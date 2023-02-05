from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import torch
from jax import random
from scipy.optimize import linear_sum_assignment
from natsort import natsorted

from extensions.sd_webui_runtime_block_merge.scripts.utils import rngmix


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


# def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
#   """We assume that one permutation cannot appear in two axes of the same weight array."""
#   assert num_hidden_layers >= 1
#   return PermutationSpec(
#       perm_to_axes={
#           f"P_{i}": [(f"Dense_{i}/kernel", 1), (f"Dense_{i}/bias", 0), (f"Dense_{i+1}/kernel", 0)]
#           for i in range(num_hidden_layers)
#       },
#       axes_to_perm={
#           "Dense_0/kernel": (None, "P_0"),
#           **{f"Dense_{i}/kernel": (f"P_{i-1}", f"P_{i}")
#              for i in range(1, num_hidden_layers)},
#           **{f"Dense_{i}/bias": (f"P_{i}", )
#              for i in range(num_hidden_layers)},
#           f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers-1}", None),
#           f"Dense_{num_hidden_layers}/bias": (None, ),
#       })

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm({
        "Dense_0/kernel": (None, "P_0"),
        **{f"Dense_{i}/kernel": (f"P_{i - 1}", f"P_{i}")
           for i in range(1, num_hidden_layers)},
        **{f"Dense_{i}/bias": (f"P_{i}",)
           for i in range(num_hidden_layers)},
        f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers - 1}", None),
        f"Dense_{num_hidden_layers}/bias": (None,),
    })


def vgg16_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
        "Conv_0/kernel": (None, None, None, "P_Conv_0"),
        **{f"Conv_{i}/kernel": (None, None, f"P_Conv_{i - 1}", f"P_Conv_{i}")
           for i in range(1, 13)},
        **{f"Conv_{i}/bias": (f"P_Conv_{i}",)
           for i in range(13)},
        **{f"LayerNorm_{i}/scale": (f"P_Conv_{i}",)
           for i in range(13)},
        **{f"LayerNorm_{i}/bias": (f"P_Conv_{i}",)
           for i in range(13)},
        "Dense_0/kernel": ("P_Conv_12", "P_Dense_0"),
        "Dense_0/bias": ("P_Dense_0",),
        "Dense_1/kernel": ("P_Dense_0", "P_Dense_1"),
        "Dense_1/bias": ("P_Dense_1",),
        "Dense_2/kernel": ("P_Dense_1", None),
        "Dense_2/bias": (None,),
    })


def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
    norm = lambda name, p: {f"{name}/scale": (p,), f"{name}/bias": (p,)}
    dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out,)}

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **conv(f"{name}/conv1", p, f"P_{name}_inner"),
        **norm(f"{name}/norm1", f"P_{name}_inner"),
        **conv(f"{name}/conv2", f"P_{name}_inner", p),
        **norm(f"{name}/norm2", p)
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}/conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}/norm1", f"P_{name}_inner"),
        **conv(f"{name}/conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}/norm2", p_out),
        **conv(f"{name}/shortcut/layers_0", p_in, p_out),
        **norm(f"{name}/shortcut/layers_1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("norm1", "P_bg0"),
        #
        **easyblock("blockgroups_0/blocks_0", "P_bg0"),
        **easyblock("blockgroups_0/blocks_1", "P_bg0"),
        **easyblock("blockgroups_0/blocks_2", "P_bg0"),
        #
        **shortcutblock("blockgroups_1/blocks_0", "P_bg0", "P_bg1"),
        **easyblock("blockgroups_1/blocks_1", "P_bg1"),
        **easyblock("blockgroups_1/blocks_2", "P_bg1"),
        #
        **shortcutblock("blockgroups_2/blocks_0", "P_bg1", "P_bg2"),
        **easyblock("blockgroups_2/blocks_1", "P_bg2"),
        **easyblock("blockgroups_2/blocks_2", "P_bg2"),
        #
        **dense("dense", "P_bg2", None),
    })


def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}/kernel": (None, None, p_in, p_out)}
    norm = lambda name, p: {f"{name}/scale": (p,), f"{name}/bias": (p,)}
    dense = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out,)}

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **conv(f"{name}/conv1", p, f"P_{name}_inner1"),
        **norm(f"{name}/norm1", f"P_{name}_inner1"),
        #
        **conv(f"{name}/conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
        **norm(f"{name}/norm2", f"P_{name}_inner2"),
        #
        **conv(f"{name}/conv3", f"P_{name}_inner2", p),
        **norm(f"{name}/norm3", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}/conv1", p_in, f"P_{name}_inner1"),
        **norm(f"{name}/norm1", f"P_{name}_inner1"),
        #
        **conv(f"{name}/conv2", f"P_{name}_inner1", f"P_{name}_inner2"),
        **norm(f"{name}/norm2", f"P_{name}_inner2"),
        #
        **conv(f"{name}/conv2", f"P_{name}_inner2", p_out),
        **norm(f"{name}/norm2", p_out),
        #
        **conv(f"{name}/shortcut/layers_0", p_in, p_out),
        **norm(f"{name}/shortcut/layers_1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("norm1", "P_bg0"),
        #
        **shortcutblock("blockgroups_0/blocks_0", "P_bg0", "P_bg1"),
        **easyblock("blockgroups_0/blocks_1", "P_bg1"),
        **easyblock("blockgroups_0/blocks_2", "P_bg1"),
        #
        **shortcutblock("blockgroups_1/blocks_0", "P_bg1", "P_bg2"),
        **easyblock("blockgroups_1/blocks_1", "P_bg2"),
        **easyblock("blockgroups_1/blocks_2", "P_bg2"),
        **easyblock("blockgroups_1/blocks_3", "P_bg2"),
        #
        **shortcutblock("blockgroups_2/blocks_0", "P_bg2", "P_bg3"),
        **easyblock("blockgroups_2/blocks_1", "P_bg3"),
        **easyblock("blockgroups_2/blocks_2", "P_bg3"),
        **easyblock("blockgroups_2/blocks_3", "P_bg3"),
        **easyblock("blockgroups_2/blocks_4", "P_bg3"),
        **easyblock("blockgroups_2/blocks_5", "P_bg3"),
        #
        **shortcutblock("blockgroups_3/blocks_0", "P_bg3", "P_bg4"),
        **easyblock("blockgroups_3/blocks_1", "P_bg4"),
        **easyblock("blockgroups_3/blocks_2", "P_bg4"),
        #
        **dense("dense", "P_bg4", None),
    })


# input_blocks
# ModuleList(
#   (0): TimestepEmbedSequential(
#     (0): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   )
#   (1-2): 2 x TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=320, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Identity()
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#       (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=320, out_features=320, bias=False)
#             (to_v): Linear(in_features=320, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=320, out_features=2560, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=1280, out_features=320, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=768, out_features=320, bias=False)
#             (to_v): Linear(in_features=768, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (3): TimestepEmbedSequential(
#     (0): Downsample(
#       (op): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     )
#   )
#   (4): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=640, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#       (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=640, out_features=640, bias=False)
#             (to_v): Linear(in_features=640, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=640, out_features=5120, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=2560, out_features=640, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=768, out_features=640, bias=False)
#             (to_v): Linear(in_features=768, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (5): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=640, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Identity()
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#       (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=640, out_features=640, bias=False)
#             (to_v): Linear(in_features=640, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=640, out_features=5120, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=2560, out_features=640, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=768, out_features=640, bias=False)
#             (to_v): Linear(in_features=768, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (6): TimestepEmbedSequential(
#     (0): Downsample(
#       (op): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     )
#   )
#   (7): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#       (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=1280, out_features=10240, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=5120, out_features=1280, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=768, out_features=1280, bias=False)
#             (to_v): Linear(in_features=768, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (8): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Identity()
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#       (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=1280, out_features=10240, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=5120, out_features=1280, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=768, out_features=1280, bias=False)
#             (to_v): Linear(in_features=768, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (9): TimestepEmbedSequential(
#     (0): Downsample(
#       (op): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#     )
#   )
#   (10-11): 2 x TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Identity()
#     )
#   )
# )
#
#
# middle_block
# TimestepEmbedSequential(
#   (0): ResBlock(
#     (in_layers): Sequential(
#       (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#       (1): SiLU()
#       (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#     (h_upd): Identity()
#     (x_upd): Identity()
#     (emb_layers): Sequential(
#       (0): SiLU()
#       (1): Linear(in_features=1280, out_features=1280, bias=True)
#     )
#     (out_layers): Sequential(
#       (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#       (1): SiLU()
#       (2): Dropout(p=0, inplace=False)
#       (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#     (skip_connection): Identity()
#   )
#   (1): SpatialTransformer(
#     (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#     (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#     (transformer_blocks): ModuleList(
#       (0): BasicTransformerBlock(
#         (attn1): CrossAttention(
#           (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#           (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#           (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#           (to_out): Sequential(
#             (0): Linear(in_features=1280, out_features=1280, bias=True)
#             (1): Dropout(p=0.0, inplace=False)
#           )
#         )
#         (ff): FeedForward(
#           (net): Sequential(
#             (0): GEGLU(
#               (proj): Linear(in_features=1280, out_features=10240, bias=True)
#             )
#             (1): Dropout(p=0.0, inplace=False)
#             (2): Linear(in_features=5120, out_features=1280, bias=True)
#           )
#         )
#         (attn2): CrossAttention(
#           (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#           (to_k): Linear(in_features=768, out_features=1280, bias=False)
#           (to_v): Linear(in_features=768, out_features=1280, bias=False)
#           (to_out): Sequential(
#             (0): Linear(in_features=1280, out_features=1280, bias=True)
#             (1): Dropout(p=0.0, inplace=False)
#           )
#         )
#         (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#       )
#     )
#     (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (2): ResBlock(
#     (in_layers): Sequential(
#       (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#       (1): SiLU()
#       (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#     (h_upd): Identity()
#     (x_upd): Identity()
#     (emb_layers): Sequential(
#       (0): SiLU()
#       (1): Linear(in_features=1280, out_features=1280, bias=True)
#     )
#     (out_layers): Sequential(
#       (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#       (1): SiLU()
#       (2): Dropout(p=0, inplace=False)
#       (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#     (skip_connection): Identity()
#   )
# )
#
#
#
# output_blocks
# ModuleList(
#   (0-1): 2 x TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (2): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): Upsample(
#       (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (3-4): 2 x TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#       (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=1280, out_features=10240, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=5120, out_features=1280, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=768, out_features=1280, bias=False)
#             (to_v): Linear(in_features=768, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (5): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=1280, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#       (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=1280, out_features=10240, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=5120, out_features=1280, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#             (to_k): Linear(in_features=768, out_features=1280, bias=False)
#             (to_v): Linear(in_features=768, out_features=1280, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=1280, out_features=1280, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (2): Upsample(
#       (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (6): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=640, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#       (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=640, out_features=640, bias=False)
#             (to_v): Linear(in_features=640, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=640, out_features=5120, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=2560, out_features=640, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=768, out_features=640, bias=False)
#             (to_v): Linear(in_features=768, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (7): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=640, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#       (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=640, out_features=640, bias=False)
#             (to_v): Linear(in_features=640, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=640, out_features=5120, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=2560, out_features=640, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=768, out_features=640, bias=False)
#             (to_v): Linear(in_features=768, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (8): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=640, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#       (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=640, out_features=640, bias=False)
#             (to_v): Linear(in_features=640, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=640, out_features=5120, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=2560, out_features=640, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=640, out_features=640, bias=False)
#             (to_k): Linear(in_features=768, out_features=640, bias=False)
#             (to_v): Linear(in_features=768, out_features=640, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=640, out_features=640, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (2): Upsample(
#       (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     )
#   )
#   (9): TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=320, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#       (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=320, out_features=320, bias=False)
#             (to_v): Linear(in_features=320, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=320, out_features=2560, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=1280, out_features=320, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=768, out_features=320, bias=False)
#             (to_v): Linear(in_features=768, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
#   (10-11): 2 x TimestepEmbedSequential(
#     (0): ResBlock(
#       (in_layers): Sequential(
#         (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (h_upd): Identity()
#       (x_upd): Identity()
#       (emb_layers): Sequential(
#         (0): SiLU()
#         (1): Linear(in_features=1280, out_features=320, bias=True)
#       )
#       (out_layers): Sequential(
#         (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#         (1): SiLU()
#         (2): Dropout(p=0, inplace=False)
#         (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       )
#       (skip_connection): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#     )
#     (1): SpatialTransformer(
#       (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#       (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#       (transformer_blocks): ModuleList(
#         (0): BasicTransformerBlock(
#           (attn1): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=320, out_features=320, bias=False)
#             (to_v): Linear(in_features=320, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (ff): FeedForward(
#             (net): Sequential(
#               (0): GEGLU(
#                 (proj): Linear(in_features=320, out_features=2560, bias=True)
#               )
#               (1): Dropout(p=0.0, inplace=False)
#               (2): Linear(in_features=1280, out_features=320, bias=True)
#             )
#           )
#           (attn2): CrossAttention(
#             (to_q): Linear(in_features=320, out_features=320, bias=False)
#             (to_k): Linear(in_features=768, out_features=320, bias=False)
#             (to_v): Linear(in_features=768, out_features=320, bias=False)
#             (to_out): Sequential(
#               (0): Linear(in_features=320, out_features=320, bias=True)
#               (1): Dropout(p=0.0, inplace=False)
#             )
#           )
#           (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#           (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#         )
#       )
#       (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#     )
#   )
# )
#
#
# out
# Sequential(
#   (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
#   (1): SiLU()
#   (2): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )
def unet686_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}  # weight as kernel
    conv_bias = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None), f"{name}.bias": (p_out,)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    linear = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in),
                                        f"{name}.bias": (p_out,)}  # dense, no activation, equal to linear in pytorch
    linear_nobias = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in)}

    # class ResBlock(TimestepBlock):
    #     """
    #     A residual block that can optionally change the number of channels.
    #     :param channels: the number of input channels.
    #     :param emb_channels: the number of timestep embedding channels.
    #     :param dropout: the rate of dropout.
    #     :param out_channels: if specified, the number of out channels.
    #     :param use_conv: if True and out_channels is specified, use a spatial
    #         convolution instead of a smaller 1x1 convolution to change the
    #         channels in the skip connection.
    #     :param dims: determines if the signal is 1D, 2D, or 3D.
    #     :param use_checkpoint: if True, use gradient checkpointing on this module.
    #     :param up: if True, use this block for upsampling.
    #     :param down: if True, use this block for downsampling.
    #     """
    #
    #     def __init__(
    #             self,
    #             channels,
    #             emb_channels,
    #             dropout,
    #             out_channels=None,
    #             use_conv=False,
    #             use_scale_shift_norm=False,
    #             dims=2,
    #             use_checkpoint=False,
    #             up=False,
    #             down=False,
    #     ):
    #         super().__init__()
    #         self.channels = channels
    #         self.emb_channels = emb_channels
    #         self.dropout = dropout
    #         self.out_channels = out_channels or channels
    #         self.use_conv = use_conv
    #         self.use_checkpoint = use_checkpoint
    #         self.use_scale_shift_norm = use_scale_shift_norm
    #
    #         self.in_layers = nn.Sequential(
    #             normalization(channels),
    #             nn.SiLU(),
    #             conv_nd(dims, channels, self.out_channels, 3, padding=1),
    #         )
    #
    #         self.updown = up or down
    #
    #         if up:
    #             self.h_upd = Upsample(channels, False, dims)
    #             self.x_upd = Upsample(channels, False, dims)
    #         elif down:
    #             self.h_upd = Downsample(channels, False, dims)
    #             self.x_upd = Downsample(channels, False, dims)
    #         else:
    #             self.h_upd = self.x_upd = nn.Identity()
    #
    #         self.emb_layers = nn.Sequential(
    #             nn.SiLU(),
    #             linear(
    #                 emb_channels,
    #                 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
    #             ),
    #         )
    #         self.out_layers = nn.Sequential(
    #             normalization(self.out_channels),
    #             nn.SiLU(),
    #             nn.Dropout(p=dropout),
    #             zero_module(
    #                 conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
    #             ),
    #         )
    #
    #         if self.out_channels == channels:
    #             self.skip_connection = nn.Identity()
    #         elif use_conv:
    #             self.skip_connection = conv_nd(
    #                 dims, channels, self.out_channels, 3, padding=1
    #             )
    #         else:
    #             self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
    #
    #     def forward(self, x, emb):
    #         """
    #         Apply the block to a Tensor, conditioned on a timestep embedding.
    #         :param x: an [N x C x ...] Tensor of features.
    #         :param emb: an [N x emb_channels] Tensor of timestep embeddings.
    #         :return: an [N x C x ...] Tensor of outputs.
    #         """
    #         return checkpoint(
    #             self._forward, (x, emb), self.parameters(), self.use_checkpoint
    #         )
    #
    #     def _forward(self, x, emb):
    #         if self.updown:
    #             in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
    #             h = in_rest(x)
    #             h = self.h_upd(h)
    #             x = self.x_upd(x)
    #             h = in_conv(h)
    #         else:
    #             h = self.in_layers(x)
    #         emb_out = self.emb_layers(emb).type(h.dtype)
    #         while len(emb_out.shape) < len(h.shape):
    #             emb_out = emb_out[..., None]
    #         if self.use_scale_shift_norm:
    #             out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
    #             scale, shift = th.chunk(emb_out, 2, dim=1)
    #             h = out_norm(h) * (1 + scale) + shift
    #             h = out_rest(h)
    #         else:
    #             h = h + emb_out
    #             h = self.out_layers(h)
    #         return self.skip_connection(x) + h

    resblock = lambda name, p_in, p_out: {
        # note: be wary of variants that contain Upsample/Downsample Blocks and non-identity skip connections
        # in_layers
        **norm(f"{name}.in_layers.0", p_in),
        **conv_bias(f"{name}.in_layers.2", p_in, f"P_{name}_inner1"),
        # out_layers
        **norm(f"{name}.out_layers.0", f"P_{name}_inner1"),
        **conv_bias(f"{name}.out_layers.3", f"P_{name}_inner1", p_out),
    }

    resblock_with_skip = lambda name, p_in, p_out: {
        # note: be wary of variants that contain Upsample/Downsample Blocks and non-identity skip connections
        # in_layers
        **norm(f"{name}.in_layers.0", p_in),
        **conv_bias(f"{name}.in_layers.2", p_in, f"P_{name}_inner1"),
        # out_layers
        **norm(f"{name}.out_layers.0", f"P_{name}_inner1"),
        **conv_bias(f"{name}.out_layers.3", f"P_{name}_inner1", p_out),
        # skip_connection
        **conv_bias(f"{name}.skip_connection", p_in, p_out),
    }

    upresblock = lambda name, p_in, p_in_cat, p_out: {
        # in_layers
        **norm(f"{name}.in_layers.0", p_in),
        **conv_bias(f"{name}.in_layers.2", p_in, f"P_{name}_inner1"),
        **norm(f"{name}.in_layers.0", p_in_cat),
        **conv_bias(f"{name}.in_layers.2", p_in_cat, f"P_{name}_inner1"),
        # out_layers
        **norm(f"{name}.out_layers.0", f"P_{name}_inner1"),
        **conv_bias(f"{name}.out_layers.3", f"P_{name}_inner1", p_out),
    }

    upresblock_with_skip = lambda name, p_in, p_in_cat, p_out: {
        # in_layers
        **norm(f"{name}.in_layers.0", p_in),
        **conv_bias(f"{name}.in_layers.2", p_in, f"P_{name}_inner1"),
        **norm(f"{name}.in_layers.0", p_in_cat),
        **conv_bias(f"{name}.in_layers.2", p_in_cat, f"P_{name}_inner1"),
        # out_layers
        **norm(f"{name}.out_layers.0", f"P_{name}_inner1"),
        **conv_bias(f"{name}.out_layers.3", f"P_{name}_inner1", p_out),
        # skip_connection
        **conv_bias(f"{name}.skip_connection", p_in, p_out),
        **conv_bias(f"{name}.skip_connection", p_in_cat, p_out),
    }

    # class CrossAttention(nn.Module):
    #     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
    #         super().__init__()
    #         inner_dim = dim_head * heads
    #         context_dim = default(context_dim, query_dim)
    #
    #         self.scale = dim_head ** -0.5
    #         self.heads = heads
    #
    #         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    #         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
    #         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
    #
    #         self.to_out = nn.Sequential(
    #             nn.Linear(inner_dim, query_dim),
    #             nn.Dropout(dropout)
    #         )
    #
    #     def forward(self, x, context=None, mask=None):
    #         h = self.heads
    #
    #         q = self.to_q(x)
    #         context = default(context, x)
    #         k = self.to_k(context)
    #         v = self.to_v(context)
    #
    #         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    #
    #         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    #         del q, k
    #
    #         if exists(mask):
    #             mask = rearrange(mask, 'b ... -> b (...)')
    #             max_neg_value = -torch.finfo(sim.dtype).max
    #             mask = repeat(mask, 'b j -> (b h) () j', h=h)
    #             sim.masked_fill_(~mask, max_neg_value)
    #
    #         # attention, what we cannot get enough of
    #         sim = sim.softmax(dim=-1)
    #
    #         out = einsum('b i j, b j d -> b i d', sim, v)
    #         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    #         return self.to_out(out)

    self_attention = lambda name, p_in, p_out: {  # CrossAttention without context input
        # to_q
        **linear_nobias(f"{name}.to_q", p_in, f"P_{name}_crossattninner1"),
        # no context, self attention
        # to_k
        **linear_nobias(f"{name}.to_k", p_in, f"P_{name}_crossattninner1"),
        # to_v
        **linear_nobias(f"{name}.to_v", p_in, f"P_{name}_crossattninner1"),
        # to_out
        **linear(f"{name}.to_out.0", f"P_{name}_crossattninner1", p_out),
    }

    cross_attention = lambda name, p_in, p_out: {  # CrossAttention
        # to_q
        **linear_nobias(f"{name}.to_q", p_in, f"P_{name}_crossattninner1"),
        # has context, cross attention
        # to_k
        **linear_nobias(f"{name}.to_k", f"P_{name}_crossattncontext1", f"P_{name}_crossattninner1"),
        # to_v
        **linear_nobias(f"{name}.to_v", f"P_{name}_crossattncontext1", f"P_{name}_crossattninner1"),
        # to_out
        **linear(f"{name}.to_out.0", f"P_{name}_crossattninner1", p_out),
    }

    # class FeedForward(nn.Module):
    #     def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
    #         super().__init__()
    #         inner_dim = int(dim * mult)
    #         dim_out = default(dim_out, dim)
    #         project_in = nn.Sequential(
    #             nn.Linear(dim, inner_dim),
    #             nn.GELU()
    #         ) if not glu else GEGLU(dim, inner_dim)
    #
    #         self.net = nn.Sequential(
    #             project_in,
    #             nn.Dropout(dropout),
    #             nn.Linear(inner_dim, dim_out)
    #         )
    #
    #     def forward(self, x):
    #         return self.net(x)

    feed_forward = lambda name, p_in, p_out: {
        # project_in
        **linear(f"{name}.net.0.proj", p_in, f"P_{name}_ffinner1"),
        # net
        **linear(f"{name}.net.2", f"P_{name}_ffinner1", p_out),
    }

    # class BasicTransformerBlock(nn.Module):
    #     ATTENTION_MODES = {
    #         "softmax": CrossAttention,  # vanilla attention
    #         "softmax-xformers": MemoryEfficientCrossAttention
    #     }
    #
    #     def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
    #                  disable_self_attn=False):
    #         super().__init__()
    #         attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
    #         assert attn_mode in self.ATTENTION_MODES
    #         attn_cls = self.ATTENTION_MODES[attn_mode]
    #         self.disable_self_attn = disable_self_attn
    #         self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
    #                               context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
    #         self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
    #         self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
    #                               heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
    #         self.norm1 = nn.LayerNorm(dim)
    #         self.norm2 = nn.LayerNorm(dim)
    #         self.norm3 = nn.LayerNorm(dim)
    #         self.checkpoint = checkpoint
    #
    #     def forward(self, x, context=None):
    #         return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    #
    #     def _forward(self, x, context=None):
    #         x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
    #         x = self.attn2(self.norm2(x), context=context) + x
    #         x = self.ff(self.norm3(x)) + x
    #         return x

    basic_cross_transformer_block = lambda name, p_in, p_out: {  # BasicTransformerBlock with context
        # norm1 
        **norm(f"{name}.norm1", p_in),
        # attn1
        **cross_attention(f"{name}.attn1", p_in, p_out),
        # norm2
        **norm(f"{name}.norm2", p_in),
        # attn2
        **cross_attention(f"{name}.attn2", p_in, p_out),
        # norm3
        **norm(f"{name}.norm3", p_in),
        # ff
        **feed_forward(f"{name}.ff", p_in, p_out),
    }

    basic_transformer_block = lambda name, p_in, p_out: {  # BasicTransformerBlock with context
        # norm1
        **norm(f"{name}.norm1", p_in),
        # attn1
        **self_attention(f"{name}.attn1", p_in, p_out),
        # norm2
        **norm(f"{name}.norm2", p_in),
        # attn2
        **cross_attention(f"{name}.attn2", p_in, p_out),
        # norm3
        **norm(f"{name}.norm3", p_in),
        # ff
        **feed_forward(f"{name}.ff", p_in, p_out),

    }

    # class SpatialTransformer(nn.Module):
    #     """
    #     Transformer block for image-like data.
    #     First, project the input (aka embedding)
    #     and reshape to b, t, d.
    #     Then apply standard transformer action.
    #     Finally, reshape to image
    #     NEW: use_linear for more efficiency instead of the 1x1 convs
    #     """
    #
    #     def __init__(self, in_channels, n_heads, d_head,
    #                  depth=1, dropout=0., context_dim=None,
    #                  disable_self_attn=False, use_linear=False,
    #                  use_checkpoint=True):
    #         super().__init__()
    #         if exists(context_dim) and not isinstance(context_dim, list):
    #             context_dim = [context_dim]
    #         self.in_channels = in_channels
    #         inner_dim = n_heads * d_head
    #         self.norm = Normalize(in_channels)
    #         if not use_linear:
    #             self.proj_in = nn.Conv2d(in_channels,
    #                                      inner_dim,
    #                                      kernel_size=1,
    #                                      stride=1,
    #                                      padding=0)
    #         else:
    #             self.proj_in = nn.Linear(in_channels, inner_dim)
    #
    #         self.transformer_blocks = nn.ModuleList(
    #             [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
    #                                    disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
    #              for d in range(depth)]
    #         )
    #         if not use_linear:
    #             self.proj_out = zero_module(nn.Conv2d(inner_dim,
    #                                                   in_channels,
    #                                                   kernel_size=1,
    #                                                   stride=1,
    #                                                   padding=0))
    #         else:
    #             self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
    #         self.use_linear = use_linear
    #
    #     def forward(self, x, context=None):
    #         # note: if no context is given, cross-attention defaults to self-attention
    #         if not isinstance(context, list):
    #             context = [context]
    #         b, c, h, w = x.shape
    #         x_in = x
    #         x = self.norm(x)
    #         if not self.use_linear:
    #             x = self.proj_in(x)
    #         x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
    #         if self.use_linear:
    #             x = self.proj_in(x)
    #         for i, block in enumerate(self.transformer_blocks):
    #             x = block(x, context=context[i])
    #         if self.use_linear:
    #             x = self.proj_out(x)
    #         x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
    #         if not self.use_linear:
    #             x = self.proj_out(x)
    #         return x + x_in

    spatial_transformer_block = lambda name, p_in, p_out: {
        # norm
        **norm(f"{name}.norm", p_in),
        # proj_in
        **conv_bias(f"{name}.proj_in", p_in, f"P_{name}_inner1"),
        # transformer_blocks
        **basic_transformer_block(f"{name}.transformer_blocks.0", f"P_{name}_inner1", f"P_{name}_inner2"),
        # proj_out
        **conv_bias(f"{name}.proj_out", f"P_{name}_inner2", p_out),
        # shortcut connection here
    }

    return permutation_spec_from_axes_to_perm({
        **conv_bias("input_blocks.0.0", None, "P_input0"),

        **resblock("input_blocks.1.0", "P_input0", "P_input1_0"),
        **spatial_transformer_block("input_blocks.1.1", "P_input1_0", "P_input1_1"),

        **resblock("input_blocks.2.0", "P_input1_1", "P_input2_0"),
        **spatial_transformer_block("input_blocks.2.1", "P_input2_0", "P_input2_1"),

        # downsample
        **conv_bias("input_blocks.3.0.op", "P_input2_1", "P_input3"),

        **resblock_with_skip("input_blocks.4.0", "P_input3", "P_input4_0"),
        **spatial_transformer_block("input_blocks.4.1", "P_input4_0", "P_input4_1"),

        **resblock("input_blocks.5.0", "P_input4_1", "P_input5_0"),
        **spatial_transformer_block("input_blocks.5.1", "P_input5_0", "P_input5_1"),

        # downsample
        **conv_bias("input_blocks.6.0.op", "P_input5_1", "P_input6"),

        **resblock_with_skip("input_blocks.7.0", "P_input6", "P_input7_0"),
        **spatial_transformer_block("input_blocks.7.1", "P_input7_0", "P_input7_1"),

        **resblock("input_blocks.8.0", "P_input7_1", "P_input8_0"),
        **spatial_transformer_block("input_blocks.8.1", "P_input8_0", "P_input8_1"),

        # downsample
        **conv_bias("input_blocks.9.0.op", "P_input8_1", "P_input9_0"),

        **resblock("input_blocks.10.0", "P_input9", "P_input10_0"),

        **resblock("input_blocks.11.0", "P_input10_0", "P_input11_0"),

        # middle
        **resblock("middle_block.0", "P_input11_1", "P_middle_0"),
        **spatial_transformer_block("middle_block.1", "P_middle_0", "P_middle_1"),
        **resblock("middle_block.2", "P_middle_1", "P_middle_2"),

        # **upresblock_with_skip("output_blocks.0.0", "P_middle_2", "P_input11_0", "P_output0_0"),
        **resblock_with_skip("output_blocks.0.0", "special_P_outputb0", "P_output0_0"),

        # **upresblock_with_skip("output_blocks.1.0", "P_output0_0", "P_input10_0", "P_output1_0"),
        **resblock_with_skip("output_blocks.1.0", "special_P_outputb1", "P_output1_0"),

        # upsample
        # **upresblock_with_skip("output_blocks.2.0", "P_output1_0", "P_input9_0", "P_output2_0"),
        **resblock_with_skip("output_blocks.2.0", "special_P_outputb2", "P_output2_0"),
        **conv_bias("output_blocks.2.1.conv", "P_output2_0", "P_output2_1"),

        # **upresblock_with_skip("output_blocks.3.0", "P_output2_1", "P_input8_1", "P_output3_0"),
        **resblock_with_skip("output_blocks.3.0", "special_P_outputb3", "P_output3_0"),
        **spatial_transformer_block("output_blocks.3.1", "P_output3_0", "P_output3_1"),

        # **upresblock_with_skip("output_blocks.4.0", "P_output3_1", "P_input7_1", "P_output4_0"),
        **resblock_with_skip("output_blocks.4.0", "special_P_outputb4", "P_output4_0"),
        **spatial_transformer_block("output_blocks.4.1", "P_output4_0", "P_output4_1"),

        # upsample
        # **upresblock_with_skip("output_blocks.5.0", "P_output4_1", "P_input6", "P_output5_0"),
        **resblock_with_skip("output_blocks.5.0", "special_P_outputb5", "P_output5_0"),
        **spatial_transformer_block("output_blocks.5.1", "P_output5_0", "P_output5_1"),
        **conv_bias("output_blocks.5.2.conv", "P_output5_1", "P_output5_2"),

        # **upresblock_with_skip("output_blocks.6.0", "P_output5_2", "P_input5_1", "P_output6_0"),
        **resblock_with_skip("output_blocks.6.0", "special_P_outputb6", "P_output6_0"),
        **spatial_transformer_block("output_blocks.6.1", "P_output6_0", "P_output6_1"),

        # **upresblock_with_skip("output_blocks.7.0", "P_output6_1", "P_input4_1", "P_output7_0"),
        **resblock_with_skip("output_blocks.7.0", "special_P_outputb7", "P_output7_0"),
        **spatial_transformer_block("output_blocks.7.1", "P_output7_0", "P_output7_1"),

        # upsample
        # **upresblock_with_skip("output_blocks.8.0", "P_output7_1", "P_input3", "P_output8_0"),
        **resblock_with_skip("output_blocks.8.0", "special_P_outputb8", "P_output8_0"),
        **spatial_transformer_block("output_blocks.8.1", "P_output8_0", "P_output8_1"),
        **conv_bias("output_blocks.8.2.conv", "P_output8_1", "P_output8_2"),

        # **upresblock_with_skip("output_blocks.9.0", "P_output8_2", "P_input2_1", "P_output9_0"),
        **resblock_with_skip("output_blocks.9.0", "special_P_outputb9", "P_output9_0"),
        **spatial_transformer_block("output_blocks.9.1", "P_output9_0", "P_output9_1"),

        # **upresblock_with_skip("output_blocks.10.0", "P_output9_1", "P_input1_1", "P_output10_0"),
        **resblock_with_skip("output_blocks.10.0", "special_P_outputb10", "P_output10_0"),
        **spatial_transformer_block("output_blocks.10.1", "P_output10_0", "P_output10_1"),

        # **upresblock_with_skip("output_blocks.11.0", "P_output10_1", "P_input0", "P_output11_0"),
        **resblock_with_skip("output_blocks.11.0", "special_P_outputb11", "P_output11_0"),
        **spatial_transformer_block("output_blocks.11.1", "P_output11_0", "P_output11_1"),

        # out
        **norm("out.0", "P_output11_1"),
        **conv_bias("out.2", "P_output11_1", None),

    })


def get_permuted_param_jax(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = jnp.take(w, perm[p], axis=axis)

    return w


# special_perm_dict = {
#     'P_output_blocks.6.1.transformer_blocks.0.ff_ffinner1': lambda x: x[:2560],
# }

def assemble_special_P(cur_specialP_name, perm):
    matching_dict={
        'special_P_outputb0':['P_middle_2','P_input11_0'],
        'special_P_outputb1':['P_output0_0','P_input10_0'],
        'special_P_outputb2':['P_output1_0','P_input9_0'],
        'special_P_outputb3':['P_output2_1','P_input8_1'],
        'special_P_outputb4':['P_output3_1','P_input7_1'],
        'special_P_outputb5':['P_output4_1','P_input6'],
        'special_P_outputb6':['P_output5_2','P_input5_1'],
        'special_P_outputb7':['P_output6_1','P_input4_1'],
        'special_P_outputb8':['P_output7_1','P_input3'],
        'special_P_outputb9':['P_output8_2','P_input2_1'],
        'special_P_outputb10':['P_output9_1','P_input1_1'],
        'special_P_outputb11':['P_output10_1','P_input0'],
    }
    first_P = perm[matching_dict[cur_specialP_name][0]].int()
    second_P = perm[matching_dict[cur_specialP_name][1]].int() + len(first_P)
    final_P = torch.cat([first_P, second_P])
    return final_P

def ff_GEGLU_firsthalf(cur_specialP_name, perm):


    ori_length = perm[cur_specialP_name].shape[0]
    result = torch.narrow(perm[cur_specialP_name], 0, 0, int(ori_length/2))
    return result

special_perm_matching_dict = {
    'special_P_outputb': assemble_special_P,
    # 'transformer_blocks.0.ff_ffinner1': ff_GEGLU_firsthalf,
}

sec_special_perm_matching_dict = {
    'transformer_blocks.0.ff_ffinner1': ff_GEGLU_firsthalf,
}


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            target_perm = perm[p].int()
            for special_key in special_perm_matching_dict.keys():
                if special_key in p:
                    target_perm = special_perm_matching_dict[special_key](p, perm)
                    break
            print(f"Permuting {k} along axis {axis} with permutation {p}")
            try:
                w = torch.index_select(w, axis, target_perm)
            except RuntimeError as e:
                if e.args[0].startswith('INDICES element is out of DATA bounds'):
                    for special_key in sec_special_perm_matching_dict.keys():
                        if special_key in p:
                            target_perm = sec_special_perm_matching_dict[special_key](p, perm)
                            break

                    w = torch.index_select(w, axis, target_perm)


    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching_jax(rng,
                        ps: PermutationSpec,
                        params_a,
                        params_b,
                        max_iter=100,
                        init_perm=None,
                        silent=False):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]

            for skip_key in special_perm_matching_dict.keys():
                if skip_key in p:
                    continue
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            if not silent: print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm


def weight_matching_test(rng,
                         ps: PermutationSpec,
                         params_a,
                         params_b,
                         max_iter=100,
                         init_perm=None,
                         silent=False):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            if not silent: print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm


def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=1000, init_perm=None):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    torchRandomGen = torch.Generator()
    initial_seed = 2147483647

    for iteration in range(max_iter):
        progress = False
        cur_seed = initial_seed + hash(iteration)
        torchRandomGen.manual_seed(cur_seed)
        for p_ix in torch.randperm(len(perm_names), generator=torchRandomGen):
            p = perm_names[p_ix]
            if p == 'P_output_blocks.10.1_inner1':
                print('debug')
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()

            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci).int()

        if not progress:
            break

    return perm


def test_weight_matching_jax():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=1)
    rng = random.PRNGKey(123)
    num_hidden = 10
    shapes = {
        "Dense_0/kernel": (2, num_hidden),
        "Dense_0/bias": (num_hidden,),
        "Dense_1/kernel": (num_hidden, 3),
        "Dense_1/bias": (3,)
    }
    params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
    params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
    perm = weight_matching(rng, ps, params_a, params_b)
    print(perm)


def filter_unet_state_dict(input_dict):
    filtered_dict = {}
    for key, value in input_dict.items():

        if key.startswith('model.diffusion_model'):
            filtered_dict[key[22:]] = value
    filtered_dict_keys = natsorted(filtered_dict.keys())
    filtered_dict = {k: filtered_dict[k] for k in filtered_dict_keys}

    return filtered_dict


def test_verify_weight_matching():
    ps_unet = unet686_permutation_spec()
    test_modela = torch.load('../../../models/Stable-diffusion/pastelmix.ckpt')
    filtered_test_modela_dict = filter_unet_state_dict(test_modela['state_dict'])
    # compare dict with ps_unet keys, then print missing element
    filtered_dict_keys = natsorted(filtered_test_modela_dict.keys())
    filtered_test_modela_dict = {k: filtered_test_modela_dict[k] for k in filtered_dict_keys}

    filter_ps_unet_keys = natsorted(ps_unet.axes_to_perm.keys())
    filtered_ps_unet_axes_to_perm = {k: ps_unet.axes_to_perm[k] for k in filter_ps_unet_keys}

    sd_skip_count = 0
    for i in range(len(filtered_dict_keys)):
        if i - sd_skip_count < len(filter_ps_unet_keys):

            if filter_ps_unet_keys[i - sd_skip_count] != filtered_dict_keys[i]:
                if 'emb_layers' in filtered_dict_keys[i]:
                    sd_skip_count += 1
                    continue
                print(filter_ps_unet_keys[i])
        else:
            if 'time_emb' in filtered_dict_keys[i]:
                continue
            print(filtered_dict_keys[i])

    # perm = weight_matching(rng, ps, params_a, params_b)
    print('finish')
    # perm = weight_matching(rng, ps, params_a, params_b)
    # print(perm)


def test_weight_matching():
    ps_unet = unet686_permutation_spec()
    test_modela = torch.load('../../../models/Stable-diffusion/pastelmix.ckpt')
    filtered_test_modela_dict = filter_unet_state_dict(test_modela['state_dict'])
    test_modelb = torch.load('../../../models/Stable-diffusion/AbyssOrangeMix2_sfw.ckpt')
    filtered_test_modelb_dict = filter_unet_state_dict(test_modelb['state_dict'])
    perm = weight_matching(ps_unet, filtered_test_modela_dict, filtered_test_modelb_dict)
    print(perm)


if __name__ == "__main__":
    test_weight_matching()
