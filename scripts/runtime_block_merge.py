import copy
import itertools
import json
from datetime import datetime

import modules.scripts as scripts
import gradio as gr

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import sd_models, shared, devices
from scripts.mbw_util.preset_weights import PresetWeights
import torch
from natsort import natsorted

from pathlib import Path
import safetensors.torch

presetWeights = PresetWeights()

shared.UNetBManager = None

known_block_prefixes = [
    'input_blocks.0.',
    'input_blocks.1.',
    'input_blocks.2.',
    'input_blocks.3.',
    'input_blocks.4.',
    'input_blocks.5.',
    'input_blocks.6.',
    'input_blocks.7.',
    'input_blocks.8.',
    'input_blocks.9.',
    'input_blocks.10.',
    'input_blocks.11.',
    'middle_block.',
    'out.',
    'output_blocks.0.',
    'output_blocks.1.',
    'output_blocks.2.',
    'output_blocks.3.',
    'output_blocks.4.',
    'output_blocks.5.',
    'output_blocks.6.',
    'output_blocks.7.',
    'output_blocks.8.',
    'output_blocks.9.',
    'output_blocks.10.',
    'output_blocks.11.',
    'time_embed.'
]

class UNetStateManager(object):
    def __init__(self, org_unet: UNetModel):
        super().__init__()
        self.modelB_state_dict_by_blocks = []
        self.torch_unet = org_unet
        # self.modelA_state_dict = copy.deepcopy(org_unet.state_dict())
        self.modelA_state_dict = None
        self.dtype = devices.dtype
        self.modelA_state_dict_by_blocks = []
        # self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        self.modelB_state_dict = None
        # self.unet_block_module_list = []
        self.unet_block_module_list = [*self.torch_unet.input_blocks, self.torch_unet.middle_block, self.torch_unet.out,
                                       *self.torch_unet.output_blocks, self.torch_unet.time_embed]
        self.applied_weights = [0] * 27
        # self.gui_weights = [0.5] * 27
        self.enabled = False
        self.modelA_path = shared.sd_model.sd_model_checkpoint
        self.modelB_path = ''
        self.force_cpu = False
        self.modelA_dtype = None
        self.modelB_dtype = None
        self.device = devices.get_cuda_device_string() if (torch.cuda.is_available() and not shared.cmd_opts.lowvram) else "cpu"

    # def set_gui_weights(self, current_weights):
    #     self.gui_weights = current_weights

    def reload_modelA(self):
        if not self.enabled:
            return

        if self.modelA_path == shared.sd_model.sd_model_checkpoint and self.modelA_state_dict is not None:
            return
        self.modelA_path = shared.sd_model.sd_model_checkpoint

        del self.modelA_state_dict_by_blocks
        self.modelA_state_dict_by_blocks = []
        # orig_modelA_state_dict_keys = list(self.modelA_state_dict.keys())
        # for key in orig_modelA_state_dict_keys:
        #     del self.modelA_state_dict[key]
        del self.modelA_state_dict
        torch.cuda.empty_cache()
        if self.force_cpu:
            self.modelA_state_dict = self.filter_unet_state_dict(
                sd_models.read_state_dict(self.modelA_path, map_location="cpu"))
            self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
            self.modelA_dtype = itertools.islice(self.modelA_state_dict.items(), 1).__next__()[1].dtype
        else:
            self.modelA_state_dict = copy.deepcopy(self.torch_unet.state_dict())
            self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        # if self.enabled:
        # self.model_state_apply(self.gui_weights)
        self.model_state_apply(self.applied_weights)
        print('model A reloaded')

    def load_modelB(self, modelB_path, force_cpu_checkbox, current_weights):
        self.force_cpu = force_cpu_checkbox
        self.device = devices.get_cuda_device_string() if (torch.cuda.is_available() and not shared.cmd_opts.lowvram) else "cpu"
        if self.force_cpu:
            self.device = "cpu"
        model_info = sd_models.get_closet_checkpoint_match(modelB_path)
        checkpoint_file = model_info.filename
        self.modelB_path = checkpoint_file


        if self.modelA_path == checkpoint_file:
            if not self.modelB_state_dict:
                self.enabled = False
            # self.gui_weights = current_weights
            return False

        # move initialization of model A to here
        if not self.modelA_state_dict:
            if self.force_cpu:
                self.modelA_path = shared.sd_model.sd_model_checkpoint
                self.modelA_state_dict = self.filter_unet_state_dict(
                    sd_models.read_state_dict(self.modelA_path, map_location="cpu"))
                self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)

            else:
                self.modelA_state_dict = copy.deepcopy(self.torch_unet.state_dict())
                self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
                # self.modelA_dtype = self.torch_unet.dtype
            self.modelA_dtype = itertools.islice(self.modelA_state_dict.items(), 1).__next__()[1].dtype
        sd_model_hash = model_info.hash
        cache_enabled = shared.opts.sd_checkpoint_cache > 0

        # if cache_enabled and model_info in sd_models.checkpoints_loaded:
        #     # use checkpoint cache
        #     print(f"Loading weights [{sd_model_hash}] from cache")
        #     self.modelB_state_dict = sd_models.checkpoints_loaded[model_info]

        if self.modelB_state_dict:
            # orig_modelB_state_dict_keys = list(self.modelB_state_dict.keys())
            # for key in orig_modelB_state_dict_keys:
            #     del self.modelB_state_dict[key]
            del self.modelB_state_dict_by_blocks
            del self.modelB_state_dict
            torch.cuda.empty_cache()
        self.modelB_state_dict_by_blocks = []
        self.modelB_state_dict = self.filter_unet_state_dict(
            sd_models.read_state_dict(checkpoint_file, map_location=self.device))
        self.modelB_dtype = itertools.islice(self.modelB_state_dict.items(), 1).__next__()[1].dtype
        if len(self.modelA_state_dict) != len(self.modelB_state_dict):
            print('modelA and modelB state dict have different length, aborting')
            return False
        self.map_blocks(self.modelB_state_dict, self.modelB_state_dict_by_blocks)
        # verify self.modelA_state_dict and self.modelB_state_dict have same structure
        self.model_state_apply(current_weights)

        print('model B loaded')
        self.enabled = True
        return True

    def model_state_apply(self, current_weights):
        # self.gui_weights = current_weights
        # ensuring maximum precision
        operation_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        for i in range(27):
            cur_block_state_dict = {}
            for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                if operation_dtype == torch.float32:
                    # try:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 current_weights[i]).to(self.dtype)
                    # except RuntimeError:
                    #     # self.modelB_state_dict_by_blocks[i][cur_layer_key] = self.modelB_state_dict_by_blocks[i][cur_layer_key].to('cpu')
                    #     self.modelA_state_dict_by_blocks[i][cur_layer_key] = self.modelA_state_dict_by_blocks[i][
                    #         cur_layer_key].to('cpu')
                    #     curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                    #                                  self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                    #                                  current_weights[i]).to(self.dtype)
                else:
                    if self.force_cpu:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     current_weights[i]).to(self.dtype)
                    else:
                    # try:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])
                    # except RuntimeError:
                    #     # self.modelB_state_dict_by_blocks[i][cur_layer_key] = self.modelB_state_dict_by_blocks[i][cur_layer_key].to('cpu')
                    #     self.modelA_state_dict_by_blocks[i][cur_layer_key] = self.modelA_state_dict_by_blocks[i][cur_layer_key].to('cpu')
                    #     curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                    #                                  self.modelB_state_dict_by_blocks[i][cur_layer_key],
                    #                                  current_weights[i])
                if str(shared.device) != self.device:
                    curlayer_tensor = curlayer_tensor.to(shared.device)
                cur_block_state_dict[cur_layer_key] = curlayer_tensor
            self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights

    def model_state_construct(self, current_weights):
        precision_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        result_state_dict = {}
        for i in range(27):
            cur_block_state_dict = {}
            for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                if precision_dtype == torch.float32:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                 current_weights[i])
                else:
                    if self.force_cpu:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                                                     current_weights[i]).to(torch.float16)
                    else:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])

                result_state_dict[known_block_prefixes[i] + cur_layer_key] = curlayer_tensor
        return result_state_dict



    def model_state_apply_modified_blocks(self, current_weights, current_model_B):
        if not self.enabled:
            return
        modelB_info = sd_models.get_closet_checkpoint_match(current_model_B)
        checkpoint_file_B = modelB_info.filename
        if checkpoint_file_B != self.modelB_path:
            print('model B changed, shouldn\'t happen')
            self.load_modelB(current_model_B, current_weights)
            return
        if self.applied_weights == current_weights:
            return
        operation_dtype = torch.float32 if self.modelA_dtype == torch.float32 or self.modelB_dtype == torch.float32 else torch.float16
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    if operation_dtype == torch.float32:
                        curlayer_tensor = torch.lerp(
                            self.modelA_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                            self.modelB_state_dict_by_blocks[i][cur_layer_key].to(torch.float32),
                            current_weights[i]).to(self.dtype)
                    else:
                        curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                     self.modelB_state_dict_by_blocks[i][cur_layer_key],
                                                     current_weights[i])
                    if str(shared.device) != self.device:
                        curlayer_tensor = curlayer_tensor.to(shared.device)
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights




    # diff current_weights and self.applied_weights, apply only the difference
    def model_state_apply_block(self, current_weights):
        # self.gui_weights = current_weights
        if not self.enabled:
            return self.applied_weights
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights
        return self.applied_weights

    # filter input_dict to include only keys starting with 'model.diffusion_model'
    def filter_unet_state_dict(self, input_dict):
        filtered_dict = {}
        for key, value in input_dict.items():

            if key.startswith('model.diffusion_model'):
                filtered_dict[key[22:]] = value
        filtered_dict_keys = natsorted(filtered_dict.keys())
        filtered_dict = {k: filtered_dict[k] for k in filtered_dict_keys}

        return filtered_dict

    def map_blocks(self, model_state_dict_input, model_state_dict_by_blocks):
        if model_state_dict_by_blocks:
            print('mapping to non empty list')
            return
        model_state_dict_sorted_keys = natsorted(model_state_dict_input.keys())
        # sort model_state_dict by model_state_dict_sorted_keys
        model_state_dict = {k: model_state_dict_input[k] for k in model_state_dict_sorted_keys}


        current_block_index = 0
        processing_block_dict = {}
        for key in model_state_dict:
            # print(key)
            if not key.startswith(known_block_prefixes[current_block_index]):
                if not key.startswith(known_block_prefixes[current_block_index + 1]):
                    print(
                        f"unknown key {key} in statedict after block {known_block_prefixes[current_block_index]}, possible UNet structure deviation"
                    )
                    continue
                else:
                    model_state_dict_by_blocks.append(processing_block_dict)
                    processing_block_dict = {}
                    current_block_index += 1
            block_local_key = key[len(known_block_prefixes[current_block_index]):]
            processing_block_dict[block_local_key] = model_state_dict[key]

        model_state_dict_by_blocks.append(processing_block_dict)
        print('mapping complete')
        return

    def restore_original_unet(self):
        self.torch_unet.load_state_dict(self.modelA_state_dict)
        return

    def unload_all(self):
        self.modelA_path = ''
        self.modelB_path = ''
        self.applied_weights = [0.0] * 27
        del self.modelA_state_dict
        self.modelA_state_dict = None
        del self.modelA_state_dict_by_blocks
        self.modelA_state_dict_by_blocks = []
        del self.modelB_state_dict
        self.modelB_state_dict = None
        del self.modelB_state_dict_by_blocks
        self.modelB_state_dict_by_blocks = []
        # self.unet_block_module_list = []
        self.enabled = False


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        if shared.UNetBManager is None:
            shared.UNetBManager = UNetStateManager(shared.sd_model.model.diffusion_model)
            from modules.call_queue import wrap_queued_call

            def reload_modelA_checkpoint():
                if shared.opts.sd_model_checkpoint == shared.sd_model.sd_checkpoint_info.title:
                    return
                sd_models.reload_model_weights()
                shared.UNetBManager.reload_modelA()

            shared.opts.onchange("sd_model_checkpoint",
                                 wrap_queued_call(reload_modelA_checkpoint), call=False)

    def title(self):
        return "Runtime block merging for UNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        process_script_params = []
        with gr.Accordion('Runtime Block Merge', open=False):
            hidden_title = gr.Textbox(label='Runtime Block Merge Title', value='Runtime Block Merge',
                                      visible=False, interactive=False)
            with gr.Row():
                enabled = gr.Checkbox(label='Enable', value=False, interactive=False)
                unload_button = gr.Button(value='Unload and Disable', elem_id="rbm_unload", visible=False)
            experimental_range_checkbox = gr.Checkbox(label='Enable Experimental Range', value=False)
            force_cpu_checkbox = gr.Checkbox(label='Force CPU (Max Precision)', value=True, interactive=True)
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        dd_preset_weight = gr.Dropdown(label="Preset Weights",
                                                       choices=presetWeights.get_preset_name_list())
                        config_paste_button = gr.Button(value='Generate Merge Block Weighted Config\u2199\ufe0f',
                                                        elem_id="rbm_config_paste",
                                                        title="Paste Current Block Configs Into Weight Command. Useful for copying to \"Merge Block Weighted\" extension")
                        weight_command_textbox = gr.Textbox(label="Weight Command",
                                                            placeholder="Input weight command, then press enter. \nExample: base:0.5, in00:1, out09:0.8, time_embed:0, out:0")
                        # weight_config_textbox_readonly = gr.Textbox(label="Weight Config For Merge Block Weighted", interactive=False)

                        # btn_apply_block_weight_from_txt = gr.Button(value="Apply block weight from text")
                        # with gr.Row():
                        #     sl_base_alpha = gr.Slider(label="base_alpha", minimum=0, maximum=1, step=0.01, value=0)
                        #     chk_verbose_mbw = gr.Checkbox(label="verbose console output", value=False)
                        # with gr.Row():
                        #     with gr.Column(scale=3):
                        #         with gr.Row():
                        #             chk_save_as_half = gr.Checkbox(label="Save as half", value=False)
                        #             chk_save_as_safetensors = gr.Checkbox(label="Save as safetensors", value=False)
                        # with gr.Column(scale=4):
                        #     radio_position_ids = gr.Radio(label="Skip/Reset CLIP position_ids",
                        #                                   choices=["None", "Skip", "Force Reset"], value="None",
                        #                                   type="index")
                with gr.Row():
                    # model_A = gr.Dropdown(label="Model A", choices=sd_models.checkpoint_tiles())
                    model_B = gr.Dropdown(label="Model B", choices=sd_models.checkpoint_tiles())
                    refresh_button = gr.Button(variant='tool', value='\U0001f504', elem_id='rbm_modelb_refresh')

                    # txt_model_O = gr.Text(label="Output Model Name")
                with gr.Row():
                    sl_TIME_EMBED = gr.Slider(label="TIME_EMBED", minimum=0, maximum=1, step=0.01, value=0)
                    sl_OUT = gr.Slider(label="OUT", minimum=0, maximum=1, step=0.01, value=0)
                with gr.Row():
                    with gr.Column(min_width=100):
                        sl_IN_00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_IN_11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.01, value=0.5)
                    with gr.Column(min_width=100):
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        gr.Slider(visible=False)
                        sl_M_00 = gr.Slider(label="M00", minimum=0, maximum=1, step=0.01, value=0.5,
                                            elem_id="mbw_sl_M00")
                    with gr.Column(min_width=100):
                        sl_OUT_11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.01, value=0.5)
                        sl_OUT_00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.01, value=0.5)

            sl_INPUT = [
                sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
                sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11]
            sl_MID = [sl_M_00]
            sl_OUTPUT = [
                sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
                sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11]
            sl_ALL_nat = [*sl_INPUT, *sl_MID, sl_OUT, *sl_OUTPUT, sl_TIME_EMBED]
            sl_ALL = [*sl_INPUT, *sl_MID, *sl_OUTPUT, sl_TIME_EMBED, sl_OUT]





            def handle_modelB_load(modelB, force_cpu_checkbox, *slALL):

                load_flag = shared.UNetBManager.load_modelB(modelB, force_cpu_checkbox, slALL)
                if load_flag:
                    return modelB, True, gr.update(interactive=False), gr.update(visible=True), gr.update(visible=True)
                else:
                    return None, False, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False)

            def handle_unload():
                shared.UNetBManager.restore_original_unet()
                shared.UNetBManager.unload_all()
                return None, False, gr.update(interactive=True), gr.update(visible=False), gr.update(visible=False)

            def handle_weight_change(*slALL):
                # convert float list to string+
                slALL_str = [str(sl) for sl in slALL]
                old_config_str = ','.join(slALL_str[:25])
                return old_config_str

            # for slider in sl_ALL:
            #     # slider.change(fn=handle_weight_change, inputs=sl_ALL, outputs=sl_ALL)
            #     slider.change(fn=handle_weight_change, inputs=sl_ALL, outputs=[weight_config_textbox_readonly])


            def on_weight_command_submit(command_str, *current_weights):
                weight_list = parse_weight_str_to_list(command_str, list(current_weights))
                if not weight_list:
                    return [gr.update() for _ in range(27)]
                if len(weight_list) == 25:
                    # noinspection PyTypeChecker
                    weight_list.extend([gr.update(), gr.update()])
                return weight_list

            weight_command_textbox.submit(
                fn=on_weight_command_submit,
                inputs=[weight_command_textbox, *sl_ALL],
                outputs=sl_ALL
            )

            def parse_weight_str_to_list(weightstr, current_weights):
                weightstr = weightstr[:500]
                if ':' in weightstr:
                    # parse as json
                    weightstr = weightstr.replace(' ', '')
                    cmd_segments = weightstr.split(',')
                    constructed_json_segments = [f'"{key.upper()}":{value}' for key, value in
                                                 [x.split(':') for x in cmd_segments]]
                    constructed_json = '{' + ','.join(constructed_json_segments) + '}'
                    try:
                        parsed_json = json.loads(constructed_json)

                    except Exception as e:
                        print(e)
                        return None
                    weight_name_map = {
                        'IN00': 0,
                        'IN01': 1,
                        'IN02': 2,
                        'IN03': 3,
                        'IN04': 4,
                        'IN05': 5,
                        'IN06': 6,
                        'IN07': 7,
                        'IN08': 8,
                        'IN09': 9,
                        'IN10': 10,
                        'IN11': 11,
                        'M00': 12,
                        'OUT00': 13,
                        'OUT01': 14,
                        'OUT02': 15,
                        'OUT03': 16,
                        'OUT04': 17,
                        'OUT05': 18,
                        'OUT06': 19,
                        'OUT07': 20,
                        'OUT08': 21,
                        'OUT09': 22,
                        'OUT10': 23,
                        'OUT11': 24,
                        'TIME_EMBED': 25,
                        'OUT': 26
                    }
                    extra_commands = ['BASE']
                    # type check
                    for key, value in parsed_json.items():
                        if key not in weight_name_map and key not in extra_commands:
                            print(f'invalid key: {key}')
                            return None
                        if not (isinstance(value, (float, int))) or value < -1 or value > 2:
                            print(f'{key} value {value} out of range')
                            return None

                    weight_list = current_weights
                    if 'BASE' in parsed_json:
                        weight_list = [float(parsed_json['BASE'])] * 27
                        del parsed_json['BASE']
                    for key, value in parsed_json.items():
                        weight_list[weight_name_map[key]] = value
                    return weight_list
                else:
                    # parse as list
                    _list = [x.strip() for x in weightstr.split(",")]
                    if len(_list) != 25 and len(_list) != 27:
                        return None
                    validated_float_weight_list = []
                    for x in _list:
                        try:
                            validated_float_weight_list.append(float(x))
                        except ValueError:
                            return None
                    return validated_float_weight_list

            def on_change_dd_preset_weight(preset_weight_name, *current_weights):
                _weights = presetWeights.find_weight_by_name(preset_weight_name)
                weight_list = parse_weight_str_to_list(_weights, list(current_weights))
                if not weight_list:
                    return [gr.update() for _ in range(27)]
                if len(weight_list) == 25:
                    # noinspection PyTypeChecker
                    weight_list.extend([gr.update(), gr.update()])
                return weight_list

            dd_preset_weight.change(
                fn=on_change_dd_preset_weight,
                inputs=[dd_preset_weight, *sl_ALL],
                outputs=sl_ALL
            )

            def update_slider_range(experimental_range_flag):
                if experimental_range_flag:
                    return [gr.update(minimum=-1, maximum=2) for _ in sl_ALL]
                else:
                    return [gr.update(minimum=0, maximum=1) for _ in sl_ALL]

            experimental_range_checkbox.change(fn=update_slider_range, inputs=[experimental_range_checkbox],
                                               outputs=sl_ALL)

            def on_config_paste(*current_weights):
                slALL_str = [str(sl) for sl in current_weights]
                old_config_str = ','.join(slALL_str[:25])
                return old_config_str

            config_paste_button.click(fn=on_config_paste, inputs=[*sl_ALL], outputs=[weight_command_textbox])

            def refresh_modelB_dropdown():
                return gr.update(choices=sd_models.checkpoint_tiles())

            refresh_button.click(
                fn=refresh_modelB_dropdown,
                inputs=None,
                outputs=[model_B]
            )

            # process_script_params.append(hidden_title)
            process_script_params.extend(sl_ALL_nat)
            process_script_params.append(model_B)
            process_script_params.append(enabled)

            with gr.Row():
                output_mode_radio = gr.Radio(label="Output Mode",choices=["Max Precision", "Runtime Snapshot"],
                                             value="Max Precision", type="value", interactive=True)
                position_id_fix_radio = gr.Radio(label="Skip/Reset CLIP position_ids",
                                                 choices=["Keep Original", "Fix"], value="Keep Original", type="value", interactive=True)

                output_format_radio = gr.Radio(label="Output Format",
                                               choices=[".ckpt", ".safetensors"], value=".ckpt", type="value",
                                               interactive=True)
            with gr.Row():
                output_recipe_checkbox = gr.Checkbox(label="Output Recipe", value=True, interactive=True)


            # with gr.Row():
            #     save_snapshot_checkbox = gr.Checkbox(label="Save Snapshot", value=False)
            with gr.Row():
                save_checkpoint_name_textbox = gr.Textbox(label="New Checkpoint Name")
                save_checkpoint_button = gr.Button(value="Save Runtime Checkpoint", elem_id="mbw_save_checkpoint_button", variant='primary', interactive=False, visible=False, )

            def on_save_checkpoint(output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name, output_recipe_checkbox, *weights,
                                   ):
                current_weights_nat = weights[:27]

                weights_output_recipe = weights[27:]
                if not save_checkpoint_name:
                    # current timestamp
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_checkpoint_name = f"mbw_{timestamp_str}"
                save_checkpoint_namewext = save_checkpoint_name + output_format_radio
                loaded_sd_model_path = Path(shared.sd_model.sd_model_checkpoint)
                model_ext = loaded_sd_model_path.suffix
                if model_ext == '.ckpt':

                    model_A_raw_state_dict = torch.load(shared.sd_model.sd_model_checkpoint, map_location='cpu')
                    if 'state_dict' in model_A_raw_state_dict:
                        model_A_raw_state_dict = model_A_raw_state_dict['state_dict']
                elif model_ext == '.safetensors':
                    model_A_raw_state_dict = safetensors.torch.load_file(shared.sd_model.sd_model_checkpoint, device="cpu")
                save_checkpoint_path = Path(shared.sd_model.sd_model_checkpoint).parent / save_checkpoint_namewext

                if output_mode_radio == 'Runtime Snapshot':
                    snapshot_state_dict = shared.sd_model.model.diffusion_model.state_dict()

                elif output_mode_radio == 'Max Precision':
                    snapshot_state_dict = shared.UNetBManager.model_state_construct(current_weights_nat)

                snapshot_state_dict_prefixed = {'model.diffusion_model.' + key: value for key, value in
                                                snapshot_state_dict.items()}
                if not set(snapshot_state_dict_prefixed.keys()).issubset(set(model_A_raw_state_dict.keys())):
                    print(
                        'warning: snapshot state_dict keys are not subset of model A state_dict keys, possible structural deviation')

                combined_state_dict = {**model_A_raw_state_dict, **snapshot_state_dict_prefixed}
                if position_id_fix_radio == 'Fix':
                    combined_state_dict['cond_stage_model.transformer.text_model.embeddings.position_ids'] = torch.tensor([list(range(77))], dtype=torch.int)

                if output_format_radio == '.ckpt':
                    state_dict_save = {'state_dict': combined_state_dict}
                    torch.save(state_dict_save, save_checkpoint_path)
                elif output_format_radio == '.safetensors':
                    safetensors.torch.save_file(combined_state_dict, save_checkpoint_path)

                if output_recipe_checkbox:
                    recipe_path = Path(shared.sd_model.sd_model_checkpoint).parent / f"{save_checkpoint_name}.recipe.txt"
                    with open(recipe_path, 'w') as f:
                        f.write(f"modelA={shared.sd_model.sd_model_checkpoint}\n")
                        f.write(f"modelB={shared.UNetBManager.modelB_path}\n")
                        f.write(f"position_id_fix={position_id_fix_radio}\n")
                        f.write(f"output_mode={output_mode_radio}\n")
                        f.write(f"{','.join([str(w) for w in weights_output_recipe])}\n")

                return gr.update(value=save_checkpoint_name)


            def on_change_force_cpu(force_cpu_flag):
                if not force_cpu_flag:
                    return gr.update(choices=["Runtime Snapshot"], value="Runtime Snapshot")
                else:
                    return gr.update(choices=["Max Precision", "Runtime Snapshot"], value="Max Precision")


            save_checkpoint_button.click(
                fn=on_save_checkpoint,
                inputs=[output_mode_radio, position_id_fix_radio, output_format_radio, save_checkpoint_name_textbox, output_recipe_checkbox, *sl_ALL_nat, *sl_ALL],
                outputs=[save_checkpoint_name_textbox],
                show_progress=True
            )
            force_cpu_checkbox.change(fn=on_change_force_cpu, inputs=[force_cpu_checkbox], outputs=[output_mode_radio])
            model_B.change(fn=handle_modelB_load, inputs=[model_B, force_cpu_checkbox, *sl_ALL_nat],
                           outputs=[model_B, enabled, force_cpu_checkbox, save_checkpoint_button, unload_button])
            unload_button.click(fn=handle_unload, inputs=[], outputs=[model_B, enabled, force_cpu_checkbox, save_checkpoint_button, unload_button])

        return process_script_params

    def process(self, p, *args):
        gui_weights = args[:27]
        modelB = args[27]
        enabled = args[28]
        if not enabled:
            return
        shared.UNetBManager.model_state_apply_modified_blocks(gui_weights, modelB)
