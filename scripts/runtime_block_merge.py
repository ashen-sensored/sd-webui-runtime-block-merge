import modules.scripts as scripts
import gradio as gr

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from modules import sd_models, shared
from scripts.mbw_util.preset_weights import PresetWeights
import torch
from natsort import natsorted

presetWeights = PresetWeights()

shared.UNetBManager = None


class UNetStateManager(object):
    def __init__(self, org_unet: UNetModel):
        super().__init__()
        self.modelB_state_dict_by_blocks = []
        self.torch_unet = org_unet
        self.modelA_state_dict = org_unet.state_dict()
        self.modelA_state_dict_by_blocks = []
        self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        self.modelB_state_dict = None

    def load_modelB(self, modelB_path, *current_weights):
        model_info = sd_models.get_closet_checkpoint_match(modelB_path)
        checkpoint_file = model_info.filename
        sd_model_hash = model_info.hash
        cache_enabled = shared.opts.sd_checkpoint_cache > 0

        if cache_enabled and model_info in sd_models.checkpoints_loaded:
            # use checkpoint cache
            print(f"Loading weights [{sd_model_hash}] from cache")
            self.modelB_state_dict = sd_models.checkpoints_loaded[model_info]
        self.modelB_state_dict = self.filter_unet_state_dict(sd_models.read_state_dict(checkpoint_file))
        self.modelB_state_dict_by_blocks = []
        self.map_blocks(self.modelB_state_dict, self.modelB_state_dict_by_blocks)

        print('model B loaded')

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
        current_block_index = 0
        processing_block_dict = {}
        for key in model_state_dict:
            # print(key)
            if not key.startswith(known_block_prefixes[current_block_index]):
                if not key.startswith(known_block_prefixes[current_block_index+1]):
                    print(
                        f"{key} in statedict after block {known_block_prefixes[current_block_index]}, possible UNet structure deviation"
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


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        if shared.UNetBManager is None:
            shared.UNetBManager = UNetStateManager(shared.sd_model.model.diffusion_model)
        # getting reference for model A

    def title(self):
        return "Runtime block merging for UNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Runtime Block Merge', open=False):
            enabled = gr.Checkbox(label='Enable', value=False)
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        dd_preset_weight = gr.Dropdown(label="Preset Weights",
                                                       choices=presetWeights.get_preset_name_list())
                        txt_block_weight = gr.Text(label="Weight values",
                                                   placeholder="Put weight sets. float number x 25")
                        btn_apply_block_weithg_from_txt = gr.Button(value="Apply block weight from text")
                        with gr.Row():
                            sl_base_alpha = gr.Slider(label="base_alpha", minimum=0, maximum=1, step=0.01, value=0)
                            chk_verbose_mbw = gr.Checkbox(label="verbose console output", value=False)
                        with gr.Row():
                            with gr.Column(scale=3):
                                with gr.Row():
                                    chk_save_as_half = gr.Checkbox(label="Save as half", value=False)
                                    chk_save_as_safetensors = gr.Checkbox(label="Save as safetensors", value=False)
                            with gr.Column(scale=4):
                                radio_position_ids = gr.Radio(label="Skip/Reset CLIP position_ids",
                                                              choices=["None", "Skip", "Force Reset"], value="None",
                                                              type="index")
                with gr.Row():
                    # model_A = gr.Dropdown(label="Model A", choices=sd_models.checkpoint_tiles())
                    model_B = gr.Dropdown(label="Model B", choices=sd_models.checkpoint_tiles())
                    txt_model_O = gr.Text(label="Output Model Name")
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

            sl_IN = [
                sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
                sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11]
            sl_MID = [sl_M_00]
            sl_OUT = [
                sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
                sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11]
            sl_ALL = [*sl_IN, *sl_MID, *sl_OUT]

            def handle_modelB_load(modelB, *slALL):
                if model_B == 'None':
                    return
                shared.UNetBManager.load_modelB(modelB)
                pass
            model_B.change(fn=handle_modelB_load, inputs=[model_B,*sl_ALL], outputs=None)