import modules.scripts as scripts
import gradio as gr
from modules import sd_models, shared
from scripts.mbw_util.preset_weights import PresetWeights
import torch

presetWeights = PresetWeights()


class UNetBlockManager(object):
    def __init__(self, org_unet: torch.nn.Module):
        super().__init__()
        self.torch_unet = org_unet
        self.modelA_block_weights = []


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        # getting reference for model A
        self.model_A = shared.sd_model

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
                        btn_apply_block_weithg_from_txt = gr.Button(value="Apply block weight from text",
                                                                    variant="primary")
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
