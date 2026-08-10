"""
Public-facing ViCCT crowd-counting demo — self-hosted version.

This mirrors notebooks/Make_image_prediction.ipynb exactly (same model, same
preprocessing, same overlay math) but swaps the notebook's file-drop workflow
for a Gradio UI, running as a normal always-on Python process on your own
server (no ZeroGPU / Hugging Face Spaces machinery needed).

Expected project layout (same as the main ViCCT project):
    app.py                  <- this file
    models/
        __init__.py
        Swin_ViCCT_models.py
    datasets/
        __init__.py
        dataset_utils.py
    requirements.txt
"""

import os
import shutil

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from huggingface_hub import hf_hub_download
from matplotlib import cm
from PIL import Image
from timm.models import create_model

import models.Swin_ViCCT_models  # noqa: F401  (registers 'Swin_ViCCT_large_22k' with timm)
from datasets.dataset_utils import img_equal_split, img_equal_unsplit

# --------------------------------------------------------------------------- #
# Config — copied from the notebook's settings cell. Change here if you swap
# to a different trained checkpoint / model_name.
# --------------------------------------------------------------------------- #
MODEL_NAME = "Swin_ViCCT_large_22k"
WEIGHTS_PATH = "models/trained_models/Swin_ViCCT_large_22k_generic_1600_epochs.pth"
HF_REPO_ID = "Lauraponoran/vicct-swin-large-22k"

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean/std
OVERLAP = 32          # min pixels of overlap between adjacent crops
IGNORE_BUFFER = 16    # pixels ignored at crop borders when reconstructing
CROP_SIZE = 224
DEFAULT_SCALE_FACTOR = 0.5   # bumped from 0.4 on 2026-08-10 to match the
                             # resolution the working dashboard actually feeds

# Density-map -> headcount calibration. The model's raw output sums to some
# multiple of the true head count rather than the count itself, so this
# divides it back down. Was briefly recalibrated to 4365 on 2026-08-10, but
# that was likely compensating for test images being fed in at a higher
# resolution than the working dashboard's real pipeline uses, not a genuine
# model miscalibration. Reverted to the notebook's original 3000 now that
# DEFAULT_SCALE_FACTOR is set to match. Re-tune if counts drift again:
# new_divisor = old_divisor * (shown_count / true_count).
COUNT_SCALE_DIVISOR = 3000

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*MEAN_STD),
])


# --------------------------------------------------------------------------- #
# Model setup — runs once at Space startup, on CPU. Every request in
# predict() just moves this same model to whatever device ZeroGPU hands us.
# --------------------------------------------------------------------------- #
def _ensure_weights() -> str:
    """Download the checkpoint from the Hub the first time the Space boots
    (same fallback the notebook uses), skip if it's already on disk."""
    if not os.path.exists(WEIGHTS_PATH):
        print("Checkpoint not found locally, downloading from Hugging Face...")
        downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=os.path.basename(WEIGHTS_PATH))
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        shutil.copy(downloaded, WEIGHTS_PATH)
        print(f"Downloaded to {WEIGHTS_PATH}")
    else:
        print(f"Checkpoint already present at {WEIGHTS_PATH}")
    return WEIGHTS_PATH


def _load_model() -> torch.nn.Module:
    weights_path = _ensure_weights()
    model = create_model(
        MODEL_NAME,
        init_path=weights_path,
        pretrained_cc=True,
        # Swin path takes drop_rate/drop_path_rate=None; see the notebook + the
        # _clean_kwargs() comment in Swin_ViCCT_models.py for why.
        drop_rate=None,
        drop_path_rate=None,
        drop_block_rate=None,
    )
    model.eval()
    return model


MODEL = _load_model()


# --------------------------------------------------------------------------- #
# Post-processing — same math as the notebook's display + EXPERIMENTAL cells.
# --------------------------------------------------------------------------- #
def _visible_density_mask(den: torch.Tensor, threshold: float = 50) -> np.ndarray:
    """Which pixels actually show up in the heatmap overlay, i.e. cleared the
    same normalize -> sqrt -> threshold pipeline used for display. Pulled out
    on its own so predict() can total only the density that's visibly 'real'
    signal, instead of summing background noise the eye never sees."""
    den_heat = den.clone().numpy() / COUNT_SCALE_DIVISOR
    den_heat[den_heat < 0] = 0

    max_val = den_heat.max()
    if max_val > 0:
        den_heat = den_heat / max_val
    den_heat **= 0.5
    den_heat *= 255

    return den_heat >= threshold


def _heatmap_overlay(input_image: Image.Image, den: torch.Tensor, mask: np.ndarray) -> Image.Image:
    img_heat = np.array(input_image).copy()
    den_heat = den.clone().numpy() / COUNT_SCALE_DIVISOR
    den_heat[den_heat < 0] = 0

    max_val = den_heat.max()
    if max_val > 0:
        den_heat = den_heat / max_val
    den_heat **= 0.5
    den_heat *= 255
    den_heat[~mask] = 0

    img_heat[:, :, 0][mask] = img_heat[:, :, 0][mask] / 2
    img_heat[:, :, 1][mask] = img_heat[:, :, 1][mask] / 2
    img_heat[:, :, 2][mask] = den_heat[mask]

    return Image.fromarray(img_heat.astype(np.uint8))


def _density_map_image(den: torch.Tensor) -> Image.Image:
    """Render the raw density map with the same 'jet' colormap the notebook
    uses via plt.imshow(den, cmap=cm.jet)."""
    den_np = den.numpy()
    den_np = den_np - den_np.min()
    max_val = den_np.max()
    if max_val > 0:
        den_np = den_np / max_val
    colored = (cm.jet(den_np) * 255).astype(np.uint8)[:, :, :3]
    return Image.fromarray(colored)


# --------------------------------------------------------------------------- #
# Inference. Model is built on CPU above; here we move it to whatever device
# is actually available. On a plain UpCloud server with no NVIDIA GPU, that's
# always CPU — slower per image than a GPU, but the 1-core/2GB box has no
# trouble with the ~1.2GB the model needs to load.
# --------------------------------------------------------------------------- #
def predict(image: Image.Image, scale_factor: float):
    if image is None:
        raise gr.Error("Upload an image first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL.to(device)

    input_image = image.convert("RGB")
    if scale_factor != 1.0:
        w, h = input_image.size
        input_image = input_image.resize((round(w * scale_factor), round(h * scale_factor)))

    img_w, img_h = input_image.size
    img_t = img_transform(input_image)
    img_stack = img_equal_split(img_t, CROP_SIZE, OVERLAP).to(device)

    pred_stack = torch.zeros(img_stack.shape[0], 1, CROP_SIZE, CROP_SIZE)

    with torch.no_grad():
        for idx, img_crop in enumerate(img_stack):
            pred_stack[idx] = MODEL.forward(img_crop.unsqueeze(0)).cpu()

    den = img_equal_unsplit(pred_stack, OVERLAP, IGNORE_BUFFER, img_h, img_w, 1)
    den = den.squeeze()

    mask = _visible_density_mask(den)
    den_clamped = den.clamp(min=0).numpy()
    pred_count = float(den_clamped[mask].sum() / COUNT_SCALE_DIVISOR)

    heatmap = _heatmap_overlay(input_image, den, mask)
    density_map = _density_map_image(den)

    return heatmap, density_map, f"Predicted count: {pred_count:.1f}"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
custom_css = """
footer {visibility: hidden}
"""

with gr.Blocks(title="ViCCT Crowd Counting") as demo:
    gr.Markdown(
        "# ViCCT Crowd Counting\n"
        "Upload a crowd photo to get a predicted head count and density map, "
        "using the Swin-based ViCCT model (ImageNet-22k pretrained backbone, "
        "trained on generic crowd-counting data)."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input image")
            scale_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=DEFAULT_SCALE_FACTOR, step=0.05,
                label="Scale factor",
                info=(
                    "Downscale large images so heads aren't too large for the model to "
                    "recognise. 0.4 matches the notebook's default."
                ),
            )
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column():
            heatmap_output = gr.Image(label="Heatmap overlay")
            density_output = gr.Image(label="Raw density map")
            count_output = gr.Textbox(label="Result")

    run_btn.click(
        fn=predict,
        inputs=[image_input, scale_slider],
        outputs=[heatmap_output, density_output, count_output],
    )

if __name__ == "__main__":
    # 0.0.0.0 so the app is reachable from outside this server, not just
    # localhost. Change the port here (and in the systemd unit / firewall
    # rule) together if you need a different one.
    demo.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
