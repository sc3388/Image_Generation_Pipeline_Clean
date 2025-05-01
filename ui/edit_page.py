import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from gradio_image_annotation import image_annotator

import sys
sys.path.insert(0, '/content/IP-Adapter')

# For Colab: download if missing, and always use /content path
MODEL_ID = "/content/sdxl-inpaint"
HF_REPO_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

import os
if not os.path.exists(MODEL_ID):
    from huggingface_hub import snapshot_download
    print("Downloading SDXL inpainting model to /content/sdxl-inpaint...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=MODEL_ID, repo_type="model", ignore_patterns=["*.msgpack", "*.bin"])

pipe = None

# === LoRA path for edit pipeline ===
LORA_PATH = "/content/drive/MyDrive/Image_Generation_Pipeline_code/models/samsung_line_art.safetensors"

# === Helper to load LoRA weights into pipe ===
from safetensors.torch import load_file
def load_lora_weights(pipe, lora_path, lora_scale=0.8):
    lora_state_dict = load_file(lora_path)
    compatible_keys = [k for k in lora_state_dict.keys() if 'lora' in k]
    if hasattr(pipe, 'unet'):
        pipe.unet.load_state_dict({k.replace('unet.', ''): lora_state_dict[k] for k in compatible_keys if 'unet' in k}, strict=False)
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder.load_state_dict({k.replace('text_encoder.', ''): lora_state_dict[k] for k in compatible_keys if 'text_encoder' in k}, strict=False)
    print(f"[LOAD] LoRA {os.path.basename(lora_path)} applied at scale {lora_scale}")
    return pipe

def load_sdxl_inpaint():
    global pipe
    if pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            add_watermarker=False
        ).to(device)
        pipe.safety_checker = None

        # === Inject LoRA after loading Inpaint Model ===
        pipe = load_lora_weights(pipe, LORA_PATH, lora_scale=1.3)
        
    return pipe

def box_to_mask(box, shape):
    """Generate a binary mask from a box annotation and image shape."""
    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
    return mask

def create_with_edit(annotation_data):
    # === Debug block for development ===
    debug_info = f"annotation_data = {repr(annotation_data)}\n"
    boxes = annotation_data.get("boxes", [])
    debug_info += f"\nFound {len(boxes)} boxes:\n"
    for i, box in enumerate(boxes, 1):
        debug_info += f"Box {i}: color={box.get('color')}, label='{box.get('label')}', xy=({box['xmin']},{box['ymin']})-({box['xmax']},{box['ymax']})\n"
    # ==========================

    if annotation_data is None or "image" not in annotation_data or not boxes:
        return None, None, None, debug_info + "[No boxes detected!]"

    orig_img = annotation_data["image"]
    if isinstance(orig_img, np.ndarray):
        orig_img = Image.fromarray(orig_img)
    elif not isinstance(orig_img, Image.Image):
        return None, None, None, debug_info + "[Image type error.]"

    # For mask overlay
    overlay_img = orig_img.convert("RGBA")
    mask_overlay = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))

    img_for_edit = orig_img.copy().convert("RGB")
    pipe = load_sdxl_inpaint()
    summary = ""
    
    for i, box in enumerate(boxes, 1):
        prompt = box.get('label', '').strip()
        if not prompt:
            continue
        color = box.get('color', (255, 0, 0))
        mask = box_to_mask(box, orig_img.size[::-1])  # shape = (height, width)
        # Visualize this mask in overlay (with some transparency)
        overlay_rgba = color if isinstance(color, tuple) else (255, 0, 0)
        overlay_rgba = overlay_rgba[:3] + (120,)  # (R,G,B,alpha)
        color_layer = Image.new("RGBA", overlay_img.size, (0, 0, 0, 0))
        color_layer.paste(Image.new("RGBA", overlay_img.size, overlay_rgba), mask=mask)
        mask_overlay = Image.alpha_composite(mask_overlay, color_layer)

        # === Run SDXL inpaint for this mask region ===
        try:
            img_for_edit = pipe(
                prompt=prompt,
                image=img_for_edit,
                mask_image=mask,
                guidance_scale=15,
            ).images[0]
            summary += f"Box {i}: color={color}, prompt='{prompt}' (edited!)\n"
        except Exception as e:
            summary += f"Box {i}: Error: {str(e)}\n"
    
    overlay_preview = Image.alpha_composite(overlay_img, mask_overlay).convert("RGB")

    return orig_img, overlay_preview, img_for_edit, summary + "\n" + debug_info

def clear_editor():
    return None, ""

def edit_page():
    with gr.Blocks() as demo:
        gr.Markdown("## 📝 Multi-Box Edit with SDXL Inpainting (`gradio-image-annotation` + SDXL)")

        with gr.Row():
            with gr.Column(scale=2, min_width=480):
                annotation = image_annotator(
                    label="Upload and draw boxes, then label each region for SDXL editing",
                    show_label=True,
                )
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Editor", scale=1)
                    create_btn = gr.Button("✨ Create with Edit", scale=2)
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("#### Original Image")
                output_orig = gr.Image(label="Original", elem_id="edit-original", height=180, show_download_button=True)
                gr.Markdown("#### Mask Overlay (All Boxes)")
                output_edit = gr.Image(label="Mask Overlay", elem_id="edit-edited", height=180, show_download_button=True)
                gr.Markdown("#### Generated Result")
                output_gen = gr.Image(label="Edited Output", elem_id="edit-result", height=230, show_download_button=True)
                summary_box = gr.Textbox(label="Debug Info", elem_id="edit-summary", lines=14, interactive=False)

        clear_btn.click(
            fn=clear_editor,
            outputs=[annotation, summary_box]
        )
        create_btn.click(
            fn=create_with_edit,
            inputs=[annotation],
            outputs=[output_orig, output_edit, output_gen, summary_box]
        )
    return demo
