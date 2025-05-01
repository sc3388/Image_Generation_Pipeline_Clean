import os
import sys
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

# === Ensure Colab can import the IP-Adapter code from /content/IP-Adapter/ip_adapter ===
if '/content/IP-Adapter' not in sys.path:
    sys.path.insert(0, '/content/IP-Adapter')
from ip_adapter import IPAdapterXL

# === Cacheable global models ===
_sdxl_pipe = None
_ip_model = None

# === LoRA Path ===
LORA_PATH = "/content/drive/MyDrive/Image_Generation_Pipeline_code/models/samsung_line_art.safetensors"

def reset_models():
    global _sdxl_pipe, _ip_model
    _sdxl_pipe = None
    _ip_model = None
    print("[RESET] Unloaded SDXL and IPAdapter models.")
    cleanup_gpu()

def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# === Helper: Load LoRA weights into a pipeline ===
def load_lora_weights(pipe, lora_path, lora_scale=1.3):
    """Apply LoRA to a pipeline."""
    lora_state_dict = load_file(lora_path)
    compatible_keys = [k for k in lora_state_dict.keys() if 'lora' in k]
    if hasattr(pipe, 'unet'):
        pipe.unet.load_state_dict({k.replace('unet.', ''): lora_state_dict[k] for k in compatible_keys if 'unet' in k}, strict=False)
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder.load_state_dict({k.replace('text_encoder.', ''): lora_state_dict[k] for k in compatible_keys if 'text_encoder' in k}, strict=False)
    print(f"[LOAD] LoRA {os.path.basename(lora_path)} applied at scale {lora_scale}")
    return pipe

# === One-time setup: Load SDXL pipeline + IPAdapterXL wrapper ===
def load_ip_adapter_model(device="cuda"):
    global _ip_model

    if _ip_model is None:
        print("[LOAD] Loading IP-AdapterXL + SDXL pipeline...")

        # Load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            add_watermarker=False,
        )
        pipe.safety_checker = None  # Optionally disable safety checker

        # === Load SamsungLineart2 LoRA ===
        pipe = load_lora_weights(pipe, LORA_PATH, lora_scale=1.3)

        # Adapter weights and vision encoder folder
        ip_ckpt = "/content/IP-Adapter/sdxl_models/ip-adapter_sdxl.safetensors"
        image_encoder_path = "/content/IP-Adapter/sdxl_models/image_encoder"

        _ip_model = IPAdapterXL(
            pipe,
            image_encoder_path,
            ip_ckpt,
            device=device
        )

    return _ip_model

# === Generate with IP-AdapterXL reference image ===
def generate_image(prompt, reference_image, style_strength=0.7, size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cleanup_gpu()
    model = load_ip_adapter_model(device)

    images = model.generate(
        pil_image=reference_image,
        prompt=prompt,
        ip_adapter_scale=style_strength,
        height=size,
        width=size,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    print("Image generation finished, result passed")
    return images[0]

# === Generate pure text-based image ===
def generate_image_pure_text(prompt, size=512):
    global _sdxl_pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cleanup_gpu()

    if _sdxl_pipe is None:
        print("[LOAD] Loading SDXL pipeline (pure text)...")
        _sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        _sdxl_pipe.safety_checker = None

        # === Load SamsungLineart2 LoRA into pure text pipeline ===
        _sdxl_pipe = load_lora_weights(_sdxl_pipe, LORA_PATH, lora_scale=1.3)

    print("[RUN] Generating image from pure text prompt...")
    return _sdxl_pipe(prompt, height=size, width=size, num_inference_steps=30, guidance_scale=7.5).images[0]