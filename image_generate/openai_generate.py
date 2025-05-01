import os
import openai
from PIL import Image
import io
import requests
import base64

from utils.image_search import load_api_key

# === Load OpenAI key (try ENV first, then file) ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    try:
        # Colab: e.g., "/content/Image Generation Pipeline_code/credentials/google_keys.txt"
        OPENAI_API_KEY = load_api_key(
            path=os.getenv("OPENAI_API_KEY_FILE", "/content/drive/MyDrive/Image_Generation_Pipeline_code/credentials/google_keys.txt"), label="GPT"
        )
    except Exception as e:
        print("[OPENAI] Failed to load API key:", e)
        OPENAI_API_KEY = None

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def set_openai_key(key: str):
    global OPENAI_API_KEY
    OPENAI_API_KEY = key
    openai.api_key = key

def generate_image_openai(prompt, size=1024, model="dall-e-3"):
    print(f"[OPENAI] Requesting DALL·E 3 image: prompt={prompt}")
    try:
        response = openai.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=f"{size}x{size}",
            response_format="url"
        )
        image_url = response.data[0].url
        image_bytes = requests.get(image_url).content
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print("[OPENAI] Error during generation:", e)
        return None

def enrich_prompt_openai(prompt, model="gpt-4o"):
    print(f"[OPENAI] Calling GPT for prompt engineering: {prompt}")
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert image prompt engineer. Enhance the user's prompt for DALL·E 3 to achieve a creative, detailed result."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )
        refined = completion.choices[0].message.content
        print(f"[OPENAI] GPT enriched prompt: {refined}")
        return refined
    except Exception as e:
        print("[OPENAI] Error during prompt enrichment:", e)
        return prompt

def extract_style_description(reference_image, model="gpt-4o"):
    print("[OPENAI] Extracting style description from reference image (GPT-4o vision).")
    try:
        img_bytes = io.BytesIO()
        reference_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        b64_img = base64.b64encode(img_bytes.read()).decode('utf-8')
        messages = [
            {"role": "system", "content": "You are an art expert. Describe the visual style, color palette, and illustration technique of the uploaded image for use in image prompt engineering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the style, color, and mood of this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_img}"
                        }
                    }
                ]
            }
        ]
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.6
        )
        description = completion.choices[0].message.content
        print(f"[OPENAI] Extracted style: {description}")
        return description
    except Exception as e:
        print("[OPENAI] Error during style extraction:", e)
        return ""

def generate_image_openai_with_reference(prompt, reference_image, size=1024, model="dall-e-3"):
    print("[OPENAI] Reference-based generation (style transfer simulation).")
    style_desc = extract_style_description(reference_image)
    full_prompt = f"{prompt}\n\nIn the style of: {style_desc}"
    return generate_image_openai(full_prompt, size=size, model=model)
