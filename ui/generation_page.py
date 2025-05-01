import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/content/IP-Adapter')

import gradio as gr
from PIL import Image as PILImage
from image_generate.generate import generate_image, generate_image_pure_text, cleanup_gpu, reset_models
from image_generate.prompt_engineering import enrich_prompt_llama
from image_generate.openai_generate import (
    generate_image_openai,
    enrich_prompt_openai,
    extract_style_description
)

from utils.keyword_extractor import extract_keywords
from utils.image_search import search_google_cse_images

GEN_CSS = """
#sidebar-card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 3px 18px 0 #e8eefc;
    padding: 26px 26px 18px 26px;
    margin: 24px 0 18px 18px;
    min-width: 320px;
    max-width: 430px;
}
#gen-main-content {
    padding: 20px 20px 18px 10px;
    background: #f6f7fa;
    min-height: 650px;
}
#output-card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 3px 18px 0 #e8eefc;
    padding: 28px 36px 18px 36px;
    min-height: 600px;
    margin: 18px 0 0 0;
    display: flex; flex-direction: column; align-items: center;
}
.selected-image-box img {
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    max-height: 200px;
    border-radius: 8px;
    margin-bottom: 6px;
}
.gallery .wrap .item {
    width: 90px !important;
    height: 90px !important;
}
.gr-textbox textarea, .gr-markdown {
    font-size: 1.04rem !important;
    line-height: 1.35 !important;
    resize: vertical;
}
@media (max-width: 1000px) {
    #sidebar-card { min-width: 160px; padding: 8px 2px 6px 2px; }
    #gen-main-content { padding: 6px; }
    #output-card { padding: 10px 5px; }
    .selected-image-box img { max-height: 110px; }
}
"""

REFERENCE_IMAGE_PATH = "/content/drive/MyDrive/Image_Generation_Pipeline_code/assets/debug_reference.png"
REFERENCE_IMAGE = PILImage.open(REFERENCE_IMAGE_PATH).convert("RGB")

def is_valid_pil_image(img):
    return isinstance(img, PILImage.Image) and img.size != (0, 0)

def refine_prompt(prompt, current_mode, gen_mode, selected_image, uploaded_image):
    if current_mode == "Pro (OpenAI API)":
        if gen_mode == "Pure Text":
            refined = enrich_prompt_openai(prompt)
            return gr.update(value=refined)
        else:
            return gr.update(value=prompt)
    else:
        refined = enrich_prompt_llama(prompt)
        return gr.update(value=refined)

# === True image search callback ===
def load_reference_images(prompt):
    try:
        print(f"🔎 [Ref Search] Extracting keywords from prompt: {repr(prompt)}")
        keywords = extract_keywords(prompt, max_keywords=5)
        print(f"🔑 [Ref Search] Keywords: {keywords}")
        images = search_google_cse_images(keywords, max_results=8)
        print(f"🖼️ [Ref Search] Retrieved {len(images)} images")
        if not images:
            return [], [], None, None
        gallery_items = [(img, f"Result {i+1}") for i, img in enumerate(images)]
        return gallery_items, images, None, None
    except Exception as e:
        print(f"❌ [Ref Search Error]: {e}")
        return [], [], None, None

def on_image_click(evt: gr.SelectData, all_images):
    print("DEBUG: on_image_click: evt=", evt, "all_images type:", type(all_images), "len:", len(all_images))
    if not all_images or evt.index is None:
        return None, None
    idx = evt.index
    item = all_images[idx]
    if isinstance(item, tuple):
        image = item[0]
    else:
        image = item
    print("DEBUG: Returning image from gallery click: type:", type(image), "size:", getattr(image, "size", None))
    return image, image

def generate_image_router(prompt, selected_image, uploaded_image, style_strength, gen_mode, current_mode):
    print("\n=== GENERATION ROUTER DEBUG ===")
    print("Prompt:", repr(prompt))
    print("Selected Image:", type(selected_image), "|", repr(selected_image)[:80])
    print("Uploaded Image:", type(uploaded_image), "|", repr(uploaded_image)[:80])
    print("Generation Mode:", gen_mode)
    print("Current System Mode:", current_mode)
    print("selected_image is valid PIL?", is_valid_pil_image(selected_image))
    print("uploaded_image is valid PIL?", is_valid_pil_image(uploaded_image))
    print("===============================\n")
    yield None, gr.update(value="🖌️ Generating image...", visible=True), gr.update(value=prompt)

    def assert_pil(img, label="img"):
        assert isinstance(img, PILImage.Image), f"{label} is not PIL.Image.Image! Got {type(img)} | Value: {repr(img)[:80]}"
        assert is_valid_pil_image(img), f"{label} failed is_valid_pil_image: {img}"

    if current_mode == "Pro (OpenAI API)":
        if gen_mode == "Pure Text":
            img = generate_image_openai(prompt)
            yield img, gr.update(value="✅ Done!", visible=True), gr.update(value=prompt)
            cleanup_gpu()
            return
        else:
            ref_img = selected_image if gen_mode == "Reference: Search Result" else uploaded_image
            if not is_valid_pil_image(ref_img):
                yield None, gr.update(value="⚠️ Please provide a reference image."), gr.update(value=prompt)
                return
            style_desc = extract_style_description(ref_img)
            full_prompt = f"{prompt}\n\nIn the style of: {style_desc}"
            img = generate_image_openai(full_prompt)
            yield img, gr.update(value="✅ Done!", visible=True), gr.update(value=full_prompt)
            cleanup_gpu()
            return

    if gen_mode == "Reference: Search Result":
        if not is_valid_pil_image(selected_image):
            yield None, gr.update(value="⚠️ Please select a reference image."), gr.update(value=prompt)
            return
        assert_pil(selected_image, label="selected_image")
        img = generate_image(prompt, selected_image, style_strength=style_strength)
    elif gen_mode == "Reference: User Upload":
        if not is_valid_pil_image(uploaded_image):
            yield None, gr.update(value="⚠️ Please upload a reference image."), gr.update(value=prompt)
            return
        assert_pil(uploaded_image, label="uploaded_image")
        img = generate_image(prompt, uploaded_image, style_strength=style_strength)
    elif gen_mode == "Pure Text":
        img = generate_image_pure_text(prompt)
    else:
        img = None

    yield img, gr.update(value="✅ Done!", visible=True), gr.update(value=prompt)
    cleanup_gpu()

def reset_state_keep_prompt(prompt):
    cleanup_gpu()
    return (
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None),
        gr.update(value="", visible=False),
        gr.update(value=prompt)
    )

def generation_page():
    with gr.Blocks(css=GEN_CSS, elem_id="main-bg", title="Image Generator Assistance") as group:
        with gr.Row():
            # Sidebar card (controls)
            with gr.Column(scale=1, min_width=320):
                with gr.Column(elem_id="sidebar-card"):
                    gr.Markdown("#### 📝 Compose Prompt")
                    prompt_input = gr.Textbox(label=None, placeholder="E.g. cartoon of AI agents collaborating...", lines=3)
                    refine_btn = gr.Button("✨ Refine Prompt (Llama/GPT)", elem_id="refine-btn")

                    gr.Markdown("#### 🧠 System Mode")
                    mode_toggle = gr.Radio(
                        choices=["Lite (Local)", "Pro (OpenAI API)"],
                        value="Lite (Local)",
                        interactive=True,
                        show_label=False
                    )

                    gr.Markdown("#### 🎨 Generation Mode")
                    gen_mode = gr.Radio(
                        choices=["Pure Text", "Reference: Search Result", "Reference: User Upload"],
                        value="Pure Text",
                        interactive=True,
                        show_label=False
                    )
                    search_btn = gr.Button("🔎 Search Reference Image", visible=False)
                    uploaded_image = gr.Image(label="Upload Reference", visible=False, type="pil", elem_id="upload-img", height=100)
                    gallery = gr.Gallery(label="Select Reference", columns=2, rows=1, height=120, visible=False)

                    all_images_state = gr.State([])
                    selected_image_state = gr.State(None)

                    with gr.Column(elem_classes="selected-image-box"):
                        selected_image = gr.Image(
                            label="Preview Selected",
                            visible=False,
                            type="pil",
                            elem_id="selected-image",
                            height=200
                        )

                    style_slider = gr.Slider(
                        label="Style Influence",
                        minimum=0.2,
                        maximum=1.2,
                        value=0.7,
                        step=0.05,
                        visible=False
                    )

                    generate_btn = gr.Button("🚀 Generate Image", elem_id="generate-btn", visible=True)

            with gr.Column(scale=3, min_width=520, elem_id="gen-main-content"):
                with gr.Column(elem_id="output-card"):
                    gr.Markdown("#### 🖼️ Result")
                    output_image = gr.Image(
                        label="Generated Result",
                        visible=True,
                        type="pil",
                        height=420,
                        show_label=False,
                        show_download_button=True,
                        value=None
                    )
                    final_prompt_display = gr.Textbox(
                        label="Final Prompt Used",
                        lines=3,
                        interactive=False,
                        visible=True,
                        elem_id="final-prompt"
                    )
                    status_text = gr.Markdown("", visible=False)
                    pro_info = gr.Markdown(
                        "<div style='color:#ffa500;'><b>Note:</b> In <b>Pro mode + Reference image</b>, prompt will be auto-enriched using your reference image and sent to DALL·E 3. Manual refinement is disabled.</div>",
                        visible=False
                    )

        def update_visibility(mode, prompt_val, system_mode):
            reset_models()
            show_refine_btn = (system_mode == "Lite (Local)") or (mode == "Pure Text")
            show_pro_info = (system_mode == "Pro (OpenAI API)") and (mode != "Pure Text")
            return {
                search_btn: gr.update(visible=(mode == "Reference: Search Result")),
                gallery: gr.update(visible=(mode == "Reference: Search Result")),
                selected_image: gr.update(visible=(mode == "Reference: Search Result")),
                uploaded_image: gr.update(visible=(mode == "Reference: User Upload")),
                style_slider: gr.update(visible=(mode != "Pure Text") and (system_mode == "Lite (Local)")),
                prompt_input: gr.update(value=prompt_val),
                refine_btn: gr.update(visible=show_refine_btn),
                pro_info: gr.update(visible=show_pro_info)
            }

        def visibility_wrapper(gen_mode_val, prompt_val, system_mode_val):
            return update_visibility(gen_mode_val, prompt_val, system_mode_val)

        mode_toggle.change(
            fn=visibility_wrapper,
            inputs=[gen_mode, prompt_input, mode_toggle],
            outputs=[search_btn, gallery, selected_image, uploaded_image, style_slider, prompt_input, refine_btn, pro_info]
        )

        gen_mode.change(
            fn=visibility_wrapper,
            inputs=[gen_mode, prompt_input, mode_toggle],
            outputs=[search_btn, gallery, selected_image, uploaded_image, style_slider, prompt_input, refine_btn, pro_info]
        )

        search_btn.click(
            fn=load_reference_images,
            inputs=prompt_input,
            outputs=[gallery, all_images_state, selected_image, selected_image_state]
        )

        gallery.select(
            fn=on_image_click,
            inputs=[all_images_state],
            outputs=[selected_image, selected_image_state]
        )

        refine_btn.click(
            fn=refine_prompt,
            inputs=[prompt_input, mode_toggle, gen_mode, selected_image, uploaded_image],
            outputs=prompt_input
        )

        generate_btn.click(
            fn=generate_image_router,
            inputs=[prompt_input, selected_image_state, uploaded_image, style_slider, gen_mode, mode_toggle],
            outputs=[output_image, status_text, final_prompt_display],
            show_progress=False
        )

    return group
