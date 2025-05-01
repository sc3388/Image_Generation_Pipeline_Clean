import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/content/IP-Adapter')


import gradio as gr
from generation_page import generation_page
from edit_page import edit_page

with gr.Blocks(title="Image Generator Assistance") as demo:
    with gr.Tabs():
        with gr.TabItem("🎨 Generate", id="generate-tab"):
            generation_page()
        with gr.TabItem("✏️ Edit", id="edit-tab"):
            edit_page()

if __name__ == "__main__":
    demo.launch(share = True)

#$env:PYTHONPATH="C:\Cornell\ENGMT 5400\Final Project\Image Generation Pipeline"