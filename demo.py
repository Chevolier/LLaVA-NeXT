import os
import gradio as gr
import requests
import tempfile
import shutil
from typing import Union

# Services URLs
LLAVA_SERVICE_URL = "http://localhost:8001"
QWEN_SERVICE_URL = "http://localhost:8002"

def inference(prompt, image=None, video=None, max_frames=32, fps=1, max_new_tokens=1024, model_name="LLaVA-Video-7B-Qwen2"):
    """Inference function that calls the appropriate service API."""
    
    service_url = LLAVA_SERVICE_URL if model_name == "LLaVA-Video-7B-Qwen2" else QWEN_SERVICE_URL
    
    # Text-only inference
    if not image and not video:
        response = requests.post(
            f"{service_url}/inference/text", 
            json={
                "prompt": prompt,
                "max_frames": max_frames,
                "fps": fps,
                "max_new_tokens": max_new_tokens
            }
        )
        return response.json()["response"]
    
    # Image inference
    elif image and not video:
        # Create a temporary file if image is not a string path
        temp_dir = None
        if not isinstance(image, str):
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, "image.jpg")
            image.save(image_path)
        else:
            image_path = image
            
        with open(image_path, "rb") as image_file:
            files = {"image": (os.path.basename(image_path), image_file, "image/jpeg")}
            data = {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens
            }
            response = requests.post(
                f"{service_url}/inference/image",
                files=files,
                data=data
            )
        
        # Clean up temp directory if created
        if temp_dir:
            shutil.rmtree(temp_dir)
        
        return response.json()["response"]
    
    # Video inference
    elif video:
        with open(video, "rb") as video_file:
            files = {"video": (os.path.basename(video), video_file, "video/mp4")}
            data = {
                "prompt": prompt,
                "max_frames": max_frames,
                "fps": fps,
                "max_new_tokens": max_new_tokens
            }
            response = requests.post(
                f"{service_url}/inference/video",
                files=files,
                data=data
            )
        
        return response.json()["response"]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Models Demo")
    gr.Markdown("Upload images, videos, or both along with your text prompt to get a response.")
    
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                choices=["LLaVA-Video-7B-Qwen2", "Qwen2.5-VL-7B-Instruct"],
                value="LLaVA-Video-7B-Qwen2",
                label="Select Model"
            )
            prompt = gr.Textbox(label="Enter your prompt", lines=3)
            max_frames = gr.Slider(label="Max video frames", minimum=1, maximum=200, value=32, step=1)
            fps = gr.Slider(label="Video frames per second", minimum=0.5, maximum=10, value=2, step=0.5)
            max_tokens = gr.Slider(label="Max response tokens", minimum=64, maximum=4096, value=1024, step=64)
            
            # Input options
            with gr.Tabs():
                with gr.TabItem("Images"):
                    images_input = gr.Image(label="Upload Images", type="filepath")
                
                with gr.TabItem("Video"):
                    video_input = gr.Video(label="Upload Video")
            
            submit_btn = gr.Button("Generate Response")
        
        with gr.Column():
            output = gr.Textbox(label="Response", lines=15)
    
    # Connect the button to the inference function
    submit_btn.click(
        fn=inference,
        inputs=[prompt, images_input, video_input, max_frames, fps, max_tokens, model_name],
        outputs=[output]
    )
    
    # Examples
    gr.Examples(
        [
            ["Describe what's happening in this video.", None, "data/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03.mp4", 80, 2, 1024, "LLaVA-Video-7B-Qwen2"],
            ["Extract all texts in the image, return only the extracted texts and no other information.", "data/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03frames/LeagueofLegends2025.03.21-10.51.35.03.mp40002.jpeg", None, 80, 2, 1024, "Qwen2.5-VL-7B-Instruct"],
        ],
        fn=inference,
        inputs=[prompt, images_input, video_input, max_frames, fps, max_tokens, model_name],
        outputs=[output],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")