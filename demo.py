import os
import gradio as gr
from typing import List, Union
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np

# Configuration
MODEL_NAME = "LLaVA-Video-7B-Qwen2"  # Update with your model name
MODEL_DIR = "/home/ec2-user/SageMaker/efs/Models/LLaVA-Video-7B-Qwen2"  # Update with your model directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto" if DEVICE == "cuda" else None

# Load the model
tokenizer, model, image_processor, max_length = load_pretrained_model(
    MODEL_DIR, None, MODEL_NAME, 
    torch_dtype="bfloat16", 
    device_map=DEVICE_MAP
)
model.eval()

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load video frames from a video file."""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps_ratio = round(vr.get_avg_fps()/fps)
    
    # Get frame indices
    frame_idx = [i for i in range(0, len(vr), fps_ratio)]
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    # Sample frames if needed
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, frame_time_str, video_time

def process_images(images):
    """Process multiple images for the model."""
    if not images:
        return None
    
    processed_images = []
    for img in images:
        if isinstance(img, str):  # If it's a file path
            img = Image.open(img).convert('RGB')
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert('RGB')
            
        # Process each image
        processed = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(DEVICE)
        if DEVICE == "cuda":
            processed = processed.bfloat16()
        processed_images.append(processed)
    
    return processed_images if processed_images else None

def process_video(video_path, max_frames=80, fps=2):
    """Process video for the model."""
    if not video_path:
        return None, "", 0
    
    # Load video frames
    video_frames, frame_time, video_time = load_video(
        video_path, max_frames_num=max_frames, fps=fps, force_sample=True
    )
    
    # Process frames
    processed_video = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(DEVICE)
    if DEVICE == "cuda":
        processed_video = processed_video.bfloat16()
        
    return [processed_video], frame_time, video_time

def inference(prompt, image=None, video=None, max_frames=32, fps=1, max_new_tokens=1024):
    """Run inference with the model."""
    conv_template = "qwen_1_5"  # Use appropriate template for your model
    modalities = []
    
    # Prepare the conversation
    conv = copy.deepcopy(conv_templates[conv_template])
    
    # Process video if provided
    video_data, frame_time, video_time = None, "", 0
    if video:
        video_data, frame_time, video_time = process_video(video, max_frames, fps)
        if video_data:
            modalities.append("video")
    
    # Process images if provided
    image_data = None
    if image:
        image_data = process_images([image])
        if image_data:
            modalities.append("image")
    
    # Prepare the input based on what's provided
    if video_data:
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video_data[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
    elif image_data:
        question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
    else:
        question = prompt
    
    # Build the conversation prompt
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)
    
    # Run the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=video_data if video_data else image_data,
            modalities=modalities,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
        )
    
    # Decode output
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract just the assistant's response
    response_prefix = conv.roles[1] + ": "
    if response_prefix in response:
        response = response.split(response_prefix)[-1].strip()
    
    return response

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# LLaVA-Video Demo")
    gr.Markdown("Upload images, videos, or both along with your text prompt to get a response.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Enter your prompt", lines=3)
            max_frames = gr.Slider(label="Max video frames", minimum=1, maximum=200, value=32, step=1)
            fps = gr.Slider(label="Video frames per second", minimum=0.5, maximum=10, value=2, step=0.5)
            max_tokens = gr.Slider(label="Max response tokens", minimum=64, maximum=4096, value=1024, step=64)
            
            # Input options
            with gr.Tabs():
                with gr.TabItem("Images"):
                    # images_input = gr.File(file_count="multiple", label="Upload Images", type="filepath")
                    images_input = gr.Image(label="Upload Images", type="filepath")
                
                with gr.TabItem("Video"):
                    video_input = gr.Video(label="Upload Video")
            
            submit_btn = gr.Button("Generate Response")
        
        with gr.Column():
            output = gr.Textbox(label="Response", lines=15)
    
    # Connect the button to the inference function
    submit_btn.click(
        fn=inference,
        inputs=[prompt, images_input, video_input, max_frames, fps, max_tokens],
        outputs=[output]
    )
    
    # Examples
    gr.Examples(
        [
            ["Describe what's happening in this video.", None, "data/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03.mp4", 80, 2, 1024],
            ["Extract all texts in the image, return only the extracted texts and no other information.", "data/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03frames/LeagueofLegends2025.03.21-10.51.35.03.mp40002.jpeg", None, 80, 2, 1024],
        ],
        fn=inference,
        inputs=[prompt, images_input, video_input, max_frames, fps, max_tokens],
        outputs=[output],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")