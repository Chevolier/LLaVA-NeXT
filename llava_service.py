from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
import tempfile
import os
import shutil
from typing import Optional, List
import numpy as np
from PIL import Image
import copy
from decord import VideoReader, cpu

# Import LLaVA specific modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

app = FastAPI()

# Configuration
MODEL_PATH = "/home/ec2-user/SageMaker/efs/Models/LLaVA-Video-7B-Qwen2"
MODEL_NAME = "LLaVA-Video-7B-Qwen2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto" if DEVICE == "cuda" else None

# Global variables for model
tokenizer, model, image_processor, max_length = None, None, None, None

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, image_processor, max_length
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        MODEL_PATH, None, MODEL_NAME, 
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

def process_image(image_path):
    """Process image for LLaVA model."""
    img = Image.open(image_path).convert('RGB')
    processed = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(DEVICE)
    if DEVICE == "cuda":
        processed = processed.bfloat16()
    return [processed]

def process_video(video_path, max_frames=80, fps=2):
    """Process video for LLaVA model."""
    # Load video frames
    video_frames, frame_time, video_time = load_video(
        video_path, max_frames_num=max_frames, fps=fps, force_sample=True
    )
    
    processed_video = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(DEVICE)
    if DEVICE == "cuda":
        processed_video = processed_video.bfloat16()
        
    return [processed_video], frame_time, video_time

class InferenceRequest(BaseModel):
    prompt: str
    max_frames: int = 32
    fps: float = 1.0
    max_new_tokens: int = 1024

@app.post("/inference/image")
async def inference_image(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    image: UploadFile = File(...),
    max_new_tokens: int = Form(1024)
):
    # Save uploaded image to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, image.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    # Process the image
    image_data = process_image(temp_path)
    
    # Set up conversation
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    
    # Add image token to the prompt
    question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)
    
    # Run model inference
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_data,
            modalities=["image"],
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
    
    # Clean up temp files
    background_tasks.add_task(lambda: shutil.rmtree(temp_dir))
    
    return {"response": response}

@app.post("/inference/video")
async def inference_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    video: UploadFile = File(...),
    max_frames: int = Form(32),
    fps: float = Form(1.0),
    max_new_tokens: int = Form(1024)
):
    # Save uploaded video to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Process the video
    video_data, frame_time, video_time = process_video(temp_path, max_frames, fps)
    
    # Set up conversation
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    
    # Add video information to the prompt
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video_data[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}\n{prompt}"
    
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # Tokenize input
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)
    
    # Run model inference
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=video_data,
            modalities=["video"],
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
    
    # Clean up temp files
    background_tasks.add_task(lambda: shutil.rmtree(temp_dir))
    
    return {"response": response}

@app.post("/inference/text")
async def inference_text(request: InferenceRequest):
    # Set up conversation
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    
    # Add the prompt
    conv.append_message(conv.roles[0], request.prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    # Tokenize input
    input_ids = tokenizer(prompt_question, return_tensors="pt").input_ids.to(DEVICE)
    
    # Run model inference
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            do_sample=False,
            temperature=0,
            max_new_tokens=request.max_new_tokens,
        )
    
    # Decode output
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract just the assistant's response
    response_prefix = conv.roles[1] + ": "
    if response_prefix in response:
        response = response.split(response_prefix)[-1].strip()
    
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)