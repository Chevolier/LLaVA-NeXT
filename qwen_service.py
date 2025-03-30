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
from decord import VideoReader, cpu

# Import Qwen specific modules
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI()

# Configuration
MODEL_PATH = "/home/ec2-user/SageMaker/efs/Models/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for model
model, processor = None, None

@app.on_event("startup")
async def startup_event():
    global model, processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load video frames from a video file."""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), 0
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps_ratio = round(vr.get_avg_fps()/fps)
    
    # Get frame indices
    frame_idx = [i for i in range(0, len(vr), fps_ratio)]
    
    # Sample frames if needed
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    
    return spare_frames, video_time

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
    
    # Create messages
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://" + temp_path,
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]
    
    # Process for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up temp files
    background_tasks.add_task(lambda: shutil.rmtree(temp_dir))
    
    return {"response": output_text}

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
    
    # Create messages
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file://" + temp_path,
                "max_frames": max_frames,
                "fps": fps,
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]
    
    # Process for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Clean up temp files
    background_tasks.add_task(lambda: shutil.rmtree(temp_dir))
    
    return {"response": output_text}

@app.post("/inference/text")
async def inference_text(request: InferenceRequest):
    # Create messages
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": request.prompt
            }
        ]
    }]
    
    # Process for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return {"response": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)