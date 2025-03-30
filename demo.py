import os
import gradio as gr
import requests
import tempfile
import shutil
from typing import Union
import boto3
import cv2
import subprocess
import uuid
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import base64

# Services URLs
LLAVA_SERVICE_URL = "http://localhost:8001"
QWEN_SERVICE_URL = "http://localhost:8002"

# S3 configuration
S3_BUCKET = "sagemaker-us-west-2-452145973879"  # Update with your bucket name
S3_OUTPUT_PREFIX = "projects/game-highlights/highlight-clips/"

s3_client = boto3.client('s3')

def download_from_s3(s3_path):
    """Download a file from S3 to a local temp file."""
    temp_dir = tempfile.mkdtemp()
    local_path = os.path.join(temp_dir, os.path.basename(s3_path))
    
    # Parse bucket name and key from s3 path
    if s3_path.startswith('s3://'):
        parts = s3_path[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        raise ValueError("Invalid S3 path format. Should be s3://bucket-name/path/to/file")
    
    # Download the file
    s3_client.download_file(bucket, key, local_path)
    
    return local_path, temp_dir

def upload_to_s3(local_path):
    """Upload a file to S3 and return the S3 path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"highlight_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
    s3_key = f"{S3_OUTPUT_PREFIX}{filename}"
    
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    
    return f"s3://{S3_BUCKET}/{s3_key}", f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

def extract_frames(video_path, fps):
    """Extract frames from a video at the specified fps."""
    temp_dir = tempfile.mkdtemp()
    frames_info = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], temp_dir
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / video_fps
    
    # Calculate frame interval based on target fps
    frame_interval = max(1, round(video_fps / fps))
    
    # Extract frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only save frames at the desired interval
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_info.append({
                "path": frame_path,
                "timestamp": timestamp,
                "frame_idx": frame_idx
            })
        
        frame_idx += 1
    
    cap.release()
    return frames_info, temp_dir

def extract_texts_from_frame(frame_path, model_name):
    """Extract text from an image using the selected model."""
    service_url = LLAVA_SERVICE_URL if model_name == "LLaVA-Video-7B-Qwen2" else QWEN_SERVICE_URL
    
    with open(frame_path, "rb") as image_file:
        files = {"image": (os.path.basename(frame_path), image_file, "image/jpeg")}
        data = {
            "prompt": "Extract all texts in this image. Only return the extracted texts, nothing else.",
            "max_new_tokens": 512
        }
        response = requests.post(
            f"{service_url}/inference/image",
            files=files,
            data=data
        )
    
    return response.json()["response"]

def process_frame(frame_info, keywords, model_name):
    """Process a single frame to check for keywords."""
    # Extract text from the frame
    texts = extract_texts_from_frame(frame_info["path"], model_name)
    
    # Check if any keyword is in the extracted text
    matched_keywords = []
    for keyword in keywords:
        if keyword.strip() and keyword.strip().lower() in texts.lower():
            matched_keywords.append(keyword.strip())
    
    if matched_keywords:
        return {
            "frame_info": frame_info,
            "matched_keywords": matched_keywords,
            "texts": texts
        }
    
    return None

def create_highlight_clip(video_path, highlights, buffer_seconds, output_path):
    """Create a highlight video from the specified segments."""
    if not highlights:
        return False
    
    # Create a file to store the list of segments for ffmpeg
    segments_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    segments_path = segments_file.name
    
    # Group consecutive highlights (within buffer_seconds of each other)
    highlight_segments = []
    current_segment = None
    
    for highlight in sorted(highlights, key=lambda x: x["frame_info"]["timestamp"]):
        timestamp = highlight["frame_info"]["timestamp"]
        
        if current_segment is None:
            current_segment = {
                "start": max(0, timestamp - buffer_seconds),
                "end": timestamp + buffer_seconds,
                "keywords": highlight["matched_keywords"]
            }
        elif timestamp - buffer_seconds <= current_segment["end"]:
            # Extend the current segment
            current_segment["end"] = timestamp + buffer_seconds
            current_segment["keywords"].extend(highlight["matched_keywords"])
            current_segment["keywords"] = list(set(current_segment["keywords"]))  # Remove duplicates
        else:
            # Start a new segment
            highlight_segments.append(current_segment)
            current_segment = {
                "start": max(0, timestamp - buffer_seconds),
                "end": timestamp + buffer_seconds,
                "keywords": highlight["matched_keywords"]
            }
    
    # Add the last segment if it exists
    if current_segment:
        highlight_segments.append(current_segment)
    
    # Write segments to the file in ffmpeg format
    for i, segment in enumerate(highlight_segments):
        start = segment["start"]
        duration = segment["end"] - segment["start"]
        
        # Format the timestamps as string with fixed decimal precision
        start_str = f"{start:.2f}"  # Format with 2 decimal places
        
        segments_file.write(f"file '{video_path}'\n")
        segments_file.write(f"inpoint {start_str}\n")
        segments_file.write(f"outpoint {segment['end']:.2f}\n")
    
    segments_file.close()
    
    # For debugging: print the contents of the segments file
    with open(segments_path, 'r') as f:
        print("Segments file content:")
        print(f.read())
    
    # Use ffmpeg to create the highlight video
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", segments_path,
        "-c", "copy", output_path
    ]
    
    try:
        process = subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print("FFmpeg output:", process.stdout.decode())
        os.unlink(segments_path)  # Remove temporary file
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating highlight clips: {e.stderr}")
        print(f"Command was: {' '.join(ffmpeg_cmd)}")
        os.unlink(segments_path)  # Remove temporary file
        return False

def extract_highlights(s3_video_path, highlight_keywords, extraction_fps, buffer_seconds, model_name, create_clip=True):
    """Extract highlights from a video based on keywords."""
    progress_updates = []
    
    # Parse keywords (split by comma instead of newlines)
    keywords_list = [k.strip() for k in re.split(r'[,，]', highlight_keywords) if k.strip()]
    
    if not keywords_list:
        return "No valid keywords provided.", None, progress_updates, None
    
    progress_updates.append(f"Looking for keywords: {', '.join(keywords_list)}")
    
    # Download video from S3
    progress_updates.append("Downloading video from S3...")
    try:
        local_video_path, temp_video_dir = download_from_s3(s3_video_path)
    except Exception as e:
        return f"Error downloading video: {str(e)}", None, progress_updates, None

    # Extract frames
    progress_updates.append(f"Extracting frames at {extraction_fps} fps...")
    frames_info, temp_frames_dir = extract_frames(local_video_path, extraction_fps)
    
    if not frames_info:
        shutil.rmtree(temp_video_dir)
        shutil.rmtree(temp_frames_dir)
        return "No frames could be extracted from the video.", None, progress_updates, None

    # Process frames to find keywords
    progress_updates.append(f"Processing {len(frames_info)} frames to find keywords...")
    
    highlights = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_frame = {
            executor.submit(process_frame, frame, keywords_list, model_name): frame 
            for frame in frames_info
        }
        
        for i, future in enumerate(future_to_frame.keys()):
            if i % 20 == 0:  # Update progress every 20 frames
                progress_updates.append(f"Processed {i}/{len(frames_info)} frames...")
                
            result = future.result()
            if result:
                highlights.append(result)
    
    if not highlights:
        shutil.rmtree(temp_video_dir)
        shutil.rmtree(temp_frames_dir)
        return "No highlights found matching the keywords.", None, progress_updates, None
    
    # Create highlight video if requested
    s3_path = None
    
    if create_clip:
        progress_updates.append(f"Found {len(highlights)} frames with matching keywords.")
        progress_updates.append("Creating highlight video...")
        
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "highlights.mp4")
        
        success = create_highlight_clip(local_video_path, highlights, buffer_seconds, output_path)
        
        if not success:
            progress_updates.append("Failed to create highlight video. Showing detected frames only.")
        else:
            # Upload to S3
            progress_updates.append("Uploading highlight video to S3...")
            s3_path, download_url = upload_to_s3(output_path)
            progress_updates.append(f"Video uploaded to {s3_path}")
            
            # Clean up output directory
            shutil.rmtree(output_dir)
    else:
        progress_updates.append(f"Found {len(highlights)} frames with matching keywords. Skipping clip creation.")
    
    # Prepare HTML for highlight frames with text
    html_parts = ["<style>",
             ".highlight-container { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 8px; background-color: #f9f9f9; }",
             ".highlight-image-container { text-align: center; margin-bottom: 20px; }",
             ".highlight-image { width: auto; max-width: 90%; max-height: 500px; object-fit: contain; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }",
             ".highlight-info { background-color: white; padding: 20px; border-radius: 8px; border: 1px solid #eee; }",
             ".highlight-text { max-height: 200px; overflow-y: auto; background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 15px; }",
             ".highlight-metadata { display: flex; justify-content: space-between; margin-bottom: 15px; }",
             ".timestamp { font-size: 1.2em; font-weight: bold; color: #444; }",
             ".keywords { color: #d32f2f; font-weight: bold; }",
             "</style>",
             "<div style='max-width: 1200px; margin: 0 auto;'>",
             f"<h2>Detected Highlights ({min(5, len(highlights))} of {len(highlights)})</h2>"]
    
    # Create temp directory for hosting highlight frames
    highlight_img_dir = tempfile.mkdtemp()
    
    # Create base64 encodings of images to embed directly in HTML
    for i, highlight in enumerate(highlights[:5]):  # Limit to first 5
        # Read the frame image
        frame_path = highlight["frame_info"]["path"]
        frame_img = cv2.imread(frame_path)
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        
        # Save the image to the temp directory with a unique name
        temp_img_path = os.path.join(highlight_img_dir, f"highlight_{i}.jpg")
        cv2.imwrite(temp_img_path, frame_img)
        
        # Convert image to base64 for embedding directly in HTML
        with open(temp_img_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}"
        
        # Format timestamp
        timestamp = highlight["frame_info"]["timestamp"]
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # Add this highlight to the HTML
        html_parts.append(f"""
        <div class="highlight-container">
            <div class="highlight-image-container">
                <img src="{img_base64}" class="highlight-image" alt="Highlight at {time_str}">
            </div>
            <div class="highlight-info">
                <div class="highlight-metadata">
                    <div class="timestamp">Timestamp: {time_str}</div>
                    <div class="keywords">Keywords: {', '.join(highlight['matched_keywords'])}</div>
                </div>
                <div class="highlight-text">
                    <p><strong>Extracted Text:</strong><br>{highlight['texts']}</p>
                </div>
            </div>
        </div>
        """)
    
    html_parts.append("</div>")
    highlights_html = "".join(html_parts)
    
    # Clean up temp files
    shutil.rmtree(temp_video_dir)
    shutil.rmtree(temp_frames_dir)
    # Note: We don't delete highlight_img_dir as we may need it for other things
    
    result_message = f"Found {len(highlights)} frames with matching keywords."
    if create_clip and s3_path:
        result_message += " Highlight video created and uploaded to S3."
    elif create_clip:
        result_message += " Failed to create highlight video."
    else:
        result_message += " Clip creation skipped as requested."
    
    return result_message, s3_path, progress_updates, highlights_html

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
    
    with gr.Tab("Model Inference"):
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

    with gr.Tab("Video Game Highlights Extraction"):
        # Input section
        gr.Markdown("### 1. Configure Highlight Extraction")
        with gr.Row():
            with gr.Column(scale=1):
                highlight_model = gr.Dropdown(
                    choices=["LLaVA-Video-7B-Qwen2", "Qwen2.5-VL-7B-Instruct"],
                    value="Qwen2.5-VL-7B-Instruct",
                    label="Select Model for Text Extraction"
                )
                extraction_fps = gr.Slider(
                    label="Frame Extraction FPS", 
                    minimum=0.5, maximum=10, value=2, step=0.5
                )
                buffer_seconds = gr.Slider(
                    label="Highlight Buffer (seconds before/after)", 
                    minimum=1, maximum=30, value=5, step=1
                )
                create_clip = gr.Checkbox(
                    label="Create Highlight Clip", 
                    value=True,
                    info="Uncheck to only detect highlights without creating a video"
                )
            
            with gr.Column(scale=2):
                s3_video_path = gr.Textbox(
                    label="S3 Video Path (s3://bucket-name/path/to/video.mp4)",
                    placeholder="s3://your-bucket/video/game_replay.mp4"
                )
                highlight_keywords = gr.Textbox(
                    label="Highlight Keywords (comma separated)",
                    placeholder="Triple Kill, Quadra Kill, Penta Kill, Ace, First Blood",
                    lines=3,
                    info="Enter keywords separated by commas"
                )
                highlight_btn = gr.Button("Extract Highlights", size="lg", variant="primary")
        
        # Results summary section
        gr.Markdown("### 2. Results Summary")
        with gr.Row():
            highlight_result = gr.Textbox(label="Status", lines=2)
            s3_output_path = gr.Textbox(label="Highlight Video S3 Path", lines=1)
        
        # Progress logs
        gr.Markdown("### 3. Processing Logs")
        highlight_progress = gr.JSON(show_label=False)
        
        # Full-width highlight content
        gr.Markdown("### 4. Detected Highlights")
        highlight_content = gr.HTML()
        
        # Connect the highlight extraction button
        def cleanup_temp_dir(result, s3_path, progress, content, temp_dir):
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return result, s3_path, progress, content
        
        highlight_btn.click(
            fn=extract_highlights,
            inputs=[s3_video_path, highlight_keywords, extraction_fps, buffer_seconds, highlight_model, create_clip],
            outputs=[highlight_result, s3_output_path, highlight_progress, highlight_content]
        ).then(
            fn=cleanup_temp_dir,
            inputs=[highlight_result, s3_output_path, highlight_progress, highlight_content, gr.State(value="_")],
            outputs=[highlight_result, s3_output_path, highlight_progress, highlight_content]
        )
        
        # Example for highlight extraction
        gr.Examples(
            [
                ["s3://sagemaker-us-west-2-452145973879/datasets/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03.mp4", 
                "击杀,双杀,三杀,Ace,First Blood", 
                0.5, 2, "Qwen2.5-VL-7B-Instruct", True],
                ["s3://sagemaker-us-west-2-452145973879/datasets/绿树电竞/英雄联盟/LeagueofLegends2025.03.21-10.51.35.03.mp4", 
                "击杀,双杀,三杀,Ace,First Blood",
                0.5, 2, "Qwen2.5-VL-7B-Instruct", False],
            ],
            fn=extract_highlights,
            inputs=[s3_video_path, highlight_keywords, extraction_fps, buffer_seconds, highlight_model, create_clip],
            outputs=[highlight_result, s3_output_path, highlight_progress, highlight_content]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")