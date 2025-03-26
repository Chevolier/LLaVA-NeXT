# Environment Configuration
## 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

## 2. **Install the training package:**
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

# Data Preparation
Follow steps in prepare_data.ipynb.


# Training
## Training in EC2 or notebook
Suggested instance: g6e.48xlarge
```bash
bash scripts/video/train/SO400M_Qwen2_7B.sh
```

## Training in training-jobs
Suggested instance: ml.g63.48xlarge, ml.p4d.24xlarge
1. Build image: 
```bash
cd training-jobs
bash build_and_push.sh
```
2. Start training job:
Follow steps in training-jobs/llava-video-full-finetuning-sagemaker.ipynb, need to change image to the image you build in Step 1.

# Checkpoint
After saving model checkpoint, add a vocab_size in the config.json
```
  "vocab_size": 152064
```

# SGLang deployment
## LLaVA-Video-Qwen2
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.4.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3
```

Need to copy config.json and preprocess_config.json to the trained folder

Run sglang 
```bash

python3 -m sglang.launch_server --model-path /home/ec2-user/SageMaker/efs/Projects/LLaVA-NeXT/checkpoints/llava-video-task-full-2025-03-19-05-52-43/checkpoint-100 \
    --chat-template=chatml-llava --chunked-prefill-size 2048 --max-running-requests 50 --mem-fraction-static 0.85 --port 30008 
```

## Qwen2.5-VL-7B
```bash
# python local-deploy/sglang_server.py

python3 -m sglang.launch_server --model-path /home/ec2-user/SageMaker/efs/Models/Qwen2.5-VL-7B-Instruct \
    --chat-template=qwen2-vl --chunked-prefill-size 1024 --max-running-requests 50 --mem-fraction-static 0.6 --port 30008
```

