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
