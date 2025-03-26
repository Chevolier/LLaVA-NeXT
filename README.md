# Environment Configuration
```bash
conda create -n llava python=3.10 -y
conda activate llava

# For local training
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"

# For SGLang
pip install uv
uv pip install "sglang[all]>=0.4.4.post1" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
pip install transformers==4.48.3

pip install -U "huggingface_hub[cli]"

# For SageMaker
pip install -U sagemaker

pip install rouge
pip install nltk
```

# Data Preparation

Put your data in the data/ folder in the following structure, video is the folder of all videos

```
data
└── hualai_sft_data
    ├── output_0314.json
    └── video
```

and then follow steps in prepare_data.ipynb.


# Download Model

Download the pretrained version to your local disk, this could be used for training locally.
```bash
mkdir -p models
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --local-dir models/LLaVA-Video-7B-Qwen2
```

# Training

We have two options, training in EC2/notebook locally or using SageMaker training-jobs.

## Training in EC2 or notebook
Suggested instance: g6e.48xlarge

```bash
# Change the following parameters to your directory in SO400M_Qwen2_7B.sh
# PREV_STAGE_CHECKPOINT
# --image_folder  since we only have video, this can be set the same with --video_folder
# --video_folder 
# --output_dir
# Then run
bash scripts/video/train/SO400M_Qwen2_7B.sh
```

## Training in training-jobs
Suggested instance: ml.g6e.48xlarge, ml.p4d.24xlarge
1. Build image: 
```bash
cd training-jobs
bash build_and_push.sh
```
2. Start training job:
Follow steps in training-jobs/llava-video-full-finetuning-sagemaker.ipynb for full finetuning or training-jobs/llava-video-lora-finetuning-sagemaker.ipynb for lora finetuning, need to change image to the image you build in Step 1.

### Checkpointing
After saving model checkpoint, add a vocab_size in the config.json
```
  "vocab_size": 152064
```

# Evaluation

To evaluate the finetuned model, following steps in evaluate.ipynb. Suggested instance: g6e.xlarge.


# SGLang deployment
Need to further copy config.json and preprocess_config.json from downloaded pretrained folder to the checkpoint folder.

Run sglang 
```bash
python3 -m sglang.launch_server --model-path your_path_to_checkpoint/checkpoint-100 \
    --chat-template=chatml-llava --chunked-prefill-size 2048 --max-running-requests 50 \
    --mem-fraction-static 0.85 --port 30008 
```
To test the started service, run the part in evaluate.ipynb.

