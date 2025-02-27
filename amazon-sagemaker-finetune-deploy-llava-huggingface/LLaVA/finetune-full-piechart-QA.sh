
#!/bin/bash
export WANDB_MODE=offline

# cd /opt/ml/code
# pip install -e . --no-deps

# pip list |grep peft

pip install pyav
pip install open_clip_torch
pip install peft==0.10.0
pip install datasets
pip install tyro
pip install transformers==4.40.0
# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
pip install deepspeed==0.14.4
pip install accelerate==0.27.0
pip install wandb

pip list |grep peft

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmms-lab/LLaVA-Video-7B-Qwen2 \
    --version v1 \
    --data_path /opt/ml/input/data/piechart/piechart-QA.jsonl \
    --image_folder /opt/ml/input/data//piechart/piechart-QA \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /opt/ml/checkpoints/$(job_id) \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
