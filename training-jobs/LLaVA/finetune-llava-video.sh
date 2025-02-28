
#!/bin/bash
export WANDB_MODE=offline

WORKING_DIR=/opt/ml/code
SM_WORKING_DIR=/opt/ml/model

#The related information about multi-nodes cluster.
MASTER_HOST=$SM_MASTER
MASTER_ADDR=$SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES="$NODE_NUMBER"
NODE_RANK="$NODE_INDEX"
GPUS_PER_NODE="$SM_NUM_GPUS"

echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "job_id: ${job_id}"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="Qwen2-7B-Instruct"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="siglip-so400m-patch14-384"

PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"

BASE_RUN_NAME="LLaVA-Video-7B-Qwen2"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ov_to_video_am9_aug17"
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

export AV_LOG_LEVEL=error  # Suppress FFmpeg info/warning messages
export PYTHONWARNINGS="ignore::UserWarning"  # Filter Python warnings

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path /opt/ml/input/data/training/0_30_s_activitynetqa_oe_qa_processed.json \
    --image_folder /opt/ml/input/data/training \
    --video_folder /opt/ml/input/data/training \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /opt/ml/checkpoints/${job_id} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 16 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
