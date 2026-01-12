#!/bin/bash
ml load gcc/13
ml load cuda
export LOGDIR=checkpoints
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p $LOGDIR

NUM_ITER=1
# Supported datasets: gsm8k, math500
DATASET="math500"
BETA=0.0 # KL divergence penalty
# MODEL_PATH=sengi/LLaDOU_planner
MODEL_PATH=sengi/lladou_gsm8k
# MODEL_PATH=sengi/lladou-grpo-sft
NUM_PROCESSES=1
PROMPT_MODE="thinking"
NORMALIZE=false
SCALE=30.0
USE_PEFT=false
USE_SCHEDULER=false
max_completion_length=256
block_length=256
learning_rate=5e-6
advantage_min_clip=0.0
RUN_NAME=debug

# CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1
# --max_unmasking_prob 0.9 \
# --config_file accelerate.yaml \
 accelerate launch \
    --num_processes $NUM_PROCESSES \
    --main_process_port 12346 diffu_grpo_train.py \
    --config sbatch_scripts/train.yaml \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir ./checkpoints/$RUN_NAME \
    --num_iterations $NUM_ITER \
    --beta $BETA \
    --normalize $NORMALIZE \
    --scale $SCALE \
    --use_scheduler $USE_SCHEDULER \
    --max_completion_length $max_completion_length \
    --block_length $block_length \
    --learning_rate $learning_rate \
    --advantage_min_clip $advantage_min_clip \
    --num_generations 12 \
    --per_device_train_batch_size 12 \
    --prompt_mode "$PROMPT_MODE" \
    --report_to none
