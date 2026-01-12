#!/bin/bash
#SBATCH --job-name=diffu-grpo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out

set -euo pipefail

ml load gcc/13
ml load cuda
export LOGDIR=checkpoints
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOGDIR"

NUM_ITER=1
DATASET=${1:-"gsm8k"}
BLOCK_LENGTH=${2:-32}
ADVANTAGE_MIN_CLIP=${3:-0.1}
PORT=${4:-12345}
MODEL_PATH=${5:-"sengi/dUltra-coding"}
NUM_GENERATIONS=${6:-12}
PER_DEVICE_TRAIN_BATCH_SIZE=${7:-12}
NUM_PROCESSES=${8:-1}
TEMPERATURE=${9:-0.1}
PROMPT_MODE=${10:-"thinking"}

if [[ -z "$DATASET" ]]; then
    echo "Usage: $0 DATASET [BLOCK_LENGTH] [ADVANTAGE_MIN_CLIP] [PORT] [MODEL_PATH] [NUM_GENERATIONS] [PER_DEVICE_TRAIN_BATCH_SIZE] [NUM_PROCESSES] [TEMPERATURE] [PROMPT_MODE]" >&2
    exit 1
fi

BETA=0.0
# MODEL_PATH=sengi/LLaDOU-planner_balanced
NORMALIZE=false
SCALE=30.0
USE_PEFT=false
USE_SCHEDULER=false
MAX_COMPLETION_LENGTH=256
LEARNING_RATE=5e-6
FREEZE_UNMASKING_HEAD=false
ROLLOUT_MODE="training"
scale_reward="none"

RUN_NAME=${DATASET}_lladou_onpolicy_lr${LEARNING_RATE}_block${BLOCK_LENGTH}_advclip${ADVANTAGE_MIN_CLIP}_temp${TEMPERATURE}_freeze${FREEZE_UNMASKING_HEAD}_ngen${NUM_GENERATIONS}_mode_${ROLLOUT_MODE}_${scale_reward}_$(basename "$MODEL_PATH")_${PROMPT_MODE}

accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$PORT" diffu_grpo_train.py \
    --config sbatch_scripts/train.yaml \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --run_name "$RUN_NAME" \
    --output_dir "./checkpoints/$RUN_NAME" \
    --num_iterations "$NUM_ITER" \
    --beta "$BETA" \
    --normalize "$NORMALIZE" \
    --scale "$SCALE" \
    --use_scheduler "$USE_SCHEDULER" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --block_length "$BLOCK_LENGTH" \
    --learning_rate "$LEARNING_RATE" \
    --advantage_min_clip "$ADVANTAGE_MIN_CLIP" \
    --temperature "$TEMPERATURE" \
    --freeze_unmasking_head "$FREEZE_UNMASKING_HEAD" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --rollout_mode $ROLLOUT_MODE \
    --scale_reward $scale_reward \
    --prompt_mode "$PROMPT_MODE" \
    # --report_to none
    # --resume_from_checkpoint ./checkpoints/gsm8k_p_iter1_beta0.0_lladou_onpolicy_peftfalse_lr5e-6_block32_len256/checkpoint-1000 \
