#!/bin/bash
#SBATCH --job-name=run-eval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/run_eval-%j.out

set -euo pipefail

# Default GPU configuration (can be overridden by passing GPU IDs as arguments)
GPU_IDS=(0)
MASTER_PORT=29411
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Evaluation configuration (dataset "math" corresponds to the Math500 split)
# Set `DATASET=all` to run both GSM8K and Math500 (via `--dataset math`).
# You can also pass a comma-separated list, e.g. `DATASET=math,gsm8k`.
DATASET="${DATASET:-gsm8k}"
GEN_LENGTH=256
FEW_SHOT=0
SCALE=30.0
NORMALIZE=false
USE_SCHEDULER=false
MODE="training" # "training" or "inference"

# Optional: set `MODEL_PATH` to force a single model for all runs (HF repo or local path).
# If unset/empty, the per-block checkpoints in CHECKPOINT_PATHS are used.
MODEL_PATH="${MODEL_PATH:-}"

DATASETS=()
if [[ "$DATASET" == "all" ]]; then
  DATASETS=(math gsm8k)
elif [[ "$DATASET" == *","* ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASET"
else
  DATASETS=("$DATASET")
fi

# Allow overriding GPU list from arguments
if [ $# -gt 0 ]; then
  GPU_IDS=("$@")
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

for DATASET in "${DATASETS[@]}"; do
  unset CHECKPOINT_PATHS
  declare -A CHECKPOINT_PATHS=()

  BLOCK_LENGTHS=(16 32 128 256)
  BLOCK_LENGTHS=(32)

  echo "Starting dataset=$DATASET"

  for block_length in "${BLOCK_LENGTHS[@]}"; do
    model_path="$MODEL_PATH"
    model_path="sengi/dUltra-math"
    if [[ -z "$model_path" ]]; then
      model_path="${CHECKPOINT_PATHS[$block_length]:-}"
    fi

    if [[ -z "${model_path}" ]]; then
      echo "No MODEL_PATH set and no checkpoint configured for block_length=${block_length}, skipping..."
      continue
    fi

    batch_size=1

    base_model_name=$(basename "$model_path")
    suffix="block${block_length}_${MODE}_mode_${base_model_name}"
    output_dir="eval_results/${OUTPUT_PREFIX}_${suffix}"

    echo "Running eval.py on model=${base_model_name}, dataset=${DATASET}, block_length=${block_length}, batch_size=${batch_size}"

    normalize_flag=()
    if [ "$NORMALIZE" = true ]; then
      normalize_flag+=(--normalize)
    fi

    scheduler_flag=()
    if [ "$USE_SCHEDULER" = true ]; then
      scheduler_flag+=(--use_scheduler)
    fi

    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset "$DATASET" \
      --gen_length $GEN_LENGTH \
      --block_length $block_length \
      --batch_size $batch_size \
      --few_shot $FEW_SHOT \
      --scale $SCALE \
      --model_path "$model_path" \
      --output_dir "$output_dir" \
      --suffix "$suffix" \
      --mode "$MODE" \
      --inference_type lladou \
      "${normalize_flag[@]}" \
      "${scheduler_flag[@]}"
  done
done

echo "All evaluations completed!"
