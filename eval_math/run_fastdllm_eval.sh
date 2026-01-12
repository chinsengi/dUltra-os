#!/bin/bash
#SBATCH --job-name=fastdllm-eval
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/run_fastdllm_eval-%j.out

set -euo pipefail

# Default GPU configuration (can be overridden by passing GPU IDs as arguments)
GPU_IDS=(0)
MASTER_PORT=29612
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Evaluation configuration
DATASETS=("math500" "gsm8k") # Accepts "gsm8k" or "math500"
GEN_LENGTH=256
FEW_SHOT=0
TEMPERATURE=0.0
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
MODE="inference"
REMASKING="low_confidence"
THRESHOLD=""

FACTORS=("0.5" "0.8" "1.0" "1.2" "1.5")
BASELINES=(fastdllm dparallel d3llm)
# BASELINES=(dUltra-coding dUltra-coding-b128)
declare -A BASELINE_MODELS=(
  [dUltra-math]="sengi/dUltra-math"
  [dUltra-math-b128]="sengi/dUltra-math-b128"
  [dUltra-coding]="sengi/dUltra-coding"
  [dUltra-coding-b128]="sengi/dUltra-coding-b128"
  [fastdllm]="$MODEL_PATH"
  [dparallel]="Zigeng/dParallel-LLaDA-8B-instruct"
  [d3llm]="d3LLM/d3LLM_LLaDA"
)

# Baseline-specific block length selection.
# Override via env var, e.g.:
#   BASELINE_BLOCK_LENGTHS_OVERRIDES="dUltra-coding=32 dUltra-coding-b128=128 fastdllm=16,32,128,256"
declare -A BASELINE_BLOCK_LENGTHS=(
  [fastdllm]="128"
  [dparallel]="128"
  [d3llm]="128"
  [dUltra-math]="32"
  [dUltra-math-b128]="128"
  [dUltra-coding]="32"
  [dUltra-coding-b128]="128"
)
if [[ -n "${BASELINE_BLOCK_LENGTHS_OVERRIDES:-}" ]]; then
  for kv in ${BASELINE_BLOCK_LENGTHS_OVERRIDES}; do
    k="${kv%%=*}"
    v="${kv#*=}"
    v="${v//,/ }"
    if [[ -n "$k" && -n "$v" ]]; then
      BASELINE_BLOCK_LENGTHS["$k"]="$v"
    fi
  done
fi

# Allow overriding GPU list from arguments
if [ $# -gt 0 ]; then
  GPU_IDS=("$@")
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

threshold_flag=()
if [ -n "$THRESHOLD" ]; then
  threshold_flag=(--threshold "$THRESHOLD")
fi

for FACTOR in "${FACTORS[@]}"; do
  echo "Starting evaluations with factor=$FACTOR"
  for DATASET in "${DATASETS[@]}"; do
    case "$DATASET" in
      math|math500)
        OUTPUT_PREFIX="math500"
        EVAL_DATASET="math"
        ;;
      gsm8k)
        OUTPUT_PREFIX="gsm8k"
        EVAL_DATASET="gsm8k"
        ;;
      *)
        echo "Unsupported dataset '$DATASET'. Update run_fastdllm_eval.sh with the correct settings." >&2
        exit 1
        ;;
    esac

    echo "Starting dataset=$EVAL_DATASET"

    for baseline in "${BASELINES[@]}"; do
      model_path="${BASELINE_MODELS[$baseline]}"
      if [[ -z "${model_path}" ]]; then
        echo "No model configured for baseline '${baseline}'" >&2
        exit 1
      fi

      block_lengths_str="${BASELINE_BLOCK_LENGTHS[$baseline]:-}"
      if [[ -z "$block_lengths_str" ]]; then
        echo "No block lengths configured for baseline '${baseline}'" >&2
        exit 1
      fi
      read -r -a BLOCK_LENGTHS <<< "$block_lengths_str"

      for block_length in "${BLOCK_LENGTHS[@]}"; do
        if [ "$block_length" -le 0 ]; then
          echo "Invalid block_length=${block_length}, skipping..."
          continue
        fi

        batch_size=1 # the nfe calculation requires batch_size=1

        steps=$((GEN_LENGTH / block_length))
        if [ "$steps" -le 0 ]; then
          steps=1
        fi

        base_model_name=$(basename "$model_path")
        suffix="${base_model_name}_block${block_length}_fastdllm"
        output_dir="fastdllm_eval_results/${OUTPUT_PREFIX}_${suffix}"

        echo "Running eval.py (${baseline}) on dataset=${EVAL_DATASET}, block_length=${block_length}, steps=${steps}"

        CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
          --nproc_per_node $NUM_GPUS \
          --master_port $MASTER_PORT \
          eval.py \
          --dataset "$EVAL_DATASET" \
          --gen_length $GEN_LENGTH \
          --block_length $block_length \
          --batch_size $batch_size \
          --few_shot $FEW_SHOT \
          --temperature $TEMPERATURE \
          --model_path "$model_path" \
          --output_dir "$output_dir" \
          --suffix "$suffix" \
          --mode "$MODE" \
          --inference_type fastdllm \
          --steps $steps \
          --remasking "$REMASKING" \
          --factor $FACTOR \
          "${threshold_flag[@]}"
      done
    done
  done
done

echo "All fast-dllm baseline evaluations completed!"
