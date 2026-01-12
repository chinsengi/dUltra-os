#!/bin/bash
#SBATCH --job-name=dultra-coding-math-seed-sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/dultra_coding_math_seed_sweep-%j.out

set -euo pipefail

# Seed sweep for dUltra-coding models on GSM8K + Math500 using the lladou inference path.
# Factor is fixed at 1.0 to evaluate robustness across different random seeds.
#
# Usage (interactive):
#   bash eval/run_dultra_coding_math_seed_sweep.sh
#
# Common overrides:
#   MODEL_PATH=/path/or/hf/repo MODE=inference SEEDS="42 123" bash eval/run_dultra_coding_math_seed_sweep.sh


GPU_LIST="${GPU_LIST:-0}"
MASTER_PORT="${MASTER_PORT:-12345}"
NUM_GPUS="${NUM_GPUS:-1}"

DATASET="${DATASET:-all}"            # Use "all" for gsm8k + math500; "math" corresponds to Math500 in eval/eval.py
GEN_LENGTH="${GEN_LENGTH:-256}"
FEW_SHOT="${FEW_SHOT:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SCALE="${SCALE:-30.0}"
NORMALIZE="${NORMALIZE:-false}"
USE_SCHEDULER="${USE_SCHEDULER:-false}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-}"
MODE="${MODE:-training}"            # factor only affects lladou in mode=inference
INFERENCE_TYPE="${INFERENCE_TYPE:-lladou}"
FACTOR="${FACTOR:-1.0}"              # Fixed factor for this sweep

MODEL_PATH="${MODEL_PATH:-}" # optional single-model override
read -r -a MODELS <<< "${MODELS:-sengi/dUltra-coding sengi/dUltra-coding-b128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-coding_math_seed_sweep_results}"

read -r -a SEEDS <<< "${SEEDS:-42 123 456 789 1000}"
read -r -a BLOCK_LENGTHS <<< "${BLOCK_LENGTHS:-256}" # fallback if model-specific block lengths not set

# Model -> block lengths mapping (defaults match model naming conventions).
# Override/extend via env var, e.g.:
#   MODEL_BLOCK_LENGTHS="sengi/dUltra-coding=32 sengi/dUltra-coding-b128=128"
declare -A MODEL_BLOCK_LENGTH=(
  ["sengi/dUltra-coding"]=32
  ["sengi/dUltra-coding-b128"]=128
)
if [[ -n "${MODEL_BLOCK_LENGTHS:-}" ]]; then
  for kv in ${MODEL_BLOCK_LENGTHS}; do
    k="${kv%%=*}"
    v="${kv#*=}"
    if [[ -n "$k" && -n "$v" ]]; then
      MODEL_BLOCK_LENGTH["$k"]="$v"
    fi
  done
fi

normalize_flag=()
if [[ "$NORMALIZE" == "true" ]]; then
  normalize_flag+=(--normalize)
fi

scheduler_flag=()
if [[ "$USE_SCHEDULER" == "true" ]]; then
  scheduler_flag+=(--use_scheduler)
fi

planner_temp_flag=()
if [[ -n "$PLANNER_TEMPERATURE" ]]; then
  planner_temp_flag=(--planner_temperature "$PLANNER_TEMPERATURE")
fi

DATASETS=()
if [[ "$DATASET" == "all" ]]; then
  DATASETS=(gsm8k math)
elif [[ "$DATASET" == *","* ]]; then
  IFS=',' read -r -a DATASETS <<< "$DATASET"
else
  DATASETS=("$DATASET")
fi

for eval_dataset in "${DATASETS[@]}"; do
  case "$eval_dataset" in
    math|math500)
      OUTPUT_PREFIX="math500"
      EVAL_DATASET="math"
      ;;
    gsm8k)
      OUTPUT_PREFIX="gsm8k"
      EVAL_DATASET="gsm8k"
      ;;
    *)
      echo "Unsupported dataset '$eval_dataset' (use gsm8k, math, math500, or all)." >&2
      exit 1
      ;;
  esac

  for model in "${MODELS[@]}"; do
    if [[ -n "$MODEL_PATH" ]]; then
      model="$MODEL_PATH"
    fi
    block_length="${MODEL_BLOCK_LENGTH[$model]:-}"
    block_lengths_to_run=()
    if [[ -n "$block_length" ]]; then
      block_lengths_to_run=("$block_length")
    else
      block_lengths_to_run=("${BLOCK_LENGTHS[@]}")
    fi
    model_tag="${model//\//_}"

    for block_length in "${block_lengths_to_run[@]}"; do
      suffix="${model_tag}_block${block_length}_mode_${MODE}_factor${FACTOR}"
      output_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}_${suffix}"

      for seed in "${SEEDS[@]}"; do
        echo "Running eval: dataset=${EVAL_DATASET} model=${model} block_length=${block_length} factor=${FACTOR} seed=${seed} output_dir=${output_dir}"

        CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun \
          --nproc_per_node "$NUM_GPUS" \
          --master_port "$MASTER_PORT" \
          eval.py \
          --dataset "$EVAL_DATASET" \
          --gen_length "$GEN_LENGTH" \
          --block_length "$block_length" \
          --batch_size 1 \
          --few_shot "$FEW_SHOT" \
          --temperature "$TEMPERATURE" \
          --scale "$SCALE" \
          --model_path "$model" \
          --output_dir "$output_dir" \
          --suffix "$suffix" \
          --mode "$MODE" \
          --inference_type "$INFERENCE_TYPE" \
          --factor "$FACTOR" \
          --seed "$seed" \
          "${normalize_flag[@]}" \
          "${scheduler_flag[@]}" \
          "${planner_temp_flag[@]}"
      done
    done
  done
done

echo "Done."
