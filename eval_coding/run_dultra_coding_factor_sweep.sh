#!/bin/bash
#SBATCH --job-name=dultra-coding-factor-sweep
#SBATCH --mail-type=END,FAIL
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/dultra_coding_factor_sweep-%j.out

set -euo pipefail

# Factor sweep for dUltra-coding on coding benchmarks via lm-eval-harness.
#
# Usage (from repo root):
#   bash eval/lm_eval/run_dultra_coding_factor_sweep.sh
#
# Common overrides:
#   MODEL_PATH=/path/or/hf/repo BLOCK_LENGTH=256 GEN_LENGTH=256 FACTORS="0.5 0.8 1.2 1.5" SEEDS="42 123" TASKS="humaneval_instruct mbpp_instruct" bash eval/lm_eval/run_dultra_coding_factor_sweep.sh

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

MODEL_PATH="${MODEL_PATH:-}" # optional single-model override
read -r -a MODELS <<< "${MODELS:-sengi/dUltra-coding sengi/dUltra-coding-b128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-factor_eval_results}"

GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-256}" # fallback if model-specific block length not set
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
MODE="${MODE:-training}" # factor only affects lladou in mode=inference

NORMALIZE="${NORMALIZE:-false}"
USE_SCHEDULER="${USE_SCHEDULER:-false}"
SCALE="${SCALE:-30.0}"

LIMIT="${NUM_SAMPLES:-}" # optional subset size

read -r -a FACTORS <<< "${FACTORS:-0.5 0.8 1.0 1.2 1.5}"
read -r -a SEEDS <<< "${SEEDS:-42 123 456 789 1000}"
read -r -a TASKS <<< "${TASKS:-mbpp_instruct humaneval_instruct}"

# Model -> block length mapping (defaults match model naming conventions).
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

limit_arg=()
if [[ -n "$LIMIT" ]]; then
  limit_arg=(--limit "$LIMIT")
fi

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ -n "$MODEL_PATH" ]]; then
      model="$MODEL_PATH"
    fi
    block_length="${MODEL_BLOCK_LENGTH[$model]:-$BLOCK_LENGTH}"
    model_tag="${model//\//_}"
    for factor in "${FACTORS[@]}"; do
      suffix="${model_tag}_${task}_block${block_length}_mode_${MODE}_factor${factor}"
      base_output_dir="${OUTPUT_ROOT}/${suffix}"

      for seed in "${SEEDS[@]}"; do
        output_dir="${base_output_dir}"
        save_dir="${output_dir}/grpo_save"

        echo "Running coding eval: task=${task} model=${model} block_length=${block_length} factor=${factor} seed=${seed} output_dir=${output_dir}"

        PYTHONPATH=. accelerate launch --gpu_ids=1,2 --num_processes=1 -m lm_eval \
          --tasks "${task}" \
          --num_fewshot "${NUM_FEWSHOT}" \
          --confirm_run_unsafe_code \
          --model grpo_lladou \
          --device cuda \
          --batch_size 1 \
          --model_args "model_path=${model},gen_length=${GEN_LENGTH},block_length=${block_length},save_dir=${save_dir},mode=${MODE},factor=${factor},normalize=${NORMALIZE},scale=${SCALE},use_scheduler=${USE_SCHEDULER},seed=${seed}" \
          --output_path "${output_dir}" \
          --log_samples \
          "${limit_arg[@]}"
      done
    done
  done
done

echo "Done."
