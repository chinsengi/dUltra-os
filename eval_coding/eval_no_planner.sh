#!/bin/bash
# Set the environment variables first before running the command.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT/eval/lm_eval"

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

tasks=("mbpp_instruct" "humaneval_instruct")
# tasks=("humaneval_instruct")
length=256
num_fewshot=0
normalize=False
use_scheduler=False
scale=30.0
mode="training"
factor=1
# Set NUM_SAMPLES env var to override; leave empty for full eval
limit=${NUM_SAMPLES:-}

# Optional single-model override; otherwise we sweep MODELS below.
MODEL_PATH="${MODEL_PATH:-}"
read -r -a MODELS <<< "${MODELS:-sengi/dUltra-coding sengi/dUltra-coding-b128}"

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

# Fallback if model-specific mapping is missing; override with env var if needed.
read -r -a BLOCK_LENGTHS <<< "${BLOCK_LENGTHS:-128}"

for task in "${tasks[@]}"; do
  echo "Starting evaluations for task=${task}"

  for model in "${MODELS[@]}"; do
    model_path="$model"

    block_length="${MODEL_BLOCK_LENGTH[$model]:-}"
    if [[ -z "$block_length" ]]; then
      if [[ ${#BLOCK_LENGTHS[@]} -gt 0 ]]; then
        block_length="${BLOCK_LENGTHS[0]}"
      else
        echo "No block length set for model=${model}. Set MODEL_BLOCK_LENGTHS or BLOCK_LENGTHS." >&2
        exit 1
      fi
    fi

    steps=$((length / block_length))
    model_tag="${model//\//_}"
    fastdllm_suffix="${task}_${model_tag}_block${block_length}_factor${factor}"

    echo "Evaluating task=${task}, model=${model}, block_length=${block_length}, factor=${factor}"
    fast_output="eval_results_no_planner/fast_dllm_${fastdllm_suffix}"
    fast_save_dir="${fast_output}/fast_save"
    limit_arg=""
    if [[ -n "${limit}" ]]; then
      limit_arg="--limit ${limit}"
    fi

    # dUltra (no planner) via fast_dllm wrapper
    HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --num_processes=1 -m lm_eval \
      --tasks ${task} \
      --num_fewshot ${num_fewshot} \
      --confirm_run_unsafe_code \
      --model fast_dllm \
      --device cuda \
      --batch_size 1 \
      --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True,save_dir=${fast_save_dir} \
      --output_path "${fast_output}" \
      ${limit_arg}

    # dparallel baseline
    HF_ALLOW_CODE_EVAL=1 accelerate launch --num_processes=1 -m lm_eval \
      --tasks ${task} \
      --num_fewshot 3 \
      --confirm_run_unsafe_code \
      --model fast_dllm \
      --device cuda \
      --batch_size 1 \
      --model_args model_path="Zigeng/dParallel-LLaDA-8B-instruct",gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True,save_dir=${dparallel_save_dir} \
      --output_path "${dparallel_output}" \
      ${limit_arg}

    # d3llm baseline
    HF_ALLOW_CODE_EVAL=1 accelerate launch --num_processes=1 -m lm_eval \
      --tasks ${task} \
      --num_fewshot 3 \
      --confirm_run_unsafe_code \
      --model fast_dllm \
      --device cuda \
      --batch_size 1 \
      --model_args model_path="d3LLM/d3LLM_LLaDA",gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True,save_dir=${d3llm_save_dir} \
      --output_path "${d3llm_output}" \
      ${limit_arg}
  done
done
