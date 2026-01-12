# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

tasks=("mbpp_instruct" "humaneval_instruct")
tasks=("humaneval_instruct")
length=256
num_fewshot=0
normalize=False
use_scheduler=False
scale=30.0
mode="training"
factor=1.0
# Set NUM_SAMPLES env var to override; leave empty for full eval
limit=${NUM_SAMPLES:-}

block_lengths=(16 32 128 256)
block_lengths=(128)


for task in "${tasks[@]}"; do
  echo "Starting evaluations for task=${task}"

  for block_length in "${block_lengths[@]}"; do
    model_path="${CHECKPOINT_PATHS[$block_length]}"
    model_path="sengi/dUltra-coding-b128"

    if [[ -z "${model_path}" ]]; then
      echo "No checkpoint configured for block_length=${block_length}, skipping..."
      continue
    fi

    steps=$((length / block_length))
    model_dir="$(basename "$(dirname "${model_path}")")"
    model_base="$(basename "${model_path}")"
    model_name="${model_dir}_${model_base}"
    grpo_suffix="${task}_${model_name}_block${block_length}_mode_${mode}_factor${factor}"
    fastdllm_suffix="${task}_block${block_length}"

    echo "Evaluating task=${task}, block_length=${block_length} with checkpoint=${model_path}, factor=${factor}, mode=${mode}"
    grpo_output="final_eval_results/${grpo_suffix}_grpo_lladou"
    fast_output="final_eval_results/fast_dllm_${fastdllm_suffix}_baseline"
    dparallel_output="final_eval_results/dparallel_${fastdllm_suffix}_baseline"
    d3llm_output="final_eval_results/d3llm_${fastdllm_suffix}_baseline"
    grpo_save_dir="${grpo_output}/grpo_save"
    fast_save_dir="${fast_output}/fast_save"
    dparallel_save_dir="${dparallel_output}/dparallel_save"
    d3llm_save_dir="${d3llm_output}/d3llm_save"
    limit_arg=""
    if [[ -n "${limit}" ]]; then
      limit_arg="--limit ${limit}"
    fi

    # dUltra
    HF_ALLOW_CODE_EVAL=1 accelerate launch \
      --num_processes=1 \
      --main_process_port 36234 \
      -m lm_eval \
      --tasks ${task} \
      --num_fewshot ${num_fewshot} \
      --confirm_run_unsafe_code \
      --model grpo_lladou \
      --device cuda \
      --batch_size 1 \
      --model_args model_path=${model_path},gen_length=${length},show_speed=True,normalize=${normalize},scale=${scale},use_scheduler=${use_scheduler},block_length=${block_length},save_dir=${grpo_save_dir},mode=${mode},factor=${factor} \
      --output_path "${grpo_output}" \
      ${limit_arg} \

    # fast dllm parallel factor
    HF_ALLOW_CODE_EVAL=1 accelerate launch --num_processes=1 -m lm_eval \
      --tasks ${task} \
      --num_fewshot 3 \
      --confirm_run_unsafe_code \
      --model fast_dllm \
      --device cuda \
      --batch_size 1 \
      --model_args model_path="GSAI-ML/LLaDA-8B-Instruct",gen_length=${length},steps=${steps},block_length=${block_length},factor=${factor},show_speed=True,save_dir=${fast_save_dir} \
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
