#!/bin/bash

# Define your hyperparameters
# block_sizes=(16 32 128 256)
block_sizes=(128)
adv_clips=()
adv_clips=(-0.5 0.0 0.1 0.5)
# datasets=("gsm8k" "apps")
datasets=("apps")
fixed_adv_clip=0.0
fixed_block_size=256
port=12446
model_path="sengi/lladou-coding"
prompt_mode="non-thinking"
num_processes=1 # # of GPUs
num_generations=12 # increase this for multiGPU
per_device_train_batch_size=12 # increase this for B200
temperature=0.5

# Sweep over block sizes (advantage clip fixed)
for ds in "${datasets[@]}"; do
    for bs in "${block_sizes[@]}"; do
        sbatch --gres=gpu:$num_processes sbatch_scripts/grpo_exp.sh \
            "$ds" "$bs" "$fixed_adv_clip" "$port" "$model_path" \
            "$num_generations" "$per_device_train_batch_size" \
            "$num_processes" "$temperature" "$prompt_mode"
        port=$((port + 1))
    done
done

# Sweep over advantage clips (block length fixed)
for ds in "${datasets[@]}"; do
    for ac in "${adv_clips[@]}"; do
        if [[ "$ac" == "$fixed_adv_clip" ]]; then
            continue
        fi
        sbatch --gres=gpu:$num_processes sbatch_scripts/grpo_exp.sh \
            "$ds" "$fixed_block_size" "$ac" "$port" "$model_path" \
            "$num_generations" "$per_device_train_batch_size" \
            "$num_processes" "$temperature" "$prompt_mode"
        port=$((port + 1))
    done
done
