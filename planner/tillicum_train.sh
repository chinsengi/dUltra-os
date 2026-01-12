#!/bin/bash

#SBATCH --job-name=planner-pretrain
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sc256@uw.edu
#SBATCH --qos=normal
#SBATCH --mem=240G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

#SBATCH --chdir=.
#SBATCH --output=./slurm_out/slurm-%j.out

ml load gcc/13.4.0
ml load cuda
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL

accelerate launch --num_processes=1 planner_train.py
