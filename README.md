# dUltra: Ultra-Fast Diffusion Large Language Models via Reinforcement Learning

<p align="center">
<a href="https://arxiv.org/abs/2512.21446"><img src="https://img.shields.io/badge/arXiv-2505.10446-b31b1b.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/sengi/dUltra-math"><img src="https://img.shields.io/badge/Huggingface-dUltra math-yellow" alt="Checkpoint"></a>
<a href="https://huggingface.co/sengi/dUltra-coding"><img src="https://img.shields.io/badge/Huggingface-dUltra coding-yellow" alt="Checkpoint"></a>
<a href="https://huggingface.co/sengi/dUltra-math-b128"><img src="https://img.shields.io/badge/Huggingface-dUltra math b128-yellow" alt="Checkpoint"></a>
<a href="https://huggingface.co/sengi/dUltra-coding-b128"><img src="https://img.shields.io/badge/Huggingface-dUltra coding b128-yellow" alt="Checkpoint"></a>
</p>

## Table of Contents

- [dUltra: Ultra-Fast Diffusion Large Language Models via Reinforcement Learning](#dultra-ultra-fast-diffusion-large-language-models-via-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Results](#results)
  - [Environment Setup](#environment-setup)
  - [Model Checkpoints](#model-checkpoints)
  - [Hardware Requirements](#hardware-requirements)
  - [Training](#training)
    - [Planner Training  (`planner/`)](#planner-training--planner)
    - [GRPO Training (`diffu_grpo/`)](#grpo-training-diffu_grpo)
  - [Evaluation](#evaluation)
    - [Inference Strategies](#inference-strategies)
    - [Math Evaluation](#math-evaluation)
    - [Coding Evaluation](#coding-evaluation)
  - [Troubleshooting](#troubleshooting)
    - [Unable to download APPS dataset](#unable-to-download-apps-dataset)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)

## Overview

This repository implements **dUltra**, a learned path planning framework for masked diffusion language models using Group Relative Policy Optimization (GRPO). By training an unmasking planner head with reinforcement learning, we enable diffusion language models to achieve SOTA performance both in terms of NFE (number of function evaluations) and TPF (token per forward) with high accuracy.

## Results

![Performance Comparison](media/overview.png)

**Figure**: dUltra achieves state-of-the-art accuracy-efficiency trade-offs on mathematical reasoning (GSM8K, MATH500) and code generation (HumanEval, MBPP) tasks. Left panels show accuracy vs. number of function evaluations (NFE) for different block sizes. Right panel illustrates the architecture with the unmasking planner head. Here we use a block size of 128 and a generation length of 256

## Environment Setup

```bash
pip install uv
uv sync
uv pip install -e .
```
You may need to downgrade to `datasets==3.6.0` to download the APPS dataset.

## Model Checkpoints

We provide trained model checkpoints on Hugging Face Hub:

| Model | Block Length | Training Dataset | Description |
|-------|--------------|------------------|-------------|
| [sengi/dUltra-math](https://huggingface.co/sengi/dUltra-math) | 32 | GSM8K | Optimized for math reasoning tasks |
| [sengi/dUltra-math-b128](https://huggingface.co/sengi/dUltra-math-b128) | 128 | GSM8K | Math model with larger block length for faster inference |
| [sengi/dUltra-coding](https://huggingface.co/sengi/dUltra-coding) | 32 | APPS | Optimized for code generation tasks |
| [sengi/dUltra-coding-b128](https://huggingface.co/sengi/dUltra-coding-b128) | 128 | APPS | Coding model with larger block length for faster inference |

To use a trained dUltra model:
```python
from model.llada.lladou import LLaDOUModelLM
from transformers import AutoTokenizer

model = LLaDOUModelLM.from_pretrained(
            "sengi/dUltra-math",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
tokenizer = AutoTokenizer.from_pretrained("sengi/dUltra-math")
```

## Hardware Requirements
All experiments were conducted on NVIDIA H200 GPUs with DDP. Usage of DeepSpeed is not fully supported yet.

## Training

### Planner Training  (`planner/`)

We initialize the planner head by training it to mimic confidence-based sampling (Fast-dLLM method). This provides a strong baseline and avoids degenerate solutions. The planner head learns when to unmask tokens in parallel during denoising.

```bash
cd planner
accelerate launch planner_train.py
```

This pre-trains the planner to mimic confidence-based sampling before GRPO fine-tuning.

### GRPO Training (`diffu_grpo/`)

After initialization, we jointly optimize the base model and planner head using reinforcement learning to discover task-specific unmasking strategies that improve both accuracy and efficiency. The GRPO trainer (`diffu_grpo/diffu_grpo_trainer.py`) extends TRL's `GRPOTrainer` to jointly optimize the diffusion model and unmasking planner head.

To sweep through configurations:
```bash
cd diffu_grpo

sh sbatch_scripts/run.sh
```

To run a specific configuration:
```bash
sh sbatch_scripts/grpo_exp.sh [dataset] [block_length] [adv_clip] [port] [model_path] [num_generations] [per_device_train_batch_size] [num_processes] [temperature]
```

**Parameters**:
1. `dataset` (default: "apps"): Dataset name (gsm8k, apps)
2. `block_length` (default: 32): Maximum tokens unmasked per denoising step
3. `adv_clip` (default: 0.1): Minimum advantage clipping threshold
4. `port` (default: 12345): Main process port for distributed training
5. `model_path` (default: "sengi/dUltra-coding"): Path to model checkpoint or HuggingFace model ID
6. `num_generations` (default: 12): Number of rollout trajectories per prompt
7. `per_device_train_batch_size` (default: 12): Training batch size per GPU
8. `num_processes` (default: 1): Number of GPUs/processes for distributed training
9. `temperature` (default: 0.1): Sampling temperature for token generation

## Evaluation
### Inference Strategies

Located in `model/inference/`:
- **Standard** (`inference_lladou.py`): Iterative denoising with learned unmasking probabilities
- **FastDLLM** (`inference_fastdllm.py`): Confidence-based parallel decoding for faster inference

### Math Evaluation

Instead of `lm_eval`, We use more robust mathematical expression evaluator [math-verify](https://huggingface.co/blog/math_verify_leaderboard).

```bash
cd eval_math
sh run_eval.sh
```

After running `run_eval.sh`, we need to parse the results:
```bash
cd eval_math
python parse_and_get_acc.py eval_dir/
```

### Coding Evaluation

For coding tasks, we evaluate on HumanEval and MBPP datasets using adapted evaluation code from the [Dream-Instruct Evaluatino Toolkit](https://github.com/DreamLM/Dream/tree/31f94a60d187e3fd481fee3bbc2c732eb94a879c/eval_instruct).

```bash
cd eval_coding
bash eval_coding.sh
```

## Troubleshooting

### Unable to download APPS dataset
**Issue**: Error when loading APPS dataset
**Solution**: Downgrade to `datasets==3.6.0`:
```bash
uv pip install datasets==3.6.0
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chen2025dultraultrafastdiffusionlanguage,
      title={dUltra: Ultra-Fast Diffusion Language Models via Reinforcement Learning},
      author={Shirui Chen and Jiantao Jiao and Lillian J. Ratliff and Banghua Zhu},
      year={2025},
      eprint={2512.21446},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.21446},
}
```

## Acknowledgments

This project builds upon and acknowledges the following excellent works:

- **Math Evaluation Code**: The evaluation pipeline is adapted from [d1](https://github.com/dllm-reasoning/d1/)
- **Coding Evaluation Code**: The HumanEval and MBPP evaluation code is adapted from [Dream](https://github.com/DreamLM/Dream/tree/main/eval_instruct)
- **Model Architecture**: The model architecture code is based on [LLaDOU](https://github.com/maple-research-lab/LLaDOU)
