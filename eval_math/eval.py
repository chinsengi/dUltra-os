import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from model.inference.inference_fastdllm import generate as fastdllm_generate
from model.inference.inference_lladou import generate as lladou_generate
from model.llada.configuration_llada import LLaDAConfig
from model.llada.lladou import LLaDOUModelLM
from model.llada.modeling_llada import LLaDAModelLM
from model.path_utils import lladou_config_dir
from utils import set_random_seed

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "math500": MATH500Dataset,
}


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    block_length=32,
    scale=30.0,
    normalize=False,
    use_scheduler=False,
    planner_temperature=None,
    mode="training",
    inference_type="lladou",
    steps=None,
    remasking="low_confidence",
    threshold=None,
    factor=1.0,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    per_sample_nfe = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        prompt_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        with torch.no_grad():
            if inference_type == "lladou":
                out, _, _, sampling_traj = lladou_generate(
                    model,
                    prompt_inputs,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    scale=scale,
                    normalize=normalize,
                    use_scheduler=use_scheduler,
                    planner_temperature=planner_temperature,
                    mode=mode,
                    factor=factor,
                    return_sampling_traj=True,
                )
                generated_texts = tokenizer.batch_decode(
                    out[:, -gen_length:], skip_special_tokens=False
                )
                batch_nfe = [
                    sum(1 for step in traj if len(step) > 0) for traj in sampling_traj
                ]
            else:
                out, nfe = fastdllm_generate(
                    model,
                    input_ids,
                    steps=steps if steps is not None else gen_length // block_length,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    remasking=remasking,
                    threshold=threshold,
                    factor=factor,
                )
                generated_texts = tokenizer.batch_decode(
                    out[:, input_ids.shape[1] : input_ids.shape[1] + gen_length],
                    skip_special_tokens=False,
                )
                batch_nfe = [nfe for _ in range(len(gt_answers))]
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
                "nfe": batch_nfe[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        per_sample_nfe.extend(batch_nfe)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        # if dist.get_rank() == 0:
        #     idx = random.randint(0, len(questions) - 1)
        #     print(f"Question: {questions[idx]}")
        #     print("-" * 50)
        #     print("Generation:")
        #     print(generated_texts[idx])
        #     print("-" * 50)
        #     print(f"Ground truth: {gt_answers[idx]}")

    if wall_times:
        total_wall_time = sum(wall_times)
        avg_wall_time = total_wall_time / len(wall_times)
    else:
        total_wall_time = 0.0
        avg_wall_time = 0.0
    avg_nfe = sum(per_sample_nfe) / len(per_sample_nfe) if per_sample_nfe else 0.0
    metrics = {
        "wall_time": avg_wall_time,
        "total_wall_time": total_wall_time,
        "avg_nfe": avg_nfe,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/"
    )
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "math500"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--planner_temperature", type=float, default=None)
    parser.add_argument(
        "--mode", choices=["training", "inference"], default="inference"
    )
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument(
        "--inference_type", choices=["lladou", "fastdllm"], default="lladou"
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--factor", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    num_evals = {
        "gsm8k": -1,
        "math": -1,
        "math500": -1,
    }

    if args.inference_type == "lladou":
        config = LLaDAConfig.from_pretrained(lladou_config_dir())
        model = LLaDOUModelLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
        ).to(local_rank)
    elif args.inference_type == "fastdllm":
        config = AutoConfig.from_pretrained(args.model_path)
        config.flash_attention = True
        model = LLaDAModelLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
        ).to(local_rank)
    else:
        raise ValueError(f"Unknown inference type: {args.inference_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    model_path = Path(args.model_path)
    if len(model_path.parts) >= 2:
        model_name = f"{model_path.parts[-2]}_{model_path.parts[-1]}"
    else:
        model_name = model_path.name or "model"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length}_seed{args.seed}_{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        scale=args.scale,
        normalize=args.normalize,
        use_scheduler=args.use_scheduler,
        planner_temperature=args.planner_temperature,
        mode=args.mode,
        inference_type=args.inference_type,
        steps=args.steps,
        remasking=args.remasking,
        threshold=args.threshold,
        factor=args.factor,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            payload = {
                "generations": metrics["generations"],
                "metrics": {
                    "wall_time": metrics["wall_time"],
                    "total_wall_time": metrics["total_wall_time"],
                    "avg_nfe": metrics["avg_nfe"],
                    "total_processed": metrics["total_processed"],
                },
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
            }
            json.dump(payload, f, indent=2)

    cleanup_ddp()
