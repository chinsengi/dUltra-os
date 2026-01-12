import logging
import random
import re
from typing import Dict

import torch
from datasets import interleave_datasets, load_dataset
from transformers import DefaultDataCollator

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


def _delete_thinking(ans: str) -> str:
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL)


def preprocess_nvidia_dataset(
    tokenizer,
    split: str = "train",
    max_length: int = 2048,
    test_split: float = 0.001,
    num_proc: int = 8,
):
    name = "nvidia/Nemotron-Post-Training-Dataset-v1"

    # Support multiple splits separated by comma
    splits = [s.strip() for s in split.split(",")]
    logging.info(f"Loading {name} dataset with splits: {splits}")

    if len(splits) == 1:
        data = load_dataset(name, split=splits[0], streaming=True)
    else:
        datasets = [load_dataset(name, split=s, streaming=True) for s in splits]
        data = interleave_datasets(datasets)

    def preprocess_and_tokenize(example: Dict):
        messages = example["messages"]

        prompt = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False
        )
        prompt_tokens = tokenizer(
            prompt, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        prompt_len = len(prompt_tokens)
        assistant_text = messages[-1]["content"]
        assistant_text = _delete_thinking(assistant_text)
        ground_truth = prompt + assistant_text + "<|eot_id|>"
        ground_truth_tokens = tokenizer(
            ground_truth,
            add_special_tokens=False,
            return_attention_mask=False,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            padding_side="right",
        )["input_ids"]

        tokens = {
            "prompt_lengths": min(prompt_len, max_length),
            "total_len": max_length,
            "answer": ground_truth_tokens,
        }
        return tokens

    logging.info(f"Tokenizing {name} dataset")

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        remove_columns=data.column_names,
    )
    logging.info(f"Finished tokenizing {name} dataset")

    # Streaming datasets don't support train_test_split, return None for eval
    return tokenized_dataset, None


# tokenized_dataset = tokenized_dataset.select(range(10))


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        self.truncate = kwargs.get("truncate", False)
        self.mask_type = kwargs.get("mask_type", "random")
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert "mask_token_id" in kwargs, (
                "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            )
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch):
        input_ids = batch["answer"].clone()
        B, N = input_ids.shape
        prompt_len = batch["prompt_lengths"]
        if self.truncate:
            max_prompt_len = prompt_len.max()
            N = random.randint(max_prompt_len, N)
        input_ids = input_ids[..., :N]
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        block_index = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(B):
            if self.mask_type == "random":
                if N - prompt_len[i] <= 2:
                    continue
                block = sorted(random.sample(range(prompt_len[i], N - 1), 2))
                # make sure that the block is at least 32 tokens long
                block[0] = min(block[0], N - 32)
                block[1] = max(block[1], block[0] + 32)
                block_index[i, block[0] : block[1]] = True
                n_mask = max(1, torch.ceil((block[1] - block[0]) * t[i]).int())
                mask_index = random.sample(range(block[0], block[1]), n_mask)
                block_index[i] = torch.zeros((N,), dtype=torch.bool)
                block_index[i, mask_index] = 1
                input_ids[i, block_index[i]] = self.mask_token_id
                input_ids[i, block[1] :] = self.mask_token_id
            elif self.mask_type == "ar":
                n_mask = max(1, torch.ceil((N - prompt_len[i]) * t[i]).int())
                input_ids[i, N - n_mask :] = self.mask_token_id
                block_index[i, N - n_mask :] = True
            else:
                raise ValueError(f"Invalid mask type: {self.mask_type}")
        return input_ids, block_index

    def __call__(self, batch):
        batch = super().__call__(batch)
        noisy_batch, batch["block_index"] = self.forward_process(batch)
        batch["input_ids"] = noisy_batch.long()
        batch.pop("total_len")
        if "t" in batch:
            batch.pop("t")
        return batch


def preprocess_s1k_dataset(tokenizer, max_length, test_split=0.01, num_proc=8):
    data = load_dataset("simplescaling/s1K", split="train")

    # Use datasets.map() for efficient preprocessing
    def preprocess_function(example):
        """Preprocessing function for datasets.map()"""
        question = SYSTEM_PROMPT + "\n\n" + example["question"]
        trajectory = f"<reasoning>{example['thinking_trajectories'][0]}</reasoning>\n<answer>{example['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).input_ids.squeeze(0)
        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        return {
            "input_ids": tokenized_input,
            "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
        }

    preprocessed_dataset = data.map(
        preprocess_function,
        num_proc=num_proc,
        remove_columns=data.column_names,  # Remove original columns
        desc="Preprocessing dataset",
    )

    preprocessed_data = list(preprocessed_dataset)

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
