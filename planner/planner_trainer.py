from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer

from model.inference.inference_fastdllm import get_transfer_index_dynamic
from model.inference.inference_lladou import generate

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    use_fast=True,
)


class PlannerTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,  # keys of inputs is determined by the data collator
        return_outputs=False,
        num_items_in_batch=None,
    ):
        input_ids = inputs.pop("input_ids")
        inputs.pop("answer")
        inputs.pop("prompt_lengths")
        # Boolean mask marking the token positions belonging to the masked block
        # that was sampled by the planner data collator (planner/data.py).
        mask_index = inputs.pop("block_index")
        mask_id = self.model.config.mask_token_id
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
        seq_len = input_ids.shape[1]
        block_timestep = 1.0 - (input_ids == mask_id).sum(dim=-1).float() / seq_len
        remask_logits = model(
            last_hidden_states,
            pred_mask_prob=True,
            timestep=block_timestep,
            mask_index=mask_index,
        )
        _, transfer_index = get_transfer_index_dynamic(
            outputs.logits, 0.0, "low_confidence", mask_index, input_ids, 1.0
        )
        valid_mask = mask_index
        logits_for_loss = remask_logits[valid_mask]
        targets_for_loss = transfer_index[valid_mask].to(remask_logits.dtype)

        if logits_for_loss.numel() == 0:
            loss = remask_logits.new_tensor(0.0)
        else:
            pos_count = targets_for_loss.sum().item()
            neg_count = targets_for_loss.numel() - pos_count

            if pos_count > 0 and neg_count > 0:
                pos_weight = torch.tensor(
                    neg_count / pos_count,
                    device=logits_for_loss.device,
                    dtype=logits_for_loss.dtype,
                )
                loss = F.binary_cross_entropy_with_logits(
                    logits_for_loss, targets_for_loss, pos_weight=pos_weight
                )
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits_for_loss, targets_for_loss
                )
        return loss if not return_outputs else (loss, remask_logits.sigmoid())

    # This is for evaluation. Not used for training.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids = inputs.pop("input_ids")
        inputs.pop("answer")
        prompt_lengths = inputs.pop("prompt_lengths")
        inputs.pop("block_index")
        batch_size = input_ids.shape[0]
        max_length = input_ids.shape[1]
        loss = torch.tensor(0.0, device=model.device)
        # Get prompts and pad them to max length
        prompts = []
        for i in range(batch_size):
            prompt = input_ids[i, : prompt_lengths[i]]
            prompts.append(prompt)

        # Pad prompts to same length
        max_prompt_len = max(prompt_lengths)
        padded_prompts = []
        for prompt in prompts:
            pad_len = max_prompt_len - len(prompt)
            padded_prompt = torch.cat(
                [
                    torch.full(
                        (pad_len,), tokenizer.pad_token_id, device=prompt.device
                    ),
                    prompt,
                ]
            )
            padded_prompts.append(padded_prompt)

        padded_prompts = torch.stack(padded_prompts)
        prompt_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": None,
        }
        max_length = max(max_length, prompt_lengths[i])
        out, _, _, sampling_traj = generate(
            model,
            prompt_inputs,
            gen_length=256,
            block_length=32,
            temperature=0.0,
            mask_id=126336,
            verbose=False,
            return_x0_hist=False,
            return_sampling_traj=True,
            use_scheduler=False,
        )
        output = tokenizer.batch_decode(out, skip_special_tokens=False)
        Path("samples").mkdir(parents=True, exist_ok=True)
        with open("samples/generate_output.log", "a") as f:
            for i, answer in enumerate(output):
                f.write(
                    "=" * 50
                    + "\n"
                    + answer.replace("\n", "\\n").replace("<|endoftext|>", "<e>")
                    + "\n"
                    + "=" * 25
                    + f"# Step {len([x for x in sampling_traj[i] if len(x) > 0])}"
                    + "=" * 25
                    + "\n"
                )

        return (loss, None, None)
