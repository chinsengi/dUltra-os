import copy
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from accelerate import DistributedType
from accelerate.utils import gather_object
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers.utils import is_peft_available, logging
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import (
    nanmax,
    nanmin,
    nanstd,
    selective_log_softmax,
    truncate_with_protected_tokens,
)

from model.inference.inference_lladou import generate
from utils import accel_break

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# from SFT.inference import generate

if is_peft_available():
    pass
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.
    """

    _tag_names = ["trl", "grpo", "dllm"]

    def __init__(self, **kwargs):
        # Hack to keep the ref model to be none
        beta = kwargs["args"].beta
        kwargs["args"].beta = 0.0
        super().__init__(**kwargs)
        self.beta = beta

        self.model_wrapped = self.model_wrapped.to(torch.bfloat16)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions for prompts and compute associated rewards and advantages. The output of this method is used as the input field to the _get_per_token_logps_and_entropies and reward functions.

        Args:
            inputs: the data batch. format: [{'question': str, 'answer': int, "prompt": str}]

        Returns:
            output: dict[str, Union[torch.Tensor, Any]]
                - prompt_ids: torch.Tensor
                - prompt_mask: torch.Tensor
                - completion_ids: torch.Tensor
                - completion_mask: torch.Tensor
                - advantages: torch.Tensor
                - num_items_in_batch: int
                - old_per_token_logps: torch.Tensor
                - ref_per_token_logps: torch.Tensor
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)
        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            raise ValueError("Images are not supported for DiffuGRPOTrainer")

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,  # pad to the longest sequence in the batch
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ]
            protected = [token for token in protected if token is not None]
            # breakpoint()
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.pad_token)})+", "", text)
                for text in prompts_text
            ]

            if self.image_token is not None:
                raise ValueError("Images are not supported for DiffuGRPOTrainer")

        if self.use_vllm:
            raise ValueError(
                "VLLM is not supported for masked diffusion language models."
            )

        if self.use_transformers_paged:
            raise ValueError(
                "Transformers Paged is not supported for masked diffusion language models."
            )

        # Regular generation path
        prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = (
            prompt_ids,
            prompt_mask,
        )

        # Rollout
        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext(),
        ):
            (
                prompt_completion_ids,
                x0_hist,
                sequence_logp,
                sampling_traj,
            ) = generate(
                unwrapped_model,
                prompt_inputs,
                gen_length=self.max_completion_length,
                block_length=self.args.block_length,
                temperature=self.temperature,
                mask_id=self.args.mask_id,
                return_sampling_traj=True,
                return_sequence_logp=True,
                return_x0_hist=True,
                mode=self.args.rollout_mode,
                normalize=self.args.normalize,
                scale=self.args.scale,
                verbose=False,
                use_scheduler=self.args.use_scheduler,
            )
            logger.info("Rollout completed")
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                torch.cuda.empty_cache()

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            row[mask_row].tolist()
            for row, mask_row in zip(completion_ids, completion_mask.bool())
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = (
            agg_completion_lengths.sum()
        )  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out the entire sequence of truncated completions using completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        # prompt mask is 1 for the prompt tokens, 0 for the completion tokens
        # completion mask is 1 for the completion tokens, 0 for the discarded tokens
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        if sequence_logp is not None and getattr(self.args, "use_reverse_kl", False):
            total_sequence_logp = sequence_logp.sum(dim=1)
            for example, logprob in zip(inputs, total_sequence_logp):
                example["student_logprob"] = logprob

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # **samples** may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # This will only run when self._step % generate_every == 0 or self._buffered_inputs is None
            # generate_every = (
            #     self.args.steps_per_generation * self.num_iterations
            # )  # generation frequency
            old_per_token_logps = sequence_logp.clone().detach()
            # if self.args.gradient_accumulation_steps % generate_every != 0:
            #     old_per_token_logps = sequence_logp.clone().detach()
            # else:
            #     old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                ref_per_token_logps = old_per_token_logps.clone().detach()
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        for i, example in enumerate(inputs):
            example["sampling_traj"] = sampling_traj[i]
        rewards_per_func = self._calculate_rewards(
            inputs, original_prompts, completions, completion_ids_list
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        ########################################################
        ####################### LOGGING ########################
        ########################################################
        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(
                std_func_rewards
            )
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        ########################################################
        ####################### OUTPUT #########################
        ########################################################
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "sampling_traj": sampling_traj,
            "x0_hist": x0_hist,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _compute_unmasking_prob(
        self, model, last_hidden_states, cur_input, cur_block, prompt_len
    ):
        mask_id = self.args.mask_id
        mask_index = cur_input == mask_id
        block_length = self.args.block_length
        mask_index[:, prompt_len + (cur_block + 1) * block_length :] = 0
        timestep = 1.0 - (cur_input == mask_id).sum(dim=-1) / cur_input.shape[1]
        remask_logits = model(
            last_hidden_states,
            pred_mask_prob=True,
            timestep=timestep,
            mask_index=mask_index,
            # current_block=cur_block,
        )

        unmasking_prob = remask_logits.sigmoid()
        unmasking_prob = torch.where(mask_index, unmasking_prob, 0.0)
        # unmasking_prob = unmasking_prob.to(last_hidden_states.dtype)
        # scale = self.args.scale
        # if self.args.normalize:
        #     scale = unmasking_prob.sum(dim=-1, keepdim=True)
        # factor = (timestep.unsqueeze(-1) + 1e-10) * self.max_completion_length / 4
        # unmasking_prob = unmasking_prob * factor
        # unmasking_prob = unmasking_prob / scale

        # unmasking_prob = torch.clamp(unmasking_prob, min=1e-6, max=1.0 - 1e-6)
        # unmasking_prob = torch.where(mask_index, unmasking_prob, 0.0)

        return unmasking_prob

    def _compute_loss(self, model, inputs, num_items_in_batch):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        prompt_len = prompt_ids.size(1)
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Replay must mirror the rollout mask: during generation only the prompt tokens
        # were marked as valid, so keep zeros on the completion portion.
        prompt_only_mask = torch.ones_like(prompt_ids, dtype=prompt_mask.dtype)
        attention_mask = torch.cat(
            [
                prompt_only_mask,
                torch.zeros_like(completion_ids, dtype=prompt_mask.dtype),
            ],
            dim=1,
        )
        sampling_traj = inputs["sampling_traj"]
        x0_hist = inputs["x0_hist"]
        all_advantages = inputs["advantages"]
        # we don't need the attention mask here because llada will ignore it anyways
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # chunk inputs into smaller batches to reduce memory peak
        batch_size = input_ids.size(0) // 2
        assert input_ids.size(0) % batch_size == 0
        return_loss = 0.0
        loss_list = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]

            cur_traj = sampling_traj[start : start + batch_size]
            cur_x0_hist = x0_hist[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            traj_len = len(cur_traj[0])

            all_traj_len = self.accelerator.gather(
                torch.tensor(traj_len, device=input_ids_batch.device)
            )
            max_traj_len = all_traj_len.max().item()

            mask_id = self.args.mask_id
            cur_input = input_ids_batch.clone()
            cur_input[:, prompt_len:] = mask_id
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                torch.cuda.empty_cache()
            for step in tqdm(range(max_traj_len), desc="Computing per-token logps"):
                # logger.info(f"Step {step} of {traj_len}")
                # running the model in batches per step
                # breakpoint()
                outputs = model(
                    cur_input,
                    output_hidden_states=True,
                    attention_mask=attention_mask_batch,
                    final_hidden_state_only=True,
                )
                # breakpoint()
                last_hidden_states = outputs.hidden_states[-1]
                logits = outputs.logits
                # logits = logits.to(model.dtype)
                logits[:, :, mask_id] = -torch.inf
                bad_flag = False
                if step < traj_len:
                    token_in_block = [x[step] for x in cur_traj if len(x[step]) > 0]
                else:
                    token_in_block = []
                if step >= traj_len or len(token_in_block) == 0:
                    # construct dummy loss
                    next_input = cur_input.clone()
                    unmasking_prob = self._compute_unmasking_prob(
                        model, last_hidden_states, cur_input, 0, prompt_len
                    )
                    loss = logits.exp().sum() * 0.0 + unmasking_prob.sum() * 0.0
                else:
                    # Divide logits by sampling temperature. (not sure if it will work)
                    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
                    # logits = logits / self.temperature
                    x0_logp = selective_log_softmax(logits, cur_x0_hist[:, step])
                    cur_block = (
                        token_in_block[0][0] - prompt_len
                    ) // self.args.block_length
                    unmasking_prob = self._compute_unmasking_prob(
                        model, last_hidden_states, cur_input, cur_block, prompt_len
                    )

                    per_token_logps = []
                    next_input = cur_input.clone()
                    for batch in range(batch_size):
                        single_input = cur_input[batch, :].unsqueeze(0)
                        mask_token_mask = (single_input == mask_id).squeeze()
                        unmasking_index_mask = torch.zeros_like(
                            mask_token_mask, dtype=torch.bool
                        )
                        unmasking_index_mask[cur_traj[batch][step]] = True
                        keep_mask_index_mask = mask_token_mask & ~unmasking_index_mask
                        cur_logp = torch.zeros_like(
                            unmasking_prob[batch], dtype=torch.float32
                        ).unsqueeze(0)
                        if len(cur_traj[batch][step]) > 0:
                            cur_logp[:, keep_mask_index_mask] = torch.log1p(
                                -unmasking_prob[batch, keep_mask_index_mask]
                            )
                            cur_logp[:, unmasking_index_mask] = (
                                torch.log(unmasking_prob[batch, unmasking_index_mask])
                                + x0_logp[batch, unmasking_index_mask]
                            )
                        if (
                            torch.isnan(cur_logp).sum() > 0
                            or not torch.isfinite(cur_logp).all()
                        ):
                            isnan = torch.isnan(cur_logp).sum() > 0
                            isinf = not torch.isfinite(cur_logp).all()
                            isx0_logp_inf = not torch.isfinite(x0_logp).all()
                            bad_flag = True
                            bad_process_index = self.accelerator.process_index
                            raise ValueError(
                                f"cur_logp is nan: {isnan}, isinf: {isinf}, isx0_logp_inf: {isx0_logp_inf}"
                            )

                        per_token_logps.append(cur_logp)

                        # unmask cur_input
                        next_input[batch, unmasking_index_mask] = input_ids_batch[
                            batch, unmasking_index_mask
                        ]

                    per_token_logps = torch.cat(per_token_logps)

                    # Compute the KL divergence between the model and the reference model
                    if self.beta != 0.0:
                        ref_per_token_logps = inputs["ref_per_token_logps"]
                        per_token_kl = (
                            torch.exp(ref_per_token_logps[:, step, :] - per_token_logps)
                            - (ref_per_token_logps[:, step, :] - per_token_logps)
                            - 1
                        )

                    # Compute the loss
                    advantages = all_advantages[start : start + batch_size]
                    # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
                    # old_per_token_logps == per_token_logps. In this case we can skip its computation
                    # (see _generate_and_score_completions) and instead use per_token_logps.detach().
                    old_per_token_logps = inputs.get("old_per_token_logps")
                    old_per_token_logps = (
                        per_token_logps.detach()
                        if old_per_token_logps is None
                        else old_per_token_logps[start : start + batch_size, step, :]
                    )

                    log_ratio = per_token_logps - old_per_token_logps
                    if log_ratio.abs().max() > 1e-2:
                        logger.warning(
                            "not on policy, max log ratio: "
                            + str(log_ratio.abs().max().item())
                        )

                    log_ratio = log_ratio[:, prompt_len:]
                    # TODO: how should we mask the log_ratio?
                    log_importance_weights = log_ratio.sum(-1)
                    log_importance_weights = log_importance_weights.unsqueeze(-1)

                    # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
                    # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

                    coef_1 = torch.exp(log_importance_weights)
                    coef_2 = torch.clamp(
                        coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high
                    )

                    # Two-sided clipping
                    if self.args.delta is not None:
                        coef_1 = torch.clamp(coef_1, max=self.args.delta)
                    advantages = torch.where(
                        advantages < self.args.advantage_min_clip,
                        torch.zeros_like(
                            advantages
                        ),  # ignores advantages below a threshold
                        advantages,
                    )

                    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

                    if self.beta != 0.0:
                        per_token_loss = per_token_loss + self.beta * per_token_kl

                    loss = (
                        per_token_loss.sum()
                        / per_token_loss.size(0)
                        / self.max_completion_length
                    )
                    loss = loss / self.current_gradient_accumulation_steps
                    if loss.grad_fn is None:
                        # this means that no token is unmasked, this can happen because generated completion rollout is splitted into smaller batches
                        # raise ValueError("No gradient found")
                        loss = logits.exp().sum() * 0.0 + unmasking_prob.sum() * 0.0
                loss_list.append(loss.item())
                # print(f"Loss: {loss}")
                # Backward pass
                if bad_flag or loss.isnan():
                    accel_break(bad_process_index)
                # logger.info(f"[Rank {self.accelerator.process_index}]Loss: {loss}")
                self.backward(loss, num_items_in_batch)
                return_loss += loss.detach()

                del cur_input
                cur_input = next_input

                # Log the metrics
                mode = "train" if self.model.training else "eval"

                completion_token_count = completion_mask.sum().clamp(min=1.0)

                def masked_batch_mean(x):
                    if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                        return x.mean()
                    else:
                        return (x * completion_mask).sum() / completion_token_count

                # Compute the clipped probability ratios
                is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (
                    advantages.unsqueeze(1) < 0
                )
                is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
                    advantages.unsqueeze(1) > 0
                )
                is_region_clipped = is_low_clipped | is_high_clipped

                low_clip = masked_batch_mean(is_low_clipped.float())
                high_clip = masked_batch_mean(is_high_clipped.float())
                clip_ratio = masked_batch_mean(is_region_clipped.float())

                gathered_low_clip = self.accelerator.gather(low_clip)
                self._metrics[mode]["clip_ratio/low_mean"].append(
                    gathered_low_clip.nanmean().item()
                )
                self._metrics[mode]["clip_ratio/low_min"].append(
                    nanmin(gathered_low_clip).item()
                )
                gathered_high_clip = self.accelerator.gather(high_clip)
                self._metrics[mode]["clip_ratio/high_mean"].append(
                    gathered_high_clip.nanmean().item()
                )
                self._metrics[mode]["clip_ratio/high_max"].append(
                    nanmax(gathered_high_clip).item()
                )
                gathered_clip_ratio = self.accelerator.gather(clip_ratio)
                self._metrics[mode]["clip_ratio/region_mean"].append(
                    gathered_clip_ratio.nanmean().item()
                )
            # breakpoint()
        del cur_input
        return_loss = self.accelerator.reduce(return_loss, reduction="mean")
        # if(return_loss.item()>0.01):
        #     breakpoint()
        return return_loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        execute N gradient steps for each diffusion rollout, where N is the number of diffusion steps.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # Because there needs to be multiple forward passes for importance weight computation, we will backprop in compute_loss
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        del inputs

        return loss

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            raise NotImplementedError(
                "Liger loss is not supported for DiffuGRPOTrainer"
            )
        else:
            return self._compute_loss(model, inputs, num_items_in_batch)

    def backward(self, loss: torch.Tensor, num_items_in_batch):
        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
        if (
            not self.model_accepts_loss_kwargs or num_items_in_batch is None
        ) and self.compute_loss_func is None:
            # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
            loss = loss / self.current_gradient_accumulation_steps

        # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        # https://github.com/huggingface/transformers/pull/35808
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss, **kwargs)
