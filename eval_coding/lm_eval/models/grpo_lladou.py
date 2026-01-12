# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
More documentatio: https://github.com/EleutherAI/lm-evaluation-harness/blob/de496b80d60c267a2d7eea3b3c1dc40f693daee7/docs/model_guide.md
"""

import json
import os
import time

import accelerate
import torch
from huggingface_hub import snapshot_download
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer

from model.inference.inference_lladou import generate
from model.llada.configuration_llada import LLaDAConfig
from model.llada.lladou import LLaDOUModelLM
from model.path_utils import lladou_config_dir
from utils import set_random_seed


@register_model("grpo_lladou")
class LLaDOGRPOEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        device="cuda",
        normalize=True,
        scale=30.0,
        use_scheduler=False,
        save_dir="./.cache",
        show_speed=True,
        mode="training",
        factor=1.0,
        seed=42,
        **kwargs,
    ):
        """
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        set_random_seed(seed)
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        local_dir = ""
        if os.path.isdir(model_path):
            local_dir = model_path
        else:
            if self.accelerator is None or self.accelerator.is_main_process:
                local_dir = snapshot_download(model_path)
            if self.accelerator is not None:
                local_dir = self.accelerator.gather_for_metrics([local_dir])[0]
        lladou_config = LLaDAConfig.from_pretrained(lladou_config_dir())
        assert lladou_config.flash_attention
        self.model = LLaDOUModelLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=lladou_config,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        # Fix known Mistral regex bug when loading tokenizers from certain checkpoints.
        # If the flag is unsupported in older transformers versions, it will be ignored.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True
        )

        self.mode = mode
        self.batch_size = int(batch_size)
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.normalize = normalize
        self.scale = scale
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.use_scheduler = use_scheduler
        self.factor = factor

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _forward_process(self, batch, prompt_index):
        raise NotImplementedError

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        raise NotImplementedError

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        raise NotImplementedError

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        total_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None and self.save_dir != "":
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        if self.accelerator is not None:
            # Ensure all ranks are ready before timing starts
            self.accelerator.wait_for_everyone()
        start_time = time.time()

        profile_stats = None
        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(
                    m, add_generation_prompt=True, tokenize=False
                )
                input_ids = self.tokenizer(user_input)["input_ids"]
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            # pad batched_input_ids to the same length
            batched_input_ids = [
                torch.cat(
                    [
                        torch.full(
                            (1, max_len - len(input_ids)),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.tensor(
                            input_ids, dtype=torch.long, device=self.device
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            attention_mask = None

            # stop_tokens = req.args[1]["until"]
            input_ids = batched_input_ids
            inputs = {
                "input_ids": batched_input_ids,
                "attention_mask": attention_mask,
            }
            assert self.gen_length % self.block_length == 0, (
                f"gen_length {self.gen_length} % block_length {self.block_length} != 0"
            )
            (
                generated_answer,
                _,
                _,
                _,
                num_steps,
                # profile_stats,
            ) = generate(
                self.model,
                inputs,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                mask_id=self.mask_id,
                return_x0_hist=False,
                return_sampling_traj=False,
                return_sequence_logp=False,
                normalize=self.normalize,
                scale=self.scale,
                use_scheduler=self.use_scheduler,
                mode=self.mode,
                factor=self.factor,
                return_num_steps=True,
                profile_forwards=False,
            )
            nfe_per_sample = [num_steps for _ in range(len(batch))]
            # if "task_id" in req.doc and str(req.doc["task_id"]).lower().startswith(
            #     "humaneval"
            # ):
            generated_answer_ids = generated_answer[:, input_ids.shape[1] :]
            if self.show_speed:
                num_tokens += (generated_answer_ids != 126081).sum()
                num_nfe += sum(nfe_per_sample)
                total_tokens += generated_answer_ids.shape[0] * self.gen_length
            batched_generated_answer = [
                self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True)
                for i in range(len(generated_answer_ids))
            ]
            # else:
            #     batched_generated_answer = []
            #     for i in range(len(generated_answer)):
            #         generated_answer_i = self.tokenizer.decode(
            #             generated_answer[i][input_ids.shape[1] :],
            #             skip_special_tokens=False,
            #         )
            #         for stop_seq in stop_tokens:
            #             if stop_seq in generated_answer_i:
            #                 generated_answer_i = generated_answer_i.split(stop_seq)[0]
            #         generated_answer_ids = torch.tensor(
            #             self.tokenizer(generated_answer_i)["input_ids"]
            #         )
            #         if self.show_speed:
            #             num_tokens += (generated_answer_ids != 126081).sum()
            #             num_nfe += nfe_per_sample[i]
            #         generated_answer_i = self.tokenizer.decode(
            #             generated_answer_ids, skip_special_tokens=True
            #         )
            #         batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                with open(os.path.join(self.save_dir, "generation_log.txt"), "a") as f:
                    for i in range(len(batched_generated_answer)):
                        f.write("=" * 20 + "\n")
                        f.write(
                            f"input: \n{self.tokenizer.decode(input_ids[i], skip_special_tokens=True)}\n"
                        )
                        f.write(f"answer: \n{batched_generated_answer[i]}\n")
                        f.write(f"nfe: {nfe_per_sample[i]}\n")
                        f.write(f"avg nfe: {num_nfe / len(output)}\n")
                        f.write("=" * 20 + "\n\n")
        end_time = time.time()
        total_logits_forward = 0.0
        total_planner_forward = 0.0
        profile_batches = 0

        if self.show_speed:
            elapsed_time = end_time - start_time
            time_per_nfe = elapsed_time / num_nfe if num_nfe else None
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {elapsed_time} seconds")
            print(f"Tokens per second: {num_tokens / elapsed_time}")
            print(f"Total NFE is {num_nfe}")
            if time_per_nfe is not None:
                print(f"Time per NFE: {time_per_nfe} seconds")
            if profile_stats is not None:
                print(
                    f"Forward time logits/planner: {profile_stats['logits_forward_sec']:.3f}s / {profile_stats['planner_forward_sec']:.3f}s"
                )
                total_logits_forward += profile_stats["logits_forward_sec"]
                total_planner_forward += profile_stats["planner_forward_sec"]
                profile_batches += 1
                avg_logits = total_logits_forward / profile_batches
                avg_planner = total_planner_forward / profile_batches
                print(
                    f"Average forward time logits/planner: {avg_logits:.3f}s / {avg_planner:.3f}s"
                )
            if self.save_dir is not None:
                with open(os.path.join(self.save_dir, "speed_log.txt"), "a") as f:
                    f.write(f"Total number of tokens generated: {num_tokens}\n")
                    f.write(f"Total time taken: {elapsed_time} seconds\n")
                    f.write(f"Tokens per second: {num_tokens / elapsed_time}\n")
                    f.write(f"Total NFE is {num_nfe}\n")
                    if time_per_nfe is not None:
                        f.write(f"Time per NFE: {time_per_nfe} seconds\n")
                    if profile_stats is not None:
                        f.write(
                            f"Forward time logits/planner: {profile_stats['logits_forward_sec']:.6f}s / {profile_stats['planner_forward_sec']:.6f}s\n"
                        )
                        f.write(
                            f"Average forward time logits/planner: {avg_logits:.6f}s / {avg_planner:.6f}s\n"
                        )

        return output
