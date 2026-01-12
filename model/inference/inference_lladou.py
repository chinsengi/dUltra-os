import argparse
import time

import torch
from transformers import AutoTokenizer
from transformers.utils import logging
from trl.trainer.utils import selective_log_softmax

from model.llada.configuration_llada import LLaDAConfig
from model.llada.lladou import LLaDOUModelLM
from model.path_utils import lladou_config_dir

logger = logging.get_logger("transformers")
logging.set_verbosity(logging.INFO)


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    dtype = logits.dtype
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (logits.exp() / gumbel_noise).to(dtype)


def generate(
    model,
    prompt_inputs,
    gen_length=256,
    block_length=256,
    temperature=0.3,
    mask_id=126336,
    return_sequence_logp=False,
    return_sampling_traj=False,
    return_x0_hist=True,
    verbose=False,
    scale=30.0,
    normalize=False,
    mode="training",
    use_scheduler=False,
    planner_temperature=None,
    factor=1.0,
    return_num_steps=False,
    profile_forwards=False,
):
    """
    implements the confidence-threshold diffusion generation.
    Args:
        model: Mask predictor.
        prompt_inputs: a dict containing the A tensor of shape (batch_size, L).
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
        planner_temperature: Optional temperature applied to planner logits.
    """
    # ignore attention mask here
    prompt, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    batch_size = prompt.shape[0]
    prompt_length = prompt.shape[1]
    x = torch.full(
        (batch_size, gen_length + prompt_length), mask_id, dtype=torch.long
    ).to(model.device)
    x[:, :prompt_length] = prompt.clone()

    if gen_length == 0:
        return x
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    sequence_logp = (
        torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        if return_sequence_logp
        else None
    )
    sampling_traj = [[] for _ in range(batch_size)] if return_sampling_traj else None
    num_steps = 0
    logp_list = []
    x0_list = []
    time_logits_forward = 0.0
    time_planner_forward = 0.0
    for num_block in range(num_blocks):
        num_masks = block_length * batch_size
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length
        while num_masks > 0:
            logits_start = time.perf_counter() if profile_forwards else None
            outputs = model(
                x,
                output_hidden_states=True,
                attention_mask=attention_mask,
                final_hidden_state_only=True,
            )
            if profile_forwards:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time_logits_forward += time.perf_counter() - logits_start
            last_hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            # logits = logits.to(model.dtype)
            logits[:, :, mask_id] = -torch.inf

            # sample using Gumbel-max
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.max(logits_with_noise, dim=-1).indices  # b, l

            # get log probability
            x0_logp = (
                selective_log_softmax(logits, x0) if return_sequence_logp else None
            )
            if return_x0_hist:
                x0_list.append(x0)

            # get unmasking probability using
            mask_index = x == mask_id
            mask_index[:, block_end:] = False
            # timestep = 1.0 - (x[:, block_start:block_end] == mask_id).sum(dim=-1) / block_length
            # timestep = 1.0 - (x == mask_id).sum(dim=-1) / x.shape[1]
            # cur_block = torch.zeros(
            #     batch_size, x.shape[1], device=x.device, dtype=torch.bool
            # )
            # cur_block[:, block_start:block_end] = True

            block_timestep = 1.0 - (x == mask_id).sum(dim=-1) / x.shape[1]
            planner_start = time.perf_counter() if profile_forwards else None
            unmasking_prob = model(
                last_hidden_states,
                pred_mask_prob=True,
                timestep=block_timestep,  # global unmasking ratio
                mask_index=mask_index,  # only mark mask tokens within current block
            )
            if profile_forwards:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time_planner_forward += time.perf_counter() - planner_start
            if planner_temperature is not None:
                unmasking_prob = unmasking_prob / planner_temperature
            unmasking_prob = unmasking_prob.sigmoid()
            # unmasking_prob = torch.where(
            #     mask_index, unmasking_prob, 0.0
            # )  # this has higher precision than unmasking_prob * mask_index
            unmasking_prob = unmasking_prob * mask_index

            # monotonically increase the expectation of the number of unmasked tokens
            if use_scheduler:
                if normalize:
                    scale = unmasking_prob.sum(dim=-1, keepdim=True)
                timestep = (
                    1.0
                    - (x[:, block_start:block_end] == mask_id).sum(dim=-1)
                    / block_length
                )
                factor = (timestep.unsqueeze(-1) + 1e-10) * gen_length / 4
                unmasking_prob = unmasking_prob * factor
                unmasking_prob = unmasking_prob / scale

                unmasking_prob = torch.clamp(unmasking_prob, min=1e-6, max=1.0 - 1e-6)
                unmasking_prob = torch.where(mask_index, unmasking_prob, 0.0)

            if num_steps == 0 and verbose:
                with open("debug.txt", "a") as f:
                    f.write(f"{x[0]=}\n")
                    f.write(f"{logits[0]=}\n")
                    f.write(f"{x0_logp[0]=}\n")
            # transfer tokens
            unmasking_prob = unmasking_prob * float(factor)
            if mode == "training":
                unmasking_index = (
                    torch.rand(batch_size, x.shape[1], device=x.device) < unmasking_prob
                ) & mask_index
            elif mode == "inference":
                # during inference, we unmask the expected number of tokens with the highest unmasking probability
                num_unmasking_tokens = torch.sum(
                    unmasking_prob, dim=-1, keepdim=True
                ).int()
                unmasking_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                for b in range(batch_size):
                    _, select_index = torch.topk(
                        unmasking_prob[b], k=num_unmasking_tokens[b]
                    )
                    unmasking_index[b, select_index] = True
            else:
                raise ValueError(f"Invalid mode: {mode}")
            force_sampling_flag = False
            batch_num_masks = unmasking_index.sum(-1)
            if (batch_num_masks == 0).any():
                force_sampling_flag = True
                for b in range(batch_size):
                    if batch_num_masks[b] == 0 and mask_index[b].any():
                        confidence = unmasking_prob[b]
                        if confidence.sum() <= 0:
                            # in very rare cases, the confidence is all 0, we need to transfer the first mask token
                            idx = torch.where(mask_index[b])[0][0]
                        else:
                            idx = torch.argmax(confidence)
                            # idx = torch.multinomial(confidence, 1)
                        unmasking_index[b, idx] = True
            x[unmasking_index] = x0[unmasking_index]
            if verbose:
                logger.info(x0[unmasking_index])
                logger.info(force_sampling_flag)
            num_masks -= unmasking_index.sum().item()

            if return_sequence_logp:
                keep_mask_index = x == mask_id
                sequence_logp = torch.zeros(
                    x.shape, dtype=torch.float32, device=x.device
                )
                for i in range(batch_size):
                    if unmasking_index[i].sum().item() > 0:
                        # calculate the sequence log probability
                        sequence_logp[i, unmasking_index[i]] = (
                            torch.log(unmasking_prob[i, unmasking_index[i]])
                            + x0_logp[i, unmasking_index[i]]
                        )
                        sequence_logp[i, keep_mask_index[i]] = torch.log1p(
                            -unmasking_prob[i, keep_mask_index[i]]
                        )
                if verbose:
                    with open("debug.txt", "a") as f:
                        f.write(f"{num_steps=}{sequence_logp[0]=}\n")
                        f.write(f"{num_steps=}{unmasking_prob[0]=}\n")
                logp_list.append(sequence_logp.to(torch.float32))

            if return_sampling_traj:
                for i in range(batch_size):
                    col_idx = torch.where(unmasking_index[i])[0]
                    col_idx = col_idx.tolist()
                    # we need to ignore the empty list later on
                    sampling_traj[i].append(col_idx)

            num_steps += 1
            # if verbose:
            if verbose:
                logger.info(
                    f"Block {num_block} Step {num_steps} completed, {num_masks} masks left"
                )
    # the size of first dimension of old_logps needs to be batch size, TRL GRPO Trainer will shuffle along this dimension
    old_logps = (
        torch.stack(logp_list, dim=1).to(torch.float32)
        if return_sequence_logp
        else None
    )
    x0_hist = torch.stack(x0_list, dim=1) if return_x0_hist else None
    profile_stats = (
        {
            "logits_forward_sec": time_logits_forward,
            "planner_forward_sec": time_planner_forward,
        }
        if profile_forwards
        else None
    )
    if return_num_steps and profile_forwards:
        return x, x0_hist, old_logps, sampling_traj, num_steps, profile_stats
    if return_num_steps:
        return x, x0_hist, old_logps, sampling_traj, num_steps
    if profile_forwards:
        return x, x0_hist, old_logps, sampling_traj, profile_stats
    return x, x0_hist, old_logps, sampling_traj


def main():
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sengi/lladou-gsm8k-b32",
        help="Model name",
    )
    # Gen-Verse/TraDo-4B-Instruct
    # GSAI-ML/LLaDA-8B-Instruct
    parser.add_argument("--gen_length", type=int, default=256, help="Generation length")
    parser.add_argument(
        "--use_official",
        action="store_true",
        help="Use official model",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--use_scheduler", action="store_true", help="Use scheduler")
    parser.add_argument("--normalize", action="store_true", help="Normalize")
    parser.add_argument(
        "--mode", default="inference", type=str, help="Mode (training | inference)"
    )
    parser.add_argument("--log_steps", action="store_true", help="Log NFE per block")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
    )
    config = LLaDAConfig.from_pretrained(lladou_config_dir())
    model = LLaDOUModelLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config,
    ).cuda()

    prompt = r"def valid_date(date):\n    \"\"\"You have to write a function which validates a given date string and\n    returns True if the date is valid otherwise False.\n    The date is valid if all of the following rules are satisfied:\n    1. The date string is not empty.\n    2. The number of days is not less than 1 or higher than 31 days for months 1,3,5,7,8,10,12. And the number of days is not less than 1 or higher than 30 days for months 4,6,9,11. And, the number of days is not less than 1 or higher than 29 for the month 2.\n    3. The months should not be less than 1 or higher than 12.\n    4. The date should be in the format: mm-dd-yyyy\n\n    for example: \n    valid_date(\'03-11-2000\') => True\n\n    valid_date(\'15-01-2012\') => False\n\n    valid_date(\'04-0-2040\') => False\n\n    valid_date(\'06-04-2020\') => True\n\n    valid_date(\'06/04/2020\') => False\n    \"\"\" "
    SYSTEM_PROMPT = r"""
You are an expert Python programmer. Carefully read the task description and unit tests,and write a correct and efficient python implementation. Enclose your final code within <answer></answer> tags, i.e., <answer> code here </answer>.
"""

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )

    # prompt = prompt.replace("\n", r"\n")
    tokenized_prompt = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        padding_side="left",
    )
    # breakpoint()
    n_repeats = 1
    input_ids = torch.tensor(
        [tokenized_prompt["input_ids"] for i in range(n_repeats)]
    ).to(device)
    attention_mask = (
        torch.tensor(tokenized_prompt["attention_mask"]).to(device).unsqueeze(0)
    )
    prompt_inputs = {
        "input_ids": input_ids,
        # "attention_mask": torch.ones_like(input_ids),
        "attention_mask": attention_mask,
    }
    mask_id = 126336
    nfe_list = []
    gen_time_list = []
    with torch.no_grad():
        for i in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            out, _, _, sampling_traj = generate(
                model,
                prompt_inputs,
                gen_length=args.gen_length,
                temperature=args.temperature,
                mask_id=mask_id,
                return_sequence_logp=False,
                return_sampling_traj=args.log_steps,
                verbose=False,
                mode=args.mode,
                normalize=args.normalize,
                scale=30,
                block_length=args.block_length,
                use_scheduler=args.use_scheduler,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time = time.perf_counter() - start_time
            gen_time_list.append(gen_time)

            # out_str = tokenizer.batch_decode(
            #     out[:, input_ids.shape[1] :], skip_special_tokens=True
            # )
            # output = out_str[0]
            # .replace("\n", "\\n").replace("<|endoftext|>", "<e>")
            # output = (
            #     output
            #     + "\n"
            #     + "=" * 25
            #     + f"# Block_len {args.block_length}, Step {len(sampling_traj[0])}, temperature {args.temperature}"
            #     + "=" * 25
            #     + "\n"
            # )
            # with open("output.txt", "a") as f:
            #     f.write(output)
            # print(output)
            # print("=" * 25)
            # print(out_str[0])
            # for s in sampling_traj[0]:
            #     print(s)
            #     print("-" * 25)
            if args.log_steps:
                print(f"nfe = {len(sampling_traj[0])}")
                nfe_list.append(len(sampling_traj[0]))
    if args.log_steps:
        print(f"avg nfe = {sum(nfe_list) / len(nfe_list)}")
        print(f"variance = {torch.var(torch.tensor(nfe_list).float())}")
    if gen_time_list:
        avg_gen_time = sum(gen_time_list) / len(gen_time_list)
        print(
            f"average generate() time: {avg_gen_time:.3f}s over {len(gen_time_list)} runs"
        )
        print(f"last generate() time: {gen_time_list[-1]:.3f}s")
    if gen_time_list and nfe_list:
        total_time = sum(gen_time_list)
        total_nfe = sum(nfe_list)
        if total_nfe > 0:
            print(f"average time per nfe: {total_time / total_nfe:.6f}s")
            print(f"last run time per nfe: {gen_time_list[-1] / nfe_list[-1]:.6f}s")


if __name__ == "__main__":
    main()
