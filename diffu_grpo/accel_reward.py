import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import selective_log_softmax

TEACHER_MODELS = {
    # "math": "Qwen/Qwen3-8B", #
    "math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "coding": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

_teacher_cache: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def _get_teacher(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load (or reuse) the verifier teacher model and tokenizer."""
    if model_name not in _teacher_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        _teacher_cache[model_name] = (tokenizer, model)
    return _teacher_cache[model_name]


def _select_teacher_domain(kwargs: dict) -> str:
    """Pick math vs coding teacher based on available metadata."""
    coding_keys = ("entry_point", "test", "imports", "starter_code", "input_output")
    for key in coding_keys:
        values = kwargs.get(key)
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            if any(v is not None for v in values):
                return "coding"
        else:
            return "coding"
    return "math"


@torch.no_grad()
def verifier_reward(prompts, completions, **kwargs):
    teacher_domain = _select_teacher_domain(kwargs)
    tokenizer, model = _get_teacher(TEACHER_MODELS[teacher_domain])
    student_logprobs = kwargs.get("student_logprob")

    rewards = []
    for prompt, completion in zip(prompts, completions):
        example = {"prompt": prompt, "completion": completion}
        example = maybe_apply_chat_template(example, tokenizer)
        completion_ids = tokenizer(example["completion"])["input_ids"]
        completion_length = len(completion_ids)
        input_ids = tokenizer(
            example["prompt"] + example["completion"],
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        )["input_ids"].to(model.device)
        # Get log probabilities from model
        with torch.inference_mode():
            logits = model(input_ids).logits[:, -completion_length - 1 : -1, :]
        completion_ids = torch.tensor(completion_ids, device=logits.device).view(1, -1)
        logprobs = selective_log_softmax(logits, completion_ids)

        if student_logprobs is not None:
            student_logprob = student_logprobs[len(rewards)]
            student_logprob = student_logprob[-completion_length:]
            student_logprob = student_logprob.view(1, -1)
            logprobs -= student_logprob

        # higher precision when casting to float first
        total_log_prob = logprobs.float().sum().item()
        avg_log_prob = total_log_prob / max(completion_length, 1)
        rewards.append(avg_log_prob)
    return rewards


def gen_step_reward(sampling_traj, mode="sum", **kwargs):
    # sqrt sum
    if mode == "sqrt_sum":
        rewards = [
            -sum([math.sqrt(len(step)) for step in traj])
            / sum([len(step) for step in traj])
            for traj in sampling_traj
        ]
    elif mode == "sqrt":
        rewards = [
            -sum([math.sqrt(len(step)) for step in traj]) for traj in sampling_traj
        ]
    elif mode == "sum":
        rewards = [
            -sum([1 for step in traj if len(step) > 0]) / 50 for traj in sampling_traj
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # sqrt
    return rewards


def print_completion_reward(
    prompts, completions, answer, sampling_traj, **kwargs
) -> list[float]:
    """
    placeholder reward function that prints the completions
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    responses = [completion[0]["content"] for completion in completions]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    for i in range(len(completions)):
        q = prompts[i][-1]["content"]
        response = responses[i].replace("\n", "\\n")
        print(
            "-" * 20,
            f"\n{RED}Prompt:{RESET}\n{q}\n",
            "-" * 20,
            f"\n{GREEN}Ground Truth:{RESET}\n{answer[i]}\n",
            "-" * 20,
            f"\n{BLUE}Response:{RESET}\n{response}\n",
            "-" * 20,
            f"\n{YELLOW}Sample Traj length:{RESET}\n{sum([1 for step in sampling_traj[i] if len(step) > 0])}\n",
            # f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
        )
    return [0.0] * len(completions)
