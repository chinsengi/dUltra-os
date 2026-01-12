import textwrap

from datasets import Dataset, load_dataset
from reward_func import extract_hash_answer

PROMPT_MODE_THINKING = "thinking"
PROMPT_MODE_NON_THINKING = "non-thinking"

# Constants for prompts
MATH_SYSTEM_PROMPT_NON_THINKING = r"""
You are a math expert.
Follow these rules:
1) Carefully read the question and identify what is being asked.
2) Solve the problem step by step, showing the intermediate reasoning.
3) Present the final answer inside \boxed{}.
"""

MATH_SYSTEM_PROMPT_THINKING = r"""
You are a math expert. To answer the user's question, first write your full reasoning inside <think></think>. After thinking, provide only the final result inside an <answer></answer> block, wrapping the answer in \boxed{}.
"""

CODING_SYSTEM_PROMPT_THINKING = r"""
You are an expert Python programmer. Carefully read the task description and unit tests,and write a correct and efficient python implementation. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and a ```python``` code block, respectively, i.e., <think> reasoning process here </think>
```python
answer code here
```.
"""

CODING_SYSTEM_PROMPT_NON_THINKING = r"""
You are an expert Python programmer. Carefully read and think about the task provided and write correct, efficient Python code that solves the problem and passes the hidden tests. Wrap the final code inside a ```python``` code block.
"""


def _build_prompt(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": textwrap.dedent(system_prompt).strip(),
        },
        {
            "role": "user",
            "content": user_content.strip(),
        },
    ]


def _get_math_system_prompt(prompt_mode: str) -> str:
    prompt_mode = prompt_mode.lower()
    if prompt_mode == PROMPT_MODE_THINKING:
        return MATH_SYSTEM_PROMPT_THINKING
    if prompt_mode == PROMPT_MODE_NON_THINKING:
        return MATH_SYSTEM_PROMPT_NON_THINKING
    raise ValueError(
        f"Unknown prompt mode '{prompt_mode}'. Expected '{PROMPT_MODE_THINKING}' or '{PROMPT_MODE_NON_THINKING}'."
    )


def _get_coding_system_prompt(prompt_mode: str) -> str:
    prompt_mode = prompt_mode.lower()
    if prompt_mode == PROMPT_MODE_THINKING:
        return CODING_SYSTEM_PROMPT_THINKING
    if prompt_mode == PROMPT_MODE_NON_THINKING:
        return CODING_SYSTEM_PROMPT_NON_THINKING
    raise ValueError(
        f"Unknown prompt mode '{prompt_mode}'. Expected '{PROMPT_MODE_THINKING}' or '{PROMPT_MODE_NON_THINKING}'."
    )


def get_gsm8k_questions(
    split="train", prompt_mode: str = PROMPT_MODE_THINKING
) -> Dataset:
    system_prompt = _get_math_system_prompt(prompt_mode)
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": _build_prompt(system_prompt, str(x["question"])),
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def _format_math_prompt(question: str) -> str:
    return textwrap.dedent(str(question)).strip()


def get_math500_questions(
    split="train", prompt_mode: str = PROMPT_MODE_THINKING
) -> Dataset:
    system_prompt = _get_math_system_prompt(prompt_mode)
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {
            "prompt": _build_prompt(system_prompt, _format_math_prompt(x["problem"])),
            "answer": x["solution"],
        }
    )
    return data


def _format_apps_prompt(question: str, input_output: str, starter_code: str) -> str:
    sections = [textwrap.dedent(str(question)).strip()]
    if starter_code:
        sections.append(f"Starter code:\n{starter_code.strip()}")
    if input_output:
        sections.append(f"Sample input/output:\n{input_output.strip()}")
    return "\n\n".join([s for s in sections if s])


def get_dapo17_data(
    split: str = "train", prompt_mode: str = PROMPT_MODE_THINKING
) -> Dataset:
    """Load and format the DAPO Math 17k dataset for GRPO training."""

    system_prompt = _get_math_system_prompt(prompt_mode)
    data = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split=split)
    question_key = "prompt"
    # Column that carries the final boxed answer/label
    final_answer_key = "solution"

    def _map_row(row: dict) -> dict:
        prompt_text = _format_math_prompt(str(row[question_key]).strip())
        mapped: dict[str, object] = {
            "prompt": _build_prompt(system_prompt, prompt_text),
        }

        if final_answer_key is not None:
            mapped["answer"] = str(row[final_answer_key]).strip()

        return mapped

    return data.map(_map_row)


def get_apps_questions(
    split: str = "train",
    difficulty=["introductory"],
    prompt_mode: str = PROMPT_MODE_THINKING,
) -> Dataset:
    """Load and format the APPS programming dataset for GRPO training."""

    normalized_prompt_mode = prompt_mode.lower()
    system_prompt = _get_coding_system_prompt(normalized_prompt_mode)
    data = load_dataset("codeparrot/apps", split=split)  # type: ignore

    def _map_row(row: dict) -> dict:
        prompt_text = _format_apps_prompt(
            row.get("question", ""),
            # row.get("input_output", ""),
            None,  # no input/output provided during training, since app has little or no hidden tests
            row.get("starter_code", ""),
        )

        instruction_suffix = (
            "Make sure to include all your thinking steps."
            if normalized_prompt_mode == PROMPT_MODE_THINKING
            else ""
        )
        user_prompt = textwrap.dedent(
            f"Write a solution to the following problem:\n{prompt_text}. {instruction_suffix}"
        ).strip()

        return {
            "prompt": _build_prompt(system_prompt, user_prompt),
            "problem_id": row.get("problem_id"),
            "input_output": row.get("input_output", ""),
            "solutions": row.get("solutions", []),
            "starter_code": row.get("starter_code", ""),
        }

    data = data.filter(
        lambda x: x["difficulty"] in difficulty and "fn_name" in x["input_output"]
    )
    mapped_data = data.map(_map_row)
    return mapped_data
