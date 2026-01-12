from __future__ import annotations

import builtins
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import textwrap
from typing import Optional

import numpy as np
from latex2sympy2_extended import NormalizationConfig
from math500_utils import (
    boxed_in_answer,
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)
from math_verify import LatexExtractionConfig
from math_verify import parse as math_parse
from math_verify import verify as math_verify_equiv
from openai import APIStatusError, OpenAI
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

OPENAI_MATH_MODEL = os.getenv("OPENAI_MATH_VERIFIER_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_MATH_VERIFIER_TIMEOUT", "30"))
APPS_TIMEOUT = 10.0


def _prepare_candidate_code(code: str, fn_name: str) -> str:
    """
    Apply the same adjustments we use for APPS reference solutions:
    - strip/dedent
    - if the solution is a LeetCode-style class, expose the method.
    Imports are left untouched so the model must provide them.
    """
    cleaned = textwrap.dedent(code).strip()
    if not cleaned:
        return ""

    prepared = cleaned

    # Many APPS references are written as LeetCode-style classes; expose the method.
    if "class Solution" in cleaned and fn_name:
        wrapper = textwrap.dedent(
            f"""
            def {fn_name}(*args, **kwargs):
                return Solution().{fn_name}(*args, **kwargs)
            """
        ).strip()
        prepared = prepared.rstrip() + "\n\n" + wrapper + "\n"

    return prepared


def _strip_code_fence(text: str) -> str:
    """Remove Markdown code fences from a string if present."""

    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("\n")
        if len(parts) >= 2:
            # Drop opening fence (and optional language) and closing fence
            parts = parts[1:]
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            stripped = "\n".join(parts)
    return stripped.strip()


def _extract_apps_solution(response: str) -> str:
    """
    Extract APPS solution code from model output.

    Accepts plain code, python fences, generic fences, or optional <answer> wrappers.
    """

    body = response
    if "<answer>" in body:
        body = body.split("<answer>", 1)[-1]
        body = body.split("</answer>", 1)[0]

    if "```python\n" in body:
        body = body.split("```python\n", 1)[-1]
    elif "```python" in body:
        body = body.split("```python", 1)[-1]
    elif "```" in body:
        body = body.split("```", 1)[-1]
    body = body.split("```", 1)[0]

    return textwrap.dedent(body).strip()


def _code_fence_score(text: str) -> float:
    """
    Score how well a response follows a python code fence format.

    +0.25 for having any fence, +0.25 for a ```python fence, +0.25 for a closing fence,
    +0.25 if there is no extra text after fences. Small penalty for stray text.
    """

    if not text:
        return 0.0

    score = 0.0
    has_fence = "```" in text
    has_python = "```python" in text.lower()
    fence_count = text.count("```")

    if has_fence:
        score += 0.25
    if has_python:
        score += 0.25
    if fence_count >= 2:
        score += 0.25

    stripped = text.strip()
    last_fence = stripped.rfind("```")
    suffix = stripped[last_fence + 3 :].strip() if last_fence != -1 else ""

    # Allow explanatory text before the first fence; only penalize trailing junk.
    if not suffix:
        score += 0.25
    else:
        score -= min(len(suffix), 200) * 0.001

    return max(0.0, min(1.0, score))


_OPENAI_VERIFIER_AVAILABLE = True
_OPENAI_CLIENT: Optional["OpenAI"] = None


def _math_verify_compare(reference: str, candidate: str) -> Optional[bool]:
    """Attempt to compare two expressions with math_verify."""

    if not reference or not candidate:
        return None

    try:
        ref_parsed = math_parse(
            reference,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("math_verify failed to parse reference '%s': %s", reference, exc)
        return None

    if not ref_parsed:
        return None

    try:
        cand_parsed = math_parse(
            candidate,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("math_verify failed to parse candidate '%s': %s", candidate, exc)
        return None

    if not cand_parsed:
        return None

    try:
        return bool(math_verify_equiv(ref_parsed, cand_parsed))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "math_verify compare failed for reference '%s' and candidate '%s': %s",
            reference,
            candidate,
            exc,
        )
        return None


def _get_openai_client() -> "OpenAI":
    """Lazily construct an OpenAI client instance."""

    if OpenAI is None:
        logger.warning(
            "openai package not installed; skipping OpenAI math verification."
        )
        raise ValueError("openai package not installed")

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def _call_openai_math_verifier(
    problem: str, candidate_answer: str, reference_answer: str
) -> Optional[bool]:
    """
    Use OpenAI's Responses API to judge whether the candidate answer matches the reference.

    Returns True for correct, False for incorrect, and None if the request failed or the API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping OpenAI math verification.")
        raise ValueError("OPENAI_API_KEY not set")

    client = _get_openai_client()
    client = client.with_options(timeout=OPENAI_TIMEOUT)
    prompt = (
        "You are a meticulous mathematics grader. You are given the student's final boxed answer, "
        "and the authoritative reference answer. Decide if the student's answer matches the reference mathematically. "
        "Reply with a single word: 'correct' or 'incorrect'. If you are uncertain, answer 'incorrect'.\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Student answer:\n{candidate_answer}\n"
    )

    try:
        response = client.responses.create(
            model=OPENAI_MATH_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a strict grader for math contest answers. "
                                "Respond with only 'correct' or 'incorrect'."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ],
            temperature=0,
            max_output_tokens=16,
        )

        if hasattr(response, "output_text"):
            message_text = response.output_text or ""
        else:
            data = response.model_dump()
            pieces = []
            for item in data.get("output", []) or []:
                for content in item.get("content", []) or []:
                    text = content.get("text")
                    if text:
                        pieces.append(text)
            message_text = " ".join(pieces)

        message = message_text.strip().lower()
        if "correct" in message and "incorrect" not in message:
            return True
        if "incorrect" in message:
            return False
    except APIStatusError as exc:  # pragma: no cover - network interaction
        status_code = getattr(exc, "status_code", None)
        logger.warning(
            "OpenAI math verification failed with HTTP %s: %s",
            status_code if status_code is not None else "unknown",
            exc,
        )
        raise exc
    return None


def extract_xml_answer(text: str) -> str:
    """
    Extract the answer content from XML-tagged text.

    This function parses text that contains XML tags and extracts the content
    between <answer> and </answer> tags. It takes the last occurrence of the
    answer tag in case there are multiple.

    Args:
        text (str): The input text containing XML answer tags.

    Returns:
        str: The extracted answer content, stripped of whitespace.

    Example:
        >>> extract_xml_answer("Question: 2+2? <answer>4</answer>")
        '4'
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """
    Extract the answer content from text marked with #### delimiter.

    This function parses text that uses "####" as a delimiter to separate
    the reasoning/explanation from the final answer, commonly used in
    mathematical problem solving.

    Args:
        text (str): The input text containing #### delimiter.

    Returns:
        str | None: The extracted answer content after ####, or None if
                   no #### delimiter is found.

    Example:
        >>> extract_hash_answer("2 + 2 = 4 #### 4")
        '4'
        >>> extract_hash_answer("No answer here")
        None
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Reward function that evaluates correctness by comparing extracted answers to ground truth.

    This function extracts answers from XML-tagged completions and compares them to the
    expected answers, providing a binary reward (2.0 for correct, 0.0 for incorrect).
    It also prints detailed debugging information showing the prompt, ground truth,
    response, and extracted answer for the first sample.

    Args:
        prompts: List of conversation prompts, where each prompt is a list of message dicts.
        completions: List of model completions, where each completion is a list containing
                    a dict with "content" key.
        answer: List of ground truth answers to compare against.
        step: Current training step (for logging).
        run_name: Name of the training run (for logging).
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: List of reward scores (2.0 for correct answers, 0.0 for incorrect).

    Note:
        This function assumes completions use XML <answer> tags and performs exact string matching.
    """
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    logger.info(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that encourages integer answers.

    This function provides a reward when the extracted answer from XML tags
    consists entirely of digits (i.e., is an integer). This is useful for
    training models to produce numerical answers in appropriate contexts.

    Args:
        completions: List of model completions, where each completion is a list
                    containing a dict with "content" key.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of reward scores (0.5 for integer answers, 0.0 otherwise).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that enforces strict XML formatting with reasoning and answer tags.

    This function rewards completions that follow the exact expected format:
    <think>
    [think content]
    </think>
    <answer>
    [answer content]
    </answer>

    The pattern requires precise newlines and structure, making it very strict.

    Args:
        completions: List of model completions, where each completion is a list
                    containing a dict with "content" key.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of reward scores (0.5 for correctly formatted responses, 0.0 otherwise).
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def _parse_apps_io(io_block: str) -> list[tuple[list, list]]:
    if not io_block:
        return []

    tests: list[tuple[str, str]] = []
    sys.set_int_max_str_digits(0)
    io_dict = json.loads(io_block) if isinstance(io_block, str) else io_block
    if "fn_name" in io_dict:
        inputs = io_dict.get("inputs", [])
        outputs = io_dict.get("outputs", [])
        for inp, outp in zip(inputs, outputs):
            tests.append((inp, outp))
    else:
        raise ValueError("No 'fn_name' found in input_output block.")

    return io_dict["fn_name"], tests


def _normalize_output(value: object, seen: set[int] | None = None) -> object:
    """Convert numpy arrays/tuples to Python lists recursively for safe comparison."""
    if seen is None:
        seen = set()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        if obj_id in seen:
            return "<recursion>"
        seen.add(obj_id)
        normalized = [_normalize_output(v, seen) for v in value]
        seen.discard(obj_id)
        return normalized

    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return "<recursion>"
        seen.add(obj_id)
        normalized = {k: _normalize_output(v, seen) for k, v in value.items()}
        seen.discard(obj_id)
        return normalized

    return value


def _apps_worker(
    candidate_code: str, fn_name: str, test_input: object, result_queue
) -> None:
    """Execute APPS candidate function with a single test input inside a subprocess."""
    env: dict[str, object] = {"__builtins__": builtins.__dict__}
    sys.setrecursionlimit(5000)
    try:
        exec(candidate_code, env)
        candidate = env.get(fn_name)
        if candidate is None or not callable(candidate):
            raise ValueError(f"Function {fn_name!r} not defined or not callable.")

        args = ()
        kwargs: dict[str, object] = {}
        if isinstance(test_input, dict) and (
            "args" in test_input or "kwargs" in test_input
        ):
            raw_args = test_input.get("args", [])
            if isinstance(raw_args, (list, tuple)):
                args = tuple(raw_args)
            elif raw_args is None:
                args = ()
            else:
                args = (raw_args,)
            raw_kwargs = test_input.get("kwargs", {})
            if isinstance(raw_kwargs, dict):
                kwargs = raw_kwargs
        elif isinstance(test_input, (list, tuple)):
            args = tuple(test_input)
        elif test_input is None:
            args = ()
        else:
            args = (test_input,)

        result = candidate(*args, **kwargs)
        result_queue.put((True, result, None))
    except Exception as exc:  # pragma: no cover
        try:
            result_queue.put((False, None, repr(exc)))
        except Exception:
            pass


def _run_apps_with_timeout(
    candidate_code: str,
    fn_name: str,
    test_input: object,
    timeout: float,
) -> tuple[bool, object | None, str | None]:
    """Run APPS candidate function with a wall-clock timeout."""
    ctx = mp.get_context()
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_apps_worker,
        args=(candidate_code, fn_name, test_input, result_queue),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        result: tuple[bool, object | None, str | None] = (
            False,
            None,
            "Timed out while running candidate solution.",
        )
    else:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = (
                False,
                None,
                "Candidate process finished without returning a result.",
            )

    try:
        result_queue.close()
        result_queue.join_thread()
    except Exception:
        pass

    passed, output, error_message = result
    return passed, output, error_message


def apps_reward_func(
    prompts,
    completions,
    input_output,
    problem_id=None,
    sampling_traj=None,
    **kwargs,
) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    rewards: list[float] = []
    for idx, response in enumerate(responses):
        candidate_code = _extract_apps_solution(response)
        prompt_text = (
            prompts[idx][-2].get("content", "")
            + "\n\n"
            + prompts[idx][-1].get("content", "")
        )

        io_block = ""
        if isinstance(input_output, (list, tuple)) and idx < len(input_output):
            io_block = input_output[idx] or ""
        elif isinstance(input_output, str):
            io_block = input_output

        fn_name, tests = _parse_apps_io(io_block)
        prepared_code = _prepare_candidate_code(candidate_code, fn_name)

        if not prepared_code.strip() or not fn_name or not tests:
            rewards.append(0.0)
            continue

        outcome = "pass"
        error_message = ""
        passed_tests = 0
        total_tests = len(tests)
        for test_input, expected_output in tests:
            expected_value: object = (
                expected_output[0]
                if isinstance(expected_output, list)
                else expected_output
            )

            ok, actual_output, exec_error = _run_apps_with_timeout(
                prepared_code, fn_name, test_input, APPS_TIMEOUT
            )
            if not ok:
                outcome = "runtime_error"
                if not error_message:
                    error_message = exec_error
                break

            actual_output = _normalize_output(actual_output)
            expected_value = _normalize_output(expected_value)
            if actual_output != expected_value:
                outcome = "wrong_output"
                if not error_message:
                    try:
                        error_message = f"Output mismatch. Expected {expected_value!r}, got {actual_output!r}."
                    except Exception as e:
                        error_message = (
                            f"Output mismatch. (Error formatting message: {e})"
                        )
                continue

            passed_tests += 1

        pass_rate = passed_tests / total_tests if total_tests else 0.0
        reward = pass_rate * 0.0 if pass_rate < 1.0 else 1.0
        if outcome == "runtime_error":
            error_msg = error_message.lower()
            if "syntax" in error_msg or "indentation" in error_msg:
                reward = 0.0
            else:
                reward = -0.05
        rewards.append(reward)

        task_label = (
            problem_id[idx]
            if isinstance(problem_id, (list, tuple)) and idx < len(problem_id)
            else problem_id
        )
        steps_taken = 0
        if isinstance(sampling_traj, (list, tuple)) and idx < len(sampling_traj):
            steps_taken = sum(1 for step in sampling_traj[idx] if len(step) > 0)
        status_text = (
            "PASS" if pass_rate == 1.0 else "PARTIAL" if pass_rate > 0 else "FAIL"
        )
        logger.info(
            "-" * 20
            + f"\nPrompt:\n{prompt_text}\n"
            + "-" * 20
            + f"\nTask ID:\n{task_label if task_label is not None else 'N/A'}\n"
            + "-" * 20
            + f"\nResponse:\n{response}\n"
            + "-" * 20
            + f"\nExtracted Code:\n{candidate_code}\n"
            + "-" * 20
            + f"\nResult: {status_text}\n"
            + f"Tests passed: {passed_tests}/{total_tests}\n"
            + (
                "-" * 20 + f"\nError: {error_message}\n"
                if pass_rate < 1.0 and error_message
                else ""
            )
            + f"Sampling steps: {steps_taken}"
        )
        logger.info("✅" if pass_rate == 1.0 else "❌")

    return rewards


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that encourages inclusion of <think> blocks.

    This function rewards completions that contain both opening and closing
    <think> tags anywhere in the response. It does not require an <answer>
    tag, matching the thinking prompt format.

    Args:
        completions: List of model completions, where each completion is a list
                    containing a dict with "content" key.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of reward scores (0.5 for responses with <think> tags, 0.0 otherwise).
    """
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if ("<think>" in r and "</think>" in r) else 0.0 for r in responses]


def count_xml(text) -> float:
    """
    Count XML formatting elements and penalize extra content after answer tags.

    This function evaluates how well a text follows XML formatting conventions,
    rewarding proper tag usage and penalizing extra content after the closing
    answer tag. It's designed to encourage clean, structured responses.

    Scoring breakdown:
    - 0.125 points for each correctly formatted tag pair
    - Penalty of 0.001 per character of extra content after </answer>

    Args:
        text (str): The text to evaluate for XML formatting.

    Returns:
        float: Score between 0.0 and 0.5 based on formatting quality.
    """
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function based on XML formatting quality.

    This function evaluates the quality of XML formatting in model completions
    by applying the count_xml scoring function to each completion. It rewards
    proper use of <think> and <answer> tags while penalizing extra content
    after the answer tag.

    Args:
        completions: List of model completions, where each completion is a list
                    containing a dict with "content" key.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of XML formatting scores (0.0 to 0.5 range).
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def codefence_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that encourages python code fences in model outputs.

    Scoring favors responses that:
    - include a code fence (bonus for ```python),
    - close the fence,
    - avoid extra prose outside the fenced code.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [_code_fence_score(c) for c in contents]


def reward_len(completions, **kwargs):
    """
    Reward function that penalizes response length.

    This function provides negative rewards proportional to the length of the response,
    encouraging the model to be more concise. It's primarily used for sanity checking
    and understanding reward function behavior.

    The current implementation returns negative length values, so longer responses
    receive more negative (worse) rewards. Alternative implementations could target
    specific optimal lengths.

    Args:
        completions: List of model completions, where each completion is a list
                    containing a dict with "content" key.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of negative reward scores based on response length.
    """
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def correctness_reward_func_math(
    prompts, completions, answer, sampling_traj, **kwargs
) -> list[float]:
    """
    Reward function for mathematical correctness using boxed answer format.

    This function evaluates mathematical problem solutions by extracting answers
    from LaTeX boxed environments and comparing them to ground truth using
    mathematical equivalence checking. It provides binary rewards (2.0 for correct,
    0.0 for incorrect) and includes detailed debugging output.

    The function uses specialized utilities to handle mathematical notation and
    equivalence checking that accounts for different valid representations of
    the same mathematical answer.

    Args:
        prompts: List of conversation prompts, where each prompt is a list of message dicts.
        completions: List of model completions, where each completion is a list containing
                    a dict with "content" key.
        answer: List of ground truth answers (may contain LaTeX boxed notation).
        step: Current training step (for logging).
        run_name: Name of the training run (for logging).
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: List of reward scores (2.0 for mathematically equivalent answers, 0.0 otherwise).

    Note:
        This function assumes mathematical answers are in LaTeX boxed format and uses
        specialized equivalence checking for mathematical expressions.
    """
    # boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except Exception:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    n_samples = len(completions)
    dataset_name = kwargs.get("dataset_name")
    allowed_openai_datasets = {}
    if isinstance(dataset_name, (list, tuple)):
        dataset_name = dataset_name[0] if dataset_name else None
    use_openai_for_sample = (
        isinstance(dataset_name, str)
        and dataset_name.lower() in allowed_openai_datasets
    )

    rewards: list[float] = []
    for i in range(n_samples):
        q = prompts[i][-1]["content"]
        response = responses[i]
        extracted_response = extracted_responses[i]
        verifier_result = None
        if extracted_responses[i].strip() and use_openai_for_sample:
            try:
                verifier_result = _call_openai_math_verifier(
                    q, extracted_responses[i], answer[i]
                )
            except ValueError:
                verifier_result = None
        if verifier_result is None:
            verifier_source = "symbolic"
            math_verify_result = _math_verify_compare(answer[i], extracted_responses[i])
            if math_verify_result is not None:
                is_correct = math_verify_result
                verifier_source = "math_verify"
            else:
                is_correct = is_equiv(extracted_responses[i], answer[i])
        else:
            is_correct = verifier_result
            verifier_source = "openai"
        logger.info(
            "-" * 20
            + f"\n{RED}Question:{RESET}\n{q}\n"
            + "-" * 20
            + f"\n{GREEN}Ground Truth:{RESET}\n{answer[i]}\n"
            + "-" * 20
            + f"\n{BLUE}Response:{RESET}\n{response}\n"
            + "-" * 20
            + f"\n{YELLOW}Extracted:{RESET}\n{extracted_response}\n"
            + "-" * 20
            + f"\n{YELLOW}Verifier:{RESET} {verifier_source}\n"
            + "-" * 20
            + f"\n{YELLOW}Sampling steps:{RESET}\n{sum([1 for step in sampling_traj[i] if len(step) > 0])}\n",
        )
        logger.info("✅" if is_correct else "❌")
        rewards.append(2.0 if is_correct else 0.0)

    return rewards


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    """
    Reward function for LaTeX boxed answer format compliance.

    This function rewards completions that properly use LaTeX boxed notation
    for answers in mathematical problem solving. It provides a scaled reward
    (0.5) when answers are correctly boxed, encouraging proper mathematical
    formatting.

    Args:
        prompts: List of conversation prompts, where each prompt is a list of message dicts.
        completions: List of model completions, where each completion is a list containing
                    a dict with "content" key.
        answer: List of ground truth answers (unused in format checking).
        step: Current training step (passed to boxed_in_answer utility).
        run_name: Name of the training run (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        list[float]: List of format compliance scores (0.5 for properly boxed answers, 0.0 otherwise).
    """
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


if __name__ == "__main__":
    test_str = """
     <think>
1. We need to generate a sequence based on the given pattern.
2. If `n` is 0, return an empty list.
3. If `n` is positive, initialize the sequence with 0.
4. If `n` is negative, initialize the sequence with 0.
   For negative `n`, the terms should be negative.
5. Calculate a term where each term is the sum of numbers from 0 to `n`.
6. Append each term to a list and return the list.
</think>

```python
def sum_of_n(n):
    if n == 0:
        return []
    elif n > 0:
        sequence = [0]
        for i in range(1, n + 1):
            sequence.append(i)
        return sequence
    else:
        sequence = [0]
        for i in range(1, abs(n) + 1):
            sequence.append(-i)
        return sequence
```
    """
    print(_code_fence_score(test_str))
