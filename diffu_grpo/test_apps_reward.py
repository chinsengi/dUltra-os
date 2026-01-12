from __future__ import annotations

import contextlib
import io
import json
import sys
import textwrap
import types
import unittest
import warnings
from typing import List, Tuple


def _install_dependency_stubs() -> None:
    """Provide minimal stubs so reward_func imports even without optional deps."""
    if "latex2sympy2_extended" not in sys.modules:
        latex_stub = types.ModuleType("latex2sympy2_extended")
        latex_stub.NormalizationConfig = object
        sys.modules["latex2sympy2_extended"] = latex_stub

    if "math_verify" not in sys.modules:
        math_verify_stub = types.ModuleType("math_verify")
        math_verify_stub.LatexExtractionConfig = object
        math_verify_stub.parse = lambda *args, **kwargs: ()
        math_verify_stub.verify = lambda *args, **kwargs: False
        sys.modules["math_verify"] = math_verify_stub

    if "sympy" not in sys.modules:
        sympy_stub = types.ModuleType("sympy")
        sympy_stub.EX = object()
        sys.modules["sympy"] = sympy_stub

    if "openai" not in sys.modules:

        class _DummyAPIStatusError(Exception):
            status_code = None

        class _DummyOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def with_options(self, **kwargs):
                return self

            class responses:
                @staticmethod
                def create(*args, **kwargs):
                    return types.SimpleNamespace(output_text="")

        openai_stub = types.ModuleType("openai")
        openai_stub.APIStatusError = _DummyAPIStatusError
        openai_stub.OpenAI = _DummyOpenAI
        sys.modules["openai"] = openai_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        utils_stub = types.ModuleType("transformers.utils")

        class _DummyLogger:
            def info(self, *args, **kwargs):
                pass

            def debug(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        class _LoggingNS:
            @staticmethod
            def set_verbosity_info():
                pass

            @staticmethod
            def get_logger(name):
                return _DummyLogger()

        utils_stub.logging = _LoggingNS
        transformers_stub.utils = utils_stub
        sys.modules["transformers"] = transformers_stub
        sys.modules["transformers.utils"] = utils_stub


_install_dependency_stubs()

from reward_func import apps_reward_func  # noqa: E402

from diffu_grpo.data_utils import get_apps_questions  # noqa: E402

COMMON_IMPORTS = """\
import math
import sys
from typing import *
import collections
from collections import *
"""


def with_common_imports(code: str) -> str:
    """Prepend common imports to help APPS reference solutions execute."""
    return COMMON_IMPORTS.rstrip() + "\n\n" + code.lstrip()


def _format_completion(code: str) -> List[dict]:
    """Wrap candidate code inside a python code fence, matching training prompts."""
    cleaned = textwrap.dedent(code).strip()
    return [{"content": (f"```python\n{cleaned}\n```")}]


def _build_io_block(fn_name: str) -> str:
    """Create a minimal APPS-style input_output payload for the given function."""
    return json.dumps(
        {
            "fn_name": fn_name,
            "inputs": [[1, 2], [3, -1]],
            "outputs": [3, 2],
        }
    )


def _build_prompts(count: int, prompt_text: str) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": prompt_text}]] * count


def _parse_apps_tests(io_block: str) -> Tuple[str, list[tuple[object, object]]]:
    """Parse APPS-style input_output JSON into fn_name and test pairs."""
    if not io_block:
        return "", []
    try:
        io_dict = json.loads(io_block)
    except Exception:
        return "", []

    fn_name = io_dict.get("fn_name") or ""
    inputs = io_dict.get("inputs", []) or []
    outputs = io_dict.get("outputs", []) or []

    parsed_inputs = []
    for raw in inputs:
        try:
            parsed_inputs.append(json.loads(raw))
        except Exception:
            parsed_inputs.append(raw)

    tests: list[tuple[object, object]] = list(zip(parsed_inputs, outputs))
    return str(fn_name), tests


def _select_solution_code(raw_solutions: object, fn_name: str | None = None) -> str:
    """Choose a non-empty solution snippet from the APPS sample."""

    def _clean(code: str) -> str:
        return textwrap.dedent(code).strip()

    if isinstance(raw_solutions, list):
        for candidate in raw_solutions:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            cleaned = _clean(candidate)
            if fn_name and f"def {fn_name}" not in cleaned:
                continue
            return cleaned
        return ""

    if isinstance(raw_solutions, str):
        stripped = raw_solutions.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return _select_solution_code(parsed)
        except Exception:
            pass
        cleaned = _clean(stripped)
        if fn_name and f"def {fn_name}" not in cleaned:
            return ""
        return cleaned

    return ""


def _prepare_solution_code(solution_code: str, fn_name: str) -> str:
    """Make APPS reference solutions executable under the reward harness."""
    # solution_code = _rewrite_is_literal_comparisons(solution_code)
    code = with_common_imports(solution_code)

    # Many APPS references are written as LeetCode-style classes; expose the method.
    if "class Solution" in solution_code:
        wrapper = textwrap.dedent(
            f"""
            def {fn_name}(*args, **kwargs):
                return Solution().{fn_name}(*args, **kwargs)
            """
        ).strip()
        code = code.rstrip() + "\n\n" + wrapper + "\n"

    return code


@contextlib.contextmanager
def _suppress_output():
    """
    Silence stdout/stderr and collect SyntaxWarnings from reference solutions during tests.
    """
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(buf_out),
        contextlib.redirect_stderr(buf_err),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("default")
        yield caught


def _iter_solution_strings(raw_solutions: object) -> list[str]:
    """Return a flat list of solution strings from raw APPS sample field."""
    if isinstance(raw_solutions, list):
        return [s for s in raw_solutions if isinstance(s, str)]
    if isinstance(raw_solutions, str):
        try:
            parsed = json.loads(raw_solutions)
            if isinstance(parsed, list):
                return [s for s in parsed if isinstance(s, str)]
        except Exception:
            pass
        return [raw_solutions]
    return []


def _find_passing_solution(
    raw_solutions: object, fn_name: str, sample: dict
) -> tuple[str, list[str]]:
    """Use the reward function itself to find a reference solution that scores 1."""
    warnings_found: list[str] = []
    normalized_io = sample.get("input_output", "")
    prompt = [
        {
            "role": "user",
            "content": str(sample.get("question", "")).strip(),
        }
    ]

    candidate = _iter_solution_strings(raw_solutions)[0]
    cleaned = textwrap.dedent(candidate).strip()
    if not cleaned:
        return "", warnings_found
    if fn_name and f"{fn_name}" not in cleaned and "class Solution" not in cleaned:
        return "", warnings_found

    prepared = _prepare_solution_code(cleaned, fn_name)
    rewards = apps_reward_func(
        [prompt],
        [_format_completion(prepared)],
        [normalized_io],
        problem_id=[sample.get("problem_id")],
    )

    if abs(rewards[0] - 1.0) < 1e-9:
        return prepared, warnings_found
    else:
        print(sample.get("problem_id"), rewards[0])

    return "", warnings_found


class AppsRewardFuncTests(unittest.TestCase):
    def test_apps_reward_func_assigns_expected_scores(self) -> None:
        fn_name = "add_numbers"
        prompts = _build_prompts(
            3, f"Write {fn_name}(a, b) that returns the sum of two integers."
        )
        input_output = [_build_io_block(fn_name)] * 3

        completions = [
            _format_completion(
                f"""
                def {fn_name}(a, b):
                    return a + b
                """
            ),
            _format_completion(
                f"""
                def {fn_name}(a, b):
                    return a - b
                """
            ),
            _format_completion(
                f"""
                def {fn_name}(a, b):
                    raise RuntimeError("boom")
                """
            ),
        ]

        rewards = apps_reward_func(
            prompts,
            completions,
            input_output,
            problem_id=["correct", "wrong", "error"],
        )

        self.assertEqual(rewards, [1.0, 0.3, 0.15])

    def test_apps_reward_func_on_200_apps_samples(self) -> None:
        target_samples = 100
        dataset = get_apps_questions("train")

        prompts: list[list[dict[str, str]]] = []
        completions: list[list[dict[str, str]]] = []
        problem_ids: list[object] = []
        warning_messages: list[str] = []

        for sample in dataset:
            fn_name, tests = _parse_apps_tests(sample.get("input_output", ""))
            if not fn_name or not tests:
                continue

            solution_code, collected_warnings = _find_passing_solution(
                sample.get("solutions", ""), fn_name, sample
            )
            warning_messages.extend(collected_warnings)
            # if not solution_code:
            #     breakpoint()
            #     _find_passing_solution( sample.get("solutions", ""), fn_name, sample )

            completions.append(_format_completion(solution_code))
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": str(sample.get("question", "")).strip(),
                    }
                ]
            )
            problem_ids.append(sample.get("problem_id"))

            if len(completions) >= target_samples:
                break

        if len(completions) < target_samples:
            self.skipTest(
                f"Only gathered {len(completions)} valid APPS samples (needed {target_samples})."
            )

        self.assertFalse(
            warning_messages,
            f"Reference solutions emitted warnings during selection: {warning_messages}",
        )

        self.assertEqual(len(completions), target_samples)


if __name__ == "__main__":
    unittest.main()
