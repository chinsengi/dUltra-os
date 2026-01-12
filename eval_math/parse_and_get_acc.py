import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None

from transformers import AutoTokenizer

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify

    MATH_VERIFY_AVAILABLE = True
except ModuleNotFoundError:
    MATH_VERIFY_AVAILABLE = False

import sys

from parser_helper import is_equiv, last_boxed_only_string, remove_boxed
from parsers import Parser


def count_effective_tokens(text, tokenizer=None):
    """
    Count effective tokens in text.
    If tokenizer is provided, uses the model's tokenizer.
    Otherwise falls back to tiktoken cl100k_base.
    """
    if not text:
        return 0

    # Remove special tokens
    text = text.replace("<|endoftext|>", "").replace("<|eot_id|>", "")

    if tokenizer is not None:
        # Use model's tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    elif tiktoken is not None:
        # Fallback to tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        return len(tokens)
    else:
        return 0


def parse_gsm_answers(json_path=None, json_data=None, tokenizer=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = Parser.to_float(item.get("ground_truth"))
        raw_generation = item.get("generations", "")
        question = item.get("question", "")
        nfe = item.get("nfe")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation, tokenizer)
        total_effective_tokens += effective_tokens

        parsed_answer = Parser.extract_answer_gsm8k_fallback(raw_generation)

        is_correct = (
            parsed_answer is not None
            and ground_truth is not None
            and abs(parsed_answer - ground_truth) < 1e-5
        )
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
                "nfe": nfe,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def parse_math_answers(json_path=None, json_data=None, tokenizer=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_correct_math_verify = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")
        nfe = item.get("nfe")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation, tokenizer)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_answer = last_boxed_only_string(raw_generation)
        parsed_answer = None
        if boxed_answer:
            try:
                parsed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                parsed_answer = boxed_answer

        if not parsed_answer:
            answer_match = re.search(
                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
            )
            if answer_match:
                parsed_answer = answer_match.group(1).strip()

        # Compute is_equiv (string-based comparison)
        is_correct_equiv = False
        if parsed_answer is not None and ground_truth is not None:
            is_correct_equiv = is_equiv(parsed_answer, ground_truth)

        # Compute math_verify (symbolic math comparison)
        is_correct_math_verify = False
        math_verify_attempted = False
        if (
            MATH_VERIFY_AVAILABLE
            and ground_truth is not None
            and parsed_answer is not None
        ):
            try:
                # Parse both the ground truth and the parsed answer
                gt_parsed = math_parse("$" + ground_truth + "$")
                ans_parsed = math_parse("$" + parsed_answer + "$")

                # Only use math_verify if both parsed successfully (non-empty)
                if gt_parsed and ans_parsed:
                    is_correct_math_verify = math_verify(gt_parsed, ans_parsed)
                    math_verify_attempted = True
            except Exception:
                pass

        # Hybrid approach: Use math_verify if it could parse both expressions,
        # otherwise fall back to is_equiv (which handles more LaTeX variations)
        if MATH_VERIFY_AVAILABLE and math_verify_attempted:
            is_correct = is_correct_math_verify
        else:
            is_correct = is_correct_equiv

        if is_correct:
            # print(f"Correct: GT='{ground_truth}' | Ans='{parsed_answer}'")
            # if ground_truth =='8n^2 + 4n + 1':
            #     breakpoint()
            total_correct += 1
        if is_correct_math_verify:
            total_correct_math_verify += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "is_correct_equiv": is_correct_equiv,
                "is_correct_math_verify": is_correct_math_verify,
                "effective_tokens": effective_tokens,
                "nfe": nfe,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


def extract_setup_name(filename):
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def detect_dataset(setup_name: str) -> str:
    lowered = setup_name.lower()
    # Check dataset prefix first (more specific patterns)
    # This ensures "math_..." isn't misclassified due to "gsm8k" in model name
    if lowered.startswith("math500") or lowered.startswith("math_"):
        return "math"
    if lowered.startswith("gsm8k") or lowered.startswith("gsm_"):
        return "gsm8k"

    # Fallback to substring matching for backward compatibility
    if "math500" in lowered or "math" in lowered:
        return "math"
    if "gsm8k" in lowered or "gsm" in lowered:
        return "gsm8k"
    raise ValueError(f"Unknown dataset for setup '{setup_name}'")


PARSERS = {
    "gsm8k": parse_gsm_answers,
    "math": parse_math_answers,
}


def iter_generation_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    # Search recursively for generation files in subdirectories
    for json_file in sorted(path.glob("**/*_generations.json")):
        if json_file.is_file():
            yield json_file


def aggregate_results(
    path: Path, dataset_filter: Optional[str] = None
) -> Dict[str, Dict]:
    setups: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "questions": [],
            "total_effective_tokens": 0,
            "dataset": "",
            "total_nfe": 0.0,
            "total_wall_time": 0.0,
        }
    )

    dataset_filter = dataset_filter.lower() if dataset_filter else None
    tokenizer_cache = {}  # Cache tokenizers by model_path

    for json_file in iter_generation_files(path):
        setup_name = extract_setup_name(json_file.name)
        if setup_name is None:
            continue
        dataset = detect_dataset(setup_name)
        if dataset_filter and dataset_filter not in setup_name.lower():
            continue

        with open(json_file, "r") as file:
            data = json.load(file)

        # Load tokenizer from model_path
        tokenizer = None
        model_path = data.get("model_path")
        if model_path:
            if model_path not in tokenizer_cache:
                try:
                    tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to load tokenizer from {model_path}: {e}",
                        file=sys.stderr,
                    )
                    tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
                        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
                    )
            tokenizer = tokenizer_cache[model_path]

        parser_fn = PARSERS[dataset]
        correct, processed, detailed_results, total_tokens = parser_fn(
            json_data=data, tokenizer=tokenizer
        )

        metrics_info = data.get("metrics", {})
        wall_time_value = metrics_info.get("total_wall_time")
        if wall_time_value is None:
            wall_time_value = metrics_info.get("wall_time")

        setup_metrics = setups[setup_name]
        setup_metrics["correct"] += correct
        setup_metrics["processed"] += processed
        setup_metrics["total_effective_tokens"] += total_tokens
        setup_metrics["questions"].extend(detailed_results)
        setup_metrics["dataset"] = dataset
        setup_metrics["total_wall_time"] += wall_time_value or 0.0
        setup_metrics["total_nfe"] += sum(
            nfe
            for nfe in (item.get("nfe") for item in detailed_results)
            if isinstance(nfe, (int, float))
        )

    for setup, results in setups.items():
        processed = results["processed"]
        if processed:
            results["accuracy"] = 100 * results["correct"] / processed
            results["avg_effective_tokens"] = (
                results["total_effective_tokens"] / processed
            )
            results["avg_nfe"] = (
                results["total_nfe"] / processed if results["total_nfe"] else 0.0
            )
        else:
            results["accuracy"] = 0.0
            results["avg_effective_tokens"] = 0.0
            results["avg_nfe"] = 0.0

        # Compute token_per_forward
        if results["avg_nfe"] > 0 and results["avg_effective_tokens"] > 0:
            results["token_per_forward"] = (
                results["avg_effective_tokens"] / results["avg_nfe"]
            )
        else:
            results["token_per_forward"] = 0.0

        wall_time = results.get("total_wall_time", 0.0)
        if wall_time:
            results["tokens_per_sec"] = results["total_effective_tokens"] / wall_time
        else:
            results["tokens_per_sec"] = 0.0

    return dict(sorted(setups.items()))


def print_summary(results: Dict[str, Dict]) -> None:
    if not results:
        print("No generation files found.")
        return

    header_format = "{:<45} {:>12} {:>12} {:>18} {:>10} {:>14} {:>14}"
    print(
        header_format.format(
            "Setup",
            "Dataset",
            "Accuracy",
            "Avg Eff. Tokens",
            "Avg NFE",
            "Tok/Fwd",
            "Tokens/s",
        )
    )
    print("-" * 135)
    row_format = "{:<45} {:>12} {:>10.2f}% {:>18.2f} {:>10.2f} {:>14.2f} {:>14.2f}"
    for setup, metrics in results.items():
        print(
            row_format.format(
                setup,
                metrics["dataset"],
                metrics["accuracy"],
                metrics["avg_effective_tokens"],
                metrics.get("avg_nfe", 0.0),
                metrics.get("token_per_forward", 0.0),
                metrics.get("tokens_per_sec", 0.0),
            )
        )
    print("=" * 135)


def build_save_payload(
    results: Dict[str, Dict], include_questions: bool = False
) -> Dict[str, Dict]:
    payload: Dict[str, Dict] = {}
    for setup, metrics in results.items():
        entry = {
            "dataset": metrics.get("dataset", ""),
            "correct": metrics.get("correct", 0),
            "processed": metrics.get("processed", 0),
            "accuracy": metrics.get("accuracy", 0.0),
            "avg_effective_tokens": metrics.get("avg_effective_tokens", 0.0),
            "avg_nfe": metrics.get("avg_nfe", 0.0),
            "token_per_forward": metrics.get("token_per_forward", 0.0),
            "tokens_per_sec": metrics.get("tokens_per_sec", 0.0),
            "total_wall_time": metrics.get("total_wall_time", 0.0),
            "total_effective_tokens": metrics.get("total_effective_tokens", 0),
        }
        if include_questions:
            entry["questions"] = metrics.get("questions", [])
        payload[setup] = entry
    return payload


def default_output_path(input_path: Path, dataset_filter: Optional[str]) -> Path:
    out_dir = input_path if input_path.is_dir() else input_path.parent
    suffix = f"_{dataset_filter.lower()}" if dataset_filter else ""
    return out_dir / f"acc_summary{suffix}.json"


def save_summary(
    results: Dict[str, Dict],
    input_path: Path,
    dataset_filter: Optional[str],
    output_path: Optional[Path] = None,
    include_questions: bool = False,
) -> Path:
    output_path = output_path or default_output_path(input_path, dataset_filter)
    payload = {
        "input_path": str(input_path),
        "dataset_filter": dataset_filter,
        "setups": build_save_payload(results, include_questions=include_questions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def print_detailed_results(
    results: Dict[str, Dict],
    max_questions: Optional[int] = None,
    setups_filter: Optional[List[str]] = None,
) -> None:
    if not results:
        return

    normalized_filter = [s.lower() for s in setups_filter] if setups_filter else None

    for setup, metrics in results.items():
        questions = metrics.get("questions") or []

        if normalized_filter and setup.lower() not in normalized_filter:
            continue

        if not questions:
            continue

        print(
            f"\n=== {setup} ({metrics.get('dataset', 'unknown')}) "
            f"- showing {min(len(questions), max_questions or len(questions))} "
            f"of {len(questions)} questions ==="
        )
        items = questions[:max_questions] if max_questions else questions
        for idx, item in enumerate(items, 1):
            ground_truth = item.get("ground_truth")
            parsed = item.get("extracted_answer")
            status = item.get("is_correct")
            raw_generation = item.get("raw_generation")
            _ = item.get("question")
            eff_tokens = item.get("effective_tokens")
            nfe = item.get("nfe")

            correctness_str = (
                "unknown" if status is None else ("correct" if status else "wrong")
            )

            if nfe is not None:
                print(f"[{idx}] {correctness_str}, tokens={eff_tokens}, nfe={nfe}")
            else:
                print(f"[{idx}] {correctness_str}, tokens={eff_tokens}")
            # if question:
            #     print(f"  Q: {question}")
            print(f"  GT: {ground_truth}")
            if parsed is not None:
                print(f"  Parsed: {parsed}")
            else:
                print("  Parsed: <missing>")
                if raw_generation is not None:
                    print(f"  Raw: {raw_generation}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse saved generations and compute accuracy for reasoning tasks."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("eval_baselines"),
        help="Path to a generation JSON file or a directory containing *_generations.json files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset substring to filter files (e.g., math500).",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print question-level comparisons (ground truth vs parsed answers).",
    )
    parser.add_argument(
        "--detail-limit",
        type=int,
        default=None,
        help="Maximum number of questions to show per setup when --show-details is set.",
    )
    parser.add_argument(
        "--detail-setups",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of exact setup names to display when using --show-details.",
    )
    parser.add_argument(
        "--dont-save",
        action="store_true",
        help="Do not save the aggregated results JSON next to the input path.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to write the aggregated results JSON (defaults to input directory).",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Include question-level details in the saved JSON (can be large).",
    )
    parser.add_argument(
        "--per-subdir",
        action="store_true",
        help="Process each immediate subdirectory separately (useful for dir of dirs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # Check if we should process subdirectories separately
    if cli_args.per_subdir and cli_args.input_path.is_dir():
        # Get immediate subdirectories that contain generation files
        subdirs = []
        for item in sorted(cli_args.input_path.iterdir()):
            if item.is_dir():
                # Check if this subdirectory has generation files
                gen_files = list(item.glob("**/*_generations.json"))
                if gen_files:
                    subdirs.append(item)

        if not subdirs:
            print(
                f"No subdirectories with generation files found in {cli_args.input_path}"
            )
            exit(1)

        print(f"Found {len(subdirs)} subdirectories to process\n")

        # Process each subdirectory
        for subdir in subdirs:
            print(f"\n{'=' * 135}")
            print(f"Processing: {subdir.name}")
            print(f"{'=' * 135}")

            summary = aggregate_results(subdir, cli_args.dataset)
            print_summary(summary)

            if not cli_args.dont_save:
                # Save to the subdirectory
                save_path = cli_args.save_path
                if save_path is None:
                    save_path = default_output_path(subdir, cli_args.dataset)
                saved_to = save_summary(
                    summary,
                    input_path=subdir,
                    dataset_filter=cli_args.dataset,
                    output_path=save_path,
                    include_questions=cli_args.save_details,
                )
                print(f"Saved results to {saved_to}")

            if cli_args.show_details:
                print_detailed_results(
                    summary, cli_args.detail_limit, cli_args.detail_setups
                )
    else:
        # Original behavior: process single directory or file
        summary = aggregate_results(cli_args.input_path, cli_args.dataset)
        print_summary(summary)
        if not cli_args.dont_save:
            saved_to = save_summary(
                summary,
                input_path=cli_args.input_path,
                dataset_filter=cli_args.dataset,
                output_path=cli_args.save_path,
                include_questions=cli_args.save_details,
            )
            print(f"Saved results to {saved_to}")
        if cli_args.show_details:
            print_detailed_results(
                summary, cli_args.detail_limit, cli_args.detail_setups
            )
