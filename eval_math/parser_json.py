import glob
import json
import os
import re
from collections import defaultdict

import tiktoken
from parser_helper import is_equiv, last_boxed_only_string, remove_boxed


def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


def parse_gsm_answers(json_path=None, json_data=None):
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
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if (
                    boxed_content
                    and boxed_content != "..."
                    and not re.match(r"^\.+$", boxed_content)
                ):
                    try:
                        parsed_answer = float(boxed_content)
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(
                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
            )
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
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
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    # Try to import math_verify for hybrid approach
    try:
        from math_verify import parse as math_parse
        from math_verify import verify as math_verify

        MATH_VERIFY_AVAILABLE = True
    except ModuleNotFoundError:
        MATH_VERIFY_AVAILABLE = False

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        try:
            parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
        except Exception:
            parsed_answer = None

        if not parsed_answer:
            answer_match = re.search(
                r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
            )
            if answer_match:
                parsed_answer = answer_match.group(1).strip()

        # Hybrid approach: Use math_verify if available and both expressions parse,
        # otherwise fall back to is_equiv (which handles more LaTeX variations)
        is_correct = False
        if parsed_answer is not None and ground_truth is not None:
            # First try is_equiv
            is_correct_equiv = is_equiv(parsed_answer, ground_truth)

            # Then try math_verify if available
            is_correct_math_verify = False
            math_verify_attempted = False
            if MATH_VERIFY_AVAILABLE:
                try:
                    gt_parsed = math_parse(ground_truth)
                    ans_parsed = math_parse(parsed_answer)
                    # Only use math_verify if both parsed successfully (non-empty)
                    if gt_parsed and ans_parsed:
                        is_correct_math_verify = math_verify(gt_parsed, ans_parsed)
                        math_verify_attempted = True
                except Exception:
                    pass

            # Use math_verify result if it could parse both, else use is_equiv
            if MATH_VERIFY_AVAILABLE and math_verify_attempted:
                is_correct = is_correct_math_verify
            else:
                is_correct = is_correct_equiv

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
            }
        )

    return total_correct, total_processed, processed_items, total_effective_tokens


def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None


def aggregate_results(directory=".", save_detailed=True):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
            "total_effective_tokens": 0,
        }
    )

    for json_file in json_files:
        if "scratch" in json_file or "llada_math_seq160" in json_file:
            continue
        filename = os.path.basename(json_file)
        setup_name = extract_setup_name(filename)

        if setup_name:
            # print(f"Processing {filename}...")
            if "gsm" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = (
                    parse_gsm_answers(json_path=json_file)
                )
            elif "math" in setup_name:
                correct, processed, detailed_results, total_effective_tokens = (
                    parse_math_answers(json_path=json_file)
                )

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            setups[setup_name]["total_effective_tokens"] += total_effective_tokens
            setups[setup_name]["questions"].extend(detailed_results)

    # Calculate final accuracy and save results
    print("\n===== AGGREGATED RESULTS =====")
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100
            if results["processed"] > 0
            else 0
        )
        results["avg_effective_tokens"] = results["total_effective_tokens"] / len(
            results["questions"]
        )
        print(
            f"{setup}: {results['correct']}/{results['processed']} correct ({results['accuracy']:.2f}% accuracy), avg effective tokens: {results['avg_effective_tokens']:.2f}"
        )

        if save_detailed:
            output_filename = f"{setup}_aggregated_results.json"
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)
            # print(f"Saved detailed results to {output_filename}")


if __name__ == "__main__":
    aggregate_results()
