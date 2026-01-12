#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

matplotlib.use("Agg")  # Use non-interactive backend

RE_RUN_DIR = re.compile(
    r"^(?P<dataset>gsm8k|math500|math)_(?P<model>.+)_block(?P<block>\d+)_mode_(?P<mode>[A-Za-z0-9_-]+)_factor(?P<factor>[0-9.]+)$"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_eval_to_syspath() -> None:
    eval_dir = repo_root() / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))


def iter_generation_files(run_dir: Path) -> Iterable[Path]:
    for p in sorted(run_dir.glob("*_generations.json")):
        if p.is_file():
            yield p


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_gsm8k_accuracy(data: Dict) -> Tuple[int, int]:
    from parsers import Parser

    correct = 0
    processed = 0
    for item in data.get("generations", []):
        processed += 1
        gt = Parser.to_float(item.get("ground_truth"))
        raw = item.get("generations", "")
        pred = Parser.extract_answer_gsm8k_fallback(raw)
        is_correct = (
            pred is not None and gt is not None and abs(float(pred) - float(gt)) < 1e-5
        )
        correct += int(is_correct)
    return correct, processed


def parse_math_accuracy(data: Dict) -> Tuple[int, int]:
    from parser_helper import is_equiv, last_boxed_only_string, remove_boxed

    # Try to import math_verify for hybrid approach
    try:
        from math_verify import parse as math_parse
        from math_verify import verify as math_verify

        MATH_VERIFY_AVAILABLE = True
    except ModuleNotFoundError:
        MATH_VERIFY_AVAILABLE = False

    correct = 0
    processed = 0
    for item in data.get("generations", []):
        processed += 1
        gt = item.get("ground_truth", "")
        raw = item.get("generations", "")

        parsed = None
        boxed = last_boxed_only_string(raw)
        if boxed:
            try:
                parsed = remove_boxed(boxed)
            except AssertionError:
                parsed = boxed
        if not parsed:
            m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
            if m:
                parsed = m.group(1).strip()

        # Hybrid approach: Use math_verify if available and both expressions parse,
        # otherwise fall back to is_equiv (which handles more LaTeX variations)
        is_correct = False
        if parsed is not None and gt is not None:
            # First try is_equiv
            is_correct_equiv = is_equiv(parsed, gt)

            # Then try math_verify if available
            is_correct_math_verify = False
            math_verify_attempted = False
            if MATH_VERIFY_AVAILABLE:
                try:
                    gt_parsed = math_parse("$" + gt + "$")
                    ans_parsed = math_parse("$" + parsed + "$")
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

        correct += int(is_correct)
    return correct, processed


def parse_avg_nfe(data: Dict) -> float | None:
    nfe_values: List[float] = []
    for item in data.get("generations", []):
        nfe = item.get("nfe")
        if isinstance(nfe, (int, float)):
            nfe_values.append(float(nfe))
    if nfe_values:
        return sum(nfe_values) / len(nfe_values)
    metrics = data.get("metrics", {})
    if isinstance(metrics, dict) and isinstance(metrics.get("avg_nfe"), (int, float)):
        return float(metrics["avg_nfe"])
    return None


def parse_gen_length(data: Dict) -> int | None:
    """Extract gen_length from the JSON data."""
    gen_length = data.get("gen_length")
    if isinstance(gen_length, (int, float)):
        return int(gen_length)
    return None


def count_effective_tokens(text: str, tokenizer) -> int:
    """Count effective tokens in text using the model's tokenizer."""
    if not text:
        return 0
    # Remove special tokens
    text = text.replace("<|endoftext|>", "").replace("<|eot_id|>", "")
    # Tokenize using model's tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def parse_effective_tokens(data: Dict, tokenizer) -> Tuple[float, int]:
    """
    Parse effective tokens from generation data.
    Returns (avg_effective_tokens, count).
    """
    total_tokens = 0
    count = 0
    for item in data.get("generations", []):
        raw_generation = item.get("generations", "")
        if raw_generation:
            total_tokens += count_effective_tokens(raw_generation, tokenizer)
            count += 1
    avg_tokens = (total_tokens / count) if count > 0 else 0.0
    return avg_tokens, count


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    model: str
    block_length: int
    mode: str
    factor: float
    accuracy: float
    avg_nfe: float | None
    avg_effective_tokens: float | None
    token_per_forward: float | None
    processed: int


def parse_run_dir(run_dir: Path) -> RunRecord | None:
    m = RE_RUN_DIR.match(run_dir.name)
    if not m:
        return None

    dataset = m.group("dataset")
    if dataset == "math500":
        dataset = "math"
    model = m.group("model")
    block_length = int(m.group("block"))
    mode = m.group("mode")
    factor = float(m.group("factor"))

    files = list(iter_generation_files(run_dir))
    if not files:
        return None

    total_correct = 0
    total_processed = 0
    total_nfe_weighted = 0.0
    total_nfe_count = 0
    total_effective_tokens_weighted = 0.0
    total_effective_tokens_count = 0
    model_path = None
    tokenizer = None

    for gen_file in files:
        data = read_json(gen_file)
        if dataset == "gsm8k":
            correct, processed = parse_gsm8k_accuracy(data)
        elif dataset == "math":
            correct, processed = parse_math_accuracy(data)
        else:
            return None

        total_correct += correct
        total_processed += processed

        # Load tokenizer from first file
        if tokenizer is None:
            model_path = data.get("model_path")
            if model_path:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to load tokenizer from {model_path}: {e}",
                        file=sys.stderr,
                    )

        # Parse effective tokens if tokenizer is available
        if tokenizer is not None:
            avg_eff_tokens, eff_count = parse_effective_tokens(data, tokenizer)
            if eff_count > 0:
                total_effective_tokens_weighted += avg_eff_tokens * eff_count
                total_effective_tokens_count += eff_count

        avg_nfe = parse_avg_nfe(data)
        if avg_nfe is not None and processed:
            total_nfe_weighted += avg_nfe * processed
            total_nfe_count += processed

    accuracy = 100.0 * total_correct / total_processed if total_processed else 0.0
    avg_nfe = (total_nfe_weighted / total_nfe_count) if total_nfe_count else None
    avg_effective_tokens = (
        (total_effective_tokens_weighted / total_effective_tokens_count)
        if total_effective_tokens_count
        else None
    )

    # Compute token_per_forward using effective tokens
    token_per_forward = None
    if avg_effective_tokens is not None and avg_nfe is not None and avg_nfe > 0:
        token_per_forward = avg_effective_tokens / avg_nfe

    return RunRecord(
        dataset=dataset,
        model=model,
        block_length=block_length,
        mode=mode,
        factor=factor,
        accuracy=accuracy,
        avg_nfe=avg_nfe,
        avg_effective_tokens=avg_effective_tokens,
        token_per_forward=token_per_forward,
        processed=total_processed,
    )


def write_csv(records: List[RunRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "model",
                "block_length",
                "mode",
                "factor",
                "accuracy",
                "avg_nfe",
                "avg_effective_tokens",
                "token_per_forward",
                "processed",
            ]
        )
        for r in sorted(
            records,
            key=lambda x: (x.dataset, x.model, x.block_length, x.mode, x.factor),
        ):
            w.writerow(
                [
                    r.dataset,
                    r.model,
                    r.block_length,
                    r.mode,
                    f"{r.factor:.6g}",
                    f"{r.accuracy:.6g}",
                    "" if r.avg_nfe is None else f"{r.avg_nfe:.6g}",
                    ""
                    if r.avg_effective_tokens is None
                    else f"{r.avg_effective_tokens:.6g}",
                    "" if r.token_per_forward is None else f"{r.token_per_forward:.6g}",
                    r.processed,
                ]
            )


def render_plot(
    dataset: str,
    series: Dict[Tuple[str, int, str], List[RunRecord]],
    out_path: Path,
) -> None:
    # Blue shades for accuracy
    acc_palette = [
        "#1f77b4",  # Medium blue
        "#3182bd",  # Steel blue
        "#6baed6",  # Light blue
        "#2171b5",  # Royal blue
        "#08519c",  # Dark blue
        "#4292c6",  # Sky blue
        "#5a9bd5",  # Cornflower blue
        "#0c5da5",  # Navy blue
        "#4169e1",  # Cobalt blue
        "#1e90ff",  # Dodger blue
    ]

    # Orange shades for NFE
    nfe_palette = [
        "#ff7f0e",  # Medium orange
        "#fd8d3c",  # Light orange
        "#e6550d",  # Dark orange
        "#ff9933",  # Bright orange
        "#d95f0e",  # Burnt orange
        "#ffa500",  # Pure orange
        "#ff8c00",  # Dark orange
        "#ff6600",  # Red-orange
        "#cc5500",  # Deep orange
        "#ff9966",  # Peach orange
    ]

    # Check if we have NFE data
    nfe_values = [
        float(p.avg_nfe)
        for pts in series.values()
        for p in pts
        if p.avg_nfe is not None
    ]
    have_nfe = bool(nfe_values)

    # Create figure with single plot and dual y-axes
    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{dataset.upper()}", fontsize=18, fontweight="bold")

    # Create second y-axis for NFE
    ax2 = ax1.twinx()

    # Plot each series
    for idx, ((model, block_length, mode), points) in enumerate(sorted(series.items())):
        acc_color = acc_palette[idx % len(acc_palette)]
        nfe_color = nfe_palette[idx % len(nfe_palette)]
        points = sorted(points, key=lambda x: x.factor)

        factors = [p.factor for p in points]
        accuracies = [p.accuracy for p in points]

        # Create label that includes block length to differentiate models
        model_short = model.split("_")[-1]
        label = f"{model_short}-b{block_length}"

        # Use solid line for default block length, dotted for b128
        linestyle = ":" if block_length == 128 else "-"

        # Plot accuracy on left y-axis
        ax1.plot(
            factors,
            accuracies,
            marker="o",
            linestyle=linestyle,
            color=acc_color,
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=label,
        )

        # Plot NFE on right y-axis
        nfe_pts = [p for p in points if p.avg_nfe is not None]
        if nfe_pts:
            nfe_factors = [p.factor for p in nfe_pts]
            nfe_vals = [float(p.avg_nfe) for p in nfe_pts]
            ax2.plot(
                nfe_factors,
                nfe_vals,
                marker="s",
                linestyle=linestyle,
                color=nfe_color,
                linewidth=2,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=1.5,
                alpha=0.7,
            )

    # Configure left y-axis (accuracy)
    ax1.set_xlabel("Factor", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold", color="#1f77b4")
    # ax1.grid(True, alpha=0.3, linestyle='-', linewidth=1)
    ax1.tick_params(axis="y", labelsize=12, labelcolor="#1f77b4")
    ax1.tick_params(axis="x", labelsize=12)

    # Auto-adjust accuracy y-axis range
    all_accuracies = [p.accuracy for pts in series.values() for p in pts]
    if all_accuracies:
        acc_min = min(all_accuracies)
        acc_max = max(all_accuracies)
        acc_range = acc_max - acc_min

        # Add padding (8% of range, or at least 2%)
        padding = max(acc_range * 0.08, 2.0)
        y_min = max(0, acc_min - padding)  # Don't go below 0
        y_max = min(100, acc_max + padding)  # Don't go above 100

        # If range is very small, ensure minimum visible range of 10%
        if y_max - y_min < 10:
            center = (y_min + y_max) / 2
            y_min = max(0, center - 5)
            y_max = min(100, center + 5)

        ax1.set_ylim(y_min, y_max)

    # Configure right y-axis (NFE)
    if have_nfe:
        ax2.set_ylabel("Avg NFE", fontsize=14, fontweight="bold", color="#ff7f0e")
        ax2.tick_params(axis="y", labelsize=12, labelcolor="#ff7f0e")
        ax2.invert_yaxis()  # Invert so low NFE is at top
    else:
        ax2.set_ylabel("")
        ax2.set_yticks([])

    # Legend - only show accuracy lines (NFE uses same colors)
    # ax1.legend(loc='upper left', framealpha=0.9, edgecolor='none', fontsize=12)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_dataset(records: List[RunRecord], output_dir: Path, dataset: str) -> Path:
    dataset_records = [r for r in records if r.dataset == dataset]
    if not dataset_records:
        raise ValueError(f"No records found for dataset={dataset}")

    series: Dict[Tuple[str, int, str], List[RunRecord]] = {}
    for r in dataset_records:
        key = (r.model, r.block_length, r.mode)
        series.setdefault(key, []).append(r)

    out_path = output_dir / f"factor_sweep_{dataset}.pdf"
    render_plot(dataset=dataset, series=series, out_path=out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot accuracy and NFE vs factor from eval/factor_eval_results runs to PDF using matplotlib."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root() / "eval_math" / "factor_eval_results",
        help="Directory containing per-run subdirectories like gsm8k_*_factor0.8.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots (defaults to input-dir).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k,math",
        help="Comma-separated list of datasets to plot (e.g. gsm8k,math).",
    )
    args = parser.parse_args()

    add_eval_to_syspath()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir or input_dir

    if not input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {input_dir}")

    records: List[RunRecord] = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        rec = parse_run_dir(child)
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit(f"No runnable sweep directories found under: {input_dir}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, output_dir / "factor_sweep_summary.csv")

    saved: List[Path] = []
    for d in datasets:
        saved.append(plot_dataset(records, output_dir=output_dir, dataset=d))

    print("Saved:")
    print(f"- {output_dir / 'factor_sweep_summary.csv'}")
    for p in saved:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
