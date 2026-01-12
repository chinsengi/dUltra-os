#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend

# Pattern to match directory names like:
# sengi_dUltra-coding-b128_humaneval_instruct_block128_mode_training_factor0.5
RE_RUN_DIR = re.compile(
    r"^(?P<model>.+)_(?P<task>humaneval_instruct|mbpp_instruct)_block(?P<block>\d+)_mode_(?P<mode>[A-Za-z0-9_-]+)_factor(?P<factor>[0-9.]+)$"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_result_jsons(run_dir: Path) -> List[Path]:
    """Find all results JSON files in the run directory."""
    return list(run_dir.rglob("results_*.json"))


def find_speed_log(run_dir: Path) -> Path | None:
    """Find the speed_log.txt file in the grpo_save directory."""
    speed_log_path = run_dir / "grpo_save" / "speed_log.txt"
    if speed_log_path.is_file():
        return speed_log_path
    return None


def parse_avg_nfe_per_sample(log_path: Path, task: str) -> float | None:
    """
    Parse average NFE per sample from speed_log.txt.
    Looks for lines like: "Total NFE is 6312" and calculates average NFE per sample.

    Args:
        log_path: Path to speed_log.txt
        task: Task name (humaneval_instruct or mbpp_instruct) to determine sample count

    Returns:
        Average NFE per sample across all seeds
    """
    # Determine number of samples based on task
    if "humaneval" in task:
        n_samples = 164
    elif "mbpp" in task:
        n_samples = 500
    else:
        print(
            f"Warning: Unknown task {task}, cannot determine sample count",
            file=sys.stderr,
        )
        return None

    try:
        with log_path.open("r", encoding="utf-8") as f:
            content = f.read()
            # Find all "Total NFE is X" lines (should be 5 for 5 seeds)
            matches = re.findall(r"Total NFE is (\d+)", content)
            if matches:
                # Calculate NFE per sample for each seed, then average across seeds
                nfe_per_sample_per_seed = [int(match) / n_samples for match in matches]
                return sum(nfe_per_sample_per_seed) / len(nfe_per_sample_per_seed)
    except Exception as e:
        print(f"Warning: Failed to parse NFE from {log_path}: {e}", file=sys.stderr)
    return None


def parse_avg_effective_tokens_per_sample(log_path: Path, task: str) -> float | None:
    """
    Parse average effective tokens per sample from speed_log.txt.
    Looks for lines like: "Total effective tokens is 123456" and calculates average per sample.

    Args:
        log_path: Path to speed_log.txt
        task: Task name (humaneval_instruct or mbpp_instruct) to determine sample count

    Returns:
        Average effective tokens per sample across all seeds
    """
    # Determine number of samples based on task
    if "humaneval" in task:
        n_samples = 164
    elif "mbpp" in task:
        n_samples = 500
    else:
        print(
            f"Warning: Unknown task {task}, cannot determine sample count",
            file=sys.stderr,
        )
        return None

    try:
        with log_path.open("r", encoding="utf-8") as f:
            content = f.read()
            # Find all "Total effective tokens is X" lines (should be 5 for 5 seeds)
            matches = re.findall(r"Total number of tokens generated: (\d+)", content)
            if matches:
                # Calculate effective tokens per sample for each seed, then average across seeds
                tokens_per_sample_per_seed = [
                    int(match) / n_samples for match in matches
                ]
                return sum(tokens_per_sample_per_seed) / len(tokens_per_sample_per_seed)
    except Exception as e:
        print(
            f"Warning: Failed to parse effective tokens from {log_path}: {e}",
            file=sys.stderr,
        )
    return None


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_lm_eval_result(data: Dict, task: str) -> Tuple[float, int]:
    """
    Parse lm_eval result JSON to extract accuracy.
    Returns (accuracy_percentage, n_samples).
    """
    results = data.get("results", {})
    task_results = results.get(task, {})

    # Find the pass@1 metric (could be pass@1,create_test or pass_at_1,strip_code_fence)
    accuracy = None
    for key in task_results:
        if key.startswith("pass") and not key.endswith("_stderr"):
            accuracy = task_results[key]
            break

    if accuracy is None:
        raise ValueError(f"No pass@1 metric found in results for task {task}")

    # Convert to percentage
    accuracy_pct = accuracy * 100.0

    # Get number of samples
    n_samples_dict = data.get("n-samples", {}).get(task, {})
    n_samples = n_samples_dict.get("effective", n_samples_dict.get("original", 0))

    return accuracy_pct, n_samples


def parse_multiple_results(json_files: List[Path], task: str) -> Tuple[float, int]:
    """
    Parse multiple results JSON files and average the accuracy.
    Returns (average_accuracy_percentage, n_samples_from_first_file).
    """
    if not json_files:
        raise ValueError("No JSON files provided")

    accuracies = []
    n_samples = 0

    for json_file in json_files:
        try:
            data = read_json(json_file)
            accuracy_pct, n_samp = parse_lm_eval_result(data, task)
            accuracies.append(accuracy_pct)
            if n_samples == 0:  # Use n_samples from first file
                n_samples = n_samp
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
            continue

    if not accuracies:
        raise ValueError(f"No valid results found for task {task}")

    # Return average accuracy
    return sum(accuracies) / len(accuracies), n_samples


@dataclass(frozen=True)
class RunRecord:
    task: str
    model: str
    block_length: int
    mode: str
    factor: float
    accuracy: float
    n_samples: int
    avg_nfe: float | None = None
    avg_effective_tokens: float | None = None
    token_per_forward: float | None = None


def parse_run_dir(run_dir: Path) -> RunRecord | None:
    m = RE_RUN_DIR.match(run_dir.name)
    if not m:
        return None

    model = m.group("model")
    task = m.group("task")
    block_length = int(m.group("block"))
    mode = m.group("mode")
    factor = float(m.group("factor"))

    result_jsons = find_result_jsons(run_dir)
    if not result_jsons:
        print(f"Warning: No result JSON found in {run_dir.name}", file=sys.stderr)
        return None

    try:
        accuracy, n_samples = parse_multiple_results(result_jsons, task)
    except Exception as e:
        print(
            f"Warning: Failed to parse results in {run_dir.name}: {e}", file=sys.stderr
        )
        return None

    # Parse NFE and effective tokens per sample from speed_log.txt in grpo_save directory
    avg_nfe = None
    avg_effective_tokens = None
    speed_log = find_speed_log(run_dir)
    if speed_log:
        avg_nfe = parse_avg_nfe_per_sample(speed_log, task)
        avg_effective_tokens = parse_avg_effective_tokens_per_sample(speed_log, task)

    # Compute token_per_forward using effective tokens
    token_per_forward = None
    if avg_effective_tokens is not None and avg_nfe is not None and avg_nfe > 0:
        token_per_forward = avg_effective_tokens / avg_nfe

    return RunRecord(
        task=task,
        model=model,
        block_length=block_length,
        mode=mode,
        factor=factor,
        accuracy=accuracy,
        n_samples=n_samples,
        avg_nfe=avg_nfe,
        avg_effective_tokens=avg_effective_tokens,
        token_per_forward=token_per_forward,
    )


def write_csv(records: List[RunRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "model",
                "block_length",
                "mode",
                "factor",
                "accuracy",
                "n_samples",
                "avg_nfe_per_sample",
                "avg_effective_tokens_per_sample",
                "token_per_forward",
            ]
        )
        for r in sorted(
            records,
            key=lambda x: (x.task, x.model, x.block_length, x.mode, x.factor),
        ):
            w.writerow(
                [
                    r.task,
                    r.model,
                    r.block_length,
                    r.mode,
                    f"{r.factor:.6g}",
                    f"{r.accuracy:.6g}",
                    r.n_samples,
                    f"{r.avg_nfe:.6g}" if r.avg_nfe is not None else "",
                    f"{r.avg_effective_tokens:.6g}"
                    if r.avg_effective_tokens is not None
                    else "",
                    f"{r.token_per_forward:.6g}"
                    if r.token_per_forward is not None
                    else "",
                ]
            )


def render_plot(
    task: str,
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

    # Format task name for title
    task_display = task.replace("_instruct", "").replace("_", " ").title()
    if task_display.lower() == "mbpp":
        task_display = "MBPP"
    elif task_display.lower() == "humaneval":
        task_display = "HumanEval"
    fig.suptitle(f"{task_display}", fontsize=18, fontweight="bold")

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

    # Configure right y-axis (NFE per sample)
    if have_nfe:
        ax2.set_ylabel(
            "Avg NFE per Sample", fontsize=14, fontweight="bold", color="#ff7f0e"
        )
        ax2.tick_params(axis="y", labelsize=12, labelcolor="#ff7f0e")
        ax2.invert_yaxis()  # Invert so low NFE is at top
    else:
        ax2.set_ylabel("")
        ax2.set_yticks([])

    # Legend - only show accuracy lines (NFE uses same colors)
    # ax1.legend(loc='best', framealpha=0.7, edgecolor='none', fontsize=12)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_task(records: List[RunRecord], output_dir: Path, task: str) -> Path:
    task_records = [r for r in records if r.task == task]
    if not task_records:
        raise ValueError(f"No records found for task={task}")

    series: Dict[Tuple[str, int, str], List[RunRecord]] = {}
    for r in task_records:
        key = (r.model, r.block_length, r.mode)
        series.setdefault(key, []).append(r)

    out_path = output_dir / f"factor_sweep_{task}.pdf"
    render_plot(task=task, series=series, out_path=out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot pass@1 accuracy vs factor from lm_eval factor_eval_results to PDF using matplotlib."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root() / "eval_coding" / "factor_eval_results",
        help="Directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots (defaults to input-dir).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="humaneval_instruct,mbpp_instruct",
        help="Comma-separated list of tasks to plot (e.g. humaneval_instruct,mbpp_instruct).",
    )
    args = parser.parse_args()

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

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, output_dir / "factor_sweep_summary.csv")

    saved: List[Path] = []
    for t in tasks:
        try:
            saved.append(plot_task(records, output_dir=output_dir, task=t))
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    print("Saved:")
    print(f"- {output_dir / 'factor_sweep_summary.csv'}")
    for p in saved:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
