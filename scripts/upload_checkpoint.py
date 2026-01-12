#!/usr/bin/env python3
"""
Upload a Diffu-GRPO checkpoint to the Hugging Face Hub without training state files.

Example:
    python scripts/upload_checkpoint.py \
        --checkpoint-dir diffu_grpo/checkpoints/.../checkpoint-8000 \
        --repo-id your-name/lladou-grpo \
        --branch main

    python scripts/upload_checkpoint.py \
        --checkpoint-dir "." \
        --repo-id sengi/dUltra-math-b32\
        --branch main
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi, HfFolder

DEFAULT_IGNORE = (
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
    "trainer_state.json",
    "rng_state.pth",
    "global_step*",
    "*.ckpt",
    "checkpoint*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a checkpoint directory to the Hugging Face Hub while omitting "
            "training-state artifacts such as optimizer or scheduler payloads."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Path to the local checkpoint directory to upload.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository id on the Hugging Face Hub (e.g. user/model-name).",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Hub branch to push to (default: main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hub access token. Defaults to the token stored by `huggingface-cli login`.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Subdirectory inside the repo where files will be uploaded.",
    )
    parser.add_argument(
        "--extra-ignore",
        nargs="*",
        default=(),
        help="Additional glob patterns to ignore (space separated).",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo on the hub if it does not exist yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir: Path = args.checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    token = args.token or HfFolder.get_token()
    if token is None:
        raise RuntimeError(
            "No Hugging Face token found. Provide --token or run `huggingface-cli login`."
        )

    ignore_patterns: Sequence[str] = tuple(DEFAULT_IGNORE) + tuple(args.extra_ignore)

    api = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, exist_ok=True, token=token)

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.branch,
        path_in_repo=args.path_in_repo,
        ignore_patterns=list(ignore_patterns),
    )


if __name__ == "__main__":
    main()
