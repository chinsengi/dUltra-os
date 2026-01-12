from __future__ import annotations

from pathlib import Path


def model_dir() -> Path:
    """Absolute path to the `model/` source directory, independent of CWD."""
    return Path(__file__).resolve().parent


def llada_dir() -> Path:
    return model_dir() / "llada"


def llada_config_json_path() -> Path:
    return llada_dir() / "config.json"


def lladou_config_dir() -> Path:
    return llada_dir() / "lladou_config"
