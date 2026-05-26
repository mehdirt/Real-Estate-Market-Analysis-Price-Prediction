"""Paths and YAML configuration loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# src/divar/config.py -> project root is two levels up from src/
PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", SRC_ROOT.parent))

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DIVAR_CSV = Path(os.getenv("DIVAR_CSV", RAW_DATA_DIR / "Divar.csv"))
IRAN_CITY_CSV = Path(os.getenv("IRAN_CITY_CSV", RAW_DATA_DIR / "iran_city_classification.csv"))


def get_divar_path() -> Path:
    """
    Resolve the main Divar CSV path from ``DIVAR_CSV`` or ``data/raw/Divar.csv``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = DIVAR_CSV.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Divar dataset not found at {path}. "
            "Place Divar.csv under data/raw/ or set DIVAR_CSV. See data/README.md."
        )
    return path


@lru_cache
def load_config(name: str) -> dict[str, Any]:
    """
    Load a YAML config from ``configs/{name}.yaml``.

    Parameters
    ----------
    name
        Config stem without extension (e.g. ``"price"``, ``"credit"``, ``"mlflow"``).
    """
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
