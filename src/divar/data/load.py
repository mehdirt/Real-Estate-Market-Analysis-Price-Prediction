"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from divar.config import IRAN_CITY_CSV, get_divar_path


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def load_divar(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the main Divar CSV.

    Parameters
    ----------
    path
        Optional override. Defaults to ``DIVAR_CSV`` / ``data/raw/Divar.csv``.
    """
    csv_path = Path(path) if path is not None else get_divar_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Divar dataset not found at {csv_path}")
    return _read_csv(csv_path)


def load_iran_city_classification(path: Path | str | None = None) -> pd.DataFrame:
    """Load optional Iran city classification CSV used in statistical analysis."""
    csv_path = Path(path) if path is not None else IRAN_CITY_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"City classification file not found at {csv_path}")
    return pd.read_csv(csv_path)
