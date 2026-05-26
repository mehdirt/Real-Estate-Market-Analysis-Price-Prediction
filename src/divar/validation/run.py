"""Run Pandera validation checks."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from divar.config import load_config
from divar.validation.schemas import (
    processed_credit_schema,
    processed_price_schema,
    raw_divar_schema,
)

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when Pandera schema validation fails."""


def validate_raw_divar(df: pd.DataFrame) -> None:
    """Validate raw Divar CSV column structure."""
    schema = raw_divar_schema()
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        raise DataValidationError(f"Raw data validation failed: {exc}") from exc
    logger.info("Raw data validation passed (%d rows, %d columns)", len(df), len(df.columns))


def validate_processed_price(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Validate prepared price train/val frames (target and building_size bounds).

    Raises ``DataValidationError`` on failure.
    """
    cfg = config or load_config("price")
    schema = processed_price_schema(cfg)
    for name, frame in ("train", train), ("val", val):
        try:
            schema.validate(frame, lazy=True)
        except Exception as exc:
            raise DataValidationError(f"Price {name} validation failed: {exc}") from exc
    logger.info("Processed price validation passed (train=%d, val=%d)", len(train), len(val))


def validate_processed_credit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Validate prepared credit train/val/(optional) test frames.

    Raises ``DataValidationError`` on failure.
    """
    cfg = config or load_config("credit")
    schema = processed_credit_schema(cfg)
    splits = [("train", train), ("val", val)]
    if test is not None:
        splits.append(("test", test))
    for name, frame in splits:
        try:
            schema.validate(frame, lazy=True)
        except Exception as exc:
            raise DataValidationError(f"Credit {name} validation failed: {exc}") from exc
    logger.info(
        "Processed credit validation passed (train=%d, val=%d%s)",
        len(train),
        len(val),
        f", test={len(test)}" if test is not None else "",
    )
