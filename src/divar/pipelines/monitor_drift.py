"""Generate Evidently drift reports for price or credit datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from divar.config import PROCESSED_DATA_DIR, PROJECT_ROOT, load_config
from divar.models.sklearn_pipeline import inference_feature_columns
from divar.monitoring.drift import generate_drift_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: compare train vs val distributions and write an HTML drift report."""
    parser = argparse.ArgumentParser(description="Generate Evidently drift HTML report.")
    parser.add_argument("--task", choices=["price", "credit"], required=True)
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "drift"),
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    task = args.task

    if task == "price":
        reference = pd.read_parquet(processed / "price_train.parquet")
        current = pd.read_parquet(processed / "price_val.parquet")
    else:
        reference = pd.read_parquet(processed / "credit_train.parquet")
        current = pd.read_parquet(processed / "credit_val.parquet")

    load_config(task)
    columns = inference_feature_columns(task)
    out = Path(args.output_dir) / f"{task}_drift.html"

    generate_drift_report(reference, current, out, column_subset=columns)
    logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
