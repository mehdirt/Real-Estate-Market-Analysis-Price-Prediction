"""Prepare and persist credit train/val/test datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from divar.config import PROCESSED_DATA_DIR, load_config
from divar.data.load import load_divar
from divar.features.credit import prepare_credit_dataset
from divar.validation import validate_processed_credit, validate_raw_divar
from divar.validation.run import DataValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: build credit features, validate, and write train/val/test parquet files."""
    parser = argparse.ArgumentParser(description="Prepare rent/credit train/val/test parquet files.")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config("credit")

    try:
        logger.info("Loading Divar dataset...")
        df = load_divar()
        if not args.skip_validation:
            validate_raw_divar(df)

        logger.info("Preparing credit features and splits...")
        train_data, val_data, test_data = prepare_credit_dataset(df, cfg)

        if not args.skip_validation:
            validate_processed_credit(train_data, val_data, test_data, cfg)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "credit_train.parquet": train_data,
        "credit_val.parquet": val_data,
        "credit_test.parquet": test_data,
    }
    for name, frame in paths.items():
        path = out_dir / name
        frame.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(frame))


if __name__ == "__main__":
    main()
