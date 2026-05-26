"""DVC stage: validate processed parquet splits."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from divar.config import PROCESSED_DATA_DIR
from divar.validation import validate_processed_credit, validate_processed_price
from divar.validation.run import DataValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """CLI entry: validate processed price or credit parquet splits."""
    parser = argparse.ArgumentParser(description="Validate processed parquet datasets.")
    parser.add_argument(
        "--task",
        choices=["price", "credit"],
        required=True,
        help="Which prepared dataset to validate",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
    )
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)

    try:
        if args.task == "price":
            train = pd.read_parquet(processed / "price_train.parquet")
            val = pd.read_parquet(processed / "price_val.parquet")
            validate_processed_price(train, val)
        else:
            train = pd.read_parquet(processed / "credit_train.parquet")
            val = pd.read_parquet(processed / "credit_val.parquet")
            test = pd.read_parquet(processed / "credit_test.parquet")
            validate_processed_credit(train, val, test)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
