"""Prepare and persist price train/val datasets."""

from __future__ import annotations

import argparse
import logging

from divar.config import PROCESSED_DATA_DIR, load_config
from divar.data.load import load_divar
from divar.features.price import prepare_price_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare sale price train/val parquet files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Directory for processed parquet outputs",
    )
    args = parser.parse_args(argv)

    cfg = load_config("price")
    logger.info("Loading Divar dataset...")
    df = load_divar()

    logger.info("Preparing price features and splits...")
    train_data, val_data = prepare_price_dataset(df, cfg)

    out_dir = __import__("pathlib").Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "price_train.parquet"
    val_path = out_dir / "price_val.parquet"

    train_data.to_parquet(train_path, index=False)
    val_data.to_parquet(val_path, index=False)

    logger.info("Wrote %s (%d rows)", train_path, len(train_data))
    logger.info("Wrote %s (%d rows)", val_path, len(val_data))


if __name__ == "__main__":
    main()
