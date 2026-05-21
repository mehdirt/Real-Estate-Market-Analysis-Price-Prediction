"""Train sale price models from raw data or prepared parquet."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from divar.config import MODELS_DIR, PROCESSED_DATA_DIR, load_config
from divar.data.load import load_divar
from divar.features.price import prepare_price_dataset
from divar.models.train_price import save_price_artifacts, train_price_models
from divar.tracking.mlflow_utils import log_price_training_run, write_dvc_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "price_train.parquet"
    val_path = processed_dir / "price_val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"Missing processed data in {processed_dir}. Run divar-prepare-price first."
        )
    return pd.read_parquet(train_path), pd.read_parquet(val_path)


def _metrics_for_dvc(metrics: dict) -> dict[str, float]:
    flat = {}
    for model_name, model_metrics in metrics.items():
        for metric_name, value in model_metrics.items():
            flat[f"{model_name}_{metric_name}"] = float(value)
    return flat


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train sale price RF and LightGBM models.")
    parser.add_argument(
        "--from-processed",
        action="store_true",
        help="Load train/val from data/processed instead of raw CSV",
    )
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR / "price"))
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    parser.add_argument("--metrics-file", type=str, default="metrics/price.json")
    args = parser.parse_args(argv)

    cfg = load_config("price")

    if args.from_processed:
        logger.info("Loading processed splits from %s", args.processed_dir)
        train_data, val_data = _load_splits(Path(args.processed_dir))
    else:
        logger.info("Loading raw data and preparing splits...")
        df = load_divar()
        train_data, val_data = prepare_price_dataset(df, cfg)

    logger.info("Training models (train=%d, val=%d)...", len(train_data), len(val_data))
    artifacts = train_price_models(train_data, val_data, cfg)

    save_price_artifacts(artifacts, args.models_dir)

    metrics_path = Path(args.models_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts["metrics"], f, indent=2)

    dvc_metrics = _metrics_for_dvc(artifacts["metrics"])
    write_dvc_metrics(dvc_metrics, Path(args.metrics_file))
    logger.info("DVC metrics written to %s", args.metrics_file)

    if args.mlflow or os.getenv("MLFLOW_ENABLE", "false").lower() == "true":
        run_id = log_price_training_run(
            artifacts, cfg, models_dir=Path(args.models_dir)
        )
        logger.info("MLflow run id: %s", run_id)

    logger.info("Validation metrics: %s", artifacts["metrics"])
    logger.info("Saved models to %s", args.models_dir)


if __name__ == "__main__":
    main()
