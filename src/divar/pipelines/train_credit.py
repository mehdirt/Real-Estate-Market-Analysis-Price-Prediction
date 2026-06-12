"""Train rent/credit models from raw or prepared data."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from divar.config import MODELS_DIR, PROCESSED_DATA_DIR, load_config
from divar.data.load import load_divar
from divar.features.credit import prepare_credit_dataset
from divar.models.train_credit import save_credit_artifacts, train_credit_models
from divar.tracking.deployment import write_deployment_manifest
from divar.tracking.mlflow_utils import log_credit_training_run, write_dvc_metrics
from divar.tracking.registry import get_val_r2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load credit train/val/test parquet files from ``divar-prepare-credit``."""
    paths = {
        "train": processed_dir / "credit_train.parquet",
        "val": processed_dir / "credit_val.parquet",
        "test": processed_dir / "credit_test.parquet",
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files {missing} in {processed_dir}. Run divar-prepare-credit first."
        )
    return (
        pd.read_parquet(paths["train"]),
        pd.read_parquet(paths["val"]),
        pd.read_parquet(paths["test"]),
    )


def _metrics_for_dvc(metrics: dict) -> dict[str, float]:
    """Flatten val/test metrics per algorithm for DVC JSON output."""
    flat = {}
    for split_name, split_models in metrics.items():
        for model_name, model_metrics in split_models.items():
            for metric_name, value in model_metrics.items():
                flat[f"{split_name}_{model_name}_{metric_name}"] = float(value)
    return flat


def main(argv: list[str] | None = None) -> None:
    """CLI entry: train credit models, save artifacts, optional MLflow + DVC metrics."""
    parser = argparse.ArgumentParser(description="Train credit RF and LightGBM models.")
    parser.add_argument("--from-processed", action="store_true")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR / "credit"))
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    parser.add_argument("--metrics-file", type=str, default="metrics/credit.json")
    args = parser.parse_args(argv)

    cfg = load_config("credit")

    if args.from_processed:
        train_data, val_data, test_data = _load_splits(Path(args.processed_dir))
    else:
        df = load_divar()
        train_data, val_data, test_data = prepare_credit_dataset(df, cfg)

    logger.info(
        "Training credit model (train=%d, val=%d, test=%d)...",
        len(train_data),
        len(val_data),
        len(test_data),
    )
    artifacts = train_credit_models(train_data, val_data, test_data, cfg)
    save_credit_artifacts(artifacts, args.models_dir)

    write_deployment_manifest(
        "credit",
        models_dir=Path(args.models_dir),
        val_r2={m: get_val_r2(artifacts["metrics"], m) for m in ("random_forest", "lightgbm")},
    )

    metrics_path = Path(args.models_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts["metrics"], f, indent=2)

    dvc_metrics = _metrics_for_dvc(artifacts["metrics"])
    write_dvc_metrics(dvc_metrics, Path(args.metrics_file))
    logger.info("DVC metrics written to %s", args.metrics_file)

    if args.mlflow or os.getenv("MLFLOW_ENABLE", "false").lower() == "true":
        run_id = log_credit_training_run(
            artifacts, cfg, models_dir=Path(args.models_dir)
        )
        logger.info("MLflow run id: %s", run_id)

    logger.info("Metrics: %s", artifacts["metrics"])
    logger.info("Saved models to %s", args.models_dir)


if __name__ == "__main__":
    main()
