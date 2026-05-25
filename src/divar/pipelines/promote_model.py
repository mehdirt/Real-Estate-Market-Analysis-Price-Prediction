"""Promote a qualified model version from Staging to Production."""

from __future__ import annotations

import argparse
import logging
import sys

from divar.tracking.registry import promote_to_production

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote MLflow model version to Production (requires val R² >= 0.65 at register time)."
    )
    parser.add_argument("--task", choices=["price", "credit"], required=True)
    parser.add_argument(
        "--model",
        choices=["random_forest", "lightgbm"],
        required=True,
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Registry version (default: latest Staging)",
    )
    args = parser.parse_args(argv)

    try:
        version = promote_to_production(args.task, args.model, args.version)
        logger.info(
            "Production ready: task=%s model=%s version=%s. "
            "Start API with MODEL_SOURCE=mlflow to serve this version.",
            args.task,
            args.model,
            version,
        )
    except Exception as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
