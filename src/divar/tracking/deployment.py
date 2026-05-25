"""Local deployment manifest (which models the API uses on disk)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from divar.config import MODELS_DIR


def write_deployment_manifest(
    task: str,
    *,
    models_dir: Path,
    run_id: str | None = None,
    registry_versions: dict[str, int | None] | None = None,
    val_r2: dict[str, float | None] | None = None,
) -> Path:
    """
    Record what was last written under ``models/{task}/`` for operators and the API.

    The API in **local** mode loads ``*_pipeline.joblib`` from this directory;
    files are overwritten on each successful training run.
    """
    manifest = {
        "task": task,
        "source": "local",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "models": {},
    }
    registry_versions = registry_versions or {}
    val_r2 = val_r2 or {}
    for model in ("random_forest", "lightgbm"):
        manifest["models"][model] = {
            "pipeline": str(models_dir / f"{model}_pipeline.joblib"),
            "val_r2": val_r2.get(model),
            "registry_version": registry_versions.get(model),
        }

    path = models_dir / "deployment.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_deployment_manifest(task: str, models_root: Path | None = None) -> dict[str, Any] | None:
    """Read deployment manifest if present."""
    root = models_root or MODELS_DIR
    path = root / task / "deployment.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)
