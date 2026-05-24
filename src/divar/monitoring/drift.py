"""Evidently data drift reports."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path,
    *,
    column_subset: list[str] | None = None,
) -> Path:
    """
    Build an HTML drift report comparing reference vs current data.

    Uses Evidently's DataDriftPreset.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    ref = reference.copy()
    cur = current.copy()

    if column_subset:
        cols = [c for c in column_subset if c in ref.columns and c in cur.columns]
        ref = ref[cols]
        cur = cur[cols]

    # Drop target if present for drift on features only
    for col in ("price_value", "total_credit"):
        if col in ref.columns:
            ref = ref.drop(columns=[col])
        if col in cur.columns:
            cur = cur.drop(columns=[col])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    logger.info("Drift report saved to %s", output_path)
    return output_path
