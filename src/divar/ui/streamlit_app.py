"""Streamlit UI: pick a val-set listing, tweak features, call /predict/{task}.

Launched via ``divar-streamlit`` (see ``divar.ui.launch``). Requires the FastAPI
service running at ``DIVAR_API_URL`` (default ``http://127.0.0.1:8000``).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import pandas as pd
import requests
import streamlit as st

from divar.config import PROCESSED_DATA_DIR, load_config
from divar.models.sklearn_pipeline import inference_feature_columns

API_URL = os.getenv("DIVAR_API_URL", "http://127.0.0.1:8000")

EDITABLE_FIELDS: dict[str, list[str]] = {
    "price": [
        "building_size",
        "rooms_count",
        "building_age",
        "luxury_score",
        "non_luxury_score",
    ],
    "credit": [
        "building_size",
        "rooms_count",
        "building_age",
        "floor",
        "total_floors_count",
    ],
}


def _check_api() -> dict[str, Any] | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


@st.cache_data
def _load_val(task: str) -> pd.DataFrame | None:
    path = PROCESSED_DATA_DIR / f"{task}_val.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _row_to_record(
    row: pd.Series, feature_cols: list[str], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Build a JSON-safe record from a val row with optional field overrides."""
    base = json.loads(row[feature_cols].to_frame().T.to_json(orient="records"))[0]
    base.update(overrides)
    return base


st.set_page_config(page_title="Divar ML", page_icon="🏠", layout="wide")
st.title("Divar — Price & Credit Prediction")
st.caption(
    "FastAPI-backed Streamlit demo. Pick a listing from the val set, tweak key features, predict."
)

with st.sidebar:
    st.header("Settings")
    task = st.selectbox("Task", ["price", "credit"], help="Sale price or rent/credit")

    health = _check_api()
    if health is None:
        st.error(f"API unreachable at `{API_URL}`")
        st.code("divar-serve", language="bash")
        st.caption("Set `DIVAR_API_URL` to point elsewhere.")
        st.stop()

    st.success(f"API @ {API_URL}")
    st.caption(f"Source: `{health.get('model_source', '?')}`")
    available = health.get("models_loaded", {}).get(task, [])
    st.caption(f"Available for {task}: {available or '—'}")
    model = st.selectbox("Model", available or ["lightgbm", "random_forest"])

val = _load_val(task)
if val is None:
    st.error(
        f"No val parquet at `data/processed/{task}_val.parquet`. "
        f"Run `divar-prepare-{task}` first."
    )
    st.stop()

cfg = load_config(task)
target_col = cfg["target_column"]
feature_cols = inference_feature_columns(task)

st.subheader("Pick a listing")
state_key = f"row_idx_{task}"
if state_key not in st.session_state:
    st.session_state[state_key] = 0

picker_col, random_col = st.columns([4, 1])
with picker_col:
    idx = st.number_input(
        "Row from val set",
        min_value=0,
        max_value=len(val) - 1,
        value=int(st.session_state[state_key]),
        key=f"input_{task}",
    )
    st.session_state[state_key] = int(idx)
with random_col:
    st.write("")
    st.write("")
    if st.button("🎲 Random", use_container_width=True):
        st.session_state[state_key] = random.randrange(len(val))
        st.rerun()

row = val.iloc[int(st.session_state[state_key])]

features_col, map_col = st.columns([3, 2])

with features_col:
    st.subheader("Features")
    overrides: dict[str, Any] = {}
    edit_fields = [f for f in EDITABLE_FIELDS[task] if f in feature_cols]
    grid = st.columns(2)
    for i, field in enumerate(edit_fields):
        with grid[i % 2]:
            current = row[field]
            try:
                overrides[field] = st.number_input(
                    field,
                    value=float(current) if pd.notna(current) else 0.0,
                    key=f"edit_{task}_{field}",
                )
            except (TypeError, ValueError):
                overrides[field] = st.text_input(
                    field, value=str(current), key=f"edit_{task}_{field}"
                )

    with st.expander("Other features (read-only)"):
        readonly = row[feature_cols].drop(labels=edit_fields, errors="ignore")
        st.dataframe(readonly, use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        record = _row_to_record(row, feature_cols, overrides)
        try:
            resp = requests.post(
                f"{API_URL}/predict/{task}",
                json={"model": model, "records": [record]},
                timeout=10,
            )
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        if resp.status_code != 200:
            st.error(f"API {resp.status_code}: {resp.text}")
        else:
            pred = float(resp.json()["predictions"][0])
            actual = float(row[target_col]) if pd.notna(row[target_col]) else None
            st.subheader("Result")
            cols = st.columns(3)
            cols[0].metric("Predicted", f"{pred:,.0f}")
            if actual is not None:
                cols[1].metric("Actual", f"{actual:,.0f}")
                err_pct = 100.0 * (pred - actual) / actual if actual else 0.0
                cols[2].metric("Error", f"{err_pct:+.1f}%", delta_color="off")

with map_col:
    st.subheader("Location")
    lat = row.get("location_latitude")
    lng = row.get("location_longitude")
    if pd.notna(lat) and pd.notna(lng):
        st.map(
            pd.DataFrame({"lat": [float(lat)], "lon": [float(lng)]}),
            zoom=11,
        )
    else:
        st.info("No coordinates for this row.")
