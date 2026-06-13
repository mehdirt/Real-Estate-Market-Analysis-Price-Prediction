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

ROOMS_CHOICES = [
    "unselect",
    "0_rooms",
    "1_room",
    "2_rooms",
    "3_rooms",
    "4_rooms",
    "5plus_rooms",
]
LUXURY_CHOICES = ["unselect", "none", "low", "medium", "high"]
BUILDING_AGE_CHOICES = [
    "unselect",
    "new",
    "relatively_new",
    "mid_age",
    "old",
    "very_old",
]
DEED_CHOICES = ["unselect", "single_page", "multi_page", "written_agreement", "other"]
FLOOR_MATERIAL_CHOICES = [
    "unselect",
    "ceramic",
    "stone",
    "laminate_parquet",
    "wood_parquet",
    "floor_covering",
    "carpet",
    "mosaic",
]
FLOOR_CAT_CHOICES = ["basement", "low-height", "mid-height", "high-height", "very-high"]
TOTAL_FLOORS_CAT_CHOICES = ["low", "medium", "high", "very-high"]
CREDIT_FLOOR_CHOICES = [
    "unselect",
    "underground",
    "low_floors",
    "mid_low_floors",
    "mid_floors",
    "high_floors",
    "very_high",
]
CREDIT_TOTAL_FLOORS_CHOICES = [
    "unselect",
    "low_floors",
    "mid_low_floors",
    "mid_floors",
    "high_floors",
    "very_high",
]
UNIT_PER_FLOOR_CHOICES = ["unselect", "1", "2", "3", "4", "5-6", "7-8", "9+"]
WARM_WATER_CHOICES = ["unselect", "package", "powerhouse", "water_heater"]
HEATING_CHOICES = [
    "unselect",
    "split",
    "duct_split",
    "fan_coil",
    "floor_heating",
    "heater",
    "shoofaj",
    "fireplace",
]
COOLING_CHOICES = [
    "unselect",
    "split",
    "duct_split",
    "fan_coil",
    "air_conditioner",
    "water_cooler",
]
RESTROOM_CHOICES = ["unselect", "seat", "squat", "squat_seat"]
DIRECTION_CHOICES = ["unselect", "north", "south", "east", "west"]
PRICE_CAT2_CHOICES = ["residential-sell", "commercial-sell", "real-estate-services"]
PRICE_CAT3_CHOICES = [
    "apartment-sell",
    "house-villa-sell",
    "office-sell",
    "shop-sell",
    "plot-old",
    "presell",
    "industry-agriculture-business-sell",
]
CREDIT_CAT2_CHOICES = ["residential-rent", "commercial-rent", "Other"]
CREDIT_CAT3_CHOICES = [
    "apartment-rent",
    "house-villa-rent",
    "office-rent",
    "shop-rent",
    "industry-agriculture-business-rent",
    "Other",
]

GROUP_BASICS = "Property basics"
GROUP_BUILDING = "Building details"
GROUP_AMENITIES = "Amenities & features"
GROUP_LOCATION = "Location"

# Per-task field metadata. ``kind`` ∈ {"num", "cat", "bool", "cat_dynamic"}.
# ``cat_dynamic`` sources options from val parquet at runtime (high-cardinality).
FIELD_META: dict[str, dict[str, dict[str, Any]]] = {
    "price": {
        "cat2_slug": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Listing category",
            "options": PRICE_CAT2_CHOICES,
        },
        "cat3_slug": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Listing sub-category",
            "options": PRICE_CAT3_CHOICES,
        },
        "rooms_count": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Rooms",
            "options": ROOMS_CHOICES,
        },
        "building_size": {
            "kind": "num",
            "group": GROUP_BASICS,
            "label": "Building size",
            "unit": "m²",
            "min": 20.0,
            "max": 3000.0,
            "step": 5.0,
        },
        "building_age": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Building age",
            "options": BUILDING_AGE_CHOICES,
        },
        "floor_cat": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Floor (binned)",
            "options": FLOOR_CAT_CHOICES,
        },
        "total_floors_cat": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Total floors (binned)",
            "options": TOTAL_FLOORS_CAT_CHOICES,
        },
        "floor_material": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Floor material",
            "options": FLOOR_MATERIAL_CHOICES,
        },
        "is_rebuilt": {
            "kind": "bool",
            "group": GROUP_BUILDING,
            "label": "Rebuilt / renovated",
        },
        "deed_type": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Deed type",
            "options": DEED_CHOICES,
        },
        "has_business_deed": {
            "kind": "bool",
            "group": GROUP_BUILDING,
            "label": "Business deed",
        },
        "luxury_score": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Luxury score",
            "options": LUXURY_CHOICES,
            "help": "Pool / sauna / barbecue / jacuzzi / security guard count, binned.",
        },
        "non_luxury_score": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Standard amenities score",
            "options": LUXURY_CHOICES,
            "help": "Balcony / elevator / warehouse / parking count, binned.",
        },
        "city_slug": {
            "kind": "cat_dynamic",
            "group": GROUP_LOCATION,
            "label": "City",
        },
        "neighborhood_slug": {
            "kind": "cat_dynamic",
            "group": GROUP_LOCATION,
            "label": "Neighborhood",
        },
        "location_latitude": {
            "kind": "num",
            "group": GROUP_LOCATION,
            "label": "Latitude",
            "min": 23.0,
            "max": 41.0,
            "step": 0.001,
            "format": "%.5f",
        },
        "location_longitude": {
            "kind": "num",
            "group": GROUP_LOCATION,
            "label": "Longitude",
            "min": 43.0,
            "max": 65.0,
            "step": 0.001,
            "format": "%.5f",
        },
    },
    "credit": {
        "cat2_slug": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Listing category",
            "options": CREDIT_CAT2_CHOICES,
        },
        "cat3_slug": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Listing sub-category",
            "options": CREDIT_CAT3_CHOICES,
        },
        "rooms_count": {
            "kind": "cat",
            "group": GROUP_BASICS,
            "label": "Rooms",
            "options": ROOMS_CHOICES,
        },
        "building_size": {
            "kind": "num",
            "group": GROUP_BASICS,
            "label": "Building size",
            "unit": "m²",
            "min": 20.0,
            "max": 3000.0,
            "step": 5.0,
        },
        "building_age": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Building age",
            "options": BUILDING_AGE_CHOICES,
        },
        "floor": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Floor (binned)",
            "options": CREDIT_FLOOR_CHOICES,
        },
        "total_floors_count": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Total floors (binned)",
            "options": CREDIT_TOTAL_FLOORS_CHOICES,
        },
        "unit_per_floor": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Units per floor",
            "options": UNIT_PER_FLOOR_CHOICES,
        },
        "floor_material": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Floor material",
            "options": FLOOR_MATERIAL_CHOICES,
        },
        "is_rebuilt": {
            "kind": "bool",
            "group": GROUP_BUILDING,
            "label": "Rebuilt / renovated",
        },
        "building_direction": {
            "kind": "cat",
            "group": GROUP_BUILDING,
            "label": "Building direction",
            "options": DIRECTION_CHOICES,
        },
        "has_warm_water_provider": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Warm water provider",
            "options": WARM_WATER_CHOICES,
        },
        "has_heating_system": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Heating system",
            "options": HEATING_CHOICES,
        },
        "has_cooling_system": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Cooling system",
            "options": COOLING_CHOICES,
        },
        "has_restroom": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Restroom type",
            "options": RESTROOM_CHOICES,
        },
        "luxury_items": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Luxury score",
            "options": LUXURY_CHOICES,
            "help": "Pool / sauna / barbecue / jacuzzi / security guard, binned.",
        },
        "non_luxury_items": {
            "kind": "cat",
            "group": GROUP_AMENITIES,
            "label": "Standard amenities score",
            "options": LUXURY_CHOICES,
            "help": "Balcony / elevator / warehouse / parking, binned.",
        },
        "city_slug": {
            "kind": "cat_dynamic",
            "group": GROUP_LOCATION,
            "label": "City",
        },
        "neighborhood_slug": {
            "kind": "cat_dynamic",
            "group": GROUP_LOCATION,
            "label": "Neighborhood",
        },
        "location_latitude": {
            "kind": "num",
            "group": GROUP_LOCATION,
            "label": "Latitude",
            "min": 23.0,
            "max": 41.0,
            "step": 0.001,
            "format": "%.5f",
        },
        "location_longitude": {
            "kind": "num",
            "group": GROUP_LOCATION,
            "label": "Longitude",
            "min": 43.0,
            "max": 65.0,
            "step": 0.001,
            "format": "%.5f",
        },
    },
}

GROUP_ORDER = [GROUP_BASICS, GROUP_BUILDING, GROUP_AMENITIES, GROUP_LOCATION]
GROUP_ICONS = {
    GROUP_BASICS: "🏷️",
    GROUP_BUILDING: "🏗️",
    GROUP_AMENITIES: "✨",
    GROUP_LOCATION: "📍",
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


@st.cache_data
def _load_metrics(task: str) -> dict[str, float]:
    """Return val R² per model from ``metrics/{task}.json``. Empty if missing."""
    from divar.config import PROJECT_ROOT

    path = PROJECT_ROOT / "metrics" / f"{task}.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, float] = {}
    for model_full in ("random_forest", "lightgbm"):
        for prefix in ("val_", ""):
            r2_key = f"{prefix}{model_full}_r2"
            if r2_key in raw:
                out[model_full] = float(raw[r2_key])
                break
    return out


@st.cache_data
def _dynamic_options(task: str, field: str) -> list[str]:
    val = _load_val(task)
    if val is None or field not in val.columns:
        return []
    return sorted(str(v) for v in val[field].dropna().unique())


def _safe_index(options: list[Any], value: Any) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _coerce_num(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _render_field(
    task: str,
    field: str,
    meta: dict[str, Any],
    current: Any,
    key_prefix: str,
) -> Any:
    """Render the right widget per field metadata and return the chosen value."""
    label = meta.get("label", field)
    help_text = meta.get("help")
    key = f"{key_prefix}_{task}_{field}"
    kind = meta["kind"]

    if kind == "num":
        return st.number_input(
            label,
            value=_coerce_num(current, meta.get("min", 0.0)),
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            step=float(meta["step"]),
            format=meta.get("format"),
            help=help_text,
            key=key,
        )

    if kind == "cat":
        options = list(meta["options"])
        current_str = str(current) if pd.notna(current) else options[0]
        if current_str not in options:
            options = [current_str, *options]
        return st.selectbox(
            label,
            options,
            index=_safe_index(options, current_str),
            help=help_text,
            key=key,
        )

    if kind == "cat_dynamic":
        options = _dynamic_options(task, field)
        if not options:
            return st.text_input(label, value=str(current) if pd.notna(current) else "", key=key)
        current_str = str(current) if pd.notna(current) else options[0]
        if current_str not in options:
            options = [current_str, *options]
        return st.selectbox(
            label,
            options,
            index=_safe_index(options, current_str),
            help=help_text,
            key=key,
        )

    if kind == "bool":
        choices = ["Yes", "No", "unselect"]
        if current is True:
            idx = 0
        elif current is False:
            idx = 1
        else:
            idx = 2
        choice = st.radio(
            label,
            choices,
            index=idx,
            horizontal=True,
            help=help_text,
            key=key,
        )
        return {"Yes": True, "No": False, "unselect": None}[choice]

    return st.text_input(label, value=str(current) if pd.notna(current) else "", key=key)


def _row_to_record(
    row: pd.Series, feature_cols: list[str], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Build a JSON-safe record from a val row with field overrides for ALL features."""
    base = json.loads(row[feature_cols].to_frame().T.to_json(orient="records"))[0]
    for k, v in overrides.items():
        if isinstance(v, float) and pd.isna(v):
            base[k] = None
        else:
            base[k] = v
    return base


def _format_money(value: float) -> str:
    """Display Iranian Rial amount with a short suffix."""
    if value >= 1e12:
        return f"{value / 1e12:,.2f}T IRR"
    if value >= 1e9:
        return f"{value / 1e9:,.2f}B IRR"
    if value >= 1e6:
        return f"{value / 1e6:,.1f}M IRR"
    return f"{value:,.0f} IRR"


def _call_predict(task: str, model: str, record: dict[str, Any]) -> tuple[float | None, str | None]:
    try:
        resp = requests.post(
            f"{API_URL}/predict/{task}",
            json={"model": model, "records": [record]},
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"
    if resp.status_code != 200:
        return None, f"API {resp.status_code}: {resp.text}"
    return float(resp.json()["predictions"][0]), None


# --------------------------------------------------------------------------- UI

st.set_page_config(
    page_title="Divar ML — Price & Credit",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 1.5rem; max-width: 1400px; }
h1, h2, h3, h4, h5 { letter-spacing: -0.01em; }
h5 { margin: 0.25rem 0 0.5rem 0 !important; font-size: 0.95rem; font-weight: 600; color: #444; }
.divar-hero {
    background: linear-gradient(135deg, #A4133C 0%, #FF6B6B 100%);
    border-radius: 12px; padding: 0.85rem 1.2rem; margin: 0 0 0.85rem 0; color: #fff;
}
.divar-hero h1 { color: #fff; margin: 0 0 0.15rem 0; font-size: 1.35rem; font-weight: 700; }
.divar-hero p { color: rgba(255,255,255,0.92); margin: 0; font-size: 0.85rem; }
.divar-pill {
    display: inline-block; padding: 1px 9px; border-radius: 999px;
    font-size: 0.7rem; background: rgba(255,255,255,0.18); color: #fff;
    margin-right: 5px; vertical-align: middle; font-weight: 500;
}
.divar-section {
    font-size: 0.8rem; font-weight: 600; color: #888; text-transform: uppercase;
    letter-spacing: 0.06em; margin: 1rem 0 0.25rem 0;
}
section[data-testid="stSidebar"] h3 { margin-top: 0.5rem; }
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
div[data-testid="stMetricLabel"] { font-size: 0.78rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid #eee; }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; font-size: 0.9rem; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0.75rem; }
div[data-testid="column"] > div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
hr { margin: 0.5rem 0 0.75rem 0; }
.stRadio > label { font-size: 0.85rem; }
</style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    task = st.selectbox(
        "Prediction task",
        ["price", "credit"],
        format_func=lambda x: "💰 Sale price" if x == "price" else "🏠 Rent / total credit",
        help="Pick the regression task — sell vs rent listings.",
    )

    health = _check_api()
    if health is None:
        st.error(f"API unreachable at `{API_URL}`")
        st.caption("Start the FastAPI service first:")
        st.code("divar-serve", language="bash")
        st.caption("Or set `DIVAR_API_URL` to point elsewhere.")
        st.stop()

    st.success(f"✅ API connected · `{health.get('model_source', '?')}`")
    available = health.get("models_loaded", {}).get(task, [])
    if not available:
        st.warning(f"No models loaded for `{task}`.")
        st.caption(f"Run `divar-train-{task}` first.")
        st.stop()

    model = st.selectbox(
        "Model",
        available,
        format_func=lambda m: "🌲 Random Forest" if m == "random_forest" else "⚡ LightGBM",
    )
    compare_models = st.checkbox(
        "Compare both models",
        value=False,
        help="Run prediction with every loaded model and show them side-by-side.",
        disabled=len(available) < 2,
    )

    metrics = _load_metrics(task)
    if metrics:
        st.markdown("##### Validation R²")
        cols = st.columns(len(metrics))
        for (m, r2), col in zip(metrics.items(), cols):
            short = "RF" if m == "random_forest" else "LGBM"
            col.metric(short, f"{r2:.3f}")

    if health.get("deployment"):
        with st.expander("Deployment manifest"):
            st.json(health["deployment"], expanded=False)

    st.caption(f"API · `{API_URL}`")

st.markdown(
    f"""
<div class="divar-hero">
  <h1>🏠 Divar — {"Sale Price" if task == "price" else "Rent / Total Credit"} Prediction</h1>
  <p>
    <span class="divar-pill">FastAPI</span>
    <span class="divar-pill">Streamlit</span>
    <span class="divar-pill">{"Random Forest + LightGBM" if len(available) >= 2 else available[0]}</span>
    Pick a real listing from the validation set, tweak any feature, and run live inference against the API.
  </p>
</div>
    """,
    unsafe_allow_html=True,
)

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

# ----------------------------------------------------- Picker (single key)
input_key = f"input_{task}"
last_row_key = f"last_row_{task}"
result_key = f"result_{task}"
if input_key not in st.session_state:
    st.session_state[input_key] = 0


def _randomize_row() -> None:
    st.session_state[input_key] = random.randrange(len(val))
    st.session_state.pop(result_key, None)


def _reset_edits() -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(f"edit_{task}_"):
            del st.session_state[k]
    st.session_state.pop(result_key, None)


st.markdown('<div class="divar-section">① Pick a listing</div>', unsafe_allow_html=True)
picker_col, random_col, info_col = st.columns([3, 1, 2], vertical_alignment="bottom")
with picker_col:
    idx = int(
        st.number_input(
            f"Row from val set (0 – {len(val) - 1})",
            min_value=0,
            max_value=len(val) - 1,
            step=1,
            key=input_key,
        )
    )
with random_col:
    st.button(
        "🎲 Random listing",
        use_container_width=True,
        key=f"rand_{task}",
        on_click=_randomize_row,
    )
with info_col:
    st.metric(
        "🎯 Actual target",
        _format_money(float(val.iloc[idx][target_col])),
        help=f"`{target_col}` from the val set for this row.",
    )

# When the row changes, clear field-widget session state so widgets re-init
# from the new row's values (Streamlit ignores ``value=`` once a key has state).
if st.session_state.get(last_row_key) != idx:
    for k in list(st.session_state.keys()):
        if k.startswith(f"edit_{task}_"):
            del st.session_state[k]
    st.session_state[last_row_key] = idx
    st.session_state.pop(result_key, None)

row = val.iloc[idx]

# ---------------------------------------------------------------- Features
st.markdown('<div class="divar-section">② Features</div>', unsafe_allow_html=True)

task_meta = FIELD_META[task]
features_col, map_col = st.columns([3, 2], gap="large")
overrides: dict[str, Any] = {}

with features_col:
    tabs = st.tabs([f"{GROUP_ICONS[g]} {g}" for g in GROUP_ORDER])
    for tab, group in zip(tabs, GROUP_ORDER):
        with tab:
            group_fields = [
                c for c in feature_cols if task_meta.get(c, {}).get("group") == group
            ]
            if not group_fields:
                st.caption("(no fields in this group)")
                continue
            grid = st.columns(2)
            for i, field in enumerate(group_fields):
                with grid[i % 2]:
                    overrides[field] = _render_field(
                        task, field, task_meta[field], row[field], "edit"
                    )

with map_col:
    st.markdown("##### 📍 Location preview")
    lat = overrides.get("location_latitude", row.get("location_latitude"))
    lng = overrides.get("location_longitude", row.get("location_longitude"))
    if pd.notna(lat) and pd.notna(lng):
        st.map(
            pd.DataFrame({"lat": [float(lat)], "lon": [float(lng)]}),
            zoom=11,
            use_container_width=True,
        )
    else:
        st.info("No coordinates for this listing.")
    with st.expander("📋 Raw row snapshot"):
        st.dataframe(row, use_container_width=True)

# ---------------------------------------------------------------- Predict
st.markdown('<div class="divar-section">③ Predict</div>', unsafe_allow_html=True)
predict_col, reset_col, _ = st.columns([2, 1, 4])
with predict_col:
    predict_clicked = st.button(
        "🚀 Run prediction", type="primary", use_container_width=True, key=f"predict_{task}"
    )
with reset_col:
    st.button(
        "↺ Reset edits",
        use_container_width=True,
        key=f"reset_{task}",
        on_click=_reset_edits,
        help="Restore every feature to the val row's original value.",
    )

if predict_clicked:
    record = _row_to_record(row, feature_cols, overrides)
    actual = float(row[target_col]) if pd.notna(row[target_col]) else None
    models_to_run = available if compare_models else [model]
    results: list[tuple[str, float]] = []
    with st.spinner("Calling /predict …"):
        for m in models_to_run:
            pred, err = _call_predict(task, m, record)
            if err is not None:
                st.error(f"{m}: {err}")
                continue
            results.append((m, pred))
    st.session_state[result_key] = {
        "results": results,
        "actual": actual,
        "record": record,
        "model_for_payload": models_to_run[0] if models_to_run else model,
    }

result = st.session_state.get(result_key)
if result and result["results"]:
    st.markdown("#### Result")
    actual = result["actual"]
    results = result["results"]
    cards = st.columns(len(results) + (1 if actual is not None else 0))
    for (m, pred), col in zip(results, cards):
        with col:
            label = "🌲 Random Forest" if m == "random_forest" else "⚡ LightGBM"
            delta = None
            if actual is not None:
                err_pct = 100.0 * (pred - actual) / actual if actual else 0.0
                delta = f"{err_pct:+.1f}% vs actual"
            st.metric(label, _format_money(pred), delta=delta, delta_color="inverse")
    if actual is not None:
        with cards[-1]:
            st.metric("🎯 Actual", _format_money(actual))
    with st.expander("Show request payload"):
        st.code(
            json.dumps(
                {"model": result["model_for_payload"], "records": [result["record"]]},
                indent=2,
                default=str,
            ),
            language="json",
        )

st.markdown("---")
st.caption(
    f"Task `{task}` · target `{target_col}` · "
    f"{len(feature_cols)} features · "
    f"val set: {len(val):,} rows · API: `{API_URL}`"
)
