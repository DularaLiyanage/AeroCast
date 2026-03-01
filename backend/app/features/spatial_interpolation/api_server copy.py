# backend/api_server.py
from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import base64
import io
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from joblib import load as joblib_load
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths (IMPORTANT because this file is inside backend/)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_DIR = BASE_DIR / "city_predictor_model_dl_md"
OBS_CSV = BASE_DIR / "clean" / "cea_hourly_2019_2024_clean_dl_md.csv"
SL_GEOJSON = BASE_DIR / "data" / "sri_lanka_boundary.geojson"

WINDOW = 24  # must match training

STATION_META = {
    "Battaramulla": {"lat": 6.901035, "lon": 79.926513},
    "Kandy":        {"lat": 7.292651, "lon": 80.635649},
}

TARGETS = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
]

DEFAULT_VALS = {
    "PM25": 40.0, "PM10": 60.0, "NO2": 25.0, "SO2": 5.0, "O3": 10.0,
    "CO": 400.0, "NOX": 30.0, "WS": 1.5, "WD": 120.0, "AT": 30.0,
    "RH": 70.0, "BP": 1010.0, "SolarRad": 500.0, "Rain": 0.0,
}
OVERRIDE_COLS = list(DEFAULT_VALS.keys())

# Optional: keep outputs within realistic limits (same as web_app)
BOUNDS = {
    "PM25": (0, 1000),
    "PM10": (0, 1500),
    "NO2": (0, 1000),
    "SO2": (0, 1000),
    "O3": (0, 1000),
    "CO": (0, 50000),
    "NOX": (0, 2000),
    "WS": (0, 60),
    "WD": (0, 360),
    "AT": (-5, 50),
    "RH": (0, 100),
    "BP": (800, 1100),
    "SolarRad": (0, 1400),
    "Rain": (0, 500),
}


def clamp_value(target: str, x: float) -> float:
    if target == "WD":
        return float(x) % 360.0
    if target in BOUNDS:
        lo, hi = BOUNDS[target]
        return float(min(max(x, lo), hi))
    return float(x)


# ============================================================
# Load observed CSV once (exactly like web_app)
# ============================================================
if not OBS_CSV.exists():
    raise FileNotFoundError(f"OBS_CSV not found: {OBS_CSV}")

obs = pd.read_csv(OBS_CSV, parse_dates=["datetime"])
keep_cols = [
    "station", "datetime",
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
    "month","day",
    "hour","hour_sin","hour_cos"
]
missing = [c for c in keep_cols if c not in obs.columns]
if missing:
    raise RuntimeError(f"OBS_CSV missing columns: {missing}")

obs = obs[keep_cols].copy()


# ============================================================
# Feature helpers (copied from web_app logic)
# ============================================================
def add_month_day_features_row(dt: pd.Timestamp, row: dict):
    row["month"] = int(dt.month)
    row["day"] = int(dt.day)

def add_hour_features_row(dt: pd.Timestamp, row: dict):
    h = int(dt.hour)
    row["hour_sin"] = float(np.sin(2 * np.pi * h / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * h / 24))

def add_wind_components_row(row: dict):
    rad = np.deg2rad(float(row["WD"]))
    row["u"] = float(float(row["WS"]) * np.cos(rad))
    row["v"] = float(float(row["WS"]) * np.sin(rad))

def compute_lag_roll_from_history(hist: pd.DataFrame, col: str, lags, rolls):
    """
    hist: station history sorted by datetime, includes latest row as "current"
    lagL at current time uses value from L hours before => index -(L+1)
    rolling mean uses past only (exclude current row)
    """
    out = {}
    s = hist[col].astype(float).reset_index(drop=True)

    for L in lags:
        idx = -(L + 1)
        out[f"{col}_lag{L}"] = float(s.iloc[idx]) if len(s) >= (L + 1) else float("nan")

    past = s.iloc[:-1]  # exclude current
    for W in rolls:
        if len(past) == 0:
            out[f"{col}_roll{W}"] = float("nan")
            continue
        w = past.iloc[-W:] if len(past) >= W else past
        out[f"{col}_roll{W}"] = float(w.mean()) if len(w) else float("nan")

    return out


# ============================================================
# Model cache (loads once)
# ============================================================
_model_cache: dict[str, tuple[tf.keras.Model, object, dict]] = {}

def load_artifacts(base_target: str):
    if base_target in _model_cache:
        return _model_cache[base_target]

    model_path = MODEL_DIR / f"dl_city_predictor_{base_target}_lead1.keras"
    prep_path  = MODEL_DIR / f"dl_preprocess_{base_target}_lead1.joblib"
    cfg_path   = MODEL_DIR / f"dl_config_{base_target}_lead1.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing preprocess: {prep_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    model = tf.keras.models.load_model(model_path)
    preprocess = joblib_load(prep_path)
    cfg = json.loads(cfg_path.read_text())

    _model_cache[base_target] = (model, preprocess, cfg)
    return model, preprocess, cfg


# ============================================================
# IDW heatmap + Sri Lanka boundary (same approach as web_app)
# ============================================================
def idw_interpolate_grid(lats, lons, vals, grid_lat, grid_lon, power=2.0, eps=1e-12):
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    vals = np.asarray(vals, dtype=float)

    Z = np.zeros_like(grid_lat, dtype=float)
    Wsum = np.zeros_like(grid_lat, dtype=float)

    for la, lo, v in zip(lats, lons, vals):
        d2 = (grid_lat - la) ** 2 + (grid_lon - lo) ** 2
        w = 1.0 / (np.power(np.sqrt(d2) + eps, power))
        Z += w * v
        Wsum += w

    return Z / (Wsum + eps)


def make_heatmap_png_base64(points, values, title="Heatmap", power=2.0, grid_size=190):
    """
    Accurate Sri Lanka boundary (offline geojson) + IDW overlay
    zoomed to Battaramulla–Kandy region.
    """
    if not SL_GEOJSON.exists():
        raise FileNotFoundError(f"Sri Lanka geojson not found: {SL_GEOJSON}")

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    pad_lat = 0.35
    pad_lon = 0.55
    lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat
    lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

    Z = idw_interpolate_grid(lats, lons, values, grid_lat, grid_lon, power=power)

    sl = gpd.read_file(SL_GEOJSON).to_crs("EPSG:4326")
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    sl_clip = gpd.clip(sl, bbox)

    fig = plt.figure(figsize=(7.8, 5.2))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    sl_clip.plot(ax=ax, color="white", edgecolor="black", linewidth=1.3, zorder=1)

    plt.pcolormesh(grid_lon, grid_lat, Z, shading="auto", alpha=0.65, zorder=2)
    plt.colorbar(label="Predicted value (t+1)")

    plt.scatter(lons, lats, s=90, c="red", edgecolors="black", zorder=3)
    for (lat, lon), v in zip(points, values):
        plt.text(lon, lat, f"{v:.2f}", fontsize=9, weight="bold", zorder=4)

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# ============================================================
# ✅ Accurate prediction function (same as web_app predict_one_station)
# ============================================================
def predict_one_station(
    model,
    preprocess,
    cfg,
    station: str,
    dt_base: pd.Timestamp,
    overrides: dict | None = None
) -> float:

    st_obs = obs[obs["station"] == station]
    st_min = st_obs["datetime"].min()
    st_max = st_obs["datetime"].max()

    dt_base = pd.to_datetime(dt_base).floor("h")
    if dt_base < st_min:
        dt_base = pd.to_datetime(st_min).floor("h")
    if dt_base > st_max:
        dt_base = pd.to_datetime(st_max).floor("h")

    lags = tuple(cfg.get("lags", [1, 3, 6, 12, 24]))
    rolls = tuple(cfg.get("rolls", [3, 6, 24]))
    lag_roll_cols = cfg.get("lag_roll_cols", TARGETS)

    max_lag = max(lags) if len(lags) else 1
    need_rows = WINDOW + max_lag

    hist = (
        obs[(obs["station"] == station) & (obs["datetime"] <= dt_base)]
        .sort_values("datetime")
        .tail(need_rows)
    )

    if len(hist) < need_rows:
        # fallback persistence
        t = cfg["base_target"]
        return float(hist.iloc[-1][t]) if len(hist) else 0.0

    # Apply overrides only if provided (selected station)
    if overrides:
        for k, v in overrides.items():
            if k in OVERRIDE_COLS and v is not None:
                hist.loc[hist.index[-1], k] = float(v)

    hist_full = hist.sort_values("datetime").copy().reset_index(drop=True)
    hist_for_seq = hist_full.tail(WINDOW).reset_index(drop=True)

    num_features = cfg["num_features"]
    cat_features = cfg["cat_features"]

    seq_rows = []
    for i in range(len(hist_for_seq)):
        dt_i = pd.to_datetime(hist_for_seq.loc[i, "datetime"])
        sub = hist_full[hist_full["datetime"] <= dt_i].copy().sort_values("datetime")

        base = {
            "station": station,
            "lat": STATION_META[station]["lat"],
            "lon": STATION_META[station]["lon"],
        }

        # current values for all variables
        for col in OVERRIDE_COLS:
            base[col] = float(hist_for_seq.loc[i, col])

        add_month_day_features_row(dt_i, base)
        add_hour_features_row(dt_i, base)
        add_wind_components_row(base)

        for c in lag_roll_cols:
            if c in sub.columns:
                base.update(compute_lag_roll_from_history(sub, c, lags=lags, rolls=rolls))
            else:
                for L in lags:
                    base[f"{c}_lag{L}"] = np.nan
                for W in rolls:
                    base[f"{c}_roll{W}"] = np.nan

        seq_rows.append(base)

    X_df = pd.DataFrame(seq_rows).ffill().bfill()
    X_flat = preprocess.transform(X_df[num_features + cat_features]).astype(np.float32)
    X_seq = X_flat.reshape(1, WINDOW, -1)

    pred = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
    return pred


# ============================================================
# API
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],  # includes OPTIONS
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    station: str
    target: str
    datetime_str: str
    overrides: dict | None = None  # {"PM25": 50.0, "WS": 2.0, ...} or nulls


@app.post("/predict")
def predict(req: PredictRequest):
    station = req.station
    target = req.target
    if target not in TARGETS:
        return {"error": f"Invalid target '{target}'. Valid: {TARGETS}"}
    if station not in STATION_META:
        return {"error": f"Invalid station '{station}'. Valid: {list(STATION_META.keys())}"}

    dt_base = pd.to_datetime(req.datetime_str, errors="coerce")
    if pd.isna(dt_base):
        return {"error": "Invalid datetime_str. Use 'YYYY-MM-DD HH:MM:SS'."}

    model, preprocess, cfg = load_artifacts(target)

    # Main prediction (selected station with overrides)
    pred_main_raw = predict_one_station(
        model, preprocess, cfg, station, dt_base, overrides=req.overrides
    )
    pred_main = clamp_value(target, pred_main_raw)

    # Heatmap station predictions for BOTH stations
    b_raw = predict_one_station(
        model, preprocess, cfg, "Battaramulla", dt_base,
        overrides=req.overrides if station == "Battaramulla" else None
    )
    k_raw = predict_one_station(
        model, preprocess, cfg, "Kandy", dt_base,
        overrides=req.overrides if station == "Kandy" else None
    )

    b_pred = clamp_value(target, b_raw)
    k_pred = clamp_value(target, k_raw)

    points = [
        (STATION_META["Battaramulla"]["lat"], STATION_META["Battaramulla"]["lon"]),
        (STATION_META["Kandy"]["lat"], STATION_META["Kandy"]["lon"]),
    ]
    heatmap_b64 = make_heatmap_png_base64(
        points, [b_pred, k_pred],
        title=f"{target} (t+1) Heatmap • {pd.to_datetime(dt_base).floor('h')}",
        power=2.0,
        grid_size=190
    )

    return {
        "station": station,
        "target": target,
        "base_time": pd.to_datetime(dt_base).floor("h").strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_time": (pd.to_datetime(dt_base).floor("h") + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": float(pred_main),
        "station_preds": {"Battaramulla": float(b_pred), "Kandy": float(k_pred)},
        "heatmap_png_base64": heatmap_b64,
    }
