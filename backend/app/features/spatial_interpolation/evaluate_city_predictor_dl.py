# evaluate_city_predictor_dl.py
# Evaluate trained DL city models vs a simple baseline (lag1 persistence)
#
# Run:
#   python evaluate_city_predictor_dl.py --target PM25
#   python evaluate_city_predictor_dl.py --target PM10
#   python evaluate_city_predictor_dl.py --target NO2

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

DATA = Path("clean/cea_hourly_2019_2024_clean.csv")
MODEL_DIR = Path("city_predictor_model_dl")

TARGETS = ["PM25", "PM10", "NO2"]


# ---------- feature helpers (must match training) ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    # Not used directly in num_features in your DL config, but safe to keep consistent
    if "WD" in df.columns and "WS" in df.columns:
        rad = np.deg2rad(df["WD"])
        df["u"] = df["WS"] * np.cos(rad)
        df["v"] = df["WS"] * np.sin(rad)
    return df


def one_hot_station(stations: pd.Series, categories: list[str]) -> np.ndarray:
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(stations), len(categories)), dtype=np.float32)
    vals = stations.astype(str).values
    for r, st in enumerate(vals):
        if st in cat_to_idx:
            out[r, cat_to_idx[st]] = 1.0
    return out


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = std.copy()
    std[std < 1e-8] = 1.0
    return (X - mean) / std


def make_X(df: pd.DataFrame, num_features: list[str], station_categories: list[str], mean, std) -> np.ndarray:
    X_num = df[num_features].to_numpy(np.float32)
    X_num = standardize_apply(X_num, mean, std)
    X_cat = one_hot_station(df["station"], station_categories)
    X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
    return X


# ---------- metrics ----------
def rmse(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")


def evaluate_block(df_block: pd.DataFrame, target: str, pred_col: str, baseline_col: str):
    y = df_block[target].to_numpy(float)
    p = df_block[pred_col].to_numpy(float)
    b = df_block[baseline_col].to_numpy(float)

    return {
        "MAE_model": mae(y, p),
        "RMSE_model": rmse(y, p),
        "R2_model": r2(y, p),
        "MAE_baseline": mae(y, b),
        "RMSE_baseline": rmse(y, b),
        "R2_baseline": r2(y, b),
        "n": int(len(df_block)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=TARGETS)
    args = ap.parse_args()
    target = args.target

    cfg_path = MODEL_DIR / f"city_config_{target}.json"
    model_path = MODEL_DIR / f"city_predictor_{target}.keras"

    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    cfg = json.loads(cfg_path.read_text())
    num_features = cfg["num_features"]
    station_categories = cfg["station_categories"]
    mean = np.array(cfg["normalization"]["mean"], dtype=np.float32)
    std = np.array(cfg["normalization"]["std"], dtype=np.float32)

    # Load data
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    needed_cols = ["station", "datetime", target] + list(set(num_features))
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV for evaluation: {missing}")

    df = df.sort_values(["station", "datetime"]).copy()

    # Ensure features exist (time features & lags) if not already in CSV
    # NOTE: your cleaned CSV already contains PM25/PM10/NO2 etc,
    # but not necessarily engineered time features or lag columns.
    # We'll rebuild those here safely.
    if "hour_sin" not in df.columns or "dow_cos" not in df.columns:
        df = add_time_features(df)
    df = add_wind_components(df)

    # Lag columns needed for baseline and possibly features
    if "PM25_lag1" not in df.columns:
        df["PM25_lag1"] = df.groupby("station")["PM25"].shift(1)
    if "PM10_lag1" not in df.columns:
        df["PM10_lag1"] = df.groupby("station")["PM10"].shift(1)
    if "NO2_lag1" not in df.columns:
        df["NO2_lag1"] = df.groupby("station")["NO2"].shift(1)

    # Drop NaNs for required columns (target + num_features + station)
    req = [target, "station"] + num_features
    df = df.dropna(subset=req).copy()

    # Same split logic used in training: last 6 months as test
    last_ts = df["datetime"].max()
    test_start = last_ts - pd.DateOffset(months=6)
    test_df = df[df["datetime"] >= test_start].copy()

    if test_df.empty:
        raise SystemExit("TEST split is empty. Check your CSV time range.")

    # Build X for TEST
    X_test = make_X(test_df, num_features, station_categories, mean, std)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    pred = model.predict(X_test, verbose=0).reshape(-1)

    test_df["pred_model"] = pred

    # Baseline: persistence using lag1 of the same pollutant
    baseline_map = {
        "PM25": "PM25_lag1",
        "PM10": "PM10_lag1",
        "NO2":  "NO2_lag1",
    }
    base_col = baseline_map[target]
    test_df["pred_baseline"] = test_df[base_col].astype(float)

    # Overall metrics
    overall = evaluate_block(test_df, target, "pred_model", "pred_baseline")
    print("\n==============================")
    print(f"TARGET: {target}")
    print(f"TEST range: {test_df['datetime'].min()}  →  {test_df['datetime'].max()}")
    print(f"Rows: {overall['n']}")
    print("------------------------------")
    print("MODEL:")
    print(f"  MAE :  {overall['MAE_model']:.4f}")
    print(f"  RMSE:  {overall['RMSE_model']:.4f}")
    print(f"  R²  :  {overall['R2_model']:.4f}")
    print("BASELINE (lag1 persistence):")
    print(f"  MAE :  {overall['MAE_baseline']:.4f}")
    print(f"  RMSE:  {overall['RMSE_baseline']:.4f}")
    print(f"  R²  :  {overall['R2_baseline']:.4f}")
    print("==============================\n")

    # Per-station metrics
    rows = []
    for st, g in test_df.groupby("station"):
        m = evaluate_block(g, target, "pred_model", "pred_baseline")
        m["station"] = st
        rows.append(m)

    per_station = pd.DataFrame(rows).sort_values("station")
    print("Per-station metrics (TEST):")
    print(per_station[[
        "station", "n",
        "MAE_model", "RMSE_model", "R2_model",
        "MAE_baseline", "RMSE_baseline", "R2_baseline"
    ]].to_string(index=False))

    # Save to CSV
    out_csv = MODEL_DIR / f"eval_{target}_test_metrics.csv"
    per_station.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved per-station metrics -> {out_csv}")

    # Save a small sample of predictions for inspection
    sample_csv = MODEL_DIR / f"eval_{target}_test_sample_preds.csv"
    test_df[["datetime", "station", target, "pred_model", "pred_baseline"]].head(5000).to_csv(sample_csv, index=False)
    print(f"[OK] Saved sample predictions -> {sample_csv}")


if __name__ == "__main__":
    main()
