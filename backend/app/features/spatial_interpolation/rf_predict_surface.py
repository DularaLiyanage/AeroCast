# rf_predict_surface.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import geopandas as gpd
from shapely.geometry import Point

DATA = Path("clean/cea_hourly_2019_2024_clean.csv")
ART  = Path("ml_baseline")
OUT  = Path("outputs_rf_surfaces"); OUT.mkdir(exist_ok=True)

def add_time_features(df):
    dt = pd.to_datetime(df["datetime"])
    df["hour"]  = dt.dt.hour
    df["dow"]   = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def add_wind_components(df):
    rad = np.deg2rad(df["WD"])
    df["u"] = df["WS"] * np.cos(rad)
    df["v"] = df["WS"] * np.sin(rad)
    return df

def add_lag_roll(df, cols, lags=(1,3,6,24), rolls=(3,6,24)):
    df = df.sort_values(["station","datetime"]).copy()
    for col in cols:
        for L in lags: df[f"{col}_lag{L}"] = df.groupby("station")[col].shift(L)
        for W in rolls:
            df[f"{col}_roll{W}"] = (df.groupby("station")[col]
                                      .rolling(W, min_periods=max(1, W//2)).mean()
                                      .reset_index(level=0, drop=True))
    return df

def idw_predict(xy_known, v_known, xy_query, p=2, eps=1e-9):
    out = np.empty(len(xy_query), dtype=float)
    for i, q in enumerate(xy_query):
        d = np.linalg.norm(xy_known - q, axis=1)
        j0 = np.argmin(d)
        if d[j0] < eps: out[i] = v_known[j0]; continue
        w = 1.0 / (d ** p)
        out[i] = np.sum(w * v_known) / np.sum(w)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--when", required=True)
    ap.add_argument("--target", default="PM25")
    args = ap.parse_args()

    target = args.target
    when   = pd.to_datetime(args.when)

    # load artifacts
    from joblib import load
    rf = load(ART / f"rf_{target}.joblib")
    feature_cols = json.loads((ART / f"rf_feature_cols_{target}.json").read_text())

    # load data + build features
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    df = df[["station","datetime","PM25","PM10","NO2","SO2","O3","CO","NOX",
             "WS","WD","AT","RH","BP","SolarRad","Rain","lat","lon"]].copy()
    df = add_time_features(df)
    df = add_wind_components(df)
    df = add_lag_roll(df, cols=[target,"WS","AT","RH","BP","NO2","O3"], lags=(1,3,6,24), rolls=(3,6,24))

    # pick rows at `when` (one per station if features exist)
    # feature_cols already loaded from JSON
    df = df.sort_values(["station","datetime"]).copy()

    # forward/back-fill only the features we’ll feed the model
    feat_cols_in_df = [c for c in feature_cols if c in df.columns]
    df[feat_cols_in_df] = (df.groupby("station")[feat_cols_in_df]
                            .apply(lambda g: g.ffill().bfill())
                            .reset_index(level=0, drop=True))

    # pick rows at `when`; if none, take last available <= when per station
    now_rows = df.loc[df["datetime"] == when].copy()
    if now_rows.empty:
        now_rows = (df.loc[df["datetime"] <= when]
                    .groupby("station", as_index=False)
                    .last())
        if now_rows.empty:
            raise SystemExit(f"No history found up to {when}. Try an earlier time.")


    X_now = now_rows[feature_cols]
    preds = rf.predict(X_now)

    # geodataframe in UTM 44N
    g = gpd.GeoDataFrame(
        {"station": now_rows["station"], "pred": preds},
        geometry=[Point(xy) for xy in zip(now_rows["lon"], now_rows["lat"])],
        crs=4326
    ).to_crs(32644)

    # make a ~1 km grid around the two stations
    pad = 20_000
    minx, miny, maxx, maxy = g.total_bounds
    minx -= pad; miny -= pad; maxx += pad; maxy += pad
    res = 1_000
    xs = np.arange(minx, maxx, res)
    ys = np.arange(miny, maxy, res)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    xy_known = np.c_[g.geometry.x.values, g.geometry.y.values]
    v_known  = g["pred"].to_numpy(float)
    grid_vals = idw_predict(xy_known, v_known, grid_xy, p=2)

    out = gpd.GeoDataFrame({"value": grid_vals},
                           geometry=[Point(x,y) for x,y in grid_xy],
                           crs=32644).to_crs(4326)
    fname = OUT / f"rf_idw_{target}_{str(when).replace(':','-').replace(' ','_')}_wgs84.csv"
    out.to_csv(fname, index=False)
    print(f"[OK] Saved RF→IDW surface → {fname}")
