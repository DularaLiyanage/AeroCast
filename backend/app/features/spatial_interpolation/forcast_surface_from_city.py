# forecast_surface_from_city.py
# LSTM forecast → IDW spatial surface

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tensorflow.keras.models import load_model

# -----------------------------
# Paths
# -----------------------------
DATA = Path("clean/cea_hourly_2019_2024_clean.csv")
MODEL = Path("city_predictor_dl/lstm_PM25")
OUT = Path("outputs_city_forecast_surfaces")
OUT.mkdir(exist_ok=True)

WINDOW = 24
FEATURES = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","AT","RH","BP","lat","lon"
]

# -----------------------------
# IDW
# -----------------------------
def idw_predict(xy_known, v_known, xy_query, p=2):
    out = np.empty(len(xy_query))
    for i, q in enumerate(xy_query):
        d = np.linalg.norm(xy_known - q, axis=1)
        w = 1.0 / (d ** p)
        out[i] = np.sum(w * v_known) / np.sum(w)
    return out

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--when", required=True)
    args = ap.parse_args()

    base_time = pd.to_datetime(args.when)

    df = pd.read_csv(DATA, parse_dates=["datetime"])
    model = load_model(MODEL)

    preds = []

    for station, g in df.groupby("station"):
        g = g.sort_values("datetime")
        g = g[g["datetime"] <= base_time].tail(WINDOW)

        if len(g) < WINDOW:
            continue

        X = g[FEATURES].values.reshape(1, WINDOW, len(FEATURES))
        pred = model.predict(X)[0, 0]

        preds.append({
            "station": station,
            "pred": pred,
            "lat": g.iloc[-1]["lat"],
            "lon": g.iloc[-1]["lon"]
        })

    gdf = gpd.GeoDataFrame(
        preds,
        geometry=[Point(xy) for xy in zip(
            [p["lon"] for p in preds],
            [p["lat"] for p in preds]
        )],
        crs="EPSG:4326"
    ).to_crs(32644)

    pad = 20_000
    minx, miny, maxx, maxy = gdf.total_bounds
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    xs = np.arange(minx, maxx, 1000)
    ys = np.arange(miny, maxy, 1000)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    xy_known = np.c_[gdf.geometry.x, gdf.geometry.y]
    v_known = gdf["pred"].values

    grid_vals = idw_predict(xy_known, v_known, grid)

    out = gpd.GeoDataFrame(
        {"value": grid_vals},
        geometry=[Point(xy) for xy in grid],
        crs=32644
    ).to_crs(4326)

    fname = OUT / f"forecast_PM25_{str(base_time).replace(':','-').replace(' ','_')}.csv"
    out.to_csv(fname, index=False)

    print(f"[OK] Saved forecast surface → {fname}")
