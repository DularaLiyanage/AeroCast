# app.py
# ------------------------------------------------------------
# DL-ready Cleaner (Month+Day only, no Hour feature generation)
#
# - Reads Battaramulla/Kandy Excel/CSV files in the same folder.
# - Standardizes columns -> creates continuous hourly timeline per station.
# - Cleans impossible values with bounds.
# - Fills missing values in a DL-friendly way:
#     1) interpolate small gaps (<= MAX_GAP_HOURS)
#     2) limited forward-fill/back-fill (<= FILL_LIMIT_HOURS)
#     3) final fallback station median
#
# - Adds ONLY:
#     month (1..12)
#     day   (1..31)
#
# Output:
#   clean/cea_hourly_2019_2024_clean_dl_md.csv
#
# Run:
#   (airvenv) python app.py
# ------------------------------------------------------------

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
HERE = Path(__file__).parent.resolve()
DATA_DIR = HERE

OUT_DIR = HERE / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Interpolate only small gaps (hours)
MAX_GAP_HOURS = 3

# After interpolation, fill longer gaps but only up to this many hours
FILL_LIMIT_HOURS = 24

# Physical plausibility bounds (adjust if units differ)
BOUNDS = {
    "PM25":  (0, 1000),
    "PM10":  (0, 1500),
    "NO2":   (0, 1000),
    "SO2":   (0, 1000),
    "O3":    (0, 1000),
    "CO":    (0, 50000),
    "NOX":   (0, 2000),
    "WS":    (0, 60),
    "WD":    (0, 360),
    "AT":    (-5, 50),
    "RH":    (0, 100),
    "BP":    (800, 1100),
    "SolarRad": (0, 1400),
    "Rain":  (0, 500),
}

STATION_META = {
    "Battaramulla": {"lat": 6.901035, "lon": 79.926513},
    "Kandy":        {"lat": 7.292651, "lon": 80.635649},
}

# Final columns used by the new DL model
SCHEMA = [
    "station", "lat", "lon", "datetime",
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
]

# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

COLMAP = {
    # datetime
    "period start time": "datetime",
    "period start": "datetime",
    "date time": "datetime",
    "timestamp": "datetime",
    "datetime": "datetime",
    "date": "datetime",

    # PM / gases
    "pm2.5 conc": "PM25", "pm2.5": "PM25", "pm 2.5": "PM25", "pm₂.₅": "PM25", "pm2.5 concentration": "PM25",
    "pm10 conc": "PM10",  "pm10": "PM10",  "pm 10": "PM10",  "pm10 concentration": "PM10",
    "no2 conc": "NO2",    "no2": "NO2",    "no2 concentration": "NO2",
    "so2 conc": "SO2",    "so2": "SO2",    "so2 concentration": "SO2",
    "o3 conc": "O3",      "o3": "O3",      "o3 concentration": "O3",
    "co conc": "CO",      "co": "CO",      "co concentration": "CO",
    "nox conc": "NOX",    "nox": "NOX",    "nox concentration": "NOX",

    # met
    "ws average": "WS", "ws": "WS", "wind speed": "WS", "windspeed": "WS",
    "wd average": "WD", "wd": "WD", "wind direction": "WD", "winddirection": "WD",
    "at": "AT", "ambient temperature": "AT", "temperature": "AT",
    "rh": "RH", "relative humidity": "RH", "humidity": "RH",
    "bp": "BP", "barometric pressure": "BP", "pressure": "BP",
    "solar rad": "SolarRad", "solar radiation": "SolarRad", "solar": "SolarRad",
    "rain gauge": "Rain", "rain": "Rain", "rainfall": "Rain",
}

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for c in df.columns:
        nc = norm(c)
        if nc in COLMAP:
            m[c] = COLMAP[nc]
    return df.rename(columns=m)

def find_all_data_files(base: Path) -> list[Path]:
    return [p for p in base.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")]

def station_from_filename(path: Path) -> str | None:
    n = path.name.lower()
    if "battaramulla" in n:
        return "Battaramulla"
    if "kandy" in n:
        return "Kandy"
    return None

def read_any(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path.name}: {e}")
        return pd.DataFrame()

def clip_bounds(df: pd.DataFrame, col: str) -> pd.DataFrame:
    low, high = BOUNDS[col]
    bad = (df[col] < low) | (df[col] > high)
    if bad.any():
        df.loc[bad, col] = np.nan
    return df

def standardize_hourly(path: Path) -> pd.DataFrame:
    raw = read_any(path)
    if raw.empty:
        return raw

    df = rename_columns(raw).copy()

    stn = station_from_filename(path)
    if stn is None:
        print(f"[WARN] Could not infer station from filename: {path.name} (skipping)")
        return pd.DataFrame()

    df["station"] = stn
    df["lat"] = STATION_META[stn]["lat"]
    df["lon"] = STATION_META[stn]["lon"]

    if "datetime" not in df.columns:
        print(f"[WARN] No datetime column in {path.name} (skipping)")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()

    # Hour alignment (indexing only; we won't create hour features)
    df["datetime"] = df["datetime"].dt.floor("h")

    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan

    out = df[SCHEMA].copy()

    # numeric coercion
    for c in SCHEMA:
        if c in ("station", "datetime"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def fill_station_block(g: pd.DataFrame) -> pd.DataFrame:
    """
    - Continuous hourly timeline
    - Small-gap interpolation
    - Limited ffill/bfill
    - Final fallback station median
    """
    g = g.sort_values("datetime").copy()
    g = g.drop_duplicates(subset=["datetime"], keep="last")
    g = g.set_index("datetime")

    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
    g = g.reindex(full_idx)

    g["station"] = g["station"].ffill().bfill()
    g["lat"] = g["lat"].ffill().bfill()
    g["lon"] = g["lon"].ffill().bfill()

    numeric_cols = ["PM25","PM10","NO2","SO2","O3","CO","NOX","WS","WD","AT","RH","BP","SolarRad","Rain"]

    for col in numeric_cols:
        if col not in g.columns:
            g[col] = np.nan

        # bounds clip before fill
        if col in BOUNDS:
            g = clip_bounds(g, col)

        # small gap interpolation
        g[col] = g[col].interpolate("time", limit=MAX_GAP_HOURS, limit_direction="both")

        # limited ffill/bfill for longer gaps
        g[col] = g[col].ffill(limit=FILL_LIMIT_HOURS).bfill(limit=FILL_LIMIT_HOURS)

        # final fallback median
        med = float(pd.to_numeric(g[col], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.0
        g[col] = g[col].fillna(med)

        # bounds clip after fill too
        if col in BOUNDS:
            g = clip_bounds(g, col)

    g = g.reset_index().rename(columns={"index": "datetime"})
    return g

def quick_summary(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"[SUM] {label}: 0 rows")
        return
    print(f"[SUM] {label}: {len(df):,} rows")
    print("      stations:", df["station"].unique().tolist())
    print(f"      time range: {df['datetime'].min()} → {df['datetime'].max()}")

def main():
    all_files = find_all_data_files(DATA_DIR)
    if not all_files:
        print(f"ERROR: No .xlsx/.xls/.csv found in {DATA_DIR}")
        sys.exit(1)

    hourly_files = [p for p in all_files if station_from_filename(p) is not None]

    print("----- DISCOVERY -----")
    print(f"Base folder: {DATA_DIR}")
    print(f"Total files found: {len(all_files)}")
    print(f"Hourly candidates (Battaramulla/Kandy): {len(hourly_files)}")
    for f in hourly_files:
        print("  -", f.name)
    print("---------------------")

    if not hourly_files:
        print("[ERROR] No Battaramulla/Kandy files detected by filename.")
        sys.exit(1)

    parts = []
    for f in hourly_files:
        d = standardize_hourly(f)
        if d.empty:
            print(f"[WARN] Skipped: {f.name}")
            continue
        for col in BOUNDS:
            if col in d.columns:
                d = clip_bounds(d, col)
        parts.append(d)
        print(f"[OK] Loaded: {f.name} rows={len(d):,}")

    if not parts:
        print("[ERROR] No usable tables parsed.")
        sys.exit(1)

    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.sort_values(["station", "datetime"])
             .drop_duplicates(["station", "datetime"], keep="last")
             .reset_index(drop=True)
    )

    hourly_clean = (
        hourly.groupby("station", group_keys=False)
              .apply(fill_station_block)
              .reset_index(drop=True)
    )

    # Add ONLY month/day (no hour)
    dt = pd.to_datetime(hourly_clean["datetime"])
    hourly_clean["month"] = dt.dt.month.astype(int)
    hourly_clean["day"] = dt.dt.day.astype(int)

    hourly_clean["hour"] = dt.dt.hour.astype(int)
    hourly_clean["hour_sin"] = np.sin(2 * np.pi * hourly_clean["hour"] / 24)
    hourly_clean["hour_cos"] = np.cos(2 * np.pi * hourly_clean["hour"] / 24)

    out_path = OUT_DIR / "cea_hourly_2019_2024_clean_dl_md.csv"
    hourly_clean.to_csv(out_path, index=False)

    print(f"\n[OK] Saved DL-ready dataset (month+day only) → {out_path}")
    quick_summary(hourly_clean, "hourly_clean_md")

    # Missing report (should be near zero after filling)
    numeric_cols = ["PM25","PM10","NO2","SO2","O3","CO","NOX","WS","WD","AT","RH","BP","SolarRad","Rain"]
    miss = (hourly_clean[numeric_cols].isna().mean() * 100.0).sort_values(ascending=False)
    print("\n[INFO] Missing % by column (after cleaning):")
    print(miss.round(2).to_string())

if __name__ == "__main__":
    main()
