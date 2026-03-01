# train_city_predictor.py

from pathlib import Path
import pandas as pd, numpy as np, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
CSV = Path("clean/cea_hourly_2019_2024_clean.csv")   # cleaned data from app.py
OUT = Path("city_predictor_model")
OUT.mkdir(exist_ok=True)

# We will train 3 separate models: PM25, PM10, NO2
TARGETS = ["PM25", "PM10", "NO2"]

# City metadata (for spatial differences)
STATION_META = {
    "Battaramulla": {"lat": 6.901035, "lon": 79.926513},
    "Kandy":        {"lat": 7.292651, "lon": 80.635649},
}

# ------------------------------------------------------------------
# Feature Functions
# ------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df["hour"]  = dt.dt.hour
    df["dow"]   = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"] / 24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"] / 7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"] / 7.0)
    return df

def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    rad = np.deg2rad(df["WD"])
    df["u"] = df["WS"] * np.cos(rad)
    df["v"] = df["WS"] * np.sin(rad)
    return df

# ------------------------------------------------------------------
# Load + Prepare Data (common for all targets)
# ------------------------------------------------------------------
df = pd.read_csv(CSV, parse_dates=["datetime"])

base_features = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
]

df = df[["station","datetime"] + base_features].copy()

# Add lat/lon per station (city)
df["lat"] = df["station"].map(lambda s: STATION_META[s]["lat"])
df["lon"] = df["station"].map(lambda s: STATION_META[s]["lon"])

# Add time & wind features
df = add_time_features(df)
df = add_wind_components(df)

# Add lag1 for the three pollutants we might predict
df = df.sort_values(["station", "datetime"]).copy()
df["PM25_lag1"] = df.groupby("station")["PM25"].shift(1)
df["PM10_lag1"] = df.groupby("station")["PM10"].shift(1)
df["NO2_lag1"]  = df.groupby("station")["NO2"].shift(1)

# Drop rows with any NaN in features (after lagging)
df = df.dropna().reset_index(drop=True)

# ------------------------------------------------------------------
# Train/Test Split (same split used for all targets)
# ------------------------------------------------------------------
last_ts = df["datetime"].max()
test_start = last_ts - pd.DateOffset(months=6)

train = df[df["datetime"] < test_start].copy()
test  = df[df["datetime"] >= test_start].copy()

# Numeric & categorical feature lists
num_features = base_features + [
    "PM25_lag1","PM10_lag1","NO2_lag1",
    "hour","dow","month","is_weekend",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "lat","lon",
]
cat_features = ["station"]

# Preprocessing (shared)
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# ------------------------------------------------------------------
# Helper to train one model
# ------------------------------------------------------------------
def train_one_target(target: str):
    print(f"\n[INFO] Training model for target = {target}")

    model = RandomForestRegressor(
        n_estimators=450,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    X_train = train[num_features + cat_features]
    y_train = train[target]

    pipe.fit(X_train, y_train)

    # Save model + config
    model_path = OUT / f"city_predictor_{target}.joblib"
    cfg_path   = OUT / f"city_config_{target}.json"

    dump(pipe, model_path)

    with open(cfg_path, "w") as f:
        json.dump({
            "num_features": num_features,
            "cat_features": cat_features,
            "target": target
        }, f, indent=2)

    print(f"[OK] Saved model → {model_path}")
    print(f"[OK] Saved config → {cfg_path}")

# ------------------------------------------------------------------
# Main: train all three models
# ------------------------------------------------------------------
if __name__ == "__main__":
    for tgt in TARGETS:
        train_one_target(tgt)

    print("\n[DONE] Trained models for:", ", ".join(TARGETS))
