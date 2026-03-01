# ------------------------------------------------------------
# Deep Learning (LSTM) CITY predictor for ALL variables:
#   PM25, PM10, NO2, SO2, O3, CO, NOX, WS, WD, AT, RH, BP, SolarRad, Rain
#
# Predicts 1-hour ahead (lead1): value at t+1
#
# Uses:
#   clean/cea_hourly_2019_2024_clean_dl_md.csv
#
# Saves:
#   city_predictor_model_dl_md/
#     dl_city_predictor_<TARGET>_lead1.keras
#     dl_preprocess_<TARGET>_lead1.joblib
#     dl_config_<TARGET>_lead1.json
#
# Run:
#   (airvenv) python train_city_predictor_dl.py
# ------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------
# Settings
# -----------------------------
CSV = Path("clean/cea_hourly_2019_2024_clean_dl_md.csv")
OUT = Path("city_predictor_model_dl_md")
OUT.mkdir(exist_ok=True)

# ✅ Train a model per target (all variables)
BASE_TARGETS = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
]

STATION_META = {
    "Battaramulla": {"lat": 6.901035, "lon": 79.926513},
    "Kandy":        {"lat": 7.292651, "lon": 80.635649},
}

# Physical plausibility bounds (used only for optional clamping in UI; training uses raw cleaned)
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

WINDOW = 24
TEST_MONTHS = 6
BATCH = 256
EPOCHS = 40
SEED = 42

# Lags / rolls
LAGS  = (1, 3, 6, 12, 24)
ROLLS = (3, 6, 24)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Feature helpers
# -----------------------------
def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    # WD degrees, WS m/s
    rad = np.deg2rad(df["WD"].astype(float))
    df["u"] = df["WS"].astype(float) * np.cos(rad)
    df["v"] = df["WS"].astype(float) * np.sin(rad)
    return df

def add_lag_roll_features(df: pd.DataFrame, cols, lags=LAGS, rolls=ROLLS) -> pd.DataFrame:
    """
    Adds lag and rolling mean features per station.
    Rolling means are shifted by 1 hour so they use past-only values.
    """
    df = df.sort_values(["station", "datetime"]).copy()

    for col in cols:
        if col not in df.columns:
            continue

        for L in lags:
            df[f"{col}_lag{L}"] = df.groupby("station")[col].shift(L)

        for W in rolls:
            past = df.groupby("station")[col].shift(1)
            df[f"{col}_roll{W}"] = (
                past.groupby(df["station"])
                    .rolling(W, min_periods=max(1, W // 2))
                    .mean()
                    .reset_index(level=0, drop=True)
            )
    return df

def metrics(y, p):
    mse = mean_squared_error(y, p)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, p))
    r2  = float(r2_score(y, p))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def seq_from_arrays(X: np.ndarray, y: np.ndarray, window: int):
    if len(y) < window:
        return None, None
    Xs, ys = [], []
    for i in range(window - 1, len(y)):
        Xs.append(X[i - window + 1 : i + 1])
        ys.append(y[i])
    return np.stack(Xs), np.array(ys)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV, parse_dates=["datetime"])

needed = [
    "station", "datetime", "lat", "lon",
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
    "month","day",
    "hour","hour_sin","hour_cos"
]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in CSV: {missing}. Run app.py cleaner first.")

df = df[needed].copy()

# enforce stable lat/lon from metadata
df["lat"] = df["station"].map(lambda s: STATION_META.get(s, {}).get("lat", np.nan))
df["lon"] = df["station"].map(lambda s: STATION_META.get(s, {}).get("lon", np.nan))
df = df.dropna(subset=["lat", "lon"]).copy()

# wind components (u,v)
df = add_wind_components(df)

# -----------------------------
# Lag/Roll columns (use ALL variables)
# -----------------------------
LAG_ROLL_COLS = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
]
df = add_lag_roll_features(df, cols=LAG_ROLL_COLS, lags=LAGS, rolls=ROLLS)

# -----------------------------
# Lead1 targets for ALL variables
# -----------------------------
df = df.sort_values(["station", "datetime"]).copy()
for t in BASE_TARGETS:
    df[f"{t}_lead1"] = df.groupby("station")[t].shift(-1)

# Drop NaNs created by lag/roll/lead
df = df.dropna().reset_index(drop=True)

# -----------------------------
# Per-station split (last N months each station)
# -----------------------------
train_parts, test_parts = [], []
split_info = {}

for stn, g in df.groupby("station"):
    g = g.sort_values("datetime").copy()
    last_ts = g["datetime"].max()
    test_start = last_ts - pd.DateOffset(months=TEST_MONTHS)

    tr = g[g["datetime"] < test_start].copy()
    te = g[g["datetime"] >= test_start].copy()

    train_parts.append(tr)
    test_parts.append(te)

    split_info[stn] = {
        "last_ts": str(last_ts),
        "test_start": str(test_start),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
    }

train_df = pd.concat(train_parts, ignore_index=True)
test_df  = pd.concat(test_parts, ignore_index=True)

print("[INFO] Train rows by station:", train_df["station"].value_counts().to_dict())
print("[INFO] Test rows by station :", test_df["station"].value_counts().to_dict())
print("[INFO] WINDOW =", WINDOW)

# -----------------------------
# Build numeric features (shared for all targets)
# -----------------------------
engineered = []
for c in LAG_ROLL_COLS:
    for L in LAGS:
        engineered.append(f"{c}_lag{L}")
    for W in ROLLS:
        engineered.append(f"{c}_roll{W}")

base_features = [
    "PM25","PM10","NO2","SO2","O3","CO","NOX",
    "WS","WD","AT","RH","BP","SolarRad","Rain",
    "u","v",
    "lat","lon",
    "month","day",
    "hour_sin","hour_cos"
]

num_features = base_features + engineered
num_features = [c for c in num_features if c in train_df.columns]
cat_features = ["station"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="drop"
)

preprocess.fit(train_df[num_features + cat_features])

X_train_flat = preprocess.transform(train_df[num_features + cat_features]).astype(np.float32)
X_test_flat  = preprocess.transform(test_df[num_features + cat_features]).astype(np.float32)

F = X_train_flat.shape[1]
print("[INFO] Final feature dim F =", F)

train_df2 = train_df[["station","datetime"]].copy()
test_df2  = test_df[["station","datetime"]].copy()
train_df2["__X"] = list(X_train_flat)
test_df2["__X"]  = list(X_test_flat)

# -----------------------------
# Train one model per target
# -----------------------------
def train_one_target(base_target: str):
    target = f"{base_target}_lead1"
    print(f"\n[DL] Training target = {target}")

    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []

    stations = sorted(set(train_df2["station"].unique()) | set(test_df2["station"].unique()))

    for stn in stations:
        tr_block = train_df2[train_df2["station"] == stn].sort_values("datetime").copy()
        te_block = test_df2[test_df2["station"] == stn].sort_values("datetime").copy()

        tr_y = train_df[train_df["station"] == stn].sort_values("datetime")[target].to_numpy(dtype=np.float32)
        te_y = test_df[test_df["station"] == stn].sort_values("datetime")[target].to_numpy(dtype=np.float32)

        if len(tr_block) < WINDOW or len(te_block) < WINDOW:
            print(f"[WARN] Skip {stn}: train_rows={len(tr_block)} test_rows={len(te_block)} < WINDOW={WINDOW}")
            continue

        tr_X = np.stack(tr_block["__X"].to_list())
        te_X = np.stack(te_block["__X"].to_list())

        m_tr = min(len(tr_X), len(tr_y))
        m_te = min(len(te_X), len(te_y))
        tr_X, tr_y = tr_X[:m_tr], tr_y[:m_tr]
        te_X, te_y = te_X[:m_te], te_y[:m_te]

        Xs_tr, ys_tr = seq_from_arrays(tr_X, tr_y, WINDOW)
        Xs_te, ys_te = seq_from_arrays(te_X, te_y, WINDOW)

        if Xs_tr is None or Xs_te is None:
            print(f"[WARN] Skip {stn}: sequence build failed")
            continue

        Xtr_list.append(Xs_tr); ytr_list.append(ys_tr)
        Xte_list.append(Xs_te); yte_list.append(ys_te)

        print(f"[OK] {stn}: train_seq={Xs_tr.shape} test_seq={Xs_te.shape}")

    if len(Xtr_list) == 0 or len(Xte_list) == 0:
        raise SystemExit("No valid sequences built. Reduce WINDOW or check station coverage.")

    X_train = np.concatenate(Xtr_list, axis=0)
    y_train = np.concatenate(ytr_list, axis=0)
    X_test  = np.concatenate(Xte_list, axis=0)
    y_test  = np.concatenate(yte_list, axis=0)

    print("[DL] X_train:", X_train.shape, "y_train:", y_train.shape)
    print("[DL] X_test :", X_test.shape,  "y_test :", y_test.shape)

    model = models.Sequential([
        layers.Input(shape=(WINDOW, F)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cb,
        verbose=1
    )

    pred = model.predict(X_test, verbose=0).reshape(-1)
    m = metrics(y_test, pred)
    print("[DL] TEST metrics:", m)

    # Save
    model_path = OUT / f"dl_city_predictor_{base_target}_lead1.keras"
    prep_path  = OUT / f"dl_preprocess_{base_target}_lead1.joblib"
    cfg_path   = OUT / f"dl_config_{base_target}_lead1.json"

    model.save(model_path)
    dump(preprocess, prep_path)

    with open(cfg_path, "w") as f:
        json.dump({
            "base_target": base_target,
            "target": target,
            "horizon_hours": 1,
            "window": WINDOW,
            "test_months": TEST_MONTHS,
            "lags": list(LAGS),
            "rolls": list(ROLLS),
            "lag_roll_cols": LAG_ROLL_COLS,
            "num_features": num_features,
            "cat_features": cat_features,
            "final_feature_dim": int(F),
            "split_info": split_info,
            "bounds": {k: [float(v[0]), float(v[1])] for k, v in BOUNDS.items()},
            "time_features": {"month": True, "day": True, "hour": "sin_cos"},
        }, f, indent=2)

    print(f"[OK] Saved DL model  → {model_path}")
    print(f"[OK] Saved preprocess→ {prep_path}")
    print(f"[OK] Saved config    → {cfg_path}")

if __name__ == "__main__":
    for tgt in BASE_TARGETS:
        train_one_target(tgt)

    print("\n[DONE] Trained DL lead1 models for:", ", ".join(BASE_TARGETS))
