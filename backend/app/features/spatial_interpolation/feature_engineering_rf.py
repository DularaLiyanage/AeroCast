# feature_engineering_rf.py
from pathlib import Path
import pandas as pd, numpy as np, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV = Path("clean/cea_hourly_2019_2024_clean.csv")
OUT = Path("ml_baseline"); OUT.mkdir(exist_ok=True)

TARGET = "PM25"  # change to PM10 later

def add_time_features(df):
    dt = pd.to_datetime(df["datetime"])
    df["hour"]  = dt.dt.hour
    df["dow"]   = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # cyclical encodings
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
        for L in lags:
            df[f"{col}_lag{L}"] = df.groupby("station")[col].shift(L)
        for W in rolls:
            df[f"{col}_roll{W}"] = (df.groupby("station")[col]
                                      .rolling(W, min_periods=max(1, W//2)).mean()
                                      .reset_index(level=0, drop=True))
    return df

def metrics(y, p, tag):
    # version-proof RMSE (works even if sklearn's mean_squared_error lacks squared=False)
    mse = mean_squared_error(y, p)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, p))
    r2  = float(r2_score(y, p))
    return {"tag": tag, "MAE": mae, "RMSE": rmse, "R2": r2}

# -------- load --------
df = pd.read_csv(CSV, parse_dates=["datetime"])
keep = ["station","datetime","PM25","PM10","NO2","SO2","O3","CO","NOX",
        "WS","WD","AT","RH","BP","SolarRad","Rain","lat","lon"]
df = df[keep].copy()

# -------- features --------
df = add_time_features(df)
df = add_wind_components(df)
df = add_lag_roll(df, cols=[TARGET,"WS","AT","RH","BP","NO2","O3"], lags=(1,3,6,24), rolls=(3,6,24))
df = df.dropna().reset_index(drop=True)

# -------- splits by time --------
last_date = df["datetime"].max()
test_start = (last_date - pd.DateOffset(months=6))
val_start  = (test_start - pd.DateOffset(months=6))

train = df[df["datetime"] < val_start]
val   = df[(df["datetime"] >= val_start) & (df["datetime"] < test_start)]
test  = df[df["datetime"] >= test_start]

exclude_raw_targets = ["PM25","PM10","NO2","SO2","O3","CO","NOX"]
feature_cols = [c for c in df.columns if c not in (["datetime","station","lat","lon"] + exclude_raw_targets)]

X_train, y_train = train[feature_cols], train[TARGET]
X_val,   y_val   = val[feature_cols],   val[TARGET]
X_test,  y_test  = test[feature_cols],  test[TARGET]

# -------- model --------
rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=4,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(pd.concat([X_train,X_val]), pd.concat([y_train,y_val]))

# -------- eval --------
import pandas as pd
res = []
res.append(metrics(y_val,  rf.predict(X_val),  "VAL"))
res.append(metrics(y_test, rf.predict(X_test), "TEST"))
pd.DataFrame(res).to_csv(OUT / f"rf_metrics_{TARGET}.csv", index=False)

# per-station on TEST
rows=[]
for stn, d in test.groupby("station"):
    rows.append(metrics(d[TARGET], rf.predict(d[feature_cols]), f"TEST_{stn}"))
pd.DataFrame(rows).to_csv(OUT / f"rf_metrics_by_station_{TARGET}.csv", index=False)

# feature importance
imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})\
        .sort_values("importance", ascending=False)
imp.to_csv(OUT / f"rf_feature_importance_{TARGET}.csv", index=False)

# save artifacts
dump(rf, OUT / f"rf_{TARGET}.joblib")
with open(OUT / f"rf_feature_cols_{TARGET}.json","w") as f:
    json.dump(feature_cols, f)

print("[OK] RF trained and saved. See ml_baseline/")
