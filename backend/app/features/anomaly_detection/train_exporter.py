import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
import os

def train_and_save_model(location_name, data_path="Pollution_Research_Hourly_Series.parquet", save_dir="backend/models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_full = pd.read_parquet(data_path)
    # Filter for specific location
    print(f"Filtering data for {location_name}...")
    df = df_full[df_full['Location'] == location_name].copy()
    
    if df.empty:
        print(f"No data found for {location_name}, skipping.")
        return

    # Define features based on 1.5 notebook
    features_to_use = [
        'AT', 'RH', 'BP', 'Solar Rad', 'Rain Gauge', 'WD Raw', 
        'O3 Conc', 'NO Conc', 'NO2 Conc', 'NOx Conc', 'SO2 Conc', 'PM2.5 Conc', 'PM10 Conc'
    ]
    targets = ['PM2.5 Conc', 'PM10 Conc', 'O3 Conc', 'SO2 Conc']
    target_indices = [features_to_use.index(t) for t in targets]

    # Scale the data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features_to_use])
    
    prefix = location_name.lower().replace(" ", "_")
    joblib.dump(scaler, f"{save_dir}/{prefix}_scaler.joblib")

    def create_sequences(data, window_size, horizon, target_indices):
        X, y = [], []
        for i in range(len(data) - window_size - horizon):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size + horizon, target_indices])
        return np.array(X), np.array(y)

    # Window size 24h, Predict 7 days (168h) ahead
    X, y = create_sequences(df_scaled, 24, 168, target_indices)

    if len(X) == 0:
        print(f"Not enough data to create sequences for {location_name}.")
        return

    # Simple split
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    # Build model (same architecture as notebook 1.5)
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        GRU(units=64, return_sequences=True),
        Dropout(0.2),
        GRU(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=len(targets))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Training model for {location_name}...")
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

    model.save(f"{save_dir}/{prefix}_gru_model.h5")
    
    # Save a small portion of X_train for SHAP background data
    np.save(f"{save_dir}/{prefix}_background_data.npy", X_train[:100])
    
    print(f"Model, scaler, and background data saved for {location_name}")

if __name__ == "__main__":
    for loc in ["Battaramulla", "Kandy"]:
        train_and_save_model(loc)
