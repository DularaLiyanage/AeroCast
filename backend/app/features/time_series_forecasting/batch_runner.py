import joblib
import torch
import pandas as pd
import numpy as np
import gc
import json
import datetime
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
import holidays
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

import warnings
warnings.filterwarnings("ignore")

# Config - Use absolute paths based on this module's location
current_file_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_file_dir, "../../"))

model_path = os.path.join(backend_dir, "models", "time_series_forecasting")
forecast_dir = os.path.join(backend_dir, "forecast", "time_series_forecasting")
output_file = os.path.join(forecast_dir, "daily_forecast.json")

# Ensure output directory exists
os.makedirs(forecast_dir, exist_ok=True)

locations = ["kandy", "baththaramulla"]
cordinates = {
    "kandy": {"lat": 7.2906, "lon": 80.6337},
    "baththaramulla": {"lat": 6.9271, "lon": 79.8612} 
}
pollutants = ['PM2 5 Conc', 'PM10 Conc', 'NO2 Conc', 'O3 Conc', 'SO2 Conc'] 
forecast_horizon = 24

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Function to get AQ history with lag and rolling features
def build_live_inference_matrix(location_name, forecast_horizon=24):
    """
    Builds a continuous 192-hour dataframe (168h past + 24h future) 
    directly from Open-Meteo APIs, completely eliminating the need for old pickle files.
    """
    coords = cordinates.get(location_name.lower(), cordinates["baththaramulla"])
    
    # 1. Define the exact temporal window
    today = datetime.date.today()
    forecast_date = today + datetime.timedelta(days=1)
    start_date = today - datetime.timedelta(days=8)
    
    print(f"Building continuous 192-hour matrix for {location_name.upper()}...")
    
    # 2. Fetch Weather (Known Inputs)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": forecast_date.strftime("%Y-%m-%d")
    }
    w_resp = openmeteo.weather_api(weather_url, params=weather_params)[0].Hourly()
    
    weather_df = pd.DataFrame({
        "Date": pd.date_range(
            start=pd.to_datetime(w_resp.Time(), unit="s", utc=True),
            end=pd.to_datetime(w_resp.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=w_resp.Interval()),
            inclusive="left"
        ),
        "AT": w_resp.Variables(0).ValuesAsNumpy(),
        "RH": w_resp.Variables(1).ValuesAsNumpy(),
        "Rain Gauge": w_resp.Variables(2).ValuesAsNumpy(),
        "BP": w_resp.Variables(3).ValuesAsNumpy(),
        "wind_speed": w_resp.Variables(4).ValuesAsNumpy(),
        "wind_deg": w_resp.Variables(5).ValuesAsNumpy()
    })

    # 3. Fetch Air Quality (Observed Inputs)
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "sulphur_dioxide"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": forecast_date.strftime("%Y-%m-%d")
    }
    aq_resp = openmeteo.weather_api(aq_url, params=aq_params)[0].Hourly()
    
    aq_df = pd.DataFrame({
        "Date": pd.date_range(
            start=pd.to_datetime(aq_resp.Time(), unit="s", utc=True),
            end=pd.to_datetime(aq_resp.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=aq_resp.Interval()),
            inclusive="left"
        ),
        "PM10 Conc": aq_resp.Variables(0).ValuesAsNumpy(),
        "PM2 5 Conc": aq_resp.Variables(1).ValuesAsNumpy(),
        "NO2 Conc": aq_resp.Variables(2).ValuesAsNumpy(),
        "O3 Conc": aq_resp.Variables(3).ValuesAsNumpy(),
        "SO2 Conc": aq_resp.Variables(4).ValuesAsNumpy()
    })

    # 4. Merge DataFrames
    df = pd.merge(weather_df, aq_df, on="Date", how="inner")

    # 5. PREVENT DATA LEAKAGE: Mask Open-Meteo's future AQ predictions
    future_mask = df['Date'].dt.date >= forecast_date
    pollutants_list = ["PM10 Conc", "PM2 5 Conc", "NO2 Conc", "O3 Conc", "SO2 Conc"]
    df.loc[future_mask, pollutants_list] = np.nan

    # 6. Time-Series Feature Engineering
    for p in pollutants_list:
        df[f"{p}_lag24"] = df[p].shift(24)
        df[f"{p}_rolling24_mean"] = df[p].rolling(window=24).mean()
        df[f"{p}_rolling24_mean"] = df[f"{p}_rolling24_mean"].ffill()

    # 7. Weather Feature Engineering
    wd_rad = df['wind_deg'] * np.pi / 180
    df['WD_sin'] = np.sin(wd_rad)
    df['WD_cos'] = np.cos(wd_rad)
    df['Heat_Humidity_Interaction'] = df['AT'] * df['RH']

    # 8. Calendar Flags
    df['hour'] = df['Date'].dt.hour
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['traffic_hour'] = (df['hour'].isin(range(7, 10)) | df['hour'].isin(range(17, 20))).astype(int)
    
    sl_holidays = holidays.SriLanka(years=[2025, 2026])
    df['is_holiday'] = df['Date'].dt.date.isin(sl_holidays).astype(int)
    
    df['monsoon_phase_Northeast Monsoon'] = df['month'].isin([12, 1, 2]).astype(int)
    df['monsoon_phase_First Inter-monsoon'] = df['month'].isin([3, 4]).astype(int)
    df['monsoon_phase_Southwest Monsoon'] = df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['monsoon_phase_Second Inter-monsoon'] = df['month'].isin([10, 11]).astype(int)

    # 9. Clean and Trim to exactly 192 hours (168 Encoder + 24 Decoder)
    df = df.dropna(subset=['PM2 5 Conc_lag24', 'AT']) 
    df = df.tail(168 + forecast_horizon).reset_index(drop=True)
    
    # 10. Initialize target variables for PyTorch
    df['residual_value'] = 0.0
    df['sarimax_pred_scaled'] = 0.0 
    df['time_idx'] = range(len(df))

    return df

def run_batch():
    full_forecast = {}

    for loc in locations:
        print(f"\nProcessing {loc.upper()}...")
        loc_results = {}
        
        # 1. Build the continuous matrix dynamically
        try:
            combined_df_base = build_live_inference_matrix(loc, forecast_horizon)
        except Exception as e:
            print(f"Failed to build matrix for {loc}: {e}")
            continue
            
        if combined_df_base.empty:
            print(f"Matrix for {loc} is empty. Skipping.")
            continue

        # Loop Pollutants
        for poll in pollutants:
            clean_poll = poll.replace(' ', '_')
            sarimax_path = f"{model_path}/{loc}_sarimax_{clean_poll}.pkl"
            
            if not os.path.exists(sarimax_path):
                continue
                
            print(f"  Generating {poll}...", end=" ")
            try:
                # Create a copy for this specific pollutant model
                combined_df = combined_df_base.copy()
                combined_df['pollutant_id'] = poll

                # SARIMAX Prediction
                with open(sarimax_path, "rb") as f:
                    sarimax = joblib.load(f)
                with open(f"{model_path}/{loc}_scaler_{clean_poll}.pkl", "rb") as f:
                    scaler = joblib.load(f)

                sarimax_features = [
                    'PM2 5 Conc_lag24', 'PM2 5 Conc_rolling24_mean',
                    'PM10 Conc_lag24', 'PM10 Conc_rolling24_mean',
                    'AT', 'RH', 'BP', 'Rain Gauge', 'WD_sin', 'WD_cos',
                    'monsoon_phase_First Inter-monsoon', 'monsoon_phase_Northeast Monsoon',
                    'monsoon_phase_Second Inter-monsoon',
                    'traffic_hour', 'is_weekend', 'is_holiday', 'month', 'hour',
                    'Heat_Humidity_Interaction'
                ]
                
                for feat in sarimax_features:
                    if feat not in combined_df.columns: combined_df[feat] = 0.0
                    
                future_exog = combined_df.iloc[-forecast_horizon:][sarimax_features].fillna(0)
                exog_array = scaler.transform(future_exog)
                
                start_idx = int(sarimax.nobs)
                end_idx = start_idx + forecast_horizon - 1
                pred_obj = sarimax.predict(start=start_idx, end=end_idx, exog=exog_array, dynamic=False)
                
                log_pred = pred_obj.values if hasattr(pred_obj, 'values') else pred_obj
                base_pred = np.expm1(log_pred)
                base_pred = np.maximum(base_pred, 0.0)
                
                # TFT Prediction
                # Update TFT Input
                p_mean = base_pred.mean()
                p_std = base_pred.std() + 1e-6
                scaled_pred = (base_pred - p_mean) / p_std
                
                target_idx = len(combined_df) - forecast_horizon
                col_idx = combined_df.columns.get_loc('sarimax_pred_scaled')
                combined_df.iloc[target_idx:, col_idx] = scaled_pred

                # Load TFT
                original_load = torch.load
                torch.load = lambda f, map_location=None, weights_only=False, **kwargs: original_load(f, map_location=map_location, weights_only=False, **kwargs)
                try:
                    # Fix the monotone_constaints typo before loading
                    checkpoint_path = f"{model_path}/{loc}_tft_model_final.ckpt"
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    
                    # Fix typo in hyperparameters if it exists
                    if "hyper_parameters" in checkpoint:
                        hparams = checkpoint["hyper_parameters"]
                        if 'monotone_constaints' in hparams:
                            hparams['monotone_constraints'] = hparams.pop('monotone_constaints')
                            torch.save(checkpoint, checkpoint_path)  # Save corrected checkpoint
                    
                    tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
                    with open(f"{model_path}/{loc}_training_dataset_params.pkl", "rb") as f:
                        tft_params = joblib.load(f)
                        
                    inf_ds = TimeSeriesDataSet.from_parameters(
                        tft_params, combined_df, predict=True, stop_randomization=True
                    )
                    inf_loader = inf_ds.to_dataloader(train=False, batch_size=1, num_workers=11)
                    tft_pred = tft.predict(inf_loader, mode="prediction").detach().cpu().numpy().flatten()
                    
                    # Get the reason 
                    try:
                        out = tft.predict(inf_loader, mode="raw", return_x=True)
                        
                        # Extract manually
                        raw_prediction = out[0]
                        x = out[1]
                        
                        # Calculate Importance
                        interpretation = tft.interpret_output(raw_prediction, reduction="sum") 
                        
                        # Extract Weights
                        decoder_weights = interpretation["decoder_variables"].detach().cpu().numpy()  
                        decoder_names = tft.decoder_variables 
                        
                        # Normalize to % 
                        total_weight = decoder_weights.sum()
                        if total_weight > 0:
                            norm_weights = decoder_weights / total_weight
                        else:
                            norm_weights = decoder_weights

                        # Save as readable dictionary
                        xai_results = {name: float(weight) for name, weight in zip(decoder_names, norm_weights)}
                        
                        # Sort by highest impact
                        sorted_xai = dict(sorted(xai_results.items(), key=lambda item: item[1], reverse=True))
                        
                        # Add to results
                        loc_results[f"{poll}_xai"] = sorted_xai
                        
                        # Optional: Print top driver to console for debugging
                        top_driver = list(sorted_xai.keys())[0]
                        print(f" (Driver: {top_driver})", end="")
                        
                    except Exception as e:
                        print(f"XAI Failed: {e}")
                        loc_results[f"{poll}_xai"] = {}

                    # Hybrid
                    min_len = min(len(base_pred), len(tft_pred))

                    # Calculate Hybrid
                    hybrid_val = base_pred[:min_len] + tft_pred[:min_len]

                    # Fallback in case of negative predictions
                    final_val = np.where(hybrid_val < 0, base_pred[:min_len], hybrid_val)

                    final_val = np.maximum(final_val, 0.0)
                            
                    loc_results[poll] = final_val.tolist()
                    print("Done")

                finally:
                    torch.load = original_load

                # Clean Memory
                del sarimax, scaler, tft, tft_params
                del combined_df, inf_ds, inf_loader
                gc.collect()

            except Exception as e:
                print(f" {e}")

        # Store results 
        full_forecast[loc] = loc_results
        
        del combined_df_base
        gc.collect()

    # Save Final as a JSON
    with open(output_file, "w") as f:
        json.dump({"updated_at": str(datetime.datetime.now()), "forecasts": full_forecast}, f)
    print(f"\nForecast saved to {output_file}")

if __name__ == "__main__":
    run_batch()