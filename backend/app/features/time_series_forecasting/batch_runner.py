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
def get_open_meteo_aq_with_features(location_name):
    coords = cordinates.get(location_name.lower(), cordinates["baththaramulla"])
    
    today = datetime.date.today() 
    target_date = today - datetime.timedelta(days=1)

    # Pull 8 days of data to safely calculate 24h lags/rolling means
    start_date = target_date - datetime.timedelta(days=8)
    
    print(f"Fetching AQ History for {location_name.title()} from {start_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}...")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ["pm10", "pm2_5"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": target_date.strftime("%Y-%m-%d")
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "Date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
    })

    # Shift data by 24 hours to create the exact lags
    df['PM2 5 Conc_lag24'] = df['pm2_5'].shift(24)
    df['PM10 Conc_lag24'] = df['pm10'].shift(24)
    
    # Calculate the 24-hour rolling mean
    df['PM2 5 Conc_rolling24_mean'] = df['pm2_5'].rolling(window=24).mean()
    df['PM10 Conc_rolling24_mean'] = df['pm10'].rolling(window=24).mean()

    # Drop NaNs and return exactly 168 hours (7 days)
    df = df.dropna()
    return df.tail(168).reset_index(drop=True)

# Get the weather for the location
def get_real_weather_forecast(location_name):

    coords = cordinates.get(location_name, cordinates["baththaramulla"])

    # Current Dates
    today = datetime.date.today()
    start_date = today + datetime.timedelta(days=1)
    end_date = start_date + datetime.timedelta(days=2)

    print(f"Fetching Real Weather Forecast for: {start_date} to {end_date}...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }

    # Send Request
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process Data
    hourly = response.Hourly()

    # Extract arrays
    weather_data = {
        "Date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        ),
        "AT": hourly.Variables(0).ValuesAsNumpy(),
        "RH": hourly.Variables(1).ValuesAsNumpy(),
        "Rain Gauge": hourly.Variables(2).ValuesAsNumpy(),
        "BP": hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed": hourly.Variables(4).ValuesAsNumpy(),
        "wind_deg": hourly.Variables(5).ValuesAsNumpy()
    }

    weather_df = pd.DataFrame(data=weather_data)

    # Feature Engineering
    # Wind
    wd_rad = weather_df['wind_deg'] * np.pi / 180
    weather_df['WD_sin'] = np.sin(wd_rad)
    weather_df['WD_cos'] = np.cos(wd_rad)

    # Time
    weather_df['hour'] = weather_df['Date'].dt.hour
    weather_df['month'] = weather_df['Date'].dt.month
    weather_df['day_of_week'] = weather_df['Date'].dt.dayofweek
    weather_df['is_weekend'] = weather_df['day_of_week'].isin([5, 6]).astype(int)
    weather_df['traffic_hour'] = (
        weather_df['hour'].isin(range(7, 10)) |
        weather_df['hour'].isin(range(17, 20))
    ).astype(int)

    # Holiday
    sl_holidays = holidays.SriLanka(years=[2025, 2026])

    # Check if the Forecast Date is in the holiday list
    weather_df['is_holiday'] = weather_df['Date'].dt.date.isin(sl_holidays).astype(int)

    # Monsoon (Dec/Jan is Northeast)
    weather_df['monsoon_phase_Northeast Monsoon'] = weather_df['month'].isin([12, 1, 2]).astype(int)
    weather_df['monsoon_phase_First Inter-monsoon'] = weather_df['month'].isin([3, 4]).astype(int)
    weather_df['monsoon_phase_Southwest Monsoon'] = weather_df['month'].isin([5, 6, 7, 8, 9]).astype(int)
    weather_df['monsoon_phase_Second Inter-monsoon'] = weather_df['month'].isin([10, 11]).astype(int)

    # Interaction
    weather_df['Heat_Humidity_Interaction'] = weather_df['AT'] * weather_df['RH']

    return weather_df.iloc[:forecast_horizon]

def run_batch():
    full_forecast = {}

    for loc in locations:
        print(f"\nProcessing {loc.upper()}...")
        loc_results = {}
        
        # 1. Fetch Live APIs First
        try:
            aq_history_df = get_open_meteo_aq_with_features(loc)
            weather_df = get_real_weather_forecast(loc)
        except Exception as e:
            print(f"Failed to fetch live API data for {loc}: {e}")
            continue

        if weather_df.empty or aq_history_df.empty:
            print("Skipping (Missing API Data)")
            continue

        # 2. Load Local History
        hist_file = f"{model_path}/{loc}_tft_input_data.pkl"
        try:
            print(f"  Loading long-term history: {hist_file} ...")
            history_df = pd.read_pickle(hist_file)
        except Exception as e:
            print(f"Failed to load history for {loc}: {e}")
            history_df = pd.DataFrame()

        # Loop Pollutants
        for poll in pollutants:
            clean_poll = poll.replace(' ', '_')
            sarimax_path = f"{model_path}/{loc}_sarimax_{clean_poll}.pkl"
            
            if not os.path.exists(sarimax_path):
                continue
                
            print(f"  Generating {poll}...", end=" ")
            try:
                # Prepare Data 
                if not history_df.empty:
                    poll_history = history_df[history_df['pollutant_id'] == poll].tail(168).copy()
                else:
                    poll_history = pd.DataFrame()

                poll_future = weather_df.copy()
                poll_future['pollutant_id'] = poll

                # 3. Inject Actual Live Data for Future Lags
                if len(aq_history_df) >= 24:
                    # Get the actual PM values from the past 24 hours
                    past_24_pm25 = aq_history_df['pm2_5'].values[-24:]
                    past_24_pm10 = aq_history_df['pm10'].values[-24:]

                    # Map past actuals to future lag columns
                    poll_future['PM2 5 Conc_lag24'] = past_24_pm25
                    poll_future['PM10 Conc_lag24'] = past_24_pm10
                    
                    # Set the baseline rolling mean to the mean of the past 24 hours
                    poll_future['PM2 5 Conc_rolling24_mean'] = past_24_pm25.mean()
                    poll_future['PM10 Conc_rolling24_mean'] = past_24_pm10.mean()
                else:
                    poll_future['PM2 5 Conc_lag24'] = 0.0
                    poll_future['PM10 Conc_lag24'] = 0.0
                    poll_future['PM2 5 Conc_rolling24_mean'] = 0.0
                    poll_future['PM10 Conc_rolling24_mean'] = 0.0
                
                # Add Targets
                poll_future['residual_value'] = 0.0
                poll_future['sarimax_pred_scaled'] = 0.0 

                # Combine
                combined_df = pd.concat([poll_history, poll_future], ignore_index=True)
                combined_df = combined_df.fillna(0.0)
                
                # Fix Time Index
                if not poll_history.empty:
                    last_idx = int(poll_history['time_idx'].max())
                    new_indices = range(last_idx + 1, last_idx + 1 + len(poll_future))
                    combined_df.iloc[-len(poll_future):, combined_df.columns.get_loc('time_idx')] = new_indices
                else:
                    combined_df['time_idx'] = range(len(combined_df))
                
                combined_df['time_idx'] = combined_df['time_idx'].astype(int)

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
        
        del history_df
        gc.collect()

    # Save Final as a JSON
    with open(output_file, "w") as f:
        json.dump({"updated_at": str(datetime.datetime.now()), "forecasts": full_forecast}, f)
    print(f"\nForecast saved to {output_file}")

if __name__ == "__main__":
    run_batch()