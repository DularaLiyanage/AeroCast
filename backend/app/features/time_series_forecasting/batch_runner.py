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

# Config
model_path= "../../models/time_series_forecasting" 
locations = ["kandy", "baththaramulla"]
output_file = "../../forecast/time_series_forecasting/daily_forecast.json"
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
        
        # Load History
        hist_file = f"{model_path}/{loc}_tft_input_data.pkl"
        
        try:
            print(f"  Loading history: {hist_file} ...")
            history_df = pd.read_pickle(hist_file)
            
            # Pre-calculate lags
            latest_values = {}
            for p in ["PM2 5 Conc", "PM10 Conc", "NO2 Conc", "SO2 Conc", "O3 Conc"]:
                # Adjust 'residual_value' to actual value column if needed
                if p in history_df['pollutant_id'].unique():
                    latest_values[p] = history_df[history_df['pollutant_id'] == p]['residual_value'].iloc[-1]
                else:
                    latest_values[p] = 0.0
            print("History loaded.")
            
        except Exception as e:
            print(f"Failed to load history for {loc}: {e}")
            history_df = pd.DataFrame()
            latest_values = {p: 0.0 for p in ["PM2 5 Conc", "PM10 Conc", "NO2 Conc", "SO2 Conc", "O3 Conc"]}
        
        # Get Weather
        weather_df = get_real_weather_forecast(loc)
        if weather_df.empty:
            print("Skipping (No Weather Data)")
            del history_df
            gc.collect()
            continue

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

                # Fill Lags 
                last_pm25 = poll_history['PM2 5 Conc_lag24'].iloc[-1] if 'PM2 5 Conc_lag24' in poll_history else 0
                last_pm10 = poll_history['PM10 Conc_lag24'].iloc[-1] if 'PM10 Conc_lag24' in poll_history else 0

                # Fill Lags
                poll_future['PM2 5 Conc_lag24'] = last_pm25
                poll_future['PM2 5 Conc_rolling24_mean'] = last_pm25
                poll_future['PM10 Conc_lag24'] = last_pm10
                poll_future['PM10 Conc_rolling24_mean'] = last_pm10
                
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
                    tft = TemporalFusionTransformer.load_from_checkpoint(f"{model_path}/{loc}_tft_model_final.ckpt")
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