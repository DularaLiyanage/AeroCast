from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
from . import utils

router = APIRouter()

# Global Assets Storage
ASSETS = {}
LOCATIONS = ["battaramulla", "kandy"]

# Config - Make sure the 'forecast' folder exists in your root directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to reach the 'app' folder
app_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# Build absolute path to models directory
ASSETS_DIR = os.path.join(app_dir, "models", "risk_scoring")

def load_assets():
    """ 
    Loads models, scalers, and configs for available locations on startup.
    """
    for loc in LOCATIONS:
        loc_path = os.path.join(ASSETS_DIR, loc)
        if not os.path.exists(loc_path):
            print(f"Warning: Assets for {loc} not found at {loc_path}. Skipping.")
            continue
            
        try:
            print(f"Loading assets for {loc}...")
            assets = {}
            
            # Load Config
            with open(os.path.join(loc_path, "config.json"), "r") as f:
                assets["config"] = json.load(f)
                
            # Load Hourly Averages
            with open(os.path.join(loc_path, "hourly_averages.json"), "r") as f:
                assets["hourly_averages"] = json.load(f)
                
            # Load Scalers
            assets["feature_scaler"] = joblib.load(os.path.join(loc_path, "feature_scaler.pkl"))
            assets["target_scaler"] = joblib.load(os.path.join(loc_path, "target_scaler.pkl"))
            
            # Load Top Drivers
            assets["top_drivers"] = joblib.load(os.path.join(loc_path, "top_drivers.pkl"))
            
            # Load Model
        
            model_path = os.path.join(loc_path, "optimized_attention_model.keras")
            if not os.path.exists(model_path):
                 # Fallback to the other one if exists
                 model_path = os.path.join(loc_path, "final_aqi_model.keras")
            
            assets["model"] = tf.keras.models.load_model(model_path, compile=False) # compile=False for safety/speed
            
            ASSETS[loc] = assets
            print(f"Successfully loaded {loc}.")
            
        except Exception as e:
            print(f"Error loading assets for {loc}: {e}")

@router.on_event("startup")
def startup_event():
    load_assets()

def get_risk_info(aqi_value, config):
    """
    Determines risk level and color based on AQI and config.
    """
    categories = config.get("cea_categories", {})
    
    # Sort categories just in case
    # Assume categories is dict of "Level": {"min":, "max":, "color":}
    
    for level, info in categories.items():
        if info['min'] <= aqi_value <= info['max']:
            return level, info['color']
            
    # Fallback if out of range (e.g. > 500)
    if aqi_value > 500:
        return "Hazardous", "maroon"
    return "Unknown", "gray"

    return "Unknown", "gray"

class PredictionRequest(BaseModel):
    location: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "battaramulla"
            }
        }
    }

class PredictionResponse(BaseModel):
    date: str
    time: str
    aqi: float
    min_aqi: float
    max_aqi: float
    risk_level: str
    risk_color: str
    reasoning: str
    health_alert: Optional[dict] = None
    hourly_forecast: list[dict] = []
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2026-01-03",
                "time": "14:00",
                "aqi": 85.5,
                "min_aqi": 75.0,
                "max_aqi": 95.0,
                "risk_level": "Moderate",
                "risk_color": "yellow",
                "reasoning": "High traffic congestion during peak hours.",
                "hourly_forecast": [
                     {
                         "date": "2026-01-03", 
                         "time": "15:00", 
                         "aqi": 87.0, 
                         "min_aqi": 77.0,
                         "max_aqi": 97.0,
                         "risk_level": "Moderate", 
                         "risk_color": "yellow",
                         "reasoning": "High solar radiation levels"
                     }
                ]
            }
        }
    }

@router.post("/predict_24h", response_model=PredictionResponse)
async def predict_24h(request: PredictionRequest):
    """
    Endpoint to predict current AQI based on past 24h weather data.
    """
    location = request.location.lower()
    print(f"[DEBUG] Received location: '{location}', Available assets: {list(ASSETS.keys())}")
    
    if location not in ASSETS:
        raise HTTPException(status_code=400, detail=f"Location '{location}' not supported or assets not loaded. Available: {list(ASSETS.keys())}")
    
    assets = ASSETS[location]
    
    try:
        # 1. Fetch Weather
        weather_df = utils.get_open_meteo_data(location)
        
        # 2. Prepare Input
        input_array, input_df_features = utils.prepare_input(weather_df, assets["hourly_averages"])
        
        # 3. Scale Input
        reshaped_input = input_array.reshape(24, 26)
        scaled_input = assets["feature_scaler"].transform(reshaped_input)
        final_input = scaled_input.reshape(1, 24, 26)
        
        # 4. Predict
        prediction_scaled = assets["model"].predict(final_input, verbose=0)
        
        # Handle 3D output (Sequence) -> Take last step
        if prediction_scaled.ndim == 3:
             prediction_scaled = prediction_scaled[:, -1, :]
        
        # 5. Inverse Transform
        aqi_pred = assets["target_scaler"].inverse_transform(prediction_scaled)[0][0]
        
        # 6. Min/Max Calculation
        now = datetime.now()
        current_hour = now.hour
        
        q_hats = assets["config"].get("q_hat_90", [])
        q_val = 0
        if len(q_hats) > current_hour:
             q_val = q_hats[current_hour]
        elif len(q_hats) > 0:
             q_val = q_hats[0]
             
        min_aqi = max(0, aqi_pred - q_val)
        max_aqi = aqi_pred + q_val
        
        # 7. Risk Level, Reasoning & Alert
        risk_level, risk_color = get_risk_info(aqi_pred, assets["config"])
        reasoning = utils.generate_reasoning(input_df_features, assets["top_drivers"], aqi_pred, risk_level)
        reasoning = utils.generate_reasoning(input_df_features, assets["top_drivers"], aqi_pred, risk_level)
        health_alert = None # Will populate based on next hour
        
        # 8. Future Forecast (Next 24h)
        hourly_forecast = []
        try:
            # A. Fetch Future Weather
            forecast_df = utils.get_open_meteo_forecast(location)
            
            # B. Prepare Input (Similar to history, but for future)
            # Use 'hourly_averages' fallback same as before. 
            forecast_input_array, forecast_input_df = utils.prepare_input(forecast_df, assets["hourly_averages"])
            
            # C. Reshape & Scale
            forecast_reshaped = forecast_input_array.reshape(24, 26)
            forecast_scaled = assets["feature_scaler"].transform(forecast_reshaped)
            forecast_final = forecast_scaled.reshape(1, 24, 26)
            
            # D. Predict Sequence
            future_pred_scaled = assets["model"].predict(forecast_final, verbose=0)
            
            if future_pred_scaled.ndim == 3:
                future_pred_2d = future_pred_scaled[0] 
                future_aqi_values = assets["target_scaler"].inverse_transform(future_pred_2d) # (24, 1)
                
                # Format Output
                start_next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                q_hats = assets["config"].get("q_hat_90", [])
                
                for i in range(24):
                     val = future_aqi_values[i][0]
                     ts = start_next_hour + timedelta(hours=i)
                     hour_idx = ts.hour
                     
                     # Uncertainty
                     q_val = q_hats[hour_idx] if len(q_hats) > hour_idx else (q_hats[0] if q_hats else 0)
                     f_min = max(0, val - q_val)
                     f_max = val + q_val
                     
                     # Risk & Reasoning
                     r_level, r_color = get_risk_info(val, assets["config"])
                     
                     # Generate reasoning for this specific future hour
                     # We pass the single row dataframe corresponding to this hour
                     # utils.generate_reasoning looks at .iloc[-1], so passing 1-row DF works.
                     row_df = forecast_input_df.iloc[[i]]
                     f_reasoning = utils.generate_reasoning(row_df, assets["top_drivers"], val, r_level)
                     
                     hourly_forecast.append({
                         "date": ts.strftime("%Y-%m-%d"),
                         "time": ts.strftime("%H:%M"),
                         "aqi": round(float(val), 2),
                         "min_aqi": round(float(f_min), 2),
                         "max_aqi": round(float(f_max), 2),
                         "risk_level": r_level,
                         "risk_color": r_color,
                         "reasoning": f_reasoning
                     })
                     

                # --- ALERT LOGIC FOR NEXT HOUR ---
                if len(future_aqi_values) > 0:
                     next_hour_val = future_aqi_values[0][0]
                     
                     next_hour_time = (now + timedelta(hours=1)).replace(minute=0, second=0).strftime("%H:%M")
                     health_alert = utils.get_cea_alert(next_hour_val)
                     
                     if health_alert:
                         # Append time to message
                         health_alert["message"] = f"{health_alert['message']} (Expected around {next_hour_time})"
                         
                         # Add color based on risk level
                         alert_level, alert_color = get_risk_info(next_hour_val, assets["config"])
                         health_alert["color"] = alert_color
                # ---------------------------------
                     
            else:
                 print("Warning: Model does not support sequence output for forecast. Skipping.")
                 
        except Exception as e:
            print(f"Forecast Error: {e}")
            # Don't fail the whole request, just return empty forecast
        
        # --- TEST ALERT OVERRIDE ---
        test_aqi = 105.0
        now_next = datetime.now() + timedelta(hours=1)
        test_time_str = now_next.replace(minute=0, second=0).strftime("%H:%M")
        
        health_alert = utils.get_cea_alert(test_aqi)
        if health_alert:
            health_alert["message"] = f"{health_alert['message']} (Expected around {test_time_str})"
            _, alert_color = get_risk_info(test_aqi, assets["config"])
            health_alert["color"] = alert_color
            
            # --- SYNC FORECAST WITH TEST ALERT ---
            if hourly_forecast:
                # Update the first hour (next hour) to match the test alert
                hourly_forecast[0]["aqi"] = test_aqi
                hourly_forecast[0]["min_aqi"] = max(0, test_aqi - 10) # Mock uncertainty
                hourly_forecast[0]["max_aqi"] = test_aqi + 10
                hourly_forecast[0]["risk_level"], hourly_forecast[0]["risk_color"] = get_risk_info(test_aqi, assets["config"])
                # We can also update reasoning if needed, or leave as is (likely won't match but acceptable for test)
                hourly_forecast[0]["reasoning"] = "Approaching high pollution levels."
            # -------------------------------------
        # -----------------------------------------------
        
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "aqi": round(float(aqi_pred), 2),
            "min_aqi": round(float(min_aqi), 2),
            "max_aqi": round(float(max_aqi), 2),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "reasoning": reasoning,
            "health_alert": health_alert,
            "hourly_forecast": hourly_forecast
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))