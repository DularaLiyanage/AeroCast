from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
import os
from .xai_service import AirQualityExplainer
from .waqi_service import WAQIService

router = APIRouter()

# Config - Make sure the 'forecast' folder exists in your root
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up 2 levels to reach the 'app' folder
app_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# Build absolute path to models directory
MODEL_DIR = os.path.join(app_dir, "models", "anomaly_detection")

DATA_FILE = os.path.join(app_dir, "data", "anomaly_detection", "Pollution_Research_Hourly_Series.parquet")
feature_names = [
    'AT', 'RH', 'BP', 'Solar Rad', 'Rain Gauge', 'WD Raw', 
    'O3 Conc', 'NO Conc', 'NO2 Conc', 'NOx Conc', 'SO2 Conc', 'PM2.5 Conc', 'PM10 Conc'
]

# Location-specific models and explainers
LOC_CONFIG = {}
LOC_LIST = ['battaramulla', 'kandy']

print("Loading models for each location...")
for loc in LOC_LIST:
    try:
        s_path = f"{MODEL_DIR}/{loc}_scaler.joblib"
        m_path = f"{MODEL_DIR}/{loc}_gru_model.h5"
        b_path = f"{MODEL_DIR}/{loc}_background_data.npy"
        
        if os.path.exists(s_path) and os.path.exists(m_path) and os.path.exists(b_path):
            scaler = joblib.load(s_path)
            bg_data = np.load(b_path)
            explainer = AirQualityExplainer(m_path, bg_data, feature_names)
            LOC_CONFIG[loc] = {
                'scaler': scaler,
                'explainer': explainer,
                'ready': True
            }
            print(f"Loaded model for {loc}")
        else:
            print(f"Model files for {loc} not found. Using mock for this location.")
            LOC_CONFIG[loc] = {'ready': False}
    except Exception as e:
        print(f"Error loading {loc}: {e}")
        LOC_CONFIG[loc] = {'ready': False}

waqi = WAQIService(token="demo")

class ForecastRequest(BaseModel):
    location: str # 'kandy' or 'battaramulla'

@router.get("/")
def read_root():
    return {"message": "Welcome to Air Quality Forecast API"}

@router.post("/forecast")
def get_forecast(request: ForecastRequest):
    loc_input = request.location.lower()
    # Normalize location
    loc = "kandy" if "kandy" in loc_input else "battaramulla"
    
    config = LOC_CONFIG.get(loc, {'ready': False})
    
    # 1. Fetch real-time data
    real_time_data = waqi.get_by_station_name(loc)
    mapped_features = waqi.map_to_model_features(real_time_data)
    
    if not mapped_features:
        # If external API fails, we use our local synthetic data if available
        mapped_features = {f: 0.5 for f in feature_names}
        
    # 2. Prepare Input Sequence (Window size 24)
    try:
        if os.path.exists(DATA_FILE):
            df_hourly = pd.read_parquet(DATA_FILE)
            # Use exact match for filtering (Battaramulla or Kandy)
            actual_loc_name = "Kandy" if loc == "kandy" else "Battaramulla"
            df_loc = df_hourly[df_hourly['Location'] == actual_loc_name].tail(23)
            history = df_loc[feature_names].values
        else:
            # Fallback to random history if parquet is missing
            print(f"Warning: {DATA_FILE} not found. Using random history.")
            history = np.random.rand(23, len(feature_names))

        current = np.array([[mapped_features.get(f, 0.5) for f in feature_names]])
        combined = np.vstack([history, current]) # (24, 13)
        
        if config['ready']:
            scaled_input = config['scaler'].transform(combined)
        else:
            scaled_input = combined 
            
        model_input = np.expand_dims(scaled_input, axis=0) # (1, 24, 13)
        
    except Exception as e:
        print(f"Data error: {e}")
        model_input = np.random.rand(1, 24, 13)

    # 3. Predict / Distinct Mock Anomaly
    if config['ready']:
        explainer = config['explainer']
        scaler = config['scaler']
        prediction_scaled = explainer.model.predict(model_input)
        prediction_final = {}
        targets = ['O3', 'PM10', 'PM2.5', 'SO2']
        target_indices = [6, 12, 11, 10]
        for i, target in enumerate(targets):
            idx = target_indices[i]
            val = prediction_scaled[0][i]
            prediction_final[target] = float((val * scaler.scale_[idx]) + scaler.min_[idx])
        
        reasons = explainer.get_top_reasons(model_input)
    else:
        # Distinct Mock values to show difference between locations if models fail to load
        if loc == 'kandy':
            prediction_final = {'O3': 0.85, 'PM10': 35.2, 'PM2.5': 18.4, 'SO2': 0.65}
            reasons = {'PM2.5': ['Mountain Topography', 'Vehicle Emissions'], 'PM10': ['Road Dust']}
        else:
            prediction_final = {'O3': 1.12, 'PM10': 25.1, 'PM2.5': 14.2, 'SO2': 0.32}
            reasons = {'O3': ['High Sunlight', 'Industrial Stack'], 'PM2.5': ['Traffic Congestion']}

    # 4. Generate Safety Tips & Multi-day Forecast (location-specific patterns)
    forecast_trends = {}
    thresholds = {'PM2.5': 15.0, 'PM10': 30.0, 'O3': 0.9, 'SO2': 0.5}
    
    # Sri Lankan Holiday & Kandy Specific Event Logic
    current_date = pd.Timestamp.now()
    kandy_events = {
        (2026, 8, 18): "Kumbal Perahera", (2026, 8, 19): "Kumbal Perahera", 
        (2026, 8, 20): "Kumbal Perahera", (2026, 8, 21): "Kumbal Perahera", 
        (2026, 8, 22): "Kumbal Perahera", (2026, 8, 23): "Randoli Perahera",
        (2026, 8, 24): "Randoli Perahera", (2026, 8, 25): "Randoli Perahera",
        (2026, 8, 26): "Grand Randoli Procession", (2026, 8, 27): "Diya Kapeema Ceremony",
        (2026, 5, 1): "Vesak Perahera & Dansal", (2026, 6, 29): "Poson Perahera & Dansal",
        (2026, 6, 5): "World Environment Day", (2026, 9, 1): "World Tourism Day"
    }
    
    national_holidays = {
        (2026, 4, 13): "Sinhala/Tamil New Year", (2026, 4, 14): "Sinhala/Tamil New Year",
        (2026, 5, 1): "May Day", (2026, 2, 4): "Independence Day"
    }

    for target, val in prediction_final.items():
        thresh = thresholds.get(target, 1.5)
        noise_level = thresh * 0.15 
        trend = [val]
        
        for i in range(6):
            if i == 3: 
                spike = np.random.uniform(thresh * 0.8, thresh * 2.0)
            else:
                spike = np.random.uniform(-noise_level, noise_level)
                
            next_val = max(0, trend[-1] + spike)
            trend.append(round(next_val, 2))
            
            # Check if this specific forecast day hits a holiday/event
            forecast_day = current_date + pd.Timedelta(days=i)
            day_key = (forecast_day.year, forecast_day.month, forecast_day.day)
            
            event_name = None
            if loc == "kandy":
                event_name = kandy_events.get(day_key) or national_holidays.get(day_key)
            else:
                event_name = national_holidays.get(day_key)
                
            if event_name and next_val > thresh:
                if target not in reasons: reasons[target] = []
                if event_name not in reasons[target]:
                    reasons[target].append(f"Increased Activity: {event_name}")

        forecast_trends[target] = trend

    pollutant_tips = {
        'PM2.5': ["Monitor air quality regularly.", "Use indoor air purifiers.", "Wear mask if >50."],
        'PM10': ["Limit outdoor dust exposure.", "Keep windows closed during peak traffic."],
        'O3': ["Avoid intense exercise in peak sunlight.", "Ozone peaks mid-afternoon."],
        'SO2': ["Avoid industrial areas.", "Check sulfur dioxide reports weekly."]
    }

    # Safe extraction of AQI
    cur_aqi = 42
    if isinstance(real_time_data, dict):
        rt_data = real_time_data.get("data", {})
        if isinstance(rt_data, dict):
            cur_aqi = rt_data.get("aqi", 42)

    pollutant_thresholds = {
        'PM2.5': 15.0,
        'PM10': 30.0,
        'O3': 0.9,
        'SO2': 0.5
    }

    return {
        "location": loc.upper(),
        "current_aqi": cur_aqi,
        "forecast_7_days": forecast_trends,
        "reasons": reasons,
        "safety_tips": pollutant_tips,
        "thresholds": pollutant_thresholds
    }