import pandas as pd
import numpy as np
import requests
import holidays
from datetime import datetime, timedelta

# Constants
LOCATION_COORDS = {
    "battaramulla": {"latitude": 6.9016, "longitude": 79.9234},
    "kandy": {"latitude": 7.2906, "longitude": 80.6337}
}

REQUIRED_FEATURES = [
    'comp_daily', 'comp_weekly', 'comp_monsoon', 'comp_residual', 
    'AT', 'BP', 'RH', 'Solar Rad', 'Spd_100m_mps', 'Dir_100m_deg', 'Prs_0m_hPa', 
    'PM2.5 Conc_lag1', 'PM10 Conc_lag24', 'SO2 Conc_lag24', 'O3 Conc_lag1', 'NO2 Conc_lag1', 
    'TrafficRiskScore', 'TrafficRiskScore_roll_mean_24h', 'IsHoliday', 'IsWeekend', 
    'is_peak_hour', 'india_transport_intensity_roll_mean_24h', 
    'transboundary_risk_index', 'transboundary_risk_index_roll_mean_24h', 
    'hour_sin', 'hour_cos'
]

def get_open_meteo_data(location_name: str):
    """
    Fetches past 24 hours of weather data from Open-Meteo API.
    """
    if location_name.lower() not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")
    
    coords = LOCATION_COORDS[location_name.lower()]
    
    # We want the *past* 24 hours. 
    # Open-Meteo 'past_days=1' gives yesterday and today so far.
    # We need to be careful to select the correct 24h window (e.g. up to current hour).
    # For simplicity/robustness, we'll request past 2 days to ensure we have enough coverage,
    # then filter for the last 24 records ending at current_hour - 1.
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,shortwave_radiation,wind_speed_10m,wind_direction_10m",
        "past_days": 3, # Increased from 2
        "forecast_days": 1,
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    
    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Handle gaps via resampling
    df = df.set_index('time').sort_index()
    # Resample to hourly to ensure no gaps, interpolate small gaps
    df = df.resample('h').interpolate(method='linear')
    df = df.reset_index()
    
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    # Filter <= current_hour (History)
    df = df[df['time'] < current_hour] 
    
    # Take the last 24 entries
    if len(df) < 24:
        raise ValueError(f"Insufficient weather data fetched. Got {len(df)} rows, expected >= 24.")
    
    df = df.tail(24).reset_index(drop=True)
    return df

def get_open_meteo_forecast(location_name: str):
    """
    Fetches NEXT 24 hours of weather forecast.
    """
    if location_name.lower() not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")
    
    coords = LOCATION_COORDS[location_name.lower()]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,shortwave_radiation,wind_speed_10m,wind_direction_10m",
        "past_days": 0,
        "forecast_days": 3, # Increased from 2 to 3 to be safe
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df['time'] = pd.to_datetime(df['time'])
    
    # Resample/Interpolate
    df = df.set_index('time').sort_index()
    df = df.resample('h').interpolate(method='linear')
    df = df.reset_index()
    
    # Filter for Future
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    df = df[df['time'] > current_hour]
    
    if len(df) < 24:
         raise ValueError(f"Insufficient forecast data. Got {len(df)} rows.")
         
    df = df.head(24).reset_index(drop=True)
    return df

def prepare_input(weather_df: pd.DataFrame, hourly_averages: dict):
    """
    Constructs the (1, 24, 26) input array for the model.
    """
    # Initialize an empty DataFrame with 24 rows and the required column order
    input_df = pd.DataFrame(index=range(24), columns=REQUIRED_FEATURES)
    
    # --- SOURCE A: Open-Meteo Mapping ---
    input_df['AT'] = weather_df['temperature_2m']
    input_df['RH'] = weather_df['relative_humidity_2m']
    input_df['BP'] = weather_df['surface_pressure']
    input_df['Prs_0m_hPa'] = weather_df['surface_pressure'] # Mapped to both
    input_df['Solar Rad'] = weather_df['shortwave_radiation']
    input_df['Spd_100m_mps'] = weather_df['wind_speed_10m'] * (5/18) # Conversion
    input_df['Dir_100m_deg'] = weather_df['wind_direction_10m']
    
    # --- SOURCE B: Calculated Features ---
    # We need the 'time' column from weather_df to determine hour, weekday etc.
    times = weather_df['time']
    hours = times.dt.hour
    
    # Hour sin/cos
    input_df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    input_df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    # Holidays & Weekends
    lk_holidays = holidays.SriLanka()
    input_df['IsHoliday'] = times.apply(lambda x: 1.0 if x in lk_holidays else 0.0)
    input_df['IsWeekend'] = times.apply(lambda x: 1.0 if x.weekday() >= 5 else 0.0)
    
    # Peak Hour & Traffic Risk
    # Peak: 7-9 (7,8,9) or 16-19 (16,17,18,19)? 
    # User said: "hour is (7-9) or (16-19)" -> literally 7,8,9 and 16,17,18,19? 
    # Usually "7-9" means 7:00 to 9:59 (hours 7, 8, 9). I will assume inclusive integers.
    def is_peak(h):
        return 1.0 if (7 <= h <= 9) or (16 <= h <= 19) else 0.0
    
    input_df['is_peak_hour'] = hours.apply(is_peak)
    input_df['TrafficRiskScore'] = input_df['is_peak_hour'].apply(lambda x: 0.8 if x == 1.0 else 0.2)
    
    # --- SOURCE C: Hourly Averages (Fallback) ---
    # For columns not yet filled, look up in hourly_averages using the hour of that row
    # hourly_averages structure: { "feature_name": { "0": val, "1": val, ... "23": val } }
    
    columns_to_fill = [c for c in REQUIRED_FEATURES if input_df[c].isnull().any()]
    
    for col in columns_to_fill:
        if col not in hourly_averages:
            # If a feature is missing from averages, fill with 0 or warn?
            # Assuming it exists as per requirements.
            input_df[col] = 0.0
            continue
            
        # Map hour -> average value
        # Note: hourly_averages keys are strings "0", "1", ...
        avg_map = hourly_averages[col]
        # Create a list of values for the 24 rows
        values = [avg_map.get(str(h), 0.0) for h in hours]
        input_df[col] = values

    # Ensure all are float
    input_df = input_df.astype(float)
    
    # Check shape
    if input_df.shape != (24, 26):
        raise ValueError(f"Shape mismatch error. Got {input_df.shape}, expected (24, 26)")
        
    return input_df.values.reshape(1, 24, 26), input_df

def generate_reasoning(input_df_24h: pd.DataFrame, top_drivers: list, prediction_val: float, risk_level: str):
    """
    Generates a reasoning string based on the *latest* hour's conditions and top drivers.
    """
    # Look at the last row (most recent time)
    latest_row = input_df_24h.iloc[-1]
    current_hour = int(pd.Timestamp.now().hour) # Or the hour from the row? Use row.
    # Actually, let's use the row's hour to be consistent with the data used for prediction
    # But wait, input_df_24h doesn't have the 'hour' column explicitly as 0-23, it has sin/cos.
    # But we can assume the logic requested: check driver conditions.
    
    # Deduplicate reasons
    unique_reasons = []
    seen = set()
    
    # Helper to check conditions
    for driver in top_drivers:
        driver_clean = driver.strip()
        
        # Latest row value
        val = latest_row.get(driver_clean, 0)
        
        msg = ""
        if "Traffic" in driver_clean:
            if latest_row.get('is_peak_hour', 0) == 1.0:
                 msg = "High traffic congestion during peak hours"
        elif "Solar" in driver_clean:
             if val > 500: # High solar
                 msg = "High solar radiation levels"
        elif "PM" in driver_clean:
             if val > 50: # Threshold
                 msg = f"Elevated {driver_clean} levels"
        elif "transboundary" in driver_clean.lower():
             if val > 0.5:
                 msg = "Significant transboundary pollution detected"
        elif "Wind" in driver_clean or "Spd" in driver_clean:
             if val < 1.0:
                 msg = "Low wind speed trapping pollutants"
        
        if msg and msg not in seen:
            unique_reasons.append(msg)
            seen.add(msg)
    
    # Contextualize based on Risk Level (AQI)
    result_string = ""
    
    if risk_level == "Good":
        if unique_reasons:
            # If there are reasons (like traffic) but AQI is still Good
            joined_reasons = ", ".join(unique_reasons).lower()
            return f"Air quality is good despite {joined_reasons}."
        else:
            return "Air quality is good. Weather conditions are favorable."
            
    elif risk_level == "Moderate":
        if unique_reasons:
            return f"Air quality is acceptable. Main contributors: {', '.join(unique_reasons)}."
        else:
             return "Air quality is moderate."
             
    else: # Unhealthy or worse
        if unique_reasons:
            return f"Warning: {risk_level} conditions driven by {', '.join(unique_reasons)}."
        else:
            return f"Air quality is {risk_level}. Reduce outdoor exposure."

# CEA Alert Configuration
CEA_ALERTS = [
    {
        "min": 101, "max": 150,
        "title": "Unhealthy for Sensitive Groups",
        "message": "Sensitive groups should reduce prolonged or heavy outdoor exertion"
    },
    {
        "min": 151, "max": 200,
        "title": "Unhealthy",
        "message": "Everyone may begin to experience health effects; sensitive groups members may experience more serious health effects"
    },
    {
        "min": 201, "max": 300,
        "title": "Very Unhealthy",
        "message": "Health alert: everyone may experience more serious health effects."
    },
    {
        "min": 301, "max": 5000, # Using big max to catch > 500 safely
        "title": "Hazardous",
        "message": "Health warning of emergency conditions. The entire population is more likely to be affected"
    }
]

def get_cea_alert(aqi_value: float):
    """
    Returns an alert dictionary if AQI > 100 based on user-defined thresholds.
    """
    for alert in CEA_ALERTS:
        if alert["min"] <= aqi_value <= alert["max"]:
            return {"title": alert["title"], "message": alert["message"]}

    # Any outlier > 5000 (unlikely) falls through to None or use last logic
    if aqi_value > 300: # Safety net for very high values not covered by loop if any
        return {
            "title": "Hazardous", 
            "message": "Health warning of emergency conditions. The entire population is more likely to be affected"
        }
            
    return None
