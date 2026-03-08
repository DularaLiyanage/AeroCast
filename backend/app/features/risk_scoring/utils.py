import pandas as pd
import numpy as np
import requests
import holidays
from datetime import datetime, timedelta
from typing import Optional
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

OPEN_METEO_TIMEOUT_SECONDS = 12
OPEN_METEO_CACHE_TTL_SECONDS = 600  # 10 minutes

AIR_QUALITY_FEATURE_MAP = {
    "PM2.5 Conc_lag1": ("pm2_5", 1),
    "PM10 Conc_lag24": ("pm10", 24),
    "SO2 Conc_lag24": ("sulphur_dioxide", 24),
    "O3 Conc_lag1": ("ozone", 1),
    "NO2 Conc_lag1": ("nitrogen_dioxide", 1),
}

# In-memory weather cache: key -> (timestamp, dataframe)
_WEATHER_CACHE = {}


def _get_cache(cache_key: str, allow_stale: bool = False):
    cached = _WEATHER_CACHE.get(cache_key)
    if not cached:
        return None

    ts, df = cached
    if allow_stale or (time.time() - ts) <= OPEN_METEO_CACHE_TTL_SECONDS:
        return df.copy()

    return None


def _set_cache(cache_key: str, df: pd.DataFrame):
    _WEATHER_CACHE[cache_key] = (time.time(), df.copy())

def get_open_meteo_data(location_name: str):
    """
    Fetches past 24 hours of weather data from Open-Meteo API.
    """
    location_key = location_name.lower()
    if location_key not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")

    cache_key = f"history:{location_key}"
    cached_df = _get_cache(cache_key)
    if cached_df is not None:
        return cached_df

    coords = LOCATION_COORDS[location_key]
    
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
    
    try:
        response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        stale_df = _get_cache(cache_key, allow_stale=True)
        if stale_df is not None:
            print(f"Warning: Open-Meteo history fetch failed, using stale cache for {location_key}: {e}")
            return stale_df
        raise ConnectionError(f"Open-Meteo history fetch failed for {location_key}: {e}") from e
    
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
    _set_cache(cache_key, df)
    return df

def get_open_meteo_forecast(location_name: str):
    """
    Fetches NEXT 24 hours of weather forecast.
    """
    location_key = location_name.lower()
    if location_key not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")

    cache_key = f"forecast:{location_key}"
    cached_df = _get_cache(cache_key)
    if cached_df is not None:
        return cached_df

    coords = LOCATION_COORDS[location_key]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,shortwave_radiation,wind_speed_10m,wind_direction_10m",
        "past_days": 0,
        "forecast_days": 3, # Increased from 2 to 3 to be safe
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        stale_df = _get_cache(cache_key, allow_stale=True)
        if stale_df is not None:
            print(f"Warning: Open-Meteo forecast fetch failed, using stale cache for {location_key}: {e}")
            return stale_df
        raise ConnectionError(f"Open-Meteo forecast fetch failed for {location_key}: {e}") from e
    
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
    _set_cache(cache_key, df)
    return df


def get_open_meteo_air_quality_data(location_name: str):
    """
    Fetches historical hourly air-quality data from Open-Meteo for lag feature construction.
    We fetch enough history so lag24 is available for each of the past-24h model rows.
    """
    location_key = location_name.lower()
    if location_key not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")

    cache_key = f"air-quality-history:{location_key}"
    cached_df = _get_cache(cache_key)
    if cached_df is not None:
        return cached_df

    coords = LOCATION_COORDS[location_key]

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        stale_df = _get_cache(cache_key, allow_stale=True)
        if stale_df is not None:
            print(f"Warning: Open-Meteo air-quality fetch failed, using stale cache for {location_key}: {e}")
            return stale_df
        print(f"Warning: Open-Meteo air-quality fetch failed for {location_key}: {e}")
        return None

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty or "time" not in df.columns:
        print(f"Warning: Open-Meteo air-quality data missing/empty for {location_key}")
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df.resample("h").interpolate(method="linear")
    df = df.reset_index()

    _set_cache(cache_key, df)
    return df


def get_open_meteo_air_quality_forecast(location_name: str):
    """
    Fetches future air quality forecast data from Open-Meteo.
    Returns next 24+ hours of forecasted air quality data.
    """
    location_key = location_name.lower()
    if location_key not in LOCATION_COORDS:
        raise ValueError(f"Unknown location: {location_name}")

    cache_key = f"air-quality-forecast:{location_key}"
    cached_df = _get_cache(cache_key)
    if cached_df is not None:
        return cached_df

    coords = LOCATION_COORDS[location_key]

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "hourly": "pm10,pm2_5,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": 1,  # Need some past data for lag calculation
        "forecast_days": 3,
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=OPEN_METEO_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        stale_df = _get_cache(cache_key, allow_stale=True)
        if stale_df is not None:
            print(f"Warning: Open-Meteo air-quality forecast fetch failed, using stale cache for {location_key}: {e}")
            return stale_df
        print(f"Warning: Open-Meteo air-quality forecast fetch failed for {location_key}: {e}")
        return None

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    if df.empty or "time" not in df.columns:
        print(f"Warning: Open-Meteo air-quality forecast data missing/empty for {location_key}")
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df.resample("h").interpolate(method="linear")
    df = df.reset_index()

    _set_cache(cache_key, df)
    return df


def _build_real_lag_features(
    weather_times: pd.Series,
    air_quality_df: pd.DataFrame,
    air_quality_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Builds real lagged pollutant features aligned to the 24 weather timestamps.
    Uses historical data first, then falls back to forecasted data for future lag requirements.
    
    Parameters:
    - weather_times: Series of datetime objects for which lag features are needed
    - air_quality_df: Historical air quality data (measured, more reliable)
    - air_quality_forecast_df: Forecasted air quality data (for future lag requirements)
    
    Returns:
    - DataFrame with lag features, using hybrid historical+forecast data
    """
    lag_features = pd.DataFrame(index=range(len(weather_times)))
    
    # Prepare historical data
    aq_hist = air_quality_df.copy() if air_quality_df is not None else None
    if aq_hist is not None:
        aq_hist["time"] = pd.to_datetime(aq_hist["time"])
        aq_hist = aq_hist.set_index("time").sort_index()
        aq_hist = aq_hist.resample("h").interpolate(method="linear")
    
    # Prepare forecasted data
    aq_fcst = air_quality_forecast_df.copy() if air_quality_forecast_df is not None else None
    if aq_fcst is not None:
        aq_fcst["time"] = pd.to_datetime(aq_fcst["time"])
        aq_fcst = aq_fcst.set_index("time").sort_index()
        aq_fcst = aq_fcst.resample("h").interpolate(method="linear")
    
    # Normalize input times to hour precision
    normalized_times = pd.to_datetime(weather_times).dt.floor("h")
    now = pd.Timestamp.now().floor("h")
    
    for target_feature, (aq_column, lag_hours) in AIR_QUALITY_FEATURE_MAP.items():
        values = []
        
        for target_time in normalized_times:
            lag_time = target_time - pd.Timedelta(hours=lag_hours)
            
            value = None
            
            # Try historical data first (more reliable)
            if aq_hist is not None:
                try:
                    value = aq_hist.loc[lag_time, aq_column]
                    if pd.notna(value):
                        values.append(float(value))
                        continue
                except (KeyError, TypeError):
                    pass
            
            # Fall back to forecasted data if lag_time is in future relative to now
            if aq_fcst is not None and lag_time > now:
                try:
                    value = aq_fcst.loc[lag_time, aq_column]
                    if pd.notna(value):
                        values.append(float(value))
                        continue
                except (KeyError, TypeError):
                    pass
            
            # If no data found, append NaN (will be filled by hourly averages later)
            values.append(np.nan)
        
        lag_features[target_feature] = values
    
    return lag_features

def prepare_input(
    weather_df: pd.DataFrame,
    hourly_averages: dict,
    air_quality_df: Optional[pd.DataFrame] = None,
    air_quality_forecast_df: Optional[pd.DataFrame] = None,
):
    """
    Constructs the (1, 24, 26) input array for the model.
    Hybrid approach: uses historical AQ data first, then forecasted AQ data for future requirements.
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

    # --- SOURCE C: Real Open-Meteo Air-Quality Lag Features (Hybrid: Historical + Forecasted) ---
    if air_quality_df is not None or air_quality_forecast_df is not None:
        try:
            real_lags = _build_real_lag_features(
                times,
                air_quality_df,
                air_quality_forecast_df=air_quality_forecast_df,
            )
            for col in real_lags.columns:
                if col in input_df.columns:
                    input_df[col] = real_lags[col].values
        except Exception as e:
            print(f"Warning: Failed to build real lag features; falling back to hourly averages: {e}")
    
    # --- SOURCE D: Hourly Averages (Fallback) ---
    # For columns not yet filled, look up in hourly_averages using the hour of that row
    # hourly_averages structure: { "feature_name": { "0": val, "1": val, ... "23": val } }
    
    columns_to_fill = [c for c in REQUIRED_FEATURES if input_df[c].isnull().any()]
    
    for col in columns_to_fill:
        if col not in hourly_averages:
            # If a feature is missing from averages, fill with 0 or warn?
            # Assuming it exists as per requirements.
            input_df[col] = input_df[col].fillna(0.0)
            continue
            
        # Map hour -> average value
        # Note: hourly_averages keys are strings "0", "1", ...
        avg_map = hourly_averages[col]
        # Create a list of values for the 24 rows
        values = [avg_map.get(str(h), 0.0) for h in hours]
        fallback_series = pd.Series(values, index=input_df.index)
        input_df[col] = input_df[col].fillna(fallback_series)

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
