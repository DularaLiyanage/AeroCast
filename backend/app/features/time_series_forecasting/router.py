import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Config - Make sure the 'forecast' folder exists in your root directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up 2 levels to reach the 'app' folder
# Level 1 up: .../backend/app/features
# Level 2 up: .../backend/app
app_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# 3. Build the path to the JSON file from the 'app' folder
output_file = os.path.join(app_dir, "forecast", "time_series_forecasting", "daily_forecast.json")

# Debug Print: Check your server logs to see if this matches!
print(f"Looking for forecast file at: {output_file}")

class ForecastRequest(BaseModel):
    location: str

# 2. Add endpoints using @router instead of @app
@router.get("/status")
def health_check():
    return {"status": "Time Series Module Online", "mode": "Lightweight"}

@router.post("/forecast")
def get_forecast(req: ForecastRequest):
    loc = req.location.lower()
    
    if not os.path.exists(output_file):
        print(f"ERROR: File not found at {output_file}")
        return {"error": "Forecast data not ready. Please run batch_runner.py."}
    
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
            
        all_forecasts = data.get("forecasts", {})
        
        if loc in all_forecasts:
            return {"location": loc, "forecast": all_forecasts[loc]}
        else:
            raise HTTPException(status_code=404, detail="Location not found in cache")
            
    except Exception as e:
        return {"error": f"Failed to read cache: {str(e)}"}