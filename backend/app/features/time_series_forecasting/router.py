import os
import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, "../../"))
forecast_dir = os.path.join(app_dir, "forecast", "time_series_forecasting")

print(f"Forecast directory: {forecast_dir}")


class ForecastRequest(BaseModel):
    location: str
    date: Optional[str] = None  # YYYY-MM-DD, defaults to latest available


def _list_available_dates() -> list[str]:
    if not os.path.exists(forecast_dir):
        return []
    dates = []
    for fname in os.listdir(forecast_dir):
        match = re.match(r"forecast_(\d{4}-\d{2}-\d{2})\.json", fname)
        if match:
            dates.append(match.group(1))
    dates.sort(reverse=True)
    return dates


@router.get("/status")
def health_check():
    return {"status": "Time Series Module Online", "mode": "Lightweight"}


@router.get("/dates")
def get_available_dates():
    return {"dates": _list_available_dates()}


@router.post("/forecast")
def get_forecast(req: ForecastRequest):
    loc = req.location.lower()

    if req.date:
        file_path = os.path.join(forecast_dir, f"forecast_{req.date}.json")
        if not os.path.exists(file_path):
            return {"error": f"Forecast for date {req.date} not found."}
    else:
        dates = _list_available_dates()
        if not dates:
            return {"error": "Forecast data not ready. Please run batch_runner.py."}
        file_path = os.path.join(forecast_dir, f"forecast_{dates[0]}.json")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        all_forecasts = data.get("forecasts", {})
        forecast_date = data.get("forecast_date", "unknown")

        if loc in all_forecasts:
            return {"location": loc, "forecast_date": forecast_date, "forecast": all_forecasts[loc]}
        else:
            raise HTTPException(status_code=404, detail="Location not found in cache")

    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"Failed to read cache: {str(e)}"}
