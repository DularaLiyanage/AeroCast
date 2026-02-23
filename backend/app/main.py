from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.features.time_series_forecasting.router import router as time_forecast_router
from app.features.spatial_interpolation.router import router as spatial_router
from app.features.risk_scoring.router import router as aqi_router
from app.features.anomaly_detection.router import router as anomaly_router

app = FastAPI(title="AeroCast Backend")

app.include_router(time_forecast_router, prefix="/time_forecast", tags=["Time Series"])
app.include_router(spatial_router, prefix="/spatial", tags=["Spatial"])
app.include_router(aqi_router, prefix="/aqi", tags=["AQI"])
app.include_router(anomaly_router, prefix="/anomaly", tags=["Anomaly"])

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AeroCast Backend Running"}