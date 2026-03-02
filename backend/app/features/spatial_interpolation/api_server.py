# backend/api_server.py - API endpoints for air quality prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from datetime import timedelta
import traceback

# Direct import - works when running directly from backend/ directory
from .air_quality_logic import (
    TARGETS, STATION_META, load_artifacts, predict_one_station,
    clamp_value, make_heatmap_png_base64, ensemble_spatial_prediction
)


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Air Quality Prediction API", version="2.0")


# ============================================================
# CORS Configuration
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,
)


# ============================================================
# Request Models
# ============================================================
class PredictRequest(BaseModel):
    station: str
    target: str
    datetime_str: str
    overrides: dict | None = None


class SpatialPredictRequest(BaseModel):
    target: str
    datetime_str: str
    lat: float
    lon: float
    overrides: dict | None = None


# ============================================================
# Root Endpoint
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Air Quality Prediction API",
        "version": "2.0",
        "endpoints": ["/predict", "/predict_spatial", "/health"],
        "stations": list(STATION_META.keys()),
        "targets": TARGETS
    }


# ============================================================
# Station Prediction Endpoint
# ============================================================
@app.post("/predict")
def predict(req: PredictRequest):
    """Predict air quality at a specific station"""
    try:
        print(f"\n[/predict] Request: {req.station} {req.target} @ {req.datetime_str}", flush=True)

        # Validate station
        if req.station not in STATION_META:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid station '{req.station}'. Valid: {list(STATION_META.keys())}"
            )

        # Validate target
        if req.target not in TARGETS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target '{req.target}'. Valid: {TARGETS}"
            )

        # Parse datetime
        dt_base = pd.to_datetime(req.datetime_str, errors="coerce")
        if pd.isna(dt_base):
            raise HTTPException(
                status_code=400,
                detail="Invalid datetime_str. Use format: 'YYYY-MM-DD HH:MM:SS'"
            )

        print(f"  Loading model artifacts for {req.target}...", flush=True)
        model, preprocess, cfg = load_artifacts(req.target)

        # Predict main station
        print(f"  Predicting {req.station}...", flush=True)
        pred_main_raw = predict_one_station(
            model, preprocess, cfg, req.station, dt_base, overrides=req.overrides
        )
        pred_main = clamp_value(req.target, pred_main_raw)
        print(f"  ✓ {req.station}: {pred_main:.2f}", flush=True)

        # Predict both stations for heatmap
        print(f"  Predicting all stations for heatmap...", flush=True)
        b_raw = predict_one_station(
            model, preprocess, cfg, "Battaramulla", dt_base,
            overrides=req.overrides if req.station == "Battaramulla" else None
        )
        k_raw = predict_one_station(
            model, preprocess, cfg, "Kandy", dt_base,
            overrides=req.overrides if req.station == "Kandy" else None
        )

        b_pred = clamp_value(req.target, b_raw)
        k_pred = clamp_value(req.target, k_raw)
        print(f"  ✓ Battaramulla: {b_pred:.2f}, Kandy: {k_pred:.2f}", flush=True)

        # Generate heatmap
        points = [
            (STATION_META["Battaramulla"]["lat"], STATION_META["Battaramulla"]["lon"]),
            (STATION_META["Kandy"]["lat"], STATION_META["Kandy"]["lon"]),
        ]

        print(f"  Generating spatial heatmap...", flush=True)
        heatmap_b64, interpolation_bbox = make_heatmap_png_base64(
            points, [b_pred, k_pred],
            title=f"{req.target} (t+1) • {pd.to_datetime(dt_base).floor('h').strftime('%Y-%m-%d %H:%M')}",
            power=2.0,
            grid_size=80,
            target=req.target
        )
        print(f"  ✓ Heatmap generated", flush=True)

        # Build response
        response = {
            "success": True,
            "station": req.station,
            "target": req.target,
            "base_time": pd.to_datetime(dt_base).floor("h").strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_time": (pd.to_datetime(dt_base).floor("h") + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": float(pred_main),
            "station_preds": {
                "Battaramulla": float(b_pred),
                "Kandy": float(k_pred)
            },
            "spatial_interpolation": {
                "method": "ensemble_optimized",
                "components": ["rbf", "gp", "knn", "idw"],
                "power": 2.0,
                "grid_size": 80,
                "bbox": interpolation_bbox
            },
            "heatmap_png_base64": heatmap_b64,
        }

        print(f"  ✓ Request completed successfully\n", flush=True)
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR in /predict:", flush=True)
        print(f"  {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================
# Spatial Prediction Endpoint
# ============================================================
@app.post("/predict_spatial")
def predict_spatial(req: SpatialPredictRequest):
    """Predict air quality at arbitrary coordinates using spatial interpolation"""
    try:
        print(f"\n[/predict_spatial] Request: {req.target} @ ({req.lat}, {req.lon})", flush=True)

        # Validate target
        if req.target not in TARGETS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target '{req.target}'. Valid: {TARGETS}"
            )

        # Parse datetime
        dt_base = pd.to_datetime(req.datetime_str, errors="coerce")
        if pd.isna(dt_base):
            raise HTTPException(
                status_code=400,
                detail="Invalid datetime_str. Use format: 'YYYY-MM-DD HH:MM:SS'"
            )

        # Validate coordinates (basic check for Sri Lanka region)
        if not (5.0 < req.lat < 10.0 and 79.0 < req.lon < 82.0):
            print(f"  ⚠️  Warning: Coordinates outside Sri Lanka region", flush=True)

        print(f"  Loading model artifacts for {req.target}...", flush=True)
        model, preprocess, cfg = load_artifacts(req.target)

        # Predict at both stations
        print(f"  Predicting station values...", flush=True)
        b_raw = predict_one_station(
            model, preprocess, cfg, "Battaramulla", dt_base, overrides=req.overrides
        )
        k_raw = predict_one_station(
            model, preprocess, cfg, "Kandy", dt_base, overrides=req.overrides
        )

        b_pred = clamp_value(req.target, b_raw)
        k_pred = clamp_value(req.target, k_raw)
        print(f"  ✓ Battaramulla: {b_pred:.2f}, Kandy: {k_pred:.2f}", flush=True)

        # Spatial interpolation
        station_coords = [
            (STATION_META["Battaramulla"]["lat"], STATION_META["Battaramulla"]["lon"]),
            (STATION_META["Kandy"]["lat"], STATION_META["Kandy"]["lon"]),
        ]
        station_values = [b_pred, k_pred]
        target_coord = (req.lat, req.lon)

        print(f"  Performing ensemble spatial interpolation...", flush=True)
        ensemble_pred, ensemble_std, individual_preds = ensemble_spatial_prediction(
            station_coords, station_values, target_coord
        )
        print(f"  ✓ Interpolated value: {ensemble_pred:.2f} ± {ensemble_std:.2f}", flush=True)

        # Build response
        response = {
            "success": True,
            "target": req.target,
            "base_time": pd.to_datetime(dt_base).floor("h").strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_time": (pd.to_datetime(dt_base).floor("h") + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "location": {
                "lat": float(req.lat),
                "lon": float(req.lon)
            },
            "prediction": float(ensemble_pred),
            "uncertainty": float(ensemble_std),
            "confidence_interval": {
                "lower": float(ensemble_pred - 1.96 * ensemble_std),
                "upper": float(ensemble_pred + 1.96 * ensemble_std)
            },
            "station_contributions": {
                "Battaramulla": float(b_pred),
                "Kandy": float(k_pred)
            },
            "method_contributions": {
                k: float(v) for k, v in individual_preds.items() 
                if not k.endswith('_std')
            },
            "interpolation_method": "ensemble"
        }

        print(f"  ✓ Request completed successfully\n", flush=True)
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR in /predict_spatial:", flush=True)
        print(f"  {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================
# Hover Value Endpoint
# ============================================================
@app.get("/hover_value")
def hover_value(lat: float, lon: float, target: str, datetime_str: str):
    """Get air quality value at hover coordinates"""
    try:
        print(f"\n[/hover_value] Request: {target} @ ({lat}, {lon}) {datetime_str}", flush=True)

        # Validate target
        if target not in TARGETS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target '{target}'. Valid: {TARGETS}"
            )

        # Parse datetime
        dt_base = pd.to_datetime(datetime_str, errors="coerce")
        if pd.isna(dt_base):
            raise HTTPException(
                status_code=400,
                detail="Invalid datetime_str. Use format: 'YYYY-MM-DD HH:MM:SS'"
            )

        # Calculate distance to nearest station
        import math
        def haversine_distance(lat1, lon1, lat2, lon2):
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return 6371 * c  # km

        distances = []
        for station, meta in STATION_META.items():
            dist = haversine_distance(lat, lon, meta["lat"], meta["lon"])
            distances.append(dist)
        min_distance = min(distances)

        # If too far from stations, return out_of_range
        MAX_INTERPOLATION_DISTANCE_KM = 50.0  # Adjust as needed
        if min_distance > MAX_INTERPOLATION_DISTANCE_KM:
            return {
                "method": "out_of_range",
                "value": None,
                "distance_km": float(min_distance)
            }

        # Otherwise, perform spatial prediction
        print(f"  Loading model artifacts for {target}...", flush=True)
        model, preprocess, cfg = load_artifacts(target)

        # Predict at both stations
        print(f"  Predicting station values...", flush=True)
        b_raw = predict_one_station(
            model, preprocess, cfg, "Battaramulla", dt_base, overrides=None
        )
        k_raw = predict_one_station(
            model, preprocess, cfg, "Kandy", dt_base, overrides=None
        )

        b_pred = clamp_value(target, b_raw)
        k_pred = clamp_value(target, k_raw)
        print(f"  ✓ Battaramulla: {b_pred:.2f}, Kandy: {k_pred:.2f}", flush=True)

        # Spatial interpolation
        station_coords = [
            (STATION_META["Battaramulla"]["lat"], STATION_META["Battaramulla"]["lon"]),
            (STATION_META["Kandy"]["lat"], STATION_META["Kandy"]["lon"]),
        ]
        station_values = [b_pred, k_pred]
        target_coord = (lat, lon)

        print(f"  Performing ensemble spatial interpolation...", flush=True)
        ensemble_pred, ensemble_std, individual_preds = ensemble_spatial_prediction(
            station_coords, station_values, target_coord
        )
        print(f"  ✓ Interpolated value: {ensemble_pred:.2f}", flush=True)

        return {
            "method": "ensemble",
            "value": float(ensemble_pred),
            "distance_km": None
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ERROR in /hover_value:", flush=True)
        print(f"  {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================
# Health Check Endpoint
# ============================================================
@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        import air_quality_logic
        
        return {
            "status": "healthy",
            "models_loaded": list(air_quality_logic._model_cache.keys()),
            "spatial_models_cached": len(air_quality_logic._spatial_model_cache),
            "observation_rows": len(air_quality_logic.obs),
            "stations": list(STATION_META.keys()),
            "targets": TARGETS
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ============================================================
# OPTIONS handlers for CORS preflight
# ============================================================
@app.options("/predict")
async def options_predict():
    return JSONResponse(content={"message": "OK"}, status_code=200)


@app.options("/predict_spatial")
async def options_predict_spatial():
    return JSONResponse(content={"message": "OK"}, status_code=200)


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🌍 Air Quality Prediction API Server")
    print("="*60)
    print(f"Stations: {list(STATION_META.keys())}")
    print(f"Targets: {TARGETS}")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )