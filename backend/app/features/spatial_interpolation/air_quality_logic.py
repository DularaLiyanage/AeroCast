# air_quality_logic.py - Business logic for air quality prediction and spatial interpolation

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import base64
import io
import json

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import joblib
import pickle
from scipy.interpolate import Rbf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBF_kernel, ConstantKernel as C

# Spatial auxiliary libraries
try:
    import rasterio
    from pyproj import Transformer
    from shapely.geometry import Point
    SPATIAL_LIBRARIES_AVAILABLE = True
    print("Spatial auxiliary libraries loaded successfully")
except ImportError as e:
    print(f"Warning: Spatial auxiliary libraries not available: {e}")
    SPATIAL_LIBRARIES_AVAILABLE = False


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "spatial_interpolation" / "city_predictor_model_dl_md"
OBS_CSV = BASE_DIR / "spatial_interpolation" / "clean" / "cea_hourly_2019_2024_clean_dl_md.csv"

# Cache file for observation data
OBS_CACHE = BASE_DIR / "spatial_interpolation" / "clean" / "obs_cache.pkl"
SL_GEOJSON = BASE_DIR / "spatial_interpolation" / "data" / "sri_lanka_boundary.geojson"

# Geospatial data paths
GEOSPATIAL_DIR = BASE_DIR / "spatial_interpolation" / "sri_lanka_data"
ELEVATION_TIF = GEOSPATIAL_DIR / "srtm_68_11.tif"
LANDCOVER_TIFS = [
    GEOSPATIAL_DIR / "landcover_N06E078.tif",
    GEOSPATIAL_DIR / "landcover_N06E081.tif",
    GEOSPATIAL_DIR / "landcover_N09E078.tif",
    GEOSPATIAL_DIR / "landcover_N09E081.tif"
]
POPULATION_TIF = GEOSPATIAL_DIR / "lka_ppp_2020.tif"
ROADS_SHP = GEOSPATIAL_DIR / "sri-lanka-251231-free.shp"

# Cache file for spatial auxiliary data
SPATIAL_CACHE = GEOSPATIAL_DIR / "spatial_auxiliary_cache.pkl"


# ============================================================
# Configuration
# ============================================================
WINDOW = 24

STATION_META = {
    "Battaramulla": {"lat": 6.901035, "lon": 79.926513},
    "Kandy": {"lat": 7.292651, "lon": 80.635649},
}

TARGETS = [
    "PM25", "PM10", "NO2", "SO2", "O3", "CO", "NOX",
    "WS", "WD", "AT", "RH", "BP", "SolarRad", "Rain",
]

DEFAULT_VALS = {
    "PM25": 40.0, "PM10": 60.0, "NO2": 25.0, "SO2": 5.0, "O3": 10.0,
    "CO": 400.0, "NOX": 30.0, "WS": 1.5, "WD": 120.0, "AT": 30.0,
    "RH": 70.0, "BP": 1010.0, "SolarRad": 500.0, "Rain": 0.0,
}
OVERRIDE_COLS = list(DEFAULT_VALS.keys())

BOUNDS = {
    "PM25": (0, 1000), "PM10": (0, 1500), "NO2": (0, 1000),
    "SO2": (0, 1000), "O3": (0, 1000), "CO": (0, 50000),
    "NOX": (0, 2000), "WS": (0, 60), "WD": (0, 360),
    "AT": (-5, 50), "RH": (0, 100), "BP": (800, 1100),
    "SolarRad": (0, 1400), "Rain": (0, 500),
}

# Color scale bounds for heatmap visualization (optimized for Sri Lankan air quality ranges)
COLOR_SCALE = {
    "PM25": (0, 15), "PM10": (0, 40), "NO2": (0, 30),
    "SO2": (0, 15), "O3": (0, 30), "CO": (0, 2000),
    "NOX": (0, 50), "WS": (0, 5), "WD": (0, 360),
    "AT": (20, 35), "RH": (0, 100), "BP": (980, 1020),
    "SolarRad": (0, 300), "Rain": (0, 15),
}


# ============================================================
# Spatial Auxiliary Data
# ============================================================
class SpatialAuxiliaryData:
    """Manages geospatial data for enhanced interpolation"""

    def __init__(self):
        self.elevation_data = None
        self.landcover_data = []
        self.population_data = None
        self.roads_data = None

        if SPATIAL_LIBRARIES_AVAILABLE:
            try:
                self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            except:
                pass
        
        # Try to load from cache first
        if SPATIAL_CACHE.exists():
            try:
                import pickle
                with open(SPATIAL_CACHE, 'rb') as f:
                    cached_data = pickle.load(f)
                self.elevation_data = cached_data['elevation']
                self.landcover_data = cached_data['landcover']
                self.population_data = cached_data['population']
                self.roads_data = cached_data['roads']
                self.roads_sindex = cached_data.get('roads_sindex')
                print("   Loaded spatial auxiliary data from cache")
                return
            except Exception as e:
                print(f"   Cache load failed: {e}, loading from files")
        
        self._load_real_geospatial_data()

    def _load_real_geospatial_data(self):
        """Load geospatial datasets with error handling"""
        if not SPATIAL_LIBRARIES_AVAILABLE:
            print("   Using simplified auxiliary features")
            return

        try:
            # Elevation
            if ELEVATION_TIF.exists():
                with rasterio.open(ELEVATION_TIF) as src:
                    self.elevation_data = {
                        'array': src.read(1),
                        'meta': src.meta
                    }
                print(f"   Loaded elevation data")

            # Land cover
            for lc_tif in LANDCOVER_TIFS:
                if lc_tif.exists():
                    with rasterio.open(lc_tif) as src:
                        self.landcover_data.append({
                            'array': src.read(1),
                            'meta': src.meta
                        })
            if self.landcover_data:
                print(f"   Loaded {len(self.landcover_data)} land cover tiles")

            # Population
            if POPULATION_TIF.exists():
                with rasterio.open(POPULATION_TIF) as src:
                    self.population_data = {
                        'array': src.read(1),
                        'meta': src.meta
                    }
                print(f"   Loaded population data")

            # Roads
            if ROADS_SHP.exists():
                roads_shp_path = ROADS_SHP / "gis_osm_roads_free_1.shp"
                if roads_shp_path.exists():
                    self.roads_data = gpd.read_file(roads_shp_path)
                    self.roads_sindex = self.roads_data.sindex
                    print(f"   Loaded {len(self.roads_data)} road segments")
                else:
                    print(f"   ⚠️  Roads shapefile not found at {roads_shp_path}")
                    self.roads_data = None
                    self.roads_sindex = None

            # Save to cache
            try:
                import pickle
                cached_data = {
                    'elevation': self.elevation_data,
                    'landcover': self.landcover_data,
                    'population': self.population_data,
                    'roads': self.roads_data,
                    'roads_sindex': self.roads_sindex
                }
                with open(SPATIAL_CACHE, 'wb') as f:
                    pickle.dump(cached_data, f)
                print("   Saved spatial auxiliary data to cache")
            except Exception as e:
                print(f"   Cache save failed: {e}")

        except Exception as e:
            print(f"⚠️  Warning loading geospatial data: {e}")
            print("   Using fallback features")

    def get_auxiliary_features(self, lat: float, lon: float) -> dict:
        """Extract auxiliary features"""
        try:
            features = {
                'elevation': self._sample_raster(self.elevation_data, lat, lon, default=100.0),
                'is_urban': self._get_land_cover_class(lat, lon),
                'population_density': self._sample_raster(self.population_data, lat, lon, default=1000.0),
                'road_distance': self._get_road_distance(lat, lon)
            }
        except:
            features = self._get_fallback_features(lat, lon)
        
        return features

    def _sample_raster(self, data, lat: float, lon: float, default: float = 0.0) -> float:
        """Sample raster value at coordinates"""
        if data is None:
            return default

        # Handle cached data (dict with array and meta)
        if isinstance(data, dict) and 'array' in data:
            array = data['array']
            meta = data['meta']
        else:
            return default

        try:
            # Transform lat/lon to pixel coordinates
            from rasterio.transform import rowcol
            row, col = rowcol(meta['transform'], lon, lat)
            if 0 <= row < meta['height'] and 0 <= col < meta['width']:
                value = array[row, col]
                nodata = meta.get('nodata')
                if nodata is not None and value == nodata:
                    return default
                return float(value)
        except:
            pass
        return default

    def _get_land_cover_class(self, lat: float, lon: float) -> float:
        """Land cover classification"""
        for lc_raster in self.landcover_data:
            value = self._sample_raster(lc_raster, lat, lon, default=-1)
            if value != -1:
                if value == 50:  # Built-up
                    return 1.0
                elif value in [10, 20, 30, 40]:  # Vegetation
                    return 0.0
                else:
                    return 0.5
        return 0.5

    def _get_road_distance(self, lat: float, lon: float) -> float:
        """Distance to nearest road"""
        if self.roads_data is None or self.roads_sindex is None:
            return self._get_fallback_road_distance(lat, lon)

        try:
            point = Point(lon, lat)
            point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
            point_projected = point_gdf.to_crs("EPSG:32644")
            buffer_geom = point_projected.buffer(5000).iloc[0]  # 5km buffer
            possible_indices = list(self.roads_sindex.query(buffer_geom, predicate='intersects'))
            if possible_indices:
                possible_roads = self.roads_data.iloc[possible_indices].to_crs("EPSG:32644")
                distances = possible_roads.distance(point_projected.iloc[0])
                return float(distances.min() / 1000)  # km
            else:
                return self._get_fallback_road_distance(lat, lon)
        except:
            return self._get_fallback_road_distance(lat, lon)

    def _get_fallback_features(self, lat: float, lon: float) -> dict:
        """Fallback when real data unavailable"""
        return {
            'elevation': self._get_fallback_elevation(lat, lon),
            'is_urban': self._get_fallback_land_cover(lat, lon),
            'population_density': self._get_fallback_population(lat, lon),
            'road_distance': self._get_fallback_road_distance(lat, lon)
        }

    def _get_fallback_elevation(self, lat: float, lon: float) -> float:
        if 6.8 < lat < 7.3 and 79.8 < lon < 80.7:
            return 500 + np.random.normal(0, 50)
        return 50 + np.random.normal(0, 20)

    def _get_fallback_land_cover(self, lat: float, lon: float) -> float:
        if abs(lat - 6.927) < 0.1 and abs(lon - 79.861) < 0.1:
            return 1.0  # Colombo
        elif abs(lat - 6.901) < 0.05 and abs(lon - 79.926) < 0.05:
            return 0.8  # Battaramulla
        return 0.2

    def _get_fallback_population(self, lat: float, lon: float) -> float:
        if abs(lat - 6.927) < 0.1 and abs(lon - 79.861) < 0.1:
            return 5000
        elif abs(lat - 6.901) < 0.05 and abs(lon - 79.926) < 0.05:
            return 2000
        return 500

    def _get_fallback_road_distance(self, lat: float, lon: float) -> float:
        road_coords = [(6.927, 79.861), (7.292, 80.635)]
        min_distance = float('inf')
        for road_lat, road_lon in road_coords:
            dist = np.sqrt((lat - road_lat)**2 + (lon - road_lon)**2) * 111
            min_distance = min(min_distance, dist)
        return min_distance


def haversine_grid(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c  # km


# Global instance
_aux_data = None
_grid_aux_cache = None

def get_auxiliary_data():
    """Lazy initialization of auxiliary data"""
    global _aux_data
    if _aux_data is None:
        print("Initializing SpatialAuxiliaryData (one-time setup)...")
        _aux_data = SpatialAuxiliaryData()
        print("SpatialAuxiliaryData ready")
    return _aux_data


def get_grid_auxiliary_features(grid_lat, grid_lon):
    """Get cached auxiliary features for grid points"""
    global _grid_aux_cache
    # Create a key based on grid bounds and size
    lat_min, lat_max = grid_lat.min(), grid_lat.max()
    lon_min, lon_max = grid_lon.min(), grid_lon.max()
    grid_size = grid_lat.shape[0]
    key = (round(lat_min, 3), round(lat_max, 3), round(lon_min, 3), round(lon_max, 3), grid_size)
    
    if _grid_aux_cache is None or _grid_aux_cache.get('key') != key:
        print("Precomputing auxiliary features for heatmap grid...")
        aux_data = get_auxiliary_data()
        grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        grid_features = []
        for lat, lon in grid_points:
            features = aux_data.get_auxiliary_features(lat, lon)
            grid_features.append([
                features['elevation'],
                features['is_urban'],
                features['road_distance'],
                features['population_density']
            ])
        _grid_aux_cache = {'key': key, 'features': np.array(grid_features)}
        print(f"Precomputed auxiliary features for {len(grid_features)} grid points")
    return _grid_aux_cache['features']


# ============================================================
# Load Data
# ============================================================
print("Loading observation data...", flush=True)
if OBS_CACHE.exists():
    try:
        import pickle
        with open(OBS_CACHE, 'rb') as f:
            obs = pickle.load(f)
        print(f"Loaded {len(obs)} observation rows from cache", flush=True)
    except Exception as e:
        print(f"Cache load failed: {e}, loading from CSV")
        if not OBS_CSV.exists():
            raise FileNotFoundError(f"OBS_CSV not found: {OBS_CSV}")
        obs = pd.read_csv(OBS_CSV, parse_dates=["datetime"])
        keep_cols = [
            "station", "datetime",
            "PM25", "PM10", "NO2", "SO2", "O3", "CO", "NOX",
            "WS", "WD", "AT", "RH", "BP", "SolarRad", "Rain",
            "month", "day", "hour", "hour_sin", "hour_cos"
        ]
        obs = obs[keep_cols].copy()
        try:
            with open(OBS_CACHE, 'wb') as f:
                pickle.dump(obs, f)
            print("Saved observation data to cache")
        except Exception as e:
            print(f"Cache save failed: {e}")
else:
    if not OBS_CSV.exists():
        raise FileNotFoundError(f"OBS_CSV not found: {OBS_CSV}")
    obs = pd.read_csv(OBS_CSV, parse_dates=["datetime"])
    keep_cols = [
        "station", "datetime",
        "PM25", "PM10", "NO2", "SO2", "O3", "CO", "NOX",
        "WS", "WD", "AT", "RH", "BP", "SolarRad", "Rain",
        "month", "day", "hour", "hour_sin", "hour_cos"
    ]
    obs = obs[keep_cols].copy()
    print(f"Loaded {len(obs)} observation rows", flush=True)
    try:
        with open(OBS_CACHE, 'wb') as f:
            pickle.dump(obs, f)
        print("Saved observation data to cache")
    except Exception as e:
        print(f"Cache save failed: {e}")


# ============================================================
# Helper Functions
# ============================================================
def clamp_value(target: str, x: float) -> float:
    if target == "WD":
        return float(x) % 360.0
    if target in BOUNDS:
        lo, hi = BOUNDS[target]
        return float(min(max(x, lo), hi))
    return float(x)


def add_month_day_features_row(dt: pd.Timestamp, row: dict):
    row["month"] = int(dt.month)
    row["day"] = int(dt.day)


def add_hour_features_row(dt: pd.Timestamp, row: dict):
    h = int(dt.hour)
    row["hour_sin"] = float(np.sin(2 * np.pi * h / 24))
    row["hour_cos"] = float(np.cos(2 * np.pi * h / 24))


def add_wind_components_row(row: dict):
    rad = np.deg2rad(float(row["WD"]))
    row["u"] = float(float(row["WS"]) * np.cos(rad))
    row["v"] = float(float(row["WS"]) * np.sin(rad))


def compute_lag_roll_from_history(hist: pd.DataFrame, col: str, lags, rolls):
    """
    Compute lag and rolling features from history
    hist: sorted by datetime, last row is current time
    """
    out = {}
    
    # Ensure column exists and convert to float
    if col not in hist.columns:
        for L in lags:
            out[f"{col}_lag{L}"] = float("nan")
        for W in rolls:
            out[f"{col}_roll{W}"] = float("nan")
        return out
    
    s = hist[col].astype(float).values
    
    # Lag features: lag L uses value from L hours ago
    # If we have [t-3, t-2, t-1, t], lag1 at time t uses t-1 (index -2)
    for L in lags:
        idx = -(L + 1)
        if len(s) >= abs(idx):
            out[f"{col}_lag{L}"] = float(s[idx])
        else:
            out[f"{col}_lag{L}"] = float("nan")
    
    # Rolling features: exclude current value
    past = s[:-1] if len(s) > 0 else s
    for W in rolls:
        if len(past) == 0:
            out[f"{col}_roll{W}"] = float("nan")
        else:
            window = past[-W:] if len(past) >= W else past
            out[f"{col}_roll{W}"] = float(np.nanmean(window)) if len(window) > 0 else float("nan")
    
    return out


# ============================================================
# Model Cache
# ============================================================
_model_cache: dict[str, tuple[object, object, dict]] = {}
_spatial_model_cache = {}


def load_artifacts(base_target: str):
    if base_target in _model_cache:
        return _model_cache[base_target]

    import tensorflow as tf

    model_path = MODEL_DIR / f"dl_city_predictor_{base_target}_lead1.keras"
    prep_path = MODEL_DIR / f"dl_preprocess_{base_target}_lead1.joblib"
    cfg_path = MODEL_DIR / f"dl_config_{base_target}_lead1.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing preprocess: {prep_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    print(f"Loading model for {base_target}...", flush=True)
    model = tf.keras.models.load_model(model_path)
    preprocess = joblib.load(prep_path)
    cfg = json.loads(cfg_path.read_text())

    _model_cache[base_target] = (model, preprocess, cfg)
    print(f"Model {base_target} loaded", flush=True)
    return model, preprocess, cfg


# ============================================================
# Spatial Interpolation
# ============================================================
def ensemble_spatial_prediction(station_coords, station_values, target_coord):
    """Ensemble spatial prediction with caching"""
    predictions = {}
    station_coords = np.array(station_coords)
    station_values = np.array(station_values)

    # Cache key
    cache_key = tuple(round(v, 1) for v in station_values)

    if cache_key not in _spatial_model_cache:
        cached_models = {}

        # RBF - only create if we have multiple points
        if len(station_coords) > 1:
            try:
                cached_models['rbf'] = Rbf(
                    station_coords[:, 0], station_coords[:, 1], station_values,
                    function='multiquadric', smooth=0
                )
            except:
                cached_models['rbf'] = None  # Will fall back to mean
        else:
            cached_models['rbf'] = None  # Single point - will fall back to mean

        # Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF_kernel(1.0, (0.1, 10))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)
        gp.fit(station_coords, station_values)
        cached_models['gp'] = gp

        # KNN
        knn = KNeighborsRegressor(n_neighbors=len(station_coords), weights='distance')
        knn.fit(station_coords, station_values)
        cached_models['knn'] = knn

        _spatial_model_cache[cache_key] = cached_models

    models = _spatial_model_cache[cache_key]

    # Predictions
    try:
        if models['rbf'] is not None:
            predictions['rbf'] = models['rbf'](target_coord[0], target_coord[1])
        else:
            predictions['rbf'] = np.mean(station_values)
    except:
        predictions['rbf'] = np.mean(station_values)

    try:
        gp_pred, gp_std = models['gp'].predict([target_coord], return_std=True)
        predictions['gp'] = gp_pred[0]
        predictions['gp_std'] = gp_std[0]
    except:
        predictions['gp'] = np.mean(station_values)
        predictions['gp_std'] = np.std(station_values)

    try:
        predictions['knn'] = models['knn'].predict([target_coord])[0]
    except:
        predictions['knn'] = np.mean(station_values)

    # IDW
    def haversine_distance(lat1, lon1, lat2, lon2):
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return 6371 * c

    distances = np.array([haversine_distance(target_coord[0], target_coord[1], c[0], c[1])
                         for c in station_coords])
    weights = 1.0 / (distances ** 2 + 1e-12)
    predictions['idw'] = np.sum(weights * station_values) / np.sum(weights)

    # Ensemble
    ensemble_weights = {'rbf': 0.25, 'gp': 0.25, 'knn': 0.20, 'idw': 0.30}
    ensemble_pred = sum(ensemble_weights[k] * predictions[k] for k in ['rbf', 'gp', 'knn', 'idw'])
    ensemble_uncertainty = predictions['gp_std']

    return ensemble_pred, ensemble_uncertainty, predictions


def ensemble_interpolate_grid_optimized(lats, lons, vals, grid_lat, grid_lon):
    """Vectorized grid interpolation with auxiliary data"""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    vals = np.asarray(vals, dtype=float)
    station_coords = np.column_stack((lats, lons))

    aux_data = get_auxiliary_data()

    # Station features
    station_features = []
    for lat, lon in zip(lats, lons):
        features = aux_data.get_auxiliary_features(lat, lon)
        station_features.append([
            features['elevation'],
            features['is_urban'],
            features['road_distance'],
            features['population_density']
        ])
    station_features = np.array(station_features)

    # Fit models
    rbf_model = Rbf(lats, lons, vals, function='multiquadric', smooth=0)

    extended_coords = np.column_stack([station_coords, station_features])
    kernel = C(1.0, (1e-3, 1e3)) * RBF_kernel(1.0, (0.1, 10))
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=42)
    gp_model.fit(extended_coords, vals)

    knn_model = KNeighborsRegressor(n_neighbors=len(vals), weights='distance')
    knn_model.fit(extended_coords, vals)

    # Grid predictions
    grid_shape = grid_lat.shape
    grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    grid_features = get_grid_auxiliary_features(grid_lat, grid_lon)
    extended_grid = np.column_stack([grid_points, grid_features])

    rbf_preds = np.array([rbf_model(pt[0], pt[1]) for pt in grid_points]).reshape(grid_shape)
    gp_preds, _ = gp_model.predict(extended_grid, return_std=True)
    gp_preds = gp_preds.reshape(grid_shape)
    knn_preds = knn_model.predict(extended_grid).reshape(grid_shape)

    # IDW
    lats_bc = lats.reshape(-1, 1, 1)
    lons_bc = lons.reshape(-1, 1, 1)
    vals_bc = vals.reshape(-1, 1, 1)

    dlat = grid_lat[np.newaxis, :, :] - lats_bc
    dlon = grid_lon[np.newaxis, :, :] - lons_bc
    distances = np.sqrt(dlat**2 + dlon**2)
    distances = np.maximum(distances, 1e-10)

    weights = 1.0 / (distances ** 2)
    idw_preds = np.sum(weights * vals_bc, axis=0) / np.sum(weights, axis=0)

    # Ensemble
    Z = 0.15 * rbf_preds + 0.35 * gp_preds + 0.30 * knn_preds + 0.20 * idw_preds

    return Z


def make_heatmap_png_base64(points, values, title="Heatmap", power=2.0, grid_size=80, target=None):
    """Generate heatmap PNG"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not SL_GEOJSON.exists():
        raise FileNotFoundError(f"Sri Lanka geojson not found: {SL_GEOJSON}")

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    pad_lat, pad_lon = 0.35, 0.55
    lat_min, lat_max = min(lats) - pad_lat, max(lats) + pad_lat
    lon_min, lon_max = min(lons) - pad_lon, max(lons) + pad_lon

    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)

    Z = ensemble_interpolate_grid_optimized(lats, lons, values, grid_lat, grid_lon)

    # ----------------------------------------------------
    # Distance-constrained spatial influence (50 km)
    # ----------------------------------------------------
    max_km = 50

    dist_to_station = np.full(Z.shape, np.inf)
    for (lat_s, lon_s) in points:
        d = haversine_grid(grid_lat, grid_lon, lat_s, lon_s)
        dist_to_station = np.minimum(dist_to_station, d)

    # Soft decay (no hard cutoff)
    decay = np.clip(1 - (dist_to_station / max_km), 0, 1)

    background = np.mean(values)
    Z = Z * decay + background * (1 - decay)

    sl = gpd.read_file(SL_GEOJSON).to_crs("EPSG:4326")
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    sl_clip = gpd.clip(sl, bbox)

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    sl_clip.plot(ax=ax, color="white", edgecolor="black", linewidth=1.2, zorder=1)
    
    local_mask = dist_to_station <= max_km

    Z_local = Z[local_mask & np.isfinite(Z)]

    if len(Z_local) > 0:
        local_min, local_max = np.min(Z_local), np.max(Z_local)
        range_padding = 0.1 * (local_max - local_min) if local_max > local_min else 1.0
        vmin = max(0, local_min - range_padding)
        vmax = local_max + range_padding
    else:
        vmin, vmax = None, None
    
    plt.pcolormesh(grid_lon, grid_lat, Z, shading="auto", alpha=0.65, zorder=2, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    plt.colorbar(label="Predicted (t+1)", shrink=0.8)

    plt.scatter(lons, lats, s=85, c="red", edgecolors="black", linewidths=1.5, zorder=3)
    for (lat, lon), v in zip(points, values):
        plt.text(lon, lat, f"{v:.1f}", fontsize=8, weight="bold", zorder=4,
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
    plt.title(title, fontsize=10, weight='bold')
    plt.xlabel("Longitude", fontsize=9)
    plt.ylabel("Latitude", fontsize=9)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

    bbox_meta = {
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
    }

    return heatmap_b64, bbox_meta


# ============================================================
# Prediction Function
# ============================================================
def predict_one_station(
    model,
    preprocess,
    cfg,
    station: str,
    dt_base: pd.Timestamp,
    overrides: dict | None = None
) -> float:
    """Predict one station at given time"""

    st_obs = obs[obs["station"] == station]
    st_min = st_obs["datetime"].min()
    st_max = st_obs["datetime"].max()

    dt_base = pd.to_datetime(dt_base).floor("h")
    if dt_base < st_min:
        dt_base = pd.to_datetime(st_min).floor("h")
    if dt_base > st_max:
        dt_base = pd.to_datetime(st_max).floor("h")

    lags = tuple(cfg.get("lags", [1, 3, 6, 12, 24]))
    rolls = tuple(cfg.get("rolls", [3, 6, 24]))
    lag_roll_cols = cfg.get("lag_roll_cols", TARGETS)

    max_lag = max(lags) if len(lags) else 1
    need_rows = WINDOW + max_lag

    hist = (
        obs[(obs["station"] == station) & (obs["datetime"] <= dt_base)]
        .sort_values("datetime")
        .tail(need_rows)
    )

    if len(hist) < need_rows:
        t = cfg["base_target"]
        return float(hist.iloc[-1][t]) if len(hist) else 0.0

    # Apply overrides
    if overrides:
        for k, v in overrides.items():
            if k in OVERRIDE_COLS and v is not None:
                hist.loc[hist.index[-1], k] = float(v)

    hist_full = hist.sort_values("datetime").copy().reset_index(drop=True)
    hist_for_seq = hist_full.tail(WINDOW).reset_index(drop=True)

    num_features = cfg["num_features"]
    cat_features = cfg["cat_features"]

    # Build sequence
    seq_rows = []
    for i in range(len(hist_for_seq)):
        dt_i = pd.to_datetime(hist_for_seq.loc[i, "datetime"])
        sub = hist_full[hist_full["datetime"] <= dt_i].copy().sort_values("datetime")

        base = {
            "station": station,
            "lat": STATION_META[station]["lat"],
            "lon": STATION_META[station]["lon"],
        }

        for col in OVERRIDE_COLS:
            base[col] = float(hist_for_seq.loc[i, col])

        add_month_day_features_row(dt_i, base)
        add_hour_features_row(dt_i, base)
        add_wind_components_row(base)

        for c in lag_roll_cols:
            base.update(compute_lag_roll_from_history(sub, c, lags=lags, rolls=rolls))

        seq_rows.append(base)

    X_df = pd.DataFrame(seq_rows).ffill().bfill()
    
    # Handle missing features gracefully
    feature_cols = num_features + cat_features
    missing_cols = [col for col in feature_cols if col not in X_df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing features {missing_cols}, filling with 0")
        for col in missing_cols:
            X_df[col] = 0.0
    
    X_flat = preprocess.transform(X_df[feature_cols]).astype(np.float32)
    X_seq = X_flat.reshape(1, WINDOW, -1)

    pred = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
    return pred


def proxy_based_spatial_evaluation(
    model, preprocess, cfg,
    timestamps: list,
    pollutant: str = "PM25"
) -> pd.DataFrame:
    """
    Proxy-based spatial evaluation using surrogate reference values.

    IMPORTANT: This evaluation uses LSTM predictions as SURROGATE REFERENCE VALUES,
    NOT as true ground truth measurements. The LSTM predictions serve as proxy
    benchmarks to assess spatial interpolation consistency.

    Methodology:
    - At each timestamp, obtain 1-hour ahead LSTM predictions for both stations
    - Scenario A: Use only Battaramulla LSTM prediction to estimate Kandy value
    - Scenario B: Use only Kandy LSTM prediction to estimate Battaramulla value
    - Compare spatial estimates with LSTM predictions (proxy references)

    Limitations:
    - LSTM predictions are model outputs, not measured ground truth
    - Evaluates relative spatial consistency, not absolute accuracy
    - Assumes LSTM predictions are reasonable proxy references

    Parameters:
    ----------
    model : LSTM model (same model used for both stations)
    preprocess : Preprocessor (same for both stations)
    cfg : Configuration dictionary
    timestamps : List of timestamps to evaluate
    pollutant : Target pollutant ("PM25", "PM10", "NO2")

    Returns:
    -------
    pd.DataFrame : Evaluation results with MAE, RMSE, MAPE for both scenarios
    """

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    results = []

    # Station coordinates
    coords = {
        "Battaramulla": (STATION_META["Battaramulla"]["lat"], STATION_META["Battaramulla"]["lon"]),
        "Kandy": (STATION_META["Kandy"]["lat"], STATION_META["Kandy"]["lon"])
    }

    print(f"Running proxy-based spatial evaluation for {pollutant}")
    print(f"Evaluating {len(timestamps)} timestamps...")

    for i, timestamp in enumerate(timestamps):
        if i % 50 == 0:
            print(f"Processing timestamp {i+1}/{len(timestamps)}: {timestamp}")

        try:
            # Get LSTM predictions (surrogate reference values) for both stations
            lstm_b = predict_one_station(
                model, preprocess, cfg, "Battaramulla", timestamp
            )
            lstm_k = predict_one_station(
                model, preprocess, cfg, "Kandy", timestamp
            )

            # Skip if either prediction is invalid or too small for MAPE calculation
            if (lstm_b is None or lstm_k is None or
                lstm_b <= 1e-6 or lstm_k <= 1e-6 or
                not np.isfinite(lstm_b) or not np.isfinite(lstm_k)):
                continue

            # Scenario A: Use only Battaramulla LSTM to predict at Kandy
            spatial_pred_k = ensemble_spatial_prediction(
                [coords["Battaramulla"]], [lstm_b], coords["Kandy"]
            )[0]  # Take ensemble prediction

            # Scenario B: Use only Kandy LSTM to predict at Battaramulla
            spatial_pred_b = ensemble_spatial_prediction(
                [coords["Kandy"]], [lstm_k], coords["Battaramulla"]
            )[0]  # Take ensemble prediction

            # Store results for this timestamp
            results.append({
                "timestamp": timestamp,
                "scenario": "A_Battaramulla_to_Kandy",
                "lstm_reference": lstm_k,  # Kandy LSTM as proxy reference
                "spatial_prediction": spatial_pred_k,
                "absolute_error": abs(spatial_pred_k - lstm_k),
                "squared_error": (spatial_pred_k - lstm_k) ** 2,
                "percentage_error": abs((spatial_pred_k - lstm_k) / lstm_k) * 100 if lstm_k != 0 else np.nan
            })

            results.append({
                "timestamp": timestamp,
                "scenario": "B_Kandy_to_Battaramulla",
                "lstm_reference": lstm_b,  # Battaramulla LSTM as proxy reference
                "spatial_prediction": spatial_pred_b,
                "absolute_error": abs(spatial_pred_b - lstm_b),
                "squared_error": (spatial_pred_b - lstm_b) ** 2,
                "percentage_error": abs((spatial_pred_b - lstm_b) / lstm_b) * 100 if lstm_b != 0 else np.nan
            })

        except Exception as e:
            print(f"Error at timestamp {timestamp}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No valid results generated")
        return pd.DataFrame()

    # Calculate aggregate metrics for each scenario
    summary_results = []

    for scenario in df["scenario"].unique():
        scenario_data = df[df["scenario"] == scenario]

        # Remove NaN values for MAPE calculation
        valid_data = scenario_data.dropna(subset=["percentage_error"])

        mae = scenario_data["absolute_error"].mean()
        rmse = np.sqrt(scenario_data["squared_error"].mean())
        mape = valid_data["percentage_error"].mean() if len(valid_data) > 0 else np.nan

        summary_results.append({
            "scenario": scenario,
            "n_samples": len(scenario_data),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mape_valid_samples": len(valid_data)
        })

    summary_df = pd.DataFrame(summary_results)

    print("\\nEvaluation complete!")
    print(f"Processed {len(timestamps)} timestamps")
    print(f"Generated {len(df)} prediction pairs")
    print("\\nSummary metrics:")
    print(summary_df.round(3))

    return summary_df

    print(f"Running proxy-based spatial evaluation for {pollutant}")
    print(f"Evaluating {len(timestamps)} timestamps...")

    for i, timestamp in enumerate(timestamps):
        if i % 50 == 0:
            print(f"Processing timestamp {i+1}/{len(timestamps)}: {timestamp}")

        try:
            # Get LSTM predictions (surrogate reference values)
            lstm_b = predict_one_station(
                model_battaramulla, preprocess_battaramulla, cfg_battaramulla,
                "Battaramulla", timestamp
            )
            lstm_k = predict_one_station(
                model_kandy, preprocess_kandy, cfg_kandy,
                "Kandy", timestamp
            )

            # Skip if either prediction is invalid
            if lstm_b <= 0 or lstm_k <= 0:
                continue

            # Scenario A: Use only Battaramulla LSTM to predict at Kandy
            spatial_pred_k = ensemble_spatial_prediction(
                [coords["Battaramulla"]], [lstm_b], coords["Kandy"]
            )[0]  # Take ensemble prediction

            # Scenario B: Use only Kandy LSTM to predict at Battaramulla
            spatial_pred_b = ensemble_spatial_prediction(
                [coords["Kandy"]], [lstm_k], coords["Battaramulla"]
            )[0]  # Take ensemble prediction

            # Store results for this timestamp
            results.append({
                "timestamp": timestamp,
                "scenario": "A_Battaramulla_to_Kandy",
                "lstm_reference": lstm_k,  # Kandy LSTM as proxy reference
                "spatial_prediction": spatial_pred_k,
                "absolute_error": abs(spatial_pred_k - lstm_k),
                "squared_error": (spatial_pred_k - lstm_k) ** 2,
                "percentage_error": abs((spatial_pred_k - lstm_k) / lstm_k) * 100 if lstm_k != 0 else np.nan
            })

            results.append({
                "timestamp": timestamp,
                "scenario": "B_Kandy_to_Battaramulla",
                "lstm_reference": lstm_b,  # Battaramulla LSTM as proxy reference
                "spatial_prediction": spatial_pred_b,
                "absolute_error": abs(spatial_pred_b - lstm_b),
                "squared_error": (spatial_pred_b - lstm_b) ** 2,
                "percentage_error": abs((spatial_pred_b - lstm_b) / lstm_b) * 100 if lstm_b != 0 else np.nan
            })

        except Exception as e:
            print(f"Error at timestamp {timestamp}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No valid results generated")
        return pd.DataFrame()

    # Calculate aggregate metrics for each scenario
    summary_results = []

    for scenario in df["scenario"].unique():
        scenario_data = df[df["scenario"] == scenario]

        # Remove NaN values for MAPE calculation
        valid_data = scenario_data.dropna(subset=["percentage_error"])

        mae = scenario_data["absolute_error"].mean()
        rmse = np.sqrt(scenario_data["squared_error"].mean())
        mape = valid_data["percentage_error"].mean() if len(valid_data) > 0 else np.nan

        summary_results.append({
            "scenario": scenario,
            "n_samples": len(scenario_data),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mape_valid_samples": len(valid_data)
        })

    summary_df = pd.DataFrame(summary_results)

    print("\\nEvaluation complete!")
    print(f"Processed {len(timestamps)} timestamps")
    print(f"Generated {len(df)} prediction pairs")
    print("\\nSummary metrics:")
    print(summary_df.round(3))

    return summary_df