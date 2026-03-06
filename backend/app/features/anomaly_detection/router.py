from fastapi import APIRouter

router = APIRouter()


# Config - Make sure the 'forecast' folder exists in your root directory
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
def test_component1():
    return {"message": "Component 1 working"}