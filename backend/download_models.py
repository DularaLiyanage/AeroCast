# backend/download_models.py
import os
import urllib.request

try:
    import gdown  # pip install gdown
    GDOWN_AVAILABLE = True
except ImportError:
    print("⚠️  gdown not installed - Google Drive downloads will not work")
    GDOWN_AVAILABLE = False

# Map each file to its download source
# For Google Drive: use file ID (e.g., "1aBcDeFgHiJkLmNoPqRsTuVwXyZ")
# For Zenodo: use full download URL (e.g., "https://zenodo.org/records/1234567/files/filename?download=1")
LARGE_FILES = {
    # Google Drive files (use file IDs)
    "backend/app/features/spatial_interpolation/sri_lanka_data/spatial_auxiliary_cache.pkl": "YOUR_GOOGLE_DRIVE_FILE_ID_FOR_CACHE",
    
    # Zenodo files (use full URLs)
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_NO2_lead1.joblib":  "YOUR_ZENODO_URL_1",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM10_lead1.joblib": "YOUR_ZENODO_URL_2",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM25_lead1.joblib": "YOUR_ZENODO_URL_3",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_NO2.joblib":  "YOUR_ZENODO_URL_4",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM10.joblib": "YOUR_ZENODO_URL_5",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM25.joblib": "YOUR_ZENODO_URL_6",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.dbf": "YOUR_ZENODO_URL_8",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.shp": "YOUR_ZENODO_URL_9",
}

def download_if_missing():
    """Download large files from Google Drive or Zenodo if they don't exist locally."""
    for filepath, source in LARGE_FILES.items():
        if source.startswith("YOUR_"):
            print(f"⚠️  Skipping {filepath} - Download source not configured")
            continue

        if not os.path.exists(filepath):
            print(f"📥 Downloading missing file: {filepath}")
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Check if it's a Google Drive file ID or Zenodo URL
                if source.startswith("https://"):
                    # Zenodo URL
                    urllib.request.urlretrieve(source, filepath)
                else:
                    # Google Drive file ID
                    if not GDOWN_AVAILABLE:
                        print(f"❌ Cannot download {filepath} - gdown not available for Google Drive")
                        continue
                    gdown.download(f"https://drive.google.com/uc?id={source}", filepath, quiet=False)
                
                print(f"✅ Downloaded: {filepath}")
            except Exception as e:
                print(f"❌ Failed to download {filepath}: {e}")
        else:
            print(f"✅ Already exists, skipping: {filepath}")

if __name__ == "__main__":
    download_if_missing()