# backend/download_models.py
import os
import urllib.request

# Map each file to its Zenodo download URL
# Replace 'YOUR_ZENODO_URL_X' with actual Zenodo download URLs
# URLs will look like: https://zenodo.org/records/1234567/files/filename.joblib?download=1
LARGE_FILES = {
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_NO2_lead1.joblib":  "YOUR_ZENODO_URL_1",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM10_lead1.joblib": "YOUR_ZENODO_URL_2",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM25_lead1.joblib": "YOUR_ZENODO_URL_3",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_NO2.joblib":  "YOUR_ZENODO_URL_4",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM10.joblib": "YOUR_ZENODO_URL_5",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM25.joblib": "YOUR_ZENODO_URL_6",
    "backend/app/features/spatial_interpolation/sri_lanka_data/spatial_auxiliary_cache.pkl": "YOUR_ZENODO_URL_7",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.dbf": "YOUR_ZENODO_URL_8",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.shp": "YOUR_ZENODO_URL_9",
}

def download_if_missing():
    """Download large files from Zenodo if they don't exist locally."""
    for filepath, url in LARGE_FILES.items():
        if url == "YOUR_ZENODO_URL_X" or url.startswith("YOUR_ZENODO_URL_"):
            print(f"⚠️  Skipping {filepath} - Zenodo URL not configured")
            continue

        if not os.path.exists(filepath):
            print(f"📥 Downloading missing file: {filepath}")
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded: {filepath}")
            except Exception as e:
                print(f"❌ Failed to download {filepath}: {e}")
        else:
            print(f"✅ Already exists, skipping: {filepath}")

if __name__ == "__main__":
    download_if_missing()