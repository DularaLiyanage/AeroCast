# backend/download_models.py
import os
import gdown  # pip install gdown

# Map each file to its Google Drive file ID
# Get file ID from share link: https://drive.google.com/file/d/FILE_ID_HERE/view
# Replace 'YOUR_FILE_ID_X' with actual Google Drive file IDs
LARGE_FILES = {
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_NO2_lead1.joblib":  "YOUR_FILE_ID_1",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM10_lead1.joblib": "YOUR_FILE_ID_2",
    "backend/app/features/spatial_interpolation/city_predictor_model/city_predictor_PM25_lead1.joblib": "YOUR_FILE_ID_3",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_NO2.joblib":  "YOUR_FILE_ID_4",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM10.joblib": "YOUR_FILE_ID_5",
    "backend/app/features/spatial_interpolation/ml_baseline/rf_PM25.joblib": "YOUR_FILE_ID_6",
    "backend/app/features/spatial_interpolation/sri_lanka_data/spatial_auxiliary_cache.pkl": "YOUR_FILE_ID_7",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.dbf": "YOUR_FILE_ID_8",
    "backend/app/features/spatial_interpolation/sri_lanka_data/sri-lanka-251231-free.shp/gis_osm_buildings_a_free_1.shp": "YOUR_FILE_ID_9",
}

def download_if_missing():
    """Download large files from Google Drive if they don't exist locally."""
    for filepath, file_id in LARGE_FILES.items():
        if file_id == "YOUR_FILE_ID_X" or file_id.startswith("YOUR_FILE_ID_"):
            print(f"⚠️  Skipping {filepath} - File ID not configured")
            continue

        if not os.path.exists(filepath):
            print(f"📥 Downloading missing file: {filepath}")
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    filepath,
                    quiet=False
                )
                print(f"✅ Downloaded: {filepath}")
            except Exception as e:
                print(f"❌ Failed to download {filepath}: {e}")
        else:
            print(f"✅ Already exists, skipping: {filepath}")

if __name__ == "__main__":
    download_if_missing()