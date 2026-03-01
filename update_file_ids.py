# Helper script to update Google Drive file IDs
# Run this after uploading files to Google Drive

# Replace these with your actual Google Drive file IDs
# Get file ID from share link: https://drive.google.com/file/d/FILE_ID_HERE/view

FILE_IDS = {
    "city_predictor_NO2": "YOUR_FILE_ID_HERE",
    "city_predictor_PM10": "YOUR_FILE_ID_HERE",
    "city_predictor_PM25": "YOUR_FILE_ID_HERE",
    "rf_NO2": "YOUR_FILE_ID_HERE",
    "rf_PM10": "YOUR_FILE_ID_HERE",
    "rf_PM25": "YOUR_FILE_ID_HERE",
    "spatial_auxiliary_cache": "YOUR_FILE_ID_HERE",
    "gis_buildings_dbf": "YOUR_FILE_ID_HERE",
    "gis_buildings_shp": "YOUR_FILE_ID_HERE",
}

print("Update these file IDs in backend/download_models.py:")
for name, file_id in FILE_IDS.items():
    print(f'    "{name}": "{file_id}",')