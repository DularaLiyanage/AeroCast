# backend/download_models.py
import os
import urllib.request

try:
    import gdown  # pip install gdown
    GDOWN_AVAILABLE = True
except ImportError:
    print("⚠️  gdown not installed - Google Drive downloads will not work")
    GDOWN_AVAILABLE = False

# Only the spatial auxiliary cache file is needed for predictions
# This contains processed geospatial data (elevation, landcover, population, roads)
LARGE_FILES = {
    "backend/app/features/spatial_interpolation/sri_lanka_data/spatial_auxiliary_cache.pkl": "YOUR_GOOGLE_DRIVE_FILE_ID_FOR_CACHE",
}

def download_if_missing():
    """Download the spatial auxiliary cache file if missing."""
    for filepath, file_id in LARGE_FILES.items():
        if file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_FOR_CACHE":
            print(f"⚠️  Skipping {filepath} - Google Drive file ID not configured")
            continue

        if not os.path.exists(filepath):
            print(f"📥 Downloading spatial auxiliary cache: {filepath}")
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                if not GDOWN_AVAILABLE:
                    print(f"❌ Cannot download {filepath} - gdown not available")
                    continue
                gdown.download(f"https://drive.google.com/uc?id={file_id}", filepath, quiet=False)
                print(f"✅ Downloaded: {filepath}")
            except Exception as e:
                print(f"❌ Failed to download {filepath}: {e}")
        else:
            print(f"✅ Spatial auxiliary cache already exists: {filepath}")

if __name__ == "__main__":
    download_if_missing()