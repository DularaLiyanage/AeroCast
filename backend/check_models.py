# backend/download_models.py
import os

def check_required_files():
    """Check if required large files exist locally."""
    required_files = [
        "app/features/spatial_interpolation/sri_lanka_data/spatial_auxiliary_cache.pkl",
    ]

    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"❌ Missing required file: {filepath}")
            print("   Please manually copy the file to this location.")
        else:
            print(f"✅ Found: {filepath}")

if __name__ == "__main__":
    check_required_files()