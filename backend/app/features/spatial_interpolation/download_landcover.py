import requests
from pathlib import Path
import time

# Create data folder on your Desktop
data_dir = Path.home() / "Desktop" / "sri_lanka_data"
data_dir.mkdir(exist_ok=True)

print(f"📁 Saving files to: {data_dir}\n")

# CORRECT URLs for ESA WorldCover v200 (2021)
# Sri Lanka is covered by these tiles
tiles = [
    ("N06E078", "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N06E078_Map.tif"),
    ("N06E081", "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N06E081_Map.tif"),
    ("N09E078", "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N09E078_Map.tif"),
    ("N09E081", "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N09E081_Map.tif"),
]

success_count = 0
failed_tiles = []

for name, url in tiles:
    filename = data_dir / f"landcover_{name}.tif"
    
    # Skip if already downloaded
    if filename.exists():
        print(f"✓ {name} already exists, skipping...")
        success_count += 1
        continue
    
    print(f"⬇️  Downloading tile {name}...")
    print(f"    URL: {url}")
    
    try:
        # Try to download
        response = requests.get(url, stream=True, timeout=30)
        
        # Check if URL exists
        if response.status_code == 404:
            print(f"⚠️  Tile {name} not found (404) - might not cover Sri Lanka")
            failed_tiles.append(name)
            continue
        
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        total_mb = total_size / (1024 * 1024)
        print(f"    File size: {total_mb:.1f} MB")
        
        # Download with progress
        with open(filename, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                # Show progress every 10MB
                if total_size > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                    percent = (downloaded / total_size) * 100
                    mb_done = downloaded / (1024 * 1024)
                    print(f"    Progress: {mb_done:.1f}/{total_mb:.1f} MB ({percent:.0f}%)")
        
        print(f"✅ {name} downloaded successfully!\n")
        success_count += 1
        time.sleep(1)  # Small delay between downloads
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {name}: {e}\n")
        failed_tiles.append(name)
        # Clean up partial download
        if filename.exists():
            filename.unlink()

print("\n" + "="*50)
print(f"📊 Download Summary:")
print(f"   ✅ Successful: {success_count}/{len(tiles)}")
if failed_tiles:
    print(f"   ❌ Failed: {', '.join(failed_tiles)}")
print("="*50)

if success_count > 0:
    print(f"\n🎉 Downloaded {success_count} tile(s) to: {data_dir}")
    print("\nTo view the files, you can use QGIS or Python with rasterio/geopandas")
else:
    print("\n⚠️  No tiles downloaded. Trying alternative method...")
    print("\nAlternative: Use Google Earth Engine (see instructions below)")