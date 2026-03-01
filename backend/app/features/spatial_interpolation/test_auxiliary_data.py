#!/usr/bin/env python3
"""
Test script to verify geospatial auxiliary data integration
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from api_server import get_auxiliary_data

def test_auxiliary_data():
    """Test that auxiliary data is loaded and working"""
    print("🔍 Testing geospatial auxiliary data integration...")

    # Get auxiliary data instance
    aux_data = get_auxiliary_data()

    # Show what data sources are loaded
    print("\n📂 Data sources loaded:")
    print(f"   Elevation: {'✅' if aux_data.elevation_data else '❌'}")
    print(f"   Land cover tiles: {len(aux_data.landcover_data)}")
    print(f"   Population: {'✅' if aux_data.population_data else '❌'}")
    print(f"   Roads: {'✅' if aux_data.roads_data is not None else '❌'}")

    # Test locations
    test_locations = [
        (6.927, 79.861, "Colombo"),
        (6.901, 79.926, "Battaramulla"),
        (7.292, 80.635, "Kandy"),
        (6.500, 80.000, "Rural area")
    ]

    print("\n📊 Auxiliary features for test locations:")
    print("-" * 60)

    for lat, lon, name in test_locations:
        features = aux_data.get_auxiliary_features(lat, lon)
        print(f"\n📍 {name} ({lat:.3f}, {lon:.3f}):")
        print(".1f")
        print(".3f")
        print(".1f")
        print(".0f")

    print("\n✅ Auxiliary data test completed!")
    print("\n💡 If you see real elevation/population values, geospatial data is loaded!")
    print("💡 If you see fallback values, check your data files in sri_lanka_data/")

if __name__ == "__main__":
    test_auxiliary_data()