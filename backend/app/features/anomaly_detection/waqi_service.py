import requests
import os

class WAQIService:
    def __init__(self, token: str = "demo"):
        """
        token: API token from aqicn.org. 'demo' is supported for some stations.
        """
        self.token = token
        self.base_url = "https://api.waqi.info/feed"

    def get_by_station_name(self, station_name: str):
        """
        Example station names: 'battaramulla', 'kandy'
        """
        url = f"{self.base_url}/{station_name}/?token={self.token}"
        response = requests.get(url)
        return response.json()

    def get_by_lat_lon(self, lat: float, lon: float):
        url = f"{self.base_url}/geo:{lat};{lon}/?token={self.token}"
        response = requests.get(url)
        return response.json()

    def map_to_model_features(self, waqi_data: dict):
        """
        Maps WAQI JSON response to the 13 features expected by the model.
        Returns a dict of feature names and values.
        """
        if not isinstance(waqi_data, dict) or waqi_data.get("status") != "ok":
            return None
            
        data = waqi_data["data"]["iaqi"]
        # Feature list from 1.5 notebook:
        # ['AT', 'RH', 'BP', 'Solar Rad', 'Rain Gauge', 'WD Raw', 
        #  'O3 Conc', 'NO Conc', 'NO2 Conc', 'NOx Conc', 'SO2 Conc', 'PM2.5 Conc', 'PM10 Conc']
        
        mapping = {
            "AT": data.get("t", {}).get("v", 25),      # Temperature
            "RH": data.get("h", {}).get("v", 80),      # Humidity
            "BP": data.get("p", {}).get("v", 1013),    # Pressure
            "Solar Rad": 0.0, # Not usually in WAQI, default or zero
            "Rain Gauge": 0.0,
            "WD Raw": data.get("wd", {}).get("v", 0),
            "O3 Conc": data.get("o3", {}).get("v", 0),
            "NO Conc": 0.0,
            "NO2 Conc": data.get("no2", {}).get("v", 0),
            "NOx Conc": 0.0,
            "SO2 Conc": data.get("so2", {}).get("v", 0),
            "PM2.5 Conc": data.get("pm25", {}).get("v", 0),
            "PM10 Conc": data.get("pm10", {}).get("v", 0)
        }
        return mapping
