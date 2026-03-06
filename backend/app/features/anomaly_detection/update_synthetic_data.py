import pandas as pd
import numpy as np

features = [
    'AT', 'RH', 'BP', 'Solar Rad', 'Rain Gauge', 'WD Raw', 
    'O3 Conc', 'NO Conc', 'NO2 Conc', 'NOx Conc', 'SO2 Conc', 'PM2.5 Conc', 'PM10 Conc'
]

def generate_noise(size):
    return np.random.rand(size) * 10

# Generate 500 hours for both locations
data = []
for loc in ["Battaramulla", "Kandy"]:
    # Slightly different base values for each city to show diversity
    bias = 5 if loc == "Battaramulla" else 15 
    for i in range(500):
        row = {f: (np.random.rand() * 20 + bias) for f in features}
        row['Location'] = loc
        row['Time'] = pd.Timestamp('2026-01-01') + pd.Timedelta(hours=i)
        data.append(row)

df = pd.DataFrame(data)
df.to_parquet('Pollution_Research_Hourly_Series.parquet')
print("Pollution_Research_Hourly_Series.parquet updated with Battaramulla and Kandy data.")
