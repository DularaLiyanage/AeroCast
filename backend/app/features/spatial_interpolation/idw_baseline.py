import pandas as pd, numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from pathlib import Path

CSV = Path("clean/cea_hourly_2019_2024_clean.csv")
OUT = Path("outputs_idw"); OUT.mkdir(exist_ok=True)

# ----------------------------
# 1) Load + pick slice
# ----------------------------
POLLUTANT = "PM25"      
WHEN = "2024-12-31 12:00:00"  

df = pd.read_csv(CSV, parse_dates=["datetime"])
WHEN_TS = pd.to_datetime(WHEN)
mask = df["datetime"] == WHEN_TS
slice_ = df.loc[mask, ["station","lat","lon", POLLUTANT]].dropna(subset=[POLLUTANT])

if slice_.empty:
    raise SystemExit(f"No data for {WHEN} {POLLUTANT}")

g = gpd.GeoDataFrame(
    slice_,
    geometry=[Point(xy) for xy in zip(slice_["lon"], slice_["lat"])],
    crs="EPSG:4326"  # WGS84 lon/lat
)

# ----------------------------
# 2) Work in meters (UTM 44N)
# ----------------------------
g_m = g.to_crs(32644)  # EPSG:32644 (UTM zone 44N, meters)

# ----------------------------
# 3) Build ~1 km grid over bbox (+ padding)
# ----------------------------
pad_km = 20_000  # 20 km padding
minx, miny, maxx, maxy = g_m.total_bounds
minx -= pad_km; miny -= pad_km; maxx += pad_km; maxy += pad_km

res = 1_000  # 1 km
xs = np.arange(minx, maxx, res)
ys = np.arange(miny, maxy, res)

xx, yy = np.meshgrid(xs, ys)
grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

# ----------------------------
# 4) IDW function (power p=2)
# ----------------------------
def idw_predict(xy_known, v_known, xy_query, p=2, eps=1e-9, rmax=None):

    out = np.empty(len(xy_query), dtype=float)
    for i, q in enumerate(xy_query):
        d = np.linalg.norm(xy_known - q, axis=1)
        # exact hit
        j0 = np.argmin(d)
        if d[j0] < eps:
            out[i] = v_known[j0]; continue
        if rmax is not None:
            mask = d <= rmax
            if not mask.any():
                out[i] = np.nan; continue
            d = d[mask]; vals = v_known[mask]
        else:
            vals = v_known
        w = 1.0 / (d ** p)
        out[i] = np.sum(w * vals) / np.sum(w)
    return out

xy_known = np.vstack([g_m.geometry.x.values, g_m.geometry.y.values]).T
v_known  = g_m[POLLUTANT].values.astype(float)

pred = idw_predict(xy_known, v_known, grid_pts, p=2, rmax=None)

grid = gpd.GeoDataFrame(
    {"value": pred},
    geometry=[Point(x, y) for x, y in grid_pts],
    crs=32644
)

# ----------------------------
# 5) Save outputs
# ----------------------------
grid_proj_csv = OUT / f"idw_{POLLUTANT}_{WHEN.replace(':','-').replace(' ','_')}_utm.csv"
grid.to_csv(grid_proj_csv, index=False)

grid_ll = grid.to_crs(4326)
grid_ll_csv = OUT / f"idw_{POLLUTANT}_{WHEN.replace(':','-').replace(' ','_')}_wgs84.csv"
grid_ll.to_csv(grid_ll_csv, index=False)

print(f"[OK] Saved grid (UTM):  {grid_proj_csv}")
print(f"[OK] Saved grid (WGS84): {grid_ll_csv}")

try:
    import rasterio
    from rasterio.transform import from_origin
    arr = grid["value"].to_numpy().reshape(len(ys), len(xs))
    transform = from_origin(minx, maxy, res, res)  
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "crs": "EPSG:32644",
        "transform": transform,
        "nodata": np.nan
    }
    tif = OUT / f"idw_{POLLUTANT}_{WHEN.replace(':','-').replace(' ','_')}.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"[OK] Saved GeoTIFF:   {tif}")
except Exception as e:
    print("[INFO] Skipped GeoTIFF (install rasterio to enable).", e)

# ---- Quick IDW parameter sweep (optional) ----
from math import sqrt
def rmse(a,b): 
    a=np.asarray(a); b=np.asarray(b)
    return float(np.sqrt(np.nanmean((a-b)**2)))

def loso_rmse_for(powers=(1.5,2,3)):
    results=[]
    for p in powers:
        # leave-one-station-out on THIS timestamp
        by_stn = g_m.groupby("station")
        for stn, test_df in by_stn:
            train_df = g_m.loc[g_m["station"] != stn]
            xy_tr = np.c_[train_df.geometry.x.values, train_df.geometry.y.values]
            v_tr  = train_df[POLLUTANT].values.astype(float)
            xy_te = np.c_[test_df.geometry.x.values, test_df.geometry.y.values]
            v_te  = test_df[POLLUTANT].values.astype(float)
            pred_te = idw_predict(xy_tr, v_tr, xy_te, p=p)
            results.append({"power":p,"station":stn,"RMSE":rmse(v_te,pred_te)})
    return pd.DataFrame(results)

sweep = loso_rmse_for()
sweep.to_csv(OUT / f"idw_param_sweep_{POLLUTANT}_{WHEN.replace(':','-').replace(' ','_')}.csv", index=False)
print(sweep)
