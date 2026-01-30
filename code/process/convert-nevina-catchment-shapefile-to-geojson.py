"""
Converts catchment shapefiles from NVE's nevina webpage into geojson 
for later use 
"""

import json
import numpy               as np
import xarray              as xr
import geopandas           as gpd
from Dunnsigouin_etal_2026 import config, misc

# input ---------------------------------------
name         = 'losna'
path_in      = config.dirs['nve_catchment']
path_out     = config.dirs['nve_catchment']
filename_in  = f'{path_in}{name}-vassdraget/zipfolder/NedbfeltF_v4.shp' 
filename_out = f'{path_out}nve_nevina_{name}vassdraget.geojson' 
write2file   = False
# ---------------------------------------------

# 1) Read the shapefile (just point to the .shp)
gdf = gpd.read_file(filename_in)

# 2) (Optional but common) reproject to WGS84 so the GeoJSON is "web-friendly"
#    GeoJSON is typically expected to be EPSG:4326 (lon/lat)
gdf = gdf.to_crs(epsg=4326)

# 3) Write to GeoJSON
if write2file:
    gdf.to_file(filename_out, driver="GeoJSON")

"""    
def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

gj1 = read_geojson(filename_out)
gj2 = read_geojson('/nird/datapeak/NS9873K/etdu/processed/cf-Dunnsigouin_etal_2026/nve_catchment/nve_regine_enhet_002_glommavassdraget_entire_catchment.geojson')

print(gj2)
"""
