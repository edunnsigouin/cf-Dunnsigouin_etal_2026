"""
Create an xy (lat,lon) weight field in [0,1] that maps an ERA5-style model grid to an
NVE regine catchment by area fraction.

Method (Shapely + pyproj area-fraction):
- Treat each model grid cell (dlon x dlat in degrees) as a polygon in EPSG:4326.
- Reproject catchment and grid-cell polygons to a metric CRS (EPSG:25833).
- weight = area(catchment ∩ cell) / area(cell)

Includes:
1) Compute weights on the full model grid.
2) Write the weights to NetCDF (xarray DataArray) for later use.
"""

import json
import numpy as np
import xarray as xr

from shapely.geometry import shape, box
from shapely.ops import unary_union, transform
from shapely.prepared import prep
from pyproj import Transformer

from Dunnsigouin_etal_2026 import config


# input ------------------------------------------------------------------
resolution            = "0.25x0.25"
path_in_catchment     = config.dirs["nve"]
dlon                  = 0.25  # degrees
dlat                  = 0.25
dst_epsg              = "EPSG:25833"

catchment             = "regine_drammen"
filename_in_catchment = f"{path_in_catchment}catchment_nve_{catchment}.geojson"
out_xy                = f"{path_in_catchment}weights_catchment_{catchment}_era5_{resolution}.nc"
write2file            = True
# -------------------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dissolve_catchment_geometry(gj: dict):
    """
    Dissolve all polygonal features into a single Shapely geometry
    (Polygon or MultiPolygon) in lon/lat (EPSG:4326).
    """
    geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
    if not geoms:
        raise ValueError("No geometries found in catchment GeoJSON.")
    out = unary_union(geoms)
    if out.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {out.geom_type}")
    return out


def make_model_grid(resolution: str):
    """
    Hard-coded grid (cell centers).
    Returns:
      model_latitude  (descending)
      model_longitude (ascending)
    """
    if resolution == "0.5x0.5":
        model_latitude = np.arange(73.5, 32.5, -0.5)
        model_longitude = np.arange(-27.0, 45.5, 0.5)
    elif resolution == "0.25x0.25":
        model_latitude = np.arange(73.5, 32.25, -0.5)
        model_longitude = np.arange(-27.0, 45.25, 0.5)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    return model_latitude, model_longitude


def centers_to_edges(centers: np.ndarray, d: float) -> np.ndarray:
    """
    Convert 1D grid centers to edges for uniform spacing d.
    Works for BOTH ascending and descending center arrays.

    Ascending centers:  [c0-d/2, c0+d/2, c1+d/2, ..., cN-1+d/2]
    Descending centers: [c0+d/2, c0-d/2, c1-d/2, ..., cN-1-d/2]
    """
    centers = np.asarray(centers)
    if centers.size < 2:
        raise ValueError("Need at least 2 centers to infer edges.")

    if centers[1] > centers[0]:
        return np.concatenate(([centers[0] - d / 2], centers + d / 2))
    else:
        return np.concatenate((centers + d / 2, [centers[-1] - d / 2]))


def build_projector(src_epsg: str, dst_epsg: str):
    """Return a Shapely-compatible coordinate transform function."""
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transformer.transform


def compute_area_fraction_weights_on_model_grid(
    catchment_ll,
    model_latitude: np.ndarray,
    model_longitude: np.ndarray,
    dlat: float,
    dlon: float,
    dst_epsg: str,
):
    """
    Compute weights on the model grid as area fractions:
      weight[i,j] = area(catchment ∩ cell_ij) / area(cell_ij)
    """
    proj_fn = build_projector("EPSG:4326", dst_epsg)

    # Reproject catchment to metric CRS for area computations
    catchment = transform(proj_fn, catchment_ll)
    if not catchment.is_valid:
        catchment = catchment.buffer(0)

    catchment_prep = prep(catchment)

    lon_edges = centers_to_edges(model_longitude, dlon)
    lat_edges = centers_to_edges(model_latitude, dlat)

    weights = np.zeros((len(model_latitude), len(model_longitude)), dtype=np.float32)

    for i in range(len(model_latitude)):
        lat0 = min(lat_edges[i], lat_edges[i + 1])
        lat1 = max(lat_edges[i], lat_edges[i + 1])

        for j in range(len(model_longitude)):
            lon0 = min(lon_edges[j], lon_edges[j + 1])
            lon1 = max(lon_edges[j], lon_edges[j + 1])

            # Cell polygon defined in lon/lat, then projected to metric CRS
            cell_ll = box(lon0, lat0, lon1, lat1)
            cell = transform(proj_fn, cell_ll)

            if not catchment_prep.intersects(cell):
                continue

            inter = catchment.intersection(cell)
            if (not inter.is_empty) and (cell.area > 0):
                weights[i, j] = inter.area / cell.area

    return weights


def weights_to_xarray(weights, model_latitude, model_longitude, dst_epsg):
    """Wrap weights into an xarray DataArray with coordinates and metadata."""
    return xr.DataArray(
        weights,
        dims=("latitude", "longitude"),
        coords={"latitude": model_latitude, "longitude": model_longitude},
        name="catchment_weight",
        attrs={
            "description": "Area fraction of model grid cell within catchment",
            "range": "[0,1]",
            "area_crs": dst_epsg,
        },
    )


# ------------------------------------------- main -------------------------------------------

if __name__ == "__main__":

    # Read and dissolve catchment polygons (EPSG:4326 lon/lat)
    gj = read_geojson(filename_in_catchment)
    catchment_ll = dissolve_catchment_geometry(gj)

    # Build model grid (cell centers)
    model_latitude, model_longitude = make_model_grid(resolution)

    # Compute area-fraction weights on the full grid
    weights = compute_area_fraction_weights_on_model_grid(
        catchment_ll=catchment_ll,
        model_latitude=model_latitude,
        model_longitude=model_longitude,
        dlat=dlat,
        dlon=dlon,
        dst_epsg=dst_epsg,
    )

    # Package as DataArray and write to NetCDF
    da_weights = weights_to_xarray(weights, model_latitude, model_longitude, dst_epsg)

    if write2file:
        da_weights.to_netcdf(out_xy)
        print("Wrote:", out_xy)
