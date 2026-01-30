"""
Makes an xy file with weights [0,1] mapping the model grid to an NVE regine catchment.

Method using area-fraction using Shapely + pyproj:
- Treat each model grid cell (0.5x0.5) as a polygon.
- Project catchment + grid-cell polygons to a metric CRS (EPSG:25833).
- Weight = area(catchment ∩ cell) / area(cell)

Note: tested an alternate method that interpolates the low res grid to high res,
then labels the high res grid with 1/0 if in catchement, then aggregates the high-res
grid to low grid to define a 0.5x0.5 grid of weights. The results are the same, but slower.
"""

import json
import numpy               as np
import xarray              as xr
from shapely.geometry      import shape, box
from shapely.ops           import unary_union, transform
from shapely.prepared      import prep
from pyproj                import Transformer
from Dunnsigouin_etal_2026 import config, misc


# input ------------------------------------------------------------------
resolution            = "0.5x0.5"
path_in_catchment     = config.dirs["nve_catchment"]
dlon                  = 0.5 # Grid spacing in degrees (for the model grid you hard-coded) 
dlat                  = 0.5
dst_epsg              = "EPSG:25833" # grid for Norway area calculations   
#filename_in_catchment = f"{path_in_catchment}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"
#out_xy                = f"{path_in_catchment}weights_regine_012_drammensvassdraget_{resolution}.nc"
filename_in_catchment = f"{path_in_catchment}nve_nevina_bergheimvassdraget.geojson"
out_xy                = f"{path_in_catchment}weights_nevina_bergheimvassdraget_{resolution}.nc"
write2file            = True
# -------------------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def make_model_grid(resolution: str):
    """
    Hard-coded grid from ECMWF subseasonal forecast/hindcast (cell centers).
    Returns:
      model_latitude  (descending)
      model_longitude (ascending)
    """
    if resolution == "0.5x0.5":
        model_latitude = np.arange(73.5, 32.5, -0.5)
        model_longitude = np.arange(-27.0, 45.5, 0.5)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    return model_latitude, model_longitude


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


def centers_to_edges(centers: np.ndarray, d: float) -> np.ndarray:
    """Convert 1D grid centers to edges, assuming uniform spacing d."""
    return np.concatenate(([centers[0] - d / 2], centers + d / 2))


def build_projector(src_epsg: str, dst_epsg: str):
    """Create a Shapely-compatible transform function for reprojection."""
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

    Steps:
    - Project catchment to dst_epsg (metric)
    - For each model cell: build lon/lat box -> project -> intersect -> area ratio

    Assumptions:
    - model_latitude/model_longitude are cell centers (regular grid).
    - catchment_ll is in EPSG:4326.
    """
    # Project catchment once
    proj_fn = build_projector("EPSG:4326", dst_epsg)
    catchment = transform(proj_fn, catchment_ll)
    catchment_prep = prep(catchment)

    # Cell edges
    lon_edges = centers_to_edges(model_longitude, dlon)
    lat_edges_desc = centers_to_edges(model_latitude, dlat)  # descending lat centers

    weights = np.zeros((len(model_latitude), len(model_longitude)), dtype=np.float32)

    for i in range(len(model_latitude)):
        lat_a = lat_edges_desc[i]
        lat_b = lat_edges_desc[i + 1]
        lat0, lat1 = (lat_b, lat_a) if lat_a > lat_b else (lat_a, lat_b)

        for j in range(len(model_longitude)):
            lon0, lon1 = lon_edges[j], lon_edges[j + 1]

            # Cell polygon in lon/lat degrees
            cell_ll = box(lon0, lat0, lon1, lat1)

            # Project cell to metric CRS
            cell = transform(proj_fn, cell_ll)

            # Quick reject (fast): if no intersection, keep weight=0
            if not catchment_prep.intersects(cell):
                continue

            inter = catchment.intersection(cell)
            if not inter.is_empty and cell.area > 0:
                weights[i, j] = inter.area / cell.area

    return weights


def weights_to_xarray(weights, model_latitude, model_longitude, dst_epsg):
    """Wrap weights into an xarray DataArray with coords."""
    da = xr.DataArray(
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
    return da


if __name__ == "__main__":

    # Read catchment polygon GeoJSON (Polygon/MultiPolygon)
    gj           = read_geojson(filename_in_catchment)
    catchment_ll = dissolve_catchment_geometry(gj)

    # Build model grid
    model_latitude, model_longitude = make_model_grid(resolution)

    # Compute area-fraction weights (recommended method)
    weights = compute_area_fraction_weights_on_model_grid(
        catchment_ll=catchment_ll,
        model_latitude=model_latitude,
        model_longitude=model_longitude,
        dlat=dlat,
        dlon=dlon,
        dst_epsg=dst_epsg)

    # Wrap in xarray
    da_weights = weights_to_xarray(weights, model_latitude, model_longitude, dst_epsg)

    # Sanity checks
    print(da_weights)
    print("Min/Max weight:", float(np.nanmin(weights)), float(np.nanmax(weights)))
    print("Non-zero cells:", int(np.sum(weights > 0)))

    # Optionally write to NetCDF
    if write2file:
        da_weights.to_netcdf(out_xy)
        print("Wrote:", out_xy)
