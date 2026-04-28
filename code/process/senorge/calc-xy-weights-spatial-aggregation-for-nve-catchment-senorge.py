"""
Make an XY weights file [0,1] mapping the SeNorge grid (projected X/Y) to an NVE catchment.

Method: area fraction using Shapely + pyproj
- Treat each SeNorge grid cell (X/Y) as a polygon in projected meters.
- Project catchment polygon from lon/lat (EPSG:4326) to the same metric CRS as the grid.
- Weight = area(catchment ∩ cell) / area(cell)

Includes:
1) Read/dissolve the catchment GeoJSON to a single Polygon/MultiPolygon in lon/lat.
2) Load the SeNorge grid coordinates (X, Y) from an example file.
3) Compute area-fraction weights on the full (Y, X) grid.
4) Write the weights to NetCDF (xarray DataArray) for later use.
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
catchment             = "nevina_losna"
path_in_catchment     = config.dirs["nve_catchment"]
filename_in_catchment = f"{path_in_catchment}catchment_nve_{catchment}.geojson"

path_in_senorge       = config.dirs["senorge_continuous_daily"]  # adjust if needed
example_year          = 2006
example_file          = f"{path_in_senorge}rr/rr_{example_year}.nc"

grid_epsg             = "EPSG:25833" # SeNorge grid CRS (typical): ETRS89 / UTM 33N 
ll_epsg               = "EPSG:4326" # catchment geojson grid

out_xy                = f"{path_in_catchment}weights_catchment_{catchment}_senorge.nc"
write2file            = True
# ------------------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dissolve_catchment_geometry(gj: dict):
    """Dissolve all polygonal features into one Shapely geometry in lon/lat (EPSG:4326)."""
    geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
    if not geoms:
        raise ValueError("No geometries found in catchment GeoJSON.")
    out = unary_union(geoms)
    if out.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {out.geom_type}")
    return out


def build_projector(src_epsg: str, dst_epsg: str):
    """Return a Shapely-compatible coordinate transform function."""
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transformer.transform


def centers_to_edges_any_order(centers: np.ndarray) -> np.ndarray:
    """
    Convert 1D grid centers to edges using median spacing.
    Works for both increasing and decreasing center arrays.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least 2 centers to infer spacing.")
    d = float(np.median(np.abs(np.diff(centers))))

    if centers[1] > centers[0]:
        return np.concatenate(([centers[0] - d / 2.0], centers + d / 2.0))
    else:
        return np.concatenate((centers + d / 2.0, [centers[-1] - d / 2.0]))


def load_senorge_grid(example_file: str):
    """Read X and Y coordinate vectors from one SeNorge NetCDF file."""
    ds = xr.open_dataset(example_file)
    try:
        if "X" not in ds or "Y" not in ds:
            raise KeyError(
                f"Expected coords 'X' and 'Y' in {example_file}. Found: {list(ds.variables)}"
            )
        X = ds["X"].values.astype(float)
        Y = ds["Y"].values.astype(float)
    finally:
        ds.close()

    # Keep X increasing for consistency; Y can be either (handled by edges function)
    if np.any(np.diff(X) <= 0):
        X = np.sort(X)

    return X, Y


def compute_area_fraction_weights_on_xy_grid_full(
    catchment_ll,
    X: np.ndarray,
    Y: np.ndarray,
    ll_epsg: str,
    grid_epsg: str,
):
    """
    Compute weights on the full SeNorge (Y, X) grid as area fractions.
    catchment_ll is EPSG:4326; grid is grid_epsg (meters).
    """
    proj_fn = build_projector(ll_epsg, grid_epsg)

    # Project catchment to grid CRS for area computations
    catchment = transform(proj_fn, catchment_ll)
    if not catchment.is_valid:
        catchment = catchment.buffer(0)

    catchment_prep = prep(catchment)

    x_edges = centers_to_edges_any_order(X)
    y_edges = centers_to_edges_any_order(Y)

    weights = np.zeros((len(Y), len(X)), dtype=np.float32)

    for i in range(len(Y)):
        y0, y1 = y_edges[i], y_edges[i + 1]
        y_lo, y_hi = (y0, y1) if y0 < y1 else (y1, y0)

        for j in range(len(X)):
            x0, x1 = x_edges[j], x_edges[j + 1]
            x_lo, x_hi = (x0, x1) if x0 < x1 else (x1, x0)

            # Cell polygon is already in projected meters
            cell = box(x_lo, y_lo, x_hi, y_hi)

            if not catchment_prep.intersects(cell):
                continue

            inter = catchment.intersection(cell)
            if (not inter.is_empty) and (cell.area > 0):
                weights[i, j] = inter.area / cell.area

    return weights


def weights_to_xarray(weights: np.ndarray, X: np.ndarray, Y: np.ndarray, grid_epsg: str) -> xr.DataArray:
    """Wrap weights into an xarray DataArray with coordinates and metadata."""
    return xr.DataArray(
        weights,
        dims=("Y", "X"),
        coords={"Y": Y, "X": X},
        name="catchment_weight",
        attrs={
            "description": "Area fraction of SeNorge grid cell within catchment",
            "range": "[0,1]",
            "grid_crs": grid_epsg,
        },
    )


# ------------------------------------------- main -------------------------------------------

if __name__ == "__main__":

    # Read and dissolve catchment polygons (EPSG:4326 lon/lat)
    gj = read_geojson(filename_in_catchment)
    catchment_ll = dissolve_catchment_geometry(gj)

    # Load SeNorge grid coordinate vectors (projected meters)
    X, Y = load_senorge_grid(example_file)

    # Compute area-fraction weights on the full (Y, X) grid in the grid CRS
    weights = compute_area_fraction_weights_on_xy_grid_full(
        catchment_ll=catchment_ll,
        X=X,
        Y=Y,
        ll_epsg=ll_epsg,
        grid_epsg=grid_epsg,
    )

    # Package as DataArray and write to NetCDF
    da_weights = weights_to_xarray(weights, X, Y, grid_epsg)

    if write2file:
        da_weights.to_netcdf(out_xy)
        print("Wrote:", out_xy)
