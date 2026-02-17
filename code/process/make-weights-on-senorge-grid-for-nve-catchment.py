"""
Make an XY weights file [0,1] mapping the SeNorge grid (projected X/Y) to an NVE catchment.

Method: area fraction using Shapely + pyproj
- Treat each SeNorge grid cell (X/Y) as a polygon in projected meters.
- Project catchment polygon to the same metric CRS as the grid.
- Weight = area(catchment ∩ cell) / area(cell)

Notes:
- SeNorge files are large; we only open ONE example file to read X/Y grid.
- This version computes weights over the ENTIRE grid (all Y, X).
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
path_in_catchment = config.dirs["nve_catchment"]
filename_in_catchment = f"{path_in_catchment}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"

path_in_senorge = config.dirs["senorge_continuous_daily"]  # adjust if needed
example_year = 2006
example_file = f"{path_in_senorge}rr/rr_{example_year}.nc"

# SeNorge grid CRS (typical): ETRS89 / UTM 33N
grid_epsg = "EPSG:25833"

out_xy = f"{path_in_catchment}weights_regine_012_drammensvassdraget_senorge.nc"
write2file = True
# ------------------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
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
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transformer.transform


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert 1D grid centers to edges using median spacing (assumes regular spacing)."""
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least 2 centers to infer spacing.")
    d = float(np.median(np.diff(centers)))
    return np.concatenate(([centers[0] - d / 2.0], centers + d / 2.0))


def load_senorge_grid(example_file: str):
    """Read X and Y from one SeNorge file."""
    ds = xr.open_dataset(example_file)
    if "X" not in ds or "Y" not in ds:
        raise KeyError(f"Expected coords 'X' and 'Y' in {example_file}. Found: {list(ds.variables)}")

    X = ds["X"].values.astype(float)
    Y = ds["Y"].values.astype(float)
    ds.close()

    # Sort X to increasing; keep Y order but edges logic handles both
    if np.any(np.diff(X) <= 0):
        X = np.sort(X)

    return X, Y


def compute_area_fraction_weights_on_xy_grid_full(
    catchment_ll,
    X: np.ndarray,
    Y: np.ndarray,
    grid_epsg: str,
):
    """
    Compute weights on the full SeNorge (Y, X) grid as area fractions.

    catchment_ll: shapely (EPSG:4326)
    X, Y: 1D arrays of cell centers (meters) in grid_epsg
    """
    proj_fn = build_projector("EPSG:4326", grid_epsg)
    catchment = transform(proj_fn, catchment_ll)
    catchment_prep = prep(catchment)

    x_edges = centers_to_edges(X)
    y_edges = centers_to_edges(Y)

    weights = np.zeros((len(Y), len(X)), dtype=np.float32)

    for i in range(len(Y)):
        y0, y1 = y_edges[i], y_edges[i + 1]
        y_lo, y_hi = (y0, y1) if y0 < y1 else (y1, y0)

        for j in range(len(X)):
            x0, x1 = x_edges[j], x_edges[j + 1]
            x_lo, x_hi = (x0, x1) if x0 < x1 else (x1, x0)

            cell = box(x_lo, y_lo, x_hi, y_hi)

            if not catchment_prep.intersects(cell):
                continue

            inter = catchment.intersection(cell)
            if not inter.is_empty and cell.area > 0:
                weights[i, j] = inter.area / cell.area

    return weights


def weights_to_xarray(weights: np.ndarray, X: np.ndarray, Y: np.ndarray, grid_epsg: str) -> xr.DataArray:
    da = xr.DataArray(
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
    return da


if __name__ == "__main__":

    gj = read_geojson(filename_in_catchment)
    catchment_ll = dissolve_catchment_geometry(gj)

    X, Y = load_senorge_grid(example_file)

    weights = compute_area_fraction_weights_on_xy_grid_full(
        catchment_ll=catchment_ll,
        X=X,
        Y=Y,
        grid_epsg=grid_epsg,
    )

    da_weights = weights_to_xarray(weights, X, Y, grid_epsg)

    print(da_weights)
    print("Min/Max weight:", float(np.nanmin(weights)), float(np.nanmax(weights)))
    print("Non-zero cells:", int(np.sum(weights > 0)))

    if write2file:
        da_weights.to_netcdf(out_xy)
        print("Wrote:", out_xy)
