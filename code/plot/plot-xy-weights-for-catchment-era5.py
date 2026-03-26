"""
Plot model-grid catchment weights (0–1) and overlay the catchment border
on a Cartopy map.
"""

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape
from shapely.ops import unary_union
from Dunnsigouin_etal_2026 import config

# input -------------------------------------------------------------------------
path_in = config.dirs["nve_catchment"]
path_out = config.dirs["fig"]
catchment_name = "regine_drammen"
weights_nc = f"{path_in}weights_catchment_{catchment_name}_era5_0.5x0.5.nc"
catchment_geojson = f"{path_in}catchment_nve_{catchment_name}.geojson"
fig_out = f"{path_out}weights_catchement_{catchment_name}_era5_0.5x0.5.pdf"
map_extent = [4.5, 14.0, 57.5, 64.0]
figure_size = (6, 6)
outline_width = 2.0
write2file = False
# -------------------------------------------------------------------------------


def read_geojson(filepath: str) -> dict:
    """Read a GeoJSON file into a Python dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dissolve_polygon_geojson(gj: dict):
    """Dissolve Polygon/MultiPolygon features into one geometry."""
    geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
    geom = unary_union(geoms)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {geom.geom_type}")
    return geom


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """
    Convert 1D grid-cell centers to cell edges.

    Works for both increasing and decreasing coordinates.
    """
    centers = np.asarray(centers)

    if centers.ndim != 1:
        raise ValueError("centers must be 1D")
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer edges")

    edges = np.empty(centers.size + 1, dtype=float)

    # interior edges = midpoints
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])

    # extrapolate outer edges from nearest spacing
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def setup_cartopy_ax(fig_size, extent):
    """Create Cartopy axis with background layers."""
    proj = ccrs.LambertConformal(
        central_longitude=15,
        central_latitude=65,
        standard_parallels=(63, 70),
    )

    fig = plt.figure(figsize=fig_size)
    ax = plt.axes(projection=proj)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def plot_weights(ax, da_weights: xr.DataArray):
    """Plot weights as a pcolormesh."""
    lats = da_weights.latitude.values
    lons = da_weights.longitude.values
    w = da_weights.values

    lat_edges = centers_to_edges(lats)
    lon_edges = centers_to_edges(lons)

    w_plot = w.copy()

    # pcolormesh expects increasing coordinates
    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        w_plot = w_plot[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        w_plot = w_plot[:, ::-1]

    lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)

    m = ax.pcolormesh(
        lon_e,
        lat_e,
        w_plot,
        cmap="BuGn",
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    cb = plt.colorbar(m, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("Catchment weight (fraction)")

    return lat_edges, lon_edges


def plot_catchment_border(ax, catchment_geom, linewidth=2.0):
    """Overlay catchment outline."""
    if catchment_geom.geom_type == "Polygon":
        x, y = catchment_geom.exterior.xy
        ax.plot(
            x,
            y,
            linewidth=linewidth,
            color="tab:red",
            transform=ccrs.PlateCarree(),
        )

    elif catchment_geom.geom_type == "MultiPolygon":
        for poly in catchment_geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                linewidth=linewidth,
                color="tab:red",
                transform=ccrs.PlateCarree(),
            )


if __name__ == "__main__":

    # Load weights
    ds = xr.open_dataset(weights_nc)
    da = ds["catchment_weight"]

    # Load catchment polygon
    gj = read_geojson(catchment_geojson)
    catchment_geom = dissolve_polygon_geojson(gj)

    # Plot
    fig, ax = setup_cartopy_ax(figure_size, map_extent)
    plot_weights(ax, da)
    plot_catchment_border(ax, catchment_geom, linewidth=outline_width)
    ax.set_title("Catchment weights on 0.5° model grid", fontsize=12)

    if write2file:
        fig.savefig(fig_out, bbox_inches="tight")
        print("Saved:", fig_out)

    plt.show()
