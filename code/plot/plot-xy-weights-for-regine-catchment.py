"""
Plot model-grid catchment weights (0–1) and overlay the catchment border
and model grid cell boundaries (0.5x0.5) on a Cartopy map.
"""

# =============================================================================
# 1) imports
# =============================================================================
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import shape
from shapely.ops import unary_union

from Dunnsigouin_etal_2026 import config


# =============================================================================
# 2) user-defined input parameters
# =============================================================================
path_in = config.dirs["nve_catchment"]
path_out = config.dirs["fig"]

weights_nc = f"{path_in}weights_regine_012_drammensvassdraget_0.5x0.5.nc"
catchment_geojson = f"{path_in}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"

map_extent = [4.5, 14.0, 57.5, 64.0]

figure_size = (6, 6)
outline_width = 2.0
gridline_width = 0.4
gridline_color = "black"

write2file = True
fig_out = f"{path_out}weights_regine_012_drammensvassdraget_0.5x0.5.pdf"


# =============================================================================
# 3) functions
# =============================================================================
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


def centers_to_edges(centers: np.ndarray, d: float) -> np.ndarray:
    """Convert grid centers to grid edges."""
    return np.concatenate(([centers[0] - d / 2], centers + d / 2))


def infer_grid_spacing(centers: np.ndarray) -> float:
    """Infer uniform grid spacing."""
    return float(np.median(np.abs(np.diff(centers))))


def setup_cartopy_ax(fig_size, extent):
    """Create Cartopy axis with background layers (no gridlines)."""
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
    #ax.add_feature(cfeature.LAKES, alpha=0.5)
    #ax.add_feature(cfeature.RIVERS, linewidth=0.3)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def plot_weights(ax, da_weights: xr.DataArray):
    """
    Plot weights as a pcolormesh using BuGn colormap.
    """
    lats = da_weights.latitude.values
    lons = da_weights.longitude.values
    w = da_weights.values

    dlat = infer_grid_spacing(lats)
    dlon = infer_grid_spacing(lons)

    lat_edges_desc = centers_to_edges(lats, dlat)
    lon_edges = centers_to_edges(lons, dlon)

    # pcolormesh needs increasing latitude
    if lat_edges_desc[0] > lat_edges_desc[-1]:
        lat_edges = lat_edges_desc[::-1]
        w_plot = w[::-1, :]
    else:
        lat_edges = lat_edges_desc
        w_plot = w

    LON_E, LAT_E = np.meshgrid(lon_edges, lat_edges)

    m = ax.pcolormesh(
        LON_E,
        LAT_E,
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


def plot_model_grid(ax, lat_edges, lon_edges):
    """
    Plot model grid boundaries (0.5x0.5) as thin lines.
    """
    for lon in lon_edges:
        ax.plot(
            [lon, lon],
            [lat_edges[0], lat_edges[-1]],
            linewidth=gridline_width,
            color=gridline_color,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )

    for lat in lat_edges:
        ax.plot(
            [lon_edges[0], lon_edges[-1]],
            [lat, lat],
            linewidth=gridline_width,
            color=gridline_color,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
        )


def plot_catchment_border(ax, catchment_geom, linewidth=2.0):
    """Overlay catchment outline."""
    if catchment_geom.geom_type == "Polygon":
        x, y = catchment_geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, color = 'tab:red',transform=ccrs.PlateCarree())

    elif catchment_geom.geom_type == "MultiPolygon":
        for poly in catchment_geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=linewidth, color = 'tab:red',transform=ccrs.PlateCarree())


# =============================================================================
# 4) main script
# =============================================================================
if __name__ == "__main__":

    # Load weights
    ds = xr.open_dataset(weights_nc)
    da = ds["catchment_weight"]

    # Load catchment polygon
    gj = read_geojson(catchment_geojson)
    catchment = dissolve_polygon_geojson(gj)

    # Plot
    fig, ax = setup_cartopy_ax(figure_size, map_extent)
    lat_edges, lon_edges = plot_weights(ax, da)
    #plot_model_grid(ax, lat_edges, lon_edges)
    plot_catchment_border(ax, catchment, linewidth=outline_width)

    ax.set_title("Catchment weights on 0.5° model grid", fontsize=12)

    if write2file:
        fig.savefig(fig_out, bbox_inches="tight")
        print("Saved:", fig_out)

    plt.show()
