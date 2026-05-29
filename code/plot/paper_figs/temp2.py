"""
Fig S1 for Hans paper.

Plot catchment weights for five catchments.

Layout:
- 3 panels in the top row
- 2 panels in the bottom row
- shared colorbar in the empty 6th panel
- publication-style golden-ratio figure
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

path_in_catchment = config.dirs["nve"]
path_out = config.dirs["fig"]

filename_out = f"{path_out}fig-S1.png"
write2file = True

# --- Publication layout
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
FIG_WIDTH_IN = 16
FIG_HEIGHT_IN = FIG_WIDTH_IN / GOLDEN_RATIO

MAP_WSPACE = -0.5
MAP_HSPACE = 0.1

# --- Font sizes
tick_labelsize = 11
axis_labelsize = 12
title_fontsize = 12
station_labelsize = 8

# --- Map projection and extent
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

# --- Catchment weights
WEIGHT_CMAP = "BuGn"
WEIGHT_VMIN = 0.0
WEIGHT_VMAX = 1.0

# --- Catchment CRS if not present in source file
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

# --- Catchments to plot
CATCHMENTS = [
    {
        "label": "Bergheim",
        "weights": "weights_catchment_nevina_bergheim_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_nevina_bergheim.geojson",
    },
    {
        "label": "Hønefoss",
        "weights": "weights_catchment_nevina_hønnefoss_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_nevina_hønnefoss.geojson",
    },
    {
        "label": "Losna",
        "weights": "weights_catchment_nevina_losna_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_nevina_losna.geojson",
    },
    {
        "label": "Drammen",
        "weights": "weights_catchment_regine_drammen_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_regine_drammen.geojson",
    },
    {
        "label": "Glomma",
        "weights": "weights_catchment_regine_glomma_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_regine_glomma.geojson",
    },
]

# --- Stations to plot, optional
STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
    {"name": "Ål III", "lon": 8.5609, "lat": 60.6391},
]


# =============================================================================
# Data loading
# =============================================================================
def load_weights_data(filename):
    """
    Load catchment weights from NetCDF.
    """
    ds = xr.open_dataset(filename)
    da_weights = ds["catchment_weight"]
    return ds, da_weights


# =============================================================================
# Catchment geometry helpers
# =============================================================================
def load_single_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
    color="red",
):
    """
    Load one catchment polygon, dissolve it to one geometry,
    and keep only the outer boundary.
    """
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    gdf = gpd.read_file(base_dir + filename)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)

    gdf_metric = gdf.to_crs(metric_crs)
    union_geom = gdf_metric.geometry.union_all()

    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)
    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon(
            [Polygon(poly.exterior) for poly in union_geom.geoms]
        )
    else:
        outer_geom = union_geom

    outer_gdf = (
        gpd.GeoDataFrame(geometry=[outer_geom], crs=metric_crs)
        .to_crs(plot_crs)
    )

    return {
        "color": color,
        "geometry": outer_gdf.geometry.iloc[0],
    }


# =============================================================================
# Plot setup helpers
# =============================================================================
def make_weight_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
    """
    Create a 2 x 3 publication-style Lambert Conformal map layout.
    """
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        subplot_kw={"projection": proj_map},
        constrained_layout=False,
    )

    for i, ax in enumerate(axes.flat):

        if i == 5:
            ax.set_axis_off()
            continue
        
        ax.coastlines(resolution="10m", linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3)

        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


def format_colorbar(cbar, label, tick_labelsize=8, axis_labelsize=9):
    """
    Apply consistent publication-style colorbar formatting.
    """
    cbar.set_label(label, fontsize=axis_labelsize, labelpad=2)
    cbar.ax.tick_params(labelsize=tick_labelsize, length=2, pad=1)


# =============================================================================
# Plotting helpers
# =============================================================================
def centers_to_edges(centers):
    """
    Convert 1D grid-cell centers to edges.
    """
    centers = np.asarray(centers)

    if centers.ndim != 1:
        raise ValueError("centers must be 1D")
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer edges")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def plot_weights(ax, da_weights, proj_data):
    """
    Plot catchment weights as a pcolormesh.

    Zero-valued weights are masked and shown in white.
    """
    lats = da_weights.latitude.values
    lons = da_weights.longitude.values
    weights = da_weights.values.copy()

    lat_edges = centers_to_edges(lats)
    lon_edges = centers_to_edges(lons)

    weights = np.where(weights == 0, np.nan, weights)

    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        weights = weights[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        weights = weights[:, ::-1]

    lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)

    cmap = plt.get_cmap(WEIGHT_CMAP).copy()
    cmap.set_bad("white")

    mesh = ax.pcolormesh(
        lon_e,
        lat_e,
        weights,
        cmap=cmap,
        vmin=WEIGHT_VMIN,
        vmax=WEIGHT_VMAX,
        shading="auto",
        transform=proj_data,
    )

    return mesh


def plot_catchment_boundary(ax, boundary, proj_data):
    """
    Plot the outer catchment border.
    """
    ax.add_geometries(
        [boundary["geometry"]],
        crs=proj_data,
        facecolor="none",
        edgecolor=boundary["color"],
        linewidth=1.4,
        zorder=5,
    )


def plot_station_markers(ax, stations, proj_data, fontsize=8):
    """
    Plot station markers as yellow dots with labels.
    Currently optional and not called in the main loop.
    """
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markerfacecolor="yellow",
            transform=proj_data,
            zorder=6,
        )

        if station["name"] == "Bergheim":
            dx, dy = 0.05, 0.05
        elif station["name"] == "Ål III":
            dx, dy = -0.65, 0.05
        else:
            dx, dy = 0.05, 0.05

        ax.text(
            station["lon"] + dx,
            station["lat"] + dy,
            station["name"],
            fontsize=fontsize,
            color="yellow",
            transform=proj_data,
            zorder=7,
        )


def finalize_figure(fig, axes, mesh, savepath=None, write2file=False):
    """
    Add titles, use the sixth panel as a shared colorbar slot,
    tighten layout, optionally save, and show figure.
    """
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, panel_label, catchment in zip(plot_axes, panel_labels, CATCHMENTS):
        ax.set_title(
            f"{panel_label} {catchment['label']}",
            fontsize=title_fontsize,
            pad=2,
        )

    # Use lower-right panel as the colorbar container
    cax_container = axes[1, 2]
    cax_container.set_axis_off()

    # Place colorbar inside the empty sixth grid cell
    bbox = cax_container.get_position()

    cbar_height = 0.025
    cbar_width = bbox.width * 0.85
    cbar_x = bbox.x0 + 0.5 * (bbox.width - cbar_width)
    cbar_y = bbox.y0 + 0.48 * bbox.height

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )
    format_colorbar(
        cbar,
        label="Catchment weight (fraction)",
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    fig, axes, proj_data = make_weight_map_axes(
        central_lon=CENTRAL_LON,
        central_lat=CENTRAL_LAT,
        extent=MAP_EXTENT,
    )

    plot_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    ]

    datasets = []
    mesh = None

    for ax, catchment in zip(plot_axes, CATCHMENTS):

        ds_weights, da_weights = load_weights_data(
            path_in_catchment + catchment["weights"]
        )
        datasets.append(ds_weights)

        boundary = load_single_catchment_outer_boundary(
            catchment["geojson"],
            base_dir=path_in_catchment,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
            color="red",
        )

        mesh = plot_weights(ax, da_weights, proj_data)
        plot_catchment_boundary(ax, boundary, proj_data)

    finalize_figure(
        fig,
        axes,
        mesh,
        savepath=filename_out,
        write2file=write2file,
    )

    for ds in datasets:
        ds.close()
