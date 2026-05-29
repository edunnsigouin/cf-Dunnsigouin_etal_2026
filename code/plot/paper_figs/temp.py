"""
Fig 1 for Hans paper.

Plot catchment weights for five catchments.

Layout:
- 3 panels in the top row
- 2 panels in the bottom row
- shared colorbar in the empty 6th panel
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
write2file = False

tick_labelsize = 12
axis_labelsize = 12
title_fontsize = 12
station_labelsize = 11

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

WEIGHT_CMAP = "BuGn"
WEIGHT_VMIN = 0.0
WEIGHT_VMAX = 1.0

CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

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
        "label": "Drammensvassdraget",
        "weights": "weights_catchment_regine_drammen_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_regine_drammen.geojson",
    },
    {
        "label": "Glomma",
        "weights": "weights_catchment_regine_glomma_era5_0.5x0.5.nc",
        "geojson": "catchment_nve_regine_glomma.geojson",
    },
]

STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
    {"name": "Ål III", "lon": 8.5609, "lat": 60.6391},
]


# =============================================================================
# Data loading
# =============================================================================
def load_weights_data(filename):
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
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
        subplot_kw={"projection": proj_map},
    )

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)
        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


def format_colorbar(cbar, label, tick_labelsize=12, axis_labelsize=12):
    cbar.set_label(label, fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)


# =============================================================================
# Plotting helpers
# =============================================================================
def centers_to_edges(centers):
    centers = np.asarray(centers)

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def plot_weights(ax, da_weights, proj_data):
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
    ax.add_geometries(
        [boundary["geometry"]],
        crs=proj_data,
        facecolor="none",
        edgecolor=boundary["color"],
        linewidth=2.5,
        zorder=5,
    )


def plot_station_markers(ax, stations, proj_data, fontsize=11):
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=8,
            markeredgecolor="black",
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
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, panel_label, catchment in zip(plot_axes, panel_labels, CATCHMENTS):
        ax.set_title(
            f"{panel_label} {catchment['label']}",
            fontsize=title_fontsize,
        )

    # Use lower-right panel as colorbar area
    cax = axes[1, 2]
    cax.set_axis_off()

    cbar = fig.colorbar(
        mesh,
        ax=cax,
        orientation="horizontal",
        fraction=0.7,
        pad=0.35,
    )
    format_colorbar(
        cbar,
        label="Catchment weight (fraction)",
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )

    fig.subplots_adjust(wspace=0.02, hspace=0.08)

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

    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

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
