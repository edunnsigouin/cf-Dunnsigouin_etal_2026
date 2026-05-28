"""
Fig 1 for Hans paper.

Plot Storm Hans precipitation and return period side by side.

The script:
1. Reads precipitation and return period from seNorge NetCDF
2. Loads catchment geometry
3. Creates two map panels
4. Plots precipitation and return period
5. Overlays catchment borders and station markers on both panels
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

# --- Paths
path_in_catchment = config.dirs["nve"]
path_in_senorge = config.dirs["senorge_processed"]
path_out = config.dirs["fig"]

# --- Files
filename_in_senorge = (
    f"{path_in_senorge}returnperiod_rr_2dayacc_senorge_1957-2023_20230809.nc"
)
filename_out = f"{path_out}temp.png"
write2file = True

# --- Publication output settings
# 180 mm x 90 mm is a typical two-column scientific figure size.
MM_TO_INCH = 1 / 25.4
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 90
FIGSIZE = (FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH)
FIG_DPI = 300

# --- Font sizes
tick_labelsize = 8
axis_labelsize = 8
title_fontsize = 9
station_labelsize = 8

# --- Map projection and extent
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

# --- Panel a: precipitation
PRECIP_LEVELS = np.arange(0, 160, 20)
PRECIP_CMAP = "GnBu"

# --- Panel b: return period
CATEGORY_EDGES = np.array([1.0, 5.0, 10.0, 50.0, 100.0, np.inf], dtype=float)
CATEGORY_LABELS = ["1–5", "5–10", "10–50", "50–100", "> 100"]
RETURN_CMAP = "PuBuGn"

# --- Catchment CRS if not present in source file
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

# --- Catchments to plot
CATCHMENTS = [
    {
        "label": "Drammensvassdraget",
        "geojson": "catchment_nve_regine_drammen.geojson",
        "color": "tab:red",
        "inset_m": 0,
    },
]

# --- Stations to plot
STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
    {"name": "Ål III", "lon": 8.5609, "lat": 60.6391},
]


# =============================================================================
# Data loading
# =============================================================================
def load_event_and_return_period_data(filename):
    ds = xr.open_dataset(filename)
    precip = ds["event_accum"]
    return_period = ds["return_period_years"]
    lon = ds["lon"]
    lat = ds["lat"]
    return ds, precip, return_period, lon, lat


# =============================================================================
# Catchment geometry helpers
# =============================================================================
def load_catchment_outer_boundaries(catchments, base_dir, crs_if_missing="EPSG:4326"):
    boundaries = []

    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    for catchment in catchments:
        filepath = base_dir + catchment["geojson"]
        gdf = gpd.read_file(filepath)

        if gdf.crs is None:
            gdf = gdf.set_crs(crs_if_missing)

        gdf_metric = gdf.to_crs(metric_crs)
        union_geom = gdf_metric.geometry.union_all()

        inset_m = catchment.get("inset_m", 0)
        if inset_m > 0:
            union_geom = union_geom.buffer(-inset_m)

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

        boundaries.append(
            {
                "label": catchment["label"],
                "color": catchment["color"],
                "geometry": outer_gdf.geometry.iloc[0],
            }
        )

    return boundaries


# =============================================================================
# Plot setup helpers
# =============================================================================
def make_two_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        dpi=FIG_DPI,
        subplot_kw={"projection": proj_map},
        constrained_layout=True,
    )

    for ax in axes:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.35)
        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


def format_colorbar(cbar, label, tick_labels=None):
    cbar.set_label(label, fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize, length=2)

    if tick_labels is not None:
        cbar.ax.set_xticklabels(tick_labels, fontsize=tick_labelsize)


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_precipitation(ax, da, lon, lat, proj_data):
    cf = ax.contourf(
        lon.values,
        lat.values,
        da.values,
        levels=PRECIP_LEVELS,
        cmap=PRECIP_CMAP,
        transform=proj_data,
        extend="max",
    )

    cbar = plt.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        shrink=0.88,
        pad=0.035,
        aspect=30,
    )

    format_colorbar(
        cbar,
        label="2-day accumulated precipitation (mm)",
        tick_labels=np.arange(0, 160, 20),
    )

    return cf


def categorize_return_period(T, edges):
    values = T.values
    categories = np.digitize(values, edges, right=False) - 1
    categories = categories.astype(float)

    categories[values < 1.0] = np.nan
    categories[~np.isfinite(values)] = np.nan

    return categories


def build_return_period_cmap(n_labels):
    base_cmap = plt.get_cmap(RETURN_CMAP)
    colors = base_cmap(np.linspace(0.3, 0.95, n_labels))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
    return mcolors.ListedColormap(colors)


def plot_return_period(ax, da, lon, lat, proj_data):
    cat = categorize_return_period(da, CATEGORY_EDGES)
    cmap = build_return_period_cmap(len(CATEGORY_LABELS))
    cmap.set_bad("white")

    mesh = ax.pcolormesh(
        lon.values,
        lat.values,
        cat,
        transform=proj_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(CATEGORY_LABELS) - 0.5,
        shading="auto",
    )

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        shrink=0.88,
        pad=0.035,
        aspect=30,
        ticks=np.arange(len(CATEGORY_LABELS)),
    )

    format_colorbar(
        cbar,
        label="Return period (years)",
        tick_labels=CATEGORY_LABELS,
    )

    return mesh


def plot_catchment_boundaries(ax, catchment_boundaries, proj_data):
    for item in catchment_boundaries:
        ax.add_geometries(
            [item["geometry"]],
            crs=proj_data,
            facecolor="none",
            edgecolor=item["color"],
            linewidth=1.4,
            zorder=5,
        )


def plot_station_markers(ax, stations, proj_data, fontsize=8):
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.6,
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


def finalize_figure(fig, axes, savepath=None, write2file=False):
    axes[0].set_title(
        "(a) Storm Hans precipitation 2023-08-07 to 2023-08-09",
        fontsize=title_fontsize,
    )
    axes[1].set_title(
        "(b) Storm Hans return period",
        fontsize=title_fontsize,
    )

    if write2file:
        fig.savefig(
            savepath,
            dpi=FIG_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    ds_senorge, precip, return_period, lon, lat = load_event_and_return_period_data(
        filename_in_senorge
    )

    catchment_boundaries = load_catchment_outer_boundaries(
        CATCHMENTS,
        base_dir=path_in_catchment,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, axes, proj_data = make_two_map_axes(
        central_lon=CENTRAL_LON,
        central_lat=CENTRAL_LAT,
        extent=MAP_EXTENT,
    )

    plot_precipitation(axes[0], precip, lon, lat, proj_data)
    plot_catchment_boundaries(axes[0], catchment_boundaries, proj_data)
    plot_station_markers(
        axes[0],
        STATIONS,
        proj_data,
        fontsize=station_labelsize,
    )

    plot_return_period(axes[1], return_period, lon, lat, proj_data)
    plot_catchment_boundaries(axes[1], catchment_boundaries, proj_data)
    plot_station_markers(
        axes[1],
        STATIONS,
        proj_data,
        fontsize=station_labelsize,
    )

    finalize_figure(
        fig,
        axes,
        savepath=filename_out,
        write2file=write2file,
    )

    ds_senorge.close()
