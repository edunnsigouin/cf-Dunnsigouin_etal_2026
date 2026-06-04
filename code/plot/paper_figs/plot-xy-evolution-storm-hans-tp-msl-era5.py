#!/usr/bin/env python3
"""
Plot ERA5 reanalysis evolution of precipitation and mean sea level pressure.

The script:
1. Uses fixed event-relative dates:
       -2 = 2023-08-06
       -1 = 2023-08-07
        0 = 2023-08-08
       +1 = 2023-08-09
       +2 = 2023-08-10
2. Plots ERA5 daily precipitation as shading.
3. Plots ERA5 mean sea level pressure as grey labelled contours.
4. Converts precipitation from m to mm.
5. Converts MSLP from Pa to hPa.
6. Overlays the Drammen catchment boundary in red.
7. Uses the same figure format as the S2S event-evolution plots.
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

path_out = config.dirs["fig"]
path_in_catchment = config.dirs["nve"]

YEAR = 2023

EVENT_LAGS = [-2, -1, 0, 1, 2]
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
    "2023-08-10",
]

ACCUMULATION_DAYS = 1  # choose 1 or 2

PRECIP_VAR = "tp24"
MSL_VAR = "msl"

PRECIP_FILE = (
    Path(config.dirs["era5_continuous_daily"])
    / PRECIP_VAR
    / f"{PRECIP_VAR}_0.5x0.5_{YEAR}.nc"
)

MSL_FILE = (
    Path(config.dirs["era5_continuous_daily"])
    / MSL_VAR
    / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"
)

CATCHMENT_FILE = "catchment_nve_regine_drammen.geojson"
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"
CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0

filename_out = (
    f"{path_out}xy-hans-evolution-era5-tp-msl-"
    f"{EVENT_DATES[0]}-{EVENT_DATES[-1]}.png"
)
write2file = True

FIG_WIDTH_IN = 16
FIG_HEIGHT_IN = 12

MAP_WSPACE = 0.0
MAP_HSPACE = 0.08

tick_labelsize = 12
axis_labelsize = 13
title_fontsize = 14
contour_labelsize = 9

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [-10, 25, 50, 70]

# --- Precipitation shading
if ACCUMULATION_DAYS == 1:
    PRECIP_LEVELS = np.arange(5, 55, 5)
elif ACCUMULATION_DAYS == 2:
    PRECIP_LEVELS = np.arange(0, 121, 10)
else:
    raise ValueError("ACCUMULATION_DAYS must be either 1 or 2.")

PRECIP_CMAP = plt.get_cmap("GnBu").copy()
PRECIP_CMAP.set_under("white")
PRECIP_ZERO_THRESHOLD = 5.0

# --- MSLP contours
MSL_CONTOUR_LEVELS = np.arange(960, 1045, 5)
MSL_CONTOUR_COLOR = "0.7"
MSL_CONTOUR_LINEWIDTH = 2


# =============================================================================
# Coordinate helpers
# =============================================================================
def get_time_coord_name(da):
    """Identify time coordinate name."""
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]
    return lon, lat


def centers_to_edges(centers):
    """Convert 1D grid-cell centers to edges."""
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


# =============================================================================
# Data loading
# =============================================================================
def open_era5_file(filename, variable):
    """Open ERA5 file and convert units."""
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in {filename}")

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    if variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    return ds


def select_date(da, target_date):
    """Select one date from a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date})


def load_precipitation(ds_tp, target_date, accumulation_days=1):
    """Load 1-day or 2-day accumulated precipitation."""
    target_date = np.datetime64(target_date, "D")

    if accumulation_days == 1:
        da_tp = select_date(ds_tp[PRECIP_VAR], str(target_date))
    elif accumulation_days == 2:
        da_previous = select_date(
            ds_tp[PRECIP_VAR],
            str(target_date - np.timedelta64(1, "D")),
        )
        da_current = select_date(ds_tp[PRECIP_VAR], str(target_date))
        da_tp = da_previous + da_current
        da_tp.attrs["units"] = "mm"
    else:
        raise ValueError("accumulation_days must be either 1 or 2.")

    return da_tp.load()


def load_msl(ds_msl, target_date):
    """Load MSLP for one date."""
    return select_date(ds_msl[MSL_VAR], target_date).load()


# =============================================================================
# Catchment helpers
# =============================================================================
def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
    """Load catchment polygon, dissolve it, and keep only outer boundary."""
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

    return outer_gdf.geometry.iloc[0]


# =============================================================================
# Plot setup
# =============================================================================
def make_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
    """Create 2 x 3 Lambert Conformal map layout."""
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

        ax.coastlines(resolution="10m", linewidth=0.5)

        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_precipitation(ax, da_precip, proj_data):
    """Plot precipitation as shaded field."""
    lon, lat = get_lon_lat(da_precip)
    precip = da_precip.values

    lon_edges = centers_to_edges(lon.values)
    lat_edges = centers_to_edges(lat.values)

    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        precip = precip[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        precip = precip[:, ::-1]

    lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)

    mesh = ax.pcolormesh(
        lon_e,
        lat_e,
        precip,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_ZERO_THRESHOLD,
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )

    return mesh


def plot_msl_contours(ax, da_msl, proj_data):
    """Plot labelled mean sea level pressure contours."""
    lon, lat = get_lon_lat(da_msl)
    msl = da_msl.values

    contour = ax.contour(
        lon.values,
        lat.values,
        msl,
        levels=MSL_CONTOUR_LEVELS,
        colors=MSL_CONTOUR_COLOR,
        linewidths=MSL_CONTOUR_LINEWIDTH,
        transform=proj_data,
        zorder=6,
    )

    ax.clabel(
        contour,
        inline=True,
        inline_spacing=4,
        fontsize=contour_labelsize,
        fmt="%d",
        colors=MSL_CONTOUR_COLOR,
    )

    return contour


def plot_catchment_boundary(ax, geometry, proj_data):
    """Overlay catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )


def finalize_figure(
    fig,
    axes,
    mesh,
    event_lags,
    event_dates,
    savepath=None,
    write2file=False,
):
    """Add titles, colorbar, save, and show."""
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    panel_labels = ["a)", "b)", "c)", "d)", "e)"]

    for ax, label, lag, date in zip(
            plot_axes,
            panel_labels,
            event_lags,
            event_dates,
    ):
        
        ax.set_title(
            f"{label} Day {lag:+d}: {date}",
            fontsize=title_fontsize,
            pad=3,
        )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    cax = fig.add_axes([0.675, 0.38, 0.255, 0.025])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_label(
        f"accumulated total precipitation (mm/day)",
        fontsize=axis_labelsize,
    )
    cbar.ax.tick_params(labelsize=tick_labelsize)

    legend_ax = axes[1, 2]
    legend_ax.set_axis_off()

    legend_handles = [
        Line2D(
            [0], [0],
            color=CATCHMENT_EDGE_COLOR,
            linewidth=2,
            label="Drammen catchment",
        ),
        Line2D(
            [0], [0],
            color=MSL_CONTOUR_COLOR,
            linewidth=MSL_CONTOUR_LINEWIDTH,
            label="Mean sea level pressure contours (hPa)",
        ),
    ]

    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=axis_labelsize,
    )


    
    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    ds_tp = open_era5_file(PRECIP_FILE, PRECIP_VAR)
    ds_msl = open_era5_file(MSL_FILE, MSL_VAR)

    catchment_boundary = load_catchment_outer_boundary(
        CATCHMENT_FILE,
        base_dir=path_in_catchment,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, axes, proj_data = make_map_axes(
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

    mesh = None

    for ax, target_date in zip(plot_axes, EVENT_DATES):

        da_precip = load_precipitation(
            ds_tp,
            target_date=target_date,
            accumulation_days=ACCUMULATION_DAYS,
        )

        da_msl = load_msl(
            ds_msl,
            target_date=target_date,
        )

        mesh = plot_precipitation(
            ax,
            da_precip,
            proj_data,
        )

        plot_msl_contours(
            ax,
            da_msl,
            proj_data,
        )

        plot_catchment_boundary(
            ax,
            catchment_boundary,
            proj_data,
        )

    finalize_figure(
        fig,
        axes,
        mesh,
        EVENT_LAGS,
        EVENT_DATES,
        savepath=filename_out,
        write2file=write2file,
    )

    ds_tp.close()
    ds_msl.close()
