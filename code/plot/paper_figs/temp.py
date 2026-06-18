#!/usr/bin/env python3
"""
Figure 1: ERA5 Storm Hans maps plus Bergheim streamflow time series.

The figure contains:
1. Four map panels for 2023-08-06 to 2023-08-09.
2. ERA5 daily precipitation as shading.
3. ERA5 mean sea level pressure as labelled grey contours.
4. Selected catchment boundary in red.
5. Station locations as yellow dots.
6. A bottom streamflow time-series panel for Bergheim station.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

YEAR = 2023

EVENT_LAGS = [-2, -1, 0, 1]
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"

PRECIP_VAR = "tp24"
MSL_VAR = "msl"
STREAMFLOW_VAR = "vannforing"

WRITE_TO_FILE = True


# =============================================================================
# Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]
PATH_ERA5 = Path(config.dirs["era5_continuous_daily"])
PATH_STATION = config.dirs["station"]

PRECIP_FILE = PATH_ERA5 / PRECIP_VAR / f"{PRECIP_VAR}_0.5x0.5_{YEAR}.nc"
MSL_FILE = PATH_ERA5 / MSL_VAR / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"
STREAMFLOW_FILE = f"{PATH_STATION}streamflow.Bergheim.nc"


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 11.2

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9
LEGEND_FONTSIZE = 9


# =============================================================================
# Plot styling
# =============================================================================

PRECIP_LEVELS = np.arange(5, 55, 5)
PRECIP_ZERO_THRESHOLD = 5.0
PRECIP_CMAP = plt.get_cmap("GnBu").copy()
PRECIP_CMAP.set_under("white")

MSL_CONTOUR_LEVELS = np.arange(975, 1045, 5)
MSL_CONTOUR_COLOR = "0.7"
MSL_CONTOUR_LINEWIDTH = 1.5

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

STATION_MARKER_SIZE = 5
STATION_MARKER_FACE_COLOR = "yellow"
STATION_MARKER_EDGE_COLOR = "black"
STATION_MARKER_EDGE_WIDTH = 0.6

STREAMFLOW_RANGE_FILL_ALPHA = 0.25
STREAMFLOW_MEDIAN_LINE_COLOR = "tab:red"
STREAMFLOW_MEDIAN_LINEWIDTH = 1.4
STREAMFLOW_YEAR_LINEWIDTH = 1.2


# =============================================================================
# Catchment and station metadata
# =============================================================================

CATCHMENTS = {
    "drammen": {
        "filename": "catchment_nve_regine_drammen.geojson",
        "label": "Drammen catchment",
    },
    "glomma": {
        "filename": "catchment_nve_regine_glomma.geojson",
        "label": "Glomma catchment",
    },
}


STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
]


# =============================================================================
# Settings helpers
# =============================================================================

def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. "
            f"Valid options are: {valid_names}."
        )

    return CATCHMENTS[catchment_name]


def make_output_filename(catchment_name):
    """Create output filename."""
    return f"{PATH_OUT}fig-temp.png"


# =============================================================================
# Coordinate helpers
# =============================================================================

def get_time_coord_name(da):
    """Return the time coordinate name."""
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
    """Convert 1D grid-cell centers to grid-cell edges."""
    centers = np.asarray(centers)

    if centers.ndim != 1:
        raise ValueError("centers must be 1D.")
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer edges.")

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# =============================================================================
# Data loading
# =============================================================================

def open_era5_variable(filename, variable):
    """Open one ERA5 variable and convert to plotting units."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in {filename}")

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    return ds


def load_streamflow(filename):
    """Load Bergheim streamflow data."""
    ds = xr.open_dataset(filename)
    ds = ds.sel(time=slice("1921-01-01", "2025-12-31"))
    return ds


def select_date(da, target_date):
    """Select one date from a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    return da.sel({time_name: target_date}).load()


def load_precipitation(ds_tp, target_date):
    """Load daily precipitation for one date."""
    return select_date(ds_tp[PRECIP_VAR], target_date)


def load_msl(ds_msl, target_date):
    """Load mean sea level pressure for one date."""
    return select_date(ds_msl[MSL_VAR], target_date)


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load the catchment and keep only its outer boundary."""
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    catchment_path = Path(base_dir) / filename
    gdf = gpd.read_file(catchment_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)

    union_geom = gdf.to_crs(metric_crs).geometry.union_all()

    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)

    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon(
            [Polygon(poly.exterior) for poly in union_geom.geoms]
        )

    else:
        outer_geom = union_geom

    outer_gdf = gpd.GeoDataFrame(
        geometry=[outer_geom],
        crs=metric_crs,
    ).to_crs(plot_crs)

    return outer_gdf.geometry.iloc[0]


# =============================================================================
# Streamflow climatology helpers
# =============================================================================

def year_series_and_climatology_by_doy(da, year):
    """
    Extract one full year's daily series and compute day-of-year climatology.

    Returns selected-year values plus 2.5%, 50%, and 97.5% day-of-year
    statistics over all available years.
    """
    da = da.dropna("time")

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    x_dates = pd.date_range(start, end, freq="D")

    da_year = da.sel(time=slice(start, end))

    if da_year.sizes.get("time", 0) != len(x_dates):
        da_year = da_year.resample(time="1D").mean().sel(time=slice(start, end))

    y_year = da_year.values

    q = da.groupby("time.dayofyear").quantile(
        [0.025, 0.5, 0.975],
        dim="time",
    )

    doy = np.arange(1, 366)
    q_low = q.sel(quantile=0.025).sel(dayofyear=doy, drop=True).values
    q_median = q.sel(quantile=0.5).sel(dayofyear=doy, drop=True).values
    q_high = q.sel(quantile=0.975).sel(dayofyear=doy, drop=True).values

    return x_dates, y_year, q_low, q_high, q_median


# =============================================================================
# Figure setup
# =============================================================================

def make_figure_axes():
    """Create four map panels, a right colorbar, and a bottom time-series panel."""
    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 0.45],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[0:2, 2])
    ts_ax = fig.add_subplot(gs[2, 0:2])

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, axes, ts_ax, cbar_ax, proj_map, proj_data


def get_plot_axes(axes):
    """Return the four map panels."""
    return list(axes.flat)


# =============================================================================
# Map plotting functions
# =============================================================================

def plot_precipitation(ax, da_precip, proj_data):
    """Plot precipitation as shaded grid cells."""
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

    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)

    return ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        precip,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_ZERO_THRESHOLD,
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )


def plot_msl_contours(ax, da_msl, proj_data):
    """Plot labelled mean sea level pressure contours."""
    lon, lat = get_lon_lat(da_msl)

    contour = ax.contour(
        lon.values,
        lat.values,
        da_msl.values,
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
        fontsize=CONTOUR_LABELSIZE,
        fmt="%d",
        colors=MSL_CONTOUR_COLOR,
    )


def plot_catchment_boundary(ax, geometry, proj_data, linewidth=CATCHMENT_LINEWIDTH):
    """Overlay the selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=linewidth,
        zorder=9,
    )


def plot_stations(ax, stations, proj_data):
    """Overlay station locations."""
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=STATION_MARKER_SIZE,
            markeredgecolor=STATION_MARKER_EDGE_COLOR,
            markeredgewidth=STATION_MARKER_EDGE_WIDTH,
            markerfacecolor=STATION_MARKER_FACE_COLOR,
            linestyle="none",
            transform=proj_data,
            zorder=12,
        )


def plot_event_panel(ax, ds_tp, ds_msl, catchment_boundary, target_date, proj_data):
    """Plot precipitation, pressure, catchment, and stations for one date."""
    da_precip = load_precipitation(ds_tp, target_date)
    da_msl = load_msl(ds_msl, target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    plot_stations(ax, STATIONS, proj_data)

    return mesh


# =============================================================================
# Time-series plotting
# =============================================================================

def plot_streamflow_timeseries(ts_ax, ds_streamflow, year):
    """Plot Bergheim streamflow for the selected year and climatology."""
    da = ds_streamflow[STREAMFLOW_VAR]

    x, y, lo, hi, med = year_series_and_climatology_by_doy(da, year)

    ts_ax.fill_between(
        x,
        lo,
        hi,
        alpha=STREAMFLOW_RANGE_FILL_ALPHA,
        linewidth=0,
        label="95% interval over all years",
    )

    ts_ax.plot(
        x,
        med,
        linewidth=STREAMFLOW_MEDIAN_LINEWIDTH,
        color=STREAMFLOW_MEDIAN_LINE_COLOR,
        label="Median over all years",
    )

    ts_ax.plot(
        x,
        y,
        linewidth=STREAMFLOW_YEAR_LINEWIDTH,
        label=f"{year}",
    )

    ts_ax.set_title(
        "e) Bergheim station 1921-2025",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel("Streamflow (m³/s)", fontsize=AXIS_LABELSIZE)
    ts_ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)

    ts_ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    ts_ax.set_xlim(start, end)
    ts_ax.margins(x=0)

    ts_ax.xaxis.set_major_locator(mdates.MonthLocator())
    ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ts_ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ts_ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)


# =============================================================================
# Figure finishing
# =============================================================================

def add_panel_titles(axes):
    """Add panel labels and date titles."""
    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, label, lag, date in zip(
        get_plot_axes(axes),
        panel_labels,
        EVENT_LAGS,
        EVENT_DATES,
    ):
        formatted_date = (
            np.datetime64(date)
            .astype("datetime64[D]")
            .astype(object)
            .strftime("%B %-d")
        )

        ax.set_title(
            f"{label} Day {lag:+d}: {formatted_date} 2023",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_colorbar(fig, mesh, cbar_ax):
    """Add vertical precipitation colorbar beside the map panels."""
    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.set_label(
        "Daily accumulated precipitation (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )

    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_legend(axes, catchment_label):
    """Add map legend inside panel a."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CATCHMENT_EDGE_COLOR,
            linewidth=2,
            label=catchment_label,
        ),
        Line2D(
            [0],
            [0],
            color=MSL_CONTOUR_COLOR,
            linewidth=MSL_CONTOUR_LINEWIDTH,
            label="Mean sea level pressure (hPa)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STATION_MARKER_FACE_COLOR,
            markeredgecolor=STATION_MARKER_EDGE_COLOR,
            markeredgewidth=STATION_MARKER_EDGE_WIDTH,
            markersize=6,
            label="Bergheim station",
        ),
    ]

    legend = axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)


def align_timeseries_axis_to_map_panels(fig, axes, ts_ax):
    """Align panel e with the combined left/right borders of panels a-d."""
    fig.canvas.draw()

    left = min(
        axes[0, 0].get_position().x0,
        axes[1, 0].get_position().x0,
    )

    right = max(
        axes[0, 1].get_position().x1,
        axes[1, 1].get_position().x1,
    )

    pos = ts_ax.get_position()

    ts_ax.set_position(
        [
            left,
            pos.y0,
            right - left,
            pos.height,
        ]
    )


def finalize_figure(
    fig,
    axes,
    ts_ax,
    cbar_ax,
    mesh,
    ds_streamflow,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, streamflow panel, save, and show."""
    add_panel_titles(axes)
    plot_streamflow_timeseries(ts_ax, ds_streamflow, YEAR)

    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, catchment_label)

    fig.subplots_adjust(
        left=0.065,
        right=0.96,
        bottom=0.075,
        top=0.96,
    )

    align_timeseries_axis_to_map_panels(fig, axes, ts_ax)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run the full plotting workflow."""
    catchment = get_catchment_settings(CATCHMENT_NAME)
    savepath = make_output_filename(CATCHMENT_NAME)

    ds_tp = open_era5_variable(PRECIP_FILE, PRECIP_VAR)
    ds_msl = open_era5_variable(MSL_FILE, MSL_VAR)
    ds_streamflow = load_streamflow(STREAMFLOW_FILE)

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        fig, axes, ts_ax, cbar_ax, proj_map, proj_data = make_figure_axes()

        mesh = None

        for ax, target_date in zip(get_plot_axes(axes), EVENT_DATES):
            mesh = plot_event_panel(
                ax=ax,
                ds_tp=ds_tp,
                ds_msl=ds_msl,
                catchment_boundary=catchment_boundary,
                target_date=target_date,
                proj_data=proj_data,
            )

        finalize_figure(
            fig=fig,
            axes=axes,
            ts_ax=ts_ax,
            cbar_ax=cbar_ax,
            mesh=mesh,
            ds_streamflow=ds_streamflow,
            catchment_label=catchment["label"],
            savepath=savepath,
        )

    finally:
        ds_tp.close()
        ds_msl.close()
        ds_streamflow.close()


if __name__ == "__main__":
    main()
