#!/usr/bin/env python3
"""
Plot time evolution of one top S2S precipitation event.

The script:
1. Selects one catchment and one ranked event.
2. Uses date_of_max as event day 0.
3. Plots event-relative days -2, -1, 0, +1, +2.
4. Plots precipitation as shading.
5. Plots MSLP as grey labelled contours.
6. Plots snowmelt where:
       sd(day L) - sd(day L - 1) < SNOWMELT_THRESHOLD
   using either hatching or stippling.
7. Converts precipitation from m to mm, MSLP from Pa to hPa,
   and snow water equivalent from m to mm.
8. Uses the 0.5x0.5 file first, then falls back to 0.25x0.25.
9. Overlays the selected catchment boundary in red.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

path_in_catchment = config.dirs["nve"]
path_out = config.dirs["fig"]

CATCHMENT_NAME = "glomma"   # "drammen" or "glomma"
EVENT_RANK = 1               # choose 1-5
ACCUMULATION_DAYS = 1        # choose 1 or 2

filename_out = (
    f"{path_out}xy-tp-msl-snowmelt-event-evolution-"
    f"{CATCHMENT_NAME}-rank{EVENT_RANK}-{ACCUMULATION_DAYS}day.png"
)
write2file = True

PRECIP_VAR = "tp24"
MSL_VAR = "msl"
SNOW_VAR = "sd"

EVENT_LAGS = [-2, -1, 0, 1, 2]

FIG_WIDTH_IN = 16
FIG_HEIGHT_IN = 12

MAP_WSPACE = 0.02
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
    PRECIP_LEVELS = np.arange(5, 65, 5)
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

# --- Snowmelt overlay
SNOWMELT_OVERLAY = "hatching"  # Options: "hatching", "stippling"
SNOWMELT_THRESHOLD = 0.0       # melt if SWE change < threshold

# Hatching settings
SNOWMELT_HATCH_PATTERN = "////"
SNOWMELT_HATCH_COLOR = "orange"
SNOWMELT_HATCH_LINEWIDTH = 1.0

# Stippling settings
SNOWMELT_DOT_COLOR = "darkorange"
SNOWMELT_DOT_SIZE = 4
SNOWMELT_DOT_ALPHA = 0.7
SNOWMELT_DOT_STRIDE = 1

SNOWMELT_ZORDER = 8

# --- Catchment
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"
CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0


# =============================================================================
# Catchment metadata and event metadata
# =============================================================================
def get_catchment_metadata(catchment_name):
    catchments = {
        "drammen": {
            "label": "Drammen",
            "geojson": "catchment_nve_regine_drammen.geojson",
        },
        "glomma": {
            "label": "Glomma",
            "geojson": "catchment_nve_regine_glomma.geojson",
        },
    }

    if catchment_name not in catchments:
        raise ValueError(
            f"Unknown catchment: {catchment_name}. "
            f"Available catchments: {list(catchments.keys())}"
        )

    return catchments[catchment_name]


def get_top_events(catchment_name):
    events = {
        "drammen": [
            {
                "rank": 1,
                "max_value": 100.26,
                "model_type": "hindcast",
                "forecast_date": "2021-04-26",
                "date_of_max": "2021-06-06",
                "hdate": 20150426.0,
                "ensemble_member": 7,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2021-04-26.nc",
            },
            {
                "rank": 2,
                "max_value": 85.55,
                "model_type": "hindcast",
                "forecast_date": "2022-04-28",
                "date_of_max": "2022-06-02",
                "hdate": 20140428.0,
                "ensemble_member": 2,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-28.nc",
            },
            {
                "rank": 3,
                "max_value": 84.83,
                "model_type": "hindcast",
                "forecast_date": "2021-04-29",
                "date_of_max": "2021-05-28",
                "hdate": 20190429.0,
                "ensemble_member": 4,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2021-04-29.nc",
            },
            {
                "rank": 4,
                "max_value": 81.35,
                "model_type": "hindcast",
                "forecast_date": "2020-04-23",
                "date_of_max": "2020-06-03",
                "hdate": 20150423.0,
                "ensemble_member": 51,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2020-04-23.nc",
            },
            {
                "rank": 5,
                "max_value": 77.56,
                "model_type": "forecast",
                "forecast_date": "2021-04-26",
                "date_of_max": "2021-06-07",
                "hdate": None,
                "ensemble_member": 17,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "forecast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2021-04-26.nc",
            },
        ],
        "glomma": [
            {
                "rank": 1,
                "max_value": 101.00,
                "model_type": "hindcast",
                "forecast_date": "2022-04-28",
                "date_of_max": "2022-06-02",
                "hdate": 20140428.0,
                "ensemble_member": 2,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-28.nc",
            },
            {
                "rank": 2,
                "max_value": 73.97,
                "model_type": "hindcast",
                "forecast_date": "2022-04-21",
                "date_of_max": "2022-05-31",
                "hdate": 20160421.0,
                "ensemble_member": 2,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-21.nc",
            },
            {
                "rank": 3,
                "max_value": 63.55,
                "model_type": "hindcast",
                "forecast_date": "2022-04-11",
                "date_of_max": "2022-05-10",
                "hdate": 20150411.0,
                "ensemble_member": 3,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-11.nc",
            },
            {
                "rank": 4,
                "max_value": 61.12,
                "model_type": "hindcast",
                "forecast_date": "2021-04-15",
                "date_of_max": "2021-05-29",
                "hdate": 20150415.0,
                "ensemble_member": 8,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2021-04-15.nc",
            },
            {
                "rank": 5,
                "max_value": 57.02,
                "model_type": "forecast",
                "forecast_date": "2023-04-03",
                "date_of_max": "2023-04-29",
                "hdate": None,
                "ensemble_member": 43,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "forecast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2023-04-03.nc",
            },
        ],
    }

    if catchment_name not in events:
        raise ValueError(f"No event metadata available for {catchment_name}")

    return events[catchment_name]


def get_selected_event(catchment_name, event_rank):
    for event in get_top_events(catchment_name):
        if event["rank"] == event_rank:
            return event

    raise ValueError(f"Rank {event_rank} not found for {catchment_name}.")


# =============================================================================
# File helpers
# =============================================================================
def get_early_lead_file(source_file):
    return source_file.replace("0.5x0.5", "0.25x0.25")


def get_msl_file(precip_file):
    return precip_file.replace("/tp24/", "/msl/").replace("tp24_", "msl_")


def get_snow_file(precip_file):
    return precip_file.replace("/tp24/", "/sd/").replace("tp24_", "sd_")


def get_time_coord_name(da):
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]
    return lon, lat


# =============================================================================
# Data loading and selection
# =============================================================================
def load_dataset(filename, variable):
    """Open dataset and convert units."""
    ds = xr.open_dataset(filename)

    if variable == PRECIP_VAR and variable in ds:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    if variable == MSL_VAR and variable in ds:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    if variable == SNOW_VAR and variable in ds:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    return ds


def select_event_member(ds, event, variable):
    da = ds[variable]

    hdate_dim_candidates = ["hdate", "hindcast_date"]
    member_dim_candidates = ["number", "member", "ensemble_member", "realization"]

    if event["model_type"] == "hindcast":
        for dim in hdate_dim_candidates:
            if dim in da.dims or dim in da.coords:
                da = da.sel({dim: event["hdate"]})
                break

    for dim in member_dim_candidates:
        if dim in da.dims or dim in da.coords:
            da = da.sel({dim: event["ensemble_member"]})
            break

    return da


def has_date(da, target_date):
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    times = da[time_name].values.astype("datetime64[ns]")
    return target_date in times


def select_date(da, target_date):
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    return da.sel({time_name: target_date})


def load_single_day_variable(event, target_date, variable):
    """Load one daily field for precipitation, MSLP, or SWE."""
    if variable == PRECIP_VAR:
        base_file = event["source_file"]
    elif variable == MSL_VAR:
        base_file = get_msl_file(event["source_file"])
    elif variable == SNOW_VAR:
        base_file = get_snow_file(event["source_file"])
    else:
        raise ValueError(f"Unknown variable: {variable}")

    files_to_try = [
        base_file,
        get_early_lead_file(base_file),
    ]

    for filename in files_to_try:

        if not os.path.exists(filename):
            continue

        ds = load_dataset(filename, variable=variable)

        try:
            da = select_event_member(ds, event, variable=variable)

            if has_date(da, target_date):
                da_day = select_date(da, target_date).load()
                return da_day, filename

        finally:
            ds.close()

    raise ValueError(
        f"Could not find date {target_date} for {variable} in either "
        f"0.5x0.5 or 0.25x0.25 file."
    )


def load_event_precip(event, target_date, accumulation_days=1):
    """Load 1-day or 2-day accumulated precipitation."""
    if accumulation_days not in [1, 2]:
        raise ValueError("accumulation_days must be either 1 or 2.")

    target_date = np.datetime64(target_date, "D")

    if accumulation_days == 1:
        dates_needed = [target_date]
    else:
        dates_needed = [
            target_date - np.timedelta64(1, "D"),
            target_date,
        ]

    daily_fields = []
    source_files_used = []

    for date_needed in dates_needed:
        da_day, source_used = load_single_day_variable(
            event,
            target_date=date_needed,
            variable=PRECIP_VAR,
        )
        daily_fields.append(da_day)
        source_files_used.append(source_used)

    if accumulation_days == 1:
        da_out = daily_fields[0]
    else:
        da_out = daily_fields[0] + daily_fields[1]
        da_out.attrs["units"] = "mm"

    return da_out, source_files_used


def load_event_msl(event, target_date):
    """Load MSLP for the target day only."""
    return load_single_day_variable(
        event,
        target_date=target_date,
        variable=MSL_VAR,
    )


def load_event_snowmelt(event, lag):
    """
    Compute SWE change for one event-relative lag.

    SWE change = sd(day L) - sd(day L - 1).
    Negative values indicate snowmelt / snow loss.
    """
    date_of_max = np.datetime64(event["date_of_max"], "D")

    date_previous = date_of_max + np.timedelta64(lag - 1, "D")
    date_current = date_of_max + np.timedelta64(lag, "D")

    da_previous, source_previous = load_single_day_variable(
        event,
        target_date=str(date_previous),
        variable=SNOW_VAR,
    )

    da_current, source_current = load_single_day_variable(
        event,
        target_date=str(date_current),
        variable=SNOW_VAR,
    )

    da_change = da_current - da_previous
    da_change.attrs["units"] = "mm"
    da_change.attrs["long_name"] = "Snow water equivalent change"

    return da_change, source_previous, source_current


# =============================================================================
# Catchment geometry helper
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
# Plot setup helpers
# =============================================================================
def make_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
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


def centers_to_edges(centers):
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
# Plotting helpers
# =============================================================================
def plot_precipitation(ax, da_precip, proj_data):
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


def plot_snowmelt_overlay(ax, da_snowmelt, proj_data):
    """
    Plot snowmelt locations.

    Snowmelt is defined as:
        SWE change < SNOWMELT_THRESHOLD

    Options:
        SNOWMELT_OVERLAY = "hatching"
        SNOWMELT_OVERLAY = "stippling"
    """
    lon, lat = get_lon_lat(da_snowmelt)
    melt_mask = np.isfinite(da_snowmelt.values) & (
        da_snowmelt.values < SNOWMELT_THRESHOLD
    )

    overlay = SNOWMELT_OVERLAY.lower()

    if overlay == "hatching":

        old_hatch_color = plt.rcParams["hatch.color"]
        old_hatch_linewidth = plt.rcParams["hatch.linewidth"]

        plt.rcParams["hatch.color"] = SNOWMELT_HATCH_COLOR
        plt.rcParams["hatch.linewidth"] = SNOWMELT_HATCH_LINEWIDTH

        ax.contourf(
            lon.values,
            lat.values,
            melt_mask.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[SNOWMELT_HATCH_PATTERN],
            transform=proj_data,
            zorder=SNOWMELT_ZORDER,
        )

        plt.rcParams["hatch.color"] = old_hatch_color
        plt.rcParams["hatch.linewidth"] = old_hatch_linewidth

    elif overlay == "stippling":

        iy, ix = np.where(melt_mask)

        if SNOWMELT_DOT_STRIDE > 1:
            iy = iy[::SNOWMELT_DOT_STRIDE]
            ix = ix[::SNOWMELT_DOT_STRIDE]

        ax.scatter(
            lon.values[ix],
            lat.values[iy],
            s=SNOWMELT_DOT_SIZE,
            c=SNOWMELT_DOT_COLOR,
            alpha=SNOWMELT_DOT_ALPHA,
            transform=proj_data,
            zorder=SNOWMELT_ZORDER,
            linewidths=0,
        )

    else:
        raise ValueError(
            "SNOWMELT_OVERLAY must be either 'hatching' or 'stippling'."
        )


def plot_catchment_boundary(ax, geometry, proj_data):
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
    event,
    catchment_metadata,
    event_days,
    savepath=None,
    write2file=False,
):
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, lag, date in zip(plot_axes, EVENT_LAGS, event_days):
        ax.set_title(
            f"Day {lag:+d}: {date}",
            fontsize=title_fontsize,
            pad=3,
        )

    fig.suptitle(
        (
            f"{catchment_metadata['label']} | "
            f"Rank {event['rank']} event | "
            f"TP shading, MSLP contours, snowmelt {SNOWMELT_OVERLAY}"
        ),
        fontsize=title_fontsize + 2,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.06,
        top=0.92,
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    cax = fig.add_axes([0.69, 0.27, 0.23, 0.025])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_label(
        f"{ACCUMULATION_DAYS}-day accumulated precipitation (mm)",
        fontsize=axis_labelsize,
    )
    cbar.ax.tick_params(labelsize=tick_labelsize)

    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    catchment_metadata = get_catchment_metadata(CATCHMENT_NAME)
    event = get_selected_event(CATCHMENT_NAME, EVENT_RANK)

    date_of_max = np.datetime64(event["date_of_max"], "D")

    event_days = [
        str(date_of_max + np.timedelta64(lag, "D"))
        for lag in EVENT_LAGS
    ]

    catchment_boundary = load_catchment_outer_boundary(
        catchment_metadata["geojson"],
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

    for ax, lag, target_date in zip(plot_axes, EVENT_LAGS, event_days):

        da_precip, precip_files_used = load_event_precip(
            event,
            target_date=target_date,
            accumulation_days=ACCUMULATION_DAYS,
        )

        da_msl, msl_file_used = load_event_msl(
            event,
            target_date=target_date,
        )

        da_snowmelt, snow_previous_file, snow_current_file = load_event_snowmelt(
            event,
            lag=lag,
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

        plot_snowmelt_overlay(
            ax,
            da_snowmelt,
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
        event,
        catchment_metadata,
        event_days,
        savepath=filename_out,
        write2file=write2file,
    )
