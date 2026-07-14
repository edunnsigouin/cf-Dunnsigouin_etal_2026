#!/usr/bin/env python3
"""
Plot one high-ranking S2S precipitation event.

The figure contains:
1. Four map panels around the selected event date.
2. Daily precipitation as shading.
3. Mean sea level pressure as labelled grey contours.
4. Snowmelt, defined as daily change in SWE < 0, as stippling or hatching.
5. The map catchment boundary in red.
6. Panel e showing snow water equivalent (sd) averaged over a separate
   time-series catchment.

The snow water equivalent panel contains:
- the selected event averaged over the catchment
- the 95% interval from all hindcast years and ensemble members
- event-date markers for the four map-panel dates.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Catchment shown in panels a-d and used to select the ranked event
MAP_CATCHMENT_NAME = "drammen"

# Catchment used for the spatial averages in panels e-f
TIMESERIES_CATCHMENT_NAME = "bergheim"

EVENT_RANK = 1              # options: 1-5

# These offsets define which dates are plotted relative to date_of_max.
# They are not used in the panel titles.
EVENT_LAGS = [-2, -1, 0, 1]

PRECIP_VAR = "tp24"
MSL_VAR = "msl"
SNOW_VAR = "sd"

WRITE_TO_FILE = False


# =============================================================================
# Drammen city settings
# =============================================================================

DRAMMEN_LON = 10.2045
DRAMMEN_LAT = 59.7440
DRAMMEN_LABEL = "Drammen"

DRAMMEN_MARKER_SIZE = 5
DRAMMEN_MARKER_FACE_COLOR = "yellow"
DRAMMEN_MARKER_EDGE_COLOR = "black"
DRAMMEN_MARKER_EDGE_WIDTH = 0.6


# =============================================================================
# Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 13.4

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9


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

SNOWMELT_OVERLAY = "stippling"  # options: "stippling", "hatching"
SNOWMELT_THRESHOLD = 0.0

SNOWMELT_DOT_COLOR = "orange"
SNOWMELT_DOT_SIZE = 4
SNOWMELT_DOT_ALPHA = 1.0
SNOWMELT_DOT_STRIDE = 1

SNOWMELT_HATCH_PATTERN = "////"
SNOWMELT_HATCH_COLOR = "orange"
SNOWMELT_HATCH_LINEWIDTH = 1.0

SNOWMELT_ZORDER = 8

TIMESERIES_RANGE_FILL_COLOR = "tab:blue"
TIMESERIES_RANGE_FILL_ALPHA = 0.20
TIMESERIES_EVENT_LINE_COLOR = "tab:blue"
TIMESERIES_EVENT_LINEWIDTH = 2.0
TIMESERIES_MEDIAN_LINE_COLOR = "tab:red"
TIMESERIES_MEDIAN_LINEWIDTH = 2.0


# =============================================================================
# Catchment and event metadata
# =============================================================================

CATCHMENTS = {
    "drammen": {
        "filename": "catchment_nve_regine_drammen.geojson",
        "weights_id": "regine_drammen",
        "label": "Drammen catchment",
    },
    "glomma": {
        "filename": "catchment_nve_regine_glomma.geojson",
        "weights_id": "regine_glomma",
        "label": "Glomma catchment",
    },
    "bergheim": {
        "filename": "catchment_nve_nevina_bergheim.geojson",
        "weights_id": "nevina_bergheim",
        "label": "Bergheim catchment",
    },
}



TOP_EVENTS = {
    "drammen": [
        {
            "rank": 1,
            "model_type": "hindcast",
            "forecast_date": "2021-04-26",
            "date_of_max": "2021-06-06",
            "hdate": 20150426.0,
            "ensemble_member": 7,
        },
        {
            "rank": 2,
            "model_type": "hindcast",
            "forecast_date": "2022-04-28",
            "date_of_max": "2022-06-02",
            "hdate": 20140428.0,
            "ensemble_member": 2,
        },
        {
            "rank": 3,
            "model_type": "hindcast",
            "forecast_date": "2021-04-29",
            "date_of_max": "2021-05-28",
            "hdate": 20190429.0,
            "ensemble_member": 4,
        },
        {
            "rank": 4,
            "model_type": "hindcast",
            "forecast_date": "2020-04-23",
            "date_of_max": "2020-06-03",
            "hdate": 20150423.0,
            "ensemble_member": 11,
        },
        {
            "rank": 5,
            "model_type": "forecast",
            "forecast_date": "2021-04-26",
            "date_of_max": "2021-06-07",
            "hdate": None,
            "ensemble_member": 17,
        },
    ],
    "glomma": [
        {
            "rank": 1,
            "model_type": "hindcast",
            "forecast_date": "2022-04-28",
            "date_of_max": "2022-06-02",
            "hdate": 20140428.0,
            "ensemble_member": 2,
        },
        {
            "rank": 2,
            "model_type": "hindcast",
            "forecast_date": "2022-04-21",
            "date_of_max": "2022-05-31",
            "hdate": 20160421.0,
            "ensemble_member": 2,
        },
        {
            "rank": 3,
            "model_type": "hindcast",
            "forecast_date": "2022-04-11",
            "date_of_max": "2022-05-10",
            "hdate": 20150411.0,
            "ensemble_member": 3,
        },
        {
            "rank": 4,
            "model_type": "hindcast",
            "forecast_date": "2021-04-15",
            "date_of_max": "2021-05-29",
            "hdate": 20150415.0,
            "ensemble_member": 8,
        },
        {
            "rank": 5,
            "model_type": "forecast",
            "forecast_date": "2023-04-03",
            "date_of_max": "2023-04-29",
            "hdate": None,
            "ensemble_member": 43,
        },
    ],
}


# =============================================================================
# Metadata helpers
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


def get_selected_event(catchment_name, event_rank):
    """Return metadata for the selected ranked event."""
    if catchment_name not in TOP_EVENTS:
        raise ValueError(f"No event metadata available for '{catchment_name}'.")

    for event in TOP_EVENTS[catchment_name]:
        if event["rank"] == event_rank:
            return event

    raise ValueError(f"Rank {event_rank} not found for '{catchment_name}'.")


def get_event_dates(event):
    """Return dates to plot as strings."""
    date_of_max = np.datetime64(event["date_of_max"], "D")

    return [
        str(date_of_max + np.timedelta64(lag, "D"))
        for lag in EVENT_LAGS
    ]


def make_s2s_file(event, variable, grid):
    """Create the expected S2S file path."""
    return (
        S2S_BASE_DIR
        / event["model_type"]
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{grid}_{event['forecast_date']}.nc"
    )


def make_output_filename(catchment_name, event_rank):
    """Create output filename."""
    return f"{PATH_OUT}fig-0X1.png"


# =============================================================================
# Coordinate helpers
# =============================================================================

def get_time_coord_name(da):
    """Return the time coordinate name used by the DataArray."""
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]

    return lon, lat


def get_member_coord_name(da):
    """Return ensemble member coordinate name."""
    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify ensemble member coordinate.")


def get_hdate_coord_name(da):
    """Return hindcast date coordinate name."""
    for name in ["hdate", "hindcast_date"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify hindcast date coordinate.")


def centers_to_edges(centers):
    """Convert 1D grid-cell center coordinates to grid-cell edges."""
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
# Catchment-weight helpers
# =============================================================================

def make_catchment_weights_file(catchment_name, grid):
    """Create the expected catchment-weight file path for the S2S/ERA5 grid."""
    catchment = get_catchment_settings(catchment_name)

    return (
        Path(PATH_CATCHMENT)
        / f"weights_catchment_{catchment['weights_id']}_era5_{grid}.nc"
    )


def check_dims(da, expected_dims, name):
    """Check that required dimensions exist."""
    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def load_weights(filename, spatial_dims):
    """Load predefined catchment weights."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Catchment weights not found: {filename}")

    ds = xr.open_dataset(filename)

    if "catchment_weight" not in ds:
        ds.close()
        raise KeyError(
            f"'catchment_weight' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    weights = ds["catchment_weight"].astype("float32").load()
    ds.close()

    weights.name = "catchment_weight"
    check_dims(weights, spatial_dims, "Catchment weights")

    return weights


def align_weights(da, weights):
    """Align catchment weights to the data grid."""
    time_name = get_time_coord_name(da) if any(
        name in da.dims or name in da.coords
        for name in ["time", "valid_time"]
    ) else None

    if time_name is not None and time_name in da.dims:
        grid_template = da.isel({time_name: 0}, drop=True)
    else:
        grid_template = da

    try:
        return weights.reindex_like(grid_template)
    except Exception:
        return weights.broadcast_like(grid_template)


def catchment_mean(da, weights, spatial_dims):
    """Calculate a catchment-weighted spatial mean."""
    weights = align_weights(da, weights)

    valid = xr.ufuncs.isfinite(da) & xr.ufuncs.isfinite(weights) & (weights > 0)

    weighted_sum = (da.where(valid) * weights.where(valid)).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = weights.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    out = weighted_sum / weight_sum
    out.name = "catchment_mean"
    out.attrs["units"] = da.attrs.get("units", "mm/day")

    return out


# =============================================================================
# Data loading
# =============================================================================

def open_s2s_variable(filename, variable):
    """Open one S2S variable and convert it to plotting units."""
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

    elif variable == SNOW_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    return ds


def select_event_member(ds, event, variable):
    """Select hindcast date and ensemble member when those dimensions exist."""
    da = ds[variable]

    if event["model_type"] == "hindcast":
        for name in ["hdate", "hindcast_date"]:
            if name in da.dims or name in da.coords:
                da = da.sel({name: event["hdate"]})
                break

    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            if ((event["model_type"] == "hindcast") and (event["ensemble_member"] == 11) and (variable == 'tp24')):
                da = da.sel({name: 51})
            else:
                da = da.sel({name: event["ensemble_member"]})
            break

    return da


def date_exists(da, target_date):
    """Check whether a target date exists in a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    available_dates = da[time_name].values.astype("datetime64[ns]")

    return target_date in available_dates


def select_date(da, target_date):
    """Select one date from a DataArray and load it into memory."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_daily_variable(event, target_date, variable):
    """
    Load one daily S2S field.

    The script first tries the 0.5-degree file. If the target date is missing,
    it then tries the 0.25-degree file.
    """
    grids_to_try = ["0.5x0.5", "0.25x0.25"]

    for grid in grids_to_try:
        filename = make_s2s_file(event, variable, grid)

        if not filename.exists():
            continue

        ds = open_s2s_variable(filename, variable)

        try:
            da = select_event_member(ds, event, variable)

            if date_exists(da, target_date):
                return select_date(da, target_date)

        finally:
            ds.close()

    raise ValueError(
        f"Could not find {variable} for {target_date} "
        f"in either 0.5x0.5 or 0.25x0.25 files."
    )


def load_precipitation(event, target_date):
    """Load daily precipitation in mm/day."""
    return load_daily_variable(event, target_date, PRECIP_VAR)


def load_msl(event, target_date):
    """Load mean sea level pressure in hPa."""
    return load_daily_variable(event, target_date, MSL_VAR)


def load_snowmelt(event, lag):
    """
    Compute daily SWE change.

    Snowmelt is later plotted where:
        SWE(current day) - SWE(previous day) < 0
    """
    date_of_max = np.datetime64(event["date_of_max"], "D")
    previous_date = date_of_max + np.timedelta64(lag - 1, "D")
    current_date = date_of_max + np.timedelta64(lag, "D")

    da_previous = load_daily_variable(event, str(previous_date), SNOW_VAR)
    da_current = load_daily_variable(event, str(current_date), SNOW_VAR)

    da_change = da_current - da_previous
    da_change.attrs["units"] = "mm"
    da_change.attrs["long_name"] = "Daily change in snow water equivalent"

    return da_change


def load_catchment_timeseries_for_variable(
    event,
    catchment_name,
    variable,
    daily_change=False,
):
    """Load the selected event time series averaged over the catchment.

    When daily_change=True, return the change from the previous day after
    calculating the catchment mean.
    """
    grids_to_try = ["0.5x0.5", "0.25x0.25"]
    spatial_dims = ("latitude", "longitude")

    for grid in grids_to_try:
        filename = make_s2s_file(event, variable, grid)
        weights_filename = make_catchment_weights_file(catchment_name, grid)

        if not filename.exists() or not weights_filename.exists():
            continue

        ds = open_s2s_variable(filename, variable)

        try:
            da = select_event_member(ds, event, variable)
            weights = load_weights(weights_filename, spatial_dims)

            da_mean = catchment_mean(
                da=da,
                weights=weights,
                spatial_dims=spatial_dims,
            )

            if daily_change:
                time_name = get_time_coord_name(da_mean)
                da_mean = da_mean.diff(time_name, label="upper")
                da_mean.attrs["units"] = "mm/day"
                da_mean.attrs["long_name"] = (
                    "Daily change in catchment-mean snow water equivalent"
                )

            da_mean = da_mean.load()
            da_mean.name = (
                f"catchment_mean_daily_change_{variable}"
                if daily_change
                else f"catchment_mean_{variable}"
            )
            da_mean.attrs["selected_grid"] = grid
            da_mean.attrs["catchment_name"] = catchment_name
            da_mean.attrs["variable"] = variable

            return da_mean

        finally:
            ds.close()

    raise FileNotFoundError(
        f"Could not find selected {variable} catchment-mean time series "
        f"for {event['forecast_date']} and catchment '{catchment_name}'."
    )


def load_hindcast_member_stats_for_variable(
    event,
    catchment_name,
    variable,
    daily_change=False,
):
    """
    Load the 95% interval and median for a catchment-mean variable.

    Statistics are computed across all hindcast dates and ensemble members
    in the file for the selected forecast initialization date.
    """
    if event["model_type"] != "hindcast":
        raise ValueError(
            "Hindcast/member statistics are only defined for hindcast events."
        )

    grids_to_try = ["0.5x0.5", "0.25x0.25"]
    spatial_dims = ("latitude", "longitude")

    for grid in grids_to_try:
        filename = make_s2s_file(event, variable, grid)
        weights_filename = make_catchment_weights_file(catchment_name, grid)

        if not filename.exists() or not weights_filename.exists():
            continue

        ds = open_s2s_variable(filename, variable)

        try:
            da = ds[variable]
            weights = load_weights(weights_filename, spatial_dims)

            da_mean = catchment_mean(
                da=da,
                weights=weights,
                spatial_dims=spatial_dims,
            )

            if daily_change:
                time_name = get_time_coord_name(da_mean)
                da_mean = da_mean.diff(time_name, label="upper")
                da_mean.attrs["units"] = "mm/day"
                da_mean.attrs["long_name"] = (
                    "Daily change in catchment-mean snow water equivalent"
                )

            hdate_name = get_hdate_coord_name(da_mean)
            member_name = get_member_coord_name(da_mean)
            sample_dims = [hdate_name, member_name]

            n_samples_total = (
                da_mean.sizes[hdate_name]
                * da_mean.sizes[member_name]
            )

            da_lower = da_mean.quantile(
                0.025,
                dim=sample_dims,
                skipna=True,
            ).load().squeeze(drop=True)

            da_upper = da_mean.quantile(
                0.975,
                dim=sample_dims,
                skipna=True,
            ).load().squeeze(drop=True)

            da_median = da_mean.median(
                dim=sample_dims,
                skipna=True,
            ).load().squeeze(drop=True)

            attrs = {
                "selected_grid": grid,
                "catchment_name": catchment_name,
                "variable": variable,
                "n_samples_total": n_samples_total,
                "n_samples_used": n_samples_total,
            }

            da_lower.attrs.update(attrs)
            da_upper.attrs.update(attrs)
            da_median.attrs.update(attrs)

            return da_lower, da_upper, da_median

        finally:
            ds.close()

    raise FileNotFoundError(
        f"Could not find {variable} hindcast/member data and weights "
        f"for {event['forecast_date']} and catchment '{catchment_name}'."
    )


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load the catchment and keep only the outer boundary."""
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
# Figure setup
# =============================================================================

def make_figure_axes():
    """Create four maps, a colorbar, and two SWE time-series panels."""
    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    gs = GridSpec(
        4,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 0.40, 0.40],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[0:2, 2])
    swe_ax = fig.add_subplot(gs[2, 0:2])
    swe_change_ax = fig.add_subplot(gs[3, 0:2], sharex=swe_ax)

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return (
        fig, axes, swe_ax, swe_change_ax, cbar_ax, proj_map, proj_data
    )


def get_plot_axes(axes):
    """Return the four map panels."""
    return list(axes.flat)


# =============================================================================
# Plotting functions
# =============================================================================

def plot_drammen_city(ax, proj_data):
    """Mark the city of Drammen."""
    ax.plot(
        DRAMMEN_LON,
        DRAMMEN_LAT,
        marker="o",
        markersize=DRAMMEN_MARKER_SIZE,
        markeredgecolor=DRAMMEN_MARKER_EDGE_COLOR,
        markeredgewidth=DRAMMEN_MARKER_EDGE_WIDTH,
        markerfacecolor=DRAMMEN_MARKER_FACE_COLOR,
        linestyle="none",
        transform=proj_data,
        zorder=12,
    )


def plot_precipitation(ax, da_precip, proj_data):
    """Plot daily precipitation as shaded grid cells."""
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


def plot_snowmelt(ax, da_snowmelt, proj_data):
    """Overlay locations where daily SWE change is negative."""
    lon, lat = get_lon_lat(da_snowmelt)

    snowmelt = np.isfinite(da_snowmelt.values)
    snowmelt &= da_snowmelt.values < SNOWMELT_THRESHOLD

    if SNOWMELT_OVERLAY == "stippling":
        plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data)

    elif SNOWMELT_OVERLAY == "hatching":
        plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data)

    else:
        raise ValueError(
            "SNOWMELT_OVERLAY must be either 'stippling' or 'hatching'."
        )


def plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data):
    """Plot snowmelt as orange stippling."""
    iy, ix = np.where(snowmelt)

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


def plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data):
    """Plot snowmelt as orange hatching."""
    old_hatch_color = plt.rcParams["hatch.color"]
    old_hatch_linewidth = plt.rcParams["hatch.linewidth"]

    try:
        plt.rcParams["hatch.color"] = SNOWMELT_HATCH_COLOR
        plt.rcParams["hatch.linewidth"] = SNOWMELT_HATCH_LINEWIDTH

        ax.contourf(
            lon.values,
            lat.values,
            snowmelt.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[SNOWMELT_HATCH_PATTERN],
            transform=proj_data,
            zorder=SNOWMELT_ZORDER,
        )

    finally:
        plt.rcParams["hatch.color"] = old_hatch_color
        plt.rcParams["hatch.linewidth"] = old_hatch_linewidth


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


def plot_event_panel(ax, event, lag, target_date, catchment_boundary, proj_data):
    """Plot one map panel."""
    da_precip = load_precipitation(event, target_date)
    da_msl = load_msl(event, target_date)
    da_snowmelt = load_snowmelt(event, lag)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_snowmelt(ax, da_snowmelt, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    # plot_drammen_city(ax, proj_data)

    return mesh


def plot_catchment_timeseries(
    ax,
    event,
    event_dates,
    catchment_name,
    variable,
    panel_label,
    title,
    ylabel,
    event_line_label,
    daily_change=False,
    add_zero_line=False,
):
    """Plot one catchment-mean event series and hindcast/member interval."""
    da_event = load_catchment_timeseries_for_variable(
        event=event,
        catchment_name=catchment_name,
        variable=variable,
        daily_change=daily_change,
    )

    da_lower, da_upper, da_median = (
        load_hindcast_member_stats_for_variable(
            event=event,
            catchment_name=catchment_name,
            variable=variable,
            daily_change=daily_change,
        )
    )

    time_name = get_time_coord_name(da_event)

    ax.fill_between(
        da_lower[time_name].values,
        da_lower.values,
        da_upper.values,
        color=TIMESERIES_RANGE_FILL_COLOR,
        alpha=TIMESERIES_RANGE_FILL_ALPHA,
        linewidth=0,
        label=(
            "95% interval over all years and members "
            f"(n={da_lower.attrs['n_samples_used']})"
        ),
    )

    # Uncomment to display the hindcast/member median.
    # ax.plot(
    #     da_median[time_name].values,
    #     da_median.values,
    #     color=TIMESERIES_MEDIAN_LINE_COLOR,
    #     linewidth=TIMESERIES_MEDIAN_LINEWIDTH,
    #     label="Median over all hindcast years and members",
    # )

    if add_zero_line:
        ax.axhline(0.0, color="0.5", linewidth=1.0, zorder=1)

    ax.plot(
        da_event[time_name].values,
        da_event.values,
        color=TIMESERIES_EVENT_LINE_COLOR,
        linewidth=TIMESERIES_EVENT_LINEWIDTH,
        label=event_line_label,
    )

    for date in event_dates:
        value = da_event.sel(
            {time_name: np.datetime64(date)},
            method="nearest",
        )

        ax.scatter(
            value[time_name].values,
            value.values,
            color=TIMESERIES_EVENT_LINE_COLOR,
            s=35,
            zorder=5,
        )

    ax.set_title(
        f"{panel_label} {title}",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )
    ax.set_ylabel(ylabel, fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ax.tick_params(labelsize=TICK_LABELSIZE)

    start_date = da_event[time_name].values[0]
    end_date = da_event[time_name].values[-1]

    ax.set_xlim(start_date, end_date)
    ax.margins(x=0)

    tick_interval_days = 2
    tick_dates = np.arange(
        start_date,
        end_date + np.timedelta64(1, "D"),
        np.timedelta64(tick_interval_days, "D"),
    )
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    ax.legend(frameon=False, fontsize=9)


def plot_swe_timeseries(ax, event, event_dates, catchment_name):
    """Plot panel e: catchment-mean snow water equivalent."""
    catchment_label = get_catchment_settings(catchment_name)["label"]

    plot_catchment_timeseries(
        ax=ax,
        event=event,
        event_dates=event_dates,
        catchment_name=catchment_name,
        variable=SNOW_VAR,
        panel_label="e)",
        title=f"{catchment_label} snow water equivalent",
        ylabel="mm",
        event_line_label="Counterfactual Storm Hans",
    )


def plot_swe_change_timeseries(ax, event, event_dates, catchment_name):
    """Plot panel f: daily change in catchment-mean SWE."""
    catchment_label = get_catchment_settings(catchment_name)["label"]

    plot_catchment_timeseries(
        ax=ax,
        event=event,
        event_dates=event_dates,
        catchment_name=catchment_name,
        variable=SNOW_VAR,
        panel_label="f)",
        title=f"{catchment_label} daily change in snow water equivalent",
        ylabel="mm/day",
        event_line_label="Counterfactual Storm Hans",
        daily_change=True,
        add_zero_line=True,
    )


# =============================================================================
# Figure finishing
# =============================================================================

def format_panel_date(date):
    """Format a date as 'Jun 4' for panel titles."""
    date_object = (
        np.datetime64(date)
        .astype("datetime64[D]")
        .astype(object)
    )

    return f"{date_object.strftime('%B')} {date_object.day}"


def add_panel_titles(axes, event_dates):
    """Add panel labels and calendar dates."""
    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, date in zip(
        axes,
        panel_labels,
        event_dates,
    ):
        ax.set_title(
            f"{panel_label} {format_panel_date(date)}",
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
        "precipitation (mm)",
        fontsize=AXIS_LABELSIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def get_snowmelt_legend_handle():
    """Return legend handle for the selected snowmelt overlay."""
    if SNOWMELT_OVERLAY == "hatching":
        return Patch(
            facecolor="white",
            edgecolor=SNOWMELT_HATCH_COLOR,
            hatch=SNOWMELT_HATCH_PATTERN,
            label=r"Snowmelt ($\Delta$SWE < 0)",
        )

    return Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=SNOWMELT_DOT_COLOR,
        markersize=5,
        label=r"Snowmelt ($\Delta$SWE < 0)",
    )


def add_legend(axes, map_catchment_label):
    """Add map legend inside the upper-left panel."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CATCHMENT_EDGE_COLOR,
            linewidth=2,
            label=map_catchment_label,
        ),
        Line2D(
            [0],
            [0],
            color=MSL_CONTOUR_COLOR,
            linewidth=MSL_CONTOUR_LINEWIDTH,
            label="Mean sea level pressure (hPa)",
        ),
        get_snowmelt_legend_handle(),
    ]

    legend = axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)


def align_timeseries_axes_to_map_panels(fig, axes, timeseries_axes):
    """Align panels e and f with the combined borders of panels a-d."""
    fig.canvas.draw()

    left = min(
        axes[0, 0].get_position().x0,
        axes[1, 0].get_position().x0,
    )
    right = max(
        axes[0, 1].get_position().x1,
        axes[1, 1].get_position().x1,
    )

    for ax in timeseries_axes:
        pos = ax.get_position()
        ax.set_position(
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
    swe_ax,
    swe_change_ax,
    cbar_ax,
    mesh,
    event,
    event_dates,
    map_catchment_label,
    timeseries_catchment_name,
    savepath,
):
    """Add the SWE series, titles, colorbar and legend, then save/show."""
    plot_axes = get_plot_axes(axes)
    add_panel_titles(plot_axes, event_dates)

    plot_swe_timeseries(
        ax=swe_ax,
        event=event,
        event_dates=event_dates,
        catchment_name=timeseries_catchment_name,
    )

    plot_swe_change_timeseries(
        ax=swe_change_ax,
        event=event,
        event_dates=event_dates,
        catchment_name=timeseries_catchment_name,
    )

    # Only the lower time-series panel needs x-axis labels.
    swe_ax.set_xlabel("")
    swe_ax.tick_params(labelbottom=False)

    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, map_catchment_label)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.055,
        top=0.97,
    )

    align_timeseries_axes_to_map_panels(
        fig=fig,
        axes=axes,
        timeseries_axes=[swe_ax, swe_change_ax],
    )

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run the full plotting workflow."""
    map_catchment = get_catchment_settings(MAP_CATCHMENT_NAME)
    get_catchment_settings(TIMESERIES_CATCHMENT_NAME)

    event = get_selected_event(MAP_CATCHMENT_NAME, EVENT_RANK)
    event_dates = get_event_dates(event)
    savepath = make_output_filename(MAP_CATCHMENT_NAME, EVENT_RANK)

    catchment_boundary = load_catchment_outer_boundary(
        filename=map_catchment["filename"],
        base_dir=PATH_CATCHMENT,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    (
        fig, axes, swe_ax, swe_change_ax, cbar_ax, proj_map, proj_data
    ) = make_figure_axes()

    mesh = None

    for ax, lag, target_date in zip(
        get_plot_axes(axes),
        EVENT_LAGS,
        event_dates,
    ):
        mesh = plot_event_panel(
            ax=ax,
            event=event,
            lag=lag,
            target_date=target_date,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

    finalize_figure(
        fig=fig,
        axes=axes,
        swe_ax=swe_ax,
        swe_change_ax=swe_change_ax,
        cbar_ax=cbar_ax,
        mesh=mesh,
        event=event,
        event_dates=event_dates,
        map_catchment_label=map_catchment["label"],
        timeseries_catchment_name=TIMESERIES_CATCHMENT_NAME,
        savepath=savepath,
    )


if __name__ == "__main__":
    main()
