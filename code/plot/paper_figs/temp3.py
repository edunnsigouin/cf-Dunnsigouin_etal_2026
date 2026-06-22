#!/usr/bin/env python3
"""
Plot precipitation ensemble time series, event-relative precipitation maps,
and runoff ensemble time series.

The figure contains:
1. Panel a):
   - catchment-mean forecast precipitation ensemble
   - ERA5 precipitation
   - SeNorge precipitation
   - forecast initialization date

2. Panels b-e):
   - daily precipitation as shading
   - mean sea level pressure as labelled grey contours
   - Drammen catchment boundary in red
   - Drammen city marker
   - zoomed inset in panel e

3. Panel f):
   - catchment-mean forecast runoff ensemble
   - selected runoff ensemble member highlighted
   - ERA5 runoff
   - forecast initialization date

Panels a and f have the same horizontal span and the same vertical height.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# 1. User-defined input parameters
# =============================================================================

CATCHMENT_NAME = "drammen"
CATCHMENT_ID = "regine_drammen"

FORECAST_DATE = "2023-08-05"
ENSEMBLE_MEMBER = 48
MODEL_TYPE = "forecast"
GRID = "0.25x0.25"

DAY_ZERO_DATE = "2023-08-08"

EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

EVENT_LAGS = [-2, -1, 0, 1]

PRECIP_VAR = "tp24"
MSL_VAR = "msl"

RUNOFF_VAR = "ro24"       # S2S forecast runoff
ERA5_RUNOFF_VAR = "ro"    # ERA5 runoff

# Panel a) precipitation time-series settings
X_DAYS = 1
N_DAYS_BEFORE = 3
M_DAYS_LEAD = 6
EXTREME_MEMBER_MODE = "max"  # options: "max", "min"

WRITE_TO_FILE = True


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

OUTPUT_FILENAME = f"{PATH_OUT}temp.png"


# =============================================================================
# 2.1 Panel a precipitation paths
# =============================================================================

FORECAST_PRECIP_FILENAME = (
    config.dirs["s2s_forecast_daily"]
    + PRECIP_VAR
    + "/"
    + f"{PRECIP_VAR}_{GRID}_{FORECAST_DATE}.nc"
)

FORECAST_PRECIP_WEIGHTS_FILENAME = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT_ID}_era5_{GRID}.nc"
)

ERA5_PRECIP_VAR = "tp24"
ERA5_PRECIP_GRID = "0.25x0.25"
ERA5_PRECIP_DOMAIN = "norway"

ERA5_PRECIP_PATH = (
    config.dirs["era5_continuous_daily"]
    + ERA5_PRECIP_VAR
    + "/"
)

ERA5_PRECIP_FILE_PATTERN = (
    f"{ERA5_PRECIP_VAR}_{ERA5_PRECIP_GRID}"
    + "_{year}.nc"
)

ERA5_PRECIP_WEIGHTS_FILENAME = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT_ID}_era5_{ERA5_PRECIP_GRID}.nc"
)

SENORGE_VAR = "rr"

SENORGE_PATH = (
    config.dirs["senorge_continuous_daily"]
    + SENORGE_VAR
    + "/"
)

SENORGE_FILE_PATTERN = SENORGE_VAR + "_{year}.nc"

SENORGE_WEIGHTS_FILENAME = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT_ID}_senorge.nc"
)


# =============================================================================
# 2.2 Panel f runoff paths
# =============================================================================

ERA5_RUNOFF_DOMAIN = "norway"

ERA5_RUNOFF_PATH = (
    config.dirs["era5_continuous_daily_scandinavia"]
    + ERA5_RUNOFF_VAR
    + "/"
)

ERA5_RUNOFF_FILE_PATTERN = (
    f"{ERA5_RUNOFF_VAR}_{GRID}"
    + "_{year}.nc"
)

RUNOFF_WEIGHTS_FILENAME = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT_ID}_era5_{GRID}.nc"
)


# =============================================================================
# 3. Figure and map settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 15.2

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9

DRAMMEN_LON = 10.2045
DRAMMEN_LAT = 59.7440
DRAMMEN_LABEL = "Drammen"

ZOOM_MAP_EXTENT = [6.5, 11.5, 59, 61.5]

DRAMMEN_MARKER_SIZE = 5
DRAMMEN_MARKER_FACE_COLOR = "yellow"
DRAMMEN_MARKER_EDGE_COLOR = "black"
DRAMMEN_MARKER_EDGE_WIDTH = 0.6


# =============================================================================
# 4. Plot styling
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

FORECAST_COLOR = "0.7"
FORECAST_LINEWIDTH = 1.0
FORECAST_ALPHA = 0.6

ERA5_PRECIP_COLOR = "tab:blue"
SENORGE_COLOR = "tab:red"
EXTREME_MEMBER_COLOR = "tab:green"
OBS_LINEWIDTH = 2.5

RUNOFF_ENSEMBLE_COLOR = "0.7"
RUNOFF_ENSEMBLE_LINEWIDTH = 1.0
RUNOFF_ENSEMBLE_ALPHA = 0.6

RUNOFF_SELECTED_COLOR = "tab:blue"
RUNOFF_SELECTED_LINEWIDTH = 2.5

ERA5_RUNOFF_COLOR = "tab:red"
ERA5_RUNOFF_LINEWIDTH = 2.5

EVENT_MARKER_SIZE = 35


# =============================================================================
# 5. Catchment metadata
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
}


# =============================================================================
# 6. General helper functions
# =============================================================================

def get_catchment_settings(catchment_name):
    """Return metadata for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. "
            f"Valid options are: {valid_names}."
        )

    return CATCHMENTS[catchment_name]


def make_s2s_file(variable):
    """Create S2S forecast file path."""
    return (
        S2S_BASE_DIR
        / MODEL_TYPE
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{GRID}_{FORECAST_DATE}.nc"
    )


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


def get_member_coord_name(da):
    """Return ensemble member coordinate name."""
    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify ensemble member coordinate.")


def centers_to_edges(centers):
    """Convert 1D grid-cell centers to grid-cell edges."""
    centers = np.asarray(centers)

    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional.")

    if centers.size < 2:
        raise ValueError("At least two center points are needed to infer edges.")

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def check_dims(da, expected_dims, name):
    """Check that required dimensions exist."""
    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def load_weights(filename, spatial_dims):
    """Load catchment weights."""
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
    time_name = get_time_coord_name(da)

    if time_name in da.dims:
        grid_template = da.isel({time_name: 0}, drop=True)
    else:
        grid_template = da

    try:
        return weights.reindex_like(grid_template)
    except Exception:
        return weights.broadcast_like(grid_template)


def catchment_mean(da, weights, spatial_dims):
    """Calculate catchment-weighted spatial mean."""
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
    out.attrs["units"] = da.attrs.get("units", "")

    return out


def get_plot_period(forecast_date, n_days_before, m_days_lead, x_days=1):
    """Define plot and loading periods."""
    init_date = pd.to_datetime(forecast_date)

    plot_start = init_date - pd.Timedelta(days=n_days_before)
    plot_end = init_date + pd.Timedelta(days=m_days_lead)

    load_start = plot_start - pd.Timedelta(days=x_days + 1)
    load_end = plot_end + pd.Timedelta(days=x_days + 1)

    years = np.arange(load_start.year, load_end.year + 1)

    return init_date, plot_start, plot_end, load_start, load_end, years


def round_time_to_nearest_day(da):
    """Round timestamps to nearest calendar day."""
    time_name = get_time_coord_name(da)
    rounded_time = pd.to_datetime(da[time_name].values).round("D")

    return da.assign_coords({time_name: rounded_time})


def subset_to_period(da, start_date, end_date):
    """Subset data to a calendar-date period."""
    time_name = get_time_coord_name(da)

    return da.sel({time_name: slice(start_date, end_date)})


# =============================================================================
# 7. Unit conversion helpers
# =============================================================================

def standardize_precip_units(da, variable):
    """Convert precipitation to mm."""
    units = str(da.attrs.get("units", "")).strip().lower()

    if variable == "tp24" or units in {"m", "meter", "metre"}:
        da = da * 1000.0
        da.attrs["units"] = "mm"

    elif units in {"kg/m^2", "kg/m2", "kg m-2"}:
        da.attrs["units"] = "mm"

    return da


def standardize_runoff_units(da):
    """Convert runoff to mm/day if stored in metres."""
    units = str(da.attrs.get("units", "")).strip().lower()

    if units in {"m", "meter", "metre", ""}:
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif units in {"mm", "mm/day", "mm d-1"}:
        da.attrs["units"] = "mm/day"

    return da


# =============================================================================
# 8. Panel a precipitation data
# =============================================================================

def load_forecast_precipitation(filename, variable):
    """Load forecast precipitation."""
    with xr.open_dataset(filename) as ds:
        da = ds[variable].load()

    da = standardize_precip_units(da, variable)

    check_dims(
        da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast precipitation",
    )

    return da


def preprocess_era5(ds):
    """Drop unnecessary ERA5 ensemble dimension if present."""
    return ds.drop_vars("number", errors="ignore")


def load_era5_precipitation(years, time_start, time_end):
    """Load ERA5 precipitation."""
    filenames = [
        ERA5_PRECIP_PATH + ERA5_PRECIP_FILE_PATTERN.format(year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if ERA5_PRECIP_DOMAIN is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(ERA5_PRECIP_DOMAIN)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    da = ds[ERA5_PRECIP_VAR].sel(time=slice(time_start, time_end))
    da = standardize_precip_units(da, ERA5_PRECIP_VAR)

    check_dims(
        da,
        expected_dims=("time", "latitude", "longitude"),
        name="ERA5 precipitation",
    )

    return da


def load_senorge_precipitation(years, time_start, time_end):
    """Load SeNorge precipitation year by year."""
    yearly_data = []

    for year in years:
        filename = SENORGE_PATH + SENORGE_FILE_PATTERN.format(year=int(year))

        ds = xr.open_dataset(filename)
        ds = xr.decode_cf(ds)

        if SENORGE_VAR not in ds:
            ds.close()
            raise KeyError(
                f"'{SENORGE_VAR}' not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[SENORGE_VAR].sel(time=slice(time_start, time_end))

        fill_value = da.attrs.get("_FillValue")
        if fill_value is not None:
            da = da.where(da != fill_value)

        da = standardize_precip_units(da, SENORGE_VAR)

        check_dims(
            da,
            expected_dims=("time", "Y", "X"),
            name="SeNorge precipitation",
        )

        yearly_data.append(da.load())
        ds.close()

    return xr.concat(yearly_data, dim="time").sortby("time")


def xday_accumulation(da, x_days):
    """Calculate trailing x-day accumulated precipitation."""
    time_name = get_time_coord_name(da)

    out = (
        da
        .rolling({time_name: x_days}, min_periods=x_days)
        .sum()
        .dropna(time_name, how="any")
    )

    out.name = f"{x_days}day_accumulated_precipitation"
    out.attrs["units"] = "mm"

    return out


def shift_senorge_time_back_one_day(da):
    """Shift SeNorge timestamps one day earlier."""
    shifted_time = pd.to_datetime(da.time.values) - pd.Timedelta(days=1)

    return da.assign_coords(time=shifted_time)


def restrict_observations_to_common_dates(era5, senorge):
    """Restrict ERA5 and SeNorge to common dates."""
    common_dates = np.intersect1d(era5.time.values, senorge.time.values)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between ERA5 and SeNorge.")

    era5 = era5.sel(time=common_dates)
    senorge = senorge.sel(time=common_dates)

    return era5, senorge


def restrict_forecast_to_observation_dates(forecast, obs_dates):
    """Restrict forecast to dates also available in observations."""
    common_dates = np.intersect1d(forecast.time.values, obs_dates)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between forecast and observations.")

    return forecast.sel(time=common_dates)


def get_extreme_ensemble_member(forecast, mode="max"):
    """Return ensemble member with largest or smallest value in time."""
    if mode not in ["max", "min"]:
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

    if mode == "max":
        member_extreme = forecast.max(dim="time")
        member = int(member_extreme.idxmax(dim="number"))
        value = float(member_extreme.max())

        member_series = forecast.sel(number=member)
        time = pd.Timestamp(
            member_series.time[
                member_series.argmax(dim="time")
            ].values
        )

    else:
        member_extreme = forecast.min(dim="time")
        member = int(member_extreme.idxmin(dim="number"))
        value = float(member_extreme.min())

        member_series = forecast.sel(number=member)
        time = pd.Timestamp(
            member_series.time[
                member_series.argmin(dim="time")
            ].values
        )

    return member, value, time


def prepare_panel_a_data():
    """Load and process forecast, ERA5, and SeNorge data for panel a."""
    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        obs_years,
    ) = get_plot_period(
        forecast_date=FORECAST_DATE,
        n_days_before=N_DAYS_BEFORE,
        m_days_lead=M_DAYS_LEAD,
        x_days=X_DAYS,
    )

    forecast_weights = load_weights(
        filename=FORECAST_PRECIP_WEIGHTS_FILENAME,
        spatial_dims=("latitude", "longitude"),
    )

    forecast = load_forecast_precipitation(
        filename=FORECAST_PRECIP_FILENAME,
        variable=PRECIP_VAR,
    )

    forecast_mean = catchment_mean(
        da=forecast,
        weights=forecast_weights,
        spatial_dims=("latitude", "longitude"),
    )

    forecast_accumulated = xday_accumulation(
        da=forecast_mean,
        x_days=X_DAYS,
    )

    forecast_accumulated = round_time_to_nearest_day(forecast_accumulated)

    forecast_accumulated = subset_to_period(
        da=forecast_accumulated,
        start_date=init_date,
        end_date=plot_end,
    )

    era5_weights = load_weights(
        filename=ERA5_PRECIP_WEIGHTS_FILENAME,
        spatial_dims=("latitude", "longitude"),
    )

    era5 = load_era5_precipitation(
        years=obs_years,
        time_start=load_start,
        time_end=load_end,
    )

    era5_mean = catchment_mean(
        da=era5,
        weights=era5_weights,
        spatial_dims=("latitude", "longitude"),
    )

    era5_accumulated = xday_accumulation(
        da=era5_mean,
        x_days=X_DAYS,
    )

    era5_accumulated = round_time_to_nearest_day(era5_accumulated)

    era5_accumulated = subset_to_period(
        da=era5_accumulated,
        start_date=plot_start,
        end_date=plot_end,
    )

    senorge_weights = load_weights(
        filename=SENORGE_WEIGHTS_FILENAME,
        spatial_dims=("Y", "X"),
    )

    senorge = load_senorge_precipitation(
        years=obs_years,
        time_start=load_start,
        time_end=load_end,
    )

    senorge = shift_senorge_time_back_one_day(senorge)

    senorge_mean = catchment_mean(
        da=senorge,
        weights=senorge_weights,
        spatial_dims=("Y", "X"),
    )

    senorge_accumulated = xday_accumulation(
        da=senorge_mean,
        x_days=X_DAYS,
    )

    senorge_accumulated = round_time_to_nearest_day(senorge_accumulated)

    senorge_accumulated = subset_to_period(
        da=senorge_accumulated,
        start_date=plot_start,
        end_date=plot_end,
    )

    era5_plot, senorge_plot = restrict_observations_to_common_dates(
        era5=era5_accumulated,
        senorge=senorge_accumulated,
    )

    forecast_plot = restrict_forecast_to_observation_dates(
        forecast=forecast_accumulated,
        obs_dates=era5_plot.time.values,
    )

    return (
        forecast_plot,
        era5_plot,
        senorge_plot,
        init_date,
        plot_start,
        plot_end,
    )


# =============================================================================
# 9. Map data loading
# =============================================================================

def open_s2s_variable(variable):
    """Open one S2S variable and convert to plotting units."""
    filename = make_s2s_file(variable)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    return ds


def select_member(da):
    """Select the requested ensemble member."""
    member_name = get_member_coord_name(da)

    return da.sel({member_name: ENSEMBLE_MEMBER})


def select_date(da, target_date):
    """Select one target date and load it into memory."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_daily_variable(variable, target_date):
    """Load one daily S2S field for the selected member and date."""
    ds = open_s2s_variable(variable)

    try:
        da = ds[variable]
        da = select_member(da)
        da = select_date(da, target_date)

    finally:
        ds.close()

    return da


def load_precipitation(target_date):
    """Load daily precipitation in mm/day."""
    return load_daily_variable(PRECIP_VAR, target_date)


def load_msl(target_date):
    """Load mean sea level pressure in hPa."""
    return load_daily_variable(MSL_VAR, target_date)


# =============================================================================
# 10. Panel f runoff data
# =============================================================================

def open_forecast_runoff_dataset():
    """Open S2S forecast ro24 and convert to mm/day."""
    filename = make_s2s_file(RUNOFF_VAR)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if RUNOFF_VAR not in ds:
        raise KeyError(
            f"Variable '{RUNOFF_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    ds[RUNOFF_VAR] = standardize_runoff_units(ds[RUNOFF_VAR])

    return ds


def load_all_forecast_runoff_members():
    """Load S2S ro24 for all forecast ensemble members."""
    ds = open_forecast_runoff_dataset()

    try:
        da = ds[RUNOFF_VAR].load()

    finally:
        ds.close()

    return da


def load_era5_runoff(years, time_start, time_end):
    """Load ERA5 ro over the requested period."""
    filenames = [
        ERA5_RUNOFF_PATH + ERA5_RUNOFF_FILE_PATTERN.format(year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if ERA5_RUNOFF_DOMAIN is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(ERA5_RUNOFF_DOMAIN)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if ERA5_RUNOFF_VAR not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{ERA5_RUNOFF_VAR}' not found in ERA5 files. "
            f"Available variables: {available}"
        )

    da = ds[ERA5_RUNOFF_VAR].sel(time=slice(time_start, time_end))
    da = standardize_runoff_units(da)

    check_dims(
        da=da,
        expected_dims=("time", "latitude", "longitude"),
        name="ERA5 runoff",
    )

    return da


def load_forecast_runoff_catchment_mean(init_date, plot_end):
    """Load catchment-mean S2S ro24 for all ensemble members."""
    spatial_dims = ("latitude", "longitude")

    da = load_all_forecast_runoff_members()

    check_dims(
        da=da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast runoff",
    )

    weights = load_weights(
        filename=RUNOFF_WEIGHTS_FILENAME,
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=init_date,
        end_date=plot_end,
    )

    return da_mean


def load_era5_runoff_catchment_mean(
    years,
    load_start,
    load_end,
    plot_start,
    plot_end,
):
    """Load catchment-mean ERA5 ro for panel f."""
    spatial_dims = ("latitude", "longitude")

    da = load_era5_runoff(
        years=years,
        time_start=load_start,
        time_end=load_end,
    )

    weights = load_weights(
        filename=RUNOFF_WEIGHTS_FILENAME,
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=plot_start,
        end_date=plot_end,
    )

    return da_mean


# =============================================================================
# 11. Catchment boundary
# =============================================================================

def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
    """Load catchment and keep only the outer boundary."""
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
# 12. Figure setup
# =============================================================================

def make_figure_axes():
    """
    Create panel a, four map panels b-e, panel f, and a map colorbar.

    Panels a and f have identical height ratios and span the same horizontal
    distance as the two map columns.
    """
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
        height_ratios=[0.58, 1.0, 1.0, 0.58],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    precip_ts_ax = fig.add_subplot(gs[0, 0:2])

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[2, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[2, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[1:3, 2])
    runoff_ts_ax = fig.add_subplot(gs[3, 0:2])

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, precip_ts_ax, axes, runoff_ts_ax, cbar_ax, proj_map, proj_data


def get_plot_axes(axes):
    """Return map panels as a flat list."""
    return list(axes.flat)


# =============================================================================
# 13. Panel a plotting
# =============================================================================

def plot_panel_a_timeseries(ts_ax):
    """Plot forecast ensemble, ERA5, and SeNorge precipitation in panel a."""
    (
        forecast,
        era5,
        senorge,
        init_date,
        plot_start,
        plot_end,
    ) = prepare_panel_a_data()

    extreme_member, _, _ = get_extreme_ensemble_member(
        forecast,
        mode=EXTREME_MEMBER_MODE,
    )

    ts_ax.plot(
        [],
        [],
        color=FORECAST_COLOR,
        linewidth=FORECAST_LINEWIDTH,
        alpha=FORECAST_ALPHA,
        label="Forecast ensemble",
    )

    for member in forecast["number"].values:
        if member == extreme_member:
            continue

        ts_ax.plot(
            forecast["time"],
            forecast.sel(number=member),
            color=FORECAST_COLOR,
            linewidth=FORECAST_LINEWIDTH,
            alpha=FORECAST_ALPHA,
        )

    extreme_label = (
        "Wettest ensemble member"
        if EXTREME_MEMBER_MODE == "max"
        else "Driest ensemble member"
    )

    ts_ax.plot(
        forecast["time"],
        forecast.sel(number=extreme_member),
        color=EXTREME_MEMBER_COLOR,
        linewidth=OBS_LINEWIDTH,
        zorder=10,
        label=extreme_label,
    )

    ts_ax.plot(
        era5["time"],
        era5,
        color=ERA5_PRECIP_COLOR,
        linewidth=OBS_LINEWIDTH,
        label="ERA5",
    )

    ts_ax.plot(
        senorge["time"],
        senorge,
        color=SENORGE_COLOR,
        linewidth=OBS_LINEWIDTH,
        label="SeNorge",
    )

    ts_ax.axvline(
        init_date,
        color="k",
        linewidth=1.2,
        linestyle="--",
        label="Forecast initialization",
    )

    ts_ax.set_title(
        (
            f"a) {X_DAYS}-day accumulated precipitation, "
            f"{CATCHMENT_ID}, initialized {FORECAST_DATE}"
        ),
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel(
        f"{X_DAYS}-day precipitation (mm)",
        fontsize=AXIS_LABELSIZE,
    )

    ts_ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(labelsize=TICK_LABELSIZE)

    ts_ax.grid(True, alpha=0.3)
    ts_ax.set_xlim(plot_start, plot_end)
    ts_ax.margins(x=0)

    plt.setp(
        ts_ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    ts_ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ts_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
    )


# =============================================================================
# 14. Map plotting
# =============================================================================

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


def plot_catchment_boundary(ax, geometry, proj_data, linewidth=CATCHMENT_LINEWIDTH):
    """Overlay catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=linewidth,
        zorder=9,
    )


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


def plot_event_panel(ax, target_date, catchment_boundary, proj_data):
    """Plot one daily map panel."""
    da_precip = load_precipitation(target_date)
    da_msl = load_msl(target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    

    return mesh


def add_zoom_inset(parent_ax, proj_map, proj_data, catchment_boundary):
    """Add zoomed inset map to panel e."""
    inset_ax = parent_ax.inset_axes(
        [0.01, 0.01, 0.3, 0.3],
        projection=proj_map,
        zorder=20,
    )

    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(1.0)
    inset_ax.set_extent(ZOOM_MAP_EXTENT, crs=proj_data)

    inset_ax.coastlines(
        resolution="10m",
        linewidth=0.4,
        color="black",
        zorder=2,
    )

    plot_catchment_boundary(
        inset_ax,
        catchment_boundary,
        proj_data,
        linewidth=1.0,
    )

    plot_drammen_city(inset_ax, proj_data)

    txt = inset_ax.text(
        DRAMMEN_LON,
        DRAMMEN_LAT + 0.06,
        DRAMMEN_LABEL,
        fontsize=7,
        color="yellow",
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=proj_data,
        zorder=13,
    )

    txt.set_path_effects(
        [
            pe.Stroke(linewidth=1.5, foreground="black"),
            pe.Normal(),
        ]
    )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    return inset_ax


# =============================================================================
# 15. Panel f plotting
# =============================================================================

def plot_panel_f_runoff_timeseries(ts_ax):
    """Plot forecast ensemble and ERA5 catchment-mean runoff in panel f."""
    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        years,
    ) = get_plot_period(
        forecast_date=FORECAST_DATE,
        n_days_before=N_DAYS_BEFORE,
        m_days_lead=M_DAYS_LEAD,
        x_days=1,
    )

    forecast = load_forecast_runoff_catchment_mean(
        init_date=init_date,
        plot_end=plot_end,
    )

    era5 = load_era5_runoff_catchment_mean(
        years=years,
        load_start=load_start,
        load_end=load_end,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    time_name = get_time_coord_name(forecast)
    member_name = get_member_coord_name(forecast)

    ts_ax.plot(
        [],
        [],
        color=RUNOFF_ENSEMBLE_COLOR,
        linewidth=RUNOFF_ENSEMBLE_LINEWIDTH,
        alpha=RUNOFF_ENSEMBLE_ALPHA,
        label="Forecast ensemble",
    )

    for member in forecast[member_name].values:
        if member == ENSEMBLE_MEMBER:
            continue

        ts_ax.plot(
            forecast[time_name].values,
            forecast.sel({member_name: member}).values,
            color=RUNOFF_ENSEMBLE_COLOR,
            linewidth=RUNOFF_ENSEMBLE_LINEWIDTH,
            alpha=RUNOFF_ENSEMBLE_ALPHA,
        )

    selected = forecast.sel({member_name: ENSEMBLE_MEMBER})

    ts_ax.plot(
        selected[time_name].values,
        selected.values,
        color=RUNOFF_SELECTED_COLOR,
        linewidth=RUNOFF_SELECTED_LINEWIDTH,
        label=f"Selected member {ENSEMBLE_MEMBER}",
        zorder=10,
    )

    for date in EVENT_DATES:
        value = selected.sel(
            {time_name: np.datetime64(date)},
            method="nearest",
        )

        ts_ax.scatter(
            value[time_name].values,
            value.values,
            color=RUNOFF_SELECTED_COLOR,
            s=EVENT_MARKER_SIZE,
            zorder=11,
        )

    era5_time_name = get_time_coord_name(era5)

    ts_ax.plot(
        era5[era5_time_name].values,
        era5.values,
        color=ERA5_RUNOFF_COLOR,
        linewidth=ERA5_RUNOFF_LINEWIDTH,
        label="ERA5",
        zorder=9,
    )

    ts_ax.axvline(
        init_date,
        color="k",
        linewidth=1.2,
        linestyle="--",
        label="Forecast initialization",
    )

    ts_ax.set_title(
        "f) Drammen catchment mean runoff",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel(
        "Runoff (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )

    ts_ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(labelsize=TICK_LABELSIZE)

    ts_ax.grid(True, alpha=0.3)
    ts_ax.set_xlim(plot_start, plot_end)
    ts_ax.margins(x=0)

    plt.setp(
        ts_ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    ts_ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ts_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
    )


# =============================================================================
# 16. Figure finishing
# =============================================================================

def add_map_panel_titles(axes):
    """Add panel labels b-e and event-relative dates."""
    panel_labels = ["b)", "c)", "d)", "e)"]

    for ax, panel_label, lag, date in zip(
        axes,
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
            f"{panel_label} Day {lag:+d}: {formatted_date}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_colorbar(fig, mesh, cbar_ax):
    """Add precipitation colorbar."""
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


def add_map_legend(axes, catchment_label):
    """Add map legend inside panel b."""
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


def finalize_figure(
    fig,
    precip_ts_ax,
    axes,
    runoff_ts_ax,
    cbar_ax,
    proj_map,
    proj_data,
    mesh,
    catchment_boundary,
    catchment_label,
    savepath,
):
    """Finalize and show figure."""
    plot_panel_a_timeseries(precip_ts_ax)

    plot_axes = get_plot_axes(axes)
    add_map_panel_titles(plot_axes)

    add_colorbar(fig, mesh, cbar_ax)
    add_map_legend(axes, catchment_label)

    plot_panel_f_runoff_timeseries(runoff_ts_ax)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.055,
        top=0.965,
    )

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 17. Main workflow
# =============================================================================

def main():
    """Run full plotting workflow."""
    catchment = get_catchment_settings(CATCHMENT_NAME)

    catchment_boundary = load_catchment_outer_boundary(
        filename=catchment["filename"],
        base_dir=PATH_CATCHMENT,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    (
        fig,
        precip_ts_ax,
        axes,
        runoff_ts_ax,
        cbar_ax,
        proj_map,
        proj_data,
    ) = make_figure_axes()

    mesh = None

    for ax, target_date in zip(get_plot_axes(axes), EVENT_DATES):
        mesh = plot_event_panel(
            ax=ax,
            target_date=target_date,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

    finalize_figure(
        fig=fig,
        precip_ts_ax=precip_ts_ax,
        axes=axes,
        runoff_ts_ax=runoff_ts_ax,
        cbar_ax=cbar_ax,
        proj_map=proj_map,
        proj_data=proj_data,
        mesh=mesh,
        catchment_boundary=catchment_boundary,
        catchment_label=catchment["label"],
        savepath=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()
