#!/usr/bin/env python3
"""
Combined figure for Storm Hans.

The figure contains:

Top panel:
- Catchment-mean accumulated precipitation time series
- S2S forecast ensemble
- highlighted wettest forecast ensemble member
- ERA5 Storm Hans
- seNorge Storm Hans
- forecast initialization date

Panels a-d:
- S2S daily accumulated precipitation as shading
- S2S mean sea level pressure as grey labelled contours
- selected catchment boundary in red

Bottom panel e):
- S2S catchment-mean runoff ensemble
- selected ensemble member highlighted
- ERA5 runoff
- forecast initialization date

The second script is used as the base.
The first script's precipitation time-series figure is added as a new top panel.
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

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# 1. User-defined input parameters
# =============================================================================

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"

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

# Map-panel variables
PRECIP_VAR = "tp24"
MSL_VAR = "msl"

# Bottom runoff-panel variables
RUNOFF_VAR = "sro24"       # S2S forecast runoff
ERA5_RUNOFF_VAR = "sro"    # ERA5 runoff

# Top precipitation-panel variables
PRECIP_TS_FORECAST_VAR = "tp24"
PRECIP_TS_ERA5_VAR = "tp24"
PRECIP_TS_SENORGE_VAR = "rr"

# Accumulation length for the top precipitation panel
PRECIP_TS_X_DAYS = 1

# Top precipitation-panel date-window settings
PRECIP_TS_N_DAYS_BEFORE = 2
PRECIP_TS_M_DAYS_LEAD = 6

# Bottom runoff-panel date-window settings
RUNOFF_TS_N_DAYS_BEFORE = 3
RUNOFF_TS_M_DAYS_LEAD = 6

WRITE_TO_FILE = True


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

# ERA5 runoff files used in the bottom panel
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

# Forecast precipitation files used in the top panel
PRECIP_TS_FORECAST_FILE = (
    Path(config.dirs["s2s_forecast_daily"])
    / PRECIP_TS_FORECAST_VAR
    / f"{PRECIP_TS_FORECAST_VAR}_{GRID}_{FORECAST_DATE}.nc"
)

# ERA5 precipitation files used in the top panel
PRECIP_TS_ERA5_DOMAIN = "norway"

PRECIP_TS_ERA5_PATH = (
    Path(config.dirs["era5_continuous_daily"])
    / PRECIP_TS_ERA5_VAR
)

PRECIP_TS_ERA5_FILE_PATTERN = (
    f"{PRECIP_TS_ERA5_VAR}_{GRID}"
    + "_{year}.nc"
)

# seNorge precipitation files used in the top panel
PRECIP_TS_SENORGE_PATH = (
    Path(config.dirs["senorge_continuous_daily"])
    / PRECIP_TS_SENORGE_VAR
)

PRECIP_TS_SENORGE_FILE_PATTERN = (
    f"{PRECIP_TS_SENORGE_VAR}"
    + "_{year}.nc"
)

OUTPUT_FILENAME = f"{PATH_OUT}fig-temp.png"


# =============================================================================
# 3. Figure and map settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 14.0

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.12

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9


# =============================================================================
# 4. Plot styling
# =============================================================================

# Map precipitation
PRECIP_LEVELS = np.arange(5, 55, 5)
PRECIP_ZERO_THRESHOLD = 5.0
PRECIP_CMAP = plt.get_cmap("GnBu").copy()
PRECIP_CMAP.set_under("white")

# Map mean sea level pressure
MSL_CONTOUR_LEVELS = np.arange(975, 1045, 5)
MSL_CONTOUR_COLOR = "0.7"
MSL_CONTOUR_LINEWIDTH = 1.5

# Catchment
CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

# Top precipitation time series
PRECIP_TS_FORECAST_COLOR = "0.7"
PRECIP_TS_FORECAST_LINEWIDTH = 0.8
PRECIP_TS_FORECAST_ALPHA = 0.35

PRECIP_TS_EXTREME_MEMBER_COLOR = "tab:green"
PRECIP_TS_ERA5_COLOR = "tab:blue"
PRECIP_TS_SENORGE_COLOR = "tab:red"
PRECIP_TS_OBS_LINEWIDTH = 2.5

# Bottom runoff time series
RUNOFF_ENSEMBLE_COLOR = "0.7"
RUNOFF_ENSEMBLE_LINEWIDTH = 1.0
RUNOFF_ENSEMBLE_ALPHA = 0.6

RUNOFF_SELECTED_MEMBER_COLOR = "tab:blue"
RUNOFF_SELECTED_MEMBER_LINEWIDTH = 2.5

RUNOFF_ERA5_COLOR = "tab:red"
RUNOFF_ERA5_LINEWIDTH = 2.5

EVENT_MARKER_SIZE = 35

# Shared time-series styling
INITIALIZATION_LINE_COLOR = "k"
INITIALIZATION_LINE_WIDTH = 1.2
INITIALIZATION_LINE_STYLE = "--"

DATE_TICK_FORMAT = "%d %b"
DATE_TICK_INTERVAL_DAYS = 1
DATE_TICK_ROTATION = 30

LEGEND_FONTSIZE = 9


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
    """Return settings for the selected catchment."""

    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. "
            f"Valid options are: {valid_names}."
        )

    return CATCHMENTS[catchment_name]


def make_s2s_file(variable):
    """Create S2S forecast file path for the map panels and runoff panel."""

    return (
        S2S_BASE_DIR
        / MODEL_TYPE
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{GRID}_{FORECAST_DATE}.nc"
    )


def make_era5_weights_file(catchment_name, grid):
    """Return the ERA5-grid catchment-weight file."""

    catchment = get_catchment_settings(catchment_name)

    return (
        Path(PATH_CATCHMENT)
        / f"weights_catchment_{catchment['weights_id']}_era5_{grid}.nc"
    )


def make_senorge_weights_file(catchment_name):
    """Return the seNorge-grid catchment-weight file."""

    catchment = get_catchment_settings(catchment_name)

    return (
        Path(PATH_CATCHMENT)
        / f"weights_catchment_{catchment['weights_id']}_senorge.nc"
    )


def get_time_coord_name(da):
    """Return the time coordinate name."""

    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_member_coord_name(da):
    """Return the ensemble member coordinate name."""

    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify ensemble member coordinate.")


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""

    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]

    return lon, lat


def centers_to_edges(centers):
    """Convert one-dimensional grid-cell centers to grid-cell edges."""

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
    """Check that required dimensions are present."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def round_time_to_nearest_day(da):
    """Round timestamps to the nearest calendar day."""

    time_name = get_time_coord_name(da)
    rounded_time = pd.to_datetime(da[time_name].values).round("D")

    return da.assign_coords({time_name: rounded_time})


def shift_time_back_one_day(da):
    """Shift timestamps one day earlier."""

    time_name = get_time_coord_name(da)
    shifted_time = pd.to_datetime(da[time_name].values) - pd.Timedelta(days=1)

    return da.assign_coords({time_name: shifted_time})


def subset_to_period(da, start_date, end_date):
    """Subset a DataArray to a date period."""

    time_name = get_time_coord_name(da)

    return da.sel({time_name: slice(start_date, end_date)})


def get_plot_period(forecast_date, n_days_before, m_days_lead, x_days=1):
    """
    Return initialization date, plot window, loading window, and years.

    The loading window is wider than the plot window to allow rolling
    accumulations to be calculated safely.
    """

    init_date = pd.to_datetime(forecast_date)

    plot_start = init_date - pd.Timedelta(days=n_days_before)
    plot_end = init_date + pd.Timedelta(days=m_days_lead)

    load_start = plot_start - pd.Timedelta(days=x_days + 1)
    load_end = plot_end + pd.Timedelta(days=x_days + 1)

    years = np.arange(load_start.year, load_end.year + 1)

    return init_date, plot_start, plot_end, load_start, load_end, years


def remove_era5_ensemble_dimension_if_present(ds):
    """Remove unnecessary ERA5 ensemble coordinate if present."""

    return ds.drop_vars("number", errors="ignore")


# =============================================================================
# 7. Unit handling
# =============================================================================

def standardize_precipitation_units(da, variable_name):
    """Convert precipitation to millimetres when needed."""

    units = str(da.attrs.get("units", "")).strip().lower()

    if variable_name == "tp24" or units in {"m", "meter", "metre"}:
        da = da * 1000.0
        da.attrs["units"] = "mm"

    elif units in {"kg/m^2", "kg/m2", "kg m-2", "mm", "mm/day", "mm d-1"}:
        da.attrs["units"] = "mm"

    return da


def standardize_runoff_units(da):
    """
    Convert runoff to mm/day if it is stored in metres.

    S2S sro24 and ERA5 sro are assumed to represent daily accumulated runoff.
    """

    units = str(da.attrs.get("units", "")).strip().lower()

    if units in {"m", "meter", "metre", ""}:
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif units in {"mm", "mm/day", "mm d-1"}:
        da.attrs["units"] = "mm/day"

    return da


# =============================================================================
# 8. Catchment weighting
# =============================================================================

def load_weights(filename, spatial_dims):
    """Load predefined catchment weights."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Catchment weights not found: {filename}")

    ds = xr.open_dataset(filename)

    if "catchment_weight" not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"'catchment_weight' not found in {filename}. "
            f"Available variables: {available}"
        )

    weights = ds["catchment_weight"].astype("float32").load()
    ds.close()

    weights.name = "catchment_weight"

    check_dims(
        da=weights,
        expected_dims=spatial_dims,
        name="Catchment weights",
    )

    return weights


def align_weights_to_data_grid(da, weights):
    """
    Align two-dimensional catchment weights to the spatial grid of the data.

    The data may also have non-spatial dimensions such as time or ensemble member.
    """

    spatial_dims = tuple(weights.dims)

    grid_template = da

    for dim in da.dims:
        if dim not in spatial_dims:
            grid_template = grid_template.isel({dim: 0}, drop=True)

    weights_on_grid = weights.reindex_like(grid_template)

    if weights_on_grid.shape != grid_template.shape:
        raise ValueError(
            f"Weights shape {weights_on_grid.shape} does not match "
            f"data grid shape {grid_template.shape}."
        )

    if np.isfinite(weights_on_grid).sum().item() == 0:
        raise ValueError(
            "All aligned catchment weights are NaN. "
            "This usually means the weight coordinates and data coordinates "
            "do not overlap."
        )

    return weights_on_grid


def catchment_weighted_mean(da, weights, spatial_dims, output_name):
    """Calculate catchment-weighted spatial mean."""

    weights_on_grid = align_weights_to_data_grid(da, weights)

    valid = (
        xr.ufuncs.isfinite(da)
        & xr.ufuncs.isfinite(weights_on_grid)
        & (weights_on_grid > 0)
    )

    weighted_sum = (
        da.where(valid)
        * weights_on_grid.where(valid)
    ).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = weights_on_grid.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    out = weighted_sum / weight_sum
    out.name = output_name
    out.attrs["units"] = da.attrs.get("units", "")

    return out


def trailing_xday_accumulation(da, x_days):
    """Calculate trailing X-day accumulation."""

    accumulated = (
        da
        .rolling(time=x_days, min_periods=x_days)
        .sum()
        .dropna("time", how="any")
    )

    accumulated.name = f"{x_days}day_accumulation"
    accumulated.attrs["units"] = da.attrs.get("units", "mm")

    return accumulated


# =============================================================================
# 9. Top precipitation-panel data processing
# =============================================================================

def load_precip_ts_forecast():
    """Load S2S forecast precipitation for the top panel."""

    filename = Path(PRECIP_TS_FORECAST_FILE)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if PRECIP_TS_FORECAST_VAR not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"'{PRECIP_TS_FORECAST_VAR}' not found in {filename}. "
            f"Available variables: {available}"
        )

    da = ds[PRECIP_TS_FORECAST_VAR].load()
    ds.close()

    da = standardize_precipitation_units(
        da=da,
        variable_name=PRECIP_TS_FORECAST_VAR,
    )

    check_dims(
        da=da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast precipitation",
    )

    return da


def load_precip_ts_era5(years, loading_start, loading_end):
    """Load ERA5 precipitation for the top panel."""

    filenames = [
        str(PRECIP_TS_ERA5_PATH / PRECIP_TS_ERA5_FILE_PATTERN.format(year=int(year)))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=remove_era5_ensemble_dimension_if_present,
        combine="by_coords",
    )

    if PRECIP_TS_ERA5_DOMAIN is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(PRECIP_TS_ERA5_DOMAIN)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if PRECIP_TS_ERA5_VAR not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"'{PRECIP_TS_ERA5_VAR}' not found in ERA5 files. "
            f"Available variables: {available}"
        )

    da = ds[PRECIP_TS_ERA5_VAR].sel(
        time=slice(loading_start, loading_end)
    )

    da = standardize_precipitation_units(
        da=da,
        variable_name=PRECIP_TS_ERA5_VAR,
    )

    check_dims(
        da=da,
        expected_dims=("time", "latitude", "longitude"),
        name="ERA5 precipitation",
    )

    return da


def load_precip_ts_senorge(years, loading_start, loading_end):
    """Load seNorge precipitation for the top panel."""

    yearly_data = []

    for year in years:
        filename = PRECIP_TS_SENORGE_PATH / PRECIP_TS_SENORGE_FILE_PATTERN.format(
            year=int(year)
        )

        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        with xr.open_dataset(filename) as ds:
            ds = xr.decode_cf(ds)

            if PRECIP_TS_SENORGE_VAR not in ds:
                raise KeyError(
                    f"'{PRECIP_TS_SENORGE_VAR}' not found in {filename}. "
                    f"Available variables: {list(ds.data_vars)}"
                )

            da_one_year = ds[PRECIP_TS_SENORGE_VAR].sel(
                time=slice(loading_start, loading_end)
            )

            fill_value = da_one_year.attrs.get("_FillValue")

            if fill_value is not None:
                da_one_year = da_one_year.where(da_one_year != fill_value)

            da_one_year = standardize_precipitation_units(
                da=da_one_year,
                variable_name=PRECIP_TS_SENORGE_VAR,
            )

            check_dims(
                da=da_one_year,
                expected_dims=("time", "Y", "X"),
                name="seNorge precipitation",
            )

            yearly_data.append(da_one_year.load())

    da = xr.concat(yearly_data, dim="time").sortby("time")

    return da


def process_precip_ts_forecast(dates):
    """Process forecast precipitation for the top panel."""

    weights = load_weights(
        filename=make_era5_weights_file(CATCHMENT_NAME, GRID),
        spatial_dims=("latitude", "longitude"),
    )

    da = load_precip_ts_forecast()

    da_mean = catchment_weighted_mean(
        da=da,
        weights=weights,
        spatial_dims=("latitude", "longitude"),
        output_name="catchment_mean_precipitation",
    )

    da_accumulated = trailing_xday_accumulation(
        da=da_mean,
        x_days=PRECIP_TS_X_DAYS,
    )

    da_accumulated = round_time_to_nearest_day(da_accumulated)

    da_accumulated = subset_to_period(
        da=da_accumulated,
        start_date=dates["init_date"],
        end_date=dates["plot_end"],
    )

    return da_accumulated


def process_precip_ts_era5(dates):
    """Process ERA5 precipitation for the top panel."""

    weights = load_weights(
        filename=make_era5_weights_file(CATCHMENT_NAME, GRID),
        spatial_dims=("latitude", "longitude"),
    )

    da = load_precip_ts_era5(
        years=dates["years"],
        loading_start=dates["load_start"],
        loading_end=dates["load_end"],
    )

    da_mean = catchment_weighted_mean(
        da=da,
        weights=weights,
        spatial_dims=("latitude", "longitude"),
        output_name="catchment_mean_precipitation",
    )

    da_accumulated = trailing_xday_accumulation(
        da=da_mean,
        x_days=PRECIP_TS_X_DAYS,
    )

    da_accumulated = round_time_to_nearest_day(da_accumulated)

    da_accumulated = subset_to_period(
        da=da_accumulated,
        start_date=dates["plot_start"],
        end_date=dates["plot_end"],
    )

    return da_accumulated


def process_precip_ts_senorge(dates):
    """Process seNorge precipitation for the top panel."""

    weights = load_weights(
        filename=make_senorge_weights_file(CATCHMENT_NAME),
        spatial_dims=("Y", "X"),
    )

    da = load_precip_ts_senorge(
        years=dates["years"],
        loading_start=dates["load_start"],
        loading_end=dates["load_end"],
    )

    # Keep the time convention used in the first script.
    da = shift_time_back_one_day(da)

    da_mean = catchment_weighted_mean(
        da=da,
        weights=weights,
        spatial_dims=("Y", "X"),
        output_name="catchment_mean_precipitation",
    )

    da_accumulated = trailing_xday_accumulation(
        da=da_mean,
        x_days=PRECIP_TS_X_DAYS,
    )

    da_accumulated = round_time_to_nearest_day(da_accumulated)

    da_accumulated = subset_to_period(
        da=da_accumulated,
        start_date=dates["plot_start"],
        end_date=dates["plot_end"],
    )

    return da_accumulated


def keep_only_common_observation_dates(era5, senorge):
    """Keep only dates available in both ERA5 and seNorge."""

    common_dates = np.intersect1d(era5.time.values, senorge.time.values)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between ERA5 and seNorge.")

    return era5.sel(time=common_dates), senorge.sel(time=common_dates)


def keep_forecast_dates_available_in_observations(forecast, observation_dates):
    """Keep forecast dates that are also available in the observations."""

    common_dates = np.intersect1d(forecast.time.values, observation_dates)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between forecast and observations.")

    return forecast.sel(time=common_dates)


def find_wettest_ensemble_member(forecast):
    """
    Find the ensemble member with the largest accumulated precipitation value.
    """

    member_name = get_member_coord_name(forecast)

    maximum_by_member = forecast.max(dim="time")

    wettest_member = int(maximum_by_member.idxmax(dim=member_name))
    maximum_value = float(maximum_by_member.max())

    wettest_series = forecast.sel({member_name: wettest_member})
    time_index = wettest_series.argmax(dim="time")
    maximum_date = pd.Timestamp(wettest_series.time[time_index].values)

    return wettest_member, maximum_value, maximum_date


# =============================================================================
# 10. Map-panel data loading
# =============================================================================

def open_s2s_variable(variable):
    """Open one S2S variable and convert to plotting units."""

    filename = make_s2s_file(variable)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {available}"
        )

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    elif variable == RUNOFF_VAR:
        ds[variable] = standardize_runoff_units(ds[variable])

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


def load_daily_s2s_variable(variable, target_date):
    """Load one daily S2S field for the selected member and date."""

    ds = open_s2s_variable(variable)

    try:
        da = ds[variable]
        da = select_member(da)
        da = select_date(da, target_date)

    finally:
        ds.close()

    return da


def load_map_precipitation(target_date):
    """Load daily S2S precipitation in mm/day."""

    return load_daily_s2s_variable(PRECIP_VAR, target_date)


def load_map_msl(target_date):
    """Load S2S mean sea level pressure in hPa."""

    return load_daily_s2s_variable(MSL_VAR, target_date)


# =============================================================================
# 11. Bottom runoff-panel data loading
# =============================================================================

def open_forecast_runoff_dataset():
    """Open S2S forecast runoff and convert to mm/day."""

    filename = make_s2s_file(RUNOFF_VAR)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if RUNOFF_VAR not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{RUNOFF_VAR}' not found in {filename}. "
            f"Available variables: {available}"
        )

    ds[RUNOFF_VAR] = standardize_runoff_units(ds[RUNOFF_VAR])

    return ds


def load_all_forecast_runoff_members():
    """Load S2S runoff for all forecast ensemble members."""

    ds = open_forecast_runoff_dataset()

    try:
        da = ds[RUNOFF_VAR].load()

    finally:
        ds.close()

    return da


def load_era5_runoff(years, time_start, time_end):
    """Load ERA5 runoff over the requested period."""

    filenames = [
        ERA5_RUNOFF_PATH + ERA5_RUNOFF_FILE_PATTERN.format(year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=remove_era5_ensemble_dimension_if_present,
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


def load_forecast_catchment_mean_runoff(catchment_name, init_date, plot_end):
    """Load catchment-mean S2S runoff for all ensemble members."""

    spatial_dims = ("latitude", "longitude")

    da = load_all_forecast_runoff_members()

    check_dims(
        da=da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast runoff",
    )

    weights = load_weights(
        filename=make_era5_weights_file(catchment_name, GRID),
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_weighted_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
        output_name="catchment_mean_runoff",
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=init_date,
        end_date=plot_end,
    )

    return da_mean


def load_era5_catchment_mean_runoff(
    catchment_name,
    years,
    load_start,
    load_end,
    plot_start,
    plot_end,
):
    """Load catchment-mean ERA5 runoff for the bottom panel."""

    spatial_dims = ("latitude", "longitude")

    da = load_era5_runoff(
        years=years,
        time_start=load_start,
        time_end=load_end,
    )

    weights = load_weights(
        filename=make_era5_weights_file(catchment_name, GRID),
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_weighted_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
        output_name="catchment_mean_runoff",
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=plot_start,
        end_date=plot_end,
    )

    return da_mean


# =============================================================================
# 12. Catchment boundary
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

    if not catchment_path.exists():
        raise FileNotFoundError(f"Catchment file not found: {catchment_path}")

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
# 13. Figure setup
# =============================================================================

def make_figure_axes():
    """
    Create the full combined figure.

    Layout:
    - Row 0: top precipitation time-series panel
    - Row 1: map panels a and b
    - Row 2: map panels c and d
    - Row 3: bottom runoff time-series panel e
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
        height_ratios=[0.45, 1.0, 1.0, 0.45],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    top_ts_ax = fig.add_subplot(gs[0, 0:2])

    map_axes = np.empty((2, 2), dtype=object)
    map_axes[0, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    map_axes[0, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)
    map_axes[1, 0] = fig.add_subplot(gs[2, 0], projection=proj_map)
    map_axes[1, 1] = fig.add_subplot(gs[2, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[1:3, 2])

    runoff_ts_ax = fig.add_subplot(gs[3, 0:2])

    for ax in map_axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, top_ts_ax, map_axes, runoff_ts_ax, cbar_ax, proj_map, proj_data


def get_map_axes(map_axes):
    """Return map panels as a flat list."""

    return list(map_axes.flat)


def align_axis_to_map_panels(fig, map_axes, ax_to_align):
    """Align a time-series axis with the combined width of the map panels."""

    fig.canvas.draw()

    left = min(
        map_axes[0, 0].get_position().x0,
        map_axes[1, 0].get_position().x0,
    )

    right = max(
        map_axes[0, 1].get_position().x1,
        map_axes[1, 1].get_position().x1,
    )

    pos = ax_to_align.get_position()

    ax_to_align.set_position(
        [
            left,
            pos.y0,
            right - left,
            pos.height,
        ]
    )


# =============================================================================
# 14. Map plotting functions
# =============================================================================

def plot_precipitation_map(ax, da_precip, proj_data):
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

    mesh = ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
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


def plot_event_panel(ax, target_date, catchment_boundary, proj_data):
    """Plot one daily map panel."""

    da_precip = load_map_precipitation(target_date)
    da_msl = load_map_msl(target_date)

    mesh = plot_precipitation_map(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


# =============================================================================
# 15. Top precipitation time-series plotting
# =============================================================================

def plot_top_precipitation_timeseries(ax, catchment_label):
    """
    Plot the precipitation time-series panel from the first script.

    This panel compares:
    - S2S forecast ensemble
    - wettest forecast member
    - ERA5
    - seNorge
    """

    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        years,
    ) = get_plot_period(
        forecast_date=FORECAST_DATE,
        n_days_before=PRECIP_TS_N_DAYS_BEFORE,
        m_days_lead=PRECIP_TS_M_DAYS_LEAD,
        x_days=PRECIP_TS_X_DAYS,
    )

    dates = {
        "init_date": init_date,
        "plot_start": plot_start,
        "plot_end": plot_end,
        "load_start": load_start,
        "load_end": load_end,
        "years": years,
    }

    forecast = process_precip_ts_forecast(dates)
    era5 = process_precip_ts_era5(dates)
    senorge = process_precip_ts_senorge(dates)

    era5, senorge = keep_only_common_observation_dates(
        era5=era5,
        senorge=senorge,
    )

    forecast = keep_forecast_dates_available_in_observations(
        forecast=forecast,
        observation_dates=era5.time.values,
    )

    wettest_member, maximum_precipitation, maximum_date = find_wettest_ensemble_member(
        forecast=forecast,
    )

    print(
        "\nTop precipitation panel:"
        f"\nWettest ensemble member: {wettest_member}"
        f"\nMaximum precipitation: {maximum_precipitation:.1f} mm"
        f"\nDate: {maximum_date:%Y-%m-%d}\n"
    )

    member_name = get_member_coord_name(forecast)

    # Dummy line for the ensemble legend entry.
    ax.plot(
        [],
        [],
        color=PRECIP_TS_FORECAST_COLOR,
        linewidth=PRECIP_TS_FORECAST_LINEWIDTH,
        alpha=PRECIP_TS_FORECAST_ALPHA,
        label="Forecast ensemble",
    )

    # Plot all non-highlighted forecast members.
    for member in forecast[member_name].values:
        if member == wettest_member:
            continue

        ax.plot(
            forecast["time"],
            forecast.sel({member_name: member}),
            color=PRECIP_TS_FORECAST_COLOR,
            linewidth=PRECIP_TS_FORECAST_LINEWIDTH,
            alpha=PRECIP_TS_FORECAST_ALPHA,
        )

    # Plot the wettest member last so it appears on top.
    ax.plot(
        forecast["time"],
        forecast.sel({member_name: wettest_member}),
        color=PRECIP_TS_EXTREME_MEMBER_COLOR,
        linewidth=PRECIP_TS_OBS_LINEWIDTH,
        zorder=10,
        label="Counterfactual Storm Hans",
    )

    ax.plot(
        era5["time"],
        era5,
        color=PRECIP_TS_ERA5_COLOR,
        linewidth=PRECIP_TS_OBS_LINEWIDTH,
        label="ERA5 Storm Hans",
    )

    ax.plot(
        senorge["time"],
        senorge,
        color=PRECIP_TS_SENORGE_COLOR,
        linewidth=PRECIP_TS_OBS_LINEWIDTH,
        label="seNorge Storm Hans",
    )

    ax.axvline(
        init_date,
        color=INITIALIZATION_LINE_COLOR,
        linewidth=INITIALIZATION_LINE_WIDTH,
        linestyle=INITIALIZATION_LINE_STYLE,
        label="Forecast initialization",
    )

    ax.set_title(
        f"{catchment_label}, {PRECIP_TS_X_DAYS}-day accumulated precipitation",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ax.set_ylabel("Precipitation (mm)", fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)

    ax.set_xlim(plot_start, plot_end)
    ax.margins(x=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=DATE_TICK_INTERVAL_DAYS))

    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    plt.setp(
        ax.get_xticklabels(),
        rotation=DATE_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )


# =============================================================================
# 16. Bottom runoff time-series plotting
# =============================================================================

def plot_bottom_runoff_timeseries(ax, catchment_name):
    """
    Plot forecast ensemble and ERA5 catchment-mean runoff in panel e).

    This keeps the same logic as panel e in the second script.
    """

    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        years,
    ) = get_plot_period(
        forecast_date=FORECAST_DATE,
        n_days_before=RUNOFF_TS_N_DAYS_BEFORE,
        m_days_lead=RUNOFF_TS_M_DAYS_LEAD,
        x_days=1,
    )

    forecast = load_forecast_catchment_mean_runoff(
        catchment_name=catchment_name,
        init_date=init_date,
        plot_end=plot_end,
    )

    era5 = load_era5_catchment_mean_runoff(
        catchment_name=catchment_name,
        years=years,
        load_start=load_start,
        load_end=load_end,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    time_name = get_time_coord_name(forecast)
    member_name = get_member_coord_name(forecast)

    # Dummy line for the ensemble legend entry.
    ax.plot(
        [],
        [],
        color=RUNOFF_ENSEMBLE_COLOR,
        linewidth=RUNOFF_ENSEMBLE_LINEWIDTH,
        alpha=RUNOFF_ENSEMBLE_ALPHA,
        label="Forecast ensemble",
    )

    # Plot all non-selected forecast members.
    for member in forecast[member_name].values:
        if member == ENSEMBLE_MEMBER:
            continue

        ax.plot(
            forecast[time_name].values,
            forecast.sel({member_name: member}).values,
            color=RUNOFF_ENSEMBLE_COLOR,
            linewidth=RUNOFF_ENSEMBLE_LINEWIDTH,
            alpha=RUNOFF_ENSEMBLE_ALPHA,
        )

    selected = forecast.sel({member_name: ENSEMBLE_MEMBER})

    ax.plot(
        selected[time_name].values,
        selected.values,
        color=RUNOFF_SELECTED_MEMBER_COLOR,
        linewidth=RUNOFF_SELECTED_MEMBER_LINEWIDTH,
        label=f"Selected member {ENSEMBLE_MEMBER}",
        zorder=10,
    )

    # Keep the original second-script behavior:
    # add dots for the four map-panel dates.
    for date in EVENT_DATES:
        value = selected.sel(
            {time_name: np.datetime64(date)},
            method="nearest",
        )

        ax.scatter(
            value[time_name].values,
            value.values,
            color=RUNOFF_SELECTED_MEMBER_COLOR,
            s=EVENT_MARKER_SIZE,
            zorder=11,
        )

    era5_time_name = get_time_coord_name(era5)

    ax.plot(
        era5[era5_time_name].values,
        era5.values,
        color=RUNOFF_ERA5_COLOR,
        linewidth=RUNOFF_ERA5_LINEWIDTH,
        label="ERA5",
        zorder=9,
    )

    ax.axvline(
        init_date,
        color=INITIALIZATION_LINE_COLOR,
        linewidth=INITIALIZATION_LINE_WIDTH,
        linestyle=INITIALIZATION_LINE_STYLE,
        label="Forecast initialization",
    )

    ax.set_title(
        "e) Drammen catchment mean runoff",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ax.set_ylabel("mm/day", fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)

    ax.tick_params(labelsize=TICK_LABELSIZE)

    ax.set_xlim(plot_start, plot_end)
    ax.margins(x=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=DATE_TICK_INTERVAL_DAYS))

    plt.setp(
        ax.get_xticklabels(),
        rotation=DATE_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )


# =============================================================================
# 17. Figure finishing
# =============================================================================

def add_panel_titles(map_axes):
    """Add panel labels and event-relative dates to the map panels."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, lag, date in zip(
        get_map_axes(map_axes),
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
    """Add precipitation colorbar beside the map panels."""

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


def add_map_legend(map_axes, catchment_label):
    """Add map legend inside the upper-left map panel."""

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

    legend = map_axes[0, 0].legend(
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


def finalize_figure(
    fig,
    top_ts_ax,
    map_axes,
    runoff_ts_ax,
    cbar_ax,
    mesh,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legends, layout, save, and show."""

    add_panel_titles(map_axes)
    add_colorbar(fig, mesh, cbar_ax)
    add_map_legend(map_axes, catchment_label)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.065,
        top=0.965,
    )

    align_axis_to_map_panels(fig, map_axes, top_ts_ax)
    align_axis_to_map_panels(fig, map_axes, runoff_ts_ax)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 18. Main workflow
# =============================================================================

def main():
    """Run the full combined plotting workflow."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    catchment_boundary = load_catchment_outer_boundary(
        filename=catchment["filename"],
        base_dir=PATH_CATCHMENT,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    (
        fig,
        top_ts_ax,
        map_axes,
        runoff_ts_ax,
        cbar_ax,
        proj_map,
        proj_data,
    ) = make_figure_axes()

    # Top precipitation time-series panel from the first script.
    plot_top_precipitation_timeseries(
        ax=top_ts_ax,
        catchment_label=catchment["label"],
    )

    # Four map panels from the second script.
    mesh = None

    for ax, target_date in zip(get_map_axes(map_axes), EVENT_DATES):
        mesh = plot_event_panel(
            ax=ax,
            target_date=target_date,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

    # Bottom runoff time-series panel from the second script.
    plot_bottom_runoff_timeseries(
        ax=runoff_ts_ax,
        catchment_name=CATCHMENT_NAME,
    )

    finalize_figure(
        fig=fig,
        top_ts_ax=top_ts_ax,
        map_axes=map_axes,
        runoff_ts_ax=runoff_ts_ax,
        cbar_ax=cbar_ax,
        mesh=mesh,
        catchment_label=catchment["label"],
        savepath=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()
