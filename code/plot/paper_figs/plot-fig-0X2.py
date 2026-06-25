"""
Plot accumulated catchment-mean precipitation for one forecast event.

The figure compares:
1. S2S forecast ensemble members from the forecast initialization date onward.
2. ERA5 over the same catchment, including days before initialization.
3. SeNorge over the same catchment, including days before initialization.

Important time-handling note
----------------------------
SeNorge timestamps are shifted one day earlier before daily rounding and
accumulation. This preserves the convention used in the original script.

Script layout
-------------
1. Import statements
2. User-defined input parameters
3. Functions that factor out tasks
4. Main function that calls clearly named functions
"""


# =============================================================================
# 1. Import statements
# =============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# 2. User-defined input parameters
# =============================================================================

# -----------------------------------------------------------------------------
# Event and catchment settings
# -----------------------------------------------------------------------------

FORECAST_DATE = "2023-08-05"
CATCHMENT = "regine_drammen"
X_DAYS = 2

# -----------------------------------------------------------------------------
# Plot-window settings
# -----------------------------------------------------------------------------
# The plot will show:
# - N_DAYS_BEFORE days before the forecast initialization date.
# - M_DAYS_LEAD days after the forecast initialization date.

N_DAYS_BEFORE = 2
M_DAYS_LEAD = 6

# -----------------------------------------------------------------------------
# Forecast data settings
# -----------------------------------------------------------------------------

FORECAST_VARIABLE = "tp24"
FORECAST_GRID = "0.25x0.25"

FORECAST_FILE = (
    config.dirs["s2s_forecast_daily"]
    + FORECAST_VARIABLE
    + "/"
    + f"{FORECAST_VARIABLE}_{FORECAST_GRID}_{FORECAST_DATE}.nc"
)

FORECAST_WEIGHTS_FILE = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT}_era5_{FORECAST_GRID}.nc"
)

# -----------------------------------------------------------------------------
# ERA5 data settings
# -----------------------------------------------------------------------------

ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.25x0.25"
ERA5_DOMAIN = "norway"

ERA5_PATH = config.dirs["era5_continuous_daily"] + ERA5_VARIABLE + "/"
ERA5_FILE_PATTERN = f"{ERA5_VARIABLE}_{ERA5_GRID}" + "_{year}.nc"

ERA5_WEIGHTS_FILE = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT}_era5_{ERA5_GRID}.nc"
)

# -----------------------------------------------------------------------------
# SeNorge data settings
# -----------------------------------------------------------------------------

SENORGE_VARIABLE = "rr"

SENORGE_PATH = config.dirs["senorge_continuous_daily"] + SENORGE_VARIABLE + "/"
SENORGE_FILE_PATTERN = SENORGE_VARIABLE + "_{year}.nc"

SENORGE_WEIGHTS_FILE = (
    config.dirs["nve"]
    + f"weights_catchment_{CATCHMENT}_senorge.nc"
)

# -----------------------------------------------------------------------------
# Plot text
# -----------------------------------------------------------------------------
if CATCHMENT == 'regine_glomma':
    PLOT_TITLE = f"Glomma catchment, {X_DAYS}-day accumulated precipitation"
elif CATCHMENT == 'regine_drammen':
    PLOT_TITLE = f" Drammen catchment, {X_DAYS}-day accumulated precipitation"
X_LABEL = "Date"
Y_LABEL = "Precipitation [mm]"

FORECAST_ENSEMBLE_LABEL = "Forecast ensemble"
EXTREME_MEMBER_LABEL = "Counterfactual Storm Hans"
ERA5_LABEL = "ERA5 Storm Hans"
SENORGE_LABEL = "SeNorge Storm Hans"
INITIALIZATION_LABEL = "Forecast initialization"

# -----------------------------------------------------------------------------
# Plot appearance
# -----------------------------------------------------------------------------

FIGURE_SIZE = (11, 5)

FORECAST_COLOR = "0.7"
FORECAST_LINE_WIDTH = 0.8
FORECAST_ALPHA = 0.35

EXTREME_MEMBER_COLOR = "tab:green"
ERA5_COLOR = "tab:blue"
SENORGE_COLOR = "tab:red"
OBS_LINE_WIDTH = 2.5

INITIALIZATION_LINE_COLOR = "k"
INITIALIZATION_LINE_WIDTH = 1.2
INITIALIZATION_LINE_STYLE = "--"

LEGEND_LOCATION = "upper left"
SHOW_LEGEND_FRAME = False

# -----------------------------------------------------------------------------
# Font sizes
# -----------------------------------------------------------------------------
# Tune these values when preparing figures for papers, presentations, or posters.

TITLE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10

# -----------------------------------------------------------------------------
# X-axis date formatting
# -----------------------------------------------------------------------------

DATE_TICK_FORMAT = "%d %b"
DATE_TICK_INTERVAL_DAYS = 1
DATE_TICK_ROTATION = 30

# -----------------------------------------------------------------------------
# Output settings
# -----------------------------------------------------------------------------

SAVE_FIGURE = True
FIGURE_OUTPUT_DIR = config.dirs["fig"]
FIGURE_FILENAME = f"{FIGURE_OUTPUT_DIR}fig-0X2.png"
FIGURE_DPI = 300


# =============================================================================
# 3. Functions that factor out tasks
# =============================================================================

# -----------------------------------------------------------------------------
# Basic validation and unit handling
# -----------------------------------------------------------------------------

def convert_precipitation_to_mm(data_array, variable_name):
    """
    Convert precipitation to millimetres when needed.

    Parameters
    ----------
    data_array : xarray.DataArray
        Precipitation field.
    variable_name : str
        Name of the variable, for example "tp24" or "rr".

    Returns
    -------
    xarray.DataArray
        Precipitation in mm.
    """

    units = str(data_array.attrs.get("units", "")).strip().lower()

    if variable_name == "tp24" or units in {"m", "meter", "metre"}:
        data_array = data_array * 1000.0
        data_array.attrs["units"] = "mm"

    elif units in {"kg/m^2", "kg/m2", "kg m-2"}:
        data_array.attrs["units"] = "mm"

    return data_array


def require_dimensions(data_array, required_dimensions, data_name):
    """
    Stop with a clear error if a DataArray lacks required dimensions.
    """

    missing_dimensions = [
        dimension
        for dimension in required_dimensions
        if dimension not in data_array.dims
    ]

    if missing_dimensions:
        raise ValueError(
            f"{data_name} is missing dimensions {missing_dimensions}. "
            f"Found dimensions: {data_array.dims}"
        )


# -----------------------------------------------------------------------------
# Date handling
# -----------------------------------------------------------------------------

def define_plot_and_loading_periods(
    forecast_date,
    n_days_before,
    m_days_lead,
    x_days,
):
    """
    Define dates needed for plotting and data loading.

    The loading period is slightly wider than the plot period because an X-day
    accumulation needs data before the first plotted date.
    """

    initialization_date = pd.to_datetime(forecast_date)

    plot_start_date = initialization_date - pd.Timedelta(days=n_days_before)
    plot_end_date = initialization_date + pd.Timedelta(days=m_days_lead)

    loading_start_date = plot_start_date - pd.Timedelta(days=x_days + 1)
    loading_end_date = plot_end_date + pd.Timedelta(days=x_days + 1)

    years_to_load = np.arange(
        loading_start_date.year,
        loading_end_date.year + 1,
    )

    return {
        "initialization_date": initialization_date,
        "plot_start_date": plot_start_date,
        "plot_end_date": plot_end_date,
        "loading_start_date": loading_start_date,
        "loading_end_date": loading_end_date,
        "years_to_load": years_to_load,
    }


def round_time_coordinate_to_nearest_day(data_array):
    """
    Round time coordinates to the nearest calendar day.
    """

    rounded_time = pd.to_datetime(data_array.time.values).round("D")

    return data_array.assign_coords(time=rounded_time)


def shift_time_coordinate_back_one_day(data_array):
    """
    Shift all timestamps one day earlier.

    This is used for SeNorge before accumulation.
    """

    shifted_time = pd.to_datetime(data_array.time.values) - pd.Timedelta(days=1)

    return data_array.assign_coords(time=shifted_time)


def select_time_period(data_array, start_date, end_date):
    """
    Select data between two dates.
    """

    return data_array.sel(time=slice(start_date, end_date))


# -----------------------------------------------------------------------------
# Weight handling
# -----------------------------------------------------------------------------

def load_catchment_weights(weights_file, spatial_dimensions):
    """
    Load catchment weights from file.

    The expected variable name is "catchment_weight".
    """

    with xr.open_dataset(weights_file) as dataset:
        if "catchment_weight" not in dataset:
            raise KeyError(
                f"'catchment_weight' not found in {weights_file}. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        weights = dataset["catchment_weight"].astype("float32").load()

    weights.name = "catchment_weight"

    require_dimensions(
        data_array=weights,
        required_dimensions=spatial_dimensions,
        data_name="Catchment weights",
    )

    return weights


def align_weights_to_data_grid(data_array, weights):
    """
    Align 2D catchment weights to the spatial grid of the data.

    The data may have extra non-spatial dimensions, such as:
        - time
        - number

    The weights should only have spatial dimensions, for example:
        - latitude, longitude
        - Y, X
    """

    spatial_dimensions = tuple(weights.dims)

    # Select one slice from all non-spatial dimensions.
    # This leaves only the spatial grid.
    grid_template = data_array

    for dim in data_array.dims:
        if dim not in spatial_dimensions:
            grid_template = grid_template.isel({dim: 0}, drop=True)

    # Align weights to the spatial grid only.
    weights_on_grid = weights.reindex_like(grid_template)

    if weights_on_grid.shape != grid_template.shape:
        raise ValueError(
            f"Weights shape {weights_on_grid.shape} does not match "
            f"spatial data grid shape {grid_template.shape} after alignment."
        )

    if np.isfinite(weights_on_grid).sum().item() == 0:
        raise ValueError(
            "Weights were aligned to the data grid, but all values are NaN. "
            "This usually means the weight coordinates and data coordinates do not overlap."
        )

    return weights_on_grid


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def remove_era5_ensemble_dimension_if_present(dataset):
    """
    Remove the ERA5 'number' variable if it exists.
    """

    return dataset.drop_vars("number", errors="ignore")


def load_forecast_precipitation(forecast_file, variable_name):
    """
    Load S2S forecast precipitation.

    Expected dimensions are:
    - time
    - number
    - latitude
    - longitude
    """

    with xr.open_dataset(forecast_file) as dataset:
        if variable_name not in dataset:
            raise KeyError(
                f"'{variable_name}' not found in {forecast_file}. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        forecast = dataset[variable_name].load()

    forecast = convert_precipitation_to_mm(
        data_array=forecast,
        variable_name=variable_name,
    )

    require_dimensions(
        data_array=forecast,
        required_dimensions=("time", "number", "latitude", "longitude"),
        data_name="Forecast",
    )

    return forecast


def load_era5_precipitation(years, loading_start_date, loading_end_date):
    """
    Load ERA5 precipitation for the requested years and time period.
    """

    era5_files = [
        ERA5_PATH + ERA5_FILE_PATTERN.format(year=int(year))
        for year in years
    ]

    dataset = xr.open_mfdataset(
        era5_files,
        preprocess=remove_era5_ensemble_dimension_if_present,
        combine="by_coords",
    )

    if ERA5_DOMAIN is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(ERA5_DOMAIN)
        dataset = dataset.sel(latitude=domain_lats, longitude=domain_lons)

    if ERA5_VARIABLE not in dataset:
        raise KeyError(
            f"'{ERA5_VARIABLE}' not found in ERA5 files. "
            f"Available variables: {list(dataset.data_vars)}"
        )

    era5 = dataset[ERA5_VARIABLE].sel(
        time=slice(loading_start_date, loading_end_date)
    )

    era5 = convert_precipitation_to_mm(
        data_array=era5,
        variable_name=ERA5_VARIABLE,
    )

    require_dimensions(
        data_array=era5,
        required_dimensions=("time", "latitude", "longitude"),
        data_name="ERA5",
    )

    return era5


def load_senorge_precipitation(years, loading_start_date, loading_end_date):
    """
    Load SeNorge precipitation year by year.

    Loading year by year avoids assuming that all years can be opened as one
    multi-file dataset.
    """

    yearly_data = []

    for year in years:
        senorge_file = SENORGE_PATH + SENORGE_FILE_PATTERN.format(year=int(year))

        with xr.open_dataset(senorge_file) as dataset:
            dataset = xr.decode_cf(dataset)

            if SENORGE_VARIABLE not in dataset:
                raise KeyError(
                    f"'{SENORGE_VARIABLE}' not found in {senorge_file}. "
                    f"Available variables: {list(dataset.data_vars)}"
                )

            senorge_one_year = dataset[SENORGE_VARIABLE].sel(
                time=slice(loading_start_date, loading_end_date)
            )

            fill_value = senorge_one_year.attrs.get("_FillValue")
            if fill_value is not None:
                senorge_one_year = senorge_one_year.where(
                    senorge_one_year != fill_value
                )

            senorge_one_year = convert_precipitation_to_mm(
                data_array=senorge_one_year,
                variable_name=SENORGE_VARIABLE,
            )

            require_dimensions(
                data_array=senorge_one_year,
                required_dimensions=("time", "Y", "X"),
                data_name="SeNorge",
            )

            yearly_data.append(senorge_one_year.load())

    senorge = xr.concat(yearly_data, dim="time").sortby("time")

    return senorge


# -----------------------------------------------------------------------------
# Catchment averaging and accumulation
# -----------------------------------------------------------------------------

def calculate_catchment_weighted_mean(data_array, weights, spatial_dimensions):
    """
    Calculate catchment-weighted spatial mean precipitation.

    Formula
    -------
    catchment mean = sum(precipitation * weight) / sum(weight)
    """

    weights_on_data_grid = align_weights_to_data_grid(
        data_array=data_array,
        weights=weights,
    )

    valid_points = (
        xr.ufuncs.isfinite(data_array)
        & xr.ufuncs.isfinite(weights_on_data_grid)
        & (weights_on_data_grid > 0)
    )

    weighted_sum = (
        data_array.where(valid_points)
        * weights_on_data_grid.where(valid_points)
    ).sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    weight_sum = weights_on_data_grid.where(valid_points).sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    catchment_mean = weighted_sum / weight_sum
    catchment_mean.name = "catchment_mean_precipitation"
    catchment_mean.attrs["units"] = data_array.attrs.get("units", "mm")

    return catchment_mean


def calculate_trailing_xday_accumulation(data_array, x_days):
    """
    Calculate trailing X-day accumulated precipitation.
    """

    accumulated = (
        data_array
        .rolling(time=x_days, min_periods=x_days)
        .sum()
        .dropna("time", how="any")
    )

    accumulated.name = f"{x_days}day_accumulated_precipitation"
    accumulated.attrs["units"] = "mm"

    return accumulated


# -----------------------------------------------------------------------------
# Dataset-specific processing pipelines
# -----------------------------------------------------------------------------

def process_forecast_for_plotting(dates):
    """
    Load, catchment-average, accumulate, round, and subset forecast data.
    """

    forecast_weights = load_catchment_weights(
        weights_file=FORECAST_WEIGHTS_FILE,
        spatial_dimensions=("latitude", "longitude"),
    )

    forecast = load_forecast_precipitation(
        forecast_file=FORECAST_FILE,
        variable_name=FORECAST_VARIABLE,
    )

    forecast_mean = calculate_catchment_weighted_mean(
        data_array=forecast,
        weights=forecast_weights,
        spatial_dimensions=("latitude", "longitude"),
    )

    forecast_accumulated = calculate_trailing_xday_accumulation(
        data_array=forecast_mean,
        x_days=X_DAYS,
    )

    forecast_accumulated = round_time_coordinate_to_nearest_day(
        forecast_accumulated
    )

    # Forecasts are only plotted from initialization onward.
    forecast_accumulated = select_time_period(
        data_array=forecast_accumulated,
        start_date=dates["initialization_date"],
        end_date=dates["plot_end_date"],
    )

    return forecast_accumulated


def process_era5_for_plotting(dates):
    """
    Load, catchment-average, accumulate, round, and subset ERA5 data.
    """

    era5_weights = load_catchment_weights(
        weights_file=ERA5_WEIGHTS_FILE,
        spatial_dimensions=("latitude", "longitude"),
    )

    era5 = load_era5_precipitation(
        years=dates["years_to_load"],
        loading_start_date=dates["loading_start_date"],
        loading_end_date=dates["loading_end_date"],
    )

    era5_mean = calculate_catchment_weighted_mean(
        data_array=era5,
        weights=era5_weights,
        spatial_dimensions=("latitude", "longitude"),
    )

    era5_accumulated = calculate_trailing_xday_accumulation(
        data_array=era5_mean,
        x_days=X_DAYS,
    )

    era5_accumulated = round_time_coordinate_to_nearest_day(era5_accumulated)

    era5_accumulated = select_time_period(
        data_array=era5_accumulated,
        start_date=dates["plot_start_date"],
        end_date=dates["plot_end_date"],
    )

    return era5_accumulated


def process_senorge_for_plotting(dates):
    """
    Load, shift time, catchment-average, accumulate, round, and subset SeNorge.
    """

    senorge_weights = load_catchment_weights(
        weights_file=SENORGE_WEIGHTS_FILE,
        spatial_dimensions=("Y", "X"),
    )

    senorge = load_senorge_precipitation(
        years=dates["years_to_load"],
        loading_start_date=dates["loading_start_date"],
        loading_end_date=dates["loading_end_date"],
    )

    senorge = shift_time_coordinate_back_one_day(senorge)

    senorge_mean = calculate_catchment_weighted_mean(
        data_array=senorge,
        weights=senorge_weights,
        spatial_dimensions=("Y", "X"),
    )

    senorge_accumulated = calculate_trailing_xday_accumulation(
        data_array=senorge_mean,
        x_days=X_DAYS,
    )

    senorge_accumulated = round_time_coordinate_to_nearest_day(senorge_accumulated)

    senorge_accumulated = select_time_period(
        data_array=senorge_accumulated,
        start_date=dates["plot_start_date"],
        end_date=dates["plot_end_date"],
    )

    return senorge_accumulated


# -----------------------------------------------------------------------------
# Matching dates across datasets
# -----------------------------------------------------------------------------

def keep_only_common_observation_dates(era5, senorge):
    """
    Keep only dates available in both ERA5 and SeNorge.
    """

    common_dates = np.intersect1d(era5.time.values, senorge.time.values)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between ERA5 and SeNorge.")

    era5_common = era5.sel(time=common_dates)
    senorge_common = senorge.sel(time=common_dates)

    print(
        f"\nObservations plotted from {common_dates[0]} "
        f"to {common_dates[-1]}.\n"
    )

    return era5_common, senorge_common


def keep_forecast_dates_available_in_observations(forecast, observation_dates):
    """
    Keep forecast dates that are also available in the observations.
    """

    common_dates = np.intersect1d(forecast.time.values, observation_dates)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between forecast and observations.")

    forecast_common = forecast.sel(time=common_dates)

    print(
        f"Forecast plotted from {common_dates[0]} "
        f"to {common_dates[-1]}.\n"
    )

    return forecast_common


# -----------------------------------------------------------------------------
# Extreme ensemble member
# -----------------------------------------------------------------------------

def find_extreme_ensemble_member(forecast, mode="max"):
    """
    Find the ensemble member with the largest or smallest value.

    Parameters
    ----------
    forecast : xarray.DataArray
        Forecast data with dimensions (time, number).
    mode : {"max", "min"}
        Use "max" to find the wettest member.
        Use "min" to find the driest member.

    Returns
    -------
    member : int
        Ensemble member number.
    value : float
        Extreme precipitation value.
    time : pandas.Timestamp
        Date when the extreme value occurs.
    """

    if mode not in {"max", "min"}:
        raise ValueError(f"mode must be 'max' or 'min', got '{mode}'.")

    if mode == "max":
        extreme_by_member = forecast.max(dim="time")
        member = int(extreme_by_member.idxmax(dim="number"))
        value = float(extreme_by_member.max())

        member_series = forecast.sel(number=member)
        time_index = member_series.argmax(dim="time")

    else:
        extreme_by_member = forecast.min(dim="time")
        member = int(extreme_by_member.idxmin(dim="number"))
        value = float(extreme_by_member.min())

        member_series = forecast.sel(number=member)
        time_index = member_series.argmin(dim="time")

    time = pd.Timestamp(member_series.time[time_index].values)

    return member, value, time


def print_extreme_member_summary(member, value, time):
    """
    Print a simple summary of the highlighted ensemble member.
    """

    print(
        f"Wettest ensemble member: {member}\n"
        f"Maximum precipitation: {value:.1f} mm\n"
        f"Date: {time:%Y-%m-%d}"
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_forecast_era5_senorge_comparison(
    forecast,
    era5,
    senorge,
    initialization_date,
    plot_start_date,
    plot_end_date,
):
    """
    Plot forecast ensemble members, ERA5, and SeNorge.
    """

    wettest_member, _, _ = find_extreme_ensemble_member(
        forecast=forecast,
        mode="max",
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    add_observations_to_axis(
        ax=ax,
        era5=era5,
        senorge=senorge,
    )


    add_initialization_line_to_axis(
        ax=ax,
        initialization_date=initialization_date,
    )
    add_forecast_ensemble_to_axis(
        ax=ax,
        forecast=forecast,
        highlighted_member=wettest_member,
    )

    format_precipitation_axis(
        ax=ax,
        plot_start_date=plot_start_date,
        plot_end_date=plot_end_date,
    )

    save_or_show_figure(fig)


def add_forecast_ensemble_to_axis(ax, forecast, highlighted_member):
    """
    Add all forecast ensemble members to a matplotlib axis.

    The highlighted member is plotted last so it appears on top.
    """

    # Dummy line for the ensemble legend entry.
    ax.plot(
        [],
        [],
        color=FORECAST_COLOR,
        linewidth=FORECAST_LINE_WIDTH,
        alpha=FORECAST_ALPHA,
        label=FORECAST_ENSEMBLE_LABEL,
    )

    for member in forecast["number"].values:
        if member == highlighted_member:
            continue

        ax.plot(
            forecast["time"],
            forecast.sel(number=member),
            color=FORECAST_COLOR,
            linewidth=FORECAST_LINE_WIDTH,
            alpha=FORECAST_ALPHA,
        )

    ax.plot(
        forecast["time"],
        forecast.sel(number=highlighted_member),
        color=EXTREME_MEMBER_COLOR,
        linewidth=OBS_LINE_WIDTH,
        zorder=10,
        label=EXTREME_MEMBER_LABEL,
    )


def add_observations_to_axis(ax, era5, senorge):
    """
    Add ERA5 and SeNorge time series to a matplotlib axis.
    """

    ax.plot(
        era5["time"],
        era5,
        color=ERA5_COLOR,
        linewidth=OBS_LINE_WIDTH,
        label=ERA5_LABEL,
    )

    ax.plot(
        senorge["time"],
        senorge,
        color=SENORGE_COLOR,
        linewidth=OBS_LINE_WIDTH,
        label=SENORGE_LABEL,
    )


def add_initialization_line_to_axis(ax, initialization_date):
    """
    Add a vertical line marking forecast initialization.
    """

    ax.axvline(
        initialization_date,
        color=INITIALIZATION_LINE_COLOR,
        linewidth=INITIALIZATION_LINE_WIDTH,
        linestyle=INITIALIZATION_LINE_STYLE,
        label=INITIALIZATION_LABEL,
    )


def format_precipitation_axis(ax, plot_start_date, plot_end_date):
    """
    Apply titles, labels, ticks, date formatting, and legend styling.
    """

    ax.set_title(PLOT_TITLE, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_xlim(plot_start_date, plot_end_date)
    ax.margins(x=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_TICK_FORMAT))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=DATE_TICK_INTERVAL_DAYS))

    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)

    plt.setp(
        ax.get_xticklabels(),
        rotation=DATE_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )

    ax.legend(
        loc=LEGEND_LOCATION,
        frameon=SHOW_LEGEND_FRAME,
        fontsize=LEGEND_FONT_SIZE,
    )


def save_or_show_figure(fig):
    """
    Save the figure if requested, then display it.
    """

    fig.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(
            FIGURE_FILENAME,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

    plt.show()


# =============================================================================
# 4. Main function
# =============================================================================

def main():
    """
    Run the complete plotting workflow.

    Workflow
    --------
    1. Define the plot and loading dates.
    2. Process forecast data.
    3. Process ERA5 data.
    4. Process SeNorge data.
    5. Match dates across datasets.
    6. Identify the wettest ensemble member.
    7. Plot the comparison.
    """

    dates = define_plot_and_loading_periods(
        forecast_date=FORECAST_DATE,
        n_days_before=N_DAYS_BEFORE,
        m_days_lead=M_DAYS_LEAD,
        x_days=X_DAYS,
    )

    forecast = process_forecast_for_plotting(dates)
    era5 = process_era5_for_plotting(dates)
    senorge = process_senorge_for_plotting(dates)

    era5, senorge = keep_only_common_observation_dates(
        era5=era5,
        senorge=senorge,
    )

    forecast = keep_forecast_dates_available_in_observations(
        forecast=forecast,
        observation_dates=era5.time.values,
    )

    wettest_member, maximum_precipitation, date_of_maximum = (
        find_extreme_ensemble_member(
            forecast=forecast,
            mode="max",
        )
    )

    print_extreme_member_summary(
        member=wettest_member,
        value=maximum_precipitation,
        time=date_of_maximum,
    )

    plot_forecast_era5_senorge_comparison(
        forecast=forecast,
        era5=era5,
        senorge=senorge,
        initialization_date=dates["initialization_date"],
        plot_start_date=dates["plot_start_date"],
        plot_end_date=dates["plot_end_date"],
    )


if __name__ == "__main__":
    main()
