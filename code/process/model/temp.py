"""
Calculate ensemble-member independence for ECMWF S2S precipitation.

For each forecast lead time, the script:

1. Reads all available operational forecasts and corresponding hindcasts.
2. Calculates catchment-weighted daily precipitation.
3. Calculates trailing N-day accumulated precipitation.
4. Constructs one time series for each ensemble member at each lead time.
5. Calculates the Spearman rank correlation for every unique pair of
   ensemble members.
6. Saves the individual correlations and summary statistics to NetCDF.

Forecasts and hindcasts are handled as separate ensemble systems:

    Forecast:
        Each forecast date is one initialization with 51 members.

    Hindcast:
        Each hdate is one initialization with 11 members.

The pairwise correlations from the forecast and hindcast systems are also
pooled to provide combined summary statistics. The raw forecast and hindcast
correlations remain identifiable in the output.

Input file dimensions
---------------------
Forecast:
    time, number, latitude, longitude

Hindcast:
    time, number, hdate, latitude, longitude

Output
------
The NetCDF file contains:

1. Individual pairwise correlations:
       spearman_rho(pair)
       lead_time(pair)
       valid_month(pair)
       member_1(pair)
       member_2(pair)
       model_type(pair)
       n_samples(pair)

2. Summary statistics:
       mean_rho(summary)
       median_rho(summary)
       q05_rho(summary)
       q25_rho(summary)
       q75_rho(summary)
       q95_rho(summary)
       minimum_rho(summary)
       maximum_rho(summary)
       n_pairs(summary)
       n_samples_summary(summary)
       lead_time_summary(summary)
       valid_month_summary(summary)
       model_type_summary(summary)

Notes
-----
The 0.5 x 0.5 degree files contain daily values for lead days 16–46.

For an N-day accumulation, the first usable ending lead time is:

    16 + N - 1

For example, a 2-day accumulation can first be calculated for lead day 17,
because the file does not contain lead day 15.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Variable and accumulation
variable = "tp24"
x_days = 2

# Catchment
catchment = "regine_drammen"

# Forecast initialization dates to include
forecast_date_range = ["2020-01-02", "2020-01-06"] #["2020-01-02", "2023-06-26"]

# ECMWF files are available on Mondays and Thursdays.
forecast_date_option = "mt"       # "mt" or "all"

# Lead times contained in each 0.5 x 0.5 degree input file
first_input_lead = 16
last_input_lead = 46

# Independence-test grouping
#
# "valid_month":
#     Calculate correlations separately for each calendar month.
#     This is recommended for monthly UNSEEN analyses because it prevents
#     the seasonal precipitation cycle from inflating correlations.
#
# "all":
#     Pool all initialization dates into one correlation calculation at
#     each lead time.
grouping = "valid_month"

# Detrending method
#
# "first_difference":
#     Replace each member time series by differences between consecutive
#     values. This follows the approach described by Kelder et al.
#
# "none":
#     Use the original accumulated precipitation values.
detrend_method = "first_difference"

# Minimum number of paired values needed to calculate a correlation
minimum_samples = 10

# File handling
skip_missing_files = True
write_to_file = True

# Input paths
path_in_forecast = (
    config.dirs["s2s_forecast_daily"]
    + variable
    + "/"
)

path_in_hindcast = (
    config.dirs["s2s_hindcast_daily"]
    + variable
    + "/"
)

filename_weights = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
)

# Output path
path_out = config.dirs["s2s_processed"]


# =============================================================================
# Dates and filenames
# =============================================================================

def get_forecast_dates(date_range, option="mt"):
    """
    Return forecast initialization dates.

    Parameters
    ----------
    date_range : list of str
        Start and end dates in YYYY-MM-DD format.

    option : str
        "mt"  : include Mondays and Thursdays.
        "all" : include every calendar day.

    Returns
    -------
    list of str
        Forecast initialization dates in YYYY-MM-DD format.
    """

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if option == "mt":
        mondays = pd.date_range(start_date, end_date, freq="W-MON")
        thursdays = pd.date_range(start_date, end_date, freq="W-THU")
        dates = mondays.union(thursdays)

    elif option == "all":
        dates = pd.date_range(start_date, end_date, freq="D")

    else:
        raise ValueError(
            "forecast_date_option must be either 'mt' or 'all'."
        )

    return dates.sort_values().strftime("%Y-%m-%d").tolist()


def get_model_filenames(forecast_date):
    """
    Return forecast and hindcast filenames for one initialization date.
    """

    forecast_filename = os.path.join(
        path_in_forecast,
        f"{variable}_0.5x0.5_{forecast_date}.nc",
    )

    hindcast_filename = os.path.join(
        path_in_hindcast,
        f"{variable}_0.5x0.5_{forecast_date}.nc",
    )

    return forecast_filename, hindcast_filename


def make_output_filename():
    """Construct the output NetCDF filename."""

    first_valid_lead = first_input_lead + x_days - 1

    filename = (
        f"independence_spearman_{variable}_"
        f"{x_days}dayacc_"
        f"nve_catchment_{catchment}_"
        f"lead{first_valid_lead}-{last_input_lead}_"
        f"{grouping}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )

    return os.path.join(path_out, filename)


# =============================================================================
# Validation
# =============================================================================

def validate_user_settings():
    """Check the user settings before processing begins."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    expected_time_length = last_input_lead - first_input_lead + 1

    if x_days > expected_time_length:
        raise ValueError(
            f"x_days={x_days} exceeds the {expected_time_length} daily "
            "lead times available in each file."
        )

    if grouping not in {"all", "valid_month"}:
        raise ValueError(
            "grouping must be either 'all' or 'valid_month'."
        )

    if detrend_method not in {"none", "first_difference"}:
        raise ValueError(
            "detrend_method must be either 'none' or "
            "'first_difference'."
        )

    if minimum_samples < 3:
        raise ValueError(
            "minimum_samples must be at least 3."
        )


# =============================================================================
# Catchment weights
# =============================================================================

def load_weights(filename):
    """
    Load catchment weights.

    The weights should have dimensions latitude and longitude and contain
    a variable called catchment_weight.
    """

    with xr.open_dataset(filename) as dataset:

        if "catchment_weight" not in dataset:
            raise KeyError(
                f"'catchment_weight' was not found in:\n{filename}\n"
                f"Available variables: {list(dataset.data_vars)}"
            )

        weights = dataset["catchment_weight"].load().astype("float32")

    required_dimensions = {"latitude", "longitude"}

    if not required_dimensions.issubset(weights.dims):
        raise ValueError(
            "Catchment weights must contain latitude and longitude "
            f"dimensions. Found: {weights.dims}"
        )

    weights.name = "catchment_weight"

    return weights


# =============================================================================
# General data-processing functions
# =============================================================================

def convert_precipitation_to_mm(data):
    """
    Convert precipitation from metres to millimetres when required.
    """

    units = str(data.attrs.get("units", "")).strip().lower()

    metre_units = {
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    }

    if units in metre_units:
        data = data * 1000.0
        data.attrs["units"] = "mm"

    elif units in {"mm", "millimeter", "millimetre"}:
        data.attrs["units"] = "mm"

    else:
        print(
            f"Warning: unrecognized units '{units}'. "
            "No unit conversion was applied."
        )

    return data


def assign_lead_time_coordinate(data):
    """
    Assign lead-day numbers to the file's time dimension.

    The first time step in the input files represents lead day 16 and
    the last represents lead day 46.
    """

    if "time" not in data.dims:
        raise ValueError(
            "The input variable does not contain a time dimension."
        )

    n_times = data.sizes["time"]
    expected_n_times = last_input_lead - first_input_lead + 1

    if n_times != expected_n_times:
        raise ValueError(
            f"Expected {expected_n_times} time steps representing lead "
            f"days {first_input_lead}–{last_input_lead}, but found "
            f"{n_times} time steps."
        )

    lead_times = np.arange(
        first_input_lead,
        last_input_lead + 1,
        dtype="int16",
    )

    return data.assign_coords(
        lead_time=("time", lead_times)
    )


def catchment_mean(
    data,
    weights,
    spatial_dimensions=("latitude", "longitude"),
):
    """
    Calculate the catchment-weighted spatial mean.

    The mean is calculated independently for every time, member, and
    hindcast date:

        sum(data * weight) / sum(valid weights)

    Grid cells with missing data or non-positive weights are ignored.
    """

    missing_dimensions = [
        dimension
        for dimension in spatial_dimensions
        if dimension not in data.dims
    ]

    if missing_dimensions:
        raise ValueError(
            "Input data are missing the following spatial dimensions: "
            f"{missing_dimensions}"
        )

    # Align the grid coordinates exactly before multiplication.
    data, aligned_weights = xr.align(
        data,
        weights,
        join="exact",
    )

    valid = (
        np.isfinite(data)
        & np.isfinite(aligned_weights)
        & (aligned_weights > 0)
    )

    weighted_sum = (
        data.where(valid)
        * aligned_weights.where(valid)
    ).sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    weight_sum = aligned_weights.where(valid).sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    mean_data = weighted_sum / weight_sum
    mean_data = mean_data.where(weight_sum > 0)

    mean_data.attrs["description"] = (
        "Catchment-weighted daily mean precipitation"
    )
    mean_data.attrs["units"] = data.attrs.get("units", "")

    return mean_data


def calculate_nday_accumulation(data, n_days):
    """
    Calculate trailing N-day accumulated precipitation.

    The accumulated value assigned to lead day L covers the period ending
    on lead day L.

    Example for a 2-day accumulation:
        value at lead day 17 = daily values at lead days 16 and 17.
    """

    accumulated = data.rolling(
        time=n_days,
        min_periods=n_days,
    ).sum()

    # The first N - 1 values cannot be calculated because earlier daily
    # lead times are not present in the file.
    accumulated = accumulated.isel(
        time=slice(n_days - 1, None)
    )

    accumulated.attrs["description"] = (
        f"Trailing {n_days}-day accumulated catchment-mean precipitation"
    )
    accumulated.attrs["units"] = "mm"

    return accumulated


def replace_time_with_lead_time(data):
    """
    Replace the time dimension with the integer lead_time dimension.
    """

    if "lead_time" not in data.coords:
        raise KeyError(
            "A lead_time coordinate must be assigned before this function "
            "is called."
        )

    data = data.swap_dims({"time": "lead_time"})
    data = data.drop_vars("time")

    return data


# =============================================================================
# Initialization-date handling
# =============================================================================

def convert_hdate_to_datetime(hdate_values):
    """
    Convert hindcast hdate values to pandas datetime values.

    Supported hdate representations include:

    - integer YYYYMMDD, such as 20030403
    - string YYYYMMDD
    - datetime64 values
    """

    values = np.asarray(hdate_values)

    if np.issubdtype(values.dtype, np.datetime64):
        dates = pd.to_datetime(values)

    else:
        date_strings = [
            str(int(value)).zfill(8)
            for value in values
        ]

        dates = pd.to_datetime(
            date_strings,
            format="%Y%m%d",
            errors="raise",
        )

    return dates.to_numpy(dtype="datetime64[ns]")


def calculate_valid_dates(initialization_dates, lead_times):
    """
    Calculate the valid date for each initialization and lead time.

    Lead day 1 is the initialization date itself plus zero days.
    Therefore:

        valid date = initialization date + (lead time - 1) days
    """

    initialization_dates = np.asarray(
        initialization_dates,
        dtype="datetime64[ns]",
    )

    lead_times = np.asarray(
        lead_times,
        dtype="int64",
    )

    day_offsets = (
        lead_times - 1
    ).astype("timedelta64[D]")

    valid_dates = (
        initialization_dates[:, np.newaxis]
        + day_offsets[np.newaxis, :]
    )

    return valid_dates


# =============================================================================
# Read and process forecast files
# =============================================================================

def process_forecast_file(filename, forecast_date, weights):
    """
    Read and process one operational forecast file.

    Returns
    -------
    xarray.DataArray
        Dimensions:
            initialization, number, lead_time

        Coordinates include:
            initialization_date(initialization)
            valid_date(initialization, lead_time)
            valid_month(initialization, lead_time)
    """

    with xr.open_dataset(filename) as dataset:

        if variable not in dataset:
            raise KeyError(
                f"'{variable}' was not found in:\n{filename}\n"
                f"Available variables: {list(dataset.data_vars)}"
            )

        data = dataset[variable].load()

    data = convert_precipitation_to_mm(data)
    data = assign_lead_time_coordinate(data)
    data = catchment_mean(data, weights)
    data = calculate_nday_accumulation(data, x_days)
    data = replace_time_with_lead_time(data)

    # Make the member dimension order explicit.
    data = data.transpose("number", "lead_time")

    initialization_date = np.datetime64(
        pd.to_datetime(forecast_date),
        "ns",
    )

    data = data.expand_dims(
        initialization=[initialization_date]
    )

    data = data.assign_coords(
        initialization_date=(
            "initialization",
            np.array([initialization_date]),
        )
    )

    valid_dates = calculate_valid_dates(
        initialization_dates=[initialization_date],
        lead_times=data["lead_time"].values,
    )

    data = data.assign_coords(
        valid_date=(
            ("initialization", "lead_time"),
            valid_dates,
        ),
        valid_month=(
            ("initialization", "lead_time"),
            pd.DatetimeIndex(valid_dates.ravel())
            .month
            .to_numpy()
            .reshape(valid_dates.shape)
            .astype("int8"),
        ),
    )

    data.name = "accumulated_precipitation"

    return data


def process_hindcast_file(filename, weights):
    """
    Read and process one hindcast file.

    Each hdate is treated as one independent initialization containing
    11 ensemble members.

    Returns
    -------
    xarray.DataArray
        Dimensions:
            initialization, number, lead_time
    """

    with xr.open_dataset(filename) as dataset:

        if variable not in dataset:
            raise KeyError(
                f"'{variable}' was not found in:\n{filename}\n"
                f"Available variables: {list(dataset.data_vars)}"
            )

        data = dataset[variable].load()

    required_dimensions = {
        "time",
        "number",
        "hdate",
        "latitude",
        "longitude",
    }

    missing_dimensions = required_dimensions.difference(data.dims)

    if missing_dimensions:
        raise ValueError(
            f"Hindcast file is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    data = convert_precipitation_to_mm(data)
    data = assign_lead_time_coordinate(data)
    data = catchment_mean(data, weights)
    data = calculate_nday_accumulation(data, x_days)
    data = replace_time_with_lead_time(data)

    # Ensure a clear and consistent dimension order.
    data = data.transpose("hdate", "number", "lead_time")

    initialization_dates = convert_hdate_to_datetime(
        data["hdate"].values
    )

    # Replace hdate with a general initialization dimension. The original
    # hdate values are retained as a coordinate.
    original_hdates = data["hdate"].values

    data = data.rename({"hdate": "initialization"})

    data = data.assign_coords(
        initialization=np.arange(
            data.sizes["initialization"],
            dtype="int32",
        ),
        initialization_date=(
            "initialization",
            initialization_dates,
        ),
        hdate=(
            "initialization",
            original_hdates,
        ),
    )

    valid_dates = calculate_valid_dates(
        initialization_dates=initialization_dates,
        lead_times=data["lead_time"].values,
    )

    valid_months = (
        pd.DatetimeIndex(valid_dates.ravel())
        .month
        .to_numpy()
        .reshape(valid_dates.shape)
        .astype("int8")
    )

    data = data.assign_coords(
        valid_date=(
            ("initialization", "lead_time"),
            valid_dates,
        ),
        valid_month=(
            ("initialization", "lead_time"),
            valid_months,
        ),
    )

    data.name = "accumulated_precipitation"

    return data


# =============================================================================
# Build forecast and hindcast archives
# =============================================================================

def collect_processed_data(forecast_dates, weights):
    """
    Read all available forecast and hindcast files.

    Returns
    -------
    forecast_archive : xarray.DataArray or None
        All operational forecast initializations.

    hindcast_archive : xarray.DataArray or None
        All hindcast initializations.
    """

    forecast_list = []
    hindcast_list = []

    for index, forecast_date in enumerate(forecast_dates, start=1):

        print()
        print("=" * 72)
        print(
            f"Processing initialization {index} of "
            f"{len(forecast_dates)}: {forecast_date}"
        )
        print("=" * 72)

        forecast_filename, hindcast_filename = get_model_filenames(
            forecast_date
        )

        # ---------------------------------------------------------------------
        # Forecast
        # ---------------------------------------------------------------------

        if os.path.exists(forecast_filename):

            print("Forecast:", forecast_filename)

            forecast_data = process_forecast_file(
                filename=forecast_filename,
                forecast_date=forecast_date,
                weights=weights,
            )

            forecast_list.append(forecast_data)

        elif skip_missing_files:
            print(
                "Warning: forecast file does not exist and will be skipped:"
            )
            print(forecast_filename)

        else:
            raise FileNotFoundError(forecast_filename)

        # ---------------------------------------------------------------------
        # Hindcast
        # ---------------------------------------------------------------------

        if os.path.exists(hindcast_filename):

            print("Hindcast:", hindcast_filename)

            hindcast_data = process_hindcast_file(
                filename=hindcast_filename,
                weights=weights,
            )

            hindcast_list.append(hindcast_data)

        elif skip_missing_files:
            print(
                "Warning: hindcast file does not exist and will be skipped:"
            )
            print(hindcast_filename)

        else:
            raise FileNotFoundError(hindcast_filename)

    forecast_archive = concatenate_initializations(
        forecast_list,
        model_type="forecast",
    )

    hindcast_archive = concatenate_initializations(
        hindcast_list,
        model_type="hindcast",
    )

    return forecast_archive, hindcast_archive


def concatenate_initializations(data_list, model_type):
    """
    Combine processed files along the initialization dimension.
    """

    if len(data_list) == 0:
        print(f"No {model_type} data were loaded.")
        return None

    archive = xr.concat(
        data_list,
        dim="initialization",
        coords="minimal",
        compat="override",
        join="exact",
    )

    # Sort chronologically before first differencing.
    order = np.argsort(
        archive["initialization_date"].values
    )

    archive = archive.isel(initialization=order)

    # Use a simple sequential initialization coordinate after concatenation.
    archive = archive.assign_coords(
        initialization=np.arange(
            archive.sizes["initialization"],
            dtype="int32",
        )
    )

    archive.attrs["model_type"] = model_type
    archive.attrs["description"] = (
        f"{x_days}-day accumulated catchment-mean precipitation "
        f"for {model_type} initializations"
    )

    print(
        f"Loaded {archive.sizes['initialization']} "
        f"{model_type} initializations."
    )

    return archive


# =============================================================================
# Spearman rank correlation
# =============================================================================

def first_difference(values):
    """
    First-difference each ensemble-member time series.

    Parameters
    ----------
    values : numpy.ndarray
        Shape:
            number of samples, number of ensemble members

    Returns
    -------
    numpy.ndarray
        First-differenced values with one fewer sample.
    """

    return np.diff(values, axis=0)


def spearman_correlation(x, y, minimum_valid_samples):
    """
    Calculate Spearman rank correlation between two one-dimensional series.

    Missing values are removed pairwise.

    Returns
    -------
    rho : float
        Spearman rank correlation.

    n_valid : int
        Number of paired samples used.
    """

    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(valid.sum())

    if n_valid < minimum_valid_samples:
        return np.nan, n_valid

    x_valid = x[valid]
    y_valid = y[valid]

    # Correlation is undefined when either series is constant.
    if (
        np.all(x_valid == x_valid[0])
        or np.all(y_valid == y_valid[0])
    ):
        return np.nan, n_valid

    x_ranks = rankdata(x_valid, method="average")
    y_ranks = rankdata(y_valid, method="average")

    rho = np.corrcoef(x_ranks, y_ranks)[0, 1]

    return float(rho), n_valid


def select_group_values(data, lead_time, month=None):
    """
    Extract a sample-by-member matrix for one lead time and optional month.
    """

    lead_data = data.sel(lead_time=lead_time)

    if month is not None:
        month_mask = lead_data["valid_month"] == month
        lead_data = lead_data.where(
            month_mask,
            drop=True,
        )

    # Make the orientation explicit:
    # rows are initialization samples and columns are members.
    lead_data = lead_data.transpose(
        "initialization",
        "number",
    )

    values = lead_data.values.astype("float64")

    if detrend_method == "first_difference":
        values = first_difference(values)

    return values


def calculate_correlations_for_archive(data, model_type):
    """
    Calculate all unique pairwise member correlations for one archive.

    Correlations are calculated separately at each lead time and, when
    requested, separately for each valid calendar month.

    Returns
    -------
    pandas.DataFrame
        One row for every unique member pair, lead time, and month.
    """

    if data is None:
        return pd.DataFrame()

    member_labels = data["number"].values
    lead_times = data["lead_time"].values.astype(int)

    if grouping == "valid_month":
        months = np.arange(1, 13, dtype=int)
    else:
        # Zero means that all calendar months were pooled.
        months = np.array([0], dtype=int)

    records = []

    total_groups = len(lead_times) * len(months)
    group_counter = 0

    for lead_time in lead_times:

        for month in months:

            group_counter += 1

            month_selection = None if month == 0 else int(month)

            values = select_group_values(
                data=data,
                lead_time=int(lead_time),
                month=month_selection,
            )

            n_raw_samples = values.shape[0]

            if n_raw_samples < minimum_samples:
                print(
                    f"{model_type:8s} | lead {lead_time:2d} | "
                    f"month {month:2d} | too few samples "
                    f"({n_raw_samples})"
                )
                continue

            print(
                f"{model_type:8s} | lead {lead_time:2d} | "
                f"month {month:2d} | "
                f"group {group_counter}/{total_groups} | "
                f"samples={n_raw_samples}"
            )

            for member_index_1, member_index_2 in combinations(
                range(len(member_labels)),
                2,
            ):

                rho, n_valid = spearman_correlation(
                    x=values[:, member_index_1],
                    y=values[:, member_index_2],
                    minimum_valid_samples=minimum_samples,
                )

                records.append(
                    {
                        "model_type": model_type,
                        "lead_time": int(lead_time),
                        "valid_month": int(month),
                        "member_1": int(
                            member_labels[member_index_1]
                        ),
                        "member_2": int(
                            member_labels[member_index_2]
                        ),
                        "spearman_rho": rho,
                        "n_samples": int(n_valid),
                    }
                )

    return pd.DataFrame.from_records(records)


# =============================================================================
# Correlation summaries
# =============================================================================

def summarize_correlations(pairwise_correlations):
    """
    Calculate summary statistics for pairwise correlations.

    Summaries are produced for:

    - forecasts;
    - hindcasts;
    - the pooled forecast and hindcast correlation distributions.

    The combined summary pools the calculated correlation coefficients.
    It does not treat forecast and hindcast members as one larger ensemble.
    """

    if pairwise_correlations.empty:
        return pd.DataFrame()

    grouping_columns = [
        "model_type",
        "lead_time",
        "valid_month",
    ]

    individual_summaries = calculate_group_summaries(
        pairwise_correlations,
        grouping_columns=grouping_columns,
    )

    combined_data = pairwise_correlations.copy()
    combined_data["model_type"] = "combined"

    combined_summaries = calculate_group_summaries(
        combined_data,
        grouping_columns=grouping_columns,
    )

    summaries = pd.concat(
        [individual_summaries, combined_summaries],
        ignore_index=True,
    )

    return summaries.sort_values(
        ["model_type", "valid_month", "lead_time"]
    ).reset_index(drop=True)


def calculate_group_summaries(dataframe, grouping_columns):
    """Calculate correlation statistics for grouped DataFrame rows."""

    summary_records = []

    for group_values, group in dataframe.groupby(
        grouping_columns,
        sort=True,
    ):

        model_type, lead_time, valid_month = group_values

        correlations = group["spearman_rho"].to_numpy()
        correlations = correlations[np.isfinite(correlations)]

        if correlations.size == 0:
            continue

        sample_counts = group.loc[
            np.isfinite(group["spearman_rho"]),
            "n_samples",
        ].to_numpy()

        summary_records.append(
            {
                "model_type": str(model_type),
                "lead_time": int(lead_time),
                "valid_month": int(valid_month),
                "mean_rho": float(np.mean(correlations)),
                "median_rho": float(np.median(correlations)),
                "q05_rho": float(np.quantile(correlations, 0.05)),
                "q25_rho": float(np.quantile(correlations, 0.25)),
                "q75_rho": float(np.quantile(correlations, 0.75)),
                "q95_rho": float(np.quantile(correlations, 0.95)),
                "minimum_rho": float(np.min(correlations)),
                "maximum_rho": float(np.max(correlations)),
                "n_pairs": int(correlations.size),
                "n_samples_summary": int(
                    np.median(sample_counts)
                ),
            }
        )

    return pd.DataFrame.from_records(summary_records)


# =============================================================================
# Convert results to xarray
# =============================================================================

def results_to_dataset(pairwise_correlations, summaries):
    """
    Convert correlation results and summaries to an xarray Dataset.
    """

    if pairwise_correlations.empty:
        raise ValueError(
            "No pairwise correlations were calculated."
        )

    pairwise_correlations = pairwise_correlations.reset_index(drop=True)
    summaries = summaries.reset_index(drop=True)

    dataset = xr.Dataset(
        data_vars={
            # Individual pairwise correlations
            "spearman_rho": (
                "pair",
                pairwise_correlations["spearman_rho"]
                .to_numpy(dtype="float32"),
            ),
            "lead_time": (
                "pair",
                pairwise_correlations["lead_time"]
                .to_numpy(dtype="int16"),
            ),
            "valid_month": (
                "pair",
                pairwise_correlations["valid_month"]
                .to_numpy(dtype="int8"),
            ),
            "member_1": (
                "pair",
                pairwise_correlations["member_1"]
                .to_numpy(dtype="int16"),
            ),
            "member_2": (
                "pair",
                pairwise_correlations["member_2"]
                .to_numpy(dtype="int16"),
            ),
            "model_type": (
                "pair",
                pairwise_correlations["model_type"]
                .to_numpy(dtype="U9"),
            ),
            "n_samples": (
                "pair",
                pairwise_correlations["n_samples"]
                .to_numpy(dtype="int32"),
            ),

            # Summary statistics
            "mean_rho": (
                "summary",
                summaries["mean_rho"].to_numpy(dtype="float32"),
            ),
            "median_rho": (
                "summary",
                summaries["median_rho"].to_numpy(dtype="float32"),
            ),
            "q05_rho": (
                "summary",
                summaries["q05_rho"].to_numpy(dtype="float32"),
            ),
            "q25_rho": (
                "summary",
                summaries["q25_rho"].to_numpy(dtype="float32"),
            ),
            "q75_rho": (
                "summary",
                summaries["q75_rho"].to_numpy(dtype="float32"),
            ),
            "q95_rho": (
                "summary",
                summaries["q95_rho"].to_numpy(dtype="float32"),
            ),
            "minimum_rho": (
                "summary",
                summaries["minimum_rho"].to_numpy(dtype="float32"),
            ),
            "maximum_rho": (
                "summary",
                summaries["maximum_rho"].to_numpy(dtype="float32"),
            ),
            "n_pairs": (
                "summary",
                summaries["n_pairs"].to_numpy(dtype="int32"),
            ),
            "n_samples_summary": (
                "summary",
                summaries["n_samples_summary"]
                .to_numpy(dtype="int32"),
            ),
            "lead_time_summary": (
                "summary",
                summaries["lead_time"].to_numpy(dtype="int16"),
            ),
            "valid_month_summary": (
                "summary",
                summaries["valid_month"].to_numpy(dtype="int8"),
            ),
            "model_type_summary": (
                "summary",
                summaries["model_type"].to_numpy(dtype="U9"),
            ),
        },
        coords={
            "pair": np.arange(
                len(pairwise_correlations),
                dtype="int32",
            ),
            "summary": np.arange(
                len(summaries),
                dtype="int32",
            ),
        },
    )

    add_output_metadata(dataset)

    return dataset


def add_output_metadata(dataset):
    """Add descriptions and processing information to the output."""

    dataset["spearman_rho"].attrs = {
        "long_name": "Spearman rank correlation",
        "description": (
            "Correlation between one unique pair of ensemble-member "
            "time series at a fixed lead time"
        ),
    }

    dataset["lead_time"].attrs = {
        "long_name": "forecast lead day",
        "units": "days",
    }

    dataset["valid_month"].attrs = {
        "long_name": "calendar month of forecast valid date",
        "description": (
            "Values 1–12 indicate calendar month; 0 indicates that "
            "all calendar months were pooled"
        ),
    }

    dataset["member_1"].attrs["long_name"] = (
        "first ensemble member in pair"
    )
    dataset["member_2"].attrs["long_name"] = (
        "second ensemble member in pair"
    )

    dataset["model_type"].attrs = {
        "long_name": "ensemble source",
        "description": "Either forecast or hindcast",
    }

    dataset["n_samples"].attrs = {
        "long_name": "number of paired samples",
        "description": (
            "Number of valid paired initialization samples used to "
            "calculate the correlation"
        ),
    }

    dataset["mean_rho"].attrs["long_name"] = (
        "mean pairwise Spearman correlation"
    )
    dataset["median_rho"].attrs["long_name"] = (
        "median pairwise Spearman correlation"
    )
    dataset["q05_rho"].attrs["long_name"] = (
        "5th percentile pairwise Spearman correlation"
    )
    dataset["q25_rho"].attrs["long_name"] = (
        "25th percentile pairwise Spearman correlation"
    )
    dataset["q75_rho"].attrs["long_name"] = (
        "75th percentile pairwise Spearman correlation"
    )
    dataset["q95_rho"].attrs["long_name"] = (
        "95th percentile pairwise Spearman correlation"
    )

    dataset.attrs = {
        "title": (
            "ECMWF S2S ensemble-member independence test"
        ),
        "variable": variable,
        "catchment": catchment,
        "accumulation_days": int(x_days),
        "first_input_lead": int(first_input_lead),
        "last_input_lead": int(last_input_lead),
        "first_valid_accumulation_lead": int(
            first_input_lead + x_days - 1
        ),
        "forecast_date_start": forecast_date_range[0],
        "forecast_date_end": forecast_date_range[1],
        "grouping": grouping,
        "detrend_method": detrend_method,
        "minimum_samples": int(minimum_samples),
        "forecast_interpretation": (
            "Each forecast date is one 51-member initialization"
        ),
        "hindcast_interpretation": (
            "Each hdate is one separate 11-member initialization"
        ),
        "combined_summary_interpretation": (
            "Forecast and hindcast pairwise correlations are pooled; "
            "their members are not treated as one larger ensemble"
        ),
    }


# =============================================================================
# Output
# =============================================================================

def write_output(dataset):
    """Write the independence-test results to NetCDF."""

    os.makedirs(path_out, exist_ok=True)

    filename_out = make_output_filename()

    encoding = {
        "spearman_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "mean_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "median_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "q05_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "q25_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "q75_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "q95_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
    }

    dataset.to_netcdf(
        filename_out,
        encoding=encoding,
    )

    print()
    print("=" * 72)
    print("Output written successfully")
    print("=" * 72)
    print(filename_out)


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    forecast_dates = get_forecast_dates(
        date_range=forecast_date_range,
        option=forecast_date_option,
    )

    print("=" * 72)
    print("ECMWF S2S ensemble-member independence test")
    print("=" * 72)
    print(f"Variable:             {variable}")
    print(f"Catchment:            {catchment}")
    print(f"Accumulation:         {x_days} days")
    print(f"Input leads:          {first_input_lead}–{last_input_lead}")
    print(
        "Usable ending leads: "
        f"{first_input_lead + x_days - 1}–{last_input_lead}"
    )
    print(f"Grouping:             {grouping}")
    print(f"Detrending:           {detrend_method}")
    print(f"Forecast dates:       {len(forecast_dates)}")
    print(f"Minimum samples:      {minimum_samples}")

    weights = load_weights(filename_weights)

    forecast_archive, hindcast_archive = collect_processed_data(
        forecast_dates=forecast_dates,
        weights=weights,
    )

    forecast_correlations = calculate_correlations_for_archive(
        data=forecast_archive,
        model_type="forecast",
    )

    hindcast_correlations = calculate_correlations_for_archive(
        data=hindcast_archive,
        model_type="hindcast",
    )

    pairwise_correlations = pd.concat(
        [
            forecast_correlations,
            hindcast_correlations,
        ],
        ignore_index=True,
    )

    if pairwise_correlations.empty:
        raise RuntimeError(
            "No pairwise correlations were calculated. Check the input "
            "paths, forecast dates, grouping, and minimum_samples."
        )

    correlation_summaries = summarize_correlations(
        pairwise_correlations
    )

    output_dataset = results_to_dataset(
        pairwise_correlations=pairwise_correlations,
        summaries=correlation_summaries,
    )

    print()
    print(output_dataset)

    if write_to_file:
        write_output(output_dataset)
