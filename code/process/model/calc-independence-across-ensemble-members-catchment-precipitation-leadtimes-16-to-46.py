"""
Calculate ensemble-member independence for ECMWF S2S precipitation.

For each lead time, this script:

1. Reads operational forecasts and corresponding hindcasts.
2. Calculates catchment-weighted mean daily precipitation.
3. Calculates trailing N-day accumulated precipitation.
4. Builds one time series for each ensemble member across initialization dates.
5. Calculates the Spearman rank correlation for every unique pair of members.
6. Saves the forecast and hindcast pairwise correlations to NetCDF.

Forecast and hindcast ensembles are handled separately:

Forecast:
    Each forecast date is one initialization with 51 ensemble members.

Hindcast:
    Each hdate is one separate initialization with 11 ensemble members.

Output data variables
---------------------
forecast_spearman_rho(valid_month, lead_time, forecast_pair)
hindcast_spearman_rho(valid_month, lead_time, hindcast_pair)

Only these two data variables are written to the output file. Coordinates such
as valid_month, lead_time, forecast_pair, and hindcast_pair are retained so
that the correlation arrays can be selected and plotted easily.

Notes
-----
The 0.5 x 0.5 degree files contain daily values for lead days 16–46.

For an N-day accumulation, the first usable ending lead time is:

    16 + N - 1

For example, a 2-day accumulation can first be calculated for lead day 17,
because lead day 15 is not available in these files.
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
forecast_date_range = ["2020-01-02", "2023-06-26"]

# ECMWF files are normally available on Mondays and Thursdays.
forecast_date_option = "mt"  # "mt" or "all"

# Lead times contained in each 0.5 x 0.5 degree input file
first_input_lead = 16
last_input_lead = 46

# Grouping used for the independence calculation
#
# "valid_month":
#     Calculate correlations separately for each calendar month.
#     This is recommended when the UNSEEN analysis is performed by month.
#
# "all":
#     Pool all calendar months at each lead time.
grouping = "valid_month"


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
    """Return forecast and hindcast filenames for one forecast date."""

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
    """Check user settings before processing."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    number_of_input_leads = last_input_lead - first_input_lead + 1

    if x_days > number_of_input_leads:
        raise ValueError(
            f"x_days={x_days} exceeds the {number_of_input_leads} "
            "daily lead times available in each input file."
        )

    if grouping not in {"all", "valid_month"}:
        raise ValueError(
            "grouping must be either 'all' or 'valid_month'."
        )


    if minimum_samples < 3:
        raise ValueError(
            "minimum_samples must be at least 3."
        )


# =============================================================================
# Loading and basic processing
# =============================================================================

def load_weights(filename):
    """
    Load catchment weights.

    The file must contain the variable catchment_weight with latitude and
    longitude dimensions.
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


def convert_precipitation_to_mm(data):
    """Convert precipitation from metres to millimetres when needed."""

    units = str(data.attrs.get("units", "")).strip().lower()

    metre_units = {
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
    }

    millimetre_units = {
        "mm",
        "millimeter",
        "millimeters",
        "millimetre",
        "millimetres",
    }

    if units in metre_units:
        data = data * 1000.0
        data.attrs["units"] = "mm"

    elif units in millimetre_units:
        data.attrs["units"] = "mm"

    else:
        print(
            f"Warning: unrecognized precipitation units '{units}'. "
            "No unit conversion was applied."
        )

    return data


def assign_lead_time_coordinate(data):
    """
    Assign integer lead-day numbers to the time dimension.

    The first time step is assumed to represent first_input_lead and the last
    time step is assumed to represent last_input_lead.
    """

    if "time" not in data.dims:
        raise ValueError(
            "The input variable does not contain a time dimension."
        )

    expected_number_of_times = last_input_lead - first_input_lead + 1
    actual_number_of_times = data.sizes["time"]

    if actual_number_of_times != expected_number_of_times:
        raise ValueError(
            f"Expected {expected_number_of_times} time steps representing "
            f"lead days {first_input_lead}–{last_input_lead}, but found "
            f"{actual_number_of_times}."
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

    Formula
    -------
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
            "Input data are missing these spatial dimensions: "
            f"{missing_dimensions}"
        )

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
        value at lead day 17 = lead-day-16 value + lead-day-17 value.
    """

    accumulated = data.rolling(
        time=n_days,
        min_periods=n_days,
    ).sum()

    accumulated = accumulated.isel(
        time=slice(n_days - 1, None)
    )

    accumulated.attrs["description"] = (
        f"Trailing {n_days}-day accumulated catchment-mean precipitation"
    )
    accumulated.attrs["units"] = "mm"

    return accumulated


def replace_time_with_lead_time(data):
    """Replace the time dimension with the integer lead_time dimension."""

    if "lead_time" not in data.coords:
        raise KeyError(
            "A lead_time coordinate must be assigned before this function "
            "is called."
        )

    data = data.swap_dims({"time": "lead_time"})

    if "time" in data.coords:
        data = data.drop_vars("time")

    return data


# =============================================================================
# Initialization dates and valid dates
# =============================================================================

def convert_hdate_to_datetime(hdate_values):
    """
    Convert hindcast hdate values to datetime64 values.

    Supported representations include integer YYYYMMDD, string YYYYMMDD,
    and datetime64 values.
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
    Calculate valid dates for each initialization and lead time.

    This assumes:

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

    return (
        initialization_dates[:, np.newaxis]
        + day_offsets[np.newaxis, :]
    )


# =============================================================================
# Process forecast and hindcast files
# =============================================================================

def process_forecast_file(filename, forecast_date, weights):
    """
    Read and process one operational forecast file.

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

    data = convert_precipitation_to_mm(data)
    data = assign_lead_time_coordinate(data)
    data = catchment_mean(data, weights)
    data = calculate_nday_accumulation(data, x_days)
    data = replace_time_with_lead_time(data)
    data = data.transpose("number", "lead_time")

    initialization_date = np.datetime64(
        pd.to_datetime(forecast_date),
        "ns",
    )

    data = data.expand_dims(
        initialization=[0]
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


def process_hindcast_file(filename, weights):
    """
    Read and process one hindcast file.

    Each hdate is treated as one separate initialization with 11 members.

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
            "Hindcast file is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    data = convert_precipitation_to_mm(data)
    data = assign_lead_time_coordinate(data)
    data = catchment_mean(data, weights)
    data = calculate_nday_accumulation(data, x_days)
    data = replace_time_with_lead_time(data)
    data = data.transpose("hdate", "number", "lead_time")

    original_hdates = data["hdate"].values
    initialization_dates = convert_hdate_to_datetime(original_hdates)

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

def concatenate_initializations(data_list, model_type):
    """Combine processed files along the initialization dimension."""

    if not data_list:
        print(f"No {model_type} data were loaded.")
        return None

    archive = xr.concat(
        data_list,
        dim="initialization",
        coords="minimal",
        compat="override",
        join="exact",
    )

    chronological_order = np.argsort(
        archive["initialization_date"].values
    )

    archive = archive.isel(
        initialization=chronological_order
    )

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


def collect_processed_data(forecast_dates, weights):
    """
    Read all available forecast and hindcast files.

    Returns
    -------
    forecast_archive : xarray.DataArray or None
    hindcast_archive : xarray.DataArray or None
    """

    forecast_list = []
    hindcast_list = []

    for index, forecast_date in enumerate(forecast_dates, start=1):

        print()
        print("=" * 72)
        print(
            f"Processing forecast date {index} of "
            f"{len(forecast_dates)}: {forecast_date}"
        )
        print("=" * 72)

        forecast_filename, hindcast_filename = get_model_filenames(
            forecast_date
        )

        if os.path.exists(forecast_filename):

            print("Forecast:", forecast_filename)

            forecast_data = process_forecast_file(
                filename=forecast_filename,
                forecast_date=forecast_date,
                weights=weights,
            )

            forecast_list.append(forecast_data)

        elif skip_missing_files:

            print("Warning: missing forecast file:")
            print(forecast_filename)

        else:
            raise FileNotFoundError(forecast_filename)

        if os.path.exists(hindcast_filename):

            print("Hindcast:", hindcast_filename)

            hindcast_data = process_hindcast_file(
                filename=hindcast_filename,
                weights=weights,
            )

            hindcast_list.append(hindcast_data)

        elif skip_missing_files:

            print("Warning: missing hindcast file:")
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


# =============================================================================
# Spearman rank correlations
# =============================================================================

def make_member_pairs(number_of_members):
    """
    Return all unique pairs of ensemble-member column indices.

    For 51 forecast members, this returns 1275 pairs.
    For 11 hindcast members, this returns 55 pairs.
    """

    return list(combinations(range(number_of_members), 2))


def select_group_values(data, lead_time, month=None):
    """
    Extract a sample-by-member matrix for one lead time and optional month.

    Rows are initialization samples and columns are ensemble members.
    """

    lead_data = data.sel(lead_time=lead_time)

    if month is not None:
        month_mask = lead_data["valid_month"] == month
        lead_data = lead_data.where(month_mask, drop=True)

    lead_data = lead_data.transpose(
        "initialization",
        "number",
    )

    return lead_data.values.astype("float64")


def spearman_correlation(x, y, minimum_valid_samples):
    """
    Calculate Spearman rank correlation for two one-dimensional series.

    Missing values are removed pairwise.
    """

    valid = np.isfinite(x) & np.isfinite(y)
    number_of_valid_samples = int(valid.sum())

    if number_of_valid_samples < minimum_valid_samples:
        return np.nan, number_of_valid_samples

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        np.all(x_valid == x_valid[0])
        or np.all(y_valid == y_valid[0])
    ):
        return np.nan, number_of_valid_samples

    x_ranks = rankdata(x_valid, method="average")
    y_ranks = rankdata(y_valid, method="average")

    rho = np.corrcoef(x_ranks, y_ranks)[0, 1]

    return float(rho), number_of_valid_samples


def calculate_pairwise_correlations(data, model_type):
    """
    Calculate all unique pairwise correlations for one ensemble system.

    Parameters
    ----------
    data : xarray.DataArray or None
        Processed forecast or hindcast archive with dimensions:
        initialization, number, lead_time.

    model_type : str
        Either "forecast" or "hindcast". Used only for printed progress.

    Returns
    -------
    xarray.DataArray or None
        Dimensions:
            valid_month, lead_time, pair
    """

    if data is None:
        return None

    pair_indices = make_member_pairs(
        number_of_members=data.sizes["number"]
    )

    lead_times = data["lead_time"].values.astype("int16")

    if grouping == "valid_month":
        months = np.arange(1, 13, dtype="int8")
    else:
        months = np.array([0], dtype="int8")

    number_of_months = len(months)
    number_of_leads = len(lead_times)
    number_of_pairs = len(pair_indices)

    correlations = np.full(
        (number_of_months, number_of_leads, number_of_pairs),
        np.nan,
        dtype="float32",
    )

    total_groups = number_of_months * number_of_leads
    group_number = 0

    for month_index, month in enumerate(months):

        selected_month = None if month == 0 else int(month)

        for lead_index, lead_time in enumerate(lead_times):

            group_number += 1

            values = select_group_values(
                data=data,
                lead_time=int(lead_time),
                month=selected_month,
            )

            print(
                f"{model_type:8s} | "
                f"month={int(month):2d} | "
                f"lead={int(lead_time):2d} | "
                f"group={group_number}/{total_groups} | "
                f"available samples={values.shape[0]}"
            )

            for pair_index, (
                member_index_1,
                member_index_2,
            ) in enumerate(pair_indices):

                rho, _ = spearman_correlation(
                    x=values[:, member_index_1],
                    y=values[:, member_index_2],
                    minimum_valid_samples=minimum_samples,
                )

                correlations[
                    month_index,
                    lead_index,
                    pair_index,
                ] = rho

    return xr.DataArray(
        correlations,
        dims=("valid_month", "lead_time", "pair"),
        coords={
            "valid_month": months,
            "lead_time": lead_times,
            "pair": np.arange(number_of_pairs, dtype="int16"),
        },
        name=f"{model_type}_spearman_rho",
    )


# =============================================================================
# Build output dataset
# =============================================================================

def build_output_dataset(
    forecast_correlations,
    hindcast_correlations,
):
    """
    Build the final output Dataset.

    Only two data variables are included:

        forecast_spearman_rho
        hindcast_spearman_rho
    """

    data_variables = {}

    if forecast_correlations is not None:
        data_variables["forecast_spearman_rho"] = (
            forecast_correlations.rename(
                {"pair": "forecast_pair"}
            )
        )

    if hindcast_correlations is not None:
        data_variables["hindcast_spearman_rho"] = (
            hindcast_correlations.rename(
                {"pair": "hindcast_pair"}
            )
        )

    if not data_variables:
        raise RuntimeError(
            "No forecast or hindcast correlations were calculated."
        )

    dataset = xr.Dataset(data_variables)

    add_output_metadata(dataset)

    return dataset


def add_output_metadata(dataset):
    """Add concise metadata to the output Dataset."""

    if "forecast_spearman_rho" in dataset:
        dataset["forecast_spearman_rho"].attrs = {
            "long_name": (
                "pairwise Spearman rank correlation between forecast members"
            ),
            "description": (
                "Calculated across operational forecast initialization dates "
                "at a fixed lead time and, when requested, valid month"
            ),
        }

    if "hindcast_spearman_rho" in dataset:
        dataset["hindcast_spearman_rho"].attrs = {
            "long_name": (
                "pairwise Spearman rank correlation between hindcast members"
            ),
            "description": (
                "Calculated across hindcast hdate initializations at a fixed "
                "lead time and, when requested, valid month"
            ),
        }

    dataset["lead_time"].attrs = {
        "long_name": "forecast lead day",
        "units": "days",
    }

    dataset["valid_month"].attrs = {
        "long_name": "calendar month of valid date",
        "description": (
            "Values 1–12 indicate calendar month; 0 means that all "
            "calendar months were pooled"
        ),
    }

    dataset.attrs = {
        "title": "ECMWF S2S ensemble-member independence test",
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
        "minimum_samples": int(minimum_samples),
        "forecast_interpretation": (
            "Each operational forecast date is one 51-member initialization"
        ),
        "hindcast_interpretation": (
            "Each hindcast hdate is one separate 11-member initialization"
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
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable_name in dataset.data_vars
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
    print(f"Forecast dates:       {len(forecast_dates)}")
    print(f"Minimum samples:      {minimum_samples}")

    weights = load_weights(filename_weights)

    forecast_archive, hindcast_archive = collect_processed_data(
        forecast_dates=forecast_dates,
        weights=weights,
    )

    forecast_correlations = calculate_pairwise_correlations(
        data=forecast_archive,
        model_type="forecast",
    )

    hindcast_correlations = calculate_pairwise_correlations(
        data=hindcast_archive,
        model_type="hindcast",
    )

    output_dataset = build_output_dataset(
        forecast_correlations=forecast_correlations,
        hindcast_correlations=hindcast_correlations,
    )

    print()
    print(output_dataset)

    if write_to_file:
        write_output(output_dataset)
