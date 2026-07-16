"""
Build a lead-time-dependent distribution of accumulated precipitation for one
selected valid month from S2S forecasts and hindcasts.

The script processes one valid calendar month per run.

Before loading an input variable into memory, the script first inspects only
the file coordinates and determines whether any complete N-day accumulations
can be valid in the requested month. Files that cannot contribute samples
are skipped. Forecast and hindcast files then use the same processing path.

For relevant files, the script:

1. Loads daily precipitation.
2. Computes catchment-weighted mean precipitation.
3. Computes trailing N-day accumulated precipitation.
4. Keeps values whose accumulation end date falls in `valid_month`.
5. Stores all retained samples in:

       accumulated_value(lead_time, index)

The `index` dimension combines samples from different initialization dates,
hindcast dates, and ensemble members.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"

# Number of consecutive days included in each accumulation.
x_days = 2

catchment = "regine_drammen"

# Valid calendar month to retain:
# 1 = January, 2 = February, ..., 12 = December.
valid_month = 1

# Forecast initialization dates to consider.
forecast_date_range = ["2023-03-16","2023-03-16"]#["2020-01-02","2023-06-26"]

# Forecast initializations are normally available on Mondays and Thursdays.
forecast_date_option = "mt"  # "mt" or "all"

path_in_forecast = Path(config.dirs["s2s_forecast_daily"]) / variable
path_in_hindcast = Path(config.dirs["s2s_hindcast_daily"]) / variable

filename_weights = (
    Path(config.dirs["nve"])
    / f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
)

path_out = Path(config.dirs["s2s_processed"])

write_to_file = True


# =============================================================================
# User-setting validation
# =============================================================================

def validate_user_settings():
    """Check the main user settings before processing begins."""

    if not 1 <= valid_month <= 12:
        raise ValueError(
            f"valid_month must be between 1 and 12. Received {valid_month}."
        )

    if x_days < 1:
        raise ValueError(
            f"x_days must be at least 1. Received {x_days}."
        )

    if forecast_date_option not in {"mt", "all"}:
        raise ValueError(
            "forecast_date_option must be either 'mt' or 'all'."
        )


# =============================================================================
# Dates and filenames
# =============================================================================

def get_forecast_dates(date_range, option="mt"):
    """
    Return forecast initialization dates.

    Parameters
    ----------
    date_range : sequence of str
        Start and end dates in YYYY-MM-DD format.

    option : {"mt", "all"}
        "mt" returns Mondays and Thursdays.
        "all" returns every calendar day.

    Returns
    -------
    list of str
        Initialization dates in YYYY-MM-DD format.
    """

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    if option == "mt":
        mondays = pd.date_range(start_date, end_date, freq="W-MON")
        thursdays = pd.date_range(start_date, end_date, freq="W-THU")
        dates = mondays.union(thursdays)

    elif option == "all":
        dates = pd.date_range(start_date, end_date, freq="D")

    else:
        raise ValueError("option must be either 'mt' or 'all'.")

    return dates.sort_values().strftime("%Y-%m-%d").tolist()


def get_model_filenames(forecast_date):
    """Return forecast and hindcast filenames for one initialization date."""

    forecast_filename = (
        path_in_forecast
        / f"{variable}_0.5x0.5_{forecast_date}.nc"
    )

    hindcast_filename = (
        path_in_hindcast
        / f"{variable}_0.5x0.5_{forecast_date}.nc"
    )

    return forecast_filename, hindcast_filename


def make_output_filename():
    """Return the output filename for the selected valid month."""

    month_name = pd.Timestamp(2000, valid_month, 1).strftime("%B").lower()

    filename = (
        f"distribution_valid_month_{valid_month:02d}_{month_name}_"
        f"{variable}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )

    return path_out / filename


# =============================================================================
# Lightweight coordinate checks
# =============================================================================

def get_complete_accumulation_times(time_values):
    """
    Return dates representing complete trailing N-day accumulations.

    The first `x_days - 1` daily dates are removed because a complete
    accumulation cannot yet be calculated.
    """

    times = pd.DatetimeIndex(pd.to_datetime(time_values)).normalize()

    if len(times) < x_days:
        return pd.DatetimeIndex([])

    return times[x_days - 1:]


def calculate_lead_times(valid_times, forecast_date):
    """
    Calculate lead times in whole days.

    Lead time is defined as:

        accumulation end date - archive forecast initialization date

    Returns
    -------
    numpy.ndarray
        Lead times as integer days.
    """

    initialization_date = pd.Timestamp(forecast_date).normalize()

    valid_times = pd.DatetimeIndex(valid_times).normalize()

    lead_times = np.asarray(
        (valid_times - initialization_date).days,
        dtype="int32",
    )

    if np.any(lead_times < 0):
        raise ValueError(
            f"Negative lead times were found for initialization "
            f"{forecast_date}."
        )

    return lead_times


def hdate_values_to_timestamps(hdate_values):
    """
    Convert hindcast initialization dates to normalized pandas timestamps.
    """

    values = np.asarray(hdate_values)

    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(values)).normalize()

    return pd.DatetimeIndex(
        pd.to_datetime(
            values.astype("int64").astype(str),
            format="%Y%m%d",
        )
    ).normalize()


def file_has_requested_month(filename, forecast_date):
    """
    Check whether a forecast or hindcast file can contribute to the requested
    valid month.

    Only coordinates are read. The precipitation variable is not loaded.

    Forecast files:
        valid date = time coordinate

    Hindcast files:
        valid date = hdate + lead time
    """

    if not filename.exists():
        raise FileNotFoundError(f"Input file not found:\n{filename}")

    with xr.open_dataset(filename) as dataset:

        if "time" not in dataset.coords:
            raise KeyError(
                f"The input file has no 'time' coordinate:\n{filename}"
            )

        accumulation_times = get_complete_accumulation_times(
            dataset["time"].values
        )

        if len(accumulation_times) == 0:
            return False

        # Hindcasts contain an hdate coordinate. Forecasts do not.
        if "hdate" in dataset.coords:

            lead_times = calculate_lead_times(
                accumulation_times,
                forecast_date,
            )

            hdates = hdate_values_to_timestamps(
                dataset["hdate"].values
            )

            valid_dates = (
                hdates.values[:, np.newaxis]
                + lead_times.astype("timedelta64[D]")[np.newaxis, :]
            )

            valid_months = pd.DatetimeIndex(
                valid_dates.ravel()
            ).month

        else:
            valid_months = accumulation_times.month

    return np.any(valid_months == valid_month)


# =============================================================================
# Loading
# =============================================================================

def load_weights(filename):
    """Load catchment weights."""

    with xr.open_dataset(filename) as dataset:

        if "catchment_weight" not in dataset:
            raise KeyError(
                f"'catchment_weight' was not found in {filename}.\n"
                f"Available variables: {list(dataset.data_vars)}"
            )

        weights = dataset["catchment_weight"].load().astype("float32")

    weights.name = "catchment_weight"

    return weights


def convert_precipitation_to_mm(data):
    """Convert precipitation from metres to millimetres when necessary."""

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

    return data


def standardize_input_data(data):
    """
    Standardize coordinates shared by forecasts and hindcasts.

    Ensemble-member coordinates are stored as integers. Hindcast hdates are
    stored as integer YYYYMMDD values when present.
    """

    if "number" not in data.dims:
        raise KeyError(
            "The input data do not contain a 'number' dimension."
        )

    if "hdate" in data.dims and data.sizes["number"] == 11:
        member_numbers = np.arange(1, 12, dtype="int32")
    else:
        member_numbers = data["number"].values.astype("int32")

    data = data.assign_coords(number=member_numbers)

    if "hdate" in data.coords:

        hdate_values = data["hdate"].values

        if np.issubdtype(hdate_values.dtype, np.datetime64):
            hdate_values = (
                pd.to_datetime(hdate_values)
                .strftime("%Y%m%d")
                .astype("int32")
            )
        else:
            hdate_values = hdate_values.astype("int32")

        data = data.assign_coords(hdate=hdate_values)

    return data


def load_input_data(filename):
    """
    Load and standardize one forecast or hindcast variable.
    """

    with xr.open_dataset(filename) as dataset:
        data = dataset[variable].load()

    data = convert_precipitation_to_mm(data)
    data = standardize_input_data(data)

    return data


# =============================================================================
# Catchment processing
# =============================================================================

def catchment_mean(
    data,
    weights,
    spatial_dimensions=("latitude", "longitude"),
):
    """
    Calculate the catchment-weighted spatial mean.

    For each time, hindcast date, and ensemble member:

        catchment mean =
            sum(data * catchment weight) / sum(valid catchment weights)
    """

    missing_dimensions = [
        dimension
        for dimension in spatial_dimensions
        if dimension not in data.dims
    ]

    if missing_dimensions:
        raise ValueError(
            "The following spatial dimensions are missing from the data: "
            f"{missing_dimensions}"
        )

    valid = (
        np.isfinite(data)
        & np.isfinite(weights)
        & (weights > 0)
    )

    valid_weights = weights.where(valid)

    weighted_sum = (
        data.where(valid)
        * valid_weights
    ).sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    weight_sum = valid_weights.sum(
        dim=spatial_dimensions,
        skipna=True,
    )

    output = weighted_sum / weight_sum

    output.name = variable
    output.attrs["units"] = data.attrs.get("units", "mm")

    return output


def calculate_nday_accumulation(data):
    """
    Calculate trailing N-day accumulated precipitation.

    Each value is labelled by the final day of its accumulation period.
    """

    if data.sizes["time"] < x_days:
        raise ValueError(
            f"The input contains only {data.sizes['time']} time steps, "
            f"which is insufficient for a {x_days}-day accumulation."
        )

    accumulated = data.rolling(
        time=x_days,
        min_periods=x_days,
    ).sum()

    # Remove dates without a complete N-day accumulation.
    accumulated = accumulated.isel(
        time=slice(x_days - 1, None)
    )

    accumulated.name = "accumulated_value"
    accumulated.attrs["units"] = "mm"

    return accumulated


# =============================================================================
# Sample collection
# =============================================================================

def initialize_sample_collection():
    """
    Create an empty collection grouped by lead time.

    Each dictionary entry contains a list of accumulated precipitation values.
    """

    return defaultdict(list)


def collect_values(
    sample_collection,
    accumulated_data,
    forecast_date,
):
    """
    Store forecast or hindcast values valid in the selected month.

    Forecast data have dimensions:
        time, number

    Hindcast data have dimensions:
        time, hdate, number

    Both are stored in the same output distribution. The presence of the
    `hdate` dimension determines how valid dates are calculated.
    """

    accumulation_times = pd.DatetimeIndex(
        pd.to_datetime(accumulated_data["time"].values)
    ).normalize()

    lead_times = calculate_lead_times(
        accumulation_times,
        forecast_date,
    )

    is_hindcast = "hdate" in accumulated_data.dims

    if is_hindcast:
        hdates = hdate_values_to_timestamps(
            accumulated_data["hdate"].values
        )

    for time_index, lead_time in enumerate(lead_times):

        if is_hindcast:

            valid_dates = (
                hdates
                + pd.to_timedelta(int(lead_time), unit="D")
            )

            matching_hdates = np.where(
                valid_dates.month == valid_month
            )[0]

            if matching_hdates.size == 0:
                continue

            values = (
                accumulated_data
                .isel(
                    time=time_index,
                    hdate=matching_hdates,
                )
                .values
                .ravel()
            )

        else:

            valid_date = accumulation_times[time_index]

            if valid_date.month != valid_month:
                continue

            values = (
                accumulated_data
                .isel(time=time_index)
                .values
                .ravel()
            )

        values = values[np.isfinite(values)]

        if values.size > 0:
            sample_collection[int(lead_time)].extend(
                values.astype("float32").tolist()
            )


# =============================================================================
# Output dataset
# =============================================================================

def build_output_dataset(sample_collection):
    """
    Convert collected values into accumulated_value(lead_time, index).

    The index dimension is padded with NaN where different lead times contain
    different numbers of samples.
    """

    if not sample_collection:
        month_name = pd.Timestamp(
            2000,
            valid_month,
            1,
        ).strftime("%B")

        raise ValueError(
            f"No valid samples were collected for {month_name}."
        )

    lead_times = np.asarray(
        sorted(sample_collection),
        dtype="int32",
    )

    maximum_sample_count = max(
        len(values)
        for values in sample_collection.values()
    )

    accumulated_values = np.full(
        (len(lead_times), maximum_sample_count),
        np.nan,
        dtype="float32",
    )

    for lead_index, lead_time in enumerate(lead_times):

        values = np.asarray(
            sample_collection[int(lead_time)],
            dtype="float32",
        )

        accumulated_values[
            lead_index,
            :values.size,
        ] = values

    output = xr.Dataset(
        data_vars={
            "accumulated_value": (
                ("lead_time", "index"),
                accumulated_values,
            ),
        },
        coords={
            "lead_time": lead_times,
            "index": np.arange(
                maximum_sample_count,
                dtype="int32",
            ),
        },
    )

    add_output_metadata(output)

    return output


def add_output_metadata(output):
    """Add minimal metadata to the output dataset."""

    month_name = pd.Timestamp(
        2000,
        valid_month,
        1,
    ).strftime("%B")

    output.attrs["description"] = (
        f"Trailing {x_days}-day accumulated catchment-mean precipitation "
        f"valid in {month_name}, grouped by lead time."
    )
    output.attrs["variable"] = variable
    output.attrs["catchment"] = catchment
    output.attrs["accumulation_days"] = np.int32(x_days)
    output.attrs["valid_month"] = np.int32(valid_month)
    output.attrs["valid_month_name"] = month_name
    output.attrs["forecast_date_start"] = forecast_date_range[0]
    output.attrs["forecast_date_end"] = forecast_date_range[1]

    output["lead_time"].attrs["description"] = (
        "Days from initialization to the final day of the N-day accumulation"
    )
    output["lead_time"].attrs["units"] = "days"

    output["index"].attrs["description"] = (
        "Sample index within each lead-time distribution"
    )

    output["accumulated_value"].attrs["description"] = (
        f"Trailing {x_days}-day accumulated catchment-mean precipitation"
    )
    output["accumulated_value"].attrs["units"] = "mm"


# =============================================================================
# Diagnostics and writing
# =============================================================================

def print_output_summary(output, file_counts):
    """Print a concise processing and output summary."""

    month_name = pd.Timestamp(
        2000,
        valid_month,
        1,
    ).strftime("%B")

    print("====================================================")
    print("Output summary")
    print("====================================================")
    print(f"Valid month:                {valid_month:02d} ({month_name})")
    print(f"Input files processed:      {file_counts['processed']}")
    print(f"Input files skipped:        {file_counts['skipped']}")
    print(f"Number of lead times:       {output.sizes['lead_time']}")
    print(f"Maximum samples per lead:   {output.sizes['index']}")
    print(
        f"Lead-time range:            "
        f"{int(output.lead_time.min())}–"
        f"{int(output.lead_time.max())} days"
    )
    print(
        f"Total stored values:        "
        f"{int(np.isfinite(output['accumulated_value']).sum().values)}"
    )


def write_output(output):
    """Write the selected valid-month dataset to NetCDF."""

    path_out.mkdir(parents=True, exist_ok=True)

    filename_out = make_output_filename()

    encoding = {
        "accumulated_value": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
    }

    output.to_netcdf(
        filename_out,
        encoding=encoding,
    )

    print("\nWrote:")
    print(filename_out)


# =============================================================================
# Main processing
# =============================================================================

def main():
    """Process all relevant forecast and hindcast files identically."""

    validate_user_settings()

    forecast_dates = get_forecast_dates(
        date_range=forecast_date_range,
        option=forecast_date_option,
    )

    if not forecast_dates:
        raise ValueError(
            "No forecast initialization dates were found in the "
            "requested date range."
        )

    weights = load_weights(filename_weights)
    sample_collection = initialize_sample_collection()

    file_counts = {
        "processed": 0,
        "skipped": 0,
    }

    for initialization_number, forecast_date in enumerate(
        forecast_dates,
        start=1,
    ):

        print("\n====================================================")
        print(
            f"Checking {forecast_date} "
            f"({initialization_number}/{len(forecast_dates)})"
        )
        print("====================================================")

        forecast_filename, hindcast_filename = get_model_filenames(
            forecast_date
        )

        # Forecasts and hindcasts are handled through the same workflow.
        input_filenames = [
            forecast_filename,
            hindcast_filename,
        ]

        for filename in input_filenames:

            if not file_has_requested_month(
                filename=filename,
                forecast_date=forecast_date,
            ):
                print(f"Skipping:   {filename.name}")
                file_counts["skipped"] += 1
                continue

            print(f"Processing: {filename.name}")

            data = load_input_data(filename)
            data = catchment_mean(data, weights)
            data = calculate_nday_accumulation(data)

            collect_values(
                sample_collection=sample_collection,
                accumulated_data=data,
                forecast_date=forecast_date,
            )

            del data
            file_counts["processed"] += 1

    output = build_output_dataset(sample_collection)

    print_output_summary(
        output=output,
        file_counts=file_counts,
    )

    if write_to_file:
        write_output(output)


if __name__ == "__main__":
    main()
