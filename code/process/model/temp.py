"""
Build a lead-time-dependent distribution of accumulated precipitation for one
selected valid month from S2S forecasts and hindcasts.

The script processes one valid calendar month per run. For every forecast
initialization date, it:

1. Loads forecast and hindcast daily precipitation.
2. Computes catchment-weighted mean precipitation.
3. Computes trailing N-day accumulated precipitation.
4. Retains the accumulated value at every lead time.
5. Keeps only samples whose accumulation end date falls in `valid_month`.
6. Stores all retained samples with dimensions:

       accumulated_value(lead_time, index)

The `index` dimension contains samples from different forecast initialization
dates, hindcast dates, and ensemble members.

Running one month at a time reduces peak memory use compared with constructing
all 12 valid months in one dataset.
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

# Forecast initialization dates to process.
forecast_date_range = ["2020-01-02","2020-01-09"] #["2020-01-02","2023-06-26"]

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
# Validation
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
        raise ValueError(
            "option must be either 'mt' or 'all'."
        )

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


def standardize_hindcast_hdates(hindcast):
    """
    Store hindcast initialization dates as integer YYYYMMDD values.
    """

    if "hdate" not in hindcast.coords:
        raise KeyError(
            "The hindcast data do not contain an 'hdate' coordinate."
        )

    hdate_values = hindcast["hdate"].values

    if np.issubdtype(hdate_values.dtype, np.datetime64):
        hdate_values = (
            pd.to_datetime(hdate_values)
            .strftime("%Y%m%d")
            .astype("int32")
        )
    else:
        hdate_values = hdate_values.astype("int32")

    return hindcast.assign_coords(hdate=hdate_values)


def standardize_ensemble_members(data, model_type):
    """Ensure that ensemble-member coordinates are stored as integers."""

    if "number" not in data.dims:
        raise KeyError(
            f"The {model_type} data do not contain a 'number' dimension."
        )

    if model_type == "hindcast" and data.sizes["number"] == 11:
        member_numbers = np.arange(1, 12, dtype="int32")
    else:
        member_numbers = data["number"].values.astype("int32")

    return data.assign_coords(number=member_numbers)


def load_model_data(forecast_date):
    """
    Load forecast and hindcast precipitation for one initialization date.
    """

    forecast_filename, hindcast_filename = get_model_filenames(
        forecast_date
    )

    if not forecast_filename.exists():
        raise FileNotFoundError(
            f"Forecast file not found:\n{forecast_filename}"
        )

    if not hindcast_filename.exists():
        raise FileNotFoundError(
            f"Hindcast file not found:\n{hindcast_filename}"
        )

    with xr.open_dataset(forecast_filename) as dataset:
        forecast = dataset[variable].load()

    with xr.open_dataset(hindcast_filename) as dataset:
        hindcast = dataset[variable].load()

    forecast = convert_precipitation_to_mm(forecast)
    hindcast = convert_precipitation_to_mm(hindcast)

    forecast = standardize_ensemble_members(
        forecast,
        model_type="forecast",
    )

    hindcast = standardize_ensemble_members(
        hindcast,
        model_type="hindcast",
    )

    hindcast = standardize_hindcast_hdates(hindcast)

    return forecast, hindcast


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
    output.attrs["description"] = (
        "Catchment-weighted daily mean precipitation"
    )
    output.attrs["units"] = data.attrs.get("units", "mm")

    return output


def calculate_nday_accumulation(data, number_of_days):
    """
    Calculate trailing N-day accumulated precipitation.

    Each output value is labelled by the final day of its accumulation period.
    For a two-day accumulation, the value labelled 10 May is the sum of
    precipitation on 9 May and 10 May.
    """

    if data.sizes["time"] < number_of_days:
        raise ValueError(
            f"The input contains only {data.sizes['time']} time steps, "
            f"which is insufficient for a {number_of_days}-day accumulation."
        )

    accumulated = data.rolling(
        time=number_of_days,
        min_periods=number_of_days,
    ).sum()

    # Remove the first N - 1 entries because they do not contain complete
    # accumulation periods.
    accumulated = accumulated.isel(
        time=slice(number_of_days - 1, None)
    )

    accumulated.name = "accumulated_value"
    accumulated.attrs["description"] = (
        f"Trailing {number_of_days}-day accumulated "
        "catchment-weighted mean precipitation"
    )
    accumulated.attrs["units"] = "mm"

    return accumulated


# =============================================================================
# Lead times and valid dates
# =============================================================================

def calculate_lead_times(time_coordinate, forecast_date):
    """
    Calculate lead time in whole days.

    Lead time is defined as:

        accumulation end date - forecast initialization date
    """

    initialization_date = pd.Timestamp(forecast_date).normalize()

    valid_dates = pd.DatetimeIndex(
        pd.to_datetime(time_coordinate.values)
    ).normalize()

    lead_times = (
        valid_dates - initialization_date
    ).days.astype("int32")

    if np.any(lead_times < 0):
        raise ValueError(
            f"Negative lead times were found for initialization "
            f"{forecast_date}."
        )

    return lead_times


def hdate_integer_to_timestamp(hdate):
    """Convert an integer YYYYMMDD hindcast date to a pandas Timestamp."""

    return pd.to_datetime(
        str(int(hdate)),
        format="%Y%m%d",
    ).normalize()


# =============================================================================
# Sample collection
# =============================================================================

def initialize_sample_collection():
    """
    Create an empty collection grouped by lead time.

    Only samples whose valid date falls in `valid_month` are added.
    """

    return defaultdict(list)


def make_sample_record(
    accumulated_value,
    valid_date,
    forecast_date,
    hdate,
    ensemble_member,
    model_type,
    source_file,
):
    """Create one sample record."""

    return {
        "accumulated_value": np.float32(accumulated_value),
        "valid_date": np.datetime64(valid_date, "ns"),
        "forecast_date": np.datetime64(forecast_date, "ns"),
        "hdate": np.int32(hdate),
        "ensemble_member": np.int32(ensemble_member),
        "model_type": str(model_type),
        "source_file": str(source_file),
    }


def collect_forecast_samples(
    sample_collection,
    accumulated_forecast,
    forecast_date,
    source_file,
):
    """
    Add forecast samples valid in the selected month.

    One sample is retained for every ensemble member and lead time whose
    accumulation end date falls in `valid_month`.
    """

    lead_times = calculate_lead_times(
        accumulated_forecast["time"],
        forecast_date,
    )

    ensemble_members = (
        accumulated_forecast["number"]
        .values
        .astype("int32")
    )

    initialization_date = pd.Timestamp(forecast_date).normalize()

    for time_index, lead_time in enumerate(lead_times):

        valid_date = pd.Timestamp(
            accumulated_forecast["time"].values[time_index]
        ).normalize()

        if valid_date.month != valid_month:
            continue

        values = (
            accumulated_forecast
            .isel(time=time_index)
            .transpose("number")
            .values
        )

        for member_index, member in enumerate(ensemble_members):

            value = values[member_index]

            if not np.isfinite(value):
                continue

            record = make_sample_record(
                accumulated_value=value,
                valid_date=valid_date,
                forecast_date=initialization_date,
                hdate=-99999999,
                ensemble_member=member,
                model_type="forecast",
                source_file=source_file,
            )

            sample_collection[int(lead_time)].append(record)


def collect_hindcast_samples(
    sample_collection,
    accumulated_hindcast,
    forecast_date,
    source_file,
):
    """
    Add hindcast samples valid in the selected month.

    For each historical hindcast initialization date:

        valid date = hdate + lead time

    The valid month is therefore determined separately for every hdate.
    """

    required_dimensions = {"time", "hdate", "number"}

    missing_dimensions = (
        required_dimensions
        - set(accumulated_hindcast.dims)
    )

    if missing_dimensions:
        raise ValueError(
            "The hindcast data are missing required dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    lead_times = calculate_lead_times(
        accumulated_hindcast["time"],
        forecast_date,
    )

    hdates = (
        accumulated_hindcast["hdate"]
        .values
        .astype("int32")
    )

    ensemble_members = (
        accumulated_hindcast["number"]
        .values
        .astype("int32")
    )

    archive_initialization_date = (
        pd.Timestamp(forecast_date).normalize()
    )

    for time_index, lead_time in enumerate(lead_times):

        data_at_lead = (
            accumulated_hindcast
            .isel(time=time_index)
            .transpose("hdate", "number")
        )

        for hdate_index, hdate in enumerate(hdates):

            hindcast_initialization_date = (
                hdate_integer_to_timestamp(hdate)
            )

            valid_date = (
                hindcast_initialization_date
                + pd.Timedelta(days=int(lead_time))
            )

            if valid_date.month != valid_month:
                continue

            values = data_at_lead.isel(
                hdate=hdate_index
            ).values

            for member_index, member in enumerate(ensemble_members):

                value = values[member_index]

                if not np.isfinite(value):
                    continue

                record = make_sample_record(
                    accumulated_value=value,
                    valid_date=valid_date,
                    forecast_date=archive_initialization_date,
                    hdate=hdate,
                    ensemble_member=member,
                    model_type="hindcast",
                    source_file=source_file,
                )

                sample_collection[int(lead_time)].append(record)


# =============================================================================
# Output dataset
# =============================================================================

def get_output_dimensions(sample_collection):
    """Determine the lead-time coordinate and required index length."""

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

    maximum_number_of_samples = max(
        len(records)
        for records in sample_collection.values()
    )

    return lead_times, maximum_number_of_samples


def initialize_output_dataset(lead_times, number_of_indices):
    """
    Create an empty output dataset with dimensions (lead_time, index).
    """

    shape = (
        len(lead_times),
        number_of_indices,
    )

    output = xr.Dataset(
        data_vars={
            "accumulated_value": (
                ("lead_time", "index"),
                np.full(shape, np.nan, dtype="float32"),
            ),
            "valid_date": (
                ("lead_time", "index"),
                np.full(
                    shape,
                    np.datetime64("NaT"),
                    dtype="datetime64[ns]",
                ),
            ),
            "forecast_date": (
                ("lead_time", "index"),
                np.full(
                    shape,
                    np.datetime64("NaT"),
                    dtype="datetime64[ns]",
                ),
            ),
            "hdate": (
                ("lead_time", "index"),
                np.full(
                    shape,
                    -99999999,
                    dtype="int32",
                ),
            ),
            "ensemble_member": (
                ("lead_time", "index"),
                np.full(
                    shape,
                    -999,
                    dtype="int32",
                ),
            ),
            "model_type": (
                ("lead_time", "index"),
                np.full(shape, "", dtype=object),
            ),
            "source_file": (
                ("lead_time", "index"),
                np.full(shape, "", dtype=object),
            ),
            "sample_count": (
                ("lead_time",),
                np.zeros(
                    len(lead_times),
                    dtype="int32",
                ),
            ),
        },
        coords={
            "lead_time": lead_times,
            "index": np.arange(
                number_of_indices,
                dtype="int32",
            ),
        },
    )

    add_output_metadata(output)

    return output


def add_output_metadata(output):
    """Add descriptions and units to the output dataset."""

    month_name = pd.Timestamp(
        2000,
        valid_month,
        1,
    ).strftime("%B")

    output.attrs["description"] = (
        "Lead-time-dependent distributions of trailing "
        f"{x_days}-day accumulated catchment-mean precipitation "
        f"valid in {month_name}."
    )

    output.attrs["variable"] = variable
    output.attrs["catchment"] = catchment
    output.attrs["accumulation_days"] = np.int32(x_days)
    output.attrs["valid_month"] = np.int32(valid_month)
    output.attrs["valid_month_name"] = month_name
    output.attrs["forecast_date_start"] = forecast_date_range[0]
    output.attrs["forecast_date_end"] = forecast_date_range[1]

    output["lead_time"].attrs["description"] = (
        "Days from initialization to the final day of the "
        "N-day accumulation"
    )
    output["lead_time"].attrs["units"] = "days"

    output["index"].attrs["description"] = (
        "Sample index within each lead-time distribution"
    )

    output["accumulated_value"].attrs["description"] = (
        f"Trailing {x_days}-day accumulated catchment-mean precipitation"
    )
    output["accumulated_value"].attrs["units"] = "mm"

    output["valid_date"].attrs["description"] = (
        "Final date of the N-day accumulation"
    )

    output["forecast_date"].attrs["description"] = (
        "Archive forecast initialization date"
    )

    output["hdate"].attrs["description"] = (
        "Hindcast initialization date as YYYYMMDD; "
        "-99999999 for forecasts"
    )
    output["hdate"].attrs["_FillValue"] = np.int32(-99999999)

    output["ensemble_member"].attrs["description"] = (
        "Forecast or hindcast ensemble-member number"
    )
    output["ensemble_member"].attrs["_FillValue"] = np.int32(-999)

    output["model_type"].attrs["description"] = (
        "Sample type: 'forecast' or 'hindcast'"
    )

    output["source_file"].attrs["description"] = (
        "Input NetCDF file containing the sample"
    )

    output["sample_count"].attrs["description"] = (
        "Number of valid samples stored at each lead time"
    )


def fill_output_dataset(output, sample_collection):
    """Copy the collected sample records into the output dataset."""

    for lead_time, records in sample_collection.items():

        number_of_samples = len(records)

        if number_of_samples == 0:
            continue

        index_values = np.arange(
            number_of_samples,
            dtype="int32",
        )

        selection = {
            "lead_time": lead_time,
            "index": index_values,
        }

        output["accumulated_value"].loc[selection] = np.asarray(
            [
                record["accumulated_value"]
                for record in records
            ],
            dtype="float32",
        )

        output["valid_date"].loc[selection] = np.asarray(
            [
                record["valid_date"]
                for record in records
            ],
            dtype="datetime64[ns]",
        )

        output["forecast_date"].loc[selection] = np.asarray(
            [
                record["forecast_date"]
                for record in records
            ],
            dtype="datetime64[ns]",
        )

        output["hdate"].loc[selection] = np.asarray(
            [
                record["hdate"]
                for record in records
            ],
            dtype="int32",
        )

        output["ensemble_member"].loc[selection] = np.asarray(
            [
                record["ensemble_member"]
                for record in records
            ],
            dtype="int32",
        )

        output["model_type"].loc[selection] = np.asarray(
            [
                record["model_type"]
                for record in records
            ],
            dtype=object,
        )

        output["source_file"].loc[selection] = np.asarray(
            [
                record["source_file"]
                for record in records
            ],
            dtype=object,
        )

        output["sample_count"].loc[
            {"lead_time": lead_time}
        ] = number_of_samples

    return output


def build_output_dataset(sample_collection):
    """Convert collected samples into the final xarray dataset."""

    lead_times, number_of_indices = get_output_dimensions(
        sample_collection
    )

    output = initialize_output_dataset(
        lead_times=lead_times,
        number_of_indices=number_of_indices,
    )

    return fill_output_dataset(
        output=output,
        sample_collection=sample_collection,
    )


# =============================================================================
# Diagnostics and writing
# =============================================================================

def print_output_summary(output):
    """Print a concise summary of the generated distributions."""

    month_name = pd.Timestamp(
        2000,
        valid_month,
        1,
    ).strftime("%B")

    print("\n====================================================")
    print("Output summary")
    print("====================================================")
    print(f"Valid month:  {valid_month:02d} ({month_name})")
    print(f"Lead times:   {output.sizes['lead_time']}")
    print(f"Index size:   {output.sizes['index']}")
    print(
        f"Lead range:   "
        f"{int(output.lead_time.min())}–"
        f"{int(output.lead_time.max())} days"
    )
    print(
        f"Total values: "
        f"{int(output['sample_count'].sum().values)}"
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
        },
        "hdate": {
            "zlib": True,
            "complevel": 4,
            "dtype": "int32",
        },
        "ensemble_member": {
            "zlib": True,
            "complevel": 4,
            "dtype": "int32",
        },
        "sample_count": {
            "zlib": True,
            "complevel": 4,
            "dtype": "int32",
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
    """Process all requested initialization dates for one valid month."""

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

    for initialization_number, forecast_date in enumerate(
        forecast_dates,
        start=1,
    ):

        print("\n====================================================")
        print(
            f"Processing {forecast_date} "
            f"({initialization_number}/{len(forecast_dates)})"
        )
        print("====================================================")

        forecast_filename, hindcast_filename = get_model_filenames(
            forecast_date
        )

        forecast, hindcast = load_model_data(forecast_date)

        forecast = catchment_mean(
            data=forecast,
            weights=weights,
        )

        hindcast = catchment_mean(
            data=hindcast,
            weights=weights,
        )

        forecast_accumulated = calculate_nday_accumulation(
            data=forecast,
            number_of_days=x_days,
        )

        hindcast_accumulated = calculate_nday_accumulation(
            data=hindcast,
            number_of_days=x_days,
        )

        collect_forecast_samples(
            sample_collection=sample_collection,
            accumulated_forecast=forecast_accumulated,
            forecast_date=forecast_date,
            source_file=forecast_filename,
        )

        collect_hindcast_samples(
            sample_collection=sample_collection,
            accumulated_hindcast=hindcast_accumulated,
            forecast_date=forecast_date,
            source_file=hindcast_filename,
        )

        # Explicitly release the large arrays before the next initialization.
        del forecast
        del hindcast
        del forecast_accumulated
        del hindcast_accumulated

    output = build_output_dataset(sample_collection)

    print_output_summary(output)

    if write_to_file:
        write_output(output)


if __name__ == "__main__":
    main()
