#!/usr/bin/env python3  
"""
Preprocess ECMWF S2S daily precipitation for one catchment.

This script is a structured rewrite of the original preprocess_s2s.py.

For each selected forecast and hindcast file, it:

1. Reads daily S2S precipitation.
2. Keeps all 31 time steps in older files or the last 31 of 46 in newer files.
3. Relabels the retained time positions as lead days 16-46.
4. Assigns unique positional ensemble-member IDs while retaining all members.
5. Sets negative precipitation values to zero.
6. Calculates a catchment-weighted and latitude-weighted spatial mean.
7. Converts precipitation from metres to millimetres.
8. Combines all forecast initializations.
9. Combines all selected hindcast initializations.
10. Writes forecast-only, hindcast-only, and combined NetCDF files.

No X-day accumulation is calculated here. The output contains daily
catchment-average precipitation for lead days 16 through 46, indexed by
lead_day, number, and i_date.

Expected combined output structure
----------------------------------
dimensions:
    lead_day
    number
    i_date

variables:
    tp24(lead_day, number, i_date)
    f_date(i_date, lead_day)
    number(number)
    lead_day(lead_day)
    i_date(i_date)
    model_type(i_date)
    hdate(i_date)
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"

# Catchment name used in the NVE weight filename.
catchment = "regine_drammen"

# Inclusive forecast initialization-date range.
forecast_date_range = ["2020-01-02", "2023-12-28"]

# Inclusive lead-day range written to every output file.
#
# The resulting lead_day coordinate is:
#     16, 17, 18, ..., 46
first_lead_day = 16
last_lead_day = 46

# Input files must start with this text.
input_filename_prefix = f"{variable}_0.5x0.5"

# Write the forecast-only and hindcast-only files in addition to the
# combined output file.
write_separate_files = False

write2file = True


# =============================================================================
# Catchment and filename helpers
# =============================================================================

def get_file_id(
    catchment,
):
    """
    Return the short catchment label used in output filenames.

    Examples:
        regine_drammen -> drammen
        regine_glomma  -> glomma

    For any other catchment beginning with "regine_", the prefix is removed.
    """

    known_catchments = {
        "regine_drammen": "drammen",
        "regine_glomma": "glomma",
    }

    if catchment in known_catchments:
        return known_catchments[
            catchment
        ]

    if catchment.startswith(
        "regine_"
    ):
        return catchment.replace(
            "regine_",
            "",
            1,
        )

    return catchment


def get_forecast_date_label(
    forecast_date_range,
):
    """Return the selected initialization-date range for filenames."""

    return (
        f"{forecast_date_range[0]}_"
        f"{forecast_date_range[1]}"
    )


file_id = get_file_id(
    catchment
)

forecast_date_label = get_forecast_date_label(
    forecast_date_range
)


# =============================================================================
# Paths
# =============================================================================

path_in_forecast = Path(
    config.dirs[
        "s2s_forecast_daily"
    ]
) / variable

path_in_hindcast = Path(
    config.dirs[
        "s2s_hindcast_daily"
    ]
) / variable

filename_weights = Path(
    config.dirs[
        "nve"
    ]
) / (
    f"weights_catchment_"
    f"{catchment}_"
    f"era5_0.5x0.5.nc"
)

path_out = Path(
    config.dirs[
        "s2s_processed"
    ]
)


# =============================================================================
# Output filenames
# =============================================================================

filename_out_forecast = (
    path_out
    / (
        f"preprocessed_model_fc_{variable}_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)

filename_out_hindcast = (
    path_out
    / (
        f"preprocessed_model_hc_{variable}_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)

filename_out_combined = (
    path_out
    / (
        f"preprocessed_model_{variable}_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)


# =============================================================================
# User-setting validation
# =============================================================================

def validate_user_settings():
    """Check the user settings and required input paths."""

    start_date = np.datetime64(
        forecast_date_range[0]
    )

    end_date = np.datetime64(
        forecast_date_range[1]
    )

    if start_date > end_date:
        raise ValueError(
            "The forecast start date must not be later than the end date."
        )

    if first_lead_day > last_lead_day:
        raise ValueError(
            "first_lead_day must not be greater than last_lead_day."
        )

    if not path_in_forecast.is_dir():
        raise FileNotFoundError(
            f"Forecast directory not found: {path_in_forecast}"
        )

    if not path_in_hindcast.is_dir():
        raise FileNotFoundError(
            f"Hindcast directory not found: {path_in_hindcast}"
        )

    if not filename_weights.is_file():
        raise FileNotFoundError(
            f"Catchment-weight file not found: {filename_weights}"
        )


# =============================================================================
# Filename and date helpers
# =============================================================================

def initialization_date_from_filename(
    filename,
):
    """
    Extract the initialization date from a model filename.

    Expected filename ending:
        ..._YYYY-MM-DD.nc
    """

    date_text = (
        Path(filename)
        .stem
        .split("_")[-1]
    )

    try:
        return np.datetime64(
            date_text
        )

    except ValueError as error:
        raise ValueError(
            f"Could not extract an initialization date from "
            f"'{filename}'. Expected the filename to end in "
            f"'_YYYY-MM-DD.nc'."
        ) from error


def date_is_selected(
    initialization_date,
):
    """Return True when the initialization date is inside the selected range."""

    start_date = np.datetime64(
        forecast_date_range[0]
    )

    end_date = np.datetime64(
        forecast_date_range[1]
    )

    return bool(
        start_date
        <= initialization_date
        <= end_date
    )


def list_input_files(
    directory,
):
    """
    Return selected NetCDF files sorted by filename.

    Only files matching input_filename_prefix and forecast_date_range
    are retained.
    """

    files = []

    for filename in sorted(
        os.listdir(
            directory
        )
    ):

        if not filename.startswith(
            input_filename_prefix
        ):
            continue

        if not filename.endswith(
            ".nc"
        ):
            continue

        initialization_date = (
            initialization_date_from_filename(
                filename
            )
        )

        if date_is_selected(
            initialization_date
        ):
            files.append(
                filename
            )

    return files


# =============================================================================
# Catchment weights
# =============================================================================

def load_catchment_weights(
    filename,
):
    """Load the catchment-weight field into memory."""

    with xr.open_dataset(
        filename
    ) as ds:

        if "catchment_weight" not in ds:
            raise KeyError(
                f"'catchment_weight' was not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        weights = (
            ds[
                "catchment_weight"
            ]
            .load()
        )

    return weights


# =============================================================================
# Spatial averaging
# =============================================================================

def catchment_weighted_mean(
    ds,
    catchment_weight,
):
    """
    Calculate catchment- and latitude-weighted mean precipitation.

    This reproduces the original source.weighted_mean behavior:

    1. Replace negative tp24 values with zero.
    2. Calculate cosine-latitude area weights.
    3. Multiply area weights by catchment weights.
    4. Average over latitude and longitude.
    5. Convert precipitation from metres to millimetres.
    """

    if variable not in ds:
        raise KeyError(
            f"'{variable}' was not found in the model dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    for dimension in [
        "latitude",
        "longitude",
    ]:
        if dimension not in ds[
            variable
        ].dims:
            raise ValueError(
                f"'{variable}' does not contain the expected "
                f"'{dimension}' dimension. "
                f"Found dimensions: {ds[variable].dims}"
            )

    clean = ds.copy()

    clean[
        variable
    ] = xr.where(
        clean[
            variable
        ] < 0,
        0,
        clean[
            variable
        ],
    )

    latitude_weight = np.cos(
        np.deg2rad(
            clean[
                "latitude"
            ]
        )
    )

    combined_weight = (
        catchment_weight
        * latitude_weight
    )

    spatial_mean = (
        clean
        .weighted(
            combined_weight
        )
        .mean(
            [
                "latitude",
                "longitude",
            ]
        )
    )

    spatial_mean[
        variable
    ] = (
        spatial_mean[
            variable
        ]
        * 1000.0
    )

    spatial_mean[
        variable
    ].attrs[
        "units"
    ] = "mm"

    spatial_mean[
        variable
    ].attrs[
        "description"
    ] = (
        "Daily catchment- and latitude-weighted mean precipitation"
    )

    return spatial_mean


# =============================================================================
# Forecast/hindcast preprocessing
# =============================================================================

def normalize_time_dimension(ds):
    """
    Return the lead-day subset expected by the downstream processing.

    Older forecast and hindcast files contain 31 time steps corresponding to
    lead days 16-46. Newer files contain 46 time steps corresponding to lead
    days 1-46; for these files, the first 15 steps are discarded.
    """

    expected_times = last_lead_day - first_lead_day + 1
    number_of_times = ds.sizes.get("time", 0)

    if number_of_times == expected_times:
        return ds

    if number_of_times == last_lead_day:
        return ds.isel(time=slice(first_lead_day - 1, None))

    raise ValueError(
        f"Expected either {expected_times} time positions for lead days "
        f"{first_lead_day}-{last_lead_day} or {last_lead_day} positions for "
        f"lead days 1-{last_lead_day}, but found {number_of_times}."
    )


def normalize_ensemble_members(ds):
    """Assign unique positional ensemble-member IDs while preserving all members."""

    if "number" not in ds.dims:
        raise ValueError("The input dataset does not contain a 'number' dimension.")

    return ds.assign_coords(number=np.arange(ds.sizes["number"], dtype="int32"))


def calculate_lead_days(time_coordinate):
    """Map the retained time positions directly to lead days 16 through 46."""

    expected_times = last_lead_day - first_lead_day + 1

    if time_coordinate.size != expected_times:
        raise ValueError(
            f"Expected {expected_times} retained time positions for lead days "
            f"{first_lead_day}-{last_lead_day}, but found {time_coordinate.size}."
        )

    return xr.DataArray(
        np.arange(first_lead_day, last_lead_day + 1, dtype="int64"),
        dims=time_coordinate.dims,
    )


def preprocess_one_dataset(
    ds,
    initialization_date,
    catchment_weight,
    hindcast=False,
    hindcast_file_index=0,
):
    """
    Preprocess one forecast or hindcast dataset.

    Files with 31 time steps are used unchanged. Files with 46 time steps are
    reduced to their final 31 steps so that every output contains lead days
    16-46. Raw valid dates are retained as f_date.

    i_date is the unique coordinate used to combine forecast and hindcast
    initializations. model_type identifies the source. hdate stores the
    original hindcast date as YYYYMMDD and is 0 for forecast rows.
    """

    if "time" not in ds.coords:
        raise KeyError("The input dataset does not contain a 'time' coordinate.")

    ds = normalize_time_dimension(ds)
    ds = normalize_ensemble_members(ds)
    initialization_date = np.datetime64(initialization_date, "ns")
    lead_days = calculate_lead_days(ds["time"])

    # Preserve decoded valid dates from the retained raw time positions.
    forecast_dates = ds["time"].values.astype("datetime64[ns]")
    working = ds.assign_coords(f_date=("time", forecast_dates), time=lead_days)

    spatial_mean = catchment_weighted_mean(
        ds=working,
        catchment_weight=catchment_weight,
    ).rename({"time": "lead_day"})

    if hindcast:
        if "hdate" not in spatial_mean.dims:
            raise ValueError("The hindcast dataset does not contain an 'hdate' dimension.")

        original_hdates = spatial_mean["hdate"].values.astype("int32")
        number_of_hdates = spatial_mean.sizes["hdate"]

        # Preserve the existing synthetic i_date scheme used for concatenation.
        file_offset = np.timedelta64(hindcast_file_index, "ns")
        row_offsets = np.arange(1, number_of_hdates + 1, dtype="timedelta64[us]")
        synthetic_initialization_dates = initialization_date + file_offset + row_offsets

        spatial_mean = (
            spatial_mean.assign_coords(hdate=synthetic_initialization_dates)
            .rename({"hdate": "i_date"})
        )

        spatial_mean["model_type"] = xr.DataArray(
            np.full(number_of_hdates, "hindcast", dtype="U8"),
            dims=("i_date",),
            coords={"i_date": spatial_mean["i_date"]},
        )
        spatial_mean["hdate"] = xr.DataArray(
            original_hdates,
            dims=("i_date",),
            coords={"i_date": spatial_mean["i_date"]},
        )

    else:
        spatial_mean = spatial_mean.expand_dims(i_date=[initialization_date])
        spatial_mean["model_type"] = xr.DataArray(
            np.array(["forecast"], dtype="U8"),
            dims=("i_date",),
            coords={"i_date": spatial_mean["i_date"]},
        )
        spatial_mean["hdate"] = xr.DataArray(
            np.array([0], dtype="int32"),
            dims=("i_date",),
            coords={"i_date": spatial_mean["i_date"]},
        )

    return spatial_mean


# =============================================================================
# File processing
# =============================================================================

def preprocess_forecast_files(
    filenames,
    catchment_weight,
):
    """Preprocess all selected forecast files."""

    processed = []

    for file_number, filename in enumerate(
        filenames,
        start=1,
    ):

        initialization_date = (
            initialization_date_from_filename(
                filename
            )
        )

        full_path = (
            path_in_forecast
            / filename
        )

        print(
            f"Forecast {file_number:>4}/{len(filenames)}: "
            f"{filename}"
        )

        with xr.open_dataset(
            full_path
        ) as opened:

            ds = opened.load()

        result = preprocess_one_dataset(
            ds=ds,
            initialization_date=initialization_date,
            catchment_weight=catchment_weight,
            hindcast=False,
        )

        processed.append(
            result
        )

    return processed


def preprocess_hindcast_files(
    filenames,
    catchment_weight,
):
    """Preprocess all selected hindcast files."""

    processed = []

    for file_index, filename in enumerate(
        filenames
    ):

        initialization_date = (
            initialization_date_from_filename(
                filename
            )
        )

        full_path = (
            path_in_hindcast
            / filename
        )

        print(
            f"Hindcast {file_index + 1:>4}/{len(filenames)}: "
            f"{filename}"
        )

        with xr.open_dataset(
            full_path
        ) as opened:

            ds = opened.load()

        result = preprocess_one_dataset(
            ds=ds,
            initialization_date=initialization_date,
            catchment_weight=catchment_weight,
            hindcast=True,
            hindcast_file_index=file_index,
        )

        processed.append(
            result
        )

    return processed


# =============================================================================
# Dataset assembly
# =============================================================================

def concatenate_initializations(
    datasets,
    label,
):
    """
    Concatenate preprocessed datasets along i_date.

    The input-list order is preserved deliberately to match the original
    preprocess_s2s.py organization. No chronological sorting is applied.

    Consequently:

    - the forecast-only output follows forecast-file order;
    - the hindcast-only output follows hindcast-file order;
    - the combined output contains all forecasts first, followed by all
      hindcasts.
    """

    if not datasets:
        raise ValueError(
            f"No {label} datasets were available for concatenation."
        )

    return xr.concat(
        datasets,
        dim="i_date",
        join="outer",
    )


def build_output_datasets(
    processed_forecasts,
    processed_hindcasts,
):
    """Build forecast-only, hindcast-only, and combined datasets."""

    forecast = concatenate_initializations(
        datasets=processed_forecasts,
        label="forecast",
    )

    hindcast = concatenate_initializations(
        datasets=processed_hindcasts,
        label="hindcast",
    )

    combined = concatenate_initializations(
        datasets=(
            processed_forecasts
            + processed_hindcasts
        ),
        label="forecast/hindcast",
    )

    add_output_metadata(
        forecast,
        dataset_type="forecast",
    )

    add_output_metadata(
        hindcast,
        dataset_type="hindcast",
    )

    add_output_metadata(
        combined,
        dataset_type="forecast and hindcast",
    )

    return (
        forecast,
        hindcast,
        combined,
    )


def add_output_metadata(ds, dataset_type):
    """Add concise global and variable metadata without changing data values."""

    ds.attrs.update(
        {
            "description": "Preprocessed ECMWF S2S daily catchment-average precipitation",
            "dataset_type": dataset_type,
            "variable": variable,
            "catchment": catchment,
            "forecast_initialization_start": forecast_date_range[0],
            "forecast_initialization_end": forecast_date_range[1],
            "first_lead_day": first_lead_day,
            "last_lead_day": last_lead_day,
        }
    )

    variable_descriptions = {
        variable: "Daily catchment- and latitude-weighted mean precipitation.",
        "f_date": "Forecast valid date for each initialization and lead day.",
        "number": "Ensemble member identifier.",
        "lead_day": "Forecast lead time in days.",
        "i_date": "Unique initialization coordinate used to combine forecast and hindcast rows.",
        "model_type": "Source type for each initialization: forecast or hindcast.",
        "hdate": "Original hindcast initialization date as YYYYMMDD; 0 for forecast rows.",
    }

    for name, description in variable_descriptions.items():
        if name in ds.variables:
            ds[name].attrs["description"] = description


# =============================================================================
# Output checks
# =============================================================================

def print_dataset_summary(
    ds,
    label,
):
    """Print the dimensions and date coverage of one output dataset."""

    print()
    print(
        label
    )
    print(
        "-" * len(
            label
        )
    )
    print(
        ds
    )

    print(
        "Initialization count:",
        ds.sizes.get(
            "i_date",
            0,
        ),
    )

    print(
        "Lead-day count:",
        ds.sizes.get(
            "lead_day",
            0,
        ),
    )

    print(
        "Ensemble-member count:",
        ds.sizes.get(
            "number",
            0,
        ),
    )

    if "i_date" in ds.coords:
        print(
            "First i_date:",
            ds[
                "i_date"
            ]
            .min()
            .values,
        )

        print(
            "Last i_date:",
            ds[
                "i_date"
            ]
            .max()
            .values,
        )


# =============================================================================
# NetCDF writing
# =============================================================================

def write_output(
    forecast,
    hindcast,
    combined,
):
    """Write the same three NetCDF products as the original script."""

    path_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {
        variable: {
            "dtype": "float32",
            "_FillValue": np.float32(
                np.nan
            ),
        },
    }

    encoding["hdate"] = {"dtype": "int32"}

    if write_separate_files:

        forecast.to_netcdf(
            filename_out_forecast,
            encoding=encoding,
        )

        print(
            "Wrote:",
            filename_out_forecast,
        )

        hindcast.to_netcdf(
            filename_out_hindcast,
            encoding=encoding,
        )

        print(
            "Wrote:",
            filename_out_hindcast,
        )

    combined.to_netcdf(
        filename_out_combined,
        encoding=encoding,
    )

    print(
        "Wrote:",
        filename_out_combined,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()


    forecast_files = list_input_files(
        path_in_forecast
    )

    hindcast_files = list_input_files(
        path_in_hindcast
    )


    print(
        "Input summary"
    )
    print(
        "-------------"
    )
    print(
        "Forecast directory:",
        path_in_forecast,
    )
    print(
        "Hindcast directory:",
        path_in_hindcast,
    )
    print(
        "Catchment:",
        catchment,
    )
    print(
        "Output file ID:",
        file_id,
    )
    print(
        "Catchment weights:",
        filename_weights,
    )
    print(
        "Selected forecast files:",
        len(
            forecast_files
        ),
    )
    print(
        "Selected hindcast files:",
        len(
            hindcast_files
        ),
    )
    print(
        "Initialization-date range:",
        (
            f"{forecast_date_range[0]} "
            f"to {forecast_date_range[1]}"
        ),
    )
    print(
        "Output lead-day range:",
        (
            f"{first_lead_day} "
            f"to {last_lead_day}"
        ),
    )


    if not forecast_files:
        raise FileNotFoundError(
            "No forecast files matched the selected settings."
        )

    if not hindcast_files:
        raise FileNotFoundError(
            "No hindcast files matched the selected settings."
        )


    catchment_weight = load_catchment_weights(
        filename_weights
    )


    processed_forecasts = preprocess_forecast_files(
        filenames=forecast_files,
        catchment_weight=catchment_weight,
    )

    processed_hindcasts = preprocess_hindcast_files(
        filenames=hindcast_files,
        catchment_weight=catchment_weight,
    )


    (
        output_forecast,
        output_hindcast,
        output_combined,
    ) = build_output_datasets(
        processed_forecasts=processed_forecasts,
        processed_hindcasts=processed_hindcasts,
    )


    print_dataset_summary(
        output_forecast,
        "Forecast output",
    )

    print_dataset_summary(
        output_hindcast,
        "Hindcast output",
    )

    print_dataset_summary(
        output_combined,
        "Combined output",
    )


    if write2file:

        write_output(
            forecast=output_forecast,
            hindcast=output_hindcast,
            combined=output_combined,
        )
