"""
Preprocess ECMWF S2S daily precipitation for one catchment.

This script is a structured rewrite of the original preprocess_s2s.py.

For each selected forecast and hindcast file, it:

1. Reads daily S2S precipitation.
2. Converts valid forecast dates to integer lead days.
3. Sets negative precipitation values to zero.
4. Calculates a catchment-weighted and latitude-weighted spatial mean.
5. Converts precipitation from metres to millimetres.
6. Combines all forecast initializations.
7. Combines all acceptable hindcast initializations.
8. Writes forecast-only, hindcast-only, and combined NetCDF files.

No X-day accumulation is calculated here. The output contains daily
catchment-average precipitation indexed by lead_day, number, and i_date.

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
forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

# Input files must start with this text.
input_filename_prefix = (
    f"{variable}_0.5x0.5"
)

# Hindcast datasets whose minimum lead is below this value are excluded.
minimum_acceptable_hindcast_lead = 15

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
        "sipa_processed"
    ]
)


# =============================================================================
# Output filenames
# =============================================================================

filename_out_forecast = (
    path_out
    / (
        f"sipa_preprocessed_s2s_fc_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)

filename_out_hindcast = (
    path_out
    / (
        f"sipa_preprocessed_s2s_hc_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)

filename_out_combined = (
    path_out
    / (
        f"sipa_preprocessed_s2s_"
        f"{file_id}_"
        f"{forecast_date_label}.nc"
    )
)


# =============================================================================
# User-setting validation
# =============================================================================

def validate_user_settings():
    """Check settings before opening the input files."""

    start_date = np.datetime64(
        forecast_date_range[0]
    )

    end_date = np.datetime64(
        forecast_date_range[1]
    )

    if start_date > end_date:
        raise ValueError(
            "The first forecast_date_range value must not be later "
            "than the second value."
        )

    if minimum_acceptable_hindcast_lead < 0:
        raise ValueError(
            "minimum_acceptable_hindcast_lead must be non-negative."
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

def calculate_lead_days(
    valid_dates,
    initialization_date,
):
    """Convert valid forecast dates to integer lead days."""

    one_day = np.timedelta64(
        1,
        "D",
    )

    return (
        (
            valid_dates
            - initialization_date
        )
        / one_day
    ).astype(
        "int64"
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

    Forecast output:
        tp24(lead_day, number)
        f_date(lead_day)
        scalar i_date

    Hindcast output:
        tp24(i_date, lead_day, number)
        f_date(i_date, lead_day)
        i_date(i_date)

    For compatibility with the original preprocess_s2s.py, hindcast i_date
    values are synthetic unique timestamps close to the parent forecast
    initialization date. The first offset identifies the hindcast row, while
    a nanosecond offset distinguishes separate source files.
    """

    if "time" not in ds.coords:
        raise KeyError(
            "The input dataset does not contain a 'time' coordinate."
        )

    initialization_date = np.datetime64(
        initialization_date
    )

    lead_days = calculate_lead_days(
        valid_dates=ds[
            "time"
        ],
        initialization_date=initialization_date,
    )

    working = ds.assign_coords(
        f_date=ds[
            "time"
        ]
    )

    working = working.assign_coords(
        time=lead_days
    )

    spatial_mean = catchment_weighted_mean(
        ds=working,
        catchment_weight=catchment_weight,
    )

    spatial_mean = spatial_mean.rename(
        {
            "time": "lead_day",
        }
    )

    if hindcast:

        if "hdate" not in spatial_mean.dims:
            raise ValueError(
                "The hindcast dataset does not contain an 'hdate' dimension."
            )

        number_of_hdates = spatial_mean.sizes[
            "hdate"
        ]

        file_offset = np.timedelta64(
            hindcast_file_index,
            "ns",
        )

        row_offsets = np.arange(
            1,
            number_of_hdates + 1,
            dtype="timedelta64[us]",
        )

        synthetic_initialization_dates = (
            initialization_date
            + file_offset
            + row_offsets
        )

        spatial_mean = spatial_mean.assign_coords(
            hdate=synthetic_initialization_dates
        )

        spatial_mean = spatial_mean.rename(
            {
                "hdate": "i_date",
            }
        )

    else:

        spatial_mean = spatial_mean.assign_coords(
            i_date=initialization_date
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
# Hindcast quality control
# =============================================================================

def select_acceptable_hindcasts(
    processed_hindcasts,
):
    """
    Exclude hindcast datasets that begin before the accepted lead threshold.

    This reproduces the original preprocess_s2s.py quality-control rule.
    """

    accepted = []
    rejected_indices = []

    for index, ds in enumerate(
        processed_hindcasts
    ):

        minimum_lead = int(
            ds[
                "lead_day"
            ]
            .min()
            .values
        )

        if (
            minimum_lead
            < minimum_acceptable_hindcast_lead
        ):
            rejected_indices.append(
                index
            )

        else:
            accepted.append(
                ds
            )

    print()
    print(
        "Hindcast lead-time quality control"
    )
    print(
        "----------------------------------"
    )
    print(
        f"Accepted files: {len(accepted)}"
    )
    print(
        f"Rejected files: {len(rejected_indices)}"
    )

    if rejected_indices:
        print(
            "Rejected processed-file indices:",
            rejected_indices,
        )

    return accepted


# =============================================================================
# Dataset assembly
# =============================================================================

def concatenate_initializations(
    datasets,
    label,
):
    """Concatenate preprocessed datasets along i_date."""

    if not datasets:
        raise ValueError(
            f"No {label} datasets were available for concatenation."
        )

    combined = xr.concat(
        datasets,
        dim="i_date",
    )

    combined = combined.sortby(
        "i_date"
    )

    return combined


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


def add_output_metadata(
    ds,
    dataset_type,
):
    """Add descriptive global metadata without changing the data structure."""

    ds.attrs.update(
        {
            "description": (
                "Preprocessed ECMWF S2S daily catchment-average precipitation"
            ),
            "dataset_type": dataset_type,
            "variable": variable,
            "catchment": catchment,
            "forecast_initialization_start": forecast_date_range[0],
            "forecast_initialization_end": forecast_date_range[1],
            "minimum_acceptable_hindcast_lead": (
                minimum_acceptable_hindcast_lead
            ),
        }
    )


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
            "dtype": "float64",
            "_FillValue": np.nan,
        },
    }

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

    processed_hindcasts_all = preprocess_hindcast_files(
        filenames=hindcast_files,
        catchment_weight=catchment_weight,
    )


    processed_hindcasts = select_acceptable_hindcasts(
        processed_hindcasts_all
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
