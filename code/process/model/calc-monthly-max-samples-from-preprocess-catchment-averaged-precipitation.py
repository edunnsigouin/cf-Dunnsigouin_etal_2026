#!/usr/bin/env python3
"""
Create monthly precipitation-maximum samples from one preprocessed S2S file.

The input can be either:

1. raw daily forecast/hindcast data produced by the preprocessing script; or
2. an already X-day-accumulated bias-corrected file produced by the
   bias-correction script.

Both input types are expected to contain:

    tp24(lead_day, number, i_date)
    f_date(i_date, lead_day)
    model_type(i_date)
    hdate(i_date)

This script does not repeat the spatial averaging or bias correction.

For raw input, the script calculates the requested trailing X-day
accumulation. For bias-corrected input, the accumulation has already been
performed, so the script does not accumulate the data again.

Calendar-month assignment
-------------------------
The calendar month is always assigned from the original daily valid dates for
lead days 16-46. In other words, month assignment always uses the N=1 forecast
window, even when the precipitation maxima are calculated from 2-day, 3-day,
or another N-day accumulation.

For each i_date:

1. Read the 31 daily valid dates corresponding to lead days 16-46.
2. Count how many of those dates fall in each calendar month.
3. Assign the initialization to the month containing the most dates.

Because there are 31 daily lead dates, there must always be one calendar month
with a strict majority. A tie cannot occur.

This gives every forecast/hindcast initialization one fixed calendar-month
assignment that does not change when accumulation_days changes.

The precipitation calculation remains accumulation-dependent. For example:

    input daily leads = 16-46
    accumulation_days = 2
    usable accumulated ending leads = 17-46

The month is assigned using daily leads 16-46, but the 2-day maximum is
calculated only over accumulated ending leads 17-46.

Workflow
--------
1. Read either raw daily or already accumulated bias-corrected
   catchment-mean precipitation.
2. Assign each i_date to a calendar month using the 31 valid dates stored in
   f_date for lead days 16-46.
3. For raw input, calculate trailing N-day precipitation accumulations along
   lead_day. For bias-corrected input, use the existing N-day accumulation.
4. For every (number, i_date), find the maximum accumulated precipitation over
   the complete usable accumulated lead range.
5. Record the valid calendar date and ending lead day of that maximum.
6. Copy each complete-range maximum into exactly one lead-bin variable,
   according to the ending lead day at which the maximum occurred.
7. Write one compact NetCDF sample file.

Lead-bin example
----------------
For:

    accumulation_days = 2
    number_of_lead_bins = 2

the usable accumulated ending leads are 17-46, and the bins are:

    17-31
    32-46

The full-window maximum is calculated once. It is then placed into the one
lead-bin variable corresponding to its lead_of_max. The other lead-bin
variables remain NaN at that same (number, i_date) position.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"

catchment = "regine_drammen"

forecast_date_range = [
    "2020-01-02",
    "2022-12-29",
]

# Number of consecutive daily values included in each trailing accumulation.
#
# Examples:
#     1 -> daily precipitation
#     2 -> two-day accumulated precipitation
#     3 -> three-day accumulated precipitation
accumulation_days = 2

# Number of consecutive lead-location bins used to partition the full-window
# maxima according to where each maximum occurs.
number_of_lead_bins = 2

# Input type.
#
# "raw":
#     Read daily preprocessed model data and calculate accumulation_days here.
#
# "bias_corrected":
#     Read an already accumulated and bias-corrected model file. No additional
#     rolling accumulation is performed.
input_data_type = "bias_corrected"

# Settings used only when input_data_type == "bias_corrected".
#
# bias_correction_method options:
#     "q", "doy", "ld", "q_doy"
#
# bias_correction_reference options:
#     "senorge", "era5"
bias_correction_method = "q_doy"
bias_correction_reference = "senorge"

# Original accumulation already contained in the bias-corrected input file.
# This must equal accumulation_days because no further accumulation is applied.
bias_corrected_input_x_days = 2

# Optional explicit input filename.
#
# Leave as None to use the automatically generated filename.
input_filename_override = None

# Optional explicit output filename.
#
# Leave as None to use the automatically generated filename.
output_filename_override = None

write2file = True


# =============================================================================
# Input lead-day settings
# =============================================================================

# These must match the preprocessed input file.
first_input_lead = 16
last_input_lead = 46


# =============================================================================
# Paths and filenames
# =============================================================================

path_in = Path(
    config.dirs[
        "s2s_processed"
    ]
)

path_out = Path(
    config.dirs[
        "s2s_processed"
    ]
)


def get_file_id(
    catchment_name,
):
    """Return the short catchment label used in filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


def make_input_filename():
    """Return the selected raw or bias-corrected input filename."""

    if input_filename_override is not None:
        return Path(
            input_filename_override
        )

    stem = (
        f"preprocessed_model_{variable}_"
        f"{get_file_id(catchment)}_"
        f"{forecast_date_range[0]}_"
        f"{forecast_date_range[1]}"
    )

    if input_data_type == "raw":

        return (
            path_in
            / f"{stem}.nc"
        )

    if input_data_type == "bias_corrected":

        return (
            path_in
            / (
                f"{stem}_"
                f"{bias_corrected_input_x_days}dayacc_"
                f"bc_{bias_correction_method}_"
                f"{bias_correction_reference}.nc"
            )
        )

    raise ValueError(
        "input_data_type must be 'raw' or 'bias_corrected'."
    )


def make_output_filename(
    lead_bins,
):
    """Return the monthly-sample output filename."""

    if output_filename_override is not None:
        return Path(
            output_filename_override
        )

    first_usable_lead = (
        first_input_lead
        + accumulation_days
        - 1
    )

    bin_label = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in lead_bins
    )

    source_label = ""

    if input_data_type == "bias_corrected":

        source_label = (
            f"_bc_{bias_correction_method}_"
            f"{bias_correction_reference}"
        )

    return (
        path_out
        / (
            f"monthly_max_samples_{variable}_"
            f"{accumulation_days}dayacc_"
            f"{get_file_id(catchment)}_"
            f"lead{first_usable_lead}-{last_input_lead}_"
            f"split{number_of_lead_bins}_{bin_label}_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}"
            f"{source_label}.nc"
        )
    )


# =============================================================================
# Validation
# =============================================================================

def validate_user_settings():
    """Check the user settings before reading the input file."""

    valid_input_types = {
        "raw",
        "bias_corrected",
    }

    if input_data_type not in valid_input_types:
        raise ValueError(
            f"input_data_type must be one of {sorted(valid_input_types)}."
        )

    if input_data_type == "bias_corrected":

        valid_methods = {
            "q",
            "doy",
            "ld",
            "q_doy",
        }

        valid_references = {
            "senorge",
            "era5",
        }

        if bias_correction_method not in valid_methods:
            raise ValueError(
                f"bias_correction_method must be one of "
                f"{sorted(valid_methods)}."
            )

        if bias_correction_reference not in valid_references:
            raise ValueError(
                f"bias_correction_reference must be one of "
                f"{sorted(valid_references)}."
            )

        if bias_corrected_input_x_days < 1:
            raise ValueError(
                "bias_corrected_input_x_days must be at least 1."
            )

        if bias_corrected_input_x_days != accumulation_days:
            raise ValueError(
                "For bias-corrected input, bias_corrected_input_x_days must "
                "equal accumulation_days because the script does not perform "
                "another accumulation."
            )

    if accumulation_days < 1:
        raise ValueError(
            "accumulation_days must be at least 1."
        )

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = (
        first_input_lead
        + accumulation_days
        - 1
    )

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "accumulation_days is too large for the input lead range."
        )

    number_of_usable_leads = (
        last_input_lead
        - first_usable_lead
        + 1
    )

    if not isinstance(
        number_of_lead_bins,
        int,
    ):
        raise TypeError(
            "number_of_lead_bins must be an integer."
        )

    if number_of_lead_bins < 1:
        raise ValueError(
            "number_of_lead_bins must be at least 1."
        )

    if number_of_lead_bins > number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins cannot exceed the number of usable leads."
        )

    filename = make_input_filename()

    if not filename.is_file():
        raise FileNotFoundError(
            f"Preprocessed input file not found: {filename}"
        )


def validate_input_dataset(
    ds,
):
    """Check that the preprocessed input has the required structure."""

    required_variables = {
        variable,
        "f_date",
        "model_type",
        "hdate",
    }

    missing_variables = (
        required_variables
        - set(
            ds.variables
        )
    )

    if missing_variables:
        raise ValueError(
            f"Input file is missing variables: "
            f"{sorted(missing_variables)}"
        )

    required_dimensions = {
        "lead_day",
        "number",
        "i_date",
    }

    missing_dimensions = (
        required_dimensions
        - set(
            ds.dims
        )
    )

    if missing_dimensions:
        raise ValueError(
            f"Input file is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    expected_leads = np.arange(
        first_input_lead,
        last_input_lead + 1,
        dtype="int64",
    )

    if not np.array_equal(
        ds[
            "lead_day"
        ].values,
        expected_leads,
    ):
        raise ValueError(
            "The input lead_day coordinate does not exactly match "
            f"{first_input_lead}-{last_input_lead}."
        )

    expected_tp24_dimensions = {
        "lead_day",
        "number",
        "i_date",
    }

    if set(
        ds[
            variable
        ].dims
    ) != expected_tp24_dimensions:
        raise ValueError(
            f"{variable} must contain dimensions "
            f"{sorted(expected_tp24_dimensions)}, "
            f"but has {ds[variable].dims}."
        )

    expected_f_date_dimensions = {
        "i_date",
        "lead_day",
    }

    if set(
        ds[
            "f_date"
        ].dims
    ) != expected_f_date_dimensions:
        raise ValueError(
            "f_date must contain dimensions "
            f"{sorted(expected_f_date_dimensions)}, "
            f"but has {ds['f_date'].dims}."
        )


# =============================================================================
# Lead-bin configuration
# =============================================================================

def split_usable_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """
    Split an inclusive lead interval into consecutive, near-equal bins.

    Extra leads are assigned to the later bins.

    Example:
        17-46 split into two bins -> 17-31 and 32-46
    """

    number_of_leads = (
        last_lead
        - first_lead
        + 1
    )

    base_size = (
        number_of_leads
        // number_of_bins
    )

    remainder = (
        number_of_leads
        % number_of_bins
    )

    bin_sizes = [
        base_size
        + int(
            bin_number
            >= number_of_bins - remainder
        )
        for bin_number in range(
            number_of_bins
        )
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:

        current_end = (
            current_start
            + bin_size
            - 1
        )

        bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + 1
        )

    return bins


def build_lead_bins():
    """Build lead bins after accounting for the accumulation length."""

    first_usable_lead = (
        first_input_lead
        + accumulation_days
        - 1
    )

    return split_usable_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )


def lead_bin_variable_name(
    lead_start,
    lead_end,
):
    """Return the output variable name for one lead-location bin."""

    return (
        f"tp24_max_lead"
        f"{lead_start}_{lead_end}"
    )


# =============================================================================
# N-day accumulation
# =============================================================================

def calculate_accumulation(
    ds,
):
    """
    Return precipitation over the usable accumulated ending leads.

    Raw input:
        Calculate a trailing accumulation_days sum.

    Bias-corrected input:
        Use tp24 directly because it is already accumulated. The first
        accumulation_days - 1 lead positions are excluded from the usable
        range.
    """

    tp24 = ds[
        variable
    ].transpose(
        "lead_day",
        "number",
        "i_date",
    )

    first_usable_lead = (
        first_input_lead
        + accumulation_days
        - 1
    )

    usable_leads = np.arange(
        first_usable_lead,
        last_input_lead + 1,
        dtype="int64",
    )

    if input_data_type == "raw":

        accumulated_tp24 = (
            tp24
            .rolling(
                lead_day=accumulation_days,
                min_periods=accumulation_days,
            )
            .sum()
            .sel(
                lead_day=usable_leads
            )
        )

    elif input_data_type == "bias_corrected":

        # Bias correction was calculated after the X-day accumulation.
        # Do not sum these values again.
        accumulated_tp24 = tp24.sel(
            lead_day=usable_leads
        )

    else:

        raise ValueError(
            "input_data_type must be 'raw' or 'bias_corrected'."
        )

    usable_f_dates = (
        ds[
            "f_date"
        ]
        .transpose(
            "i_date",
            "lead_day",
        )
        .sel(
            lead_day=usable_leads
        )
    )

    return (
        accumulated_tp24,
        usable_f_dates,
    )


# =============================================================================
# Calendar-month assignment
# =============================================================================

def month_with_most_daily_valid_dates(
    valid_dates,
):
    """
    Return the month containing the most daily valid dates.

    Month assignment always uses the complete N=1 valid-date window for input
    lead days 16-46. There are 31 dates, so one month must have a strict
    majority and a tie is impossible.
    """

    finite_dates = valid_dates[
        ~np.isnat(
            valid_dates
        )
    ]

    expected_number_of_dates = (
        last_input_lead
        - first_input_lead
        + 1
    )

    if finite_dates.size != expected_number_of_dates:
        raise ValueError(
            f"Expected {expected_number_of_dates} daily valid dates for "
            f"lead days {first_input_lead}-{last_input_lead}, "
            f"but found {finite_dates.size}."
        )

    month_numbers = (
        finite_dates
        .astype(
            "datetime64[M]"
        )
        .astype(
            "int64"
        )
        % 12
        + 1
    )

    months, counts = np.unique(
        month_numbers,
        return_counts=True,
    )

    largest_count = counts.max()

    if np.sum(
        counts == largest_count
    ) != 1:
        raise ValueError(
            "Calendar-month assignment produced a tie, even though the "
            "daily lead window contains 31 dates."
        )

    selected_position = np.argmax(
        counts
    )

    return np.int8(
        months[
            selected_position
        ]
    )


def calculate_month_coordinate(
    input_f_dates,
):
    """
    Assign one fixed calendar month to every i_date.

    The assignment uses all daily f_date values for lead days 16-46,
    regardless of accumulation_days.
    """

    daily_f_dates = (
        input_f_dates
        .transpose(
            "i_date",
            "lead_day",
        )
        .sel(
            lead_day=slice(
                first_input_lead,
                last_input_lead,
            )
        )
    )

    months = np.array(
        [
            month_with_most_daily_valid_dates(
                daily_f_dates.isel(
                    i_date=index
                ).values
            )
            for index in range(
                daily_f_dates.sizes[
                    "i_date"
                ]
            )
        ],
        dtype="int8",
    )

    return xr.DataArray(
        months,
        dims=(
            "i_date",
        ),
        coords={
            "i_date": daily_f_dates[
                "i_date"
            ],
        },
        name="month",
    )


# =============================================================================
# Maximum extraction
# =============================================================================

def extract_full_window_maximum(
    accumulated_tp24,
    usable_f_dates,
):
    """
    Extract the maximum, ending lead, and valid date for each sample.

    Equal maxima are resolved by taking the first occurrence in lead-day order,
    matching xarray/numpy argmax behavior.
    """

    finite = np.isfinite(
        accumulated_tp24
    )

    has_valid_value = finite.any(
        dim="lead_day"
    )

    values_for_argmax = accumulated_tp24.where(
        finite,
        other=-np.inf,
    )

    index_of_max = values_for_argmax.argmax(
        dim="lead_day"
    )

    tp24_max = accumulated_tp24.max(
        dim="lead_day",
        skipna=True,
    ).where(
        has_valid_value
    )

    lead_of_max = (
        accumulated_tp24[
            "lead_day"
        ]
        .isel(
            lead_day=index_of_max
        )
        .where(
            has_valid_value
        )
    )

    # f_date has no number dimension. Broadcast it to match
    # (lead_day, number, i_date), then select the date at index_of_max.
    f_date_broadcast = usable_f_dates.broadcast_like(
        accumulated_tp24
    )

    date_of_max = (
        f_date_broadcast
        .isel(
            lead_day=index_of_max
        )
        .where(
            has_valid_value
        )
    )

    tp24_max.name = "tp24_max"
    lead_of_max.name = "lead_of_max"
    date_of_max.name = "date_of_max"

    return (
        tp24_max,
        lead_of_max,
        date_of_max,
    )


# =============================================================================
# Lead-bin variables
# =============================================================================

def make_lead_bin_variables(
    tp24_max,
    lead_of_max,
    lead_bins,
):
    """
    Partition the same full-window maxima by lead of maximum.

    Each finite tp24_max appears in exactly one lead-bin variable at the same
    (number, i_date) position. It is not recalculated inside each bin.
    """

    output = {}

    assignment_count = xr.zeros_like(
        tp24_max,
        dtype="int8",
    )

    for lead_start, lead_end in lead_bins:

        in_bin = (
            (lead_of_max >= lead_start)
            & (lead_of_max <= lead_end)
        )

        variable_name = lead_bin_variable_name(
            lead_start,
            lead_end,
        )

        output[
            variable_name
        ] = tp24_max.where(
            in_bin
        )

        assignment_count = (
            assignment_count
            + xr.where(
                in_bin,
                1,
                0,
            )
        )

    valid_full = np.isfinite(
        tp24_max
    )

    bad_valid = (
        valid_full
        & (
            assignment_count
            != 1
        )
    )

    bad_missing = (
        (~valid_full)
        & (
            assignment_count
            != 0
        )
    )

    if bool(
        bad_valid.any().values
        or bad_missing.any().values
    ):
        raise ValueError(
            "Lead-bin variables do not form an exact partition of tp24_max."
        )

    return output


# =============================================================================
# Output dataset
# =============================================================================

def build_output_dataset(
    input_ds,
    tp24_max,
    lead_of_max,
    date_of_max,
    month,
    lead_bin_variables,
    lead_bins,
):
    """Build the compact monthly-maximum sample dataset."""

    output = xr.Dataset(
        data_vars={
            "tp24_max": tp24_max.astype(
                "float32"
            ),
            "date_of_max": date_of_max,
            "lead_of_max": lead_of_max.astype(
                "float32"
            ),
            "month": month,
            "model_type": input_ds[
                "model_type"
            ],
            "hdate": input_ds[
                "hdate"
            ],
        },
        coords={
            "number": input_ds[
                "number"
            ],
            "i_date": input_ds[
                "i_date"
            ],
        },
    )

    for variable_name, values in lead_bin_variables.items():
        output[
            variable_name
        ] = values.astype(
            "float32"
        )

    first_usable_lead = (
        first_input_lead
        + accumulation_days
        - 1
    )

    output[
        "tp24_max"
    ].attrs.update(
        {
            "units": "mm",
            "description": (
                f"Maximum {accumulation_days}-day accumulated precipitation "
                f"over ending lead days "
                f"{first_usable_lead}-{last_input_lead}"
            ),
            "lead_start": first_usable_lead,
            "lead_end": last_input_lead,
        }
    )

    output[
        "month"
    ].attrs.update(
        {
            "description": (
                "Calendar month containing the largest number of daily valid "
                "dates across input lead days 16-46"
            ),
            "assignment_window": (
                "Fixed N=1 daily valid-date window for lead days 16-46"
            ),
            "tie_handling": (
                "No tie is possible because the assignment window contains "
                "31 daily valid dates"
            ),
        }
    )

    output[
        "date_of_max"
    ].attrs.update(
        {
            "description": (
                "Calendar valid date on which tp24_max occurs"
            ),
        }
    )

    output[
        "lead_of_max"
    ].attrs.update(
        {
            "description": (
                "Accumulated ending lead day on which tp24_max occurs"
            ),
            "units": "days",
        }
    )

    output[
        "model_type"
    ].attrs.update(
        {
            "description": (
                "Source type for each initialization: forecast or hindcast"
            ),
        }
    )

    output[
        "hdate"
    ].attrs.update(
        {
            "description": (
                "Original hindcast initialization date; NaT for forecast rows"
            ),
        }
    )

    for bin_number, (
        lead_start,
        lead_end,
    ) in enumerate(
        lead_bins,
        start=1,
    ):

        variable_name = lead_bin_variable_name(
            lead_start,
            lead_end,
        )

        output[
            variable_name
        ].attrs.update(
            {
                "units": "mm",
                "description": (
                    "Subset of the complete-window tp24_max values whose "
                    f"maximum occurs at ending leads {lead_start}-{lead_end}; "
                    f"lead bin {bin_number} of {number_of_lead_bins}"
                ),
                "lead_start": lead_start,
                "lead_end": lead_end,
                "range_type": (
                    f"lead-location bin {bin_number} "
                    f"of {number_of_lead_bins}"
                ),
            }
        )

    output.attrs.update(
        {
            "description": (
                "Monthly S2S precipitation-maximum samples created from a "
                "preprocessed combined forecast/hindcast file"
            ),
            "source_file": str(
                make_input_filename()
            ),
            "input_data_type": input_data_type,
            "bias_correction_method": (
                bias_correction_method
                if input_data_type == "bias_corrected"
                else "none"
            ),
            "bias_correction_reference": (
                bias_correction_reference
                if input_data_type == "bias_corrected"
                else "none"
            ),
            "input_accumulation_days": (
                bias_corrected_input_x_days
                if input_data_type == "bias_corrected"
                else 1
            ),
            "accumulation_performed_in_this_script": (
                "false"
                if input_data_type == "bias_corrected"
                else "true"
            ),
            "variable": variable,
            "catchment": catchment,
            "forecast_initialization_start": forecast_date_range[0],
            "forecast_initialization_end": forecast_date_range[1],
            "accumulation_days": accumulation_days,
            "first_input_lead": first_input_lead,
            "last_input_lead": last_input_lead,
            "first_usable_accumulated_lead": first_usable_lead,
            "last_usable_accumulated_lead": last_input_lead,
            "number_of_lead_bins": number_of_lead_bins,
            "calendar_month_assignment": (
                "Month containing the largest number of N=1 daily valid dates "
                "across input lead days 16-46; independent of "
                "accumulation_days"
            ),
            "calendar_month_assignment_number_of_dates": 31,
            "calendar_month_tie_handling": (
                "No tie possible because 31 daily valid dates are used"
            ),
            "lead_bin_sampling": (
                "Complete-window maxima partitioned by ending lead day of "
                "maximum; maxima are not recalculated within lead bins"
            ),
        }
    )

    return output


# =============================================================================
# Reporting and NetCDF writing
# =============================================================================

def print_summary(
    output,
    lead_bins,
):
    """Print a readable summary of the completed sample file."""

    print()
    print(
        "Output summary"
    )
    print(
        "--------------"
    )
    print(
        output
    )

    print()
    print(
        "Usable accumulated lead range:",
        (
            f"{first_input_lead + accumulation_days - 1}-"
            f"{last_input_lead}"
        ),
    )

    print(
        "Lead bins:",
        lead_bins,
    )

    print()
    print(
        "Samples assigned to each month"
    )
    print(
        "------------------------------"
    )

    for month_number in range(
        1,
        13,
    ):

        initialization_mask = (
            output[
                "month"
            ]
            == month_number
        )

        sample_count = np.isfinite(
            output[
                "tp24_max"
            ].where(
                initialization_mask
            )
        ).sum().item()

        print(
            f"Month {month_number:>2}: "
            f"{sample_count:>8} finite samples"
        )


def write_output(
    output,
    filename,
    lead_bin_variables,
):
    """Write the completed sample dataset to NetCDF."""

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    precipitation_variables = [
        "tp24_max",
        *lead_bin_variables,
    ]

    encoding = {
        variable_name: {
            "dtype": "float32",
            "_FillValue": np.float32(
                np.nan
            ),
            "zlib": True,
            "complevel": 4,
        }
        for variable_name in precipitation_variables
    }

    encoding[
        "lead_of_max"
    ] = {
        "dtype": "float32",
        "_FillValue": np.float32(
            np.nan
        ),
    }

    encoding[
        "hdate"
    ] = {
        "dtype": "int64",
    }

    output.to_netcdf(
        filename,
        encoding=encoding,
    )

    print()
    print(
        "Wrote:",
        filename,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    lead_bins = build_lead_bins()

    filename_input = make_input_filename()

    filename_output = make_output_filename(
        lead_bins
    )

    print(
        "Reading:",
        filename_input,
    )

    print(
        "Writing:",
        filename_output,
    )

    print()
    print(
        "Calculation settings"
    )
    print(
        "--------------------"
    )
    print(
        "Input data type:",
        input_data_type,
    )

    if input_data_type == "bias_corrected":

        print(
            "Bias-correction method:",
            bias_correction_method,
        )

        print(
            "Bias-correction reference:",
            bias_correction_reference,
        )

        print(
            "Input accumulation days:",
            bias_corrected_input_x_days,
        )

        print(
            "Additional accumulation:",
            "none",
        )

    else:

        print(
            "Input accumulation days:",
            1,
        )

        print(
            "Additional accumulation:",
            f"{accumulation_days}-day trailing sum",
        )

    print(
        "Accumulation days:",
        accumulation_days,
    )
    print(
        "Month-assignment leads:",
        (
            f"{first_input_lead}-"
            f"{last_input_lead} "
            f"(N=1 daily dates)"
        ),
    )
    print(
        "Input leads:",
        (
            f"{first_input_lead}-"
            f"{last_input_lead}"
        ),
    )
    print(
        "Usable accumulated leads:",
        (
            f"{first_input_lead + accumulation_days - 1}-"
            f"{last_input_lead}"
        ),
    )
    print(
        "Number of lead bins:",
        number_of_lead_bins,
    )
    print(
        "Lead bins:",
        lead_bins,
    )

    with xr.open_dataset(
        filename_input,
        decode_timedelta=False,
    ) as opened:

        input_ds = opened.load()

    validate_input_dataset(
        input_ds
    )

    (
        accumulated_tp24,
        usable_f_dates,
    ) = calculate_accumulation(
        input_ds
    )

    # Assign the calendar month from the full N=1 daily valid-date window
    # (lead days 16-46), independently of accumulation_days.
    month = calculate_month_coordinate(
        input_ds[
            "f_date"
        ]
    )

    (
        tp24_max,
        lead_of_max,
        date_of_max,
    ) = extract_full_window_maximum(
        accumulated_tp24=accumulated_tp24,
        usable_f_dates=usable_f_dates,
    )

    lead_bin_variables = make_lead_bin_variables(
        tp24_max=tp24_max,
        lead_of_max=lead_of_max,
        lead_bins=lead_bins,
    )

    output = build_output_dataset(
        input_ds=input_ds,
        tp24_max=tp24_max,
        lead_of_max=lead_of_max,
        date_of_max=date_of_max,
        month=month,
        lead_bin_variables=lead_bin_variables,
        lead_bins=lead_bins,
    )

    print_summary(
        output=output,
        lead_bins=lead_bins,
    )

    if write2file:

        write_output(
            output=output,
            filename=filename_output,
            lead_bin_variables=list(
                lead_bin_variables
            ),
        )
