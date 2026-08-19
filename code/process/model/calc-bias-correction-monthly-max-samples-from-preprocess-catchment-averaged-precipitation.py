#!/usr/bin/env python3
"""
Apply a two-stage multiplicative bias correction to the NEW compact S2S
monthly-maximum sample dataset.

Expected model input structure
------------------------------
The model sample is produced by the new sample-building script and has the
compact structure:

    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)
    month(i_date)
    date_of_max(number, i_date)
    lead_of_max(number, i_date)
    model_type(i_date)
    hdate(i_date)
    number(number)
    i_date(i_date)

The sample month is stored as sample_month(i_date) in YYYYMM format. Monthly
statistics are therefore calculated by:

1. deriving calendar month as sample_month % 100 and selecting all i_date values
   for the requested calendar month;
2. pooling all finite ensemble-member values across number and i_date;
3. calculating the mean of that pooled monthly sample.

Bias-correction method
----------------------
The correction follows the same two-stage procedure as the old script.

Stage 1: lead-time correction
~~~~~~~~~~~~~~~~~~~~~~~~~~
For every lead-location split and calendar month:

    lead_time_ratio
        = original_full_sample_monthly_mean
          / original_split_sample_monthly_mean

Each split sample is multiplied by its monthly ratio. Its original NaN mask
and its exact (number, i_date) positions are preserved.

The corrected splits are then recombined position-by-position to rebuild the
full sample. Every finite full-sample value must correspond to exactly one
finite split value at the same (number, i_date).

Stage 2: reference correction
~~~~~~~~~~~~~~~~~~~~~~~~~~
For every calendar month:

    reference_ratio
        = reference_monthly_mean
          / rebuilt_full_sample_monthly_mean

The same reference ratio is applied to the rebuilt full sample and all
lead-time-corrected split samples.

Output structure
----------------
The output keeps the same compact organization and provenance variables as the
input. The original uncorrected tp24_max variables are replaced in place by
their bias-corrected versions, using the same variable names:

    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)

The output also stores:

    bias_correction_ratio(month_of_year)
    lead_time_bias_correction_ratio_lead<start>_<end>(month_of_year)

The input variable sample_month(i_date), stored as YYYYMM, is retained unchanged.
Calendar month 1-12 is derived internally as sample_month % 100. The separate
month_of_year coordinate is used only for the 12 monthly correction ratios.

where <reference> is either "era5" or "senorge".
"""

from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"

accumulation_days = 2

forecast_date_range = [
    "2020-01-02",
    "2023-12-28",
]

observation_years = [
    "1957",
    "2025",
]

era5_grid = "0.5x0.5"

# These settings must match the new monthly-sample file.
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Choose one reference dataset:
#
#     "era5"
#     "senorge"
REFERENCE_DATASET = "senorge"

# Optional explicit paths.
#
# Leave as None to build filenames automatically.
model_filename_override = None
output_filename_override = None

write2file = True


# =============================================================================
# Dataset-specific settings
# =============================================================================

MODEL_VARIABLE = "tp24"

ERA5_VARIABLE = "tp24"

SENORGE_VARIABLE = "rr"


# =============================================================================
# General helpers
# =============================================================================

MONTHS = np.arange(
    1,
    13,
    dtype="int8",
)


def get_file_id(
    catchment_name,
):
    """Return the short catchment name used in the new sample filename."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


# =============================================================================
# Lead-time configuration
# =============================================================================

def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """
    Split usable accumulated ending leads into near-equal consecutive bins.

    Extra lead times are assigned to the later bins.
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
            bin_index
            >= number_of_bins - remainder
        )
        for bin_index in range(
            number_of_bins
        )
    ]

    lead_bins = []
    current_start = first_lead

    for bin_size in bin_sizes:

        current_end = (
            current_start
            + bin_size
            - 1
        )

        lead_bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + 1
        )

    return lead_bins


def get_full_lead_range():
    """Return the usable accumulated ending-lead range."""

    return (
        first_input_lead
        + accumulation_days
        - 1,
        last_input_lead,
    )


def build_lead_bins():
    """Return the lead-location bins used in the model sample file."""

    full_start, full_end = (
        get_full_lead_range()
    )

    return split_usable_accumulated_leads(
        first_lead=full_start,
        last_lead=full_end,
        number_of_bins=number_of_lead_bins,
    )


def lead_split_filename_label():
    """Return the lead-range text used in the new sample filename."""

    full_start, full_end = (
        get_full_lead_range()
    )

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in build_lead_bins()
    )

    return (
        f"lead{full_start}-{full_end}_"
        f"split{number_of_lead_bins}_"
        f"{split_text}"
    )


def full_model_variable_name():
    """Return the complete-window variable in the new sample file."""

    return "tp24_max"


def split_model_variable_name(
    lead_start,
    lead_end,
):
    """Return one split variable in the new sample file."""

    return (
        f"tp24_max_lead"
        f"{lead_start}_{lead_end}"
    )


# =============================================================================
# Filenames
# =============================================================================

def make_model_filename():
    """Create the filename of the new compact monthly sample dataset."""

    if model_filename_override is not None:
        return Path(
            model_filename_override
        )

    return (
        Path(
            config.dirs[
                "s2s_processed"
            ]
        )
        / (
            f"test-monthly_max_samples_"
            f"{MODEL_VARIABLE}_"
            f"{accumulation_days}dayacc_"
            f"{get_file_id(catchment)}_"
            f"{lead_split_filename_label()}_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        )
    )


def make_era5_filename():
    """Create the ERA5 reference filename."""

    return Path(
        (
            f"{config.dirs['era5_processed']}"
            f"monthly_max_samples_"
            f"{ERA5_VARIABLE}_{accumulation_days}dayacc_"
            f"{catchment}_"
            f"{observation_years[0]}-"
            f"{observation_years[1]}.nc"
        )
    )


def make_senorge_filename():
    """Create the SeNorge reference filename."""

    return Path(
        (
            f"{config.dirs['senorge_processed']}"
            f"monthly_max_samples_"
            f"{SENORGE_VARIABLE}_{accumulation_days}dayacc_"
            f"{catchment}_"
            f"{observation_years[0]}-"
            f"{observation_years[1]}.nc"
        )
    )


def make_output_filename(
    model_filename,
):
    """Append the bias-correction reference to the model filename."""

    if output_filename_override is not None:
        return Path(
            output_filename_override
        )

    return model_filename.with_name(
        (
            f"{model_filename.stem}_"
            f"bc_mm_{REFERENCE_DATASET}"
            f"{model_filename.suffix}"
        )
    )


# =============================================================================
# Validation
# =============================================================================

def validate_user_settings():
    """Validate user settings and required files."""

    valid_references = {
        "era5",
        "senorge",
    }

    if REFERENCE_DATASET not in valid_references:
        raise ValueError(
            f"REFERENCE_DATASET must be one of "
            f"{sorted(valid_references)}. "
            f"Got '{REFERENCE_DATASET}'."
        )

    if accumulation_days < 1:
        raise ValueError(
            "accumulation_days must be at least 1."
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

    if not (
        1
        <= number_of_lead_bins
        <= number_of_usable_leads
    ):
        raise ValueError(
            "number_of_lead_bins must be between 1 and the number "
            "of usable accumulated lead times."
        )

    model_filename = (
        make_model_filename()
    )

    if not model_filename.is_file():
        raise FileNotFoundError(
            f"Model sample file not found: {model_filename}"
        )


def validate_model_dataset(
    model_ds,
):
    """Check the structure of the new compact model sample dataset."""

    required_dimensions = {
        "number",
        "i_date",
    }

    missing_dimensions = (
        required_dimensions
        - set(
            model_ds.dims
        )
    )

    if missing_dimensions:
        raise ValueError(
            f"Model dataset is missing dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    required_variables = {
        "sample_month",
        "model_type",
        "hdate",
        "date_of_max",
        full_model_variable_name(),
    }

    required_variables.update(
        {
            split_model_variable_name(
                lead_start,
                lead_end,
            )
            for lead_start, lead_end in build_lead_bins()
        }
    )

    missing_variables = (
        required_variables
        - set(
            model_ds.variables
        )
    )

    if missing_variables:
        raise ValueError(
            f"Model dataset is missing variables: "
            f"{sorted(missing_variables)}"
        )

    sample_month_values = np.asarray(model_ds["sample_month"].values)

    if model_ds["sample_month"].dims != ("i_date",):
        raise ValueError(
            "sample_month must have dimension ('i_date',), "
            f"but has {model_ds['sample_month'].dims}."
        )

    finite_sample_months = sample_month_values[np.isfinite(sample_month_values)]
    calendar_months = finite_sample_months.astype("int64") % 100

    if not np.all(np.isin(calendar_months, MONTHS)):
        raise ValueError("sample_month(i_date) contains invalid YYYYMM values.")

    expected_sample_dims = {
        "number",
        "i_date",
    }

    sample_variables = [
        full_model_variable_name(),
        *[
            split_model_variable_name(
                lead_start,
                lead_end,
            )
            for lead_start, lead_end in build_lead_bins()
        ],
    ]

    for variable_name in sample_variables:

        if set(
            model_ds[
                variable_name
            ].dims
        ) != expected_sample_dims:
            raise ValueError(
                f"{variable_name} must contain dimensions "
                f"{sorted(expected_sample_dims)}, but has "
                f"{model_ds[variable_name].dims}."
            )

    expected_metadata = {
        "first_usable_accumulated_lead": (
            first_input_lead
            + accumulation_days
            - 1
        ),
        "last_usable_accumulated_lead": (
            last_input_lead
        ),
        "number_of_lead_bins": (
            number_of_lead_bins
        ),
    }

    for attribute, expected_value in expected_metadata.items():

        if attribute not in model_ds.attrs:
            raise KeyError(
                f"Model dataset is missing global attribute "
                f"'{attribute}'."
            )

        actual_value = int(
            model_ds.attrs[
                attribute
            ]
        )

        if actual_value != expected_value:
            raise ValueError(
                f"Model attribute '{attribute}' is {actual_value}, "
                f"but the script expects {expected_value}."
            )


def get_calendar_month(model_ds):
    """Return calendar month 1-12 derived from sample_month(i_date) YYYYMM."""
    calendar_month = (model_ds["sample_month"].astype("int64") % 100).rename("calendar_month")
    return calendar_month


# =============================================================================
# Reference data
# =============================================================================

def load_reference_dataset():
    """Open the selected reference dataset."""

    if REFERENCE_DATASET == "era5":

        filename = (
            make_era5_filename()
        )

        variable_name = (
            ERA5_VARIABLE
        )

    else:

        filename = (
            make_senorge_filename()
        )

        variable_name = (
            SENORGE_VARIABLE
        )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Reference file not found: {filename}"
        )

    reference_ds = xr.open_dataset(
        filename
    )

    if variable_name not in reference_ds:
        reference_ds.close()

        raise KeyError(
            f"Variable '{variable_name}' was not found in "
            f"{filename}."
        )

    return (
        reference_ds,
        variable_name,
        filename,
    )


def get_reference_monthly_mean(
    reference_ds,
    reference_variable,
):
    """
    Calculate one reference mean for every calendar month.

    Expected reference structure:

        reference_variable(year, month)
    """

    values = reference_ds[
        reference_variable
    ]

    required_dimensions = {
        "year",
        "month",
    }

    if not required_dimensions.issubset(
        set(
            values.dims
        )
    ):
        raise ValueError(
            f"Reference variable '{reference_variable}' must contain "
            "dimensions 'year' and 'month'."
        )

    monthly_mean = values.mean(
        dim="year",
        skipna=True,
    )

    monthly_mean = monthly_mean.sel(
        month=MONTHS
    )

    monthly_mean.name = (
        "reference_monthly_mean"
    )

    monthly_mean.attrs[
        "reference_year_start"
    ] = int(
        values[
            "year"
        ].min().values
    )

    monthly_mean.attrs[
        "reference_year_end"
    ] = int(
        values[
            "year"
        ].max().values
    )

    return monthly_mean


# =============================================================================
# Monthly model statistics
# =============================================================================

def calculate_model_monthly_mean(
    values,
    month_coordinate,
):
    """
    Calculate a pooled model mean for each calendar month.

    For each month, all finite values are pooled across both number and i_date.
    """

    monthly_means = []

    for month_number in MONTHS:

        selected = values.where(
            month_coordinate
            == month_number
        )

        mean_value = selected.mean(
            dim=(
                "number",
                "i_date",
            ),
            skipna=True,
        )

        monthly_means.append(
            mean_value
        )

    result = xr.concat(
        monthly_means,
        dim=xr.DataArray(
            MONTHS,
            dims=(
                "month",
            ),
            name="month",
        ),
    )

    return result


def check_monthly_means(
    monthly_mean,
    label,
):
    """Require finite, positive means in every calendar month."""

    if np.any(
        ~np.isfinite(
            monthly_mean.values
        )
    ):
        raise ValueError(
            f"At least one monthly mean is non-finite for {label}."
        )

    if np.any(
        monthly_mean.values
        <= 0
    ):
        raise ValueError(
            f"At least one monthly mean is zero or negative for {label}."
        )


def expand_monthly_ratio_to_i_date(
    ratio,
    month_coordinate,
):
    """
    Expand ratio(month) to ratio(i_date) using month(i_date).

    The result broadcasts automatically across the number dimension.
    """

    ratio_by_i_date = ratio.sel(
        month=month_coordinate
    )

    # The selection adds month as an auxiliary coordinate on i_date.
    # It is not needed in the corrected precipitation variables.
    if "month" in ratio_by_i_date.coords:
        ratio_by_i_date = ratio_by_i_date.drop_vars(
            "month"
        )

    return ratio_by_i_date


# =============================================================================
# Variable discovery
# =============================================================================

def get_full_model_variable(
    model_ds,
):
    """Return the complete-window maximum variable."""

    variable_name = (
        full_model_variable_name()
    )

    if variable_name not in model_ds:
        raise KeyError(
            f"Expected complete-window variable "
            f"'{variable_name}' was not found."
        )

    return variable_name


def get_split_model_variables(
    model_ds,
):
    """Return the split lead-location variables in lead-bin order."""

    split_variables = [
        split_model_variable_name(
            lead_start,
            lead_end,
        )
        for lead_start, lead_end in build_lead_bins()
    ]

    missing = [
        variable_name
        for variable_name in split_variables
        if variable_name not in model_ds
    ]

    if missing:
        raise KeyError(
            f"Missing split variables: {missing}"
        )

    return split_variables


# =============================================================================
# Stage 1: lead-time correction
# =============================================================================

def calculate_lead_time_correction_ratios(
    model_ds,
    full_model_variable,
    split_variables,
):
    """
    Calculate one lead-time correction ratio for each split and month.

        ratio(month)
            = original full-sample mean(month)
              / original split-sample mean(month)
    """

    month_coordinate = get_calendar_month(model_ds)

    full_monthly_mean = (
        calculate_model_monthly_mean(
            values=model_ds[
                full_model_variable
            ],
            month_coordinate=month_coordinate,
        )
    )

    check_monthly_means(
        full_monthly_mean,
        full_model_variable,
    )

    ratios = {}

    for variable_name in split_variables:

        split_monthly_mean = (
            calculate_model_monthly_mean(
                values=model_ds[
                    variable_name
                ],
                month_coordinate=month_coordinate,
            )
        )

        check_monthly_means(
            split_monthly_mean,
            variable_name,
        )

        ratio = (
            full_monthly_mean
            / split_monthly_mean
        )

        lead_label = variable_name.replace(
            "tp24_max_",
            "",
            1,
        )

        ratio.name = (
            f"lead_time_bias_correction_ratio_"
            f"{lead_label}"
        )

        ratio.attrs = {
            "description": (
                "Monthly multiplicative lead-time correction ratio "
                "calculated as the original complete-window monthly "
                "mean divided by the original split monthly mean"
            ),
            "formula": (
                "original_full_monthly_mean / "
                "original_split_monthly_mean"
            ),
            "target_variable": full_model_variable,
            "source_variable": variable_name,
            "units": "1",
        }

        ratios[
            variable_name
        ] = ratio

    return ratios


def apply_lead_time_correction(
    model_ds,
    split_variables,
    lead_time_ratios,
):
    """
    Apply monthly lead-time ratios without changing split membership masks.
    """

    month_coordinate = get_calendar_month(model_ds)

    corrected_splits = {}

    for variable_name in split_variables:

        ratio_by_i_date = (
            expand_monthly_ratio_to_i_date(
                ratio=lead_time_ratios[
                    variable_name
                ],
                month_coordinate=month_coordinate,
            )
        )

        corrected = (
            model_ds[
                variable_name
            ]
            * ratio_by_i_date
        )

        corrected = corrected.transpose(
            *model_ds[
                variable_name
            ].dims
        )

        corrected.attrs = (
            model_ds[
                variable_name
            ]
            .attrs
            .copy()
        )

        corrected.attrs[
            "lead_time_bias_correction_method"
        ] = (
            "monthly multiplicative mean scaling"
        )

        corrected.attrs[
            "lead_time_bias_correction_formula"
        ] = (
            "corrected_split = original_split * "
            "original_full_monthly_mean / original_split_monthly_mean"
        )

        corrected_splits[
            variable_name
        ] = corrected

    return corrected_splits


def rebuild_full_sample(
    model_ds,
    full_model_variable,
    split_variables,
    corrected_splits,
):
    """
    Rebuild the full sample at the same (number, i_date) positions.

    Every finite original full-sample value must correspond to exactly one
    finite original split value.
    """

    original_full = model_ds[
        full_model_variable
    ]

    finite_split_count = xr.zeros_like(
        original_full,
        dtype="int16",
    )

    for variable_name in split_variables:

        finite_split_count = (
            finite_split_count
            + xr.where(
                np.isfinite(
                    model_ds[
                        variable_name
                    ]
                ),
                1,
                0,
            )
        )

    original_full_finite = np.isfinite(
        original_full
    )

    invalid_finite = (
        original_full_finite
        & (
            finite_split_count
            != 1
        )
    )

    invalid_missing = (
        (~original_full_finite)
        & (
            finite_split_count
            != 0
        )
    )

    if bool(
        invalid_finite.any().values
        or invalid_missing.any().values
    ):
        raise ValueError(
            "The split variables do not form an exact position-aligned "
            "partition of tp24_max. Every finite tp24_max must have exactly "
            "one finite split value at the same (number, i_date), and missing "
            "tp24_max positions must have no finite split value."
        )

    rebuilt = xr.full_like(
        original_full,
        np.nan,
        dtype="float64",
    )

    for variable_name in split_variables:

        corrected_split = corrected_splits[
            variable_name
        ]

        rebuilt = xr.where(
            np.isfinite(
                corrected_split
            ),
            corrected_split,
            rebuilt,
        )

    rebuilt = rebuilt.transpose(
        *original_full.dims
    )

    rebuilt.name = (
        full_model_variable
    )

    rebuilt.attrs = (
        original_full
        .attrs
        .copy()
    )

    rebuilt.attrs[
        "lead_time_rebuild_method"
    ] = (
        "Rebuilt position-by-position from the finite lead-time-corrected "
        "split value at each original (number, i_date)."
    )

    return rebuilt


# =============================================================================
# Stage 2: reference correction
# =============================================================================

def calculate_reference_bias_correction_ratio(
    model_ds,
    rebuilt_full_sample,
    reference_monthly_mean,
):
    """
    Calculate the monthly reference correction ratio.

        ratio(month)
            = reference mean(month)
              / rebuilt full-sample mean(month)
    """

    rebuilt_monthly_mean = (
        calculate_model_monthly_mean(
            values=rebuilt_full_sample,
            month_coordinate=get_calendar_month(model_ds),
        )
    )

    check_monthly_means(
        rebuilt_monthly_mean,
        "rebuilt full sample",
    )

    reference_mean_aligned = (
        reference_monthly_mean
        .sel(
            month=MONTHS
        )
    )

    if np.any(
        ~np.isfinite(
            reference_mean_aligned.values
        )
    ):
        raise ValueError(
            "At least one reference monthly mean is non-finite."
        )

    if np.any(
        reference_mean_aligned.values
        <= 0
    ):
        raise ValueError(
            "At least one reference monthly mean is zero or negative."
        )

    ratio = (
        reference_mean_aligned
        / rebuilt_monthly_mean
    )

    ratio.name = (
        "bias_correction_ratio"
    )

    ratio.attrs = {
        "description": (
            "Monthly multiplicative reference correction ratio calculated "
            "as the reference mean divided by the rebuilt complete-window "
            "model mean"
        ),
        "formula": (
            "reference_monthly_mean / "
            "rebuilt_full_model_monthly_mean"
        ),
        "reference_dataset": REFERENCE_DATASET,
        "reference_year_start": (
            reference_monthly_mean.attrs[
                "reference_year_start"
            ]
        ),
        "reference_year_end": (
            reference_monthly_mean.attrs[
                "reference_year_end"
            ]
        ),
        "units": "1",
    }

    return ratio


# =============================================================================
# Output construction
# =============================================================================

def get_original_maximum_variables(
    model_ds,
):
    """Return uncorrected compact maximum variables."""

    variables = [
        variable_name
        for variable_name in model_ds.data_vars
        if (
            variable_name
            == "tp24_max"
            or variable_name.startswith(
                "tp24_max_lead"
            )
        )
        and "_bc_" not in variable_name
    ]

    return variables


def build_final_bias_corrected_dataset(
    model_ds,
    full_model_variable,
    split_variables,
    corrected_splits,
    rebuilt_full_sample,
    reference_ratio,
    lead_time_ratios,
):
    """
    Build the compact bias-corrected output dataset.

    Provenance variables and coordinates remain unchanged.
    """

    output_ds = (
        model_ds
        .drop_vars(
            get_original_maximum_variables(
                model_ds
            )
        )
        .copy(
            deep=True
        )
    )

    reference_ratio_by_i_date = (
        expand_monthly_ratio_to_i_date(
            ratio=reference_ratio,
            month_coordinate=get_calendar_month(model_ds),
        )
    )

    final_full = (
        rebuilt_full_sample
        * reference_ratio_by_i_date
    )

    final_full = final_full.transpose(
        *rebuilt_full_sample.dims
    )

    final_full.attrs = (
        rebuilt_full_sample
        .attrs
        .copy()
    )

    final_full.attrs.update(
        {
            "bias_correction_method": (
                "two-stage monthly multiplicative correction: "
                "lead-time correction followed by reference correction"
            ),
            "bias_correction_formula": (
                "rebuilt_full_sample * reference_ratio"
            ),
            "bias_correction_reference": (
                REFERENCE_DATASET
            ),
            "original_variable": (
                full_model_variable
            ),
        }
    )

    output_ds[
        full_model_variable
    ] = final_full.astype(
        "float32"
    )

    for variable_name in split_variables:

        final_split = (
            corrected_splits[
                variable_name
            ]
            * reference_ratio_by_i_date
        )

        final_split = final_split.transpose(
            *model_ds[
                variable_name
            ].dims
        )

        final_split.attrs = (
            corrected_splits[
                variable_name
            ]
            .attrs
            .copy()
        )

        final_split.attrs.update(
            {
                "bias_correction_method": (
                    "two-stage monthly multiplicative correction: "
                    "lead-time correction followed by common reference "
                    "correction"
                ),
                "bias_correction_formula": (
                    "original_split * lead_time_ratio * reference_ratio"
                ),
                "bias_correction_reference": (
                    REFERENCE_DATASET
                ),
                "original_variable": (
                    variable_name
                ),
            }
        )

        output_ds[
            variable_name
        ] = final_split.astype(
            "float32"
        )

    # The compact input contains sample_month(i_date) as YYYYMM. Correction
    # ratios use a separate 12-value calendar-month axis in the output dataset.
    reference_ratio_for_output = (
        reference_ratio
        .rename(
            {
                "month": "month_of_year",
            }
        )
        .astype(
            "float32"
        )
    )

    output_ds[
        reference_ratio.name
    ] = reference_ratio_for_output

    for ratio in lead_time_ratios.values():

        ratio_for_output = (
            ratio
            .rename(
                {
                    "month": "month_of_year",
                }
            )
            .astype(
                "float32"
            )
        )

        output_ds[
            ratio.name
        ] = ratio_for_output

    output_ds.attrs = (
        model_ds
        .attrs
        .copy()
    )

    output_ds.attrs.update(
        {
            "bias_correction": (
                "Two-stage monthly multiplicative correction applied to "
                "the compact (number, i_date) sample. First, each split "
                "lead-time sample is scaled to the original full-sample "
                "monthly mean. Second, the full sample is rebuilt at the "
                "same positions. Third, one reference correction is "
                "calculated from the rebuilt full sample and applied to "
                "the rebuilt full and split samples."
            ),
            "bias_correction_reference_dataset": (
                REFERENCE_DATASET
            ),
            "bias_correction_scope": (
                "monthly-mean correction"
            ),
            "bias_corrected_variable_naming": (
                "Original tp24_max variable names retained"
            ),
            "bias_correction_reference_year_start": (
                reference_ratio.attrs[
                    "reference_year_start"
                ]
            ),
            "bias_correction_reference_year_end": (
                reference_ratio.attrs[
                    "reference_year_end"
                ]
            ),
            "sample_alignment": (
                "Preserved exactly in (number, i_date). The original sample_month, "
                "date_of_max, lead_of_max, model_type, hdate, number, and i_date "
                "information is retained unchanged."
            ),
        }
    )

    return output_ds


# =============================================================================
# Reporting
# =============================================================================

def print_lead_time_correction_table(
    model_ds,
    full_model_variable,
    split_variables,
    lead_time_ratios,
):
    """Print monthly model means and lead-time correction ratios."""

    full_mean = calculate_model_monthly_mean(
        values=model_ds[
            full_model_variable
        ],
        month_coordinate=get_calendar_month(model_ds),
    )

    print()
    print(
        "Lead-time correction"
    )
    print(
        "--------------------"
    )

    for variable_name in split_variables:

        split_mean = calculate_model_monthly_mean(
            values=model_ds[
                variable_name
            ],
            month_coordinate=get_calendar_month(model_ds),
        )

        ratio = lead_time_ratios[
            variable_name
        ]

        print()
        print(
            variable_name
        )

        print(
            f"{'Month':>5}"
            f"{'Full mean':>14}"
            f"{'Split mean':>14}"
            f"{'Lead ratio':>14}"
        )

        print(
            "-" * 47
        )

        for month_number in MONTHS:

            print(
                f"{int(month_number):>5d}"
                f"{float(full_mean.sel(month=month_number).values):>14.3f}"
                f"{float(split_mean.sel(month=month_number).values):>14.3f}"
                f"{float(ratio.sel(month=month_number).values):>14.4f}"
            )


def print_reference_correction_table(
    model_ds,
    rebuilt_full_sample,
    reference_monthly_mean,
    reference_ratio,
):
    """Print rebuilt model means and reference correction ratios."""

    rebuilt_mean = calculate_model_monthly_mean(
        values=rebuilt_full_sample,
        month_coordinate=get_calendar_month(model_ds),
    )

    print()
    print(
        "Reference period:",
        (
            f"{reference_monthly_mean.attrs['reference_year_start']}-"
            f"{reference_monthly_mean.attrs['reference_year_end']}"
        ),
    )

    print()
    print(
        "Reference bias correction"
    )
    print(
        "-------------------------"
    )

    print(
        f"{'Month':>5}"
        f"{'Rebuilt mean':>16}"
        f"{'Reference mean':>18}"
        f"{'Ratio':>12}"
    )

    print(
        "-" * 51
    )

    for month_number in MONTHS:

        print(
            f"{int(month_number):>5d}"
            f"{float(rebuilt_mean.sel(month=month_number).values):>16.3f}"
            f"{float(reference_monthly_mean.sel(month=month_number).values):>18.3f}"
            f"{float(reference_ratio.sel(month=month_number).values):>12.4f}"
        )


def print_output_summary(
    output_ds,
):
    """Print the final dataset and finite monthly sample counts."""

    corrected_full_variable = (
        full_model_variable_name()
    )

    print()
    print(
        "Output dataset"
    )
    print(
        "--------------"
    )
    print(
        output_ds
    )

    print()
    print(
        "Finite corrected full-sample values by month"
    )
    print(
        "--------------------------------------------"
    )

    for month_number in MONTHS:

        calendar_month = output_ds["sample_month"].astype("int64") % 100
        count = np.isfinite(
            output_ds[corrected_full_variable].where(calendar_month == month_number)
        ).sum().item()

        print(
            f"Month {int(month_number):>2}: "
            f"{count:>8}"
        )


# =============================================================================
# NetCDF writing
# =============================================================================

def write_output(
    output_ds,
    filename,
):
    """Write the compact bias-corrected dataset."""

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    corrected_variables = [
        variable_name
        for variable_name in output_ds.data_vars
        if (
            variable_name == "tp24_max"
            or variable_name.startswith(
                "tp24_max_lead"
            )
        )
    ]

    ratio_variables = [
        variable_name
        for variable_name in output_ds.data_vars
        if (
            variable_name == "bias_correction_ratio"
            or variable_name.startswith(
                "lead_time_bias_correction_ratio_"
            )
        )
    ]

    encoding = {}

    for variable_name in (
        corrected_variables
        + ratio_variables
    ):

        encoding[
            variable_name
        ] = {
            "dtype": "float32",
            "_FillValue": (
                np.float32(
                    np.nan
                )
                if variable_name in corrected_variables
                else None
            ),
            "zlib": True,
            "complevel": 4,
        }

    # NetCDF does not accept _FillValue=None in all backends.
    for variable_name in ratio_variables:

        encoding[
            variable_name
        ].pop(
            "_FillValue",
            None,
        )

    if "hdate" in output_ds:

        encoding[
            "hdate"
        ] = {
            "dtype": "int64",
        }

    output_ds.to_netcdf(
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

    filename_model = (
        make_model_filename()
    )

    filename_output = (
        make_output_filename(
            filename_model
        )
    )

    print(
        "Reading model file:     ",
        filename_model,
    )

    print(
        "Writing output file:    ",
        filename_output,
    )

    model_ds = xr.open_dataset(
        filename_model,
        decode_timedelta=False,
    )

    (
        reference_ds,
        reference_variable,
        filename_reference,
    ) = load_reference_dataset()

    print(
        "Reading reference file: ",
        filename_reference,
    )

    try:

        validate_model_dataset(
            model_ds
        )

        full_model_variable = (
            get_full_model_variable(
                model_ds
            )
        )

        split_variables = (
            get_split_model_variables(
                model_ds
            )
        )

        print()
        print(
            "Complete-window variable:"
        )
        print(
            "   ",
            full_model_variable,
        )

        print()
        print(
            "Split variables:"
        )

        for variable_name in split_variables:

            print(
                "   ",
                variable_name,
            )

        reference_monthly_mean = (
            get_reference_monthly_mean(
                reference_ds=reference_ds,
                reference_variable=reference_variable,
            )
        )

        # ---------------------------------------------------------------------
        # Stage 1: lead-time correction.
        # ---------------------------------------------------------------------

        lead_time_ratios = (
            calculate_lead_time_correction_ratios(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
                split_variables=split_variables,
            )
        )

        corrected_splits = (
            apply_lead_time_correction(
                model_ds=model_ds,
                split_variables=split_variables,
                lead_time_ratios=lead_time_ratios,
            )
        )

        rebuilt_full_sample = (
            rebuild_full_sample(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
                split_variables=split_variables,
                corrected_splits=corrected_splits,
            )
        )

        # ---------------------------------------------------------------------
        # Stage 2: reference correction.
        # ---------------------------------------------------------------------

        reference_ratio = (
            calculate_reference_bias_correction_ratio(
                model_ds=model_ds,
                rebuilt_full_sample=rebuilt_full_sample,
                reference_monthly_mean=reference_monthly_mean,
            )
        )

        print_lead_time_correction_table(
            model_ds=model_ds,
            full_model_variable=full_model_variable,
            split_variables=split_variables,
            lead_time_ratios=lead_time_ratios,
        )

        print_reference_correction_table(
            model_ds=model_ds,
            rebuilt_full_sample=rebuilt_full_sample,
            reference_monthly_mean=reference_monthly_mean,
            reference_ratio=reference_ratio,
        )

        output_ds = (
            build_final_bias_corrected_dataset(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
                split_variables=split_variables,
                corrected_splits=corrected_splits,
                rebuilt_full_sample=rebuilt_full_sample,
                reference_ratio=reference_ratio,
                lead_time_ratios=lead_time_ratios,
            )
        )

        print_output_summary(
            output_ds
        )

        if write2file:

            write_output(
                output_ds=output_ds,
                filename=filename_output,
            )

        output_ds.close()

    finally:

        model_ds.close()
        reference_ds.close()
