"""
Apply a Kelder et al. (2020)-style multiplicative bias correction to the
sampled monthly precipitation-extreme statistics from the S2S model.

The correction is performed separately for each calendar month in two stages.

Stage 1: lead-time correction
-----------------------------
Each split lead-time distribution is multiplicatively scaled to the mean of
the ORIGINAL all-lead distribution:

    lead_time_ratio = original_all_lead_mean / split_lead_mean

The split NaN masks and (month_of_year, index) positions are preserved. The
corrected split samples are then recombined index-by-index to rebuild the
all-lead sample, so each index continues to reference the same initialization.

Stage 2: reference correction
-----------------------------
A Kelder et al. (2020)-style multiplicative correction is calculated from the
REBUILT all-lead sample:

    reference_ratio = reference_mean / rebuilt_all_lead_mean

The same reference ratio is applied to the rebuilt all-lead sample and all
lead-time-corrected split samples.

The output contains only the final bias-corrected max_value variables.
All other model variables are retained unchanged and index-aligned.
"""

import os

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"
x_days = 2

forecast_date_range = [
    "2020-01-02",
    "2022-12-29",
]

observation_years = [
    "1957",
    "2023",
]

era5_grid = "0.5x0.5"

# Must match the settings used to create the model sample file.
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Choose ONE reference dataset at a time: "era5" or "senorge".
REFERENCE_DATASET = "senorge"

# If True, exclude 2023 when calculating the reference monthly means.
# This is useful for excluding Storm Hans from the ERA5/SeNorge reference
# climatology used to calculate the bias-correction ratios.
#
# True  -> use 1957-2022
# False -> use 1957-2023
EXCLUDE_2023_FROM_REFERENCE = True

write2file = False

# =============================================================================
# Dataset-specific settings
# =============================================================================

MODEL_VARIABLE = "tp24"
ERA5_VARIABLE = "tp24"
SENORGE_VARIABLE = "rr"


# =============================================================================
# Lead-time and filename helpers
# =============================================================================

def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """Split usable accumulated ending leads into approximately equal bins."""

    number_of_leads = last_lead - first_lead + 1
    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    # As in the sample-building workflow, later bins receive extra leads.
    bin_sizes = [
        base_size + int(i >= number_of_bins - remainder)
        for i in range(number_of_bins)
    ]

    lead_bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        lead_bins.append((current_start, current_end))
        current_start = current_end + 1

    return lead_bins


def get_full_lead_range():
    """Return the complete usable accumulated lead range."""

    return (
        first_input_lead + x_days - 1,
        last_input_lead,
    )


def build_lead_bins():
    """Return the split lead bins used by the model sample file."""

    full_start, full_end = get_full_lead_range()

    return split_usable_accumulated_leads(
        first_lead=full_start,
        last_lead=full_end,
        number_of_bins=number_of_lead_bins,
    )


def lead_split_filename_label():
    """Return the lead-range label used in the model sample filename."""

    full_start, full_end = get_full_lead_range()

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in build_lead_bins()
    )

    return (
        f"lead{full_start}-{full_end}_"
        f"split{number_of_lead_bins}_"
        f"{split_text}"
    )


def make_model_filename():
    """Create the S2S model input filename."""

    return os.path.join(
        config.dirs["s2s_processed"],
        (
            "unseen_sample_monthly_catchment_precipitation_extremes_"
            f"{MODEL_VARIABLE}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_split_filename_label()}_"
            "forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )


def make_era5_filename():
    """Create the ERA5 input filename."""

    return (
        f"{config.dirs['era5_processed']}"
        "distribution_monthly_extremes_"
        f"{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{observation_years[0]}-{observation_years[1]}.nc"
    )


def make_senorge_filename():
    """Create the SeNorge input filename."""

    return (
        f"{config.dirs['senorge_processed']}"
        "distribution_monthly_extremes_"
        f"{SENORGE_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_senorge_"
        f"{observation_years[0]}-{observation_years[1]}.nc"
    )


def make_output_filename(model_filename, reference_name):
    """Append the reference dataset name to the model filename."""

    stem, extension = os.path.splitext(model_filename)
    return f"{stem}_bc_{reference_name}{extension}"


# =============================================================================
# Validation and variable discovery
# =============================================================================

def validate_reference_dataset():
    """Check that a supported reference dataset was selected."""

    valid = {"era5", "senorge"}

    if REFERENCE_DATASET not in valid:
        raise ValueError(
            f"REFERENCE_DATASET must be one of {sorted(valid)}. "
            f"Got '{REFERENCE_DATASET}'."
        )


def get_full_model_variable(model_ds):
    """
    Return the maximum variable for the complete usable lead window.

    Prefer the lead-range metadata stored in the NetCDF file itself.
    """

    try:
        lead_start = int(model_ds.attrs["first_usable_accumulated_lead"])
        lead_end = int(model_ds.attrs["last_usable_accumulated_lead"])
    except KeyError as exc:
        raise KeyError(
            "The model file must contain global attributes "
            "'first_usable_accumulated_lead' and "
            "'last_usable_accumulated_lead'."
        ) from exc

    variable = f"max_value_lead{lead_start}_{lead_end}"

    if variable not in model_ds:
        raise KeyError(
            f"Expected full lead-window variable '{variable}' "
            "was not found in the model dataset."
        )

    return variable


def get_model_maximum_variables(model_ds):
    """Return the full and split sampled maximum variables automatically."""

    variables = [
        variable
        for variable in model_ds.data_vars
        if variable.startswith("max_value_lead")
        and "_bc_" not in variable
    ]

    if not variables:
        raise ValueError(
            "No variables beginning with 'max_value_lead' "
            "were found in the model dataset."
        )

    return variables


# =============================================================================
# Reference data
# =============================================================================

def load_reference_dataset():
    """Open the selected reference dataset."""

    if REFERENCE_DATASET == "era5":
        filename = make_era5_filename()
        variable = ERA5_VARIABLE
    else:
        filename = make_senorge_filename()
        variable = SENORGE_VARIABLE

    ds = xr.open_dataset(filename)

    if variable not in ds:
        ds.close()
        raise KeyError(
            f"Variable '{variable}' was not found in "
            f"{REFERENCE_DATASET} file: {filename}"
        )

    return ds, variable, filename


def get_reference_monthly_mean(reference_ds, reference_variable):
    """
    Calculate the mean sampled extreme for each calendar month.

    Expected reference structure:
        reference_ds[reference_variable](year, month)
    """

    values = reference_ds[reference_variable]

    if "year" not in values.dims or "month" not in values.dims:
        raise ValueError(
            f"Reference variable '{reference_variable}' must have "
            "dimensions 'year' and 'month'."
        )

    # Optionally exclude 2023 so that Storm Hans does not influence
    # the monthly reference means used for bias correction.
    if EXCLUDE_2023_FROM_REFERENCE:
        values = values.sel(
            year=values["year"] < 2023
        )

    reference_monthly_mean = values.mean(
        dim="year",
        skipna=True,
    )

    # Record the actual reference period used.
    reference_year_start = int(values["year"].min().values)
    reference_year_end = int(values["year"].max().values)

    reference_monthly_mean.attrs["reference_year_start"] = reference_year_start
    reference_monthly_mean.attrs["reference_year_end"] = reference_year_end
    reference_monthly_mean.attrs["exclude_2023"] = str(
        EXCLUDE_2023_FROM_REFERENCE
    )

    # Rename to match the model calendar-month dimension.
    return reference_monthly_mean.rename(
        {"month": "month_of_year"}
    )


# =============================================================================
# Bias correction
# =============================================================================

def get_split_model_variables(
    model_ds,
    full_model_variable,
):
    """Return all split lead-time maximum variables."""

    maximum_variables = get_model_maximum_variables(
        model_ds
    )

    split_variables = [
        variable
        for variable in maximum_variables
        if variable != full_model_variable
    ]

    if len(split_variables) != number_of_lead_bins:
        raise ValueError(
            f"Expected {number_of_lead_bins} split variables, "
            f"but found {len(split_variables)}: {split_variables}"
        )

    return split_variables


def calculate_lead_time_correction_ratios(
    model_ds,
    full_model_variable,
    split_variables,
):
    """
    Calculate one multiplicative lead-time correction ratio per split/month.

    For each split:

        lead_time_ratio(month)
            = original_all_lead_mean(month)
              / original_split_mean(month)

    The ratio is calculated from finite values only.

    Applying this ratio preserves the original (month_of_year, index)
    structure and NaN mask of each split variable.
    """

    full_monthly_mean = (
        model_ds[
            full_model_variable
        ]
        .mean(
            dim="index",
            skipna=True,
        )
    )

    ratios = {}

    for variable in split_variables:

        split_monthly_mean = (
            model_ds[
                variable
            ]
            .mean(
                dim="index",
                skipna=True,
            )
        )

        if np.any(
            ~np.isfinite(
                split_monthly_mean.values
            )
        ):
            raise ValueError(
                f"At least one monthly mean is non-finite "
                f"for '{variable}'."
            )

        if np.any(
            split_monthly_mean.values <= 0
        ):
            raise ValueError(
                f"At least one monthly mean is zero or negative "
                f"for '{variable}'."
            )

        ratio = (
            full_monthly_mean
            / split_monthly_mean
        )

        ratio.name = (
            "lead_time_bias_correction_ratio_"
            + variable.replace(
                "max_value_",
                "",
            )
        )

        ratio.attrs = {
            "description": (
                "Monthly multiplicative lead-time correction ratio "
                "calculated as original all-lead mean divided by "
                "original split lead-time mean"
            ),
            "formula": (
                "original_all_lead_monthly_mean / "
                "original_split_monthly_mean"
            ),
            "target_variable": full_model_variable,
            "source_variable": variable,
            "units": "1",
        }

        ratios[
            variable
        ] = ratio

    return ratios


def apply_lead_time_correction(
    model_ds,
    split_variables,
    lead_time_ratios,
):
    """
    Apply lead-time correction while preserving every index position.

    Multiplication by a monthly ratio leaves NaNs as NaNs, so each split
    retains exactly the same (month_of_year, index) membership mask as the
    original model file.
    """

    corrected_splits = {}

    for variable in split_variables:

        corrected = (
            model_ds[
                variable
            ]
            * lead_time_ratios[
                variable
            ]
        )

        corrected.attrs = (
            model_ds[
                variable
            ]
            .attrs
            .copy()
        )

        corrected.attrs[
            "lead_time_bias_correction_method"
        ] = (
            "multiplicative monthly mean scaling"
        )

        corrected.attrs[
            "lead_time_bias_correction_formula"
        ] = (
            "corrected_split = original_split * "
            "original_all_lead_mean / original_split_mean"
        )

        corrected.attrs[
            "lead_time_bias_correction_target"
        ] = (
            "original all-lead monthly mean"
        )

        corrected_splits[
            variable
        ] = corrected

    return corrected_splits


def rebuild_all_lead_sample(
    model_ds,
    full_model_variable,
    split_variables,
    corrected_splits,
):
    """
    Rebuild the all-lead sample index-by-index from corrected split samples.

    The expected model-file structure is:

        all lead:   finite finite finite ...
        split 1:    finite NaN    finite ...
        split 2:    NaN    finite NaN    ...

    Thus every finite original all-lead value should correspond to exactly
    one finite split value at the SAME (month_of_year, index).

    The rebuilt all-lead sample therefore preserves the unique initialization
    represented by each index, along with date_of_max, forecast_date,
    ensemble_member, and the other provenance variables.
    """

    original_full = (
        model_ds[
            full_model_variable
        ]
    )

    # Count how many ORIGINAL split variables are finite at every
    # (month_of_year, index).
    finite_split_count = xr.zeros_like(
        original_full,
        dtype=np.int16,
    )

    for variable in split_variables:

        finite_split_count = (
            finite_split_count
            + xr.where(
                np.isfinite(
                    model_ds[
                        variable
                    ]
                ),
                1,
                0,
            )
        )

    original_full_finite = np.isfinite(
        original_full
    )

    # Wherever the all-lead value exists, exactly one split must exist.
    bad_finite = (
        original_full_finite
        & (finite_split_count != 1)
    )

    # Wherever the all-lead value is NaN, all split values must also be NaN.
    bad_missing = (
        (~original_full_finite)
        & (finite_split_count != 0)
    )

    if bool(
        bad_finite.any().values
        or bad_missing.any().values
    ):
        raise ValueError(
            "Split lead-time variables do not form an index-aligned "
            "partition of the original all-lead sample. Expected exactly "
            "one finite split value for every finite all-lead "
            "(month_of_year, index), and no finite split values where "
            "the all-lead value is NaN."
        )

    # Start as all NaN with exactly the same dimensions and coordinates
    # as the original all-lead variable.
    rebuilt = xr.full_like(
        original_full,
        np.nan,
        dtype=np.float64,
    )

    # Insert each corrected split into its original index positions.
    for variable in split_variables:

        split_corrected = (
            corrected_splits[
                variable
            ]
        )

        rebuilt = xr.where(
            np.isfinite(
                split_corrected
            ),
            split_corrected,
            rebuilt,
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
        "description"
    ] = (
        rebuilt.attrs.get(
            "description",
            full_model_variable,
        )
        + "; rebuilt index-by-index from lead-time-corrected split samples"
    )

    rebuilt.attrs[
        "lead_time_rebuild_method"
    ] = (
        "At each original (month_of_year, index), the finite corrected "
        "split value was inserted into the corresponding all-lead position."
    )

    return rebuilt


def calculate_reference_bias_correction_ratio(
    rebuilt_all_lead,
    full_model_variable,
    reference_monthly_mean,
):
    """
    Calculate the reference correction from the rebuilt all-lead sample.

        reference_ratio(month)
            = reference_mean(month)
              / rebuilt_all_lead_mean(month)
    """

    rebuilt_monthly_mean = (
        rebuilt_all_lead
        .mean(
            dim="index",
            skipna=True,
        )
    )

    (
        reference_mean_aligned,
        rebuilt_mean_aligned,
    ) = xr.align(
        reference_monthly_mean,
        rebuilt_monthly_mean,
        join="exact",
    )

    if np.any(
        ~np.isfinite(
            rebuilt_mean_aligned.values
        )
    ):
        raise ValueError(
            "At least one rebuilt all-lead monthly mean is non-finite."
        )

    if np.any(
        rebuilt_mean_aligned.values <= 0
    ):
        raise ValueError(
            "At least one rebuilt all-lead monthly mean is zero or negative."
        )

    if np.any(
        ~np.isfinite(
            reference_mean_aligned.values
        )
    ):
        raise ValueError(
            "At least one monthly reference mean is non-finite."
        )

    ratio = (
        reference_mean_aligned
        / rebuilt_mean_aligned
    )

    ratio.name = (
        f"bias_correction_ratio_"
        f"{REFERENCE_DATASET}"
    )

    ratio.attrs = {
        "description": (
            "Monthly multiplicative reference bias-correction ratio "
            "calculated as reference mean divided by rebuilt "
            "lead-time-corrected all-lead mean"
        ),
        "reference_dataset": REFERENCE_DATASET,
        "reference_year_start": reference_monthly_mean.attrs[
            "reference_year_start"
        ],
        "reference_year_end": reference_monthly_mean.attrs[
            "reference_year_end"
        ],
        "exclude_2023_from_reference": str(
            EXCLUDE_2023_FROM_REFERENCE
        ),
        "formula": (
            "reference_monthly_mean / "
            "rebuilt_all_lead_monthly_mean"
        ),
        "model_variable_used_for_ratio": full_model_variable,
        "units": "1",
    }

    return ratio


def build_final_bias_corrected_dataset(
    model_ds,
    full_model_variable,
    split_variables,
    corrected_splits,
    rebuilt_all_lead,
    reference_ratio,
    lead_time_ratios,
):
    """
    Build the final output dataset.

    Final workflow:
        1. split samples corrected relative to original all-lead mean;
        2. all-lead sample rebuilt at the same index positions;
        3. one reference correction calculated from rebuilt all-lead sample;
        4. same reference ratio applied to rebuilt all-lead and all splits.

    The original uncorrected max_value variables are omitted.
    All non-maximum variables remain unchanged and index-aligned.
    """

    maximum_variables = (
        get_model_maximum_variables(
            model_ds
        )
    )

    output_ds = (
        model_ds
        .drop_vars(
            maximum_variables
        )
        .copy(
            deep=True
        )
    )

    suffix = (
        f"_bc_"
        f"{REFERENCE_DATASET}"
    )

    # Final rebuilt all-lead sample.
    final_all = (
        rebuilt_all_lead
        * reference_ratio
    )

    final_all.attrs = (
        rebuilt_all_lead
        .attrs
        .copy()
    )

    final_all.attrs[
        "bias_correction_method"
    ] = (
        "two-stage multiplicative correction: lead-time correction "
        "followed by reference correction"
    )

    final_all.attrs[
        "bias_correction_formula"
    ] = (
        "rebuilt_all_lead * reference_ratio"
    )

    final_all.attrs[
        "bias_correction_reference"
    ] = (
        REFERENCE_DATASET
    )

    output_ds[
        f"{full_model_variable}{suffix}"
    ] = final_all

    # Final corrected split samples.
    for variable in split_variables:

        final_split = (
            corrected_splits[
                variable
            ]
            * reference_ratio
        )

        final_split.attrs = (
            corrected_splits[
                variable
            ]
            .attrs
            .copy()
        )

        final_split.attrs[
            "bias_correction_method"
        ] = (
            "two-stage multiplicative correction: split lead-time "
            "correction followed by common reference correction"
        )

        final_split.attrs[
            "bias_correction_formula"
        ] = (
            "original_split * lead_time_ratio * reference_ratio"
        )

        final_split.attrs[
            "bias_correction_reference"
        ] = (
            REFERENCE_DATASET
        )

        final_split.attrs[
            "original_variable"
        ] = (
            variable
        )

        output_ds[
            f"{variable}{suffix}"
        ] = final_split

    # Store common reference correction ratio.
    output_ds[
        reference_ratio.name
    ] = (
        reference_ratio
    )

    # Store split-specific lead-time correction ratios.
    for ratio in lead_time_ratios.values():

        output_ds[
            ratio.name
        ] = (
            ratio
        )

    output_ds.attrs = (
        model_ds
        .attrs
        .copy()
    )

    output_ds.attrs[
        "bias_correction"
    ] = (
        "Two-stage monthly multiplicative correction. First, each split "
        "lead-time sample is scaled to the original all-lead monthly mean "
        "without changing its index membership. Second, the all-lead sample "
        "is rebuilt index-by-index from the corrected split samples. Third, "
        "a common reference correction is calculated from the rebuilt "
        "all-lead sample and applied to the rebuilt all-lead and all splits."
    )

    output_ds.attrs[
        "bias_correction_reference_dataset"
    ] = (
        REFERENCE_DATASET
    )

    output_ds.attrs[
        "bias_correction_reference_year_start"
    ] = (
        reference_ratio.attrs[
            "reference_year_start"
        ]
    )

    output_ds.attrs[
        "bias_correction_reference_year_end"
    ] = (
        reference_ratio.attrs[
            "reference_year_end"
        ]
    )

    output_ds.attrs[
        "bias_correction_exclude_2023"
    ] = str(
        EXCLUDE_2023_FROM_REFERENCE
    )

    output_ds.attrs[
        "index_alignment"
    ] = (
        "Preserved. Each (month_of_year, index) retains the original "
        "initialization/provenance represented by date_of_max, "
        "forecast_date, hdate, ensemble_member, model_type, and source_file."
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
    """Print monthly lead-time means and correction ratios."""

    full_mean = (
        model_ds[
            full_model_variable
        ]
        .mean(
            dim="index",
            skipna=True,
        )
    )

    print()
    print(
        "Lead-time correction"
    )
    print(
        "--------------------"
    )

    for variable in split_variables:

        split_mean = (
            model_ds[
                variable
            ]
            .mean(
                dim="index",
                skipna=True,
            )
        )

        ratio = (
            lead_time_ratios[
                variable
            ]
        )

        print()
        print(
            variable
        )

        print(
            f"{'Month':>5}"
            f"{'All mean':>14}"
            f"{'Split mean':>14}"
            f"{'Lead ratio':>14}"
        )

        print(
            "-" * 47
        )

        for month in range(
            1,
            13,
        ):

            print(
                f"{month:>5d}"
                f"{float(full_mean.sel(month_of_year=month).values):>14.3f}"
                f"{float(split_mean.sel(month_of_year=month).values):>14.3f}"
                f"{float(ratio.sel(month_of_year=month).values):>14.4f}"
            )


def print_reference_correction_table(
    rebuilt_all_lead,
    reference_monthly_mean,
    reference_ratio,
):
    """Print rebuilt all-lead means and reference correction ratios."""

    rebuilt_mean = (
        rebuilt_all_lead
        .mean(
            dim="index",
            skipna=True,
        )
    )

    print()
    print(
        "Reference period:",
        f"{reference_monthly_mean.attrs['reference_year_start']}-"
        f"{reference_monthly_mean.attrs['reference_year_end']}",
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

    for month in range(
        1,
        13,
    ):

        print(
            f"{month:>5d}"
            f"{float(rebuilt_mean.sel(month_of_year=month).values):>16.3f}"
            f"{float(reference_monthly_mean.sel(month_of_year=month).values):>18.3f}"
            f"{float(reference_ratio.sel(month_of_year=month).values):>12.4f}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_reference_dataset()

    filename_model = (
        make_model_filename()
    )

    filename_output = (
        make_output_filename(
            model_filename=filename_model,
            reference_name=REFERENCE_DATASET,
        )
    )

    print(
        "Reading model file:     ",
        filename_model,
    )

    model_ds = xr.open_dataset(
        filename_model
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

    print(
        "Writing output file:    ",
        filename_output,
    )

    try:

        # ---------------------------------------------------------------------
        # Identify the original all-lead and split variables.
        # ---------------------------------------------------------------------

        full_model_variable = (
            get_full_model_variable(
                model_ds
            )
        )

        split_variables = (
            get_split_model_variables(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
            )
        )

        print()
        print(
            "Original all-lead variable:"
        )
        print(
            "   ",
            full_model_variable,
        )

        print()
        print(
            "Split lead-time variables:"
        )

        for variable in split_variables:
            print(
                "   ",
                variable,
            )

        # ---------------------------------------------------------------------
        # Reference monthly means.
        # ---------------------------------------------------------------------

        reference_monthly_mean = (
            get_reference_monthly_mean(
                reference_ds=reference_ds,
                reference_variable=reference_variable,
            )
        )

        # ---------------------------------------------------------------------
        # 1. Correct split lead-time samples relative to the original all-lead
        #    monthly mean. Index positions and NaN masks are preserved.
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

        # ---------------------------------------------------------------------
        # 2. Rebuild the all-lead sample at the SAME (month_of_year, index)
        #    positions from the corrected split samples.
        # ---------------------------------------------------------------------

        rebuilt_all_lead = (
            rebuild_all_lead_sample(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
                split_variables=split_variables,
                corrected_splits=corrected_splits,
            )
        )

        # ---------------------------------------------------------------------
        # 3. Calculate a reference correction from the rebuilt all-lead sample.
        # ---------------------------------------------------------------------

        reference_ratio = (
            calculate_reference_bias_correction_ratio(
                rebuilt_all_lead=rebuilt_all_lead,
                full_model_variable=full_model_variable,
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
            rebuilt_all_lead=rebuilt_all_lead,
            reference_monthly_mean=reference_monthly_mean,
            reference_ratio=reference_ratio,
        )

        # ---------------------------------------------------------------------
        # 4. Apply the reference ratio to rebuilt all-lead and corrected split
        #    samples and create the output dataset.
        # ---------------------------------------------------------------------

        output_ds = (
            build_final_bias_corrected_dataset(
                model_ds=model_ds,
                full_model_variable=full_model_variable,
                split_variables=split_variables,
                corrected_splits=corrected_splits,
                rebuilt_all_lead=rebuilt_all_lead,
                reference_ratio=reference_ratio,
                lead_time_ratios=lead_time_ratios,
            )
        )

        if write2file:

            output_ds.to_netcdf(
                filename_output
            )

            print()
            print(
                "Finished."
            )

            print(
                "Wrote:",
                filename_output,
            )

        output_ds.close()

    finally:

        model_ds.close()
        reference_ds.close()
