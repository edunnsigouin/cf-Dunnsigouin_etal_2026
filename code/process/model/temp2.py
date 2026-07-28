"""
Apply a Kelder et al. (2020)-style multiplicative bias correction to the
sampled monthly precipitation-extreme statistics from the S2S model.

The correction factor is calculated separately for each calendar month as:

    bias_correction_ratio = reference_mean / model_mean

where:
    - reference_mean is the mean monthly extreme in ERA5 or SeNorge
      across all available reference years;
    - model_mean is the mean of the FULL lead-window sampled model statistic
      for the same calendar month.

The same monthly ratio is then applied to:
    1. the full lead-window maximum variable; and
    2. every split lead-window maximum variable in the model file.

All original model variables are retained unchanged.
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
    "2023-06-26",
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

write2file = True

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

def calculate_monthly_bias_correction_ratio(
    model_ds,
    full_model_variable,
    reference_monthly_mean,
):
    """
    Calculate the Kelder-style multiplicative correction factor.

        ratio(month) = reference_mean(month) / model_mean(month)

    The model mean is calculated ONLY from the complete lead-window sampled
    statistic. The resulting monthly ratio is later applied to the complete
    distribution and all split lead-time distributions.
    """

    model_monthly_mean = model_ds[full_model_variable].mean(
        dim="index",
        skipna=True,
    )

    reference_mean_aligned, model_mean_aligned = xr.align(
        reference_monthly_mean,
        model_monthly_mean,
        join="exact",
    )

    if np.any(~np.isfinite(model_mean_aligned.values)):
        raise ValueError(
            "At least one monthly model mean is non-finite."
        )

    if np.any(model_mean_aligned.values <= 0):
        raise ValueError(
            "At least one monthly model mean is zero or negative; "
            "multiplicative precipitation bias correction is not valid."
        )

    if np.any(~np.isfinite(reference_mean_aligned.values)):
        raise ValueError(
            "At least one monthly reference mean is non-finite."
        )

    ratio = reference_mean_aligned / model_mean_aligned
    ratio.name = f"bias_correction_ratio_{REFERENCE_DATASET}"

    ratio.attrs = {
        "description": (
            "Monthly multiplicative bias-correction ratio calculated as "
            "reference mean divided by model mean using the complete "
            "lead-window sampled maximum statistic"
        ),
        "reference_dataset": REFERENCE_DATASET,
        "reference_year_start": reference_monthly_mean.attrs[
            "reference_year_start"
        ],
        "reference_year_end": reference_monthly_mean.attrs[
            "reference_year_end"
        ],
        "exclude_2023_from_reference": str(EXCLUDE_2023_FROM_REFERENCE),
        "formula": "reference_monthly_mean / model_monthly_mean",
        "model_variable_used_for_ratio": full_model_variable,
        "application": (
            "Same calendar-month ratio applied to the complete lead-window "
            "and all split lead-window maximum variables"
        ),
        "units": "1",
    }

    return ratio


def add_bias_corrected_variables(model_ds, ratio):
    """
    Copy the model dataset and add corrected maximum variables.

    The original maximum variables and all metadata/provenance variables are
    retained unchanged.
    """

    output_ds = model_ds.copy(deep=True)
    maximum_variables = get_model_maximum_variables(model_ds)
    suffix = f"_bc_{REFERENCE_DATASET}"

    for variable in maximum_variables:
        output_variable = f"{variable}{suffix}"

        # xarray broadcasts the 12 monthly ratios across the 'index' dimension.
        corrected = model_ds[variable] * ratio

        # Preserve original metadata and append bias-correction metadata.
        corrected.attrs = model_ds[variable].attrs.copy()

        original_description = corrected.attrs.get(
            "description",
            variable,
        )

        corrected.attrs["description"] = (
            f"{original_description}; multiplicatively bias corrected "
            f"relative to {REFERENCE_DATASET.upper()}"
        )
        corrected.attrs["bias_correction_method"] = (
            "multiplicative monthly mean scaling"
        )
        corrected.attrs["bias_correction_formula"] = (
            "corrected_value = original_value * "
            f"bias_correction_ratio_{REFERENCE_DATASET}"
        )
        corrected.attrs["bias_correction_reference"] = REFERENCE_DATASET

        output_ds[output_variable] = corrected

    output_ds[ratio.name] = ratio

    output_ds.attrs = model_ds.attrs.copy()
    output_ds.attrs["bias_correction"] = (
        f"Monthly multiplicative bias correction relative to "
        f"{REFERENCE_DATASET}. Ratio = reference monthly mean / model "
        "monthly mean, calculated from the complete lead-window sampled "
        "maximum and applied to all maximum variables."
    )
    output_ds.attrs["bias_correction_reference_dataset"] = REFERENCE_DATASET
    output_ds.attrs["bias_correction_reference_year_start"] = (
        ratio.attrs["reference_year_start"]
    )
    output_ds.attrs["bias_correction_reference_year_end"] = (
        ratio.attrs["reference_year_end"]
    )
    output_ds.attrs["bias_correction_exclude_2023"] = str(
        EXCLUDE_2023_FROM_REFERENCE
    )

    return output_ds


# =============================================================================
# Reporting
# =============================================================================

def print_bias_correction_table(
    model_ds,
    full_model_variable,
    reference_monthly_mean,
    ratio,
):
    """Print model means, reference means, and monthly correction factors."""

    model_monthly_mean = model_ds[full_model_variable].mean(
        dim="index",
        skipna=True,
    )

    print()
    print(
        "Reference period:",
        f"{reference_monthly_mean.attrs['reference_year_start']}-"
        f"{reference_monthly_mean.attrs['reference_year_end']}",
    )

    print()
    print("Monthly multiplicative bias correction")
    print("--------------------------------------")
    print(
        f"{'Month':>5}"
        f"{'Model mean':>15}"
        f"{'Reference mean':>18}"
        f"{'Ratio':>12}"
    )
    print("-" * 50)

    for month in range(1, 13):
        model_mean = float(
            model_monthly_mean.sel(month_of_year=month).values
        )
        reference_mean = float(
            reference_monthly_mean.sel(month_of_year=month).values
        )
        month_ratio = float(
            ratio.sel(month_of_year=month).values
        )

        print(
            f"{month:>5d}"
            f"{model_mean:>15.3f}"
            f"{reference_mean:>18.3f}"
            f"{month_ratio:>12.4f}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_reference_dataset()

    filename_model = make_model_filename()
    filename_output = make_output_filename(
        model_filename=filename_model,
        reference_name=REFERENCE_DATASET,
    )

    print("Reading model file:     ", filename_model)
    model_ds = xr.open_dataset(filename_model)

    reference_ds, reference_variable, filename_reference = (
        load_reference_dataset()
    )

    print("Reading reference file: ", filename_reference)
    print("Writing output file:    ", filename_output)

    try:
        full_model_variable = get_full_model_variable(model_ds)
        maximum_variables = get_model_maximum_variables(model_ds)

        print()
        print("Full lead-window variable used to calculate ratio:")
        print("   ", full_model_variable)

        print()
        print("Maximum variables to bias correct:")
        for variable in maximum_variables:
            print("   ", variable)

        reference_monthly_mean = get_reference_monthly_mean(
            reference_ds=reference_ds,
            reference_variable=reference_variable,
        )

        ratio = calculate_monthly_bias_correction_ratio(
            model_ds=model_ds,
            full_model_variable=full_model_variable,
            reference_monthly_mean=reference_monthly_mean,
        )

        print_bias_correction_table(
            model_ds=model_ds,
            full_model_variable=full_model_variable,
            reference_monthly_mean=reference_monthly_mean,
            ratio=ratio,
        )

        output_ds = add_bias_corrected_variables(
            model_ds=model_ds,
            ratio=ratio,
        )

        if write2file:
            output_ds.to_netcdf(filename_output)

            print()
            print("Finished.")
            print("Wrote:", filename_output)

        output_ds.close()
        
    finally:
        model_ds.close()
