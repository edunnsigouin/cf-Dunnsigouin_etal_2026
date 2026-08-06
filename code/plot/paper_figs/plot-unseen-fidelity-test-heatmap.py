#!/usr/bin/env python3
"""
Create a raw-versus-bias-corrected fidelity-test heatmap for all 12 months.

This script combines:

1. the heatmap summary used by the raw-only fidelity script; and
2. the four fidelity calculations used by the monthly six-panel UNSEEN
   diagnostic script.

Only the four fidelity tests are performed:

    mean
    standard deviation
    skewness
    kurtosis

The independence and stability tests are not calculated here.

Input model structure
---------------------
The raw compact model sample file is expected to contain:

    tp24_max(number, i_date)
    month(i_date)

The bias-corrected compact model sample file is expected to contain:

    tp24_max_bc_<reference>(number, i_date)
    month(i_date)

where <reference> is either "era5" or "senorge".

Calendar-month samples
----------------------
For each month:

1. Select every i_date for which month(i_date) equals that month.
2. Pool all finite ensemble-member values across number and i_date.
3. Extract the corresponding reference monthly-extreme sample.
4. Draw bootstrap samples from the raw and bias-corrected model samples using
   the same bootstrap indices.
5. Calculate bootstrap distributions of mean, sample standard deviation,
   skewness, and excess kurtosis.
6. A fidelity test passes when the reference statistic falls inside the
   selected central bootstrap confidence interval.

Heatmap
-------
The output heatmap has:

    rows:
        raw
        bias corrected

    columns:
        mean
        std
        skewness
        kurtosis

Each cell contains the number of calendar months, out of 12, that pass the
corresponding fidelity test.

Optional selective bias correction
----------------------------------
When BIAS_CORRECT_ONLY_FAILED_MONTHS is True, the raw fidelity tests are first
evaluated for each month. A month that fails at least one of the four raw tests
uses the bias-corrected sample for all four tests in the "bias corrected" row.
A month that passes all four raw tests keeps the original raw sample in that
row. The row remains labelled "bias corrected".

The plot layout and colour logic follow the original raw-only heatmap:
darker shading indicates fewer failed months, and the cell text gives the
number of passed months.
"""

import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kurtosis, skew

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Accumulation period used in the compact model and reference sample files.
x_days = 2

catchment = "regine_drammen"

forecast_date_range = (
    "2020-01-02",
    "2022-12-29",
)

reference_years = (
    "1957",
    "2022",
)

era5_grid = "0.5x0.5"

# Must match the compact sample file.
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Reference used for both:
#   1. the observed monthly statistic; and
#   2. selecting the corresponding bias-corrected model file.
#
# Options:
#     "era5"
#     "senorge"
REFERENCE_DATASET = "era5"

# Bias-correction application mode.
#
# False:
#     The "bias corrected" heatmap row uses the bias-corrected sample for
#     every calendar month.
#
# True:
#     First run all four fidelity tests on the raw sample for each month.
#     If a month fails at least one raw test, use the bias-corrected sample
#     for that entire month. If it passes all four raw tests, retain the raw
#     sample for that month.
#
# The resulting heatmap row is still labelled "bias corrected".
BIAS_CORRECT_ONLY_FAILED_MONTHS = True

number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0
random_seed = 42

# Optional explicit input filenames.
#
# Leave as None to construct filenames automatically.
raw_model_filename_override = None
bias_corrected_model_filename_override = None
reference_filename_override = None

write2file = False
show_figure = True

path_out = Path(
    config.dirs[
        "fig"
    ]
)

filename_heatmap = (
    path_out
    / (
        f"fidelity_heatmap_raw_bc_"
        f"{REFERENCE_DATASET}_"
        f"{x_days}dayacc_"
        f"{catchment}_"
        f"{forecast_date_range[0]}_"
        f"{forecast_date_range[1]}.png"
    )
)


# =============================================================================
# Dataset settings
# =============================================================================

MODEL_VARIABLE = "tp24"
MODEL_MONTH_VARIABLE = "month"

ERA5_VARIABLE = "tp24"

SENORGE_VARIABLE = "rr"
SENORGE_LABEL = "SeNorge"

MONTH_LABELS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

STATISTICS = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
)

HEATMAP_COLUMN_LABELS = {
    "mean": "mean",
    "std": "std",
    "skewness": "skewness",
    "kurtosis": "kurtosis",
}


# =============================================================================
# General helpers
# =============================================================================

def get_file_id(
    catchment_name: str,
) -> str:
    """Return the short catchment label used in compact sample filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


def remove_missing_values(
    values: np.ndarray,
) -> np.ndarray:
    """Flatten an array and retain only finite values."""

    values = np.asarray(
        values
    ).ravel()

    return values[
        np.isfinite(
            values
        )
    ]


# =============================================================================
# Lead-time and filename helpers
# =============================================================================

def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """Split usable ending leads into approximately equal consecutive bins."""

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
            index
            >= number_of_bins - remainder
        )
        for index in range(
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


def get_lead_ranges():
    """Return the full accumulated lead range and all lead-location bins."""

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    full_range = (
        first_usable_lead,
        last_input_lead,
    )

    split_ranges = (
        split_usable_accumulated_leads(
            first_lead=first_usable_lead,
            last_lead=last_input_lead,
            number_of_bins=number_of_lead_bins,
        )
    )

    return (
        full_range,
        split_ranges,
    )


def lead_filename_label() -> str:
    """Return the lead-window label used in compact sample filenames."""

    full_range, split_ranges = (
        get_lead_ranges()
    )

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in split_ranges
    )

    return (
        f"lead{full_range[0]}-{full_range[1]}_"
        f"split{number_of_lead_bins}_"
        f"{split_text}"
    )


def build_raw_model_filename() -> Path:
    """Build the raw compact model sample filename."""

    if raw_model_filename_override is not None:
        return Path(
            raw_model_filename_override
        )

    return (
        Path(
            config.dirs[
                "s2s_processed"
            ]
        )
        / (
            f"monthly_max_samples_"
            f"{MODEL_VARIABLE}_"
            f"{x_days}dayacc_"
            f"{get_file_id(catchment)}_"
            f"{lead_filename_label()}_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        )
    )


def build_bias_corrected_model_filename() -> Path:
    """Build the bias-corrected compact model sample filename."""

    if bias_corrected_model_filename_override is not None:
        return Path(
            bias_corrected_model_filename_override
        )

    raw_filename = (
        build_raw_model_filename()
    )

    return raw_filename.with_name(
        (
            f"{raw_filename.stem}_"
            f"bc_{REFERENCE_DATASET}"
            f"{raw_filename.suffix}"
        )
    )


def build_era5_filename() -> Path:
    """Build the ERA5 monthly-extreme reference filename."""

    return Path(
        (
            f"{config.dirs['era5_processed']}"
            f"distribution_monthly_extremes_"
            f"{ERA5_VARIABLE}_{x_days}dayacc_"
            f"{catchment}_era5_{era5_grid}_"
            f"{reference_years[0]}-"
            f"{reference_years[1]}.nc"
        )
    )


def build_senorge_filename() -> Path:
    """Build the SeNorge monthly-extreme reference filename."""

    return Path(
        (
            f"{config.dirs['senorge_processed']}"
            f"distribution_monthly_extremes_"
            f"{SENORGE_VARIABLE}_{x_days}dayacc_"
            f"{catchment}_senorge_"
            f"{reference_years[0]}-"
            f"{reference_years[1]}.nc"
        )
    )


def get_reference_configuration() -> tuple[Path, str, str]:
    """Return reference filename, variable name, and display label."""

    if reference_filename_override is not None:

        filename = Path(
            reference_filename_override
        )

        if REFERENCE_DATASET == "era5":
            return (
                filename,
                ERA5_VARIABLE,
                "ERA5",
            )

        return (
            filename,
            SENORGE_VARIABLE,
            SENORGE_LABEL,
        )

    if REFERENCE_DATASET == "era5":

        return (
            build_era5_filename(),
            ERA5_VARIABLE,
            "ERA5",
        )

    return (
        build_senorge_filename(),
        SENORGE_VARIABLE,
        SENORGE_LABEL,
    )


def get_model_variable_names() -> tuple[str, str]:
    """Return raw and bias-corrected complete-window variable names."""

    raw_variable = "tp24_max"

    bias_corrected_variable = (
        f"{raw_variable}_"
        f"bc_{REFERENCE_DATASET}"
    )

    return (
        raw_variable,
        bias_corrected_variable,
    )


# =============================================================================
# Validation
# =============================================================================

def validate_user_settings() -> None:
    """Validate settings and input files."""

    if x_days < 1:
        raise ValueError(
            "x_days must be at least 1."
        )

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available lead range."
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
            "of usable accumulated leads."
        )

    if number_of_bootstrap_samples < 1:
        raise ValueError(
            "number_of_bootstrap_samples must be at least 1."
        )

    if not (
        0.0
        < confidence_level_percent
        < 100.0
    ):
        raise ValueError(
            "confidence_level_percent must be between 0 and 100."
        )

    valid_references = {
        "era5",
        "senorge",
    }

    if REFERENCE_DATASET not in valid_references:
        raise ValueError(
            f"REFERENCE_DATASET must be one of "
            f"{sorted(valid_references)}."
        )

    (
        reference_filename,
        _,
        reference_label,
    ) = get_reference_configuration()

    files = {
        "Raw model": build_raw_model_filename(),
        "Bias-corrected model": (
            build_bias_corrected_model_filename()
        ),
        reference_label: reference_filename,
    }

    for label, filename in files.items():

        if not filename.is_file():
            raise FileNotFoundError(
                f"{label} file not found: {filename}"
            )


def check_model_dataset(
    ds: xr.Dataset,
    variable_name: str,
    dataset_label: str,
) -> None:
    """Check the compact model dataset structure."""

    required_variables = {
        variable_name,
        MODEL_MONTH_VARIABLE,
    }

    missing = (
        required_variables
        - set(
            ds.variables
        )
    )

    if missing:
        raise KeyError(
            f"{dataset_label} is missing variables: "
            f"{sorted(missing)}"
        )

    expected_dimensions = {
        "number",
        "i_date",
    }

    if set(
        ds[
            variable_name
        ].dims
    ) != expected_dimensions:
        raise ValueError(
            f"{dataset_label} variable '{variable_name}' must contain "
            f"dimensions {sorted(expected_dimensions)}, but has "
            f"{ds[variable_name].dims}."
        )

    if ds[
        MODEL_MONTH_VARIABLE
    ].dims != (
        "i_date",
    ):
        raise ValueError(
            f"{dataset_label} month variable must have dimensions "
            "('i_date',)."
        )


# =============================================================================
# Monthly sample extraction
# =============================================================================

def get_model_values_for_month(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
) -> np.ndarray:
    """Pool finite model values across number and selected i_date rows."""

    selected = ds[
        variable_name
    ].where(
        ds[
            MODEL_MONTH_VARIABLE
        ]
        == month_number,
        drop=True,
    )

    return remove_missing_values(
        selected.values
    )


def get_reference_values_for_month(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
) -> np.ndarray:
    """Return finite reference monthly-extreme values for one month."""

    if variable_name not in ds:
        raise KeyError(
            f"Reference variable '{variable_name}' was not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    data = ds[
        variable_name
    ]

    available_names = (
        set(
            data.coords
        )
        | set(
            data.dims
        )
    )

    if "month" not in available_names:
        raise KeyError(
            "Reference variable must contain a 'month' coordinate "
            "or dimension."
        )

    return remove_missing_values(
        data.sel(
            month=month_number
        ).values
    )


# =============================================================================
# Fidelity statistics and bootstrap
# =============================================================================

def calculate_statistic(
    values: np.ndarray,
    statistic_name: str,
) -> float:
    """Calculate one statistic exactly as in the monthly diagnostic script."""

    if statistic_name == "mean":

        return float(
            np.mean(
                values
            )
        )

    if statistic_name == "std":

        return float(
            np.std(
                values,
                ddof=1,
            )
        )

    if statistic_name == "skewness":

        return float(
            skew(
                values,
                bias=True,
            )
        )

    if statistic_name == "kurtosis":

        return float(
            kurtosis(
                values,
                fisher=True,
                bias=True,
            )
        )

    raise ValueError(
        f"Unsupported statistic: {statistic_name}"
    )


def get_vectorized_statistic_function(
    statistic_name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a statistic function operating along bootstrap axis 1."""

    if statistic_name == "mean":

        return lambda samples: np.mean(
            samples,
            axis=1,
        )

    if statistic_name == "std":

        return lambda samples: np.std(
            samples,
            axis=1,
            ddof=1,
        )

    if statistic_name == "skewness":

        return lambda samples: skew(
            samples,
            axis=1,
            bias=True,
        )

    if statistic_name == "kurtosis":

        return lambda samples: kurtosis(
            samples,
            axis=1,
            fisher=True,
            bias=True,
        )

    raise ValueError(
        f"Unsupported statistic: {statistic_name}"
    )


def calculate_confidence_interval(
    bootstrap_values: np.ndarray,
) -> tuple[float, float]:
    """Return the central bootstrap confidence interval."""

    alpha_percent = (
        100.0
        - confidence_level_percent
    )

    lower = np.percentile(
        bootstrap_values,
        alpha_percent / 2.0,
    )

    upper = np.percentile(
        bootstrap_values,
        100.0 - alpha_percent / 2.0,
    )

    return (
        float(
            lower
        ),
        float(
            upper
        ),
    )


def validate_month_samples(
    raw_values: np.ndarray,
    bias_corrected_values: np.ndarray,
    reference_values: np.ndarray,
    month_number: int,
) -> None:
    """Validate samples needed by the four monthly fidelity tests."""

    minimum_sample_size = 4

    for label, values in (
        (
            "raw model",
            raw_values,
        ),
        (
            "bias-corrected model",
            bias_corrected_values,
        ),
        (
            "reference",
            reference_values,
        ),
    ):

        if values.size < minimum_sample_size:
            raise ValueError(
                f"Only {values.size} finite {label} values were found "
                f"for {MONTH_LABELS[month_number]}. At least "
                f"{minimum_sample_size} are required."
            )

    if (
        raw_values.size
        != bias_corrected_values.size
    ):
        raise ValueError(
            "Raw and bias-corrected samples have different finite "
            f"sizes for {MONTH_LABELS[month_number]}: "
            f"raw={raw_values.size}, "
            f"bias corrected={bias_corrected_values.size}."
        )


def perform_month_fidelity_tests(
    raw_values: np.ndarray,
    comparison_values: np.ndarray,
    reference_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    """
    Perform all four fidelity tests for one calendar month.

    comparison_values contains either:
        - the fully bias-corrected monthly sample; or
        - the raw monthly sample when selective correction keeps that month raw.

    The same bootstrap indices are used for raw and comparison samples.
    """

    sample_size = (
        reference_values.size
    )

    sample_indices = rng.integers(
        low=0,
        high=raw_values.size,
        size=(
            number_of_bootstrap_samples,
            sample_size,
        ),
    )

    raw_resampled = raw_values[
        sample_indices
    ]

    comparison_resampled = comparison_values[
        sample_indices
    ]

    results = {}

    for statistic_name in STATISTICS:

        statistic_function = (
            get_vectorized_statistic_function(
                statistic_name
            )
        )

        raw_bootstrap = remove_missing_values(
            statistic_function(
                raw_resampled
            )
        )

        comparison_bootstrap = remove_missing_values(
            statistic_function(
                comparison_resampled
            )
        )

        if (
            raw_bootstrap.size == 0
            or comparison_bootstrap.size == 0
        ):
            raise ValueError(
                f"No finite bootstrap {statistic_name} values "
                "were produced."
            )

        raw_low, raw_high = (
            calculate_confidence_interval(
                raw_bootstrap
            )
        )

        comparison_low, comparison_high = (
            calculate_confidence_interval(
                comparison_bootstrap
            )
        )

        reference_value = calculate_statistic(
            reference_values,
            statistic_name,
        )

        results[
            statistic_name
        ] = {
            "reference_value": (
                reference_value
            ),
            "raw_low": (
                raw_low
            ),
            "raw_high": (
                raw_high
            ),
            "bc_low": (
                comparison_low
            ),
            "bc_high": (
                comparison_high
            ),
            "raw_passes": bool(
                raw_low
                <= reference_value
                <= raw_high
            ),
            "bc_passes": bool(
                comparison_low
                <= reference_value
                <= comparison_high
            ),
            "raw_sample_size": int(
                raw_values.size
            ),
            "bc_sample_size": int(
                comparison_values.size
            ),
            "reference_sample_size": int(
                reference_values.size
            ),
        }

    return results


def calculate_all_months_fidelity(
    raw_model_ds: xr.Dataset,
    bias_corrected_model_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    raw_variable: str,
    bias_corrected_variable: str,
    reference_variable: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the four fidelity tests over all months.

    The calculation is performed in two passes.

    Pass 1:
        Run all four raw fidelity tests and determine whether each month fails
        at least one test.

    Pass 2:
        Build the sample used for the "bias corrected" row.

        If BIAS_CORRECT_ONLY_FAILED_MONTHS is False:
            use bias-corrected values for every month.

        If BIAS_CORRECT_ONLY_FAILED_MONTHS is True:
            use bias-corrected values only for months that failed at least one
            raw fidelity test; retain raw values for months that passed all
            four raw tests.
    """

    monthly_samples = {}

    # -------------------------------------------------------------------------
    # Pass 1: collect samples and determine raw pass/fail status.
    # -------------------------------------------------------------------------

    for month_number in range(
        1,
        13,
    ):

        raw_values = get_model_values_for_month(
            ds=raw_model_ds,
            variable_name=raw_variable,
            month_number=month_number,
        )

        fully_bias_corrected_values = (
            get_model_values_for_month(
                ds=bias_corrected_model_ds,
                variable_name=bias_corrected_variable,
                month_number=month_number,
            )
        )

        reference_values = (
            get_reference_values_for_month(
                ds=reference_ds,
                variable_name=reference_variable,
                month_number=month_number,
            )
        )

        validate_month_samples(
            raw_values=raw_values,
            bias_corrected_values=fully_bias_corrected_values,
            reference_values=reference_values,
            month_number=month_number,
        )

        raw_rng = np.random.default_rng(
            random_seed
            + month_number
        )

        raw_only_results = (
            perform_month_fidelity_tests(
                raw_values=raw_values,
                comparison_values=raw_values,
                reference_values=reference_values,
                rng=raw_rng,
            )
        )

        raw_failed_any_test = any(
            not raw_only_results[
                statistic_name
            ][
                "raw_passes"
            ]
            for statistic_name in STATISTICS
        )

        if (
            BIAS_CORRECT_ONLY_FAILED_MONTHS
            and not raw_failed_any_test
        ):
            comparison_values = raw_values
            correction_applied = False
        else:
            comparison_values = fully_bias_corrected_values
            correction_applied = True

        monthly_samples[
            month_number
        ] = {
            "raw_values": raw_values,
            "comparison_values": comparison_values,
            "reference_values": reference_values,
            "raw_failed_any_test": raw_failed_any_test,
            "correction_applied": correction_applied,
        }

    # -------------------------------------------------------------------------
    # Pass 2: run paired raw versus selected comparison calculations.
    # -------------------------------------------------------------------------

    counts = {
        "raw": {
            statistic_name: 0
            for statistic_name in STATISTICS
        },
        "bias corrected": {
            statistic_name: 0
            for statistic_name in STATISTICS
        },
    }

    monthly_rows = []

    for month_number in range(
        1,
        13,
    ):

        samples = monthly_samples[
            month_number
        ]

        rng = np.random.default_rng(
            random_seed
            + month_number
        )

        month_results = (
            perform_month_fidelity_tests(
                raw_values=samples[
                    "raw_values"
                ],
                comparison_values=samples[
                    "comparison_values"
                ],
                reference_values=samples[
                    "reference_values"
                ],
                rng=rng,
            )
        )

        for statistic_name in STATISTICS:

            result = month_results[
                statistic_name
            ]

            counts[
                "raw"
            ][
                statistic_name
            ] += int(
                result[
                    "raw_passes"
                ]
            )

            counts[
                "bias corrected"
            ][
                statistic_name
            ] += int(
                result[
                    "bc_passes"
                ]
            )

            monthly_rows.append(
                {
                    "month": (
                        month_number
                    ),
                    "month_name": (
                        MONTH_LABELS[
                            month_number
                        ]
                    ),
                    "statistic": (
                        statistic_name
                    ),
                    "reference_value": (
                        result[
                            "reference_value"
                        ]
                    ),
                    "raw_low": (
                        result[
                            "raw_low"
                        ]
                    ),
                    "raw_high": (
                        result[
                            "raw_high"
                        ]
                    ),
                    "raw_passes": (
                        result[
                            "raw_passes"
                        ]
                    ),
                    "bc_low": (
                        result[
                            "bc_low"
                        ]
                    ),
                    "bc_high": (
                        result[
                            "bc_high"
                        ]
                    ),
                    "bc_passes": (
                        result[
                            "bc_passes"
                        ]
                    ),
                    "raw_failed_any_test": (
                        samples[
                            "raw_failed_any_test"
                        ]
                    ),
                    "bias_correction_applied": (
                        samples[
                            "correction_applied"
                        ]
                    ),
                    "raw_sample_size": (
                        result[
                            "raw_sample_size"
                        ]
                    ),
                    "bc_sample_size": (
                        result[
                            "bc_sample_size"
                        ]
                    ),
                    "reference_sample_size": (
                        result[
                            "reference_sample_size"
                        ]
                    ),
                }
            )

    fidelity_counts = pd.DataFrame.from_dict(
        counts,
        orient="index",
    )

    fidelity_counts = fidelity_counts.loc[
        [
            "raw",
            "bias corrected",
        ],
        list(
            STATISTICS
        ),
    ]

    monthly_results = pd.DataFrame(
        monthly_rows
    )

    return (
        fidelity_counts,
        monthly_results,
    )


# =============================================================================
# Reporting
# =============================================================================

def print_fidelity_counts(
    fidelity_counts: pd.DataFrame,
) -> None:
    """Print the heatmap values."""

    print()
    print(
        "Fidelity counts (months passed out of 12)"
    )
    print(
        "----------------------------------------"
    )
    print(
        fidelity_counts
    )


def print_monthly_results(
    monthly_results: pd.DataFrame,
) -> None:
    """Print detailed pass/fail results for raw and bias-corrected data."""

    for statistic_name in STATISTICS:

        selected = monthly_results.loc[
            monthly_results[
                "statistic"
            ]
            == statistic_name
        ]

        print()
        print(
            f"Monthly fidelity results: {statistic_name}"
        )
        print(
            "-" * 116
        )
        print(
            f"{'Month':<12}"
            f"{'Reference':>12}"
            f"{'Raw low':>12}"
            f"{'Raw high':>12}"
            f"{'Raw':>8}"
            f"{'BC low':>12}"
            f"{'BC high':>12}"
            f"{'BC':>8}"
            f"{'Applied':>10}"
        )
        print(
            "-" * 116
        )

        for _, row in selected.iterrows():

            raw_status = (
                "PASS"
                if row[
                    "raw_passes"
                ]
                else "FAIL"
            )

            bc_status = (
                "PASS"
                if row[
                    "bc_passes"
                ]
                else "FAIL"
            )

            print(
                f"{row['month_name']:<12}"
                f"{row['reference_value']:>12.4f}"
                f"{row['raw_low']:>12.4f}"
                f"{row['raw_high']:>12.4f}"
                f"{raw_status:>8}"
                f"{row['bc_low']:>12.4f}"
                f"{row['bc_high']:>12.4f}"
                f"{bc_status:>8}"
                f"{str(bool(row['bias_correction_applied'])):>10}"
            )


# =============================================================================
# Heatmap
# =============================================================================

def make_fidelity_heatmap(
    fidelity_counts: pd.DataFrame,
    filename: Path | None = None,
) -> None:
    """
    Plot the raw and bias-corrected fidelity counts.

    The visual design follows the original raw-only heatmap.
    """

    figure, axis = plt.subplots(
        figsize=(
            8,
            3.0,
        )
    )

    axis.imshow(
        12
        - fidelity_counts.values,
        cmap=colormaps[
            "Blues"
        ],
        vmin=0,
        vmax=6,
        aspect="auto",
    )

    for (
        row,
        column,
    ), value in np.ndenumerate(
        fidelity_counts.values
    ):

        axis.text(
            column,
            row,
            int(
                value
            ),
            ha="center",
            va="center",
        )

    axis.set_xticks(
        range(
            len(
                fidelity_counts.columns
            )
        ),
        [
            HEATMAP_COLUMN_LABELS[
                column
            ]
            for column in fidelity_counts.columns
        ],
    )

    axis.set_yticks(
        range(
            len(
                fidelity_counts.index
            )
        ),
        fidelity_counts.index,
    )

    axis.set_title(
        "Fidelity counts (months out of 12)"
    )

    figure.tight_layout()

    if filename is not None:

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            filename,
            bbox_inches="tight",
            dpi=400,
        )

        print(
            "Wrote:",
            filename,
        )

    if show_figure:

        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    raw_model_filename = (
        build_raw_model_filename()
    )

    bias_corrected_model_filename = (
        build_bias_corrected_model_filename()
    )

    (
        reference_filename,
        reference_variable,
        reference_label,
    ) = get_reference_configuration()

    (
        raw_variable,
        bias_corrected_variable,
    ) = get_model_variable_names()

    print(
        "Input files"
    )
    print(
        "-----------"
    )
    print(
        "Raw model:          ",
        raw_model_filename,
    )
    print(
        "Bias-corrected model:",
        bias_corrected_model_filename,
    )
    print(
        f"{reference_label}:".ljust(
            22
        ),
        reference_filename,
    )

    print()
    print(
        "Model variables"
    )
    print(
        "---------------"
    )
    print(
        "Raw:           ",
        raw_variable,
    )
    print(
        "Bias corrected:",
        bias_corrected_variable,
    )

    print()
    print(
        "Analysis settings"
    )
    print(
        "-----------------"
    )
    print(
        "Reference dataset:",
        REFERENCE_DATASET,
    )
    print(
        "Accumulation days:",
        x_days,
    )
    print(
        "Bootstrap samples:",
        f"{number_of_bootstrap_samples:,}",
    )
    print(
        "Confidence level:",
        f"{confidence_level_percent:.1f}%",
    )
    print(
        "Bias correct only failed months:",
        BIAS_CORRECT_ONLY_FAILED_MONTHS,
    )

    with (
        xr.open_dataset(
            raw_model_filename,
            decode_timedelta=False,
        ) as raw_model_ds,
        xr.open_dataset(
            bias_corrected_model_filename,
            decode_timedelta=False,
        ) as bias_corrected_model_ds,
        xr.open_dataset(
            reference_filename
        ) as reference_ds,
    ):

        check_model_dataset(
            ds=raw_model_ds,
            variable_name=raw_variable,
            dataset_label="Raw model dataset",
        )

        check_model_dataset(
            ds=bias_corrected_model_ds,
            variable_name=bias_corrected_variable,
            dataset_label="Bias-corrected model dataset",
        )

        (
            fidelity_counts,
            monthly_results,
        ) = calculate_all_months_fidelity(
            raw_model_ds=raw_model_ds,
            bias_corrected_model_ds=(
                bias_corrected_model_ds
            ),
            reference_ds=reference_ds,
            raw_variable=raw_variable,
            bias_corrected_variable=(
                bias_corrected_variable
            ),
            reference_variable=reference_variable,
        )

    print_fidelity_counts(
        fidelity_counts
    )

    print_monthly_results(
        monthly_results
    )

    make_fidelity_heatmap(
        fidelity_counts=fidelity_counts,
        filename=(
            filename_heatmap
            if write2file
            else None
        ),
    )
