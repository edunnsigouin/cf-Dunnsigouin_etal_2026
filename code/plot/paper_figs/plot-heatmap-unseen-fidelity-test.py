#!/usr/bin/env python3
"""
Create a raw-versus-multiple-bias-corrections fidelity heatmap for all 12 months.

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

The bias-corrected compact model sample file is expected to contain the same
sample variable names as the raw file:

    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)
    month(i_date)

The filename identifies the monthly-mean bias correction and reference dataset.

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
        mm
        q
        doy
        ld
        q_doy

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
evaluated for each month. For every correction method, a month that fails at
least one raw test uses that method's corrected sample. A month that passes all
four raw tests keeps the original raw sample in every corrected row.

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
REFERENCE_DATASET = "senorge"

# Bias-correction methods to compare.
#
# "mm" uses the monthly-mean corrected compact sample file:
#     ..._bc_mm_<reference>.nc
#
# The other methods use:
#     ..._bc_<method>_<reference>.nc
BIAS_CORRECTION_METHODS = [
    "mm",
    "q",
    "doy",
    "ld",
    "q_doy",
]

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
BIAS_CORRECT_ONLY_FAILED_MONTHS = False

number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0
random_seed = 42

# Optional explicit input filenames.
#
# Leave as None to construct filenames automatically.
raw_model_filename_override = None

# Optional per-method filename overrides. Leave values as None to construct
# filenames automatically.
bias_corrected_model_filename_overrides = {
    "mm": None,
    "q": None,
    "doy": None,
    "ld": None,
    "q_doy": None,
}

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
        f"fidelity_heatmap_raw_all_bc_"
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


def build_bias_corrected_model_filename(
    method: str,
) -> Path:
    """Build one bias-corrected compact model sample filename."""

    override = bias_corrected_model_filename_overrides.get(
        method
    )

    if override is not None:
        return Path(
            override
        )

    raw_filename = build_raw_model_filename()

    if method == "mm":
        suffix = (
            f"bc_mm_{REFERENCE_DATASET}"
        )
    else:
        suffix = (
            f"bc_{method}_{REFERENCE_DATASET}"
        )

    return raw_filename.with_name(
        (
            f"{raw_filename.stem}_"
            f"{suffix}"
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


def get_model_variable_name() -> str:
    """Return the complete-window variable used by every model file."""

    return "tp24_max"


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

    valid_methods = {
        "mm",
        "q",
        "doy",
        "ld",
        "q_doy",
    }

    invalid_methods = (
        set(
            BIAS_CORRECTION_METHODS
        )
        - valid_methods
    )

    if invalid_methods:
        raise ValueError(
            f"Unsupported bias-correction methods: "
            f"{sorted(invalid_methods)}."
        )

    if len(
        set(
            BIAS_CORRECTION_METHODS
        )
    ) != len(
        BIAS_CORRECTION_METHODS
    ):
        raise ValueError(
            "BIAS_CORRECTION_METHODS contains duplicate entries."
        )

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
        reference_label: reference_filename,
    }

    for method in BIAS_CORRECTION_METHODS:

        files[
            f"Bias-corrected model ({method})"
        ] = build_bias_corrected_model_filename(
            method
        )

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
    corrected_model_datasets: dict[str, xr.Dataset],
    reference_ds: xr.Dataset,
    model_variable: str,
    reference_variable: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the four fidelity tests for raw data and every correction method.

    When selective correction is enabled, corrected values are used only for
    months that fail at least one raw fidelity test.
    """

    row_names = [
        "raw",
        *BIAS_CORRECTION_METHODS,
    ]

    counts = {
        row_name: {
            statistic_name: 0
            for statistic_name in STATISTICS
        }
        for row_name in row_names
    }

    monthly_rows = []

    for month_number in range(
        1,
        13,
    ):

        raw_values = get_model_values_for_month(
            ds=raw_model_ds,
            variable_name=model_variable,
            month_number=month_number,
        )

        corrected_values_by_method = {
            method: get_model_values_for_month(
                ds=corrected_model_datasets[
                    method
                ],
                variable_name=model_variable,
                month_number=month_number,
            )
            for method in BIAS_CORRECTION_METHODS
        }

        reference_values = get_reference_values_for_month(
            ds=reference_ds,
            variable_name=reference_variable,
            month_number=month_number,
        )

        for method, corrected_values in (
            corrected_values_by_method.items()
        ):

            validate_month_samples(
                raw_values=raw_values,
                bias_corrected_values=corrected_values,
                reference_values=reference_values,
                month_number=month_number,
            )

        raw_rng = np.random.default_rng(
            random_seed
            + month_number
        )

        raw_results = perform_month_fidelity_tests(
            raw_values=raw_values,
            comparison_values=raw_values,
            reference_values=reference_values,
            rng=raw_rng,
        )

        raw_failed_any_test = any(
            not raw_results[
                statistic_name
            ][
                "raw_passes"
            ]
            for statistic_name in STATISTICS
        )

        for statistic_name in STATISTICS:

            result = raw_results[
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

            monthly_rows.append(
                {
                    "dataset": "raw",
                    "month": month_number,
                    "month_name": MONTH_LABELS[
                        month_number
                    ],
                    "statistic": statistic_name,
                    "reference_value": result[
                        "reference_value"
                    ],
                    "low": result[
                        "raw_low"
                    ],
                    "high": result[
                        "raw_high"
                    ],
                    "passes": result[
                        "raw_passes"
                    ],
                    "raw_failed_any_test": (
                        raw_failed_any_test
                    ),
                    "bias_correction_applied": False,
                    "model_sample_size": result[
                        "raw_sample_size"
                    ],
                    "reference_sample_size": result[
                        "reference_sample_size"
                    ],
                }
            )

        for method in BIAS_CORRECTION_METHODS:

            if (
                BIAS_CORRECT_ONLY_FAILED_MONTHS
                and not raw_failed_any_test
            ):
                comparison_values = raw_values
                correction_applied = False
            else:
                comparison_values = (
                    corrected_values_by_method[
                        method
                    ]
                )
                correction_applied = True

            # Reuse identical bootstrap indices across raw and all methods.
            method_rng = np.random.default_rng(
                random_seed
                + month_number
            )

            method_results = perform_month_fidelity_tests(
                raw_values=raw_values,
                comparison_values=comparison_values,
                reference_values=reference_values,
                rng=method_rng,
            )

            for statistic_name in STATISTICS:

                result = method_results[
                    statistic_name
                ]

                counts[
                    method
                ][
                    statistic_name
                ] += int(
                    result[
                        "bc_passes"
                    ]
                )

                monthly_rows.append(
                    {
                        "dataset": method,
                        "month": month_number,
                        "month_name": MONTH_LABELS[
                            month_number
                        ],
                        "statistic": statistic_name,
                        "reference_value": result[
                            "reference_value"
                        ],
                        "low": result[
                            "bc_low"
                        ],
                        "high": result[
                            "bc_high"
                        ],
                        "passes": result[
                            "bc_passes"
                        ],
                        "raw_failed_any_test": (
                            raw_failed_any_test
                        ),
                        "bias_correction_applied": (
                            correction_applied
                        ),
                        "model_sample_size": result[
                            "bc_sample_size"
                        ],
                        "reference_sample_size": result[
                            "reference_sample_size"
                        ],
                    }
                )

    fidelity_counts = pd.DataFrame.from_dict(
        counts,
        orient="index",
    ).loc[
        row_names,
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
    """Print detailed monthly results for every dataset row."""

    for dataset_name in [
        "raw",
        *BIAS_CORRECTION_METHODS,
    ]:

        selected_dataset = monthly_results.loc[
            monthly_results[
                "dataset"
            ]
            == dataset_name
        ]

        print()
        print(
            f"Monthly fidelity results: {dataset_name}"
        )
        print(
            "-" * 92
        )
        print(
            f"{'Month':<12}"
            f"{'Statistic':<12}"
            f"{'Reference':>12}"
            f"{'Low':>12}"
            f"{'High':>12}"
            f"{'Result':>10}"
            f"{'Applied':>10}"
        )
        print(
            "-" * 92
        )

        for _, row in selected_dataset.iterrows():

            status = (
                "PASS"
                if row[
                    "passes"
                ]
                else "FAIL"
            )

            print(
                f"{row['month_name']:<12}"
                f"{row['statistic']:<12}"
                f"{row['reference_value']:>12.4f}"
                f"{row['low']:>12.4f}"
                f"{row['high']:>12.4f}"
                f"{status:>10}"
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
            1.0
            + 0.65
            * len(
                fidelity_counts.index
            ),
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

    raw_model_filename = build_raw_model_filename()

    corrected_model_filenames = {
        method: build_bias_corrected_model_filename(
            method
        )
        for method in BIAS_CORRECTION_METHODS
    }

    (
        reference_filename,
        reference_variable,
        reference_label,
    ) = get_reference_configuration()

    model_variable = get_model_variable_name()

    print(
        "Input files"
    )
    print(
        "-----------"
    )
    print(
        "Raw model:",
        raw_model_filename,
    )

    for method in BIAS_CORRECTION_METHODS:

        print(
            f"{method:>5}:",
            corrected_model_filenames[
                method
            ],
        )

    print(
        f"{reference_label}:",
        reference_filename,
    )

    print()
    print(
        "Model variable:",
        model_variable,
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
        "Bias-correction methods:",
        BIAS_CORRECTION_METHODS,
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

    raw_model_ds = xr.open_dataset(
        raw_model_filename,
        decode_timedelta=False,
    )

    corrected_model_datasets = {
        method: xr.open_dataset(
            filename,
            decode_timedelta=False,
        )
        for method, filename in (
            corrected_model_filenames.items()
        )
    }

    reference_ds = xr.open_dataset(
        reference_filename
    )

    try:

        check_model_dataset(
            ds=raw_model_ds,
            variable_name=model_variable,
            dataset_label="Raw model dataset",
        )

        for method, dataset in (
            corrected_model_datasets.items()
        ):

            check_model_dataset(
                ds=dataset,
                variable_name=model_variable,
                dataset_label=(
                    f"Bias-corrected model dataset "
                    f"({method})"
                ),
            )

        (
            fidelity_counts,
            monthly_results,
        ) = calculate_all_months_fidelity(
            raw_model_ds=raw_model_ds,
            corrected_model_datasets=(
                corrected_model_datasets
            ),
            reference_ds=reference_ds,
            model_variable=model_variable,
            reference_variable=reference_variable,
        )

    finally:

        raw_model_ds.close()

        for dataset in (
            corrected_model_datasets.values()
        ):
            dataset.close()

        reference_ds.close()

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
