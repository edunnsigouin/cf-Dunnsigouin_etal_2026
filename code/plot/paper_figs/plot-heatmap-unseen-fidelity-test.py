#!/usr/bin/env python3
"""
Create an all-month UNSEEN summary heatmap for raw and bias-corrected S2S data.

Rows are raw, mm, q, doy, ld, and q_doy. Columns are independence, four
fidelity tests, and stability. Each cell is the number of calendar months
(out of 12) that pass the corresponding test.

Test reasoning
--------------
Independence:
For each month, pairwise Spearman rank correlations are calculated between
ensemble members across initialization dates. Forecast and hindcast
correlations are calculated separately and then pooled. The monthly summary is
the median pairwise correlation. Independence passes when

    abs(median Spearman correlation) < INDEPENDENCE_CORRELATION_THRESHOLD

The default threshold is 0.10 and is a user setting. This is intentionally a
simple effect-size criterion: a median correlation sufficiently close to zero
is treated as weak ensemble-member dependence.

Fidelity:
For each month, the complete all-lead model sample is bootstrapped using the
same sample size as the selected reference dataset. Mean, sample standard
deviation, skewness, and excess kurtosis are calculated. A fidelity test passes
when the reference statistic falls inside the selected central model bootstrap
confidence interval.

Stability:
For each month, the Early and Late lead-location samples are compared with a
two-sample Kolmogorov-Smirnov (KS) test. These are the same full-window maxima
classified by the lead at which the maximum occurred; maxima are not
recalculated over shorter windows. The null hypothesis is that Early and Late
come from the same continuous distribution. Stability passes when this null is
not rejected. At the default 95% level, a month therefore fails when p < 0.05.
The confidence level is controlled by STABILITY_CONFIDENCE_LEVEL_PERCENT.

Bias-correction handling
------------------------
When BIAS_CORRECT_ONLY_FAILED_MONTHS is False, every corrected row uses its
bias-corrected sample for all 12 months and for all six tests.

When BIAS_CORRECT_ONLY_FAILED_MONTHS is True, the script first evaluates ALL
six tests on the raw sample for each month:

    independence
    fidelity: mean
    fidelity: std
    fidelity: skewness
    fidelity: kurtosis
    stability

If the raw month passes all six tests, that month stays raw in every corrected
row. If the raw month fails at least one test, the selected bias-corrected
sample is used for that entire month and all six tests in that corrected row.

This keeps the selective-correction decision month-based and consistent across
independence, fidelity, and stability.

Expected compact model variables
--------------------------------
    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)
    month(i_date)
    model_type(i_date)
    number(number)
    i_date(i_date)
"""

import os
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ks_2samp, kurtosis, rankdata, skew

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
#     First run ALL SIX tests on the raw sample for each month:
#       independence, four fidelity tests, and stability.
#     If a month fails at least one raw test, use the bias-corrected sample
#     for that entire month and all six tests. If it passes all six raw tests,
#     retain the raw sample for that month in every corrected row.
#
# The resulting heatmap rows are still labelled by bias-correction method.
BIAS_CORRECT_ONLY_FAILED_MONTHS = True

# Independence settings.
# A month passes when abs(median Spearman correlation) is below this threshold.
INDEPENDENCE_CORRELATION_THRESHOLD = 0.10

# Minimum paired initialization values required for one Spearman correlation.
minimum_independence_samples = 10

# Fidelity bootstrap settings.
number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0
random_seed = 42

# Stability KS-test settings.
STABILITY_CONFIDENCE_LEVEL_PERCENT = 95.0
ks_alternative = "two-sided"
ks_method = "auto"

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

write2file = True
show_figure = True

path_out = Path(
    config.dirs[
        "fig"
    ]
)

correction_mode = (
    "failedmonths"
    if BIAS_CORRECT_ONLY_FAILED_MONTHS
    else "allmonths"
)

HEATMAP_NUMBER_COLOR = "tab:red"

filename_heatmap = (
    path_out
    / (
        f"Heatmap_UNSEEN_test_summary_"
        f"{REFERENCE_DATASET}_"
        f"{correction_mode}_"
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
    "independence": "independence",
    "mean": "fidelity: mean",
    "std": "fidelity: std",
    "skewness": "fidelity: skewness",
    "kurtosis": "fidelity: kurtosis",
    "stability": "stability",
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

    if not 0.0 < INDEPENDENCE_CORRELATION_THRESHOLD <= 1.0:
        raise ValueError(
            "INDEPENDENCE_CORRELATION_THRESHOLD must be in (0, 1]."
        )

    if minimum_independence_samples < 3:
        raise ValueError(
            "minimum_independence_samples must be at least 3."
        )

    if not 0.0 < STABILITY_CONFIDENCE_LEVEL_PERCENT < 100.0:
        raise ValueError(
            "STABILITY_CONFIDENCE_LEVEL_PERCENT must be between 0 and 100."
        )

    if ks_alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(
            "ks_alternative must be 'two-sided', 'less', or 'greater'."
        )

    if ks_method not in {"auto", "exact", "asymp"}:
        raise ValueError(
            "ks_method must be 'auto', 'exact', or 'asymp'."
        )

    if number_of_lead_bins != 2:
        raise ValueError(
            "The stability test requires exactly two lead bins."
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


def check_independence_stability_structure(
    ds: xr.Dataset,
    dataset_label: str,
) -> None:
    """Check variables needed by independence and stability."""

    early_variable, late_variable = get_stability_variable_names()

    required = {
        "tp24_max",
        early_variable,
        late_variable,
        MODEL_MONTH_VARIABLE,
        "model_type",
        "number",
        "i_date",
    }

    missing = required - set(ds.variables)

    if missing:
        raise KeyError(
            f"{dataset_label} is missing independence/stability variables: "
            f"{sorted(missing)}"
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
# Independence and stability
# =============================================================================

def normalize_model_type(values: np.ndarray) -> np.ndarray:
    """Return model-type labels as stripped lowercase strings."""
    return np.array(
        [
            (value.decode("utf-8") if isinstance(value, bytes) else str(value))
            .strip()
            .lower()
            for value in np.asarray(values).ravel()
        ],
        dtype=object,
    )


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate one pairwise Spearman rank correlation."""
    valid = np.isfinite(x) & np.isfinite(y)

    if int(valid.sum()) < minimum_independence_samples:
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return np.nan

    x_rank = rankdata(x_valid, method="average")
    y_rank = rankdata(y_valid, method="average")

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def get_month_member_matrix(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
    model_type: str,
) -> np.ndarray:
    """Return an initialization-by-member matrix for one month/model type."""

    if model_type not in {"forecast", "hindcast"}:
        raise ValueError("model_type must be 'forecast' or 'hindcast'.")

    required = {
        variable_name,
        MODEL_MONTH_VARIABLE,
        "model_type",
        "number",
        "i_date",
    }
    missing = required - set(ds.variables)

    if missing:
        raise KeyError(
            "Model dataset is missing independence variables: "
            f"{sorted(missing)}"
        )

    types = normalize_model_type(ds["model_type"].values)

    selected_rows = (
        (ds[MODEL_MONTH_VARIABLE].values == month_number)
        & (types == model_type)
    )

    return (
        ds[variable_name]
        .isel(i_date=selected_rows)
        .transpose("i_date", "number")
        .values
        .astype("float64")
    )


def calculate_pairwise_correlations_from_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """Return all finite member-pair Spearman correlations."""

    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return np.array([], dtype="float64")

    correlations = np.array(
        [
            spearman_correlation(
                matrix[:, member_1],
                matrix[:, member_2],
            )
            for member_1, member_2 in combinations(
                range(matrix.shape[1]),
                2,
            )
        ],
        dtype="float64",
    )

    return remove_missing_values(correlations)


def calculate_month_independence(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
) -> dict[str, object]:
    """
    Calculate panel-(a)-style independence for one month.

    Forecast and hindcast correlations are calculated separately, then pooled.
    """

    forecast = calculate_pairwise_correlations_from_matrix(
        get_month_member_matrix(
            ds,
            variable_name,
            month_number,
            "forecast",
        )
    )

    hindcast = calculate_pairwise_correlations_from_matrix(
        get_month_member_matrix(
            ds,
            variable_name,
            month_number,
            "hindcast",
        )
    )

    correlations = np.concatenate([forecast, hindcast])

    if correlations.size == 0:
        raise ValueError(
            "No finite pairwise Spearman correlations for "
            f"{MONTH_LABELS[month_number]}."
        )

    median_correlation = float(np.median(correlations))

    return {
        "median_correlation": median_correlation,
        "passes": bool(
            abs(median_correlation)
            < INDEPENDENCE_CORRELATION_THRESHOLD
        ),
        "number_of_pairwise_correlations": int(correlations.size),
    }


def get_stability_variable_names() -> tuple[str, str]:
    """Return Early and Late compact-sample variable names."""
    _, split_ranges = get_lead_ranges()

    if len(split_ranges) != 2:
        raise ValueError(
            "The stability test requires exactly two lead bins."
        )

    early_range, late_range = split_ranges

    return (
        f"tp24_max_lead{early_range[0]}_{early_range[1]}",
        f"tp24_max_lead{late_range[0]}_{late_range[1]}",
    )


def get_stability_p_value_threshold() -> float:
    """Convert stability confidence level to the KS p-value threshold."""
    return 1.0 - STABILITY_CONFIDENCE_LEVEL_PERCENT / 100.0


def calculate_month_stability(
    ds: xr.Dataset,
    month_number: int,
) -> dict[str, object]:
    """Perform the panel-(f)-style Early/Late two-sample KS test."""

    early_variable, late_variable = get_stability_variable_names()

    for variable_name in (early_variable, late_variable):
        if variable_name not in ds:
            raise KeyError(
                f"Stability variable '{variable_name}' was not found."
            )

    early_values = get_model_values_for_month(
        ds,
        early_variable,
        month_number,
    )
    late_values = get_model_values_for_month(
        ds,
        late_variable,
        month_number,
    )

    if early_values.size == 0 or late_values.size == 0:
        raise ValueError(
            "Early or Late stability sample is empty for "
            f"{MONTH_LABELS[month_number]}."
        )

    result = ks_2samp(
        early_values,
        late_values,
        alternative=ks_alternative,
        method=ks_method,
    )

    p_value = float(result.pvalue)
    passes = bool(p_value >= get_stability_p_value_threshold())

    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "passes": passes,
        "early_sample_size": int(early_values.size),
        "late_sample_size": int(late_values.size),
    }


def calculate_all_months_independence_and_stability(
    raw_model_ds: xr.Dataset,
    corrected_model_datasets: dict[str, xr.Dataset],
    model_variable: str,
    month_correction_lookup: dict[int, bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Count monthly independence and stability passes for all rows.

    When selective correction is enabled, corrected rows use raw data for a
    month that passed all six raw tests, and corrected data otherwise.
    """

    dataset_names = [
        "raw",
        *BIAS_CORRECTION_METHODS,
    ]

    counts = {
        dataset_name: {
            "independence": 0,
            "stability": 0,
        }
        for dataset_name in dataset_names
    }

    monthly_rows = []

    for dataset_name in dataset_names:
        for month_number in range(1, 13):

            correction_applied = (
                dataset_name != "raw"
                and (
                    not BIAS_CORRECT_ONLY_FAILED_MONTHS
                    or month_correction_lookup[
                        month_number
                    ]
                )
            )

            if dataset_name == "raw" or not correction_applied:
                dataset = raw_model_ds
            else:
                dataset = corrected_model_datasets[dataset_name]

            independence = calculate_month_independence(
                dataset,
                model_variable,
                month_number,
            )
            stability = calculate_month_stability(
                dataset,
                month_number,
            )

            counts[dataset_name]["independence"] += int(
                independence["passes"]
            )
            counts[dataset_name]["stability"] += int(
                stability["passes"]
            )

            monthly_rows.append(
                {
                    "dataset": dataset_name,
                    "month": month_number,
                    "month_name": MONTH_LABELS[month_number],
                    "median_correlation": independence[
                        "median_correlation"
                    ],
                    "independence_passes": independence["passes"],
                    "number_of_pairwise_correlations": independence[
                        "number_of_pairwise_correlations"
                    ],
                    "ks_statistic": stability["statistic"],
                    "ks_p_value": stability["p_value"],
                    "stability_passes": stability["passes"],
                    "early_sample_size": stability["early_sample_size"],
                    "late_sample_size": stability["late_sample_size"],
                    "bias_correction_applied": correction_applied,
                    "raw_failed_any_test": month_correction_lookup[
                        month_number
                    ],
                }
            )

    row_names = ["raw", *BIAS_CORRECTION_METHODS]

    count_frame = pd.DataFrame.from_dict(
        counts,
        orient="index",
    ).loc[row_names, ["independence", "stability"]]

    return count_frame, pd.DataFrame(monthly_rows)


def combine_test_counts(
    independence_stability_counts: pd.DataFrame,
    fidelity_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all six requested heatmap columns."""

    return pd.concat(
        [
            independence_stability_counts[["independence"]],
            fidelity_counts[list(STATISTICS)],
            independence_stability_counts[["stability"]],
        ],
        axis=1,
    )



def calculate_raw_month_gate(
    raw_model_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    model_variable: str,
    reference_variable: str,
) -> pd.DataFrame:
    """
    Evaluate all six tests on raw data and decide which months need correction.

    A month requires bias correction when ANY raw test fails:
        independence,
        mean fidelity,
        standard-deviation fidelity,
        skewness fidelity,
        kurtosis fidelity,
        or stability.
    """

    rows = []

    for month_number in range(
        1,
        13,
    ):
        raw_values = get_model_values_for_month(
            ds=raw_model_ds,
            variable_name=model_variable,
            month_number=month_number,
        )

        reference_values = get_reference_values_for_month(
            ds=reference_ds,
            variable_name=reference_variable,
            month_number=month_number,
        )

        validate_month_samples(
            raw_values=raw_values,
            bias_corrected_values=raw_values,
            reference_values=reference_values,
            month_number=month_number,
        )

        raw_rng = np.random.default_rng(
            random_seed
            + month_number
        )

        raw_fidelity = perform_month_fidelity_tests(
            raw_values=raw_values,
            comparison_values=raw_values,
            reference_values=reference_values,
            rng=raw_rng,
        )

        independence = calculate_month_independence(
            ds=raw_model_ds,
            variable_name=model_variable,
            month_number=month_number,
        )

        stability = calculate_month_stability(
            ds=raw_model_ds,
            month_number=month_number,
        )

        raw_test_passes = {
            "independence": bool(
                independence["passes"]
            ),
            **{
                statistic_name: bool(
                    raw_fidelity[
                        statistic_name
                    ][
                        "raw_passes"
                    ]
                )
                for statistic_name in STATISTICS
            },
            "stability": bool(
                stability["passes"]
            ),
        }

        failed_tests = [
            test_name
            for test_name, passes in raw_test_passes.items()
            if not passes
        ]

        rows.append(
            {
                "month": month_number,
                "month_name": MONTH_LABELS[month_number],
                **{
                    f"raw_{test_name}_passes": passes
                    for test_name, passes in raw_test_passes.items()
                },
                "raw_failed_any_test": bool(failed_tests),
                "raw_failed_tests": ",".join(failed_tests),
            }
        )

    return pd.DataFrame(rows)


def build_month_correction_lookup(
    raw_month_gate: pd.DataFrame,
) -> dict[int, bool]:
    """Return month -> whether the corrected sample should be used."""

    return {
        int(row["month"]): bool(row["raw_failed_any_test"])
        for _, row in raw_month_gate.iterrows()
    }


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
    month_correction_lookup: dict[int, bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the four fidelity tests for raw data and every correction method.

    When selective correction is enabled, corrected values are used only for
    months that fail at least one of the six raw tests.
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
                        month_correction_lookup[month_number]
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
                and not month_correction_lookup[month_number]
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
                            month_correction_lookup[month_number]
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



def print_raw_month_gate(
    raw_month_gate: pd.DataFrame,
) -> None:
    """Print which raw months trigger selective bias correction."""

    print()
    print("Raw all-test screening")
    print("----------------------")
    print(
        f"{'Month':<12}"
        f"{'Needs BC':>10}"
        f"{'Failed tests':>42}"
    )
    print("-" * 64)

    for _, row in raw_month_gate.iterrows():
        failed_tests = (
            row["raw_failed_tests"]
            if row["raw_failed_tests"]
            else "-"
        )

        print(
            f"{row['month_name']:<12}"
            f"{str(bool(row['raw_failed_any_test'])):>10}"
            f"{failed_tests:>42}"
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


def print_independence_stability_results(
    counts: pd.DataFrame,
    monthly_results: pd.DataFrame,
) -> None:
    """Print counts and detailed monthly independence/stability results."""

    print()
    print("Independence and stability counts (months passed out of 12)")
    print("------------------------------------------------------------")
    print(counts)

    for dataset_name in ["raw", *BIAS_CORRECTION_METHODS]:
        selected = monthly_results.loc[
            monthly_results["dataset"] == dataset_name
        ]

        print()
        print(
            f"Monthly independence/stability results: {dataset_name}"
        )
        print("-" * 92)
        print(
            f"{'Month':<12}"
            f"{'Median rho':>12}"
            f"{'Indep.':>10}"
            f"{'KS D':>12}"
            f"{'KS p':>12}"
            f"{'Stability':>12}"
        )
        print("-" * 92)

        for _, row in selected.iterrows():
            independence_status = (
                "PASS" if row["independence_passes"] else "FAIL"
            )
            stability_status = (
                "PASS" if row["stability_passes"] else "FAIL"
            )

            print(
                f"{row['month_name']:<12}"
                f"{row['median_correlation']:>12.4f}"
                f"{independence_status:>10}"
                f"{row['ks_statistic']:>12.4f}"
                f"{row['ks_p_value']:>12.4g}"
                f"{stability_status:>12}"
            )


# =============================================================================
# Heatmap
# =============================================================================

def make_summary_heatmap(
    summary_counts: pd.DataFrame,
    filename: Path | None = None,
) -> None:
    """
    Plot monthly pass counts for independence, fidelity, and stability.
    """

    figure, axis = plt.subplots(
        figsize=(
            10,
            1.0
            + 0.65
            * len(
                summary_counts.index
            ),
        )
    )

    axis.imshow(
        12
        - summary_counts.values,
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
        summary_counts.values
    ):

        axis.text(
            column,
            row,
            int(
                value
            ),
            ha="center",
            va="center",
            color=HEATMAP_NUMBER_COLOR,
        )

    axis.set_xticks(
        range(
            len(
                summary_counts.columns
            )
        ),
        [
            HEATMAP_COLUMN_LABELS[
                column
            ]
            for column in summary_counts.columns
        ],
    )
    
    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    axis.set_yticks(
        range(
            len(
                summary_counts.index
            )
        ),
        summary_counts.index,
    )

    axis.set_title(
        "UNSEEN tests"
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
    print(
        "Independence |median rho| threshold:",
        INDEPENDENCE_CORRELATION_THRESHOLD,
    )
    print(
        "Stability confidence level:",
        f"{STABILITY_CONFIDENCE_LEVEL_PERCENT:.1f}%",
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

        check_independence_stability_structure(
            ds=raw_model_ds,
            dataset_label="Raw model dataset",
        )

        for method, dataset in corrected_model_datasets.items():
            check_independence_stability_structure(
                ds=dataset,
                dataset_label=(
                    f"Bias-corrected model dataset ({method})"
                ),
            )

        raw_month_gate = calculate_raw_month_gate(
            raw_model_ds=raw_model_ds,
            reference_ds=reference_ds,
            model_variable=model_variable,
            reference_variable=reference_variable,
        )

        month_correction_lookup = build_month_correction_lookup(
            raw_month_gate
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
            month_correction_lookup=month_correction_lookup,
        )

        (
            independence_stability_counts,
            independence_stability_monthly_results,
        ) = calculate_all_months_independence_and_stability(
            raw_model_ds=raw_model_ds,
            corrected_model_datasets=corrected_model_datasets,
            model_variable=model_variable,
            month_correction_lookup=month_correction_lookup,
        )

        summary_counts = combine_test_counts(
            independence_stability_counts,
            fidelity_counts,
        )

    finally:

        raw_model_ds.close()

        for dataset in (
            corrected_model_datasets.values()
        ):
            dataset.close()

        reference_ds.close()

    print_raw_month_gate(
        raw_month_gate
    )

    print_fidelity_counts(
        fidelity_counts
    )

    print_monthly_results(
        monthly_results
    )

    print_independence_stability_results(
        independence_stability_counts,
        independence_stability_monthly_results,
    )

    print()
    print("Combined heatmap counts")
    print("-----------------------")
    print(summary_counts)

    make_summary_heatmap(
        summary_counts=summary_counts,
        filename=(
            filename_heatmap
            if write2file
            else None
        ),
    )
