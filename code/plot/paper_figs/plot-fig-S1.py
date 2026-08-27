#!/usr/bin/env python3
"""
Create an all-month UNSEEN summary heatmap using the same compact inputs and tests
as the six-panel monthly diagnostic script.

Each heatmap cell is the number of calendar months, out of 12, that pass a test.
Rows are raw plus the bias-correction methods selected in BIAS_CORRECTION_METHODS.
The first five columns are fixed: independence plus fidelity of the mean,
standard deviation, skewness, and kurtosis.

The final column is selected with PANEL_F_TEST:

    "ks_test"
        Compare the complete model monthly-maximum sample with the selected
        reference sample using a two-sample Kolmogorov-Smirnov (KS) test. A month
        passes when the equal-distribution null hypothesis is not rejected.

    "stability_test"
        Compare the Early and Late lead-location samples using a two-sample KS
        test. These are the same complete-window maxima classified by the ending
        lead on which the maximum occurred; maxima are not recalculated within
        shorter lead windows. A month passes when the equal-distribution null
        hypothesis is not rejected.

The script uses the same compact S2S filenames as the six-panel script:
    monthly_max_samples_<variable>_<N>dayacc_<catchment>_<start>_<end>_raw.nc
    monthly_max_samples_<variable>_<N>dayacc_<catchment>_<start>_<end>_
        bc_<method>_<reference>_<reference-years>.nc

The reference files always cover REFERENCE_FILE_YEARS, while reference_years
selects the subset used in the tests. August 2023 Storm Hans can optionally be
excluded from the reference sample.

If BIAS_CORRECT_ONLY_FAILED_MONTHS is True, each raw month is first screened with
all six heatmap tests. A corrected row uses its bias-corrected sample only for
months that fail at least one raw test; otherwise that month remains raw for all
tests in that row.
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

x_days = 2
catchment = "regine_drammen"
forecast_date_range = ("2020-01-02", "2023-12-28")

reference_years = ("1957", "2025")
REFERENCE_FILE_YEARS = ("1957", "2025")
era5_grid = "0.5x0.5"

first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Options: "era5", "senorge"
REFERENCE_DATASET = "senorge"

# Final heatmap column.
#
# Options:
#     "ks_test"        : complete model vs reference KS test
#     "stability_test" : Early vs Late lead-location KS test
PANEL_F_TEST = "ks_test"

# Heatmap rows = raw + only the corrected methods listed here.
# Any non-empty subset of the supported methods may be selected.
#BIAS_CORRECTION_METHODS = [
#    "mm_1step",
#    "mm_2step",
#    "q",
#    "doy",
#    "ld",
#    "q_doy",
#]

BIAS_CORRECTION_METHODS = [
    "mm_1step",
]

# If True, apply each correction only to months that fail at least one raw test.
BIAS_CORRECT_ONLY_FAILED_MONTHS = True

EXCLUDE_STORM_HANS_FROM_REFERENCE = True

# Independence passes when abs(median pairwise Spearman correlation) is below
# this threshold. The six-panel figure itself shows the correlation distribution;
# this threshold converts it to a monthly pass/fail result for the heatmap.
INDEPENDENCE_CORRELATION_THRESHOLD = 0.10
minimum_independence_samples = 10

number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0
random_seed = 42

ks_alternative = "two-sided"
ks_method = "auto"
ks_significance_level_percent = 95.0

raw_model_filename_override = None
bias_corrected_model_filename_overrides = {
    "mm_1step": None,
    "mm_2step": None,
    "q": None,
    "doy": None,
    "ld": None,
    "q_doy": None,
}
reference_filename_override = None

write2file = False
show_figure = True


# =============================================================================
# Fixed settings
# =============================================================================

MODEL_VARIABLE = "tp24"
MODEL_MONTH_COORDINATE = "sample_month"

ERA5_VARIABLE = "tp24"
SENORGE_VARIABLE = "rr"
SENORGE_LABEL = "SeNorge"

STATISTICS = ("mean", "std", "skewness", "kurtosis")

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

HEATMAP_NUMBER_COLOR = "tab:red"


# =============================================================================
# General helpers
# =============================================================================

def get_file_id(catchment_name: str) -> str:
    """Return the short catchment name used in compact sample filenames."""
    return catchment_name.removeprefix("regine_")


def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """Flatten an array and retain only finite values."""
    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def get_model_calendar_month(model_ds: xr.Dataset) -> xr.DataArray:
    """Return calendar month 1-12 from sample_month(i_date) stored as YYYYMM."""
    if MODEL_MONTH_COORDINATE not in model_ds:
        raise KeyError(
            f"Model dataset is missing '{MODEL_MONTH_COORDINATE}'. "
            f"Available variables: {list(model_ds.variables)}"
        )

    sample_month = model_ds[MODEL_MONTH_COORDINATE]
    if sample_month.dims != ("i_date",):
        raise ValueError(
            f"{MODEL_MONTH_COORDINATE} must have dimension ('i_date',), "
            f"but has {sample_month.dims}."
        )

    values = np.asarray(sample_month.values)
    finite = np.isfinite(values)
    calendar_month = np.full(values.shape, -1, dtype="int16")
    calendar_month[finite] = values[finite].astype("int64") % 100

    if np.any(finite & ~np.isin(calendar_month, np.arange(1, 13))):
        raise ValueError(f"{MODEL_MONTH_COORDINATE} contains invalid YYYYMM values.")

    return xr.DataArray(
        calendar_month,
        dims=("i_date",),
        coords={"i_date": model_ds["i_date"]},
        name="calendar_month",
    )


# =============================================================================
# Lead bins and filenames
# =============================================================================

def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """Split usable accumulated ending leads into near-equal consecutive bins."""
    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder)
        for index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead
    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1
    return bins


def get_stability_lead_ranges() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return Early and Late accumulated ending-lead ranges."""
    first_usable_lead = first_input_lead + x_days - 1
    split_ranges = split_usable_accumulated_leads(
        first_usable_lead, last_input_lead, number_of_lead_bins
    )
    if len(split_ranges) != 2:
        raise ValueError("The stability test requires exactly two lead bins.")
    return split_ranges[0], split_ranges[1]


def get_stability_variable_names() -> tuple[str, str]:
    """Return Early and Late compact model-variable names."""
    early_range, late_range = get_stability_lead_ranges()
    return (
        f"tp24_max_lead{early_range[0]}_{early_range[1]}",
        f"tp24_max_lead{late_range[0]}_{late_range[1]}",
    )


def build_model_filename(method: str) -> Path:
    """Build one compact monthly-sample filename used by the six-panel script."""
    if method == "raw" and raw_model_filename_override is not None:
        return Path(raw_model_filename_override)

    if method != "raw":
        override = bias_corrected_model_filename_overrides.get(method)
        if override is not None:
            return Path(override)

    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}"
    )
    correction_label = (
        "raw"
        if method == "raw"
        else f"bc_{method}_{REFERENCE_DATASET}_{reference_years[0]}-{reference_years[-1]}"
    )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction_label}.nc"


def build_era5_filename() -> Path:
    """Build the fixed-range ERA5 monthly-maximum reference filename."""
    return Path(
        f"{config.dirs['era5_processed']}"
        f"monthly_max_samples_{ERA5_VARIABLE}_{x_days}dayacc_{catchment}_"
        f"{REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}.nc"
    )


def build_senorge_filename() -> Path:
    """Build the fixed-range SeNorge monthly-maximum reference filename."""
    return Path(
        f"{config.dirs['senorge_processed']}"
        f"monthly_max_samples_{SENORGE_VARIABLE}_{x_days}dayacc_{catchment}_"
        f"{REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}.nc"
    )


def get_reference_configuration() -> tuple[Path, str, str]:
    """Return selected reference filename, variable, and display label."""
    if reference_filename_override is not None:
        filename = Path(reference_filename_override)
    else:
        filename = build_era5_filename() if REFERENCE_DATASET == "era5" else build_senorge_filename()

    if REFERENCE_DATASET == "era5":
        return filename, ERA5_VARIABLE, "ERA5"
    return filename, SENORGE_VARIABLE, SENORGE_LABEL


def build_output_filename() -> Path:
    """Return the heatmap output filename."""
    if len(BIAS_CORRECTION_METHODS) == 1:
        return Path(config.dirs["fig"]) / "fig-S1.png"

    correction_mode = (
        "fixed-failed-months"
        if BIAS_CORRECT_ONLY_FAILED_MONTHS
        else "fixed-all-months"
    )
    return Path(config.dirs["fig"]) / (
        f"Heatmap_UNSEEN_tests_with_{PANEL_F_TEST}_{correction_mode}_"
        f"{x_days}dayacc_{catchment}_{forecast_date_range[0]}_"
        f"{forecast_date_range[1]}_{REFERENCE_DATASET}_"
        f"{reference_years[0]}-{reference_years[-1]}.png"
    )



# =============================================================================
# Validation and sample extraction
# =============================================================================

def validate_user_settings() -> None:
    """Validate settings and required files."""
    valid_methods = {"mm_1step", "mm_2step", "q", "doy", "ld", "q_doy"}
    valid_references = {"era5", "senorge"}
    valid_panel_f_tests = {"ks_test", "stability_test"}

    if not BIAS_CORRECTION_METHODS:
        raise ValueError("BIAS_CORRECTION_METHODS must contain at least one method.")

    invalid_methods = set(BIAS_CORRECTION_METHODS) - valid_methods
    if invalid_methods:
        raise ValueError(
            "BIAS_CORRECTION_METHODS contains invalid method(s): "
            f"{sorted(invalid_methods)}. Valid methods are {sorted(valid_methods)}."
        )

    if len(BIAS_CORRECTION_METHODS) != len(set(BIAS_CORRECTION_METHODS)):
        raise ValueError("BIAS_CORRECTION_METHODS must not contain duplicate methods.")
    if REFERENCE_DATASET not in valid_references:
        raise ValueError(f"REFERENCE_DATASET must be one of {sorted(valid_references)}.")
    if PANEL_F_TEST not in valid_panel_f_tests:
        raise ValueError(f"PANEL_F_TEST must be one of {sorted(valid_panel_f_tests)}.")
    if x_days < 1:
        raise ValueError("x_days must be at least 1.")
    if first_input_lead > last_input_lead:
        raise ValueError("first_input_lead must not exceed last_input_lead.")
    if first_input_lead + x_days - 1 > last_input_lead:
        raise ValueError("x_days is too large for the available lead window.")
    if number_of_lead_bins != 2:
        raise ValueError("The stability test requires exactly two lead bins.")
    if minimum_independence_samples < 3:
        raise ValueError("minimum_independence_samples must be at least 3.")
    if not 0.0 < INDEPENDENCE_CORRELATION_THRESHOLD <= 1.0:
        raise ValueError("INDEPENDENCE_CORRELATION_THRESHOLD must be in (0, 1].")
    if number_of_bootstrap_samples < 1:
        raise ValueError("number_of_bootstrap_samples must be at least 1.")
    if not 0.0 < confidence_level_percent < 100.0:
        raise ValueError("confidence_level_percent must be between 0 and 100.")
    if not 0.0 < ks_significance_level_percent < 100.0:
        raise ValueError("ks_significance_level_percent must be between 0 and 100.")
    if ks_alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("ks_alternative must be 'two-sided', 'less', or 'greater'.")
    if ks_method not in {"auto", "exact", "asymp"}:
        raise ValueError("ks_method must be 'auto', 'exact', or 'asymp'.")

    first_reference_year, last_reference_year = map(int, reference_years)
    file_start, file_end = map(int, REFERENCE_FILE_YEARS)
    if first_reference_year > last_reference_year:
        raise ValueError("reference_years must be increasing.")
    if first_reference_year < file_start or last_reference_year > file_end:
        raise ValueError(
            f"reference_years must fall within {REFERENCE_FILE_YEARS[0]}-"
            f"{REFERENCE_FILE_YEARS[1]}."
        )

    files = {"Raw model": build_model_filename("raw")}
    files.update(
        {
            f"Bias-corrected model ({method})": build_model_filename(method)
            for method in BIAS_CORRECTION_METHODS
        }
    )
    reference_filename, _, reference_label = get_reference_configuration()
    files[reference_label] = reference_filename

    missing = [f"{label}: {filename}" for label, filename in files.items() if not filename.is_file()]
    if missing:
        raise FileNotFoundError("Required input file(s) not found:\n" + "\n".join(missing))


def check_model_dataset(ds: xr.Dataset, dataset_label: str) -> None:
    """Check the compact model variables needed by the selected tests."""
    required = {"tp24_max", MODEL_MONTH_COORDINATE, "model_type", "number", "i_date"}
    if PANEL_F_TEST == "stability_test":
        required.update(get_stability_variable_names())

    missing = required - set(ds.variables)
    if missing:
        raise KeyError(f"{dataset_label} is missing variables: {sorted(missing)}")

    if set(ds["tp24_max"].dims) != {"number", "i_date"}:
        raise ValueError(f"{dataset_label} tp24_max must have dimensions number and i_date.")
    if ds[MODEL_MONTH_COORDINATE].dims != ("i_date",):
        raise ValueError(
            f"{dataset_label} {MODEL_MONTH_COORDINATE} must have dimension ('i_date',)."
        )


def get_model_values_for_month(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
) -> np.ndarray:
    """Return finite compact model values for one calendar month."""
    if variable_name not in ds:
        raise KeyError(f"Model variable '{variable_name}' was not found.")

    calendar_month = get_model_calendar_month(ds)
    selected = ds[variable_name].where(calendar_month == month_number, drop=True)
    return remove_missing_values(selected.values)


def get_reference_values_for_month(
    ds: xr.Dataset,
    variable_name: str,
    month_number: int,
) -> np.ndarray:
    """Return selected reference years for one month, optionally excluding Hans."""
    if variable_name not in ds:
        raise KeyError(
            f"Reference variable '{variable_name}' was not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    data = ds[variable_name]
    if not {"year", "month"}.issubset(set(data.dims) | set(data.coords)):
        raise ValueError("Reference variable must contain year and month.")

    first_year, last_year = map(int, reference_years)
    selected = data.sel(year=slice(first_year, last_year), month=month_number)

    if (
        EXCLUDE_STORM_HANS_FROM_REFERENCE
        and month_number == 8
        and first_year <= 2023 <= last_year
    ):
        if 2023 not in np.asarray(selected["year"].values):
            raise ValueError("Cannot exclude Storm Hans: August 2023 is absent.")
        selected = selected.sel(year=selected["year"] != 2023)

    return remove_missing_values(selected.values)


# =============================================================================
# Independence
# =============================================================================

def normalize_model_type(values: np.ndarray) -> np.ndarray:
    """Return model-type labels as stripped lowercase strings."""
    return np.array(
        [
            (value.decode("utf-8") if isinstance(value, bytes) else str(value)).strip().lower()
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

    return float(
        np.corrcoef(
            rankdata(x_valid, method="average"),
            rankdata(y_valid, method="average"),
        )[0, 1]
    )


def get_month_member_matrix(
    ds: xr.Dataset,
    month_number: int,
    model_type: str,
) -> np.ndarray:
    """Return an initialization-by-member matrix for one month and model type."""
    if model_type not in {"forecast", "hindcast"}:
        raise ValueError("model_type must be 'forecast' or 'hindcast'.")

    calendar_month = get_model_calendar_month(ds).values
    types = normalize_model_type(ds["model_type"].values)
    selected_rows = (calendar_month == month_number) & (types == model_type)

    return (
        ds["tp24_max"]
        .isel(i_date=selected_rows)
        .transpose("i_date", "number")
        .values.astype("float64")
    )


def calculate_pairwise_correlations(matrix: np.ndarray) -> np.ndarray:
    """Return all finite member-pair Spearman correlations."""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        return np.array([], dtype="float64")

    correlations = np.array(
        [
            spearman_correlation(matrix[:, member_1], matrix[:, member_2])
            for member_1, member_2 in combinations(range(matrix.shape[1]), 2)
        ],
        dtype="float64",
    )
    return remove_missing_values(correlations)


def calculate_month_independence(ds: xr.Dataset, month_number: int) -> dict[str, object]:
    """Calculate monthly independence and convert it to a heatmap pass/fail."""
    forecast = calculate_pairwise_correlations(
        get_month_member_matrix(ds, month_number, "forecast")
    )
    hindcast = calculate_pairwise_correlations(
        get_month_member_matrix(ds, month_number, "hindcast")
    )
    correlations = np.concatenate([forecast, hindcast])

    if correlations.size == 0:
        raise ValueError(
            f"No finite pairwise Spearman correlations for {MONTH_LABELS[month_number]}."
        )

    median_correlation = float(np.median(correlations))
    return {
        "median_correlation": median_correlation,
        "passes": abs(median_correlation) < INDEPENDENCE_CORRELATION_THRESHOLD,
        "number_of_pairwise_correlations": int(correlations.size),
    }


# =============================================================================
# Fidelity bootstrap
# =============================================================================

def calculate_statistic(values: np.ndarray, statistic_name: str) -> float:
    """Calculate one fidelity statistic."""
    if statistic_name == "mean":
        return float(np.mean(values))
    if statistic_name == "std":
        return float(np.std(values, ddof=1))
    if statistic_name == "skewness":
        return float(skew(values, bias=True))
    if statistic_name == "kurtosis":
        return float(kurtosis(values, fisher=True, bias=True))
    raise ValueError(f"Unsupported statistic: {statistic_name}")


def get_vectorized_statistic_function(
    statistic_name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a statistic function operating along bootstrap axis 1."""
    if statistic_name == "mean":
        return lambda samples: np.mean(samples, axis=1)
    if statistic_name == "std":
        return lambda samples: np.std(samples, axis=1, ddof=1)
    if statistic_name == "skewness":
        return lambda samples: skew(samples, axis=1, bias=True)
    if statistic_name == "kurtosis":
        return lambda samples: kurtosis(samples, axis=1, fisher=True, bias=True)
    raise ValueError(f"Unsupported statistic: {statistic_name}")


def calculate_confidence_interval(bootstrap_values: np.ndarray) -> tuple[float, float]:
    """Return the central bootstrap confidence interval."""
    alpha = 100.0 - confidence_level_percent
    return (
        float(np.percentile(bootstrap_values, alpha / 2.0)),
        float(np.percentile(bootstrap_values, 100.0 - alpha / 2.0)),
    )


def validate_fidelity_samples(
    model_values: np.ndarray,
    reference_values: np.ndarray,
    month_number: int,
) -> None:
    """Validate samples needed by the four fidelity tests."""
    for label, values in (("model", model_values), ("reference", reference_values)):
        if values.size < 4:
            raise ValueError(
                f"Only {values.size} finite {label} values for "
                f"{MONTH_LABELS[month_number]}; at least 4 are required."
            )


def perform_month_fidelity_tests(
    model_values: np.ndarray,
    reference_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    """Run the four panel-(b)-(e) fidelity tests for one model sample."""
    sample_indices = rng.integers(
        0,
        model_values.size,
        size=(number_of_bootstrap_samples, reference_values.size),
    )
    resampled = model_values[sample_indices]
    results = {}

    for statistic_name in STATISTICS:
        statistic_function = get_vectorized_statistic_function(statistic_name)
        bootstrap = remove_missing_values(statistic_function(resampled))
        if bootstrap.size == 0:
            raise ValueError(f"No finite bootstrap {statistic_name} values were produced.")

        low, high = calculate_confidence_interval(bootstrap)
        reference_value = calculate_statistic(reference_values, statistic_name)
        results[statistic_name] = {
            "reference_value": reference_value,
            "low": low,
            "high": high,
            "passes": low <= reference_value <= high,
        }

    return results


# =============================================================================
# Panel-(f) tests
# =============================================================================

def get_ks_p_value_threshold() -> float:
    """Convert the selected KS confidence level to a p-value threshold."""
    return 1.0 - ks_significance_level_percent / 100.0


def perform_distribution_ks_test(
    model_values: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, object]:
    """Perform the panel-(f) model-versus-reference KS test."""
    if model_values.size == 0 or reference_values.size == 0:
        raise ValueError("Model or reference KS sample is empty.")

    result = ks_2samp(
        model_values,
        reference_values,
        alternative="two-sided",
        method=ks_method,
    )
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "passes": p_value >= get_ks_p_value_threshold(),
    }


def perform_stability_ks_test(
    ds: xr.Dataset,
    month_number: int,
) -> dict[str, object]:
    """Perform the panel-(f) Early-versus-Late stability KS test."""
    early_variable, late_variable = get_stability_variable_names()
    early_values = get_model_values_for_month(ds, early_variable, month_number)
    late_values = get_model_values_for_month(ds, late_variable, month_number)

    if early_values.size == 0 or late_values.size == 0:
        raise ValueError(
            f"Early or Late stability sample is empty for {MONTH_LABELS[month_number]}."
        )

    result = ks_2samp(
        early_values,
        late_values,
        alternative=ks_alternative,
        method=ks_method,
    )
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "passes": p_value >= get_ks_p_value_threshold(),
        "early_sample_size": int(early_values.size),
        "late_sample_size": int(late_values.size),
    }


def perform_panel_f_test(
    ds: xr.Dataset,
    month_number: int,
    reference_values: np.ndarray,
) -> dict[str, object]:
    """Run the selected panel-(f) test for one model dataset and month."""
    if PANEL_F_TEST == "ks_test":
        model_values = get_model_values_for_month(ds, "tp24_max", month_number)
        return perform_distribution_ks_test(model_values, reference_values)

    return perform_stability_ks_test(ds, month_number)


# =============================================================================
# Monthly evaluation
# =============================================================================

def evaluate_month(
    ds: xr.Dataset,
    reference_values: np.ndarray,
    month_number: int,
) -> dict[str, object]:
    """Run all six heatmap tests for one model dataset and calendar month."""
    model_values = get_model_values_for_month(ds, "tp24_max", month_number)
    validate_fidelity_samples(model_values, reference_values, month_number)

    fidelity = perform_month_fidelity_tests(
        model_values,
        reference_values,
        np.random.default_rng(random_seed + month_number),
    )

    return {
        "independence": calculate_month_independence(ds, month_number),
        "fidelity": fidelity,
        "panel_f": perform_panel_f_test(ds, month_number, reference_values),
    }


def get_test_passes(results: dict[str, object]) -> dict[str, bool]:
    """Return the six pass/fail values from one monthly evaluation."""
    return {
        "independence": bool(results["independence"]["passes"]),
        **{
            statistic_name: bool(results["fidelity"][statistic_name]["passes"])
            for statistic_name in STATISTICS
        },
        PANEL_F_TEST: bool(results["panel_f"]["passes"]),
    }


def calculate_raw_month_gate(
    raw_model_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    reference_variable: str,
) -> pd.DataFrame:
    """Evaluate all six raw tests and identify months needing correction."""
    rows = []

    for month_number in range(1, 13):
        reference_values = get_reference_values_for_month(
            reference_ds, reference_variable, month_number
        )
        results = evaluate_month(raw_model_ds, reference_values, month_number)
        passes = get_test_passes(results)
        failed_tests = [name for name, passed in passes.items() if not passed]

        rows.append(
            {
                "month": month_number,
                "month_name": MONTH_LABELS[month_number],
                **{f"raw_{name}_passes": passed for name, passed in passes.items()},
                "raw_failed_any_test": bool(failed_tests),
                "raw_failed_tests": ",".join(failed_tests),
            }
        )

    return pd.DataFrame(rows)


def build_month_correction_lookup(raw_month_gate: pd.DataFrame) -> dict[int, bool]:
    """Return month -> whether corrected data should be used."""
    return {
        int(row["month"]): bool(row["raw_failed_any_test"])
        for _, row in raw_month_gate.iterrows()
    }


def calculate_summary_counts(
    raw_model_ds: xr.Dataset,
    corrected_model_datasets: dict[str, xr.Dataset],
    reference_ds: xr.Dataset,
    reference_variable: str,
    month_correction_lookup: dict[int, bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate all heatmap counts and detailed monthly results."""
    row_names = ["raw", *BIAS_CORRECTION_METHODS]
    column_names = ["independence", *STATISTICS, PANEL_F_TEST]
    counts = pd.DataFrame(0, index=row_names, columns=column_names, dtype="int64")
    monthly_rows = []

    for month_number in range(1, 13):
        reference_values = get_reference_values_for_month(
            reference_ds, reference_variable, month_number
        )

        raw_results = evaluate_month(raw_model_ds, reference_values, month_number)
        raw_passes = get_test_passes(raw_results)

        for test_name, passes in raw_passes.items():
            counts.loc["raw", test_name] += int(passes)

        monthly_rows.append(
            {
                "dataset": "raw",
                "month": month_number,
                "month_name": MONTH_LABELS[month_number],
                "bias_correction_applied": False,
                **{f"{name}_passes": passes for name, passes in raw_passes.items()},
            }
        )

        for method in BIAS_CORRECTION_METHODS:
            use_corrected = (
                not BIAS_CORRECT_ONLY_FAILED_MONTHS
                or month_correction_lookup[month_number]
            )
            selected_ds = corrected_model_datasets[method] if use_corrected else raw_model_ds
            results = evaluate_month(selected_ds, reference_values, month_number)
            passes_by_test = get_test_passes(results)

            for test_name, passes in passes_by_test.items():
                counts.loc[method, test_name] += int(passes)

            monthly_rows.append(
                {
                    "dataset": method,
                    "month": month_number,
                    "month_name": MONTH_LABELS[month_number],
                    "bias_correction_applied": use_corrected,
                    **{
                        f"{name}_passes": passes
                        for name, passes in passes_by_test.items()
                    },
                }
            )

    return counts, pd.DataFrame(monthly_rows)


# =============================================================================
# Reporting and heatmap
# =============================================================================

def print_raw_month_gate(raw_month_gate: pd.DataFrame) -> None:
    """Print which raw months trigger selective correction."""
    print("\nRaw all-test screening\n----------------------")
    print(f"{'Month':<12}{'Needs BC':>10}  Failed tests")
    print("-" * 64)

    for _, row in raw_month_gate.iterrows():
        failed_tests = row["raw_failed_tests"] or "-"
        print(
            f"{row['month_name']:<12}"
            f"{str(bool(row['raw_failed_any_test'])):>10}  "
            f"{failed_tests}"
        )


def print_monthly_results(monthly_results: pd.DataFrame) -> None:
    """Print monthly pass/fail results for every heatmap row."""
    test_names = ["independence", *STATISTICS, PANEL_F_TEST]

    for dataset_name in ["raw", *BIAS_CORRECTION_METHODS]:
        selected = monthly_results[monthly_results["dataset"] == dataset_name]
        print(f"\nMonthly results: {dataset_name}\n" + "-" * 100)

        header = f"{'Month':<12}" + "".join(f"{name:>14}" for name in test_names)
        header += f"{'Applied':>10}"
        print(header)
        print("-" * len(header))

        for _, row in selected.iterrows():
            statuses = "".join(
                f"{('PASS' if row[f'{name}_passes'] else 'FAIL'):>14}"
                for name in test_names
            )
            print(
                f"{row['month_name']:<12}{statuses}"
                f"{str(bool(row['bias_correction_applied'])):>10}"
            )


def get_heatmap_column_labels() -> dict[str, str]:
    """Return readable heatmap column labels."""
    final_label = "fidelity: KS test" if PANEL_F_TEST == "ks_test" else "stability"
    return {
        "independence": "independence",
        "mean": "fidelity: mean",
        "std": "fidelity: std",
        "skewness": "fidelity: skewness",
        "kurtosis": "fidelity: kurtosis",
        PANEL_F_TEST: final_label,
    }


def make_summary_heatmap(
    summary_counts: pd.DataFrame,
    filename: Path | None = None,
) -> None:
    """Plot monthly pass counts for the six selected tests."""
    figure, axis = plt.subplots(
        figsize=(10, 1.0 + 0.65 * len(summary_counts.index))
    )

    axis.imshow(
        12 - summary_counts.values,
        cmap=colormaps["Blues"],
        vmin=0,
        vmax=6,
        aspect="auto",
    )

    for (row, column), value in np.ndenumerate(summary_counts.values):
        axis.text(
            column,
            row,
            int(value),
            ha="center",
            va="center",
            color=HEATMAP_NUMBER_COLOR,
        )

    labels = get_heatmap_column_labels()
    axis.set_xticks(
        range(len(summary_counts.columns)),
        [labels[column] for column in summary_counts.columns],
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    #axis.set_yticks(range(len(summary_counts.index)), summary_counts.index)
    if len(BIAS_CORRECTION_METHODS) == 1:
        y_labels = ["raw", "model BC"]
    else:
        y_labels = list(summary_counts.index)

    axis.set_yticks(range(len(summary_counts.index)), y_labels)
    axis.set_title("UNSEEN tests")
    figure.tight_layout()

    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, bbox_inches="tight", dpi=400)
        print("Wrote:", filename)

    if show_figure:
        plt.show()
    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run all monthly tests and create the summary heatmap."""
    validate_user_settings()

    raw_model_filename = build_model_filename("raw")
    corrected_model_filenames = {
        method: build_model_filename(method) for method in BIAS_CORRECTION_METHODS
    }
    reference_filename, reference_variable, reference_label = get_reference_configuration()
    output_filename = build_output_filename()

    print("Input files\n-----------")
    print("Raw model:", raw_model_filename)
    for method, filename in corrected_model_filenames.items():
        print(f"{method:>8}:", filename)
    print(f"{reference_label}:", reference_filename)

    print("\nAnalysis settings\n-----------------")
    print("Panel-f test:", PANEL_F_TEST)
    print("Reference dataset:", REFERENCE_DATASET)
    print("Reference years:", reference_years)
    print("Bias-correction methods:", BIAS_CORRECTION_METHODS)
    print("Bias correct only failed months:", BIAS_CORRECT_ONLY_FAILED_MONTHS)

    with (
        xr.open_dataset(raw_model_filename, decode_timedelta=False) as raw_model_ds,
        xr.open_dataset(reference_filename) as reference_ds,
    ):
        corrected_model_datasets = {
            method: xr.open_dataset(filename, decode_timedelta=False)
            for method, filename in corrected_model_filenames.items()
        }

        try:
            check_model_dataset(raw_model_ds, "Raw model dataset")
            for method, dataset in corrected_model_datasets.items():
                check_model_dataset(dataset, f"Bias-corrected model dataset ({method})")

            raw_month_gate = calculate_raw_month_gate(
                raw_model_ds,
                reference_ds,
                reference_variable,
            )
            month_correction_lookup = build_month_correction_lookup(raw_month_gate)

            summary_counts, monthly_results = calculate_summary_counts(
                raw_model_ds,
                corrected_model_datasets,
                reference_ds,
                reference_variable,
                month_correction_lookup,
            )
        finally:
            for dataset in corrected_model_datasets.values():
                dataset.close()

    print_raw_month_gate(raw_month_gate)
    print_monthly_results(monthly_results)

    print("\nHeatmap counts (months passed out of 12)\n----------------------------------------")
    print(summary_counts)

    make_summary_heatmap(
        summary_counts,
        output_filename if write2file else None,
    )


if __name__ == "__main__":
    main()
