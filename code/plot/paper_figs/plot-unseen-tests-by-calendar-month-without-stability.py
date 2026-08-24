"""
Create a six-panel UNSEEN diagnostic figure for one selected calendar month.

The figure combines two checks:
    1. ensemble-member independence;
    2. fidelity of raw and bias-corrected S2S distributions relative to one
       selected reference dataset.

Inputs
------
The script reads three NetCDF files:
    - the raw compact S2S monthly extreme-sample file;
    - the bias-corrected compact S2S monthly extreme-sample file;
    - ONE selected reference dataset: ERA5 or SeNorge.

The compact S2S files organize samples as (number, i_date). Sample-month
membership is stored as sample_month(i_date) in YYYYMM format, model source is
stored in model_type(i_date), and the unique i_date coordinate identifies each
forecast or hindcast initialization row. Calendar month is derived internally
as sample_month % 100.

Panel (a) calculates independence from the raw all-lead sample.
Panels (b)-(e) overlay raw and bias-corrected bootstrap distributions and their
95% intervals, with one vertical line for the selected reference dataset. The reference files are always read for 1957-2025, while `reference_years` selects
the subset used in the fidelity calculations. An optional user setting can exclude
August 2023 Storm Hans when 2023 is included in `reference_years`.
Panel (f) plots the complete selected-month raw model, bias-corrected model,
and reference distributions. Two-sided Kolmogorov-Smirnov tests compare the raw
and bias-corrected model samples directly with the reference sample.

For the default settings
    first_input_lead = 16
    last_input_lead = 46
    x_days = 2
    number_of_lead_bins = 2

the usable accumulated ending leads are 17-46 and the raw S2S variables are:
    all leads : tp24_max
    early     : tp24_max_lead17_31
    late      : tp24_max_lead32_46

Bias-corrected compact files preserve the same maximum-variable names. The
bias-correction method and reference dataset are encoded in the filename.


Panel (a): Independence
-----------------------
Shows one boxplot of pairwise Spearman rank correlations between ensemble
members for the selected month. Forecast and hindcast correlations are pooled.

Correlations near zero indicate weak dependence between ensemble members.
Larger positive or negative correlations indicate stronger dependence.


Panel (b): Fidelity of the mean
-------------------------------
Uses the complete all-lead S2S sample.

The model sample is repeatedly resampled with replacement using the same sample
size as the observational datasets. The resulting bootstrap distribution of
the mean is shown together with the central model confidence interval and the
ERA5 and SeNorge means.


Panel (c): Fidelity of the standard deviation
---------------------------------------------
Uses the same bootstrap procedure, but for sample standard deviation.

This tests whether the observed spread of monthly extremes is consistent with the spread expected from the S2S distribution.


Panel (d): Fidelity of the skewness
-----------------------------------
Uses the same bootstrap procedure for skewness.

This tests whether the asymmetry of the observed extreme-precipitation
distribution is consistent with the S2S distribution.


Panel (e): Fidelity of the kurtosis
-----------------------------------
Uses the same bootstrap procedure for excess kurtosis.

This tests whether the tail-heaviness / peakedness of the observed extreme
distribution is consistent with the S2S distribution.


Panel (f): Distributional fidelity
----------------------------------
Uses all finite values for the selected calendar month.

The raw and bias-corrected model distributions are plotted together with the
selected reference distribution. Two-sided two-sample Kolmogorov-Smirnov tests
compare each model distribution directly with the reference distribution. The
panel reports the KS D statistic and p-value.

Data used by each panel
-----------------------
Panel (a):
    complete all-lead S2S sample + i_date + number + model_type from the
    shared compact S2S extreme-sample file. hdate is retained as provenance
    but is not required because every hindcast row already has a unique i_date.

Panels (b)-(e):
    complete all-lead S2S sample + ERA5 + SeNorge.

Panel (f):
    complete all-lead raw model sample + complete all-lead bias-corrected model
    sample + selected reference sample.
"""


import os
from itertools import combinations
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp, kurtosis, rankdata, skew

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Calendar month to plot: 1=January, ..., 12=December.
selected_month = 8

# Accumulation period.
x_days = 2

# Catchment used consistently for all input datasets and figure labels.
catchment = "regine_drammen"

forecast_date_range = ("2020-01-02", "2023-12-28")

reference_years = ("1957", "2022")
REFERENCE_FILE_YEARS = ("1957", "2025")

era5_grid = "0.5x0.5"

# Lead-location sampling used by panels (b)-(f).
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Model-data / bias-correction method.
#
# Options:
#     "raw"   : uncorrected compact monthly-maximum sample
#     "mm"    : monthly-mean multiplicative correction from script 2
#     "q"     : quantile-corrected compact sample from script 3
#     "ld"    : lead-day-corrected compact sample from script 3
#     "doy"   : day-of-year-corrected compact sample from script 3
#     "q_doy" : quantile/day-of-year-corrected compact sample from script 3
#
# For any method other than "raw", the selected corrected model is compared
# with the raw model in panels (b)-(f). For "raw", only the raw model is shown.
BIAS_CORRECTION_METHOD = "mm_2step"

# Reference dataset used for BOTH:
#   1. the vertical reference line in panels (b)-(e); and
#   2. selecting the corresponding corrected model file.
#
# Options:
#     "era5"
#     "senorge"
REFERENCE_DATASET = "senorge"

# Exclude the August 2023 Storm Hans reference value before fidelity calculations.
EXCLUDE_STORM_HANS_FROM_REFERENCE = False

# Independence-test settings.
# Minimum number of paired initialization values required for one
# ensemble-member Spearman correlation.
minimum_samples = 10

# Bootstrap settings.
number_of_bootstrap_samples = 10000
confidence_level_percent = 95.0
random_seed = 42

# Kolmogorov-Smirnov distribution-test settings.
ks_alternative = "two-sided"
ks_method = "auto"
ks_significance_level_percent = 95.0

# Histogram settings.
number_of_bins = 30
plot_probability_density = True
y_axis_margin_fraction = 0.08

# Figure output.
figure_width = 13.0
figure_height = 8.0
figure_dpi = 300

write2file = False
show_figure = True


# =============================================================================
# Dataset configuration
# =============================================================================

MODEL_VARIABLE = "tp24"
# S2S maximum-variable names are built automatically from lead ranges.
# Scripts 2 and 3 both store sample_month(i_date) as YYYYMM.
MODEL_MONTH_COORDINATE = "sample_month"

ERA5_VARIABLE = "tp24"

SENORGE_VARIABLE = "rr"
SENORGE_LABEL = "SeNorge"


# =============================================================================
# Labels and plotting constants
# =============================================================================

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

STATISTICS = ("mean", "std", "skewness", "kurtosis")

STATISTIC_LABELS = {
    "mean": "Mean",
    "std": "Standard deviation",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
}


STATISTIC_AXIS_LABELS = {
    "mean": f"Maximum monthly {x_days}-day precipitation [mm]",
    "std": f"Maximum monthly {x_days}-day precipitation [mm]",
    "skewness": "unitless",
    "kurtosis": "unitless",
}



# Raw and bias-corrected bootstrap distributions in panels (b)-(e).
# Semi-transparent filled histograms make their overlap visually apparent,
# similar in spirit to Kelder et al. (2020), Fig. 4.
RAW_MODEL_COLOR = "0.45"
BIAS_CORRECTED_COLOR = "goldenrod"
BOOTSTRAP_ALPHA = 0.45

MODEL_COLOR = "black"
ERA5_COLOR = "tab:blue"
SENORGE_COLOR = "tab:red"
EARLY_COLOR = "tab:green"
LATE_COLOR = "tab:purple"

HISTOGRAM_LINEWIDTH = 2
REFERENCE_LINEWIDTH = 2
CONFIDENCE_LINEWIDTH = 2

TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12
AXIS_LABELSIZE = 10
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 10


# =============================================================================
# General helpers
# =============================================================================

def readable_catchment_name(catchment_name: str) -> str:
    """Convert a technical catchment identifier into a readable name."""

    name = catchment_name
    for prefix in ("nve_catchment_regine_", "nve_catchment_", "regine_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    return name.replace("_", " ").title()


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


def validate_user_settings() -> None:
    """Validate settings that affect both analyses."""

    if selected_month not in MONTH_LABELS:
        raise ValueError("selected_month must be an integer from 1 to 12.")

    if not isinstance(EXCLUDE_STORM_HANS_FROM_REFERENCE, bool):
        raise TypeError("EXCLUDE_STORM_HANS_FROM_REFERENCE must be True or False.")

    first_reference_year, last_reference_year = map(int, reference_years)
    if first_reference_year > last_reference_year:
        raise ValueError("reference_years must be increasing.")

    file_start, file_end = map(int, REFERENCE_FILE_YEARS)
    if first_reference_year < file_start or last_reference_year > file_end:
        raise ValueError(
            f"reference_years must fall within the fixed reference file range "
            f"{file_start}-{file_end}."
        )

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if number_of_bootstrap_samples < 1:
        raise ValueError("number_of_bootstrap_samples must be at least 1.")

    if number_of_bins < 1:
        raise ValueError("number_of_bins must be at least 1.")

    if not 0.0 < confidence_level_percent < 100.0:
        raise ValueError(
            "confidence_level_percent must be between 0 and 100."
        )

    valid_ks_alternatives = {"two-sided", "less", "greater"}
    if ks_alternative not in valid_ks_alternatives:
        raise ValueError(
            f"ks_alternative must be one of {sorted(valid_ks_alternatives)}."
        )

    valid_ks_methods = {"auto", "exact", "asymp"}
    if ks_method not in valid_ks_methods:
        raise ValueError(
            f"ks_method must be one of {sorted(valid_ks_methods)}."
        )

    if not 0.0 < ks_significance_level_percent < 100.0:
        raise ValueError(
            "ks_significance_level_percent must be between 0 and 100."
        )

    if y_axis_margin_fraction < 0:
        raise ValueError(
            "y_axis_margin_fraction must be non-negative."
        )

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = first_input_lead + x_days - 1
    number_of_usable_leads = last_input_lead - first_usable_lead + 1

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )

    if number_of_lead_bins != 2:
        raise ValueError(
            "This combined stability figure is configured for exactly two "
            "lead bins: Early and Late."
        )

    if number_of_lead_bins > number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins exceeds the number of usable leads."
        )

    if minimum_samples < 3:
        raise ValueError(
            "minimum_samples must be at least 3."
        )


    valid_methods = {
        "raw",
        "mm",
        "q",
        "ld",
        "doy",
        "q_doy",
    }

    if BIAS_CORRECTION_METHOD not in valid_methods:
        raise ValueError(
            f"BIAS_CORRECTION_METHOD must be one of "
            f"{sorted(valid_methods)}. "
            f"Got '{BIAS_CORRECTION_METHOD}'."
        )

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


# =============================================================================
# Filename helpers
# =============================================================================

def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """Split usable accumulated ending leads into approximately equal bins."""

    number_of_leads = last_lead - first_lead + 1
    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

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


def get_stability_lead_ranges() -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    """Return complete, early, and late accumulated lead ranges."""

    first_usable_lead = first_input_lead + x_days - 1

    full_range = (first_usable_lead, last_input_lead)

    split_ranges = split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )

    return full_range, split_ranges[0], split_ranges[1]


def get_stability_variable_names() -> tuple[str, str, str]:
    """Return compact complete, early, and late model-variable names.

    Scripts 2 and 3 preserve the compact variable names in their output files;
    the correction method and reference dataset are encoded in the filename.
    """

    _, early_range, late_range = get_stability_lead_ranges()

    all_variable = "tp24_max"
    early_variable = (
        f"tp24_max_lead{early_range[0]}_{early_range[1]}"
    )
    late_variable = (
        f"tp24_max_lead{late_range[0]}_{late_range[1]}"
    )

    return all_variable, early_variable, late_variable


def build_model_filename(method: str) -> str:
    """Build a compact S2S filename produced by script 2 or script 3."""

    full_range, early_range, late_range = get_stability_lead_ranges()
    lead_label = (
        f"lead{full_range[0]}-{full_range[1]}_split{number_of_lead_bins}_"
        f"{early_range[0]}-{early_range[1]}_{late_range[0]}-{late_range[1]}"
    )
    stem = (
        f"test-monthly_max_samples_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"{get_file_id(catchment)}_{lead_label}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}"
    )

    filename = os.path.join(config.dirs["s2s_processed"], stem)
    suffix = "" if method == "raw" else f"_bc_{method}_{REFERENCE_DATASET}"
    return f"{filename}{suffix}.nc"

def resolve_model_input_filenames() -> tuple[str, str]:
    """Return raw and selected corrected compact S2S input filenames."""

    raw_filename = build_model_filename("raw")
    selected_filename = build_model_filename(BIAS_CORRECTION_METHOD)
    return raw_filename, selected_filename

def build_era5_filename() -> str:
    """Build the fixed 1957-2025 ERA5 reference filename."""

    return (
        f"{config.dirs['era5_processed']}"
        f"monthly_max_samples_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_"
        f"{REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}.nc"
    )


def build_senorge_filename() -> str:
    """Build the fixed 1957-2025 SeNorge reference filename."""

    return (
        f"{config.dirs['senorge_processed']}"
        f"monthly_max_samples_{SENORGE_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_"
        f"{REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}.nc"
    )


def get_reference_configuration() -> tuple[str, str, str]:
    """Return selected reference filename, variable, and display label."""

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


def build_output_filename() -> str:
    """Create a descriptive filename for the six-panel figure."""

    month_name = MONTH_LABELS[selected_month].lower()

    return os.path.join(
        config.dirs["fig"],
        (
            f"UNSEEN_independence_fidelity_no_stability_tests_"
            f"{month_name}_{x_days}dayacc_{catchment}_"
            f"{forecast_date_range[0]}_{forecast_date_range[1]}_"
            f"raw_{BIAS_CORRECTION_METHOD}_{REFERENCE_DATASET}_{reference_years[0]}-{reference_years[-1]}.png"
        ),
    )

# =============================================================================
# Independence calculation from the shared S2S sample
# =============================================================================

def normalize_model_type(values: np.ndarray) -> np.ndarray:
    """Return model-type values as stripped lowercase strings."""

    flat_values = np.asarray(values).ravel()

    return np.array(
        [
            (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
            ).strip().lower()
            for value in flat_values
        ],
        dtype=object,
    )


def datetime_values_to_key(values: np.ndarray) -> np.ndarray:
    """Convert forecast_date values to datetime64[ns] keys."""

    return pd.to_datetime(
        np.asarray(values).ravel(),
        errors="coerce",
    ).to_numpy(
        dtype="datetime64[ns]"
    )


def hdate_values_to_key(values: np.ndarray) -> np.ndarray:
    """Convert hdate values to integer YYYYMMDD keys."""

    values = np.asarray(values).ravel()

    if np.issubdtype(
        values.dtype,
        np.datetime64,
    ):
        dates = pd.to_datetime(
            values,
            errors="coerce",
        )

        out = np.full(
            values.size,
            -99999999,
            dtype="int64",
        )

        valid = ~pd.isna(dates)

        out[valid] = (
            dates[valid]
            .strftime("%Y%m%d")
            .astype("int64")
        )

        return out

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    out = np.full(
        values.size,
        -99999999,
        dtype="int64",
    )

    valid = np.isfinite(
        numeric_values
    )

    out[valid] = (
        numeric_values[valid]
        .astype("int64")
    )

    return out


def get_independence_month_samples(
    model_ds: xr.Dataset,
    all_variable: str,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract finite selected-month samples for the independence calculation."""

    required_variables = {
        all_variable,
        MODEL_MONTH_COORDINATE,
        "model_type",
        "number",
        "i_date",
    }
    missing = required_variables.difference(model_ds.variables)
    if missing:
        raise KeyError(
            "Model dataset is missing variables needed for the independence "
            f"calculation: {sorted(missing)}"
        )

    if model_type not in {"forecast", "hindcast"}:
        raise ValueError("model_type must be 'forecast' or 'hindcast'.")

    calendar_month = get_model_calendar_month(model_ds).values
    normalized_types = normalize_model_type(model_ds["model_type"].values)
    initialization_mask = (
        (calendar_month == selected_month) & (normalized_types == model_type)
    )

    selected_i_dates = model_ds["i_date"].values[initialization_mask]
    selected_values = (
        model_ds[all_variable]
        .isel(i_date=initialization_mask)
        .transpose("i_date", "number")
        .values.astype("float64")
    )

    member_labels = model_ds["number"].values.astype("int64")

    values = selected_values.ravel()
    initialization_keys = np.repeat(selected_i_dates, member_labels.size)
    members = np.tile(member_labels, selected_i_dates.size)

    valid = np.isfinite(values)

    return (
        values[valid],
        initialization_keys[valid],
        members[valid],
    )

def reconstruct_member_matrix(
    values: np.ndarray,
    initialization_keys: np.ndarray,
    member_labels: np.ndarray,
    model_type: str,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Reconstruct an initialization-by-member matrix."""

    if values.size == 0:
        return (
            np.empty(
                (0, 0),
                dtype="float64",
            ),
            [],
            np.array([]),
        )

    unique_initializations = []
    initialization_lookup = {}

    for initialization in initialization_keys:

        if initialization not in initialization_lookup:

            initialization_lookup[
                initialization
            ] = len(
                unique_initializations
            )

            unique_initializations.append(
                initialization
            )

    unique_members = np.unique(member_labels)

    if unique_members.size < 2:
        raise ValueError(
            f"{model_type.capitalize()} data contain fewer than two ensemble members."
        )

    member_lookup = {
        member: index
        for index, member in enumerate(unique_members)
    }

    matrix = np.full(
        (
            len(
                unique_initializations
            ),
            unique_members.size,
        ),
        np.nan,
        dtype="float64",
    )

    for value, initialization, member in zip(
        values,
        initialization_keys,
        member_labels,
    ):

        row = initialization_lookup[
            initialization
        ]

        column = member_lookup[
            member
        ]

        if np.isfinite(
            matrix[
                row,
                column,
            ]
        ):
            raise ValueError(
                "Duplicate sample found for "
                f"{model_type} initialization "
                f"{initialization!r}, member {member!r}."
            )

        matrix[
            row,
            column,
        ] = value

    return (
        matrix,
        unique_initializations,
        unique_members,
    )


def spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
    minimum_valid_samples: int,
) -> float:
    """Calculate one pairwise Spearman rank correlation."""

    valid = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
    )

    number_of_valid_samples = int(
        valid.sum()
    )

    if (
        number_of_valid_samples
        < minimum_valid_samples
    ):
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        np.all(
            x_valid
            == x_valid[0]
        )
        or np.all(
            y_valid
            == y_valid[0]
        )
    ):
        return np.nan

    x_ranks = rankdata(
        x_valid,
        method="average",
    )

    y_ranks = rankdata(
        y_valid,
        method="average",
    )

    return float(
        np.corrcoef(
            x_ranks,
            y_ranks,
        )[0, 1]
    )


def calculate_selected_month_correlations(
    model_ds: xr.Dataset,
    all_variable: str,
    model_type: str,
) -> np.ndarray:
    """
    Calculate all member-pair correlations for the selected month.
    """

    (
        values,
        initialization_keys,
        member_labels,
    ) = get_independence_month_samples(
        model_ds=model_ds,
        all_variable=all_variable,
        model_type=model_type,
    )

    (
        matrix,
        _,
        unique_members,
    ) = reconstruct_member_matrix(
        values=values,
        initialization_keys=initialization_keys,
        member_labels=member_labels,
        model_type=model_type,
    )

    if matrix.size == 0:
        return np.array(
            [],
            dtype="float64",
        )

    pair_indices = list(
        combinations(
            range(
                unique_members.size
            ),
            2,
        )
    )

    correlations = np.array(
        [
            spearman_correlation(
                x=matrix[
                    :,
                    index_1,
                ],
                y=matrix[
                    :,
                    index_2,
                ],
                minimum_valid_samples=minimum_samples,
            )
            for index_1, index_2
            in pair_indices
        ],
        dtype="float64",
    )

    return remove_missing_values(
        correlations
    )


def calculate_independence_values(
    model_ds: xr.Dataset,
    all_variable: str,
) -> np.ndarray:
    """
    Calculate and pool forecast/hindcast correlations for panel (a).
    """

    forecast = calculate_selected_month_correlations(
        model_ds=model_ds,
        all_variable=all_variable,
        model_type="forecast",
    )

    hindcast = calculate_selected_month_correlations(
        model_ds=model_ds,
        all_variable=all_variable,
        model_type="hindcast",
    )

    combined = np.concatenate(
        [
            forecast,
            hindcast,
        ]
    )

    if combined.size == 0:
        raise ValueError(
            f"No finite independence correlations could be calculated "
            f"for {MONTH_LABELS[selected_month]}."
        )

    return combined


# =============================================================================
# Moments data: script 2 logic for one month
# =============================================================================

def check_variable_exists(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> None:
    """Raise a clear error when a required variable is missing."""

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables: {list(ds.data_vars)}"
        )


def check_coordinate_exists(
    data: xr.DataArray,
    coordinate: str,
    dataset_name: str,
) -> None:
    """Raise a clear error when a required coordinate is missing."""

    available_names = set(data.coords) | set(data.dims)

    if coordinate not in available_names:
        raise KeyError(
            f"Coordinate/dimension '{coordinate}' was not found in "
            f"{dataset_name}. Dimensions: {data.dims}; "
            f"coordinates: {list(data.coords)}."
        )


def get_model_values_for_selected_month(
    model_ds: xr.Dataset,
    variable_name: str,
) -> np.ndarray:
    """Extract one compact model sample for the selected calendar month."""

    check_variable_exists(model_ds, variable_name, "model dataset")
    data = model_ds[variable_name]

    required_dimensions = {"number", "i_date"}
    if set(data.dims) != required_dimensions:
        raise ValueError(
            f"Variable '{variable_name}' must contain dimensions "
            f"{sorted(required_dimensions)}, but has {data.dims}."
        )

    calendar_month = get_model_calendar_month(model_ds)
    selected = data.where(calendar_month == selected_month, drop=True)
    return remove_missing_values(selected.values)

def get_reference_values_for_selected_month(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> np.ndarray:
    """Extract selected reference years, optionally excluding August 2023."""

    check_variable_exists(ds, variable, dataset_name)
    data = ds[variable]

    check_coordinate_exists(data, "month", dataset_name)
    check_coordinate_exists(data, "year", dataset_name)

    first_year, last_year = map(int, reference_years)
    selected = data.sel(
        year=slice(first_year, last_year),
        month=selected_month,
    )

    hans_in_reference_years = first_year <= 2023 <= last_year
    if (
        EXCLUDE_STORM_HANS_FROM_REFERENCE
        and selected_month == 8
        and hans_in_reference_years
    ):
        if 2023 not in np.asarray(selected["year"].values):
            raise ValueError(
                f"Cannot exclude Storm Hans because August 2023 is not present "
                f"in the {dataset_name} reference sample."
            )

        selected = selected.sel(year=selected["year"] != 2023)

    return remove_missing_values(selected.values)


def load_model_and_reference_values(
    raw_model_filename: str,
    selected_model_filename: str,
    reference_filename: str,
    reference_variable: str,
    reference_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the samples needed by the six diagnostic panels."""

    for dataset_name, filename in (
        ("raw model", raw_model_filename),
        ("selected model", selected_model_filename),
        (reference_label, reference_filename),
    ):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{dataset_name} input file does not exist:\n{filename}"
            )

    all_variable, _, _ = get_stability_variable_names()

    with (
        xr.open_dataset(raw_model_filename, decode_timedelta=False) as raw_model_ds,
        xr.open_dataset(selected_model_filename, decode_timedelta=False) as bc_model_ds,
        xr.open_dataset(reference_filename) as reference_ds,
    ):
        independence_values = calculate_independence_values(
            model_ds=raw_model_ds,
            all_variable=all_variable,
        )
        raw_all_values = get_model_values_for_selected_month(
            raw_model_ds,
            all_variable,
        )
        bc_all_values = get_model_values_for_selected_month(
            bc_model_ds,
            all_variable,
        )
        reference_values = get_reference_values_for_selected_month(
            reference_ds,
            reference_variable,
            reference_label,
        )

    return independence_values, raw_all_values, bc_all_values, reference_values


def validate_moments_samples(
    raw_model_values: np.ndarray,
    bias_corrected_model_values: np.ndarray,
    reference_values: np.ndarray,
    reference_label: str,
) -> None:
    """Check the samples required by the four fidelity tests."""

    minimum_sample_size = 4

    for dataset_name, values in (
        ("raw model", raw_model_values),
        ("bias-corrected model", bias_corrected_model_values),
        (reference_label, reference_values),
    ):
        if values.size < minimum_sample_size:
            raise ValueError(
                f"Only {values.size} finite {dataset_name} values were found "
                f"for {MONTH_LABELS[selected_month]}. At least "
                f"{minimum_sample_size} are required."
            )

    if raw_model_values.size != bias_corrected_model_values.size:
        raise ValueError(
            "Raw and bias-corrected all-lead samples have different finite "
            f"sample sizes: raw={raw_model_values.size}, "
            f"bias-corrected={bias_corrected_model_values.size}."
        )


# =============================================================================
# Statistics and bootstrap
# =============================================================================

def calculate_statistic(
    values: np.ndarray,
    statistic_name: str,
) -> float:
    """Calculate one requested sample statistic."""

    if statistic_name == "mean":
        return float(np.mean(values))

    if statistic_name == "std":
        return float(np.std(values, ddof=1))

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

    raise ValueError(f"Unsupported statistic: {statistic_name}")


def get_vectorized_statistic_function(
    statistic_name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a statistic function that operates along bootstrap axis 1."""

    if statistic_name == "mean":
        return lambda samples: np.mean(samples, axis=1)

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

    raise ValueError(f"Unsupported statistic: {statistic_name}")


def calculate_confidence_interval(
    bootstrap_values: np.ndarray,
) -> tuple[float, float]:
    """Return the central bootstrap confidence interval."""

    alpha_percent = 100.0 - confidence_level_percent

    lower = np.percentile(
        bootstrap_values,
        alpha_percent / 2.0,
    )

    upper = np.percentile(
        bootstrap_values,
        100.0 - alpha_percent / 2.0,
    )

    return float(lower), float(upper)


def perform_all_moments_tests(
    raw_model_values: np.ndarray,
    bias_corrected_model_values: np.ndarray,
    reference_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    """
    Run the four fidelity diagnostics for raw and bias-corrected model samples.

    The SAME bootstrap indices are used for raw and bias-corrected samples.
    This makes their distributions directly comparable because the two model
    arrays represent the same index-aligned events.
    """

    sample_size = reference_values.size

    sample_indices = rng.integers(
        low=0,
        high=raw_model_values.size,
        size=(
            number_of_bootstrap_samples,
            sample_size,
        ),
    )

    raw_resampled = (
        raw_model_values[
            sample_indices
        ]
    )

    bc_resampled = (
        bias_corrected_model_values[
            sample_indices
        ]
    )

    results = {}

    for statistic_name in STATISTICS:

        statistic_function = (
            get_vectorized_statistic_function(
                statistic_name
            )
        )

        raw_bootstrap_values = remove_missing_values(
            statistic_function(
                raw_resampled
            )
        )

        bc_bootstrap_values = remove_missing_values(
            statistic_function(
                bc_resampled
            )
        )

        if (
            raw_bootstrap_values.size == 0
            or bc_bootstrap_values.size == 0
        ):
            raise ValueError(
                f"No finite bootstrap {statistic_name} values were produced."
            )

        raw_confidence_interval = (
            calculate_confidence_interval(
                raw_bootstrap_values
            )
        )

        bc_confidence_interval = (
            calculate_confidence_interval(
                bc_bootstrap_values
            )
        )

        reference_value = calculate_statistic(
            reference_values,
            statistic_name,
        )

        raw_lower, raw_upper = (
            raw_confidence_interval
        )

        bc_lower, bc_upper = (
            bc_confidence_interval
        )

        results[
            statistic_name
        ] = {
            "raw_bootstrap_values": raw_bootstrap_values,
            "bc_bootstrap_values": bc_bootstrap_values,
            "raw_confidence_interval": raw_confidence_interval,
            "bc_confidence_interval": bc_confidence_interval,
            "sample_size": sample_size,
            "reference_value": reference_value,
            "raw_passes": (
                raw_lower
                <= reference_value
                <= raw_upper
            ),
            "bc_passes": (
                bc_lower
                <= reference_value
                <= bc_upper
            ),
        }

    return results



# =============================================================================
# Distributional fidelity KS test
# =============================================================================

def get_ks_significance_threshold() -> float:
    """Convert the selected KS confidence level to a p-value threshold."""
    return 1.0 - ks_significance_level_percent / 100.0


def perform_fidelity_ks_test(
    model_values: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, object]:
    """Compare model and reference samples with a two-sided two-sample KS test."""

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
        "reject_null": p_value < get_ks_significance_threshold(),
    }


def format_ks_p_value(p_value: float) -> str:
    """Format a KS p-value compactly."""
    return f"{p_value:.1e}" if p_value < 0.001 else f"{p_value:.3f}"


# =============================================================================
# Plot helpers
# =============================================================================

def get_histogram_y_label() -> str:
    """Return the histogram y-axis label."""

    if plot_probability_density:
        return "Probability density"

    return "Bootstrap samples"


def calculate_bin_edges(
    result: dict[str, object],
) -> np.ndarray:
    """Create common bins for raw, bias-corrected, and reference values."""

    combined = np.concatenate(
        [
            np.asarray(
                result[
                    "raw_bootstrap_values"
                ]
            ),
            np.asarray(
                result[
                    "bc_bootstrap_values"
                ]
            ),
            np.asarray(
                [
                    result[
                        "reference_value"
                    ]
                ]
            ),
        ]
    )

    x_min = float(
        np.min(
            combined
        )
    )

    x_max = float(
        np.max(
            combined
        )
    )

    if np.isclose(
        x_min,
        x_max,
    ):
        padding = max(
            abs(
                x_min
            )
            * 0.05,
            0.5,
        )
    else:
        padding = (
            0.03
            * (
                x_max
                - x_min
            )
        )

    return np.linspace(
        x_min - padding,
        x_max + padding,
        number_of_bins + 1,
    )


def format_axis(ax: plt.Axes) -> None:
    """Apply consistent, light formatting to one panel."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABELSIZE,
        direction="out",
    )


def plot_independence_panel(
    ax: plt.Axes,
    correlations: np.ndarray,
) -> None:
    """Plot the selected-month boxplot from script 1."""

    boxplot = ax.boxplot(
        [correlations],
        positions=[1],
        widths=0.55,
        patch_artist=False,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": "black",
            "linewidth": 1.4,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "0.6",
            "markersize": 3.5,
            "linestyle": "none",
        },
    )

    for key in ("boxes", "whiskers", "caps"):
        for artist in boxplot[key]:
            artist.set_linewidth(1.0)

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.9,
        zorder=0,
    )

    ax.set_xticks([1])
    ax.set_xticklabels(
        [MONTH_LABELS[selected_month]],
        fontsize=TICK_LABELSIZE,
    )

    ax.set_ylabel(
        "Spearman rank correlation",
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_title(
        "Independence",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    format_axis(ax)


def add_failure_text(
    ax: plt.Axes,
    result: dict[str, object],
) -> None:
    """Mark raw and bias-corrected fidelity failures.

    A model fails when the selected reference statistic lies outside that
    model's bootstrap confidence interval. Failure labels are drawn in the
    selected reference-dataset color (ERA5 blue or SeNorge red).
    """

    reference_color = (
        ERA5_COLOR
        if REFERENCE_DATASET == "era5"
        else SENORGE_COLOR
    )

    if not result["raw_passes"]:
        ax.text(
            0.97,
            0.96,
            "raw\nfail",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=LEGEND_FONTSIZE,
            color=reference_color,
        )

    if (
        BIAS_CORRECTION_METHOD != "raw"
        and not result["bc_passes"]
    ):
        ax.text(
            0.97,
            0.8,
            (
                f"BC\nfail"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=LEGEND_FONTSIZE,
            color=reference_color,
        )


def plot_moment_panel(
    ax: plt.Axes,
    statistic_name: str,
    result: dict[str, object],
    reference_label: str,
) -> None:
    """
    Plot raw and bias-corrected bootstrap distributions.

    Semi-transparent filled histograms use common bins so their overlap forms
    a visible mixture of the two colors, following the visual idea used in
    Kelder et al. (2020), Fig. 4.
    """

    bin_edges = calculate_bin_edges(
        result
    )

    raw_counts, _, _ = ax.hist(
        result[
            "raw_bootstrap_values"
        ],
        bins=bin_edges,
        density=plot_probability_density,
        histtype="stepfilled",
        color=RAW_MODEL_COLOR,
        edgecolor=RAW_MODEL_COLOR,
        alpha=BOOTSTRAP_ALPHA,
        linewidth=HISTOGRAM_LINEWIDTH,
        zorder=1,
    )

    bc_counts = np.array([])

    if BIAS_CORRECTION_METHOD != "raw":
        bc_counts, _, _ = ax.hist(
            result[
                "bc_bootstrap_values"
            ],
            bins=bin_edges,
            density=plot_probability_density,
            histtype="stepfilled",
            color=BIAS_CORRECTED_COLOR,
            edgecolor=BIAS_CORRECTED_COLOR,
            alpha=BOOTSTRAP_ALPHA,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=2,
        )

    for confidence_limit in result[
        "raw_confidence_interval"
    ]:
        ax.axvline(
            confidence_limit,
            color=RAW_MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            zorder=3,
        )

    if BIAS_CORRECTION_METHOD != "raw":
        for confidence_limit in result[
            "bc_confidence_interval"
        ]:
            ax.axvline(
                confidence_limit,
                color=BIAS_CORRECTED_COLOR,
                linewidth=CONFIDENCE_LINEWIDTH,
                linestyle="--",
                zorder=4,
            )

    reference_color = (
        ERA5_COLOR
        if REFERENCE_DATASET
        == "era5"
        else SENORGE_COLOR
    )

    ax.axvline(
        result[
            "reference_value"
        ],
        color=reference_color,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=5,
    )

    ax.set_xlim(
        bin_edges[0],
        bin_edges[-1],
    )

    maximum_count = max(
        float(
            np.max(
                raw_counts
            )
        )
        if raw_counts.size
        else 0.0,
        float(
            np.max(
                bc_counts
            )
        )
        if bc_counts.size
        else 0.0,
    )

    if maximum_count > 0:
        ax.set_ylim(
            0,
            maximum_count
            * (
                1.0
                + y_axis_margin_fraction
            ),
        )

    ax.set_xlabel(
        STATISTIC_AXIS_LABELS[
            statistic_name
        ],
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_ylabel(
        get_histogram_y_label(),
        fontsize=AXIS_LABELSIZE,
    )

    panel_title = (
        "Fidelity: Kurtosis"
        if statistic_name
        == "kurtosis"
        else (
            f"Fidelity: "
            f"{STATISTIC_LABELS[statistic_name]}"
        )
    )

    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    format_axis(
        ax
    )

    add_failure_text(
        ax,
        result,
    )


def make_shared_legend_handles(reference_label: str) -> list:
    """Create legend handles arranged as model, interval, and reference columns."""

    reference_color = ERA5_COLOR if REFERENCE_DATASET == "era5" else SENORGE_COLOR
    raw_model = Patch(
        facecolor=RAW_MODEL_COLOR,
        edgecolor=RAW_MODEL_COLOR,
        alpha=BOOTSTRAP_ALPHA,
        label="Model raw",
    )
    raw_interval = Line2D(
        [0],
        [0],
        color=RAW_MODEL_COLOR,
        linewidth=CONFIDENCE_LINEWIDTH,
        linestyle="--",
        label=f"Model raw {confidence_level_percent:g}% interval",
    )
    reference = Line2D(
        [0],
        [0],
        color=reference_color,
        linewidth=REFERENCE_LINEWIDTH,
        label=reference_label,
    )

    if BIAS_CORRECTION_METHOD == "raw":
        return [raw_model, raw_interval, reference]

    bc_model = Patch(
        facecolor=BIAS_CORRECTED_COLOR,
        edgecolor=BIAS_CORRECTED_COLOR,
        alpha=BOOTSTRAP_ALPHA,
        label=f"Model BC {BIAS_CORRECTION_METHOD}",
    )
    bc_interval = Line2D(
        [0],
        [0],
        color=BIAS_CORRECTED_COLOR,
        linewidth=CONFIDENCE_LINEWIDTH,
        linestyle="--",
        label=f"Model BC {BIAS_CORRECTION_METHOD} {confidence_level_percent:g}% interval",
    )
    #blank = Line2D([], [], linestyle="none", marker="", label="")

    # Matplotlib fills legends by columns for ncol=3, so this ordering gives:
    # column 1: model raw / model BC
    # column 2: raw interval / BC interval
    # column 3: reference / blank
    #return [raw_model, bc_model, raw_interval, bc_interval, reference, blank]
    return [raw_model, bc_model, raw_interval, bc_interval, reference]

def calculate_distribution_bin_edges(*samples: np.ndarray) -> np.ndarray:
    """Create common bins for the model and reference distributions."""

    combined = np.concatenate(samples)
    x_min = float(np.min(combined))
    x_max = float(np.max(combined))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
        x_min -= padding
        x_max += padding

    return np.linspace(x_min, x_max, number_of_bins + 1)


def plot_distribution_fidelity_panel(
    ax: plt.Axes,
    raw_model_values: np.ndarray,
    bias_corrected_model_values: np.ndarray,
    reference_values: np.ndarray,
    reference_label: str,
    raw_ks: dict[str, object],
    bc_ks: dict[str, object],
) -> None:
    """Plot selected-month model/reference distributions and KS-test results."""

    reference_color = ERA5_COLOR if REFERENCE_DATASET == "era5" else SENORGE_COLOR
    samples = [raw_model_values, reference_values]
    if BIAS_CORRECTION_METHOD != "raw":
        samples.append(bias_corrected_model_values)

    bin_edges = calculate_distribution_bin_edges(*samples)
    maximum_density = 0.0

    distributions = [
        (raw_model_values, RAW_MODEL_COLOR, "Model raw", 1),
    ]
    if BIAS_CORRECTION_METHOD != "raw":
        distributions.append(
            (
                bias_corrected_model_values,
                BIAS_CORRECTED_COLOR,
                f"Model BC {BIAS_CORRECTION_METHOD}",
                2,
            )
        )
    distributions.append((reference_values, reference_color, reference_label, 3))

    for values, color, label, zorder in distributions:
        is_reference = label == reference_label
        density, _, _ = ax.hist(
            values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step" if is_reference else "stepfilled",
            color=color,
            edgecolor=color,
            alpha=1.0 if is_reference else BOOTSTRAP_ALPHA,
            linewidth=HISTOGRAM_LINEWIDTH,
            label=label,
            zorder=zorder,
        )
        if density.size:
            maximum_density = max(maximum_density, float(np.nanmax(density)))

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    if maximum_density > 0:
        ax.set_ylim(0, maximum_density * (1.0 + y_axis_margin_fraction))

    ax.set_xlabel(
        f"Maximum monthly {x_days}-day precipitation [mm]",
        fontsize=AXIS_LABELSIZE,
    )
    ax.set_ylabel(
        "Probability density" if plot_probability_density else "Samples",
        fontsize=AXIS_LABELSIZE,
    )
    ax.set_title(
        "Fidelity: KS-test",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    threshold = get_ks_significance_threshold()
    raw_color = reference_color if raw_ks["p_value"] < threshold else "black"
    bc_color = reference_color if bc_ks["p_value"] < threshold else "black"

    ax.text(
        0.97,
        0.95,
        (
            f"raw: D={raw_ks['statistic']:.3f}, "
            f"p={format_ks_p_value(raw_ks['p_value'])}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=raw_color,
    )

    if BIAS_CORRECTION_METHOD != "raw":
        ax.text(
            0.97,
            0.87,
            (
                f"BC: D={bc_ks['statistic']:.3f}, "
                f"p={format_ks_p_value(bc_ks['p_value'])}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=bc_color,
        )

    format_axis(ax)


def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    """Place a publication-style panel label in the upper-left corner."""

    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        clip_on=False,
    )


def build_figure_title() -> str:
    """Create the figure title for the selected catchment and month."""

    catchment_name = readable_catchment_name(catchment)
    month_name = MONTH_LABELS[selected_month]

    return (
        f"{month_name}: {x_days}-day accumulated precipitation maxima\n"
        f"{catchment_name} catchment"
    )


def create_combined_figure(
    independence_values: np.ndarray,
    moments_results: dict[str, dict[str, object]],
    raw_model_values: np.ndarray,
    bias_corrected_model_values: np.ndarray,
    reference_values: np.ndarray,
    reference_label: str,
    raw_ks: dict[str, object],
    bc_ks: dict[str, object],
) -> plt.Figure:
    """Create the publication-style 2 x 3 diagnostic figure."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    plot_independence_panel(
        ax=axes[0, 0],
        correlations=independence_values,
    )
    add_panel_label(axes[0, 0], "(a)")

    panel_locations = {
        "mean": (0, 1),
        "std": (0, 2),
        "skewness": (1, 0),
        "kurtosis": (1, 1),
    }
    panel_labels = {
        "mean": "(b)",
        "std": "(c)",
        "skewness": "(d)",
        "kurtosis": "(e)",
    }

    for statistic_name in STATISTICS:
        row, column = panel_locations[statistic_name]
        plot_moment_panel(
            ax=axes[row, column],
            statistic_name=statistic_name,
            result=moments_results[statistic_name],
            reference_label=reference_label,
        )
        add_panel_label(axes[row, column], panel_labels[statistic_name])

    plot_distribution_fidelity_panel(
        ax=axes[1, 2],
        raw_model_values=raw_model_values,
        bias_corrected_model_values=bias_corrected_model_values,
        reference_values=reference_values,
        reference_label=reference_label,
        raw_ks=raw_ks,
        bc_ks=bc_ks,
    )
    add_panel_label(axes[1, 2], "(f)")

    fig.legend(
        handles=make_shared_legend_handles(reference_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.09,
        top=0.84,
        wspace=0.32,
        hspace=0.38,
    )
    return fig


# =============================================================================
# Terminal output
# =============================================================================

def format_statistic_value(
    statistic_name: str,
    value: float,
) -> str:
    """Format values compactly for terminal output."""

    if statistic_name in {"mean", "std"}:
        return f"{value:.1f}"

    return f"{value:.2f}"


def print_moments_results(
    results: dict[str, dict[str, object]],
    reference_label: str,
) -> None:
    """Print raw and bias-corrected fidelity results."""

    print()
    print(
        f"{MONTH_LABELS[selected_month]} moments-test results"
    )
    print(
        "-" * 70
    )

    for statistic_name in STATISTICS:

        result = results[
            statistic_name
        ]

        raw_lower, raw_upper = result[
            "raw_confidence_interval"
        ]

        bc_lower, bc_upper = result[
            "bc_confidence_interval"
        ]

        raw_marker = (
            ""
            if result[
                "raw_passes"
            ]
            else "*"
        )

        bc_marker = (
            ""
            if result[
                "bc_passes"
            ]
            else "*"
        )

        print(
            f"{STATISTIC_LABELS[statistic_name]:>18s} | "
            f"n={result['sample_size']:>3d} | "
            f"raw=["
            f"{format_statistic_value(statistic_name, raw_lower)}, "
            f"{format_statistic_value(statistic_name, raw_upper)}]"
            f"{raw_marker} | "
            f"BC=["
            f"{format_statistic_value(statistic_name, bc_lower)}, "
            f"{format_statistic_value(statistic_name, bc_upper)}]"
            f"{bc_marker} | "
            f"{reference_label}="
            f"{format_statistic_value(statistic_name, result['reference_value'])}"
        )

    print(
        "* reference value outside the corresponding central model "
        "bootstrap interval"
    )


def print_distribution_ks_results(
    raw_model_values: np.ndarray,
    bias_corrected_model_values: np.ndarray,
    reference_values: np.ndarray,
    reference_label: str,
    raw_ks: dict[str, object],
    bc_ks: dict[str, object],
) -> None:
    """Print two-sided model-versus-reference KS-test results."""

    print()
    print(f"{MONTH_LABELS[selected_month]} distribution fidelity KS test")
    print("-" * 55)
    print(
        f"Samples: raw={raw_model_values.size}, "
        f"BC={bias_corrected_model_values.size}, "
        f"{reference_label}={reference_values.size}"
    )
    print(f"Raw vs {reference_label}: D={raw_ks['statistic']:.3f}, p={raw_ks['p_value']:.4g}")

    if BIAS_CORRECTION_METHOD != "raw":
        print(
            f"BC vs {reference_label}:  "
            f"D={bc_ks['statistic']:.3f}, p={bc_ks['p_value']:.4g}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    (
        raw_model_filename,
        bc_model_filename,
    ) = resolve_model_input_filenames()

    (
        reference_filename,
        reference_variable,
        reference_label,
    ) = get_reference_configuration()

    output_filename = (
        build_output_filename()
    )

    print(
        "Selected month"
    )
    print(
        "--------------"
    )
    print(
        MONTH_LABELS[
            selected_month
        ]
    )

    print()
    print(
        "Input files"
    )
    print(
        "-----------"
    )

    print(
        f"Raw S2S model:          "
        f"{raw_model_filename}"
    )

    print(
        f"Selected S2S ({BIAS_CORRECTION_METHOD}): "
        f"{bc_model_filename}"
    )

    print(
        f"{reference_label}:".ljust(
            24
        ),
        reference_filename,
    )

    print()
    print(
        f"Reference dataset: "
        f"{reference_label}"
    )

    all_variable, _, _ = get_stability_variable_names()

    print()
    print("Model variable")
    print("--------------")
    print(f"All leads: {all_variable}")

    (
        independence_values,
        raw_all_values,
        bc_all_values,
        reference_values,
    ) = load_model_and_reference_values(
        raw_model_filename=raw_model_filename,
        selected_model_filename=bc_model_filename,
        reference_filename=reference_filename,
        reference_variable=reference_variable,
        reference_label=reference_label,
    )

    validate_moments_samples(
        raw_model_values=raw_all_values,
        bias_corrected_model_values=bc_all_values,
        reference_values=reference_values,
        reference_label=reference_label,
    )

    rng = np.random.default_rng(
        random_seed
    )

    moments_results = perform_all_moments_tests(
        raw_model_values=raw_all_values,
        bias_corrected_model_values=bc_all_values,
        reference_values=reference_values,
        rng=rng,
    )

    raw_fidelity_ks = perform_fidelity_ks_test(
        model_values=raw_all_values,
        reference_values=reference_values,
    )
    bc_fidelity_ks = perform_fidelity_ks_test(
        model_values=bc_all_values,
        reference_values=reference_values,
    )

    print()
    print(
        f"Independence pairs: "
        f"{independence_values.size} "
        f"finite pooled correlations "
        f"(raw all-lead sample)"
    )

    print_moments_results(
        results=moments_results,
        reference_label=reference_label,
    )

    print_distribution_ks_results(
        raw_model_values=raw_all_values,
        bias_corrected_model_values=bc_all_values,
        reference_values=reference_values,
        reference_label=reference_label,
        raw_ks=raw_fidelity_ks,
        bc_ks=bc_fidelity_ks,
    )

    figure = create_combined_figure(
        independence_values=independence_values,
        moments_results=moments_results,
        raw_model_values=raw_all_values,
        bias_corrected_model_values=bc_all_values,
        reference_values=reference_values,
        reference_label=reference_label,
        raw_ks=raw_fidelity_ks,
        bc_ks=bc_fidelity_ks,
    )

    if write2file:

        output_directory = os.path.dirname(
            output_filename
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        figure.savefig(
            output_filename,
            dpi=figure_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print()
        print(
            f"Wrote figure: "
            f"{output_filename}"
        )

    if show_figure:
        plt.show()
    else:
        plt.close(
            figure
        )
