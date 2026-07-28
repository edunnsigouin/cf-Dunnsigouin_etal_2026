"""
Create a six-panel UNSEEN diagnostic figure for one selected calendar month.

The figure combines three checks:
    1. ensemble-member independence;
    2. fidelity of raw and bias-corrected S2S distributions relative to one
       selected reference dataset;
    3. stability of raw and bias-corrected S2S distributions across lead-time
       subgroups.

Inputs
------
The script reads three NetCDF files:
    - the raw S2S monthly extreme-sample file;
    - the bias-corrected S2S monthly extreme-sample file;
    - ONE selected reference dataset: ERA5 or SeNorge.

Panel (a) calculates independence from the raw all-lead sample.
Panels (b)-(e) overlay raw and bias-corrected bootstrap distributions and their
95% intervals, with one vertical line for the selected reference dataset.
Panel (f) plots the bias-corrected Early/Late distributions and reports both
raw and bias-corrected KS statistics.

For the default settings
    first_input_lead = 16
    last_input_lead = 46
    x_days = 2
    number_of_lead_bins = 2

the usable accumulated ending leads are 17-46 and the S2S variables are:
    all leads : max_value_lead17_46
    early     : max_value_lead17_31
    late      : max_value_lead32_46


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

This tests whether the observed spread of monthly extremes is consistent with
the spread expected from the S2S distribution.


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


Panel (f): Lead-time stability
------------------------------
Compares the complete all-lead S2S distribution with the lead-location
subgroups.

The subgroup values are not maxima recalculated over shorter lead windows.
They are the SAME full-window maxima, classified by the lead time at which each
maximum occurred.

For the default two-bin setup:
    early = maxima occurring at ending leads 17-31
    late  = maxima occurring at ending leads 32-46

The panel shows probability-density distributions for all leads, early leads,
and late leads. A two-sample Kolmogorov-Smirnov test compares the early and
late subgroups and reports sample counts, KS statistic, p-value, and whether
the equal-distribution null hypothesis is rejected.


Data used by each panel
-----------------------
Panel (a):
    complete all-lead S2S sample + forecast_date + hdate + ensemble_member
    + model_type from the shared S2S extreme-sample file.

Panels (b)-(e):
    complete all-lead S2S sample + ERA5 + SeNorge.

Panel (f):
    complete, early, and late samples from the same S2S extreme-sample file.
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
selected_month = 1

# Accumulation period.
x_days = 2

# Catchment used consistently for all input datasets and figure labels.
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

# Lead-location sampling used by panels (b)-(f).
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Reference dataset used for BOTH:
#   1. the vertical reference line in panels (b)-(e); and
#   2. selecting the corresponding bias-corrected model file.
#
# Options:
#     "era5"
#     "senorge"
REFERENCE_DATASET = "senorge"

# Independence-test settings.
# Minimum number of paired initialization values required for one
# ensemble-member Spearman correlation.
minimum_samples = 10

# Expected ensemble sizes.
n_forecast_members = 51
n_hindcast_members = 11

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

write2file = True
show_figure = True


# =============================================================================
# Dataset configuration
# =============================================================================

MODEL_VARIABLE = "tp24"
# S2S maximum-variable names are built automatically from lead ranges.
MODEL_MONTH_COORDINATE = "month_of_year"

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

STATISTICS = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
)

STATISTIC_LABELS = {
    "mean": "Mean",
    "std": "Standard deviation",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
}

STATISTIC_AXIS_LABELS = {
    "mean": "Mean precipitation [mm]",
    "std": "Precipitation standard deviation [mm]",
    "skewness": "Precipitation skewness",
    "kurtosis": "Precipitation excess kurtosis",
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

    for prefix in (
        "nve_catchment_regine_",
        "nve_catchment_",
        "regine_",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    return name.replace("_", " ").title()


def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """Flatten an array and retain only finite values."""

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def validate_user_settings() -> None:
    """Validate settings that affect both analyses."""

    if selected_month not in MONTH_LABELS:
        raise ValueError("selected_month must be an integer from 1 to 12.")

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


def get_stability_variable_names(
    bias_corrected: bool,
) -> tuple[str, str, str]:
    """Return complete, early, and late model-variable names."""

    full_range, early_range, late_range = get_stability_lead_ranges()

    all_variable = f"max_value_lead{full_range[0]}_{full_range[1]}"
    early_variable = f"max_value_lead{early_range[0]}_{early_range[1]}"
    late_variable = f"max_value_lead{late_range[0]}_{late_range[1]}"

    if bias_corrected:
        suffix = f"_bc_{REFERENCE_DATASET}"
        all_variable += suffix
        early_variable += suffix
        late_variable += suffix

    return all_variable, early_variable, late_variable


def build_model_filename(
    bias_corrected: bool,
) -> str:
    """Build the raw or bias-corrected S2S sample filename."""

    full_range, early_range, late_range = get_stability_lead_ranges()

    lead_label = (
        f"lead{full_range[0]}-{full_range[1]}_"
        f"split{number_of_lead_bins}_"
        f"{early_range[0]}-{early_range[1]}_"
        f"{late_range[0]}-{late_range[1]}"
    )

    filename = os.path.join(
        config.dirs["s2s_processed"],
        (
            f"unseen_sample_monthly_catchment_precipitation_extremes_"
            f"{MODEL_VARIABLE}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_label}_"
            f"forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )

    if bias_corrected:
        stem, extension = os.path.splitext(filename)
        filename = (
            f"{stem}_bc_{REFERENCE_DATASET}{extension}"
        )

    return filename


def resolve_model_input_filenames() -> tuple[str, str]:
    """Construct raw and bias-corrected S2S sample filenames."""

    raw_filename = build_model_filename(
        bias_corrected=False,
    )

    bias_corrected_filename = build_model_filename(
        bias_corrected=True,
    )

    return raw_filename, bias_corrected_filename


def build_era5_filename() -> str:
    """Build the ERA5 filename exactly as in script 2."""

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def build_senorge_filename() -> str:
    """Build the SeNorge monthly-extremes filename."""

    return (
        f"{config.dirs['senorge_processed']}"
        f"distribution_monthly_extremes_{SENORGE_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_senorge_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
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
            f"UNSEEN_independence_fidelity_stability_tests_"
            f"{month_name}_{x_days}dayacc_{catchment}_"
            f"{forecast_date_range[0]}_{forecast_date_range[1]}_"
            f"raw_bc_{REFERENCE_DATASET}.png"
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
    """
    Extract the selected-month all-lead samples needed for independence.

    Forecast initialization key:
        forecast_date

    Hindcast initialization key:
        (forecast_date, hdate)
    """

    required_variables = {
        all_variable,
        "forecast_date",
        "hdate",
        "ensemble_member",
        "model_type",
    }

    missing = required_variables.difference(
        model_ds.data_vars
    )

    if missing:
        raise KeyError(
            "Model dataset is missing variables needed for the "
            f"independence calculation: {sorted(missing)}"
        )

    values = (
        model_ds[all_variable]
        .sel(
            month_of_year=selected_month
        )
        .values
        .ravel()
        .astype("float64")
    )

    model_types = normalize_model_type(
        model_ds["model_type"]
        .sel(
            month_of_year=selected_month
        )
        .values
    )

    members = (
        model_ds["ensemble_member"]
        .sel(
            month_of_year=selected_month
        )
        .values
        .ravel()
        .astype("int64")
    )

    forecast_date_keys = datetime_values_to_key(
        model_ds["forecast_date"]
        .sel(
            month_of_year=selected_month
        )
        .values
    )

    if model_type == "forecast":

        initialization_keys = forecast_date_keys

        valid_initialization = (
            ~np.isnat(
                forecast_date_keys
            )
        )

    elif model_type == "hindcast":

        hdate_keys = hdate_values_to_key(
            model_ds["hdate"]
            .sel(
                month_of_year=selected_month
            )
            .values
        )

        initialization_keys = np.empty(
            hdate_keys.size,
            dtype=object,
        )

        for index in range(
            hdate_keys.size
        ):
            initialization_keys[index] = (
                forecast_date_keys[index],
                int(
                    hdate_keys[index]
                ),
            )

        valid_initialization = (
            ~np.isnat(
                forecast_date_keys
            )
            & (
                hdate_keys
                != -99999999
            )
        )

    else:
        raise ValueError(
            "model_type must be 'forecast' or 'hindcast'."
        )

    valid = (
        np.isfinite(
            values
        )
        & (
            model_types
            == model_type
        )
        & valid_initialization
        & (
            members
            >= 0
        )
    )

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

    unique_members = np.unique(
        member_labels
    )

    expected_members = (
        n_forecast_members
        if model_type == "forecast"
        else n_hindcast_members
    )

    if unique_members.size != expected_members:
        raise ValueError(
            f"{model_type.capitalize()} data contain "
            f"{unique_members.size} unique ensemble-member labels, "
            f"but {expected_members} were expected. "
            f"Found labels: {unique_members.tolist()}"
        )

    member_lookup = {
        member: index
        for index, member
        in enumerate(
            unique_members
        )
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
    """Extract one model lead-group sample for the selected month."""

    check_variable_exists(
        model_ds,
        variable_name,
        "model dataset",
    )

    data = model_ds[variable_name]

    check_coordinate_exists(
        data,
        MODEL_MONTH_COORDINATE,
        "model dataset",
    )

    values = data.sel(
        {MODEL_MONTH_COORDINATE: selected_month}
    ).values

    return remove_missing_values(values)


def get_reference_values_for_selected_month(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> np.ndarray:
    """Extract ERA5 or SeNorge monthly maxima for the selected month."""

    check_variable_exists(ds, variable, dataset_name)

    data = ds[variable]

    check_coordinate_exists(
        data,
        "month",
        dataset_name,
    )

    values = data.sel(month=selected_month).values
    return remove_missing_values(values)


def load_model_and_reference_values(
    raw_model_filename: str,
    bias_corrected_model_filename: str,
    reference_filename: str,
    reference_variable: str,
    reference_label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load all samples needed by the six panels.

    Panel (a):
        independence is calculated from the RAW all-lead model sample.

    Panels (b)-(e):
        raw all-lead, bias-corrected all-lead, and one reference sample.

    Panel (f):
        raw Early/Late are used for the raw KS test;
        bias-corrected Early/Late are used for the BC KS test and plotted.
    """

    for dataset_name, filename in (
        ("raw model", raw_model_filename),
        ("bias-corrected model", bias_corrected_model_filename),
        (reference_label, reference_filename),
    ):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{dataset_name} input file does not exist:\\n{filename}"
            )

    (
        raw_all_variable,
        raw_early_variable,
        raw_late_variable,
    ) = get_stability_variable_names(
        bias_corrected=False,
    )

    (
        bc_all_variable,
        bc_early_variable,
        bc_late_variable,
    ) = get_stability_variable_names(
        bias_corrected=True,
    )

    with (
        xr.open_dataset(raw_model_filename) as raw_model_ds,
        xr.open_dataset(bias_corrected_model_filename) as bc_model_ds,
        xr.open_dataset(reference_filename) as reference_ds,
    ):
        independence_values = calculate_independence_values(
            model_ds=raw_model_ds,
            all_variable=raw_all_variable,
        )

        raw_all_values = get_model_values_for_selected_month(
            raw_model_ds,
            raw_all_variable,
        )

        raw_early_values = get_model_values_for_selected_month(
            raw_model_ds,
            raw_early_variable,
        )

        raw_late_values = get_model_values_for_selected_month(
            raw_model_ds,
            raw_late_variable,
        )

        bc_all_values = get_model_values_for_selected_month(
            bc_model_ds,
            bc_all_variable,
        )

        bc_early_values = get_model_values_for_selected_month(
            bc_model_ds,
            bc_early_variable,
        )

        bc_late_values = get_model_values_for_selected_month(
            bc_model_ds,
            bc_late_variable,
        )

        reference_values = get_reference_values_for_selected_month(
            reference_ds,
            reference_variable,
            reference_label,
        )

    return (
        independence_values,
        raw_all_values,
        raw_early_values,
        raw_late_values,
        bc_all_values,
        bc_early_values,
        bc_late_values,
        reference_values,
    )


def validate_model_partition(
    model_all_values: np.ndarray,
    model_early_values: np.ndarray,
    model_late_values: np.ndarray,
) -> None:
    """Check that Early + Late partition the complete selected-month sample."""

    if model_all_values.size != (
        model_early_values.size + model_late_values.size
    ):
        raise ValueError(
            "Early + Late sample counts do not equal the all-lead sample "
            "for the selected month."
        )


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
# Stability KS test
# =============================================================================

def get_ks_significance_threshold() -> float:
    """Convert the selected KS confidence level to a p-value threshold."""

    return 1.0 - ks_significance_level_percent / 100.0


def perform_stability_ks_test(
    early_values: np.ndarray,
    late_values: np.ndarray,
) -> dict[str, object]:
    """
    Compare Early and Late model subgroups with a two-sided two-sample KS test.

    Null hypothesis:
        Early and Late samples come from the same continuous distribution.
    """

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
        "reject_null": p_value < get_ks_significance_threshold(),
    }


def format_ks_p_value(p_value: float) -> str:
    """Format a KS p-value compactly."""

    if p_value < 0.001:
        return f"{p_value:.1e}"

    return f"{p_value:.3f}"


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
    """Annotate only failed raw/BC fidelity checks."""

    failure_lines = []

    if not result["raw_passes"]:
        failure_lines.append(
            (
                "Raw fail",
                RAW_MODEL_COLOR,
            )
        )

    if not result["bc_passes"]:
        failure_lines.append(
            (
                "BC fail",
                BIAS_CORRECTED_COLOR,
            )
        )

    for index, (
        label,
        color,
    ) in enumerate(
        failure_lines
    ):
        ax.text(
            0.97,
            0.96
            - index
            * 0.08,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LEGEND_FONTSIZE,
            color=color,
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

    #add_failure_text(
    #    ax,
    #    result,
    #)


def make_shared_legend_handles(
    reference_label: str,
) -> list:
    """Create the shared legend used by the six-panel figure."""

    reference_color = (
        ERA5_COLOR
        if REFERENCE_DATASET
        == "era5"
        else SENORGE_COLOR
    )

    return [
        Patch(
            facecolor=RAW_MODEL_COLOR,
            edgecolor=RAW_MODEL_COLOR,
            alpha=BOOTSTRAP_ALPHA,
            label="Model raw",
        ),
        Patch(
            facecolor=BIAS_CORRECTED_COLOR,
            edgecolor=BIAS_CORRECTED_COLOR,
            alpha=BOOTSTRAP_ALPHA,
            label="Model BC",
        ),
        Line2D(
            [0],
            [0],
            color=RAW_MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            label=(
                f"Model raw "
                f"{confidence_level_percent:g}% interval"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=BIAS_CORRECTED_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            label=(
                f"Model BC "
                f"{confidence_level_percent:g}% interval"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=EARLY_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model BC early lead days (17-31)",
        ),
        Line2D(
            [0],
            [0],
            color=LATE_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model BC late lead days (32-46)",
        ),
        Line2D(
            [0],
            [0],
            color=reference_color,
            linewidth=REFERENCE_LINEWIDTH,
            label=reference_label,
        ),

    ]


def calculate_stability_bin_edges(
    all_values: np.ndarray,
    early_values: np.ndarray,
    late_values: np.ndarray,
) -> np.ndarray:
    """Create common precipitation bins for all, Early, and Late samples."""

    combined = np.concatenate(
        [all_values, early_values, late_values]
    )

    x_min = float(np.min(combined))
    x_max = float(np.max(combined))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
        x_min -= padding
        x_max += padding

    return np.linspace(
        x_min,
        x_max,
        number_of_bins + 1,
    )


def plot_stability_panel(
    ax: plt.Axes,
    bc_early_values: np.ndarray,
    bc_late_values: np.ndarray,
    raw_stability_ks: dict[str, object],
    bc_stability_ks: dict[str, object],
) -> None:
    """
    Plot only bias-corrected Early/Late distributions.

    The annotation reports sample sizes and both the raw and bias-corrected
    KS statistics, avoiding four overlaid probability-density curves.
    """

    bin_edges = calculate_stability_bin_edges(
        bc_early_values,
        bc_early_values,
        bc_late_values,
    )

    maximum_density = 0.0

    for values, color, zorder in (
        (
            bc_early_values,
            EARLY_COLOR,
            2,
        ),
        (
            bc_late_values,
            LATE_COLOR,
            1,
        ),
    ):
        density, _, _ = ax.hist(
            values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=color,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=zorder,
        )

        if density.size > 0:
            maximum_density = max(
                maximum_density,
                float(
                    np.nanmax(
                        density
                    )
                ),
            )

    ax.set_xlim(
        bin_edges[0],
        bin_edges[-1],
    )

    if maximum_density > 0:
        ax.set_ylim(
            0,
            maximum_density
            * (
                1.0
                + y_axis_margin_fraction
            ),
        )

    ax.set_xlabel(
        "Precipitation [mm]",
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_ylabel(
        get_histogram_y_label(),
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_title(
        "Stability",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    annotation = (
        f"Early n={bc_early_values.size}\n"
        f"Late n={bc_late_values.size}\n"
        f"Model raw: D={raw_stability_ks['statistic']:.3f}, "
        f"p={format_ks_p_value(raw_stability_ks['p_value'])}\n"
        f"Model BC:   D={bc_stability_ks['statistic']:.3f}, "
        f"p={format_ks_p_value(bc_stability_ks['p_value'])}"
    )

    ax.text(
        0.97,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    format_axis(
        ax
    )


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
    bc_early_values: np.ndarray,
    bc_late_values: np.ndarray,
    reference_label: str,
    raw_stability_ks: dict[str, object],
    bc_stability_ks: dict[str, object],
) -> plt.Figure:
    """Create the publication-style 2 x 3 diagnostic figure."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(
            figure_width,
            figure_height,
        ),
        squeeze=False,
    )

    # (a) Ensemble-member independence from raw all-lead data.
    plot_independence_panel(
        ax=axes[0, 0],
        correlations=independence_values,
    )

    add_panel_label(
        axes[0, 0],
        "(a)",
    )

    panel_locations = {
        "mean": (
            0,
            1,
        ),
        "std": (
            0,
            2,
        ),
        "skewness": (
            1,
            0,
        ),
        "kurtosis": (
            1,
            1,
        ),
    }

    panel_labels = {
        "mean": "(b)",
        "std": "(c)",
        "skewness": "(d)",
        "kurtosis": "(e)",
    }

    for statistic_name in STATISTICS:

        row, column = panel_locations[
            statistic_name
        ]

        plot_moment_panel(
            ax=axes[
                row,
                column,
            ],
            statistic_name=statistic_name,
            result=moments_results[
                statistic_name
            ],
            reference_label=reference_label,
        )

        add_panel_label(
            axes[
                row,
                column,
            ],
            panel_labels[
                statistic_name
            ],
        )

    # (f) Bias-corrected lead-time distributions, with raw and BC KS values.
    plot_stability_panel(
        ax=axes[1, 2],
        bc_early_values=bc_early_values,
        bc_late_values=bc_late_values,
        raw_stability_ks=raw_stability_ks,
        bc_stability_ks=bc_stability_ks,
    )

    add_panel_label(
        axes[1, 2],
        "(f)",
    )

    fig.legend(
        handles=make_shared_legend_handles(
            reference_label
        ),
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.96,
        ),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.2,
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


def print_stability_results(
    raw_early_values: np.ndarray,
    raw_late_values: np.ndarray,
    bc_early_values: np.ndarray,
    bc_late_values: np.ndarray,
    raw_stability_ks: dict[str, object],
    bc_stability_ks: dict[str, object],
) -> None:
    """Print raw and bias-corrected lead-time stability results."""

    print()
    print(
        f"{MONTH_LABELS[selected_month]} stability test"
    )
    print(
        "-" * 45
    )

    print(
        f"Early n={bc_early_values.size}, "
        f"Late n={bc_late_values.size}"
    )

    print(
        f"Raw: D={raw_stability_ks['statistic']:.3f}, "
        f"p={raw_stability_ks['p_value']:.4g}"
    )

    print(
        f"BC:  D={bc_stability_ks['statistic']:.3f}, "
        f"p={bc_stability_ks['p_value']:.4g}"
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
        f"Bias-corrected S2S:     "
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
        f"Bias-correction reference: "
        f"{reference_label}"
    )

    (
        raw_all_variable,
        raw_early_variable,
        raw_late_variable,
    ) = get_stability_variable_names(
        bias_corrected=False,
    )

    (
        bc_all_variable,
        bc_early_variable,
        bc_late_variable,
    ) = get_stability_variable_names(
        bias_corrected=True,
    )

    print()
    print(
        "Raw variables"
    )
    print(
        "-------------"
    )
    print(
        f"All leads:   "
        f"{raw_all_variable}"
    )
    print(
        f"Early leads: "
        f"{raw_early_variable}"
    )
    print(
        f"Late leads:  "
        f"{raw_late_variable}"
    )

    print()
    print(
        "Bias-corrected variables"
    )
    print(
        "------------------------"
    )
    print(
        f"All leads:   "
        f"{bc_all_variable}"
    )
    print(
        f"Early leads: "
        f"{bc_early_variable}"
    )
    print(
        f"Late leads:  "
        f"{bc_late_variable}"
    )

    (
        independence_values,
        raw_all_values,
        raw_early_values,
        raw_late_values,
        bc_all_values,
        bc_early_values,
        bc_late_values,
        reference_values,
    ) = load_model_and_reference_values(
        raw_model_filename=raw_model_filename,
        bias_corrected_model_filename=bc_model_filename,
        reference_filename=reference_filename,
        reference_variable=reference_variable,
        reference_label=reference_label,
    )

    validate_model_partition(
        model_all_values=raw_all_values,
        model_early_values=raw_early_values,
        model_late_values=raw_late_values,
    )

    validate_model_partition(
        model_all_values=bc_all_values,
        model_early_values=bc_early_values,
        model_late_values=bc_late_values,
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

    raw_stability_ks = perform_stability_ks_test(
        early_values=raw_early_values,
        late_values=raw_late_values,
    )

    bc_stability_ks = perform_stability_ks_test(
        early_values=bc_early_values,
        late_values=bc_late_values,
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

    print_stability_results(
        raw_early_values=raw_early_values,
        raw_late_values=raw_late_values,
        bc_early_values=bc_early_values,
        bc_late_values=bc_late_values,
        raw_stability_ks=raw_stability_ks,
        bc_stability_ks=bc_stability_ks,
    )

    figure = create_combined_figure(
        independence_values=independence_values,
        moments_results=moments_results,
        bc_early_values=bc_early_values,
        bc_late_values=bc_late_values,
        reference_label=reference_label,
        raw_stability_ks=raw_stability_ks,
        bc_stability_ks=bc_stability_ks,
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
