"""
Create a six-panel UNSEEN diagnostic figure for one selected calendar month.

The figure combines three checks:
    1. ensemble-member independence;
    2. fidelity of the S2S distribution relative to ERA5 and SeNorge;
    3. stability of the S2S distribution across lead-time subgroups.

Inputs
------
The script reads four NetCDF files:
    - a precomputed Spearman-correlation file for panel (a);
    - one S2S monthly extreme-sample file for panels (b)-(f), which can be
      either the original or multiplicatively bias-corrected model sample;
    - ERA5 monthly extremes;
    - SeNorge monthly extremes.

Panel (a) always uses the original precomputed independence file. The
``USE_BIAS_CORRECTED_MODEL`` and ``BIAS_CORRECTION_REFERENCE`` settings affect
only panels (b)-(f).

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
    precomputed independence file.

Panels (b)-(e):
    complete all-lead S2S sample + ERA5 + SeNorge.

Panel (f):
    complete, early, and late samples from the same S2S extreme-sample file.
"""


import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp, kurtosis, skew

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Calendar month to plot: 1=January, ..., 12=December.
selected_month = 6

# Accumulation period.
x_days = 2

# Catchment used consistently for all input datasets and figure labels.
catchment = "regine_drammen"

forecast_date_range = (
    "2020-01-02",
    "2023-06-26",
)

reference_years = (
    "1957",
    "2022",
)

era5_grid = "0.5x0.5"

# Optional full-path override for the independence file used by panel (a).
# Leave as None to construct the standard filename automatically.
independence_filename_override = None

# Lead-location sampling used by panels (b)-(f).
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Optional full-path override for the monthly extreme-sample model file.
# Leave as None to construct the filename automatically.
stability_model_filename_override = None

# Choose whether panels (b)-(f) use the original or bias-corrected S2S sample.
#
# False -> original model sample
# True  -> bias-corrected model sample created by the bias-correction script
#
# Panel (a) always continues to use the original independence file.
USE_BIAS_CORRECTED_MODEL = True

# Reference dataset used for the bias correction.
# Only used when USE_BIAS_CORRECTED_MODEL = True.
#
# Options:
#     "era5"
#     "senorge"
BIAS_CORRECTION_REFERENCE = "era5"

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

MODEL_COLOR = "black"
ERA5_COLOR = "tab:blue"
SENORGE_COLOR = "tab:red"
EARLY_COLOR = "tab:green"
LATE_COLOR = "tab:purple"

HISTOGRAM_LINEWIDTH = 1.4
REFERENCE_LINEWIDTH = 1.6
CONFIDENCE_LINEWIDTH = 1.2

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


    if USE_BIAS_CORRECTED_MODEL:
        valid_references = {
            "era5",
            "senorge",
        }

        if BIAS_CORRECTION_REFERENCE not in valid_references:
            raise ValueError(
                f"BIAS_CORRECTION_REFERENCE must be one of "
                f"{sorted(valid_references)}. "
                f"Got '{BIAS_CORRECTION_REFERENCE}'."
            )


# =============================================================================
# Filename helpers
# =============================================================================

def build_independence_filename() -> str:
    """Build the standard independence filename."""

    first_usable_lead = first_input_lead + x_days - 1

    return (
        config.dirs["s2s_processed"]
        + f"independence_spearman_monthly_max_{MODEL_VARIABLE}_"
        + f"{x_days}dayacc_"
        + f"nve_catchment_{catchment}_"
        + f"lead{first_usable_lead}-{last_input_lead}_"
        + f"{forecast_date_range[0]}_"
        + f"{forecast_date_range[1]}.nc"
    )

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
    """Return complete, early, and late variable names in the model file."""

    full_range, early_range, late_range = get_stability_lead_ranges()

    all_variable = f"max_value_lead{full_range[0]}_{full_range[1]}"
    early_variable = f"max_value_lead{early_range[0]}_{early_range[1]}"
    late_variable = f"max_value_lead{late_range[0]}_{late_range[1]}"

    if USE_BIAS_CORRECTED_MODEL:
        suffix = f"_bc_{BIAS_CORRECTION_REFERENCE}"

        all_variable += suffix
        early_variable += suffix
        late_variable += suffix

    return all_variable, early_variable, late_variable


def build_stability_model_filename() -> str:
    """
    Build the shared model filename exactly as written by the revised
    monthly extreme-sample script.

    Example for x_days=2 and number_of_lead_bins=2:

        unseen_sample_monthly_catchment_precipitation_extremes_
        tp24_2dayacc_regine_drammen_
        lead17-46_split2_17-31_32-46_
        forecast_hindcast_2020-01-02_2023-06-26.nc
    """

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

    if USE_BIAS_CORRECTED_MODEL:
        stem, extension = os.path.splitext(filename)

        filename = (
            f"{stem}_bc_"
            f"{BIAS_CORRECTION_REFERENCE}"
            f"{extension}"
        )

    return filename


def resolve_model_input_filenames() -> tuple[str, str]:
    """Return the independence file and shared S2S extreme-sample file."""

    independence_filename = (
        independence_filename_override
        if independence_filename_override is not None
        else build_independence_filename()
    )

    shared_model_filename = (
        stability_model_filename_override
        if stability_model_filename_override is not None
        else build_stability_model_filename()
    )

    return independence_filename, shared_model_filename

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

def build_output_filename() -> str:
    """Create a descriptive filename for the six-panel figure."""

    month_name = MONTH_LABELS[selected_month].lower()

    if USE_BIAS_CORRECTED_MODEL:
        model_label = (
            f"bc_{BIAS_CORRECTION_REFERENCE}"
        )
    else:
        model_label = "raw"

    return os.path.join(
        config.dirs["fig"],
        (
            f"UNSEEN_independence_fidelity_stability_tests_"
            f"{month_name}_{x_days}dayacc_{catchment}_{model_label}.png"
        ),
    )


# =============================================================================
# Independence data: script 1 logic for one month
# =============================================================================

def load_independence_values(filename: str) -> np.ndarray:
    """
    Load and pool forecast/hindcast pairwise correlations for one month.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Independence input file does not exist:\n{filename}"
        )

    with xr.open_dataset(filename) as ds:
        required = {
            "forecast_spearman_rho",
            "hindcast_spearman_rho",
        }

        missing = required.difference(ds.data_vars)

        if missing:
            raise KeyError(
                "Independence file is missing variables: "
                f"{sorted(missing)}"
            )

        if "month_of_year" not in ds.coords:
            raise KeyError(
                "Independence file has no 'month_of_year' coordinate."
            )

        forecast = remove_missing_values(
            ds["forecast_spearman_rho"]
            .sel(month_of_year=selected_month)
            .values
        )

        hindcast = remove_missing_values(
            ds["hindcast_spearman_rho"]
            .sel(month_of_year=selected_month)
            .values
        )

    combined = np.concatenate([forecast, hindcast])

    if combined.size == 0:
        raise ValueError(
            f"No finite independence values found for "
            f"{MONTH_LABELS[selected_month]}."
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
    model_filename: str,
    era5_filename: str,
    senorge_filename: str,
    senorge_variable: str,
    senorge_label: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load all-lead, Early, Late, ERA5, and SeNorge samples for one month.

    The all-lead model sample is used by panels (b)-(e). The all-lead, Early,
    and Late model samples are used by panel (f).
    """

    for dataset_name, filename in (
        ("model", model_filename),
        ("ERA5", era5_filename),
        (senorge_label, senorge_filename),
    ):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{dataset_name} input file does not exist:\n{filename}"
            )

    all_variable, early_variable, late_variable = (
        get_stability_variable_names()
    )

    with (
        xr.open_dataset(model_filename) as model_ds,
        xr.open_dataset(era5_filename) as era5_ds,
        xr.open_dataset(senorge_filename) as senorge_ds,
    ):
        model_all_values = get_model_values_for_selected_month(
            model_ds,
            all_variable,
        )

        model_early_values = get_model_values_for_selected_month(
            model_ds,
            early_variable,
        )

        model_late_values = get_model_values_for_selected_month(
            model_ds,
            late_variable,
        )

        era5_values = get_reference_values_for_selected_month(
            era5_ds,
            ERA5_VARIABLE,
            "ERA5 dataset",
        )

        senorge_values = get_reference_values_for_selected_month(
            senorge_ds,
            senorge_variable,
            f"{senorge_label} dataset",
        )

    return (
        model_all_values,
        model_early_values,
        model_late_values,
        era5_values,
        senorge_values,
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
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
    senorge_label: str,
) -> None:
    """Check the samples required by all four moments tests."""

    minimum_sample_size = 4

    for dataset_name, values in (
        ("model", model_values),
        ("ERA5", era5_values),
        (senorge_label, senorge_values),
    ):
        if values.size < minimum_sample_size:
            raise ValueError(
                f"Only {values.size} finite {dataset_name} values were found "
                f"for {MONTH_LABELS[selected_month]}. At least "
                f"{minimum_sample_size} are required."
            )

    if era5_values.size != senorge_values.size:
        raise ValueError(
            f"ERA5 and {senorge_label} have different sample sizes for "
            f"{MONTH_LABELS[selected_month]}: ERA5={era5_values.size}, "
            f"{senorge_label}={senorge_values.size}."
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
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, object]]:
    """
    Run mean, standard deviation, skewness, and kurtosis tests.

    One set of bootstrap sample indices is generated and reused for all four
    statistics. Thus every moment is calculated from the same resampled model
    samples.
    """

    sample_size = era5_values.size

    sample_indices = rng.integers(
        low=0,
        high=model_values.size,
        size=(number_of_bootstrap_samples, sample_size),
    )

    resampled_model = model_values[sample_indices]

    results = {}

    for statistic_name in STATISTICS:
        statistic_function = get_vectorized_statistic_function(
            statistic_name
        )

        bootstrap_values = remove_missing_values(
            statistic_function(resampled_model)
        )

        if bootstrap_values.size == 0:
            raise ValueError(
                f"No finite bootstrap {statistic_name} values were produced."
            )

        confidence_interval = calculate_confidence_interval(
            bootstrap_values
        )

        era5_value = calculate_statistic(
            era5_values,
            statistic_name,
        )

        senorge_value = calculate_statistic(
            senorge_values,
            statistic_name,
        )

        lower, upper = confidence_interval

        results[statistic_name] = {
            "bootstrap_values": bootstrap_values,
            "confidence_interval": confidence_interval,
            "sample_size": sample_size,
            "era5_value": era5_value,
            "senorge_value": senorge_value,
            "era5_passes": lower <= era5_value <= upper,
            "senorge_passes": lower <= senorge_value <= upper,
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
    """Create histogram bins that also include both reference values."""

    combined = np.concatenate(
        [
            np.asarray(result["bootstrap_values"]),
            np.asarray(
                [
                    result["era5_value"],
                    result["senorge_value"],
                ]
            ),
        ]
    )

    x_min = float(np.min(combined))
    x_max = float(np.max(combined))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
    else:
        padding = 0.03 * (x_max - x_min)

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
    senorge_label: str,
) -> None:
    """
    Add a simple failure label only when a reference dataset fails the test.

    No annotation is drawn when both ERA5 and SeNorge pass.
    """

    failure_lines = []

    if not result["era5_passes"]:
        failure_lines.append(("ERA5 fail", ERA5_COLOR))

    if not result["senorge_passes"]:
        failure_lines.append((f"{senorge_label} fail", SENORGE_COLOR))

    for index, (label, color) in enumerate(failure_lines):
        ax.text(
            0.97,
            0.96 - index * 0.08,
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
    senorge_label: str,
    show_legend: bool,
) -> None:
    """Plot one of the four moments bootstrap diagnostics."""

    bin_edges = calculate_bin_edges(result)

    counts, _, _ = ax.hist(
        result["bootstrap_values"],
        bins=bin_edges,
        density=plot_probability_density,
        histtype="step",
        color=MODEL_COLOR,
        linewidth=HISTOGRAM_LINEWIDTH,
        zorder=1,
    )

    lower, upper = result["confidence_interval"]

    for confidence_limit in (lower, upper):
        ax.axvline(
            confidence_limit,
            color=MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            zorder=2,
        )

    ax.axvline(
        result["era5_value"],
        color=ERA5_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=3,
    )

    ax.axvline(
        result["senorge_value"],
        color=SENORGE_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=3,
    )

    ax.set_xlim(bin_edges[0], bin_edges[-1])

    if counts.size > 0:
        ax.set_ylim(
            0,
            float(np.max(counts)) * (1.0 + y_axis_margin_fraction),
        )

    ax.set_xlabel(
        STATISTIC_AXIS_LABELS[statistic_name],
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_ylabel(
        get_histogram_y_label(),
        fontsize=AXIS_LABELSIZE,
    )

    panel_title = (
        "Fidelity: Kurtosis"
        if statistic_name == "kurtosis"
        else f"Fidelity: {STATISTIC_LABELS[statistic_name]}"
    )

    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    format_axis(ax)
    add_failure_text(ax, result, senorge_label)

    # Dataset/style information is shown once in the shared figure legend.



def calculate_distribution_bin_edges(
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
) -> np.ndarray:
    """Create common precipitation bins for all three distributions."""

    combined = np.concatenate(
        [model_values, era5_values, senorge_values]
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


def make_shared_legend_handles(
    senorge_label: str,
) -> list[Line2D]:
    """Create the shared legend used by the six-panel figure."""

    return [
        Line2D(
            [0], [0],
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model / all leads (17-46)",
        ),
        Line2D(
            [0], [0],
            color=MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            label=f"Model / {confidence_level_percent:g}% bootstrap interval",
        ),
        Line2D(
            [0], [0],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            label="ERA5",
        ),
        Line2D(
            [0], [0],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            label=senorge_label,
        ),
        Line2D(
            [0], [0],
            color=EARLY_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model / early leads (17-31)",
        ),
        Line2D(
            [0], [0],
            color=LATE_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model / late leads (32-46)",
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
    all_values: np.ndarray,
    early_values: np.ndarray,
    late_values: np.ndarray,
    stability_ks: dict[str, object],
) -> None:
    """Plot all, Early, and Late model distributions for the stability test."""

    bin_edges = calculate_stability_bin_edges(
        all_values,
        early_values,
        late_values,
    )

    maximum_density = 0.0

    for values, color, zorder in (
        (early_values, EARLY_COLOR, 2),
        (late_values, LATE_COLOR, 1),
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
                float(np.nanmax(density)),
            )

    ax.set_xlim(bin_edges[0], bin_edges[-1])

    if maximum_density > 0:
        ax.set_ylim(
            0,
            maximum_density * (1.0 + y_axis_margin_fraction),
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

    decision = (
        r"Reject $H_0$"
        if stability_ks["reject_null"]
        else r"Do not reject $H_0$"
    )

    annotation = (
        f"Early n={early_values.size}\n"
        f"Late n={late_values.size}\n"
        f"KS D={stability_ks['statistic']:.3f}\n"
        f"p={format_ks_p_value(stability_ks['p_value'])}\n"
        f"{decision} ({ks_significance_level_percent:g}% level)"
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
    model_all_values: np.ndarray,
    model_early_values: np.ndarray,
    model_late_values: np.ndarray,
    senorge_label: str,
    stability_ks: dict[str, object],
) -> plt.Figure:
    """
    Create the publication-style 2 x 3 diagnostic figure.

    Layout:
        (a) Independence      (b) Mean              (c) Standard deviation
        (d) Skewness          (e) Excess kurtosis   (f) Stability
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )

    # (a) Ensemble-member independence.
    plot_independence_panel(
        ax=axes[0, 0],
        correlations=independence_values,
    )
    add_panel_label(axes[0, 0], "(a)")

    # (b)-(e) Moments tests.
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
            senorge_label=senorge_label,
            show_legend=False,
        )

        add_panel_label(
            axes[row, column],
            panel_labels[statistic_name],
        )

    # (f) Lead-time stability.
    plot_stability_panel(
        ax=axes[1, 2],
        all_values=model_all_values,
        early_values=model_early_values,
        late_values=model_late_values,
        stability_ks=stability_ks,
    )
    add_panel_label(axes[1, 2], "(f)")

    #fig.suptitle(
    #    build_figure_title(),
    #    fontsize=SUPTITLE_FONTSIZE,
    #    fontweight="normal",
    #    y=0.985,
    #)

    # One legend applies to all six panels.
    fig.legend(
        handles=make_shared_legend_handles(senorge_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=6,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.09,
        top=0.86,
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
    senorge_label: str,
) -> None:
    """Print the four selected-month moments-test results."""

    print()
    print(f"{MONTH_LABELS[selected_month]} moments-test results")
    print("-" * 45)

    for statistic_name in STATISTICS:
        result = results[statistic_name]
        lower, upper = result["confidence_interval"]

        era5_marker = "" if result["era5_passes"] else "*"
        senorge_marker = "" if result["senorge_passes"] else "*"

        print(
            f"{STATISTIC_LABELS[statistic_name]:>18s} | "
            f"n={result['sample_size']:>3d} | "
            f"model interval=["
            f"{format_statistic_value(statistic_name, lower)}, "
            f"{format_statistic_value(statistic_name, upper)}] | "
            f"ERA5={format_statistic_value(statistic_name, result['era5_value'])}"
            f"{era5_marker} | "
            f"{senorge_label}="
            f"{format_statistic_value(statistic_name, result['senorge_value'])}"
            f"{senorge_marker}"
        )

    print("* outside the central model bootstrap interval")



def print_stability_results(
    model_all_values: np.ndarray,
    model_early_values: np.ndarray,
    model_late_values: np.ndarray,
    stability_ks: dict[str, object],
) -> None:
    """Print selected-month lead-time stability results."""

    alpha = get_ks_significance_threshold()

    decision = (
        "Reject H0"
        if stability_ks["reject_null"]
        else "Do not reject H0"
    )

    print()
    print(f"{MONTH_LABELS[selected_month]} stability test")
    print("-" * 45)
    print(
        f"All={model_all_values.size}, "
        f"Early={model_early_values.size}, "
        f"Late={model_late_values.size}"
    )
    print(
        "Partition check: "
        f"{model_early_values.size + model_late_values.size} "
        f"= {model_all_values.size}"
    )
    print(
        f"KS: D={stability_ks['statistic']:.3f}, "
        f"p={stability_ks['p_value']:.4g}"
    )
    print(
        f"Decision: {decision} at "
        f"{ks_significance_level_percent:g}% confidence "
        f"(alpha={alpha:.4f})"
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    senorge_variable = SENORGE_VARIABLE
    senorge_label = SENORGE_LABEL

    (
        independence_filename,
        shared_model_filename,
    ) = resolve_model_input_filenames()

    era5_filename = build_era5_filename()
    senorge_filename = build_senorge_filename()
    output_filename = build_output_filename()

    print("Selected month")
    print("--------------")
    print(MONTH_LABELS[selected_month])

    print()
    print("Input files")
    print("-----------")
    print(f"Independence model: {independence_filename}")
    print(f"Panels (b)-(f):     {shared_model_filename}")
    print(f"ERA5:               {era5_filename}")
    print(f"{senorge_label}:".ljust(20), senorge_filename)

    print()
    print("Model data for panels (b)-(f)")
    print("-----------------------------")
    print(f"Bias corrected: {USE_BIAS_CORRECTED_MODEL}")

    if USE_BIAS_CORRECTED_MODEL:
        print(
            f"BC reference:   "
            f"{BIAS_CORRECTION_REFERENCE}"
        )

    all_variable, early_variable, late_variable = (
        get_stability_variable_names()
    )

    print(f"All leads:      {all_variable}")
    print(f"Early leads:    {early_variable}")
    print(f"Late leads:     {late_variable}")

    independence_values = load_independence_values(
        independence_filename
    )

    (
        model_all_values,
        model_early_values,
        model_late_values,
        era5_values,
        senorge_values,
    ) = load_model_and_reference_values(
        model_filename=shared_model_filename,
        era5_filename=era5_filename,
        senorge_filename=senorge_filename,
        senorge_variable=senorge_variable,
        senorge_label=senorge_label,
    )

    validate_model_partition(
        model_all_values=model_all_values,
        model_early_values=model_early_values,
        model_late_values=model_late_values,
    )

    validate_moments_samples(
        model_values=model_all_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        senorge_label=senorge_label,
    )

    rng = np.random.default_rng(random_seed)

    moments_results = perform_all_moments_tests(
        model_values=model_all_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        rng=rng,
    )

    stability_ks = perform_stability_ks_test(
        early_values=model_early_values,
        late_values=model_late_values,
    )

    print()
    print(
        f"Independence pairs: {independence_values.size} "
        f"finite pooled correlations"
    )

    print_moments_results(
        results=moments_results,
        senorge_label=senorge_label,
    )

    print_stability_results(
        model_all_values=model_all_values,
        model_early_values=model_early_values,
        model_late_values=model_late_values,
        stability_ks=stability_ks,
    )

    figure = create_combined_figure(
        independence_values=independence_values,
        moments_results=moments_results,
        model_all_values=model_all_values,
        model_early_values=model_early_values,
        model_late_values=model_late_values,
        senorge_label=senorge_label,
        stability_ks=stability_ks,
    )

    if write2file:
        output_directory = os.path.dirname(output_filename)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        figure.savefig(
            output_filename,
            dpi=figure_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print()
        print(f"Wrote figure: {output_filename}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)
