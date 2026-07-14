"""
Perform a monthly UNSEEN moments-based model fidelity test.

For each calendar month, this script:

1. Extracts the full model sample of X-day accumulated precipitation maxima.
2. Extracts the ERA5 and SeNorge observational/reference samples.
3. Repeatedly resamples the model, with replacement, using the same sample
   length as the reference datasets.
4. Calculates one selected statistic for every model resample:
      - mean
      - standard deviation
      - skewness
      - kurtosis
5. Compares the ERA5 and SeNorge statistics with the resulting model
   bootstrap distribution.
6. Plots the model bootstrap distribution and the two reference values in a
   publication-quality 3 x 4 panel figure.

A reference dataset passes the moments test when its statistic lies inside
the central confidence interval of the bootstrapped model distribution.

Notes
-----
- The ERA5 and SeNorge monthly samples are expected to have the same length.
  This is normally true when they cover the same years and contain no missing
  values.
- Standard deviation is calculated using ddof=1.
- Skewness and kurtosis are calculated using bias-corrected estimators.
- Kurtosis is Fisher kurtosis, so a normal distribution has kurtosis = 0.

Inputs
------
- S2S monthly extreme distribution file.
- ERA5 monthly extreme distribution file.
- SeNorge or regridded SeNorge monthly extreme distribution file.

Output
------
- One publication-quality 3 x 4 panel PNG figure for the selected statistic.
"""

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import kurtosis, skew

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User-defined input parameters
# =============================================================================

# Catchment and accumulation period
catchment = "regine_drammen"
x_days = 2

# Select either native-grid or regridded SeNorge data
reference_dataset = "senorge"  # "senorge" or "senorge_regrid"

# Date ranges used in the input filenames
forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

# ERA5 grid used for the reference dataset
era5_grid = "0.5x0.5"

# Statistic to test and plot:
# "mean", "std", "skewness", or "kurtosis"
statistic = "kurtosis"

# Number of model resamples used to construct the bootstrap distribution
number_of_bootstrap_samples = 10_000

# Number of histogram bins
number_of_bins = 30

# Confidence level used for the moments test.
# At 95%, a reference value fails if it lies outside the central 95% interval
# of the bootstrapped model statistic.
confidence_level_percent = 95.0

# Seed makes the random resampling reproducible.
# Set to None for a different random result every time.
random_seed = 42

# Normalize the histograms to probability density
plot_probability_density = True

# Use identical x- and y-axis limits in all monthly panels
use_common_axis_limits = True

# Add a small margin above the largest histogram value
y_axis_margin_fraction = 0.08

# Figure output
write2file = True
filename_out = os.path.join(
    config.dirs["fig"],
    (
        f"monthly_moments_test_{statistic}_{x_days}dayacc_"
        f"{catchment}_model_era5_{reference_dataset}.png"
    ),
)


# =============================================================================
# Dataset-specific settings
# =============================================================================

# Model data
MODEL_VARIABLE = "tp"
MODEL_EXTREME_VARIABLE = "max_value"
MODEL_MONTH_COORDINATE = "month_of_year"

# ERA5 data
ERA5_VARIABLE = "tp"

# SeNorge variable names
REFERENCE_VARIABLES = {
    "senorge": "tp",
    "senorge_regrid": "rr",
}

# Plot labels
REFERENCE_LABELS = {
    "senorge": "SeNorge",
    "senorge_regrid": "SeNorge regrid",
}


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 13.0
FIG_HEIGHT_IN = 8

TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12
AXIS_LABELSIZE = 10
TICK_LABELSIZE = 9
LEGEND_FONTSIZE = 7.5

HISTOGRAM_LINEWIDTH = 1.4
REFERENCE_LINEWIDTH = 1.6

MODEL_COLOR = "black"
SENORGE_COLOR = "tab:red"
ERA5_COLOR = "tab:blue"

MONTHS = np.arange(1, 13)

MONTH_LABELS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

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
    "kurtosis": "Precipitation kurtosis",
}


# =============================================================================
# Configuration helpers
# =============================================================================

def get_reference_variable(reference_dataset: str) -> str:
    """
    Return the precipitation variable used by the selected SeNorge dataset.
    """

    if reference_dataset not in REFERENCE_VARIABLES:
        valid_options = ", ".join(REFERENCE_VARIABLES)

        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return REFERENCE_VARIABLES[reference_dataset]


def get_reference_label(reference_dataset: str) -> str:
    """
    Return a publication-friendly name for the selected SeNorge dataset.
    """

    if reference_dataset not in REFERENCE_LABELS:
        valid_options = ", ".join(REFERENCE_LABELS)

        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return REFERENCE_LABELS[reference_dataset]


def get_catchment_label(catchment: str) -> str:
    """
    Return a publication-friendly catchment name.
    """

    labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }

    return labels.get(
        catchment,
        catchment.replace("_", " ").title(),
    )


def validate_user_settings() -> None:
    """
    Check user-defined settings before reading or processing data.
    """

    valid_statistics = set(STATISTIC_LABELS)

    if statistic not in valid_statistics:
        raise ValueError(
            f"Unknown statistic '{statistic}'. "
            f"Valid options are: {sorted(valid_statistics)}."
        )

    if not isinstance(number_of_bootstrap_samples, int):
        raise TypeError("number_of_bootstrap_samples must be an integer.")

    if number_of_bootstrap_samples < 1:
        raise ValueError(
            "number_of_bootstrap_samples must be greater than or equal to 1."
        )

    if not isinstance(number_of_bins, int):
        raise TypeError("number_of_bins must be an integer.")

    if number_of_bins < 1:
        raise ValueError(
            "number_of_bins must be greater than or equal to 1."
        )

    if not 0.0 < confidence_level_percent < 100.0:
        raise ValueError(
            "confidence_level_percent must be greater than 0 and less than 100."
        )

    if y_axis_margin_fraction < 0:
        raise ValueError(
            "y_axis_margin_fraction must be greater than or equal to zero."
        )

    if len(reference_years) != 2:
        raise ValueError(
            "reference_years must contain a start year and an end year."
        )

    if len(forecast_date_range) != 2:
        raise ValueError(
            "forecast_date_range must contain a start date and an end date."
        )


# =============================================================================
# Filename helpers
# =============================================================================

def make_model_filename() -> str:
    """
    Create the S2S forecast and hindcast input filename.
    """

    return (
        f"{config.dirs['s2s_processed']}"
        f"distribution_monthly_extremes_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )


def make_era5_filename() -> str:
    """
    Create the ERA5 input filename.
    """

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def make_reference_filename(
    reference_dataset: str,
    reference_variable: str,
) -> str:
    """
    Create the SeNorge or SeNorge-regrid input filename.
    """

    return (
        f"{config.dirs[f'{reference_dataset}_processed']}"
        f"distribution_monthly_extremes_{reference_variable}_{x_days}dayacc_"
        f"{catchment}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


# =============================================================================
# Data loading and validation
# =============================================================================

def load_datasets(
    filename_model: str,
    filename_era5: str,
    filename_reference: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Open the model, ERA5, and SeNorge datasets.
    """

    model_ds = xr.open_dataset(filename_model)
    era5_ds = xr.open_dataset(filename_era5)
    reference_ds = xr.open_dataset(filename_reference)

    return model_ds, era5_ds, reference_ds


def check_variable_exists(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> None:
    """
    Raise a clear error when a required variable is missing.
    """

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are: {list(ds.data_vars)}"
        )


def check_coordinate_exists(
    data: xr.DataArray,
    coordinate: str,
    dataset_name: str,
) -> None:
    """
    Raise a clear error when a required coordinate or dimension is missing.
    """

    available_names = set(data.coords) | set(data.dims)

    if coordinate not in available_names:
        raise KeyError(
            f"Coordinate or dimension '{coordinate}' was not found in "
            f"{dataset_name}. Available dimensions are {data.dims}, and "
            f"available coordinates are {list(data.coords)}."
        )


# =============================================================================
# Data extraction
# =============================================================================

def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """
    Flatten an array and retain only finite values.
    """

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def get_model_values_by_month(
    model_ds: xr.Dataset,
    variable: str = MODEL_EXTREME_VARIABLE,
) -> dict[int, np.ndarray]:
    """
    Extract one array of model precipitation extremes for each month.

    Expected model structure:

        max_value(month_of_year, index)

    Additional dimensions are allowed because the selected monthly values are
    flattened before analysis.
    """

    check_variable_exists(
        model_ds,
        variable,
        dataset_name="model dataset",
    )

    data = model_ds[variable]

    check_coordinate_exists(
        data,
        MODEL_MONTH_COORDINATE,
        dataset_name="model dataset",
    )

    values_by_month = {}

    for month in MONTHS:
        monthly_data = data.sel(
            {MODEL_MONTH_COORDINATE: month}
        )

        values_by_month[month] = remove_missing_values(
            monthly_data.values
        )

    return values_by_month


def get_reference_values_by_month(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> dict[int, np.ndarray]:
    """
    Extract annual monthly precipitation extremes for each calendar month.

    Expected reference-data structure:

        variable(year, month)
    """

    check_variable_exists(
        ds,
        variable,
        dataset_name=dataset_name,
    )

    data = ds[variable]

    check_coordinate_exists(
        data,
        "month",
        dataset_name=dataset_name,
    )

    values_by_month = {}

    for month in MONTHS:
        monthly_data = data.sel(month=month)

        values_by_month[month] = remove_missing_values(
            monthly_data.values
        )

    return values_by_month


def check_monthly_samples(
    values_by_month: dict[int, np.ndarray],
    dataset_name: str,
) -> None:
    """
    Check that each month has enough values for the selected statistic.
    """

    minimum_sample_size = 4 if statistic in {"skewness", "kurtosis"} else 2

    for month, values in values_by_month.items():
        if values.size < minimum_sample_size:
            raise ValueError(
                f"Only {values.size} finite {dataset_name} values were found "
                f"for {MONTH_LABELS[month - 1]}. At least "
                f"{minimum_sample_size} values are required for {statistic}."
            )


def check_reference_sample_sizes(
    era5_values_by_month: dict[int, np.ndarray],
    reference_values_by_month: dict[int, np.ndarray],
    reference_label: str,
) -> None:
    """
    Require ERA5 and SeNorge to have equal sample sizes in each month.

    A single model bootstrap distribution is plotted in each panel. Therefore,
    both reference statistics must be compared with a model distribution
    generated using the same sample size.
    """

    for month in MONTHS:
        era5_size = era5_values_by_month[month].size
        reference_size = reference_values_by_month[month].size

        if era5_size != reference_size:
            raise ValueError(
                f"ERA5 and {reference_label} have different sample sizes for "
                f"{MONTH_LABELS[month - 1]}: ERA5={era5_size}, "
                f"{reference_label}={reference_size}. A single bootstrap "
                "distribution cannot exactly match both reference lengths. "
                "Check missing data or use a common period before running."
            )


# =============================================================================
# Statistic and bootstrap helpers
# =============================================================================

def calculate_statistic(values: np.ndarray, statistic_name: str) -> float:
    """
    Calculate the selected sample statistic.

    Parameters
    ----------
    values
        One-dimensional sample.
    statistic_name
        One of: "mean", "std", "skewness", or "kurtosis".
    """

    if statistic_name == "mean":
        return float(np.mean(values))

    if statistic_name == "std":
        return float(np.std(values, ddof=1))

    if statistic_name == "skewness":
        return float(skew(values, bias=False))

    if statistic_name == "kurtosis":
        return float(
            kurtosis(
                values,
                fisher=True,
                bias=False,
            )
        )

    raise ValueError(f"Unsupported statistic: {statistic_name}")


def get_vectorized_statistic_function(
    statistic_name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a function that calculates the selected statistic along axis 1.

    This allows all bootstrap resamples to be processed efficiently.
    """

    if statistic_name == "mean":
        return lambda samples: np.mean(samples, axis=1)

    if statistic_name == "std":
        return lambda samples: np.std(samples, axis=1, ddof=1)

    if statistic_name == "skewness":
        return lambda samples: skew(
            samples,
            axis=1,
            bias=False,
        )

    if statistic_name == "kurtosis":
        return lambda samples: kurtosis(
            samples,
            axis=1,
            fisher=True,
            bias=False,
        )

    raise ValueError(f"Unsupported statistic: {statistic_name}")


def bootstrap_model_statistic(
    model_values: np.ndarray,
    sample_size: int,
    number_of_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct a bootstrap distribution of the selected model statistic.

    The model is sampled with replacement. Every resample has the same number
    of values as the observational/reference samples.
    """

    if sample_size > model_values.size:
        print(
            "Warning: the reference sample is larger than the model sample. "
            "Sampling with replacement remains valid, but this is unusual."
        )

    sample_indices = rng.integers(
        low=0,
        high=model_values.size,
        size=(number_of_samples, sample_size),
    )

    resampled_values = model_values[sample_indices]

    statistic_function = get_vectorized_statistic_function(statistic)

    bootstrap_statistics = statistic_function(resampled_values)
    bootstrap_statistics = remove_missing_values(bootstrap_statistics)

    if bootstrap_statistics.size != number_of_samples:
        print(
            f"Warning: retained {bootstrap_statistics.size} finite bootstrap "
            f"statistics out of {number_of_samples} for {statistic}."
        )

    return bootstrap_statistics


def calculate_confidence_interval(
    bootstrap_values: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the central bootstrap confidence interval.
    """

    alpha_percent = 100.0 - confidence_level_percent
    lower_percentile = alpha_percent / 2.0
    upper_percentile = 100.0 - lower_percentile

    lower = np.percentile(
        bootstrap_values,
        lower_percentile,
    )

    upper = np.percentile(
        bootstrap_values,
        upper_percentile,
    )

    return float(lower), float(upper)


def reference_passes_test(
    reference_value: float,
    confidence_interval: tuple[float, float],
) -> bool:
    """
    Return True when the reference statistic is inside the model interval.
    """

    lower, upper = confidence_interval
    return lower <= reference_value <= upper


def format_statistic_value(value: float) -> str:
    """
    Format a statistic compactly for legends and terminal output.
    """

    if statistic in {"mean", "std"}:
        return f"{value:.1f}"

    return f"{value:.2f}"


def format_reference_label(
    dataset_label: str,
    value: float,
    passes: bool,
) -> str:
    """
    Create a legend label and mark a failed moments test with an asterisk.
    """

    asterisk = "" if passes else "*"

    return (
        f"{dataset_label}: "
        f"{format_statistic_value(value)}{asterisk}"
    )


# =============================================================================
# Analysis
# =============================================================================

def perform_monthly_moments_test(
    model_values_by_month: dict[int, np.ndarray],
    era5_values_by_month: dict[int, np.ndarray],
    reference_values_by_month: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[int, dict[str, object]]:
    """
    Perform the bootstrap moments test independently for every month.
    """

    results = {}

    for month in MONTHS:
        model_values = model_values_by_month[month]
        era5_values = era5_values_by_month[month]
        reference_values = reference_values_by_month[month]

        sample_size = era5_values.size

        bootstrap_values = bootstrap_model_statistic(
            model_values=model_values,
            sample_size=sample_size,
            number_of_samples=number_of_bootstrap_samples,
            rng=rng,
        )

        confidence_interval = calculate_confidence_interval(
            bootstrap_values
        )

        era5_value = calculate_statistic(
            era5_values,
            statistic,
        )

        reference_value = calculate_statistic(
            reference_values,
            statistic,
        )

        era5_passes = reference_passes_test(
            era5_value,
            confidence_interval,
        )

        reference_passes = reference_passes_test(
            reference_value,
            confidence_interval,
        )

        results[month] = {
            "bootstrap_values": bootstrap_values,
            "confidence_interval": confidence_interval,
            "sample_size": sample_size,
            "era5_value": era5_value,
            "reference_value": reference_value,
            "era5_passes": era5_passes,
            "reference_passes": reference_passes,
        }

    return results


# =============================================================================
# Histogram and axis helpers
# =============================================================================

def calculate_global_histogram_settings(
    results: dict[int, dict[str, object]],
) -> tuple[np.ndarray, float]:
    """
    Calculate common histogram bins and a common y-axis maximum.

    The x-axis covers all model bootstrap values and both reference values over
    all 12 months. The y-axis is based on the tallest monthly histogram.
    """

    all_values = []

    for month in MONTHS:
        month_results = results[month]

        all_values.append(
            np.asarray(month_results["bootstrap_values"])
        )

        all_values.append(
            np.asarray(
                [
                    month_results["era5_value"],
                    month_results["reference_value"],
                ]
            )
        )

    combined_values = np.concatenate(all_values)

    x_min = float(np.min(combined_values))
    x_max = float(np.max(combined_values))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
        x_min -= padding
        x_max += padding
    else:
        padding = 0.03 * (x_max - x_min)
        x_min -= padding
        x_max += padding

    common_bin_edges = np.linspace(
        x_min,
        x_max,
        number_of_bins + 1,
    )

    y_max = 0.0

    for month in MONTHS:
        counts, _ = np.histogram(
            results[month]["bootstrap_values"],
            bins=common_bin_edges,
            density=plot_probability_density,
        )

        y_max = max(y_max, float(np.max(counts)))

    y_max *= 1.0 + y_axis_margin_fraction

    return common_bin_edges, y_max


def calculate_monthly_bin_edges(
    month_results: dict[str, object],
) -> np.ndarray:
    """
    Calculate panel-specific bins when common axes are not requested.
    """

    combined_values = np.concatenate(
        [
            np.asarray(month_results["bootstrap_values"]),
            np.asarray(
                [
                    month_results["era5_value"],
                    month_results["reference_value"],
                ]
            ),
        ]
    )

    x_min = float(np.min(combined_values))
    x_max = float(np.max(combined_values))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
        x_min -= padding
        x_max += padding
    else:
        padding = 0.03 * (x_max - x_min)
        x_min -= padding
        x_max += padding

    return np.linspace(
        x_min,
        x_max,
        number_of_bins + 1,
    )


def get_histogram_y_label() -> str:
    """
    Return the y-axis label corresponding to histogram normalization.
    """

    if plot_probability_density:
        return "Probability density"

    return "Bootstrap samples"


def make_panel_legend_handles(
    reference_label: str,
    month_results: dict[str, object],
) -> list[Line2D]:
    """
    Create legend handles for one monthly panel.
    """

    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model bootstrap",
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            linewidth=1.2,
            linestyle=":",
            label=f"{confidence_level_percent:g}% model interval",
        ),
        Line2D(
            [0],
            [0],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="--",
            label=format_reference_label(
                dataset_label="ERA5",
                value=float(month_results["era5_value"]),
                passes=bool(month_results["era5_passes"]),
            ),
        ),
        Line2D(
            [0],
            [0],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="--",
            label=format_reference_label(
                dataset_label=reference_label,
                value=float(month_results["reference_value"]),
                passes=bool(month_results["reference_passes"]),
            ),
        ),
    ]


# =============================================================================
# Plot formatting
# =============================================================================

def apply_axis_formatting(
    ax: plt.Axes,
    month_label: str,
    row_index: int,
    column_index: int,
) -> None:
    """
    Apply consistent publication-style formatting to one panel.
    """

    ax.set_title(
        month_label,
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=5,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABELSIZE,
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.0,
        width=0.6,
    )

    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if row_index == 2:
        ax.set_xlabel(
            STATISTIC_AXIS_LABELS[statistic],
            fontsize=AXIS_LABELSIZE,
        )

    if column_index == 0:
        ax.set_ylabel(
            get_histogram_y_label(),
            fontsize=AXIS_LABELSIZE,
        )


# =============================================================================
# Main plotting function
# =============================================================================

def plot_monthly_moments_test(
    results: dict[int, dict[str, object]],
    reference_label: str,
    filename_out: str,
    write2file: bool,
) -> None:
    """
    Plot the monthly model bootstrap distributions and reference statistics.
    """

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        squeeze=False,
    )

    if use_common_axis_limits:
        common_bin_edges, common_y_max = (
            calculate_global_histogram_settings(results)
        )
    else:
        common_bin_edges = None
        common_y_max = None

    for month_index, month in enumerate(MONTHS):
        row_index = month_index // 4
        column_index = month_index % 4

        ax = axes[row_index, column_index]
        month_results = results[month]

        if use_common_axis_limits:
            bin_edges = common_bin_edges
        else:
            bin_edges = calculate_monthly_bin_edges(
                month_results
            )

        ax.hist(
            month_results["bootstrap_values"],
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=1,
        )

        # Plot the model confidence interval.
        confidence_lower, confidence_upper = (
            month_results["confidence_interval"]
        )

        ax.axvline(
            confidence_lower,
            color=MODEL_COLOR,
            linewidth=1.2,
            linestyle=":",
            zorder=2,
        )

        ax.axvline(
            confidence_upper,
            color=MODEL_COLOR,
            linewidth=1.2,
            linestyle=":",
            zorder=2,
        )

        # Plot the reference statistics.
        ax.axvline(
            month_results["era5_value"],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="--",
            zorder=3,
        )

        ax.axvline(
            month_results["reference_value"],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="--",
            zorder=3,
        )

        ax.set_xlim(
            bin_edges[0],
            bin_edges[-1],
        )

        if use_common_axis_limits:
            ax.set_ylim(
                0,
                common_y_max,
            )
        else:
            ax.set_ylim(bottom=0)

        apply_axis_formatting(
            ax=ax,
            month_label=MONTH_LABELS[month - 1],
            row_index=row_index,
            column_index=column_index,
        )

        ax.legend(
            handles=make_panel_legend_handles(
                reference_label=reference_label,
                month_results=month_results,
            ),
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="0.8",
            framealpha=0.9,
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0.4,
            labelspacing=0.25,
        )

    catchment_label = get_catchment_label(catchment)
    statistic_label = STATISTIC_LABELS[statistic]

    fig.suptitle(
        (
            f"{catchment_label}: monthly {statistic_label.lower()} "
            f"moments test for {x_days}-day accumulated precipitation maxima"
        ),
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="normal",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.28,
        hspace=0.38,
    )

    if write2file:
        output_directory = os.path.dirname(filename_out)

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        fig.savefig(
            filename_out,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Wrote figure: {filename_out}")

    plt.show()
    plt.close(fig)


# =============================================================================
# Terminal output
# =============================================================================

def print_test_results(
    results: dict[int, dict[str, object]],
    reference_label: str,
) -> None:
    """
    Print a concise table of monthly moments-test results.
    """

    print()
    print("Monthly moments-test results")
    print("----------------------------")
    print(f"Statistic: {STATISTIC_LABELS[statistic]}")
    print(
        f"Model bootstrap samples: {number_of_bootstrap_samples:,}"
    )
    print(
        f"Confidence level: {confidence_level_percent:g}%"
    )
    print("An asterisk marks a failed test.")
    print()

    for month in MONTHS:
        month_results = results[month]
        lower, upper = month_results["confidence_interval"]

        era5_marker = (
            ""
            if month_results["era5_passes"]
            else "*"
        )

        reference_marker = (
            ""
            if month_results["reference_passes"]
            else "*"
        )

        print(
            f"{MONTH_LABELS[month - 1]:>9s} | "
            f"n={month_results['sample_size']:>3d} | "
            f"model interval=[{format_statistic_value(lower)}, "
            f"{format_statistic_value(upper)}] | "
            f"ERA5={format_statistic_value(month_results['era5_value'])}"
            f"{era5_marker} | "
            f"{reference_label}="
            f"{format_statistic_value(month_results['reference_value'])}"
            f"{reference_marker}"
        )


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    reference_variable = get_reference_variable(
        reference_dataset
    )

    reference_label = get_reference_label(
        reference_dataset
    )

    filename_model = make_model_filename()
    filename_era5 = make_era5_filename()

    filename_reference = make_reference_filename(
        reference_dataset=reference_dataset,
        reference_variable=reference_variable,
    )

    print("Reading input datasets")
    print("----------------------")
    print(f"Model:       {filename_model}")
    print(f"ERA5:        {filename_era5}")
    print(f"{reference_label}: {filename_reference}")

    model_ds, era5_ds, reference_ds = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_reference=filename_reference,
    )

    try:
        model_values_by_month = get_model_values_by_month(
            model_ds=model_ds,
            variable=MODEL_EXTREME_VARIABLE,
        )

        era5_values_by_month = get_reference_values_by_month(
            ds=era5_ds,
            variable=ERA5_VARIABLE,
            dataset_name="ERA5 dataset",
        )

        reference_values_by_month = get_reference_values_by_month(
            ds=reference_ds,
            variable=reference_variable,
            dataset_name=f"{reference_label} dataset",
        )

        check_monthly_samples(
            values_by_month=model_values_by_month,
            dataset_name="model",
        )

        check_monthly_samples(
            values_by_month=era5_values_by_month,
            dataset_name="ERA5",
        )

        check_monthly_samples(
            values_by_month=reference_values_by_month,
            dataset_name=reference_label,
        )

        check_reference_sample_sizes(
            era5_values_by_month=era5_values_by_month,
            reference_values_by_month=reference_values_by_month,
            reference_label=reference_label,
        )

        rng = np.random.default_rng(
            random_seed
        )

        results = perform_monthly_moments_test(
            model_values_by_month=model_values_by_month,
            era5_values_by_month=era5_values_by_month,
            reference_values_by_month=reference_values_by_month,
            rng=rng,
        )

        print_test_results(
            results=results,
            reference_label=reference_label,
        )

        plot_monthly_moments_test(
            results=results,
            reference_label=reference_label,
            filename_out=filename_out,
            write2file=write2file,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        reference_ds.close()
