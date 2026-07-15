"""
Perform a monthly UNSEEN moments-based fidelity test.

For each month, the script:

1. Reads the full model sample of X-day accumulated precipitation maxima.
2. Reads the ERA5 and SeNorge samples.
3. Resamples the model with replacement using the same sample size as the
   ERA5 and SeNorge samples.
4. Calculates one selected statistic for every model resample:
      - mean
      - standard deviation
      - skewness
      - kurtosis
5. Compares the ERA5 and SeNorge statistic with the model bootstrap
   distribution.
6. Plots the model bootstrap distribution and the ERA5 and SeNorge values in
   a publication-quality 3 x 4 panel figure.

ERA5 or SeNorge fails the moments test when its statistic lies outside the
central confidence interval of the model bootstrap distribution.
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
catchment = "regine_glomma"
x_days = 2

# Date ranges used in the input filenames
forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

# ERA5 grid
era5_grid = "0.5x0.5"

# SeNorge dataset
senorge_dataset = "senorge"  # "senorge" or "senorge_regrid"

# Statistic to test and plot:
# "mean", "std", "skewness", or "kurtosis"
statistic = "kurtosis"

# Apply finite-sample bias correction to skewness and kurtosis
bias_correct_skewness = True
bias_correct_kurtosis = True

# Number of model resamples
number_of_bootstrap_samples = 10000

# Number of histogram bins
number_of_bins = 30

# Confidence level for the moments test
confidence_level_percent = 95.0

# Random seed for reproducible resampling
random_seed = 42

# Normalize histograms to probability density
plot_probability_density = True

# Use the same x- and y-axis limits for all months
use_common_axis_limits = True

# Add a small margin above the highest histogram
y_axis_margin_fraction = 0.08

# Figure output
write2file = True
filename_out = os.path.join(
    config.dirs["fig"],
    (
        f"monthly_moments_test_{statistic}_{x_days}dayacc_"
        f"{catchment}_model_{forecast_date_range[0]}_{forecast_date_range[1]}_"
        f"era5_senorge_{reference_years[0]}-{reference_years[1]}.png"
    ),
)


# =============================================================================
# Dataset settings
# =============================================================================

MODEL_VARIABLE = "tp24"
MODEL_EXTREME_VARIABLE = "max_value"
MODEL_MONTH_COORDINATE = "month_of_year"

ERA5_VARIABLE = "tp24"

SENORGE_VARIABLES = {
    "senorge": "rr",
    "senorge_regrid": "rr",
}

SENORGE_LABELS = {
    "senorge": "SeNorge",
    "senorge_regrid": "SeNorge regrid",
}


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 13.0
FIG_HEIGHT_IN = 8.0

TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12
AXIS_LABELSIZE = 10
TICK_LABELSIZE = 9
LEGEND_FONTSIZE = 7.5

HISTOGRAM_LINEWIDTH = 1.4
REFERENCE_LINEWIDTH = 1.6
CONFIDENCE_LINEWIDTH = 1.2

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

def get_senorge_variable() -> str:
    """Return the variable name used by the selected SeNorge dataset."""

    if senorge_dataset not in SENORGE_VARIABLES:
        valid_options = ", ".join(SENORGE_VARIABLES)

        raise ValueError(
            f"Unknown senorge_dataset '{senorge_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return SENORGE_VARIABLES[senorge_dataset]


def get_senorge_label() -> str:
    """Return the plot label for the selected SeNorge dataset."""

    if senorge_dataset not in SENORGE_LABELS:
        valid_options = ", ".join(SENORGE_LABELS)

        raise ValueError(
            f"Unknown senorge_dataset '{senorge_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return SENORGE_LABELS[senorge_dataset]


def get_catchment_label() -> str:
    """Return a readable catchment name."""

    labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }

    return labels.get(
        catchment,
        catchment.replace("_", " ").title(),
    )


def get_bias_correction_label() -> str:
    """Describe the bias-correction setting for the selected statistic."""

    if statistic == "skewness":
        return "on" if bias_correct_skewness else "off"

    if statistic == "kurtosis":
        return "on" if bias_correct_kurtosis else "off"

    return "not applicable"


def validate_user_settings() -> None:
    """Check the user-defined settings."""

    if statistic not in STATISTIC_LABELS:
        raise ValueError(
            f"Unknown statistic '{statistic}'. "
            f"Valid options are: {sorted(STATISTIC_LABELS)}."
        )

    if not isinstance(bias_correct_skewness, bool):
        raise TypeError("bias_correct_skewness must be True or False.")

    if not isinstance(bias_correct_kurtosis, bool):
        raise TypeError("bias_correct_kurtosis must be True or False.")

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
    """Create the model input filename."""

    return (
        f"{config.dirs['s2s_processed']}"
        f"distribution_monthly_extremes_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )


def make_era5_filename() -> str:
    """Create the ERA5 input filename."""

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def make_senorge_filename(
    senorge_variable: str,
) -> str:
    """Create the SeNorge or SeNorge-regrid input filename."""

    return (
        f"{config.dirs[f'{senorge_dataset}_processed']}"
        f"distribution_monthly_extremes_{senorge_variable}_{x_days}dayacc_"
        f"{catchment}_{senorge_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


# =============================================================================
# Data loading and validation
# =============================================================================

def load_datasets(
    filename_model: str,
    filename_era5: str,
    filename_senorge: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open the model, ERA5, and SeNorge datasets."""

    model_ds = xr.open_dataset(filename_model)
    era5_ds = xr.open_dataset(filename_era5)
    senorge_ds = xr.open_dataset(filename_senorge)

    return model_ds, era5_ds, senorge_ds


def check_variable_exists(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> None:
    """Raise an error when a required variable is missing."""

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
    """Raise an error when a required coordinate is missing."""

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
    """Flatten an array and keep only finite values."""

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def get_model_values_by_month(
    model_ds: xr.Dataset,
) -> dict[int, np.ndarray]:
    """Extract one model array for each calendar month."""

    check_variable_exists(
        model_ds,
        MODEL_EXTREME_VARIABLE,
        "model dataset",
    )

    data = model_ds[MODEL_EXTREME_VARIABLE]

    check_coordinate_exists(
        data,
        MODEL_MONTH_COORDINATE,
        "model dataset",
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


def get_dataset_values_by_month(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> dict[int, np.ndarray]:
    """Extract annual monthly extremes for each calendar month."""

    check_variable_exists(
        ds,
        variable,
        dataset_name,
    )

    data = ds[variable]

    check_coordinate_exists(
        data,
        "month",
        dataset_name,
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
    """Check that every month has enough values."""

    minimum_sample_size = (
        4 if statistic in {"skewness", "kurtosis"} else 2
    )

    for month, values in values_by_month.items():
        if values.size < minimum_sample_size:
            raise ValueError(
                f"Only {values.size} finite {dataset_name} values were found "
                f"for {MONTH_LABELS[month - 1]}. At least "
                f"{minimum_sample_size} values are required."
            )


def check_era5_senorge_sample_sizes(
    era5_values_by_month: dict[int, np.ndarray],
    senorge_values_by_month: dict[int, np.ndarray],
    senorge_label: str,
) -> None:
    """Require ERA5 and SeNorge to have equal monthly sample sizes."""

    for month in MONTHS:
        era5_size = era5_values_by_month[month].size
        senorge_size = senorge_values_by_month[month].size

        if era5_size != senorge_size:
            raise ValueError(
                f"ERA5 and {senorge_label} have different sample sizes for "
                f"{MONTH_LABELS[month - 1]}: ERA5={era5_size}, "
                f"{senorge_label}={senorge_size}. "
                "Use a common period or remove missing values consistently."
            )


# =============================================================================
# Statistic and bootstrap helpers
# =============================================================================

def calculate_statistic(
    values: np.ndarray,
) -> float:
    """Calculate the selected statistic."""

    if statistic == "mean":
        return float(np.mean(values))

    if statistic == "std":
        return float(np.std(values, ddof=1))

    if statistic == "skewness":
        return float(
            skew(
                values,
                bias=not bias_correct_skewness,
            )
        )

    if statistic == "kurtosis":
        return float(
            kurtosis(
                values,
                fisher=True,
                bias=not bias_correct_kurtosis,
            )
        )

    raise ValueError(f"Unsupported statistic: {statistic}")


def get_vectorized_statistic_function() -> Callable[[np.ndarray], np.ndarray]:
    """Return the selected statistic calculated along axis 1."""

    if statistic == "mean":
        return lambda samples: np.mean(samples, axis=1)

    if statistic == "std":
        return lambda samples: np.std(
            samples,
            axis=1,
            ddof=1,
        )

    if statistic == "skewness":
        return lambda samples: skew(
            samples,
            axis=1,
            bias=not bias_correct_skewness,
        )

    if statistic == "kurtosis":
        return lambda samples: kurtosis(
            samples,
            axis=1,
            fisher=True,
            bias=not bias_correct_kurtosis,
        )

    raise ValueError(f"Unsupported statistic: {statistic}")


def bootstrap_model_statistic(
    model_values: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create the bootstrap distribution of the selected model statistic."""

    sample_indices = rng.integers(
        low=0,
        high=model_values.size,
        size=(number_of_bootstrap_samples, sample_size),
    )

    resampled_values = model_values[sample_indices]

    statistic_function = get_vectorized_statistic_function()
    bootstrap_values = statistic_function(resampled_values)

    return remove_missing_values(bootstrap_values)


def calculate_confidence_interval(
    bootstrap_values: np.ndarray,
) -> tuple[float, float]:
    """Calculate the central model confidence interval."""

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


def value_passes_test(
    value: float,
    confidence_interval: tuple[float, float],
) -> bool:
    """Return True when a value lies inside the model interval."""

    lower, upper = confidence_interval
    return lower <= value <= upper


def format_statistic_value(value: float) -> str:
    """Format a statistic for legends and terminal output."""

    if statistic in {"mean", "std"}:
        return f"{value:.1f}"

    return f"{value:.2f}"


# =============================================================================
# Analysis
# =============================================================================

def perform_monthly_moments_test(
    model_values_by_month: dict[int, np.ndarray],
    era5_values_by_month: dict[int, np.ndarray],
    senorge_values_by_month: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[int, dict[str, object]]:
    """Perform the moments test separately for each month."""

    results = {}

    for month in MONTHS:
        model_values = model_values_by_month[month]
        era5_values = era5_values_by_month[month]
        senorge_values = senorge_values_by_month[month]

        sample_size = era5_values.size

        bootstrap_values = bootstrap_model_statistic(
            model_values=model_values,
            sample_size=sample_size,
            rng=rng,
        )

        confidence_interval = calculate_confidence_interval(
            bootstrap_values
        )

        era5_value = calculate_statistic(
            era5_values
        )

        senorge_value = calculate_statistic(
            senorge_values
        )

        results[month] = {
            "bootstrap_values": bootstrap_values,
            "confidence_interval": confidence_interval,
            "sample_size": sample_size,
            "era5_value": era5_value,
            "senorge_value": senorge_value,
            "era5_passes": value_passes_test(
                era5_value,
                confidence_interval,
            ),
            "senorge_passes": value_passes_test(
                senorge_value,
                confidence_interval,
            ),
        }

    return results


# =============================================================================
# Histogram and axis helpers
# =============================================================================

def calculate_global_histogram_settings(
    results: dict[int, dict[str, object]],
) -> tuple[np.ndarray, float]:
    """Calculate common histogram bins and a common y-axis maximum."""

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
                    month_results["senorge_value"],
                ]
            )
        )

    combined_values = np.concatenate(all_values)

    x_min = float(np.min(combined_values))
    x_max = float(np.max(combined_values))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
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

        y_max = max(
            y_max,
            float(np.max(counts)),
        )

    y_max *= 1.0 + y_axis_margin_fraction

    return common_bin_edges, y_max


def calculate_monthly_bin_edges(
    month_results: dict[str, object],
) -> np.ndarray:
    """Calculate bins for one month."""

    combined_values = np.concatenate(
        [
            np.asarray(month_results["bootstrap_values"]),
            np.asarray(
                [
                    month_results["era5_value"],
                    month_results["senorge_value"],
                ]
            ),
        ]
    )

    x_min = float(np.min(combined_values))
    x_max = float(np.max(combined_values))

    if np.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
    else:
        padding = 0.03 * (x_max - x_min)

    return np.linspace(
        x_min - padding,
        x_max + padding,
        number_of_bins + 1,
    )


def get_histogram_y_label() -> str:
    """Return the histogram y-axis label."""

    if plot_probability_density:
        return "Probability density"

    return "Bootstrap samples"


# =============================================================================
# Legend helpers
# =============================================================================

def make_full_legend_handles(
    senorge_label: str,
    month_results: dict[str, object],
) -> list[Line2D]:
    """
    Create the full legend used in the top-left panel.

    Failure labels are appended below the main legend entries when ERA5 or
    SeNorge fails the moments test in January.
    """

    handles = [
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
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            label=f"{confidence_level_percent:g}% model interval",
        ),
        Line2D(
            [0],
            [0],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="-",
            label="ERA5",
        ),
        Line2D(
            [0],
            [0],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="-",
            label=senorge_label,
        ),
    ]

    if not month_results["era5_passes"]:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                linewidth=0,
                marker="",
                label="ERA5 fail",
            )
        )

    if not month_results["senorge_passes"]:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                linewidth=0,
                marker="",
                label=f"{senorge_label} fail",
            )
        )

    return handles


def make_failure_legend_handles(
    month_results: dict[str, object],
    senorge_label: str,
) -> list[Line2D]:
    """
    Create text-only legend entries for datasets that fail the test.

    The handles are invisible. The corresponding legend text is colored later.
    """

    handles = []

    if not month_results["era5_passes"]:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                linewidth=0,
                marker="",
                label="ERA5 fail",
            )
        )

    if not month_results["senorge_passes"]:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                linewidth=0,
                marker="",
                label=f"{senorge_label} fail",
            )
        )

    return handles


def color_failure_legend_text(
    legend,
    senorge_label: str,
) -> None:
    """
    Color failure labels without adding a symbol beside the text.
    """

    for legend_text in legend.get_texts():
        label = legend_text.get_text()

        if label == "ERA5 fail":
            legend_text.set_color(ERA5_COLOR)

        elif label == f"{senorge_label} fail":
            legend_text.set_color(SENORGE_COLOR)


# =============================================================================
# Plot formatting
# =============================================================================

def apply_axis_formatting(
    ax: plt.Axes,
    month_label: str,
    row_index: int,
    column_index: int,
) -> None:
    """Apply consistent formatting to one panel."""

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
# Plotting
# =============================================================================

def plot_monthly_moments_test(
    results: dict[int, dict[str, object]],
    senorge_label: str,
) -> None:
    """Plot the monthly model bootstrap distributions and dataset values."""

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

        confidence_lower, confidence_upper = (
            month_results["confidence_interval"]
        )

        ax.axvline(
            confidence_lower,
            color=MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            zorder=2,
        )

        ax.axvline(
            confidence_upper,
            color=MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            zorder=2,
        )

        ax.axvline(
            month_results["era5_value"],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="-",
            zorder=3,
        )

        ax.axvline(
            month_results["senorge_value"],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle="-",
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

        if month_index == 0:
            legend_handles = make_full_legend_handles(
                senorge_label=senorge_label,
                month_results=month_results,
            )
        else:
            legend_handles = make_failure_legend_handles(
                month_results,
                senorge_label,
            )

        if legend_handles:
            legend = ax.legend(
                handles=legend_handles,
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

            color_failure_legend_text(
                legend=legend,
                senorge_label=senorge_label,
            )

    fig.suptitle(
        (
            f"{get_catchment_label()}: monthly "
            f"{STATISTIC_LABELS[statistic].lower()} moments test for "
            f"{x_days}-day accumulated precipitation maxima"
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
    senorge_label: str,
) -> None:
    """Print monthly moments-test results."""

    print()
    print("Monthly moments-test results")
    print("----------------------------")
    print(f"Statistic: {STATISTIC_LABELS[statistic]}")
    print(
        f"Bias correction for selected statistic: "
        f"{get_bias_correction_label()}"
    )
    print(
        f"Model bootstrap samples: {number_of_bootstrap_samples}"
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

        senorge_marker = (
            ""
            if month_results["senorge_passes"]
            else "*"
        )

        print(
            f"{MONTH_LABELS[month - 1]:>9s} | "
            f"n={month_results['sample_size']:>3d} | "
            f"model interval=[{format_statistic_value(lower)}, "
            f"{format_statistic_value(upper)}] | "
            f"ERA5={format_statistic_value(month_results['era5_value'])}"
            f"{era5_marker} | "
            f"{senorge_label}="
            f"{format_statistic_value(month_results['senorge_value'])}"
            f"{senorge_marker}"
        )


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    senorge_variable = get_senorge_variable()
    senorge_label = get_senorge_label()

    filename_model = make_model_filename()
    filename_era5 = make_era5_filename()
    filename_senorge = make_senorge_filename(
        senorge_variable
    )

    print("Reading input datasets")
    print("----------------------")
    print(f"Model:   {filename_model}")
    print(f"ERA5:    {filename_era5}")
    print(f"{senorge_label}: {filename_senorge}")

    model_ds, era5_ds, senorge_ds = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_senorge=filename_senorge,
    )

    try:
        model_values_by_month = get_model_values_by_month(
            model_ds
        )

        era5_values_by_month = get_dataset_values_by_month(
            ds=era5_ds,
            variable=ERA5_VARIABLE,
            dataset_name="ERA5 dataset",
        )

        senorge_values_by_month = get_dataset_values_by_month(
            ds=senorge_ds,
            variable=senorge_variable,
            dataset_name=f"{senorge_label} dataset",
        )

        check_monthly_samples(
            model_values_by_month,
            "model",
        )

        check_monthly_samples(
            era5_values_by_month,
            "ERA5",
        )

        check_monthly_samples(
            senorge_values_by_month,
            senorge_label,
        )

        check_era5_senorge_sample_sizes(
            era5_values_by_month=era5_values_by_month,
            senorge_values_by_month=senorge_values_by_month,
            senorge_label=senorge_label,
        )

        rng = np.random.default_rng(
            random_seed
        )

        results = perform_monthly_moments_test(
            model_values_by_month=model_values_by_month,
            era5_values_by_month=era5_values_by_month,
            senorge_values_by_month=senorge_values_by_month,
            rng=rng,
        )

        print_test_results(
            results=results,
            senorge_label=senorge_label,
        )

        plot_monthly_moments_test(
            results=results,
            senorge_label=senorge_label,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        senorge_ds.close()
