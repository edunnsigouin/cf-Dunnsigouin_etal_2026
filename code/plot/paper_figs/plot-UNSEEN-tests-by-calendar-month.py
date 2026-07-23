"""
Create a five-panel monthly diagnostic figure for one selected calendar month.

Panel 1
-------
Ensemble-member independence, following script 1:
forecast and hindcast pairwise Spearman correlations are pooled and shown as
one boxplot for the selected month.

Panels 2-5
----------
UNSEEN moments-based fidelity tests, following script 2:
    2. mean
    3. standard deviation
    4. skewness
    5. kurtosis

Panel 6
-------
Model, ERA5, and SeNorge precipitation distributions with two-sample
Kolmogorov-Smirnov tests comparing the model with each reference dataset.

For each moment, the model is resampled with replacement using the same sample
size as ERA5/SeNorge. The model bootstrap distribution, its central confidence
interval, and the ERA5/SeNorge values are plotted.

The data-selection layer is kept separate from the analysis. This makes it
straightforward to add alternative model inputs later, for example:
    - leads 17-30;
    - leads 31-46;
    - bias-corrected model data.

Only the current input-file conventions from scripts 1 and 2 are configured below.
The same catchment is used for all datasets.
"""

from dataclasses import dataclass
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
senorge_dataset = "senorge"  # "senorge" or "senorge_regrid"

# Select which model input configuration to use.
# Add future lead-window or bias-corrected datasets in MODEL_INPUT_VARIANTS.
model_input_variant = "current_raw"

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

# Finite-sample correction used for shape statistics.
bias_correct_skewness = False
bias_correct_kurtosis = False

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


@dataclass(frozen=True)
class ModelInputVariant:
    """
    Describe one pair of model input files.

    For the current dataset, filenames are built exactly as in scripts 1 and 2.
    For future datasets, set explicit filename overrides without changing any
    of the loading, analysis, bootstrap, or plotting functions.
    """

    label: str

    # Settings used by the current independence filename convention.
    first_input_lead: int = 16
    last_input_lead: int = 46

    # Optional full-path overrides for future datasets.
    independence_filename: str | None = None
    moments_filename: str | None = None


MODEL_INPUT_VARIANTS = {
    "current_raw": ModelInputVariant(
        label="Current raw model input",
        first_input_lead=16,
        last_input_lead=46,
    ),

    # Future examples:
    #
    # "raw_lead17_30": ModelInputVariant(
    #     label="Raw model, leads 17-30",
    #     independence_filename="/path/to/independence_17_30.nc",
    #     moments_filename="/path/to/monthly_extremes_17_30.nc",
    # ),
    #
    # "raw_lead31_46": ModelInputVariant(
    #     label="Raw model, leads 31-46",
    #     independence_filename="/path/to/independence_31_46.nc",
    #     moments_filename="/path/to/monthly_extremes_31_46.nc",
    # ),
    #
    # "bias_corrected_lead17_46": ModelInputVariant(
    #     label="Bias-corrected model, leads 17-46",
    #     independence_filename="/path/to/bc_independence_17_46.nc",
    #     moments_filename="/path/to/bc_monthly_extremes_17_46.nc",
    # ),
}


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


def get_senorge_variable() -> str:
    """Return the precipitation variable for the selected SeNorge product."""

    if senorge_dataset not in SENORGE_VARIABLES:
        raise ValueError(
            f"Unknown senorge_dataset '{senorge_dataset}'. "
            f"Choose from {sorted(SENORGE_VARIABLES)}."
        )

    return SENORGE_VARIABLES[senorge_dataset]


def get_senorge_label() -> str:
    """Return a readable label for the selected SeNorge product."""

    if senorge_dataset not in SENORGE_LABELS:
        raise ValueError(
            f"Unknown senorge_dataset '{senorge_dataset}'. "
            f"Choose from {sorted(SENORGE_LABELS)}."
        )

    return SENORGE_LABELS[senorge_dataset]


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

    if model_input_variant not in MODEL_INPUT_VARIANTS:
        raise ValueError(
            f"Unknown model_input_variant '{model_input_variant}'. "
            f"Choose from {sorted(MODEL_INPUT_VARIANTS)}."
        )

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


# =============================================================================
# Filename helpers
# =============================================================================

def build_current_independence_filename(
    variant: ModelInputVariant,
) -> str:
    """Build the independence filename exactly as in script 1."""

    first_usable_accumulation_lead = (
        variant.first_input_lead + x_days - 1
    )

    return (
        config.dirs["s2s_processed"]
        + f"independence_spearman_monthly_max_{MODEL_VARIABLE}_"
        + f"{x_days}dayacc_"
        + f"nve_catchment_{catchment}_"
        + f"lead{first_usable_accumulation_lead}-"
        + f"{variant.last_input_lead}_"
        + f"{forecast_date_range[0]}_"
        + f"{forecast_date_range[1]}.nc"
    )


def build_current_moments_model_filename() -> str:
    """Build the model-extremes filename exactly as in script 2."""

    return (
        f"{config.dirs['s2s_processed']}"
        f"distribution_monthly_extremes_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )


def resolve_model_input_filenames() -> tuple[str, str]:
    """
    Return independence and moments files for the selected model variant.

    Future variants can provide explicit paths in MODEL_INPUT_VARIANTS.
    The rest of the script does not need to know which variant is active.
    """

    variant = MODEL_INPUT_VARIANTS[model_input_variant]

    independence_filename = (
        variant.independence_filename
        if variant.independence_filename is not None
        else build_current_independence_filename(variant)
    )

    moments_filename = (
        variant.moments_filename
        if variant.moments_filename is not None
        else build_current_moments_model_filename()
    )

    return independence_filename, moments_filename


def build_era5_filename() -> str:
    """Build the ERA5 filename exactly as in script 2."""

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def build_senorge_filename(senorge_variable: str) -> str:
    """Build the SeNorge filename exactly as in script 2."""

    return (
        f"{config.dirs[f'{senorge_dataset}_processed']}"
        f"distribution_monthly_extremes_{senorge_variable}_{x_days}dayacc_"
        f"{catchment}_{senorge_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def build_output_filename() -> str:
    """Create a descriptive filename for the combined five-panel figure."""

    month_name = MONTH_LABELS[selected_month].lower()

    return os.path.join(
        config.dirs["fig"],
        (
            f"monthly_combined_diagnostics_{month_name}_"
            f"{x_days}dayacc_{model_input_variant}_"
            f"{catchment}.png"
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

        if "assigned_month" not in ds.coords:
            raise KeyError(
                "Independence file has no 'assigned_month' coordinate."
            )

        forecast = remove_missing_values(
            ds["forecast_spearman_rho"]
            .sel(assigned_month=selected_month)
            .values
        )

        hindcast = remove_missing_values(
            ds["hindcast_spearman_rho"]
            .sel(assigned_month=selected_month)
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
) -> np.ndarray:
    """Extract model monthly maxima for the selected month."""

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

    values = data.sel(month=selected_month).values
    return remove_missing_values(values)


def load_moments_values(
    model_filename: str,
    era5_filename: str,
    senorge_filename: str,
    senorge_variable: str,
    senorge_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model, ERA5, and SeNorge values for the selected month."""

    for dataset_name, filename in (
        ("model", model_filename),
        ("ERA5", era5_filename),
        (senorge_label, senorge_filename),
    ):
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{dataset_name} input file does not exist:\n{filename}"
            )

    with (
        xr.open_dataset(model_filename) as model_ds,
        xr.open_dataset(era5_filename) as era5_ds,
        xr.open_dataset(senorge_filename) as senorge_ds,
    ):
        model_values = get_model_values_for_selected_month(model_ds)

        era5_values = get_reference_values_for_selected_month(
            ds=era5_ds,
            variable=ERA5_VARIABLE,
            dataset_name="ERA5 dataset",
        )

        senorge_values = get_reference_values_for_selected_month(
            ds=senorge_ds,
            variable=senorge_variable,
            dataset_name=f"{senorge_label} dataset",
        )

    return model_values, era5_values, senorge_values


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
                bias=not bias_correct_skewness,
            )
        )

    if statistic_name == "kurtosis":
        return float(
            kurtosis(
                values,
                fisher=True,
                bias=not bias_correct_kurtosis,
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
            bias=not bias_correct_skewness,
        )

    if statistic_name == "kurtosis":
        return lambda samples: kurtosis(
            samples,
            axis=1,
            fisher=True,
            bias=not bias_correct_kurtosis,
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
# Kolmogorov-Smirnov distribution test
# =============================================================================

def perform_ks_test(
    model_values: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[float, float]:
    """
    Compare the full model and reference distributions with a two-sample KS test.

    The null hypothesis is that both samples come from the same continuous
    distribution.
    """

    result = ks_2samp(
        model_values,
        reference_values,
        alternative=ks_alternative,
        method=ks_method,
    )

    return float(result.statistic), float(result.pvalue)


def get_ks_significance_threshold() -> float:
    """Convert the selected KS confidence level to a p-value threshold."""

    return 1.0 - ks_significance_level_percent / 100.0


def ks_test_failed(p_value: float) -> bool:
    """Return True when equal model/reference distributions are rejected."""

    return p_value < get_ks_significance_threshold()


def format_ks_p_value(p_value: float) -> str:
    """Format a KS p-value and append an asterisk when the test fails."""

    if p_value < 0.001:
        text = f"{p_value:.1e}"
    else:
        text = f"{p_value:.3f}"

    if ks_test_failed(p_value):
        text += "*"

    return text


def perform_distribution_tests(
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
) -> dict[str, float]:
    """Run the two selected-month KS tests used in panel 6."""

    era5_statistic, era5_p_value = perform_ks_test(
        model_values=model_values,
        reference_values=era5_values,
    )

    senorge_statistic, senorge_p_value = perform_ks_test(
        model_values=model_values,
        reference_values=senorge_values,
    )

    return {
        "era5_statistic": era5_statistic,
        "era5_p_value": era5_p_value,
        "senorge_statistic": senorge_statistic,
        "senorge_p_value": senorge_p_value,
    }


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
        showfliers=True,
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
    """Create one legend used by the complete six-panel figure."""

    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model",
        ),
        Line2D(
            [0],
            [0],
            color=ERA5_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            label="ERA5",
        ),
        Line2D(
            [0],
            [0],
            color=SENORGE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            label=senorge_label,
        ),
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            linewidth=CONFIDENCE_LINEWIDTH,
            linestyle="--",
            label=f"{confidence_level_percent:g}% model interval",
        ),
    ]


def add_ks_failure_text(
    ax: plt.Axes,
    ks_results: dict[str, float],
    senorge_label: str,
) -> None:
    """
    Add a simple label only when a KS comparison rejects equal distributions.
    """

    failure_lines = []

    if ks_test_failed(ks_results["era5_p_value"]):
        failure_lines.append(("ERA5 fail", ERA5_COLOR))

    if ks_test_failed(ks_results["senorge_p_value"]):
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


def plot_distribution_panel(
    ax: plt.Axes,
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
    senorge_label: str,
    ks_results: dict[str, float],
) -> None:
    """
    Plot the three selected-month precipitation distributions and KS results.
    """

    bin_edges = calculate_distribution_bin_edges(
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
    )

    maximum_density = 0.0

    for values, color, zorder in (
        (model_values, MODEL_COLOR, 3),
        (senorge_values, SENORGE_COLOR, 2),
        (era5_values, ERA5_COLOR, 1),
    ):
        counts, _, _ = ax.hist(
            values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=color,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=zorder,
        )

        if counts.size > 0:
            maximum_density = max(
                maximum_density,
                float(np.nanmax(counts)),
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
        "Fidelity: Kolmolgorov-Smirnov",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    format_axis(ax)

    add_ks_failure_text(
        ax=ax,
        ks_results=ks_results,
        senorge_label=senorge_label,
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
    variant_label = MODEL_INPUT_VARIANTS[model_input_variant].label

    return (
        f"{month_name}: {x_days}-day accumulated precipitation maxima\n"
        f"{catchment_name} catchment | {variant_label}"
    )


def create_combined_figure(
    independence_values: np.ndarray,
    moments_results: dict[str, dict[str, object]],
    model_values: np.ndarray,
    era5_values: np.ndarray,
    senorge_values: np.ndarray,
    senorge_label: str,
    ks_results: dict[str, float],
) -> plt.Figure:
    """
    Create the publication-style 2 x 3 diagnostic figure.

    Layout:
        (a) Independence      (b) Mean              (c) Standard deviation
        (d) Skewness          (e) Excess kurtosis   (f) KS test
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

    # (f) Full-distribution comparison and KS test.
    plot_distribution_panel(
        ax=axes[1, 2],
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        senorge_label=senorge_label,
        ks_results=ks_results,
    )
    add_panel_label(axes[1, 2], "(f)")

    fig.suptitle(
        build_figure_title(),
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="normal",
        y=0.985,
    )

    # One legend applies to all six panels.
    fig.legend(
        handles=make_shared_legend_handles(senorge_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
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



def print_ks_results(
    ks_results: dict[str, float],
    senorge_label: str,
) -> None:
    """Print the selected-month KS distribution-test results."""

    alpha = get_ks_significance_threshold()

    era5_marker = "*" if ks_test_failed(
        ks_results["era5_p_value"]
    ) else ""

    senorge_marker = "*" if ks_test_failed(
        ks_results["senorge_p_value"]
    ) else ""

    print()
    print(f"{MONTH_LABELS[selected_month]} distribution KS tests")
    print("-" * 45)
    print(
        f"Failure threshold: p < {alpha:.4f} "
        f"({ks_significance_level_percent:g}% confidence)"
    )
    print(
        f"ERA5: D={ks_results['era5_statistic']:.3f}, "
        f"p={ks_results['era5_p_value']:.4g}{era5_marker}"
    )
    print(
        f"{senorge_label}: D={ks_results['senorge_statistic']:.3f}, "
        f"p={ks_results['senorge_p_value']:.4g}{senorge_marker}"
    )
    print("* model and reference distributions differ significantly")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    senorge_variable = get_senorge_variable()
    senorge_label = get_senorge_label()

    (
        independence_filename,
        moments_model_filename,
    ) = resolve_model_input_filenames()

    era5_filename = build_era5_filename()
    senorge_filename = build_senorge_filename(
        senorge_variable
    )
    output_filename = build_output_filename()

    print("Selected month")
    print("--------------")
    print(MONTH_LABELS[selected_month])

    print()
    print("Input files")
    print("-----------")
    print(f"Independence model: {independence_filename}")
    print(f"Moments model:      {moments_model_filename}")
    print(f"ERA5:               {era5_filename}")
    print(f"{senorge_label}:".ljust(20), senorge_filename)

    independence_values = load_independence_values(
        independence_filename
    )

    (
        model_values,
        era5_values,
        senorge_values,
    ) = load_moments_values(
        model_filename=moments_model_filename,
        era5_filename=era5_filename,
        senorge_filename=senorge_filename,
        senorge_variable=senorge_variable,
        senorge_label=senorge_label,
    )

    validate_moments_samples(
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        senorge_label=senorge_label,
    )

    rng = np.random.default_rng(random_seed)

    moments_results = perform_all_moments_tests(
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        rng=rng,
    )

    ks_results = perform_distribution_tests(
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
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

    print_ks_results(
        ks_results=ks_results,
        senorge_label=senorge_label,
    )

    figure = create_combined_figure(
        independence_values=independence_values,
        moments_results=moments_results,
        model_values=model_values,
        era5_values=era5_values,
        senorge_values=senorge_values,
        senorge_label=senorge_label,
        ks_results=ks_results,
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
