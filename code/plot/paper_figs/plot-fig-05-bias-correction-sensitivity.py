"""
Sensitivity of the Storm Hans return-period panels of fig-05 (c-d) to the model
bias-correction method.

Layout
------
    a) August return period for the Storm Hans threshold
    b) May return period for the Storm Hans threshold

For each extreme-value distribution (GEV, Gumbel, GenEx) the reference dataset
and every model dataset in MODEL_DATA_METHODS are shown side by side, each in
its own color. Each entry shows the original fitted return period as a dot and
the central CONFIDENCE_LEVEL bootstrap interval as a capped vertical line.

The reference fit uses OBSERVATION_YEARS, with INCLUDE_STORM_HANS_IN_FIT
controlling whether August 2023 is part of the August fit (no effect when 2023
is outside OBSERVATION_YEARS).

Infinite return-period estimates are retained when calculating the percentiles.
Intervals extending beyond the plotting limits end in outward arrows, not caps.
Off-scale fitted estimates use directional triangles at the corresponding limit.

Bootstrap seeds follow fig-05: the reference uses group index 0 and every model
dataset uses group index 1. The reference and mm_1step entries therefore
reproduce fig-05 panels c-d exactly for identical settings.

Model files that do not exist are skipped with a warning when
SKIP_MISSING_MODEL_FILES is True; otherwise a FileNotFoundError is raised.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r
from Dunnsigouin_etal_2026 import config

# =============================================================================
# USER INPUTS — data selection and analysis
# =============================================================================
# Data selection
REFERENCE_DATASET = "senorge"  # "senorge" or "era5"
CATCHMENT = "regine_drammen"
X_DAYS = 2
OBSERVATION_YEARS = [1957, 2025]  # Years used for the reference fit.
REFERENCE_FILE_YEARS = [1957, 2025]  # Year range encoded in the reference filename.
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

# Model datasets compared side by side (plotted in this order after the reference).
MODEL_DATA_METHODS = ["raw", "mm_1step", "q", "ld", "doy", "q_doy"]
SKIP_MISSING_MODEL_FILES = True

# Model processing and sampling
MODEL_VARIABLE = "tp24"
MODEL_SAMPLING_GROUP = "full"  # "full", "split1", "split2", ...
FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Affects only the August reference fit when 2023 is in OBSERVATION_YEARS.
INCLUDE_STORM_HANS_IN_FIT = True

# Bootstrap uncertainty
BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric"
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

# Optional reference input path; None uses the filename constructed above.
REFERENCE_FILENAME_OVERRIDE = None

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIG_WIDTH_IN = 14
FIG_HEIGHT_IN = 10
SHOW_GRID = True
RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e8

# =============================================================================
# FIGURE OUTPUT
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_DPI = 300
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = "fig-05-sensitivity-bc-methods.png"
OUTPUT_PATH = OUTPUT_DIRECTORY / OUTPUT_FILENAME

# =============================================================================
# Plot constants
# =============================================================================
MAY = 5
AUGUST = 8
PANEL_MONTHS = [AUGUST, MAY]  # Same order as fig-05, so bootstrap seeds match.
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST
SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
METHODS = ["GEV", "Gumbel", "GenEx"]
SUPPORTED_MODEL_METHODS = ["raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"]
OBSERVATION_COLOR = "tab:orange"
MODEL_COLORS = {
    "raw": "tab:blue",
    "mm_1step": "tab:cyan",
    "mm_2step": "tab:cyan",
    "q": "tab:green",
    "ld": "tab:red",
    "doy": "tab:purple",
    "q_doy": "tab:brown",
}
MODEL_LABELS = {
    "raw": "Model raw",
    "mm_1step": "BC mm_1step",
    "mm_2step": "BC mm_2step",
    "q": "BC q",
    "ld": "BC ld",
    "doy": "BC doy",
    "q_doy": "BC q_doy",
}
DATASET_SPACING = 0.11  # Separation between neighbouring entries at each tick.
INTERVAL_LINEWIDTH = 1.4
INTERVAL_CAP_WIDTH = 0.06
CLIPPED_ARROW_LENGTH = 14  # Points; fixed visual length on logarithmic axes.
CLIPPED_ARROW_HEAD_SIZE = 10
FITTED_MARKER_SIZE = 5
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 10

# =============================================================================
# Validation, labels, and filenames
# =============================================================================


def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    unsupported = set(MODEL_DATA_METHODS) - set(SUPPORTED_MODEL_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported MODEL_DATA_METHODS: {sorted(unsupported)}")
    if len(set(MODEL_DATA_METHODS)) != len(MODEL_DATA_METHODS):
        raise ValueError("MODEL_DATA_METHODS contains duplicates.")
    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if RETURN_PERIOD_MIN < 1:
        raise ValueError("RETURN_PERIOD_MIN must be at least 1.")
    if RETURN_PERIOD_MAX <= RETURN_PERIOD_MIN:
        raise ValueError("RETURN_PERIOD_MAX must exceed RETURN_PERIOD_MIN.")
    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError("NUMBER_OF_BOOTSTRAPS must be at least 1.")
    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must lie between 0 and 1.")
    if not 0 < MIN_SUCCESSFUL_BOOTSTRAP_FRACTION <= 1:
        raise ValueError("MIN_SUCCESSFUL_BOOTSTRAP_FRACTION must lie in (0, 1].")
    if not isinstance(INCLUDE_STORM_HANS_IN_FIT, bool):
        raise TypeError("INCLUDE_STORM_HANS_IN_FIT must be True or False.")
    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    number_of_usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1
    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError("X_DAYS is too large for the configured lead range.")
    if not 1 <= NUMBER_OF_LEAD_BINS <= number_of_usable_leads:
        raise ValueError("NUMBER_OF_LEAD_BINS is invalid for the usable lead range.")
    valid_groups = {"full", *(f"split{i}" for i in range(1, NUMBER_OF_LEAD_BINS + 1))}
    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}.")


def get_reference_name():
    """Return the display name of the selected reference dataset."""
    return {"senorge": "seNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_model_file_id(catchment_name):
    """Return the short catchment identifier used in model filenames."""
    return catchment_name.removeprefix("regine_")


def split_usable_leads(first_lead, last_lead, number_of_bins):
    """Split an inclusive lead range into near-equal consecutive bins."""
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


def build_lead_bins():
    """Return configured accumulated ending-lead bins."""
    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    return split_usable_leads(first_usable_lead, LAST_INPUT_LEAD, NUMBER_OF_LEAD_BINS)


def get_model_variable():
    """Return the compact model precipitation variable to read."""
    if MODEL_SAMPLING_GROUP == "full":
        return "tp24_max"
    lead_start, lead_end = build_lead_bins()[int(MODEL_SAMPLING_GROUP.removeprefix("split")) - 1]
    return f"tp24_max_lead{lead_start}_{lead_end}"


def make_reference_filename():
    """Construct the selected observational input filename."""
    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)
    first_year, last_year = REFERENCE_FILE_YEARS
    variable = get_reference_variable()
    filename = (
        f"monthly_max_samples_{variable}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_{first_year}-{last_year}.nc"
    )
    directory = "senorge_processed" if REFERENCE_DATASET == "senorge" else "era5_processed"
    return Path(config.dirs[directory]) / filename


def make_model_filename(model_data_method):
    """Construct the compact model filename for one model-data method."""
    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    if model_data_method == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{model_data_method}_{REFERENCE_DATASET}_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
        )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction_label}.nc"


def build_datasets():
    """Return the plotted datasets, dropping missing model files if requested."""
    datasets = [
        {"key": "reference", "label": get_reference_name(),
         "color": OBSERVATION_COLOR, "seed_group": 0}
    ]
    for model_data_method in MODEL_DATA_METHODS:
        filename = make_model_filename(model_data_method)
        if not filename.is_file():
            if SKIP_MISSING_MODEL_FILES:
                print(f"Warning: skipping '{model_data_method}', file not found: {filename}")
                continue
            raise FileNotFoundError(f"Model file not found: {filename}")
        datasets.append(
            {
                "key": model_data_method,
                "label": MODEL_LABELS[model_data_method],
                "color": MODEL_COLORS[model_data_method],
                "seed_group": 1,  # Matches the model seed used in fig-05.
            }
        )
    return datasets

# =============================================================================
# Data reading
# =============================================================================


def read_reference_month(month):
    """Read the reference-month fit sample and the Storm Hans threshold."""
    filename = make_reference_filename()
    variable = get_reference_variable()
    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")
    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")
        selected = ds[variable].sel(
            year=slice(OBSERVATION_YEARS[0], OBSERVATION_YEARS[1]), month=month
        ).load()
        storm_hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )
    years = np.asarray(selected["year"].values)
    values = np.asarray(selected.values, dtype=float)
    finite = np.isfinite(values)
    years, values = years[finite], values[finite]
    hans_in_observation_range = OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
    fit_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST and hans_in_observation_range and not INCLUDE_STORM_HANS_IN_FIT:
        fit_mask &= years != STORM_HANS_YEAR
    fit_values = values[fit_mask]
    if fit_values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain in the fit."
        )
    return {"fit_values": fit_values, "storm_hans_value": storm_hans_value}


def read_model_month(month, model_data_method):
    """Read one calendar-month model sample from sample_month(YYYYMM)."""
    filename = make_model_filename(model_data_method)
    variable = get_model_variable()
    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        if "sample_month" not in ds:
            raise KeyError(f"Variable 'sample_month' was not found in {filename}.")
        if set(ds[variable].dims) != {"number", "i_date"}:
            raise ValueError(
                f"'{variable}' must have dimensions ('number', 'i_date'); "
                f"found {ds[variable].dims}."
            )
        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("'sample_month' must have dimension ('i_date',).")
        calendar_month = ds["sample_month"] % 100
        values = np.asarray(
            ds[variable].where(calendar_month == month, drop=True).values, dtype=float
        ).ravel()
    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite '{model_data_method}' values were found for month {month}."
        )
    return values

# =============================================================================
# Distribution methods
# =============================================================================


def genex_negative_log_likelihood(log_parameters, values):
    """Return the GenEx negative log-likelihood."""
    shape, scale = np.exp(log_parameters)
    if shape <= 0 or scale <= 0 or np.any(values < 0):
        return np.inf
    z = values / scale
    log_pdf = np.log(shape) - np.log(scale) - z + (shape - 1.0) * np.log(-np.expm1(-z))
    return np.inf if not np.isfinite(log_pdf).all() else -np.sum(log_pdf)


def fit_distribution(values, method, initial_parameters=None):
    """Fit one supported extreme-value distribution."""
    if method == "GEV":
        parameters = genextreme.fit(values)
    elif method == "Gumbel":
        parameters = gumbel_r.fit(values)
    elif method == "GenEx":
        positive = values[values > 0]
        if np.any(values < 0) or positive.size == 0:
            raise ValueError("GenEx requires non-negative values with at least one positive value.")
        initial_parameters = (
            initial_parameters
            if initial_parameters is not None
            else (1.0, np.mean(positive))
        )
        result = minimize(
            genex_negative_log_likelihood,
            x0=np.log(initial_parameters),
            args=(values,),
            method="Nelder-Mead",
            options={"maxiter": 5000},
        )
        if not result.success:
            raise RuntimeError(f"GenEx fit failed: {result.message}")
        parameters = tuple(np.exp(result.x))
    else:
        raise ValueError(f"Unsupported distribution: {method}")
    if not np.isfinite(parameters).all() or parameters[-1] <= 0:
        raise RuntimeError(f"{method} fit returned invalid parameters.")
    return parameters


def exceedance_probability(event_value, parameters, method):
    """Return the fitted probability of exceeding one event value."""
    if method == "GEV":
        shape, location, scale = parameters
        probability = genextreme.sf(event_value, shape, loc=location, scale=scale)
    elif method == "Gumbel":
        location, scale = parameters
        probability = gumbel_r.sf(event_value, loc=location, scale=scale)
    else:
        shape, scale = parameters
        probability = (
            1.0 if event_value < 0
            else 1.0 - (1.0 - np.exp(-event_value / scale)) ** shape
        )
    if not np.isfinite(probability):
        raise RuntimeError(f"{method} produced a non-finite exceedance probability.")
    return float(np.clip(probability, 0.0, 1.0))


def simulate_distribution(parameters, method, sample_size, rng):
    """Simulate from one fitted distribution."""
    if method == "GEV":
        shape, location, scale = parameters
        return genextreme.rvs(shape, loc=location, scale=scale, size=sample_size, random_state=rng)
    if method == "Gumbel":
        location, scale = parameters
        return gumbel_r.rvs(loc=location, scale=scale, size=sample_size, random_state=rng)
    shape, scale = parameters
    probabilities = rng.random(sample_size)
    return -scale * np.log1p(-np.power(probabilities, 1.0 / shape))


def make_bootstrap_sample(values, method, rng, fitted_parameters=None):
    """Create one nonparametric or parametric bootstrap sample."""
    if BOOTSTRAP_METHOD == "nonparametric":
        return rng.choice(values, size=values.size, replace=True)
    if fitted_parameters is None:
        raise ValueError("Parametric bootstrap requires fitted parameters.")
    return simulate_distribution(fitted_parameters, method, values.size, rng)


def return_period_from_probability(probability):
    """Convert annual exceedance probability to return period."""
    return np.inf if probability <= 0 else 1.0 / probability

# =============================================================================
# Bootstrap analyses
# =============================================================================


class ProgressTracker:
    """Print integer percentage completion for the requested bootstrap fits."""

    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.last_percent = -1

    def update(self):
        """Advance one bootstrap fit and print when the integer percentage changes."""
        self.completed += 1
        percent = min(100, int(100 * self.completed / self.total))
        if percent != self.last_percent:
            print(f"Progress: {percent:3d}%", end="\r", flush=True)
            self.last_percent = percent
        if self.completed == self.total:
            print()


def bootstrap_distribution(values, method, random_seed, progress=None):
    """Fit one distribution and create bootstrap parameter sets."""
    parameters = fit_distribution(values, method)
    rng = np.random.default_rng(random_seed)
    bootstrap_parameters = []
    for _ in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = make_bootstrap_sample(values, method, rng, fitted_parameters=parameters)
            fitted = fit_distribution(
                sample,
                method,
                initial_parameters=parameters if method == "GenEx" else None,
            )
            bootstrap_parameters.append(fitted)
        except (RuntimeError, ValueError, FloatingPointError):
            pass
        finally:
            if progress is not None:
                progress.update()
    minimum = int(np.ceil(MIN_SUCCESSFUL_BOOTSTRAP_FRACTION * NUMBER_OF_BOOTSTRAPS))
    if len(bootstrap_parameters) < minimum:
        raise RuntimeError(
            f"Only {len(bootstrap_parameters)} of {NUMBER_OF_BOOTSTRAPS} "
            f"{method} bootstrap fits succeeded."
        )
    return {"parameters": parameters, "bootstrap_parameters": bootstrap_parameters}


def analyse_event_return_period(fit, event_value, method):
    """Evaluate the Storm Hans return period for one fit and its bootstraps."""
    probability = exceedance_probability(event_value, fit["parameters"], method)
    bootstrap_return_periods = np.array(
        [
            return_period_from_probability(exceedance_probability(event_value, parameters, method))
            for parameters in fit["bootstrap_parameters"]
        ],
        dtype=float,
    )
    bootstrap_return_periods = bootstrap_return_periods[~np.isnan(bootstrap_return_periods)]
    if bootstrap_return_periods.size == 0:
        raise RuntimeError(f"{method} produced no valid bootstrap return periods.")
    return {
        "return_period": return_period_from_probability(probability),
        "metric_samples": bootstrap_return_periods,
    }


def analyse_month(month, month_index, datasets, progress=None):
    """Fit every dataset and distribution for one month at the Storm Hans threshold."""
    reference = read_reference_month(month)
    event_value = reference["storm_hans_value"]
    results = {}
    sample_sizes = {}
    for dataset in datasets:
        if dataset["key"] == "reference":
            values = reference["fit_values"]
        else:
            values = read_model_month(month, dataset["key"])
        sample_sizes[dataset["key"]] = values.size
        for method_index, method in enumerate(METHODS):
            seed = (
                RANDOM_SEED + 10000 * month_index
                + 1000 * dataset["seed_group"] + method_index
            )
            fit = bootstrap_distribution(values, method, seed, progress)
            results[(dataset["key"], method)] = analyse_event_return_period(
                fit, event_value, method
            )
    return {
        "month": month,
        "event_value": event_value,
        "sample_sizes": sample_sizes,
        "results": results,
    }

# =============================================================================
# Plotting
# =============================================================================


def format_metric_return_period(value, _position):
    """Format return periods as powers of ten and mark the configured upper limit."""
    if value <= 0:
        return ""
    exponent = np.log10(value)
    rounded_exponent = int(np.round(exponent))
    if not np.isclose(exponent, rounded_exponent, atol=1e-10):
        return ""
    if np.isclose(value, RETURN_PERIOD_MAX):
        return rf"$>10^{{{rounded_exponent}}}$"
    return rf"$10^{{{rounded_exponent}}}$"


def configure_metric_axis(axis):
    """Apply common formatting to a return-period panel."""
    axis.set_yscale("log")
    axis.set_ylim(RETURN_PERIOD_MIN, 1.15 * RETURN_PERIOD_MAX)
    axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))
    axis.set_xlim(-0.5, len(METHODS) - 0.5)
    axis.set_xticks(range(len(METHODS)))
    axis.set_xticklabels(METHODS)
    axis.set_ylabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    axis.set_xlabel("Extreme-value distribution", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for boundary in np.arange(0.5, len(METHODS) - 0.5):
        axis.axvline(boundary, color="lightgrey", linewidth=0.8, zorder=0)
    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.7, alpha=0.45)


def percentile_preserving_infinity(values, percentile):
    """Calculate a percentile without discarding positive infinity."""
    values = np.sort(np.asarray(values, dtype=float))
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValueError("Cannot calculate a percentile from an empty sample.")
    position = (values.size - 1) * percentile / 100.0
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    lower = values[lower_index]
    upper = values[upper_index]
    if lower_index == upper_index:
        return float(lower)
    if np.isposinf(upper):
        return np.inf
    weight = position - lower_index
    return float(lower + weight * (upper - lower))


def clip_metric(value):
    """Clip one plotted return period to the shared limits."""
    return float(np.clip(value, RETURN_PERIOD_MIN, RETURN_PERIOD_MAX))


def plot_metric_interval(axis, position, lower, upper, fitted, color):
    """Show clipped intervals with arrows and off-scale estimates with triangles."""
    minimum, maximum = RETURN_PERIOD_MIN, RETURN_PERIOD_MAX
    axis.vlines(
        position, clip_metric(lower), clip_metric(upper),
        color=color, linewidth=INTERVAL_LINEWIDTH,
    )
    visible_endpoints = [value for value in (lower, upper) if minimum <= value <= maximum]
    if visible_endpoints:
        axis.hlines(
            visible_endpoints,
            position - INTERVAL_CAP_WIDTH / 2, position + INTERVAL_CAP_WIDTH / 2,
            color=color, linewidth=INTERVAL_LINEWIDTH,
        )
    # Arrow tails are inside the displayed range; arrowheads point outwards.
    for clipped, boundary, direction in [
        (upper > maximum, maximum, 1),
        (lower < minimum, minimum, -1),
    ]:
        if clipped:
            axis.annotate(
                "", xy=(position, boundary), xycoords="data",
                xytext=(0, -direction * CLIPPED_ARROW_LENGTH), textcoords="offset points",
                arrowprops={
                    "arrowstyle": "-|>", "color": color,
                    "lw": INTERVAL_LINEWIDTH, "mutation_scale": CLIPPED_ARROW_HEAD_SIZE,
                    "shrinkA": 0, "shrinkB": 0,
                },
                annotation_clip=False, zorder=4,
            )
    if not np.isnan(fitted):
        marker = "^" if fitted > maximum else "v" if fitted < minimum else "o"
        axis.plot(
            position, clip_metric(fitted), marker=marker, linestyle="none", color=color,
            markersize=FITTED_MARKER_SIZE, clip_on=False, zorder=5,
        )


def dataset_offsets(number_of_datasets):
    """Return symmetric x offsets around each distribution tick."""
    return (np.arange(number_of_datasets) - (number_of_datasets - 1) / 2) * DATASET_SPACING


def plot_month_panel(axis, panel_label, panel_output, datasets):
    """Plot fitted dots and bootstrap intervals for all datasets in one month."""
    tail_percent = 100 * (1 - CONFIDENCE_LEVEL) / 2
    offsets = dataset_offsets(len(datasets))
    for method_index, method in enumerate(METHODS):
        for dataset, offset in zip(datasets, offsets):
            analysis = panel_output["results"][(dataset["key"], method)]
            lower = percentile_preserving_infinity(analysis["metric_samples"], tail_percent)
            upper = percentile_preserving_infinity(analysis["metric_samples"], 100 - tail_percent)
            plot_metric_interval(
                axis, method_index + offset, lower, upper,
                analysis["return_period"], dataset["color"],
            )
    configure_metric_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[panel_output['month'] - 1]}: Storm Hans threshold "
        f"({panel_output['event_value']:.1f} mm)",
        loc="center", fontsize=TITLE_FONTSIZE, fontweight="normal",
    )


def make_figure(month_results, datasets):
    """Create the two-row sensitivity figure (August top, May bottom)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": TICK_LABELSIZE,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True
    )
    plot_month_panel(axes[0], "a", month_results[AUGUST], datasets)
    plot_month_panel(axes[1], "b", month_results[MAY], datasets)
    handles = [
        Line2D([0], [0], linestyle="-", marker="o", color=dataset["color"],
               linewidth=INTERVAL_LINEWIDTH, label=dataset["label"])
        for dataset in datasets
    ]
    figure.legend(
        handles=handles, loc="outside upper center", ncol=len(datasets),
        frameon=False, fontsize=LEGEND_FONTSIZE,
    )
    if WRITE_TO_FILE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PATH, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        print("Wrote:", OUTPUT_PATH)
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Reporting
# =============================================================================


def interval_width_ratio(reference_width, model_width):
    """Return reference/model interval width, preserving undefined and unbounded cases."""
    if np.isnan(reference_width) or np.isnan(model_width):
        return np.nan
    if np.isinf(reference_width) and np.isinf(model_width):
        return np.nan
    if model_width == 0:
        return np.nan if reference_width == 0 else np.inf
    return reference_width / model_width


def print_summary(month_results, datasets):
    """Print settings, sample sizes, and return-period intervals for every entry."""
    tail_percent = 100 * (1 - CONFIDENCE_LEVEL) / 2
    reference_name = get_reference_name()
    print("\nSelected settings")
    print("-----------------")
    print(f"Reference dataset: {reference_name}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Model datasets:    {[d['key'] for d in datasets if d['key'] != 'reference']}")
    print(f"Bootstrap method:  {BOOTSTRAP_METHOD}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS}")
    print(f"Include Hans fit:  {INCLUDE_STORM_HANS_IN_FIT}")
    print(f"\n{CONFIDENCE_LEVEL:.0%} bootstrap return-period intervals [years]")
    print(f"Factor = {reference_name} interval width / dataset interval width.")
    for month in PANEL_MONTHS:
        panel = month_results[month]
        print(
            f"\n{MONTH_NAMES[month - 1]}: Storm Hans threshold ({panel['event_value']:.3f} mm)"
        )
        print(
            f"{'Distribution':<13} {'Dataset':<13} {'N':>7} {'Fitted':>12} {'Lower':>12} "
            f"{'Upper':>12} {'Width':>12} {'Factor':>12}"
        )
        for method in METHODS:
            reference_width = None
            for dataset in datasets:
                analysis = panel["results"][(dataset["key"], method)]
                lower = percentile_preserving_infinity(analysis["metric_samples"], tail_percent)
                upper = percentile_preserving_infinity(
                    analysis["metric_samples"], 100 - tail_percent
                )
                width = np.nan if lower == upper == np.inf else upper - lower
                if dataset["key"] == "reference":
                    reference_width = width
                    factor_text = ""
                else:
                    factor = interval_width_ratio(reference_width, width)
                    factor_text = "undefined" if np.isnan(factor) else f"{factor:.4g}x"
                print(
                    f"{method:<13} {dataset['label']:<13} "
                    f"{panel['sample_sizes'][dataset['key']]:>7d} "
                    f"{analysis['return_period']:12.5g} {lower:12.5g} {upper:12.5g} "
                    f"{width:12.5g} {factor_text:>12}"
                )
    print("\ninf = unbounded; nan/undefined = no defined width or ratio.")

# =============================================================================
# Main
# =============================================================================


def main():
    """Run the bias-correction sensitivity analysis and plot both months."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)
    datasets = build_datasets()
    total_bootstraps = len(PANEL_MONTHS) * len(datasets) * len(METHODS) * NUMBER_OF_BOOTSTRAPS
    progress = ProgressTracker(total_bootstraps)
    print("Running bootstrap fits...")
    print("Progress:   0%", end="\r", flush=True)
    month_results = {
        month: analyse_month(month, index, datasets, progress)
        for index, month in enumerate(PANEL_MONTHS)
    }
    print_summary(month_results, datasets)
    make_figure(month_results, datasets)


if __name__ == "__main__":
    main()
