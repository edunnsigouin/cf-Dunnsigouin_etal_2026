"""
Create a 2 x 2 extreme-precipitation figure for August and May.

Layout
------
    a) August return-level distribution
    b) May return-level distribution
    c) August return period for the Storm Hans threshold
    d) May return period for the Storm Hans threshold

Reference and model samples
---------------------------
The reference fit uses OBSERVATION_YEARS from the 1957-2025 reference file, with
INCLUDE_STORM_HANS_IN_FIT controlling whether August 2023 (Storm Hans) is part
of the August fit (no effect when 2023 is outside OBSERVATION_YEARS). The model
sample is the complete-lead monthly-maximum sample (tp24_max) of the selected
MODEL_DATA_METHOD; all finite values for a calendar month are pooled.

Panels (a)-(b) show the fitted TOP_DISTRIBUTION return-level curves with
CONFIDENCE_LEVEL bootstrap bands, Weibull empirical points ((n + 1) / rank) and
the Storm Hans threshold as an orange dashed line labelled with the reference
name (e.g. "seNorge Storm Hans threshold").

Panels (c)-(d) show, for each distribution, the fitted return period of the
Storm Hans threshold as a dot and the central CONFIDENCE_LEVEL bootstrap interval
as a capped vertical line, with the reference on the left and the model on the
right of each tick. Infinite bootstrap return periods are retained in the
percentiles. Intervals beyond the plotting limits end in outward arrows;
off-scale fitted values are drawn as triangles at the limit.

Empirical estimate (panels c-d)
-------------------------------
Panels (c)-(d) also show an "Empirical" estimate for the reference (left) and
model (right) as solid dots without uncertainty. If k of the n sample values are
at least as large as the Storm Hans threshold, the return period is (n + 1) / k
years, the Weibull plotting position of the k-th largest value, matching the
empirical points in panels (a)-(b) (e.g. 70 years for one seNorge event in
1957-2025, n = 69). If k = 0 the dot is placed at (n + 1) years, the plotting
position of the sample maximum, which is then a lower bound; the printed
summary flags this case.

Output
------
FIGURE_FORMAT = "png" or "pdf"; fonts are embedded as TrueType. A percentage
progress indicator is printed while the bootstrap fits run.
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
REFERENCE_DATASET = "senorge"  # "senorge" or "era5"
CATCHMENT = "regine_drammen"
X_DAYS = 2
OBSERVATION_YEARS = [1957, 2025]  # Years used for the reference fit.
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]
MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"

TOP_DISTRIBUTION = "GEV"  # Panels a-b: "GEV", "Gumbel", or "GenEx".
INCLUDE_STORM_HANS_IN_FIT = True  # Affects only the August reference fit.

# Bootstrap uncertainty
BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric"
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 9.3
SHOW_GRID = True
RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e8
PRECIPITATION_YMIN = 0.0
PRECIPITATION_YMAX = 200.0

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_FORMAT = "pdf"  # "png" or "pdf".
FIGURE_DPI = 300  # Resolution of PNG output.
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = f"fig-05.{FIGURE_FORMAT}"
OUTPUT_PATH = OUTPUT_DIRECTORY / OUTPUT_FILENAME

# =============================================================================
# Dataset conventions (fixed; must match the input files)
# =============================================================================
REFERENCE_FILE_YEARS = (1957, 2025)  # Year range encoded in the reference filename.
SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
MODEL_FILE_VARIABLE = "tp24"  # Variable name in the model sample filename.
MODEL_SAMPLE_VARIABLE = "tp24_max"  # Complete-lead monthly maxima.
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
MAY = 5
AUGUST = 8
PANEL_MONTHS = [AUGUST, MAY]
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
METHODS = ["GEV", "Gumbel", "GenEx"]
EMPIRICAL_LABEL = "Empirical"

# =============================================================================
# Plot constants
# =============================================================================
NUMBER_OF_RETURN_PERIODS = 500  # Resolution of the return-level curves.
DATASET_OFFSET = 0.075  # Symmetric separation around each tick.
OBSERVATION_COLOR = "tab:orange"
MODEL_COLOR = "tab:blue"
STORM_HANS_COLOR = "tab:orange"  # Same colour as the reference dataset.
STORM_HANS_LINESTYLE = "--"
CONFIDENCE_ALPHA = 0.15
CURVE_LINEWIDTH = 2.0
REFERENCE_LINEWIDTH = 1.5
MARKER_SIZE = 30
MARKER_LINEWIDTH = 0.8
INTERVAL_LINEWIDTH = 1.4
INTERVAL_CAP_WIDTH = 0.10
CLIPPED_ARROW_LENGTH = 14  # Points; fixed visual length on logarithmic axes.
CLIPPED_ARROW_HEAD_SIZE = 10
FITTED_MARKER_SIZE = 6
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
    if MODEL_DATA_METHOD not in {"raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:
        raise ValueError("Unsupported MODEL_DATA_METHOD.")
    if TOP_DISTRIBUTION not in METHODS:
        raise ValueError(f"TOP_DISTRIBUTION must be one of {METHODS}.")
    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if not REFERENCE_FILE_YEARS[0] <= OBSERVATION_YEARS[0] <= OBSERVATION_YEARS[1] <= REFERENCE_FILE_YEARS[1]:
        raise ValueError(
            f"OBSERVATION_YEARS must lie within {REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}."
        )
    if RETURN_PERIOD_MIN < 1:
        raise ValueError("RETURN_PERIOD_MIN must be at least 1.")
    if RETURN_PERIOD_MAX <= RETURN_PERIOD_MIN:
        raise ValueError("RETURN_PERIOD_MAX must exceed RETURN_PERIOD_MIN.")
    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError("NUMBER_OF_BOOTSTRAPS must be at least 1.")
    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must lie between 0 and 1.")
    if not isinstance(INCLUDE_STORM_HANS_IN_FIT, bool):
        raise TypeError("INCLUDE_STORM_HANS_IN_FIT must be True or False.")
    if FIGURE_FORMAT not in {"png", "pdf"}:
        raise ValueError("FIGURE_FORMAT must be 'png' or 'pdf'.")


def get_reference_name():
    """Return the display name of the selected reference dataset."""
    return {"senorge": "seNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_model_label():
    """Return the display label for the selected model data."""
    return "Model" if MODEL_DATA_METHOD == "raw" else "Model BC"


def get_storm_hans_label():
    """Return the label of the Storm Hans threshold line in panels a-b."""
    return f"{get_reference_name()} Storm Hans threshold"


def make_reference_filename():
    """Construct the reference monthly-maximum filename."""
    first_year, last_year = REFERENCE_FILE_YEARS
    filename = (
        f"monthly_max_samples_{get_reference_variable()}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_{first_year}-{last_year}.nc"
    )
    directory = "senorge_processed" if REFERENCE_DATASET == "senorge" else "era5_processed"
    return Path(config.dirs[directory]) / filename


def make_model_filename():
    """Construct the compact model filename written by the sample-building script."""
    stem = (
        f"monthly_max_samples_{MODEL_FILE_VARIABLE}_{X_DAYS}dayacc_"
        f"{CATCHMENT.removeprefix('regine_')}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    if MODEL_DATA_METHOD == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
        )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction_label}.nc"

# =============================================================================
# Data reading
# =============================================================================


def read_reference_month(month):
    """Read one reference-month fit sample and the Storm Hans threshold."""
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
    if values.size < 10:
        raise ValueError(f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain.")
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


def read_model_month(month):
    """Read one calendar-month model sample from sample_month(YYYYMM)."""
    filename = make_model_filename()
    variable = MODEL_SAMPLE_VARIABLE
    if not filename.is_file():
        raise FileNotFoundError(f"Model file not found: {filename}")
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
        raise ValueError(f"Fewer than 10 finite model values were found for month {month}.")
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


def distribution_ppf(probabilities, parameters, method):
    """Evaluate the fitted quantile function."""
    if method == "GEV":
        shape, location, scale = parameters
        return genextreme.ppf(probabilities, shape, loc=location, scale=scale)
    if method == "Gumbel":
        location, scale = parameters
        return gumbel_r.ppf(probabilities, loc=location, scale=scale)
    shape, scale = parameters
    return -scale * np.log1p(-np.power(probabilities, 1.0 / shape))


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


def empirical_return_periods(values):
    """Return Weibull empirical return periods and descending values."""
    sorted_values = np.sort(values)[::-1]
    ranks = np.arange(1, sorted_values.size + 1)
    return (sorted_values.size + 1) / ranks, sorted_values


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


def make_return_period_grid():
    """Return the return-period grid used by panels a-b."""
    grid_min = max(RETURN_PERIOD_MIN, 1.0 + np.finfo(float).eps)
    return np.geomspace(grid_min, RETURN_PERIOD_MAX, NUMBER_OF_RETURN_PERIODS)


def bootstrap_distribution(values, method, random_seed, progress=None):
    """Fit one distribution and create reusable bootstrap parameter sets."""
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


def build_month_analysis(month, month_index, progress=None):
    """Read one month and create shared reference/model bootstrap fits."""
    reference = read_reference_month(month)
    samples = {"reference": reference["fit_values"], "model": read_model_month(month)}
    bootstrap = {}
    for group_index, (group, values) in enumerate(samples.items()):
        for method_index, method in enumerate(METHODS):
            seed = RANDOM_SEED + 10000 * month_index + 1000 * group_index + method_index
            bootstrap[(group, method)] = bootstrap_distribution(values, method, seed, progress)
    return {"reference": reference, "samples": samples, "bootstrap": bootstrap}


def evaluate_return_levels(values, fit, return_periods, method):
    """Evaluate fitted and bootstrap return-level curves."""
    probabilities = 1.0 - 1.0 / return_periods
    fitted_levels = distribution_ppf(probabilities, fit["parameters"], method)
    bootstrap_levels = np.array(
        [
            distribution_ppf(probabilities, parameters, method)
            for parameters in fit["bootstrap_parameters"]
        ]
    )
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = np.percentile(bootstrap_levels, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(bootstrap_levels, 100.0 * (1.0 - alpha / 2.0), axis=0)
    empirical_rp, empirical_values = empirical_return_periods(values)
    return {
        "fitted_levels": fitted_levels,
        "lower": lower,
        "upper": upper,
        "empirical_rp": empirical_rp,
        "empirical_values": empirical_values,
    }


def analyse_top_month(month_analysis, return_periods):
    """Prepare reference and model return-level analyses for one month."""
    return {
        "reference": month_analysis["reference"],
        "reference_analysis": evaluate_return_levels(
            month_analysis["samples"]["reference"],
            month_analysis["bootstrap"][("reference", TOP_DISTRIBUTION)],
            return_periods,
            TOP_DISTRIBUTION,
        ),
        "model_analysis": evaluate_return_levels(
            month_analysis["samples"]["model"],
            month_analysis["bootstrap"][("model", TOP_DISTRIBUTION)],
            return_periods,
            TOP_DISTRIBUTION,
        ),
    }


def analyse_event_return_period(fit, event_value, method):
    """Evaluate one event threshold using reusable fitted bootstrap parameters."""
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


def empirical_event_metric(values, event_value):
    """
    Return the empirical estimate for one threshold, using Weibull plotting positions.

    k of the n sample values are at least as large as the threshold. As in
    empirical_return_periods, the k-th largest value has return period
    (n + 1) / k. With k = 0 the estimate is set to the plotting position of the
    sample maximum, (n + 1) years, a lower bound.
    """
    values = np.asarray(values, dtype=float)
    sample_size = values.size
    exceedances = int(np.count_nonzero(values >= event_value))
    return {
        "return_period": (sample_size + 1) / max(exceedances, 1),
        "exceedances": exceedances,
        "sample_size": sample_size,
    }


def calculate_metric_panel(month_analysis, month):
    """Calculate one lower-row Storm Hans panel from shared bootstrap fits."""
    event_value = month_analysis["reference"]["storm_hans_value"]
    results = {}
    for group in ["reference", "model"]:
        for method in METHODS:
            results[(group, method)] = analyse_event_return_period(
                month_analysis["bootstrap"][(group, method)],
                event_value,
                method,
            )
    empirical = {
        group: empirical_event_metric(month_analysis["samples"][group], event_value)
        for group in ["reference", "model"]
    }
    return {"month": month, "event_value": event_value, "results": results, "empirical": empirical}

# =============================================================================
# Plot formatting
# =============================================================================


def format_power_of_ten(value, _position):
    """Format positive logarithmic ticks as powers of ten."""
    if value <= 0:
        return ""
    exponent = np.log10(value)
    rounded_exponent = int(np.round(exponent))
    if not np.isclose(exponent, rounded_exponent, atol=1e-10):
        return ""
    return rf"$10^{{{rounded_exponent}}}$"


def format_metric_return_period(value, _position):
    """Format lower-panel return periods and mark the configured upper limit."""
    if value <= 0:
        return ""
    exponent = np.log10(value)
    rounded_exponent = int(np.round(exponent))
    if not np.isclose(exponent, rounded_exponent, atol=1e-10):
        return ""
    if np.isclose(value, RETURN_PERIOD_MAX):
        return rf"$>10^{{{rounded_exponent}}}$"
    return rf"$10^{{{rounded_exponent}}}$"


def format_top_axis(axis):
    """Format a return-level panel."""
    axis.set_xscale("log")
    axis.set_xlim(RETURN_PERIOD_MIN, RETURN_PERIOD_MAX)
    axis.set_xlabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    axis.xaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
    axis.set_ylim(PRECIPITATION_YMIN, PRECIPITATION_YMAX)
    axis.set_ylabel(f"Monthly maximum {X_DAYS}-day precipitation [mm]", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_top_panel(axis, panel_label, month, result, return_periods, show_legend=False):
    """Plot one return-level distribution panel with reference data on top."""
    for analysis, color, zorder in [
        (result["model_analysis"], MODEL_COLOR, 2),
        (result["reference_analysis"], OBSERVATION_COLOR, 3),
    ]:
        axis.fill_between(
            return_periods,
            analysis["lower"],
            analysis["upper"],
            color=color,
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
            zorder=zorder,
        )
        axis.plot(
            return_periods,
            analysis["fitted_levels"],
            color=color,
            linewidth=CURVE_LINEWIDTH,
            zorder=zorder,
        )
        axis.scatter(
            analysis["empirical_rp"],
            analysis["empirical_values"],
            facecolors="none",
            edgecolors=color,
            linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE,
            zorder=zorder,
        )
    axis.axhline(
        result["reference"]["storm_hans_value"],
        color=STORM_HANS_COLOR,
        linestyle=STORM_HANS_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=4,
    )
    format_top_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[month - 1]}: {TOP_DISTRIBUTION} fit",
        loc="center",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )
    if show_legend:
        handles = [
            Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=CURVE_LINEWIDTH,
                   label=get_reference_name()),
            Line2D([0], [0], color=STORM_HANS_COLOR, linestyle=STORM_HANS_LINESTYLE,
                   linewidth=REFERENCE_LINEWIDTH, label=get_storm_hans_label()),
            Line2D([0], [0], color=MODEL_COLOR, linewidth=CURVE_LINEWIDTH,
                   label=get_model_label()),
        ]
        axis.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")


def configure_metric_axis(axis):
    """Apply common formatting to a lower-row return-period panel."""
    tick_labels = [*METHODS, EMPIRICAL_LABEL]
    axis.set_yscale("log")
    axis.set_ylim(RETURN_PERIOD_MIN, 1.15 * RETURN_PERIOD_MAX)
    axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))
    axis.set_xlim(-0.5, len(tick_labels) - 0.5)
    axis.set_xticks(range(len(tick_labels)))
    axis.set_xticklabels(tick_labels)
    axis.set_ylabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    axis.set_xlabel("Estimation method", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
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
    """Show clipped intervals with arrows and off-scale estimates with triangles.
    Caps mark actual interval endpoints only. Arrows indicate continuation beyond
    the displayed range, including infinite return periods.
    """
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


def plot_empirical_estimates(axis, position, empirical):
    """Plot reference and model empirical estimates as solid dots."""
    for group, offset, color in [
        ("reference", -DATASET_OFFSET, OBSERVATION_COLOR),
        ("model", DATASET_OFFSET, MODEL_COLOR),
    ]:
        axis.plot(
            position + offset, clip_metric(empirical[group]["return_period"]),
            marker="o", linestyle="none", color=color,
            markersize=FITTED_MARKER_SIZE, clip_on=False, zorder=5,
        )


def plot_metric_panel(axis, panel_label, panel_output):
    """Plot fitted dots and central bootstrap intervals for reference and model."""
    tail_percent = 100 * (1 - CONFIDENCE_LEVEL) / 2
    for method_index, method in enumerate(METHODS):
        for group, offset, color in [
            ("reference", -DATASET_OFFSET, OBSERVATION_COLOR),
            ("model", DATASET_OFFSET, MODEL_COLOR),
        ]:
            analysis = panel_output["results"][(group, method)]
            lower = percentile_preserving_infinity(analysis["metric_samples"], tail_percent)
            upper = percentile_preserving_infinity(analysis["metric_samples"], 100 - tail_percent)
            # The percentile interval need not contain the fitted estimate.
            plot_metric_interval(
                axis, method_index + offset, lower, upper, analysis["return_period"], color
            )

    plot_empirical_estimates(axis, len(METHODS), panel_output["empirical"])
    configure_metric_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[panel_output['month'] - 1]}: Storm Hans threshold",
        loc="center",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

# =============================================================================
# Figure and reporting
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


def print_uncertainty_comparison(metric_results):
    """Report untruncated bootstrap interval widths for both months."""
    reference_name = get_reference_name()
    tail_percent = 100 * (1 - CONFIDENCE_LEVEL) / 2
    print(f"\n{CONFIDENCE_LEVEL:.0%} bootstrap uncertainty comparison")
    print("Return-period interval widths in years.")
    print(f"Factor = {reference_name} interval width / {get_model_label()} interval width.")
    print("Factor > 1: narrower model interval; factor < 1: wider model interval.")
    print("Widths use the original scale, before logarithms or plot-limit clipping.")
    for month in PANEL_MONTHS:
        panel = metric_results[month]
        print(f"\n{MONTH_NAMES[month - 1]}: Storm Hans threshold ({panel['event_value']:.3f} mm)")
        print(
            f"{'Distribution':<13} {'Dataset':<12} {'Lower':>12} {'Upper':>12} "
            f"{'Width':>12} {'Factor':>12}"
        )
        for method in METHODS:
            intervals = []
            for group in ["reference", "model"]:
                samples = panel["results"][(group, method)]["metric_samples"]
                lower = percentile_preserving_infinity(samples, tail_percent)
                upper = percentile_preserving_infinity(samples, 100 - tail_percent)
                # An interval [inf, inf] has no defined numerical width.
                width = np.nan if lower == upper == np.inf else upper - lower
                intervals.append((lower, upper, width))
            factor = interval_width_ratio(intervals[0][2], intervals[1][2])
            factor_text = "undefined" if np.isnan(factor) else f"{factor:.4g}x"
            for index, (lower, upper, width) in enumerate(intervals):
                label = reference_name if index == 0 else get_model_label()
                ratio_label = factor_text if index == 1 else ""
                print(
                    f"{method:<13} {label:<12} {lower:12.5g} {upper:12.5g} "
                    f"{width:12.5g} {ratio_label:>12}"
                )
    print("\ninf = unbounded; nan/undefined = no defined width or ratio (including 0/0).")
    print("An infinite factor can also result from a zero model interval width.")


def print_summary(top_results, metric_results):
    """Print the key input, Storm Hans threshold, and empirical-estimate information."""
    print("Selected settings")
    print("-----------------")
    print(f"Reference dataset: {get_reference_name()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")
    print(f"Model file:        {make_model_filename()}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Top distribution:  {TOP_DISTRIBUTION}")
    print(f"Bootstrap method:  {BOOTSTRAP_METHOD}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS}")
    print(f"Include Hans fit:  {INCLUDE_STORM_HANS_IN_FIT}")
    for month in PANEL_MONTHS:
        reference = top_results[month]["reference"]
        print()
        print(f"{MONTH_NAMES[month - 1]} reference fit: {reference['fit_values'].size} values")
        print(f"Storm Hans threshold: {reference['storm_hans_value']:.3f} mm")
        empirical = metric_results[month]["empirical"]
        for group, label in [("reference", get_reference_name()), ("model", get_model_label())]:
            k, n = empirical[group]["exceedances"], empirical[group]["sample_size"]
            if k == 0:
                estimate = (
                    f"no exceedances; plotted at {n + 1} years "
                    f"(lower bound: return period > {n + 1} years)"
                )
            else:
                estimate = f"empirical return period {(n + 1) / k:.4g} years"
            print(f"{label} empirical: {k} of {n} values >= Storm Hans; {estimate}")


def make_figure(top_results, metric_results, return_periods):
    """Create the 2 x 2 figure (August left, May right)."""
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
        2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True
    )
    # August is always the left column; May is always the right column.
    plot_top_panel(axes[0, 0], "a", AUGUST, top_results[AUGUST], return_periods, show_legend=True)
    plot_top_panel(axes[0, 1], "b", MAY, top_results[MAY], return_periods)
    plot_metric_panel(axes[1, 0], "c", metric_results[AUGUST])
    plot_metric_panel(axes[1, 1], "d", metric_results[MAY])

    if WRITE_TO_FILE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PATH, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
                       bbox_inches="tight", facecolor="white")
        print("Wrote:", OUTPUT_PATH)
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Main
# =============================================================================


def main():
    """Run the May/August analysis and plot the selected panels."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)
    return_periods = make_return_period_grid()
    progress = ProgressTracker(len(PANEL_MONTHS) * 2 * len(METHODS) * NUMBER_OF_BOOTSTRAPS)
    print("Running bootstrap fits...")
    print("Progress:   0%", end="\r", flush=True)
    month_analyses = {
        month: build_month_analysis(month, index, progress)
        for index, month in enumerate(PANEL_MONTHS)
    }
    top_results = {
        month: analyse_top_month(month_analyses[month], return_periods)
        for month in PANEL_MONTHS
    }
    metric_results = {
        month: calculate_metric_panel(month_analyses[month], month) for month in PANEL_MONTHS
    }
    print_summary(top_results, metric_results)
    print_uncertainty_comparison(metric_results)
    make_figure(top_results, metric_results, return_periods)


if __name__ == "__main__":
    main()
