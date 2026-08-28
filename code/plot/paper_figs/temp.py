"""
Compare two bootstrap/resampling procedures for monthly extreme model precipitation.

Purpose
-------
This script modifies the original reference-versus-model analysis so that both
samples come from the same S2S model dataset. The comparison isolates the effect
of how model realizations are resampled before fitting an extreme-value model.

The two procedures are:

1. Pooled nonparametric bootstrap
   For one calendar month, all finite forecast and hindcast realizations from all
   available years are pooled. Each bootstrap replicate draws, with replacement,
   the same number of values as the original pooled sample. This is the standard
   nonparametric bootstrap used for the model in the original script.

2. Year-balanced repeated sampling
   The same calendar month is kept separate by YYYYMM. The smallest finite sample
   count among those YYYYMM groups defines a common target count. For every
   replicate, exactly that many values are sampled without replacement from each
   YYYYMM group, the equally represented years are pooled, and an extreme-value
   distribution is fitted. Repeating this procedure quantifies sensitivity to
   unequal realization counts across years. Strictly speaking, this is repeated
   balanced subsampling rather than a classical bootstrap because sampling within
   each year is without replacement.

For the upper panels, the pooled method is fitted once to the complete pooled
sample. The balanced method needs a representative central fit because it has no
single natural original sample, so one reproducible balanced draw is used for its
central curve. Confidence bands for both methods come from their repeated fits.

The observational dataset is not bootstrapped or fitted. It is read only to retain
Storm Hans and calendar-record precipitation thresholds used by the lower panels
and as horizontal reference lines in the upper panels.

The figure retains the original 3 x 2 structure for August and May. The upper row
compares return-level curves for one selected distribution. The lower two rows
compare bootstrap/resampling return metrics for GEV, Gumbel, and GenEx.
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
# User settings
# =============================================================================

REFERENCE_DATASET = "senorge"  # "senorge" or "era5"; thresholds only
CATCHMENT = "regine_drammen"
X_DAYS = 2

OBSERVATION_YEARS = [1957, 2025]
REFERENCE_FILE_YEARS = [1957, 2025]
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"
MODEL_SAMPLING_GROUP = "full"  # "full", "split1", "split2", ...
MODEL_VARIABLE = "tp24"

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Used for panels a-b. Options: "GEV", "Gumbel", "GenEx".
TOP_DISTRIBUTION = "GEV"

# Used by all panels. Options: "return_period" or "aep".
PLOT_METRIC = "return_period"
AEP_YEARS = 1

NUMBER_OF_BOOTSTRAPS = 10
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

FIGURE_DPI = 300
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 14

RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e7
NUMBER_OF_RETURN_PERIODS = 500

PRECIPITATION_YMIN = 0.0
PRECIPITATION_YMAX = 200.0

SHOW_GRID = True
WRITE_TO_FILE = False
SHOW_FIGURE = True


# =============================================================================
# Plot constants
# =============================================================================

MAY = 5
AUGUST = 8
PANEL_MONTHS = [AUGUST, MAY]
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

METHODS = ["GEV", "Gumbel", "GenEx"]
METHOD_COLORS = {"GEV": "tab:pink", "Gumbel": "tab:green", "GenEx": "tab:purple"}
METHOD_OFFSETS = {"GEV": -0.18, "Gumbel": 0.0, "GenEx": 0.18}

POOLED_GROUP = "pooled"
BALANCED_GROUP = "balanced"
RESAMPLING_GROUPS = [POOLED_GROUP, BALANCED_GROUP]
GROUP_COLORS = {POOLED_GROUP: "tab:blue", BALANCED_GROUP: "tab:orange"}
GROUP_LABELS = {
    POOLED_GROUP: "Pooled bootstrap",
    BALANCED_GROUP: "Year-balanced sampling",
}

STORM_HANS_COLOR = "grey"
RECORD_COLOR = "grey"
STORM_HANS_LINESTYLE = "--"
RECORD_LINESTYLE = ":"

CONFIDENCE_ALPHA = 0.15
CURVE_LINEWIDTH = 2.0
REFERENCE_LINEWIDTH = 2.0
MARKER_SIZE = 30
MARKER_LINEWIDTH = 0.8

INTERVAL_LINEWIDTH = 1.4
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

    valid_data_methods = {"raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}
    if MODEL_DATA_METHOD not in valid_data_methods:
        raise ValueError("Unsupported MODEL_DATA_METHOD.")
    if TOP_DISTRIBUTION not in METHODS:
        raise ValueError(f"TOP_DISTRIBUTION must be one of {METHODS}.")
    if PLOT_METRIC not in {"return_period", "aep"}:
        raise ValueError("PLOT_METRIC must be 'return_period' or 'aep'.")
    if AEP_YEARS < 1:
        raise ValueError("AEP_YEARS must be at least 1.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if RETURN_PERIOD_MIN < 1:
        raise ValueError("RETURN_PERIOD_MIN must be at least 1.")
    if RETURN_PERIOD_MAX <= RETURN_PERIOD_MIN:
        raise ValueError("RETURN_PERIOD_MAX must exceed RETURN_PERIOD_MIN.")
    if NUMBER_OF_RETURN_PERIODS < 2:
        raise ValueError("NUMBER_OF_RETURN_PERIODS must be at least 2.")
    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError("NUMBER_OF_BOOTSTRAPS must be at least 1.")
    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must lie between 0 and 1.")
    if not 0 < MIN_SUCCESSFUL_BOOTSTRAP_FRACTION <= 1:
        raise ValueError("MIN_SUCCESSFUL_BOOTSTRAP_FRACTION must lie in (0, 1].")

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    number_of_usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1
    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError("X_DAYS is too large for the configured lead range.")
    if not 1 <= NUMBER_OF_LEAD_BINS <= number_of_usable_leads:
        raise ValueError("NUMBER_OF_LEAD_BINS is invalid for the usable lead range.")

    valid_groups = {"full", *(f"split{i}" for i in range(1, NUMBER_OF_LEAD_BINS + 1))}
    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}.")


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_model_label():
    """Return the display label for the selected model data."""
    return "Model" if MODEL_DATA_METHOD == "raw" else "Model BC"


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

    split_index = int(MODEL_SAMPLING_GROUP.removeprefix("split")) - 1
    lead_start, lead_end = build_lead_bins()[split_index]
    return f"tp24_max_lead{lead_start}_{lead_end}"


def make_reference_filename():
    """Construct the observational filename used only for event thresholds."""
    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)

    first_year, last_year = REFERENCE_FILE_YEARS
    if REFERENCE_DATASET == "senorge":
        filename = (
            f"monthly_max_samples_{SENORGE_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_{first_year}-{last_year}.nc"
        )
        return Path(config.dirs["senorge_processed"]) / filename

    filename = (
        f"monthly_max_samples_{ERA5_VARIABLE}_{X_DAYS}dayacc_{CATCHMENT}_"
        f"{first_year}-{last_year}.nc"
    )
    return Path(config.dirs["era5_processed"]) / filename


def make_model_filename():
    """Construct the compact model filename."""
    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)

    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )

    correction_label = "raw" if MODEL_DATA_METHOD == "raw" else (
        f"bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}_"
        f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
    )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction_label}.nc"


def make_figure_filename():
    """Construct the six-panel output figure filename."""
    model_label = "model-raw" if MODEL_DATA_METHOD == "raw" else f"model-bc-{MODEL_DATA_METHOD}"
    return Path(config.dirs["fig"]) / (
        f"bootstrap-comparison-{PLOT_METRIC}-{TOP_DISTRIBUTION}-{model_label}-"
        f"{FORECAST_DATE_RANGE[0]}-{FORECAST_DATE_RANGE[-1]}.png"
    )


# =============================================================================
# Data reading and sample construction
# =============================================================================

def read_reference_thresholds(month):
    """Read Storm Hans and calendar-record thresholds without fitting observations."""
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

    if values.size == 0:
        raise ValueError(f"No finite {MONTH_NAMES[month - 1]} reference values were found.")

    hans_in_range = OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
    if month == AUGUST and hans_in_range:
        keep = years != STORM_HANS_YEAR
        years, values = years[keep], values[keep]

    if values.size == 0:
        raise ValueError(f"No values remain for the {MONTH_NAMES[month - 1]} record.")

    record_index = int(np.argmax(values))
    return {
        "storm_hans_value": storm_hans_value,
        "record_value": float(values[record_index]),
        "record_year": int(years[record_index]),
    }


def read_model_month_by_year(month):
    """Read one calendar month and retain separate finite samples for each YYYYMM."""
    filename = make_model_filename()
    variable = get_model_variable()

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
                f"'{variable}' must have dimensions ('number', 'i_date'); found {ds[variable].dims}."
            )
        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("'sample_month' must have dimension ('i_date',).")

        sample_month = np.asarray(ds["sample_month"].values, dtype="int64")
        values = np.asarray(ds[variable].transpose("number", "i_date").values, dtype=float)

    selected_months = np.unique(sample_month[sample_month % 100 == month])
    if selected_months.size == 0:
        raise ValueError(f"No model samples were found for calendar month {month}.")

    values_by_yyyymm = {}
    for yyyymm in selected_months:
        month_values = values[:, sample_month == yyyymm].ravel()
        month_values = month_values[np.isfinite(month_values)]
        if month_values.size == 0:
            raise ValueError(f"No finite model values were found for {int(yyyymm)}.")
        values_by_yyyymm[int(yyyymm)] = month_values

    pooled_values = np.concatenate(list(values_by_yyyymm.values()))
    if pooled_values.size < 10:
        raise ValueError(f"Fewer than 10 finite model values were found for month {month}.")

    target_count = min(values.size for values in values_by_yyyymm.values())
    if target_count < 1:
        raise ValueError(f"The year-balanced target count is zero for month {month}.")

    return {
        "values_by_yyyymm": values_by_yyyymm,
        "pooled_values": pooled_values,
        "target_count": target_count,
    }


def make_balanced_sample(values_by_yyyymm, target_count, rng):
    """Sample the same number of realizations without replacement from each YYYYMM."""
    selected = []
    for yyyymm in sorted(values_by_yyyymm):
        values = values_by_yyyymm[yyyymm]
        indices = rng.choice(values.size, size=target_count, replace=False)
        selected.append(values[indices])
    return np.concatenate(selected)


# =============================================================================
# Extreme-value distributions
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

        initial_parameters = initial_parameters or (1.0, np.mean(positive))
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


def empirical_return_periods(values):
    """Return Weibull empirical return periods and descending values."""
    sorted_values = np.sort(values)[::-1]
    ranks = np.arange(1, sorted_values.size + 1)
    return (sorted_values.size + 1) / ranks, sorted_values


def return_period_from_probability(probability):
    """Convert annual exceedance probability to return period."""
    return np.inf if probability <= 0 else 1.0 / probability


def horizon_aep(probability):
    """Convert annual exceedance probability to AEP_YEARS exceedance probability."""
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(-np.expm1(AEP_YEARS * np.log1p(-probability)))


# =============================================================================
# Bootstrap and repeated balanced sampling
# =============================================================================

class ProgressTracker:
    """Print integer percentage completion for the requested distribution fits."""

    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.last_percent = -1

    def update(self):
        """Advance one fit and print when the integer percentage changes."""
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


def fit_repeated_samples(sample_factory, central_values, method, random_seed, progress=None):
    """Fit a central sample and NUMBER_OF_BOOTSTRAPS independently generated samples."""
    parameters = fit_distribution(central_values, method)
    rng = np.random.default_rng(random_seed)
    bootstrap_parameters = []

    for _ in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = sample_factory(rng)
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
            f"{method} repeated fits succeeded."
        )

    return {
        "parameters": parameters,
        "bootstrap_parameters": bootstrap_parameters,
    }


def build_month_analysis(month, month_index, progress=None):
    """Create pooled-bootstrap and year-balanced fits for one calendar month."""
    thresholds = read_reference_thresholds(month)
    model = read_model_month_by_year(month)
    pooled_values = model["pooled_values"]
    values_by_yyyymm = model["values_by_yyyymm"]
    target_count = model["target_count"]

    central_rng = np.random.default_rng(RANDOM_SEED + 100_000 + month_index)
    balanced_values = make_balanced_sample(values_by_yyyymm, target_count, central_rng)

    samples = {
        POOLED_GROUP: pooled_values,
        BALANCED_GROUP: balanced_values,
    }
    repeated_fits = {}

    for group_index, group in enumerate(RESAMPLING_GROUPS):
        for method_index, method in enumerate(METHODS):
            seed = RANDOM_SEED + 10_000 * month_index + 1_000 * group_index + method_index

            if group == POOLED_GROUP:
                sample_factory = lambda rng, values=pooled_values: rng.choice(
                    values, size=values.size, replace=True
                )
            else:
                sample_factory = lambda rng: make_balanced_sample(
                    values_by_yyyymm, target_count, rng
                )

            repeated_fits[(group, method)] = fit_repeated_samples(
                sample_factory, samples[group], method, seed, progress
            )

    return {
        "thresholds": thresholds,
        "model": model,
        "samples": samples,
        "repeated_fits": repeated_fits,
    }


def evaluate_return_levels(values, fit, return_periods, method):
    """Evaluate the central fit and repeated-fit return-level uncertainty."""
    probabilities = 1.0 - 1.0 / return_periods
    fitted_levels = distribution_ppf(probabilities, fit["parameters"], method)
    bootstrap_levels = np.array([
        distribution_ppf(probabilities, parameters, method)
        for parameters in fit["bootstrap_parameters"]
    ])

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
    """Prepare return-level analyses for the two model resampling procedures."""
    analyses = {}
    for group in RESAMPLING_GROUPS:
        analyses[group] = evaluate_return_levels(
            month_analysis["samples"][group],
            month_analysis["repeated_fits"][(group, TOP_DISTRIBUTION)],
            return_periods,
            TOP_DISTRIBUTION,
        )

    return {
        "thresholds": month_analysis["thresholds"],
        "analyses": analyses,
    }


def metric_samples_from_probabilities(probabilities):
    """Convert repeated-fit probabilities while retaining unbounded return periods."""
    probabilities = np.asarray(probabilities, dtype=float)
    if PLOT_METRIC == "aep":
        samples = 100.0 * np.array([horizon_aep(value) for value in probabilities])
        return samples[np.isfinite(samples)]

    samples = np.array([return_period_from_probability(value) for value in probabilities])
    return samples[~np.isnan(samples)]


def analyse_event_metric(fit, event_value, method):
    """Evaluate one event threshold using repeated fitted parameter sets."""
    probability = exceedance_probability(event_value, fit["parameters"], method)
    bootstrap_probabilities = np.array([
        exceedance_probability(event_value, parameters, method)
        for parameters in fit["bootstrap_parameters"]
    ])

    metric_samples = metric_samples_from_probabilities(bootstrap_probabilities)
    if metric_samples.size == 0:
        raise RuntimeError(f"{method} produced no valid repeated-fit metric values.")

    return {
        "probability": probability,
        "return_period": return_period_from_probability(probability),
        "metric_samples": metric_samples,
    }


def calculate_metric_panel(month_analysis, month, threshold_type):
    """Calculate one lower-row panel for both model resampling procedures."""
    thresholds = month_analysis["thresholds"]
    event_value = (
        thresholds["storm_hans_value"]
        if threshold_type == "storm_hans"
        else thresholds["record_value"]
    )

    results = {}
    for group in RESAMPLING_GROUPS:
        for method in METHODS:
            results[(group, method)] = analyse_event_metric(
                month_analysis["repeated_fits"][(group, method)], event_value, method
            )

    return {
        "month": month,
        "threshold_type": threshold_type,
        "event_value": event_value,
        "record_year": thresholds["record_year"],
        "results": results,
    }


# =============================================================================
# Plot formatting
# =============================================================================

def return_period_to_aep_percent(return_period):
    """Convert a return period to the configured multi-year AEP percentage."""
    annual_probability = 1.0 / return_period
    probability = -np.expm1(AEP_YEARS * np.log1p(-annual_probability))
    return 100.0 * probability


def top_x_values(return_periods):
    """Convert return periods to the selected top-row x coordinate."""
    if PLOT_METRIC == "return_period":
        return return_periods
    return return_period_to_aep_percent(return_periods)


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


def get_aep_limits():
    """Return AEP limits equivalent to the shared return-period range."""
    return return_period_to_aep_percent(RETURN_PERIOD_MAX), return_period_to_aep_percent(
        RETURN_PERIOD_MIN
    )


def format_top_axis(axis):
    """Format a return-level panel."""
    axis.set_xscale("log")

    if PLOT_METRIC == "return_period":
        axis.set_xlim(RETURN_PERIOD_MIN, RETURN_PERIOD_MAX)
        axis.set_xlabel("Return period [years]", fontsize=AXIS_LABELSIZE)
        axis.xaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
    else:
        aep_min, aep_max = get_aep_limits()
        axis.set_xlim(aep_max, aep_min)
        axis.set_xlabel(f"{AEP_YEARS}-year exceedance probability [%]", fontsize=AXIS_LABELSIZE)

    axis.set_ylim(PRECIPITATION_YMIN, PRECIPITATION_YMAX)
    axis.set_ylabel(
        f"Monthly maximum {X_DAYS}-day precipitation [mm]", fontsize=AXIS_LABELSIZE
    )
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_top_panel(axis, panel_label, month, result, return_periods, show_legend=False):
    """Plot return-level curves for pooled and year-balanced model resampling."""
    x_values = top_x_values(return_periods)

    for group in RESAMPLING_GROUPS:
        analysis = result["analyses"][group]
        color = GROUP_COLORS[group]
        axis.fill_between(
            x_values,
            analysis["lower"],
            analysis["upper"],
            color=color,
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
        )
        axis.plot(
            x_values,
            analysis["fitted_levels"],
            color=color,
            linewidth=CURVE_LINEWIDTH,
        )
        axis.scatter(
            top_x_values(analysis["empirical_rp"]),
            analysis["empirical_values"],
            facecolors="none",
            edgecolors=color,
            linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE,
        )

    thresholds = result["thresholds"]
    axis.axhline(
        thresholds["storm_hans_value"],
        color=STORM_HANS_COLOR,
        linestyle=STORM_HANS_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
    )
    axis.axhline(
        thresholds["record_value"],
        color=RECORD_COLOR,
        linestyle=RECORD_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
    )

    format_top_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[month - 1]}: {TOP_DISTRIBUTION} fit",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D(
                [0], [0], color=GROUP_COLORS[group], linewidth=CURVE_LINEWIDTH,
                label=GROUP_LABELS[group],
            )
            for group in RESAMPLING_GROUPS
        ]
        handles.extend([
            Line2D(
                [0], [0], color=STORM_HANS_COLOR, linestyle=STORM_HANS_LINESTYLE,
                linewidth=REFERENCE_LINEWIDTH, label="Storm Hans August 2023",
            ),
            Line2D(
                [0], [0], color=RECORD_COLOR, linestyle=RECORD_LINESTYLE,
                linewidth=REFERENCE_LINEWIDTH, label="Monthly record excluding Storm Hans",
            ),
        ])
        axis.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")


def metric_axis_label():
    """Return the lower-row y-axis label."""
    if PLOT_METRIC == "aep":
        return f"{AEP_YEARS}-year exceedance probability [%]"
    return "Return period [years]"


def configure_metric_axis(axis):
    """Apply common formatting to a lower-row return-metric panel."""
    axis.set_yscale("log")

    if PLOT_METRIC == "aep":
        aep_min, aep_max = get_aep_limits()
        axis.set_ylim(aep_min, aep_max)
    else:
        axis.set_ylim(RETURN_PERIOD_MIN, 1.15 * RETURN_PERIOD_MAX)
        axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))

    axis.set_xlim(-0.55, 1.55)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([GROUP_LABELS[group] for group in RESAMPLING_GROUPS])
    axis.set_ylabel(metric_axis_label(), fontsize=AXIS_LABELSIZE)
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


def summarize_boxplot_samples(samples):
    """Return median, IQR, and central 95% limits while preserving infinity."""
    return {
        "q1": percentile_preserving_infinity(samples, 25.0),
        "median": percentile_preserving_infinity(samples, 50.0),
        "q3": percentile_preserving_infinity(samples, 75.0),
        "whislo": percentile_preserving_infinity(samples, 2.5),
        "whishi": percentile_preserving_infinity(samples, 97.5),
    }


def clip_metric(value):
    """Clip one plotted lower-panel metric value to the shared metric limits."""
    if PLOT_METRIC == "aep":
        aep_min, aep_max = get_aep_limits()
        return float(np.clip(value, aep_min, aep_max))
    return float(np.clip(value, RETURN_PERIOD_MIN, RETURN_PERIOD_MAX))


def plot_metric_panel(axis, panel_label, panel_output, show_legend=False):
    """Plot IQR boxes and central 95% intervals for both resampling procedures."""
    for group_index, group in enumerate(RESAMPLING_GROUPS):
        for method in METHODS:
            analysis = panel_output["results"][(group, method)]
            position = group_index + METHOD_OFFSETS[method]
            summary = summarize_boxplot_samples(analysis["metric_samples"])
            color = METHOD_COLORS[method]

            plot_summary = {
                "label": "",
                "q1": clip_metric(summary["q1"]),
                "med": clip_metric(summary["median"]),
                "q3": clip_metric(summary["q3"]),
                "whislo": clip_metric(summary["whislo"]),
                "whishi": RETURN_PERIOD_MAX if np.isposinf(summary["whishi"]) else clip_metric(
                    summary["whishi"]
                ),
                "fliers": [],
            }

            axis.bxp(
                [plot_summary],
                positions=[position],
                widths=0.14,
                showfliers=False,
                patch_artist=False,
                manage_ticks=False,
                boxprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},
                whiskerprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},
                capprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},
                medianprops={"color": color, "linewidth": 1.8},
            )

    configure_metric_axis(axis)
    panel_titles = {
        "c": "August: Storm Hans threshold",
        "d": "May: Storm Hans threshold",
        "e": "August: record threshold excluding Storm Hans",
        "f": "May: record threshold",
    }
    axis.set_title(
        f"{panel_label}) {panel_titles[panel_label]}",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D(
                [0], [0], linestyle="-", color=METHOD_COLORS[method],
                linewidth=INTERVAL_LINEWIDTH, label=method,
            )
            for method in METHODS
        ]
        axis.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")


# =============================================================================
# Figure and reporting
# =============================================================================

def print_summary(month_analyses):
    """Print model sample sizes, balancing targets, and event thresholds."""
    print("Selected settings")
    print("-----------------")
    print(f"Model data:       {MODEL_DATA_METHOD}")
    print(f"Model file:       {make_model_filename()}")
    print(f"Threshold file:   {make_reference_filename()}")
    print(f"Top distribution: {TOP_DISTRIBUTION}")
    print(f"Metric:           {PLOT_METRIC}")
    print(f"Repeated fits:    {NUMBER_OF_BOOTSTRAPS}")

    for month in PANEL_MONTHS:
        analysis = month_analyses[month]
        model = analysis["model"]
        thresholds = analysis["thresholds"]
        counts = [values.size for values in model["values_by_yyyymm"].values()]

        print()
        print(MONTH_NAMES[month - 1])
        print(f"  YYYYMM groups:                 {len(counts)}")
        print(f"  Pooled model realizations:     {model['pooled_values'].size}")
        print(f"  Realizations per YYYYMM range: {min(counts)}-{max(counts)}")
        print(f"  Balanced target per YYYYMM:    {model['target_count']}")
        print(
            f"  Balanced sample size:          "
            f"{model['target_count'] * len(model['values_by_yyyymm'])}"
        )
        print(f"  Storm Hans threshold:          {thresholds['storm_hans_value']:.3f} mm")
        print(
            f"  Calendar record:               {thresholds['record_value']:.3f} mm "
            f"({thresholds['record_year']})"
        )


def make_figure(top_results, metric_results, return_periods):
    """Create the combined 3 x 2 model-bootstrap comparison figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": TICK_LABELSIZE,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(
        3, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True
    )

    plot_top_panel(
        axes[0, 0], "a", AUGUST, top_results[AUGUST], return_periods, show_legend=True
    )
    plot_top_panel(axes[0, 1], "b", MAY, top_results[MAY], return_periods)

    plot_metric_panel(axes[1, 0], "c", metric_results[(AUGUST, "storm_hans")], True)
    plot_metric_panel(axes[1, 1], "d", metric_results[(MAY, "storm_hans")])
    plot_metric_panel(axes[2, 0], "e", metric_results[(AUGUST, "calendar_record")])
    plot_metric_panel(axes[2, 1], "f", metric_results[(MAY, "calendar_record")])

    if WRITE_TO_FILE:
        filename = make_figure_filename()
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the six-panel comparison of two model resampling procedures."""
    validate_settings()
    return_periods = make_return_period_grid()

    total_fits = len(PANEL_MONTHS) * len(RESAMPLING_GROUPS) * len(METHODS) * NUMBER_OF_BOOTSTRAPS
    progress = ProgressTracker(total_fits)
    print("Running repeated extreme-value fits...")
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
        (month, threshold_type): calculate_metric_panel(
            month_analyses[month], month, threshold_type
        )
        for month, threshold_type in [
            (MAY, "storm_hans"),
            (AUGUST, "storm_hans"),
            (MAY, "calendar_record"),
            (AUGUST, "calendar_record"),
        ]
    }

    print_summary(month_analyses)
    make_figure(top_results, metric_results, return_periods)


if __name__ == "__main__":
    main()
