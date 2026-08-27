"""
Create a 3 x 2 extreme-precipitation figure for May and August.

Layout
------
    a) May return-level distribution
    b) August return-level distribution
    c) May return metric for the Storm Hans threshold
    d) August return metric for the Storm Hans threshold
    e) May return metric for the calendar-month record
    f) August return metric for the calendar-month record

The reference fit uses OBSERVATION_YEARS, with a user option controlling whether
August 2023 (Storm Hans) is included in the August fit. The option has no effect
when 2023 is outside OBSERVATION_YEARS. Calendar-record thresholds use the same
year range, but August 2023 is always excluded from the August record so Storm
Hans does not define its own comparison threshold. May 2023 is retained.

Panels (c)-(f) summarize the bootstrap return metric with box-and-whisker plots:
the center line is the median, the box spans the interquartile range, and the
whiskers mark the 2.5th and 97.5th percentiles. Infinite return-period estimates
are retained when calculating these percentiles. If a percentile is unbounded,
its whisker is drawn to the plotting limit and marked with an arrow rather than
silently discarding the infinite bootstrap values.

The compact model input is expected to contain sample_month(i_date) as YYYYMM and
precipitation maxima with dimensions (number, i_date). Finite values are pooled,
so files containing padded 51-, 101-, and 11-member samples are handled directly.

A percentage-complete progress indicator is printed while the bootstrap fits are
running. Progress is based on the total requested bootstrap fits across both
months, both datasets, and all fitted distributions.
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

REFERENCE_DATASET = "senorge"  # "senorge" or "era5"
CATCHMENT = "regine_drammen"
X_DAYS = 2

OBSERVATION_YEARS = [1957, 2025]
REFERENCE_FILE_YEARS = [1957, 2025]
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "ld", "doy", or "q_doy"
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

BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric"
NUMBER_OF_BOOTSTRAPS = 50
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

SUBSAMPLE_MODEL_TO_REFERENCE_LENGTH = False

# Include August 2023 in the observational August fit when 2023 is inside
# OBSERVATION_YEARS. This setting has no effect when 2023 is outside that range.
INCLUDE_STORM_HANS_IN_FIT = True

REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

FIGURE_DPI = 300
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 14

# Shared return-period range for all six panels.
RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e7
NUMBER_OF_RETURN_PERIODS = 500

PRECIPITATION_YMIN = 0.0
PRECIPITATION_YMAX = 200.0

SHOW_GRID = True
WRITE_TO_FILE = True
SHOW_FIGURE = False

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

OBSERVATION_COLOR = "tab:orange"
MODEL_COLOR = 'tab:blue'
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
INTERVAL_CAP_WIDTH = 0.10
MEDIAN_MARKER_SIZE = 5.5

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
    if PLOT_METRIC not in {"return_period", "aep"}:
        raise ValueError("PLOT_METRIC must be 'return_period' or 'aep'.")
    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")
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
    return {"senorge": "Senorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_reference_label():
    """Return the reference label including the fitted year range."""
    return f"{get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"


def get_model_label():
    """Return the display label for the selected model data."""
    return "Model" if MODEL_DATA_METHOD == "raw" else "Model BC"


def get_record_label():
    """Return the calendar-record label for the configured observation range."""
    return "Monthly record excluding Storm Hans"


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


def lead_split_filename_label():
    """Return the lead-bin label used by the compact model filename."""
    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    split_text = "_".join(f"{start}-{end}" for start, end in build_lead_bins())
    return (
        f"lead{first_usable_lead}-{LAST_INPUT_LEAD}_"
        f"split{NUMBER_OF_LEAD_BINS}_{split_text}"
    )


def make_reference_filename():
    """Construct the selected observational input filename."""
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
    """Construct the compact model filename written by the sample-building script."""
    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)

    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )

    if MODEL_DATA_METHOD == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
        )

    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction_label}.nc"


def make_figure_filename():
    """Construct the six-panel output figure filename."""
    model_label = "model-raw" if MODEL_DATA_METHOD == "raw" else f"model-bc-{MODEL_DATA_METHOD}"
    hans_fit_label = "with-hans-fit" if INCLUDE_STORM_HANS_IN_FIT else "without-hans-fit"

    return Path(config.dirs["fig"]) / (
        f"fig-04-{PLOT_METRIC}-{TOP_DISTRIBUTION}-{model_label}-"
        f"{FORECAST_DATE_RANGE[0]}-{FORECAST_DATE_RANGE[-1]}-{REFERENCE_DATASET}-"
        f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}-{hans_fit_label}.png"
    )


# =============================================================================
# Data reading
# =============================================================================

def read_reference_month(month):
    """Read a complete reference-month sample and its two event thresholds."""
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

    record_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST and hans_in_observation_range:
        record_mask &= years != STORM_HANS_YEAR

    record_values = values[record_mask]
    record_years = years[record_mask]

    fit_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST and hans_in_observation_range and not INCLUDE_STORM_HANS_IN_FIT:
        fit_mask &= years != STORM_HANS_YEAR

    fit_values = values[fit_mask]
    fit_years = years[fit_mask]
    if fit_values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain in the fit."
        )
    if record_values.size == 0:
        raise ValueError(f"No values remain for the {MONTH_NAMES[month - 1]} record.")

    record_index = int(np.argmax(record_values))
    return {
        "fit_values": fit_values,
        "fit_years": fit_years,
        "storm_hans_value": storm_hans_value,
        "record_value": float(record_values[record_index]),
        "record_year": int(record_years[record_index]),
    }


def read_model_month(month):
    """Read one calendar-month model sample from sample_month(YYYYMM)."""
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


def subsample_model_values(values, reference_size, random_seed):
    """Optionally subsample model values to the reference sample size."""
    if not SUBSAMPLE_MODEL_TO_REFERENCE_LENGTH:
        return values
    if values.size < reference_size:
        raise ValueError("The model sample is smaller than the reference sample.")

    rng = np.random.default_rng(random_seed)
    return values[rng.choice(values.size, size=reference_size, replace=False)]


# =============================================================================
# Shared distribution methods
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
        return genextreme.rvs(
            shape, loc=location, scale=scale, size=sample_size, random_state=rng
        )
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


def horizon_aep(probability):
    """Convert annual exceedance probability to AEP_YEARS exceedance probability."""
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(-np.expm1(AEP_YEARS * np.log1p(-probability)))


# =============================================================================
# Shared bootstrap analyses
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
            sample = make_bootstrap_sample(
                values,
                method,
                rng,
                fitted_parameters=parameters,
            )
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

    return {
        "parameters": parameters,
        "bootstrap_parameters": bootstrap_parameters,
    }


def build_month_analysis(month, month_index, progress=None):
    """Read one month and create shared reference/model bootstrap fits."""
    reference = read_reference_month(month)
    model_values = read_model_month(month)
    model_values = subsample_model_values(
        model_values,
        reference["fit_values"].size,
        RANDOM_SEED + 100 * month_index,
    )

    samples = {
        "reference": reference["fit_values"],
        "model": model_values,
    }
    bootstrap = {}

    for group_index, (group, values) in enumerate(samples.items()):
        for method_index, method in enumerate(METHODS):
            seed = (
                RANDOM_SEED
                + 10_000 * month_index
                + 1_000 * group_index
                + method_index
            )
            bootstrap[(group, method)] = bootstrap_distribution(
                values, method, seed, progress
            )

    return {
        "reference": reference,
        "samples": samples,
        "bootstrap": bootstrap,
    }


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
    upper = np.percentile(
        bootstrap_levels,
        100.0 * (1.0 - alpha / 2.0),
        axis=0,
    )
    empirical_rp, empirical_values = empirical_return_periods(values)

    return {
        "values": values,
        "parameters": fit["parameters"],
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


def metric_samples_from_probabilities(probabilities):
    """Convert bootstrap probabilities while retaining unbounded return periods."""
    probabilities = np.asarray(probabilities, dtype=float)

    if PLOT_METRIC == "aep":
        samples = 100.0 * np.array([horizon_aep(value) for value in probabilities])
        return samples[np.isfinite(samples)]

    samples = np.array(
        [return_period_from_probability(value) for value in probabilities],
        dtype=float,
    )
    return samples[~np.isnan(samples)]


def analyse_event_metric(fit, event_value, method):
    """Evaluate one event threshold using reusable fitted bootstrap parameters."""
    probability = exceedance_probability(event_value, fit["parameters"], method)
    bootstrap_probabilities = np.array(
        [
            exceedance_probability(event_value, parameters, method)
            for parameters in fit["bootstrap_parameters"]
        ]
    )
    metric_samples = metric_samples_from_probabilities(bootstrap_probabilities)
    if metric_samples.size == 0:
        raise RuntimeError(f"{method} produced no valid bootstrap metric values.")

    return {
        "probability": probability,
        "return_period": return_period_from_probability(probability),
        "metric_samples": metric_samples,
    }


def calculate_metric_panel(month_analysis, month, threshold_type):
    """Calculate one lower-row panel from shared bootstrap fits."""
    reference = month_analysis["reference"]
    event_value = (
        reference["storm_hans_value"]
        if threshold_type == "storm_hans"
        else reference["record_value"]
    )

    results = {}
    for group in ["reference", "model"]:
        for method in METHODS:
            results[(group, method)] = analyse_event_metric(
                month_analysis["bootstrap"][(group, method)],
                event_value,
                method,
            )

    return {
        "month": month,
        "threshold_type": threshold_type,
        "event_value": event_value,
        "record_year": reference["record_year"],
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
    lower = return_period_to_aep_percent(RETURN_PERIOD_MAX)
    upper = return_period_to_aep_percent(RETURN_PERIOD_MIN)
    return lower, upper


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
        axis.set_xlabel(
            f"{AEP_YEARS}-year exceedance probability [%]",
            fontsize=AXIS_LABELSIZE,
        )

    axis.set_ylim(PRECIPITATION_YMIN, PRECIPITATION_YMAX)
    axis.set_ylabel(
        f"Monthly maximum {X_DAYS}-day precipitation [mm]",
        fontsize=AXIS_LABELSIZE,
    )
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_top_panel(axis, panel_label, month, result, return_periods, show_legend=False):
    """Plot one return-level distribution panel with reference data on top."""

    x_values = top_x_values(return_periods)
    reference_analysis = result["reference_analysis"]
    model_analysis = result["model_analysis"]

    for analysis, color, zorder in [
        (model_analysis, MODEL_COLOR, 2),
        (reference_analysis, OBSERVATION_COLOR, 3),
    ]:
        axis.fill_between(
            x_values,
            analysis["lower"],
            analysis["upper"],
            color=color,
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
            zorder=zorder,
        )
        axis.plot(
            x_values,
            analysis["fitted_levels"],
            color=color,
            linewidth=CURVE_LINEWIDTH,
            zorder=zorder,
        )
        axis.scatter(
            top_x_values(analysis["empirical_rp"]),
            analysis["empirical_values"],
            facecolors="none",
            edgecolors=color,
            linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE,
            zorder=zorder,
        )

    reference = result["reference"]
    axis.axhline(
        reference["storm_hans_value"],
        color=STORM_HANS_COLOR,
        linestyle=STORM_HANS_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=4,
    )
    axis.axhline(
        reference["record_value"],
        color=RECORD_COLOR,
        linestyle=RECORD_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        zorder=4,
    )

    format_top_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[month - 1]} {TOP_DISTRIBUTION} fit",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=OBSERVATION_COLOR,
                linewidth=CURVE_LINEWIDTH,
                label=get_reference_label(),
            ),
            Line2D(
                [0],
                [0],
                color=MODEL_COLOR,
                linewidth=CURVE_LINEWIDTH,
                label=get_model_label(),
            ),
            Line2D(
                [0],
                [0],
                color=STORM_HANS_COLOR,
                linestyle=STORM_HANS_LINESTYLE,
                linewidth=REFERENCE_LINEWIDTH,
                label="Storm Hans August 2023",
            ),
            Line2D(
                [0],
                [0],
                color=RECORD_COLOR,
                linestyle=RECORD_LINESTYLE,
                linewidth=REFERENCE_LINEWIDTH,
                label=get_record_label(),
            ),
        ]
        axis.legend(
            handles=handles,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            loc="upper left",
        )
        

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
        axis.set_ylim(RETURN_PERIOD_MIN, 1.15*RETURN_PERIOD_MAX)
        axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))

    axis.set_xlim(-0.55, 1.55)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([get_reference_name(), get_model_label()])
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


def plot_metric_limit_marker(axis, position, upper=True, color="black"):
    """Mark a confidence limit that extends beyond the plotted metric range."""
    if PLOT_METRIC == "return_period":
        limit = RETURN_PERIOD_MAX if upper else RETURN_PERIOD_MIN
        direction = 1 if upper else -1
    else:
        aep_min, aep_max = get_aep_limits()
        limit = aep_max if upper else aep_min
        direction = 1 if upper else -1

    axis.annotate(
        "",
        xy=(position, limit),
        xytext=(position, limit / 1.6 if direction > 0 else limit * 1.6),
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": INTERVAL_LINEWIDTH},
        annotation_clip=False,
        zorder=5,
    )


def clip_metric(value):
    """Clip one plotted lower-panel metric value to the shared metric limits."""
    if PLOT_METRIC == "aep":
        aep_min, aep_max = get_aep_limits()
        return float(np.clip(value, aep_min, aep_max))
    return float(np.clip(value, RETURN_PERIOD_MIN, RETURN_PERIOD_MAX))


def plot_metric_panel(axis, panel_label, panel_output, show_legend=False):
    """Plot IQR boxes and central 95% whiskers, preserving unbounded limits."""
    pending_limit_markers = []

    for group_index, group in enumerate(["reference", "model"]):
        for method in METHODS:
            analysis = panel_output["results"][(group, method)]
            position = group_index + METHOD_OFFSETS[method]
            summary = summarize_boxplot_samples(analysis["metric_samples"])
            color = METHOD_COLORS[method]

            upper_unbounded = np.isposinf(summary["whishi"])
            lower_unbounded = np.isneginf(summary["whislo"])

            plot_summary = {
                "label": "",
                "q1": clip_metric(summary["q1"]),
                "med": clip_metric(summary["median"]),
                "q3": clip_metric(summary["q3"]),
                "whislo": clip_metric(summary["whislo"]),
                "whishi": RETURN_PERIOD_MAX if upper_unbounded else clip_metric(summary["whishi"]),
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

            if upper_unbounded:
                pending_limit_markers.append((position, True, color))
            if lower_unbounded:
                pending_limit_markers.append((position, False, color))

    configure_metric_axis(axis)

    #for position, upper, color in pending_limit_markers:
    #    plot_metric_limit_marker(axis, position, upper=upper, color=color)
    month = panel_output["month"]
    title = (
        f"{MONTH_NAMES[month - 1]}: Storm Hans threshold"
        if panel_output["threshold_type"] == "storm_hans"
        else f"{MONTH_NAMES[month - 1]}: record excluding Storm Hans"
    )
    axis.set_title(
        f"{panel_label}) {title}",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D([0], [0], linestyle="-", color=METHOD_COLORS[method],
                   linewidth=INTERVAL_LINEWIDTH, label=method)
            for method in METHODS
        ]
        axis.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")


# =============================================================================
# Figure and reporting
# =============================================================================

def print_summary(top_results, metric_results):
    """Print the key input and threshold information."""
    print("Selected settings")
    print("-----------------")
    print(f"Reference dataset: {get_reference_label()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")
    print(f"Model file:        {make_model_filename()}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Top distribution:  {TOP_DISTRIBUTION}")
    print(f"Metric:            {PLOT_METRIC}")
    print(f"Bootstrap method:  {BOOTSTRAP_METHOD}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS}")
    print(f"Include Hans fit:  {INCLUDE_STORM_HANS_IN_FIT}")

    for month in PANEL_MONTHS:
        reference = top_results[month]["reference"]
        hans_in_observation_range = (
            OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
        )
        record_range = (
            f"{OBSERVATION_YEARS[0]}-{STORM_HANS_YEAR - 1}"
            if month == AUGUST and hans_in_observation_range
            else f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
        )
        print()
        print(f"{MONTH_NAMES[month - 1]} reference fit: {reference['fit_values'].size} values")
        print(
            f"{MONTH_NAMES[month - 1]} record {record_range}: "
            f"{reference['record_value']:.3f} mm ({reference['record_year']})"
        )
        print(f"Storm Hans threshold: {reference['storm_hans_value']:.3f} mm")


def make_figure(top_results, metric_results, return_periods):
    """Create the combined 3 x 2 figure."""
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
        3, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True
    )

    # August is always the left column; May is always the right column.
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
    """Run the six-panel May/August analysis."""
    validate_settings()
    return_periods = make_return_period_grid()

    total_bootstraps = (
        len(PANEL_MONTHS) * 2 * len(METHODS) * NUMBER_OF_BOOTSTRAPS
    )
    progress = ProgressTracker(total_bootstraps)
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
        (month, threshold_type): calculate_metric_panel(
            month_analyses[month],
            month,
            threshold_type,
        )
        for month, threshold_type in [
            (MAY, "storm_hans"),
            (AUGUST, "storm_hans"),
            (MAY, "calendar_record"),
            (AUGUST, "calendar_record"),
        ]
    }

    print_summary(top_results, metric_results)
    make_figure(top_results, metric_results, return_periods)


if __name__ == "__main__":
    main()
