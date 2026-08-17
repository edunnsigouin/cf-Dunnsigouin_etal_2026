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

The reference fit always uses the complete OBSERVATION_YEARS range. Calendar-record
thresholds use the same range, except August 2023 is excluded from the August
record so Storm Hans does not define its own comparison threshold. May 2023 is
retained in the May record calculation.

The compact model input is expected to contain sample_month(i_date) as YYYYMM and
precipitation maxima with dimensions (number, i_date). Finite values are pooled,
so files containing padded 51-, 101-, and 11-member samples are handled directly.
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

OBSERVATION_YEARS = [1957, 2023]
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

MODEL_DATA_METHOD = "raw"  # "raw", "mm", "q", "ld", "doy", or "q_doy"
MODEL_SAMPLING_GROUP = "full"  # "full", "split1", "split2", ...
MODEL_VARIABLE = "tp24"

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Used for panels a-b. Options: "GEV", "Gumbel", "GenEx".
TOP_DISTRIBUTION = "Gumbel"

# Used by all panels. Options: "return_period" or "aep".
PLOT_METRIC = "return_period"
AEP_YEARS = 1

BOOTSTRAP_METHOD = "parametric"  # "nonparametric" or "parametric"
NUMBER_OF_BOOTSTRAPS = 10
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

SUBSAMPLE_MODEL_TO_REFERENCE_LENGTH = False

REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

WRITE_TO_FILE = False
SHOW_FIGURE = True
FIGURE_DPI = 300
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 14

MIN_RETURN_PERIOD = 1.01
MAX_RETURN_PERIOD = 10_000_000.0
NUMBER_OF_RETURN_PERIODS = 500

TOP_XMIN_RETURN_PERIOD = 1.0
TOP_XMAX_RETURN_PERIOD = 1.0e6
TOP_XMIN_AEP = 0.0001
TOP_XMAX_AEP = 100.0
PRECIPITATION_YMIN = 0.0
PRECIPITATION_YMAX = 200.0

METRIC_AEP_MIN_PERCENT = 0.0001
METRIC_AEP_MAX_PERCENT = 100.0
METRIC_RP_MIN_YEARS = 1.0
METRIC_RP_MAX_YEARS = 10_000_000.0

SHOW_GRID = True


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
METHOD_COLORS = {"GEV": "tab:blue", "Gumbel": "tab:orange", "GenEx": "tab:green"}
METHOD_OFFSETS = {"GEV": -0.18, "Gumbel": 0.0, "GenEx": 0.18}

OBSERVATION_COLOR = "tab:blue"
MODEL_COLOR = "goldenrod"
STORM_HANS_COLOR = "grey"
RECORD_COLOR = "grey"

STORM_HANS_LINESTYLE = "--"
RECORD_LINESTYLE = ":"
CONFIDENCE_ALPHA = 0.2
CURVE_LINEWIDTH = 2.0
REFERENCE_LINEWIDTH = 2.0
MARKER_SIZE = 35
MARKER_LINEWIDTH = 1.0

INTERVAL_LINEWIDTH = 1.4
INTERVAL_CAP_WIDTH = 0.10
MEDIAN_MARKER_SIZE = 5.5

AXIS_LABELSIZE = 11
TICK_LABELSIZE = 10
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 9


# =============================================================================
# Validation, labels, and filenames
# =============================================================================

def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if MODEL_DATA_METHOD not in {"raw", "mm", "q", "ld", "doy", "q_doy"}:
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


def get_reference_name():
    """Return the display name of the selected reference dataset."""
    return {"senorge": "SeNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_reference_label():
    """Return the reference label including the fitted year range."""
    return f"{get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"


def get_model_label():
    """Return the display label for the selected model data."""
    if MODEL_DATA_METHOD == "raw":
        return "Model raw"
    return f"Model BC ({MODEL_DATA_METHOD})"


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

    if REFERENCE_DATASET == "senorge":
        filename = (
            f"distribution_monthly_extremes_{SENORGE_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_senorge_{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
        )
        return Path(config.dirs["senorge_processed"]) / filename

    filename = (
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{X_DAYS}dayacc_{CATCHMENT}_"
        f"era5_{ERA5_GRID}_{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
    )
    return Path(config.dirs["era5_processed"]) / filename


def make_model_filename():
    """Construct the selected raw or bias-corrected compact model filename."""
    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)

    filename = (
        f"test-monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{lead_split_filename_label()}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    if MODEL_DATA_METHOD != "raw":
        filename += f"_bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}"
    return Path(config.dirs["s2s_processed"]) / f"{filename}.nc"


def make_figure_filename():
    """Construct the six-panel output figure filename."""
    model_label = "raw" if MODEL_DATA_METHOD == "raw" else f"bc-{MODEL_DATA_METHOD}"
    return Path(config.dirs["fig"]) / (
        f"fig-combined_extremes_{PLOT_METRIC}_{model_label}-{REFERENCE_DATASET}.png"
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

    # The fit uses the full requested time series. Only the August record threshold
    # excludes August 2023 so Storm Hans does not define its own record threshold.
    record_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST:
        record_mask &= years != STORM_HANS_YEAR

    record_values = values[record_mask]
    record_years = years[record_mask]
    if record_values.size == 0:
        raise ValueError(f"No values remain for the {MONTH_NAMES[month - 1]} record.")

    record_index = int(np.argmax(record_values))
    return {
        "fit_values": values,
        "fit_years": years,
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


def fit_distribution(values, method):
    """Fit one supported extreme-value distribution."""
    if method == "GEV":
        parameters = genextreme.fit(values)
    elif method == "Gumbel":
        parameters = gumbel_r.fit(values)
    elif method == "GenEx":
        positive = values[values > 0]
        if np.any(values < 0) or positive.size == 0:
            raise ValueError("GenEx requires non-negative values with at least one positive value.")
        result = minimize(
            genex_negative_log_likelihood,
            x0=np.log([1.0, np.mean(positive)]),
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
# Top-row return-level analyses
# =============================================================================

def make_return_period_grid():
    """Return the logarithmic grid used by panels a-b."""
    return np.geomspace(MIN_RETURN_PERIOD, MAX_RETURN_PERIOD, NUMBER_OF_RETURN_PERIODS)


def analyse_return_levels(values, return_periods, method, random_seed):
    """Fit one distribution and bootstrap its return-level curve."""
    parameters = fit_distribution(values, method)
    probabilities = 1.0 - 1.0 / return_periods
    fitted_levels = distribution_ppf(probabilities, parameters, method)

    rng = np.random.default_rng(random_seed)
    bootstrap_levels = np.full((NUMBER_OF_BOOTSTRAPS, return_periods.size), np.nan)
    base_parameters = parameters if BOOTSTRAP_METHOD == "parametric" else None

    for index in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = make_bootstrap_sample(values, method, rng, base_parameters)
            fitted = fit_distribution(sample, method)
            levels = distribution_ppf(probabilities, fitted, method)
            if np.isfinite(levels).all():
                bootstrap_levels[index] = levels
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    successful = np.isfinite(bootstrap_levels).all(axis=1).sum()
    minimum = int(np.ceil(MIN_SUCCESSFUL_BOOTSTRAP_FRACTION * NUMBER_OF_BOOTSTRAPS))
    if successful < minimum:
        raise RuntimeError(
            f"Only {successful} of {NUMBER_OF_BOOTSTRAPS} {method} bootstrap fits succeeded."
        )

    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = np.nanpercentile(bootstrap_levels, 100.0 * alpha / 2.0, axis=0)
    upper = np.nanpercentile(bootstrap_levels, 100.0 * (1.0 - alpha / 2.0), axis=0)
    empirical_rp, empirical_values = empirical_return_periods(values)

    return {
        "values": values,
        "parameters": parameters,
        "fitted_levels": fitted_levels,
        "lower": lower,
        "upper": upper,
        "empirical_rp": empirical_rp,
        "empirical_values": empirical_values,
    }


def analyse_top_month(month, month_index, return_periods):
    """Prepare reference and model return-level analyses for one month."""
    reference = read_reference_month(month)
    model_values = read_model_month(month)
    model_values = subsample_model_values(
        model_values, reference["fit_values"].size, RANDOM_SEED + 100 * month_index
    )

    return {
        "reference": reference,
        "reference_analysis": analyse_return_levels(
            reference["fit_values"], return_periods, TOP_DISTRIBUTION,
            RANDOM_SEED + 100 * month_index + 1,
        ),
        "model_analysis": analyse_return_levels(
            model_values, return_periods, TOP_DISTRIBUTION,
            RANDOM_SEED + 100 * month_index + 2,
        ),
    }


# =============================================================================
# Lower-row return-metric analyses
# =============================================================================

def bootstrap_event_metric(values, event_value, method, random_seed):
    """Bootstrap exceedance probabilities for one sample/event pair."""
    rng = np.random.default_rng(random_seed)
    probabilities = np.full(NUMBER_OF_BOOTSTRAPS, np.nan)
    base_parameters = (
        fit_distribution(values, method) if BOOTSTRAP_METHOD == "parametric" else None
    )

    for index in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = make_bootstrap_sample(values, method, rng, base_parameters)
            parameters = fit_distribution(sample, method)
            probabilities[index] = exceedance_probability(event_value, parameters, method)
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    probabilities = probabilities[np.isfinite(probabilities)]
    minimum = int(np.ceil(MIN_SUCCESSFUL_BOOTSTRAP_FRACTION * NUMBER_OF_BOOTSTRAPS))
    if probabilities.size < minimum:
        raise RuntimeError(
            f"Only {probabilities.size} of {NUMBER_OF_BOOTSTRAPS} {method} "
            "bootstrap fits succeeded."
        )
    return probabilities


def summarize_metric_probabilities(probabilities):
    """Return the bootstrap interval in the selected plotting metric."""
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    if PLOT_METRIC == "aep":
        samples = 100.0 * np.array([horizon_aep(value) for value in probabilities])
        return np.percentile(samples, [lower_percentile, 50.0, upper_percentile])

    # Return period is the inverse of exceedance probability. Calculate quantiles
    # in finite probability space first, then invert them. This avoids NumPy
    # interpolation warnings when zero probabilities imply infinite return periods.
    probability_high, probability_median, probability_low = np.percentile(
        probabilities,
        [upper_percentile, 50.0, lower_percentile],
    )
    return np.array(
        [
            return_period_from_probability(probability_high),
            return_period_from_probability(probability_median),
            return_period_from_probability(probability_low),
        ]
    )


def analyse_event_metric(values, event_value, method, random_seed):
    """Fit and bootstrap one return metric for an event threshold."""
    parameters = fit_distribution(values, method)
    probability = exceedance_probability(event_value, parameters, method)
    bootstrap_probabilities = bootstrap_event_metric(
        values, event_value, method, random_seed
    )
    low, median, high = summarize_metric_probabilities(bootstrap_probabilities)

    return {
        "probability": probability,
        "return_period": return_period_from_probability(probability),
        "low": low,
        "median": median,
        "high": high,
    }


def calculate_metric_panel(month, threshold_type, panel_index):
    """Calculate one lower-row panel for May or August."""
    reference = read_reference_month(month)
    model_values = read_model_month(month)
    event_value = (
        reference["storm_hans_value"]
        if threshold_type == "storm_hans"
        else reference["record_value"]
    )

    samples = {"reference": reference["fit_values"], "model": model_values}
    results = {}
    for group_index, (group, values) in enumerate(samples.items()):
        for method_index, method in enumerate(METHODS):
            seed = RANDOM_SEED + 10_000 * panel_index + 100 * group_index + method_index
            results[(group, method)] = analyse_event_metric(
                values, event_value, method, seed
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

def top_x_values(return_periods):
    """Convert return periods to the selected top-row x coordinate."""
    if PLOT_METRIC == "return_period":
        return return_periods

    annual_probability = 1.0 / return_periods
    probability = -np.expm1(AEP_YEARS * np.log1p(-annual_probability))
    return 100.0 * probability


def format_top_axis(axis):
    """Format a return-level panel."""
    axis.set_xscale("log")
    if PLOT_METRIC == "return_period":
        axis.set_xlim(TOP_XMIN_RETURN_PERIOD, TOP_XMAX_RETURN_PERIOD)
        axis.set_xlabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    else:
        axis.set_xlim(TOP_XMAX_AEP, TOP_XMIN_AEP)
        axis.set_xlabel(f"{AEP_YEARS}-year exceedance probability [%]", fontsize=AXIS_LABELSIZE)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}" if value > 0 else ""))

    axis.set_ylim(PRECIPITATION_YMIN, PRECIPITATION_YMAX)
    axis.set_ylabel(f"Monthly maximum {X_DAYS}-day precipitation [mm]", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_top_panel(axis, panel_label, month, result, return_periods, show_legend=False):
    """Plot one return-level distribution panel."""
    x_values = top_x_values(return_periods)
    reference_analysis = result["reference_analysis"]
    model_analysis = result["model_analysis"]

    for analysis, color in [
        (reference_analysis, OBSERVATION_COLOR),
        (model_analysis, MODEL_COLOR),
    ]:
        axis.fill_between(
            x_values, analysis["lower"], analysis["upper"],
            color=color, alpha=CONFIDENCE_ALPHA, linewidth=0, zorder=1,
        )
        axis.plot(x_values, analysis["fitted_levels"], color=color, linewidth=CURVE_LINEWIDTH)
        axis.scatter(
            top_x_values(analysis["empirical_rp"]),
            analysis["empirical_values"],
            facecolors="none", edgecolors=color, linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE, zorder=3,
        )

    reference = result["reference"]
    axis.axhline(
        reference["storm_hans_value"], color=STORM_HANS_COLOR,
        linestyle=STORM_HANS_LINESTYLE, linewidth=REFERENCE_LINEWIDTH,
    )
    axis.axhline(
        reference["record_value"], color=RECORD_COLOR,
        linestyle=RECORD_LINESTYLE, linewidth=REFERENCE_LINEWIDTH,
    )

    format_top_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[month - 1]}", loc="left",
        fontsize=TITLE_FONTSIZE, fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D([0], [0], color=OBSERVATION_COLOR, linewidth=CURVE_LINEWIDTH,
                   label=get_reference_label()),
            Line2D([0], [0], color=MODEL_COLOR, linewidth=CURVE_LINEWIDTH,
                   label=get_model_label()),
            Line2D([0], [0], color=STORM_HANS_COLOR, linestyle=STORM_HANS_LINESTYLE,
                   linewidth=REFERENCE_LINEWIDTH, label="Storm Hans, August 2023"),
            Line2D([0], [0], color=RECORD_COLOR, linestyle=RECORD_LINESTYLE,
                   linewidth=REFERENCE_LINEWIDTH,
                   label=f"Pre-Hans {MONTH_NAMES[month - 1]} record"),
        ]
        axis.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")


def metric_axis_label():
    """Return the lower-row y-axis label."""
    if PLOT_METRIC == "aep":
        return f"{AEP_YEARS}-year exceedance probability [%]"
    return "Return period [years]"


def format_metric_tick(value, _position):
    """Format lower-row logarithmic metric ticks."""
    if value <= 0:
        return ""
    return f"{value:g}"


def configure_metric_axis(axis):
    """Apply common formatting to a lower-row return-metric panel."""
    axis.set_yscale("log")
    if PLOT_METRIC == "aep":
        axis.set_ylim(0.7 * METRIC_AEP_MIN_PERCENT, 1.3 * METRIC_AEP_MAX_PERCENT)
    else:
        axis.set_ylim(0.7 * METRIC_RP_MIN_YEARS, 1.3 * METRIC_RP_MAX_YEARS)

    axis.set_xlim(-0.55, 1.55)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([get_reference_name(), get_model_label()])
    axis.set_ylabel(metric_axis_label(), fontsize=AXIS_LABELSIZE)
    axis.yaxis.set_major_formatter(FuncFormatter(format_metric_tick))
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.7, alpha=0.45)


def clip_metric(value):
    """Clip one plotted lower-panel metric value to the configured limits."""
    if PLOT_METRIC == "aep":
        return float(np.clip(value, METRIC_AEP_MIN_PERCENT, METRIC_AEP_MAX_PERCENT))
    return float(np.clip(value, METRIC_RP_MIN_YEARS, METRIC_RP_MAX_YEARS))


def plot_metric_panel(axis, panel_label, panel_output, show_legend=False):
    """Plot bootstrap intervals for GEV, Gumbel, and GenEx."""
    for group_index, group in enumerate(["reference", "model"]):
        for method in METHODS:
            analysis = panel_output["results"][(group, method)]
            position = group_index + METHOD_OFFSETS[method]
            low = clip_metric(analysis["low"])
            median = clip_metric(analysis["median"])
            high = clip_metric(analysis["high"])
            color = METHOD_COLORS[method]

            axis.vlines(position, low, high, color=color, linewidth=INTERVAL_LINEWIDTH)
            axis.hlines(
                [low, high], position - INTERVAL_CAP_WIDTH / 2,
                position + INTERVAL_CAP_WIDTH / 2, color=color,
                linewidth=INTERVAL_LINEWIDTH,
            )
            axis.plot(
                position, median, marker="o", markersize=MEDIAN_MARKER_SIZE,
                markerfacecolor=color, markeredgecolor=color, linestyle="none",
            )

    configure_metric_axis(axis)
    threshold_label = (
        "Storm Hans threshold"
        if panel_output["threshold_type"] == "storm_hans"
        else "calendar-month record threshold"
    )
    month = panel_output["month"]
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[month - 1]} — {threshold_label}",
        loc="left", fontsize=TITLE_FONTSIZE, fontweight="normal",
    )

    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="-", color=METHOD_COLORS[method],
                   markersize=MEDIAN_MARKER_SIZE, label=method)
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

    for month in PANEL_MONTHS:
        reference = top_results[month]["reference"]
        record_range = (
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
            if month != AUGUST
            else f"{OBSERVATION_YEARS[0]}-{STORM_HANS_YEAR - 1}"
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
    plot_top_panel(
        axes[0, 1], "b", MAY, top_results[MAY], return_periods, show_legend=True
    )

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

    top_results = {
        month: analyse_top_month(month, index, return_periods)
        for index, month in enumerate(PANEL_MONTHS)
    }

    metric_results = {}
    panel_specs = [
        (MAY, "storm_hans"),
        (AUGUST, "storm_hans"),
        (MAY, "calendar_record"),
        (AUGUST, "calendar_record"),
    ]
    for panel_index, spec in enumerate(panel_specs):
        metric_results[spec] = calculate_metric_panel(*spec, panel_index)

    print_summary(top_results, metric_results)
    make_figure(top_results, metric_results, return_periods)


if __name__ == "__main__":
    main()
