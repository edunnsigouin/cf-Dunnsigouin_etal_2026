"""
Create a 2 x 2 figure comparing extreme-precipitation return metrics for one
reference dataset and one model dataset. The panels show the Storm Hans 2023
threshold and calendar-record threshold for two user-selected calendar months.
Month 0 retains the annual any-month calculation.

For each panel and dataset, GEV, Gumbel, and GenEx are fitted to the same sample.
Bootstrap uncertainty can be estimated in two ways:

    "nonparametric"
        Draw each bootstrap sample with replacement from the original values at
        the original sample size, then refit the selected distribution.

    "parametric"
        Fit the selected distribution once to the original values, simulate each
        same-size bootstrap sample from that fitted distribution, then refit it.

Both methods therefore propagate fitting uncertainty through repeated refitting.

The plotted uncertainty summaries show the central 95% bootstrap interval for
the selected return metric for each fitted distribution. A vertical line with
end caps spans the 2.5th-97.5th percentiles, and a circle marks the bootstrap
median. GEV, Gumbel, and GenEx are distinguished by color and positioned side by
side, centered on each dataset label.

For annual / any-month panels, each bootstrap replicate generates a new sample
for all 12 months using the selected bootstrap method, refits every month, and
then combines the 12 monthly exceedance probabilities into one annual probability
before calculating the return period.

Exactly one REFERENCE_DATASET is used throughout. It supplies the observational
sample and event thresholds and is also the reference encoded in a
bias-corrected model filename.
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

PLOT_METRIC = "return_period"  # "aep" or "return_period"
N_AEP_YEARS = 1  # Used only when PLOT_METRIC == "aep".

REFERENCE_DATASET = "senorge"  # "senorge" or "era5"

# Model data to plot:
#     "raw", "mm", "q", "ld", "doy", or "q_doy"
MODEL_DATA_METHOD = "raw"

# Calendar months for the two figure rows. Use 0 for annual / any month.
AUGUST_MONTH = 8
MAY_MONTH = 5

STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

RECORD_START_YEAR = 1957
AUGUST_RECORD_END_YEAR = 2022
MAY_RECORD_END_YEAR = 2022

OBSERVATION_YEARS = ["1957", "2023"]
EXCLUDE_2023_FROM_REFERENCE_FIT = True

CATCHMENT = "regine_drammen"
X_DAYS = 2
MODEL_VARIABLE = "tp24"

FORECAST_DATE_RANGE = ["2020-01-02", "2022-12-29"]
FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2
MODEL_SAMPLING_GROUP = "full"

REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

# Bootstrap method:
#     "nonparametric" -> resample original values with replacement
#     "parametric"    -> fit once, simulate from the fitted distribution, then refit
BOOTSTRAP_METHOD = "nonparametric"

NUMBER_OF_BOOTSTRAPS = 50
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90

FIG_WIDTH_IN = 10.5
FIG_HEIGHT_IN = 8.0
FIGURE_DPI = 400

# Bootstrap interval layout. Method offsets average to zero, so the three
# methods are centered on each reference/model dataset label.
METHOD_COLORS = {
    "GEV": "tab:blue",
    "Gumbel": "tab:orange",
    "GenEx": "tab:green",
}
METHOD_OFFSETS = {"GEV": -0.18, "Gumbel": 0.0, "GenEx": 0.18}
METHODS = list(METHOD_COLORS)

INTERVAL_LINEWIDTH = 1.4
INTERVAL_CAP_WIDTH = 0.10
MEDIAN_MARKER_SIZE = 5.5
MEDIAN_MARKER_EDGE_WIDTH = 0.9

AXIS_LABELSIZE = 12
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 10

AEP_YMIN_PERCENT = 0.0001
AEP_YMAX_PERCENT = 100.0
RETURN_PERIOD_YMIN_YEARS = 1.0
RETURN_PERIOD_YMAX_YEARS = 10_000_000.0

SHARE_Y_AXES = False
SHOW_GRID = True
WRITE_TO_FILE = True
SHOW_FIGURE = True


# =============================================================================
# Constants
# =============================================================================

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PLOT_GROUPS = ["reference", "model"]


# =============================================================================
# Validation and labels
# =============================================================================

def validate_user_settings():
    """Validate user-configurable settings."""

    if PLOT_METRIC not in {"aep", "return_period"}:
        raise ValueError("PLOT_METRIC must be 'aep' or 'return_period'.")

    if not isinstance(N_AEP_YEARS, int) or N_AEP_YEARS < 1:
        raise ValueError("N_AEP_YEARS must be a positive integer.")

    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")

    valid_model_methods = {"raw", "mm", "q", "ld", "doy", "q_doy"}
    if MODEL_DATA_METHOD not in valid_model_methods:
        raise ValueError(f"MODEL_DATA_METHOD must be one of {sorted(valid_model_methods)}.")

    for name, month in {"AUGUST_MONTH": AUGUST_MONTH, "MAY_MONTH": MAY_MONTH}.items():
        if not isinstance(month, int) or month not in range(13):
            raise ValueError(f"{name} must be 0 (annual) or an integer from 1 to 12.")

    if AUGUST_RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError("AUGUST_RECORD_END_YEAR must be >= RECORD_START_YEAR.")

    if MAY_RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError("MAY_RECORD_END_YEAR must be >= RECORD_START_YEAR.")

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError("X_DAYS is too large for the configured lead range.")

    usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1
    if not isinstance(NUMBER_OF_LEAD_BINS, int) or not 1 <= NUMBER_OF_LEAD_BINS <= usable_leads:
        raise ValueError("NUMBER_OF_LEAD_BINS is invalid for the usable lead range.")

    valid_groups = {"full", *(f"split{i}" for i in range(1, NUMBER_OF_LEAD_BINS + 1))}
    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}.")

    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")

    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError("NUMBER_OF_BOOTSTRAPS must be at least 1.")

    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must lie between 0 and 1.")

    if not 0 < MIN_SUCCESSFUL_BOOTSTRAP_FRACTION <= 1:
        raise ValueError("MIN_SUCCESSFUL_BOOTSTRAP_FRACTION must lie in (0, 1].")

    if set(METHOD_OFFSETS) != set(METHODS):
        raise ValueError("METHOD_OFFSETS must define one position for each method.")

    if set(METHOD_COLORS) != set(METHODS):
        raise ValueError("METHOD_COLORS must define one color for each method.")


def get_reference_name():
    """Return the publication-style reference dataset name."""

    return {"senorge": "SeNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the selected reference variable."""

    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_model_label():
    """Return the x-axis label for the selected model data."""

    if MODEL_DATA_METHOD == "raw":
        return "Model raw"
    return f"Model BC\n({MODEL_DATA_METHOD})"


def get_panel_month_label(month):
    """Return the calendar-month name or 'Annual' for month 0."""

    return "Annual" if month == 0 else MONTH_NAMES[month - 1]


# =============================================================================
# Filenames and lead ranges
# =============================================================================

def get_model_file_id(catchment_name):
    """Return the short catchment identifier used in model filenames."""

    return catchment_name.removeprefix("regine_")


def split_usable_accumulated_leads(first_lead, last_lead, number_of_bins):
    """Split usable ending leads into near-equal bins."""

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
    """Return configured model lead bins."""

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    return split_usable_accumulated_leads(
        first_usable_lead, LAST_INPUT_LEAD, NUMBER_OF_LEAD_BINS
    )


def get_full_lead_range():
    """Return the complete usable ending-lead range."""

    return FIRST_INPUT_LEAD + X_DAYS - 1, LAST_INPUT_LEAD


def get_selected_model_lead_range():
    """Return the lead range selected by MODEL_SAMPLING_GROUP."""

    if MODEL_SAMPLING_GROUP == "full":
        return get_full_lead_range()

    split_number = int(MODEL_SAMPLING_GROUP.removeprefix("split"))
    return build_lead_bins()[split_number - 1]


def get_model_variable():
    """Return the compact precipitation variable to read."""

    if MODEL_SAMPLING_GROUP == "full":
        return "tp24_max"

    lead_start, lead_end = get_selected_model_lead_range()
    return f"tp24_max_lead{lead_start}_{lead_end}"


def lead_split_filename_label():
    """Return the lead-bin label used in compact model filenames."""

    full_start, full_end = get_full_lead_range()
    split_text = "_".join(f"{start}-{end}" for start, end in build_lead_bins())
    return f"lead{full_start}-{full_end}_split{NUMBER_OF_LEAD_BINS}_{split_text}"


def make_reference_filename():
    """Construct the selected reference-data filename."""

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


def make_raw_model_filename():
    """Construct the raw compact model filename."""

    filename = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{lead_split_filename_label()}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}.nc"
    )
    return Path(config.dirs["s2s_processed"]) / filename


def make_model_filename():
    """Construct the selected raw or bias-corrected model filename."""

    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)

    raw_filename = make_raw_model_filename()
    if MODEL_DATA_METHOD == "raw":
        return raw_filename

    return raw_filename.with_name(
        f"{raw_filename.stem}_bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}.nc"
    )


def make_figure_filename():
    """Construct the output figure filename."""

    metric_label = (
        f"{N_AEP_YEARS}year-aep" if PLOT_METRIC == "aep" else "return-period"
    )
    model_label = (
        "raw" if MODEL_DATA_METHOD == "raw"
        else f"bc-{MODEL_DATA_METHOD}"
    )
    filename = f"fig-04_{metric_label}_{model_label}-{REFERENCE_DATASET}.png"
    return Path(config.dirs["fig"]) / filename


# =============================================================================
# Data reading
# =============================================================================

def read_reference_month(month, record_end_year):
    """Read one reference month and its Storm Hans and record thresholds."""

    filename = make_reference_filename()
    variable = get_reference_variable()

    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")

    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")

        selected = ds[variable].sel(
            year=slice(int(OBSERVATION_YEARS[0]), int(OBSERVATION_YEARS[1])),
            month=month,
        ).load()

        record_data = ds[variable].sel(
            year=slice(RECORD_START_YEAR, record_end_year), month=month
        ).load()

        storm_hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )

    years = np.asarray(selected["year"].values)
    values = np.asarray(selected.values, dtype=float)
    finite = np.isfinite(values)
    years, values = years[finite], values[finite]

    if EXCLUDE_2023_FROM_REFERENCE_FIT:
        values = values[years != STORM_HANS_YEAR]

    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {get_reference_name()} {MONTH_NAMES[month - 1]} "
            "values remain for fitting."
        )

    record_values = np.asarray(record_data.values, dtype=float)
    record_years = np.asarray(record_data["year"].values)
    finite = np.isfinite(record_values)

    if not finite.any():
        raise ValueError(f"No finite {MONTH_NAMES[month - 1]} record values were found.")

    record_values, record_years = record_values[finite], record_years[finite]
    record_index = int(np.argmax(record_values))

    return {
        "fit_values": values,
        "storm_hans_value": storm_hans_value,
        "record_value": float(record_values[record_index]),
        "record_year": int(record_years[record_index]),
    }


def read_model_month(month):
    """Read one calendar-month sample from the selected compact model file."""

    filename = make_model_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Model file not found: {filename}")

    variable = get_model_variable()
    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        if "month" not in ds:
            raise KeyError(f"Variable 'month' was not found in {filename}.")

        selected = ds[variable].where(ds["month"] == month, drop=True)
        values = np.asarray(selected.values, dtype=float).ravel()

    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite model values were found for {MONTH_NAMES[month - 1]}."
        )

    return values


# =============================================================================
# Extreme-value methods
# =============================================================================

def fit_gev(values):
    """Fit a stationary three-parameter GEV distribution."""

    shape_c, location, scale = genextreme.fit(values)
    if not np.isfinite([shape_c, location, scale]).all() or scale <= 0:
        raise RuntimeError("The GEV fit returned invalid parameters.")
    return shape_c, location, scale


def fit_gumbel(values):
    """Fit a stationary two-parameter Gumbel distribution."""

    location, scale = gumbel_r.fit(values)
    if not np.isfinite([location, scale]).all() or scale <= 0:
        raise RuntimeError("The Gumbel fit returned invalid parameters.")
    return location, scale


def genex_negative_log_likelihood(log_parameters, values):
    """Return the GenEx negative log-likelihood."""

    shape, scale = np.exp(log_parameters)
    if (
        not np.isfinite(shape)
        or not np.isfinite(scale)
        or shape <= 0
        or scale <= 0
        or np.any(values < 0)
    ):
        return np.inf

    z = values / scale
    log_pdf = np.log(shape) - np.log(scale) - z + (shape - 1.0) * np.log(-np.expm1(-z))
    return np.inf if not np.isfinite(log_pdf).all() else -np.sum(log_pdf)


def fit_genex(values):
    """Fit a two-parameter Generalized Exponential distribution."""

    if np.any(values < 0):
        raise ValueError("GenEx requires non-negative values.")

    positive_values = values[values > 0]
    if positive_values.size == 0:
        raise RuntimeError("GenEx cannot be fitted without positive values.")

    result = minimize(
        genex_negative_log_likelihood,
        x0=np.log([1.0, np.mean(positive_values)]),
        args=(values,),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    if not result.success:
        raise RuntimeError(f"GenEx fit failed: {result.message}")

    shape, scale = np.exp(result.x)
    if not np.isfinite([shape, scale]).all() or shape <= 0 or scale <= 0:
        raise RuntimeError("The GenEx fit returned invalid parameters.")

    return shape, scale


def calculate_exceedance_probability(values, event_value, method):
    """Fit one distribution and calculate exceedance probability."""

    if method == "GEV":
        shape_c, location, scale = fit_gev(values)
        probability = genextreme.sf(
            event_value, shape_c, loc=location, scale=scale
        )
    elif method == "Gumbel":
        location, scale = fit_gumbel(values)
        probability = gumbel_r.sf(event_value, loc=location, scale=scale)
    elif method == "GenEx":
        shape, scale = fit_genex(values)
        probability = 1.0 if event_value < 0 else 1.0 - (1.0 - np.exp(-event_value / scale)) ** shape
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not np.isfinite(probability):
        raise RuntimeError(f"{method} produced a non-finite probability.")

    return float(np.clip(probability, 0.0, 1.0))


def analyse_method(values, event_value, method):
    """Calculate exceedance probability and return period for the original sample."""

    probability = calculate_exceedance_probability(values, event_value, method)
    return_period = np.inf if probability <= 0 else 1.0 / probability
    return {"annual_exceedance_probability": probability, "return_period": return_period}


def combine_monthly_probabilities(monthly_probabilities):
    """Combine 12 monthly probabilities into an annual any-month probability."""

    probabilities = np.asarray(monthly_probabilities, dtype=float)
    if probabilities.size != 12 or not np.isfinite(probabilities).all():
        raise ValueError("Annual analysis requires 12 finite monthly probabilities.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    return float(np.clip(1.0 - np.prod(1.0 - probabilities), 0.0, 1.0))


def calculate_horizon_aep(annual_probability, horizon_years):
    """Convert annual exceedance probability to an N-year probability."""

    annual_probability = float(np.clip(annual_probability, 0.0, 1.0))
    return float(-np.expm1(horizon_years * np.log1p(-annual_probability)))


def return_period_from_probability(probability):
    """Convert annual exceedance probability to return period."""

    return np.inf if probability <= 0 else 1.0 / probability


def fit_method(values, method):
    """Fit one selected extreme-value distribution."""

    if method == "GEV":
        return fit_gev(values)
    if method == "Gumbel":
        return fit_gumbel(values)
    if method == "GenEx":
        return fit_genex(values)
    raise ValueError(f"Unsupported method: {method}")


def simulate_from_fitted_distribution(parameters, method, sample_size, rng):
    """Simulate one same-size sample from a fitted extreme-value distribution."""

    if method == "GEV":
        shape_c, location, scale = parameters
        return genextreme.rvs(
            shape_c, loc=location, scale=scale, size=sample_size, random_state=rng
        )

    if method == "Gumbel":
        location, scale = parameters
        return gumbel_r.rvs(loc=location, scale=scale, size=sample_size, random_state=rng)

    if method == "GenEx":
        shape, scale = parameters
        probabilities = rng.random(sample_size)
        return -scale * np.log1p(-np.power(probabilities, 1.0 / shape))

    raise ValueError(f"Unsupported method: {method}")


def make_bootstrap_sample(values, method, rng, fitted_parameters=None):
    """Generate one bootstrap sample using the configured bootstrap method."""

    if BOOTSTRAP_METHOD == "nonparametric":
        return rng.choice(values, size=values.size, replace=True)

    if fitted_parameters is None:
        raise ValueError("Parametric bootstrap requires fitted parameters.")

    return simulate_from_fitted_distribution(
        fitted_parameters,
        method,
        values.size,
        rng,
    )


def bootstrap_month_method(values, event_value, method, random_seed):
    """Bootstrap one monthly sample using the configured bootstrap method."""

    rng = np.random.default_rng(random_seed)
    probabilities = np.full(NUMBER_OF_BOOTSTRAPS, np.nan)
    fitted_parameters = fit_method(values, method) if BOOTSTRAP_METHOD == "parametric" else None

    for bootstrap_number in range(NUMBER_OF_BOOTSTRAPS):
        try:
            bootstrap_values = make_bootstrap_sample(
                values,
                method,
                rng,
                fitted_parameters=fitted_parameters,
            )
            probabilities[bootstrap_number] = calculate_exceedance_probability(
                bootstrap_values, event_value, method
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    return summarize_bootstrap_probabilities(probabilities, method)


def bootstrap_annual_method(monthly_values, monthly_event_values, method, random_seed):
    """Bootstrap 12 monthly samples and combine each replicate to an annual result."""

    rng = np.random.default_rng(random_seed)
    probabilities = np.full(NUMBER_OF_BOOTSTRAPS, np.nan)

    fitted_parameters = None
    if BOOTSTRAP_METHOD == "parametric":
        fitted_parameters = {
            month: fit_method(monthly_values[month], method)
            for month in range(1, 13)
        }

    for bootstrap_number in range(NUMBER_OF_BOOTSTRAPS):
        monthly_probabilities = []
        fit_failed = False

        for month in range(1, 13):
            values = monthly_values[month]
            parameters = None if fitted_parameters is None else fitted_parameters[month]

            try:
                bootstrap_values = make_bootstrap_sample(
                    values,
                    method,
                    rng,
                    fitted_parameters=parameters,
                )
                probability = calculate_exceedance_probability(
                    bootstrap_values,
                    monthly_event_values[month],
                    method,
                )
            except (RuntimeError, ValueError, FloatingPointError):
                fit_failed = True
                break

            monthly_probabilities.append(probability)

        if not fit_failed:
            probabilities[bootstrap_number] = combine_monthly_probabilities(
                monthly_probabilities
            )

    return summarize_bootstrap_probabilities(probabilities, method)


def summarize_bootstrap_probabilities(probabilities, method):
    """Summarize successful bootstrap probabilities and return periods."""

    probabilities = probabilities[np.isfinite(probabilities)]
    minimum_successful = int(
        np.ceil(MIN_SUCCESSFUL_BOOTSTRAP_FRACTION * NUMBER_OF_BOOTSTRAPS)
    )
    if probabilities.size < minimum_successful:
        raise RuntimeError(
            f"Only {probabilities.size} of {NUMBER_OF_BOOTSTRAPS} {method} "
            "bootstrap fits succeeded."
        )

    return_periods = np.divide(
        1.0,
        probabilities,
        out=np.full(probabilities.shape, np.inf),
        where=probabilities > 0,
    )

    return {
        "bootstrap_probabilities": probabilities,
        "bootstrap_return_periods": return_periods,
        "successful_bootstraps": probabilities.size,
    }


def add_bootstrap_summary(analysis, bootstrap):
    """Attach bootstrap samples and percentile summaries to one fitted result."""

    probabilities = bootstrap["bootstrap_probabilities"]
    return_periods = bootstrap["bootstrap_return_periods"]

    alpha = 1.0 - CONFIDENCE_LEVEL
    rp_percentiles = [
        100.0 * alpha / 2.0,
        25.0,
        50.0,
        75.0,
        100.0 * (1.0 - alpha / 2.0),
    ]
    probability_percentiles = [100.0 - percentile for percentile in rp_percentiles]
    probability_quantiles = np.percentile(probabilities, probability_percentiles)
    rp_low, rp_q1, rp_median, rp_q3, rp_high = [
        return_period_from_probability(probability) for probability in probability_quantiles
    ]

    analysis.update(
        {
            **bootstrap,
            "return_period_low": rp_low,
            "return_period_q1": rp_q1,
            "return_period_median": rp_median,
            "return_period_q3": rp_q3,
            "return_period_high": rp_high,
        }
    )

    if PLOT_METRIC == "aep":
        metric_samples = 100.0 * np.array(
            [calculate_horizon_aep(p, N_AEP_YEARS) for p in probabilities]
        )
        low, q1, median, q3, high = np.percentile(metric_samples, rp_percentiles)
    else:
        low, q1, median, q3, high = rp_low, rp_q1, rp_median, rp_q3, rp_high

    analysis.update(
        {
            "metric_low": low,
            "metric_q1": q1,
            "metric_median": median,
            "metric_q3": q3,
            "metric_high": high,
        }
    )
    return analysis


def get_metric_value(analysis):
    """Return the original-sample plotting metric within configured limits."""

    if PLOT_METRIC == "aep":
        value = 100.0 * calculate_horizon_aep(
            analysis["annual_exceedance_probability"], N_AEP_YEARS
        )
        return float(np.clip(value, AEP_YMIN_PERCENT, AEP_YMAX_PERCENT))

    value = analysis["return_period"]
    if not np.isfinite(value):
        value = RETURN_PERIOD_YMAX_YEARS
    return float(np.clip(value, RETURN_PERIOD_YMIN_YEARS, RETURN_PERIOD_YMAX_YEARS))


def get_interval_values(analysis):
    """Return the clipped 95% interval and bootstrap median for plotting."""

    if PLOT_METRIC == "aep":
        lower_limit, upper_limit = AEP_YMIN_PERCENT, AEP_YMAX_PERCENT
    else:
        lower_limit, upper_limit = RETURN_PERIOD_YMIN_YEARS, RETURN_PERIOD_YMAX_YEARS

    return (
        float(np.clip(analysis["metric_low"], lower_limit, upper_limit)),
        float(np.clip(analysis["metric_median"], lower_limit, upper_limit)),
        float(np.clip(analysis["metric_high"], lower_limit, upper_limit)),
    )


# =============================================================================
# Panel calculations
# =============================================================================

def get_analysis_seed(panel_index, group_index, method_index):
    """Return a deterministic, distinct random seed for one bootstrap analysis."""

    return RANDOM_SEED + 10_000 * panel_index + 100 * group_index + method_index


def calculate_month_panel(panel_index, panel_label, month, threshold_type, record_end_year):
    """Calculate and bootstrap one single-month panel."""

    reference = read_reference_month(month, record_end_year)
    model_values = read_model_month(month)

    if threshold_type == "storm_hans":
        event_value = reference["storm_hans_value"]
        threshold_label = "Storm Hans 2023"
    else:
        event_value = reference["record_value"]
        threshold_label = f"calendar record {RECORD_START_YEAR}-{record_end_year}"

    samples = {"reference": reference["fit_values"], "model": model_values}
    results = {}

    for group_index, (group, values) in enumerate(samples.items()):
        for method_index, method in enumerate(METHODS):
            analysis = analyse_method(values, event_value, method)
            bootstrap = bootstrap_month_method(
                values,
                event_value,
                method,
                get_analysis_seed(panel_index, group_index, method_index),
            )
            results[(group, method)] = add_bootstrap_summary(analysis, bootstrap)

    return {
        "title": f"{panel_label}) {get_panel_month_label(month)} {threshold_label}",
        "month": month,
        "threshold_type": threshold_type,
        "event_value": event_value,
        "record_year": reference["record_year"],
        "results": results,
    }


def calculate_annual_panel(panel_index, panel_label, threshold_type, record_end_year):
    """Calculate and bootstrap an annual any-month panel from 12 monthly samples."""

    monthly_samples = {group: {} for group in PLOT_GROUPS}
    monthly_event_values = {}

    for month in range(1, 13):
        reference = read_reference_month(month, record_end_year)
        monthly_samples["reference"][month] = reference["fit_values"]
        monthly_samples["model"][month] = read_model_month(month)
        monthly_event_values[month] = (
            reference["storm_hans_value"]
            if threshold_type == "storm_hans"
            else reference["record_value"]
        )

    results = {}
    for group_index, group in enumerate(PLOT_GROUPS):
        for method_index, method in enumerate(METHODS):
            monthly_probabilities = [
                calculate_exceedance_probability(
                    monthly_samples[group][month], monthly_event_values[month], method
                )
                for month in range(1, 13)
            ]
            probability = combine_monthly_probabilities(monthly_probabilities)
            analysis = {
                "annual_exceedance_probability": probability,
                "return_period": return_period_from_probability(probability),
            }
            bootstrap = bootstrap_annual_method(
                monthly_samples[group],
                monthly_event_values,
                method,
                get_analysis_seed(panel_index, group_index, method_index),
            )
            results[(group, method)] = add_bootstrap_summary(analysis, bootstrap)

    threshold_label = (
        "Storm Hans 2023"
        if threshold_type == "storm_hans"
        else f"calendar record {RECORD_START_YEAR}-{record_end_year}"
    )

    return {
        "title": f"{panel_label}) Annual {threshold_label}",
        "month": 0,
        "threshold_type": threshold_type,
        "event_value": None,
        "monthly_event_values": monthly_event_values,
        "record_year": None,
        "results": results,
    }


def calculate_panel(panel_index, panel_label, month, threshold_type, record_end_year):
    """Calculate a single-month or annual panel."""

    if month == 0:
        return calculate_annual_panel(
            panel_index, panel_label, threshold_type, record_end_year
        )
    return calculate_month_panel(
        panel_index, panel_label, month, threshold_type, record_end_year
    )


def calculate_all_panels():
    """Calculate the four panels in reading order."""

    panel_specs = [
        ("a", AUGUST_MONTH, "storm_hans", AUGUST_RECORD_END_YEAR),
        ("b", AUGUST_MONTH, "calendar_record", AUGUST_RECORD_END_YEAR),
        ("c", MAY_MONTH, "storm_hans", MAY_RECORD_END_YEAR),
        ("d", MAY_MONTH, "calendar_record", MAY_RECORD_END_YEAR),
    ]
    return [calculate_panel(index, *spec) for index, spec in enumerate(panel_specs)]


# =============================================================================
# Reporting and plotting
# =============================================================================

def print_panel_results(panel_outputs):
    """Print fitted estimates and 95% bootstrap return-period ranges."""

    print(f"\nReference dataset: {get_reference_name()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")
    print(f"Bootstrap method:  {BOOTSTRAP_METHOD}")
    print(f"Bootstrap samples: {NUMBER_OF_BOOTSTRAPS}")
    print(f"Confidence level:  {100.0 * CONFIDENCE_LEVEL:g}%")

    for panel in panel_outputs:
        print(f"\n{panel['title']}")
        if panel["month"] != 0:
            print(f"Event value: {panel['event_value']:.4f} mm")

        for group in PLOT_GROUPS:
            group_label = get_reference_name() if group == "reference" else get_model_label()
            for method in METHODS:
                analysis = panel["results"][(group, method)]
                return_period = analysis["return_period"]
                rp_text = f"{return_period:.4g}" if np.isfinite(return_period) else "inf"
                print(
                    f"  {group_label:>14} | {method:>7} | RP={rp_text} y | "
                    f"95% bootstrap={analysis['return_period_low']:.4g}-"
                    f"{analysis['return_period_high']:.4g} y | "
                    f"n={analysis['successful_bootstraps']}"
                )


def metric_tick_formatter(value, _position):
    """Format logarithmic metric-axis ticks."""

    if value <= 0:
        return ""

    label = f"{value:g}"
    if PLOT_METRIC == "aep" and np.isclose(value, AEP_YMIN_PERCENT):
        return f"<{label}"
    if PLOT_METRIC == "return_period" and np.isclose(value, RETURN_PERIOD_YMAX_YEARS):
        return f">{label}"
    return label


def get_y_axis_label():
    """Return the y-axis label."""

    if PLOT_METRIC == "aep":
        return f"{N_AEP_YEARS}-year exceedance probability [%]"
    return "Return period [years]"


def configure_axis(axis):
    """Apply common axis formatting."""

    axis.set_yscale("log")
    if PLOT_METRIC == "aep":
        axis.set_ylim(0.7 * AEP_YMIN_PERCENT, 1.3 * AEP_YMAX_PERCENT)
    else:
        axis.set_ylim(0.7 * RETURN_PERIOD_YMIN_YEARS, 1.3 * RETURN_PERIOD_YMAX_YEARS)

    axis.yaxis.set_major_formatter(FuncFormatter(metric_tick_formatter))
    axis.set_xlim(-0.55, 1.55)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([get_reference_name(), get_model_label()], fontsize=TICK_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.7, alpha=0.45, zorder=0)


def plot_one_panel(axis, panel_output):
    """Plot side-by-side 95% bootstrap intervals and medians for one panel."""

    for group_index, group in enumerate(PLOT_GROUPS):
        for method in METHODS:
            analysis = panel_output["results"][(group, method)]
            position = group_index + METHOD_OFFSETS[method]
            low, median, high = get_interval_values(analysis)
            color = METHOD_COLORS[method]

            axis.vlines(
                position,
                low,
                high,
                color=color,
                linewidth=INTERVAL_LINEWIDTH,
                zorder=3,
            )
            axis.hlines(
                [low, high],
                position - INTERVAL_CAP_WIDTH / 2,
                position + INTERVAL_CAP_WIDTH / 2,
                color=color,
                linewidth=INTERVAL_LINEWIDTH,
                zorder=3,
            )
            axis.plot(
                position,
                median,
                marker="o",
                markersize=MEDIAN_MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=MEDIAN_MARKER_EDGE_WIDTH,
                linestyle="none",
                zorder=4,
            )

    configure_axis(axis)
    axis.set_title(
        panel_output["title"],
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        loc="left",
        pad=8,
    )


def make_figure(panel_outputs):
    """Create the 2 x 2 publication figure."""

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
        2,
        2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharey=SHARE_Y_AXES,
        constrained_layout=True,
    )

    for axis, panel_output in zip(axes.ravel(), panel_outputs):
        plot_one_panel(axis, panel_output)

    for row in range(2):
        axes[row, 0].set_ylabel(get_y_axis_label(), fontsize=AXIS_LABELSIZE)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=METHOD_COLORS[method],
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=METHOD_COLORS[method],
            linewidth=INTERVAL_LINEWIDTH,
            markersize=MEDIAN_MARKER_SIZE + 1,
            label=method,
        )
        for method in METHODS
    ]
    axes[0, 0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="best",
        handletextpad=0.5,
        borderaxespad=0.4)

    filename = make_figure_filename()
    if WRITE_TO_FILE:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()

    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate_user_settings()

    print("Selected settings")
    print("-----------------")
    print(f"Reference dataset: {get_reference_name()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")
    print(f"August row month:  {AUGUST_MONTH}")
    print(f"May row month:     {MAY_MONTH}")
    print(f"Model file:        {make_model_filename()}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Bootstrap method:  {BOOTSTRAP_METHOD}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS}")
    print(f"Confidence level:  {100.0 * CONFIDENCE_LEVEL:g}%")

    panels = calculate_all_panels()
    print_panel_results(panels)
    make_figure(panels)
