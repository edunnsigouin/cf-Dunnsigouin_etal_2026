"""
Return period of the Storm Hans threshold for the reference dataset and five
variants of the model dataset, in August and May.

Layout
------
    a) August return period for the Storm Hans threshold
    b) May return period for the Storm Hans threshold

Each panel has three ticks, GEV, Gumbel and GenEx, and at every tick six series, from left to right:

    reference (seNorge or ERA5)        fitted to OBSERVATION_YEARS
    model raw                          all raw model samples for the month, pooled
    model raw, first period            raw samples from RAW_PERIOD_YEARS["raw_first_half"]
    model raw, second period           raw samples from RAW_PERIOD_YEARS["raw_second_half"]
    model bias corrected               all BIAS_CORRECTION_METHOD samples, pooled
    model raw, equal-year weighted     error bars only (no dot)

The Storm Hans threshold (August 2023) is taken from the reference dataset and
applied identically to all six series. INCLUDE_STORM_HANS_IN_FIT controls
whether August 2023 is part of the August reference fit.

Fitted estimates and uncertainty
--------------------------------
For GEV, Gumbel and GenEx, a dot shows the fitted return period and a capped
vertical line the central CONFIDENCE_LEVEL interval of NUMBER_OF_BOOTSTRAPS
refits. The reference, raw, raw-period and bias-corrected series use a
nonparametric bootstrap: each replicate draws, with replacement, as many values
as the sample holds. The two raw-period series are the raw model sample
restricted to the sample years (true valid year) in RAW_PERIOD_YEARS, pooled and
bootstrapped exactly like the full raw sample. The equal-year-weighted series is an alternative uncertainty
estimate for the raw model, so it is drawn as an error bar only, without a dot
(its central estimate is the raw-model dot next to it). Its bootstrap keeps the
raw values separate by year and uses only the years in EQUAL_YEAR_YEARS
(inclusive); each replicate draws
EQUAL_YEAR_SAMPLES_PER_YEAR values with replacement from every year and pools
them, so every year carries equal weight in the uncertainty however many model
realisations it has. Infinite bootstrap return periods are retained in the
percentiles.
Intervals beyond the plotting limits end in outward arrows; off-scale fitted
values are drawn as triangles at the limit. The empirical return period of the
threshold can be read from the Weibull points in fig-05 (a)-(b).

Bootstrap seeds follow the companion scripts: the reference, raw and
bias-corrected series reproduce fig-05 (run with the matching MODEL_DATA_METHOD),
and the equal-year-weighted bootstrap replicates are drawn as in the equal-year
bootstrap script. The raw-period series use their own seed groups.

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
BIAS_CORRECTION_METHOD = "mm_1step"  # "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"

INCLUDE_STORM_HANS_IN_FIT = True  # Affects only the August reference fit.
EQUAL_YEAR_SAMPLES_PER_YEAR = 400  # Draws with replacement from every year per replicate.
EQUAL_YEAR_YEARS = [2004, 2023]  # Years used by the equal-year bootstrap only (inclusive).
RAW_PERIOD_YEARS = {  # Sample years of the two raw-period series (inclusive).
    "raw_first_half": [2000, 2011],
    "raw_second_half": [2012, 2023],
}

# Bootstrap uncertainty
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIG_WIDTH_IN = 13
FIG_HEIGHT_IN = 5.2
SHOW_GRID = True
RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e8
SERIES_COLORS = {
    "reference": "tab:orange",
    "raw": "tab:blue",
    "raw_first_half": "tab:green",
    "raw_second_half": "tab:purple",
    "bias_corrected": "tab:grey",
    "raw_equal_year": "tab:pink",
}
LEGEND_PANEL = "a"  # Panel that carries the legend: "a" (August) or "b" (May).
LEGEND_LOCATION = "upper right"

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_FORMAT = "pdf"  # "png" or "pdf".
FIGURE_DPI = 300  # Resolution of PNG output.
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = f"fig-S2.{FIGURE_FORMAT}"
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
SERIES = ["reference", "raw", "raw_first_half", "raw_second_half",
          "bias_corrected", "raw_equal_year"]
RAW_PERIOD_SERIES = ["raw_first_half", "raw_second_half"]
INTERVAL_ONLY_SERIES = {"raw_equal_year"}  # Drawn as error bars without a dot.
# Seed group per series; 0/1 match fig-05 (reference/model), 1 matches the
# equal-year group of the equal-year bootstrap script; 2/3 are the raw periods.
SERIES_SEED_GROUP = {
    "reference": 0, "raw": 1, "raw_first_half": 2, "raw_second_half": 3,
    "bias_corrected": 1, "raw_equal_year": 1,
}

# =============================================================================
# Plot constants
# =============================================================================
SERIES_SPACING = 0.12  # Separation between neighbouring series at each tick.
INTERVAL_LINEWIDTH = 1.4
INTERVAL_CAP_WIDTH = 0.06
CLIPPED_ARROW_LENGTH = 14  # Points; fixed visual length on logarithmic axes.
CLIPPED_ARROW_HEAD_SIZE = 10
FITTED_MARKER_SIZE = 6
AXIS_LABELSIZE = 12
TICK_LABELSIZE = 12
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 11

# =============================================================================
# Validation, labels, and filenames
# =============================================================================


def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if BIAS_CORRECTION_METHOD not in {"mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:
        raise ValueError("Unsupported BIAS_CORRECTION_METHOD.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if not REFERENCE_FILE_YEARS[0] <= OBSERVATION_YEARS[0] <= OBSERVATION_YEARS[1] <= REFERENCE_FILE_YEARS[1]:
        raise ValueError(
            f"OBSERVATION_YEARS must lie within {REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}."
        )
    if EQUAL_YEAR_SAMPLES_PER_YEAR < 1:
        raise ValueError("EQUAL_YEAR_SAMPLES_PER_YEAR must be at least 1.")
    if EQUAL_YEAR_YEARS[0] > EQUAL_YEAR_YEARS[1]:
        raise ValueError("EQUAL_YEAR_YEARS must be increasing.")
    if set(RAW_PERIOD_YEARS) != set(RAW_PERIOD_SERIES):
        raise ValueError(f"RAW_PERIOD_YEARS must contain exactly {RAW_PERIOD_SERIES}.")
    for series, (first_year, last_year) in RAW_PERIOD_YEARS.items():
        if first_year > last_year:
            raise ValueError(f"RAW_PERIOD_YEARS['{series}'] must be increasing.")
    if LEGEND_PANEL not in {"a", "b"}:
        raise ValueError("LEGEND_PANEL must be 'a' or 'b'.")
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
    if set(SERIES_COLORS) != set(SERIES):
        raise ValueError(f"SERIES_COLORS must contain exactly {SERIES}.")
    if FIGURE_FORMAT not in {"png", "pdf"}:
        raise ValueError("FIGURE_FORMAT must be 'png' or 'pdf'.")


def get_reference_name():
    """Return the display name of the selected reference dataset."""
    return {"senorge": "seNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the variable name in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def series_label(series):
    """Return the legend label of one series."""
    return {
        "reference": get_reference_name(),
        "raw": "Model raw",
        "raw_first_half": (f"Model raw {RAW_PERIOD_YEARS['raw_first_half'][0]}"
                           f"\u2013{RAW_PERIOD_YEARS['raw_first_half'][1]}"),
        "raw_second_half": (f"Model raw {RAW_PERIOD_YEARS['raw_second_half'][0]}"
                            f"\u2013{RAW_PERIOD_YEARS['raw_second_half'][1]}"),
        "bias_corrected": f"Model bias corrected",
        "raw_equal_year": "Model raw, equal-year weighted bootstrap",
    }[series]


def series_offsets():
    """Return the x offset of each series around a tick, left to right."""
    centre = (len(SERIES) - 1) / 2
    return {series: (index - centre) * SERIES_SPACING for index, series in enumerate(SERIES)}


def make_reference_filename():
    """Construct the reference monthly-maximum filename."""
    first_year, last_year = REFERENCE_FILE_YEARS
    filename = (
        f"monthly_max_samples_{get_reference_variable()}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_{first_year}-{last_year}.nc"
    )
    directory = "senorge_processed" if REFERENCE_DATASET == "senorge" else "era5_processed"
    return Path(config.dirs[directory]) / filename


def make_model_filename(data_method):
    """Construct the compact model filename for "raw" or a bias-correction method."""
    stem = (
        f"monthly_max_samples_{MODEL_FILE_VARIABLE}_{X_DAYS}dayacc_"
        f"{CATCHMENT.removeprefix('regine_')}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    if data_method == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{data_method}_{REFERENCE_DATASET}_"
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


def read_model_month(data_method, month):
    """Read one calendar-month model sample, pooled and split by year."""
    filename = make_model_filename(data_method)
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
        sample_month = np.asarray(ds["sample_month"].values, dtype="int64")
        values = np.asarray(ds[variable].transpose("number", "i_date").values, dtype=float)
    in_month = sample_month % 100 == month
    pooled = values[:, in_month].ravel()
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 10:
        raise ValueError(
            f"Fewer than 10 finite '{data_method}' model values were found for month {month}."
        )
    values_by_year = {}
    for yyyymm in np.unique(sample_month[in_month]):
        year_values = values[:, sample_month == yyyymm].ravel()
        year_values = year_values[np.isfinite(year_values)]
        if year_values.size:
            values_by_year[int(yyyymm) // 100] = year_values
    return {"pooled": pooled, "by_year": values_by_year}


def select_equal_year_years(values_by_year, month):
    """Keep only the years in EQUAL_YEAR_YEARS for the equal-year bootstrap."""
    first_year, last_year = EQUAL_YEAR_YEARS
    selected = {
        year: values for year, values in values_by_year.items()
        if first_year <= year <= last_year
    }
    if not selected:
        raise ValueError(
            f"No {MONTH_NAMES[month - 1]} raw model values fall in EQUAL_YEAR_YEARS "
            f"{first_year}-{last_year}; available years: "
            f"{min(values_by_year)}-{max(values_by_year)}."
        )
    return selected


def select_raw_period(values_by_year, series, month):
    """Pool the raw values whose sample year lies in RAW_PERIOD_YEARS[series]."""
    first_year, last_year = RAW_PERIOD_YEARS[series]
    years = sorted(year for year in values_by_year if first_year <= year <= last_year)
    if not years:
        raise ValueError(
            f"No {MONTH_NAMES[month - 1]} raw model values fall in RAW_PERIOD_YEARS"
            f"['{series}'] {first_year}-{last_year}; available years: "
            f"{min(values_by_year)}-{max(values_by_year)}."
        )
    pooled = np.concatenate([values_by_year[year] for year in years])
    if pooled.size < 10:
        raise ValueError(
            f"Fewer than 10 {MONTH_NAMES[month - 1]} raw model values fall in "
            f"RAW_PERIOD_YEARS['{series}'] {first_year}-{last_year}."
        )
    return {"pooled": pooled, "years": years}


def make_equal_year_sample(values_by_year, rng):
    """Draw EQUAL_YEAR_SAMPLES_PER_YEAR values with replacement from every year."""
    return np.concatenate([
        rng.choice(values_by_year[year], size=EQUAL_YEAR_SAMPLES_PER_YEAR, replace=True)
        for year in sorted(values_by_year)
    ])

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


def fit_with_bootstrap(central_values, sample_factory, method, random_seed, progress=None):
    """Fit the central sample and NUMBER_OF_BOOTSTRAPS replicates from sample_factory."""
    parameters = fit_distribution(central_values, method)
    rng = np.random.default_rng(random_seed)
    bootstrap_parameters = []
    for _ in range(NUMBER_OF_BOOTSTRAPS):
        try:
            fitted = fit_distribution(
                sample_factory(rng),
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


def resample_with_replacement(values):
    """Return a nonparametric bootstrap sample factory for one sample."""
    return lambda rng: rng.choice(values, size=values.size, replace=True)


def build_month_analysis(month, month_index, progress=None):
    """Read one month and fit every series and distribution with its bootstrap."""
    reference = read_reference_month(month)
    raw = read_model_month("raw", month)
    equal_year_values = select_equal_year_years(raw["by_year"], month)
    bias_corrected = read_model_month(BIAS_CORRECTION_METHOD, month)
    raw_periods = {
        series: select_raw_period(raw["by_year"], series, month)
        for series in RAW_PERIOD_SERIES
    }
    # The equal-year series is fitted to the pooled raw sample (same dots as raw);
    # only its bootstrap replicates are drawn year by year.
    samples = {
        "reference": reference["fit_values"],
        "raw": raw["pooled"],
        **{series: raw_periods[series]["pooled"] for series in RAW_PERIOD_SERIES},
        "bias_corrected": bias_corrected["pooled"],
        "raw_equal_year": raw["pooled"],
    }
    factories = {
        "reference": resample_with_replacement(samples["reference"]),
        "raw": resample_with_replacement(samples["raw"]),
        **{series: resample_with_replacement(samples[series]) for series in RAW_PERIOD_SERIES},
        "bias_corrected": resample_with_replacement(samples["bias_corrected"]),
        "raw_equal_year": lambda rng: make_equal_year_sample(equal_year_values, rng),
    }
    fits = {}
    for series in SERIES:
        for method_index, method in enumerate(METHODS):
            seed = (RANDOM_SEED + 10000 * month_index
                    + 1000 * SERIES_SEED_GROUP[series] + method_index)
            fits[(series, method)] = fit_with_bootstrap(
                samples[series], factories[series], method, seed, progress
            )
    return {
        "month": month,
        "storm_hans_value": reference["storm_hans_value"],
        "samples": samples,
        "raw_by_year": raw["by_year"],
        "equal_year_by_year": equal_year_values,
        "raw_period_years": {series: raw_periods[series]["years"] for series in RAW_PERIOD_SERIES},
        "fits": fits,
    }


def analyse_event_return_period(fit, event_value, method):
    """Evaluate the Storm Hans threshold with the fitted and bootstrap parameters."""
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


def calculate_panel(month_analysis):
    """Calculate fitted and bootstrap results for one month."""
    event_value = month_analysis["storm_hans_value"]
    results = {
        (series, method): analyse_event_return_period(
            month_analysis["fits"][(series, method)], event_value, method
        )
        for series in SERIES
        for method in METHODS
    }
    return {
        "month": month_analysis["month"],
        "event_value": event_value,
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


def configure_axis(axis):
    """Apply common formatting to one return-period panel."""
    axis.set_yscale("log")
    axis.set_ylim(RETURN_PERIOD_MIN, 1.15 * RETURN_PERIOD_MAX)
    axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))
    axis.set_xlim(-0.5, len(METHODS) - 0.5)
    axis.set_xticks(range(len(METHODS)))
    axis.set_xticklabels(METHODS)
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


def interval_bounds(metric_samples):
    """Return the central CONFIDENCE_LEVEL bootstrap interval."""
    tail_percent = 100 * (1 - CONFIDENCE_LEVEL) / 2
    return (
        percentile_preserving_infinity(metric_samples, tail_percent),
        percentile_preserving_infinity(metric_samples, 100 - tail_percent),
    )


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


def plot_panel(axis, panel_label, panel_output):
    """Plot fitted and bootstrap estimates of every series for one month."""
    offsets = series_offsets()
    for method_index, method in enumerate(METHODS):
        for series in SERIES:
            analysis = panel_output["results"][(series, method)]
            lower, upper = interval_bounds(analysis["metric_samples"])
            # The percentile interval need not contain the fitted estimate.
            fitted = np.nan if series in INTERVAL_ONLY_SERIES else analysis["return_period"]
            plot_metric_interval(
                axis, method_index + offsets[series], lower, upper,
                fitted, SERIES_COLORS[series],
            )
    configure_axis(axis)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[panel_output['month'] - 1]}: Storm Hans threshold",
        loc="center", fontsize=TITLE_FONTSIZE, fontweight="normal",
    )


def make_figure(panels):
    """Create the 1 x 2 figure (August left, May right) with the legend in one panel."""
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
        1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), constrained_layout=True
    )
    plot_panel(axes[0], "a", panels[AUGUST])
    plot_panel(axes[1], "b", panels[MAY])
    legend_axis = axes[0] if LEGEND_PANEL == "a" else axes[1]
    handles = [
        Line2D([0], [0], linestyle="-", color=SERIES_COLORS[series],
               marker=None if series in INTERVAL_ONLY_SERIES else "o",
               linewidth=INTERVAL_LINEWIDTH, markersize=FITTED_MARKER_SIZE,
               label=series_label(series))
        for series in SERIES
    ]
    legend = legend_axis.legend(
        handles=handles, loc=LEGEND_LOCATION, frameon=True, framealpha=0.9,
        edgecolor="none", fontsize=LEGEND_FONTSIZE,
    )
    legend.set_zorder(10)
    if WRITE_TO_FILE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PATH, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
                       bbox_inches="tight", facecolor="white")
        print("Wrote:", OUTPUT_PATH)
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Reporting
# =============================================================================


def print_summary(month_analyses, panels):
    """Print settings, sample sizes, and fitted return periods."""
    print("Selected settings")
    print("-----------------")
    print(f"Reference:         {get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Raw model file:    {make_model_filename('raw')}")
    print(f"BC model file:     {make_model_filename(BIAS_CORRECTION_METHOD)}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS} (nonparametric)")
    print(f"Draws per year:    {EQUAL_YEAR_SAMPLES_PER_YEAR} (equal-year weighted, "
          f"years {EQUAL_YEAR_YEARS[0]}-{EQUAL_YEAR_YEARS[1]})")
    print(f"Include Hans fit:  {INCLUDE_STORM_HANS_IN_FIT}")
    for month in PANEL_MONTHS:
        analysis = month_analyses[month]
        panel = panels[month]
        all_years = analysis["raw_by_year"]
        by_year = analysis["equal_year_by_year"]
        counts = [values.size for values in by_year.values()]
        print(f"\n{MONTH_NAMES[month - 1]}: Storm Hans threshold {panel['event_value']:.3f} mm")
        print(f"  Sample sizes: {get_reference_name()} {analysis['samples']['reference'].size}, "
              f"raw {analysis['samples']['raw'].size}, "
              f"bias corrected {analysis['samples']['bias_corrected'].size}")
        for series in RAW_PERIOD_SERIES:
            years = analysis["raw_period_years"][series]
            print(f"  {series_label(series)}: {analysis['samples'][series].size} values "
                  f"from {len(years)} years ({years[0]}-{years[-1]} present)")
        print("  Raw values per year (* = used by the equal-year bootstrap):")
        print("    " + ", ".join(
            f"{year}{'*' if year in by_year else ''}: {values.size}"
            for year, values in sorted(all_years.items())
        ))
        print(f"  Equal-year bootstrap: {len(by_year)} years {min(by_year)}-{max(by_year)}, "
              f"{min(counts)}-{max(counts)} raw values per year, "
              f"{EQUAL_YEAR_SAMPLES_PER_YEAR * len(by_year)} values per replicate")
        print(f"  {'Distribution':<13} {'Series':<40} {'Fitted':>12} {'Lower':>12} {'Upper':>12}")
        for method in METHODS:
            for series in SERIES:
                result = panel["results"][(series, method)]
                lower, upper = interval_bounds(result["metric_samples"])
                fitted = (f"{'-':>12}" if series in INTERVAL_ONLY_SERIES
                          else f"{result['return_period']:12.5g}")
                print(f"  {method:<13} {series_label(series):<40} "
                      f"{fitted} {lower:12.5g} {upper:12.5g}")
        print("  (Equal-year weighted: interval only; its central estimate is the Model raw value.)")
    print("\ninf = unbounded return period.")

# =============================================================================
# Main
# =============================================================================


def main():
    """Fit every series for August and May and plot the Storm Hans return periods."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)
    progress = ProgressTracker(len(PANEL_MONTHS) * len(SERIES) * len(METHODS) * NUMBER_OF_BOOTSTRAPS)
    print("Running bootstrap fits...")
    print("Progress:   0%", end="\r", flush=True)
    month_analyses = {
        month: build_month_analysis(month, index, progress)
        for index, month in enumerate(PANEL_MONTHS)
    }
    panels = {month: calculate_panel(month_analyses[month]) for month in PANEL_MONTHS}
    print_summary(month_analyses, panels)
    make_figure(panels)


if __name__ == "__main__":
    main()
