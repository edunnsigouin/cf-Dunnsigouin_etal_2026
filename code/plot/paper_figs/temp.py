"""
Return period of the calendar-month record threshold in every calendar month.

Layout
------
Twelve panels, one per calendar month (a) January to (l) December, arranged in
4 rows x 3 columns. Each panel has four ticks, GEV, Gumbel, GenEx and
"Empirical", and at every tick three series, from left to right:

    reference (seNorge or ERA5)   fitted to OBSERVATION_YEARS for that month
    model raw                     all raw model samples for the month, pooled
    model bias corrected          all BIAS_CORRECTION_METHOD samples, pooled

Thresholds (as the "monthly_record" threshold of fig-06)
---------------------------------------------------------
Each panel uses its own calendar-month record: the highest reference value of
that calendar month in OBSERVATION_YEARS, including 2023. The August record is
therefore Storm Hans itself (August 2023); every other month uses its own
highest seNorge (or ERA5) value. The thresholds are always taken from the
reference dataset and applied identically to all three series in a panel; the
printed summary lists each month's record value and year.
INCLUDE_STORM_HANS_IN_FIT controls whether August 2023 is part of the August
reference fit (no effect on other months); the August record always includes it.

Fitted estimates and uncertainty
--------------------------------
For GEV, Gumbel and GenEx, a dot shows the fitted return period and a capped
vertical line the central CONFIDENCE_LEVEL interval of NUMBER_OF_BOOTSTRAPS
refits (BOOTSTRAP_METHOD "nonparametric" or "parametric"). Infinite bootstrap
return periods are retained in the percentiles. Intervals beyond the plotting
limits end in outward arrows; off-scale fitted values are drawn as triangles at
the limit.

Empirical estimate
------------------
Solid dots without uncertainty, using the Weibull plotting position as in
fig-05: if k of the n sample values are at least as large as the month's record
threshold, the return period is (n + 1) / k years. For the reference this is
(n + 1) years by construction (k = 1, the record itself). If k = 0 the dot is placed at
(n + 1) years, a lower bound; the printed summary flags this case.

Reproducibility
---------------
Bootstrap seeds follow fig-05/fig-S2 (August is month index 0, May index 1; the
other months follow in calendar order), so the August and May panels reproduce
the same bootstrap replicates as fig-05 (with the matching MODEL_DATA_METHOD) and
fig-S2; the August values are identical to those (August record = Storm Hans).

Output
------
FIGURE_FORMAT = "png" or "pdf"; fonts are embedded as TrueType. The default size
(7.0 x 8.6 in, about 178 x 218 mm) fits a full-page-width journal figure. A
percentage progress indicator is printed while the bootstrap fits run.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
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

# Bootstrap uncertainty
BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric"
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIG_WIDTH_IN = 7.0  # Full page width (about 178 mm).
FIG_HEIGHT_IN = 8.6
SHOW_GRID = True
RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e8
SERIES_COLORS = {
    "reference": "tab:orange",
    "raw": "tab:blue",
    "bias_corrected": "tab:grey",
}
LEGEND_LOCATION = "upper right"  # Position of the legend inside panel a) (January).

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_FORMAT = "pdf"  # "png" or "pdf".
FIGURE_DPI = 300  # Resolution of PNG output.
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = f"fig-S4.{FIGURE_FORMAT}"
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
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST
MONTHS = list(range(1, 13))  # Panel order: January to December.
# Seed order: August and May first, as in fig-05/fig-S2, then the other months.
SEED_MONTH_ORDER = [AUGUST, MAY] + [month for month in MONTHS if month not in (AUGUST, MAY)]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
METHODS = ["GEV", "Gumbel", "GenEx"]
EMPIRICAL_LABEL = "Empirical"
SERIES = ["reference", "raw", "bias_corrected"]
SERIES_SEED_GROUP = {"reference": 0, "raw": 1, "bias_corrected": 1}  # As in fig-05.

# =============================================================================
# Plot constants (sized for a 7-inch-wide figure)
# =============================================================================
N_ROWS = 4
N_COLUMNS = 3
SERIES_SPACING = 0.22  # Separation between neighbouring series at each tick.
INTERVAL_LINEWIDTH = 1.0
INTERVAL_CAP_WIDTH = 0.12
CLIPPED_ARROW_LENGTH = 8  # Points; fixed visual length on logarithmic axes.
CLIPPED_ARROW_HEAD_SIZE = 6
FITTED_MARKER_SIZE = 3.5
FONT_SIZE = 11
TITLE_FONTSIZE = 11
LEGEND_FONTSIZE = 10
Y_TICK_DECADES = [0, 2, 4, 6, 8]  # Labelled powers of ten on the return-period axis.

# =============================================================================
# Validation, labels, and filenames
# =============================================================================


def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if BIAS_CORRECTION_METHOD not in {"mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:
        raise ValueError("Unsupported BIAS_CORRECTION_METHOD.")
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
        "bias_corrected": "Model bias corrected",
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
# Data reading (each file is read once for all months)
# =============================================================================


def read_reference_months():
    """Read the reference fit sample and calendar-month record of every month."""
    filename = make_reference_filename()
    variable = get_reference_variable()
    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")
    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")
        storm_hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )
        selected = {
            month: ds[variable].sel(
                year=slice(OBSERVATION_YEARS[0], OBSERVATION_YEARS[1]), month=month
            ).load()
            for month in MONTHS
        }
    hans_in_observation_range = OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
    fit_values = {}
    records = {}
    for month, data in selected.items():
        years = np.asarray(data["year"].values)
        values = np.asarray(data.values, dtype=float)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 10:
            raise ValueError(f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain.")
        # Record over all finite years in OBSERVATION_YEARS, including August 2023 (Hans).
        record_index = int(np.argmax(np.where(finite, values, -np.inf)))
        records[month] = {"value": float(values[record_index]), "year": int(years[record_index])}
        keep = finite.copy()
        if month == AUGUST and hans_in_observation_range and not INCLUDE_STORM_HANS_IN_FIT:
            keep &= years != STORM_HANS_YEAR
        if np.count_nonzero(keep) < 10:
            raise ValueError(
                f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain in the fit."
            )
        fit_values[month] = values[keep]
    if hans_in_observation_range and not np.isclose(records[AUGUST]["value"], storm_hans_value):
        print(f"Note: the August record ({records[AUGUST]['value']:.3f} mm in "
              f"{records[AUGUST]['year']}) is not Storm Hans ({storm_hans_value:.3f} mm).")
    return {"fit_values": fit_values, "records": records, "storm_hans_value": storm_hans_value}


def read_model_months(data_method):
    """Read the pooled model sample of every calendar month from sample_month (YYYYMM)."""
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
        calendar_month = np.asarray(ds["sample_month"].values, dtype="int64") % 100
        values = np.asarray(ds[variable].transpose("number", "i_date").values, dtype=float)
    pooled = {}
    for month in MONTHS:
        month_values = values[:, calendar_month == month].ravel()
        month_values = month_values[np.isfinite(month_values)]
        if month_values.size < 10:
            raise ValueError(
                f"Fewer than 10 finite '{data_method}' model values were found for "
                f"{MONTH_NAMES[month - 1]}."
            )
        pooled[month] = month_values
    return pooled

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


def make_bootstrap_sample(values, method, rng, fitted_parameters):
    """Create one nonparametric or parametric bootstrap sample."""
    if BOOTSTRAP_METHOD == "nonparametric":
        return rng.choice(values, size=values.size, replace=True)
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
    """Fit one distribution and NUMBER_OF_BOOTSTRAPS bootstrap refits."""
    parameters = fit_distribution(values, method)
    rng = np.random.default_rng(random_seed)
    bootstrap_parameters = []
    for _ in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = make_bootstrap_sample(values, method, rng, parameters)
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
    """Evaluate one threshold with the fitted and bootstrap parameters."""
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
    """Return the Weibull empirical estimate (n + 1) / max(k, 1) for one sample."""
    values = np.asarray(values, dtype=float)
    exceedances = int(np.count_nonzero(values >= event_value))
    return {
        "return_period": (values.size + 1) / max(exceedances, 1),
        "exceedances": exceedances,
        "sample_size": values.size,
    }


def analyse_month(month, samples, event_value, progress=None, event_year=None):
    """Fit every series and distribution for one month and evaluate its threshold."""
    month_index = SEED_MONTH_ORDER.index(month)
    results = {}
    for series in SERIES:
        for method_index, method in enumerate(METHODS):
            seed = (RANDOM_SEED + 10000 * month_index
                    + 1000 * SERIES_SEED_GROUP[series] + method_index)
            fit = bootstrap_distribution(samples[series], method, seed, progress)
            results[(series, method)] = analyse_event_return_period(fit, event_value, method)
    empirical = {
        series: empirical_event_metric(samples[series], event_value) for series in SERIES
    }
    return {
        "month": month,
        "event_value": event_value,
        "event_year": event_year,
        "sample_sizes": {series: samples[series].size for series in SERIES},
        "results": results,
        "empirical": empirical,
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


def configure_axis(axis, show_x_labels, show_y_labels):
    """Apply common formatting to one monthly panel."""
    tick_labels = [*METHODS, EMPIRICAL_LABEL]
    axis.set_yscale("log")
    axis.set_ylim(RETURN_PERIOD_MIN, 1.15 * RETURN_PERIOD_MAX)
    y_ticks = [
        10.0 ** decade for decade in Y_TICK_DECADES
        if RETURN_PERIOD_MIN <= 10.0 ** decade <= RETURN_PERIOD_MAX
    ]
    axis.yaxis.set_major_locator(FixedLocator(y_ticks))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_formatter(FuncFormatter(format_metric_return_period))
    axis.set_xlim(-0.5, len(tick_labels) - 0.5)
    axis.set_xticks(range(len(tick_labels)))
    axis.set_xticklabels(tick_labels if show_x_labels else [])
    axis.tick_params(axis="both", labelsize=FONT_SIZE, length=2.5, pad=2)
    axis.tick_params(axis="y", labelleft=show_y_labels)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)


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
        color=color, linewidth=INTERVAL_LINEWIDTH, zorder=3,
    )
    visible_endpoints = [value for value in (lower, upper) if minimum <= value <= maximum]
    if visible_endpoints:
        axis.hlines(
            visible_endpoints,
            position - INTERVAL_CAP_WIDTH / 2, position + INTERVAL_CAP_WIDTH / 2,
            color=color, linewidth=INTERVAL_LINEWIDTH, zorder=3,
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
    marker = "^" if fitted > maximum else "v" if fitted < minimum else "o"
    axis.plot(
        position, clip_metric(fitted), marker=marker, linestyle="none", color=color,
        markersize=FITTED_MARKER_SIZE, markeredgewidth=0, clip_on=False, zorder=5,
    )


def plot_panel(axis, panel_label, panel_output, show_x_labels, show_y_labels):
    """Plot fitted, bootstrap and empirical estimates of every series for one month."""
    offsets = series_offsets()
    for method_index, method in enumerate(METHODS):
        for series in SERIES:
            analysis = panel_output["results"][(series, method)]
            lower, upper = interval_bounds(analysis["metric_samples"])
            # The percentile interval need not contain the fitted estimate.
            plot_metric_interval(
                axis, method_index + offsets[series], lower, upper,
                analysis["return_period"], SERIES_COLORS[series],
            )
    for series in SERIES:
        axis.plot(
            len(METHODS) + offsets[series],
            clip_metric(panel_output["empirical"][series]["return_period"]),
            marker="o", linestyle="none", color=SERIES_COLORS[series],
            markersize=FITTED_MARKER_SIZE, markeredgewidth=0, clip_on=False, zorder=5,
        )
    configure_axis(axis, show_x_labels, show_y_labels)
    axis.set_title(
        f"{panel_label}) {MONTH_NAMES[panel_output['month'] - 1]}",
        loc="left", fontsize=TITLE_FONTSIZE, fontweight="normal", pad=3,
    )


def make_figure(panels):
    """Create the 4 x 3 figure (January top left, December bottom right)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": FONT_SIZE,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        N_ROWS, N_COLUMNS, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharex=True, sharey=True, layout="constrained",
    )
    figure.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.03, hspace=0.04)
    for index, month in enumerate(MONTHS):
        row, column = divmod(index, N_COLUMNS)
        plot_panel(
            axes[row, column], chr(ord("a") + index), panels[month],
            show_x_labels=row == N_ROWS - 1, show_y_labels=column == 0,
        )
    figure.supylabel("Return period [years]", fontsize=FONT_SIZE + 1)
    figure.supxlabel("Estimation method", fontsize=FONT_SIZE + 1)
    handles = [
        Line2D([0], [0], linestyle="-", color=SERIES_COLORS[series], marker="o",
               markeredgewidth=0, linewidth=INTERVAL_LINEWIDTH,
               markersize=FITTED_MARKER_SIZE + 0.5, label=series_label(series))
        for series in SERIES
    ]
    legend = axes[0, 0].legend(
        handles=handles, loc=LEGEND_LOCATION, frameon=True, framealpha=0.9,
        edgecolor="none", fontsize=LEGEND_FONTSIZE, handlelength=1.6,
        borderaxespad=0.3, borderpad=0.3, labelspacing=0.3,
    )
    legend.set_zorder(10)
    if WRITE_TO_FILE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PATH, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
                       bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print("Wrote:", OUTPUT_PATH)
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Reporting
# =============================================================================


def print_summary(panels):
    """Print settings and, per month, sample sizes and fitted/empirical return periods."""
    print("Selected settings")
    print("-----------------")
    print(f"Reference:         {get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}")
    print(f"Reference file:    {make_reference_filename()}")
    print(f"Raw model file:    {make_model_filename('raw')}")
    print(f"BC model file:     {make_model_filename(BIAS_CORRECTION_METHOD)}")
    print(f"Bootstraps:        {NUMBER_OF_BOOTSTRAPS} ({BOOTSTRAP_METHOD})")
    print(f"Include Hans fit:  {INCLUDE_STORM_HANS_IN_FIT}")
    print("Thresholds:        calendar-month records (August = Storm Hans)")
    for month in MONTHS:
        panel = panels[month]
        sizes = panel["sample_sizes"]
        print(f"\n{MONTH_NAMES[month - 1]}: record threshold {panel['event_value']:.3f} mm "
              f"({panel['event_year']})")
        print("  Sample sizes: "
              + ", ".join(f"{series_label(series)} {sizes[series]}" for series in SERIES))
        print(f"  {'Distribution':<13} {'Series':<32} {'Fitted':>12} {'Lower':>12} {'Upper':>12}")
        for method in METHODS:
            for series in SERIES:
                result = panel["results"][(series, method)]
                lower, upper = interval_bounds(result["metric_samples"])
                print(f"  {method:<13} {series_label(series):<32} "
                      f"{result['return_period']:12.5g} {lower:12.5g} {upper:12.5g}")
        for series in SERIES:
            empirical = panel["empirical"][series]
            k, n = empirical["exceedances"], empirical["sample_size"]
            estimate = (f"no exceedances; plotted at {n + 1} years (lower bound)" if k == 0
                        else f"{empirical['return_period']:.4g} years")
            print(f"  {EMPIRICAL_LABEL:<13} {series_label(series):<32} "
                  f"{k} of {n} values >= record; {estimate}")
    print("\ninf = unbounded return period.")

# =============================================================================
# Main
# =============================================================================


def main():
    """Fit every series for every month and plot the record-threshold return periods."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)
    reference = read_reference_months()
    raw = read_model_months("raw")
    bias_corrected = read_model_months(BIAS_CORRECTION_METHOD)
    progress = ProgressTracker(len(MONTHS) * len(SERIES) * len(METHODS) * NUMBER_OF_BOOTSTRAPS)
    print("Running bootstrap fits...")
    print("Progress:   0%", end="\r", flush=True)
    panels = {
        month: analyse_month(
            month,
            {
                "reference": reference["fit_values"][month],
                "raw": raw[month],
                "bias_corrected": bias_corrected[month],
            },
            reference["records"][month]["value"],
            progress,
            event_year=reference["records"][month]["year"],
        )
        for month in MONTHS
    }
    print_summary(panels)
    make_figure(panels)


if __name__ == "__main__":
    main()
