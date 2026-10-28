"""
Probability of at least one monthly exceedance within N years: sensitivity of
fig-06 (a, b) to the distribution and to bias correction.

Layout (3 rows x 2 columns)
---------------------------
Each row repeats fig-06 (a, b) for one distribution in ROW_DISTRIBUTIONS, by
default:

    row 1 (a, b)   GEV
    row 2 (c, d)   Gumbel
    row 3 (e, f)   GenEx

Every panel shows the reference (seNorge or ERA5), the raw model and the
bias-corrected model (BIAS_CORRECTION_METHOD), all fitted with the row's
distribution. Left column: Storm Hans threshold (log y-axis). Right column:
calendar-month record thresholds (linear y-axis). Panels in a column share their
y-axis so the rows can be compared directly. One legend is drawn in LEGEND_PANEL
at LEGEND_LOCATION.

Calculation (identical to fig-06)
---------------------------------
For each calendar month m, the distribution is fitted to the monthly-maximum
X-day precipitation sample and evaluated at the threshold x_m:
    p_m    = P(M_m > x_m)
    p_year = 1 - product_m (1 - p_m)          (months independent)
    P_N    = 1 - (1 - p_year)^N               (years independent, stationary)
for N = 1..AEP_YEARS_MAX, reported in percent.

Thresholds are always taken from the reference dataset: "storm_hans" is the
August 2023 value in every month; "monthly_record" is each calendar month's
highest value in OBSERVATION_YEARS, including 2023 (so the August record is
Storm Hans).

Bootstrap uncertainty: each month and dataset is bootstrapped independently
(BOOTSTRAP_METHOD "nonparametric" or "parametric"); replicate b combines the b-th
successful refit of every month into one P_N curve, and the band is the central
CONFIDENCE_LEVEL interval across replicates. Seeds follow fig-06, so each curve
reproduces fig-06 run with the same distribution and model data.

Output
------
FIGURE_FORMAT = "png" or "pdf"; fonts are embedded as TrueType. The default size
(7.0 x 8.0 in, about 178 x 203 mm) fits a full-page-width journal figure. On the
log axis, values below the lower limit are drawn at the limit (fitted values of
exactly 0 % are not drawn); the printed summary gives the unclipped values.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter)
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r
from Dunnsigouin_etal_2026 import config

# =============================================================================
# USER INPUTS — data selection and analysis
# =============================================================================
REFERENCE_DATASET = "senorge"  # "senorge" or "era5".
CATCHMENT = "regine_drammen"
X_DAYS = 2  # Precipitation accumulation length in days.
OBSERVATION_YEARS = [1957, 2025]  # Reference fitting period and record thresholds.
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]
BIAS_CORRECTION_METHOD = "mm_1step"  # "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy".

# One row per distribution, top to bottom: "GEV", "Gumbel", or "GenEx".
ROW_DISTRIBUTIONS = ["GEV", "Gumbel", "GenEx"]

AEP_YEARS_MAX = 10  # Horizons N = 1, 2, ..., AEP_YEARS_MAX years.
INCLUDE_STORM_HANS_IN_FIT = True  # August reference fit only; the record always includes Hans.
BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric".
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
# One column per threshold; panels in a column share these y-axis settings.
#   "y_scale": "linear" or "log".
#   "ylim":    (lower, upper) in percent; None for either limit selects it from
#              the data in that column (log: decade below/above, capped at 100 %;
#              linear: 0 and the next multiple of 10).
#   "yticks":  list of tick positions in percent, or None for automatic ticks.
COLUMNS = {
    "storm_hans": {
        "title": "Storm Hans threshold",
        "y_scale": "log",
        "ylim": (0.001, 10.0),
        "yticks": None,
    },
    "monthly_record": {
        "title": "Calendar-month record thresholds",
        "y_scale": "linear",
        "ylim": (0.0, 100.0),
        "yticks": [0, 20, 40, 60, 80, 100],
    },
}
FIGURE_WIDTH_IN = 7.0  # Full page width (about 178 mm).
FIGURE_HEIGHT_IN = 8.0
SHOW_GRID = True
LEGEND_PANEL = "b"  # Panel that carries the legend: "a" (top left) ... "f" (bottom right).
LEGEND_LOCATION = "upper left"  # Matplotlib location inside that panel, e.g. "upper left".
SERIES_COLORS = {
    "reference": "tab:orange",
    "raw": "tab:blue",
    "bias_corrected": "tab:grey",
}

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_FORMAT = "pdf"  # "png" or "pdf".
FIGURE_DPI = 300  # Resolution of PNG output.
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = f"fig-S5.{FIGURE_FORMAT}"
OUTPUT_PATH = OUTPUT_DIRECTORY / OUTPUT_FILENAME

# =============================================================================
# DATASET CONVENTIONS (fixed; must match the input files)
# =============================================================================
REFERENCE_FILE_YEARS = (1957, 2025)  # Year range encoded in the reference filename.
SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
MODEL_FILE_VARIABLE = "tp24"  # Variable name in the model sample filename.
MODEL_SAMPLE_VARIABLE = "tp24_max"  # Complete-lead monthly maxima.
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
MIN_AUTO_YLIM_PERCENT = 1e-4  # Floor for an automatic lower limit on a log axis.
AUGUST = 8
ALL_MONTHS = range(1, 13)
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST
THRESHOLD_ORDER = ["storm_hans", "monthly_record"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# Fixed orders so bootstrap seeds match fig-06 for the same distribution.
SEED_METHOD_ORDER = ["GEV", "Gumbel", "GenEx"]
SEED_GROUP_INDEX = {"reference": 0, "raw": 1, "bias_corrected": 1}
GROUPS = ["reference", "raw", "bias_corrected"]  # Plotting and legend order in every panel.
MODEL_GROUPS = ["raw", "bias_corrected"]

# =============================================================================
# Plot constants (sized for a 7-inch-wide figure)
# =============================================================================
FONT_SIZE = 10
TITLE_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9
X_LABEL = "years"
Y_LABEL = "Probability of at least one monthly exceedance within N years [%]"
DATASET_MARKERS = {"reference": "s", "raw": "o", "bias_corrected": "o"}
LINEWIDTH = 1.1
MARKER_SIZE = 3.5
BAND_ALPHA = 0.20
BAND_X_PADDING = 0.05  # Years the shading and x-axis extend beyond the first and last horizon.
GRID_ALPHA = 0.5

# =============================================================================
# Validation and labels
# =============================================================================


def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if BIAS_CORRECTION_METHOD not in {"mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:
        raise ValueError("Unsupported BIAS_CORRECTION_METHOD.")
    if not ROW_DISTRIBUTIONS:
        raise ValueError("ROW_DISTRIBUTIONS must contain at least one distribution.")
    for distribution in ROW_DISTRIBUTIONS:
        if distribution not in SEED_METHOD_ORDER:
            raise ValueError(f"ROW_DISTRIBUTIONS entries must be one of {SEED_METHOD_ORDER}.")
    panel_letters = panel_labels()
    if LEGEND_PANEL not in panel_letters:
        raise ValueError(f"LEGEND_PANEL must be one of {panel_letters}.")
    if not isinstance(AEP_YEARS_MAX, int) or AEP_YEARS_MAX < 1:
        raise ValueError("AEP_YEARS_MAX must be a positive integer.")
    if set(COLUMNS) != set(THRESHOLD_ORDER):
        raise ValueError(f"COLUMNS must contain exactly {THRESHOLD_ORDER}.")
    for name, column in COLUMNS.items():
        if column["y_scale"] not in {"linear", "log"}:
            raise ValueError(f"COLUMNS['{name}']['y_scale'] must be 'linear' or 'log'.")
        lower, upper = column["ylim"]
        if upper is not None and not 0 < upper <= 100:
            raise ValueError(f"COLUMNS['{name}'] upper y-limit must lie in (0, 100].")
        if lower is not None and column["y_scale"] == "log" and not lower > 0:
            raise ValueError(f"COLUMNS['{name}'] needs a positive lower limit on a log axis.")
        if lower is not None and upper is not None and not lower < upper:
            raise ValueError(f"COLUMNS['{name}'] lower y-limit must be below the upper.")
    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if not REFERENCE_FILE_YEARS[0] <= OBSERVATION_YEARS[0] <= OBSERVATION_YEARS[1] <= REFERENCE_FILE_YEARS[1]:
        raise ValueError(
            f"OBSERVATION_YEARS must lie within {REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}."
        )
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


def group_label(group):
    """Return the legend label of one dataset."""
    return {
        "reference": get_reference_name(),
        "raw": "Model raw",
        "bias_corrected": "Model bias corrected",
    }[group]


def panel_labels():
    """Return the panel letters row by row: a, b (row 1), c, d (row 2), ..."""
    n_panels = len(ROW_DISTRIBUTIONS) * len(THRESHOLD_ORDER)
    return [chr(ord("a") + index) for index in range(n_panels)]


def horizon_years():
    """Return the evaluated horizons 1..AEP_YEARS_MAX."""
    return np.arange(1, AEP_YEARS_MAX + 1)

# =============================================================================
# Input filenames
# =============================================================================


def make_reference_filename():
    """Construct the selected reference input filename."""
    first_year, last_year = REFERENCE_FILE_YEARS
    filename = (
        f"monthly_max_samples_{get_reference_variable()}_{X_DAYS}dayacc_{CATCHMENT}_"
        f"{first_year}-{last_year}.nc"
    )
    directory_key = "senorge_processed" if REFERENCE_DATASET == "senorge" else "era5_processed"
    return Path(config.dirs[directory_key]) / filename


def make_model_filename(group):
    """Construct the compact model input filename for "raw" or "bias_corrected"."""
    stem = (
        f"monthly_max_samples_{MODEL_FILE_VARIABLE}_{X_DAYS}dayacc_"
        f"{CATCHMENT.removeprefix('regine_')}_{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    correction = (
        "raw" if group == "raw"
        else f"bc_{BIAS_CORRECTION_METHOD}_{REFERENCE_DATASET}_"
        f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
    )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction}.nc"

# =============================================================================
# Read monthly samples and thresholds (each file is read once)
# =============================================================================


def read_reference_months():
    """Read every month's reference fit sample and both threshold definitions."""
    filename = make_reference_filename()
    variable = get_reference_variable()
    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")
    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")
        hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )
        selected = {
            month: ds[variable].sel(
                year=slice(OBSERVATION_YEARS[0], OBSERVATION_YEARS[1]), month=month
            ).load()
            for month in ALL_MONTHS
        }
    hans_in_range = OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
    fit_values = {}
    records = {}
    for month, data in selected.items():
        years = np.asarray(data["year"].values)
        values = np.asarray(data.values, dtype=float)
        finite = np.isfinite(values)
        years, values = years[finite], values[finite]
        if values.size < 10:
            raise ValueError(f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain.")
        # Record over all years in the observation period, including August 2023 (Hans).
        record_index = int(np.argmax(values))
        records[month] = {"value": float(values[record_index]), "year": int(years[record_index])}
        keep = np.ones(values.size, dtype=bool)
        if month == AUGUST and hans_in_range and not INCLUDE_STORM_HANS_IN_FIT:
            keep &= years != STORM_HANS_YEAR
        if np.count_nonzero(keep) < 10:
            raise ValueError(
                f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain in the fit."
            )
        fit_values[month] = values[keep]
    thresholds = {
        "storm_hans": {month: {"value": hans_value, "year": STORM_HANS_YEAR}
                       for month in ALL_MONTHS},
        "monthly_record": records,
    }
    return fit_values, thresholds


def read_model_months(group):
    """Read the pooled model sample of every calendar month from sample_month (YYYYMM)."""
    filename = make_model_filename(group)
    variable = MODEL_SAMPLE_VARIABLE
    if not filename.is_file():
        raise FileNotFoundError(f"Model file not found: {filename}")
    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")
        if "sample_month" not in ds:
            raise KeyError(f"Variable 'sample_month' was not found in {filename}.")
        if set(ds[variable].dims) != {"number", "i_date"}:
            raise ValueError(f"'{variable}' must have dimensions ('number', 'i_date').")
        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("'sample_month' must have dimension ('i_date',).")
        calendar_month = np.asarray(ds["sample_month"].values, dtype="int64") % 100
        values = np.asarray(ds[variable].transpose("number", "i_date").values, dtype=float)
    pooled = {}
    for month in ALL_MONTHS:
        month_values = values[:, calendar_month == month].ravel()
        month_values = month_values[np.isfinite(month_values)]
        if month_values.size < 10:
            raise ValueError(
                f"Fewer than 10 finite {group} model values were found for "
                f"{MONTH_NAMES[month - 1]}."
            )
        pooled[month] = month_values
    return pooled

# =============================================================================
# Distribution fitting and bootstrap uncertainty
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


def exceedance_probability(event_value, parameters, method):
    """Return the fitted probability of exceeding one threshold."""
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


class ProgressTracker:
    """Print integer percentage completion for requested bootstrap fits."""

    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.last_percent = -1

    def update(self):
        """Advance one bootstrap fit and print when percentage changes."""
        self.completed += 1
        percent = min(100, int(100 * self.completed / self.total))
        if percent != self.last_percent:
            print(f"Progress: {percent:3d}%", end="\r", flush=True)
            self.last_percent = percent
        if self.completed == self.total:
            print()


def bootstrap_distribution(values, method, random_seed, progress=None):
    """Fit a distribution and generate reusable bootstrap parameter sets."""
    parameters = fit_distribution(values, method)
    rng = np.random.default_rng(random_seed)
    bootstrap_parameters = []
    for _ in range(NUMBER_OF_BOOTSTRAPS):
        try:
            sample = make_bootstrap_sample(values, method, rng, parameters)
            fitted = fit_distribution(
                sample, method, initial_parameters=parameters if method == "GenEx" else None
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

# =============================================================================
# Monthly fits and N-year probability curves
# =============================================================================


def required_fits():
    """Return the (dataset, distribution) pairs needed by all panels, without duplicates."""
    pairs = []
    for distribution in ROW_DISTRIBUTIONS:
        for group in GROUPS:
            if (group, distribution) not in pairs:
                pairs.append((group, distribution))
    return pairs


def fit_all_months(samples, progress=None):
    """Fit every required (dataset, distribution) pair for all 12 months."""
    analyses = {}
    for group, method in required_fits():
        method_index = SEED_METHOD_ORDER.index(method)
        for month_index, month in enumerate(ALL_MONTHS):
            seed = (RANDOM_SEED + 10000 * month_index
                    + 1000 * SEED_GROUP_INDEX[group] + method_index)
            analyses[(group, method, month)] = bootstrap_distribution(
                samples[group][month], method, seed, progress
            )
    return analyses


def annual_probability(monthly_probabilities):
    """Combine monthly probabilities (last axis = month) into p_year."""
    monthly_probabilities = np.clip(np.asarray(monthly_probabilities, dtype=float), 0.0, 1.0)
    return 1.0 - np.prod(1.0 - monthly_probabilities, axis=-1)


def horizon_probability(p_year, years):
    """Return P_N = 1 - (1 - p_year)^N for every N, accurate for tiny p_year."""
    p_year = np.clip(np.asarray(p_year, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        log_survival = np.log1p(-p_year)
    return -np.expm1(np.multiply.outer(log_survival, years))


def calculate_curve(group, method, monthly_thresholds, analyses):
    """Return the fitted and bootstrap P_N curve (in %) of one dataset and distribution."""
    years = horizon_years()
    alpha = 1.0 - CONFIDENCE_LEVEL
    fitted_monthly = []
    bootstrap_monthly = []
    for month in ALL_MONTHS:
        fit = analyses[(group, method, month)]
        threshold = monthly_thresholds[month]["value"]
        fitted_monthly.append(exceedance_probability(threshold, fit["parameters"], method))
        bootstrap_monthly.append([
            exceedance_probability(threshold, parameters, method)
            for parameters in fit["bootstrap_parameters"]
        ])
    # Replicate b uses the b-th successful fit in every month, as in fig-06.
    n_bootstraps = min(map(len, bootstrap_monthly))
    bootstrap_monthly = np.array([values[:n_bootstraps] for values in bootstrap_monthly]).T
    fitted_year = float(annual_probability(fitted_monthly))
    bootstrap_year = annual_probability(bootstrap_monthly)
    bootstrap_curves = 100.0 * horizon_probability(bootstrap_year, years)
    lower, upper = np.percentile(
        bootstrap_curves, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0
    )
    return {
        "annual_probability": 100.0 * fitted_year,
        "annual_bootstrap": 100.0 * bootstrap_year,
        "probability": 100.0 * horizon_probability(fitted_year, years),
        "lower": lower,
        "upper": upper,
        "n_bootstraps": n_bootstraps,
    }


def calculate_all_curves(thresholds, analyses):
    """Return curves[row_index][threshold_type][group] for every panel."""
    curves = []
    for method in ROW_DISTRIBUTIONS:
        curves.append({
            threshold_type: {
                group: calculate_curve(group, method, thresholds[threshold_type], analyses)
                for group in GROUPS
            }
            for threshold_type in THRESHOLD_ORDER
        })
    return curves

# =============================================================================
# Plotting
# =============================================================================


def apply_publication_style():
    """Set consistent, journal-friendly matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.linewidth": 0.6,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def resolve_ylim(column, column_curves):
    """Return (lower, upper) y-limits shared by one column, filling None from the data."""
    lower, upper = column["ylim"]
    values = np.concatenate([
        np.concatenate([curve["probability"], curve["lower"], curve["upper"]])
        for curve in column_curves
    ])
    if column["y_scale"] == "linear":
        if lower is None:
            lower = 0.0
        if upper is None:
            upper = min(100.0, 10.0 * np.ceil(np.nanmax(values) / 10.0))
        return lower, upper
    positive = values[values > 0]
    if lower is None:
        smallest = positive.min() if positive.size else MIN_AUTO_YLIM_PERCENT
        lower = max(10.0 ** np.floor(np.log10(smallest)), MIN_AUTO_YLIM_PERCENT)
    if upper is None:
        largest = positive.max() if positive.size else 100.0
        upper = min(10.0 ** np.ceil(np.log10(largest)), 100.0)
    return lower, upper


def format_percent(value, _position):
    """Label ticks as plain percentages (0.01, 0.1, 1, 10, 100)."""
    return f"{value:g}" if value > 0 else "0"


def apply_y_scale(axis, column, ylim):
    """Apply one column's y scale, limits and ticks."""
    axis.set_yscale(column["y_scale"])
    axis.set_ylim(*ylim)
    if column["yticks"] is not None:
        axis.yaxis.set_major_locator(FixedLocator(column["yticks"]))
    elif column["y_scale"] == "log":
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    else:
        axis.yaxis.set_major_locator(MaxNLocator(5))
    if column["y_scale"] == "log":
        axis.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        axis.yaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FuncFormatter(format_percent))


def draw_curve(axis, years, curve, group, ylim):
    """Draw one fitted P_N curve with its bootstrap band."""
    minimum, maximum = ylim
    color = SERIES_COLORS[group]
    band_lower = np.clip(curve["lower"], minimum, maximum)
    band_upper = np.clip(curve["upper"], minimum, maximum)
    # Extend the band flat by BAND_X_PADDING beyond the first and last horizon.
    band_years = np.concatenate(([years[0] - BAND_X_PADDING], years,
                                 [years[-1] + BAND_X_PADDING]))
    band_lower = np.concatenate(([band_lower[0]], band_lower, [band_lower[-1]]))
    band_upper = np.concatenate(([band_upper[0]], band_upper, [band_upper[-1]]))
    axis.fill_between(band_years, band_lower, band_upper, color=color, alpha=BAND_ALPHA,
                      linewidth=0, zorder=2)
    # Values below the lower limit (e.g. exactly 0 % on a log axis) are not drawn.
    fitted = np.where(curve["probability"] >= minimum, curve["probability"], np.nan)
    axis.plot(
        years, fitted, color=color, linewidth=LINEWIDTH,
        marker=DATASET_MARKERS[group], markersize=MARKER_SIZE,
        markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.4,
        zorder=3, clip_on=False,
    )


def draw_legend(axis):
    """One legend for the whole figure: colour and marker identify the dataset."""
    handles = [
        Line2D([0], [0], color=SERIES_COLORS[group], linewidth=LINEWIDTH,
               marker=DATASET_MARKERS[group], markersize=MARKER_SIZE + 0.5,
               markerfacecolor=SERIES_COLORS[group], markeredgecolor="white",
               markeredgewidth=0.4, label=group_label(group))
        for group in GROUPS
    ]
    legend = axis.legend(handles=handles, loc=LEGEND_LOCATION, frameon=False, framealpha=0.9,
                         edgecolor="none", handlelength=1.8, borderaxespad=0.3,
                         borderpad=0.3, labelspacing=0.3)
    legend.set_zorder(10)


def format_panel(axis, panel_label, title, show_x_labels):
    """Title, x-axis, spines and grid for one panel."""
    axis.set_title(f"{panel_label}) {title}", loc="left", pad=3)
    axis.set_xlim(1 - BAND_X_PADDING, AEP_YEARS_MAX + BAND_X_PADDING)
    if AEP_YEARS_MAX <= 12:
        axis.set_xticks(horizon_years())
    else:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.tick_params(axis="x", labelbottom=show_x_labels)
    axis.spines[["top", "right"]].set_visible(False)
    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.5, alpha=GRID_ALPHA)
        axis.set_axisbelow(True)


def plot_figure(curves):
    """Plot P_N against horizon N: one row per distribution, one column per threshold."""
    apply_publication_style()
    n_rows = len(ROW_DISTRIBUTIONS)
    figure, axes = plt.subplots(
        n_rows, len(THRESHOLD_ORDER), figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        sharex=True, sharey="col", squeeze=False, layout="constrained",
    )
    figure.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.06)
    years = horizon_years()
    for column_index, threshold_type in enumerate(THRESHOLD_ORDER):
        column = COLUMNS[threshold_type]
        ylim = resolve_ylim(column, [
            curve for row_curves in curves for curve in row_curves[threshold_type].values()
        ])
        for row_index, distribution in enumerate(ROW_DISTRIBUTIONS):
            axis = axes[row_index, column_index]
            for group in GROUPS:
                draw_curve(axis, years, curves[row_index][threshold_type][group], group, ylim)
            apply_y_scale(axis, column, ylim)
            panel_label = panel_labels()[row_index * len(THRESHOLD_ORDER) + column_index]
            format_panel(axis, panel_label, f"{distribution}: {column['title']}",
                         show_x_labels=row_index == n_rows - 1)
    legend_row, legend_column = divmod(panel_labels().index(LEGEND_PANEL), len(THRESHOLD_ORDER))
    draw_legend(axes[legend_row, legend_column])
    figure.supxlabel(X_LABEL, fontsize=FONT_SIZE + 1)
    figure.supylabel(Y_LABEL, fontsize=FONT_SIZE + 1)
    if WRITE_TO_FILE:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(OUTPUT_PATH, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
                       bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print("Wrote:", OUTPUT_PATH)
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Reporting and main workflow
# =============================================================================


def print_summary(curves, thresholds):
    """Print thresholds, annual probabilities, and P_N curves with intervals."""
    alpha = 1.0 - CONFIDENCE_LEVEL
    print("Selected settings")
    print("-----------------")
    print(f"Reference:        {get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}")
    print(f"Reference file:   {make_reference_filename()}")
    for group in MODEL_GROUPS:
        print(f"{group_label(group) + ' file:':<18}{make_model_filename(group)}")
    print(f"Horizons:         1-{AEP_YEARS_MAX} years")
    print(f"Bootstrap:        {BOOTSTRAP_METHOD}, n={NUMBER_OF_BOOTSTRAPS}")
    print(f"Include Hans fit: {INCLUDE_STORM_HANS_IN_FIT}")
    print("Assumption: exceedances are independent between months and years.")
    print(f"\nStorm Hans threshold: {thresholds['storm_hans'][AUGUST]['value']:.3f} mm in every month")
    print("Calendar-month record thresholds:")
    for month in ALL_MONTHS:
        record = thresholds["monthly_record"][month]
        print(f"  {MONTH_NAMES[month - 1]:9s} {record['value']:.3f} mm ({record['year']})")
    for row_index, distribution in enumerate(ROW_DISTRIBUTIONS):
        for threshold_type in THRESHOLD_ORDER:
            title = f"Row {row_index + 1}: {distribution}, {COLUMNS[threshold_type]['title']}"
            print(f"\n{title}\n{'-' * len(title)}")
            for group in GROUPS:
                curve = curves[row_index][threshold_type][group]
                annual = curve["annual_probability"]
                low, high = np.percentile(curve["annual_bootstrap"],
                                          [100 * alpha / 2, 100 * (1 - alpha / 2)])
                period = f"{100.0 / annual:.4g} years" if annual > 0 else "unbounded"
                print(f"{group_label(group)}  (bootstrap replicates: {curve['n_bootstraps']})")
                print(f"  Annual probability {annual:.4g}% [{low:.4g}, {high:.4g}]%; "
                      f"return period {period}")
                print(f"  {'N':>4} {'P_N [%]':>11} {'Lower':>11} {'Upper':>11}")
                for index, years in enumerate(horizon_years()):
                    print(f"  {years:>4d} {curve['probability'][index]:11.4g} "
                          f"{curve['lower'][index]:11.4g} {curve['upper'][index]:11.4g}")


def main():
    """Fit the monthly distributions and plot the N-year exceedance panels."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)
    reference_fit_values, thresholds = read_reference_months()
    samples = {"reference": reference_fit_values}
    for group in MODEL_GROUPS:
        samples[group] = read_model_months(group)
    progress = ProgressTracker(len(required_fits()) * 12 * NUMBER_OF_BOOTSTRAPS)
    print("Running bootstrap fits for all 12 months...")
    analyses = fit_all_months(samples, progress)
    curves = calculate_all_curves(thresholds, analyses)
    print_summary(curves, thresholds)
    plot_figure(curves)


if __name__ == "__main__":
    main()
