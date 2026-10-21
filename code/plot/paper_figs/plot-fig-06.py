"""
Probability of exceeding a precipitation threshold in any calendar month as a
function of the horizon N (years), using one monthly extreme-value distribution.

For each calendar month m, the selected distribution (DISTRIBUTION) is fitted to
the sample of monthly maximum X-day precipitation values. For a threshold x_m,
the fitted monthly exceedance probability is
    p_m = P(M_m > x_m),
where M_m is the maximum precipitation in month m of a given year. Because M_m
is already a monthly maximum, p_m is also the probability that at least one
X-day event in that month exceeds x_m during that year.
Assuming exceedances are independent between calendar months, the probability
of at least one exceedance in any month during one year is
    p_year = 1 - product_m(1 - p_m).
Assuming years are independent and have the same exceedance probabilities, the
probability of at least one exceedance within N years is
    P_N = 1 - (1 - p_year)^N,
evaluated for every N in 1..AEP_YEARS_MAX and reported as 100 * P_N (%).
These conversions rely on the stated independence and stationarity assumptions.

Thresholds
----------
"storm_hans": the August 2023 Storm Hans monthly-maximum value, the same in
every calendar month.
"monthly_record_without_hans": each calendar month's own record in the
configured observation period. August 2023 is excluded from the August record;
other months in 2023, including May, remain eligible.
Both thresholds are always taken from the reference dataset.

Bootstrap uncertainty
---------------------
Each month and dataset is bootstrapped independently. With BOOTSTRAP_METHOD =
"nonparametric", a sample of the same size as the monthly sample is drawn with
replacement and refitted; with "parametric", a sample of the same size is
simulated from the fitted distribution and refitted. For replicate b, the 12
refitted monthly probabilities p_m^(b) give p_year^(b), and
    P_N^(b) = 1 - (1 - p_year^(b))^N
for every N, so each replicate is one complete curve. The band at each N is the
central CONFIDENCE_LEVEL percentile interval of P_N^(b) across replicates.
Because P_N increases monotonically with p_year, the band is essentially the
annual-probability interval mapped through the same formula.
Failed fits are skipped; at least MIN_SUCCESSFUL_BOOTSTRAP_FRACTION of the
requested fits must succeed for every month and dataset.

Datasets
--------
The model is always analysed. With INCLUDE_REFERENCE = True the same calculation
is made for the reference dataset (seNorge or ERA5); with False its fits are
skipped entirely, which halves the run time.

Figure
------
One panel: horizon N on the x-axis, P_N on the y-axis (logarithmic by default).
Colour identifies the dataset (orange reference, blue model). Line style and
marker identify the threshold: Storm Hans uses solid lines with filled circles,
the calendar-month records dashed lines with open squares. Shading shows the
bootstrap interval. On a log axis, fitted values of exactly 0% are not drawn and
band edges below the lower limit are drawn at the limit; the printed summary
gives the unclipped values.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r
from Dunnsigouin_etal_2026 import config

# =============================================================================
# USER INPUTS — data selection
# =============================================================================
REFERENCE_DATASET = "senorge"  # "senorge" or "era5".
CATCHMENT = "regine_drammen"
X_DAYS = 2  # Precipitation accumulation length in days.
OBSERVATION_YEARS = [1957, 2025]  # Reference fitting period and record thresholds.
REFERENCE_FILE_YEARS = [1957, 2025]  # Full period encoded in the reference filename.
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

# Must match the settings used to generate the model sample file.
MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy".
MODEL_VARIABLE = "tp24"
MODEL_SAMPLING_GROUP = "full"  # "full", "split1", "split2", ...
FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2
SUBSAMPLE_MODEL_TO_REFERENCE_LENGTH = False  # Match reference sample size in each month.

# Optional explicit input paths; None builds filenames from the settings above.
REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================
DISTRIBUTION = "GEV"  # "GEV", "Gumbel", or "GenEx".
AEP_YEARS_MAX = 10  # Horizons N = 1, 2, ..., AEP_YEARS_MAX years.
INCLUDE_REFERENCE = True  # False analyses the model only (thresholds still from reference).
INCLUDE_STORM_HANS_IN_FIT = True # August reference fit only; not the record threshold.
BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric".
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
WRITE_TO_FILE = True  # Set True to save the figure.
SHOW_FIGURE = True
FIGURE_DPI = 300
OUTPUT_DIRECTORY = Path(config.dirs["fig"])
OUTPUT_FILENAME = "fig-06.png"
OUTPUT_PATH = OUTPUT_DIRECTORY / OUTPUT_FILENAME

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIGURE_SIZE = (7.5, 5.0)  # Width and height in inches.
Y_SCALE = "log"  # "log" or "linear".

# Y-axis limits in percent. With a log axis, None for the lower limit selects
# the decade below the smallest positive value plotted, floored at
# MIN_AUTO_YLIM_PERCENT. With a linear axis, None for the lower limit gives 0.
PROBABILITY_YLIM = (None, 100.0)
MIN_AUTO_YLIM_PERCENT = 1e-4

THRESHOLD_STYLES = {
    "storm_hans": {
        "label": "Storm Hans threshold", "marker": "o", "filled": True, "linestyle": "-",
    },
    "monthly_record_without_hans": {
        "label": "Calendar-month record thresholds", "marker": "s", "filled": False,
        "linestyle": "--",
    },
}
THRESHOLD_ORDER = ["storm_hans", "monthly_record_without_hans"]
SHOW_MARKERS = True
SHOW_GRID = True
GRID_ALPHA = 0.35
BAND_ALPHA = 0.18
BAND_X_PADDING = 0.05  # Years the shading and x-axis extend beyond the first and last horizon.
OBSERVATION_COLOR = "tab:orange"
MODEL_COLOR = "tab:blue"
LINEWIDTH = 1.8
MARKER_SIZE = 4.5
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
LEGEND_FONTSIZE = 10
LEGEND_LOCATION = "lower right"  # Any matplotlib legend location, e.g. "upper left".
LEGEND_FRAME = True  # White background so the legend stays readable over bands.

# =============================================================================
# DATASET CONVENTIONS
# =============================================================================
AUGUST = 8
ALL_MONTHS = range(1, 13)
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST
SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# Fixed order so bootstrap seeds match fig-06 for the same distribution.
SEED_METHOD_ORDER = ["GEV", "Gumbel", "GenEx"]
SEED_GROUP_INDEX = {"reference": 0, "model": 1}

# =============================================================================
# Validation and labels
# =============================================================================


def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if MODEL_DATA_METHOD not in {"raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:
        raise ValueError("Unsupported MODEL_DATA_METHOD.")
    if DISTRIBUTION not in SEED_METHOD_ORDER:
        raise ValueError(f"DISTRIBUTION must be one of {SEED_METHOD_ORDER}.")
    if not isinstance(AEP_YEARS_MAX, int) or AEP_YEARS_MAX < 1:
        raise ValueError("AEP_YEARS_MAX must be a positive integer.")
    if not isinstance(INCLUDE_REFERENCE, bool):
        raise TypeError("INCLUDE_REFERENCE must be True or False.")
    if set(THRESHOLD_STYLES) != set(THRESHOLD_ORDER):
        raise ValueError(f"THRESHOLD_STYLES must contain exactly {THRESHOLD_ORDER}.")
    if Y_SCALE not in {"log", "linear"}:
        raise ValueError("Y_SCALE must be 'log' or 'linear'.")
    lower, upper = PROBABILITY_YLIM
    if upper is None or not 0 < upper <= 100:
        raise ValueError("PROBABILITY_YLIM upper limit must lie in (0, 100].")
    if lower is not None:
        if Y_SCALE == "log" and not 0 < lower < upper:
            raise ValueError("With a log axis the lower limit must be positive and below the upper.")
        if Y_SCALE == "linear" and not 0 <= lower < upper:
            raise ValueError("PROBABILITY_YLIM lower limit must be in [0, upper).")
    if not MIN_AUTO_YLIM_PERCENT > 0:
        raise ValueError("MIN_AUTO_YLIM_PERCENT must be positive.")
    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:
        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
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


def get_model_label():
    """Return the model display label."""
    return "Model" if MODEL_DATA_METHOD == "raw" else "Model bias corrected"


def analysed_groups():
    """Return the datasets analysed and plotted, in plotting order."""
    return ["reference", "model"] if INCLUDE_REFERENCE else ["model"]


def group_label(group):
    """Return the display label of one dataset."""
    return get_reference_name() if group == "reference" else get_model_label()


def group_color(group):
    """Return the colour of one dataset."""
    return OBSERVATION_COLOR if group == "reference" else MODEL_COLOR


def horizon_years():
    """Return the evaluated horizons 1..AEP_YEARS_MAX."""
    return np.arange(1, AEP_YEARS_MAX + 1)

# =============================================================================
# Lead selection and input filenames
# =============================================================================


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
    index = int(MODEL_SAMPLING_GROUP.removeprefix("split")) - 1
    lead_start, lead_end = build_lead_bins()[index]
    return f"tp24_max_lead{lead_start}_{lead_end}"


def make_reference_filename():
    """Construct the selected reference input filename."""
    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)
    first_year, last_year = REFERENCE_FILE_YEARS
    variable = get_reference_variable()
    filename = (
        f"monthly_max_samples_{variable}_{X_DAYS}dayacc_{CATCHMENT}_"
        f"{first_year}-{last_year}.nc"
    )
    directory_key = "senorge_processed" if REFERENCE_DATASET == "senorge" else "era5_processed"
    return Path(config.dirs[directory_key]) / filename


def make_model_filename():
    """Construct the compact model input filename."""
    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)
    catchment_id = CATCHMENT.removeprefix("regine_")
    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_{catchment_id}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )
    correction = (
        "raw" if MODEL_DATA_METHOD == "raw"
        else f"bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}_"
        f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
    )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{correction}.nc"

# =============================================================================
# Read and prepare monthly samples
# =============================================================================


def read_reference_month(month):
    """Read one reference monthly-maximum sample and its observed record."""
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
        hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )

    years = np.asarray(selected["year"].values)
    values = np.asarray(selected.values, dtype=float)
    finite = np.isfinite(values)
    years, values = years[finite], values[finite]
    if values.size < 10:
        raise ValueError(f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain.")

    hans_in_range = OBSERVATION_YEARS[0] <= STORM_HANS_YEAR <= OBSERVATION_YEARS[1]
    record_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST and hans_in_range:
        record_mask &= years != STORM_HANS_YEAR
    record_values = values[record_mask]
    record_years = years[record_mask]
    if record_values.size == 0:
        raise ValueError(f"No values remain for the {MONTH_NAMES[month - 1]} record.")

    fit_mask = np.ones(values.size, dtype=bool)
    if month == AUGUST and hans_in_range and not INCLUDE_STORM_HANS_IN_FIT:
        fit_mask &= years != STORM_HANS_YEAR
    fit_values = values[fit_mask]
    if fit_values.size < 10:
        raise ValueError(f"Fewer than 10 finite {MONTH_NAMES[month - 1]} values remain in the fit.")

    record_index = int(np.argmax(record_values))
    return {
        "fit_values": fit_values,
        "storm_hans_value": hans_value,
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
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")
        if "sample_month" not in ds:
            raise KeyError(f"Variable 'sample_month' was not found in {filename}.")
        if set(ds[variable].dims) != {"number", "i_date"}:
            raise ValueError(f"'{variable}' must have dimensions ('number', 'i_date').")
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


def read_all_reference_months():
    """Read all reference months and construct both threshold definitions."""
    monthly = {month: read_reference_month(month) for month in ALL_MONTHS}
    hans_value = monthly[AUGUST]["storm_hans_value"]
    thresholds = {
        "storm_hans": {
            month: {"value": hans_value, "year": STORM_HANS_YEAR}
            for month in ALL_MONTHS
        },
        "monthly_record_without_hans": {
            month: {
                "value": monthly[month]["record_value"],
                "year": monthly[month]["record_year"],
            }
            for month in ALL_MONTHS
        },
    }
    return monthly, thresholds


def build_all_month_analysis(progress=None):
    """Fit the selected distribution to every analysed dataset for all 12 months."""
    reference_months, thresholds = read_all_reference_months()
    method_index = SEED_METHOD_ORDER.index(DISTRIBUTION)
    analyses = {}
    for month_index, month in enumerate(ALL_MONTHS):
        samples = {}
        if INCLUDE_REFERENCE:
            samples["reference"] = reference_months[month]["fit_values"]
        samples["model"] = subsample_model_values(
            read_model_month(month), reference_months[month]["fit_values"].size,
            RANDOM_SEED + 100 * month_index,
        )
        for group, values in samples.items():
            seed = (RANDOM_SEED + 10000 * month_index
                    + 1000 * SEED_GROUP_INDEX[group] + method_index)
            analyses[(month, group)] = bootstrap_distribution(
                values, DISTRIBUTION, seed, progress
            )
    return thresholds, analyses


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


def calculate_threshold_curves(monthly_thresholds, analyses):
    """Return fitted and bootstrap P_N curves (in %) for one threshold definition."""
    years = horizon_years()
    alpha = 1.0 - CONFIDENCE_LEVEL
    results = {}
    for group in analysed_groups():
        fitted_monthly = []
        bootstrap_monthly = []
        for month in ALL_MONTHS:
            fit = analyses[(month, group)]
            threshold = monthly_thresholds[month]["value"]
            fitted_monthly.append(exceedance_probability(threshold, fit["parameters"], DISTRIBUTION))
            bootstrap_monthly.append([
                exceedance_probability(threshold, parameters, DISTRIBUTION)
                for parameters in fit["bootstrap_parameters"]
            ])
        # Replicate b uses the b-th successful fit in every month, as in fig-06.
        n_bootstraps = min(map(len, bootstrap_monthly))
        bootstrap_monthly = np.array([values[:n_bootstraps] for values in bootstrap_monthly]).T

        fitted_year = float(annual_probability(fitted_monthly))
        bootstrap_year = annual_probability(bootstrap_monthly)  # (n_bootstraps,)
        bootstrap_curves = 100.0 * horizon_probability(bootstrap_year, years)
        lower, upper = np.percentile(
            bootstrap_curves, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0
        )
        results[group] = {
            "annual_probability": 100.0 * fitted_year,
            "annual_bootstrap": 100.0 * bootstrap_year,
            "probability": 100.0 * horizon_probability(fitted_year, years),
            "lower": lower,
            "upper": upper,
            "n_bootstraps": n_bootstraps,
        }
    return results


def calculate_all_curves(thresholds, analyses):
    """Calculate P_N curves for both threshold definitions."""
    return {
        threshold_type: calculate_threshold_curves(thresholds[threshold_type], analyses)
        for threshold_type in THRESHOLD_ORDER
    }


def threshold_label(threshold_type, thresholds):
    """Return a descriptive threshold label for summaries."""
    if threshold_type == "storm_hans":
        return f"Storm Hans, August 2023 ({thresholds[AUGUST]['value']:.1f} mm) in every month"
    return "Calendar-month records; August 2023 excluded from the August record"

# =============================================================================
# Plotting
# =============================================================================


def resolve_ylim(results):
    """Return y-axis limits, choosing the lower limit from the data if unset."""
    lower, upper = PROBABILITY_YLIM
    if lower is not None:
        return lower, upper
    if Y_SCALE == "linear":
        return 0.0, upper
    values = np.concatenate([
        np.concatenate([result["probability"], result["lower"], result["upper"]])
        for threshold_results in results.values()
        for result in threshold_results.values()
    ])
    positive = values[values > 0]
    smallest = positive.min() if positive.size else MIN_AUTO_YLIM_PERCENT
    lower = 10.0 ** np.floor(np.log10(smallest))
    return max(lower, MIN_AUTO_YLIM_PERCENT), upper


def format_percent(value, _position):
    """Label log ticks as plain percentages (0.01, 0.1, 1, 10, 100)."""
    if value <= 0:
        return ""
    exponent = np.log10(value)
    if not np.isclose(exponent, np.round(exponent), atol=1e-10):
        return ""
    return f"{value:g}"

def draw_curve(axis, years, result, color, style, ylim):
    """Draw one fitted P_N curve with its bootstrap band."""
    minimum, maximum = ylim
    band_lower = np.clip(result["lower"], minimum, maximum)
    band_upper = np.clip(result["upper"], minimum, maximum)
    # Extend the band flat by BAND_X_PADDING beyond the first and last horizon.
    band_years = np.concatenate(([years[0] - BAND_X_PADDING], years,
                                 [years[-1] + BAND_X_PADDING]))
    band_lower = np.concatenate(([band_lower[0]], band_lower, [band_lower[-1]]))
    band_upper = np.concatenate(([band_upper[0]], band_upper, [band_upper[-1]]))
    axis.fill_between(band_years, band_lower, band_upper, color=color, alpha=BAND_ALPHA,
                      linewidth=0, zorder=2)
    fitted = np.where(result["probability"] >= minimum, result["probability"], np.nan)
    axis.plot(
        years, fitted, color=color, linestyle=style["linestyle"], linewidth=LINEWIDTH,
        marker=style["marker"] if SHOW_MARKERS else None, markersize=MARKER_SIZE,
        markerfacecolor=color if style["filled"] else "white", markeredgecolor=color,
        zorder=3, clip_on=False,
    )
    


def draw_legend(axis):
    """Explain colour (dataset) and line style (threshold) separately."""
    handles = [
        Line2D([0], [0], color=group_color(group), linewidth=LINEWIDTH, label=group_label(group))
        for group in analysed_groups()
    ]
    for threshold_type in THRESHOLD_ORDER:
        style = THRESHOLD_STYLES[threshold_type]
        handles.append(Line2D(
            [0], [0], color="0.25", linewidth=LINEWIDTH, linestyle=style["linestyle"],
            marker=style["marker"] if SHOW_MARKERS else None, markersize=MARKER_SIZE,
            markeredgecolor="0.25", markerfacecolor="0.25" if style["filled"] else "white",
            label=style["label"],
        ))
    axis.legend(handles=handles, frameon=LEGEND_FRAME, framealpha=0.9, edgecolor="none",
                fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOCATION, ncol=1, handlelength=2.6)


def plot_horizon_curves(results):
    """Plot P_N against horizon N for both thresholds and the analysed datasets."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    years = horizon_years()
    ylim = resolve_ylim(results)
    for threshold_type in THRESHOLD_ORDER:
        style = THRESHOLD_STYLES[threshold_type]
        for group in analysed_groups():
            draw_curve(axis, years, results[threshold_type][group], group_color(group),
                       style, ylim)

    axis.set_yscale(Y_SCALE)
    axis.set_ylim(*ylim)
    if Y_SCALE == "log":
        axis.yaxis.set_major_formatter(FuncFormatter(format_percent))
    axis.set_xlim(1 - BAND_X_PADDING, AEP_YEARS_MAX + BAND_X_PADDING)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("years", fontsize=AXIS_LABELSIZE)
    axis.set_ylabel("Probability of at least one monthly\nexceedance within N years [%]",
                    fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE, direction="out")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if SHOW_GRID:
        axis.grid(axis="y", which="major", linestyle=":", linewidth=0.7, alpha=GRID_ALPHA)
        axis.set_axisbelow(True)
    draw_legend(axis)

    if WRITE_TO_FILE:
        filename = OUTPUT_PATH
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

# =============================================================================
# Reporting and main workflow
# =============================================================================


def print_summary(results, thresholds):
    """Print thresholds, annual probabilities, and P_N curves with intervals."""
    alpha = 1.0 - CONFIDENCE_LEVEL
    print("Selected settings")
    print("-----------------")
    print(f"Distribution:     {DISTRIBUTION}")
    print(f"Reference:        {get_reference_name()} {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
          f"{'' if INCLUDE_REFERENCE else ' (thresholds only; not fitted)'}")
    print(f"Reference file:   {make_reference_filename()}")
    print(f"Model file:       {make_model_filename()}")
    print(f"Horizons:         1-{AEP_YEARS_MAX} years")
    print(f"Bootstrap:        {BOOTSTRAP_METHOD}, n={NUMBER_OF_BOOTSTRAPS}")
    print(f"Include Hans fit: {INCLUDE_STORM_HANS_IN_FIT}")
    print("Assumption: exceedances are independent between months and years.")
    for threshold_type in THRESHOLD_ORDER:
        title = THRESHOLD_STYLES[threshold_type]["label"]
        print(f"\n{title}\n{'-' * len(title)}")
        print(f"Threshold: {threshold_label(threshold_type, thresholds[threshold_type])}")
        if threshold_type == "monthly_record_without_hans":
            for month in ALL_MONTHS:
                threshold = thresholds[threshold_type][month]
                print(f"  {MONTH_NAMES[month - 1]:9s} "
                      f"{threshold['value']:.3f} mm ({threshold['year']})")
        for group in analysed_groups():
            result = results[threshold_type][group]
            annual = result["annual_probability"]
            low, high = np.percentile(result["annual_bootstrap"],
                                      [100 * alpha / 2, 100 * (1 - alpha / 2)])
            period = f"{100.0 / annual:.4g} years" if annual > 0 else "unbounded"
            print(f"\n{group_label(group)}  (bootstrap replicates: {result['n_bootstraps']})")
            print(f"  Annual probability {annual:.4g}% [{low:.4g}, {high:.4g}]%; "
                  f"return period {period}")
            print(f"  {'N':>4} {'P_N [%]':>11} {'Lower':>11} {'Upper':>11}")
            for index, years in enumerate(horizon_years()):
                print(f"  {years:>4d} {result['probability'][index]:11.4g} "
                      f"{result['lower'][index]:11.4g} {result['upper'][index]:11.4g}")


def main():
    """Fit the monthly distributions and plot N-year exceedance curves."""
    validate_settings()
    print("Figure output:", OUTPUT_PATH)

    total = 12 * len(analysed_groups()) * NUMBER_OF_BOOTSTRAPS
    progress = ProgressTracker(total)
    print(f"Running {DISTRIBUTION} bootstrap fits for all 12 months...")

    thresholds, analyses = build_all_month_analysis(progress)
    results = calculate_all_curves(thresholds, analyses)
    print_summary(results, thresholds)
    plot_horizon_curves(results)


if __name__ == "__main__":
    main()
