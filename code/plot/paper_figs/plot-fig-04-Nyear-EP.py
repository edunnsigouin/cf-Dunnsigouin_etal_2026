"""

Estimate the probability of exceeding a precipitation threshold in any calendar

month over an N-year horizon using monthly extreme-value fits.

For each calendar month m, one distribution is fitted to the sample of monthly

maximum X-day precipitation values. For a threshold x, the fitted monthly

exceedance probability is

    p_m = P(M_m > x),

where M_m is the maximum precipitation in month m of a given year. Because M_m

is already a monthly maximum, p_m is also the probability that at least one

X-day event in that month exceeds x during that year.

Assuming exceedances are independent between calendar months, the probability

of no threshold exceedance anywhere in one year is

    P(no exceedance in one year) = product_m(1 - p_m),

so the probability of at least one exceedance in any month during one year is

    p_year = 1 - product_m(1 - p_m).

Assuming years are independent and have the same exceedance probabilities, the

probability of at least one exceedance over N years is

    P_N = 1 - (1 - p_year)^N.

The script reports 100 * P_N as a percentage. These month-to-year and year-to-N-

year conversions rely on the stated independence and stationarity assumptions.

Bootstrap uncertainty

---------------------

Each month, dataset (reference/model), and fitted distribution is bootstrapped

independently. With BOOTSTRAP_METHOD = "nonparametric", a bootstrap sample of

the same size as the original monthly sample is drawn with replacement and the

distribution is refitted. With BOOTSTRAP_METHOD = "parametric", a sample of the

same size is simulated from the fitted distribution and then refitted. This is

repeated NUMBER_OF_BOOTSTRAPS times.

For bootstrap replicate b, the refitted distribution for each month gives

p_m^(b). The 12 monthly probabilities from the same replicate index are then

combined as

    p_year^(b) = 1 - product_m(1 - p_m^(b)),

    P_N^(b) = 1 - (1 - p_year^(b))^N.

The resulting P_N bootstrap sample is summarized by its median, interquartile

range, and central CONFIDENCE_LEVEL interval. Failed distribution fits are

skipped; at least MIN_SUCCESSFUL_BOOTSTRAP_FRACTION of the requested fits must

succeed for every month/dataset/distribution combination.

Threshold options

-----------------

"storm_hans": use the August 2023 Storm Hans monthly-maximum value.

"monthly_record_without_hans": use the largest calendar-month record in the

configured observation period after excluding August 2023 from the August

record calculation. Other months in 2023, including May 2023, remain eligible.

"""

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

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

MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"

MODEL_VARIABLE = "tp24"

MODEL_SAMPLING_GROUP = "full"  # "full", "split1", "split2", ...

FIRST_INPUT_LEAD = 16

LAST_INPUT_LEAD = 46

NUMBER_OF_LEAD_BINS = 2

SUBSAMPLE_MODEL_TO_REFERENCE_LENGTH = False


AEP_YEARS = 10

INCLUDE_STORM_HANS_IN_FIT = True

BOOTSTRAP_METHOD = "nonparametric"  # "nonparametric" or "parametric"

NUMBER_OF_BOOTSTRAPS = 50

CONFIDENCE_LEVEL = 0.95

MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90

RANDOM_SEED = 42

REFERENCE_FILENAME_OVERRIDE = None

MODEL_FILENAME_OVERRIDE = None

WRITE_TO_FILE = False
SHOW_FIGURE = True
FIGURE_DPI = 300
FIGURE_SIZE = (10.0, 4.6)
PANEL_WSPACE = 0.22

PANEL_SETTINGS = {
    "storm_hans": {
        "title": "Storm Hans threshold",
        "ylim": (0, 10),
        "yticks": None,
    },
    "monthly_record_without_hans": {
        "title": "Monthly records without Storm Hans",
        "ylim": (0, 100),
        "yticks": None,
    },
}

SHOW_GRID = True
GRID_ALPHA = 0.35

# =============================================================================

# Constants

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

METHODS = ["GEV", "Gumbel", "GenEx"]

METHOD_COLORS = {"GEV": "tab:pink", "Gumbel": "tab:green", "GenEx": "tab:purple"}

METHOD_OFFSETS = {"GEV": -0.18, "Gumbel": 0.0, "GenEx": 0.18}

INTERVAL_LINEWIDTH = 1.4

AXIS_LABELSIZE = 11

TICK_LABELSIZE = 11

TITLE_FONTSIZE = 12

LEGEND_FONTSIZE = 10


def validate_settings():

    """Validate user-configurable settings."""

    if REFERENCE_DATASET not in {"senorge", "era5"}:

        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")

    if MODEL_DATA_METHOD not in {"raw", "mm_1step", "mm_2step", "q", "ld", "doy", "q_doy"}:

        raise ValueError("Unsupported MODEL_DATA_METHOD.")

    required_panels = {"storm_hans", "monthly_record_without_hans"}

    if set(PANEL_SETTINGS) != required_panels:
        raise ValueError(f"PANEL_SETTINGS must contain exactly {sorted(required_panels)}.")

    if BOOTSTRAP_METHOD not in {"nonparametric", "parametric"}:

        raise ValueError("BOOTSTRAP_METHOD must be 'nonparametric' or 'parametric'.")

    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:

        raise ValueError("OBSERVATION_YEARS must be increasing.")

    if AEP_YEARS < 1:

        raise ValueError("AEP_YEARS must be at least 1.")

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


def get_model_label():

    """Return the model display label."""

    return "Model" if MODEL_DATA_METHOD == "raw" else "Model BC"


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

    directory = config.dirs[directory_key]

    return Path(directory) / filename


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

        return genextreme.rvs(

            shape, loc=location, scale=scale, size=sample_size, random_state=rng

        )

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


def combine_monthly_probabilities(probabilities, years=AEP_YEARS):

    """Return P(at least one exceedance in any month over the selected years)."""

    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)

    annual_probability = 1.0 - np.prod(1.0 - probabilities)

    return float(1.0 - (1.0 - annual_probability) ** years)


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

    """Fit reference and model distributions independently for all 12 months."""

    reference_months, thresholds = read_all_reference_months()

    analyses = {}

    for month_index, month in enumerate(ALL_MONTHS):

        model_values = read_model_month(month)

        model_values = subsample_model_values(

            model_values, reference_months[month]["fit_values"].size,

            RANDOM_SEED + 100 * month_index,

        )

        samples = {

            "reference": reference_months[month]["fit_values"],

            "model": model_values,

        }

        for group_index, (group, values) in enumerate(samples.items()):

            for method_index, method in enumerate(METHODS):

                seed = (

                    RANDOM_SEED + 10_000 * month_index + 1_000 * group_index + method_index

                )

                analyses[(month, group, method)] = bootstrap_distribution(

                    values, method, seed, progress

                )

    return thresholds, analyses


def calculate_threshold_probabilities(monthly_thresholds, analyses):

    """Combine monthly fitted and bootstrap probabilities for one threshold set."""

    results = {}

    for group in ["reference", "model"]:

        for method in METHODS:

            fitted_monthly = []

            bootstrap_monthly = []

            for month in ALL_MONTHS:

                fit = analyses[(month, group, method)]

                threshold = monthly_thresholds[month]["value"]

                fitted_monthly.append(

                    exceedance_probability(threshold, fit["parameters"], method)

                )

                bootstrap_monthly.append([

                    exceedance_probability(threshold, parameters, method)

                    for parameters in fit["bootstrap_parameters"]

                ])

            fitted = combine_monthly_probabilities(fitted_monthly)

            n_bootstraps = min(map(len, bootstrap_monthly))

            bootstrap = np.array([

                combine_monthly_probabilities([values[i] for values in bootstrap_monthly])

                for i in range(n_bootstraps)

            ])

            results[(group, method)] = {

                "probability": 100.0 * fitted,

                "metric_samples": 100.0 * bootstrap,

            }

    return results


def calculate_all_probabilities(thresholds, analyses):

    """Calculate results for both threshold definitions."""

    return {

        threshold_type: calculate_threshold_probabilities(monthly_thresholds, analyses)

        for threshold_type, monthly_thresholds in thresholds.items()

    }

def threshold_label(threshold_type, thresholds):

    """Return a descriptive threshold label for summaries."""

    if threshold_type == "storm_hans":

        return f"Storm Hans, August 2023 ({thresholds[AUGUST]['value']:.1f} mm)"

    return "Calendar-month records; August 2023 excluded from the August record"


def draw_probability_panel(axis, results, settings, panel_label):

    """Draw one threshold panel."""

    alpha = 1.0 - CONFIDENCE_LEVEL

    for group_index, group in enumerate(["reference", "model"]):

        for method in METHODS:

            samples = results[(group, method)]["metric_samples"]

            lower, q1, median, q3, upper = np.percentile(

                samples, [100 * alpha / 2, 25, 50, 75, 100 * (1 - alpha / 2)]

            )

            position = group_index + METHOD_OFFSETS[method]

            color = METHOD_COLORS[method]

            axis.bxp(

                [{

                    "label": "", "q1": q1, "med": median, "q3": q3,

                    "whislo": lower, "whishi": upper, "fliers": [],

                }],

                positions=[position], widths=0.14, showfliers=False, manage_ticks=False,

                boxprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},

                whiskerprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},

                capprops={"color": color, "linewidth": INTERVAL_LINEWIDTH},

                medianprops={"color": color, "linewidth": 1.8},

            )

    axis.set_xlim(-0.55, 1.55)

    axis.set_ylim(*settings["ylim"])

    if settings["yticks"] is not None:

        axis.set_yticks(settings["yticks"])

    axis.set_xticks([0, 1])

    axis.set_xticklabels([get_reference_name(), get_model_label()])

    axis.set_title(settings["title"], loc="left", fontsize=TITLE_FONTSIZE, fontweight="normal")

    axis.text(

        0.0, 1.03, panel_label, transform=axis.transAxes,

        fontsize=TITLE_FONTSIZE, fontweight="normal", ha="left", va="bottom",

    )

    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE, direction="out")

    axis.spines["top"].set_visible(False)

    axis.spines["right"].set_visible(False)

    if SHOW_GRID:

        axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=GRID_ALPHA)

        axis.set_axisbelow(True)


def plot_all_month_probabilities(results):

    """Plot Storm Hans and monthly-record exceedance probabilities side by side."""

    figure, axes = plt.subplots(

        1, 2, figsize=FIGURE_SIZE,

        gridspec_kw={"wspace": PANEL_WSPACE},

        constrained_layout=False,

    )

    panel_types = ["storm_hans", "monthly_record_without_hans"]

    for axis, threshold_type, panel_label in zip(axes, panel_types, ["a)", "b)"]):

        draw_probability_panel(

            axis, results[threshold_type], PANEL_SETTINGS[threshold_type], panel_label

        )

    axes[0].set_ylabel(

        f"Probability of at least one exceedance in {AEP_YEARS} years [%]",

        fontsize=AXIS_LABELSIZE,

    )

    handles = [

        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2, label=method)

        for method in METHODS

    ]

    figure.legend(

        handles=handles, loc="lower center", ncol=len(METHODS), frameon=False,

        fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, -0.01),

    )

    figure.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.18)

    if WRITE_TO_FILE:

        filename = Path(config.dirs["fig"]) / (

            f"all-month-{AEP_YEARS}yr-exceedance-two-panels-"

            f"{MODEL_DATA_METHOD}-{REFERENCE_DATASET}.png"

        )

        filename.parent.mkdir(parents=True, exist_ok=True)

        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")

        print("Wrote:", filename)

    if SHOW_FIGURE:

        plt.show()

    plt.close(figure)


def print_summary(results, thresholds):

    """Print thresholds and fitted N-year exceedance probabilities for both panels."""

    print("Selected settings")

    print("-----------------")

    print(

        f"Reference:        {get_reference_name()} "

        f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"

    )

    print(f"Reference file:   {make_reference_filename()}")

    print(f"Model file:       {make_model_filename()}")

    print(f"Horizon:          {AEP_YEARS} years")

    print(f"Bootstrap:        {BOOTSTRAP_METHOD}, n={NUMBER_OF_BOOTSTRAPS}")

    print(f"Include Hans fit: {INCLUDE_STORM_HANS_IN_FIT}")

    print("Assumption: exceedances are independent between months and years.")

    for threshold_type in ["storm_hans", "monthly_record_without_hans"]:

        print()

        print(PANEL_SETTINGS[threshold_type]["title"])

        print("-" * len(PANEL_SETTINGS[threshold_type]["title"]))

        print(f"Threshold: {threshold_label(threshold_type, thresholds[threshold_type])}")

        if threshold_type == "monthly_record_without_hans":

            for month in ALL_MONTHS:

                threshold = thresholds[threshold_type][month]

                print(

                    f"  {MONTH_NAMES[month - 1]:9s} "

                    f"{threshold['value']:.3f} mm ({threshold['year']})"

                )

        for group in ["reference", "model"]:

            print()

            print(get_reference_name() if group == "reference" else get_model_label())

            for method in METHODS:

                probability = results[threshold_type][(group, method)]["probability"]

                print(f"  {method}: {probability:.3f}%")

def main():

    """Run the two-panel all-calendar-month exceedance analysis."""

    validate_settings()

    total = 12 * 2 * len(METHODS) * NUMBER_OF_BOOTSTRAPS

    progress = ProgressTracker(total)

    print("Running bootstrap fits for all 12 months...")

    thresholds, analyses = build_all_month_analysis(progress)

    results = calculate_all_probabilities(thresholds, analyses)

    print_summary(results, thresholds)

    plot_all_month_probabilities(results)


if __name__ == "__main__":

    main()
