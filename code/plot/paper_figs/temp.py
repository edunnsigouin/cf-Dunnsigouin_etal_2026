"""
Compare the M-year exceedance probabilities of Storm Hans and the
1957-2022 August observational record across:

1. observational reference data,
2. raw UNSEEN model data, and
3. bias-corrected UNSEEN model data.

For each group, GEV, Gumbel, and GenEx distributions are fitted, and a
fourth empirical estimate is calculated directly from the event rank.

Threshold logic
---------------
Reference:
    SeNorge fits use the SeNorge August 2023 Storm Hans threshold.
    ERA5 fits use the ERA5 August 2023 Storm Hans threshold.

Model raw:
    The same raw model sample is evaluated twice:
    first using the SeNorge threshold and then using the ERA5 threshold.

Model BC:
    SeNorge-bias-corrected model data use the SeNorge threshold.
    ERA5-bias-corrected model data use the ERA5 threshold.

For every dataset/method position, Storm Hans is shown with a filled marker
and the 1957-2022 August observational record is shown with the same marker
but no face color. Vertical error bars are percentile confidence intervals
from a parametric bootstrap: simulate from the fitted distribution, refit the
same distribution, and recalculate the probability of exceeding the fixed
threshold.

Colors identify the threshold / bias-correction reference:
    SeNorge = tab:red
    ERA5    = tab:blue

Markers identify the calculation method:
    GEV       = circle
    Gumbel    = square
    GenEx     = diamond
    Empirical = triangle

SciPy's genextreme shape c has the opposite sign from conventional GEV xi:
xi = -c.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User inputs
# =============================================================================

# Distribution month fitted to the observational monthly maxima.
# 1 = January, ..., 8 = August, ..., 12 = December.
SELECTED_MONTH = 8

# Storm Hans is always read from August 2023.
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

# Historical August record used as a second event threshold. The maximum is
# found separately in SeNorge and ERA5 over this inclusive period.
RECORD_START_YEAR = 1957
RECORD_END_YEAR = 2022

# Probability of at least one exceedance in this many independent years.
AEP_YEARS = 1

CATCHMENT = "regine_drammen"
X_DAYS = 2

OBSERVATION_YEARS = [
    "1957",
    "2023",
]

# Read 2023 to obtain Storm Hans, but optionally omit it from fitting.
EXCLUDE_2023_FROM_FIT = True

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

REFERENCE_DATASETS = [
    "senorge",
    "era5",
]

METHODS = [
    "GEV",
    "Gumbel",
    "GenEx",
    "Empirical",
]

# Parametric distributions only. The empirical method is calculated directly
# from the event rank and is not fitted or parametrically bootstrapped.
DISTRIBUTIONS = [
    "GEV",
    "Gumbel",
    "GenEx",
]


# -----------------------------------------------------------------------------
# Parametric bootstrap
# -----------------------------------------------------------------------------

NUMBER_OF_BOOTSTRAPS = 10
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
MINIMUM_BOOTSTRAP_SUCCESS_FRACTION = 0.90


# -----------------------------------------------------------------------------
# Raw and bias-corrected UNSEEN inputs retained for later use
# -----------------------------------------------------------------------------

# Model data are required for the full reference/raw/bias-corrected figure.
READ_MODEL_DATA = True

MODEL_VARIABLE = "tp24"

FORECAST_DATE_RANGE = [
    "2020-01-02",
    "2022-12-29",
]

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2
MODEL_SAMPLING_GROUP = "full"


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

FIG_WIDTH_IN = 13.0
FIG_HEIGHT_IN = 6.0
FIGURE_DPI = 300

POINT_SIZE = 55
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPSIZE = 4
GROUP_SEPARATOR_LINEWIDTH = 0.8

# Horizontal x-axis spacing controls.
#
# ESTIMATE_SPACING controls the distance between every adjacent estimate
# within a group. Each group contains eight estimates:
#   SeNorge: GEV, Gumbel, GenEx, Empirical
#   ERA5:    GEV, Gumbel, GenEx, Empirical
#
# GROUP_SPACING controls the empty horizontal gap between the final estimate
# of one group and the first estimate of the next group.
ESTIMATE_SPACING = 0.55
GROUP_SPACING = 1.40

METHOD_MARKERS = {
    "GEV": "o",
    "Gumbel": "s",
    "GenEx": "D",
    "Empirical": "^",
}

SENORGE_COLOR = "tab:red"
ERA5_COLOR = "tab:blue"

AXIS_LABELSIZE = 11
TICK_LABELSIZE = 10
TITLE_FONTSIZE = 11
ANNOTATION_FONTSIZE = 9

# Logarithmic AEP axis limits in percent. Leave either as None to determine
# that limit automatically from the plotted estimates and confidence intervals.
YMIN_PERCENT = 0.000001
YMAX_PERCENT = 100

# Multiplicative padding applied when an automatic log-axis limit is used.
YMIN_PADDING_FACTOR = 0.5
YMAX_PADDING_FACTOR = 2.0

SHOW_POINT_LABELS = False
WRITE_TO_FILE = False


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


# =============================================================================
# Validation and labels
# =============================================================================

def validate_user_settings():
    """Validate user-defined settings."""

    if SELECTED_MONTH not in range(1, 13):
        raise ValueError(
            "SELECTED_MONTH must be an integer from 1 to 12."
        )

    if STORM_HANS_MONTH not in range(1, 13):
        raise ValueError(
            "STORM_HANS_MONTH must be an integer from 1 to 12."
        )

    if RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError(
            "RECORD_END_YEAR must be greater than or equal to "
            "RECORD_START_YEAR."
        )

    if not isinstance(AEP_YEARS, int) or AEP_YEARS < 1:
        raise ValueError(
            "AEP_YEARS must be a positive integer."
        )

    if X_DAYS < 1:
        raise ValueError(
            "X_DAYS must be at least 1."
        )

    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError(
            "NUMBER_OF_BOOTSTRAPS must be at least 1."
        )

    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError(
            "CONFIDENCE_LEVEL must lie between 0 and 1."
        )

    if not 0 < MINIMUM_BOOTSTRAP_SUCCESS_FRACTION <= 1:
        raise ValueError(
            "MINIMUM_BOOTSTRAP_SUCCESS_FRACTION must lie in (0, 1]."
        )

    if YMIN_PERCENT is not None and YMIN_PERCENT <= 0:
        raise ValueError(
            "YMIN_PERCENT must be greater than zero on a logarithmic axis."
        )

    if YMAX_PERCENT is not None and YMAX_PERCENT <= 0:
        raise ValueError(
            "YMAX_PERCENT must be greater than zero on a logarithmic axis."
        )

    if (
        YMIN_PERCENT is not None
        and YMAX_PERCENT is not None
        and YMAX_PERCENT <= YMIN_PERCENT
    ):
        raise ValueError(
            "YMAX_PERCENT must exceed YMIN_PERCENT."
        )

    if YMIN_PADDING_FACTOR <= 0 or YMAX_PADDING_FACTOR <= 0:
        raise ValueError(
            "YMIN_PADDING_FACTOR and YMAX_PADDING_FACTOR must be positive."
        )

    if ESTIMATE_SPACING <= 0:
        raise ValueError(
            "ESTIMATE_SPACING must be greater than zero."
        )

    if GROUP_SPACING <= 0:
        raise ValueError(
            "GROUP_SPACING must be greater than zero."
        )

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1

    if FIRST_INPUT_LEAD > LAST_INPUT_LEAD:
        raise ValueError(
            "FIRST_INPUT_LEAD must not exceed LAST_INPUT_LEAD."
        )

    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError(
            "X_DAYS is too large for the requested lead window."
        )

    number_of_usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1

    if (
        not isinstance(NUMBER_OF_LEAD_BINS, int)
        or NUMBER_OF_LEAD_BINS < 1
        or NUMBER_OF_LEAD_BINS > number_of_usable_leads
    ):
        raise ValueError(
            "NUMBER_OF_LEAD_BINS must be valid for the usable lead range."
        )

    valid_groups = {"full"}
    valid_groups.update(
        {
            f"split{number}"
            for number in range(1, NUMBER_OF_LEAD_BINS + 1)
        }
    )

    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(
            f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}."
        )


def get_reference_name(dataset):
    """Return publication-style dataset name."""

    return {
        "senorge": "SeNorge",
        "era5": "ERA5",
    }[dataset]


def get_reference_variable(dataset):
    """Return observational variable name."""

    return {
        "senorge": SENORGE_VARIABLE,
        "era5": ERA5_VARIABLE,
    }[dataset]


def get_reference_color(dataset):
    """Return plot color for a dataset."""

    return {
        "senorge": SENORGE_COLOR,
        "era5": ERA5_COLOR,
    }[dataset]


# =============================================================================
# Filenames
# =============================================================================

def make_reference_filename(dataset):
    """Construct an observational filename."""

    if dataset == "senorge":
        filename = (
            f"distribution_monthly_extremes_"
            f"{SENORGE_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_senorge_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
        )

        return os.path.join(
            config.dirs["senorge_processed"],
            filename,
        )

    if dataset == "era5":
        filename = (
            f"distribution_monthly_extremes_"
            f"{ERA5_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_era5_{ERA5_GRID}_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
        )

        return os.path.join(
            config.dirs["era5_processed"],
            filename,
        )

    raise ValueError(
        f"Unsupported reference dataset: {dataset}"
    )


def make_figure_filename():
    """Construct output figure filename."""

    month_name = MONTH_NAMES[SELECTED_MONTH - 1].lower()

    filename = (
        f"storm-hans-and-record-aep-{AEP_YEARS}year-"
        f"{month_name}-reference-raw-bc-"
        f"senorge-era5-gev-gumbel-genex-empirical-"
        f"{CATCHMENT}-{X_DAYS}dayacc-log-y.png"
    )

    return os.path.join(
        config.dirs["fig"],
        filename,
    )


# =============================================================================
# Optional UNSEEN support
# =============================================================================

def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """Split usable ending leads into approximately equal bins."""

    number_of_leads = last_lead - first_lead + 1
    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder)
        for index in range(number_of_bins)
    ]

    lead_bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        lead_bins.append((current_start, current_end))
        current_start = current_end + 1

    return lead_bins


def build_lead_bins():
    """Return configured lead bins."""

    return split_usable_accumulated_leads(
        first_lead=FIRST_INPUT_LEAD + X_DAYS - 1,
        last_lead=LAST_INPUT_LEAD,
        number_of_bins=NUMBER_OF_LEAD_BINS,
    )


def get_full_lead_range():
    """Return complete usable lead range."""

    return (
        FIRST_INPUT_LEAD + X_DAYS - 1,
        LAST_INPUT_LEAD,
    )


def get_selected_model_lead_range():
    """Return selected model lead range."""

    if MODEL_SAMPLING_GROUP == "full":
        return get_full_lead_range()

    split_number = int(
        MODEL_SAMPLING_GROUP.replace("split", "")
    )

    return build_lead_bins()[split_number - 1]


def get_raw_model_variable():
    """Return raw model variable."""

    lead_start, lead_end = get_selected_model_lead_range()

    return f"max_value_lead{lead_start}_{lead_end}"


def get_bias_corrected_model_variable(reference_dataset):
    """Return bias-corrected model variable."""

    return (
        f"{get_raw_model_variable()}_bc_"
        f"{reference_dataset}"
    )


def lead_split_filename_label():
    """Return lead-bin filename label."""

    full_start, full_end = get_full_lead_range()

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in build_lead_bins()
    )

    return (
        f"lead{full_start}-{full_end}_"
        f"split{NUMBER_OF_LEAD_BINS}_"
        f"{split_text}"
    )


def make_raw_model_filename():
    """Construct raw UNSEEN filename."""

    filename = (
        f"unseen_sample_monthly_catchment_precipitation_extremes_"
        f"{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_"
        f"{lead_split_filename_label()}_"
        f"forecast_hindcast_"
        f"{FORECAST_DATE_RANGE[0]}_"
        f"{FORECAST_DATE_RANGE[1]}.nc"
    )

    return os.path.join(
        config.dirs["s2s_processed"],
        filename,
    )


def make_bias_corrected_model_filename(reference_dataset):
    """Construct bias-corrected UNSEEN filename."""

    raw_filename = make_raw_model_filename()
    stem, extension = os.path.splitext(raw_filename)

    return (
        f"{stem}_bc_{reference_dataset}"
        f"{extension}"
    )


def read_model_month(
    filename,
    variable,
    month,
    dataset_name,
):
    """Read finite model values for one month."""

    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {dataset_name}. "
                f"Available variables are: {list(ds.data_vars)}"
            )

        values = np.asarray(
            ds[variable]
            .sel(month_of_year=month)
            .values,
            dtype=float,
        )

    values = values[np.isfinite(values)]

    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite values were found in {dataset_name}."
        )

    return values


def read_model_data():
    """Read raw and both bias-corrected UNSEEN samples."""

    if not READ_MODEL_DATA:
        raise ValueError(
            "READ_MODEL_DATA must be True for this comparison."
        )

    raw_filename = make_raw_model_filename()

    model_data = {
        "raw": read_model_month(
            filename=raw_filename,
            variable=get_raw_model_variable(),
            month=SELECTED_MONTH,
            dataset_name="raw UNSEEN dataset",
        ),
        "bias_corrected": {},
        "filenames": {
            "raw": raw_filename,
            "bias_corrected": {},
        },
    }

    for reference_dataset in REFERENCE_DATASETS:
        bc_filename = make_bias_corrected_model_filename(
            reference_dataset
        )

        model_data["bias_corrected"][reference_dataset] = (
            read_model_month(
                filename=bc_filename,
                variable=get_bias_corrected_model_variable(
                    reference_dataset
                ),
                month=SELECTED_MONTH,
                dataset_name=(
                    f"bias-corrected UNSEEN dataset "
                    f"({get_reference_name(reference_dataset)} reference)"
                ),
            )
        )

        model_data["filenames"]["bias_corrected"][
            reference_dataset
        ] = bc_filename

    return model_data


# =============================================================================
# Observational data
# =============================================================================

def read_reference_data(dataset):
    """Read fitted sample and dataset-specific Storm Hans value."""

    filename = make_reference_filename(dataset)
    variable = get_reference_variable(dataset)

    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in "
                f"{get_reference_name(dataset)}."
            )

        selected_month_data = (
            ds[variable]
            .sel(
                year=slice(
                    int(OBSERVATION_YEARS[0]),
                    int(OBSERVATION_YEARS[1]),
                ),
                month=SELECTED_MONTH,
            )
            .load()
        )

        storm_hans_value = float(
            ds[variable]
            .sel(
                year=STORM_HANS_YEAR,
                month=STORM_HANS_MONTH,
            )
            .load()
            .values
        )

        record_data = (
            ds[variable]
            .sel(
                year=slice(
                    RECORD_START_YEAR,
                    RECORD_END_YEAR,
                ),
                month=STORM_HANS_MONTH,
            )
            .load()
        )

    years = np.asarray(
        selected_month_data["year"].values
    )

    values = np.asarray(
        selected_month_data.values,
        dtype=float,
    )

    finite = np.isfinite(values)
    years = years[finite]
    values = values[finite]

    record_years = np.asarray(
        record_data["year"].values
    )

    record_values = np.asarray(
        record_data.values,
        dtype=float,
    )

    record_finite = np.isfinite(
        record_values
    )

    if not np.any(record_finite):
        raise ValueError(
            f"No finite {get_reference_name(dataset)} August values were "
            f"found from {RECORD_START_YEAR} to {RECORD_END_YEAR}."
        )

    finite_record_values = record_values[
        record_finite
    ]
    finite_record_years = record_years[
        record_finite
    ]

    record_index = int(
        np.argmax(
            finite_record_values
        )
    )

    record_value = float(
        finite_record_values[
            record_index
        ]
    )

    record_year = int(
        finite_record_years[
            record_index
        ]
    )

    if not np.isfinite(storm_hans_value):
        raise ValueError(
            f"The {get_reference_name(dataset)} Storm Hans value is not finite."
        )

    fit_mask = np.ones(
        years.size,
        dtype=bool,
    )

    if EXCLUDE_2023_FROM_FIT:
        fit_mask &= years != 2023

    fit_years = years[fit_mask]
    fit_values = values[fit_mask]

    if fit_values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {get_reference_name(dataset)} values "
            f"remain in the fitted sample."
        )

    return {
        "filename": filename,
        "fit_years": fit_years,
        "fit_values": fit_values,
        "storm_hans_value": storm_hans_value,
        "record_value": record_value,
        "record_year": record_year,
    }


# =============================================================================
# Distribution fitting
# =============================================================================

def fit_gev(values):
    """Fit stationary three-parameter GEV."""

    shape_c, location, scale = genextreme.fit(values)

    if (
        not np.isfinite([shape_c, location, scale]).all()
        or scale <= 0
    ):
        raise RuntimeError(
            "The GEV fit returned invalid parameters."
        )

    return shape_c, location, scale


def fit_gumbel(values):
    """Fit stationary two-parameter Gumbel."""

    location, scale = gumbel_r.fit(values)

    if (
        not np.isfinite([location, scale]).all()
        or scale <= 0
    ):
        raise RuntimeError(
            "The Gumbel fit returned invalid parameters."
        )

    return location, scale


def genex_negative_log_likelihood(
    log_parameters,
    values,
):
    """Negative log-likelihood for two-parameter GenEx."""

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

    log_one_minus_exp = np.log(
        -np.expm1(-z)
    )

    log_pdf = (
        np.log(shape)
        - np.log(scale)
        - z
        + (shape - 1.0) * log_one_minus_exp
    )

    if not np.isfinite(log_pdf).all():
        return np.inf

    return -np.sum(log_pdf)


def fit_genex(values):
    """Fit two-parameter Generalized Exponential."""

    if np.any(values < 0):
        raise ValueError(
            "GenEx requires non-negative values."
        )

    positive_values = values[values > 0]

    if positive_values.size == 0:
        raise RuntimeError(
            "GenEx cannot be fitted without positive values."
        )

    result = minimize(
        genex_negative_log_likelihood,
        x0=np.log(
            [
                1.0,
                np.mean(positive_values),
            ]
        ),
        args=(values,),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    if not result.success:
        raise RuntimeError(
            f"GenEx fit failed: {result.message}"
        )

    shape, scale = np.exp(result.x)

    if (
        not np.isfinite([shape, scale]).all()
        or shape <= 0
        or scale <= 0
    ):
        raise RuntimeError(
            "The GenEx fit returned invalid parameters."
        )

    return shape, scale


def fit_distribution(
    values,
    distribution_name,
):
    """Fit one named distribution."""

    if distribution_name == "GEV":
        return fit_gev(values)

    if distribution_name == "Gumbel":
        return fit_gumbel(values)

    if distribution_name == "GenEx":
        return fit_genex(values)

    raise ValueError(
        f"Unsupported distribution: {distribution_name}"
    )


def calculate_event_exceedance_probability(
    event_value,
    fitted_parameters,
    distribution_name,
):
    """Calculate one-year fitted exceedance probability."""

    if distribution_name == "GEV":
        shape_c, location, scale = fitted_parameters

        probability = genextreme.sf(
            event_value,
            shape_c,
            loc=location,
            scale=scale,
        )

    elif distribution_name == "Gumbel":
        location, scale = fitted_parameters

        probability = gumbel_r.sf(
            event_value,
            loc=location,
            scale=scale,
        )

    elif distribution_name == "GenEx":
        shape, scale = fitted_parameters

        if event_value < 0:
            probability = 1.0
        else:
            cdf = (
                1.0
                - np.exp(
                    -event_value / scale
                )
            ) ** shape

            probability = 1.0 - cdf

    else:
        raise ValueError(
            f"Unsupported distribution: {distribution_name}"
        )

    if not np.isfinite(probability):
        raise RuntimeError(
            "The fitted event probability is not finite."
        )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


def calculate_m_year_aep(
    annual_exceedance_probability,
):
    """Convert annual exceedance probability to M-year AEP."""

    annual_exceedance_probability = float(
        np.clip(
            annual_exceedance_probability,
            0.0,
            1.0,
        )
    )

    return float(
        -np.expm1(
            AEP_YEARS
            * np.log1p(
                -annual_exceedance_probability
            )
        )
    )


def generate_random_sample(
    fitted_parameters,
    distribution_name,
    sample_size,
    rng,
):
    """Generate one parametric-bootstrap sample."""

    if distribution_name == "GEV":
        shape_c, location, scale = fitted_parameters

        return genextreme.rvs(
            shape_c,
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    if distribution_name == "Gumbel":
        location, scale = fitted_parameters

        return gumbel_r.rvs(
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    if distribution_name == "GenEx":
        shape, scale = fitted_parameters

        uniforms = rng.uniform(
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
            size=sample_size,
        )

        return (
            -scale
            * np.log1p(
                -np.power(
                    uniforms,
                    1.0 / shape,
                )
            )
        )

    raise ValueError(
        f"Unsupported distribution: {distribution_name}"
    )


# =============================================================================
# Bootstrap and analysis
# =============================================================================

def bootstrap_event_aep(
    sample_values,
    event_value,
    fitted_parameters,
    distribution_name,
    random_seed,
):
    """Parametric-bootstrap confidence interval for fitted event AEP."""

    rng = np.random.default_rng(random_seed)

    bootstrap_aeps = np.full(
        NUMBER_OF_BOOTSTRAPS,
        np.nan,
        dtype=float,
    )

    successful_fits = 0

    for bootstrap_number in range(NUMBER_OF_BOOTSTRAPS):
        simulated_values = generate_random_sample(
            fitted_parameters=fitted_parameters,
            distribution_name=distribution_name,
            sample_size=sample_values.size,
            rng=rng,
        )

        try:
            bootstrap_parameters = fit_distribution(
                simulated_values,
                distribution_name,
            )

            annual_probability = (
                calculate_event_exceedance_probability(
                    event_value=event_value,
                    fitted_parameters=bootstrap_parameters,
                    distribution_name=distribution_name,
                )
            )

            bootstrap_aep = calculate_m_year_aep(
                annual_probability
            )

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ):
            continue

        if not np.isfinite(bootstrap_aep):
            continue

        bootstrap_aeps[bootstrap_number] = bootstrap_aep
        successful_fits += 1

    minimum_successful_fits = int(
        np.ceil(
            MINIMUM_BOOTSTRAP_SUCCESS_FRACTION
            * NUMBER_OF_BOOTSTRAPS
        )
    )

    if successful_fits < minimum_successful_fits:
        raise RuntimeError(
            f"Only {successful_fits} of {NUMBER_OF_BOOTSTRAPS} bootstrap "
            f"fits succeeded for {distribution_name}."
        )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(
        np.nanpercentile(
            bootstrap_aeps,
            100.0 * alpha / 2.0,
        )
    )

    upper = float(
        np.nanpercentile(
            bootstrap_aeps,
            100.0 * (1.0 - alpha / 2.0),
        )
    )

    return lower, upper, successful_fits


def analyse_event_aep(
    sample_values,
    event_value,
    distribution_name,
    random_seed,
):
    """Fit one distribution and calculate event AEP with uncertainty."""

    fitted_parameters = fit_distribution(
        sample_values,
        distribution_name,
    )

    annual_probability = calculate_event_exceedance_probability(
        event_value=event_value,
        fitted_parameters=fitted_parameters,
        distribution_name=distribution_name,
    )

    return_period = (
        np.inf
        if annual_probability <= 0
        else 1.0 / annual_probability
    )

    estimate = calculate_m_year_aep(
        annual_probability
    )

    lower, upper, successful_fits = bootstrap_event_aep(
        sample_values=sample_values,
        event_value=event_value,
        fitted_parameters=fitted_parameters,
        distribution_name=distribution_name,
        random_seed=random_seed,
    )

    return {
        "fitted_parameters": fitted_parameters,
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
        "aep": estimate,
        "lower": lower,
        "upper": upper,
        "successful_bootstraps": successful_fits,
    }


def analyse_empirical_event_aep(
    sample_values,
    event_value,
):
    """
    Calculate event probability empirically from its rank.

    The event rank is defined as:

        rank = 1 + number of sample values strictly greater than the event

    and the annual exceedance probability is:

        rank / sample_size

    Therefore, an event larger than all values in a 66-value sample has rank 1,
    annual exceedance probability 1/66, and return period 66 years.

    No parametric confidence interval is assigned to the empirical estimate;
    lower and upper are set equal to the point estimate for plotting.
    """

    sample_values = np.asarray(
        sample_values,
        dtype=float,
    )

    sample_values = sample_values[
        np.isfinite(sample_values)
    ]

    if sample_values.size < 1:
        raise ValueError(
            "At least one finite value is required for the empirical method."
        )

    rank = 1 + int(
        np.sum(
            sample_values > event_value
        )
    )

    annual_probability = rank / sample_values.size
    return_period = sample_values.size / rank

    estimate = calculate_m_year_aep(
        annual_probability
    )

    return {
        "fitted_parameters": None,
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
        "aep": estimate,
        "lower": estimate,
        "upper": estimate,
        "successful_bootstraps": None,
        "empirical_rank": rank,
        "sample_size": sample_values.size,
    }


def analyse_method_event_aep(
    sample_values,
    event_value,
    method_name,
    random_seed,
):
    """Run either one fitted distribution or the empirical rank method."""

    if method_name == "Empirical":
        return analyse_empirical_event_aep(
            sample_values=sample_values,
            event_value=event_value,
        )

    return analyse_event_aep(
        sample_values=sample_values,
        event_value=event_value,
        distribution_name=method_name,
        random_seed=random_seed,
    )


# =============================================================================
# Reporting and plotting
# =============================================================================

def print_result(
    group_name,
    threshold_dataset,
    distribution_name,
    sample_values,
    event_value,
    event_label,
    analysis,
):
    """Print one fitted probability result."""

    print()
    print(
        f"{group_name} — {get_reference_name(threshold_dataset)} threshold "
        f"— {distribution_name} — {event_label}"
    )
    print(
        "-" * 72
    )
    print(
        f"Fitted month:                 "
        f"{MONTH_NAMES[SELECTED_MONTH - 1]}"
    )
    print(
        f"Sample size:                  "
        f"{sample_values.size}"
    )
    print(
        f"Event threshold:              "
        f"{event_value:.3f} mm "
        f"({get_reference_name(threshold_dataset)})"
    )

    if np.isfinite(analysis["return_period"]):
        print(
            f"Return period:                "
            f"{analysis['return_period']:.3f} years"
        )
    else:
        print(
            "Return period:                infinite"
        )

    print(
        f"{AEP_YEARS}-year AEP:                 "
        f"{100.0 * analysis['aep']:.3f}%"
    )
    print(
        f"{CONFIDENCE_LEVEL:.0%} bootstrap interval:      "
        f"{100.0 * analysis['lower']:.3f}% to "
        f"{100.0 * analysis['upper']:.3f}%"
    )
    if distribution_name == "Empirical":
        print(
            f"Empirical rank:               "
            f"{analysis['empirical_rank']}/"
            f"{analysis['sample_size']}"
        )
        print(
            "Confidence interval:          not calculated "
            "for the empirical rank estimate"
        )
    else:
        print(
            f"Successful bootstrap fits:    "
            f"{analysis['successful_bootstraps']}/"
            f"{NUMBER_OF_BOOTSTRAPS}"
        )


def build_plot_records(
    results,
):
    """Build one x-axis record containing both events at each position."""

    records = []

    group_definitions = [
        (
            "Reference",
            "reference",
        ),
        (
            "Model raw",
            "raw",
        ),
        (
            "Model BC",
            "bias_corrected",
        ),
    ]

    for group_label, group_key in group_definitions:
        for threshold_dataset in REFERENCE_DATASETS:
            for distribution_name in METHODS:
                records.append(
                    {
                        "group_label": group_label,
                        "group_key": group_key,
                        "threshold_dataset": threshold_dataset,
                        "distribution_name": distribution_name,
                        "storm_hans_analysis": results[
                            (
                                group_key,
                                threshold_dataset,
                                distribution_name,
                                "storm_hans",
                            )
                        ],
                        "record_analysis": results[
                            (
                                group_key,
                                threshold_dataset,
                                distribution_name,
                                "record",
                            )
                        ],
                    }
                )

    return records


def calculate_plot_positions(
    records,
):
    """
    Assign sequential, non-overlapping positions to every estimate.

    Every adjacent estimate within a major group is separated by
    ESTIMATE_SPACING. The empty gap between major groups is GROUP_SPACING.
    Group tick locations are placed at the midpoint of each complete group.
    """

    group_order = [
        "reference",
        "raw",
        "bias_corrected",
    ]

    positions = np.full(
        len(records),
        np.nan,
        dtype=float,
    )

    group_centers = {}
    group_bounds = {}

    next_x = 0.0

    for group_key in group_order:
        group_indices = [
            index
            for index, record in enumerate(records)
            if record["group_key"] == group_key
        ]

        if not group_indices:
            raise ValueError(
                f"No plotting records were found for group '{group_key}'."
            )

        group_positions = (
            next_x
            + ESTIMATE_SPACING
            * np.arange(
                len(group_indices),
                dtype=float,
            )
        )

        for index, position in zip(
            group_indices,
            group_positions,
        ):
            positions[index] = position

        group_start = float(
            group_positions[0]
        )
        group_end = float(
            group_positions[-1]
        )

        group_bounds[group_key] = (
            group_start,
            group_end,
        )

        group_centers[group_key] = (
            0.5
            * (
                group_start
                + group_end
            )
        )

        next_x = (
            group_end
            + GROUP_SPACING
        )

    if not np.isfinite(positions).all():
        raise RuntimeError(
            "One or more plotting positions were not assigned."
        )

    return (
        positions,
        group_centers,
        group_bounds,
    )


def plot_results(
    results,
    filename_out,
):
    """Plot Storm Hans and historical-record estimates at shared x positions."""

    records = build_plot_records(
        results
    )

    x, group_centers, group_bounds = calculate_plot_positions(
        records
    )

    plotted_bounds = []

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        )
    )

    event_definitions = [
        (
            "storm_hans_analysis",
            "Storm Hans, August 2023",
            True,
            4,
        ),
        (
            "record_analysis",
            (
                f"August record "
                f"{RECORD_START_YEAR}-{RECORD_END_YEAR}"
            ),
            False,
            3,
        ),
    ]

    for index, record in enumerate(records):
        threshold_dataset = record["threshold_dataset"]
        method_name = record["distribution_name"]

        color = get_reference_color(
            threshold_dataset
        )

        marker = METHOD_MARKERS[
            method_name
        ]

        for (
            analysis_key,
            event_label,
            filled_marker,
            zorder,
        ) in event_definitions:
            analysis = record[
                analysis_key
            ]

            estimate = (
                100.0
                * analysis["aep"]
            )

            lower = (
                100.0
                * analysis["lower"]
            )

            upper = (
                100.0
                * analysis["upper"]
            )

            lower_error = max(
                0.0,
                estimate - lower,
            )

            upper_error = max(
                0.0,
                upper - estimate,
            )

            plotted_bounds.append(
                (
                    lower,
                    upper,
                )
            )

            marker_facecolor = (
                color
                if filled_marker
                else "none"
            )

            if method_name == "Empirical":
                ax.plot(
                    x[index],
                    estimate,
                    marker=marker,
                    markersize=np.sqrt(POINT_SIZE),
                    markerfacecolor=marker_facecolor,
                    markeredgecolor=color,
                    markeredgewidth=ERRORBAR_LINEWIDTH,
                    color=color,
                    linestyle="none",
                    zorder=zorder,
                )
            else:
                ax.errorbar(
                    x[index],
                    estimate,
                    yerr=np.array(
                        [
                            [lower_error],
                            [upper_error],
                        ]
                    ),
                    fmt=marker,
                    markersize=np.sqrt(POINT_SIZE),
                    markerfacecolor=marker_facecolor,
                    markeredgecolor=color,
                    markeredgewidth=ERRORBAR_LINEWIDTH,
                    color=color,
                    ecolor=color,
                    elinewidth=ERRORBAR_LINEWIDTH,
                    capsize=ERRORBAR_CAPSIZE,
                    capthick=ERRORBAR_LINEWIDTH,
                    linestyle="none",
                    zorder=zorder,
                )

            if SHOW_POINT_LABELS:
                vertical_offset = (
                    6
                    if filled_marker
                    else -8
                )

                vertical_alignment = (
                    "bottom"
                    if filled_marker
                    else "top"
                )

                ax.annotate(
                    f"{estimate:.2f}%",
                    xy=(
                        x[index],
                        estimate,
                    ),
                    xytext=(
                        0,
                        vertical_offset,
                    ),
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=ANNOTATION_FONTSIZE,
                )

    # Draw separators at the midpoint of the user-defined gaps.
    group_boundaries = [
        0.5
        * (
            group_bounds["reference"][1]
            + group_bounds["raw"][0]
        ),
        0.5
        * (
            group_bounds["raw"][1]
            + group_bounds["bias_corrected"][0]
        ),
    ]

    for boundary in group_boundaries:
        ax.axvline(
            boundary,
            color="0.78",
            linewidth=GROUP_SEPARATOR_LINEWIDTH,
            zorder=0,
        )

    ax.set_xticks(
        [
            group_centers["reference"],
            group_centers["raw"],
            group_centers["bias_corrected"],
        ]
    )

    ax.set_xticklabels(
        [
            "Reference",
            "Model raw",
            "Model BC",
        ],
        fontsize=TICK_LABELSIZE,
    )

    ax.set_xlim(
        x.min() - ESTIMATE_SPACING,
        x.max() + ESTIMATE_SPACING,
    )

    ax.set_ylabel(
        (
            f"Probability of at least one exceedance "
            f"in {AEP_YEARS} years [%]"
        ),
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_title(
        (
            f"August 2023 Storm Hans and "
            f"{RECORD_START_YEAR}-{RECORD_END_YEAR} August record"
        ),
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=8,
    )

    ax.set_yscale(
        "log"
    )

    lower_bounds = np.asarray(
        [
            lower
            for lower, upper in plotted_bounds
        ],
        dtype=float,
    )

    upper_bounds = np.asarray(
        [
            upper
            for lower, upper in plotted_bounds
        ],
        dtype=float,
    )

    positive_lower_bounds = lower_bounds[
        np.isfinite(lower_bounds)
        & (lower_bounds > 0)
    ]

    positive_upper_bounds = upper_bounds[
        np.isfinite(upper_bounds)
        & (upper_bounds > 0)
    ]

    if positive_lower_bounds.size == 0:
        raise ValueError(
            "No positive lower confidence limits are available "
            "for the logarithmic y-axis."
        )

    if positive_upper_bounds.size == 0:
        raise ValueError(
            "No positive upper confidence limits are available "
            "for the logarithmic y-axis."
        )

    automatic_ymin = (
        YMIN_PADDING_FACTOR
        * np.min(
            positive_lower_bounds
        )
    )

    automatic_ymax = (
        YMAX_PADDING_FACTOR
        * np.max(
            positive_upper_bounds
        )
    )

    plot_ymin = (
        automatic_ymin
        if YMIN_PERCENT is None
        else YMIN_PERCENT
    )

    plot_ymax = (
        automatic_ymax
        if YMAX_PERCENT is None
        else YMAX_PERCENT
    )

    if plot_ymax <= plot_ymin:
        raise ValueError(
            "The final logarithmic y-axis maximum must exceed its minimum."
        )

    ax.set_ylim(
        bottom=plot_ymin,
        top=plot_ymax,
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, position: (
                ""
                if y <= 0
                else f"{y:g}%"
            )
        )
    )

    ax.tick_params(
        axis="y",
        labelsize=TICK_LABELSIZE,
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )
    ax.spines["left"].set_linewidth(
        0.8
    )
    ax.spines["bottom"].set_linewidth(
        0.8
    )

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=get_reference_color(
                dataset
            ),
            markerfacecolor=get_reference_color(
                dataset
            ),
            markeredgecolor=get_reference_color(
                dataset
            ),
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label=(
                f"{get_reference_name(dataset)} threshold / "
                f"bias correction"
            ),
        )
        for dataset in REFERENCE_DATASETS
    ]

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[
                method_name
            ],
            linestyle="none",
            color="0.25",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label=method_name,
        )
        for method_name in METHODS
    ]

    event_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="0.25",
            markerfacecolor="0.25",
            markeredgecolor="0.25",
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label="Storm Hans, August 2023",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="0.25",
            markerfacecolor="none",
            markeredgecolor="0.25",
            markeredgewidth=ERRORBAR_LINEWIDTH,
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label=(
                f"August record "
                f"{RECORD_START_YEAR}-{RECORD_END_YEAR}"
            ),
        ),
    ]

    legend_handles = (
        color_handles
        + method_handles
        + event_handles
    )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=TICK_LABELSIZE,
        ncol=4,
        columnspacing=1.6,
        handletextpad=0.6,
        labelspacing=0.5,
    )

    fig.tight_layout()

    if WRITE_TO_FILE:
        fig.savefig(
            filename_out,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )

        print()
        print(
            "Wrote:",
            filename_out,
        )

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Storm Hans and historical-record calculations for all samples."""

    validate_user_settings()

    if SELECTED_MONTH != STORM_HANS_MONTH:
        print()
        print(
            "Warning: SELECTED_MONTH differs from STORM_HANS_MONTH. "
            "Both August event thresholds will be evaluated against the "
            "selected calendar-month distributions."
        )

    reference_data = {}

    for dataset in REFERENCE_DATASETS:
        reference_data[dataset] = read_reference_data(
            dataset
        )

        print()
        print(
            f"Reading {get_reference_name(dataset)}:"
        )
        print(
            reference_data[dataset]["filename"]
        )
        print(
            f"  Storm Hans August {STORM_HANS_YEAR}: "
            f"{reference_data[dataset]['storm_hans_value']:.3f} mm"
        )
        print(
            f"  August record {RECORD_START_YEAR}-{RECORD_END_YEAR}: "
            f"{reference_data[dataset]['record_value']:.3f} mm "
            f"({reference_data[dataset]['record_year']})"
        )

    model_data = read_model_data()

    print()
    print(
        "Reading raw UNSEEN:"
    )
    print(
        model_data["filenames"]["raw"]
    )

    for dataset in REFERENCE_DATASETS:
        print()
        print(
            f"Reading {get_reference_name(dataset)}-bias-corrected UNSEEN:"
        )
        print(
            model_data["filenames"]["bias_corrected"][dataset]
        )

    event_thresholds = {
        dataset: {
            "storm_hans": reference_data[dataset]["storm_hans_value"],
            "record": reference_data[dataset]["record_value"],
        }
        for dataset in REFERENCE_DATASETS
    }

    event_labels = {
        dataset: {
            "storm_hans": (
                f"Storm Hans, August {STORM_HANS_YEAR}"
            ),
            "record": (
                f"August record {RECORD_START_YEAR}-{RECORD_END_YEAR} "
                f"({reference_data[dataset]['record_year']})"
            ),
        }
        for dataset in REFERENCE_DATASETS
    }

    results = {}
    seed_counter = 0

    def calculate_group(
        group_key,
        group_label,
        threshold_dataset,
        sample_values,
    ):
        """Calculate both events for every method in one sample."""

        nonlocal seed_counter

        for method_name in METHODS:
            for event_key in [
                "storm_hans",
                "record",
            ]:
                event_value = event_thresholds[
                    threshold_dataset
                ][
                    event_key
                ]

                analysis = analyse_method_event_aep(
                    sample_values=sample_values,
                    event_value=event_value,
                    method_name=method_name,
                    random_seed=RANDOM_SEED + seed_counter,
                )

                results[
                    (
                        group_key,
                        threshold_dataset,
                        method_name,
                        event_key,
                    )
                ] = analysis

                print_result(
                    group_name=group_label,
                    threshold_dataset=threshold_dataset,
                    distribution_name=method_name,
                    sample_values=sample_values,
                    event_value=event_value,
                    event_label=event_labels[
                        threshold_dataset
                    ][
                        event_key
                    ],
                    analysis=analysis,
                )

                if method_name != "Empirical":
                    seed_counter += 1

    # 1. Reference samples and their matching event thresholds.
    for dataset in REFERENCE_DATASETS:
        calculate_group(
            group_key="reference",
            group_label="Reference",
            threshold_dataset=dataset,
            sample_values=reference_data[dataset]["fit_values"],
        )

    # 2. One raw model sample evaluated against both reference thresholds.
    for threshold_dataset in REFERENCE_DATASETS:
        calculate_group(
            group_key="raw",
            group_label="Model raw",
            threshold_dataset=threshold_dataset,
            sample_values=model_data["raw"],
        )

    # 3. Each bias-corrected sample uses events from its correction reference.
    for reference_dataset in REFERENCE_DATASETS:
        calculate_group(
            group_key="bias_corrected",
            group_label="Model BC",
            threshold_dataset=reference_dataset,
            sample_values=model_data["bias_corrected"][
                reference_dataset
            ],
        )

    plot_results(
        results=results,
        filename_out=make_figure_filename(),
    )


if __name__ == "__main__":
    main()
