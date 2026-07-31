"""Compare exceedance probabilities for two event thresholds across observational
reference data and one selected UNSEEN model form (raw or bias-corrected).

Two panels are plotted:
1. Storm Hans (August 2023) threshold.
2. The historical record threshold for SELECTED_MONTH over RECORD_START_YEAR to
   RECORD_END_YEAR.

Each panel has four x-axis columns:
    Reference - SeNorge
    Reference - ERA5
    Model - SeNorge threshold/reference
    Model - ERA5 threshold/reference

GEV, Gumbel, GenEx, and empirical estimates are plotted at the same x position
within each column. There is one point estimate per dataset/method/threshold; no
bootstrap confidence intervals are calculated or plotted.

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
# from the event rank and is not fitted.
DISTRIBUTIONS = [
    "GEV",
    "Gumbel",
    "GenEx",
]




# -----------------------------------------------------------------------------
# Raw and bias-corrected UNSEEN inputs retained for later use
# -----------------------------------------------------------------------------

# Select exactly one model form for calculation and plotting:
#     "raw" or "bias_corrected"
MODEL_DATA_MODE = "raw"

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

# Horizontal spacing between the four calculation columns.
COLUMN_SPACING = 1.0
PANEL_SPACING = 0.30

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
# that limit automatically from the plotted point estimates.
YMIN_PERCENT = 0.001
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

    if MODEL_DATA_MODE not in {"raw", "bias_corrected"}:
        raise ValueError(
            'MODEL_DATA_MODE must be either "raw" or "bias_corrected".'
        )

    if COLUMN_SPACING <= 0:
        raise ValueError(
            "COLUMN_SPACING must be greater than zero."
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
    model_label = "raw" if MODEL_DATA_MODE == "raw" else "bc"

    filename = (
        f"storm-hans-and-{month_name}-record-aep-{AEP_YEARS}year-"
        f"reference-model-{model_label}-senorge-era5-"
        f"gev-gumbel-genex-empirical-{CATCHMENT}-"
        f"{X_DAYS}dayacc-log-y.png"
    )

    return os.path.join(config.dirs["fig"], filename)


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
    """Read only the user-selected UNSEEN model form."""

    if not READ_MODEL_DATA:
        raise ValueError("READ_MODEL_DATA must be True for this comparison.")

    if MODEL_DATA_MODE == "raw":
        filename = make_raw_model_filename()
        values = read_model_month(
            filename=filename,
            variable=get_raw_model_variable(),
            month=SELECTED_MONTH,
            dataset_name="raw UNSEEN dataset",
        )
        return {
            "mode": "raw",
            "label": "Model raw",
            "samples": {dataset: values for dataset in REFERENCE_DATASETS},
            "filenames": {dataset: filename for dataset in REFERENCE_DATASETS},
        }

    samples = {}
    filenames = {}
    for reference_dataset in REFERENCE_DATASETS:
        filename = make_bias_corrected_model_filename(reference_dataset)
        samples[reference_dataset] = read_model_month(
            filename=filename,
            variable=get_bias_corrected_model_variable(reference_dataset),
            month=SELECTED_MONTH,
            dataset_name=(
                f"bias-corrected UNSEEN dataset "
                f"({get_reference_name(reference_dataset)} reference)"
            ),
        )
        filenames[reference_dataset] = filename

    return {
        "mode": "bias_corrected",
        "label": "Model BC",
        "samples": samples,
        "filenames": filenames,
    }


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
                month=SELECTED_MONTH,
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
            f"No finite {get_reference_name(dataset)} {MONTH_NAMES[SELECTED_MONTH - 1]} values were "
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


def analyse_event_aep(
    sample_values,
    event_value,
    distribution_name,
):
    """Fit one distribution and calculate a single event AEP estimate."""

    fitted_parameters = fit_distribution(sample_values, distribution_name)
    annual_probability = calculate_event_exceedance_probability(
        event_value=event_value,
        fitted_parameters=fitted_parameters,
        distribution_name=distribution_name,
    )
    return_period = np.inf if annual_probability <= 0 else 1.0 / annual_probability
    estimate = calculate_m_year_aep(annual_probability)

    return {
        "fitted_parameters": fitted_parameters,
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
        "aep": estimate,
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
        "empirical_rank": rank,
        "sample_size": sample_values.size,
    }


def analyse_method_event_aep(
    sample_values,
    event_value,
    method_name,
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
    )


# =============================================================================
# Reporting and plotting
# =============================================================================

def print_result(group_name, threshold_dataset, method_name, event_label, analysis):
    """Print only the return period and AEP for one calculation."""

    return_period = analysis["return_period"]
    return_period_text = (
        f"{return_period:.3f} years" if np.isfinite(return_period) else "infinite"
    )
    print(
        f"{event_label} | {group_name} | "
        f"{get_reference_name(threshold_dataset)} | {method_name}: "
        f"return period = {return_period_text}; "
        f"{AEP_YEARS}-year AEP = {100.0 * analysis['aep']:.6g}%"
    )


def display_aep_percent(aep):
    """Convert AEP to percent and place exact zeros at the plot minimum."""

    percent = 100.0 * float(aep)
    if percent == 0.0:
        if YMIN_PERCENT is None:
            raise ValueError(
                "YMIN_PERCENT must be set when an AEP can equal zero on a log axis."
            )
        return float(YMIN_PERCENT)
    return percent


def build_plot_columns(model_label):
    """Return the four shared x-axis calculation columns."""

    return [
        ("reference", "senorge", "Ref–SeNorge"),
        ("reference", "era5", "Ref–ERA5"),
        ("model", "senorge", f"{model_label}–SeNorge"),
        ("model", "era5", f"{model_label}–ERA5"),
    ]


def plot_results(results, model_label, filename_out):
    """Plot four methods over one another for each calculation column."""

    event_definitions = [
        ("storm_hans", "Storm Hans threshold (August 2023)"),
        (
            "calendar_record",
            f"{MONTH_NAMES[SELECTED_MONTH - 1]} record threshold "
            f"({RECORD_START_YEAR}-{RECORD_END_YEAR})",
        ),
    ]
    columns = build_plot_columns(model_label)
    x = COLUMN_SPACING * np.arange(len(columns), dtype=float)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        gridspec_kw={"hspace": PANEL_SPACING},
    )

    plotted_values = []
    for ax, (event_key, panel_title) in zip(axes, event_definitions):
        for column_index, (group_key, threshold_dataset, _) in enumerate(columns):
            color = get_reference_color(threshold_dataset)
            for method_name in METHODS:
                analysis = results[(group_key, threshold_dataset, method_name, event_key)]
                estimate = display_aep_percent(analysis["aep"])
                plotted_values.append(estimate)
                ax.plot(
                    x[column_index],
                    estimate,
                    marker=METHOD_MARKERS[method_name],
                    markersize=np.sqrt(POINT_SIZE),
                    markerfacecolor=color,
                    markeredgecolor=color,
                    linestyle="none",
                    zorder=3,
                )
                if SHOW_POINT_LABELS:
                    ax.annotate(
                        f"{estimate:.3g}%",
                        (x[column_index], estimate),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=ANNOTATION_FONTSIZE,
                    )

        ax.set_title(panel_title, fontsize=TITLE_FONTSIZE, fontweight="normal")
        ax.set_yscale("log")
        ax.set_ylabel(f"{AEP_YEARS}-year AEP [%]", fontsize=AXIS_LABELSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, position: "" if y <= 0 else f"{y:g}%")
        )

    positive_values = np.asarray([v for v in plotted_values if np.isfinite(v) and v > 0])
    if positive_values.size == 0:
        raise ValueError("No positive AEP values are available for the log axis.")
    plot_ymin = (
        YMIN_PADDING_FACTOR * positive_values.min()
        if YMIN_PERCENT is None
        else YMIN_PERCENT
    )
    plot_ymax = (
        YMAX_PADDING_FACTOR * positive_values.max()
        if YMAX_PERCENT is None
        else YMAX_PERCENT
    )
    for ax in axes:
        ax.set_ylim(plot_ymin, plot_ymax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [label for _, _, label in columns],
        fontsize=TICK_LABELSIZE,
    )
    axes[-1].tick_params(axis="x", length=0)
    axes[-1].set_xlim(x.min() - 0.5 * COLUMN_SPACING, x.max() + 0.5 * COLUMN_SPACING)

    method_handles = [
        Line2D(
            [0], [0], marker=METHOD_MARKERS[method], linestyle="none",
            color="0.25", markerfacecolor="0.25", markersize=np.sqrt(POINT_SIZE),
            label=method,
        )
        for method in METHODS
    ]
    color_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            color=get_reference_color(dataset),
            markerfacecolor=get_reference_color(dataset),
            markersize=np.sqrt(POINT_SIZE),
            label=get_reference_name(dataset),
        )
        for dataset in REFERENCE_DATASETS
    ]
    axes[0].legend(
        handles=method_handles + color_handles,
        loc="upper left", frameon=False, fontsize=TICK_LABELSIZE, ncol=3,
    )

    fig.tight_layout()
    if WRITE_TO_FILE:
        fig.savefig(
            filename_out, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white"
        )
        print(f"Wrote: {filename_out}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run both threshold calculations for references and one model form."""

    validate_user_settings()

    if SELECTED_MONTH != STORM_HANS_MONTH:
        print(
            "Note: Storm Hans remains the August 2023 threshold, while the "
            "calendar-record threshold uses "
            f"{MONTH_NAMES[SELECTED_MONTH - 1]}."
        )

    reference_data = {
        dataset: read_reference_data(dataset)
        for dataset in REFERENCE_DATASETS
    }
    model_data = read_model_data()

    thresholds = {
        dataset: {
            "storm_hans": reference_data[dataset]["storm_hans_value"],
            "calendar_record": reference_data[dataset]["record_value"],
        }
        for dataset in REFERENCE_DATASETS
    }
    event_labels = {
        dataset: {
            "storm_hans": f"Storm Hans {STORM_HANS_YEAR}",
            "calendar_record": (
                f"{MONTH_NAMES[SELECTED_MONTH - 1]} record "
                f"{reference_data[dataset]['record_year']}"
            ),
        }
        for dataset in REFERENCE_DATASETS
    }

    samples = {
        ("reference", dataset): reference_data[dataset]["fit_values"]
        for dataset in REFERENCE_DATASETS
    }
    samples.update({
        ("model", dataset): model_data["samples"][dataset]
        for dataset in REFERENCE_DATASETS
    })

    group_labels = {
        "reference": "Reference",
        "model": model_data["label"],
    }
    results = {}

    for group_key in ["reference", "model"]:
        for threshold_dataset in REFERENCE_DATASETS:
            sample_values = samples[(group_key, threshold_dataset)]
            for method_name in METHODS:
                for event_key in ["storm_hans", "calendar_record"]:
                    analysis = analyse_method_event_aep(
                        sample_values=sample_values,
                        event_value=thresholds[threshold_dataset][event_key],
                        method_name=method_name,
                    )
                    results[(group_key, threshold_dataset, method_name, event_key)] = analysis
                    print_result(
                        group_name=group_labels[group_key],
                        threshold_dataset=threshold_dataset,
                        method_name=method_name,
                        event_label=event_labels[threshold_dataset][event_key],
                        analysis=analysis,
                    )

    plot_results(
        results=results,
        model_label=model_data["label"],
        filename_out=make_figure_filename(),
    )


if __name__ == "__main__":
    main()
