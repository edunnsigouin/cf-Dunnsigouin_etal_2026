"""Create a five-panel publication figure of exceedance probabilities.

Panels a-d reproduce the four-panel workflow:
    a) left month, top threshold
    b) right month, top threshold
    c) left month, bottom threshold
    d) right month, bottom threshold

Panel e spans the full figure width and shows January-December AEP medians for
both Storm Hans and calendar-record thresholds.

Panels a-d use one shared legend. Panel e has its own legend:
    - color distinguishes Reference and Model;
    - filled versus open markers distinguish calendar-record and Storm Hans
      thresholds.

For panel e, Reference medians use GEV, Gumbel, and GenEx. Model medians use
the methods listed in METHODS. No min-max ranges are plotted.

Only the user-selected N_AEP_YEARS probability horizon is plotted. Values below
YMIN_PERCENT are plotted at YMIN_PERCENT, and the lowest y-axis tick is labeled
with a leading "<".

SciPy's genextreme shape parameter c has the opposite sign from conventional
GEV xi: xi = -c.
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

# Figure layout
# -------------
# Calendar month used in the left-column panels.
# 1 = January, ..., 8 = August, ..., 12 = December.
LEFT_PANEL_MONTH = 8

# Calendar month used in the right-column panels.
RIGHT_PANEL_MONTH = 5

# Threshold used in the top-row panels.
# Options: "storm_hans" or "calendar_record".
TOP_ROW_THRESHOLD = "storm_hans"

# Threshold used in the bottom-row panels.
# Options: "storm_hans" or "calendar_record".
BOTTOM_ROW_THRESHOLD = "calendar_record"

# Panel e settings
# ----------------
# Observational dataset used for the Reference sample in panel e.
# Options: "senorge" or "era5".
PANEL_E_REFERENCE_DATASET = "senorge"

# Dataset used to define the threshold applied to the Model sample in panel e.
# Options: "senorge" or "era5".
PANEL_E_MODEL_THRESHOLD_DATASET = "senorge"

# Panel e always shows both threshold types:
#     "storm_hans" and "calendar_record".
PANEL_E_THRESHOLD_TYPES = [
    "storm_hans",
    "calendar_record",
]


# Storm Hans event date.
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

# Inclusive period used to calculate calendar-month record thresholds.
RECORD_START_YEAR = 1957
RECORD_END_YEAR = 2022

# These values are assigned internally while each panel is calculated.
SELECTED_MONTH = LEFT_PANEL_MONTH
EVENT_THRESHOLD = TOP_ROW_THRESHOLD

# Probability horizon shown in every panel.
# Set to 1 for annual exceedance probability.
N_AEP_YEARS = 1

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

# Plot the empirical estimate for the observational Reference group.
# The model empirical estimate is always calculated and plotted.
PLOT_REFERENCE_EMPIRICAL = False

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
MODEL_DATA_MODE = "bias_corrected"

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

FIG_WIDTH_IN = 10
FIG_HEIGHT_IN = 16
FIGURE_DPI = 300

POINT_SIZE = 60

# Horizontal spacing between the Reference and Model groups.
GROUP_SPACING = 1.4

# Dataset offset from each group tick. SeNorge is plotted to the left and ERA5
# to the right.
DATASET_X_OFFSET = 0.16

METHOD_MARKERS = {
    "GEV": "o",
    "Gumbel": "s",
    "GenEx": "D",
    "Empirical": "^",
}

# Colors used only in panel e.
PANEL_E_REFERENCE_COLOR = "tab:blue"
PANEL_E_MODEL_COLOR = "goldenrod"

AXIS_LABELSIZE = 12
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 12
LEGEND_LABELSIZE = 10
ANNOTATION_FONTSIZE = 8

# Logarithmic AEP axis limits in percent. Leave either as None to determine
# that limit automatically from the plotted point estimates.
YMIN_PERCENT = 0.0001
YMAX_PERCENT = 100

# Multiplicative padding applied when an automatic log-axis limit is used.
YMIN_PADDING_FACTOR = 0.5
YMAX_PADDING_FACTOR = 2.0

# Share x-axis limits and tick locations across all four panels.
SHARE_X_AXES = True

# Share y-axis limits and tick locations across all four panels.
SHARE_Y_AXES = True

SHOW_POINT_LABELS = False
WRITE_TO_FILE = True


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

    for parameter_name, month in [
        ("LEFT_PANEL_MONTH", LEFT_PANEL_MONTH),
        ("RIGHT_PANEL_MONTH", RIGHT_PANEL_MONTH),
        ("STORM_HANS_MONTH", STORM_HANS_MONTH),
    ]:
        if month not in range(1, 13):
            raise ValueError(
                f"{parameter_name} must be an integer from 1 to 12."
            )

    for parameter_name, threshold_type in [
        ("TOP_ROW_THRESHOLD", TOP_ROW_THRESHOLD),
        ("BOTTOM_ROW_THRESHOLD", BOTTOM_ROW_THRESHOLD),
    ]:
        if threshold_type not in {"storm_hans", "calendar_record"}:
            raise ValueError(
                f"{parameter_name} must be either 'storm_hans' "
                f"or 'calendar_record'."
            )

    if set(PANEL_E_THRESHOLD_TYPES) != {
        "storm_hans",
        "calendar_record",
    }:
        raise ValueError(
            "PANEL_E_THRESHOLD_TYPES must contain exactly "
            "'storm_hans' and 'calendar_record'."
        )

    for parameter_name, dataset in [
        ("PANEL_E_REFERENCE_DATASET", PANEL_E_REFERENCE_DATASET),
        (
            "PANEL_E_MODEL_THRESHOLD_DATASET",
            PANEL_E_MODEL_THRESHOLD_DATASET,
        ),
    ]:
        if dataset not in REFERENCE_DATASETS:
            raise ValueError(
                f"{parameter_name} must be one of {REFERENCE_DATASETS}."
            )

    if RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError(
            "RECORD_END_YEAR must be greater than or equal to "
            "RECORD_START_YEAR."
        )

    if not isinstance(N_AEP_YEARS, int) or N_AEP_YEARS < 1:
        raise ValueError(
            "N_AEP_YEARS must be a positive integer."
        )

    if not isinstance(SHARE_X_AXES, bool):
        raise TypeError("SHARE_X_AXES must be either True or False.")

    if not isinstance(SHARE_Y_AXES, bool):
        raise TypeError("SHARE_Y_AXES must be either True or False.")

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

    if EVENT_THRESHOLD not in {"storm_hans", "calendar_record"}:
        raise ValueError(
            'EVENT_THRESHOLD must be either "storm_hans" or '
            '"calendar_record".'
        )

    if not isinstance(PLOT_REFERENCE_EMPIRICAL, bool):
        raise TypeError(
            "PLOT_REFERENCE_EMPIRICAL must be either True or False."
        )

    if GROUP_SPACING <= 0:
        raise ValueError(
            "GROUP_SPACING must be greater than zero."
        )

    if DATASET_X_OFFSET <= 0 or DATASET_X_OFFSET >= 0.5 * GROUP_SPACING:
        raise ValueError(
            "DATASET_X_OFFSET must be positive and less than half "
            "GROUP_SPACING."
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
    threshold_label = (
        "storm-hans"
        if EVENT_THRESHOLD == "storm_hans"
        else f"{month_name}-record"
    )

    filename = 'fig-04.png'


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


def calculate_horizon_aep(
    annual_exceedance_probability,
    horizon_years,
):
    """Convert annual exceedance probability to a horizon-year AEP."""

    annual_exceedance_probability = float(
        np.clip(
            annual_exceedance_probability,
            0.0,
            1.0,
        )
    )

    return float(
        -np.expm1(
            horizon_years
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
    return {
        "fitted_parameters": fitted_parameters,
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
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

    return {
        "fitted_parameters": None,
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
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
    """Print the return period and N-year AEP."""

    return_period = analysis["return_period"]
    return_period_text = (
        f"{return_period:.3f} years" if np.isfinite(return_period) else "infinite"
    )
    annual_probability = analysis["annual_exceedance_probability"]
    n_year_aep = calculate_horizon_aep(
        annual_probability,
        N_AEP_YEARS,
    )
    print(
        f"{event_label} | {group_name} | "
        f"{get_reference_name(threshold_dataset)} | {method_name}: "
        f"return period = {return_period_text}; "
        f"{N_AEP_YEARS}-year AEP = {100.0 * n_year_aep:.6g}%"
    )


def display_aep_percent(aep):
    """Convert AEP to percent and apply the lower plotting limit."""

    percent = 100.0 * float(aep)

    if YMIN_PERCENT is None:
        if percent <= 0.0:
            raise ValueError(
                "YMIN_PERCENT must be set when an AEP can be zero "
                "or negative on a logarithmic axis."
            )
        return percent

    return max(percent, float(YMIN_PERCENT))


def build_plot_groups(model_label):
    """Return the two Reference and Model x-axis groups."""

    return [
        ("reference", "Reference"),
        ("model", model_label),
    ]



# =============================================================================
# Five-panel publication figure
# =============================================================================

def build_panel_configurations():
    """Return the four panels in reading order.

    Columns are controlled by LEFT_PANEL_MONTH and RIGHT_PANEL_MONTH.
    Rows are controlled by TOP_ROW_THRESHOLD and BOTTOM_ROW_THRESHOLD.
    """

    return [
        ("a", LEFT_PANEL_MONTH, TOP_ROW_THRESHOLD),
        ("b", RIGHT_PANEL_MONTH, TOP_ROW_THRESHOLD),
        ("c", LEFT_PANEL_MONTH, BOTTOM_ROW_THRESHOLD),
        ("d", RIGHT_PANEL_MONTH, BOTTOM_ROW_THRESHOLD),
    ]


def calculate_panel(panel_label, month, threshold_type):
    """Calculate one month/threshold panel using the existing workflow."""

    global SELECTED_MONTH
    global EVENT_THRESHOLD

    SELECTED_MONTH = month
    EVENT_THRESHOLD = threshold_type

    reference_data = {
        dataset: read_reference_data(dataset)
        for dataset in REFERENCE_DATASETS
    }
    model_data = read_model_data()

    thresholds = {
        dataset: (
            reference_data[dataset]["storm_hans_value"]
            if threshold_type == "storm_hans"
            else reference_data[dataset]["record_value"]
        )
        for dataset in REFERENCE_DATASETS
    }
    event_labels = {
        dataset: (
            f"Storm Hans {STORM_HANS_YEAR}"
            if threshold_type == "storm_hans"
            else (
                f"{MONTH_NAMES[month - 1]} record "
                f"{reference_data[dataset]['record_year']}"
            )
        )
        for dataset in REFERENCE_DATASETS
    }

    samples = {
        ("reference", dataset): reference_data[dataset]["fit_values"]
        for dataset in REFERENCE_DATASETS
    }
    samples.update(
        {
            ("model", dataset): model_data["samples"][dataset]
            for dataset in REFERENCE_DATASETS
        }
    )

    group_labels = {
        "reference": "Reference",
        "model": model_data["label"],
    }
    results = {}

    for group_key in ["reference", "model"]:
        for threshold_dataset in REFERENCE_DATASETS:
            sample_values = samples[(group_key, threshold_dataset)]
            event_value = thresholds[threshold_dataset]

            for method_name in METHODS:
                if (
                    group_key == "reference"
                    and method_name == "Empirical"
                    and not PLOT_REFERENCE_EMPIRICAL
                ):
                    continue

                analysis = analyse_method_event_aep(
                    sample_values=sample_values,
                    event_value=event_value,
                    method_name=method_name,
                )
                results[
                    (group_key, threshold_dataset, method_name)
                ] = analysis
                print_result(
                    group_name=group_labels[group_key],
                    threshold_dataset=threshold_dataset,
                    method_name=method_name,
                    event_label=event_labels[threshold_dataset],
                    analysis=analysis,
                )

    month_name = MONTH_NAMES[month - 1]
    if threshold_type == "storm_hans":
        title = (
            f"{month_name} exceedance probability\nfor Storm Hans 2023"
        )
    else:
        title = (
            f"{month_name} exceedance probability\nfor "
            f"{month_name} {RECORD_START_YEAR}-{RECORD_END_YEAR} record"
        )

    return {
        "panel_label": panel_label,
        "title": f"{panel_label}) {title}",
        "results": results,
        "model_label": model_data["label"],
    }


def make_five_panel_figure_filename():
    """Construct the publication figure filename."""

    model_label = "raw" if MODEL_DATA_MODE == "raw" else "bc"
    filename = 'fig-04.png'

    return os.path.join(config.dirs["fig"], filename)



def get_panel_e_threshold(reference_data, threshold_type):
    """Return the selected panel e threshold and a title label."""

    if threshold_type == "storm_hans":
        return (
            reference_data["storm_hans_value"],
            f"Storm Hans {STORM_HANS_YEAR}",
        )

    return (
        reference_data["record_value"],
        (
            f"{MONTH_NAMES[SELECTED_MONTH - 1]} "
            f"{RECORD_START_YEAR}-{RECORD_END_YEAR} record"
        ),
    )


def calculate_panel_e():
    """Calculate monthly Reference and Model medians for both thresholds."""

    global SELECTED_MONTH
    global EVENT_THRESHOLD

    monthly_outputs = []

    for month in range(1, 13):
        SELECTED_MONTH = month

        reference_data = read_reference_data(
            PANEL_E_REFERENCE_DATASET
        )
        threshold_data = read_reference_data(
            PANEL_E_MODEL_THRESHOLD_DATASET
        )
        model_data = read_model_data()

        threshold_results = {}

        for threshold_type in PANEL_E_THRESHOLD_TYPES:
            EVENT_THRESHOLD = threshold_type

            reference_threshold, reference_threshold_label = (
                get_panel_e_threshold(
                    reference_data=reference_data,
                    threshold_type=threshold_type,
                )
            )
            model_threshold, model_threshold_label = (
                get_panel_e_threshold(
                    reference_data=threshold_data,
                    threshold_type=threshold_type,
                )
            )

            reference_sample = reference_data["fit_values"]
            model_sample = model_data["samples"][
                PANEL_E_REFERENCE_DATASET
            ]

            results = {}

            # Reference excludes empirical.
            for method in DISTRIBUTIONS:
                results[("reference", method)] = (
                    analyse_method_event_aep(
                        sample_values=reference_sample,
                        event_value=reference_threshold,
                        method_name=method,
                    )
                )

            # Model uses the methods listed in METHODS.
            for method in METHODS:
                results[("model", method)] = analyse_method_event_aep(
                    sample_values=model_sample,
                    event_value=model_threshold,
                    method_name=method,
                )

            threshold_results[threshold_type] = {
                "reference_threshold_label": (
                    reference_threshold_label
                ),
                "model_threshold_label": model_threshold_label,
                "results": results,
            }

        monthly_outputs.append(
            {
                "month": month,
                "threshold_results": threshold_results,
            }
        )

    return {
        "panel_label": "e",
        "title": (
            "e) Median monthly exceedance probability for Storm Hans "
            "and calendar-month records"
        ),
        "monthly_outputs": monthly_outputs,
        "model_label": model_data["label"],
        "reference_label": (
            f"Reference: "
            f"{get_reference_name(PANEL_E_REFERENCE_DATASET)}"
        ),
        "model_legend_label": (
            f"{model_data['label']}: "
            f"{get_reference_name(PANEL_E_MODEL_THRESHOLD_DATASET)} threshold"
        ),
    }


def get_panel_e_median(results, group_key, methods):
    """Return the median panel e AEP across selected methods."""

    values = [
        display_aep_percent(
            calculate_horizon_aep(
                results[(group_key, method)][
                    "annual_exceedance_probability"
                ],
                N_AEP_YEARS,
            )
        )
        for method in methods
    ]

    return float(np.median(values))


def probability_tick_formatter(y, position, ymin):
    """Format probability ticks and mark the lower plotting bound."""

    if y <= 0:
        return ""

    label = f"{y:g}%"
    if np.isclose(y, ymin):
        return f"<{label}"

    return label


def configure_probability_axis(ax, ymin, ymax):
    """Apply shared logarithmic probability-axis formatting."""

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, position: probability_tick_formatter(
                y,
                position,
                ymin,
            )
        )
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        which="major",
        linestyle=":",
        linewidth=0.6,
        alpha=0.45,
    )


def plot_panel_e(ax, panel_e_output, ymin, ymax):
    """Plot monthly medians for both threshold types."""

    month_x = np.arange(1, 13, dtype=float)

    group_offsets = {
        "reference": -0.12,
        "model": 0.12,
    }
    threshold_offsets = {
        "storm_hans": -0.035,
        "calendar_record": 0.035,
    }
    colors = {
        "reference": PANEL_E_REFERENCE_COLOR,
        "model": PANEL_E_MODEL_COLOR,
    }
    method_sets = {
        "reference": DISTRIBUTIONS,
        "model": METHODS,
    }
    marker_face = {
        "storm_hans": "none",
        "calendar_record": "full",
    }

    for group_key in ["reference", "model"]:
        for threshold_type in PANEL_E_THRESHOLD_TYPES:
            x_values = []
            medians = []

            for month_index, monthly_output in enumerate(
                panel_e_output["monthly_outputs"],
                start=1,
            ):
                results = monthly_output["threshold_results"][
                    threshold_type
                ]["results"]

                x_values.append(
                    month_index
                    + group_offsets[group_key]
                    + threshold_offsets[threshold_type]
                )
                medians.append(
                    get_panel_e_median(
                        results=results,
                        group_key=group_key,
                        methods=method_sets[group_key],
                    )
                )

            ax.plot(
                x_values,
                medians,
                linestyle="none",
                marker="o",
                markersize=np.sqrt(POINT_SIZE) * 0.85,
                markerfacecolor=(
                    colors[group_key]
                    if marker_face[threshold_type] == "full"
                    else "none"
                ),
                markeredgecolor=colors[group_key],
                markeredgewidth=1.4,
                color=colors[group_key],
                zorder=4,
            )

    configure_probability_axis(ax, ymin, ymax)
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(month_x)
    ax.set_xticklabels(
        [month[:3] for month in MONTH_NAMES],
        fontsize=TICK_LABELSIZE,
    )
    ax.set_xlabel("Calendar month", fontsize=AXIS_LABELSIZE)
    ax.set_title(
        panel_e_output["title"],
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        loc="center",
        pad=8,
    )

    panel_e_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=PANEL_E_REFERENCE_COLOR,
            markerfacecolor=PANEL_E_REFERENCE_COLOR,
            markeredgecolor=PANEL_E_REFERENCE_COLOR,
            markersize=np.sqrt(POINT_SIZE) * 0.85,
            label=panel_e_output["reference_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=PANEL_E_MODEL_COLOR,
            markerfacecolor=PANEL_E_MODEL_COLOR,
            markeredgecolor=PANEL_E_MODEL_COLOR,
            markersize=np.sqrt(POINT_SIZE) * 0.85,
            label=panel_e_output["model_legend_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.4,
            markersize=np.sqrt(POINT_SIZE) * 0.85,
            label="Storm Hans threshold",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=np.sqrt(POINT_SIZE) * 0.85,
            label="Calendar-month record threshold",
        ),
    ]

    ax.legend(
        handles=panel_e_handles,
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_LABELSIZE,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def plot_five_panels(panel_outputs, panel_e_output, model_label, filename_out):
    """Plot panels a-d above one full-width panel e."""

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

    groups = build_plot_groups(model_label)
    group_x = GROUP_SPACING * np.arange(len(groups), dtype=float)
    dataset_offsets = {
        "senorge": -DATASET_X_OFFSET,
        "era5": DATASET_X_OFFSET,
    }
    dataset_facecolors = {
        "senorge": "black",
        "era5": "none",
    }

    all_values = []
    for panel_output in panel_outputs:
        results = panel_output["results"]
        for group_key, _ in groups:
            for threshold_dataset in REFERENCE_DATASETS:
                for method_name in METHODS:
                    if (
                        group_key == "reference"
                        and method_name == "Empirical"
                        and not PLOT_REFERENCE_EMPIRICAL
                    ):
                        continue
                    analysis = results[
                        (group_key, threshold_dataset, method_name)
                    ]
                    all_values.append(
                        display_aep_percent(
                            calculate_horizon_aep(
                                analysis["annual_exceedance_probability"],
                                N_AEP_YEARS,
                            )
                        )
                    )

    for monthly_output in panel_e_output["monthly_outputs"]:
        for threshold_type in PANEL_E_THRESHOLD_TYPES:
            results = monthly_output["threshold_results"][
                threshold_type
            ]["results"]

            all_values.append(
                get_panel_e_median(
                    results=results,
                    group_key="reference",
                    methods=DISTRIBUTIONS,
                )
            )
            all_values.append(
                get_panel_e_median(
                    results=results,
                    group_key="model",
                    methods=METHODS,
                )
            )

    positive_values = np.asarray(
        [
            value
            for value in all_values
            if np.isfinite(value) and value > 0
        ],
        dtype=float,
    )
    if positive_values.size == 0:
        raise ValueError(
            "No positive AEP values are available for the log axis."
        )

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

    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.15],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(
        grid[0, 1],
        sharex=ax_a if SHARE_X_AXES else None,
        sharey=ax_a if SHARE_Y_AXES else None,
    )
    ax_c = fig.add_subplot(
        grid[1, 0],
        sharex=ax_a if SHARE_X_AXES else None,
        sharey=ax_a if SHARE_Y_AXES else None,
    )
    ax_d = fig.add_subplot(
        grid[1, 1],
        sharex=ax_a if SHARE_X_AXES else None,
        sharey=ax_a if SHARE_Y_AXES else None,
    )
    ax_e = fig.add_subplot(
        grid[2, :],
        sharey=ax_a if SHARE_Y_AXES else None,
    )
    axes = np.asarray([ax_a, ax_b, ax_c, ax_d])

    for ax, panel_output in zip(axes, panel_outputs):
        results = panel_output["results"]

        for group_index, (group_key, _) in enumerate(groups):
            for threshold_dataset in REFERENCE_DATASETS:
                point_x = (
                    group_x[group_index]
                    + dataset_offsets[threshold_dataset]
                )

                for method_name in METHODS:
                    if (
                        group_key == "reference"
                        and method_name == "Empirical"
                        and not PLOT_REFERENCE_EMPIRICAL
                    ):
                        continue

                    analysis = results[
                        (group_key, threshold_dataset, method_name)
                    ]
                    estimate = display_aep_percent(
                        calculate_horizon_aep(
                            analysis["annual_exceedance_probability"],
                            N_AEP_YEARS,
                        )
                    )

                    ax.plot(
                        point_x,
                        estimate,
                        marker=METHOD_MARKERS[method_name],
                        markersize=np.sqrt(POINT_SIZE),
                        markerfacecolor=(
                            dataset_facecolors[threshold_dataset]
                        ),
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        color="black",
                        linestyle="none",
                        zorder=(
                            4
                            if threshold_dataset == "senorge"
                            else 3
                        ),
                    )

                    if SHOW_POINT_LABELS:
                        ax.annotate(
                            f"{estimate:.3g}%",
                            (point_x, estimate),
                            xytext=(
                                -5
                                if threshold_dataset == "senorge"
                                else 5,
                                4,
                            ),
                            textcoords="offset points",
                            ha=(
                                "right"
                                if threshold_dataset == "senorge"
                                else "left"
                            ),
                            fontsize=ANNOTATION_FONTSIZE,
                        )

        ax.set_yscale("log")
        ax.set_ylim(plot_ymin, plot_ymax)
        ax.set_xlim(
            group_x.min() - 0.5 * GROUP_SPACING,
            group_x.max() + 0.5 * GROUP_SPACING,
        )
        ax.set_xticks(group_x)
        ax.set_xticklabels(
            [label for _, label in groups],
            fontsize=TICK_LABELSIZE,
        )
        ax.tick_params(axis="x", length=0, pad=5)
        ax.tick_params(axis="y", labelsize=TICK_LABELSIZE)
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, position: probability_tick_formatter(
                    y,
                    position,
                    plot_ymin,
                )
            )
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(
            axis="y",
            which="major",
            linestyle=":",
            linewidth=0.6,
            alpha=0.45,
        )

        ax.set_title(
            panel_output["title"],
            fontsize=TITLE_FONTSIZE,
            fontweight="normal",
            loc="center",
            pad=8,
        )
        
    axes[0].set_ylabel(
        f"{N_AEP_YEARS}-year exceedance probability [%]",
        fontsize=AXIS_LABELSIZE,
    )
    axes[2].set_ylabel(
        f"{N_AEP_YEARS}-year exceedance probability [%]",
        fontsize=AXIS_LABELSIZE,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            linestyle="none",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=np.sqrt(POINT_SIZE),
            label=method,
        )
        for method in METHODS
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=np.sqrt(POINT_SIZE),
            label="SeNorge",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=np.sqrt(POINT_SIZE),
            label="ERA5",
        ),
    ]

    axes[0].legend(
        handles=method_handles + dataset_handles,
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_LABELSIZE,
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    plot_panel_e(
        ax=ax_e,
        panel_e_output=panel_e_output,
        ymin=plot_ymin,
        ymax=plot_ymax,
    )
    ax_e.set_ylabel(
        f"{N_AEP_YEARS}-year exceedance probability [%]",
        fontsize=AXIS_LABELSIZE,
    )

    if WRITE_TO_FILE:
        fig.savefig(
            filename_out,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Wrote: {filename_out}")

    plt.show()


def main():
    """Calculate and plot the five publication panels."""

    validate_user_settings()

    panel_outputs = [
        calculate_panel(
            panel_label=panel_label,
            month=month,
            threshold_type=threshold_type,
        )
        for panel_label, month, threshold_type
        in build_panel_configurations()
    ]

    panel_e_output = calculate_panel_e()

    model_labels = {
        panel_output["model_label"]
        for panel_output in panel_outputs
    }
    model_labels.add(panel_e_output["model_label"])

    if len(model_labels) != 1:
        raise RuntimeError(
            "All panels must use the same model configuration."
        )

    plot_five_panels(
        panel_outputs=panel_outputs,
        panel_e_output=panel_e_output,
        model_label=panel_outputs[0]["model_label"],
        filename_out=make_five_panel_figure_filename(),
    )

if __name__ == "__main__":
    main()
