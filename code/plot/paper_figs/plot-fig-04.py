"""
Create a 3 x 2 publication figure comparing AEP or return period for:

    Reference
    Model raw
    Model bias corrected

The figure is organized by row and column.

Columns
-------
Left column:
    Storm Hans threshold.

    For a single-month row, the August 2023 Storm Hans precipitation from
    REFERENCE_DATASET is compared with that row's calendar-month distribution.

    For an annual row (month = 0), the same Storm Hans value is compared with
    each of the 12 monthly distributions. The 12 monthly exceedance
    probabilities are then combined into the probability of at least one
    exceedance somewhere in the year.

Right column:
    Calendar-month record threshold.

    For a single-month row, the threshold is the largest value for that
    calendar month in REFERENCE_DATASET during
    RECORD_START_YEAR-RECORD_END_YEAR.

    For an annual row (month = 0), each of the 12 months uses its own
    calendar-month record. The 12 monthly exceedance probabilities are then
    combined into the probability that at least one month exceeds its own
    historical record during a year.

Rows
----
The three rows are controlled by the user input ROW_CALENDAR_MONTHS.

Calendar-month values are:

    0  = Annual / any month of the year
    1  = January
    2  = February
    ...
    8  = August
    ...
    12 = December

For example:

    ROW_CALENDAR_MONTHS = [8, 5, 0]

produces:

    Row 1: August
    Row 2: May
    Row 3: Annual

Panel letters follow normal reading order:

    a) row 1, Storm Hans
    b) row 1, calendar record
    c) row 2, Storm Hans
    d) row 2, calendar record
    e) row 3, Storm Hans
    f) row 3, calendar record

Annual probability
------------------
For an annual row, let p_m be the exceedance probability calculated separately
for calendar month m.

Assuming exceedances in different calendar months are independent:

    p_any = 1 - product over m=1,...,12 of (1 - p_m)

The annual return period is:

    T_any = 1 / p_any

The independence assumption means that an exceedance in one month is assumed
not to change the chance of an exceedance in another month. This is a useful
approximation, but monthly precipitation extremes can still share some
climate-scale dependence.

Methods and datasets
--------------------
Every panel compares:

    Reference
    Model raw
    Model BC

using four estimation methods:

    GEV
    Gumbel
    GenEx
    Empirical

Different marker shapes identify the four methods. All markers have empty faces
and are vertically aligned within each dataset group.

Set PLOT_METRIC to:

    "aep"
        Plot the probability [%] of at least one exceedance during
        N_AEP_YEARS.

    "return_period"
        Plot return period [years], where T = 1 / p.

Model input
-----------
The compact model files contain:

    tp24_max(number, i_date)

or, for a selected lead-location split:

    tp24_max_lead<start>_<end>(number, i_date)

Calendar-month membership is stored in month(i_date).

Raw files have the form:

    monthly_max_samples_<variable>_<N>dayacc_<catchment>_
    lead<full>_split<...>_<forecast_start>_<forecast_end>.nc

Bias-corrected files append:

    _bc_<method>_<reference>.nc

Supported correction methods are:

    mm
    q
    ld
    doy
    q_doy

REFERENCE_DATASET supplies the observational sample and thresholds and is also
used to select the bias-corrected model file.

Note
----
SciPy's genextreme shape parameter c has the opposite sign from conventional
GEV xi:

    xi = -c
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
# USER INPUTS
# =============================================================================

# -----------------------------------------------------------------------------
# Main analysis choices
# -----------------------------------------------------------------------------

# Quantity shown on the y-axis.
#
# Options:
#     "aep"
#     "return_period"
PLOT_METRIC = "return_period"

# Used only when PLOT_METRIC == "aep".
# Set to 1 for annual exceedance probability.
N_AEP_YEARS = 1

# Observational dataset used for:
#     1. the Reference distribution;
#     2. the Storm Hans threshold;
#     3. the calendar-record threshold;
#     4. selection of the bias-corrected model file.
#
# Options:
#     "senorge"
#     "era5"
REFERENCE_DATASET = "senorge"

# Bias-correction method shown as the third x-axis group.
#
# Options:
#     "mm"
#     "q"
#     "ld"
#     "doy"
#     "q_doy"
BIAS_CORRECTION_METHOD = "mm"


# -----------------------------------------------------------------------------
# Event and observational settings
# -----------------------------------------------------------------------------

# Calendar month represented by each figure row.
#
#     0       = annual / any month
#     1...12  = January...December
#
# Example below gives August, May, and Annual.
ROW_CALENDAR_MONTHS = [
    8,
    5,
    0,
]

STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

# Inclusive period used for the calendar-month record threshold.
RECORD_START_YEAR = 1957
RECORD_END_YEAR = 2022

OBSERVATION_YEARS = [
    "1957",
    "2023",
]

# Read 2023 to obtain Storm Hans, but omit 2023 from the fitted observational
# Reference distribution if True.
EXCLUDE_2023_FROM_REFERENCE_FIT = True


# -----------------------------------------------------------------------------
# Catchment and compact UNSEEN input
# -----------------------------------------------------------------------------

CATCHMENT = "regine_drammen"
X_DAYS = 2

MODEL_VARIABLE = "tp24"

FORECAST_DATE_RANGE = [
    "2020-01-02",
    "2022-12-29",
]

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Which compact model variable is used.
#
# Options:
#     "full"
#     "split1"
#     "split2"
#     ...
MODEL_SAMPLING_GROUP = "full"

# Optional explicit filenames.
# Leave as None to construct them automatically.
REFERENCE_FILENAME_OVERRIDE = None
RAW_MODEL_FILENAME_OVERRIDE = None
BIAS_CORRECTED_MODEL_FILENAME_OVERRIDE = None


# -----------------------------------------------------------------------------
# Figure appearance
# -----------------------------------------------------------------------------

FIG_WIDTH_IN = 10.5
FIG_HEIGHT_IN = 12.0
FIGURE_DPI = 400

POINT_SIZE = 68
MARKER_EDGE_WIDTH = 1.4

METHOD_MARKERS = {
    "GEV": "o",
    "Gumbel": "s",
    "GenEx": "D",
    "Empirical": "^",
}

# All four methods use exactly the same x coordinate within a dataset group.
METHOD_X_JITTER = 0.0

# Marker faces are intentionally empty in all panels.
MARKER_FACECOLOR = "none"

# One edge color is used for all calculations.
MARKER_EDGECOLOR = "0.20"

AXIS_LABELSIZE = 12
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 10

# AEP y-axis limits in percent.
AEP_YMIN_PERCENT = 0.0001
AEP_YMAX_PERCENT = 100.0

# Return-period y-axis limits in years.
RETURN_PERIOD_YMIN_YEARS = 1.0
RETURN_PERIOD_YMAX_YEARS = 1_000_000.0

SHARE_Y_AXES = False
SHOW_GRID = True

WRITE_TO_FILE = True
SHOW_FIGURE = True


# =============================================================================
# CONSTANTS
# =============================================================================

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

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

#METHODS = [
#    "GEV",
#    "Gumbel",
#    "GenEx",
#    "Empirical",
#]

METHODS = [
    "GEV",
    "Gumbel",
    "GenEx",
]

PLOT_GROUPS = [
    "reference",
    "raw",
    "bias_corrected",
]

GROUP_LABELS = {
    "reference": "Reference",
    "raw": "Model raw",
    "bias_corrected": "Model BC",
}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_user_settings():
    """Check user settings before opening any data files."""

    if PLOT_METRIC not in {
        "aep",
        "return_period",
    }:
        raise ValueError(
            "PLOT_METRIC must be 'aep' or 'return_period'."
        )

    if not isinstance(
        N_AEP_YEARS,
        int,
    ) or N_AEP_YEARS < 1:
        raise ValueError(
            "N_AEP_YEARS must be a positive integer."
        )

    if REFERENCE_DATASET not in {
        "senorge",
        "era5",
    }:
        raise ValueError(
            "REFERENCE_DATASET must be 'senorge' or 'era5'."
        )

    if BIAS_CORRECTION_METHOD not in {
        "mm",
        "q",
        "ld",
        "doy",
        "q_doy",
    }:
        raise ValueError(
            "BIAS_CORRECTION_METHOD must be one of "
            "'mm', 'q', 'ld', 'doy', or 'q_doy'."
        )

    if (
        not isinstance(
            ROW_CALENDAR_MONTHS,
            (list, tuple),
        )
        or len(
            ROW_CALENDAR_MONTHS
        ) != 3
    ):
        raise ValueError(
            "ROW_CALENDAR_MONTHS must contain exactly three month values."
        )

    for month in ROW_CALENDAR_MONTHS:

        if (
            not isinstance(
                month,
                int,
            )
            or month not in range(
                0,
                13,
            )
        ):
            raise ValueError(
                "Each ROW_CALENDAR_MONTHS value must be 0 (annual) "
                "or an integer from 1 to 12."
            )

    if RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError(
            "RECORD_END_YEAR must be >= RECORD_START_YEAR."
        )

    if X_DAYS < 1:
        raise ValueError(
            "X_DAYS must be at least 1."
        )

    first_usable_lead = (
        FIRST_INPUT_LEAD
        + X_DAYS
        - 1
    )

    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError(
            "X_DAYS is too large for the configured lead range."
        )

    number_of_usable_leads = (
        LAST_INPUT_LEAD
        - first_usable_lead
        + 1
    )

    if (
        not isinstance(
            NUMBER_OF_LEAD_BINS,
            int,
        )
        or NUMBER_OF_LEAD_BINS < 1
        or NUMBER_OF_LEAD_BINS > number_of_usable_leads
    ):
        raise ValueError(
            "NUMBER_OF_LEAD_BINS is invalid for the usable lead range."
        )

    valid_groups = {
        "full",
        *[
            f"split{number}"
            for number in range(
                1,
                NUMBER_OF_LEAD_BINS + 1,
            )
        ],
    }

    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(
            f"MODEL_SAMPLING_GROUP must be one of "
            f"{sorted(valid_groups)}."
        )


# =============================================================================
# LABELS AND FILENAMES
# =============================================================================

def get_reference_name():
    """Return publication-style reference name."""

    return {
        "senorge": "SeNorge",
        "era5": "ERA5",
    }[
        REFERENCE_DATASET
    ]


def get_reference_variable():
    """Return the selected observational variable name."""

    return {
        "senorge": SENORGE_VARIABLE,
        "era5": ERA5_VARIABLE,
    }[
        REFERENCE_DATASET
    ]


def get_model_file_id(
    catchment_name,
):
    """Return the short catchment identifier used in model filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """Split usable ending leads into near-equal bins."""

    number_of_leads = (
        last_lead
        - first_lead
        + 1
    )

    base_size = (
        number_of_leads
        // number_of_bins
    )

    remainder = (
        number_of_leads
        % number_of_bins
    )

    bin_sizes = [
        base_size
        + int(
            index
            >= number_of_bins - remainder
        )
        for index in range(
            number_of_bins
        )
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:

        current_end = (
            current_start
            + bin_size
            - 1
        )

        bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + 1
        )

    return bins


def build_lead_bins():
    """Return configured lead bins."""

    return split_usable_accumulated_leads(
        first_lead=(
            FIRST_INPUT_LEAD
            + X_DAYS
            - 1
        ),
        last_lead=LAST_INPUT_LEAD,
        number_of_bins=NUMBER_OF_LEAD_BINS,
    )


def get_full_lead_range():
    """Return complete usable ending-lead range."""

    return (
        FIRST_INPUT_LEAD
        + X_DAYS
        - 1,
        LAST_INPUT_LEAD,
    )


def get_selected_model_lead_range():
    """Return the lead range selected by MODEL_SAMPLING_GROUP."""

    if MODEL_SAMPLING_GROUP == "full":
        return get_full_lead_range()

    split_number = int(
        MODEL_SAMPLING_GROUP.replace(
            "split",
            "",
        )
    )

    return build_lead_bins()[
        split_number
        - 1
    ]


def get_model_variable():
    """Return the compact precipitation variable to read."""

    if MODEL_SAMPLING_GROUP == "full":
        return "tp24_max"

    lead_start, lead_end = (
        get_selected_model_lead_range()
    )

    return (
        f"tp24_max_lead"
        f"{lead_start}_{lead_end}"
    )


def lead_split_filename_label():
    """Return the lead-bin label used in compact model filenames."""

    full_start, full_end = (
        get_full_lead_range()
    )

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in build_lead_bins()
    )

    return (
        f"lead{full_start}-{full_end}_"
        f"split{NUMBER_OF_LEAD_BINS}_"
        f"{split_text}"
    )


def make_reference_filename():
    """Construct the observational filename."""

    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(
            REFERENCE_FILENAME_OVERRIDE
        )

    if REFERENCE_DATASET == "senorge":

        filename = (
            f"distribution_monthly_extremes_"
            f"{SENORGE_VARIABLE}_"
            f"{X_DAYS}dayacc_"
            f"{CATCHMENT}_senorge_"
            f"{OBSERVATION_YEARS[0]}-"
            f"{OBSERVATION_YEARS[1]}.nc"
        )

        return (
            Path(
                config.dirs[
                    "senorge_processed"
                ]
            )
            / filename
        )

    filename = (
        f"distribution_monthly_extremes_"
        f"{ERA5_VARIABLE}_"
        f"{X_DAYS}dayacc_"
        f"{CATCHMENT}_era5_"
        f"{ERA5_GRID}_"
        f"{OBSERVATION_YEARS[0]}-"
        f"{OBSERVATION_YEARS[1]}.nc"
    )

    return (
        Path(
            config.dirs[
                "era5_processed"
            ]
        )
        / filename
    )


def make_raw_model_filename():
    """Construct the raw compact model filename."""

    if RAW_MODEL_FILENAME_OVERRIDE is not None:
        return Path(
            RAW_MODEL_FILENAME_OVERRIDE
        )

    filename = (
        f"monthly_max_samples_"
        f"{MODEL_VARIABLE}_"
        f"{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_"
        f"{lead_split_filename_label()}_"
        f"{FORECAST_DATE_RANGE[0]}_"
        f"{FORECAST_DATE_RANGE[1]}.nc"
    )

    return (
        Path(
            config.dirs[
                "s2s_processed"
            ]
        )
        / filename
    )


def make_bias_corrected_model_filename():
    """Construct the selected bias-corrected compact model filename."""

    if BIAS_CORRECTED_MODEL_FILENAME_OVERRIDE is not None:
        return Path(
            BIAS_CORRECTED_MODEL_FILENAME_OVERRIDE
        )

    raw_filename = make_raw_model_filename()

    return raw_filename.with_name(
        (
            f"{raw_filename.stem}_"
            f"bc_{BIAS_CORRECTION_METHOD}_"
            f"{REFERENCE_DATASET}.nc"
        )
    )


def make_figure_filename():
    """Construct output figure filename."""

    metric_label = (
        f"{N_AEP_YEARS}year-aep"
        if PLOT_METRIC == "aep"
        else "return-period"
    )

    return (
        Path(
            config.dirs[
                "fig"
            ]
        )
        / (
            f"fig-04_{metric_label}_bc-{BIAS_CORRECTION_METHOD}-{REFERENCE_DATASET}.png"
        )
    )


# =============================================================================
# DATA READING
# =============================================================================

def read_reference_month(
    month,
):
    """
    Read one observational month and return:
        fitted sample,
        Storm Hans value,
        calendar-month record value.
    """

    filename = make_reference_filename()
    variable = get_reference_variable()

    if not filename.is_file():
        raise FileNotFoundError(
            f"Reference file not found: {filename}"
        )

    with xr.open_dataset(
        filename
    ) as ds:

        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}."
            )

        selected_month_data = (
            ds[
                variable
            ]
            .sel(
                year=slice(
                    int(
                        OBSERVATION_YEARS[
                            0
                        ]
                    ),
                    int(
                        OBSERVATION_YEARS[
                            1
                        ]
                    ),
                ),
                month=month,
            )
            .load()
        )

        record_data = (
            ds[
                variable
            ]
            .sel(
                year=slice(
                    RECORD_START_YEAR,
                    RECORD_END_YEAR,
                ),
                month=month,
            )
            .load()
        )

        storm_hans_value = float(
            ds[
                variable
            ]
            .sel(
                year=STORM_HANS_YEAR,
                month=STORM_HANS_MONTH,
            )
            .load()
            .values
        )

    years = np.asarray(
        selected_month_data[
            "year"
        ].values
    )

    values = np.asarray(
        selected_month_data.values,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    years = years[
        finite
    ]

    values = values[
        finite
    ]

    fit_mask = np.ones(
        values.size,
        dtype=bool,
    )

    if EXCLUDE_2023_FROM_REFERENCE_FIT:

        fit_mask &= (
            years
            != STORM_HANS_YEAR
        )

    fit_values = values[
        fit_mask
    ]

    if fit_values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {get_reference_name()} "
            f"{MONTH_NAMES[month - 1]} values remain for fitting."
        )

    record_values = np.asarray(
        record_data.values,
        dtype=float,
    )

    record_years = np.asarray(
        record_data[
            "year"
        ].values
    )

    record_finite = np.isfinite(
        record_values
    )

    if not np.any(
        record_finite
    ):
        raise ValueError(
            f"No finite {MONTH_NAMES[month - 1]} record values were found."
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

    return {
        "filename": filename,
        "fit_values": fit_values,
        "storm_hans_value": storm_hans_value,
        "record_value": record_value,
        "record_year": record_year,
    }


def read_model_month(
    filename,
    month,
    dataset_label,
):
    """Read one calendar-month sample from a compact UNSEEN file."""

    if not filename.is_file():
        raise FileNotFoundError(
            f"{dataset_label} file not found: {filename}"
        )

    variable = get_model_variable()

    with xr.open_dataset(
        filename,
        decode_timedelta=False,
    ) as ds:

        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        if "month" not in ds:
            raise KeyError(
                f"Variable 'month' was not found in {filename}."
            )

        selected = ds[
            variable
        ].where(
            ds[
                "month"
            ]
            == month,
            drop=True,
        )

        values = np.asarray(
            selected.values,
            dtype=float,
        ).ravel()

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite values were found in {dataset_label} "
            f"for {MONTH_NAMES[month - 1]}."
        )

    return values


# =============================================================================
# EXTREME-VALUE METHODS
# =============================================================================

def fit_gev(
    values,
):
    """Fit stationary three-parameter GEV."""

    shape_c, location, scale = (
        genextreme.fit(
            values
        )
    )

    if (
        not np.isfinite(
            [
                shape_c,
                location,
                scale,
            ]
        ).all()
        or scale <= 0
    ):
        raise RuntimeError(
            "The GEV fit returned invalid parameters."
        )

    return (
        shape_c,
        location,
        scale,
    )


def fit_gumbel(
    values,
):
    """Fit stationary two-parameter Gumbel."""

    location, scale = (
        gumbel_r.fit(
            values
        )
    )

    if (
        not np.isfinite(
            [
                location,
                scale,
            ]
        ).all()
        or scale <= 0
    ):
        raise RuntimeError(
            "The Gumbel fit returned invalid parameters."
        )

    return (
        location,
        scale,
    )


def genex_negative_log_likelihood(
    log_parameters,
    values,
):
    """Negative log-likelihood for the two-parameter GenEx distribution."""

    shape, scale = np.exp(
        log_parameters
    )

    if (
        not np.isfinite(
            shape
        )
        or not np.isfinite(
            scale
        )
        or shape <= 0
        or scale <= 0
        or np.any(
            values < 0
        )
    ):
        return np.inf

    z = (
        values
        / scale
    )

    log_one_minus_exp = np.log(
        -np.expm1(
            -z
        )
    )

    log_pdf = (
        np.log(
            shape
        )
        - np.log(
            scale
        )
        - z
        + (
            shape
            - 1.0
        )
        * log_one_minus_exp
    )

    if not np.isfinite(
        log_pdf
    ).all():
        return np.inf

    return -np.sum(
        log_pdf
    )


def fit_genex(
    values,
):
    """Fit a two-parameter Generalized Exponential distribution."""

    if np.any(
        values < 0
    ):
        raise ValueError(
            "GenEx requires non-negative values."
        )

    positive_values = values[
        values
        > 0
    ]

    if positive_values.size == 0:
        raise RuntimeError(
            "GenEx cannot be fitted without positive values."
        )

    result = minimize(
        genex_negative_log_likelihood,
        x0=np.log(
            [
                1.0,
                np.mean(
                    positive_values
                ),
            ]
        ),
        args=(
            values,
        ),
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"GenEx fit failed: {result.message}"
        )

    shape, scale = np.exp(
        result.x
    )

    if (
        not np.isfinite(
            [
                shape,
                scale,
            ]
        ).all()
        or shape <= 0
        or scale <= 0
    ):
        raise RuntimeError(
            "The GenEx fit returned invalid parameters."
        )

    return (
        shape,
        scale,
    )


def calculate_parametric_probability(
    values,
    event_value,
    method,
):
    """Fit one distribution and calculate annual exceedance probability."""

    if method == "GEV":

        shape_c, location, scale = fit_gev(
            values
        )

        probability = genextreme.sf(
            event_value,
            shape_c,
            loc=location,
            scale=scale,
        )

    elif method == "Gumbel":

        location, scale = fit_gumbel(
            values
        )

        probability = gumbel_r.sf(
            event_value,
            loc=location,
            scale=scale,
        )

    elif method == "GenEx":

        shape, scale = fit_genex(
            values
        )

        if event_value < 0:

            probability = 1.0

        else:

            cdf = (
                1.0
                - np.exp(
                    -event_value
                    / scale
                )
            ) ** shape

            probability = (
                1.0
                - cdf
            )

    else:

        raise ValueError(
            f"Unsupported parametric method: {method}"
        )

    if not np.isfinite(
        probability
    ):
        raise RuntimeError(
            f"{method} produced a non-finite probability."
        )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


def calculate_empirical_probability(
    values,
    event_value,
):
    """Estimate annual exceedance probability from the event rank."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size < 1:
        raise ValueError(
            "At least one finite value is required."
        )

    rank = (
        1
        + int(
            np.sum(
                values
                > event_value
            )
        )
    )

    return float(
        rank
        / values.size
    )


def analyse_method(
    values,
    event_value,
    method,
):
    """Calculate annual exceedance probability and return period."""

    if method == "Empirical":

        annual_probability = (
            calculate_empirical_probability(
                values,
                event_value,
            )
        )

    else:

        annual_probability = (
            calculate_parametric_probability(
                values,
                event_value,
                method,
            )
        )

    return_period = (
        np.inf
        if annual_probability <= 0
        else 1.0
        / annual_probability
    )

    return {
        "annual_exceedance_probability": annual_probability,
        "return_period": return_period,
    }


def calculate_horizon_aep(
    annual_probability,
    horizon_years,
):
    """Convert annual exceedance probability to N-year exceedance probability."""

    annual_probability = float(
        np.clip(
            annual_probability,
            0.0,
            1.0,
        )
    )

    return float(
        -np.expm1(
            horizon_years
            * np.log1p(
                -annual_probability
            )
        )
    )


def get_metric_value(
    analysis,
):
    """Return the selected plotting quantity, clamped to axis limits."""

    if PLOT_METRIC == "aep":

        value = (
            100.0
            * calculate_horizon_aep(
                analysis[
                    "annual_exceedance_probability"
                ],
                N_AEP_YEARS,
            )
        )

        return float(
            np.clip(
                value,
                AEP_YMIN_PERCENT,
                AEP_YMAX_PERCENT,
            )
        )

    value = float(
        analysis[
            "return_period"
        ]
    )

    if not np.isfinite(
        value
    ):
        value = RETURN_PERIOD_YMAX_YEARS

    return float(
        np.clip(
            value,
            RETURN_PERIOD_YMIN_YEARS,
            RETURN_PERIOD_YMAX_YEARS,
        )
    )


def combine_monthly_probabilities(
    monthly_probabilities,
):
    """
    Combine 12 monthly exceedance probabilities into one annual probability.

    The calculation assumes independence between monthly exceedance events:

        p_any = 1 - product(1 - p_m)
    """

    probabilities = np.asarray(
        monthly_probabilities,
        dtype=float,
    )

    if probabilities.size != 12:
        raise ValueError(
            "Annual analysis requires exactly 12 monthly probabilities."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "All monthly probabilities must be finite."
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    probability_none = float(
        np.prod(
            1.0
            - probabilities
        )
    )

    probability_any = (
        1.0
        - probability_none
    )

    return float(
        np.clip(
            probability_any,
            0.0,
            1.0,
        )
    )


def get_panel_month_label(
    month,
):
    """Return 'Annual' for month 0, otherwise the calendar-month name."""

    if month == 0:
        return "Annual"

    return MONTH_NAMES[
        month - 1
    ]


# =============================================================================
# PANEL CALCULATIONS
# =============================================================================

def calculate_panel(
    panel_label,
    month,
    threshold_type,
):
    """Calculate Reference, raw, and bias-corrected results for one panel."""

    reference = read_reference_month(
        month
    )

    raw_values = read_model_month(
        filename=make_raw_model_filename(),
        month=month,
        dataset_label="raw model",
    )

    bias_corrected_values = read_model_month(
        filename=make_bias_corrected_model_filename(),
        month=month,
        dataset_label=(
            f"bias-corrected model "
            f"({BIAS_CORRECTION_METHOD}, {REFERENCE_DATASET})"
        ),
    )

    if threshold_type == "storm_hans":

        event_value = reference[
            "storm_hans_value"
        ]

        threshold_label = "Storm Hans"

    else:

        event_value = reference[
            "record_value"
        ]

        threshold_label = "calendar record"

    samples = {
        "reference": reference[
            "fit_values"
        ],
        "raw": raw_values,
        "bias_corrected": bias_corrected_values,
    }

    results = {}

    for group_name in PLOT_GROUPS:

        for method in METHODS:

            results[
                (
                    group_name,
                    method,
                )
            ] = analyse_method(
                values=samples[
                    group_name
                ],
                event_value=event_value,
                method=method,
            )

    title = (
        f"{panel_label}) "
        f"{MONTH_NAMES[month - 1]} "
        f"{threshold_label}"
    )

    return {
        "panel_label": panel_label,
        "month": month,
        "threshold_type": threshold_type,
        "title": title,
        "event_value": event_value,
        "record_year": reference[
            "record_year"
        ],
        "results": results,
    }


def calculate_annual_panel(
    panel_label,
    threshold_type,
):
    """
    Calculate the annual any-month panel.

    Each calendar month is analysed separately using the same four methods.
    The 12 monthly exceedance probabilities are then combined using
    combine_monthly_probabilities().
    """

    raw_filename = make_raw_model_filename()
    bias_corrected_filename = (
        make_bias_corrected_model_filename()
    )

    monthly_results = {
        group_name: {
            method: []
            for method in METHODS
        }
        for group_name in PLOT_GROUPS
    }

    monthly_event_values = {}

    for month in range(
        1,
        13,
    ):

        reference = read_reference_month(
            month
        )

        raw_values = read_model_month(
            filename=raw_filename,
            month=month,
            dataset_label="raw model",
        )

        bias_corrected_values = read_model_month(
            filename=bias_corrected_filename,
            month=month,
            dataset_label=(
                f"bias-corrected model "
                f"({BIAS_CORRECTION_METHOD}, {REFERENCE_DATASET})"
            ),
        )

        if threshold_type == "storm_hans":

            event_value = reference[
                "storm_hans_value"
            ]

        else:

            event_value = reference[
                "record_value"
            ]

        monthly_event_values[
            month
        ] = event_value

        samples = {
            "reference": reference[
                "fit_values"
            ],
            "raw": raw_values,
            "bias_corrected": bias_corrected_values,
        }

        for group_name in PLOT_GROUPS:

            for method in METHODS:

                analysis = analyse_method(
                    values=samples[
                        group_name
                    ],
                    event_value=event_value,
                    method=method,
                )

                monthly_results[
                    group_name
                ][
                    method
                ].append(
                    analysis[
                        "annual_exceedance_probability"
                    ]
                )

    results = {}

    for group_name in PLOT_GROUPS:

        for method in METHODS:

            monthly_probabilities = monthly_results[
                group_name
            ][
                method
            ]

            annual_probability = (
                combine_monthly_probabilities(
                    monthly_probabilities
                )
            )

            return_period = (
                np.inf
                if annual_probability <= 0
                else 1.0
                / annual_probability
            )

            results[
                (
                    group_name,
                    method,
                )
            ] = {
                "annual_exceedance_probability": annual_probability,
                "return_period": return_period,
                "monthly_exceedance_probabilities": {
                    month: probability
                    for month, probability in zip(
                        range(
                            1,
                            13,
                        ),
                        monthly_probabilities,
                    )
                },
            }

    threshold_label = (
        "Storm Hans"
        if threshold_type == "storm_hans"
        else "calendar record"
    )

    return {
        "panel_label": panel_label,
        "month": 0,
        "threshold_type": threshold_type,
        "title": (
            f"{panel_label}) "
            f"Annual "
            f"{threshold_label}"
        ),
        "event_value": None,
        "monthly_event_values": monthly_event_values,
        "record_year": None,
        "results": results,
    }


def calculate_panel_for_month_mode(
    panel_label,
    month,
    threshold_type,
):
    """
    Calculate one panel for either a single month or the annual any-month mode.
    """

    if month == 0:

        return calculate_annual_panel(
            panel_label=panel_label,
            threshold_type=threshold_type,
        )

    return calculate_panel(
        panel_label=panel_label,
        month=month,
        threshold_type=threshold_type,
    )


def calculate_all_panels():
    """
    Calculate the six panels in 3-row x 2-column reading order.

    Left column always uses the Storm Hans threshold.
    Right column always uses the calendar-record threshold.
    Each row uses the corresponding value in ROW_CALENDAR_MONTHS.
    """

    panel_letters = [
        (
            "a",
            "b",
        ),
        (
            "c",
            "d",
        ),
        (
            "e",
            "f",
        ),
    ]

    panel_outputs = []

    for row_index, month in enumerate(
        ROW_CALENDAR_MONTHS
    ):

        storm_hans_label, calendar_record_label = (
            panel_letters[
                row_index
            ]
        )

        panel_outputs.append(
            calculate_panel_for_month_mode(
                panel_label=storm_hans_label,
                month=month,
                threshold_type="storm_hans",
            )
        )

        panel_outputs.append(
            calculate_panel_for_month_mode(
                panel_label=calendar_record_label,
                month=month,
                threshold_type="calendar_record",
            )
        )

    return panel_outputs


# =============================================================================
# REPORTING
# =============================================================================

def print_panel_results(
    panel_outputs,
):
    """Print all calculated AEP and return-period values."""

    print()
    print(
        "Reference dataset:",
        get_reference_name(),
    )

    print(
        "Bias correction:",
        BIAS_CORRECTION_METHOD,
    )

    for panel in panel_outputs:

        print()
        print(
            panel[
                "title"
            ]
        )

        if panel[
            "month"
        ] == 0:

            if panel[
                "threshold_type"
            ] == "storm_hans":

                unique_event_values = set(
                    panel[
                        "monthly_event_values"
                    ].values()
                )

                if len(
                    unique_event_values
                ) == 1:

                    event_value = next(
                        iter(
                            unique_event_values
                        )
                    )

                    print(
                        f"Storm Hans event value used for all months: "
                        f"{event_value:.4f} mm"
                    )

            else:

                print(
                    "Each month uses its own calendar-record event value."
                )

        else:

            print(
                f"Event value: "
                f"{panel['event_value']:.4f} mm"
            )

        for group_name in PLOT_GROUPS:

            for method in METHODS:

                analysis = panel[
                    "results"
                ][
                    (
                        group_name,
                        method,
                    )
                ]

                probability = analysis[
                    "annual_exceedance_probability"
                ]

                return_period = analysis[
                    "return_period"
                ]

                return_period_text = (
                    f"{return_period:.4g}"
                    if np.isfinite(
                        return_period
                    )
                    else "inf"
                )

                print(
                    f"  {GROUP_LABELS[group_name]:>12} | "
                    f"{method:>9} | "
                    f"annual p={100.0 * probability:.6g}% | "
                    f"RP={return_period_text} y"
                )


# =============================================================================
# FIGURE
# =============================================================================

def metric_tick_formatter(
    value,
    position,
):
    """Format logarithmic metric-axis ticks."""

    if value <= 0:
        return ""

    if PLOT_METRIC == "aep":

        label = (
            f"{value:g}"
        )

        if np.isclose(
            value,
            AEP_YMIN_PERCENT,
        ):
            return (
                f"<{label}"
            )

        return label

    label = (
        f"{value:g}"
    )

    if np.isclose(
        value,
        RETURN_PERIOD_YMAX_YEARS,
    ):
        return (
            f">{label}"
        )

    return label


def get_y_axis_label():
    """Return the y-axis label."""

    if PLOT_METRIC == "aep":

        return (
            f"{N_AEP_YEARS}-year exceedance probability [%]"
        )

    return (
        "Return period [years]"
    )


def configure_axis(
    axis,
):
    """Apply common publication-style axis formatting."""

    axis.set_yscale(
        "log"
    )

    if PLOT_METRIC == "aep":

        axis.set_ylim(
            0.7
            * AEP_YMIN_PERCENT,
            1.3
            * AEP_YMAX_PERCENT,
        )

    else:

        axis.set_ylim(
            0.7
            * RETURN_PERIOD_YMIN_YEARS,
            1.3
            * RETURN_PERIOD_YMAX_YEARS,
        )

    axis.yaxis.set_major_formatter(
        FuncFormatter(
            metric_tick_formatter
        )
    )

    axis.set_xlim(
        -0.55,
        len(
            PLOT_GROUPS
        )
        - 0.45,
    )

    axis.set_xticks(
        np.arange(
            len(
                PLOT_GROUPS
            )
        )
    )

    axis.set_xticklabels(
        [
            GROUP_LABELS[
                group
            ]
            if group != "bias_corrected"
            else (
                f"Model BC\n"
                f"({BIAS_CORRECTION_METHOD})"
            )
            for group in PLOT_GROUPS
        ],
        fontsize=TICK_LABELSIZE,
    )

    axis.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    if SHOW_GRID:

        axis.grid(
            axis="y",
            which="major",
            linestyle=":",
            linewidth=0.7,
            alpha=0.45,
            zorder=0,
        )


def plot_one_panel(
    axis,
    panel_output,
):
    """Plot one month/threshold panel."""

    results = panel_output[
        "results"
    ]

    method_offsets = {
        method: (
            (
                index
                - (
                    len(
                        METHODS
                    )
                    - 1
                )
                / 2.0
            )
            * METHOD_X_JITTER
        )
        for index, method in enumerate(
            METHODS
        )
    }

    for group_index, group_name in enumerate(
        PLOT_GROUPS
    ):

        for method in METHODS:

            y_value = get_metric_value(
                results[
                    (
                        group_name,
                        method,
                    )
                ]
            )

            axis.scatter(
                group_index
                + method_offsets[
                    method
                ],
                y_value,
                s=POINT_SIZE,
                marker=METHOD_MARKERS[
                    method
                ],
                facecolor=MARKER_FACECOLOR,
                edgecolor=MARKER_EDGECOLOR,
                linewidth=MARKER_EDGE_WIDTH,
                zorder=4,
            )

    configure_axis(
        axis
    )

    axis.set_title(
        panel_output[
            "title"
        ],
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        loc="left",
        pad=8,
    )


def make_figure(
    panel_outputs,
):
    """Create the requested 3 x 2 publication figure."""

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
        3,
        2,
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        ),
        sharey=SHARE_Y_AXES,
        constrained_layout=True,
    )

    axes_flat = axes.ravel()

    for axis, panel_output in zip(
        axes_flat,
        panel_outputs,
    ):

        plot_one_panel(
            axis=axis,
            panel_output=panel_output,
        )

    for row in range(
        3
    ):

        axes[
            row,
            0,
        ].set_ylabel(
            get_y_axis_label(),
            fontsize=AXIS_LABELSIZE,
        )

    for column in range(
        2
    ):

        axes[
            2,
            column,
        ].set_xlabel(
            "Dataset",
            fontsize=AXIS_LABELSIZE,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label=method,
        )
        for method in METHODS
    ]

    axes[0, 0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="best",
        ncol=1,
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    filename = make_figure_filename()

    if WRITE_TO_FILE:

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            filename,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        print(
            "Wrote:",
            filename,
        )

    if SHOW_FIGURE:

        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    panel_outputs = (
        calculate_all_panels()
    )

    print_panel_results(
        panel_outputs
    )

    make_figure(
        panel_outputs
    )
