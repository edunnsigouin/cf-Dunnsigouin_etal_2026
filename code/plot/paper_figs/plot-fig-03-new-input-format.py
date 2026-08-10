"""
Create a 2 x 2 panel extreme-value figure comparing observations with one
selected UNSEEN model sample.

Default layout
--------------
(a) August, M = 1 year
(b) August, M = 10 years
(c) May,    M = 1 year
(d) May,    M = 10 years

The two calendar months, the two M-year horizons, the x-axis mode, the model
data method, and the observational reference dataset are user inputs.

Model input
-----------
The model input is the compact monthly-maximum sample produced by scripts 2
and 3. The supported model-data methods are:

    raw
    mm
    q
    ld
    doy
    q_doy

The raw compact filename is:

    monthly_max_samples_<variable>_<N>dayacc_<catchment>_
    lead<full>_split<...>_<forecast_start>_<forecast_end>.nc

Bias-corrected files use the same compact structure and variable names, but
the correction method and reference dataset are encoded in the filename:

    ..._bc_mm_<reference>.nc
    ..._bc_q_<reference>.nc
    ..._bc_ld_<reference>.nc
    ..._bc_doy_<reference>.nc
    ..._bc_q_doy_<reference>.nc

The compact precipitation variables are:

    full sample:
        tp24_max(number, i_date)

    lead-location split:
        tp24_max_lead<start>_<end>(number, i_date)

Calendar-month membership is stored in month(i_date). Therefore one monthly
model sample is obtained by selecting i_date rows for the requested month and
pooling all finite values across number and i_date.

The corrected files keep exactly the same precipitation variable names as the
raw compact file. Bias correction is identified by the filename, not by a
suffix on the variable name.

Reference dataset
-----------------
REFERENCE_DATASET selects both:
    1. the observational dataset plotted in the figure; and
    2. the reference encoded in the selected bias-corrected model filename.

For MODEL_DATA_METHOD = "raw", the reference choice affects only the
observational dataset.

Extreme-value calculation
-------------------------
Each calendar month is fitted only once for each dataset. The fitted return-
level curve is then displayed either against return period or against the
probability of at least one exceedance in M independent years:

    p_M = 1 - (1 - 1 / T) ** M

Storm Hans is always the August 2023 observational value. Each panel also
shows the observational record for that panel's calendar month over
RECORD_START_YEAR-RECORD_END_YEAR.

Important
---------
SciPy's ``genextreme`` shape parameter ``c`` has the opposite sign from the
conventional GEV shape parameter xi:

    xi = -c
"""

# =============================================================================
# Imports
# =============================================================================

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
# User-defined input parameters
# =============================================================================

# -----------------------------------------------------------------------------
# Panel configuration
# -----------------------------------------------------------------------------

# The two columns of the 2 x 2 figure.
# 1 = January, ..., 5 = May, ..., 8 = August, ..., 12 = December.
PANEL_MONTHS = [
    8,
    5,
]

# The two rows of the 2 x 2 figure.
PANEL_M_YEARS = [
    1,
    10,
]

# Options:
#     "return_period" -> return period in years
#     "aep"           -> probability of at least one exceedance in M years
X_AXIS_MODE = "aep"


# -----------------------------------------------------------------------------
# Observational reference dataset
# -----------------------------------------------------------------------------

# Options:
#     "senorge"
#     "era5"
REFERENCE_DATASET = "senorge"

CATCHMENT = "regine_drammen"
X_DAYS = 2

OBSERVATION_YEARS = [
    "1957",
    "2023",
]

# If True, 2023 is read but excluded from each observational fit.
EXCLUDE_2023_FROM_FIT = True

SENORGE_VARIABLE = "rr"
if EXCLUDE_2023_FROM_FIT == False:
    SENORGE_LABEL = f"SeNorge {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
else:
    SENORGE_LABEL = f"SeNorge {OBSERVATION_YEARS[0]}-2022"
    
ERA5_VARIABLE = "tp24"
ERA5_LABEL = f"ERA5 {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[-1]}"
ERA5_GRID = "0.5x0.5"


# -----------------------------------------------------------------------------
# UNSEEN compact model sample
# -----------------------------------------------------------------------------

MODEL_VARIABLE = "tp24"

FORECAST_DATE_RANGE = [
    "2020-01-02",
    "2022-12-29",
]

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Which compact sample variable to use.
#
# Options:
#     "full"
#     "split1"
#     "split2"
#     ...
MODEL_SAMPLING_GROUP = "full"

# Model-data / bias-correction method.
#
# Options:
#     "raw"   -> uncorrected compact monthly-maximum sample
#     "mm"    -> monthly-mean correction from script 2
#     "q"     -> quantile correction
#     "ld"    -> lead-day correction
#     "doy"   -> day-of-year correction
#     "q_doy" -> combined quantile/day-of-year correction
MODEL_DATA_METHOD = "raw"

# Optional explicit compact model filename. Leave as None to construct the
# filename automatically from the settings above.
MODEL_FILENAME_OVERRIDE = None


# -----------------------------------------------------------------------------
# Extreme-value distribution
# -----------------------------------------------------------------------------

# Options:
#     1 -> GEV
#     2 -> Gumbel
#     3 -> GenEx
EXTREME_VALUE_DISTRIBUTION = 2


# -----------------------------------------------------------------------------
# Reference events
# -----------------------------------------------------------------------------

RECORD_START_YEAR = 1957
RECORD_END_YEAR = 2022

STORM_HANS_LINESTYLE = "--"
RECORD_LINESTYLE = ":"


# -----------------------------------------------------------------------------
# Return-period and bootstrap settings
# -----------------------------------------------------------------------------

MIN_RETURN_PERIOD = 1.01
MAX_RETURN_PERIOD = 10000000.0
NUMBER_OF_RETURN_PERIODS = 500

NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

# Share only the axis labels, not the underlying x/y scales.
# Shared x labels use a generic description because M differs between columns.
SHARE_X_LABEL = False
SHARE_Y_LABEL = False

# Panel-title options.
# False gives titles such as "a)", "b)", "c)", and "d)".
# True gives titles such as "a) August".
INCLUDE_MONTH_IN_PANEL_TITLE = False

# Y-axis label options.
# True gives, for example:
#     "Maximum August 2-day precipitation [mm]"
# False gives:
#     "Maximum monthly 2-day precipitation [mm]"
#
# Month-specific labels cannot be shared across panels when PANEL_MONTHS
# contains different months. In that case set SHARE_Y_LABEL = False.
INCLUDE_MONTH_IN_Y_LABEL = True

FIG_WIDTH_IN = 14
FIG_HEIGHT_IN = 10

TITLE_FONTSIZE = 14
AXIS_LABELSIZE = 12
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 11
ANNOTATION_FONTSIZE = 11

OBSERVATION_COLOR = "tab:blue"
RAW_UNSEEN_COLOR = "goldenrod"
BIAS_CORRECTED_UNSEEN_COLOR = "forestgreen"

# Horizontal reference-event colors. These colors are also used for the
# corresponding legend entries.
STORM_HANS_COLOR = "grey"
RECORD_COLOR = "grey"

CONFIDENCE_ALPHA = 0.2

CURVE_LINEWIDTH = 2
REFERENCE_LINEWIDTH = 2
RECORD_LINEWIDTH = 2
MARKER_SIZE = 35
MARKER_LINEWIDTH = 1.0

# X-axis limits for return-period mode [years].
XMIN_RETURN_PERIOD = 1.0
XMAX_RETURN_PERIOD = 1.0e6

# X-axis limits for AEP mode [%].
# For example, 0.01 means 0.01%, not a fractional probability of 0.01.
XMIN_AEP = 0.0001
XMAX_AEP = 100.0

YMIN = 0.0
YMAX = 200

WRITE_TO_FILE = False
FIGURE_DPI = 300


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
    """Check user-defined settings before any calculations."""

    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError(
            "REFERENCE_DATASET must be 'senorge' or 'era5'."
        )

    if X_AXIS_MODE not in {"return_period", "aep"}:
        raise ValueError(
            "X_AXIS_MODE must be 'return_period' or 'aep'."
        )

    valid_model_methods = {
        "raw",
        "mm",
        "q",
        "ld",
        "doy",
        "q_doy",
    }

    if MODEL_DATA_METHOD not in valid_model_methods:
        raise ValueError(
            f"MODEL_DATA_METHOD must be one of "
            f"{sorted(valid_model_methods)}."
        )

    if len(PANEL_MONTHS) != 2:
        raise ValueError(
            "PANEL_MONTHS must contain exactly two calendar months."
        )

    if len(PANEL_M_YEARS) != 2:
        raise ValueError(
            "PANEL_M_YEARS must contain exactly two time horizons."
        )

    if (
        SHARE_Y_LABEL
        and INCLUDE_MONTH_IN_Y_LABEL
        and len(set(PANEL_MONTHS)) > 1
    ):
        raise ValueError(
            "SHARE_Y_LABEL cannot be True when "
            "INCLUDE_MONTH_IN_Y_LABEL is True and PANEL_MONTHS contains "
            "different months. Set SHARE_Y_LABEL = False or "
            "INCLUDE_MONTH_IN_Y_LABEL = False."
        )

    for month in PANEL_MONTHS:
        if not isinstance(month, int) or month not in range(1, 13):
            raise ValueError(
                "Each entry in PANEL_MONTHS must be an integer from 1 to 12."
            )

    for m_years in PANEL_M_YEARS:
        if not isinstance(m_years, int) or m_years < 1:
            raise ValueError(
                "Each entry in PANEL_M_YEARS must be a positive integer."
            )

    if EXTREME_VALUE_DISTRIBUTION not in {1, 2, 3}:
        raise ValueError(
            "EXTREME_VALUE_DISTRIBUTION must be 1, 2, or 3."
        )

    if X_DAYS < 1:
        raise ValueError(
            "X_DAYS must be at least 1."
        )

    if FIRST_INPUT_LEAD > LAST_INPUT_LEAD:
        raise ValueError(
            "FIRST_INPUT_LEAD must not exceed LAST_INPUT_LEAD."
        )

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1

    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError(
            "X_DAYS is too large for the requested input lead window."
        )

    number_of_usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1

    if not isinstance(NUMBER_OF_LEAD_BINS, int):
        raise TypeError(
            "NUMBER_OF_LEAD_BINS must be an integer."
        )

    if (
        NUMBER_OF_LEAD_BINS < 1
        or NUMBER_OF_LEAD_BINS > number_of_usable_leads
    ):
        raise ValueError(
            "NUMBER_OF_LEAD_BINS must be between 1 and the number "
            "of usable accumulated leads."
        )

    valid_groups = {"full"}
    valid_groups.update(
        {
            f"split{bin_number}"
            for bin_number in range(1, NUMBER_OF_LEAD_BINS + 1)
        }
    )

    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(
            f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}."
        )

    if MIN_RETURN_PERIOD <= 1:
        raise ValueError(
            "MIN_RETURN_PERIOD must be greater than 1."
        )

    if MAX_RETURN_PERIOD <= MIN_RETURN_PERIOD:
        raise ValueError(
            "MAX_RETURN_PERIOD must exceed MIN_RETURN_PERIOD."
        )

    if XMIN_RETURN_PERIOD <= 0:
        raise ValueError(
            "XMIN_RETURN_PERIOD must be greater than 0 for a logarithmic axis."
        )

    if XMAX_RETURN_PERIOD <= XMIN_RETURN_PERIOD:
        raise ValueError(
            "XMAX_RETURN_PERIOD must exceed XMIN_RETURN_PERIOD."
        )

    if XMIN_AEP <= 0:
        raise ValueError(
            "XMIN_AEP must be greater than 0 for a logarithmic axis."
        )

    if XMAX_AEP <= XMIN_AEP:
        raise ValueError(
            "XMAX_AEP must exceed XMIN_AEP."
        )

    if XMAX_AEP > 100:
        raise ValueError(
            "XMAX_AEP cannot exceed 100%."
        )

    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError(
            "NUMBER_OF_BOOTSTRAPS must be at least 1."
        )

    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError(
            "CONFIDENCE_LEVEL must lie between 0 and 1."
        )


def get_distribution_name():
    """Return the selected distribution name."""

    return {
        1: "GEV",
        2: "Gumbel",
        3: "GenEx",
    }[EXTREME_VALUE_DISTRIBUTION]


def get_reference_variable():
    """Return the selected observational variable name."""

    if REFERENCE_DATASET == "senorge":
        return SENORGE_VARIABLE

    return ERA5_VARIABLE


def get_reference_name():
    """Return the publication-style reference name."""

    return {
        "senorge": "SeNorge",
        "era5": "ERA5",
    }[REFERENCE_DATASET]


def get_reference_label():
    """Return the reference label including its year range."""

    if REFERENCE_DATASET == "senorge":
        return SENORGE_LABEL

    return ERA5_LABEL


def model_is_raw():
    """Return True when the selected compact model sample is uncorrected."""

    return MODEL_DATA_METHOD == "raw"


def get_model_method_label():
    """Return a short publication-style label for the selected model sample."""

    labels = {
        "raw": "Model raw",
        "mm": "Model BC (MM)",
        "q": "Model BC (Q)",
        "ld": "Model BC (LD)",
        "doy": "Model BC (DOY)",
        "q_doy": "Model BC (Q-DOY)",
    }

    return labels[MODEL_DATA_METHOD]


def get_model_file_id(
    catchment_name,
):
    """Return the short catchment identifier used in compact sample filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


# =============================================================================
# UNSEEN lead configuration
# =============================================================================

def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """Split usable accumulated ending leads into near-equal bins."""

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
    """Return the configured UNSEEN lead bins."""

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1

    return split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=LAST_INPUT_LEAD,
        number_of_bins=NUMBER_OF_LEAD_BINS,
    )


def get_full_lead_range():
    """Return the complete usable lead range."""

    return (
        FIRST_INPUT_LEAD + X_DAYS - 1,
        LAST_INPUT_LEAD,
    )


def get_selected_model_lead_range():
    """Return the lead range for MODEL_SAMPLING_GROUP."""

    if MODEL_SAMPLING_GROUP == "full":
        return get_full_lead_range()

    split_number = int(
        MODEL_SAMPLING_GROUP.replace("split", "")
    )

    return build_lead_bins()[split_number - 1]


def get_model_variable():
    """
    Return the selected compact precipitation variable.

    Raw and corrected compact files preserve the same variable names.
    """

    if MODEL_SAMPLING_GROUP == "full":
        return "tp24_max"

    lead_start, lead_end = get_selected_model_lead_range()

    return (
        f"tp24_max_lead"
        f"{lead_start}_{lead_end}"
    )


def lead_split_filename_label():
    """Return the lead-split label used in the model filename."""

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


# =============================================================================
# Filename helpers
# =============================================================================

def make_reference_filename():
    """Construct the observational filename."""

    if REFERENCE_DATASET == "senorge":
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


def make_model_filename():
    """
    Construct the selected compact monthly-maximum model filename.

    The correction method/reference are encoded in the filename while the
    precipitation variable names remain unchanged.
    """

    if MODEL_FILENAME_OVERRIDE is not None:
        return str(
            MODEL_FILENAME_OVERRIDE
        )

    base_filename = (
        f"monthly_max_samples_"
        f"{MODEL_VARIABLE}_"
        f"{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_"
        f"{lead_split_filename_label()}_"
        f"{FORECAST_DATE_RANGE[0]}_"
        f"{FORECAST_DATE_RANGE[1]}"
    )

    if MODEL_DATA_METHOD == "raw":
        filename = (
            f"{base_filename}.nc"
        )
    else:
        filename = (
            f"{base_filename}_"
            f"bc_{MODEL_DATA_METHOD}_"
            f"{REFERENCE_DATASET}.nc"
        )

    return os.path.join(
        config.dirs["s2s_processed"],
        filename,
    )


def make_figure_filename():
    """Construct the 2 x 2 figure filename."""

    months_label = "-".join(
        MONTH_NAMES[month - 1].lower()
        for month in PANEL_MONTHS
    )

    horizons_label = "-".join(
        f"{m_years}year"
        for m_years in PANEL_M_YEARS
    )

    model_label = (
        MODEL_DATA_METHOD
        if MODEL_DATA_METHOD == "raw"
        else (
            f"{MODEL_DATA_METHOD}_"
            f"{REFERENCE_DATASET}"
        )
    )

    filename = (
        f"fig-03-{X_AXIS_MODE}_"
        f"{model_label}.png"
    )

    return os.path.join(
        config.dirs["fig"],
        filename,
    )


# =============================================================================
# Data reading
# =============================================================================

def check_variable_exists(
    ds,
    variable,
    dataset_name,
):
    """Raise an informative error if a variable is absent."""

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are: {list(ds.data_vars)}"
        )


def read_reference_month(
    filename,
    month,
):
    """
    Read one observational calendar month plus its record and Storm Hans.

    Storm Hans is always the August 2023 observational value, regardless of
    the calendar month represented by the panel.
    """

    variable = get_reference_variable()

    with xr.open_dataset(filename) as ds:
        check_variable_exists(
            ds,
            variable,
            get_reference_label(),
        )

        selected_data = (
            ds[variable]
            .sel(
                year=slice(
                    int(OBSERVATION_YEARS[0]),
                    int(OBSERVATION_YEARS[1]),
                ),
                month=month,
            )
            .load()
        )

        record_data = (
            ds[variable]
            .sel(
                year=slice(
                    RECORD_START_YEAR,
                    RECORD_END_YEAR,
                ),
                month=month,
            )
            .load()
        )

        storm_hans_value = None

        try:
            candidate = float(
                ds[variable]
                .sel(
                    year=2023,
                    month=8,
                )
                .load()
                .values
            )

            if np.isfinite(candidate):
                storm_hans_value = candidate

        except KeyError:
            storm_hans_value = None

    years = np.asarray(
        selected_data["year"].values
    )

    values = np.asarray(
        selected_data.values,
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

    record_finite = np.isfinite(record_values)

    record_value = None
    record_year = None

    if np.any(record_finite):
        finite_record_values = record_values[record_finite]
        finite_record_years = record_years[record_finite]
        record_index = int(np.argmax(finite_record_values))
        record_value = float(finite_record_values[record_index])
        record_year = int(finite_record_years[record_index])

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
            f"Fewer than 10 finite observational values remain for "
            f"{MONTH_NAMES[month - 1]}."
        )

    return {
        "years": years,
        "values": values,
        "fit_years": fit_years,
        "fit_values": fit_values,
        "storm_hans_value": storm_hans_value,
        "record_value": record_value,
        "record_year": record_year,
    }


def read_model_month(
    filename,
    variable,
    month,
    dataset_name,
):
    """
    Read one calendar month from a compact monthly-maximum model sample.

    month(i_date) identifies the calendar month. Finite values are pooled
    across ensemble member (number) and initialization row (i_date).
    """

    with xr.open_dataset(
        filename,
        decode_timedelta=False,
    ) as ds:

        check_variable_exists(
            ds,
            variable,
            dataset_name,
        )

        check_variable_exists(
            ds,
            "month",
            dataset_name,
        )

        if set(
            ds[
                variable
            ].dims
        ) != {
            "number",
            "i_date",
        }:
            raise ValueError(
                f"Variable '{variable}' in {dataset_name} must have "
                "dimensions ('number', 'i_date'). "
                f"Found {ds[variable].dims}."
            )

        if ds[
            "month"
        ].dims != (
            "i_date",
        ):
            raise ValueError(
                f"Variable 'month' in {dataset_name} must have "
                "dimension ('i_date',)."
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
                f"Fewer than 10 finite values were found in {dataset_name} "
                f"for {MONTH_NAMES[month - 1]}."
            )

        print(
            f"{dataset_name}, {MONTH_NAMES[month - 1]}: "
            f"{values.size} finite values."
        )

    return values


# =============================================================================
# Distribution fitting
# =============================================================================

def fit_gev(values):
    """Fit a stationary three-parameter GEV."""

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
    """Fit a stationary two-parameter Gumbel distribution."""

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
    """Negative log-likelihood for the two-parameter GenEx."""

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
    """Fit a two-parameter GenEx distribution."""

    if np.any(values < 0):
        raise ValueError(
            "The GenEx implementation requires non-negative values."
        )

    positive_values = values[values > 0]

    if positive_values.size == 0:
        raise RuntimeError(
            "GenEx cannot be fitted because there are no positive values."
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


def fit_distribution(values):
    """Fit the selected distribution."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        return fit_gev(values)

    if EXTREME_VALUE_DISTRIBUTION == 2:
        return fit_gumbel(values)

    return fit_genex(values)


def genex_ppf(
    probabilities,
    parameters,
):
    """GenEx quantile function."""

    shape, scale = parameters
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return (
        -scale
        * np.log1p(
            -np.power(
                probabilities,
                1.0 / shape,
            )
        )
    )


def calculate_return_levels(
    return_periods,
    fitted_parameters,
):
    """Calculate fitted return levels."""

    probabilities = 1.0 - 1.0 / return_periods

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters

        return genextreme.ppf(
            probabilities,
            shape_c,
            loc=location,
            scale=scale,
        )

    if EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters

        return gumbel_r.ppf(
            probabilities,
            loc=location,
            scale=scale,
        )

    return genex_ppf(
        probabilities,
        fitted_parameters,
    )


def generate_random_sample(
    fitted_parameters,
    sample_size,
    rng,
):
    """Generate one parametric-bootstrap sample."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters

        return genextreme.rvs(
            shape_c,
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    if EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters

        return gumbel_r.rvs(
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    uniforms = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=sample_size,
    )

    return genex_ppf(
        uniforms,
        fitted_parameters,
    )


def calculate_empirical_return_periods(values):
    """Calculate empirical Weibull return periods."""

    sorted_values = np.sort(values)[::-1]

    ranks = np.arange(
        1,
        sorted_values.size + 1,
    )

    return_periods = (
        sorted_values.size + 1
    ) / ranks

    return return_periods, sorted_values


def calculate_event_return_period(
    event_value,
    fitted_parameters,
):
    """Calculate the fitted return period corresponding to one event."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters

        exceedance_probability = genextreme.sf(
            event_value,
            shape_c,
            loc=location,
            scale=scale,
        )

    elif EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters

        exceedance_probability = gumbel_r.sf(
            event_value,
            loc=location,
            scale=scale,
        )

    else:
        shape, scale = fitted_parameters

        if event_value < 0:
            exceedance_probability = 1.0
        else:
            cdf = (
                1.0
                - np.exp(
                    -event_value / scale
                )
            ) ** shape

            exceedance_probability = 1.0 - cdf

    if (
        not np.isfinite(exceedance_probability)
        or exceedance_probability <= 0
    ):
        return np.inf

    return 1.0 / exceedance_probability


def calculate_m_year_exceedance_probability(
    return_period,
    m_years,
):
    """
    Convert a fitted return period to the probability of at least one
    exceedance in m_years independent years.
    """

    if (
        not np.isfinite(return_period)
        or return_period <= 0
    ):
        return 0.0

    annual_exceedance_probability = 1.0 / return_period

    return float(
        -np.expm1(
            m_years
            * np.log1p(
                -annual_exceedance_probability
            )
        )
    )


def format_return_period(
    return_period,
):
    """Format a finite or infinite return period for console output."""

    if np.isinf(return_period):
        return "infinite"

    return f"{return_period:.2f} years"


def print_panel_event_statistics(
    panel_label,
    month,
    m_years,
    reference_data,
    observation_analysis,
    raw_analysis,
    bias_corrected_analysis,
):
    """Print fitted return periods and M-year AEPs for one panel."""

    print()
    print(
        f"Panel {panel_label}) — "
        f"{MONTH_NAMES[month - 1]}, M = {m_years}"
    )
    print(
        "-" * 60
    )

    events = [
        (
            "Storm Hans, August 2023",
            reference_data["storm_hans_value"],
        ),
        (
            (
                f"{MONTH_NAMES[month - 1]} record "
                f"{RECORD_START_YEAR}-{RECORD_END_YEAR}"
            ),
            reference_data["record_value"],
        ),
    ]

    fitted_datasets = [
        (
            get_reference_label(),
            observation_analysis,
        ),
    ]

    if raw_analysis is not None:
        fitted_datasets.append(
            (
                "Model raw",
                raw_analysis,
            )
        )

    if bias_corrected_analysis is not None:
        fitted_datasets.append(
            (
                get_model_method_label(),
                bias_corrected_analysis,
            )
        )

    for event_label, event_value in events:
        if event_value is None:
            print()
            print(
                f"{event_label}: unavailable"
            )
            continue

        print()
        print(
            f"{event_label}: {event_value:.3f} mm"
        )

        for dataset_label, analysis in fitted_datasets:
            return_period = calculate_event_return_period(
                event_value=event_value,
                fitted_parameters=analysis["fitted_parameters"],
            )

            m_year_aep = calculate_m_year_exceedance_probability(
                return_period=return_period,
                m_years=m_years,
            )

            print(
                f"  {dataset_label}: "
                f"return period = {format_return_period(return_period)}; "
                f"{m_years}-year AEP = {100.0 * m_year_aep:.3f}%"
            )


# =============================================================================
# Bootstrap and analysis
# =============================================================================

def make_return_period_grid():
    """Create a logarithmic return-period grid."""

    return np.geomspace(
        MIN_RETURN_PERIOD,
        MAX_RETURN_PERIOD,
        NUMBER_OF_RETURN_PERIODS,
    )


def parametric_bootstrap_return_levels(
    sample_values,
    fitted_parameters,
    return_periods,
    random_seed,
):
    """Estimate confidence limits for one fitted curve."""

    rng = np.random.default_rng(random_seed)
    sample_size = sample_values.size

    bootstrap_levels = np.full(
        (
            NUMBER_OF_BOOTSTRAPS,
            return_periods.size,
        ),
        np.nan,
        dtype=float,
    )

    successful_fits = 0

    for bootstrap_number in range(NUMBER_OF_BOOTSTRAPS):
        simulated_values = generate_random_sample(
            fitted_parameters=fitted_parameters,
            sample_size=sample_size,
            rng=rng,
        )

        try:
            bootstrap_parameters = fit_distribution(
                simulated_values
            )

            levels = calculate_return_levels(
                return_periods,
                bootstrap_parameters,
            )

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ):
            continue

        if not np.isfinite(levels).all():
            continue

        bootstrap_levels[
            bootstrap_number,
            :,
        ] = levels

        successful_fits += 1

    minimum_successful_fits = int(
        0.90 * NUMBER_OF_BOOTSTRAPS
    )

    if successful_fits < minimum_successful_fits:
        raise RuntimeError(
            f"Only {successful_fits} of {NUMBER_OF_BOOTSTRAPS} "
            f"bootstrap fits succeeded."
        )

    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = np.nanpercentile(
        bootstrap_levels,
        100.0 * alpha / 2.0,
        axis=0,
    )

    upper = np.nanpercentile(
        bootstrap_levels,
        100.0 * (1.0 - alpha / 2.0),
        axis=0,
    )

    return lower, upper, successful_fits


def analyse_distribution(
    values,
    return_periods,
    random_seed,
):
    """Fit and summarize one sample."""

    parameters = fit_distribution(values)

    fitted_levels = calculate_return_levels(
        return_periods,
        parameters,
    )

    lower, upper, successful_fits = (
        parametric_bootstrap_return_levels(
            sample_values=values,
            fitted_parameters=parameters,
            return_periods=return_periods,
            random_seed=random_seed,
        )
    )

    empirical_return_periods, empirical_values = (
        calculate_empirical_return_periods(values)
    )

    return {
        "values": values,
        "fitted_parameters": parameters,
        "fitted_return_levels": fitted_levels,
        "lower_confidence_limit": lower,
        "upper_confidence_limit": upper,
        "successful_bootstraps": successful_fits,
        "empirical_return_periods": empirical_return_periods,
        "empirical_values": empirical_values,
    }


# =============================================================================
# Axis transformation and formatting
# =============================================================================

def convert_return_periods_to_plot_x(
    return_periods,
    m_years,
):
    """Convert return periods to the selected panel x-coordinate."""

    return_periods = np.asarray(
        return_periods,
        dtype=float,
    )

    if X_AXIS_MODE == "return_period":
        return return_periods

    annual_probability = 1.0 / return_periods

    m_year_probability = -np.expm1(
        m_years
        * np.log1p(
            -annual_probability
        )
    )

    return 100.0 * m_year_probability


def get_panel_x_axis_label(m_years):
    """Return the x-axis label for one panel."""

    if X_AXIS_MODE == "return_period":
        return "Return period [years]"

    return f"{m_years}-year exceedance probability [%]"


def get_shared_x_axis_label():
    """Return a generic shared x-axis label."""

    if X_AXIS_MODE == "return_period":
        return "Return period [years]"

    return "M-year exceedance probability [%]"


def get_y_axis_label(
    month=None,
):
    """Return a generic or calendar-month-specific precipitation label."""

    if INCLUDE_MONTH_IN_Y_LABEL:
        if month is None:
            raise ValueError(
                "A calendar month is required when "
                "INCLUDE_MONTH_IN_Y_LABEL is True."
            )

        return (
            f"Maximum {MONTH_NAMES[month - 1]} "
            f"{X_DAYS}-day precipitation [mm]"
        )

    return f"Maximum monthly {X_DAYS}-day precipitation [mm]"


def format_x_axis(
    ax,
    m_years,
):
    """Format one panel's x-axis."""

    ax.set_xscale("log")

    if X_AXIS_MODE == "return_period":
        ax.set_xlim(
            XMIN_RETURN_PERIOD,
            XMAX_RETURN_PERIOD,
        )

    else:
        # AEP decreases as return period increases, so reverse the axis.
        ax.set_xlim(
            XMAX_AEP,
            XMIN_AEP,
        )

        def percent_formatter(x, pos):
            if x <= 0:
                return ""

            return f"{x:g}"

        ax.xaxis.set_major_formatter(
            FuncFormatter(percent_formatter)
        )


# =============================================================================
# Plotting
# =============================================================================

def plot_panel(
    ax,
    panel_label,
    month,
    m_years,
    return_periods,
    reference_data,
    observation_analysis,
    raw_analysis,
    bias_corrected_analysis,
    show_legend,
):
    """Plot one publication-style panel."""

    plot_x = convert_return_periods_to_plot_x(
        return_periods,
        m_years,
    )

    observation_empirical_x = convert_return_periods_to_plot_x(
        observation_analysis["empirical_return_periods"],
        m_years,
    )

    ax.fill_between(
        plot_x,
        observation_analysis["lower_confidence_limit"],
        observation_analysis["upper_confidence_limit"],
        color=OBSERVATION_COLOR,
        alpha=CONFIDENCE_ALPHA,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        plot_x,
        observation_analysis["fitted_return_levels"],
        color=OBSERVATION_COLOR,
        linewidth=CURVE_LINEWIDTH,
        zorder=4,
    )

    ax.scatter(
        observation_empirical_x,
        observation_analysis["empirical_values"],
        facecolors="none",
        edgecolors=OBSERVATION_COLOR,
        linewidths=MARKER_LINEWIDTH,
        s=MARKER_SIZE,
        zorder=5,
    )

    if raw_analysis is not None:
        raw_empirical_x = convert_return_periods_to_plot_x(
            raw_analysis["empirical_return_periods"],
            m_years,
        )

        ax.fill_between(
            plot_x,
            raw_analysis["lower_confidence_limit"],
            raw_analysis["upper_confidence_limit"],
            color=RAW_UNSEEN_COLOR,
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            plot_x,
            raw_analysis["fitted_return_levels"],
            color=RAW_UNSEEN_COLOR,
            linewidth=CURVE_LINEWIDTH,
            zorder=4,
        )

        ax.scatter(
            raw_empirical_x,
            raw_analysis["empirical_values"],
            facecolors="none",
            edgecolors=RAW_UNSEEN_COLOR,
            linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE,
            zorder=2,
        )

    if bias_corrected_analysis is not None:
        bias_corrected_empirical_x = convert_return_periods_to_plot_x(
            bias_corrected_analysis["empirical_return_periods"],
            m_years,
        )

        ax.fill_between(
            plot_x,
            bias_corrected_analysis["lower_confidence_limit"],
            bias_corrected_analysis["upper_confidence_limit"],
            color=BIAS_CORRECTED_UNSEEN_COLOR,
            alpha=CONFIDENCE_ALPHA,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            plot_x,
            bias_corrected_analysis["fitted_return_levels"],
            color=BIAS_CORRECTED_UNSEEN_COLOR,
            linewidth=CURVE_LINEWIDTH,
            zorder=4,
        )

        ax.scatter(
            bias_corrected_empirical_x,
            bias_corrected_analysis["empirical_values"],
            facecolors="none",
            edgecolors=BIAS_CORRECTED_UNSEEN_COLOR,
            linewidths=MARKER_LINEWIDTH,
            s=MARKER_SIZE,
            alpha=0.55,
            zorder=2,
        )

    storm_hans_value = reference_data["storm_hans_value"]
    record_value = reference_data["record_value"]

    if storm_hans_value is not None:
        ax.axhline(
            storm_hans_value,
            color=STORM_HANS_COLOR,
            linestyle=STORM_HANS_LINESTYLE,
            linewidth=REFERENCE_LINEWIDTH,
            zorder=2,
        )

    if record_value is not None:
        ax.axhline(
            record_value,
            color=RECORD_COLOR,
            linestyle=RECORD_LINESTYLE,
            linewidth=RECORD_LINEWIDTH,
            zorder=2,
        )

    format_x_axis(
        ax,
        m_years,
    )

    ax.set_ylim(
        bottom=YMIN,
        top=YMAX,
    )

    if not SHARE_X_LABEL:
        ax.set_xlabel(
            get_panel_x_axis_label(m_years),
            fontsize=AXIS_LABELSIZE,
        )

    if not SHARE_Y_LABEL:
        ax.set_ylabel(
            get_y_axis_label(
                month=month,
            ),
            fontsize=AXIS_LABELSIZE,
        )

    if INCLUDE_MONTH_IN_PANEL_TITLE:
        panel_title = (
            f"{panel_label}) "
            f"{MONTH_NAMES[month - 1]}"
        )
    else:
        panel_title = f"{panel_label})"

    ax.set_title(
        panel_title,
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=5,
    )

    #if X_AXIS_MODE == "aep":
        #ax.text(
        #    0.98,
        #    0.97,
        #    f"M = {m_years} year" if m_years == 1 else f"M = {m_years} years",
        #    transform=ax.transAxes,
        #    ha="right",
        #    va="top",
        #    fontsize=ANNOTATION_FONTSIZE,
        #)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if show_legend:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=OBSERVATION_COLOR,
                linewidth=CURVE_LINEWIDTH,
                label=get_reference_label(),
            )
        ]

        if raw_analysis is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=RAW_UNSEEN_COLOR,
                    linewidth=CURVE_LINEWIDTH,
                    label="Model raw",
                )
            )

        if bias_corrected_analysis is not None:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=BIAS_CORRECTED_UNSEEN_COLOR,
                    linewidth=CURVE_LINEWIDTH,
                    label=get_model_method_label(),
                )
            )

        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=STORM_HANS_COLOR,
                    linestyle=STORM_HANS_LINESTYLE,
                    linewidth=REFERENCE_LINEWIDTH,
                    label=f"{get_reference_name()} Storm Hans, August 2023",
                ),
                Line2D(
                    [0],
                    [0],
                    color=RECORD_COLOR,
                    linestyle=RECORD_LINESTYLE,
                    linewidth=RECORD_LINEWIDTH,
                    label=(
                        f"{get_reference_name()} calendar-month record "
                        f"{RECORD_START_YEAR}-{RECORD_END_YEAR}"
                    ),
                ),
            ]
        )

        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            handlelength=2.5,
            borderaxespad=0.4,
            labelspacing=0.4,
        )


def plot_figure(
    return_periods,
    month_results,
    filename_out,
):
    """Create the 2 x 2 publication-quality figure."""

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        ),
        constrained_layout=True,
    )

    panel_labels = ["a", "b", "c", "d"]
    panel_number = 0

    # Month-major order:
    # a) first month, first M
    # b) first month, second M
    # c) second month, first M
    # d) second month, second M
    for row_index, month in enumerate(PANEL_MONTHS):
        for column_index, m_years in enumerate(PANEL_M_YEARS):
            result = month_results[month]

            plot_panel(
                ax=axes[row_index, column_index],
                panel_label=panel_labels[panel_number],
                month=month,
                m_years=m_years,
                return_periods=return_periods,
                reference_data=result["reference_data"],
                observation_analysis=result["observation_analysis"],
                raw_analysis=result["raw_analysis"],
                bias_corrected_analysis=result[
                    "bias_corrected_analysis"
                ],
                show_legend=panel_number == 0,
            )

            panel_number += 1

    if SHARE_X_LABEL:
        fig.supxlabel(
            get_shared_x_axis_label(),
            fontsize=AXIS_LABELSIZE,
        )

    if SHARE_Y_LABEL:
        shared_y_month = (
            PANEL_MONTHS[0]
            if INCLUDE_MONTH_IN_Y_LABEL
            else None
        )

        fig.supylabel(
            get_y_axis_label(
                month=shared_y_month,
            ),
            fontsize=AXIS_LABELSIZE,
        )

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
    """Run the two-month, two-horizon analysis."""

    validate_user_settings()

    filename_reference = make_reference_filename()
    filename_model = make_model_filename()
    filename_out = make_figure_filename()

    model_variable = get_model_variable()

    if not os.path.isfile(
        filename_model
    ):
        raise FileNotFoundError(
            f"Selected compact model file not found: {filename_model}"
        )

    print(
        f"Reference dataset:             {get_reference_label()}"
    )
    print(
        f"Reference file:                {filename_reference}"
    )
    print(
        f"Model-data method:             {MODEL_DATA_METHOD}"
    )
    print(
        f"Model file:                    {filename_model}"
    )
    print(
        f"Model variable:                {model_variable}"
    )

    print(
        f"Months:                        "
        f"{', '.join(MONTH_NAMES[m - 1] for m in PANEL_MONTHS)}"
    )
    print(
        f"M-year horizons:               {PANEL_M_YEARS}"
    )
    print(
        f"X-axis mode:                   {X_AXIS_MODE}"
    )
    print(
        f"Distribution:                  {get_distribution_name()}"
    )

    return_periods = make_return_period_grid()
    month_results = {}

    for month_index, month in enumerate(PANEL_MONTHS):
        print()
        print(
            f"Analysing {MONTH_NAMES[month - 1]}"
        )

        reference_data = read_reference_month(
            filename_reference,
            month,
        )

        observation_analysis = analyse_distribution(
            values=reference_data["fit_values"],
            return_periods=return_periods,
            random_seed=RANDOM_SEED + 20 * month_index,
        )

        model_values = read_model_month(
            filename=filename_model,
            variable=model_variable,
            month=month,
            dataset_name=(
                f"UNSEEN model dataset "
                f"({MODEL_DATA_METHOD})"
            ),
        )

        model_analysis = analyse_distribution(
            values=model_values,
            return_periods=return_periods,
            random_seed=RANDOM_SEED + 20 * month_index + 1,
        )

        if model_is_raw():
            raw_analysis = model_analysis
            bias_corrected_analysis = None
        else:
            raw_analysis = None
            bias_corrected_analysis = model_analysis

        month_results[month] = {
            "reference_data": reference_data,
            "observation_analysis": observation_analysis,
            "raw_analysis": raw_analysis,
            "bias_corrected_analysis": bias_corrected_analysis,
        }

        print(
            f"Observation sample size:       "
            f"{observation_analysis['values'].size}"
        )

        print(
            f"{get_model_method_label()} sample size: "
            f"{model_analysis['values'].size}"
        )

        print(
            f"{MONTH_NAMES[month - 1]} record "
            f"{RECORD_START_YEAR}-{RECORD_END_YEAR}: "
            f"{reference_data['record_value']:.3f} mm "
            f"({reference_data['record_year']})"
        )

        if reference_data["storm_hans_value"] is not None:
            print(
                f"Storm Hans August 2023:        "
                f"{reference_data['storm_hans_value']:.3f} mm"
            )

    panel_labels = [
        "a",
        "b",
        "c",
        "d",
    ]

    panel_number = 0

    # Use the same month-major order as the plotted panels:
    # a) first month, first M
    # b) first month, second M
    # c) second month, first M
    # d) second month, second M
    for month in PANEL_MONTHS:
        for m_years in PANEL_M_YEARS:
            result = month_results[month]

            print_panel_event_statistics(
                panel_label=panel_labels[panel_number],
                month=month,
                m_years=m_years,
                reference_data=result["reference_data"],
                observation_analysis=result["observation_analysis"],
                raw_analysis=result["raw_analysis"],
                bias_corrected_analysis=result[
                    "bias_corrected_analysis"
                ],
            )

            panel_number += 1

    plot_figure(
        return_periods=return_periods,
        month_results=month_results,
        filename_out=filename_out,
    )


if __name__ == "__main__":
    main()
