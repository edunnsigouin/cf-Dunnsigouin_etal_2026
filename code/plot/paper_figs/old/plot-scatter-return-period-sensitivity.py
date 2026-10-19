#!/usr/bin/env python3
"""
Plot one-panel AEP or return-period estimates for two observational references,
raw UNSEEN, and all supported bias-correction methods.

Calendar-month modes
--------------------
CALENDAR_MONTH controls whether the analysis is for one calendar month or for
the chance of an exceedance in ANY month of a year:

    CALENDAR_MONTH = 1, ..., 12
        Analyse one calendar month exactly as before.

    CALENDAR_MONTH = 0
        Analyse all 12 calendar months separately and combine their monthly
        exceedance probabilities into one annual "any month" probability.

For one estimation method and one dataset, let p_m be the probability that the
event value is exceeded in calendar month m during a year. The probability
that month m does NOT exceed the event value is:

    1 - p_m

Assuming exceedances in different calendar months are independent, the
probability that none of the 12 months exceeds the event value is:

    product over m=1,...,12 of (1 - p_m)

Therefore the probability of at least one exceedance somewhere in the year is:

    p_any = 1 - product over m=1,...,12 of (1 - p_m)

The corresponding return period is:

    T_any = 1 / p_any

This is preferable to simply adding monthly probabilities because the product
formula correctly accounts for the possibility that more than one month could
exceed the event value in the same year.

Important assumption
--------------------
The equation above assumes that exceedance events in different calendar months
are independent. In plain language, an exceedance in one month must not change
the probability of an exceedance in another month. This is an approximation.
Monthly precipitation extremes are often much less dependent than daily values,
but adjacent months can still share some climate-scale dependence. The annual
"any month" result should therefore be interpreted as an independence-based
combination of the 12 monthly probabilities.

Event-value handling in CALENDAR_MONTH = 0 mode
-----------------------------------------------
EVENT_THRESHOLD = "storm_hans":
    Each reference dataset supplies one Storm Hans value from August 2023.
    That same reference-specific value is tested against all 12 monthly
    distributions before their probabilities are combined.

EVENT_THRESHOLD = "calendar_record":
    Each calendar month uses its own historical record over
    RECORD_START_YEAR-RECORD_END_YEAR. The combined event therefore means:
    "at least one month exceeds that month's historical calendar-month record."

Figure design
-------------
The x-axis contains:

    Reference
    raw
    mm
    q
    doy
    ld
    q_doy

At every x-axis entry there are two horizontally separated stacks:

    SeNorge at x - REFERENCE_X_DELTA / 2
    ERA5    at x + REFERENCE_X_DELTA / 2

For each reference stack, four extreme-value estimation methods are plotted
using different marker shapes:

    GEV
    Gumbel
    GenEx
    Empirical

The marker face color and edge color are user inputs for each reference
dataset. A face color can be a valid Matplotlib color or the literal string
"empty" for an unfilled marker.

Model input
-----------
The compact UNSEEN files contain:

    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)
    month(i_date)

Raw file:

    monthly_max_samples_<variable>_<N>dayacc_<catchment>_
    lead<full>_split<...>_<forecast_start>_<forecast_end>.nc

Bias-corrected files append:

    _bc_<method>_<reference>.nc

where method is mm, q, doy, ld, or q_doy and reference is senorge or era5.

Statistical methods
-------------------
GEV:
    stationary three-parameter generalized extreme-value distribution.

Gumbel:
    stationary two-parameter Gumbel distribution.

GenEx:
    two-parameter generalized exponential distribution fitted by maximum
    likelihood.

Empirical:
    rank-based estimate:
        rank = 1 + number of sample values strictly greater than the event value
        annual exceedance probability = rank / sample size
        return period = sample size / rank

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

# Calendar-month mode.
#
#     0       -> combine January-December probabilities to estimate the chance
#                of at least one exceedance in any month of a year
#     1...12  -> analyse one calendar month only
CALENDAR_MONTH = 8

# Both reference datasets are plotted. Their order also controls which one
# appears on the left/right side of each x-axis label.
REFERENCE_DATASETS = [
    "era5",
    "senorge",
]

# Event threshold.
# Options:
#     "storm_hans"
#     "calendar_record"
EVENT_THRESHOLD = "calendar_record"

# Quantity plotted on the y-axis.
# Options:
#     "aep"
#     "return_period"
PLOT_METRIC = "aep"

# Used only when PLOT_METRIC == "aep".
# 1 gives annual exceedance probability.
N_AEP_YEARS = 1


# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------

CATCHMENT = "regine_drammen"

# Consecutive-day precipitation accumulation.
X_DAYS = 2

# Observational file period. 2023 is needed for Storm Hans.
OBSERVATION_YEARS = [
    "1957",
    "2023",
]

# If True, 2023 is read for the threshold but excluded from the fitted
# observational Reference distribution.
EXCLUDE_2023_FROM_REFERENCE_FIT = True

# Period used to define the calendar-month record threshold.
RECORD_START_YEAR = 1957
RECORD_END_YEAR = 2022

# Compact UNSEEN sample configuration.
FORECAST_DATE_RANGE = [
    "2020-01-02",
    "2022-12-29",
]

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Model sample to use from each compact file.
#
# Options:
#     "full"
#     "split1"
#     "split2"
#     ...
MODEL_SAMPLING_GROUP = "full"

# Optional explicit input paths.
# Leave as None to construct filenames automatically.
REFERENCE_FILENAME_OVERRIDES = {
    "senorge": None,
    "era5": None,
}

# Optional explicit model paths. Keys are (method, reference).
#
# Raw is reference-independent, so only ("raw", None) is used.
# Corrected methods use one file for each correction reference.
MODEL_FILENAME_OVERRIDES = {
    ("raw", None): None,
    ("mm", "senorge"): None,
    ("mm", "era5"): None,
    ("q", "senorge"): None,
    ("q", "era5"): None,
    ("doy", "senorge"): None,
    ("doy", "era5"): None,
    ("ld", "senorge"): None,
    ("ld", "era5"): None,
    ("q_doy", "senorge"): None,
    ("q_doy", "era5"): None,
}


# -----------------------------------------------------------------------------
# Figure appearance
# -----------------------------------------------------------------------------

FIG_WIDTH_IN = 10.5
FIG_HEIGHT_IN = 6.5
FIGURE_DPI = 400

POINT_SIZE = 72
MARKER_EDGE_WIDTH = 1.2

METHOD_MARKERS = {
    "GEV": "s",
    "Gumbel": "v",
    "GenEx": "o",
    "Empirical": "D",
}

# All four methods are plotted at the same x position for each dataset.
# Keep this at 0.0 for exact vertical alignment. Increase it slightly only if
# you temporarily want to separate overlapping markers.
METHOD_X_JITTER = 0.0

# Horizontal separation between SeNorge and ERA5 values at one x-axis entry.
# The two positions are x - delta/2 and x + delta/2, so they remain centered.
REFERENCE_X_DELTA = 0.24

# Marker colors for each reference dataset.
#
# Face colors:
#     - use any valid Matplotlib color, e.g. "tab:red", "white", "#444444"
#     - use the literal string "empty" for an unfilled marker
#
# Edge colors:
#     - use any valid Matplotlib color
#
# Example:
#     SeNorge filled red, ERA5 empty with blue outline.
REFERENCE_MARKER_FACECOLORS = {
    "senorge": "empty",
    "era5": "empty",
}

REFERENCE_MARKER_EDGECOLORS = {
    "senorge": "tab:red",
    "era5": "tab:blue",
}

AXIS_LABELSIZE = 12
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 10

# AEP y-axis limits in percent.
AEP_YMIN_PERCENT = 0.0001
AEP_YMAX_PERCENT = 100.0

# Return-period y-axis limits in years.
RETURN_PERIOD_YMIN_YEARS = 1.0
RETURN_PERIOD_YMAX_YEARS = 1_000_000.0

SHOW_GRID = True

WRITE_TO_FILE = True
SHOW_FIGURE = True


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_VARIABLE = "tp24"

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

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

MODEL_DATASETS = [
    "raw",
    "mm",
    "q",
    "doy",
    "ld",
    "q_doy",
]

PLOT_DATASETS = [
    "reference",
    *MODEL_DATASETS,
]

DISPLAY_LABELS = {
    "reference": "Reference",
    "raw": "raw",
    "mm": "mm",
    "q": "q",
    "doy": "doy",
    "ld": "ld",
    "q_doy": "q_doy",
}

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
# SCRIPT WORKFLOW
# =============================================================================
#
# 1. validate_user_settings()
#       Checks the user settings before any files are opened.
#
# 2. read_reference_data()
#       Reads SeNorge and ERA5 separately and determines each event value.
#
# 3. read_model_month()
#       Reads raw once and each corrected sample for both correction references.
#
# 4. calculate_all_results()
#       Applies all four methods month by month. If CALENDAR_MONTH = 0, it then
#       combines the 12 monthly probabilities into one annual any-month value.
#
# 5. make_figure()
#       Places SeNorge and ERA5 on opposite sides of each dataset label.
#
# The functions below are arranged in roughly this same order.
# =============================================================================


# =============================================================================
# VALIDATION
# =============================================================================

def validate_user_settings():
    """Check the user settings before any files are opened."""

    if CALENDAR_MONTH not in range(0, 13):
        raise ValueError(
            "CALENDAR_MONTH must be 0 (any month) or an integer from 1 to 12."
        )

    if REFERENCE_DATASETS != [
        "era5",
        "senorge",
    ]:
        raise ValueError(
            "REFERENCE_DATASETS must be ['era5', 'senorge'] so both "
            "references are plotted in the expected left/right order."
        )

    if REFERENCE_X_DELTA <= 0:
        raise ValueError(
            "REFERENCE_X_DELTA must be greater than zero."
        )

    for color_mapping_name, color_mapping in [
        (
            "REFERENCE_MARKER_FACECOLORS",
            REFERENCE_MARKER_FACECOLORS,
        ),
        (
            "REFERENCE_MARKER_EDGECOLORS",
            REFERENCE_MARKER_EDGECOLORS,
        ),
    ]:
        if set(
            color_mapping
        ) != set(
            REFERENCE_DATASETS
        ):
            raise ValueError(
                f"{color_mapping_name} must contain exactly the keys "
                f"{REFERENCE_DATASETS}."
            )

    for reference_dataset in REFERENCE_DATASETS:

        facecolor = REFERENCE_MARKER_FACECOLORS[
            reference_dataset
        ]

        if not isinstance(
            facecolor,
            str,
        ):
            raise TypeError(
                "REFERENCE_MARKER_FACECOLORS values must be strings: "
                'a Matplotlib color or "empty".'
            )

    if EVENT_THRESHOLD not in {
        "storm_hans",
        "calendar_record",
    }:
        raise ValueError(
            "EVENT_THRESHOLD must be 'storm_hans' or 'calendar_record'."
        )

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

    if X_DAYS < 1:
        raise ValueError(
            "X_DAYS must be at least 1."
        )

    if FIRST_INPUT_LEAD > LAST_INPUT_LEAD:
        raise ValueError(
            "FIRST_INPUT_LEAD must not exceed LAST_INPUT_LEAD."
        )

    first_usable_lead = (
        FIRST_INPUT_LEAD
        + X_DAYS
        - 1
    )

    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError(
            "X_DAYS is too large for the requested lead window."
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

    if RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError(
            "RECORD_END_YEAR must be >= RECORD_START_YEAR."
        )

    if not (
        0
        < AEP_YMIN_PERCENT
        < AEP_YMAX_PERCENT
        <= 100
    ):
        raise ValueError(
            "AEP y-axis limits must satisfy "
            "0 < AEP_YMIN_PERCENT < AEP_YMAX_PERCENT <= 100."
        )

    if not (
        0
        < RETURN_PERIOD_YMIN_YEARS
        < RETURN_PERIOD_YMAX_YEARS
    ):
        raise ValueError(
            "Return-period y-axis limits must be positive and increasing."
        )


# =============================================================================
# LABELS AND FILENAMES
# =============================================================================

def get_reference_name(
    reference_dataset,
):
    """Return the publication-style name of one observational reference."""

    return {
        "senorge": "SeNorge",
        "era5": "ERA5",
    }[
        reference_dataset
    ]


def get_reference_variable(
    reference_dataset,
):
    """Return the precipitation variable used by one reference dataset."""

    return {
        "senorge": SENORGE_VARIABLE,
        "era5": ERA5_VARIABLE,
    }[
        reference_dataset
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
    """Split the usable accumulated ending leads into near-equal bins."""

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
    """Return the complete usable ending-lead range."""

    return (
        FIRST_INPUT_LEAD
        + X_DAYS
        - 1,
        LAST_INPUT_LEAD,
    )


def get_selected_model_lead_range():
    """Return selected lead range."""

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
    """Return the compact precipitation variable selected by MODEL_SAMPLING_GROUP."""

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
    """Return the lead-bin label used in compact filenames."""

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


def make_reference_filename(
    reference_dataset,
):
    """Construct the observational filename for one reference dataset."""

    override = REFERENCE_FILENAME_OVERRIDES[
        reference_dataset
    ]

    if override is not None:
        return Path(
            override
        )

    if reference_dataset == "senorge":

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


def make_model_filename(
    method,
    reference_dataset=None,
):
    """
    Construct one raw or bias-corrected compact model filename.

    Raw data are reference-independent. Bias-corrected filenames include the
    reference used to calculate the correction.
    """

    override_key = (
        ("raw", None)
        if method == "raw"
        else (
            method,
            reference_dataset,
        )
    )

    override = MODEL_FILENAME_OVERRIDES.get(
        override_key
    )

    if override is not None:
        return Path(
            override
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

    if method == "raw":

        filename = (
            f"{base_filename}.nc"
        )

    else:

        filename = (
            f"{base_filename}_"
            f"bc_{method}_"
            f"{reference_dataset}.nc"
        )

    return (
        Path(
            config.dirs[
                "s2s_processed"
            ]
        )
        / filename
    )


def make_figure_filename():
    """Construct output figure filename."""

    threshold_label = (
        "storm-hans"
        if EVENT_THRESHOLD == "storm_hans"
        else "calendar-record"
    )

    metric_label = (
        f"{N_AEP_YEARS}year-aep"
        if PLOT_METRIC == "aep"
        else "return-period"
    )

    month_label = (
        "any-month"
        if CALENDAR_MONTH == 0
        else MONTH_NAMES[
            CALENDAR_MONTH - 1
        ].lower()
    )

    return (
        Path(
            config.dirs[
                "fig"
            ]
        )
        / (
            f"scatter_{metric_label}_sensitivity_"
            f"{month_label}_"
            f"senorge-era5_"
            f"{threshold_label}.png"
        )
    )


def get_analysis_months():
    """Return the calendar months used by the selected analysis mode."""

    if CALENDAR_MONTH == 0:
        return list(
            range(
                1,
                13,
            )
        )

    return [
        CALENDAR_MONTH
    ]


def get_analysis_month_label():
    """Return a readable label for console output and the figure title."""

    if CALENDAR_MONTH == 0:
        return "Any month"

    return MONTH_NAMES[
        CALENDAR_MONTH - 1
    ]


# =============================================================================
# DATA READING
# =============================================================================

def read_reference_data(
    reference_dataset,
):
    """
    Read one observational reference for every month needed by the analysis.

    Returns one fitted observational sample and one event value per month.
    """

    filename = make_reference_filename(
        reference_dataset
    )

    variable = get_reference_variable(
        reference_dataset
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Reference file not found: {filename}"
        )

    months = get_analysis_months()

    monthly = {}

    with xr.open_dataset(
        filename
    ) as ds:

        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}."
            )

        # Storm Hans is one fixed reference-specific value. In any-month mode
        # it is compared with every monthly distribution.
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

        if not np.isfinite(
            storm_hans_value
        ):
            raise ValueError(
                f"The {get_reference_name(reference_dataset)} Storm Hans "
                "value is not finite."
            )

        for month in months:

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
                    f"Fewer than 10 finite values remain in the "
                    f"{get_reference_name(reference_dataset)} "
                    f"{MONTH_NAMES[month - 1]} observational fit sample."
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
                    f"No finite {MONTH_NAMES[month - 1]} values are available "
                    "for the calendar-record period."
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

            if EVENT_THRESHOLD == "storm_hans":

                threshold_value = (
                    storm_hans_value
                )

                threshold_label = (
                    f"Storm Hans {STORM_HANS_YEAR}"
                )

            else:

                threshold_value = (
                    record_value
                )

                threshold_label = (
                    f"{MONTH_NAMES[month - 1]} record "
                    f"({record_year})"
                )

            monthly[
                month
            ] = {
                "fit_values": fit_values,
                "threshold_value": threshold_value,
                "threshold_label": threshold_label,
                "record_value": record_value,
                "record_year": record_year,
            }

    return {
        "reference_dataset": reference_dataset,
        "reference_name": get_reference_name(
            reference_dataset
        ),
        "filename": filename,
        "storm_hans_value": storm_hans_value,
        "monthly": monthly,
    }


def read_model_month(
    method,
    month,
    reference_dataset=None,
):
    """
    Read one model sample for one calendar month.

    Raw uses one common file. Corrected methods use the file associated with
    reference_dataset.
    """

    filename = make_model_filename(
        method=method,
        reference_dataset=reference_dataset,
    )

    if not filename.is_file():
        raise FileNotFoundError(
            f"Model file not found for '{method}': "
            f"{filename}"
        )

    variable = get_model_variable()

    with xr.open_dataset(
        filename,
        decode_timedelta=False,
    ) as ds:

        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in "
                f"{filename}. Available variables: "
                f"{list(ds.data_vars)}"
            )

        if "month" not in ds:
            raise KeyError(
                f"Variable 'month' was not found in {filename}."
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
                f"Variable '{variable}' must contain "
                "dimensions 'number' and 'i_date'."
            )

        if ds[
            "month"
        ].dims != (
            "i_date",
        ):
            raise ValueError(
                "Variable 'month' must have dimension ('i_date',)."
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
            f"Fewer than 10 finite model values were found for "
            f"{method}, {MONTH_NAMES[month - 1]}."
        )

    return {
        "filename": filename,
        "values": values,
        "reference_dataset": reference_dataset,
    }


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
    """Negative log-likelihood for two-parameter GenEx."""

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
    """Fit two-parameter Generalized Exponential."""

    if np.any(
        values < 0
    ):
        raise ValueError(
            "GenEx requires non-negative values."
        )

    positive_values = (
        values[
            values > 0
        ]
    )

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
    threshold_value,
    method,
):
    """Return annual exceedance probability from one fitted distribution."""

    if method == "GEV":

        shape_c, location, scale = (
            fit_gev(
                values
            )
        )

        probability = genextreme.sf(
            threshold_value,
            shape_c,
            loc=location,
            scale=scale,
        )

    elif method == "Gumbel":

        location, scale = (
            fit_gumbel(
                values
            )
        )

        probability = gumbel_r.sf(
            threshold_value,
            loc=location,
            scale=scale,
        )

    elif method == "GenEx":

        shape, scale = (
            fit_genex(
                values
            )
        )

        if threshold_value < 0:

            probability = 1.0

        else:

            cdf = (
                1.0
                - np.exp(
                    -threshold_value
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
    threshold_value,
):
    """Calculate annual exceedance probability from threshold rank."""

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
                > threshold_value
            )
        )
    )

    return float(
        rank
        / values.size
    )


def analyse_method(
    values,
    threshold_value,
    method,
):
    """Calculate annual probability and return period for one method."""

    if method == "Empirical":

        annual_probability = (
            calculate_empirical_probability(
                values,
                threshold_value,
            )
        )

    else:

        annual_probability = (
            calculate_parametric_probability(
                values,
                threshold_value,
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


def combine_monthly_probabilities(
    monthly_probabilities,
):
    """
    Combine monthly exceedance probabilities into an annual any-month value.

    Assumes independence between monthly exceedance events:

        p_any = 1 - product(1 - p_m)
    """

    probabilities = np.asarray(
        monthly_probabilities,
        dtype=float,
    )

    if probabilities.size != 12:
        raise ValueError(
            "Any-month analysis requires exactly 12 monthly probabilities."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Monthly probabilities must all be finite."
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


def calculate_horizon_aep(
    annual_probability,
    horizon_years,
):
    """Convert annual exceedance probability to N-year AEP."""

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


def get_plot_value(
    analysis,
):
    """Return selected metric, clamped to plotting limits."""

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

        value = (
            RETURN_PERIOD_YMAX_YEARS
        )

    return float(
        np.clip(
            value,
            RETURN_PERIOD_YMIN_YEARS,
            RETURN_PERIOD_YMAX_YEARS,
        )
    )


# =============================================================================
# ANALYSIS
# =============================================================================

def calculate_all_results():
    """
    Calculate all methods for both references.

    For one-month mode, each result is calculated directly from that month.

    For CALENDAR_MONTH == 0, each method is first calculated separately for
    January through December. The 12 monthly exceedance probabilities are then
    combined using:

        p_any = 1 - product(1 - p_m)

    and the return period is 1 / p_any.
    """

    months = get_analysis_months()

    references = {
        reference_dataset: read_reference_data(
            reference_dataset
        )
        for reference_dataset in REFERENCE_DATASETS
    }

    samples = {}

    # Read raw once per month because it is reference-independent.
    raw_monthly_samples = {
        month: read_model_month(
            method="raw",
            month=month,
            reference_dataset=None,
        )
        for month in months
    }

    for reference_dataset in REFERENCE_DATASETS:

        reference = references[
            reference_dataset
        ]

        for month in months:

            samples[
                (
                    "reference",
                    reference_dataset,
                    month,
                )
            ] = {
                "values": reference[
                    "monthly"
                ][
                    month
                ][
                    "fit_values"
                ],
                "filename": reference[
                    "filename"
                ],
            }

            samples[
                (
                    "raw",
                    reference_dataset,
                    month,
                )
            ] = raw_monthly_samples[
                month
            ]

            for dataset_name in MODEL_DATASETS:

                if dataset_name == "raw":
                    continue

                samples[
                    (
                        dataset_name,
                        reference_dataset,
                        month,
                    )
                ] = read_model_month(
                    method=dataset_name,
                    month=month,
                    reference_dataset=reference_dataset,
                )

    monthly_results = {}

    for dataset_name in PLOT_DATASETS:

        for reference_dataset in REFERENCE_DATASETS:

            for month in months:

                values = samples[
                    (
                        dataset_name,
                        reference_dataset,
                        month,
                    )
                ][
                    "values"
                ]

                event_value = references[
                    reference_dataset
                ][
                    "monthly"
                ][
                    month
                ][
                    "threshold_value"
                ]

                for method in METHODS:

                    monthly_results[
                        (
                            dataset_name,
                            reference_dataset,
                            month,
                            method,
                        )
                    ] = analyse_method(
                        values=values,
                        threshold_value=event_value,
                        method=method,
                    )

    results = {}

    for dataset_name in PLOT_DATASETS:

        for reference_dataset in REFERENCE_DATASETS:

            for method in METHODS:

                if CALENDAR_MONTH != 0:

                    month = CALENDAR_MONTH

                    results[
                        (
                            dataset_name,
                            reference_dataset,
                            method,
                        )
                    ] = monthly_results[
                        (
                            dataset_name,
                            reference_dataset,
                            month,
                            method,
                        )
                    ].copy()

                    continue

                monthly_probabilities = [
                    monthly_results[
                        (
                            dataset_name,
                            reference_dataset,
                            month,
                            method,
                        )
                    ][
                        "annual_exceedance_probability"
                    ]
                    for month in months
                ]

                annual_probability = combine_monthly_probabilities(
                    monthly_probabilities
                )

                return_period = (
                    np.inf
                    if annual_probability <= 0
                    else 1.0
                    / annual_probability
                )

                results[
                    (
                        dataset_name,
                        reference_dataset,
                        method,
                    )
                ] = {
                    "annual_exceedance_probability": annual_probability,
                    "return_period": return_period,
                    "monthly_exceedance_probabilities": {
                        month: probability
                        for month, probability in zip(
                            months,
                            monthly_probabilities,
                        )
                    },
                }

    return (
        references,
        samples,
        results,
        monthly_results,
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_results(
    references,
    samples,
    results,
    monthly_results,
):
    """Print configuration, event values, and calculated AEP/return periods."""

    print()
    print(
        "Analysis configuration"
    )
    print(
        "----------------------"
    )
    print(
        "Calendar-month mode:",
        get_analysis_month_label(),
    )
    print(
        "Event type:",
        EVENT_THRESHOLD,
    )
    print(
        "Metric:",
        PLOT_METRIC,
    )

    print()
    print(
        "Reference event values"
    )
    print(
        "----------------------"
    )

    for reference_dataset in REFERENCE_DATASETS:

        reference = references[
            reference_dataset
        ]

        print()
        print(
            reference[
                "reference_name"
            ]
        )

        if EVENT_THRESHOLD == "storm_hans":

            print(
                f"  Storm Hans {STORM_HANS_YEAR}: "
                f"{reference['storm_hans_value']:.4f} mm"
            )

        else:

            for month in get_analysis_months():

                month_info = reference[
                    "monthly"
                ][
                    month
                ]

                print(
                    f"  {MONTH_NAMES[month - 1]:>9}: "
                    f"{month_info['threshold_value']:.4f} mm "
                    f"({month_info['record_year']})"
                )

    print()
    print(
        "Results"
    )
    print(
        "-------"
    )

    for dataset_name in PLOT_DATASETS:

        for reference_dataset in REFERENCE_DATASETS:

            for method in METHODS:

                analysis = results[
                    (
                        dataset_name,
                        reference_dataset,
                        method,
                    )
                ]

                annual_probability = analysis[
                    "annual_exceedance_probability"
                ]

                horizon_aep = calculate_horizon_aep(
                    annual_probability,
                    N_AEP_YEARS,
                )

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
                    f"{DISPLAY_LABELS[dataset_name]:>10} | "
                    f"{get_reference_name(reference_dataset):>7} | "
                    f"{method:>9} | "
                    f"AEP({N_AEP_YEARS}y)="
                    f"{100.0 * horizon_aep:.6g}% | "
                    f"RP={return_period_text} y"
                )

                if CALENDAR_MONTH == 0:

                    monthly_text = ", ".join(
                        (
                            f"{MONTH_NAMES[month - 1][:3]}="
                            f"{100.0 * probability:.4g}%"
                        )
                        for month, probability in analysis[
                            "monthly_exceedance_probabilities"
                        ].items()
                    )

                    print(
                        f"    monthly p: {monthly_text}"
                    )


def resolve_marker_facecolor(
    facecolor,
):
    """
    Convert the user-friendly facecolor setting to a Matplotlib value.

    The user can write:
        "empty"  -> marker has no fill
        any valid Matplotlib color -> marker is filled with that color
    """

    if (
        isinstance(
            facecolor,
            str,
        )
        and facecolor.strip().lower()
        == "empty"
    ):
        return "none"

    return facecolor


# =============================================================================
# PLOT
# =============================================================================

def metric_tick_formatter(
    value,
    position,
):
    """Format log-axis ticks."""

    if value <= 0:
        return ""

    if PLOT_METRIC == "aep":

        if np.isclose(
            value,
            AEP_YMIN_PERCENT,
        ):
            return (
                f"<{value:g}%"
            )

        return (
            f"{value:g}%"
        )

    if np.isclose(
        value,
        RETURN_PERIOD_YMAX_YEARS,
    ):
        return (
            f">{value:g}"
        )

    return (
        f"{value:g}"
    )


def get_y_axis_label():
    """Return publication-style y-axis label."""

    if PLOT_METRIC == "aep":

        if N_AEP_YEARS == 1:
            return (
                "Annual exceedance probability [%]"
            )

        return (
            f"{N_AEP_YEARS}-year exceedance probability [%]"
        )

    return (
        "Return period [years]"
    )


def get_title():
    """Return a concise title for the two-reference comparison."""

    month_name = get_analysis_month_label()

    event_text = (
        "Storm Hans"
        if EVENT_THRESHOLD == "storm_hans"
        else "calendar-month record"
    )

    return (
        f"{month_name}: {event_text}"
    )


def make_figure(
    results,
):
    """
    Plot two reference-specific stacks around every dataset x-axis position.

    SeNorge and ERA5 are separated by REFERENCE_X_DELTA and centered on the
    category label. Marker shape identifies the statistical method; marker
    color identifies the reference dataset.
    """

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

    figure, axis = plt.subplots(
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        ),
        constrained_layout=True,
    )

    x_positions = np.arange(
        len(
            PLOT_DATASETS
        ),
        dtype=float,
    )

    reference_offsets = {
        "era5": (
            -REFERENCE_X_DELTA
            / 2.0
        ),
        "senorge": (
            REFERENCE_X_DELTA
            / 2.0
        ),
    }

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

    for dataset_index, dataset_name in enumerate(
        PLOT_DATASETS
    ):

        for reference_dataset in REFERENCE_DATASETS:

            x_value = (
                dataset_index
                + reference_offsets[
                    reference_dataset
                ]
            )

            for method in METHODS:

                analysis = results[
                    (
                        dataset_name,
                        reference_dataset,
                        method,
                    )
                ]

                y_value = get_plot_value(
                    analysis
                )

                axis.scatter(
                    x_value
                    + method_offsets[
                        method
                    ],
                    y_value,
                    s=POINT_SIZE,
                    marker=METHOD_MARKERS[
                        method
                    ],
                    facecolor=resolve_marker_facecolor(
                        REFERENCE_MARKER_FACECOLORS[
                            reference_dataset
                        ]
                    ),
                    edgecolor=REFERENCE_MARKER_EDGECOLORS[
                        reference_dataset
                    ],
                    linewidth=MARKER_EDGE_WIDTH,
                    zorder=4,
                )

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
        -0.65,
        len(
            PLOT_DATASETS
        )
        - 0.35,
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            DISPLAY_LABELS[
                dataset
            ]
            for dataset in PLOT_DATASETS
        ],
        fontsize=TICK_LABELSIZE,
    )

    axis.set_xlabel(
        "Datasets",
        fontsize=AXIS_LABELSIZE,
    )

    axis.set_ylabel(
        get_y_axis_label(),
        fontsize=AXIS_LABELSIZE,
    )

    axis.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
    )

    axis.set_title(
        get_title(),
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=10,
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

    # One combined legend:
    #   - the first four entries show the symbols used for the calculation methods;
    #   - the final two entries show the colors used for SeNorge and ERA5.
    #
    # The method entries use a neutral grey so marker shape, rather than
    # reference color, is what the reader notices there.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[
                method
            ],
            linestyle="none",
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=np.sqrt(
                POINT_SIZE
            ),
            label=method,
        )
        for method in METHODS
    ]

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=resolve_marker_facecolor(
                    REFERENCE_MARKER_FACECOLORS[
                        reference_dataset
                    ]
                ),
                markeredgecolor=REFERENCE_MARKER_EDGECOLORS[
                    reference_dataset
                ],
                markeredgewidth=MARKER_EDGE_WIDTH,
                markersize=np.sqrt(
                    POINT_SIZE
                ),
                label=get_reference_name(
                    reference_dataset
                ),
            )
            for reference_dataset in REFERENCE_DATASETS
        ]
    )

    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc='best',
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
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

    (
        references,
        samples,
        results,
        monthly_results,
    ) = calculate_all_results()

    print_results(
        references=references,
        samples=samples,
        results=results,
        monthly_results=monthly_results,
    )

    make_figure(
        results=results,
    )
