"""
Plot monthly distributions of catchment precipitation extremes.

The S2S model input is read from the compact monthly maximum-sample file
produced from the preprocessed combined forecast/hindcast dataset.

Expected raw model structure
----------------------------
The model sample is organized by ensemble member and initialization:

    tp24_max(number, i_date)
    tp24_max_lead<start>_<end>(number, i_date)
    month(i_date)
    model_type(i_date)
    hdate(i_date)
    date_of_max(number, i_date)
    lead_of_max(number, i_date)

Unlike the older sample format, calendar month is not a dimension. Instead,
`month(i_date)` assigns every initialization to one month. To obtain a monthly
distribution, the script:

1. selects i_date values whose `month` equals the requested calendar month;
2. reads all ensemble-member values for those initializations;
3. flattens the selected (number, i_date) values;
4. removes NaNs.

Forecasts have 51 ensemble members. Hindcasts have 11 real members and NaN
padding for the remaining member positions, so removing NaNs produces the
correct monthly sample automatically.

Model sample selection
----------------------
The model sample can be selected in two ways:

    model_sampling_group = "full"
        Read tp24_max over the complete usable accumulated lead range.

    model_sampling_group = "split1", "split2", ...
        Read one lead-location subset such as tp24_max_lead17_31.

The number and bounds of the lead bins are determined from the same settings
used by the sample-building script.

Model-data method
-----------------
The model sample can be selected with one user parameter:

    MODEL_DATA_METHOD = "raw"
        Read the raw monthly maximum-sample file.

    MODEL_DATA_METHOD = "mm", "q", "doy", "ld", or "q_doy"
        Read a bias-corrected monthly maximum-sample file whose filename ends
        in `_bc_<method>_<reference>.nc`.

For bias-corrected input, BIAS_CORRECTION_REFERENCE selects either "senorge"
or "era5". The compact output files from the bias-correction/sample-building
scripts retain the original model variable names, so this plotting script always
reads `tp24_max` or `tp24_max_lead<start>_<end>` without a BC suffix.

Example
-------
With:

    first_input_lead = 16
    last_input_lead = 46
    x_days = 2
    number_of_lead_bins = 2

the usable accumulated ending leads are 17-46, and the available variables are:

    "full"   -> tp24_max
    "split1" -> tp24_max_lead17_31
    "split2" -> tp24_max_lead32_46
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"
x_days = 2

forecast_date_range = [
    "2020-01-02",
    "2022-12-29",
]

observation_years = [
    "1957",
    "2023",
]

era5_grid = "0.5x0.5"


# Daily lead range used when creating the S2S model sample.
first_input_lead = 16
last_input_lead = 46


# Must be identical to the value used by the sample-building script.
number_of_lead_bins = 2


# Select which S2S distribution to plot.
#
# Options:
#     "full"
#     "split1"
#     "split2"
#     ...
#
# The valid split numbers depend on number_of_lead_bins.
model_sampling_group = "full"


# Select the model-data source / bias-correction method.
#
# Options:
#     "raw"   -> uncorrected monthly maximum sample
#     "mm"    -> monthly-mean multiplicative correction
#     "q"     -> quantile-based correction
#     "doy"   -> day-of-year correction
#     "ld"    -> lead-day correction
#     "q_doy" -> combined quantile/day-of-year correction
MODEL_DATA_METHOD = "q_doy"

# Reference dataset used for bias correction. Ignored when
# MODEL_DATA_METHOD == "raw".
#
# Options:
#     "era5"
#     "senorge"
BIAS_CORRECTION_REFERENCE = "era5"


write2file = False


# =============================================================================
# Dataset-specific settings
# =============================================================================

MODEL_VARIABLE = "tp24"

ERA5_VARIABLE = "tp24"

SENORGE_VARIABLE = "rr"
SENORGE_LABEL = "SeNorge"


# =============================================================================
# Plot settings
# =============================================================================

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.4

TITLE_FONTSIZE = 11
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
LEGEND_FONTSIZE = 7

YMIN = 0
YMAX = 135

BOX_WIDTH = 0.58

REFERENCE_COLOR = "tab:red"
ERA5_COLOR = "tab:blue"
COUNTERFACTUAL_COLOR = "tab:green"

MONTHS = np.arange(
    1,
    13,
)

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


# =============================================================================
# Lead-time configuration
# =============================================================================

def validate_model_sampling_settings():
    """Check the lead-time settings and selected sampling group."""

    if x_days < 1:
        raise ValueError(
            "x_days must be at least 1."
        )

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )


    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )


    number_of_usable_leads = (
        last_input_lead
        - first_usable_lead
        + 1
    )


    if not isinstance(
        number_of_lead_bins,
        int,
    ):
        raise TypeError(
            "number_of_lead_bins must be an integer."
        )


    if number_of_lead_bins < 1:
        raise ValueError(
            "number_of_lead_bins must be at least 1."
        )


    if (
        number_of_lead_bins
        > number_of_usable_leads
    ):
        raise ValueError(
            "number_of_lead_bins cannot exceed the number of "
            "usable accumulated lead times."
        )


    valid_groups = {
        "full"
    }

    valid_groups.update(
        {
            f"split{bin_number}"
            for bin_number in range(
                1,
                number_of_lead_bins + 1,
            )
        }
    )


    if model_sampling_group not in valid_groups:
        raise ValueError(
            f"Unknown model_sampling_group "
            f"'{model_sampling_group}'. "
            f"Valid options are: "
            f"{sorted(valid_groups)}."
        )


    valid_methods = {
        "raw",
        "mm",
        "q",
        "doy",
        "ld",
        "q_doy",
    }

    if MODEL_DATA_METHOD not in valid_methods:
        raise ValueError(
            f"MODEL_DATA_METHOD must be one of "
            f"{sorted(valid_methods)}. "
            f"Got '{MODEL_DATA_METHOD}'."
        )

    if MODEL_DATA_METHOD != "raw":

        valid_references = {
            "era5",
            "senorge",
        }

        if BIAS_CORRECTION_REFERENCE not in valid_references:
            raise ValueError(
                f"BIAS_CORRECTION_REFERENCE must be one of "
                f"{sorted(valid_references)}. "
                f"Got '{BIAS_CORRECTION_REFERENCE}'."
            )


def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """
    Split usable accumulated ending leads into approximately equal bins.

    Extra lead times are assigned to the later bins.

    Example:
        17-46 contains 30 leads.

        With 2 bins:
            17-31
            32-46

        18-46 contains 29 leads.

        With 2 bins:
            18-31
            32-46
    """

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


    # Later bins receive any extra lead times.
    bin_sizes = [
        base_size
        + int(
            bin_index
            >= number_of_bins - remainder
        )
        for bin_index in range(
            number_of_bins
        )
    ]


    lead_bins = []
    current_start = first_lead


    for bin_size in bin_sizes:

        current_end = (
            current_start
            + bin_size
            - 1
        )

        lead_bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + 1
        )


    return lead_bins


def build_lead_bins():
    """
    Return the lead-location bins used by the model sample-building script.

    Splitting occurs AFTER accounting for X-day accumulation.
    """

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )


    return split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )


def get_full_lead_range():
    """Return the complete usable accumulated lead range."""

    return (
        first_input_lead
        + x_days
        - 1,
        last_input_lead,
    )


def get_selected_model_lead_range():
    """
    Return the lead range corresponding to model_sampling_group.

    "full" returns the complete usable lead window.

    "splitN" returns lead bin N.
    """

    if model_sampling_group == "full":
        return get_full_lead_range()


    lead_bins = (
        build_lead_bins()
    )


    split_number = int(
        model_sampling_group.replace(
            "split",
            "",
        )
    )


    # Convert the user-facing 1-based split number to a Python list index.
    return lead_bins[
        split_number - 1
    ]


def get_selected_model_variable():
    """Return the variable name for the selected S2S distribution."""

    if model_sampling_group == "full":

        variable = "tp24_max"

    else:

        (
            lead_start,
            lead_end,
        ) = get_selected_model_lead_range()

        variable = (
            f"tp24_max_lead"
            f"{lead_start}_{lead_end}"
        )


    # Bias-corrected compact sample files retain the same variable names as
    # the raw compact sample file. The method/reference are encoded only in
    # the filename.
    return variable


def get_selected_sample_count_variable():
    """
    Return the optional stored-count variable name.

    The current compact sample files do not store monthly count variables.
    This name is retained only for compatibility with a future file that may
    choose to include them.
    """

    if model_sampling_group == "full":

        return "sample_count_tp24_max"

    (
        lead_start,
        lead_end,
    ) = get_selected_model_lead_range()

    return (
        f"sample_count_tp24_max_lead"
        f"{lead_start}_{lead_end}"
    )


def lead_split_filename_label():
    """
    Return the lead-split filename label used by the sample-building script.

    Example:
        lead17-46_split2_17-31_32-46
    """

    (
        full_start,
        full_end,
    ) = get_full_lead_range()


    lead_bins = (
        build_lead_bins()
    )


    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    )


    return (
        f"lead{full_start}-{full_end}_"
        f"split{number_of_lead_bins}_"
        f"{split_text}"
    )


def get_model_sampling_label():
    """Return a plot-friendly label for the selected S2S sampling group."""

    (
        lead_start,
        lead_end,
    ) = get_selected_model_lead_range()


    if model_sampling_group == "full":

        if MODEL_DATA_METHOD != "raw":

            return (
                f"Model extremes, {MODEL_DATA_METHOD} bias correction "
                f"to {BIAS_CORRECTION_REFERENCE.upper()}"
            )

        return "Model extremes (raw)"


    split_number = int(
        model_sampling_group.replace(
            "split",
            "",
        )
    )


    label = (
        f"S2S split {split_number}, "
        f"ending leads {lead_start}-{lead_end}"
    )


    if MODEL_DATA_METHOD != "raw":

        label += (
            f", {MODEL_DATA_METHOD} bias correction to "
            f"{BIAS_CORRECTION_REFERENCE.upper()}"
        )
    else:
        label += ", raw"


    return label


# =============================================================================
# Configuration helpers
# =============================================================================

def get_catchment_label(
    catchment,
):
    """Return a plot-friendly catchment name."""

    labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }


    return labels.get(
        catchment,
        catchment,
    )


# =============================================================================
# Filename helpers
# =============================================================================

def get_file_id(
    catchment_name,
):
    """Return the short catchment label used in model-sample filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name



def make_model_filename():
    """
    Create the compact S2S monthly-sample input filename.

    Bias-corrected compact sample files use the same structure and append
    `_bc_<method>_<reference>` to the raw sample filename.
    """

    lead_label = (
        lead_split_filename_label()
    )

    filename = os.path.join(
        config.dirs[
            "s2s_processed"
        ],
        (
            f"monthly_max_samples_"
            f"{MODEL_VARIABLE}_"
            f"{x_days}dayacc_"
            f"{get_file_id(catchment)}_"
            f"{lead_label}_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )

    if MODEL_DATA_METHOD != "raw":

        stem, extension = os.path.splitext(
            filename
        )

        filename = (
            f"{stem}_bc_"
            f"{MODEL_DATA_METHOD}_"
            f"{BIAS_CORRECTION_REFERENCE}"
            f"{extension}"
        )

    return filename


def make_era5_filename():
    """Create the ERA5 input filename."""

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_"
        f"{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{observation_years[0]}-"
        f"{observation_years[1]}.nc"
    )


def make_senorge_filename():
    """Create the SeNorge input filename."""

    return (
        f"{config.dirs['senorge_processed']}"
        f"distribution_monthly_extremes_"
        f"{SENORGE_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_senorge_"
        f"{observation_years[0]}-"
        f"{observation_years[1]}.nc"
    )


def make_figure_filename():
    """Create an output figure name that records the model-data choice."""

    if MODEL_DATA_METHOD != "raw":
        suffix = (
            f"bc-{MODEL_DATA_METHOD}-"
            f"{BIAS_CORRECTION_REFERENCE}"
        )
    else:
        suffix = "raw"


    if model_sampling_group != "full":
        suffix = (
            f"{suffix}-{model_sampling_group}"
        )


    if catchment == "regine_drammen":
        filename = (
            f"fig-02-{suffix}.png"
        )
    else:
        filename = (
            f"fig-02-{catchment}-{suffix}.png"
        )


    return os.path.join(
        config.dirs["fig"],
        filename,
    )


# =============================================================================
# Data loading
# =============================================================================

def load_datasets(
    filename_model,
    filename_era5,
    filename_senorge,
):
    """Open the model, ERA5, and SeNorge datasets."""

    model_ds = xr.open_dataset(
        filename_model
    )

    era5_ds = xr.open_dataset(
        filename_era5
    )

    senorge_ds = xr.open_dataset(
        filename_senorge
    )


    return (
        model_ds,
        era5_ds,
        senorge_ds,
    )


def check_variable_exists(
    ds,
    variable,
    dataset_name,
):
    """Raise a clear error if a required variable is missing."""

    if variable not in ds:

        raise KeyError(
            f"Variable '{variable}' was not found in "
            f"{dataset_name}. "
            f"Available variables are: "
            f"{list(ds.data_vars)}"
        )


# =============================================================================
# Model data extraction
# =============================================================================

def validate_compact_model_structure(
    model_ds,
    variable,
):
    """Check the compact `(number, i_date)` monthly-sample structure."""

    check_variable_exists(
        model_ds,
        variable,
        "model dataset",
    )

    check_variable_exists(
        model_ds,
        "month",
        "model dataset",
    )

    required_dimensions = {
        "number",
        "i_date",
    }

    if set(
        model_ds[
            variable
        ].dims
    ) != required_dimensions:

        raise ValueError(
            f"Model variable '{variable}' must contain dimensions "
            f"{sorted(required_dimensions)}, but has "
            f"{model_ds[variable].dims}."
        )

    if model_ds[
        "month"
    ].dims != (
        "i_date",
    ):

        raise ValueError(
            "Model variable 'month' must have dimensions ('i_date',), "
            f"but has {model_ds['month'].dims}."
        )


def get_model_values_by_month(
    model_ds,
    variable,
):
    """
    Return one flattened finite-value array for each calendar month.

    For a requested month:

    1. identify i_date positions where month(i_date) matches;
    2. retain every ensemble member for those initializations;
    3. flatten number and i_date;
    4. remove NaNs.

    NaN removal also removes the padded forecast-member positions that do not
    exist for the 11-member hindcasts.
    """

    validate_compact_model_structure(
        model_ds,
        variable,
    )

    values_by_month = []

    for month_number in MONTHS:

        selected = (
            model_ds[
                variable
            ]
            .where(
                model_ds[
                    "month"
                ]
                == month_number,
                drop=True,
            )
        )

        values = selected.values.ravel()

        values = values[
            np.isfinite(
                values
            )
        ]

        values_by_month.append(
            values
        )

    return values_by_month


def check_model_sample_counts(
    model_ds,
    model_values_by_month,
    sample_count_variable,
):
    """
    Print monthly finite-value counts.

    The compact model sample normally does not contain stored sample-count
    variables. When an optional count variable exists, compare it with the
    values read from the selected model variable.
    """

    calculated_counts = np.array(
        [
            values.size
            for values in model_values_by_month
        ],
        dtype=int,
    )

    print()
    print(
        "Model sample counts"
    )
    print(
        "-------------------"
    )

    if sample_count_variable not in model_ds:

        print(
            f"{'Month':<8}"
            f"{'read':>10}"
        )

        print(
            "-" * 18
        )

        for month_label, calculated in zip(
            MONTH_LABELS,
            calculated_counts,
        ):

            print(
                f"{month_label:<8}"
                f"{calculated:>10}"
            )

        print(
            "-" * 18
        )

        print(
            "No stored sample-count variable was present; "
            "counts above were calculated from finite values."
        )

        return

    stored_counts = (
        model_ds[
            sample_count_variable
        ]
        .values
        .astype(int)
    )

    if stored_counts.size != 12:

        raise ValueError(
            f"Stored sample-count variable '{sample_count_variable}' "
            "must contain 12 monthly values."
        )

    print(
        f"{'Month':<8}"
        f"{'stored':>10}"
        f"{'read':>10}"
        f"{'check':>10}"
    )

    print(
        "-" * 38
    )

    all_ok = True

    for (
        month_label,
        stored,
        calculated,
    ) in zip(
        MONTH_LABELS,
        stored_counts,
        calculated_counts,
    ):

        check = (
            "OK"
            if stored == calculated
            else "FAIL"
        )

        if check == "FAIL":
            all_ok = False

        print(
            f"{month_label:<8}"
            f"{stored:>10}"
            f"{calculated:>10}"
            f"{check:>10}"
        )

    print(
        "-" * 38
    )

    if not all_ok:

        raise ValueError(
            "Stored model sample counts do not match the finite "
            "values read from the selected model distribution."
        )

    print(
        "Sample-count check passed."
    )


# =============================================================================
# ERA5 and SeNorge data extraction
# =============================================================================

def get_monthly_records_before_hans(
    ds,
    variable,
):
    """
    Get monthly records before Storm Hans.

    Uses 1957-2022, so Storm Hans in 2023 is excluded.
    """

    check_variable_exists(
        ds,
        variable,
        "input dataset",
    )


    before_hans = (
        ds[variable]
        .sel(
            year=slice(
                1957,
                2022,
            )
        )
    )


    return before_hans.max(
        dim="year"
    )


def get_storm_hans_event(
    ds,
    variable,
):
    """
    Get the largest 2023 event.

    This assumes the largest 2023 value corresponds to Storm Hans.
    """

    check_variable_exists(
        ds,
        variable,
        "input dataset",
    )


    values_2023 = (
        ds[variable]
        .sel(
            year=2023
        )
    )


    flat = values_2023.stack(
        z=("month",)
    )


    max_index = flat.argmax(
        "z"
    )

    max_value = flat.isel(
        z=max_index
    )

    max_month = flat[
        "month"
    ].isel(
        z=max_index
    )


    return (
        int(
            max_month.values
        ),
        float(
            max_value.values
        ),
    )


def get_highest_may_model_event(
    model_ds,
    variable,
):
    """Get the largest May event from the compact S2S distribution."""

    validate_compact_model_structure(
        model_ds,
        variable,
    )

    may_values = (
        model_ds[
            variable
        ]
        .where(
            model_ds[
                "month"
            ]
            == 5,
            drop=True,
        )
        .values
        .ravel()
    )

    finite_values = may_values[
        np.isfinite(
            may_values
        )
    ]

    if finite_values.size == 0:

        raise ValueError(
            f"No finite May values were found in '{variable}'."
        )

    return (
        5,
        float(
            finite_values.max()
        ),
    )


# =============================================================================
# Plotting helpers
# =============================================================================

def make_legend_handles():
    """Create legend handles for the plot."""

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=(
                f"{SENORGE_LABEL} "
                f"record 1957–2022"
            ),
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 record 1957–2022",
        ),

        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=(
                f"{SENORGE_LABEL} "
                f"Storm Hans 2023"
            ),
        ),

        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 Storm Hans 2023",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.6",
            markeredgewidth=0.8,
            markersize=5,
            label=get_model_sampling_label(),
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label=(
                "Counterfactual Storm Hans"
            ),
        ),
    ]


def apply_axis_formatting(
    ax,
):
    """Apply consistent axis formatting."""

    catchment_label = (
        get_catchment_label(
            catchment
        )
    )


    (
        lead_start,
        lead_end,
    ) = get_selected_model_lead_range()


    ax.set_title(
        (
            f"{catchment_label}, monthly "
            f"{x_days}-day accumulated "
            f"precipitation maxima\n"
        ),
        fontsize=TITLE_FONTSIZE,
        pad=8,
    )


    ax.set_ylabel(
        "mm",
        fontsize=AXIS_LABELSIZE,
    )


    ax.set_xlabel(
        "Month",
        fontsize=AXIS_LABELSIZE,
    )


    ax.set_xlim(
        0.4,
        12.6,
    )

    ax.set_ylim(
        YMIN,
        YMAX,
    )


    ax.set_xticks(
        MONTHS
    )

    ax.set_xticklabels(
        MONTH_LABELS
    )


    ax.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# =============================================================================
# Main plotting function
# =============================================================================

def plot_monthly_extreme_distributions(
    model_values_by_month,
    era5_ds,
    senorge_ds,
    model_ds,
    model_variable,
    filename_out,
    write2file,
):
    """Create the monthly precipitation extreme-distribution figure."""

    era5_records = (
        get_monthly_records_before_hans(
            era5_ds,
            variable=ERA5_VARIABLE,
        )
    )


    senorge_records = (
        get_monthly_records_before_hans(
            senorge_ds,
            variable=SENORGE_VARIABLE,
        )
    )


    (
        era5_hans_month,
        era5_hans_value,
    ) = get_storm_hans_event(
        era5_ds,
        variable=ERA5_VARIABLE,
    )


    (
        senorge_hans_month,
        senorge_hans_value,
    ) = get_storm_hans_event(
        senorge_ds,
        variable=SENORGE_VARIABLE,
    )


    (
        counterfactual_month,
        counterfactual_value,
    ) = get_highest_may_model_event(
        model_ds,
        variable=model_variable,
    )


    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        ),
    )


    ax.boxplot(
        model_values_by_month,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,

        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.6",
            markersize=4,
            linestyle="none",
            markeredgewidth=0.8,
        ),

        boxprops=dict(
            color="0.25",
            linewidth=1.0,
        ),

        whiskerprops=dict(
            color="0.25",
            linewidth=1.0,
        ),

        capprops=dict(
            color="0.25",
            linewidth=1.0,
        ),

        medianprops=dict(
            color="black",
            linewidth=1.4,
        ),
    )


    ax.scatter(
        MONTHS,
        senorge_records.values,
        facecolors=REFERENCE_COLOR,
        edgecolors=REFERENCE_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )


    ax.scatter(
        MONTHS,
        era5_records.values,
        facecolors=ERA5_COLOR,
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )


    ax.scatter(
        senorge_hans_month,
        senorge_hans_value,
        facecolors=REFERENCE_COLOR,
        edgecolors=REFERENCE_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
    )


    ax.scatter(
        era5_hans_month,
        era5_hans_value,
        facecolors=ERA5_COLOR,
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
    )


    ax.scatter(
        counterfactual_month,
        counterfactual_value,
        facecolors=COUNTERFACTUAL_COLOR,
        edgecolors=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        marker="o",
        s=20,
        zorder=6,
    )


    apply_axis_formatting(
        ax
    )


    ax.legend(
        handles=make_legend_handles(),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
    )


    fig.tight_layout()


    if write2file:

        fig.savefig(
            filename_out,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "Wrote:",
            filename_out,
        )


    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_model_sampling_settings()


    model_extreme_variable = (
        get_selected_model_variable()
    )


    model_sample_count_variable = (
        get_selected_sample_count_variable()
    )


    filename_model = (
        make_model_filename()
    )


    filename_era5 = (
        make_era5_filename()
    )


    filename_senorge = (
        make_senorge_filename()
    )


    filename_out = (
        make_figure_filename()
    )


    (
        lead_start,
        lead_end,
    ) = get_selected_model_lead_range()


    print(
        "Selected model sampling group"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Group:          "
        f"{model_sampling_group}"
    )

    print(
        f"Leads:          "
        f"{lead_start}-{lead_end}"
    )

    print(
        f"Maximum variable: "
        f"{model_extreme_variable}"
    )

    print(
        f"Count variable:   "
        f"{model_sample_count_variable}"
    )

    print(
        f"Data method:       "
        f"{MODEL_DATA_METHOD}"
    )

    if MODEL_DATA_METHOD != "raw":

        print(
            f"BC reference:     "
            f"{BIAS_CORRECTION_REFERENCE}"
        )


    print()
    print(
        "Available lead-time selections"
    )
    print(
        "------------------------------"
    )


    (
        full_start,
        full_end,
    ) = get_full_lead_range()


    full_variable = "tp24_max"

    print(
        f"full:   "
        f"{full_variable}"
    )


    for bin_number, (
        bin_start,
        bin_end,
    ) in enumerate(
        build_lead_bins(),
        start=1,
    ):

        split_variable = (
            f"tp24_max_lead"
            f"{bin_start}_{bin_end}"
        )

        print(
            f"split{bin_number}: "
            f"{split_variable}"
        )


    print()
    print(
        "Reading model file:    ",
        filename_model,
    )

    print(
        "Reading ERA5 file:     ",
        filename_era5,
    )

    print(
        "Reading SeNorge file:  ",
        filename_senorge,
    )


    (
        model_ds,
        era5_ds,
        senorge_ds,
    ) = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_senorge=filename_senorge,
    )


    try:

        model_values_by_month = (
            get_model_values_by_month(
                model_ds,
                variable=model_extreme_variable,
            )
        )


        check_model_sample_counts(
            model_ds=model_ds,
            model_values_by_month=model_values_by_month,
            sample_count_variable=model_sample_count_variable,
        )


        plot_monthly_extreme_distributions(
            model_values_by_month=model_values_by_month,
            era5_ds=era5_ds,
            senorge_ds=senorge_ds,
            model_ds=model_ds,
            model_variable=model_extreme_variable,
            filename_out=filename_out,
            write2file=write2file,
        )


    finally:

        model_ds.close()
        era5_ds.close()
        senorge_ds.close()
