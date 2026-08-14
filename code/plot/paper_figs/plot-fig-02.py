"""
Plot monthly S2S precipitation-extreme distributions together with the
1957–2022 record and the 2023 Storm Hans value from one reference dataset
(ERA5 or SeNorge). The same reference dataset is used for model bias correction.
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

catchment = "regine_glomma"
x_days = 2

forecast_date_range = ["2020-01-02", "2023-12-28"]
observation_years = ["1957", "2023"]

# Daily lead range used when creating the S2S model sample.
first_input_lead = 16
last_input_lead = 46

# Must match the value used by the sample-building script.
number_of_lead_bins = 2

# S2S distribution to plot: "full", "split1", "split2", ...
model_sampling_group = "full"

# Model-data source / bias-correction method:
# "raw", "mm", "q", "doy", "ld", or "q_doy".
MODEL_DATA_METHOD = "raw"

# Reference dataset used both in the figure and for bias correction.
# Options: "era5" or "senorge".
REFERENCE_DATASET = "senorge"

# ERA5 grid used when REFERENCE_DATASET == "era5".
era5_grid = "0.5x0.5"

write2file = True


# =============================================================================
# Dataset and plot settings
# =============================================================================

MODEL_VARIABLE = "tp24"
REFERENCE_SETTINGS = {
    "era5": {
        "variable": "tp24",
        "label": "ERA5",
        "directory": "era5_processed",
    },
    "senorge": {
        "variable": "rr",
        "label": "SeNorge",
        "directory": "senorge_processed",
    },
}

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.4
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
LEGEND_FONTSIZE = 7
YMIN = 0
YMAX = 135
BOX_WIDTH = 0.58
REFERENCE_COLOR = "tab:red"
COUNTERFACTUAL_COLOR = "tab:green"

MONTHS = np.arange(1, 13)
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Lead-time configuration
# =============================================================================

def validate_settings():
    """Validate the user-selectable model and reference settings."""
    if x_days < 1:
        raise ValueError("x_days must be at least 1.")
    if first_input_lead > last_input_lead:
        raise ValueError("first_input_lead must not exceed last_input_lead.")

    first_usable_lead = first_input_lead + x_days - 1
    if first_usable_lead > last_input_lead:
        raise ValueError("x_days is too large for the available input lead window.")

    number_of_usable_leads = last_input_lead - first_usable_lead + 1
    if not isinstance(number_of_lead_bins, int):
        raise TypeError("number_of_lead_bins must be an integer.")
    if not 1 <= number_of_lead_bins <= number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins must be between 1 and the number of usable lead times."
        )

    valid_groups = {"full"} | {
        f"split{number}" for number in range(1, number_of_lead_bins + 1)
    }
    if model_sampling_group not in valid_groups:
        raise ValueError(
            f"Unknown model_sampling_group '{model_sampling_group}'. "
            f"Valid options are {sorted(valid_groups)}."
        )

    valid_methods = {"raw", "mm", "q", "doy", "ld", "q_doy"}
    if MODEL_DATA_METHOD not in valid_methods:
        raise ValueError(
            f"MODEL_DATA_METHOD must be one of {sorted(valid_methods)}. "
            f"Got '{MODEL_DATA_METHOD}'."
        )

    if REFERENCE_DATASET not in REFERENCE_SETTINGS:
        raise ValueError(
            f"REFERENCE_DATASET must be one of {sorted(REFERENCE_SETTINGS)}. "
            f"Got '{REFERENCE_DATASET}'."
        )


def split_usable_accumulated_leads(first_lead, last_lead, number_of_bins):
    """Split accumulated ending leads into approximately equal contiguous bins."""
    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
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
    """Return the lead bins used by the model sample-building script."""
    first_usable_lead = first_input_lead + x_days - 1
    return split_usable_accumulated_leads(
        first_usable_lead, last_input_lead, number_of_lead_bins
    )


def get_full_lead_range():
    """Return the complete usable accumulated lead range."""
    return first_input_lead + x_days - 1, last_input_lead


def get_selected_model_lead_range():
    """Return the lead range for the selected S2S sampling group."""
    if model_sampling_group == "full":
        return get_full_lead_range()

    split_number = int(model_sampling_group.removeprefix("split"))
    return build_lead_bins()[split_number - 1]


def get_selected_model_variable():
    """Return the model variable for the selected S2S sampling group."""
    if model_sampling_group == "full":
        return "tp24_max"

    lead_start, lead_end = get_selected_model_lead_range()
    return f"tp24_max_lead{lead_start}_{lead_end}"


def lead_split_filename_label():
    """Return the lead-bin label used in model sample filenames."""
    full_start, full_end = get_full_lead_range()
    split_text = "_".join(f"{start}-{end}" for start, end in build_lead_bins())
    return f"lead{full_start}-{full_end}_split{number_of_lead_bins}_{split_text}"


def get_model_sampling_label():
    """Return a plot-friendly label for the selected S2S sample."""
    if model_sampling_group == "full":
        group_label = "Model extremes"
    else:
        lead_start, lead_end = get_selected_model_lead_range()
        split_number = model_sampling_group.removeprefix("split")
        group_label = f"S2S split {split_number}, ending leads {lead_start}-{lead_end}"

    if MODEL_DATA_METHOD == "raw":
        return f"{group_label} (raw)"

    reference_label = REFERENCE_SETTINGS[REFERENCE_DATASET]["label"]
    return f"{group_label}, {MODEL_DATA_METHOD} bias correction to {reference_label}"


# =============================================================================
# Filenames
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in model-sample filenames."""
    return catchment_name.removeprefix("regine_")


def make_model_filename():
    """Create the raw or bias-corrected compact S2S sample filename."""
    filename = os.path.join(
        config.dirs["s2s_processed"],
        (
            f"monthly_max_samples_{MODEL_VARIABLE}_{x_days}dayacc_"
            f"{get_file_id(catchment)}_{lead_split_filename_label()}_"
            f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
        ),
    )

    if MODEL_DATA_METHOD == "raw":
        return filename

    stem, extension = os.path.splitext(filename)
    return f"{stem}_bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}{extension}"


def make_reference_filename():
    """Create the input filename for the selected reference dataset."""
    settings = REFERENCE_SETTINGS[REFERENCE_DATASET]
    directory = config.dirs[settings["directory"]]
    variable = settings["variable"]

    if REFERENCE_DATASET == "era5":
        dataset_suffix = f"era5_{era5_grid}"
    else:
        dataset_suffix = "senorge"

    return (
        f"{directory}distribution_monthly_extremes_{variable}_{x_days}dayacc_"
        f"{catchment}_{dataset_suffix}_{observation_years[0]}-"
        f"{observation_years[1]}.nc"
    )


def make_figure_filename():
    """Create the output figure filename."""

    if MODEL_DATA_METHOD == "raw":
        suffix = "raw"
    else:
        suffix = f"bc-{MODEL_DATA_METHOD}"

    if model_sampling_group != "full":
        suffix += f"-{model_sampling_group}"

    filename = f"fig-02-{catchment}-{suffix}-{REFERENCE_DATASET}.png"

    return os.path.join(config.dirs["fig"], filename)


# =============================================================================
# Data extraction
# =============================================================================

def check_variable_exists(ds, variable, dataset_name):
    """Raise a clear error if a required variable is missing."""
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are {list(ds.data_vars)}."
        )


def validate_compact_model_structure(model_ds, variable):
    """Check the compact (number, i_date) model sample structure."""
    check_variable_exists(model_ds, variable, "model dataset")
    check_variable_exists(model_ds, "month", "model dataset")

    if set(model_ds[variable].dims) != {"number", "i_date"}:
        raise ValueError(
            f"Model variable '{variable}' must have dimensions 'number' and 'i_date'; "
            f"got {model_ds[variable].dims}."
        )
    if model_ds["month"].dims != ("i_date",):
        raise ValueError(
            "Model variable 'month' must have dimensions ('i_date',); "
            f"got {model_ds['month'].dims}."
        )


def get_model_values_by_month(model_ds, variable):
    """Return one flattened array of finite model values for each calendar month."""
    validate_compact_model_structure(model_ds, variable)

    values_by_month = []
    for month_number in MONTHS:
        selected = model_ds[variable].where(
            model_ds["month"] == month_number, drop=True
        )
        values = selected.values.ravel()
        values_by_month.append(values[np.isfinite(values)])

    return values_by_month


def get_monthly_records_before_hans(reference_ds, variable):
    """Return monthly record values from 1957–2022, excluding Storm Hans in 2023."""
    check_variable_exists(reference_ds, variable, "reference dataset")
    return reference_ds[variable].sel(year=slice(1957, 2022)).max(dim="year")


def get_storm_hans_event(reference_ds, variable):
    """Return the month and value of the largest reference event in 2023."""
    check_variable_exists(reference_ds, variable, "reference dataset")
    values_2023 = reference_ds[variable].sel(year=2023)
    flat = values_2023.stack(z=("month",))
    max_index = flat.argmax("z")

    max_value = float(flat.isel(z=max_index).values)
    max_month = int(flat["month"].isel(z=max_index).values)
    return max_month, max_value


def get_highest_may_model_event(model_ds, variable):
    """Return the largest May value in the selected S2S model distribution."""
    validate_compact_model_structure(model_ds, variable)
    may_values = model_ds[variable].where(model_ds["month"] == 5, drop=True).values.ravel()
    finite_values = may_values[np.isfinite(may_values)]

    if finite_values.size == 0:
        raise ValueError(f"No finite May values were found in '{variable}'.")

    return 5, float(finite_values.max())


# =============================================================================
# Plotting
# =============================================================================

def make_legend_handles():
    """Create legend handles for the plotted data."""
    reference_label = REFERENCE_SETTINGS[REFERENCE_DATASET]["label"]
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=REFERENCE_COLOR,
            markersize=6, label=f"{reference_label} record 1957–2022",
        ),
        Line2D(
            [0], [0], marker="^", linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=REFERENCE_COLOR,
            markersize=6, label=f"{reference_label} Storm Hans 2023",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor="0.6", markersize=5, label=get_model_sampling_label(),
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markersize=5, label="Counterfactual Storm Hans",
        ),
    ]


def apply_axis_formatting(ax):
    """Apply labels, limits, ticks, and simple panel styling."""
    ax.set_ylabel(
        f"Monthly maximum {x_days}-day precipitation",
        fontsize=AXIS_LABELSIZE,
    )
    ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_monthly_extreme_distributions(
    model_values_by_month,
    reference_ds,
    model_ds,
    model_variable,
    filename_out,
):
    """Create the monthly precipitation extreme-distribution figure."""
    reference_settings = REFERENCE_SETTINGS[REFERENCE_DATASET]
    reference_variable = reference_settings["variable"]

    reference_records = get_monthly_records_before_hans(
        reference_ds, reference_variable
    )
    hans_month, hans_value = get_storm_hans_event(reference_ds, reference_variable)
    counterfactual_month, counterfactual_value = get_highest_may_model_event(
        model_ds, model_variable
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    ax.boxplot(
        model_values_by_month,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,
        flierprops={
            "marker": "o", "markerfacecolor": "none", "markeredgecolor": "0.6",
            "markersize": 4, "linestyle": "none", "markeredgewidth": 0.8,
        },
        boxprops={"color": "0.25", "linewidth": 1.0},
        whiskerprops={"color": "0.25", "linewidth": 1.0},
        capprops={"color": "0.25", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.4},
    )

    ax.scatter(
        MONTHS,
        reference_records.values,
        color=REFERENCE_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )
    ax.scatter(
        hans_month,
        hans_value,
        color=REFERENCE_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
    )
    ax.scatter(
        counterfactual_month,
        counterfactual_value,
        color=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        s=20,
        zorder=6,
    )

    apply_axis_formatting(ax)
    ax.legend(
        handles=make_legend_handles(),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()

    if write2file:
        fig.savefig(filename_out, dpi=300, bbox_inches="tight")
        print("Wrote:", filename_out)

    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate_settings()

    model_variable = get_selected_model_variable()
    filename_model = make_model_filename()
    filename_reference = make_reference_filename()
    filename_out = make_figure_filename()
    lead_start, lead_end = get_selected_model_lead_range()

    print("Selected settings")
    print("-----------------")
    print(f"Model group:       {model_sampling_group}")
    print(f"Model leads:       {lead_start}-{lead_end}")
    print(f"Model variable:    {model_variable}")
    print(f"Data method:       {MODEL_DATA_METHOD}")
    print(f"Reference dataset: {REFERENCE_DATASET}")
    print()
    print("Reading model file:    ", filename_model)
    print("Reading reference file:", filename_reference)

    model_ds = xr.open_dataset(filename_model, decode_timedelta=False)
    reference_ds = xr.open_dataset(filename_reference)
    
    try:
        model_values_by_month = get_model_values_by_month(model_ds, model_variable)
        plot_monthly_extreme_distributions(
            model_values_by_month=model_values_by_month,
            reference_ds=reference_ds,
            model_ds=model_ds,
            model_variable=model_variable,
            filename_out=filename_out,
        )
    finally:
        model_ds.close()
        reference_ds.close()
