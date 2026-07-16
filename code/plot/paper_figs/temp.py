"""
Plot lead-time-dependent distributions of accumulated precipitation.

The input NetCDF file must contain:

    accumulated_value(lead_time, index)

The script creates one box-and-whisker distribution for every available lead
time. For a 1-day accumulation, the expected lead times are 16 through 46.
For an N-day accumulation, the first available accumulated lead is:

    first_input_lead + x_days - 1

The plotting style follows the ensemble-independence plotting script used in
this project: unfilled boxes, black median lines, grey outliers, outward ticks,
and hidden top and right spines.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Data settings ---------------------------------------------------------------

variable = "tp24"
x_days = 2

catchment = "regine_drammen"

valid_month = 1

forecast_date_range = (
    "2020-01-02",
    "2023-06-26",
)

first_input_lead = 16
last_input_lead = 46

# The first complete N-day accumulation is labelled by its final lead day.
first_valid_accumulation_lead = first_input_lead + x_days - 1


# Figure settings -------------------------------------------------------------

figure_width = 11.0
figure_height = 5.5
figure_dpi = 300

box_width = 0.65
show_outliers = True

label_fontsize = 11
title_fontsize = 12
tick_fontsize = 10

# Show one x-axis label for every N lead times.
label_every_n_leads = 2

# Optional y-axis limits. Use None for automatic limits.
y_axis_min = None
y_axis_max = None


# Save settings ---------------------------------------------------------------

path_out = config.dirs["fig"]

save_pdf = False
save_png = False
show_figure = True


# =============================================================================
# Labels and filenames
# =============================================================================

MONTH_ABBREVIATIONS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

VARIABLE_LABELS = {
    "tp": "precipitation",
    "tp24": "precipitation",
    "rr": "precipitation",
    "sro": "surface runoff",
    "gwb_q": "surface runoff",
    "ro": "total runoff",
    "sd": "snow water equivalent",
}

VARIABLE_UNITS = {
    "tp": "mm",
    "tp24": "mm",
    "rr": "mm",
    "sro": "mm",
    "gwb_q": "mm",
    "ro": "mm",
    "sd": "mm",
}


def readable_catchment_name(catchment_name):
    """Convert a technical catchment identifier into a readable name."""

    name = catchment_name

    prefixes = (
        "nve_catchment_regine_",
        "nve_catchment_",
        "regine_",
    )

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    return name.replace("_", " ").title()


def variable_description(variable_name):
    """Return a readable physical-variable description."""

    return VARIABLE_LABELS.get(
        variable_name,
        variable_name.replace("_", " "),
    )


def variable_units(variable_name):
    """Return plotting units for the selected variable."""

    return VARIABLE_UNITS.get(variable_name, "")


def build_plot_title():
    """Construct a descriptive figure title."""

    catchment_label = readable_catchment_name(catchment)
    variable_label = variable_description(variable)
    month_name = MONTH_NAMES[valid_month]

    return (
        f"Lead-time distributions of {x_days}-day accumulated "
        f"{variable_label} over {catchment_label} catchment "
        f"({month_name} valid dates)"
    )


def build_input_filename():
    """
    Construct the filename produced by the valid-month calculation script.
    """

    month_name = MONTH_NAMES[valid_month].lower()

    return os.path.join(
        config.dirs["s2s_processed"],
        (
            f"distribution_valid_month_{valid_month:02d}_{month_name}_"
            f"{variable}_{x_days}dayacc_"
            f"nve_catchment_{catchment}_forecast_hindcast_"
            f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
        ),
    )


def build_output_filename_stem():
    """Construct a descriptive output filename stem."""

    month_abbreviation = MONTH_ABBREVIATIONS[valid_month]

    return (
        f"distribution_by_lead-time_"
        f"{variable}_"
        f"{x_days}day-accumulation_"
        f"{catchment}_"
        f"lead{first_valid_accumulation_lead}-{last_input_lead}_"
        f"valid-month-{month_abbreviation}_"
        f"{forecast_date_range[0]}-to-{forecast_date_range[1]}"
    )


filename_in = build_input_filename()
filename_stem = build_output_filename_stem()
plot_title = build_plot_title()


# =============================================================================
# Validation and loading
# =============================================================================

def validate_user_settings():
    """Check user settings before reading and plotting the data."""

    if valid_month not in range(1, 13):
        raise ValueError("valid_month must be an integer from 1 to 12.")

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead < 0:
        raise ValueError("first_input_lead must be non-negative.")

    if last_input_lead < first_input_lead:
        raise ValueError(
            "last_input_lead must be greater than or equal to "
            "first_input_lead."
        )

    if label_every_n_leads < 1:
        raise ValueError("label_every_n_leads must be at least 1.")

    if min(label_fontsize, title_fontsize, tick_fontsize) <= 0:
        raise ValueError("All font sizes must be greater than zero.")

    if (
        y_axis_min is not None
        and y_axis_max is not None
        and y_axis_min >= y_axis_max
    ):
        raise ValueError("y_axis_min must be smaller than y_axis_max.")


def load_distribution(filename):
    """
    Load accumulated_value(lead_time, index) from the input NetCDF file.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Input file does not exist:\n{filename}"
        )

    with xr.open_dataset(
        filename,
        decode_timedelta=False,
    ) as dataset:

        if "accumulated_value" not in dataset:
            raise KeyError(
                "The input file does not contain 'accumulated_value'. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        data = dataset["accumulated_value"].load()

    required_dimensions = {"lead_time", "index"}
    missing_dimensions = required_dimensions.difference(data.dims)

    if missing_dimensions:
        raise ValueError(
            "accumulated_value is missing required dimensions: "
            f"{sorted(missing_dimensions)}"
        )

    return normalize_lead_time_coordinate(data)


def normalize_lead_time_coordinate(data):
    """Ensure that lead time is stored as integer day numbers."""

    lead_time = data["lead_time"]

    if np.issubdtype(lead_time.dtype, np.timedelta64):
        lead_time_days = (
            lead_time.values / np.timedelta64(1, "D")
        ).astype("int16")
    else:
        lead_time_days = lead_time.values.astype("int16")

    data = data.assign_coords(
        lead_time=("lead_time", lead_time_days)
    )

    data["lead_time"].attrs = {
        "long_name": "forecast lead day",
        "units": "days",
    }

    return data


def select_requested_lead_times(data):
    """
    Select the expected accumulated lead-time range.

    For x_days = 1, this selects lead days 16–46.
    For x_days = 2, this selects lead days 17–46, and so on.
    """

    available_leads = data["lead_time"].values.astype(int)

    selected_leads = available_leads[
        (available_leads >= first_valid_accumulation_lead)
        & (available_leads <= last_input_lead)
    ]

    if selected_leads.size == 0:
        raise ValueError(
            "No lead times fall within the requested plotting range "
            f"{first_valid_accumulation_lead}–{last_input_lead}. "
            f"Available leads are {available_leads.tolist()}."
        )

    return data.sel(lead_time=selected_leads)


# =============================================================================
# Boxplot preparation
# =============================================================================

def prepare_boxplot_values(data):
    """Return one finite value array for each lead time."""

    lead_times = data["lead_time"].values.astype(int)
    values_by_lead = []

    for lead_time in lead_times:

        values = (
            data
            .sel(lead_time=lead_time)
            .values
            .ravel()
        )

        values = values[np.isfinite(values)]

        if values.size == 0:
            raise ValueError(
                f"No finite accumulated values are available at "
                f"lead time {lead_time}."
            )

        values_by_lead.append(values)

    return lead_times, values_by_lead


def print_distribution_summary(lead_times, values_by_lead):
    """Print sample counts and basic distribution information."""

    print()
    print("Distribution summary")
    print("--------------------")

    for lead_time, values in zip(lead_times, values_by_lead):
        print(
            f"Lead {lead_time:2d}: "
            f"n={values.size:6d}, "
            f"median={np.median(values):8.3f}, "
            f"q25={np.percentile(values, 25):8.3f}, "
            f"q75={np.percentile(values, 75):8.3f}"
        )


# =============================================================================
# Plotting
# =============================================================================

def style_boxplot(boxplot):
    """Apply clear publication-style formatting to the boxplots."""

    for box in boxplot["boxes"]:
        box.set_linewidth(1.0)

    for whisker in boxplot["whiskers"]:
        whisker.set_linewidth(0.9)

    for cap in boxplot["caps"]:
        cap.set_linewidth(0.9)

    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)

    for outliers in boxplot["fliers"]:
        outliers.set_markeredgecolor("0.6")
        outliers.set_markerfacecolor("none")


def create_figure(data):
    """Create the lead-time box-and-whisker figure."""

    lead_times, values_by_lead = prepare_boxplot_values(data)

    positions = np.arange(1, len(lead_times) + 1)

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    boxplot = axis.boxplot(
        values_by_lead,
        positions=positions,
        widths=box_width,
        patch_artist=False,
        showfliers=show_outliers,
        whis=1.5,
        medianprops={
            "color": "black",
            "linewidth": 1.4,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "0.6",
            "markersize": 3.5,
            "linestyle": "none",
        },
    )

    style_boxplot(boxplot)

    visible_labels = [
        str(lead_time)
        if index % label_every_n_leads == 0
        else ""
        for index, lead_time in enumerate(lead_times)
    ]

    axis.set_xticks(positions)
    axis.set_xticklabels(
        visible_labels,
        fontsize=tick_fontsize,
    )

    axis.set_xlabel(
        "Lead time (days)",
        fontsize=label_fontsize,
    )

    units = variable_units(variable)

    y_label = (
        f"{x_days}-day accumulated "
        f"{variable_description(variable).capitalize()}"
    )

    if units:
        y_label += f" ({units})"

    axis.set_ylabel(
        y_label,
        fontsize=label_fontsize,
    )

    axis.set_title(
        plot_title,
        fontsize=title_fontsize,
        fontweight="normal",
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=tick_fontsize,
    )

    axis.set_xlim(
        0.3,
        len(lead_times) + 0.7,
    )

    if y_axis_min is not None or y_axis_max is not None:
        current_min, current_max = axis.get_ylim()

        axis.set_ylim(
            y_axis_min if y_axis_min is not None else current_min,
            y_axis_max if y_axis_max is not None else current_max,
        )

    print_distribution_summary(
        lead_times=lead_times,
        values_by_lead=values_by_lead,
    )

    return figure


# =============================================================================
# Output
# =============================================================================

def save_figure(figure):
    """Save the figure as PDF and/or PNG."""

    os.makedirs(path_out, exist_ok=True)

    if save_pdf:
        filename_pdf = os.path.join(
            path_out,
            f"{filename_stem}.pdf",
        )

        figure.savefig(
            filename_pdf,
            bbox_inches="tight",
        )

        print("Wrote:", filename_pdf)

    if save_png:
        filename_png = os.path.join(
            path_out,
            f"{filename_stem}.png",
        )

        figure.savefig(
            filename_png,
            dpi=figure_dpi,
            bbox_inches="tight",
        )

        print("Wrote:", filename_png)


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    print("Input file:")
    print(filename_in)
    print()

    print("Figure title:")
    print(plot_title)
    print()

    print("Output filename stem:")
    print(filename_stem)

    accumulated_values = load_distribution(
        filename=filename_in
    )

    accumulated_values = select_requested_lead_times(
        accumulated_values
    )

    figure = create_figure(
        data=accumulated_values
    )

    save_figure(figure)

    if show_figure:
        plt.show()
    else:
        plt.close(figure)
