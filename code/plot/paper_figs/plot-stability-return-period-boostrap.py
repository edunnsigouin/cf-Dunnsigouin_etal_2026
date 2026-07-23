"""
Plot empirical extreme-value distributions for grouped forecast lead times.

Input NetCDF variable
---------------------
accumulated_value(lead_time, index)

What this script does
---------------------
1. Selects the requested accumulated lead times.
2. Combines consecutive lead times into groups.
3. Pools all finite values within each group.
4. Calculates one empirical return-value curve for each group.
5. Calculates a bootstrap confidence interval using all selected lead times.
6. Uses that same confidence interval for both the grey return-period band
   and the inset boxplot whiskers.
7. Plots grouped probability-density functions in panel 1.
8. Plots grouped return-period curves in panel 2.
9. Uses the same six Matplotlib ``tab:`` colors in both panels.

For a 2-day accumulation with input leads 16-46, the valid accumulated lead
times are 17-46. With five lead times per group, the six groups are:

    17-21, 22-26, 27-31, 32-36, 37-41, and 42-46.

Note about the empirical return period estimate:
Empirical return values were estimated directly from the sample quantiles, 
with return period T related to non-exceedance probability as p=1−1/T. 
Quantiles were calculated using linear interpolation (Type 7), 
without fitting a theoretical extreme-value distribution.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Physical variable and accumulation period.
variable = "tp24"
x_days = 2

# Catchment and valid month used in the input filename.
catchment = "regine_drammen"
valid_month = 12

# Date range used in the input filename.
forecast_date_range = (
    "2020-01-02",
    "2023-06-26",
)

# The input lead range used when the accumulated data were produced.
first_input_lead = 16
last_input_lead = 46

# An x-day accumulation beginning at first_input_lead first becomes valid at:
# first_input_lead + x_days - 1.
first_valid_accumulation_lead = first_input_lead + x_days - 1

# Number of consecutive lead times pooled into each distribution.
# For leads 17-46, a value of 5 creates six groups.
lead_times_per_group = 5


# Bootstrap settings ----------------------------------------------------------

number_of_bootstrap_samples = 1000
confidence_interval_percent = 95.0

# The bootstrap sample length is based on the grouped samples.
# Options:
#   "minimum" -> use the smallest grouped sample size
#   "median"  -> use the median grouped sample size
#   integer   -> use that exact sample size
bootstrap_sample_size = "minimum"

random_seed = 42


# Empirical return-period grid ------------------------------------------------

minimum_return_period = 1.01
maximum_return_period = 1000.0
number_of_return_period_points = 300

# Inset settings ---------------------------------------------------------------
inset_centre_return_period = 100.0
inset_half_width_decades = 0.1
inset_padding_fraction = 0.2


# Figure settings -------------------------------------------------------------

figure_width = 12.0
figure_height = 5.2
figure_dpi = 300

label_fontsize = 11
title_fontsize = 12
tick_fontsize = 10
legend_fontsize = 9

group_line_width = 1.8
confidence_alpha = 0.28

# Histogram settings
number_of_histogram_bins = 50
density_x_min = None
density_x_max = None

# Six explicit Matplotlib tab colors, one for each expected group.
group_colors = None

return_value_y_min = None
return_value_y_max = None
density_y_min = 0.0
density_y_max = None


# Save settings ---------------------------------------------------------------

path_out = config.dirs["fig"]
save_pdf = False
save_png = True
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

    for prefix in (
        "nve_catchment_regine_",
        "nve_catchment_",
        "regine_",
    ):
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
    """Construct a descriptive plot title."""

    return (
        f"Grouped empirical extreme-value distributions of "
        f"{x_days}-day accumulated {variable_description(variable)} over "
        f"{readable_catchment_name(catchment)} catchment "
        f"({MONTH_NAMES[valid_month]} valid dates)"
    )


def build_input_filename():
    """Construct the expected NetCDF input filename."""

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
    """Construct the output figure filename without an extension."""

    return (
        f"stability-return-period-boostrap_"
        f"{variable}_{x_days}day-accumulation_{catchment}_"
        f"{lead_times_per_group}-leads-per-group_"
        f"valid-month-{MONTH_ABBREVIATIONS[valid_month]}_"
        f"{forecast_date_range[0]}-to-{forecast_date_range[1]}"
    )


filename_in = build_input_filename()
filename_stem = build_output_filename_stem()
plot_title = build_plot_title()


# =============================================================================
# Validation and loading
# =============================================================================


def validate_user_settings():
    """Check user settings before reading or processing data."""

    if valid_month not in range(1, 13):
        raise ValueError("valid_month must be an integer from 1 to 12.")

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if lead_times_per_group < 1:
        raise ValueError("lead_times_per_group must be at least 1.")

    if number_of_bootstrap_samples < 1:
        raise ValueError("number_of_bootstrap_samples must be at least 1.")

    if not 0.0 < confidence_interval_percent < 100.0:
        raise ValueError(
            "confidence_interval_percent must be between 0 and 100."
        )

    if minimum_return_period <= 1.0:
        raise ValueError("minimum_return_period must be greater than 1.")

    if maximum_return_period <= minimum_return_period:
        raise ValueError(
            "maximum_return_period must exceed minimum_return_period."
        )

    if number_of_return_period_points < 2:
        raise ValueError(
            "number_of_return_period_points must be at least 2."
        )
    if (
        density_x_min is not None
        and density_x_max is not None
        and density_x_min >= density_x_max
    ):
        raise ValueError("density_x_min must be smaller than density_x_max.")

    if (
        density_y_min is not None
        and density_y_max is not None
        and density_y_min >= density_y_max
    ):
        raise ValueError("density_y_min must be smaller than density_y_max.")

    if (
        return_value_y_min is not None
        and return_value_y_max is not None
        and return_value_y_min >= return_value_y_max
    ):
        raise ValueError(
            "return_value_y_min must be smaller than return_value_y_max."
        )


def load_distribution(filename):
    """Load accumulated_value(lead_time, index) from the NetCDF file."""

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file does not exist:\n{filename}")

    with xr.open_dataset(filename, decode_timedelta=False) as dataset:

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

    # Convert timedelta lead coordinates to whole days when necessary.
    lead_time = data["lead_time"]

    if np.issubdtype(lead_time.dtype, np.timedelta64):
        lead_days = (
            lead_time.values / np.timedelta64(1, "D")
        ).astype("int16")
    else:
        lead_days = lead_time.values.astype("int16")

    return data.assign_coords(
        lead_time=("lead_time", lead_days)
    )


def select_requested_lead_times(data):
    """Select the requested valid accumulated lead-time range."""

    available_leads = data["lead_time"].values.astype(int)

    selected_leads = available_leads[
        (available_leads >= first_valid_accumulation_lead)
        & (available_leads <= last_input_lead)
    ]

    if selected_leads.size == 0:
        raise ValueError(
            "No lead times fall within the requested range "
            f"{first_valid_accumulation_lead}-{last_input_lead}."
        )

    return data.sel(lead_time=selected_leads)


# =============================================================================
# Build grouped samples
# =============================================================================


def make_lead_time_groups(lead_times):
    """
    Split ordered lead times into consecutive groups.

    Example
    -------
    Leads 17-46 with lead_times_per_group = 5 become:

        [17, 18, 19, 20, 21]
        [22, 23, 24, 25, 26]
        ...
        [42, 43, 44, 45, 46]
    """

    lead_times = np.asarray(lead_times, dtype=int)
    lead_times = np.sort(lead_times)

    groups = []

    for start_index in range(0, lead_times.size, lead_times_per_group):
        group = lead_times[
            start_index:start_index + lead_times_per_group
        ]
        groups.append(group)

    return groups


def extract_grouped_values(data):
    """
    Pool all finite values from the lead times belonging to each group.

    Returns
    -------
    lead_groups : list of numpy arrays
        Lead-time numbers included in each group.
    grouped_values : list of numpy arrays
        One one-dimensional sample containing all values in each group.
    """

    lead_times = data["lead_time"].values.astype(int)
    lead_groups = make_lead_time_groups(lead_times)
    grouped_values = []

    for lead_group in lead_groups:

        # Select every lead in this group, flatten lead_time and index into one
        # dimension, and discard missing or infinite values.
        values = data.sel(lead_time=lead_group).values.ravel()
        values = values[np.isfinite(values)].astype("float64")

        if values.size < 2:
            group_label = f"{lead_group[0]}-{lead_group[-1]}"
            raise ValueError(
                f"Lead group {group_label} has fewer than two finite values."
            )

        grouped_values.append(values)

    return lead_groups, grouped_values


def resolve_bootstrap_sample_size(grouped_values):
    """Choose the common length of each bootstrap sample."""

    sample_counts = np.asarray(
        [values.size for values in grouped_values],
        dtype=int,
    )

    if bootstrap_sample_size == "minimum":
        return int(sample_counts.min())

    if bootstrap_sample_size == "median":
        return int(np.median(sample_counts))

    if isinstance(bootstrap_sample_size, (int, np.integer)):
        if bootstrap_sample_size < 2:
            raise ValueError(
                "An integer bootstrap_sample_size must be at least 2."
            )
        return int(bootstrap_sample_size)

    raise ValueError(
        "bootstrap_sample_size must be 'minimum', 'median', or an integer."
    )


# =============================================================================
# Empirical extreme-value calculations
# =============================================================================


def make_return_period_grid():
    """Create the common logarithmic return-period grid."""

    return np.geomspace(
        minimum_return_period,
        maximum_return_period,
        number_of_return_period_points,
    )


def return_periods_to_probabilities(return_periods):
    """Convert return periods T to non-exceedance probabilities p = 1 - 1/T."""

    return 1.0 - 1.0 / np.asarray(return_periods, dtype="float64")


def empirical_return_values(values, probabilities):
    """
    Calculate empirical return values at the requested probabilities.

    NumPy's linear quantile method corresponds to the commonly used type-7
    quantile. This is an empirical calculation; no distribution is fitted.
    """

    return np.quantile(
        np.asarray(values, dtype="float64"),
        probabilities,
        method="linear",
    )


def calculate_grouped_curves(grouped_values):
    """Calculate one empirical return-value curve for every lead group."""

    return_periods = make_return_period_grid()
    probabilities = return_periods_to_probabilities(return_periods)

    group_curves = [
        empirical_return_values(values, probabilities)
        for values in grouped_values
    ]

    # Pool every selected lead time and calculate its empirical reference curve.
    all_values = np.concatenate(grouped_values)
    all_lead_curve = empirical_return_values(all_values, probabilities)

    return (
        return_periods,
        probabilities,
        all_values,
        group_curves,
        all_lead_curve,
    )


def calculate_bootstrap_interval(
    all_values,
    probabilities,
    sample_size,
):
    """
    Bootstrap an empirical interval from all selected lead-time values.

    Sampling is with replacement. Every bootstrap sample has the same length,
    allowing the resulting return-value curves to be compared directly.
    """

    random_generator = np.random.default_rng(random_seed)

    bootstrap_curves = np.empty(
        (
            number_of_bootstrap_samples,
            probabilities.size,
        ),
        dtype="float32",
    )

    print(
        f"Calculating {number_of_bootstrap_samples:,} bootstrap curves "
        f"with sample size {sample_size:,}..."
    )

    for bootstrap_index in range(number_of_bootstrap_samples):

        bootstrap_values = random_generator.choice(
            all_values,
            size=sample_size,
            replace=True,
        )

        bootstrap_curves[bootstrap_index] = empirical_return_values(
            bootstrap_values,
            probabilities,
        )

    tail = (100.0 - confidence_interval_percent) / 2.0

    lower_curve = np.percentile(
        bootstrap_curves,
        tail,
        axis=0,
    )

    upper_curve = np.percentile(
        bootstrap_curves,
        100.0 - tail,
        axis=0,
    )

    return lower_curve, upper_curve, bootstrap_curves


# =============================================================================
# Plotting
# =============================================================================


def create_figure(
    grouped_values,
    return_periods,
    lead_groups,
    group_curves,
    all_lead_curve,
    lower_curve,
    upper_curve,
    bootstrap_curves,
):
    """Create a publication-quality two-panel grouped-distribution figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": tick_fontsize,
            "axes.labelsize": label_fontsize,
            "axes.titlesize": title_fontsize,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, (density_axis, return_axis) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    units = variable_units(variable)
    cmap=plt.get_cmap("tab20")
    group_colors=[cmap(i % cmap.N) for i in range(len(lead_groups))]
    value_label = (
        f"{x_days}-day accumulated "
        f"{variable_description(variable).capitalize()}"
    )
    if units:
        value_label += f" ({units})"

    # -------------------------------------------------------------------------
    # Panel 1: grouped probability-density functions
    # -------------------------------------------------------------------------
    for lead_group, values, color in zip(lead_groups, grouped_values, group_colors):
        group_label=f"Leads {lead_group[0]}-{lead_group[-1]}"
        density_axis.hist(values,bins=number_of_histogram_bins,density=True,
                          histtype="step",linewidth=1.5,
                          color=color,label=group_label)

    density_axis.set_xlabel(value_label)
    density_axis.set_ylabel("Probability density")
    # automatic histogram limits

    if density_y_min is not None or density_y_max is not None:
        current_min, current_max = density_axis.get_ylim()
        density_axis.set_ylim(
            density_y_min if density_y_min is not None else current_min,
            density_y_max if density_y_max is not None else current_max,
        )

    # Use simple colored line handles in the legend rather than the
    # rectangular handles returned by the step histograms.
    # Build compact line-style legend entries. For single-lead groups,
    # use "Lead 17" rather than the redundant "Leads 17-17".
    density_legend_handles = []

    for lead_group, color in zip(lead_groups, group_colors):

        if len(lead_group) == 1:
            group_label = f"Lead {lead_group[0]}"
        else:
            group_label = f"Leads {lead_group[0]}-{lead_group[-1]}"

        density_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.5,
                label=group_label,
            )
        )

    # Adapt the legend layout to the number of lead-time groups so that
    # large legends remain inside the left-hand panel.
    number_of_groups = len(lead_groups)

    if number_of_groups <= 6:
        legend_ncol = 1
        legend_fontsize_adaptive = legend_fontsize
    elif number_of_groups <= 12:
        legend_ncol = 2
        legend_fontsize_adaptive = legend_fontsize - 1
    elif number_of_groups <= 20:
        legend_ncol = 3
        legend_fontsize_adaptive = legend_fontsize - 1
    else:
        legend_ncol = 4
        legend_fontsize_adaptive = legend_fontsize - 2

    density_axis.legend(
        handles=density_legend_handles,
        frameon=False,
        loc="upper right",
        ncol=legend_ncol,
        fontsize=legend_fontsize_adaptive,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.5,
        labelspacing=0.4,
    )

    # -------------------------------------------------------------------------
    # Panel 2: grouped empirical return-period curves
    # -------------------------------------------------------------------------
    return_axis.fill_between(
        return_periods,
        lower_curve,
        upper_curve,
        color="0.6",
        alpha=confidence_alpha,
        linewidth=0.0,
        label=f"{confidence_interval_percent:g}% pooled bootstrap interval",
        zorder=1,
    )

    #return_axis.plot(
    #    return_periods,
    #    all_lead_curve,
    #    color="black",
    #    linewidth=2.4,
    #    label="All selected lead times",
    #    zorder=3,
    #)

    for lead_group, group_curve, color in zip(
        lead_groups,
        group_curves,
        group_colors,
    ):
        group_label = f"Leads {lead_group[0]}-{lead_group[-1]}"

        return_axis.plot(
            return_periods,
            group_curve,
            color=color,
            linewidth=group_line_width,
            label=group_label,
            zorder=2,
        )

    return_axis.set_xscale("log")
    return_axis.set_xlim(minimum_return_period, maximum_return_period)
    return_axis.set_xlabel("Return period (years)")
    return_axis.set_ylabel(value_label)

    if return_value_y_min is not None or return_value_y_max is not None:
        current_min, current_max = return_axis.get_ylim()
        return_axis.set_ylim(
            return_value_y_min
            if return_value_y_min is not None
            else current_min,
            return_value_y_max
            if return_value_y_max is not None
            else current_max,
        )

    
    # -------------------------------------------------------------------------
    # Inset: bootstrap distribution and lead-group values at one return period
    # -------------------------------------------------------------------------
    inset = inset_axes(
        return_axis,
        width="42%",
        height="42%",
        loc="upper left",
        bbox_to_anchor=(0.12, 0.02, 0.86, 0.93),
        bbox_transform=return_axis.transAxes,
    )

    inset_index = int(
        np.argmin(np.abs(return_periods - inset_centre_return_period))
    )
    inset_return_period = return_periods[inset_index]
    bootstrap_values_at_inset = bootstrap_curves[:, inset_index]
    group_values_at_inset = np.asarray(
        [curve[inset_index] for curve in group_curves]
    )
    all_lead_value_at_inset = all_lead_curve[inset_index]

    # Use the same percentile limits for the inset whiskers as for the
    # grey bootstrap confidence interval in the full return-period plot.
    #
    # For confidence_interval_percent = 95, this gives whiskers at the
    # 2.5th and 97.5th percentiles. Therefore, at the inset return period,
    # the whisker endpoints correspond exactly to the lower and upper
    # edges of the grey confidence band.
    tail = (100.0 - confidence_interval_percent) / 2.0

    inset.boxplot(
        bootstrap_values_at_inset,
        positions=[1.0],
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        whis=[tail, 100.0 - tail],
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.5},
    )

    dot_offsets = np.linspace(-0.12, 0.12, len(group_values_at_inset))
    for offset, group_value, color in zip(
        dot_offsets,
        group_values_at_inset,
        group_colors,
    ):
        inset.scatter(
            1.0 + offset,
            group_value,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            s=28,
            zorder=3,
        )

    #inset.scatter(
    #    1.0,
    #    all_lead_value_at_inset,
    #    color="black",
    #    marker="D",
    #    s=30,
    #    zorder=4,
    #)

    inset.set_xlim(0.55, 1.45)
    inset.set_xticks([1.0])
    inset.set_xticklabels([f"{int(round(inset_return_period))}-year return period"])
    inset.set_ylabel('mm', fontsize=tick_fontsize - 1)
    inset.tick_params(labelsize=tick_fontsize - 2)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    # The first panel already identifies the six color-coded lead groups.
    # Panel 2 therefore only needs to identify the bootstrap interval.
    bootstrap_handle, bootstrap_label = return_axis.get_legend_handles_labels()
    return_axis.legend(
        [bootstrap_handle[0]],
        [bootstrap_label[0]],
        frameon=False,
        loc="lower right",
        handlelength=2.4,
    )

    for axis in (density_axis, return_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=4.0,
            width=0.8,
        )
        axis.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=2.5,
            width=0.6,
        )

    figure.suptitle(
        plot_title,
        fontsize=title_fontsize,
        fontweight="normal",
    )

    return figure


# =============================================================================
# Diagnostics and output
# =============================================================================


def print_sample_summary(
    lead_groups,
    grouped_values,
    all_values,
    resample_size,
):
    """Print the group definitions and the number of values in each sample."""

    print()
    print("Grouped sample summary")
    print("----------------------")
    print(f"All selected leads:    {all_values.size:,} finite samples")
    print(f"Bootstrap sample size: {resample_size:,}")
    print()

    for lead_group, values in zip(lead_groups, grouped_values):
        group_label = f"{lead_group[0]}-{lead_group[-1]}"
        print(
            f"Leads {group_label:>5}: "
            f"{values.size:,} finite samples "
            f"from {lead_group.size} lead times"
        )


def save_figure(figure):
    """Save the figure as PDF and/or PNG according to the user settings."""

    os.makedirs(path_out, exist_ok=True)

    if save_pdf:
        filename_pdf = os.path.join(path_out, f"{filename_stem}.pdf")
        figure.savefig(filename_pdf, bbox_inches="tight")
        print("Wrote:", filename_pdf)

    if save_png:
        filename_png = os.path.join(path_out, f"{filename_stem}.png")
        figure.savefig(
            filename_png,
            dpi=figure_dpi,
            bbox_inches="tight",
        )
        print("Wrote:", filename_png)


# =============================================================================
# Main program
# =============================================================================


def main():
    """Run the grouped empirical extreme-value analysis."""

    validate_user_settings()

    print("Input file:")
    print(filename_in)
    print()
    print("Figure title:")
    print(plot_title)
    print()

    data = load_distribution(filename_in)
    data = select_requested_lead_times(data)

    lead_groups, grouped_values = extract_grouped_values(data)

    (
        return_periods,
        probabilities,
        all_values,
        group_curves,
        all_lead_curve,
    ) = calculate_grouped_curves(grouped_values)

    resample_size = resolve_bootstrap_sample_size(grouped_values)

    print_sample_summary(
        lead_groups,
        grouped_values,
        all_values,
        resample_size,
    )

    lower_curve, upper_curve, bootstrap_curves = calculate_bootstrap_interval(
        all_values,
        probabilities,
        resample_size,
    )

    figure = create_figure(
        grouped_values,
        return_periods,
        lead_groups,
        group_curves,
        all_lead_curve,
        lower_curve,
        upper_curve,
        bootstrap_curves,
    )

    save_figure(figure)

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
