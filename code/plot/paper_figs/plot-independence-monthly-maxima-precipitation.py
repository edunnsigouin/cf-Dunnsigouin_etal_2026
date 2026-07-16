"""
Plot ensemble-member independence for monthly maximum N-day precipitation.

This script reads the NetCDF output from:

    calculate_s2s_independence_monthly_maxima.py

Forecast and hindcast pairwise Spearman correlations are pooled into one
distribution for each assigned calendar month. The figure therefore contains
12 boxplots, one for January through December.

Each monthly distribution contains:

    - 1275 forecast member-pair correlations, when 51 forecast members exist;
    - 55 hindcast member-pair correlations, when 11 hindcast members exist.

Optional significance testing
-----------------------------
When show_significance = True, a two-sided Wilcoxon signed-rank test is
calculated for each monthly distribution to assess whether its centre differs
from zero. P-values can be corrected across the 12 calendar months.

Important:
The pairwise correlations are not fully independent because many pairs share
an ensemble member. The Wilcoxon test should therefore be interpreted as a
diagnostic rather than a fully rigorous formal test.

Required input variables
------------------------
forecast_spearman_rho(assigned_month, forecast_pair)
hindcast_spearman_rho(assigned_month, hindcast_pair)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import wilcoxon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Data settings ---------------------------------------------------------------

variable = "tp24"
x_days = 2

# Use the same catchment name as in the calculation script.
catchment = "regine_drammen"

forecast_date_range = (
    "2020-01-02",
    "2023-06-26",
)

# Daily lead times available in the original files.
first_input_lead = 16
last_input_lead = 46

# The first usable ending lead for an N-day accumulation.
first_usable_accumulation_lead = (
    first_input_lead + x_days - 1
)


# Significance settings -------------------------------------------------------

# Set to False to skip both the statistical test and significance asterisks.
show_significance = False

significance_level = 0.05

# Correction for testing 12 calendar months:
#
# "holm"       : Holm step-down correction; recommended
# "bonferroni" : Bonferroni correction; more conservative
# "none"       : no correction
multiple_testing_correction = "holm"


# Figure settings -------------------------------------------------------------

figure_width = 10.5
figure_height = 5.5
figure_dpi = 300

box_width = 0.62
show_outliers = True

label_fontsize = 11
title_fontsize = 12
tick_fontsize = 10
significance_fontsize = 8


# Save settings ---------------------------------------------------------------

path_out = config.dirs["fig"]

save_pdf = False
save_png = True
show_figure = True


# =============================================================================
# Descriptive labels and filenames
# =============================================================================

MONTH_ABBREVIATIONS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
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


def build_plot_title():
    """Construct a publication-style title."""

    catchment_label = readable_catchment_name(catchment)
    variable_label = variable_description(variable)

    return (
        f"Ensemble-member independence of monthly maximum "
        f"{x_days}-day accumulated {variable_label} "
        f"over the {catchment_label} catchment"
    )


def build_input_filename():
    """Construct the NetCDF filename produced by the calculation script."""

    return (
        config.dirs["s2s_processed"]
        + f"independence_spearman_monthly_max_{variable}_"
        + f"{x_days}dayacc_"
        + f"nve_catchment_{catchment}_"
        + f"lead{first_usable_accumulation_lead}-"
        + f"{last_input_lead}_"
        + f"{forecast_date_range[0]}_"
        + f"{forecast_date_range[1]}.nc"
    )


def build_output_filename_stem():
    """Construct a descriptive output filename stem."""

    significance_text = (
        f"significance-{multiple_testing_correction}"
        if show_significance
        else "no-significance"
    )

    return (
        f"independence_test_monthly_maxima_"
        f"{variable}_"
        f"{x_days}day-accumulation_"
        f"{catchment}_"
        f"lead{first_usable_accumulation_lead}-{last_input_lead}_"
        f"{forecast_date_range[0]}-to-{forecast_date_range[1]}"
    )


filename_in = build_input_filename()
filename_stem = build_output_filename_stem()
plot_title = build_plot_title()


# =============================================================================
# Validation and loading
# =============================================================================

def validate_user_settings():
    """Check settings before reading and plotting the data."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    if min(
        label_fontsize,
        title_fontsize,
        tick_fontsize,
        significance_fontsize,
    ) <= 0:
        raise ValueError(
            "All font sizes must be greater than zero."
        )

    if show_significance:

        if not 0 < significance_level < 1:
            raise ValueError(
                "significance_level must be between 0 and 1."
            )

        valid_corrections = {
            "holm",
            "bonferroni",
            "none",
        }

        if multiple_testing_correction not in valid_corrections:
            raise ValueError(
                "multiple_testing_correction must be one of "
                f"{sorted(valid_corrections)}."
            )


def load_independence_results(filename):
    """Load monthly forecast and hindcast correlation distributions."""

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Input file does not exist:\n{filename}"
        )

    dataset = xr.open_dataset(filename)

    required_variables = {
        "forecast_spearman_rho",
        "hindcast_spearman_rho",
    }

    missing_variables = required_variables.difference(
        dataset.data_vars
    )

    if missing_variables:
        dataset.close()

        raise KeyError(
            "The input file is missing these required variables: "
            f"{sorted(missing_variables)}"
        )

    if "assigned_month" not in dataset.coords:
        dataset.close()

        raise KeyError(
            "The input file does not contain the assigned_month coordinate."
        )

    return dataset


def combine_forecast_and_hindcast(
    forecast_correlations,
    hindcast_correlations,
):
    """
    Pool forecast and hindcast pair distributions for every month.

    The forecast_pair and hindcast_pair dimensions are renamed to a common
    dimension called pair before concatenation.
    """

    forecast = forecast_correlations.rename(
        {"forecast_pair": "pair"}
    )

    hindcast = hindcast_correlations.rename(
        {"hindcast_pair": "pair"}
    )

    # Give the two sets of pairs non-overlapping coordinate values.
    forecast = forecast.assign_coords(
        pair=np.arange(
            forecast.sizes["pair"],
            dtype="int32",
        )
    )

    hindcast = hindcast.assign_coords(
        pair=np.arange(
            forecast.sizes["pair"],
            forecast.sizes["pair"] + hindcast.sizes["pair"],
            dtype="int32",
        )
    )

    combined = xr.concat(
        [forecast, hindcast],
        dim="pair",
        join="exact",
    )

    combined.name = "combined_spearman_rho"

    return combined


# =============================================================================
# Optional statistical testing
# =============================================================================

def one_sample_wilcoxon(values):
    """Test whether one monthly correlation distribution is centred on zero."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    # Exact zeros do not contribute information to the signed-rank test.
    values = values[values != 0.0]

    if values.size == 0:
        return np.nan, np.nan

    statistic, p_value = wilcoxon(
        values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return float(statistic), float(p_value)


def adjust_p_values(p_values, method):
    """Correct p-values for multiple monthly tests."""

    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)

    valid = np.isfinite(p_values)
    valid_p_values = p_values[valid]

    if valid_p_values.size == 0:
        return adjusted

    if method == "none":
        adjusted[valid] = valid_p_values
        return adjusted

    number_of_tests = valid_p_values.size

    if method == "bonferroni":
        adjusted[valid] = np.minimum(
            valid_p_values * number_of_tests,
            1.0,
        )
        return adjusted

    # Holm step-down correction.
    order = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[order]

    adjusted_sorted = np.empty_like(sorted_p_values)

    for rank, p_value in enumerate(sorted_p_values):
        adjusted_sorted[rank] = (
            number_of_tests - rank
        ) * p_value

    adjusted_sorted = np.maximum.accumulate(
        adjusted_sorted
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted,
        1.0,
    )

    restored_order = np.empty_like(adjusted_sorted)
    restored_order[order] = adjusted_sorted

    adjusted[valid] = restored_order

    return adjusted


def calculate_test_results(correlations):
    """Calculate one Wilcoxon test for each assigned calendar month."""

    months = correlations["assigned_month"].values.astype(int)

    medians = []
    raw_p_values = []

    for month in months:

        values = correlations.sel(
            assigned_month=month
        ).values.ravel()

        values = values[np.isfinite(values)]

        medians.append(
            np.median(values)
            if values.size > 0
            else np.nan
        )

        _, p_value = one_sample_wilcoxon(values)
        raw_p_values.append(p_value)

    raw_p_values = np.asarray(
        raw_p_values,
        dtype=float,
    )

    adjusted_p_values = adjust_p_values(
        raw_p_values,
        method=multiple_testing_correction,
    )

    return {
        "assigned_month": months,
        "median_rho": np.asarray(
            medians,
            dtype=float,
        ),
        "raw_p_value": raw_p_values,
        "adjusted_p_value": adjusted_p_values,
        "significant": (
            adjusted_p_values < significance_level
        ),
    }


def print_test_results(test_results):
    """Print monthly significance-test results."""

    print()
    print("Monthly test results")
    print("--------------------")

    for (
        month,
        median_rho,
        raw_p_value,
        adjusted_p_value,
    ) in zip(
        test_results["assigned_month"],
        test_results["median_rho"],
        test_results["raw_p_value"],
        test_results["adjusted_p_value"],
    ):

        month_label = MONTH_ABBREVIATIONS[int(month)]

        print(
            f"{month_label}: "
            f"median rho={median_rho: .4f}, "
            f"raw p={raw_p_value:.4g}, "
            f"adjusted p={adjusted_p_value:.4g}"
        )


# =============================================================================
# Plotting
# =============================================================================

def significance_label(p_value):
    """Return the conventional asterisk label for a p-value."""

    if not np.isfinite(p_value):
        return ""

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return ""


def prepare_boxplot_values(correlations):
    """Return one finite correlation array for each calendar month."""

    months = correlations["assigned_month"].values.astype(int)
    values_by_month = []

    for month in months:

        values = correlations.sel(
            assigned_month=month
        ).values.ravel()

        values = values[np.isfinite(values)]
        values_by_month.append(values)

    return months, values_by_month


def style_boxplot(boxplot):
    """Apply clear publication-style formatting."""

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


def add_significance_markers(
    axis,
    adjusted_p_values,
    values_by_month,
):
    """Add adjusted-p-value significance asterisks above boxplots."""

    nonempty_values = [
        values
        for values in values_by_month
        if values.size > 0
    ]

    if not nonempty_values:
        return

    all_values = np.concatenate(nonempty_values)

    data_min = np.nanmin(all_values)
    data_max = np.nanmax(all_values)
    data_range = data_max - data_min

    if data_range == 0:
        data_range = 1.0

    marker_height = data_max + 0.08 * data_range

    for position, p_value in enumerate(
        adjusted_p_values,
        start=1,
    ):

        label = significance_label(p_value)

        if label:
            axis.text(
                position,
                marker_height,
                label,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=significance_fontsize,
            )

    current_lower, current_upper = axis.get_ylim()

    axis.set_ylim(
        current_lower,
        max(
            current_upper,
            marker_height + 0.08 * data_range,
        ),
    )


def create_figure(
    combined_correlations,
    test_results=None,
):
    """Create the monthly box-and-whisker figure."""

    months, values_by_month = prepare_boxplot_values(
        combined_correlations
    )

    positions = np.arange(
        1,
        len(months) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    boxplot = axis.boxplot(
        values_by_month,
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

    axis.axhline(
        0.0,
        color="black",
        linewidth=0.9,
        linestyle="-",
        zorder=0,
    )

    month_labels = [
        MONTH_ABBREVIATIONS[int(month)]
        for month in months
    ]

    axis.set_xticks(positions)
    axis.set_xticklabels(
        month_labels,
        fontsize=tick_fontsize,
    )

    axis.set_xlabel(
        "Month",
        fontsize=label_fontsize,
    )

    axis.set_ylabel(
        "Spearman rank correlation",
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
        len(months) + 0.7,
    )

    if show_significance and test_results is not None:
        add_significance_markers(
            axis=axis,
            adjusted_p_values=test_results[
                "adjusted_p_value"
            ],
            values_by_month=values_by_month,
        )

    return figure


# =============================================================================
# Output
# =============================================================================

def save_figure(figure):
    """Save the figure as PDF and/or PNG."""

    os.makedirs(
        path_out,
        exist_ok=True,
    )

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

    dataset = load_independence_results(
        filename=filename_in
    )

    try:
        forecast_correlations = dataset[
            "forecast_spearman_rho"
        ].load()

        hindcast_correlations = dataset[
            "hindcast_spearman_rho"
        ].load()

    finally:
        dataset.close()

    combined_correlations = combine_forecast_and_hindcast(
        forecast_correlations=forecast_correlations,
        hindcast_correlations=hindcast_correlations,
    )

    test_results = None

    if show_significance:

        test_results = calculate_test_results(
            correlations=combined_correlations
        )

        print_test_results(test_results)

    else:

        print()
        print(
            "Significance calculation and asterisks are disabled."
        )

    figure = create_figure(
        combined_correlations=combined_correlations,
        test_results=test_results,
    )

    save_figure(figure)

    if show_figure:
        plt.show()
    else:
        plt.close(figure)
