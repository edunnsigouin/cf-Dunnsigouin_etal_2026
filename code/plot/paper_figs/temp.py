"""
Plot combined ECMWF S2S ensemble-member independence results.

This script reads the NetCDF output produced by the independence-calculation
script and combines forecast and hindcast pairwise Spearman correlations.

At each lead time, the plotted distribution contains:

    - all 1275 forecast member-pair correlations;
    - all 55 hindcast member-pair correlations.

The result is one combined box-and-whisker distribution at every lead time.

A two-sided Wilcoxon signed-rank test is applied at each lead time to assess
whether the centre of the combined correlation distribution differs from zero.
P-values can be corrected for testing multiple lead times.

IMPORTANT STATISTICAL NOTE
--------------------------
The pairwise correlations are not fully independent because many pairs share
an ensemble member. For example, pairs (1, 2) and (1, 3) both contain member 1.

The Wilcoxon test should therefore be interpreted as a diagnostic indication,
rather than as a fully rigorous formal test of ensemble independence.

Output
------
The script writes a publication-quality PDF and/or high-resolution PNG figure.

Required input variables
------------------------
forecast_spearman_rho(valid_month, lead_time, forecast_pair)
hindcast_spearman_rho(valid_month, lead_time, hindcast_pair)
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

# Input independence-test file
filename_in = (
    config.dirs["s2s_processed"]
    + "independence_spearman_tp24_2dayacc_"
    + "nve_catchment_regine_drammen_"
    + "lead17-46_all_2020-01-02_2020-01-30.nc"
)

# Output directory and filename stem
path_out = config.dirs["fig"]

filename_stem = (
    "independence_spearman_combined_tp24_2dayacc_"
    "nve_catchment_regine_drammen"
)

# Select which valid-month group to plot.
#
# Use:
#     0    when the calculation script used grouping = "all"
#     1–12 when the calculation script used grouping = "valid_month"
valid_month_selection = 0

# Statistical significance level
significance_level = 0.05

# Correction for testing multiple lead times:
#
# "holm"       : recommended
# "bonferroni" : more conservative
# "none"       : no correction
multiple_testing_correction = "holm"

# Figure settings
figure_width = 11.0
figure_height = 5.5
figure_dpi = 300

box_width = 0.65
show_outliers = False

# Show one x-axis label for every N lead times
label_every_n_leads = 2

# Save options
save_pdf = False
save_png = True
show_figure = True


# =============================================================================
# Validation and loading
# =============================================================================

def validate_user_settings():
    """Check plotting settings before reading the data."""

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

    if label_every_n_leads < 1:
        raise ValueError(
            "label_every_n_leads must be at least 1."
        )


def load_independence_results(filename):
    """Load forecast and hindcast pairwise Spearman correlations."""

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

    if "lead_time" not in dataset.coords:
        dataset.close()

        raise KeyError(
            "The input file does not contain a lead_time coordinate."
        )

    return dataset


def select_valid_month(data_array, valid_month):
    """
    Select one valid-month group.

    If the DataArray has no valid_month dimension, it is returned unchanged.
    """

    if "valid_month" not in data_array.dims:
        return data_array

    available_months = data_array["valid_month"].values.astype(int)

    if valid_month not in available_months:
        raise ValueError(
            f"valid_month_selection={valid_month} is unavailable. "
            f"Available values are {available_months.tolist()}."
        )

    return data_array.sel(valid_month=valid_month)


def combine_forecast_and_hindcast(
    forecast_correlations,
    hindcast_correlations,
):
    """
    Combine forecast and hindcast correlation distributions.

    The original forecast_pair and hindcast_pair dimensions are both renamed
    to a common dimension called pair before concatenation.

    Returns
    -------
    xarray.DataArray
        Dimensions:
            lead_time, pair
    """

    forecast = forecast_correlations.rename(
        {"forecast_pair": "pair"}
    )

    hindcast = hindcast_correlations.rename(
        {"hindcast_pair": "pair"}
    )

    # Replace the original pair coordinates so they do not overlap.
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
# Statistical testing
# =============================================================================

def one_sample_wilcoxon(values):
    """
    Test whether a correlation distribution is centred on zero.

    The two-sided Wilcoxon signed-rank test evaluates the null hypothesis
    that the distribution is symmetric around zero.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    # Exact zeros carry no information for the signed-rank test.
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
    """Correct p-values for multiple lead-time tests."""

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

    # Holm step-down correction
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
    """
    Calculate one Wilcoxon test for each lead time.

    Returns
    -------
    dict
        Arrays containing lead times, medians, raw p-values, adjusted
        p-values, and significance flags.
    """

    lead_times = correlations["lead_time"].values.astype(int)

    medians = []
    raw_p_values = []

    for lead_time in lead_times:

        values = correlations.sel(
            lead_time=lead_time
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
        "lead_time": lead_times,
        "median_rho": np.asarray(medians, dtype=float),
        "raw_p_value": raw_p_values,
        "adjusted_p_value": adjusted_p_values,
        "significant": (
            adjusted_p_values < significance_level
        ),
    }


# =============================================================================
# Plotting
# =============================================================================

def significance_label(p_value):
    """Return a conventional significance label."""

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
    """Return one finite correlation array per lead time."""

    lead_times = correlations["lead_time"].values.astype(int)

    values_by_lead = []

    for lead_time in lead_times:

        values = correlations.sel(
            lead_time=lead_time
        ).values.ravel()

        values = values[np.isfinite(values)]

        values_by_lead.append(values)

    return lead_times, values_by_lead


def style_boxplot(boxplot):
    """Apply restrained publication-style boxplot formatting."""

    for box in boxplot["boxes"]:
        box.set_linewidth(1.0)

    for whisker in boxplot["whiskers"]:
        whisker.set_linewidth(0.9)

    for cap in boxplot["caps"]:
        cap.set_linewidth(0.9)

    for median in boxplot["medians"]:
        median.set_linewidth(1.4)


def add_significance_markers(
    axis,
    adjusted_p_values,
    values_by_lead,
):
    """Add adjusted-p-value significance symbols above the boxplots."""

    nonempty_values = [
        values
        for values in values_by_lead
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
                fontsize=8,
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
    test_results,
):
    """Create the publication-quality combined boxplot figure."""

    lead_times, values_by_lead = prepare_boxplot_values(
        combined_correlations
    )

    positions = np.arange(
        1,
        len(lead_times) + 1,
    )

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
    )

    style_boxplot(boxplot)

    axis.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
        zorder=0,
    )

    visible_labels = [
        str(lead_time)
        if index % label_every_n_leads == 0
        else ""
        for index, lead_time in enumerate(lead_times)
    ]

    axis.set_xticks(positions)
    axis.set_xticklabels(visible_labels)

    axis.set_xlabel("Lead time (days)")
    axis.set_ylabel("Spearman rank correlation")

    axis.set_title(
        "Ensemble-member independence by forecast lead time",
        fontweight="bold",
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
    )

    axis.set_xlim(
        0.3,
        len(lead_times) + 0.7,
    )

    add_significance_markers(
        axis=axis,
        adjusted_p_values=test_results[
            "adjusted_p_value"
        ],
        values_by_lead=values_by_lead,
    )

    correction_text = {
        "holm": "Holm-adjusted",
        "bonferroni": "Bonferroni-adjusted",
        "none": "Unadjusted",
    }[multiple_testing_correction]

    figure.text(
        0.5,
        0.005,
        (
            "Forecast and hindcast pairwise correlations are pooled. "
            "Boxes show the interquartile range, centre lines show medians, "
            "and whiskers extend to 1.5 × IQR. "
            f"Asterisks indicate {correction_text} two-sided Wilcoxon "
            f"p-values: * p<0.05, ** p<0.01, *** p<0.001."
        ),
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8,
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

    dataset = load_independence_results(
        filename=filename_in
    )

    try:
        forecast_correlations = select_valid_month(
            dataset["forecast_spearman_rho"],
            valid_month=valid_month_selection,
        ).load()

        hindcast_correlations = select_valid_month(
            dataset["hindcast_spearman_rho"],
            valid_month=valid_month_selection,
        ).load()

    finally:
        dataset.close()

    combined_correlations = combine_forecast_and_hindcast(
        forecast_correlations=forecast_correlations,
        hindcast_correlations=hindcast_correlations,
    )

    test_results = calculate_test_results(
        correlations=combined_correlations
    )

    print()
    print("Lead-time test results")
    print("----------------------")

    for (
        lead_time,
        median_rho,
        raw_p_value,
        adjusted_p_value,
    ) in zip(
        test_results["lead_time"],
        test_results["median_rho"],
        test_results["raw_p_value"],
        test_results["adjusted_p_value"],
    ):
        print(
            f"Lead {lead_time:2d}: "
            f"median rho={median_rho: .4f}, "
            f"raw p={raw_p_value:.4g}, "
            f"adjusted p={adjusted_p_value:.4g}"
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
