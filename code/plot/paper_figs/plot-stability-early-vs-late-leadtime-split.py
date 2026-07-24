"""
Plot monthly probability-density distributions of complete-window and
lead-location extreme precipitation samples.

Purpose
-------
This script reads the NetCDF file produced by the lead-location sampling
script. That input contains:

    1. The complete-window maximum precipitation sample:
           max_value_lead<start>_<end>

    2. Lead-location subsets of those SAME maxima:
           max_value_lead<start>_<end>
           max_value_lead<start>_<end>
           ...

For the default two-bin case used here:

    all leads : complete-window maxima over ending leads 17-46
    early     : subset of those maxima occurring at ending leads 17-31
    late      : subset of those maxima occurring at ending leads 32-46

The early and late distributions are therefore subsets of the all-lead
distribution; they are not maxima recomputed over shorter lead windows.

Figure
------
A 3 x 4 publication-style figure is created, with one panel per calendar
month. Each panel shows the probability density of:

    - early lead-time subgroup: tab:blue
    - late lead-time subgroup: tab:orange

Each panel legend includes the sample number in each group. A two-sample
Kolmogorov-Smirnov (KS) test is also performed between the Early and Late
samples for each month. The null hypothesis is that both samples come from
the same continuous distribution. The user-defined significance level is
used to decide whether to reject the null hypothesis.

Within each month all three distributions use the same histogram bin edges,
which allows their probability-density shapes to be compared directly.

The script also prints the number of finite samples in each group for every
calendar month. For an input generated with two lead bins, the expected
relationship is:

    n_all = n_early + n_late

for each month.

The script is intentionally written in a simple structure:
    1. imports
    2. user settings
    3. constants
    4. helper functions
    5. main program
"""

# =============================================================================
# Imports
# =============================================================================

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import ks_2samp

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Sampling setup. These settings are used to construct the expected input
# filename and variable names.
variable = "tp24"
x_days = 2
catchment = "regine_drammen"

forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Optional explicit input filename.
#
# Leave as None to construct the filename automatically from the settings
# above. Set this to a complete path if the NetCDF file has been renamed.
input_filename_override = None

# Histogram settings.
number_of_bins = 35

# Two-sample KS test settings.
# Example: significance_level_percent = 95.0 corresponds to alpha = 0.05.
significance_level_percent = 95.0
ks_alternative = "two-sided"
ks_method = "auto"

# Set True to use the same x-axis limits for all 12 panels. This makes the
# precipitation magnitudes especially easy to compare across seasons.
use_common_x_limits = False

# Small fractional padding added to plotted x/y ranges.
x_axis_margin_fraction = 0.02
y_axis_margin_fraction = 0.08

# Figure settings.
figure_width = 13.0
figure_height = 9.5
figure_dpi = 300

write2file = False
show_figure = True


# =============================================================================
# Plotting constants
# =============================================================================

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

# Requested colors.
ALL_COLOR = "black"
EARLY_COLOR = "tab:blue"
LATE_COLOR = "tab:orange"

LINE_WIDTH = 1.6

TITLE_FONTSIZE = 10
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12


# =============================================================================
# Lead-time and filename helpers
# =============================================================================

def validate_user_settings() -> None:
    """Check the user settings before opening the data."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = first_input_lead + x_days - 1

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )

    number_of_usable_leads = (
        last_input_lead - first_usable_lead + 1
    )

    if number_of_lead_bins != 2:
        raise ValueError(
            "This plotting script is designed for exactly two lead bins "
            "(early and late). Set number_of_lead_bins = 2."
        )

    if number_of_lead_bins > number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins exceeds the number of usable leads."
        )

    if number_of_bins < 1:
        raise ValueError("number_of_bins must be at least 1.")

    if not 0.0 < significance_level_percent < 100.0:
        raise ValueError(
            "significance_level_percent must be between 0 and 100."
        )

    if ks_alternative not in {"two-sided", "less", "greater"}:
        raise ValueError(
            "ks_alternative must be 'two-sided', 'less', or 'greater'."
        )

    if ks_method not in {"auto", "exact", "asymp"}:
        raise ValueError(
            "ks_method must be 'auto', 'exact', or 'asymp'."
        )


def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """
    Split usable accumulated ending leads into approximately equal bins.

    For the default two-day setup:
        usable leads = 17-46
        two bins     = 17-31 and 32-46
    """

    number_of_leads = last_lead - first_lead + 1

    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size
        + int(bin_index >= number_of_bins - remainder)
        for bin_index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1

    return bins


def get_lead_ranges() -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    """Return the full, early, and late accumulated lead ranges."""

    first_usable_lead = first_input_lead + x_days - 1

    full_range = (
        first_usable_lead,
        last_input_lead,
    )

    split_ranges = split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=2,
    )

    early_range = split_ranges[0]
    late_range = split_ranges[1]

    return full_range, early_range, late_range


def build_input_filename() -> str:
    """Construct the output filename generated by the sampling script."""

    full_range, early_range, late_range = get_lead_ranges()

    lead_label = (
        f"lead{full_range[0]}-{full_range[1]}_"
        f"split2_"
        f"{early_range[0]}-{early_range[1]}_"
        f"{late_range[0]}-{late_range[1]}"
    )

    return os.path.join(
        config.dirs["s2s_processed"],
        (
            f"lt_maxima_binning_distribution_monthly_extremes_"
            f"{variable}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_label}_"
            f"forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )


def resolve_input_filename() -> str:
    """Return either the explicit filename or the automatically built one."""

    if input_filename_override is not None:
        return input_filename_override

    return build_input_filename()


def get_variable_names() -> tuple[str, str, str]:
    """Return expected NetCDF variable names for all, early, and late groups."""

    full_range, early_range, late_range = get_lead_ranges()

    all_variable = (
        f"max_value_lead{full_range[0]}_{full_range[1]}"
    )

    early_variable = (
        f"max_value_lead{early_range[0]}_{early_range[1]}"
    )

    late_variable = (
        f"max_value_lead{late_range[0]}_{late_range[1]}"
    )

    return all_variable, early_variable, late_variable


def build_output_filename() -> str:
    """Build a descriptive figure filename."""

    full_range, early_range, late_range = get_lead_ranges()

    return os.path.join(
        config.dirs["fig"],
        (
            f"monthly_probability_density_"
            f"{variable}_{x_days}dayacc_"
            f"{catchment}_"
            f"lead{full_range[0]}-{full_range[1]}_"
            f"early{early_range[0]}-{early_range[1]}_"
            f"late{late_range[0]}-{late_range[1]}.png"
        ),
    )


# =============================================================================
# Data loading
# =============================================================================

def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """Flatten an array and retain only finite values."""

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def check_input_dataset(
    ds: xr.Dataset,
    variable_names: tuple[str, str, str],
) -> None:
    """Check that the required monthly coordinate and variables exist."""

    if "month_of_year" not in ds.coords:
        raise KeyError(
            "Input dataset does not contain the 'month_of_year' coordinate."
        )

    missing = [
        name
        for name in variable_names
        if name not in ds.data_vars
    ]

    if missing:
        raise KeyError(
            "Input dataset is missing expected variables:\n"
            + "\n".join(f"  {name}" for name in missing)
            + "\n\nAvailable variables:\n"
            + "\n".join(f"  {name}" for name in ds.data_vars)
        )


def load_monthly_distributions(
    filename: str,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Load all, early, and late extreme samples for every calendar month.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Input file does not exist:\n{filename}"
        )

    (
        all_variable,
        early_variable,
        late_variable,
    ) = get_variable_names()

    monthly_data = {}

    with xr.open_dataset(filename) as ds:

        check_input_dataset(
            ds=ds,
            variable_names=(
                all_variable,
                early_variable,
                late_variable,
            ),
        )

        for month in range(1, 13):

            all_values = remove_missing_values(
                ds[all_variable]
                .sel(month_of_year=month)
                .values
            )

            early_values = remove_missing_values(
                ds[early_variable]
                .sel(month_of_year=month)
                .values
            )

            late_values = remove_missing_values(
                ds[late_variable]
                .sel(month_of_year=month)
                .values
            )

            monthly_data[month] = {
                "all": all_values,
                "early": early_values,
                "late": late_values,
            }

    return monthly_data


# =============================================================================
# Diagnostics and histogram helpers
# =============================================================================

def validate_partition(
    monthly_data: dict[int, dict[str, np.ndarray]],
) -> None:
    """
    Check that early and late samples partition the complete-window sample.
    """

    failed_months = []

    for month in range(1, 13):
        n_all = monthly_data[month]["all"].size
        n_early = monthly_data[month]["early"].size
        n_late = monthly_data[month]["late"].size

        if n_all != n_early + n_late:
            failed_months.append(month)

    if failed_months:
        month_text = ", ".join(
            MONTH_NAMES[month - 1]
            for month in failed_months
        )

        raise ValueError(
            "Early + late sample counts do not equal the all-lead sample "
            f"for: {month_text}."
        )


def print_sample_counts(
    monthly_data: dict[int, dict[str, np.ndarray]],
) -> None:
    """Print monthly all/early/late sample sizes."""

    print()
    print("Sample counts")
    print("-------------")

    header = (
        f"{'Month':<12}"
        f"{'All':>10}"
        f"{'Early':>10}"
        f"{'Late':>10}"
        f"{'Check':>10}"
    )

    print(header)
    print("-" * len(header))

    total_all = 0
    total_early = 0
    total_late = 0

    for month, month_name in enumerate(MONTH_NAMES, start=1):

        n_all = monthly_data[month]["all"].size
        n_early = monthly_data[month]["early"].size
        n_late = monthly_data[month]["late"].size

        check = "OK" if n_all == n_early + n_late else "FAIL"

        print(
            f"{month_name:<12}"
            f"{n_all:>10d}"
            f"{n_early:>10d}"
            f"{n_late:>10d}"
            f"{check:>10}"
        )

        total_all += n_all
        total_early += n_early
        total_late += n_late

    print("-" * len(header))

    total_check = (
        "OK"
        if total_all == total_early + total_late
        else "FAIL"
    )

    print(
        f"{'TOTAL':<12}"
        f"{total_all:>10d}"
        f"{total_early:>10d}"
        f"{total_late:>10d}"
        f"{total_check:>10}"
    )


def get_global_x_limits(
    monthly_data: dict[int, dict[str, np.ndarray]],
) -> tuple[float, float]:
    """Return precipitation limits covering all months and groups."""

    values = np.concatenate(
        [
            monthly_data[month][group]
            for month in range(1, 13)
            for group in ("all", "early", "late")
            if monthly_data[month][group].size > 0
        ]
    )

    x_min = float(np.min(values))
    x_max = float(np.max(values))

    if np.isclose(x_min, x_max):
        padding = max(1.0, abs(x_min) * 0.05)
    else:
        padding = (x_max - x_min) * x_axis_margin_fraction

    return x_min - padding, x_max + padding


def calculate_month_bin_edges(
    month_data: dict[str, np.ndarray],
    common_x_limits: tuple[float, float] | None,
) -> np.ndarray:
    """
    Calculate histogram edges shared by all three groups in one month.
    """

    if common_x_limits is not None:
        x_min, x_max = common_x_limits
    else:
        combined = np.concatenate(
            [
                month_data["all"],
                month_data["early"],
                month_data["late"],
            ]
        )

        x_min = float(np.min(combined))
        x_max = float(np.max(combined))

        if np.isclose(x_min, x_max):
            padding = max(1.0, abs(x_min) * 0.05)
        else:
            padding = (
                (x_max - x_min)
                * x_axis_margin_fraction
            )

        x_min -= padding
        x_max += padding

    return np.linspace(
        x_min,
        x_max,
        number_of_bins + 1,
    )



def get_significance_alpha() -> float:
    """Return the KS-test alpha threshold from the confidence percentage."""

    return 1.0 - significance_level_percent / 100.0


def perform_ks_test(
    early_values: np.ndarray,
    late_values: np.ndarray,
) -> tuple[float, float, bool]:
    """
    Perform a two-sample Kolmogorov-Smirnov test.

    Returns
    -------
    d_statistic : float
        Maximum distance between the two empirical cumulative distributions.

    p_value : float
        Probability, under the null hypothesis of equal distributions, of
        obtaining a KS statistic at least this large.

    reject_null : bool
        True when p_value < alpha.
    """

    if early_values.size == 0 or late_values.size == 0:
        return np.nan, np.nan, False

    result = ks_2samp(
        early_values,
        late_values,
        alternative=ks_alternative,
        method=ks_method,
    )

    d_statistic = float(result.statistic)
    p_value = float(result.pvalue)
    reject_null = p_value < get_significance_alpha()

    return d_statistic, p_value, reject_null


def format_p_value(p_value: float) -> str:
    """Format p-values compactly for panel annotations."""

    if not np.isfinite(p_value):
        return "NA"

    if p_value < 0.001:
        return f"{p_value:.1e}"

    return f"{p_value:.3f}"


# =============================================================================
# Plotting
# =============================================================================

def format_axis(ax: plt.Axes) -> None:
    """Apply consistent publication-style axis formatting."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=TICK_LABEL_FONTSIZE,
    )


def plot_month_panel(
    ax: plt.Axes,
    month: int,
    month_data: dict[str, np.ndarray],
    bin_edges: np.ndarray,
) -> None:
    """
    Plot Early and Late probability densities and annotate the KS test.
    """

    early_values = month_data["early"]
    late_values = month_data["late"]

    maximum_density = 0.0

    plot_specs = (
        (
            early_values,
            EARLY_COLOR,
            f"Early leads (n={early_values.size})",
            2,
        ),
        (
            late_values,
            LATE_COLOR,
            f"Late leads (n={late_values.size})",
            1,
        ),
    )

    for values, color, label, zorder in plot_specs:

        if values.size == 0:
            continue

        density, _, _ = ax.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=LINE_WIDTH,
            color=color,
            label=label,
            zorder=zorder,
        )

        if density.size > 0:
            maximum_density = max(
                maximum_density,
                float(np.nanmax(density)),
            )

    d_statistic, p_value, reject_null = perform_ks_test(
        early_values=early_values,
        late_values=late_values,
    )

    decision_text = (
        r"Reject $H_0$"
        if reject_null
        else r"Do not reject $H_0$"
    )

    ks_label = (
        f"KS: D={d_statistic:.3f}, "
        f"p={format_p_value(p_value)}\n"
        f"{decision_text} "
        f"({significance_level_percent:g}% level)"
    )

    # Add a dummy line so the KS result appears in the panel legend.
    ax.plot(
        [],
        [],
        linestyle="none",
        label=ks_label,
    )

    ax.set_xlim(
        bin_edges[0],
        bin_edges[-1],
    )

    if maximum_density > 0:
        ax.set_ylim(
            0,
            maximum_density
            * (1.0 + y_axis_margin_fraction),
        )

    ax.set_title(
        MONTH_NAMES[month - 1],
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        borderaxespad=0.2,
    )

    format_axis(ax)


def create_figure(
    monthly_data: dict[int, dict[str, np.ndarray]],
) -> plt.Figure:
    """Create the 3 x 4 monthly probability-density figure."""

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(figure_width, figure_height),
        squeeze=False,
        sharex=use_common_x_limits,
    )

    common_x_limits = (
        get_global_x_limits(monthly_data)
        if use_common_x_limits
        else None
    )

    for month in range(1, 13):

        row = (month - 1) // 4
        column = (month - 1) % 4

        ax = axes[row, column]

        bin_edges = calculate_month_bin_edges(
            month_data=monthly_data[month],
            common_x_limits=common_x_limits,
        )

        plot_month_panel(
            ax=ax,
            month=month,
            month_data=monthly_data[month],
            bin_edges=bin_edges,
        )

        # Show x-axis labels only on the bottom row.
        if row == 2:
            ax.set_xlabel(
                f"{x_days}-day maximum precipitation [mm]",
                fontsize=AXIS_LABEL_FONTSIZE,
            )

        # Show y-axis labels only on the left column.
        if column == 0:
            ax.set_ylabel(
                "Probability density",
                fontsize=AXIS_LABEL_FONTSIZE,
            )

    full_range, early_range, late_range = get_lead_ranges()

    fig.suptitle(
        (
            f"Monthly {x_days}-day accumulated precipitation maxima\n"
            f"Early {early_range[0]}-{early_range[1]} vs "
            f"Late {late_range[0]}-{late_range[1]}"
        ),
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="normal",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.86,
        wspace=0.24,
        hspace=0.30,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    input_filename = resolve_input_filename()
    output_filename = build_output_filename()

    full_range, early_range, late_range = get_lead_ranges()

    print("Input file")
    print("----------")
    print(input_filename)

    print()
    print("Lead-time groups")
    print("----------------")
    print(
        f"All leads:   {full_range[0]}-{full_range[1]}"
    )
    print(
        f"Early leads: {early_range[0]}-{early_range[1]}"
    )
    print(
        f"Late leads:  {late_range[0]}-{late_range[1]}"
    )

    monthly_data = load_monthly_distributions(
        input_filename
    )

    validate_partition(
        monthly_data
    )

    print_sample_counts(
        monthly_data
    )

    print()
    print("Two-sample KS tests")
    print("-------------------")
    print(
        f"Significance level: {significance_level_percent:g}% "
        f"(alpha={get_significance_alpha():.3f})"
    )

    print(
        f"{'Month':<12}"
        f"{'D':>10}"
        f"{'p-value':>14}"
        f"{'Decision':>22}"
    )
    print("-" * 58)

    for month, month_name in enumerate(MONTH_NAMES, start=1):
        d_statistic, p_value, reject_null = perform_ks_test(
            early_values=monthly_data[month]["early"],
            late_values=monthly_data[month]["late"],
        )

        decision = (
            "Reject H0"
            if reject_null
            else "Do not reject H0"
        )

        print(
            f"{month_name:<12}"
            f"{d_statistic:>10.3f}"
            f"{format_p_value(p_value):>14}"
            f"{decision:>22}"
        )

    figure = create_figure(
        monthly_data
    )

    if write2file:

        output_directory = os.path.dirname(
            output_filename
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        figure.savefig(
            output_filename,
            dpi=figure_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print()
        print(f"Wrote figure: {output_filename}")

    if show_figure:
        plt.show()
    else:
        plt.close(figure)
