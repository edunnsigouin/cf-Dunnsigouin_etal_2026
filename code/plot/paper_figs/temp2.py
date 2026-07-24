"""
Permutation-based two-sample Kolmogorov-Smirnov (KS) stability test
===================================================================

Purpose
-------
This script tests whether precipitation extremes sampled from an EARLY and a
LATE forecast lead-time window have the same distribution for each calendar
month.

The input is the NetCDF file produced by the monthly extreme-sampling script.
For example, with:

    first_input_lead = 16
    last_input_lead  = 46
    x_days           = 2
    number_of_lead_bins = 2

the usable 2-day accumulated lead times are 17-46 and the sampling script
creates two equal lead-time groups:

    early leads : 17-31
    late leads  : 32-46

Each split is already assigned independently to the calendar month containing
the majority of its own valid dates. This script therefore compares, for each
calendar month, the early- and late-lead extreme samples assigned to that same
calendar month.

Statistical test
----------------
For each calendar month, the null hypothesis is:

    H0: the early- and late-lead extremes come from the same distribution.

The two-sided KS statistic is:

    D = max_x |F_early(x) - F_late(x)|

where F_early and F_late are the empirical cumulative distribution functions
of the two samples. D is the largest vertical distance between the two CDFs.
A larger D means the two distributions are more different.

Two p-values are calculated for comparison:

1. p_perm: a permutation-based p-value estimated by randomly shuffling the
   early/late labels.
2. p_reg: the regular two-sample KS p-value returned by scipy.stats.ks_2samp
   using its standard exact/asymptotic KS null distribution as appropriate.

The permutation p-value is estimated as follows:

1. Calculate the observed KS statistic, D_observed, from the real early and
   late samples.
2. Pool all early and late values for that calendar month.
3. Randomly shuffle the pooled values and split them back into two groups with
   the original early and late sample sizes.
4. Calculate a new KS statistic, D_permuted.
5. Repeat the shuffle many times to create the null distribution of D.
6. Estimate the p-value as the proportion of permuted D values that are at
   least as large as D_observed.

A small p-value means that a difference as large as the observed one is rare
when the early/late labels are interchangeable. If p < alpha, H0 is rejected
and the lead-time distributions are considered statistically different for
that month.

Figure
------
The output figure contains one panel for each calendar month. Each panel shows:

    - the permutation distribution of KS D under H0;
    - the right-tail area corresponding to D_permuted >= D_observed;
    - a vertical line showing D_observed;
    - the permutation p-value, p_perm;
    - the regular KS p-value, p_reg;
    - whether H0 is rejected using the permutation p-value at the selected
      significance level.

The same x-axis is used for all months so the size of D can be compared
visually across the year.
"""


# =============================================================================
# Imports
# =============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

# Variable and sampling settings. These should match the script that generated
# the monthly-extremes NetCDF file.
variable = "tp24"
x_days = 2
catchment = "regine_drammen"

forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

# Original daily lead-time range.
first_input_lead = 16
last_input_lead = 46

# This analysis is a TWO-SAMPLE test, so exactly two lead-time bins are used.
# For the settings above and x_days=2, the usable accumulated leads are 17-46,
# which are split equally into 17-31 and 32-46.
number_of_lead_bins = 2

# Number of random label permutations used to estimate each monthly p-value.
number_of_permutations = 100

# Significance level. alpha=0.05 corresponds to a 95% significance level.
alpha = 0.05

# Random seed makes the permutation results reproducible.
random_seed = 42

# Histogram settings.
number_of_bins = 35
plot_probability_density = True

# Input/output directories.
path_in = config.dirs["s2s_processed"]
path_out = config.dirs["fig"]

# Optional explicit input-file override. Leave as None to construct the input
# filename automatically from the settings above.
input_filename_override = None

# Figure settings.
figure_width = 14.0
figure_height = 10.0
figure_dpi = 300

write2file = False
show_figure = True


# =============================================================================
# Functions
# =============================================================================

MONTH_LABELS = {
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


def validate_user_settings() -> None:
    """Check user settings before opening the input file."""

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

    if number_of_lead_bins != 2:
        raise ValueError(
            "This script performs a two-sample KS test and therefore "
            "requires number_of_lead_bins = 2."
        )

    if number_of_permutations < 1:
        raise ValueError(
            "number_of_permutations must be at least 1."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if number_of_bins < 1:
        raise ValueError("number_of_bins must be at least 1.")


def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """Flatten an array and retain only finite values."""

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """
    Split usable accumulated ending leads into approximately equal bins.

    Extra lead times, if present, are assigned to the later bins. For the
    standard 2-day example, 17-46 contains 30 lead times and therefore splits
    exactly into 17-31 and 32-46.
    """

    number_of_leads = last_lead - first_lead + 1
    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size
        + int(bin_index >= number_of_bins - remainder)
        for bin_index in range(number_of_bins)
    ]

    lead_bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        lead_bins.append((current_start, current_end))
        current_start = current_end + 1

    return lead_bins


def get_lead_ranges() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return the full, early, and late usable accumulated lead ranges."""

    first_usable_lead = first_input_lead + x_days - 1

    full_range = (
        first_usable_lead,
        last_input_lead,
    )

    split_ranges = split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )

    early_range = split_ranges[0]
    late_range = split_ranges[1]

    return full_range, early_range, late_range


def lead_range_variable_name(
    lead_start: int,
    lead_end: int,
) -> str:
    """Return the NetCDF variable name for one lead-time range."""

    return f"max_value_lead{lead_start}_{lead_end}"


def lead_split_filename_label(
    full_range: tuple[int, int],
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> str:
    """Return the lead-time label used by the sampling-script filename."""

    full_start, full_end = full_range
    early_start, early_end = early_range
    late_start, late_end = late_range

    return (
        f"lead{full_start}-{full_end}_"
        f"split{number_of_lead_bins}_"
        f"{early_start}-{early_end}_"
        f"{late_start}-{late_end}"
    )


def build_input_filename(
    full_range: tuple[int, int],
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> str:
    """Build the NetCDF filename written by the sampling script."""

    if input_filename_override is not None:
        return input_filename_override

    lead_label = lead_split_filename_label(
        full_range=full_range,
        early_range=early_range,
        late_range=late_range,
    )

    return os.path.join(
        path_in,
        (
            f"distribution_monthly_extremes_"
            f"{variable}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_label}_"
            f"forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )


def build_output_filename(
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> str:
    """Build a descriptive PNG filename for the 12-panel figure."""

    early_start, early_end = early_range
    late_start, late_end = late_range

    confidence_percent = 100.0 * (1.0 - alpha)

    return os.path.join(
        path_out,
        (
            f"permutation_ks_lead_stability_"
            f"{variable}_{x_days}dayacc_"
            f"{catchment}_"
            f"early{early_start}-{early_end}_"
            f"late{late_start}-{late_end}_"
            f"{confidence_percent:g}pct.png"
        ),
    )


def load_monthly_split_samples(
    filename: str,
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Load early- and late-lead extreme samples for all 12 calendar months.

    Returns
    -------
    dictionary
        Keys are calendar months 1-12. Each value is a tuple containing
        (early_values, late_values).
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Input file does not exist:\n{filename}"
        )

    early_variable = lead_range_variable_name(*early_range)
    late_variable = lead_range_variable_name(*late_range)

    monthly_samples = {}

    with xr.open_dataset(filename) as ds:

        for variable_name in (early_variable, late_variable):
            if variable_name not in ds:
                raise KeyError(
                    f"Variable '{variable_name}' was not found in the input "
                    f"file. Available variables: {list(ds.data_vars)}"
                )

        if "month_of_year" not in ds.coords:
            raise KeyError(
                "The input dataset has no 'month_of_year' coordinate."
            )

        for month in range(1, 13):
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

            if early_values.size < 2:
                raise ValueError(
                    f"Only {early_values.size} early values were found for "
                    f"{MONTH_LABELS[month]}. At least 2 are required."
                )

            if late_values.size < 2:
                raise ValueError(
                    f"Only {late_values.size} late values were found for "
                    f"{MONTH_LABELS[month]}. At least 2 are required."
                )

            monthly_samples[month] = (
                early_values,
                late_values,
            )

    return monthly_samples


def calculate_regular_ks_test(
    sample_1: np.ndarray,
    sample_2: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the regular two-sided two-sample KS test.

    Returns
    -------
    observed_d : float
        The KS D statistic.
    regular_p_value : float
        The standard KS p-value returned by scipy.stats.ks_2samp.
    """

    result = ks_2samp(
        sample_1,
        sample_2,
        alternative="two-sided",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def calculate_ks_statistic(
    sample_1: np.ndarray,
    sample_2: np.ndarray,
) -> float:
    """Calculate only the two-sided two-sample KS D statistic."""

    observed_d, _ = calculate_regular_ks_test(
        sample_1,
        sample_2,
    )

    return observed_d


def permutation_ks_test(
    early_values: np.ndarray,
    late_values: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    """
    Perform one permutation-based two-sided KS test.

    The early and late values are pooled. For each permutation, the pooled
    values are randomly reordered and split back into groups with the original
    sample sizes. A KS D statistic is then calculated for that random split.

    The p-value uses the common +1 correction:

        p = (number of D_permuted >= D_observed + 1)
            / (number_of_permutations + 1)

    This prevents an estimated p-value of exactly zero when using a finite
    number of random permutations.
    """

    n_early = early_values.size
    n_late = late_values.size

    observed_d, regular_p_value = calculate_regular_ks_test(
        early_values,
        late_values,
    )

    pooled_values = np.concatenate(
        [early_values, late_values]
    )

    permuted_d = np.empty(
        number_of_permutations,
        dtype="float64",
    )

    for permutation_index in range(number_of_permutations):
        shuffled_values = rng.permutation(pooled_values)

        permuted_early = shuffled_values[:n_early]
        permuted_late = shuffled_values[n_early:]

        permuted_d[permutation_index] = calculate_ks_statistic(
            permuted_early,
            permuted_late,
        )

    number_at_least_observed = int(
        np.count_nonzero(permuted_d >= observed_d)
    )

    p_value = (
        number_at_least_observed + 1
    ) / (
        number_of_permutations + 1
    )

    reject_null = p_value < alpha

    return {
        "observed_d": observed_d,
        "permuted_d": permuted_d,
        "p_value": float(p_value),
        "regular_p_value": float(regular_p_value),
        "reject_null": reject_null,
        "n_early": n_early,
        "n_late": n_late,
    }


def run_all_monthly_tests(
    monthly_samples: dict[int, tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> dict[int, dict[str, object]]:
    """Run permutation and regular KS tests independently for all 12 months."""

    results = {}

    for month in range(1, 13):
        early_values, late_values = monthly_samples[month]

        results[month] = permutation_ks_test(
            early_values=early_values,
            late_values=late_values,
            rng=rng,
        )

    return results


def get_common_x_limit(
    results: dict[int, dict[str, object]],
) -> float:
    """Return one common upper x-axis limit for all 12 panels."""

    maximum_d = 0.0

    for result in results.values():
        maximum_d = max(
            maximum_d,
            float(result["observed_d"]),
            float(np.max(result["permuted_d"])),
        )

    # KS D cannot exceed 1. Add a small margin while respecting that bound.
    return min(1.0, maximum_d * 1.08)


def create_histogram_bin_edges(
    common_x_max: float,
) -> np.ndarray:
    """Create identical histogram bins for every calendar-month panel."""

    if common_x_max <= 0.0:
        common_x_max = 1.0

    return np.linspace(
        0.0,
        common_x_max,
        number_of_bins + 1,
    )


def format_p_value(p_value: float) -> str:
    """Format p-values compactly for figure annotations and terminal output."""

    minimum_resolvable = 1.0 / (number_of_permutations + 1)

    if np.isclose(p_value, minimum_resolvable):
        return f"< {2.0 * minimum_resolvable:.4f}"

    if p_value < 0.001:
        return f"{p_value:.2e}"

    return f"{p_value:.3f}"


def format_regular_p_value(p_value: float) -> str:
    """Format the regular KS p-value for the figure and terminal output."""

    if p_value < 0.001:
        return f"{p_value:.2e}"

    return f"{p_value:.3f}"


def plot_month_panel(
    ax: plt.Axes,
    month: int,
    result: dict[str, object],
    bin_edges: np.ndarray,
    common_x_max: float,
) -> None:
    """Plot the permutation null distribution and observed D for one month."""

    permuted_d = np.asarray(result["permuted_d"])
    observed_d = float(result["observed_d"])
    p_value = float(result["p_value"])
    regular_p_value = float(result["regular_p_value"])
    reject_null = bool(result["reject_null"])

    # Draw the complete permutation distribution first.
    counts, _, patches = ax.hist(
        permuted_d,
        bins=bin_edges,
        density=plot_probability_density,
        histtype="bar",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.35,
    )

    # Shade the part of the permutation distribution corresponding to
    # D_permuted >= D_observed. This is the tail used to calculate the p-value.
    for patch, left_edge, right_edge in zip(
        patches,
        bin_edges[:-1],
        bin_edges[1:],
    ):
        if right_edge > observed_d:
            patch.set_alpha(0.8)

    ax.axvline(
        observed_d,
        color="black",
        linewidth=1.8,
        linestyle="--",
        zorder=5,
    )

    decision_text = (
        "Reject H₀"
        if reject_null
        else "Do not reject H₀"
    )

    annotation = (
        f"D = {observed_d:.3f}\n"
        f"p_perm = {format_p_value(p_value)}\n"
        f"p_reg = {format_regular_p_value(regular_p_value)}\n"
        f"{decision_text}"
    )

    ax.text(
        0.97,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax.set_title(
        MONTH_LABELS[month],
        fontsize=11,
        fontweight="normal",
    )

    ax.set_xlim(0.0, common_x_max)

    if counts.size > 0 and np.nanmax(counts) > 0:
        ax.set_ylim(0.0, float(np.nanmax(counts)) * 1.12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=9)


def create_figure(
    results: dict[int, dict[str, object]],
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> plt.Figure:
    """Create the 3 x 4 calendar-month permutation-KS diagnostic figure."""

    common_x_max = get_common_x_limit(results)
    bin_edges = create_histogram_bin_edges(common_x_max)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    for month in range(1, 13):
        row = (month - 1) // 4
        column = (month - 1) % 4

        plot_month_panel(
            ax=axes[row, column],
            month=month,
            result=results[month],
            bin_edges=bin_edges,
            common_x_max=common_x_max,
        )

    # Shared axis labels keep individual panels uncluttered.
    fig.supxlabel(
        "Two-sample KS statistic, D",
        fontsize=11,
        y=0.045,
    )

    y_label = (
        "Probability density"
        if plot_probability_density
        else "Number of permutations"
    )

    fig.supylabel(
        y_label,
        fontsize=11,
        x=0.025,
    )

    early_start, early_end = early_range
    late_start, late_end = late_range
    confidence_percent = 100.0 * (1.0 - alpha)

    fig.suptitle(
        (
            f"Permutation-based and regular two-sided KS lead-time stability test\n"
            f"Early leads {early_start}-{early_end} vs late leads "
            f"{late_start}-{late_end} | "
            f"{x_days}-day precipitation extremes | "
            f"{confidence_percent:g}% significance level"
        ),
        fontsize=13,
        fontweight="normal",
        y=0.985,
    )

    legend_handles = [
        Patch(
            facecolor="0.7",
            edgecolor="black",
            alpha=0.35,
            label="Permutation null distribution",
        ),
        Patch(
            facecolor="0.7",
            edgecolor="black",
            alpha=0.8,
            label="Permuted D ≥ observed D",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Observed D",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.86,
        wspace=0.23,
        hspace=0.34,
    )

    return fig


def print_results(
    results: dict[int, dict[str, object]],
    early_range: tuple[int, int],
    late_range: tuple[int, int],
) -> None:
    """Print a concise table comparing monthly permutation and regular KS results."""

    early_start, early_end = early_range
    late_start, late_end = late_range

    print()
    print("Permutation-based and regular two-sided KS tests")
    print("-----------------------------------------------")
    print(f"Early leads: {early_start}-{early_end}")
    print(f"Late leads:  {late_start}-{late_end}")
    print(f"Permutations: {number_of_permutations}")
    print(f"alpha: {alpha:.3f}")
    print()

    header = (
        f"{'Month':<10s} "
        f"{'n early':>8s} "
        f"{'n late':>8s} "
        f"{'D':>8s} "
        f"{'p_perm':>10s} "
        f"{'p_reg':>10s} "
        f"{'Decision':>18s}"
    )

    print(header)
    print("-" * len(header))

    for month in range(1, 13):
        result = results[month]

        decision = (
            "Reject H0"
            if result["reject_null"]
            else "Do not reject H0"
        )

        print(
            f"{MONTH_LABELS[month]:<10s} "
            f"{result['n_early']:>8d} "
            f"{result['n_late']:>8d} "
            f"{result['observed_d']:>8.3f} "
            f"{format_p_value(result['p_value']):>10s} "
            f"{format_regular_p_value(result['regular_p_value']):>10s} "
            f"{decision:>18s}"
        )


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    (
        full_range,
        early_range,
        late_range,
    ) = get_lead_ranges()

    input_filename = build_input_filename(
        full_range=full_range,
        early_range=early_range,
        late_range=late_range,
    )

    output_filename = build_output_filename(
        early_range=early_range,
        late_range=late_range,
    )

    print("Input file")
    print("----------")
    print(input_filename)

    print()
    print("Lead-time groups")
    print("----------------")
    print(f"Full usable range: {full_range[0]}-{full_range[1]}")
    print(f"Early range:       {early_range[0]}-{early_range[1]}")
    print(f"Late range:        {late_range[0]}-{late_range[1]}")

    monthly_samples = load_monthly_split_samples(
        filename=input_filename,
        early_range=early_range,
        late_range=late_range,
    )

    rng = np.random.default_rng(random_seed)

    results = run_all_monthly_tests(
        monthly_samples=monthly_samples,
        rng=rng,
    )

    print_results(
        results=results,
        early_range=early_range,
        late_range=late_range,
    )

    figure = create_figure(
        results=results,
        early_range=early_range,
        late_range=late_range,
    )

    if write2file:
        os.makedirs(path_out, exist_ok=True)

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
