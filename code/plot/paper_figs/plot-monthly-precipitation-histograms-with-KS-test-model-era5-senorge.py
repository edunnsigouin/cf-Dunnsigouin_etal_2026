"""
Plot monthly distributions of X-day accumulated precipitation extremes.

The figure compares monthly extreme precipitation distributions from:

1. S2S forecast and hindcast model data.
2. ERA5.
3. SeNorge or regridded SeNorge.

Each calendar month is plotted in a separate panel. The three distributions
are shown as unfilled histogram outlines so that they can be overlaid and
compared directly.

For every month, a two-sample Kolmogorov-Smirnov test compares:

- ERA5 with the model distribution.
- SeNorge or SeNorge-regrid with the model distribution.

The p-values are displayed in the legend of each panel using the color of the
corresponding reference dataset. An asterisk marks a failed test at the
user-defined confidence level.

Inputs:
- S2S monthly extreme distribution file.
- ERA5 monthly extreme distribution file.
- SeNorge or SeNorge-regrid monthly extreme distribution file.

Output:
- One publication-quality 3 x 4 panel PNG figure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User-defined input parameters
# =============================================================================

# Catchment and accumulation period
catchment = "regine_glomma"
x_days = 2

# Select either native-grid or regridded SeNorge data
reference_dataset = "senorge"  # "senorge" or "senorge_regrid"

# Date ranges used in the input filenames
forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

# ERA5 grid used for the reference dataset
era5_grid = "0.5x0.5"

# Number of histogram bins in every monthly panel
number_of_bins = 20

# Normalize each histogram so that distributions with different sample sizes
# can be compared directly
plot_probability_density = True

# Kolmogorov-Smirnov test settings
ks_alternative = "two-sided"
ks_method = "auto"

# Confidence level used to decide whether a KS test fails.
# For example, 95 means that p-values below 0.05 are marked with an asterisk.
significance_level_percent = 95.0

# Figure output
write2file = True
filename_out = os.path.join(
    config.dirs["fig"],
    (
        f"monthly_histograms_ks_test_{x_days}dayacc_"
        f"{catchment}_model_era5_{reference_dataset}.png"
    ),
)


# =============================================================================
# Dataset-specific settings
# =============================================================================

# Model data
MODEL_VARIABLE = "tp"
MODEL_EXTREME_VARIABLE = "max_value"
MODEL_MONTH_COORDINATE = "month_of_year"

# ERA5 data
ERA5_VARIABLE = "tp"

# SeNorge variable names
REFERENCE_VARIABLES = {
    "senorge": "tp",
    "senorge_regrid": "rr",
}

# Labels used in the plot
REFERENCE_LABELS = {
    "senorge": "SeNorge",
    "senorge_regrid": "SeNorge regrid",
}


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 13.0
FIG_HEIGHT_IN = 8.0

TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12
AXIS_LABELSIZE = 10
TICK_LABELSIZE = 9
LEGEND_FONTSIZE = 7.5

HISTOGRAM_LINEWIDTH = 1.5

MODEL_COLOR = "black"
SENORGE_COLOR = "tab:red"
ERA5_COLOR = "tab:blue"

MONTHS = np.arange(1, 13)

MONTH_LABELS = [
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
# Configuration helpers
# =============================================================================

def get_reference_variable(reference_dataset: str) -> str:
    """
    Return the precipitation variable used by the selected SeNorge dataset.
    """

    if reference_dataset not in REFERENCE_VARIABLES:
        valid_options = ", ".join(REFERENCE_VARIABLES)

        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return REFERENCE_VARIABLES[reference_dataset]


def get_reference_label(reference_dataset: str) -> str:
    """
    Return a publication-friendly name for the selected SeNorge dataset.
    """

    if reference_dataset not in REFERENCE_LABELS:
        valid_options = ", ".join(REFERENCE_LABELS)

        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid_options}."
        )

    return REFERENCE_LABELS[reference_dataset]


def get_catchment_label(catchment: str) -> str:
    """
    Return a publication-friendly catchment name.
    """

    catchment_labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }

    return catchment_labels.get(
        catchment,
        catchment.replace("_", " ").title(),
    )


def validate_user_settings() -> None:
    """
    Check the user-defined settings before reading any data.
    """

    if not isinstance(number_of_bins, int) or number_of_bins < 1:
        raise ValueError(
            "number_of_bins must be an integer greater than or equal to 1."
        )

    if len(reference_years) != 2:
        raise ValueError(
            "reference_years must contain a start year and an end year."
        )

    if len(forecast_date_range) != 2:
        raise ValueError(
            "forecast_date_range must contain a start date and an end date."
        )

    valid_alternatives = {"two-sided", "less", "greater"}

    if ks_alternative not in valid_alternatives:
        raise ValueError(
            f"ks_alternative must be one of {sorted(valid_alternatives)}."
        )

    valid_methods = {"auto", "exact", "asymp"}

    if ks_method not in valid_methods:
        raise ValueError(
            f"ks_method must be one of {sorted(valid_methods)}."
        )

    if not 0.0 < significance_level_percent < 100.0:
        raise ValueError(
            "significance_level_percent must be greater than 0 and less than 100."
        )


# =============================================================================
# Filename helpers
# =============================================================================

def make_model_filename() -> str:
    """
    Create the S2S forecast and hindcast input filename.
    """

    return (
        f"{config.dirs['s2s_processed']}"
        f"distribution_monthly_extremes_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )


def make_era5_filename() -> str:
    """
    Create the ERA5 input filename.
    """

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def make_reference_filename(
    reference_dataset: str,
    reference_variable: str,
) -> str:
    """
    Create the SeNorge or SeNorge-regrid input filename.
    """

    return (
        f"{config.dirs[f'{reference_dataset}_processed']}"
        f"distribution_monthly_extremes_{reference_variable}_{x_days}dayacc_"
        f"{catchment}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


# =============================================================================
# Data loading and validation
# =============================================================================

def load_datasets(
    filename_model: str,
    filename_era5: str,
    filename_reference: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Open the model, ERA5, and SeNorge datasets.
    """

    model_ds = xr.open_dataset(filename_model)
    era5_ds = xr.open_dataset(filename_era5)
    reference_ds = xr.open_dataset(filename_reference)

    return model_ds, era5_ds, reference_ds


def check_variable_exists(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> None:
    """
    Raise a clear error when a required variable is missing.
    """

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available data variables are: {list(ds.data_vars)}"
        )


def check_coordinate_exists(
    data: xr.DataArray,
    coordinate: str,
    dataset_name: str,
) -> None:
    """
    Raise a clear error when a required coordinate or dimension is missing.
    """

    available_names = set(data.coords) | set(data.dims)

    if coordinate not in available_names:
        raise KeyError(
            f"Coordinate or dimension '{coordinate}' was not found in "
            f"{dataset_name}. Available dimensions are {data.dims}, and "
            f"available coordinates are {list(data.coords)}."
        )


# =============================================================================
# Data extraction
# =============================================================================

def remove_missing_values(values: np.ndarray) -> np.ndarray:
    """
    Flatten an array and remove NaN and infinite values.
    """

    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def get_model_values_by_month(
    model_ds: xr.Dataset,
    variable: str = MODEL_EXTREME_VARIABLE,
) -> dict[int, np.ndarray]:
    """
    Extract one array of model extremes for each calendar month.

    The expected model structure is approximately:

        max_value(month_of_year, index)

    Additional dimensions are allowed because all selected monthly values are
    flattened before plotting.
    """

    check_variable_exists(
        model_ds,
        variable,
        dataset_name="model dataset",
    )

    model_data = model_ds[variable]

    check_coordinate_exists(
        model_data,
        MODEL_MONTH_COORDINATE,
        dataset_name="model dataset",
    )

    values_by_month = {}

    for month in MONTHS:
        monthly_data = model_data.sel(
            {MODEL_MONTH_COORDINATE: month}
        )

        values_by_month[month] = remove_missing_values(
            monthly_data.values
        )

    return values_by_month


def get_reference_values_by_month(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> dict[int, np.ndarray]:
    """
    Extract one array of annual monthly extremes for each calendar month.

    The expected reference-data structure is approximately:

        variable(year, month)

    The function selects each month and then flattens all remaining dimensions.
    """

    check_variable_exists(
        ds,
        variable,
        dataset_name=dataset_name,
    )

    data = ds[variable]

    check_coordinate_exists(
        data,
        "month",
        dataset_name=dataset_name,
    )

    values_by_month = {}

    for month in MONTHS:
        monthly_data = data.sel(month=month)

        values_by_month[month] = remove_missing_values(
            monthly_data.values
        )

    return values_by_month


def check_monthly_samples(
    values_by_month: dict[int, np.ndarray],
    dataset_name: str,
) -> None:
    """
    Check that each dataset contains at least two finite values per month.

    At least two values are required for a meaningful distribution comparison.
    """

    for month, values in values_by_month.items():
        if values.size < 2:
            raise ValueError(
                f"Fewer than two finite {dataset_name} values were found for "
                f"{MONTH_LABELS[month - 1]}."
            )


# =============================================================================
# Statistical-test helpers
# =============================================================================

def perform_ks_test(
    model_values: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[float, float]:
    """
    Perform a two-sample Kolmogorov-Smirnov test.

    The null hypothesis is that the model and reference samples are drawn
    from the same continuous distribution.

    Returns
    -------
    statistic : float
        Maximum absolute difference between the two empirical cumulative
        distribution functions.
    p_value : float
        Probability of obtaining a KS statistic at least this large if the
        null hypothesis is true.
    """

    result = ks_2samp(
        model_values,
        reference_values,
        alternative=ks_alternative,
        method=ks_method,
    )

    return float(result.statistic), float(result.pvalue)


def get_significance_threshold() -> float:
    """
    Convert the user-defined confidence level to a p-value threshold.

    For example:
        95% confidence level -> alpha = 0.05
        99% confidence level -> alpha = 0.01
    """

    return 1.0 - significance_level_percent / 100.0


def test_failed(p_value: float) -> bool:
    """
    Return True when the KS test fails at the selected confidence level.

    A failed test means that the null hypothesis of equal distributions is
    rejected because the p-value is smaller than the significance threshold.
    """

    return p_value < get_significance_threshold()


def format_p_value(p_value: float) -> str:
    """
    Format a p-value compactly and mark failed tests with an asterisk.
    """

    if p_value < 0.001:
        formatted = f"{p_value:.1e}"
    else:
        formatted = f"{p_value:.3f}"

    if test_failed(p_value):
        formatted += "*"

    return formatted


# =============================================================================
# Histogram helpers
# =============================================================================

def calculate_common_bin_edges(
    model_values: np.ndarray,
    era5_values: np.ndarray,
    reference_values: np.ndarray,
    number_of_bins: int,
) -> np.ndarray:
    """
    Calculate common histogram bins for the three datasets in one panel.

    Common bin edges ensure that the overlaid histograms describe identical
    precipitation intervals.
    """

    combined_values = np.concatenate(
        [
            model_values,
            era5_values,
            reference_values,
        ]
    )

    minimum = float(np.nanmin(combined_values))
    maximum = float(np.nanmax(combined_values))

    if np.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 0.5)
        minimum -= padding
        maximum += padding

    return np.linspace(
        minimum,
        maximum,
        number_of_bins + 1,
    )


def get_histogram_y_label() -> str:
    """
    Return the appropriate y-axis label.
    """

    if plot_probability_density:
        return "Probability density"

    return "Number of events"


def make_panel_legend_handles(
    reference_label: str,
    era5_p_value: float,
    reference_p_value: float,
) -> list[Line2D]:
    """
    Create histogram and p-value legend entries for one monthly panel.

    The p-value for each reference dataset is shown in the color used for its
    histogram.
    """

    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model",
        ),
        Line2D(
            [0],
            [0],
            color=SENORGE_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label=(
                f"{reference_label}: "
                f"$p_{{KS}}$={format_p_value(reference_p_value)}"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=ERA5_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label=f"ERA5: $p_{{KS}}$={format_p_value(era5_p_value)}",
        ),
    ]


# =============================================================================
# Plot formatting
# =============================================================================

def apply_axis_formatting(
    ax: plt.Axes,
    month_label: str,
    row_index: int,
    column_index: int,
) -> None:
    """
    Apply consistent publication-style formatting to one panel.
    """

    ax.set_title(
        month_label,
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=5,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABELSIZE,
        direction="out",
        length=3.5,
        width=0.8,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.0,
        width=0.6,
    )

    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if row_index == 2:
        ax.set_xlabel(
            "Precipitation [mm]",
            fontsize=AXIS_LABELSIZE,
        )

    if column_index == 0:
        ax.set_ylabel(
            get_histogram_y_label(),
            fontsize=AXIS_LABELSIZE,
        )


# =============================================================================
# Main plotting function
# =============================================================================

def plot_monthly_histograms(
    model_values_by_month: dict[int, np.ndarray],
    era5_values_by_month: dict[int, np.ndarray],
    reference_values_by_month: dict[int, np.ndarray],
    reference_label: str,
    filename_out: str,
    write2file: bool,
) -> None:
    """
    Plot monthly precipitation distributions and KS-test p-values.

    For each month:

    - All three histograms use common bin edges.
    - Histograms are outlines without shading.
    - ERA5 is compared with the model using a two-sample KS test.
    - SeNorge or SeNorge-regrid is compared with the model using the same test.
    - Both p-values are shown in the panel legend.
    """

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        squeeze=False,
    )

    # -----------------------------------------------------------------
    # Calculate common x- and y-axis limits across all 12 months.
    # -----------------------------------------------------------------

    global_x_min = np.inf
    global_x_max = -np.inf
    global_y_max = 0.0

    for month in MONTHS:

        model_values = model_values_by_month[month]
        era5_values = era5_values_by_month[month]
        reference_values = reference_values_by_month[month]

        bin_edges = calculate_common_bin_edges(
            model_values=model_values,
            era5_values=era5_values,
            reference_values=reference_values,
            number_of_bins=number_of_bins,
        )

        global_x_min = min(global_x_min, bin_edges[0])
        global_x_max = max(global_x_max, bin_edges[-1])

        for values in (model_values, era5_values, reference_values):

            counts, _ = np.histogram(
                values,
                bins=bin_edges,
                density=plot_probability_density,
            )

            global_y_max = max(global_y_max, np.nanmax(counts))

    global_y_max *= 1.05

    for month_index, month in enumerate(MONTHS):
        row_index = month_index // 4
        column_index = month_index % 4

        ax = axes[row_index, column_index]

        model_values = model_values_by_month[month]
        era5_values = era5_values_by_month[month]
        reference_values = reference_values_by_month[month]

        bin_edges = calculate_common_bin_edges(
            model_values=model_values,
            era5_values=era5_values,
            reference_values=reference_values,
            number_of_bins=number_of_bins,
        )

        _, era5_p_value = perform_ks_test(
            model_values=model_values,
            reference_values=era5_values,
        )

        _, reference_p_value = perform_ks_test(
            model_values=model_values,
            reference_values=reference_values,
        )

        ax.hist(
            model_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=3,
        )

        ax.hist(
            reference_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=SENORGE_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=2,
        )

        ax.hist(
            era5_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=ERA5_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            zorder=1,
        )

        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, global_y_max)

        apply_axis_formatting(
            ax=ax,
            month_label=MONTH_LABELS[month - 1],
            row_index=row_index,
            column_index=column_index,
        )

        ax.legend(
            handles=make_panel_legend_handles(
                reference_label=reference_label,
                era5_p_value=era5_p_value,
                reference_p_value=reference_p_value,
            ),
            loc="upper right",
            frameon=False,
            facecolor="white",
            edgecolor="0.8",
            framealpha=0.9,
            fontsize=LEGEND_FONTSIZE,
            handlelength=1.8,
            handletextpad=0.5,
            borderaxespad=0.4,
            labelspacing=0.25,
        )

    catchment_label = get_catchment_label(catchment)

    fig.suptitle(
        (
            f"{catchment_label}: monthly distributions of "
            f"{x_days}-day accumulated precipitation maxima"
        ),
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="normal",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.93,
        wspace=0.28,
        hspace=0.38,
    )

    if write2file:
        output_directory = os.path.dirname(filename_out)

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        fig.savefig(
            filename_out,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Wrote figure: {filename_out}")

    plt.show()
    plt.close(fig)


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    reference_variable = get_reference_variable(
        reference_dataset
    )

    reference_label = get_reference_label(
        reference_dataset
    )

    filename_model = make_model_filename()
    filename_era5 = make_era5_filename()

    filename_reference = make_reference_filename(
        reference_dataset=reference_dataset,
        reference_variable=reference_variable,
    )

    print("Reading input datasets")
    print("----------------------")
    print(f"Model:     {filename_model}")
    print(f"ERA5:      {filename_era5}")
    print(f"{reference_label}: {filename_reference}")

    model_ds, era5_ds, reference_ds = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_reference=filename_reference,
    )

    try:
        model_values_by_month = get_model_values_by_month(
            model_ds=model_ds,
            variable=MODEL_EXTREME_VARIABLE,
        )

        era5_values_by_month = get_reference_values_by_month(
            ds=era5_ds,
            variable=ERA5_VARIABLE,
            dataset_name="ERA5 dataset",
        )

        reference_values_by_month = get_reference_values_by_month(
            ds=reference_ds,
            variable=reference_variable,
            dataset_name=f"{reference_label} dataset",
        )

        check_monthly_samples(
            values_by_month=model_values_by_month,
            dataset_name="model",
        )

        check_monthly_samples(
            values_by_month=era5_values_by_month,
            dataset_name="ERA5",
        )

        check_monthly_samples(
            values_by_month=reference_values_by_month,
            dataset_name=reference_label,
        )

        print()
        print("Monthly Kolmogorov-Smirnov tests")
        print("--------------------------------")
        print(
            f"Failure threshold: p < {get_significance_threshold():.4f} "
            f"({significance_level_percent:g}% confidence level)"
        )
        print("An asterisk marks a failed test.")

        for month in MONTHS:
            era5_statistic, era5_p_value = perform_ks_test(
                model_values=model_values_by_month[month],
                reference_values=era5_values_by_month[month],
            )

            reference_statistic, reference_p_value = perform_ks_test(
                model_values=model_values_by_month[month],
                reference_values=reference_values_by_month[month],
            )

            print(
                f"{MONTH_LABELS[month - 1]:>9s} | "
                f"ERA5: D={era5_statistic:.3f}, "
                f"p={era5_p_value:.4g} | "
                f"{reference_label}: D={reference_statistic:.3f}, "
                f"p={reference_p_value:.4g}"
            )

        plot_monthly_histograms(
            model_values_by_month=model_values_by_month,
            era5_values_by_month=era5_values_by_month,
            reference_values_by_month=reference_values_by_month,
            reference_label=reference_label,
            filename_out=filename_out,
            write2file=write2file,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        reference_ds.close()
