"""
Plot monthly distributions of X-day accumulated precipitation extremes.

The figure compares monthly extreme precipitation distributions from:

1. S2S forecast and hindcast model data.
2. ERA5.
3. SeNorge or regridded SeNorge.

Each calendar month is plotted in a separate panel. The three distributions
are shown as unfilled histogram outlines so that they can be overlaid and
compared directly.

Inputs:
- S2S monthly extreme distribution file.
- ERA5 monthly extreme distribution file.
- SeNorge or SeNorge-regrid monthly extreme distribution file.

Output:
- One publication-quality 3 × 4 panel PNG figure.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User-defined input parameters
# =============================================================================

# Catchment and accumulation period
catchment = "regine_drammen"
x_days = 2

# Select either native-grid or regridded SeNorge data
reference_dataset = "senorge"  # "senorge" or "senorge_regrid"

# Date ranges used in the input filenames
forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

# ERA5 grid used for the reference dataset
era5_grid = "0.5x0.5"

# Number of histogram bins in every monthly panel
number_of_bins = 30

# Normalize each histogram so that distributions with different sample sizes
# can be compared directly
plot_probability_density = True

# Figure output
write2file = False
filename_out = os.path.join(
    config.dirs["fig"],
    (
        f"monthly_histograms_{x_days}dayacc_"
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

FIG_WIDTH_IN = 10.0
FIG_HEIGHT_IN = 7.5

TITLE_FONTSIZE = 10
SUPTITLE_FONTSIZE = 12
AXIS_LABELSIZE = 10
TICK_LABELSIZE = 9
LEGEND_FONTSIZE = 9

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

    if number_of_bins < 1:
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
    Check that each dataset contains at least one finite value per month.
    """

    for month, values in values_by_month.items():
        if values.size == 0:
            raise ValueError(
                f"No finite {dataset_name} values were found for "
                f"{MONTH_LABELS[month - 1]}."
            )


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

    Using common bin edges is essential because it ensures that the three
    overlaid histograms describe precipitation values over identical intervals.
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


def make_legend_handles(
    reference_label: str,
) -> list[Line2D]:
    """
    Create line-style legend handles matching the histogram outlines.
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
            label=reference_label,
        ),
        Line2D(
            [0],
            [0],
            color=ERA5_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="ERA5",
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
        fontweight="bold",
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

    # Only the bottom row receives an x-axis label.
    if row_index == 2:
        ax.set_xlabel(
            "Precipitation [mm]",
            fontsize=AXIS_LABELSIZE,
        )

    # Only the first column receives a y-axis label.
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
    Plot monthly model, ERA5, and SeNorge precipitation distributions.

    Each month uses bin edges calculated from the combined values of all three
    datasets for that month. The histograms are outlines without shading.
    """

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        squeeze=False,
    )

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

        # Model histogram
        ax.hist(
            model_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=MODEL_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="Model",
            zorder=3,
        )

        # SeNorge histogram
        ax.hist(
            reference_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=SENORGE_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label=reference_label,
            zorder=2,
        )

        # ERA5 histogram
        ax.hist(
            era5_values,
            bins=bin_edges,
            density=plot_probability_density,
            histtype="step",
            color=ERA5_COLOR,
            linewidth=HISTOGRAM_LINEWIDTH,
            label="ERA5",
            zorder=1,
        )

        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(bottom=0)

        apply_axis_formatting(
            ax=ax,
            month_label=MONTH_LABELS[month - 1],
            row_index=row_index,
            column_index=column_index,
        )

    catchment_label = get_catchment_label(catchment)

    fig.suptitle(
        (
            f"{catchment_label}: monthly distributions of "
            f"{x_days}-day accumulated precipitation maxima"
        ),
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="bold",
        y=0.985,
    )

    fig.legend(
        handles=make_legend_handles(reference_label),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.5,
        columnspacing=2.0,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.88,
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
        print("Number of values in January")
        print("---------------------------")
        print(f"Model:     {model_values_by_month[1].size}")
        print(f"ERA5:      {era5_values_by_month[1].size}")
        print(
            f"{reference_label}: "
            f"{reference_values_by_month[1].size}"
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

