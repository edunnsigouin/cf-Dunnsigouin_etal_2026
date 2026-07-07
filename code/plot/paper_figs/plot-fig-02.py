"""
Plot monthly distributions of catchment precipitation extremes.

The figure compares:
1. S2S forecast/hindcast monthly extreme distributions.
2. ERA5 monthly records before Storm Hans.
3. SeNorge or regridded SeNorge monthly records before Storm Hans.
4. Storm Hans 2023 in ERA5 and SeNorge / SeNorge-regrid.
5. The largest May S2S event, interpreted as a counterfactual spring Hans.

Inputs:
- S2S monthly extreme distribution file.
- ERA5 monthly extreme distribution file.
- SeNorge or SeNorge-regrid monthly extreme distribution file.

Output:
- One publication-quality PNG figure.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"
x_days = 2

reference_dataset = "senorge"  # "senorge" or "senorge_regrid"

forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

era5_grid = "0.5x0.5"

write2file = True

filename_out = config.dirs["fig"] + "fig-02.png"


# =============================================================================
# Dataset-specific settings
# =============================================================================

MODEL_VARIABLE = "tp"
MODEL_EXTREME_VARIABLE = "max_value"

ERA5_VARIABLE = "tp"

REFERENCE_VARIABLES = {
    "senorge": "tp",
    "senorge_regrid": "rr",
}

REFERENCE_LABELS = {
    "senorge": "SeNorge",
    "senorge_regrid": "SeNorge regrid",
}


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

REFERENCE_COLOR = "tab:blue"
ERA5_COLOR = "tab:red"
COUNTERFACTUAL_COLOR = "tab:green"

MONTHS = np.arange(1, 13)
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Configuration helpers
# =============================================================================

def get_reference_variable(reference_dataset: str) -> str:
    """Return the variable name used in the selected reference dataset."""

    if reference_dataset not in REFERENCE_VARIABLES:
        valid = ", ".join(REFERENCE_VARIABLES)
        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid}."
        )

    return REFERENCE_VARIABLES[reference_dataset]


def get_reference_label(reference_dataset: str) -> str:
    """Return a plot-friendly label for the selected reference dataset."""

    if reference_dataset not in REFERENCE_LABELS:
        valid = ", ".join(REFERENCE_LABELS)
        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid}."
        )

    return REFERENCE_LABELS[reference_dataset]


def get_catchment_label(catchment: str) -> str:
    """Return a plot-friendly catchment name."""

    labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }

    return labels.get(catchment, catchment)


# =============================================================================
# Filename helpers
# =============================================================================

def make_model_filename() -> str:
    """Create the S2S input filename."""

    return (
        f"{config.dirs['s2s_processed']}"
        f"distribution_monthly_extremes_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )


def make_era5_filename() -> str:
    """Create the ERA5 input filename."""

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
    """Create the SeNorge or SeNorge-regrid input filename."""

    return (
        f"{config.dirs[f'{reference_dataset}_processed']}"
        f"distribution_monthly_extremes_{reference_variable}_{x_days}dayacc_"
        f"{catchment}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


# =============================================================================
# Data loading
# =============================================================================

def load_datasets(
    filename_model: str,
    filename_era5: str,
    filename_reference: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open model, ERA5, and reference datasets."""

    model_ds = xr.open_dataset(filename_model)
    era5_ds = xr.open_dataset(filename_era5)
    reference_ds = xr.open_dataset(filename_reference)

    return model_ds, era5_ds, reference_ds


def check_variable_exists(ds: xr.Dataset, variable: str, dataset_name: str) -> None:
    """Raise a clear error if a required variable is missing."""

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are: {list(ds.data_vars)}"
        )


# =============================================================================
# Data extraction
# =============================================================================

def get_model_values_by_month(
    model_ds: xr.Dataset,
    variable: str = MODEL_EXTREME_VARIABLE,
) -> list[np.ndarray]:
    """
    Convert S2S extremes into one array per month.

    Expected input:
        model_ds[variable](month_of_year, index)
    """

    check_variable_exists(model_ds, variable, "model dataset")

    values_by_month = []

    for month in MONTHS:
        values = model_ds[variable].sel(month_of_year=month).values
        values = values[np.isfinite(values)]
        values_by_month.append(values)

    return values_by_month


def get_monthly_records_before_hans(
    ds: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """
    Get monthly records before Storm Hans.

    Uses 1957–2022, so Storm Hans in 2023 is excluded.
    """

    check_variable_exists(ds, variable, "reference dataset")

    before_hans = ds[variable].sel(year=slice(1957, 2022))
    records = before_hans.max(dim="year")

    return records


def get_storm_hans_event(
    ds: xr.Dataset,
    variable: str,
) -> tuple[int, float]:
    """
    Get the largest 2023 event.

    This assumes the largest 2023 value corresponds to Storm Hans.
    Returns:
        month, value
    """

    check_variable_exists(ds, variable, "reference dataset")

    values_2023 = ds[variable].sel(year=2023)

    flat = values_2023.stack(z=("month",))
    max_index = flat.argmax("z")

    max_value = flat.isel(z=max_index)
    max_month = flat["month"].isel(z=max_index)

    return int(max_month.values), float(max_value.values)


def get_highest_may_model_event(
    model_ds: xr.Dataset,
    variable: str = MODEL_EXTREME_VARIABLE,
) -> tuple[int, float]:
    """
    Get the largest May event in the S2S archive.

    Returns:
        month, value
    """

    check_variable_exists(model_ds, variable, "model dataset")

    may_values = model_ds[variable].sel(month_of_year=5)
    may_max = may_values.max()

    return 5, float(may_max.values)


# =============================================================================
# Plotting helpers
# =============================================================================

def make_legend_handles(reference_label: str) -> list[Line2D]:
    """Create legend handles for the plot."""

    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=f"{reference_label} record 1957–2022",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 record 1957–2022",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=f"{reference_label} Storm Hans 2023",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 Storm Hans 2023",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.6",
            markeredgewidth=0.8,
            markersize=5,
            label="Model extremes",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label="Counterfactual spring Storm Hans",
        ),
    ]


def apply_axis_formatting(ax) -> None:
    """Apply consistent axis formatting."""

    catchment_label = get_catchment_label(catchment)

    ax.set_title(
        f"{catchment_label}, monthly {x_days}-day accumulated precipitation maxima",
        fontsize=TITLE_FONTSIZE,
        pad=8,
    )

    ax.set_ylabel("Precipitation [mm]", fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)

    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS)

    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Main plotting function
# =============================================================================

def plot_monthly_extreme_distributions(
    model_values_by_month: list[np.ndarray],
    era5_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    model_ds: xr.Dataset,
    reference_variable: str,
    reference_label: str,
    filename_out: str,
    write2file: bool,
) -> None:
    """Create the monthly precipitation extreme distribution figure."""

    era5_records = get_monthly_records_before_hans(
        era5_ds,
        variable=ERA5_VARIABLE,
    )

    reference_records = get_monthly_records_before_hans(
        reference_ds,
        variable=reference_variable,
    )

    era5_hans_month, era5_hans_value = get_storm_hans_event(
        era5_ds,
        variable=ERA5_VARIABLE,
    )

    reference_hans_month, reference_hans_value = get_storm_hans_event(
        reference_ds,
        variable=reference_variable,
    )

    counterfactual_month, counterfactual_value = get_highest_may_model_event(
        model_ds,
        variable=MODEL_EXTREME_VARIABLE,
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
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
        boxprops=dict(color="0.25", linewidth=1.0),
        whiskerprops=dict(color="0.25", linewidth=1.0),
        capprops=dict(color="0.25", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.4),
    )

    ax.scatter(
        MONTHS,
        reference_records.values,
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
        reference_hans_month,
        reference_hans_value,
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

    apply_axis_formatting(ax)

    ax.legend(
        handles=make_legend_handles(reference_label),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
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

    reference_variable = get_reference_variable(reference_dataset)
    reference_label = get_reference_label(reference_dataset)

    filename_model = make_model_filename()
    filename_era5 = make_era5_filename()
    filename_reference = make_reference_filename(
        reference_dataset=reference_dataset,
        reference_variable=reference_variable,
    )

    print("Reading model file:    ", filename_model)
    print("Reading ERA5 file:     ", filename_era5)
    print("Reading reference file:", filename_reference)

    model_ds, era5_ds, reference_ds = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_reference=filename_reference,
    )

    try:
        model_values_by_month = get_model_values_by_month(
            model_ds,
            variable=MODEL_EXTREME_VARIABLE,
        )

        plot_monthly_extreme_distributions(
            model_values_by_month=model_values_by_month,
            era5_ds=era5_ds,
            reference_ds=reference_ds,
            model_ds=model_ds,
            reference_variable=reference_variable,
            reference_label=reference_label,
            filename_out=filename_out,
            write2file=write2file,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        reference_ds.close()
