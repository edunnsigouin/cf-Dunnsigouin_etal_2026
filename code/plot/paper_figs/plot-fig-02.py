"""
Plots fig. 2/3 in the paper: monthly S2S distributions of precipitation extremes.

Single publication-quality panel:
- Box-and-whisker distributions show S2S forecast/hindcast monthly extremes.
- Blue dots show SeNorge monthly records before Storm Hans.
- Red dots show ERA5 monthly records before Storm Hans.
- Blue triangle shows SeNorge Storm Hans.
- Red triangle shows ERA5 Storm Hans.
- Green dot shows the highest May S2S event, labelled counterfactual event.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from Dunnsigouin_etal_2026 import config


# =============================================================================
# Input
# =============================================================================

variable = "tp"
x_days = 2
catchment = "regine_drammen"
forecast_date_range = ["2020-01-02", "2023-06-26"]

grid = "0.5x0.5"
reference_years = ["1957", "2023"]

path_in_model = config.dirs["s2s_processed"]
path_out = config.dirs["fig"]

path_in_era5 = config.dirs["era5_processed"]
path_in_senorge = config.dirs["senorge_processed"]

filename_in_model = (
    f"{path_in_model}"
    f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
    f"nve_catchment_{catchment}_forecast_hindcast_"
    f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
)

filename_in_era5 = (
    f"{path_in_era5}"
    f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
    f"nve_catchment_{catchment}_era5_{grid}_"
    f"{reference_years[0]}-{reference_years[1]}.nc"
)

filename_in_senorge = (
    f"{path_in_senorge}"
    f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
    f"nve_catchment_{catchment}_senorge_"
    f"{reference_years[0]}-{reference_years[1]}.nc"
)

filename_out = f"{path_out}fig-02.png"
write2file = True


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

SENORGE_COLOR = "tab:blue"
ERA5_COLOR = "tab:red"
COUNTERFACTUAL_COLOR = "tab:green"

MONTHS = np.arange(1, 13)
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Data loading
# =============================================================================

def load_data(filename_in_model, filename_in_era5, filename_in_senorge):
    """Load model, ERA5, and SeNorge datasets."""
    model_ds = xr.open_dataset(filename_in_model)
    era5_ds = xr.open_dataset(filename_in_era5)
    senorge_ds = xr.open_dataset(filename_in_senorge)

    return model_ds, era5_ds, senorge_ds


def get_model_monthly_values(model_ds, variable="max_value"):
    """
    Convert model data to a list of arrays, one per month.

    Expected input:
        model_ds[variable](month_of_year, index)
    """
    values_by_month = []

    for month in MONTHS:
        vals = model_ds[variable].sel(month_of_year=month).values
        vals = vals[np.isfinite(vals)]
        values_by_month.append(vals)

    return values_by_month


def get_reference_monthly_records(ref_ds, variable="tp"):
    """
    Get monthly records before Storm Hans.

    Uses 1957-2022, so the 2023 Storm Hans event is not included
    in the monthly record markers.
    """
    ref_before_hans = ref_ds[variable].sel(year=slice(1957, 2022))
    return ref_before_hans.max(dim="year")


def get_reference_storm_hans_event(ref_ds, variable="tp"):
    """
    Get the 2023 Storm Hans event.

    This assumes the largest 2023 reference value is Storm Hans.
    """
    ref_2023 = ref_ds[variable].sel(year=2023)

    ref_flat = ref_2023.stack(z=("month",))
    ref_abs_idx = ref_flat.argmax("z")

    ref_abs_max = ref_flat.isel(z=ref_abs_idx)
    ref_abs_month = ref_flat["month"].isel(z=ref_abs_idx)

    return int(ref_abs_month.values), float(ref_abs_max.values)


def get_highest_may_model_event(model_ds, variable="max_value"):
    """Get the highest May model event."""
    may_values = model_ds[variable].sel(month_of_year=5)
    may_max = may_values.max()

    return 5, float(may_max.values)


# =============================================================================
# Plotting
# =============================================================================

def plot_monthly_boxplots(
    model_values,
    era5_ds,
    senorge_ds,
    model_ds,
    filename_out=None,
    write2file=False,
):
    """Plot publication-quality monthly S2S boxplots with reference markers."""

    senorge_records = get_reference_monthly_records(
        senorge_ds,
        variable=variable,
    )

    era5_records = get_reference_monthly_records(
        era5_ds,
        variable=variable,
    )

    senorge_hans_month, senorge_hans_value = get_reference_storm_hans_event(
        senorge_ds,
        variable=variable,
    )

    era5_hans_month, era5_hans_value = get_reference_storm_hans_event(
        era5_ds,
        variable=variable,
    )

    counterfactual_month, counterfactual_value = get_highest_may_model_event(
        model_ds,
        variable="max_value",
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
    )

    flierprops = dict(
        marker="o",
        markerfacecolor="none",
        markeredgecolor="0.25",
        markersize=4,
        linestyle="none",
        markeredgewidth=0.8,
    )

    boxprops = dict(
        color="0.25",
        linewidth=1.0,
    )

    whiskerprops = dict(
        color="0.25",
        linewidth=1.0,
    )

    capprops = dict(
        color="0.25",
        linewidth=1.0,
    )

    medianprops = dict(
        color="black",
        linewidth=1.4,
    )

    ax.boxplot(
        model_values,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,
        flierprops=flierprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        medianprops=medianprops,
    )

    ax.scatter(
        MONTHS,
        senorge_records.values,
        facecolors="w",
        edgecolors=SENORGE_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
        label="SeNorge record",
    )

    ax.scatter(
        MONTHS,
        era5_records.values,
        facecolors="w",
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
        label="ERA5 record",
    )

    ax.scatter(
        senorge_hans_month,
        senorge_hans_value,
        facecolors="w",
        edgecolors=SENORGE_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
        label="SeNorge Storm Hans",
    )

    ax.scatter(
        era5_hans_month,
        era5_hans_value,
        facecolors="w",
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
        label="ERA5 Storm Hans",
    )

    ax.scatter(
        counterfactual_month,
        counterfactual_value,
        facecolors="w",
        marker="o",
        edgecolors=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        s=20,
        zorder=6,
        label="Counterfactual spring storm Hans",
    )

    if catchment == 'regine_drammen':
        ax.set_title("Drammen catchment, monthly 2-day accumulated precipitation maxima",fontsize=TITLE_FONTSIZE,pad=8)
    elif catchment == 'regine_glomma':
        ax.set_title("Glomma catchment, monthly 2-day accumulated precipitation maxima",fontsize=TITLE_FONTSIZE,pad=8)
        
    ax.set_ylabel("Precipitation [mm]", fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)

    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS)

    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [

         Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=SENORGE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="SeNorge record 1957–2022",
        ),
        
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 record 1957–2022",
        ),
        
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=SENORGE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="SeNorge Storm Hans 2023",
        ),

        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
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
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=5,
            label="Model extremes",
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label=f"Counterfactual spring storm Hans",
        ),
    ]

    ax.legend(
        handles=legend_handles,
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

    model_ds, era5_ds, senorge_ds = load_data(
        filename_in_model=filename_in_model,
        filename_in_era5=filename_in_era5,
        filename_in_senorge=filename_in_senorge,
    )

    try:
        model_values = get_model_monthly_values(
            model_ds,
            variable="max_value",
        )

        plot_monthly_boxplots(
            model_values=model_values,
            era5_ds=era5_ds,
            senorge_ds=senorge_ds,
            model_ds=model_ds,
            filename_out=filename_out,
            write2file=write2file,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        senorge_ds.close()
