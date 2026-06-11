"""
Plots fig. 3 in the paper: monthly distributions of precipitation extremes.

The figure has two panels:
1. Reference data, e.g. ERA5 or SeNorge
2. Model data, i.e. S2S forecast/hindcast extremes

Each panel shows one box-and-whisker distribution per month of year.
Outliers are shown as empty circles.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from Dunnsigouin_etal_2026 import config


# Input -------------------------------------------------
variable            = "tp"
x_days              = 2
catchment           = "regine_drammen"
forecast_date_range = ["2020-01-02", "2023-06-26"]

reference_dataset = "senorge"
grid              = "0.5x0.5"
reference_years   = ["1957", "2023"]

path_in_model = config.dirs["s2s_processed"]
path_in_ref   = config.dirs[f"{reference_dataset}_processed"]
path_out      = config.dirs["fig"]

filename_in_model = (
    f"{path_in_model}"
    f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
    f"nve_catchment_{catchment}_forecast_hindcast_"
    f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
)

if reference_dataset == 'era5':
    filename_in_ref = (
        f"{path_in_ref}"
        f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_{reference_dataset}_{grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )
else:
    filename_in_ref = (
        f"{path_in_ref}"
        f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )

filename_out = f"{path_out}fig-03.png"
write2file   = False
# -------------------------------------------------------


def load_data(filename_in_model, filename_in_ref):
    """Load model and reference datasets."""

    model_ds = xr.open_dataset(filename_in_model)
    ref_ds   = xr.open_dataset(filename_in_ref)

    return model_ds, ref_ds


def get_reference_monthly_values(ref_ds, variable="tp"):
    """
    Convert reference data to a list of arrays, one per month.

    Expected input:
        ref_ds[variable](year, month)
    """

    values_by_month = []

    for month in range(1, 13):
        vals = ref_ds[variable].sel(month=month).values
        vals = vals[np.isfinite(vals)]
        values_by_month.append(vals)

    return values_by_month


def get_model_monthly_values(model_ds, variable="max_value"):
    """
    Convert model data to a list of arrays, one per month.

    Expected input:
        model_ds[variable](month_of_year, index)
    """

    values_by_month = []

    for month in range(1, 13):
        vals = model_ds[variable].sel(month_of_year=month).values
        vals = vals[np.isfinite(vals)]
        values_by_month.append(vals)

    return values_by_month


def plot_monthly_boxplots(ref_values, model_values, ref_ds, filename_out=None, write2file=False):
    """Plot monthly boxplots and overlay reference maxima."""

    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Blue dots:
    # largest reference value in each month, using only 1957–2022
    ref_1957_2022 = ref_ds[variable].sel(year=slice(1957, 2022))
    ref_monthly_max_1957_2022 = ref_1957_2022.max(dim="year")

    # Red dot:
    # single largest reference value over all months and years, using 1957–2023
    ref_1957_2023 = ref_ds[variable].sel(year=slice(1957, 2023))

    ref_flat = ref_1957_2023.stack(z=("year", "month"))
    ref_abs_idx = ref_flat.argmax("z")

    ref_abs_max = ref_flat.isel(z=ref_abs_idx)
    ref_abs_month = ref_flat["month"].isel(z=ref_abs_idx)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11, 7),
        sharex=True,
    )

    flierprops = dict(
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        markersize=4,
        linestyle="none",
    )

    boxplot_kwargs = dict(
        positions=months,
        widths=0.6,
        patch_artist=False,
        showfliers=True,
        flierprops=flierprops,
    )

    # Reference panel
    axes[0].boxplot(ref_values, **boxplot_kwargs)
    axes[0].set_title("Reference data")
    axes[0].set_ylabel(f"{x_days}-day precipitation extremes (mm)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 125)

    # Model panel
    axes[1].boxplot(model_values, **boxplot_kwargs)

    axes[1].scatter(
        months,
        ref_monthly_max_1957_2022.values,
        color="blue",
        s=35,
        zorder=4,
        label="record for 1957–2022",
    )

    axes[1].scatter(
        int(ref_abs_month.values),
        float(ref_abs_max.values),
        color="red",
        s=45,
        zorder=5,
        label="Storm Hans",
    )

    axes[1].set_title("Model data")
    axes[1].set_ylabel(f"{x_days}-day precipitation extremes (mm)")
    axes[1].set_xlabel("Month of year")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 125)

    axes[1].set_xticks(months)
    axes[1].set_xticklabels(month_labels)
    axes[1].legend(loc='best',frameon=False)

    fig.tight_layout()

    if write2file:
        fig.savefig(filename_out, dpi=300, bbox_inches="tight")
        print("Wrote:", filename_out)

    plt.show()


    
if __name__ == "__main__":

    model_ds, ref_ds = load_data(filename_in_model, filename_in_ref)

    ref_values = get_reference_monthly_values(
        ref_ds,
        variable=variable,
    )

    model_values = get_model_monthly_values(
        model_ds,
        variable="max_value",
    )

    plot_monthly_boxplots(
        ref_values=ref_values,
        model_values=model_values,
        ref_ds=ref_ds,
        filename_out=filename_out,
        write2file=write2file,
    )
