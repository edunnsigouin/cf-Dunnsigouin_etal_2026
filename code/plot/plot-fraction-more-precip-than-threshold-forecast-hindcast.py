"""
Plots the fraction of ensemble members with X-day accumulated precipitation
greater than a user-defined threshold in forecasts and hindcasts.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Dunnsigouin_etal_2026 import config

# input --------------------------------------
forecast_date_range = ["2020-01-02", "2023-06-26"]
acc_days            = 2
rain_threshold_mm   = 30.0   # must match nc file
path_in             = config.dirs["s2s_processed"]
path_out            = config.dirs["fig"]
write2file          = True
# --------------------------------------------


def load_data(forecast_date_range, acc_days, threshold_mm, path_in):
    filename = (
        f"{path_in}xyt_model_events_with_more_{acc_days}_day_accumulated_precip_than_"
        f"{threshold_mm:g}mm_{forecast_date_range[0]}-{forecast_date_range[-1]}.nc"
    )
    return xr.open_dataset(filename)


def plot_fraction_map(count, *,
                      cmap="RdPu_r",
                      levels=None,
                      extend="max",
                      title=None,
                      figsize=(6, 5),
                      dpi=150):
    """
    Plot a single-panel map of fraction/count with contourf + colorbar.
    """

    lat_name = "latitude"
    lon_name = "longitude"

    lats = count[lat_name].values
    lons = count[lon_name].values
    extent = [float(lons.min()), float(lons.max()),
              float(lats.min()), float(lats.max())]

    if levels is None:
        levels = np.arange(0, 1.05, 0.05)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": proj},
        constrained_layout=True
    )

    ax.set_extent(extent, crs=proj)
    ax.coastlines(resolution="50m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7)

    cf = ax.contourf(
        count[lon_name], count[lat_name], count,
        levels=levels,
        cmap=cmap,
        extend=extend,
        transform=proj
    )

    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.9, pad=0.05)
    cb.set_label("fraction")

    if title is not None:
        ax.set_title(title)

    return fig, ax


if __name__ == "__main__":

    ds = load_data(forecast_date_range, acc_days, rain_threshold_mm, path_in)
    count = ds["count"]

    fig, ax = plot_fraction_map(
        count,
        cmap="PiYG_r",
        levels=np.arange(0, 0.52, 0.02),
        title=f"Fraction of forecast + hindcast members with\n"
              f"{acc_days}-day accumulated precipitation > {rain_threshold_mm:g} mm",
    )

    if write2file:
        outname = (
            f"{path_out}/xy_fraction_{acc_days}day_accumulated_precip_"
            f"greater_than_{rain_threshold_mm:g}mm_"
            f"forecast_hindcast_{forecast_date_range[0]}_{forecast_date_range[-1]}.png"
        )
        fig.savefig(outname, bbox_inches="tight")

    plt.show()
