#!/usr/bin/env python3
"""
Compute ERA5 daily snowmelt climatologies from yearly snow water equivalent files.

Steps:
1. Read yearly NetCDF files with xarray.open_mfdataset.
2. Compute daily snowmelt as positive decreases in sd:
      melt = sd(t-1) - sd(t)
3. Convert melt from meters to millimeters.
4. Calculate monthly climatological mean, median, or standard deviation.
5. Plot 12 monthly panels on a Lambert Conformal map.
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from Dunnsigouin_etal_2026 import config


# ============================================================
# User input parameters
# ============================================================

START_YEAR = 1995
END_YEAR = 2025

STATISTIC = "mean"  # Options: "mean", "median", "std"

VAR_NAME = "sd"

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

PATH_IN = Path(config.dirs["era5_continuous_daily_scandinavia"],VAR_NAME)
FILE_TEMPLATE = "sd_0.25x0.25_{year}.nc"

CMAP = "RdBu_r"
CBAR_MIN = -10
CBAR_MAX = 10
CBAR_INTERVAL = 2

write2file = False
OUTPUT_FIG = f"xy_era5_monthly_snowmelt_climatology_{STATISTIC}_{START_YEAR}_{END_YEAR}.png"


# ============================================================
# Functions
# ============================================================

def get_file_list(path_in, start_year, end_year, file_template):
    """Return list of yearly NetCDF files within the selected year range."""
    files = [
        path_in / file_template.format(year=year)
        for year in range(start_year, end_year + 1)
    ]

    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files, e.g. {missing[:5]}")

    return files


def open_sd_dataset(files, var_name):
    """Open yearly sd files as one xarray dataset."""
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"time": 365},
        parallel=True,
    )

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    return ds


def calculate_daily_snowmelt(ds, var_name):
    """
    Calculate daily change as sd(t-1) - sd(t).

    Positive values mean sd decreased.
    Negative values mean sd increased.
    Output is in mm water equivalent per day.
    """
    sd = ds[var_name]

    melt = sd.diff("time")   # equivalent to sd(t-1) - sd(t)
    melt = melt * 1000.0      # m to mm

    melt.name = "snowmelt"
    melt.attrs["units"] = "mm day-1"
    melt.attrs["long_name"] = "Daily sd(t-1) minus sd(t)"

    return melt


def monthly_climatology(melt, statistic):
    """Calculate monthly climatological mean, median, or standard deviation."""
    statistic = statistic.lower()

    if statistic == "mean":
        clim = melt.groupby("time.month").mean("time", skipna=True)
    elif statistic == "median":
        clim = melt.groupby("time.month").median("time", skipna=True)
    elif statistic == "std":
        clim = melt.groupby("time.month").std("time", skipna=True)
    else:
        raise ValueError("STATISTIC must be one of: 'mean', 'median', 'std'.")

    clim.name = f"monthly_snowmelt_{statistic}"
    return clim


def plot_monthly_climatology(
    clim,
    statistic,
    output_fig,
    write2file,
    cmap,
    cbar_min,
    cbar_max,
    cbar_interval,
):
    """Plot 12 monthly climatology panels on a Lambert Conformal map."""
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    proj = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )

    data_crs = ccrs.PlateCarree()

    lon = clim["longitude"]
    lat = clim["latitude"]

    levels = np.arange(cbar_min, cbar_max + cbar_interval, cbar_interval)

    fig, axes = plt.subplots(
        3, 4,
        figsize=(14, 10),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    axes = axes.ravel()

    for i, ax in enumerate(axes):
        month = i + 1
        field = clim.sel(month=month)

        im = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=data_crs,
            shading="auto",
            cmap=cmap,
            vmin=cbar_min,
            vmax=cbar_max,
        )

        ax.set_extent(MAP_EXTENT, crs=data_crs)
        ax.coastlines(resolution="10m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, linewidth=0.3, alpha=0.5)
        ax.set_title(month_names[i])

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.04,
        ticks=levels,
        extend="both",
    )

    cbar.set_label(f"Snowmelt {statistic} [mm day$^{{-1}}$]")

    fig.suptitle(
        f"Monthly climatological snowmelt {statistic}, "
        f"{START_YEAR}–{END_YEAR}",
        fontsize=15,
    )

    if write2file:
        fig.savefig(output_fig, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_fig}")
        plt.close(fig)

    plt.show()        

# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":

    files = get_file_list(PATH_IN, START_YEAR, END_YEAR, FILE_TEMPLATE)

    ds = open_sd_dataset(files, VAR_NAME)

    melt = calculate_daily_snowmelt(ds, VAR_NAME)

    clim = monthly_climatology(melt, STATISTIC)

    plot_monthly_climatology(
        clim,
        STATISTIC,
        OUTPUT_FIG,
        write2file,
        CMAP,
        CBAR_MIN,
        CBAR_MAX,
        CBAR_INTERVAL,
    )

    ds.close()

    print(f"Saved figure to: {OUTPUT_FIG}")


