"""
Draft figure 01 for Hans paper: observational time series only.

3-panel plot:
  (1) Snow depth
  (2) Precipitation
  (3) Streamflow

Each panel shows:
  - daily values for 2023
  - shaded day-of-year 95% interval across all years
  - median day-of-year climatology in red

User-configurable plotting options:
  - figure size
  - x/y tick label font size
  - x/y axis label font size
  - title font size
  - legend font size
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Dunnsigouin_etal_2026 import config


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
path_in_obs               = config.dirs["obs"]
path_out                  = config.dirs["fig"]
filename_in_streamflow    = f"{path_in_obs}streamflow.Bergheim.nc"
filename_in_precipitation = f"{path_in_obs}precipitation.ål.III.nc"
filename_in_snowdepth     = f"{path_in_obs}snowdepth.ål.III.nc"
filename_out              = f"{path_out}fig-02.png"
write2file                = True


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_obs_data(filename_streamflow, filename_precipitation, filename_snowdepth):
    """
    Load the three observational datasets.

    Returns
    -------
    ds_streamflow, ds_precipitation, ds_snowdepth : xr.Dataset
        Observational datasets for streamflow, precipitation, and snow depth.
    """
    ds_streamflow = xr.open_dataset(filename_streamflow)
    ds_precipitation = xr.open_dataset(filename_precipitation)
    ds_snowdepth = xr.open_dataset(filename_snowdepth)

    # Restrict streamflow period if desired
    ds_streamflow = ds_streamflow.sel(time=slice("1921-01-01", "2025-12-31"))

    return ds_streamflow, ds_precipitation, ds_snowdepth


# -----------------------------------------------------------------------------
# Time-series helpers
# -----------------------------------------------------------------------------
def year_series_and_climatology_by_doy(da: xr.DataArray, year: int):
    """
    Extract one full year's daily series and compute day-of-year climatological
    statistics across all available years.

    Parameters
    ----------
    da : xr.DataArray
        Time series with a 'time' coordinate.
    year : int
        Year to extract, e.g. 2023.

    Returns
    -------
    x_dates : pandas.DatetimeIndex
        Daily dates for the selected year.
    y_year : np.ndarray
        Daily values for the selected year.
    q_low : np.ndarray
        2.5th percentile by day-of-year.
    q_high : np.ndarray
        97.5th percentile by day-of-year.
    q_median : np.ndarray
        Median by day-of-year.
    """
    da = da.dropna("time")

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    x_dates = pd.date_range(start, end, freq="D")

    # Extract selected year and ensure daily resolution
    da_year = da.sel(time=slice(start, end))
    if da_year.sizes.get("time", 0) != len(x_dates):
        da_year = da_year.resample(time="1D").mean().sel(time=slice(start, end))

    y_year = da_year.values

    # Day-of-year climatology across all years
    q = da.groupby("time.dayofyear").quantile([0.025, 0.5, 0.975], dim="time")

    # Keep same behavior as your original script: 365-day climatology
    doy = np.arange(1, 366)
    q_low = q.sel(quantile=0.025).sel(dayofyear=doy, drop=True).values
    q_median = q.sel(quantile=0.5).sel(dayofyear=doy, drop=True).values
    q_high = q.sel(quantile=0.975).sel(dayofyear=doy, drop=True).values

    return x_dates, y_year, q_low, q_high, q_median


def format_timeseries_axis(
    ax,
    year: int,
    tick_labelsize=10,
    axis_labelsize=11,
):
    """
    Apply consistent formatting to a time-series axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format.
    year : int
        Year shown on the x-axis.
    tick_labelsize : int or float
        Font size for x and y tick labels.
    axis_labelsize : int or float
        Font size for x and y axis labels.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.set_xlim(start, end)
    ax.margins(x=0)

    ax.set_xlabel("Month", fontsize=axis_labelsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)


# -----------------------------------------------------------------------------
# Panel plotting functions
# -----------------------------------------------------------------------------
def plot_panel_streamflow(
    ax,
    ds_streamflow: xr.Dataset,
    year=2023,
    var="vannforing",
    axis_labelsize=11,
    title_fontsize=12,
):
    """
    Plot streamflow panel.
    """
    da = ds_streamflow[var]
    x, y, lo, hi, med = year_series_and_climatology_by_doy(da, year)

    ax.fill_between(x, lo, hi, alpha=0.25, label="95% interval over all years")
    ax.plot(x, med, linewidth=1.4, color="tab:red", label="Median over all years")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title("c) Bergheim station streamflow", fontsize=title_fontsize)
    ax.set_ylabel("m³/s", fontsize=axis_labelsize)


def plot_panel_precipitation(
    ax,
    ds_precipitation: xr.Dataset,
    year=2023,
    var="precipitation",
    axis_labelsize=11,
    title_fontsize=12,
):
    """
    Plot 2-day accumulated precipitation panel.
    """
    da = ds_precipitation[var]

    # Convert daily precipitation to 2-day accumulated precipitation
    da_2day = da.rolling(time=2, min_periods=2).sum()

    x, y, lo, hi, med = year_series_and_climatology_by_doy(da_2day, year)

    ax.fill_between(x, lo, hi, alpha=0.25, label="95% interval over all years")
    ax.plot(x, med, linewidth=1.4, color="tab:red", label="Median over all years")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title(
        "b) Ål III station 2-day accumulated precipitation",
        fontsize=title_fontsize,
    )
    ax.set_ylabel("mm / 2 days", fontsize=axis_labelsize)


def plot_panel_snowdepth(
    ax,
    ds_snowdepth: xr.Dataset,
    year=2023,
    var="snowdepth",
    axis_labelsize=11,
    title_fontsize=12,
):
    """
    Plot snow depth panel.
    """
    da = ds_snowdepth[var]
    x, y, lo, hi, med = year_series_and_climatology_by_doy(da, year)

    ax.fill_between(x, lo, hi, alpha=0.25, label="95% interval over all years")
    ax.plot(x, med, linewidth=1.4, color="tab:red", label="Median over all years")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title("a) Ål III station snow depth", fontsize=title_fontsize)
    ax.set_ylabel("cm", fontsize=axis_labelsize)


# -----------------------------------------------------------------------------
# Main plotting routine
# -----------------------------------------------------------------------------
def plot_all_panels(
    ds_streamflow,
    ds_precipitation,
    ds_snowdepth,
    year=2023,
    outfile=None,
    write2file=False,
    figsize=(20, 5),
    tick_labelsize=11,
    axis_labelsize=11,
    title_fontsize=12,
    legend_fontsize=11,
):
    """
    Create a 3-panel horizontal figure with snow depth, precipitation,
    and streamflow.

    Parameters
    ----------
    ds_streamflow, ds_precipitation, ds_snowdepth : xr.Dataset
        Input datasets.
    year : int
        Year to plot.
    outfile : str or None
        Output filename if saving figure.
    write2file : bool
        If True, save the figure to file.
    figsize : tuple
        Figure size, e.g. (20, 5).
    tick_labelsize : int or float
        Font size for x and y tick labels.
    axis_labelsize : int or float
        Font size for x and y axis labels.
    title_fontsize : int or float
        Font size for panel titles.
    legend_fontsize : int or float
        Font size for legend text.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    # Panel order: snow depth, precipitation, streamflow
    ax_snowdepth, ax_precipitation, ax_streamflow = axes

    plot_panel_snowdepth(
        ax_snowdepth,
        ds_snowdepth,
        year=year,
        var="snowdepth",
        axis_labelsize=axis_labelsize,
        title_fontsize=title_fontsize,
    )

    plot_panel_precipitation(
        ax_precipitation,
        ds_precipitation,
        year=year,
        var="precipitation",
        axis_labelsize=axis_labelsize,
        title_fontsize=title_fontsize,
    )

    plot_panel_streamflow(
        ax_streamflow,
        ds_streamflow,
        year=year,
        var="vannforing",
        axis_labelsize=axis_labelsize,
        title_fontsize=title_fontsize,
    )

    for ax in axes:
        format_timeseries_axis(
            ax,
            year=year,
            tick_labelsize=tick_labelsize,
            axis_labelsize=axis_labelsize,
        )

    # Use one shared legend
    handles, labels = ax_streamflow.get_legend_handles_labels()
    ax_snowdepth.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
        fontsize=legend_fontsize,
    )

    fig.tight_layout()

    if write2file and outfile:
        fig.savefig(outfile, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ds_streamflow, ds_precipitation, ds_snowdepth = load_obs_data(
        filename_in_streamflow,
        filename_in_precipitation,
        filename_in_snowdepth,
    )

    plot_all_panels(
        ds_streamflow,
        ds_precipitation,
        ds_snowdepth,
        year=2023,
        outfile=filename_out,
        write2file=write2file,
        figsize=(20, 5),        
        tick_labelsize=12,      
        axis_labelsize=12,      
        title_fontsize=12,      
        legend_fontsize=11,     
    )
