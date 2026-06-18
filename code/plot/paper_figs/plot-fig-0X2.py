"""
Plot 2-day accumulated precipitation over the Drammen catchment.

The figure shows:
1. Forecast ensemble members from initialization through M lead days.
2. ERA5 from N days before initialization through M lead days.
3. SeNorge from N days before initialization through M lead days.

SeNorge timestamps are shifted back one day before daily rounding.
"""

# =============================================================================
# 1. Import statements
# =============================================================================

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# 2. User-defined input parameters
# =============================================================================

# General settings
forecast_date = "2023-08-05"
catchment = "regine_drammen"
x_days = 1

# Plot-window settings
N_days_before = 3
M_days_lead = 6

# Forecast settings
forecast_variable = "tp24"
forecast_grid = "0.25x0.25"

forecast_filename = (
    config.dirs["s2s_forecast_daily"]
    + forecast_variable
    + "/"
    + f"{forecast_variable}_{forecast_grid}_{forecast_date}.nc"
)

forecast_weights_filename = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_era5_{forecast_grid}.nc"
)

# ERA5 settings
era5_variable = "tp24"
era5_grid = "0.25x0.25"
era5_domain = "norway"

era5_path = config.dirs["era5_continuous_daily"] + era5_variable + "/"
era5_file_pattern = f"{era5_variable}_{era5_grid}" + "_{year}.nc"

era5_weights_filename = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_era5_{era5_grid}.nc"
)

# SeNorge settings
senorge_variable = "rr"

senorge_path = config.dirs["senorge_continuous_daily"] + senorge_variable + "/"
senorge_file_pattern = senorge_variable + "_{year}.nc"

senorge_weights_filename = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_senorge.nc"
)

# Plot settings
figure_size = (11, 5)

forecast_color = "0.7"
forecast_line_width = 1.0
forecast_alpha = 0.6

era5_color = "tab:blue"
senorge_color = "tab:red"
obs_line_width = 2.5

title = (
    f"{x_days}-day accumulated precipitation\n"
    f"{catchment} catchment, forecast initialized {forecast_date}"
)

xlabel = "Date"
ylabel = f"{x_days}-day accumulated precipitation [mm]"

save_figure = False
figure_filename = (
    f"ensemble_timeseries_with_era5_senorge_"
    f"{forecast_variable}_{x_days}dayacc_"
    f"{catchment}_{forecast_date}.png"
)


# =============================================================================
# 3. Functions
# =============================================================================

def standardize_precip_units(da, variable):
    """Convert precipitation to mm."""

    units = str(da.attrs.get("units", "")).strip().lower()

    if variable == "tp24" or units in {"m", "meter", "metre"}:
        da = da * 1000.0
        da.attrs["units"] = "mm"

    elif units in {"kg/m^2", "kg/m2", "kg m-2"}:
        da.attrs["units"] = "mm"

    return da


def check_dims(da, expected_dims, name):
    """Check that required dimensions exist."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def load_weights(filename, spatial_dims):
    """Load catchment weights."""

    ds = xr.open_dataset(filename)

    if "catchment_weight" not in ds:
        raise KeyError(
            f"'catchment_weight' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    weights = ds["catchment_weight"].astype("float32")
    weights.name = "catchment_weight"

    check_dims(weights, spatial_dims, "Catchment weights")

    return weights


def load_forecast(filename, variable):
    """Load forecast precipitation."""

    with xr.open_dataset(filename) as ds:
        da = ds[variable].load()

    da = standardize_precip_units(da, variable)

    check_dims(
        da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast",
    )

    return da


def preprocess_era5(ds):
    """Drop unnecessary ERA5 ensemble dimension if present."""

    return ds.drop_vars("number", errors="ignore")


def load_era5(years, time_start, time_end):
    """Load ERA5 precipitation."""

    filenames = [
        era5_path + era5_file_pattern.format(year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if era5_domain is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(era5_domain)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    da = ds[era5_variable].sel(time=slice(time_start, time_end))
    da = standardize_precip_units(da, era5_variable)

    check_dims(
        da,
        expected_dims=("time", "latitude", "longitude"),
        name="ERA5",
    )

    return da


def load_senorge(years, time_start, time_end):
    """Load SeNorge precipitation year by year."""

    yearly_data = []

    for year in years:
        filename = senorge_path + senorge_file_pattern.format(year=int(year))

        ds = xr.open_dataset(filename)
        ds = xr.decode_cf(ds)

        if senorge_variable not in ds:
            raise KeyError(
                f"'{senorge_variable}' not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[senorge_variable].sel(time=slice(time_start, time_end))

        fill_value = da.attrs.get("_FillValue")
        if fill_value is not None:
            da = da.where(da != fill_value)

        da = standardize_precip_units(da, senorge_variable)

        check_dims(
            da,
            expected_dims=("time", "Y", "X"),
            name="SeNorge",
        )

        yearly_data.append(da.load())
        ds.close()

    return xr.concat(yearly_data, dim="time").sortby("time")


def align_weights(da, weights):
    """Align catchment weights to precipitation grid."""

    grid_template = da.isel(time=0, drop=True)

    try:
        return weights.reindex_like(grid_template)
    except Exception:
        return weights.broadcast_like(grid_template)


def catchment_mean(da, weights, spatial_dims):
    """Calculate catchment-weighted spatial mean precipitation."""

    weights = align_weights(da, weights)

    valid = xr.ufuncs.isfinite(da) & xr.ufuncs.isfinite(weights) & (weights > 0)

    weighted_sum = (da.where(valid) * weights.where(valid)).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = weights.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    out = weighted_sum / weight_sum
    out.name = "catchment_mean_precipitation"
    out.attrs["units"] = da.attrs.get("units", "mm")

    return out


def xday_accumulation(da, x_days):
    """Calculate trailing x-day accumulated precipitation."""

    out = (
        da
        .rolling(time=x_days, min_periods=x_days)
        .sum()
        .dropna("time", how="any")
    )

    out.name = f"{x_days}day_accumulated_precipitation"
    out.attrs["units"] = "mm"

    return out


def get_plot_period(forecast_date, N_days_before, M_days_lead, x_days):
    """
    Define plot and loading periods.

    The loading period includes extra days before the plot start so the first
    x-day accumulated value can be calculated.
    """

    init_date = pd.to_datetime(forecast_date)

    plot_start = init_date - pd.Timedelta(days=N_days_before)
    plot_end = init_date + pd.Timedelta(days=M_days_lead)

    load_start = plot_start - pd.Timedelta(days=x_days + 1)
    load_end = plot_end + pd.Timedelta(days=x_days + 1)

    years = np.arange(load_start.year, load_end.year + 1)

    return init_date, plot_start, plot_end, load_start, load_end, years


def round_time_to_nearest_day(da):
    """Round timestamps to nearest calendar day."""

    rounded_time = pd.to_datetime(da.time.values).round("D")

    return da.assign_coords(time=rounded_time)


def shift_senorge_time_back_one_day(da):
    """Shift SeNorge timestamps one day earlier."""

    shifted_time = pd.to_datetime(da.time.values) - pd.Timedelta(days=1)

    return da.assign_coords(time=shifted_time)


def subset_to_period(da, start_date, end_date):
    """Subset data to a calendar-date period."""

    return da.sel(time=slice(start_date, end_date))


def print_time_diagnostics(forecast, era5, senorge, n=5):
    """Print timestamps before final matching."""

    print("\nTimestamps before final matching:")
    print("Forecast:", forecast.time.values[:n])
    print("ERA5:    ", era5.time.values[:n])
    print("SeNorge: ", senorge.time.values[:n])
    print("")


def restrict_observations_to_common_dates(era5, senorge):
    """
    Restrict ERA5 and SeNorge to their common dates.

    This keeps the pre-initialization observation period.
    """

    common_dates = np.intersect1d(era5.time.values, senorge.time.values)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between ERA5 and SeNorge.")

    era5 = era5.sel(time=common_dates)
    senorge = senorge.sel(time=common_dates)

    print(
        f"\nObservations plotted from {common_dates[0]} "
        f"to {common_dates[-1]}.\n"
    )

    return era5, senorge


def restrict_forecast_to_observation_dates(forecast, obs_dates):
    """
    Restrict forecast to dates that are also available in observations.

    Forecast remains limited to initialization date onward.
    """

    common_dates = np.intersect1d(forecast.time.values, obs_dates)

    if len(common_dates) == 0:
        raise ValueError("No common dates found between forecast and observations.")

    forecast = forecast.sel(time=common_dates)

    print(
        f"Forecast plotted from {common_dates[0]} "
        f"to {common_dates[-1]}.\n"
    )

    return forecast


def plot_ensemble_timeseries(
    forecast,
    era5,
    senorge,
    init_date,
    plot_start,
    plot_end,
):
    """
    Plot forecast ensemble, ERA5, and SeNorge.

    The ensemble member with the highest maximum precipitation is highlighted.
    """

    wettest_member, max_precip, max_time = (
        get_wettest_ensemble_member(forecast)
    )

    fig, ax = plt.subplots(figsize=figure_size)

    # ---------------------------------------------------------------------
    # Dummy line for ensemble legend entry
    # ---------------------------------------------------------------------

    ax.plot(
        [],
        [],
        color=forecast_color,
        linewidth=forecast_line_width,
        alpha=forecast_alpha,
        label="Forecast ensemble",
    )

    # ---------------------------------------------------------------------
    # Plot all non-extreme ensemble members
    # ---------------------------------------------------------------------

    for member in forecast["number"].values:

        if member == wettest_member:
            continue

        ax.plot(
            forecast["time"],
            forecast.sel(number=member),
            color=forecast_color,
            linewidth=forecast_line_width,
            alpha=forecast_alpha,
        )

    # ---------------------------------------------------------------------
    # Highlight wettest ensemble member
    # ---------------------------------------------------------------------

    ax.plot(
        forecast["time"],
        forecast.sel(number=wettest_member),
        color="tab:green",
        linewidth=obs_line_width,
        zorder=10,
        label=(
            f"Wettest ensemble member"
        ),
    )

    # ---------------------------------------------------------------------
    # ERA5
    # ---------------------------------------------------------------------

    ax.plot(
        era5["time"],
        era5,
        color=era5_color,
        linewidth=obs_line_width,
        label="ERA5",
    )

    # ---------------------------------------------------------------------
    # SeNorge
    # ---------------------------------------------------------------------

    ax.plot(
        senorge["time"],
        senorge,
        color=senorge_color,
        linewidth=obs_line_width,
        label="SeNorge",
    )

    # ---------------------------------------------------------------------
    # Forecast initialization
    # ---------------------------------------------------------------------

    ax.axvline(
        init_date,
        color="k",
        linewidth=1.2,
        linestyle="--",
        label="Forecast initialization",
    )

    # ---------------------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------------------

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3)

    # Remove whitespace at ends of x-axis
    ax.set_xlim(plot_start, plot_end)
    ax.margins(x=0)

    # Rotate date labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    ax.legend(loc='upper left',frameon=False)

    plt.tight_layout()

    if save_figure:
        plt.savefig(
            figure_filename,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


    
def get_wettest_ensemble_member(forecast):
    """
    Identify the ensemble member with the highest accumulated precipitation
    at any point during the forecast period.

    Parameters
    ----------
    forecast : xarray.DataArray
        Forecast accumulated precipitation with dimensions
        (time, number).

    Returns
    -------
    member : int
        Ensemble member number.

    max_precip : float
        Maximum precipitation [mm].

    max_time : pandas.Timestamp
        Date of maximum precipitation.
    """

    # Maximum value for each member
    member_max = forecast.max(dim="time")

    # Member producing largest value
    member = int(member_max.idxmax(dim="number"))

    # Value itself
    max_precip = float(member_max.max())

    # Date of occurrence
    member_series = forecast.sel(number=member)

    max_time = pd.Timestamp(
        member_series.time[
            member_series.argmax(dim="time")
        ].values
    )

    return member, max_precip, max_time


# =============================================================================
# 4. Main script
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------------------

    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        obs_years,
    ) = get_plot_period(
        forecast_date=forecast_date,
        N_days_before=N_days_before,
        M_days_lead=M_days_lead,
        x_days=x_days,
    )

    # -------------------------------------------------------------------------
    # Forecast
    # -------------------------------------------------------------------------

    forecast_weights = load_weights(
        filename=forecast_weights_filename,
        spatial_dims=("latitude", "longitude"),
    )

    forecast = load_forecast(
        filename=forecast_filename,
        variable=forecast_variable,
    )

    forecast_mean = catchment_mean(
        da=forecast,
        weights=forecast_weights,
        spatial_dims=("latitude", "longitude"),
    )

    forecast_accumulated = xday_accumulation(
        da=forecast_mean,
        x_days=x_days,
    )

    forecast_accumulated = round_time_to_nearest_day(forecast_accumulated)

    # Forecast only exists from initialization onward.
    forecast_accumulated = subset_to_period(
        da=forecast_accumulated,
        start_date=init_date,
        end_date=plot_end,
    )

    # -------------------------------------------------------------------------
    # ERA5
    # -------------------------------------------------------------------------

    era5_weights = load_weights(
        filename=era5_weights_filename,
        spatial_dims=("latitude", "longitude"),
    )

    era5 = load_era5(
        years=obs_years,
        time_start=load_start,
        time_end=load_end,
    )

    era5_mean = catchment_mean(
        da=era5,
        weights=era5_weights,
        spatial_dims=("latitude", "longitude"),
    )

    era5_accumulated = xday_accumulation(
        da=era5_mean,
        x_days=x_days,
    )

    era5_accumulated = round_time_to_nearest_day(era5_accumulated)

    era5_accumulated = subset_to_period(
        da=era5_accumulated,
        start_date=plot_start,
        end_date=plot_end,
    )

    # -------------------------------------------------------------------------
    # SeNorge
    # -------------------------------------------------------------------------

    senorge_weights = load_weights(
        filename=senorge_weights_filename,
        spatial_dims=("Y", "X"),
    )

    senorge = load_senorge(
        years=obs_years,
        time_start=load_start,
        time_end=load_end,
    )

    # Shift SeNorge timestamps one day earlier before accumulation.
    senorge = shift_senorge_time_back_one_day(senorge)

    senorge_mean = catchment_mean(
        da=senorge,
        weights=senorge_weights,
        spatial_dims=("Y", "X"),
    )

    senorge_accumulated = xday_accumulation(
        da=senorge_mean,
        x_days=x_days,
    )

    senorge_accumulated = round_time_to_nearest_day(senorge_accumulated)

    senorge_accumulated = subset_to_period(
        da=senorge_accumulated,
        start_date=plot_start,
        end_date=plot_end,
    )

    # -------------------------------------------------------------------------
    # Observation and forecast matching
    # -------------------------------------------------------------------------

    era5_plot, senorge_plot = restrict_observations_to_common_dates(
        era5=era5_accumulated,
        senorge=senorge_accumulated,
    )

    forecast_accumulated = restrict_forecast_to_observation_dates(
        forecast=forecast_accumulated,
        obs_dates=era5_plot.time.values,
    )

    # ------------------------------------------------------------------------- 
    # get info on wettest member
    # -------------------------------------------------------------------------    
    member, max_precip, max_time = get_wettest_ensemble_member(
    forecast_accumulated
    )

    print(
        f"Wettest ensemble member: {member}\n"
        f"Maximum precipitation: {max_precip:.1f} mm\n"
        f"Date: {max_time:%Y-%m-%d}"
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    
    plot_ensemble_timeseries(
        forecast=forecast_accumulated,
        era5=era5_plot,
        senorge=senorge_plot,
        init_date=init_date,
        plot_start=plot_start,
        plot_end=plot_end,
    )
