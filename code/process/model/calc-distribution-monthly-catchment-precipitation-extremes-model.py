"""
Calculates distribution of monthly extremes for catchment averaged and time accumulated
total precipitation in model forecasts and hindcasts. 
"""

import numpy as np
import xarray as xr
import pandas as pd
from Dunnsigouin_etal_2026 import config

# input ------------------------------------------------
variable            = 'tp24'
x_days              = 2
catchment           = "regine_drammen"
forecast_date_range = ['2020-01-02','2023-06-26']
path_in_forecast    = config.dirs['s2s_forecast_daily'] + variable + '/'
path_in_hindcast    = config.dirs['s2s_hindcast_daily'] + variable + '/'
filename_weights    = config.dirs["nve"] + f'weights_catchment_{catchment}_era5_0.5x0.5.nc' # same weights as era5 since same grid
path_out            = config.dirs['s2s_processed']
write2file          = False
# ------------------------------------------------------


def get_forecast_dates(forecast_date_range, option="mt"):
    """
    Generate a list of forecast dates within a given date range.

    Parameters
    ----------
    forecast_date_range : list-like
        ['YYYY-MM-DD', 'YYYY-MM-DD']
    option : str
        'mt'  -> Mondays and Thursdays
        'all' -> all calendar days
    """

    start_date = pd.to_datetime(forecast_date_range[0])
    end_date   = pd.to_datetime(forecast_date_range[1])

    if option == "mt":
        mondays        = pd.date_range(start_date, end_date, freq="W-MON")
        thursdays      = pd.date_range(start_date, end_date, freq="W-THU")
        forecast_dates = mondays.union(thursdays)

    else:
        forecast_dates = pd.date_range(start_date, end_date, freq="D")

    return forecast_dates.sort_values().strftime("%Y-%m-%d").tolist()


def load_weights(filename_weights):
    """Load catchment weights"""
    
    ds = xr.open_dataset(filename_weights)

    if "catchment_weight" not in ds:
        raise KeyError(
            f"'catchment_weight' not found in {path_weights}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    w = ds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"

    return w


def load_model_data(path_in_forecast,path_in_hindcast,variable,date):

    forecast_filename  = path_in_forecast + f'{variable}_0.5x0.5_{date}.nc'
    hindcast_filename = path_in_hindcast + f'{variable}_0.5x0.5_{date}.nc'

    forecast_da = xr.open_dataset(forecast_filename)[variable]
    hindcast_da = xr.open_dataset(hindcast_filename)[variable]
    
    return forecast_da, hindcast_da


def catchment_mean(da,w,spatial_dims):
    """
    Compute catchment-weighted spatial mean precipitation.                        
    Formula:                                                                                                                                                  
        sum(precip * weight) / sum(weight)
    Only finite precipitation values and positive finite weights are used.
    """

    valid = xr.ufuncs.isfinite(da) & xr.ufuncs.isfinite(w) & (w > 0)

    weighted_sum = (da.where(valid) * w.where(valid)).sum(dim=spatial_dims,skipna=True)
    weight_sum   = w.where(valid).sum(dim=spatial_dims,skipna=True)
    ts           = weighted_sum / weight_sum

    ts.attrs["description"] = "Catchment-weighted daily mean precipitation"
    ts.attrs["units"] = da.attrs.get("units", "")

    return ts


def xday_accumulation(da: xr.DataArray, x_days: int) -> xr.DataArray:
    """Compute trailing X-day accumulated precipitation."""

    out = (da.rolling(time=x_days, min_periods=x_days).sum().dropna("time", how="any"))

    # Standardized output variable name.
    out.name = "tp"
    out.attrs["description"] = (f"{x_days}-day accumulated catchment-weighted mean precipitation")
    out.attrs["units"] = "m"

    return out


def sort_maximum_into_monthly_bin(da):

    months = da["time"].dt.month
    counts = months.groupby(months).count()
    month_with_most = counts.idxmax()

    da = da.max(dim='time')

    print(da)
    
    return da #bin_counts



if __name__ == "__main__":

    forecast_dates = get_forecast_dates(forecast_date_range, option="mt")
    weights        = load_weights(filename_weights)

    for date in forecast_dates[0:1]:

        forecast_da, hindcast_da = load_model_data(path_in_forecast,path_in_hindcast,variable,date)
        forecast_da              = catchment_mean(forecast_da,weights,spatial_dims=("latitude", "longitude"))
        hindcast_da              = catchment_mean(hindcast_da,weights,spatial_dims=("latitude", "longitude"))
        forecast_da              = xday_accumulation(forecast_da,x_days)
        hindcast_da              = xday_accumulation(hindcast_da,x_days)

        
        bin_counts               = sort_maximum_into_monthly_bin(forecast_da)

        
