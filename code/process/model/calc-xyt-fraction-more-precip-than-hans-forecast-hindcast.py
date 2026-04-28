"""
Calculates an x,y,month file with the number of ensemble members in all forecast/hindcasts
with x-day accumulated precipitation > storm hans.

The key idea is to look at how the frequency of hans-like events varies by gridpoint so that
we can understand why the return period of hans is so dependent on the xy box size and location
in the unseen analysis. 
"""

import numpy               as np
import xarray              as xr
import pandas              as pd
from datetime              import datetime
from Dunnsigouin_etal_2026 import config, misc
from matplotlib            import pyplot as plt

# input ----------------------------------------
variable            = 'tp24'
date_hans           = '2023-08-07'
acc_days            = 2
domain              = 'norway'
forecast_date_range = ['2020-01-02','2023-06-26']
path_in_era5        = config.dirs['era5_continuous_daily'] + variable + '/'
path_in_forecast    = config.dirs['s2s_forecast_daily'] + variable + '/'
path_in_hindcast    = config.dirs['s2s_hindcast_daily'] + variable + '/'
path_out            = config.dirs['s2s_processed']
write2file          = True
# ----------------------------------------------

def accumulate_along_dim(ds, acc_days, dim="time"):
    """
    Rolling N-step accumulation along `dim` (default: time).
    Returns only *complete* accumulations (drops the first acc_days-1 steps).
    """
    if acc_days < 1:
        raise ValueError("acc_days must be >= 1")

    if dim not in ds.dims:
        raise ValueError(f"'{dim}' is not a dimension of the dataset.")

    # Rolling sum across the dimension
    out = ds.rolling({dim: acc_days}, min_periods=acc_days).sum()

    # Keep only complete accumulation windows 
    out = out.dropna(dim=dim, how="all")

    return out


def read_accumulated_hans_era5_data(variable,date_hans,path_in_era5,acc_days,domain):
    
    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    year_hans                = datetime.strptime(date_hans, "%Y-%m-%d").year
    filename                 = path_in_era5 + f'{variable}_0.5x0.5_{year_hans}.nc'
    ds_era5                  = xr.open_dataset(filename).sel(latitude=domain_lats,longitude=domain_lons)
    ds_era5                  = accumulate_along_dim(ds_era5,acc_days,dim='time')

    obs = ds_era5[variable].sel(time=date_hans)
    
    return obs


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
        mondays = pd.date_range(start_date, end_date, freq="W-MON")
        thursdays = pd.date_range(start_date, end_date, freq="W-THU")
        forecast_dates = mondays.union(thursdays)

    else:
        forecast_dates = pd.date_range(start_date, end_date, freq="D")

    return forecast_dates.sort_values().strftime("%Y-%m-%d").tolist()


def read_accumulated_model_data(variable, date, path_in_model, acc_days, domain):
    """
    Read model data for one forecast date, subset to domain,
    accumulate along time, and return the variable as a DataArray.
    """

    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    filename                 = f"{path_in_model}{variable}_0.5x0.5_{date}.nc"

    with xr.open_dataset(filename) as ds:
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)
        ds = accumulate_along_dim(ds, acc_days, dim="time")
        out = ds[variable].load()

    return out


def count_model_events_greater_than_hans(obs,forecast_dates,variable,path_in_forecast,acc_days,domain):
    """
    loops through all forecasts and hindcasts in forecast_dates and counts, gridpointwise, how many days are greater
    than hans
    """

    # initialize count array
    obs   = obs.drop_vars({'time','number'})
    count = xr.zeros_like(obs, dtype="float32").rename("count")
    
    # loop through all forecasts & hindcasts
    for date in forecast_dates:

        print(f'reading forecast date: ',date)
        
        # read in data + accumulate in time
        forecast = read_accumulated_model_data(variable,date,path_in_forecast,acc_days,domain)
        hindcast = read_accumulated_model_data(variable,date,path_in_hindcast,acc_days,domain)
        
        # find maximum time for each ensemble-member, hdate and lat-lon gridpoint.   
        forecast = forecast.max(dim='time')
        hindcast = hindcast.max(dim='time')
        
        # Count members where model > obs, then sum over ensemble members
        total_count = np.float32((forecast['number'].size + hindcast['number'].size*hindcast['hdate'].size)*len(forecast_dates))
        count       = count + (forecast > obs).sum(dim="number")/total_count
        count       = count + (hindcast > obs).sum(dim={"number","hdate"})/total_count

    return count.rename("count")


def write_count_to_file(write2file,count,obs,date_hans,forecast_dates,acc_days,path_out):

    if not write2file:
        return None
    
    # Ensure obs has a time dimension + correct units
    obs                = obs.expand_dims(time=[pd.to_datetime(date_hans)])
    obs                = obs*1000 # m/day to mm/day
    obs.attrs['units'] = 'mm/day'
    
    # Combine into single Dataset
    ds_out = xr.Dataset({"count": count,"obs": obs})

    filename = f'{path_out}xyt_model_events_with_more_{acc_days}_day_accumulated_precip_than_hans_{forecast_dates[0]}-{forecast_dates[-1]}.nc'
    ds_out.to_netcdf(filename)
        


if __name__ == "__main__":

    obs            = read_accumulated_hans_era5_data(variable,date_hans,path_in_era5,acc_days,domain)
    forecast_dates = get_forecast_dates(forecast_date_range, option="mt")
    count          = count_model_events_greater_than_hans(obs,forecast_dates,variable,path_in_forecast,acc_days,domain)
    write_count_to_file(write2file,count,obs,date_hans,forecast_dates,acc_days,path_out)

