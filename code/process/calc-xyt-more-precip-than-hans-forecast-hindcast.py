"""
Calculates an x,y,doy file with the number of ensemble members in all forecast/hindcasts
with x-day accumulated precipitation > storm hans.

The key idea is to look at how the frequency of hans-like events varies by gridpoint so that
we can understand why the retrun period of hans is so dependent on the xy box size and location
in the unseen analysis. 
"""

import numpy               as np
import xarray              as xr
from datetime              import datetime
from Dunnsigouin_etal_2026 import config, misc

# input ----------------------------------------
variable         = 'tp24'
date_hans        = '2023-08-07'
acc_days         = 2
domain           = 'norway'
path_in_era5     = config.dirs['era5_continuous_daily'] + variable + '/'
path_in_forecast = config.dirs['s2s_forecast_daily'] + variable + '/'
path_in_hindcast = config.dirs['s2s_hindcast_daily'] + variable + '/'
write2file       = False
# ----------------------------------------------

def calc_accumulation(ds_era5, acc_days, dim="time"):
    """
    Rolling N-step accumulation along `dim` (default: time).

    Returns only *complete* accumulations (drops the first acc_days-1 steps).
    """
    if acc_days < 1:
        raise ValueError("acc_days must be >= 1")

    ds = ds_era5

    # If `dim` is a scalar coord (like in your example), make it a real dimension
    if dim not in ds.dims:
        if dim in ds.coords and ds[dim].ndim == 0:
            ds = ds.expand_dims({dim: [ds[dim].values]})
        else:
            raise ValueError(f"'{dim}' is neither a dimension nor a scalar coordinate in the dataset.")

    # Rolling sum across the dimension
    out = ds.rolling({dim: acc_days}, min_periods=acc_days).sum()

    # Keep only complete windows (optional but usually what "accumulation" means)
    out = out.dropna(dim=dim, how="all")

    return out


def read_hans_precip_from_era5_data(variable,date_hans,path_in_era5,acc_days,domain):
    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    year_hans                = datetime.strptime(date_hans, "%Y-%m-%d").year
    filename                 = path_in_era5 + f'{variable}_0.5x0.5_{year_hans}.nc'
    ds_era5                  = xr.open_dataset(filename).sel(latitude=domain_lats,longitude=domain_lons)
    ds_era5                  = calc_accumulation(ds_era5,acc_days,dim='time')
    return ds_era5.sel(time=date_hans)


if __name__ == "__main__":

    ds_era5 = read_hans_precip_from_era5_data(variable,date_hans,path_in_era5,acc_days,domain)

    print(ds_era5)
