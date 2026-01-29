"""
calculates the xyt climatology of a given era5 variable as a function
of month and over all years
"""

import numpy               as np
import xarray              as xr
from Dunnsigouin_etal_2026 import config, misc

# input -------------------------------
variable   = 'tp24'
years      = np.arange(2020,2023,1)
grid       = '0.5x0.5'
domain     = 'norway'
path_in    = config.dirs['era5_continuous_daily'] + variable + '/'
path_out   = config.dirs['era5_processed']
write2file = True
# -------------------------------------

def preprocess_func(ds):
    return ds.drop_vars("number", errors="ignore")

def load_data(variable, years, grid, path_in):

    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    filenames                = [f'{path_in}{variable}_{grid}_{year}.nc' for year in years]
    ds                       = xr.open_mfdataset(filenames,preprocess=preprocess_func,combine="by_coords")
    ds                       = ds.sel(latitude=domain_lats,longitude=domain_lons).compute()*1000 # convert to mm/day
    
    return ds

def daily_std_by_month_and_year(da: xr.DataArray) -> xr.DataArray:
    """
    Compute daily standard deviation at each grid point:
      - month=1..12: std across all daily values in that calendar month (all years)
      - month=0:     std across all daily values (whole period)

    Returns: DataArray with dims (latitude, longitude, month)
    """
    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")

    # std for each calendar month (1..12)
    std_month = da.groupby("time.month").std("time", ddof=0)  # dims: (month, latitude, longitude)

    # std for whole year / whole period
    std_year = da.std("time", ddof=0)  # dims: (latitude, longitude)

    # add a "month" dimension with value 0 for the whole-year std
    std_year = std_year.expand_dims(month=[0])  # dims: (month, latitude, longitude)

    # concatenate month=0 with month=1..12
    out = xr.concat([std_year, std_month], dim="month")  # month coord becomes [0,1..12]

    # reorder to (latitude, longitude, month)
    out = out.transpose("latitude", "longitude", "month")

    # keep a helpful name
    out.name = f"{da.name}_daily_std"

    return out


if __name__ == "__main__":
    
    ds = load_data(variable, years, grid, path_in)

    da_std = daily_std_by_month_and_year(ds[variable])

    if write2file:
        filename_out = f'{path_out}xym_{variable}_climatology_{grid}_{years[0]}-{years[-1]}.nc'
        da_std.to_netcdf(filename_out)
