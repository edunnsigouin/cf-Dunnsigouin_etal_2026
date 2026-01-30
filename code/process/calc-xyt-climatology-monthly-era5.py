"""
calculates the xyt climatology of a given era5 variable as a function
of month and over all years
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config, misc

# input -------------------------------
variable   = 'tp24'
years      = np.arange(1941, 2023, 1)
grid       = '0.25x0.25'
domain     = 'norway'
x_days     = 2        
path_in    = config.dirs['era5_continuous_daily'] + variable + '/'
path_out   = config.dirs['era5_processed']
write2file = True
# -------------------------------------

def preprocess_func(ds):
    return ds.drop_vars("number", errors="ignore")

def load_data(variable, years, grid, path_in):

    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    filenames                = [f'{path_in}{variable}_{grid}_{year}.nc' for year in years]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_func,
        combine="by_coords"
    )
    ds = ds.sel(latitude=domain_lats, longitude=domain_lons).compute()

    # Convert tp24 to mm/day (ERA5 total precipitation is typically in meters)
    if variable == "tp24":
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    return ds

def xday_accum_then_clim_mean_std(da: xr.DataArray,x_days: int,*,keep_full_windows_only: bool = True) -> xr.Dataset:
    """
    1) Compute X-day accumulated series (rolling sum) => units become (input units) * day,
       e.g. mm/day summed over X days -> mm/X day (accumulated amount over X days).
    2) Compute climatological mean and std:
       - month=1..12: mean/std across all (rolling) daily values in that calendar month (all years)
       - month=0:     mean/std across all (rolling) daily values (whole period)

    Returns: xr.Dataset with variables:
      - mean (latitude, longitude, month)
      - std  (latitude, longitude, month)
    """
    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")
    if not isinstance(x_days, int) or x_days < 1:
        raise ValueError("x_days must be an integer >= 1.")

    # Rolling X-day accumulation (sum over last X days including current day)
    acc = da.rolling(time=x_days,min_periods=x_days if keep_full_windows_only else 1).sum()

    # If we required full windows, drop the initial NaNs
    if keep_full_windows_only:
        acc = acc.dropna("time", how="any")

    # Monthly mean/std (months 1..12)
    mean_month = acc.groupby("time.month").mean("time")
    std_month  = acc.groupby("time.month").std("time", ddof=0)

    # Whole-period mean/std (month 0)
    mean_year = acc.mean("time").expand_dims(month=[0])
    std_year  = acc.std("time", ddof=0).expand_dims(month=[0])

    # Concatenate month=0 with month=1..12
    mean_out = xr.concat([mean_year, mean_month], dim="month").transpose("latitude", "longitude", "month")
    std_out  = xr.concat([std_year,  std_month ], dim="month").transpose("latitude", "longitude", "month")

    # Units: convert "mm/day" -> f"mm/{x_days}day" for tp24 after accumulation
    in_units = da.attrs.get("units", None)
    if in_units is not None and in_units.strip().lower() == "mm/day":
        out_units = f"mm/{x_days}day"
    else:
        # generic fallback: keep original units but note it's an X-day accumulation
        out_units = in_units

    mean_out.attrs["units"] = out_units
    std_out.attrs["units"]  = out_units

    mean_out.name = "mean"
    std_out.name  = "std"

    mean_out.attrs["description"] = f"Climatological mean of {x_days}-day accumulated values"
    std_out.attrs["description"]  = f"Climatological std of {x_days}-day accumulated values"

    return xr.Dataset({"mean": mean_out, "std": std_out})


if __name__ == "__main__":
    
    ds = load_data(variable, years, grid, path_in)

    clim = xday_accum_then_clim_mean_std(ds[variable], x_days=x_days)

    if write2file:
        filename_out = f"{path_out}xyt_climatology_{variable}_{x_days}dayacc_monthly_{grid}_{years[0]}-{years[-1]}.nc"
        clim.to_netcdf(filename_out)
