"""
Calculates the distribution of monthly precipitation extremes for a given catchment 
in senorge over all years. 
Output has format (month of year,n_years)
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config, misc

# input -------------------------------
variable        = "tp"
years           = np.arange(1957, 2024, 1)
dataset         = "era5"
grid            = '0.5x0.5'
x_days          = 2
path_in         = config.dirs[f"{dataset}_processed"] 
catchment       = "regine_drammen"
path_out        = config.dirs[f"{dataset}_processed"]
write2file      = True
# -------------------------------------  


def load_data(path_in,variable,x_days,catchment,dataset,years):

    if dataset == 'senorge':
        filename_in = f"{path_in}t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
    elif dataset == 'era5':
        filename_in = f"{path_in}t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_{grid}_{years[0]}-{years[-1]}.nc"

    da = xr.open_dataset(filename_in)[f'{variable}_{x_days}day_catchment_acc']

    return da


def calc_monthly_maximum_distribution(da):

    monthly_max        = da.resample(time="1MS").max()
    annual_monthly_max = monthly_max.groupby("time.year").max()

    annual_monthly_max = (
        monthly_max
        .assign_coords(
            year=monthly_max.time.dt.year,
            month=monthly_max.time.dt.month
        )
        .set_index(time=["year", "month"])
        .unstack("time")
    )
    return annual_monthly_max


if __name__ == "__main__":

    da                 = load_data(path_in,variable,x_days,catchment,dataset,years)
    annual_monthly_max = calc_monthly_maximum_distribution(da)
    
    if write2file:
        filename_out = f"{path_out}distribution_monthly_extremes_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
        annual_monthly_max.to_netcdf(filename_out)
        print("Wrote:", filename_out)

