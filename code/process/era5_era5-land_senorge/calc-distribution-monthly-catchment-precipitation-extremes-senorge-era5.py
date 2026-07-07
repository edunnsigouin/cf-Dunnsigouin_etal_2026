"""
Calculate the distribution of monthly precipitation extremes for a catchment.

Input:
- Output files from the catchment-mean precipitation script
- File contains one variable: tp(time)

Output:
- Monthly maxima arranged as (year, month)
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config


# input -----------------------------------------------------------------
dataset    = "senorge"          # "senorge" or "era5"
variable   = "tp"
years      = np.arange(1957, 2024)
grid       = "0.5x0.5"       # ERA5 only
x_days     = 2
catchment  = "regine_glomma"
write2file = True

path_in  = config.dirs[f"{dataset}_processed"]
path_out = config.dirs[f"{dataset}_processed"]
# -----------------------------------------------------------------------


def make_input_filename(
    path_in: str,
    dataset: str,
    variable: str,
    x_days: int,
    catchment: str,
    years: np.ndarray,
    grid: str | None = None,
) -> str:
    """Create filename matching the standardized output from the previous script."""

    if dataset == "era5":
        return (
            f"{path_in}"
            f"t_{variable}_{x_days}dayacc_nve_catchment_"
            f"{catchment}_{dataset}_{grid}_{years[0]}-{years[-1]}.nc"
        )

    return (
        f"{path_in}"
        f"t_{variable}_{x_days}dayacc_nve_catchment_"
        f"{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
    )


def load_data(
    path_in: str,
    dataset: str,
    variable: str,
    x_days: int,
    catchment: str,
    years: np.ndarray,
    grid: str | None = None,
) -> xr.DataArray:
    """Load standardized catchment-mean accumulated precipitation."""

    filename_in = make_input_filename(
        path_in=path_in,
        dataset=dataset,
        variable=variable,
        x_days=x_days,
        catchment=catchment,
        years=years,
        grid=grid,
    )

    ds = xr.open_dataset(filename_in)

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename_in}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    return ds[variable]


def calc_monthly_maximum_distribution(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate monthly maxima for each year.

    Output dimensions:
        year, month
    """

    monthly_max = da.resample(time="1MS").max()

    annual_monthly_max = (
        monthly_max
        .assign_coords(
            year=monthly_max.time.dt.year,
            month=monthly_max.time.dt.month,
        )
        .set_index(time=["year", "month"])
        .unstack("time")
    )

    annual_monthly_max.name = da.name
    annual_monthly_max.attrs["description"] = (
        "Monthly maxima of X-day accumulated catchment-mean precipitation"
    )
    annual_monthly_max.attrs["units"] = da.attrs.get("units", "")

    return annual_monthly_max


def make_output_filename(
    path_out: str,
    dataset: str,
    variable: str,
    x_days: int,
    catchment: str,
    years: np.ndarray,
    grid: str | None = None,
) -> str:
    """Create output filename."""

    if dataset == "era5":
        return (
            f"{path_out}"
            f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
            f"nve_catchment_{catchment}_{dataset}_{grid}_"
            f"{years[0]}-{years[-1]}.nc"
        )

    return (
        f"{path_out}"
        f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
        f"nve_catchment_{catchment}_{dataset}_"
        f"{years[0]}-{years[-1]}.nc"
    )


def write_output(
    da: xr.DataArray,
    path_out: str,
    dataset: str,
    variable: str,
    x_days: int,
    catchment: str,
    years: np.ndarray,
    grid: str | None = None,
    write2file: bool = True,
) -> xr.Dataset:
    """Create output dataset and optionally write to NetCDF."""

    out = xr.Dataset({variable: da})

    if write2file:
        filename_out = make_output_filename(
            path_out=path_out,
            dataset=dataset,
            variable=variable,
            x_days=x_days,
            catchment=catchment,
            years=years,
            grid=grid,
        )

        out.to_netcdf(filename_out)
        print("Wrote:", filename_out)

    return out


if __name__ == "__main__":

    da = load_data(
        path_in=path_in,
        dataset=dataset,
        variable=variable,
        x_days=x_days,
        catchment=catchment,
        years=years,
        grid=grid,
    )

    monthly_extremes = calc_monthly_maximum_distribution(da)

    out = write_output(
        da=monthly_extremes,
        path_out=path_out,
        dataset=dataset,
        variable=variable,
        x_days=x_days,
        catchment=catchment,
        years=years,
        grid=grid,
        write2file=write2file,
    )
