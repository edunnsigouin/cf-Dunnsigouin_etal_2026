"""
Create a time series of 2-day accumulated, catchment-weighted spatial mean precipitation.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Dunnsigouin_etal_2026 import config, misc

# input -------------------------------
variable        = "tp24"
years           = np.arange(1957, 2024, 1)
grid            = "0.5x0.5"  
domain          = "norway"   # optional
x_days          = 2
path_in         = config.dirs["era5_continuous_daily"] + variable + "/"
catchment       = 'regine_glomma'
path_in_weights = config.dirs["nve_catchment"] + "weights_catchment_" + catchment + "_era5_0.5x0.5.nc"
path_out        = config.dirs["era5_processed"]
write2file      = True
# -------------------------------------


def preprocess_func(ds):
    return ds.drop_vars("number", errors="ignore")


def load_data(variable, years, grid, path_in, domain=None):

    filenames = [f"{path_in}{variable}_{grid}_{year}.nc" for year in years]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_func,
        combine="by_coords"
    )

    # Optional trim for IO/performance
    if domain is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(domain)
        ds                       = ds.sel(latitude=domain_lats, longitude=domain_lons)
    
    # Convert tp24 to mm/day (ERA5 total precipitation is typically in meters)
    if variable == "tp24":
        ds[variable]                = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    return ds


def load_weights(path_in_weights):
    wds    = xr.open_dataset(path_in_weights)
    w      = wds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"
    return w


def catchment_weighted_mean_timeseries(tp_da: xr.DataArray, w: xr.DataArray):
    """
    Weighted spatial mean time series:
      sum(w * tp) / sum(w), NaN-safe.
    """

    valid = xr.ufuncs.isfinite(tp_da) & xr.ufuncs.isfinite(w) & (w > 0)

    tp_masked = tp_da.where(valid)
    w_masked  = w.where(valid)

    num = (tp_masked * w_masked).sum(dim=("latitude", "longitude"), skipna=True)
    den = w_masked.sum(dim=("latitude", "longitude"), skipna=True)

    ts = num / den
    ts.name = "tp_catchment_mean"
    ts.attrs["description"] = "Catchment-weighted spatial mean of daily accumulated precipitation"
    ts.attrs["units"] = tp_da.attrs.get("units", "")

    return ts


def xday_accum_timeseries(ts: xr.DataArray, x_days: int, keep_full_windows_only: bool = True):

    minp = x_days if keep_full_windows_only else 1
    out  = ts.rolling(time=x_days, min_periods=minp).sum()

    if keep_full_windows_only:
        out = out.dropna("time", how="any")

    in_units = ts.attrs.get("units", "")
    # If daily is mm/day, sum over days -> mm accumulated
    out_units = "mm" if in_units.strip().lower() == "mm/day" else in_units

    out.name                 = f"{ts.name}_{x_days}dayacc"
    out.attrs["description"] = f"{x_days}-day accumulated catchment-weighted mean precipitation"
    out.attrs["units"]       = out_units

    return out


def plot_timeseries(ts: xr.DataArray, title: str = None):
    """
    Simple line plot of a 1D time series (must contain 'time' dimension).
    """

    if "time" not in ts.dims:
        raise ValueError("Input must have 'time' dimension.")

    plt.figure(figsize=(10, 4))
    plt.plot(ts["time"].values, ts.values)

    plt.xlabel("Time")

    units = ts.attrs.get("units", "")
    ylabel = f"{ts.name} ({units})" if units else ts.name
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":

    ds = load_data(variable, years, grid, path_in, domain=domain)
    tp = ds[variable]  # (time, lat, lon)
    w  = load_weights(path_in_weights)  # (lat, lon)

    ts_daily_mean = catchment_weighted_mean_timeseries(tp, w)
    ts_2day_acc   = xday_accum_timeseries(ts_daily_mean, x_days=x_days)

    out = xr.Dataset(
        {
            "tp_daily_catchment_mean": ts_daily_mean,
            f"tp_{x_days}day_catchment_acc": ts_2day_acc,
        }
    )

    plot_timeseries(ts_2day_acc,title=f"{x_days}-day accumulated catchment mean precipitation")
    
    if write2file:
        filename_out = f"{path_out}t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_era5_{grid}_{years[0]}-{years[-1]}.nc"
        out.to_netcdf(filename_out)
