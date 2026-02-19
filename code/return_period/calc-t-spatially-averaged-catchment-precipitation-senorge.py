"""
Create a time series of X-day accumulated, catchment-weighted spatial mean precipitation
from SeNorge daily precipitation files (rr_{year}.nc).

Notes:
- SeNorge files are large, so we read year-by-year in a loop (no open_mfdataset).
- SeNorge grid is projected (Y, X) in meters; weights must be on the same (Y, X) grid.
- rr units are kg/m^2 for daily totals, which equals mm water equivalent.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Dunnsigouin_etal_2026 import config, misc

# input -------------------------------
variable        = "rr"
years           = np.arange(1957, 2024, 1)
dataset         = "senorge"
x_days          = 3

# SeNorge input: yearly files rr_{year}.nc
path_in         = config.dirs["senorge_continuous_daily"] + 'rr/'  # folder containing rr_YYYY.nc
file_pattern    = "rr_{year}.nc"

catchment       = "regine_drammen"
path_in_weights = config.dirs["nve_catchment"] + f"weights_catchment_{catchment}_senorge.nc"  # adjust if needed

path_out        = config.dirs["senorge_processed"]  # keep your existing output dir convention
write2file      = True
# -------------------------------------


def load_weights(path_in_weights: str) -> xr.DataArray:
    """Load catchment weights on the SeNorge dataset (Y, X)."""
    wds = xr.open_dataset(path_in_weights)

    if "catchment_weight" not in wds:
        raise KeyError(f"'catchment_weight' not found in {path_in_weights}. Available: {list(wds.data_vars)}")

    w = wds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"

    # Ensure expected dims exist
    if not (("Y" in w.dims) and ("X" in w.dims)):
        raise ValueError(f"Expected weights dims ('Y','X'), got {w.dims}")

    return w


def load_senorge_year(path_in: str, year: int, variable: str, file_pattern: str) -> xr.DataArray:
    """Load one SeNorge yearly file and return rr(time, Y, X) with fill values masked."""
    fn = path_in + file_pattern.format(year=year)
    ds = xr.open_dataset(fn)

    if variable not in ds:
        raise KeyError(f"'{variable}' not found in {fn}. Available: {list(ds.data_vars)}")

    # Decode CF time (hours since 1900-01-01)
    ds = xr.decode_cf(ds)

    da = ds[variable]

    # Mask common fill values
    fv = da.attrs.get("_FillValue", None)
    if fv is not None:
        da = da.where(da != fv)

    # SeNorge rr is usually kg/m^2 == mm
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"kg/m^2", "kg/m2", "kg m-2"}:
        da.attrs["units"] = "mm"

    # Ensure expected dims
    if not (("time" in da.dims) and ("Y" in da.dims) and ("X" in da.dims)):
        raise ValueError(f"Expected dims ('time','Y','X') for {variable}, got {da.dims}")

    # Close dataset handle; DataArray keeps a reference to the file lazily.
    # We will .load() only the reduced time series later (cheap).
    ds.close()

    return da


def catchment_weighted_mean_timeseries(rr_da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """
    Weighted spatial mean time series:
      sum(w * rr) / sum(w), NaN-safe.

    rr_da: (time, Y, X)
    w:     (Y, X)
    """
    # Align weights to rr dataset if coords match
    w_aligned = w
    try:
        w_aligned = w_aligned.reindex_like(rr_da.isel(time=0, drop=True), method=None)
    except Exception:
        # If coords aren't identical, fall back to broadcasting by dimension only
        w_aligned = w.broadcast_like(rr_da.isel(time=0, drop=True))

    valid = xr.ufuncs.isfinite(rr_da) & xr.ufuncs.isfinite(w_aligned) & (w_aligned > 0)

    rr_masked = rr_da.where(valid)
    w_masked  = w_aligned.where(valid)

    num = (rr_masked * w_masked).sum(dim=("Y", "X"), skipna=True)
    den = w_masked.sum(dim=("Y", "X"), skipna=True)

    ts = (num / den).load()  # load small 1D result into memory
    ts.name = "rr_catchment_mean"
    ts.attrs["description"] = "Catchment-weighted spatial mean of daily accumulated precipitation"
    ts.attrs["units"] = rr_da.attrs.get("units", "")

    return ts


def xday_accum_timeseries(ts: xr.DataArray, x_days: int, keep_full_windows_only: bool = True) -> xr.DataArray:
    """Trailing X-day rolling sum along time."""
    minp = x_days if keep_full_windows_only else 1
    out = ts.rolling(time=x_days, min_periods=minp).sum()

    if keep_full_windows_only:
        out = out.dropna("time", how="any")

    in_units = ts.attrs.get("units", "")
    # Daily totals in mm -> accumulated also mm
    out_units = "mm" if in_units.strip().lower() in {"kg/m^2", "kg/m2", "kg m-2", "mm"} else in_units

    out.name = f"{ts.name}_{x_days}dayacc"
    out.attrs["description"] = f"{x_days}-day accumulated catchment-weighted mean precipitation"
    out.attrs["units"] = out_units

    return out


def plot_timeseries(ts: xr.DataArray, title: str = None):
    """Simple line plot of a 1D time series (must contain 'time' dimension)."""
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

    # Load catchment weights once
    w = load_weights(path_in_weights)  # (Y, X)

    # Build daily catchment mean time series year-by-year
    ts_list = []
    for year in years:
        print(year)
        rr      = load_senorge_year(path_in, int(year), variable, file_pattern)  # (time, Y, X)
        ts_year = catchment_weighted_mean_timeseries(rr, w)                # (time,)
        ts_list.append(ts_year)

    ts_daily_mean      = xr.concat(ts_list, dim="time").sortby("time")
    ts_daily_mean.name = "rr_daily_catchment_mean"

    # X-day accumulation
    ts_xday_acc = xday_accum_timeseries(ts_daily_mean, x_days=x_days)

    out = xr.Dataset(
        {
            "rr_daily_catchment_mean": ts_daily_mean,
            f"rr_{x_days}day_catchment_acc": ts_xday_acc,
        }
    )

    plot_timeseries(ts_xday_acc, title=f"{x_days}-day accumulated catchment mean precipitation (SeNorge)")

    if write2file:
        filename_out = f"{path_out}t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
        out.to_netcdf(filename_out)
        print("Wrote:", filename_out)
