"""
Read annual maxima per grid point, fit stationary GEV per grid point (chunked + parallel),
compute return period map for an event magnitude (x-day accumulation), and write to NetCDF.

Inputs:
  annual_max(year, Y, X) NetCDF created by the annual-max script.

Outputs:
  return_period_years(Y, X) + fitted GEV params + event_accum
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt  # kept to match your import set (unused)
import cartopy.crs as ccrs       # kept to match your import set (unused)
import cartopy.feature as cfeature  # kept to match your import set (unused)
from scipy.stats import genextreme
import matplotlib.colors as mcolors  # kept to match your import set (unused)
import geopandas as gpd              # kept to match your import set (unused)
from shapely.geometry import Polygon, MultiPolygon  # kept to match your import set (unused)
from pyproj import Transformer       # kept to match your import set (unused)

from dask.diagnostics import ProgressBar

from Dunnsigouin_etal_2026 import config

# ------------------------------- config
dataset          = "senorge"
variable         = "rr"
years            = np.arange(1957, 2024, 1)

x_days           = 2
event_date_str   = "2023-08-09"
event_sel_method = "nearest"

min_T            = 1.0
max_T            = 500.0

# Daily files (for computing event x-day accumulation)
path_in_daily    = config.dirs["senorge_continuous_daily"] + variable + "/"
file_pattern     = f"{variable}_{{year}}.nc"  # adjust if needed, e.g. "rr_{year}.nc"

# Annual maxima file (from Script 1)
path_in_annmax   = config.dirs["senorge_processed"]

# Output
path_out         = config.dirs["senorge_processed"]
write2file       = True

# Chunking (tune to your machine; year must be whole for GEV fit)
chunk_Y          = 200
chunk_X          = 200
# -------------------------------------


def _open_senorge_year(path_in: str, file_pattern: str, year: int, variable: str) -> xr.DataArray:
    """Open one SeNorge yearly file as DataArray, decode CF time."""
    fn = path_in + file_pattern.format(year=int(year))
    ds = xr.open_dataset(fn)
    try:
        ds = xr.decode_cf(ds)
        if variable not in ds:
            raise KeyError(f"'{variable}' not found in {fn}. Available: {list(ds.data_vars)}")
        da = ds[variable]

        # Units: kg/m^2 == mm (daily totals)
        units = str(da.attrs.get("units", "")).strip().lower()
        if units in {"kg/m^2", "kg/m2", "kg m-2"}:
            da.attrs["units"] = "mm"

        return da.astype(np.float32)
    finally:
        ds.close()


def get_event_field(
    event_date_str: str,
    *,
    path_in: str,
    file_pattern: str,
    variable: str,
    x_days: int,
    method: str = "nearest",
) -> xr.DataArray:
    """
    Compute x-day trailing accumulation for the event year (with prev-year tail)
    and select the event field (Y, X).
    """
    event_time = np.datetime64(event_date_str)
    event_year = int(str(event_time)[:4])

    da_year = _open_senorge_year(path_in, file_pattern, event_year, variable)

    if x_days > 1:
        prev_year = event_year - 1
        try:
            da_prev = _open_senorge_year(path_in, file_pattern, prev_year, variable)
            tail = da_prev.isel(time=slice(-(x_days - 1), None))
            da_ext = xr.concat([tail, da_year], dim="time")
        except FileNotFoundError:
            da_ext = da_year
    else:
        da_ext = da_year

    # trailing rolling sum, require full windows
    acc = da_ext.rolling(time=x_days, min_periods=x_days).sum()

    # keep full windows based on time-sample count (not NaNs in the data)
    ones = xr.DataArray(
        np.ones(da_ext.sizes["time"], dtype=np.int16),
        coords={"time": da_ext["time"]},
        dims=("time",),
    )
    full = ones.rolling(time=x_days, min_periods=1).sum() == x_days
    acc = acc.sel(time=full)

    out = acc.sel(time=event_time, method=method).squeeze(drop=True)
    out.name = "event_accum"
    out.attrs["requested_time"] = str(event_time)[:10]
    if "time" in out.coords:
        out.attrs["selected_time"] = str(np.datetime64(out["time"].values))[:10]
    out.attrs["x_days"] = int(x_days)
    out.attrs["units"] = da_year.attrs.get("units", "")
    return out


def _fit_gev_1d(x: np.ndarray) -> tuple[float, float, float]:
    """Fit stationary GEV to 1D annual maxima sample; return (c, loc, scale) or NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    try:
        c, loc, scale = genextreme.fit(x)
        return float(c), float(loc), float(scale)
    except Exception:
        return np.nan, np.nan, np.nan


def fit_gev_per_gridpoint(ann_max: xr.DataArray) -> xr.Dataset:
    """
    Fit stationary GEV per gridpoint using annual maxima.

    ann_max dims: (year, Y, X) -> outputs c, loc, scale on (Y, X).

    NOTE: If ann_max is chunked over Y/X with year unchunked, this runs in parallel with Dask.
    """
    c, loc, scale = xr.apply_ufunc(
        _fit_gev_1d,
        ann_max,
        input_core_dims=[["year"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return xr.Dataset({"c": c, "loc": loc, "scale": scale})


def return_period_map_from_event(
    event_field: xr.DataArray,
    gev_params: xr.Dataset,
    *,
    min_T: float = 1.0,
    max_T: float = 500.0,
) -> xr.DataArray:
    """Compute T = 1/(1-F(z)) at each gridpoint using fitted GEV CDF."""
    def _cdf(z, c, loc, scale):
        if not (np.isfinite(z) and np.isfinite(c) and np.isfinite(loc) and np.isfinite(scale) and scale > 0):
            return np.nan
        try:
            return float(genextreme.cdf(z, c=c, loc=loc, scale=scale))
        except Exception:
            return np.nan

    Fz = xr.apply_ufunc(
        _cdf,
        event_field,
        gev_params["c"],
        gev_params["loc"],
        gev_params["scale"],
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    eps = 1e-12
    Fz = Fz.clip(min=eps, max=1.0 - eps)

    T = 1.0 / (1.0 - Fz)
    T = T.clip(min=min_T, max=max_T)

    T.name = "return_period_years"
    T.attrs["description"] = "Equivalent return period (years) under annual-max GEV"
    T.attrs["units"] = "years"
    T.attrs["event_date_requested"] = event_field.attrs.get("requested_time", "")
    T.attrs["event_date_selected"] = event_field.attrs.get("selected_time", "")
    T.attrs["x_days"] = event_field.attrs.get("x_days", "")
    return T


if __name__ == "__main__":

    # ---- 1) Load annual maxima
    fn_annmax = f"{path_in_annmax}xy_annualmax_{variable}_{x_days}dayacc_{dataset}_{years[0]}-{years[-1]}.nc"
    ds_ann = xr.open_dataset(fn_annmax)

    if "annual_max" not in ds_ann:
        raise KeyError(f"'annual_max' not found in {fn_annmax}. Found: {list(ds_ann.data_vars)}")

    ann_max = ds_ann["annual_max"]

    # Make a land mask based on "enough" valid annual maxima in each cell
    min_years_for_fit = 5  # must match _fit_gev_1d threshold
    n_valid = xr.ufuncs.isfinite(ann_max).sum("year")
    land_mask = n_valid >= min_years_for_fit

    # Mask ann_max so ocean cells become all-NaN along 'year'
    ann_max = ann_max.where(land_mask)

    # ---- 2) Chunk for parallelism (year must be whole)
    # If your dims are lowercase, adjust keys accordingly.
    ann_max = ann_max.chunk({"year": -1, "Y": chunk_Y, "X": chunk_X})

    # ---- 3) Fit GEV per gridpoint with progress bar
    params_lazy = fit_gev_per_gridpoint(ann_max)
    with ProgressBar():
        params = params_lazy.compute()

    # ---- 4) Compute event field (x-day accumulation at event date)
    event_acc = get_event_field(
        event_date_str,
        path_in=path_in_daily,
        file_pattern=file_pattern,
        variable=variable,
        x_days=x_days,
        method=event_sel_method,
    )

    event_acc = event_acc.where(land_mask)
    
    # ---- 5) Return period map (cheap relative to fitting; keep lazy until write)
    T_map = return_period_map_from_event(event_acc, params, min_T=min_T, max_T=max_T)

    # ---- 6) Write outputs (with progress bar during write)
    ds_out = xr.Dataset(
        {
            "return_period_years": T_map,
            "gev_shape": params["c"],
            "gev_loc": params["loc"],
            "gev_scale": params["scale"],
            "event_accum": event_acc,
        }
    )

    if write2file:
        event_tag = (
            str(np.datetime64(event_acc["time"].values))[:10].replace("-", "")
            if "time" in event_acc.coords
            else "event"
        )
        fn_out = f"{path_out}returnperiod_{variable}_{x_days}dayacc_{dataset}_{years[0]}-{years[-1]}_{event_tag}.nc"

        with ProgressBar():
            ds_out.to_netcdf(fn_out)

        print("Wrote:", fn_out)
    else:
        print("write2file=False (nothing written).")
