"""
Compute annual maxima of x-day trailing accumulated SeNorge precipitation (rr) per grid point,
year-by-year (memory-safe streaming), and write to NetCDF.

Key points:
- SeNorge files are large: do NOT build (time,Y,X) rolling cubes.
- Avoid operations that force loading the full year (e.g., da.astype(float32) on the full cube).
- Stream day-by-day: keep only a small x_days buffer + running sum + running max.
- To compute the x-day accumulation on Jan 1, prepend the last (x_days-1) days from the previous year.
  If previous year is missing, the first (x_days-1) windows are naturally invalid and never enter the max.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt  # kept to match your import set (unused)
import cartopy.crs as ccrs       # kept to match your import set (unused)
import cartopy.feature as cfeature  # kept to match your import set (unused)
from scipy.stats import genextreme  # kept to match your import set (unused)
import matplotlib.colors as mcolors  # kept to match your import set (unused)
import geopandas as gpd              # kept to match your import set (unused)
from shapely.geometry import Polygon, MultiPolygon  # kept to match your import set (unused)
from pyproj import Transformer       # kept to match your import set (unused)

from Dunnsigouin_etal_2026 import config

# ------------------------------- config
dataset      = "senorge"
variable     = "rr"
years        = np.arange(1957, 2024, 1)

x_days       = 3  # trailing window length

path_in      = config.dirs["senorge_continuous_daily"] + variable + "/"
file_pattern = f"{variable}_{{year}}.nc"  # adjust to "rr_{year}.nc" if needed

path_out     = config.dirs["senorge_processed"]
write2file   = True
# -------------------------------------


def build_annual_maxima_xday_streaming(
    *,
    years: np.ndarray,
    path_in: str,
    file_pattern: str,
    variable: str,
    x_days: int,
) -> xr.DataArray:
    """
    Memory-safe annual maxima of x-day trailing accumulation on a (Y, X) grid.

    Returns:
      annual_max(year, Y, X)
    """
    if not isinstance(x_days, int) or x_days < 1:
        raise ValueError("x_days must be an integer >= 1.")

    years_sorted = np.sort(np.asarray(years, dtype=int))

    ann_max_list: list[xr.DataArray] = []

    # Keep previous-year tail as a SMALL in-memory numpy array: (x_days-1, Y, X)
    prev_tail_np: np.ndarray | None = None

    for y in years_sorted:
        y = int(y)
        print(f"Loading {y} ...")

        fn = path_in + file_pattern.format(year=y)

        with xr.open_dataset(fn, decode_cf=True) as ds:
            if variable not in ds:
                raise KeyError(f"'{variable}' not found in {fn}. Available: {list(ds.data_vars)}")

            rr = ds[variable]  # lazy-backed (time, Y, X)

            # Units bookkeeping only (no compute)
            units = str(rr.attrs.get("units", "")).strip().lower()
            if units in {"kg/m^2", "kg/m2", "kg m-2"}:
                rr.attrs["units"] = "mm"

            if rr.sizes.get("time", 0) == 0:
                raise RuntimeError(f"{y}: '{variable}' has zero timesteps in {fn}.")

            template_2d = rr.isel(time=0, drop=True)  # (Y, X), lazy metadata

            running_sum = xr.zeros_like(template_2d, dtype=np.float32)
            running_max = xr.full_like(template_2d, -np.inf, dtype=np.float32)

            # Ring buffer holds only x_days 2D numpy arrays (float32)
            buf: list[np.ndarray] = []
            buf_size = 0

            # ---- seed the rolling window with previous-year tail (if available)
            if x_days > 1 and prev_tail_np is not None:
                for k in range(prev_tail_np.shape[0]):
                    day_np = prev_tail_np[k].astype(np.float32, copy=False)
                    buf.append(day_np)
                    running_sum = running_sum + xr.DataArray(day_np, coords=template_2d.coords, dims=template_2d.dims)
                    buf_size += 1

            # ---- stream through the current year, one day at a time
            ntime = rr.sizes["time"]
            for t in range(ntime):
                # Load ONE day (2D) into memory (this is the core memory-safe trick)
                day_np = rr.isel(time=t).values.astype(np.float32, copy=False)

                buf.append(day_np)
                running_sum = running_sum + xr.DataArray(day_np, coords=template_2d.coords, dims=template_2d.dims)
                buf_size += 1

                if buf_size > x_days:
                    oldest_np = buf.pop(0)
                    running_sum = running_sum - xr.DataArray(oldest_np, coords=template_2d.coords, dims=template_2d.dims)
                    buf_size -= 1

                if buf_size == x_days:
                    running_max = xr.ufuncs.maximum(running_max, running_sum)

            # Convert -inf (never updated) to NaN
            ann_max_y = running_max.where(np.isfinite(running_max) & (running_max > -np.inf))

            ann_max_y.name = "annual_max"
            ann_max_y.attrs["units"] = rr.attrs.get("units", "")
            ann_max_y.attrs["x_days"] = int(x_days)
            ann_max_y.attrs["description"] = f"Annual maximum of {x_days}-day trailing accumulation"

            ann_max_list.append(ann_max_y.expand_dims(year=[y]))

            # ---- update tail for next year (load only last x_days-1 days; small)
            if x_days > 1:
                prev_tail_np = rr.isel(time=slice(-(x_days - 1), None)).values
            else:
                prev_tail_np = None

    ann_max = xr.concat(ann_max_list, dim="year")
    ann_max.name = "annual_max"
    ann_max.attrs["description"] = f"Annual maxima of {x_days}-day trailing accumulation ({variable})"
    ann_max.attrs["units"] = ann_max_list[0].attrs.get("units", "") if ann_max_list else ""
    ann_max.attrs["x_days"] = int(x_days)
    return ann_max



if __name__ == "__main__":

    ann_max = build_annual_maxima_xday_streaming(
        years=years,
        path_in=path_in,
        file_pattern=file_pattern,
        variable=variable,
        x_days=x_days,
    )

    ds_out = xr.Dataset({"annual_max": ann_max})

    if write2file:
        fn_out = f"{path_out}xy_annualmax_{variable}_{x_days}dayacc_{dataset}_{years[0]}-{years[-1]}.nc"
        ds_out.to_netcdf(fn_out)
        print("Wrote:", fn_out)
