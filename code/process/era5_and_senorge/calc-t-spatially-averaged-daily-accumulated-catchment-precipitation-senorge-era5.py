"""
Create an X-day accumulated, catchment-weighted mean precipitation time series
for either SeNorge or ERA5.

Output:
- One NetCDF variable named `tp`
- Units: mm
- `tp` is the X-day accumulated catchment-mean precipitation

Notes:
- SeNorge is read year-by-year because files are large.
- ERA5 is read with open_mfdataset because files are smaller.
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# User settings
# =============================================================================

dataset    = "senorge"          # "senorge" or "era5"
years      = np.arange(1957, 2024)
x_days     = 2
catchment  = "nevina_bergheim"
write2file = True

# ERA5-only settings
grid   = "0.5x0.5"
domain = "norway"


# =============================================================================
# Dataset-specific settings
# =============================================================================

DATASETS = {
    "senorge": {
        "variable": "rr",
        "path_in": config.dirs["senorge_continuous_daily"] + "rr/",
        "file_pattern": "rr_{year}.nc",
        "path_weights": lambda catchment: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_senorge.nc"
        ),
        "path_out": config.dirs["senorge_processed"],
        "spatial_dims": ("Y", "X"),
        "read_mode": "yearly",
    },

    "era5": {
        "variable": "tp24",
        "path_in": config.dirs["era5_continuous_daily"] + "tp24/",
        "file_pattern": "tp24_{grid}_{year}.nc",
        "path_weights": lambda catchment: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_era5_{grid}.nc"
        ),
        "path_out": config.dirs["era5_processed"],
        "spatial_dims": ("latitude", "longitude"),
        "read_mode": "mfdataset",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def standardize_precip_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    """
    Convert precipitation to daily mm where needed.

    SeNorge:
        rr is usually kg/m^2, equivalent to mm.

    ERA5:
        tp24 is assumed to be in m/day and is converted to mm/day.
    """

    units = str(da.attrs.get("units", "")).strip().lower()

    if variable == "tp24":
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif units in {"kg/m^2", "kg/m2", "kg m-2"}:
        da.attrs["units"] = "mm"

    return da


def check_dims(da: xr.DataArray, expected_dims: tuple[str, ...], name: str) -> None:
    """Raise a clear error if required dimensions are missing."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing expected dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def load_weights(path_weights: str, spatial_dims: tuple[str, str]) -> xr.DataArray:
    """Load catchment weights and check that they match the expected grid."""

    ds = xr.open_dataset(path_weights)

    if "catchment_weight" not in ds:
        raise KeyError(
            f"'catchment_weight' not found in {path_weights}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    w = ds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"

    check_dims(w, spatial_dims, "Catchment weights")

    return w


def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Drop unnecessary ERA5 ensemble dimension if present."""

    return ds.drop_vars("number", errors="ignore")


def load_senorge_year(cfg: dict, year: int) -> xr.DataArray:
    """Load one SeNorge yearly precipitation file."""

    variable = cfg["variable"]
    filename = cfg["path_in"] + cfg["file_pattern"].format(year=year)

    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]

    # Mask fill values if they have not already been handled by xarray.
    fill_value = da.attrs.get("_FillValue")
    if fill_value is not None:
        da = da.where(da != fill_value)

    da = standardize_precip_units(da, variable)
    check_dims(da, ("time", *cfg["spatial_dims"]), filename)

    ds.close()

    return da


def load_era5(cfg: dict, years: np.ndarray, grid: str, domain: str | None) -> xr.DataArray:
    """Load all ERA5 files at once."""

    variable = cfg["variable"]

    filenames = [
        cfg["path_in"] + cfg["file_pattern"].format(grid=grid, year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    # Optional geographical subsetting.
    if domain is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(domain)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in ERA5 files. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = standardize_precip_units(ds[variable], variable)
    check_dims(da, ("time", *cfg["spatial_dims"]), "ERA5 precipitation")

    return da


def align_weights(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """
    Align weights to the precipitation grid.

    First tries coordinate-aware alignment. If that fails, it falls back to
    dimension-based broadcasting.
    """

    grid_template = da.isel(time=0, drop=True)

    try:
        return w.reindex_like(grid_template)
    except Exception:
        return w.broadcast_like(grid_template)


def catchment_mean(
    da: xr.DataArray,
    w: xr.DataArray,
    spatial_dims: tuple[str, str],
    load_result: bool = False,
) -> xr.DataArray:
    """
    Compute catchment-weighted spatial mean precipitation.

    Formula:
        sum(precip * weight) / sum(weight)

    Only finite precipitation values and positive finite weights are used.
    """

    w = align_weights(da, w)

    valid = xr.ufuncs.isfinite(da) & xr.ufuncs.isfinite(w) & (w > 0)

    weighted_sum = (da.where(valid) * w.where(valid)).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = w.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    ts = weighted_sum / weight_sum

    if load_result:
        ts = ts.load()

    ts.name = "daily_catchment_mean_precip"
    ts.attrs["description"] = "Catchment-weighted daily mean precipitation"
    ts.attrs["units"] = da.attrs.get("units", "")

    return ts


def xday_accumulation(ts: xr.DataArray, x_days: int) -> xr.DataArray:
    """Compute trailing X-day accumulated precipitation."""

    out = (
        ts
        .rolling(time=x_days, min_periods=x_days)
        .sum()
        .dropna("time", how="any")
    )

    # Standardized output variable name.
    out.name = "tp"
    out.attrs["description"] = (
        f"{x_days}-day accumulated catchment-weighted mean precipitation"
    )
    out.attrs["units"] = "mm"

    return out


def build_daily_catchment_mean(
    dataset: str,
    years: np.ndarray,
    catchment: str,
    grid: str | None = None,
    domain: str | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load precipitation and return the daily catchment-mean time series."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'senorge' or 'era5'.")

    cfg = DATASETS[dataset]
    spatial_dims = cfg["spatial_dims"]

    w = load_weights(cfg["path_weights"](catchment), spatial_dims)

    if cfg["read_mode"] == "yearly":
        # SeNorge: reduce each year to a 1D time series before loading the next.
        yearly_series = []

        for year in years:
            print(f"Processing {dataset} {year}")

            da = load_senorge_year(cfg, int(year))

            ts_year = catchment_mean(
                da=da,
                w=w,
                spatial_dims=spatial_dims,
                load_result=True,
            )

            yearly_series.append(ts_year)

        ts_daily = xr.concat(yearly_series, dim="time").sortby("time")

    elif cfg["read_mode"] == "mfdataset":
        # ERA5: load all files lazily as one combined dataset.
        da = load_era5(
            cfg=cfg,
            years=years,
            grid=grid,
            domain=domain,
        )

        ts_daily = catchment_mean(
            da=da,
            w=w,
            spatial_dims=spatial_dims,
            load_result=False,
        )

    else:
        raise ValueError(f"Unknown read mode: {cfg['read_mode']}")

    return ts_daily, cfg


def make_output_filename(
    cfg: dict,
    dataset: str,
    catchment: str,
    years: np.ndarray,
    x_days: int,
    grid: str | None = None,
) -> str:
    """Create standardized output filename."""

    if dataset == "era5":
        return (
            f"{cfg['path_out']}"
            f"t_tp_{x_days}dayacc_nve_catchment_"
            f"{catchment}_{dataset}_{grid}_{years[0]}-{years[-1]}.nc"
        )

    return (
        f"{cfg['path_out']}"
        f"t_tp_{x_days}dayacc_nve_catchment_"
        f"{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
    )


def write_output(
    tp: xr.DataArray,
    cfg: dict,
    dataset: str,
    catchment: str,
    years: np.ndarray,
    x_days: int,
    grid: str | None = None,
    write2file: bool = True,
) -> xr.Dataset:
    """Create output dataset and optionally write it to NetCDF."""

    out = xr.Dataset({"tp": tp})

    if write2file:
        filename_out = make_output_filename(
            cfg=cfg,
            dataset=dataset,
            catchment=catchment,
            years=years,
            x_days=x_days,
            grid=grid,
        )

        out.to_netcdf(filename_out)
        print("Wrote:", filename_out)

    return out


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    ts_daily, cfg = build_daily_catchment_mean(
        dataset=dataset,
        years=years,
        catchment=catchment,
        grid=grid,
        domain=domain if dataset == "era5" else None,
    )

    tp = xday_accumulation(ts_daily, x_days=x_days)

    out = write_output(
        tp=tp,
        cfg=cfg,
        dataset=dataset,
        catchment=catchment,
        years=years,
        x_days=x_days,
        grid=grid,
        write2file=write2file,
    )
