"""
Create an X-day accumulated, catchment-weighted mean time series
for either SeNorge or ERA5.

Supported variables:
- SeNorge precipitation: rr
- SeNorge runoff:        gwb_q
- ERA5 precipitation:    tp24
- ERA5 runoff:           ro

Output:
- One NetCDF variable named after the selected variable.
- Units: mm
- Values are X-day accumulated catchment-mean values.
"""

import numpy as np
import xarray as xr
from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# User settings
# =============================================================================

dataset    = "senorge"          # "senorge" or "era5"
variable   = "gwb_q"            # "rr", "gwb_q", "tp24", or "ro"
years      = np.arange(1958, 2024)
x_days     = 1
catchment  = "regine_drammen"
write2file = True

# ERA5-only settings
grid   = "0.25x0.25"
domain = "norway"


# =============================================================================
# Dataset and variable settings
# =============================================================================

VARIABLES = {
    "senorge": {
        "rr": {
            "path_in": config.dirs["senorge_continuous_daily"] + "rr/",
            "file_pattern": "rr_{year}.nc",
            "spatial_dims": ("Y", "X"),
            "output_name": "rr",
            "description": "precipitation",
        },
        "gwb_q": {
            "path_in": config.dirs["senorge_continuous_daily"] + "gwb_q/",
            "file_pattern": "gwb_q_{year}.nc",
            "spatial_dims": ("y", "x"),
            "output_name": "gwb_q",
            "description": "runoff",
        },
    },

    "era5": {
        "tp24": {
            "path_in": config.dirs["era5_continuous_daily"] + "tp24/",
            "file_pattern": "tp24_{grid}_{year}.nc",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "tp24",
            "description": "precipitation",
        },
        "ro": {
            "path_in": config.dirs["era5_continuous_daily_scandinavia"] + "ro/",
            "file_pattern": "ro_{grid}_{year}.nc",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "ro",
            "description": "runoff",
        },
    },
}


DATASETS = {
    "senorge": {
        "read_mode": "yearly",
        "path_weights": lambda catchment, grid=None: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_senorge.nc"
        ),
        "path_out": config.dirs["senorge_processed"],
    },

    "era5": {
        "read_mode": "mfdataset",
        "path_weights": lambda catchment, grid: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_era5_{grid}.nc"
        ),
        "path_out": config.dirs["era5_processed"],
    },
}


# =============================================================================
# Configuration helpers
# =============================================================================

def get_config(dataset: str, variable: str) -> dict:
    """Return combined dataset and variable configuration."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'senorge' or 'era5'.")

    if variable not in VARIABLES[dataset]:
        valid = ", ".join(VARIABLES[dataset])
        raise ValueError(
            f"Variable '{variable}' is not available for dataset '{dataset}'. "
            f"Valid options are: {valid}."
        )

    cfg = {}
    cfg.update(DATASETS[dataset])
    cfg.update(VARIABLES[dataset][variable])
    cfg["dataset"] = dataset
    cfg["variable"] = variable

    return cfg


def make_input_filename(cfg: dict, year: int, grid: str | None = None) -> str:
    """Create input filename for one year."""

    if cfg["dataset"] == "era5":
        return cfg["path_in"] + cfg["file_pattern"].format(
            grid=grid,
            year=int(year),
        )

    return cfg["path_in"] + cfg["file_pattern"].format(
        year=int(year),
    )


# =============================================================================
# Unit handling
# =============================================================================

def standardize_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    """
    Convert daily variable to mm/day where needed.

    SeNorge:
        rr is usually kg/m2, equivalent to mm.
        gwb_q is already mm/day.

    ERA5:
        tp24 and ro are usually in m/day and are converted to mm/day.
    """

    units = str(da.attrs.get("units", "")).strip().lower()

    if variable in {"tp24", "ro"}:
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif variable == "rr":
        if units in {"kg/m^2", "kg/m2", "kg m-2"}:
            da.attrs["units"] = "mm/day"
        else:
            da.attrs["units"] = "mm/day"

    elif variable == "gwb_q":
        da.attrs["units"] = "mm/day"

    return da


# =============================================================================
# General helpers
# =============================================================================

def check_dims(da: xr.DataArray, expected_dims: tuple[str, ...], name: str) -> None:
    """Raise a clear error if required dimensions are missing."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing expected dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Drop unnecessary ERA5 ensemble dimension if present."""

    return ds.drop_vars("number", errors="ignore")


# =============================================================================
# Weight loading and alignment
# =============================================================================

def load_weights(path_weights: str, spatial_dims: tuple[str, str]) -> xr.DataArray:
    """Load catchment weights and rename dimensions when needed."""

    ds = xr.open_dataset(path_weights)

    if "catchment_weight" not in ds:
        raise KeyError(
            f"'catchment_weight' not found in {path_weights}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    w = ds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"

    rename_dims = {}

    if "Y" in w.dims and "y" in spatial_dims:
        rename_dims["Y"] = "y"

    if "X" in w.dims and "x" in spatial_dims:
        rename_dims["X"] = "x"

    if rename_dims:
        w = w.rename(rename_dims)

    if "y" in w.dims:
        w = w.sortby("y")

    check_dims(w, spatial_dims, "Catchment weights")

    return w


def align_weights(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """
    Align weights to the data grid.

    First tries coordinate-aware alignment. If that produces no valid weights,
    it falls back to positional alignment.
    """

    grid_template = da.isel(time=0, drop=True)

    try:
        w_aligned = w.reindex_like(grid_template)

        if np.isfinite(w_aligned).sum().item() > 0:
            return w_aligned

    except Exception:
        pass

    if w.shape != grid_template.shape:
        raise ValueError(
            f"Weights shape {w.shape} does not match data grid shape "
            f"{grid_template.shape}."
        )

    return xr.DataArray(
        w.values,
        dims=grid_template.dims,
        coords=grid_template.coords,
        name=w.name,
    )


# =============================================================================
# Data loading
# =============================================================================

def load_senorge_year(cfg: dict, year: int) -> xr.DataArray:
    """Load one SeNorge yearly file."""

    variable = cfg["variable"]
    filename = make_input_filename(cfg, year)

    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]

    fill_value = da.attrs.get("_FillValue")
    if fill_value is not None:
        da = da.where(da != fill_value)

    da = standardize_units(da, variable)

    check_dims(
        da,
        ("time", *cfg["spatial_dims"]),
        f"SeNorge {variable}",
    )

    ds.close()

    return da


def load_era5(
    cfg: dict,
    years: np.ndarray,
    grid: str,
    domain: str | None,
) -> xr.DataArray:
    """Load all ERA5 files at once."""

    variable = cfg["variable"]

    filenames = [
        make_input_filename(cfg, int(year), grid=grid)
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if domain is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(domain)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in ERA5 files. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = standardize_units(ds[variable], variable)

    check_dims(
        da,
        ("time", *cfg["spatial_dims"]),
        f"ERA5 {variable}",
    )

    return da


# =============================================================================
# Catchment averaging and accumulation
# =============================================================================

def catchment_mean(
    da: xr.DataArray,
    w: xr.DataArray,
    spatial_dims: tuple[str, str],
    variable: str,
    load_result: bool = False,
) -> xr.DataArray:
    """
    Compute catchment-weighted spatial mean.

    Formula:
        sum(variable * weight) / sum(weight)
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

    ts.name = f"daily_catchment_mean_{variable}"
    ts.attrs["description"] = f"Catchment-weighted daily mean {variable}"
    ts.attrs["units"] = da.attrs.get("units", "mm/day")

    return ts


def xday_accumulation(
    ts: xr.DataArray,
    x_days: int,
    output_name: str,
    description: str,
) -> xr.DataArray:
    """Compute trailing X-day accumulated catchment-mean values."""

    out = (
        ts
        .rolling(time=x_days, min_periods=x_days)
        .sum()
        .dropna("time", how="any")
    )

    out.name = output_name
    out.attrs["description"] = (
        f"{x_days}-day accumulated catchment-weighted mean {description}"
    )
    out.attrs["units"] = "mm"

    return out


def build_daily_catchment_mean(
    dataset: str,
    variable: str,
    years: np.ndarray,
    catchment: str,
    grid: str | None = None,
    domain: str | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load data and return the daily catchment-mean time series."""

    cfg = get_config(dataset, variable)
    spatial_dims = cfg["spatial_dims"]

    path_weights = cfg["path_weights"](catchment, grid)
    w = load_weights(path_weights, spatial_dims)

    if cfg["read_mode"] == "yearly":
        yearly_series = []

        for year in years:
            print(f"Processing {dataset} {variable} {year}")

            da = load_senorge_year(cfg, int(year))
            
            ts_year = catchment_mean(
                da=da,
                w=w,
                spatial_dims=spatial_dims,
                variable=variable,
                load_result=True,
            )

            yearly_series.append(ts_year)

        ts_daily = xr.concat(yearly_series, dim="time").sortby("time")

    elif cfg["read_mode"] == "mfdataset":
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
            variable=variable,
            load_result=False,
        )

    else:
        raise ValueError(f"Unknown read mode: {cfg['read_mode']}")

    return ts_daily, cfg


# =============================================================================
# Output
# =============================================================================

def make_output_filename(
    cfg: dict,
    dataset: str,
    variable: str,
    catchment: str,
    years: np.ndarray,
    x_days: int,
    grid: str | None = None,
) -> str:
    """Create standardized output filename."""

    if dataset == "era5":
        return (
            f"{cfg['path_out']}"
            f"t_{variable}_{x_days}dayacc_"
            f"{catchment}_{dataset}_{grid}_{years[0]}-{years[-1]}.nc"
        )

    return (
        f"{cfg['path_out']}"
        f"t_{variable}_{x_days}dayacc_"
        f"{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
    )


def write_output(
    da_out: xr.DataArray,
    cfg: dict,
    dataset: str,
    variable: str,
    catchment: str,
    years: np.ndarray,
    x_days: int,
    grid: str | None = None,
    write2file: bool = True,
) -> xr.Dataset:
    """Create output dataset and optionally write it to NetCDF."""

    out = xr.Dataset({da_out.name: da_out})

    if write2file:
        filename_out = make_output_filename(
            cfg=cfg,
            dataset=dataset,
            variable=variable,
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
        variable=variable,
        years=years,
        catchment=catchment,
        grid=grid,
        domain=domain if dataset == "era5" else None,
    )

    da_acc = xday_accumulation(
        ts=ts_daily,
        x_days=x_days,
        output_name=cfg["output_name"],
        description=cfg["description"],
    )

    out = write_output(
        da_out=da_acc,
        cfg=cfg,
        dataset=dataset,
        variable=variable,
        catchment=catchment,
        years=years,
        x_days=x_days,
        grid=grid,
        write2file=write2file,
    )
