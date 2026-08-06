"""
Create an X-day accumulated, catchment-weighted mean time series
for SeNorge, regridded SeNorge, ERA5, or ERA5-Land.

Supported variables:
- SeNorge precipitation:          rr
- SeNorge runoff:                 gwb_q
- Regridded SeNorge precipitation: rr
- Regridded SeNorge runoff:        gwb_q
- ERA5 precipitation:             tp24
- ERA5 runoff:                    ro
- ERA5 surface runoff:            sro
- ERA5-Land runoff:               ro
- ERA5-Land surface runoff:       sro

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

dataset    = "era5"      # "senorge", "senorge_regrid", "era5", or "era5_land"
variable   = "tp24"                  # "rr", "gwb_q", "tp24", "ro", "sro"
years      = np.arange(1957, 2024)
x_days     = 1
catchment  = "regine_drammen"
write2file = True

# Optional spatial subset for ERA5 / ERA5-Land only.
# For senorge_regrid, the data are already a subset of the ERA5 grid,
# so weights are aligned directly to the data grid.
domain = "norway"


# =============================================================================
# Dataset and variable settings
# =============================================================================

VARIABLES = {
    "senorge": {
        "rr": {
            "path_in": config.dirs["senorge_continuous_daily"] + "rr/",
            "file_pattern": "rr_{year}.nc",
            "grid": None,
            "spatial_dims": ("Y", "X"),
            "output_name": "rr",
            "description": "precipitation",
        },
        "gwb_q": {
            "path_in": config.dirs["senorge_continuous_daily"] + "gwb_q/",
            "file_pattern": "gwb_q_{year}.nc",
            "grid": None,
            "spatial_dims": ("y", "x"),
            "output_name": "gwb_q",
            "description": "runoff",
        },
    },

    "senorge_regrid": {
        "rr": {
            "path_in": config.dirs["senorge_continuous_daily_regrid"] + "rr/",
            "file_pattern": "rr_regrid_0.5x0.5_{year}.nc",
            "grid": "0.5x0.5",
            "spatial_dims": ("lat", "lon"),
            "output_name": "rr",
            "description": "regridded precipitation",
        },
        "gwb_q": {
            "path_in": config.dirs["senorge_continuous_daily_regrid"] + "gwb_q/",
            "file_pattern": "gwb_q_regrid_0.5x0.5_{year}.nc",
            "grid": "0.5x0.5",
            "spatial_dims": ("lat", "lon"),
            "output_name": "gwb_q",
            "description": "regridded runoff",
        },
    },

    "era5": {
        "tp24": {
            "path_in": config.dirs["era5_continuous_daily"] + "tp24/",
            "file_pattern": "tp24_{grid}_{year}.nc",
            "grid": "0.5x0.5",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "tp24",
            "description": "precipitation",
        },
        "ro": {
            "path_in": config.dirs["era5_continuous_daily_scandinavia"] + "ro/",
            "file_pattern": "ro_{grid}_{year}.nc",
            "grid": "0.5x0.5",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "ro",
            "description": "runoff",
        },
        "sro": {
            "path_in": config.dirs["era5_continuous_daily_scandinavia"] + "sro/",
            "file_pattern": "sro_{grid}_{year}.nc",
            "grid": "0.5x0.5",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "sro",
            "description": "surface_runoff",
        },
    },

    "era5_land": {
        "ro": {
            "path_in": config.dirs["era5_land_continuous_daily_scandinavia"] + "ro/",
            "file_pattern": "ro_{grid}_{year}.nc",
            "grid": "0.1x0.1",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "ro",
            "description": "runoff",
        },
        "sro": {
            "path_in": config.dirs["era5_land_continuous_daily_scandinavia"] + "sro/",
            "file_pattern": "sro_{grid}_{year}.nc",
            "grid": "0.1x0.1",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "sro",
            "description": "surface_runoff",
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

    "senorge_regrid": {
        "read_mode": "yearly",
        "path_weights": lambda catchment, grid: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_era5_{grid}.nc"
        ),
        "path_out": config.dirs["senorge_regrid_processed"],
    },

    "era5": {
        "read_mode": "mfdataset",
        "path_weights": lambda catchment, grid: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_era5_{grid}.nc"
        ),
        "path_out": config.dirs["era5_processed"],
    },

    "era5_land": {
        "read_mode": "mfdataset",
        "path_weights": lambda catchment, grid: (
            config.dirs["nve"]
            + f"weights_catchment_{catchment}_era5_land_{grid}.nc"
        ),
        "path_out": config.dirs["era5_land_processed"],
    },
}


# =============================================================================
# Configuration helpers
# =============================================================================

def get_config(dataset: str, variable: str) -> dict:
    """Return combined dataset and variable configuration."""

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Use 'senorge', 'senorge_regrid', 'era5', or 'era5_land'."
        )

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


def make_input_filename(cfg: dict, year: int) -> str:
    """Create input filename for one year."""

    return cfg["path_in"] + cfg["file_pattern"].format(
        grid=cfg["grid"],
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
        tp24 is usually in m/day and is converted to mm/day.
        ro and sro are assumed to already be in mm/day in these files.
    """

    units = str(da.attrs.get("units", "")).strip().lower()

    if variable == "tp24":
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif variable == "rr":
        da.attrs["units"] = "mm/day"

    elif variable == "gwb_q":
        da.attrs["units"] = "mm/day"

    elif variable == "ro":
        da.attrs["units"] = "mm/day"

    elif variable == "sro":
        da.attrs["units"] = "mm/day"

    return da


# =============================================================================
# General helpers
# =============================================================================

def check_dims(
    da: xr.DataArray,
    expected_dims: tuple[str, ...],
    name: str,
) -> None:
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

def get_domain_latlon(domain: str):
    """Return latitude and longitude slices for predefined domains."""

    if domain == "norway":
        domain_lats = slice(72, 57)
        domain_lons = slice(4, 32)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return domain_lats, domain_lons


def load_weights(
    path_weights: str,
    spatial_dims: tuple[str, str],
    domain: str | None = None,
) -> xr.DataArray:
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

    if "latitude" in w.dims and "lat" in spatial_dims:
        rename_dims["latitude"] = "lat"

    if "longitude" in w.dims and "lon" in spatial_dims:
        rename_dims["longitude"] = "lon"
        
    if rename_dims:
        w = w.rename(rename_dims)

    if "y" in w.dims:
        w = w.sortby("y")

    if domain is not None and {"latitude", "longitude"}.issubset(set(w.dims)):
        domain_lats, domain_lons = get_domain_latlon(domain)
        w = w.sel(latitude=domain_lats, longitude=domain_lons)

    check_dims(w, spatial_dims, "Catchment weights")

    return w


def align_weights(da: xr.DataArray, w: xr.DataArray) -> xr.DataArray:
    """
    Align weights to the data grid.

    This is especially important for senorge_regrid, because the data are
    a subset of the ERA5 0.5x0.5 grid, while the weight file is defined on
    the full ERA5 0.5x0.5 grid.
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

def load_yearly_data(cfg: dict, year: int) -> xr.DataArray:
    """Load one yearly file for SeNorge or regridded SeNorge."""

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
        f"{cfg['dataset']} {variable}",
    )

    ds.close()

    return da


def load_era5_like(
    cfg: dict,
    years: np.ndarray,
    domain: str | None = None,
) -> xr.DataArray:
    """Load all ERA5 or ERA5-Land files at once."""

    variable = cfg["variable"]

    filenames = [
        make_input_filename(cfg, int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if domain is not None:
        domain_lats, domain_lons = get_domain_latlon(domain)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in files. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = standardize_units(ds[variable], variable)

    check_dims(
        da,
        ("time", *cfg["spatial_dims"]),
        f"{cfg['dataset']} {variable}",
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
    domain: str | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load data and return the daily catchment-mean time series."""

    cfg = get_config(dataset, variable)

    spatial_dims = cfg["spatial_dims"]
    grid = cfg["grid"]

    path_weights = cfg["path_weights"](catchment, grid)

    use_domain_for_weights = domain if dataset in {"era5", "era5_land"} else None

    w = load_weights(
        path_weights=path_weights,
        spatial_dims=spatial_dims,
        domain=use_domain_for_weights,
    )

    if cfg["read_mode"] == "yearly":

        yearly_series = []

        for year in years:
            print(f"Processing {dataset}: {variable}, {year}")

            da = load_yearly_data(cfg, int(year))

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

        da = load_era5_like(
            cfg=cfg,
            years=years,
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
) -> str:
    """Create standardized output filename."""

    # Only include the grid in filenames for ERA5 datasets.
    if dataset in {"era5", "era5_land"}:
        return (
            f"{cfg['path_out']}"
            f"t_{variable}_{x_days}dayacc_"
            f"{catchment}_{dataset}_{cfg['grid']}_"
            f"{years[0]}-{years[-1]}.nc"
        )

    return (
        f"{cfg['path_out']}"
        f"t_{variable}_{x_days}dayacc_"
        f"{catchment}_{dataset}_"
        f"{years[0]}-{years[-1]}.nc"
    )


def write_output(
    da_out: xr.DataArray,
    cfg: dict,
    dataset: str,
    variable: str,
    catchment: str,
    years: np.ndarray,
    x_days: int,
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
        )

        out.to_netcdf(filename_out)
        print("Wrote:", filename_out)

    return out


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    cfg_preview = get_config(dataset, variable)

    use_domain = domain if dataset in {"era5", "era5_land"} else None

    ts_daily, cfg = build_daily_catchment_mean(
        dataset=dataset,
        variable=variable,
        years=years,
        catchment=catchment,
        domain=use_domain,
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
        write2file=write2file,
    )
