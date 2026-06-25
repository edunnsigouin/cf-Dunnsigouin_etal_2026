"""
Create an X-day accumulated, catchment-weighted mean time series
for SeNorge, ERA5, or ERA5-Land.

Supported variables
-------------------
SeNorge:
    rr      precipitation
    gwb_q   runoff

ERA5:
    tp24    precipitation
    ro      runoff
    sro     surface runoff

ERA5-Land:
    ro      runoff
    sro     surface runoff

Main idea
---------
1. Load the requested dataset and variable.
2. Optionally subset the data to a named domain, for example Norway.
3. Load catchment weights. The weight grid may be larger than the data domain.
4. Subset and align the weights to the exact data grid.
5. Compute the catchment-weighted daily mean.
6. Accumulate the daily values over X days.
7. Optionally write the result to NetCDF.

Output
------
One NetCDF variable named after the selected variable.
Units are mm.
Values are X-day accumulated catchment-mean values.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# User settings
# =============================================================================

DATASET = "senorge"          # "senorge", "era5", or "era5_land"
VARIABLE = "gwb_q"               # "rr", "gwb_q", "tp24", "ro", or "sro"
YEARS = np.arange(2023, 2024)
X_DAYS = 1
CATCHMENT = "regine_drammen"
GRID = "0.25x0.25"
DOMAIN = "norway"

# Coordinate matching tolerance for regular lat/lon grids.
# 1e-4 degrees is about 10 m in latitude, and is generous enough for rounding
# differences while still being much smaller than a 0.1 degree grid cell.
LATLON_TOLERANCE = 1e-4

WRITE_TO_FILE = False

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
        "sro": {
            "path_in": config.dirs["era5_continuous_daily_scandinavia"] + "sro/",
            "file_pattern": "sro_{grid}_{year}.nc",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "sro",
            "description": "surface_runoff",
        },
    },
    "era5_land": {
        "ro": {
            "path_in": config.dirs["era5_land_continuous_daily_scandinavia"] + "ro/",
            "file_pattern": "ro_{grid}_{year}.nc",
            "spatial_dims": ("latitude", "longitude"),
            "output_name": "ro",
            "description": "runoff",
        },
        "sro": {
            "path_in": config.dirs["era5_land_continuous_daily_scandinavia"] + "sro/",
            "file_pattern": "sro_{grid}_{year}.nc",
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
    """Return one combined configuration dictionary."""

    if dataset not in DATASETS:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{dataset}'. Valid datasets are: {valid}.")

    if variable not in VARIABLES[dataset]:
        valid = ", ".join(VARIABLES[dataset])
        raise ValueError(
            f"Variable '{variable}' is not available for dataset '{dataset}'. "
            f"Valid variables are: {valid}."
        )

    cfg = {}
    cfg.update(DATASETS[dataset])
    cfg.update(VARIABLES[dataset][variable])
    cfg["dataset"] = dataset
    cfg["variable"] = variable

    return cfg


def make_input_filename(cfg: dict, year: int, grid: str | None = None) -> str:
    """Create the input filename for one year."""

    if cfg["dataset"] in {"era5", "era5_land"}:
        if grid is None:
            raise ValueError(f"grid must be provided for dataset '{cfg['dataset']}'.")
        return cfg["path_in"] + cfg["file_pattern"].format(grid=grid, year=int(year))

    return cfg["path_in"] + cfg["file_pattern"].format(year=int(year))


def make_weight_filename(cfg: dict, catchment: str, grid: str | None = None) -> str:
    """Create the catchment-weight filename."""

    if cfg["dataset"] in {"era5", "era5_land"} and grid is None:
        raise ValueError(f"grid must be provided for dataset '{cfg['dataset']}'.")

    return cfg["path_weights"](catchment, grid)


# =============================================================================
# Unit handling
# =============================================================================

def standardize_units(da: xr.DataArray, variable: str) -> xr.DataArray:
    """
    Convert daily variables to mm/day when needed.

    Notes
    -----
    - kg m-2 is equivalent to mm water depth.
    - ERA5 and ERA5-Land accumulated runoff variables are commonly stored in m.
    - If a runoff variable is already labelled as mm, it is not multiplied again.
    """

    units = str(da.attrs.get("units", "")).strip().lower()

    units_in_metres = units in {"m", "meter", "metre", "meters", "metres"}
    units_in_mm = units in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}
    units_as_water_mass = units in {"kg/m^2", "kg/m2", "kg m-2", "kg m**-2"}

    if variable in {"tp24", "ro", "sro"}:
        if units_in_metres or units == "":
            da = da * 1000.0
        elif not units_in_mm:
            print(
                f"Warning: variable '{variable}' has unrecognized units '{units}'. "
                "Leaving values unchanged, but setting output units to mm/day."
            )
        da.attrs["units"] = "mm/day"

    elif variable == "rr":
        # SeNorge precipitation is often kg m-2, which is equivalent to mm.
        if not (units_in_mm or units_as_water_mass or units == ""):
            print(
                f"Warning: variable 'rr' has unrecognized units '{units}'. "
                "Leaving values unchanged, but setting output units to mm/day."
            )
        da.attrs["units"] = "mm/day"

    elif variable == "gwb_q":
        da.attrs["units"] = "mm/day"

    return da


# =============================================================================
# General validation and reporting helpers
# =============================================================================

def check_dims(da: xr.DataArray, expected_dims: tuple[str, ...], name: str) -> None:
    """Raise a clear error if required dimensions are missing."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing expected dimensions {missing}. "
            f"Found dimensions: {da.dims}."
        )


def print_grid_summary(label: str, da: xr.DataArray, spatial_dims: tuple[str, str]) -> None:
    """Print a compact summary of the spatial grid."""

    ydim, xdim = spatial_dims
    print(f"{label} grid:")
    print(f"  shape: ({da.sizes[ydim]}, {da.sizes[xdim]})")

    for dim in spatial_dims:
        if dim in da.coords:
            coord = da[dim]
            print(
                f"  {dim}: {float(coord.min()):.6f} to "
                f"{float(coord.max()):.6f}, n={coord.size}"
            )
        else:
            print(f"  {dim}: no coordinate values found")


def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Drop unnecessary ERA5 ensemble dimension/coordinate if present."""

    return ds.drop_vars("number", errors="ignore")


# =============================================================================
# Domain and grid alignment helpers
# =============================================================================

def subset_latlon_domain(ds: xr.Dataset, domain: str | None) -> xr.Dataset:
    """
    Subset an ERA5/ERA5-Land dataset to a named lat/lon domain.

    The domain coordinates are supplied by misc.get_domain_latlon(domain).
    If domain is None, the full dataset grid is returned.
    """

    if domain is None:
        return ds

    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    return ds.sel(latitude=domain_lats, longitude=domain_lons)


def rename_weight_dims_if_needed(
    w: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    """
    Rename weight dimensions so they match the data dimensions.

    This mainly handles the SeNorge case where some files may use Y/X and
    others may use y/x.
    """

    rename_dims = {}

    if "Y" in w.dims and "y" in spatial_dims:
        rename_dims["Y"] = "y"
    if "X" in w.dims and "x" in spatial_dims:
        rename_dims["X"] = "x"
    if "y" in w.dims and "Y" in spatial_dims:
        rename_dims["y"] = "Y"
    if "x" in w.dims and "X" in spatial_dims:
        rename_dims["x"] = "X"

    if rename_dims:
        w = w.rename(rename_dims)

    return w


def sort_spatial_coordinates(
    da: xr.DataArray,
    spatial_dims: tuple[str, str],
) -> xr.DataArray:
    """
    Sort spatial coordinates where helpful.

    Sorting avoids confusing alignment behavior when one object is ascending and
    the other is descending. xarray can usually handle this, but sorting makes
    the later logic easier to reason about.
    """

    for dim in spatial_dims:
        if dim in da.coords:
            da = da.sortby(dim)

    return da


def load_weights(path_weights: str, spatial_dims: tuple[str, str]) -> xr.DataArray:
    """
    Load catchment weights.

    The loaded weights do not have to be on the same domain as the data. They
    only need to be on the same grid spacing and cover at least the data domain.
    The actual subsetting to the data grid happens later in align_weights_to_data_grid().
    """

    ds = xr.open_dataset(path_weights)

    if "catchment_weight" not in ds:
        raise KeyError(
            f"'catchment_weight' not found in {path_weights}. "
            f"Available variables: {list(ds.data_vars)}."
        )

    w = ds["catchment_weight"].astype("float32")
    w.name = "catchment_weight"

    w = rename_weight_dims_if_needed(w, spatial_dims)
    check_dims(w, spatial_dims, "Catchment weights")
    w = sort_spatial_coordinates(w, spatial_dims)

    return w


def align_weights_to_data_grid(
    da: xr.DataArray,
    w: xr.DataArray,
    spatial_dims: tuple[str, str],
    tolerance: float = LATLON_TOLERANCE,
) -> xr.DataArray:
    """
    Subset and align catchment weights to the exact data grid.

    Why this exists
    ---------------
    The catchment-weight files can be larger than the selected data domain.
    For example, weights may cover Scandinavia while data are subset to Norway.
    Before multiplying data by weights, both arrays must have exactly the same
    spatial shape and coordinates.

    Strategy
    --------
    1. Take one time slice from the data to define the target spatial grid.
    2. Select the same coordinates from the weight grid.
    3. Reindex to the data grid so dimension order and coordinates match.
    4. Check that the result has valid, non-empty weights.
    """

    grid_template = da.isel(time=0, drop=True)
    grid_template = grid_template.transpose(*spatial_dims)

    # Make sure weight dimensions are ordered like the data dimensions.
    w = w.transpose(*spatial_dims)

    # Lat/lon grids sometimes differ by tiny floating-point rounding errors.
    # For those grids, nearest-neighbor selection with a small tolerance is safer
    # than exact coordinate matching.
    if spatial_dims == ("latitude", "longitude"):
        try:
            w_on_data_grid = w.sel(
                latitude=grid_template.latitude,
                longitude=grid_template.longitude,
                method="nearest",
                tolerance=tolerance,
            )
        except KeyError as exc:
            raise ValueError(
                "Could not subset weights to the data latitude/longitude grid. "
                "This usually means the weight file does not cover the selected "
                "data domain, or the coordinate values differ by more than the "
                f"tolerance={tolerance}."
            ) from exc
    else:
        # For projected or index-like grids, first try exact coordinate-aware
        # subsetting. If coordinates are not available, xarray will fall back to
        # dimension-based behavior in the later shape check.
        selectors = {
            dim: grid_template[dim]
            for dim in spatial_dims
            if dim in w.coords and dim in grid_template.coords
        }

        if selectors:
            w_on_data_grid = w.sel(selectors)
        else:
            w_on_data_grid = w

    # Final coordinate-aware alignment to the data grid.
    w_on_data_grid = w_on_data_grid.reindex_like(grid_template)

    if w_on_data_grid.shape != grid_template.shape:
        raise ValueError(
            "Weights and data do not have the same spatial shape after subsetting.\n"
            f"  Weights shape: {w_on_data_grid.shape}\n"
            f"  Data shape:    {grid_template.shape}\n"
            "Check that the weight grid covers the selected data domain and "
            "uses the same spatial resolution."
        )

    valid_weight_count = np.isfinite(w_on_data_grid).sum().item()
    positive_weight_count = ((w_on_data_grid > 0) & np.isfinite(w_on_data_grid)).sum().item()

    if valid_weight_count == 0:
        raise ValueError(
            "Weights were subset to the data grid, but all values are NaN. "
            "This means the data and weight coordinates do not overlap."
        )

    if positive_weight_count == 0:
        raise ValueError(
            "Weights were subset to the data grid, but no positive catchment "
            "weights were found. Check that the selected catchment intersects "
            "the selected domain."
        )

    return w_on_data_grid


# =============================================================================
# Data loading
# =============================================================================

def load_senorge_year(cfg: dict, year: int) -> xr.DataArray:
    """Load one yearly SeNorge file."""

    variable = cfg["variable"]
    filename = make_input_filename(cfg, year)

    print(f"Opening {filename}")

    with xr.open_dataset(filename) as ds:
        ds = xr.decode_cf(ds)

        if variable not in ds:
            raise KeyError(
                f"'{variable}' not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}."
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

        da = sort_spatial_coordinates(da, cfg["spatial_dims"])
        da = da.load()

    return da


def load_era5_like_dataset(
    cfg: dict,
    years: np.ndarray,
    grid: str,
    domain: str | None,
) -> xr.DataArray:
    """
    Load ERA5 or ERA5-Land files and optionally subset to a domain.

    This function is used for both ERA5 and ERA5-Land because the file layout,
    coordinates, and processing steps are the same.
    """

    variable = cfg["variable"]

    filenames = [
        make_input_filename(cfg, int(year), grid=grid)
        for year in years
    ]

    print("Opening files:")
    for filename in filenames:
        print(f"  {filename}")

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    ds = subset_latlon_domain(ds, domain)

    if variable not in ds:
        raise KeyError(
            f"'{variable}' not found in files. "
            f"Available variables: {list(ds.data_vars)}."
        )

    da = standardize_units(ds[variable], variable)

    check_dims(
        da,
        ("time", *cfg["spatial_dims"]),
        f"{cfg['dataset']} {variable}",
    )

    da = sort_spatial_coordinates(da, cfg["spatial_dims"])

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

    Formula
    -------
    sum(variable * weight) / sum(weight)

    The weights are first subset and aligned to the exact data grid. This is the
    step that makes it safe for the original weight file to cover a larger domain
    than the selected data domain.
    """

    print_grid_summary("Data before weighting", da.isel(time=0, drop=True), spatial_dims)
    print_grid_summary("Weights before alignment", w, spatial_dims)

    w = align_weights_to_data_grid(da, w, spatial_dims)

    print_grid_summary("Weights after alignment", w, spatial_dims)

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

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

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


# =============================================================================
# Main processing function
# =============================================================================

def build_daily_catchment_mean(
    dataset: str,
    variable: str,
    years: np.ndarray,
    catchment: str,
    grid: str | None = None,
    domain: str | None = None,
) -> tuple[xr.DataArray, dict]:
    """
    Load data, load weights, align weights to data grid, and return daily means.
    """

    cfg = get_config(dataset, variable)
    spatial_dims = cfg["spatial_dims"]

    path_weights = make_weight_filename(cfg, catchment, grid)
    print(f"Opening weights: {path_weights}")
    weights = load_weights(path_weights, spatial_dims)

    if cfg["read_mode"] == "yearly":
        yearly_series = []

        for year in years:
            print(f"Processing {dataset} {variable} {year}")

            da_year = load_senorge_year(cfg, int(year))

            ts_year = catchment_mean(
                da=da_year,
                w=weights,
                spatial_dims=spatial_dims,
                variable=variable,
                load_result=True,
            )

            yearly_series.append(ts_year)

        ts_daily = xr.concat(yearly_series, dim="time").sortby("time")

    elif cfg["read_mode"] == "mfdataset":
        da = load_era5_like_dataset(
            cfg=cfg,
            years=years,
            grid=grid,
            domain=domain,
        )

        ts_daily = catchment_mean(
            da=da,
            w=weights,
            spatial_dims=spatial_dims,
            variable=variable,
            load_result=False,
        )

    else:
        raise ValueError(f"Unknown read mode: {cfg['read_mode']}.")

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

    if dataset in {"era5", "era5_land"}:
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
    write_to_file: bool = True,
) -> xr.Dataset:
    """Create output dataset and optionally write it to NetCDF."""

    out = xr.Dataset({da_out.name: da_out})

    if write_to_file:
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
        print(f"Wrote: {filename_out}")

    return out


# =============================================================================
# Script entry point
# =============================================================================

def main() -> xr.Dataset:
    """Run the full processing chain."""

    domain_for_processing = DOMAIN if DATASET in {"era5", "era5_land"} else None
    grid_for_processing = GRID if DATASET in {"era5", "era5_land"} else None

    ts_daily, cfg = build_daily_catchment_mean(
        dataset=DATASET,
        variable=VARIABLE,
        years=YEARS,
        catchment=CATCHMENT,
        grid=grid_for_processing,
        domain=domain_for_processing,
    )

    da_acc = xday_accumulation(
        ts=ts_daily,
        x_days=X_DAYS,
        output_name=cfg["output_name"],
        description=cfg["description"],
    )

    out = write_output(
        da_out=da_acc,
        cfg=cfg,
        dataset=DATASET,
        variable=VARIABLE,
        catchment=CATCHMENT,
        years=YEARS,
        x_days=X_DAYS,
        grid=grid_for_processing,
        write_to_file=WRITE_TO_FILE,
    )

    return out


if __name__ == "__main__":
    main()
