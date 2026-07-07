#!/usr/bin/env python3
"""
Regrid daily ERA5-Land and seNorge variables to a regular 0.5° grid.

Supported variables:
    ERA5-Land:
        sro      surface runoff

    seNorge:
        gwb_q    surface runoff
        rr       precipitation

Method:
    - Uses xESMF conservative_normed regridding.
    - Adds a binary source mask to the source grid.
    - mask = 1 means valid source cell.
    - mask = 0 means masked source cell.
    - ERA5-Land mask is based on the land-sea mask file.
    - seNorge mask is based directly on non-NaN values in the data file.

NOTE: you need to use the conda environment called "regrid" where xesmf is installed.
"""

import glob
import os

import numpy as np
import xarray as xr
import xesmf as xe
from pyproj import Transformer

from Dunnsigouin_etal_2026 import config


# ====================================================
# User settings
# ====================================================

DATASET = "senorge"          # "era5_land" or "senorge"
SENORGE_VARIABLE = "rr"      # "rr" or "gwb_q"

YEARS = np.arange(1957, 1958)

WRITE_TO_FILE = True

TEST_ONE_DAY = False
TEST_TIME_INDEX = 0


# ====================================================
# Target 0.5° grid
# ====================================================

TARGET_LATS = np.arange(73.5, 52.5, -0.5)
TARGET_LONS = np.arange(2.0, 33.0, 0.5)


# ====================================================
# Dataset configuration
# ====================================================

DATASET_CONFIGS = {
    "era5_land_sro": {
        "dataset": "era5_land",
        "variable": "sro",
        "path_in": os.path.join(
            config.dirs["era5_land_continuous_daily_scandinavia"],
            "sro",
        ),
        "path_out": os.path.join(
            config.dirs["era5_land_continuous_daily_scandinavia"],
            "sro",
        ),
        "file_pattern": "sro_0.1x0.1_{year}.nc",
        "grid_type": "regular_lat_lon",
        "units_out": "mm/day",
        "description": "ERA5-Land daily accumulated surface runoff",
        "mask_type": "era5_land_lsm",
        "mask_file": os.path.join(
            config.dirs["era5_land_continuous_daily_scandinavia"],
            "lsm",
            "lsm_0.1x0.1_scandinavia.nc",
        ),
        "mask_variables": ["lsm", "land_sea_mask"],
        "mask_threshold": 0.5,
    },
    "senorge_gwb_q": {
        "dataset": "senorge",
        "variable": "gwb_q",
        "path_in": os.path.join(
            config.dirs["senorge_continuous_daily"],
            "gwb_q",
        ),
        "path_out": os.path.join(
            config.dirs["senorge_continuous_daily_regrid"],
            "gwb_q",
        ),
        "file_pattern": "gwb_q_{year}.nc",
        "grid_type": "curvilinear_lat_lon",
        "units_out": "mm",
        "description": "seNorge daily surface runoff",
        "mask_type": "valid_data",
    },
    "senorge_rr": {
        "dataset": "senorge",
        "variable": "rr",
        "path_in": os.path.join(
            config.dirs["senorge_continuous_daily"],
            "rr",
        ),
        "path_out": os.path.join(
            config.dirs["senorge_continuous_daily_regrid"],
            "rr",
        ),
        "file_pattern": "rr_{year}.nc",
        "grid_type": "curvilinear_lat_lon",
        "units_out": "kg/m2",
        "description": "seNorge daily total precipitation",
        "mask_type": "valid_data",
    },
}


# ====================================================
# Configuration helpers
# ====================================================

def get_config_key():
    """Return the selected configuration key."""

    if DATASET == "era5_land":
        return "era5_land_sro"

    if DATASET == "senorge":
        return f"senorge_{SENORGE_VARIABLE}"

    raise ValueError(f"Unknown DATASET: {DATASET}")


def get_config():
    """Return the selected dataset configuration."""

    config_key = get_config_key()

    if config_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_key}")

    return DATASET_CONFIGS[config_key]


# ====================================================
# General helpers
# ====================================================

def normalise_units(units):
    """Return a simplified unit string for comparison."""

    if units is None:
        return ""

    return (
        units.strip()
        .lower()
        .replace(" ", "")
        .replace("−", "-")
        .replace("_", "")
    )


def standardise_coordinate_names(ds):
    """
    Standardise coordinate names.

    ERA5-Land may use latitude/longitude.
    seNorge files may use X/Y or x/y.
    """

    rename_dict = {}

    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if "X" in ds.dims:
        rename_dict["X"] = "x"

    if "Y" in ds.dims:
        rename_dict["Y"] = "y"

    if "X" in ds.coords:
        rename_dict["X"] = "x"

    if "Y" in ds.coords:
        rename_dict["Y"] = "y"

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def sort_coordinates(ds):
    """Sort latitude/y coordinates where needed."""

    if "lat" in ds.coords and ds["lat"].ndim == 1:
        if ds.lat[0] > ds.lat[-1]:
            ds = ds.sortby("lat")

    if "y" in ds.coords:
        if ds.y[0] > ds.y[-1]:
            ds = ds.sortby("y")

    return ds


def prepare_dataset(ds):
    """Prepare dataset coordinates before regridding."""

    ds = standardise_coordinate_names(ds)
    ds = sort_coordinates(ds)

    return ds


def find_input_file(path_in, file_pattern, year):
    """Find one yearly input file."""

    pattern = os.path.join(path_in, file_pattern.format(year=year))
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No file found:\n{pattern}")

    if len(files) > 1:
        raise ValueError(f"More than one file found:\n{files}")

    return files[0]


def get_first_existing_variable(ds, variable_names):
    """Return the first available variable from a list."""

    for variable_name in variable_names:
        if variable_name in ds:
            return ds[variable_name]

    raise ValueError(
        "None of these variables were found: "
        + ", ".join(variable_names)
    )


# ====================================================
# Variable preparation
# ====================================================

def convert_era5_sro_to_mm_per_day(da):
    """Ensure ERA5-Land surface runoff is in mm/day."""

    original_units = da.attrs.get("units", "")
    units_clean = normalise_units(original_units)

    if units_clean in ["mm", "mm/day", "mmday-1", "mmd-1", "mmperday"]:
        conversion_note = "No conversion applied; input already mm/day or mm."

    elif units_clean in ["m", "meter", "metre", "meters", "metres"]:
        da = da * 1000.0
        conversion_note = "Converted from m to mm/day using factor 1000."

    else:
        raise ValueError(f"Unknown ERA5-Land sro units: '{original_units}'")

    da.attrs["units"] = "mm/day"
    da.attrs["original_units"] = original_units
    da.attrs["conversion_note"] = conversion_note

    return da


def prepare_variable(da, cfg):
    """Apply variable-specific preparation."""

    if cfg["dataset"] == "era5_land" and cfg["variable"] == "sro":
        da = convert_era5_sro_to_mm_per_day(da)
    else:
        da.attrs["units"] = da.attrs.get("units", cfg["units_out"])

    return da


# ====================================================
# Grid creation
# ====================================================

def make_target_grid():
    """Create regular 0.5° target grid."""

    target_grid = xr.Dataset(
        coords={
            "lat": TARGET_LATS,
            "lon": TARGET_LONS,
        }
    )

    return sort_coordinates(target_grid)


def add_curvilinear_bounds(source_grid, ds):
    """
    Add lon/lat corner coordinates for a seNorge 2D curvilinear grid.

    seNorge is defined on a regular UTM Zone 33 grid. It is safer to compute
    cell corners in projected x/y coordinates and then transform those corners
    to lon/lat, rather than estimating corners directly from 2D lon/lat.
    """

    x = ds["x"].values
    y = ds["y"].values

    dx = np.median(np.diff(x))
    dy = np.median(np.diff(y))

    x_b = np.empty(x.size + 1)
    y_b = np.empty(y.size + 1)

    x_b[1:-1] = 0.5 * (x[:-1] + x[1:])
    y_b[1:-1] = 0.5 * (y[:-1] + y[1:])

    x_b[0] = x[0] - 0.5 * dx
    x_b[-1] = x[-1] + 0.5 * dx

    y_b[0] = y[0] - 0.5 * dy
    y_b[-1] = y[-1] + 0.5 * dy

    x_b_2d, y_b_2d = np.meshgrid(x_b, y_b)

    transformer = Transformer.from_crs(
        "EPSG:32633",   # UTM Zone 33N, WGS84
        "EPSG:4326",    # lon/lat, WGS84
        always_xy=True,
    )

    lon_b, lat_b = transformer.transform(x_b_2d, y_b_2d)

    source_grid["lon_b"] = (("y_b", "x_b"), lon_b)
    source_grid["lat_b"] = (("y_b", "x_b"), lat_b)

    return source_grid


def make_source_grid(ds, cfg):
    """Create the source grid expected by xESMF."""

    if cfg["grid_type"] == "regular_lat_lon":
        return xr.Dataset(
            coords={
                "lat": ds["lat"],
                "lon": ds["lon"],
            }
        )

    if cfg["grid_type"] == "curvilinear_lat_lon":
        source_grid = xr.Dataset(
            {
                "lat": ds["lat"],
                "lon": ds["lon"],
            }
        )

        return add_curvilinear_bounds(source_grid, ds)

    raise ValueError(f"Unknown grid type: {cfg['grid_type']}")


# ====================================================
# Mask creation
# ====================================================

def load_era5_land_mask(cfg, ds_source):
    """Load ERA5-Land land-sea mask and convert it to binary xESMF mask."""

    if not os.path.exists(cfg["mask_file"]):
        raise FileNotFoundError(f"Mask file not found:\n{cfg['mask_file']}")

    ds_mask = xr.open_dataset(cfg["mask_file"])
    ds_mask = prepare_dataset(ds_mask)

    mask_source = get_first_existing_variable(ds_mask, cfg["mask_variables"])

    for dim in ["time", "valid_time"]:
        if dim in mask_source.dims:
            mask_source = mask_source.isel({dim: 0}, drop=True)

    mask_source = mask_source.squeeze(drop=True)
    mask_source = mask_source.clip(min=0.0, max=1.0)

    mask_source = mask_source.sel(
        lat=ds_source.lat,
        lon=ds_source.lon,
    )

    mask = xr.where(
        mask_source >= cfg["mask_threshold"],
        1,
        0,
    ).astype("int32")

    mask.name = "mask"

    return mask


def make_valid_data_mask(da, cfg):
    """
    Create a binary mask from valid data values.

    General seNorge rule:
        non-NaN values = valid source cells
        NaN values     = masked source cells

    Special case for seNorge gwb_q:
        older files may use -0.1 for sea points, so these are also masked.
    """

    valid = da.notnull()

    if cfg["dataset"] == "senorge" and cfg["variable"] == "gwb_q":
        valid = valid & (da != -0.1)

    if "time" in da.dims:
        valid = valid.any(dim="time")

    mask = xr.where(valid, 1, 0).astype("int32")
    mask.name = "mask"

    return mask


def make_source_mask(cfg, ds_source, da):
    """Create source mask according to the selected dataset."""

    if cfg["mask_type"] == "era5_land_lsm":
        return load_era5_land_mask(cfg, ds_source)

    if cfg["mask_type"] == "valid_data":
        return make_valid_data_mask(da, cfg)

    raise ValueError(f"Unknown mask type: {cfg['mask_type']}")


def add_mask_to_source_grid(source_grid, mask):
    """Attach a binary xESMF mask to the source grid."""

    source_grid = source_grid.copy()
    source_grid["mask"] = mask

    return source_grid


# ====================================================
# Regridding
# ====================================================

def make_regridder(source_grid, target_grid):
    """Create masked conservative-normalized xESMF regridder."""

    return xe.Regridder(
        source_grid,
        target_grid,
        method="conservative_normed",
        periodic=False,
        reuse_weights=False,
        unmapped_to_nan=True,
    )


def regrid_data(da, regridder):
    """Apply the xESMF regridder."""

    return regridder(da)


# ====================================================
# Diagnostics
# ====================================================

def print_summary(da, source_mask):
    """Print simple checks before regridding."""

    print("")
    print("Source mask check")
    print("-----------------")
    print(f"mask dims: {source_mask.dims}")
    print(f"mask shape: {source_mask.shape}")
    print(f"valid fraction: {float(source_mask.mean().values):.4f}")
    print(f"valid cells: {int(source_mask.sum().values)}")
    print(f"total cells: {source_mask.size}")

    print("")
    print("Input data check")
    print("----------------")
    print(f"data dims: {da.dims}")
    print(f"data shape: {da.shape}")
    print(f"valid fraction: {float(da.notnull().mean().values):.4f}")
    print(f"min: {float(da.min(skipna=True).values)}")
    print(f"max: {float(da.max(skipna=True).values)}")


# ====================================================
# Main processing
# ====================================================

def process_one_year(year, cfg):
    """Regrid one year of data."""

    variable = cfg["variable"]

    print("")
    print("====================================================")
    print(f"Processing {cfg['dataset']}: {variable}, {year}")
    print("====================================================")

    input_file = find_input_file(
        path_in=cfg["path_in"],
        file_pattern=cfg["file_pattern"],
        year=year,
    )

    print(f"Input file: {input_file}")

    ds = xr.open_dataset(input_file)
    ds = prepare_dataset(ds)

    da = ds[variable]
    da = prepare_variable(da, cfg)

    if TEST_ONE_DAY:
        da = da.isel(time=TEST_TIME_INDEX)

    source_mask = make_source_mask(
        cfg=cfg,
        ds_source=ds,
        da=da,
    )


    if cfg["dataset"] == "senorge":
        da = da.where(source_mask == 1)
        da = da.fillna(0.0)
    
    source_grid = make_source_grid(ds, cfg)
    source_grid = add_mask_to_source_grid(source_grid, source_mask)

    target_grid = make_target_grid()

    regridder = make_regridder(source_grid, target_grid)
    da_out = regrid_data(da, regridder)

    da_out.name = variable
    da_out.attrs["units"] = cfg["units_out"]
    da_out.attrs["long_name"] = (
        f"{cfg['description']} interpolated to 0.5 degree grid"
    )
    da_out.attrs["dataset"] = cfg["dataset"]
    da_out.attrs["source_variable"] = variable
    da_out.attrs["target_grid"] = "regular 0.5x0.5"
    da_out.attrs["interpolation_method"] = "conservative_normed"
    da_out.attrs["mask_convention"] = "1 = valid source cell, 0 = masked source cell"

    if cfg["dataset"] == "era5_land":
        da_out.attrs["source_grid"] = "ERA5-Land 0.1x0.1"
        da_out.attrs["source_mask"] = "lsm >= 0.5"
    else:
        da_out.attrs["source_grid"] = "seNorge native 1 km grid"
        da_out.attrs["source_mask"] = f"non-NaN {variable} values"

    ds_out = da_out.to_dataset()

    if ds_out.lat[0] < ds_out.lat[-1]:
        ds_out = ds_out.sortby("lat", ascending=False)

    return ds_out


def write_output(ds, year, cfg):
    """Write one yearly output file."""

    os.makedirs(cfg["path_out"], exist_ok=True)

    if TEST_ONE_DAY:
        filename = f"{cfg['variable']}_regrid_0.5x0.5_{year}_test_one_day.nc"
    else:
        filename = f"{cfg['variable']}_regrid_0.5x0.5_{year}.nc"

    output_file = os.path.join(cfg["path_out"], filename)

    print(f"Writing: {output_file}")
    ds.to_netcdf(output_file)

    return output_file


# ====================================================
# Run script
# ====================================================

if __name__ == "__main__":

    cfg = get_config()

    for year in YEARS:

        ds_out = process_one_year(year, cfg)

        if WRITE_TO_FILE:
            write_output(ds_out, year, cfg)
        else:
            print(ds_out)
