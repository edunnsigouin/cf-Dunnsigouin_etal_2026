#!/usr/bin/env python3
"""
Interpolate daily ERA5-Land surface runoff from 0.1° to 0.5°.

Input:
    sro/sro_0.1x0.1_[year].nc

Output:
    sro/sro_model_grid_0.5x0.5_[year].nc

Notes:
    - Conservative interpolation is used because surface runoff is an
      accumulated hydrological variable.
    - No land-sea mask is used. ERA5-Land surface runoff over sea is assumed
      to be zero or effectively zero.
"""

import glob
import os

import numpy as np
import xarray as xr
import xesmf as xe

from Dunnsigouin_etal_2026 import config


# ====================================================
# User settings
# ====================================================

VARIABLE = "sro"
YEARS = np.arange(2023, 2024)

WRITE_TO_FILE = True

PATH_BASE = config.dirs["era5_land_continuous_daily_scandinavia"]

PATH_IN = os.path.join(PATH_BASE, VARIABLE)
PATH_OUT = os.path.join(PATH_BASE, VARIABLE)


# ====================================================
# Target 0.5° grid
# ====================================================

TARGET_LATS = np.arange(73.5, 52.5, -0.5)
TARGET_LONS = np.arange(2.0, 33.0, 0.5)


# ====================================================
# Helper functions
# ====================================================

def normalise_units(units):
    """Return a simplified unit string for easier comparison."""

    if units is None:
        return ""

    return (
        units.strip()
        .lower()
        .replace(" ", "")
        .replace("−", "-")
        .replace("_", "")
    )


def standardise_lat_lon_names(ds):
    """Rename latitude/longitude coordinates to lat/lon if needed."""

    rename_dict = {}

    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"

    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def sort_lat_increasing(ds):
    """Sort latitude from south to north."""

    if ds.lat[0] > ds.lat[-1]:
        ds = ds.sortby("lat")

    return ds


def prepare_dataset(ds):
    """Standardise coordinate names and latitude order."""

    ds = standardise_lat_lon_names(ds)
    ds = sort_lat_increasing(ds)

    return ds


def find_input_file(year):
    """Find the yearly ERA5-Land surface runoff file."""

    pattern = os.path.join(PATH_IN, f"{VARIABLE}_0.1x0.1_{year}.nc")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No file found:\n{pattern}")

    if len(files) > 1:
        raise ValueError(f"More than one file found:\n{files}")

    return files[0]


def convert_sro_to_mm_per_day(da):
    """
    Ensure surface runoff is in mm/day.

    If the input is already mm/day, no conversion is applied.
    If the input is in metres, it is multiplied by 1000.
    """

    original_units = da.attrs.get("units", "")
    units_clean = normalise_units(original_units)

    if units_clean in ["mm", "mm/day", "mmday-1", "mmd-1", "mmperday"]:
        conversion_note = "No conversion applied; input already mm/day."

    elif units_clean in ["m", "meter", "metre", "meters", "metres"]:
        da = da * 1000.0
        conversion_note = "Converted from m to mm/day using factor 1000."

    else:
        raise ValueError(f"Unknown {VARIABLE} units: '{original_units}'")

    da.attrs["units"] = "mm/day"
    da.attrs["original_units"] = original_units
    da.attrs["conversion_note"] = conversion_note

    return da


def make_source_grid(ds):
    """Create the source grid from the ERA5-Land input file."""

    return xr.Dataset(
        coords={
            "lat": ds["lat"],
            "lon": ds["lon"],
        }
    )


def make_target_grid():
    """Create the regular 0.5° target grid."""

    target_grid = xr.Dataset(
        coords={
            "lat": TARGET_LATS,
            "lon": TARGET_LONS,
        }
    )

    return sort_lat_increasing(target_grid)


def interpolate_to_target_grid(da, source_grid, target_grid):
    """Conservatively interpolate data from source grid to target grid."""

    regridder = xe.Regridder(
        source_grid,
        target_grid,
        method="conservative",
        periodic=False,
        reuse_weights=False,
        unmapped_to_nan=True,
    )

    da_out = regridder(
        da,
        skipna=True,
        na_thres=1.0,
    )

    return da_out


def process_one_year(year):
    """Process one year of ERA5-Land surface runoff."""

    print("")
    print("====================================================")
    print(f"Processing {year}")
    print("====================================================")

    input_file = find_input_file(year)
    print(f"Input file: {input_file}")

    ds = xr.open_dataset(input_file)
    ds = prepare_dataset(ds)

    da = ds[VARIABLE]
    da = convert_sro_to_mm_per_day(da)

    source_grid = make_source_grid(ds)
    target_grid = make_target_grid()

    da_out = interpolate_to_target_grid(
        da=da,
        source_grid=source_grid,
        target_grid=target_grid,
    )

    da_out.name = VARIABLE
    da_out.attrs["units"] = "mm/day"
    da_out.attrs["long_name"] = (
        "ERA5-Land daily accumulated surface runoff "
        "interpolated to 0.5 degree grid"
    )
    da_out.attrs["source_grid"] = "ERA5-Land 0.1x0.1"
    da_out.attrs["target_grid"] = "regular 0.5x0.5"
    da_out.attrs["interpolation_method"] = "conservative"
    da_out.attrs["land_mask_used"] = "False"

    ds_out = da_out.to_dataset()

    # Return latitude from north to south, matching TARGET_LATS.
    if ds_out.lat[0] < ds_out.lat[-1]:
        ds_out = ds_out.sortby("lat", ascending=False)

    return ds_out


def write_output(ds, year):
    """Write one yearly output file."""

    os.makedirs(PATH_OUT, exist_ok=True)

    output_file = os.path.join(
        PATH_OUT,
        f"{VARIABLE}_model_grid_0.5x0.5_{year}.nc",
    )

    print(f"Writing: {output_file}")
    ds.to_netcdf(output_file)

    return output_file


if __name__ == "__main__":

    for year in YEARS:

        ds_out = process_one_year(year)

        if WRITE_TO_FILE:
            write_output(ds_out, year)
        else:
            print(ds_out)
