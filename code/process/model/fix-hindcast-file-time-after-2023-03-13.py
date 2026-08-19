#!/usr/bin/env python3
"""
Fix incorrect time units in ECMWF S2S hindcast NetCDF files.

For the known file formats:
- 31 time steps represent lead days 16-46, so time index 0 is initialization + 15 days.
- 46 time steps represent lead days 1-46, so time index 0 is the initialization date.

Only the time:units attribute is changed. Data values are not rewritten.
"""

from datetime import date, timedelta
from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset


# =============================================================================
# User settings
# =============================================================================

path_in = Path(
    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/hindcast/sfc/daily/europe/tp24"
)

first_date = date.fromisoformat("2023-03-20")
last_date = date.fromisoformat("2023-12-28")

# True: only report proposed changes. False: modify files in place.
dry_run = True


# =============================================================================
# Time-unit repair
# =============================================================================

def initialization_date_from_filename(filename):
    """Extract YYYY-MM-DD from filenames ending in _YYYY-MM-DD.nc."""
    match = re.search(r"_(\d{4}-\d{2}-\d{2})\.nc$", filename.name)
    if match is None:
        return None
    return date.fromisoformat(match.group(1))


def expected_time_origin(initialization_date, number_of_times):
    """Return the correct reference date for the known 31- and 46-step formats."""
    if number_of_times == 31:
        return initialization_date + timedelta(days=15)
    if number_of_times == 46:
        return initialization_date
    raise ValueError(f"Unsupported time dimension: {number_of_times}")


def fix_file(filename):
    """Check one file and optionally replace its time:units attribute."""
    initialization_date = initialization_date_from_filename(filename)
    if initialization_date is None or not first_date <= initialization_date <= last_date:
        return

    mode = "r" if dry_run else "r+"
    with Dataset(filename, mode) as ds:
        if "time" not in ds.variables:
            raise KeyError(f"{filename.name}: missing time variable")

        time = ds.variables["time"]
        number_of_times = time.size

        values = np.asarray(time[:])
        expected_values = np.arange(number_of_times)
        if not np.array_equal(values, expected_values):
            raise ValueError(
                f"{filename.name}: expected time values 0-{number_of_times - 1}, "
                f"found {values}"
            )

        origin = expected_time_origin(initialization_date, number_of_times)
        expected_units = f"days since {origin.isoformat()}"
        current_units = getattr(time, "units", None)

        if current_units == expected_units:
            print(f"OK        {filename.name}: {current_units}")
            return

        action = "WOULD FIX" if dry_run else "FIXED"
        print(f"{action:9} {filename.name}: {current_units!r} -> {expected_units!r}")

        if not dry_run:
            time.units = expected_units


if __name__ == "__main__":
    for filename in sorted(path_in.glob("tp24_0.5x0.5_*.nc")):
        fix_file(filename)
