#!/usr/bin/env python3
"""
List the largest S2S monthly precipitation-maximum events for a selected month.

The input is the monthly_max_samples NetCDF produced by the monthly-maximum
sample script. For each event, print rank, maximum value, date of maximum,
forecast or hindcast, forecast initialization date, hindcast date if relevant,
ensemble member, lead day of maximum, and original raw S2S source file.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"
catchment = "regine_drammen"
forecast_date_range = ["2020-01-02", "2023-12-28"]
observation_years = ["1957", "2025"]
accumulation_days = 2

# Options: "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
bias_correction_method = "raw"

# Options: "senorge", "era5"
bias_correction_reference = "senorge"

month_of_year = 5
n_top = 10

input_filename_override = None

path_s2s = Path(config.dirs["s2s_processed"])
raw_s2s_base_dir = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

# Raw files are checked in this order.
raw_grids = ["0.5x0.5", "0.25x0.25"]


# =============================================================================
# Filename helpers
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in S2S filenames."""
    if catchment_name.startswith("regine_"):
        return catchment_name.replace("regine_", "", 1)
    return catchment_name


def make_input_filename():
    """Return the monthly-maximum sample filename."""
    if input_filename_override is not None:
        return Path(input_filename_override)

    if bias_correction_method == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{bias_correction_method}_{bias_correction_reference}_"
            f"{observation_years[0]}-{observation_years[1]}"
        )

    return path_s2s / (
        f"monthly_max_samples_{variable}_{accumulation_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}_"
        f"{correction_label}.nc"
    )


def make_raw_s2s_filename(model_type, forecast_date, grid):
    """Return the expected original raw S2S file path."""
    return (
        raw_s2s_base_dir
        / model_type
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{grid}_{forecast_date}.nc"
    )


def find_raw_s2s_filename(model_type, forecast_date):
    """Return the existing raw S2S file, checking configured grids in order."""
    candidates = [
        make_raw_s2s_filename(model_type, forecast_date, grid)
        for grid in raw_grids
    ]

    for filename in candidates:
        if filename.is_file():
            return filename

    return candidates[0]


# =============================================================================
# Event extraction
# =============================================================================

def validate_input_dataset(ds):
    """Check that the monthly-maximum sample dataset has the required structure."""
    required = {
        "tp24_max",
        "date_of_max",
        "lead_of_max",
        "sample_month",
        "model_type",
        "hdate",
    }
    missing = required - set(ds.variables)
    if missing:
        raise ValueError(f"Input file is missing variables: {sorted(missing)}")

    if set(ds["tp24_max"].dims) != {"number", "i_date"}:
        raise ValueError("tp24_max must have dimensions number and i_date.")

    if set(ds["sample_month"].dims) != {"i_date"}:
        raise ValueError("sample_month must have dimension i_date.")


def decode_hdate(value):
    """Return YYYY-MM-DD for hindcast hdate, or '-' for forecast rows."""
    value = int(value)
    if value == 0:
        return "-"

    text = f"{value:08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def format_date(value):
    """Return a compact YYYY-MM-DD date string."""
    value = np.asarray(value).astype("datetime64[ns]")
    if np.isnat(value):
        return "-"
    return np.datetime_as_string(value.astype("datetime64[D]"), unit="D")


def get_top_event_indices(model_ds, month_of_year, n_top):
    """Return positional i_date and number indices for the largest events."""
    values = model_ds["tp24_max"].transpose("i_date", "number").values
    sample_month = model_ds["sample_month"].values.astype("int64")
    month_mask = sample_month % 100 == month_of_year

    valid = np.isfinite(values) & month_mask[:, np.newaxis]
    i_indices, number_indices = np.where(valid)

    if i_indices.size == 0:
        return []

    event_values = values[i_indices, number_indices]
    order = np.argsort(event_values)[::-1][:n_top]

    return [
        (int(i_indices[index]), int(number_indices[index]))
        for index in order
    ]


def print_top_monthly_model_events(model_ds, month_of_year, n_top=10):
    """Print the largest model events for one calendar month."""
    if not 1 <= month_of_year <= 12:
        raise ValueError("month_of_year must be between 1 and 12.")
    if n_top < 1:
        raise ValueError("n_top must be at least 1.")

    top_indices = get_top_event_indices(model_ds, month_of_year, n_top)

    if not top_indices:
        print(f"No valid events found for month {month_of_year}.")
        return

    print(f"\nTop {len(top_indices)} model events for month {month_of_year}")
    print("-" * 70)

    for rank, (i_index, number_index) in enumerate(top_indices, start=1):
        sample = model_ds.isel(i_date=i_index, number=number_index)

        max_value = sample["tp24_max"].item()
        date_of_max = sample["date_of_max"].values
        lead_of_max = sample["lead_of_max"].item()

        i_date = model_ds["i_date"].isel(i_date=i_index).values
        forecast_date = format_date(i_date)
        number = model_ds["number"].isel(number=number_index).item()
        model_type = str(model_ds["model_type"].isel(i_date=i_index).item())
        hdate = model_ds["hdate"].isel(i_date=i_index).item()

        source_file = find_raw_s2s_filename(
            model_type=model_type,
            forecast_date=forecast_date,
        )

        print(f"\nRank {rank}")
        print(f"  max_value        : {max_value:.2f} mm")
        print(f"  date_of_max      : {format_date(date_of_max)}")
        print(f"  model_type       : {model_type}")
        print(f"  forecast_date    : {forecast_date}")
        print(f"  hdate            : {decode_hdate(hdate)}")
        print(f"  ensemble_member  : {number}")
        print(f"  lead_of_max      : {lead_of_max:.0f} days")
        print(f"  source_file      : {source_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Load monthly-maximum samples and print the largest selected-month events."""
    filename = make_input_filename()

    if not filename.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")

    print("Reading:", filename)

    with xr.open_dataset(filename, decode_timedelta=False) as opened:
        model_ds = opened.load()

    validate_input_dataset(model_ds)
    print_top_monthly_model_events(
        model_ds=model_ds,
        month_of_year=month_of_year,
        n_top=n_top,
    )


if __name__ == "__main__":
    main()
