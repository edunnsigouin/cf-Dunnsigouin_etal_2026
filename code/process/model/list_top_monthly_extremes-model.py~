"""
List the 5 largest model monthly precipitation maxima for a selected month.

For each event, print:
- rank
- maximum value
- date of maximum
- forecast or hindcast
- forecast initialization date
- hindcast date, if relevant
- ensemble member
- source file
"""

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# Input -------------------------------------------------
variable            = "tp"
x_days              = 2
catchment           = "regine_glomma"
forecast_date_range = ["2020-01-02", "2023-06-26"]

month_of_year = 5   # 1=Jan, 2=Feb, ..., 12=Dec
n_top         = 5

path_in_model = config.dirs["s2s_processed"]

filename_in_model = (
    f"{path_in_model}"
    f"distribution_monthly_extremes_{variable}_{x_days}dayacc_"
    f"nve_catchment_{catchment}_forecast_hindcast_"
    f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
)
# -------------------------------------------------------


def load_model_data(filename_in_model):
    """Load model monthly extreme dataset."""

    return xr.open_dataset(filename_in_model)


def print_top_monthly_model_events(model_ds, month_of_year, n_top=5):
    """Print the largest model events for one month of year."""

    month_ds = model_ds.sel(month_of_year=month_of_year)

    values = month_ds["max_value"].values

    valid = np.isfinite(values)

    if valid.sum() == 0:
        print(f"No valid events found for month {month_of_year}.")
        return

    valid_indices = np.where(valid)[0]
    valid_values = values[valid]

    order = np.argsort(valid_values)[::-1]
    top_indices = valid_indices[order[:n_top]]

    print(f"\nTop {n_top} model events for month {month_of_year}")
    print("-" * 70)

    for rank, idx in enumerate(top_indices, start=1):

        max_value = month_ds["max_value"].isel(index=idx).item()
        date_of_max = month_ds["date_of_max"].isel(index=idx).values
        forecast_date = month_ds["forecast_date"].isel(index=idx).values
        hdate = month_ds["hdate"].isel(index=idx).values
        ensemble_member = month_ds["ensemble_member"].isel(index=idx).item()
        model_type = month_ds["model_type"].isel(index=idx).item()
        source_file = month_ds["source_file"].isel(index=idx).item()

        print(f"\nRank {rank}")
        print(f"  max_value       : {max_value:.2f} mm")
        print(f"  date_of_max     : {date_of_max}")
        print(f"  model_type      : {model_type}")
        print(f"  forecast_date   : {forecast_date}")
        print(f"  hdate           : {hdate}")
        print(f"  ensemble_member : {ensemble_member}")
        print(f"  source_file     : {source_file}")


if __name__ == "__main__":

    model_ds = load_model_data(filename_in_model)

    print_top_monthly_model_events(
        model_ds=model_ds,
        month_of_year=month_of_year,
        n_top=n_top,
    )
