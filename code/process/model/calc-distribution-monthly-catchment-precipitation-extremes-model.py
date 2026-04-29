"""
Calculates distribution of monthly extremes for catchment averaged and time accumulated
total precipitation in model forecasts and hindcasts. 
"""

import numpy as np
import xarray as xr
import pandas as pd
from Dunnsigouin_etal_2026 import config

# input ------------------------------------------------
variable            = 'tp'
x_days              = 2
catchment           = "regine_drammen"
forecast_date_range = ['2020-01-02','2023-06-26']
path_in_forecast    = config.dirs['s2s_forecast_daily'] + variable + '/'
path_in_hindcast    = config.dirs['s2s_hindcast_daily'] + variable + '/'
path_out            = config.dirs['s2s_processed']
write2file          = False
# ------------------------------------------------------


def get_forecast_dates(forecast_date_range, option="mt"):
    """
    Generate a list of forecast dates within a given date range.

    Parameters
    ----------
    forecast_date_range : list-like
        ['YYYY-MM-DD', 'YYYY-MM-DD']
    option : str
        'mt'  -> Mondays and Thursdays
        'all' -> all calendar days
    """

    start_date = pd.to_datetime(forecast_date_range[0])
    end_date   = pd.to_datetime(forecast_date_range[1])

    if option == "mt":
        mondays        = pd.date_range(start_date, end_date, freq="W-MON")
        thursdays      = pd.date_range(start_date, end_date, freq="W-THU")
        forecast_dates = mondays.union(thursdays)

    else:
        forecast_dates = pd.date_range(start_date, end_date, freq="D")

    return forecast_dates.sort_values().strftime("%Y-%m-%d").tolist()



if __name__ == "__main__":

    forecast_dates = get_forecast_dates(forecast_date_range, option="mt")

    print(len(forecast_dates))
