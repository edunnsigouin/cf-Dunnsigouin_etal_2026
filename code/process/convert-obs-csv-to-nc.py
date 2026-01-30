"""
Converts observational data from nve Sildre and seklima.no in csv format
to nc.
"""

import numpy as np
import xarray as xr
import pandas as pd
from Dunnsigouin_etal_2026 import config, misc


# input ----------------------------------------
path_in_obs                = config.dirs['obs']
filename_in_streamflow     = f'{path_in_obs}streamflow.kolbjørnshus.csv'
filename_in_precipitation  = f'{path_in_obs}precipitation.ål.III.csv'
filename_in_snowdepth      = f'{path_in_obs}snowdepth.ål.III.csv'
filename_out_streamflow    = filename_in_streamflow.replace(".csv", ".nc")
filename_out_precipitation = filename_in_precipitation.replace(".csv", ".nc")
filename_out_snowdepth     = filename_in_snowdepth.replace(".csv", ".nc")
write2file                 = True
# ----------------------------------------------

def nve_sildre_csv_to_dataarray(filename, time_col="Tidspunkt", value_col="Vannføring (m³/s)"):
    df = pd.read_csv(filename, comment="#", sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    # Parse time and DROP timezone immediately
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)

    da = (
        df.set_index(time_col)[value_col]
          .rename("vannforing")
          .to_xarray()
          .rename({time_col: "time"})
    )
    da.attrs["units"] = "m3 s-1"
    da.attrs["long_name"] = value_col
    return da


def seklima_csv_to_dataarray(
    filename,
    time_col="Tid(norsk normaltid)",
    value_col="Nedbør (døgn)",
    value_col_new="precipitation",
    units="mm/day",
):
    df = pd.read_csv(filename, comment="#", sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    df = df.iloc[:-1]  # drop footer row

    # Parse dd.mm.yyyy and ensure timezone-naive
    df[time_col] = (
        pd.to_datetime(df[time_col], format="%d.%m.%Y", errors="coerce")
          .dt.tz_localize(None)
    )

    df = df.dropna(subset=[time_col])

    da = (
        df.set_index(time_col)[value_col]
          .rename(value_col_new)
          .to_xarray()
          .rename({time_col: "time"})
    )
    da.attrs["units"] = units
    da.attrs["long_name"] = value_col_new
    return da


if __name__ == "__main__":

    da_streamflow = nve_sildre_csv_to_dataarray(filename_in_streamflow)

    da_precipitation = seklima_csv_to_dataarray(filename_in_precipitation,time_col="Tid(norsk normaltid)",value_col="Nedbør (døgn)",value_col_new='precipitation',units='mm/day')

    da_snowdepth = seklima_csv_to_dataarray(filename_in_snowdepth,time_col="Tid(norsk normaltid)",value_col="Snødybde",value_col_new='snowdepth',units='cm')

    if write2file:
        da_streamflow.to_netcdf(filename_out_streamflow)
        da_precipitation.to_netcdf(filename_out_precipitation)
        da_snowdepth.to_netcdf(filename_out_snowdepth)
