"""
plots A) a map of X-day accumulated rain during Strom Hans and 
B) the fraction of ensemble members with greater X-day accumulated rain
than Hans in forecasts and hincasts throughout the year
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Dunnsigouin_etal_2026 import config, misc

# input --------------------------------------
forecast_date_range = ['2020-01-02','2023-06-26']
acc_days            = 2
path_in             = config.dirs['s2s_processed']
path_out            = config.dirs['fig']
write2file          = True
# --------------------------------------------
def load_data(forecast_date_range,acc_days,path_in):
    filename = f'{path_in}xyt_model_events_with_more_{acc_days}_day_accumulated_precip_than_hans_{forecast_date_range[0]}-{forecast_date_range[-1]}.nc'
    return xr.open_dataset(filename)

def plot_obs_and_count_map(obs, count, *, time_index=0,
                           cmap_obs="Blues", cmap_count="Reds",
                           nlevels_obs=20, nlevels_count=20,
                           extend_obs="max", extend_count="neither",
                           title_obs=None, title_count=None,
                           figsize=(12, 5), dpi=150):
    """
    Plot:
      A) accumulated rain (obs) as a map
      B) count as a side panel map
    Both with contourf + colorbar, different colormaps, and with coastlines + borders.

    Parameters
    ----------
    obs : xr.DataArray
        Expected dims: (time, latitude, longitude) OR (latitude, longitude).
    count : xr.DataArray
        Expected dims: (latitude, longitude).
    time_index : int
        If obs has a time dimension, which index to plot.
    """

    # --- handle obs time dimension ---
    if "time" in obs.dims:
        obs2d = obs.isel(time=time_index)
        tval = obs2d["time"].values
    else:
        obs2d = obs
        tval = None

    # --- coordinate names (your dataset uses latitude/longitude) ---
    lat_name = "latitude"
    lon_name = "longitude"

    lats = obs2d[lat_name].values
    lons = obs2d[lon_name].values

    # Robust extent
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    levels_obs = np.arange(5,55,5)
    levels_count = np.arange(0,1.05,0.05)
    
    proj = ccrs.PlateCarree()
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi,
        subplot_kw={"projection": proj},
        constrained_layout=True
    )

    def add_map_features(ax):
        ax.set_extent(extent, crs=proj)
        ax.coastlines(resolution="50m", linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.7)

    # --- Panel A: obs accumulated rain ---
    add_map_features(ax0)
    cf0 = ax0.contourf(
        obs2d[lon_name], obs2d[lat_name], obs2d,
        levels=levels_obs, cmap=cmap_obs, extend=extend_obs,
        transform=proj
    )
    cb0 = fig.colorbar(cf0, ax=ax0, orientation="horizontal", shrink=0.9, pad=0.02)
    cb0.set_label(f"mm/day")
    ax0.set_title(title_obs)

    # --- Panel B: count ---
    add_map_features(ax1)
    cf1 = ax1.contourf(
        count[lon_name], count[lat_name], count,
        levels=levels_count, cmap=cmap_count, extend=extend_count,
        transform=proj
    )
    cb1 = fig.colorbar(cf1, ax=ax1, orientation="horizontal", shrink=0.9, pad=0.02)
    cb1.set_label('fraction')
    ax1.set_title(title_count)

    
    return fig, (ax0, ax1)



if __name__ == "__main__":

    ds = load_data(forecast_date_range,acc_days,path_in)

    date_hans = ds['obs'].time.values[0].astype('datetime64[D]').astype(str)
    
    fig, axes = plot_obs_and_count_map(
        ds["obs"],
        ds["count"],
        cmap_obs="GnBu",
        cmap_count="RdPu_r",
        nlevels_obs=21,
        nlevels_count=21,
        title_obs=f"{acc_days}-day accumulated rain during Hans ({date_hans})",
        title_count="fraction of forecast + hindcast ensemble members > hans")

    if write2file:
        outname = f"{path_out}/xy_fraction_{acc_days}day_accumulated_precip_greater_than_hans_forecast_hindcast_{forecast_date_range[0]}_{forecast_date_range[-1]}.png"
        fig.savefig(outname, bbox_inches="tight")

    plt.show()    
