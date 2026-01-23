"""
draft figure 01 for hans paper showing obs and hans
4-panel plot:
  (1) ERA5 tp24 map (PlateCarree, pcolormesh, GnBu, coastlines+borders + catchment border)
  (2-4) streamflow, precipitation, snowdepth
Each time-series panel: 2023 daily line + shaded day-of-year 95% interval across all years
PLUS: median (all years, by day-of-year) in tab:red
"""

import json
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

from shapely.geometry import shape
from shapely.ops import unary_union

from Dunnsigouin_etal_2026 import config, misc


# import ---------------------------------------
path_in_obs                = config.dirs['obs']
path_in_era5               = config.dirs['era5_continuous_daily'] + 'tp24/'
path_in_catchment          = config.dirs['nve_catchment']
path_out                   = config.dirs['fig']
filename_in_streamflow     = f'{path_in_obs}streamflow.Bergheim.nc'
filename_in_precipitation  = f'{path_in_obs}precipitation.tunhovd.nc'
filename_in_snowdepth      = f'{path_in_obs}snowdepth.tunhovd.nc'
filename_in_era5           = f'{path_in_era5}tp24_0.5x0.5_2023.nc'
filename_in_catchment      = f"{path_in_catchment}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"
filename_out               = f'{path_out}fig-01.pdf'
write2file                 = False
# ----------------------------------------------


def load_obs_data(filename_01, filename_02, filename_03):
    ds_01 = xr.open_dataset(filename_01).sel(time=slice('1921-01-01', '2025-12-31'))
    ds_02 = xr.open_dataset(filename_02)
    ds_03 = xr.open_dataset(filename_03)
    return ds_01, ds_02, ds_03


def load_era5_data(filename_in_era5):
    ds_era5         = xr.open_dataset(filename_in_era5)
    ds_era5         = ds_era5.rolling(time=2, min_periods=2).sum()
    ds_era5["tp24"] = ds_era5["tp24"] * 1000  # m/2day -> mm/2day
    return ds_era5.sel(time="2023-08-08")


def read_geojson(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dissolve_polygon_geojson(gj: dict):
    geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
    geom = unary_union(geoms)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {geom.geom_type}")
    return geom


def plot_catchment_border(ax, catchment_geom, linewidth=2.0):
    if catchment_geom is None:
        return

    if catchment_geom.geom_type == "Polygon":
        x, y = catchment_geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, color="tab:red", transform=ccrs.PlateCarree())
    elif catchment_geom.geom_type == "MultiPolygon":
        for poly in catchment_geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=linewidth, color="tab:red", transform=ccrs.PlateCarree())


def _infer_edges_1d(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers)
    d = np.median(np.diff(centers))
    return np.concatenate(([centers[0] - d / 2], centers + d / 2))


def plot_station_marker(ax, lon, lat, label=None):
    ax.plot(
        lon, lat,
        marker="o",
        markersize=5,
        markeredgecolor="black",
        markerfacecolor="yellow",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    if label:
        ax.text(
            lon + 0.05, lat + 0.05, label,
            color="yellow",
            fontsize=10,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )


def _year_series_and_climstats_by_doy(da: xr.DataArray, year: int):
    da = da.dropna("time")

    t0, t1 = f"{year}-01-01", f"{year}-12-31"
    x_dates = pd.date_range(t0, t1, freq="D")

    da_y = da.sel(time=slice(t0, t1))
    if da_y.sizes.get("time", 0) != len(x_dates):
        da_y = da_y.resample(time="1D").mean().sel(time=slice(t0, t1))
    y_year = da_y.values

    q = da.groupby("time.dayofyear").quantile([0.025, 0.5, 0.975], dim="time")

    doy = np.arange(1, 366)
    q_low = q.sel(quantile=0.025).sel(dayofyear=doy, drop=True).values
    q_med = q.sel(quantile=0.5).sel(dayofyear=doy, drop=True).values
    q_hi  = q.sel(quantile=0.975).sel(dayofyear=doy, drop=True).values

    return x_dates, y_year, q_low, q_hi, q_med


def plot_panel_era5_tp24_map(
    ax,
    ds_era5: xr.Dataset,
    var="tp24",
    extent=(5, 13.0, 58, 63.0),
    catchment_geom=None,
    catchment_linewidth=2.0,
):
    da = ds_era5[var]
    lon = ds_era5["longitude"].values
    lat = ds_era5["latitude"].values

    lon_e = _infer_edges_1d(lon)
    lat_e = _infer_edges_1d(lat)

    z = da.values
    if lat_e[0] > lat_e[-1]:
        lat_e = lat_e[::-1]
        z = z[::-1, :]

    LON_E, LAT_E = np.meshgrid(lon_e, lat_e)

    m = ax.pcolormesh(
        LON_E, LAT_E, z,
        cmap="GnBu",
        shading="auto",
        vmin=0.0,
        vmax=100.0,
        transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    plot_catchment_border(ax, catchment_geom, linewidth=catchment_linewidth)

    plot_station_marker(ax, lon=9.2483, lat=60.4761, label="Bergheim")
    plot_station_marker(ax, lon=8.7521, lat=60.4629, label="Tunhovd")

    # Gridline labels (easy + clean on PlateCarree)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    t = ds_era5.coords.get("time", None)
    tstr = ""
    if t is not None and np.ndim(t.values) == 0:
        tstr = str(pd.to_datetime(t.values).date())
    ax.set_title(f"a) Storm Hans 2-day precipitation {tstr}".strip())

    return m


def plot_panel_streamflow(ax, ds_streamflow: xr.Dataset, year=2023, var="vannforing"):
    da = ds_streamflow[var]
    x, y, lo, hi, med = _year_series_and_climstats_by_doy(da, year)

    ax.fill_between(x, lo, hi, alpha=0.25, label="95% interval over all years")
    ax.plot(x, med, linewidth=1.4, color="tab:red", label="Median over all years")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title("b) Bergheim station streamflow 2023")
    ax.set_ylabel("m³/s")


def plot_panel_precipitation(ax, ds_precip: xr.Dataset, year=2023, var="precipitation"):
    da = ds_precip[var].rolling(time=2, min_periods=2).sum()
    x, y, lo, hi, med = _year_series_and_climstats_by_doy(da, year)

    ax.fill_between(x, lo, hi, alpha=0.25)
    ax.plot(x, med, linewidth=1.4, color="tab:red")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title("c) Tunhovd station 2-day precipitation 2023")
    ax.set_ylabel("mm / 2 days")


def plot_panel_snowdepth(ax, ds_snow: xr.Dataset, year=2023, var="snowdepth"):
    da = ds_snow[var]
    x, y, lo, hi, med = _year_series_and_climstats_by_doy(da, year)

    ax.fill_between(x, lo, hi, alpha=0.25)
    ax.plot(x, med, linewidth=1.4, color="tab:red")
    ax.plot(x, y, linewidth=1.2, label=f"{year}")

    ax.set_title("d) Tunhovd station snowdepth 2023")
    ax.set_ylabel("cm")


def plot_all_panels(
    ds_era5,
    ds_streamflow,
    ds_precipitation,
    ds_snowdepth,
    catchment_geom=None,
    year=2023,
    outfile=None,
    write2file=write2file,
):
    # PlateCarree is simpler for equal panel sizing
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)

    ax_map = fig.add_subplot(gs[0, 0], projection=proj)
    ax_sf  = fig.add_subplot(gs[0, 1])
    ax_pr  = fig.add_subplot(gs[1, 0])
    ax_sd  = fig.add_subplot(gs[1, 1])

    m = plot_panel_era5_tp24_map(
        ax_map,
        ds_era5,
        var="tp24",
        extent=(5, 13.0, 58, 63.0),
        catchment_geom=catchment_geom,
        catchment_linewidth=2.0,
    )

    cbar = fig.colorbar(m, ax=ax_map, orientation="vertical", pad=0.02, fraction=0.04)
    cbar.set_label("mm / 2 days")

    plot_panel_streamflow(ax_sf, ds_streamflow, year=year, var="vannforing")
    plot_panel_precipitation(ax_pr, ds_precipitation, year=year, var="precipitation")
    plot_panel_snowdepth(ax_sd, ds_snowdepth, year=year, var="snowdepth")

    box_aspect = 1/2
    for ax in (ax_sf, ax_pr, ax_sd):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.set_xlabel("Month")
        #ax.set_box_aspect(box_aspect)

    # Try to enforce the same box aspect on the map as well
    try:
        ax_map.set_box_aspect(box_aspect)
    except Exception:
        pass

    # Legend on one panel
    handles_labels = {}
    for ax in (ax_sf, ax_pr, ax_sd):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            handles_labels[ll] = hh

    ax_sf.legend(
        list(handles_labels.values()),
        list(handles_labels.keys()),
        frameon=False,
        loc="upper left",
        fontsize=10,
    )

    if write2file and outfile:
        fig.savefig(outfile)

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ds_streamflow, ds_precipitation, ds_snowdepth = load_obs_data(
        filename_in_streamflow, filename_in_precipitation, filename_in_snowdepth
    )
    ds_era5 = load_era5_data(filename_in_era5)

    gj = read_geojson(filename_in_catchment)
    catchment = dissolve_polygon_geojson(gj)

    plot_all_panels(
        ds_era5,
        ds_streamflow,
        ds_precipitation,
        ds_snowdepth,
        catchment_geom=catchment,
        year=2023,
        outfile=filename_out,
        write2file=write2file,
    )
