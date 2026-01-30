"""
Plots a 1x3 cartopy map panel for accumulated precip during storm Hans:
  (1) Hans X-day accumulated precip (mm/Xday)
  (2) Climatological monthly std of X-day accumulated precip (mm/Xday)
  (3) Hans / climatological std (unitless)
"""

import json
import numpy               as np
import xarray              as xr
import matplotlib.pyplot   as plt
import cartopy.crs         as ccrs
import cartopy.feature     as cfeature
from shapely.geometry      import shape
from shapely.ops           import unary_union
from Dunnsigouin_etal_2026 import config, misc


# input -------------------------------------
clim_years               = np.arange(1941, 2023, 1)
grid                     = '0.5x0.5'
domain                   = 'norway'
x_days                   = 2
hans_date                = '2023-08-08'
path_in_clim             = config.dirs['era5_processed']
path_in_hans             = config.dirs['era5_continuous_daily'] + 'tp24/'
path_in_catchment        = config.dirs['nve_catchment']
path_out                 = config.dirs['fig']
filename_in_hans         = f'{path_in_hans}tp24_{grid}_2023.nc'
filename_in_clim         = f'{path_in_clim}xyt_climatology_tp24_{x_days}dayacc_monthly_{grid}_{clim_years[0]}-{clim_years[-1]}.nc'
filename_in_catchment_01 = f"{path_in_catchment}nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson"
filename_in_catchment_02 = f"{path_in_catchment}nve_regine_enhet_002_glommavassdraget_entire_catchment.geojson"
filename_out             = f"{path_out}xy_storm_hans_3panel_tp24_{x_days}dayacc_normalized_{grid}.pdf"
write2file               = True
# -------------------------------------------


def load_hans_data(filename_in_hans, hans_date, x_days):
    
    domain_lats, domain_lons = misc.get_domain_latlon(domain)
    ds_hans                  = xr.open_dataset(filename_in_hans).sel(latitude=domain_lats, longitude=domain_lons)
    ds_hans                  = ds_hans.rolling(time=x_days, min_periods=x_days).sum()
    da_hans                  = ds_hans["tp24"] * 1000.0
    da_hans.attrs["units"]   = f"mm/{x_days}day"

    return da_hans.sel(time=hans_date)


def load_clim_data(filename_in_clim, month):
    da_std = xr.open_dataset(filename_in_clim)["std"]
    return da_std.sel(month=month)


# ---------------- Catchment helpers ----------------

def read_geojson(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dissolve_polygon_geojson(gj: dict):
    geoms = [shape(feat["geometry"]) for feat in gj.get("features", [])]
    geom = unary_union(geoms)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {geom.geom_type}")
    return geom


def load_catchment_geom(filepath: str):
    gj = read_geojson(filepath)
    return dissolve_polygon_geojson(gj)


def plot_catchment_border(ax, catchment_geom, linewidth=2.0, color="tab:red", zorder=6):
    if catchment_geom is None:
        return

    if catchment_geom.geom_type == "Polygon":
        x, y = catchment_geom.exterior.xy
        ax.plot(x, y, linewidth=linewidth, color=color, transform=ccrs.PlateCarree(), zorder=zorder)

    elif catchment_geom.geom_type == "MultiPolygon":
        for poly in catchment_geom.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=linewidth, color=color, transform=ccrs.PlateCarree(), zorder=zorder)


# ---------------- Plot helpers ----------------

def _setup_map_ax(ax, extent_south_norway=None, catchment_geoms=None, catchment_linewidth=2.0):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, zorder=3)

    if extent_south_norway is None:
        extent_south_norway = [5, 13.0, 58, 63.0]
    ax.set_extent(extent_south_norway, crs=ccrs.PlateCarree())

    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.4, alpha=0.4)

    # Overlay multiple catchments
    if catchment_geoms is not None:
        for geom in catchment_geoms:
            plot_catchment_border(ax, geom, linewidth=catchment_linewidth)

    return ax


def _lonlat_mesh(da):
    lon = da["longitude"].values
    lat = da["latitude"].values
    Lon, Lat = np.meshgrid(lon, lat)
    return Lon, Lat


def plot_panel_hans(ax, da_hans, catchment_geoms=None):
    _setup_map_ax(ax, catchment_geoms=catchment_geoms)
    Lon, Lat = _lonlat_mesh(da_hans)

    m = ax.pcolormesh(
        Lon, Lat, da_hans.values,
        cmap="GnBu",
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=0.0,
        vmax=100.0,
    )
    ax.set_title(f"Hans precip ({da_hans.attrs.get('units','')})\n{np.datetime_as_string(da_hans['time'].values, unit='D')}")
    cb = plt.colorbar(m, ax=ax, orientation="vertical", pad=0.04, fraction=0.05)
    cb.set_label(da_hans.attrs.get("units", ""))
    return m


def plot_panel_clim_std(ax, da_std, catchment_geoms=None):
    _setup_map_ax(ax, catchment_geoms=catchment_geoms)
    Lon, Lat = _lonlat_mesh(da_std)

    m = ax.pcolormesh(
        Lon, Lat, da_std.values,
        cmap="GnBu",
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    ax.set_title(f"Climatology std ({da_std.attrs.get('units','')})\nMonth = {int(da_std['month'].values)}")
    cb = plt.colorbar(m, ax=ax, orientation="vertical", pad=0.04, fraction=0.05)
    cb.set_label(da_std.attrs.get("units", ""))
    return m


def plot_panel_ratio(ax, da_hans, da_std, catchment_geoms=None, eps=1e-12):
    _setup_map_ax(ax, catchment_geoms=catchment_geoms)
    Lon, Lat = _lonlat_mesh(da_hans)

    ratio = da_hans / (da_std + eps)

    m = ax.pcolormesh(
        Lon, Lat, ratio.values,
        cmap="PiYG",
        transform=ccrs.PlateCarree(),
        shading="auto",
        vmin=0.0,
        vmax=10.0,
    )
    ax.set_title("Hans / climatology std (unitless)")
    cb = plt.colorbar(m, ax=ax, orientation="vertical", pad=0.04, fraction=0.05)
    cb.set_label("standard deviations")
    return m


def plot_all_panels(da_hans, da_std, catchment_geoms, write2file, filename_out):

    proj = ccrs.LambertConformal(
        central_longitude=15,
        central_latitude=65,
        standard_parallels=(63, 70),
    )

    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(15, 7.5),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    plot_panel_hans(axes[0], da_hans, catchment_geoms=catchment_geoms)
    plot_panel_clim_std(axes[1], da_std, catchment_geoms=catchment_geoms)
    plot_panel_ratio(axes[2], da_hans, da_std, catchment_geoms=catchment_geoms)

    fig.suptitle(f"{x_days}-day accumulated precip: Storm Hans vs climatology", y=1.02)

    if write2file:
        fig.savefig(filename_out, bbox_inches="tight")
   
    plt.show()

    
# ---------------- main ----------------

if __name__ == "__main__":

    da_hans = load_hans_data(filename_in_hans, hans_date, x_days=x_days)
    da_std  = load_clim_data(filename_in_clim, month=8)

    # Load both catchments
    catchment_01 = load_catchment_geom(filename_in_catchment_01)
    catchment_02 = load_catchment_geom(filename_in_catchment_02)
    catchments   = [catchment_01, catchment_02]
    
    plot_all_panels(da_hans, da_std, catchments, write2file, filename_out)

