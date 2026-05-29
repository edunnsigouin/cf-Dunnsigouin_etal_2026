"""
Plot one top-5 S2S 2-day accumulated precipitation event.

The raw files contain 1-day accumulated precipitation. This script:
1. Opens one forecast or hindcast NetCDF file
2. Selects one ensemble member and, for hindcasts, one hdate
3. Computes the 2-day precipitation accumulation ending on DATE_OF_MAX
4. Plots the event on a Lambert Conformal Cartopy map
5. Overlays the Drammen catchment boundary in red
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

path_in_catchment = config.dirs["nve"]
path_out = config.dirs["fig"]

filename_out = f"{path_out}fig-top5-event-precip.png"
write2file = False

# Choose which event to plot (1-5)
EVENT_NUMBER = 4

TOP_EVENTS = {

    1: {
        "label": "Rank 1",
        "max_value": 100.26,
        "model_type": "hindcast",
        "forecast_date": "2021-04-26",
        "date_of_max": "2021-06-06",
        "hdate": 20150426.0,
        "ensemble_member": 7,
        "source_file":
            "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
            "hindcast/sfc/daily/europe/tp24/"
            "tp24_0.5x0.5_2021-04-26.nc",
    },

    2: {
        "label": "Rank 2",
        "max_value": 85.55,
        "model_type": "hindcast",
        "forecast_date": "2022-04-28",
        "date_of_max": "2022-06-02",
        "hdate": 20140428.0,
        "ensemble_member": 2,
        "source_file":
            "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
            "hindcast/sfc/daily/europe/tp24/"
            "tp24_0.5x0.5_2022-04-28.nc",
    },

    3: {
        "label": "Rank 3",
        "max_value": 84.83,
        "model_type": "hindcast",
        "forecast_date": "2021-04-29",
        "date_of_max": "2021-05-28",
        "hdate": 20190429.0,
        "ensemble_member": 4,
        "source_file":
            "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
            "hindcast/sfc/daily/europe/tp24/"
            "tp24_0.5x0.5_2021-04-29.nc",
    },

    4: {
        "label": "Rank 4",
        "max_value": 81.35,
        "model_type": "hindcast",
        "forecast_date": "2020-04-23",
        "date_of_max": "2020-06-03",
        "hdate": 20150423.0,
        "ensemble_member": 51, # member 11 is labelled wrong as 51!
        "source_file":
            "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
            "hindcast/sfc/daily/europe/tp24/"
            "tp24_0.5x0.5_2020-04-23.nc",
    },

    5: {
        "label": "Rank 5",
        "max_value": 77.56,
        "model_type": "forecast",
        "forecast_date": "2021-04-26",
        "date_of_max": "2021-06-07",
        "hdate": None,
        "ensemble_member": 17,
        "source_file":
            "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
            "forecast/sfc/daily/europe/tp24/"
            "tp24_0.5x0.5_2021-04-26.nc",
    },
}

EVENT = TOP_EVENTS[EVENT_NUMBER]

EVENT_LABEL = EVENT["label"]
MAX_VALUE = EVENT["max_value"]
MODEL_TYPE = EVENT["model_type"]
FORECAST_DATE = EVENT["forecast_date"]
DATE_OF_MAX = EVENT["date_of_max"]
HDATE = EVENT["hdate"]
ENSEMBLE_MEMBER = EVENT["ensemble_member"]
SOURCE_FILE = EVENT["source_file"]


PRECIP_VAR = "tp24"

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 5.5

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

PRECIP_LEVELS = np.arange(0, 121, 10)
PRECIP_CMAP = "GnBu"

tick_labelsize = 10
axis_labelsize = 11
title_fontsize = 12

CATCHMENT_GEOJSON = "catchment_nve_regine_drammen.geojson"
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"



# =============================================================================
# Data loading and processing
# =============================================================================
def load_dataset(filename):
    """
    Open the S2S precipitation dataset.

    ECMWF precipitation is stored in meters and is converted
    immediately to millimeters.
    """
    ds = xr.open_dataset(filename)

    ds[PRECIP_VAR] = ds[PRECIP_VAR] * 1000.0
    ds[PRECIP_VAR].attrs["units"] = "mm"

    return ds

def select_event_member(ds, model_type, hdate=None, ensemble_member=None):
    """
    Select the requested hindcast/forecast member.

    This function is intentionally flexible about dimension names.
    Adjust the candidate names if your files use different names.
    """
    da = ds[PRECIP_VAR]

    hdate_dim_candidates = ["hdate", "hindcast_date"]
    member_dim_candidates = ["number", "member", "ensemble_member", "realization"]

    if model_type == "hindcast":
        for dim in hdate_dim_candidates:
            if dim in da.dims or dim in da.coords:
                da = da.sel({dim: hdate},method='nearest')
                break

    if ensemble_member is not None:
        for dim in member_dim_candidates:
            if dim in da.dims or dim in da.coords:
                da = da.sel({dim: ensemble_member},method='nearest')
                break

    return da


def compute_2day_accumulation(da, date_of_max):
    """
    Compute 2-day accumulated precipitation ending on date_of_max.

    The raw data are 1-day accumulated precipitation, so this sums:
    date_of_max - 1 day and date_of_max.
    """
    date_of_max = np.datetime64(date_of_max)
    date_start = date_of_max - np.timedelta64(1, "D")

    time_dim_candidates = ["time", "valid_time", "step"]

    for dim in time_dim_candidates:
        if dim in da.dims or dim in da.coords:
            time_dim = dim
            break
    else:
        raise ValueError("Could not identify time dimension.")

    da_2day = da.sel({time_dim: slice(date_start, date_of_max)}).sum(time_dim)

    return da_2day


def get_lon_lat(da):
    """
    Identify longitude and latitude coordinates.
    """
    lon_candidates = ["longitude", "lon"]
    lat_candidates = ["latitude", "lat"]

    for name in lon_candidates:
        if name in da.coords:
            lon = da[name]
            break
    else:
        raise ValueError("Could not identify longitude coordinate.")

    for name in lat_candidates:
        if name in da.coords:
            lat = da[name]
            break
    else:
        raise ValueError("Could not identify latitude coordinate.")

    return lon, lat


# =============================================================================
# Catchment geometry helper
# =============================================================================
def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
    """
    Load Drammen catchment polygon, dissolve it, and keep only the outer boundary.
    """
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    gdf = gpd.read_file(base_dir + filename)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)

    gdf_metric = gdf.to_crs(metric_crs)
    union_geom = gdf_metric.geometry.union_all()

    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)
    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon(
            [Polygon(poly.exterior) for poly in union_geom.geoms]
        )
    else:
        outer_geom = union_geom

    outer_gdf = (
        gpd.GeoDataFrame(geometry=[outer_geom], crs=metric_crs)
        .to_crs(plot_crs)
    )

    return outer_gdf.geometry.iloc[0]


# =============================================================================
# Plot helpers
# =============================================================================
def make_map_axis(central_lon=10.0, central_lat=62.0, extent=None):
    """
    Create one Lambert Conformal map panel.
    """
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        subplot_kw={"projection": proj_map},
    )

    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)

    if extent is not None:
        ax.set_extent(extent, crs=proj_data)

    return fig, ax, proj_data


def centers_to_edges(centers):
    """
    Convert 1D grid-cell centers to edges.
    """
    centers = np.asarray(centers)

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def plot_precipitation(ax, da_precip, lon, lat, proj_data):
    """
    Plot 2-day accumulated precipitation.
    """
    precip = da_precip.values

    if lon.ndim == 1 and lat.ndim == 1:
        lon_edges = centers_to_edges(lon.values)
        lat_edges = centers_to_edges(lat.values)

        if lat_edges[0] > lat_edges[-1]:
            lat_edges = lat_edges[::-1]
            precip = precip[::-1, :]

        if lon_edges[0] > lon_edges[-1]:
            lon_edges = lon_edges[::-1]
            precip = precip[:, ::-1]

        lon_plot, lat_plot = np.meshgrid(lon_edges, lat_edges)

    else:
        lon_plot = lon.values
        lat_plot = lat.values

    mesh = ax.pcolormesh(
        lon_plot,
        lat_plot,
        precip,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_LEVELS.min(),
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )

    return mesh


def plot_catchment_boundary(ax, geometry, proj_data):
    """
    Overlay Drammen catchment boundary.
    """
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor="red",
        linewidth=1.8,
        zorder=5,
    )


def finalize_figure(fig, ax, mesh, savepath=None, write2file=False):
    """
    Add title, colorbar, save, and show.
    """
    title = (
        f"{EVENT_LABEL}: 2-day accumulated precipitation\n"
        f"valid ending {DATE_OF_MAX}, {MODEL_TYPE}, "
        f"member {ENSEMBLE_MEMBER}"
    )

    ax.set_title(title, fontsize=title_fontsize, pad=4)

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        shrink=0.82,
        pad=0.04,
    )
    cbar.set_label("2-day accumulated precipitation (mm)", fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        bottom=0.08,
        top=0.90,
    )

    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    ds = load_dataset(SOURCE_FILE)

    print(ds)
    
    da_event = select_event_member(
        ds,
        model_type=MODEL_TYPE,
        hdate=HDATE,
        ensemble_member=ENSEMBLE_MEMBER,
    )


    da_2day = compute_2day_accumulation(
        da_event,
        date_of_max=DATE_OF_MAX,
    )

    lon, lat = get_lon_lat(da_2day)

    drammen_boundary = load_catchment_outer_boundary(
        CATCHMENT_GEOJSON,
        base_dir=path_in_catchment,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, ax, proj_data = make_map_axis(
        central_lon=CENTRAL_LON,
        central_lat=CENTRAL_LAT,
        extent=MAP_EXTENT,
    )

    mesh = plot_precipitation(
        ax,
        da_2day,
        lon,
        lat,
        proj_data,
    )

    plot_catchment_boundary(
        ax,
        drammen_boundary,
        proj_data,
    )

    finalize_figure(
        fig,
        ax,
        mesh,
        savepath=filename_out,
        write2file=write2file,
    )

    ds.close()

