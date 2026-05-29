"""
Plot top-5 S2S 2-day accumulated precipitation events for one catchment.

Raw data are 1-day accumulated precipitation in meters.
The script converts to mm when reading, then computes 2-day accumulations.

Layout:
- 5 event panels
- shared colorbar in empty 6th panel
- 16 x 12 inch publication-style figure
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

# --- Choose catchment
CATCHMENT_NAME = "drammen"

filename_out = f"{path_out}xy-tp-2day-top5-events-{CATCHMENT_NAME}.png"
write2file = True

# --- Variable names
PRECIP_VAR = "tp24"

# --- Figure settings
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 8

MAP_WSPACE = 0.0
MAP_HSPACE = 0.08

tick_labelsize = 12
axis_labelsize = 13
title_fontsize = 14

# --- Map projection and extent
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

# --- Precipitation plotting
PRECIP_LEVELS = np.arange(0, 121, 10)
PRECIP_CMAP = "GnBu"

# --- Catchment CRS if missing
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"


# =============================================================================
# Catchment metadata and event metadata
# =============================================================================
def get_catchment_metadata(catchment_name):
    """
    Return catchment-specific metadata.
    """
    catchments = {
        "drammen": {
            "label": "Drammen",
            "geojson": "catchment_nve_regine_drammen.geojson",
        },
        "glomma": {
            "label": "Glomma",
            "geojson": "catchment_nve_regine_glomma.geojson",
        },
    }

    if catchment_name not in catchments:
        raise ValueError(
            f"Unknown catchment: {catchment_name}. "
            f"Available catchments: {list(catchments.keys())}"
        )

    return catchments[catchment_name]


def get_top_events(catchment_name):
    """
    Return top-5 event metadata for selected catchment.
    """
    events = {
        "drammen": [
            {
                "rank": 1,
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
            {
                "rank": 2,
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
            {
                "rank": 3,
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
            {
                "rank": 4,
                "max_value": 81.35,
                "model_type": "hindcast",
                "forecast_date": "2020-04-23",
                "date_of_max": "2020-06-03",
                "hdate": 20150423.0,
                "ensemble_member": 51,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2020-04-23.nc",
            },
            {
                "rank": 5,
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
        ],

        "glomma": [
            {
                "rank": 1,
                "max_value": 101.00,
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
            {
                "rank": 2,
                "max_value": 73.97,
                "model_type": "hindcast",
                "forecast_date": "2022-04-21",
                "date_of_max": "2022-05-31",
                "hdate": 20160421.0,
                "ensemble_member": 2,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-21.nc",
            },
            {
                "rank": 3,
                "max_value": 63.55,
                "model_type": "hindcast",
                "forecast_date": "2022-04-11",
                "date_of_max": "2022-05-10",
                "hdate": 20150411.0,
                "ensemble_member": 3,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2022-04-11.nc",
            },
            {
                "rank": 4,
                "max_value": 61.12,
                "model_type": "hindcast",
                "forecast_date": "2021-04-15",
                "date_of_max": "2021-05-29",
                "hdate": 20150415.0,
                "ensemble_member": 8,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "hindcast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2021-04-15.nc",
            },
            {
                "rank": 5,
                "max_value": 57.02,
                "model_type": "forecast",
                "forecast_date": "2023-04-03",
                "date_of_max": "2023-04-29",
                "hdate": None,
                "ensemble_member": 43,
                "source_file":
                    "/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf/"
                    "forecast/sfc/daily/europe/tp24/"
                    "tp24_0.5x0.5_2023-04-03.nc",
            },
        ],
    }

    if catchment_name not in events:
        raise ValueError(
            f"No event metadata available for catchment: {catchment_name}"
        )

    return events[catchment_name]


# =============================================================================
# Data loading and processing
# =============================================================================
def load_dataset(filename):
    """
    Open dataset and convert precipitation from meters to millimeters.
    """
    ds = xr.open_dataset(filename)

    ds[PRECIP_VAR] = ds[PRECIP_VAR] * 1000.0
    ds[PRECIP_VAR].attrs["units"] = "mm"

    return ds


def select_event_member(ds, event):
    """
    Select hdate and ensemble member for one event.
    """
    da = ds[PRECIP_VAR]

    hdate_dim_candidates = ["hdate", "hindcast_date"]
    member_dim_candidates = ["number", "member", "ensemble_member", "realization"]

    if event["model_type"] == "hindcast":
        for dim in hdate_dim_candidates:
            if dim in da.dims or dim in da.coords:
                da = da.sel({dim: event["hdate"]},method='nearest')
                break

    for dim in member_dim_candidates:
        if dim in da.dims or dim in da.coords:
            da = da.sel({dim: event["ensemble_member"]},method='nearest')
            break

    return da


def compute_2day_accumulation(da, date_of_max):
    """
    Sum 1-day precipitation over date_of_max - 1 day and date_of_max.
    """
    date_of_max = np.datetime64(date_of_max)
    date_start = date_of_max - np.timedelta64(1, "D")

    time_dim_candidates = ["time", "valid_time"]

    for dim in time_dim_candidates:
        if dim in da.dims or dim in da.coords:
            time_dim = dim
            break
    else:
        raise ValueError("Could not identify time dimension.")

    da_2day = da.sel({time_dim: slice(date_start, date_of_max)}).sum(time_dim)
    da_2day.attrs["units"] = "mm"

    return da_2day


def get_lon_lat(da):
    """
    Return longitude and latitude coordinates.
    """
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]

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
    Load catchment polygon, dissolve it, and keep only the outer boundary.
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
# Plot setup helpers
# =============================================================================
def make_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
    """
    Create 2 x 3 Lambert Conformal map layout.
    """
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        subplot_kw={"projection": proj_map},
        constrained_layout=False,
    )

    for i, ax in enumerate(axes.flat):
        if i == 5:
            ax.set_axis_off()
            continue

        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)

        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


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


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_precipitation(ax, da_precip, lon, lat, proj_data):
    """
    Plot 2-day accumulated precipitation.
    """
    precip = da_precip.values

    lon_edges = centers_to_edges(lon.values)
    lat_edges = centers_to_edges(lat.values)

    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        precip = precip[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        precip = precip[:, ::-1]

    lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)

    mesh = ax.pcolormesh(
        lon_e,
        lat_e,
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
    Overlay selected catchment boundary.
    """
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor="red",
        linewidth=1.8,
        zorder=5,
    )


def finalize_figure(fig, axes, mesh, events, savepath=None, write2file=False):
    """
    Add titles, shared colorbar, layout, save, and show.
    """
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, event in zip(plot_axes, events):
        ax.set_title(
            f"Rank {event['rank']}: {event['max_value']:.1f} mm",
            fontsize=title_fontsize,
            pad=3,
        )

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.06,
        top=0.94,
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    # Colorbar in empty sixth panel
    cax = fig.add_axes([0.69, 0.27, 0.23, 0.025])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )
    cbar.set_label("2-day accumulated precipitation (mm)", fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    catchment_metadata = get_catchment_metadata(CATCHMENT_NAME)
    events = get_top_events(CATCHMENT_NAME)

    catchment_boundary = load_catchment_outer_boundary(
        catchment_metadata["geojson"],
        base_dir=path_in_catchment,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, axes, proj_data = make_map_axes(
        central_lon=CENTRAL_LON,
        central_lat=CENTRAL_LAT,
        extent=MAP_EXTENT,
    )

    plot_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        axes[1, 0],
        axes[1, 1],
    ]

    datasets = []
    mesh = None

    for ax, event in zip(plot_axes, events):

        ds = load_dataset(event["source_file"])
        datasets.append(ds)

        da_event = select_event_member(ds, event)

        da_2day = compute_2day_accumulation(
            da_event,
            date_of_max=event["date_of_max"],
        )

        lon, lat = get_lon_lat(da_2day)

        mesh = plot_precipitation(
            ax,
            da_2day,
            lon,
            lat,
            proj_data,
        )

        plot_catchment_boundary(
            ax,
            catchment_boundary,
            proj_data,
        )

    finalize_figure(
        fig,
        axes,
        mesh,
        events,
        savepath=filename_out,
        write2file=write2file,
    )

    for ds in datasets:
        ds.close()
