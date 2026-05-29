"""
Plot time evolution of one top S2S precipitation event.

The script:
1. Selects one catchment and one ranked event
2. Uses date_of_max as event day 0
3. Plots daily precipitation for event-relative days -2, -1, 0, +1, +2
4. Converts raw precipitation from meters to mm when reading
5. Uses the 0.5x0.5 file first, then falls back to the matching 0.25x0.25 file
6. Overlays the selected catchment boundary in red

Note:
- The 0.5x0.5 files contain lead days 16-46.
- The matching 0.25x0.25 files contain lead days 1-15.
- msl contour plotting can be added later.
"""

# =============================================================================
# Imports
# =============================================================================
import os
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

# --- Choose catchment and event
CATCHMENT_NAME = "drammen"   # "drammen" or "glomma"
EVENT_RANK = 1               # choose 1-5

filename_out = f"{path_out}fig-event-evolution-{CATCHMENT_NAME}-rank{EVENT_RANK}.png"
write2file = False

# --- Variable names
PRECIP_VAR = "tp24"
MSL_VAR = "msl"  # for later use

# --- Event-relative days to plot
EVENT_LAGS = [-2, -1, 0, 1, 2]

# --- Figure settings
FIG_WIDTH_IN = 16
FIG_HEIGHT_IN = 12

MAP_WSPACE = 0.02
MAP_HSPACE = 0.08

tick_labelsize = 12
axis_labelsize = 13
title_fontsize = 14

# --- Map projection and extent
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

# --- Precipitation plotting
PRECIP_LEVELS = np.arange(0, 61, 5)
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


def get_selected_event(catchment_name, event_rank):
    """
    Return one selected event by catchment and rank.
    """
    events = get_top_events(catchment_name)

    for event in events:
        if event["rank"] == event_rank:
            return event

    raise ValueError(
        f"Rank {event_rank} not found for catchment {catchment_name}."
    )


# =============================================================================
# File helpers
# =============================================================================
def get_early_lead_file(source_file):
    """
    Convert the 0.5x0.5 lead-day 16-46 file path to the matching
    0.25x0.25 lead-day 1-15 file path.
    """
    return source_file.replace("0.5x0.5", "0.25x0.25")


def get_time_coord_name(da):
    """
    Identify time coordinate name.
    """
    time_candidates = ["time", "valid_time"]

    for name in time_candidates:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    """
    Return longitude and latitude coordinates.
    """
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]

    return lon, lat


# =============================================================================
# Data loading and selection
# =============================================================================
def load_dataset(filename):
    """
    Open dataset and convert precipitation from meters to millimeters.
    """
    ds = xr.open_dataset(filename)

    if PRECIP_VAR in ds:
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
                da = da.sel({dim: event["hdate"]})
                break

    for dim in member_dim_candidates:
        if dim in da.dims or dim in da.coords:
            da = da.sel({dim: event["ensemble_member"]})
            break

    return da


def has_date(da, target_date):
    """
    Check whether target_date exists in the DataArray time coordinate.
    """
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    times = da[time_name].values.astype("datetime64[ns]")

    return target_date in times


def select_date(da, target_date):
    """
    Select one valid date from DataArray.
    """
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date})


def load_event_day_precip(event, target_date):
    """
    Load daily precipitation for one event-relative target date.

    Try the original 0.5x0.5 file first. If the date is missing,
    try the matching 0.25x0.25 early-lead file.
    """
    files_to_try = [
        event["source_file"],
        get_early_lead_file(event["source_file"]),
    ]

    opened_datasets = []

    try:
        for filename in files_to_try:

            if not os.path.exists(filename):
                continue

            ds = load_dataset(filename)
            opened_datasets.append(ds)

            da = select_event_member(ds, event)

            if has_date(da, target_date):
                da_day = select_date(da, target_date).load()
                source_used = filename
                return da_day, source_used

        raise ValueError(
            f"Could not find target date {target_date} in either "
            f"0.5x0.5 or 0.25x0.25 file."
        )

    finally:
        for ds in opened_datasets:
            ds.close()


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

    if centers.ndim != 1:
        raise ValueError("centers must be 1D")
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer edges")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_precipitation(ax, da_precip, proj_data):
    """
    Plot daily precipitation.
    """
    lon, lat = get_lon_lat(da_precip)

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


# Placeholder for later
def plot_msl_contours(ax, da_msl, proj_data):
    """
    Placeholder for later mean-sea-level pressure contours.

    Example later:
    ax.contour(
        lon,
        lat,
        msl / 100.0,
        levels=np.arange(960, 1045, 5),
        colors="0.4",
        linewidths=0.8,
        transform=proj_data,
    )
    """
    pass


def finalize_figure(
    fig,
    axes,
    mesh,
    event,
    catchment_metadata,
    event_days,
    source_files_used,
    savepath=None,
    write2file=False,
):
    """
    Add titles, shared colorbar, layout, save, and show.
    """
    plot_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, lag, date, source_file in zip(
        plot_axes,
        EVENT_LAGS,
        event_days,
        source_files_used,
    ):
        basename = os.path.basename(source_file)
        resolution = "0.25°" if "0.25x0.25" in basename else "0.5°"

        ax.set_title(
            f"Day {lag:+d}: {date} ({resolution})",
            fontsize=title_fontsize,
            pad=3,
        )

    fig.suptitle(
        (
            f"{catchment_metadata['label']} | "
            f"Rank {event['rank']} event | "
            f"max 2-day precipitation = {event['max_value']:.1f} mm"
        ),
        fontsize=title_fontsize + 2,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.06,
        top=0.92,
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
    cbar.set_label("Daily precipitation (mm)", fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    if write2file:
        fig.savefig(savepath, dpi=300)

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    catchment_metadata = get_catchment_metadata(CATCHMENT_NAME)
    event = get_selected_event(CATCHMENT_NAME, EVENT_RANK)

    date_of_max = np.datetime64(event["date_of_max"], "D")
    event_days = [
        str(date_of_max + np.timedelta64(lag, "D"))
        for lag in EVENT_LAGS
    ]

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

    mesh = None
    source_files_used = []

    for ax, lag, target_date in zip(plot_axes, EVENT_LAGS, event_days):

        da_day, source_used = load_event_day_precip(
            event,
            target_date=target_date,
        )

        source_files_used.append(source_used)

        mesh = plot_precipitation(
            ax,
            da_day,
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
        event,
        catchment_metadata,
        event_days,
        source_files_used,
        savepath=filename_out,
        write2file=write2file,
    )
