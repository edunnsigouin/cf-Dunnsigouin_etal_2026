#!/usr/bin/env python3
"""
Plot one high-ranking S2S precipitation event.

The figure follows the same layout as the ERA5 Storm Hans plot:
1. Four event-relative panels: -2, -1, 0, +1.
2. Daily precipitation as shading.
3. Mean sea level pressure as labelled grey contours.
4. Snowmelt, defined as daily change in SWE < 0, as stippling or hatching.
5. Selected catchment boundary in red.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"
EVENT_RANK = 1              # options: 1-5

EVENT_LAGS = [-2, -1, 0, 1]

PRECIP_VAR = "tp24"
MSL_VAR = "msl"
SNOW_VAR = "sd"

WRITE_TO_FILE = True


# =============================================================================
# Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")


# =============================================================================
# Figure settings
# =============================================================================

FIG_WIDTH_IN = 9
FIG_HEIGHT_IN = 10

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.0
MAP_HSPACE = 0.08

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9


# =============================================================================
# Plot styling
# =============================================================================

PRECIP_LEVELS = np.arange(5, 55, 5)
PRECIP_ZERO_THRESHOLD = 5.0
PRECIP_CMAP = plt.get_cmap("GnBu").copy()
PRECIP_CMAP.set_under("white")

MSL_CONTOUR_LEVELS = np.arange(975, 1045, 5)
MSL_CONTOUR_COLOR = "0.7"
MSL_CONTOUR_LINEWIDTH = 1.5

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

SNOWMELT_OVERLAY = "stippling"  # options: "stippling", "hatching"
SNOWMELT_THRESHOLD = 0.0

SNOWMELT_DOT_COLOR = "orange"
SNOWMELT_DOT_SIZE = 4
SNOWMELT_DOT_ALPHA = 1.0
SNOWMELT_DOT_STRIDE = 1

SNOWMELT_HATCH_PATTERN = "////"
SNOWMELT_HATCH_COLOR = "orange"
SNOWMELT_HATCH_LINEWIDTH = 1.0

SNOWMELT_ZORDER = 8


# =============================================================================
# Catchment and event metadata
# =============================================================================

CATCHMENTS = {
    "drammen": {
        "filename": "catchment_nve_regine_drammen.geojson",
        "label": "Drammen catchment",
    },
    "glomma": {
        "filename": "catchment_nve_regine_glomma.geojson",
        "label": "Glomma catchment",
    },
}


TOP_EVENTS = {
    "drammen": [
        {
            "rank": 1,
            "model_type": "hindcast",
            "forecast_date": "2021-04-26",
            "date_of_max": "2021-06-06",
            "hdate": 20150426.0,
            "ensemble_member": 7,
        },
        {
            "rank": 2,
            "model_type": "hindcast",
            "forecast_date": "2022-04-28",
            "date_of_max": "2022-06-02",
            "hdate": 20140428.0,
            "ensemble_member": 2,
        },
        {
            "rank": 3,
            "model_type": "hindcast",
            "forecast_date": "2021-04-29",
            "date_of_max": "2021-05-28",
            "hdate": 20190429.0,
            "ensemble_member": 4,
        },
        {
            "rank": 4,
            "model_type": "hindcast",
            "forecast_date": "2020-04-23",
            "date_of_max": "2020-06-03",
            "hdate": 20150423.0,
            "ensemble_member": 51,
        },
        {
            "rank": 5,
            "model_type": "forecast",
            "forecast_date": "2021-04-26",
            "date_of_max": "2021-06-07",
            "hdate": None,
            "ensemble_member": 17,
        },
    ],
    "glomma": [
        {
            "rank": 1,
            "model_type": "hindcast",
            "forecast_date": "2022-04-28",
            "date_of_max": "2022-06-02",
            "hdate": 20140428.0,
            "ensemble_member": 2,
        },
        {
            "rank": 2,
            "model_type": "hindcast",
            "forecast_date": "2022-04-21",
            "date_of_max": "2022-05-31",
            "hdate": 20160421.0,
            "ensemble_member": 2,
        },
        {
            "rank": 3,
            "model_type": "hindcast",
            "forecast_date": "2022-04-11",
            "date_of_max": "2022-05-10",
            "hdate": 20150411.0,
            "ensemble_member": 3,
        },
        {
            "rank": 4,
            "model_type": "hindcast",
            "forecast_date": "2021-04-15",
            "date_of_max": "2021-05-29",
            "hdate": 20150415.0,
            "ensemble_member": 8,
        },
        {
            "rank": 5,
            "model_type": "forecast",
            "forecast_date": "2023-04-03",
            "date_of_max": "2023-04-29",
            "hdate": None,
            "ensemble_member": 43,
        },
    ],
}


# =============================================================================
# Metadata helpers
# =============================================================================

def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. "
            f"Valid options are: {valid_names}."
        )

    return CATCHMENTS[catchment_name]


def get_selected_event(catchment_name, event_rank):
    """Return metadata for the selected ranked event."""
    if catchment_name not in TOP_EVENTS:
        raise ValueError(f"No event metadata available for '{catchment_name}'.")

    for event in TOP_EVENTS[catchment_name]:
        if event["rank"] == event_rank:
            return event

    raise ValueError(f"Rank {event_rank} not found for '{catchment_name}'.")


def get_event_dates(event):
    """Return event-relative dates as strings."""
    date_of_max = np.datetime64(event["date_of_max"], "D")

    return [
        str(date_of_max + np.timedelta64(lag, "D"))
        for lag in EVENT_LAGS
    ]


def make_s2s_file(event, variable, grid):
    """Create the expected S2S file path."""
    return (
        S2S_BASE_DIR
        / event["model_type"]
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{grid}_{event['forecast_date']}.nc"
    )


def make_output_filename(catchment_name, event_rank):
    """Create output filename."""
    return f"{PATH_OUT}fig-0X1.png"
    


# =============================================================================
# Coordinate helpers
# =============================================================================

def get_time_coord_name(da):
    """Return the time coordinate name used by the DataArray."""
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""
    lon = da["longitude"] if "longitude" in da.coords else da["lon"]
    lat = da["latitude"] if "latitude" in da.coords else da["lat"]

    return lon, lat


def centers_to_edges(centers):
    """Convert 1D grid-cell center coordinates to grid-cell edges."""
    centers = np.asarray(centers)

    if centers.ndim != 1:
        raise ValueError("centers must be 1D.")
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer edges.")

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# =============================================================================
# Data loading
# =============================================================================

def open_s2s_variable(filename, variable):
    """Open one S2S variable and convert it to plotting units."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        raise KeyError(f"Variable '{variable}' not found in {filename}")

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    elif variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    elif variable == SNOW_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm"

    return ds


def select_event_member(ds, event, variable):
    """Select hindcast date and ensemble member when those dimensions exist."""
    da = ds[variable]

    if event["model_type"] == "hindcast":
        for name in ["hdate", "hindcast_date"]:
            if name in da.dims or name in da.coords:
                da = da.sel({name: event["hdate"]})
                break

    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            da = da.sel({name: event["ensemble_member"]})
            break

    return da


def date_exists(da, target_date):
    """Check whether a target date exists in a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    available_dates = da[time_name].values.astype("datetime64[ns]")

    return target_date in available_dates


def select_date(da, target_date):
    """Select one date from a DataArray and load it into memory."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_daily_variable(event, target_date, variable):
    """
    Load one daily S2S field.

    The script first tries the 0.5-degree file. If the target date is missing,
    it then tries the 0.25-degree file.
    """
    grids_to_try = ["0.5x0.5", "0.25x0.25"]

    for grid in grids_to_try:
        filename = make_s2s_file(event, variable, grid)

        if not filename.exists():
            continue

        ds = open_s2s_variable(filename, variable)

        try:
            da = select_event_member(ds, event, variable)

            if date_exists(da, target_date):
                return select_date(da, target_date)

        finally:
            ds.close()

    raise ValueError(
        f"Could not find {variable} for {target_date} "
        f"in either 0.5x0.5 or 0.25x0.25 files."
    )


def load_precipitation(event, target_date):
    """Load daily precipitation in mm/day."""
    return load_daily_variable(event, target_date, PRECIP_VAR)


def load_msl(event, target_date):
    """Load mean sea level pressure in hPa."""
    return load_daily_variable(event, target_date, MSL_VAR)


def load_snowmelt(event, lag):
    """
    Compute daily SWE change.

    Snowmelt is later plotted where:
        SWE(current day) - SWE(previous day) < 0
    """
    date_of_max = np.datetime64(event["date_of_max"], "D")
    previous_date = date_of_max + np.timedelta64(lag - 1, "D")
    current_date = date_of_max + np.timedelta64(lag, "D")

    da_previous = load_daily_variable(event, str(previous_date), SNOW_VAR)
    da_current = load_daily_variable(event, str(current_date), SNOW_VAR)

    da_change = da_current - da_previous
    da_change.attrs["units"] = "mm"
    da_change.attrs["long_name"] = "Daily change in snow water equivalent"

    return da_change


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load the catchment and keep only the outer boundary."""
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    catchment_path = Path(base_dir) / filename
    gdf = gpd.read_file(catchment_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)

    union_geom = gdf.to_crs(metric_crs).geometry.union_all()

    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)

    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon(
            [Polygon(poly.exterior) for poly in union_geom.geoms]
        )

    else:
        outer_geom = union_geom

    outer_gdf = gpd.GeoDataFrame(
        geometry=[outer_geom],
        crs=metric_crs,
    ).to_crs(plot_crs)

    return outer_gdf.geometry.iloc[0]


# =============================================================================
# Figure setup
# =============================================================================

def make_map_axes():
    """Create a 2 x 2 Lambert Conformal map layout."""
    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        subplot_kw={"projection": proj_map},
        constrained_layout=False,
    )

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, axes, proj_data


def get_plot_axes(axes):
    """Return the four map panels."""
    return list(axes.flat)


# =============================================================================
# Plotting functions
# =============================================================================

def plot_precipitation(ax, da_precip, proj_data):
    """Plot daily precipitation as shaded grid cells."""
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

    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)

    return ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        precip,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_ZERO_THRESHOLD,
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )


def plot_msl_contours(ax, da_msl, proj_data):
    """Plot labelled mean sea level pressure contours."""
    lon, lat = get_lon_lat(da_msl)

    contour = ax.contour(
        lon.values,
        lat.values,
        da_msl.values,
        levels=MSL_CONTOUR_LEVELS,
        colors=MSL_CONTOUR_COLOR,
        linewidths=MSL_CONTOUR_LINEWIDTH,
        transform=proj_data,
        zorder=6,
    )

    ax.clabel(
        contour,
        inline=True,
        inline_spacing=4,
        fontsize=CONTOUR_LABELSIZE,
        fmt="%d",
        colors=MSL_CONTOUR_COLOR,
    )


def plot_snowmelt(ax, da_snowmelt, proj_data):
    """Overlay locations where daily SWE change is negative."""
    lon, lat = get_lon_lat(da_snowmelt)

    snowmelt = np.isfinite(da_snowmelt.values)
    snowmelt &= da_snowmelt.values < SNOWMELT_THRESHOLD

    if SNOWMELT_OVERLAY == "stippling":
        plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data)

    elif SNOWMELT_OVERLAY == "hatching":
        plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data)

    else:
        raise ValueError(
            "SNOWMELT_OVERLAY must be either 'stippling' or 'hatching'."
        )


def plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data):
    """Plot snowmelt as orange stippling."""
    iy, ix = np.where(snowmelt)

    if SNOWMELT_DOT_STRIDE > 1:
        iy = iy[::SNOWMELT_DOT_STRIDE]
        ix = ix[::SNOWMELT_DOT_STRIDE]

    ax.scatter(
        lon.values[ix],
        lat.values[iy],
        s=SNOWMELT_DOT_SIZE,
        c=SNOWMELT_DOT_COLOR,
        alpha=SNOWMELT_DOT_ALPHA,
        transform=proj_data,
        zorder=SNOWMELT_ZORDER,
        linewidths=0,
    )


def plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data):
    """Plot snowmelt as orange hatching."""
    old_hatch_color = plt.rcParams["hatch.color"]
    old_hatch_linewidth = plt.rcParams["hatch.linewidth"]

    try:
        plt.rcParams["hatch.color"] = SNOWMELT_HATCH_COLOR
        plt.rcParams["hatch.linewidth"] = SNOWMELT_HATCH_LINEWIDTH

        ax.contourf(
            lon.values,
            lat.values,
            snowmelt.astype(int),
            levels=[0.5, 1.5],
            colors="none",
            hatches=[SNOWMELT_HATCH_PATTERN],
            transform=proj_data,
            zorder=SNOWMELT_ZORDER,
        )

    finally:
        plt.rcParams["hatch.color"] = old_hatch_color
        plt.rcParams["hatch.linewidth"] = old_hatch_linewidth


def plot_catchment_boundary(ax, geometry, proj_data):
    """Overlay the selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )


def plot_event_panel(ax, event, lag, target_date, catchment_boundary, proj_data):
    """Plot one event-relative panel."""
    da_precip = load_precipitation(event, target_date)
    da_msl = load_msl(event, target_date)
    da_snowmelt = load_snowmelt(event, lag)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_snowmelt(ax, da_snowmelt, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


# =============================================================================
# Figure finishing
# =============================================================================

def add_panel_titles(axes, event_dates):
    """Add panel labels and event-relative dates."""
    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, lag, date in zip(
        axes,
        panel_labels,
        EVENT_LAGS,
        event_dates,
    ):
        ax.set_title(
            f"{panel_label} Day {lag:+d}: {date}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_colorbar(fig, mesh):
    """Add horizontal precipitation colorbar below the panels."""
    cax = fig.add_axes([0.072, 0.1, 0.41, 0.025])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="horizontal",
    )

    cbar.set_label(
        "accumulated total precipitation (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def get_snowmelt_legend_handle():
    """Return legend handle for the selected snowmelt overlay."""
    if SNOWMELT_OVERLAY == "hatching":
        return Patch(
            facecolor="white",
            edgecolor=SNOWMELT_HATCH_COLOR,
            hatch=SNOWMELT_HATCH_PATTERN,
            label=r"Snowmelt ($\Delta$SWE < 0)",
        )

    return Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=SNOWMELT_DOT_COLOR,
        markersize=5,
        label=r"Snowmelt ($\Delta$SWE < 0)",
    )


def add_legend(fig, catchment_label):
    """Add legend below the right-hand panels."""
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CATCHMENT_EDGE_COLOR,
            linewidth=2,
            label=catchment_label,
        ),
        Line2D(
            [0],
            [0],
            color=MSL_CONTOUR_COLOR,
            linewidth=MSL_CONTOUR_LINEWIDTH,
            label="Mean sea level pressure (hPa)",
        ),
        get_snowmelt_legend_handle(),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.725, 0.05),
        frameon=False,
        fontsize=AXIS_LABELSIZE,
        ncol=1,
    )


def finalize_figure(
    fig,
    axes,
    mesh,
    event_dates,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, save the figure, and show it."""
    plot_axes = get_plot_axes(axes)

    add_panel_titles(plot_axes, event_dates)

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.15,
        top=0.95,
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    add_colorbar(fig, mesh)
    add_legend(fig, catchment_label)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Run the full plotting workflow."""
    catchment = get_catchment_settings(CATCHMENT_NAME)
    event = get_selected_event(CATCHMENT_NAME, EVENT_RANK)
    event_dates = get_event_dates(event)
    savepath = make_output_filename(CATCHMENT_NAME, EVENT_RANK)

    catchment_boundary = load_catchment_outer_boundary(
        filename=catchment["filename"],
        base_dir=PATH_CATCHMENT,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, axes, proj_data = make_map_axes()

    mesh = None

    for ax, lag, target_date in zip(get_plot_axes(axes), EVENT_LAGS, event_dates):
        mesh = plot_event_panel(
            ax=ax,
            event=event,
            lag=lag,
            target_date=target_date,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

    finalize_figure(
        fig=fig,
        axes=axes,
        mesh=mesh,
        event_dates=event_dates,
        catchment_label=catchment["label"],
        savepath=savepath,
    )


if __name__ == "__main__":
    main()
