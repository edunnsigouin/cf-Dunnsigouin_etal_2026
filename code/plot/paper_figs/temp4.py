#!/usr/bin/env python3
"""
Plot one high-ranking S2S runoff event.

The figure contains:
1. Four event-relative map panels: -2, -1, 0, +1.
2. Daily runoff as shading.
3. Selected catchment boundary in red.
4. Drammen city location as a yellow circle with text label.
5. A bottom time-series panel showing:
   - the selected runoff event at the grid point nearest Drammen
   - the full min-max range from all other hindcast years and ensemble members
   - the median of all other hindcast years and ensemble members.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"
EVENT_RANK = 1              # options: 1-5

EVENT_LAGS = [-2, -1, 0, 1]

RUNOFF_VAR = "ro24"

WRITE_TO_FILE = True


# =============================================================================
# City marker settings
# =============================================================================

DRAMMEN_LON = 10.2045
DRAMMEN_LAT = 59.7440
DRAMMEN_LABEL = "Drammen"


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
FIG_HEIGHT_IN = 12

MAP_EXTENT = [6, 12, 59, 62]
MAP_WSPACE = 0.0
MAP_HSPACE = 0.08

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CITY_LABEL_FONTSIZE = 10


# =============================================================================
# Plot styling
# =============================================================================

RUNOFF_LEVELS = np.arange(5, 35, 5)
RUNOFF_ZERO_THRESHOLD = 5.0
RUNOFF_CMAP = plt.get_cmap("GnBu").copy()
RUNOFF_CMAP.set_under("white")

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

CITY_MARKER_COLOR = "yellow"
CITY_MARKER_EDGE_COLOR = "black"
CITY_MARKER_SIZE = 55

EVENT_LINE_COLOR = "tab:blue"
EVENT_LINEWIDTH = 2.0

RANGE_FILL_COLOR = "tab:blue"
RANGE_FILL_ALPHA = 0.20

MEDIAN_LINE_COLOR = "tab:red"
MEDIAN_LINEWIDTH = 2.0

EVENT_DATE_LINEWIDTH = 1.0
EVENT_DATE_ALPHA = 0.6


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
    return f"{PATH_OUT}test.png"


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


def get_member_coord_name(da):
    """Return ensemble member coordinate name."""
    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify ensemble member coordinate.")


def get_hdate_coord_name(da):
    """Return hindcast date coordinate name."""
    for name in ["hdate", "hindcast_date"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify hindcast date coordinate.")


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

    if variable == RUNOFF_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

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


def load_runoff(event, target_date):
    """Load daily runoff in mm/day."""
    return load_daily_variable(event, target_date, RUNOFF_VAR)


def load_runoff_timeseries_at_drammen(event):
    """
    Load the selected runoff event time series at the grid point nearest Drammen.
    """
    grids_to_try = ["0.5x0.5", "0.25x0.25"]

    for grid in grids_to_try:
        filename = make_s2s_file(event, RUNOFF_VAR, grid)

        if not filename.exists():
            continue

        ds = open_s2s_variable(filename, RUNOFF_VAR)

        try:
            da = select_event_member(ds, event, RUNOFF_VAR)
            lon, lat = get_lon_lat(da)

            da_point = da.sel(
                {
                    lon.name: DRAMMEN_LON,
                    lat.name: DRAMMEN_LAT,
                },
                method="nearest",
            ).load()

            da_point.attrs["selected_grid"] = grid
            da_point.attrs["selected_lon"] = float(da_point[lon.name].values)
            da_point.attrs["selected_lat"] = float(da_point[lat.name].values)

            return da_point

        finally:
            ds.close()

    raise FileNotFoundError(
        f"Could not find selected {RUNOFF_VAR} time series "
        f"for {event['forecast_date']}."
    )


def load_runoff_hindcast_member_stats_at_drammen(event):
    """
    Load min, max, and median runoff at the grid point nearest Drammen.

    Statistics are computed from all hindcast dates and ensemble members
    for the same forecast initialization date, excluding the selected event.
    """
    if event["model_type"] != "hindcast":
        raise ValueError(
            "The hindcast/member statistics are only defined for hindcast events."
        )

    grids_to_try = ["0.5x0.5", "0.25x0.25"]

    for grid in grids_to_try:
        filename = make_s2s_file(event, RUNOFF_VAR, grid)

        if not filename.exists():
            continue

        ds = open_s2s_variable(filename, RUNOFF_VAR)

        try:
            da = ds[RUNOFF_VAR]
            lon, lat = get_lon_lat(da)

            da_point = da.sel(
                {
                    lon.name: DRAMMEN_LON,
                    lat.name: DRAMMEN_LAT,
                },
                method="nearest",
            )

            hdate_name = get_hdate_coord_name(da_point)
            member_name = get_member_coord_name(da_point)
            time_name = get_time_coord_name(da_point)

            hdate_values = da_point[hdate_name]
            member_values = da_point[member_name]

            selected_hdate = event["hdate"]
            selected_member = event["ensemble_member"]

            is_selected_hdate = hdate_values == selected_hdate
            is_selected_member = member_values == selected_member

            keep_mask = ~(
                is_selected_hdate.broadcast_like(da_point)
                & is_selected_member.broadcast_like(da_point)
            )

            da_others = da_point.where(keep_mask)

            sample_dims = [hdate_name, member_name]
            n_samples_total = da_point.sizes[hdate_name] * da_point.sizes[member_name]
            n_samples_used = int(keep_mask.any(time_name).sum().values)

            da_min = da_others.min(sample_dims, skipna=True).load()
            da_max = da_others.max(sample_dims, skipna=True).load()
            da_median = da_others.median(sample_dims, skipna=True).load()

            attrs = {
                "selected_grid": grid,
                "selected_lon": float(da_point[lon.name].values),
                "selected_lat": float(da_point[lat.name].values),
                "n_samples_total": n_samples_total,
                "n_samples_used": n_samples_used,
            }

            da_min.attrs.update(attrs)
            da_max.attrs.update(attrs)
            da_median.attrs.update(attrs)

            return da_min, da_max, da_median

        finally:
            ds.close()

    raise FileNotFoundError(
        f"Could not find hindcast/member statistics file "
        f"for {event['forecast_date']}."
    )


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

def make_figure_axes():
    """Create four map panels and one full-width time-series panel."""
    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.55],
        hspace=MAP_HSPACE,
        wspace=MAP_WSPACE,
    )

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    ts_ax = fig.add_subplot(gs[2, :])

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, axes, ts_ax, proj_data


def get_plot_axes(axes):
    """Return the four map panels."""
    return list(axes.flat)


# =============================================================================
# Plotting functions
# =============================================================================

def plot_runoff(ax, da_runoff, proj_data):
    """Plot daily runoff as shaded grid cells."""
    lon, lat = get_lon_lat(da_runoff)
    runoff = da_runoff.values

    lon_edges = centers_to_edges(lon.values)
    lat_edges = centers_to_edges(lat.values)

    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        runoff = runoff[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        runoff = runoff[:, ::-1]

    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)

    return ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        runoff,
        cmap=RUNOFF_CMAP,
        vmin=RUNOFF_ZERO_THRESHOLD,
        vmax=RUNOFF_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )


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


def plot_drammen_location(ax, proj_data):
    """Plot Drammen city location and label."""
    ax.scatter(
        DRAMMEN_LON,
        DRAMMEN_LAT,
        s=CITY_MARKER_SIZE,
        marker="o",
        facecolor=CITY_MARKER_COLOR,
        edgecolor=CITY_MARKER_EDGE_COLOR,
        linewidth=0.8,
        transform=proj_data,
        zorder=10,
    )

def plot_event_panel(ax, event, lag, target_date, catchment_boundary, proj_data):
    """Plot one event-relative runoff panel."""
    da_runoff = load_runoff(event, target_date)

    mesh = plot_runoff(ax, da_runoff, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    plot_drammen_location(ax, proj_data)

    return mesh


def plot_runoff_timeseries(ts_ax, event, event_dates):
    """Plot selected runoff time series and hindcast/member statistics."""
    da_event = load_runoff_timeseries_at_drammen(event)
    da_min, da_max, da_median = load_runoff_hindcast_member_stats_at_drammen(event)

    time_name = get_time_coord_name(da_event)

    ts_ax.fill_between(
        da_min[time_name].values,
        da_min.values,
        da_max.values,
        color=RANGE_FILL_COLOR,
        alpha=RANGE_FILL_ALPHA,
        linewidth=0,
        label=(
            "Full range, excluding selected event "
            f"(n={da_min.attrs['n_samples_used']})"
        ),
    )

    ts_ax.plot(
        da_median[time_name].values,
        da_median.values,
        color=MEDIAN_LINE_COLOR,
        linewidth=MEDIAN_LINEWIDTH,
        label="Median of hindcast members",
    )

    ts_ax.plot(
        da_event[time_name].values,
        da_event.values,
        color=EVENT_LINE_COLOR,
        linewidth=EVENT_LINEWIDTH,
        label="Selected event",
    )

    for date in event_dates:
        ts_ax.axvline(
            np.datetime64(date),
            color="0.4",
            linewidth=EVENT_DATE_LINEWIDTH,
            alpha=EVENT_DATE_ALPHA,
            linestyle="--",
        )

        value = da_event.sel({time_name: np.datetime64(date)}, method="nearest")

        ts_ax.scatter(
            value[time_name].values,
            value.values,
            color=EVENT_LINE_COLOR,
            s=35,
            zorder=5,
        )

    selected_lon = da_event.attrs["selected_lon"]
    selected_lat = da_event.attrs["selected_lat"]
    selected_grid = da_event.attrs["selected_grid"]

    ts_ax.set_title(
        (
            f"e) Runoff at grid point nearest {DRAMMEN_LABEL} "
            f"({selected_lat:.2f}°N, {selected_lon:.2f}°E; {selected_grid})"
        ),
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel("Runoff (mm/day)", fontsize=AXIS_LABELSIZE)
    ts_ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(labelsize=TICK_LABELSIZE)

    plt.setp(
        ts_ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    ts_ax.grid(True, linewidth=0.5, alpha=0.4)
    ts_ax.legend(frameon=False, fontsize=AXIS_LABELSIZE)


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
    """Add vertical runoff colorbar on the right side of the map panels."""
    cax = fig.add_axes([0.93, 0.39, 0.02, 0.50])

    cbar = fig.colorbar(
        mesh,
        cax=cax,
        orientation="vertical",
    )

    cbar.set_label(
        "Runoff (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_legend(axes, catchment_label):
    """Add legend inside the upper-left map panel."""

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
            marker="o",
            color="none",
            markerfacecolor=CITY_MARKER_COLOR,
            markeredgecolor=CITY_MARKER_EDGE_COLOR,
            markersize=8,
            label="city of Drammen",
        ),
    ]

    axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=AXIS_LABELSIZE,
    )


def finalize_figure(
    fig,
    axes,
    ts_ax,
    mesh,
    event,
    event_dates,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, save the figure, and show it."""
    plot_axes = get_plot_axes(axes)

    add_panel_titles(plot_axes, event_dates)
    plot_runoff_timeseries(ts_ax, event, event_dates)

    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.08,
        top=0.95,
    )

    add_colorbar(fig, mesh)
    add_legend(axes, catchment_label)

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

    fig, axes, ts_ax, proj_data = make_figure_axes()

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
        ts_ax=ts_ax,
        mesh=mesh,
        event=event,
        event_dates=event_dates,
        catchment_label=catchment["label"],
        savepath=savepath,
    )


if __name__ == "__main__":
    main()
