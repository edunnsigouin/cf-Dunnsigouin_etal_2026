#!/usr/bin/env python3
"""
Plot ERA5 runoff maps and runoff time series near Drammen.

The figure contains:
1. Four ERA5 runoff map panels for 2023-08-06 to 2023-08-09.
2. ERA5 daily runoff as shading.
3. Drammen catchment boundary in red.
4. Drammen city as a yellow marker.
5. A bottom runoff time-series panel at the ERA5 grid point nearest Drammen.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# 1. User-defined input parameters
# =============================================================================

YEAR = 2023
CATCHMENT_NAME = "drammen"

EVENT_LAGS = [-2, -1, 0, 1]
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

RUNOFF_VAR = "ro"
GRID = "0.25x0.25"

TIMESERIES_START = "2023-08-01"
TIMESERIES_END = "2023-08-15"

WRITE_TO_FILE = False


# =============================================================================
# 2. Drammen city and inset settings
# =============================================================================

DRAMMEN_LON = 10.2045
DRAMMEN_LAT = 59.7440
DRAMMEN_LABEL = "Drammen"

ZOOM_MAP_EXTENT = [6, 12, 59, 62] #[6.5, 11.5, 59, 61.5]

DRAMMEN_MARKER_SIZE = 5
DRAMMEN_MARKER_FACE_COLOR = "yellow"
DRAMMEN_MARKER_EDGE_COLOR = "black"
DRAMMEN_MARKER_EDGE_WIDTH = 0.6


# =============================================================================
# 3. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]
PATH_ERA5 = Path(config.dirs["era5_continuous_daily_scandinavia"])

RUNOFF_FILE = PATH_ERA5 / RUNOFF_VAR / f"{RUNOFF_VAR}_{GRID}_{YEAR}.nc"

OUTPUT_FILENAME = (
    f"{PATH_OUT}"
    f"era5_runoff_drammen_{GRID}_{EVENT_DATES[0]}_{EVENT_DATES[-1]}.png"
)


# =============================================================================
# 4. Figure settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 11.2

MAP_EXTENT = [4.75, 12.75, 58.0, 63.0] #[-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 9


# =============================================================================
# 5. Plot styling
# =============================================================================

RUNOFF_LEVELS = np.arange(1, 31, 2)
RUNOFF_ZERO_THRESHOLD = 1.0
RUNOFF_CMAP = plt.get_cmap("Blues").copy()
RUNOFF_CMAP.set_under("white")

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

RUNOFF_LINE_COLOR = "tab:blue"
RUNOFF_LINEWIDTH = 2.0
EVENT_MARKER_SIZE = 35


# =============================================================================
# 6. Catchment metadata
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


# =============================================================================
# 7. Helper functions
# =============================================================================

def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""

    if catchment_name not in CATCHMENTS:
        raise ValueError(f"Unknown catchment: {catchment_name}")

    return CATCHMENTS[catchment_name]


def get_time_coord_name(da):
    """Return time coordinate name."""

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
    """Convert 1D grid-cell centers to grid-cell edges."""

    centers = np.asarray(centers)

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# =============================================================================
# 8. Data loading
# =============================================================================

def open_era5_runoff(filename):
    """
    Open ERA5 runoff.

    Assumes runoff is already in mm/day unless metadata clearly says metres.
    """

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if RUNOFF_VAR not in ds:
        raise KeyError(
            f"Variable '{RUNOFF_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    units = str(ds[RUNOFF_VAR].attrs.get("units", "")).strip().lower()

    if units in {"m", "meter", "metre", "m/day", "m d-1", "m/day"}:
        ds[RUNOFF_VAR] = ds[RUNOFF_VAR] * 1000.0
        ds[RUNOFF_VAR].attrs["units"] = "mm/day"

    else:
        ds[RUNOFF_VAR].attrs["units"] = "mm/day"

    return ds


def select_date(da, target_date):
    """Select one date from a DataArray."""

    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_runoff_map(ds_ro, target_date):
    """Load ERA5 runoff map for one date."""

    return select_date(ds_ro[RUNOFF_VAR], target_date)


def load_runoff_timeseries_nearest_drammen(ds_ro):
    """Load runoff time series nearest Drammen city."""

    da = ds_ro[RUNOFF_VAR]
    lon, lat = get_lon_lat(da)

    ts = da.sel(
        {
            lon.name: DRAMMEN_LON,
            lat.name: DRAMMEN_LAT,
        },
        method="nearest",
    )

    ts = ts.sel(time=slice(TIMESERIES_START, TIMESERIES_END)).load()

    ts.attrs["selected_lon"] = float(ts[lon.name].values)
    ts.attrs["selected_lat"] = float(ts[lat.name].values)

    return ts


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load catchment and keep only the outer boundary."""

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
# 9. Figure setup
# =============================================================================

def make_figure_axes():
    """Create four map panels, colorbar axis, and bottom time-series panel."""

    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    gs = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 0.45],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[0:2, 2])
    ts_ax = fig.add_subplot(gs[2, 0:2])

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, axes, ts_ax, cbar_ax, proj_map, proj_data


def get_plot_axes(axes):
    """Return four map panels as a flat list."""

    return list(axes.flat)


# =============================================================================
# 10. Map plotting functions
# =============================================================================

def plot_runoff_map(ax, da_runoff, proj_data):
    """Plot ERA5 daily runoff as shaded grid cells."""

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


def plot_catchment_boundary(ax, geometry, proj_data, linewidth=CATCHMENT_LINEWIDTH):
    """Overlay catchment boundary."""

    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=linewidth,
        zorder=9,
    )


def plot_drammen_city(ax, proj_data):
    """Mark Drammen city."""

    ax.plot(
        DRAMMEN_LON,
        DRAMMEN_LAT,
        marker="o",
        markersize=DRAMMEN_MARKER_SIZE,
        markeredgecolor=DRAMMEN_MARKER_EDGE_COLOR,
        markeredgewidth=DRAMMEN_MARKER_EDGE_WIDTH,
        markerfacecolor=DRAMMEN_MARKER_FACE_COLOR,
        linestyle="none",
        transform=proj_data,
        zorder=12,
    )


def plot_event_panel(ax, ds_ro, target_date, catchment_boundary, proj_data):
    """Plot one runoff map panel."""

    da_runoff = load_runoff_map(ds_ro, target_date)

    mesh = plot_runoff_map(ax, da_runoff, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    plot_drammen_city(ax, proj_data)

    return mesh


def add_zoom_inset(parent_ax, proj_map, proj_data, catchment_boundary):
    """Add zoomed inset map to panel d."""

    inset_ax = parent_ax.inset_axes(
        [0.01, 0.01, 0.3, 0.3],
        projection=proj_map,
        zorder=20,
    )

    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(1.0)
    inset_ax.set_extent(ZOOM_MAP_EXTENT, crs=proj_data)

    inset_ax.coastlines(
        resolution="10m",
        linewidth=0.4,
        color="black",
        zorder=2,
    )

    plot_catchment_boundary(
        inset_ax,
        catchment_boundary,
        proj_data,
        linewidth=1.0,
    )

    plot_drammen_city(inset_ax, proj_data)

    txt = inset_ax.text(
        DRAMMEN_LON,
        DRAMMEN_LAT + 0.06,
        DRAMMEN_LABEL,
        fontsize=7,
        color="yellow",
        fontweight="bold",
        ha="right",
        va="bottom",
        transform=proj_data,
        zorder=13,
    )

    txt.set_path_effects(
        [
            pe.Stroke(linewidth=1.5, foreground="black"),
            pe.Normal(),
        ]
    )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    return inset_ax


# =============================================================================
# 11. Time-series plotting
# =============================================================================

def plot_runoff_timeseries(ts_ax, ds_ro):
    """Plot ERA5 runoff time series nearest Drammen."""

    da_event = load_runoff_timeseries_nearest_drammen(ds_ro)
    time_name = get_time_coord_name(da_event)

    ts_ax.plot(
        da_event[time_name].values,
        da_event.values,
        color=RUNOFF_LINE_COLOR,
        linewidth=RUNOFF_LINEWIDTH,
        label="ERA5 runoff",
    )

    for date in EVENT_DATES:
        value = da_event.sel({time_name: np.datetime64(date)}, method="nearest")

        ts_ax.scatter(
            value[time_name].values,
            value.values,
            color=RUNOFF_LINE_COLOR,
            s=EVENT_MARKER_SIZE,
            zorder=5,
        )

    ts_ax.set_title(
        f"e) ERA5 runoff nearest {DRAMMEN_LABEL}",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel("Runoff (mm/day)", fontsize=AXIS_LABELSIZE)
    ts_ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(labelsize=TICK_LABELSIZE)

    ts_ax.set_xlim(
        np.datetime64(TIMESERIES_START),
        np.datetime64(TIMESERIES_END),
    )
    ts_ax.margins(x=0)

    tick_interval_days = 2

    tick_dates = np.arange(
        np.datetime64(TIMESERIES_START),
        np.datetime64(TIMESERIES_END) + np.timedelta64(1, "D"),
        np.timedelta64(tick_interval_days, "D"),
    )

    ts_ax.set_xticks(tick_dates)

    ts_ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    plt.setp(
        ts_ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    ts_ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)


# =============================================================================
# 12. Figure finishing
# =============================================================================

def add_panel_titles(axes):
    """Add panel labels and event-relative dates."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, lag, date in zip(
        get_plot_axes(axes),
        panel_labels,
        EVENT_LAGS,
        EVENT_DATES,
    ):
        formatted_date = (
            np.datetime64(date)
            .astype("datetime64[D]")
            .astype(object)
            .strftime("%B %-d")
        )

        ax.set_title(
            f"{panel_label} Day {lag:+d}: {formatted_date} {YEAR}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_colorbar(fig, mesh, cbar_ax):
    """Add vertical runoff colorbar beside map panels."""

    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.set_label(
        "Daily runoff (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )

    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_legend(axes, catchment_label):
    """Add map legend inside panel a."""

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
            markerfacecolor=DRAMMEN_MARKER_FACE_COLOR,
            markeredgecolor=DRAMMEN_MARKER_EDGE_COLOR,
            markeredgewidth=DRAMMEN_MARKER_EDGE_WIDTH,
            markersize=6,
            label=f"City of {DRAMMEN_LABEL}",
        ),
    ]

    legend = axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)


def align_timeseries_axis_to_map_panels(fig, axes, ts_ax):
    """Align panel e with the combined left/right borders of panels a-d."""

    fig.canvas.draw()

    left = min(
        axes[0, 0].get_position().x0,
        axes[1, 0].get_position().x0,
    )

    right = max(
        axes[0, 1].get_position().x1,
        axes[1, 1].get_position().x1,
    )

    pos = ts_ax.get_position()

    ts_ax.set_position(
        [
            left,
            pos.y0,
            right - left,
            pos.height,
        ]
    )


def finalize_figure(
    fig,
    axes,
    ts_ax,
    cbar_ax,
    proj_map,
    proj_data,
    mesh,
    ds_ro,
    catchment_boundary,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, inset, time series, save, and show."""

    add_panel_titles(axes)

    plot_runoff_timeseries(ts_ax, ds_ro)

    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, catchment_label)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.075,
        top=0.96,
    )

    align_timeseries_axis_to_map_panels(fig, axes, ts_ax)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 13. Main workflow
# =============================================================================

def main():
    """Run full plotting workflow."""

    catchment = get_catchment_settings(CATCHMENT_NAME)
    savepath = OUTPUT_FILENAME

    ds_ro = open_era5_runoff(RUNOFF_FILE)

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        fig, axes, ts_ax, cbar_ax, proj_map, proj_data = make_figure_axes()

        mesh = None

        for ax, target_date in zip(get_plot_axes(axes), EVENT_DATES):
            mesh = plot_event_panel(
                ax=ax,
                ds_ro=ds_ro,
                target_date=target_date,
                catchment_boundary=catchment_boundary,
                proj_data=proj_data,
            )

        finalize_figure(
            fig=fig,
            axes=axes,
            ts_ax=ts_ax,
            cbar_ax=cbar_ax,
            proj_map=proj_map,
            proj_data=proj_data,
            mesh=mesh,
            ds_ro=ds_ro,
            catchment_boundary=catchment_boundary,
            catchment_label=catchment["label"],
            savepath=savepath,
        )

    finally:
        ds_ro.close()


if __name__ == "__main__":
    main()
