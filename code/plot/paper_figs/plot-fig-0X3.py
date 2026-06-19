#!/usr/bin/env python3
"""
Plot daily precipitation and mean sea level pressure for one S2S ensemble member.

The figure contains four event-relative map panels:
    Day -2 : 2023-08-06
    Day -1 : 2023-08-07
    Day  0 : 2023-08-08
    Day +1 : 2023-08-09

Each panel shows:
1. Daily precipitation as shading.
2. Mean sea level pressure as labelled grey contours.
3. Drammen catchment boundary in red.
4. Drammen city marker.
5. A zoomed inset in panel d.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
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

CATCHMENT_NAME = "drammen"

FORECAST_DATE = "2023-08-05"
ENSEMBLE_MEMBER = 48
MODEL_TYPE = "forecast"
GRID = "0.25x0.25"

DAY_ZERO_DATE = "2023-08-08"
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]
EVENT_LAGS = [-2, -1, 0, 1]

PRECIP_VAR = "tp24"
MSL_VAR = "msl"

WRITE_TO_FILE = True


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

OUTPUT_FILENAME = f"{PATH_OUT}temp.png"



# =============================================================================
# 3. Figure and map settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 9.2

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9

DRAMMEN_LON = 10.2045
DRAMMEN_LAT = 59.7440
DRAMMEN_LABEL = "Drammen"

ZOOM_MAP_EXTENT = [6.5, 11.5, 59, 61.5]

DRAMMEN_MARKER_SIZE = 5
DRAMMEN_MARKER_FACE_COLOR = "yellow"
DRAMMEN_MARKER_EDGE_COLOR = "black"
DRAMMEN_MARKER_EDGE_WIDTH = 0.6


# =============================================================================
# 4. Plot styling
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


# =============================================================================
# 5. Catchment metadata
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
# 6. Helper functions
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


def make_s2s_file(variable):
    """Create S2S forecast file path."""

    return (
        S2S_BASE_DIR
        / MODEL_TYPE
        / "sfc"
        / "daily"
        / "europe"
        / variable
        / f"{variable}_{GRID}_{FORECAST_DATE}.nc"
    )


def get_time_coord_name(da):
    """Return the time coordinate name."""

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


def centers_to_edges(centers):
    """Convert 1D grid-cell centers to grid-cell edges."""

    centers = np.asarray(centers)

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


# =============================================================================
# 7. Data loading
# =============================================================================

def open_s2s_variable(variable):
    """Open one S2S variable and convert to plotting units."""

    filename = make_s2s_file(variable)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if variable == PRECIP_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    return ds


def select_member(da):
    """Select the requested ensemble member."""

    member_name = get_member_coord_name(da)

    return da.sel({member_name: ENSEMBLE_MEMBER})


def select_date(da, target_date):
    """Select one target date and load it into memory."""

    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_daily_variable(variable, target_date):
    """Load one daily S2S field for the selected member and date."""

    ds = open_s2s_variable(variable)

    try:
        da = ds[variable]
        da = select_member(da)
        da = select_date(da, target_date)

    finally:
        ds.close()

    return da


def load_precipitation(target_date):
    """Load daily precipitation in mm/day."""

    return load_daily_variable(PRECIP_VAR, target_date)


def load_msl(target_date):
    """Load mean sea level pressure in hPa."""

    return load_daily_variable(MSL_VAR, target_date)


def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
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
# 8. Figure setup
# =============================================================================

def make_figure_axes():
    """Create four map panels and a right-side colorbar."""

    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[:, 2])

    for ax in axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, axes, cbar_ax, proj_map, proj_data


def get_plot_axes(axes):
    """Return map panels as a flat list."""

    return list(axes.flat)


# =============================================================================
# 9. Plotting functions
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
    """Mark the city of Drammen."""

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


def plot_event_panel(ax, target_date, catchment_boundary, proj_data):
    """Plot one daily map panel."""

    da_precip = load_precipitation(target_date)
    da_msl = load_msl(target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
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
# 10. Figure finishing
# =============================================================================

def add_panel_titles(axes):
    """Add panel labels and event-relative dates."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, lag, date in zip(
        axes,
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
            f"{panel_label} Day {lag:+d}: {formatted_date}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_colorbar(fig, mesh, cbar_ax):
    """Add precipitation colorbar."""

    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.set_label(
        "Daily accumulated precipitation (mm/day)",
        fontsize=AXIS_LABELSIZE,
    )

    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_legend(axes, catchment_label):
    """Add map legend inside the upper-left panel."""

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
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DRAMMEN_MARKER_FACE_COLOR,
            markeredgecolor=DRAMMEN_MARKER_EDGE_COLOR,
            markeredgewidth=DRAMMEN_MARKER_EDGE_WIDTH,
            markersize=6,
            label="City of Drammen",
        ),
    ]

    legend = axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=9,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)


def finalize_figure(
    fig,
    axes,
    cbar_ax,
    proj_map,
    proj_data,
    mesh,
    catchment_boundary,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, inset, save, and show."""

    plot_axes = get_plot_axes(axes)

    add_panel_titles(plot_axes)

    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, catchment_label)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.06,
        top=0.96,
    )

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 11. Main workflow
# =============================================================================

def main():
    """Run full plotting workflow."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    catchment_boundary = load_catchment_outer_boundary(
        filename=catchment["filename"],
        base_dir=PATH_CATCHMENT,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    fig, axes, cbar_ax, proj_map, proj_data = make_figure_axes()

    mesh = None

    for ax, target_date in zip(get_plot_axes(axes), EVENT_DATES):
        mesh = plot_event_panel(
            ax=ax,
            target_date=target_date,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

    finalize_figure(
        fig=fig,
        axes=axes,
        cbar_ax=cbar_ax,
        proj_map=proj_map,
        proj_data=proj_data,
        mesh=mesh,
        catchment_boundary=catchment_boundary,
        catchment_label=catchment["label"],
        savepath=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()
