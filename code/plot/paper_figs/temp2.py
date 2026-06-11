#!/usr/bin/env python3
"""
Plot ERA5 daily precipitation and mean sea level pressure during Storm Hans.

The script:
1. Plots four fixed event-relative dates from 2023-08-06 to 2023-08-09.
2. Shows daily ERA5 precipitation as shading.
3. Shows ERA5 mean sea level pressure as labelled grey contours.
4. Overlays a selected catchment boundary.
5. Adds a zoomed inset map in the bottom-right panel showing the catchment
   boundary and two station locations with labels.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon
import cartopy.feature as cfeature

from Dunnsigouin_etal_2026 import config


# =============================================================================
# Settings
# =============================================================================

YEAR = 2023

EVENT_LAGS = [-2, -1, 0, 1]
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"

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

STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
    {"name": "Ål III", "lon": 8.5609, "lat": 60.6391},
]

PRECIP_VAR = "tp24"
MSL_VAR = "msl"

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]
PATH_ERA5 = Path(config.dirs["era5_continuous_daily"])

PRECIP_FILE = PATH_ERA5 / PRECIP_VAR / f"{PRECIP_VAR}_0.5x0.5_{YEAR}.nc"
MSL_FILE = PATH_ERA5 / MSL_VAR / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"

CATCHMENT_CRS_IF_MISSING = "EPSG:4326"
CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0

FIG_WIDTH_IN = 9
FIG_HEIGHT_IN = 10

MAP_WSPACE = 0.0
MAP_HSPACE = 0.08
MAP_EXTENT = [-10, 25, 50, 70]
ZOOM_MAP_EXTENT = [6, 12, 59, 62]

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9

PRECIP_LEVELS = np.arange(5, 55, 5)
PRECIP_ZERO_THRESHOLD = 5.0
PRECIP_CMAP = plt.get_cmap("GnBu").copy()
PRECIP_CMAP.set_under("white")

MSL_CONTOUR_LEVELS = np.arange(975, 1045, 5)
MSL_CONTOUR_COLOR = "0.7"
MSL_CONTOUR_LINEWIDTH = 1.5

STATION_MARKER_SIZE = 5
STATION_MARKER_FACE_COLOR = "yellow"
STATION_MARKER_EDGE_COLOR = "black"
STATION_MARKER_EDGE_WIDTH = 0.6

WRITE_TO_FILE = True


# =============================================================================
# Settings helpers
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


def make_output_filename(catchment_name):
    """Create output filename for the selected catchment."""
    return (
        f"{PATH_OUT}xy-hans-evolution-era5-tp-msl-"
        f"{catchment_name}-"
        f"{EVENT_DATES[0]}-{EVENT_DATES[-1]}-2x2-inset.png"
    )


# =============================================================================
# Coordinate helpers
# =============================================================================

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


def centers_to_edges(centers):
    """Convert 1D grid-cell centers to grid-cell edges."""
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

def open_era5_variable(filename, variable):
    """Open one ERA5 variable and convert to plotting units."""
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

    return ds


def select_date(da, target_date):
    """Select one date from a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")
    return da.sel({time_name: target_date}).load()


def load_precipitation(ds_tp, target_date):
    """Load daily precipitation for one date."""
    return select_date(ds_tp[PRECIP_VAR], target_date)


def load_msl(ds_msl, target_date):
    """Load mean sea level pressure for one date."""
    return select_date(ds_msl[MSL_VAR], target_date)


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load the catchment and keep only its outer boundary."""
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

def make_map_axes(central_lon, central_lat, extent):
    """Create the 2 x 2 Lambert Conformal map layout."""
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
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
        ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_map, proj_data


def get_event_axes(axes):
    """Return the four map panels used for the event dates."""
    return list(axes.flat)


# =============================================================================
# Plotting
# =============================================================================

def plot_precipitation(ax, da_precip, proj_data):
    """Plot precipitation as shading."""
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
    """Overlay the selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=linewidth,
        zorder=9,
    )


def plot_stations(ax, stations, proj_data, fontsize=8):
    """Overlay station locations and names."""
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=STATION_MARKER_SIZE,
            markeredgecolor=STATION_MARKER_EDGE_COLOR,
            markeredgewidth=STATION_MARKER_EDGE_WIDTH,
            markerfacecolor=STATION_MARKER_FACE_COLOR,
            linestyle="none",
            transform=proj_data,
            zorder=12,
        )

        if station["name"] == "Bergheim":
            dx, dy = 0.05, 0.075
            ha = "left"
        elif station["name"] == "Ål III":
            dx, dy = -0.075, 0.1
            ha = "right"
        else:
            dx, dy = 0.05, 0.05
            ha = "left"

        txt = ax.text(
            station["lon"] + dx,
            station["lat"] + dy,
            station["name"],
            fontsize=fontsize,
            color="yellow",
            fontweight="bold",
            ha=ha,
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



def plot_event_panel(ax, ds_tp, ds_msl, catchment_boundary, target_date, proj_data):
    """Plot precipitation, pressure, and catchment boundary for one date."""
    da_precip = load_precipitation(ds_tp, target_date)
    da_msl = load_msl(ds_msl, target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


def add_zoom_inset(parent_ax, proj_map, proj_data, catchment_boundary, stations):
    """Add zoomed inset map to the lower-right corner of a parent axis."""
    inset_ax = parent_ax.inset_axes(
        [0.025, 0.025, 0.35, 0.35],
        projection=proj_map,
        zorder=20,
    )

    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(1.0)
    inset_ax.patch.set_zorder(20)

    inset_ax.set_extent(ZOOM_MAP_EXTENT, crs=proj_data)

    # Coastline
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
    plot_stations(
        inset_ax,
        stations,
        proj_data,
        fontsize=6,
    )

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")
        spine.set_zorder(30)

    return inset_ax


# =============================================================================
# Figure finishing
# =============================================================================

def add_panel_titles(axes):
    """Add panel labels and date titles."""
    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, label, lag, date in zip(
        get_event_axes(axes),
        panel_labels,
        EVENT_LAGS,
        EVENT_DATES,
    ):
        ax.set_title(
            f"{label} Day {lag:+d}: {date}",
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
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATION_MARKER_FACE_COLOR,
            markeredgecolor=STATION_MARKER_EDGE_COLOR,
            markersize=7,
            label="Stations",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.725, 0.055),
        frameon=False,
        fontsize=AXIS_LABELSIZE,
        ncol=1,
    )


def finalize_figure(
    fig,
    axes,
    mesh,
    catchment_label,
    savepath,
    write_to_file,
):
    """Add final figure elements, save, and show."""
    add_panel_titles(axes)

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

    if write_to_file:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the full plotting workflow."""
    catchment = get_catchment_settings(CATCHMENT_NAME)
    savepath = make_output_filename(CATCHMENT_NAME)

    ds_tp = open_era5_variable(PRECIP_FILE, PRECIP_VAR)
    ds_msl = open_era5_variable(MSL_FILE, MSL_VAR)

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        fig, axes, proj_map, proj_data = make_map_axes(
            central_lon=CENTRAL_LON,
            central_lat=CENTRAL_LAT,
            extent=MAP_EXTENT,
        )

        mesh = None

        for ax, target_date in zip(get_event_axes(axes), EVENT_DATES):
            mesh = plot_event_panel(
                ax=ax,
                ds_tp=ds_tp,
                ds_msl=ds_msl,
                catchment_boundary=catchment_boundary,
                target_date=target_date,
                proj_data=proj_data,
            )

        add_zoom_inset(
            parent_ax=axes[1, 1],
            proj_map=proj_map,
            proj_data=proj_data,
            catchment_boundary=catchment_boundary,
            stations=STATIONS,
        )

        finalize_figure(
            fig=fig,
            axes=axes,
            mesh=mesh,
            catchment_label=catchment["label"],
            savepath=savepath,
            write_to_file=WRITE_TO_FILE,
        )

    finally:
        ds_tp.close()
        ds_msl.close()


if __name__ == "__main__":
    main()
