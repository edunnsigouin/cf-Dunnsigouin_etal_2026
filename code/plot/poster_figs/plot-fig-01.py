#!/usr/bin/env python3
"""Plot Storm Hans precipitation and pressure maps for four selected dates.

Panels a-d show precipitation shading, ERA5 mean sea level pressure contours,
and the selected NVE catchment boundary. The precipitation reference can be
seNorge or ERA5; pressure remains ERA5 in both cases.

The figure geometry preserves the physical size and spacing of panels a-d and
the precipitation colorbar from the earlier five-panel layout, while removing
panel e and the Bergheim station marker.
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
# 1. USER SETTINGS
# =============================================================================
YEAR = 2023
CATCHMENT_NAME = "drammen"
REFERENCE_DATASET = "senorge"  # "senorge" or "era5"
EVENT_DATES = ["2023-08-06", "2023-08-07", "2023-08-08", "2023-08-09"]
WRITE_TO_FILE = True

# =============================================================================
# 2. VARIABLES AND PATHS
# =============================================================================
SENORGE_PRECIP_VAR = "rr"
ERA5_PRECIP_VAR = "tp24"
MSL_VAR = "msl"

PATH_OUT = Path(config.dirs["fig"]) / 'poster_figs/'
PATH_CATCHMENT = Path(config.dirs["nve"])
PATH_SENORGE = Path(config.dirs["senorge_continuous_daily"])
PATH_ERA5 = Path(config.dirs["era5_continuous_daily"])

SENORGE_PRECIP_FILE = PATH_SENORGE / SENORGE_PRECIP_VAR / f"{SENORGE_PRECIP_VAR}_{YEAR}.nc"
ERA5_PRECIP_FILE = PATH_ERA5 / ERA5_PRECIP_VAR / f"{ERA5_PRECIP_VAR}_0.5x0.5_{YEAR}.nc"
MSL_FILE = PATH_ERA5 / MSL_VAR / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"
OUTPUT_FILE = PATH_OUT / f"fig-01_{REFERENCE_DATASET}.png"

# =============================================================================
# 3. FIGURE SETTINGS
# =============================================================================
FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 9.183464285714285
MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.08166666666666667
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9
LEGEND_FONTSIZE = 9

# Preserve the original physical top and bottom margins after removing panel e.
FIG_LEFT = 0.065
FIG_RIGHT = 0.96
FIG_BOTTOM_IN = 0.84
FIG_TOP_IN = 0.448

# =============================================================================
# 4. PLOT STYLING
# =============================================================================
PRECIP_LEVELS = np.arange(5, 65, 5)
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
# 5. CATCHMENT METADATA
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


def validate_user_settings():
    """Check user settings before opening files."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if len(EVENT_DATES) != 4:
        raise ValueError("EVENT_DATES must contain exactly four dates.")


def get_reference_name():
    """Return the publication-style precipitation reference name."""
    return {"senorge": "seNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_precip_variable():
    """Return the precipitation variable for the selected reference."""
    return {"senorge": SENORGE_PRECIP_VAR, "era5": ERA5_PRECIP_VAR}[REFERENCE_DATASET]


def get_precip_filename():
    """Return the yearly precipitation file for the selected reference."""
    return {"senorge": SENORGE_PRECIP_FILE, "era5": ERA5_PRECIP_FILE}[REFERENCE_DATASET]


def get_catchment_settings(catchment_name):
    """Return metadata for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(f"Unknown catchment '{catchment_name}'. Valid options are: {valid_names}.")
    return CATCHMENTS[catchment_name]


def get_time_coord_name(da):
    """Return the time-coordinate name used by a DataArray."""
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name
    raise ValueError("Could not identify time coordinate.")


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""
    if "longitude" in da.coords:
        lon = da["longitude"]
    elif "lon" in da.coords:
        lon = da["lon"]
    else:
        raise KeyError("Could not find longitude coordinate.")

    if "latitude" in da.coords:
        lat = da["latitude"]
    elif "lat" in da.coords:
        lat = da["lat"]
    else:
        raise KeyError("Could not find latitude coordinate.")
    return lon, lat


def centers_to_edges(centers):
    """Convert one-dimensional grid-cell centres to grid-cell edges."""
    centers = np.asarray(centers)
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional.")
    if centers.size < 2:
        raise ValueError("At least two grid-cell centres are required.")

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def check_file_exists(filename):
    """Raise a clear error if an input file is missing."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")


def open_precipitation():
    """Open precipitation for the selected reference dataset."""
    filename = get_precip_filename()
    variable = get_precip_variable()
    check_file_exists(filename)

    ds = xr.decode_cf(xr.open_dataset(filename))
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if REFERENCE_DATASET == "senorge":
        if "time" not in ds.coords:
            raise KeyError("The seNorge precipitation file must contain a 'time' coordinate.")
        ds = ds.assign_coords(time=ds.time - np.timedelta64(6, "h") - np.timedelta64(24, "h"))
    else:
        ds[variable] = ds[variable] * 1000.0

    ds[variable].attrs["units"] = "mm/day"
    return ds


def open_era5_msl(filename):
    """Open ERA5 mean sea level pressure and convert Pa to hPa."""
    check_file_exists(filename)
    ds = xr.open_dataset(filename)
    if MSL_VAR not in ds:
        raise KeyError(
            f"Variable '{MSL_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    ds[MSL_VAR] = ds[MSL_VAR] / 100.0
    ds[MSL_VAR].attrs["units"] = "hPa"
    return ds


def select_date(da, target_date):
    """Select one date from a DataArray."""
    time_name = get_time_coord_name(da)
    return da.sel({time_name: np.datetime64(target_date, "ns")}).load()


def load_precipitation(ds_precip, target_date):
    """Load selected-reference precipitation for one date."""
    return select_date(ds_precip[get_precip_variable()], target_date)


def load_msl(ds_msl, target_date):
    """Load ERA5 pressure for one date."""
    return select_date(ds_msl[MSL_VAR], target_date)


def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load a catchment polygon and keep only its outer boundary."""
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"
    catchment_path = Path(base_dir) / filename
    check_file_exists(catchment_path)

    gdf = gpd.read_file(catchment_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)

    union_geom = gdf.to_crs(metric_crs).geometry.union_all()
    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)
    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon([Polygon(polygon.exterior) for polygon in union_geom.geoms])
    else:
        outer_geom = union_geom

    outer_gdf = gpd.GeoDataFrame(geometry=[outer_geom], crs=metric_crs).to_crs(plot_crs)
    return outer_gdf.geometry.iloc[0]


def make_figure_axes():
    """Create the four map panels and precipitation colorbar."""
    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    map_axes = np.empty((2, 2), dtype=object)
    map_axes[0, 0] = fig.add_subplot(grid[0, 0], projection=proj_map)
    map_axes[0, 1] = fig.add_subplot(grid[0, 1], projection=proj_map)
    map_axes[1, 0] = fig.add_subplot(grid[1, 0], projection=proj_map)
    map_axes[1, 1] = fig.add_subplot(grid[1, 1], projection=proj_map)
    cbar_ax = fig.add_subplot(grid[:, 2])

    for ax in map_axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)
    return fig, map_axes, cbar_ax, proj_data


def plot_precipitation(ax, da_precip, proj_data):
    """Plot precipitation for the selected reference."""
    lon, lat = get_lon_lat(da_precip)
    if REFERENCE_DATASET == "era5":
        precipitation = np.asarray(da_precip.values)
        lon_edges = centers_to_edges(lon.values)
        lat_edges = centers_to_edges(lat.values)

        if lat_edges[0] > lat_edges[-1]:
            lat_edges = lat_edges[::-1]
            precipitation = precipitation[::-1, :]
        if lon_edges[0] > lon_edges[-1]:
            lon_edges = lon_edges[::-1]
            precipitation = precipitation[:, ::-1]

        lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)
        return ax.pcolormesh(
            lon_edges_2d,
            lat_edges_2d,
            precipitation,
            cmap=PRECIP_CMAP,
            vmin=PRECIP_ZERO_THRESHOLD,
            vmax=PRECIP_LEVELS.max(),
            shading="auto",
            transform=proj_data,
        )

    return ax.pcolormesh(
        lon.values,
        lat.values,
        da_precip.values,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_ZERO_THRESHOLD,
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )


def plot_msl_contours(ax, da_msl, proj_data):
    """Plot labelled ERA5 mean sea level pressure contours."""
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


def plot_catchment_boundary(ax, geometry, proj_data):
    """Plot the selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )


def plot_event_map_panel(ax, ds_precip, ds_msl, catchment_boundary, target_date, proj_data):
    """Plot precipitation, pressure, and catchment for one date."""
    da_precip = load_precipitation(ds_precip, target_date)
    da_msl = load_msl(ds_msl, target_date)
    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    return mesh


def plot_all_map_panels(map_axes, ds_precip, ds_msl, catchment_boundary, proj_data):
    """Plot panels a-d."""
    mesh = None
    for ax, target_date in zip(map_axes.flat, EVENT_DATES):
        mesh = plot_event_map_panel(
            ax,
            ds_precip,
            ds_msl,
            catchment_boundary,
            target_date,
            proj_data,
        )
    return mesh


def add_panel_titles(map_axes):
    """Add panel labels and dates to panels a-d."""
    for ax, label, date in zip(map_axes.flat, ["a)", "b)", "c)", "d)"], EVENT_DATES):
        formatted_date = np.datetime64(date).astype("datetime64[D]").astype(object).strftime("%B %-d")
        ax.set_title(f"{label} {formatted_date} {YEAR}", fontsize=TITLE_FONTSIZE, pad=3)


def add_precip_colorbar(fig, mesh, cbar_ax):
    """Add the precipitation colorbar for the selected reference dataset."""
    colorbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical")
    colorbar.set_label(f"{get_reference_name()} precipitation (mm)", fontsize=AXIS_LABELSIZE)
    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_map_legend(map_axes, catchment_label):
    """Add catchment and pressure legend inside panel a."""
    legend_handles = [
        Line2D([0], [0], color=CATCHMENT_EDGE_COLOR, linewidth=2, label=catchment_label),
        Line2D(
            [0],
            [0],
            color=MSL_CONTOUR_COLOR,
            linewidth=MSL_CONTOUR_LINEWIDTH,
            label="ERA5 mean sea level pressure (hPa)",
        ),
    ]
    legend = map_axes[0, 0].legend(
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


def finalize_layout_and_save(fig, savepath):
    """Apply the preserved map layout, optionally save, and show the figure."""
    bottom = FIG_BOTTOM_IN / FIG_HEIGHT_IN
    top = 1.0 - FIG_TOP_IN / FIG_HEIGHT_IN
    fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=bottom, top=top)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        print("Wrote:", savepath)

    plt.show()
    plt.close(fig)


def main():
    """Run the plotting workflow."""
    validate_user_settings()
    catchment = get_catchment_settings(CATCHMENT_NAME)
    ds_precip = open_precipitation()
    ds_msl = open_era5_msl(MSL_FILE)

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )
        fig, map_axes, cbar_ax, proj_data = make_figure_axes()
        mesh = plot_all_map_panels(
            map_axes=map_axes,
            ds_precip=ds_precip,
            ds_msl=ds_msl,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )
        add_panel_titles(map_axes)
        add_precip_colorbar(fig, mesh, cbar_ax)
        add_map_legend(map_axes, catchment["label"])
        finalize_layout_and_save(fig, OUTPUT_FILE)
    finally:
        ds_precip.close()
        ds_msl.close()


if __name__ == "__main__":
    main()
