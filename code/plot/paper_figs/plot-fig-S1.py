"""
Fig 1 for Hans paper.

Plot Storm Hans precipitation, return period, and catchment weights side by side.

The script:
1. Reads precipitation and return period from seNorge NetCDF
2. Reads catchment weights from an ERA5-grid NetCDF
3. Loads catchment geometry
4. Creates three map panels
5. Plots precipitation, return period, and catchment weights
6. Overlays catchment borders and station markers on all panels
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User input parameters
# =============================================================================

# --- Paths
path_in_catchment = config.dirs["nve"]
path_in_senorge = config.dirs["senorge_processed"]
path_out = config.dirs["fig"]

# --- Files
filename_in_senorge = (
    f"{path_in_senorge}returnperiod_rr_2dayacc_senorge_1957-2023_20230809.nc"
)
filename_in_weights = (
    f"{path_in_catchment}weights_catchment_regine_drammen_era5_0.5x0.5.nc"
)
filename_out = f"{path_out}fig-01.png"
write2file = False

# --- Font sizes
tick_labelsize = 12
axis_labelsize = 12
title_fontsize = 12
station_labelsize = 11

# --- Map projection and extent
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]

# --- Left panel: precipitation
PRECIP_LEVELS = np.arange(0, 160, 20)
PRECIP_CMAP = "GnBu"

# --- Middle panel: return period
CATEGORY_EDGES = np.array([1.0, 5.0, 10.0, 50.0, 100.0, np.inf], dtype=float)
CATEGORY_LABELS = ["1–5", "5–10", "10–50", "50–100", "> 100"]
RETURN_CMAP = "PuBuGn"

# --- Right panel: catchment weights
WEIGHT_CMAP = "BuGn"
WEIGHT_VMIN = 0.0
WEIGHT_VMAX = 1.0

# --- Catchment CRS if not present in source file
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

# --- Catchments to plot
CATCHMENTS = [
    {
        "label": "Drammensvassdraget",
        "geojson": "catchment_nve_regine_drammen.geojson",
        "color": "tab:red",
        "inset_m": 0,
    },
]

# --- Stations to plot
STATIONS = [
    {"name": "Bergheim", "lon": 9.2483, "lat": 60.4761},
    {"name": "Ål III", "lon": 8.5609, "lat": 60.6391},
]


# =============================================================================
# Data loading
# =============================================================================
def load_event_and_return_period_data(filename):
    """
    Load precipitation event accumulation and return period from NetCDF.

    Returns
    -------
    ds : xr.Dataset
        Open dataset.
    precip : xr.DataArray
        Event precipitation accumulation.
    return_period : xr.DataArray
        Return period in years.
    lon : xr.DataArray
        Longitude grid.
    lat : xr.DataArray
        Latitude grid.
    """
    ds = xr.open_dataset(filename)
    precip = ds["event_accum"]
    return_period = ds["return_period_years"]
    lon = ds["lon"]
    lat = ds["lat"]
    return ds, precip, return_period, lon, lat


def load_weights_data(filename):
    """
    Load catchment weights from NetCDF.

    Returns
    -------
    ds : xr.Dataset
        Open dataset.
    da_weights : xr.DataArray
        Catchment weights.
    """
    ds = xr.open_dataset(filename)
    da_weights = ds["catchment_weight"]
    return ds, da_weights


# =============================================================================
# Catchment geometry helpers
# =============================================================================
def load_catchment_outer_boundaries(catchments, base_dir, crs_if_missing="EPSG:4326"):
    """
    Load catchment polygons, dissolve each catchment to one geometry,
    optionally apply an inset, and keep only the outer boundary.

    Parameters
    ----------
    catchments : list of dict
        Catchment metadata.
    base_dir : str
        Directory containing catchment files.
    crs_if_missing : str
        CRS to assign if missing in the input file.

    Returns
    -------
    boundaries : list of dict
        List containing label, color, and dissolved boundary geometry.
    """
    boundaries = []

    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    for catchment in catchments:
        filepath = base_dir + catchment["geojson"]
        gdf = gpd.read_file(filepath)

        if gdf.crs is None:
            gdf = gdf.set_crs(crs_if_missing)

        # Work in projected coordinates for optional buffering/insetting
        gdf_metric = gdf.to_crs(metric_crs)
        union_geom = gdf_metric.geometry.union_all()

        inset_m = catchment.get("inset_m", 0)
        if inset_m > 0:
            union_geom = union_geom.buffer(-inset_m)

        # Keep only the outer boundaries
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

        boundaries.append(
            {
                "label": catchment["label"],
                "color": catchment["color"],
                "geometry": outer_gdf.geometry.iloc[0],
            }
        )

    return boundaries


# =============================================================================
# Plot setup helpers
# =============================================================================
def make_three_map_axes(central_lon=10.0, central_lat=62.0, extent=None):
    """
    Create a figure with three Lambert Conformal map panels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib axes
    proj_data : cartopy CRS
        CRS of the input longitude/latitude data.
    """
    proj_map = ccrs.LambertConformal(
        central_longitude=central_lon,
        central_latitude=central_lat,
    )
    proj_data = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 7.5),
        subplot_kw={"projection": proj_map},
    )

    for ax in axes:
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4)
        if extent is not None:
            ax.set_extent(extent, crs=proj_data)

    return fig, axes, proj_data


def format_colorbar(cbar, label, tick_labels=None, tick_labelsize=12, axis_labelsize=12):
    """
    Apply consistent formatting to a horizontal colorbar.
    """
    cbar.set_label(label, fontsize=axis_labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)

    if tick_labels is not None:
        cbar.ax.set_xticklabels(tick_labels, fontsize=tick_labelsize)


# =============================================================================
# Plotting helpers
# =============================================================================
def plot_precipitation(
    ax,
    da,
    lon,
    lat,
    proj_data,
    tick_labelsize=12,
    axis_labelsize=12,
):
    """
    Plot precipitation with contourf.
    """
    cf = ax.contourf(
        lon.values,
        lat.values,
        da.values,
        levels=PRECIP_LEVELS,
        cmap=PRECIP_CMAP,
        transform=proj_data,
        extend="max",
    )

    cbar = plt.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        shrink=0.9,
        pad=0.02,
    )
    format_colorbar(
        cbar,
        label="2-day accumulated precipitation (mm)",
        tick_labels=np.arange(0, 160, 20),
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )

    return cf


def categorize_return_period(T, edges):
    """
    Convert return period values to category indices.

    Values < 1 year and non-finite values are set to NaN.
    """
    values = T.values
    categories = np.digitize(values, edges, right=False) - 1
    categories = categories.astype(float)

    categories[values < 1.0] = np.nan
    categories[~np.isfinite(values)] = np.nan

    return categories


def build_return_period_cmap(n_labels):
    """
    Build a discrete colormap for return period categories.

    The first category (1-5 years) is set to white to match the intended styling.
    """
    base_cmap = plt.get_cmap(RETURN_CMAP)
    colors = base_cmap(np.linspace(0.3, 0.95, n_labels))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
    return mcolors.ListedColormap(colors)


def plot_return_period(
    ax,
    da,
    lon,
    lat,
    proj_data,
    tick_labelsize=12,
    axis_labelsize=12,
):
    """
    Plot categorized return period map.
    """
    cat = categorize_return_period(da, CATEGORY_EDGES)
    cmap = build_return_period_cmap(len(CATEGORY_LABELS))

    mesh = ax.pcolormesh(
        lon.values,
        lat.values,
        cat,
        transform=proj_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(CATEGORY_LABELS) - 0.5,
    )

    mesh.cmap.set_bad("white")

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        shrink=0.9,
        pad=0.02,
        ticks=np.arange(len(CATEGORY_LABELS)),
    )
    format_colorbar(
        cbar,
        label="Return period (years)",
        tick_labels=CATEGORY_LABELS,
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )

    return mesh


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


def plot_weights(
    ax,
    da_weights,
    proj_data,
    tick_labelsize=12,
    axis_labelsize=12,
):
    """
    Plot catchment weights as a pcolormesh.

    Important styling choice:
    Zero-valued weights are masked so that they are shown in white,
    matching the white low-end category used in the other two panels.
    """
    lats = da_weights.latitude.values
    lons = da_weights.longitude.values
    weights = da_weights.values

    lat_edges = centers_to_edges(lats)
    lon_edges = centers_to_edges(lons)

    weights_plot = weights.copy()

    # Make zero weights white by masking them
    weights_plot = np.where(weights_plot == 0, np.nan, weights_plot)

    # Ensure ascending edge order for plotting
    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        weights_plot = weights_plot[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        weights_plot = weights_plot[:, ::-1]

    lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)

    cmap = plt.get_cmap(WEIGHT_CMAP).copy()
    cmap.set_bad("white")

    mesh = ax.pcolormesh(
        lon_e,
        lat_e,
        weights_plot,
        cmap=cmap,
        vmin=WEIGHT_VMIN,
        vmax=WEIGHT_VMAX,
        shading="auto",
        transform=proj_data,
    )

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        shrink=0.9,
        pad=0.02,
    )
    format_colorbar(
        cbar,
        label="Catchment weight (fraction)",
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )

    return mesh


def plot_catchment_boundaries(ax, catchment_boundaries, proj_data):
    """
    Plot outer borders of the catchments.
    """
    for item in catchment_boundaries:
        ax.add_geometries(
            [item["geometry"]],
            crs=proj_data,
            facecolor="none",
            edgecolor=item["color"],
            linewidth=2.0,
            zorder=5,
        )


def plot_station_markers(ax, stations, proj_data, fontsize=11):
    """
    Plot station markers as yellow dots with labels.
    """
    for station in stations:
        ax.plot(
            station["lon"],
            station["lat"],
            marker="o",
            markersize=8,
            markeredgecolor="black",
            markerfacecolor="yellow",
            transform=proj_data,
            zorder=6,
        )

        if station["name"] == "Bergheim":
            dx, dy = 0.05, 0.05
        elif station["name"] == "Ål III":
            dx, dy = -0.65, 0.05
        else:
            dx, dy = 0.05, 0.05

        ax.text(
            station["lon"] + dx,
            station["lat"] + dy,
            station["name"],
            fontsize=fontsize,
            color="yellow",
            transform=proj_data,
            zorder=7,
        )


def finalize_figure(fig, axes, savepath=None, write2file=False, title_fontsize=12):
    """
    Add panel titles, finalize layout, optionally save, and show figure.
    """
    axes[0].set_title(
        "(a) Storm Hans precipitation 2023-08-07 to 2023-08-09",
        fontsize=title_fontsize,
    )
    axes[1].set_title(
        "(b) Storm Hans return period",
        fontsize=title_fontsize,
    )
    axes[2].set_title(
        "(c) Catchment weights",
        fontsize=title_fontsize,
    )

    fig.subplots_adjust(wspace=0.01, bottom=0.0, top=1.0)
    plt.tight_layout()

    if write2file:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()


# =============================================================================
# Main script
# =============================================================================
if __name__ == "__main__":

    # 1) Load precipitation and return period data
    ds_senorge, precip, return_period, lon, lat = load_event_and_return_period_data(
        filename_in_senorge
    )

    # 2) Load weights
    ds_weights, da_weights = load_weights_data(filename_in_weights)

    # 3) Load catchment borders
    catchment_boundaries = load_catchment_outer_boundaries(
        CATCHMENTS,
        base_dir=path_in_catchment,
        crs_if_missing=CATCHMENT_CRS_IF_MISSING,
    )

    # 4) Create figure and map axes
    fig, axes, proj_data = make_three_map_axes(
        central_lon=CENTRAL_LON,
        central_lat=CENTRAL_LAT,
        extent=MAP_EXTENT,
    )

    # 5) Left panel: precipitation
    plot_precipitation(
        axes[0],
        precip,
        lon,
        lat,
        proj_data,
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )
    plot_catchment_boundaries(axes[0], catchment_boundaries, proj_data)
    plot_station_markers(
        axes[0],
        STATIONS,
        proj_data,
        fontsize=station_labelsize,
    )

    # 6) Middle panel: return period
    plot_return_period(
        axes[1],
        return_period,
        lon,
        lat,
        proj_data,
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )
    plot_catchment_boundaries(axes[1], catchment_boundaries, proj_data)
    plot_station_markers(
        axes[1],
        STATIONS,
        proj_data,
        fontsize=station_labelsize,
    )

    # 7) Right panel: catchment weights
    plot_weights(
        axes[2],
        da_weights,
        proj_data,
        tick_labelsize=tick_labelsize,
        axis_labelsize=axis_labelsize,
    )
    plot_catchment_boundaries(axes[2], catchment_boundaries, proj_data)
    plot_station_markers(
        axes[2],
        STATIONS,
        proj_data,
        fontsize=station_labelsize,
    )

    # 8) Finalize and optionally save figure
    finalize_figure(
        fig,
        axes,
        savepath=filename_out,
        write2file=write2file,
        title_fontsize=title_fontsize,
    )

    # 9) Close datasets
    ds_senorge.close()
    ds_weights.close()
