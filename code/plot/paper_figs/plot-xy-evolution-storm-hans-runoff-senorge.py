#!/usr/bin/env python3
"""
Plot seNorge runoff during Storm Hans for the Drammen catchment.

Workflow:
1. Load yearly seNorge runoff.
2. Load Drammen catchment weights.
3. Calculate catchment-weighted runoff time series.
4. Load catchment boundary.
5. Plot four runoff maps.
6. Plot catchment-mean runoff time series.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config


# =============================================================================
# 1. User settings
# =============================================================================

YEAR = 2023
CATCHMENT_NAME = "drammen"

EVENT_LAGS = [-2, -1, 0, 1]
EVENT_DATES = [
    "2023-08-08",
    "2023-08-09",
    "2023-08-10",
    "2023-08-11",
]

RUNOFF_VAR = "gwb_q"
WEIGHT_VAR = "catchment_weight"

TIMESERIES_START = "2023-01-01"
TIMESERIES_END = "2023-12-31"

WRITE_TO_FILE = True


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

PATH_SENORGE_RUNOFF = Path(config.dirs["senorge_continuous_daily"]) / RUNOFF_VAR

RUNOFF_FILE = PATH_SENORGE_RUNOFF / f"{RUNOFF_VAR}_{YEAR}.nc"

WEIGHTS_FILE = (
    Path(config.dirs["nve"])
    / "weights_catchment_regine_drammen_senorge.nc"
)

OUTPUT_FILENAME = f"{PATH_OUT}xy_evolution_storm_hans_runoff_senorge.png"



# =============================================================================
# 3. Figure settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 11.2

MAP_EXTENT = [4.75, 12.75, 58.0, 63.0]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
LEGEND_FONTSIZE = 9


# =============================================================================
# 4. Plot styling
# =============================================================================

RUNOFF_LEVELS = np.arange(1, 31, 2)
RUNOFF_ZERO_THRESHOLD = 1.0
RUNOFF_CMAP = plt.get_cmap("Blues").copy()
RUNOFF_CMAP.set_under("white")

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

RUNOFF_LINE_COLOR = "tab:blue"
RUNOFF_LINEWIDTH = 1.6
EVENT_MARKER_SIZE = 35


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
# 6. General helper functions
# =============================================================================

def get_catchment_settings(catchment_name):
    """Return metadata for selected catchment."""

    if catchment_name not in CATCHMENTS:
        raise ValueError(f"Unknown catchment: {catchment_name}")

    return CATCHMENTS[catchment_name]


def get_time_coord_name(da):
    """Return time coordinate name."""

    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name

    raise ValueError("Could not identify time coordinate.")


def check_dims(da, expected_dims, name):
    """Check that expected dimensions exist."""

    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def get_lon_lat(da):
    """Return seNorge longitude and latitude coordinates."""

    if "lon" not in da.coords or "lat" not in da.coords:
        raise KeyError("Expected coordinates 'lon' and 'lat'.")

    return da["lon"], da["lat"]


# =============================================================================
# 7. Load runoff, weights, and calculate time series
# =============================================================================

def load_runoff_file(filename, variable):
    """Load the yearly seNorge runoff file."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]
    da.attrs["units"] = "mm/day"

    check_dims(da, ("time", "y", "x"), "Runoff")

    return ds, da


def load_catchment_weights(filename, weight_var):
    """Load catchment weights and orient them like the seNorge runoff grid."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds_w = xr.open_dataset(filename)

    if weight_var not in ds_w:
        raise KeyError(
            f"Variable '{weight_var}' not found in {filename}. "
            f"Available variables: {list(ds_w.data_vars)}"
        )

    w = ds_w[weight_var].astype("float32")

    if "Y" in w.dims:
        w = w.rename({"Y": "y"})
    if "X" in w.dims:
        w = w.rename({"X": "x"})

    # Important: runoff y increases, weights y decreases
    w = w.sortby("y")

    return w


def match_weights_to_runoff_grid(da_runoff, weights):
    """Put weights onto the runoff grid using positional matching."""

    runoff_grid = da_runoff.isel(time=0, drop=True)

    if weights.shape != runoff_grid.shape:
        raise ValueError(
            f"Weight grid shape {weights.shape} does not match runoff grid "
            f"shape {runoff_grid.shape}."
        )

    weights_on_grid = xr.DataArray(
        weights.values,
        dims=runoff_grid.dims,
        coords=runoff_grid.coords,
        name=WEIGHT_VAR,
    )

    return weights_on_grid


def calculate_catchment_mean_runoff_timeseries(
    da_runoff,
    weights,
    start_date,
    end_date,
):
    """
    Calculate catchment-weighted mean runoff time series.

    Formula:
        sum(runoff * weight) / sum(weight)

    Only finite runoff values and positive finite weights are used.
    """

    da = da_runoff.sel(time=slice(start_date, end_date))
    w = match_weights_to_runoff_grid(da, weights)

    valid = np.isfinite(da) & np.isfinite(w) & (w > 0)

    weighted_sum = (da.where(valid) * w.where(valid)).sum(
        dim=("y", "x"),
        skipna=True,
    )

    weight_sum = w.where(valid).sum(
        dim=("y", "x"),
        skipna=True,
    )

    da_ts = weighted_sum / weight_sum

    da_ts.name = "daily_catchment_mean_runoff"
    da_ts.attrs["description"] = "Catchment-weighted daily mean runoff"
    da_ts.attrs["units"] = da.attrs.get("units", "mm/day")

    return da_ts.load()


# =============================================================================
# 8. Load catchment boundary
# =============================================================================

def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
    """Load catchment polygon and keep only the outer boundary."""

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
    """Create four map panels, one colorbar axis, and one time-series panel."""

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

    map_axes = np.empty((2, 2), dtype=object)
    map_axes[0, 0] = fig.add_subplot(gs[0, 0], projection=proj_map)
    map_axes[0, 1] = fig.add_subplot(gs[0, 1], projection=proj_map)
    map_axes[1, 0] = fig.add_subplot(gs[1, 0], projection=proj_map)
    map_axes[1, 1] = fig.add_subplot(gs[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(gs[0:2, 2])
    ts_ax = fig.add_subplot(gs[2, 0:2])

    for ax in map_axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data


def get_map_axes(map_axes):
    """Return map axes as a flat list."""

    return list(map_axes.flat)


def align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax):
    """Align panel e with the combined left/right borders of panels a-d."""

    fig.canvas.draw()

    left = min(
        map_axes[0, 0].get_position().x0,
        map_axes[1, 0].get_position().x0,
    )

    right = max(
        map_axes[0, 1].get_position().x1,
        map_axes[1, 1].get_position().x1,
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
# =============================================================================
# 10. Map plotting functions
# =============================================================================

def select_runoff_date(da_runoff, date):
    """Select runoff for one date."""

    time_name = get_time_coord_name(da_runoff)
    return da_runoff.sel({time_name: np.datetime64(date, "ns")}).load()


def plot_runoff_map(ax, da_map, proj_data):
    """Plot one daily runoff map."""

    lon, lat = get_lon_lat(da_map)

    mesh = ax.pcolormesh(
        lon.values,
        lat.values,
        da_map.values,
        cmap=RUNOFF_CMAP,
        vmin=RUNOFF_ZERO_THRESHOLD,
        vmax=RUNOFF_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )

    return mesh


def plot_catchment_boundary(ax, geometry, proj_data):
    """Plot catchment boundary."""

    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )


def plot_map_panels(map_axes, da_runoff, catchment_boundary, proj_data):
    """Plot the four Storm Hans runoff map panels."""

    mesh = None

    for ax, date in zip(get_map_axes(map_axes), EVENT_DATES):
        da_map = select_runoff_date(da_runoff, date)
        mesh = plot_runoff_map(ax, da_map, proj_data)
        plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


# =============================================================================
# 11. Time-series plotting function
# =============================================================================

def plot_runoff_timeseries(ts_ax, da_ts):
    """Plot the 2023 catchment-mean runoff time series."""

    time_name = get_time_coord_name(da_ts)

    ts_ax.plot(
        da_ts[time_name].values,
        da_ts.values,
        color=RUNOFF_LINE_COLOR,
        linewidth=RUNOFF_LINEWIDTH,
        label="Catchment-mean runoff",
    )

    for date in EVENT_DATES:
        event_value = da_ts.sel(
            {time_name: np.datetime64(date)},
            method="nearest",
        )

        ts_ax.scatter(
            event_value[time_name].values,
            event_value.values,
            color=RUNOFF_LINE_COLOR,
            s=EVENT_MARKER_SIZE,
            zorder=5,
        )

    ts_ax.set_title(
        "e) 2023 Drammen catchment-mean runoff",
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

    ts_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ts_ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)


# =============================================================================
# 12. Figure finishing functions
# =============================================================================

def add_panel_titles(map_axes):
    """Add titles to map panels."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, lag, date in zip(
        get_map_axes(map_axes),
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
    """Add runoff colorbar."""

    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.set_label("Daily runoff (mm/day)", fontsize=AXIS_LABELSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_legend(map_axes, catchment_label):
    """Add map legend."""

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=CATCHMENT_EDGE_COLOR,
            linewidth=2,
            label=catchment_label,
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


def finalize_layout_and_save(fig, map_axes, ts_ax, savepath):
    """Apply final figure formatting, save, and show."""

    fig.subplots_adjust(
        left=0.065,
        right=0.96,
        bottom=0.075,
        top=0.96,
    )

    align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax)

    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 13. Main script
# =============================================================================

def main():
    """Run workflow in a clear sequence."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    # 1. Load runoff data
    ds_runoff, da_runoff = load_runoff_file(
        filename=RUNOFF_FILE,
        variable=RUNOFF_VAR,
    )

    try:
        # 2. Load catchment weights
        weights = load_catchment_weights(
            filename=WEIGHTS_FILE,
            weight_var=WEIGHT_VAR,
        )

        # 3. Calculate catchment-mean runoff time series
        da_ts = calculate_catchment_mean_runoff_timeseries(
            da_runoff=da_runoff,
            weights=weights,
            start_date=TIMESERIES_START,
            end_date=TIMESERIES_END,
        )

        # 4. Load catchment boundary
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        # 5. Create figure
        fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data = make_figure_axes()

        # 6. Plot maps
        mesh = plot_map_panels(
            map_axes=map_axes,
            da_runoff=da_runoff,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

        # 7. Plot time series
        plot_runoff_timeseries(
            ts_ax=ts_ax,
            da_ts=da_ts,
        )

        # 8. Add titles, colorbar, and legend
        add_panel_titles(map_axes)
        add_colorbar(fig, mesh, cbar_ax)
        add_legend(map_axes, catchment["label"])

        # 9. Save and show
        finalize_layout_and_save(
            fig=fig,
            map_axes=map_axes,
            ts_ax=ts_ax,
            savepath=OUTPUT_FILENAME,
        )

    finally:
        ds_runoff.close()


if __name__ == "__main__":
    main()
