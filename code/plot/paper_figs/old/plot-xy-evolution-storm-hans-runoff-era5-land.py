#!/usr/bin/env python3
"""
Plot ERA5-Land surface runoff during Storm Hans for the Drammen catchment.

The figure contains:
1. Four ERA5-Land surface runoff map panels for 2023-08-07 to 2023-08-10.
2. Daily surface runoff as shading.
3. Drammen catchment boundary in red.
4. A bottom time-series panel showing:
   - 2023 Drammen catchment-mean surface runoff
   - 95% interval over all years
   - Median over all years
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

NDAY = 1 # timeseries Nday accumulation

EVENT_LAGS = [-2, -1, 0, 1]
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

RUNOFF_VAR = "sro"

WRITE_TO_FILE = True


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

PATH_ERA5_LAND_RUNOFF = (
    Path(config.dirs["era5_land_continuous_daily_scandinavia"]) / RUNOFF_VAR
)

RUNOFF_FILE = PATH_ERA5_LAND_RUNOFF / f"{RUNOFF_VAR}_0.1x0.1_{YEAR}.nc"

RUNOFF_TIMESERIES_FILE = (
    config.dirs["era5_land_processed"]
    + f"t_{RUNOFF_VAR}_{NDAY}dayacc_regine_drammen_era5_land_0.1x0.1_1950-2023.nc"
)

OUTPUT_FILENAME = f"{PATH_OUT}xy_evolution_storm_hans_sro_era5_land.png"


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

RUNOFF_LEVELS = np.arange(1, 51, 2)#np.arange(1, 31, 2)
RUNOFF_ZERO_THRESHOLD = 1.0
RUNOFF_CMAP = plt.get_cmap("Blues").copy()
RUNOFF_CMAP.set_under("white")

CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"

RUNOFF_RANGE_FILL_ALPHA = 0.25
RUNOFF_MEDIAN_LINE_COLOR = "tab:red"
RUNOFF_MEDIAN_LINEWIDTH = 1.4
RUNOFF_YEAR_LINEWIDTH = 1.2

EVENT_MARKER_SIZE = 20


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


def get_lon_lat(da):
    """Return longitude and latitude coordinates."""

    lon_names = ["longitude", "lon"]
    lat_names = ["latitude", "lat"]

    lon_name = next((name for name in lon_names if name in da.coords), None)
    lat_name = next((name for name in lat_names if name in da.coords), None)

    if lon_name is None or lat_name is None:
        raise KeyError(
            "Could not find longitude/latitude coordinates. "
            f"Available coordinates: {list(da.coords)}"
        )

    return da[lon_name], da[lat_name]


def check_spatial_dims(da, name):
    """Check that data has recognizable spatial dimensions."""

    possible_lon_dims = ["longitude", "lon", "x"]
    possible_lat_dims = ["latitude", "lat", "y"]

    has_lon_dim = any(dim in da.dims for dim in possible_lon_dims)
    has_lat_dim = any(dim in da.dims for dim in possible_lat_dims)

    if not has_lon_dim or not has_lat_dim:
        raise ValueError(
            f"{name} must have longitude/latitude spatial dimensions. "
            f"Found dimensions: {da.dims}"
        )


# =============================================================================
# 7. Data loading
# =============================================================================

def load_runoff_file(filename, variable):
    """Load the yearly ERA5-Land surface runoff file used for map panels."""

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

    time_name = get_time_coord_name(da)
    check_spatial_dims(da, "ERA5-Land surface runoff")

    # The processed daily file is assumed to already be in mm/day.
    da.attrs["units"] = "mm/day"

    # Put time first if possible. This makes later selection clearer.
    other_dims = [dim for dim in da.dims if dim != time_name]
    da = da.transpose(time_name, *other_dims)

    return ds, da


def load_runoff_timeseries(filename):
    """Load processed Drammen catchment-mean ERA5-Land surface runoff time series."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)

    if RUNOFF_VAR not in ds:
        raise KeyError(
            f"Variable '{RUNOFF_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[RUNOFF_VAR]
    da.attrs["units"] = "mm/day"

    return ds, da


# =============================================================================
# 8. Catchment boundary
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
# 9. Runoff time-series climatology
# =============================================================================

def year_series_and_climatology_by_doy(da, year):
    """
    Extract one full year's daily runoff and compute day-of-year climatology.

    Returns:
    - selected-year runoff
    - 2.5% quantile
    - 50% quantile
    - 97.5% quantile

    The climatology is calculated over all available years in the input file.
    """

    da = da.dropna("time")

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    x_dates = pd.date_range(start, end, freq="D")

    da_year = da.sel(time=slice(start, end))

    if da_year.sizes.get("time", 0) != len(x_dates):
        da_year = da_year.resample(time="1D").mean().sel(time=slice(start, end))

    y_year = da_year.values

    q = da.groupby("time.dayofyear").quantile(
        [0.025, 0.5, 0.975],
        dim="time",
    )

    doy = np.arange(1, 366)

    q_low = q.sel(quantile=0.025).sel(dayofyear=doy, drop=True).values
    q_median = q.sel(quantile=0.5).sel(dayofyear=doy, drop=True).values
    q_high = q.sel(quantile=0.975).sel(dayofyear=doy, drop=True).values

    return x_dates, y_year, q_low, q_high, q_median


# =============================================================================
# 10. Figure setup
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
# 11. Map plotting functions
# =============================================================================

def select_runoff_date(da_runoff, date):
    """Select runoff for one date."""

    time_name = get_time_coord_name(da_runoff)

    return da_runoff.sel(
        {time_name: np.datetime64(date, "ns")}
    ).load()


def plot_runoff_map(ax, da_map, proj_data):
    """Plot one daily surface runoff map."""

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
    """Plot the four Storm Hans ERA5-Land surface runoff map panels."""

    mesh = None

    for ax, date in zip(get_map_axes(map_axes), EVENT_DATES):
        da_map = select_runoff_date(da_runoff, date)
        mesh = plot_runoff_map(ax, da_map, proj_data)
        plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


# =============================================================================
# 12. Time-series plotting function
# =============================================================================

def plot_runoff_timeseries(ts_ax, da_runoff_ts, year):
    """Plot Drammen catchment surface runoff for selected year and climatology."""

    x, y, lo, hi, med = year_series_and_climatology_by_doy(
        da=da_runoff_ts,
        year=year,
    )

    ts_ax.fill_between(
        x,
        lo,
        hi,
        alpha=RUNOFF_RANGE_FILL_ALPHA,
        linewidth=0,
        label="95% interval over all years",
    )

    ts_ax.plot(
        x,
        med,
        linewidth=RUNOFF_MEDIAN_LINEWIDTH,
        color=RUNOFF_MEDIAN_LINE_COLOR,
        label="Median over all years",
    )

    ts_ax.plot(
        x,
        y,
        linewidth=RUNOFF_YEAR_LINEWIDTH,
        color="tab:blue",
        label=f"{year}",
    )

    for date in EVENT_DATES:
        event_date = pd.Timestamp(date)

        if event_date in x:
            idx = np.where(x == event_date)[0][0]

            ts_ax.scatter(
                event_date,
                y[idx],
                color="tab:blue",
                s=EVENT_MARKER_SIZE,
                zorder=6,
            )

    ts_ax.set_title(
        f"e) Drammen catchment {NDAY}-day accumulated surface runoff 1950-2023",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel("mm", fontsize=AXIS_LABELSIZE)
    ts_ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)

    ts_ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    ts_ax.set_xlim(start, end)
    ts_ax.margins(x=0)

    ts_ax.xaxis.set_major_locator(mdates.MonthLocator())
    ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ts_ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ts_ax.legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
    )


# =============================================================================
# 13. Figure finishing functions
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
    """Add surface runoff colorbar."""

    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    cbar.set_label("Surface runoff (mm/day)", fontsize=AXIS_LABELSIZE)
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
# 14. Main script
# =============================================================================

def main():
    """Run workflow in a clear sequence."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    ds_runoff, da_runoff = load_runoff_file(
        filename=RUNOFF_FILE,
        variable=RUNOFF_VAR,
    )

    ds_runoff_ts, da_runoff_ts = load_runoff_timeseries(
        filename=RUNOFF_TIMESERIES_FILE,
    )

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data = make_figure_axes()

        mesh = plot_map_panels(
            map_axes=map_axes,
            da_runoff=da_runoff,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

        plot_runoff_timeseries(
            ts_ax=ts_ax,
            da_runoff_ts=da_runoff_ts,
            year=YEAR,
        )

        add_panel_titles(map_axes)
        add_colorbar(fig, mesh, cbar_ax)
        add_legend(map_axes, catchment["label"])

        finalize_layout_and_save(
            fig=fig,
            map_axes=map_axes,
            ts_ax=ts_ax,
            savepath=OUTPUT_FILENAME,
        )

    finally:
        ds_runoff.close()
        ds_runoff_ts.close()


if __name__ == "__main__":
    main()
