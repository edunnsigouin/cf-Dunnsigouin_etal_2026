#!/usr/bin/env python3
"""
Figure 1: ERA5 precipitation and mean sea level pressure, together with
ERA5-Land surface runoff during Storm Hans.

The figure contains:

Panels a-d:
- ERA5 daily accumulated precipitation as shading
- ERA5 mean sea level pressure as labelled grey contours
- Drammen catchment boundary in red

Panel e:
- 2023 Drammen catchment-mean ERA5-Land surface runoff
- 95% interval over all available years (1950-2023)
- Median over all available years
- One blue marker on the user-specified Storm Hans date

No station locations or station data are used in this script.
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

# Dates shown in panels a-d.
EVENT_DATES = [
    "2023-08-06",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
]

# Date marked with one blue dot in panel e.
HANS_DATE = "2023-08-07"

# ERA5 variables used in panels a-d.
PRECIP_VAR = "tp24"
MSL_VAR = "msl"

# ERA5-Land variable used in panel e.
RUNOFF_VAR = "ro"
NDAY = 1

WRITE_TO_FILE = False


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = Path(config.dirs["fig"])
PATH_CATCHMENT = Path(config.dirs["nve"])
PATH_ERA5 = Path(config.dirs["era5_continuous_daily"])
PATH_ERA5_LAND_PROCESSED = Path(config.dirs["era5_land_processed"])

PRECIP_FILE = (
    PATH_ERA5
    / PRECIP_VAR
    / f"{PRECIP_VAR}_0.5x0.5_{YEAR}.nc"
)

MSL_FILE = (
    PATH_ERA5
    / MSL_VAR
    / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"
)

RUNOFF_TIMESERIES_FILE = (
    PATH_ERA5_LAND_PROCESSED
    / (
        f"t_{RUNOFF_VAR}_{NDAY}dayacc_regine_{CATCHMENT_NAME}_"
        "era5_land_0.1x0.1_1950-2023.nc"
    )
)

OUTPUT_FILE = PATH_OUT / "fig-01.png"


# =============================================================================
# 3. Figure settings
# =============================================================================

FIG_WIDTH_IN = 9.4
FIG_HEIGHT_IN = 11.2

MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.10

CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
CONTOUR_LABELSIZE = 9
LEGEND_FONTSIZE = 9


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
    """Return metadata for the selected catchment."""

    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. "
            f"Valid options are: {valid_names}."
        )

    return CATCHMENTS[catchment_name]


def get_time_coord_name(da):
    """Return the name of the time coordinate."""

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
    """Raise a clear error if an input file does not exist."""

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")


# =============================================================================
# 7. Data loading
# =============================================================================


def open_era5_variable(filename, variable):
    """Open one ERA5 variable and convert it to plotting units."""

    check_file_exists(filename)

    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if variable == PRECIP_VAR:
        # ERA5 precipitation is stored in metres.
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:
        # ERA5 mean sea level pressure is stored in Pa.
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"

    return ds


def open_runoff_timeseries(filename):
    """Open processed catchment-mean ERA5-Land surface runoff."""

    check_file_exists(filename)

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


def select_date(da, target_date):
    """Select one date from a DataArray."""

    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()


def load_precipitation(ds_precip, target_date):
    """Load ERA5 daily precipitation for one date."""

    return select_date(ds_precip[PRECIP_VAR], target_date)


def load_msl(ds_msl, target_date):
    """Load ERA5 mean sea level pressure for one date."""

    return select_date(ds_msl[MSL_VAR], target_date)


# =============================================================================
# 8. Catchment boundary
# =============================================================================


def load_catchment_outer_boundary(
    filename,
    base_dir,
    crs_if_missing="EPSG:4326",
):
    """
    Load a catchment polygon and retain only its outer boundary.

    Removing internal polygon boundaries produces a cleaner map when a
    catchment file consists of several adjacent polygons.
    """

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
# 9. ERA5-Land runoff climatology for panel e
# =============================================================================


def year_series_and_climatology_by_doy(da, year):
    """
    Extract one full year and calculate the day-of-year climatology.

    Returns
    -------
    x_dates : pandas.DatetimeIndex
        Daily dates in the selected year.
    y_year : numpy.ndarray
        Daily runoff values in the selected year.
    q_low : numpy.ndarray
        2.5th percentile for each day of year.
    q_high : numpy.ndarray
        97.5th percentile for each day of year.
    q_median : numpy.ndarray
        Median for each day of year.
    """

    da = da.dropna("time")

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    x_dates = pd.date_range(start, end, freq="D")

    da_year = da.sel(time=slice(start, end))

    if da_year.sizes.get("time", 0) != len(x_dates):
        da_year = (
            da_year
            .resample(time="1D")
            .mean()
            .sel(time=slice(start, end))
        )

    y_year = da_year.values

    quantiles = da.groupby("time.dayofyear").quantile(
        [0.025, 0.5, 0.975],
        dim="time",
    )

    # YEAR = 2023 is not a leap year, so the panel contains 365 days.
    day_of_year = np.arange(1, 366)

    q_low = (
        quantiles
        .sel(quantile=0.025)
        .sel(dayofyear=day_of_year, drop=True)
        .values
    )
    q_median = (
        quantiles
        .sel(quantile=0.5)
        .sel(dayofyear=day_of_year, drop=True)
        .values
    )
    q_high = (
        quantiles
        .sel(quantile=0.975)
        .sel(dayofyear=day_of_year, drop=True)
        .values
    )

    return x_dates, y_year, q_low, q_high, q_median


# =============================================================================
# 10. Figure setup
# =============================================================================


def make_figure_axes():
    """Create four map panels, one colorbar, and panel e."""

    proj_map = ccrs.LambertConformal(
        central_longitude=CENTRAL_LON,
        central_latitude=CENTRAL_LAT,
    )
    proj_data = ccrs.PlateCarree()

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    grid = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 0.45],
        wspace=MAP_WSPACE,
        hspace=MAP_HSPACE,
    )

    map_axes = np.empty((2, 2), dtype=object)
    map_axes[0, 0] = fig.add_subplot(grid[0, 0], projection=proj_map)
    map_axes[0, 1] = fig.add_subplot(grid[0, 1], projection=proj_map)
    map_axes[1, 0] = fig.add_subplot(grid[1, 0], projection=proj_map)
    map_axes[1, 1] = fig.add_subplot(grid[1, 1], projection=proj_map)

    cbar_ax = fig.add_subplot(grid[0:2, 2])
    ts_ax = fig.add_subplot(grid[2, 0:2])

    for ax in map_axes.flat:
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)

    return fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data


def get_map_axes(map_axes):
    """Return the four map axes as a flat list."""

    return list(map_axes.flat)


def align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax):
    """Align panel e with the combined outer edges of panels a-d."""

    fig.canvas.draw()

    left = min(
        map_axes[0, 0].get_position().x0,
        map_axes[1, 0].get_position().x0,
    )

    right = max(
        map_axes[0, 1].get_position().x1,
        map_axes[1, 1].get_position().x1,
    )

    position = ts_ax.get_position()

    ts_ax.set_position(
        [
            left,
            position.y0,
            right - left,
            position.height,
        ]
    )


# =============================================================================
# 11. Map plotting functions for panels a-d
# =============================================================================


def plot_precipitation(ax, da_precip, proj_data):
    """Plot ERA5 precipitation using explicit grid-cell edges."""

    lon, lat = get_lon_lat(da_precip)
    precip = da_precip.values

    lon_edges = centers_to_edges(lon.values)
    lat_edges = centers_to_edges(lat.values)

    # Ensure that coordinates and data increase in the same direction.
    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        precip = precip[::-1, :]

    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        precip = precip[:, ::-1]

    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)

    mesh = ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        precip,
        cmap=PRECIP_CMAP,
        vmin=PRECIP_ZERO_THRESHOLD,
        vmax=PRECIP_LEVELS.max(),
        shading="auto",
        transform=proj_data,
    )

    return mesh


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


def plot_event_map_panel(
    ax,
    ds_precip,
    ds_msl,
    catchment_boundary,
    target_date,
    proj_data,
):
    """Plot precipitation, pressure, and the catchment for one date."""

    da_precip = load_precipitation(ds_precip, target_date)
    da_msl = load_msl(ds_msl, target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh


def plot_all_map_panels(
    map_axes,
    ds_precip,
    ds_msl,
    catchment_boundary,
    proj_data,
):
    """Plot panels a-d."""

    mesh = None

    for ax, target_date in zip(get_map_axes(map_axes), EVENT_DATES):
        mesh = plot_event_map_panel(
            ax=ax,
            ds_precip=ds_precip,
            ds_msl=ds_msl,
            catchment_boundary=catchment_boundary,
            target_date=target_date,
            proj_data=proj_data,
        )

    return mesh


# =============================================================================
# 12. Panel e: ERA5-Land surface runoff
# =============================================================================


def plot_runoff_timeseries(ts_ax, da_runoff_ts, year):
    """
    Plot panel e exactly as in script 2: ERA5-Land surface runoff for the
    selected year, its climatological median, and its 95% interval.
    """

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
        label="95% interval 1950-2023",
    )

    ts_ax.plot(
        x,
        med,
        linewidth=RUNOFF_MEDIAN_LINEWIDTH,
        color=RUNOFF_MEDIAN_LINE_COLOR,
        label="Median 1950-2023",
    )

    ts_ax.plot(
        x,
        y,
        linewidth=RUNOFF_YEAR_LINEWIDTH,
        color="tab:blue",
        label=f"{year}",
    )

    hans_date = pd.Timestamp(HANS_DATE)

    if hans_date in x:
        index = np.where(x == hans_date)[0][0]

        ts_ax.scatter(
            hans_date,
            y[index],
            color="tab:blue",
            s=EVENT_MARKER_SIZE,
            zorder=6,
            label="Storm Hans",
        )
    else:
        print(
            f"Warning: HANS_DATE {HANS_DATE} was not found "
            f"in the {year} runoff time series."
        )

    ts_ax.set_title(
        "e) Drammen catchment surface runoff",
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
    """Add panel labels and calendar dates to panels a-d."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, label, date in zip(
        get_map_axes(map_axes),
        panel_labels,
        EVENT_DATES,
    ):
        formatted_date = (
            np.datetime64(date)
            .astype("datetime64[D]")
            .astype(object)
            .strftime("%B %-d")
        )

        ax.set_title(
            f"{label} {formatted_date} {YEAR}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )


def add_precip_colorbar(fig, mesh, cbar_ax):
    """Add the ERA5 precipitation colorbar."""

    colorbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
    )

    colorbar.set_label(
        "Precipitation (mm)",
        fontsize=AXIS_LABELSIZE,
    )

    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE)


def add_map_legend(map_axes, catchment_label):
    """Add the catchment and pressure legend to panel a."""

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


def finalize_layout_and_save(fig, map_axes, ts_ax, savepath):
    """Apply the final layout, optionally save the figure, and display it."""

    fig.subplots_adjust(
        left=0.065,
        right=0.96,
        bottom=0.075,
        top=0.96,
    )

    align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax)

    if WRITE_TO_FILE:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()


# =============================================================================
# 14. Main workflow
# =============================================================================


def main():
    """Run the complete plotting workflow."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    ds_precip = open_era5_variable(PRECIP_FILE, PRECIP_VAR)
    ds_msl = open_era5_variable(MSL_FILE, MSL_VAR)
    ds_runoff_ts, da_runoff_ts = open_runoff_timeseries(
        RUNOFF_TIMESERIES_FILE
    )

    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )

        fig, map_axes, ts_ax, cbar_ax, _, proj_data = make_figure_axes()

        mesh = plot_all_map_panels(
            map_axes=map_axes,
            ds_precip=ds_precip,
            ds_msl=ds_msl,
            catchment_boundary=catchment_boundary,
            proj_data=proj_data,
        )

        plot_runoff_timeseries(
            ts_ax=ts_ax,
            da_runoff_ts=da_runoff_ts,
            year=YEAR,
        )

        add_panel_titles(map_axes)
        add_precip_colorbar(fig, mesh, cbar_ax)
        add_map_legend(map_axes, catchment["label"])

        finalize_layout_and_save(
            fig=fig,
            map_axes=map_axes,
            ts_ax=ts_ax,
            savepath=OUTPUT_FILE,
        )

    finally:
        ds_precip.close()
        ds_msl.close()
        ds_runoff_ts.close()


if __name__ == "__main__":
    main()
