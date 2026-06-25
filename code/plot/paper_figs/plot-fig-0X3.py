#!/usr/bin/env python3
"""
Plot daily precipitation and mean sea level pressure for one S2S ensemble member,
with an additional runoff time-series panel.

The figure contains:
1. Four event-relative map panels:
   - daily precipitation as shading
   - mean sea level pressure as labelled grey contours
   - selected catchment boundary in red
   - Drammen city marker
2. A bottom panel, e), showing Drammen catchment-spatial-mean runoff:
   - all S2S forecast ensemble members from initialization onward
   - the selected ensemble member highlighted
   - ERA5 runoff for the same catchment
   - a dashed vertical line marking forecast initialization

SeNorge is not included.
"""

from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config, misc


# =============================================================================
# 1. User-defined input parameters
# =============================================================================

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"

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

RUNOFF_VAR = "ro24"       # S2S forecast runoff
ERA5_RUNOFF_VAR = "ro"    # ERA5 runoff

# Panel e) date-window settings
N_DAYS_BEFORE = 3
M_DAYS_LEAD = 6

WRITE_TO_FILE = False


# =============================================================================
# 2. Paths
# =============================================================================

PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

ERA5_DOMAIN = "norway"

ERA5_PATH = (
    config.dirs["era5_continuous_daily_scandinavia"]
    + ERA5_RUNOFF_VAR
    + "/"
)

ERA5_FILE_PATTERN = (
    f"{ERA5_RUNOFF_VAR}_{GRID}"
    + "_{year}.nc"
)

OUTPUT_FILENAME = f"{PATH_OUT}fig-0X3.png"



# =============================================================================
# 3. Figure and map settings
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

ENSEMBLE_COLOR = "0.7"
ENSEMBLE_LINEWIDTH = 1.0
ENSEMBLE_ALPHA = 0.6

SELECTED_MEMBER_COLOR = "tab:blue"
SELECTED_MEMBER_LINEWIDTH = 2.5

ERA5_COLOR = "tab:red"
ERA5_LINEWIDTH = 2.5

EVENT_MARKER_SIZE = 35


# =============================================================================
# 5. Catchment metadata
# =============================================================================

CATCHMENTS = {
    "drammen": {
        "filename": "catchment_nve_regine_drammen.geojson",
        "weights_id": "regine_drammen",
        "label": "Drammen catchment",
    },
    "glomma": {
        "filename": "catchment_nve_regine_glomma.geojson",
        "weights_id": "regine_glomma",
        "label": "Glomma catchment",
    },
}


# =============================================================================
# 6. General helper functions
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


def make_catchment_weights_file(catchment_name, grid):
    """Return the catchment-weight file for a given grid."""
    catchment = get_catchment_settings(catchment_name)

    return (
        Path(PATH_CATCHMENT)
        / f"weights_catchment_{catchment['weights_id']}_era5_{grid}.nc"
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

    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional.")

    if centers.size < 2:
        raise ValueError("At least two center points are needed to infer edges.")

    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])

    return edges


def check_dims(da, expected_dims, name):
    """Check that required dimensions are present."""
    missing = [dim for dim in expected_dims if dim not in da.dims]

    if missing:
        raise ValueError(
            f"{name} is missing dimensions {missing}. "
            f"Found dimensions: {da.dims}"
        )


def get_plot_period(forecast_date, n_days_before, m_days_lead):
    """Return initialization date, plot window, loading window, and years."""
    init_date = pd.to_datetime(forecast_date)

    plot_start = init_date - pd.Timedelta(days=n_days_before)
    plot_end = init_date + pd.Timedelta(days=m_days_lead)

    load_start = plot_start - pd.Timedelta(days=1)
    load_end = plot_end + pd.Timedelta(days=1)

    years = np.arange(load_start.year, load_end.year + 1)

    return init_date, plot_start, plot_end, load_start, load_end, years


def round_time_to_nearest_day(da):
    """Round timestamps to nearest calendar day."""
    time_name = get_time_coord_name(da)
    rounded_time = pd.to_datetime(da[time_name].values).round("D")

    return da.assign_coords({time_name: rounded_time})


def subset_to_period(da, start_date, end_date):
    """Subset a DataArray to a date period."""
    time_name = get_time_coord_name(da)

    return da.sel({time_name: slice(start_date, end_date)})


# =============================================================================
# 7. Map data loading
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

    elif variable == RUNOFF_VAR:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

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


# =============================================================================
# 8. Runoff time-series data loading
# =============================================================================

def standardize_runoff_units(da):
    """
    Convert runoff to mm/day if it is stored in metres.

    S2S ro24 and ERA5 ro are assumed to represent daily accumulated runoff.
    """
    units = str(da.attrs.get("units", "")).strip().lower()

    if units in {"m", "meter", "metre", ""}:
        da = da * 1000.0
        da.attrs["units"] = "mm/day"

    elif units in {"mm", "mm/day", "mm d-1"}:
        da.attrs["units"] = "mm/day"

    return da


def open_forecast_runoff_dataset():
    """Open S2S forecast ro24 and convert to mm/day."""
    filename = make_s2s_file(RUNOFF_VAR)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if RUNOFF_VAR not in ds:
        raise KeyError(
            f"Variable '{RUNOFF_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    ds[RUNOFF_VAR] = standardize_runoff_units(ds[RUNOFF_VAR])

    return ds


def load_all_forecast_runoff_members():
    """Load S2S ro24 for all forecast ensemble members."""
    ds = open_forecast_runoff_dataset()

    try:
        da = ds[RUNOFF_VAR].load()

    finally:
        ds.close()

    return da


def preprocess_era5(ds):
    """Drop unnecessary ERA5 ensemble dimension if present."""
    return ds.drop_vars("number", errors="ignore")


def load_era5_runoff(years, time_start, time_end):
    """Load ERA5 ro over the requested period."""
    filenames = [
        ERA5_PATH + ERA5_FILE_PATTERN.format(year=int(year))
        for year in years
    ]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_era5,
        combine="by_coords",
    )

    if ERA5_DOMAIN is not None:
        domain_lats, domain_lons = misc.get_domain_latlon(ERA5_DOMAIN)
        ds = ds.sel(latitude=domain_lats, longitude=domain_lons)

    if ERA5_RUNOFF_VAR not in ds:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Variable '{ERA5_RUNOFF_VAR}' not found in ERA5 files. "
            f"Available variables: {available}"
        )

    da = ds[ERA5_RUNOFF_VAR].sel(time=slice(time_start, time_end))
    da = standardize_runoff_units(da)

    check_dims(
        da=da,
        expected_dims=("time", "latitude", "longitude"),
        name="ERA5 runoff",
    )

    return da


def load_weights(filename, spatial_dims):
    """Load predefined catchment weights."""
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Catchment weights not found: {filename}")

    ds = xr.open_dataset(filename)

    if "catchment_weight" not in ds:
        ds.close()
        raise KeyError(
            f"'catchment_weight' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    weights = ds["catchment_weight"].astype("float32").load()
    ds.close()

    weights.name = "catchment_weight"

    check_dims(
        da=weights,
        expected_dims=spatial_dims,
        name="Catchment weights",
    )

    return weights


def align_weights(da, weights):
    """Align catchment weights to the data grid."""
    time_name = get_time_coord_name(da)

    if time_name in da.dims:
        grid_template = da.isel({time_name: 0}, drop=True)
    else:
        grid_template = da

    try:
        return weights.reindex_like(grid_template)
    except Exception:
        return weights.broadcast_like(grid_template)


def catchment_mean(da, weights, spatial_dims):
    """Calculate catchment-weighted spatial mean."""
    weights = align_weights(da, weights)

    valid = xr.ufuncs.isfinite(da) & xr.ufuncs.isfinite(weights) & (weights > 0)

    weighted_sum = (da.where(valid) * weights.where(valid)).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = weights.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    out = weighted_sum / weight_sum
    out.name = "catchment_mean_runoff"
    out.attrs["units"] = da.attrs.get("units", "mm/day")

    return out


def load_forecast_catchment_mean(catchment_name, init_date, plot_end):
    """Load catchment-mean S2S ro24 for all ensemble members."""
    spatial_dims = ("latitude", "longitude")

    da = load_all_forecast_runoff_members()

    check_dims(
        da=da,
        expected_dims=("time", "number", "latitude", "longitude"),
        name="Forecast runoff",
    )

    weights_filename = make_catchment_weights_file(
        catchment_name=catchment_name,
        grid=GRID,
    )

    weights = load_weights(
        filename=weights_filename,
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=init_date,
        end_date=plot_end,
    )

    return da_mean


def load_era5_catchment_mean(
    catchment_name,
    years,
    load_start,
    load_end,
    plot_start,
    plot_end,
):
    """Load catchment-mean ERA5 ro for the panel-e date window."""
    spatial_dims = ("latitude", "longitude")

    da = load_era5_runoff(
        years=years,
        time_start=load_start,
        time_end=load_end,
    )

    weights_filename = make_catchment_weights_file(
        catchment_name=catchment_name,
        grid=GRID,
    )

    weights = load_weights(
        filename=weights_filename,
        spatial_dims=spatial_dims,
    )

    da_mean = catchment_mean(
        da=da,
        weights=weights,
        spatial_dims=spatial_dims,
    ).load()

    da_mean = round_time_to_nearest_day(da_mean)

    da_mean = subset_to_period(
        da=da_mean,
        start_date=plot_start,
        end_date=plot_end,
    )

    return da_mean


# =============================================================================
# 9. Catchment boundary
# =============================================================================

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
# 10. Figure setup
# =============================================================================

def make_figure_axes():
    """Create four map panels, one colorbar, and one time-series panel."""
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
    """Return map panels as a flat list."""
    return list(axes.flat)


# =============================================================================
# 11. Map plotting functions
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


def plot_event_panel(ax, target_date, catchment_boundary, proj_data):
    """Plot one daily map panel."""
    da_precip = load_precipitation(target_date)
    da_msl = load_msl(target_date)

    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    return mesh



# =============================================================================
# 12. Time-series plotting
# =============================================================================

def plot_catchment_mean_timeseries(ts_ax, catchment_name):
    """Plot forecast ensemble and ERA5 catchment-mean runoff in panel e)."""
    (
        init_date,
        plot_start,
        plot_end,
        load_start,
        load_end,
        years,
    ) = get_plot_period(
        forecast_date=FORECAST_DATE,
        n_days_before=N_DAYS_BEFORE,
        m_days_lead=M_DAYS_LEAD,
    )

    forecast = load_forecast_catchment_mean(
        catchment_name=catchment_name,
        init_date=init_date,
        plot_end=plot_end,
    )

    era5 = load_era5_catchment_mean(
        catchment_name=catchment_name,
        years=years,
        load_start=load_start,
        load_end=load_end,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    time_name = get_time_coord_name(forecast)
    member_name = get_member_coord_name(forecast)

    ts_ax.plot(
        [],
        [],
        color=ENSEMBLE_COLOR,
        linewidth=ENSEMBLE_LINEWIDTH,
        alpha=ENSEMBLE_ALPHA,
        label="Forecast ensemble",
    )

    for member in forecast[member_name].values:
        if member == ENSEMBLE_MEMBER:
            continue

        ts_ax.plot(
            forecast[time_name].values,
            forecast.sel({member_name: member}).values,
            color=ENSEMBLE_COLOR,
            linewidth=ENSEMBLE_LINEWIDTH,
            alpha=ENSEMBLE_ALPHA,
        )

    selected = forecast.sel({member_name: ENSEMBLE_MEMBER})

    ts_ax.plot(
        selected[time_name].values,
        selected.values,
        color=SELECTED_MEMBER_COLOR,
        linewidth=SELECTED_MEMBER_LINEWIDTH,
        label=f"Selected member {ENSEMBLE_MEMBER}",
        zorder=10,
    )

    for date in EVENT_DATES:
        value = selected.sel(
            {time_name: np.datetime64(date)},
            method="nearest",
        )

        ts_ax.scatter(
            value[time_name].values,
            value.values,
            color=SELECTED_MEMBER_COLOR,
            s=EVENT_MARKER_SIZE,
            zorder=11,
        )

    era5_time_name = get_time_coord_name(era5)

    ts_ax.plot(
        era5[era5_time_name].values,
        era5.values,
        color=ERA5_COLOR,
        linewidth=ERA5_LINEWIDTH,
        label="ERA5",
        zorder=9,
    )

    ts_ax.axvline(
        init_date,
        color="k",
        linewidth=1.2,
        linestyle="--",
        label="Forecast initialization",
    )

    ts_ax.set_title(
        "e) Drammen catchment mean runoff",
        fontsize=TITLE_FONTSIZE,
        pad=5,
    )

    ts_ax.set_ylabel(
        "mm/day",
        fontsize=AXIS_LABELSIZE,
    )

    ts_ax.set_xlabel("Date", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(labelsize=TICK_LABELSIZE)

    ts_ax.grid(True, alpha=0.3)

    ts_ax.set_xlim(plot_start, plot_end)
    ts_ax.margins(x=0)

    plt.setp(
        ts_ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    ts_ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    ts_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
    )


# =============================================================================
# 13. Figure finishing
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


def align_timeseries_axis_to_map_panels(fig, axes, ts_ax):
    """Align panel e with the combined width of panels a-d."""
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
    catchment_boundary,
    catchment_label,
    savepath,
):
    """Add titles, colorbar, legend, inset, panel e, save, and show."""
    plot_axes = get_plot_axes(axes)

    add_panel_titles(plot_axes)
    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, catchment_label)

    plot_catchment_mean_timeseries(
        ts_ax=ts_ax,
        catchment_name=CATCHMENT_NAME,
    )

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
# 14. Main workflow
# =============================================================================

def main():
    """Run full plotting workflow."""
    catchment = get_catchment_settings(CATCHMENT_NAME)

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
        catchment_boundary=catchment_boundary,
        catchment_label=catchment["label"],
        savepath=OUTPUT_FILENAME,
    )


if __name__ == "__main__":
    main()
