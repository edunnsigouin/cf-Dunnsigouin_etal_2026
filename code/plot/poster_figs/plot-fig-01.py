#!/usr/bin/env python3
"""
Figure 1: Storm Hans precipitation and pressure maps plus Bergheim streamflow.

Figure idea
-----------
Panels a-d show the weather during four selected Storm Hans dates:

    - precipitation is shaded;
    - ERA5 mean sea level pressure is shown as grey contours;
    - the Drammen catchment boundary is shown in red;
    - the NVE Bergheim hydrological station (station 12.97) is shown as a
      yellow dot.

The precipitation reference is a user choice:

    REFERENCE_DATASET = "senorge"
        plots seNorge daily precipitation.

    REFERENCE_DATASET = "era5"
        plots ERA5 daily accumulated precipitation.

Only the precipitation field changes with REFERENCE_DATASET. The pressure
contours remain ERA5 in both cases.

Panel e shows Bergheim station streamflow during 2023 together with a
historical day-of-year climatology calculated over the user-selected
STREAMFLOW_CLIMATOLOGY_YEARS:

    - the blue shading is the daily 95% interval (2.5th-97.5th percentiles);
    - the red line is the daily median or mean, selected by the user;
    - an optional grey line shows the daily maximum across all available years;
    - the remaining line is the observed daily streamflow during 2023.

For example, STREAMFLOW_CLIMATOLOGY_YEARS = [1921, 2022] keeps the Storm Hans
year (2023) out of the historical comparison.

Station location
----------------
The script first tries to read Bergheim longitude/latitude from the NVE
streamflow NetCDF metadata. Optional user overrides are provided if the local
file does not contain geographic metadata.

Data handling
-------------
seNorge precipitation:
    Variable "rr" is read from the yearly seNorge file. The timestamp shift
    used in the original seNorge script is retained.

ERA5 precipitation:
    Variable "tp24" is read from the yearly ERA5 0.5 x 0.5 degree file and
    converted from metres to millimetres.

ERA5 mean sea level pressure:
    Variable "msl" is converted from Pa to hPa.

Bergheim streamflow:
    Variable "vannforing" is read from streamflow.Bergheim.nc.
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
# 1. USER SETTINGS
# =============================================================================

YEAR = 2023
CATCHMENT_NAME = "drammen"

# Precipitation reference for panels a-d: "senorge" or "era5".
REFERENCE_DATASET = "senorge"

# Historical period used for the panel-e climatology, inclusive.
STREAMFLOW_CLIMATOLOGY_YEARS = [1921, 2025]

# Red climatology line in panel e: "median" or "mean".
STREAMFLOW_CENTER_STATISTIC = "median"

# Plot the maximum for each day of year across all available years in grey.
PLOT_STREAMFLOW_DAILY_MAX = False

# Optional Bergheim station coordinate overrides in decimal degrees.
# Leave both as None to read coordinates from the NVE station NetCDF metadata.
BERGHEIM_LONGITUDE_OVERRIDE = 9.231
BERGHEIM_LATITUDE_OVERRIDE = 60.472

EVENT_DATES = ["2023-08-06", "2023-08-07", "2023-08-08", "2023-08-09"]
WRITE_TO_FILE = False


# =============================================================================
# 2. VARIABLES AND PATHS
# =============================================================================

SENORGE_PRECIP_VAR = "rr"
ERA5_PRECIP_VAR = "tp24"
MSL_VAR = "msl"
STREAMFLOW_VAR = "vannforing"

PATH_OUT = Path(config.dirs["fig"])
PATH_CATCHMENT = Path(config.dirs["nve"])
PATH_SENORGE = Path(config.dirs["senorge_continuous_daily"])
PATH_ERA5 = Path(config.dirs["era5_continuous_daily"])
PATH_STATION = Path(config.dirs["station"])

SENORGE_PRECIP_FILE = PATH_SENORGE / SENORGE_PRECIP_VAR / f"{SENORGE_PRECIP_VAR}_{YEAR}.nc"
ERA5_PRECIP_FILE = PATH_ERA5 / ERA5_PRECIP_VAR / f"{ERA5_PRECIP_VAR}_0.5x0.5_{YEAR}.nc"

MSL_FILE = PATH_ERA5 / MSL_VAR / f"{MSL_VAR}_0.5x0.5_{YEAR}.nc"
STREAMFLOW_FILE = PATH_STATION / "streamflow.Bergheim.nc"
OUTPUT_FILE = PATH_OUT / f"fig-01_{REFERENCE_DATASET}.png"


# =============================================================================
# 3. FIGURE SETTINGS
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

BERGHEIM_MARKER_COLOR = "yellow"
BERGHEIM_MARKER_EDGE_COLOR = "black"
BERGHEIM_MARKER_SIZE = 30
BERGHEIM_MARKER_EDGE_WIDTH = 0.8

TIMESERIES_RANGE_FILL_COLOR = "tab:blue"
TIMESERIES_RANGE_FILL_ALPHA = 0.25
TIMESERIES_CENTER_LINE_COLOR = "tab:red"
TIMESERIES_CENTER_LINEWIDTH = 1.4
TIMESERIES_MAX_LINE_COLOR = "0.55"
TIMESERIES_MAX_LINEWIDTH = 1.0
TIMESERIES_YEAR_LINEWIDTH = 1.2


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

    if STREAMFLOW_CENTER_STATISTIC not in {"median", "mean"}:
        raise ValueError("STREAMFLOW_CENTER_STATISTIC must be 'median' or 'mean'.")

    if not isinstance(PLOT_STREAMFLOW_DAILY_MAX, bool):
        raise TypeError("PLOT_STREAMFLOW_DAILY_MAX must be True or False.")

    if len(EVENT_DATES) != 4:
        raise ValueError("EVENT_DATES must contain exactly four dates.")

    if (
        not isinstance(STREAMFLOW_CLIMATOLOGY_YEARS, (list, tuple))
        or len(STREAMFLOW_CLIMATOLOGY_YEARS) != 2
    ):
        raise ValueError(
            "STREAMFLOW_CLIMATOLOGY_YEARS must contain [first_year, last_year]."
        )

    if STREAMFLOW_CLIMATOLOGY_YEARS[1] < STREAMFLOW_CLIMATOLOGY_YEARS[0]:
        raise ValueError(
            "The last STREAMFLOW_CLIMATOLOGY_YEARS value must be greater than "
            "or equal to the first."
        )

    coordinate_overrides = (
        BERGHEIM_LONGITUDE_OVERRIDE,
        BERGHEIM_LATITUDE_OVERRIDE,
    )
    if any(value is None for value in coordinate_overrides) and not all(
        value is None for value in coordinate_overrides
    ):
        raise ValueError(
            "Set both BERGHEIM_LONGITUDE_OVERRIDE and BERGHEIM_LATITUDE_OVERRIDE, "
            "or leave both as None."
        )

def get_reference_name():
    """Return the publication-style precipitation reference name."""
    return {'senorge': 'seNorge', 'era5': 'ERA5'}[REFERENCE_DATASET]

def get_precip_variable():
    """Return the precipitation variable for the selected reference."""
    return {'senorge': SENORGE_PRECIP_VAR, 'era5': ERA5_PRECIP_VAR}[REFERENCE_DATASET]

def get_precip_filename():
    """Return the yearly precipitation file for the selected reference."""
    return {'senorge': SENORGE_PRECIP_FILE, 'era5': ERA5_PRECIP_FILE}[REFERENCE_DATASET]

def get_catchment_settings(catchment_name):
    """Return metadata for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ', '.join(CATCHMENTS)
        raise ValueError(f"Unknown catchment '{catchment_name}'. Valid options are: {valid_names}.")
    return CATCHMENTS[catchment_name]

def get_time_coord_name(da):
    """Return the time-coordinate name used by a DataArray."""
    for name in ['time', 'valid_time']:
        if name in da.dims or name in da.coords:
            return name
    raise ValueError('Could not identify time coordinate.')

def get_lon_lat(da):
    """Return longitude and latitude coordinates."""
    if 'longitude' in da.coords:
        lon = da['longitude']
    elif 'lon' in da.coords:
        lon = da['lon']
    else:
        raise KeyError('Could not find longitude coordinate.')
    if 'latitude' in da.coords:
        lat = da['latitude']
    elif 'lat' in da.coords:
        lat = da['lat']
    else:
        raise KeyError('Could not find latitude coordinate.')
    return (lon, lat)

def centers_to_edges(centers):
    """Convert one-dimensional grid-cell centres to grid-cell edges."""
    centers = np.asarray(centers)
    if centers.ndim != 1:
        raise ValueError('centers must be one-dimensional.')
    if centers.size < 2:
        raise ValueError('At least two grid-cell centres are required.')
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

def check_file_exists(filename):
    """Raise a clear error if an input file is missing."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f'File not found: {filename}')

def open_precipitation():
    """
    Open precipitation for the selected reference dataset.

    seNorge keeps the timestamp adjustment used in the original script.
    ERA5 precipitation is converted from metres to millimetres.
    """
    filename = get_precip_filename()
    variable = get_precip_variable()
    check_file_exists(filename)
    ds = xr.open_dataset(filename)
    ds = xr.decode_cf(ds)
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )
    if REFERENCE_DATASET == 'senorge':
        if 'time' not in ds.coords:
            raise KeyError("The seNorge precipitation file must contain a 'time' coordinate.")
        ds = ds.assign_coords(time=ds.time - np.timedelta64(6, 'h') - np.timedelta64(24, 'h'))
        ds[variable].attrs['units'] = 'mm/day'
    else:
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs['units'] = 'mm/day'
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
    ds[MSL_VAR].attrs['units'] = 'hPa'
    return ds

def open_streamflow(filename):
    """Open the full NVE Bergheim station streamflow dataset."""
    check_file_exists(filename)
    ds = xr.open_dataset(filename)
    if STREAMFLOW_VAR not in ds:
        raise KeyError(
            f"Variable '{STREAMFLOW_VAR}' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )
    return ds

def get_bergheim_station_coordinates(ds_streamflow):
    """
    Return Bergheim longitude and latitude in decimal degrees.

    The NVE station file is checked for common coordinate names and metadata
    keys. User overrides are used when supplied.
    """
    if BERGHEIM_LONGITUDE_OVERRIDE is not None and BERGHEIM_LATITUDE_OVERRIDE is not None:
        return (float(BERGHEIM_LONGITUDE_OVERRIDE), float(BERGHEIM_LATITUDE_OVERRIDE))
    longitude_names = ['longitude', 'lon', 'station_longitude', 'station_lon']
    latitude_names = ['latitude', 'lat', 'station_latitude', 'station_lat']
    longitude = None
    latitude = None
    for name in longitude_names:
        if name in ds_streamflow.coords:
            value = np.asarray(ds_streamflow.coords[name].values).squeeze()
            if np.size(value) == 1 and np.isfinite(value):
                longitude = float(value)
                break
        if name in ds_streamflow.variables:
            value = np.asarray(ds_streamflow[name].values).squeeze()
            if np.size(value) == 1 and np.isfinite(value):
                longitude = float(value)
                break
        if name in ds_streamflow.attrs:
            try:
                value = float(ds_streamflow.attrs[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                longitude = value
                break
    for name in latitude_names:
        if name in ds_streamflow.coords:
            value = np.asarray(ds_streamflow.coords[name].values).squeeze()
            if np.size(value) == 1 and np.isfinite(value):
                latitude = float(value)
                break
        if name in ds_streamflow.variables:
            value = np.asarray(ds_streamflow[name].values).squeeze()
            if np.size(value) == 1 and np.isfinite(value):
                latitude = float(value)
                break
        if name in ds_streamflow.attrs:
            try:
                value = float(ds_streamflow.attrs[name])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                latitude = value
                break
    if longitude is None or latitude is None:
        raise ValueError(
            "Could not find Bergheim station longitude/latitude in the NVE "
            "streamflow NetCDF metadata. Set BERGHEIM_LONGITUDE_OVERRIDE and "
            "BERGHEIM_LATITUDE_OVERRIDE in the USER SETTINGS section."
        )
    return (longitude, latitude)

def select_date(da, target_date):
    """Select one date from a DataArray."""
    time_name = get_time_coord_name(da)
    target_date = np.datetime64(target_date, 'ns')
    return da.sel({time_name: target_date}).load()

def load_precipitation(ds_precip, target_date):
    """Load selected-reference precipitation for one date."""
    return select_date(ds_precip[get_precip_variable()], target_date)

def load_msl(ds_msl, target_date):
    """Load ERA5 pressure for one date."""
    return select_date(ds_msl[MSL_VAR], target_date)

def load_catchment_outer_boundary(filename, base_dir, crs_if_missing='EPSG:4326'):
    """Load a catchment polygon and keep only its outer boundary."""
    plot_crs = 'EPSG:4326'
    metric_crs = 'EPSG:32633'
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

def year_series_and_climatology_by_doy(da, year):
    """Return the displayed year and historical day-of-year statistics."""
    da = da.dropna("time")

    start = f"{year}-01-01"
    end = f"{year}-12-31"
    x_dates = pd.date_range(start, end, freq="D")

    da_year = da.sel(time=slice(start, end))
    if da_year.sizes.get("time", 0) != len(x_dates):
        da_year = da_year.resample(time="1D").mean().sel(time=slice(start, end))
    y_year = da_year.values

    first_year, last_year = STREAMFLOW_CLIMATOLOGY_YEARS
    historical = da.sel(time=slice(f"{first_year}-01-01", f"{last_year}-12-31"))
    if historical.sizes.get("time", 0) == 0:
        raise ValueError(
            "No Bergheim streamflow data were found in STREAMFLOW_CLIMATOLOGY_YEARS."
        )

    grouped = historical.groupby("time.dayofyear")
    quantiles = grouped.quantile([0.025, 0.5, 0.975], dim="time")
    day_of_year = np.arange(1, 366)

    q_low = quantiles.sel(quantile=0.025, dayofyear=day_of_year, drop=True).values
    q_high = quantiles.sel(quantile=0.975, dayofyear=day_of_year, drop=True).values

    if STREAMFLOW_CENTER_STATISTIC == "median":
        center = quantiles.sel(quantile=0.5, dayofyear=day_of_year, drop=True).values
    else:
        center = grouped.mean(dim="time").sel(dayofyear=day_of_year, drop=True).values

    all_years = da.groupby("time.dayofyear")
    daily_max = all_years.max(dim="time").sel(dayofyear=day_of_year, drop=True).values
    return x_dates, y_year, q_low, q_high, center, daily_max

def make_figure_axes():
    """Create four map panels, one colorbar, and bottom panel e."""
    proj_map = ccrs.LambertConformal(central_longitude=CENTRAL_LON, central_latitude=CENTRAL_LAT)
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
        ax.coastlines(resolution='10m', linewidth=0.5)
        ax.set_extent(MAP_EXTENT, crs=proj_data)
    return (fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data)

def get_map_axes(map_axes):
    """Return the four map axes as a flat list."""
    return list(map_axes.flat)

def align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax):
    """Align panel e with the outer edges of panels a-d."""
    fig.canvas.draw()
    left = min(map_axes[0, 0].get_position().x0, map_axes[1, 0].get_position().x0)
    right = max(map_axes[0, 1].get_position().x1, map_axes[1, 1].get_position().x1)
    position = ts_ax.get_position()
    ts_ax.set_position([left, position.y0, right - left, position.height])

def plot_precipitation(ax, da_precip, proj_data):
    """
    Plot precipitation for the selected reference.

    ERA5 uses explicit grid-cell edges because its longitude/latitude
    coordinates are one-dimensional cell centres. seNorge retains the
    original pcolormesh approach, which also supports its curvilinear grid.
    """
    lon, lat = get_lon_lat(da_precip)
    if REFERENCE_DATASET == 'era5':
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
    """Plot selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )

def plot_bergheim_station(ax, station_longitude, station_latitude, proj_data):
    """Plot the NVE Bergheim station as a yellow map marker."""
    ax.scatter(
        station_longitude,
        station_latitude,
        s=BERGHEIM_MARKER_SIZE,
        marker="o",
        facecolor=BERGHEIM_MARKER_COLOR,
        edgecolor=BERGHEIM_MARKER_EDGE_COLOR,
        linewidth=BERGHEIM_MARKER_EDGE_WIDTH,
        transform=proj_data,
        zorder=10,
    )

def plot_event_map_panel(
    ax,
    ds_precip,
    ds_msl,
    catchment_boundary,
    station_longitude,
    station_latitude,
    target_date,
    proj_data,
):
    """Plot precipitation, pressure, and catchment for one date."""
    da_precip = load_precipitation(ds_precip, target_date)
    da_msl = load_msl(ds_msl, target_date)
    mesh = plot_precipitation(ax, da_precip, proj_data)
    plot_msl_contours(ax, da_msl, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    plot_bergheim_station(
        ax=ax,
        station_longitude=station_longitude,
        station_latitude=station_latitude,
        proj_data=proj_data,
    )
    return mesh

def plot_all_map_panels(
    map_axes,
    ds_precip,
    ds_msl,
    catchment_boundary,
    station_longitude,
    station_latitude,
    proj_data,
):
    """Plot panels a-d, including the Bergheim station marker."""
    mesh = None
    for ax, target_date in zip(get_map_axes(map_axes), EVENT_DATES):
        mesh = plot_event_map_panel(
            ax=ax,
            ds_precip=ds_precip,
            ds_msl=ds_msl,
            catchment_boundary=catchment_boundary,
            station_longitude=station_longitude,
            station_latitude=station_latitude,
            target_date=target_date,
            proj_data=proj_data,
        )
    return mesh

def plot_streamflow_timeseries(ts_ax, ds_streamflow, year):
    """Plot Bergheim streamflow and the selected historical statistics."""
    da = ds_streamflow[STREAMFLOW_VAR]
    x, y, lo, hi, center, daily_max = year_series_and_climatology_by_doy(da, year)

    first_year, last_year = STREAMFLOW_CLIMATOLOGY_YEARS
    period_label = f"{first_year}-{last_year}"

    ts_ax.fill_between(
        x,
        lo,
        hi,
        color=TIMESERIES_RANGE_FILL_COLOR,
        alpha=TIMESERIES_RANGE_FILL_ALPHA,
        linewidth=0,
        label=f"95% interval {period_label}",
    )

    center_label = STREAMFLOW_CENTER_STATISTIC.capitalize()
    ts_ax.plot(
        x,
        center,
        linewidth=TIMESERIES_CENTER_LINEWIDTH,
        color=TIMESERIES_CENTER_LINE_COLOR,
        label=f"{center_label} {period_label}",
    )

    if PLOT_STREAMFLOW_DAILY_MAX:
        ts_ax.plot(
            x,
            daily_max,
            linewidth=TIMESERIES_MAX_LINEWIDTH,
            color=TIMESERIES_MAX_LINE_COLOR,
            label="Daily maximum (all years)",
        )

    ts_ax.plot(x, y, linewidth=TIMESERIES_YEAR_LINEWIDTH, label=f"{year}")

    ts_ax.set_title("e) Bergheim station", fontsize=TITLE_FONTSIZE, pad=5)
    ts_ax.set_ylabel("streamflow [m³/s]", fontsize=AXIS_LABELSIZE)
    ts_ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)
    ts_ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)

    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    ts_ax.set_xlim(start, end)
    ts_ax.margins(x=0)

    ts_ax.xaxis.set_major_locator(mdates.MonthLocator())
    ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ts_ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ts_ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right")

def add_panel_titles(map_axes):
    """Add panel labels and dates to panels a-d."""
    panel_labels = ['a)', 'b)', 'c)', 'd)']
    for ax, label, date in zip(get_map_axes(map_axes), panel_labels, EVENT_DATES):
        formatted_date = (
            np.datetime64(date)
            .astype("datetime64[D]")
            .astype(object)
            .strftime("%B %-d")
        )
        ax.set_title(f'{label} {formatted_date} {YEAR}', fontsize=TITLE_FONTSIZE, pad=3)

def add_precip_colorbar(fig, mesh, cbar_ax):
    """Add precipitation colorbar for the selected reference dataset."""
    colorbar = fig.colorbar(mesh, cax=cbar_ax, orientation='vertical')
    colorbar.set_label(f'{get_reference_name()} precipitation (mm)', fontsize=AXIS_LABELSIZE)
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
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BERGHEIM_MARKER_COLOR,
            markeredgecolor=BERGHEIM_MARKER_EDGE_COLOR,
            markeredgewidth=BERGHEIM_MARKER_EDGE_WIDTH,
            markersize=np.sqrt(BERGHEIM_MARKER_SIZE),
            label="Bergheim station",
        ),
    ]
    legend = map_axes[0, 0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)

def finalize_layout_and_save(fig, map_axes, ts_ax, savepath):
    """Apply final layout, optionally save, and show the figure."""
    fig.subplots_adjust(left=0.065, right=0.96, bottom=0.075, top=0.96)
    align_timeseries_axis_to_map_panels(fig, map_axes, ts_ax)
    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print('Wrote:', savepath)
    plt.show()
    plt.close(fig)

def main():
    """Run the full plotting workflow."""
    validate_user_settings()
    catchment = get_catchment_settings(CATCHMENT_NAME)
    ds_precip = open_precipitation()
    ds_msl = open_era5_msl(MSL_FILE)
    ds_streamflow = open_streamflow(STREAMFLOW_FILE)
    bergheim_longitude, bergheim_latitude = get_bergheim_station_coordinates(ds_streamflow)
    print(f'Bergheim station coordinates: {bergheim_latitude:.5f}°N, {bergheim_longitude:.5f}°E')
    try:
        catchment_boundary = load_catchment_outer_boundary(
            filename=catchment["filename"],
            base_dir=PATH_CATCHMENT,
            crs_if_missing=CATCHMENT_CRS_IF_MISSING,
        )
        fig, map_axes, ts_ax, cbar_ax, proj_map, proj_data = make_figure_axes()
        mesh = plot_all_map_panels(
            map_axes=map_axes,
            ds_precip=ds_precip,
            ds_msl=ds_msl,
            catchment_boundary=catchment_boundary,
            station_longitude=bergheim_longitude,
            station_latitude=bergheim_latitude,
            proj_data=proj_data,
        )
        #plot_streamflow_timeseries(ts_ax=ts_ax, ds_streamflow=ds_streamflow, year=YEAR)
        add_panel_titles(map_axes)
        add_precip_colorbar(fig, mesh, cbar_ax)
        add_map_legend(map_axes, catchment['label'])
        finalize_layout_and_save(fig=fig, map_axes=map_axes, ts_ax=ts_ax, savepath=OUTPUT_FILE)
    finally:
        ds_precip.close()
        ds_msl.close()
        ds_streamflow.close()
if __name__ == '__main__':
    main()
