#!/usr/bin/env python3

"""

Plot one high-ranking S2S precipitation event as a four-panel map figure.

Figure idea

-----------

Panels a-d show four consecutive dates around the selected ranked S2S event.

Each panel contains:

    - daily precipitation as colored shading;

    - mean sea level pressure as labelled grey contours;

    - snowmelt, defined as a negative daily change in snow water equivalent

      (ΔSWE < 0), shown as stippling or hatching;

    - the selected catchment boundary in red.

The four dates are set relative to the event's date of maximum precipitation by

EVENT_LAGS. For example, EVENT_LAGS = [-2, -1, 0, 1] plots two days before the

maximum, one day before, the maximum day, and one day after.

The selected event itself is controlled by:

    CATCHMENT_NAME

        chooses the catchment.

    EVENT_MONTH

        chooses the calendar month used for event ranking.

    EVENT_RANK

        chooses the Nth-largest event read from the monthly_max_samples file.

Panels a-d therefore provide a compact spatial view of how precipitation,

pressure, and snowmelt evolve around one extreme S2S event.

"""

from pathlib import Path

import cartopy.crs as ccrs

import geopandas as gpd

import matplotlib.pyplot as plt

import numpy as np

import xarray as xr

from matplotlib.gridspec import GridSpec

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from shapely.geometry import MultiPolygon, Polygon

from Dunnsigouin_etal_2026 import config

# =============================================================================

# User settings

# =============================================================================

CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"

EVENT_MONTH = 5

EVENT_RANK = 10

FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

OBSERVATION_YEARS = ["1957", "2025"]

ACCUMULATION_DAYS = 2

# Options: "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"

BIAS_CORRECTION_METHOD = "raw"

# Options: "senorge", "era5"

BIAS_CORRECTION_REFERENCE = "senorge"

SAMPLE_FILENAME_OVERRIDE = None

# Dates plotted relative to date_of_max.

EVENT_LAGS = [-2, -1, 0, 1]

PRECIP_VAR = "tp24"

MSL_VAR = "msl"

SNOW_VAR = "sd"

WRITE_TO_FILE = False

# =============================================================================

# Drammen city settings

# =============================================================================

DRAMMEN_LON = 10.2045

DRAMMEN_LAT = 59.7440

DRAMMEN_LABEL = "Drammen"

DRAMMEN_MARKER_SIZE = 5

DRAMMEN_MARKER_FACE_COLOR = "yellow"

DRAMMEN_MARKER_EDGE_COLOR = "black"

DRAMMEN_MARKER_EDGE_WIDTH = 0.6

# =============================================================================

# Paths

# =============================================================================

PATH_OUT = config.dirs["fig"]

PATH_CATCHMENT = config.dirs["nve"]

PATH_S2S_PROCESSED = Path(config.dirs["s2s_processed"])

S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")

# =============================================================================

# Figure settings

# =============================================================================

FIG_WIDTH_IN = 9.4

FIG_HEIGHT_IN = 8.2

MAP_EXTENT = [-10, 25, 50, 70]

MAP_WSPACE = 0.02

MAP_HSPACE = 0.10

CENTRAL_LON = 10.0

CENTRAL_LAT = 62.0

TICK_LABELSIZE = 12

AXIS_LABELSIZE = 11

TITLE_FONTSIZE = 13

CONTOUR_LABELSIZE = 9

# =============================================================================

# Plot styling

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

SNOWMELT_OVERLAY = "stippling"  # options: "stippling", "hatching"

SNOWMELT_THRESHOLD = 0.0

SNOWMELT_DOT_COLOR = "orange"

SNOWMELT_DOT_SIZE = 4

SNOWMELT_DOT_ALPHA = 1.0

SNOWMELT_DOT_STRIDE = 1

SNOWMELT_HATCH_PATTERN = "////"

SNOWMELT_HATCH_COLOR = "orange"

SNOWMELT_HATCH_LINEWIDTH = 1.0

SNOWMELT_ZORDER = 8

# =============================================================================

# Catchment and event metadata

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

# Metadata helpers

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

def get_file_id(catchment_name):

    """Return the short catchment label used in S2S filenames."""

    if catchment_name.startswith("regine_"):

        return catchment_name.replace("regine_", "", 1)

    return catchment_name

def make_sample_filename(catchment_name):

    """Return the monthly-maximum sample filename used for ranking."""

    if SAMPLE_FILENAME_OVERRIDE is not None:

        return Path(SAMPLE_FILENAME_OVERRIDE)

    if BIAS_CORRECTION_METHOD == "raw":

        correction_label = "raw"

    else:

        correction_label = (

            f"bc_{BIAS_CORRECTION_METHOD}_{BIAS_CORRECTION_REFERENCE}_"

            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"

        )

    return PATH_S2S_PROCESSED / (

        f"monthly_max_samples_{PRECIP_VAR}_{ACCUMULATION_DAYS}dayacc_"

        f"{get_file_id(catchment_name)}_"

        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}_"

        f"{correction_label}.nc"

    )

def validate_sample_dataset(ds):

    """Check variables needed to identify and verify the ranked event."""

    max_var = f"{PRECIP_VAR}_max"

    required = {

        max_var,

        "date_of_max",

        "lead_of_max",

        "sample_month",

        "model_type",

        "hdate",

    }

    missing = required - set(ds.variables)

    if missing:

        raise ValueError(f"Sample file is missing variables: {sorted(missing)}")

    if set(ds[max_var].dims) != {"number", "i_date"}:

        raise ValueError(f"{max_var} must have dimensions number and i_date.")

    if set(ds["sample_month"].dims) != {"i_date"}:

        raise ValueError("sample_month must have dimension i_date.")

def format_date(value):

    """Return a compact YYYY-MM-DD date string."""

    value = np.asarray(value).astype("datetime64[ns]")

    if np.isnat(value):

        return "-"

    return np.datetime_as_string(value.astype("datetime64[D]"), unit="D")

def decode_hdate(value):

    """Return YYYY-MM-DD for hindcast hdate, or '-' for forecast rows."""

    if value is None:

        return "-"

    value = int(value)

    if value == 0:

        return "-"

    text = f"{value:08d}"

    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

def get_ranked_event_indices(ds, month_of_year):

    """Return positional event indices ranked from largest to smallest."""

    max_var = f"{PRECIP_VAR}_max"

    values = ds[max_var].transpose("i_date", "number").values

    sample_month = ds["sample_month"].values.astype("int64")

    month_mask = sample_month % 100 == month_of_year

    valid = np.isfinite(values) & month_mask[:, np.newaxis]

    i_indices, number_indices = np.where(valid)

    if i_indices.size == 0:

        return []

    event_values = values[i_indices, number_indices]

    order = np.argsort(event_values)[::-1]

    return [

        (int(i_indices[index]), int(number_indices[index]))

        for index in order

    ]

def make_s2s_file(event, variable, grid):

    """Create the expected raw S2S file path."""

    return (

        S2S_BASE_DIR

        / event["model_type"]

        / "sfc"

        / "daily"

        / "europe"

        / variable

        / f"{variable}_{grid}_{event['forecast_date']}.nc"

    )

def find_raw_s2s_file(event, variable):

    """Return the existing raw S2S file, checking available grids."""

    candidates = [

        make_s2s_file(event, variable, grid)

        for grid in ["0.5x0.5", "0.25x0.25"]

    ]

    for filename in candidates:

        if filename.is_file():

            return filename

    return candidates[0]

def get_selected_event(catchment_name, month_of_year, event_rank):

    """Read the sample file and return metadata for the Nth ranked event."""

    if not 1 <= month_of_year <= 12:

        raise ValueError("EVENT_MONTH must be between 1 and 12.")

    if event_rank < 1:

        raise ValueError("EVENT_RANK must be at least 1.")

    filename = make_sample_filename(catchment_name)

    if not filename.is_file():

        raise FileNotFoundError(f"Sample file not found: {filename}")

    print("Reading ranked events:", filename)

    with xr.open_dataset(filename, decode_timedelta=False) as opened:

        ds = opened.load()

    validate_sample_dataset(ds)

    ranked_indices = get_ranked_event_indices(ds, month_of_year)

    if event_rank > len(ranked_indices):

        raise ValueError(

            f"EVENT_RANK={event_rank} exceeds the {len(ranked_indices)} "

            f"valid events available for month {month_of_year}."

        )

    i_index, number_index = ranked_indices[event_rank - 1]

    sample = ds.isel(i_date=i_index, number=number_index)

    max_var = f"{PRECIP_VAR}_max"

    hdate = int(ds["hdate"].isel(i_date=i_index).item())

    event = {

        "rank": event_rank,

        "max_value": float(sample[max_var].item()),

        "date_of_max": format_date(sample["date_of_max"].values),

        "model_type": str(ds["model_type"].isel(i_date=i_index).item()),

        "forecast_date": format_date(ds["i_date"].isel(i_date=i_index).values),

        "hdate": None if hdate == 0 else hdate,

        # Sample members are zero-based; logical ECMWF members are 1-11.

        "ensemble_member": int(ds["number"].isel(number=number_index).item()) + 1,

        "lead_of_max": float(sample["lead_of_max"].item()),

    }

    event["source_file"] = find_raw_s2s_file(event, PRECIP_VAR)

    return event

def get_event_dates(event):

    """Return dates to plot as strings."""

    date_of_max = np.datetime64(event["date_of_max"], "D")

    return [

        str(date_of_max + np.timedelta64(lag, "D"))

        for lag in EVENT_LAGS

    ]

def print_selected_event(event):
    """Print ranked-event metadata and plotted lead/valid dates."""
    print(f"\nRank {event['rank']}")
    print(f"  max_value        : {event['max_value']:.2f} mm")
    print(f"  date_of_max      : {event['date_of_max']}")
    print(f"  model_type       : {event['model_type']}")
    print(f"  forecast_date    : {event['forecast_date']}")
    print(f"  hdate            : {decode_hdate(event['hdate'])}")
    print(f"  ensemble_member  : {event['ensemble_member']}")
    print(f"  lead_of_max      : {event['lead_of_max']:.0f} days")
    print(f"  source_file      : {event['source_file']}")

    print("\nDates used for plotting")
    for lag, valid_date in zip(EVENT_LAGS, get_event_dates(event)):
        lead_day = event["lead_of_max"] + lag
        print(
            f"  lag {lag:+d}: lead_day {lead_day:.0f}, "
            f"valid_date {valid_date}"
        )


def make_output_filename(catchment_name, event_rank):

    """Create output filename."""

    return f"{PATH_OUT}fig-05.png"

# =============================================================================

# Coordinate helpers

# =============================================================================

def get_time_coord_name(da):

    """Return the time coordinate name used by the DataArray."""

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

def get_hdate_coord_name(da):

    """Return hindcast date coordinate name."""

    for name in ["hdate", "hindcast_date"]:

        if name in da.dims or name in da.coords:

            return name

    raise ValueError("Could not identify hindcast date coordinate.")

def centers_to_edges(centers):

    """Convert 1D grid-cell center coordinates to grid-cell edges."""

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

def open_s2s_variable(filename, variable):

    """Open one S2S variable and convert it to plotting units."""

    filename = Path(filename)

    if not filename.exists():

        raise FileNotFoundError(f"File not found: {filename}")

    ds = xr.open_dataset(filename)

    if variable not in ds:

        raise KeyError(f"Variable '{variable}' not found in {filename}")

    if variable == PRECIP_VAR:

        ds[variable] = ds[variable] * 1000.0

        ds[variable].attrs["units"] = "mm/day"

    elif variable == MSL_VAR:

        ds[variable] = ds[variable] / 100.0

        ds[variable].attrs["units"] = "hPa"

    elif variable == SNOW_VAR:

        ds[variable] = ds[variable] * 1000.0

        ds[variable].attrs["units"] = "mm"

    return ds

def get_raw_member_label(da, event):
    """Return the member label available in the current raw S2S variable."""
    member = int(event["ensemble_member"])
    member_name = get_member_coord_name(da)
    available = np.asarray(da[member_name].values)

    if member in available:
        return member

    if event["model_type"] == "hindcast" and member == 11 and 51 in available:
        return 51

    raise KeyError(
        f"Could not find ensemble member {member} in raw {member_name} "
        f"coordinate. Available values: {available.tolist()}"
    )


def select_event_member(ds, event, variable):
    """Select hindcast date and ensemble member from the raw S2S file."""
    da = ds[variable]

    if event["model_type"] == "hindcast":
        for name in ["hdate", "hindcast_date"]:
            if name in da.dims or name in da.coords:
                da = da.sel({name: event["hdate"]})
                break

    member_name = get_member_coord_name(da)
    raw_member = get_raw_member_label(da, event)

    if raw_member != event["ensemble_member"]:
        print(
            f"Using raw {variable} {member_name}={raw_member} "
            f"for logical ensemble member {event['ensemble_member']}."
        )

    return da.sel({member_name: raw_member})


def date_exists(da, target_date):

    """Check whether a target date exists in a DataArray."""

    time_name = get_time_coord_name(da)

    target_date = np.datetime64(target_date, "ns")

    available_dates = da[time_name].values.astype("datetime64[ns]")

    return target_date in available_dates

def select_date(da, target_date):

    """Select one date from a DataArray and load it into memory."""

    time_name = get_time_coord_name(da)

    target_date = np.datetime64(target_date, "ns")

    return da.sel({time_name: target_date}).load()

def load_daily_variable(event, target_date, variable):

    """

    Load one daily S2S field.

    The script first tries the 0.5-degree file. If the target date is missing,

    it then tries the 0.25-degree file.

    """

    grids_to_try = ["0.5x0.5", "0.25x0.25"]

    for grid in grids_to_try:

        filename = make_s2s_file(event, variable, grid)

        if not filename.exists():

            continue

        ds = open_s2s_variable(filename, variable)

        try:

            da = select_event_member(ds, event, variable)

            if date_exists(da, target_date):

                return select_date(da, target_date)

        finally:

            ds.close()

    raise ValueError(

        f"Could not find {variable} for {target_date} "

        f"in either 0.5x0.5 or 0.25x0.25 files."

    )

def load_precipitation(event, target_date):

    """Load daily precipitation in mm/day."""

    return load_daily_variable(event, target_date, PRECIP_VAR)

def load_msl(event, target_date):

    """Load mean sea level pressure in hPa."""

    return load_daily_variable(event, target_date, MSL_VAR)

def load_snowmelt(event, lag):

    """

    Compute daily SWE change.

    Snowmelt is later plotted where:

        SWE(current day) - SWE(previous day) < 0

    """

    date_of_max = np.datetime64(event["date_of_max"], "D")

    previous_date = date_of_max + np.timedelta64(lag - 1, "D")

    current_date = date_of_max + np.timedelta64(lag, "D")

    da_previous = load_daily_variable(event, str(previous_date), SNOW_VAR)

    da_current = load_daily_variable(event, str(current_date), SNOW_VAR)

    da_change = da_current - da_previous

    da_change.attrs["units"] = "mm"

    da_change.attrs["long_name"] = "Daily change in snow water equivalent"

    return da_change

def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):

    """Load the catchment and keep only the outer boundary."""

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

def make_figure_axes():

    """Create four map panels and one vertical precipitation colorbar."""

    proj_map = ccrs.LambertConformal(

        central_longitude=CENTRAL_LON,

        central_latitude=CENTRAL_LAT,

    )

    proj_data = ccrs.PlateCarree()

    fig = plt.figure(

        figsize=(

            FIG_WIDTH_IN,

            FIG_HEIGHT_IN,

        )

    )

    gs = GridSpec(

        2,

        3,

        figure=fig,

        width_ratios=[

            1.0,

            1.0,

            0.045,

        ],

        height_ratios=[

            1.0,

            1.0,

        ],

        wspace=MAP_WSPACE,

        hspace=MAP_HSPACE,

    )

    axes = np.empty(

        (

            2,

            2,

        ),

        dtype=object,

    )

    axes[

        0,

        0,

    ] = fig.add_subplot(

        gs[

            0,

            0,

        ],

        projection=proj_map,

    )

    axes[

        0,

        1,

    ] = fig.add_subplot(

        gs[

            0,

            1,

        ],

        projection=proj_map,

    )

    axes[

        1,

        0,

    ] = fig.add_subplot(

        gs[

            1,

            0,

        ],

        projection=proj_map,

    )

    axes[

        1,

        1,

    ] = fig.add_subplot(

        gs[

            1,

            1,

        ],

        projection=proj_map,

    )

    cbar_ax = fig.add_subplot(

        gs[

            :,

            2,

        ]

    )

    for ax in axes.flat:

        ax.coastlines(

            resolution="10m",

            linewidth=0.5,

        )

        ax.set_extent(

            MAP_EXTENT,

            crs=proj_data,

        )

    return (

        fig,

        axes,

        cbar_ax,

        proj_map,

        proj_data,

    )

def get_plot_axes(axes):

    """Return the four map panels."""

    return list(axes.flat)

# =============================================================================

# Plotting functions

# =============================================================================

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

def plot_snowmelt(ax, da_snowmelt, proj_data):

    """Overlay locations where daily SWE change is negative."""

    lon, lat = get_lon_lat(da_snowmelt)

    snowmelt = np.isfinite(da_snowmelt.values)

    snowmelt &= da_snowmelt.values < SNOWMELT_THRESHOLD

    if SNOWMELT_OVERLAY == "stippling":

        plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data)

    elif SNOWMELT_OVERLAY == "hatching":

        plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data)

    else:

        raise ValueError(

            "SNOWMELT_OVERLAY must be either 'stippling' or 'hatching'."

        )

def plot_snowmelt_stippling(ax, lon, lat, snowmelt, proj_data):

    """Plot snowmelt as orange stippling."""

    iy, ix = np.where(snowmelt)

    if SNOWMELT_DOT_STRIDE > 1:

        iy = iy[::SNOWMELT_DOT_STRIDE]

        ix = ix[::SNOWMELT_DOT_STRIDE]

    ax.scatter(

        lon.values[ix],

        lat.values[iy],

        s=SNOWMELT_DOT_SIZE,

        c=SNOWMELT_DOT_COLOR,

        alpha=SNOWMELT_DOT_ALPHA,

        transform=proj_data,

        zorder=SNOWMELT_ZORDER,

        linewidths=0,

    )

def plot_snowmelt_hatching(ax, lon, lat, snowmelt, proj_data):

    """Plot snowmelt as orange hatching."""

    old_hatch_color = plt.rcParams["hatch.color"]

    old_hatch_linewidth = plt.rcParams["hatch.linewidth"]

    try:

        plt.rcParams["hatch.color"] = SNOWMELT_HATCH_COLOR

        plt.rcParams["hatch.linewidth"] = SNOWMELT_HATCH_LINEWIDTH

        ax.contourf(

            lon.values,

            lat.values,

            snowmelt.astype(int),

            levels=[0.5, 1.5],

            colors="none",

            hatches=[SNOWMELT_HATCH_PATTERN],

            transform=proj_data,

            zorder=SNOWMELT_ZORDER,

        )

    finally:

        plt.rcParams["hatch.color"] = old_hatch_color

        plt.rcParams["hatch.linewidth"] = old_hatch_linewidth

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

def plot_event_panel(ax, event, lag, target_date, catchment_boundary, proj_data):

    """Plot one map panel."""

    da_precip = load_precipitation(event, target_date)

    da_msl = load_msl(event, target_date)

    da_snowmelt = load_snowmelt(event, lag)

    mesh = plot_precipitation(ax, da_precip, proj_data)

    plot_msl_contours(ax, da_msl, proj_data)

    plot_snowmelt(ax, da_snowmelt, proj_data)

    plot_catchment_boundary(ax, catchment_boundary, proj_data)

    # plot_drammen_city(ax, proj_data)

    return mesh

# =============================================================================

# Figure finishing

# =============================================================================

def format_panel_date(date):

    """Format a date as 'Jun 4' for panel titles."""

    date_object = (

        np.datetime64(date)

        .astype("datetime64[D]")

        .astype(object)

    )

    return f"{date_object.strftime('%B')} {date_object.day}"

def add_panel_titles(axes, event_dates):

    """Add panel labels and calendar dates."""

    panel_labels = ["a)", "b)", "c)", "d)"]

    for ax, panel_label, date in zip(

        axes,

        panel_labels,

        event_dates,

    ):

        ax.set_title(

            f"{panel_label} {format_panel_date(date)}",

            fontsize=TITLE_FONTSIZE,

            pad=3,

        )

def add_colorbar(fig, mesh, cbar_ax):

    """Add vertical precipitation colorbar beside the map panels."""

    cbar = fig.colorbar(

        mesh,

        cax=cbar_ax,

        orientation="vertical",

    )

    cbar.set_label(

        "precipitation (mm)",

        fontsize=AXIS_LABELSIZE,

    )

    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)

def get_snowmelt_legend_handle():

    """Return legend handle for the selected snowmelt overlay."""

    if SNOWMELT_OVERLAY == "hatching":

        return Patch(

            facecolor="white",

            edgecolor=SNOWMELT_HATCH_COLOR,

            hatch=SNOWMELT_HATCH_PATTERN,

            label=r"Snowmelt ($\Delta$SWE < 0)",

        )

    return Line2D(

        [0],

        [0],

        marker="o",

        color="none",

        markerfacecolor=SNOWMELT_DOT_COLOR,

        markersize=5,

        label=r"Snowmelt ($\Delta$SWE < 0)",

    )

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

        get_snowmelt_legend_handle(),

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

    mesh,

    event_dates,

    catchment_label,

    savepath,

):

    """Add titles, colorbar, legend, save the figure, and show it."""

    plot_axes = get_plot_axes(

        axes

    )

    add_panel_titles(

        plot_axes,

        event_dates,

    )

    add_colorbar(

        fig,

        mesh,

        cbar_ax,

    )

    add_legend(

        axes,

        catchment_label,

    )

    fig.subplots_adjust(

        left=0.07,

        right=0.96,

        bottom=0.06,

        top=0.96,

    )

    if WRITE_TO_FILE:

        fig.savefig(

            savepath,

            dpi=300,

            bbox_inches="tight",

        )

    plt.show()

    plt.close(

        fig

    )

# =============================================================================

# Main workflow

# =============================================================================

def main():

    """Run the full plotting workflow."""

    catchment = get_catchment_settings(CATCHMENT_NAME)

    event = get_selected_event(

        catchment_name=catchment["weights_id"],

        month_of_year=EVENT_MONTH,

        event_rank=EVENT_RANK,

    )

    print_selected_event(event)

    event_dates = get_event_dates(event)

    savepath = make_output_filename(CATCHMENT_NAME, EVENT_RANK)

    catchment_boundary = load_catchment_outer_boundary(

        filename=catchment["filename"],

        base_dir=PATH_CATCHMENT,

        crs_if_missing=CATCHMENT_CRS_IF_MISSING,

    )

    fig, axes, cbar_ax, proj_map, proj_data = make_figure_axes()

    mesh = None

    for ax, lag, target_date in zip(

        get_plot_axes(axes),

        EVENT_LAGS,

        event_dates,

    ):

        mesh = plot_event_panel(

            ax=ax,

            event=event,

            lag=lag,

            target_date=target_date,

            catchment_boundary=catchment_boundary,

            proj_data=proj_data,

        )

    finalize_figure(

        fig=fig,

        axes=axes,

        cbar_ax=cbar_ax,

        mesh=mesh,

        event_dates=event_dates,

        catchment_label=catchment["label"],

        savepath=savepath,

    )

if __name__ == "__main__":

    main()
