#!/usr/bin/env python3
"""
Plot one high-ranking S2S precipitation event as a four-panel map figure.
The event is still ranked using accumulated precipitation, but each map shades only
one user-selected variable. Set PLOT_VARIABLE to "tp24", "msl", "sd", or "sm".
The raw-variable choices match the S2S variable names. "sm" is derived from "sd"
as daily change in snow water equivalent:
    ΔSWE = SWE(current day) - SWE(previous day)
Negative values therefore indicate snowmelt.
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
# User settings

# =============================================================================
CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"
EVENT_MONTH = 5
EVENT_RANK = 1
EVENT_LAGS = [-2, -1, 0, 1]
# Variable shown as colored shading.
# Options: "tp24" (precipitation), "msl", "sd" (snow depth/SWE), "sm" (snowmelt)
PLOT_VARIABLE = "sd"
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]
OBSERVATION_YEARS = ["1957", "2025"]
ACCUMULATION_DAYS = 2
# Options: "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
BIAS_CORRECTION_METHOD = "raw"
# Options: "senorge", "era5"
BIAS_CORRECTION_REFERENCE = "senorge"
SAMPLE_FILENAME_OVERRIDE = None
WRITE_TO_FILE = False
# Raw S2S variable used to rank events.
RANK_VARIABLE = "tp24"

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
FIG_HEIGHT_IN = 9.183464285714285
MAP_EXTENT = [-10, 25, 50, 70]
MAP_WSPACE = 0.02
MAP_HSPACE = 0.08166666666666667
CENTRAL_LON = 10.0
CENTRAL_LAT = 62.0
TICK_LABELSIZE = 12
AXIS_LABELSIZE = 11
TITLE_FONTSIZE = 13
# Preserve the physical panel geometry of the original script.
FIG_LEFT = 0.065
FIG_RIGHT = 0.96
FIG_BOTTOM_IN = 0.84
FIG_TOP_IN = 0.448
CATCHMENT_EDGE_COLOR = "red"
CATCHMENT_LINEWIDTH = 1.0
CATCHMENT_CRS_IF_MISSING = "EPSG:4326"
# Each selectable variable has its own colormap and colorbar configuration.
VARIABLE_SETTINGS = {
    "tp24": {
        "cmap": "GnBu",
        "vmin": 5.0,
        "vmax": 60.0,
        "label": "Precipitation (mm/day)",
        "extend": "max",
    },
    "msl": {
        "cmap": "viridis",
        "vmin": 975.0,
        "vmax": 1040.0,
        "label": "Mean sea level pressure (hPa)",
        "extend": "both",
    },
    "sd": {
        "cmap": "Blues",
        "vmin": 0.0,
        "vmax": 100.0,
        "label": "Snow depth / SWE (mm)",
        "extend": "max",
    },
    "sm": {
        "cmap": "RdBu",
        "vmin": -30.0,
        "vmax": 30.0,
        "label": r"Daily $\Delta$SWE (mm)",
        "extend": "both",
    },
}

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
# Event metadata

# =============================================================================

def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. Valid options are: {valid_names}."
        )
    return CATCHMENTS[catchment_name]

def get_variable_settings(variable):
    """Validate and return plotting settings for the selected variable."""
    if variable not in VARIABLE_SETTINGS:
        valid_variables = ", ".join(VARIABLE_SETTINGS)
        raise ValueError(
            f"Unknown PLOT_VARIABLE '{variable}'. Valid options are: {valid_variables}."
        )
    return VARIABLE_SETTINGS[variable]

def get_file_id(catchment_name):
    """Return the short catchment label used in S2S filenames."""
    if catchment_name.startswith("regine_"):
        return catchment_name.replace("regine_", "", 1)
    return catchment_name

def make_sample_filename(catchment_name):
    """Return the monthly-maximum precipitation sample filename used for ranking."""
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
        f"monthly_max_samples_{RANK_VARIABLE}_{ACCUMULATION_DAYS}dayacc_"
        f"{get_file_id(catchment_name)}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}_{correction_label}.nc"
    )

def validate_sample_dataset(ds):
    """Check variables needed to identify the ranked precipitation event."""
    max_var = f"{RANK_VARIABLE}_max"
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
    if value is None or int(value) == 0:
        return "-"
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"

def get_ranked_event_indices(ds, month_of_year):
    """Return event indices ranked from largest to smallest precipitation."""
    max_var = f"{RANK_VARIABLE}_max"
    values = ds[max_var].transpose("i_date", "number").values
    sample_month = ds["sample_month"].values.astype("int64")
    valid = np.isfinite(values) & (sample_month % 100 == month_of_year)[:, None]
    i_indices, number_indices = np.where(valid)
    if i_indices.size == 0:
        return []
    order = np.argsort(values[i_indices, number_indices])[::-1]
    return [(int(i_indices[i]), int(number_indices[i])) for i in order]

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
        make_s2s_file(event, variable, grid) for grid in ["0.5x0.5", "0.25x0.25"]
    ]
    return next((filename for filename in candidates if filename.is_file()), candidates[0])

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
            f"EVENT_RANK={event_rank} exceeds the {len(ranked_indices)} valid events "
            f"available for month {month_of_year}."
        )
    i_index, number_index = ranked_indices[event_rank - 1]
    sample = ds.isel(i_date=i_index, number=number_index)
    max_var = f"{RANK_VARIABLE}_max"
    hdate = int(ds["hdate"].isel(i_date=i_index).item())
    event = {
        "rank": event_rank,
        "max_value": float(sample[max_var].item()),
        "date_of_max": format_date(sample["date_of_max"].values),
        "model_type": str(ds["model_type"].isel(i_date=i_index).item()),
        "forecast_date": format_date(ds["i_date"].isel(i_date=i_index).values),
        "hdate": None if hdate == 0 else hdate,
        "ensemble_member": int(ds["number"].isel(number=number_index).item()) + 1,
        "lead_of_max": float(sample["lead_of_max"].item()),
    }
    event["source_file"] = find_raw_s2s_file(event, RANK_VARIABLE)
    return event

def get_event_dates(event):
    """Return plotted dates as YYYY-MM-DD strings."""
    date_of_max = np.datetime64(event["date_of_max"], "D")
    return [str(date_of_max + np.timedelta64(lag, "D")) for lag in EVENT_LAGS]

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
    print(f"  plot_variable    : {PLOT_VARIABLE}")
    print("\nDates used for plotting")
    for lag, valid_date in zip(EVENT_LAGS, get_event_dates(event)):
        lead_day = event["lead_of_max"] + lag
        print(f"  lag {lag:+d}: lead_day {lead_day:.0f}, valid_date {valid_date}")

def make_output_filename(catchment_name, event_rank):
    """Create output filename including the selected plotted variable."""
    return (
        f"{PATH_OUT}fig-05-{FORECAST_DATE_RANGE[0]}-{FORECAST_DATE_RANGE[-1]}-"
        f"{catchment_name}-month-{EVENT_MONTH}-rank-{event_rank}-{PLOT_VARIABLE}.png"
    )

# =============================================================================
# Data loading

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

def open_s2s_variable(filename, variable):
    """Open one S2S variable and convert it to plotting units."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    ds = xr.open_dataset(filename)
    if variable not in ds:
        ds.close()
        raise KeyError(f"Variable '{variable}' not found in {filename}")
    if variable == "tp24":
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"
    elif variable == "msl":
        ds[variable] = ds[variable] / 100.0
        ds[variable].attrs["units"] = "hPa"
    elif variable == "sd":
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
        f"Could not find ensemble member {member} in raw {member_name} coordinate. "
        f"Available values: {available.tolist()}"
    )

def select_event_member(ds, event, variable):
    """Select hindcast date and ensemble member from a raw S2S variable."""
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
            f"Using raw {variable} {member_name}={raw_member} for logical ensemble "
            f"member {event['ensemble_member']}."
        )
    return da.sel({member_name: raw_member})

def date_exists(da, target_date):
    """Check whether a target date exists in a DataArray."""
    time_name = get_time_coord_name(da)
    target = np.datetime64(target_date, "ns")
    return target in da[time_name].values.astype("datetime64[ns]")

def select_date(da, target_date):
    """Select one date and load it into memory."""
    time_name = get_time_coord_name(da)
    return da.sel({time_name: np.datetime64(target_date, "ns")}).load()

def load_daily_variable(event, target_date, variable):
    """Load one daily S2S field, trying 0.5° and then 0.25° data."""
    for grid in ["0.5x0.5", "0.25x0.25"]:
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
        f"Could not find {variable} for {target_date} in either 0.5x0.5 or 0.25x0.25 files."
    )

def load_snowmelt(event, target_date):
    """Return daily SWE change from raw sd; negative values indicate snowmelt."""
    current_date = np.datetime64(target_date, "D")
    previous_date = current_date - np.timedelta64(1, "D")
    previous = load_daily_variable(event, str(previous_date), "sd")
    current = load_daily_variable(event, str(current_date), "sd")

    change = current - previous
    change.name = "sm"
    change.attrs.update(units="mm", long_name="Daily change in snow water equivalent")
    return change

def load_plot_variable(event, target_date):
    """Load the selected raw or derived field for one map panel."""
    if PLOT_VARIABLE == "sm":
        return load_snowmelt(event, target_date)
    return load_daily_variable(event, target_date, PLOT_VARIABLE)

def load_catchment_outer_boundary(filename, base_dir, crs_if_missing="EPSG:4326"):
    """Load the catchment and retain only its outer boundary."""
    catchment_path = Path(base_dir) / filename
    gdf = gpd.read_file(catchment_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_if_missing)
    metric_crs = "EPSG:32633"
    union_geom = gdf.to_crs(metric_crs).geometry.union_all()
    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)
    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon([Polygon(poly.exterior) for poly in union_geom.geoms])
    else:
        outer_geom = union_geom
    outer_gdf = gpd.GeoDataFrame(geometry=[outer_geom], crs=metric_crs)
    return outer_gdf.to_crs("EPSG:4326").geometry.iloc[0]

# =============================================================================
# Plotting

# =============================================================================

def make_figure_axes():
    """Create four map panels and one vertical colorbar axis."""
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
    return fig, axes, cbar_ax, proj_data

def plot_shaded_field(ax, da, proj_data):
    """Plot the selected variable as colored grid cells."""
    settings = get_variable_settings(PLOT_VARIABLE)
    lon, lat = get_lon_lat(da)
    values = da.values
    lon_edges = centers_to_edges(lon.values)
    lat_edges = centers_to_edges(lat.values)
    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        values = values[::-1, :]
    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        values = values[:, ::-1]
    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)
    cmap = plt.get_cmap(settings["cmap"]).copy()
    if PLOT_VARIABLE == "tp24":
        cmap.set_under("white")
    return ax.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        values,
        cmap=cmap,
        vmin=settings["vmin"],
        vmax=settings["vmax"],
        shading="auto",
        transform=proj_data,
    )

def plot_catchment_boundary(ax, geometry, proj_data):
    """Overlay the selected catchment boundary."""
    ax.add_geometries(
        [geometry],
        crs=proj_data,
        facecolor="none",
        edgecolor=CATCHMENT_EDGE_COLOR,
        linewidth=CATCHMENT_LINEWIDTH,
        zorder=9,
    )

def plot_event_panel(ax, event, target_date, catchment_boundary, proj_data):
    """Plot one date using only the selected shaded variable."""
    field = load_plot_variable(event, target_date)
    mesh = plot_shaded_field(ax, field, proj_data)
    plot_catchment_boundary(ax, catchment_boundary, proj_data)
    return mesh

def format_panel_date(date):
    """Format a date as 'June 4'."""
    date_object = np.datetime64(date).astype("datetime64[D]").astype(object)
    return f"{date_object.strftime('%B')} {date_object.day}"

def add_panel_titles(axes, event_dates):
    """Add panel labels and calendar dates."""
    panel_labels = ["a)", "b)", "c)", "d)"]
    for ax, panel_label, date in zip(axes.flat, panel_labels, event_dates):
        ax.set_title(
            f"{panel_label} {format_panel_date(date)}",
            fontsize=TITLE_FONTSIZE,
            pad=3,
        )

def add_colorbar(fig, mesh, cbar_ax):
    """Add the variable-specific colorbar."""
    settings = get_variable_settings(PLOT_VARIABLE)
    cbar = fig.colorbar(
        mesh,
        cax=cbar_ax,
        orientation="vertical",
        extend=settings["extend"],
    )
    cbar.set_label(settings["label"], fontsize=AXIS_LABELSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)

def add_legend(axes, catchment_label):
    """Add a legend for the catchment boundary."""
    handle = Line2D(
        [0],
        [0],
        color=CATCHMENT_EDGE_COLOR,
        linewidth=2,
        label=catchment_label,
    )
    legend = axes[0, 0].legend(
        handles=[handle], loc="upper left", frameon=True, fontsize=9
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(100)

def finalize_figure(fig, axes, cbar_ax, mesh, event_dates, catchment_label, savepath):
    """Finish, save, show, and close the figure."""
    add_panel_titles(axes, event_dates)
    add_colorbar(fig, mesh, cbar_ax)
    add_legend(axes, catchment_label)
    bottom = FIG_BOTTOM_IN / FIG_HEIGHT_IN
    top = 1.0 - FIG_TOP_IN / FIG_HEIGHT_IN
    fig.subplots_adjust(
        left=FIG_LEFT,
        right=FIG_RIGHT,
        bottom=bottom,
        top=top,
    )
    if WRITE_TO_FILE:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# =============================================================================
# Main workflow

# =============================================================================

def main():
    """Run the full plotting workflow."""
    get_variable_settings(PLOT_VARIABLE)
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
    fig, axes, cbar_ax, proj_data = make_figure_axes()
    mesh = None
    for ax, target_date in zip(axes.flat, event_dates):
        mesh = plot_event_panel(
            ax=ax,
            event=event,
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
