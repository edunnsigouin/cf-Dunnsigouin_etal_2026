#!/usr/bin/env python3
"""
Create a four-panel schematic of the Drammen catchment and S2S sampling strategy.

Panel (a) shows the Drammen catchment weights on the ERA5 0.5-degree grid and
the catchment boundary. Panel (b) shows one ECMWF S2S forecast after catchment
averaging and temporal accumulation. All ensemble members are grey, one example
member is highlighted in blue, its later-period maximum is marked, and a dashed
line marks the start of the later lead-day sampling period.

Panels (c) and (d) use the compact monthly-maximum sample file. Panel (c)
aggregates Hindcast, Forecast, and All realization counts by calendar month across
all years. Panel (d) shows the number of realizations assigned to each YYYYMM.

The figure uses a 2 x 2 layout with equal grid-cell sizes. Panel (a) uses a
regular subplot as an outer container, with the Cartopy map drawn as an inset
inside that container. This preserves the map projection while making the outer
panel geometry align more closely with panels (b)-(d). A shared typography block
controls title, axis-label, tick-label, legend, and colorbar font sizes consistently
across all four panels.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings: shared data
# =============================================================================

variable = "tp24"
catchment = "regine_drammen"
accumulation_days = 2

write2file = True
show_figure = True


# =============================================================================
# User settings: panel (a) Drammen catchment map
# =============================================================================

map_title = "a) Model weights for drammen catchment"
map_title_x = -0.3  # Shift title left so it begins above the colorbar.
map_cmap = "BuGn"
map_vmin = 0.0
map_vmax = 1.0

map_central_lon = 10.0
map_central_lat = 62.0
map_extent = [4.75, 12.75, 58.0, 63.0]

map_coastline_width = 0.5
map_border_width = 0.4

catchment_boundary_color = "red"
catchment_boundary_width = 1.5
catchment_crs_if_missing = "EPSG:4326"

colorbar_label = "Area fraction"

# Position of the Cartopy map inside the equal-sized panel-(a) container:
# [left, bottom, width, height] in container-axis coordinates.
#map_inset_bounds = [0.08, 0.06, 0.84, 0.88]
map_inset_bounds = [0.0, 0.0, 1.0, 1.0]

# Position of the vertical colorbar inside the same panel container.
map_colorbar_bounds = [0.05, 0.0, 0.075, 1.0]


# =============================================================================
# User settings: panel (b) example forecast
# =============================================================================

forecast_date = "2023-07-24"
input_filename_prefix = f"{variable}_0.5x0.5"

forecast_title = "b) Example of forecast sampling for August"
# Ensemble member to highlight by positional index: 0 is the first member.
highlight_member_index = 0

ensemble_color = "0.65"
ensemble_alpha = 0.35
ensemble_line_width = 0.8

highlight_color = "tab:blue"
highlight_line_width = 2.0
maximum_marker_size = 60

later_group_shading_color = "tab:orange"
later_group_shading_alpha = 0.25

x_label_interval = 5
x_label_rotation = 30
forecast_y_limits = [0, 90]

forecast_grid_linewidth = 0.7
forecast_grid_alpha = 0.4


# =============================================================================
# User settings: panels (c) and (d) realization counts
# =============================================================================

# Optional explicit compact sample filename. Leave as None to construct it.
sample_input_filename_override = None

forecast_date_range = ["2020-01-02", "2023-12-28"]
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2

# Compact sample source.
input_data_type = "raw"  # "raw" or "bias_corrected"
bias_correction_method = "ld"  # "q", "doy", "ld", or "q_doy"
bias_correction_reference = "era5"  # "senorge" or "era5"

# Optional time limits. Use None for the complete sample_month range.
plot_start_month = None  # e.g. 200001
plot_end_month = None  # e.g. 202212

count_colors = {
    "All": "tab:blue",
    "Forecast": "tab:orange",
    "Hindcast": "tab:green",
}
count_linewidth = 1.7
count_marker_size = 4.0

count_show_grid = True
count_grid_linewidth = 0.7
count_grid_alpha = 0.45

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# User settings: overall figure
# =============================================================================

figure_width = 12
figure_height = 10
figure_dpi = 300
figure_wspace = 0.15
figure_hspace = 0.35

# Shared typography for all panels.
title_fontsize = 12
axis_label_fontsize = 11
tick_label_fontsize = 10
legend_fontsize = 9

filename_out = (
    Path(config.dirs["fig"])
    / f"fig-02-{forecast_date_range[0]}-{forecast_date_range[-1]}.png"
)


# =============================================================================
# Paths
# =============================================================================

path_nve = Path(config.dirs["nve"])
path_in_forecast = Path(config.dirs["s2s_forecast_daily"]) / variable

filename_weights = path_nve / f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
filename_catchment = path_nve / f"catchment_nve_{catchment}.geojson"


# =============================================================================
# Filename helpers
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in filenames."""
    return catchment_name.removeprefix("regine_")


def split_usable_leads(first_lead, last_lead, number_of_bins):
    """Split an inclusive lead interval into consecutive near-equal bins."""
    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder)
        for index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1

    return bins


def build_lead_bins():
    """Return the usable accumulated lead bins encoded in the sample filename."""
    first_usable_lead = first_input_lead + accumulation_days - 1
    return split_usable_leads(
        first_usable_lead,
        last_input_lead,
        number_of_lead_bins,
    )


def make_sample_input_filename():
    """Construct the compact monthly-maximum sample filename."""
    if sample_input_filename_override is not None:
        return Path(sample_input_filename_override)

    first_usable_lead = first_input_lead + accumulation_days - 1
    bin_label = "_".join(f"{start}-{end}" for start, end in build_lead_bins())

    stem = (
        f"test-monthly_max_samples_{variable}_{accumulation_days}dayacc_"
        f"{get_file_id(catchment)}_lead{first_usable_lead}-{last_input_lead}_"
        f"split{number_of_lead_bins}_{bin_label}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}"
    )

    if input_data_type == "bias_corrected":
        stem += f"_bc_{bias_correction_method}_{bias_correction_reference}"

    return Path(config.dirs["s2s_processed"]) / f"{stem}.nc"


def find_forecast_file():
    """Return the single raw forecast file matching forecast_date."""
    matches = sorted(
        path
        for path in path_in_forecast.glob(f"{input_filename_prefix}*.nc")
        if path.stem.endswith(f"_{forecast_date}")
    )

    if not matches:
        raise FileNotFoundError(
            f"No forecast file ending in _{forecast_date}.nc was found in "
            f"{path_in_forecast}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple forecast files for {forecast_date}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


# =============================================================================
# Validation
# =============================================================================

def validate_sample_month_value(value, name):
    """Validate one YYYYMM integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer YYYYMM value.")

    year = int(value) // 100
    month = int(value) % 100
    if year < 1 or month not in range(1, 13):
        raise ValueError(f"{name} must use valid YYYYMM format.")


def validate_user_settings():
    """Validate user settings and required input files."""
    try:
        np.datetime64(forecast_date)
    except ValueError as exc:
        raise ValueError("forecast_date must have format YYYY-MM-DD.") from exc

    if not 1 <= accumulation_days <= 31:
        raise ValueError("accumulation_days must be between 1 and 31.")
    if highlight_member_index < 0:
        raise ValueError("highlight_member_index must be zero or greater.")
    if x_label_interval < 1:
        raise ValueError("x_label_interval must be at least 1.")
    if map_vmin >= map_vmax:
        raise ValueError("map_vmin must be smaller than map_vmax.")

    if first_input_lead > last_input_lead:
        raise ValueError("first_input_lead must not exceed last_input_lead.")

    first_usable_lead = first_input_lead + accumulation_days - 1
    if first_usable_lead > last_input_lead:
        raise ValueError("accumulation_days is too large for the sample lead range.")

    usable_leads = last_input_lead - first_usable_lead + 1
    if not isinstance(number_of_lead_bins, int) or not 1 <= number_of_lead_bins <= usable_leads:
        raise ValueError("number_of_lead_bins is invalid for the usable lead range.")

    if input_data_type not in {"raw", "bias_corrected"}:
        raise ValueError("input_data_type must be 'raw' or 'bias_corrected'.")

    if input_data_type == "bias_corrected":
        if bias_correction_method not in {"q", "doy", "ld", "q_doy"}:
            raise ValueError(
                "bias_correction_method must be 'q', 'doy', 'ld', or 'q_doy'."
            )
        if bias_correction_reference not in {"senorge", "era5"}:
            raise ValueError(
                "bias_correction_reference must be 'senorge' or 'era5'."
            )

    for name, value in {
        "plot_start_month": plot_start_month,
        "plot_end_month": plot_end_month,
    }.items():
        if value is not None:
            validate_sample_month_value(value, name)

    if plot_start_month is not None and plot_end_month is not None:
        if plot_end_month < plot_start_month:
            raise ValueError("plot_end_month must not precede plot_start_month.")

    for filename in [filename_weights, filename_catchment, make_sample_input_filename()]:
        if not filename.is_file():
            raise FileNotFoundError(f"Required file not found: {filename}")

    if not path_in_forecast.is_dir():
        raise FileNotFoundError(f"Forecast directory not found: {path_in_forecast}")

    find_forecast_file()


# =============================================================================
# Panel (a): Drammen catchment map
# =============================================================================

def load_catchment_weights():
    """Load the Drammen catchment weights."""
    with xr.open_dataset(filename_weights) as ds:
        if "catchment_weight" not in ds:
            raise KeyError(
                f"'catchment_weight' was not found in {filename_weights}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return ds["catchment_weight"].load()


def load_catchment_boundary():
    """Load and dissolve the Drammen catchment to its outer boundary."""
    plot_crs = "EPSG:4326"
    metric_crs = "EPSG:32633"

    gdf = gpd.read_file(filename_catchment)
    if gdf.crs is None:
        gdf = gdf.set_crs(catchment_crs_if_missing)

    union_geom = gdf.to_crs(metric_crs).geometry.union_all()
    if isinstance(union_geom, Polygon):
        outer_geom = Polygon(union_geom.exterior)
    elif isinstance(union_geom, MultiPolygon):
        outer_geom = MultiPolygon([Polygon(poly.exterior) for poly in union_geom.geoms])
    else:
        outer_geom = union_geom

    return (
        gpd.GeoDataFrame(geometry=[outer_geom], crs=metric_crs)
        .to_crs(plot_crs)
        .geometry.iloc[0]
    )


def centers_to_edges(centers):
    """Convert one-dimensional grid-cell centers to cell edges."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("centers must be one-dimensional with at least two values.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def plot_catchment_panel(axis, weights, boundary, data_crs):
    """Plot the Drammen catchment weights and boundary."""
    lats = weights["latitude"].values
    lons = weights["longitude"].values
    values = weights.values.copy()

    lat_edges = centers_to_edges(lats)
    lon_edges = centers_to_edges(lons)
    values = np.where(values == 0, np.nan, values)

    if lat_edges[0] > lat_edges[-1]:
        lat_edges = lat_edges[::-1]
        values = values[::-1, :]
    if lon_edges[0] > lon_edges[-1]:
        lon_edges = lon_edges[::-1]
        values = values[:, ::-1]

    lon_edges_2d, lat_edges_2d = np.meshgrid(lon_edges, lat_edges)
    cmap = plt.get_cmap(map_cmap).copy()
    cmap.set_bad("white")

    mesh = axis.pcolormesh(
        lon_edges_2d,
        lat_edges_2d,
        values,
        cmap=cmap,
        vmin=map_vmin,
        vmax=map_vmax,
        shading="auto",
        transform=data_crs,
    )
    axis.add_geometries(
        [boundary],
        crs=data_crs,
        facecolor="none",
        edgecolor=catchment_boundary_color,
        linewidth=catchment_boundary_width,
        zorder=5,
    )

    axis.coastlines(resolution="10m", linewidth=map_coastline_width)
    axis.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=map_border_width)
    axis.set_extent(map_extent, crs=data_crs)
    axis.set_title(
        map_title,
        loc="left",
        x=map_title_x,
        fontsize=title_fontsize,
    )

    return mesh


# =============================================================================
# Panel (b): forecast sampling
# =============================================================================

def catchment_weighted_mean(ds, catchment_weight):
    """Calculate catchment- and latitude-weighted mean precipitation in mm."""
    if variable not in ds:
        raise KeyError(
            f"'{variable}' was not found in the forecast dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    for dimension in ["latitude", "longitude"]:
        if dimension not in ds[variable].dims:
            raise ValueError(
                f"'{variable}' does not contain the expected '{dimension}' dimension. "
                f"Found dimensions: {ds[variable].dims}"
            )

    precipitation = xr.where(ds[variable] < 0, 0, ds[variable])
    latitude_weight = np.cos(np.deg2rad(ds["latitude"]))
    combined_weight = catchment_weight * latitude_weight
    spatial_mean = precipitation.weighted(combined_weight).mean(
        ["latitude", "longitude"]
    )
    return (spatial_mean * 1000.0).rename(variable)


def calculate_accumulation(precipitation):
    """Calculate trailing accumulation and assign ending lead days 1-46."""
    if "time" not in precipitation.dims:
        raise ValueError(
            f"'{variable}' must contain a 'time' dimension; found {precipitation.dims}."
        )

    number_of_times = precipitation.sizes["time"]
    if number_of_times != 46:
        raise ValueError(
            f"This schematic expects 46 raw forecast time steps, but found "
            f"{number_of_times}."
        )

    accumulated = precipitation.rolling(
        time=accumulation_days,
        min_periods=accumulation_days,
    ).sum()

    lead_days = xr.DataArray(
        np.arange(1, 47, dtype="int64"),
        dims=("time",),
        coords={"time": accumulated["time"]},
        name="lead_day",
    )
    return accumulated.assign_coords(lead_day=lead_days)


def get_lead_groups():
    """Return early and later accumulated ending-lead ranges."""
    first_usable_lead = accumulation_days
    later_start = 15 + accumulation_days
    early = np.arange(first_usable_lead, later_start, dtype="int64")
    later = np.arange(later_start, 47, dtype="int64")
    return early, later


def get_labelled_lead_days():
    """Return lead days labelled on both x-axes."""
    return np.arange(1, 47, x_label_interval, dtype="int64")


def plot_forecast_panel(axis, accumulated):
    """Plot the ensemble and one highlighted member with its later-period maximum."""
    if "number" not in accumulated.dims:
        raise ValueError(
            f"'{variable}' must contain an ensemble 'number' dimension; "
            f"found {accumulated.dims}."
        )
    if highlight_member_index >= accumulated.sizes["number"]:
        raise ValueError(
            f"highlight_member_index={highlight_member_index} is outside the available "
            f"range 0-{accumulated.sizes['number'] - 1}."
        )

    _, later_leads = get_lead_groups()
    later = accumulated.where(accumulated["lead_day"].isin(later_leads), drop=True)

    for member_index in range(accumulated.sizes["number"]):
        if member_index == highlight_member_index:
            continue

        member = accumulated.isel(number=member_index).squeeze(drop=True)
        axis.plot(
            member["time"].values,
            member.values,
            color=ensemble_color,
            alpha=ensemble_alpha,
            linewidth=ensemble_line_width,
        )

    highlighted = accumulated.isel(number=highlight_member_index).squeeze(drop=True)
    highlighted_later = later.isel(number=highlight_member_index).squeeze(drop=True)

    axis.plot(
        highlighted["time"].values,
        highlighted.values,
        color=highlight_color,
        linewidth=highlight_line_width,
        zorder=4,
    )

    later_values = np.asarray(highlighted_later.values, dtype=float)
    if np.isfinite(later_values).any():
        maximum_index = int(np.nanargmax(later_values))
        axis.scatter(
            highlighted_later["time"].values[maximum_index],
            later_values[maximum_index],
            marker="o",
            s=maximum_marker_size,
            color=highlight_color,
            edgecolors="none",
            zorder=5,
        )

    all_dates = accumulated["time"].values.astype("datetime64[ns]")
    initialization_date = np.datetime64(forecast_date, "ns")
    later_start_date = later["time"].values[0]

    later_end_date = later["time"].values[-1]
    axis.axvspan(
        later_start_date,
        later_end_date,
        color=later_group_shading_color,
        alpha=later_group_shading_alpha,
        zorder=0,
    )
    axis.set_xlim(initialization_date, all_dates[-1])

    labelled_leads = get_labelled_lead_days()
    labelled_dates = all_dates[labelled_leads - 1]

    axis.set_xticks(all_dates, minor=True)
    axis.set_xticks(labelled_dates)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.tick_params(axis="x", which="minor", length=3.5, labelbottom=False)
    axis.tick_params(
        axis="x",
        which="major",
        length=6.0,
        labelsize=tick_label_fontsize,
        rotation=x_label_rotation,
    )
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    #lead_axis = axis.twiny()
    #lead_axis.set_xlim(axis.get_xlim())
    #lead_axis.set_xticks(all_dates, minor=True)
    #lead_axis.set_xticks(labelled_dates)
    #lead_axis.set_xticklabels([str(lead) for lead in labelled_leads])
    #lead_axis.tick_params(axis="x", which="minor", length=3.5, labeltop=False)
    #lead_axis.tick_params(
    #    axis="x", which="major", length=6.0, labelsize=tick_label_fontsize
    #)
    #lead_axis.set_xlabel("Lead day", fontsize=axis_label_fontsize)

    axis.set_title(forecast_title, loc="left", fontsize=title_fontsize)
    axis.set_xlabel("Date [day]", fontsize=axis_label_fontsize)
    axis.set_ylabel(
        f"{accumulation_days}-day accumulated precipitation [mm]",
        fontsize=axis_label_fontsize,
    )
    axis.tick_params(axis="y", labelsize=tick_label_fontsize)

    legend_handles = [
        Line2D(
            [0], [0],
            color=ensemble_color,
            alpha=ensemble_alpha,
            linewidth=ensemble_line_width,
            label="Forecast members",
        ),
        Patch(
            facecolor=later_group_shading_color,
            edgecolor="none",
            alpha=later_group_shading_alpha,
            label="Sampling period (leads 17-46)",
        ),
        Line2D(
            [0], [0],
            color=highlight_color,
            linewidth=highlight_line_width,
            marker="o",
            markersize=np.sqrt(maximum_marker_size),
            label=f"Ensemble member maximum",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=legend_fontsize,
        loc="upper left",
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    #lead_axis.spines["right"].set_visible(False)
    #axis.grid(
    #    axis="y",
    #    linestyle=":",
    #    linewidth=forecast_grid_linewidth,
    #    alpha=forecast_grid_alpha,
    #)
    axis.set_ylim(forecast_y_limits)


# =============================================================================
# Panels (c) and (d): realization counts
# =============================================================================

def read_sample_metadata():
    """Read sample metadata and count finite realizations for each i_date."""
    filename = make_sample_input_filename()

    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        required = {"sample_month", "model_type", "tp24_max"}
        missing = required - set(ds.variables)
        if missing:
            raise KeyError(f"Input file is missing variables: {sorted(missing)}")

        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("sample_month must have dimension ('i_date',).")
        if ds["model_type"].dims != ("i_date",):
            raise ValueError("model_type must have dimension ('i_date',).")

        expected_dims = {"number", "i_date"}
        if set(ds["tp24_max"].dims) != expected_dims:
            raise ValueError(
                f"tp24_max must contain dimensions {sorted(expected_dims)}, "
                f"but has {ds['tp24_max'].dims}."
            )

        sample_month = np.asarray(ds["sample_month"].load().values, dtype="int64")
        model_type = np.char.lower(
            np.asarray(ds["model_type"].load().values).astype(str)
        )
        realization_count = (
            np.isfinite(ds["tp24_max"])
            .sum(dim="number")
            .transpose("i_date")
            .load()
            .values
            .astype("int64")
        )

    unknown = sorted(set(model_type) - {"forecast", "hindcast"})
    if unknown:
        raise ValueError(f"Unsupported model_type values: {unknown}")

    for value in np.unique(sample_month):
        validate_sample_month_value(int(value), "sample_month")

    return sample_month, model_type, realization_count


def sample_month_to_period(sample_month):
    """Convert one YYYYMM integer to a pandas monthly Period."""
    year = int(sample_month) // 100
    month = int(sample_month) % 100
    return pd.Period(year=year, month=month, freq="M")


def calculate_monthly_counts(sample_month, model_type, realization_count):
    """Calculate monthly forecast, hindcast, and combined realization counts."""
    periods = pd.PeriodIndex(
        [sample_month_to_period(value) for value in sample_month]
    )
    all_months = pd.period_range(periods.min(), periods.max(), freq="M")

    if plot_start_month is not None:
        all_months = all_months[
            all_months >= sample_month_to_period(plot_start_month)
        ]
    if plot_end_month is not None:
        all_months = all_months[
            all_months <= sample_month_to_period(plot_end_month)
        ]
    if all_months.empty:
        raise ValueError("No months remain after applying the plot limits.")

    monthly_counts = {}
    for label, type_name in [("Forecast", "forecast"), ("Hindcast", "hindcast")]:
        mask = model_type == type_name
        counts = (
            pd.Series(realization_count[mask], index=periods[mask])
            .groupby(level=0)
            .sum()
        )
        monthly_counts[label] = counts.reindex(all_months, fill_value=0).astype(int)

    monthly_counts["All"] = monthly_counts["Forecast"] + monthly_counts["Hindcast"]
    return all_months.to_timestamp(how="start"), monthly_counts


def calculate_calendar_month_counts(plot_dates, monthly_counts):
    """Sum realization counts across years for January through December."""
    calendar_month_counts = {}

    for label, counts in monthly_counts.items():
        values = pd.Series(counts.values, index=pd.DatetimeIndex(plot_dates))
        grouped = values.groupby(values.index.month).sum()
        calendar_month_counts[label] = grouped.reindex(range(1, 13), fill_value=0)

    return calendar_month_counts


def format_count_axis(axis):
    """Apply common formatting to panels (c) and (d)."""
    axis.tick_params(
        axis="both",
        labelsize=tick_label_fontsize,
        direction="out",
        length=3.5,
        width=0.8,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(bottom=0)
    axis.margins(x=0.01)

    #if count_show_grid:
    #    axis.grid(
    #        axis="y",
    #        linestyle=":",
    #        linewidth=count_grid_linewidth,
    #        alpha=count_grid_alpha,
    #    )


def plot_monthly_count_panel(axis, plot_dates, monthly_counts):
    """Plot realization counts for each YYYYMM in panel (d)."""
    for label in ["Forecast","Hindcast","All"]:
        if label == 'All':
            axis.plot(
                plot_dates[0],
                np.nan,
                linewidth=count_linewidth,
                color=count_colors[label],
                label=label,
            )
        else:
            axis.plot(
                plot_dates,
                monthly_counts[label].values,
                linewidth=count_linewidth,
                color=count_colors[label],
                label=label,
            )

    axis.set_title("d) Ensemble members by year", loc="left", fontsize=title_fontsize, pad=8)
    axis.set_xlabel(
        "Time [year]",
        fontsize=axis_label_fontsize,
    )
    axis.set_ylabel("Number", fontsize=axis_label_fontsize)

    start_year = plot_dates.min().year
    end_year = plot_dates.max().year
    tick_years = np.arange(start_year, end_year + 1)
    label_years = set(tick_years[::2])

    year_ticks = [pd.Timestamp(year=year, month=1, day=1) for year in tick_years]
    year_labels = [str(year) if year in label_years else "" for year in tick_years]

    axis.set_xticks(year_ticks)
    axis.set_xticklabels(
        year_labels,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    format_count_axis(axis)


def plot_calendar_month_count_panel(axis, calendar_month_counts):
    """Plot realization counts aggregated by calendar month in panel (c)."""
    month_numbers = np.arange(1, 13)

    for label in ["All","Hindcast","Forecast"]:
        axis.plot(
            month_numbers,
            calendar_month_counts[label].values,
            marker="o",
            markersize=count_marker_size,
            linewidth=count_linewidth,
            color=count_colors[label],
            label=label,
        )

    axis.set_title("c) Ensemble members by month", loc="left", fontsize=title_fontsize, pad=8)
    axis.set_xlabel("Time [month]", fontsize=axis_label_fontsize)
    axis.set_ylabel("Number", fontsize=axis_label_fontsize)
    axis.set_xlim(0.5, 12.5)
    axis.set_xticks(month_numbers)
    axis.set_xticklabels(month_names)

    format_count_axis(axis)
    axis.legend(
        frameon=False,
        fontsize=legend_fontsize,
        loc="center left",
    )


# =============================================================================
# Figure assembly
# =============================================================================

def make_figure(
    weights,
    boundary,
    accumulated,
    plot_dates,
    monthly_counts,
    calendar_month_counts,
):
    """Create the combined 2 x 2 figure with an inset Cartopy map in panel (a)."""
    map_projection = ccrs.LambertConformal(
        central_longitude=map_central_lon,
        central_latitude=map_central_lat,
    )
    data_crs = ccrs.PlateCarree()

    figure = plt.figure(figsize=(figure_width, figure_height))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        wspace=figure_wspace,
        hspace=figure_hspace,
    )

    map_container = figure.add_subplot(grid[0, 0])
    forecast_axis = figure.add_subplot(grid[0, 1])
    calendar_month_axis = figure.add_subplot(grid[1, 0])
    monthly_axis = figure.add_subplot(grid[1, 1])

    map_container.set_axis_off()

    container_position = map_container.get_position()
    map_left = container_position.x0 + map_inset_bounds[0] * container_position.width
    map_bottom = container_position.y0 + map_inset_bounds[1] * container_position.height
    map_width = map_inset_bounds[2] * container_position.width
    map_height = map_inset_bounds[3] * container_position.height

    map_axis = figure.add_axes(
        [map_left, map_bottom, map_width, map_height],
        projection=map_projection,
    )

    mesh = plot_catchment_panel(map_axis, weights, boundary, data_crs)
    plot_forecast_panel(forecast_axis, accumulated)
    plot_monthly_count_panel(monthly_axis, plot_dates, monthly_counts)
    plot_calendar_month_count_panel(calendar_month_axis, calendar_month_counts)

    colorbar_left = (
        container_position.x0 + map_colorbar_bounds[0] * container_position.width
    )
    colorbar_bottom = (
        container_position.y0 + map_colorbar_bounds[1] * container_position.height
    )
    colorbar_width = map_colorbar_bounds[2] * container_position.width
    colorbar_height = map_colorbar_bounds[3] * container_position.height

    colorbar_axis = figure.add_axes(
        [colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]
    )
    colorbar = figure.colorbar(
        mesh,
        cax=colorbar_axis,
        orientation="vertical",
    )
    
    colorbar.set_label(colorbar_label, fontsize=axis_label_fontsize)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.tick_params(
        axis="y",
        labelsize=tick_label_fontsize,
        labelleft=True,
        labelright=False,
    )
    
    return figure


# =============================================================================
# Reporting
# =============================================================================

def print_sample_summary(sample_month, model_type, realization_count):
    """Print compact-sample totals."""
    forecast_mask = model_type == "forecast"
    hindcast_mask = model_type == "hindcast"

    print("Sample file:", make_sample_input_filename())
    print(f"Forecast samples total: {int(realization_count[forecast_mask].sum()):,}")
    print(f"Hindcast samples total: {int(realization_count[hindcast_mask].sum()):,}")
    print(f"First sample_month: {int(np.min(sample_month))}")
    print(f"Last sample_month: {int(np.max(sample_month))}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Read the map, forecast, and compact sample data and make four panels."""
    validate_user_settings()

    filename_forecast = find_forecast_file()
    print("Forecast initialization:", forecast_date)
    print("Forecast file:", filename_forecast)
    print("Catchment weights:", filename_weights)
    print("Catchment boundary:", filename_catchment)

    weights = load_catchment_weights()
    boundary = load_catchment_boundary()

    with xr.open_dataset(filename_forecast) as opened:
        forecast_ds = opened.load()

    precipitation = catchment_weighted_mean(forecast_ds, weights)
    accumulated = calculate_accumulation(precipitation)

    sample_month, model_type, realization_count = read_sample_metadata()
    plot_dates, monthly_counts = calculate_monthly_counts(
        sample_month,
        model_type,
        realization_count,
    )
    calendar_month_counts = calculate_calendar_month_counts(
        plot_dates,
        monthly_counts,
    )
    print_sample_summary(sample_month, model_type, realization_count)

    figure = make_figure(
        weights,
        boundary,
        accumulated,
        plot_dates,
        monthly_counts,
        calendar_month_counts,
    )

    if write2file:
        filename_out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            filename_out,
            dpi=figure_dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Wrote:", filename_out)

    if show_figure:
        plt.show()

    plt.close(figure)


if __name__ == "__main__":
    main()
