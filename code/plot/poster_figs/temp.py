"""
Create a two-panel schematic of the Drammen catchment and S2S sampling strategy.

Panel (a) shows the Drammen catchment weights on the ERA5 0.5-degree grid and
the catchment boundary. Panel (b) shows one ECMWF S2S forecast after catchment
averaging and temporal accumulation. All ensemble members are grey, one example
member is highlighted in blue, its later-period maximum is marked, and shading
marks the later lead-day sampling period.

The original four-panel figure used a 2 x 2 layout. This version retains only
the original top row. The figure height is reduced from 10 inches to
10 / (2 + 0.35) inches so panels (a) and (b) keep their original physical
height. Figure width, horizontal spacing, map inset geometry, and colorbar
geometry are unchanged.
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

# Positions inside the panel-(a) container: [left, bottom, width, height].
map_inset_bounds = [0.0, 0.0, 1.0, 1.0]
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


# =============================================================================
# User settings: overall figure
# =============================================================================

figure_width = 12
original_figure_height = 10
original_figure_hspace = 0.35
figure_height = original_figure_height / (2 + original_figure_hspace)
figure_dpi = 300
figure_wspace = 0.05

title_fontsize = 12
axis_label_fontsize = 11
tick_label_fontsize = 10
legend_fontsize = 9

filename_out = Path(config.dirs["fig"]) / "poster_figs/" / f"fig-02-{forecast_date}.png"


# =============================================================================
# Paths
# =============================================================================

path_nve = Path(config.dirs["nve"])
path_in_forecast = Path(config.dirs["s2s_forecast_daily"]) / variable
filename_weights = path_nve / f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
filename_catchment = path_nve / f"catchment_nve_{catchment}.geojson"


# =============================================================================
# Validation and input helpers
# =============================================================================

def find_forecast_file():
    """Return the single raw forecast file matching forecast_date."""
    matches = sorted(
        path
        for path in path_in_forecast.glob(f"{input_filename_prefix}*.nc")
        if path.stem.endswith(f"_{forecast_date}")
    )

    if not matches:
        raise FileNotFoundError(
            f"No forecast file ending in _{forecast_date}.nc was found in {path_in_forecast}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple forecast files for {forecast_date}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def validate_user_settings():
    """Validate settings and required input files."""
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

    for filename in [filename_weights, filename_catchment]:
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
    axis.set_title(map_title, loc="left", x=map_title_x, fontsize=title_fontsize)
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
    spatial_mean = precipitation.weighted(combined_weight).mean(["latitude", "longitude"])
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
            f"This schematic expects 46 raw forecast time steps, but found {number_of_times}."
        )

    accumulated = precipitation.rolling(
        time=accumulation_days, min_periods=accumulation_days
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
    """Return lead days labelled on the x-axis."""
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

    axis.set_title(forecast_title, loc="left", fontsize=title_fontsize)
    axis.set_xlabel("Date [day]", fontsize=axis_label_fontsize)
    axis.set_ylabel(
        f"{accumulation_days}-day accumulated precipitation [mm]",
        fontsize=axis_label_fontsize,
    )
    axis.tick_params(axis="y", labelsize=tick_label_fontsize)

    legend_handles = [
        Line2D(
            [0],
            [0],
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
            [0],
            [0],
            color=highlight_color,
            linewidth=highlight_line_width,
            marker="o",
            markersize=np.sqrt(maximum_marker_size),
            label="Ensemble member maximum",
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
    axis.set_ylim(forecast_y_limits)


# =============================================================================
# Figure assembly
# =============================================================================

def make_figure(weights, boundary, accumulated):
    """Create panels (a) and (b) at their original physical dimensions."""
    map_projection = ccrs.LambertConformal(
        central_longitude=map_central_lon,
        central_latitude=map_central_lat,
    )
    data_crs = ccrs.PlateCarree()

    figure = plt.figure(figsize=(figure_width, figure_height))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=figure_wspace,
    )

    map_container = figure.add_subplot(grid[0, 0])
    forecast_axis = figure.add_subplot(grid[0, 1])
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
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="vertical")
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
# Main
# =============================================================================

def main():
    """Read the map and forecast data and make panels (a) and (b)."""
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
    figure = make_figure(weights, boundary, accumulated)

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
