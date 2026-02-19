"""
Map (annual-maxima) GEV return period categories of an event at each grid point over southern Norway
using X-day accumulated precipitation from daily tp24.

Overlays: outer boundaries of selected NVE catchments (GeoJSON), dissolved to remove internal borders.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import genextreme
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from Dunnsigouin_etal_2026 import config

# ------------------------------- config
variable               = "tp24"
grid                   = "0.5x0.5"
years                  = np.arange(1957, 2024, 1)

# Southern Norway bounds (deg)
lat_min, lat_max  = 57.75, 64.0
lon_min, lon_max  = 4.5, 13.0

# Event selection
event_date_str         = "2023-08-08"
event_sel_method       = "nearest"
x_days                 = 2 # accumulated up to the event date 
keep_full_windows_only = True

# Return period map settings
exclude_year_2023 = True
max_return_period = 500.0
min_return_period = 0.0

# Catchment overlays (label, GeoJSON relative to config.dirs["nve_catchment"], color)
catchments = [
    {
        "label": "Drammensvassdraget",
        "geojson": "catchement_nve_regine_drammen.geojson",
        "color": "tab:blue",
    },
    {
        "label": "Glommavassdraget",
        "geojson": "catchment_nve_regine_glomma.geojson",
        "color": "tab:blue",
    },
    {
        "label": "hønnefossvassdraget",
        "geojson": "catchment_nve_nevina_hønnefoss.geojson",
        "color": "tab:green",
    },
    {
        "label": "losnavassdraget",
        "geojson": "catchment_nve_nevina_losna.geojson",
        "color": "tab:green",
    },
    {
        "label": "bergheimvassdraget",
        "geojson": "catchment_nve_nevina_bergheim.geojson",
        "color": "tab:green",
    },
]

# If GeoJSON has no CRS, assume this
catchment_crs_if_missing = "EPSG:4326"

# IO
path_in    = config.dirs["era5_continuous_daily"] + variable + "/"
path_out   = config.dirs["fig"]
write2file = True
# -------------------------------------


def preprocess_func(ds):
    return ds.drop_vars("number", errors="ignore")


def load_tp24_daily(variable, years, grid, path_in, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    """Load daily tp24 files and subset to bounds; convert tp24 from meters to mm/day."""
    filenames = [f"{path_in}{variable}_{grid}_{year}.nc" for year in years]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_func,
        combine="by_coords",
    )

    if (lat_min is not None) and (lat_max is not None):
        if ds.latitude.values[0] > ds.latitude.values[-1]:
            ds = ds.sel(latitude=slice(lat_max, lat_min))
        else:
            ds = ds.sel(latitude=slice(lat_min, lat_max))

    if (lon_min is not None) and (lon_max is not None):
        ds = ds.sel(longitude=slice(lon_min, lon_max))

    if variable == "tp24":
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    return ds


def xday_accum_field(tp_da: xr.DataArray, x_days: int, keep_full_windows_only: bool = True) -> xr.DataArray:
    """Trailing X-day rolling sum along time."""
    if "time" not in tp_da.dims:
        raise ValueError("tp_da must have a 'time' dimension.")
    if not isinstance(x_days, int) or x_days < 1:
        raise ValueError("x_days must be an integer >= 1.")

    minp = x_days if keep_full_windows_only else 1
    acc = tp_da.rolling(time=x_days, min_periods=minp).sum()

    if keep_full_windows_only:
        acc = acc.dropna("time", how="any")

    in_units = tp_da.attrs.get("units", "")
    out_units = "mm" if in_units.strip().lower() == "mm/day" else in_units

    acc.name = f"{tp_da.name}_{x_days}dayacc"
    acc.attrs["description"] = f"{x_days}-day accumulated precipitation (rolling sum)"
    acc.attrs["units"] = out_units
    return acc


def get_event_field(acc_da: xr.DataArray, event_date_str: str, method: str = "nearest") -> xr.DataArray:
    """Select the event field at a requested date (nearest by default)."""
    requested_time = np.datetime64(event_date_str)
    point = acc_da.sel(time=requested_time, method=method)
    selected_time = np.datetime64(point["time"].values)

    out = point.squeeze(drop=True)
    out.name = "event_accum"
    out.attrs["requested_time"] = str(requested_time)[:10]
    out.attrs["selected_time"] = str(selected_time)[:10]
    out.attrs["units"] = acc_da.attrs.get("units", "")
    out.attrs["x_days"] = int(x_days)
    return out


def annual_block_maxima_field(acc_da: xr.DataArray, *, exclude_year: int | None = None) -> xr.DataArray:
    """Compute annual maxima of the accumulated field; optionally drop one year."""
    ann_max = acc_da.groupby("time.year").max("time", skipna=True)
    if exclude_year is not None:
        ann_max = ann_max.sel(year=ann_max["year"] != exclude_year)

    ann_max.name = "annual_max"
    ann_max.attrs["units"] = acc_da.attrs.get("units", "")
    ann_max.attrs["description"] = f"Annual maxima of {acc_da.attrs.get('description', 'X-day accumulated precipitation')}"
    return ann_max


def _fit_gev_1d(x: np.ndarray):
    """Fit GEV to a 1D sample; return (shape, loc, scale) or NaNs if insufficient/failed."""
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan, np.nan, np.nan
    try:
        c, loc, scale = genextreme.fit(x)
        return c, loc, scale
    except Exception:
        return np.nan, np.nan, np.nan


def fit_gev_per_gridpoint(ann_max: xr.DataArray) -> xr.Dataset:
    """Fit a GEV at each grid point along the 'year' dimension."""
    if hasattr(ann_max.data, "chunks") and ann_max.chunks is not None:
        ann_max = ann_max.chunk({"year": -1})

    c, loc, scale = xr.apply_ufunc(
        _fit_gev_1d,
        ann_max,
        input_core_dims=[["year"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return xr.Dataset({"c": c, "loc": loc, "scale": scale})


def return_period_map(event_field: xr.DataArray, gev_params: xr.Dataset, *, min_T=0.0, max_T=500.0) -> xr.DataArray:
    """Compute equivalent return period T=1/(1-F(z)) using GEV CDF at each grid point."""
    def _cdf(z, c, loc, scale):
        if not (np.isfinite(z) and np.isfinite(c) and np.isfinite(loc) and np.isfinite(scale) and scale > 0):
            return np.nan
        try:
            return float(genextreme.cdf(z, c=c, loc=loc, scale=scale))
        except Exception:
            return np.nan

    Fz = xr.apply_ufunc(
        _cdf,
        event_field, gev_params["c"], gev_params["loc"], gev_params["scale"],
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    eps = 1e-12
    Fz = Fz.clip(min=eps, max=1.0 - eps)

    T = 1.0 / (1.0 - Fz)
    T = T.clip(min=min_T, max=max_T)

    T.name = "return_period_years"
    T.attrs["description"] = "Equivalent return period (years) of event magnitude under annual-max GEV"
    T.attrs["units"] = "years"
    T.attrs["event_date_requested"] = event_field.attrs.get("requested_time", "")
    T.attrs["event_date_selected"] = event_field.attrs.get("selected_time", "")
    T.attrs["x_days"] = event_field.attrs.get("x_days", "")
    return T


def load_catchment_outer_boundaries(
    catchments: list[dict],
    crs_if_missing: str = "EPSG:4326"
) -> list[dict]:
    """Load catchments and return geometries with ONLY outer borders (no interior holes/borders)."""
    out = []

    for c in catchments:
        path_geojson = config.dirs["nve_catchment"] + c["geojson"]
        gdf = gpd.read_file(path_geojson)

        if gdf.crs is None:
            gdf = gdf.set_crs(crs_if_missing)

        gdf = gdf.to_crs("EPSG:4326")

        # Dissolve to one geometry
        union_geom = gdf.geometry.union_all()

        # Keep ONLY exterior rings (remove holes / interior boundaries)
        if isinstance(union_geom, Polygon):
            outer_only = Polygon(union_geom.exterior)
        elif isinstance(union_geom, MultiPolygon):
            outer_only = MultiPolygon([Polygon(p.exterior) for p in union_geom.geoms])
        else:
            # Fallback: keep as-is if geometry type is unexpected
            outer_only = union_geom

        out.append(
            {
                "label": c.get("label", c["geojson"]),
                "color": c.get("color", "blue"),
                "geometry": outer_only,
            }
        )

    return out


def plot_return_period_map(
    T: xr.DataArray,
    *,
    catchment_boundaries: list[dict] | None = None,
    savepath: str | None = None,
    dpi: int = 300,
):
    """
    Plot categorical return period map with bins: 1–5, 5–10, 10–50, 50–100, >100 years.
    If savepath is provided, save as PDF and close; otherwise show interactively.
    """
    proj_data = ccrs.PlateCarree()
    proj_map = ccrs.LambertConformal(central_longitude=10.0, central_latitude=62.0)

    edges = np.array([1.0, 5.0, 10.0, 50.0, 100.0, np.inf], dtype=float)
    labels = ["1–5", "5–10", "10–50", "50–100", "> 100"]

    def _categorize(x):
        out = np.digitize(x, edges, right=False) - 1
        out = out.astype(np.int16)
        out[~np.isfinite(x)] = -1
        return out

    cat = xr.apply_ufunc(
        _categorize,
        T,
        input_core_dims=[[]],
        output_core_dims=[[]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.int16],
    )
    cat_plot = cat.where(cat >= 0)

    base_cmap = plt.get_cmap("YlOrRd")
    colors = base_cmap(np.linspace(0.25, 0.95, len(labels)))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])  # 1–5 years as white
    cmap = mcolors.ListedColormap(colors)

    fig = plt.figure(figsize=(9, 8))
    ax = plt.axes(projection=proj_map)

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5)

    mesh = ax.pcolormesh(
        T["longitude"].values,
        T["latitude"].values,
        cat_plot.values,
        transform=proj_data,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
    )

    if catchment_boundaries is not None:
        for item in catchment_boundaries:
            ax.add_geometries(
                [item["geometry"]],
                crs=proj_data,
                facecolor="none",
                edgecolor=item.get("color", "blue"),
                linewidth=1.8,
                zorder=5,
            )

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Return period category (years)")

    ax.set_extent(
        [float(T.longitude.min()), float(T.longitude.max()),
         float(T.latitude.min()), float(T.latitude.max())],
        crs=proj_data,
    )

    event_date = T.attrs.get("event_date_selected", "")
    xdays = T.attrs.get("x_days", "")
    title = (
        f"Storm Hans return period categories\n"
        f"{event_date} ({xdays}-day accumulation)"
    )
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        
    plt.show()
    
    return fig, ax


if __name__ == "__main__":

    # 1) Load daily precipitation and subset to bounds
    ds = load_tp24_daily(
        variable, years, grid, path_in,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    tp = ds[variable]  # (time, lat, lon) in mm/day

    # 2) X-day accumulation (trailing rolling sum)
    tp_acc = xday_accum_field(tp, x_days=x_days, keep_full_windows_only=keep_full_windows_only)

    # 3) Event accumulation field
    event_acc = get_event_field(tp_acc, event_date_str, method=event_sel_method)

    # 4) Annual maxima (optionally exclude 2023)
    exclude_year = 2023 if exclude_year_2023 else None
    ann_max = annual_block_maxima_field(tp_acc, exclude_year=exclude_year)

    # 5) Fit GEV per grid point
    params = fit_gev_per_gridpoint(ann_max)

    # 6) Return period map for the event magnitude
    T_map = return_period_map(
        event_acc,
        params,
        min_T=min_return_period,
        max_T=max_return_period,
    )

    # 7) Load catchment boundaries and plot
    catchment_boundaries = load_catchment_outer_boundaries(
        catchments,
        crs_if_missing=catchment_crs_if_missing,
    )

    # 8) Save figure as PDF (or show interactively)
    suffix = "_excl2023" if exclude_year_2023 else ""
    event_tag = event_acc.attrs.get("selected_time", "").replace("-", "")
    pdf_out = (
        f"{path_out}xy_returnperiod_annual_era5_{grid}_{variable}_{x_days}dayacc_"
        f"{years[0]}-{years[-1]}_{event_tag}{suffix}.pdf"
    )

    plot_return_period_map(
        T_map,
        catchment_boundaries=catchment_boundaries,
        savepath=pdf_out if write2file else None,
    )
