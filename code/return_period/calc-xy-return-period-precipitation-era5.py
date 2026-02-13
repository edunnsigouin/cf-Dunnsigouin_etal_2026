"""
Map the (annual-maxima) GEV return period of an event for each grid point over southern Norway,
using X-day accumulated precipitation derived from daily accumulated tp24.

Adds:
- Overlay boundaries for N catchments from GeoJSONs (each with its own color).
- Plots ONLY the OUTER border of each catchment (dissolve -> outer boundary).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import genextreme
import matplotlib.colors as mcolors
import geopandas as gpd

from Dunnsigouin_etal_2026 import config

# input -------------------------------
variable         = "tp24"
grid             = "0.5x0.5"
years            = np.arange(1957, 2024, 1)

# Accumulation length
x_days                 = 2
keep_full_windows_only = True

# Southern Norway bounds (degrees)
lat_min, lat_max = 57.75, 64.0
lon_min, lon_max = 4.5, 13.0

# Event to map
event_date_str    = "2023-08-08"
event_sel_method  = "nearest"

# Return period map settings
exclude_year_2023 = True
max_return_period = 500.0
min_return_period = 0.0

# Catchment boundary overlays (N catchments)
# Provide label + geojson filename (relative to config.dirs["nve_catchment"]) + color
catchments = [
    {
        "label": "Drammensvassdraget",
        "geojson": "nve_regine_enhet_012_drammensvassdraget_entire_catchment.geojson",
        "color": "tab:blue",
    },
    {"label": "Glommavassdraget",
     "geojson": "nve_regine_enhet_002_glommavassdraget_entire_catchment.geojson",
     "color": "tab:blue",
    },
    {"label": "hønnefossvassdraget",
     "geojson": "nve_nevina_hønnefossvassdraget.geojson",
     "color": "tab:green",
    },
     {"label": "losnavassdraget",
      "geojson": "nve_nevina_losnavassdraget.geojson",
     "color": "tab:green",
    },
    {"label": "bergheimvassdraget",
     "geojson": "nve_nevina_bergheimvassdraget.geojson",
     "color": "tab:green",
    }
]

# If GeoJSON has missing CRS metadata, assume this (change if needed, e.g. EPSG:25833)
catchment_crs_if_missing = "EPSG:4326"

# IO
path_in      = config.dirs["era5_continuous_daily"] + variable + "/"
path_out     = config.dirs["era5_processed"]
write2file   = False
# -------------------------------------


def preprocess_func(ds):
    return ds.drop_vars("number", errors="ignore")


def load_tp24_daily(variable, years, grid, path_in,
                    lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    filenames = [f"{path_in}{variable}_{grid}_{year}.nc" for year in years]

    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess_func,
        combine="by_coords"
    )

    # Geographic subset
    if (lat_min is not None) and (lat_max is not None):
        if ds.latitude.values[0] > ds.latitude.values[-1]:
            ds = ds.sel(latitude=slice(lat_max, lat_min))
        else:
            ds = ds.sel(latitude=slice(lat_min, lat_max))

    if (lon_min is not None) and (lon_max is not None):
        ds = ds.sel(longitude=slice(lon_min, lon_max))

    # Convert tp24 meters -> mm/day
    if variable == "tp24":
        ds[variable] = ds[variable] * 1000.0
        ds[variable].attrs["units"] = "mm/day"

    return ds


def xday_accum_field(tp_da: xr.DataArray, x_days: int, keep_full_windows_only: bool = True) -> xr.DataArray:
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
    ann_max = acc_da.groupby("time.year").max("time", skipna=True)
    if exclude_year is not None:
        ann_max = ann_max.sel(year=ann_max["year"] != exclude_year)

    ann_max.name = "annual_max"
    ann_max.attrs["units"] = acc_da.attrs.get("units", "")
    ann_max.attrs["description"] = f"Annual maxima of {acc_da.attrs.get('description', 'X-day accumulated precipitation')}"
    return ann_max


def _fit_gev_1d(x: np.ndarray):
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
    # Ensure core dimension 'year' is a single chunk for apply_ufunc with dask parallelized
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


def return_period_map(event_field: xr.DataArray, gev_params: xr.Dataset,
                      *, min_T=0.0, max_T=500.0) -> xr.DataArray:
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


def load_catchment_outer_boundaries(catchments: list[dict], crs_if_missing: str = "EPSG:4326") -> list[dict]:
    """
    Load multiple catchments and return a list of dicts with:
      {'label': str, 'color': str, 'geometry': shapely geometry (outer boundary)}
    Outer boundary is computed by dissolving all polygons to one geometry.
    """
    out = []
    for c in catchments:
        path_geojson = config.dirs["nve_catchment"] + c["geojson"]
        gdf = gpd.read_file(path_geojson)

        if gdf.crs is None:
            gdf = gdf.set_crs(crs_if_missing)

        gdf = gdf.to_crs("EPSG:4326")

        # Dissolve to single geometry (removes internal borders)
        union_geom = gdf.geometry.union_all()
        
        out.append(
            {
                "label": c.get("label", c["geojson"]),
                "color": c.get("color", "blue"),
                "geometry": union_geom,  # plot boundary of this
            }
        )
    return out


def plot_return_period_map(
    T: xr.DataArray,
    *,
    catchment_boundaries: list[dict] | None = None,
    title=None
):
    """
    Categorical return period map with bins:
      <1, 1-5, 5-10, 10-50, 50-100, >100 years

    - 1–5 year category is white.
    - Can overlay N catchment OUTER boundaries (dissolved) with specified colors.
    """
    proj_data = ccrs.PlateCarree()
    proj_map  = ccrs.LambertConformal(central_longitude=10.0, central_latitude=62.0)

    edges  = np.array([1.0, 5.0, 10.0, 50.0, 100.0, np.inf], dtype=float)
    labels = ["1–5", "5–10", "10–50", "50–100", "> 100"]

    def _categorize(x):
        out = np.digitize(x, edges, right=False) - 1  # 0..5
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

    # Sequential discrete colormap + make 1–5 bin white (index 1)
    base_cmap = plt.get_cmap("YlOrRd")
    colors = base_cmap(np.linspace(0.25, 0.95, len(labels)))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])  # 1–5 years
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
        vmax=len(labels) - 0.5
    )

    # Overlay N catchment outer boundaries
    if catchment_boundaries is not None:
        for item in catchment_boundaries:
            geom = item["geometry"]
            ax.add_geometries(
                [geom],
                crs=proj_data,
                facecolor="none",
                edgecolor=item.get("color", "blue"),
                linewidth=1.8,
                zorder=5,
            )
            # Optional label near centroid (comment out if you don't want text)
            #try:
            #    cx, cy = geom.centroid.x, geom.centroid.y
            #    ax.text(cx, cy, item.get("label", ""), transform=proj_data, fontsize=9)
            #except Exception:
            #    pass

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.03, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Return period category (years)")

    ax.set_extent(
        [float(T.longitude.min()), float(T.longitude.max()),
         float(T.latitude.min()), float(T.latitude.max())],
        crs=proj_data
    )

    if title is None:
        title = (
            f"Return period categories (annual-max GEV)\n"
            f"Event: {T.attrs.get('event_date_selected','')}"
        )
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 1) Load daily precipitation over southern Norway (no Norway domain subselection)
    ds = load_tp24_daily(
        variable, years, grid, path_in,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
    )
    tp = ds[variable]  # (time, lat, lon) in mm/day

    # 2) Compute X-day accumulated field (rolling sum)
    tp_acc = xday_accum_field(tp, x_days=x_days, keep_full_windows_only=keep_full_windows_only)

    # 3) Extract event accumulated field
    event_acc = get_event_field(tp_acc, event_date_str, method=event_sel_method)

    # 4) Annual maxima of X-day accumulated field (optionally exclude 2023)
    exclude_year = 2023 if exclude_year_2023 else None
    ann_max = annual_block_maxima_field(tp_acc, exclude_year=exclude_year)

    # 5) Fit GEV per grid point
    params = fit_gev_per_gridpoint(ann_max)

    # 6) Return period map for the event magnitude
    T_map = return_period_map(
        event_acc,
        params,
        min_T=min_return_period,
        max_T=max_return_period
    )

    # 7) Load N catchment outer boundaries and plot
    catchment_boundaries = load_catchment_outer_boundaries(
        catchments,
        crs_if_missing=catchment_crs_if_missing
    )
    plot_return_period_map(
        T_map,
        catchment_boundaries=catchment_boundaries,
        title="Storm Hans return period categories"
    )

    # 8) Optional save
    if write2file:
        suffix = "_excl2023" if exclude_year_2023 else ""
        event_tag = event_acc.attrs.get("selected_time", "").replace("-", "")
        file_out = (
            f"{path_out}map_returnperiod_gev_annualmax_{variable}_{x_days}dayacc_{grid}_"
            f"{years[0]}-{years[-1]}_{event_tag}{suffix}.nc"
        )

        xr.Dataset(
            {
                "event_accum": event_acc,
                "return_period_years": T_map,
                "gev_shape_c": params["c"],
                "gev_loc": params["loc"],
                "gev_scale": params["scale"],
            },
            attrs={
                "method": "Block maxima (annual) + GEV per grid cell (SciPy genextreme MLE)",
                "variable": variable,
                "grid": grid,
                "x_days": int(x_days),
                "keep_full_windows_only": bool(keep_full_windows_only),
                "event_date_requested": event_acc.attrs.get("requested_time", ""),
                "event_date_selected": event_acc.attrs.get("selected_time", ""),
                "exclude_year_2023": bool(exclude_year_2023),
                "lat_min": float(lat_min), "lat_max": float(lat_max),
                "lon_min": float(lon_min), "lon_max": float(lon_max),
                "return_period_cap_max": float(max_return_period),
                "return_period_floor_min": float(min_return_period),
                "catchments": str(catchments),
                "catchment_crs_if_missing": catchment_crs_if_missing,
            },
        ).to_netcdf(file_out)
