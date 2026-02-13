"""
Read an existing catchment-averaged precipitation time series, compute annual block maxima,
fit a stationary GEV distribution, compute return levels, and plot:

Panel 1: Full time series with event date marked + inset showing +/- N days around event.
Panel 2: Return period plot (empirical dots + fitted curve over a full spectrum of T),
         with a horizontal line at the event magnitude and an annotation of its
         "equivalent return period" under the annual-max model.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import genextreme  # GEV
from Dunnsigouin_etal_2026 import config

# input -------------------------------
variable        = "tp24"
years           = np.arange(1957, 2024, 1)
grid            = "0.5x0.5"
catchment       = "regine_012_drammensvassdraget"
x_days          = 2  # accumulation used when creating the saved time series

# File containing your already-created spatially averaged time series
path_in_ts      = config.dirs["era5_processed"]
file_in_ts      = f"t_{variable}_{x_days}dayacc_catchment_{catchment}_{grid}_{years[0]}-{years[-1]}.nc"

# Variable in the file to use
ts_varname      = f"tp_{x_days}day_catchment_acc"

# Return period plotting options
T_min           = 1.01   # years (must be >1)
T_max           = 500.0  # years
n_T             = 300    # number of points in fitted curve
plot_T_ticks    = np.array([2, 5, 10, 20, 50, 100, 200, 500], dtype=float)

# Event settings (Storm Hans default)
event_date_str   = "2023-08-08"
window_days      = 15            # +/- days shown in inset on time series
event_sel_method = "nearest"     # xarray .sel(time=..., method=...)

# Option to exclude year 2023 from GEV fit / return period calculation
exclude_year_2023 = True

path_out        = config.dirs["era5_processed"]
write2file      = False
# -------------------------------------


def load_timeseries(path_in_ts: str, file_in_ts: str, ts_varname: str) -> xr.DataArray:
    ds = xr.open_dataset(path_in_ts + file_in_ts)

    if ts_varname not in ds:
        raise KeyError(f"'{ts_varname}' not found in {file_in_ts}. Available: {list(ds.data_vars)}")

    ts = ds[ts_varname]
    if "time" not in ts.dims:
        raise ValueError(f"'{ts_varname}' must have a 'time' dimension.")

    # Ensure datetime time axis if possible
    if not np.issubdtype(ts["time"].dtype, np.datetime64):
        ts = xr.decode_cf(ts.to_dataset(name="ts"))["ts"]

    return ts


def annual_block_maxima(ts: xr.DataArray, *, exclude_year: int | None = None) -> xr.DataArray:
    ann_max = ts.groupby("time.year").max("time", skipna=True)
    ann_max.name = "annual_max"
    ann_max.attrs["description"] = "Annual block maxima of the input time series"
    ann_max.attrs["units"] = ts.attrs.get("units", "")

    if exclude_year is not None:
        if "year" not in ann_max.dims:
            raise ValueError("Annual maxima should have 'year' dimension.")
        ann_max = ann_max.sel(year=ann_max["year"] != exclude_year)

    return ann_max


def fit_gev(annual_max: xr.DataArray):
    x = annual_max.values.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError(f"Too few valid annual maxima to fit GEV (n={x.size}).")
    c, loc, scale = genextreme.fit(x)
    return c, loc, scale


def gev_return_levels(T: np.ndarray, c: float, loc: float, scale: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1):
        raise ValueError("All return periods must be > 1 year.")
    p = 1.0 - 1.0 / T
    return genextreme.ppf(p, c=c, loc=loc, scale=scale)


def empirical_return_periods(annual_max: xr.DataArray, plotting_position: str = "weibull"):
    x = annual_max.values.astype(float)
    x = x[np.isfinite(x)]
    x_sorted = np.sort(x)
    n = x_sorted.size

    m = np.arange(1, n + 1)
    if plotting_position.lower() == "weibull":
        p = m / (n + 1.0)
    elif plotting_position.lower() == "gringorten":
        p = (m - 0.44) / (n + 0.12)
    else:
        raise ValueError("plotting_position must be 'weibull' or 'gringorten'.")

    T_emp = 1.0 / (1.0 - p)
    return x_sorted, T_emp


def get_event_value(ts: xr.DataArray, event_date_str: str, method: str = "nearest"):
    requested_time = np.datetime64(event_date_str)
    ts_point = ts.sel(time=requested_time, method=method)
    selected_time = np.datetime64(ts_point["time"].values)
    event_value = float(ts_point.values)
    return requested_time, selected_time, event_value


def plot_timeseries_and_returnperiods(
    ts: xr.DataArray,
    annual_max: xr.DataArray,
    *,
    T_curve: np.ndarray,
    z_curve: np.ndarray,
    T_emp: np.ndarray,
    x_emp: np.ndarray,
    requested_time: np.datetime64,
    selected_time: np.datetime64,
    event_value: float,
    window_days: int,
    gev_params: tuple,
    T_ticks: np.ndarray,
    title: str = None,
):
    c, loc, scale = gev_params

    # Equivalent return period under annual-max model
    Fz = float(genextreme.cdf(event_value, c=c, loc=loc, scale=scale))
    Fz = min(max(Fz, 1e-12), 1 - 1e-12)
    T_event = 1.0 / (1.0 - Fz)

    units = ts.attrs.get("units", "")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    # -------------------------
    # Panel 1: time series + inset zoom
    # -------------------------
    ax1.plot(ts["time"].values, ts.values)
    ax1.axvline(requested_time, linestyle="--")
    ax1.axvline(selected_time, linestyle=":")
    ax1.scatter([selected_time], [event_value], zorder=3)

    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"{ts.name} ({units})" if units else ts.name)
    ax1.set_title("Time series (event marked)")

    label_req = str(requested_time)[:10]
    label_sel = str(selected_time)[:10]
    ax1.annotate(
        f"Requested: {label_req}\nSelected:  {label_sel}\n{event_value:.1f} {units}",
        xy=(selected_time, event_value),
        xytext=(10, 10),
        textcoords="offset points",
    )

    # Inset: place top-left OUTSIDE the axes so it doesn't cover the line
    # bbox_to_anchor is in axes fraction; negative y moves it above the axes area.
    inset = ax1.inset_axes(
        [0.0, 1.02, 0.45, 0.42],  # x0, y0, w, h in axes fraction
        transform=ax1.transAxes
    )

    t0 = requested_time - np.timedelta64(window_days, "D")
    t1 = requested_time + np.timedelta64(window_days, "D")
    ts_zoom = ts.sel(time=slice(t0, t1))

    inset.plot(ts_zoom["time"].values, ts_zoom.values)
    inset.axvline(requested_time, linestyle="--")
    inset.axvline(selected_time, linestyle=":")
    inset.scatter([selected_time], [event_value], zorder=3)
    inset.set_title(f"Zoom: ±{window_days} days", fontsize=9)
    inset.tick_params(labelsize=8)

    # Make inset visually distinct (optional)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)

    # -------------------------
    # Panel 2: return periods (empirical dots + fitted curve)
    # -------------------------
    ax2.plot(T_curve, z_curve)
    ax2.scatter(T_emp, x_emp, s=18)

    ax2.set_xscale("log")
    ax2.set_xlabel("Return period, T (years)")
    ax2.set_ylabel(f"Return level ({units})" if units else "Return level")
    ax2.set_title("Return period plot (GEV fit + empirical)")

    if T_ticks is not None and len(T_ticks) > 0:
        ax2.set_xticks(T_ticks)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.get_xaxis().set_minor_formatter(plt.NullFormatter())

    # Event magnitude as a horizontal line
    ax2.axhline(event_value, linestyle="--")
    ax2.annotate(
        f"Event level: {event_value:.1f} {units}\nEquivalent T ≈ {T_event:.1f} years",
        xy=(T_curve[0], event_value),
        xytext=(10, 10),
        textcoords="offset points",
    )

    ax2.legend(["GEV fit", "Empirical (Weibull PP)"], loc="best")

    if title is not None:
        fig.suptitle(title)

    plt.show()


if __name__ == "__main__":

    # Read the already-created catchment time series
    ts = load_timeseries(path_in_ts, file_in_ts, ts_varname)

    # Annual maxima (optionally exclude 2023)
    exclude_year = 2023 if exclude_year_2023 else None
    ann_max = annual_block_maxima(ts, exclude_year=exclude_year)

    # GEV fit
    c, loc, scale = fit_gev(ann_max)

    # Full spectrum of return periods for smooth curve (log-spaced)
    T_curve = np.logspace(np.log10(T_min), np.log10(T_max), n_T)
    z_curve = gev_return_levels(T_curve, c=c, loc=loc, scale=scale)

    # Empirical return periods (dots)
    x_emp, T_emp = empirical_return_periods(ann_max, plotting_position="weibull")

    # Event value (Storm Hans by default) taken from the full time series
    requested_time, selected_time, event_value = get_event_value(ts, event_date_str, method=event_sel_method)

    # Plot
    plot_timeseries_and_returnperiods(
        ts,
        ann_max,
        T_curve=T_curve,
        z_curve=z_curve,
        T_emp=T_emp,
        x_emp=x_emp,
        requested_time=requested_time,
        selected_time=selected_time,
        event_value=event_value,
        window_days=window_days,
        gev_params=(c, loc, scale),
        T_ticks=plot_T_ticks,
        title=f"{catchment} — {ts_varname}" + (" (excl. 2023)" if exclude_year_2023 else ""),
    )

    # Optional write out fitted results
    ds_out = xr.Dataset(
        {
            "annual_max": ann_max,
            "return_level_curve": xr.DataArray(
                z_curve,
                dims=("return_period",),
                coords={"return_period": T_curve},
                attrs={"units": ts.attrs.get("units", ""), "description": "GEV return level curve z_T"},
            ),
        },
        attrs={
            "method": "Block maxima (annual) + GEV (SciPy genextreme MLE)",
            "gev_shape_c": float(c),
            "gev_loc": float(loc),
            "gev_scale": float(scale),
            "plotting_position": "weibull",
            "exclude_year": int(exclude_year) if exclude_year is not None else -1,
            "event_date_requested": str(requested_time)[:10],
            "event_date_selected": str(selected_time)[:10],
            "event_value": float(event_value),
            "source_file": file_in_ts,
            "ts_varname": ts_varname,
            "T_min": float(T_min),
            "T_max": float(T_max),
            "n_T": int(n_T),
        },
    )

    if write2file:
        suffix = "_excl2023" if exclude_year_2023 else ""
        filename_out = (
            f"{path_out}returnlevels_gev_annualmax_{variable}_{x_days}dayacc_"
            f"catchment_{catchment}_{grid}_{years[0]}-{years[-1]}{suffix}.nc"
        )
        ds_out.to_netcdf(filename_out)
