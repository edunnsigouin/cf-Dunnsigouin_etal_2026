"""
Read an existing catchment-averaged precipitation time series (ERA5 or SeNorge),
compute annual block maxima, fit a stationary GEV distribution, compute return levels,
and plot (publication-style):

Panel A: Full time series with event date marked.
Panel B: Zoomed time series around event (±N days).
Panel C: Return period plot (empirical dots + fitted GEV curve on log-T axis),
         with event level line and annotation of its equivalent return period.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import genextreme
from Dunnsigouin_etal_2026 import config

# input -------------------------------
dataset         = "era5"   # "era5" or "senorge"

years           = np.arange(1957, 2024, 1)
catchment       = "nevina_hønnefoss"
x_days          = 2

# Event settings (Storm Hans default)
event_date_str   = "2023-08-08"
window_days      = 15
event_sel_method = "nearest"

# Return period plotting options
T_min        = 1.01
T_max        = 1000.0
n_T          = 300
plot_T_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)

exclude_year_2023 = True

# IO
path_in_ts = config.dirs[f"{dataset}_processed"]  # where your aggregated ERA5/SeNorge files live
path_out   = config.dirs["fig"]
write2file = True
# -------------------------------------


def infer_filenames(dataset: str, years: np.ndarray, catchment: str, x_days: int):
    dataset = dataset.lower().strip()
    if dataset == "era5":
        variable = "tp24"
        grid = "0.5x0.5"
        file_in_ts = f"t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_era5_{grid}_{years[0]}-{years[-1]}.nc"
        ts_varname = f"tp_{x_days}day_catchment_acc"
    elif dataset == "senorge":
        variable = "rr"
        file_in_ts = f"t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_{years[0]}-{years[-1]}.nc"
        ts_varname = f"rr_{x_days}day_catchment_acc"
    else:
        raise ValueError("dataset must be 'era5' or 'senorge'.")
    return variable, grid, file_in_ts, ts_varname


def load_timeseries(path_in_ts: str, file_in_ts: str, ts_varname: str) -> xr.DataArray:
    ds = xr.open_dataset(path_in_ts + file_in_ts)

    if ts_varname not in ds:
        raise KeyError(f"'{ts_varname}' not found in {file_in_ts}. Available: {list(ds.data_vars)}")

    ts = ds[ts_varname]
    if "time" not in ts.dims:
        raise ValueError(f"'{ts_varname}' must have a 'time' dimension.")

    if not np.issubdtype(ts["time"].dtype, np.datetime64):
        ts = xr.decode_cf(ts.to_dataset(name="ts"))["ts"]

    ts = ts.astype(float).where(np.isfinite(ts))
    return ts


def annual_block_maxima(ts: xr.DataArray, *, exclude_year: int | None = None) -> xr.DataArray:
    ann_max = ts.groupby("time.year").max("time", skipna=True)
    ann_max.name = "annual_max"
    ann_max.attrs["description"] = "Annual block maxima of the input time series"
    ann_max.attrs["units"] = ts.attrs.get("units", "")

    if exclude_year is not None:
        ann_max = ann_max.sel(year=ann_max["year"] != exclude_year)

    return ann_max


def fit_gev(annual_max: xr.DataArray):
    x = annual_max.values.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError(f"Too few valid annual maxima to fit GEV (n={x.size}).")
    c, loc, scale = genextreme.fit(x)
    return float(c), float(loc), float(scale)


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


def _nice_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    #ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
    #ax.grid(True, which="minor", linewidth=0.25, alpha=0.25)


def plot_three_panel_pub(
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
    gev_params: tuple[float, float, float],
    T_ticks: np.ndarray,
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 300,
):
    """
    Three-panel publication-style figure with robust layout so Panel A y-label is visible.

    Panel A: Full time series with event marked.
    Panel B: Zoomed time series around event (±window_days).
    Panel C: Return period plot (empirical Weibull PP + GEV fit).
    """
    c, loc, scale = gev_params

    # Equivalent return period under annual-max model
    Fz = float(genextreme.cdf(event_value, c=c, loc=loc, scale=scale))
    Fz = min(max(Fz, 1e-12), 1 - 1e-12)
    T_event = 1.0 / (1.0 - Fz)

    units = ts.attrs.get("units", "")

    # Zoom series for Panel B
    t0 = requested_time - np.timedelta64(window_days, "D")
    t1 = requested_time + np.timedelta64(window_days, "D")
    ts_zoom = ts.sel(time=slice(t0, t1))

    fig = plt.figure(figsize=(10.5, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[2, 0])

    # Reserve vertical space for suptitle
    fig.subplots_adjust(
    left=0.1,
        right=0.95,
        bottom=0.05,
        top=0.92   # ← this creates space between suptitle and Panel A
    )

    # -------------------------
    # Panel A: full time series
    # -------------------------
    t_vals = ts["time"].values
    y_vals = ts.values.astype(float)

    t_min, t_max = t_vals.min(), t_vals.max()
    y_min = np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)
    y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0

    axA.plot(t_vals, y_vals, linewidth=0.9)
    axA.scatter([selected_time], [event_value], color="k", s=25, zorder=4)

    axA.set_ylabel(f"{ts.name} ({units})" if units else ts.name)
    axA.set_title("A) Full time series", loc="left", fontsize=12)

    # Decadal ticks (every 10 years)
    years_all = pd.to_datetime(t_vals).year
    start_year = int(np.floor(years_all.min() / 10) * 10)
    end_year = int(np.ceil(years_all.max() / 10) * 10)
    tick_years = np.arange(start_year, end_year + 1, 10)
    tick_dates = pd.to_datetime([f"{y}-01-01" for y in tick_years])

    axA.set_xticks(tick_dates)
    axA.set_xticklabels([str(y) for y in tick_years])

    axA.set_xlim(t_min, t_max)
    axA.set_ylim(y_min, y_max + y_pad)
    
    # -------------------------
    # Panel B: zoom around event
    # -------------------------
    aug_start = np.datetime64("2023-08-01")
    aug_end   = np.datetime64("2023-08-31")
    ts_aug = ts.sel(time=slice(aug_start, aug_end))

    axB.plot(ts_aug["time"].values, ts_aug.values, linewidth=1.2)
    axB.plot(ts_aug["time"].values, ts_aug.values, marker='o',color='tab:blue',linewidth=1.2)
    axB.axvline(selected_time, linestyle="--", linewidth=1.1, color="k")
    axB.scatter([selected_time], [event_value], color="k", s=35, zorder=4)

    axB.set_xlim(aug_start, aug_end)
    axB.set_ylabel(f"{ts.name} ({units})" if units else ts.name)

    # Ticks: every 3 days, rotated and right-aligned
    axB.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(axB.get_xticklabels(), rotation=30, ha="right")

    # Clean y-limits for zoom clarity
    if np.isfinite(ts_aug.values).any():
        y_aug_min = np.nanmin(ts_aug.values)
        y_aug_max = np.nanmax(ts_aug.values)
        y_pad_aug = 0.05 * (y_aug_max - y_aug_min) if y_aug_max > y_aug_min else 1.0
        axB.set_ylim(y_aug_min, y_aug_max + y_pad_aug)

    storm_date = str(selected_time)[:10]
    axB.set_title(f"B) Storm Hans maximum precipitation: {storm_date}", loc="left", fontsize=12)


    # -------------------------
    # Panel C: return periods
    # -------------------------
    axC.plot(T_curve, z_curve, linewidth=1.6, label="GEV fit")
    axC.scatter(T_emp, x_emp, s=22, alpha=0.85, label="Empirical (Weibull PP)")

    axC.set_xscale("log")
    axC.set_xlabel("Return period, T (years)")
    axC.set_ylabel(f"{ts.name} ({units})" if units else ts.name) 
    axC.set_title("C) Annual return period", loc="left", fontsize=12)

    if T_ticks is not None and len(T_ticks) > 0:
        axC.set_xticks(T_ticks)
        axC.get_xaxis().set_major_formatter(ScalarFormatter())
        axC.get_xaxis().set_minor_formatter(NullFormatter())

    axC.axhline(event_value, linestyle="--", color='k',linewidth=1.1)
    axC.axvline(T_event, linestyle="--", linewidth=1.1, color="k")
    axC.scatter([T_event], [event_value], color="k", s=45, zorder=5)

    axC.set_title(f"C) Storm Hans return period ≈ {T_event:.1f} years", loc="left", fontsize=12)

    axC.legend(frameon=False, loc="upper left")

    # Simple clean style
    for ax in (axA, axB, axC):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=4, width=0.8)

    fig.align_ylabels([axA, axB, axC])

    if title is not None:
        fig.suptitle(title, y=1.0, fontsize=15)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        
    plt.show()
    return fig, (axA, axB, axC)



if __name__ == "__main__":

    variable, grid, file_in_ts, ts_varname = infer_filenames(dataset, years, catchment, x_days)

    # Read aggregated catchment time series
    ts = load_timeseries(path_in_ts, file_in_ts, ts_varname)

    # Annual maxima (optionally exclude 2023)
    exclude_year = 2023 if exclude_year_2023 else None
    ann_max = annual_block_maxima(ts, exclude_year=exclude_year)

    # GEV fit
    c, loc, scale = fit_gev(ann_max)

    # Smooth return level curve
    T_curve = np.logspace(np.log10(T_min), np.log10(T_max), n_T)
    z_curve = gev_return_levels(T_curve, c=c, loc=loc, scale=scale)

    # Empirical dots
    x_emp, T_emp = empirical_return_periods(ann_max, plotting_position="weibull")

    # Event value from full time series
    requested_time, selected_time, event_value = get_event_value(ts, event_date_str, method=event_sel_method)

    suffix = "_excl2023" if exclude_year_2023 else ""
    event_tag = str(selected_time)[:10].replace("-", "")

    fig_out = f"{path_out}returnperiod_hans_{event_tag}_{dataset}_{grid}_{variable}_{x_days}dayacc_catchment_{catchment}_{years[0]}-{years[-1]}.pdf"

    plot_three_panel_pub(
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
        title=f"{dataset.upper()} — {catchment} — {x_days}-day accumulation",
        savepath=fig_out if write2file else None,
    )

    print(f"Event selected date: {str(selected_time)[:10]}")
    print(f"Event value: {event_value:.3f} {ts.attrs.get('units','')}")
    print(f"GEV params: c={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    print(f"Figure: {fig_out if write2file else '(not saved)'}")
