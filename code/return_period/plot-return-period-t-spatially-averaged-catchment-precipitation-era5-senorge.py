"""
Read an existing catchment-averaged precipitation time series (ERA5 or SeNorge),
compute annual block maxima, fit a stationary GEV distribution, compute return levels,
and plot (publication-style):

Panel A: Full time series with the selected August-2023 event date marked.
Panel B: August 2023 time series (full month) with the event date marked.
Panel C: Return period plot (empirical dots + fitted GEV curve on log-T axis),
         with event level line and annotation of its equivalent return period.

Event definition (automatic):
- Event date = the date in August 2023 with the maximum accumulated precipitation
  in the input time series (already x_days-accumulated).
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
dataset   = "senorge"   # "era5" or "senorge"
years     = np.arange(1957, 2024, 1)
catchment = "nevina_losna"
x_days    = 2

# Return period plotting options
T_min             = 1.01
T_max             = 1000.0
n_T               = 300
plot_T_ticks      = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], dtype=float)
exclude_year_2023 = True

# IO
path_in_ts = config.dirs[f"{dataset}_processed"]
path_out   = config.dirs["fig"]
write2file = True
# -------------------------------------


def infer_filenames(dataset: str, years: np.ndarray, catchment: str, x_days: int):
    dataset = dataset.lower().strip()

    if dataset == "era5":
        variable = "tp24"
        grid = "0.5x0.5"
        file_in_ts = (
            f"t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_era5_{grid}_"
            f"{years[0]}-{years[-1]}.nc"
        )
        ts_varname = f"tp_{x_days}day_catchment_acc"

    elif dataset == "senorge":
        variable = "rr"
        grid = "senorge"
        file_in_ts = (
            f"t_{variable}_{x_days}dayacc_nve_catchment_{catchment}_{dataset}_"
            f"{years[0]}-{years[-1]}.nc"
        )
        ts_varname = f"rr_{x_days}day_catchment_acc"

    else:
        raise ValueError("dataset must be 'era5' or 'senorge'.")

    return variable, grid, file_in_ts, ts_varname


def load_timeseries(path_in_ts: str, file_in_ts: str, ts_varname: str) -> xr.DataArray:
    ds = xr.open_dataset(path_in_ts + file_in_ts)
    try:
        if ts_varname not in ds:
            raise KeyError(
                f"'{ts_varname}' not found in {file_in_ts}. Available: {list(ds.data_vars)}"
            )

        ts = ds[ts_varname]
        if "time" not in ts.dims:
            raise ValueError(f"'{ts_varname}' must have a 'time' dimension.")

        if not np.issubdtype(ts["time"].dtype, np.datetime64):
            ts = xr.decode_cf(ts.to_dataset(name="ts"))["ts"]

        return ts.astype(float).where(np.isfinite(ts))
    finally:
        ds.close()


def annual_block_maxima(ts: xr.DataArray, *, exclude_year: int | None = None) -> xr.DataArray:
    ann_max = ts.groupby("time.year").max("time", skipna=True)
    ann_max.name = "annual_max"
    ann_max.attrs["description"] = "Annual block maxima of the input time series"
    ann_max.attrs["units"] = ts.attrs.get("units", "")

    if exclude_year is not None:
        ann_max = ann_max.sel(year=ann_max["year"] != exclude_year)

    return ann_max


def fit_gev(annual_max: xr.DataArray) -> tuple[float, float, float]:
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
    pp = plotting_position.lower()
    if pp == "weibull":
        p = m / (n + 1.0)
    elif pp == "gringorten":
        p = (m - 0.44) / (n + 0.12)
    else:
        raise ValueError("plotting_position must be 'weibull' or 'gringorten'.")

    T_emp = 1.0 / (1.0 - p)
    return x_sorted, T_emp


def select_event_max_august_2023(ts: xr.DataArray) -> tuple[np.datetime64, float]:
    """Return (selected_time, event_value) for the maximum value in August 2023."""
    aug_start = np.datetime64("2023-08-01")
    aug_end   = np.datetime64("2023-09-01")  # exclusive end

    ts_aug = ts.sel(time=slice(aug_start, aug_end))
    if ts_aug.size == 0 or not np.isfinite(ts_aug.values).any():
        raise ValueError("No valid data found in August 2023 for event selection.")

    sel_time = np.datetime64(ts_aug.idxmax("time").values)
    sel_val = float(ts.sel(time=sel_time).values)
    return sel_time, sel_val


def plot_three_panel_pub(
    ts: xr.DataArray,
    annual_max: xr.DataArray,
    *,
    T_curve: np.ndarray,
    z_curve: np.ndarray,
    T_emp: np.ndarray,
    x_emp: np.ndarray,
    selected_time: np.datetime64,
    event_value: float,
    gev_params: tuple[float, float, float],
    T_ticks: np.ndarray,
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 300,
):
    """
    Panel A: full time series with event marked.
    Panel B: August 2023 time series (full month) with event marked.
    Panel C: return-period plot (empirical Weibull PP + GEV fit), with event annotated.
    """
    c, loc, scale = gev_params
    units = ts.attrs.get("units", "")

    # Equivalent return period under annual-max model
    Fz = float(genextreme.cdf(event_value, c=c, loc=loc, scale=scale))
    Fz = min(max(Fz, 1e-12), 1 - 1e-12)
    T_event = 1.0 / (1.0 - Fz)

    fig = plt.figure(figsize=(10.5, 9.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[2, 0])

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.05, top=0.92)

    # -------------------------
    # Panel A: full time series
    # -------------------------
    t_vals = ts["time"].values
    y_vals = ts.values.astype(float)

    y_min = np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)
    y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0

    axA.plot(t_vals, y_vals, linewidth=0.9)
    axA.scatter([selected_time], [event_value], color="k", s=25, zorder=4)

    axA.set_ylabel(f"{ts.name} ({units})" if units else ts.name)
    axA.set_title("A) Full time series", loc="left", fontsize=12)

    # Decadal ticks
    years_all = pd.to_datetime(t_vals).year
    start_year = int(np.floor(years_all.min() / 10) * 10)
    end_year = int(np.ceil(years_all.max() / 10) * 10)
    tick_years = np.arange(start_year, end_year + 1, 10)
    tick_dates = pd.to_datetime([f"{y}-01-01" for y in tick_years])

    axA.set_xticks(tick_dates)
    axA.set_xticklabels([str(y) for y in tick_years])
    axA.set_xlim(t_vals.min(), t_vals.max())
    axA.set_ylim(y_min, y_max + y_pad)

    # -------------------------
    # Panel B: August 2023 only
    # -------------------------
    aug_start = np.datetime64("2023-08-01")
    aug_end   = np.datetime64("2023-09-01")  # exclusive end

    ts_aug = ts.sel(time=slice(aug_start, aug_end))

    axB.plot(ts_aug["time"].values, ts_aug.values, linewidth=1.2, marker="o")
    axB.axvline(selected_time, linestyle="--", linewidth=1.1, color="k")
    axB.scatter([selected_time], [event_value], color="k", s=35, zorder=4)

    axB.set_xlim(aug_start, aug_end - np.timedelta64(1, "D"))
    axB.set_ylabel(f"{ts.name} ({units})" if units else ts.name)

    axB.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(axB.get_xticklabels(), rotation=30, ha="right")

    if ts_aug.size > 0 and np.isfinite(ts_aug.values).any():
        y_b_min = np.nanmin(ts_aug.values)
        y_b_max = np.nanmax(ts_aug.values)
        y_pad_b = 0.05 * (y_b_max - y_b_min) if y_b_max > y_b_min else 1.0
        axB.set_ylim(y_b_min, y_b_max + y_pad_b)

    event_date = str(selected_time)[:10]
    axB.set_title(f"B) August 2023 (event: {event_date})", loc="left", fontsize=12)

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

    axC.axhline(event_value, linestyle="--", color="k", linewidth=1.1)
    axC.axvline(T_event, linestyle="--", linewidth=1.1, color="k")
    axC.scatter([T_event], [event_value], color="k", s=45, zorder=5)

    axC.set_title(f"C) Event return period ≈ {T_event:.1f} years", loc="left", fontsize=12)
    axC.legend(frameon=False, loc="upper left")

    # Clean style
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

    # Annual maxima (optionally exclude 2023 from the fit)
    exclude_year = 2023 if exclude_year_2023 else None
    ann_max = annual_block_maxima(ts, exclude_year=exclude_year)

    # GEV fit
    c, loc, scale = fit_gev(ann_max)

    # Smooth return level curve
    T_curve = np.logspace(np.log10(T_min), np.log10(T_max), n_T)
    z_curve = gev_return_levels(T_curve, c=c, loc=loc, scale=scale)

    # Empirical dots
    x_emp, T_emp = empirical_return_periods(ann_max, plotting_position="weibull")

    # Automatic event selection: maximum accumulated precipitation in August 2023
    selected_time, event_value = select_event_max_august_2023(ts)

    if dataset == 'era5':
        fig_out = f"{path_out}returnperiod_stormhans_{dataset}_{grid}_catchment_{catchment}_{variable}_{x_days}dayacc_{years[0]}-{years[-1]}.pdf"
    elif dataset == 'senorge':
        fig_out = f"{path_out}returnperiod_stormhans_{dataset}_catchment_{catchment}_{variable}_{x_days}dayacc_{years[0]}-{years[-1]}.pdf"
        
    plot_three_panel_pub(
        ts,
        ann_max,
        T_curve=T_curve,
        z_curve=z_curve,
        T_emp=T_emp,
        x_emp=x_emp,
        selected_time=selected_time,
        event_value=event_value,
        gev_params=(c, loc, scale),
        T_ticks=plot_T_ticks,
        title=f"{dataset.upper()} — {catchment} — {x_days}-day accumulation",
        savepath=fig_out if write2file else None,
    )

    print(f"Event selected date (max in Aug 2023): {str(selected_time)[:10]}")
    print(f"Event value: {event_value:.3f} {ts.attrs.get('units','')}")
    print(f"GEV params: c={c:.4f}, loc={loc:.4f}, scale={scale:.4f}")
    print(f"Figure: {fig_out if write2file else '(not saved)'}")
