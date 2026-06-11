"""
Read an existing catchment-averaged precipitation time series and plot the distribution of
X-day accumulated precipitation over all years.

Plot includes:
- Histogram of all values in the time series (all years).
- Vertical lines for: event value, mean, mean ± 1 standard deviation.
- Annotation of event z-score (how many standard deviations from the mean).
- Optional PDF output when write2file=True.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Dunnsigouin_etal_2026 import config

# input -------------------------------
variable        = "tp24"
years           = np.arange(1957, 2024, 1)
grid            = "0.5x0.5"
catchment       = "regine_002_drammenvassdraget"
x_days          = 2  # accumulation used when creating the saved time series

# File containing your already-created spatially averaged time series
path_in_ts      = config.dirs["era5_processed"]
file_in_ts      = f"t_{variable}_{x_days}dayacc_catchment_{catchment}_{grid}_{years[0]}-{years[-1]}.nc"

# Variable in the file to use
ts_varname      = f"tp_{x_days}day_catchment_acc"

# Event settings (Storm Hans default)
event_date_str   = "2023-08-08"
event_sel_method = "nearest"     # xarray .sel(time=..., method=...)

# Histogram settings
n_bins           = 40

# IO
path_out        = config.dirs["fig"]
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


def get_event_value(ts: xr.DataArray, event_date_str: str, method: str = "nearest"):
    requested_time = np.datetime64(event_date_str)
    ts_point = ts.sel(time=requested_time, method=method)
    selected_time = np.datetime64(ts_point["time"].values)
    event_value = float(ts_point.values)
    return requested_time, selected_time, event_value


def summary_stats(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("Need at least two finite values to compute mean/std.")
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))  # sample std
    return mu, sigma, x.size


def plot_distribution_with_event(
    ts: xr.DataArray,
    *,
    requested_time: np.datetime64,
    selected_time: np.datetime64,
    event_value: float,
    n_bins: int = 40,
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 300,
):
    """Histogram of all values + event/mean/std markers; optionally save to PDF."""
    units = ts.attrs.get("units", "")
    x = ts.values.astype(float)
    x = x[np.isfinite(x)]

    mu, sigma, n = summary_stats(x)
    z = np.nan if sigma <= 0 else (event_value - mu) / sigma

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8), constrained_layout=True)

    ax.hist(x, bins=n_bins, density=False)
    ax.set_xlabel(f"{ts.name} ({units})" if units else ts.name)
    ax.set_ylabel("Count")
    ax.set_title("Distribution of X-day accumulated precipitation")

    # Lines
    ax.axvline(mu, linestyle="--")
    ax.axvline(mu - sigma, linestyle=":")
    ax.axvline(mu + sigma, linestyle=":")
    ax.axvline(event_value, linestyle="-")

    # Labels/annotation
    label_req = str(requested_time)[:10]
    label_sel = str(selected_time)[:10]

    text = (
        f"Event (requested): {label_req}\n"
        f"Event (selected):  {label_sel}\n"
        f"Event value: {event_value:.1f} {units}\n"
        f"Mean: {mu:.1f} {units}\n"
        f"Std:  {sigma:.1f} {units}\n"
    )

    if np.isfinite(z):
        text += f"z = (event - mean)/std = {z:.2f}"
    else:
        text += "z = NaN (std <= 0)"

    ax.annotate(
        text,
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
    )

    # Legend (keep short)
    ax.legend(
        ["Mean", "Mean ± 1σ", "", "Event"],
        loc="upper left"
    )

    if title is None:
        title = f"{catchment} — {ts_varname}"
    fig.suptitle(title)

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig, ax

    plt.show()
    return fig, ax


if __name__ == "__main__":

    # Read the already-created catchment time series
    ts = load_timeseries(path_in_ts, file_in_ts, ts_varname)

    # Event value from the full time series
    requested_time, selected_time, event_value = get_event_value(ts, event_date_str, method=event_sel_method)

    # Output filename
    event_tag = str(selected_time)[:10].replace("-", "")
    pdf_out = (
        f"{path_out}distribution_{variable}_{x_days}dayacc_"
        f"catchment_{catchment}_{grid}_{years[0]}-{years[-1]}_{event_tag}.pdf"
    )

    # Plot
    plot_distribution_with_event(
        ts,
        requested_time=requested_time,
        selected_time=selected_time,
        event_value=event_value,
        n_bins=n_bins,
        title=f"{catchment} — {ts_varname}",
        savepath=pdf_out if write2file else None,
    )
