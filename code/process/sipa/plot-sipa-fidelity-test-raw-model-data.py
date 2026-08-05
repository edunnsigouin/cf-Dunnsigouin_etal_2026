"""
Raw-only fidelity analysis based on debug.ipynb.

Produces:
1. A one-row heatmap for raw S2S data with mean, std, skew, and kurtosis.
2. A 12-panel bootstrap-distribution figure for one user-selected statistic.
"""

from pathlib import Path
import copy

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

reference_dataset = "senorge"

reference_date_range = [
    "1957-01-01",
    "2022-12-31",
]

model_forecast_date_range = [
    "2020-01-02",
    "2022-12-29",
]

catchment = "regine_drammen"

analysis_x_days = 2
reference_file_x_days = 1
model_file_x_days = 1

# Choose one: "mean", "std", "skew", or "kurtosis".
distribution_statistic = "kurtosis"

n_bootstrap_samples = 10_000
bootstrap_quantiles = [0.025, 0.975]

REMOVE_HANS = False

path_in = Path(config.dirs["sipa_processed"])
path_out = Path(config.dirs["fig"])

reference_filename_override = None
model_filename_override = None

write2file = False
show_plots = True
write_counts_csv = False

filename_heatmap = path_out / "raw_fidelity_heatmap.png"
filename_distribution = path_out / f"raw_fidelity_distribution_{distribution_statistic}.png"
filename_counts_csv = path_out / "raw_fidelity_counts.csv"


# =============================================================================
# Fixed settings
# =============================================================================

STATISTICS = {
    "mean": np.mean,
    "std": np.std,
    "skew": st.skew,
    "kurtosis": st.kurtosis,
}

MONTH_LABELS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# =============================================================================
# Filename helpers
# =============================================================================

def get_file_id(catchment):
    if catchment.startswith("regine_"):
        return catchment.replace("regine_", "", 1)
    return catchment


def make_reference_filename():
    if reference_filename_override is not None:
        return Path(reference_filename_override)
    file_id = get_file_id(catchment)
    return path_in / (
        f"sipa_preprocessed_{reference_dataset}_{file_id}_"
        f"{reference_date_range[0]}_{reference_date_range[1]}.nc"
    )


def make_model_filename():
    if model_filename_override is not None:
        return Path(model_filename_override)
    file_id = get_file_id(catchment)
    return path_in / (
        f"sipa_preprocessed_s2s_{file_id}_"
        f"{model_forecast_date_range[0]}_{model_forecast_date_range[1]}.nc"
    )


# =============================================================================
# Validation
# =============================================================================

def validate_date_range(date_range, name):
    if len(date_range) != 2:
        raise ValueError(f"{name} must contain exactly two dates.")
    if np.datetime64(date_range[0]) > np.datetime64(date_range[1]):
        raise ValueError(f"The start of {name} must not be later than its end.")


def validate_user_settings():
    validate_date_range(reference_date_range, "reference_date_range")
    validate_date_range(model_forecast_date_range, "model_forecast_date_range")

    if distribution_statistic not in STATISTICS:
        raise ValueError(
            f"Unknown distribution_statistic '{distribution_statistic}'. "
            f"Valid choices are: {', '.join(STATISTICS)}."
        )

    for name, value in {
        "analysis_x_days": analysis_x_days,
        "reference_file_x_days": reference_file_x_days,
        "model_file_x_days": model_file_x_days,
        "n_bootstrap_samples": n_bootstrap_samples,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1.")

    if reference_file_x_days > analysis_x_days:
        raise ValueError("reference_file_x_days cannot exceed analysis_x_days.")
    if model_file_x_days > analysis_x_days:
        raise ValueError("model_file_x_days cannot exceed analysis_x_days.")

    if len(bootstrap_quantiles) != 2:
        raise ValueError("bootstrap_quantiles must contain two probabilities.")
    if not (0 <= bootstrap_quantiles[0] < bootstrap_quantiles[1] <= 1):
        raise ValueError("bootstrap_quantiles must be ordered values between 0 and 1.")

    for label, filename in {
        "Reference": make_reference_filename(),
        "Model": make_model_filename(),
    }.items():
        if not filename.is_file():
            raise FileNotFoundError(f"{label} file not found: {filename}")


# =============================================================================
# Loading and accumulation
# =============================================================================

def rolling_window(target_days, file_days):
    if target_days == file_days:
        return 1
    if file_days != 1:
        raise ValueError(
            "Longer accumulations can only be built safely from daily input data."
        )
    return target_days


def load_reference(filename):
    with xr.open_dataset(filename) as opened:
        ds = opened.load()

    if "tp24" not in ds or "date" not in ds.dims:
        raise ValueError("Reference file must contain tp24(date).")

    # Rolling works on array order, so ensure chronological reference dates.
    ds = ds.sortby("date")

    window = rolling_window(analysis_x_days, reference_file_x_days)
    if window > 1:
        ds = (
            ds.rolling(date=window)
            .sum()
            .shift(date=-(window - 1))
            .dropna("date")
        )

    return ds.assign_coords(
        month=ds.date.dt.month,
        year=ds.date.dt.year,
    )


def load_model(filename):
    with xr.open_dataset(filename) as opened:
        ds = opened.load()

    required_vars = {"tp24", "f_date"}
    required_dims = {"lead_day", "number", "i_date"}

    if not required_vars.issubset(ds.variables):
        raise ValueError("Model file must contain tp24 and f_date.")
    if not required_dims.issubset(ds.dims):
        raise ValueError("Model file must contain lead_day, number, and i_date.")

    window = rolling_window(analysis_x_days, model_file_x_days)
    if window > 1:
        ds = (
            ds.rolling(lead_day=window)
            .sum()
            .shift(lead_day=-(window - 1))
        )

    if ds.sizes["lead_day"] <= 15:
        raise ValueError("At least 16 lead-day positions are required.")

    # Match debug.ipynb month assignment.
    month = (
        ds.isel(lead_day=15)
        .f_date.dt.month
        .drop_vars(["lead_day", "f_date"], errors="ignore")
    )

    return ds.assign_coords(month=month)


# =============================================================================
# Monthly samples
# =============================================================================

def monthly_reference_maxima(reference):
    grouped = reference.groupby(["month", "year"]).max()
    monthly = []

    for month in range(1, 13):
        values = grouped.sel(month=month).tp24.values
        monthly.append(values[np.isfinite(values)])

    if REMOVE_HANS:
        if len(monthly[7]) == 0:
            raise ValueError("The August reference sample is empty.")
        monthly = copy.deepcopy(monthly)
        monthly[7] = monthly[7][:-1]

    return monthly


def monthly_raw_model_maxima(model):
    maxima = model.tp24.max("lead_day")
    monthly = []

    for month in range(1, 13):
        values = maxima.where(maxima.month == month, drop=True).values.ravel()
        monthly.append(values[np.isfinite(values)])

    return monthly


# =============================================================================
# Bootstrap fidelity test
# =============================================================================

def bootstrap_statistics(values, sample_size, statistic, seed):
    if len(values) == 0:
        raise ValueError("A raw-model monthly sample is empty.")
    if sample_size == 0:
        raise ValueError("A reference monthly sample is empty.")

    random_state = np.random.RandomState(seed)
    indices = random_state.randint(
        0,
        len(values),
        size=(n_bootstrap_samples, sample_size),
    )

    return statistic(values[indices], axis=1)


def calculate_raw_fidelity(reference_monthly, model_monthly):
    counts = {}
    diagnostics = {}

    for statistic_index, (name, statistic) in enumerate(STATISTICS.items(), start=1):
        passed_count = 0
        monthly_results = []

        for month in range(12):
            boot = bootstrap_statistics(
                model_monthly[month],
                len(reference_monthly[month]),
                statistic,
                seed=(month + 1) * statistic_index,
            )

            low, high = np.quantile(boot, bootstrap_quantiles)
            reference_value = float(statistic(reference_monthly[month]))
            passed = bool(low <= reference_value <= high)
            passed_count += int(passed)

            monthly_results.append({
                "month": month + 1,
                "reference_value": reference_value,
                "bootstrap_low": float(low),
                "bootstrap_high": float(high),
                "bootstrap_values": boot,
                "passed": passed,
            })

        counts[name] = passed_count
        diagnostics[name] = monthly_results

    return pd.DataFrame([counts], index=["raw"]), diagnostics


# =============================================================================
# Printed results
# =============================================================================

def print_raw_results(statistic_name, monthly_results):
    print()
    print(f"Raw monthly results for {statistic_name}")
    print("-" * 84)
    print(
        f"{'Month':<12}{'Reference':>14}{'Bootstrap low':>16}"
        f"{'Bootstrap high':>17}{'Result':>12}"
    )
    print("-" * 84)

    for result in monthly_results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{MONTH_LABELS[result['month'] - 1]:<12}"
            f"{result['reference_value']:>14.4f}"
            f"{result['bootstrap_low']:>16.4f}"
            f"{result['bootstrap_high']:>17.4f}"
            f"{status:>12}"
        )

    passed = sum(int(result["passed"]) for result in monthly_results)
    print("-" * 84)
    print(f"Passed {passed} of 12 months; failed {12 - passed} of 12 months.")


# =============================================================================
# Plotting
# =============================================================================

def make_raw_heatmap(fidelity_counts, filename=None):
    figure, axis = plt.subplots(figsize=(8, 2.4))

    axis.imshow(
        12 - fidelity_counts.values,
        cmap=colormaps["Blues"],
        vmin=0,
        vmax=6,
        aspect="auto",
    )

    for (row, column), value in np.ndenumerate(fidelity_counts.values):
        axis.text(column, row, int(value), ha="center", va="center")

    axis.set_xticks(
        range(len(fidelity_counts.columns)),
        fidelity_counts.columns,
    )
    axis.set_yticks([0], ["raw"])
    axis.set_title("Raw-data fidelity counts (months out of 12)")
    figure.tight_layout()

    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, bbox_inches="tight", dpi=400)
        print("Wrote:", filename)

    if show_plots:
        plt.show()
    plt.close(figure)


def make_raw_distribution_plot(statistic_name, monthly_results, filename=None):
    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()

    # Use common x-limits across all 12 panels for direct comparison.
    all_values = np.concatenate([
        result["bootstrap_values"][np.isfinite(result["bootstrap_values"])]
        for result in monthly_results
    ])
    reference_values = np.array([
        result["reference_value"]
        for result in monthly_results
    ])
    xmin = min(all_values.min(), reference_values.min())
    xmax = max(all_values.max(), reference_values.max())

    for month_index, (axis, result) in enumerate(zip(axes, monthly_results)):
        values = result["bootstrap_values"]
        values = values[np.isfinite(values)]

        axis.hist(values, bins=30, density=True)
        axis.axvline(result["bootstrap_low"], linestyle="--", linewidth=1.2)
        axis.axvline(result["bootstrap_high"], linestyle="--", linewidth=1.2)
        axis.axvline(result["reference_value"], linewidth=2,color='r')
        axis.set_xlim(xmin, xmax)

        status = "PASS" if result["passed"] else "FAIL"
        axis.set_title(f"{MONTH_LABELS[month_index]} — {status}")
        axis.set_xlabel(statistic_name)
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.3)

    figure.suptitle(
        f"Raw-model bootstrap distributions: {statistic_name}",
        y=1.01,
    )
    figure.tight_layout()

    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, bbox_inches="tight", dpi=400)
        print("Wrote:", filename)

    if show_plots:
        plt.show()
    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate_user_settings()

    filename_reference = make_reference_filename()
    filename_model = make_model_filename()

    print("Input files")
    print("-----------")
    print("Reference:", filename_reference)
    print("Model:    ", filename_model)

    print()
    print("Analysis settings")
    print("-----------------")
    print("Reference dataset:", reference_dataset)
    print("Catchment:", catchment)
    print("Accumulation days:", analysis_x_days)
    print("Bootstrap samples:", f"{n_bootstrap_samples:,}")
    print("Distribution statistic:", distribution_statistic)
    print("Remove Hans:", REMOVE_HANS)

    reference = load_reference(filename_reference)
    model = load_model(filename_model)

    reference_monthly = monthly_reference_maxima(reference)
    model_monthly = monthly_raw_model_maxima(model)

    fidelity_counts, diagnostics = calculate_raw_fidelity(
        reference_monthly,
        model_monthly,
    )

    print()
    print("Raw fidelity counts (months out of 12)")
    print("--------------------------------------")
    print(fidelity_counts)

    print_raw_results(
        distribution_statistic,
        diagnostics[distribution_statistic],
    )

    if write_counts_csv:
        filename_counts_csv.parent.mkdir(parents=True, exist_ok=True)
        fidelity_counts.to_csv(filename_counts_csv)
        print("Wrote:", filename_counts_csv)

    make_raw_heatmap(
        fidelity_counts,
        filename_heatmap if write2file else None,
    )

    make_raw_distribution_plot(
        distribution_statistic,
        diagnostics[distribution_statistic],
        filename_distribution if write2file else None,
    )
