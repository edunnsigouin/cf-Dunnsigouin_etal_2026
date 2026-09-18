#!/usr/bin/env python3
"""Compare monthly fidelity percentiles for raw and bias-corrected model data.

The upper panel uses raw data; the lower optionally corrects only failed months.
Columns are calendar months; rows are mean, std, kurtosis, and skewness.
Numbers are reference-statistic percentiles in model bootstrap distributions.
Blue cells pass the central bootstrap interval test; orange cells fail.

Bootstrap samples are drawn with replacement and match the reference sample
size. Ties receive half weight in the percentile calculation. Colours use the
unrounded interval test, not the displayed percentile. Statistic definitions,
monthly random seeds, and reference selection follow the original script.

Bias-corrected filenames follow script 1. If BIAS_CORRECT_ONLY_FAILED_MONTHS is
True, a month uses corrected data for all four tests in panel (b) only if at
least one raw fidelity test fails. Otherwise its raw results are reused exactly.
If False, panel (b) uses corrected data for every month.
Requires numpy, scipy, pandas, xarray, matplotlib, and the project config.
"""

from calendar import month_abbr
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kurtosis, skew

from Dunnsigouin_etal_2026 import config


# User settings
x_days = 2
catchment = "regine_drammen"
forecast_date_range = ("2020-01-02", "2023-12-28")
reference_years = ("1957", "2025")
REFERENCE_FILE_YEARS = ("1957", "2025")
REFERENCE_DATASET = "senorge"  # "era5" or "senorge"
EXCLUDE_STORM_HANS_FROM_REFERENCE = True
BIAS_CORRECTION_METHOD = "mm_1step"  # "mm_1step", "mm_2step", "q", "doy", "ld", "q_doy"

# True: correct months failing any raw fidelity test; False: correct every month.
BIAS_CORRECT_ONLY_FAILED_MONTHS = False

number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0
random_seed = 42

raw_model_filename_override = None
bias_corrected_model_filename_override = None
reference_filename_override = None
output_filename_override = None  # Optional .pdf, .svg, or .png path
write2file = True
show_figure = True

figure_size = (10.0, 6.4)
figure_dpi = 400
annotation_decimals = 1
PASS_COLOR = "#0072B2"  # Okabe-Ito blue
FAIL_COLOR = "#E69F00"  # Okabe-Ito orange

# Fixed settings
MODEL_MONTH_COORDINATE = "sample_month"
STATISTICS = ("mean", "std", "kurtosis", "skewness")
TEST_LABELS = ("Mean", "Standard deviation", "Kurtosis", "Skewness")
MONTHS = np.arange(1, 13)


def build_model_filename(method: str) -> Path:
    """Use script 1's compact raw and bias-corrected filename convention."""
    override = (
        raw_model_filename_override if method == "raw"
        else bias_corrected_model_filename_override
    )
    if override is not None:
        return Path(override)
    stem = (
        f"monthly_max_samples_tp24_{x_days}dayacc_{catchment.removeprefix('regine_')}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}"
    )
    suffix = "raw" if method == "raw" else (
        f"bc_{method}_{REFERENCE_DATASET}_{reference_years[0]}-{reference_years[1]}"
    )
    return Path(config.dirs["s2s_processed"]) / f"{stem}_{suffix}.nc"


def get_reference_file() -> tuple[Path, str, str]:
    """Resolve the compact reference filename, variable, and display label."""
    variable, label = ("tp24", "ERA5") if REFERENCE_DATASET == "era5" else ("rr", "SeNorge")
    filename = reference_filename_override
    if filename is None:
        filename = Path(config.dirs[f"{REFERENCE_DATASET}_processed"]) / (
            f"monthly_max_samples_{variable}_{x_days}dayacc_{catchment}_"
            f"{REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}.nc"
        )
    return Path(filename), variable, label


def validate_settings() -> None:
    """Check settings before loading data or generating bootstrap samples."""
    if REFERENCE_DATASET not in {"era5", "senorge"}:
        raise ValueError("REFERENCE_DATASET must be 'era5' or 'senorge'.")
    valid_methods = {"mm_1step", "mm_2step", "q", "doy", "ld", "q_doy"}
    if BIAS_CORRECTION_METHOD not in valid_methods:
        raise ValueError(f"BIAS_CORRECTION_METHOD must be one of {sorted(valid_methods)}.")
    if x_days < 1 or number_of_bootstrap_samples < 1:
        raise ValueError("x_days and number_of_bootstrap_samples must be positive.")
    if not 0 < confidence_level_percent < 100:
        raise ValueError("confidence_level_percent must be between 0 and 100.")
    if not 0 <= annotation_decimals <= 4:
        raise ValueError("annotation_decimals must be between 0 and 4.")
    first_year, last_year = map(int, reference_years)
    file_start, file_end = map(int, REFERENCE_FILE_YEARS)
    if not file_start <= first_year <= last_year <= file_end:
        raise ValueError("reference_years must be increasing and within REFERENCE_FILE_YEARS.")


def finite_values(values: np.ndarray) -> np.ndarray:
    """Flatten values and discard missing or infinite entries."""
    values = np.asarray(values).ravel()
    return values[np.isfinite(values)]


def get_calendar_month(model_ds: xr.Dataset) -> xr.DataArray:
    """Extract calendar month from the compact YYYYMM coordinate."""
    if "tp24_max" not in model_ds or MODEL_MONTH_COORDINATE not in model_ds:
        raise KeyError("Model data must contain tp24_max and sample_month.")
    if set(model_ds["tp24_max"].dims) != {"number", "i_date"}:
        raise ValueError("tp24_max must have dimensions number and i_date.")
    sample_month = model_ds[MODEL_MONTH_COORDINATE]
    if sample_month.dims != ("i_date",):
        raise ValueError("sample_month must have dimension ('i_date',).")

    values = np.asarray(sample_month.values)
    valid = np.isfinite(values)
    months = np.full(values.shape, -1, dtype="int16")
    months[valid] = values[valid].astype("int64") % 100
    if np.any(valid & ~np.isin(months, MONTHS)):
        raise ValueError("sample_month contains invalid YYYYMM values.")
    return xr.DataArray(months, dims="i_date", coords={"i_date": model_ds["i_date"]})


def get_reference_values(ds: xr.Dataset, variable: str, month: int) -> np.ndarray:
    """Select reference years and optionally remove August 2023 Storm Hans."""
    if variable not in ds:
        raise KeyError(f"Reference variable {variable!r} was not found.")
    data = ds[variable]
    if not {"year", "month"}.issubset(set(data.dims) | set(data.coords)):
        raise ValueError("Reference data must contain year and month.")
    first_year, last_year = map(int, reference_years)
    selected = data.sel(year=slice(first_year, last_year), month=month)
    if EXCLUDE_STORM_HANS_FROM_REFERENCE and month == 8 and first_year <= 2023 <= last_year:
        if 2023 not in selected["year"].values:
            raise ValueError("Cannot exclude Storm Hans: August 2023 is absent.")
        selected = selected.sel(year=selected["year"] != 2023)
    return finite_values(selected.values)


def calculate_statistic(values: np.ndarray, statistic: str, axis=None):
    """Use the original sample std and biased skewness/excess-kurtosis estimators."""
    if statistic == "mean":
        return np.mean(values, axis=axis)
    if statistic == "std":
        return np.std(values, axis=axis, ddof=1)
    if statistic == "kurtosis":
        return kurtosis(values, axis=axis, fisher=True, bias=True)
    if statistic == "skewness":
        return skew(values, axis=axis, bias=True)
    raise ValueError(f"Unsupported statistic: {statistic}")


def evaluate_month(model: np.ndarray, reference: np.ndarray, month: int) -> list[dict]:
    """Return reference percentiles and the unchanged interval-based decisions."""
    if min(model.size, reference.size) < 4:
        raise ValueError(f"{month_abbr[month]} requires at least four model and reference values.")
    rng = np.random.default_rng(random_seed + month)
    indices = rng.integers(0, model.size, size=(number_of_bootstrap_samples, reference.size))
    resampled = model[indices]
    tail = (100.0 - confidence_level_percent) / 2.0
    results = []

    for statistic in STATISTICS:
        bootstrap = finite_values(calculate_statistic(resampled, statistic, axis=1))
        observed = float(calculate_statistic(reference, statistic))
        if bootstrap.size == 0 or not np.isfinite(observed):
            raise ValueError(f"{month_abbr[month]}: {statistic} is undefined for these samples.")
        low, high = np.percentile(bootstrap, [tail, 100.0 - tail])
        percentile = 100.0 * (
            np.count_nonzero(bootstrap < observed) + 0.5 * np.count_nonzero(bootstrap == observed)
        ) / bootstrap.size
        results.append({
            "month": month,
            "test": statistic,
            "reference_value": observed,
            "percentile": percentile,
            "low": float(low),
            "high": float(high),
            "passes": bool(low <= observed <= high),
            "bootstrap_count": bootstrap.size,
        })
    return results


def evaluate_fidelity(model_ds: xr.Dataset, reference_ds: xr.Dataset, variable: str, months=MONTHS):
    """Evaluate the four fidelity tests for the requested calendar months."""
    calendar_month = get_calendar_month(model_ds)
    results = []
    for month in months:
        selected = model_ds["tp24_max"].where(calendar_month == month, drop=True)
        model = finite_values(selected.values)
        reference = get_reference_values(reference_ds, variable, month)
        results.extend(evaluate_month(model, reference, int(month)))
    return pd.DataFrame(results)


def evaluate_panel_b(raw_results, corrected_ds, reference_ds, reference_variable):
    """Use corrected data for selected whole months and retain other raw results."""
    if BIAS_CORRECT_ONLY_FAILED_MONTHS:
        months = raw_results.loc[~raw_results["passes"], "month"].unique()
    else:
        months = MONTHS
    print("Panel (b) corrected months:", ", ".join(month_abbr[m] for m in months) or "None")
    if len(months) == 0:
        return raw_results.copy()

    corrected = evaluate_fidelity(corrected_ds, reference_ds, reference_variable, months)
    retained = raw_results.loc[~raw_results["month"].isin(months)]
    return pd.concat([retained, corrected], ignore_index=True)


def plot_fidelity_heatmap(axis, results: pd.DataFrame, title: str) -> None:
    """Draw on supplied axes so this panel can be reused in a comparison figure."""
    percentiles = results.pivot(index="test", columns="month", values="percentile")
    passes = results.pivot(index="test", columns="month", values="passes")
    percentiles = percentiles.loc[list(STATISTICS), MONTHS].to_numpy()
    passes = passes.loc[list(STATISTICS), MONTHS].to_numpy(dtype=bool)
    axis.imshow(
        passes.astype(int), cmap=ListedColormap([FAIL_COLOR, PASS_COLOR]),
        vmin=0, vmax=1, aspect="auto", interpolation="nearest",alpha=1.0,
    )
    for (row, column), percentile in np.ndenumerate(percentiles):
        axis.text(
            column, row, f"{percentile:.{annotation_decimals}f}",
            ha="center", va="center", fontsize=10,
            color="white" if passes[row, column] else "#1A1A1A",
        )

    axis.set_xticks(np.arange(12), [month_abbr[month] for month in MONTHS])
    axis.set_yticks(np.arange(4), TEST_LABELS)
    axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="both", length=0, pad=7)
    axis.set_ylabel("Fidelity test", labelpad=12)
    axis.set_title(title, loc="center", fontsize=12, fontweight="normal", pad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_figure(raw_results: pd.DataFrame, corrected_results: pd.DataFrame, filename=None):
    """Plot raw and corrected results with shared month axes and a common legend."""
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True)
        plot_fidelity_heatmap(axes[0], raw_results, "(a) Raw model")
        plot_fidelity_heatmap(axes[1], corrected_results, f"(b) Bias-corrected model")
        axes[0].tick_params(axis="x", labelbottom=True)
        axes[1].set_xlabel("Month", labelpad=10)
        axes[0].set_xlabel("Month", labelpad=10)
        figure.subplots_adjust(left=0.20, right=0.99, top=0.93, bottom=0.17, hspace=0.42)
        figure.legend(
            handles=[Patch(facecolor=PASS_COLOR, label="Pass"),
                     Patch(facecolor=FAIL_COLOR, label="Fail")],
            title=f"Significance level: {confidence_level_percent:g}%",
            loc="lower left", bbox_to_anchor=(0.82, 0.025), ncol=2,
            frameon=False, handlelength=1.2, handleheight=1.0,
            title_fontsize=10,
        )
        if filename is not None:
            filename.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(filename, dpi=figure_dpi, bbox_inches="tight", facecolor="white")
            print("Wrote:", filename)
        if show_figure:
            plt.show()
        plt.close(figure)


def main() -> None:
    validate_settings()
    raw_filename = build_model_filename("raw")
    corrected_filename = build_model_filename(BIAS_CORRECTION_METHOD)
    reference_filename, reference_variable, reference_label = get_reference_file()
    for filename in (raw_filename, corrected_filename, reference_filename):
        if not filename.is_file():
            raise FileNotFoundError(f"Required input file not found: {filename}")
    print("Raw model:", raw_filename)
    print(f"Bias-corrected model ({BIAS_CORRECTION_METHOD}):", corrected_filename)
    print(f"{reference_label}:", reference_filename)

    with (
        xr.open_dataset(raw_filename, decode_timedelta=False) as raw_ds,
        xr.open_dataset(corrected_filename, decode_timedelta=False) as corrected_ds,
        xr.open_dataset(reference_filename) as reference_ds,
    ):
        raw_results = evaluate_fidelity(raw_ds, reference_ds, reference_variable)
        corrected_results = evaluate_panel_b(
            raw_results, corrected_ds, reference_ds, reference_variable
        )

    for label, results in [("Raw model", raw_results), ("Bias-corrected model", corrected_results)]:
        print(f"\n{label}: reference-statistic percentiles (0–100)")
        table = results.pivot(index="test", columns="month", values="percentile")
        table = table.loc[list(STATISTICS), MONTHS]
        table.columns = [month_abbr[month] for month in MONTHS]
        print(table.to_string(float_format=lambda value: f"{value:.{annotation_decimals}f}"))

    filename = None
    correction_mode = "failed-months" if BIAS_CORRECT_ONLY_FAILED_MONTHS else "all-months"
    if write2file:
        filename = Path(output_filename_override) if output_filename_override else (Path(config.dirs["fig"]) / ("fig-S4.png"))
    make_figure(raw_results, corrected_results, filename)


if __name__ == "__main__":
    main()
