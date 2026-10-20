#!/usr/bin/env python3
"""Sensitivity of the monthly fidelity tests (fig-S1) to the bias-correction method.

Layout (3 rows x 2 columns, read left to right, top to bottom)
------
    (a) Raw model            (b) BC mm_1step
    (c) BC q                 (d) BC doy
    (e) BC ld                (f) BC q_doy

Each panel repeats the fig-S1 heatmap: columns are calendar months, rows are
mean, std, kurtosis, and skewness. Numbers are reference-statistic percentiles
in the model bootstrap distributions. Blue cells pass the central bootstrap
interval test; orange cells fail. Colours use the unrounded interval test.

Statistic definitions, bootstrap settings, monthly random seeds (random_seed +
month), reference selection, and filenames follow fig-S1, so the raw panel and
the panel for fig-S1's BIAS_CORRECTION_METHOD reproduce fig-S1 exactly for
identical settings.

If BIAS_CORRECT_ONLY_FAILED_MONTHS is True, each corrected panel uses corrected
data only for months failing at least one raw test and reuses the raw results
for all other months. If False, corrected panels use corrected data throughout.

Bias-corrected files that do not exist are skipped with a warning when
SKIP_MISSING_MODEL_FILES is True (the grid shrinks accordingly); otherwise a
FileNotFoundError is raised.

Requires numpy, scipy, pandas, xarray, matplotlib, and the project config.
"""
from calendar import month_abbr
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kurtosis, skew

from Dunnsigouin_etal_2026 import config

# =============================================================================
# USER INPUTS — data selection
# =============================================================================
x_days = 2  # Precipitation accumulation length in days.
catchment = "regine_drammen"
forecast_date_range = ("2020-01-02", "2023-12-28")
reference_years = ("1957", "2025")  # Reference period used in the fidelity tests.
REFERENCE_FILE_YEARS = ("1957", "2025")  # Full period encoded in the reference filename.
REFERENCE_DATASET = "senorge"  # "era5" or "senorge".
EXCLUDE_STORM_HANS_FROM_REFERENCE = True  # Remove August 2023 from the reference sample.

# Bias-correction methods shown after the raw panel, in this order.
BIAS_CORRECTION_METHODS = ["mm_1step", "q", "doy", "ld", "q_doy"]
SKIP_MISSING_MODEL_FILES = True
# True: correct all four tests for any month failing at least one raw test.
# False: use corrected data for every month in every corrected panel.
BIAS_CORRECT_ONLY_FAILED_MONTHS = False

# Optional explicit reference path; None builds the filename from the settings above.
reference_filename_override = None

# =============================================================================
# BOOTSTRAP SETTINGS
# =============================================================================
number_of_bootstrap_samples = 10_000
confidence_level_percent = 95.0  # Central bootstrap interval used for pass/fail decisions.
random_seed = 42  # Each calendar month uses random_seed + month.

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
write2file = True  # Set True to save the figure.
show_figure = True
output_directory = Path(config.dirs["fig"])
output_filename = "fig-S1-sensitivity-bc-methods.png"  # Also supports .pdf and .svg.
output_filename_override = None  # Optional full path; takes precedence over the two above.
output_path = (
    Path(output_filename_override) if output_filename_override
    else output_directory / output_filename
)
figure_dpi = 400

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
NUMBER_OF_COLUMNS = 2
figure_width = 14.0  # Inches.
panel_row_height = 2.6  # Inches per row of panels.
annotation_decimals = 1  # Display precision only; does not affect pass/fail decisions.
annotation_fontsize = 8
PASS_COLOR = "#0072B2"  # Okabe-Ito blue.
FAIL_COLOR = "#E69F00"  # Okabe-Ito orange.

# =============================================================================
# DATASET AND TEST CONVENTIONS
# =============================================================================
MODEL_MONTH_COORDINATE = "sample_month"
STATISTICS = ("mean", "std", "kurtosis", "skewness")
TEST_LABELS = ("Mean", "Standard deviation", "Kurtosis", "Skewness")
MONTHS = np.arange(1, 13)
VALID_BC_METHODS = ("mm_1step", "mm_2step", "q", "doy", "ld", "q_doy")

# =============================================================================
# Input filenames and validation
# =============================================================================


def build_model_filename(method: str) -> Path:
    """Use script 1's compact raw and bias-corrected filename convention."""
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
    unsupported = set(BIAS_CORRECTION_METHODS) - set(VALID_BC_METHODS)
    if unsupported:
        raise ValueError(f"Unsupported BIAS_CORRECTION_METHODS: {sorted(unsupported)}")
    if len(set(BIAS_CORRECTION_METHODS)) != len(BIAS_CORRECTION_METHODS):
        raise ValueError("BIAS_CORRECTION_METHODS contains duplicates.")
    if x_days < 1 or number_of_bootstrap_samples < 1:
        raise ValueError("x_days and number_of_bootstrap_samples must be positive.")
    if not 0 < confidence_level_percent < 100:
        raise ValueError("confidence_level_percent must be between 0 and 100.")
    if not 0 <= annotation_decimals <= 4:
        raise ValueError("annotation_decimals must be between 0 and 4.")
    if NUMBER_OF_COLUMNS < 1:
        raise ValueError("NUMBER_OF_COLUMNS must be at least 1.")

    first_year, last_year = map(int, reference_years)
    file_start, file_end = map(int, REFERENCE_FILE_YEARS)
    if not file_start <= first_year <= last_year <= file_end:
        raise ValueError("reference_years must be increasing and within REFERENCE_FILE_YEARS.")


def resolve_corrected_methods() -> list[str]:
    """Return the bias-correction methods whose files exist."""
    methods = []
    for method in BIAS_CORRECTION_METHODS:
        filename = build_model_filename(method)
        if filename.is_file():
            methods.append(method)
        elif SKIP_MISSING_MODEL_FILES:
            print(f"Warning: skipping '{method}', file not found: {filename}")
        else:
            raise FileNotFoundError(f"Required input file not found: {filename}")
    return methods

# =============================================================================
# Data preparation
# =============================================================================


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

# =============================================================================
# Bootstrap fidelity calculations
# =============================================================================


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
    """Return reference percentiles and the interval-based decisions."""
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


def evaluate_corrected(method, raw_results, reference_ds, reference_variable):
    """Evaluate one bias-correction method, optionally only in raw-failing months."""
    if BIAS_CORRECT_ONLY_FAILED_MONTHS:
        months = raw_results.loc[~raw_results["passes"], "month"].unique()
    else:
        months = MONTHS
    print(f"{method}: corrected months:", ", ".join(month_abbr[m] for m in months) or "None")
    if len(months) == 0:
        return raw_results.copy()

    with xr.open_dataset(build_model_filename(method), decode_timedelta=False) as corrected_ds:
        corrected = evaluate_fidelity(corrected_ds, reference_ds, reference_variable, months)
    retained = raw_results.loc[~raw_results["month"].isin(months)]
    return pd.concat([retained, corrected], ignore_index=True)

# =============================================================================
# Plotting
# =============================================================================


def panel_title(method: str) -> str:
    """Return the display name of one panel."""
    return "Raw model" if method == "raw" else f"Bias-corrected: {method}"


def plot_fidelity_heatmap(axis, results: pd.DataFrame, title: str,
                          show_xlabels: bool, show_ylabels: bool) -> None:
    """Draw one fidelity heatmap on the supplied axes."""
    percentiles = results.pivot(index="test", columns="month", values="percentile")
    passes = results.pivot(index="test", columns="month", values="passes")
    percentiles = percentiles.loc[list(STATISTICS), MONTHS].to_numpy()
    passes = passes.loc[list(STATISTICS), MONTHS].to_numpy(dtype=bool)
    axis.imshow(
        passes.astype(int), cmap=ListedColormap([FAIL_COLOR, PASS_COLOR]),
        vmin=0, vmax=1, aspect="auto", interpolation="nearest",
    )

    for (row, column), percentile in np.ndenumerate(percentiles):
        axis.text(
            column, row, f"{percentile:.{annotation_decimals}f}",
            ha="center", va="center", fontsize=annotation_fontsize,
            color="white" if passes[row, column] else "#1A1A1A",
        )

    axis.set_xticks(np.arange(12), [month_abbr[month] for month in MONTHS])
    axis.set_yticks(np.arange(4), TEST_LABELS)
    axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="both", length=0, pad=5,
                     labelbottom=show_xlabels, labelleft=show_ylabels)
    if show_xlabels:
        axis.set_xlabel("Month", labelpad=6)
    n_pass = int(passes.sum())
    axis.set_title(f"{title}  ({n_pass}/{passes.size} pass)",
                   loc="left", fontsize=11, fontweight="normal", pad=6)
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_figure(panel_results: dict[str, pd.DataFrame], filename=None):
    """Plot all panels in a grid with shared axes labels and a common legend."""
    methods = list(panel_results)
    number_of_columns = min(NUMBER_OF_COLUMNS, len(methods))
    number_of_rows = int(np.ceil(len(methods) / number_of_columns))
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            number_of_rows, number_of_columns,
            figsize=(figure_width, panel_row_height * number_of_rows + 0.8),
            squeeze=False, constrained_layout=True,
        )
        for index, method in enumerate(methods):
            row, column = divmod(index, number_of_columns)
            # Month labels only where no panel sits directly below.
            show_xlabels = index + number_of_columns >= len(methods)
            plot_fidelity_heatmap(
                axes[row, column], panel_results[method],
                f"({ascii_lowercase[index]}) {panel_title(method)}",
                show_xlabels=show_xlabels, show_ylabels=(column == 0),
            )
        for index in range(len(methods), number_of_rows * number_of_columns):
            axes.flat[index].set_visible(False)

        figure.legend(
            handles=[Patch(facecolor=PASS_COLOR, label="Pass"),
                     Patch(facecolor=FAIL_COLOR, label="Fail")],
            title=f"Significance level: {confidence_level_percent:g}%",
            loc="outside lower center", ncol=2,
            frameon=False, handlelength=1.2, handleheight=1.0, title_fontsize=10,
        )

        if filename is not None:
            filename.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(filename, dpi=figure_dpi, bbox_inches="tight", facecolor="white")
            print("Wrote:", filename)
        if show_figure:
            plt.show()
        plt.close(figure)

# =============================================================================
# Reporting
# =============================================================================


def print_tables(panel_results: dict[str, pd.DataFrame]) -> None:
    """Print percentile tables and a pass-count summary for every panel."""
    for method, results in panel_results.items():
        print(f"\n{panel_title(method)}: reference-statistic percentiles (0–100)")
        table = results.pivot(index="test", columns="month", values="percentile")
        table = table.loc[list(STATISTICS), MONTHS]
        table.columns = [month_abbr[month] for month in MONTHS]
        print(table.to_string(float_format=lambda value: f"{value:.{annotation_decimals}f}"))

    print("\nPassing tests per method (out of 12 months)")
    rows = {
        panel_title(method): results.groupby("test")["passes"].sum()
        .reindex(list(STATISTICS)).astype(int)
        for method, results in panel_results.items()
    }
    summary = pd.DataFrame(rows).T
    summary.columns = TEST_LABELS
    summary["Total (of 48)"] = summary.sum(axis=1)
    print(summary.to_string())

# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    """Evaluate raw and every bias-corrected dataset, then plot all panels."""
    validate_settings()
    raw_filename = build_model_filename("raw")
    reference_filename, reference_variable, reference_label = get_reference_file()
    for filename in (raw_filename, reference_filename):
        if not filename.is_file():
            raise FileNotFoundError(f"Required input file not found: {filename}")
    corrected_methods = resolve_corrected_methods()

    print("Figure output:", output_path, "(saving enabled)" if write2file else "(saving disabled)")
    print("Raw model:", raw_filename)
    for method in corrected_methods:
        print(f"Bias-corrected model ({method}):", build_model_filename(method))
    print(f"{reference_label}:", reference_filename)

    with (
        xr.open_dataset(raw_filename, decode_timedelta=False) as raw_ds,
        xr.open_dataset(reference_filename) as reference_ds,
    ):
        raw_results = evaluate_fidelity(raw_ds, reference_ds, reference_variable)
        panel_results = {"raw": raw_results}
        for method in corrected_methods:
            panel_results[method] = evaluate_corrected(
                method, raw_results, reference_ds, reference_variable
            )

    print_tables(panel_results)
    make_figure(panel_results, output_path if write2file else None)


if __name__ == "__main__":
    main()
