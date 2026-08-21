"""
Diagnose the August reference-data GEV fit and bootstrap uncertainty.

Panels
------
a) August return-level curve for the reference observations only.
b) Bootstrap distributions of GEV shape, location, and scale parameters.
c) Bootstrap distribution of the finite GEV upper endpoint for fits with shape > 0.
d) Bootstrap return levels at selected return periods.

Additional printed diagnostics compare GEV and Gumbel fits and report sensitivity
of the GEV curve to the minimum plotted return period.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import genextreme, gumbel_r

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

REFERENCE_DATASET = "senorge"  # "senorge" or "era5"
CATCHMENT = "regine_drammen"
X_DAYS = 2

OBSERVATION_YEARS = [1957, 2022]
REFERENCE_FILE_YEARS = [1957, 2025]
REFERENCE_FILENAME_OVERRIDE = None

NUMBER_OF_BOOTSTRAPS = 1000
CONFIDENCE_LEVEL = 0.95
MIN_SUCCESSFUL_BOOTSTRAP_FRACTION = 0.90
RANDOM_SEED = 42

RETURN_PERIOD_MIN = 1.0
RETURN_PERIOD_MAX = 1.0e7
NUMBER_OF_RETURN_PERIODS = 500
SELECTED_RETURN_PERIODS = np.array([2, 5, 10, 20, 50, 100, 1000, 10000], dtype=float)
RETURN_PERIOD_STARTS = [1.01, 1.1, 1.5, 2.0]

PRECIPITATION_YMIN = 0.0
PRECIPITATION_YMAX = 200.0

FIGURE_DPI = 300
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 10
WRITE_TO_FILE = True
SHOW_FIGURE = True

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"

AUGUST = 8
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = AUGUST

OBSERVATION_COLOR = "tab:blue"
STORM_HANS_COLOR = "grey"
CONFIDENCE_ALPHA = 0.2
CURVE_LINEWIDTH = 2.0
MARKER_SIZE = 35
MARKER_LINEWIDTH = 1.0

AXIS_LABELSIZE = 11
TICK_LABELSIZE = 10
TITLE_FONTSIZE = 12


# =============================================================================
# Input and validation
# =============================================================================

def validate_settings():
    """Validate user-configurable settings."""
    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")
    if OBSERVATION_YEARS[0] > OBSERVATION_YEARS[1]:
        raise ValueError("OBSERVATION_YEARS must be increasing.")
    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError("NUMBER_OF_BOOTSTRAPS must be at least 1.")
    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError("CONFIDENCE_LEVEL must lie between 0 and 1.")
    if not 0 < MIN_SUCCESSFUL_BOOTSTRAP_FRACTION <= 1:
        raise ValueError("MIN_SUCCESSFUL_BOOTSTRAP_FRACTION must lie in (0, 1].")
    if RETURN_PERIOD_MIN < 1:
        raise ValueError("RETURN_PERIOD_MIN must be at least 1.")
    if RETURN_PERIOD_MAX <= RETURN_PERIOD_MIN:
        raise ValueError("RETURN_PERIOD_MAX must exceed RETURN_PERIOD_MIN.")


def get_reference_name():
    """Return the display name of the selected reference dataset."""
    return {"senorge": "seNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the precipitation variable in the selected reference dataset."""
    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def make_reference_filename():
    """Construct the hard-coded 1957-2025 reference filename."""
    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)

    first_year, last_year = REFERENCE_FILE_YEARS
    variable = get_reference_variable()
    filename = (
        f"monthly_max_samples_{variable}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_{first_year}-{last_year}.nc"
    )
    directory = (
        config.dirs["senorge_processed"]
        if REFERENCE_DATASET == "senorge"
        else config.dirs["era5_processed"]
    )
    return Path(directory) / filename


def make_figure_filename():
    """Return the diagnostic figure filename."""
    return Path(config.dirs["fig"]) / (
        f"test.png"
    )


def read_august_reference():
    """Read August observations for fitting and August 2023 Storm Hans."""
    filename = make_reference_filename()
    variable = get_reference_variable()

    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")

    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")

        selected = ds[variable].sel(
            year=slice(OBSERVATION_YEARS[0], OBSERVATION_YEARS[1]),
            month=AUGUST,
        ).load()
        storm_hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )

    years = np.asarray(selected["year"].values)
    values = np.asarray(selected.values, dtype=float)
    finite = np.isfinite(values)
    years, values = years[finite], values[finite]

    if values.size < 10:
        raise ValueError("Fewer than 10 finite August observations remain.")

    return years, values, storm_hans_value


# =============================================================================
# GEV and Gumbel methods
# =============================================================================

def fit_gev(values):
    """Fit SciPy's GEV and return shape, location, and scale."""
    parameters = genextreme.fit(values)
    if not np.isfinite(parameters).all() or parameters[-1] <= 0:
        raise RuntimeError("GEV fit returned invalid parameters.")
    return parameters


def fit_gumbel(values):
    """Fit a Gumbel distribution and return location and scale."""
    parameters = gumbel_r.fit(values)
    if not np.isfinite(parameters).all() or parameters[-1] <= 0:
        raise RuntimeError("Gumbel fit returned invalid parameters.")
    return parameters


def gev_return_levels(return_periods, parameters):
    """Evaluate GEV return levels."""
    probabilities = 1.0 - 1.0 / return_periods
    shape, location, scale = parameters
    return genextreme.ppf(probabilities, shape, loc=location, scale=scale)


def gumbel_return_levels(return_periods, parameters):
    """Evaluate Gumbel return levels."""
    probabilities = 1.0 - 1.0 / return_periods
    location, scale = parameters
    return gumbel_r.ppf(probabilities, loc=location, scale=scale)


def empirical_return_periods(values):
    """Return Weibull empirical return periods and descending values."""
    sorted_values = np.sort(values)[::-1]
    ranks = np.arange(1, sorted_values.size + 1)
    return (sorted_values.size + 1) / ranks, sorted_values


def gev_upper_endpoint(parameters):
    """Return the finite upper endpoint for SciPy GEV fits with shape > 0."""
    shape, location, scale = parameters
    return np.inf if shape <= 0 else location + scale / shape


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_gev(values):
    """Fit nonparametric bootstrap GEV parameter sets."""
    fitted_parameters = fit_gev(values)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_parameters = []

    for _ in range(NUMBER_OF_BOOTSTRAPS):
        sample = rng.choice(values, size=values.size, replace=True)
        try:
            bootstrap_parameters.append(fit_gev(sample))
        except (RuntimeError, ValueError, FloatingPointError):
            continue

    minimum = int(np.ceil(MIN_SUCCESSFUL_BOOTSTRAP_FRACTION * NUMBER_OF_BOOTSTRAPS))
    if len(bootstrap_parameters) < minimum:
        raise RuntimeError(
            f"Only {len(bootstrap_parameters)} of {NUMBER_OF_BOOTSTRAPS} "
            "GEV bootstrap fits succeeded."
        )

    return fitted_parameters, np.asarray(bootstrap_parameters, dtype=float)


def bootstrap_return_level_summary(bootstrap_parameters, return_periods):
    """Return pointwise bootstrap confidence intervals for GEV return levels."""
    bootstrap_levels = np.array(
        [gev_return_levels(return_periods, parameters) for parameters in bootstrap_parameters]
    )
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = np.percentile(bootstrap_levels, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(bootstrap_levels, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return bootstrap_levels, lower, upper


# =============================================================================
# Diagnostics
# =============================================================================

def print_diagnostics(values, gev_parameters, bootstrap_parameters):
    """Print concise GEV/Gumbel and bootstrap diagnostics."""
    gumbel_parameters = fit_gumbel(values)
    shape, location, scale = gev_parameters
    endpoint = gev_upper_endpoint(gev_parameters)

    print("August reference GEV diagnostics")
    print("--------------------------------")
    print(f"Reference file:       {make_reference_filename()}")
    print(f"Observation years:    {OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}")
    print(f"Number of values:     {values.size}")
    print(f"GEV shape:            {shape:.6f}")
    print(f"GEV location:         {location:.3f}")
    print(f"GEV scale:            {scale:.3f}")
    print(
        f"GEV upper endpoint:   {endpoint:.3f} mm"
        if np.isfinite(endpoint)
        else "GEV upper endpoint:   unbounded"
    )
    print(f"Gumbel location:      {gumbel_parameters[0]:.3f}")
    print(f"Gumbel scale:         {gumbel_parameters[1]:.3f}")
    print(f"Successful bootstraps:{bootstrap_parameters.shape[0]}")

    positive_shape = bootstrap_parameters[:, 0] > 0
    print(f"Bootstrap shape > 0:  {100.0 * positive_shape.mean():.1f}%")

    print()
    print("Sensitivity to minimum plotted return period")
    for start in RETURN_PERIOD_STARTS:
        grid = np.geomspace(start, RETURN_PERIOD_MAX, NUMBER_OF_RETURN_PERIODS)
        first_level = gev_return_levels(grid, gev_parameters)[0]
        print(f"Tmin={start:>4}: first fitted return level = {first_level:.3f} mm")


# =============================================================================
# Plotting
# =============================================================================

def plot_return_level_panel(axis, values, storm_hans_value, gev_parameters, bootstrap_parameters):
    """Plot the reference-only August GEV fit and bootstrap confidence interval."""
    return_periods = np.geomspace(
        max(RETURN_PERIOD_MIN, 1.0 + np.finfo(float).eps),
        RETURN_PERIOD_MAX,
        NUMBER_OF_RETURN_PERIODS,
    )
    fitted_levels = gev_return_levels(return_periods, gev_parameters)
    _, lower, upper = bootstrap_return_level_summary(bootstrap_parameters, return_periods)
    empirical_rp, empirical_values = empirical_return_periods(values)

    axis.fill_between(
        return_periods,
        lower,
        upper,
        alpha=CONFIDENCE_ALPHA,
        linewidth=0,
        zorder=1,
    )
    axis.plot(
        return_periods,
        fitted_levels,
        linewidth=CURVE_LINEWIDTH,
        label=f"{get_reference_name()} GEV fit",
        zorder=2,
    )
    axis.scatter(
        empirical_rp,
        empirical_values,
        facecolors="none",
        linewidths=MARKER_LINEWIDTH,
        s=MARKER_SIZE,
        label="Empirical observations",
        zorder=3,
    )
    axis.axhline(
        storm_hans_value,
        linestyle="--",
        linewidth=1.8,
        label="Storm Hans August 2023",
        zorder=4,
    )

    axis.set_xscale("log")
    axis.set_xlim(RETURN_PERIOD_MIN, RETURN_PERIOD_MAX)
    axis.set_ylim(PRECIPITATION_YMIN, PRECIPITATION_YMAX)
    axis.set_xlabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    axis.set_ylabel(
        f"Monthly maximum {X_DAYS}-day precipitation [mm]",
        fontsize=AXIS_LABELSIZE,
    )
    axis.set_title("a) August reference GEV fit", loc="left", fontsize=TITLE_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=9, loc="upper left")


def plot_parameter_panel(axis, gev_parameters, bootstrap_parameters):
    """Plot bootstrap distributions of GEV shape, location, and scale."""
    labels = ["Shape", "Location", "Scale"]
    data = [bootstrap_parameters[:, i] for i in range(3)]

    axis.boxplot(data, labels=labels, showfliers=True)
    axis.scatter(np.arange(1, 4), gev_parameters, zorder=3, label="Original fit")

    axis.set_title("b) Bootstrap GEV parameters", loc="left", fontsize=TITLE_FONTSIZE)
    axis.set_ylabel("Parameter value", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=9, loc="best")


def plot_endpoint_panel(axis, values, storm_hans_value, gev_parameters, bootstrap_parameters):
    """Plot finite upper endpoints implied by bootstrap GEV fits."""
    endpoints = np.array(
        [gev_upper_endpoint(parameters) for parameters in bootstrap_parameters],
        dtype=float,
    )
    endpoints = endpoints[np.isfinite(endpoints)]

    if endpoints.size:
        axis.boxplot([endpoints], labels=["Upper endpoint"], showfliers=True)
        fitted_endpoint = gev_upper_endpoint(gev_parameters)
        if np.isfinite(fitted_endpoint):
            axis.scatter(1, fitted_endpoint, zorder=3, label="Original-fit endpoint")
    else:
        axis.text(
            0.5,
            0.5,
            "No bootstrap fits have a finite upper endpoint",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.axhline(np.max(values), linestyle=":", linewidth=1.6, label="Observed August maximum")
    axis.axhline(storm_hans_value, linestyle="--", linewidth=1.6, label="Storm Hans")

    axis.set_title("c) Finite GEV upper endpoints", loc="left", fontsize=TITLE_FONTSIZE)
    axis.set_ylabel("Precipitation [mm]", fontsize=AXIS_LABELSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=9, loc="best")


def plot_selected_return_period_panel(axis, gev_parameters, bootstrap_parameters):
    """Plot bootstrap GEV return levels at selected return periods."""
    bootstrap_levels = np.array(
        [
            gev_return_levels(SELECTED_RETURN_PERIODS, parameters)
            for parameters in bootstrap_parameters
        ]
    )
    fitted_levels = gev_return_levels(SELECTED_RETURN_PERIODS, gev_parameters)

    axis.boxplot(
        [bootstrap_levels[:, i] for i in range(SELECTED_RETURN_PERIODS.size)],
        labels=[f"{value:g}" for value in SELECTED_RETURN_PERIODS],
        showfliers=True,
    )
    axis.scatter(
        np.arange(1, SELECTED_RETURN_PERIODS.size + 1),
        fitted_levels,
        zorder=3,
        label="Original fit",
    )

    axis.set_title(
        "d) Bootstrap return levels at selected periods",
        loc="left",
        fontsize=TITLE_FONTSIZE,
    )
    axis.set_xlabel("Return period [years]", fontsize=AXIS_LABELSIZE)
    axis.set_ylabel(
        f"Return level [mm / {X_DAYS} days]",
        fontsize=AXIS_LABELSIZE,
    )
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=9, loc="best")


def make_figure(values, storm_hans_value, gev_parameters, bootstrap_parameters):
    """Create the four-panel August GEV diagnostic figure."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    plot_return_level_panel(
        axes[0, 0],
        values,
        storm_hans_value,
        gev_parameters,
        bootstrap_parameters,
    )
    plot_parameter_panel(axes[0, 1], gev_parameters, bootstrap_parameters)
    plot_endpoint_panel(
        axes[1, 0],
        values,
        storm_hans_value,
        gev_parameters,
        bootstrap_parameters,
    )
    plot_selected_return_period_panel(
        axes[1, 1],
        gev_parameters,
        bootstrap_parameters,
    )

    if WRITE_TO_FILE:
        filename = make_figure_filename()
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()

    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the August reference-data GEV diagnostics."""
    validate_settings()

    _, values, storm_hans_value = read_august_reference()
    gev_parameters, bootstrap_parameters = bootstrap_gev(values)

    print_diagnostics(values, gev_parameters, bootstrap_parameters)
    make_figure(values, storm_hans_value, gev_parameters, bootstrap_parameters)


if __name__ == "__main__":
    main()
