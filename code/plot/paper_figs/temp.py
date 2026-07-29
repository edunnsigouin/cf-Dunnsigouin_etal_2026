"""
Plot a stationary GEV return-period curve for one calendar month of a
reference precipitation-extreme dataset.

The reference dataset can be either SeNorge or ERA5. Both datasets contain
one monthly maximum X-day accumulated precipitation value for each year and
calendar month:

    SeNorge: rr(year, month)
    ERA5:    tp24(year, month)

For the selected calendar month, this script:

1. Reads the reference data over OBSERVATION_YEARS.
2. Optionally excludes 2023 from the sample used to fit the GEV.
3. Fits a stationary three-parameter Generalized Extreme Value (GEV)
   distribution by maximum likelihood.
4. Calculates the fitted return-level curve.
5. Estimates a 95% confidence interval using a parametric bootstrap.
6. Plots the observations used in the GEV fit at empirical return periods.
7. Optionally plots Storm Hans separately.

Storm Hans occurred on 2023-08-08. The input files contain monthly maxima,
not daily event values, so when SELECTED_MONTH = 8 the script uses the
August 2023 monthly maximum as the Storm Hans magnitude.

This follows the stationary-observation part of Fig. 8 in
Kelder et al. (2020).

Important
---------
SciPy's ``genextreme`` shape parameter ``c`` has the opposite sign from the
conventional extreme-value shape parameter xi:

    xi = -c
"""

# =============================================================================
# Imports
# =============================================================================

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import genextreme

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User-defined input parameters
# =============================================================================

# Choose the observational reference dataset.
#
# Options:
#     "senorge"
#     "era5"
REFERENCE_DATASET = "senorge"

CATCHMENT = "regine_drammen"
X_DAYS = 2

# Full period read from the NetCDF file.
OBSERVATION_YEARS = [
    "1957",
    "2023",
]

# If True, 2023 is read from the file but excluded from the GEV fit.
# With the settings above, the fitted observational record is therefore
# 1957-2022.
EXCLUDE_2023_FROM_FIT = True

# Dataset-specific settings.
SENORGE_VARIABLE = "rr"
SENORGE_LABEL = "SeNorge"

ERA5_VARIABLE = "tp24"
ERA5_LABEL = "ERA5"
ERA5_GRID = "0.5x0.5"

# Plot colors for the selected reference dataset and Storm Hans.
SENORGE_COLOR = "tab:blue"
ERA5_COLOR = "tab:blue"

# Calendar month to analyse:
# 1 = January, ..., 8 = August, ..., 12 = December.
SELECTED_MONTH = 5

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Storm Hans settings.
#
# Since these NetCDF files contain monthly maxima, STORM_HANS_DATE is used
# for the plot label and to identify the relevant year/month. The actual
# plotted magnitude is the monthly maximum for August 2023.
STORM_HANS_DATE = "2023-08-08"
PLOT_STORM_HANS = True

# Return-period range shown on the fitted curve.
MIN_RETURN_PERIOD = 1.01
MAX_RETURN_PERIOD = 10_000.0
NUMBER_OF_RETURN_PERIODS = 500

# Parametric-bootstrap settings.
# Use 10,000 for the final figure. A smaller value can be useful while testing.
NUMBER_OF_BOOTSTRAPS = 100
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Plot settings.
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.8

TITLE_FONTSIZE = 11
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 9

YMIN = 0.0
YMAX = None

WRITE_TO_FILE = False


# =============================================================================
# Functions
# =============================================================================

def validate_user_settings():
    """Check the user-defined settings before doing any calculations."""

    valid_reference_datasets = {
        "senorge",
        "era5",
    }

    if REFERENCE_DATASET not in valid_reference_datasets:
        raise ValueError(
            f"REFERENCE_DATASET must be one of "
            f"{sorted(valid_reference_datasets)}. "
            f"Got '{REFERENCE_DATASET}'."
        )

    if SELECTED_MONTH not in range(1, 13):
        raise ValueError(
            "SELECTED_MONTH must be an integer from 1 to 12."
        )

    if X_DAYS < 1:
        raise ValueError(
            "X_DAYS must be at least 1."
        )

    if MIN_RETURN_PERIOD <= 1:
        raise ValueError(
            "MIN_RETURN_PERIOD must be greater than 1 year."
        )

    if MAX_RETURN_PERIOD <= MIN_RETURN_PERIOD:
        raise ValueError(
            "MAX_RETURN_PERIOD must be larger than MIN_RETURN_PERIOD."
        )

    if NUMBER_OF_BOOTSTRAPS < 1:
        raise ValueError(
            "NUMBER_OF_BOOTSTRAPS must be at least 1."
        )

    if not 0 < CONFIDENCE_LEVEL < 1:
        raise ValueError(
            "CONFIDENCE_LEVEL must lie between 0 and 1."
        )


def get_reference_variable():
    """Return the variable name for the selected reference dataset."""

    if REFERENCE_DATASET == "senorge":
        return SENORGE_VARIABLE

    return ERA5_VARIABLE


def get_reference_label():
    """Return a plot-friendly label for the selected reference dataset."""

    if REFERENCE_DATASET == "senorge":
        return SENORGE_LABEL

    return ERA5_LABEL


def get_reference_color():
    """Return the plot color for the selected reference dataset."""

    if REFERENCE_DATASET == "senorge":
        return SENORGE_COLOR

    return ERA5_COLOR


def make_reference_filename():
    """Construct the input filename for the selected reference dataset."""

    if REFERENCE_DATASET == "senorge":

        filename = (
            f"distribution_monthly_extremes_"
            f"{SENORGE_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_senorge_"
            f"{OBSERVATION_YEARS[0]}-"
            f"{OBSERVATION_YEARS[1]}.nc"
        )

        return os.path.join(
            config.dirs["senorge_processed"],
            filename,
        )

    filename = (
        f"distribution_monthly_extremes_"
        f"{ERA5_VARIABLE}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_era5_{ERA5_GRID}_"
        f"{OBSERVATION_YEARS[0]}-"
        f"{OBSERVATION_YEARS[1]}.nc"
    )

    return os.path.join(
        config.dirs["era5_processed"],
        filename,
    )


def make_figure_filename():
    """Construct the output figure filename."""

    month_name = MONTH_NAMES[
        SELECTED_MONTH - 1
    ].lower()

    filename = (
        f"return-period-gev-"
        f"{REFERENCE_DATASET}-"
        f"{CATCHMENT}-"
        f"{X_DAYS}dayacc-"
        f"{month_name}.png"
    )

    return os.path.join(
        config.dirs["fig"],
        filename,
    )


def check_variable_exists(
    ds,
    variable,
):
    """Raise a clear error if the required variable is absent."""

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found. "
            f"Available data variables are: {list(ds.data_vars)}"
        )


def read_reference_data(
    filename,
):
    """
    Read the selected calendar month for the GEV analysis and, independently,
    read the August 2023 maximum for Storm Hans.

    This means Storm Hans can be plotted as a horizontal reference line
    regardless of which calendar month is selected for the GEV fit.

    Returns
    -------
    years : ndarray
        Years available for SELECTED_MONTH.
    values : ndarray
        Monthly maxima for SELECTED_MONTH.
    hans_value : float or None
        August 2023 monthly maximum when PLOT_STORM_HANS is True.
    """

    variable = get_reference_variable()

    with xr.open_dataset(filename) as ds:

        check_variable_exists(
            ds,
            variable,
        )

        if "year" not in ds[variable].dims:
            raise ValueError(
                f"'{variable}' must contain a 'year' dimension."
            )

        if "month" not in ds[variable].dims:
            raise ValueError(
                f"'{variable}' must contain a 'month' dimension."
            )

        # Data for the calendar month being analysed.
        selected_month_data = (
            ds[variable]
            .sel(
                year=slice(
                    int(OBSERVATION_YEARS[0]),
                    int(OBSERVATION_YEARS[1]),
                ),
                month=SELECTED_MONTH,
            )
            .load()
        )

        # Storm Hans is always taken from August 2023, independently of
        # SELECTED_MONTH.
        hans_value = None

        if PLOT_STORM_HANS:

            try:
                hans_data = (
                    ds[variable]
                    .sel(
                        year=2023,
                        month=8,
                    )
                    .load()
                )

                hans_value_candidate = float(
                    hans_data.values
                )

                if np.isfinite(
                    hans_value_candidate
                ):
                    hans_value = (
                        hans_value_candidate
                    )

            except KeyError:
                hans_value = None

    years = np.asarray(
        selected_month_data["year"].values
    )

    values = np.asarray(
        selected_month_data.values,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    years = years[
        finite
    ]

    values = values[
        finite
    ]

    if PLOT_STORM_HANS:

        if hans_value is None:
            print()
            print(
                "Warning: no finite August 2023 value was found, "
                "so Storm Hans will not be plotted."
            )

        else:
            print()
            print(
                f"Storm Hans ({STORM_HANS_DATE}) "
                f"{get_reference_label()} August 2023 maximum: "
                f"{hans_value:.3f} mm"
            )

    return (
        years,
        values,
        hans_value,
    )

def create_fit_sample(
    years,
    values,
):
    """
    Create the sample used for the GEV fit.

    When EXCLUDE_2023_FROM_FIT is True, the selected-month value for 2023 is
    removed from the fitted sample.

    Storm Hans is handled separately and is always the August 2023 maximum.
    """

    fit_mask = np.ones(
        years.size,
        dtype=bool,
    )

    if EXCLUDE_2023_FROM_FIT:
        fit_mask &= (
            years != 2023
        )

    fit_years = years[
        fit_mask
    ]

    fit_values = values[
        fit_mask
    ]

    if fit_values.size < 10:
        raise ValueError(
            "Fewer than 10 finite values remain in the GEV fitting sample."
        )

    return (
        fit_years,
        fit_values,
    )

def fit_gev(
    values,
):
    """
    Fit a stationary three-parameter GEV by maximum likelihood.

    Returns
    -------
    shape_c : float
        SciPy's GEV shape parameter. Conventional xi = -shape_c.
    location : float
        GEV location parameter.
    scale : float
        GEV scale parameter.
    """

    shape_c, location, scale = genextreme.fit(
        values
    )

    if not np.isfinite(
        [shape_c, location, scale]
    ).all():
        raise RuntimeError(
            "The fitted GEV contains non-finite parameters."
        )

    if scale <= 0:
        raise RuntimeError(
            "The fitted GEV scale parameter is not positive."
        )

    return (
        shape_c,
        location,
        scale,
    )


def make_return_period_grid():
    """Create logarithmically spaced return periods for the fitted curve."""

    return np.geomspace(
        MIN_RETURN_PERIOD,
        MAX_RETURN_PERIOD,
        NUMBER_OF_RETURN_PERIODS,
    )


def calculate_gev_return_levels(
    return_periods,
    gev_parameters,
):
    """
    Convert return periods to fitted GEV return levels.

    For block maxima:

        p = 1 - 1 / T
    """

    shape_c, location, scale = gev_parameters

    probabilities = (
        1.0
        - 1.0 / return_periods
    )

    return genextreme.ppf(
        probabilities,
        shape_c,
        loc=location,
        scale=scale,
    )


def calculate_empirical_return_periods(
    values,
):
    """
    Calculate empirical return periods for plotting the observations.

    A Weibull plotting position is used:

        T = (n + 1) / rank

    with rank = 1 for the largest event.

    These empirical points are not used to fit the GEV.
    """

    sorted_values = np.sort(
        values
    )[::-1]

    ranks = np.arange(
        1,
        sorted_values.size + 1,
    )

    return_periods = (
        sorted_values.size + 1
    ) / ranks

    return (
        return_periods,
        sorted_values,
    )


def calculate_event_return_period(
    event_value,
    gev_parameters,
):
    """
    Calculate the fitted-GEV return period for one event.

    The survival function is used directly because it is numerically more
    stable in the far upper tail than calculating 1 - CDF.
    """

    shape_c, location, scale = gev_parameters

    exceedance_probability = genextreme.sf(
        event_value,
        shape_c,
        loc=location,
        scale=scale,
    )

    if (
        not np.isfinite(exceedance_probability)
        or exceedance_probability <= 0
    ):
        return np.inf

    return (
        1.0
        / exceedance_probability
    )

def parametric_bootstrap_return_levels(
    observed_values,
    fitted_parameters,
    return_periods,
):
    """
    Estimate uncertainty in the fitted GEV return-level curve.

    Each bootstrap replicate:
      1. draws a sample from the fitted GEV;
      2. uses the same sample size as the observations used in the fit;
      3. refits a three-parameter GEV;
      4. calculates a new return-level curve.

    The confidence interval is taken across these bootstrap curves at each
    return period.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    sample_size = observed_values.size

    bootstrap_return_levels = np.full(
        (
            NUMBER_OF_BOOTSTRAPS,
            return_periods.size,
        ),
        np.nan,
        dtype=float,
    )

    shape_c, location, scale = fitted_parameters

    successful_fits = 0

    for bootstrap_number in range(
        NUMBER_OF_BOOTSTRAPS
    ):

        simulated_values = genextreme.rvs(
            shape_c,
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

        try:
            bootstrap_parameters = fit_gev(
                simulated_values
            )

            bootstrap_levels = (
                calculate_gev_return_levels(
                    return_periods,
                    bootstrap_parameters,
                )
            )

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
        ):
            continue

        if not np.isfinite(
            bootstrap_levels
        ).all():
            continue

        bootstrap_return_levels[
            bootstrap_number,
            :
        ] = bootstrap_levels

        successful_fits += 1

    minimum_successful_fits = int(
        0.90
        * NUMBER_OF_BOOTSTRAPS
    )

    if successful_fits < minimum_successful_fits:
        raise RuntimeError(
            f"Only {successful_fits} of "
            f"{NUMBER_OF_BOOTSTRAPS} bootstrap GEV fits succeeded."
        )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower_percentile = (
        100.0
        * alpha / 2.0
    )

    upper_percentile = (
        100.0
        * (1.0 - alpha / 2.0)
    )

    lower = np.nanpercentile(
        bootstrap_return_levels,
        lower_percentile,
        axis=0,
    )

    upper = np.nanpercentile(
        bootstrap_return_levels,
        upper_percentile,
        axis=0,
    )

    return (
        lower,
        upper,
        successful_fits,
    )


def print_fit_summary(
    years,
    values,
    gev_parameters,
    successful_bootstraps,
    hans_value,
    hans_return_period,
):
    """Print a short summary of the fitted observational distribution."""

    shape_c, location, scale = gev_parameters

    shape_xi = -shape_c

    print()
    print(
        "Reference GEV fit"
    )
    print(
        "-----------------"
    )
    print(
        f"Dataset:          {get_reference_label()}"
    )
    print(
        f"Calendar month:   "
        f"{MONTH_NAMES[SELECTED_MONTH - 1]} "
        f"({SELECTED_MONTH})"
    )
    print(
        f"Years used:       "
        f"{int(years.min())}-{int(years.max())}"
    )
    print(
        f"Sample size:      {values.size}"
    )
    print(
        f"Location, mu:     {location:.3f} mm"
    )
    print(
        f"Scale, sigma:     {scale:.3f} mm"
    )
    print(
        f"Shape, xi:        {shape_xi:.4f}"
    )
    print(
        f"SciPy shape, c:   {shape_c:.4f}"
    )
    print(
        f"Bootstrap fits:   "
        f"{successful_bootstraps}/"
        f"{NUMBER_OF_BOOTSTRAPS}"
    )

    if hans_value is not None:
        print(
            f"Storm Hans value: {hans_value:.3f} mm"
        )
        print(
            f"Hans fitted RP:   {hans_return_period:.2f} years"
        )


def plot_return_period_curve(
    empirical_return_periods,
    empirical_values,
    fitted_return_periods,
    fitted_return_levels,
    lower_confidence_limit,
    upper_confidence_limit,
    hans_value,
    hans_return_period,
    filename_out,
):
    """Plot the fitted GEV, uncertainty, observations, and Storm Hans."""

    reference_label = get_reference_label()
    reference_color = get_reference_color()

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH_IN,
            FIG_HEIGHT_IN,
        )
    )

    ax.fill_between(
        fitted_return_periods,
        lower_confidence_limit,
        upper_confidence_limit,
        color=reference_color,
        alpha=0.20,
        label=(
            f"{int(CONFIDENCE_LEVEL * 100)}% "
            "parametric-bootstrap interval"
        ),
    )

    ax.plot(
        fitted_return_periods,
        fitted_return_levels,
        color=reference_color,
        linewidth=2.0,
        label="Fitted stationary GEV",
    )

    ax.scatter(
        empirical_return_periods,
        empirical_values,
        facecolors="none",
        edgecolors=reference_color,
        linewidths=1.2,
        s=30,
        zorder=3,
        label=(
            f"{reference_label} values used in fit"
        ),
    )

    if hans_value is not None:
        ax.axhline(
            y=hans_value,
            color=reference_color,
            linestyle="--",
            linewidth=1.5,
            zorder=4,
            label=(
                f"Storm Hans ({STORM_HANS_DATE})"
            ),
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        0,
        MAX_RETURN_PERIOD,
    )

    ax.set_ylim(
        bottom=YMIN,
        top=YMAX,
    )

    ax.set_xlabel(
        "Return period (years)",
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_ylabel(
        (
            f"{X_DAYS}-day accumulated "
            "precipitation (mm)"
        ),
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_title(
        (
            f"{reference_label}: "
            f"{MONTH_NAMES[SELECTED_MONTH - 1]} "
            f"{X_DAYS}-day precipitation maxima"
        ),
        fontsize=TITLE_FONTSIZE,
        pad=8,
    )

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )

    fig.tight_layout()

    if WRITE_TO_FILE:
        fig.savefig(
            filename_out,
            dpi=300,
            bbox_inches="tight",
        )

        print()
        print(
            "Wrote:",
            filename_out,
        )

    plt.show()


def main():
    """Run the complete observational return-period analysis."""

    validate_user_settings()

    filename_reference = (
        make_reference_filename()
    )

    filename_out = (
        make_figure_filename()
    )

    print(
        f"Reading {get_reference_label()} file:"
    )
    print(
        filename_reference
    )

    # Read the selected calendar month for the GEV analysis and separately
    # read the August 2023 maximum for Storm Hans.
    (
        years,
        all_values,
        hans_value,
    ) = read_reference_data(
        filename_reference
    )

    # Create the GEV-fitting sample. Storm Hans is independent of this step.
    (
        fit_years,
        fit_values,
    ) = create_fit_sample(
        years,
        all_values,
    )

    gev_parameters = (
        fit_gev(
            fit_values
        )
    )

    fitted_return_periods = (
        make_return_period_grid()
    )

    fitted_return_levels = (
        calculate_gev_return_levels(
            fitted_return_periods,
            gev_parameters,
        )
    )

    (
        lower_confidence_limit,
        upper_confidence_limit,
        successful_bootstraps,
    ) = parametric_bootstrap_return_levels(
        observed_values=fit_values,
        fitted_parameters=gev_parameters,
        return_periods=fitted_return_periods,
    )

    (
        empirical_return_periods,
        empirical_values,
    ) = calculate_empirical_return_periods(
        fit_values
    )

    hans_return_period = np.nan

    if hans_value is not None:
        hans_return_period = (
            calculate_event_return_period(
                event_value=hans_value,
                gev_parameters=gev_parameters,
            )
        )

        if not np.isfinite(hans_return_period):
            print()
            print(
                "Warning: the fitted GEV gives an effectively infinite "
                "return period for Storm Hans, so its x-position cannot "
                "be represented on the logarithmic return-period axis."
            )

    print_fit_summary(
        years=fit_years,
        values=fit_values,
        gev_parameters=gev_parameters,
        successful_bootstraps=successful_bootstraps,
        hans_value=hans_value,
        hans_return_period=hans_return_period,
    )

    plot_return_period_curve(
        empirical_return_periods=empirical_return_periods,
        empirical_values=empirical_values,
        fitted_return_periods=fitted_return_periods,
        fitted_return_levels=fitted_return_levels,
        lower_confidence_limit=lower_confidence_limit,
        upper_confidence_limit=upper_confidence_limit,
        hans_value=hans_value,
        hans_return_period=hans_return_period,
        filename_out=filename_out,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    main()
