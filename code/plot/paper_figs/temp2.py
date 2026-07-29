"""
Plot a stationary return-period curve for one calendar month of a
reference precipitation-extreme dataset.

The reference dataset can be either SeNorge or ERA5. Both datasets contain
one monthly maximum X-day accumulated precipitation value for each year and
calendar month:

    SeNorge: rr(year, month)
    ERA5:    tp24(year, month)

For the selected calendar month, this script:

1. Reads the reference data over OBSERVATION_YEARS.
2. Optionally excludes 2023 from the sample used to fit the selected distribution.
3. Fits the selected stationary extreme-value distribution by maximum likelihood.
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
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r

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

# Choose the extreme-value distribution to fit.
#
# Options:
#     1 -> GEV
#     2 -> Gumbel
#     3 -> GenEx (two-parameter Generalized Exponential)
EXTREME_VALUE_DISTRIBUTION = 3

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
SELECTED_MONTH = 8

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

    if EXTREME_VALUE_DISTRIBUTION not in {1, 2, 3}:
        raise ValueError(
            "EXTREME_VALUE_DISTRIBUTION must be 1, 2, or 3."
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


def get_distribution_name():
    """Return the name of the selected fitted distribution."""

    return {
        1: "GEV",
        2: "Gumbel",
        3: "GenEx",
    }[EXTREME_VALUE_DISTRIBUTION]


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
        f"return-period-{get_distribution_name().lower()}-"
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

def fit_gev(values):
    """Fit a stationary three-parameter GEV by maximum likelihood."""

    shape_c, location, scale = genextreme.fit(values)

    if (
        not np.isfinite([shape_c, location, scale]).all()
        or scale <= 0
    ):
        raise RuntimeError("The GEV fit returned invalid parameters.")

    return shape_c, location, scale


def fit_gumbel(values):
    """Fit a stationary two-parameter Gumbel distribution by MLE."""

    location, scale = gumbel_r.fit(values)

    if (
        not np.isfinite([location, scale]).all()
        or scale <= 0
    ):
        raise RuntimeError("The Gumbel fit returned invalid parameters.")

    return location, scale


def genex_negative_log_likelihood(log_parameters, values):
    """
    Negative log-likelihood for the two-parameter Generalized Exponential.

    The parameterization used here is:

        F(x) = [1 - exp(-x / scale)] ** shape,  x >= 0

    with shape > 0 and scale > 0.
    """

    shape, scale = np.exp(log_parameters)

    if (
        not np.isfinite(shape)
        or not np.isfinite(scale)
        or shape <= 0
        or scale <= 0
        or np.any(values < 0)
    ):
        return np.inf

    z = values / scale

    # Stable evaluation of log(1 - exp(-z)).
    log_one_minus_exp = np.log(-np.expm1(-z))

    log_pdf = (
        np.log(shape)
        - np.log(scale)
        - z
        + (shape - 1.0) * log_one_minus_exp
    )

    if not np.isfinite(log_pdf).all():
        return np.inf

    return -np.sum(log_pdf)


def fit_genex(values):
    """Fit a two-parameter Generalized Exponential distribution by MLE."""

    if np.any(values < 0):
        raise ValueError(
            "The GenEx implementation requires non-negative values."
        )

    positive_values = values[values > 0]

    if positive_values.size == 0:
        raise RuntimeError(
            "GenEx cannot be fitted because there are no positive values."
        )

    result = minimize(
        genex_negative_log_likelihood,
        x0=np.log([1.0, np.mean(positive_values)]),
        args=(values,),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    if not result.success:
        raise RuntimeError(f"GenEx fit failed: {result.message}")

    shape, scale = np.exp(result.x)

    if (
        not np.isfinite([shape, scale]).all()
        or shape <= 0
        or scale <= 0
    ):
        raise RuntimeError("The GenEx fit returned invalid parameters.")

    return shape, scale


def fit_distribution(values):
    """Fit the distribution selected by EXTREME_VALUE_DISTRIBUTION."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        return fit_gev(values)

    if EXTREME_VALUE_DISTRIBUTION == 2:
        return fit_gumbel(values)

    return fit_genex(values)


def genex_ppf(probabilities, parameters):
    """Quantile function of the two-parameter Generalized Exponential."""

    shape, scale = parameters
    probabilities = np.asarray(probabilities, dtype=float)

    return -scale * np.log1p(
        -np.power(probabilities, 1.0 / shape)
    )


def genex_cdf(values, parameters):
    """CDF of the two-parameter Generalized Exponential."""

    shape, scale = parameters
    values = np.asarray(values, dtype=float)
    cdf = np.zeros_like(values, dtype=float)
    mask = values >= 0
    cdf[mask] = np.power(
        1.0 - np.exp(-values[mask] / scale),
        shape,
    )
    return cdf


def calculate_return_levels(return_periods, fitted_parameters):
    """Calculate fitted return levels for the selected distribution."""

    probabilities = 1.0 - 1.0 / return_periods

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters
        return genextreme.ppf(
            probabilities,
            shape_c,
            loc=location,
            scale=scale,
        )

    if EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters
        return gumbel_r.ppf(
            probabilities,
            loc=location,
            scale=scale,
        )

    return genex_ppf(
        probabilities,
        fitted_parameters,
    )


def generate_random_sample(fitted_parameters, sample_size, rng):
    """Generate one parametric-bootstrap sample from the fitted model."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters
        return genextreme.rvs(
            shape_c,
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    if EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters
        return gumbel_r.rvs(
            loc=location,
            scale=scale,
            size=sample_size,
            random_state=rng,
        )

    uniforms = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=sample_size,
    )

    return genex_ppf(uniforms, fitted_parameters)


def make_return_period_grid():
    """Create logarithmically spaced return periods for the fitted curve."""

    return np.geomspace(
        MIN_RETURN_PERIOD,
        MAX_RETURN_PERIOD,
        NUMBER_OF_RETURN_PERIODS,
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
    fitted_parameters,
):
    """Calculate the fitted return period corresponding to one event."""

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters
        exceedance_probability = genextreme.sf(
            event_value,
            shape_c,
            loc=location,
            scale=scale,
        )

    elif EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters
        exceedance_probability = gumbel_r.sf(
            event_value,
            loc=location,
            scale=scale,
        )

    else:
        cdf = float(
            genex_cdf(
                np.array([event_value]),
                fitted_parameters,
            )[0]
        )
        exceedance_probability = 1.0 - cdf

    if (
        not np.isfinite(exceedance_probability)
        or exceedance_probability <= 0
    ):
        return np.inf

    return 1.0 / exceedance_probability


def parametric_bootstrap_return_levels(
    observed_values,
    fitted_parameters,
    return_periods,
):
    """
    Estimate uncertainty in the fitted return-level curve.

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

    successful_fits = 0

    for bootstrap_number in range(
        NUMBER_OF_BOOTSTRAPS
    ):

        simulated_values = generate_random_sample(
            fitted_parameters=fitted_parameters,
            sample_size=sample_size,
            rng=rng,
        )

        try:
            bootstrap_parameters = fit_distribution(
                simulated_values
            )

            bootstrap_levels = (
                calculate_return_levels(
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
            f"{NUMBER_OF_BOOTSTRAPS} bootstrap fits succeeded."
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
    fitted_parameters,
    successful_bootstraps,
    hans_value,
    hans_return_period,
):
    """Print a summary of the selected distribution fit."""

    distribution_name = get_distribution_name()

    print()
    print(f"Reference {distribution_name} fit")
    print("------------------------")
    print(f"Dataset:          {get_reference_label()}")
    print(
        f"Calendar month:   "
        f"{MONTH_NAMES[SELECTED_MONTH - 1]} "
        f"({SELECTED_MONTH})"
    )
    print(
        f"Years used:       "
        f"{int(years.min())}-{int(years.max())}"
    )
    print(f"Sample size:      {values.size}")

    if EXTREME_VALUE_DISTRIBUTION == 1:
        shape_c, location, scale = fitted_parameters
        print(f"Location, mu:     {location:.3f} mm")
        print(f"Scale, sigma:     {scale:.3f} mm")
        print(f"Shape, xi:        {-shape_c:.4f}")
        print(f"SciPy shape, c:   {shape_c:.4f}")

    elif EXTREME_VALUE_DISTRIBUTION == 2:
        location, scale = fitted_parameters
        print(f"Location:         {location:.3f} mm")
        print(f"Scale:            {scale:.3f} mm")

    else:
        shape, scale = fitted_parameters
        print(f"Shape:            {shape:.4f}")
        print(f"Scale:            {scale:.3f} mm")

    print(
        f"Bootstrap fits:   "
        f"{successful_bootstraps}/{NUMBER_OF_BOOTSTRAPS}"
    )

    if hans_value is not None:
        print(f"Storm Hans value: {hans_value:.3f} mm")
        print(f"Hans fitted RP:   {hans_return_period:.2f} years")


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
    """Plot the fitted distribution, uncertainty, observations, and Hans."""

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
        label=f"Fitted stationary {get_distribution_name()}",
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
        MIN_RETURN_PERIOD,
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
            f"{X_DAYS}-day precipitation maxima\n"
            f"{get_distribution_name()} fit"
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

    print()
    print(
        f"Selected distribution: "
        f"{get_distribution_name()}"
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

    fitted_parameters = (
        fit_distribution(
            fit_values
        )
    )

    fitted_return_periods = (
        make_return_period_grid()
    )

    fitted_return_levels = (
        calculate_return_levels(
            fitted_return_periods,
            fitted_parameters,
        )
    )

    (
        lower_confidence_limit,
        upper_confidence_limit,
        successful_bootstraps,
    ) = parametric_bootstrap_return_levels(
        observed_values=fit_values,
        fitted_parameters=fitted_parameters,
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
                fitted_parameters=fitted_parameters,
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
        fitted_parameters=fitted_parameters,
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
