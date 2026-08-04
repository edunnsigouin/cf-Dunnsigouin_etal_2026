"""
Reproduce the fidelity heatmap shown in debug.ipynb Out[62].

The script reads:

1. A preprocessed reference file created by the reference preprocessing script:
       sipa_preprocessed_<reference_dataset>_<file_id>_<startdate>_<enddate>.nc

2. A preprocessed S2S model file created by the model preprocessing script:
       sipa_preprocessed_s2s_<file_id>_<startdate>_<enddate>.nc

It then reproduces the workflow:

    reference/model loading
        -> X-day accumulation
        -> raw and bias-corrected S2S samples
        -> monthly maxima
        -> bootstrap fidelity tests
        -> make_fidelity_plot

The correction methods are:

    raw
    q corrected
    doy corrected
    ld corrected
    q_doy corrected

The plotted values are the number of calendar months, out of 12, for which
the reference statistic lies inside the 95% bootstrap interval of the model
sample. As in debug.ipynb Out[62], the August 2023 reference maximum ("Hans")
is removed before the plotted fidelity counts are selected.
"""

from __future__ import annotations

import copy
from pathlib import Path

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

# Reference dataset used for bias correction and fidelity testing.
#
# Options expected from the reference preprocessing script:
#     "era5"
#     "senorge"
#     "senorge_regrid"
reference_dataset = "era5"

# Calendar-date range used to create the reference file.
reference_date_range = [
    "2000-01-01",
    "2023-08-10",
]

# Forecast initialization dates used to create the S2S model file.
model_forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

# Catchment used by both preprocessing scripts.
catchment = "regine_drammen"

# Number of days accumulated before calculating correction factors and maxima.
#
# To reproduce debug.ipynb Out[62], use:
#     analysis_x_days = 2
analysis_x_days = 2

# Number of days already accumulated in the reference preprocessing file.
#
# The original debug.ipynb reference file contains daily data, so this should
# normally be 1. The script will only add the remaining accumulation needed
# to reach analysis_x_days.
reference_file_x_days = 1

# Number of days already accumulated in the S2S preprocessing file.
#
# preprocess_s2s.py writes daily data, so this should normally remain 1.
model_file_x_days = 1

# Bootstrap samples per month, correction method, and statistic.
#
# debug.ipynb uses 10,000.
n_bootstrap_samples = 10_000

# Bias-correction settings used by debug.ipynb.
quantile_cutoff = 3
reference_doy_window = 61
model_doy_window = 15
quantile_doy_rolling_window = 15

# Remove the final August reference maximum before selecting the plotted row.
#
# This reproduces the exact "without Hans" behavior used by Out[62].
REMOVE_HANS = True

# Input/Output Directory Containing The Sipa_Preprocessed Files.
path_in = Path(
    config.dirs[
        "sipa_processed"
    ]
)

# Optional explicit input filenames.
#
# Leave as None to generate filenames automatically from the settings above.
# Set either value to a Path when the actual filename differs from the
# automatic naming convention.
reference_filename_override = None
model_filename_override = None

path_out = Path(
    config.dirs[
        "fig"
    ]
)

# Plot output.
write2file = False
show_plot = True

filename_plot = (
    path_out
    / "fidelity_plot.png"
)

# Optional table of fidelity counts.
write_counts_csv = False

filename_counts_csv = (
    path_out
    / "fidelity_counts.csv"
)


# =============================================================================
# Fixed calculation settings
# =============================================================================

CORRECTION_NAMES = [
    "raw",
    "q corrected",
    "doy corrected",
    "ld corrected",
    "q_doy corrected",
]

STATISTICS = {
    "mean": np.mean,
    "std": np.std,
    "skew": st.skew,
    "kurtosis": st.kurtosis,
}

MONTH_LABELS = [
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


# =============================================================================
# Catchment and filename helpers
# =============================================================================

def get_file_id(
    catchment,
):
    """
    Return the short catchment label used in preprocessed filenames.

    Examples:
        regine_drammen -> drammen
        regine_glomma  -> glomma
    """

    if catchment.startswith(
        "regine_"
    ):
        return catchment.replace(
            "regine_",
            "",
            1,
        )

    return catchment


def make_date_label(
    date_range,
):
    """Return the date-range label used by the reference preprocessing script."""

    return (
        f"{date_range[0]}_"
        f"{date_range[1]}"
    )


def make_reference_filename():
    """Create or return the selected reference filename."""

    if reference_filename_override is not None:
        return Path(
            reference_filename_override
        )

    file_id = get_file_id(
        catchment
    )

    date_label = make_date_label(
        reference_date_range
    )

    return (path_in/(f"sipa_preprocessed_{reference_dataset}_{file_id}_{date_label}.nc"))
    #return (path_in/(f"original/sipa_preprocessed_{reference_dataset}_{file_id}.nc"))

def make_model_filename():
    """Create or return the selected S2S model filename."""

    if model_filename_override is not None:
        return Path(
            model_filename_override
        )

    file_id = get_file_id(
        catchment
    )

    return (path_in/(f"sipa_preprocessed_s2s_{file_id}_{model_forecast_date_range[0]}_{model_forecast_date_range[1]}.nc"))
    #return (path_in/(f"original/sipa_preprocessed_s2s_{file_id}.nc")) 

# =============================================================================
# User-setting validation
# =============================================================================

def validate_user_settings():
    """Validate settings and derived input filenames."""

    if analysis_x_days < 1:
        raise ValueError(
            "analysis_x_days must be at least 1."
        )

    if reference_file_x_days < 1:
        raise ValueError(
            "reference_file_x_days must be at least 1."
        )

    if model_file_x_days < 1:
        raise ValueError(
            "model_file_x_days must be at least 1."
        )

    if reference_file_x_days > analysis_x_days:
        raise ValueError(
            "reference_file_x_days cannot exceed analysis_x_days."
        )

    if model_file_x_days > analysis_x_days:
        raise ValueError(
            "model_file_x_days cannot exceed analysis_x_days."
        )

    if n_bootstrap_samples < 1:
        raise ValueError(
            "n_bootstrap_samples must be at least 1."
        )

    if len(
        reference_date_range
    ) != 2:
        raise ValueError(
            "reference_date_range must contain exactly two values: "
            "a start date and an end date."
        )

    if (
        np.datetime64(
            reference_date_range[0]
        )
        >
        np.datetime64(
            reference_date_range[1]
        )
    ):
        raise ValueError(
            "The reference start date must not be later than the end date."
        )

    if (
        np.datetime64(
            model_forecast_date_range[0]
        )
        >
        np.datetime64(
            model_forecast_date_range[1]
        )
    ):
        raise ValueError(
            "The first model forecast date must not be later "
            "than the second."
        )

    filename_reference = (
        make_reference_filename()
    )

    filename_model = (
        make_model_filename()
    )

    if not filename_reference.is_file():
        raise FileNotFoundError(
            f"Reference file not found: {filename_reference}"
        )

    if not filename_model.is_file():
        raise FileNotFoundError(
            f"Model file not found: {filename_model}"
        )


# =============================================================================
# Rolling-window helpers
# =============================================================================

def wrapped_rolling_mean(
    window_size,
    da,
    dim,
):
    """Calculate a centered rolling mean with wraparound at year boundaries."""

    half_window = (
        window_size
        // 2
    )

    wrapped = xr.concat(
        [
            da.isel(
                {
                    dim: slice(
                        -half_window,
                        None,
                    )
                }
            ),
            da,
            da.isel(
                {
                    dim: slice(
                        0,
                        half_window,
                    )
                }
            ),
        ],
        dim=dim,
    )

    rolled = wrapped.rolling(
        {
            dim: window_size
        },
        center=True,
    ).mean()

    return rolled.isel(
        {
            dim: slice(
                half_window,
                -half_window,
            )
        }
    )


def additional_accumulation_days(
    target_days,
    file_days,
):
    """
    Return the rolling window needed to reach target_days.

    The intended workflow uses daily input files, where file_days=1.
    """

    if file_days == target_days:
        return 1

    if file_days != 1:
        raise ValueError(
            "Combining already accumulated overlapping values into a longer "
            "accumulation is ambiguous. Use files with daily values "
            "(file_days=1), or set file_days equal to analysis_x_days."
        )

    return target_days


# =============================================================================
# Quantile-coordinate helpers
# =============================================================================

def add_doy_quantile(
    ds,
    quantiles,
    delta=0,
    reference=False,
):
    """Assign a day-of-year-specific precipitation quantile to every value."""

    if reference:
        data_dims = (
            "date",
        )
    else:
        data_dims = (
            "lead_day",
            "number",
            "i_date",
        )

    quantiles_by_doy = []

    for doy in range(
        1,
        367,
    ):

        subset = ds.where(
            (
                ds.doy
                >= doy - delta
            )
            & (
                ds.doy
                <= doy + delta
            ),
            drop=True,
        )

        quantiles_by_doy.append(
            subset.quantile(
                quantiles
            )
        )

    quantiles_by_doy = xr.concat(
        quantiles_by_doy,
        "doy",
    ).assign_coords(
        doy=np.arange(
            1,
            367,
        )
    )

    if reference:

        thresholds = (
            quantiles_by_doy.tp24[
                ds.doy - 1
            ]
            .T
            .values[
                ::-1,
                :,
            ]
        )

        values = ds.tp24.values[
            np.newaxis,
            :,
        ]

    else:

        thresholds = (
            quantiles_by_doy.tp24[
                ds.doy - 1
            ]
            .T
            .values[
                ::-1,
                :,
                np.newaxis,
                :,
            ]
        )

        thresholds = np.repeat(
            thresholds,
            ds.sizes[
                "number"
            ],
            axis=2,
        )

        values = ds.tp24.values[
            np.newaxis,
            :,
            :,
            :,
        ]

    comparisons = np.repeat(
        values,
        len(
            quantiles
        ),
        axis=0,
    ) >= thresholds

    allocation = (
        len(
            quantiles
        )
        - 1
        - np.argmax(
            comparisons,
            axis=0,
        )
    )

    return ds.assign_coords(
        quantile_doy=(
            data_dims,
            allocation,
        )
    )


def add_global_quantile_reference(
    ds,
    quantiles,
):
    """Assign a global quantile coordinate to reference values."""

    thresholds = np.quantile(
        ds.tp24.values,
        quantiles,
    )

    comparisons = np.repeat(
        ds.tp24.values[
            :,
            np.newaxis,
        ],
        len(
            quantiles
        ),
        axis=1,
    ) >= np.repeat(
        thresholds[
            np.newaxis,
            ::-1,
        ],
        ds.sizes[
            "date"
        ],
        axis=0,
    )

    allocation = (
        len(
            quantiles
        )
        - 1
        - np.argmax(
            comparisons,
            axis=1,
        )
    )

    return ds.assign_coords(
        quantile_global=(
            "date",
            allocation,
        )
    )


def add_global_quantile_model(
    ds,
    quantiles,
):
    """Assign a global quantile coordinate to S2S values."""

    flat_values = ds.tp24.values.ravel()

    thresholds = np.quantile(
        flat_values[
            np.isfinite(
                flat_values
            )
        ],
        quantiles,
    )

    n_lead = ds.sizes[
        "lead_day"
    ]

    n_number = ds.sizes[
        "number"
    ]

    n_date = ds.sizes[
        "i_date"
    ]

    comparisons = np.repeat(
        ds.tp24.values[
            :,
            :,
            :,
            np.newaxis,
        ],
        len(
            quantiles
        ),
        axis=3,
    ) >= np.repeat(
        thresholds[
            np.newaxis,
            np.newaxis,
            np.newaxis,
            ::-1,
        ],
        n_lead,
        axis=0,
    ).repeat(
        n_number,
        axis=1,
    ).repeat(
        n_date,
        axis=2,
    )

    allocation = (
        len(
            quantiles
        )
        - 1
        - np.argmax(
            comparisons,
            axis=3,
        )
    )

    return ds.assign_coords(
        quantile_global=(
            (
                "lead_day",
                "number",
                "i_date",
            ),
            allocation,
        )
    )


# =============================================================================
# Data loading
# =============================================================================

def load_reference(
    filename,
    quantiles,
):
    """Load and prepare the reference dataset."""

    with xr.open_dataset(
        filename
    ) as opened:

        ds = opened.load()

    if "tp24" not in ds:
        raise KeyError(
            f"'tp24' not found in {filename}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if "date" not in ds.dims:
        raise ValueError(
            f"Reference file must contain a 'date' dimension. "
            f"Found dimensions: {ds.dims}"
        )

    rolling_days = additional_accumulation_days(
        target_days=analysis_x_days,
        file_days=reference_file_x_days,
    )

    if rolling_days > 1:

        ds = (
            ds
            .rolling(
                date=rolling_days
            )
            .sum()
            .shift(
                date=-(rolling_days - 1)
            )
            .dropna(
                "date"
            )
        )

    ds = ds.assign_coords(
        month=ds.date.dt.month,
        doy=ds.date.dt.dayofyear,
    )

    ds = add_doy_quantile(
        ds=ds,
        quantiles=quantiles,
        delta=(
            reference_doy_window
            - 1
        )
        // 2,
        reference=True,
    )

    ds = add_global_quantile_reference(
        ds=ds,
        quantiles=quantiles,
    )

    return ds


def load_model(
    filename,
    quantiles,
):
    """Load and prepare the S2S model dataset."""

    with xr.open_dataset(
        filename
    ) as opened:

        ds = opened.load()

    required = {
        "tp24",
        "f_date",
    }

    missing = [
        name
        for name in required
        if name not in ds
    ]

    if missing:
        raise KeyError(
            f"Model file is missing {missing}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    required_dims = {
        "lead_day",
        "number",
        "i_date",
    }

    missing_dims = (
        required_dims
        - set(
            ds.dims
        )
    )

    if missing_dims:
        raise ValueError(
            f"Model file is missing dimensions {sorted(missing_dims)}. "
            f"Found dimensions: {ds.dims}"
        )

    rolling_days = additional_accumulation_days(
        target_days=analysis_x_days,
        file_days=model_file_x_days,
    )

    if rolling_days > 1:

        ds = (
            ds
            .rolling(
                lead_day=rolling_days
            )
            .sum()
            .shift(
                lead_day=-(rolling_days - 1)
            )
        )

    if ds.sizes[
        "lead_day"
    ] <= 15:
        raise ValueError(
            "The model dataset must contain at least 16 lead-day positions "
            "to reproduce the calendar-month assignment in debug.ipynb."
        )

    # Exact debug.ipynb behavior:
    # all samples for one i_date are assigned the month of the valid date at
    # the 16th lead_day position (isel index 15).
    month = (
        ds
        .isel(
            lead_day=15
        )
        .f_date
        .dt
        .month
        .drop_vars(
            [
                "lead_day",
                "f_date",
            ],
            errors="ignore",
        )
    )

    doy = (
        ds
        .f_date
        .dt
        .dayofyear
        .drop_vars(
            [
                "lead_day",
                "f_date",
            ],
            errors="ignore",
        )
    )

    ds = ds.assign_coords(
        month=month,
        doy=doy,
    )

    ds = add_doy_quantile(
        ds=ds,
        quantiles=quantiles,
        delta=0,
        reference=False,
    )

    ds = add_global_quantile_model(
        ds=ds,
        quantiles=quantiles,
    )

    return ds


# =============================================================================
# Bias-correction calculations
# =============================================================================

def mean_by_doy_quantile(
    ds,
    delta=0,
):
    """Calculate mean precipitation by day of year and quantile."""

    quantile_indices = np.unique(
        ds.quantile_doy.values
    )

    quantile_indices.sort()

    quantile_edges = np.hstack(
        [
            quantile_indices
            / len(
                quantile_indices
            ),
            [
                1,
            ],
        ]
    )

    output = []

    for doy in range(
        1,
        367,
    ):

        values = (
            ds
            .where(
                (
                    ds.doy
                    >= doy - delta
                )
                & (
                    ds.doy
                    <= doy + delta
                ),
                drop=True,
            )
            .tp24
            .values
            .ravel()
        )

        values = np.sort(
            values[
                np.isfinite(
                    values
                )
            ]
        )

        n_values = len(
            values
        )

        means = np.array(
            [
                np.mean(
                    values[
                        int(
                            np.floor(
                                quantile_edges[index]
                                * n_values
                            )
                        ):
                        int(
                            np.floor(
                                quantile_edges[
                                    index + 1
                                ]
                                * n_values
                            )
                        )
                    ]
                )
                for index in range(
                    len(
                        quantile_indices
                    )
                )
            ]
        )

        output.append(
            xr.DataArray(
                means,
                dims="quantile_doy",
                coords={
                    "quantile_doy": (
                        quantile_indices
                        / len(
                            quantile_indices
                        )
                    )
                },
            )
        )

    return xr.concat(
        output,
        "doy",
    ).assign_coords(
        doy=np.arange(
            1,
            367,
        )
    )


def correction_factors(
    method,
    reference,
    model,
):
    """Calculate multiplicative correction factors for one method."""

    if method == "raw":
        return 1

    if method == "q":

        factors = (
            reference
            .groupby(
                "quantile_global"
            )
            .mean()
            /
            model
            .groupby(
                "quantile_global"
            )
            .mean()
        )

        factors.tp24[
            -(
                quantile_cutoff
                - 1
            ):
        ] = factors.tp24[
            -quantile_cutoff
        ]

        return (
            factors
            .tp24
            .drop_vars(
                "number",
                errors="ignore",
            )[
                model.quantile_global
            ]
        )

    if method == "doy":

        reference_mean = (
            reference
            .groupby(
                "doy"
            )
            .mean()
            .tp24
        )

        model_mean = (
            model
            .groupby(
                "doy"
            )
            .mean()
            .mean(
                "number"
            )
            .tp24
        )

        reference_rolling = wrapped_rolling_mean(
            window_size=reference_doy_window,
            da=reference_mean,
            dim="doy",
        )

        model_rolling = wrapped_rolling_mean(
            window_size=model_doy_window,
            da=model_mean,
            dim="doy",
        )

        factors = (
            reference_rolling.values
            /
            model_rolling.values
        )

        return np.repeat(
            factors[
                model.doy - 1
            ]
            .T[
                :,
                np.newaxis,
                :,
            ],
            model.sizes[
                "number"
            ],
            axis=1,
        )

    if method == "ld":

        factors = (
            reference
            .mean(
                "date"
            )
            .tp24
            /
            model
            .mean(
                "number"
            )
            .mean(
                "i_date"
            )
            .tp24
        )

        return np.repeat(
            np.repeat(
                factors.values[
                    :,
                    np.newaxis,
                    np.newaxis,
                ],
                model.sizes[
                    "number"
                ],
                axis=1,
            ),
            model.sizes[
                "i_date"
            ],
            axis=2,
        )

    if method == "q_doy":

        reference_quantile_mean = mean_by_doy_quantile(
            ds=reference,
            delta=(
                reference_doy_window
                - 1
            )
            // 2,
        )

        model_quantile_mean = mean_by_doy_quantile(
            ds=model,
            delta=0,
        )

        factors = (
            wrapped_rolling_mean(
                window_size=quantile_doy_rolling_window,
                da=reference_quantile_mean,
                dim="doy",
            )
            /
            wrapped_rolling_mean(
                window_size=quantile_doy_rolling_window,
                da=model_quantile_mean,
                dim="doy",
            )
        )

        factors.values[
            :,
            -(
                quantile_cutoff
                - 1
            ):
        ] = np.repeat(
            factors.values[
                :,
                -quantile_cutoff
            ][
                :,
                np.newaxis,
            ],
            quantile_cutoff
            - 1,
            axis=1,
        )

        return factors[
            model.doy - 1,
            model.quantile_doy,
        ]

    raise ValueError(
        f"Unknown correction method: {method}"
    )


def corrected_monthly_maxima(
    method,
    reference,
    model,
):
    """Apply one correction and take the maximum over lead_day."""

    factors = correction_factors(
        method=method,
        reference=reference,
        model=model,
    )

    return (
        factors
        * model.tp24
    ).max(
        "lead_day"
    )


# =============================================================================
# Monthly samples
# =============================================================================

def split_to_months(
    da,
):
    """Return one finite model-sample array for each calendar month."""

    output = []

    for month in range(
        1,
        13,
    ):

        values = (
            da
            .where(
                da.month
                == month,
                drop=True,
            )
            .values
            .ravel()
        )

        output.append(
            values[
                np.isfinite(
                    values
                )
            ]
        )

    return output


def monthly_reference_maxima(
    reference,
):
    """Return annual reference maxima separately for each calendar month."""

    reference_by_year = (
        reference
        .assign_coords(
            year=reference.date.dt.year
        )
        .groupby(
            [
                "month",
                "year",
            ]
        )
        .max()
    )

    output = []

    for month in range(
        1,
        13,
    ):

        values = (
            reference_by_year
            .sel(
                month=month
            )
            .tp24
            .values
        )

        output.append(
            values[
                np.isfinite(
                    values
                )
            ]
        )

    return output


# =============================================================================
# Bootstrap fidelity tests
# =============================================================================

def bootstrap_interval(
    values,
    sample_size,
    statistic,
    seed,
    n_samples,
):
    """Return the 2.5% and 97.5% bootstrap interval."""

    random_state = np.random.RandomState(
        seed
    )

    sampled = values[
        random_state.randint(
            0,
            len(
                values
            ),
            size=(
                n_samples,
                sample_size,
            ),
        )
    ]

    bootstrap_statistics = statistic(
        sampled,
        axis=1,
    )

    return np.quantile(
        bootstrap_statistics,
        [
            0.025,
            0.975,
        ],
    )


def count_months_within(
    reference_lists,
    model_lists,
    statistic,
    n_samples,
):
    """
    Count passing months and retain month-level diagnostics.

    Diagnostics are stored for every model version and reference series.
    """

    counts = {
        name: np.zeros(
            len(
                reference_lists
            ),
            dtype=int,
        )
        for name in model_lists
    }

    diagnostics = {
        model_name: {
            reference_name: []
            for reference_name in reference_lists
        }
        for model_name in model_lists
    }

    for month in range(
        12
    ):

        sample_size = len(
            next(
                iter(
                    reference_lists.values()
                )
            )[
                month
            ]
        )

        for model_index, (
            model_name,
            monthly_values,
        ) in enumerate(
            model_lists.items()
        ):

            low, high = bootstrap_interval(
                values=monthly_values[
                    month
                ],
                sample_size=sample_size,
                statistic=statistic,
                seed=(
                    month + 1
                )
                * (
                    model_index + 1
                ),
                n_samples=n_samples,
            )

            for reference_index, (
                reference_name,
                reference_values,
            ) in enumerate(
                reference_lists.items()
            ):

                reference_value = float(
                    statistic(
                        reference_values[
                            month
                        ]
                    )
                )

                passed = bool(
                    low
                    <= reference_value
                    <= high
                )

                counts[
                    model_name
                ][
                    reference_index
                ] += int(
                    passed
                )

                diagnostics[
                    model_name
                ][
                    reference_name
                ].append(
                    {
                        "month": month + 1,
                        "reference_value": reference_value,
                        "bootstrap_low": float(
                            low
                        ),
                        "bootstrap_high": float(
                            high
                        ),
                        "passed": passed,
                    }
                )

    return (
        counts,
        diagnostics,
    )


def print_model_results(
    statistic_name,
    diagnostics,
    reference_name,
):
    """
    Print month-by-month fidelity results for every model-data version.
    """

    for model_name in CORRECTION_NAMES:

        monthly_results = diagnostics[
            model_name
        ][
            reference_name
        ]

        print()
        print(
            f"{model_name} monthly results for {statistic_name}"
        )
        print(
            f"Reference: {reference_name}"
        )
        print(
            "-" * 84
        )

        print(
            f"{'Month':<12}"
            f"{'Reference':>14}"
            f"{'Bootstrap low':>16}"
            f"{'Bootstrap high':>17}"
            f"{'Result':>12}"
        )

        print(
            "-" * 84
        )

        for result in monthly_results:

            month_number = int(
                result[
                    "month"
                ]
            )

            status = (
                "PASS"
                if result[
                    "passed"
                ]
                else "FAIL"
            )

            print(
                f"{MONTH_LABELS[month_number - 1]:<12}"
                f"{result['reference_value']:>14.4f}"
                f"{result['bootstrap_low']:>16.4f}"
                f"{result['bootstrap_high']:>17.4f}"
                f"{status:>12}"
            )

        number_passed = sum(
            int(
                result[
                    "passed"
                ]
            )
            for result in monthly_results
        )

        print(
            "-" * 84
        )

        print(
            f"Passed {number_passed} of 12 months; "
            f"failed {12 - number_passed} of 12 months."
        )


def calculate_fidelity(
    reference,
    model,
):
    """Calculate the fidelity-count table plotted in debug.ipynb Out[62]."""

    reference_with_hans = monthly_reference_maxima(
        reference
    )

    reference_without_hans = copy.deepcopy(
        reference_with_hans
    )

    if REMOVE_HANS:

        if len(
            reference_without_hans[
                7
            ]
        ) < 1:
            raise ValueError(
                "The August reference sample is empty; Hans cannot be removed."
            )

        # Exact debug.ipynb behavior: remove the final August maximum.
        reference_without_hans[
            7
        ] = reference_without_hans[
            7
        ][
            :-1
        ]

    references = {
        "Reference with Hans": reference_with_hans,
        "Reference w/o Hans": reference_without_hans,
    }

    corrected_samples = {}

    for label in CORRECTION_NAMES:

        method = label.split(
            " "
        )[0]

        print(
            f"Computing {label} data ...",
            flush=True,
        )

        corrected_samples[
            label
        ] = split_to_months(
            corrected_monthly_maxima(
                method=method,
                reference=reference,
                model=model,
            )
        )

    table = {}

    selected_reference_index = (
        1
        if REMOVE_HANS
        else 0
    )

    for statistic_name, statistic in STATISTICS.items():

        print(
            f"Bootstrapping {statistic_name} "
            f"({n_bootstrap_samples:,} samples) ...",
            flush=True,
        )

        (
            counts,
            diagnostics,
        ) = count_months_within(
            reference_lists=references,
            model_lists=corrected_samples,
            statistic=statistic,
            n_samples=n_bootstrap_samples,
        )

        table[
            statistic_name
        ] = [
            counts[
                label
            ][
                selected_reference_index
            ]
            for label in CORRECTION_NAMES
        ]

        selected_reference_name = list(
            references.keys()
        )[
            selected_reference_index
        ]

        print_model_results(
            statistic_name=statistic_name,
            diagnostics=diagnostics,
            reference_name=selected_reference_name,
        )

    return pd.DataFrame(
        table,
        index=CORRECTION_NAMES,
    )


# =============================================================================
# Plotting
# =============================================================================

def make_fidelity_plot(
    fidelity_test_counts,
    filename=None,
):
    """Reproduce make_fidelity_plot from debug.ipynb."""

    plt.figure()

    image = plt.imshow(
        12
        - fidelity_test_counts.values,
        cmap=colormaps[
            "Blues"
        ],
        vmin=0,
        vmax=6,
    )

    for (
        row,
        column,
    ), value in np.ndenumerate(
        fidelity_test_counts.values
    ):

        plt.text(
            column,
            row,
            int(
                value
            ),
            ha="center",
            va="center",
        )

    image.axes.set_xticks(
        range(
            len(
                fidelity_test_counts.columns
            )
        ),
        fidelity_test_counts.columns,
    )

    image.axes.set_yticks(
        range(
            len(
                fidelity_test_counts.index
            )
        ),
        fidelity_test_counts.index,
    )

    if filename is not None:

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            filename,
            bbox_inches="tight",
            dpi=400,
        )

        print(
            "Wrote:",
            filename,
        )

    if show_plot:
        plt.show()

    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()


    filename_reference = (
        make_reference_filename()
    )

    filename_model = (
        make_model_filename()
    )


    print(
        "Input files"
    )
    print(
        "-----------"
    )
    print(
        "Reference:",
        filename_reference,
    )
    print(
        "Model:    ",
        filename_model,
    )

    print()
    print(
        "Analysis settings"
    )
    print(
        "-----------------"
    )
    print(
        "Reference dataset:",
        reference_dataset,
    )
    print(
        "Catchment:",
        catchment,
    )
    print(
        "Accumulation days:",
        analysis_x_days,
    )
    print(
        "Bootstrap samples:",
        f"{n_bootstrap_samples:,}",
    )
    print(
        "Remove Hans:",
        REMOVE_HANS,
    )


    quantiles = np.arange(
        0,
        0.96,
        0.05,
    )


    reference = load_reference(
        filename=filename_reference,
        quantiles=quantiles,
    )

    model = load_model(
        filename=filename_model,
        quantiles=quantiles,
    )


    fidelity_test_counts = calculate_fidelity(
        reference=reference,
        model=model,
    )


    print()
    print(
        "Fidelity counts "
        "(months out of 12)"
    )
    print(
        "-------------------------"
    )
    print(
        fidelity_test_counts
    )


    if write_counts_csv:

        filename_counts_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fidelity_test_counts.to_csv(
            filename_counts_csv
        )

        print(
            "Wrote:",
            filename_counts_csv,
        )


    filename_fi = (
        filename_plot
        if write2file
        else None
    )

    make_fidelity_plot(
        fidelity_test_counts,
        filename=filename_fi,
    )
