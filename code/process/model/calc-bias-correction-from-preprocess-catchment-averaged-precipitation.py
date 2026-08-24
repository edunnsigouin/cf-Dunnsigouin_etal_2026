"""
Create four bias-corrected versions of a preprocessed ECMWF S2S dataset.

The input model file is the combined output from the S2S preprocessing script:

    preprocessed_model_tp24_<catchment>_<start>_<end>.nc

Expected model structure
------------------------
dimensions:
    lead_day
    number
    i_date

variables:
    tp24(lead_day, number, i_date)
    f_date(i_date, lead_day)
    number(number)
    lead_day(lead_day)
    i_date(i_date)
    model_type(i_date)
    hdate(i_date)

The script reads daily reference datasets:

    ERA5:
        t_tp24_1dayacc_<catchment>_era5_<grid>_<years>.nc

    SeNorge:
        t_rr_1dayacc_<catchment>_senorge_<years>.nc

The reference files are converted internally to the common structure:

    tp24(date)

before calculating four multiplicative bias corrections:

    q
        Global quantile-dependent correction.

    doy
        Day-of-year-dependent correction.

    ld
        Lead-day-dependent correction.

    q_doy
        Day-of-year- and quantile-dependent correction.

No fidelity tests, bootstrap calculations, or plots are produced.

Accumulation
------------
The correction calculations follow the original bias-correction script.

Both the reference and model inputs are daily. The single user setting
analysis_x_days controls the rolling accumulation applied to both datasets.

For example:

    analysis_x_days = 1
        use the daily values directly

    analysis_x_days = 2
        calculate two-day rolling accumulations for both reference and model

    analysis_x_days = 3
        calculate three-day rolling accumulations for both reference and model

Output
------
Each correction method is written to its own NetCDF file:

    preprocessed_model_tp24_<catchment>_<start>_<end>_bc_q_<reference>.nc
    preprocessed_model_tp24_<catchment>_<start>_<end>_bc_doy_<reference>.nc
    preprocessed_model_tp24_<catchment>_<start>_<end>_bc_ld_<reference>.nc
    preprocessed_model_tp24_<catchment>_<start>_<end>_bc_q_doy_<reference>.nc

Every output retains the model coordinates and provenance variables and
contains the bias-corrected tp24 field.

In addition, each output contains ONLY the lookup table relevant to that
correction method:

    q:
        model_mean_lookup(quantile_global)
        reference_mean_lookup(quantile_global)
        bias_correction_factor_lookup(quantile_global)
        bias_correction_factor(lead_day, number, i_date)

    doy:
        model_mean_lookup(doy)
        reference_mean_lookup(doy)
        bias_correction_factor_lookup(doy)
        bias_correction_factor(lead_day, number, i_date)

    ld:
        model_mean_lookup(lead_day)
        reference_mean
        bias_correction_factor_lookup(lead_day)
        bias_correction_factor(lead_day, number, i_date)

    q_doy:
        model_mean_lookup(doy, quantile_doy)
        reference_mean_lookup(doy, quantile_doy)
        bias_correction_factor_lookup(doy, quantile_doy)
        bias_correction_factor(lead_day, number, i_date)

The full-size bias_correction_factor is the lookup-table factor mapped onto
each model value. Helper coordinates such as doy, quantile_doy and
quantile_global are not retained on the full model dimensions in the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Reference dataset used to calculate correction factors.
#
# Options:
#     "era5"
#     "senorge"
reference_dataset = "senorge"

reference_years = [
    "1957",
    "2022",
]

era5_grid = "0.5x0.5"

model_forecast_date_range = [
    "2020-01-02",
    "2023-12-28",
]

catchment = "regine_drammen"

# Target accumulation used when calculating and applying the corrections.
analysis_x_days = 2

# Bias-correction settings retained from the original fidelity script.
quantile_cutoff = 3
reference_doy_window = 61
model_doy_window = 15
quantile_doy_rolling_window = 15

# Optional explicit input filenames.
#
# Leave as None to construct filenames automatically.
reference_filename_override = None
model_filename_override = None

write2file = True


# =============================================================================
# Fixed settings
# =============================================================================

MODEL_VARIABLE = "tp24"

ERA5_VARIABLE = "tp24"
SENORGE_VARIABLE = "rr"

CORRECTION_METHODS = (
    "q",
    "doy",
    "ld",
    "q_doy",
)

HELPER_COORDINATES = (
    "month",
    "doy",
    "quantile_global",
    "quantile_doy",
)


# =============================================================================
# Paths and filenames
# =============================================================================

path_era5_reference = Path(
    config.dirs[
        "era5_processed"
    ]
)

path_senorge_reference = Path(
    config.dirs[
        "senorge_processed"
    ]
)

path_model = Path(
    config.dirs[
        "s2s_processed"
    ]
)

path_out = Path(
    config.dirs[
        "s2s_processed"
    ]
)


def get_file_id(
    catchment_name,
):
    """Return the short catchment label used in filenames."""

    if catchment_name.startswith(
        "regine_"
    ):
        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


def make_reference_filename():
    """Return the selected already accumulated daily reference filename."""

    if reference_filename_override is not None:
        return Path(
            reference_filename_override
        )

    if reference_dataset == "era5":

        return (
            path_era5_reference
            / (
                f"t_{ERA5_VARIABLE}_"
                "1dayacc_"
                f"{catchment}_"
                f"era5_{era5_grid}_"
                f"{reference_years[0]}-"
                f"{reference_years[1]}.nc"
            )
        )

    if reference_dataset == "senorge":

        return (
            path_senorge_reference
            / (
                f"t_{SENORGE_VARIABLE}_"
                "1dayacc_"
                f"{catchment}_"
                f"senorge_"
                f"{reference_years[0]}-"
                f"{reference_years[1]}.nc"
            )
        )

    raise ValueError(
        "reference_dataset must be 'era5' or 'senorge'."
    )


def get_reference_variable():
    """Return the precipitation variable stored in the reference file."""

    if reference_dataset == "era5":
        return ERA5_VARIABLE

    if reference_dataset == "senorge":
        return SENORGE_VARIABLE

    raise ValueError(
        "reference_dataset must be 'era5' or 'senorge'."
    )


def make_model_filename():
    """Return the preprocessed combined S2S model filename."""

    if model_filename_override is not None:
        return Path(
            model_filename_override
        )

    return (
        path_model
        / (
            f"preprocessed_model_"
            f"{MODEL_VARIABLE}_"
            f"{get_file_id(catchment)}_"
            f"{model_forecast_date_range[0]}_"
            f"{model_forecast_date_range[1]}.nc"
        )
    )



def make_output_filename(
    model_filename,
    method,
):
    """Return the output filename including the analysis accumulation."""

    stem = model_filename.stem

    suffix = "_bc_"
    if suffix in stem:
        stem = stem.split(suffix)[0]

    return model_filename.with_name(
        (
            f"{stem}_"
            f"{analysis_x_days}dayacc_"
            f"bc_{method}_"
            f"{reference_dataset}"
            f"{model_filename.suffix}"
        )
    )

# =============================================================================
# Validation
# =============================================================================

def validate_user_settings():
    """Validate settings and required files."""

    valid_reference_datasets = {
        "era5",
        "senorge",
    }

    if reference_dataset not in valid_reference_datasets:
        raise ValueError(
            f"reference_dataset must be one of "
            f"{sorted(valid_reference_datasets)}."
        )

    if analysis_x_days < 1:
        raise ValueError(
            "analysis_x_days must be at least 1."
        )

    if quantile_cutoff < 1:
        raise ValueError(
            "quantile_cutoff must be at least 1."
        )

    for name, value in {
        "reference_doy_window": reference_doy_window,
        "model_doy_window": model_doy_window,
        "quantile_doy_rolling_window": quantile_doy_rolling_window,
    }.items():

        if value < 1:
            raise ValueError(
                f"{name} must be at least 1."
            )

        if value % 2 == 0:
            raise ValueError(
                f"{name} must be odd."
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
    """Calculate a centered rolling mean with wraparound."""

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


# =============================================================================
# Quantile-coordinate helpers
# =============================================================================

def add_doy_quantile(
    ds,
    quantiles,
    delta=0,
    reference=False,
):
    """Assign a day-of-year-specific quantile index to every value."""

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
            subset[
                MODEL_VARIABLE
            ].quantile(
                quantiles
            ).to_dataset(
                name=MODEL_VARIABLE
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
            quantiles_by_doy[
                MODEL_VARIABLE
            ][
                ds.doy - 1
            ]
            .T
            .values[
                ::-1,
                :,
            ]
        )

        values = ds[
            MODEL_VARIABLE
        ].values[
            np.newaxis,
            :,
        ]

    else:

        thresholds = (
            quantiles_by_doy[
                MODEL_VARIABLE
            ][
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

        values = ds[
            MODEL_VARIABLE
        ].values[
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
    """Assign a global quantile index to each reference value."""

    thresholds = np.quantile(
        ds[
            MODEL_VARIABLE
        ].values,
        quantiles,
    )

    comparisons = np.repeat(
        ds[
            MODEL_VARIABLE
        ].values[
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
    """Assign a global quantile index to each model value."""

    flat_values = ds[
        MODEL_VARIABLE
    ].values.ravel()

    finite_values = flat_values[
        np.isfinite(
            flat_values
        )
    ]

    if finite_values.size == 0:
        raise ValueError(
            "The model contains no finite precipitation values."
        )

    thresholds = np.quantile(
        finite_values,
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
        ds[
            MODEL_VARIABLE
        ].values[
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
# Input preparation
# =============================================================================

def load_reference(
    filename,
    quantiles,
):
    """
    Load and prepare an already accumulated ERA5 or SeNorge reference file.

    ERA5 input:
        tp24(time)

    SeNorge input:
        rr(time)

    Internal format:
        tp24(date)
    """

    reference_variable = get_reference_variable()

    with xr.open_dataset(
        filename
    ) as opened:

        ds = opened.load()

    if reference_variable not in ds:
        raise KeyError(
            f"Reference variable '{reference_variable}' was not found in "
            f"{filename}. Available variables: {list(ds.data_vars)}"
        )

    if "time" not in ds.dims:
        raise ValueError(
            "The reference dataset must contain a 'time' dimension."
        )

    rename_mapping = {
        "time": "date",
    }

    if reference_variable != MODEL_VARIABLE:
        rename_mapping[
            reference_variable
        ] = MODEL_VARIABLE

    ds = ds.rename(
        rename_mapping
    )

    # Retain only precipitation so string or metadata variables cannot enter
    # quantile and grouped-mean calculations.
    ds = ds[
        [
            MODEL_VARIABLE,
        ]
    ]

    ds = ds.sortby(
        "date"
    )

    # The reference file contains daily precipitation. Calculate the selected
    # N-day accumulation and keep the coordinate on the ENDING date.
    if analysis_x_days > 1:

        ds[
            MODEL_VARIABLE
        ] = (
            ds[
                MODEL_VARIABLE
            ]
            .rolling(
                date=analysis_x_days,
                min_periods=analysis_x_days,
            )
            .sum()
        )

        ds = ds.dropna(
            dim="date",
            subset=[
                MODEL_VARIABLE,
            ],
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


def ensure_model_dimension_order(
    ds,
):
    """Transpose model variables to the order used by the correction code."""

    ds = ds.copy()

    ds[
        MODEL_VARIABLE
    ] = ds[
        MODEL_VARIABLE
    ].transpose(
        "lead_day",
        "number",
        "i_date",
    )

    if "f_date" not in ds:
        raise KeyError(
            "The model dataset must contain f_date."
        )

    if set(
        ds[
            "f_date"
        ].dims
    ) == {
        "lead_day",
        "i_date",
    }:

        ds[
            "f_date"
        ] = ds[
            "f_date"
        ].transpose(
            "i_date",
            "lead_day",
        )

    elif ds[
        "f_date"
    ].dims == (
        "lead_day",
    ):

        ds[
            "f_date"
        ] = ds[
            "f_date"
        ].broadcast_like(
            ds[
                MODEL_VARIABLE
            ].isel(
                number=0,
                drop=True,
            ).transpose(
                "i_date",
                "lead_day",
            )
        )

    else:

        raise ValueError(
            "f_date must contain dimensions (i_date, lead_day), "
            "(lead_day, i_date), or (lead_day,)."
        )

    return ds


def load_model(
    filename,
    quantiles,
):
    """Load and prepare the new preprocessed S2S model dataset."""

    with xr.open_dataset(
        filename,
        decode_timedelta=False,
    ) as opened:

        ds = opened.load()

    required_dimensions = {
        "lead_day",
        "number",
        "i_date",
    }

    missing_dimensions = (
        required_dimensions
        - set(
            ds.dims
        )
    )

    if missing_dimensions:
        raise ValueError(
            f"Model file is missing dimensions "
            f"{sorted(missing_dimensions)}."
        )

    if MODEL_VARIABLE not in ds:
        raise KeyError(
            f"Model file does not contain '{MODEL_VARIABLE}'."
        )

    ds = ensure_model_dimension_order(
        ds
    )

    # The model file contains daily precipitation. Calculate the selected
    # N-day accumulation and keep the coordinate on the ENDING lead day.
    #
    # Example for analysis_x_days = 2:
    #     lead 17 = daily lead 16 + daily lead 17
    if analysis_x_days > 1:

        ds[
            MODEL_VARIABLE
        ] = (
            ds[
                MODEL_VARIABLE
            ]
            .rolling(
                lead_day=analysis_x_days,
                min_periods=analysis_x_days,
            )
            .sum()
        )

    doy = (
        ds[
            "f_date"
        ]
        .dt
        .dayofyear
    )

    ds = ds.assign_coords(
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
            )[
                MODEL_VARIABLE
            ]
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

        number_of_values = len(
            values
        )

        means = np.array(
            [
                np.mean(
                    values[
                        int(
                            np.floor(
                                quantile_edges[
                                    index
                                ]
                                * number_of_values
                            )
                        ):
                        int(
                            np.floor(
                                quantile_edges[
                                    index + 1
                                ]
                                * number_of_values
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


def correction_lookup_tables(
    method,
    reference,
    model,
):
    """
    Calculate compact lookup tables plus the mapped model-sized factors.

    Returns
    -------
    model_mean
        Model lookup-table mean before broadcasting.
    reference_mean
        Reference lookup-table mean before broadcasting.
    factor_lookup
        Compact multiplicative factor before broadcasting.
    mapped_factors
        Compact factors mapped onto every model-data position.
    """

    if method == "q":

        reference_mean = (
            reference[
                MODEL_VARIABLE
            ]
            .groupby(
                "quantile_global"
            )
            .mean()
        )

        model_mean = (
            model[
                MODEL_VARIABLE
            ]
            .groupby(
                "quantile_global"
            )
            .mean()
        )

        reference_mean = reference_mean.drop_vars(
            [
                "number",
                "lead_day",
                "i_date",
            ],
            errors="ignore",
        )

        model_mean = model_mean.drop_vars(
            [
                "number",
                "lead_day",
                "i_date",
            ],
            errors="ignore",
        )

        factor_lookup = (
            reference_mean
            / model_mean
        )

        factor_lookup[
            -(
                quantile_cutoff
                - 1
            ):
        ] = factor_lookup[
            -quantile_cutoff
        ]

        mapped_factors = factor_lookup.isel(
            quantile_global=model.quantile_global
        )

        return (
            model_mean,
            reference_mean,
            factor_lookup,
            mapped_factors,
        )

    if method == "doy":

        reference_mean_unsmoothed = (
            reference[
                MODEL_VARIABLE
            ]
            .groupby(
                "doy"
            )
            .mean()
        )

        model_mean_unsmoothed = (
            model[
                MODEL_VARIABLE
            ]
            .groupby(
                "doy"
            )
            .mean()
            .mean(
                "number"
            )
        )

        reference_mean = wrapped_rolling_mean(
            window_size=reference_doy_window,
            da=reference_mean_unsmoothed,
            dim="doy",
        )

        model_mean = wrapped_rolling_mean(
            window_size=model_doy_window,
            da=model_mean_unsmoothed,
            dim="doy",
        )

        factor_lookup = (
            reference_mean
            / model_mean
        )

        mapped_factors = np.repeat(
            factor_lookup.values[
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

        return (
            model_mean,
            reference_mean,
            factor_lookup,
            mapped_factors,
        )

    if method == "ld":

        reference_mean = (
            reference[
                MODEL_VARIABLE
            ]
            .mean(
                "date"
            )
        )

        model_mean = (
            model[
                MODEL_VARIABLE
            ]
            .mean(
                "number"
            )
            .mean(
                "i_date"
            )
        )

        factor_lookup = (
            reference_mean
            / model_mean
        )

        mapped_factors = np.repeat(
            np.repeat(
                factor_lookup.values[
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

        return (
            model_mean,
            reference_mean,
            factor_lookup,
            mapped_factors,
        )

    if method == "q_doy":

        reference_mean_raw = mean_by_doy_quantile(
            ds=reference,
            delta=(
                reference_doy_window
                - 1
            )
            // 2,
        )

        model_mean_raw = mean_by_doy_quantile(
            ds=model,
            delta=0,
        )

        reference_mean = wrapped_rolling_mean(
            window_size=quantile_doy_rolling_window,
            da=reference_mean_raw,
            dim="doy",
        )

        model_mean = wrapped_rolling_mean(
            window_size=quantile_doy_rolling_window,
            da=model_mean_raw,
            dim="doy",
        )

        factor_lookup = (
            reference_mean
            / model_mean
        )

        factor_lookup.values[
            :,
            -(
                quantile_cutoff
                - 1
            ):
        ] = np.repeat(
            factor_lookup.values[
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

        mapped_factors = factor_lookup[
            model.doy - 1,
            model.quantile_doy,
        ]

        return (
            model_mean,
            reference_mean,
            factor_lookup,
            mapped_factors,
        )

    raise ValueError(
        f"Unknown correction method: {method}"
    )


def factors_as_data_array(
    factors,
    model,
):
    """
    Return full factors with exactly the model tp24 dimension order.

    Some xarray vectorized indexing operations, especially q_doy, can return
    the correct dimensions in a different order. Reorder by dimension name
    before converting to NumPy.
    """

    template = model[
        MODEL_VARIABLE
    ]

    if isinstance(
        factors,
        xr.DataArray,
    ):

        expected_dims = tuple(
            template.dims
        )

        factor_dims = tuple(
            factors.dims
        )

        if set(
            factor_dims
        ) == set(
            expected_dims
        ):

            factors = factors.transpose(
                *expected_dims
            )

        elif factor_dims != expected_dims:

            raise ValueError(
                "Correction-factor dimensions do not match model dimensions. "
                f"Factors: {factor_dims}; model: {expected_dims}."
            )

        values = np.asarray(
            factors.values
        )

    else:

        values = np.asarray(
            factors
        )

    if values.shape != template.shape:
        raise ValueError(
            "Correction-factor shape does not match model shape after "
            "dimension reordering. "
            f"Factors: {values.shape}; model: {template.shape}."
        )

    factor_da = xr.DataArray(
        values,
        dims=template.dims,
        coords={
            "lead_day": model[
                "lead_day"
            ],
            "number": model[
                "number"
            ],
            "i_date": model[
                "i_date"
            ],
        },
        name="bias_correction_factor",
    )

    factor_da.attrs = {
        "description": (
            "Multiplicative bias-correction factor mapped onto every "
            "model-data position"
        ),
        "units": "1",
        "broadcasted_from_lookup_table": "true",
    }

    return factor_da


# =============================================================================
# Output construction
# =============================================================================

def remove_helper_coordinates(
    ds,
    method,
):
    """
    Remove helper coordinates that are not needed in the selected output.

    Retained helper dimensions by method:
        q      -> quantile_global
        doy    -> doy
        ld     -> none
        q_doy  -> doy and quantile_doy
    """

    keep_by_method = {
        "q": {
            "quantile_global",
        },
        "doy": {
            "doy",
        },
        "ld": set(),
        "q_doy": {
            "doy",
            "quantile_doy",
        },
    }

    if method not in keep_by_method:
        raise ValueError(
            f"Unknown correction method: {method}"
        )

    output = ds

    keep = keep_by_method[
        method
    ]

    for coordinate in HELPER_COORDINATES:

        if (
            coordinate in output.coords
            and coordinate not in keep
        ):

            output = output.drop_vars(
                coordinate
            )

    return output


def build_corrected_dataset(
    model,
    method,
    factor_da,
    model_mean,
    reference_mean,
    factor_lookup,
):
    """
    Build one corrected dataset with method-specific compact lookup tables.
    """

    corrected = (
        model[
            MODEL_VARIABLE
        ]
        * factor_da
    ).transpose(
        "lead_day",
        "number",
        "i_date",
    )

    output = remove_helper_coordinates(
        model.copy(
            deep=True
        ),
        method=method,
    )

    output[
        MODEL_VARIABLE
    ] = corrected.astype(
        "float32"
    )

    output[
        "bias_correction_factor"
    ] = factor_da.astype(
        "float32"
    )

    model_mean_out = model_mean.copy(
        deep=True
    )
    reference_mean_out = reference_mean.copy(
        deep=True
    )
    factor_lookup_out = factor_lookup.copy(
        deep=True
    )

    model_mean_out.name = "model_mean_lookup"
    reference_mean_out.name = "reference_mean_lookup"
    factor_lookup_out.name = (
        "bias_correction_factor_lookup"
    )

    model_mean_out.attrs = {
        "description": (
            f"Model mean used in the compact {method} bias-correction "
            "lookup table"
        ),
        "units": "mm",
    }

    reference_mean_out.attrs = {
        "description": (
            f"Reference mean used in the compact {method} bias-correction "
            "lookup table"
        ),
        "units": "mm",
        "reference_dataset": reference_dataset,
    }

    factor_lookup_out.attrs = {
        "description": (
            f"Compact multiplicative {method} bias-correction factor "
            "before mapping to model dimensions"
        ),
        "formula": "reference_mean_lookup / model_mean_lookup",
        "units": "1",
    }

    output[
        "model_mean_lookup"
    ] = model_mean_out.astype(
        "float32"
    )

    output[
        "reference_mean_lookup"
    ] = reference_mean_out.astype(
        "float32"
    )

    output[
        "bias_correction_factor_lookup"
    ] = factor_lookup_out.astype(
        "float32"
    )

    output[
        MODEL_VARIABLE
    ].attrs.update(
        {
            "units": "mm",
            "description": (
                f"{analysis_x_days}-day accumulated catchment-average "
                f"precipitation corrected using the {method} method"
            ),
            "bias_correction_method": method,
            "bias_correction_reference_dataset": reference_dataset,
        }
    )

    output.attrs.update(
        {
            "description": (
                "Bias-corrected ECMWF S2S catchment-average precipitation"
            ),
            "bias_correction_method": method,
            "bias_correction_reference_dataset": reference_dataset,
            "bias_correction_reference_year_start": reference_years[0],
            "bias_correction_reference_year_end": reference_years[1],
            "bias_correction_reference_file": str(
                make_reference_filename()
            ),
            "analysis_accumulation_days": analysis_x_days,
            "input_reference_accumulation_days": 1,
            "reference_variable_in_source_file": get_reference_variable(),
            "input_model_accumulation_days": 1,
            "quantile_cutoff": quantile_cutoff,
            "reference_doy_window": reference_doy_window,
            "model_doy_window": model_doy_window,
            "quantile_doy_rolling_window": quantile_doy_rolling_window,
            "lookup_table_variables": (
                "model_mean_lookup, reference_mean_lookup, "
                "bias_correction_factor_lookup"
            ),
            "broadcast_factor_variable": (
                "bias_correction_factor"
            ),
        }
    )

    return output


def write_corrected_dataset(
    ds,
    filename,
):
    """Write one corrected NetCDF file."""

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {}

    for variable_name in (
        MODEL_VARIABLE,
        "model_mean_lookup",
        "reference_mean_lookup",
        "bias_correction_factor_lookup",
        "bias_correction_factor",
    ):

        if variable_name in ds:

            encoding[
                variable_name
            ] = {
                "dtype": "float32",
                "_FillValue": np.float32(
                    np.nan
                ),
                "zlib": True,
                "complevel": 4,
            }

    if "hdate" in ds:

        encoding[
            "hdate"
        ] = {
            "dtype": "int64",
        }

    ds.to_netcdf(
        filename,
        encoding=encoding,
    )

    print(
        "Wrote:",
        filename,
    )


# =============================================================================
# Reporting
# =============================================================================

def print_output_summary(
    ds,
    method,
):
    """Print a concise summary for one corrected dataset."""

    values = ds[
        MODEL_VARIABLE
    ].values

    finite_count = int(
        np.isfinite(
            values
        ).sum()
    )

    print()
    print(
        f"{method} correction"
    )
    print(
        "-" * (
            len(
                method
            )
            + 11
        )
    )
    print(
        "Dimensions:",
        dict(
            ds.sizes
        ),
    )
    print(
        "Finite corrected values:",
        finite_count,
    )
    print(
        "Lookup model_mean dimensions:",
        ds[
            "model_mean_lookup"
        ].dims,
    )
    print(
        "Lookup reference_mean dimensions:",
        ds[
            "reference_mean_lookup"
        ].dims,
    )
    print(
        "Lookup factor dimensions:",
        ds[
            "bias_correction_factor_lookup"
        ].dims,
    )
    print(
        "Broadcast factor dimensions:",
        ds[
            "bias_correction_factor"
        ].dims,
    )
    print(
        "Minimum:",
        float(
            np.nanmin(
                values
            )
        ),
    )
    print(
        "Maximum:",
        float(
            np.nanmax(
                values
            )
        ),
    )


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
        "Correction settings"
    )
    print(
        "-------------------"
    )
    print(
        "Reference dataset:",
        reference_dataset,
    )
    print(
        "Analysis accumulation:",
        analysis_x_days,
    )
    print(
        "Reference input accumulation:",
        "1 day",
    )
    print(
        "Model input accumulation:",
        "1 day",
    )
    print(
        "Methods:",
        CORRECTION_METHODS,
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

    try:

        for method in CORRECTION_METHODS:

            print()
            print(
                f"Calculating {method} correction ...",
                flush=True,
            )

            (
                model_mean,
                reference_mean,
                factor_lookup,
                mapped_factors,
            ) = correction_lookup_tables(
                method=method,
                reference=reference,
                model=model,
            )

            factor_da = factors_as_data_array(
                factors=mapped_factors,
                model=model,
            )

            output = build_corrected_dataset(
                model=model,
                method=method,
                factor_da=factor_da,
                model_mean=model_mean,
                reference_mean=reference_mean,
                factor_lookup=factor_lookup,
            )

            print_output_summary(
                ds=output,
                method=method,
            )

            if write2file:

                filename_output = (
                    make_output_filename(
                        model_filename=filename_model,
                        method=method,
                    )
                )

                write_corrected_dataset(
                    ds=output,
                    filename=filename_output,
                )

            output.close()

    finally:

        reference.close()
        model.close()
