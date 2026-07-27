"""
Calculate ensemble-member independence from a precomputed monthly S2S
extreme-sample NetCDF file.

This script does NOT reopen the original daily forecast and hindcast files.
Instead, it reads the complete usable lead-window maxima and their supporting
metadata from the monthly extreme-sample file.

For every calendar month:

Forecasts
---------
1. Select full-window maxima with model_type == "forecast".
2. Reconstruct each forecast initialization using forecast_date.
3. Place each maximum into its ensemble-member column using ensemble_member.
4. Calculate Spearman rank correlation for every unique pair of forecast
   ensemble members across forecast initializations assigned to that month.

Hindcasts
---------
1. Select full-window maxima with model_type == "hindcast".
2. Reconstruct each hindcast initialization using the combination of
   forecast_date and hdate. The same hdate can occur in different reforecast
   cycles, so hdate alone is not a unique initialization identifier.
3. Place each maximum into its ensemble-member column using ensemble_member.
4. Calculate Spearman rank correlation for every unique pair of hindcast
   ensemble members across hindcast initializations assigned to that month.

Only the complete-window maximum variable is used. Lead-bin subsamples such as
max_value_lead17_31 and max_value_lead32_46 are intentionally ignored.

For example, with:
    first_input_lead = 16
    last_input_lead = 46
    x_days = 2

the input maximum variable is:
    max_value_lead17_46

Output variables
----------------
forecast_spearman_rho(month_of_year, forecast_pair)
hindcast_spearman_rho(month_of_year, hindcast_pair)

The correlation calculation matches the original independence script:
missing values are removed pairwise, at least `minimum_samples` paired
initializations are required, and constant member series return NaN.

Implementation note:
Forecast initialization keys are kept as their original numpy.datetime64
objects when constructing the row lookup. They are not converted through
.tolist(), because that conversion can change the datetime key representation
and cause valid forecast dates to fail dictionary lookup.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"
x_days = 2
catchment = "regine_drammen"

forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

first_input_lead = 16
last_input_lead = 46

# This is needed only to construct the input filename produced by the
# monthly-extreme sample-building script.
number_of_lead_bins = 2

# Minimum number of paired initialization values required for a correlation.
minimum_samples = 10

# Expected ensemble sizes. These are checked against the member labels found
# in the input file.
n_forecast_members = 51
n_hindcast_members = 11

write_to_file = True

path_in = config.dirs["s2s_processed"]
path_out = config.dirs["s2s_processed"]


# =============================================================================
# Lead-time configuration
# =============================================================================

def validate_user_settings():
    """Check settings before opening the input file."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = first_input_lead + x_days - 1

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )

    number_of_usable_leads = (
        last_input_lead - first_usable_lead + 1
    )

    if not isinstance(number_of_lead_bins, int):
        raise TypeError(
            "number_of_lead_bins must be an integer."
        )

    if number_of_lead_bins < 1:
        raise ValueError(
            "number_of_lead_bins must be at least 1."
        )

    if number_of_lead_bins > number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins cannot exceed the number of "
            "usable accumulated lead times."
        )

    if minimum_samples < 3:
        raise ValueError(
            "minimum_samples must be at least 3."
        )


def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """
    Split usable accumulated ending leads exactly as in the sample builder.

    Extra lead times are assigned to later bins.
    """

    number_of_leads = last_lead - first_lead + 1
    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size
        + int(
            bin_number
            >= number_of_bins - remainder
        )
        for bin_number in range(number_of_bins)
    ]

    lead_bins = []
    current_start = first_lead

    for bin_size in bin_sizes:

        current_end = current_start + bin_size - 1

        lead_bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = current_end + 1

    return lead_bins


def build_lead_bins():
    """Return lead bins used in the monthly-extreme input filename."""

    first_usable_lead = (
        first_input_lead + x_days - 1
    )

    return split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )


def full_range_variable_name():
    """Return the complete-window maximum variable name."""

    first_usable_lead = (
        first_input_lead + x_days - 1
    )

    return (
        f"max_value_lead"
        f"{first_usable_lead}_{last_input_lead}"
    )


# =============================================================================
# Filenames
# =============================================================================

def make_input_filename():
    """
    Construct the monthly-extreme sample filename.

    This follows the naming convention of the sample-building script.
    """

    first_usable_lead = (
        first_input_lead + x_days - 1
    )

    lead_bins = build_lead_bins()

    lead_bin_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in lead_bins
    )

    filename = (
        f"unseen_sample_monthly_catchment_precipitation_extremes_"
        f"{variable}_{x_days}dayacc_{catchment}_"
        f"lead{first_usable_lead}-{last_input_lead}_"
        f"split{number_of_lead_bins}_{lead_bin_text}_"
        f"forecast_hindcast_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )

    return os.path.join(
        path_in,
        filename,
    )


def make_output_filename():
    """Construct the Spearman-correlation output filename."""

    first_usable_lead = (
        first_input_lead + x_days - 1
    )

    filename = (
        f"independence_spearman_monthly_max_{variable}_"
        f"{x_days}dayacc_"
        f"nve_catchment_{catchment}_"
        f"lead{first_usable_lead}-{last_input_lead}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
    )

    return os.path.join(
        path_out,
        filename,
    )


# =============================================================================
# Input validation
# =============================================================================

def load_extreme_sample(filename):
    """Open and load the variables needed for the independence calculation."""

    full_variable = full_range_variable_name()

    required_variables = [
        full_variable,
        "forecast_date",
        "hdate",
        "ensemble_member",
        "model_type",
    ]

    with xr.open_dataset(filename) as dataset:

        missing = [
            name
            for name in required_variables
            if name not in dataset
        ]

        if missing:
            raise KeyError(
                "The input extreme-sample file is missing required "
                f"variables: {missing}. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        data = dataset[
            required_variables
        ].load()

        # Preserve useful global metadata for consistency checks.
        data.attrs.update(
            dataset.attrs
        )

    return data


def check_input_metadata(dataset):
    """
    Check important input metadata when it is available.

    The script still works if older files lack some global attributes.
    """

    expected = {
        "variable": variable,
        "catchment": catchment,
        "x_days": x_days,
        "first_input_lead": first_input_lead,
        "last_input_lead": last_input_lead,
    }

    for attribute, expected_value in expected.items():

        if attribute not in dataset.attrs:
            continue

        actual_value = dataset.attrs[attribute]

        if str(actual_value) != str(expected_value):
            raise ValueError(
                f"Input metadata mismatch for '{attribute}': "
                f"file contains {actual_value!r}, but the script "
                f"expects {expected_value!r}."
            )


# =============================================================================
# Helpers for decoding flattened metadata
# =============================================================================

def normalize_model_type(values):
    """Return model-type values as stripped lowercase strings."""

    flat_values = np.asarray(values).ravel()

    normalized = np.array(
        [
            (
                value.decode("utf-8")
                if isinstance(value, bytes)
                else str(value)
            ).strip().lower()
            for value in flat_values
        ],
        dtype=object,
    )

    return normalized


def datetime_values_to_key(values):
    """
    Convert forecast_date values to comparable datetime64[ns] values.

    Missing dates remain NaT.
    """

    return pd.to_datetime(
        np.asarray(values).ravel(),
        errors="coerce",
    ).to_numpy(
        dtype="datetime64[ns]"
    )


def hdate_values_to_key(values):
    """
    Convert hdate values to integer YYYYMMDD keys.

    Missing values and the forecast fill value are returned as -99999999.
    This avoids casting NaN directly to int64.
    """

    values = np.asarray(values).ravel()

    if np.issubdtype(
        values.dtype,
        np.datetime64,
    ):

        dates = pd.to_datetime(
            values,
            errors="coerce",
        )

        out = np.full(
            values.size,
            -99999999,
            dtype="int64",
        )

        valid = ~pd.isna(dates)

        out[valid] = (
            dates[valid]
            .strftime("%Y%m%d")
            .astype("int64")
        )

        return out


    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    out = np.full(
        values.size,
        -99999999,
        dtype="int64",
    )

    valid = np.isfinite(
        numeric_values
    )

    out[valid] = (
        numeric_values[valid]
        .astype("int64")
    )

    return out


# =============================================================================
# Reconstruct initialization-by-member matrices
# =============================================================================

def get_month_samples(
    dataset,
    month,
    model_type,
):
    """
    Extract valid flattened samples for one month and model type.

    Returns
    -------
    values : numpy.ndarray
    initialization_keys : numpy.ndarray
    member_labels : numpy.ndarray
    """

    full_variable = full_range_variable_name()

    values = (
        dataset[full_variable]
        .sel(month_of_year=month)
        .values
        .ravel()
        .astype("float64")
    )

    model_types = normalize_model_type(
        dataset["model_type"]
        .sel(month_of_year=month)
        .values
    )

    members = (
        dataset["ensemble_member"]
        .sel(month_of_year=month)
        .values
        .ravel()
        .astype("int64")
    )

    if model_type == "forecast":

        initialization_keys = datetime_values_to_key(
            dataset["forecast_date"]
            .sel(month_of_year=month)
            .values
        )

        valid_initialization = ~np.isnat(
            initialization_keys
        )

    elif model_type == "hindcast":

        hdate_keys = hdate_values_to_key(
            dataset["hdate"]
            .sel(month_of_year=month)
            .values
        )

        forecast_date_keys = datetime_values_to_key(
            dataset["forecast_date"]
            .sel(month_of_year=month)
            .values
        )


        # A hindcast sample is not uniquely identified by hdate alone.
        # The same historical hdate can occur in reforecast sets associated
        # with different real-time forecast initialization dates.
        #
        # Therefore one hindcast initialization is identified by the pair:
        #
        #     (forecast_date, hdate)
        #
        # This preserves separate reforecast cycles and prevents samples from
        # different cycles from being incorrectly placed in the same matrix row.
        initialization_keys = np.empty(
            hdate_keys.size,
            dtype=object,
        )

        for index in range(
            hdate_keys.size
        ):

            initialization_keys[index] = (
                forecast_date_keys[index],
                int(hdate_keys[index]),
            )


        valid_initialization = (
            ~np.isnat(
                forecast_date_keys
            )
            & (
                hdate_keys
                != -99999999
            )
        )

    else:
        raise ValueError(
            "model_type must be 'forecast' or 'hindcast'."
        )

    valid = (
        np.isfinite(values)
        & (model_types == model_type)
        & valid_initialization
        & (members >= 0)
    )

    return (
        values[valid],
        initialization_keys[valid],
        members[valid],
    )


def reconstruct_member_matrix(
    values,
    initialization_keys,
    member_labels,
    model_type,
):
    """
    Reconstruct an initialization-by-member matrix from flattened samples.

    Rows are unique forecast dates or hdates.
    Columns are ensemble-member labels.

    Duplicate initialization/member combinations are treated as an error
    because each member should contribute exactly one complete-window maximum
    to an initialization.
    """

    if values.size == 0:
        return (
            np.empty(
                (0, 0),
                dtype="float64",
            ),
            np.array([]),
            np.array([]),
        )

    # Build the initialization list and lookup together from the exact
    # objects present in initialization_keys.
    #
    # This is important because forecast keys are numpy.datetime64 objects.
    # Converting them with .tolist() can change their Python representation,
    # which can make an otherwise identical datetime fail as a dictionary key.
    unique_initializations = []
    initialization_lookup = {}

    for initialization in initialization_keys:

        if initialization not in initialization_lookup:

            initialization_lookup[
                initialization
            ] = len(
                unique_initializations
            )

            unique_initializations.append(
                initialization
            )


    unique_members = np.unique(
        member_labels
    )

    expected_members = (
        n_forecast_members
        if model_type == "forecast"
        else n_hindcast_members
    )

    if unique_members.size != expected_members:
        raise ValueError(
            f"{model_type.capitalize()} data contain "
            f"{unique_members.size} unique ensemble-member labels, "
            f"but {expected_members} were expected. "
            f"Found labels: {unique_members.tolist()}"
        )

    member_lookup = {
        member: index
        for index, member
        in enumerate(unique_members)
    }

    matrix = np.full(
        (
            len(unique_initializations),
            unique_members.size,
        ),
        np.nan,
        dtype="float64",
    )

    for value, initialization, member in zip(
        values,
        initialization_keys,
        member_labels,
    ):

        row = initialization_lookup[
            initialization
        ]

        column = member_lookup[
            member
        ]

        if np.isfinite(
            matrix[row, column]
        ):
            raise ValueError(
                "Duplicate sample found for "
                f"{model_type} initialization "
                f"{initialization!r}, member {member!r}."
            )

        matrix[
            row,
            column,
        ] = value

    return (
        matrix,
        unique_initializations,
        unique_members,
    )


# =============================================================================
# Spearman rank correlations
# =============================================================================

def make_member_pairs(number_of_members):
    """Return all unique pairs of ensemble-member column indices."""

    return list(
        combinations(
            range(number_of_members),
            2,
        )
    )


def spearman_correlation(
    x,
    y,
    minimum_valid_samples,
):
    """
    Calculate Spearman rank correlation for two member series.

    Missing values are removed pairwise.
    """

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    number_of_valid_samples = int(
        valid.sum()
    )

    if (
        number_of_valid_samples
        < minimum_valid_samples
    ):
        return np.nan

    x_valid = x[valid]
    y_valid = y[valid]

    if (
        np.all(
            x_valid == x_valid[0]
        )
        or np.all(
            y_valid == y_valid[0]
        )
    ):
        return np.nan

    x_ranks = rankdata(
        x_valid,
        method="average",
    )

    y_ranks = rankdata(
        y_valid,
        method="average",
    )

    return float(
        np.corrcoef(
            x_ranks,
            y_ranks,
        )[0, 1]
    )


def calculate_monthly_correlations(
    dataset,
    model_type,
):
    """
    Calculate member-pair Spearman correlations for all 12 calendar months.
    """

    months = np.arange(
        1,
        13,
        dtype="int8",
    )

    expected_members = (
        n_forecast_members
        if model_type == "forecast"
        else n_hindcast_members
    )

    pair_indices = make_member_pairs(
        expected_members
    )

    number_of_pairs = len(
        pair_indices
    )

    correlations = np.full(
        (
            len(months),
            number_of_pairs,
        ),
        np.nan,
        dtype="float32",
    )

    member_labels_for_output = None

    for month_index, month in enumerate(
        months
    ):

        (
            values,
            initialization_keys,
            member_labels,
        ) = get_month_samples(
            dataset=dataset,
            month=int(month),
            model_type=model_type,
        )

        if values.size == 0:

            print(
                f"{model_type:8s} | "
                f"month={int(month):2d} | "
                "initializations=0 | "
                f"member pairs={number_of_pairs}"
            )

            continue

        (
            matrix,
            unique_initializations,
            unique_members,
        ) = reconstruct_member_matrix(
            values=values,
            initialization_keys=initialization_keys,
            member_labels=member_labels,
            model_type=model_type,
        )

        if member_labels_for_output is None:
            member_labels_for_output = (
                unique_members.copy()
            )

        elif not np.array_equal(
            member_labels_for_output,
            unique_members,
        ):
            raise ValueError(
                f"{model_type.capitalize()} member labels differ "
                "between calendar months."
            )

        print(
            f"{model_type:8s} | "
            f"month={int(month):2d} | "
            f"initializations={matrix.shape[0]} | "
            f"member pairs={number_of_pairs}"
        )

        for pair_index, (
            member_index_1,
            member_index_2,
        ) in enumerate(
            pair_indices
        ):

            correlations[
                month_index,
                pair_index,
            ] = spearman_correlation(
                x=matrix[
                    :,
                    member_index_1,
                ],
                y=matrix[
                    :,
                    member_index_2,
                ],
                minimum_valid_samples=minimum_samples,
            )

    if member_labels_for_output is None:

        # This should only occur if no samples of this model type exist.
        member_labels_for_output = np.arange(
            expected_members,
            dtype="int64",
        )

    pair_member_1 = np.array(
        [
            member_labels_for_output[index_1]
            for index_1, index_2
            in pair_indices
        ],
        dtype="int64",
    )

    pair_member_2 = np.array(
        [
            member_labels_for_output[index_2]
            for index_1, index_2
            in pair_indices
        ],
        dtype="int64",
    )

    return (
        xr.DataArray(
            correlations,
            dims=(
                "month_of_year",
                "pair",
            ),
            coords={
                "month_of_year": months,
                "pair": np.arange(
                    number_of_pairs,
                    dtype="int16",
                ),
            },
            name=(
                f"{model_type}_spearman_rho"
            ),
        ),
        pair_member_1,
        pair_member_2,
    )


# =============================================================================
# Output
# =============================================================================

def build_output_dataset(
    forecast_correlations,
    forecast_member_1,
    forecast_member_2,
    hindcast_correlations,
    hindcast_member_1,
    hindcast_member_2,
):
    """Build the final Spearman-correlation Dataset."""

    forecast_correlations = (
        forecast_correlations.rename(
            {
                "pair": "forecast_pair"
            }
        )
    )

    hindcast_correlations = (
        hindcast_correlations.rename(
            {
                "pair": "hindcast_pair"
            }
        )
    )

    dataset = xr.Dataset(
        {
            "forecast_spearman_rho": (
                forecast_correlations
            ),
            "hindcast_spearman_rho": (
                hindcast_correlations
            ),
            "forecast_pair_member_1": (
                (
                    "forecast_pair",
                ),
                forecast_member_1,
            ),
            "forecast_pair_member_2": (
                (
                    "forecast_pair",
                ),
                forecast_member_2,
            ),
            "hindcast_pair_member_1": (
                (
                    "hindcast_pair",
                ),
                hindcast_member_1,
            ),
            "hindcast_pair_member_2": (
                (
                    "hindcast_pair",
                ),
                hindcast_member_2,
            ),
        }
    )

    add_output_metadata(
        dataset
    )

    return dataset


def add_output_metadata(dataset):
    """Add descriptions and global metadata."""

    full_variable = (
        full_range_variable_name()
    )

    dataset[
        "forecast_spearman_rho"
    ].attrs = {
        "long_name": (
            "pairwise Spearman rank correlation between forecast members"
        ),
        "description": (
            "Calculated across complete-window initialization maxima "
            "separately for each assigned calendar month"
        ),
    }

    dataset[
        "hindcast_spearman_rho"
    ].attrs = {
        "long_name": (
            "pairwise Spearman rank correlation between hindcast members"
        ),
        "description": (
            "Calculated across complete-window hdate maxima separately "
            "for each assigned calendar month"
        ),
    }

    for name in [
        "forecast_pair_member_1",
        "forecast_pair_member_2",
        "hindcast_pair_member_1",
        "hindcast_pair_member_2",
    ]:

        dataset[name].attrs[
            "description"
        ] = (
            "Ensemble-member label for this correlation pair"
        )

    dataset[
        "month_of_year"
    ].attrs = {
        "long_name": (
            "assigned calendar month"
        ),
        "description": (
            "Calendar-month grouping inherited directly from the "
            "monthly extreme-sample input file"
        ),
    }

    dataset.attrs = {
        "title": (
            "ECMWF S2S ensemble-member independence test from "
            "precomputed monthly complete-window maxima"
        ),
        "variable": variable,
        "catchment": catchment,
        "accumulation_days": int(
            x_days
        ),
        "first_input_lead": int(
            first_input_lead
        ),
        "last_input_lead": int(
            last_input_lead
        ),
        "first_usable_accumulation_lead": int(
            first_input_lead
            + x_days
            - 1
        ),
        "maximum_input_variable": (
            full_variable
        ),
        "forecast_date_start": (
            forecast_date_range[0]
        ),
        "forecast_date_end": (
            forecast_date_range[1]
        ),
        "minimum_samples": int(
            minimum_samples
        ),
        "input_processing": (
            "Uses precomputed complete-window maxima and metadata; "
            "original daily forecast and hindcast files are not reopened"
        ),
        "forecast_interpretation": (
            "Each forecast_date is reconstructed as one forecast "
            "ensemble initialization"
        ),
        "hindcast_interpretation": (
            "Each unique (forecast_date, hdate) combination is "
            "reconstructed as one hindcast ensemble initialization"
        ),
    }


def write_output(dataset):
    """Write results to NetCDF."""

    os.makedirs(
        path_out,
        exist_ok=True,
    )

    filename_out = (
        make_output_filename()
    )

    encoding = {
        "forecast_spearman_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
        "hindcast_spearman_rho": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        },
    }

    dataset.to_netcdf(
        filename_out,
        encoding=encoding,
    )

    print()
    print("=" * 72)
    print(
        "Output written successfully"
    )
    print("=" * 72)
    print(
        filename_out
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    filename_in = (
        make_input_filename()
    )

    print(
        "=" * 72
    )
    print(
        "Monthly-maximum ensemble-member independence test"
    )
    print(
        "=" * 72
    )

    print(
        f"Variable:               "
        f"{variable}"
    )

    print(
        f"Catchment:              "
        f"{catchment}"
    )

    print(
        f"Accumulation:           "
        f"{x_days} days"
    )

    print(
        f"Maximum variable:       "
        f"{full_range_variable_name()}"
    )

    print(
        f"Maximum ending leads:   "
        f"{first_input_lead + x_days - 1}-"
        f"{last_input_lead}"
    )

    print(
        f"Minimum samples:        "
        f"{minimum_samples}"
    )

    print()
    print(
        "Reading precomputed extreme sample:"
    )
    print(
        filename_in
    )

    extreme_sample = (
        load_extreme_sample(
            filename_in
        )
    )

    check_input_metadata(
        extreme_sample
    )

    print()
    print(
        "Calculating forecast correlations"
    )
    print(
        "---------------------------------"
    )

    (
        forecast_correlations,
        forecast_member_1,
        forecast_member_2,
    ) = calculate_monthly_correlations(
        dataset=extreme_sample,
        model_type="forecast",
    )

    print()
    print(
        "Calculating hindcast correlations"
    )
    print(
        "---------------------------------"
    )

    (
        hindcast_correlations,
        hindcast_member_1,
        hindcast_member_2,
    ) = calculate_monthly_correlations(
        dataset=extreme_sample,
        model_type="hindcast",
    )

    output_dataset = (
        build_output_dataset(
            forecast_correlations=forecast_correlations,
            forecast_member_1=forecast_member_1,
            forecast_member_2=forecast_member_2,
            hindcast_correlations=hindcast_correlations,
            hindcast_member_1=hindcast_member_1,
            hindcast_member_2=hindcast_member_2,
        )
    )

    print()
    print(
        output_dataset
    )

    if write_to_file:

        write_output(
            output_dataset
        )
