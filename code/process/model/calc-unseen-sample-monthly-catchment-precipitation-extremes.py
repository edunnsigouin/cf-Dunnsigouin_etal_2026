"""
Build a monthly distribution of S2S precipitation extremes from forecasts and
hindcasts, and additionally split the full extreme sample according to the lead
time at which each maximum occurs.

For each forecast initialization date, this script:

1. Loads forecast and hindcast daily precipitation.
2. Calculates catchment-weighted spatial mean precipitation.
3. Calculates trailing X-day accumulated precipitation.
4. Calculates ONE maximum for every forecast ensemble member and every
   hindcast date/member over the complete usable accumulated lead window.
5. Stores the full-sample maximum and its supporting information:
       max_value_lead<first_usable_lead>_<last_input_lead>
       date_of_max
       forecast_date
       hdate
       ensemble_member
       model_type
       source_file
6. Records internally the ending lead day at which each maximum occurs.
7. Splits those SAME full-window maxima into a user-defined number of
   lead-location bins.

Important: the lead-bin variables are NOT maxima recalculated over shorter
lead windows. A full-window maximum is calculated once, then copied into the
single lead-bin sample corresponding to the lead day where that maximum occurs.

Calendar-month assignment
-------------------------
The calendar month containing the largest number of usable accumulated lead
dates determines where ALL samples from one forecast/hindcast initialization
are stored.

If two calendar months contain the same number of usable lead dates, the FIRST
calendar month is used.

For example, with a 2-day accumulation and 30 usable accumulated lead times:

    20 dates in January + 10 dates in February -> January
    15 dates in January + 15 dates in February -> January

Lead-time splitting
-------------------
The usable accumulated ending leads are split into `number_of_lead_bins`
consecutive, approximately equal bins.

The split is performed AFTER accounting for the X-day accumulation.

For example:

    first_input_lead = 16
    last_input_lead  = 46
    x_days           = 2

gives usable accumulated ending leads 17-46, or 30 lead times.

With:

    number_of_lead_bins = 2

the bins are:

    17-31
    32-46

and the additional output variables are:

    max_value_lead17_31
    max_value_lead32_46

If the number of usable leads cannot be divided evenly by the requested number
of bins, the extra lead times are assigned to the later bins.

For example, 10 usable leads split into 3 bins would have sizes 3, 3, and 4.

Tie handling for equal precipitation maxima
--------------------------------------------
If exactly the same maximum precipitation value occurs at more than one lead
time, the first occurrence in lead-time order is used to determine its lead bin.

Output structure
----------------
The full sample retains all supporting information from the original
distribution script.

Each lead-bin variable contains only precipitation values. The lead-bin samples
are stored compactly within each calendar month and do not duplicate
date/member/source metadata.

For every calendar month:

    number of full-window maximum samples
        =
    sum of sample numbers across all lead-bin variables

The output also contains one monthly sample-count variable for every maximum
variable. For example, with x_days=2:

    max_value_lead17_46  -> sample_count_lead17_46
    max_value_lead17_31  -> sample_count_lead17_31
    max_value_lead32_46  -> sample_count_lead32_46

The complete-window maximum variable name is generated automatically from
x_days. For example:

    x_days=1 -> max_value_lead16_46
    x_days=2 -> max_value_lead17_46
    x_days=3 -> max_value_lead18_46

apart from genuinely missing maximum values or lead-day information.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"
x_days = 2
catchment = "regine_drammen"

#forecast_date_range = ["2020-01-02","2023-06-26"]
forecast_date_range = ["2020-01-02","2022-12-29"]

path_in_forecast = (
    config.dirs["s2s_forecast_daily"]
    + variable
    + "/"
)

path_in_hindcast = (
    config.dirs["s2s_hindcast_daily"]
    + variable
    + "/"
)

filename_weights = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
)

path_out = config.dirs["s2s_processed"]


# Daily lead times available in the original forecast/hindcast files.
first_input_lead = 16
last_input_lead = 46


# Number of lead-location bins used to partition the full-window maxima.
#
# For x_days=2 and input leads 16-46:
#     usable accumulated ending leads = 17-46
#
# number_of_lead_bins=2 gives:
#     17-31
#     32-46
number_of_lead_bins = 2


# Expected ensemble structure.
# These values are used only to determine the required storage size.
n_forecast_members = 51
n_hindcast_members = 11
n_hdates = 20


write2file = True


# =============================================================================
# Lead-time configuration
# =============================================================================

def validate_user_settings():
    """Check the user settings before opening any input files."""

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
        last_input_lead
        - first_usable_lead
        + 1
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


def split_usable_accumulated_leads(
    first_lead,
    last_lead,
    number_of_bins,
):
    """
    Split an inclusive lead interval into approximately equal consecutive bins.

    Extra lead times are assigned to the later bins.

    Example:
        first_lead = 17
        last_lead = 46
        number_of_bins = 2

    returns:
        [(17, 31), (32, 46)]
    """

    number_of_leads = (
        last_lead
        - first_lead
        + 1
    )

    base_size = (
        number_of_leads
        // number_of_bins
    )

    remainder = (
        number_of_leads
        % number_of_bins
    )

    # Later bins receive the extra lead times.
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

        current_end = (
            current_start
            + bin_size
            - 1
        )

        lead_bins.append(
            (
                current_start,
                current_end,
            )
        )

        current_start = (
            current_end
            + 1
        )

    return lead_bins


def build_lead_bins():
    """
    Build lead-location bins from the usable accumulated ending leads.

    The first usable accumulated lead is:

        first_input_lead + x_days - 1
    """

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    return split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )


def lead_bin_variable_name(
    lead_start,
    lead_end,
):
    """Return the NetCDF variable name for one lead-location sample."""

    return (
        f"max_value_lead"
        f"{lead_start}_{lead_end}"
    )


def full_range_variable_name():
    """
    Return the NetCDF variable name for the complete usable lead window.

    The first usable accumulated ending lead depends on x_days.

    Examples:
        x_days = 1 -> max_value_lead16_46
        x_days = 2 -> max_value_lead17_46
        x_days = 3 -> max_value_lead18_46
    """

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    return (
        f"max_value_lead"
        f"{first_usable_lead}_{last_input_lead}"
    )


def sample_count_variable_name(
    maximum_variable_name,
):
    """
    Return the monthly sample-count variable corresponding to one maximum
    variable.

    Example:
        max_value_lead17_46
        -> sample_count_lead17_46
    """

    return maximum_variable_name.replace(
        "max_value_",
        "sample_count_",
        1,
    )


# =============================================================================
# Dates and filenames
# =============================================================================

def get_forecast_dates(
    forecast_date_range,
    option="mt",
):
    """
    Return forecast initialization dates.

    option:
        "mt"  : Mondays and Thursdays
        "all" : all calendar days
    """

    start_date = pd.to_datetime(
        forecast_date_range[0]
    )

    end_date = pd.to_datetime(
        forecast_date_range[1]
    )

    if option == "mt":

        mondays = pd.date_range(
            start_date,
            end_date,
            freq="W-MON",
        )

        thursdays = pd.date_range(
            start_date,
            end_date,
            freq="W-THU",
        )

        dates = mondays.union(
            thursdays
        )

    elif option == "all":

        dates = pd.date_range(
            start_date,
            end_date,
            freq="D",
        )

    else:
        raise ValueError(
            "option must be 'mt' or 'all'"
        )

    return (
        dates
        .sort_values()
        .strftime("%Y-%m-%d")
        .tolist()
    )


def get_model_filenames(date):
    """Return forecast and hindcast filenames for one initialization date."""

    forecast_filename = (
        path_in_forecast
        + f"{variable}_0.5x0.5_{date}.nc"
    )

    hindcast_filename = (
        path_in_hindcast
        + f"{variable}_0.5x0.5_{date}.nc"
    )

    return (
        forecast_filename,
        hindcast_filename,
    )


def make_output_filename(
    lead_bins,
):
    """Return the output NetCDF filename."""

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    lead_bin_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in lead_bins
    )

    return os.path.join(
        path_out,
        (
            f"unseen_sample_monthly_catchment_precipitation_extremes_"
            f"{variable}_{x_days}dayacc_{catchment}_"
            f"lead{first_usable_lead}-{last_input_lead}_"
            f"split{number_of_lead_bins}_{lead_bin_text}_"
            f"forecast_hindcast_{forecast_date_range[0]}_{forecast_date_range[1]}.nc"
        ),
    )


# =============================================================================
# Loading
# =============================================================================

def load_weights(filename):
    """Load catchment weights into memory."""

    with xr.open_dataset(filename) as ds:

        if "catchment_weight" not in ds:
            raise KeyError(
                f"'catchment_weight' not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        weights = (
            ds["catchment_weight"]
            .astype("float32")
            .load()
        )

    weights.name = "catchment_weight"

    return weights


def convert_to_mm(da):
    """Convert precipitation from metres to millimetres when needed."""

    units = str(
        da.attrs.get("units", "")
    ).lower()

    if units in {
        "m",
        "meter",
        "metre",
    }:
        da = da * 1000.0
        da.attrs["units"] = "mm"

    return da


def load_model_data(date):
    """
    Load forecast and hindcast daily precipitation.

    - Precipitation in metres is converted to millimetres.
    - Hindcast ensemble-member labels are reset to 1...N.
    - Hindcast hdate values are converted to integer YYYYMMDD when necessary.
    """

    (
        forecast_filename,
        hindcast_filename,
    ) = get_model_filenames(date)

    with xr.open_dataset(
        forecast_filename
    ) as ds_forecast:
        forecast = (
            ds_forecast[variable]
            .load()
        )

    with xr.open_dataset(
        hindcast_filename
    ) as ds_hindcast:
        hindcast = (
            ds_hindcast[variable]
            .load()
        )

    forecast = convert_to_mm(
        forecast
    )

    hindcast = convert_to_mm(
        hindcast
    )


    # Hindcast ensemble-member labels.
    if "number" in hindcast.dims:

        n_members = (
            hindcast.sizes["number"]
        )

        hindcast = hindcast.assign_coords(
            number=np.arange(
                1,
                n_members + 1,
            )
        )


    # Hindcast date identifiers.
    if "hdate" in hindcast.coords:

        hdate_values = (
            hindcast["hdate"]
            .values
        )

        if np.issubdtype(
            hdate_values.dtype,
            np.datetime64,
        ):

            hdate_values = (
                pd.to_datetime(
                    hdate_values
                )
                .strftime("%Y%m%d")
                .astype(int)
            )

        else:

            hdate_values = (
                hdate_values
                .astype(int)
            )

        hindcast = hindcast.assign_coords(
            hdate=hdate_values
        )

    return (
        forecast,
        hindcast,
    )


# =============================================================================
# Catchment precipitation
# =============================================================================

def catchment_mean(
    da,
    weights,
    spatial_dims=("latitude", "longitude"),
):
    """
    Calculate catchment-weighted spatial mean precipitation.

    Formula:

        sum(precipitation * catchment_weight)
        -------------------------------------
                 sum(catchment_weight)
    """

    valid = (
        np.isfinite(da)
        & np.isfinite(weights)
        & (weights > 0)
    )

    weighted_sum = (
        da.where(valid)
        * weights.where(valid)
    ).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = (
        weights.where(valid)
        .sum(
            dim=spatial_dims,
            skipna=True,
        )
    )

    out = (
        weighted_sum
        / weight_sum
    )

    out.attrs["description"] = (
        "Catchment-weighted daily mean precipitation"
    )

    out.attrs["units"] = (
        da.attrs.get(
            "units",
            "",
        )
    )

    return out


# =============================================================================
# X-day accumulation
# =============================================================================

def xday_accumulation(da):
    """
    Calculate trailing X-day accumulated precipitation.

    An explicit `lead_day` coordinate is added after accumulation.

    Example:
        input daily leads = 16...46
        x_days = 2

    gives:
        accumulated ending leads = 17...46
    """

    expected_input_size = (
        last_input_lead
        - first_input_lead
        + 1
    )

    if da.sizes["time"] != expected_input_size:
        raise ValueError(
            "Unexpected number of daily lead times. "
            f"Expected {expected_input_size} for leads "
            f"{first_input_lead}-{last_input_lead}, "
            f"but found {da.sizes['time']}."
        )

    out = (
        da
        .rolling(
            time=x_days,
            min_periods=x_days,
        )
        .sum()
        .dropna(
            "time",
            how="any",
        )
    )

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )

    usable_leads = np.arange(
        first_usable_lead,
        last_input_lead + 1,
        dtype="int16",
    )

    if (
        out.sizes["time"]
        != usable_leads.size
    ):
        raise ValueError(
            "The accumulated time dimension does not match "
            "the expected usable accumulated lead range."
        )

    out = out.assign_coords(
        lead_day=(
            "time",
            usable_leads,
        )
    )

    out.attrs["description"] = (
        f"{x_days}-day accumulated catchment-weighted "
        "mean precipitation"
    )

    out.attrs["units"] = (
        da.attrs.get(
            "units",
            "",
        )
    )

    return out


# =============================================================================
# Calendar-month assignment
# =============================================================================

def get_month_with_most_forecast_dates(da):
    """
    Return the calendar month containing the largest number of lead dates.

    If two months are tied, the FIRST month is returned.

    This happens naturally because the month numbers are in ascending order
    and argmax/idxmax returns the first occurrence of the largest count.
    """

    lead_months = (
        da["time"]
        .dt.month
        .values
    )

    months, counts = np.unique(
        lead_months,
        return_counts=True,
    )

    first_largest_position = (
        np.argmax(counts)
    )

    return int(
        months[
            first_largest_position
        ]
    )


# =============================================================================
# Maximum extraction
# =============================================================================

def extract_max_info(da):
    """
    Calculate full-window maximum information for every sample.

    The maximum is calculated once over the complete usable accumulated lead
    window.

    Also returned:
        date_of_max
        lead_of_max
        month_with_most_forecast_dates

    If equal maximum values occur at several lead times, argmax uses the first
    occurrence in lead-time order.
    """

    max_value = da.max(
        dim="time"
    )

    max_value.name = (
        "max_value"
    )


    index_of_max = da.argmax(
        dim="time"
    )


    date_of_max = (
        da["time"]
        .isel(
            time=index_of_max
        )
    )

    date_of_max.name = (
        "date_of_max"
    )


    lead_of_max = (
        da["lead_day"]
        .isel(
            time=index_of_max
        )
    )

    lead_of_max.name = (
        "lead_of_max"
    )


    month = (
        get_month_with_most_forecast_dates(
            da
        )
    )


    return xr.Dataset(
        {
            "max_value": max_value,
            "date_of_max": date_of_max,
            "lead_of_max": lead_of_max,
            "month_with_most_forecast_dates": xr.DataArray(
                month
            ),
        }
    )


# =============================================================================
# Output storage
# =============================================================================

def initialize_extreme_store(
    n_forecasts,
    lead_bins,
):
    """
    Create storage for the full sample and all lead-bin samples.

    Full sample:
        max_value and all supporting metadata.

    Lead-bin samples:
        precipitation values only.
    """

    n_index = (
        n_forecasts
        * (
            n_forecast_members
            + n_hindcast_members
            * n_hdates
        )
    )


    full_variable = (
        full_range_variable_name()
    )

    data_vars = {
        full_variable: (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                np.nan,
                dtype="float32",
            ),
        ),

        "date_of_max": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                np.datetime64("NaT"),
                dtype="datetime64[ns]",
            ),
        ),

        "forecast_date": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                np.datetime64("NaT"),
                dtype="datetime64[ns]",
            ),
        ),

        "hdate": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                -99999999,
                dtype="int32",
            ),
        ),

        "ensemble_member": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                -999,
                dtype="int64",
            ),
        ),

        "model_type": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                "",
                dtype=object,
            ),
        ),

        "source_file": (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                "",
                dtype=object,
            ),
        ),
    }


    # Add one precipitation-only variable for every lead-location bin.
    for (
        lead_start,
        lead_end,
    ) in lead_bins:

        variable_name = (
            lead_bin_variable_name(
                lead_start=lead_start,
                lead_end=lead_end,
            )
        )

        data_vars[
            variable_name
        ] = (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                np.nan,
                dtype="float32",
            ),
        )


    # Add one monthly sample-count variable for every maximum variable.
    maximum_variables = [
        full_variable
    ] + [
        lead_bin_variable_name(
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]

    for maximum_variable in maximum_variables:

        count_variable = (
            sample_count_variable_name(
                maximum_variable
            )
        )

        data_vars[
            count_variable
        ] = (
            ("month_of_year",),
            np.zeros(
                12,
                dtype="int32",
            ),
        )


    store = xr.Dataset(
        data_vars=data_vars,

        coords={
            "month_of_year": np.arange(
                1,
                13,
                dtype="int8",
            ),

            "index": np.arange(
                n_index,
                dtype="int32",
            ),
        },
    )


    add_store_metadata(
        store=store,
        lead_bins=lead_bins,
    )

    return store


def add_store_metadata(
    store,
    lead_bins,
):
    """Add clear variable and global metadata to the output dataset."""

    first_usable_lead = (
        first_input_lead
        + x_days
        - 1
    )


    store.attrs.update(
        {
            "description": (
                "Monthly pooled forecast/hindcast maxima calculated over "
                "the complete usable accumulated lead window, plus subsets "
                "of those SAME maxima grouped by the lead-time bin in which "
                "each maximum occurs."
            ),

            "variable": variable,
            "catchment": catchment,
            "x_days": x_days,

            "first_input_lead": first_input_lead,
            "last_input_lead": last_input_lead,

            "first_usable_accumulated_lead": (
                first_usable_lead
            ),

            "last_usable_accumulated_lead": (
                last_input_lead
            ),

            "number_of_lead_bins": (
                number_of_lead_bins
            ),

            "calendar_month_binning": (
                "calendar month containing the largest number of complete "
                "usable accumulated lead dates"
            ),

            "calendar_month_tie_handling": (
                "first calendar month used when equal numbers of usable "
                "accumulated lead dates occur in multiple months"
            ),

            "lead_bin_sampling": (
                "complete-window maxima partitioned by ending lead day "
                "of maximum"
            ),

            "maximum_tie_handling": (
                "first lead-time occurrence used when equal precipitation "
                "maxima are tied"
            ),

            "forecast_date_start": (
                forecast_date_range[0]
            ),

            "forecast_date_end": (
                forecast_date_range[1]
            ),
        }
    )


    full_variable = (
        full_range_variable_name()
    )

    store[full_variable].attrs.update(
        {
            "description": (
                f"Maximum {x_days}-day accumulated catchment-mean "
                f"precipitation over complete ending leads "
                f"{first_usable_lead}-{last_input_lead}"
            ),
            "units": "mm",
            "lead_start": first_usable_lead,
            "lead_end": last_input_lead,
            "range_type": (
                "complete usable lead window"
            ),
        }
    )


    for bin_number, (
        lead_start,
        lead_end,
    ) in enumerate(
        lead_bins,
        start=1,
    ):

        variable_name = (
            lead_bin_variable_name(
                lead_start=lead_start,
                lead_end=lead_end,
            )
        )

        store[
            variable_name
        ].attrs.update(
            {
                "description": (
                    "Subset of complete-window maxima whose maximum occurs "
                    f"at ending leads {lead_start}-{lead_end}; "
                    f"lead bin {bin_number} of {number_of_lead_bins}"
                ),

                "units": "mm",
                "lead_start": lead_start,
                "lead_end": lead_end,

                "range_type": (
                    f"lead-location bin {bin_number} "
                    f"of {number_of_lead_bins}"
                ),
            }
        )


    # Metadata for the monthly sample-count variables.
    maximum_variables = [
        full_variable
    ] + [
        lead_bin_variable_name(
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]

    for maximum_variable in maximum_variables:

        count_variable = (
            sample_count_variable_name(
                maximum_variable
            )
        )

        store[count_variable].attrs.update(
            {
                "description": (
                    f"Number of finite samples stored in "
                    f"{maximum_variable} for each calendar month"
                ),
                "corresponding_maximum_variable": maximum_variable,
            }
        )


    store["date_of_max"].attrs[
        "description"
    ] = (
        "Date when max_value occurs"
    )


    store["forecast_date"].attrs[
        "description"
    ] = (
        "Forecast initialization date"
    )


    store["hdate"].attrs[
        "description"
    ] = (
        "Hindcast date as integer YYYYMMDD; "
        "-99999999 for forecasts"
    )

    store["hdate"].attrs[
        "_FillValue"
    ] = np.int32(
        -99999999
    )


    store["ensemble_member"].attrs[
        "description"
    ] = (
        "Ensemble member number"
    )


    store["model_type"].attrs[
        "description"
    ] = (
        "Either 'forecast' or 'hindcast'"
    )


    store["source_file"].attrs[
        "description"
    ] = (
        "Source NetCDF file"
    )


# =============================================================================
# Labels matching the flattened full sample
# =============================================================================

def get_member_labels(max_value):
    """
    Return ensemble-member labels matching the flattened max_value order.

    Forecast:
        number

    Hindcast:
        hdate, number
    """

    if "number" not in max_value.coords:

        return np.arange(
            max_value.size
        )


    members = (
        max_value["number"]
        .values
    )


    if "hdate" in max_value.dims:

        members = np.tile(
            members,
            max_value.sizes["hdate"],
        )


    return members


def get_hdate_labels(max_value):
    """
    Return hdate labels matching the flattened max_value order.

    Forecast:
        -99999999

    Hindcast:
        integer YYYYMMDD repeated for each ensemble member.
    """

    fill_value = np.int32(
        -99999999
    )


    if "hdate" not in max_value.dims:

        return np.full(
            max_value.size,
            fill_value,
            dtype="int32",
        )


    hdates = (
        max_value["hdate"]
        .values
        .astype("int32")
    )

    n_members = (
        max_value.sizes["number"]
    )


    return np.repeat(
        hdates,
        n_members,
    ).astype("int32")


# =============================================================================
# Finding free storage positions
# =============================================================================

def get_free_indices(
    store,
    variable_name,
    month,
    n_values,
):
    """
    Return the first available storage positions for one variable and month.

    Full-window and lead-bin variables are allowed to have different numbers
    of stored values, so free positions are found separately for each maximum
    variable.
    """

    if n_values == 0:
        return np.array(
            [],
            dtype="int32",
        )


    current_values = (
        store[variable_name]
        .sel(
            month_of_year=month
        )
        .values
    )

    free_positions = np.where(
        ~np.isfinite(
            current_values
        )
    )[0]


    if free_positions.size < n_values:
        raise ValueError(
            f"Not enough storage for {variable_name}, month {month}. "
            f"Need {n_values} positions but only "
            f"{free_positions.size} remain."
        )


    return store["index"].values[
        free_positions[:n_values]
    ]


# =============================================================================
# Store the full sample
# =============================================================================

def add_full_max_info_to_store(
    store,
    max_info,
    forecast_date,
    source_file,
    model_type,
):
    """
    Store the full-window maximum sample and all its supporting information.

    All samples from this initialization go into the same calendar month:
    the month containing the largest number of usable accumulated lead dates.

    If month counts are tied, the first month is used.
    """

    month = int(
        max_info[
            "month_with_most_forecast_dates"
        ].values
    )


    max_value = (
        max_info["max_value"]
    )


    max_values = (
        max_value
        .values
        .astype("float32")
        .ravel()
    )


    dates = (
        max_info["date_of_max"]
        .values
        .ravel()
    )


    members = (
        get_member_labels(
            max_value
        )
    )


    hdates = (
        get_hdate_labels(
            max_value
        )
    )


    # Remove genuinely missing maxima from the stored full sample.
    valid = np.isfinite(
        max_values
    )


    max_values = (
        max_values[valid]
    )

    dates = (
        dates[valid]
    )

    members = (
        members[valid]
    )

    hdates = (
        hdates[valid]
    )


    n_values = (
        max_values.size
    )


    if n_values == 0:
        return store


    full_variable = (
        full_range_variable_name()
    )


    index_values = get_free_indices(
        store=store,
        variable_name=full_variable,
        month=month,
        n_values=n_values,
    )


    store[full_variable].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = max_values


    store["date_of_max"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = dates


    store["forecast_date"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = np.datetime64(
        forecast_date
    )


    store["hdate"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = hdates


    store["ensemble_member"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = members


    store["model_type"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = model_type


    store["source_file"].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = source_file


    return store


# =============================================================================
# Split and store the SAME full-window maxima by lead location
# =============================================================================

def add_lead_bin_values_to_store(
    store,
    max_info,
    lead_bins,
):
    """
    Partition the full-window maxima according to where each maximum occurred.

    No new maximum is calculated here.

    For every sample:
        1. take its already calculated full-window max_value;
        2. inspect lead_of_max;
        3. copy max_value into exactly one lead-bin sample.

    Lead-bin samples contain precipitation values only.
    """

    month = int(
        max_info[
            "month_with_most_forecast_dates"
        ].values
    )


    max_values = (
        max_info["max_value"]
        .values
        .astype("float32")
        .ravel()
    )


    max_leads = (
        max_info["lead_of_max"]
        .values
        .astype("int16")
        .ravel()
    )


    valid = (
        np.isfinite(max_values)
        & np.isfinite(max_leads)
    )


    max_values = (
        max_values[valid]
    )

    max_leads = (
        max_leads[valid]
    )


    number_assigned = 0


    for (
        lead_start,
        lead_end,
    ) in lead_bins:

        in_bin = (
            (max_leads >= lead_start)
            & (max_leads <= lead_end)
        )


        bin_values = (
            max_values[in_bin]
        )


        number_assigned += (
            bin_values.size
        )


        if bin_values.size == 0:
            continue


        variable_name = (
            lead_bin_variable_name(
                lead_start=lead_start,
                lead_end=lead_end,
            )
        )


        index_values = get_free_indices(
            store=store,
            variable_name=variable_name,
            month=month,
            n_values=bin_values.size,
        )


        store[variable_name].loc[
            dict(
                month_of_year=month,
                index=index_values,
            )
        ] = bin_values


    if number_assigned != max_values.size:
        raise ValueError(
            "Not every valid full-window maximum was assigned to exactly "
            "one lead-time bin. Check the lead-bin definitions."
        )


    return store


# =============================================================================
# Monthly sample counts
# =============================================================================

def update_sample_counts(
    store,
    lead_bins,
):
    """
    Calculate and store the number of finite samples for every maximum variable.

    Each count variable has dimension:

        sample_count_...(month_of_year)

    and corresponds directly to one maximum variable.

    Example:
        max_value_lead17_46
            -> sample_count_lead17_46

        max_value_lead17_31
            -> sample_count_lead17_31
    """

    maximum_variables = [
        full_range_variable_name()
    ] + [
        lead_bin_variable_name(
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]


    for maximum_variable in maximum_variables:

        count_variable = (
            sample_count_variable_name(
                maximum_variable
            )
        )

        counts = np.isfinite(
            store[
                maximum_variable
            ].values
        ).sum(
            axis=1
        ).astype(
            "int32"
        )

        store[
            count_variable
        ].values[:] = counts


    return store


# =============================================================================
# Output checks
# =============================================================================

def print_monthly_sample_counts(
    store,
    lead_bins,
):
    """
    Print monthly sample counts and verify that the lead bins partition
    the full sample.
    """

    month_names = [
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


    full_variable = (
        full_range_variable_name()
    )


    bin_variables = [
        lead_bin_variable_name(
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]


    bin_labels = [
        f"{lead_start}-{lead_end}"
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]


    print()
    print(
        "Sample counts by calendar month"
    )
    print(
        "-------------------------------"
    )


    header = (
        f"{'Month':<12}"
        f"{'full':>10}"
    )

    for label in bin_labels:
        header += (
            f"{label:>12}"
        )

    header += (
        f"{'bins_sum':>12}"
        f"{'check':>10}"
    )


    print(header)
    print(
        "-" * len(header)
    )


    all_months_ok = True


    for month, month_name in enumerate(
        month_names,
        start=1,
    ):

        full_count = int(
            np.isfinite(
                store[full_variable]
                .sel(
                    month_of_year=month
                )
                .values
            ).sum()
        )


        bin_counts = []

        for variable_name in bin_variables:

            count = int(
                np.isfinite(
                    store[variable_name]
                    .sel(
                        month_of_year=month
                    )
                    .values
                ).sum()
            )

            bin_counts.append(
                count
            )


        bins_sum = sum(
            bin_counts
        )


        check = (
            "OK"
            if bins_sum == full_count
            else "FAIL"
        )


        if check == "FAIL":
            all_months_ok = False


        row = (
            f"{month_name:<12}"
            f"{full_count:>10}"
        )

        for count in bin_counts:
            row += (
                f"{count:>12}"
            )

        row += (
            f"{bins_sum:>12}"
            f"{check:>10}"
        )


        print(row)


    print(
        "-" * len(header)
    )


    full_total = int(
        np.isfinite(
            store[full_variable]
            .values
        ).sum()
    )


    bin_totals = [
        int(
            np.isfinite(
                store[variable_name]
                .values
            ).sum()
        )
        for variable_name
        in bin_variables
    ]


    bins_total = sum(
        bin_totals
    )


    total_check = (
        "OK"
        if bins_total == full_total
        else "FAIL"
    )


    total_row = (
        f"{'TOTAL':<12}"
        f"{full_total:>10}"
    )

    for count in bin_totals:
        total_row += (
            f"{count:>12}"
        )

    total_row += (
        f"{bins_total:>12}"
        f"{total_check:>10}"
    )


    print(total_row)
    print()


    if (
        all_months_ok
        and total_check == "OK"
    ):

        print(
            "Partition check passed: for every month, "
            "full count = sum of lead-bin counts."
        )

    else:

        print(
            "WARNING: at least one month failed the "
            "lead-bin partition check."
        )


# =============================================================================
# Write NetCDF
# =============================================================================

def write_output(
    store,
    lead_bins,
):
    """Write the completed dataset to NetCDF."""

    filename_out = (
        make_output_filename(
            lead_bins
        )
    )


    os.makedirs(
        path_out,
        exist_ok=True,
    )


    # Compress precipitation variables.
    precipitation_variables = [
        full_range_variable_name()
    ] + [
        lead_bin_variable_name(
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for (
            lead_start,
            lead_end,
        ) in lead_bins
    ]


    encoding = {
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable_name
        in precipitation_variables
    }


    store.to_netcdf(
        filename_out,
        encoding=encoding,
    )


    print(
        "Wrote:",
        filename_out,
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()


    lead_bins = (
        build_lead_bins()
    )


    print(
        "Lead-time configuration"
    )
    print(
        "-----------------------"
    )

    print(
        f"Input daily leads: "
        f"{first_input_lead}-{last_input_lead}"
    )

    print(
        f"{x_days}-day accumulated ending leads: "
        f"{first_input_lead + x_days - 1}-{last_input_lead}"
    )

    print(
        f"Number of lead bins: "
        f"{number_of_lead_bins}"
    )

    print()
    print(
        "Full-window maximum variable:"
    )
    print(
        f"  {full_range_variable_name()}"
    )

    print()
    print(
        "Lead-location variables:"
    )


    for (
        lead_start,
        lead_end,
    ) in lead_bins:

        variable_name = (
            lead_bin_variable_name(
                lead_start=lead_start,
                lead_end=lead_end,
            )
        )

        print(
            f"  {variable_name}: "
            f"full-window maxima occurring at ending leads "
            f"{lead_start}-{lead_end}"
        )


    print()


    forecast_dates = (
        get_forecast_dates(
            forecast_date_range,
            option="mt",
        )
    )


    weights = (
        load_weights(
            filename_weights
        )
    )


    extreme_store = (
        initialize_extreme_store(
            n_forecasts=len(
                forecast_dates
            ),
            lead_bins=lead_bins,
        )
    )


    for date in forecast_dates:

        print(date)


        (
            forecast_filename,
            hindcast_filename,
        ) = get_model_filenames(
            date
        )


        (
            forecast,
            hindcast,
        ) = load_model_data(
            date
        )


        # ---------------------------------------------------------
        # Catchment-weighted precipitation
        # ---------------------------------------------------------

        forecast = catchment_mean(
            forecast,
            weights,
        )

        hindcast = catchment_mean(
            hindcast,
            weights,
        )


        # ---------------------------------------------------------
        # X-day accumulations
        # ---------------------------------------------------------

        forecast = xday_accumulation(
            forecast
        )

        hindcast = xday_accumulation(
            hindcast
        )


        # ---------------------------------------------------------
        # Calculate each full-window maximum ONCE.
        # ---------------------------------------------------------

        forecast_max_info = (
            extract_max_info(
                forecast
            )
        )

        hindcast_max_info = (
            extract_max_info(
                hindcast
            )
        )


        # ---------------------------------------------------------
        # Store full forecast sample and metadata.
        # ---------------------------------------------------------

        extreme_store = (
            add_full_max_info_to_store(
                store=extreme_store,
                max_info=forecast_max_info,
                forecast_date=date,
                source_file=forecast_filename,
                model_type="forecast",
            )
        )


        # Split those SAME forecast maxima by lead of maximum.
        extreme_store = (
            add_lead_bin_values_to_store(
                store=extreme_store,
                max_info=forecast_max_info,
                lead_bins=lead_bins,
            )
        )


        # ---------------------------------------------------------
        # Store full hindcast sample and metadata.
        # ---------------------------------------------------------

        extreme_store = (
            add_full_max_info_to_store(
                store=extreme_store,
                max_info=hindcast_max_info,
                forecast_date=date,
                source_file=hindcast_filename,
                model_type="hindcast",
            )
        )


        # Split those SAME hindcast maxima by lead of maximum.
        extreme_store = (
            add_lead_bin_values_to_store(
                store=extreme_store,
                max_info=hindcast_max_info,
                lead_bins=lead_bins,
            )
        )


        # Explicitly release large arrays before moving to the next date.
        del forecast
        del hindcast
        del forecast_max_info
        del hindcast_max_info


    # Store the monthly sample counts as variables in the output dataset.
    extreme_store = (
        update_sample_counts(
            store=extreme_store,
            lead_bins=lead_bins,
        )
    )


    print_monthly_sample_counts(
        store=extreme_store,
        lead_bins=lead_bins,
    )


    if write2file:

        write_output(
            store=extreme_store,
            lead_bins=lead_bins,
        )
