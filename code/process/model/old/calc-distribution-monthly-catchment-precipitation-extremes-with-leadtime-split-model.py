"""
Build monthly S2S extreme-precipitation samples for the complete usable lead
window and partition those SAME full-window maxima according to the lead-time
bin in which each maximum occurs.

Sampling procedure
------------------
The script follows the original UNSEEN sampling approach:
1. Load forecast and hindcast daily precipitation.
2. Calculate catchment-weighted mean precipitation.
3. Calculate trailing X-day accumulated precipitation.
4. For every ensemble member (and hindcast date), calculate ONE maximum over
   the complete usable accumulated lead-time window.
5. Assign that full-window maximum to a calendar month using the calendar month
   containing most valid dates in the COMPLETE usable lead-time window.
6. Record the ending lead day at which that same full-window maximum occurs.
7. Place the full-window maximum into exactly one lead-time subgroup according
   to the lead bin containing its maximum lead day.

The lead-time subgroups are therefore NOT maxima recalculated over shorter
windows. They are subsets of the complete-window maxima. Every stored subgroup
value is exactly the same extreme value that appears in the full sample.

Consequently, for every calendar month:

    number of full-sample maxima
        =
    sum of the sample numbers across all lead-time bins

apart from cases where a full-window maximum or its lead day is missing.

Lead-time binning
-----------------
The usable accumulated ending leads are divided into N consecutive,
approximately equal bins. The splitting occurs AFTER accounting for the X-day
accumulation. If the number of usable accumulated leads is not divisible by N,
the extra leads are assigned to the later bins.

For:
    first_input_lead = 16
    last_input_lead  = 46
    x_days           = 2

the usable accumulated ending leads are 17-46, giving 30 usable lead times.

Examples:

    number_of_lead_bins = 1
        bin 1: 17-46

    number_of_lead_bins = 2
        bin 1: 17-31
        bin 2: 32-46

    number_of_lead_bins = 3
        bin 1: 17-26
        bin 2: 27-36
        bin 3: 37-46

    ...

    number_of_lead_bins = 30
        one ending lead per bin

For number_of_lead_bins = 2, a January full-window maximum that occurs at
lead 25 is stored in:
    max_value_lead17_46
and also in:
    max_value_bin1_lead17_31

A January full-window maximum that occurs at lead 38 is stored in:
    max_value_lead17_46
and also in:
    max_value_bin2_lead32_46

The subgroup values therefore partition the full UNSEEN sample according to
where within the forecast window the extreme occurred.

Tie handling
------------
If exactly the same maximum value occurs at more than one lead time for a
sample, the first occurrence in lead-time order is used to assign that sample
to a lead bin. The maximum value itself is unchanged.

Output dataset
--------------
Dimensions:
    month_of_year : 1-12
    index         : pooled forecast/hindcast sample storage

Variables:
    max_value_lead<start>_<end>
        Complete-window maximum sample.

    max_value_bin<N>_lead<start>_<end>
        Subset of complete-window maxima whose maximum occurs within that
        lead-time bin.

The script prints the sample number for the full sample and every lead bin for
each calendar month, plus a check that:

    full count = sum of lead-bin counts

for every month.
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

forecast_date_range = ["2020-01-02","2023-06-26"]

path_in_forecast = (
    config.dirs["s2s_forecast_daily"] + variable + "/"
)
path_in_hindcast = (
    config.dirs["s2s_hindcast_daily"] + variable + "/"
)

filename_weights = (
    config.dirs["nve"]
    + f"weights_catchment_{catchment}_era5_0.5x0.5.nc"
)

path_out = config.dirs["s2s_processed"]

# Daily lead times available in the original forecast/hindcast files.
first_input_lead = 16
last_input_lead = 46

# Number of lead-location bins used to partition the full-window maxima
number_of_lead_bins = 2

# Expected ensemble structure. These values only determine storage size.
n_forecast_members = 51
n_hindcast_members = 11
n_hdates = 20

write2file = True


# =============================================================================
# Lead-time configuration
# =============================================================================

def validate_user_settings() -> None:
    """Check settings before any files are opened."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    first_usable_lead = first_input_lead + x_days - 1
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


    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )


def split_usable_accumulated_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """
    Split the inclusive usable accumulated-lead interval into approximately
    equal consecutive bins.

    The split is performed AFTER accounting for the X-day accumulation.
    Therefore, with daily input leads 16-46 and x_days=2, the usable
    accumulated ending leads are 17-46 (30 lead times). Splitting these into
    two bins gives exactly 17-31 and 32-46.

    If the number of usable accumulated leads is not divisible by the number
    of bins, the extra leads are assigned to the later bins.
    """

    number_of_leads = last_lead - first_lead + 1

    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size
        + int(bin_index >= number_of_bins - remainder)
        for bin_index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1

    return bins


def build_accumulated_lead_ranges() -> list[tuple[int, int]]:
    """
    Return the full usable accumulated range followed by N lead-location bins.

    Importantly, the bins are created from the usable accumulated ending
    leads, not from the original daily input leads.

    Example:
        first_input_lead = 16
        last_input_lead  = 46
        x_days           = 2
        number_of_lead_bins = 2

    gives usable accumulated leads 17-46, which are split as:
        17-31
        32-46
    """

    first_usable_lead = first_input_lead + x_days - 1

    full_range = (
        first_usable_lead,
        last_input_lead,
    )

    split_ranges = split_usable_accumulated_leads(
        first_lead=first_usable_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )

    return [full_range] + split_ranges


def full_range_variable_name(
    full_range: tuple[int, int],
) -> str:
    """Return the NetCDF variable name for the complete-window maxima."""

    lead_start, lead_end = full_range
    return f"max_value_lead{lead_start}_{lead_end}"


def lead_bin_variable_name(
    bin_number: int,
    lead_start: int,
    lead_end: int,
) -> str:
    """Return the NetCDF variable name for one lead-location subgroup."""

    return (
        f"max_value_bin{bin_number}_"
        f"lead{lead_start}_{lead_end}"
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
        dates = mondays.union(thursdays)

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


def get_model_filenames(
    date: str,
) -> tuple[str, str]:
    """Return forecast and hindcast filenames for one initialization date."""

    forecast_filename = (
        path_in_forecast
        + f"{variable}_0.5x0.5_{date}.nc"
    )

    hindcast_filename = (
        path_in_hindcast
        + f"{variable}_0.5x0.5_{date}.nc"
    )

    return forecast_filename, hindcast_filename


def lead_split_filename_label(
    lead_ranges: list[tuple[int, int]],
) -> str:
    """
    Return a compact filename label describing the complete and split ranges.
    """

    full_start, full_end = lead_ranges[0]

    split_text = "_".join(
        f"{lead_start}-{lead_end}"
        for lead_start, lead_end in lead_ranges[1:]
    )

    return (
        f"lead{full_start}-{full_end}_"
        f"split{number_of_lead_bins}_"
        f"{split_text}"
    )


def make_output_filename(
    lead_ranges: list[tuple[int, int]],
) -> str:
    """Return an output filename that explicitly records all lead splits."""

    lead_label = lead_split_filename_label(
        lead_ranges
    )

    return os.path.join(
        path_out,
        (
            f"lt_maxima_distribution_monthly_extremes_"
            f"{variable}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_label}_"
            f"forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )


# =============================================================================
# Loading
# =============================================================================

def load_weights(
    filename: str,
) -> xr.DataArray:
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


def load_model_data(
    date: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Load forecast and hindcast daily precipitation.

    Precipitation in metres is converted to millimetres. Hindcast member
    labels are reset to 1-11, matching the original script.
    """

    (
        forecast_filename,
        hindcast_filename,
    ) = get_model_filenames(date)

    with xr.open_dataset(
        forecast_filename
    ) as ds_forecast:
        forecast = ds_forecast[variable].load()

    with xr.open_dataset(
        hindcast_filename
    ) as ds_hindcast:
        hindcast = ds_hindcast[variable].load()

    forecast = convert_to_mm(forecast)
    hindcast = convert_to_mm(hindcast)

    if "number" in hindcast.dims:
        expected_members = hindcast.sizes["number"]

        hindcast = hindcast.assign_coords(
            number=np.arange(
                1,
                expected_members + 1,
            )
        )

    return forecast, hindcast


def convert_to_mm(
    da: xr.DataArray,
) -> xr.DataArray:
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


# =============================================================================
# Processing
# =============================================================================

def catchment_mean(
    da: xr.DataArray,
    weights: xr.DataArray,
    spatial_dims=("latitude", "longitude"),
) -> xr.DataArray:
    """Calculate catchment-weighted spatial mean precipitation."""

    valid = (
        xr.ufuncs.isfinite(da)
        & xr.ufuncs.isfinite(weights)
        & (weights > 0)
    )

    weighted_sum = (
        da.where(valid)
        * weights.where(valid)
    ).sum(
        dim=spatial_dims,
        skipna=True,
    )

    weight_sum = weights.where(valid).sum(
        dim=spatial_dims,
        skipna=True,
    )

    out = weighted_sum / weight_sum

    out.attrs["description"] = (
        "Catchment-weighted daily mean precipitation"
    )
    out.attrs["units"] = da.attrs.get(
        "units",
        "",
    )

    return out


def xday_accumulation(
    da: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate trailing X-day precipitation and add an explicit lead_day axis.

    The original input is expected to contain daily leads
    first_input_lead ... last_input_lead.
    """

    expected_input_size = (
        last_input_lead - first_input_lead + 1
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
        first_input_lead + x_days - 1
    )

    expected_accumulated_leads = np.arange(
        first_usable_lead,
        last_input_lead + 1,
        dtype="int16",
    )

    if out.sizes["time"] != expected_accumulated_leads.size:
        raise ValueError(
            "Accumulated time dimension does not match the expected "
            "usable lead range."
        )

    out = out.assign_coords(
        lead_day=(
            "time",
            expected_accumulated_leads,
        )
    )

    out.attrs["description"] = (
        f"{x_days}-day accumulated catchment-weighted mean precipitation"
    )
    out.attrs["units"] = da.attrs.get(
        "units",
        "",
    )

    return out


def select_lead_range(
    da: xr.DataArray,
    lead_start: int,
    lead_end: int,
) -> xr.DataArray:
    """Select accumulated values whose ending lead is inside one range."""

    mask = (
        (da["lead_day"] >= lead_start)
        & (da["lead_day"] <= lead_end)
    )

    selected = da.where(
        mask,
        drop=True,
    )

    if selected.sizes["time"] == 0:
        raise ValueError(
            f"No accumulated data found for leads "
            f"{lead_start}-{lead_end}."
        )

    return selected


def get_month_with_most_lead_dates(
    da: xr.DataArray,
) -> int:
    """Return the calendar month containing most dates in one lead window."""

    months = da.time.dt.month

    month_counts = xr.DataArray(
        np.ones(
            len(months),
            dtype="int16",
        ),
        coords={
            "time": da.time,
            "month": (
                "time",
                months.data,
            ),
        },
        dims="time",
    ).groupby("month").sum()

    return int(
        month_counts
        .idxmax(dim="month")
        .values
    )


def extract_full_window_maxima_and_leads(
    da: xr.DataArray,
    full_range: tuple[int, int],
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Extract the complete-window maximum and its ending lead day for every sample.

    Returns
    -------
    month : int
        Calendar month assigned from the complete usable lead-time window.

    max_values : np.ndarray
        Flattened full-window maximum values.

    max_leads : np.ndarray
        Flattened ending lead day at which each full-window maximum occurs.

    Notes
    -----
    The lead day is found with argmax along the time dimension. If the same
    maximum occurs at multiple lead times, xarray/numpy argmax selects the
    first occurrence in lead-time order.
    """

    full_start, full_end = full_range

    selected = select_lead_range(
        da=da,
        lead_start=full_start,
        lead_end=full_end,
    )

    month = get_month_with_most_lead_dates(selected)

    # Maximum value over the complete lead window.
    max_da = selected.max(dim="time")

    # Index of the first occurrence of the maximum along time.
    argmax_da = selected.argmax(dim="time")

    # Convert the argmax indices into actual ending lead days.
    lead_days = selected["lead_day"].values
    argmax_indices = argmax_da.values.astype("int64")

    max_values = max_da.values.astype("float32").ravel()
    max_leads = lead_days[argmax_indices].astype("int16").ravel()

    valid = (
        np.isfinite(max_values)
        & np.isfinite(max_leads)
    )

    return (
        month,
        max_values[valid],
        max_leads[valid],
    )


def split_full_maxima_by_lead_bin(
    max_values: np.ndarray,
    max_leads: np.ndarray,
    split_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Partition complete-window maxima according to where the maximum occurred.

    Each full-window maximum is assigned to exactly one lead bin.
    """

    bin_values = []

    for lead_start, lead_end in split_ranges:
        in_bin = (
            (max_leads >= lead_start)
            & (max_leads <= lead_end)
        )

        bin_values.append(
            max_values[in_bin]
        )

    assigned_count = sum(values.size for values in bin_values)

    if assigned_count != max_values.size:
        raise ValueError(
            "Not every full-window maximum was assigned to exactly one "
            "lead-time bin. Check the lead-bin definitions."
        )

    return bin_values


# =============================================================================
# Storage
# =============================================================================

def initialize_extreme_store(
    n_forecasts: int,
    lead_ranges: list[tuple[int, int]],
) -> xr.Dataset:
    """
    Create storage for the complete-window maxima and lead-location subsets.

    The full sample and every lead bin use dimensions (month_of_year, index).
    """

    n_index = n_forecasts * (
        n_forecast_members
        + n_hindcast_members * n_hdates
    )

    full_range = lead_ranges[0]
    split_ranges = lead_ranges[1:]

    data_vars = {}

    full_variable = full_range_variable_name(full_range)
    data_vars[full_variable] = (
        ("month_of_year", "index"),
        np.full(
            (12, n_index),
            np.nan,
            dtype="float32",
        ),
    )

    for bin_number, (lead_start, lead_end) in enumerate(
        split_ranges,
        start=1,
    ):
        variable_name = lead_bin_variable_name(
            bin_number=bin_number,
            lead_start=lead_start,
            lead_end=lead_end,
        )

        data_vars[variable_name] = (
            ("month_of_year", "index"),
            np.full(
                (12, n_index),
                np.nan,
                dtype="float32",
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
        lead_ranges=lead_ranges,
    )

    return store


def add_store_metadata(
    store: xr.Dataset,
    lead_ranges: list[tuple[int, int]],
) -> None:
    """Add metadata describing the full-window lead-location sampling."""

    full_range = lead_ranges[0]
    split_ranges = lead_ranges[1:]
    full_start, full_end = full_range

    store.attrs.update(
        {
            "description": (
                "Monthly pooled forecast/hindcast maxima calculated over the "
                "complete usable lead window, plus subsets of those SAME "
                "maxima grouped by the lead-time bin in which each maximum "
                "occurs."
            ),
            "variable": variable,
            "catchment": catchment,
            "x_days": x_days,
            "first_input_lead": first_input_lead,
            "last_input_lead": last_input_lead,
            "first_usable_accumulated_lead": full_start,
            "last_usable_accumulated_lead": full_end,
            "number_of_lead_bins": number_of_lead_bins,
            "calendar_month_binning": "complete usable lead window",
            "lead_bin_sampling": (
                "complete-window maxima partitioned by lead day of maximum"
            ),
            "tie_handling": (
                "first lead-time occurrence used when equal maxima are tied"
            ),
            "forecast_date_start": forecast_date_range[0],
            "forecast_date_end": forecast_date_range[1],
        }
    )

    full_variable = full_range_variable_name(full_range)
    store[full_variable].attrs.update(
        {
            "description": (
                f"Maximum {x_days}-day accumulated catchment-mean "
                f"precipitation over complete ending leads "
                f"{full_start}-{full_end}"
            ),
            "units": "mm",
            "lead_start": full_start,
            "lead_end": full_end,
            "range_type": "complete usable lead window",
        }
    )

    for bin_number, (lead_start, lead_end) in enumerate(
        split_ranges,
        start=1,
    ):
        variable_name = lead_bin_variable_name(
            bin_number=bin_number,
            lead_start=lead_start,
            lead_end=lead_end,
        )

        store[variable_name].attrs.update(
            {
                "description": (
                    f"Subset of complete-window maxima whose maximum occurs "
                    f"at ending leads {lead_start}-{lead_end}; "
                    f"lead bin {bin_number} of {number_of_lead_bins}"
                ),
                "units": "mm",
                "lead_start": lead_start,
                "lead_end": lead_end,
                "range_type": (
                    f"lead-location bin {bin_number} of "
                    f"{number_of_lead_bins}"
                ),
            }
        )


def get_free_indices(
    store: xr.Dataset,
    variable_name: str,
    month: int,
    n_values: int,
) -> np.ndarray:
    """Find the first available block for one variable and calendar month."""

    current_values = store[
        variable_name
    ].sel(
        month_of_year=month
    ).values

    used = np.isfinite(
        current_values
    )

    if not np.any(~used):
        raise ValueError(
            f"No free storage slots left for {variable_name}, "
            f"month {month}."
        )

    first_free = np.where(
        ~used
    )[0][0]

    last_free = (
        first_free + n_values
    )

    if last_free > store.sizes["index"]:
        raise ValueError(
            f"Not enough storage for {variable_name}, month {month}. "
            f"Need {n_values} slots but only "
            f"{store.sizes['index'] - first_free} remain."
        )

    return store["index"].values[
        first_free:last_free
    ]


def add_values_to_store(
    store: xr.Dataset,
    variable_name: str,
    month: int,
    values: np.ndarray,
) -> None:
    """Append one flattened maximum-value sample to the selected month."""

    if values.size == 0:
        return

    index_values = get_free_indices(
        store=store,
        variable_name=variable_name,
        month=month,
        n_values=values.size,
    )

    store[variable_name].loc[
        dict(
            month_of_year=month,
            index=index_values,
        )
    ] = values


def process_and_store_all_ranges(
    store: xr.Dataset,
    da: xr.DataArray,
    lead_ranges: list[tuple[int, int]],
) -> None:
    """
    Store the complete-window maxima and partition them by maximum lead day.

    The complete-window maximum is calculated once for each sample. That same
    value is then stored in exactly one lead-bin variable according to the
    ending lead day at which the maximum occurs.
    """

    full_range = lead_ranges[0]
    split_ranges = lead_ranges[1:]

    (
        month,
        max_values,
        max_leads,
    ) = extract_full_window_maxima_and_leads(
        da=da,
        full_range=full_range,
    )

    # Store every complete-window maximum.
    full_variable = full_range_variable_name(full_range)

    add_values_to_store(
        store=store,
        variable_name=full_variable,
        month=month,
        values=max_values,
    )

    # Partition those same maxima among the lead-location bins.
    bin_values = split_full_maxima_by_lead_bin(
        max_values=max_values,
        max_leads=max_leads,
        split_ranges=split_ranges,
    )

    for bin_number, (
        (lead_start, lead_end),
        values,
    ) in enumerate(
        zip(split_ranges, bin_values),
        start=1,
    ):
        variable_name = lead_bin_variable_name(
            bin_number=bin_number,
            lead_start=lead_start,
            lead_end=lead_end,
        )

        add_values_to_store(
            store=store,
            variable_name=variable_name,
            month=month,
            values=values,
        )


def print_monthly_sample_counts(
    store: xr.Dataset,
    lead_ranges: list[tuple[int, int]],
) -> None:
    """
    Print monthly sample counts and verify that lead bins partition the full sample.
    """

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    full_range = lead_ranges[0]
    split_ranges = lead_ranges[1:]

    full_variable = full_range_variable_name(full_range)

    bin_variables = [
        lead_bin_variable_name(
            bin_number=bin_number,
            lead_start=lead_start,
            lead_end=lead_end,
        )
        for bin_number, (lead_start, lead_end) in enumerate(
            split_ranges,
            start=1,
        )
    ]

    labels = ["full"] + [
        f"bin_{bin_number}"
        for bin_number in range(1, len(split_ranges) + 1)
    ] + ["bins_sum", "check"]

    print()
    print("Sample counts by calendar month")
    print("-------------------------------")
    print(
        "Each lead-bin sample is a subset of the complete-window maxima."
    )

    header = f"{'Month':<12}" + "".join(
        f"{label:>12}" for label in labels
    )
    print(header)
    print("-" * len(header))

    all_months_ok = True

    for month, month_name in enumerate(month_names, start=1):

        full_count = int(
            np.isfinite(
                store[full_variable]
                .sel(month_of_year=month)
                .values
            ).sum()
        )

        bin_counts = []

        for variable_name in bin_variables:
            count = int(
                np.isfinite(
                    store[variable_name]
                    .sel(month_of_year=month)
                    .values
                ).sum()
            )
            bin_counts.append(count)

        bins_sum = sum(bin_counts)
        check = "OK" if bins_sum == full_count else "FAIL"

        if check == "FAIL":
            all_months_ok = False

        values_to_print = [full_count] + bin_counts + [bins_sum]

        row = f"{month_name:<12}" + "".join(
            f"{value:>12d}" for value in values_to_print
        ) + f"{check:>12}"

        print(row)

    full_total = int(
        np.isfinite(store[full_variable].values).sum()
    )

    bin_totals = [
        int(np.isfinite(store[variable_name].values).sum())
        for variable_name in bin_variables
    ]

    bins_total = sum(bin_totals)
    total_check = "OK" if bins_total == full_total else "FAIL"

    print("-" * len(header))

    total_values = [full_total] + bin_totals + [bins_total]

    print(
        f"{'TOTAL':<12}"
        + "".join(f"{value:>12d}" for value in total_values)
        + f"{total_check:>12}"
    )

    print()

    if all_months_ok and total_check == "OK":
        print(
            "Partition check passed: for every month, "
            "full count = sum of lead-bin counts."
        )
    else:
        print(
            "WARNING: at least one month failed the partition check."
        )


# =============================================================================
# Output
# =============================================================================

def write_output(
    store: xr.Dataset,
    lead_ranges: list[tuple[int, int]],
) -> None:
    """Write compact float32 output with NetCDF compression."""

    filename_out = make_output_filename(
        lead_ranges
    )

    os.makedirs(
        path_out,
        exist_ok=True,
    )

    encoding = {
        variable_name: {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
        }
        for variable_name in store.data_vars
    }

    store.to_netcdf(
        filename_out,
        encoding=encoding,
    )

    print("Wrote:", filename_out)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_user_settings()

    lead_ranges = build_accumulated_lead_ranges()

    print("Lead-time sampling")
    print("------------------")

    print(
        f"Input daily leads: "
        f"{first_input_lead}-{last_input_lead}"
    )

    print(
        f"{x_days}-day accumulation usable from lead "
        f"{first_input_lead + x_days - 1}"
    )

    print()
    print("Output maximum variables:")

    full_range = lead_ranges[0]
    split_ranges = lead_ranges[1:]

    full_variable = full_range_variable_name(full_range)

    print(
        f"  {full_variable}: "
        f"complete-window maxima over leads "
        f"{full_range[0]}-{full_range[1]}"
    )

    for bin_number, (lead_start, lead_end) in enumerate(
        split_ranges,
        start=1,
    ):
        variable_name = lead_bin_variable_name(
            bin_number=bin_number,
            lead_start=lead_start,
            lead_end=lead_end,
        )

        print(
            f"  {variable_name}: "
            f"full-window maxima occurring at leads "
            f"{lead_start}-{lead_end}"
        )

    forecast_dates = get_forecast_dates(
        forecast_date_range,
        option="mt",
    )

    weights = load_weights(
        filename_weights
    )

    extreme_store = initialize_extreme_store(
        n_forecasts=len(forecast_dates),
        lead_ranges=lead_ranges,
    )

    for date in forecast_dates:

        print()
        print(date)

        forecast, hindcast = load_model_data(
            date
        )

        forecast = catchment_mean(
            forecast,
            weights,
        )

        hindcast = catchment_mean(
            hindcast,
            weights,
        )

        forecast = xday_accumulation(
            forecast
        )

        hindcast = xday_accumulation(
            hindcast
        )

        process_and_store_all_ranges(
            store=extreme_store,
            da=forecast,
            lead_ranges=lead_ranges,
        )

        process_and_store_all_ranges(
            store=extreme_store,
            da=hindcast,
            lead_ranges=lead_ranges,
        )

        # Explicitly release the large per-initialization arrays before
        # continuing to the next forecast date.
        del forecast
        del hindcast

    print_monthly_sample_counts(
        store=extreme_store,
        lead_ranges=lead_ranges,
    )

    if write2file:
        write_output(
            store=extreme_store,
            lead_ranges=lead_ranges,
        )
