"""
Build monthly S2S extreme-precipitation samples for the full lead window and
for N approximately equal lead-time bins.

Core logic
----------
For every forecast initialization, the script first calculates the complete
X-day accumulated precipitation series over the full usable lead window.

The calendar month is then assigned ONCE using that full usable lead window:
the assigned month is the month containing the largest number of dates in the
full window. The same assigned month is used for:

    - the maximum over the full lead window;
    - the maximum over lead-time split 1;
    - the maximum over lead-time split 2;
    - and every additional split.

This is deliberate. The lead-time subsets are therefore compared using the
same calendar-month grouping. A forecast cannot move from, for example,
January in the full sample to February in a later-lead subset simply because
that subset contains more February dates. The only quantity that changes
between the stored samples is the lead-time window over which the maximum is
calculated.

This makes differences between the lead-time samples easier to interpret as
lead-time effects rather than a combination of lead-time effects and changing
calendar-month membership.

Lead-time splitting
-------------------
The original daily input window is split into N consecutive, approximately
equal bins. If the number of daily input leads is not divisible by N, the
extra days are assigned to the later bins.

Example:
    first_input_lead = 16
    last_input_lead = 46
    x_days = 2
    number_of_lead_bins = 2

Daily input bins:
    16-30
    31-46

After the 2-day accumulation, the stored variables are:
    max_value_lead17_46   # maximum over the complete usable lead window
    max_value_lead17_30   # maximum over the first lead subset
    max_value_lead31_46   # maximum over the second lead subset

All three maxima from a given forecast/hindcast sample are stored in the same
calendar month: the month assigned from the complete lead-17-46 window.

The accumulation is calculated before selecting the lead-time subsets.
Therefore, for example, the 2-day accumulation ending on lead 31 uses daily
leads 30 and 31.

Output dataset
--------------
Dimensions:
    month_of_year : 1-12
    index         : pooled forecast/hindcast sample storage

Variables:
    One float32 max_value variable for the full lead window and one for each
    lead-time split. Dates, ensemble-member identifiers, model-type strings,
    and source filenames are not stored, keeping the output relatively small.
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

forecast_date_range = [
    "2020-01-02",
    "2023-06-26",
]

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

# Number of approximately equal lead-time subsets.
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

    number_of_input_leads = (
        last_input_lead - first_input_lead + 1
    )

    if not isinstance(number_of_lead_bins, int):
        raise TypeError(
            "number_of_lead_bins must be an integer."
        )

    if number_of_lead_bins < 1:
        raise ValueError(
            "number_of_lead_bins must be at least 1."
        )

    if number_of_lead_bins > number_of_input_leads:
        raise ValueError(
            "number_of_lead_bins cannot exceed the number of "
            "available input lead days."
        )

    first_usable_lead = first_input_lead + x_days - 1

    if first_usable_lead > last_input_lead:
        raise ValueError(
            "x_days is too large for the available input lead window."
        )


def split_input_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """
    Split an inclusive input lead interval into approximately equal bins.

    When the interval cannot be divided exactly, the extra lead days are
    assigned to the later bins. Thus 16-46 split into two bins becomes
    16-30 and 31-46.
    """

    number_of_leads = last_lead - first_lead + 1

    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    # Put the larger bins at the end, matching the requested 16-30 / 31-46
    # convention for 31 lead days split into two bins.
    bin_sizes = [
        base_size
        + int(
            bin_index
            >= number_of_bins - remainder
        )
        for bin_index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1

        bins.append(
            (current_start, current_end)
        )

        current_start = current_end + 1

    return bins


def build_accumulated_lead_ranges() -> list[tuple[int, int]]:
    """
    Return the full usable accumulated range followed by the N split ranges.

    Only the beginning of the first split is affected by the trailing
    accumulation because later accumulated ending leads already exist after
    the accumulation is calculated over the complete input period.
    """

    first_usable_lead = (
        first_input_lead + x_days - 1
    )

    input_bins = split_input_leads(
        first_lead=first_input_lead,
        last_lead=last_input_lead,
        number_of_bins=number_of_lead_bins,
    )

    split_ranges = []

    for bin_index, (bin_start, bin_end) in enumerate(input_bins):

        if bin_index == 0:
            accumulated_start = max(
                bin_start,
                first_usable_lead,
            )
        else:
            accumulated_start = bin_start

        accumulated_end = bin_end

        if accumulated_start > accumulated_end:
            raise ValueError(
                "A lead-time bin contains no usable accumulated leads. "
                "Reduce number_of_lead_bins or x_days."
            )

        split_ranges.append(
            (accumulated_start, accumulated_end)
        )

    full_range = (
        first_usable_lead,
        last_input_lead,
    )

    return [full_range] + split_ranges


def lead_range_variable_name(
    lead_start: int,
    lead_end: int,
) -> str:
    """Return the NetCDF variable name for one lead-time range."""

    return f"max_value_lead{lead_start}_{lead_end}"


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
            f"distribution_monthly_extremes_"
            f"{variable}_{x_days}dayacc_"
            f"nve_catchment_{catchment}_"
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
    """Return the calendar month containing most dates in the supplied window."""

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


def extract_max_values(
    da: xr.DataArray,
    lead_start: int,
    lead_end: int,
) -> np.ndarray:
    """
    Extract flattened ensemble/hindcast maxima for one accumulated lead range.

    Calendar-month assignment is intentionally not done here. It is calculated
    once from the full usable lead window and reused for every lead subset.
    """

    selected = select_lead_range(
        da=da,
        lead_start=lead_start,
        lead_end=lead_end,
    )

    max_values = (
        selected
        .max(dim="time")
        .values
        .astype("float32")
        .ravel()
    )

    return max_values[
        np.isfinite(max_values)
    ]


def get_fixed_month_assignment(
    da: xr.DataArray,
    full_lead_range: tuple[int, int],
) -> int:
    """
    Assign one calendar month from the complete usable lead window.

    This month is reused for the full-range maximum and every lead-time split,
    so changing the lead subset cannot move a sample between calendar months.
    """

    full_start, full_end = full_lead_range

    full_window = select_lead_range(
        da=da,
        lead_start=full_start,
        lead_end=full_end,
    )

    return get_month_with_most_lead_dates(
        full_window
    )


# =============================================================================
# Storage
# =============================================================================

def initialize_extreme_store(
    n_forecasts: int,
    lead_ranges: list[tuple[int, int]],
) -> xr.Dataset:
    """
    Create compact storage containing only maximum-value variables.

    Each lead range gets a separate float32 variable with dimensions
    (month_of_year, index).
    """

    n_index = n_forecasts * (
        n_forecast_members
        + n_hindcast_members * n_hdates
    )

    data_vars = {}

    for lead_start, lead_end in lead_ranges:
        variable_name = lead_range_variable_name(
            lead_start,
            lead_end,
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
    """Add compact metadata describing the sampling configuration."""

    store.attrs.update(
        {
            "description": (
                "Monthly pooled forecast/hindcast maxima for the complete "
                "usable lead window and approximately equal lead-time splits. "
                "Calendar month is assigned from the complete usable lead "
                "window and held fixed for all lead-time splits."
            ),
            "variable": variable,
            "catchment": catchment,
            "x_days": x_days,
            "first_input_lead": first_input_lead,
            "last_input_lead": last_input_lead,
            "number_of_lead_bins": number_of_lead_bins,
            "month_assignment": (
                "Month containing most dates in the complete usable lead "
                "window; same month reused for all lead-time splits."
            ),
            "forecast_date_start": forecast_date_range[0],
            "forecast_date_end": forecast_date_range[1],
        }
    )

    for range_index, (
        lead_start,
        lead_end,
    ) in enumerate(lead_ranges):

        variable_name = lead_range_variable_name(
            lead_start,
            lead_end,
        )

        if range_index == 0:
            range_type = "complete usable lead window"
        else:
            range_type = (
                f"lead-time split {range_index} of "
                f"{number_of_lead_bins}"
            )

        store[variable_name].attrs.update(
            {
                "description": (
                    f"Maximum {x_days}-day accumulated catchment-mean "
                    f"precipitation for ending leads "
                    f"{lead_start}-{lead_end}; {range_type}"
                ),
                "units": "mm",
                "lead_start": lead_start,
                "lead_end": lead_end,
                "range_type": range_type,
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
    Store maxima for the full range and every split using one fixed month.

    The month is determined from lead_ranges[0], which is always the complete
    usable lead window. All split maxima from this forecast/hindcast sample are
    then stored in that same month.
    """

    full_lead_range = lead_ranges[0]

    assigned_month = get_fixed_month_assignment(
        da=da,
        full_lead_range=full_lead_range,
    )

    for lead_start, lead_end in lead_ranges:

        variable_name = lead_range_variable_name(
            lead_start,
            lead_end,
        )

        max_values = extract_max_values(
            da=da,
            lead_start=lead_start,
            lead_end=lead_end,
        )

        add_values_to_store(
            store=store,
            variable_name=variable_name,
            month=assigned_month,
            values=max_values,
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
    print(
        "Month assignment: determined from the complete usable lead "
        "window and reused for every split."
    )

    print()
    print("Output maximum variables:")

    for range_index, (
        lead_start,
        lead_end,
    ) in enumerate(lead_ranges):

        variable_name = lead_range_variable_name(
            lead_start,
            lead_end,
        )

        label = (
            "full range"
            if range_index == 0
            else f"split {range_index}"
        )

        print(
            f"  {variable_name}: "
            f"leads {lead_start}-{lead_end} ({label})"
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

    if write2file:
        write_output(
            store=extreme_store,
            lead_ranges=lead_ranges,
        )
