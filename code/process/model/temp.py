"""
Create calendar-month S2S precipitation-maximum samples, with optional MM correction.

For raw, the script reads raw preprocessed model data, calculates the requested trailing
accumulation, and builds monthly-maximum samples without bias correction. For q, doy, ld,
and q_doy, it reads the corresponding already accumulated, bias-corrected model file and
builds samples without applying another correction. For mm_1step and mm_2step, it reads
raw preprocessed model data, builds the samples, and applies monthly multiplicative mean
correction against ERA5 or SeNorge. mm_1step applies only the reference correction;
mm_2step first corrects lead-bin means to the full-sample mean and then applies the
reference correction. All outputs retain the configured lead-bin split.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"
catchment = "regine_drammen"
forecast_date_range = ["2020-01-02", "2023-12-28"]
observation_years = ["1957", "2022"]
accumulation_days = 2
number_of_lead_bins = 2

# Options: "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
bias_correction_method = "mm_2step"

# Options: "senorge", "era5"
bias_correction_reference = "senorge"

input_filename_override = None
output_filename_override = None
write2file = True


# =============================================================================
# Fixed dataset settings
# =============================================================================

first_input_lead = 16
last_input_lead = 46

ERA5_VARIABLE = "tp24"
SENORGE_VARIABLE = "rr"
MONTHS = np.arange(1, 13, dtype="int8")

path_s2s = Path(config.dirs["s2s_processed"])


# =============================================================================
# Filenames and validation
# =============================================================================


def get_file_id(catchment_name):
    """Return the short catchment label used in S2S filenames."""
    if catchment_name.startswith("regine_"):
        return catchment_name.replace("regine_", "", 1)
    return catchment_name


def uses_raw_input():
    """Return True for methods that start from raw preprocessed model data."""
    return bias_correction_method in {"raw", "mm_1step", "mm_2step"}


def uses_mm_correction():
    """Return True for MM methods calculated inside this script."""
    return bias_correction_method in {"mm_1step", "mm_2step"}


def make_input_filename():
    """Return the raw or already bias-corrected preprocessed model filename."""
    if input_filename_override is not None:
        return Path(input_filename_override)

    stem = (
        f"preprocessed_model_{variable}_{get_file_id(catchment)}_"
        f"{forecast_date_range[0]}_{forecast_date_range[1]}"
    )
    if uses_raw_input():
        return path_s2s / f"{stem}.nc"

    return path_s2s / (
        f"{stem}_{accumulation_days}dayacc_bc_{bias_correction_method}_"
        f"{bias_correction_reference}.nc"
    )


def make_output_filename():
    """Return a compact output filename without lead-range or split labels."""
    if output_filename_override is not None:
        return Path(output_filename_override)

    if bias_correction_method == "raw":
        correction_label = "raw"
    else:
        correction_label = f"bc_{bias_correction_method}_{bias_correction_reference}"

    return path_s2s / (
        f"test2-monthly_max_samples_{variable}_{accumulation_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}_"
        f"{correction_label}.nc"
    )


def make_reference_filename():
    """Return the selected monthly-maximum reference filename."""
    if bias_correction_reference == "era5":
        return Path(
            f"{config.dirs['era5_processed']}monthly_max_samples_{ERA5_VARIABLE}_"
            f"{accumulation_days}dayacc_{catchment}_{observation_years[0]}-"
            f"{observation_years[1]}.nc"
        )

    return Path(
        f"{config.dirs['senorge_processed']}monthly_max_samples_{SENORGE_VARIABLE}_"
        f"{accumulation_days}dayacc_{catchment}_{observation_years[0]}-"
        f"{observation_years[1]}.nc"
    )


def validate_user_settings():
    """Validate settings and required input files."""
    valid_methods = {"raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"}
    valid_references = {"senorge", "era5"}

    if bias_correction_method not in valid_methods:
        raise ValueError(f"bias_correction_method must be one of {sorted(valid_methods)}.")
    if bias_correction_method != "raw" and bias_correction_reference not in valid_references:
        raise ValueError(f"bias_correction_reference must be one of {sorted(valid_references)}.")
    if accumulation_days < 1:
        raise ValueError("accumulation_days must be at least 1.")
    if first_input_lead > last_input_lead:
        raise ValueError("first_input_lead must not exceed last_input_lead.")

    first_usable_lead = first_input_lead + accumulation_days - 1
    if first_usable_lead > last_input_lead:
        raise ValueError("accumulation_days is too large for the input lead range.")

    number_of_usable_leads = last_input_lead - first_usable_lead + 1
    if not isinstance(number_of_lead_bins, int):
        raise TypeError("number_of_lead_bins must be an integer.")
    if not 1 <= number_of_lead_bins <= number_of_usable_leads:
        raise ValueError("number_of_lead_bins must be between 1 and the number of usable leads.")

    filename = make_input_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Preprocessed input file not found: {filename}")

    if uses_mm_correction() and not make_reference_filename().is_file():
        raise FileNotFoundError(f"Reference file not found: {make_reference_filename()}")


def validate_input_dataset(ds):
    """Check that the preprocessed model input has the required structure."""
    required_variables = {variable, "f_date", "model_type", "hdate"}
    missing_variables = required_variables - set(ds.variables)
    if missing_variables:
        raise ValueError(f"Input file is missing variables: {sorted(missing_variables)}")

    required_dimensions = {"lead_day", "number", "i_date"}
    missing_dimensions = required_dimensions - set(ds.dims)
    if missing_dimensions:
        raise ValueError(f"Input file is missing dimensions: {sorted(missing_dimensions)}")

    expected_leads = np.arange(first_input_lead, last_input_lead + 1, dtype="int64")
    if not np.array_equal(ds["lead_day"].values, expected_leads):
        raise ValueError(
            f"The input lead_day coordinate does not exactly match "
            f"{first_input_lead}-{last_input_lead}."
        )

    if set(ds[variable].dims) != {"lead_day", "number", "i_date"}:
        raise ValueError(f"{variable} must have dimensions lead_day, number, and i_date.")
    if set(ds["f_date"].dims) != {"i_date", "lead_day"}:
        raise ValueError("f_date must have dimensions i_date and lead_day.")


# =============================================================================
# Lead bins and accumulation
# =============================================================================


def split_usable_leads(first_lead, last_lead, number_of_bins):
    """Split an inclusive lead interval into consecutive near-equal bins."""
    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder) for index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead
    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1
    return bins


def build_lead_bins():
    """Build lead bins after accounting for accumulation length."""
    return split_usable_leads(
        first_input_lead + accumulation_days - 1, last_input_lead, number_of_lead_bins
    )


def lead_bin_variable_name(lead_start, lead_end):
    """Return the output variable name for one lead-location bin."""
    return f"tp24_max_lead{lead_start}_{lead_end}"


def calculate_accumulation(ds):
    """Return precipitation and dates over usable accumulated ending leads."""
    tp24 = ds[variable].transpose("lead_day", "number", "i_date")
    first_usable_lead = first_input_lead + accumulation_days - 1
    usable_leads = np.arange(first_usable_lead, last_input_lead + 1, dtype="int64")

    if uses_raw_input():
        accumulated = tp24.rolling(
            lead_day=accumulation_days, min_periods=accumulation_days
        ).sum().sel(lead_day=usable_leads)
    else:
        accumulated = tp24.sel(lead_day=usable_leads)

    usable_f_dates = ds["f_date"].transpose("i_date", "lead_day").sel(lead_day=usable_leads)
    return accumulated, usable_f_dates


# =============================================================================
# Sample-month assignment
# =============================================================================


def decode_hdate_yyyymmdd(hdate_values):
    """Decode integer YYYYMMDD hdate values; 0 denotes forecast rows."""
    values = np.asarray(hdate_values)
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("hdate must contain integer YYYYMMDD values.")

    encoded = values.astype("int64")
    decoded = np.full(encoded.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for index in np.flatnonzero(encoded != 0):
        date_code = f"{encoded[index]:08d}"
        try:
            decoded[index] = np.datetime64(
                f"{date_code[:4]}-{date_code[4:6]}-{date_code[6:8]}", "ns"
            )
        except ValueError as exc:
            raise ValueError(
                f"Cannot decode hdate value {encoded[index]} as a valid YYYYMMDD date."
            ) from exc
    return decoded


def calculate_true_daily_valid_dates(input_f_dates, model_type, hdate, i_date):
    """Return true daily valid dates for every forecast and hindcast initialization."""
    daily_f_dates = input_f_dates.transpose("i_date", "lead_day").sel(
        lead_day=slice(first_input_lead, last_input_lead)
    )
    true_dates = np.asarray(daily_f_dates.values).astype("datetime64[ns]")
    model_types = np.char.lower(np.asarray(model_type.values).astype(str))
    nominal_initializations = np.asarray(i_date.values).astype("datetime64[ns]")
    decoded_hdates = decode_hdate_yyyymmdd(hdate.values)

    forecast_rows = model_types == "forecast"
    hindcast_rows = model_types == "hindcast"
    unknown_rows = ~(forecast_rows | hindcast_rows)
    if np.any(unknown_rows):
        unknown = sorted(set(np.asarray(model_type.values).astype(str)[unknown_rows]))
        raise ValueError(f"Unsupported model_type values: {unknown}")
    if np.any(np.isnat(decoded_hdates[hindcast_rows])):
        raise ValueError("One or more hindcast rows have missing decoded hdate values.")

    offsets = decoded_hdates[hindcast_rows] - nominal_initializations[hindcast_rows]
    true_dates[hindcast_rows] += offsets[:, np.newaxis]
    return xr.DataArray(
        true_dates,
        dims=("i_date", "lead_day"),
        coords={"i_date": daily_f_dates["i_date"], "lead_day": daily_f_dates["lead_day"]},
        name="true_f_date",
    )


def majority_sample_month(valid_dates):
    """Return the strict-majority YYYYMM value from the daily valid dates."""
    finite_dates = valid_dates[~np.isnat(valid_dates)]
    expected_size = last_input_lead - first_input_lead + 1
    if finite_dates.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} valid dates for lead days "
            f"{first_input_lead}-{last_input_lead}, found {finite_dates.size}."
        )

    months_since_epoch = finite_dates.astype("datetime64[M]").astype("int64")
    sample_months = 100 * (months_since_epoch // 12 + 1970) + months_since_epoch % 12 + 1
    unique_values, counts = np.unique(sample_months, return_counts=True)
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("sample_month assignment produced an unexpected tie.")
    return int(unique_values[np.argmax(counts)])


def calculate_sample_month(input_f_dates, model_type, hdate, i_date):
    """Assign one strict-majority YYYYMM sample month to every i_date."""
    true_daily_dates = calculate_true_daily_valid_dates(
        input_f_dates, model_type, hdate, i_date
    )
    sample_months = np.array(
        [
            majority_sample_month(true_daily_dates.isel(i_date=index).values)
            for index in range(true_daily_dates.sizes["i_date"])
        ],
        dtype="int32",
    )
    return xr.DataArray(
        sample_months,
        dims=("i_date",),
        coords={"i_date": true_daily_dates["i_date"]},
        name="sample_month",
    )


# =============================================================================
# Maximum extraction and lead-bin partition
# =============================================================================


def extract_full_window_maximum(accumulated_tp24, usable_f_dates):
    """Extract maximum precipitation, ending lead, and valid date for each sample."""
    finite = np.isfinite(accumulated_tp24)
    has_valid_value = finite.any(dim="lead_day")
    index_of_max = accumulated_tp24.where(finite, other=-np.inf).argmax(dim="lead_day")

    tp24_max = accumulated_tp24.max(dim="lead_day", skipna=True).where(has_valid_value)
    lead_of_max = accumulated_tp24["lead_day"].isel(lead_day=index_of_max).where(has_valid_value)
    date_of_max = usable_f_dates.broadcast_like(accumulated_tp24).isel(
        lead_day=index_of_max
    ).where(has_valid_value)

    tp24_max.name = "tp24_max"
    lead_of_max.name = "lead_of_max"
    date_of_max.name = "date_of_max"
    return tp24_max, lead_of_max, date_of_max


def make_lead_bin_variables(tp24_max, lead_of_max, lead_bins):
    """Partition each full-window maximum into exactly one ending-lead bin."""
    output = {}
    assignment_count = xr.zeros_like(tp24_max, dtype="int8")

    for lead_start, lead_end in lead_bins:
        in_bin = (lead_of_max >= lead_start) & (lead_of_max <= lead_end)
        output[lead_bin_variable_name(lead_start, lead_end)] = tp24_max.where(in_bin)
        assignment_count += xr.where(in_bin, 1, 0)

    valid_full = np.isfinite(tp24_max)
    invalid = (valid_full & (assignment_count != 1)) | (~valid_full & (assignment_count != 0))
    if bool(invalid.any().values):
        raise ValueError("Lead-bin variables do not form an exact partition of tp24_max.")
    return output


# =============================================================================
# Base sample dataset
# =============================================================================


def build_sample_dataset(input_ds, tp24_max, lead_of_max, date_of_max, sample_month, lead_bins):
    """Build the compact monthly-maximum sample dataset before optional MM correction."""
    lead_bin_variables = make_lead_bin_variables(tp24_max, lead_of_max, lead_bins)
    output = xr.Dataset(
        data_vars={
            "tp24_max": tp24_max.astype("float32"),
            "date_of_max": date_of_max,
            "lead_of_max": lead_of_max.astype("float32"),
            "sample_month": sample_month,
            "model_type": input_ds["model_type"],
            "hdate": input_ds["hdate"],
            **{name: values.astype("float32") for name, values in lead_bin_variables.items()},
        },
        coords={"number": input_ds["number"], "i_date": input_ds["i_date"]},
    )

    first_usable_lead = first_input_lead + accumulation_days - 1
    output["tp24_max"].attrs.update(
        {
            "units": "mm",
            "description": (
                f"Maximum {accumulation_days}-day accumulated precipitation over ending "
                f"lead days {first_usable_lead}-{last_input_lead}"
            ),
            "lead_start": first_usable_lead,
            "lead_end": last_input_lead,
        }
    )
    output["sample_month"].attrs.update(
        {
            "description": (
                "Calendar year-month containing the largest number of true daily valid "
                f"dates across input lead days {first_input_lead}-{last_input_lead}"
            ),
            "format": "YYYYMM",
            "assignment_window": (
                f"Fixed N=1 daily valid-date window for lead days "
                f"{first_input_lead}-{last_input_lead}"
            ),
        }
    )
    output["date_of_max"].attrs["description"] = "Calendar valid date on which tp24_max occurs"
    output["lead_of_max"].attrs.update(
        {"description": "Accumulated ending lead day on which tp24_max occurs", "units": "days"}
    )
    output["hdate"].attrs["description"] = (
        "Original hindcast initialization as YYYYMMDD; 0 for forecast rows"
    )

    for bin_number, (lead_start, lead_end) in enumerate(lead_bins, start=1):
        name = lead_bin_variable_name(lead_start, lead_end)
        output[name].attrs.update(
            {
                "units": "mm",
                "description": (
                    "Subset of complete-window tp24_max values whose maximum occurs at "
                    f"ending leads {lead_start}-{lead_end}; lead bin {bin_number} of "
                    f"{number_of_lead_bins}"
                ),
                "lead_start": lead_start,
                "lead_end": lead_end,
                "range_type": f"lead-location bin {bin_number} of {number_of_lead_bins}",
            }
        )

    output.attrs.update(
        {
            "description": "Monthly S2S precipitation-maximum samples",
            "source_file": str(make_input_filename()),
            "variable": variable,
            "catchment": catchment,
            "forecast_initialization_start": forecast_date_range[0],
            "forecast_initialization_end": forecast_date_range[1],
            "accumulation_days": accumulation_days,
            "first_input_lead": first_input_lead,
            "last_input_lead": last_input_lead,
            "first_usable_accumulated_lead": first_usable_lead,
            "last_usable_accumulated_lead": last_input_lead,
            "number_of_lead_bins": number_of_lead_bins,
            "lead_bin_sampling": (
                "Complete-window maxima partitioned by ending lead day of maximum; maxima "
                "are not recalculated within lead bins"
            ),
        }
    )
    return output


# =============================================================================
# MM reference data and monthly statistics
# =============================================================================


def load_reference_dataset():
    """Open the selected MM reference dataset and return its precipitation variable."""
    filename = make_reference_filename()
    variable_name = ERA5_VARIABLE if bias_correction_reference == "era5" else SENORGE_VARIABLE
    reference_ds = xr.open_dataset(filename)
    if variable_name not in reference_ds:
        reference_ds.close()
        raise KeyError(f"Variable '{variable_name}' was not found in {filename}.")
    return reference_ds, variable_name, filename


def get_reference_monthly_mean(reference_ds, reference_variable):
    """Calculate one reference mean for every calendar month."""
    values = reference_ds[reference_variable]
    if not {"year", "month"}.issubset(values.dims):
        raise ValueError(
            f"Reference variable '{reference_variable}' must contain dimensions 'year' and 'month'."
        )

    monthly_mean = values.mean(dim="year", skipna=True).sel(month=MONTHS)
    monthly_mean.name = "reference_monthly_mean"
    monthly_mean.attrs.update(
        {
            "reference_year_start": int(values["year"].min().values),
            "reference_year_end": int(values["year"].max().values),
        }
    )
    return monthly_mean


def get_calendar_month(model_ds):
    """Return calendar month 1-12 derived from sample_month YYYYMM."""
    return (model_ds["sample_month"].astype("int64") % 100).rename("calendar_month")


def calculate_model_monthly_mean(values, month_coordinate):
    """Calculate pooled model means across number and i_date for each calendar month."""
    monthly_means = [
        values.where(month_coordinate == month).mean(dim=("number", "i_date"), skipna=True)
        for month in MONTHS
    ]
    return xr.concat(monthly_means, dim=xr.DataArray(MONTHS, dims=("month",), name="month"))


def check_monthly_means(monthly_mean, label):
    """Require finite positive means in every calendar month."""
    if np.any(~np.isfinite(monthly_mean.values)):
        raise ValueError(f"At least one monthly mean is non-finite for {label}.")
    if np.any(monthly_mean.values <= 0):
        raise ValueError(f"At least one monthly mean is zero or negative for {label}.")


def expand_monthly_ratio_to_i_date(ratio, month_coordinate):
    """Expand ratio(month) to ratio(i_date) for broadcasting across ensemble members."""
    ratio_by_i_date = ratio.sel(month=month_coordinate)
    if "month" in ratio_by_i_date.coords:
        ratio_by_i_date = ratio_by_i_date.drop_vars("month")
    return ratio_by_i_date


# =============================================================================
# MM Stage 1: lead-time correction
# =============================================================================


def calculate_lead_time_correction_ratios(model_ds, split_variables):
    """Calculate original full-mean / split-mean ratios for every month and lead bin."""
    month_coordinate = get_calendar_month(model_ds)
    full_monthly_mean = calculate_model_monthly_mean(model_ds["tp24_max"], month_coordinate)
    check_monthly_means(full_monthly_mean, "tp24_max")

    ratios = {}
    for variable_name in split_variables:
        split_monthly_mean = calculate_model_monthly_mean(model_ds[variable_name], month_coordinate)
        check_monthly_means(split_monthly_mean, variable_name)
        ratio = full_monthly_mean / split_monthly_mean
        lead_label = variable_name.replace("tp24_max_", "", 1)
        ratio.name = f"lead_time_bias_correction_ratio_{lead_label}"
        ratio.attrs.update(
            {
                "description": (
                    "Monthly multiplicative lead-time correction ratio calculated as the "
                    "original complete-window monthly mean divided by the original split mean"
                ),
                "formula": "original_full_monthly_mean / original_split_monthly_mean",
                "source_variable": variable_name,
                "target_variable": "tp24_max",
                "units": "1",
            }
        )
        ratios[variable_name] = ratio
    return ratios


def apply_lead_time_correction(model_ds, split_variables, lead_time_ratios):
    """Apply monthly lead-time ratios while preserving each split's original mask."""
    month_coordinate = get_calendar_month(model_ds)
    corrected_splits = {}
    for variable_name in split_variables:
        ratio_by_i_date = expand_monthly_ratio_to_i_date(
            lead_time_ratios[variable_name], month_coordinate
        )
        corrected = (model_ds[variable_name] * ratio_by_i_date).transpose(
            *model_ds[variable_name].dims
        )
        corrected.attrs = model_ds[variable_name].attrs.copy()
        corrected_splits[variable_name] = corrected
    return corrected_splits


def rebuild_full_sample(model_ds, split_variables, corrected_splits):
    """Rebuild the full sample position-by-position from corrected lead-bin values."""
    original_full = model_ds["tp24_max"]
    finite_split_count = xr.zeros_like(original_full, dtype="int16")
    for variable_name in split_variables:
        finite_split_count += xr.where(np.isfinite(model_ds[variable_name]), 1, 0)

    full_finite = np.isfinite(original_full)
    invalid = (full_finite & (finite_split_count != 1)) | (~full_finite & (finite_split_count != 0))
    if bool(invalid.any().values):
        raise ValueError(
            "Lead-bin variables do not form an exact position-aligned partition of tp24_max."
        )

    rebuilt = xr.full_like(original_full, np.nan, dtype="float64")
    for variable_name in split_variables:
        corrected_split = corrected_splits[variable_name]
        rebuilt = xr.where(np.isfinite(corrected_split), corrected_split, rebuilt)

    rebuilt = rebuilt.transpose(*original_full.dims)
    rebuilt.name = "tp24_max"
    rebuilt.attrs = original_full.attrs.copy()
    return rebuilt


# =============================================================================
# MM Stage 2: reference correction
# =============================================================================


def calculate_reference_ratio(model_ds, full_sample, reference_monthly_mean):
    """Calculate reference mean / model full-sample mean for every calendar month."""
    model_monthly_mean = calculate_model_monthly_mean(full_sample, get_calendar_month(model_ds))
    check_monthly_means(model_monthly_mean, "full sample used for reference correction")

    reference_mean = reference_monthly_mean.sel(month=MONTHS)
    check_monthly_means(reference_mean, "reference sample")
    ratio = reference_mean / model_monthly_mean
    ratio.name = "bias_correction_ratio"
    ratio.attrs.update(
        {
            "description": "Monthly multiplicative reference correction ratio",
            "formula": "reference_monthly_mean / model_full_sample_monthly_mean",
            "reference_dataset": bias_correction_reference,
            "reference_year_start": reference_monthly_mean.attrs["reference_year_start"],
            "reference_year_end": reference_monthly_mean.attrs["reference_year_end"],
            "units": "1",
        }
    )
    return ratio


def apply_mm_correction(model_ds, reference_monthly_mean, split_variables):
    """Apply mm_1step or mm_2step and retain all full and lead-bin sample variables."""
    if bias_correction_method == "mm_2step":
        lead_time_ratios = calculate_lead_time_correction_ratios(model_ds, split_variables)
        corrected_splits = apply_lead_time_correction(model_ds, split_variables, lead_time_ratios)
        full_for_reference = rebuild_full_sample(model_ds, split_variables, corrected_splits)
    else:
        lead_time_ratios = {}
        corrected_splits = {name: model_ds[name].astype("float64") for name in split_variables}
        full_for_reference = model_ds["tp24_max"].astype("float64")

    reference_ratio = calculate_reference_ratio(
        model_ds, full_for_reference, reference_monthly_mean
    )
    ratio_by_i_date = expand_monthly_ratio_to_i_date(
        reference_ratio, get_calendar_month(model_ds)
    )

    output = model_ds.copy(deep=True)
    output["tp24_max"] = (full_for_reference * ratio_by_i_date).transpose(
        *model_ds["tp24_max"].dims
    ).astype("float32")
    output["tp24_max"].attrs = model_ds["tp24_max"].attrs.copy()

    for variable_name in split_variables:
        output[variable_name] = (corrected_splits[variable_name] * ratio_by_i_date).transpose(
            *model_ds[variable_name].dims
        ).astype("float32")
        output[variable_name].attrs = model_ds[variable_name].attrs.copy()

    output["bias_correction_ratio"] = reference_ratio.rename(
        {"month": "month_of_year"}
    ).astype("float32")
    for ratio in lead_time_ratios.values():
        output[ratio.name] = ratio.rename({"month": "month_of_year"}).astype("float32")

    method_description = (
        "Monthly multiplicative reference correction only"
        if bias_correction_method == "mm_1step"
        else "Monthly multiplicative lead-time correction followed by reference correction"
    )
    output.attrs.update(
        {
            "bias_correction_method": bias_correction_method,
            "bias_correction": method_description,
            "bias_correction_reference_dataset": bias_correction_reference,
            "bias_correction_reference_year_start": reference_ratio.attrs["reference_year_start"],
            "bias_correction_reference_year_end": reference_ratio.attrs["reference_year_end"],
        }
    )
    return output


# =============================================================================
# Reporting and NetCDF writing
# =============================================================================


def print_summary(output, lead_bins):
    """Print a concise summary of the completed sample dataset."""
    print("\nOutput summary\n--------------")
    print(output)
    print(
        "\nUsable accumulated lead range:",
        f"{first_input_lead + accumulation_days - 1}-{last_input_lead}",
    )
    print("Lead bins:", lead_bins)


def write_output(output, filename, lead_bin_variables):
    """Write the completed sample dataset to NetCDF."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    precipitation_variables = ["tp24_max", *lead_bin_variables]
    ratio_variables = [
        name
        for name in output.data_vars
        if name == "bias_correction_ratio" or name.startswith("lead_time_bias_correction_ratio_")
    ]

    encoding = {
        name: {
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
            "zlib": True,
            "complevel": 4,
        }
        for name in precipitation_variables
    }
    encoding.update(
        {
            name: {"dtype": "float32", "zlib": True, "complevel": 4}
            for name in ratio_variables
        }
    )
    encoding["lead_of_max"] = {"dtype": "float32", "_FillValue": np.float32(np.nan)}
    encoding["sample_month"] = {"dtype": "int32"}
    encoding["hdate"] = {"dtype": "int32"}

    output.to_netcdf(filename, encoding=encoding)
    print("\nWrote:", filename)


# =============================================================================
# Main
# =============================================================================


def main():
    """Build monthly-maximum samples and apply the selected correction workflow."""
    validate_user_settings()
    lead_bins = build_lead_bins()
    filename_input = make_input_filename()
    filename_output = make_output_filename()

    print("Reading:", filename_input)
    print("Writing:", filename_output)
    print("Bias-correction method:", bias_correction_method)
    if bias_correction_method != "raw":
        print("Bias-correction reference:", bias_correction_reference)
    print("Lead bins:", lead_bins)

    with xr.open_dataset(filename_input, decode_timedelta=False) as opened:
        input_ds = opened.load()
    validate_input_dataset(input_ds)

    accumulated_tp24, usable_f_dates = calculate_accumulation(input_ds)
    sample_month = calculate_sample_month(
        input_ds["f_date"], input_ds["model_type"], input_ds["hdate"], input_ds["i_date"]
    )
    tp24_max, lead_of_max, date_of_max = extract_full_window_maximum(
        accumulated_tp24, usable_f_dates
    )
    output = build_sample_dataset(
        input_ds, tp24_max, lead_of_max, date_of_max, sample_month, lead_bins
    )

    if uses_mm_correction():
        reference_ds, reference_variable, filename_reference = load_reference_dataset()
        print("Reading reference:", filename_reference)
        try:
            reference_monthly_mean = get_reference_monthly_mean(
                reference_ds, reference_variable
            )
            split_variables = [lead_bin_variable_name(*lead_bin) for lead_bin in lead_bins]
            output = apply_mm_correction(output, reference_monthly_mean, split_variables)
        finally:
            reference_ds.close()
    elif bias_correction_method == "raw":
        output.attrs.update(
            {
                "bias_correction_method": "raw",
                "bias_correction": "None; samples calculated directly from raw preprocessed data",
            }
        )
    else:
        output.attrs.update(
            {
                "bias_correction_method": bias_correction_method,
                "bias_correction_reference_dataset": bias_correction_reference,
                "bias_correction": "Applied upstream before monthly-sample extraction",
            }
        )

    print_summary(output, lead_bins)
    if write2file:
        write_output(
            output,
            filename_output,
            [lead_bin_variable_name(*lead_bin) for lead_bin in lead_bins],
        )


if __name__ == "__main__":
    main()
