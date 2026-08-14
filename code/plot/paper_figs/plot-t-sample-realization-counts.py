#!/usr/bin/env python3
"""
Plot monthly realization counts from a compact S2S monthly-maximum sample file.

The input file must contain:

    sample_month(i_date)
    model_type(i_date)

sample_month is an integer YYYYMM label identifying the calendar year-month
assigned to each i_date. model_type identifies each i_date as either a forecast
or hindcast row.

The figure contains two panels, each showing Hindcast, Forecast, and All.

Panel a) plots the monthly YYYYMM time series:

    Hindcast
        HINDCAST_MEMBERS × hindcast i_date count in each YYYYMM.

    Forecast
        FORECAST_MEMBERS × forecast i_date count in each YYYYMM.

    All
        Hindcast + Forecast in each YYYYMM.

Panel b) aggregates the same counts by calendar month only, summing over all
years so the x-axis is January through December.

The ensemble sizes are source-specific because hindcast and forecast rows do not
use the same number of members.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

# Optional explicit compact sample filename. Leave as None to construct it.
INPUT_FILENAME_OVERRIDE = None

VARIABLE = "tp24"
CATCHMENT = "regine_drammen"
ACCUMULATION_DAYS = 2

FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]

FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2

# Compact sample source.
INPUT_DATA_TYPE = "raw"  # "raw" or "bias_corrected"
BIAS_CORRECTION_METHOD = "ld"  # "q", "doy", "ld", or "q_doy"
BIAS_CORRECTION_REFERENCE = "era5"  # "senorge" or "era5"

# Source-specific ensemble sizes.
HINDCAST_MEMBERS = 11
FORECAST_MEMBERS = 51

# Optional time limits. Use None for the complete sample_month range.
PLOT_START_MONTH = None  # e.g. 200001
PLOT_END_MONTH = None  # e.g. 202212

FIG_WIDTH_IN = 14
FIG_HEIGHT_IN = 5.5
FIGURE_DPI = 400

LINEWIDTH = 1.7
MARKER_SIZE = 4.0

SERIES_COLORS = {
    "All": "tab:blue",
    "Forecast": "tab:orange",
    "Hindcast": "tab:green",
}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

TITLE_FONTSIZE = 13
AXIS_LABELSIZE = 12
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 10

SHOW_GRID = True
WRITE_TO_FILE = False
SHOW_FIGURE = True


# =============================================================================
# Filename helpers
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in compact filenames."""

    return catchment_name.removeprefix("regine_")


def split_usable_leads(first_lead, last_lead, number_of_bins):
    """Split an inclusive lead interval into consecutive near-equal bins."""

    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder)
        for index in range(number_of_bins)
    ]

    bins = []
    current_start = first_lead

    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        bins.append((current_start, current_end))
        current_start = current_end + 1

    return bins


def build_lead_bins():
    """Return the usable accumulated lead bins encoded in the filename."""

    first_usable_lead = FIRST_INPUT_LEAD + ACCUMULATION_DAYS - 1
    return split_usable_leads(
        first_usable_lead,
        LAST_INPUT_LEAD,
        NUMBER_OF_LEAD_BINS,
    )


def make_input_filename():
    """Construct the compact monthly-maximum input filename."""

    if INPUT_FILENAME_OVERRIDE is not None:
        return Path(INPUT_FILENAME_OVERRIDE)

    first_usable_lead = FIRST_INPUT_LEAD + ACCUMULATION_DAYS - 1
    bin_label = "_".join(f"{start}-{end}" for start, end in build_lead_bins())

    stem = (
        f"test-monthly_max_samples_{VARIABLE}_{ACCUMULATION_DAYS}dayacc_"
        f"{get_file_id(CATCHMENT)}_lead{first_usable_lead}-{LAST_INPUT_LEAD}_"
        f"split{NUMBER_OF_LEAD_BINS}_{bin_label}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}"
    )

    if INPUT_DATA_TYPE == "bias_corrected":
        stem += f"_bc_{BIAS_CORRECTION_METHOD}_{BIAS_CORRECTION_REFERENCE}"

    return Path(config.dirs["s2s_processed"]) / f"{stem}.nc"


def make_figure_filename():
    """Construct the output figure filename."""

    source = "raw"
    if INPUT_DATA_TYPE == "bias_corrected":
        source = f"bc-{BIAS_CORRECTION_METHOD}-{BIAS_CORRECTION_REFERENCE}"

    filename = (
        f"t_sample_month_realization_counts_{VARIABLE}_{ACCUMULATION_DAYS}dayacc_"
        f"{get_file_id(CATCHMENT)}_{source}.png"
    )
    return Path(config.dirs["fig"]) / filename


# =============================================================================
# Validation and reading
# =============================================================================

def validate_user_settings():
    """Validate user-configurable settings."""

    if HINDCAST_MEMBERS < 1 or FORECAST_MEMBERS < 1:
        raise ValueError("HINDCAST_MEMBERS and FORECAST_MEMBERS must be at least 1.")

    if ACCUMULATION_DAYS < 1:
        raise ValueError("ACCUMULATION_DAYS must be at least 1.")

    if FIRST_INPUT_LEAD > LAST_INPUT_LEAD:
        raise ValueError("FIRST_INPUT_LEAD must not exceed LAST_INPUT_LEAD.")

    first_usable_lead = FIRST_INPUT_LEAD + ACCUMULATION_DAYS - 1
    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError("ACCUMULATION_DAYS is too large for the input lead range.")

    usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1
    if not isinstance(NUMBER_OF_LEAD_BINS, int) or not 1 <= NUMBER_OF_LEAD_BINS <= usable_leads:
        raise ValueError("NUMBER_OF_LEAD_BINS is invalid for the usable lead range.")

    if INPUT_DATA_TYPE not in {"raw", "bias_corrected"}:
        raise ValueError("INPUT_DATA_TYPE must be 'raw' or 'bias_corrected'.")

    if INPUT_DATA_TYPE == "bias_corrected":
        if BIAS_CORRECTION_METHOD not in {"q", "doy", "ld", "q_doy"}:
            raise ValueError("BIAS_CORRECTION_METHOD must be 'q', 'doy', 'ld', or 'q_doy'.")
        if BIAS_CORRECTION_REFERENCE not in {"senorge", "era5"}:
            raise ValueError("BIAS_CORRECTION_REFERENCE must be 'senorge' or 'era5'.")

    for name, value in {
        "PLOT_START_MONTH": PLOT_START_MONTH,
        "PLOT_END_MONTH": PLOT_END_MONTH,
    }.items():
        if value is not None:
            validate_sample_month_value(value, name)

    if PLOT_START_MONTH is not None and PLOT_END_MONTH is not None:
        if PLOT_END_MONTH < PLOT_START_MONTH:
            raise ValueError("PLOT_END_MONTH must not precede PLOT_START_MONTH.")

    filename = make_input_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")


def validate_sample_month_value(value, name):
    """Validate one YYYYMM integer."""

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer YYYYMM value.")

    year = int(value) // 100
    month = int(value) % 100

    if year < 1 or month not in range(1, 13):
        raise ValueError(f"{name} must use valid YYYYMM format.")


def read_sample_metadata():
    """Read sample_month and model_type from the compact sample file."""

    filename = make_input_filename()

    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        required = {"sample_month", "model_type"}
        missing = required - set(ds.variables)
        if missing:
            raise KeyError(f"Input file is missing variables: {sorted(missing)}")

        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("sample_month must have dimension ('i_date',).")

        if ds["model_type"].dims != ("i_date",):
            raise ValueError("model_type must have dimension ('i_date',).")

        sample_month = np.asarray(ds["sample_month"].load().values, dtype="int64")
        model_type = np.char.lower(np.asarray(ds["model_type"].load().values).astype(str))

    valid_types = {"forecast", "hindcast"}
    unknown = sorted(set(model_type) - valid_types)
    if unknown:
        raise ValueError(f"Unsupported model_type values: {unknown}")

    for value in np.unique(sample_month):
        validate_sample_month_value(int(value), "sample_month")

    return sample_month, model_type


# =============================================================================
# Monthly aggregation
# =============================================================================

def sample_month_to_period(sample_month):
    """Convert one YYYYMM integer to a pandas monthly Period."""

    year = int(sample_month) // 100
    month = int(sample_month) % 100
    return pd.Period(year=year, month=month, freq="M")


def calculate_monthly_counts(sample_month, model_type):
    """Calculate forecast, hindcast, and combined realization counts."""

    periods = pd.PeriodIndex([sample_month_to_period(value) for value in sample_month])

    first_month = periods.min()
    last_month = periods.max()
    all_months = pd.period_range(first_month, last_month, freq="M")

    if PLOT_START_MONTH is not None:
        all_months = all_months[all_months >= sample_month_to_period(PLOT_START_MONTH)]

    if PLOT_END_MONTH is not None:
        all_months = all_months[all_months <= sample_month_to_period(PLOT_END_MONTH)]

    if all_months.empty:
        raise ValueError("No months remain after applying the plot limits.")

    monthly_counts = {}

    for label, type_name, members in [
        ("Forecast", "forecast", FORECAST_MEMBERS),
        ("Hindcast", "hindcast", HINDCAST_MEMBERS),
    ]:
        group_periods = periods[model_type == type_name]
        i_date_counts = pd.Series(1, index=group_periods).groupby(level=0).sum()
        i_date_counts = i_date_counts.reindex(all_months, fill_value=0).astype(int)

        monthly_counts[label] = members * i_date_counts

    monthly_counts["All"] = monthly_counts["Forecast"] + monthly_counts["Hindcast"]

    return all_months.to_timestamp(how="start"), monthly_counts



def calculate_calendar_month_counts(plot_dates, monthly_counts):
    """Sum realization counts across years for January through December."""

    calendar_month_counts = {}

    for label, counts in monthly_counts.items():
        values = pd.Series(counts.values, index=pd.DatetimeIndex(plot_dates))
        grouped = values.groupby(values.index.month).sum()
        calendar_month_counts[label] = grouped.reindex(range(1, 13), fill_value=0)

    return calendar_month_counts


# =============================================================================
# Plotting
# =============================================================================

def plot_realization_counts(plot_dates, monthly_counts, calendar_month_counts):
    """Create the two-panel publication-quality realization-count figure."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": TICK_LABELSIZE,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )

    axis = axes[0]
    for label in ["All", "Forecast", "Hindcast"]:
        axis.plot(
            plot_dates,
            monthly_counts[label].values,
            marker="none",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=SERIES_COLORS[label],
            label=label,
        )

    axis.set_title(
        "a)",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=8,
    )
    axis.set_xlabel("Calendar year and month", fontsize=AXIS_LABELSIZE)
    axis.set_ylabel("Number of realizations", fontsize=AXIS_LABELSIZE)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    axis = axes[1]
    month_numbers = np.arange(1, 13)

    for label in ["All", "Forecast", "Hindcast"]:
        axis.plot(
            month_numbers,
            calendar_month_counts[label].values,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINEWIDTH,
            color=SERIES_COLORS[label],
            label=label,
        )

    axis.set_title(
        "b)",
        loc="left",
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        pad=8,
    )
    axis.set_xlabel("Calendar month", fontsize=AXIS_LABELSIZE)
    axis.set_ylabel("Number of realizations", fontsize=AXIS_LABELSIZE)
    axis.set_xlim(0.5, 12.5)
    axis.set_xticks(month_numbers)
    axis.set_xticklabels(MONTH_NAMES)

    for axis in axes:
        axis.tick_params(
            axis="both",
            labelsize=TICK_LABELSIZE,
            direction="out",
            length=3.5,
            width=0.8,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylim(bottom=0)
        axis.margins(x=0.01)

        if SHOW_GRID:
            axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    axes[0].legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="best",
    )

    filename = make_figure_filename()

    if WRITE_TO_FILE:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            filename,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()

    plt.close(figure)


# =============================================================================
# Reporting
# =============================================================================

def print_summary(sample_month, model_type, monthly_counts):
    """Print sample sizes and monthly count ranges."""

    forecast_rows = int(np.sum(model_type == "forecast"))
    hindcast_rows = int(np.sum(model_type == "hindcast"))

    print("Input:", make_input_filename())
    print(f"Forecast i_date rows:    {forecast_rows:,}")
    print(f"Hindcast i_date rows:    {hindcast_rows:,}")
    print(f"Forecast members/row:    {FORECAST_MEMBERS}")
    print(f"Hindcast members/row:    {HINDCAST_MEMBERS}")
    print(f"Forecast samples total:  {FORECAST_MEMBERS * forecast_rows:,}")
    print(f"Hindcast samples total:  {HINDCAST_MEMBERS * hindcast_rows:,}")
    print(f"All samples total:       {FORECAST_MEMBERS * forecast_rows + HINDCAST_MEMBERS * hindcast_rows:,}")
    print(f"First sample_month:      {int(np.min(sample_month))}")
    print(f"Last sample_month:       {int(np.max(sample_month))}")

    for label in ["Forecast", "Hindcast", "All"]:
        values = monthly_counts[label].values
        print(f"{label:>8} monthly range: {int(values.min()):,} to {int(values.max()):,}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Read sample metadata, calculate monthly counts, and make the time-series plot."""

    validate_user_settings()

    sample_month, model_type = read_sample_metadata()
    plot_dates, monthly_counts = calculate_monthly_counts(sample_month, model_type)
    calendar_month_counts = calculate_calendar_month_counts(plot_dates, monthly_counts)

    print_summary(sample_month, model_type, monthly_counts)
    plot_realization_counts(
        plot_dates,
        monthly_counts,
        calendar_month_counts,
    )


if __name__ == "__main__":
    main()
