"""
Randomly balance compact monthly-maximum S2S samples across year-months and plot them.

Forecast and hindcast realizations are pooled before sampling. A single target
sample size is defined as the smallest number of finite realizations among all
year-month combinations from BALANCE_START_MONTH onward. Every year-month in
the selected period is then randomly subsampled without replacement to exactly
that number of realizations.

Panel (a) shows the equal number of realizations retained for each YYYYMM.
Panel (b) shows monthly model distributions using only the balanced subsample.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

variable = "tp24"
catchment = "regine_drammen"
accumulation_days = 2
forecast_date_range = ["2020-01-02", "2023-12-28"]

# Compact monthly-maximum sample file. Leave as None to construct the raw filename.
input_filename_override = None

# First YYYYMM included when defining and applying the balanced sampling.
BALANCE_START_MONTH = 200101

# Optional final YYYYMM. None uses the last available sample month.
BALANCE_END_MONTH = None

# Reproducible random subsampling.
random_seed = 43

write2file = False
show_figure = True

figure_width = 12
figure_height = 5
figure_dpi = 300
figure_wspace = 0.25

count_line_color = "tab:blue"
count_line_width = 1.7

box_width = 0.58
distribution_ymin = 0
distribution_ymax = None

title_fontsize = 12
axis_label_fontsize = 11
tick_label_fontsize = 10

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Paths and validation
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in compact sample filenames."""
    return catchment_name.removeprefix("regine_")


def make_input_filename():
    """Return the compact raw monthly-maximum sample filename."""
    if input_filename_override is not None:
        return Path(input_filename_override)

    return Path(config.dirs["s2s_processed"]) / (
        f"monthly_max_samples_{variable}_{accumulation_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}_raw.nc"
    )


def make_output_filename():
    """Return the output figure filename."""
    end_label = BALANCE_END_MONTH if BALANCE_END_MONTH is not None else "end"
    return Path(config.dirs["fig"]) / (
        f"balanced_sampling_{BALANCE_START_MONTH}-{end_label}.png"
    )


def validate_yyyymm(value, name):
    """Validate one integer YYYYMM value."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer YYYYMM value.")

    year, month = divmod(int(value), 100)
    if year < 1 or month not in range(1, 13):
        raise ValueError(f"{name} must use valid YYYYMM format.")


def validate_user_settings():
    """Validate settings and required input."""
    validate_yyyymm(BALANCE_START_MONTH, "BALANCE_START_MONTH")

    if BALANCE_END_MONTH is not None:
        validate_yyyymm(BALANCE_END_MONTH, "BALANCE_END_MONTH")
        if BALANCE_END_MONTH < BALANCE_START_MONTH:
            raise ValueError("BALANCE_END_MONTH must not precede BALANCE_START_MONTH.")

    if distribution_ymax is not None and distribution_ymax <= distribution_ymin:
        raise ValueError("distribution_ymax must be greater than distribution_ymin.")

    filename = make_input_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")


# =============================================================================
# Sample loading and balancing
# =============================================================================

def load_compact_sample():
    """Load sample months and precipitation from the compact input dataset."""
    filename = make_input_filename()

    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        required = {"sample_month", "model_type", "tp24_max"}
        missing = required - set(ds.variables)
        if missing:
            raise KeyError(f"Input file is missing variables: {sorted(missing)}")

        if ds["sample_month"].dims != ("i_date",):
            raise ValueError("sample_month must have dimension ('i_date',).")
        if ds["model_type"].dims != ("i_date",):
            raise ValueError("model_type must have dimension ('i_date',).")
        if set(ds["tp24_max"].dims) != {"number", "i_date"}:
            raise ValueError("tp24_max must have dimensions number and i_date.")

        sample_month = np.asarray(ds["sample_month"].values, dtype="int64")
        model_type = np.char.lower(np.asarray(ds["model_type"].values).astype(str))
        values = ds["tp24_max"].transpose("number", "i_date").values.astype("float64")

    unknown = sorted(set(model_type) - {"forecast", "hindcast"})
    if unknown:
        raise ValueError(f"Unsupported model_type values: {unknown}")

    return sample_month, values


def yyyymm_to_period(value):
    """Convert YYYYMM to a pandas monthly Period."""
    year, month = divmod(int(value), 100)
    return pd.Period(year=year, month=month, freq="M")


def get_selected_months(sample_month):
    """Return sorted YYYYMM values included in the balanced sampling period."""
    mask = sample_month >= BALANCE_START_MONTH
    if BALANCE_END_MONTH is not None:
        mask &= sample_month <= BALANCE_END_MONTH

    months = np.unique(sample_month[mask])
    if months.size == 0:
        raise ValueError("No sample months remain within the selected balancing period.")

    expected = pd.period_range(
        yyyymm_to_period(int(months.min())),
        yyyymm_to_period(int(months.max())),
        freq="M",
    )
    observed = pd.PeriodIndex([yyyymm_to_period(int(value)) for value in months])

    missing = expected.difference(observed)
    if len(missing):
        missing_labels = ", ".join(period.strftime("%Y%m") for period in missing)
        raise ValueError(f"Missing year-month combinations in selected period: {missing_labels}")

    return months


def count_finite_realizations_by_month(sample_month, values, months):
    """Count pooled forecast and hindcast realizations for every YYYYMM."""
    counts = {}

    for month in months:
        month_columns = sample_month == month
        counts[int(month)] = int(np.isfinite(values[:, month_columns]).sum())

    return pd.Series(counts, dtype="int64")


def randomly_balance_samples(sample_month, values, months, target_count, rng):
    """Randomly sample target_count pooled realizations from every YYYYMM."""
    balanced_values = {}

    for month in months:
        month_columns = np.flatnonzero(sample_month == month)
        month_values = values[:, month_columns]

        member_index, local_column_index = np.where(np.isfinite(month_values))
        if member_index.size < target_count:
            raise ValueError(
                f"{int(month)} has only {member_index.size} finite realizations, "
                f"below the target of {target_count}."
            )

        selected = rng.choice(member_index.size, size=target_count, replace=False)
        balanced_values[int(month)] = month_values[
            member_index[selected],
            local_column_index[selected],
        ]

    return balanced_values


def aggregate_balanced_values_by_calendar_month(balanced_values):
    """Pool balanced YYYYMM samples by calendar month across years."""
    values_by_month = []

    for month_number in range(1, 13):
        selected = [
            values
            for yyyymm, values in balanced_values.items()
            if yyyymm % 100 == month_number
        ]
        if not selected:
            raise ValueError(f"No balanced samples found for calendar month {month_number}.")

        values_by_month.append(np.concatenate(selected))

    return values_by_month


# =============================================================================
# Reporting and plotting
# =============================================================================

def print_sampling_summary(original_counts, target_count):
    """Print the balancing threshold and original monthly count range."""
    print("\nBalanced sampling")
    print("-----------------")
    print("Start month:", BALANCE_START_MONTH)
    print("End month:", int(original_counts.index.max()))
    print("Minimum original count:", int(original_counts.min()))
    print("Maximum original count:", int(original_counts.max()))
    print("Target realizations per YYYYMM:", target_count)

    limiting_months = original_counts[original_counts == target_count].index.tolist()
    print("Limiting YYYYMM:", ", ".join(str(month) for month in limiting_months))


def format_axis(axis):
    """Apply shared simple panel styling."""
    axis.tick_params(
        axis="both",
        labelsize=tick_label_fontsize,
        direction="out",
        length=3.5,
        width=0.8,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_balanced_count_panel(axis, months, target_count):
    """Plot equal realization counts for every selected YYYYMM."""
    plot_dates = pd.DatetimeIndex(
        [yyyymm_to_period(int(month)).to_timestamp(how="start") for month in months]
    )
    counts = np.full(len(plot_dates), target_count, dtype="int64")

    axis.plot(
        plot_dates,
        counts,
        linewidth=count_line_width,
        color=count_line_color,
    )
    axis.set_title(
        "a) Equally sampled ensemble members by year",
        loc="left",
        fontsize=title_fontsize,
        pad=8,
    )
    axis.set_xlabel("Time [year]", fontsize=axis_label_fontsize)
    axis.set_ylabel("Number", fontsize=axis_label_fontsize)

    tick_years = np.arange(plot_dates.min().year, plot_dates.max().year + 1)
    label_years = set(tick_years[::2])
    year_ticks = [pd.Timestamp(year=year, month=1, day=1) for year in tick_years]
    year_labels = [str(year) if year in label_years else "" for year in tick_years]

    axis.set_xticks(year_ticks)
    axis.set_xticklabels(
        year_labels,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylim(bottom=0)
    axis.margins(x=0.01)
    format_axis(axis)


def plot_monthly_distribution_panel(axis, values_by_month):
    """Plot monthly distributions using only the balanced model realizations."""
    axis.boxplot(
        values_by_month,
        positions=np.arange(1, 13),
        widths=box_width,
        patch_artist=False,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "0.6",
            "markersize": 4,
            "linestyle": "none",
            "markeredgewidth": 0.8,
        },
        boxprops={"color": "0.25", "linewidth": 1.0},
        whiskerprops={"color": "0.25", "linewidth": 1.0},
        capprops={"color": "0.25", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.4},
    )

    axis.set_title(
        "b) Monthly distribution of equally sampled model extremes",
        loc="left",
        fontsize=title_fontsize,
        pad=8,
    )
    axis.set_xlabel("Month", fontsize=axis_label_fontsize)
    axis.set_ylabel(
        f"Monthly maximum {accumulation_days}-day precipitation [mm]",
        fontsize=axis_label_fontsize,
    )
    axis.set_xlim(0.4, 12.6)
    axis.set_xticks(np.arange(1, 13))
    axis.set_xticklabels(month_labels)
    axis.set_ylim(bottom=distribution_ymin)
    if distribution_ymax is not None:
        axis.set_ylim(distribution_ymin, distribution_ymax)

    format_axis(axis)


def make_figure(months, target_count, balanced_values, filename=None):
    """Create count and monthly-distribution panels from the balanced sample."""
    values_by_month = aggregate_balanced_values_by_calendar_month(balanced_values)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        gridspec_kw={"wspace": figure_wspace},
    )

    plot_balanced_count_panel(axes[0], months, target_count)
    plot_monthly_distribution_panel(axes[1], values_by_month)
    figure.tight_layout()

    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=figure_dpi, bbox_inches="tight")
        print("Wrote:", filename)

    if show_figure:
        plt.show()

    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

def main():
    """Balance pooled realizations across YYYYMM and plot the balanced sample."""
    validate_user_settings()

    filename_input = make_input_filename()
    filename_output = make_output_filename()

    print("Reading:", filename_input)
    sample_month, values = load_compact_sample()

    months = get_selected_months(sample_month)
    original_counts = count_finite_realizations_by_month(sample_month, values, months)

    if np.any(original_counts <= 0):
        empty_months = original_counts[original_counts <= 0].index.tolist()
        raise ValueError(f"Empty year-month samples found: {empty_months}")

    target_count = int(original_counts.min())
    rng = np.random.default_rng(random_seed)
    balanced_values = randomly_balance_samples(
        sample_month,
        values,
        months,
        target_count,
        rng,
    )

    balanced_counts = {month: values.size for month, values in balanced_values.items()}
    if set(balanced_counts.values()) != {target_count}:
        raise RuntimeError("Balanced sampling did not produce equal YYYYMM sample sizes.")

    print_sampling_summary(original_counts, target_count)
    make_figure(
        months,
        target_count,
        balanced_values,
        filename_output if write2file else None,
    )


if __name__ == "__main__":
    main()
