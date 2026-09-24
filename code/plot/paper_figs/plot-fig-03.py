"""
Plot monthly S2S precipitation-extreme distributions together with monthly
reference records calculated over ``observation_years`` and the August 2023
Storm Hans value from one fixed 1957–2025 reference dataset (ERA5 or SeNorge).
The same reference dataset is used for model bias correction.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from Dunnsigouin_etal_2026 import config
# =============================================================================
# USER INPUTS — data selection
# =============================================================================
catchment = "regine_drammen"
x_days = 2  # Precipitation accumulation length in days.
forecast_date_range = ["2020-01-02", "2023-12-28"]
observation_years = ["1957", "2025"]  # Reference record and bias-correction period.
# Reference used for the plotted records and any model bias correction.
REFERENCE_DATASET = "senorge"  # "era5" or "senorge"
MODEL_DATA_METHOD = "raw"  # "raw", "mm_1step", "mm_2step", "q", "doy", "ld", "q_doy"
# Must match the settings used to build the model samples.
first_input_lead = 16
last_input_lead = 46
number_of_lead_bins = 2
model_sampling_group = "full"  # "full", "split1", "split2", ...
# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
write2file = True  # Set True to save the figure.
show_figure = True
output_directory = config.dirs["fig"]
output_filename = 'fig-03.png'
filename_out = os.path.join(output_directory, output_filename)
FIGURE_DPI = 300
# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.4
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
LEGEND_FONTSIZE = 9
YMIN = 0
YMAX = 135
BOX_WIDTH = 0.58
BOX_COLOR = "0.25"
BOX_LINEWIDTH = 1.0
REFERENCE_COLOR = "tab:red"
COUNTERFACTUAL_COLOR = "tab:green"
MODEL_LEGEND_LABEL = "Model sample distribution"

# Outlier sizes are in points; event sizes are scatter areas in points squared.
OUTLIER_MARKERSIZE = 4
OUTLIER_EDGEWIDTH = 1.0
OUTLIER_ALPHA = 0.7
STORM_HANS_MARKER = "^"
STORM_HANS_SIZE = 45
COUNTERFACTUAL_MARKER = "^"
COUNTERFACTUAL_SIZE = 45
# =============================================================================
# DATASET CONVENTIONS — these must match the existing input files
# =============================================================================
MODEL_VARIABLE = "tp24"
REFERENCE_FILE_YEARS = (1957, 2025)  # Full reference file; subset via observation_years.
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8
REFERENCE_SETTINGS = {
    "era5": {"variable": "tp24", "label": "ERA5", "directory": "era5_processed"},
    "senorge": {"variable": "rr", "label": "SeNorge", "directory": "senorge_processed"},
}
MONTHS = np.arange(1, 13)
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# =============================================================================
# Lead-time configuration
# =============================================================================


def validate_settings():
    """Validate the user-selectable model and reference settings."""
    if x_days < 1:
        raise ValueError("x_days must be at least 1.")
    first_observation_year, last_observation_year = map(int, observation_years)
    reference_first_year, reference_last_year = REFERENCE_FILE_YEARS
    if first_observation_year > last_observation_year:
        raise ValueError("observation_years must be ordered from first to last year.")
    if not (
        reference_first_year <= first_observation_year <= last_observation_year
        <= reference_last_year
    ):
        raise ValueError(
            "observation_years must lie within the fixed reference-file range "
            f"{reference_first_year}-{reference_last_year}."
        )
    if first_input_lead > last_input_lead:
        raise ValueError("first_input_lead must not exceed last_input_lead.")
    first_usable_lead = first_input_lead + x_days - 1
    if first_usable_lead > last_input_lead:
        raise ValueError("x_days is too large for the available input lead window.")
    number_of_usable_leads = last_input_lead - first_usable_lead + 1
    if not isinstance(number_of_lead_bins, int):
        raise TypeError("number_of_lead_bins must be an integer.")
    if not 1 <= number_of_lead_bins <= number_of_usable_leads:
        raise ValueError(
            "number_of_lead_bins must be between 1 and the number of usable lead times."
        )
    valid_groups = {'full'} | {f'split{number}' for number in range(1, number_of_lead_bins + 1)}
    if model_sampling_group not in valid_groups:
        raise ValueError(
            f"Unknown model_sampling_group '{model_sampling_group}'. "
            f"Valid options are {sorted(valid_groups)}."
        )
    valid_methods = {"raw", "mm_1step", "mm_2step", "q", "doy", "ld", "q_doy"}
    if MODEL_DATA_METHOD not in valid_methods:
        raise ValueError(
            f"MODEL_DATA_METHOD must be one of {sorted(valid_methods)}. "
            f"Got '{MODEL_DATA_METHOD}'."
        )
    if REFERENCE_DATASET not in REFERENCE_SETTINGS:
        raise ValueError(
            f"REFERENCE_DATASET must be one of {sorted(REFERENCE_SETTINGS)}. "
            f"Got '{REFERENCE_DATASET}'."
        )


def split_usable_accumulated_leads(first_lead, last_lead, number_of_bins):
    """Split accumulated ending leads into approximately equal contiguous bins."""
    number_of_leads = last_lead - first_lead + 1
    base_size, remainder = divmod(number_of_leads, number_of_bins)
    bin_sizes = [
        base_size + int(index >= number_of_bins - remainder)
        for index in range(number_of_bins)
    ]
    lead_bins = []
    current_start = first_lead
    for bin_size in bin_sizes:
        current_end = current_start + bin_size - 1
        lead_bins.append((current_start, current_end))
        current_start = current_end + 1
    return lead_bins


def build_lead_bins():
    """Return the lead bins used by the model sample-building script."""
    first_usable_lead = first_input_lead + x_days - 1
    return split_usable_accumulated_leads(first_usable_lead, last_input_lead, number_of_lead_bins)


def get_full_lead_range():
    """Return the complete usable accumulated lead range."""
    return first_input_lead + x_days - 1, last_input_lead


def get_selected_model_lead_range():
    """Return the lead range for the selected S2S sampling group."""
    if model_sampling_group == "full":
        return get_full_lead_range()
    split_number = int(model_sampling_group.removeprefix("split"))
    return build_lead_bins()[split_number - 1]


def get_selected_model_variable():
    """Return the model variable for the selected S2S sampling group."""
    if model_sampling_group == "full":
        return "tp24_max"
    lead_start, lead_end = get_selected_model_lead_range()
    return f"tp24_max_lead{lead_start}_{lead_end}"
# =============================================================================
# Filenames
# =============================================================================


def get_file_id(catchment_name):
    """Return the short catchment label used in model-sample filenames."""
    return catchment_name.removeprefix("regine_")


def make_model_filename():
    """Create the compact S2S sample filename written by script 2."""
    stem = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{x_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}"
    )
    if MODEL_DATA_METHOD == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}_"
            f"{observation_years[0]}-{observation_years[-1]}"
        )
    return os.path.join(config.dirs['s2s_processed'], f'{stem}_{correction_label}.nc')


def make_reference_filename():
    """Return the fixed 1957–2025 reference-file path."""
    settings = REFERENCE_SETTINGS[REFERENCE_DATASET]
    directory = config.dirs[settings["directory"]]
    variable = settings["variable"]
    first_year, last_year = REFERENCE_FILE_YEARS
    return os.path.join(
        directory,
        f"monthly_max_samples_{variable}_{x_days}dayacc_"
        f"{catchment}_{first_year}-{last_year}.nc",
    )
# =============================================================================
# Data extraction
# =============================================================================


def check_variable_exists(ds, variable, dataset_name):
    """Raise a clear error if a required variable is missing."""
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are {list(ds.data_vars)}."
        )


def validate_compact_model_structure(model_ds, variable):
    """Check the compact (number, i_date) model sample structure."""
    check_variable_exists(model_ds, variable, "model dataset")
    check_variable_exists(model_ds, "sample_month", "model dataset")
    if set(model_ds[variable].dims) != {"number", "i_date"}:
        raise ValueError(
            f"Model variable '{variable}' must have dimensions 'number' and 'i_date'; "
            f"got {model_ds[variable].dims}."
        )
    if model_ds["sample_month"].dims != ("i_date",):
        raise ValueError(
            "Model variable 'sample_month' must have dimensions ('i_date',); "
            f"got {model_ds['sample_month'].dims}."
        )


def get_model_values_by_month(model_ds, variable):
    """Return one flattened array of finite model values for each calendar month."""
    validate_compact_model_structure(model_ds, variable)
    calendar_month = model_ds["sample_month"] % 100
    values_by_month = []
    for month_number in MONTHS:
        selected = model_ds[variable].where(calendar_month == month_number, drop=True)
        values = selected.values.ravel()
        values_by_month.append(values[np.isfinite(values)])
    return values_by_month


def get_monthly_records_excluding_hans(reference_ds, variable):
    """Return monthly records over observation_years, excluding August 2023."""
    check_variable_exists(reference_ds, variable, "reference dataset")
    first_year, last_year = map(int, observation_years)
    selected = reference_ds[variable].sel(year=slice(first_year, last_year))
    if STORM_HANS_YEAR in selected["year"]:
        hans_mask = (selected['year'] == STORM_HANS_YEAR) & (selected['month'] == STORM_HANS_MONTH)
        selected = selected.where(~hans_mask)
    return selected.max(dim="year", skipna=True)


def get_storm_hans_event(reference_ds, variable):
    """Return the explicitly defined August 2023 Storm Hans value."""
    check_variable_exists(reference_ds, variable, "reference dataset")
    hans = reference_ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH)
    return STORM_HANS_MONTH, float(hans.values)


def get_highest_may_model_event(model_ds, variable):
    """Return the largest May value in the selected S2S model distribution."""
    validate_compact_model_structure(model_ds, variable)
    calendar_month = model_ds["sample_month"] % 100
    may_values = model_ds[variable].where(calendar_month == 5, drop=True).values.ravel()
    finite_values = may_values[np.isfinite(may_values)]
    if finite_values.size == 0:
        raise ValueError(f"No finite May values were found in '{variable}'.")
    return 5, float(finite_values.max())
# =============================================================================
# Plotting
# =============================================================================


def make_legend_handles():
    """Create legend handles for the plotted data."""
    reference_label = REFERENCE_SETTINGS[REFERENCE_DATASET]["label"]
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=REFERENCE_COLOR,
            markersize=6,
            label=('Calendar-month record'),
        ),
        Line2D(
            [0], [0], marker=STORM_HANS_MARKER, linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=REFERENCE_COLOR,
            markersize=np.sqrt(STORM_HANS_SIZE), label="Storm Hans",
        ),
        Patch(
            facecolor="none", edgecolor=BOX_COLOR, linewidth=BOX_LINEWIDTH,
            label=MODEL_LEGEND_LABEL,
        ),
        Line2D(
            [0], [0], marker=COUNTERFACTUAL_MARKER, linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markersize=np.sqrt(COUNTERFACTUAL_SIZE), label="Counterfactual Storm Hans",
        ),
    ]


def apply_axis_formatting(ax):
    """Apply labels, limits, ticks, and simple panel styling."""
    ax.set_ylabel(f'Monthly maximum {x_days}-day precipitation [mm]', fontsize=AXIS_LABELSIZE)
    ax.set_xlabel("Month", fontsize=AXIS_LABELSIZE)
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_monthly_extreme_distributions(
    model_values_by_month, reference_ds, model_ds, model_variable, filename_out
):
    """Create the monthly precipitation extreme-distribution figure."""
    reference_variable = REFERENCE_SETTINGS[REFERENCE_DATASET]["variable"]
    reference_records = get_monthly_records_excluding_hans(reference_ds, reference_variable)
    hans_month, hans_value = get_storm_hans_event(reference_ds, reference_variable)
    counterfactual_month, counterfactual_value = get_highest_may_model_event(
        model_ds, model_variable
    )
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.boxplot(
        model_values_by_month,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,
        flierprops={
            "marker": "o", "markerfacecolor": "none", "markeredgecolor": "0.6",
            "markersize": OUTLIER_MARKERSIZE, "linestyle": "none",
            "markeredgewidth": OUTLIER_EDGEWIDTH, "alpha": OUTLIER_ALPHA,
        },
        boxprops={"color": BOX_COLOR, "linewidth": BOX_LINEWIDTH},
        whiskerprops={"color": BOX_COLOR, "linewidth": BOX_LINEWIDTH},
        capprops={"color": BOX_COLOR, "linewidth": BOX_LINEWIDTH},
        medianprops={"color": "black", "linewidth": 1.4},
    )
    ax.scatter(
        MONTHS,
        reference_records.values,
        color=REFERENCE_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )
    ax.scatter(
        hans_month,
        hans_value,
        color=REFERENCE_COLOR,
        linewidths=1.5,
        marker=STORM_HANS_MARKER,
        s=STORM_HANS_SIZE,
        zorder=5,
    )
    ax.scatter(
        counterfactual_month,
        counterfactual_value,
        color=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        marker=COUNTERFACTUAL_MARKER,
        s=COUNTERFACTUAL_SIZE,
        zorder=6,
    )
    apply_axis_formatting(ax)
    ax.legend(
        handles=make_legend_handles(),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    if write2file:
        os.makedirs(output_directory, exist_ok=True)
        fig.savefig(filename_out, dpi=FIGURE_DPI, bbox_inches="tight")
        print("Wrote:", filename_out)
    if show_figure:
        plt.show()
    plt.close(fig)
# =============================================================================
# Main
# =============================================================================


def main():
    """Load the selected model and reference data, then create the figure."""
    validate_settings()
    model_variable = get_selected_model_variable()
    filename_model = make_model_filename()
    filename_reference = make_reference_filename()
    lead_start, lead_end = get_selected_model_lead_range()
    print("Selected settings")
    print("-----------------")
    print(f"Model group:       {model_sampling_group}")
    print(f"Model leads:       {lead_start}-{lead_end}")
    print(f"Model variable:    {model_variable}")
    print(f"Data method:       {MODEL_DATA_METHOD}")
    print(f"Reference dataset: {REFERENCE_DATASET}")
    print(f"Reference file:    {REFERENCE_FILE_YEARS[0]}-{REFERENCE_FILE_YEARS[1]}")
    print(f"Record years:      {observation_years[0]}-{observation_years[-1]}")
    print("Storm Hans:        August 2023")
    print()
    print("Figure output:", filename_out, "(saving enabled)" if write2file else "(saving disabled)")
    print("Reading model file:    ", filename_model)
    print("Reading reference file:", filename_reference)
    with (
        xr.open_dataset(filename_model, decode_timedelta=False) as model_ds,
        xr.open_dataset(filename_reference) as reference_ds,
    ):
        model_values_by_month = get_model_values_by_month(model_ds, model_variable)
        plot_monthly_extreme_distributions(
            model_values_by_month, reference_ds, model_ds, model_variable, filename_out
        )

if __name__ == "__main__":
    main()
