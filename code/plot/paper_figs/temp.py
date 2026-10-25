"""
Plot monthly S2S precipitation-extreme distributions together with one
reference record per calendar month (ERA5 or seNorge) and the August 2023
Storm Hans value. The same reference dataset is used for model bias correction.

Records
-------
Each calendar month shows exactly one record, calculated over
``observation_years`` from one fixed 1957-2025 reference file:

    - January-July and September-December: the monthly maximum over
      observation_years, drawn as orange circles;
    - August: Storm Hans (August 2023), drawn as a larger orange star with a
      dark edge, because it is the August record.

The script checks that Storm Hans is indeed the August record. If August 2023
lies outside observation_years, or an earlier August exceeds Storm Hans, the
pre-Hans August record is drawn as a circle as well and a warning is printed.

Counterfactual
--------------
The green triangle is the COUNTERFACTUAL_RANK-th largest May value in the
selected model sample (1 = largest), pooled over all initialisations and
ensemble members. Equal values count as separate samples. Its origin
(model type, initialisation, ensemble member, lead and date of the maximum) is
printed so the forecast can be followed up as a storyline.

Output
------
FIGURE_FORMAT = "png" or "pdf". PDFs embed fonts as TrueType so text stays
editable; the figure is small and remains vector graphics.
"""
import os
import warnings
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

# Counterfactual Storm Hans: the Nth largest May model sample (1 = largest).
COUNTERFACTUAL_MONTH = 5
COUNTERFACTUAL_RANK = 2

# =============================================================================
# FIGURE OUTPUT — edit the directory or filename here
# =============================================================================
write2file = True  # Set True to save the figure.
show_figure = True
FIGURE_FORMAT = "pdf"  # "png" or "pdf".
output_directory = config.dirs["fig"]
output_filename = f"fig-04.{FIGURE_FORMAT}"
filename_out = os.path.join(output_directory, output_filename)
FIGURE_DPI = 300  # Resolution of PNG output.

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

# Model box plots.
MODEL_COLOR = "tab:blue"
BOX_WIDTH = 0.58
BOX_LINEWIDTH = 1.0
MEDIAN_LINEWIDTH = 1.8
MODEL_LEGEND_LABEL = "Model sample distribution"
# Outlier sizes are in points.
OUTLIER_MARKERSIZE = 4
OUTLIER_EDGEWIDTH = 1.0
OUTLIER_ALPHA = 0.5

# Reference records (seNorge orange, as in the other figures).
REFERENCE_COLOR = "tab:orange"
RECORD_MARKER = "o"
RECORD_SIZE = 35  # Scatter area in points squared.
RECORD_LEGEND_LABEL = "seNorge calendar-month record"

# Storm Hans, the August record.
STORM_HANS_MARKER = "^"
STORM_HANS_SIZE = 75  # Scatter area in points squared.
STORM_HANS_EDGE_COLOR = "None"
STORM_HANS_EDGE_WIDTH = 0.0
STORM_HANS_LEGEND_LABEL = "seNorge storm Hans (august record)"

# Counterfactual.
COUNTERFACTUAL_COLOR = "tab:green"
COUNTERFACTUAL_MARKER = "^"
COUNTERFACTUAL_SIZE = 75  # Scatter area in points squared.

# =============================================================================
# DATASET CONVENTIONS — these must match the existing input files
# =============================================================================
MODEL_VARIABLE = "tp24"
REFERENCE_FILE_YEARS = (1957, 2025)  # Full reference file; subset via observation_years.
STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8
REFERENCE_SETTINGS = {
    "era5": {"variable": "tp24", "label": "ERA5", "directory": "era5_processed"},
    "senorge": {"variable": "rr", "label": "seNorge", "directory": "senorge_processed"},
}
MONTHS = np.arange(1, 13)
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# =============================================================================
# Validation
# =============================================================================


def validate_settings():
    """Validate the user-selectable model, reference and figure settings."""
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
    if COUNTERFACTUAL_MONTH not in MONTHS:
        raise ValueError("COUNTERFACTUAL_MONTH must be an integer from 1 to 12.")
    if not isinstance(COUNTERFACTUAL_RANK, (int, np.integer)) or COUNTERFACTUAL_RANK < 1:
        raise ValueError("COUNTERFACTUAL_RANK must be a positive integer (1 = largest).")
    if FIGURE_FORMAT not in {"png", "pdf"}:
        raise ValueError("FIGURE_FORMAT must be 'png' or 'pdf'.")

# =============================================================================
# Lead-time configuration
# =============================================================================


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
    return selected.max(dim="year", skipna=True).sel(month=MONTHS)


def get_storm_hans_event(reference_ds, variable):
    """Return the explicitly defined August 2023 Storm Hans value."""
    check_variable_exists(reference_ds, variable, "reference dataset")
    hans = reference_ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH)
    return STORM_HANS_MONTH, float(hans.values)


def hans_is_august_record(records_without_hans, hans_value):
    """
    Return True if Storm Hans is the August record over observation_years.

    Storm Hans must lie inside observation_years and be at least as large as
    every other August in that period.
    """
    first_year, last_year = map(int, observation_years)
    if not first_year <= STORM_HANS_YEAR <= last_year:
        warnings.warn(
            f"August {STORM_HANS_YEAR} lies outside observation_years "
            f"{first_year}-{last_year}; the August record is drawn as well as Storm Hans."
        )
        return False
    previous_record = float(records_without_hans.sel(month=STORM_HANS_MONTH).values)
    if np.isfinite(previous_record) and previous_record > hans_value:
        warnings.warn(
            f"Storm Hans ({hans_value:.1f} mm) is not the August record: "
            f"{previous_record:.1f} mm is larger. The August record is drawn as well."
        )
        return False
    return True


def ordinal(number):
    """Return 1st, 2nd, 3rd, 4th, ... for a positive integer."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def counterfactual_description():
    """Return a readable description of the counterfactual sample."""
    month = MONTH_NAMES[COUNTERFACTUAL_MONTH - 1]
    if COUNTERFACTUAL_RANK == 1:
        return f"largest {month} model sample"
    return f"{ordinal(COUNTERFACTUAL_RANK)} largest {month} model sample"


def attribute_text(value):
    """Return a value as plain text (decoding bytes from NetCDF strings)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def describe_sample_origin(model_ds, number_index, i_date_index):
    """Return the provenance of one model sample as printable lines."""
    lines = [
        f"Initialisation (i_date): {np.datetime_as_string(model_ds['i_date'].values[i_date_index], unit='D')}",
        f"Ensemble member:         {model_ds['number'].values[number_index]}",
    ]
    if "model_type" in model_ds:
        model_type = attribute_text(model_ds["model_type"].values[i_date_index])
        lines.insert(0, f"Model type:              {model_type}")
        if model_type.lower() == "hindcast" and "hdate" in model_ds:
            lines.append(f"Hindcast date (hdate):   {int(model_ds['hdate'].values[i_date_index])}")
    if "sample_month" in model_ds:
        lines.append(f"Sample month (YYYYMM):   {int(model_ds['sample_month'].values[i_date_index])}")
    for name, label in [("lead_of_max", "Lead day of maximum:"), ("date_of_max", "Date of maximum:")]:
        if name in model_ds:
            value = model_ds[name].isel(number=number_index, i_date=i_date_index).values
            if name == "lead_of_max":
                text = f"{float(value):.0f}" if np.isfinite(value) else "n/a"
            else:
                text = np.datetime_as_string(value, unit="D") if not np.isnat(value) else "n/a"
            lines.append(f"{label:<25}{text}")
    return lines


def get_ranked_counterfactual_event(model_ds, variable):
    """
    Return the COUNTERFACTUAL_RANK-th largest model value in COUNTERFACTUAL_MONTH.

    Values are pooled over all initialisations and ensemble members of the
    selected sampling group; equal values count as separate samples.
    """
    validate_compact_model_structure(model_ds, variable)
    data = model_ds[variable].transpose("number", "i_date")
    calendar_month = (model_ds["sample_month"] % 100).values
    values = np.asarray(data.values, dtype=float)
    in_month = np.broadcast_to(calendar_month == COUNTERFACTUAL_MONTH, values.shape)
    valid = in_month & np.isfinite(values)
    number_indices, i_date_indices = np.nonzero(valid)
    month_values = values[valid]
    month_name = MONTH_NAMES[COUNTERFACTUAL_MONTH - 1]
    if month_values.size == 0:
        raise ValueError(f"No finite {month_name} values were found in '{variable}'.")
    if COUNTERFACTUAL_RANK > month_values.size:
        raise ValueError(
            f"COUNTERFACTUAL_RANK={COUNTERFACTUAL_RANK} exceeds the number of "
            f"{month_name} model samples ({month_values.size})."
        )
    order = np.argsort(-month_values, kind="stable")
    selected = order[COUNTERFACTUAL_RANK - 1]
    return {
        "month": COUNTERFACTUAL_MONTH,
        "value": float(month_values[selected]),
        "number_of_samples": int(month_values.size),
        "origin": describe_sample_origin(
            model_ds, number_indices[selected], i_date_indices[selected]
        ),
    }

# =============================================================================
# Plotting
# =============================================================================

def make_legend_handles(show_august_record_circle):
    """Create legend handles: seNorge records, Storm Hans, model, counterfactual."""
    return [
        Line2D(
            [0], [0], marker=RECORD_MARKER, linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=REFERENCE_COLOR,
            markersize=np.sqrt(RECORD_SIZE), label=RECORD_LEGEND_LABEL,
        ),
        Line2D(
            [0], [0], marker=STORM_HANS_MARKER, linestyle="none",
            markerfacecolor=REFERENCE_COLOR, markeredgecolor=STORM_HANS_EDGE_COLOR,
            markeredgewidth=STORM_HANS_EDGE_WIDTH, markersize=np.sqrt(STORM_HANS_SIZE),
            label=STORM_HANS_LEGEND_LABEL if not show_august_record_circle else "Storm Hans",
        ),
        Patch(
            facecolor="none", edgecolor=MODEL_COLOR, linewidth=BOX_LINEWIDTH,
            label=MODEL_LEGEND_LABEL,
        ),
        Line2D(
            [0], [0], marker=COUNTERFACTUAL_MARKER, linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=STORM_HANS_EDGE_COLOR,  # Use COUNTERFACTUAL_COLOR if the plotted marker has no black edge.
            markeredgewidth=STORM_HANS_EDGE_WIDTH,
            markersize=np.sqrt(COUNTERFACTUAL_SIZE),
            label=f"Model counterfactual storm Hans",
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
    model_values_by_month, records_without_hans, hans_value, counterfactual, filename_out
):
    """Create the monthly precipitation extreme-distribution figure."""
    show_august_record_circle = not hans_is_august_record(records_without_hans, hans_value)
    record_months = MONTHS if show_august_record_circle else MONTHS[MONTHS != STORM_HANS_MONTH]
    record_values = records_without_hans.sel(month=record_months).values

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax.boxplot(
        model_values_by_month,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,
        flierprops={
            "marker": "o", "markerfacecolor": "none", "markeredgecolor": MODEL_COLOR,
            "markersize": OUTLIER_MARKERSIZE, "linestyle": "none",
            "markeredgewidth": OUTLIER_EDGEWIDTH, "alpha": OUTLIER_ALPHA,
        },
        boxprops={"color": MODEL_COLOR, "linewidth": BOX_LINEWIDTH},
        whiskerprops={"color": MODEL_COLOR, "linewidth": BOX_LINEWIDTH},
        capprops={"color": MODEL_COLOR, "linewidth": BOX_LINEWIDTH},
        medianprops={"color": MODEL_COLOR, "linewidth": MEDIAN_LINEWIDTH},
    )
    ax.scatter(
        record_months,
        record_values,
        color=REFERENCE_COLOR,
        marker=RECORD_MARKER,
        s=RECORD_SIZE,
        zorder=4,
    )
    ax.scatter(
        STORM_HANS_MONTH,
        hans_value,
        facecolor=REFERENCE_COLOR,
        edgecolor=STORM_HANS_EDGE_COLOR,
        linewidths=STORM_HANS_EDGE_WIDTH,
        marker=STORM_HANS_MARKER,
        s=STORM_HANS_SIZE,
        zorder=5,
    )
    ax.scatter(
        counterfactual["month"],
        counterfactual["value"],
        color=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        marker=COUNTERFACTUAL_MARKER,
        s=COUNTERFACTUAL_SIZE,
        zorder=6,
    )
    apply_axis_formatting(ax)
    ax.legend(
        handles=make_legend_handles(show_august_record_circle),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout()
    if write2file:
        os.makedirs(output_directory, exist_ok=True)
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(filename_out, format=FIGURE_FORMAT, dpi=FIGURE_DPI,
                        bbox_inches="tight", facecolor="white")
        print("Wrote:", filename_out)
    if show_figure:
        plt.show()
    plt.close(fig)

# =============================================================================
# Reporting
# =============================================================================


def print_events(records_without_hans, hans_value, counterfactual):
    """Print the plotted records, Storm Hans and the counterfactual sample."""
    reference_label = REFERENCE_SETTINGS[REFERENCE_DATASET]["label"]
    print()
    print(f"{reference_label} calendar-month records {observation_years[0]}-{observation_years[-1]}")
    print("-" * 50)
    for month in MONTHS:
        value = float(records_without_hans.sel(month=month).values)
        note = "  (pre-Hans August record)" if month == STORM_HANS_MONTH else ""
        print(f"  {MONTH_LABELS[month - 1]}  {value:7.2f} mm{note}")
    print(f"  Storm Hans (August {STORM_HANS_YEAR}): {hans_value:.2f} mm")
    print()
    print(f"Counterfactual: {counterfactual_description()}")
    print("-" * 50)
    print(f"  Value: {counterfactual['value']:.2f} mm "
          f"(rank {COUNTERFACTUAL_RANK} of {counterfactual['number_of_samples']} samples; "
          f"{counterfactual['value'] / hans_value:.0%} of Storm Hans)")
    for line in counterfactual["origin"]:
        print(f"  {line}")

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
    print(f"Counterfactual:    {counterfactual_description()}")
    print()
    print("Figure output:", filename_out, "(saving enabled)" if write2file else "(saving disabled)")
    print("Reading model file:    ", filename_model)
    print("Reading reference file:", filename_reference)
    reference_variable = REFERENCE_SETTINGS[REFERENCE_DATASET]["variable"]
    with (
        xr.open_dataset(filename_model, decode_timedelta=False) as model_ds,
        xr.open_dataset(filename_reference) as reference_ds,
    ):
        model_values_by_month = get_model_values_by_month(model_ds, model_variable)
        records_without_hans = get_monthly_records_excluding_hans(
            reference_ds, reference_variable
        ).load()
        _, hans_value = get_storm_hans_event(reference_ds, reference_variable)
        counterfactual = get_ranked_counterfactual_event(model_ds, model_variable)
    print_events(records_without_hans, hans_value, counterfactual)
    plot_monthly_extreme_distributions(
        model_values_by_month, records_without_hans, hans_value, counterfactual, filename_out
    )


if __name__ == "__main__":
    main()
