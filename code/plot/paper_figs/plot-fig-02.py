"""
Plot monthly distributions of catchment precipitation extremes.

The figure compares:
1. S2S forecast/hindcast monthly extreme distributions.
2. ERA5 monthly records before Storm Hans.
3. SeNorge or regridded SeNorge monthly records before Storm Hans.
4. Storm Hans 2023 in ERA5 and SeNorge / SeNorge-regrid.
5. The largest May S2S event, interpreted as a counterfactual spring Hans.

The S2S model input is expected to be a lead-split file produced by the
lead-bin sample-building script. A single user setting selects whether to plot
the complete lead range or one of the individual lead-time splits.

Example for x_days=2 and number_of_lead_bins=2:
    model_sampling_group = "full"   -> max_value_lead17_46
    model_sampling_group = "split1" -> max_value_lead17_30
    model_sampling_group = "split2" -> max_value_lead31_46
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"
x_days = 2

reference_dataset = "senorge"  # "senorge" or "senorge_regrid"

forecast_date_range = ["2020-01-02", "2023-06-26"]
reference_years = ["1957", "2023"]

era5_grid = "0.5x0.5"

# Daily lead range used when creating the model sample.
first_input_lead = 16
last_input_lead = 46

# Must match the number of lead bins used by the sample-building script.
number_of_lead_bins = 2

# Model sample to plot:
#     "full"   -> complete usable lead range
#     "split1" -> first lead-time subset
#     "split2" -> second lead-time subset
#     ...
model_sampling_group = "full"

if model_sampling_group == "full":
    filename_out = config.dirs["fig"] + f"fig-02.png"
else:
    filename_out = config.dirs["fig"] + f"fig-02-{model_sampling_group}.png"

write2file = False

# =============================================================================
# Dataset-specific settings
# =============================================================================

# Model files produced by the lead-bin sample-building script.
MODEL_VARIABLE = "tp24"

ERA5_VARIABLE = "tp24"

REFERENCE_VARIABLES = {
    "senorge": "rr",
    "senorge_regrid": "rr",
}

REFERENCE_LABELS = {
    "senorge": "SeNorge",
    "senorge_regrid": "SeNorge regrid",
}


# =============================================================================
# Plot settings
# =============================================================================

FIG_WIDTH_IN = 7.2
FIG_HEIGHT_IN = 4.4

TITLE_FONTSIZE = 11
AXIS_LABELSIZE = 11
TICK_LABELSIZE = 11
LEGEND_FONTSIZE = 7

YMIN = 0
YMAX = 135

BOX_WIDTH = 0.58

REFERENCE_COLOR = "tab:red"
ERA5_COLOR = "tab:blue"
COUNTERFACTUAL_COLOR = "tab:green"

MONTHS = np.arange(1, 13)
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Lead-time helpers
# =============================================================================

def validate_model_sampling_settings() -> None:
    """Check the model lead-time sampling settings."""

    if x_days < 1:
        raise ValueError("x_days must be at least 1.")

    if first_input_lead > last_input_lead:
        raise ValueError(
            "first_input_lead must not exceed last_input_lead."
        )

    if not isinstance(number_of_lead_bins, int):
        raise TypeError(
            "number_of_lead_bins must be an integer."
        )

    if number_of_lead_bins < 1:
        raise ValueError(
            "number_of_lead_bins must be at least 1."
        )

    number_of_input_leads = (
        last_input_lead - first_input_lead + 1
    )

    if number_of_lead_bins > number_of_input_leads:
        raise ValueError(
            "number_of_lead_bins cannot exceed the number of "
            "available input lead days."
        )

    valid_groups = {
        "full",
        *{
            f"split{index}"
            for index in range(1, number_of_lead_bins + 1)
        },
    }

    if model_sampling_group not in valid_groups:
        raise ValueError(
            f"Unknown model_sampling_group '{model_sampling_group}'. "
            f"Valid options are: {sorted(valid_groups)}."
        )


def split_input_leads(
    first_lead: int,
    last_lead: int,
    number_of_bins: int,
) -> list[tuple[int, int]]:
    """
    Split an inclusive input lead interval into approximately equal bins.

    Extra days are assigned to the later bins, matching the sample-building
    script. Thus 16-46 split into two bins becomes 16-30 and 31-46.
    """

    number_of_leads = last_lead - first_lead + 1

    base_size = number_of_leads // number_of_bins
    remainder = number_of_leads % number_of_bins

    bin_sizes = [
        base_size
        + int(
            bin_index >= number_of_bins - remainder
        )
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
    Return the full usable accumulated range followed by all split ranges.

    Example for 2-day accumulation over input leads 16-46 with two bins:
        [(17, 46), (17, 30), (31, 46)]
    """

    first_usable_lead = first_input_lead + x_days - 1

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

        if accumulated_start > bin_end:
            raise ValueError(
                "A split contains no usable accumulated leads. "
                "Reduce x_days or number_of_lead_bins."
            )

        split_ranges.append(
            (accumulated_start, bin_end)
        )

    return [
        (first_usable_lead, last_input_lead),
        *split_ranges,
    ]


def get_selected_model_lead_range() -> tuple[int, int]:
    """Return the lead range selected by model_sampling_group."""

    lead_ranges = build_accumulated_lead_ranges()

    if model_sampling_group == "full":
        return lead_ranges[0]

    split_number = int(
        model_sampling_group.replace("split", "")
    )

    return lead_ranges[split_number]


def get_selected_model_variable() -> str:
    """Return the max-value variable for the selected lead range."""

    lead_start, lead_end = get_selected_model_lead_range()

    return f"max_value_lead{lead_start}_{lead_end}"


def lead_split_filename_label() -> str:
    """Return the lead-split label used by the model sample filename."""

    lead_ranges = build_accumulated_lead_ranges()

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


def get_model_sampling_label() -> str:
    """Return a readable label for the selected model sampling group."""

    lead_start, lead_end = get_selected_model_lead_range()

    if model_sampling_group == "full":
        return f"model leads {lead_start}-{lead_end}"

    split_number = int(
        model_sampling_group.replace("split", "")
    )

    return (
        f"model split {split_number}: "
        f"leads {lead_start}-{lead_end}"
    )


# =============================================================================
# Configuration helpers
# =============================================================================

def get_reference_variable(reference_dataset: str) -> str:
    """Return the variable name used in the selected reference dataset."""

    if reference_dataset not in REFERENCE_VARIABLES:
        valid = ", ".join(REFERENCE_VARIABLES)
        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid}."
        )

    return REFERENCE_VARIABLES[reference_dataset]


def get_reference_label(reference_dataset: str) -> str:
    """Return a plot-friendly label for the selected reference dataset."""

    if reference_dataset not in REFERENCE_LABELS:
        valid = ", ".join(REFERENCE_LABELS)
        raise ValueError(
            f"Unknown reference_dataset '{reference_dataset}'. "
            f"Valid options are: {valid}."
        )

    return REFERENCE_LABELS[reference_dataset]


def get_catchment_label(catchment: str) -> str:
    """Return a plot-friendly catchment name."""

    labels = {
        "regine_drammen": "Drammen catchment",
        "regine_glomma": "Glomma catchment",
    }

    return labels.get(catchment, catchment)


# =============================================================================
# Filename helpers
# =============================================================================

def make_model_filename() -> str:
    """
    Create the S2S lead-split input filename.

    This follows the filename convention used by the lead-bin sample builder.
    """

    lead_label = lead_split_filename_label()

    return os.path.join(
        config.dirs["s2s_processed"],
        (
            f"all_distribution_monthly_extremes_"
            f"{MODEL_VARIABLE}_{x_days}dayacc_"
            f"{catchment}_"
            f"{lead_label}_"
            f"forecast_hindcast_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}.nc"
        ),
    )


def make_era5_filename() -> str:
    """Create the ERA5 input filename."""

    return (
        f"{config.dirs['era5_processed']}"
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{x_days}dayacc_"
        f"{catchment}_era5_{era5_grid}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def make_reference_filename(
    reference_dataset: str,
    reference_variable: str,
) -> str:
    """Create the SeNorge or SeNorge-regrid input filename."""

    return (
        f"{config.dirs[f'{reference_dataset}_processed']}"
        f"distribution_monthly_extremes_{reference_variable}_{x_days}dayacc_"
        f"{catchment}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


# =============================================================================
# Data loading
# =============================================================================

def load_datasets(
    filename_model: str,
    filename_era5: str,
    filename_reference: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open model, ERA5, and reference datasets."""

    model_ds = xr.open_dataset(filename_model)
    era5_ds = xr.open_dataset(filename_era5)
    reference_ds = xr.open_dataset(filename_reference)

    return model_ds, era5_ds, reference_ds


def check_variable_exists(
    ds: xr.Dataset,
    variable: str,
    dataset_name: str,
) -> None:
    """Raise a clear error if a required variable is missing."""

    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' was not found in {dataset_name}. "
            f"Available variables are: {list(ds.data_vars)}"
        )


# =============================================================================
# Data extraction
# =============================================================================

def get_model_values_by_month(
    model_ds: xr.Dataset,
    variable: str,
) -> list[np.ndarray]:
    """
    Convert the selected S2S lead-range sample into one array per month.

    Expected variable structure:
        model_ds[variable](month_of_year, index)
    """

    check_variable_exists(
        model_ds,
        variable,
        "model dataset",
    )

    values_by_month = []

    for month in MONTHS:
        values = model_ds[variable].sel(
            month_of_year=month
        ).values

        values = values[np.isfinite(values)]
        values_by_month.append(values)

    return values_by_month


def get_monthly_records_before_hans(
    ds: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    """
    Get monthly records before Storm Hans.

    Uses 1957-2022, so Storm Hans in 2023 is excluded.
    """

    check_variable_exists(
        ds,
        variable,
        "reference dataset",
    )

    before_hans = ds[variable].sel(
        year=slice(1957, 2022)
    )

    return before_hans.max(dim="year")


def get_storm_hans_event(
    ds: xr.Dataset,
    variable: str,
) -> tuple[int, float]:
    """
    Get the largest 2023 event.

    This assumes the largest 2023 value corresponds to Storm Hans.
    """

    check_variable_exists(
        ds,
        variable,
        "reference dataset",
    )

    values_2023 = ds[variable].sel(
        year=2023
    )

    flat = values_2023.stack(
        z=("month",)
    )

    max_index = flat.argmax("z")
    max_value = flat.isel(z=max_index)
    max_month = flat["month"].isel(z=max_index)

    return (
        int(max_month.values),
        float(max_value.values),
    )


def get_highest_may_model_event(
    model_ds: xr.Dataset,
    variable: str,
) -> tuple[int, float]:
    """
    Get the largest May event from the selected S2S sampling group.
    """

    check_variable_exists(
        model_ds,
        variable,
        "model dataset",
    )

    may_values = model_ds[variable].sel(
        month_of_year=5
    )

    may_max = may_values.max()

    return 5, float(may_max.values)


# =============================================================================
# Plotting helpers
# =============================================================================

def make_legend_handles(
    reference_label: str,
) -> list[Line2D]:
    """Create legend handles for the plot."""

    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=f"{reference_label} record 1957–2022",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 record 1957–2022",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label=f"{reference_label} Storm Hans 2023",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor=ERA5_COLOR,
            markeredgecolor=ERA5_COLOR,
            markeredgewidth=1.5,
            markersize=6,
            label="ERA5 Storm Hans 2023",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="0.6",
            markeredgewidth=0.8,
            markersize=5,
            label=get_model_sampling_label(),
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COUNTERFACTUAL_COLOR,
            markeredgecolor=COUNTERFACTUAL_COLOR,
            markeredgewidth=1.0,
            markersize=5,
            label="Counterfactual spring Storm Hans",
        ),
    ]


def apply_axis_formatting(ax) -> None:
    """Apply consistent axis formatting."""

    catchment_label = get_catchment_label(catchment)
    lead_start, lead_end = get_selected_model_lead_range()

    ax.set_title(
        (
            f"{catchment_label}, monthly {x_days}-day accumulated "
            f"precipitation maxima\n"
            f"S2S ending leads {lead_start}-{lead_end}"
        ),
        fontsize=TITLE_FONTSIZE,
        pad=8,
    )

    ax.set_ylabel(
        "Precipitation [mm]",
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_xlabel(
        "Month",
        fontsize=AXIS_LABELSIZE,
    )

    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xticks(MONTHS)
    ax.set_xticklabels(MONTH_LABELS)

    ax.tick_params(
        axis="both",
        labelsize=TICK_LABELSIZE,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Main plotting function
# =============================================================================

def plot_monthly_extreme_distributions(
    model_values_by_month: list[np.ndarray],
    era5_ds: xr.Dataset,
    reference_ds: xr.Dataset,
    model_ds: xr.Dataset,
    model_variable: str,
    reference_variable: str,
    reference_label: str,
    filename_out: str,
    write2file: bool,
) -> None:
    """Create the monthly precipitation extreme distribution figure."""

    era5_records = get_monthly_records_before_hans(
        era5_ds,
        variable=ERA5_VARIABLE,
    )

    reference_records = get_monthly_records_before_hans(
        reference_ds,
        variable=reference_variable,
    )

    era5_hans_month, era5_hans_value = get_storm_hans_event(
        era5_ds,
        variable=ERA5_VARIABLE,
    )

    (
        reference_hans_month,
        reference_hans_value,
    ) = get_storm_hans_event(
        reference_ds,
        variable=reference_variable,
    )

    (
        counterfactual_month,
        counterfactual_value,
    ) = get_highest_may_model_event(
        model_ds,
        variable=model_variable,
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
    )

    ax.boxplot(
        model_values_by_month,
        positions=MONTHS,
        widths=BOX_WIDTH,
        patch_artist=False,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.6",
            markersize=4,
            linestyle="none",
            markeredgewidth=0.8,
        ),
        boxprops=dict(
            color="0.25",
            linewidth=1.0,
        ),
        whiskerprops=dict(
            color="0.25",
            linewidth=1.0,
        ),
        capprops=dict(
            color="0.25",
            linewidth=1.0,
        ),
        medianprops=dict(
            color="black",
            linewidth=1.4,
        ),
    )

    ax.scatter(
        MONTHS,
        reference_records.values,
        facecolors=REFERENCE_COLOR,
        edgecolors=REFERENCE_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )

    ax.scatter(
        MONTHS,
        era5_records.values,
        facecolors=ERA5_COLOR,
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        s=35,
        zorder=4,
    )

    ax.scatter(
        reference_hans_month,
        reference_hans_value,
        facecolors=REFERENCE_COLOR,
        edgecolors=REFERENCE_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
    )

    ax.scatter(
        era5_hans_month,
        era5_hans_value,
        facecolors=ERA5_COLOR,
        edgecolors=ERA5_COLOR,
        linewidths=1.5,
        marker="^",
        s=35,
        zorder=5,
    )

    ax.scatter(
        counterfactual_month,
        counterfactual_value,
        facecolors=COUNTERFACTUAL_COLOR,
        edgecolors=COUNTERFACTUAL_COLOR,
        linewidths=1.0,
        marker="o",
        s=20,
        zorder=6,
    )

    apply_axis_formatting(ax)

    ax.legend(
        handles=make_legend_handles(
            reference_label
        ),
        loc="upper left",
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
    )

    fig.tight_layout()

    if write2file:
        fig.savefig(
            filename_out,
            dpi=300,
            bbox_inches="tight",
        )

        print("Wrote:", filename_out)

    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_model_sampling_settings()

    reference_variable = get_reference_variable(
        reference_dataset
    )

    reference_label = get_reference_label(
        reference_dataset
    )

    model_extreme_variable = get_selected_model_variable()

    filename_model = make_model_filename()
    filename_era5 = make_era5_filename()

    filename_reference = make_reference_filename(
        reference_dataset=reference_dataset,
        reference_variable=reference_variable,
    )

    lead_start, lead_end = get_selected_model_lead_range()

    print("Selected model sampling group")
    print("-----------------------------")
    print(f"Group:    {model_sampling_group}")
    print(f"Leads:    {lead_start}-{lead_end}")
    print(f"Variable: {model_extreme_variable}")

    print()
    print("Reading model file:    ", filename_model)
    print("Reading ERA5 file:     ", filename_era5)
    print("Reading reference file:", filename_reference)

    (
        model_ds,
        era5_ds,
        reference_ds,
    ) = load_datasets(
        filename_model=filename_model,
        filename_era5=filename_era5,
        filename_reference=filename_reference,
    )

    try:
        model_values_by_month = get_model_values_by_month(
            model_ds,
            variable=model_extreme_variable,
        )

        plot_monthly_extreme_distributions(
            model_values_by_month=model_values_by_month,
            era5_ds=era5_ds,
            reference_ds=reference_ds,
            model_ds=model_ds,
            model_variable=model_extreme_variable,
            reference_variable=reference_variable,
            reference_label=reference_label,
            filename_out=filename_out,
            write2file=write2file,
        )

    finally:
        model_ds.close()
        era5_ds.close()
        reference_ds.close()
