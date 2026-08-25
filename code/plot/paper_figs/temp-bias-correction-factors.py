#!/usr/bin/env python3
"""
Plot stored bias-correction lookup tables or monthly MM correction factors.

Supported methods
-----------------
q, doy, ld
    Read the bias-corrected preprocessed-model files produced by the lookup-table
    bias-correction script. Panel (a) shows the model and reference lookup means;
    panel (b) shows the multiplicative bias-correction factor.

q_doy
    Read the same lookup-table output family. Panels (a)-(c) show the model mean,
    reference mean, and multiplicative bias-correction factor as day-of-year by
    quantile heatmaps.

mm_1step
    Read the compact monthly-maximum sample produced by the monthly-sample script.
    Plot the stored monthly reference bias-correction ratio for calendar months
    January-December.

mm_2step
    Read the corresponding compact monthly-maximum sample. Panel (a) plots the
    stored monthly lead-time correction ratios for the Early and Late lead bins.
    Panel (b) plots the stored monthly reference correction ratio applied after
    the lead-time correction. This second-stage reference ratio is common to the
    corrected full, Early, and Late samples; separate Early/Late reference ratios
    are not stored by the sample-building script.

The script only visualizes correction factors already stored in the selected
NetCDF file. It does not recalculate bias corrections.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

catchment = "regine_drammen"

forecast_date_range = [
    "2020-01-02",
    "2023-12-28",
]

reference_years = [
    "1957",
    "2022",
]

analysis_x_days = 2

# Options: "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
bias_correction_method = "mm_2step"

# Options: "era5", "senorge"
reference_dataset = "senorge"

# Optional explicit input filename. Leave as None to construct it automatically.
input_filename_override = None

write2file = False
show_figure = True

figure_extension = "png"
figure_dpi = 300

# Optional axis limits for 1-D lookup/MM plots. Use None for automatic limits.
lookup_x_limits = (None, None)
lookup_y_limits = (None, None)
factor_x_limits = (None, None)
factor_y_limits = (None, None)

# Optional color limits for q_doy heatmaps.
q_doy_model_color_limits = (None, None)
q_doy_reference_color_limits = (None, None)
q_doy_factor_color_limits = (0.5, 1.5)


# =============================================================================
# Fixed settings
# =============================================================================

MODEL_VARIABLE = "tp24"
MONTHS = np.arange(1, 13)

LOOKUP_METHODS = {"q", "doy", "ld", "q_doy"}
MM_METHODS = {"mm_1step", "mm_2step"}
VALID_METHODS = LOOKUP_METHODS | MM_METHODS
VALID_REFERENCES = {"era5", "senorge"}

LOOKUP_VARIABLES = (
    "model_mean_lookup",
    "reference_mean_lookup",
    "bias_correction_factor_lookup",
)

path_in = Path(config.dirs["s2s_processed"])
path_out = Path(config.dirs["fig"])


# =============================================================================
# Filenames and validation
# =============================================================================

def get_file_id(catchment_name):
    """Return the short catchment label used in model filenames."""
    return catchment_name.removeprefix("regine_")


def make_input_filename():
    """Return the selected lookup-table or MM monthly-sample input file."""
    if input_filename_override is not None:
        return Path(input_filename_override)

    if bias_correction_method in LOOKUP_METHODS:
        return path_in / (
            f"preprocessed_model_{MODEL_VARIABLE}_{get_file_id(catchment)}_"
            f"{forecast_date_range[0]}_{forecast_date_range[1]}_"
            f"{analysis_x_days}dayacc_bc_{bias_correction_method}_{reference_dataset}_"
            f"{reference_years[0]}-{reference_years[1]}.nc"
        )

    return path_in / (
        f"monthly_max_samples_{MODEL_VARIABLE}_{analysis_x_days}dayacc_"
        f"{get_file_id(catchment)}_{forecast_date_range[0]}_{forecast_date_range[1]}_"
        f"bc_{bias_correction_method}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}.nc"
    )


def make_output_filename():
    """Return a descriptive figure filename."""
    return path_out / (
        f"bias_correction_factor_{bias_correction_method}_{reference_dataset}_"
        f"{reference_years[0]}-{reference_years[1]}_{analysis_x_days}dayacc_"
        f"{get_file_id(catchment)}.{figure_extension}"
    )


def validate_limits(name, limits):
    """Validate one optional two-element plotting-limit tuple."""
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError(f"{name} must be a two-element tuple.")

    lower, upper = limits
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{name} lower limit must be smaller than upper limit.")


def validate_settings():
    """Validate user settings and the selected input file."""
    if bias_correction_method not in VALID_METHODS:
        raise ValueError(
            f"bias_correction_method must be one of {sorted(VALID_METHODS)}."
        )
    if reference_dataset not in VALID_REFERENCES:
        raise ValueError(f"reference_dataset must be one of {sorted(VALID_REFERENCES)}.")
    if analysis_x_days < 1:
        raise ValueError("analysis_x_days must be at least 1.")
    if int(reference_years[0]) > int(reference_years[1]):
        raise ValueError("reference_years must be increasing.")

    for name, limits in {
        "lookup_x_limits": lookup_x_limits,
        "lookup_y_limits": lookup_y_limits,
        "factor_x_limits": factor_x_limits,
        "factor_y_limits": factor_y_limits,
        "q_doy_model_color_limits": q_doy_model_color_limits,
        "q_doy_reference_color_limits": q_doy_reference_color_limits,
        "q_doy_factor_color_limits": q_doy_factor_color_limits,
    }.items():
        validate_limits(name, limits)

    filename = make_input_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")


def validate_dataset(ds):
    """Check that the variables required by the selected method are present."""
    if bias_correction_method in LOOKUP_METHODS:
        missing = [name for name in LOOKUP_VARIABLES if name not in ds]
        if missing:
            raise KeyError(f"Input dataset is missing lookup variables: {missing}.")
        return

    if "bias_correction_ratio" not in ds:
        raise KeyError("MM input dataset is missing 'bias_correction_ratio'.")

    if bias_correction_method == "mm_2step":
        lead_ratio_names = get_lead_time_ratio_names(ds)
        if len(lead_ratio_names) < 2:
            raise KeyError(
                "mm_2step input must contain at least two "
                "'lead_time_bias_correction_ratio_*' variables."
            )


# =============================================================================
# Labels and general plotting helpers
# =============================================================================

def method_display_name():
    """Return a compact display name for the selected correction method."""
    return {
        "q": "Quantile",
        "doy": "Day of year",
        "ld": "Lead day",
        "q_doy": "Quantile + day of year",
        "mm_1step": "MM 1-step",
        "mm_2step": "MM 2-step",
    }[bias_correction_method]


def reference_display_name():
    """Return a publication-style reference label."""
    return "ERA5" if reference_dataset == "era5" else "SeNorge"


def add_panel_label(axis, label):
    """Add a panel label in the upper-left corner."""
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )


def style_axis(axis):
    """Apply restrained publication-style formatting."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=4, width=0.8)
    axis.grid(axis="y", linewidth=0.5, alpha=0.25)


def apply_axis_limits(axis, x_limits, y_limits):
    """Apply optional user-provided axis limits."""
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    current_x_min, current_x_max = axis.get_xlim()
    current_y_min, current_y_max = axis.get_ylim()

    axis.set_xlim(
        current_x_min if x_min is None else x_min,
        current_x_max if x_max is None else x_max,
    )
    axis.set_ylim(
        current_y_min if y_min is None else y_min,
        current_y_max if y_max is None else y_max,
    )


def resolved_color_limits(limits):
    """Return vmin and vmax from an optional (min, max) tuple."""
    return limits[0], limits[1]


def add_factor_reference_line(axis):
    """Draw the no-correction factor of one."""
    axis.axhline(1.0, linewidth=1.0, linestyle="--", color="black", alpha=0.6)


# =============================================================================
# Lookup-table plotting: q, doy, ld, q_doy
# =============================================================================

def get_x_axis_information(data):
    """Return x coordinates and label for one-dimensional lookup tables."""
    if bias_correction_method == "q":
        coordinate = np.asarray(data["quantile_global"].values)
        x = coordinate.astype(float)
        if np.issubdtype(coordinate.dtype, np.integer):
            x = (x + 0.5) * 5.0
        elif np.nanmax(x) <= 1.0:
            x *= 100.0
        return x, "Global quantile (%)"

    if bias_correction_method == "doy":
        return np.asarray(data["doy"].values), "Day of year"

    if bias_correction_method == "ld":
        if "lead_day" not in data.dims:
            return np.array([0.0]), "Lead day"
        return np.asarray(data["lead_day"].values), "Lead day"

    raise ValueError("1-D axis information requested for q_doy.")


def plot_combined_lookup(axis, model_data, reference_data, panel_label):
    """Plot model and reference means for q, doy, or ld."""
    x, x_label = get_x_axis_information(model_data)
    model_values = np.asarray(model_data.values).squeeze()
    reference_values = np.asarray(reference_data.values).squeeze()

    marker = "o" if bias_correction_method == "q" else None
    axis.plot(
        x,
        model_values,
        linewidth=1.8,
        marker=marker,
        markerfacecolor="none" if marker else None,
        color="tab:blue",
        label="Model",
    )

    if reference_values.ndim == 0:
        axis.axhline(
            float(reference_values),
            linewidth=1.8,
            color="tab:red",
            label=reference_display_name(),
        )
    else:
        axis.plot(
            x,
            reference_values,
            linewidth=1.8,
            marker=marker,
            markerfacecolor="none" if marker else None,
            color="tab:red",
            label=reference_display_name(),
        )

    axis.set_xlabel(x_label)
    axis.set_ylabel("Precipitation (mm)")
    axis.set_title("Lookup means", fontsize=11, pad=8)
    axis.legend(frameon=False, loc="best")
    add_panel_label(axis, panel_label)
    style_axis(axis)
    apply_axis_limits(axis, lookup_x_limits, lookup_y_limits)


def plot_factor_lookup(axis, data, panel_label):
    """Plot one compact multiplicative lookup-table factor."""
    x, x_label = get_x_axis_information(data)
    values = np.asarray(data.values).squeeze()
    marker = "o" if bias_correction_method == "q" else None

    axis.plot(
        x,
        values,
        linewidth=1.8,
        marker=marker,
        markerfacecolor="none" if marker else None,
        color="black",
    )
    add_factor_reference_line(axis)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Multiplicative factor")
    axis.set_title("Bias-correction factor", fontsize=11, pad=8)
    add_panel_label(axis, panel_label)
    style_axis(axis)
    apply_axis_limits(axis, factor_x_limits, factor_y_limits)


def plot_q_doy_lookup(
    figure,
    axis,
    data,
    title,
    colorbar_label,
    panel_label,
    color_limits,
    colormap,
):
    """Plot one day-of-year by quantile lookup table."""
    data = data.transpose("doy", "quantile_doy")
    values = np.asarray(data.values)
    quantiles = np.asarray(data["quantile_doy"].values, dtype=float)
    if np.nanmax(quantiles) <= 1.0:
        quantiles *= 100.0

    vmin, vmax = resolved_color_limits(color_limits)
    image = axis.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            float(data["doy"].min()),
            float(data["doy"].max()),
            float(np.nanmin(quantiles)),
            float(np.nanmax(quantiles)),
        ),
        interpolation="nearest",
    )

    axis.set_xlabel("Day of year")
    axis.set_ylabel("Seasonal quantile (%)")
    axis.set_title(title, fontsize=11, pad=8)
    axis.set_xticks([1, 60, 121, 182, 244, 305, 366])
    add_panel_label(axis, panel_label)

    colorbar = figure.colorbar(image, ax=axis, pad=0.02, location="bottom", fraction=0.05)
    colorbar.set_label(colorbar_label)


def make_lookup_figure(ds):
    """Create the q, doy, ld, or q_doy lookup-table figure."""
    model_lookup = ds["model_mean_lookup"]
    reference_lookup = ds["reference_mean_lookup"]
    factor_lookup = ds["bias_correction_factor_lookup"]

    if bias_correction_method == "q_doy":
        figure, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(10, 4),
            constrained_layout=True,
        )
        plot_q_doy_lookup(
            figure,
            axes[0],
            model_lookup,
            "Model mean",
            "Precipitation (mm)",
            "(a)",
            q_doy_model_color_limits,
            "GnBu",
        )
        plot_q_doy_lookup(
            figure,
            axes[1],
            reference_lookup,
            f"{reference_display_name()} mean",
            "Precipitation (mm)",
            "(b)",
            q_doy_reference_color_limits,
            "GnBu",
        )
        plot_q_doy_lookup(
            figure,
            axes[2],
            factor_lookup,
            "Bias-correction factor",
            "Multiplicative factor",
            "(c)",
            q_doy_factor_color_limits,
            "RdBu",
        )
        return figure

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.2, 3.6),
        constrained_layout=True,
    )
    plot_combined_lookup(axes[0], model_lookup, reference_lookup, "(a)")
    plot_factor_lookup(axes[1], factor_lookup, "(b)")
    return figure


# =============================================================================
# MM plotting: mm_1step and mm_2step
# =============================================================================

def get_month_coordinate(data):
    """Return the 1-12 calendar-month coordinate from one stored MM ratio."""
    if "month_of_year" not in data.dims:
        raise ValueError(
            f"MM ratio '{data.name}' must have dimension 'month_of_year'; "
            f"found {data.dims}."
        )

    months = np.asarray(data["month_of_year"].values)
    if not np.array_equal(months.astype(int), MONTHS):
        raise ValueError(
            f"MM ratio '{data.name}' must contain calendar months 1-12 in order."
        )
    return months.astype(int)


def get_lead_time_ratio_names(ds):
    """Return stored mm_2step lead-time ratio variables in lead order."""
    names = [
        name
        for name in ds.data_vars
        if name.startswith("lead_time_bias_correction_ratio_")
    ]

    def lead_start(name):
        source = ds[name].attrs.get("source_variable", name)
        lead_text = source.split("lead", 1)[-1].split("_", 1)[0]
        try:
            return int(lead_text)
        except ValueError:
            return 10_000

    return sorted(names, key=lead_start)


def lead_ratio_label(data, index, total):
    """Return a readable Early/Late label for one lead-time correction ratio."""
    source = data.attrs.get("source_variable", "")
    if "lead" in source:
        lead_range = source.split("lead", 1)[1].replace("_", "-")
    else:
        lead_range = data.name.replace("lead_time_bias_correction_ratio_lead", "").replace("_", "-")

    if total == 2:
        period = "Early" if index == 0 else "Late"
    else:
        period = f"Lead bin {index + 1}"

    return f"{period} ({lead_range})"


def plot_mm_reference_factor(axis, ratio, panel_label=None):
    """Plot the monthly MM reference correction ratio."""
    months = get_month_coordinate(ratio)
    axis.plot(months, ratio.values, marker="o", linewidth=1.8, color="black")
    add_factor_reference_line(axis)
    axis.set_xticks(MONTHS)
    axis.set_xlabel("Calendar month")
    axis.set_ylabel("Multiplicative factor")
    axis.set_title(
        f"Reference correction relative to {reference_display_name()}",
        fontsize=11,
        pad=8,
    )
    if panel_label is not None:
        add_panel_label(axis, panel_label)
    style_axis(axis)
    apply_axis_limits(axis, factor_x_limits, factor_y_limits)


def make_mm_1step_figure(ds):
    """Plot the single monthly reference factor stored for mm_1step."""
    ratio = ds["bias_correction_ratio"]
    figure, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    plot_mm_reference_factor(axis, ratio)
    return figure


def make_mm_2step_figure(ds):
    """Plot mm_2step lead-time and reference correction factors."""
    lead_ratio_names = get_lead_time_ratio_names(ds)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.0, 3.8),
        constrained_layout=True,
    )

    for index, name in enumerate(lead_ratio_names):
        ratio = ds[name]
        months = get_month_coordinate(ratio)
        axes[0].plot(
            months,
            ratio.values,
            marker="o",
            linewidth=1.8,
            label=lead_ratio_label(ratio, index, len(lead_ratio_names)),
        )

    add_factor_reference_line(axes[0])
    axes[0].set_xticks(MONTHS)
    axes[0].set_xlabel("Calendar month")
    axes[0].set_ylabel("Multiplicative factor")
    axes[0].set_title("Lead-time correction", fontsize=11, pad=8)
    axes[0].legend(frameon=False, loc="best")
    add_panel_label(axes[0], "(a)")
    style_axis(axes[0])
    apply_axis_limits(axes[0], factor_x_limits, factor_y_limits)

    plot_mm_reference_factor(axes[1], ds["bias_correction_ratio"], "(b)")
    return figure


def make_figure(ds):
    """Create the figure appropriate for the selected correction method."""
    if bias_correction_method in LOOKUP_METHODS:
        return make_lookup_figure(ds)
    if bias_correction_method == "mm_1step":
        return make_mm_1step_figure(ds)
    return make_mm_2step_figure(ds)


# =============================================================================
# Main
# =============================================================================

def main():
    """Read stored correction factors and create the selected figure."""
    validate_settings()
    filename_input = make_input_filename()
    filename_output = make_output_filename()

    print("Input file:", filename_input)
    print("Bias-correction method:", bias_correction_method)
    print("Reference dataset:", reference_dataset)
    print("Reference years:", f"{reference_years[0]}-{reference_years[1]}")
    print("Analysis accumulation:", analysis_x_days)

    with xr.open_dataset(filename_input, decode_timedelta=False) as ds:
        validate_dataset(ds)
        figure = make_figure(ds)

    figure.suptitle(
        (
            f"{method_display_name()} bias correction "
            f"({reference_display_name()}, {analysis_x_days}-day accumulation)"
        ),
        fontsize=12,
        y=1.04,
    )

    if write2file:
        filename_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename_output, dpi=figure_dpi, bbox_inches="tight")
        print("Wrote:", filename_output)

    if show_figure:
        plt.show()

    plt.close(figure)


if __name__ == "__main__":
    main()
