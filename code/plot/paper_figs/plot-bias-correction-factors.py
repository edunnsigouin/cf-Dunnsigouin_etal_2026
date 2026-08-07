"""
Plot the compact lookup tables used by one bias-correction method.

The script reads one bias-corrected S2S file produced by the lookup-table
bias-correction workflow and creates a publication-quality figure.

For q, doy and ld:
    panel (a): model_mean_lookup and reference_mean_lookup together
    panel (b): bias_correction_factor_lookup

For q_doy:
    panel (a): model_mean_lookup heatmap
    panel (b): reference_mean_lookup heatmap
    panel (c): bias_correction_factor_lookup heatmap

Supported bias-correction methods
---------------------------------
    q
        Lookup dimension:
            quantile_global

    doy
        Lookup dimension:
            doy

    ld
        Model/factor lookup dimension:
            lead_day

        reference_mean_lookup is a scalar because the lead-day correction
        compares each model lead-day mean with one overall reference mean.

    q_doy
        Lookup dimensions:
            doy, quantile_doy

        These two-dimensional lookup tables are plotted as heatmaps.

The reference dataset is selected independently:

    era5
    senorge

The expected input filename is:

    preprocessed_model_tp24_<catchment>_<start>_<end>_
    <X>dayacc_bc_<method>_<reference>.nc

An explicit filename can also be supplied with input_filename_override.

The script does not recalculate any bias correction. It only visualizes the
lookup tables already stored in the bias-corrected NetCDF file.
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
    "2022-12-29",
]

analysis_x_days = 2

# Options:
#     "q"
#     "doy"
#     "ld"
#     "q_doy"
bias_correction_method = "q_doy"

# Options:
#     "era5"
#     "senorge"
reference_dataset = "senorge"

# Optional explicit input filename.
#
# Leave as None to construct the filename automatically.
input_filename_override = None

write2file = True
show_figure = True

# Output figure format.
figure_extension = "png"
figure_dpi = 300

# Axis limits for q, doy and ld.
#
# Use None to let matplotlib choose that limit automatically.
lookup_x_limits = (
    None,
    None,
)

lookup_y_limits = (
    None,
    None,
)

factor_x_limits = (
    None,
    None,
)

factor_y_limits = (
    None,
    None,
)

# Colorbar limits for q_doy heatmaps.
#
# Use None for automatic limits.
q_doy_model_color_limits = (
    None,
    None,
)

q_doy_reference_color_limits = (
    None,
    None,
)

q_doy_factor_color_limits = (
    0.5,
    1.5,
)


# =============================================================================
# Fixed settings
# =============================================================================

MODEL_VARIABLE = "tp24"

LOOKUP_VARIABLES = (
    "model_mean_lookup",
    "reference_mean_lookup",
    "bias_correction_factor_lookup",
)

VALID_METHODS = {
    "q",
    "doy",
    "ld",
    "q_doy",
}

VALID_REFERENCES = {
    "era5",
    "senorge",
}


# =============================================================================
# Paths and filenames
# =============================================================================

path_in = Path(
    config.dirs[
        "s2s_processed"
    ]
)

path_out = Path(
    config.dirs[
        "fig"
    ]
)


def get_file_id(
    catchment_name,
):
    """Return the short catchment label used in model filenames."""

    if catchment_name.startswith(
        "regine_"
    ):

        return catchment_name.replace(
            "regine_",
            "",
            1,
        )

    return catchment_name


def make_input_filename():
    """Return the selected bias-corrected lookup-table file."""

    if input_filename_override is not None:

        return Path(
            input_filename_override
        )

    return (
        path_in
        / (
            f"preprocessed_model_{MODEL_VARIABLE}_"
            f"{get_file_id(catchment)}_"
            f"{forecast_date_range[0]}_"
            f"{forecast_date_range[1]}_"
            f"{analysis_x_days}dayacc_"
            f"bc_{bias_correction_method}_"
            f"{reference_dataset}.nc"
        )
    )


def make_output_filename():
    """Return the figure filename."""

    return (
        path_out
        / (
            f"bias_correction_factor_"
            f"{bias_correction_method}_"
            f"{reference_dataset}_"
            f"{analysis_x_days}dayacc_"
            f"{get_file_id(catchment)}."
            f"{figure_extension}"
        )
    )


# =============================================================================
# Validation
# =============================================================================

def validate_settings():
    """Validate user settings and input file."""

    if bias_correction_method not in VALID_METHODS:

        raise ValueError(
            "bias_correction_method must be one of "
            f"{sorted(VALID_METHODS)}."
        )

    if reference_dataset not in VALID_REFERENCES:

        raise ValueError(
            "reference_dataset must be one of "
            f"{sorted(VALID_REFERENCES)}."
        )

    if analysis_x_days < 1:

        raise ValueError(
            "analysis_x_days must be at least 1."
        )

    for name, limits in {
        "lookup_x_limits": lookup_x_limits,
        "lookup_y_limits": lookup_y_limits,
        "factor_x_limits": factor_x_limits,
        "factor_y_limits": factor_y_limits,
        "q_doy_model_color_limits": q_doy_model_color_limits,
        "q_doy_reference_color_limits": q_doy_reference_color_limits,
        "q_doy_factor_color_limits": q_doy_factor_color_limits,
    }.items():

        if (
            not isinstance(
                limits,
                tuple,
            )
            or len(
                limits
            ) != 2
        ):

            raise ValueError(
                f"{name} must be a two-element tuple."
            )

        lower, upper = limits

        if (
            lower is not None
            and upper is not None
            and lower >= upper
        ):

            raise ValueError(
                f"{name} lower limit must be smaller than upper limit."
            )

    filename = make_input_filename()

    if not filename.is_file():

        raise FileNotFoundError(
            f"Input file not found: {filename}"
        )


def validate_dataset(
    ds,
):
    """Check that the expected lookup variables are present."""

    missing = [
        variable_name
        for variable_name in LOOKUP_VARIABLES
        if variable_name not in ds
    ]

    if missing:

        raise KeyError(
            "Input dataset is missing lookup variables: "
            f"{missing}"
        )


# =============================================================================
# Plot helpers
# =============================================================================

def method_display_name():
    """Return a compact method label for the figure."""

    labels = {
        "q": "Quantile",
        "doy": "Day of year",
        "ld": "Lead day",
        "q_doy": "Quantile + day of year",
    }

    return labels[
        bias_correction_method
    ]


def reference_display_name():
    """Return a publication-style reference label."""

    if reference_dataset == "era5":
        return "ERA5"

    return "SeNorge"


def get_x_axis_information(
    data,
):
    """Return x coordinate, x-axis label, and tick behaviour for 1-D lookups."""

    if bias_correction_method == "q":

        coordinate = np.asarray(
            data[
                "quantile_global"
            ].values
        )

        # Convert quantile index to percentile-range label position.
        # The values are normally integer indices 0..19.
        if np.issubdtype(
            coordinate.dtype,
            np.integer,
        ):

            x = (
                coordinate.astype(
                    float
                )
                + 0.5
            ) * 5.0

        else:

            x = coordinate.astype(
                float
            )

            if np.nanmax(
                x
            ) <= 1.0:

                x = x * 100.0

        return (
            x,
            "Global quantile (%)",
        )

    if bias_correction_method == "doy":

        return (
            np.asarray(
                data[
                    "doy"
                ].values
            ),
            "Day of year",
        )

    if bias_correction_method == "ld":

        if "lead_day" not in data.dims:

            return (
                np.array(
                    [
                        0.0,
                    ]
                ),
                "Lead day",
            )

        return (
            np.asarray(
                data[
                    "lead_day"
                ].values
            ),
            "Lead day",
        )

    raise ValueError(
        "1-D axis information requested for a 2-D q_doy lookup."
    )


def add_panel_label(
    axis,
    label,
):
    """Add a small panel label in the upper-left corner."""

    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )


def style_axis(
    axis,
):
    """Apply restrained publication-style formatting."""

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        direction="out",
        length=4,
        width=0.8,
    )

    axis.grid(
        axis="y",
        linewidth=0.5,
        alpha=0.25,
    )


def apply_axis_limits(
    axis,
    x_limits,
    y_limits,
):
    """Apply optional user-provided axis limits."""

    x_min, x_max = x_limits
    y_min, y_max = y_limits

    current_x_min, current_x_max = axis.get_xlim()
    current_y_min, current_y_max = axis.get_ylim()

    axis.set_xlim(
        current_x_min
        if x_min is None
        else x_min,
        current_x_max
        if x_max is None
        else x_max,
    )

    axis.set_ylim(
        current_y_min
        if y_min is None
        else y_min,
        current_y_max
        if y_max is None
        else y_max,
    )


def resolved_color_limits(
    limits,
):
    """Return vmin and vmax from an optional (min, max) tuple."""

    return (
        limits[0],
        limits[1],
    )


def plot_combined_lookup(
    axis,
    model_data,
    reference_data,
    panel_label,
):
    """Plot model and reference lookup tables together."""

    model_values = np.asarray(
        model_data.values
    ).squeeze()

    reference_values = np.asarray(
        reference_data.values
    ).squeeze()

    if bias_correction_method == "ld":

        x, x_label = get_x_axis_information(
            model_data
        )

        axis.plot(
            x,
            model_values,
            linewidth=1.8,
            color="tab:blue",
            label="Model",
        )

        if reference_values.ndim == 0:

            axis.axhline(
                float(
                    reference_values
                ),
                linewidth=1.8,
                color="tab:red",
                label=reference_display_name(),
            )

        else:

            axis.plot(
                x,
                reference_values,
                linewidth=1.8,
                color="tab:red",
                label=reference_display_name(),
            )

    else:

        x, x_label = get_x_axis_information(
            model_data
        )

        if bias_correction_method == "q":

            axis.plot(
                x,
                model_values,
                linewidth=1.8,
                marker="o",
                markerfacecolor="none",
                markeredgecolor="tab:blue",
                markersize=5,
                color="tab:blue",
                label="Model",
            )

            axis.plot(
                x,
                reference_values,
                linewidth=1.8,
                marker="o",
                markerfacecolor="none",
                markeredgecolor="tab:red",
                markersize=5,
                color="tab:red",
                label=reference_display_name(),
            )

        else:

            axis.plot(
                x,
                model_values,
                linewidth=1.8,
                color="tab:blue",
                label="Model",
            )

            axis.plot(
                x,
                reference_values,
                linewidth=1.8,
                color="tab:red",
                label=reference_display_name(),
            )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        "Precipitation (mm)"
    )

    axis.set_title(
        "Lookup means",
        fontsize=11,
        pad=8,
    )

    axis.legend(
        frameon=False,
        loc="best",
    )

    add_panel_label(
        axis,
        panel_label,
    )

    style_axis(
        axis
    )

    apply_axis_limits(
        axis=axis,
        x_limits=lookup_x_limits,
        y_limits=lookup_y_limits,
    )


def plot_factor_lookup(
    axis,
    data,
    panel_label,
):
    """Plot the compact multiplicative correction factor."""

    values = np.asarray(
        data.values
    ).squeeze()

    x, x_label = get_x_axis_information(
        data
    )

    if bias_correction_method == "q":

        axis.plot(
            x,
            values,
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=5,
            linewidth=1.8,
            color="black",
        )

    else:

        axis.plot(
            x,
            values,
            linewidth=1.8,
            color="black",
        )

    axis.axhline(
        1.0,
        linewidth=1.0,
        linestyle="--",
        color="black",
        alpha=0.6,
    )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        "Multiplicative factor"
    )

    axis.set_title(
        "Bias-correction factor",
        fontsize=11,
        pad=8,
    )

    add_panel_label(
        axis,
        panel_label,
    )

    style_axis(
        axis
    )

    apply_axis_limits(
        axis=axis,
        x_limits=factor_x_limits,
        y_limits=factor_y_limits,
    )


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
    """Plot one two-dimensional q_doy lookup table."""

    data = data.transpose(
        "doy",
        "quantile_doy",
    )

    values = np.asarray(
        data.values
    )

    quantiles = np.asarray(
        data[
            "quantile_doy"
        ].values
    )

    if np.nanmax(
        quantiles
    ) <= 1.0:

        quantiles = (
            quantiles
            * 100.0
        )

    vmin, vmax = resolved_color_limits(
        color_limits
    )

    image = axis.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            float(
                data[
                    "doy"
                ].min()
            ),
            float(
                data[
                    "doy"
                ].max()
            ),
            float(
                np.nanmin(
                    quantiles
                )
            ),
            float(
                np.nanmax(
                    quantiles
                )
            ),
        ),
        interpolation="nearest",
    )

    axis.set_xlabel(
        "Day of year"
    )

    axis.set_ylabel(
        "Seasonal quantile (%)"
    )

    axis.set_title(
        title,
        fontsize=11,
        pad=8,
    )

    axis.set_xticks(
        [
            1,
            60,
            121,
            182,
            244,
            305,
            366,
        ]
    )

    add_panel_label(
        axis,
        panel_label,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.02,
        location='bottom',
        fraction=0.05,
    )

    colorbar.set_label(
        colorbar_label
    )


# =============================================================================
# Main figure
# =============================================================================

def make_lookup_figure(
    ds,
    filename=None,
):
    """Create the lookup-table figure."""

    model_lookup = ds[
        "model_mean_lookup"
    ]

    reference_lookup = ds[
        "reference_mean_lookup"
    ]

    factor_lookup = ds[
        "bias_correction_factor_lookup"
    ]

    if bias_correction_method == "q_doy":

        figure, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(
                10,
                4,
            ),
            constrained_layout=True,
        )

        plot_q_doy_lookup(
            figure=figure,
            axis=axes[0],
            data=model_lookup,
            title="Model mean",
            colorbar_label="Precipitation (mm)",
            panel_label="(a)",
            color_limits=q_doy_model_color_limits,
            colormap="GnBu",
        )

        plot_q_doy_lookup(
            figure=figure,
            axis=axes[1],
            data=reference_lookup,
            title=f"{reference_display_name()} mean",
            colorbar_label="Precipitation (mm)",
            panel_label="(b)",
            color_limits=q_doy_reference_color_limits,
            colormap="GnBu",
        )

        plot_q_doy_lookup(
            figure=figure,
            axis=axes[2],
            data=factor_lookup,
            title="Bias-correction factor",
            colorbar_label="Multiplicative factor",
            panel_label="(c)",
            color_limits=q_doy_factor_color_limits,
            colormap="RdBu",
        )

    else:

        figure, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(
                8.2,
                3.6,
            ),
            constrained_layout=True,
        )

        plot_combined_lookup(
            axis=axes[0],
            model_data=model_lookup,
            reference_data=reference_lookup,
            panel_label="(a)",
        )

        plot_factor_lookup(
            axis=axes[1],
            data=factor_lookup,
            panel_label="(b)",
        )

    figure.suptitle(
        (
            f"{method_display_name()} bias correction "
            f"({reference_display_name()}, "
            f"{analysis_x_days}-day accumulation)"
        ),
        fontsize=12,
        y=1.04,
    )

    if filename is not None:

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            filename,
            dpi=figure_dpi,
            bbox_inches="tight",
        )

        print(
            "Wrote:",
            filename,
        )

    if show_figure:

        plt.show()

    plt.close(
        figure
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    validate_settings()

    filename_input = make_input_filename()

    filename_output = make_output_filename()

    print(
        "Input file:"
    )
    print(
        filename_input
    )

    print()
    print(
        "Plot settings"
    )
    print(
        "-------------"
    )
    print(
        "Bias-correction method:",
        bias_correction_method,
    )
    print(
        "Reference dataset:",
        reference_dataset,
    )
    print(
        "Analysis accumulation:",
        analysis_x_days,
    )

    with xr.open_dataset(
        filename_input,
        decode_timedelta=False,
    ) as ds:

        validate_dataset(
            ds
        )

        print()
        print(
            "Lookup dimensions"
        )
        print(
            "-----------------"
        )

        for variable_name in LOOKUP_VARIABLES:

            print(
                f"{variable_name}:",
                ds[
                    variable_name
                ].dims,
            )

        if bias_correction_method in ("q", "doy", "ld"):

            print()
            print("Lookup values")
            print("-------------")

            model_lookup = ds["model_mean_lookup"]
            reference_lookup = ds["reference_mean_lookup"]

            if bias_correction_method == "q":
                coord = "quantile_global"
            elif bias_correction_method == "doy":
                coord = "doy"
            else:
                coord = "lead_day"

            print(f"{'Index':>8} {'Model':>15} {'Reference':>15}")

            for value in model_lookup[coord].values:

                m = float(model_lookup.sel({coord: value}).values)

                if reference_lookup.ndim == 0:
                    r = float(reference_lookup.values)
                else:
                    r = float(reference_lookup.sel({coord: value}).values)

                print(f"{int(value):>8d} {m:15.4f} {r:15.4f}")

        make_lookup_figure(
            ds=ds,
            filename=(
                filename_output
                if write2file
                else None
            ),
        )
