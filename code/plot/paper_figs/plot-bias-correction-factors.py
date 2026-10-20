"""
Plot the correction tables stored by one bias-correction method.

The script reads files already written by the bias-correction workflow and
visualises the tables they contain. No bias correction is recalculated.

Layout
------
q, doy, ld, mm_1step, mm_2step (1 x 2)
    (a) Model and reference means
    (b) Multiplicative bias-correction factor
q_doy (1 x 3)
    (a) Model mean heatmap        (b) Reference mean heatmap
    (c) Bias-correction factor heatmap

Input files
-----------
q, doy, ld, q_doy  (lookup-table bias-correction script):
    preprocessed_model_tp24_<catchment>_<start>_<end>_<X>dayacc_bc_<method>_
    <reference>_<year0>-<year1>.nc
    with model_mean_lookup, reference_mean_lookup, bias_correction_factor_lookup.

mm_1step, mm_2step  (monthly-maximum sample script):
    monthly_max_samples_tp24_<X>dayacc_<catchment>_<start>_<end>_bc_<method>_
    <reference>_<year0>-<year1>.nc
    with bias_correction_ratio(month_of_year) and, for mm_2step,
    lead_time_bias_correction_ratio_lead<a>_<b>(month_of_year).
    The MM files store only the ratios, so the reference monthly mean is read
    from the reference monthly-maximum file (with August 2023 excluded when
    EXCLUDE_STORM_HANS_FROM_REFERENCE is True, as in the sample script) and the
    model monthly mean is recovered exactly as reference mean / ratio. For
    mm_2step this is the mean of the lead-time-corrected sample, i.e. the model
    mean that the reference ratio was actually calculated against.

Lookup dimensions expected per method
-------------------------------------
    q          quantile_global
    doy        doy
    ld         lead_day (reference_mean_lookup is a scalar: one overall mean)
    q_doy      doy, quantile_doy
    mm_1step   month_of_year
    mm_2step   month_of_year

Quantile coordinates stored as integer bin indices 0..N-1 are plotted at bin
centres, (i + 0.5) * 100 / N; fractional coordinates (<= 1) are converted to %.

Heatmap cells are drawn with pcolormesh on cell edges derived from the stored
coordinates, so the first and last day-of-year and quantile bins are shown at
full width. The factor colour scale is centred on 1 (no correction). Model and
reference heatmaps share one colour scale by default so they can be compared.
"""
from calendar import month_abbr
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import xarray as xr

from Dunnsigouin_etal_2026 import config

# =============================================================================
# USER INPUTS — data selection
# =============================================================================
CATCHMENT = "regine_drammen"
FORECAST_DATE_RANGE = ("2020-01-02", "2023-12-28")
REFERENCE_YEARS = ("1957", "2025")  # Reference period encoded in the filenames.
X_DAYS = 2
BIAS_CORRECTION_METHOD = "q_doy"  # "q", "doy", "ld", "q_doy", "mm_1step", or "mm_2step"
REFERENCE_DATASET = "senorge"  # "era5" or "senorge"

# mm_1step and mm_2step only: must match the setting used to build the MM samples,
# since the reference monthly mean is recalculated here.
EXCLUDE_STORM_HANS_FROM_REFERENCE = True

INPUT_FILENAME_OVERRIDE = None  # Optional explicit input path.
REFERENCE_FILENAME_OVERRIDE = None  # Optional explicit reference path (MM methods only).

# =============================================================================
# FIGURE OUTPUT
# =============================================================================
WRITE_TO_FILE = True
SHOW_FIGURE = True
FIGURE_EXTENSION = "png"  # "png", "pdf", or "svg"
FIGURE_DPI = 300
OUTPUT_FILENAME_OVERRIDE = None  # Optional explicit output path.

# =============================================================================
# FIGURE APPEARANCE
# =============================================================================
SHOW_SUPTITLE = False  # Usually off for a paper; the caption carries this.

# Axis limits for the 1-D methods; None lets matplotlib choose that limit.
LOOKUP_X_LIMITS = (None, None)
LOOKUP_Y_LIMITS = (0.0, None)
FACTOR_X_LIMITS = (None, None)
FACTOR_Y_LIMITS = (None, None)
FACTOR_LOG_SCALE = False  # Log y-axis makes factors 0.5 and 2 equally far from 1.

# Colour limits for q_doy heatmaps; None uses the data range.
Q_DOY_SHARED_MEAN_COLOR_LIMITS = True  # One scale for model and reference panels.
Q_DOY_MEAN_COLOR_LIMITS = (None, None)
Q_DOY_FACTOR_COLOR_LIMITS = (0.5, 1.5)  # Must bracket 1; None gives a range symmetric about 1.

FIGURE_SIZE_1D = (9.0, 3.6)
FIGURE_SIZE_Q_DOY = (12.0, 4.4)
MEAN_COLORMAP = "GnBu"
FACTOR_COLORMAP = "BrBG"  # Brown: drier after correction; green: wetter.
MODEL_COLOR = "tab:blue"  # Same model/reference colours as fig-05.
REFERENCE_COLOR = "tab:orange"
FACTOR_COLOR = "black"
LEAD_RATIO_COLORS = ("tab:green", "tab:purple", "tab:brown", "tab:pink")
LINEWIDTH = 1.8
MARKER_SIZE = 4
TITLE_FONTSIZE = 11

# =============================================================================
# Fixed settings
# =============================================================================
MODEL_VARIABLE = "tp24"
REFERENCE_VARIABLES = {"era5": "tp24", "senorge": "rr"}
LOOKUP_METHODS = ("q", "doy", "ld", "q_doy")
MM_METHODS = ("mm_1step", "mm_2step")
LOOKUP_VARIABLES = ("model_mean_lookup", "reference_mean_lookup", "bias_correction_factor_lookup")
MM_RATIO_VARIABLE = "bias_correction_ratio"
MM_LEAD_RATIO_PREFIX = "lead_time_bias_correction_ratio_"
EXPECTED_DIMS = {
    "q": ("quantile_global",),
    "doy": ("doy",),
    "ld": ("lead_day",),
    "q_doy": ("doy", "quantile_doy"),
    "mm_1step": ("month_of_year",),
    "mm_2step": ("month_of_year",),
}
METHOD_NAMES = {
    "q": "Quantile",
    "doy": "Day of year",
    "ld": "Lead day",
    "q_doy": "Quantile + day of year",
    "mm_1step": "Monthly mean (one step)",
    "mm_2step": "Monthly mean (two step)",
}
X_LABELS = {
    "quantile_global": "Quantile (%)",
    "doy": "Day of year",
    "lead_day": "Lead day",
    "month_of_year": "Month",
}
MONTH_START_DOY = (1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
MONTHS = np.arange(1, 13)
STORM_HANS_YEAR, STORM_HANS_MONTH = 2023, 8

# =============================================================================
# Filenames and validation
# =============================================================================


def get_file_id(catchment_name):
    """Return the short catchment label used in model filenames."""
    return catchment_name.removeprefix("regine_")


def is_mm_method():
    """Return True for the monthly-mean methods stored in the sample files."""
    return BIAS_CORRECTION_METHOD in MM_METHODS


def reference_display_name():
    """Return the publication label of the reference dataset."""
    return {"era5": "ERA5", "senorge": "seNorge"}[REFERENCE_DATASET]


def correction_label():
    """Return the shared bc_<method>_<reference>_<years> filename label."""
    return (f"bc_{BIAS_CORRECTION_METHOD}_{REFERENCE_DATASET}_"
            f"{REFERENCE_YEARS[0]}-{REFERENCE_YEARS[1]}")


def make_input_filename():
    """Return the file holding the selected method's correction tables."""
    if INPUT_FILENAME_OVERRIDE is not None:
        return Path(INPUT_FILENAME_OVERRIDE)
    start, end = FORECAST_DATE_RANGE
    if is_mm_method():
        # Naming of the monthly-maximum sample script.
        name = (f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_{get_file_id(CATCHMENT)}_"
                f"{start}_{end}_{correction_label()}.nc")
    else:
        # Naming of the lookup-table bias-correction script.
        name = (f"preprocessed_model_{MODEL_VARIABLE}_{get_file_id(CATCHMENT)}_{start}_{end}_"
                f"{X_DAYS}dayacc_{correction_label()}.nc")
    return Path(config.dirs["s2s_processed"]) / name


def make_reference_filename():
    """Return the reference monthly-maximum file used by the MM correction."""
    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)
    return Path(config.dirs[f"{REFERENCE_DATASET}_processed"]) / (
        f"monthly_max_samples_{REFERENCE_VARIABLES[REFERENCE_DATASET]}_{X_DAYS}dayacc_"
        f"{CATCHMENT}_{REFERENCE_YEARS[0]}-{REFERENCE_YEARS[1]}.nc"
    )


def make_output_filename():
    """Return the figure filename."""
    if OUTPUT_FILENAME_OVERRIDE is not None:
        return Path(OUTPUT_FILENAME_OVERRIDE)
    return Path(config.dirs["fig"]) / (
        f"bias_correction_factor_{BIAS_CORRECTION_METHOD}_{REFERENCE_DATASET}_"
        f"{X_DAYS}dayacc_{get_file_id(CATCHMENT)}.{FIGURE_EXTENSION}"
    )


def check_limits(name, limits):
    """Check that a limit setting is an increasing (lower, upper) pair."""
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise ValueError(f"{name} must be a two-element tuple.")
    lower, upper = limits
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{name}: lower limit must be smaller than upper limit.")


def validate_settings():
    """Validate user settings before reading data."""
    if BIAS_CORRECTION_METHOD not in EXPECTED_DIMS:
        raise ValueError(f"BIAS_CORRECTION_METHOD must be one of {sorted(EXPECTED_DIMS)}.")
    if REFERENCE_DATASET not in REFERENCE_VARIABLES:
        raise ValueError("REFERENCE_DATASET must be 'era5' or 'senorge'.")
    if X_DAYS < 1:
        raise ValueError("X_DAYS must be at least 1.")
    if int(REFERENCE_YEARS[0]) > int(REFERENCE_YEARS[1]):
        raise ValueError("REFERENCE_YEARS must be increasing.")
    if not isinstance(EXCLUDE_STORM_HANS_FROM_REFERENCE, bool):
        raise TypeError("EXCLUDE_STORM_HANS_FROM_REFERENCE must be True or False.")
    if FIGURE_EXTENSION not in {"png", "pdf", "svg"}:
        raise ValueError("FIGURE_EXTENSION must be 'png', 'pdf', or 'svg'.")
    for name in ("LOOKUP_X_LIMITS", "LOOKUP_Y_LIMITS", "FACTOR_X_LIMITS", "FACTOR_Y_LIMITS",
                 "Q_DOY_MEAN_COLOR_LIMITS", "Q_DOY_FACTOR_COLOR_LIMITS"):
        check_limits(name, globals()[name])
    low, high = Q_DOY_FACTOR_COLOR_LIMITS
    if (low is not None and low >= 1) or (high is not None and high <= 1):
        raise ValueError("Q_DOY_FACTOR_COLOR_LIMITS must bracket 1 (no correction).")
    if FACTOR_LOG_SCALE and FACTOR_Y_LIMITS[0] is not None and FACTOR_Y_LIMITS[0] <= 0:
        raise ValueError("FACTOR_Y_LIMITS lower limit must be positive with FACTOR_LOG_SCALE.")


def require_file(filename, description):
    """Raise a clear error if an input file is missing."""
    if not filename.is_file():
        raise FileNotFoundError(f"{description} not found: {filename}")

# =============================================================================
# Reading
# =============================================================================


def quantile_percent(coordinate):
    """Convert a quantile coordinate (bin index, fraction, or %) to percent."""
    coordinate = np.asarray(coordinate)
    if np.issubdtype(coordinate.dtype, np.integer):
        return (coordinate.astype(float) + 0.5) * 100.0 / coordinate.size
    coordinate = coordinate.astype(float)
    return coordinate * 100.0 if np.nanmax(coordinate) <= 1.0 else coordinate


def coordinate_values(data, dim):
    """Return plotting values of one lookup dimension."""
    if dim == "month_of_year":
        return MONTHS.astype(float)
    values = np.asarray(data[dim].values) if dim in data.coords else np.arange(data.sizes[dim])
    return quantile_percent(values) if dim.startswith("quantile") else values.astype(float)


def read_lookups(ds):
    """Read q/doy/ld/q_doy lookup tables as plain NumPy arrays for plotting."""
    missing = [name for name in LOOKUP_VARIABLES if name not in ds]
    if missing:
        raise KeyError(f"Input dataset is missing lookup variables: {missing}")
    expected = EXPECTED_DIMS[BIAS_CORRECTION_METHOD]
    for name in LOOKUP_VARIABLES:
        dims = tuple(ds[name].dims)
        scalar_reference = BIAS_CORRECTION_METHOD == "ld" and name == "reference_mean_lookup"
        allowed = {(), expected} if scalar_reference else {expected}
        if set(dims) != set(expected) and dims not in allowed:
            raise ValueError(
                f"{name} has dimensions {dims}; expected {expected} for "
                f"'{BIAS_CORRECTION_METHOD}'. Check BIAS_CORRECTION_METHOD and the input file."
            )

    def as_array(name):
        data = ds[name]
        if data.ndim > 1:
            data = data.transpose(*expected)
        return np.asarray(data.values, dtype=float)

    model = ds["model_mean_lookup"]
    if model.ndim > 1:
        model = model.transpose(*expected)
    return {
        "dims": expected,
        "coords": [coordinate_values(model, dim) for dim in expected],
        "model": as_array("model_mean_lookup"),
        "reference": as_array("reference_mean_lookup"),
        "factor": as_array("bias_correction_factor_lookup"),
        "lead_factors": {},
    }


def read_reference_monthly_mean(reference_ds):
    """Recalculate the MM reference monthly mean exactly as the sample script does."""
    variable = REFERENCE_VARIABLES[REFERENCE_DATASET]
    if variable not in reference_ds:
        raise KeyError(f"Reference variable '{variable}' was not found.")
    values = reference_ds[variable]
    if not {"year", "month"}.issubset(values.dims):
        raise ValueError(f"Reference variable '{variable}' must have dimensions year and month.")
    if EXCLUDE_STORM_HANS_FROM_REFERENCE:
        storm_hans = (values["year"] == STORM_HANS_YEAR) & (values["month"] == STORM_HANS_MONTH)
        if not bool(storm_hans.any().values):
            raise ValueError("Cannot exclude Storm Hans: August 2023 is absent from the reference.")
        values = values.where(~storm_hans)
    return np.asarray(values.mean(dim="year", skipna=True).sel(month=MONTHS).values, dtype=float)


def month_ratio(ds, name):
    """Return one ratio(month_of_year) variable as a 12-value array."""
    data = ds[name]
    if data.dims != ("month_of_year",) or data.size != 12:
        raise ValueError(f"{name} must have dimension ('month_of_year',) with 12 values.")
    return np.asarray(data.values, dtype=float)


def lead_ratio_label(name):
    """Turn lead_time_bias_correction_ratio_lead17_31 into 'leads 17–31'."""
    lead_range = name.removeprefix(MM_LEAD_RATIO_PREFIX).removeprefix("lead")
    return "leads " + lead_range.replace("_", "–")


def read_mm_lookups(ds, reference_ds):
    """Read MM ratios and derive the monthly means they were calculated from."""
    if MM_RATIO_VARIABLE not in ds:
        raise KeyError(f"Input dataset is missing '{MM_RATIO_VARIABLE}'. Is this an MM file?")
    stored_method = ds.attrs.get("bias_correction_method")
    if stored_method is not None and stored_method != BIAS_CORRECTION_METHOD:
        raise ValueError(f"Input file was written with '{stored_method}', "
                         f"not '{BIAS_CORRECTION_METHOD}'.")
    stored_hans = ds.attrs.get("storm_hans_august_2023_excluded_from_reference")
    if stored_hans is not None and bool(int(stored_hans)) != EXCLUDE_STORM_HANS_FROM_REFERENCE:
        raise ValueError(
            "EXCLUDE_STORM_HANS_FROM_REFERENCE does not match the setting stored in the MM file "
            f"({bool(int(stored_hans))}); the derived model mean would be wrong."
        )

    factor = month_ratio(ds, MM_RATIO_VARIABLE)
    lead_names = sorted(name for name in ds.data_vars if name.startswith(MM_LEAD_RATIO_PREFIX))
    if BIAS_CORRECTION_METHOD == "mm_2step" and not lead_names:
        raise ValueError("mm_2step file contains no lead-time correction ratios.")
    reference = read_reference_monthly_mean(reference_ds)
    with np.errstate(divide="ignore", invalid="ignore"):
        model = reference / factor
    return {
        "dims": EXPECTED_DIMS[BIAS_CORRECTION_METHOD],
        "coords": [MONTHS.astype(float)],
        "model": model,
        "reference": reference,
        "factor": factor,
        "lead_factors": {lead_ratio_label(name): month_ratio(ds, name) for name in lead_names},
    }

# =============================================================================
# Plot helpers
# =============================================================================


def cell_edges(centres):
    """Return cell edges for monotonic cell centres (half a step beyond each end)."""
    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        return centres + np.array([-0.5, 0.5])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate(
        ([centres[0] - (middle[0] - centres[0])], middle, [centres[-1] + (centres[-1] - middle[-1])])
    )


def resolve_limits(limits, *arrays):
    """Fill None entries of a (lower, upper) pair from the finite data range."""
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot set colour limits: no finite lookup values.")
    lower = float(np.min(values)) if limits[0] is None else limits[0]
    upper = float(np.max(values)) if limits[1] is None else limits[1]
    return lower, upper


def factor_norm(factor):
    """Return a colour normalisation centred on a factor of 1."""
    low, high = Q_DOY_FACTOR_COLOR_LIMITS
    if low is None or high is None:
        finite = factor[np.isfinite(factor)]
        spread = max(float(np.max(np.abs(finite - 1.0))), 1e-6) if finite.size else 0.5
        low = 1.0 - spread if low is None else low
        high = 1.0 + spread if high is None else high
    return TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=high)


def apply_limits(axis, x_limits, y_limits):
    """Apply optional user axis limits; None leaves that limit unchanged."""
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)


def style_line_axis(axis, dim):
    """Apply common formatting to a 1-D lookup panel."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=4, width=0.8)
    axis.grid(axis="y", linewidth=0.5, alpha=0.25)
    axis.set_xlabel(X_LABELS[dim])
    if dim == "doy":
        set_month_ticks(axis)
    elif dim == "quantile_global":
        axis.set_xlim(0, 100)
    elif dim == "lead_day":
        axis.xaxis.get_major_locator().set_params(integer=True)
    elif dim == "month_of_year":
        axis.set_xticks(MONTHS, [month_abbr[month] for month in MONTHS])
        axis.set_xlim(0.5, 12.5)


def set_month_ticks(axis):
    """Label day-of-year axes at the first day of each month."""
    axis.set_xticks(MONTH_START_DOY, [month_abbr[month] for month in range(1, 13)])
    axis.set_xlabel("Day of year (month start)")


def set_title(axis, label, title):
    """Add a left-aligned panel label and title, as in the other figures."""
    axis.set_title(f"({label}) {title}", loc="left", fontsize=TITLE_FONTSIZE, pad=6)


def line_marker(dim):
    """Use markers for discrete lookup dimensions."""
    return "o" if dim in {"quantile_global", "month_of_year"} else None

# =============================================================================
# Panels
# =============================================================================


def plot_means_1d(axis, lookups, label):
    """Plot model and reference means against the lookup dimension."""
    dim, x = lookups["dims"][0], lookups["coords"][0]
    marker = line_marker(dim)
    axis.plot(x, lookups["model"], color=MODEL_COLOR, linewidth=LINEWIDTH, marker=marker,
              markersize=MARKER_SIZE, markerfacecolor="none", label="Model")
    reference = lookups["reference"]
    if reference.ndim == 0:
        axis.axhline(float(reference), color=REFERENCE_COLOR, linewidth=LINEWIDTH,
                     label=f"{reference_display_name()} (overall mean)")
    else:
        axis.plot(x, reference, color=REFERENCE_COLOR, linewidth=LINEWIDTH, marker=marker,
                  markersize=MARKER_SIZE, markerfacecolor="none", label=reference_display_name())
    style_line_axis(axis, dim)
    if is_mm_method():
        axis.set_ylabel(f"Mean monthly max. {X_DAYS}-day\nprecipitation (mm)")
        set_title(axis, label, "Monthly means")
    else:
        axis.set_ylabel(f"Mean {X_DAYS}-day precipitation (mm)")
        set_title(axis, label, "Lookup means")
    axis.legend(frameon=False, loc="best")
    apply_limits(axis, LOOKUP_X_LIMITS, LOOKUP_Y_LIMITS)


def plot_factor_1d(axis, lookups, label):
    """Plot the multiplicative factor(s) against the lookup dimension."""
    dim, x = lookups["dims"][0], lookups["coords"][0]
    marker = line_marker(dim)
    lead_factors = lookups["lead_factors"]
    main_label = f"{reference_display_name()} ratio" if lead_factors else None
    axis.plot(x, lookups["factor"], color=FACTOR_COLOR, linewidth=LINEWIDTH, marker=marker,
              markersize=MARKER_SIZE, markerfacecolor="none", label=main_label)
    for index, (lead_label, values) in enumerate(lead_factors.items()):
        axis.plot(x, values, color=LEAD_RATIO_COLORS[index % len(LEAD_RATIO_COLORS)],
                  linewidth=LINEWIDTH, linestyle="--", marker=marker, markersize=MARKER_SIZE,
                  markerfacecolor="none", label=f"Lead-time ratio, {lead_label}")
    axis.axhline(1.0, color=FACTOR_COLOR, linewidth=1.0, linestyle=":", alpha=0.6)
    if FACTOR_LOG_SCALE:
        axis.set_yscale("log")
    style_line_axis(axis, dim)
    axis.set_ylabel("Multiplicative factor")
    set_title(axis, label, "Bias-correction factor")
    if lead_factors:
        axis.legend(frameon=False, loc="best")
    apply_limits(axis, FACTOR_X_LIMITS, FACTOR_Y_LIMITS)


def plot_heatmap(axis, lookups, values, label, title, cmap, norm=None, limits=(None, None)):
    """Plot one doy x quantile_doy lookup on its true cell edges."""
    doy, quantile = lookups["coords"]
    image = axis.pcolormesh(
        cell_edges(doy), cell_edges(quantile), values.T,
        cmap=cmap, norm=norm, vmin=None if norm else limits[0],
        vmax=None if norm else limits[1], shading="flat", rasterized=True,
    )
    set_month_ticks(axis)
    axis.set_ylabel("Seasonal quantile (%)")
    axis.set_ylim(0, 100)
    axis.tick_params(direction="out", length=4, width=0.8)
    set_title(axis, label, title)
    return image


def make_figure_1d(lookups):
    """Create the two-panel figure for the 1-D methods."""
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_1D, constrained_layout=True)
    plot_means_1d(axes[0], lookups, "a")
    plot_factor_1d(axes[1], lookups, "b")
    return figure


def make_figure_q_doy(lookups):
    """Create the three-panel heatmap figure for q_doy."""
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_Q_DOY, constrained_layout=True,
                                sharey=True)
    if Q_DOY_SHARED_MEAN_COLOR_LIMITS:
        shared = resolve_limits(Q_DOY_MEAN_COLOR_LIMITS, lookups["model"], lookups["reference"])
        model_limits = reference_limits = shared
    else:
        model_limits = resolve_limits(Q_DOY_MEAN_COLOR_LIMITS, lookups["model"])
        reference_limits = resolve_limits(Q_DOY_MEAN_COLOR_LIMITS, lookups["reference"])

    model_image = plot_heatmap(axes[0], lookups, lookups["model"], "a", "Model mean",
                               MEAN_COLORMAP, limits=model_limits)
    reference_image = plot_heatmap(axes[1], lookups, lookups["reference"], "b",
                                   f"{reference_display_name()} mean", MEAN_COLORMAP,
                                   limits=reference_limits)
    factor_image = plot_heatmap(axes[2], lookups, lookups["factor"], "c",
                                "Bias-correction factor", FACTOR_COLORMAP,
                                norm=factor_norm(lookups["factor"]))
    for axis in axes[1:]:
        axis.set_ylabel("")

    mean_label = f"Mean {X_DAYS}-day precipitation (mm)"
    if Q_DOY_SHARED_MEAN_COLOR_LIMITS:
        figure.colorbar(model_image, ax=axes[:2], location="bottom", fraction=0.06,
                        pad=0.02, aspect=40, label=mean_label)
    else:
        for axis, image in [(axes[0], model_image), (axes[1], reference_image)]:
            figure.colorbar(image, ax=axis, location="bottom", fraction=0.06, pad=0.02,
                            label=mean_label)
    figure.colorbar(factor_image, ax=axes[2], location="bottom", fraction=0.06, pad=0.02,
                    label="Multiplicative factor", extend="both")
    return figure


def make_lookup_figure(lookups, filename=None):
    """Create, optionally save, and optionally show the lookup-table figure."""
    style = {"font.family": "sans-serif", "font.size": 10, "axes.linewidth": 0.8,
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        figure = (make_figure_q_doy(lookups) if BIAS_CORRECTION_METHOD == "q_doy"
                  else make_figure_1d(lookups))
        if SHOW_SUPTITLE:
            figure.suptitle(
                f"{METHOD_NAMES[BIAS_CORRECTION_METHOD]} bias correction "
                f"({reference_display_name()}, {X_DAYS}-day accumulation)", fontsize=12,
            )
        if filename is not None:
            filename.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
            print("Wrote:", filename)
        if SHOW_FIGURE:
            plt.show()
        plt.close(figure)

# =============================================================================
# Reporting
# =============================================================================


def print_lookups(lookups):
    """Print lookup dimensions, 1-D tables, and a factor summary."""
    print("\nLookup dimensions:", lookups["dims"],
          "| sizes:", tuple(len(coord) for coord in lookups["coords"]))
    if lookups["model"].ndim == 1:
        dim, x = lookups["dims"][0], lookups["coords"][0]
        reference = lookups["reference"]
        lead_factors = lookups["lead_factors"]
        header = (f"\n{X_LABELS[dim]:>16} {'Model':>12} {reference_display_name():>12} "
                  f"{'Factor':>10}")
        header += "".join(f" {label:>14}" for label in lead_factors)
        print(header)
        for index, value in enumerate(x):
            ref = float(reference) if reference.ndim == 0 else reference[index]
            row = (f"{value:16.4g} {lookups['model'][index]:12.4f} {ref:12.4f} "
                   f"{lookups['factor'][index]:10.4f}")
            row += "".join(f" {values[index]:14.4f}" for values in lead_factors.values())
            print(row)
    factor = lookups["factor"]
    finite = factor[np.isfinite(factor)]
    print("\nFactor summary")
    if finite.size:
        print(f"  min {finite.min():.4f}  median {np.median(finite):.4f}  max {finite.max():.4f}")
    print(f"  non-finite cells: {factor.size - finite.size} of {factor.size}")

# =============================================================================
# Main
# =============================================================================


def main():
    """Read the correction tables, print a summary, and plot them."""
    validate_settings()
    input_filename = make_input_filename()
    require_file(input_filename, "Input file")
    print("Input file:", input_filename)
    print(f"Method: {BIAS_CORRECTION_METHOD} | reference: {REFERENCE_DATASET} "
          f"{REFERENCE_YEARS[0]}-{REFERENCE_YEARS[1]} | accumulation: {X_DAYS} days")

    with xr.open_dataset(input_filename, decode_timedelta=False) as ds:
        if is_mm_method():
            reference_filename = make_reference_filename()
            require_file(reference_filename, "Reference file")
            print("Reference file:", reference_filename)
            print("Exclude Storm Hans from reference:", EXCLUDE_STORM_HANS_FROM_REFERENCE)
            with xr.open_dataset(reference_filename) as reference_ds:
                lookups = read_mm_lookups(ds, reference_ds)
        else:
            lookups = read_lookups(ds)
    print_lookups(lookups)
    make_lookup_figure(lookups, make_output_filename() if WRITE_TO_FILE else None)


if __name__ == "__main__":
    main()
