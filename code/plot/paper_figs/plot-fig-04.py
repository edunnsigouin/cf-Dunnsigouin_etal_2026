"""
Create a 2 x 2 figure comparing extreme-precipitation return metrics for one
reference dataset and one model dataset. The panels show the Storm Hans 2023
threshold and calendar-record threshold for two user-selected calendar months.
Month 0 retains the annual any-month calculation.

The selected REFERENCE_DATASET supplies the observational sample and thresholds
and is also used when selecting a bias-corrected model file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r

from Dunnsigouin_etal_2026 import config


# =============================================================================
# User settings
# =============================================================================

PLOT_METRIC = "return_period"  # "aep" or "return_period"
N_AEP_YEARS = 1  # Used only when PLOT_METRIC == "aep".

REFERENCE_DATASET = "senorge"  # "senorge" or "era5"

# Model data to plot:
#     "raw", "mm", "q", "ld", "doy", or "q_doy"
MODEL_DATA_METHOD = "raw"

# Calendar months for the two figure rows. Use 0 for annual / any month.
AUGUST_MONTH = 8
MAY_MONTH = 5

STORM_HANS_YEAR = 2023
STORM_HANS_MONTH = 8

RECORD_START_YEAR = 1957
AUGUST_RECORD_END_YEAR = 2022
MAY_RECORD_END_YEAR = 2022

OBSERVATION_YEARS = ["1957", "2023"]
EXCLUDE_2023_FROM_REFERENCE_FIT = True

CATCHMENT = "regine_drammen"
X_DAYS = 2
MODEL_VARIABLE = "tp24"

FORECAST_DATE_RANGE = ["2020-01-02", "2022-12-29"]
FIRST_INPUT_LEAD = 16
LAST_INPUT_LEAD = 46
NUMBER_OF_LEAD_BINS = 2
MODEL_SAMPLING_GROUP = "full"

REFERENCE_FILENAME_OVERRIDE = None
MODEL_FILENAME_OVERRIDE = None

FIG_WIDTH_IN = 10.5
FIG_HEIGHT_IN = 8.0
FIGURE_DPI = 400

POINT_SIZE = 68
MARKER_EDGE_WIDTH = 1.4
MARKER_FACECOLOR = "none"
MARKER_EDGECOLOR = "0.20"

METHOD_MARKERS = {"GEV": "o", "Gumbel": "s", "GenEx": "D"}
METHODS = list(METHOD_MARKERS)

AXIS_LABELSIZE = 12
TICK_LABELSIZE = 11
TITLE_FONTSIZE = 12
LEGEND_FONTSIZE = 10

AEP_YMIN_PERCENT = 0.0001
AEP_YMAX_PERCENT = 100.0
RETURN_PERIOD_YMIN_YEARS = 1.0
RETURN_PERIOD_YMAX_YEARS = 1_000_000.0

SHARE_Y_AXES = False
SHOW_GRID = True
WRITE_TO_FILE = True
SHOW_FIGURE = True


# =============================================================================
# Constants
# =============================================================================

SENORGE_VARIABLE = "rr"
ERA5_VARIABLE = "tp24"
ERA5_GRID = "0.5x0.5"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PLOT_GROUPS = ["reference", "model"]


# =============================================================================
# Validation and labels
# =============================================================================

def validate_user_settings():
    """Validate user-configurable settings."""

    if PLOT_METRIC not in {"aep", "return_period"}:
        raise ValueError("PLOT_METRIC must be 'aep' or 'return_period'.")

    if not isinstance(N_AEP_YEARS, int) or N_AEP_YEARS < 1:
        raise ValueError("N_AEP_YEARS must be a positive integer.")

    if REFERENCE_DATASET not in {"senorge", "era5"}:
        raise ValueError("REFERENCE_DATASET must be 'senorge' or 'era5'.")

    valid_model_methods = {"raw", "mm", "q", "ld", "doy", "q_doy"}
    if MODEL_DATA_METHOD not in valid_model_methods:
        raise ValueError(f"MODEL_DATA_METHOD must be one of {sorted(valid_model_methods)}.")

    for name, month in {"AUGUST_MONTH": AUGUST_MONTH, "MAY_MONTH": MAY_MONTH}.items():
        if not isinstance(month, int) or month not in range(13):
            raise ValueError(f"{name} must be 0 (annual) or an integer from 1 to 12.")

    if AUGUST_RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError("AUGUST_RECORD_END_YEAR must be >= RECORD_START_YEAR.")

    if MAY_RECORD_END_YEAR < RECORD_START_YEAR:
        raise ValueError("MAY_RECORD_END_YEAR must be >= RECORD_START_YEAR.")

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    if first_usable_lead > LAST_INPUT_LEAD:
        raise ValueError("X_DAYS is too large for the configured lead range.")

    usable_leads = LAST_INPUT_LEAD - first_usable_lead + 1
    if not isinstance(NUMBER_OF_LEAD_BINS, int) or not 1 <= NUMBER_OF_LEAD_BINS <= usable_leads:
        raise ValueError("NUMBER_OF_LEAD_BINS is invalid for the usable lead range.")

    valid_groups = {"full", *(f"split{i}" for i in range(1, NUMBER_OF_LEAD_BINS + 1))}
    if MODEL_SAMPLING_GROUP not in valid_groups:
        raise ValueError(f"MODEL_SAMPLING_GROUP must be one of {sorted(valid_groups)}.")


def get_reference_name():
    """Return the publication-style reference dataset name."""

    return {"senorge": "SeNorge", "era5": "ERA5"}[REFERENCE_DATASET]


def get_reference_variable():
    """Return the selected reference variable."""

    return {"senorge": SENORGE_VARIABLE, "era5": ERA5_VARIABLE}[REFERENCE_DATASET]


def get_model_label():
    """Return the x-axis label for the selected model data."""

    if MODEL_DATA_METHOD == "raw":
        return "Model raw"
    return f"Model BC\n({MODEL_DATA_METHOD})"


def get_panel_month_label(month):
    """Return the calendar-month name or 'Annual' for month 0."""

    return "Annual" if month == 0 else MONTH_NAMES[month - 1]


# =============================================================================
# Filenames and lead ranges
# =============================================================================

def get_model_file_id(catchment_name):
    """Return the short catchment identifier used in model filenames."""

    return catchment_name.removeprefix("regine_")


def split_usable_accumulated_leads(first_lead, last_lead, number_of_bins):
    """Split usable ending leads into near-equal bins."""

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
    """Return configured model lead bins."""

    first_usable_lead = FIRST_INPUT_LEAD + X_DAYS - 1
    return split_usable_accumulated_leads(
        first_usable_lead, LAST_INPUT_LEAD, NUMBER_OF_LEAD_BINS
    )


def get_full_lead_range():
    """Return the complete usable ending-lead range."""

    return FIRST_INPUT_LEAD + X_DAYS - 1, LAST_INPUT_LEAD


def get_selected_model_lead_range():
    """Return the lead range selected by MODEL_SAMPLING_GROUP."""

    if MODEL_SAMPLING_GROUP == "full":
        return get_full_lead_range()

    split_number = int(MODEL_SAMPLING_GROUP.removeprefix("split"))
    return build_lead_bins()[split_number - 1]


def get_model_variable():
    """Return the compact precipitation variable to read."""

    if MODEL_SAMPLING_GROUP == "full":
        return "tp24_max"

    lead_start, lead_end = get_selected_model_lead_range()
    return f"tp24_max_lead{lead_start}_{lead_end}"


def lead_split_filename_label():
    """Return the lead-bin label used in compact model filenames."""

    full_start, full_end = get_full_lead_range()
    split_text = "_".join(f"{start}-{end}" for start, end in build_lead_bins())
    return f"lead{full_start}-{full_end}_split{NUMBER_OF_LEAD_BINS}_{split_text}"


def make_reference_filename():
    """Construct the selected reference-data filename."""

    if REFERENCE_FILENAME_OVERRIDE is not None:
        return Path(REFERENCE_FILENAME_OVERRIDE)

    if REFERENCE_DATASET == "senorge":
        filename = (
            f"distribution_monthly_extremes_{SENORGE_VARIABLE}_{X_DAYS}dayacc_"
            f"{CATCHMENT}_senorge_{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
        )
        return Path(config.dirs["senorge_processed"]) / filename

    filename = (
        f"distribution_monthly_extremes_{ERA5_VARIABLE}_{X_DAYS}dayacc_{CATCHMENT}_"
        f"era5_{ERA5_GRID}_{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}.nc"
    )
    return Path(config.dirs["era5_processed"]) / filename


def make_raw_model_filename():
    """Construct the raw compact model filename."""

    filename = (
        f"monthly_max_samples_{MODEL_VARIABLE}_{X_DAYS}dayacc_"
        f"{get_model_file_id(CATCHMENT)}_{lead_split_filename_label()}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}.nc"
    )
    return Path(config.dirs["s2s_processed"]) / filename


def make_model_filename():
    """Construct the selected raw or bias-corrected model filename."""

    if MODEL_FILENAME_OVERRIDE is not None:
        return Path(MODEL_FILENAME_OVERRIDE)

    raw_filename = make_raw_model_filename()
    if MODEL_DATA_METHOD == "raw":
        return raw_filename

    return raw_filename.with_name(
        f"{raw_filename.stem}_bc_{MODEL_DATA_METHOD}_{REFERENCE_DATASET}.nc"
    )


def make_figure_filename():
    """Construct the output figure filename."""

    metric_label = (
        f"{N_AEP_YEARS}year-aep" if PLOT_METRIC == "aep" else "return-period"
    )
    model_label = (
        "raw" if MODEL_DATA_METHOD == "raw"
        else f"bc-{MODEL_DATA_METHOD}"
    )
    filename = f"fig-04_{metric_label}_{model_label}-{REFERENCE_DATASET}.png"
    return Path(config.dirs["fig"]) / filename


# =============================================================================
# Data reading
# =============================================================================

def read_reference_month(month, record_end_year):
    """Read one reference month and its Storm Hans and record thresholds."""

    filename = make_reference_filename()
    variable = get_reference_variable()

    if not filename.is_file():
        raise FileNotFoundError(f"Reference file not found: {filename}")

    with xr.open_dataset(filename) as ds:
        if variable not in ds:
            raise KeyError(f"Variable '{variable}' was not found in {filename}.")

        selected = ds[variable].sel(
            year=slice(int(OBSERVATION_YEARS[0]), int(OBSERVATION_YEARS[1])),
            month=month,
        ).load()

        record_data = ds[variable].sel(
            year=slice(RECORD_START_YEAR, record_end_year), month=month
        ).load()

        storm_hans_value = float(
            ds[variable].sel(year=STORM_HANS_YEAR, month=STORM_HANS_MONTH).load().values
        )

    years = np.asarray(selected["year"].values)
    values = np.asarray(selected.values, dtype=float)
    finite = np.isfinite(values)
    years, values = years[finite], values[finite]

    if EXCLUDE_2023_FROM_REFERENCE_FIT:
        values = values[years != STORM_HANS_YEAR]

    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite {get_reference_name()} {MONTH_NAMES[month - 1]} "
            "values remain for fitting."
        )

    record_values = np.asarray(record_data.values, dtype=float)
    record_years = np.asarray(record_data["year"].values)
    finite = np.isfinite(record_values)

    if not finite.any():
        raise ValueError(f"No finite {MONTH_NAMES[month - 1]} record values were found.")

    record_values, record_years = record_values[finite], record_years[finite]
    record_index = int(np.argmax(record_values))

    return {
        "fit_values": values,
        "storm_hans_value": storm_hans_value,
        "record_value": float(record_values[record_index]),
        "record_year": int(record_years[record_index]),
    }


def read_model_month(month):
    """Read one calendar-month sample from the selected compact model file."""

    filename = make_model_filename()
    if not filename.is_file():
        raise FileNotFoundError(f"Model file not found: {filename}")

    variable = get_model_variable()
    with xr.open_dataset(filename, decode_timedelta=False) as ds:
        if variable not in ds:
            raise KeyError(
                f"Variable '{variable}' was not found in {filename}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        if "month" not in ds:
            raise KeyError(f"Variable 'month' was not found in {filename}.")

        selected = ds[variable].where(ds["month"] == month, drop=True)
        values = np.asarray(selected.values, dtype=float).ravel()

    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError(
            f"Fewer than 10 finite model values were found for {MONTH_NAMES[month - 1]}."
        )

    return values


# =============================================================================
# Extreme-value methods
# =============================================================================

def fit_gev(values):
    """Fit a stationary three-parameter GEV distribution."""

    shape_c, location, scale = genextreme.fit(values)
    if not np.isfinite([shape_c, location, scale]).all() or scale <= 0:
        raise RuntimeError("The GEV fit returned invalid parameters.")
    return shape_c, location, scale


def fit_gumbel(values):
    """Fit a stationary two-parameter Gumbel distribution."""

    location, scale = gumbel_r.fit(values)
    if not np.isfinite([location, scale]).all() or scale <= 0:
        raise RuntimeError("The Gumbel fit returned invalid parameters.")
    return location, scale


def genex_negative_log_likelihood(log_parameters, values):
    """Return the GenEx negative log-likelihood."""

    shape, scale = np.exp(log_parameters)
    if (
        not np.isfinite(shape)
        or not np.isfinite(scale)
        or shape <= 0
        or scale <= 0
        or np.any(values < 0)
    ):
        return np.inf

    z = values / scale
    log_pdf = np.log(shape) - np.log(scale) - z + (shape - 1.0) * np.log(-np.expm1(-z))
    return np.inf if not np.isfinite(log_pdf).all() else -np.sum(log_pdf)


def fit_genex(values):
    """Fit a two-parameter Generalized Exponential distribution."""

    if np.any(values < 0):
        raise ValueError("GenEx requires non-negative values.")

    positive_values = values[values > 0]
    if positive_values.size == 0:
        raise RuntimeError("GenEx cannot be fitted without positive values.")

    result = minimize(
        genex_negative_log_likelihood,
        x0=np.log([1.0, np.mean(positive_values)]),
        args=(values,),
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    if not result.success:
        raise RuntimeError(f"GenEx fit failed: {result.message}")

    shape, scale = np.exp(result.x)
    if not np.isfinite([shape, scale]).all() or shape <= 0 or scale <= 0:
        raise RuntimeError("The GenEx fit returned invalid parameters.")

    return shape, scale


def calculate_exceedance_probability(values, event_value, method):
    """Fit one distribution and calculate exceedance probability."""

    if method == "GEV":
        shape_c, location, scale = fit_gev(values)
        probability = genextreme.sf(
            event_value, shape_c, loc=location, scale=scale
        )
    elif method == "Gumbel":
        location, scale = fit_gumbel(values)
        probability = gumbel_r.sf(event_value, loc=location, scale=scale)
    elif method == "GenEx":
        shape, scale = fit_genex(values)
        probability = 1.0 if event_value < 0 else 1.0 - (1.0 - np.exp(-event_value / scale)) ** shape
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not np.isfinite(probability):
        raise RuntimeError(f"{method} produced a non-finite probability.")

    return float(np.clip(probability, 0.0, 1.0))


def analyse_method(values, event_value, method):
    """Calculate exceedance probability and return period."""

    probability = calculate_exceedance_probability(values, event_value, method)
    return_period = np.inf if probability <= 0 else 1.0 / probability
    return {"annual_exceedance_probability": probability, "return_period": return_period}


def combine_monthly_probabilities(monthly_probabilities):
    """Combine 12 monthly probabilities into an annual any-month probability."""

    probabilities = np.asarray(monthly_probabilities, dtype=float)
    if probabilities.size != 12 or not np.isfinite(probabilities).all():
        raise ValueError("Annual analysis requires 12 finite monthly probabilities.")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    return float(np.clip(1.0 - np.prod(1.0 - probabilities), 0.0, 1.0))


def calculate_horizon_aep(annual_probability, horizon_years):
    """Convert annual exceedance probability to an N-year probability."""

    annual_probability = float(np.clip(annual_probability, 0.0, 1.0))
    return float(-np.expm1(horizon_years * np.log1p(-annual_probability)))


def get_metric_value(analysis):
    """Return the selected plotting metric within the configured limits."""

    if PLOT_METRIC == "aep":
        value = 100.0 * calculate_horizon_aep(
            analysis["annual_exceedance_probability"], N_AEP_YEARS
        )
        return float(np.clip(value, AEP_YMIN_PERCENT, AEP_YMAX_PERCENT))

    value = analysis["return_period"]
    if not np.isfinite(value):
        value = RETURN_PERIOD_YMAX_YEARS
    return float(
        np.clip(value, RETURN_PERIOD_YMIN_YEARS, RETURN_PERIOD_YMAX_YEARS)
    )


# =============================================================================
# Panel calculations
# =============================================================================

def calculate_month_panel(panel_label, month, threshold_type, record_end_year):
    """Calculate one single-month panel for the reference and selected model."""

    reference = read_reference_month(month, record_end_year)
    model_values = read_model_month(month)

    if threshold_type == "storm_hans":
        event_value = reference["storm_hans_value"]
        threshold_label = "Storm Hans 2023"
    else:
        event_value = reference["record_value"]
        threshold_label = f"calendar record {RECORD_START_YEAR}-{record_end_year}"

    samples = {"reference": reference["fit_values"], "model": model_values}
    results = {
        (group, method): analyse_method(values, event_value, method)
        for group, values in samples.items()
        for method in METHODS
    }

    return {
        "title": f"{panel_label}) {get_panel_month_label(month)} {threshold_label}",
        "month": month,
        "threshold_type": threshold_type,
        "event_value": event_value,
        "record_year": reference["record_year"],
        "results": results,
    }


def calculate_annual_panel(panel_label, threshold_type, record_end_year):
    """Calculate an annual any-month panel from 12 monthly probabilities."""

    monthly_probabilities = {
        group: {method: [] for method in METHODS} for group in PLOT_GROUPS
    }
    monthly_event_values = {}

    for month in range(1, 13):
        reference = read_reference_month(month, record_end_year)
        model_values = read_model_month(month)

        event_value = (
            reference["storm_hans_value"]
            if threshold_type == "storm_hans"
            else reference["record_value"]
        )
        monthly_event_values[month] = event_value

        samples = {"reference": reference["fit_values"], "model": model_values}
        for group, values in samples.items():
            for method in METHODS:
                analysis = analyse_method(values, event_value, method)
                monthly_probabilities[group][method].append(
                    analysis["annual_exceedance_probability"]
                )

    results = {}
    for group in PLOT_GROUPS:
        for method in METHODS:
            probability = combine_monthly_probabilities(
                monthly_probabilities[group][method]
            )
            results[(group, method)] = {
                "annual_exceedance_probability": probability,
                "return_period": np.inf if probability <= 0 else 1.0 / probability,
            }

    threshold_label = (
        "Storm Hans 2023"
        if threshold_type == "storm_hans"
        else f"calendar record {RECORD_START_YEAR}-{record_end_year}"
    )

    return {
        "title": f"{panel_label}) Annual {threshold_label}",
        "month": 0,
        "threshold_type": threshold_type,
        "event_value": None,
        "monthly_event_values": monthly_event_values,
        "record_year": None,
        "results": results,
    }


def calculate_panel(panel_label, month, threshold_type, record_end_year):
    """Calculate a single-month or annual panel."""

    if month == 0:
        return calculate_annual_panel(panel_label, threshold_type, record_end_year)
    return calculate_month_panel(panel_label, month, threshold_type, record_end_year)


def calculate_all_panels():
    """Calculate the four panels in reading order."""

    panel_specs = [
        ("a", AUGUST_MONTH, "storm_hans", AUGUST_RECORD_END_YEAR),
        ("b", AUGUST_MONTH, "calendar_record", AUGUST_RECORD_END_YEAR),
        ("c", MAY_MONTH, "storm_hans", MAY_RECORD_END_YEAR),
        ("d", MAY_MONTH, "calendar_record", MAY_RECORD_END_YEAR),
    ]
    return [calculate_panel(*spec) for spec in panel_specs]


# =============================================================================
# Reporting and plotting
# =============================================================================

def print_panel_results(panel_outputs):
    """Print calculated probabilities and return periods."""

    print(f"\nReference dataset: {get_reference_name()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")

    for panel in panel_outputs:
        print(f"\n{panel['title']}")
        if panel["month"] != 0:
            print(f"Event value: {panel['event_value']:.4f} mm")

        for group in PLOT_GROUPS:
            group_label = get_reference_name() if group == "reference" else get_model_label()
            for method in METHODS:
                analysis = panel["results"][(group, method)]
                probability = analysis["annual_exceedance_probability"]
                return_period = analysis["return_period"]
                rp_text = f"{return_period:.4g}" if np.isfinite(return_period) else "inf"
                print(
                    f"  {group_label:>14} | {method:>7} | "
                    f"annual p={100.0 * probability:.6g}% | RP={rp_text} y"
                )


def metric_tick_formatter(value, _position):
    """Format logarithmic metric-axis ticks."""

    if value <= 0:
        return ""

    label = f"{value:g}"
    if PLOT_METRIC == "aep" and np.isclose(value, AEP_YMIN_PERCENT):
        return f"<{label}"
    if PLOT_METRIC == "return_period" and np.isclose(
        value, RETURN_PERIOD_YMAX_YEARS
    ):
        return f">{label}"
    return label


def get_y_axis_label():
    """Return the y-axis label."""

    if PLOT_METRIC == "aep":
        return f"{N_AEP_YEARS}-year exceedance probability [%]"
    return "Return period [years]"


def configure_axis(axis):
    """Apply common axis formatting."""

    axis.set_yscale("log")
    if PLOT_METRIC == "aep":
        axis.set_ylim(0.7 * AEP_YMIN_PERCENT, 1.3 * AEP_YMAX_PERCENT)
    else:
        axis.set_ylim(
            0.7 * RETURN_PERIOD_YMIN_YEARS, 1.3 * RETURN_PERIOD_YMAX_YEARS
        )

    axis.yaxis.set_major_formatter(FuncFormatter(metric_tick_formatter))
    axis.set_xlim(-0.55, 1.45)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        [get_reference_name(), get_model_label()], fontsize=TICK_LABELSIZE
    )
    axis.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    if SHOW_GRID:
        axis.grid(
            axis="y", which="major", linestyle=":", linewidth=0.7, alpha=0.45, zorder=0
        )


def plot_one_panel(axis, panel_output):
    """Plot one panel."""

    for group_index, group in enumerate(PLOT_GROUPS):
        for method in METHODS:
            y_value = get_metric_value(panel_output["results"][(group, method)])
            axis.scatter(
                group_index,
                y_value,
                s=POINT_SIZE,
                marker=METHOD_MARKERS[method],
                facecolor=MARKER_FACECOLOR,
                edgecolor=MARKER_EDGECOLOR,
                linewidth=MARKER_EDGE_WIDTH,
                zorder=4,
            )

    configure_axis(axis)
    axis.set_title(
        panel_output["title"],
        fontsize=TITLE_FONTSIZE,
        fontweight="normal",
        loc="left",
        pad=8,
    )


def make_figure(panel_outputs):
    """Create the 2 x 2 publication figure."""

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
        2,
        2,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        sharey=SHARE_Y_AXES,
        constrained_layout=True,
    )

    for axis, panel_output in zip(axes.ravel(), panel_outputs):
        plot_one_panel(axis, panel_output)

    for row in range(2):
        axes[row, 0].set_ylabel(get_y_axis_label(), fontsize=AXIS_LABELSIZE)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=MARKER_EDGECOLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=np.sqrt(POINT_SIZE),
            label=method,
        )
        for method in METHODS
    ]
    axes[0, 0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="best",
        handletextpad=0.5,
        borderaxespad=0.4,
    )

    filename = make_figure_filename()
    if WRITE_TO_FILE:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
        print("Wrote:", filename)

    if SHOW_FIGURE:
        plt.show()

    plt.close(figure)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    validate_user_settings()

    print("Selected settings")
    print("-----------------")
    print(f"Reference dataset: {get_reference_name()}")
    print(f"Model data:        {MODEL_DATA_METHOD}")
    print(f"August row month:  {AUGUST_MONTH}")
    print(f"May row month:     {MAY_MONTH}")
    print(f"Model file:        {make_model_filename()}")
    print(f"Reference file:    {make_reference_filename()}")

    panels = calculate_all_panels()
    print_panel_results(panels)
    make_figure(panels)
