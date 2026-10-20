"""
Stacked time-series figure of one high-ranking S2S precipitation event, shown as
catchment means: one panel per entry in PLOT_VARIABLES (up to eight), each with
    - the selected ensemble member (line),
    - the central range over all hindcast dates and ensemble members (shading),
    - dots on the event days (date_of_max + EVENT_LAGS).

The event is the EVENT_RANK-th largest 2-day precipitation event in EVENT_MONTH,
taken from the monthly-maximum sample file.

Everything you normally change is in the USER SETTINGS block below, in this order:
    1. Event         - which event to plot
    2. Panels        - which variables, in which order, and the style of each panel
    3. Figure        - layout, fonts, lines, date axis
    4. Legend        - position and all legend text
    5. Output        - figure name, folder, formats
    6. Calculations  - climatology bands
    7. Data sources  - file locations (from config.dirs) and units (rarely changed)

Variables available for panels
    "tp24"   total precipitation (mm/day)       "rn24"  rainfall (mm/day)
    "sf24"   snowfall (mm/day)                  "sd"    snow water equivalent, SWE (mm)
    "melt"   snowmelt = sf24 - ΔSWE (mm/day)    "ro24"  total runoff (mm/day)
    "sro24"  surface runoff (mm/day)            "t2m24" daily-mean 2 m temperature (°C)
    "msl"    mean sea level pressure (hPa)
Daily totals (tp24, rn24, sf24, ro24, sro24) cover 00-24 UTC on the labelled day.
Daily snowfall "sf24" comes from the files written by convert_sf6_to_sf24.py.

Snow water equivalent change (ΔSWE) and snowmelt
-------------------------------------------------
The daily SWE files ("sd") hold daily means of the 6-hourly values at 00, 06, 12 and
18 UTC, so each daily value represents the snowpack at roughly 09 UTC. The daily
fluxes (tp24, sf24, ro24) are totals over 00-24 UTC and are labelled by the start of
that window. ΔSWE is therefore computed "forward" and labelled by its first day:

    ΔSWE(D) = SWE(D+1) - SWE(D)

This is the change from ~09 UTC on day D to ~09 UTC on day D+1. It overlaps the
00-24 UTC flux window of day D by 15 of 24 hours, and lags it by ~9 hours. The
alternative "backward" difference, SWE(D) - SWE(D-1), covers ~09 UTC on D-1 to ~09
UTC on D, overlaps day D by only 9 hours and leads it by ~15 hours, so it would place
snow changes too early relative to the precipitation. An exact 00-24 UTC window
would need 6-hourly SWE snapshots; the remaining ~9-hour offset is small compared
with the 2-day events studied here.

Snowmelt is estimated from the snow budget of each day,

    ΔSWE = snowfall - melt   =>   melt = sf24 - ΔSWE

so positive values are melt. Using snowfall recovers melt that ΔSWE alone hides on
days when snow both falls and melts (e.g. 10 mm of snowfall and 10 mm of melt gives
ΔSWE = 0 but melt = 10 mm). Small negative values mean the snowpack gained more water
than fell as snow, typically rain retained or refrozen in the snowpack. Sublimation
is neglected.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Dunnsigouin_etal_2026 import config

# #############################################################################
#                               USER SETTINGS
# #############################################################################

# =============================================================================
# 1. Event
# =============================================================================
CATCHMENT_NAME = "drammen"   # "drammen" or "glomma"
EVENT_MONTH = 5              # Calendar month of the event (1-12)
EVENT_RANK = 2               # 1 = largest 2-day precipitation event in that month
EVENT_LAGS = [-2, -1, 0, 1]  # Event days marked with dots, relative to the 2-day maximum

# =============================================================================
# 2. Panels
# =============================================================================
# Variables to plot, top to bottom (one panel each, up to eight).
PLOT_VARIABLES = ["tp24", "t2m24", "sd", "melt", "ro24"]

# Style of each variable's panel. Edit any entry to change that panel:
#   "title"       panel title (the a), b), ... label is added automatically)
#   "ylabel"      y-axis label
#   "color"       colour of the selected-member line and dots
#   "range_color" colour of the climatology bands and median; None = CLIMATOLOGY_COLOR
#   "ylim"        (bottom, top); None for either end sets it from the data
#   "zero_line"   True/False: grey reference line at 0
#   "yticks"      list of tick positions, or None for automatic ticks
MM_PER_DAY = r"[mm/day]"
VARIABLE_DEFAULTS = {
    "tp24":   {"title": "Precipitation", "ylabel": MM_PER_DAY, "color": "#1f5fa8",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "t2m24":  {"title": "Daily-mean 2 m temperature", "ylabel": r"[$^{\circ}$C]", "color": "#b2182b",
               "range_color": None, "ylim": (None, None), "zero_line": True, "yticks": None},
    "sd":     {"title": "Snow water equivalent", "ylabel": "[mm]", "color": "#6a51a3",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "melt":   {"title": r"Snowmelt (snowfall $-$ $\Delta$ snow water equivalent)",
               "ylabel": MM_PER_DAY, "color": "#7b3294",
               "range_color": None, "ylim": (None, None), "zero_line": True, "yticks": None},
    "ro24":   {"title": "Total runoff", "ylabel": MM_PER_DAY, "color": "#c2571a",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "rn24":   {"title": "Rainfall", "ylabel": MM_PER_DAY, "color": "#2a9d8f",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "sf24":   {"title": "Snowfall", "ylabel": MM_PER_DAY, "color": "#4a7fb5",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "sro24":  {"title": "Surface runoff", "ylabel": MM_PER_DAY, "color": "#d98a2b",
               "range_color": None, "ylim": (0, None), "zero_line": False, "yticks": None},
    "msl":    {"title": "Mean sea level pressure", "ylabel": "[hPa]", "color": "#555555",
               "range_color": None, "ylim": (None, None), "zero_line": False, "yticks": None},
}

# =============================================================================
# 3. Figure
# =============================================================================
# Text above the top panel; "{catchment}" is replaced by the catchment name.
# Use None for no title.
FIGURE_TITLE = None  # e.g. "{catchment} spatial mean"
FIGURE_WIDTH_IN = 7.0        # 7.0 in ≈ double-column journal width
PANEL_HEIGHT_IN = 1.75       # Height of each panel; figure height = panels x this
PANEL_SPACING = 0.32         # Vertical space between panels (fraction of panel height)

FONT_SIZE = 9                # Default for any text not set below
TICK_LABEL_SIZE = 11          # Numbers and dates on the axes
AXIS_LABEL_SIZE = 11          # Y-axis labels (e.g. "mm day-1")
TITLE_FONT_SIZE = 12        # Panel titles
FIGURE_TITLE_SIZE = 10       # Title above the top panel
LEGEND_FONT_SIZE = 9

SHOW_PANEL_LABELS = True     # Prefix titles with a), b), c) ...
LINE_WIDTH = 1.6             # Selected-member line
SHOW_EVENT_DOTS = True
EVENT_DOT_SIZE = 30
ZERO_LINE_COLOR = "0.45"     # Grey level or any matplotlib colour

# Climatology (all hindcast dates and members): nested percentile bands + median.
# "panel" uses each panel's colour; any matplotlib colour (e.g. "0.55" grey) uses that
# colour in every panel, which leaves colour for the selected member only.
CLIMATOLOGY_COLOR = "panel"
BAND_ALPHAS = [0.15, 0.30]   # Opacity of each band in CLIMATOLOGY_BANDS (outer first)
MEDIAN_COLOR = None          # None = same as the bands; or e.g. "0.35" for grey
MEDIAN_LINE_STYLE = "--"
MEDIAN_LINE_WIDTH = 1.25

# Date axis
X_LIMITS = None              # e.g. ("2023-05-12", "2023-06-10"); None = full forecast
DATE_TICK_INTERVAL_DAYS = 5  # Spacing of labelled date ticks
DATE_FORMAT = "%d %b"        # e.g. "12 May"
DATE_LABEL_ROTATION = 30     # Degrees; 0 = horizontal
DATE_LABEL_ALIGNMENT = "right"  # "left", "center" or "right" (relative to the tick)
# True: dates only under the bottom panel. False: dates under every panel
# (the space between panels grows by EXTRA_SPACING_FOR_DATES to fit them).
SHARE_X_LABELS = True
EXTRA_SPACING_FOR_DATES = 0.4

# Give all mm/day panels the same y-range (panels with their own "ylim" keep it).
SHARE_YLIM_FOR_MM_PER_DAY = False
SHARED_YLIM = None           # (bottom, top), or None to take it from the data

# =============================================================================
# 4. Legend
# =============================================================================
SHOW_LEGEND = True
LEGEND_PANEL = 0             # Panel index (0 = top) that holds the legend
LEGEND_LOCATION = "upper right"
# True: line and dots share one entry (LEGEND_LABELS["member_and_dots"]).
# False: separate entries (LEGEND_LABELS["member"] and LEGEND_LABELS["event_days"]).
COMBINE_LINE_AND_DOTS = True

# All legend text. Placeholders filled in automatically:
#   {member}  ensemble member number of the selected event
#   {n}       number of hindcasts in the climatology (e.g. 220)
LEGEND_LABELS = {
    "member": "Counterfactual Storm Hans",
    "event_days": "Event days",
    "member_and_dots": "Counterfactual Storm Hans",
    # One label per entry in CLIMATOLOGY_BANDS (same order), or None for an
    # automatic label such as "95% range, all hindcasts".
    "bands": [
        "95% range, all hindcasts members (n=220)",
        "25–75% range, all hindcast members",
    ],
    "median": "Median, all hindcast members",
}

# =============================================================================
# 5. Output
# =============================================================================
SAVE_FIGURE = True
SHOW_FIGURE = True
FIGURE_NAME = f"fig-may-counterfactual-timeseries-0{EVENT_RANK}-event"  # File name without extension
OUTPUT_DIR = Path(config.dirs["fig"])
OUTPUT_FORMATS = ["png"]
DPI = 300

# =============================================================================
# 6. Calculations
# =============================================================================
# Climatology bands as (lower, upper) percentiles, widest first. Examples:
#   [(2.5, 97.5)]              one 95% band
#   [(2.5, 97.5), (25, 75)]    95% band plus the middle 50% (interquartile range)
#   [(5, 95), (25, 75)]        90% band plus the middle 50%
CLIMATOLOGY_BANDS = [(2.5, 97.5), (25, 75)]
SHOW_MEDIAN = True
EXCLUDE_SELECTED_FROM_RANGE = False  # True: climatology from the other members only

# =============================================================================
# 7. Data sources (rarely changed)
# =============================================================================
# Sample file used to rank events
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]
OBSERVATION_YEARS = ["1957", "2025"]
ACCUMULATION_DAYS = 2
RANK_VARIABLE = "tp24"
BIAS_CORRECTION_METHOD = "raw"       # "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
BIAS_CORRECTION_REFERENCE = "senorge"  # "senorge" or "era5"
SAMPLE_FILENAME_OVERRIDE = None      # Full path to use instead of the constructed name

# All folders come from config.dirs. Daily model files are read from
#   config.dirs[S2S_DAILY_DIR_KEYS[model_type]] / <variable> / <variable>_<grid>_<date>.nc
# (the config folders already include the domain, e.g. .../daily/europe/).
PATH_CATCHMENT = Path(config.dirs["nve"])
PATH_S2S_PROCESSED = Path(config.dirs["s2s_processed"])
S2S_DAILY_DIR_KEYS = {
    "hindcast": "s2s_hindcast_daily",
    "forecast": "s2s_forecast_daily",
}

# Multiply stored values by these factors to get mm (1000 for files in metres).
TO_MM_FACTORS = {
    "tp24": 1000.0, "rn24": 1000.0, "sf24": 1000.0, "sd": 1000.0,
    "ro24": 1000.0, "sro24": 1000.0,
}
UNIT_WARNING_THRESHOLD_MM = 500.0    # Warn if a daily catchment mean exceeds this


# #############################################################################
#                         END OF USER SETTINGS
# #############################################################################

STYLE_KEYS = {"title", "ylabel", "color", "range_color", "ylim", "zero_line", "yticks"}
PANEL_LABELS = "abcdefgh"
CATCHMENTS = {
    "drammen": {
        "filename": "catchment_nve_regine_drammen.geojson",
        "weights_id": "regine_drammen",
        "label": "Drammen catchment",
    },
    "glomma": {
        "filename": "catchment_nve_regine_glomma.geojson",
        "weights_id": "regine_glomma",
        "label": "Glomma catchment",
    },
}

# =============================================================================
# Validation and event metadata
# =============================================================================


def get_plot_variables():
    """Return the variable of each panel, top to bottom."""
    return list(PLOT_VARIABLES)


def get_panel_settings(variable):
    """Return the full style of one variable's panel from VARIABLE_DEFAULTS."""
    settings = dict(VARIABLE_DEFAULTS[variable])
    settings.setdefault("range_color", None)
    settings.setdefault("yticks", None)
    settings.setdefault("zero_line", False)
    settings.setdefault("ylim", (None, None))
    if settings["range_color"] is None:
        settings["range_color"] = (
            settings["color"] if CLIMATOLOGY_COLOR == "panel" else CLIMATOLOGY_COLOR
        )
    if settings["ylim"] is None:
        settings["ylim"] = (None, None)
    return settings


def validate_settings():
    """Check user settings before reading any data."""
    if not 1 <= len(PLOT_VARIABLES) <= len(PANEL_LABELS):
        raise ValueError(f"PLOT_VARIABLES must contain 1 to {len(PANEL_LABELS)} variables.")
    for variable in PLOT_VARIABLES:
        if variable not in VARIABLE_DEFAULTS:
            raise ValueError(
                f"'{variable}' in PLOT_VARIABLES has no entry in VARIABLE_DEFAULTS. "
                f"Options: {', '.join(VARIABLE_DEFAULTS)}."
            )
    for variable, style in VARIABLE_DEFAULTS.items():
        unknown = set(style) - STYLE_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(unknown)} for '{variable}' in VARIABLE_DEFAULTS. "
                f"Allowed: {sorted(STYLE_KEYS)}."
            )
        missing = {"title", "ylabel", "color"} - set(style)
        if missing:
            raise ValueError(f"'{variable}' in VARIABLE_DEFAULTS is missing {sorted(missing)}.")
        ylim = style.get("ylim")
        if ylim is not None and len(ylim) != 2:
            raise ValueError(f"'ylim' for '{variable}' must be (bottom, top).")
    bands_labels = LEGEND_LABELS.get("bands")
    if bands_labels is not None and len(bands_labels) != len(CLIMATOLOGY_BANDS):
        raise ValueError(
            "LEGEND_LABELS['bands'] needs one label per CLIMATOLOGY_BANDS entry (or None)."
        )
    missing_labels = {"member", "event_days", "member_and_dots", "median"} - set(LEGEND_LABELS)
    if missing_labels:
        raise ValueError(f"LEGEND_LABELS is missing {sorted(missing_labels)}.")
    if not 0 <= LEGEND_PANEL < len(PLOT_VARIABLES):
        raise ValueError("LEGEND_PANEL must be the index of an existing panel (0 = top).")
    get_catchment_settings(CATCHMENT_NAME)
    if not CLIMATOLOGY_BANDS and not SHOW_MEDIAN:
        raise ValueError("Set at least one CLIMATOLOGY_BANDS entry or SHOW_MEDIAN = True.")
    for lower, upper in CLIMATOLOGY_BANDS:
        if not 0 <= lower < upper <= 100:
            raise ValueError("Each CLIMATOLOGY_BANDS entry must satisfy 0 <= lower < upper <= 100.")
    if len(BAND_ALPHAS) < len(CLIMATOLOGY_BANDS):
        raise ValueError("BAND_ALPHAS needs one opacity per CLIMATOLOGY_BANDS entry.")
    if not FIGURE_NAME:
        raise ValueError("FIGURE_NAME must be a non-empty file name.")


def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. Valid options are: {valid_names}."
        )
    return CATCHMENTS[catchment_name]


def get_file_id(catchment_name):
    """Return the short catchment label used in S2S filenames."""
    if catchment_name.startswith("regine_"):
        return catchment_name.replace("regine_", "", 1)
    return catchment_name


def make_sample_filename(catchment_name):
    """Return the monthly-maximum precipitation sample filename used for ranking."""
    if SAMPLE_FILENAME_OVERRIDE is not None:
        return Path(SAMPLE_FILENAME_OVERRIDE)
    if BIAS_CORRECTION_METHOD == "raw":
        correction_label = "raw"
    else:
        correction_label = (
            f"bc_{BIAS_CORRECTION_METHOD}_{BIAS_CORRECTION_REFERENCE}_"
            f"{OBSERVATION_YEARS[0]}-{OBSERVATION_YEARS[1]}"
        )
    return PATH_S2S_PROCESSED / (
        f"monthly_max_samples_{RANK_VARIABLE}_{ACCUMULATION_DAYS}dayacc_"
        f"{get_file_id(catchment_name)}_"
        f"{FORECAST_DATE_RANGE[0]}_{FORECAST_DATE_RANGE[1]}_{correction_label}.nc"
    )


def validate_sample_dataset(ds):
    """Check variables needed to identify the ranked precipitation event."""
    max_var = f"{RANK_VARIABLE}_max"
    required = {max_var, "date_of_max", "lead_of_max", "sample_month", "model_type", "hdate"}
    missing = required - set(ds.variables)
    if missing:
        raise ValueError(f"Sample file is missing variables: {sorted(missing)}")
    if set(ds[max_var].dims) != {"number", "i_date"}:
        raise ValueError(f"{max_var} must have dimensions number and i_date.")
    if set(ds["sample_month"].dims) != {"i_date"}:
        raise ValueError("sample_month must have dimension i_date.")


def format_date(value):
    """Return a compact YYYY-MM-DD date string."""
    value = np.asarray(value).astype("datetime64[ns]")
    if np.isnat(value):
        return "-"
    return np.datetime_as_string(value.astype("datetime64[D]"), unit="D")


def decode_hdate(value):
    """Return YYYY-MM-DD for hindcast hdate, or '-' for forecast rows."""
    if value is None or int(value) == 0:
        return "-"
    text = f"{int(value):08d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def get_ranked_event_indices(ds, month_of_year):
    """Return event indices ranked from largest to smallest precipitation."""
    max_var = f"{RANK_VARIABLE}_max"
    values = ds[max_var].transpose("i_date", "number").values
    sample_month = ds["sample_month"].values.astype("int64")
    valid = np.isfinite(values) & (sample_month % 100 == month_of_year)[:, None]
    i_indices, number_indices = np.where(valid)
    if i_indices.size == 0:
        return []
    order = np.argsort(values[i_indices, number_indices])[::-1]
    return [(int(i_indices[i]), int(number_indices[i])) for i in order]


def get_config_dir(key):
    """Return config.dirs[key] as a Path, with a clear message if the key is missing."""
    if key not in config.dirs:
        raise KeyError(
            f"config.dirs has no entry '{key}'. Add it to the config, or change "
            f"S2S_DAILY_DIR_KEYS. Available keys: {', '.join(sorted(config.dirs))}"
        )
    return Path(config.dirs[key])


def make_s2s_file(event, variable, grid):
    """Return the daily model file for one variable, grid and forecast date."""
    model_type = event["model_type"]
    if model_type not in S2S_DAILY_DIR_KEYS:
        raise ValueError(f"No entry for model type '{model_type}' in S2S_DAILY_DIR_KEYS.")
    directory = get_config_dir(S2S_DAILY_DIR_KEYS[model_type]) / variable
    return directory / f"{variable}_{grid}_{event['forecast_date']}.nc"


def get_selected_event(catchment_name, month_of_year, event_rank):
    """Read the sample file and return metadata for the Nth ranked event."""
    if not 1 <= month_of_year <= 12:
        raise ValueError("EVENT_MONTH must be between 1 and 12.")
    if event_rank < 1:
        raise ValueError("EVENT_RANK must be at least 1.")
    filename = make_sample_filename(catchment_name)
    if not filename.is_file():
        raise FileNotFoundError(f"Sample file not found: {filename}")
    print("Reading ranked events:", filename)
    with xr.open_dataset(filename, decode_timedelta=False) as opened:
        ds = opened.load()
    validate_sample_dataset(ds)
    ranked_indices = get_ranked_event_indices(ds, month_of_year)
    if event_rank > len(ranked_indices):
        raise ValueError(
            f"EVENT_RANK={event_rank} exceeds the {len(ranked_indices)} valid events "
            f"available for month {month_of_year}."
        )
    i_index, number_index = ranked_indices[event_rank - 1]
    sample = ds.isel(i_date=i_index, number=number_index)
    max_var = f"{RANK_VARIABLE}_max"
    hdate = int(ds["hdate"].isel(i_date=i_index).item())
    return {
        "rank": event_rank,
        "max_value": float(sample[max_var].item()),
        "date_of_max": format_date(sample["date_of_max"].values),
        "model_type": str(ds["model_type"].isel(i_date=i_index).item()),
        "forecast_date": format_date(ds["i_date"].isel(i_date=i_index).values),
        "hdate": None if hdate == 0 else hdate,
        "ensemble_member": int(ds["number"].isel(number=number_index).item()) + 1,
        "lead_of_max": float(sample["lead_of_max"].item()),
    }


def get_event_dates(event):
    """Return event-window dates as datetime64[D]."""
    date_of_max = np.datetime64(event["date_of_max"], "D")
    return np.array([date_of_max + np.timedelta64(lag, "D") for lag in EVENT_LAGS])


def print_selected_event(event):
    """Print ranked-event metadata."""
    print(f"\nRank {event['rank']}")
    print(f"  max_value        : {event['max_value']:.2f} mm")
    print(f"  date_of_max      : {event['date_of_max']}")
    print(f"  model_type       : {event['model_type']}")
    print(f"  forecast_date    : {event['forecast_date']}")
    print(f"  hdate            : {decode_hdate(event['hdate'])}")
    print(f"  ensemble_member  : {event['ensemble_member']}")
    print(f"  lead_of_max      : {event['lead_of_max']:.0f} days")
    print(f"  plot_variables   : {', '.join(get_plot_variables())}")


# =============================================================================
# Catchment-average helpers
# =============================================================================


def make_catchment_weights_file(catchment_name, grid):
    """Return the predefined catchment-weight file for one S2S grid."""
    catchment = get_catchment_settings(catchment_name)
    return PATH_CATCHMENT / f"weights_catchment_{catchment['weights_id']}_era5_{grid}.nc"


def load_weights(filename):
    """Load predefined catchment weights."""
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Catchment weights not found: {filename}")
    print("Reading:", filename)
    with xr.open_dataset(filename) as ds:
        if "catchment_weight" not in ds:
            raise KeyError(f"'catchment_weight' not found in {filename}")
        return ds["catchment_weight"].astype("float32").load()


def get_time_coord_name(da):
    """Return the time coordinate name used by the DataArray."""
    for name in ["time", "valid_time"]:
        if name in da.dims or name in da.coords:
            return name
    raise ValueError("Could not identify time coordinate.")


def get_member_coord_name(da):
    """Return ensemble member coordinate name."""
    for name in ["number", "member", "ensemble_member", "realization"]:
        if name in da.dims or name in da.coords:
            return name
    raise ValueError("Could not identify ensemble member coordinate.")


def get_hdate_coord_name(da):
    """Return the hindcast-date coordinate name, or None if absent."""
    for name in ["hdate", "hindcast_date"]:
        if name in da.dims or name in da.coords:
            return name
    return None


def align_weights(da, weights):
    """Align catchment weights to the variable grid."""
    time_name = get_time_coord_name(da)
    template = da.isel({time_name: 0}, drop=True) if time_name in da.dims else da
    try:
        return weights.reindex_like(template)
    except Exception:
        return weights.broadcast_like(template)


def catchment_mean(da, weights):
    """Calculate the catchment-weighted spatial mean."""
    spatial_dims = [name for name in ["latitude", "longitude", "lat", "lon"] if name in da.dims]
    if len(spatial_dims) != 2:
        raise ValueError(f"Could not identify two spatial dimensions in {da.dims}.")
    weights = align_weights(da, weights)
    valid = np.isfinite(da) & np.isfinite(weights) & (weights > 0)
    weighted_sum = (da.where(valid) * weights.where(valid)).sum(spatial_dims, skipna=True)
    weight_sum = weights.where(valid).sum(spatial_dims, skipna=True)
    out = weighted_sum / weight_sum
    out.name = "catchment_mean"
    out.attrs["units"] = da.attrs.get("units", "")
    return out

# =============================================================================
# Data loading
# =============================================================================


def convert_units(da, variable):
    """Convert one stored variable to plotting units."""
    if variable in TO_MM_FACTORS:
        return da * TO_MM_FACTORS[variable]
    if variable == "msl":
        return da / 100.0  # Pa -> hPa
    if variable == "t2m24" and da.attrs.get("units", "K") in ["K", "kelvin", "Kelvin"]:
        return da - 273.15
    return da


def get_data_variable(ds, variable, filename):
    """Return the requested variable, or the only data variable if the name differs."""
    if variable in ds:
        return ds[variable]
    if len(ds.data_vars) == 1:
        name = next(iter(ds.data_vars))
        print(f"Using variable '{name}' in {filename.name} for '{variable}'.")
        return ds[name]
    raise KeyError(f"Variable '{variable}' not found in {filename}; has {list(ds.data_vars)}")


def load_catchment_mean(event, catchment_name, variable, grid):
    """Read one daily variable and return its catchment mean in plotting units.

    Returns all hindcast dates and members. Raises FileNotFoundError if the data or
    the catchment weights for this grid do not exist.
    """
    filename = make_s2s_file(event, variable, grid)
    weights_file = make_catchment_weights_file(catchment_name, grid)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    if not weights_file.exists():
        raise FileNotFoundError(f"Catchment weights not found: {weights_file}")
    weights = load_weights(weights_file)
    print("Reading:", filename)
    with xr.open_dataset(filename) as ds:
        da = get_data_variable(ds, variable, filename)
        mean = convert_units(catchment_mean(da, weights), variable).load()
    check_units(mean, variable)
    return mean


def check_units(mean, variable):
    """Warn when water amounts look implausibly large (likely double conversion)."""
    if variable not in TO_MM_FACTORS:
        return
    largest = float(np.nanmax(np.abs(mean.values)))
    if variable == "sd":  # A state, not a daily flux, so large values are fine.
        return
    if largest > UNIT_WARNING_THRESHOLD_MM:
        print(
            f"WARNING: {variable} reaches {largest:.0f} mm/day as a catchment mean. "
            f"Check TO_MM_FACTORS['{variable}'] (file may already be in mm)."
        )


def compute_delta_swe(event, catchment_name, grid):
    """Return catchment-mean daily ΔSWE (mm/day): ΔSWE(D) = SWE(D+1) - SWE(D).

    SWE is the daily mean of the 00-18 UTC values, so ΔSWE(D) spans ~09 UTC on D to
    ~09 UTC on D+1 and is labelled by day D, like the 00-24 UTC daily fluxes
    (see the module docstring).
    """
    swe = load_catchment_mean(event, catchment_name, "sd", grid)
    time_name = get_time_coord_name(swe)
    delta = swe.diff(time_name, label="lower")
    delta = delta.assign_coords({time_name: delta[time_name].dt.floor("D")})
    delta.name = "delta_swe"
    return delta


def describe_coords(da):
    """Summarise a DataArray's dimensions and coordinate values for error messages."""
    parts = []
    for dim in da.dims:
        values = da[dim].values if dim in da.coords else np.arange(da.sizes[dim])
        if values.size == 0:
            parts.append(f"{dim}: empty")
        elif values.size <= 12:
            parts.append(f"{dim}: {values.tolist()}")
        else:
            parts.append(f"{dim} ({values.size}): {values[0]} ... {values[-1]}")
    return "; ".join(parts)


def compute_plot_variable(event, catchment_name, variable, grid):
    """Return the catchment-mean DataArray for one plotted variable on one grid."""
    if variable == "melt":
        snowfall = load_catchment_mean(event, catchment_name, "sf24", grid)
        delta_swe = compute_delta_swe(event, catchment_name, grid)
        time_name = get_time_coord_name(snowfall)
        snowfall = snowfall.assign_coords({time_name: snowfall[time_name].dt.floor("D")})
        aligned_snowfall, aligned_delta = xr.align(snowfall, delta_swe, join="inner")
        if 0 in aligned_snowfall.shape:
            raise ValueError(
                "Snowfall and ΔSWE share no common coordinates, so melt cannot be computed.\n"
                f"  sf24 : {describe_coords(snowfall)}\n"
                f"  ΔSWE : {describe_coords(delta_swe)}"
            )
        melt = aligned_snowfall - aligned_delta
        melt.name = "melt"
        return melt
    return load_catchment_mean(event, catchment_name, variable, grid)


def get_raw_member_label(da, event):
    """Return the member label available in the current raw S2S variable."""
    member = int(event["ensemble_member"])
    member_name = get_member_coord_name(da)
    available = np.asarray(da[member_name].values)
    if member in available:
        return member
    if event["model_type"] == "hindcast" and member == 11 and 51 in available:
        return 51
    raise KeyError(
        f"Could not find ensemble member {member} in raw {member_name} coordinate. "
        f"Available values: {available.tolist()}"
    )


def summarize_ensemble(mean, event, variable, grid):
    """Return the selected member and the ensemble range as plain numpy arrays."""
    time_name = get_time_coord_name(mean)
    member_name = get_member_coord_name(mean)
    hdate_name = get_hdate_coord_name(mean) if event["model_type"] == "hindcast" else None
    if event["model_type"] == "hindcast" and hdate_name is None:
        raise ValueError("Could not identify hindcast-date coordinate.")

    event_mean = mean.sel({hdate_name: event["hdate"]}) if hdate_name else mean
    selected_member = get_raw_member_label(event_mean, event)
    selected = event_mean.sel({member_name: selected_member})

    sample_dims = [member_name] if hdate_name is None else [hdate_name, member_name]
    ensemble = mean
    if EXCLUDE_SELECTED_FROM_RANGE:
        is_selected = mean[member_name] == selected_member
        if hdate_name is not None:
            is_selected = is_selected & (mean[hdate_name] == event["hdate"])
        ensemble = mean.where(~is_selected)
    percentiles = sorted({p for band in CLIMATOLOGY_BANDS for p in band} | {50.0})
    quantiles = ensemble.quantile(
        [p / 100.0 for p in percentiles], dim=sample_dims, skipna=True
    )
    climatology = {
        p: np.asarray(quantiles.isel(quantile=i).squeeze(drop=True).values, dtype=float)
        for i, p in enumerate(percentiles)
    }
    n_samples = int(np.prod([mean.sizes[dim] for dim in sample_dims]))
    n_samples -= int(EXCLUDE_SELECTED_FROM_RANGE)

    return {
        "variable": variable,
        "times": selected[time_name].values.astype("datetime64[ns]"),
        "selected": np.asarray(selected.values, dtype=float),
        "range_times": quantiles[time_name].values.astype("datetime64[ns]"),
        "climatology": climatology,  # {percentile: values over time}
        "n_samples": n_samples,
        "grid": grid,
    }


def load_variable_timeseries(event, catchment_name, variable):
    """Load one plotted variable, trying each grid until all its inputs exist."""
    errors = []
    for grid in ["0.5x0.5", "0.25x0.25"]:
        try:
            mean = compute_plot_variable(event, catchment_name, variable, grid)
        except FileNotFoundError as error:
            errors.append(str(error))
            continue
        return summarize_ensemble(mean, event, variable, grid)
    raise FileNotFoundError(
        f"Could not load '{variable}' for {event['forecast_date']}:\n  " + "\n  ".join(errors)
    )

# =============================================================================
# Plotting
# =============================================================================


def apply_publication_style():
    """Set consistent, journal-friendly matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.linewidth": 0.7,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,  # Editable text in vector output.
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
    })


def get_time_limits(series_list):
    """Return X_LIMITS, or the time span covered by every panel."""
    if X_LIMITS is not None:
        start, end = X_LIMITS
        return np.datetime64(start, "ns"), np.datetime64(end, "ns")
    start = max(series["times"][0] for series in series_list)
    end = min(series["times"][-1] for series in series_list)
    return start, end


def mark_event_days(ax, series, event_dates, color):
    """Mark the selected member's values on the event days with dots."""
    marker_dates = event_dates.astype("datetime64[ns]")
    on_event_day = np.isin(series["times"], marker_dates)
    ax.scatter(
        series["times"][on_event_day], series["selected"][on_event_day],
        s=EVENT_DOT_SIZE, color=color, edgecolor="white",
        linewidth=0.7, zorder=4, clip_on=False,
    )


def make_panel_title(settings, index):
    """Return the panel title, prefixed with a), b), ... when requested."""
    if SHOW_PANEL_LABELS:
        return f"{PANEL_LABELS[index]}) {settings['title']}"
    return settings["title"]


def plot_climatology(ax, series, settings):
    """Draw the nested percentile bands and the median of all hindcasts."""
    climatology = series["climatology"]
    for (lower, upper), alpha in zip(CLIMATOLOGY_BANDS, BAND_ALPHAS):
        ax.fill_between(
            series["range_times"], climatology[float(lower)], climatology[float(upper)],
            color=settings["range_color"], alpha=alpha, linewidth=0, zorder=2,
        )
    if SHOW_MEDIAN:
        ax.plot(
            series["range_times"], climatology[50.0],
            color=MEDIAN_COLOR or settings["range_color"], linestyle=MEDIAN_LINE_STYLE,
            linewidth=MEDIAN_LINE_WIDTH, zorder=2.5,
        )

def describe_band(lower, upper):
    """Return a legend label such as '95% range' or '25-75% range'."""
    width = upper - lower
    if width >= 90 and np.isclose(lower, 100 - upper):
        return f"{width:g}% range"
    return f"{lower:g}\u2013{upper:g}% range"



def plot_panel(ax, series, settings, event_dates, index):
    """Plot one panel: ensemble range, selected member and event days."""
    if settings["zero_line"]:
        ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.7, zorder=1)
    plot_climatology(ax, series, settings)
    ax.plot(
        series["times"], series["selected"],
        color=settings["color"], linewidth=LINE_WIDTH, solid_capstyle="round", zorder=3,
    )
    if SHOW_EVENT_DOTS:
        mark_event_days(ax, series, event_dates, settings["color"])
    ax.set_title(make_panel_title(settings, index), loc="left", pad=3)
    ax.set_ylabel(settings["ylabel"])
    bottom, top = settings["ylim"]
    ax.set_ylim(bottom=bottom, top=top)
    if settings["yticks"] is not None:
        ax.set_yticks(settings["yticks"])
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0)


def apply_shared_ylim(axes, panel_settings):
    """Give all mm/day panels whose top y-limit is not fixed the same y-range."""
    shared = [
        ax for ax, settings in zip(axes, panel_settings)
        if settings["ylabel"] == MM_PER_DAY and settings["ylim"][1] is None
    ]
    if not shared:
        return
    if SHARED_YLIM is not None:
        bottom, top = SHARED_YLIM
    else:
        bottom = min(ax.get_ylim()[0] for ax in shared)
        top = max(ax.get_ylim()[1] for ax in shared)
    for ax in shared:
        ax.set_ylim(bottom, top)


def fill_label(text, event, n_samples):
    """Fill the {member} and {n} placeholders of one legend label."""
    return text.format(member=event["ensemble_member"], n=n_samples)


def add_legend(ax, event, n_samples):
    """Add one shared legend using neutral grey handles and LEGEND_LABELS text."""
    band_color = "0.25" if CLIMATOLOGY_COLOR == "panel" else CLIMATOLOGY_COLOR
    who = "other hindcasts" if EXCLUDE_SELECTED_FROM_RANGE else "all hindcasts"
    dot_style = {"marker": "o", "markersize": 5, "markerfacecolor": "0.15",
                 "markeredgecolor": "white", "markeredgewidth": 0.7}

    if SHOW_EVENT_DOTS and COMBINE_LINE_AND_DOTS:
        handles = [Line2D([0], [0], color="0.15", linewidth=LINE_WIDTH, **dot_style,
                          label=fill_label(LEGEND_LABELS["member_and_dots"], event, n_samples))]
    else:
        handles = [Line2D([0], [0], color="0.15", linewidth=LINE_WIDTH,
                          label=fill_label(LEGEND_LABELS["member"], event, n_samples))]

    # Each band is shown as it appears on top of the wider ones.
    band_labels = LEGEND_LABELS.get("bands") or [None] * len(CLIMATOLOGY_BANDS)
    cumulative_alpha = 0.0
    for index, ((lower, upper), alpha, text) in enumerate(
        zip(CLIMATOLOGY_BANDS, BAND_ALPHAS, band_labels)
    ):
        cumulative_alpha = 1 - (1 - cumulative_alpha) * (1 - alpha)
        if text is None:
            text = f"{describe_band(lower, upper)}, {who}" + (" (n={n})" if index == 0 else "")
        handles.append(Patch(facecolor=band_color, alpha=cumulative_alpha, linewidth=0,
                             label=fill_label(text, event, n_samples)))

    if SHOW_MEDIAN:
        handles.append(Line2D([0], [0], color=MEDIAN_COLOR or band_color,
                              linestyle=MEDIAN_LINE_STYLE, linewidth=MEDIAN_LINE_WIDTH,
                              label=fill_label(LEGEND_LABELS["median"], event, n_samples)))
    if SHOW_EVENT_DOTS and not COMBINE_LINE_AND_DOTS:
        handles.append(Line2D([0], [0], linestyle="none", **dot_style,
                              label=fill_label(LEGEND_LABELS["event_days"], event, n_samples)))
    ax.legend(handles=handles, loc=LEGEND_LOCATION, frameon=False,
              handlelength=1.6, borderaxespad=0.2)


def format_time_axis(ax, start, end, show_labels=True):
    """Format one panel's date axis; tick labels only when show_labels is True."""
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=DATE_TICK_INTERVAL_DAYS))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.5)
    ax.tick_params(axis="x", which="major", labelbottom=show_labels)
    if show_labels:
        plt.setp(ax.get_xticklabels(), rotation=DATE_LABEL_ROTATION, ha=DATE_LABEL_ALIGNMENT)


def make_figure(series_list, event, event_dates, catchment_label):
    """Create the stacked time-series figure."""
    apply_publication_style()
    n_panels = len(series_list)
    height = PANEL_HEIGHT_IN * n_panels
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True, squeeze=False,
        figsize=(FIGURE_WIDTH_IN, height),
        gridspec_kw={
            "hspace": PANEL_SPACING + (0 if SHARE_X_LABELS else EXTRA_SPACING_FOR_DATES)
        },
    )
    axes = axes[:, 0]
    panel_settings = [get_panel_settings(variable) for variable in PLOT_VARIABLES]
    for index, (ax, series, settings) in enumerate(zip(axes, series_list, panel_settings)):
        plot_panel(ax, series, settings, event_dates, index)
    if SHARE_YLIM_FOR_MM_PER_DAY:
        apply_shared_ylim(axes, panel_settings)
    start, end = get_time_limits(series_list)
    for ax in axes:
        format_time_axis(ax, start, end, show_labels=(ax is axes[-1]) or not SHARE_X_LABELS)
    if SHOW_LEGEND:
        add_legend(axes[LEGEND_PANEL], event, series_list[LEGEND_PANEL]["n_samples"])
    top_margin_in = 0.15
    if FIGURE_TITLE:
        title = FIGURE_TITLE.format(catchment=catchment_label)
        fig.suptitle(title, x=0.5, y=1 - 0.05 / height, fontsize=FIGURE_TITLE_SIZE)
        top_margin_in = 0.42
    fig.align_ylabels(axes)
    # Fixed margins in inches, whatever the number of panels.
    fig.subplots_adjust(left=0.09, right=0.985, top=1 - top_margin_in / height,
                        bottom=0.35 / height)
    return fig


def save_and_show(fig):
    """Save the figure in each requested format, then show and close it."""
    if SAVE_FIGURE:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for fmt in OUTPUT_FORMATS:
            path = OUTPUT_DIR / f"{FIGURE_NAME}.{fmt}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print("Saved figure:", path)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def print_event_values(series_list, event_dates):
    """Print selected-member values and the ensemble range on the event days."""
    print("\nCatchment-mean values on the event days (selected | climatology)")
    for series in series_list:
        print(f"  {series['variable']}:")
        for date in event_dates.astype("datetime64[ns]"):
            selected = series["selected"][series["times"] == date]
            in_range = series["range_times"] == date
            if selected.size == 0 or not in_range.any():
                continue
            climatology = series["climatology"]
            parts = [f"median {climatology[50.0][in_range][0]:6.2f}"]
            for lower, upper in CLIMATOLOGY_BANDS:
                parts.append(
                    f"{lower:g}-{upper:g}%: [{climatology[float(lower)][in_range][0]:6.2f}, "
                    f"{climatology[float(upper)][in_range][0]:6.2f}]"
                )
            print(
                f"    {np.datetime_as_string(date, unit='D')}: {selected[0]:7.2f} | "
                + " | ".join(parts)
            )


def main():
    """Run the full plotting workflow."""
    validate_settings()
    catchment = get_catchment_settings(CATCHMENT_NAME)
    event = get_selected_event(
        catchment_name=catchment["weights_id"],
        month_of_year=EVENT_MONTH,
        event_rank=EVENT_RANK,
    )
    print_selected_event(event)
    event_dates = get_event_dates(event)
    series_list = [
        load_variable_timeseries(event, CATCHMENT_NAME, variable)
        for variable in get_plot_variables()
    ]
    print_event_values(series_list, event_dates)
    fig = make_figure(series_list, event, event_dates, catchment["label"])
    save_and_show(fig)


if __name__ == "__main__":
    main()
