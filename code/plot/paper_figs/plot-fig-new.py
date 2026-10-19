"""
Plot the catchment-mean evolution of one high-ranking S2S precipitation event as
vertically stacked time-series panels (one per entry in PLOT_VARIABLES, up to eight).

Event selection is unchanged from the map script: events are ranked by accumulated
precipitation (RANK_VARIABLE) in the monthly-maximum sample file, and the
EVENT_RANK-th event in EVENT_MONTH is selected.

Each panel shows one variable from PLOT_VARIABLES as a catchment-weighted spatial
mean:
    - the selected ensemble member (solid line), and
    - the central 95% range over all hindcast dates and ensemble members (shading).
The event-window days (date_of_max + EVENT_LAGS) are marked with dots on the
selected-member line in every panel.

Supported variables: "tp24", "rn24", "sf24", "msl", "sd", "sm", "melt", "ro24",
"sro24", "t2m24".

Daily accumulated variables (tp24, rn24, sf24, ro24, sro24) are labelled by the
start of their 24-hour window: the value on day D covers 00-24 UTC on day D.

Snow variables derived from snow water equivalent (SWE, stored as "sd"):
    "sm"   : daily change in SWE, ΔSWE (negative = net loss from the snowpack).
    "neg_sm": -ΔSWE (positive = net loss from the snowpack), on the same sign as
             "melt" so the two can be compared directly.
    "melt" : snowmelt estimated from the snow budget, melt = sf24 - ΔSWE
             (positive = melt). Unlike ΔSWE, this does not hide melt on days
             when snow also falls.
Daily snowfall "sf24" is read from the daily files written by convert_sf6_to_sf24.py
(00-24 UTC on the labelled day, like tp24).
How ΔSWE is computed is set by SWE_METHOD:
    "daily_mean_upper" : ΔSWE(D) = SWE(D) - SWE(D-1), daily-mean SWE.
                         Window ~09 UTC D-1 to 09 UTC D (original definition).
    "daily_mean_lower" : ΔSWE(D) = SWE(D+1) - SWE(D), daily-mean SWE.
                         Window ~09 UTC D to 09 UTC D+1 (closer to tp24).
    "snapshot_00utc"   : ΔSWE(D) = SWE(00 UTC D+1) - SWE(00 UTC D), from the
                         6-hourly SWE files. Same 00-24 UTC window as tp24 and sf24.

6-hourly SWE files are assumed to be stored like the sf6 files:
    - split into control and perturbed files, e.g. sd_CY47R3_0.5x0.5_<date>_cf.nc
      and ..._pf.nc, combined with the control appended as the last member;
    - a "time" coordinate that counts forecast steps as if the forecast had started
      on 1900-01-01 00 UTC (e.g. "hours since 1900-01-16 00:00:00"), converted here
      to valid time = forecast date + step;
    - the control member relabelled to match the daily files (e.g. 11 -> 51).
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Dunnsigouin_etal_2026 import config

# =============================================================================
# User settings: event selection
# =============================================================================
CATCHMENT_NAME = "drammen"  # options: "drammen", "glomma"
EVENT_MONTH = 5
EVENT_RANK = 1
FORECAST_DATE_RANGE = ["2020-01-02", "2023-12-28"]
OBSERVATION_YEARS = ["1957", "2025"]
ACCUMULATION_DAYS = 2
# Options: "raw", "q", "doy", "ld", "q_doy", "mm_1step", "mm_2step"
BIAS_CORRECTION_METHOD = "raw"
# Options: "senorge", "era5"
BIAS_CORRECTION_REFERENCE = "senorge"
SAMPLE_FILENAME_OVERRIDE = None
RANK_VARIABLE = "tp24"

# =============================================================================
# User settings: plotted variables and dates
# =============================================================================
# One to eight variables, plotted top to bottom as panels a, b, c, ...
# "t2m24" is daily-mean 2 m temperature in °C, with a 0 °C reference line.
# Options: "tp24", "rn24", "sf24", "msl", "sd", "sm", "neg_sm", "melt", "ro24",
#          "sro24", "t2m24"
# e.g. ["sf24", "neg_sm", "melt", "ro24"] to check melt = snowfall + (-ΔSWE).
PLOT_VARIABLES = ["tp24", "t2m24", "sd", "melt", "ro24"]
# How daily ΔSWE is computed for "sm" and "melt" (see module docstring).
# Options: "daily_mean_upper", "daily_mean_lower", "snapshot_00utc"
SWE_METHOD = "daily_mean_lower"
EVENT_LAGS = [-2, -1, 0, 1]  # Days relative to date_of_max; marked with dots.
INTERVAL_PERCENTILES = (2.5, 97.5)  # Central 95% range.
# False: range over all hindcast dates and members (as in the map script).
# True: exclude the selected member itself from the range.
EXCLUDE_SELECTED_FROM_RANGE = False
# True: all mm/day panels share the same y-axis range; other units (°C, hPa, mm of
# SWE) keep their own.
SHARE_YLIM = False
# Shared y-axis range as (bottom, top); None picks it from the data in all panels.
SHARED_YLIM = None
WRITE_TO_FILE = False
OUTPUT_FORMATS = ["png", "pdf"]

# =============================================================================
# Paths
# =============================================================================
PATH_OUT = config.dirs["fig"]
PATH_CATCHMENT = config.dirs["nve"]
PATH_S2S_PROCESSED = Path(config.dirs["s2s_processed"])
S2S_BASE_DIR = Path("/nird/datapeak/NS9873K/etdu/raw/s2s/mars/ecmwf")
# Sub-folders of S2S_BASE_DIR/<model_type>/sfc/ holding daily and 6-hourly files.
DAILY_FOLDER = "daily"
SIXHOURLY_FOLDER = "6hourly"
DOMAIN = "europe"
# Factor converting each stored variable to mm (or mm/day). Files written in metres
# need 1000; set 1.0 for any variable already stored in mm.
TO_MM_FACTORS = {
    "tp24": 1000.0, "rn24": 1000.0, "sf24": 1000.0, "sd": 1000.0, "sd6": 1000.0,
    "ro24": 1000.0, "sro24": 1000.0,
}
# 6-hourly SWE for SWE_METHOD = "snapshot_00utc": names tried in order, each used as
# the sub-folder and filename prefix, e.g. .../6hourly/europe/sd6/sd6_CY47R3_..._cf.nc.
SWE_6HOURLY_NAMES = ["sd6", "sd"]
# ECMWF model cycle in 6-hourly filenames, e.g. "CY47R3". None accepts any cycle (an
# error is raised if more than one cycle matches the same forecast date).
MODEL_CYCLE = None
# 6-hourly "time" values count forecast steps from this fake start time when the time
# units refer to a year before FAKE_TIME_YEAR_LIMIT (as in the sf6 files).
STEP_REFERENCE = "1900-01-01T00:00"
FAKE_TIME_YEAR_LIMIT = 1950
# Catchment-mean daily values above this (mm/day) trigger a unit warning.
UNIT_WARNING_THRESHOLD_MM = 500.0

# =============================================================================
# Figure settings (sized for a double-column journal figure)
# =============================================================================
FIG_WIDTH_IN = 7.0
PANEL_HEIGHT_IN = 1.75
FIG_DPI = 300
FONT_SIZE = 9
TITLE_FONTSIZE = 9.5
LEGEND_FONTSIZE = 8
RANGE_ALPHA = 0.22
SELECTED_LINEWIDTH = 1.6
EVENT_MARKER_SIZE = 22
EVENT_MARKER_EDGEWIDTH = 0.7
ZERO_LINE_COLOR = "0.45"
TICK_INTERVAL_DAYS = 5
PANEL_HSPACE = 0.32

# "color" sets both the selected-member line and its shaded range.
# "zero_line" draws a reference line at 0 (ΔSWE sign change, freezing point).
VARIABLE_SETTINGS = {
    "tp24": {"title": "Total precipitation", "units": r"mm day$^{-1}$",
             "color": "#1f5fa8", "zero_line": False, "ymin": 0.0},
    "rn24": {"title": "Rainfall", "units": r"mm day$^{-1}$",
             "color": "#2a9d8f", "zero_line": False, "ymin": 0.0},
    "sf24": {"title": "Snowfall", "units": r"mm day$^{-1}$",
             "color": "#4a7fb5", "zero_line": False, "ymin": 0.0},
    "msl": {"title": "Mean sea level pressure", "units": "hPa",
            "color": "#555555", "zero_line": False, "ymin": None},
    "sd": {"title": "Snow water equivalent", "units": "mm",
           "color": "#6a51a3", "zero_line": False, "ymin": 0.0},
    "sm": {"title": r"Daily $\Delta$SWE (negative = net snow loss)", "units": r"mm day$^{-1}$",
           "color": "#7b3294", "zero_line": True, "ymin": None},
    "neg_sm": {"title": r"$-\Delta$SWE (positive = net snow loss)", "units": r"mm day$^{-1}$",
               "color": "#9e6fb8", "zero_line": True, "ymin": None},
    "melt": {"title": r"Snowmelt (snowfall $-$ $\Delta$SWE)", "units": r"mm day$^{-1}$",
             "color": "#7b3294", "zero_line": True, "ymin": None},
    "ro24": {"title": "Total runoff", "units": r"mm day$^{-1}$",
             "color": "#c2571a", "zero_line": False, "ymin": 0.0},
    "sro24": {"title": "Surface runoff", "units": r"mm day$^{-1}$",
              "color": "#d98a2b", "zero_line": False, "ymin": 0.0},
    "t2m24": {"title": "Daily-mean 2 m temperature", "units": r"$^{\circ}$C",
              "color": "#b2182b", "zero_line": True, "ymin": None},
}
SWE_METHODS = {"daily_mean_upper", "daily_mean_lower", "snapshot_00utc"}
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


def validate_settings():
    """Check user settings before reading any data."""
    if not 1 <= len(PLOT_VARIABLES) <= len(PANEL_LABELS):
        raise ValueError(f"PLOT_VARIABLES must contain 1 to {len(PANEL_LABELS)} variables.")
    for variable in PLOT_VARIABLES:
        get_variable_settings(variable)
    get_catchment_settings(CATCHMENT_NAME)
    if SWE_METHOD not in SWE_METHODS:
        raise ValueError(f"SWE_METHOD must be one of {sorted(SWE_METHODS)}.")
    lower, upper = INTERVAL_PERCENTILES
    if not 0 <= lower < upper <= 100:
        raise ValueError("INTERVAL_PERCENTILES must satisfy 0 <= lower < upper <= 100.")


def get_catchment_settings(catchment_name):
    """Return settings for the selected catchment."""
    if catchment_name not in CATCHMENTS:
        valid_names = ", ".join(CATCHMENTS)
        raise ValueError(
            f"Unknown catchment '{catchment_name}'. Valid options are: {valid_names}."
        )
    return CATCHMENTS[catchment_name]


def get_variable_settings(variable):
    """Validate and return plotting settings for one variable."""
    if variable not in VARIABLE_SETTINGS:
        valid_variables = ", ".join(VARIABLE_SETTINGS)
        raise ValueError(
            f"Unknown variable '{variable}'. Valid options are: {valid_variables}."
        )
    return VARIABLE_SETTINGS[variable]


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


def make_s2s_file(event, variable, grid, folder=DAILY_FOLDER):
    """Create the expected raw S2S file path (daily or 6-hourly)."""
    return (
        S2S_BASE_DIR
        / event["model_type"]
        / "sfc"
        / folder
        / DOMAIN
        / variable
        / f"{variable}_{grid}_{event['forecast_date']}.nc"
    )


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
    print(f"  plot_variables   : {', '.join(PLOT_VARIABLES)}")
    if any(variable in ["sm", "neg_sm", "melt"] for variable in PLOT_VARIABLES):
        print(f"  swe_method       : {SWE_METHOD}")


def make_output_stem(catchment_name, event_rank):
    """Create the output filename stem (without extension)."""
    variables = "-".join(PLOT_VARIABLES)
    return (
        f"{PATH_OUT}storyline-timeseries-{variables}-{catchment_name}-"
        f"month-{EVENT_MONTH:02d}-rank-{event_rank:02d}-"
        f"{FORECAST_DATE_RANGE[0]}-{FORECAST_DATE_RANGE[-1]}"
    )

# =============================================================================
# Catchment-average helpers
# =============================================================================


def make_catchment_weights_file(catchment_name, grid):
    """Return the predefined catchment-weight file for one S2S grid."""
    catchment = get_catchment_settings(catchment_name)
    return Path(PATH_CATCHMENT) / f"weights_catchment_{catchment['weights_id']}_era5_{grid}.nc"


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


def find_files(event, variable, grid, folder, suffix):
    """Return files in the variable's folder for this grid and forecast date.

    Matches names with or without a model cycle, e.g. both
    sf6_0.5x0.5_2023-04-27_cf.nc and sf6_CY47R3_0.5x0.5_2023-04-27_cf.nc
    (suffix "_cf"), or sd_0.5x0.5_2023-04-27.nc (suffix "").
    """
    directory = make_s2s_file(event, variable, grid, folder).parent
    if not directory.is_dir():
        return []
    cycle = f"{MODEL_CYCLE}_" if MODEL_CYCLE else "*"
    pattern = f"{variable}_{cycle}{grid}_{event['forecast_date']}{suffix}.nc"
    return sorted(directory.glob(pattern))


def pick_one(files, description):
    """Return the single matching file, None if there is none, or raise if ambiguous."""
    if len(files) > 1:
        names = ", ".join(path.name for path in files)
        raise ValueError(
            f"Several {description} files match ({names}). Set MODEL_CYCLE to choose one."
        )
    return files[0] if files else None


def find_split_member_files(event, variable, grid, folder):
    """Return (control, perturbed) paths for a variable stored as _cf/_pf files, or None."""
    cf_file = pick_one(find_files(event, variable, grid, folder, "_cf"), "control (_cf)")
    pf_file = pick_one(find_files(event, variable, grid, folder, "_pf"), "perturbed (_pf)")
    if cf_file is None and pf_file is None:
        return None
    if cf_file is None or pf_file is None:
        found = cf_file or pf_file
        raise FileNotFoundError(f"Found {found.name} but not its matching _cf/_pf partner.")
    if cf_file.name.replace("_cf", "") != pf_file.name.replace("_pf", ""):
        raise ValueError(f"Control and perturbed files do not match: {cf_file.name}, {pf_file.name}")
    return cf_file, pf_file


def get_data_variable(ds, variable, filename):
    """Return the requested variable, or the only data variable if the name differs."""
    if variable in ds:
        return ds[variable]
    if len(ds.data_vars) == 1:
        name = next(iter(ds.data_vars))
        print(f"Using variable '{name}' in {filename.name} for '{variable}'.")
        return ds[name]
    raise KeyError(f"Variable '{variable}' not found in {filename}; has {list(ds.data_vars)}")


def fix_6hourly_time(da, forecast_date):
    """Convert an undecoded 6-hourly "time" coordinate to valid times.

    If the time units refer to a date before FAKE_TIME_YEAR_LIMIT, the values are
    forecast steps counted from STEP_REFERENCE, and valid time = forecast date + step.
    Otherwise the units already give real dates and are decoded as usual.
    """
    units = da["time"].attrs.get("units", "")
    if not units.startswith("hours since "):
        raise ValueError(f"Unexpected 6-hourly time units '{units}'; expected 'hours since ...'.")
    origin = np.datetime64(units.removeprefix("hours since ").strip().replace(" ", "T"), "h")
    hours = da["time"].values.astype("int64")
    if origin.astype("datetime64[Y]").astype(int) + 1970 < FAKE_TIME_YEAR_LIMIT:
        steps = hours + (origin - np.datetime64(STEP_REFERENCE, "h")).astype(int)
        if np.any(steps % 6 != 0) or steps.min() < 0 or steps.max() > 24 * 50:
            raise ValueError(f"Derived forecast steps look wrong: {steps[:3]} ... {steps[-3:]} h")
        valid = np.datetime64(forecast_date, "h") + steps.astype("timedelta64[h]")
        print(f"  6-hourly steps {steps[0]}-{steps[-1]} h -> valid {valid[0]} to {valid[-1]}")
    else:
        valid = origin + hours.astype("timedelta64[h]")
    return da.assign_coords(time=valid.astype("datetime64[ns]"))


def combine_control_and_perturbed(da_cf, da_pf):
    """Attach the control run as the last ensemble member, as in the download script.

    Perturbed members keep their numbers (1..N); the control becomes member N+1 and is
    concatenated after them along "number", matching how tp24 files were built.
    """
    member_name = get_member_coord_name(da_pf)
    n_ensemble = da_pf[member_name].size
    if member_name in da_cf.dims:
        da_cf = da_cf.isel({member_name: 0}, drop=True)
    elif member_name in da_cf.coords:
        da_cf = da_cf.drop_vars(member_name)
    position = da_pf.dims.index(member_name)
    da_cf = da_cf.expand_dims({member_name: [n_ensemble + 1]}, axis=position)
    return xr.concat([da_pf, da_cf], dim=member_name)


def open_variable(event, variable, grid, folder):
    """Open one variable as a lazily loaded DataArray, plus a description for printing.

    Uses the single combined file when it exists; otherwise combines _cf and _pf files.
    """
    filename = make_s2s_file(event, variable, grid, folder)
    if not filename.exists():
        # Combined file whose name includes a model cycle, e.g. sd_CY47R3_0.5x0.5_<date>.nc.
        combined = pick_one(find_files(event, variable, grid, folder, ""), "combined")
        filename = combined if combined is not None else filename
    sixhourly = folder == SIXHOURLY_FOLDER

    def prepare(ds, path):
        da = get_data_variable(ds, variable, path)
        return fix_6hourly_time(da, event["forecast_date"]) if sixhourly else da

    if filename.exists():
        print("Reading:", filename)
        ds = xr.open_dataset(filename, decode_times=not sixhourly)
        return prepare(ds, filename), [ds]
    split = find_split_member_files(event, variable, grid, folder)
    if split is None:
        raise FileNotFoundError(f"File not found: {filename} (and no _cf/_pf pair)")
    cf_file, pf_file = split
    print("Reading:", cf_file)
    print("Reading:", pf_file)
    ds_cf = xr.open_dataset(cf_file, decode_times=not sixhourly)
    ds_pf = xr.open_dataset(pf_file, decode_times=not sixhourly)
    da = combine_control_and_perturbed(prepare(ds_cf, cf_file), prepare(ds_pf, pf_file))
    return da, [ds_cf, ds_pf]


def load_catchment_mean(
    event, catchment_name, variable, grid, folder=DAILY_FOLDER, select_00utc=False
):
    """Read one stored variable and return its catchment mean in plotting units.

    Returns all hindcast dates and members. Raises FileNotFoundError if the data or
    the catchment weights for this grid do not exist.
    """
    weights_file = make_catchment_weights_file(catchment_name, grid)
    if not weights_file.exists():
        raise FileNotFoundError(f"Catchment weights not found: {weights_file}")
    da, opened = open_variable(event, variable, grid, folder)
    try:
        weights = load_weights(weights_file)
        if select_00utc:
            # Keep only the 00 UTC snapshots before averaging over the catchment.
            time_name = get_time_coord_name(da)
            da = da.where(da[time_name].dt.hour == 0, drop=True)
        mean = convert_units(catchment_mean(da, weights), variable).load()
    finally:
        for ds in opened:
            ds.close()
    check_units(mean, variable)
    return mean


def check_units(mean, variable):
    """Warn when water amounts look implausibly large (likely double conversion)."""
    if variable not in TO_MM_FACTORS:
        return
    largest = float(np.nanmax(np.abs(mean.values)))
    if variable in ["sd", "sd6"]:  # A state, not a daily flux, so large values are fine.
        return
    if largest > UNIT_WARNING_THRESHOLD_MM:
        print(
            f"WARNING: {variable} reaches {largest:.0f} mm/day as a catchment mean. "
            f"Check TO_MM_FACTORS['{variable}'] (file may already be in mm)."
        )


def compute_delta_swe(event, catchment_name, grid):
    """Return catchment-mean daily ΔSWE (mm/day) using SWE_METHOD.

    All methods return values on daily time stamps at 00 UTC. For "daily_mean_lower"
    and "snapshot_00utc" the value on day D is the change during day D, matching
    the labelling of tp24, sf24 and ro24.
    """
    if SWE_METHOD == "snapshot_00utc":
        swe = load_6hourly_swe_snapshots(event, catchment_name, grid)
        label = "lower"
    else:
        swe = load_catchment_mean(event, catchment_name, "sd", grid, DAILY_FOLDER)
        label = "upper" if SWE_METHOD == "daily_mean_upper" else "lower"
    time_name = get_time_coord_name(swe)
    delta = swe.diff(time_name, label=label)
    # Snapshots are stamped at 00 UTC already; make the daily-mean stamps match too.
    delta = delta.assign_coords({time_name: delta[time_name].dt.floor("D")})
    delta.name = "sm"
    return delta


def load_6hourly_swe_snapshots(event, catchment_name, grid):
    """Return catchment-mean SWE (mm) at 00 UTC from the 6-hourly files."""
    errors = []
    for name in SWE_6HOURLY_NAMES:
        try:
            return load_catchment_mean(
                event, catchment_name, name, grid, SIXHOURLY_FOLDER, select_00utc=True
            )
        except (FileNotFoundError, KeyError) as error:
            errors.append(str(error))
    raise FileNotFoundError("No 6-hourly SWE found:\n  " + "\n  ".join(errors))


def match_member_labels(da, reference):
    """Copy the reference member labels when only the last (control) label differs.

    The daily files label the hindcast control run 51, while combining 6-hourly
    _cf/_pf files gives N+1 (e.g. 11). Perturbed members must match exactly.
    """
    member_name = get_member_coord_name(da)
    labels, reference_labels = da[member_name].values, reference[member_name].values
    if np.array_equal(labels, reference_labels):
        return da
    if labels.size == reference_labels.size and np.array_equal(labels[:-1], reference_labels[:-1]):
        print(f"  control member relabelled {labels[-1]} -> {reference_labels[-1]}")
        return da.assign_coords({member_name: reference_labels})
    return da


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
    if variable == "sm":
        return compute_delta_swe(event, catchment_name, grid)
    if variable == "neg_sm":
        negative = -compute_delta_swe(event, catchment_name, grid)
        negative.name = "neg_sm"
        return negative
    if variable == "melt":
        snowfall = load_catchment_mean(event, catchment_name, "sf24", grid)
        delta_swe = match_member_labels(
            compute_delta_swe(event, catchment_name, grid), snowfall
        )
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
    lower_q, upper_q = (p / 100.0 for p in INTERVAL_PERCENTILES)
    lower = ensemble.quantile(lower_q, dim=sample_dims, skipna=True).squeeze(drop=True)
    upper = ensemble.quantile(upper_q, dim=sample_dims, skipna=True).squeeze(drop=True)
    n_samples = int(np.prod([mean.sizes[dim] for dim in sample_dims]))
    n_samples -= int(EXCLUDE_SELECTED_FROM_RANGE)

    return {
        "variable": variable,
        "times": selected[time_name].values.astype("datetime64[ns]"),
        "selected": np.asarray(selected.values, dtype=float),
        "range_times": lower[time_name].values.astype("datetime64[ns]"),
        "lower": np.asarray(lower.values, dtype=float),
        "upper": np.asarray(upper.values, dtype=float),
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
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.linewidth": 0.7,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": LEGEND_FONTSIZE,
        "pdf.fonttype": 42,  # Editable text in vector output.
        "ps.fonttype": 42,
        "savefig.dpi": FIG_DPI,
    })


def get_common_time_limits(series_list):
    """Return the time span covered by every panel (ΔSWE series are one day shorter)."""
    start = max(series["times"][0] for series in series_list)
    end = min(series["times"][-1] for series in series_list)
    return start, end


def mark_event_days(ax, series, event_dates, color):
    """Mark the selected member's values on the event-window days with dots."""
    marker_dates = event_dates.astype("datetime64[ns]")
    on_event_day = np.isin(series["times"], marker_dates)
    ax.scatter(
        series["times"][on_event_day], series["selected"][on_event_day],
        s=EVENT_MARKER_SIZE, color=color, edgecolor="white",
        linewidth=EVENT_MARKER_EDGEWIDTH, zorder=4, clip_on=False,
    )


def plot_variable_panel(ax, series, event_dates, panel_label):
    """Plot one variable: ensemble range, selected member, and event window."""
    settings = get_variable_settings(series["variable"])
    color = settings["color"]
    if settings["zero_line"]:
        ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=0.7, zorder=1)
    ax.fill_between(
        series["range_times"], series["lower"], series["upper"],
        color=color, alpha=RANGE_ALPHA, linewidth=0, zorder=2,
    )
    ax.plot(
        series["times"], series["selected"],
        color=color, linewidth=SELECTED_LINEWIDTH, solid_capstyle="round", zorder=3,
    )
    mark_event_days(ax, series, event_dates, color)
    ax.set_title(f"{panel_label}) {settings['title']}", loc="left", pad=3)
    ax.set_ylabel(settings["units"])
    if settings["ymin"] is not None:
        ax.set_ylim(bottom=settings["ymin"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.margins(x=0)


def add_legend(ax, event, n_samples):
    """Add one shared legend to the top panel using neutral grey handles."""
    lower, upper = INTERVAL_PERCENTILES
    range_label = f"{upper - lower:.0f}% range, all hindcast dates and members (n={n_samples})"
    if EXCLUDE_SELECTED_FROM_RANGE:
        range_label = f"{upper - lower:.0f}% range, other hindcast members (n={n_samples})"
    handles = [
        Line2D([0], [0], color="0.25", linewidth=SELECTED_LINEWIDTH,
               label=f"Selected member {event['ensemble_member']}"),
        Patch(facecolor="0.25", alpha=RANGE_ALPHA + 0.1, linewidth=0, label=range_label),
        Line2D([0], [0], linestyle="none", marker="o", markersize=5,
               markerfacecolor="0.25", markeredgecolor="white",
               markeredgewidth=EVENT_MARKER_EDGEWIDTH, label="Event days"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handlelength=1.6, borderaxespad=0.2)


def format_time_axis(ax, start, end):
    """Format the shared date axis on the bottom panel."""
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=TICK_INTERVAL_DAYS))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.5)


def apply_shared_ylim(axes, series_list):
    """Give every water-amount panel (mm/day) the same y-axis range.

    Panels in other units (e.g. temperature, pressure) keep their own range.
    """
    water_units = r"mm day$^{-1}$"
    shared = [
        ax for ax, series in zip(axes, series_list)
        if get_variable_settings(series["variable"])["units"] == water_units
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
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))


def make_figure(series_list, event, event_dates, catchment_label):
    """Create the four-panel stacked time-series figure."""
    apply_publication_style()
    fig, axes = plt.subplots(
        len(series_list), 1, sharex=True, squeeze=False,
        figsize=(FIG_WIDTH_IN, PANEL_HEIGHT_IN * len(series_list)),
        gridspec_kw={"hspace": PANEL_HSPACE},
    )
    axes = axes[:, 0]
    for ax, series, label in zip(axes, series_list, PANEL_LABELS):
        plot_variable_panel(ax, series, event_dates, label)
    if SHARE_YLIM:
        apply_shared_ylim(axes, series_list)
    start, end = get_common_time_limits(series_list)
    format_time_axis(axes[-1], start, end)
    add_legend(axes[0], event, series_list[0]["n_samples"])
    fig.suptitle(f"{catchment_label} spatial mean", x=0.5, y=0.995, fontsize=TITLE_FONTSIZE + 0.5)
    fig.align_ylabels(axes)
    # Keep fixed margins in inches whatever the number of panels.
    height = PANEL_HEIGHT_IN * len(series_list)
    fig.subplots_adjust(left=0.09, right=0.985, top=1 - 0.42 / height, bottom=0.35 / height)
    return fig


def save_and_show(fig, stem):
    """Save the figure in each requested format, then show it."""
    if WRITE_TO_FILE:
        for fmt in OUTPUT_FORMATS:
            path = f"{stem}.{fmt}"
            fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
            print("Saved figure:", path)
    plt.show()
    plt.close(fig)


def print_event_values(series_list, event_dates):
    """Print selected-member values and the ensemble range on event-window dates."""
    print("\nCatchment-mean values in the event window (selected | range)")
    for series in series_list:
        print(f"  {series['variable']}:")
        for date in event_dates.astype("datetime64[ns]"):
            selected = series["selected"][series["times"] == date]
            in_range = series["range_times"] == date
            if selected.size == 0 or not in_range.any():
                continue
            print(
                f"    {np.datetime_as_string(date, unit='D')}: {selected[0]:7.2f} | "
                f"[{series['lower'][in_range][0]:6.2f}, {series['upper'][in_range][0]:6.2f}]"
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
        for variable in PLOT_VARIABLES
    ]
    print_event_values(series_list, event_dates)
    fig = make_figure(series_list, event, event_dates, catchment["label"])
    save_and_show(fig, make_output_stem(CATCHMENT_NAME, EVENT_RANK))


if __name__ == "__main__":
    main()
