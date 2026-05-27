"""
flu_example_utils.py
====================
Shared utility functions and configuration for flu marimo notebooks
(flu_sensitivity.py and flu_scenario_analysis.py).

To switch cities or region models, change only ``CURRENT_LOCATION`` near the
bottom of the configuration section.  All module-level defaults
(``SHARED_FILES_CONFIG``, ``SUBPOP_CONFIG``) are derived from that single
variable.  Adding a new location requires one new entry in
``ALL_LOCATION_CONFIGS`` plus the corresponding raw config dicts.
"""

from pathlib import Path
import copy
import json
import warnings

import numpy as np
import clt_toolkit as clt


EXAMPLES_ROOT = Path(__file__).resolve().parent
FLU_INSTANCES_ROOT = EXAMPLES_ROOT.parent
CLT_BASEMODEL_ROOT = FLU_INSTANCES_ROOT.parent
REPO_ROOT = CLT_BASEMODEL_ROOT.parent


# ---------------------------------------------------------------------------
# Shared-file config templates  (base_path is None; filled in at runtime)
# ---------------------------------------------------------------------------

AUSTIN_L2_SHARED_FILES_CONFIG = {
    "base_path":          None,
    "common_params_file": "common_subpop_params.json",
    "mixing_params_file": "mixing_params.json",
    "settings_file":      "simulation_settings.json",
    "humidity_file":      "absolute_humidity_austin.csv",
    "mobility_file":      "mobility_modifier.csv",
}

AUSTIN_L3_SHARED_FILES_CONFIG = {
    "base_path":          None,
    "common_params_file": "common_subpop_params.json",
    "mixing_params_file": "mixing_params.json",
    "settings_file":      "simulation_settings.json",
    "humidity_file":      "absolute_humidity_austin.csv",
    "mobility_file":      "mobility_modifier.csv",
}

DALLAS_L6_SHARED_FILES_CONFIG = {
    "base_path":          None,
    "common_params_file": "common_subpop_params.json",
    "mixing_params_file": "mixing_params.json",
    "settings_file":      "simulation_settings.json",
    "humidity_file":      "absolute_humidity_dallas.csv",
    "mobility_file":      "mobility_modifier.csv",
}


# ---------------------------------------------------------------------------
# Subpopulation config lists
# ---------------------------------------------------------------------------

AUSTIN_L2_SUBPOP_CONFIG = [
    {
        "name":            "east",
        "init_vals_file":  "init_vals_east.json",
        "vaccines_file":   "daily_vaccines_East.csv",
        "calendar_file":   "school_work_calendar_austin_East.csv",
    },
    {
        "name":            "west",
        "init_vals_file":  "init_vals_west.json",
        "vaccines_file":   "daily_vaccines_West.csv",
        "calendar_file":   "school_work_calendar_austin_West.csv",
    },
]

AUSTIN_L3_SUBPOP_CONFIG = [
    {
        "name":            "east",
        "init_vals_file":  "init_vals_East.json",
        "vaccines_file":   "daily_vaccines_East.csv",
        "calendar_file":   "school_work_calendar_austin_East.csv",
    },
    {
        "name":            "mid",
        "init_vals_file":  "init_vals_Mid.json",
        "vaccines_file":   "daily_vaccines_Mid.csv",
        "calendar_file":   "school_work_calendar_austin_Mid.csv",
    },
    {
        "name":            "west",
        "init_vals_file":  "init_vals_West.json",
        "vaccines_file":   "daily_vaccines_West.csv",
        "calendar_file":   "school_work_calendar_austin_West.csv",
    },
]

# Dallas files are dummy placeholders
DALLAS_L6_SUBPOP_CONFIG = [
    {
        "name":           f"Dallas_{i + 1}",
        "init_vals_file": f"init_vals_Dallas_{i + 1}.json",
        "vaccines_file":  f"daily_vaccines_Dallas_{i + 1}.csv",
        "calendar_file":  f"school_work_calendar_dallas_Dallas_{i + 1}.csv",
    }
    for i in range(6)
]
# Dallas values are dummy placeholders
_DALLAS_HIGH_RISK_FRACTIONS = {
    f"Dallas_{i + 1}": np.array([0.050, 0.116, 0.183, 0.352, 0.550], dtype=float)
    for i in range(6)
}


# ---------------------------------------------------------------------------
# Master location registry — add new cities/models here
# ---------------------------------------------------------------------------

ALL_LOCATION_CONFIGS = {
    "Austin_L2": {
        "shared":          AUSTIN_L2_SHARED_FILES_CONFIG,
        "subpops":         AUSTIN_L2_SUBPOP_CONFIG,
        "input_prefix":    "austin_input_files",
        "report_dir":      "Austin_L2",
        "param_tag":       "Austin2",
        "high_risk_fractions": {
            "east": np.array([0.048, 0.111, 0.181, 0.350, 0.552], dtype=float),
            "west": np.array([0.053, 0.121, 0.186, 0.354, 0.549], dtype=float),
        },
        "calibration_mode_dirs": {
            "normal":  "Austin_L2",
            "rescale": "Austin_L2_updated",
            "pop":     "Austin_L2_E0_pop",
        },
    },
    "Austin_L3": {
        "shared":          AUSTIN_L3_SHARED_FILES_CONFIG,
        "subpops":         AUSTIN_L3_SUBPOP_CONFIG,
        "input_prefix":    "austin3_input_files",
        "report_dir":      "Austin_L3",
        "param_tag":       "Austin3",
        "high_risk_fractions": {
            "east": np.array([0.048, 0.111, 0.181, 0.350, 0.552], dtype=float),
            "mid":  np.array([0.0505, 0.116, 0.1835, 0.352, 0.5505], dtype=float),
            "west": np.array([0.053, 0.121, 0.186, 0.354, 0.549], dtype=float),
        },
        "calibration_mode_dirs": {
            "normal":  "Austin_L3",
            "rescale": "Austin_L3_updated",
            "pop":     "Austin_L3_E0_pop",
        },
    },
    "Dallas_L6": {
        "shared":          DALLAS_L6_SHARED_FILES_CONFIG,
        "subpops":         DALLAS_L6_SUBPOP_CONFIG,
        "input_prefix":    "dallas_input_files",
        "report_dir":      "Dallas_L6",
        "param_tag":       "Dallas6",
        "high_risk_fractions": _DALLAS_HIGH_RISK_FRACTIONS,
        "calibration_mode_dirs": {
            "normal":  "Dallas_L6",
            "rescale": "Dallas_L6_updated",
            "pop":     "Dallas_L6_E0_pop",
        },
    },
}


# ---------------------------------------------------------------------------
# Active location — change only this variable to switch cities/models
# ---------------------------------------------------------------------------

CURRENT_LOCATION = "Austin_L2"

SHARED_FILES_CONFIG = {
    **ALL_LOCATION_CONFIGS[CURRENT_LOCATION]["shared"],
    "base_path": FLU_INSTANCES_ROOT / f"{ALL_LOCATION_CONFIGS[CURRENT_LOCATION]['input_prefix']}_2024_2025",
}
SUBPOP_CONFIG = ALL_LOCATION_CONFIGS[CURRENT_LOCATION]["subpops"]


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

HIGH_RISK_IHR_MULTIPLIERS = np.array([7.8, 3.2, 6.7, 7.2, 5.1], dtype=float)
IHR_MAX_PROB = 0.999
CALIBRATION_POP_SCALE = 100.0
RATE_PARAMS_TO_SCALE = [
    "E_to_I_rate",
    "IP_to_IS_rate",
    "ISR_to_R_rate",
    "IA_to_R_rate",
    "ISH_to_H_rate",
    "HR_to_R_rate",
    "HD_to_D_rate",
    "R_to_S_rate",
]


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def get_shared_files_config(location, season):
    """Return a shared-files config dict with ``base_path`` filled in for ``season``."""
    cfg = ALL_LOCATION_CONFIGS[location]
    return {
        **cfg["shared"],
        "base_path": FLU_INSTANCES_ROOT / f"{cfg['input_prefix']}_{season}",
    }


def get_subpop_config(location):
    """Return a deep copy of the subpop config list for ``location``."""
    return copy.deepcopy(ALL_LOCATION_CONFIGS[location]["subpops"])


def get_austin_subpop_config(region_model="L2"):
    return get_subpop_config(f"Austin_{region_model}")


def get_austin_shared_files_config(season, region_model="L2"):
    return get_shared_files_config(f"Austin_{region_model}", season)


def get_calibrated_austin_l_path(
    region_model,
    season,
    flatten_contact_calendar=True,
    calibration_mode="normal",
):
    loc_cfg = ALL_LOCATION_CONFIGS[f"Austin_{region_model}"]
    report_root = loc_cfg["calibration_mode_dirs"][calibration_mode]
    base_dir = (
        REPO_ROOT
        / "0_save_reports"
        / "Austin_calibration"
        / report_root
        / f"{loc_cfg['report_dir']}_{season}_cal_{'on' if flatten_contact_calendar else 'off'}"
    )
    matches = sorted(base_dir.glob(f"calibrated_params_offset*_{loc_cfg['param_tag']}_{season}.json"))
    if not matches:
        raise FileNotFoundError(f"No calibrated parameter JSON found in {base_dir}")
    return matches[0]


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_flu_inputs(subpop_config, shared_config, clt_module, flu_module, pd_module):
    """Load all flu model input files for the given subpopulation configuration.

    Parameters
    ----------
    subpop_config : list[dict]
        List of subpopulation config dicts (see SUBPOP_CONFIG).
    shared_config : dict
        Shared file config (see SHARED_FILES_CONFIG).
    clt_module, flu_module, pd_module :
        The clt_toolkit, flu_core, and pandas modules, passed in to avoid
        import-order issues in marimo cells.

    Returns
    -------
    dict with keys:
        "states"          : {name: FluSubpopState}
        "vaccines_df"     : {name: DataFrame}
        "calendar_df"     : {name: DataFrame}
        "params_baseline" : FluSubpopParams  (shared across subpopulations)
        "mixing_params"   : FluMixingParams
        "settings_base"   : SimulationSettings
        "humidity_df"     : DataFrame
        "mobility_df"     : DataFrame
    """
    base = Path(shared_config["base_path"])

    states = {
        sp["name"]: clt_module.make_dataclass_from_json(
            base / sp["init_vals_file"], flu_module.FluSubpopState
        )
        for sp in subpop_config
    }
    vaccines_df = {
        sp["name"]: pd_module.read_csv(base / sp["vaccines_file"], index_col=0)
        for sp in subpop_config
    }
    calendar_df = {
        sp["name"]: pd_module.read_csv(base / sp["calendar_file"], index_col=0)
        for sp in subpop_config
    }

    params_baseline = clt_module.make_dataclass_from_json(
        base / shared_config["common_params_file"], flu_module.FluSubpopParams
    )
    mixing_params = clt_module.make_dataclass_from_json(
        base / shared_config["mixing_params_file"], flu_module.FluMixingParams
    )
    settings_base = clt_module.make_dataclass_from_json(
        base / shared_config["settings_file"], flu_module.SimulationSettings
    )
    humidity_df = pd_module.read_csv(base / shared_config["humidity_file"], index_col=0)
    mobility_df = pd_module.read_csv(base / shared_config["mobility_file"], index_col=0)

    return {
        "states":          states,
        "vaccines_df":     vaccines_df,
        "calendar_df":     calendar_df,
        "params_baseline": params_baseline,
        "mixing_params":   mixing_params,
        "settings_base":   settings_base,
        "humidity_df":     humidity_df,
        "mobility_df":     mobility_df,
    }


def convert_1risk_to_2risk(data_1risk, low_risk_frac=None, high_risk_frac=None):
    """Convert a trailing singleton risk dimension into two risk groups."""
    data_array = np.asarray(data_1risk, dtype=float)

    shape_2risk = list(data_array.shape)
    shape_2risk[-1] = 2
    data_2risk = np.zeros(shape_2risk, dtype=float)

    if high_risk_frac is not None:
        high_frac = np.asarray(high_risk_frac, dtype=float)
        low_frac = 1.0 - high_frac
    elif low_risk_frac is not None:
        low_frac = np.asarray(low_risk_frac, dtype=float)
        high_frac = 1.0 - low_frac
    else:
        low_frac = 0.7
        high_frac = 0.3

    data_2risk[..., 0] = data_array[..., 0] * low_frac
    data_2risk[..., 1] = data_array[..., 0] * high_frac
    return data_2risk


def convert_1risk_to_2risk2(data_1risk, low_risk_frac=None, high_risk_frac=None):
    """Convert a trailing singleton risk dimension into two risk groups."""
    data_array = np.asarray(data_1risk, dtype=float)

    shape_2risk = list(data_array.shape)
    shape_2risk[-1] = 2
    data_2risk = np.zeros(shape_2risk, dtype=float)

    if high_risk_frac is not None:
        high_frac = np.asarray(high_risk_frac, dtype=float)
        low_frac = 1.0 - high_frac
    elif low_risk_frac is not None:
        low_frac = np.asarray(low_risk_frac, dtype=float)
        high_frac = 1.0 - low_frac
    else:
        low_frac = 0.7
        high_frac = 0.3

    data_2risk[..., 0] = data_array[..., 0]
    data_2risk[..., 1] = data_array[..., 0]
    return data_2risk


def _convert_schedule_df_to_2risk(df, value_col, high_risk_frac=None, low_risk_frac=None):
    """Convert JSON-encoded age-by-risk schedule entries from 1 to 2 risk groups."""
    import pandas as _pd

    if len(df) == 0:
        return df.copy()

    first_val = np.asarray(json.loads(df.iloc[0][value_col]), dtype=float)
    if first_val.ndim == 2 and first_val.shape[1] == 2:
        converted = df.copy()
        if "date" in converted.columns:
            converted["date"] = _pd.to_datetime(converted["date"]).dt.strftime("%Y-%m-%d")
        return converted.set_index(_pd.RangeIndex(len(converted)))

    rows = []
    for _, row in df.iterrows():
        val_1risk = json.loads(row[value_col])
        val_2risk = convert_1risk_to_2risk2(
            val_1risk,
            high_risk_frac=high_risk_frac,
            low_risk_frac=low_risk_frac,
        )
        new_row = row.to_dict()
        new_row[value_col] = json.dumps(val_2risk.tolist())
        rows.append(new_row)

    converted = _pd.DataFrame(rows)
    if "date" in converted.columns:
        converted["date"] = _pd.to_datetime(converted["date"]).dt.strftime("%Y-%m-%d")
    return converted.set_index(_pd.RangeIndex(len(converted)))


def _align_vaccine_schedule_to_contact_start(vax_df, contact_start):
    """Trim any leading vaccine rows before the contact-calendar start date."""
    import pandas as _pd

    aligned = vax_df.copy()
    if "date" not in aligned.columns:
        return aligned

    _vax_dates = _pd.to_datetime(aligned["date"])
    _contact_start = _pd.to_datetime(contact_start)
    aligned = aligned.loc[_vax_dates >= _contact_start].reset_index(drop=True)
    if "date" in aligned.columns:
        aligned["date"] = _pd.to_datetime(aligned["date"]).dt.strftime("%Y-%m-%d")
    return aligned


def _expand_mobility_to_calendar_dates(mobility_df, calendar_dates):
    """Expand day-of-week mobility rows to one row per calendar date."""
    import pandas as _pd

    if "day_of_week" not in mobility_df.columns:
        expanded = mobility_df.copy()
        if "date" in expanded.columns:
            expanded["date"] = _pd.to_datetime(expanded["date"]).dt.strftime("%Y-%m-%d")
        return expanded.reset_index(drop=True)

    dates = _pd.to_datetime(calendar_dates)
    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    rows = []
    for date in dates:
        day_name = day_names[date.dayofweek]
        day_row = mobility_df.loc[mobility_df["day_of_week"] == day_name]
        if len(day_row) > 0:
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "mobility_modifier": day_row["mobility_modifier"].iloc[0],
                }
            )
        else:
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "mobility_modifier": json.dumps([[1.0], [1.0], [1.0], [1.0], [1.0]]),
                }
            )

    return _pd.DataFrame(rows).set_index(_pd.RangeIndex(len(rows)))


def _expand_age_risk_params_to_2risk(params):
    """Duplicate age-risk arrays that are still stored as (A, 1)."""
    age_risk_params = [
        "vax_induced_immune_wane",
        "vax_induced_inf_risk_reduce",
        "vax_induced_hosp_risk_reduce",
        "HR_to_R_rate",
        "HD_to_D_rate",
        "E_to_IA_prop",
        "relative_suscept",
        "ISH_to_HD_prop",
    ]

    updated = params
    for param_name in age_risk_params:
        param_val = getattr(updated, param_name, None)
        if param_val is None:
            continue
        param_arr = np.asarray(param_val, dtype=float)
        if param_arr.ndim == 2 and param_arr.shape[1] == 1:
            updated = clt.updated_dataclass(
                updated,
                {param_name: convert_1risk_to_2risk2(param_arr, low_risk_frac=1.0)},
            )
    return updated


def load_calibrated_austin_inputs(
    clt_module,
    flu_module,
    pd_module,
    calibration_path=None,
    flatten_contact_calendar=True,
    season="2024_2025",
    region_model="L2",
    calibration_mode="normal",
):
    """Load Austin calibrated inputs with calibrated beta/E0/IHR and 2 risk groups."""
    global CURRENT_LOCATION, SHARED_FILES_CONFIG, SUBPOP_CONFIG

    location = f"Austin_{region_model}"
    CURRENT_LOCATION = location
    SHARED_FILES_CONFIG = get_shared_files_config(location, season)
    SUBPOP_CONFIG = get_subpop_config(location)
    high_risk_fractions = ALL_LOCATION_CONFIGS[location]["high_risk_fractions"]

    shared_config = SHARED_FILES_CONFIG
    inputs = load_flu_inputs(SUBPOP_CONFIG, shared_config, clt_module, flu_module, pd_module)

    calibration_file = Path(
        calibration_path
        or get_calibrated_austin_l_path(
            region_model,
            season,
            flatten_contact_calendar,
            calibration_mode=calibration_mode,
        )
    )
    calibrated = json.loads(calibration_file.read_text())

    calendar_df = {}
    for sp in SUBPOP_CONFIG:
        name = sp["name"]
        df = inputs["calendar_df"][name].copy()
        if flatten_contact_calendar:
            for col in ["is_school_day", "is_work_day"]:
                if col in df.columns:
                    df[col] = df[col].mean()
        if "date" in df.columns:
            df["date"] = pd_module.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        calendar_df[name] = df

    vaccines_df = {}
    contact_start = calendar_df[SUBPOP_CONFIG[0]["name"]]["date"].iloc[0]
    for sp in SUBPOP_CONFIG:
        name = sp["name"]
        aligned_vax = _align_vaccine_schedule_to_contact_start(inputs["vaccines_df"][name], contact_start)
        vaccines_df[name] = _convert_schedule_df_to_2risk(
            aligned_vax,
            "daily_vaccines",
            high_risk_frac=high_risk_fractions[name],
        )

    mobility_expanded = _expand_mobility_to_calendar_dates(
        inputs["mobility_df"],
        calendar_df[SUBPOP_CONFIG[0]["name"]]["date"],
    )
    mobility_df = _convert_schedule_df_to_2risk(
        mobility_expanded,
        "mobility_modifier",
        low_risk_frac=1.0,
    )

    params_base = clt_module.updated_dataclass(inputs["params_baseline"], {"num_risk_groups": 2})
    params_base = _expand_age_risk_params_to_2risk(params_base)
    time_stretch = float(calibrated.get("time_stretch", 1.0))
    if time_stretch != 1.0:
        rate_updates = {}
        for rate_name in RATE_PARAMS_TO_SCALE:
            orig = getattr(params_base, rate_name)
            orig_arr = np.asarray(orig, dtype=float)
            rate_updates[rate_name] = orig_arr / time_stretch
        params_base = clt_module.updated_dataclass(params_base, rate_updates)

    params_by_subpop = {}
    states = {}
    for idx, sp in enumerate(SUBPOP_CONFIG):
        name = sp["name"]

        ihr_low = np.asarray(calibrated["IHR_low"][idx], dtype=float)
        ihr_high = np.asarray(calibrated["IHR_high"][idx], dtype=float)
        max_low = IHR_MAX_PROB / HIGH_RISK_IHR_MULTIPLIERS
        if np.any(ihr_high >= IHR_MAX_PROB) or np.any(ihr_low > max_low):
            warnings.warn(
                f"Calibrated IHR for {name} exceeds valid probability bounds; "
                "clipping low-risk IHR so derived high-risk IHR stays below 1.0."
            )
            ihr_low = np.minimum(ihr_low, max_low)
            ihr_high = ihr_low * HIGH_RISK_IHR_MULTIPLIERS
        beta = float(calibrated["beta"][idx])
        params_by_subpop[name] = clt_module.updated_dataclass(
            params_base,
            {
                "beta_baseline": beta,
                "IP_to_ISH_prop": np.column_stack([ihr_low, ihr_high]),
            },
        )

        state = copy.deepcopy(inputs["states"][name])
        high_risk_frac = high_risk_fractions[name]
        state_s_arr = np.asarray(state.S, dtype=float)
        if state_s_arr.ndim == 2 and state_s_arr.shape[1] == 1:
            for comp_name in ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D", "M", "MV"]:
                comp_val = getattr(state, comp_name)
                setattr(
                    state,
                    comp_name,
                    convert_1risk_to_2risk(comp_val, high_risk_frac=high_risk_frac),
                )

        e0_raw = np.asarray(calibrated["E0"][idx], dtype=float)
        s_arr = np.asarray(state.S, dtype=float)
        if e0_raw.ndim == 1:
            e0_2risk = np.zeros_like(s_arr, dtype=float)
            e0_2risk[:, 0] = e0_raw * (1.0 - high_risk_frac)
            e0_2risk[:, 1] = e0_raw * high_risk_frac
        elif e0_raw.ndim == 2 and e0_raw.shape[1] == 2:
            e0_2risk = e0_raw
        else:
            raise ValueError(
                f"Unsupported calibrated E0 shape for {name}: {e0_raw.shape}. "
                "Expected (age,) or (age, 2)."
            )
        state.S = s_arr - e0_2risk + np.asarray(state.E, dtype=float)
        state.E = e0_2risk
        states[name] = state

    return {
        **inputs,
        "states": states,
        "vaccines_df": vaccines_df,
        "calendar_df": calendar_df,
        "mobility_df": mobility_df,
        "params_baseline": params_by_subpop[SUBPOP_CONFIG[0]["name"]],
        "params_by_subpop": params_by_subpop,
        "calibrated_params_file": calibration_file,
        "calibrated_params": calibrated,
        "calibrated_offset": int(calibrated["offset"]),
        "pop_scale": float(calibrated.get("pop_scale", CALIBRATION_POP_SCALE)),
        "season": season,
        "region_model": region_model,
        "location": location,
        "calibration_mode": calibration_mode,
        "subpop_config": SUBPOP_CONFIG,
        "shared_config": shared_config,
    }


def load_calibrated_austin_l2_2024_2025_inputs(
    clt_module,
    flu_module,
    pd_module,
    calibration_path=None,
    flatten_contact_calendar=True,
    season="2024_2025",
    calibration_mode="normal",
):
    """Backward-compatible wrapper for the existing 2-region Austin loader."""
    return load_calibrated_austin_inputs(
        clt_module,
        flu_module,
        pd_module,
        calibration_path=calibration_path,
        flatten_contact_calendar=flatten_contact_calendar,
        season=season,
        region_model="L2",
        calibration_mode=calibration_mode,
    )


# ---------------------------------------------------------------------------
# Vaccine schedule utilities
# ---------------------------------------------------------------------------

def scale_vaccines_df(df, scale, np_module):
    """Return a copy of a daily_vaccines DataFrame with doses scaled by ``scale``.

    ``scale`` may be a scalar float or a (n_age, n_risk) numpy array.
    The ``daily_vaccines`` column contains JSON-encoded 2-D arrays, so this
    function round-trips through JSON to apply the scaling correctly.
    """
    import json
    import pandas as _pd
    scaled = df.copy()
    scaled["daily_vaccines"] = scaled["daily_vaccines"].apply(
        lambda s: json.dumps((np_module.array(json.loads(s)) * scale).tolist())
    )
    scaled["date"] = _pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    return scaled


def compute_cumulative_vax(df, scale, np_module):
    """Compute rolling annual cumulative vaccination rates.

    Returns an (n_age, n_risk) array of cumulative rates over the most recent
    year of data in ``df``, with daily doses multiplied by ``scale``.
    """
    import json
    daily_arrs = np_module.stack(
        [np_module.array(json.loads(s)) * scale for s in df["daily_vaccines"]]
    )  # (n_days, n_age, n_risk)
    window_size = min(365, len(df))
    windows = np_module.lib.stride_tricks.sliding_window_view(
        daily_arrs, window_size, axis=0
    )  # (n_windows, n_age, n_risk, window_size)
    return np_module.sum(windows, axis=-1)[-1]  # (n_age, n_risk)


def make_cumvax_markdown_table(arr, title):
    """Format a (n_age, n_risk) cumulative-vax array as a Markdown table string."""
    n_age, n_risk = arr.shape
    header = "| Age group | " + " | ".join(f"Risk {rg}" for rg in range(n_risk)) + " |"
    sep = "|-----------|" + "--------|" * n_risk
    rows = [
        f"| {ag} | " + " | ".join(f"{arr[ag, rg]:.4f}" for rg in range(n_risk)) + " |"
        for ag in range(n_age)
    ]
    return f"**{title}**\n\n" + "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# RNG construction
# ---------------------------------------------------------------------------

def make_rng_generators(seed, subpop_config, np_module):
    """Return one ``np.random.Generator`` per subpopulation.

    Uses ``MT19937(seed).jumped(i)`` for subpop index ``i``, which preserves
    the same statistical independence guarantee as the original two-subpop code
    (``bit_gen`` for east, ``bit_gen.jumped(1)`` for west).
    """
    bit_gen = np_module.random.MT19937(seed)
    return [np_module.random.Generator(bit_gen.jumped(i)) for i in range(len(subpop_config))]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_flu_metapop_model(subpop_config, inputs, params, settings,
                             rng_list, per_subpop_vaccines_df, flu_module):
    """Build a ``FluMetapopModel`` from config and loaded inputs.

    Parameters
    ----------
    subpop_config : list[dict]
        Each dict must have a ``"name"`` key.
    inputs : dict
        As returned by :func:`load_flu_inputs`.
    params : FluSubpopParams or dict[str, FluSubpopParams]
        Parameters shared across subpopulations, or a per-subpop dict.
    settings : SimulationSettings
        Simulation settings (scenario overrides applied).
    rng_list : list[np.random.Generator]
        One generator per subpopulation, in the same order as ``subpop_config``.
    per_subpop_vaccines_df : dict[str, DataFrame]
        ``{name: vaccines_df}`` — allows scenario-specific vaccine schedule
        overrides without mutating the base ``inputs`` dict.
    flu_module : module
        The ``flu_core`` module.
    """
    subpop_models = []
    for sp_cfg, rng in zip(subpop_config, rng_list):
        name = sp_cfg["name"]
        subpop_params = params[name] if isinstance(params, dict) else params
        schedules = flu_module.FluSubpopSchedules(
            absolute_humidity=inputs["humidity_df"],
            flu_contact_matrix=inputs["calendar_df"][name],
            daily_vaccines=per_subpop_vaccines_df[name],
            mobility_modifier=inputs["mobility_df"],
        )
        model = flu_module.FluSubpopModel(
            inputs["states"][name], subpop_params, settings, rng, schedules, name=name
        )
        subpop_models.append(model)
    return flu_module.FluMetapopModel(subpop_models, inputs["mixing_params"])


def apply_general_init_overrides(states, overrides, subpop_config, np_module):
    """Apply initial condition overrides to states.

    Keys in ``overrides`` follow two patterns:

    - ``"init:{sp_name}:{comp}:{i}:{j}"`` → set ``state.{comp}[i][j] = value``
    - ``"M:{sp_name}"`` or ``"M:all"`` → multiply ``state.M`` element-wise by
      ``value`` (treated as a scale factor)

    Returns a new dict of deepcopied, modified states for all subpopulations.
    """
    import copy
    modified = {sp["name"]: copy.deepcopy(states[sp["name"]]) for sp in subpop_config}

    for key, val in overrides.items():
        parts = key.split(":")
        if parts[0] == "init":
            _, sp_name, comp, i_str, j_str = parts
            getattr(modified[sp_name], comp)[int(i_str)][int(j_str)] = val
        elif parts[0] == "M":
            sp_target = parts[1]
            scale = float(val)
            for sp in subpop_config:
                if sp_target == "all" or sp["name"] == sp_target:
                    s = modified[sp["name"]]
                    s.M = np_module.asarray(s.M) * scale

    return modified


def apply_init_overrides(states, overrides, subpop_config):
    """Return a new states dict with E[2][0] values replaced per ``overrides``.

    Keys in ``overrides`` must follow the pattern ``"{subpop_name}_E_2_0"``.
    States for subpopulations not present in ``overrides`` are returned as-is
    (no deep copy).
    """
    import copy
    new_states = {}
    for sp in subpop_config:
        name = sp["name"]
        key = f"{name}_E_2_0"
        if key in overrides:
            s = copy.deepcopy(states[name])
            s.E[2][0] = overrides[key]
            new_states[name] = s
        else:
            new_states[name] = states[name]
    return new_states


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def subpop_dropdown_options(subpop_config, aggregate_label="combined"):
    """Return ``[aggregate_label, name1, name2, ...]`` for marimo dropdowns.

    The first element is the aggregate label (e.g. ``"combined"`` for
    flu_sensitivity.py or ``"all subpops"`` for flu_scenario_analysis.py).
    """
    return [aggregate_label] + [sp["name"] for sp in subpop_config]


def load_austin_observed_hosp(pd_module, season="2024_2025", region_model="L2"):
    """Load observed daily hospital admissions for the selected Austin region model/season.

    Returns a dict with:
      - ``dates``: pandas.DatetimeIndex
      - ``values``: ndarray shaped ``(T, n_regions, 5)``
      - ``subpop_names``: list of lowercase region names
    """
    season_start_year, season_end_year = season.split("_")
    start_date = pd_module.Timestamp(f"{season_start_year}-08-01")
    end_date = pd_module.Timestamp(f"{season_end_year}-05-31")

    if region_model == "L2":
        region_files = {
            "east": REPO_ROOT / "InputFiles" / "HospitalFiles" / "Hosp_Austin2_East.csv",
            "west": REPO_ROOT / "InputFiles" / "HospitalFiles" / "Hosp_Austin2_West.csv",
        }
    elif region_model == "L3":
        region_files = {
            "east": REPO_ROOT / "InputFiles" / "HospitalFiles" / "Hosp_Austin3_East.csv",
            "mid":  REPO_ROOT / "InputFiles" / "HospitalFiles" / "Hosp_Austin3_Mid.csv",
            "west": REPO_ROOT / "InputFiles" / "HospitalFiles" / "Hosp_Austin3_West.csv",
        }
    else:
        raise ValueError(f"Unknown region_model: {region_model}")

    age_cols = ["0-4", "5-17", "18-49", "50-64", "65plus"]
    dates = pd_module.date_range(start_date, end_date, freq="D")
    region_arrays = []
    for _, path in region_files.items():
        df = pd_module.read_csv(path)
        df["Dates"] = pd_module.to_datetime(df["Dates"])
        df = df[(df["Dates"] >= start_date) & (df["Dates"] <= end_date)].copy()
        df = df.set_index("Dates").reindex(dates)
        df[age_cols] = df[age_cols].fillna(0.0)
        region_arrays.append(df[age_cols].to_numpy(dtype=float))

    return {
        "dates": dates,
        "values": np.stack(region_arrays, axis=1),
        "subpop_names": list(region_files.keys()),
    }
