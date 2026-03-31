"""
flu_example_utils.py
====================
Shared utility functions and configuration for flu marimo notebooks
(flu_sensitivity.py and flu_scenario_analysis.py).

To configure for a different city, update SUBPOP_CONFIG and SHARED_FILES_CONFIG.
"""

from pathlib import Path
import clt_toolkit as clt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Shared (non-subpopulation-specific) input files.
AUSTIN_SHARED_FILES_CONFIG = {
    "base_path":          clt.utils.PROJECT_ROOT / "flu_instances" / "austin_input_files_2024_2025",
    "common_params_file": "common_subpop_params.json",
    "mixing_params_file": "mixing_params.json",
    "settings_file":      "simulation_settings.json",
    "humidity_file":      "absolute_humidity_austin.csv",
    "mobility_file":      "mobility_modifier.csv",
}

#: One entry per subpopulation.  Each dict must have the keys shown here.
#: The order of subpopulations should match the rows/columns of the travel_proportion
#: matrix in mixing_params.json.
#: To use a different city, replace this list (and update SHARED_FILES_CONFIG).
AUSTIN_SUBPOP_CONFIG = [
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

SHARED_FILES_CONFIG = AUSTIN_SHARED_FILES_CONFIG
SUBPOP_CONFIG = AUSTIN_SUBPOP_CONFIG

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
    scaled["date"] = _pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.date
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
    params : FluSubpopParams
        Parameters shared across all subpopulations (scenario overrides applied).
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
        schedules = flu_module.FluSubpopSchedules(
            absolute_humidity=inputs["humidity_df"],
            flu_contact_matrix=inputs["calendar_df"][name],
            daily_vaccines=per_subpop_vaccines_df[name],
            mobility_modifier=inputs["mobility_df"],
        )
        model = flu_module.FluSubpopModel(
            inputs["states"][name], params, settings, rng, schedules, name=name
        )
        subpop_models.append(model)
    return flu_module.FluMetapopModel(subpop_models, inputs["mixing_params"])


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
