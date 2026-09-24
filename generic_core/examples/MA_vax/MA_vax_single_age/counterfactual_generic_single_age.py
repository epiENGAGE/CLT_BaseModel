"""Single-age-group adaptation of `MA_vax/counterfactual_generic.py`: builds
and runs the generic_core-exported MA_vax_single_age model directly (against
`model_config_MA_single_age.json`/`fitted_params_MA_single_age.json`/
`schedules.json` in this folder), for the "always run fresh" sections of
`counterfactual_notebook_generic_single_age.py` -- vaccination coverage,
matched-cohort attack-probability curves, the baseline fit check, and the
interactive epi-curves-by-scenario explorer. Those sections need a handful of
cheap, deterministic simulations at notebook-render time, not the
already-computed S.A.* tables (those come from
`build_counterfactual_tables_from_db_single_age.py`'s CSVs instead, loaded via
`load_saved_tables` re-exported below).

Trimmed relative to the multi-age `counterfactual_generic.py`:

  * No `single_age_only_scenario` / per-age-group scenario entries in
    `named_scenarios()` -- with one age group they have no single-age
    equivalent, same reason `run_simulations_MA_vax_single_age.py`'s
    `SCENARIOS` dict and `build_counterfactual_tables_from_db_single_age.py`
    omit them (S.A.2/S.A.3 are not built there either).
  * No `table_S_A_*`/`save_all_tables` -- this module only backs the
    notebook's live-simulation cells; the CSV-table pipeline for this model
    is `build_counterfactual_tables_from_db_single_age.py`, which reads an
    already-run results source instead of simulating.
  * `ve_scenarios()` returns the absolute `vax_susceptibility`/`IV_to_H_prop`
    overrides copied from `run_simulations_MA_vax_single_age.py`'s
    `SCENARIOS` dict (that script's source of truth for the single-age-refit
    VE presets), rather than ratios applied to the fitted baseline.

`AGE_GROUPS`/`NUM_AGE_GROUPS` are read from `model_config_MA_single_age.json`
itself (currently one group, `"0+"`) rather than hardcoded, so nothing here
assumes a specific count.

Location: this file must sit next to model_config_MA_single_age.json /
fitted_params_MA_single_age.json / schedules.json, three directory levels
below the repo root that contains generic_core/, clt_toolkit/ and flu_core/
-- same constraint as run_simulations_MA_vax_single_age.py in this folder.

Single-population only (IS_METAPOP=False), matching the exported model.
"""

from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))  # repo root
sys.path.insert(0, str(_HERE.parent))  # MA_vax/, for ma_vax_shared
sys.path.insert(0, str(_HERE))

import clt_toolkit as clt
import flu_core as flu
from generic_core.config_parser import parse_model_config_from_dict
from generic_core.generic_model import (
    ConfigDrivenSubpopModel, build_state_from_config, build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel
from generic_core.model_factory import (
    build_compartment_init, scale_dose_schedule_df, load_schedule_csv_texts,
)
from generic_core.fitting import (
    _scale_compartment_init, _inject_tv_transmission, _tv_knot_days,
    build_transmission_multiplier_array, prepare_param_sets,
)

from ma_vax_shared import (
    COMPARTMENTS, TRANSITIONS,
    attack_probability_curves, load_saved_tables,
)

# ---- Configurable (mirrors run_simulations_MA_vax_single_age.py) ----
MODEL_CONFIG_FILE = "model_config_MA_single_age.json"
FITTED_PARAMS_FILE = "fitted_params_MA_single_age.json"
SCHEDULES_FILE = "schedules.json"
NUM_DAYS = 250
TIMESTEPS_PER_DAY = 7
START_DATE = "2025-09-01"
NUM_RISK_GROUPS = 1
SEED_BASE = 42

with open(_HERE / MODEL_CONFIG_FILE) as _f:
    _base_model_config = json.load(_f)
AGE_GROUPS: list[str] = list((_base_model_config.get("age_risk", {}) or {}).get("age_groups") or ["0+"])
NUM_AGE_GROUPS = len(AGE_GROUPS)

# Absolute vax_susceptibility/IV_to_H_prop overrides for the VE presets,
# copied from run_simulations_MA_vax_single_age.py's SCENARIOS dict (source
# of truth -- keep in sync if that script is re-fit or the presets change).
# See build_counterfactual_tables_from_db_single_age.py's identical constant.
VE_PRESET_OVERRIDES = {
    "Low VE": {"vax_susceptibility": 0.8976501514033461, "IV_to_H_prop": 0.005687597232161169},
    "High VE": {"vax_susceptibility": 0.6616378415241131, "IV_to_H_prop": 0.0049138373017620935},
}


# ── Base inputs ───────────────────────────────────────────────────────────────

def _split_pset(pset):
    model_p = {
        k: v for k, v in pset.items()
        if k != "phi" and not k.startswith("m_dlog_") and not k.startswith("seed_scale_")
    }
    scales = {
        k[len("seed_scale_"):]: float(v)
        for k, v in pset.items() if k.startswith("seed_scale_")
    }
    incr = [
        v for _, v in sorted(
            (
                (int(k[len("m_dlog_"):]), float(v))
                for k, v in pset.items()
                if k.startswith("m_dlog_") and k[len("m_dlog_"):].isdigit()
            ),
            key=lambda t: t[0],
        )
    ]
    return model_p, scales, incr


def load_base_inputs(model_config_file: str = MODEL_CONFIG_FILE,
                      fitted_params_file: str | None = FITTED_PARAMS_FILE) -> dict:
    """Build the base inputs dict: fitted params (best point estimate) merged
    into `config_dict["params"]`, plus seed scales / m(t) log-increments /
    per-age population needed by `build_model`."""
    with open(_HERE / model_config_file) as f:
        config_dict = json.load(f)

    seed_scales, tv_increments = {}, []
    tv_spacing, fit_num_days = 30, 0
    if fitted_params_file is not None and (_HERE / fitted_params_file).exists():
        with open(_HERE / fitted_params_file) as f:
            fitted_raw = json.load(f)
        best_params = fitted_raw.get("best_params", fitted_raw) if isinstance(fitted_raw, dict) else {}
        scale_groups = (fitted_raw.get("scale_groups", {}) or {}) if isinstance(fitted_raw, dict) else {}
        fit_num_days = int(fitted_raw.get("num_days", 0) or 0) if isinstance(fitted_raw, dict) else 0
        tv_spacing = int(fitted_raw.get("tv_knot_spacing_days", 30) or 30) if isinstance(fitted_raw, dict) else 30

        orig_params = dict(config_dict.get("params", {}) or {})
        expanded = prepare_param_sets([best_params], scale_groups, orig_params)[0]
        fitted_params, seed_scales, tv_increments = _split_pset(expanded)
        if tv_increments and fit_num_days <= 0:
            tv_increments = []
        config_dict["params"] = {**config_dict.get("params", {}), **fitted_params}

    ic_entry = (config_dict.get("initial_conditions", {}) or {}).get("aggregate_pop", {})
    population = np.asarray(
        ic_entry.get("population", [[0.0]] * NUM_AGE_GROUPS), dtype=float
    ).reshape(NUM_AGE_GROUPS)

    return {
        "config_dict": config_dict,
        "seed_scales": seed_scales,
        "tv_increments": tv_increments,
        "tv_spacing": tv_spacing,
        "fit_num_days": fit_num_days,
        "population": population,
    }


# ── Model building ───────────────────────────────────────────────────────────

def _build_tvm_df(base_inputs, num_days, start_date=START_DATE):
    incr = base_inputs["tv_increments"]
    fit_num_days = base_inputs["fit_num_days"]
    if not incr or fit_num_days <= 0:
        return None
    h = max(num_days + 14, 370)
    dates = pd.date_range(start=start_date, periods=h, freq="D").date
    knots = _tv_knot_days(fit_num_days, base_inputs["tv_spacing"])
    m_fit = build_transmission_multiplier_array(incr, knots, fit_num_days)
    if h <= fit_num_days:
        m_full = m_fit[:h]
    else:
        m_full = np.concatenate([m_fit, np.full(h - fit_num_days, m_fit[-1])])
    return pd.DataFrame({"date": dates, "transmission_multiplier": m_full})


_SCHEDULE_CSVS_CACHE = {}


def _load_schedule_csvs(schedules_file: str = SCHEDULES_FILE,
                        model_config_file: str = MODEL_CONFIG_FILE):
    """{df_attribute: csv_text} of the base schedules. The CSVs named in the
    model config's input_files are read fresh from disk; `schedules_file` (the
    notebook's exported snapshot) only fills in files that can't be found (see
    generic_core.model_factory.load_schedule_csv_texts). Cached per process."""
    key = (schedules_file, model_config_file)
    if key not in _SCHEDULE_CSVS_CACHE:
        with open(_HERE / model_config_file) as f:
            config = json.load(f)
        _SCHEDULE_CSVS_CACHE[key] = load_schedule_csv_texts(
            config,
            snapshot_path=(_HERE / schedules_file) if schedules_file else None,
            search_roots=[Path.cwd(), _HERE, *_HERE.parents],
        )
    return _SCHEDULE_CSVS_CACHE[key]


def _build_schedules(base_inputs, start_date, num_days, dose_mult=None):
    h = max(num_days + 14, 370)
    dates = pd.date_range(start=start_date, periods=h, freq="D").date
    mob = json.dumps(np.ones((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    vax = json.dumps(np.zeros((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    csvs = _load_schedule_csvs()

    def real_or(name, fallback):
        if name in csvs:
            return pd.read_csv(io.StringIO(csvs[name]))
        return fallback

    kwargs = {}
    tvm_df = _build_tvm_df(base_inputs, num_days, start_date)
    if tvm_df is not None:
        kwargs["transmission_multiplier_df"] = tvm_df
    return SimpleNamespace(
        absolute_humidity_df=real_or(
            "absolute_humidity_df",
            pd.DataFrame({"date": dates, "absolute_humidity": [0.01] * h})),
        school_work_calendar_df=real_or(
            "school_work_calendar_df",
            pd.DataFrame({"date": dates, "is_school_day": [1.0] * h, "is_work_day": [1.0] * h})),
        mobility_df=real_or(
            "mobility_df",
            pd.DataFrame({"day_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday",
                                           "saturday", "sunday"], "mobility_modifier": [mob] * 7})),
        daily_vaccines_df=scale_dose_schedule_df(real_or(
            "daily_vaccines_df",
            pd.DataFrame({"date": dates, "daily_vaccines": [vax] * h})), dose_mult),
        **kwargs,
    )


def build_model(base_inputs: dict, param_overrides: dict | None = None,
                 dose_mult: list | None = None, rng_seed: int = 0,
                 stochastic: bool = False) -> ConfigDrivenMetapopModel:
    """Build (but do not run) a single-population model for `base_inputs`
    with `param_overrides` merged in and `dose_mult` (per-age-group
    multiplier on the vaccine-uptake schedule) applied."""
    seed_scales = base_inputs["seed_scales"]
    tv_incr = base_inputs["tv_increments"]
    cfg = copy.deepcopy(base_inputs["config_dict"])
    if param_overrides:
        cfg["params"] = {**cfg.get("params", {}), **param_overrides}
    if tv_incr:
        cfg, _ = _inject_tv_transmission(cfg)

    sched = _build_schedules(base_inputs, START_DATE, NUM_DAYS, dose_mult)
    mc = parse_model_config_from_dict(cfg, schedules_input=sched)
    A, R = NUM_AGE_GROUPS, NUM_RISK_GROUPS
    comps = list(cfg.get("compartments", {}).keys()) if isinstance(cfg.get("compartments"), dict) else list(cfg.get("compartments", ["S"]))
    first = comps[0] if comps else "S"
    N = cfg.get("total_population", 100000)
    ic_entry = (cfg.get("initial_conditions", {}) or {}).get("aggregate_pop", {})
    if ic_entry:
        pop_arr = np.asarray(ic_entry.get("population", np.full((A, R), float(N))), dtype=float)
        seed_arrays = {
            c: np.asarray(a, dtype=float)
            for c, a in (ic_entry.get("seeds", {}) or {}).items()
            if c in comps
        }
        comp_init, _ = build_compartment_init(seed_arrays, pop_arr, comps)
    else:
        comp_init = {first: np.full((A, R), float(N))}
        for c in comps[1:]:
            comp_init.setdefault(c, np.zeros((A, R)))
    if seed_scales:
        comp_init = _scale_compartment_init(comp_init, seed_scales, comps, A, R)
    state = build_state_from_config(mc, comp_init, epi_metric_init={})
    params = build_params_from_config(mc, num_age_groups=A, num_risk_groups=R)
    tt = clt.TransitionTypes.BINOM if stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    settings = clt.SimulationSettings(
        timesteps_per_day=TIMESTEPS_PER_DAY, transition_type=tt,
        start_real_date=START_DATE, save_daily_history=True,
        transition_variables_to_save=TRANSITIONS,
    )
    subpop = ConfigDrivenSubpopModel(
        model_config=mc, state_init=state, params=params,
        simulation_settings=settings, RNG=np.random.default_rng(rng_seed),
        schedules_input=sched, name="pop",
    )
    mixing = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
    return ConfigDrivenMetapopModel(
        subpop_models=[subpop], mixing_params=mixing, model_config=mc, travel_config={},
    )


def _extract_age_arrays(subpop) -> dict[str, np.ndarray]:
    """(day, age_group) arrays for every compartment/transition in
    COMPARTMENTS/TRANSITIONS -- the per-age-group analog of
    `generic_core.model_factory.extract_history`, which sums the age axis
    away. Transition-variable history is saved once per sub-timestep (see
    that function's docstring); aggregated to daily here before returning."""
    out = {}
    for c in COMPARTMENTS:
        arr = np.array(subpop.compartments[c].history_vals_list)  # (T, A, R)
        out[c] = arr.sum(axis=2)
    ts = int(getattr(subpop.simulation_settings, "timesteps_per_day", 1) or 1)
    for t in TRANSITIONS:
        raw = np.array(subpop.transition_variables[t].history_vals_list)  # (T*ts, A, R)
        T = raw.shape[0]
        if ts > 1 and T > 0 and T % ts == 0:
            raw = raw.reshape(T // ts, ts, *raw.shape[1:]).sum(axis=1)
        out[t] = raw.sum(axis=2)
    return out


# ── Scenario builders ────────────────────────────────────────────────────────
# Each scenario is {"param_overrides": {...}, "dose_mult": [...]}; either key
# may be omitted (falls back to the fitted baseline / unscaled schedule). No
# per-age-group scenario builder here -- see module docstring.

def no_vaccine_scenario() -> dict:
    return {"dose_mult": [0.0] * NUM_AGE_GROUPS}


def baseline_scenario() -> dict:
    return {}


def infection_protection_only_scenario(base_inputs: dict) -> dict:
    """VE against infection only: IV_to_H_prop set equal to I_to_H_prop, so
    vaccination confers no hospitalization-risk reduction conditional on
    infection."""
    ih = base_inputs["config_dict"]["params"]["I_to_H_prop"]
    return {"param_overrides": {"IV_to_H_prop": copy.deepcopy(ih)}}


def ve_scenarios(base_inputs: dict) -> dict[str, dict]:
    """VE sensitivity presets. Unlike the multi-age version's
    `ve_scale_scenario` (a ratio on the fitted baseline), these are the
    absolute `VE_PRESET_OVERRIDES` values copied from
    run_simulations_MA_vax_single_age.py's SCENARIOS dict -- that script's
    source of truth for the single-age-refit presets."""
    return {
        "low_ve": {"param_overrides": {
            "vax_susceptibility": [[VE_PRESET_OVERRIDES["Low VE"]["vax_susceptibility"]]],
            "IV_to_H_prop": [[VE_PRESET_OVERRIDES["Low VE"]["IV_to_H_prop"]]],
        }},
        "baseline_ve": {},
        "high_ve": {"param_overrides": {
            "vax_susceptibility": [[VE_PRESET_OVERRIDES["High VE"]["vax_susceptibility"]]],
            "IV_to_H_prop": [[VE_PRESET_OVERRIDES["High VE"]["IV_to_H_prop"]]],
        }},
    }


def scheduled_coverage(age_idx: int | None = None) -> np.ndarray | float:
    """Cumulative coverage the BASELINE dose schedule asks for over the
    simulation window -- the sum of the daily vaccination proportions in
    `daily_vaccines_df`, before the model's per-step S-cap (§1.4) decides how
    many of those doses actually land. See
    `counterfactual_generic.scheduled_coverage`'s docstring for the full
    rationale (identical here, just over `AGE_GROUPS` as this model config
    defines it)."""
    csvs = _load_schedule_csvs()
    if "daily_vaccines_df" not in csvs:
        raise ValueError("no daily_vaccines_df schedule to derive coverage from")
    df = pd.read_csv(io.StringIO(csvs["daily_vaccines_df"]))
    dates = pd.to_datetime(df["date"], format="mixed")
    start = pd.Timestamp(START_DATE)
    window = (dates >= start) & (dates < start + pd.Timedelta(days=NUM_DAYS))
    daily = np.array([np.asarray(json.loads(v), dtype=float).ravel()
                       for v in df.loc[window, "daily_vaccines"]])
    cov = daily.sum(axis=0)
    return cov if age_idx is None else float(cov[age_idx])


def scheduled_doses(population, age_idx: int | None = None):
    cov = scheduled_coverage()
    doses = np.asarray(cov, dtype=float) * np.asarray(population, dtype=float)
    return doses if age_idx is None else float(doses[age_idx])


def coverage_multiplier_for_target(base_inputs: dict, age_idx: int, target: float,
                                    seed: int = 0) -> float:
    """Cross-product schedule multiplier to reach `target` cumulative
    coverage in `age_idx`: `target / scheduled_coverage(age_idx)`, clamped at
    1.0 -- see `counterfactual_generic.coverage_multiplier_for_target`'s
    docstring for the two deliberate properties (scaled against the
    scheduled, not realized, coverage; never de-vaccinates a group already
    above target)."""
    baseline_cov = scheduled_coverage(age_idx)
    if baseline_cov <= 0:
        raise ValueError(f"age group {AGE_GROUPS[age_idx]} has zero baseline vaccination; "
                          "cannot scale to a coverage target")
    return max(1.0, target / baseline_cov)


def coverage_70pct_scenario(base_inputs: dict, age_idx: int | None = None, seed: int = 0) -> dict:
    """`dose_mult` scenario targeting 70% cumulative coverage. `age_idx=None`
    scales every age group; an int scales only that age group."""
    mult = np.ones(NUM_AGE_GROUPS)
    indices = range(NUM_AGE_GROUPS) if age_idx is None else [age_idx]
    for i in indices:
        mult[i] = coverage_multiplier_for_target(base_inputs, i, 0.70, seed=seed)
    return {"dose_mult": mult.tolist()}


def named_scenarios(base_inputs: dict) -> dict[str, dict]:
    """Every scenario builder above, by the same canonical name used in
    `run_simulations_MA_vax_single_age.py`'s `SCENARIOS` dict."""
    return {
        "baseline": baseline_scenario(),
        "no vax": no_vaccine_scenario(),
        "Infection protection only": infection_protection_only_scenario(base_inputs),
        "Low VE": ve_scenarios(base_inputs)["low_ve"],
        "High VE": ve_scenarios(base_inputs)["high_ve"],
        "70% coverage (all ages)": coverage_70pct_scenario(base_inputs, None),
    }


# ── Paired stochastic simulation ─────────────────────────────────────────────

def _run_reps(base_inputs: dict, scenario: dict, n_reps: int, seed: int,
              stochastic: bool = True) -> xr.Dataset:
    """`n_reps` replications of `scenario`, each seeded from `seed` and its
    own replication index only (never from which scenario is running) --
    pairs replications across scenarios for variance reduction, as long as
    both are run with the same (n_reps, seed). `stochastic=False` runs a
    single deterministic replication instead."""
    reps = n_reps if stochastic else 1
    A = NUM_AGE_GROUPS
    comp_arr = {c: np.zeros((reps, NUM_DAYS, A)) for c in COMPARTMENTS}
    trans_arr = {t: np.zeros((reps, NUM_DAYS, A)) for t in TRANSITIONS}
    for r in range(reps):
        rng_seed = (seed * 1_000_003 + r) if stochastic else SEED_BASE
        m = build_model(
            base_inputs, scenario.get("param_overrides"), scenario.get("dose_mult"),
            rng_seed=rng_seed, stochastic=stochastic,
        )
        m.simulate_until_day(NUM_DAYS)
        subpop = list(m.subpop_models.values())[0]
        h = _extract_age_arrays(subpop)
        for c in COMPARTMENTS:
            comp_arr[c][r] = h[c][:NUM_DAYS]
        for t in TRANSITIONS:
            trans_arr[t][r] = h[t][:NUM_DAYS]
    dates = pd.date_range(start=START_DATE, periods=NUM_DAYS, freq="D")
    data_vars = {c: (("replication", "day", "age_group"), comp_arr[c]) for c in COMPARTMENTS}
    data_vars.update({t: (("replication", "day", "age_group"), trans_arr[t]) for t in TRANSITIONS})
    return xr.Dataset(
        data_vars,
        coords={"replication": np.arange(reps), "day": dates, "age_group": AGE_GROUPS},
    )


def scenario_totals(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                     stochastic: bool = True) -> dict:
    ds = _run_reps(base_inputs, scenario, n_reps, seed, stochastic=stochastic)
    new_H = (ds["I_to_H"] + ds["IV_to_H"]).sum(dim="day").to_numpy()  # (reps, A)
    det_ds = _run_reps(base_inputs, scenario, 1, seed, stochastic=False)
    doses = det_ds["S_to_SV"].isel(replication=0).sum(dim="day").to_numpy()  # (A,)
    return {
        "new_H": new_H,
        "doses": doses,
        "population": np.asarray(base_inputs["population"], dtype=float),
    }


def _scenario_check_sums(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                          stochastic: bool = True) -> dict[str, np.ndarray]:
    ds = _run_reps(base_inputs, scenario, n_reps, seed, stochastic=stochastic)
    flows = ["S_to_E", "S_to_SV", "SV_to_EV", "I_to_H", "IV_to_H"]
    out = {t: ds[t].sum(dim="day").to_numpy() for t in flows}
    out["S0"] = ds["S"].isel(replication=0, day=0).to_numpy()
    out["SV0"] = ds["SV"].isel(replication=0, day=0).to_numpy()
    return out


def _scenario_daily_arrays(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                           stochastic: bool = True) -> dict[str, np.ndarray]:
    ds = _run_reps(base_inputs, scenario, n_reps, seed, stochastic=stochastic)
    names = ["S", "SV", "S_to_E", "SV_to_EV", "S_to_SV"]
    return {name: ds[name].to_numpy() for name in names}
