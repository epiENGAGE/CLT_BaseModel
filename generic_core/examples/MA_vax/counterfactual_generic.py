"""Counterfactual vaccination-impact pipeline for the generic_core-exported
MA_vax model (model_config.json + fitted_params.json in this folder), ported
from `MA_vax/counterfactual.py`.

Replicates the same table structure (PNAS 2505175122 Supplementary Tables
S.A.1-S.A.6, decomposed by `MA_vax/counterfactual.py`'s docstring) but builds
every simulation through generic_core's `ConfigDrivenSubpopModel` /
model_config machinery instead of the hand-written `MA_vax.model` equations
-- i.e. this is a check that the notebook-built/exported model reproduces the
same vaccination-impact conclusions as the original hand-written model.

The pure numeric helpers that only operate on already-computed arrays
(`_summ`, `averted_summary`, `attack_probability_curves`, `_rate_ratio_col`,
`_matched_cohort_ratio_col`) and the CSV I/O (`DICT_TABLES`,
`load_saved_tables`) don't depend on which model produced the arrays, so they
are imported and reused unchanged from `MA_vax.counterfactual` rather than
duplicated. Everything that actually runs a simulation (`_run_reps`,
`scenario_totals`, `_scenario_check_sums`, `_scenario_daily_arrays`, the
`table_S_A_*` builders, `save_all_tables`) is reimplemented here against
generic_core.

Location: this file must sit exactly two directory levels below the repo
root that contains generic_core/, clt_toolkit/ and flu_core/ -- same
constraint as run_simulations_MA_vax.py in this folder.

Single-population only (this exported model has IS_METAPOP=False) --
metapop support is not implemented here.

Coverage-target scenarios (`coverage_70pct_scenario`) use a NAIVE
cross-product multiplier (`target_coverage / baseline_coverage`, one
simulation, no bisection refinement) -- unlike
`MA_vax.counterfactual.coverage_multiplier_for_target`'s optional bisection
loop, this does not verify or correct for the S-cap binding. Check the
resulting `vaccination_coverage`-style table if you need to know how far off
70% the result actually lands.
"""

from __future__ import annotations

import copy
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent))

import clt_toolkit as clt
import flu_core as flu
from generic_core.config_parser import parse_model_config_from_dict
from generic_core.generic_model import (
    ConfigDrivenSubpopModel, build_state_from_config, build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel
from generic_core.model_factory import build_compartment_init, scale_dose_schedule_df
from generic_core.fitting import (
    _scale_compartment_init, _inject_tv_transmission, _tv_knot_days,
    build_transmission_multiplier_array, prepare_param_sets,
)

from MA_vax.model import AGE_GROUP_LABELS as AGE_GROUPS, COMPARTMENTS, TRANSITIONS
from MA_vax.counterfactual import (
    _summ, averted_summary, attack_probability_curves, _rate_ratio_col,
    _matched_cohort_ratio_col, DICT_TABLES, load_saved_tables,
)

# ---- Configurable (mirrors run_simulations_MA_vax.py) ----
MODEL_CONFIG_FILE = "model_config.json"
FITTED_PARAMS_FILE = "fitted_params.json"
SCHEDULES_FILE = "schedules.json"
NUM_DAYS = 250
TIMESTEPS_PER_DAY = 7
START_DATE = "2025-09-01"
NUM_AGE_GROUPS = 7
NUM_RISK_GROUPS = 1
SEED_BASE = 42


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
    per-age population needed by `build_model`. Mirrors the setup section of
    `run_simulation_MA_vax_small.py`."""
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


def _load_schedule_csvs(schedules_file: str = SCHEDULES_FILE):
    if not schedules_file:
        return {}
    p = _HERE / schedules_file
    if not p.exists():
        print(f"Warning: {schedules_file} not found next to counterfactual_generic.py -- "
              "falling back to flat constant schedules (no seasonal forcing, no vaccination).")
        return {}
    with open(p) as f:
        return json.load(f)


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
    multiplier on the vaccine-uptake schedule) applied. Mirrors
    `run_simulation_MA_vax_small.py`'s `build_model`, single-pop path only."""
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
# may be omitted (falls back to the fitted baseline / unscaled schedule).

def no_vaccine_scenario() -> dict:
    return {"dose_mult": [0.0] * NUM_AGE_GROUPS}


def baseline_scenario() -> dict:
    return {}


def single_age_only_scenario(age_idx: int) -> dict:
    mult = [0.0] * NUM_AGE_GROUPS
    mult[age_idx] = 1.0
    return {"dose_mult": mult}


def infection_protection_only_scenario(base_inputs: dict) -> dict:
    """VE against infection only: IV_to_H_prop set equal to I_to_H_prop, so
    vaccination confers no hospitalization-risk reduction conditional on
    infection."""
    ih = base_inputs["config_dict"]["params"]["I_to_H_prop"]
    return {"param_overrides": {"IV_to_H_prop": copy.deepcopy(ih)}}


def ve_scale_scenario(base_inputs: dict, vax_susceptibility_scale, IV_to_H_prop_scale) -> dict:
    """Multipliers (scalar or length-NUM_AGE_GROUPS sequence) on the fitted
    baseline vax_susceptibility / IV_to_H_prop."""
    p = base_inputs["config_dict"]["params"]
    vs = np.asarray(p["vax_susceptibility"], dtype=float)
    ivh = np.asarray(p["IV_to_H_prop"], dtype=float)
    vs_scale = np.broadcast_to(np.asarray(vax_susceptibility_scale, dtype=float).reshape(-1, 1), vs.shape)
    ivh_scale = np.broadcast_to(np.asarray(IV_to_H_prop_scale, dtype=float).reshape(-1, 1), ivh.shape)
    return {"param_overrides": {
        "vax_susceptibility": (vs * vs_scale).tolist(),
        "IV_to_H_prop": (ivh * ivh_scale).tolist(),
    }}


def ve_scenarios(base_inputs: dict) -> dict[str, dict]:
    """VE sensitivity presets, matching `MA_vax.counterfactual.VE_SCENARIOS`
    exactly (same multipliers) so Table S.A.4/S.A.5/S.A.6 are directly
    comparable between the two pipelines."""
    return {
        "low_ve": ve_scale_scenario(
            base_inputs,
            vax_susceptibility_scale=[1.1754, 1.1754, 1.1754, 1.1754, 1.1899, 1.1899, 1],
            IV_to_H_prop_scale=[1.0962, 1.0962, 1.0962, 1.0962, 1.0535, 1.0535, 1.1449],
        ),
        "baseline_ve": ve_scale_scenario(
            base_inputs, vax_susceptibility_scale=[1.0] * 7, IV_to_H_prop_scale=[1.0] * 7,
        ),
        "high_ve": ve_scale_scenario(
            base_inputs,
            vax_susceptibility_scale=[0.8596, 0.8596, 0.8596, 0.8596, 0.8354, 0.8354, 0.86],
            IV_to_H_prop_scale=[0.4922, 0.4922, 0.4922, 0.4922, 0.9272, 0.9272, 1.028],
        ),
    }


def coverage_multiplier_for_target(base_inputs: dict, age_idx: int, target: float,
                                    seed: int = 0) -> float:
    """Naive cross-product schedule multiplier to reach `target` cumulative
    coverage in `age_idx` by season's end: `target / baseline_coverage`, from
    one deterministic baseline run. No bisection refinement/verification --
    see module docstring."""
    ds = _run_reps(base_inputs, baseline_scenario(), n_reps=1, seed=seed, stochastic=False)
    baseline_cov = float(ds["S_to_SV"].isel(replication=0).sum(dim="day").to_numpy()[age_idx]
                          / base_inputs["population"][age_idx])
    if baseline_cov <= 0:
        raise ValueError(f"age group {AGE_GROUPS[age_idx]} has zero baseline vaccination; "
                          "cannot scale to a coverage target")
    return target / baseline_cov


def coverage_70pct_scenario(base_inputs: dict, age_idx: int | None = None, seed: int = 0) -> dict:
    """`dose_mult` scenario targeting 70% cumulative coverage. `age_idx=None`
    scales every age group; an int scales only that age group (others stay
    at their baseline schedule)."""
    mult = np.ones(NUM_AGE_GROUPS)
    indices = range(NUM_AGE_GROUPS) if age_idx is None else [age_idx]
    for i in indices:
        mult[i] = coverage_multiplier_for_target(base_inputs, i, 0.70, seed=seed)
    return {"dose_mult": mult.tolist()}


def named_scenarios(base_inputs: dict) -> dict[str, dict]:
    """Every scenario builder above, by the same canonical name used in
    `run_simulations_MA_vax.py`'s `SCENARIOS` dict (so the two are directly
    cross-referenceable) -- a single source of truth for anything that wants
    to run or list scenarios by name rather than calling the builder
    functions directly (e.g. the notebook's epi-curve section). Excludes
    `'children vax only'`, which has no `counterfactual.py`-derived builder
    and only exists as a hand-written entry in `run_simulations_MA_vax.py`.
    """
    scenarios = {
        "baseline": baseline_scenario(),
        "no vax": no_vaccine_scenario(),
        "Infection protection only": infection_protection_only_scenario(base_inputs),
        "Low VE": ve_scenarios(base_inputs)["low_ve"],
        "High VE": ve_scenarios(base_inputs)["high_ve"],
        "70% coverage (all ages)": coverage_70pct_scenario(base_inputs, None),
    }
    for i, label in enumerate(AGE_GROUPS):
        scenarios[f"Vaccinate {label} only"] = single_age_only_scenario(i)
        scenarios[f"70% coverage ({label} only)"] = coverage_70pct_scenario(base_inputs, i)
    return scenarios


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


# ── Table builders (structure mirrors MA_vax.counterfactual's table_S_A_*) ──

def table_S_A_1(base_inputs: dict, n_reps: int = 200, seed: int = 0, stochastic: bool = True) -> pd.DataFrame:
    no_vax = scenario_totals(base_inputs, no_vaccine_scenario(), n_reps, seed, stochastic=stochastic)
    inf_only = scenario_totals(base_inputs, infection_protection_only_scenario(base_inputs), n_reps, seed,
                               stochastic=stochastic)
    full = scenario_totals(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)

    reduced_infection = averted_summary(no_vax, inf_only).add_suffix("_reduced_infection")
    reduced_severity = averted_summary(inf_only, full, pct_reference=no_vax).add_suffix("_reduced_severity")
    total = averted_summary(no_vax, full).add_suffix("_total")
    return reduced_infection.join(reduced_severity).join(total)


def table_S_A_2(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                stochastic: bool = True) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(base_inputs, no_vaccine_scenario(), n_reps, seed, stochastic=stochastic)
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(base_inputs, single_age_only_scenario(i), n_reps, seed, stochastic=stochastic)
        cols[label] = averted_summary(no_vax, scen)
    cols["All"] = averted_summary(
        no_vax, scenario_totals(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic))

    return {
        "pct_reduction": pd.DataFrame({label: df["pct_averted"] for label, df in cols.items()}),
        "per_100k": pd.DataFrame({label: df["per100k_averted"] for label, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({label: df["per100k_doses_averted"] for label, df in cols.items()}),
    }


def table_S_A_3(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                stochastic: bool = True) -> dict[str, pd.DataFrame]:
    baseline = scenario_totals(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(base_inputs, coverage_70pct_scenario(base_inputs, i, seed=seed), n_reps, seed,
                               stochastic=stochastic)
        cols[label] = averted_summary(baseline, scen)
    all_scen = scenario_totals(base_inputs, coverage_70pct_scenario(base_inputs, None, seed=seed), n_reps, seed,
                               stochastic=stochastic)
    cols["All"] = averted_summary(baseline, all_scen)

    return {
        "pct_reduction": pd.DataFrame({label: df["pct_averted"] for label, df in cols.items()}),
        "per_100k": pd.DataFrame({label: df["per100k_averted"] for label, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({label: df["per100k_doses_averted"] for label, df in cols.items()}),
    }


def table_S_A_4(base_inputs: dict) -> pd.DataFrame:
    rows = []
    p = base_inputs["config_dict"]["params"]
    ih = np.asarray(p["I_to_H_prop"], dtype=float).flatten()
    i_rel_inf = float(p["I_relative_infectiousness"])
    for name, scen in ve_scenarios(base_inputs).items():
        overrides = scen.get("param_overrides", {})
        vs = np.asarray(overrides.get("vax_susceptibility", p["vax_susceptibility"]), dtype=float).flatten()
        ivh = np.asarray(overrides.get("IV_to_H_prop", p["IV_to_H_prop"]), dtype=float).flatten()
        iv_rel_inf = float(overrides.get("IV_relative_infectiousness", p["IV_relative_infectiousness"]))
        ve_inf = 1.0 - vs
        ve_hosp_given_inf = 1.0 - ivh / ih
        ve_hosp_inf = 1 - (1 - ve_inf) * (1 - ve_hosp_given_inf)
        ve_transmission_blocking = 1.0 - iv_rel_inf / i_rel_inf
        for i, label in enumerate(AGE_GROUPS):
            rows.append({
                "scenario": name,
                "age_group": label,
                "VE_infection": f"{ve_inf[i] * 100:.0f}%",
                "VE_hosp_infection": f"{ve_hosp_inf[i] * 100:.0f}%",
                "VE_hosp_given_infection": f"{ve_hosp_given_inf[i] * 100:.0f}%",
                "VE_transmission_blocking": f"{ve_transmission_blocking * 100:.0f}%",
            })
    return pd.DataFrame(rows).set_index(["scenario", "age_group"])


def table_S_A_5(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                stochastic: bool = True) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(base_inputs, no_vaccine_scenario(), n_reps, seed, stochastic=stochastic)
    cols = {
        name: averted_summary(no_vax, scenario_totals(base_inputs, scen, n_reps, seed, stochastic=stochastic))
        for name, scen in ve_scenarios(base_inputs).items()
    }
    return {
        "pct_reduction": pd.DataFrame({name: df["pct_averted"] for name, df in cols.items()}),
        "per_100k": pd.DataFrame({name: df["per100k_averted"] for name, df in cols.items()}),
    }


def table_S_A_6(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                stochastic: bool = True) -> dict[str, pd.DataFrame]:
    cols = {}
    for name, ve_scen in ve_scenarios(base_inputs).items():
        # ve_scen only overrides vax_susceptibility/IV_to_H_prop; baseline_scenario()/
        # coverage_70pct_scenario() only set dose_mult -- merge both into one scenario dict.
        def _merged(dose_scen, ve_scen=ve_scen):
            merged = dict(ve_scen)
            if "dose_mult" in dose_scen:
                merged["dose_mult"] = dose_scen["dose_mult"]
            return merged
        baseline = scenario_totals(base_inputs, _merged(baseline_scenario()), n_reps, seed, stochastic=stochastic)
        target70 = scenario_totals(
            base_inputs, _merged(coverage_70pct_scenario(base_inputs, None, seed=seed)), n_reps, seed,
            stochastic=stochastic)
        cols[name] = averted_summary(baseline, target70)
    return {
        "pct_reduction": pd.DataFrame({name: df["pct_averted"] for name, df in cols.items()}),
        "per_100k": pd.DataFrame({name: df["per100k_averted"] for name, df in cols.items()}),
    }


def table_vax_efficacy_check(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                              stochastic: bool = True) -> dict[str, pd.DataFrame]:
    cols_inf, cols_hosp, cols_matched = {}, {}, {}
    for i, label in enumerate(AGE_GROUPS):
        s = _scenario_check_sums(base_inputs, single_age_only_scenario(i), n_reps, seed, stochastic=stochastic)
        vax_den = s["SV0"][None, :] + s["S_to_SV"]
        unvax_den = s["S0"][None, :] - s["S_to_SV"]
        cols_inf[label] = _rate_ratio_col(s["SV_to_EV"], vax_den, s["S_to_E"], unvax_den)
        cols_hosp[label] = _rate_ratio_col(s["IV_to_H"], s["SV_to_EV"], s["I_to_H"], s["S_to_E"])

        d = _scenario_daily_arrays(base_inputs, single_age_only_scenario(i), n_reps, seed, stochastic=stochastic)
        cols_matched[label] = _matched_cohort_ratio_col(
            d["S"], d["SV"], d["S_to_E"], d["SV_to_EV"], d["S_to_SV"])

    s_all = _scenario_check_sums(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)
    vax_den_all = s_all["SV0"][None, :] + s_all["S_to_SV"]
    unvax_den_all = s_all["S0"][None, :] - s_all["S_to_SV"]
    cols_inf["All"] = _rate_ratio_col(s_all["SV_to_EV"], vax_den_all, s_all["S_to_E"], unvax_den_all)
    cols_hosp["All"] = _rate_ratio_col(s_all["IV_to_H"], s_all["SV_to_EV"], s_all["I_to_H"], s_all["S_to_E"])

    d_all = _scenario_daily_arrays(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)
    cols_matched["All"] = _matched_cohort_ratio_col(
        d_all["S"], d_all["SV"], d_all["S_to_E"], d_all["SV_to_EV"], d_all["S_to_SV"])

    return {
        "infection_reduction": pd.DataFrame(cols_inf, index=AGE_GROUPS),
        "matched_cohort_infection_reduction": pd.DataFrame(cols_matched, index=AGE_GROUPS),
        "hospitalization_reduction": pd.DataFrame(cols_hosp, index=AGE_GROUPS),
    }


# ── Persisted-table I/O ───────────────────────────────────────────────────────
# `DICT_TABLES` / `load_saved_tables` are reused unchanged from
# MA_vax.counterfactual (pure CSV I/O keyed by the same filenames written
# below) -- see this module's imports.

def save_all_tables(base_inputs: dict, out_folder: str, n_reps: int = 500, seed: int = 0,
                    stochastic: bool = True, meta: dict | None = None, log=print) -> None:
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "meta.json"), "w") as f:
        json.dump({**(meta or {}), "n_reps": n_reps if stochastic else 1, "seed": seed,
                   "stochastic": stochastic}, f, indent=2)

    log("[1/7] Table S.A.1 (infection vs severity protection) ...")
    table_S_A_1(base_inputs, n_reps, seed, stochastic).to_csv(os.path.join(out_folder, "S_A_1.csv"))

    log("[2/7] Table S.A.2 (age group vaccinated) ...")
    for sub, df in table_S_A_2(base_inputs, n_reps, seed, stochastic).items():
        df.to_csv(os.path.join(out_folder, f"S_A_2_{sub}.csv"))

    log("[3/7] Table S.A.3 (70% coverage, single age group) ...")
    for sub, df in table_S_A_3(base_inputs, n_reps, seed, stochastic).items():
        df.to_csv(os.path.join(out_folder, f"S_A_3_{sub}.csv"))

    log("[4/7] Table S.A.4 (VE sensitivity parameters) ...")
    table_S_A_4(base_inputs).to_csv(os.path.join(out_folder, "S_A_4.csv"))

    log("[5/7] Table S.A.5 (VE sensitivity, vs no vaccine) ...")
    for sub, df in table_S_A_5(base_inputs, n_reps, seed, stochastic).items():
        df.to_csv(os.path.join(out_folder, f"S_A_5_{sub}.csv"))

    log("[6/7] Table S.A.6 (VE sensitivity, 70% coverage) ...")
    for sub, df in table_S_A_6(base_inputs, n_reps, seed, stochastic).items():
        df.to_csv(os.path.join(out_folder, f"S_A_6_{sub}.csv"))

    log("[7/7] Vaccine-efficacy mechanism check (flow-level ratios) ...")
    for sub, df in table_vax_efficacy_check(base_inputs, n_reps, seed, stochastic).items():
        df.to_csv(os.path.join(out_folder, f"VAX_CHECK_{sub}.csv"))
