#!/usr/bin/env python3
"""
Single-age-group adaptation of MA_vax/run_simulations_MA_vax.py.

Requirements: the CLT_BaseModel repo installed into the active Python
environment (from the repo root: pip install -e .), which makes
generic_core, clt_toolkit and flu_core importable from anywhere -- so
this file can live in any folder. The model config (and fitted params /
schedules.json, if used) must sit alongside it.
"""

import io
import os
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Configurable ----
MODEL_CONFIG_FILE = "model_config_MA_single_age.json"
FITTED_PARAMS_FILE = "fitted_params_MA_single_age.json"  # set None to skip
# Schedule CSVs (humidity / school-work calendar / mobility / vaccination),
# single-population only. The files named in the model config's "input_files"
# are read fresh from disk each run (resolved against the working directory,
# then this script's folder and its parents), so editing a CSV takes effect
# without re-exporting. SCHEDULES_FILE is the Export tab's snapshot of those
# same CSVs, used only for a file that can't be found (a notice is printed
# whenever the snapshot is stale); it also carries any per-scenario
# schedule-replacement CSVs. With neither, the script falls back to flat
# constants (no seasonal forcing, NO vaccination).
SCHEDULES_FILE = "schedules.json"
# Transition variables (daily flows, e.g. I_to_H / IV_to_H) to record
# alongside the compartments. None = every transition in model_config.json;
# set to a list to record only some, or [] for compartments only.
TRANSITION_VARS = None
OUTPUT_DIR = Path("simulation_output")
NUM_DAYS = 250
NUM_REPS = 1
STOCHASTIC = True
TIMESTEPS_PER_DAY = 7
START_DATE = "2025-09-01"
NUM_AGE_GROUPS = 1
NUM_RISK_GROUPS = 1
# Base RNG seed; each run uses default_rng(SEED_BASE + run_index), matching
# the notebook's Analysis tab.
SEED_BASE = 42
# Scenario x replicate runs are independent, so they run in a process pool
# (real OS processes -- CPU-bound numpy/Python simulation work would not
# benefit from threads because of the GIL). None = os.cpu_count(); set to 1
# to run serially (e.g. for easier debugging/tracebacks).
NUM_WORKERS = None

# Where the spread between replicates comes from (mirrors the Analysis tab's
# "Uncertainty source" control):
#   "transitions"            - every replicate uses the fitted BEST parameter
#                              set and differs only in the transition RNG.
#   "parameters"             - draw NUM_PARAM_SETS sets at random (without
#                              replacement) from the fit's accepted_params and
#                              run each ONE time with DETERMINISTIC transitions.
#                              The spread is parameter uncertainty alone, with
#                              no transition RNG noise, so repeating a set
#                              would only duplicate its trajectory -- NUM_REPS
#                              is ignored and the ensemble size is the number
#                              of sampled sets.
#   "parameters+transitions" - draw NUM_PARAM_SETS sets at random (without
#                              replacement) from the fit's accepted_params and
#                              spread NUM_REPS replicates evenly across them,
#                              so the ensemble carries parameter uncertainty
#                              too.
# Both sampling modes require a fitted_params.json with more than one accepted
# set, and both are ignored when STOCHASTIC is False (deterministic always runs
# once, with the best set). fitted_params_MA_single_age.json supplies 495
# accepted sets, so these modes are live here.
UNCERTAINTY_SOURCE = "parameters"
NUM_PARAM_SETS = 495
# Whether the transition engine itself is stochastic. Derived, not a setting:
# the "parameters" mode is stochastic in the sense that it samples the
# posterior, but each run's transitions are deterministic.
RUN_STOCHASTIC = STOCHASTIC and UNCERTAINTY_SOURCE != "parameters"

# Metapopulation reproduction (set by the notebook). When IS_METAPOP is
# True the script builds the model from METAPOP_FOLDER's inputs; that
# folder must exist at run time (edit the path below if you move it).
IS_METAPOP = False
METAPOP_FOLDER = ''
METAPOP_TRAVEL_CONFIG = {}

# Define scenarios: {name: {param: value}}
#
# Scalar params take a number. Array params (per age/risk group) take the
# FULL nested list, shape [num_age_groups][num_risk_groups] -- here that's
# always [[value]] since NUM_AGE_GROUPS = NUM_RISK_GROUPS = 1.
#
# Any param you set here must also be listed in DESIGNED_PARAMS below to
# survive parameter sampling -- see the note there.
#
# Each scenario sets ONLY the params it deliberately changes away from the
# fitted baseline -- everything else falls through to config_dict["params"],
# which already holds fitted_params_MA_single_age.json's best set merged over
# model_config.json. Don't re-list baseline values here: a copied-in value
# silently overrides the fit in deterministic / "transitions" runs, and goes
# stale the moment the model is re-fit. ("baseline", "no vax" and the
# coverage scenarios are therefore empty -- they differ only through
# DOSE_MULTIPLIER below.)
#
# Designed values below are DESIGNED_PARAM_RATIOS x the fitted best set (for
# IV_to_H_prop, best = 0.005051315603850319, i.e. model_config's
# 0.015785059937119972 x fitted IHR_scale 0.32000610855912565), so a
# deterministic run gives exactly what a parameter-sampling run gives for the
# best draw. vax_susceptibility is not fitted, so its values are ratio x
# model_config's 0.7836264370076347. If you RE-FIT, recompute the
# IV_to_H_prop values here (ratio x new best); under parameter sampling they
# are not read at all -- the ratio is applied to each draw instead.
#
# Age-targeted scenarios from MA_vax/run_simulations_MA_vax.py (children vax
# only, Vaccinate <age> only, 70% coverage (<age> only)) rely on zeroing/
# scaling one age band's dose schedule independently of the others and have
# no single-age equivalent, so they're omitted -- see module docstring.
# "Infection protection only" IS kept: it doesn't target an age group at
# all (no DOSE_MULTIPLIER entry, real unscaled coverage) -- it isolates
# the vaccine's infection-blocking effect from its severity-reduction
# effect by setting IV_to_H_prop = I_to_H_prop (vaccinated breakthrough
# infections hospitalize at the same rate as unvaccinated ones) while
# leaving vax_susceptibility at its normal value.
SCENARIOS = {
    "baseline": {},
    "no vax": {},
    "Infection protection only": {"IV_to_H_prop": [[0.006777229097599511]]},
    "Low VE": {"vax_susceptibility": [[0.8976501514033461]], "IV_to_H_prop": [[0.005687597232161169]]},
    "High VE": {"vax_susceptibility": [[0.6616378415241131]], "IV_to_H_prop": [[0.0049138373017620935]]},
    "70% coverage (all ages)": {},
    "Low VE + 70% coverage (all ages)": {"vax_susceptibility": [[0.8976501514033461]], "IV_to_H_prop": [[0.005687597232161169]]},
    "High VE + 70% coverage (all ages)": {"vax_susceptibility": [[0.6616378415241131]], "IV_to_H_prop": [[0.0049138373017620935]]},
}

# Per-subpopulation parameter overrides per scenario (metapop only)
# Format: {scenario_name: [override_dict_or_None, ...]} indexed by subpop order
SUBPOP_PARAM_OVERRIDES = {
}

# Parameters each scenario deliberately sets: {scenario_name: [param, ...]}.
#
# Only relevant when UNCERTAINTY_SOURCE samples parameters. Each run
# takes its parameters from a sampled fitted set, EXCEPT the ones listed here
# for that scenario, which keep the value given in SCENARIOS. Without this,
# the sampled set would overwrite the very parameter the scenario varies and
# every scenario would collapse to the same thing.
#
# A parameter that is in SCENARIOS but NOT in this list for that scenario is
# deliberately NOT protected: it comes from the sampled parameter set, and the
# SCENARIOS value is ignored. That is what makes the non-varied parameters
# carry posterior uncertainty. The notebook lists exactly the parameters you
# changed away from the fitted baseline, so its scenarios behave as designed.
#
# If you ADD a scenario by hand, add it here too. A scenario name missing from
# this dict falls back to protecting everything it sets in SCENARIOS (safe for
# hand-written scenarios, but it means none of its params carry parameter
# uncertainty). Use an explicit empty list to opt a scenario fully into
# sampling.
#
# Here every SCENARIOS entry sets exactly its designed params and nothing
# else, so each list below matches its scenario's keys. "baseline" is still
# listed explicitly ([]) so the dict reads as complete; with an empty
# SCENARIOS entry the fallback would protect nothing either way. If you
# ever give a scenario extra non-designed keys, list it here with only the
# designed ones, or the fallback will pin those extras across every draw.
DESIGNED_PARAMS = {
    "baseline": [],
    "no vax": [],
    "Infection protection only": ["IV_to_H_prop"],
    "Low VE": ["vax_susceptibility", "IV_to_H_prop"],
    "High VE": ["vax_susceptibility", "IV_to_H_prop"],
    "70% coverage (all ages)": [],
    "Low VE + 70% coverage (all ages)": ["vax_susceptibility", "IV_to_H_prop"],
    "High VE + 70% coverage (all ages)": ["vax_susceptibility", "IV_to_H_prop"],
}

# Ratio (override / catalog baseline) for designed params, where known:
# {scenario_name: {param: ratio_or_nested_list}}.
#
# Only relevant when UNCERTAINTY_SOURCE samples parameters. A designed
# param listed here has the sampled draw's own value scaled by this ratio
# instead of being pinned to the literal value in SCENARIOS -- so the
# scenario's intended scaling survives parameter uncertainty. A designed
# param NOT listed here (or added by hand) falls back to being pinned to
# the literal SCENARIOS value on every draw.
#
# Each ratio is a RATIO OF population-weighted AVERAGES (scenario's
# single-age value / baseline's), NOT an average of MA_vax's per-age
# ratios -- the latter doesn't preserve the collapsed value and was off by
# up to ~14%. Specifically:
#   - "Infection protection only" IV_to_H_prop: model_config's I_to_H_prop /
#     IV_to_H_prop (0.021178.../0.015785...). The fit's IHR_scale multiplies
#     both by the same factor, so this makes IV_to_H_prop == I_to_H_prop
#     exactly on every draw -- the scenario's whole definition.
#   - Low/High VE IV_to_H_prop and vax_susceptibility: pop-weighted average
#     of MA_vax's Low/High VE 7-age values / the same for its baseline.
# vax_susceptibility is not a fitted param, so it never appears in a draw
# and its ratio is never applied under sampling -- the SCENARIOS literal is
# used. It's kept here so it stays right if vax_susceptibility is ever fit.
DESIGNED_PARAM_RATIOS = {
    "Infection protection only": {
        "IV_to_H_prop": [[1.3416760363248001]],
    },
    "Low VE": {
        "vax_susceptibility": [[1.1455077432445029]],
        "IV_to_H_prop": [[1.1259635465710853]],
    },
    "High VE": {
        "vax_susceptibility": [[0.8443281266143231]],
        "IV_to_H_prop": [[0.9727836641243651]],
    },
    "Low VE + 70% coverage (all ages)": {
        "vax_susceptibility": [[1.1455077432445029]],
        "IV_to_H_prop": [[1.1259635465710853]],
    },
    "High VE + 70% coverage (all ages)": {
        "vax_susceptibility": [[0.8443281266143231]],
        "IV_to_H_prop": [[0.9727836641243651]],
    },
}

# Per-age-group multiplier on a `scheduled_exact` transition's daily
# schedule, per scenario (a schedule override, not a params entry).
# Despite the "vaccine" naming baked into the model config schema
# (daily_vaccines_df), scheduled_exact isn't vaccine-specific -- this
# scales whatever it represents (antiviral courses, etc.). Applies to
# every subpop identically in a metapop run, except where overridden by
# DOSE_MULTIPLIER_PER_SUBPOP below.
# Format: {scenario_name: [multiplier_per_age_group, ...]} -- here always a
# single-element list, since NUM_AGE_GROUPS = 1; or, to scale two
# `scheduled_exact` schedules by different amounts, a per-schedule dict:
#   {scenario_name: {"daily_vaccines_df": [...], "<other>_df": [...]}}
# keyed by each schedule's df_attribute (a schedule left out is unscaled).
#
# "70% coverage (all ages)"'s multiplier is the population-weighted average
# of MA_vax/run_simulations_MA_vax.py's 7-element DOSE_MULTIPLIER for that
# scenario -- a naive cross-product estimate (target coverage / baseline
# coverage), same caveat as there: not bisection-refined.
DOSE_MULTIPLIER = {
    "no vax": [0.0],
    "70% coverage (all ages)": [1.3971676077459427],
    "Low VE + 70% coverage (all ages)": [1.3971676077459427],
    "High VE + 70% coverage (all ages)": [1.3971676077459427],
}

# Per-subpopulation override of DOSE_MULTIPLIER above (metapop only).
# Format: {scenario_name: [multiplier_or_None, ...]} indexed by subpop
# order; a None entry falls back to DOSE_MULTIPLIER. Each non-None entry
# takes either shape accepted by DOSE_MULTIPLIER (list, or per-schedule
# dict keyed by df_attribute).
DOSE_MULTIPLIER_PER_SUBPOP = {
}

# Schedules REPLACED outright by an uploaded CSV, per scenario (as
# opposed to DOSE_MULTIPLIER, which scales a schedule in place). Names
# the df_attribute(s) overridden; the actual CSV data lives in
# schedules.json under "__scenario_overrides__" (see
# _schedule_overrides_from_json) -- download it from the Export tab
# alongside this script.
# Format: {scenario_name: {df_attribute: df_attribute}}
SCHEDULE_OVERRIDES = {
}

# Per-subpopulation override of SCHEDULE_OVERRIDES above (metapop only).
# Format: {scenario_name: [{df_attribute: df_attribute} | None, ...]}
# indexed by subpop order.
SCHEDULE_OVERRIDES_PER_SUBPOP = {
}

# ---- Setup ----
_HERE = Path(__file__).parent

# Write output next to this script, not into whatever the current working
# directory happens to be. An absolute OUTPUT_DIR is used as-is.
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = _HERE / OUTPUT_DIR

import duckdb
import clt_toolkit as clt
import flu_core as flu
from generic_core import results_io
from generic_core.config_parser import parse_model_config_from_dict
from generic_core.generic_model import (
    ConfigDrivenSubpopModel, build_state_from_config, build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel
from generic_core.model_factory import (
    build_compartment_init, make_metapop_from_folder, extract_history,
    extract_history_full, scale_dose_schedule_df, config_schedule_df_attributes,
    load_schedule_csv_texts, input_file_schedule_paths,
)
from generic_core.fitting import (
    _scale_compartment_init, _inject_tv_transmission, _tv_knot_days,
    build_transmission_multiplier_array, prepare_param_sets,
)

with open(_HERE / MODEL_CONFIG_FILE) as _f:
    config_dict = json.load(_f)

if TRANSITION_VARS is None:
    TRANSITION_VARS = [
        _t["name"] for _t in config_dict.get("transitions", []) if _t.get("name")
    ]
else:
    TRANSITION_VARS = list(TRANSITION_VARS)

# Fitted params can carry three kinds of entries:
#  - regular config["params"] scalars/arrays  -> merged in directly below
#  - seed_scale_<comp>  -> scales that seeded compartment's initial condition
#    (mirrors generic_core.fitting._scale_compartment_init); applied in
#    build_model(), not a config["params"] entry
#  - m_dlog_*  -> log-increments of the fitted time-varying transmission
#    multiplier m(t); reconstructed in build_model() via a
#    'transmission_multiplier' schedule (mirrors
#    generic_core.fitting._inject_tv_transmission), exact over the fit period
#    (FIT_NUM_DAYS) and held flat at its last value beyond it.
#  - phi (NB2 observation-noise dispersion) and linked-scale multiplier keys
#    (see SCALE_GROUPS) are not model parameters and are dropped/expanded.
SEED_SCALES = {}
TV_INCREMENTS = []
TV_SPACING = 30
FIT_NUM_DAYS = 0
# Accepted/posterior parameter sets behind the best set, used only when
# UNCERTAINTY_SOURCE samples parameters ("parameters" / "parameters+transitions").
PARAM_SETS = []


def _split_pset(_pset):
    # A prepared param set mixes three kinds of entry. Split them the way
    # each is actually applied: model params merge into config["params"],
    # seed_scale_* scales the initial conditions, m_dlog_* rebuilds m(t).
    # phi (NB2 observation-noise dispersion) is not a model parameter.
    _model = {
        _k: _v for _k, _v in _pset.items()
        if _k != "phi" and not _k.startswith("m_dlog_") and not _k.startswith("seed_scale_")
    }
    _scales = {
        _k[len("seed_scale_"):]: float(_v)
        for _k, _v in _pset.items() if _k.startswith("seed_scale_")
    }
    _incr = [
        _v for _, _v in sorted(
            (
                (int(_k[len("m_dlog_"):]), float(_v))
                for _k, _v in _pset.items()
                if _k.startswith("m_dlog_") and _k[len("m_dlog_"):].isdigit()
            ),
            key=lambda _t: _t[0],
        )
    ]
    return _model, _scales, _incr


if FITTED_PARAMS_FILE is not None:
    _fp = _HERE / FITTED_PARAMS_FILE
    if _fp.exists():
        with open(_fp) as _f:
            _fitted_raw = json.load(_f)
    else:
        print(f"Warning: {FITTED_PARAMS_FILE} not found")
        _fitted_raw = {}

    _best_params = _fitted_raw.get("best_params", _fitted_raw) if isinstance(_fitted_raw, dict) else {}
    _accepted = (_fitted_raw.get("accepted_params") or []) if isinstance(_fitted_raw, dict) else []
    _scale_groups = (_fitted_raw.get("scale_groups", {}) or {}) if isinstance(_fitted_raw, dict) else {}
    FIT_NUM_DAYS = int(_fitted_raw.get("num_days", 0) or 0) if isinstance(_fitted_raw, dict) else 0
    TV_SPACING = int(_fitted_raw.get("tv_knot_spacing_days", 30) or 30) if isinstance(_fitted_raw, dict) else 30

    # prepare_param_sets reassembles MCMC/ABC-SMC per-element sampler columns
    # (`pn|a0`, `pn|a1`, ... -- AR/gradient record one array-valued `pn`
    # directly) and expands linked-scale multipliers into concrete
    # base-param values (base := model_config baseline x multiplier). Both
    # must be resolved against the ORIGINAL (pre-fit) config baseline, so
    # capture it before config_dict["params"] is updated below.
    _orig_params = dict(config_dict.get("params", {}) or {})
    _expanded = prepare_param_sets([_best_params], _scale_groups, _orig_params)[0]
    PARAM_SETS = prepare_param_sets(_accepted, _scale_groups, _orig_params)

    FITTED_PARAMS, SEED_SCALES, TV_INCREMENTS = _split_pset(_expanded)
    if TV_INCREMENTS and FIT_NUM_DAYS <= 0:
        print("Warning: fitted params have m_dlog_* but no num_days -- m(t) will not be reconstructed")
        TV_INCREMENTS = []
    config_dict["params"] = {**config_dict.get("params", {}), **FITTED_PARAMS}

    if SEED_SCALES:
        print(f"Applying fitted seed scaling to: {sorted(SEED_SCALES)}")
    if TV_INCREMENTS:
        print(
            f"Reconstructing m(t) from {len(TV_INCREMENTS)} fitted log-increment(s) "
            f"(fit period {FIT_NUM_DAYS} days, knots every {TV_SPACING} days)"
        )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_tvm_df(start_date, num_days, tv_increments=None):
    # Reconstruct the fitted m(t) trajectory: exact over the fit period
    # (FIT_NUM_DAYS) and held flat at its last value beyond it. Returns None
    # when no m(t) was fit. Shared by the single-pop schedules and the metapop
    # build (where it is broadcast identically to every subpop).
    # tv_increments defaults to the best set's; a sampled param set passes
    # its own (each posterior draw carries its own m(t) trajectory).
    _incr = TV_INCREMENTS if tv_increments is None else tv_increments
    if not _incr or FIT_NUM_DAYS <= 0:
        return None
    _h = max(num_days + 14, 370)
    _dates = pd.date_range(start=start_date, periods=_h, freq="D").date
    _knots = _tv_knot_days(FIT_NUM_DAYS, TV_SPACING)
    _m_fit = build_transmission_multiplier_array(_incr, _knots, FIT_NUM_DAYS)
    if _h <= FIT_NUM_DAYS:
        _m_full = _m_fit[:_h]
    else:
        _m_full = np.concatenate([_m_fit, np.full(_h - FIT_NUM_DAYS, _m_fit[-1])])
    return pd.DataFrame({"date": _dates, "transmission_multiplier": _m_full})


_SCHEDULE_CSVS = None


def _load_schedule_csvs():
    # {df_attribute: csv_text} of the base schedules, plus SCHEDULES_FILE's
    # "__scenario_overrides__". The CSVs named in the model config's
    # input_files win; SCHEDULES_FILE only fills in files that can't be found
    # (see load_schedule_csv_texts). Any attribute absent from both had no
    # CSV in the notebook and correctly falls back to a flat constant below.
    # Metapop reads its base schedules from METAPOP_FOLDER instead, so only
    # the snapshot's overrides are used there. Loaded once per process;
    # notices print only in the main process, not in every pool worker.
    global _SCHEDULE_CSVS
    if _SCHEDULE_CSVS is None:
        _snap = (_HERE / SCHEDULES_FILE) if SCHEDULES_FILE else None
        _cfg = {} if IS_METAPOP else config_dict
        _verbose = __name__ == "__main__"
        if (
            _verbose and _snap is not None and not _snap.exists()
            and not input_file_schedule_paths(_cfg, [])
        ):
            print(
                f"Warning: {SCHEDULES_FILE} not found next to this script and "
                "the model config names no schedule CSVs -- falling back to flat constant "
                "schedules (no seasonal forcing, no vaccination). Results will NOT match "
                "the notebook."
            )
        _SCHEDULE_CSVS = load_schedule_csv_texts(
            _cfg, snapshot_path=_snap,
            search_roots=[Path.cwd(), _HERE, *_HERE.parents],
            verbose=_verbose,
        )
    return _SCHEDULE_CSVS


def _dose_mult_for(dose_mult, attr):
    # A scenario's dose multiplier is either
    #   - a per-age-group list  -> applied to EVERY vaccine_schedule, or
    #   - a {df_attribute: per-age-group list} dict -> per-schedule, letting
    #     two `scheduled_exact` schedules be scaled by different amounts
    #     (a schedule absent from the dict is left unscaled).
    # Both shapes are accepted so hand-edited scripts using the simple list
    # form keep working.
    if dose_mult is None:
        return None
    if isinstance(dose_mult, dict):
        return dose_mult.get(attr)
    return dose_mult


def _schedule_overrides_from_json(scenario_name):
    # {df_attribute: csv_text} of uploaded schedule-REPLACEMENT CSVs for
    # this scenario (see the Analysis tab's "Replace schedules from
    # uploaded CSVs" control), read from schedules.json's reserved
    # "__scenario_overrides__" key. Only the attributes SCHEDULE_OVERRIDES
    # names for this scenario are pulled out, so a schedules.json missing
    # an entry (e.g. re-downloaded before that upload was added) degrades
    # to "no override" instead of raising. Returns None if this scenario
    # has no schedule overrides.
    _attrs = SCHEDULE_OVERRIDES.get(scenario_name)
    if not _attrs:
        return None
    _csvs = _load_schedule_csvs()
    _entry = (_csvs.get("__scenario_overrides__", {}) or {}).get("per_scenario", {}).get(scenario_name, {})
    return {_a: _entry[_a] for _a in _attrs if _a in _entry} or None


def _schedule_overrides_per_subpop_from_json(scenario_name):
    # [{df_attribute: csv_text} | None, ...] indexed by subpop order --
    # the per-subpop companion to _schedule_overrides_from_json above.
    _lists = SCHEDULE_OVERRIDES_PER_SUBPOP.get(scenario_name)
    if not _lists:
        return None
    _csvs = _load_schedule_csvs()
    _entries = (_csvs.get("__scenario_overrides__", {}) or {}).get("per_subpop", {}).get(scenario_name, [])
    _out = []
    for _si, _attrs in enumerate(_lists):
        if not _attrs:
            _out.append(None)
            continue
        _sp_entry = _entries[_si] if _si < len(_entries) else {}
        _out.append({_a: _sp_entry[_a] for _a in _attrs if _a in _sp_entry} or None)
    return _out if any(_out) else None


def _build_schedules(start_date, num_days, tv_increments=None, dose_mult=None, schedule_overrides=None):
    # dose_mult: per-age-group multiplier on the `vaccine_schedule`-template
    # schedules (see scale_dose_schedule_df and _dose_mult_for above) -- e.g.
    # 0 to zero out an age group for a no-dose scenario.
    # Despite the "daily_vaccines" naming (fixed by the model config schema),
    # this isn't vaccine-specific -- it scales whatever `scheduled_exact`
    # represents (antiviral prophylaxis courses, etc.).
    # schedule_overrides: {df_attribute: csv_text} -- REPLACES the named
    # schedule outright (see _schedule_overrides_from_json above), checked
    # before the notebook's own uploaded schedules.json so a scenario's
    # replacement wins; dose_mult scaling above still applies on top of it.
    _h = max(num_days + 14, 370)
    _dates = pd.date_range(start=start_date, periods=_h, freq="D").date
    _mob = json.dumps(np.ones((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    _vax = json.dumps(np.zeros((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    _csvs = _load_schedule_csvs()
    _overrides = schedule_overrides or {}

    def _real_or(_name, _fallback):
        if _name in _overrides:
            return pd.read_csv(io.StringIO(_overrides[_name]))
        if _name in _csvs:
            return pd.read_csv(io.StringIO(_csvs[_name]))
        return _fallback

    _kwargs = {}
    _tvm_df = _build_tvm_df(start_date, num_days, tv_increments)
    if _tvm_df is not None:
        _kwargs["transmission_multiplier_df"] = _tvm_df
    # Any vaccine_schedule-template schedule other than the default-named
    # "vaccinated_transfer_schedule"/"daily_vaccines" gets its df loaded (or
    # falls back to an all-zero constant) and scaled by its OWN entry in
    # dose_mult (see _dose_mult_for).
    _extra_attrs = sorted({
        _sc.get("schedule_config", {}).get("df_attribute")
        for _sc in config_dict.get("schedules", []) or []
        if _sc.get("schedule_template") == "vaccine_schedule"
        and _sc.get("schedule_config", {}).get("df_attribute") not in (None, "daily_vaccines_df")
    })
    for _attr in _extra_attrs:
        _kwargs[_attr] = scale_dose_schedule_df(_real_or(
            _attr,
            pd.DataFrame({"date": _dates, "daily_vaccines": [_vax] * _h})),
            _dose_mult_for(dose_mult, _attr))
    return SimpleNamespace(
        absolute_humidity_df=_real_or(
            "absolute_humidity_df",
            pd.DataFrame({"date": _dates, "absolute_humidity": [0.01] * _h})),
        school_work_calendar_df=_real_or(
            "school_work_calendar_df",
            pd.DataFrame({"date": _dates, "is_school_day": [1.0] * _h, "is_work_day": [1.0] * _h})),
        mobility_df=_real_or(
            "mobility_df",
            pd.DataFrame({"day_of_week": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], "mobility_modifier": [_mob]*7})),
        daily_vaccines_df=scale_dose_schedule_df(_real_or(
            "daily_vaccines_df",
            pd.DataFrame({"date": _dates, "daily_vaccines": [_vax] * _h})),
            _dose_mult_for(dose_mult, "daily_vaccines_df")),
        **_kwargs,
    )


def build_model(cfg, param_overrides, rep, subpop_overrides=None,
                seed_scales=None, tv_increments=None,
                dose_mult=None, dose_mult_per_subpop=None,
                schedule_overrides=None, schedule_overrides_per_subpop=None):
    # subpop_overrides: list of {param: value} indexed by subpop order (metapop only)
    # seed_scales / tv_increments default to the fitted best set's; a sampled
    # param set passes its own so each run reproduces that draw's initial
    # conditions and m(t) trajectory.
    # dose_mult / dose_mult_per_subpop: per-age-group multiplier(s) on a
    # `scheduled_exact` transition's daily schedule (see
    # scale_dose_schedule_df), mirroring the Analysis tab's schedule-scaling
    # control. dose_mult applies to every subpop identically (or is the only
    # one used, single-population); dose_mult_per_subpop (metapop only) is a
    # list indexed by subpop order whose non-None entries override dose_mult
    # for that subpop.
    _seed_scales = SEED_SCALES if seed_scales is None else seed_scales
    _tv_incr = TV_INCREMENTS if tv_increments is None else tv_increments
    _cfg = copy.deepcopy(cfg)
    if param_overrides:
        _cfg["params"] = {**_cfg.get("params", {}), **param_overrides}
    if _tv_incr:
        _cfg, _ = _inject_tv_transmission(_cfg)

    if IS_METAPOP:
        # Metapop: reuse the folder-driven factory, which reads
        # metapop_config.json, per-subpop schedule CSVs, travel matrix and
        # per-subpop initial conditions. The scenario's shared overrides are
        # already merged into _cfg["params"] above (applied to every subpop);
        # subpop_overrides supplies per-subpop values. The fitted m(t) is
        # reconstructed once and broadcast identically to all subpops.
        _comps = list(_cfg.get("compartments", {}).keys()) if isinstance(_cfg.get("compartments"), dict) else list(_cfg.get("compartments", ["S"]))
        if _seed_scales:
            print(
                "Warning: seed-scale reproduction is not applied for metapop export "
                "(per-subpop seeded initial conditions are read from the metapop folder); "
                f"ignoring fitted seed scales {sorted(_seed_scales)}."
            )
        # Split the scenario's dose multiplier per schedule: the factory
        # takes the default schedule's vector as `dose_mult` and the rest
        # as an `extra_dose_mult` dict. _dose_mult_for accepts either a
        # bare per-age-group list (same scaling for every schedule) or a
        # {df_attribute: list} dict (per-schedule scaling).
        _extra_attrs = sorted(config_schedule_df_attributes(_cfg) - {"daily_vaccines_df"})
        _base_dose_mult = _dose_mult_for(dose_mult, "daily_vaccines_df")
        _extra_dose_mult = (
            {_attr: _dose_mult_for(dose_mult, _attr) for _attr in _extra_attrs}
            if (_extra_attrs and dose_mult is not None) else None
        )
        _base_dm_per_sp = _extra_dm_per_sp = None
        if dose_mult_per_subpop:
            _base_dm_per_sp = [
                _dose_mult_for(_dm, "daily_vaccines_df") for _dm in dose_mult_per_subpop
            ]
            _extra_dm_per_sp = [
                ({_attr: _dose_mult_for(_dm, _attr) for _attr in _extra_attrs}
                 if (_extra_attrs and _dm is not None) else None)
                for _dm in dose_mult_per_subpop
            ]
        # Uploaded schedule REPLACEMENTS (as opposed to dose_mult, which
        # scales a schedule in place): csv_text -> DataFrame, since
        # make_metapop_from_folder's schedule_df_overrides kwarg takes
        # actual DataFrames, not raw CSV text.
        _sched_ov_dfs = (
            {_a: pd.read_csv(io.StringIO(_t)) for _a, _t in schedule_overrides.items()}
            if schedule_overrides else None
        )
        _sched_ov_dfs_per_sp = (
            [
                ({_a: pd.read_csv(io.StringIO(_t)) for _a, _t in _d.items()} if _d else None)
                for _d in schedule_overrides_per_subpop
            ]
            if schedule_overrides_per_subpop else None
        )
        _m, _ = make_metapop_from_folder(
            METAPOP_FOLDER, _cfg, START_DATE, NUM_DAYS, _comps,
            seed_offset=rep, seed_base=SEED_BASE, ts_per_day=TIMESTEPS_PER_DAY,
            stochastic=RUN_STOCHASTIC, save_daily=True, tvs=TRANSITION_VARS,
            param_overrides=None,
            param_overrides_per_subpop=subpop_overrides,
            travel_config=METAPOP_TRAVEL_CONFIG or None,
            num_age_groups=NUM_AGE_GROUPS, num_risk_groups=NUM_RISK_GROUPS,
            transmission_multiplier_df=_build_tvm_df(START_DATE, NUM_DAYS, _tv_incr),
            dose_mult=_base_dose_mult, dose_mult_per_subpop=_base_dm_per_sp,
            extra_dose_mult=_extra_dose_mult, extra_dose_mult_per_subpop=_extra_dm_per_sp,
            schedule_df_overrides=_sched_ov_dfs, schedule_df_overrides_per_subpop=_sched_ov_dfs_per_sp,
        )
        return _m

    _sched = _build_schedules(START_DATE, NUM_DAYS, _tv_incr, dose_mult, schedule_overrides=schedule_overrides)
    _mc = parse_model_config_from_dict(_cfg, schedules_input=_sched)
    _A, _R = NUM_AGE_GROUPS, NUM_RISK_GROUPS
    _comps = list(_cfg.get("compartments", {}).keys()) if isinstance(_cfg.get("compartments"), dict) else list(_cfg.get("compartments", ["S"]))
    _first = _comps[0] if _comps else "S"
    _N = _cfg.get("total_population", 100000)
    # Seeded compartments (e.g. an initial E count) from the Builder's Step 6
    # initial-conditions table, same source Analysis/Forecast read -- not just
    # everyone starting in the first compartment.
    _ic_entry = (_cfg.get("initial_conditions", {}) or {}).get("aggregate_pop", {})
    if _ic_entry:
        _pop_arr = np.asarray(_ic_entry.get("population", np.full((_A, _R), float(_N))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in _comps
        }
        _comp_init, _ = build_compartment_init(_seed_arrays, _pop_arr, _comps)
    else:
        _comp_init = {_first: np.full((_A, _R), float(_N))}
        for _c in _comps[1:]:
            _comp_init.setdefault(_c, np.zeros((_A, _R)))
    if _seed_scales:
        _comp_init = _scale_compartment_init(_comp_init, _seed_scales, _comps, _A, _R)
    _state = build_state_from_config(_mc, _comp_init, epi_metric_init={})
    _params = build_params_from_config(_mc, num_age_groups=_A, num_risk_groups=_R)
    _tt = clt.TransitionTypes.BINOM if RUN_STOCHASTIC else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    _settings = clt.SimulationSettings(
        timesteps_per_day=TIMESTEPS_PER_DAY, transition_type=_tt,
        start_real_date=START_DATE, save_daily_history=True,
        transition_variables_to_save=TRANSITION_VARS,
    )
    _subpop = ConfigDrivenSubpopModel(
        model_config=_mc, state_init=_state, params=_params,
        simulation_settings=_settings, RNG=np.random.default_rng(SEED_BASE + rep),
        schedules_input=_sched, name="pop",
    )
    _mixing = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
    return ConfigDrivenMetapopModel(
        subpop_models=[_subpop], mixing_params=_mixing,
        model_config=_mc, travel_config={},
    )


# ---- Uncertainty source: which parameter set each replicate runs with ----
# Mirrors the Analysis tab. "transitions" (and every deterministic run) uses
# the fitted best set -- already merged into config_dict["params"] and into
# each scenario's overrides -- so replicates differ only in the transition
# RNG. Otherwise draw NUM_PARAM_SETS sets at random without replacement:
# "parameters" runs each drawn set exactly once (transitions are
# deterministic, so a repeated set would only duplicate its trajectory --
# NUM_REPS is ignored), "parameters+transitions" spreads NUM_REPS replicates
# evenly across the drawn sets (remainder distributed at random).
# fitted_params_MA_single_age.json supplies 495 accepted sets here, so the
# sampling branch below is the one that actually runs.
_param_only = UNCERTAINTY_SOURCE == "parameters"
_use_psets = (
    STOCHASTIC
    and UNCERTAINTY_SOURCE in ("parameters", "parameters+transitions")
    and len(PARAM_SETS) > 1
)
if _use_psets:
    _rng_sched = np.random.default_rng(SEED_BASE)
    _k = min(int(NUM_PARAM_SETS), len(PARAM_SETS), *(() if _param_only else (NUM_REPS,)))
    _sel = _rng_sched.choice(len(PARAM_SETS), size=_k, replace=False)
    RUN_PARAM_SETS = [PARAM_SETS[int(_i)] for _i in _sel]
    if _param_only:
        RUN_SCHEDULE = list(range(_k))
    else:
        _base_r, _extra_r = divmod(NUM_REPS, _k)
        RUN_SCHEDULE = [_i for _i in range(_k) for _ in range(_base_r)]
        if _extra_r:
            RUN_SCHEDULE += [int(_i) for _i in _rng_sched.choice(_k, size=_extra_r, replace=False)]
    if __name__ == "__main__":
        # This module-level block runs again on import in every worker
        # process (spawn re-imports the script as __main__; fork re-runs
        # it too under some pool implementations) since RUN_PARAM_SETS/
        # RUN_SCHEDULE must exist there for _apply_pset. Only print once,
        # in the parent.
        print(
            f"Parameter uncertainty: {_k} set(s) sampled from {len(PARAM_SETS)} accepted, "
            f"{len(RUN_SCHEDULE)} run(s) per scenario"
            + (" (deterministic transitions)" if _param_only else "")
        )
else:
    if (
        __name__ == "__main__"
        and UNCERTAINTY_SOURCE in ("parameters", "parameters+transitions")
        and STOCHASTIC
    ):
        _fallback = (
            "a single deterministic run of the best parameter set"
            if _param_only
            else "transition-only uncertainty with the best parameter set"
        )
        print(
            f"Warning: UNCERTAINTY_SOURCE is '{UNCERTAINTY_SOURCE}' but no fitted_params.json "
            f"with 2+ accepted_params is configured -- falling back to {_fallback}."
        )
    RUN_PARAM_SETS = []
    RUN_SCHEDULE = [None] * (NUM_REPS if RUN_STOCHASTIC else 1)


def _apply_pset(overrides, pset_idx, designed, ratios=None):
    # Layer a sampled param set under the scenario's overrides: the set
    # supplies every fitted param the scenario design did not deliberately
    # set (DESIGNED_PARAMS), so the ensemble carries parameter uncertainty
    # without undoing the sweep or the edited scenario values. Returns the
    # (param_overrides, seed_scales, tv_increments) triple for this run.
    # A designed param with a known ratio in DESIGNED_PARAM_RATIOS (the
    # scenario's override / catalog baseline at export time) instead has
    # that ratio applied to the sampled draw's own value, so the
    # scenario's scaling factor carries through posterior uncertainty
    # rather than pinning one absolute value across every draw.
    if pset_idx is None:
        return overrides, None, None
    _model, _scales, _incr = _split_pset(RUN_PARAM_SETS[pset_idx])
    _ov = dict(overrides or {})
    _ratios = ratios or {}
    for _k2, _v2 in _model.items():
        if _k2 in designed:
            _ratio = _ratios.get(_k2)
            if _ratio is None:
                continue
            if isinstance(_ratio, list):
                _ov[_k2] = (np.asarray(_ratio, dtype=float) * np.asarray(_v2, dtype=float)).tolist()
            else:
                _ov[_k2] = float(_ratio) * float(_v2)
            continue
        _ov[_k2] = _v2
    return _ov, _scales, _incr


def _run_one(scenario_name, overrides, designed, dose_mult, dose_mult_per_subpop,
             ratios, sp_overrides, rep, pset_idx):
    # One (scenario, replicate) unit of work -- everything a pool worker
    # needs is passed in as plain, picklable arguments; module globals
    # used inside (config_dict, NUM_DAYS, TRANSITION_VARS, RUN_PARAM_SETS
    # via _apply_pset, ...) are rebuilt identically in every worker
    # process when it re-imports this script, so they don't need to be
    # passed explicitly.
    _run_ov, _run_scales, _run_incr = _apply_pset(overrides, pset_idx, designed, ratios)
    _m = build_model(
        config_dict, _run_ov, rep, subpop_overrides=sp_overrides,
        seed_scales=_run_scales, tv_increments=_run_incr,
        dose_mult=dose_mult, dose_mult_per_subpop=dose_mult_per_subpop,
        schedule_overrides=_schedule_overrides_from_json(scenario_name),
        schedule_overrides_per_subpop=_schedule_overrides_per_subpop_from_json(scenario_name),
    )
    _m.simulate_until_day(NUM_DAYS)
    _sps = list(_m.subpop_models.values())
    _comp_names = list(_sps[0].compartments.keys())
    # extract_history sums over age/risk and subpops, and -- critically --
    # aggregates transition variables from per-sub-timestep to per-day.
    # Transition history is saved once per sub-timestep, so summing it
    # raw would give a series TIMESTEPS_PER_DAY x too long and each
    # value TIMESTEPS_PER_DAY x too small.
    _h = extract_history(_m, _comp_names, tvs=TRANSITION_VARS)
    # Fully-resolved companion to _h -- same compartments/transitions,
    # but per-subpop (day, age_group, risk_group) arrays instead of
    # (day,) scalars summed over subpop/age/risk. Needed by anything
    # that cares which subpop/age/risk group a flow happened in --
    # trivial here since num_age_groups = num_risk_groups = 1, but kept
    # for parity with results_explorer_notebook.py's readers.
    _h_full = extract_history_full(_m, _comp_names, tvs=TRANSITION_VARS)
    _kinds = {_k: ("transition" if _k in TRANSITION_VARS else "compartment") for _k in _h}
    return scenario_name, rep, pset_idx, _h, _h_full, _kinds


# ---- Everything below only runs in the parent process. ----
# Guarding it is required (not just tidy) on spawn-based multiprocessing
# (the default on macOS and Windows): each pool worker re-imports this
# file as __main__, so without this guard every worker would recursively
# try to build its own pool and re-run the whole scenario sweep. Under
# fork-based multiprocessing (Linux's default) it isn't strictly
# required, but keeping it makes the script behave identically on all
# three platforms.
if __name__ == "__main__":
    # Load the schedule CSVs once up front, in this process, so their
    # notices (stale schedules.json, missing CSV) print once here --
    # pool workers load their own copy silently.
    _load_schedule_csvs()
    _tasks = []
    for scenario_name, overrides in SCENARIOS.items():
        _sp_overrides = SUBPOP_PARAM_OVERRIDES.get(scenario_name)
        # A scenario absent from DESIGNED_PARAMS was added by hand, so
        # nothing says which of its params are deliberate -- protect all
        # of them rather than silently letting a sampled set overwrite
        # the values the user just wrote. An explicit (possibly empty)
        # list always wins.
        if scenario_name in DESIGNED_PARAMS:
            _designed = set(DESIGNED_PARAMS[scenario_name])
        else:
            _designed = set(overrides or {})
        _dose_mult = DOSE_MULTIPLIER.get(scenario_name)
        _dose_mult_per_subpop = DOSE_MULTIPLIER_PER_SUBPOP.get(scenario_name)
        _ratios = DESIGNED_PARAM_RATIOS.get(scenario_name, {})
        # Seed by position in the run schedule so every run gets a
        # distinct RNG stream once replicates are spread across
        # parameter sets.
        for _rep, _pset_idx in enumerate(RUN_SCHEDULE):
            _tasks.append((
                scenario_name, overrides, _designed, _dose_mult,
                _dose_mult_per_subpop, _ratios, _sp_overrides, _rep, _pset_idx,
            ))

    _n_workers = NUM_WORKERS or os.cpu_count() or 1
    _n_workers = min(_n_workers, len(_tasks)) or 1
    print(f"Running {len(_tasks)} simulation(s) across {_n_workers} worker process(es)")

    all_results = {_scen: [] for _scen in SCENARIOS}
    if _n_workers == 1:
        # Serial fallback (NUM_WORKERS = 1) -- also handy for debugging,
        # since a worker-process traceback is otherwise harder to inspect.
        for _t in _tasks:
            _scen, _rep, _psi, _h, _h_full, _kinds = _run_one(*_t)
            all_results[_scen].append((_rep, _h, _h_full, _kinds, _psi))
    else:
        with ProcessPoolExecutor(max_workers=_n_workers) as _ex:
            _futures = {_ex.submit(_run_one, *_t): _t for _t in _tasks}
            for _fut in as_completed(_futures):
                _scen_name = _futures[_fut][0]
                _scen, _rep, _psi, _h, _h_full, _kinds = _fut.result()
                print(f"Finished scenario={_scen_name} rep={_rep}")
                all_results[_scen].append((_rep, _h, _h_full, _kinds, _psi))

    # Pool completion order is nondeterministic; sort each scenario's
    # replicates back into run-schedule order so the `rep` column
    # written to results_parquet/ below matches what a serial run would
    # have produced.
    for _scen in all_results:
        all_results[_scen] = [
            (_h, _h_full, _kinds, _psi)
            for (_rep, _h, _h_full, _kinds, _psi) in sorted(all_results[_scen], key=lambda _r: _r[0])
        ]

    # Every run of this script writes a FRESH results directory, never
    # appends to one already on disk: two runs sharing one would both
    # write rep=0, rep=1, ... under the same scenario names, and a
    # reader grouping by (scenario, rep, day) would silently blend the
    # two runs' rows together into nonsense (this bit a real user: a
    # stale results.db from an earlier, differently-configured run
    # stayed on disk and got averaged in with a later correct run). So
    # results_parquet/ is used only while that exact name is free; once
    # it exists, a timestamped results_{timestamp}_parquet/ is used
    # instead (with a numeric suffix in the unlikely case two runs start
    # in the same second), so nothing already on disk is ever appended to.
    _out_dir = OUTPUT_DIR / "results_parquet"
    if _out_dir.exists():
        _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _out_dir = OUTPUT_DIR / f"results_{_stamp}_parquet"
        _suffix = 1
        while _out_dir.exists():
            _out_dir = OUTPUT_DIR / f"results_{_stamp}_{_suffix}_parquet"
            _suffix += 1
        print(f"{OUTPUT_DIR / 'results_parquet'} already exists -- writing to {_out_dir} instead")
    # Staged in a native DuckDB table first, then exported to Parquet in
    # one shot at the end (results_io.write_results_parquet) -- DuckDB is a
    # vectorized engine tuned for bulk columnar loads, not row-by-row
    # inserts (a plain executemany measured ~10s per 100k rows), so rows
    # are appended per array as a small DataFrame via
    # `con.append(table, df)`, which does the same insert as one bulk
    # write. The staging file is removed once the Parquet export below
    # succeeds; if the process dies mid-run, its presence marks the
    # Parquet directory as incomplete rather than silently half-written.
    _stage_path = OUTPUT_DIR / f"{_out_dir.name}.stage.duckdb"
    _stage_path.unlink(missing_ok=True)
    _con = duckdb.connect(str(_stage_path))
    results_io.create_results_tables(_con)
    print(f"Writing results for {len(all_results)} scenario(s) to {_out_dir}")
    for _si, (_scen, _reps_data) in enumerate(all_results.items(), start=1):
        _n_reps = len(_reps_data)
        for _ri, (_h, _h_full, _kinds, _psi) in enumerate(_reps_data):
            for _c, _arr in _h.items():
                _con.append("results", pd.DataFrame({
                    "scenario": _scen, "rep": _ri, "param_set": _psi,
                    "compartment": _c, "kind": _kinds[_c],
                    "day": np.arange(1, len(_arr) + 1),
                    "value": np.asarray(_arr, dtype=float),
                }))
            for _c, _sp_map in _h_full.items():
                for _spname, _arr_full in _sp_map.items():
                    # Vectorized index construction (numpy) instead of a
                    # triple-nested Python for-loop over day/age_group/
                    # risk_group -- meaningfully faster once that product
                    # runs into the hundreds of thousands of rows across
                    # many stochastic replicates.
                    _d_idx, _a_idx, _r_idx = np.indices(_arr_full.shape)
                    _con.append("results_full", pd.DataFrame({
                        "scenario": _scen, "rep": _ri, "param_set": _psi,
                        "compartment": _c, "kind": _kinds[_c], "subpop": _spname,
                        "age_group": _a_idx.ravel(), "risk_group": _r_idx.ravel(),
                        "day": _d_idx.ravel() + 1,
                        "value": _arr_full.ravel().astype(float),
                    }))
            # Progress within a scenario's replicates, so a long
            # stochastic run (hundreds of reps) doesn't look stuck
            # between per-scenario lines below.
            if _n_reps > 20 and (_ri + 1) % 20 == 0:
                print(f"  [{_si}/{len(all_results)}] {_scen}: wrote {_ri + 1}/{_n_reps} replicate(s)")
        print(f"[{_si}/{len(all_results)}] {_scen}: wrote {_n_reps} replicate(s)")
    # Run-level metadata the result rows themselves cannot carry: they
    # only hold day indices and 0-based age/risk indices, so without this
    # a reader (e.g. results_explorer_notebook.py) cannot plot real dates
    # or label age bands. It also saves the reader from recovering the
    # scenario/subpop lists with SELECT DISTINCT, which on a large
    # results set means a full scan -- measured at 7s over a 73M-row
    # `results` and 49s over its `results_full`, just to list values that
    # were known here at write time (back when this was SQLite).
    #
    # Key/value JSON so keys can be added later without migrating the
    # schema. Readers must tolerate meta.json being absent (files written
    # before it existed) and individual keys being missing.
    _subpop_names = sorted({
        _spname
        for _reps_data in all_results.values()
        for (_h, _h_full, _kinds, _psi) in _reps_data
        for _sp_map in _h_full.values()
        for _spname in _sp_map
    })
    _meta = {
        "schema_version": 1,
        "source": "run_simulation_script",
        "start_date": START_DATE,
        "num_days": NUM_DAYS,
        "timesteps_per_day": TIMESTEPS_PER_DAY,
        "stochastic": RUN_STOCHASTIC,
        "uncertainty_source": UNCERTAINTY_SOURCE,
        "num_age_groups": NUM_AGE_GROUPS,
        "num_risk_groups": NUM_RISK_GROUPS,
        "age_group_labels": (config_dict.get("age_risk", {}) or {}).get("age_groups"),
        "subpop_names": _subpop_names,
        # Definition order, which SELECT DISTINCT would lose -- it decides
        # scenario colour assignment and which scenario reads as baseline.
        "scenarios": list(SCENARIOS.keys()),
        # Order-bearing too: model_config.json's compartment order followed
        # by its transition order, matching the metric lists the Model
        # Builder's own selectors show. SELECT DISTINCT would return these
        # alphabetically, scrambling a model's natural S -> E -> I -> H -> R
        # reading order. Keep as-is; do not sort.
        "compartments": [
            _c for _c in (
                list(config_dict.get("compartments", {}).keys())
                if isinstance(config_dict.get("compartments"), dict)
                else list(config_dict.get("compartments", []))
            )
        ],
        "transition_vars": list(TRANSITION_VARS),
        "n_reps": len(RUN_SCHEDULE),
        "param_set_indices": list(RUN_SCHEDULE),
    }
    print(f"Writing Parquet to {_out_dir}")
    results_io.write_results_parquet(_con, _out_dir, meta=_meta)
    _con.close()
    _stage_path.unlink(missing_ok=True)
    print(f"Results saved to {_out_dir}")
