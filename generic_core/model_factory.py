"""
model_factory.py — Build ConfigDrivenMetapopModel instances from a plain
config dict, with no marimo/notebook dependency.

Extracted from generic_core/examples/_nb_shared.py and _nb_shared_factory.py
so that the model-building logic can be reused outside the notebook (e.g. by
generic_core/fitting.py and by exported standalone scripts).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import clt_toolkit as clt
import flu_core as flu

from generic_core.config_parser import parse_model_config_from_dict
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel


def build_scalar_array(value, num_age_groups: int = 1, num_risk_groups: int = 1) -> np.ndarray:
    """Return an (A×R) array filled with ``value``."""
    return np.full((num_age_groups, num_risk_groups), float(value), dtype=float)


def build_notebook_schedules_input(
    start_date,
    num_days: int,
    absolute_humidity: float,
    mobility_value: float,
    daily_vaccines_value,  # float, or an A×R nested list to vary by age/risk group
    num_age_groups: int = 1,
    num_risk_groups: int = 1,
    absolute_humidity_df=None,
    school_work_calendar_df=None,
    mobility_df=None,
    daily_vaccines_df=None,
    transmission_multiplier_df=None,
) -> SimpleNamespace:
    """Assemble a schedules-input namespace for the model from the per-schedule
    DataFrames, falling back to constant-valued DataFrames where a df is None."""
    _horizon = max(int(num_days) + 14, 370)
    _dates = pd.date_range(start=start_date, periods=_horizon, freq="D").date

    _ah_df = absolute_humidity_df
    if _ah_df is None:
        _ah_df = pd.DataFrame({
            "date": _dates,
            "absolute_humidity": [float(absolute_humidity)] * _horizon,
        })

    _cal_df = school_work_calendar_df
    if _cal_df is None:
        _cal_df = pd.DataFrame({
            "date": _dates,
            "is_school_day": [1.0] * _horizon,
            "is_work_day": [1.0] * _horizon,
        })

    _mob_payload = json.dumps(
        np.full((num_age_groups, num_risk_groups), float(mobility_value)).tolist()
    )
    _mob_df = mobility_df
    if _mob_df is None:
        _mob_df = pd.DataFrame({
            "day_of_week": [
                "monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday",
            ],
            "mobility_modifier": [_mob_payload] * 7,
        })

    if isinstance(daily_vaccines_value, (list, tuple)):
        _vax_arr = np.asarray(daily_vaccines_value, dtype=float)
    else:
        _vax_arr = np.full((num_age_groups, num_risk_groups), float(daily_vaccines_value))
    _vax_payload = json.dumps(_vax_arr.tolist())
    _vax_df = daily_vaccines_df
    if _vax_df is None:
        _vax_df = pd.DataFrame({
            "date": _dates,
            "daily_vaccines": [_vax_payload] * _horizon,
        })

    # Time-varying transmission multiplier m(t); constant 1.0 (no effect) when
    # not supplied. Only referenced by the model when a FOI rate_config sets
    # 'transmission_multiplier_schedule' (see rate_templates._beta_adjusted_*).
    _tvm_df = transmission_multiplier_df
    if _tvm_df is None:
        _tvm_df = pd.DataFrame({
            "date": _dates,
            "transmission_multiplier": [1.0] * _horizon,
        })

    return SimpleNamespace(
        absolute_humidity_df=_ah_df,
        school_work_calendar_df=_cal_df,
        mobility_df=_mob_df,
        daily_vaccines_df=_vax_df,
        transmission_multiplier_df=_tvm_df,
    )


def config_schedule_df_attributes(config) -> set[str]:
    """Names of the ``*_df`` schedule attributes referenced by ``config``.

    Every ``schedule_config`` entry whose key ends in ``df_attribute`` names a
    DataFrame the schedule reads its values from (e.g. ``df_attribute:
    absolute_humidity_df``, ``school_work_day_df_attribute:
    school_work_calendar_df``)."""
    _out = set()
    for _s in (config or {}).get("schedules", []) or []:
        for _k, _v in (_s.get("schedule_config", {}) or {}).items():
            if _k.endswith("df_attribute") and isinstance(_v, str):
                _out.add(_v)
    return _out


def warn_missing_schedule_dfs(config, schedule_dfs, context: str = ""):
    """Warn when ``config`` references a schedule DataFrame that wasn't supplied.

    ``build_notebook_schedules_input`` silently substitutes a flat constant
    series for any missing df (humidity 0, no vaccination, every day a
    school/work day). That fallback is correct for a model that genuinely has
    no such input, but when the config *does* reference the schedule it
    quietly simulates a different model than intended — e.g. dropping a fitted
    vaccination schedule leaves the whole population susceptible. Callers that
    have real schedule DataFrames available should always pass them; this warns
    rather than raises so configs that intentionally run flat still work.
    """
    _referenced = config_schedule_df_attributes(config)
    if not _referenced:
        return []
    _missing = sorted(
        _a for _a in _referenced
        if getattr(schedule_dfs, _a, None) is None
    )
    if _missing:
        _where = f" ({context})" if context else ""
        warnings.warn(
            f"Schedule input(s) {_missing} are referenced by the model config but "
            f"no DataFrame was supplied{_where}; falling back to flat constant "
            f"values (humidity 0.0, no vaccination, every day a school/work day). "
            f"The simulated model will differ from one built with the real "
            f"schedule CSVs. Pass schedule_dfs=<loaded schedules> to fix.",
            RuntimeWarning,
            stacklevel=3,
        )
    return _missing


def build_compartment_init(seed_arrays, population_AR, compartments):
    """Build a {compartment: A×R array} init dict from seed counts + population.

    ``seed_arrays`` maps non-first compartment names to A×R arrays of seeded
    counts. The first compartment receives ``population − Σ seeds`` per cell
    (clamped at 0). Returns ``(comp_init, overflow)`` where ``overflow`` is
    True if any cell's seeds exceeded its population."""
    _pop = np.asarray(population_AR, dtype=float)
    _seed_total = np.zeros_like(_pop)
    _comp_init = {}
    for _c in compartments[1:] if len(compartments) > 1 else []:
        _arr = np.asarray(seed_arrays.get(_c, np.zeros_like(_pop)), dtype=float)
        _comp_init[_c] = _arr
        _seed_total = _seed_total + _arr
    _remainder = _pop - _seed_total
    _overflow = bool(np.any(_remainder < 0))
    if compartments:
        _comp_init[compartments[0]] = np.clip(_remainder, 0.0, None)
    for _c in compartments:
        _comp_init.setdefault(_c, np.zeros_like(_pop))
    return _comp_init, _overflow


def read_initial_conditions(config, subpop_name, compartments,
                            num_age_groups, num_risk_groups):
    """Read per-subpop initial conditions from ``config['initial_conditions']``.

    Returns a ``{compartment: A×R array}`` init dict (first compartment =
    population − Σ seeds), or None when the subpop has no entry. Used by the
    metapop and shared-factory run paths, with the folder JSON as fallback."""
    _ic = (config or {}).get("initial_conditions", {})
    _entry = _ic.get(subpop_name)
    if not _entry:
        return None
    _A = int(num_age_groups)
    _R = int(num_risk_groups)
    _pop = np.asarray(_entry.get("population",
                                 np.zeros((_A, _R))), dtype=float)
    _seeds = {}
    for _c, _arr in (_entry.get("seeds", {}) or {}).items():
        if _c in compartments:
            _seeds[_c] = np.asarray(_arr, dtype=float)
    _comp_init, _ = build_compartment_init(_seeds, _pop, compartments)
    return _comp_init


def make_single_pop_metapop(
    config, start_date, num_days, compartment_init,
    seed_offset=0, seed_base=0, ts_per_day=7, stochastic=False,
    tvs=None, save_daily=True, epi_metric_init=None, param_overrides=None,
    travel_config=None,
    num_age_groups: int = 1, num_risk_groups: int = 1,
    mobility_value: float = 1.0, daily_vaccines_value=0.0,
    schedule_dfs: SimpleNamespace | None = None,
):
    """Build a single-subpopulation ConfigDrivenMetapopModel from a config dict.

    ``schedule_dfs`` (optional) supplies per-schedule DataFrames (as produced
    by ``build_notebook_schedules_input``'s ``*_df`` fields); when omitted,
    constant-valued schedules are generated from ``mobility_value`` /
    ``daily_vaccines_value``.
    """
    _A, _R = num_age_groups, num_risk_groups
    _cfg = config
    if param_overrides:
        _cfg = dict(config)
        _cfg["params"] = {**config.get("params", {}), **param_overrides}
    _sched_dfs = schedule_dfs or SimpleNamespace(
        absolute_humidity_df=None, school_work_calendar_df=None,
        mobility_df=None, daily_vaccines_df=None,
    )
    warn_missing_schedule_dfs(_cfg, _sched_dfs, context="make_single_pop_metapop")
    _sched = build_notebook_schedules_input(
        start_date=start_date, num_days=num_days,
        absolute_humidity=0.0, mobility_value=mobility_value,
        daily_vaccines_value=daily_vaccines_value,
        num_age_groups=_A, num_risk_groups=_R,
        absolute_humidity_df=_sched_dfs.absolute_humidity_df,
        school_work_calendar_df=_sched_dfs.school_work_calendar_df,
        mobility_df=_sched_dfs.mobility_df,
        daily_vaccines_df=_sched_dfs.daily_vaccines_df,
        transmission_multiplier_df=getattr(_sched_dfs, "transmission_multiplier_df", None),
    )
    _mc = parse_model_config_from_dict(_cfg, schedules_input=_sched)
    _state = build_state_from_config(_mc, compartment_init, epi_metric_init=epi_metric_init or {})
    _params = build_params_from_config(_mc, num_age_groups=_A, num_risk_groups=_R)
    _tt = clt.TransitionTypes.BINOM if stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    _settings = clt.SimulationSettings(
        timesteps_per_day=ts_per_day, transition_type=_tt,
        start_real_date=start_date, save_daily_history=save_daily,
        transition_variables_to_save=tvs or [],
    )
    _subpop = ConfigDrivenSubpopModel(
        model_config=_mc, state_init=_state, params=_params,
        simulation_settings=_settings,
        RNG=np.random.default_rng(seed_base + seed_offset),
        schedules_input=_sched, name="pop",
    )
    _mixing = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
    return ConfigDrivenMetapopModel(
        subpop_models=[_subpop], mixing_params=_mixing,
        model_config=_mc, travel_config=travel_config or {},
    ), _mc, _sched


def make_metapop_from_folder(
    folder_path, config, start_date, num_days, compartments_list,
    seed_offset=0, seed_base=0, ts_per_day=7, stochastic=False,
    tvs=None, save_daily=True, param_overrides=None, travel_config=None,
    param_overrides_per_subpop=None, init_states_override=None,
    num_age_groups: int = 1, num_risk_groups: int = 1,
    mobility_value: float = 1.0, daily_vaccines_value=0.0,
    transmission_multiplier_df=None,
):
    """Build a multi-subpopulation ConfigDrivenMetapopModel from a metapop
    input folder (``metapop_config.json`` + per-subpop schedule/IC files).

    ``transmission_multiplier_df`` (optional), if supplied, is broadcast as
    the same m(t) schedule to every subpopulation — e.g. to apply a
    single-population-fitted time-varying transmission trajectory uniformly
    across all subpops. ``config`` must already have the
    'transmission_multiplier' schedule/rate_config wired in (see
    ``generic_core.fitting._inject_tv_transmission``) for this to take effect."""
    _A, _R = num_age_groups, num_risk_groups
    _folder = Path(folder_path)
    with open(_folder / "metapop_config.json") as _f:
        _mc_cfg = json.load(_f)
    _sp_names = list(_mc_cfg["subpopulations"])
    _travel_arr = np.array(_mc_cfg["travel_matrix"], dtype=float)
    _shared_ah = _shared_mob = None
    _ah_p = _folder / "absolute_humidity.csv"
    _mob_p = _folder / "mobility_modifier.csv"
    if _ah_p.exists():
        _shared_ah = pd.read_csv(_ah_p)
        _shared_ah = _shared_ah.loc[:, ~_shared_ah.columns.str.match(r"^Unnamed")]
    if _mob_p.exists():
        _shared_mob = pd.read_csv(_mob_p)
        _shared_mob = _shared_mob.loc[:, ~_shared_mob.columns.str.match(r"^Unnamed")]
    _tt = clt.TransitionTypes.BINOM if stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    _subpops = []
    _mc_ref = None
    for _si, _sp_name in enumerate(_sp_names):
        _cal_df = _vax_df = None
        _cal_p = _folder / f"school_work_calendar_{_sp_name}.csv"
        _vax_p = _folder / f"vaccines_{_sp_name}.csv"
        _ic_p = _folder / f"initial_conditions_{_sp_name}.json"
        if _cal_p.exists():
            _cal_df = pd.read_csv(_cal_p)
            _cal_df = _cal_df.loc[:, ~_cal_df.columns.str.match(r"^Unnamed")]
        if _vax_p.exists():
            _vax_df = pd.read_csv(_vax_p)
            _vax_df = _vax_df.loc[:, ~_vax_df.columns.str.match(r"^Unnamed")]
        warn_missing_schedule_dfs(
            config,
            SimpleNamespace(
                absolute_humidity_df=_shared_ah, school_work_calendar_df=_cal_df,
                mobility_df=_shared_mob, daily_vaccines_df=_vax_df,
                transmission_multiplier_df=transmission_multiplier_df,
            ),
            context=f"subpopulation '{_sp_name}' — missing CSV(s) in {_folder}",
        )
        _sched = build_notebook_schedules_input(
            start_date=start_date, num_days=num_days,
            absolute_humidity=0.0, mobility_value=mobility_value,
            daily_vaccines_value=daily_vaccines_value,
            num_age_groups=_A, num_risk_groups=_R,
            absolute_humidity_df=_shared_ah, school_work_calendar_df=_cal_df,
            mobility_df=_shared_mob, daily_vaccines_df=_vax_df,
            transmission_multiplier_df=transmission_multiplier_df,
        )
        _comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments_list}
        _epi_init = {}
        # Precedence: explicit override > Step 6 tables (with seeds) > folder
        # JSON > population-only table > zeros.
        _ic_cfg = (config.get("initial_conditions", {}) or {}).get(_sp_name, {})
        _has_table_seeds = bool(_ic_cfg.get("seeds"))
        _table_ci = read_initial_conditions(config, _sp_name, compartments_list, _A, _R)
        if init_states_override is not None and _si < len(init_states_override):
            _comp_init, _epi_init = init_states_override[_si]
        elif _has_table_seeds and _table_ci is not None:
            _comp_init = _table_ci
        elif _ic_p.exists():
            with open(_ic_p) as _f:
                _ic = json.load(_f)
            for _c, _arr in _ic.get("compartments", {}).items():
                if _c in compartments_list:
                    _comp_init[_c] = np.array(_arr, dtype=float)
            for _m, _arr in _ic.get("epi_metrics", {}).items():
                _epi_init[_m] = np.array(_arr, dtype=float)
        elif _table_ci is not None:
            _comp_init = _table_ci
        _sp_overrides = dict(param_overrides or {})
        if param_overrides_per_subpop and _si < len(param_overrides_per_subpop) and param_overrides_per_subpop[_si]:
            _sp_overrides.update(param_overrides_per_subpop[_si])
        _sp_overrides.update(config.get("subpop_params", {}).get(_sp_name, {}))
        _cfg = config
        if _sp_overrides:
            _cfg = dict(config)
            _cfg["params"] = {**config.get("params", {}), **_sp_overrides}
        _mc_parsed = parse_model_config_from_dict(_cfg, schedules_input=_sched)
        _state = build_state_from_config(_mc_parsed, _comp_init, epi_metric_init=_epi_init)
        _params = build_params_from_config(_mc_parsed, num_age_groups=_A, num_risk_groups=_R)
        _settings = clt.SimulationSettings(
            timesteps_per_day=ts_per_day, transition_type=_tt,
            start_real_date=start_date, save_daily_history=save_daily,
            transition_variables_to_save=tvs or [],
        )
        _subpop = ConfigDrivenSubpopModel(
            model_config=_mc_parsed, state_init=_state, params=_params,
            simulation_settings=_settings,
            RNG=np.random.default_rng(seed_base + seed_offset + _si),
            schedules_input=_sched, name=_sp_name,
        )
        _subpops.append(_subpop)
        if _mc_ref is None:
            _mc_ref = _mc_parsed
    _mixing = flu.FluMixingParams(travel_proportions=_travel_arr, num_locations=len(_sp_names))
    _kwargs = {}
    if travel_config:
        _kwargs["travel_config"] = travel_config
    return ConfigDrivenMetapopModel(
        subpop_models=_subpops, mixing_params=_mixing, model_config=_mc_ref, **_kwargs
    ), _mc_ref


def extract_history(metapop, comps, tvs=None):
    """Sum per-compartment / per-transition-variable daily history across subpops."""
    _sps = list(metapop.subpop_models.values())
    _out = {}
    for _c in comps:
        _out[_c] = sum(
            np.array(_sp.compartments[_c].history_vals_list).sum(axis=(1, 2))
            for _sp in _sps
        )
    for _tv in (tvs or []):
        _candidates = [_sp for _sp in _sps if _tv in _sp.transition_variables]
        if _candidates:
            # Transition-variable history is saved once per *sub-timestep*
            # (timesteps_per_day entries per day), not per day like
            # compartments. Aggregate each day's sub-steps to a daily total
            # before summing over age/risk, otherwise (with ts_per_day > 1)
            # the series is timesteps_per_day× too long and 1/timesteps_per_day
            # too small.
            def _daily_tv_sum(_sp, _name=_tv):
                _raw = np.array(_sp.transition_variables[_name].history_vals_list)
                _ts = int(getattr(_sp.simulation_settings, "timesteps_per_day", 1) or 1)
                _T = _raw.shape[0]
                if _ts > 1 and _T > 0 and _T % _ts == 0:
                    _raw = _raw.reshape(_T // _ts, _ts, *_raw.shape[1:]).sum(axis=1)
                return _raw.sum(axis=(1, 2))
            _out[_tv] = sum(_daily_tv_sum(_sp) for _sp in _candidates)
    return _out
