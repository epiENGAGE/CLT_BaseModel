"""
fitting.py — Calibrate ConfigDrivenMetapopModel parameters to observed data.

Extracted from the Fitting tab's ``_run_fitting`` cell in
generic_core/examples/_nb_fitting.py so the optimization engine (LHS sampling,
Adam/L-BFGS gradient loop, accept-reject loop, multi-target weighted loss,
seed-scaling, epidemic-start-date offset search) can be called from a
standalone script (e.g. for a server run) as well as from the notebook.

This module has no marimo dependency. The notebook's Fitting tab is a thin
wrapper that gathers UI widget values into ``FitTarget``/``FitConfig`` and
calls ``run_fit``; an exported ``run_fitting.py`` script does the same from
plain JSON.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any

import copy
import json
import multiprocessing as mp
import os
import re
import numpy as np

from generic_core.model_factory import (
    build_compartment_init,
    build_scalar_array,
    make_metapop_from_folder,
    make_single_pop_metapop,
    read_initial_conditions,
)
from generic_core.calibration import compute_rsquared

try:
    import torch
except ImportError:
    torch = None

try:
    from generic_core.torch_generic import (
        build_generic_torch_inputs,
        generic_torch_simulate_calibration_target,
    )
    from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY
except ImportError:
    build_generic_torch_inputs = None
    generic_torch_simulate_calibration_target = None
    RATE_TEMPLATE_REGISTRY = None


# ---------------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------------

@dataclass
class FitTarget:
    """One calibration target: a sum of compartments/transition variables,
    compared against observed data under an optional age/risk/subpop slice."""

    variables: list[str]
    mode: str = "ts"  # "ts" | "scalar" | "proportion"
    weight: float = 1.0
    observed: Any = None  # np.ndarray (ts) or list[dict] (scalar/proportion)
    label: str = ""
    subpop_idx: int = -1  # -1 = "All (sum)"
    age_idx: int = -1
    risk_idx: int = -1
    # Optional per-timepoint weights for "ts" targets (e.g. from a "weight"
    # CSV column), same length/date-alignment as `observed`. Up-weights
    # specific days (typically the epidemic peak) within this target's
    # timeseries. Normalized internally to mean 1 over the days actually used
    # in the loss, so this target's overall contribution (`weight` / lambda)
    # is unchanged — only the relative emphasis between its own days shifts.
    # None means every day is weighted equally (the prior behavior).
    point_weights: Any = None  # np.ndarray | None

    def n_days(self) -> int:
        if self.mode == "ts" and self.observed is not None:
            return len(self.observed)
        return 0


@dataclass
class FitConfig:
    selected_params: list[str]
    bounds: dict[str, tuple[float, float]]
    param_dims: dict[str, list[str]] = field(default_factory=dict)
    # Linked-scale groups: one fitted multiplier scales several base params by
    # the same factor, preserving each base's config ratio. Maps a synthetic
    # multiplier name (which must appear in selected_params with its own bounds,
    # scalar only) to the list of base config params it multiplies —
    # effective[base] = config_baseline[base] × multiplier. Example:
    # {"ihr_scale": ["I_to_H_prop", "IV_to_H_prop"]}. The base params must NOT
    # also be listed in selected_params. Works across every method.
    scale_groups: dict[str, list[str]] = field(default_factory=dict)
    # Fit these parameters in log10 space (LogUniform prior / log-space sampler
    # steps / log-space gradient leaf). Names must appear in selected_params and
    # be scalar with strictly-positive bounds; `bounds` are still given in linear
    # space and transformed internally. Applies to regular params, seed_scale_*
    # keys, and scale-group multipliers. Recorded outputs are always the linear
    # (model-space) value, so downstream re-simulation needs no log awareness.
    log_params: list[str] = field(default_factory=list)
    method: str = "adam"  # "adam" | "lbfgs" | "ar" | "mcmc" | "abc-smc"
    lr: float = 0.01
    n_iter: int = 200
    n_replications: int = 5
    r2_threshold: float = 0.75
    sim_days: int = 180
    fit_start_offset: bool = False
    start_offset_bounds: tuple[int, int] = (-30, 30)
    # ── Bayesian samplers (method="mcmc" → emcee; "abc-smc" → pyabc) ──────────
    # emcee: affine-invariant ensemble MCMC with a Negative-Binomial likelihood
    #   on timeseries targets. n_walkers ensemble members run n_iter steps;
    #   the first mcmc_burnin steps are discarded and every mcmc_thin-th kept.
    #   The retained draws (across walkers) form the posterior ensemble that
    #   populates accepted_params — the same field the Forecast tab consumes.
    n_walkers: int = 32
    mcmc_burnin: int = 500
    mcmc_thin: int = 10
    # abc-smc: sequential Monte Carlo with a weighted-RMSE distance. abc_pop_size
    #   accepted particles per generation, up to abc_max_gens generations.
    abc_pop_size: int = 200
    abc_max_gens: int = 12
    # ── Time-varying transmission multiplier m(t) ────────────────────────────
    # When True, fit a low-dimensional m(t) that scales the force of infection:
    # monthly knots, piecewise-linear in log space, with a Gaussian random-walk
    # prior on the month-to-month increments (smoothness set by tv_tau). Only
    # the Bayesian samplers (mcmc/abc-smc) calibrate the increments; the
    # multiplier itself is respected by every forward simulation (see
    # rate_templates transmission_multiplier_schedule). Supported for both
    # single-population and metapopulation models — for a metapop a single
    # shared m(t) is fit and broadcast identically to every subpopulation.
    tv_transmission: bool = False
    tv_knot_spacing_days: int = 30
    tv_tau: float = 0.25
    # When True (Adam only), each step is capped to a fraction of every
    # parameter's bound width and rejected if it increases the loss
    # (backtracking line search). Keeps the optimiser from overshooting narrow
    # optima on steep / stiff landscapes (e.g. transmission rates whose epidemic
    # size grows ~exponentially). Ignored by L-BFGS, which has its own line
    # search, and by accept-reject.
    robust_steps: bool = True
    # Run in parallel where applicable:
    #   - Gradient methods (Adam/L-BFGS): replications run across a process
    #     pool (embarrassingly parallel; each starts from a different LHS point).
    #   - MCMC (emcee): walker log-prob evaluations run across a process pool.
    #     Each worker rebuilds its own simulate/likelihood context once via the
    #     Pool initializer, so nothing unpicklable crosses the boundary; only a
    #     thread pool was possible before, but the pure-Python simulation holds
    #     the GIL so threads gave no speed-up.
    parallel: bool = True
    # Workers to use when parallel=True. None means auto: half the available
    # CPUs (min 1) for gradient methods (memory-heavy replications), capped at
    # n_replications; for MCMC, nearly all cores (os.cpu_count()-1), capped at
    # n_walkers/2 (emcee evaluates half the ensemble at a time).
    n_workers: int | None = None
    # Bayesian samplers (MCMC / ABC-SMC) only: reuse a single model per process,
    # driving each likelihood/distance eval via reset_simulation() +
    # modify_subpop_params() + (for seeds) init_val recompute + (for m(t)) an
    # in-place transmission-multiplier schedule swap, instead of rebuilding the
    # model from config every eval. Verified bit-identical to the rebuild path
    # for scalar/array params, seed scaling, and m(t); the rebuild path is used
    # automatically when it can't apply (fit_start_offset, or metapopulation).
    # Set False to force the always-rebuild path.
    reuse_model: bool = True


@dataclass
class FitResult:
    best_params: dict
    loss_curve: list
    num_days: int
    observed: list
    method: str
    accepted_params: list
    sim_trajectories: list
    fit_targets: list
    target_labels: list
    target_weights: list
    target_modes: list
    r2_threshold: float | None = None
    n_ar_accepted: int | None = None
    # Linked-scale groups active in this fit ({multiplier: [base_param, ...]}).
    # best_params / accepted_params record only the multiplier value; downstream
    # re-simulation (e.g. the Forecast tab) must expand it into base-param
    # overrides via effective[base] = config_baseline[base] × multiplier.
    scale_groups: dict = field(default_factory=dict)
    # Params that were fit in log10 space. Recorded values are already linear
    # (model-space), so this is informational / for round-tripping the config.
    log_params: list = field(default_factory=list)
    # Knot spacing (days) used to build the m(t) time-varying transmission
    # multiplier from any fitted m_dlog_* log-increments — needed downstream
    # (Forecast/Analysis tabs) to reconstruct the same m(t) trajectory via
    # build_transmission_multiplier_array(). Always recorded, even when m(t)
    # fitting is off, since it's just fit_config.tv_knot_spacing_days.
    tv_knot_spacing_days: int = 30
    # Full FitConfig used to produce this result (selected_params, bounds,
    # param_dims/granularity, log_params, method hyperparameters, ...) — a
    # dict via dataclasses.asdict(fit_config). Recorded so a saved/uploaded
    # result is self-describing: what was fit and how, not just the outcome.
    fit_config: dict = field(default_factory=dict)
    # Per-target slice metadata (subpop/age/risk index, -1 = "All (sum)"),
    # aligned with fit_targets/target_labels/target_weights/target_modes.
    # Not derivable from `observed` alone, so recorded alongside it.
    target_slices: list = field(default_factory=list)


def fit_result_to_dict(result: "FitResult") -> dict:
    """Full JSON-serializable snapshot of a FitResult.

    Used for the Fitting tab's autosave/download (fitted_params.json) and as
    the schema the "load previous fitting result" upload expects — the two
    must stay in sync so a downloaded file can always be re-uploaded.
    """
    def _clean_ts(obs):
        # ts-mode observed arrays may contain NaN (gaps/lead-in days), which
        # json.dumps emits as the non-standard `NaN` token; normalize to null.
        return [None if isinstance(v, float) and np.isnan(v) else v for v in obs]

    _observed = [
        _clean_ts(o) if isinstance(o, list) and o and not isinstance(o[0], dict) else o
        for o in result.observed
    ]
    return {
        "best_params": result.best_params,
        "loss_curve": result.loss_curve,
        "num_days": result.num_days,
        "observed": _observed,
        "method": result.method,
        "accepted_params": result.accepted_params,
        "sim_trajectories": result.sim_trajectories,
        "fit_targets": result.fit_targets,
        "target_labels": result.target_labels,
        "target_weights": result.target_weights,
        "target_modes": result.target_modes,
        "r2_threshold": result.r2_threshold,
        "n_ar_accepted": result.n_ar_accepted,
        "scale_groups": result.scale_groups,
        "log_params": result.log_params,
        "tv_knot_spacing_days": result.tv_knot_spacing_days,
        "fit_config": result.fit_config,
        "target_slices": result.target_slices,
    }


def fit_result_from_dict(d: dict) -> "FitResult":
    """Inverse of fit_result_to_dict — rebuilds a FitResult from a fitting
    result JSON (as produced by the Fitting tab download or a headless
    run_fitting.py run), for display via the "load previous results" upload.
    """
    _observed = []
    for o in d.get("observed", []):
        if isinstance(o, list) and o and isinstance(o[0], dict):
            _observed.append(o)
        elif isinstance(o, list):
            _observed.append([float("nan") if v is None else v for v in o])
        else:
            _observed.append(o)
    return FitResult(
        best_params=d.get("best_params", {}),
        loss_curve=d.get("loss_curve", []),
        num_days=d.get("num_days", 0),
        observed=_observed,
        method=d.get("method", "adam"),
        accepted_params=d.get("accepted_params", []),
        sim_trajectories=d.get("sim_trajectories", []),
        fit_targets=d.get("fit_targets", []),
        target_labels=d.get("target_labels", []),
        target_weights=d.get("target_weights", []),
        target_modes=d.get("target_modes", []),
        r2_threshold=d.get("r2_threshold"),
        n_ar_accepted=d.get("n_ar_accepted"),
        scale_groups=d.get("scale_groups", {}),
        log_params=d.get("log_params", []),
        tv_knot_spacing_days=d.get("tv_knot_spacing_days", 30),
        fit_config=d.get("fit_config", {}),
        target_slices=d.get("target_slices", []),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _shift_date(base_str: str, days: int) -> str:
    return (datetime.strptime(base_str, "%Y-%m-%d") + timedelta(days=int(days))).strftime("%Y-%m-%d")


def _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
    _sp = row.get("subpopulation", None)
    _sp_idx = sp_fallback
    if _sp is not None:
        _sp_s = str(_sp).strip()
        if _sp_s.isdigit():
            _sp_idx = int(_sp_s)
        elif _sp_s in subpop_name_to_idx:
            _sp_idx = subpop_name_to_idx[_sp_s]
    _ag_idx = int(row["age"]) if "age" in row else ag_fallback
    _rk_idx = int(row["risk"]) if "risk" in row else rk_fallback
    return _sp_idx, _ag_idx, _rk_idx


def _validate_targets(targets: list[FitTarget], fit_config: FitConfig) -> None:
    for _i, _t in enumerate(targets):
        if _t.observed is None:
            raise ValueError(f"Target {_i + 1}: no observed data loaded.")
        if not _t.variables:
            raise ValueError(f"Target {_i + 1}: no variables selected.")
    if not fit_config.selected_params:
        raise ValueError("No parameters to fit. Select at least one parameter.")
    _ts_days = [_t.n_days() for _t in targets if _t.mode == "ts"]
    if len(set(_ts_days)) > 1:
        raise ValueError(
            "Timeseries targets have mismatched lengths: "
            + ", ".join(f"target {_i + 1}: {_t.n_days()} days" for _i, _t in enumerate(targets) if _t.mode == "ts")
        )


def _sim_days_for(targets: list[FitTarget], fit_config: FitConfig) -> int:
    _ts_days = [_t.n_days() for _t in targets if _t.mode == "ts"]
    return max(_ts_days) if _ts_days else int(fit_config.sim_days)


def _aligned_point_weights(target: FitTarget, n_days: int) -> np.ndarray:
    """Per-day weights for a "ts" target, sliced/padded to `n_days`. Missing
    or NaN entries default to 1.0 (no preference). Callers normalize this to
    mean 1 over the days actually used in the loss."""
    pw = target.point_weights
    if pw is None:
        return np.ones(n_days, dtype=float)
    pw_np = np.asarray(pw[:n_days], dtype=float)
    if len(pw_np) < n_days:
        pw_np = np.concatenate([pw_np, np.ones(n_days - len(pw_np), dtype=float)])
    return np.where(np.isnan(pw_np), 1.0, pw_np)


def _weighted_rsquared(obs: np.ndarray, sim: np.ndarray, w: np.ndarray) -> float:
    """Weighted R², matching `compute_rsquared` when w is uniform."""
    if len(obs) == 0:
        return 0.0
    w_sum = float(w.sum())
    if w_sum <= 0:
        return 0.0
    obs_mean = float((w * obs).sum() / w_sum)
    ss_res = float((w * (obs - sim) ** 2).sum())
    ss_tot = float((w * (obs - obs_mean) ** 2).sum())
    if ss_tot <= 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Trajectory extraction (shared by accept-reject and the Bayesian samplers)
# ---------------------------------------------------------------------------

def _read_daily_history(sp_m, var, n_days):
    """Daily-resolution history (n_days, A, R) for a target variable, or None.

    Compartments save one entry per day (via ``save_daily_history``), so their
    history is already daily. Transition variables, however, save one entry per
    *sub-timestep* (``timesteps_per_day`` entries per day) in
    ``SubpopModel.sample_transitions`` — each entry is a single sub-step flow.
    A transition-variable trajectory must therefore be summed over the
    sub-timesteps within each day to recover the daily flow (e.g. daily
    hospital admissions). Slicing ``[:n_days]`` on the raw sub-step history
    (as the old code did) grabbed only the first ``n_days / timesteps_per_day``
    days at ``1 / timesteps_per_day`` magnitude, collapsing the trajectory.
    """
    if var in sp_m.transition_variables:
        h = np.array(sp_m.transition_variables[var].history_vals_list)
        tspd = int(getattr(sp_m.simulation_settings, "timesteps_per_day", 1) or 1)
        if tspd > 1 and h.shape[0] >= n_days * tspd:
            h = h[:n_days * tspd].reshape(n_days, tspd, *h.shape[1:]).sum(axis=1)
        else:
            h = h[:n_days]
        return h
    if var in sp_m.compartments:
        return np.array(sp_m.compartments[var].history_vals_list)[:n_days]
    return None


def _extract_ts_sliced(metapop_obj, target_vars_k, sp_idx, ag_idx, rk_idx, n_days):
    sps = list(metapop_obj.subpop_models.values())
    sp_list = [sps[sp_idx]] if sp_idx >= 0 else sps
    result = None
    for sp_m in sp_list:
        for var in target_vars_k:
            h = _read_daily_history(sp_m, var, n_days)
            if h is None:
                continue
            if ag_idx >= 0:
                h = h[:, ag_idx:ag_idx + 1, :]
            if rk_idx >= 0:
                h = h[:, :, rk_idx:rk_idx + 1]
            hv = h.sum(axis=(1, 2))
            result = hv if result is None else result + hv
    return result if result is not None else np.zeros(n_days)


def _extract_scalar_rows(metapop_obj, target_vars_k, scalar_rows, n_days, subpop_name_to_idx,
                         sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
    sps = list(metapop_obj.subpop_models.values())
    row_sims = []
    for row in scalar_rows:
        sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_fallback, ag_fallback, rk_fallback)
        sp_list = [sps[sp_i]] if sp_i >= 0 else sps
        total = 0.0
        for sp_m in sp_list:
            for var in target_vars_k:
                h = _read_daily_history(sp_m, var, n_days)
                if h is None:
                    continue
                if ag_i >= 0:
                    h = h[:, ag_i:ag_i + 1, :]
                if rk_i >= 0:
                    h = h[:, :, rk_i:rk_i + 1]
                total += float(h.sum())
        row_sims.append(total)
    return row_sims


def _extract_proportion_rows(metapop_obj, target_vars_k, scalar_rows, n_days, subpop_name_to_idx,
                             sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
    sps = list(metapop_obj.subpop_models.values())
    row_sims = []
    for ri_prop, row in enumerate(scalar_rows):
        sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_fallback, ag_fallback, rk_fallback)
        if sp_i < 0 and ag_i < 0 and rk_i < 0:
            sp_i = ri_prop
        den_sp = sp_i if (sp_i >= 0 and (ag_i >= 0 or rk_i >= 0)) else -1
        sp_list_num = [sps[sp_i]] if (0 <= sp_i < len(sps)) else sps
        sp_list_den = [sps[den_sp]] if den_sp >= 0 else sps
        num = 0.0
        den = 0.0
        for sp_m in sp_list_den:
            for var in target_vars_k:
                h_full = _read_daily_history(sp_m, var, n_days)
                if h_full is None:
                    continue
                den += float(h_full.sum())
        for sp_m in sp_list_num:
            for var in target_vars_k:
                h_full = _read_daily_history(sp_m, var, n_days)
                if h_full is None:
                    continue
                h_sliced = h_full
                if ag_i >= 0:
                    h_sliced = h_sliced[:, ag_i:ag_i + 1, :]
                if rk_i >= 0:
                    h_sliced = h_sliced[:, :, rk_i:rk_i + 1]
                num += float(h_sliced.sum())
        row_sims.append(num / den if den > 1e-10 else 0.0)
    return row_sims


# ---------------------------------------------------------------------------
# Time-varying transmission multiplier m(t)
# ---------------------------------------------------------------------------

TV_INCREMENT_PREFIX = "m_dlog_"  # fitted param name prefix for m(t) log-increments


# ---------------------------------------------------------------------------
# Linked-scale groups: one fitted multiplier scales several base params
# ---------------------------------------------------------------------------

def _active_scale_groups(fit_config, selected_params) -> dict:
    """{multiplier_name: [base_param, ...]} for scale groups whose multiplier is
    actually being fitted (present in selected_params)."""
    sg = getattr(fit_config, "scale_groups", None) or {}
    sel = set(selected_params)
    return {m: list(bases) for m, bases in sg.items() if m in sel and bases}


def _resolve_scale_baselines(config_dict: dict, groups: dict) -> dict:
    """Config baseline value (scalar or list) for every base param in `groups`."""
    base_params = (config_dict or {}).get("params", {}) or {}
    out: dict = {}
    for bases in groups.values():
        for b in bases:
            if b not in base_params:
                raise ValueError(
                    f"Linked-scale base parameter '{b}' is not a model parameter "
                    f"(not found in config['params'])."
                )
            out[b] = base_params[b]
    return out


def _expand_scale_overrides(sampled: dict, groups: dict, baselines: dict) -> dict:
    """Turn a sampled dict (which holds multiplier values under their own names)
    into model param_overrides: drop the multiplier keys and, for each base
    param, set config_baseline[base] × multiplier. Ratios between the bases are
    preserved. Non-group entries pass through unchanged."""
    if not groups:
        return dict(sampled)
    out = {k: v for k, v in sampled.items() if k not in groups}
    for m, bases in groups.items():
        if m not in sampled:
            continue
        # The multiplier is a plain float for a scalar (ungranular) group, or a
        # nested (age/risk) list when the group was given granularity — in
        # which case it broadcasts elementwise against each base's own array.
        _mult_raw = sampled[m]
        _is_arr = isinstance(_mult_raw, (list, tuple, np.ndarray))
        mult = np.asarray(_mult_raw, dtype=float) if _is_arr else float(_mult_raw)
        for b in bases:
            bl = baselines[b]
            if _is_arr or isinstance(bl, (list, tuple, np.ndarray)):
                out[b] = (np.asarray(bl, dtype=float) * mult).tolist()
            else:
                out[b] = float(bl) * mult
    return out


def _active_log_params(fit_config, selected_params) -> set:
    """Names of fitted params to explore in log10 space (present in
    selected_params). Positive-bound validation happens in run_fit."""
    lp = getattr(fit_config, "log_params", None) or []
    sel = set(selected_params)
    return {p for p in lp if p in sel}


def _tv_knot_days(sim_days: int, spacing_days: int) -> list[int]:
    """Day indices of the m(t) knots: 0, spacing, 2·spacing, … up to sim_days."""
    spacing = max(1, int(spacing_days))
    knots = list(range(0, int(sim_days), spacing))
    if not knots or knots[-1] != int(sim_days) - 1:
        knots.append(int(sim_days) - 1)
    return knots


def _tv_increment_names(n_incr: int) -> list[str]:
    return [f"{TV_INCREMENT_PREFIX}{i + 1}" for i in range(n_incr)]


def build_transmission_multiplier_array(increments, knot_days, num_days: int) -> np.ndarray:
    """m(t) over `num_days`: exp of the piecewise-linear interpolation of the
    cumulative log-increments over the monthly knots (anchored g[0]=0)."""
    increments = np.asarray(increments, dtype=float)
    g = np.concatenate([[0.0], np.cumsum(increments)])
    g_interp = np.interp(np.arange(num_days), np.asarray(knot_days, dtype=float), g)
    return np.exp(g_interp)


def _build_transmission_multiplier_df(increments, knot_days, start_date: str, horizon: int):
    """DataFrame (date, transmission_multiplier) for the schedule input."""
    import pandas as pd
    m = build_transmission_multiplier_array(increments, knot_days, horizon)
    dates = pd.date_range(start=start_date, periods=horizon, freq="D").date
    return pd.DataFrame({"date": dates, "transmission_multiplier": m})


def _inject_tv_transmission(config_dict: dict) -> tuple[dict, int]:
    """Return a copy of the config with a 'transmission_multiplier' timeseries
    schedule added and referenced by every force-of-infection transition, plus
    the count of FOI transitions wired. No-op-safe if already present."""
    import copy
    cfg = copy.deepcopy(config_dict)
    sched_names = {s.get("name") for s in cfg.get("schedules", [])}
    if "transmission_multiplier" not in sched_names:
        cfg.setdefault("schedules", []).append({
            "name": "transmission_multiplier",
            "schedule_template": "timeseries_lookup",
            "schedule_config": {
                "df_attribute": "transmission_multiplier_df",
                "value_column": "transmission_multiplier",
            },
        })
    n_foi = 0
    for t in cfg.get("transitions", []):
        if "force_of_infection" in str(t.get("rate_template", "")):
            t.setdefault("rate_config", {})["transmission_multiplier_schedule"] = "transmission_multiplier"
            n_foi += 1
    return cfg, n_foi


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fit(
    config_dict: dict,
    compartments: list[str],
    is_metapop: bool,
    targets: list[FitTarget],
    fit_config: FitConfig,
    start_date: str,
    ts_per_day: int = 7,
    seed_base: int = 0,
    num_age_groups: int = 1,
    num_risk_groups: int = 1,
    metapop_folder: str | None = None,
    metapop_travel_config: dict | None = None,
    compartment_init: dict | None = None,
    subpop_names: tuple[str, ...] = (),
    mobility_value: float = 1.0,
    daily_vaccines_value: float = 0.0,
    schedule_dfs: Any = None,
    progress_callback=None,
) -> FitResult:
    """Calibrate ``fit_config.selected_params`` against ``targets``.

    ``compartment_init`` (single-population only) and ``subpop_names``
    (metapop only, for resolving ``subpopulation`` columns in scalar/
    proportion target rows) are precomputed by the caller — see
    ``generic_core/examples/_nb_fitting.py`` for how the notebook builds them
    from ``config_dict``.

    ``schedule_dfs`` (single-population only — metapop builds its own
    per-subpop schedules from the metapop folder's CSVs regardless of this
    argument) is a ``SimpleNamespace`` of ``absolute_humidity_df`` /
    ``school_work_calendar_df`` / ``mobility_df`` / ``daily_vaccines_df``, as
    produced by ``build_notebook_schedules_input``'s ``*_df`` fields. When
    omitted, constant-valued schedules are built from ``mobility_value`` /
    ``daily_vaccines_value`` instead — which silently drops any real,
    time-varying vaccination schedule (and the backfill/delay preprocessing
    ``VaccineScheduleGeneric`` applies to it). Always pass the real schedules
    here when the model config references an uploaded vaccine/humidity/
    calendar/mobility CSV.
    """
    _validate_targets(targets, fit_config)

    _n = len(targets)
    _target_vars_list = [t.variables for t in targets]
    _target_modes = [t.mode for t in targets]
    _target_weights = [float(t.weight) for t in targets]
    _weight_sum = sum(_target_weights) or 1.0
    _sp_idxs = [t.subpop_idx for t in targets]
    _age_idxs = [t.age_idx for t in targets]
    _risk_idxs = [t.risk_idx for t in targets]
    _target_labels = [t.label or f"Target {_i + 1}" for _i, t in enumerate(targets)]
    _subpop_name_to_idx = {str(_nm): _i for _i, _nm in enumerate(subpop_names)}

    _target_tvs = [[v for v in _target_vars_list[k] if v not in compartments] for k in range(_n)]
    _target_comps = [[v for v in _target_vars_list[k] if v in compartments] for k in range(_n)]

    _sim_days = _sim_days_for(targets, fit_config)
    _num_fit_days_per_target = [
        targets[k].n_days() if _target_modes[k] == "ts" else _sim_days
        for k in range(_n)
    ]

    _selected_params = list(fit_config.selected_params)
    _lo_vals = [float(fit_config.bounds[p][0]) for p in _selected_params]
    _hi_vals = [float(fit_config.bounds[p][1]) for p in _selected_params]
    _dim_vals = [list(fit_config.param_dims.get(p, [])) for p in _selected_params]
    _A, _R = num_age_groups, num_risk_groups

    _seed_scale_comps = {
        _pn[len("seed_scale_"):]: _j
        for _j, _pn in enumerate(_selected_params)
        if _pn.startswith("seed_scale_") and _pn[len("seed_scale_"):] in compartments
    }

    # Linked-scale groups (one multiplier scales several base params). Validate
    # up front so every method sees a consistent, error-checked configuration.
    _scale_groups = _active_scale_groups(fit_config, _selected_params)
    if _scale_groups:
        _cfg_params = (config_dict or {}).get("params", {}) or {}
        for _m, _bases in _scale_groups.items():
            if _m in _cfg_params:
                raise ValueError(
                    f"Linked-scale multiplier '{_m}' collides with an existing model "
                    f"parameter; choose a name that is not already a model parameter."
                )
            for _b in _bases:
                if _b not in _cfg_params:
                    raise ValueError(
                        f"Linked-scale base '{_b}' (group '{_m}') is not a model parameter."
                    )
                if _b in _selected_params:
                    raise ValueError(
                        f"Parameter '{_b}' cannot be both directly fitted and scaled by "
                        f"the linked multiplier '{_m}'. Remove it from the fitted-params list."
                    )

    # Log-space params must be scalar with strictly-positive linear bounds.
    _log_params = _active_log_params(fit_config, _selected_params)
    if _log_params:
        _pdims = getattr(fit_config, "param_dims", {}) or {}
        for _lp in _log_params:
            _lo_lp, _hi_lp = fit_config.bounds[_lp]
            if _lo_lp <= 0 or _hi_lp <= 0:
                raise ValueError(
                    f"Log-space parameter '{_lp}' needs strictly-positive bounds "
                    f"(got [{_lo_lp}, {_hi_lp}]); log10 is undefined for ≤ 0."
                )
            if _pdims.get(_lp):
                raise ValueError(
                    f"Log-space parameter '{_lp}' must be scalar; remove its "
                    f"age/risk/subpopulation granularity."
                )

    if fit_config.method in ("adam", "lbfgs"):
        _run = _run_gradient_fit(
            config_dict, compartments, is_metapop, targets, fit_config,
            start_date, ts_per_day, seed_base, _A, _R, metapop_folder,
            metapop_travel_config, compartment_init, mobility_value, daily_vaccines_value, schedule_dfs,
            _n, _target_vars_list, _target_modes, _target_weights, _weight_sum,
            _sp_idxs, _age_idxs, _risk_idxs, _subpop_name_to_idx,
            _target_tvs, _target_comps, _sim_days, _num_fit_days_per_target,
            _selected_params, _lo_vals, _hi_vals, _dim_vals, _seed_scale_comps,
            progress_callback=progress_callback,
        )
        _best_params, _loss_curve, _accepted_params_list, _accepted_trajectories, _best_trajs = _run
    elif fit_config.method == "ar":
        _run = _run_accept_reject_fit(
            config_dict, compartments, is_metapop, targets, fit_config,
            start_date, ts_per_day, seed_base, _A, _R, metapop_folder,
            metapop_travel_config, compartment_init, subpop_names, mobility_value, daily_vaccines_value, schedule_dfs,
            _n, _target_vars_list, _target_modes, _target_weights, _weight_sum,
            _sp_idxs, _age_idxs, _risk_idxs, _subpop_name_to_idx,
            _target_tvs, _target_comps, _sim_days, _num_fit_days_per_target,
            _selected_params, _lo_vals, _hi_vals, _dim_vals, _seed_scale_comps,
            progress_callback=progress_callback,
        )
        _best_params, _loss_curve, _accepted_params_list, _accepted_trajectories, _best_trajs = _run
    elif fit_config.method in ("mcmc", "abc-smc"):
        _run = _run_bayesian_fit(
            fit_config.method,
            config_dict, compartments, is_metapop, targets, fit_config,
            start_date, ts_per_day, seed_base, _A, _R, metapop_folder,
            metapop_travel_config, compartment_init, subpop_names, mobility_value, daily_vaccines_value, schedule_dfs,
            _n, _target_vars_list, _target_modes, _target_weights, _weight_sum,
            _sp_idxs, _age_idxs, _risk_idxs, _subpop_name_to_idx,
            _target_tvs, _target_comps, _sim_days, _num_fit_days_per_target,
            _selected_params, _lo_vals, _hi_vals, _dim_vals, _seed_scale_comps,
            progress_callback=progress_callback,
        )
        _best_params, _loss_curve, _accepted_params_list, _accepted_trajectories, _best_trajs = _run
    else:
        raise ValueError(f"Unknown fitting method: {fit_config.method!r}")

    _sim_trajectories = _accepted_trajectories if _accepted_trajectories else ([_best_trajs] if _best_trajs else [])
    _is_ar = fit_config.method == "ar"
    return FitResult(
        best_params=_best_params,
        loss_curve=_loss_curve,
        num_days=_sim_days,
        observed=[list(o) if isinstance(o, np.ndarray) else o for o in (t.observed for t in targets)],
        method=fit_config.method,
        accepted_params=_accepted_params_list if _accepted_params_list else [_best_params],
        sim_trajectories=_sim_trajectories,
        fit_targets=_target_vars_list,
        target_labels=_target_labels,
        target_weights=_target_weights,
        target_modes=_target_modes,
        r2_threshold=fit_config.r2_threshold if _is_ar else None,
        n_ar_accepted=len(_accepted_params_list) if _is_ar else None,
        scale_groups=_scale_groups,
        log_params=sorted(_log_params),
        tv_knot_spacing_days=int(fit_config.tv_knot_spacing_days),
        fit_config=asdict(fit_config),
        target_slices=[
            {"subpop_idx": t.subpop_idx, "age_idx": t.age_idx, "risk_idx": t.risk_idx}
            for t in targets
        ],
    )


# ---------------------------------------------------------------------------
# Gradient methods (Adam / L-BFGS)
# ---------------------------------------------------------------------------

# Scale compartment values and total population up by this factor. PyTorch
# uses softplus for non-negativity; for small counts it diverges from the
# identity — large values (~10000) make it negligible. N must scale with the
# compartments so I/N (force of infection) is unchanged.
_GRAD_SCALE = 10000.0


def _build_grad_model(
    config_dict, compartments, is_metapop, start_date_rep, sim_days, seed_offset,
    seed_base, ts_per_day, all_tvs_union, metapop_folder, metapop_travel_config,
    compartment_init, A, R, mobility_value, daily_vaccines_value, schedule_dfs,
):
    if is_metapop:
        return make_metapop_from_folder(
            metapop_folder, config_dict, start_date_rep, sim_days, list(compartments),
            seed_offset=seed_offset, seed_base=seed_base, ts_per_day=ts_per_day,
            stochastic=False, tvs=all_tvs_union, save_daily=False,
            travel_config=metapop_travel_config or None,
            num_age_groups=A, num_risk_groups=R,
            mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
        )
    _m, _mc, _ = make_single_pop_metapop(
        config_dict, start_date_rep, sim_days, compartment_init,
        ts_per_day=ts_per_day, stochastic=False, tvs=all_tvs_union,
        save_daily=False, seed_base=seed_base + seed_offset,
        travel_config=metapop_travel_config,
        num_age_groups=A, num_risk_groups=R,
        mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
        schedule_dfs=schedule_dfs,
    )
    return _m, _mc


@dataclass
class _GradFitContext:
    """Everything a single gradient-fit replication needs, independent of
    every other replication — picklable so it can cross a process boundary
    for parallel execution."""

    config_dict: dict
    compartments: list
    is_metapop: bool
    targets: list
    fit_config: FitConfig
    start_date: str
    ts_per_day: int
    seed_base: int
    A: int
    R: int
    metapop_folder: Any
    metapop_travel_config: Any
    compartment_init: Any
    mobility_value: float
    daily_vaccines_value: float
    schedule_dfs: Any
    n: int
    target_vars_list: list
    target_modes: list
    target_weights: list
    weight_sum: float
    sp_idxs: list
    age_idxs: list
    risk_idxs: list
    subpop_name_to_idx: dict
    all_tvs_union: list
    all_comps_union: list
    sim_days: int
    num_fit_days_per_target: list
    selected_params: list
    lo_vals: list
    hi_vals: list
    dim_vals: list
    seed_scale_comps: dict
    seed_scale_keys: list
    n_regular_lhs: int
    offset_lhs_col: int
    has_offset_col: bool
    # Linked-scale groups: {multiplier_name: [base_param, ...]}. The multiplier
    # is a single optimized scalar tensor; each base param tensor is set to
    # config_baseline × multiplier every forward pass (see
    # _run_one_gradient_replication). LHS columns follow the seed-scale columns.
    scale_groups: dict = field(default_factory=dict)
    scale_group_keys: list = field(default_factory=list)
    # Params fit in log10 space (regular params, seed_scale_* keys, and/or
    # scale-group multipliers). Their leaf tensor holds log10(value) and the
    # applied model quantity is 10**leaf; LHS columns / bounds are log-scaled.
    log_params: set = field(default_factory=set)


def _init_grad_worker():
    """ProcessPoolExecutor initializer: keep each worker single-threaded so
    N worker processes don't oversubscribe the CPUs they were given."""
    if torch is not None:
        torch.set_num_threads(1)


def _run_one_gradient_replication(rep_idx: int, lhs_row: np.ndarray, ctx: "_GradFitContext") -> dict:
    """Runs entirely from `ctx` + `lhs_row` — builds its own model, so it has
    no dependency on any other replication and can run in its own process."""
    if torch is None:
        raise RuntimeError("PyTorch not available. Install torch to use gradient-based fitting.")

    fit_config = ctx.fit_config
    targets = ctx.targets
    compartments = ctx.compartments
    selected_params = ctx.selected_params
    lo_vals, hi_vals, dim_vals = ctx.lo_vals, ctx.hi_vals, ctx.dim_vals
    target_vars_list, target_modes = ctx.target_vars_list, ctx.target_modes
    target_weights, weight_sum = ctx.target_weights, ctx.weight_sum
    sp_idxs, age_idxs, risk_idxs = ctx.sp_idxs, ctx.age_idxs, ctx.risk_idxs
    subpop_name_to_idx = ctx.subpop_name_to_idx
    sim_days, ts_per_day = ctx.sim_days, ctx.ts_per_day
    num_fit_days_per_target = ctx.num_fit_days_per_target
    all_tvs_union, all_comps_union = ctx.all_tvs_union, ctx.all_comps_union
    n = ctx.n

    sampled_offset = 0
    if fit_config.fit_start_offset and ctx.has_offset_col:
        sampled_offset = int(round(float(lhs_row[ctx.offset_lhs_col])))
    start_rep = _shift_date(ctx.start_date, sampled_offset) if fit_config.fit_start_offset else ctx.start_date

    metapop, mc = _build_grad_model(
        ctx.config_dict, compartments, ctx.is_metapop, start_rep, sim_days, rep_idx,
        ctx.seed_base, ts_per_day, all_tvs_union, ctx.metapop_folder, ctx.metapop_travel_config,
        ctx.compartment_init, ctx.A, ctx.R, ctx.mobility_value, ctx.daily_vaccines_value, ctx.schedule_dfs,
    )
    ti = build_generic_torch_inputs(metapop, mc, sim_days)
    ti["params_dict"]["total_pop_age_risk"] = ti["params_dict"]["total_pop_age_risk"] * _GRAD_SCALE
    ti["precomputed"].total_pop_LAR_tensor = ti["precomputed"].total_pop_LAR_tensor * _GRAD_SCALE
    ti["precomputed"].total_pop_LA = ti["precomputed"].total_pop_LA * _GRAD_SCALE

    seed_base_state = {k: v.clone().detach() * _GRAD_SCALE for k, v in ti["state_dict"].items()}
    scale_tensors: dict = {}

    log_params = set(ctx.log_params)

    def _eff(name, tens):
        # Model-space value of an optimized leaf: 10**leaf for a log param
        # (leaf stored in log10 space), else the leaf itself. Differentiable.
        return (10.0 ** tens) if name in log_params else tens

    # Linked-scale groups: one multiplier leaf tensor drives several base params.
    # Capture each base param's config baseline now (before any forward-pass
    # overwrite); the multiplier tensors themselves are created alongside the
    # seed-scale tensors below.
    mult_tensors: dict = {}
    mult_baselines: dict = {
        _m: {
            _b: ti["params_dict"][_b].detach().clone()
            for _b in ctx.scale_groups[_m] if _b in ti["params_dict"]
        }
        for _m in ctx.scale_group_keys
    }
    # Regular params fit in log space: their leaf holds log10(value) and
    # ti["params_dict"][pn] is set to 10**leaf every forward pass (like the
    # scale-group bases), so gradients flow to the log-space leaf.
    log_reg_tensors: dict = {}

    def _apply_scale_group_params():
        """Refresh every derived (non-leaf) params_dict entry from its
        optimized leaf tensor, so the forward pass sees this step's values.

        base_param := config_baseline × multiplier, differentiable in the
        multiplier. Reassigned each forward pass so gradients flow to the mult.
        """
        for _m, _tens in mult_tensors.items():
            _mult = _eff(_m, _tens)
            for _b, _bl in mult_baselines[_m].items():
                ti["params_dict"][_b] = _bl * _mult
        # Log-space regular params: params_dict := 10**leaf.
        for _pn, _tens in log_reg_tensors.items():
            ti["params_dict"][_pn] = 10.0 ** _tens

    robust_steps = bool(getattr(fit_config, "robust_steps", True))
    STEP_CAP_FRAC = 0.1
    MAX_BACKTRACKS = 8

    def _clamp_bounds(name, lo, hi):
        # Clamp interval in the leaf's own space (log10 for log params).
        if name in log_params:
            return float(np.log10(lo)), float(np.log10(hi))
        return float(lo), float(hi)

    def _apply_bounds():
        """In-place clamp every optimized leaf tensor to its fitted bound
        interval (log10 space for log params), so an Adam/L-BFGS step can
        never push a parameter outside the user-specified range."""
        for _pn, (_pmin, _pmax) in fit_config.bounds.items():
            # Log-regular leaves are clamped in log space below; their
            # params_dict entry is a derived (non-leaf) 10**leaf tensor.
            if _pn in ti["params_dict"] and _pn not in log_params:
                ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
        for _pn, _tens in log_reg_tensors.items():
            _lo, _hi = _clamp_bounds(_pn, lo_vals[selected_params.index(_pn)],
                                     hi_vals[selected_params.index(_pn)])
            _tens.data.clamp_(_lo, _hi)
        for _comp_ss in ctx.seed_scale_keys:
            _j_ss = ctx.seed_scale_comps[_comp_ss]
            _lo, _hi = _clamp_bounds(f"seed_scale_{_comp_ss}", lo_vals[_j_ss], hi_vals[_j_ss])
            scale_tensors[_comp_ss].data.clamp_(_lo, _hi)
        for _m, _tens in mult_tensors.items():
            _jm = selected_params.index(_m)
            _lo, _hi = _clamp_bounds(_m, lo_vals[_jm], hi_vals[_jm])
            _tens.data.clamp_(_lo, _hi)

    def _slice_tensor(sim, sp_idx, ag_idx, rk_idx):
        """Slice (days, L, A, R) and sum remaining dims → (days,)."""
        s = sim
        if sp_idx >= 0:
            s = s[:, sp_idx:sp_idx + 1, :, :]
        if ag_idx >= 0:
            s = s[:, :, ag_idx:ag_idx + 1, :]
        if rk_idx >= 0:
            s = s[:, :, :, rk_idx:rk_idx + 1]
        return s.sum(dim=tuple(range(1, s.dim())))

    def _scale_mult(c):
        # Effective seed multiplier for compartment c (10**leaf if log).
        return _eff(f"seed_scale_{c}", scale_tensors[c])

    def _build_scaled_state():
        """Initial (day-0) state dict for this forward pass: every seeded
        compartment scaled by its ``seed_scale_*`` multiplier, with the
        added/removed counts offset against ``compartments[0]`` (e.g.
        Susceptible) so total population stays fixed — mirroring the AR
        path's ``metapop_base_inits``/``_scale_compartment_init`` logic."""
        state_run = {}
        for ks, vs in seed_base_state.items():
            if ks in scale_tensors:
                state_run[ks] = seed_base_state[ks] * _scale_mult(ks)
            elif compartments and ks == compartments[0] and scale_tensors:
                delta = sum(
                    seed_base_state[c] * (_scale_mult(c) - 1.0)
                    for c in scale_tensors if c in seed_base_state
                )
                state_run[ks] = torch.clamp(seed_base_state[ks] - delta, min=0.0)
            else:
                state_run[ks] = vs.clone()
        return state_run

    def _compute_total_loss():
        """Differentiable scalar loss for this replication's current tensor
        state — the quantity Adam/L-BFGS descend.

        1. Apply this step's parameter values (scale-group multipliers and
           log-space regular params are derived tensors, so their
           ``params_dict``/state entries are refreshed here — see
           ``_apply_scale_group_params`` / ``_build_scaled_state``) and run one
           forward simulation via ``generic_torch_simulate_calibration_target``.
        2. For each target, sum its constituent variables, slice to the
           target's subpop/age/risk selection, and compare to observed under
           its mode:
           - "ts": per-day squared error, NaN-masked (date-alignment gaps),
             optionally re-weighted by per-timepoint ``point_weights``
             (renormalized to mean 1 so only within-target emphasis shifts),
             normalized by mean(obs²) so targets on different scales
             contribute comparably.
           - "scalar": squared error against each observed row's single
             value, summed then normalized by sum(obs²).
           - "proportion": squared error between simulated numerator/
             denominator ratio and each observed row's proportion, summed
             then normalized by sum(obs²).
        3. Combine all targets into one scalar via a convex combination
           weighted by each target's ``weight / weight_sum``.

        The `/ _GRAD_SCALE` divisions undo the population up-scaling applied
        to compartments/state at model-build time (see ``_GRAD_SCALE``),
        which exists purely so PyTorch's softplus non-negativity trick is
        numerically well-behaved for small compartment counts.
        """
        _apply_scale_group_params()
        state_run = _build_scaled_state()
        sim_all = generic_torch_simulate_calibration_target(
            state_run, ti["params_dict"], mc, RATE_TEMPLATE_REGISTRY,
            ti["precomputed"], ti["schedules_dict"],
            sim_days, ts_per_day, all_tvs_union, all_comps_union,
        )
        total = torch.tensor(0.0, dtype=torch.float64)
        for k in range(n):
            parts = [sim_all[v] for v in target_vars_list[k]]
            sim_k = parts[0]
            for pt in parts[1:]:
                sim_k = sim_k + pt
            w_k = target_weights[k] / weight_sum
            if target_modes[k] == "scalar":
                loss_k = torch.tensor(0.0, dtype=torch.float64)
                obs_sq_sum = sum(float(row["value"]) ** 2 for row in targets[k].observed) + 1e-10
                for row in targets[k].observed:
                    sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_idxs[k], age_idxs[k], risk_idxs[k])
                    sliced = _slice_tensor(sim_k, sp_i, ag_i, rk_i) / _GRAD_SCALE
                    obs_v = torch.tensor(float(row["value"]), dtype=torch.float64)
                    loss_k = loss_k + (sliced.sum() - obs_v) ** 2
                loss_k = loss_k / obs_sq_sum
            elif target_modes[k] == "proportion":
                loss_k = torch.tensor(0.0, dtype=torch.float64)
                obs_sq_sum = sum(float(row["value"]) ** 2 for row in targets[k].observed) + 1e-10
                for ri_prop, row in enumerate(targets[k].observed):
                    sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_idxs[k], age_idxs[k], risk_idxs[k])
                    if sp_i < 0 and ag_i < 0 and rk_i < 0:
                        sp_i = ri_prop
                    den_sp = sp_i if (sp_i >= 0 and (ag_i >= 0 or rk_i >= 0)) else -1
                    den = _slice_tensor(sim_k, den_sp, -1, -1).sum()
                    num = _slice_tensor(sim_k, sp_i, ag_i, rk_i).sum()
                    sim_prop = num / (den + 1e-10)
                    obs_prop = torch.tensor(float(row["value"]), dtype=torch.float64)
                    loss_k = loss_k + (sim_prop - obs_prop) ** 2
                loss_k = loss_k / obs_sq_sum
            else:
                days_k = num_fit_days_per_target[k]
                sliced = _slice_tensor(sim_k, sp_idxs[k], age_idxs[k], risk_idxs[k]) / _GRAD_SCALE
                sliced = sliced[:days_k]
                # Observed may carry NaN for days with no data (date-alignment
                # gaps / leading lead-in); mask those out of the loss.
                obs_np = np.asarray(targets[k].observed[:days_k], dtype=float)
                mask_np = ~np.isnan(obs_np)
                pw_np = _aligned_point_weights(targets[k], days_k)
                if not mask_np.all():
                    sliced = sliced[torch.from_numpy(mask_np)]
                    obs_np = obs_np[mask_np]
                    pw_np = pw_np[mask_np]
                # Normalize so per-timepoint weights average to 1 over the days
                # actually used — this target's overall contribution (w_k
                # above) is unchanged; only emphasis between its own days
                # shifts (e.g. up-weighting the epidemic peak).
                pw_mean = float(pw_np.mean()) if pw_np.size else 1.0
                if pw_mean > 0:
                    pw_np = pw_np / pw_mean
                pw_t = torch.tensor(pw_np, dtype=torch.float64)
                obs_t_k = torch.tensor(obs_np, dtype=torch.float64)
                obs_sq_mean_k = float((pw_t * obs_t_k ** 2).mean()) + 1e-10
                loss_k = (pw_t * (sliced - obs_t_k) ** 2).mean() / obs_sq_mean_k
            total = total + w_k * loss_k
        return total

    def _record_params_grad():
        """Snapshot the current optimized tensor values as a plain-Python
        params dict, in the same ``{param_name: value}`` shape used across
        every fitting method's ``best_params``/``accepted_params`` output.

        Called whenever this replication's loss improves, so ``rep_best_params``
        always reflects the tensor state at the best-seen loss rather than the
        (possibly worse) state at the end of optimization. Log params are
        converted back to linear (model-space) values; per-subpop scalar
        params are unpacked into ``{name}_subpop{i}`` keys; granular
        (age/risk/subpop) values are recorded as nested lists.
        """
        out = {}
        for pn in selected_params:
            # Log params: record the linear (model-space) value 10**leaf.
            if pn in log_reg_tensors:
                out[pn] = float(10.0 ** log_reg_tensors[pn].item())
            elif pn.startswith("seed_scale_"):
                comp = pn[len("seed_scale_"):]
                if comp in scale_tensors:
                    out[pn] = float(_eff(pn, scale_tensors[comp]).item())
            elif pn in mult_tensors:
                # Granular (age/risk/subpop) multiplier: nested list, same
                # representation as an array-valued regular param below.
                out[pn] = _eff(pn, mult_tensors[pn]).detach().tolist()
            elif pn in ti["params_dict"]:
                v = ti["params_dict"][pn].detach()
                if v.ndim == 3 and v.shape[1] == 1 and v.shape[2] == 1:
                    for spi in range(v.shape[0]):
                        out[f"{pn}_subpop{spi}"] = float(v[spi, 0, 0].item())
                else:
                    out[pn] = v.tolist()
        return out

    def _record_trajs_grad():
        """Re-run the forward simulation under ``torch.no_grad()`` at the
        current tensor state and extract each target's simulated trajectory
        (``{"target_k": [...]}``, one row/day per entry matching the target's
        mode) for plotting/inspection.

        A second simulation rather than reusing ``_compute_total_loss``'s
        output because that pass builds the *loss* (a single scalar mixing
        every target), not the per-target trajectory values needed for
        display; called alongside ``_record_params_grad`` whenever this
        replication's loss improves, so ``rep_best_trajs`` matches the
        params recorded at that same best-loss step.
        """
        trajs = {}
        with torch.no_grad():
            _apply_scale_group_params()
            state_run = _build_scaled_state()
            sim_all = generic_torch_simulate_calibration_target(
                state_run, ti["params_dict"], mc, RATE_TEMPLATE_REGISTRY,
                ti["precomputed"], ti["schedules_dict"],
                sim_days, ts_per_day, all_tvs_union, all_comps_union,
            )
            for k in range(n):
                parts = [sim_all[v] for v in target_vars_list[k]]
                sim_k = parts[0]
                for pt in parts[1:]:
                    sim_k = sim_k + pt
                if target_modes[k] == "scalar":
                    row_sims = []
                    for row in targets[k].observed:
                        sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_idxs[k], age_idxs[k], risk_idxs[k])
                        sliced = _slice_tensor(sim_k, sp_i, ag_i, rk_i)
                        row_sims.append(float(sliced.sum().item()) / _GRAD_SCALE)
                    trajs[f"target_{k}"] = row_sims
                elif target_modes[k] == "proportion":
                    row_sims = []
                    for ri_prop, row in enumerate(targets[k].observed):
                        sp_i, ag_i, rk_i = _parse_scalar_row_idxs(row, subpop_name_to_idx, sp_idxs[k], age_idxs[k], risk_idxs[k])
                        if sp_i < 0 and ag_i < 0 and rk_i < 0:
                            sp_i = ri_prop
                        den_sp = sp_i if (sp_i >= 0 and (ag_i >= 0 or rk_i >= 0)) else -1
                        den = _slice_tensor(sim_k, den_sp, -1, -1).sum()
                        num = _slice_tensor(sim_k, sp_i, ag_i, rk_i).sum()
                        row_sims.append(float((num / (den + 1e-10)).item()))
                    trajs[f"target_{k}"] = row_sims
                else:
                    days_k = num_fit_days_per_target[k]
                    sliced = _slice_tensor(sim_k, sp_idxs[k], age_idxs[k], risk_idxs[k])
                    trajs[f"target_{k}"] = (sliced[:days_k] / _GRAD_SCALE).numpy().tolist()
        return trajs

    n_it = int(fit_config.n_iter)
    lr_val = float(fit_config.lr)

    for ssi, comp_ss in enumerate(ctx.seed_scale_keys):
        sval = float(lhs_row[ctx.n_regular_lhs + ssi])
        scale_tensors[comp_ss] = torch.tensor(sval, dtype=torch.float64, requires_grad=True)

    # Scale-group multiplier tensors — their LHS columns follow the seed-scale
    # columns (see _run_gradient_fit's lhs_lo_list ordering).
    _sg_col0 = ctx.n_regular_lhs + len(ctx.seed_scale_keys)
    for gi, _m in enumerate(ctx.scale_group_keys):
        mval = float(lhs_row[_sg_col0 + gi])
        # Granularity (age/risk/subpop), same as regular fitted params below:
        # the whole tensor is seeded with this replication's scalar LHS draw,
        # then its elements diverge independently under gradient steps.
        _mdims = dim_vals[selected_params.index(_m)] if dim_vals else []
        _m_has_age = "age groups" in _mdims
        _m_has_risk = "risk groups" in _mdims
        if _m_has_age and _m_has_risk:
            mult_tensors[_m] = torch.full((ctx.A, ctx.R), mval, dtype=torch.float64, requires_grad=True)
        elif _m_has_age:
            mult_tensors[_m] = torch.full((ctx.A, 1), mval, dtype=torch.float64, requires_grad=True)
        elif _m_has_risk:
            mult_tensors[_m] = torch.full((1, ctx.R), mval, dtype=torch.float64, requires_grad=True)
        elif "subpopulation" in _mdims:
            _L = list(ti["state_dict"].values())[0].shape[0]
            mult_tensors[_m] = torch.full((_L, 1, 1), mval, dtype=torch.float64, requires_grad=True)
        else:
            mult_tensors[_m] = torch.tensor(mval, dtype=torch.float64, requires_grad=True)

    opt_tensors = list(scale_tensors.values()) + list(mult_tensors.values())
    # Per-tensor (lo, hi) bounds, parallel to opt_tensors, for the robust-step
    # move cap — in the leaf's own space (log10 for log params, via
    # _clamp_bounds). Seed-scale tensors come first (same order as
    # scale_tensors.values()), then scale-group multipliers, then fitted params
    # in selected_params order.
    opt_bounds = [
        _clamp_bounds(f"seed_scale_{_c}", lo_vals[ctx.seed_scale_comps[_c]], hi_vals[ctx.seed_scale_comps[_c]])
        for _c in ctx.seed_scale_keys
    ] + [
        _clamp_bounds(_m, lo_vals[selected_params.index(_m)], hi_vals[selected_params.index(_m)])
        for _m in ctx.scale_group_keys
    ]
    lhs_col = 0
    for j, pn in enumerate(selected_params):
        if pn not in ti["params_dict"]:
            continue
        dims = dim_vals[j] if dim_vals else []
        has_age = "age groups" in dims
        has_risk = "risk groups" in dims
        init_val = float(lhs_row[lhs_col])  # already log10 for a log param
        lhs_col += 1
        if has_age and has_risk:
            t = torch.full((ctx.A, ctx.R), init_val, dtype=torch.float64, requires_grad=True)
        elif has_age:
            t = torch.full((ctx.A, 1), init_val, dtype=torch.float64, requires_grad=True)
        elif has_risk:
            t = torch.full((1, ctx.R), init_val, dtype=torch.float64, requires_grad=True)
        elif "subpopulation" in dims:
            L = list(ti["state_dict"].values())[0].shape[0]
            t = torch.full((L, 1, 1), init_val, dtype=torch.float64, requires_grad=True)
        else:
            t = torch.tensor(init_val, dtype=torch.float64, requires_grad=True)
        if pn in log_params:
            # Leaf holds log10(value); params_dict[pn] = 10**leaf is refreshed
            # each forward by _apply_scale_group_params. Seed it now so any
            # pre-forward read sees a finite model-space value.
            log_reg_tensors[pn] = t
            ti["params_dict"][pn] = 10.0 ** t
        else:
            ti["params_dict"][pn] = t
        opt_tensors.append(t)
        opt_bounds.append(_clamp_bounds(pn, lo_vals[j], hi_vals[j]))

    rep_best_loss = float("inf")
    rep_best_params: dict = {}
    rep_best_trajs: dict = {}
    rep_loss_curve = []

    if not opt_tensors:
        with torch.no_grad():
            lv0 = float(_compute_total_loss().item())
        rep_loss_curve = [lv0]
        rep_best_loss = lv0
        rep_best_params = _record_params_grad()
        rep_best_trajs = _record_trajs_grad()
    elif fit_config.method == "adam":
        opt = torch.optim.Adam(opt_tensors, lr=lr_val)
        for _ in range(n_it):
            opt.zero_grad()
            loss = _compute_total_loss()
            loss.backward()
            prev_loss = float(loss.item())
            if robust_steps:
                prev_vals = [t.detach().clone() for t in opt_tensors]
            opt.step()
            _apply_bounds()
            if robust_steps:
                # 1) Cap each parameter's move to a fraction of its bound
                #    width so a single steep gradient can't fling it across
                #    a narrow optimum into a flat / divergent region.
                for _t, _pv, (_blo, _bhi) in zip(opt_tensors, prev_vals, opt_bounds):
                    _cap = STEP_CAP_FRAC * (_bhi - _blo)
                    if _cap > 0:
                        _t.data.copy_(_pv + (_t.data - _pv).clamp(-_cap, _cap))
                _apply_bounds()
                # 2) Backtracking line search: while the step increased the
                #    loss (or produced NaN), halve it back toward the previous
                #    point. Guarantees monotone, overshoot-proof descent.
                with torch.no_grad():
                    lv = float(_compute_total_loss().item())
                _bt = 0
                while (lv != lv or lv > prev_loss) and _bt < MAX_BACKTRACKS:
                    for _t, _pv in zip(opt_tensors, prev_vals):
                        _t.data.copy_((_t.data + _pv) / 2.0)
                    with torch.no_grad():
                        lv = float(_compute_total_loss().item())
                    _bt += 1
                if lv != lv or lv > prev_loss:
                    # No improving step found — stay put for this iteration.
                    for _t, _pv in zip(opt_tensors, prev_vals):
                        _t.data.copy_(_pv)
                    lv = prev_loss
            else:
                lv = float(loss.item())
            rep_loss_curve.append(lv)
            if lv < rep_best_loss:
                rep_best_loss = lv
                rep_best_params = _record_params_grad()
                rep_best_trajs = _record_trajs_grad()
    else:  # lbfgs
        opt = torch.optim.LBFGS(opt_tensors, lr=lr_val, max_iter=20)
        for _ in range(max(1, n_it // 20)):
            def _closure():
                opt.zero_grad()
                l = _compute_total_loss()
                l.backward()
                return l
            opt.step(_closure)
            _apply_bounds()
            with torch.no_grad():
                lv2 = float(_compute_total_loss().item())
            rep_loss_curve.append(lv2)
            if lv2 < rep_best_loss:
                rep_best_loss = lv2
                rep_best_params = _record_params_grad()
                rep_best_trajs = _record_trajs_grad()

    if fit_config.fit_start_offset:
        rep_best_params["epidemic_start_offset_days"] = sampled_offset
        rep_best_params["epidemic_start_date"] = start_rep

    return {
        "rep_loss_curve": rep_loss_curve,
        "rep_best_loss": rep_best_loss,
        "rep_best_params": rep_best_params,
        "rep_best_trajs": rep_best_trajs,
    }


def _resolve_n_workers(fit_config: FitConfig, n_rep: int) -> int:
    if not bool(getattr(fit_config, "parallel", True)):
        return 1
    explicit = getattr(fit_config, "n_workers", None)
    if explicit:
        return max(1, min(int(explicit), n_rep))
    cpu_count = os.cpu_count() or 2
    return max(1, min(cpu_count // 2, n_rep))


def _run_gradient_fit(
    config_dict, compartments, is_metapop, targets, fit_config,
    start_date, ts_per_day, seed_base, A, R, metapop_folder,
    metapop_travel_config, compartment_init, mobility_value, daily_vaccines_value, schedule_dfs,
    n, target_vars_list, target_modes, target_weights, weight_sum,
    sp_idxs, age_idxs, risk_idxs, subpop_name_to_idx,
    target_tvs, target_comps, sim_days, num_fit_days_per_target,
    selected_params, lo_vals, hi_vals, dim_vals, seed_scale_comps,
    progress_callback=None,
):
    """Multi-start Adam/L-BFGS driver: LHS-seed ``n_replications`` independent
    optimizations of ``_compute_total_loss`` and keep the best.

    Each replication is an independent local optimization (own model, own
    tensors), so different LHS starting points can land in different local
    minima — running several and keeping the lowest-loss one is the multi-
    start strategy that makes a local gradient method behave more like a
    global one. Replications are embarrassingly parallel (see
    ``_run_one_gradient_replication`` / ``_GradFitContext``) and run across a
    process pool when ``fit_config.parallel`` and ``n_replications`` allow it.

    LHS (Latin Hypercube Sampling) draws one row per replication over every
    fitted dimension (regular params in log10 space for log params, seed-scale
    factors, scale-group multipliers, and — if ``fit_start_offset`` — the
    epidemic start-date offset), so replications spread out over the full
    bounded search space rather than clustering near its center.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available. Install torch to use gradient-based fitting.")

    all_tvs_union = list(dict.fromkeys(tv for tvs in target_tvs for tv in tvs))
    all_comps_union = list(dict.fromkeys(c for cs in target_comps for c in cs))

    # One throwaway model build, purely to find which selected_params are
    # real torch-optimizable scalars (vs. seed-scale / start-offset, which
    # aren't in params_dict). Each replication below builds its own model.
    _probe_m, _probe_mc = _build_grad_model(
        config_dict, compartments, is_metapop, start_date, sim_days, 0,
        seed_base, ts_per_day, all_tvs_union, metapop_folder, metapop_travel_config,
        compartment_init, A, R, mobility_value, daily_vaccines_value, schedule_dfs,
    )
    _probe_ti = build_generic_torch_inputs(_probe_m, _probe_mc, sim_days)
    valid_pn_idxs = [j for j, pn in enumerate(selected_params) if pn in _probe_ti["params_dict"]]

    # Linked-scale groups: the multiplier is a single optimized scalar (not a
    # config param, so not in valid_pn_idxs); its base params are driven from it.
    scale_groups = _active_scale_groups(fit_config, selected_params)
    _resolve_scale_baselines(config_dict, scale_groups)  # validate bases exist
    scale_group_keys = list(scale_groups.keys())
    log_params = _active_log_params(fit_config, selected_params)

    if (not valid_pn_idxs and not seed_scale_comps and not scale_group_keys
            and not fit_config.fit_start_offset):
        raise ValueError("None of the specified parameters found in params dict. Check names.")

    seed_scale_keys = list(seed_scale_comps.keys())

    # LHS init lives in the optimizer's search space: log10(bound) for log
    # params (leaf holds log10(value)), linear otherwise.
    def _lb(name, val):
        return float(np.log10(val)) if name in log_params else float(val)
    lhs_lo_list = [_lb(selected_params[j], lo_vals[j]) for j in valid_pn_idxs]
    lhs_hi_list = [_lb(selected_params[j], hi_vals[j]) for j in valid_pn_idxs]
    for comp_ss in seed_scale_keys:
        j_ss = seed_scale_comps[comp_ss]
        lhs_lo_list.append(_lb(f"seed_scale_{comp_ss}", lo_vals[j_ss]))
        lhs_hi_list.append(_lb(f"seed_scale_{comp_ss}", hi_vals[j_ss]))
    # Scale-group multiplier columns follow the seed-scale columns.
    for _m in scale_group_keys:
        _jm = selected_params.index(_m)
        lhs_lo_list.append(_lb(_m, lo_vals[_jm]))
        lhs_hi_list.append(_lb(_m, hi_vals[_jm]))
    offset_lhs_col = len(lhs_lo_list)
    has_offset_col = bool(fit_config.fit_start_offset)
    if has_offset_col:
        lhs_lo_list.append(float(fit_config.start_offset_bounds[0]))
        lhs_hi_list.append(float(fit_config.start_offset_bounds[1]))

    n_rep = int(fit_config.n_replications)
    n_lhs_total = max(len(lhs_lo_list), 1)
    try:
        from scipy.stats.qmc import LatinHypercube
        lhs_unit = LatinHypercube(d=n_lhs_total).random(n=n_rep)
    except Exception:
        lhs_unit = np.random.rand(n_rep, n_lhs_total)
    lo_ext = np.array(lhs_lo_list) if lhs_lo_list else np.array([0.0])
    hi_ext = np.array(lhs_hi_list) if lhs_hi_list else np.array([1.0])
    lhs_scaled = lo_ext + lhs_unit[:, :len(lhs_lo_list)] * (hi_ext - lo_ext)

    ctx = _GradFitContext(
        config_dict=config_dict, compartments=compartments, is_metapop=is_metapop,
        targets=targets, fit_config=fit_config, start_date=start_date, ts_per_day=ts_per_day,
        seed_base=seed_base, A=A, R=R, metapop_folder=metapop_folder,
        metapop_travel_config=metapop_travel_config, compartment_init=compartment_init,
        mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
        schedule_dfs=schedule_dfs, n=n, target_vars_list=target_vars_list,
        target_modes=target_modes, target_weights=target_weights, weight_sum=weight_sum,
        sp_idxs=sp_idxs, age_idxs=age_idxs, risk_idxs=risk_idxs,
        subpop_name_to_idx=subpop_name_to_idx, all_tvs_union=all_tvs_union,
        all_comps_union=all_comps_union, sim_days=sim_days,
        num_fit_days_per_target=num_fit_days_per_target, selected_params=selected_params,
        lo_vals=lo_vals, hi_vals=hi_vals, dim_vals=dim_vals, seed_scale_comps=seed_scale_comps,
        seed_scale_keys=seed_scale_keys, n_regular_lhs=len(valid_pn_idxs),
        offset_lhs_col=offset_lhs_col, has_offset_col=has_offset_col,
        scale_groups=scale_groups, scale_group_keys=scale_group_keys,
        log_params=log_params,
    )

    n_workers = _resolve_n_workers(fit_config, n_rep)

    _par_label = f"parallel, {n_workers} workers" if n_workers > 1 else "sequential"
    print(
        f"[fitting] {n_rep} replications | method={fit_config.method}"
        f" | iter={fit_config.n_iter} | {_par_label}",
        flush=True,
    )

    if n_workers > 1 and n_rep > 1:
        if progress_callback:
            progress_callback({"method": fit_config.method, "phase": "start_parallel",
                               "rep": 0, "total": n_rep, "n_workers": n_workers})
        results: list = [None] * n_rep
        _done = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_grad_worker) as ex:
            futures = {
                ex.submit(_run_one_gradient_replication, rep_idx, lhs_scaled[rep_idx], ctx): rep_idx
                for rep_idx in range(n_rep)
            }
            for fut in as_completed(futures):
                rep_idx = futures[fut]
                r = fut.result()
                results[rep_idx] = r
                _done += 1
                print(
                    f"  rep {_done}/{n_rep} done"
                    f"  (rep {rep_idx + 1}, loss={r['rep_best_loss']:.4f})",
                    flush=True,
                )
                if progress_callback:
                    progress_callback({"method": fit_config.method, "phase": "rep_done",
                                       "rep": _done, "total": n_rep,
                                       "loss": r["rep_best_loss"]})
    else:
        results = []
        for rep_idx in range(n_rep):
            if progress_callback:
                progress_callback({"method": fit_config.method, "phase": "rep_starting",
                                   "rep": rep_idx + 1, "total": n_rep})
            r = _run_one_gradient_replication(rep_idx, lhs_scaled[rep_idx], ctx)
            results.append(r)
            print(
                f"  rep {rep_idx + 1}/{n_rep} done"
                f"  loss={r['rep_best_loss']:.4f}",
                flush=True,
            )
            if progress_callback:
                progress_callback({"method": fit_config.method, "phase": "rep_done",
                                   "rep": rep_idx + 1, "total": n_rep,
                                   "loss": r["rep_best_loss"]})

    best_loss = float("inf")
    best_params: dict = {}
    best_trajs: dict = {}
    all_rep_params = []
    all_rep_trajs = []
    loss_curve = []
    for r in results:
        loss_curve.append(r["rep_loss_curve"])
        all_rep_params.append(r["rep_best_params"])
        all_rep_trajs.append(r["rep_best_trajs"])
        if r["rep_best_loss"] < best_loss:
            best_loss = r["rep_best_loss"]
            best_params = r["rep_best_params"]
            best_trajs = r["rep_best_trajs"]

    print(f"[fitting] done — best loss across all reps: {best_loss:.4f}", flush=True)
    return best_params, loss_curve, all_rep_params, all_rep_trajs, best_trajs


# ---------------------------------------------------------------------------
# Accept-reject
# ---------------------------------------------------------------------------

def _run_accept_reject_fit(
    config_dict, compartments, is_metapop, targets, fit_config,
    start_date, ts_per_day, seed_base, A, R, metapop_folder,
    metapop_travel_config, compartment_init, subpop_names, mobility_value, daily_vaccines_value, schedule_dfs,
    n, target_vars_list, target_modes, target_weights, weight_sum,
    sp_idxs, age_idxs, risk_idxs, subpop_name_to_idx,
    target_tvs, target_comps, sim_days, num_fit_days_per_target,
    selected_params, lo_vals, hi_vals, dim_vals, seed_scale_comps,
    progress_callback=None,
):
    """Prior-predictive accept-reject calibration: draw ``n_iter`` independent
    uniform samples from the fitted bounds, simulate each, and keep every
    sample whose weighted R² clears ``fit_config.r2_threshold``.

    Unlike the gradient/Bayesian methods there is no optimization or adaptive
    proposal — each sample is an independent forward simulation from
    parameters drawn straight from the prior (uniform, or log-uniform for
    ``log_params``), so this method's cost scales linearly with ``n_iter``
    and its acceptance rate is entirely a function of how tight the bounds
    and threshold are relative to the model.

    Per sample:
    1. Draw a value for every ``selected_params`` entry (seed-scale factors,
       linked-scale multipliers, and array/per-subpop params expand into
       their own draws — mirroring ``_build_param_slots``' expansion rules
       for the Bayesian path) and, if enabled, a start-date offset.
    2. Build and run a fresh model with those parameters (seed scaling
       rescales the initial compartment counts, keeping total population
       fixed by pulling the delta from ``compartments[0]``; see
       ``_scale_compartment_init`` for the equivalent Bayesian-path logic).
    3. For each target, extract the simulated trajectory/rows and compute R²
       against the observed data (NaN-masked and optionally point-weighted
       for "ts" targets — see ``_weighted_rsquared``), then combine per-target
       R² into one weighted score via ``weight / weight_sum``.
    4. Track the single best-scoring sample (``best_params``/``best_trajs``)
       regardless of threshold, and separately collect every sample at or
       above ``r2_threshold`` into ``accepted_params_list`` /
       ``accepted_trajectories`` — this accepted set is the "posterior" the
       Forecast tab draws from, analogous to ``accepted_params`` from MCMC/
       ABC-SMC.
    """
    all_tvs_ar = list(dict.fromkeys(tv for tvs in target_tvs for tv in tvs))
    all_comps_ar = list(dict.fromkeys(c for cs in target_comps for c in cs))
    has_any_comp_ar = bool(all_comps_ar)

    _scale_groups = _active_scale_groups(fit_config, selected_params)
    _scale_baselines = _resolve_scale_baselines(config_dict, _scale_groups)
    _log_params = _active_log_params(fit_config, selected_params)

    n_subpops = 1
    if is_metapop and any("subpopulation" in (dim_vals[j] if dim_vals else []) for j in range(len(selected_params))):
        n_subpops = len(subpop_names) or 1

    metapop_base_inits = []
    if seed_scale_comps and is_metapop and metapop_folder:
        from pathlib import Path as _Path
        for sp_nm in subpop_names:
            ic_path = _Path(metapop_folder) / f"initial_conditions_{sp_nm}.json"
            ic_cfg = (config_dict.get("initial_conditions", {}) or {}).get(sp_nm, {})
            table_ci = read_initial_conditions(config_dict, sp_nm, compartments, A, R)
            comp_base = {c: build_scalar_array(0.0, A, R) for c in compartments}
            epi_base: dict = {}
            if ic_cfg.get("seeds") and table_ci is not None:
                comp_base = table_ci
            elif ic_path.exists():
                try:
                    with open(ic_path) as f:
                        ic_data = json.load(f)
                    for c, arr in ic_data.get("compartments", {}).items():
                        if c in compartments:
                            comp_base[c] = np.array(arr, dtype=float)
                    for em, arr in ic_data.get("epi_metrics", {}).items():
                        epi_base[em] = np.array(arr, dtype=float)
                except Exception:
                    pass
            elif table_ci is not None:
                comp_base = table_ci
            metapop_base_inits.append((comp_base, epi_base))

    rng = np.random.default_rng(seed_base)
    n_samples = int(fit_config.n_iter)
    r2_thresh = float(fit_config.r2_threshold)
    best_r2 = -float("inf")
    best_params: dict = {}
    best_trajs: dict = {}
    loss_curve = []
    accepted_params_list = []
    accepted_trajectories = []

    print(
        f"[fitting] {n_samples} accept-reject samples | R² threshold={r2_thresh:.2f}",
        flush=True,
    )
    _ar_print_every = max(1, n_samples // 20)

    for s_idx in range(n_samples):
        sampled: dict = {}
        per_subpop = [{} for _ in range(n_subpops)]
        sampled_scales: dict = {}
        sampled_offset = 0

        for j, pn in enumerate(selected_params):
            lo = float(lo_vals[j])
            hi = float(hi_vals[j])
            # Log-space params: draw uniformly in log10 and exponentiate, so the
            # stored value is the linear (model-space) quantity.
            _is_log = pn in _log_params
            _slo, _shi = (np.log10(lo), np.log10(hi)) if _is_log else (lo, hi)

            def _draw(size=None, _lo=_slo, _hi=_shi, _lg=_is_log):
                x = rng.uniform(_lo, _hi, size=size)
                return (10.0 ** x) if _lg else x

            if pn.startswith("seed_scale_"):
                comp_ss = pn[len("seed_scale_"):]
                if comp_ss in compartments:
                    sampled_scales[comp_ss] = float(_draw())
                continue
            dims = dim_vals[j] if dim_vals else []
            has_age = "age groups" in dims
            has_risk = "risk groups" in dims
            has_meta = "subpopulation" in dims
            if has_age and has_risk:
                inner_shape = (A, R)
            elif has_age:
                inner_shape = (A, 1)
            elif has_risk:
                inner_shape = (1, R)
            else:
                inner_shape = None
            if has_meta:
                for spi in range(n_subpops):
                    per_subpop[spi][pn] = (
                        _draw(inner_shape).tolist() if inner_shape else float(_draw())
                    )
            elif inner_shape:
                sampled[pn] = _draw(inner_shape).tolist()
            else:
                sampled[pn] = float(_draw())

        if fit_config.fit_start_offset:
            sampled_offset = int(rng.integers(fit_config.start_offset_bounds[0], fit_config.start_offset_bounds[1] + 1))
        start_iter = _shift_date(start_date, sampled_offset) if fit_config.fit_start_offset else start_date

        ci_iter = None
        init_override = None
        if sampled_scales:
            if is_metapop and metapop_base_inits:
                init_override = []
                for comp_base_sp, epi_base_sp in metapop_base_inits:
                    scaled_comp = {}
                    delta_sp = np.zeros_like(
                        comp_base_sp.get(compartments[0], build_scalar_array(0.0, A, R))
                    ) if compartments else None
                    for c in compartments:
                        base_arr = comp_base_sp.get(c, build_scalar_array(0.0, A, R))
                        if c in sampled_scales and c != (compartments[0] if compartments else None):
                            sc = sampled_scales[c]
                            scaled_comp[c] = base_arr * sc
                            if delta_sp is not None:
                                delta_sp += base_arr * (sc - 1.0)
                        else:
                            scaled_comp[c] = base_arr.copy()
                    if compartments and delta_sp is not None:
                        first_arr = comp_base_sp.get(compartments[0], build_scalar_array(0.0, A, R))
                        scaled_comp[compartments[0]] = np.maximum(first_arr - delta_sp, 0.0)
                    init_override.append((scaled_comp, epi_base_sp))
            elif not is_metapop and compartment_init is not None:
                ci_iter = {}
                delta = np.zeros_like(
                    compartment_init.get(compartments[0], build_scalar_array(0.0, A, R))
                ) if compartments else None
                for c in compartments:
                    base_arr = compartment_init.get(c, build_scalar_array(0.0, A, R))
                    if c in sampled_scales and c != (compartments[0] if compartments else None):
                        sc = sampled_scales[c]
                        ci_iter[c] = base_arr * sc
                        if delta is not None:
                            delta += base_arr * (sc - 1.0)
                    else:
                        ci_iter[c] = base_arr.copy()
                if compartments and delta is not None:
                    first_arr = compartment_init.get(compartments[0], build_scalar_array(0.0, A, R))
                    ci_iter[compartments[0]] = np.maximum(first_arr - delta, 0.0)

        has_per_subpop = any(per_subpop)
        # Expand any linked-scale multipliers into concrete base-param overrides
        # (multiplier stays in `sampled` for the recorded best/accepted params).
        model_overrides = _expand_scale_overrides(sampled, _scale_groups, _scale_baselines)
        if is_metapop:
            m, _ = make_metapop_from_folder(
                metapop_folder, config_dict, start_iter, sim_days, list(compartments),
                seed_offset=s_idx, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=False, tvs=all_tvs_ar, save_daily=has_any_comp_ar,
                param_overrides=model_overrides or None,
                param_overrides_per_subpop=per_subpop if has_per_subpop else None,
                travel_config=metapop_travel_config or None,
                init_states_override=init_override,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
            )
        else:
            ci_run = ci_iter if ci_iter is not None else compartment_init
            m, _, _ = make_single_pop_metapop(
                config_dict, start_iter, sim_days, ci_run,
                seed_offset=s_idx, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=False, tvs=all_tvs_ar, save_daily=has_any_comp_ar,
                param_overrides=model_overrides or None, travel_config=metapop_travel_config,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
                schedule_dfs=schedule_dfs,
            )
        m.simulate_until_day(sim_days)

        weighted_r2 = 0.0
        per_target_trajs: dict = {}
        skip = False
        for k in range(n):
            mode_k = target_modes[k]
            w_k = target_weights[k] / weight_sum
            n_days_k = num_fit_days_per_target[k]

            if mode_k == "scalar":
                row_sims = _extract_scalar_rows(
                    m, target_vars_list[k], targets[k].observed, n_days_k, subpop_name_to_idx,
                    sp_fallback=sp_idxs[k], ag_fallback=age_idxs[k], rk_fallback=risk_idxs[k],
                )
                obs_vals = [row["value"] for row in targets[k].observed]
                r2_k = compute_rsquared(obs_vals, row_sims)
                per_target_trajs[f"target_{k}"] = row_sims
            elif mode_k == "proportion":
                row_sims = _extract_proportion_rows(
                    m, target_vars_list[k], targets[k].observed, n_days_k, subpop_name_to_idx,
                    sp_fallback=sp_idxs[k], ag_fallback=age_idxs[k], rk_fallback=risk_idxs[k],
                )
                obs_vals = [row["value"] for row in targets[k].observed]
                r2_k = compute_rsquared(obs_vals, row_sims)
                per_target_trajs[f"target_{k}"] = row_sims
            else:
                sim_ts = _extract_ts_sliced(m, target_vars_list[k], sp_idxs[k], age_idxs[k], risk_idxs[k], n_days_k)
                obs_k = np.asarray(targets[k].observed[:n_days_k], dtype=float)
                if len(sim_ts) != len(obs_k):
                    skip = True
                    break
                # Mask NaN observed days (date-alignment gaps / leading lead-in).
                mask_k = ~np.isnan(obs_k)
                if targets[k].point_weights is not None:
                    pw_k = _aligned_point_weights(targets[k], n_days_k)[mask_k]
                    r2_k = _weighted_rsquared(obs_k[mask_k], np.asarray(sim_ts)[mask_k], pw_k)
                else:
                    r2_k = compute_rsquared(obs_k[mask_k].tolist(), np.asarray(sim_ts)[mask_k].tolist())
                per_target_trajs[f"target_{k}"] = sim_ts.tolist()

            weighted_r2 += w_k * r2_k

        if skip:
            continue

        loss_curve.append(float(weighted_r2))
        if (s_idx + 1) % _ar_print_every == 0 or s_idx == n_samples - 1:
            print(
                f"  sample {s_idx + 1}/{n_samples}"
                f"  R²={weighted_r2:.4f}"
                f"  accepted={len(accepted_params_list)}",
                flush=True,
            )
            if progress_callback:
                progress_callback({
                    "method": "ar",
                    "samples": s_idx + 1,
                    "total": n_samples,
                    "accepted": len(accepted_params_list),
                    "best_r2": best_r2,
                })
        if weighted_r2 > best_r2:
            best_r2 = weighted_r2
            best_params = dict(sampled)
            for spi, sp_d in enumerate(per_subpop):
                for kk, vv in sp_d.items():
                    best_params[f"{kk}_subpop{spi}"] = vv
            best_params.update({f"seed_scale_{c}": v for c, v in sampled_scales.items()})
            if fit_config.fit_start_offset:
                best_params["epidemic_start_offset_days"] = sampled_offset
                best_params["epidemic_start_date"] = start_iter
            best_trajs = dict(per_target_trajs)
        if weighted_r2 >= r2_thresh:
            acc = dict(sampled)
            for spi, sp_d in enumerate(per_subpop):
                for kk, vv in sp_d.items():
                    acc[f"{kk}_subpop{spi}"] = vv
            acc.update({f"seed_scale_{c}": v for c, v in sampled_scales.items()})
            if fit_config.fit_start_offset:
                acc["epidemic_start_offset_days"] = sampled_offset
                acc["epidemic_start_date"] = start_iter
            accepted_params_list.append(acc)
            accepted_trajectories.append(dict(per_target_trajs))

    print(
        f"[fitting] done — best R²={best_r2:.4f}"
        f"  accepted={len(accepted_params_list)}/{n_samples}",
        flush=True,
    )
    return best_params, loss_curve, accepted_params_list, accepted_trajectories, best_trajs


# ---------------------------------------------------------------------------
# Bayesian samplers (emcee MCMC / pyabc ABC-SMC)
# ---------------------------------------------------------------------------

def _scale_compartment_init(compartment_init, sampled_scales, compartments, A, R):
    """Scale seeded compartments by their sampled seed_scale factors, removing
    the added counts from the first compartment to keep the population fixed
    (mirrors the single-pop seed-scaling in accept-reject)."""
    if not sampled_scales or compartment_init is None:
        return compartment_init
    first = compartments[0] if compartments else None
    delta = np.zeros_like(
        compartment_init.get(first, build_scalar_array(0.0, A, R))
    ) if first else None
    ci = {}
    for c in compartments:
        base = compartment_init.get(c, build_scalar_array(0.0, A, R))
        if c in sampled_scales and c != first:
            sc = sampled_scales[c]
            ci[c] = base * sc
            if delta is not None:
                delta += base * (sc - 1.0)
        else:
            ci[c] = np.array(base, dtype=float)
    if first and delta is not None:
        ci[first] = np.maximum(compartment_init.get(first, build_scalar_array(0.0, A, R)) - delta, 0.0)
    return ci


def _nb2_loglike(obs: np.ndarray, mu: np.ndarray, phi: float, weights: np.ndarray | None = None) -> float:
    """Negative-Binomial (NB2) log-likelihood; gammaln handles fractional obs.

    `weights` (optional, same shape as `obs`) scales each day's log-likelihood
    contribution — the same per-timepoint emphasis (e.g. up-weighting the
    epidemic peak) the gradient/ABC-SMC paths apply via `point_weights`,
    normalized to mean 1 by the caller so the target's overall contribution
    is unchanged."""
    from scipy.special import gammaln
    mu = np.maximum(mu, 1e-6)
    ll = (
        gammaln(obs + phi) - gammaln(phi) - gammaln(obs + 1.0)
        + phi * (np.log(phi) - np.log(phi + mu))
        + obs * (np.log(mu) - np.log(phi + mu))
    )
    if weights is not None:
        ll = ll * weights
    return float(np.sum(ll))


def _build_param_slots(selected_params, dim_vals, A, R, n_subpops, lo_vals, hi_vals, log_params):
    """Expand each selected parameter into one scalar sampler "slot" per
    (subpop, age, risk) element implied by its granularity (``dim_vals``).

    A scalar parameter (empty dims) yields a single slot whose ``name`` **is**
    the parameter name, so the pre-existing single-scalar layout is preserved
    exactly (no regression for scalar fits). Array (age/risk) and per-subpop
    granularity each add a factor to the number of slots — mirroring the
    expansion rules the accept-reject path uses (see ``_run_accept_reject_fit``).
    Each slot inherits the parent's bounds and log-transform flag. Returns an
    ordered list of dicts (parents contiguous, subpop-major then age then risk).
    """
    slots = []
    for j, pn in enumerate(selected_params):
        dims = dim_vals[j] if dim_vals else []
        lo, hi, is_log = lo_vals[j], hi_vals[j], (pn in log_params)
        has_age = "age groups" in dims
        has_risk = "risk groups" in dims
        has_sub = "subpopulation" in dims
        expanded = has_age or has_risk or has_sub
        sub_slots = range(n_subpops) if has_sub else [None]
        age_slots = range(A) if has_age else [None]
        risk_slots = range(R) if has_risk else [None]
        for si in sub_slots:
            for ai in age_slots:
                for ri in risk_slots:
                    if not expanded:
                        name = pn
                    else:
                        parts = [pn]
                        if si is not None:
                            parts.append(f"sp{si}")
                        if ai is not None:
                            parts.append(f"a{ai}")
                        if ri is not None:
                            parts.append(f"r{ri}")
                        name = "|".join(parts)
                    slots.append(dict(
                        name=name, parent=pn, subpop_idx=si, age_idx=ai,
                        risk_idx=ri, lo=lo, hi=hi, is_log=is_log))
    return slots


_SLOT_TAG_RE = __import__("re").compile(r"^(a|r)(\d+)$")


def merge_param_slots(flat: dict) -> dict:
    """Reconstruct MCMC/ABC-SMC per-slot sampler output back into one
    array-valued entry per parent parameter.

    ``_build_param_slots`` gives every age/risk element of a granular fitted
    param its own scalar sampler dimension (name ``pn|a{i}``, ``pn|r{i}`` or
    ``pn|a{i}|r{i}``) so ``best_params``/``accepted_params`` can report a
    separate posterior marginal per element (see ``_out_params``) — the AR
    and gradient fitters instead record one nested-list value per param
    directly. This turns the former back into the latter representation, so
    code that treats a fitted param as one value per name (e.g. re-applying
    it as a model/scale-group override) works the same regardless of which
    method produced it.

    Non-slot keys (scalar params, ``seed_scale_*``, ``m_dlog_*``, ``phi``,
    …) and per-subpopulation slots (``pn|sp{i}...`` — a different naming
    convention, not produced for single-population fits) pass through
    unchanged.
    """
    groups: dict = {}
    out: dict = {}
    for k, v in flat.items():
        if "|" not in k:
            out[k] = v
            continue
        parts = k.split("|")
        parent = parts[0]
        tags: dict = {}
        ok = True
        for p in parts[1:]:
            m = _SLOT_TAG_RE.match(p)
            if not m:
                ok = False
                break
            tags[m.group(1)] = int(m.group(2))
        if not ok:
            out[k] = v
            continue
        groups.setdefault(parent, []).append((tags, v))

    for parent, entries in groups.items():
        A = max((t.get("a", 0) for t, _ in entries), default=0) + 1
        R = max((t.get("r", 0) for t, _ in entries), default=0) + 1
        arr = [[None] * R for _ in range(A)]
        for t, v in entries:
            a_rng = [t["a"]] if "a" in t else range(A)
            r_rng = [t["r"]] if "r" in t else range(R)
            for a in a_rng:
                for r in r_rng:
                    arr[a][r] = v
        out[parent] = arr
    return out


def prepare_param_sets(
    accepted_params: list, scale_groups: dict, config_params: dict
) -> list:
    """Turn raw fitted parameter sets into directly-applicable override dicts.

    ``accepted_params`` (posterior draws, AR-accepted samples or gradient
    replication bests) are stored in whichever representation the fitting
    method produced them. This normalises every set the same way a consumer
    re-simulating from a fit needs:

    - ``merge_param_slots`` reassembles MCMC/ABC-SMC per-element sampler
      columns (``pn|a0``, ``pn|a1``, …) into one array-valued ``pn`` entry;
    - linked ``scale_groups`` multipliers are expanded into concrete
      base-param overrides (``base := config_baseline × multiplier``) and the
      multiplier keys themselves are dropped, since they are not
      ``config["params"]`` entries and the model cannot apply them.

    ``seed_scale_*``, ``m_dlog_*`` and ``phi`` are left in place — they are
    not model params either, but each caller applies them differently
    (compartment-init scaling, an m(t) transmission schedule, observation
    noise), so splitting them out is the caller's job.

    ``config_params`` is the ``params`` block of the config the fit was run
    against, supplying the baselines the multipliers scale.
    """
    out = []
    for pset in accepted_params or []:
        merged = merge_param_slots(pset)
        if not scale_groups:
            out.append(merged)
            continue
        expanded = {k: v for k, v in merged.items() if k not in scale_groups}
        for mult_name, bases in scale_groups.items():
            if mult_name not in merged:
                continue
            raw = merged[mult_name]
            mult = (
                np.asarray(raw, dtype=float)
                if isinstance(raw, (list, tuple))
                else float(raw)
            )
            for base in bases:
                baseline = (config_params or {}).get(base)
                if isinstance(baseline, (list, tuple)) or isinstance(mult, np.ndarray):
                    expanded[base] = (np.asarray(baseline, dtype=float) * mult).tolist()
                elif baseline is not None:
                    expanded[base] = float(baseline) * mult
        out.append(expanded)
    return out


def _bayesian_setup(
    method,
    config_dict, compartments, is_metapop, targets, fit_config,
    start_date, ts_per_day, seed_base, A, R, metapop_folder,
    metapop_travel_config, compartment_init, subpop_names, mobility_value, daily_vaccines_value, schedule_dfs,
    n, target_vars_list, target_modes, target_weights, weight_sum,
    sp_idxs, age_idxs, risk_idxs, subpop_name_to_idx,
    target_tvs, target_comps, sim_days, num_fit_days_per_target,
    selected_params, lo_vals, hi_vals, dim_vals, seed_scale_comps,
):
    """Build the parameter layout, forward-simulate closure, likelihood /
    distance, and (for emcee) the ``_log_prob`` callable for a Bayesian fit.

    Kept separate from the samplers so an identical evaluation context can be
    rebuilt inside each multiprocessing worker from a picklable payload (the
    same kwargs), avoiding the need to pickle the closures themselves. Returns
    a ``SimpleNamespace`` of the pieces the sampler runners consume.
    """
    from types import SimpleNamespace

    # ── validate scope ───────────────────────────────────────────────────────
    # Array (age/risk) and per-subpop parameters ARE supported: each element is
    # expanded into its own scalar sampler dimension below (see
    # ``_build_param_slots``) and reconstructed before simulating. m(t) is
    # supported for both single-population and metapopulation models; for a
    # metapop a single shared m(t) trajectory is fit and broadcast identically
    # to every subpopulation (see the metapop branch of ``_simulate``).
    # Array-valued log params remain unsupported (run_fit's log-param check).
    tv_enabled = bool(getattr(fit_config, "tv_transmission", False))

    all_tvs = list(dict.fromkeys(tv for tvs in target_tvs for tv in tvs))
    all_comps = list(dict.fromkeys(c for cs in target_comps for c in cs))
    has_any_comp = bool(all_comps)

    _scale_groups = _active_scale_groups(fit_config, selected_params)
    _scale_baselines = _resolve_scale_baselines(config_dict, _scale_groups)
    _log_params = _active_log_params(fit_config, selected_params)

    # ── time-varying m(t) setup ───────────────────────────────────────────────
    cfg_run = config_dict
    knot_days: list[int] = []
    incr_names: list[str] = []
    if tv_enabled:
        cfg_run, n_foi = _inject_tv_transmission(config_dict)
        if n_foi == 0:
            raise ValueError(
                "Time-varying transmission enabled but the model has no "
                "force_of_infection transition to attach m(t) to."
            )
        knot_days = _tv_knot_days(sim_days, int(getattr(fit_config, "tv_knot_spacing_days", 30)))
        incr_names = _tv_increment_names(len(knot_days) - 1)
    n_incr = len(incr_names)
    tv_tau = float(getattr(fit_config, "tv_tau", 0.25))

    has_offset = bool(fit_config.fit_start_offset)
    off_lo, off_hi = fit_config.start_offset_bounds
    has_phi = (method == "mcmc")
    LOG10PHI_LO, LOG10PHI_HI = np.log10(0.1), np.log10(500.0)

    # Parameter vector layout / names. Each selected param is expanded into one
    # scalar sampler "slot" per (subpop, age, risk) element implied by its
    # granularity; scalar params keep their bare name (bit-identical layout).
    n_subpops = len(subpop_names) if subpop_names else 1
    param_slots = _build_param_slots(
        selected_params, dim_vals, A, R, n_subpops, lo_vals, hi_vals, _log_params)
    # Group slots by parent (parents are contiguous in param_slots); flag which
    # parents carry a per-subpop axis (their values go to param_overrides_per_subpop).
    _slots_by_parent: dict = {}
    for _s in param_slots:
        _slots_by_parent.setdefault(_s["parent"], []).append(_s)
    _has_subpop_param = any(_s["subpop_idx"] is not None for _s in param_slots)

    theta_names = [s["name"] for s in param_slots] + list(incr_names)
    if has_offset:
        theta_names.append("offset")
    if has_phi:
        theta_names.append("log10phi")
    ndim = len(theta_names)

    def _assemble_array(slots_for, named, has_age, has_risk, is_log):
        """Reconstruct a parameter's model-facing value from its slots. Returns
        a plain float when the parameter has no age/risk axis, else a nested
        (A×R) list, broadcasting each slot over any axis it does not index."""
        if not has_age and not has_risk:
            v = float(named[slots_for[0]["name"]])
            return float(10.0 ** v) if is_log else v
        arr = np.zeros((A, R), dtype=float)
        for s in slots_for:
            v = float(named[s["name"]])
            if is_log:
                v = float(10.0 ** v)
            a_rng = [s["age_idx"]] if s["age_idx"] is not None else range(A)
            r_rng = [s["risk_idx"]] if s["risk_idx"] is not None else range(R)
            for a in a_rng:
                for r in r_rng:
                    arr[a, r] = v
        return arr.tolist()

    def _named_to_components(named: dict):
        """Reconstruct model-facing parameters from a sampler position.

        The sampler (emcee/pyabc) works over a flat vector of scalar slots
        (see ``_build_param_slots`` / ``theta_names``); this regroups those
        slots back by parent parameter — exponentiating log-space slots back
        to linear/model space, and assembling per-age/risk slots into a
        nested array — into the ``(sampled, sampled_scales, incr, offset,
        phi, per_subpop)`` tuple ``_simulate``/``_loglike`` expect. The
        inverse of this regrouping (for output/recording) is ``_out_params``.
        """
        sampled: dict = {}
        sampled_scales: dict = {}
        per_subpop = [dict() for _ in range(n_subpops)] if _has_subpop_param else None
        for pn, slots in _slots_by_parent.items():
            is_seed = pn.startswith("seed_scale_") and pn[len("seed_scale_"):] in compartments
            is_log = slots[0]["is_log"]
            has_sub = slots[0]["subpop_idx"] is not None
            has_age = any(s["age_idx"] is not None for s in slots)
            has_risk = any(s["risk_idx"] is not None for s in slots)
            if not has_sub and not has_age and not has_risk:
                # Scalar parameter (single slot named exactly ``pn``).
                v = float(named[pn])
                if is_log:
                    v = float(10.0 ** v)
                if is_seed:
                    sampled_scales[pn[len("seed_scale_"):]] = v
                else:
                    sampled[pn] = v
            elif has_sub:
                # Per-subpopulation value (itself possibly an age/risk array).
                for si in range(n_subpops):
                    sub_slots = [s for s in slots if s["subpop_idx"] == si]
                    per_subpop[si][pn] = _assemble_array(sub_slots, named, has_age, has_risk, is_log)
            else:
                # Age/risk array, shared across subpops.
                sampled[pn] = _assemble_array(slots, named, has_age, has_risk, is_log)
        incr = np.array([float(named[nm]) for nm in incr_names], dtype=float) if incr_names else np.array([])
        offset = int(round(float(named["offset"]))) if has_offset else 0
        phi = float(10.0 ** float(named["log10phi"])) if "log10phi" in named else None
        return sampled, sampled_scales, incr, offset, phi, per_subpop

    # ── forward simulate → per-target trajectories ────────────────────────────
    def _extract_trajs(m) -> dict:
        """Pull each target's simulated trajectory/rows out of a completed
        model run `m`, in the ``{"target_k": [...]}`` shape both ``_loglike``
        and ``_distance`` compare against observed data."""
        trajs: dict = {}
        for k in range(n):
            nd = num_fit_days_per_target[k]
            if target_modes[k] == "scalar":
                trajs[f"target_{k}"] = _extract_scalar_rows(
                    m, target_vars_list[k], targets[k].observed, nd, subpop_name_to_idx,
                    sp_fallback=sp_idxs[k], ag_fallback=age_idxs[k], rk_fallback=risk_idxs[k])
            elif target_modes[k] == "proportion":
                trajs[f"target_{k}"] = _extract_proportion_rows(
                    m, target_vars_list[k], targets[k].observed, nd, subpop_name_to_idx,
                    sp_fallback=sp_idxs[k], ag_fallback=age_idxs[k], rk_fallback=risk_idxs[k])
            else:
                trajs[f"target_{k}"] = _extract_ts_sliced(
                    m, target_vars_list[k], sp_idxs[k], age_idxs[k], risk_idxs[k], nd).tolist()
        return trajs

    def _tvm_df_for(incr, start_iter):
        horizon = max(int(sim_days) + 14, 370)
        return _build_transmission_multiplier_df(incr, knot_days, start_iter, horizon)

    def _sdfs_with_tvm(tvm_df):
        base = schedule_dfs
        return SimpleNamespace(
            absolute_humidity_df=getattr(base, "absolute_humidity_df", None) if base else None,
            school_work_calendar_df=getattr(base, "school_work_calendar_df", None) if base else None,
            mobility_df=getattr(base, "mobility_df", None) if base else None,
            daily_vaccines_df=getattr(base, "daily_vaccines_df", None) if base else None,
            transmission_multiplier_df=tvm_df,
        )

    # ── build-once + reset fast path ──────────────────────────────────────────
    # Rebuilding the model from config every eval is ~40% of per-eval cost. When
    # the start date is fixed (no offset) and it's a single population, we can
    # instead build the model once per process and, each eval: swap the m(t)
    # schedule in place, restore baseline params + apply this eval's overrides
    # (modify_subpop_params is incremental), rebuild seeded compartment init_vals
    # (mirroring ConfigDrivenSubpopModel.create_compartments, including the
    # scheduled_exact pre-simulation adjustment), then reset_simulation() and
    # simulate. Verified bit-identical to the rebuild path (see reuse tests).
    _reuse = (bool(getattr(fit_config, "reuse_model", True))
              and not is_metapop and not has_offset)
    _has_seed_fit = any(p.startswith("seed_scale_") for p in selected_params)
    _cache: dict = {}

    def _reuse_build():
        """Build the single process-lifetime model instance the reuse path
        mutates and re-simulates on every eval (see ``_reuse_simulate``),
        caching it (and its pre-override baseline params) in ``_cache``.
        Runs once, lazily, on the first eval in a process — not per-draw.

        Base model: baseline params, baseline seeds, neutral (zeros) m(t).
        """
        sdfs0 = _sdfs_with_tvm(_tvm_df_for(np.zeros(n_incr), start_date)) if tv_enabled else schedule_dfs
        m, _, _ = make_single_pop_metapop(
            cfg_run, start_date, sim_days, compartment_init,
            seed_offset=0, seed_base=seed_base, ts_per_day=ts_per_day,
            stochastic=False, tvs=all_tvs, save_daily=has_any_comp,
            param_overrides=None, travel_config=metapop_travel_config,
            num_age_groups=A, num_risk_groups=R,
            mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
            schedule_dfs=sdfs0,
        )
        sub = next(iter(m.subpop_models.values()))
        _cache["m"] = m
        _cache["sub"] = sub
        _cache["name"] = next(iter(m.subpop_models.keys()))
        _cache["base_params"] = copy.deepcopy(sub.params.params)
        if tv_enabled:
            _cache["tvm_sched"] = sub.schedules["transmission_multiplier"]

    def _reuse_simulate(model_overrides, sampled_scales, incr):
        """Re-simulate the cached model (see ``_reuse_build``) for one new
        parameter draw without rebuilding it from config — the single-pop,
        fixed-start-date fast path ``_simulate`` takes when ``_reuse`` is on.

        Mutates the cached model in place, in this exact order (each step's
        comment below explains why the order matters): swap in the new m(t)
        schedule values, reset simulation state to day 0, rebuild any seeded
        compartment initial values from this draw's seed scales, then
        restore baseline params and apply this draw's overrides. Verified
        bit-identical to always rebuilding a fresh model per eval.
        """
        if not _cache:
            _reuse_build()
        m, sub = _cache["m"], _cache["sub"]
        # 1. m(t): overwrite the transmission-multiplier values in place. The
        #    schedule's date index is fixed across evals (only the increments
        #    change), so we assign the new value column onto the existing
        #    date-indexed frame — same result as rebuilding + re-parsing dates,
        #    but skips the (slow) per-eval date normalization.
        if tv_enabled:
            sched = _cache["tvm_sched"]
            sched.timeseries_df[sched.value_column] = build_transmission_multiplier_array(
                incr, knot_days, len(sched.timeseries_df))
        # 2. reset simulation state to day 0 FIRST (restores current_real_date to
        #    the start date and clears history). This must precede the seed step
        #    because the scheduled_exact pre-simulation adjustment is computed
        #    relative to current_real_date, which is left at the end of the
        #    previous eval's run otherwise.
        m.reset_simulation()
        # 3. seeds: rebuild seeded compartment init_vals from scaled seeds, then
        #    re-apply the scheduled_exact pre-simulation adjustment exactly as
        #    create_compartments does (setting init_val also sets current_val).
        #    Re-sync the state container so rate templates see the new values.
        #    Only needed when seed scales are fit.
        if _has_seed_fit:
            ci_run = _scale_compartment_init(compartment_init, sampled_scales, compartments, A, R)
            for cname in compartments:
                if ci_run is not None and cname in ci_run:
                    sub._state_init._cvals[cname] = np.asarray(ci_run[cname], dtype=float)
            adjustments = sub._compute_scheduled_exact_pre_simulation_adjustments()
            for cname in compartments:
                init = np.asarray(
                    sub._state_init._cvals.get(cname, np.zeros((A, R))), dtype=float)
                if cname in adjustments:
                    init = np.clip(init + adjustments[cname], 0.0, None)
                sub.compartments[cname].init_val = init
            sub.state.sync_to_current_vals(sub.all_state_variables)
        # 4. restore baseline params then apply this eval's overrides (list/array
        #    values → numpy, matching parse_model_config normalization).
        sub.params.params.clear()
        sub.params.params.update(copy.deepcopy(_cache["base_params"]))
        norm = {k: (np.asarray(v, dtype=float) if isinstance(v, (list, tuple)) else v)
                for k, v in (model_overrides or {}).items()}
        m.modify_subpop_params(_cache["name"], norm)
        m.simulate_until_day(sim_days)
        return _extract_trajs(m)

    def _simulate(sampled, sampled_scales, incr, offset, per_subpop=None):
        """Run one forward simulation for a candidate parameter draw and
        return its per-target trajectories (``{"target_k": [...]}``).

        Dispatches to one of two paths:
        - Reuse path (``_reuse``, single-pop + fixed start date only): swap
          the m(t) schedule and seeded-compartment init values into a model
          built once per process, then ``reset_simulation()`` + re-simulate —
          see ``_reuse_simulate`` for why the reset must precede the seed
          rebuild. ~40% cheaper per eval since it skips rebuilding the model
          from config every time.
        - Rebuild path (metapop, or any offset search): builds a fresh model
          from ``cfg_run`` each call, with ``model_overrides``/
          ``param_overrides_per_subpop`` applying this draw's parameters and
          (if ``tv_enabled``) a freshly-built transmission-multiplier
          schedule for these increments.

        Linked-scale multipliers in `sampled` are expanded into concrete
        base-param overrides via ``_expand_scale_overrides`` before either
        path runs — `sampled` itself keeps the multiplier value for
        recording/output.
        """
        start_iter = _shift_date(start_date, offset) if has_offset else start_date
        # Expand linked-scale multipliers into concrete base-param overrides;
        # `sampled` keeps the multiplier for output/recording. Array (age/risk)
        # params are already nested lists in `sampled` and pass through unchanged;
        # per-subpop params live in `per_subpop` and are routed separately below.
        model_overrides = _expand_scale_overrides(sampled, _scale_groups, _scale_baselines)
        if _reuse:
            # Single-pop fast path; per-subpop params only occur for metapop.
            return _reuse_simulate(model_overrides, sampled_scales, incr)
        sdfs_run = schedule_dfs
        if tv_enabled:
            sdfs_run = _sdfs_with_tvm(_tvm_df_for(incr, start_iter))
        if is_metapop:
            m, _ = make_metapop_from_folder(
                metapop_folder, cfg_run, start_iter, sim_days, list(compartments),
                seed_offset=0, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=False, tvs=all_tvs, save_daily=has_any_comp,
                param_overrides=model_overrides or None, travel_config=metapop_travel_config or None,
                param_overrides_per_subpop=per_subpop,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
                # One shared m(t) trajectory (from `cfg_run`'s injected schedule +
                # these per-eval increments) broadcast identically to every subpop.
                transmission_multiplier_df=(
                    sdfs_run.transmission_multiplier_df if tv_enabled else None
                ),
            )
        else:
            ci_run = _scale_compartment_init(compartment_init, sampled_scales, compartments, A, R)
            m, _, _ = make_single_pop_metapop(
                cfg_run, start_iter, sim_days, ci_run,
                seed_offset=0, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=False, tvs=all_tvs, save_daily=has_any_comp,
                param_overrides=model_overrides or None, travel_config=metapop_travel_config,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
                schedule_dfs=sdfs_run,
            )
        m.simulate_until_day(sim_days)
        return _extract_trajs(m)

    def _loglike(trajs, phi) -> float:
        """Total log-likelihood of one simulation (``_simulate`` output)
        under the emcee (MCMC) likelihood model, combined across targets by
        the same ``weight / weight_sum`` convex combination used elsewhere.

        - "ts" targets: Negative-Binomial (NB2) log-likelihood per day via
          ``_nb2_loglike``, with dispersion ``phi`` itself a fitted parameter
          (sampled in log10 space as ``log10phi``) — this lets the sampler
          learn how over-dispersed the observation noise is rather than
          assuming a fixed variance. NaN observed days are masked out, and
          ``point_weights`` (renormalized to mean 1) reweight emphasis
          between days within the target.
        - "scalar"/"proportion" targets: an approximate Gaussian
          log-likelihood with a heteroscedastic scale ``s = 0.1·|obs| + 1``
          (roughly 10% relative + a floor for near-zero observations), since
          these targets are single point estimates rather than count
          timeseries where NB2 applies.
        """
        total = 0.0
        for k in range(n):
            w_k = target_weights[k] / weight_sum
            nd = num_fit_days_per_target[k]
            sim = np.asarray(trajs[f"target_{k}"], dtype=float)
            if target_modes[k] == "ts":
                obs = np.asarray(targets[k].observed[:nd], dtype=float)
                mask = ~np.isnan(obs)
                if not mask.any():
                    continue
                pw = None
                if targets[k].point_weights is not None:
                    pw = _aligned_point_weights(targets[k], nd)[mask]
                    pw_mean = float(pw.mean()) if pw.size else 1.0
                    if pw_mean > 0:
                        pw = pw / pw_mean
                ll = _nb2_loglike(obs[mask], sim[:nd][mask], phi, weights=pw)
            else:
                obs = np.asarray([row["value"] for row in targets[k].observed], dtype=float)
                s = 0.1 * np.abs(obs) + 1.0
                resid = (sim[:len(obs)] - obs) / s
                ll = float(-0.5 * np.sum(resid ** 2) - np.sum(np.log(s)))
            total += w_k * ll
        return total

    def _distance(trajs) -> float:
        """Weighted-RMSE-style discrepancy for one simulation, the quantity
        ABC-SMC's sequential Monte Carlo minimizes (accepting particles whose
        distance falls under the generation's adaptively-shrinking epsilon).

        Per target, uses ``1 - R²`` (via ``compute_rsquared`` /
        ``_weighted_rsquared`` for point-weighted "ts" targets) rather than
        raw RMSE, so every target contributes on the same [0, ~1] scale
        regardless of its natural magnitude — mirroring how ``_compute_total_loss``
        normalizes by mean(obs²) for the same reason. Targets combine via the
        same ``weight / weight_sum`` convex combination as the other methods.
        Returns +inf (auto-reject) if any simulated value is non-finite, so a
        numerically unstable draw never accidentally scores well.
        """
        total = 0.0
        for k in range(n):
            w_k = target_weights[k] / weight_sum
            nd = num_fit_days_per_target[k]
            sim = np.asarray(trajs[f"target_{k}"], dtype=float)
            if not np.isfinite(sim).all():
                return float("inf")
            if target_modes[k] == "ts":
                obs = np.asarray(targets[k].observed[:nd], dtype=float)
                mask = ~np.isnan(obs)
                if not mask.any():
                    continue
                if targets[k].point_weights is not None:
                    pw = _aligned_point_weights(targets[k], nd)[mask]
                    r2 = _weighted_rsquared(obs[mask], sim[:nd][mask], pw)
                else:
                    r2 = compute_rsquared(obs[mask].tolist(), sim[:nd][mask].tolist())
            else:
                obs = [row["value"] for row in targets[k].observed]
                r2 = compute_rsquared(obs, list(sim[:len(obs)]))
            total += w_k * (1.0 - r2)
        return float(total)

    def _out_params(named: dict) -> dict:
        """Turn one sampler draw into the recorded params dict stored in
        ``best_params``/``accepted_params`` — the output-side counterpart of
        ``_named_to_components``.

        One recorded column per expanded sampler slot (linear space for log
        params), so each fitted degree of freedom gets its own posterior
        marginal. Scalar-param output is unchanged (slot name == param name);
        ``merge_param_slots`` later reconstructs the nested array shape from
        these per-slot keys when a granular param needs to be applied as one
        model override.
        """
        out = {}
        for s in param_slots:
            v = float(named[s["name"]])
            out[s["name"]] = float(10.0 ** v) if s["is_log"] else v
        for nm in incr_names:
            out[nm] = float(named[nm])
        if has_offset:
            off = int(round(float(named["offset"])))
            out["epidemic_start_offset_days"] = off
            out["epidemic_start_date"] = _shift_date(start_date, off)
        if "log10phi" in named:
            out["phi"] = float(10.0 ** float(named["log10phi"]))
        return out

    # ── emcee bounds + log posterior ─────────────────────────────────────────
    # Log params are explored in log10 space, so their sampler bounds (used for
    # the hard-bound check and walker init) are the log10 of the linear bounds;
    # _named_to_components exponentiates back to model space. Keyed per slot so
    # every expanded (age/risk/subpop) dimension carries the parent's bounds.
    def _sb(val, is_log):
        return float(np.log10(val)) if is_log else float(val)
    lo_by_name = {s["name"]: _sb(s["lo"], s["is_log"]) for s in param_slots}
    hi_by_name = {s["name"]: _sb(s["hi"], s["is_log"]) for s in param_slots}

    def _bounds_for(name):
        if name in lo_by_name:
            return lo_by_name[name], hi_by_name[name]
        if name == "offset":
            return float(off_lo), float(off_hi)
        if name == "log10phi":
            return LOG10PHI_LO, LOG10PHI_HI
        return None  # increment: unbounded (Gaussian prior)

    def _log_prob(theta):
        """emcee's target log-posterior for one walker position ``theta``
        (a flat vector in ``theta_names`` order — see ``_build_param_slots``
        for how granular params expand into scalar slots here).

        1. Hard-bound check: any coordinate outside its slot's ``[lo, hi]``
           (log10 space for log params; ``start_offset_bounds``/log10(phi)
           range for the offset/dispersion slots) returns ``-inf`` — a flat
           prior truncated to the fitted bounds, cheaply rejecting proposals
           before paying for a simulation.
        2. Log-prior on the m(t) increments: independent
           N(0, ``tv_tau``²) per knot-to-knot log-increment — a Gaussian
           random-walk smoothness prior (see the module-level tv_transmission
           docs) that penalizes month-to-month transmission swings without
           constraining the overall trend.
        3. Log-likelihood: reconstruct model-space parameters via
           ``_named_to_components``, forward-simulate (``_simulate``), and
           score with ``_loglike``. Any simulation exception (e.g. a
           degenerate parameter combination) is treated as ``-inf`` rather
           than propagating, so emcee's ensemble simply avoids that region.

        Returns prior + likelihood, or ``-inf`` if either is non-finite.
        """
        named = dict(zip(theta_names, theta))
        # structural / offset / phi hard bounds
        for nm in theta_names:
            b = _bounds_for(nm)
            if b is not None and not (b[0] <= named[nm] <= b[1]):
                return -np.inf
        sampled, scales, incr, offset, phi, per_sub = _named_to_components(named)
        lp = -0.5 * float(np.sum((incr / tv_tau) ** 2)) if len(incr) else 0.0
        try:
            trajs = _simulate(sampled, scales, incr, offset, per_sub)
            ll = _loglike(trajs, phi if phi is not None else 1.0)
        except Exception:
            return -np.inf
        total = lp + ll
        return total if np.isfinite(total) else -np.inf

    # ABC-SMC: pyabc stores model output as numeric per-target arrays, so build
    # an observed dict of the same shape (only used for DB storage; the actual
    # discrepancy is computed by `_distance` from the simulated trajectories).
    observed_dict = {}
    for k in range(n):
        nd = num_fit_days_per_target[k]
        if target_modes[k] == "ts":
            observed_dict[f"target_{k}"] = np.asarray(targets[k].observed[:nd], dtype=float)
        else:
            observed_dict[f"target_{k}"] = np.asarray(
                [row["value"] for row in targets[k].observed], dtype=float)

    return SimpleNamespace(
        method=method, fit_config=fit_config, seed_base=seed_base,
        theta_names=theta_names, ndim=ndim, param_slots=param_slots,
        lo_by_name=lo_by_name, hi_by_name=hi_by_name,
        selected_params=selected_params, lo_vals=lo_vals, hi_vals=hi_vals,
        log_params=_log_params, incr_names=incr_names, tv_tau=tv_tau,
        has_offset=has_offset, off_lo=off_lo, off_hi=off_hi,
        has_phi=has_phi, LOG10PHI_LO=LOG10PHI_LO, LOG10PHI_HI=LOG10PHI_HI,
        named_to_components=_named_to_components, simulate=_simulate,
        loglike=_loglike, distance=_distance, out_params=_out_params,
        bounds_for=_bounds_for, log_prob=_log_prob, observed_dict=observed_dict,
    )


# ── multiprocessing worker for emcee ──────────────────────────────────────────
# emcee's EnsembleSampler pickles the log-prob callable to each worker, so it
# must be a top-level function; the (unpicklable) simulate/likelihood closures
# are rebuilt once per process by the Pool initializer from the picklable
# payload — the same trick the MA_vax prototype uses to parallelise cleanly.
_BAYES_WORKER_LOG_PROB = None


def _init_bayes_worker(payload):
    global _BAYES_WORKER_LOG_PROB
    _BAYES_WORKER_LOG_PROB = _bayesian_setup(**payload).log_prob


def _bayes_log_prob_worker(theta):
    return _BAYES_WORKER_LOG_PROB(theta)


def _bayes_worker_count(fit_config, cap):
    """Number of processes for a Bayesian sampler.

    Bayesian log-prob / distance evals are CPU-light (one deterministic sim
    each), so — unlike the memory-heavy gradient replications that use cpu//2 —
    it pays to use nearly all cores, capped at ``cap`` (the max number of evals
    the sampler runs concurrently: nwalkers//2 for emcee, population_size for
    ABC-SMC). Honors the FitConfig ``parallel`` toggle and explicit
    ``n_workers`` override.
    """
    if not bool(getattr(fit_config, "parallel", True)):
        return 1
    if getattr(fit_config, "n_workers", None):
        return max(1, min(int(fit_config.n_workers), cap))
    return max(1, min((os.cpu_count() or 2) - 1, cap))


def _run_bayesian_fit(
    method,
    config_dict, compartments, is_metapop, targets, fit_config,
    start_date, ts_per_day, seed_base, A, R, metapop_folder,
    metapop_travel_config, compartment_init, subpop_names, mobility_value, daily_vaccines_value, schedule_dfs,
    n, target_vars_list, target_modes, target_weights, weight_sum,
    sp_idxs, age_idxs, risk_idxs, subpop_name_to_idx,
    target_tvs, target_comps, sim_days, num_fit_days_per_target,
    selected_params, lo_vals, hi_vals, dim_vals, seed_scale_comps,
    progress_callback=None,
):
    """emcee (method='mcmc') / pyabc (method='abc-smc') calibration.

    Supports scalar, array (per-age/risk) and per-subpopulation parameters: each
    element is expanded into its own scalar sampler dimension and reconstructed
    before simulating (see ``_build_param_slots`` / ``_bayesian_setup``). Returns
    the same 5-tuple as the other methods; the posterior ensemble is delivered
    through ``accepted_params`` / ``accepted_trajectories``.
    """
    # `payload` is the picklable set of kwargs `_bayesian_setup` needs; it is
    # both built once in-process (for walker init / post-processing) and shipped
    # to each multiprocessing worker to rebuild an identical eval context.
    payload = dict(
        method=method,
        config_dict=config_dict, compartments=compartments, is_metapop=is_metapop,
        targets=targets, fit_config=fit_config,
        start_date=start_date, ts_per_day=ts_per_day, seed_base=seed_base, A=A, R=R,
        metapop_folder=metapop_folder, metapop_travel_config=metapop_travel_config,
        compartment_init=compartment_init, subpop_names=subpop_names,
        mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
        schedule_dfs=schedule_dfs,
        n=n, target_vars_list=target_vars_list, target_modes=target_modes,
        target_weights=target_weights, weight_sum=weight_sum,
        sp_idxs=sp_idxs, age_idxs=age_idxs, risk_idxs=risk_idxs,
        subpop_name_to_idx=subpop_name_to_idx,
        target_tvs=target_tvs, target_comps=target_comps, sim_days=sim_days,
        num_fit_days_per_target=num_fit_days_per_target,
        selected_params=selected_params, lo_vals=lo_vals, hi_vals=hi_vals,
        dim_vals=dim_vals, seed_scale_comps=seed_scale_comps,
    )
    ctx = _bayesian_setup(**payload)
    if method == "mcmc":
        return _emcee_run(ctx, payload, progress_callback=progress_callback)
    return _abc_smc_run(ctx, payload, progress_callback=progress_callback)


def _emcee_run(ctx, payload, progress_callback=None):
    """Drive emcee's affine-invariant ensemble MCMC on ``ctx.log_prob``
    (see ``_log_prob``) and turn the resulting chain into a posterior sample.

    - Initializes ``nwalkers = max(2·ndim, fit_config.n_walkers)`` walkers
      uniformly within each bounded dimension (pulled in 20% from the edges
      to avoid starting exactly on a hard boundary); m(t) increment
      dimensions (unbounded, Gaussian-prior) start near 0 instead.
    - Runs ``fit_config.n_iter`` ensemble steps, optionally across a process
      pool (see ``_init_bayes_worker``/``_bayes_log_prob_worker`` — emcee
      pickles the log-prob to each worker, so a picklable ``payload``
      rebuilds an identical eval context once per process rather than
      pickling closures directly).
    - Post-processing: discard the first ``mcmc_burnin`` steps, keep every
      ``mcmc_thin``-th step, then drop walkers whose mean post-burn-in
      log-probability is a robust outlier (median/MAD cut, falling back to a
      25th-percentile cut if that would drop too many) — these are walkers
      stuck in a low-probability region rather than sampling the posterior.
    - The surviving (step, walker) draws are flattened into the posterior
      ensemble (capped at ``MAX_DRAWS`` for downstream plotting), each
      re-simulated to populate ``accepted_trajectories`` (capped at
      ``MAX_TRAJS``), and the highest-log-prob draw becomes ``best_params``.
    """
    try:
        import emcee
    except ImportError:
        raise RuntimeError("emcee not installed. `uv add emcee` to use MCMC fitting.")

    fit_config, seed_base = ctx.fit_config, ctx.seed_base
    theta_names, ndim = ctx.theta_names, ctx.ndim
    _bounds_for = ctx.bounds_for
    _named_to_components, _simulate, _out_params = (
        ctx.named_to_components, ctx.simulate, ctx.out_params)

    nwalkers = max(2 * ndim, int(fit_config.n_walkers))
    nsteps = int(fit_config.n_iter)
    rng = np.random.default_rng(seed_base)
    p0 = np.empty((nwalkers, ndim))
    for j, nm in enumerate(theta_names):
        b = _bounds_for(nm)
        if b is not None:
            pad = 0.2 * (b[1] - b[0])
            p0[:, j] = rng.uniform(b[0] + pad, b[1] - pad, size=nwalkers)
        else:
            p0[:, j] = rng.normal(0.0, 0.05, size=nwalkers)  # increments near 0

    # True multiprocessing: the simulation is pure-Python and holds the GIL, so
    # a thread pool gives no speed-up. Each worker rebuilds its own eval context
    # once via the Pool initializer (from the picklable `payload`), then only
    # small parameter vectors / floats cross the process boundary per step —
    # mirroring the MA_vax prototype's multiprocessing.Pool approach.
    # emcee evaluates up to nwalkers/2 log-probs at once, so cap workers there.
    n_workers = _bayes_worker_count(fit_config, cap=nwalkers // 2)
    pool = None
    log_prob_fn = ctx.log_prob
    if n_workers > 1:
        try:
            _mpctx = mp.get_context("spawn")
            pool = _mpctx.Pool(processes=n_workers,
                               initializer=_init_bayes_worker, initargs=(payload,))
            log_prob_fn = _bayes_log_prob_worker
        except Exception as _e:  # pragma: no cover - falls back to single process
            print(f"[fitting] MCMC: could not start process pool ({_e}); "
                  f"running single-process", flush=True)
            pool = None
            log_prob_fn = ctx.log_prob
    par_label = f"{n_workers} processes" if pool is not None else "single-process"
    print(f"[fitting] MCMC (emcee): {nwalkers} walkers x {nsteps} steps, ndim={ndim}, {par_label}", flush=True)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, pool=pool)
    _mcmc_report_every = max(1, nsteps // 20)
    try:
        for _step_i, _state in enumerate(sampler.sample(p0, iterations=nsteps, progress=False)):
            if progress_callback and (_step_i + 1) % _mcmc_report_every == 0:
                progress_callback({
                    "method": "mcmc",
                    "step": _step_i + 1,
                    "total": nsteps,
                    "walkers": nwalkers,
                    "mean_log_prob": float(np.mean(_state.log_prob)),
                })
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    burn = min(int(fit_config.mcmc_burnin), max(0, nsteps - 1))
    thin = max(1, int(fit_config.mcmc_thin))
    lp_chain = sampler.get_log_prob()  # (nsteps, nwalkers)
    # Drop walkers stuck far below the main cluster (robust median/MAD cut).
    mean_lp = np.mean(lp_chain[burn:], axis=0)
    med, mad = np.median(mean_lp), np.median(np.abs(mean_lp - np.median(mean_lp)))
    good = mean_lp >= med - 5.0 * (mad + 1e-9)
    if good.sum() < 0.75 * nwalkers:
        good = mean_lp >= np.percentile(mean_lp, 25)

    chain = sampler.get_chain(discard=burn, thin=thin)          # (kept, nwalkers, ndim)
    lp_kept = sampler.get_log_prob(discard=burn, thin=thin)
    flat = chain[:, good, :].reshape(-1, ndim)
    flat_lp = lp_kept[:, good].reshape(-1)
    loss_curve = [float(v) for v in np.mean(lp_chain, axis=1)]   # mean log-prob per step

    # Cap stored draws / trajectories for downstream plotting performance.
    MAX_DRAWS, MAX_TRAJS = 4000, 300
    if len(flat) > MAX_DRAWS:
        sel = rng.choice(len(flat), size=MAX_DRAWS, replace=False)
        flat, flat_lp = flat[sel], flat_lp[sel]

    accepted_params_list = []
    for row in flat:
        named = dict(zip(theta_names, row))
        accepted_params_list.append(_out_params(named))

    best_i = int(np.argmax(flat_lp)) if len(flat_lp) else 0
    best_named = dict(zip(theta_names, flat[best_i])) if len(flat) else {}
    best_sampled, best_scales, best_incr, best_offset, best_phi, best_per_sub = _named_to_components(best_named)
    best_params = _out_params(best_named)

    traj_idx = np.arange(len(flat)) if len(flat) <= MAX_TRAJS else rng.choice(len(flat), size=MAX_TRAJS, replace=False)
    accepted_trajectories = []
    for i in traj_idx:
        named = dict(zip(theta_names, flat[i]))
        s, sc, ic, off, _, ps = _named_to_components(named)
        try:
            accepted_trajectories.append(_simulate(s, sc, ic, off, ps))
        except Exception:
            pass
    best_trajs = _simulate(best_sampled, best_scales, best_incr, best_offset, best_per_sub)

    print(f"[fitting] MCMC done — {len(accepted_params_list)} posterior draws "
          f"(dropped {int(nwalkers - good.sum())}/{nwalkers} walkers)", flush=True)
    return best_params, loss_curve, accepted_params_list, accepted_trajectories, best_trajs


def _abc_smc_run(ctx, payload, progress_callback=None):
    """Drive pyabc's ABC-SMC (Approximate Bayesian Computation, Sequential
    Monte Carlo) on ``ctx.distance`` (see ``_distance``) and turn the final
    generation's weighted particle population into a posterior sample.

    ABC-SMC approximates the posterior without an explicit likelihood: each
    generation samples particles from the previous generation's weighted
    population (or the prior, for generation 0), perturbs them, simulates,
    and keeps those whose ``_distance`` falls under that generation's
    acceptance threshold epsilon. Epsilon shrinks generation-over-generation
    (pyabc's adaptive MedianEpsilon), so the population is progressively
    driven toward better-fitting regions — up to ``fit_config.abc_max_gens``
    generations of ``fit_config.abc_pop_size`` accepted particles each.

    - Priors: uniform over each slot's bounds (log10 space for log params),
      N(0, tv_tau) for m(t) increments — mirroring ``_log_prob``'s prior.
    - ``_model``/``_dist`` wrap ``_simulate``/``_distance`` in the numeric
      dict shape pyabc expects, treating a simulation exception or
      non-finite output as an automatic-reject (infinite distance).
    - Parallelized via a spawn ``multiprocessing.Pool`` fed through pyabc's
      ``MappingSampler`` (pyabc's own multicore samplers rely on fork, unsafe
      under macOS spawn) — see ``_bayes_worker_count``.
    - The generation loop is driven manually (``initialize_components_before_run``
      once, then repeated ``run_generation``) rather than via ``abc.run()``
      per generation, because re-calling ``abc.run()`` each generation resets
      epsilon back to its prior-calibration value instead of letting it
      adapt — see the inline comment at the call site for the full story
      (this was a real bug: epsilon never tightened and the posterior stayed
      ≈ the prior).
    - Final posterior: the last generation's particles/weights
      (``history.get_distribution``), resampled down to ``MAX_TRAJS`` (by
      weight) for ``accepted_trajectories``; ``best_params`` is the
      highest-weight particle.
    """
    try:
        import pyabc
    except ImportError:
        raise RuntimeError("pyabc not installed. `uv add pyabc` to use ABC-SMC fitting.")

    fit_config, seed_base = ctx.fit_config, ctx.seed_base
    theta_names = ctx.theta_names
    incr_names, tv_tau = ctx.incr_names, ctx.tv_tau
    off_lo, off_hi = ctx.off_lo, ctx.off_hi
    # Per-slot bounds already in log10 space for log params (built in setup); each
    # expanded age/risk/subpop dimension carries the parent's transformed bounds.
    lo_by_name, hi_by_name = ctx.lo_by_name, ctx.hi_by_name
    _named_to_components, _simulate = ctx.named_to_components, ctx.simulate
    _distance, _out_params, observed_dict = ctx.distance, ctx.out_params, ctx.observed_dict

    # Log params get a uniform prior in log10 space (LogUniform on the linear
    # value); _named_to_components exponentiates back to model space.
    prior_dict = {}
    for nm in theta_names:
        if nm in lo_by_name:
            prior_dict[nm] = pyabc.RV("uniform", lo_by_name[nm], hi_by_name[nm] - lo_by_name[nm])
        elif nm == "offset":
            prior_dict[nm] = pyabc.RV("uniform", float(off_lo), float(off_hi) - float(off_lo))
        else:  # increment
            prior_dict[nm] = pyabc.RV("norm", 0.0, tv_tau)
    prior = pyabc.Distribution(**prior_dict)

    def _model(named):
        s, sc, ic, off, _, ps = _named_to_components(dict(named))
        try:
            trajs = _simulate(s, sc, ic, off, ps)
            return {key: np.asarray(val, dtype=float) for key, val in trajs.items()}
        except Exception:
            # Return NaN sentinels so _dist can safely reject this particle.
            return {key: np.full_like(np.asarray(v, dtype=float), np.nan)
                    for key, v in observed_dict.items()}

    def _dist(x, x0):
        # Reject any particle whose simulation produced non-finite values.
        if any(not np.isfinite(np.asarray(v, dtype=float)).all() for v in x.values()):
            return float("inf")
        return _distance(x)

    pop = int(fit_config.abc_pop_size)
    gens = int(fit_config.abc_max_gens)

    # True multiprocessing, mirroring the emcee path. pyabc's own multicore
    # samplers rely on fork (unavailable/unsafe under macOS spawn), so instead
    # we drive a spawn multiprocessing.Pool through pyabc's MappingSampler, which
    # cloudpickles the per-particle sample function (closing over _model/_dist)
    # to each worker. Each particle still rebuilds its own model per eval — the
    # pool just spreads the population across cores. Capped at the population
    # size (the max particles sampled concurrently per generation).
    n_workers = _bayes_worker_count(fit_config, cap=pop)
    pool = None
    if n_workers > 1:
        try:
            _mpctx = mp.get_context("spawn")
            pool = _mpctx.Pool(processes=n_workers)
            sampler = pyabc.sampler.MappingSampler(map_=pool.map, mapper_pickles=False)
        except Exception as _e:  # pragma: no cover - falls back to single core
            print(f"[fitting] ABC-SMC: could not start process pool ({_e}); "
                  f"running single-core", flush=True)
            if pool is not None:
                pool.terminate()
            pool = None
            sampler = pyabc.sampler.SingleCoreSampler()
    else:
        sampler = pyabc.sampler.SingleCoreSampler()
    par_label = f"{n_workers} processes" if pool is not None else "single-core"
    print(f"[fitting] ABC-SMC (pyabc): pop={pop} x up to {gens} generations, {par_label}", flush=True)

    try:
        abc = pyabc.ABCSMC(_model, prior, _dist, population_size=pop, sampler=sampler)
        db_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"fitting_abc_{seed_base}.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        abc.new("sqlite:///" + db_path, observed_dict)

        # Run the SMC generation loop manually rather than calling
        # `abc.run(max_nr_populations=1)` per generation. Each `abc.run()` call
        # re-executes `initialize_components_before_run` →
        # `_initialize_dist_eps_acc`, which re-samples from the prior and RESETS
        # the (adaptive MedianEpsilon) threshold back to its prior-calibration
        # value — discarding the per-generation `eps.update()` that `run()`
        # normally performs via `_prepare_next_iteration`. Looping over
        # single-population runs therefore pins epsilon at the initial prior
        # tolerance forever, so the SMC never tightens and the posterior stays
        # ≈ the prior (a wide, badly-fitting ensemble). Instead we do what
        # `run()` does internally: initialize the components ONCE, then step
        # through `run_generation`, which keeps epsilon adapting while still
        # letting us emit a per-generation progress callback.
        t0 = abc.initialize_components_before_run(
            minimum_epsilon=0.0,
            max_nr_populations=gens,
            min_acceptance_rate=0.0,
            max_total_nr_simulations=np.inf,
            max_walltime=None,
            min_eps_diff=0.0,
        )
        history = abc.history
        try:
            t = t0
            for _g in range(gens):
                _current_eps = float(abc.eps(t))
                ret = abc.run_generation(t=t)
                print(f"  generation {t + 1}/{gens}  eps={_current_eps:.4g}", flush=True)
                if progress_callback:
                    progress_callback({
                        "method": "abc-smc",
                        "generation": t + 1,
                        "total": gens,
                        "epsilon": _current_eps,
                    })
                if not ret.get("successful", False) or abc.check_terminate(
                    t=t, acceptance_rate=ret.get("acceptance_rate", 0.0)
                ):
                    break
                t += 1
        finally:
            # Mirror the cleanup that ABCSMC.run's wrapper normally performs.
            abc.history.done(end_time=datetime.now())
            abc.sampler.stop()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    df, weights = history.get_distribution(t=history.max_t)
    eps = history.get_all_populations()["epsilon"].to_numpy()
    loss_curve = [float(e) for e in eps if np.isfinite(e)]

    rng = np.random.default_rng(seed_base)
    names_in_df = [nm for nm in theta_names if nm in df.columns]
    samples = df[names_in_df].to_numpy()
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else None

    accepted_params_list = []
    for row in samples:
        named = dict(zip(names_in_df, row))
        accepted_params_list.append(_out_params(named))

    # Best = highest-weight particle.
    best_i = int(np.argmax(weights)) if len(weights) else 0
    best_named = dict(zip(names_in_df, samples[best_i])) if len(samples) else {}
    bs, bsc, bic, boff, _, bps = _named_to_components(best_named)
    best_params = _out_params(best_named)

    MAX_TRAJS = 300
    idx = (np.arange(len(samples)) if len(samples) <= MAX_TRAJS
           else rng.choice(len(samples), size=MAX_TRAJS, replace=False, p=w))
    accepted_trajectories = []
    for i in idx:
        named = dict(zip(names_in_df, samples[i]))
        s, sc, ic, off, _, ps = _named_to_components(named)
        try:
            accepted_trajectories.append(_simulate(s, sc, ic, off, ps))
        except Exception:
            pass
    best_trajs = _simulate(bs, bsc, bic, boff, bps)

    print(f"[fitting] ABC-SMC done — {len(accepted_params_list)} posterior particles, "
          f"final eps={eps[-1]:.4g}", flush=True)
    return best_params, loss_curve, accepted_params_list, accepted_trajectories, best_trajs
