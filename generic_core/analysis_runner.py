"""Scenario x replicate simulation runner for the notebook's Analysis tab.

Lives outside the marimo notebook (unlike the loop it replaces, which was
nested inside a single ``@app.cell``) so its worker function is a real,
independently-importable module-level function -- required for
``multiprocessing`` to hand it to a spawned worker process. A live marimo
kernel (``marimo edit``/``marimo run``) parses the notebook file into cell
ASTs and executes each cell against an ephemeral per-session globals dict; it
never ``import``s the notebook as a module, so nothing defined inside a cell
is resolvable by a spawned worker, and cell-nested closures aren't picklable
in the first place. This mirrors the same fix already used for the Fitting
tab's Bayesian samplers (see ``_init_bayes_worker``/``_bayes_log_prob_worker``
in ``generic_core/fitting.py``): every worker-facing function stays here, a
plain-data ``payload`` dict is shipped to each worker once via the
``multiprocessing.Pool`` initializer, and each worker rebuilds its own local
run context from that payload instead of pickling a closure.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import multiprocessing as mp
import os

import numpy as np

from generic_core.model_factory import (
    make_metapop_from_folder,
    make_single_pop_metapop,
    extract_history,
    extract_history_full,
)
from generic_core.fitting import (
    _scale_compartment_init,
    _tv_knot_days,
    _build_transmission_multiplier_df,
)


def _split_dose_mult(mult):
    # Scenario dose multipliers are a {df_attribute: per-age-group vector}
    # dict, one entry per vaccine_schedule the user actually scaled.
    # make_single_pop_metapop/make_metapop_from_folder take the default
    # schedule's vector as `dose_mult` and the rest as an `extra_dose_mult`
    # dict, so split on that boundary here. Returns (dose_mult, extra_dose_mult).
    if not mult:
        return None, None
    base = mult.get("daily_vaccines_df")
    extra = {k: v for k, v in mult.items() if k != "daily_vaccines_df"}
    return base, (extra or None)


def _extract_detailed(metapop, comps, tvs=None):
    # Same data as extract_history_full, transposed from {comp: {subpop: arr}}
    # to {subpop: {comp: arr}} to match the notebook's existing result shape.
    by_comp = extract_history_full(metapop, comps, tvs=tvs)
    out: dict[str, dict[str, np.ndarray]] = {name: {} for name in metapop.subpop_models}
    for comp, per_subpop in by_comp.items():
        for subpop_name, arr in per_subpop.items():
            out[subpop_name][comp] = arr
    return out


def _analysis_setup(
    *,
    config_dict,
    compartments,
    is_metapop,
    metapop_folder,
    metapop_travel_config,
    ci_unscaled,
    ci_best,
    has_mt,
    schedule_dfs_best,
    loaded_schedule_dfs,
    psets,
    num_days,
    ts_per_day,
    seed_base,
    start_date,
    tvs,
    stoch_run,
    num_age_groups,
    num_risk_groups,
    mobility_value,
    daily_vaccines_value,
    analysis_fitted_num_days,
    analysis_fitted_tv_spacing,
    keep_full_detail,
    schedule_overrides_by_scenario=None,
    schedule_overrides_per_subpop_by_scenario=None,
) -> SimpleNamespace:
    """Rebuild, from plain data, the per-run helpers the Analysis cell used to
    nest as closures. Called once per worker process (via the Pool
    initializer) or once in-process for the serial path."""

    A, R = num_age_groups, num_risk_groups
    ci_cache: dict[int, Any] = {}
    sched_cache: dict[int, Any] = {}

    def scaled_ci(pset):
        seed_scales = {
            k[len("seed_scale_"):]: float(v)
            for k, v in pset.items()
            if k.startswith("seed_scale_") and k[len("seed_scale_"):] in compartments
        }
        if not seed_scales:
            return ci_unscaled
        return _scale_compartment_init(ci_unscaled, seed_scales, compartments, A, R)

    def pset_increments(pset):
        return [
            v for _, v in sorted(
                (
                    (int(pk[len("m_dlog_"):]), float(pv))
                    for pk, pv in pset.items()
                    if pk.startswith("m_dlog_") and pk[len("m_dlog_"):].isdigit()
                ),
                key=lambda t: t[0],
            )
        ]

    def schedule_dfs_for(increments):
        fit_days = int(analysis_fitted_num_days)
        knots = _tv_knot_days(fit_days, int(analysis_fitted_tv_spacing))
        m_df = _build_transmission_multiplier_df(increments, knots, start_date, num_days)
        return SimpleNamespace(
            **{
                f: getattr(loaded_schedule_dfs, f, None)
                for f in ("absolute_humidity_df", "school_work_calendar_df",
                          "mobility_df", "daily_vaccines_df")
            },
            transmission_multiplier_df=m_df,
            # Carry extra scheduled_exact schedules through too -- omitting
            # this silently zeroed out any such schedule whenever fitted m(t)
            # was in use (this rebuild only listed the four base attributes).
            extra_scheduled_dfs=getattr(loaded_schedule_dfs, "extra_scheduled_dfs", None),
        )

    def get_ci(pset_idx):
        if pset_idx is None or is_metapop:
            return ci_best
        if pset_idx not in ci_cache:
            ci_cache[pset_idx] = scaled_ci(psets[pset_idx])
        return ci_cache[pset_idx]

    def get_schedule_dfs(pset_idx):
        if pset_idx is None or not has_mt:
            return schedule_dfs_best
        if pset_idx not in sched_cache:
            incrs = pset_increments(psets[pset_idx])
            sched_cache[pset_idx] = schedule_dfs_for(incrs) if incrs else schedule_dfs_best
        return sched_cache[pset_idx]

    _sched_overrides_by_scen = schedule_overrides_by_scenario or {}
    _sched_overrides_per_sp_by_scen = schedule_overrides_per_subpop_by_scenario or {}
    _base_attrs = ("absolute_humidity_df", "school_work_calendar_df",
                   "mobility_df", "daily_vaccines_df")

    def apply_schedule_overrides(sched_dfs, scenario_name):
        # Merges a scenario's uploaded schedule-replacement CSVs (see the
        # Analysis tab's "Replace schedules from uploaded CSVs" control) into
        # `sched_dfs`, a schedule_dfs namespace as produced by
        # build_notebook_schedules_input / schedule_dfs_for. The upload
        # REPLACES the base schedule; dose_mult scaling (applied separately,
        # after this) still lands on top of the replacement. Extra
        # scheduled_exact schedules live in the `extra_scheduled_dfs` dict
        # rather than as flat attributes (see make_single_pop_metapop's
        # precedence between the two shapes), so overrides for those must be
        # merged into that dict, not just set as a flat attribute.
        overrides = _sched_overrides_by_scen.get(scenario_name)
        if not overrides:
            return sched_dfs
        kwargs = {f: getattr(sched_dfs, f, None) for f in _base_attrs}
        kwargs["transmission_multiplier_df"] = getattr(sched_dfs, "transmission_multiplier_df", None)
        extra = dict(getattr(sched_dfs, "extra_scheduled_dfs", None) or {})
        for attr, df in overrides.items():
            if attr in _base_attrs:
                kwargs[attr] = df
            else:
                extra[attr] = df
        kwargs["extra_scheduled_dfs"] = extra or None
        return SimpleNamespace(**kwargs)

    def schedule_overrides_per_subpop(scenario_name):
        # [{attr: df} | None, ...] indexed by subpop order, for
        # make_metapop_from_folder's schedule_df_overrides_per_subpop kwarg.
        return _sched_overrides_per_sp_by_scen.get(scenario_name)

    def apply_pset(overrides, pset_idx, designed, ratios=None):
        if pset_idx is None:
            return overrides
        ov = dict(overrides or {})
        ratios = ratios or {}
        for pk, pv in psets[pset_idx].items():
            if pk == "phi" or pk.startswith("m_dlog_") or pk.startswith("seed_scale_"):
                continue
            if pk in designed:
                ratio = ratios.get(pk)
                if ratio is None:
                    continue
                if isinstance(ratio, list):
                    ov[pk] = (np.asarray(ratio, dtype=float) * np.asarray(pv, dtype=float)).tolist()
                else:
                    ov[pk] = float(ratio) * float(pv)
                continue
            ov[pk] = list(pv) if isinstance(pv, (list, tuple)) else float(pv)
        return ov

    def run_one(task):
        (scenario_name, overrides, per_subpop, designed, dose_mult,
         dose_mult_per_subpop, ratios, pset_idx, run_i, _rep) = task
        run_ov = apply_pset(overrides, pset_idx, designed, ratios)
        run_sched = get_schedule_dfs(pset_idx)
        base_dose_mult, extra_dose_mult = _split_dose_mult(dose_mult)
        if not is_metapop:
            run_sched = apply_schedule_overrides(run_sched, scenario_name)
            m, _, _ = make_single_pop_metapop(
                config_dict, start_date, num_days, get_ci(pset_idx),
                seed_offset=run_i, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=stoch_run, tvs=tvs, save_daily=True,
                param_overrides=run_ov or None,
                travel_config=metapop_travel_config,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
                schedule_dfs=run_sched,
                dose_mult=base_dose_mult,
                extra_dose_mult=extra_dose_mult,
            )
        else:
            base_dm_per_sp = extra_dm_per_sp = None
            if dose_mult_per_subpop:
                split = [_split_dose_mult(dm) for dm in dose_mult_per_subpop]
                base_dm_per_sp = [b for b, _ in split]
                extra_dm_per_sp = [e for _, e in split]
            m, _ = make_metapop_from_folder(
                metapop_folder, config_dict, start_date, num_days, list(compartments),
                seed_offset=run_i, seed_base=seed_base, ts_per_day=ts_per_day,
                stochastic=stoch_run, tvs=tvs, save_daily=True,
                param_overrides=run_ov or None,
                param_overrides_per_subpop=per_subpop,
                travel_config=metapop_travel_config,
                num_age_groups=A, num_risk_groups=R,
                mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
                transmission_multiplier_df=getattr(run_sched, "transmission_multiplier_df", None),
                dose_mult=base_dose_mult,
                dose_mult_per_subpop=base_dm_per_sp,
                extra_dose_mult=extra_dose_mult,
                extra_dose_mult_per_subpop=extra_dm_per_sp,
                schedule_df_overrides=_sched_overrides_by_scen.get(scenario_name),
                schedule_df_overrides_per_subpop=schedule_overrides_per_subpop(scenario_name),
            )
        m.simulate_until_day(num_days)
        if keep_full_detail:
            return _extract_detailed(m, list(compartments), tvs=tvs)
        # Population-total (summed over subpop/age/risk) daily series only --
        # avoids holding a (day, age, risk) array per subpop per replicate in
        # memory, which is what makes large replicate counts (hundreds to
        # thousands, now reachable quickly thanks to parallelization) blow up
        # memory downstream (autosave, plots, tables all re-derive stats from
        # it). run_analysis_scenarios reduces these across replicates into a
        # median/95% interval per scenario/key before returning.
        return extract_history(m, list(compartments), tvs=tvs)

    return SimpleNamespace(run_one=run_one)


# ---------------------------------------------------------------------------
# multiprocessing worker
# ---------------------------------------------------------------------------
# Same trick as generic_core.fitting's _init_bayes_worker/_bayes_log_prob_worker:
# each worker process rebuilds its own run context once (from the picklable
# `payload`) via the Pool initializer, rather than pickling a closure.
_ANALYSIS_WORKER_CTX = None


def _init_analysis_worker(payload):
    global _ANALYSIS_WORKER_CTX
    _ANALYSIS_WORKER_CTX = _analysis_setup(**payload)


def _analysis_worker_task(task):
    assert _ANALYSIS_WORKER_CTX is not None
    return _ANALYSIS_WORKER_CTX.run_one(task)


def _analysis_worker_count(n_tasks: int, cap: int | None = None) -> int:
    n = max(1, (os.cpu_count() or 2) - 1)
    if cap is not None:
        n = min(n, cap)
    return max(1, min(n, n_tasks))


def run_analysis_scenarios(
    analysis_scenarios,
    run_schedule,
    *,
    config_dict,
    compartments,
    is_metapop,
    metapop_folder,
    metapop_travel_config,
    ci_unscaled,
    ci_best,
    has_mt,
    schedule_dfs_best,
    loaded_schedule_dfs,
    psets,
    num_days,
    ts_per_day,
    seed_base,
    start_date,
    tvs,
    stoch_run,
    num_age_groups,
    num_risk_groups,
    mobility_value,
    daily_vaccines_value,
    analysis_fitted_num_days,
    analysis_fitted_tv_spacing,
    keep_full_detail: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run every (scenario, run_schedule) combination. Uses a spawn-context
    process pool when there's more than one task (see module docstring for
    why plain ``ProcessPoolExecutor``/closures can't be used directly from a
    marimo cell); falls back to running serially in-process otherwise, or if
    the pool fails to start.

    ``keep_full_detail=True`` returns ``{scenario_name: [replicate_result,
    ...]}`` with the full per-subpop/(day, age, risk) detail per replicate,
    same as the notebook's original inline loop -- fine for drill-down/export
    with a modest replicate count, but holding that for every replicate of a
    large ensemble is what drove a real memory blowup on a ~3000-replicate
    run. The default, ``keep_full_detail=False``, has each worker return only
    the population-total (summed over subpop/age/risk) daily series (see
    ``generic_core.model_factory.extract_history``), then reduces those
    across replicates into a per-scenario/key ``{"median", "p2_5", "p97_5",
    "n_reps"}`` summary here -- independent of replicate count, and safe at
    any scale. Returns ``{scenario_name: {key: {"median": array, "p2_5":
    array, "p97_5": array, "n_reps": int}}}`` in that mode.
    """
    tasks: list[tuple] = []
    task_scenarios: list[str] = []
    # Uploaded schedule-replacement CSVs are per-scenario but not per-task
    # (every replicate of a scenario uses the same override), so they ship
    # once via `payload` -- keyed by scenario name -- rather than being
    # repeated (and re-pickled) in every task tuple. See
    # _analysis_setup/apply_schedule_overrides.
    schedule_overrides_by_scenario: dict[str, dict] = {}
    schedule_overrides_per_subpop_by_scenario: dict[str, list] = {}
    for scen_tuple in analysis_scenarios:
        scen_name, overrides = scen_tuple[0], scen_tuple[1]
        per_subpop = scen_tuple[2] if len(scen_tuple) > 2 else None
        designed = scen_tuple[3] if len(scen_tuple) > 3 else set()
        dose_mult = scen_tuple[4] if len(scen_tuple) > 4 else None
        dose_mult_per_subpop = scen_tuple[5] if len(scen_tuple) > 5 else None
        ratios = scen_tuple[6] if len(scen_tuple) > 6 else {}
        sched_ov = scen_tuple[7] if len(scen_tuple) > 7 else None
        sched_ov_per_subpop = scen_tuple[8] if len(scen_tuple) > 8 else None
        if sched_ov:
            schedule_overrides_by_scenario[scen_name] = sched_ov
        if sched_ov_per_subpop:
            schedule_overrides_per_subpop_by_scenario[scen_name] = sched_ov_per_subpop
        for run_i, (pset_idx, rep) in enumerate(run_schedule):
            tasks.append((
                scen_name, overrides, per_subpop, designed, dose_mult,
                dose_mult_per_subpop, ratios, pset_idx, run_i, rep,
            ))
            task_scenarios.append(scen_name)

    payload = dict(
        config_dict=config_dict, compartments=compartments, is_metapop=is_metapop,
        metapop_folder=metapop_folder, metapop_travel_config=metapop_travel_config,
        ci_unscaled=ci_unscaled, ci_best=ci_best, has_mt=has_mt, schedule_dfs_best=schedule_dfs_best,
        loaded_schedule_dfs=loaded_schedule_dfs, psets=psets, num_days=num_days,
        ts_per_day=ts_per_day, seed_base=seed_base, start_date=start_date, tvs=tvs,
        stoch_run=stoch_run, num_age_groups=num_age_groups, num_risk_groups=num_risk_groups,
        mobility_value=mobility_value, daily_vaccines_value=daily_vaccines_value,
        analysis_fitted_num_days=analysis_fitted_num_days,
        analysis_fitted_tv_spacing=analysis_fitted_tv_spacing,
        keep_full_detail=keep_full_detail,
        schedule_overrides_by_scenario=schedule_overrides_by_scenario,
        schedule_overrides_per_subpop_by_scenario=schedule_overrides_per_subpop_by_scenario,
    )

    n_workers = _analysis_worker_count(len(tasks))
    results: list[dict] | None = None
    if n_workers > 1:
        try:
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=n_workers, initializer=_init_analysis_worker, initargs=(payload,))
        except Exception as exc:  # pragma: no cover - falls back to single process
            print(f"[analysis] could not start process pool ({exc}); running single-process", flush=True)
            pool = None
        if pool is not None:
            print(f"[analysis] running {len(tasks)} task(s) across {n_workers} processes", flush=True)
            results = []
            try:
                with pool:
                    for i, result in enumerate(pool.imap(_analysis_worker_task, tasks)):
                        results.append(result)
                        if progress_callback:
                            progress_callback({
                                "done": i + 1, "total": len(tasks),
                                "scenario": task_scenarios[i], "rep": tasks[i][-1],
                            })
            except Exception as exc:  # pragma: no cover - falls back to single process
                print(f"[analysis] process pool failed ({exc}); running single-process", flush=True)
                results = None

    if results is None:
        print(f"[analysis] running {len(tasks)} task(s) single-process", flush=True)
        run_ctx = _analysis_setup(**payload)
        results = []
        for i, task in enumerate(tasks):
            results.append(run_ctx.run_one(task))
            if progress_callback:
                progress_callback({
                    "done": i + 1, "total": len(tasks),
                    "scenario": task_scenarios[i], "rep": task[-1],
                })

    all_results: dict[str, list[dict]] = {}
    for scen_name, result in zip(task_scenarios, results):
        all_results.setdefault(scen_name, []).append(result)

    if keep_full_detail:
        return all_results

    summary: dict[str, dict] = {}
    for scen_name, reps in all_results.items():
        scen_summary: dict[str, dict] = {}
        keys = {k for rep in reps for k in rep}
        for key in keys:
            arrs = [rep[key] for rep in reps if key in rep]
            if not arrs:
                continue
            stacked = np.stack(arrs, axis=0)
            scen_summary[key] = {
                "median": np.median(stacked, axis=0),
                "p2_5": np.percentile(stacked, 2.5, axis=0),
                "p97_5": np.percentile(stacked, 97.5, axis=0),
                "n_reps": len(arrs),
            }
        summary[scen_name] = scen_summary
    return summary
