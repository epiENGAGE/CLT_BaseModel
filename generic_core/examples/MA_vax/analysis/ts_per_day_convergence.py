"""
ts_per_day_convergence.py — Numerical convergence check for generic-core's
substep resolution (ts_per_day) on the MA_vax model config.

clt_toolkit converts every per-day `rate` into a per-substep transition
probability via the exact hazard formula `1 - exp(-rate * dt)`
(dt = 1/ts_per_day), in both deterministic and stochastic modes. That's only
close to the naive `rate * dt` when `rate * dt` is small; at ts_per_day=7 the
fastest rates in this model (E_to_I_rate = EV_to_IV_rate = 0.5/day) still
have rate*dt ~= 0.07 per substep, a bias that compounds over ~250 days of
simulation through the nonlinear S-I feedback.

This script runs the SAME forward simulation (fixed, realistic parameters —
no fitting involved) at a ladder of ts_per_day values and compares each to
the finest resolution run, to find the smallest ts_per_day beyond which the
trajectory stops changing materially. That's the resolution to actually fit
at, rather than guessing — and since MCMC cost scales ~linearly with
ts_per_day, this tells you the cheapest resolution that's still trustworthy.

Run from the repo root:
    uv run python generic_core/examples/MA_vax/analysis/ts_per_day_convergence.py
    uv run python generic_core/examples/MA_vax/analysis/ts_per_day_convergence.py --ts-per-day 7 14 28 56
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from generic_core.model_factory import build_compartment_init, make_single_pop_metapop
from generic_core.fitting import (
    _inject_tv_transmission,
    _read_daily_history,
    build_transmission_multiplier_array,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# The model config and input CSVs live one level up, in the MA_vax example
# folder this analysis belongs to.
PARENT = os.path.dirname(HERE)


def _load_config() -> dict:
    with open(os.path.join(PARENT, "model_config.json")) as f:
        return json.load(f)


def _load_schedule_csvs():
    ah = pd.read_csv(os.path.join(PARENT, "data", "schedules", "ma_absolute_humidity.csv"))
    cal = pd.read_csv(os.path.join(PARENT, "data", "schedules", "MA_school_work_calendar.csv"))
    vax = pd.read_csv(os.path.join(PARENT, "data", "vaccination", "MA_flu_daily_vaccinations_proportions_array.csv"))
    return ah, cal, vax


def _load_params(fitted_params_path: str):
    """Pull a realistic (not necessarily "correct") parameter set + m(t) curve
    out of a previous fitting result JSON's best_params, just to stress the
    integrator with plausible dynamics — magnitude/timing of the epidemic,
    not which fit produced it, is what matters for a resolution check."""
    with open(fitted_params_path) as f:
        d = json.load(f)
    bp = d["best_params"]
    age_labels = [f"a{i}" for i in range(7)]
    ihr = np.array([bp[f"IHR_scale|{lbl}"] for lbl in age_labels])
    incr_keys = sorted(
        [k for k in bp if k.startswith("m_dlog_")], key=lambda x: int(x.split("_")[-1])
    )
    increments = np.array([bp[k] for k in incr_keys])
    return {
        "beta_baseline": float(bp["beta_baseline"]),
        "humidity_impact": float(bp["humidity_impact"]),
        "seed_scale_E": float(bp["seed_scale_E"]),
        "ihr_scale": ihr,
        "m_increments": increments,
        "tv_knot_spacing_days": int(d["fit_config"].get("tv_knot_spacing_days", 30)),
        "num_days": int(d["num_days"]),
    }


def simulate(cfg, params, num_days, ts_per_day, schedule_csvs) -> np.ndarray:
    """Returns new_H_by_age, shape (num_days, 7)."""
    A, R = 7, 1
    start_date = "2025-09-01"
    ah_df, cal_df, vax_df = schedule_csvs

    base_params = cfg["params"]
    population = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], dtype=float)
    seed_E = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["seeds"]["E"], dtype=float)
    seed_E_scaled = seed_E * params["seed_scale_E"]
    compartment_init, overflow = build_compartment_init(
        {"E": seed_E_scaled}, population, cfg["compartments"]
    )
    assert not overflow, "seed count exceeds population in some age group"

    ihr_scale = params["ihr_scale"].reshape(A, R)
    base_I_to_H = np.asarray(base_params["I_to_H_prop"], dtype=float).reshape(A, R)
    base_IV_to_H = np.asarray(base_params["IV_to_H_prop"], dtype=float).reshape(A, R)
    param_overrides = {
        "beta_baseline": params["beta_baseline"],
        "humidity_impact": params["humidity_impact"],
        "I_to_H_prop": (base_I_to_H * ihr_scale).tolist(),
        "IV_to_H_prop": (base_IV_to_H * ihr_scale).tolist(),
    }

    cfg_run, n_foi = _inject_tv_transmission(cfg)
    assert n_foi > 0, "no force_of_infection transition found to attach m(t) to"

    spacing = params["tv_knot_spacing_days"]
    knots = list(range(0, num_days, spacing))
    if not knots or knots[-1] != num_days - 1:
        knots.append(num_days - 1)
    n_incr = len(knots) - 1
    incr = params["m_increments"][:n_incr]
    m = build_transmission_multiplier_array(incr, knots, num_days)

    dates = pd.date_range(start=start_date, periods=num_days, freq="D").date
    tvm_df = pd.DataFrame({"date": dates, "transmission_multiplier": m})

    schedule_dfs = SimpleNamespace(
        absolute_humidity_df=ah_df, school_work_calendar_df=cal_df,
        mobility_df=None, daily_vaccines_df=vax_df, transmission_multiplier_df=tvm_df,
    )

    metapop, _, _ = make_single_pop_metapop(
        cfg_run, start_date, num_days, compartment_init,
        seed_offset=0, seed_base=0, ts_per_day=ts_per_day, stochastic=False,
        tvs=["I_to_H", "IV_to_H"], save_daily=False,
        param_overrides=param_overrides,
        num_age_groups=A, num_risk_groups=R,
        mobility_value=1.0, daily_vaccines_value=0.0,
        schedule_dfs=schedule_dfs,
    )
    metapop.simulate_until_day(num_days)
    sub = next(iter(metapop.subpop_models.values()))
    i_to_h = _read_daily_history(sub, "I_to_H", num_days)
    iv_to_h = _read_daily_history(sub, "IV_to_H", num_days)
    return (i_to_h + iv_to_h).sum(axis=2)  # (num_days, A)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--params-from",
        default=os.path.join(HERE, "MA_fitted_params_one_ts_per_day.json"),
    )
    ap.add_argument("--num-days", type=int, default=None, help="default: match the source fit's num_days")
    ap.add_argument("--ts-per-day", type=int, nargs="+", default=[7, 14, 28, 56, 112])
    ap.add_argument("--tol-pct", type=float, default=1.0, help="convergence threshold, %% vs finest run")
    args = ap.parse_args()

    cfg = _load_config()
    params = _load_params(args.params_from)
    schedule_csvs = _load_schedule_csvs()
    num_days = args.num_days or params["num_days"]

    print(f"params: beta_baseline={params['beta_baseline']:.4f}  humidity_impact={params['humidity_impact']:.4f}  "
          f"seed_scale_E={params['seed_scale_E']:.3f}")
    print(f"ihr_scale: {np.round(params['ihr_scale'], 3).tolist()}")
    print(f"num_days={num_days}\n")

    results, timings = {}, {}
    for tspd in sorted(set(args.ts_per_day)):
        t0 = time.time()
        traj = simulate(cfg, params, num_days, tspd, schedule_csvs)
        dt = time.time() - t0
        results[tspd] = traj
        timings[tspd] = dt
        total = traj.sum(axis=1)
        print(f"ts_per_day={tspd:4d}  peak={total.max():9.2f} (day {int(total.argmax())})  "
              f"total={total.sum():10.1f}  wall={dt:6.2f}s")

    finest = max(results)
    ref = results[finest].sum(axis=1)
    print(f"\n=== relative to finest resolution (ts_per_day={finest}) ===")
    print(f"{'ts_per_day':>10}  {'peak %diff':>11}  {'total %diff':>12}  {'max day %diff':>14}  {'cost vs ts=7':>13}")
    base_cost = timings.get(7, min(timings.values()))
    converged_at = None
    for tspd in sorted(results):
        if tspd == finest:
            continue
        total = results[tspd].sum(axis=1)
        peak_pct = 100 * (total.max() - ref.max()) / ref.max()
        total_pct = 100 * (total.sum() - ref.sum()) / ref.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            day_pct = np.where(ref > 1.0, 100 * np.abs(total - ref) / ref, 0.0)
        cost_mult = timings[tspd] / base_cost
        print(f"{tspd:10d}  {peak_pct:10.2f}%  {total_pct:11.2f}%  {day_pct.max():13.2f}%  {cost_mult:12.2f}x")
        if converged_at is None and max(abs(peak_pct), abs(total_pct), day_pct.max()) <= args.tol_pct:
            converged_at = tspd

    print()
    if converged_at is not None:
        print(f"Smallest tested ts_per_day within {args.tol_pct}% of the finest run: {converged_at}")
    else:
        print(f"No tested ts_per_day (below {finest}) came within {args.tol_pct}% of the finest run — "
              f"try extending --ts-per-day higher, or relax --tol-pct.")

    out_csv = os.path.join(HERE, "ts_per_day_convergence.csv")
    df = pd.DataFrame({"day": np.arange(num_days)})
    for tspd, traj in results.items():
        df[f"total_ts{tspd}"] = traj.sum(axis=1)
    df.to_csv(out_csv, index=False)
    print(f"[csv] wrote day-by-day totals per ts_per_day to {out_csv}")


if __name__ == "__main__":
    main()
