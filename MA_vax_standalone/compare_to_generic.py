"""
compare_to_generic.py — Apples-to-apples check: does the generic-core engine
reproduce the same new_H trajectory as the plain-numpy MA_vax model (model.py)
for the *same* parameters?

Loads a fitted parameter set from a MA_vax_standalone/fit_bayesian.py output folder
(overrides + beta_multiplier_arr), simulates it two ways:

  1. model.py's simulate_new_H — the pure-numpy reference.
  2. generic_core's ConfigDrivenMetapopModel, built from
     generic_core/examples/massachusetts_vax/model_config_MA.json, with the
     same param overrides / seeds / m(t) curve, at ts_per_day=1.

and reports per-day / per-age-group discrepancies. This isolates genuine
model-mechanics differences from anything caused by MCMC posterior
degeneracy or fitting-config mismatches (bounds, likelihood, etc.) — those
were already ruled out/addressed separately.

Run from the repo root:
    uv run python MA_vax_standalone/compare_to_generic.py
    uv run python MA_vax_standalone/compare_to_generic.py --output-folder MA_vax_standalone/outputs_2026-07-30_age_ihr_scale
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from MA_vax_standalone import model
from MA_vax_standalone.fit_bayesian import load_fitted_run

from generic_core.model_factory import build_compartment_init, make_single_pop_metapop
from generic_core.fitting import _inject_tv_transmission, _read_daily_history

HERE = os.path.dirname(os.path.abspath(__file__))
GENERIC_DATA_FOLDER = os.path.join(
    os.path.dirname(HERE), "generic_core", "examples", "massachusetts_vax"
)


def _load_generic_config() -> dict:
    import json
    with open(os.path.join(GENERIC_DATA_FOLDER, "model_config_MA.json")) as f:
        return json.load(f)


def _load_schedule_csvs():
    """Raw (string-date) DataFrames for the schedules the generic model needs,
    read from the same source CSVs model.py's load_inputs uses."""
    ah = pd.read_csv(os.path.join(GENERIC_DATA_FOLDER, "data", "schedules", "ma_absolute_humidity.csv"))
    cal = pd.read_csv(os.path.join(GENERIC_DATA_FOLDER, "data", "schedules", "MA_school_work_calendar.csv"))
    vax = pd.read_csv(os.path.join(GENERIC_DATA_FOLDER, "data", "vaccination", "MA_flu_daily_vaccinations_proportions_array.csv"))
    return ah, cal, vax


def run_reference(overrides: dict, beta_mult: np.ndarray) -> np.ndarray:
    """model.py's new_H_by_age for `overrides` + explicit m(t) curve."""
    params = model.default_params()
    params["num_days"] = len(beta_mult)
    inputs = model.load_inputs(model.DATA_FOLDER, params)
    return model.simulate_new_H(overrides, inputs, beta_multiplier_arr=beta_mult, by_age=True)


def run_generic(overrides: dict, beta_mult: np.ndarray, num_days: int, ts_per_day: int = 1) -> np.ndarray:
    """generic_core's new_H_by_age for the same overrides + m(t) curve."""
    cfg = _load_generic_config()
    A, R = 7, 1
    start_date = "2025-09-01"

    base_params = cfg["params"]
    population = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], dtype=float)
    seed_E = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["seeds"]["E"], dtype=float)
    seed_E_scaled = seed_E * float(overrides.get("E0_scale", 1.0))
    compartment_init, overflow = build_compartment_init(
        {"E": seed_E_scaled}, population, cfg["compartments"]
    )
    assert not overflow, "seed count exceeds population in some age group"

    ihr_scale = np.asarray(overrides["ihr_scale"], dtype=float).reshape(A, R)
    base_I_to_H = np.asarray(base_params["I_to_H_prop"], dtype=float).reshape(A, R)
    base_IV_to_H = np.asarray(base_params["IV_to_H_prop"], dtype=float).reshape(A, R)
    param_overrides = {
        "beta_baseline": float(overrides["beta_baseline"]),
        "humidity_impact": float(overrides["humidity_impact"]),
        "I_to_H_prop": (base_I_to_H * ihr_scale).tolist(),
        "IV_to_H_prop": (base_IV_to_H * ihr_scale).tolist(),
    }

    cfg_run, n_foi = _inject_tv_transmission(cfg)
    assert n_foi > 0, "no force_of_infection transition found to attach m(t) to"

    dates = pd.date_range(start=start_date, periods=num_days, freq="D").date
    tvm_df = pd.DataFrame({"date": dates, "transmission_multiplier": beta_mult[:num_days]})

    ah_df, cal_df, vax_df = _load_schedule_csvs()
    from types import SimpleNamespace
    schedule_dfs = SimpleNamespace(
        absolute_humidity_df=ah_df,
        school_work_calendar_df=cal_df,
        mobility_df=None,
        daily_vaccines_df=vax_df,
        transmission_multiplier_df=tvm_df,
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

    i_to_h = _read_daily_history(sub, "I_to_H", num_days)   # (num_days, A, R)
    iv_to_h = _read_daily_history(sub, "IV_to_H", num_days)
    new_H_by_age = (i_to_h + iv_to_h).sum(axis=2)  # sum over risk -> (num_days, A)
    return new_H_by_age


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-folder", default=os.path.join(HERE, "outputs_2026-07-30_age_ihr_scale"))
    ap.add_argument("--method", default="emcee")
    ap.add_argument("--point", default="best")
    ap.add_argument("--ts-per-day", type=int, default=1)
    args = ap.parse_args()

    overrides, beta_mult = load_fitted_run(args.output_folder, args.method, point=args.point)
    if beta_mult is None:
        raise RuntimeError("no fit_beta_multiplier_*.csv found — this comparison needs a tvbeta run")
    num_days = len(beta_mult)
    print(f"Comparing over {num_days} days, ts_per_day={args.ts_per_day}")
    print(f"overrides: { {k: (v if np.isscalar(v) else list(np.round(v, 4))) for k, v in overrides.items()} }")

    ref = run_reference(overrides, beta_mult)          # (num_days, 7)
    gen = run_generic(overrides, beta_mult, num_days, ts_per_day=args.ts_per_day)  # (num_days, 7)

    assert ref.shape == gen.shape, f"shape mismatch: ref={ref.shape} generic={gen.shape}"

    diff = gen - ref
    ref_total = ref.sum(axis=1)
    gen_total = gen.sum(axis=1)

    print("\n=== Total new_H (summed over age groups) ===")
    print(f"{'day':>4}  {'ref':>10}  {'generic':>10}  {'diff':>10}  {'%diff':>8}")
    step = max(1, num_days // 25)
    for t in range(0, num_days, step):
        pct = 100 * (gen_total[t] - ref_total[t]) / ref_total[t] if ref_total[t] > 1e-9 else float("nan")
        print(f"{t:4d}  {ref_total[t]:10.3f}  {gen_total[t]:10.3f}  {gen_total[t]-ref_total[t]:10.3f}  {pct:7.1f}%")

    print(f"\npeak ref:     {ref_total.max():.2f} on day {int(ref_total.argmax())}")
    print(f"peak generic: {gen_total.max():.2f} on day {int(gen_total.argmax())}")
    print(f"total ref:     {ref_total.sum():.1f}")
    print(f"total generic: {gen_total.sum():.1f}   ({100*(gen_total.sum()-ref_total.sum())/ref_total.sum():+.1f}%)")

    print("\n=== Per-age-group totals (summed over all days) ===")
    for a, label in enumerate(model.AGE_GROUP_LABELS):
        r, g = ref[:, a].sum(), gen[:, a].sum()
        pct = 100 * (g - r) / r if r > 1e-9 else float("nan")
        print(f"  {label:>6}: ref={r:9.2f}  generic={g:9.2f}  diff={g-r:+9.2f}  ({pct:+6.1f}%)")

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs_day = int(np.argmax(np.abs(diff.sum(axis=1))))
    print(f"\nRMSE (per age-day cell): {rmse:.4f}")
    print(f"day with largest total-abs discrepancy: {max_abs_day} "
          f"(ref={ref_total[max_abs_day]:.2f}, generic={gen_total[max_abs_day]:.2f})")

    out_csv = os.path.join(HERE, "compare_to_generic_trajectories.csv")
    df = pd.DataFrame({
        "day": np.arange(num_days),
        "ref_total": ref_total,
        "generic_total": gen_total,
        "diff_total": gen_total - ref_total,
    })
    for a, label in enumerate(model.AGE_GROUP_LABELS):
        df[f"ref_{label}"] = ref[:, a]
        df[f"generic_{label}"] = gen[:, a]
    df.to_csv(out_csv, index=False)
    print(f"\n[csv] wrote day-by-day / age-by-age trajectories to {out_csv}")


if __name__ == "__main__":
    main()
