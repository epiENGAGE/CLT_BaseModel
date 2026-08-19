"""Counterfactual vaccination-impact pipeline for MA_vax_standalone.

Replicates the table structure of PNAS 2505175122 Supplementary Tables
S.A.1-S.A.6 (infection- vs severity-protection decomposition, age-specific
vaccination impact, 70%-coverage targets, and VE sensitivity), adapted to
this model's 7 age groups (`model.AGE_GROUP_LABELS`), the Massachusetts
population, and the 2025-2026 season.

Built as a pipeline: scenario dict -> paired stochastic simulation -> averted
-burden table, so the scenario definitions below (age groups, VE levels,
coverage targets) can be edited independently of the table-computation
logic. Everything downstream of `scenario_totals`/`averted_summary` only
cares that a scenario dict is accepted by `model.apply_scenario`.

Caveats to flag:

VE_SCENARIOS are placeholder multipliers (1.3/1.0/0.7 on vax_susceptibility and 
IV_to_H_prop) — Table S.A.4's output shows a naive uniform scale can push 
IV_to_H_prop above I_to_H_prop for some age groups under low_ve (negative severity 
VE), which is a modeling artifact worth fixing once you decide on real VE bounds.

The 70%-coverage bisection and "single age group only" scenarios reuse the existing 
stochastic-simulation machinery from model.py (unchanged) — no new 
correctness assumptions there.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from MA_vax_standalone import model

AGE_GROUPS = model.AGE_GROUP_LABELS


# ── Scenario builders ────────────────────────────────────────────────────────

def no_vaccine_scenario() -> dict:
    return {"vax_multiplier": [0.0] * len(AGE_GROUPS)}


def baseline_scenario() -> dict:
    return {}


def single_age_only_scenario(age_idx: int) -> dict:
    """Vaccinate only `age_idx` at the baseline schedule; every other age
    group unvaccinated."""
    mult = [0.0] * len(AGE_GROUPS)
    mult[age_idx] = 1.0
    return {"vax_multiplier": mult}


def infection_protection_only_scenario(inputs: dict) -> dict:
    """VE against infection only: `vax_susceptibility` unchanged, but
    `IV_to_H_prop` is set equal to `I_to_H_prop` so vaccination confers no
    reduction in hospitalization risk conditional on infection."""
    return {"IV_to_H_prop": np.array(inputs["params"]["I_to_H_prop"], dtype=float)}


def ve_scale_scenario(
    vax_susceptibility_scale: float | Sequence[float],
    IV_to_H_prop_scale: float | Sequence[float],
) -> dict:
    """Multipliers on `vax_susceptibility` / `IV_to_H_prop`. Each may be a
    scalar (applied to every age group) or a length-`len(AGE_GROUPS)` sequence
    of per-age-group multipliers, ordered per `model.AGE_GROUP_LABELS`.
    `model.apply_scenario` validates the length of any array-valued scale."""
    return {
        "vax_susceptibility_scale": vax_susceptibility_scale,
        "IV_to_H_prop_scale": IV_to_H_prop_scale,
    }


# VE sensitivity presets, analogous to Table S.A.4's Low/Baseline/High
# scenarios: multipliers on the fitted/default vax_susceptibility and
# IV_to_H_prop (below 1 = more protective than baseline, above 1 = less).
# Each scale may also be a per-age-group sequence (length len(AGE_GROUPS))
# instead of a scalar. Placeholder values -- swap these for whatever VE
# bounds are appropriate.
VE_SCENARIOS = {
    # "low_ve":      ve_scale_scenario(vax_susceptibility_scale=1.3, IV_to_H_prop_scale=1.3),
    # "baseline_ve": ve_scale_scenario(vax_susceptibility_scale=1.0, IV_to_H_prop_scale=1.0),
    # "high_ve":     ve_scale_scenario(vax_susceptibility_scale=0.7, IV_to_H_prop_scale=0.7),
    "low_ve": ve_scale_scenario(
        vax_susceptibility_scale=[1.1754,1.1754,1.1754,1.1754,1.1899,1.1899,1],
        IV_to_H_prop_scale=[1.0962,1.0962,1.0962,1.0962,1.0535,1.0535,1.1449],
    ),
    "baseline_ve": ve_scale_scenario(
            vax_susceptibility_scale=[1.0]*7,
            IV_to_H_prop_scale=[1.0]*7,
        ),
    "high_ve": ve_scale_scenario(
            vax_susceptibility_scale=[0.8596,0.8596,0.8596,0.8596,0.8354,0.8354,0.86],
            IV_to_H_prop_scale=[0.4922,0.4922,0.4922,0.4922,0.9272,0.9272,1.028],
        ),
}


# ── 70% coverage calibration ─────────────────────────────────────────────────

def _cumulative_coverage(inputs: dict, age_idx: int) -> float:
    """Deterministic total S_to_SV / population for one age group over the season."""
    ds = model.simulate_detailed(inputs, stochastic=False, n_reps=1)
    total_vaxxed = float(ds["S_to_SV"].isel(replication=0, age_group=age_idx).sum())
    return total_vaxxed / float(inputs["population"][age_idx])


def coverage_multiplier_for_target(inputs: dict, age_idx: int, target: float,
                                    tol: float = 0.005, max_iter: int = 15,
                                    bisect: bool = False) -> float:
    """
    Find the per-age-group vax-schedule multiplier so cumulative coverage in
    `age_idx` reaches `target` (fraction of population) by season's end,
    holding every other age group at its baseline schedule -- mirrors the
    paper's "scale daily uptake by a constant multiplier to reach 70%
    coverage, preserving the shape of the original series" construction.

    Under the "population" vax pool (see `model.VAX_POOL_DEFAULT`) the daily
    dose count is a fixed fraction of the constant population, so cumulative
    coverage is linear in the multiplier and the cross-product estimate
    `target / baseline_cov` is exact -- we try it first and accept it if it
    lands within `tol`. It only misses when the per-step S-cap binds (target
    infeasible, or near saturation), and under the "susceptible" pool coverage
    is nonlinear throughout; in both cases we fall back to bisection (unless
    `bisect=False`, see below), which is derivative-free and robust to either
    interpretation.

    `bisect=False` skips all calibration refinement and returns the raw
    cross-product guess `target / baseline_cov` after a single simulation --
    exact under "population", only approximate under "susceptible" (won't
    land exactly on `target`). Use this to match generic_core, which has no
    calibration loop of its own and would only ever apply this same naive
    scale-by-ratio multiplier to a "susceptible"-pool schedule.
    """
    baseline_cov = _cumulative_coverage(inputs, age_idx)
    if baseline_cov <= 0:
        raise ValueError(
            f"age group {AGE_GROUPS[age_idx]} has zero baseline vaccination; "
            "cannot scale to a coverage target"
        )

    guess = target / baseline_cov
    if not bisect:
        return guess

    def cov_at(mult: float) -> float:
        mvec = np.ones(len(AGE_GROUPS))
        mvec[age_idx] = mult
        return _cumulative_coverage(model.apply_vax_scenario(inputs, mvec), age_idx)

    # Fast path: closed-form cross product, exact in "population" mode absent
    # S-capping (coverage is linear in the multiplier there). One extra sim to
    # verify; cheap insurance against the S-cap caveat. Skipped entirely under
    # the "susceptible" pool, where coverage is nonlinear and the guess never
    # passes -- so we don't pay the wasted verification sim before bisecting.
    if inputs["params"].get("vax_pool", model.VAX_POOL_DEFAULT) == "population":
        if abs(cov_at(guess) - target) < tol:
            return guess

    # Fallback: bisection. Bracket-double `hi` until it reaches `target` (or the
    # S-cap makes `target` infeasible and we cap out near 1e4).
    lo, hi = 0.0, max(2.0, 1.5 * guess)
    while cov_at(hi) < target and hi < 1e4:
        hi *= 2
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c = cov_at(mid)
        if abs(c - target) < tol:
            return mid
        if c < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def coverage_70pct_scenario(inputs: dict, age_idx: int | None = None, bisect: bool = True) -> dict:
    """
    `vax_multiplier` scenario reaching 70% cumulative coverage.
    `age_idx=None` scales every age group to 70%; an int scales only that
    age group (others stay at their baseline schedule). `bisect` is passed
    through to `coverage_multiplier_for_target` -- see there.
    """
    mult = np.ones(len(AGE_GROUPS))
    indices = range(len(AGE_GROUPS)) if age_idx is None else [age_idx]
    for i in indices:
        mult[i] = coverage_multiplier_for_target(inputs, i, 0.70, bisect=bisect)
    return {"vax_multiplier": mult.tolist()}


# ── Paired stochastic simulation ─────────────────────────────────────────────

def _run_reps(inputs: dict, n_reps: int, seed: int, steps_per_day: int,
              stochastic: bool = True) -> xr.Dataset:
    """
    `n_reps` stochastic replications, each seeded from `seed` and its own
    replication index only (never from which scenario is running). This is
    what pairs two scenarios' replications on common random numbers for
    variance reduction -- mirrors the paper's "N pairs of stochastic
    simulations" -- as long as both scenarios are simulated with the same
    (n_reps, seed, steps_per_day).

    `stochastic=False` ignores `n_reps` and runs a single deterministic
    replication instead (the model is the same discrete-time Euler map
    either way -- see `model._simulate_detailed_core`). Downstream median/CI
    summaries over a single replication just collapse to that point
    estimate, which is the expected behaviour for a deterministic run.
    """
    reps = n_reps if stochastic else 1
    dates = inputs["dates"]
    num_days = len(dates)
    A = len(inputs["population"])
    comp_arr = {c: np.zeros((reps, num_days, A)) for c in model.COMPARTMENTS}
    trans_arr = {t: np.zeros((reps, num_days, A)) for t in model.TRANSITIONS}
    for r in range(reps):
        rng = np.random.default_rng(seed * 1_000_003 + r) if stochastic else None
        comps_r, trans_r = model._simulate_detailed_core(inputs, stochastic, rng, steps_per_day)
        for c in model.COMPARTMENTS:
            comp_arr[c][r] = comps_r[c]
        for name in model.TRANSITIONS:
            trans_arr[name][r] = trans_r[name]
    data_vars = {c: (("replication", "day", "age_group"), comp_arr[c]) for c in model.COMPARTMENTS}
    data_vars.update({t: (("replication", "day", "age_group"), trans_arr[t]) for t in model.TRANSITIONS})
    return xr.Dataset(
        data_vars,
        coords={"replication": np.arange(reps), "day": dates, "age_group": AGE_GROUPS},
    )


def scenario_totals(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                     steps_per_day: int = 1, stochastic: bool = True) -> dict:
    """
    Run a scenario `n_reps` times (paired on `seed`) -- or once,
    deterministically, if `stochastic=False` -- and return:
        {"new_H": (reps, A) array of total hospitalizations per age group,
         "doses": (A,) array of total (deterministic) doses administered,
         "population": (A,) array}
    """
    inputs = model.apply_scenario(base_inputs, scenario)
    ds = _run_reps(inputs, n_reps, seed, steps_per_day, stochastic=stochastic)
    new_H = (ds["I_to_H"] + ds["IV_to_H"]).sum(dim="day").to_numpy()  # (reps, A)
    det_ds = model.simulate_detailed(inputs, stochastic=False, n_reps=1, steps_per_day=steps_per_day)
    doses = det_ds["S_to_SV"].isel(replication=0).sum(dim="day").to_numpy()  # (A,)
    return {
        "new_H": new_H,
        "doses": doses,
        "population": np.asarray(base_inputs["population"], dtype=float),
    }


# ── Averted-burden summary ───────────────────────────────────────────────────

def _summ(values: np.ndarray) -> tuple[float, float, float]:
    """median, 2.5th pct, 97.5th pct across replications."""
    return (
        float(np.median(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def averted_summary(reference: dict, scenario: dict, by_age: bool = True,
                     pct_reference: dict | None = None,
                     doses_reference: dict | None = None,
                     doses_override: np.ndarray | float | None = None) -> pd.DataFrame:
    """
    Hospitalizations averted by `scenario` relative to `reference` (both
    dicts from `scenario_totals`, run with the same n_reps/seed so
    replications are paired). Returns a DataFrame indexed by age group (plus
    an aggregate "All" row) with median [95% CI] strings for percent averted,
    hospitalizations averted per 100K population, and (when doses differ)
    per 100K additional doses -- mirroring Tables S.A.1-S.A.3/S.A.5-S.A.6.

    `pct_reference` overrides the denominator used for `pct_averted` only
    (per100k/per100k_doses always use `reference`/`scenario`'s own
    population/doses). Needed when chaining multiple `averted_summary` calls
    that must decompose additively in percentage terms, e.g. Table S.A.1's
    reduced-infection + reduced-severity = total split -- the raw averted
    *counts* always sum correctly across such a chain, but the percentages
    only sum to the chain's overall percentage if every link's `pct_averted`
    shares the same denominator (the chain's starting `new_H`), not each
    link's own intermediate `reference`.

    `doses_reference` overrides which scenario's doses are subtracted to get
    the per-100K-doses denominator (default: `reference`'s). Needed when the
    two scenarios being differenced share an identical schedule, so their
    dose difference is exactly zero -- e.g. Table S.A.1's reduced-severity
    link (infection-protection-only -> full baseline), where the meaningful
    denominator is the full baseline dose count, i.e. `doses_reference=no_vax`.

    `doses_override` replaces the denominator for EVERY row (age rows and
    "All" alike) with one scalar or per-replicate dose count, instead of each
    row using the doses delivered to its own age group. Needed whenever the
    numerator and denominator refer to different age groups -- e.g. Table
    S.A.2/S.A.3's off-diagonal cells, which count hospitalizations averted in
    the ROW's age group per dose delivered to the COLUMN's age group. Without
    it those cells divide by zero doses (S.A.2) or by an incidental handful of
    dose-cap spillover doses (S.A.3) and are dropped or meaningless.
    """
    ref_H, scen_H = reference["new_H"], scenario["new_H"]   # (n_reps, A)
    averted = ref_H - scen_H
    pop = reference["population"]
    doses_ref = reference if doses_reference is None else doses_reference
    doses = scenario["doses"] - doses_ref["doses"]           # (A,) or (n_reps, A) additional doses given
    pct_denom_H = reference["new_H"] if pct_reference is None else pct_reference["new_H"]

    def row(label, averted_col, ref_col, pop_val, doses_val):
        pct = np.where(ref_col > 0, averted_col / np.where(ref_col > 0, ref_col, 1) * 100, np.nan)
        per100k = averted_col / pop_val * 1e5
        # doses_val is either a scalar (one schedule shared by every replicate)
        # or a per-replicate array (paired with averted_col) -- broadcasting
        # handles both the same way.
        doses_arr = np.broadcast_to(np.asarray(doses_val, dtype=float), np.shape(averted_col))
        if np.any(doses_arr > 0):
            per100k_doses = averted_col / np.where(doses_arr > 0, doses_arr, 1) * 1e5
            dm, dl, dh = _summ(per100k_doses)
            doses_str = f"{dm:.1f} [{dl:.1f} - {dh:.1f}]"
        else:
            doses_str = "—"
        pm, pl, ph = _summ(pct)
        rm, rl, rh = _summ(per100k)
        return {
            "age_group": label,
            "pct_averted": f"{pm:.1f}% [{pl:.1f}% - {ph:.1f}%]",
            "per100k_averted": f"{rm:.1f} [{rl:.1f} - {rh:.1f}]",
            "per100k_doses_averted": doses_str,
        }

    rows = []
    A = ref_H.shape[1]
    if by_age:
        for i in range(A):
            doses_val = (np.take(doses, i, axis=-1) if doses_override is None
                          else doses_override)
            rows.append(row(AGE_GROUPS[i], averted[:, i], pct_denom_H[:, i], pop[i], doses_val))
    all_doses = doses.sum(axis=-1) if doses_override is None else doses_override
    rows.append(row("All", averted.sum(axis=1), pct_denom_H.sum(axis=1), pop.sum(), all_doses))
    return pd.DataFrame(rows).set_index("age_group")


# ── Base inputs ───────────────────────────────────────────────────────────────

def load_base_inputs(fit_folder: str | None = None, method: str = "emcee",
                      point: str = "best") -> dict:
    """
    Build the base (baseline-vaccination) inputs dict. If `fit_folder` is
    given, applies the fitted overrides + m(t) from that `fit_bayesian.py`
    output folder; otherwise uses `default_params()`. `point` selects which
    point estimate of the posterior to use -- `"mean"` (per-parameter
    posterior mean) or `"best"` (single highest-support sampled parameter
    vector) -- see `fit_bayesian.load_fitted_run`. `"mean"` does not 
    correspond to any point actually visited by sampler and may result
    in simulations with a poor fit if there are strong correlations between
    input parameters. 
    """
    params = model.default_params()
    inputs = model.load_inputs(model.DATA_FOLDER, params)
    if fit_folder is not None:
        from MA_vax_standalone.fit_bayesian import load_fitted_run
        overrides, beta_multiplier_arr = load_fitted_run(fit_folder, method, point)
        inputs["params"] = model.apply_overrides(inputs["params"], overrides)
        if beta_multiplier_arr is not None:
            inputs["beta_multiplier_arr"] = beta_multiplier_arr
    return inputs


# ── Table builders ────────────────────────────────────────────────────────────

def table_S_A_1(base_inputs: dict, n_reps: int = 200, seed: int = 0, stochastic: bool = True) -> pd.DataFrame:
    """
    Hospitalizations averted by vaccination, decomposed into a
    reduced-infection effect (no-vaccine -> infection-protection-only) and a
    reduced-severity effect (infection-protection-only -> full baseline), by
    age group and aggregate. Analog of Table S.A.1.

    `stochastic=False` runs a single deterministic replication per scenario
    instead of `n_reps` stochastic ones (median/CI bounds then collapse to
    the point estimate).
    """
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
    """
    Hospitalizations averted by vaccinating a single age group (all other
    groups unvaccinated), relative to no vaccination. Returns
    {"pct_reduction", "per_100k", "per_100k_doses"}, each a (age group
    hospitalized) x (age group vaccinated, + "All") matrix. Analog of Table
    S.A.2. See `table_S_A_1` for `stochastic`.
    """
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
    """
    Additional hospitalizations averted if a single age group reached 70%
    coverage (others left at baseline), relative to the baseline vaccination
    scenario. Analog of Table S.A.3. See `table_S_A_1` for `stochastic`.
    """
    baseline = scenario_totals(base_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(base_inputs, coverage_70pct_scenario(base_inputs, i), n_reps, seed,
                               stochastic=stochastic)
        cols[label] = averted_summary(baseline, scen)
    all_scen = scenario_totals(base_inputs, coverage_70pct_scenario(base_inputs, None), n_reps, seed,
                               stochastic=stochastic)
    cols["All"] = averted_summary(baseline, all_scen)

    return {
        "pct_reduction": pd.DataFrame({label: df["pct_averted"] for label, df in cols.items()}),
        "per_100k": pd.DataFrame({label: df["per100k_averted"] for label, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({label: df["per100k_doses_averted"] for label, df in cols.items()}),
    }


def table_S_A_4(base_inputs: dict) -> pd.DataFrame:
    """
    Implied VE-against-infection (1 - vax_susceptibility), VE-against-
    hospitalization-given-infection (1 - IV_to_H_prop / I_to_H_prop), and
    VE-against-onward-transmission-given-infection (1 -
    IV_relative_infectiousness / I_relative_infectiousness) for each scenario
    in `VE_SCENARIOS`, by age group. Analog of Table S.A.4 (a parameter
    table, no simulation).

    `VE_transmission_blocking` is reported independently -- it's not folded
    into `VE_hosp_infection` (which stays a pure infection x hosp-given-
    infection combination) because it doesn't act on the vaccinated
    individual's own hospitalization risk at all; it reduces how infectious
    a vaccinated person is to *others*, which only shows up as an effect on
    population-level transmission dynamics (see the "reduced infection"
    share of Table S.A.1 being larger than `VE_infection` alone would
    suggest -- this is the uncounted mechanism behind that gap).
    """
    rows = []
    for name, scen in VE_SCENARIOS.items():
        inputs = model.apply_scenario(base_inputs, scen)
        vs = inputs["params"]["vax_susceptibility"]
        ivh = inputs["params"]["IV_to_H_prop"]
        ih = inputs["params"]["I_to_H_prop"]
        # I/IV_relative_infectiousness are scalars (unlike the per-age-group
        # params above), so this VE is the same across every age group.
        iv_rel_inf = float(inputs["params"]["IV_relative_infectiousness"])
        i_rel_inf = float(inputs["params"]["I_relative_infectiousness"])
        ve_inf = 1.0 - vs
        ve_hosp_given_inf = 1.0 - ivh / ih
        ve_hosp_inf = 1 - (1-ve_inf) * (1-ve_hosp_given_inf)
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
    """
    Hospitalizations averted relative to no vaccination, across the VE
    sensitivity scenarios in `VE_SCENARIOS`. Analog of Table S.A.5. See
    `table_S_A_1` for `stochastic`.
    """
    no_vax = scenario_totals(base_inputs, no_vaccine_scenario(), n_reps, seed, stochastic=stochastic)
    cols = {
        name: averted_summary(no_vax, scenario_totals(base_inputs, scen, n_reps, seed, stochastic=stochastic))
        for name, scen in VE_SCENARIOS.items()
    }
    return {
        "pct_reduction": pd.DataFrame({name: df["pct_averted"] for name, df in cols.items()}),
        "per_100k": pd.DataFrame({name: df["per100k_averted"] for name, df in cols.items()}),
    }


def table_S_A_6(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                stochastic: bool = True) -> dict[str, pd.DataFrame]:
    """
    Additional hospitalizations averted if all age groups reached 70%
    coverage, relative to that VE scenario's own baseline vaccination,
    across the VE sensitivity scenarios in `VE_SCENARIOS`. Analog of Table
    S.A.6. See `table_S_A_1` for `stochastic`.
    """
    cols = {}
    for name, ve_scen in VE_SCENARIOS.items():
        ve_inputs = model.apply_scenario(base_inputs, ve_scen)
        baseline = scenario_totals(ve_inputs, baseline_scenario(), n_reps, seed, stochastic=stochastic)
        target70 = scenario_totals(ve_inputs, coverage_70pct_scenario(ve_inputs, None), n_reps, seed,
                                   stochastic=stochastic)
        cols[name] = averted_summary(baseline, target70)
    return {
        "pct_reduction": pd.DataFrame({name: df["pct_averted"] for name, df in cols.items()}),
        "per_100k": pd.DataFrame({name: df["per100k_averted"] for name, df in cols.items()}),
    }


def _scenario_check_sums(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                          steps_per_day: int = 1, stochastic: bool = True) -> dict[str, np.ndarray]:
    """Run `scenario` and return the season totals needed by
    `table_vax_efficacy_check`: `S0`/`SV0` are (A,) initial-condition
    snapshots (identical across replications); the rest are (reps, A) season
    sums of that transition's flow."""
    inputs = model.apply_scenario(base_inputs, scenario)
    ds = _run_reps(inputs, n_reps, seed, steps_per_day, stochastic=stochastic)
    flows = ["S_to_E", "S_to_SV", "SV_to_EV", "I_to_H", "IV_to_H"]
    out = {t: ds[t].sum(dim="day").to_numpy() for t in flows}
    out["S0"] = ds["S"].isel(replication=0, day=0).to_numpy()
    out["SV0"] = ds["SV"].isel(replication=0, day=0).to_numpy()
    return out


def _rate_ratio_col(vax_num: np.ndarray, vax_den: np.ndarray,
                     unvax_num: np.ndarray, unvax_den: np.ndarray) -> list[str]:
    """Per age group (columns of the (reps, A) arrays), median [95% CI] of
    the per-replication rate ratio (vax_num/vax_den) / (unvax_num/unvax_den),
    as a percentage string. `"—"` where undefined (either denominator zero)
    in every replication."""
    A = vax_num.shape[1]
    out = []
    for i in range(A):
        with np.errstate(divide="ignore", invalid="ignore"):
            vax_rate = np.where(vax_den[:, i] > 0, vax_num[:, i] / vax_den[:, i], np.nan)
            unvax_rate = np.where(unvax_den[:, i] > 0, unvax_num[:, i] / unvax_den[:, i], np.nan)
            ratio = vax_rate / unvax_rate
        if np.all(np.isnan(ratio)):
            out.append("—")
            continue
        m = float(np.nanmedian(ratio)) * 100
        lo = float(np.nanpercentile(ratio, 2.5)) * 100
        hi = float(np.nanpercentile(ratio, 97.5)) * 100
        out.append(f"{m:.1f}% [{lo:.1f}% - {hi:.1f}%]")
    return out


def _scenario_daily_arrays(base_inputs: dict, scenario: dict, n_reps: int = 200, seed: int = 0,
                           steps_per_day: int = 1, stochastic: bool = True) -> dict[str, np.ndarray]:
    """Run `scenario` and return full (reps, day, A) arrays for the
    compartments/flows needed by `_matched_cohort_ratio_col` -- unlike
    `_scenario_check_sums`, the `day` axis is kept rather than summed away."""
    inputs = model.apply_scenario(base_inputs, scenario)
    ds = _run_reps(inputs, n_reps, seed, steps_per_day, stochastic=stochastic)
    names = ["S", "SV", "S_to_E", "SV_to_EV", "S_to_SV"]
    return {name: ds[name].to_numpy() for name in names}


def attack_probability_curves(S: np.ndarray, SV: np.ndarray, S_to_E: np.ndarray,
                               SV_to_EV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given (reps, day, A) compartment/flow arrays, return `(attack_S_from,
    attack_SV_from)`, each (reps, day, A): `attack_S_from[r, d, a]` /
    `attack_SV_from[r, d, a]` is the counterfactual season-end attack
    probability for a hypothetical individual entering the S/SV pool on day
    `d` (in replication `r`, age group `a`) and followed to season end --
    see `table_vax_efficacy_check`'s docstring (the
    "matched_cohort_infection_reduction" entry) for the full derivation.
    Used both by `_matched_cohort_ratio_col` (which reduces these into a
    single ratio) and directly by `counterfactual_notebook.py` to plot the
    curves themselves.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        haz_S = np.where(S > 0, S_to_E / S, 0.0)
        haz_SV = np.where(SV > 0, SV_to_EV / SV, 0.0)
    haz_S = np.clip(haz_S, 0.0, 1.0)
    haz_SV = np.clip(haz_SV, 0.0, 1.0)

    # Suffix log-survival along the day axis: logsurv[r, d, a] = sum_{t=d}^{T-1} log(1 - haz[r, t, a]).
    # A reversed cumsum of the reversed log-hazard array gives this in one vectorized pass.
    def suffix_log_survival(haz: np.ndarray) -> np.ndarray:
        log1m = np.log1p(-haz)
        return np.cumsum(log1m[:, ::-1, :], axis=1)[:, ::-1, :]

    attack_S_from = 1.0 - np.exp(suffix_log_survival(haz_S))    # (reps, day, A)
    attack_SV_from = 1.0 - np.exp(suffix_log_survival(haz_SV))  # (reps, day, A)
    return attack_S_from, attack_SV_from


def _matched_cohort_ratio_col(S: np.ndarray, SV: np.ndarray, S_to_E: np.ndarray,
                               SV_to_EV: np.ndarray, S_to_SV: np.ndarray) -> list[str]:
    """
    Per age group (last axis of the (reps, day, A) arrays), median [95% CI]
    of the per-replication matched-cohort attack-rate ratio -- see
    `table_vax_efficacy_check` docstring for the formula and rationale.
    `"—"` where no one was ever vaccinated in that (rep, age) slice.
    """
    attack_S_from, attack_SV_from = attack_probability_curves(S, SV, S_to_E, SV_to_EV)

    weight = S_to_SV  # (reps, day, A)
    total_weight = weight.sum(axis=1)  # (reps, A)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_attack_vax = np.where(
            total_weight > 0, (weight * attack_SV_from).sum(axis=1) / total_weight, np.nan)
        weighted_attack_unvax = np.where(
            total_weight > 0, (weight * attack_S_from).sum(axis=1) / total_weight, np.nan)
        ratio = weighted_attack_vax / weighted_attack_unvax  # (reps, A)

    A = ratio.shape[1]
    out = []
    for i in range(A):
        col = ratio[:, i]
        if np.all(np.isnan(col)):
            out.append("—")
            continue
        m = float(np.nanmedian(col)) * 100
        lo = float(np.nanpercentile(col, 2.5)) * 100
        hi = float(np.nanpercentile(col, 97.5)) * 100
        out.append(f"{m:.1f}% [{lo:.1f}% - {hi:.1f}%]")
    return out


def table_vax_efficacy_check(base_inputs: dict, n_reps: int = 200, seed: int = 0,
                              stochastic: bool = True) -> dict[str, pd.DataFrame]:
    """
    Direct mechanism check that vaccination is doing what it should, at the
    flow level rather than the population-averted level of Tables S.A.*.
    Same layout as Table S.A.2: columns are "which age group was vaccinated"
    (single age group only, plus "All" = baseline schedule), rows are the age
    group the flow is counted in. All three entries are rate ratios
    (vaccinated rate / unvaccinated rate), reported as a percentage -- below
    100% means vaccination is reducing that risk; a value near 100% would
    indicate the mechanism isn't doing anything.

    "infection_reduction": attack-rate ratio
        [sum(SV_to_EV) / (SV[0] + sum(S_to_SV))] / [sum(S_to_E) / (S[0] - sum(S_to_SV))]
    Each rate's denominator is the size of the pool that was ever at risk of
    that flow over the season. **This is biased by immortal-time bias** and
    kept only as a "what a naive analysis would show" comparison point:
    vaccination is staggered across the season, so the SV cohort's average
    follow-up window (from whenever each person was vaccinated to season
    end) is shorter than the reference S[0] cohort's full-season window.
    Verified against a case with zero modeled infection protection
    (`vax_susceptibility=1.0`, 65+ age group): this formula still reports a
    spurious ~20 percentage-point "reduction" purely from that timing
    effect, even though there's no susceptibility difference to detect.

    "matched_cohort_infection_reduction": the bias-corrected version, and
    the one actually comparable to real-world reported VE (RCTs synchronize
    enrollment; surveillance/cohort studies use vaccination as a
    time-varying covariate specifically to avoid this same immortal-time
    bias). Uses the fact that in this model the per-capita daily hazard is a
    pure function of calendar day, not of compartment size:
        haz_S(t)  = S_to_E(t)  / S(t)   == foi(t) * rel_suscept * dt
        haz_SV(t) = SV_to_EV(t) / SV(t) == foi(t) * vax_susceptibility * dt
    So for any day `d`, the counterfactual season-end attack probability for
    a hypothetical individual entering the pool on day `d` and followed to
    season end `T` is:
        attack_S_from(d)  = 1 - prod_{t=d}^{T-1} (1 - haz_S(t))
        attack_SV_from(d) = 1 - prod_{t=d}^{T-1} (1 - haz_SV(t))
    Weighting across the *actual* vaccination-day distribution using
    `S_to_SV(d)` (how many people were actually vaccinated on day `d`) and
    averaging gives a season-total attack-rate ratio with matched start
    days between the two arms -- eliminating the immortal-time bias. When
    `haz_SV(t) == haz_S(t)` for every `t` (e.g. the 65+ case above), this
    ratio is exactly 100%, with no artifact.

    "hospitalization_reduction": rate ratio of hospitalization-given-infection
        [sum(IV_to_H) / sum(SV_to_EV)] / [sum(I_to_H) / sum(S_to_E)]
    i.e. P(hospitalized | infected, vaccinated) / P(hospitalized | infected,
    unvaccinated) -- directly comparable to `IV_to_H_prop / I_to_H_prop` in
    Table S.A.4. Needs no timing correction: `I_to_H_prop`/`IV_to_H_prop`
    are constant rates conditional on already being infected, not driven by
    the time-varying force of infection the way `S_to_E`/`SV_to_EV` are, so
    there's no staggered-entry confound here.

    Off-diagonal columns/rows (age group vaccinated != age group of the
    flow) are undefined ("—") under `single_age_only_scenario`, since only
    the vaccinated age group has any SV/IV population there; they're only
    informative in the "All" column.
    """
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
# Sub-table names for the dict-valued tables -- single source of truth shared
# by `save_all_tables` (writer, used by run_counterfactual_tables.py) and
# `load_saved_tables` (reader, used by counterfactual_notebook.py). Splitting
# compute from display this way means the (potentially slow, many-replication)
# simulation only has to run once per CLI invocation, and the notebook just
# loads+renders whatever CSVs are already on disk.
DICT_TABLES = {
    "S_A_2": ["pct_reduction", "per_100k", "per_100k_doses"],
    "S_A_3": ["pct_reduction", "per_100k", "per_100k_doses"],
    "S_A_5": ["pct_reduction", "per_100k"],
    "S_A_6": ["pct_reduction", "per_100k"],
    "VAX_CHECK": ["infection_reduction", "matched_cohort_infection_reduction", "hospitalization_reduction"],
}


def save_all_tables(base_inputs: dict, out_folder: str, n_reps: int = 500, seed: int = 0,
                    stochastic: bool = True, meta: dict | None = None, log=print) -> None:
    """
    Compute every S.A.* table and write it to `out_folder` as CSV (one file
    per DataFrame; dict-valued tables get one file per sub-table, named
    `<table>_<subname>.csv`), plus a `meta.json` with run metadata.

    `stochastic=False` runs a single deterministic replication per scenario
    instead of `n_reps` stochastic ones (median/CI bounds then collapse to
    the point estimate) -- useful as a fast sanity check before committing
    to a large stochastic run.
    """
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


def load_saved_tables(results_folder: str) -> dict:
    """
    Read back everything `save_all_tables` wrote, in the same shapes the
    `table_S_A_*` functions return (a DataFrame, or a dict of DataFrame for
    the multi-metric tables), plus a `"meta"` key with the run metadata.
    Returns `None` for any table whose CSV isn't present in `results_folder`.
    """
    def read(filename, index_col: int | list[int] = 0):
        path = os.path.join(results_folder, filename)
        return pd.read_csv(path, index_col=index_col) if os.path.exists(path) else None

    meta_path = os.path.join(results_folder, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    tables = {"meta": meta, "S_A_1": read("S_A_1.csv"), "S_A_4": read("S_A_4.csv", index_col=[0, 1])}
    for name, subs in DICT_TABLES.items():
        tables[name] = {sub: read(f"{name}_{sub}.csv") for sub in subs}
    return tables
