"""Turn a results source (written by run_simulations_MA_vax.py) into the
counterfactual vaccination-impact tables that counterfactual_notebook_generic.py
loads via `cf.load_saved_tables`.

This is the missing link between two things already in this folder:

  run_simulations_MA_vax.py   -- an exported script (downloadable straight from
                                  the Model Builder notebook's Export tab) that
                                  runs every scenario in its SCENARIOS dict and
                                  writes per-replicate, per-subpop, per-age-group,
                                  per-risk-group daily compartment/transition
                                  history to simulation_output/ (a Hive-partitioned
                                  Parquet directory by default, or a SQLite
                                  results.db from an older run / the notebook's
                                  Analysis tab SQLite export).
  counterfactual_notebook_generic.py -- displays Tables S.A.1-S.A.6 + the
                                  vaccine-efficacy check, but only ever reads
                                  already-computed CSVs from a results folder
                                  (via cf.load_saved_tables); it never runs a
                                  simulation for these tables itself.

Unlike counterfactual_generic.py's own `save_all_tables` (which reimplements
build_model/scenario-running against generic_core directly -- a second,
independently-defined simulation pipeline), this script computes the exact
same tables purely by reading the results source: the notebook-exported
script is the only thing that ever runs a simulation. Reading goes through
`generic_core.results_io.load_source`, the same loader the results-explorer
notebook uses, so this script works unchanged whether `--db` points at a
Parquet directory or a SQLite `.db` file. The pure numeric helpers that turn
already-simulated (reps, day, age) arrays into table rows (`averted_summary`,
`_rate_ratio_col`, `_matched_cohort_ratio_col`, ...) are reused unchanged from
MA_vax.counterfactual, same as counterfactual_generic.py does -- they don't
care which pipeline produced the arrays.

Usage:
    cd generic_core/examples/MA_vax
    python run_simulations_MA_vax.py
    python build_counterfactual_tables_from_db.py --db simulation_output/results_parquet

Requires a results source with a `results_full` table (per-subpop/age-group/
risk-group history -- added alongside the population-summed `results` table
so per-age-group tables like S.A.2/S.A.3/S.A.5/S.A.6/VAX_CHECK are possible
at all). Re-run run_simulations_MA_vax.py if your source predates that table.

The coverage-target and coverage-decrease scenarios come from
make_scenarios_MA_vax.py (COVERAGE_TARGETS / COVERAGE_DECREASES), which writes
them into run_simulations_MA_vax.py; this script reads the same two lists, so
adding a target there is enough. Outputs, besides S.A.1/2/4/5 and VAX_CHECK:
  S_A_3_<T>pct_*        raising one age group (or all) to T% coverage, vs baseline;
                        S_A_3_* (no suffix) is the 70% one, kept under its old name
  DECREASE_<D>pct_*     cutting one age group's (or all) doses by D%: ADDITIONAL
                        hospitalizations vs baseline, per 100K doses removed
  COVERAGE_SUMMARY.csv  numeric median / 2.5% / 97.5% totals for baseline and every
                        "(all ages)" coverage scenario, for plotting
(Table S.A.6 -- VE presets combined with coverage targets -- is no longer built:
those combined scenarios were dropped from SCENARIOS.)

Location: this file must sit exactly two directory levels below the repo
root that contains generic_core/, clt_toolkit/ and flu_core/, and next to
model_config.json/fitted_params.json -- same constraint as
run_simulations_MA_vax.py / counterfactual_generic.py in this folder.

Single-population only in the table math (matches run_simulations_MA_vax.py's
IS_METAPOP=False export here): `population` comes from model_config.json's
single aggregate_pop entry, and every age-resolved array sums `results_full`
across subpop AND risk group via SQL GROUP BY. `results_full` itself does
carry subpop/risk-group resolution (harmless no-op columns here, since
MA_vax has one subpop and one risk group) -- a metapop/multi-risk-group
extension of these tables would read those columns instead of summing them
away, and would also need a per-subpop `population` (this script's single
scalar-per-age-group `population` array doesn't generalize to that).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE))

from ma_vax_shared import (
    AGE_GROUP_LABELS as AGE_GROUPS,
    averted_summary, _rate_ratio_col, _matched_cohort_ratio_col,
)

import counterfactual_generic as cf  # table_S_A_4 (parameter-only), plus the scheduled-dose helpers
from make_scenarios_MA_vax import COVERAGE_TARGETS, COVERAGE_DECREASES
from ma_vax_shared import _summ
from generic_core import results_io

MODEL_CONFIG_FILE = _HERE / "model_config_MA_vax.json"
DEFAULT_DB = _HERE / "simulation_output_MA_vax_param_set_stochastic" / "results_parquet"
DEFAULT_OUT = _HERE / "counterfactual_tables_from_db_MA_vax"

# Named-scenario -> results.db `scenario` column value. Matches the literal
# scenario names in run_simulations_MA_vax.py's SCENARIOS dict -- edit this
# map if scenarios are renamed there.
SCENARIO_DB_NAME = {
    "baseline": "baseline",
    "no vax": "no vax",
    "Infection protection only": "Infection protection only",
    **{f"Vaccinate {label} only": f"Vaccinate {label} only" for label in AGE_GROUPS},
    **{f"{pct}% coverage (all ages)": f"{pct}% coverage (all ages)" for pct in COVERAGE_TARGETS},
    **{f"{pct}% coverage ({label} only)": f"{pct}% coverage ({label} only)"
       for pct in COVERAGE_TARGETS for label in AGE_GROUPS},
    **{f"{d}% lower coverage (all ages)": f"{d}% lower coverage (all ages)" for d in COVERAGE_DECREASES},
    **{f"{d}% lower coverage ({label} only)": f"{d}% lower coverage ({label} only)"
       for d in COVERAGE_DECREASES for label in AGE_GROUPS},
}
VE_TOTALS_DB_NAME = {"low_ve": "Low VE", "baseline_ve": "baseline", "high_ve": "High VE"}


def load_population(model_config_file=MODEL_CONFIG_FILE) -> np.ndarray:
    with open(model_config_file) as f:
        config_dict = json.load(f)
    ic_entry = (config_dict.get("initial_conditions", {}) or {}).get("aggregate_pop", {})
    return np.asarray(
        ic_entry.get("population", [[0.0]] * len(AGE_GROUPS)), dtype=float
    ).reshape(len(AGE_GROUPS))


class ResultsDB:
    """Age-resolved (reps, day, age_group) arrays for a scenario, read from
    the `results_full` table run_simulations_MA_vax.py writes -- summed over
    subpop and risk group (via SQL GROUP BY), since every table this script
    builds wants population totals by age group, not by subpop/risk group.
    Single-population, single-risk-group models (MA_vax) have nothing to sum
    away there, so this is a no-op for them."""

    def __init__(self, db_path: str):
        self.path = db_path
        # results_io.load_source handles a Hive-partitioned Parquet directory
        # (current output format), a SQLite .db (older runs / the notebook's
        # Analysis tab SQLite export) or a legacy .json export uniformly,
        # returning a duckdb connection exposing `results`/`results_full`
        # views -- it also raises a clear error if `results_full` is missing,
        # so no separate check is needed here.
        try:
            self._con = results_io.load_source(db_path)
        except results_io.ResultsExplorerError as exc:
            raise SystemExit(str(exc)) from exc
        # The table builders re-query the same scenario's arrays several
        # times across different tables (e.g. "baseline" is used by S.A.1,
        # S.A.2, S.A.3, S.A.5, S.A.6, and VAX_CHECK) -- cache per (scenario,
        # names) so each combination only hits the DB once.
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def scenarios_present(self) -> set[str]:
        return {r[0] for r in self._con.execute("SELECT DISTINCT scenario FROM results_full").fetchall()}

    def n_reps(self, scenario: str) -> int:
        return self._con.execute(
            "SELECT COUNT(DISTINCT rep) FROM results_full WHERE scenario = ?", [scenario]
        ).fetchone()[0]

    def arrays(self, scenario: str, names: list[str]) -> dict[str, np.ndarray]:
        # Table builders re-request the same (scenario, compartment) pairs
        # across different tables (e.g. "baseline" x "I_to_H" is used by
        # S.A.1, S.A.2, S.A.3, S.A.5 and S.A.6) -- serve those from cache and
        # only query the DB for names not already fetched for this scenario.
        cached = self._cache.setdefault(scenario, {})
        missing = [n for n in names if n not in cached]
        if missing:
            placeholders = ",".join("?" * len(missing))
            df = self._con.execute(
                f"SELECT rep, compartment, age_group, day, SUM(value) AS value FROM results_full "
                f"WHERE scenario = ? AND compartment IN ({placeholders}) "
                f"GROUP BY rep, compartment, age_group, day",
                [scenario, *missing],
            ).df()
            if df.empty:
                raise ValueError(
                    f"No rows for scenario {scenario!r} in results_full -- check it's "
                    "in run_simulations_MA_vax.py's SCENARIOS and the script has been re-run."
                )
            reps = sorted(df["rep"].unique().tolist())
            days = sorted(df["day"].unique().tolist())
            n_age = len(AGE_GROUPS)
            rep_pos = {r: i for i, r in enumerate(reps)}
            day_pos = {d: i for i, d in enumerate(days)}
            for name in missing:
                sub = df[df["compartment"] == name]
                if sub.empty:
                    raise ValueError(
                        f"Scenario {scenario!r} has no {name!r} rows -- check TRANSITION_VARS "
                        "in run_simulations_MA_vax.py includes it."
                    )
                arr = np.zeros((len(reps), len(days), n_age))
                arr[sub["rep"].map(rep_pos).to_numpy(),
                    sub["day"].map(day_pos).to_numpy(),
                    sub["age_group"].to_numpy()] = sub["value"].to_numpy()
                cached[name] = arr
        return {name: cached[name] for name in names}

    def close(self):
        self._con.close()


def scenario_totals(db: ResultsDB, scenario: str, population: np.ndarray) -> dict:
    arrs = db.arrays(scenario, ["I_to_H", "IV_to_H", "S_to_SV"])
    new_H = (arrs["I_to_H"] + arrs["IV_to_H"]).sum(axis=1)  # (reps, A)
    # Doses are schedule-driven, not transition-RNG-driven, so under
    # UNCERTAINTY_SOURCE="transitions" every replicate's schedule is
    # identical and any one of them is representative. But under
    # UNCERTAINTY_SOURCE="parameters" each replicate is a different sampled
    # parameter set, and coverage-target scenarios (e.g. table S.A.3/S.A.6's
    # "scale to 70% coverage") derive their dose multiplier from that
    # replicate's own baseline coverage -- so the schedule itself varies
    # per replicate and must be kept per-replicate here, paired with new_H.
    doses = arrs["S_to_SV"].sum(axis=1)  # (reps, A)
    return {"new_H": new_H, "doses": doses, "population": population}


def scenario_check_sums(db: ResultsDB, scenario: str) -> dict[str, np.ndarray]:
    arrs = db.arrays(scenario, ["S_to_E", "S_to_SV", "SV_to_EV", "I_to_H", "IV_to_H", "S", "SV"])
    out = {t: arrs[t].sum(axis=1) for t in ["S_to_E", "S_to_SV", "SV_to_EV", "I_to_H", "IV_to_H"]}
    out["S0"] = arrs["S"][0, 0, :]
    out["SV0"] = arrs["SV"][0, 0, :]
    return out


def scenario_daily_arrays(db: ResultsDB, scenario: str) -> dict[str, np.ndarray]:
    return db.arrays(scenario, ["S", "SV", "S_to_E", "SV_to_EV", "S_to_SV"])


# ── Table builders -- same structure as counterfactual_generic.py's, but
# every array comes from results.db instead of a fresh simulation ──────────

def _sched_doses(totals: dict, doses) -> dict:
    """Copy of a `scenario_totals` dict with realized `S_to_SV` doses replaced by
    SCHEDULED doses. Every per-100K-doses panel uses this: a dose the §1.4 cap
    declines to deliver (because the recipient is already infected) is still
    bought and administered in the real world, so it belongs in the denominator.
    See counterfactual_generic.scheduled_doses."""
    return {**totals, "doses": np.asarray(doses, dtype=float)}


def table_S_A_1(db: ResultsDB, population: np.ndarray,
                 model_config_file=MODEL_CONFIG_FILE) -> pd.DataFrame:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    inf_only = scenario_totals(db, SCENARIO_DB_NAME["Infection protection only"], population)
    full = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    sched = cf.scheduled_doses(population, model_config_file=model_config_file)
    no_vax = _sched_doses(no_vax, np.zeros_like(sched))
    inf_only, full = _sched_doses(inf_only, sched), _sched_doses(full, sched)
    reduced_infection = averted_summary(no_vax, inf_only).add_suffix("_reduced_infection")
    # `inf_only` and `full` share an identical vaccination schedule (adding back
    # severity protection doesn't change who gets vaccinated), so their dose
    # difference is exactly zero -- divide by the full baseline dose count
    # instead, the same denominator the reduced-infection and total links use.
    reduced_severity = averted_summary(inf_only, full, pct_reference=no_vax,
                                        doses_reference=no_vax).add_suffix("_reduced_severity")
    total = averted_summary(no_vax, full).add_suffix("_total")
    return reduced_infection.join(reduced_severity).join(total)


def table_S_A_2(db: ResultsDB, population: np.ndarray,
                 model_config_file=MODEL_CONFIG_FILE) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    sched = cf.scheduled_doses(population, model_config_file=model_config_file)
    no_vax = _sched_doses(no_vax, np.zeros_like(sched))
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(db, SCENARIO_DB_NAME[f"Vaccinate {label} only"], population)
        # Off-diagonal cells count hospitalizations averted in the ROW's age group
        # per 100K doses delivered to the COLUMN's age group -- only group `i` is
        # vaccinated in this scenario, so that denominator is its own dose count
        # (per replicate, paired with the numerator). The diagonal and the "All"
        # row are unchanged by this: they already used exactly that denominator.
        cols[label] = averted_summary(no_vax, scen, doses_override=float(sched[i]))
    # The "All vaccinated" column has no single targeted age group, so each row
    # keeps its own age group's dose count (and the "All" row the total) -- the
    # same convention as table S.A.1's per-dose columns, which it must match.
    baseline = _sched_doses(scenario_totals(db, SCENARIO_DB_NAME["baseline"], population), sched)
    cols["All"] = averted_summary(no_vax, baseline)
    return {
        "absolute": pd.DataFrame({l: df["averted"] for l, df in cols.items()}),
        "pct_reduction": pd.DataFrame({l: df["pct_averted"] for l, df in cols.items()}),
        "per_100k": pd.DataFrame({l: df["per100k_averted"] for l, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({l: df["per100k_doses_averted"] for l, df in cols.items()}),
    }


def table_S_A_3(db: ResultsDB, population: np.ndarray,
                 model_config_file=MODEL_CONFIG_FILE) -> dict[str, pd.DataFrame]:
    """70%-coverage Table S.A.3 (see table_S_A_3_for_target)."""
    return table_S_A_3_for_target(db, population, 70, model_config_file=model_config_file)


def table_S_A_3_for_target(db: ResultsDB, population: np.ndarray, target_pct: int,
                            model_config_file=MODEL_CONFIG_FILE) -> dict[str, pd.DataFrame]:
    """Raising one age group (columns) -- or all, "All" -- to `target_pct`%
    coverage, vs baseline. Reads the "{target_pct}% coverage (...)" scenarios.

    Per-100K-doses cells divide by the ADDITIONAL doses going to the COLUMN's
    age group, taken from the SCHEDULE rather than from realized S_to_SV:
      * realized doses undercount -- the model's cap declines to vaccinate
        someone already infected, but that dose is still bought and given;
      * realized doses also leak across age groups (a milder epidemic leaves
        more susceptibles for the untargeted groups' unchanged schedule to
        reach), and dividing an indirect effect by a handful of incidental
        doses gives absurd magnitudes.
    The scheduled figure is max(0, target - baseline coverage) * population.
    The "All" column divides every row by the TOTAL additional doses, so rows
    also sum to the "All" row."""
    target = target_pct / 100.0
    baseline = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(db, SCENARIO_DB_NAME[f"{target_pct}% coverage ({label} only)"], population)
        col_doses = cf.additional_scheduled_doses_for_target(
            population, target, i, model_config_file=model_config_file)
        cols[label] = averted_summary(baseline, scen, doses_override=col_doses)
    all_scen = scenario_totals(db, SCENARIO_DB_NAME[f"{target_pct}% coverage (all ages)"], population)
    extra = float(cf.additional_scheduled_doses_for_target(
        population, target, model_config_file=model_config_file).sum())
    cols["All"] = averted_summary(baseline, all_scen, doses_override=extra)
    return {
        "absolute": pd.DataFrame({l: df["averted"] for l, df in cols.items()}),
        "pct_reduction": pd.DataFrame({l: df["pct_averted"] for l, df in cols.items()}),
        "per_100k": pd.DataFrame({l: df["per100k_averted"] for l, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({l: df["per100k_doses_averted"] for l, df in cols.items()}),
    }


def table_dose_accounting(db: ResultsDB, population: np.ndarray,
                           model_config_file=MODEL_CONFIG_FILE) -> pd.DataFrame:
    """Scheduled vs. delivered doses under the baseline schedule, by age group.

    "Scheduled" is what the vaccination schedule ships: `scheduled_coverage *
    population`. "Delivered" is what the model records as an actual S -> SV
    transition (median across parameter draws). The gap is the doses the §1.4
    cap declines to hand out because the intended recipient has already been
    infected -- real doses, bought and administered, that buy no protection.
    This is the reconciliation between the two dose figures, and it is why the
    per-100K-doses panels use the scheduled count as their denominator.
    """
    sched = cf.scheduled_doses(population, model_config_file=model_config_file)
    delivered = np.median(scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)["doses"],
                           axis=0)
    wasted = sched - delivered
    rows = [{"age_group": AGE_GROUPS[i],
             "population": f"{population[i]:,.0f}",
             "scheduled_doses": f"{sched[i]:,.0f}",
             "delivered_doses": f"{delivered[i]:,.0f}",
             "wasted_doses": f"{wasted[i]:,.0f}",
             "pct_wasted": f"{wasted[i] / sched[i] * 100:.1f}%",
             "scheduled_coverage": f"{sched[i] / population[i] * 100:.1f}%",
             "delivered_coverage": f"{delivered[i] / population[i] * 100:.1f}%"}
            for i in range(len(AGE_GROUPS))]
    rows.append({"age_group": "All", "population": f"{population.sum():,.0f}",
                 "scheduled_doses": f"{sched.sum():,.0f}",
                 "delivered_doses": f"{delivered.sum():,.0f}",
                 "wasted_doses": f"{wasted.sum():,.0f}",
                 "pct_wasted": f"{wasted.sum() / sched.sum() * 100:.1f}%",
                 "scheduled_coverage": f"{sched.sum() / population.sum() * 100:.1f}%",
                 "delivered_coverage": f"{delivered.sum() / population.sum() * 100:.1f}%"})
    return pd.DataFrame(rows).set_index("age_group")


def table_S_A_5(db: ResultsDB, population: np.ndarray) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    cols = {
        name: averted_summary(no_vax, scenario_totals(db, db_name, population))
        for name, db_name in VE_TOTALS_DB_NAME.items()
    }
    return {
        "absolute": pd.DataFrame({l: df["averted"] for l, df in cols.items()}),
        "pct_reduction": pd.DataFrame({n: df["pct_averted"] for n, df in cols.items()}),
        "per_100k": pd.DataFrame({n: df["per100k_averted"] for n, df in cols.items()}),
    }


def table_decrease(db: ResultsDB, population: np.ndarray, decrease_pct: int,
                   model_config_file=MODEL_CONFIG_FILE) -> dict[str, pd.DataFrame]:
    """Cutting one age group's (columns) -- or every group's, "All" -- baseline
    doses by `decrease_pct`%, vs baseline. Reads the "{D}% lower coverage (...)"
    scenarios. Cells are ADDITIONAL hospitalizations (scenario - baseline), the
    percentage is relative to baseline, and per-100K-doses divides by the
    SCHEDULED doses removed from the column's age group ("All": from everyone)."""
    frac = decrease_pct / 100.0
    baseline = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    sched = cf.scheduled_doses(population, model_config_file=model_config_file)
    cols = {}
    for i, label in enumerate(AGE_GROUPS):
        scen = scenario_totals(db, SCENARIO_DB_NAME[f"{decrease_pct}% lower coverage ({label} only)"],
                               population)
        cols[label] = averted_summary(scen, baseline, pct_reference=baseline,
                                      doses_override=frac * float(sched[i]))
    all_scen = scenario_totals(db, SCENARIO_DB_NAME[f"{decrease_pct}% lower coverage (all ages)"],
                               population)
    cols["All"] = averted_summary(all_scen, baseline, pct_reference=baseline,
                                  doses_override=frac * float(sched.sum()))
    return {
        "absolute": pd.DataFrame({l: df["averted"] for l, df in cols.items()}),
        "pct_increase": pd.DataFrame({l: df["pct_averted"] for l, df in cols.items()}),
        "per_100k": pd.DataFrame({l: df["per100k_averted"] for l, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({l: df["per100k_doses_averted"] for l, df in cols.items()}),
    }


def table_coverage_summary(db: ResultsDB, population: np.ndarray,
                           model_config_file=MODEL_CONFIG_FILE) -> pd.DataFrame:
    """One numeric row per "(all ages)" coverage scenario (plus baseline and no
    vax): scheduled coverage and doses, and median / 2.5% / 97.5% across
    parameter draws of total hospitalizations and of the change vs baseline
    (positive = more hospitalizations than baseline)."""
    base_cov = np.asarray(cf.scheduled_coverage(model_config_file=model_config_file), dtype=float)
    base_doses = base_cov * population
    baseline = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    base_H = baseline["new_H"].sum(axis=1)
    specs = [("no vax", "No vaccination", np.zeros_like(base_doses)),
             ("baseline", "Baseline", base_doses)]
    for d in sorted(COVERAGE_DECREASES, reverse=True):
        specs.append((f"{d}% lower coverage (all ages)", f"-{d}%", base_doses * (1 - d / 100.0)))
    for t in COVERAGE_TARGETS:
        specs.append((f"{t}% coverage (all ages)", f"{t}% target",
                      np.maximum(base_cov, t / 100.0) * population))
    rows = []
    for name, label, doses in specs:
        H = scenario_totals(db, SCENARIO_DB_NAME[name], population)["new_H"].sum(axis=1)
        h_m, h_lo, h_hi = _summ(H)
        d_m, d_lo, d_hi = _summ(H - base_H)
        p_m, p_lo, p_hi = _summ((H - base_H) / base_H * 100)
        rows.append({
            "scenario": name, "label": label,
            "scheduled_coverage_pct": doses.sum() / population.sum() * 100,
            "scheduled_doses": doses.sum(),
            "dose_change_vs_baseline": doses.sum() - base_doses.sum(),
            "hosp_median": h_m, "hosp_lo": h_lo, "hosp_hi": h_hi,
            "change_vs_baseline_median": d_m, "change_vs_baseline_lo": d_lo,
            "change_vs_baseline_hi": d_hi,
            "pct_change_median": p_m, "pct_change_lo": p_lo, "pct_change_hi": p_hi,
        })
    return pd.DataFrame(rows)


def table_vax_efficacy_check(db: ResultsDB) -> dict[str, pd.DataFrame]:
    cols_inf, cols_hosp, cols_matched = {}, {}, {}
    for label in AGE_GROUPS:
        db_name = SCENARIO_DB_NAME[f"Vaccinate {label} only"]
        s = scenario_check_sums(db, db_name)
        vax_den = s["SV0"][None, :] + s["S_to_SV"]
        unvax_den = s["S0"][None, :] - s["S_to_SV"]
        cols_inf[label] = _rate_ratio_col(s["SV_to_EV"], vax_den, s["S_to_E"], unvax_den)
        cols_hosp[label] = _rate_ratio_col(s["IV_to_H"], s["SV_to_EV"], s["I_to_H"], s["S_to_E"])

        d = scenario_daily_arrays(db, db_name)
        cols_matched[label] = _matched_cohort_ratio_col(
            d["S"], d["SV"], d["S_to_E"], d["SV_to_EV"], d["S_to_SV"])

    s_all = scenario_check_sums(db, SCENARIO_DB_NAME["baseline"])
    vax_den_all = s_all["SV0"][None, :] + s_all["S_to_SV"]
    unvax_den_all = s_all["S0"][None, :] - s_all["S_to_SV"]
    cols_inf["All"] = _rate_ratio_col(s_all["SV_to_EV"], vax_den_all, s_all["S_to_E"], unvax_den_all)
    cols_hosp["All"] = _rate_ratio_col(s_all["IV_to_H"], s_all["SV_to_EV"], s_all["I_to_H"], s_all["S_to_E"])

    d_all = scenario_daily_arrays(db, SCENARIO_DB_NAME["baseline"])
    cols_matched["All"] = _matched_cohort_ratio_col(
        d_all["S"], d_all["SV"], d_all["S_to_E"], d_all["SV_to_EV"], d_all["S_to_SV"])

    return {
        "infection_reduction": pd.DataFrame(cols_inf, index=AGE_GROUPS),
        "matched_cohort_infection_reduction": pd.DataFrame(cols_matched, index=AGE_GROUPS),
        "hospitalization_reduction": pd.DataFrame(cols_hosp, index=AGE_GROUPS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB),
                         help="results source written by run_simulations_MA_vax.py -- a "
                              "results_parquet/ directory (current default output) or a "
                              "SQLite results.db (older runs / the notebook's Analysis tab "
                              "SQLite export)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                         help="output folder for the CSVs counterfactual_notebook_generic.py reads "
                              "(a relative path is taken relative to the current working directory)")
    parser.add_argument("--model-config", default=str(MODEL_CONFIG_FILE),
                         help="model config JSON (population, params) -- for a run against a "
                              "differently-named/located config than this folder's default "
                              "model_config.json")
    parser.add_argument("--fitted-params", default=str(_HERE / cf.FITTED_PARAMS_FILE),
                         help="fitted params JSON merged over --model-config's baseline params for "
                              "Table S.A.4 (best point estimate) -- a path that doesn't exist is "
                              "silently skipped, same as run_simulations_MA_vax.py's own handling")
    args = parser.parse_args()

    # Resolved to absolute paths up front: load_population() and cf.load_base_inputs()
    # would otherwise interpret a relative path against two different base directories
    # (the current working directory vs. counterfactual_generic.py's own folder).
    args.model_config = os.path.abspath(args.model_config)
    args.fitted_params = os.path.abspath(args.fitted_params)

    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found -- run run_simulations_MA_vax.py first "
                          "(or pass --db pointing at its results_parquet/ or results.db).")

    population = load_population(args.model_config)
    db = ResultsDB(args.db)

    required = set(SCENARIO_DB_NAME.values()) | set(VE_TOTALS_DB_NAME.values())
    missing = sorted(required - db.scenarios_present())
    if missing:
        db.close()
        raise SystemExit(
            f"{args.db} is missing these scenarios -- add them to run_simulations_MA_vax.py's "
            f"SCENARIOS (and DOSE_MULTIPLIER, if they vaccinate) and re-run it:\n  " + "\n  ".join(missing)
        )

    os.makedirs(args.out, exist_ok=True)

    print("[1/7] Table S.A.1 (infection vs severity protection) ...")
    table_S_A_1(db, population, model_config_file=args.model_config).to_csv(
        os.path.join(args.out, "S_A_1.csv"))

    print("[2/7] Table S.A.2 (age group vaccinated) ...")
    for sub, df in table_S_A_2(db, population, model_config_file=args.model_config).items():
        df.to_csv(os.path.join(args.out, f"S_A_2_{sub}.csv"))

    for _pct in COVERAGE_TARGETS:
        print(f"[3/7] Table S.A.3 ({_pct}% coverage, single age group) ...")
        _tabs = table_S_A_3_for_target(db, population, _pct, model_config_file=args.model_config)
        for sub, df in _tabs.items():
            df.to_csv(os.path.join(args.out, f"S_A_3_{_pct}pct_{sub}.csv"))
            if _pct == 70:   # original Table S.A.3 file names (counterfactual notebook)
                df.to_csv(os.path.join(args.out, f"S_A_3_{sub}.csv"))

    print("[4/7] Table S.A.4 (VE sensitivity parameters) ...")
    cf.table_S_A_4(cf.load_base_inputs(
        model_config_file=args.model_config, fitted_params_file=args.fitted_params,
    )).to_csv(os.path.join(args.out, "S_A_4.csv"))

    print("[4b/7] Dose accounting (scheduled vs. delivered) ...")
    table_dose_accounting(db, population, model_config_file=args.model_config).to_csv(
        os.path.join(args.out, "DOSE_ACCOUNTING.csv"))

    print("[5/7] Table S.A.5 (VE sensitivity, vs no vaccine) ...")
    for sub, df in table_S_A_5(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_5_{sub}.csv"))

    for _d in COVERAGE_DECREASES:
        print(f"[6/7] Coverage decrease ({_d}% fewer doses, single age group) ...")
        for sub, df in table_decrease(db, population, _d, model_config_file=args.model_config).items():
            df.to_csv(os.path.join(args.out, f"DECREASE_{_d}pct_{sub}.csv"))

    print("[6b/7] Coverage summary (all-ages coverage scenarios) ...")
    table_coverage_summary(db, population, model_config_file=args.model_config).to_csv(
        os.path.join(args.out, "COVERAGE_SUMMARY.csv"), index=False)

    print("[7/7] Vaccine-efficacy mechanism check (flow-level ratios) ...")
    for sub, df in table_vax_efficacy_check(db).items():
        df.to_csv(os.path.join(args.out, f"VAX_CHECK_{sub}.csv"))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "source": "run_simulations_MA_vax.py -> results source (results_full table)",
            "db": os.path.abspath(args.db),
            "model_config_file": args.model_config,
            "fitted_params_file": args.fitted_params,
            "n_reps": db.n_reps(SCENARIO_DB_NAME["baseline"]),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    db.close()
    print(f"Tables saved to {args.out}/")


if __name__ == "__main__":
    main()
