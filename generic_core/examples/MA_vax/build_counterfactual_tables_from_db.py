"""Turn results.db (written by run_simulations_MA_vax.py) into the
counterfactual vaccination-impact tables that counterfactual_notebook_generic.py
loads via `cf.load_saved_tables`.

This is the missing link between two things already in this folder:

  run_simulations_MA_vax.py   -- an exported script (downloadable straight from
                                  the Model Builder notebook's Export tab) that
                                  runs every scenario in its SCENARIOS dict and
                                  writes per-replicate, per-subpop, per-age-group,
                                  per-risk-group daily compartment/transition
                                  history to simulation_output/results.db.
  counterfactual_notebook_generic.py -- displays Tables S.A.1-S.A.6 + the
                                  vaccine-efficacy check, but only ever reads
                                  already-computed CSVs from a results folder
                                  (via cf.load_saved_tables); it never runs a
                                  simulation for these tables itself.

Unlike counterfactual_generic.py's own `save_all_tables` (which reimplements
build_model/scenario-running against generic_core directly -- a second,
independently-defined simulation pipeline), this script computes the exact
same tables purely by reading results.db: the notebook-exported script is the
only thing that ever runs a simulation. The pure numeric helpers that turn
already-simulated (reps, day, age) arrays into table rows (`averted_summary`,
`_rate_ratio_col`, `_matched_cohort_ratio_col`, ...) are reused unchanged from
MA_vax.counterfactual, same as counterfactual_generic.py does -- they don't
care which pipeline produced the arrays.

Usage:
    cd generic_core/examples/MA_vax
    python run_simulations_MA_vax.py
    python build_counterfactual_tables_from_db.py

Requires a results.db with a `results_full` table (per-subpop/age-group/
risk-group history -- added alongside the population-summed `results` table
so per-age-group tables like S.A.2/S.A.3/S.A.5/S.A.6/VAX_CHECK are possible
at all). Re-run run_simulations_MA_vax.py if results.db predates that table.

Table S.A.6 additionally needs two scenarios beyond what the Model Builder
notebook's Analysis tab exports on its own -- "Low VE + 70% coverage (all
ages)" and "High VE + 70% coverage (all ages)", combining a VE preset with
70%-coverage dosing -- hand-added to SCENARIOS/DOSE_MULTIPLIER/
DESIGNED_PARAMS/DESIGNED_PARAM_RATIOS in run_simulations_MA_vax.py.

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
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE))

from MA_vax.model import AGE_GROUP_LABELS as AGE_GROUPS
from MA_vax.counterfactual import averted_summary, _rate_ratio_col, _matched_cohort_ratio_col

import counterfactual_generic as cf  # only for table_S_A_4, which is parameter-only (no simulation)

MODEL_CONFIG_FILE = _HERE / "model_config.json"
DEFAULT_DB = _HERE / "simulation_output" / "results.db"
DEFAULT_OUT = _HERE / "counterfactual_tables_from_db"

# Named-scenario -> results.db `scenario` column value. Matches the literal
# scenario names in run_simulations_MA_vax.py's SCENARIOS dict -- edit this
# map if scenarios are renamed there.
SCENARIO_DB_NAME = {
    "baseline": "baseline",
    "no vax": "no vax",
    "Infection protection only": "Infection protection only",
    **{f"Vaccinate {label} only": f"Vaccinate {label} only" for label in AGE_GROUPS},
    "70% coverage (all ages)": "70% coverage (all ages)",
    **{f"70% coverage ({label} only)": f"70% coverage ({label} only)" for label in AGE_GROUPS},
}

# VE-scenario name -> (own-baseline, own-70%-coverage) results.db scenario
# names, for Table S.A.6. "baseline_ve" reuses "baseline"/"70% coverage (all
# ages)" directly since VE_SCENARIOS' baseline_ve multipliers are all 1.0 --
# identical params to the fitted baseline.
VE_SCENARIO_DB_NAMES = {
    "low_ve": ("Low VE", "Low VE + 70% coverage (all ages)"),
    "baseline_ve": ("baseline", "70% coverage (all ages)"),
    "high_ve": ("High VE", "High VE + 70% coverage (all ages)"),
}
VE_TOTALS_DB_NAME = {"low_ve": "Low VE", "baseline_ve": "baseline", "high_ve": "High VE"}


def load_population() -> np.ndarray:
    with open(MODEL_CONFIG_FILE) as f:
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
        self._con = sqlite3.connect(db_path)
        _cols = {r[1] for r in self._con.execute("PRAGMA table_info(results_full)")}
        if not _cols:
            raise SystemExit(
                f"{db_path} has no `results_full` table -- it was written by an "
                "older version of run_simulations_MA_vax.py (before per-age-group/"
                "subpop/risk-group history was saved alongside the population-summed "
                "totals). Re-run run_simulations_MA_vax.py to regenerate results.db."
            )
        # results_full can be hundreds of millions of rows for a parameter-
        # uncertainty run; without an index every (scenario, compartment)
        # lookup is a full table scan. Older results.db files predate the
        # index run_simulations_MA_vax*.py now creates, so add it here too
        # (idempotent -- a no-op once it already exists).
        _idx = {r[0] for r in self._con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        if "idx_results_full_scenario_compartment" not in _idx:
            self._con.execute(
                "CREATE INDEX idx_results_full_scenario_compartment ON results_full (scenario, compartment)"
            )
            self._con.commit()
        # The table builders re-query the same scenario's arrays several
        # times across different tables (e.g. "baseline" is used by S.A.1,
        # S.A.2, S.A.3, S.A.5, S.A.6, and VAX_CHECK) -- cache per (scenario,
        # names) so each combination only hits the DB once.
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def scenarios_present(self) -> set[str]:
        return {r[0] for r in self._con.execute("SELECT DISTINCT scenario FROM results_full")}

    def n_reps(self, scenario: str) -> int:
        return self._con.execute(
            "SELECT COUNT(DISTINCT rep) FROM results_full WHERE scenario = ?", (scenario,)
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
            df = pd.read_sql_query(
                f"SELECT rep, compartment, age_group, day, SUM(value) AS value FROM results_full "
                f"WHERE scenario = ? AND compartment IN ({placeholders}) "
                f"GROUP BY rep, compartment, age_group, day",
                self._con, params=[scenario, *missing],
            )
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

def table_S_A_1(db: ResultsDB, population: np.ndarray) -> pd.DataFrame:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    inf_only = scenario_totals(db, SCENARIO_DB_NAME["Infection protection only"], population)
    full = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    reduced_infection = averted_summary(no_vax, inf_only).add_suffix("_reduced_infection")
    reduced_severity = averted_summary(inf_only, full, pct_reference=no_vax).add_suffix("_reduced_severity")
    total = averted_summary(no_vax, full).add_suffix("_total")
    return reduced_infection.join(reduced_severity).join(total)


def table_S_A_2(db: ResultsDB, population: np.ndarray) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    cols = {}
    for label in AGE_GROUPS:
        scen = scenario_totals(db, SCENARIO_DB_NAME[f"Vaccinate {label} only"], population)
        cols[label] = averted_summary(no_vax, scen)
    cols["All"] = averted_summary(no_vax, scenario_totals(db, SCENARIO_DB_NAME["baseline"], population))
    return {
        "pct_reduction": pd.DataFrame({l: df["pct_averted"] for l, df in cols.items()}),
        "per_100k": pd.DataFrame({l: df["per100k_averted"] for l, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({l: df["per100k_doses_averted"] for l, df in cols.items()}),
    }


def table_S_A_3(db: ResultsDB, population: np.ndarray) -> dict[str, pd.DataFrame]:
    baseline = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    cols = {}
    for label in AGE_GROUPS:
        scen = scenario_totals(db, SCENARIO_DB_NAME[f"70% coverage ({label} only)"], population)
        cols[label] = averted_summary(baseline, scen)
    all_scen = scenario_totals(db, SCENARIO_DB_NAME["70% coverage (all ages)"], population)
    cols["All"] = averted_summary(baseline, all_scen)
    return {
        "pct_reduction": pd.DataFrame({l: df["pct_averted"] for l, df in cols.items()}),
        "per_100k": pd.DataFrame({l: df["per100k_averted"] for l, df in cols.items()}),
        "per_100k_doses": pd.DataFrame({l: df["per100k_doses_averted"] for l, df in cols.items()}),
    }


def table_S_A_5(db: ResultsDB, population: np.ndarray) -> dict[str, pd.DataFrame]:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    cols = {
        name: averted_summary(no_vax, scenario_totals(db, db_name, population))
        for name, db_name in VE_TOTALS_DB_NAME.items()
    }
    return {
        "pct_reduction": pd.DataFrame({n: df["pct_averted"] for n, df in cols.items()}),
        "per_100k": pd.DataFrame({n: df["per100k_averted"] for n, df in cols.items()}),
    }


def table_S_A_6(db: ResultsDB, population: np.ndarray) -> dict[str, pd.DataFrame]:
    cols = {}
    for name, (base_name, target_name) in VE_SCENARIO_DB_NAMES.items():
        baseline = scenario_totals(db, base_name, population)
        target70 = scenario_totals(db, target_name, population)
        cols[name] = averted_summary(baseline, target70)
    return {
        "pct_reduction": pd.DataFrame({n: df["pct_averted"] for n, df in cols.items()}),
        "per_100k": pd.DataFrame({n: df["per100k_averted"] for n, df in cols.items()}),
    }


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
                         help="results.db written by run_simulations_MA_vax.py")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                         help="output folder for the CSVs counterfactual_notebook_generic.py reads")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found -- run run_simulations_MA_vax.py first.")

    population = load_population()
    db = ResultsDB(args.db)

    required = set(SCENARIO_DB_NAME.values()) | {
        n for pair in VE_SCENARIO_DB_NAMES.values() for n in pair
    }
    missing = sorted(required - db.scenarios_present())
    if missing:
        db.close()
        raise SystemExit(
            "results.db is missing these scenarios -- add them to run_simulations_MA_vax.py's "
            f"SCENARIOS (and DOSE_MULTIPLIER, if they vaccinate) and re-run it:\n  " + "\n  ".join(missing)
        )

    os.makedirs(args.out, exist_ok=True)

    print("[1/7] Table S.A.1 (infection vs severity protection) ...")
    table_S_A_1(db, population).to_csv(os.path.join(args.out, "S_A_1.csv"))

    print("[2/7] Table S.A.2 (age group vaccinated) ...")
    for sub, df in table_S_A_2(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_2_{sub}.csv"))

    print("[3/7] Table S.A.3 (70% coverage, single age group) ...")
    for sub, df in table_S_A_3(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_3_{sub}.csv"))

    print("[4/7] Table S.A.4 (VE sensitivity parameters) ...")
    cf.table_S_A_4(cf.load_base_inputs()).to_csv(os.path.join(args.out, "S_A_4.csv"))

    print("[5/7] Table S.A.5 (VE sensitivity, vs no vaccine) ...")
    for sub, df in table_S_A_5(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_5_{sub}.csv"))

    print("[6/7] Table S.A.6 (VE sensitivity, 70% coverage) ...")
    for sub, df in table_S_A_6(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_6_{sub}.csv"))

    print("[7/7] Vaccine-efficacy mechanism check (flow-level ratios) ...")
    for sub, df in table_vax_efficacy_check(db).items():
        df.to_csv(os.path.join(args.out, f"VAX_CHECK_{sub}.csv"))

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "source": "run_simulations_MA_vax.py -> results.db (results_by_age table)",
            "db": os.path.abspath(args.db),
            "model_config_file": "model_config.json",
            "fitted_params_file": "fitted_params.json",
            "n_reps": db.n_reps(SCENARIO_DB_NAME["baseline"]),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    db.close()
    print(f"Tables saved to {args.out}/")


if __name__ == "__main__":
    main()
