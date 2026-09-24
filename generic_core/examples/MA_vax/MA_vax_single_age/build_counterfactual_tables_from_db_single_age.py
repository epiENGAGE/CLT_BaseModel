"""Single-age-group adaptation of
`MA_vax/build_counterfactual_tables_from_db.py`: turns a results source
(written by `run_simulations_MA_vax_single_age.py`) into counterfactual
vaccination-impact tables, reading only `results_full` -- no simulation of
its own is run here.

Differs from the multi-age version in what it can compute, not just in
which files it points at:

  * `run_simulations_MA_vax_single_age.py`'s own `SCENARIOS` dict omits every
    age-targeted scenario ("Vaccinate <age> only", "70% coverage (<age>
    only)") -- see the comment above it -- because with one age group they
    have no single-age equivalent. Tables S.A.2 and S.A.3 (which are
    *defined* by that per-age breakdown) are therefore not built here at all.
  * Table S.A.4's VE-preset overrides (`vax_susceptibility`/`IV_to_H_prop`)
    are copied from that same `SCENARIOS` dict below (`VE_PRESET_OVERRIDES`)
    rather than recomputed from ratios -- that script already carries the
    single-age-refit absolute values as its source of truth.
  * The vaccine-efficacy mechanism check only has one column ("All"), since
    there is no per-age breakdown to compare it against.

Everything else (S.A.1, S.A.5, S.A.6, dose accounting) mirrors the multi-age
script exactly, generalized to whatever `AGE_GROUPS` this model config says
(currently one group, but nothing here hardcodes that count).

Usage:
    cd generic_core/examples/MA_vax/MA_vax_single_age
    python run_simulations_MA_vax_single_age.py
    python build_counterfactual_tables_from_db_single_age.py

Location: this file must sit next to model_config_MA_single_age.json /
fitted_params_MA_single_age.json / schedules.json, three directory levels
below the repo root that contains generic_core/, clt_toolkit/ and flu_core/
-- same constraint as run_simulations_MA_vax_single_age.py in this folder.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))  # repo root
sys.path.insert(0, str(_HERE.parent))  # MA_vax/, for ma_vax_shared
sys.path.insert(0, str(_HERE))

from ma_vax_shared import averted_summary, _rate_ratio_col, _matched_cohort_ratio_col
from generic_core import results_io
from generic_core.model_factory import load_schedule_csv_texts
from generic_core.fitting import prepare_param_sets

MODEL_CONFIG_FILE = _HERE / "model_config_MA_single_age.json"
FITTED_PARAMS_FILE = _HERE / "fitted_params_MA_single_age.json"
SCHEDULES_FILE = _HERE / "schedules.json"
DEFAULT_DB = _HERE / "simulation_output_single_age" / "results_parquet"
DEFAULT_OUT = _HERE / "counterfactual_tables_from_db_single_age"

START_DATE = "2025-09-01"
NUM_DAYS = 250

with open(MODEL_CONFIG_FILE) as _f:
    _model_config = json.load(_f)
AGE_GROUPS: list[str] = list((_model_config.get("age_risk", {}) or {}).get("age_groups") or ["0+"])

# Named-scenario -> results source `scenario` column value. Matches the
# literal scenario names in run_simulations_MA_vax_single_age.py's SCENARIOS
# dict -- edit this map if scenarios are renamed there. No per-age-group
# entries: see module docstring.
SCENARIO_DB_NAME = {
    "baseline": "baseline",
    "no vax": "no vax",
    "Infection protection only": "Infection protection only",
    "Low VE": "Low VE",
    "High VE": "High VE",
    "70% coverage (all ages)": "70% coverage (all ages)",
}

# VE-scenario name -> (own-baseline, own-70%-coverage) results source scenario
# names, for Table S.A.6. "baseline_ve" reuses "baseline"/"70% coverage (all
# ages)" directly since it's a 1.0-multiplier preset -- identical params to
# the fitted baseline.
VE_SCENARIO_DB_NAMES = {
    "low_ve": ("Low VE", "Low VE + 70% coverage (all ages)"),
    "baseline_ve": ("baseline", "70% coverage (all ages)"),
    "high_ve": ("High VE", "High VE + 70% coverage (all ages)"),
}
VE_TOTALS_DB_NAME = {"low_ve": "Low VE", "baseline_ve": "baseline", "high_ve": "High VE"}

# Absolute vax_susceptibility/IV_to_H_prop overrides for the VE presets,
# copied from run_simulations_MA_vax_single_age.py's SCENARIOS dict (source
# of truth -- keep in sync if that script is re-fit or the presets change).
VE_PRESET_OVERRIDES = {
    "Low VE": {"vax_susceptibility": 0.8976501514033461, "IV_to_H_prop": 0.005687597232161169},
    "High VE": {"vax_susceptibility": 0.6616378415241131, "IV_to_H_prop": 0.0049138373017620935},
}


def reload_model_config(path) -> None:
    """Point `_model_config`/`AGE_GROUPS` at a different model config file --
    for a run against a differently-named/located config than this folder's
    default `MODEL_CONFIG_FILE`. Unlike the multi-age
    `build_counterfactual_tables_from_db.py` (whose `AGE_GROUPS` is a fixed
    7-label constant, independent of which config file is loaded), this
    model's age groups are themselves read from the config -- so switching
    files has to refresh both. Must be called (if at all) before any other
    function in this module runs, since everything downstream (`ResultsDB`,
    the table builders, `load_population`, `load_base_params`, ...) reads
    `_model_config`/`AGE_GROUPS` at call time, not at import time."""
    global _model_config, AGE_GROUPS
    with open(path) as f:
        _model_config = json.load(f)
    AGE_GROUPS = list((_model_config.get("age_risk", {}) or {}).get("age_groups") or ["0+"])


def load_population() -> np.ndarray:
    ic_entry = (_model_config.get("initial_conditions", {}) or {}).get("aggregate_pop", {})
    return np.asarray(
        ic_entry.get("population", [[0.0]] * len(AGE_GROUPS)), dtype=float
    ).reshape(len(AGE_GROUPS))


def load_base_params(fitted_params_file=FITTED_PARAMS_FILE) -> dict:
    """Fitted best-point params merged over `_model_config`'s baseline, same
    merge `run_simulations_MA_vax_single_age.py` does -- just the static
    param dict (no seed scales / m(t) reconstruction), since that's all
    Table S.A.4 needs."""
    config_dict = json.loads(json.dumps(_model_config))  # cheap deep copy
    fitted_params_file = Path(fitted_params_file)
    if fitted_params_file.exists():
        with open(fitted_params_file) as f:
            fitted_raw = json.load(f)
        best_params = fitted_raw.get("best_params", fitted_raw) if isinstance(fitted_raw, dict) else {}
        scale_groups = (fitted_raw.get("scale_groups", {}) or {}) if isinstance(fitted_raw, dict) else {}
        orig_params = dict(config_dict.get("params", {}) or {})
        expanded = prepare_param_sets([best_params], scale_groups, orig_params)[0]
        fitted_params = {
            k: v for k, v in expanded.items()
            if k != "phi" and not k.startswith("m_dlog_") and not k.startswith("seed_scale_")
        }
        config_dict["params"] = {**config_dict.get("params", {}), **fitted_params}
    return config_dict["params"]


# ── Scheduled doses (mirrors counterfactual_generic.scheduled_doses) ───────

_SCHEDULE_CSVS_CACHE: dict | None = None


def _load_schedule_csvs() -> dict:
    global _SCHEDULE_CSVS_CACHE
    if _SCHEDULE_CSVS_CACHE is None:
        _SCHEDULE_CSVS_CACHE = load_schedule_csv_texts(
            _model_config,
            snapshot_path=SCHEDULES_FILE if SCHEDULES_FILE.exists() else None,
            search_roots=[Path.cwd(), _HERE, *_HERE.parents],
        )
    return _SCHEDULE_CSVS_CACHE


def scheduled_coverage() -> np.ndarray:
    """Cumulative coverage the BASELINE dose schedule asks for over the
    simulation window, per age group -- see
    counterfactual_generic.scheduled_coverage's docstring for why this (not
    realized S_to_SV) is the right per-100k-doses denominator."""
    csvs = _load_schedule_csvs()
    if "daily_vaccines_df" not in csvs:
        raise ValueError("no daily_vaccines_df schedule to derive coverage from")
    df = pd.read_csv(io.StringIO(csvs["daily_vaccines_df"]))
    dates = pd.to_datetime(df["date"], format="mixed")
    start = pd.Timestamp(START_DATE)
    window = (dates >= start) & (dates < start + pd.Timedelta(days=NUM_DAYS))
    daily = np.array([np.asarray(json.loads(v), dtype=float).ravel()
                       for v in df.loc[window, "daily_vaccines"]])
    return daily.sum(axis=0)


def scheduled_doses(population) -> np.ndarray:
    return scheduled_coverage() * np.asarray(population, dtype=float)


# ── Results source reader (identical to the multi-age version) ─────────────

class ResultsDB:
    """Age-resolved (reps, day, age_group) arrays for a scenario, read from
    the `results_full` table run_simulations_MA_vax_single_age.py writes --
    summed over subpop and risk group (via SQL GROUP BY), since every table
    this script builds wants population totals by age group, not by
    subpop/risk group. Single-population, single-risk-group, single-age-group
    models (MA_vax_single_age) have nothing to sum away there, so this is a
    no-op for them."""

    def __init__(self, db_path: str):
        self.path = db_path
        try:
            self._con = results_io.load_source(db_path)
        except results_io.ResultsExplorerError as exc:
            raise SystemExit(str(exc)) from exc
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def scenarios_present(self) -> set[str]:
        return {r[0] for r in self._con.execute("SELECT DISTINCT scenario FROM results_full").fetchall()}

    def n_reps(self, scenario: str) -> int:
        return self._con.execute(
            "SELECT COUNT(DISTINCT rep) FROM results_full WHERE scenario = ?", [scenario]
        ).fetchone()[0]

    def arrays(self, scenario: str, names: list[str]) -> dict[str, np.ndarray]:
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
                    "in run_simulations_MA_vax_single_age.py's SCENARIOS and the script has been re-run."
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
                        "in run_simulations_MA_vax_single_age.py includes it."
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


# ── Table builders ──────────────────────────────────────────────────────────
# S.A.2 and S.A.3 are not built here -- see module docstring.

def _sched_doses(totals: dict, doses) -> dict:
    return {**totals, "doses": np.asarray(doses, dtype=float)}


def table_S_A_1(db: ResultsDB, population: np.ndarray) -> pd.DataFrame:
    no_vax = scenario_totals(db, SCENARIO_DB_NAME["no vax"], population)
    inf_only = scenario_totals(db, SCENARIO_DB_NAME["Infection protection only"], population)
    full = scenario_totals(db, SCENARIO_DB_NAME["baseline"], population)
    sched = scheduled_doses(population)
    no_vax = _sched_doses(no_vax, np.zeros_like(sched))
    inf_only, full = _sched_doses(inf_only, sched), _sched_doses(full, sched)
    reduced_infection = averted_summary(no_vax, inf_only).add_suffix("_reduced_infection")
    reduced_severity = averted_summary(inf_only, full, pct_reference=no_vax,
                                        doses_reference=no_vax).add_suffix("_reduced_severity")
    total = averted_summary(no_vax, full).add_suffix("_total")
    return reduced_infection.join(reduced_severity).join(total)


def table_dose_accounting(db: ResultsDB, population: np.ndarray) -> pd.DataFrame:
    """Scheduled vs. delivered doses under the baseline schedule, by age
    group -- see `build_counterfactual_tables_from_db.table_dose_accounting`'s
    docstring for the full explanation (identical here, just over
    `AGE_GROUPS` as this model config defines it)."""
    sched = scheduled_doses(population)
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


def table_S_A_4(params: dict) -> pd.DataFrame:
    ih = np.asarray(params["I_to_H_prop"], dtype=float).flatten()[0]
    i_rel_inf = float(params["I_relative_infectiousness"])
    iv_rel_inf = float(params["IV_relative_infectiousness"])
    ve_transmission_blocking = 1.0 - iv_rel_inf / i_rel_inf
    presets = {
        "low_ve": VE_PRESET_OVERRIDES["Low VE"],
        "baseline_ve": {"vax_susceptibility": float(np.asarray(params["vax_susceptibility"]).flatten()[0]),
                         "IV_to_H_prop": ih},
        "high_ve": VE_PRESET_OVERRIDES["High VE"],
    }
    rows = []
    for name, ov in presets.items():
        vs, ivh = ov["vax_susceptibility"], ov["IV_to_H_prop"]
        ve_inf = 1.0 - vs
        ve_hosp_given_inf = 1.0 - ivh / ih
        ve_hosp_inf = 1 - (1 - ve_inf) * (1 - ve_hosp_given_inf)
        rows.append({
            "scenario": name,
            "age_group": AGE_GROUPS[0],
            "VE_infection": f"{ve_inf * 100:.0f}%",
            "VE_hosp_infection": f"{ve_hosp_inf * 100:.0f}%",
            "VE_hosp_given_infection": f"{ve_hosp_given_inf * 100:.0f}%",
            "VE_transmission_blocking": f"{ve_transmission_blocking * 100:.0f}%",
        })
    return pd.DataFrame(rows).set_index(["scenario", "age_group"])


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
    """Same flow-level VE check as the multi-age version, but with a single
    "All" column -- there is no per-age breakdown to compare it against
    (see module docstring)."""
    s_all = scenario_check_sums(db, SCENARIO_DB_NAME["baseline"])
    vax_den_all = s_all["SV0"][None, :] + s_all["S_to_SV"]
    unvax_den_all = s_all["S0"][None, :] - s_all["S_to_SV"]
    cols_inf = {"All": _rate_ratio_col(s_all["SV_to_EV"], vax_den_all, s_all["S_to_E"], unvax_den_all)}
    cols_hosp = {"All": _rate_ratio_col(s_all["IV_to_H"], s_all["SV_to_EV"], s_all["I_to_H"], s_all["S_to_E"])}

    d_all = scenario_daily_arrays(db, SCENARIO_DB_NAME["baseline"])
    cols_matched = {"All": _matched_cohort_ratio_col(
        d_all["S"], d_all["SV"], d_all["S_to_E"], d_all["SV_to_EV"], d_all["S_to_SV"])}

    return {
        "infection_reduction": pd.DataFrame(cols_inf, index=AGE_GROUPS),
        "matched_cohort_infection_reduction": pd.DataFrame(cols_matched, index=AGE_GROUPS),
        "hospitalization_reduction": pd.DataFrame(cols_hosp, index=AGE_GROUPS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB),
                         help="results source written by run_simulations_MA_vax_single_age.py -- a "
                              "results_parquet/ directory (current default output) or a SQLite "
                              "results.db")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                         help="output folder for the CSVs (a relative path is taken relative to "
                              "the current working directory)")
    parser.add_argument("--model-config", default=str(MODEL_CONFIG_FILE),
                         help="model config JSON (population, params, age groups) -- for a run "
                              "against a differently-named/located config than this folder's "
                              "default model_config_MA_single_age.json")
    parser.add_argument("--fitted-params", default=str(FITTED_PARAMS_FILE),
                         help="fitted params JSON merged over --model-config's baseline params for "
                              "Table S.A.4 (best point estimate) -- a path that doesn't exist is "
                              "silently skipped, same as run_simulations_MA_vax_single_age.py's own "
                              "handling")
    args = parser.parse_args()

    # Resolved to absolute paths up front, same reasoning as the multi-age script's
    # --model-config/--fitted-params handling.
    args.model_config = os.path.abspath(args.model_config)
    args.fitted_params = os.path.abspath(args.fitted_params)
    reload_model_config(args.model_config)

    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found -- run run_simulations_MA_vax_single_age.py first "
                          "(or pass --db pointing at its results_parquet/ or results.db).")

    population = load_population()
    db = ResultsDB(args.db)

    required = set(SCENARIO_DB_NAME.values()) | {
        n for pair in VE_SCENARIO_DB_NAMES.values() for n in pair
    }
    missing = sorted(required - db.scenarios_present())
    if missing:
        db.close()
        raise SystemExit(
            f"{args.db} is missing these scenarios -- add them to run_simulations_MA_vax_single_age.py's "
            f"SCENARIOS (and DOSE_MULTIPLIER, if they vaccinate) and re-run it:\n  " + "\n  ".join(missing)
        )

    os.makedirs(args.out, exist_ok=True)

    print("[1/6] Table S.A.1 (infection vs severity protection) ...")
    table_S_A_1(db, population).to_csv(os.path.join(args.out, "S_A_1.csv"))

    print("[2/6] Table S.A.4 (VE sensitivity parameters) ...")
    table_S_A_4(load_base_params(args.fitted_params)).to_csv(os.path.join(args.out, "S_A_4.csv"))

    print("[2b/6] Dose accounting (scheduled vs. delivered) ...")
    table_dose_accounting(db, population).to_csv(os.path.join(args.out, "DOSE_ACCOUNTING.csv"))

    print("[3/6] Table S.A.5 (VE sensitivity, vs no vaccine) ...")
    for sub, df in table_S_A_5(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_5_{sub}.csv"))

    print("[4/6] Table S.A.6 (VE sensitivity, 70% coverage) ...")
    for sub, df in table_S_A_6(db, population).items():
        df.to_csv(os.path.join(args.out, f"S_A_6_{sub}.csv"))

    print("[5/6] Vaccine-efficacy mechanism check (flow-level ratios) ...")
    for sub, df in table_vax_efficacy_check(db).items():
        df.to_csv(os.path.join(args.out, f"VAX_CHECK_{sub}.csv"))

    print("[6/6] Writing meta.json ...")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({
            "source": "run_simulations_MA_vax_single_age.py -> results source (results_full table)",
            "db": os.path.abspath(args.db),
            "model_config_file": args.model_config,
            "fitted_params_file": args.fitted_params,
            "age_groups": AGE_GROUPS,
            "n_reps": db.n_reps(SCENARIO_DB_NAME["baseline"]),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "omitted_tables": ["S_A_2", "S_A_3"],
            "omitted_reason": "no per-age-group scenarios in a single-age-group model",
        }, f, indent=2)

    db.close()
    print(f"Tables saved to {args.out}/")


if __name__ == "__main__":
    main()
