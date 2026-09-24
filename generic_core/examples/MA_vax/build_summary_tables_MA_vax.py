"""Build the two presentation summary tables from a run_simulations_MA_vax*.py
results source:

  1. Hospitalizations averted, by measure and age group: "Infection protection
     only" (no vax -> infection-protection-only), "Hospitalization protection
     only" (infection-protection-only -> baseline) and "Total" (no vax ->
     baseline). Same numbers as Table S.A.1 (build_counterfactual_tables_from_db.py's
     table_S_A_1), reshaped to long format.
  2. Cumulative hospitalizations by scenario (baseline, Infection protection
     only, no vax) and age group.

All cells are median [95% interval] across replicates.

Usage:
    cd generic_core/examples/MA_vax
    python build_summary_tables_MA_vax.py \
        --db simulation_output_MA_vax_param_set_stochastic/results_parquet \
        --out summary_tables_MA_vax
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import build_counterfactual_tables_from_db as cfdb
from ma_vax_shared import AGE_GROUP_LABELS, _summ

_HERE = os.path.dirname(os.path.abspath(__file__))

MEASURES = {
    "reduced_infection": "Infection protection only",
    "reduced_severity": "Hospitalization protection only",
    "total": "Total",
}
CUMULATIVE_SCENARIOS = ["baseline", "Infection protection only", "no vax"]


def _range_dash(s) -> str:
    return str(s).replace(" - ", " – ")


def _to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def hospitalizations_averted_table(db, population, model_config_file) -> pd.DataFrame:
    s_a_1 = cfdb.table_S_A_1(db, population, model_config_file=model_config_file)
    rows = []
    for key, label in MEASURES.items():
        for age_group, r in s_a_1.iterrows():
            rows.append({
                "Measure": label,
                "Age group": age_group,
                "Hospitalizations Averted": _range_dash(r[f"averted_{key}"]),
                "% Hospitalizations Averted": _range_dash(r[f"pct_averted_{key}"]),
            })
    return pd.DataFrame(rows)


def cumulative_hospitalizations_table(db, population) -> pd.DataFrame:
    rows = []
    for scenario in CUMULATIVE_SCENARIOS:
        new_H = cfdb.scenario_totals(db, cfdb.SCENARIO_DB_NAME[scenario], population)["new_H"]
        cols = [new_H[:, a] for a in range(new_H.shape[1])] + [new_H.sum(axis=1)]
        for label, values in zip(AGE_GROUP_LABELS + ["All"], cols):
            m, lo, hi = _summ(values)
            rows.append({
                "Scenario": scenario,
                "Age Group": label,
                "Hospitalizations": f"{m:,.0f} [{lo:,.0f} – {hi:,.0f}]",
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.path.join(
        _HERE, "simulation_output_MA_vax_param_set_stochastic", "results_parquet"))
    parser.add_argument("--model-config", default=os.path.join(_HERE, "model_config_MA_vax.json"))
    parser.add_argument("--out", default=os.path.join(_HERE, "summary_tables_MA_vax"))
    args = parser.parse_args()
    args.model_config = os.path.abspath(args.model_config)

    population = cfdb.load_population(args.model_config)
    db = cfdb.ResultsDB(args.db)
    try:
        averted = hospitalizations_averted_table(db, population, args.model_config)
        cumulative = cumulative_hospitalizations_table(db, population)
        n_reps = db.n_reps("baseline")
    finally:
        db.close()

    os.makedirs(args.out, exist_ok=True)
    # utf-8-sig: the BOM makes Excel read the en dashes as UTF-8 instead of Mac Roman.
    averted.to_csv(os.path.join(args.out, "hospitalizations_averted.csv"), index=False,
                   encoding="utf-8-sig")
    cumulative.to_csv(os.path.join(args.out, "cumulative_hospitalizations.csv"), index=False,
                      encoding="utf-8-sig")
    # Excel auto-converts CSV age groups like "1-4"/"5-12" to dates on open; the
    # .xlsx stores every cell as text, so open that one in Excel instead.
    with pd.ExcelWriter(os.path.join(args.out, "summary_tables.xlsx"), engine="openpyxl") as xw:
        averted.to_excel(xw, sheet_name="Hospitalizations averted", index=False)
        cumulative.to_excel(xw, sheet_name="Cumulative hospitalizations", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(
                    len(str(c.value)) for c in col) + 2
    with open(os.path.join(args.out, "summary_tables.md"), "w") as f:
        f.write(f"Source: `{os.path.relpath(args.db, _HERE)}` ({n_reps} replicates). "
                "Median [95% interval] across replicates.\n\n")
        f.write("## Hospitalizations averted\n\n" + _to_markdown(averted) + "\n\n")
        f.write("## Cumulative hospitalizations\n\n" + _to_markdown(cumulative) + "\n")

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(averted.to_string(index=False), "\n")
        print(cumulative.to_string(index=False))
    print(f"\nWrote tables to {args.out}")


if __name__ == "__main__":
    main()
