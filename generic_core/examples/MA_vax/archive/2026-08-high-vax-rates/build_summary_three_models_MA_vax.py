"""Side-by-side summary of the three Aug-2026 high-vax-rate fits, per age group
and total:

  no_R0        no initial immunity   model_config_2026-08-high-vax-rates.json + ../fitted_params.json
  R20_SSV      20% initially in R, daily doses = proportion x (S + SV)
  R20_totpop   20% initially in R, daily doses = proportion x total population

Every quantity is read from each run's param-set simulation output (one
deterministic run per posterior draw) or from its posterior draws:

  1. doses delivered (S_to_SV, baseline)
  2. hospitalizations averted, split into infection protection only (no vax ->
     infection-protection-only) and hospitalization reduction (infection-
     protection-only -> baseline), plus the total (no vax -> baseline), as
     counts and as % of the no-vax burden
     (build_counterfactual_tables_from_db.table_S_A_1)
  3. hospitalizations (I_to_H + IV_to_H) in the baseline and no-vax scenarios
  4. attack rate (N - R(0) - S(T) - SV(T)) / N in the baseline and no-vax scenarios
  5. effective IHR: IHR_scale|age x I_to_H_prop and x IV_to_H_prop
  6. fitted beta_baseline, humidity_impact, seed_scale_E

Cells are median [95% interval] across draws.

Writes summary_three_models_MA_vax/{summary.md, summary.xlsx, summary_long.csv}.

    python generic_core/examples/MA_vax/archive/2026-08-high-vax-rates/build_summary_three_models_MA_vax.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MA = HERE.parent.parent
sys.path.insert(0, str(MA))
import build_counterfactual_tables_from_db as cfdb  # noqa: E402
from ma_vax_shared import AGE_GROUP_LABELS as AGES  # noqa: E402

OUT = HERE / "summary_three_models_MA_vax"
TAG = "MA_vax_old_vax_initial_R_20pct"

MODELS = {
    "No initial immunity": dict(
        config=HERE / "model_config_2026-08-high-vax-rates.json",
        fit=HERE.parent / "fitted_params.json",
        db=HERE / "simulation_output_param_set_stochastic_2026-08-high-vax-rates" / "results_parquet"),
    "20% R, doses of S+SV": dict(
        config=HERE / f"model_config_{TAG}.json",
        fit=HERE / f"fitted_params_{TAG}.json",
        db=HERE / f"simulation_output_{TAG}_param_set_stochastic" / "results_parquet"),
    "20% R, doses of total population": dict(
        config=HERE / f"model_config_{TAG}_vax_total_pop.json",
        fit=HERE / f"fitted_params_{TAG}_vax_total_pop.json",
        db=HERE / f"simulation_output_{TAG}_vax_total_pop_param_set_stochastic" / "results_parquet"),
}
ROWS = AGES + ["All"]
# Excel sheet names (max 31 characters), one per metric, in build order
SHEETS = ["Doses delivered", "Averted - infection (n)", "Averted - infection (pct)",
          "Averted - hosp reduction (n)", "Averted - hosp reduction (pct)",
          "Averted - total (n)", "Averted - total (pct)",
          "Hospitalizations - baseline", "Hospitalizations - no vax",
          "Attack rate - baseline", "Attack rate - no vax", "IHR - I_to_H", "IHR - IV_to_H"]


def band(x: np.ndarray) -> np.ndarray:
    """(3, ...) median, 2.5%, 97.5% along the draw axis."""
    return np.percentile(x, [50, 2.5, 97.5], axis=0)


def fmt(b: np.ndarray, kind: str) -> str:
    """b = (median, 2.5%, 97.5%) from band()."""
    med, lo, hi = (float(v) for v in b)
    if kind == "count":
        return f"{med:,.0f} [{lo:,.0f} – {hi:,.0f}]"
    if kind == "pct1":
        return f"{100*med:.1f}% [{100*lo:.1f}% – {100*hi:.1f}%]"
    if kind == "pct3":
        return f"{100*med:.3f}% [{100*lo:.3f}% – {100*hi:.3f}%]"
    return f"{med:.4g} [{lo:.4g} – {hi:.4g}]"


def by_age_and_total(x: np.ndarray, kind: str, total: np.ndarray | None = None) -> list[str]:
    """x: (draws, ages). `total` (draws,) overrides the summed All row."""
    tot = x.sum(axis=1) if total is None else total
    cols = [band(x[:, a]) for a in range(x.shape[1])] + [band(tot)]
    return [fmt(c, kind) for c in cols]


def summarize(label: str, spec: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    cfg = json.loads(spec["config"].read_text())
    fit = json.loads(spec["fit"].read_text())
    ic = cfg["initial_conditions"]["aggregate_pop"]
    pop = np.asarray(ic["population"], float).ravel()
    r0 = np.asarray(ic.get("seeds", {}).get("R", np.zeros_like(pop)), float).ravel()

    db = cfdb.ResultsDB(str(spec["db"]))
    try:
        s_a_1 = cfdb.table_S_A_1(db, pop, model_config_file=spec["config"])
        out: dict[str, list[str]] = {}
        base = db.arrays("baseline", ["I_to_H", "IV_to_H", "S_to_SV", "S", "SV"])
        novax = db.arrays("no vax", ["I_to_H", "IV_to_H", "S", "SV"])
        n_reps = db.n_reps("baseline")
    finally:
        db.close()

    out["Doses delivered (baseline)"] = by_age_and_total(base["S_to_SV"].sum(axis=1), "count")
    for key, name in (("reduced_infection", "infection protection only"),
                      ("reduced_severity", "hospitalization reduction"),
                      ("total", "total")):
        out[f"Hospitalizations averted, {name}"] = [
            s_a_1.loc[r, f"averted_{key}"].replace(" - ", " – ") for r in ROWS]
        out[f"% of no-vax hospitalizations averted, {name}"] = [
            s_a_1.loc[r, f"pct_averted_{key}"].replace(" - ", " – ") for r in ROWS]
    for scen, arr in (("baseline", base), ("no vax", novax)):
        out[f"Hospitalizations, {scen}"] = by_age_and_total(
            (arr["I_to_H"] + arr["IV_to_H"]).sum(axis=1), "count")
    for scen, arr in (("baseline", base), ("no vax", novax)):
        # people seeded into R never get infected this season
        inf = pop[None, :] - r0[None, :] - arr["S"][:, -1, :] - arr["SV"][:, -1, :]
        out[f"Attack rate, {scen}"] = by_age_and_total(
            inf / pop[None, :], "pct1", total=inf.sum(axis=1) / pop.sum())

    draws = fit["accepted_params"]
    scale = np.array([[float(q[f"IHR_scale|a{a}"]) for a in range(len(AGES))] for q in draws])
    for comp, name in (("I_to_H_prop", "IHR, unvaccinated (I_to_H)"),
                       ("IV_to_H_prop", "IHR, vaccinated (IV_to_H)")):
        ihr = scale * np.asarray(cfg["params"][comp], float).ravel()[None, :]
        out[name] = [fmt(band(ihr[:, a]), "pct3") for a in range(len(AGES))] + ["n/a"]

    params = {}
    for k in ("beta_baseline", "humidity_impact", "seed_scale_E"):
        v = np.array([float(q[k]) for q in draws])
        params[k] = f"{fmt(band(v), 'num')} (best {float(fit['best_params'][k]):.4g})"
    params["posterior draws / simulations per scenario"] = f"{len(draws)} / {n_reps}"
    print(f"{label}: {n_reps} replicates")
    return out, params


def to_md(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row]
        if idx == "All":
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    results, params = {}, {}
    for label, spec in MODELS.items():
        results[label], params[label] = summarize(label, spec)

    metrics = list(next(iter(results.values())))
    tables = {m: pd.DataFrame({lab: results[lab][m] for lab in MODELS},
                              index=pd.Index(ROWS, name="Age group")) for m in metrics}
    ptab = pd.DataFrame(params)
    ptab.index.name = "Parameter"

    long = pd.concat([t.reset_index().melt(id_vars="Age group", var_name="Model", value_name="Value")
                      .assign(Metric=m) for m, t in tables.items()], ignore_index=True)
    long = long[["Metric", "Model", "Age group", "Value"]]

    OUT.mkdir(exist_ok=True)
    long.to_csv(OUT / "summary_long.csv", index=False, encoding="utf-8-sig")
    ptab.to_csv(OUT / "fitted_params.csv", encoding="utf-8-sig")

    # one sheet per metric; the .xlsx stores age groups as text (a CSV opened
    # in Excel turns "1-4"/"5-12" into dates)
    with pd.ExcelWriter(OUT / "summary.xlsx", engine="openpyxl") as xw:
        assert len(SHEETS) == len(tables)
        for sheet, t in zip(SHEETS, tables.values()):
            t.to_excel(xw, sheet_name=sheet)
        ptab.to_excel(xw, sheet_name="Fitted parameters")
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value)) for c in col) + 2

    md = ["# Aug-2026 high-vax-rate fits: summary of the three models", "",
          "Median [95% interval] across posterior draws (one deterministic simulation per draw).",
          "", "- **No initial immunity**: the earlier high-vax fit (`model_config_2026-08-high-vax-rates.json`).",
          "- **20% R, doses of S+SV**: 20% of every age group starts in R; daily doses = scheduled proportion × (S + SV).",
          "- **20% R, doses of total population**: 20% starts in R; daily doses = scheduled proportion × total population, all given to S.",
          "", "Hospitalizations averted: *infection protection only* = no vaccination → infection-protection-only "
          "scenario; *hospitalization reduction* = infection-protection-only → full baseline; *total* = no vaccination → full "
          "baseline (the sum of the two). Percentages are of the no-vaccination hospitalizations. Attack rate = (N − R(0) − S(T) − SV(T)) / N, as a share of the "
          "whole age group (excludes the initially recovered). Effective IHR = fitted IHR_scale × I_to_H_prop "
          "(unvaccinated) or × IV_to_H_prop (vaccinated).", ""]
    md += ["## Fitted parameters", "", to_md(ptab), ""]
    for m, t in tables.items():
        md += [f"## {m}", "", to_md(t), ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
