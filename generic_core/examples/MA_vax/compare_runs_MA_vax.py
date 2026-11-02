"""Compare two (or more) MA_vax simulation runs side by side: the fit, the
attack rates by vaccination status and the hospitalizations averted.

Written to test whether the IHR_scale upper bound (2.0 in the main fit) drives
the high attack rate: the IHR_ub_5 run refits with the bound raised to 5.0 and
simulates a subset of scenarios. Each run is
`label=results_dir:fitted_params[:model_config]` (model config defaults to
model_config_MA_vax.json):

    python compare_runs_MA_vax.py                       # main run vs IHR_ub_5
    python compare_runs_MA_vax.py --runs \\
        "IHR max 2=simulation_output_MA_vax_param_set_stochastic/results_parquet:fitted_params_MA_vax.json" \\
        "IHR max 5=simulation_output_MA_vax_IHR_ub_5/results_parquet:fitted_params_MA_vax_IHR_ub_5.json"

Writes to --out (default compare_IHR_ub_5/):
  fit_params.csv      posterior mean (95% interval) of each fitted parameter, the
                      share of draws near the upper bound, and fitted IHR by age
  attack_rates.csv    baseline attack rate by age group: vaccinated, unvaccinated,
                      total, median (95% interval) across draws
  symptomatic.csv     baseline symptomatic / asymptomatic attack rate by age group and
                      vaccination status (runs whose model has an A compartment)
  hospitalizations.csv  total hospitalizations per scenario
  averted.csv         total-population hospitalizations averted: baseline vs no vax,
                      and 50/60/70% all-ages coverage vs baseline
  comparison.md       the three tables as Markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "figures"))

import counterfactual_generic as cf
from build_counterfactual_tables_from_db import (
    MODEL_CONFIG_FILE, ResultsDB, load_population, scenario_totals,
)
from build_figures import attack_rates
from ma_vax_shared import AGE_GROUP_LABELS as AGE_GROUPS, _summ

DEFAULT_RUNS = [
    "IHR multiplier max 2=simulation_output_MA_vax_param_set_stochastic/results_parquet:fitted_params_MA_vax.json",
    "IHR multiplier max 5=simulation_output_MA_vax_IHR_ub_5/results_parquet:fitted_params_MA_vax_IHR_ub_5.json",
]
COVERAGE_TARGETS = [50, 60, 70]
NEAR_BOUND = 0.95   # a draw is "near the upper bound" above this share of it


def _cell(x, fmt="{:,.0f}", pct=False) -> str:
    m, lo, hi = _summ(np.asarray(x, dtype=float))
    s = "%" if pct else ""
    return f"{fmt.format(m)}{s} ({fmt.format(lo)}–{fmt.format(hi)}{s})"


def _mean_ci(x, fmt="{:.2f}") -> str:
    x = np.asarray(x, dtype=float)
    return f"{fmt.format(x.mean())} ({fmt.format(np.percentile(x, 2.5))}–{fmt.format(np.percentile(x, 97.5))})"


SCENARIOS = ["no vax", "baseline"] + [f"{t}% coverage (all ages)" for t in COVERAGE_TARGETS]


def fit_params(fitted_file: Path, model_config: Path) -> pd.Series:
    """Posterior mean (95% interval) of the scalar fit params and each age
    group's IHR multiplier, with the fitted IHR (pre-fit x multiplier) -- per
    infection, or per symptomatic infection in the asymptomatic model."""
    fit = json.loads(fitted_file.read_text())
    draws = pd.DataFrame(fit["accepted_params"])
    bounds = fit["fit_config"]["bounds"]
    pre = np.asarray(json.loads(model_config.read_text())["params"]["I_to_H_prop"],
                     dtype=float).ravel()
    out = {"posterior draws": f"{len(draws)}"}
    for p in ("beta_baseline", "humidity_impact", "seed_scale_E"):
        out[p] = _mean_ci(draws[p], "{:.3g}")
    ub = bounds["IHR_scale"][1]
    out["IHR multiplier bounds"] = f"{bounds['IHR_scale'][0]:g}–{ub:g}"
    for i, a in enumerate(AGE_GROUPS):
        x = draws[f"IHR_scale|a{i}"]
        out[f"IHR multiplier {a}"] = _mean_ci(x)
        out[f"  near upper bound {a}"] = f"{(x > NEAR_BOUND * ub).mean() * 100:.0f}%"
    for i, a in enumerate(AGE_GROUPS):
        out[f"fitted IHR {a}"] = _mean_ci(draws[f"IHR_scale|a{i}"] * pre[i] * 100, "{:.3f}") + " %"
    return pd.Series(out)


def symptomatic(con, population: np.ndarray, scenario: str = "baseline") -> pd.DataFrame | None:
    """Symptomatic and asymptomatic attack rates by age group and vaccination
    status, median [95% interval] across draws; None if the model has no
    asymptomatic compartment. Same split as build_figures.attack_rates:
    vaccinated infections go through EV (EV_to_IV / EV_to_AV), unvaccinated
    ones through E (E_to_I / E_to_A)."""
    flows = ("E_to_I", "E_to_A", "EV_to_IV", "EV_to_AV", "S_to_SV", "SV")
    df = con.execute(
        f"""SELECT rep, age_group, compartment, day, SUM(value) AS v FROM results_full
            WHERE scenario = ? AND compartment IN ({",".join("?" * len(flows))})
            GROUP BY rep, age_group, compartment, day""", [scenario, *flows]).df()
    if not (df["compartment"] == "E_to_A").any():
        return None
    reps = np.sort(df["rep"].unique())

    def cube(comp, first_day=False):
        d = df[df["compartment"] == comp]
        d = (d[d["day"] == d["day"].min()].set_index(["rep", "age_group"])["v"] if first_day
             else d.groupby(["rep", "age_group"])["v"].sum())
        return d.unstack("age_group").reindex(index=reps, columns=range(len(AGE_GROUPS))).to_numpy()

    vacc = cube("SV", first_day=True) + cube("S_to_SV")
    groups = {"vaccinated": (cube("EV_to_IV"), cube("EV_to_AV"), vacc),
              "unvaccinated": (cube("E_to_I"), cube("E_to_A"), population[None, :] - vacc)}
    groups["total"] = tuple(a + b for a, b in zip(groups["vaccinated"], groups["unvaccinated"]))

    def cell(num, den):
        x = num / den * 100
        return f"{np.median(x):.1f}% [{np.percentile(x, 2.5):.1f}–{np.percentile(x, 97.5):.1f}%]"

    rows = []
    for i, a in list(enumerate(AGE_GROUPS)) + [(None, "All")]:
        row = {"age_group": a}
        for g, (sym, asym, den) in groups.items():
            pick = (lambda x: x[:, i]) if i is not None else (lambda x: x.sum(1))
            row[f"{g} symptomatic"] = cell(pick(sym), pick(den))
            row[f"{g} asymptomatic"] = cell(pick(asym), pick(den))
            row[f"{g} share symptomatic"] = cell(pick(sym), pick(sym) + pick(asym))
        rows.append(row)
    return pd.DataFrame(rows).set_index("age_group")


def hospitalizations(db: ResultsDB, population: np.ndarray) -> dict[str, np.ndarray]:
    """Season total hospitalizations per draw, by scenario."""
    return {s: scenario_totals(db, s, population)["new_H"].sum(axis=1) for s in SCENARIOS}


def averted(tot: dict[str, np.ndarray], population: np.ndarray) -> pd.DataFrame:
    """Total-population hospitalizations averted, median (95% interval) across
    draws, paired by draw. Per-dose columns use scheduled doses (see
    build_counterfactual_tables_from_db._sched_doses)."""
    rows = []

    def row(label, ref, scen, doses):
        d = tot[ref] - tot[scen]
        rows.append({
            "comparison": label,
            "reference hospitalizations": _cell(tot[ref]),
            "scenario hospitalizations": _cell(tot[scen]),
            "averted": _cell(d),
            "% of reference": _cell(d / tot[ref] * 100, "{:.1f}", pct=True),
            "per 100K population": _cell(d / population.sum() * 1e5, "{:.1f}"),
            "per 100K doses": _cell(d / doses * 1e5, "{:.1f}"),
        })

    row("Baseline vs no vax", "no vax", "baseline", float(cf.scheduled_doses(population).sum()))
    for t in COVERAGE_TARGETS:
        extra = float(cf.additional_scheduled_doses_for_target(population, t / 100.0).sum())
        row(f"{t}% coverage (all ages) vs baseline", "baseline", f"{t}% coverage (all ages)", extra)
    return pd.DataFrame(rows).set_index("comparison")


def _md(df: pd.DataFrame) -> str:
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, metavar="LABEL=RESULTS:FITTED")
    ap.add_argument("--out", default=str(_HERE / "compare_IHR_ub_5"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    population = load_population()

    fits, attacks, sym, hosp, avert = {}, [], [], {}, []
    for spec in args.runs:
        label, paths = spec.split("=", 1)
        parts = paths.split(":")
        results, fitted = _HERE / parts[0], _HERE / parts[1]
        model_config = _HERE / (parts[2] if len(parts) > 2 else MODEL_CONFIG_FILE)
        print(f"[{label}] {results}")
        fits[label] = fit_params(fitted, model_config)
        db = ResultsDB(str(results))
        sub = out / "per_run" / "".join(c if c.isalnum() else "_" for c in label)
        sub.mkdir(parents=True, exist_ok=True)
        ar = attack_rates(db._con, sub).set_index("age_group")
        ar.columns = pd.MultiIndex.from_product([[label], ar.columns])
        attacks.append(ar)
        sy = symptomatic(db._con, population)
        if sy is not None:
            sy.columns = pd.MultiIndex.from_product([[label], sy.columns])
            sym.append(sy)
        tot = hospitalizations(db, population)
        hosp[label] = {s: _cell(x) for s, x in tot.items()}
        av = averted(tot, population)
        av.columns = pd.MultiIndex.from_product([[label], av.columns])
        avert.append(av)
        db.close()

    fit_tab = pd.DataFrame(fits)
    fit_tab.index.name = "parameter"
    ar_tab = pd.concat(attacks, axis=1)
    av_tab = pd.concat(avert, axis=1)
    hosp_tab = pd.DataFrame(hosp).rename_axis("scenario")
    fit_tab.to_csv(out / "fit_params.csv")
    ar_tab.to_csv(out / "attack_rates.csv")
    hosp_tab.to_csv(out / "hospitalizations.csv")
    if sym:
        sym_tab = pd.concat(sym, axis=1)
        sym_tab.to_csv(out / "symptomatic.csv")
    av_tab.to_csv(out / "averted.csv")

    # Markdown: one table per measure, one column (attack rates) or row
    # (averted) per run, so the runs sit next to each other.
    labels = list(fits)
    md = ["# Run comparison", "", "## Fit", "", _md(fit_tab), "",
          "## Baseline attack rate, median [95% interval]", "",
          "Vaccinated = vaccinated at any point in the season; their infections are those after vaccination.", ""]
    for m, title in (("total", "Total"), ("vaccinated", "Vaccinated"), ("unvaccinated", "Unvaccinated"),
                     ("share_vaccinated", "Share vaccinated")):
        md += [f"### {title}", "",
               _md(pd.DataFrame({l: ar_tab[(l, m)] for l in labels}).rename_axis("age group")), ""]
    if sym:
        md += ["## Baseline symptomatic / asymptomatic attack rate, median [95% interval]", "",
               "Runs whose model has asymptomatic compartments only.", ""]
        for l in sym_tab.columns.get_level_values(0).unique():
            for g in ("total", "vaccinated", "unvaccinated"):
                t = sym_tab[l][[f"{g} symptomatic", f"{g} asymptomatic", f"{g} share symptomatic"]]
                t.columns = ["symptomatic", "asymptomatic", "share of infections symptomatic"]
                md += [f"### {l}: {g}", "", _md(t.rename_axis("age group")), ""]
    md += ["## Total hospitalizations, median (95% interval)", "", _md(hosp_tab), ""]
    md += ["## Hospitalizations averted, total population, median (95% interval)", ""]
    for comp in av_tab.index:
        tab = pd.DataFrame({l: av_tab.loc[comp, l] for l in labels}).T.rename_axis("run")
        md += [f"### {comp}", "", _md(tab), ""]
    (out / "comparison.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
