"""Compare two (or more) MA_vax simulation runs side by side: the fit, the
attack rates by vaccination status and the hospitalizations averted.

Written to test whether the IHR_scale upper bound (2.0 in the main fit) drives
the high attack rate: the IHR_ub_5 run refits with the bound raised to 5.0 and
simulates a subset of scenarios. Each run is `label=results_dir:fitted_params`:

    python compare_runs_MA_vax.py                       # main run vs IHR_ub_5
    python compare_runs_MA_vax.py --runs \\
        "IHR max 2=simulation_output_MA_vax_param_set_stochastic/results_parquet:fitted_params_MA_vax.json" \\
        "IHR max 5=simulation_output_MA_vax_IHR_ub_5/results_parquet:fitted_params_MA_vax_IHR_ub_5.json"

Writes to --out (default compare_IHR_ub_5/):
  fit_params.csv      posterior mean (95% interval) of each fitted parameter, the
                      share of draws near the upper bound, and fitted IHR by age
  attack_rates.csv    baseline attack rate by age group: vaccinated, unvaccinated,
                      total, median (95% interval) across draws
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


def fit_params(fitted_file: Path) -> pd.Series:
    """Posterior mean (95% interval) of the scalar fit params and each age
    group's IHR multiplier, with the fitted infection IHR (pre-fit x multiplier)."""
    fit = json.loads(fitted_file.read_text())
    draws = pd.DataFrame(fit["accepted_params"])
    bounds = fit["fit_config"]["bounds"]
    pre = np.asarray(json.loads(Path(MODEL_CONFIG_FILE).read_text())["params"]["I_to_H_prop"],
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


def averted(db: ResultsDB, population: np.ndarray) -> pd.DataFrame:
    """Total-population hospitalizations averted, median (95% interval) across
    draws, paired by draw. Per-dose columns use scheduled doses (see
    build_counterfactual_tables_from_db._sched_doses)."""
    tot = {s: scenario_totals(db, s, population)["new_H"].sum(axis=1)
           for s in ["no vax", "baseline"] + [f"{t}% coverage (all ages)" for t in COVERAGE_TARGETS]}
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

    fits, attacks, avert = {}, [], []
    for spec in args.runs:
        label, paths = spec.split("=", 1)
        results, fitted = (_HERE / p for p in paths.rsplit(":", 1))
        print(f"[{label}] {results}")
        fits[label] = fit_params(fitted)
        db = ResultsDB(str(results))
        sub = out / "per_run" / "".join(c if c.isalnum() else "_" for c in label)
        sub.mkdir(parents=True, exist_ok=True)
        ar = attack_rates(db._con, sub).set_index("age_group")
        ar.columns = pd.MultiIndex.from_product([[label], ar.columns])
        attacks.append(ar)
        av = averted(db, population)
        av.columns = pd.MultiIndex.from_product([[label], av.columns])
        avert.append(av)
        db.close()

    fit_tab = pd.DataFrame(fits)
    fit_tab.index.name = "parameter"
    ar_tab = pd.concat(attacks, axis=1)
    av_tab = pd.concat(avert, axis=1)
    fit_tab.to_csv(out / "fit_params.csv")
    ar_tab.to_csv(out / "attack_rates.csv")
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
    md += ["## Hospitalizations averted, total population, median (95% interval)", ""]
    for comp in av_tab.index:
        tab = pd.DataFrame({l: av_tab.loc[comp, l] for l in labels}).T.rename_axis("run")
        md += [f"### {comp}", "", _md(tab), ""]
    (out / "comparison.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
