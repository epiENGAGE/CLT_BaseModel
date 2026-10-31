"""Build the MA_vax report (Markdown, HTML and PDF) from the pipeline outputs.

    python generic_core/examples/MA_vax/report/build_report.py [--no-pdf]

Every number is read from files the pipeline writes, so re-running after a
new fit / simulation (or an alternative model version) regenerates the report
unchanged:

  model_config_MA_vax.json, fit_config_MA_vax.json, fitted_params_MA_vax.json
  counterfactual_tables_from_db_MA_vax/      build_counterfactual_tables_from_db.py
  fit_comparison_output_baseline/cumulative_comparison.csv   plot_fit_vs_actual.py
  figures/output/                            figures/build_figures.py

Writes report/report_MA_vax.md, report_MA_vax.html (images referenced from
../figures) and report_MA_vax.pdf (headless Google Chrome, as the previous
report.pdf was made). The interpretive sentences quote numbers from the tables
but are otherwise generic -- re-read them after a model change.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
MA_VAX = REPORT_DIR.parent
sys.path.insert(0, str(MA_VAX.parent.parent.parent))
sys.path.insert(0, str(MA_VAX))

from make_scenarios_MA_vax import COVERAGE_TARGETS, COVERAGE_DECREASES
from ma_vax_shared import AGE_GROUP_LABELS as AGE_GROUPS

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIG = "../figures/output"
STATIC = "../figures/static"


# ── helpers ──────────────────────────────────────────────────────────────────

def med(cell) -> float:
    m = re.match(r"^\s*(-?[\d,]*\.?\d+)", str(cell))
    return float(m.group(1).replace(",", "")) if m else np.nan


def dash(cell) -> str:
    return str(cell).replace(" - ", " – ")


def md_table(df: pd.DataFrame, bold_last: bool = False, index: bool = True, align=None) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    al = align or ["---"] + ["---:"] * (len(cols) - 1)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(al) + "|"]
    rows = [[dash(v) for v in r] for r in d.itertuples(index=False)]
    for k, r in enumerate(rows):
        if bold_last and k == len(rows) - 1:
            r = [f"**{v}**" for v in r]
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def fig(name: str, caption: str, folder: str = FIG) -> str:
    return f"![{caption}]({folder}/{name})\n\n*{caption}*"


def fitted_risk(a: pd.DataFrame, col: str) -> list[str]:
    """ "mean% (lo–hi%) [pre-fit%]" per age group from age_parameters.csv."""
    return [f"{m * 100:.3f}% ({lo * 100:.3f}–{hi * 100:.3f}%) [{pre * 100:.3f}%]"
            for m, lo, hi, pre in zip(a[f"{col}_fitted_mean"], a[f"{col}_fitted_p2.5"],
                                      a[f"{col}_fitted_p97.5"], a[col])]


def pct(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}%"


# ── report ───────────────────────────────────────────────────────────────────

def build(tables: Path, figs: Path, fit_cmp: Path) -> str:
    cfg = json.loads((MA_VAX / "model_config_MA_vax.json").read_text())
    fitc = json.loads((MA_VAX / "fit_config_MA_vax.json").read_text())
    fitp = json.loads((MA_VAX / "fitted_params_MA_vax.json").read_text())
    p = cfg["params"]
    n_fit_draws = len(fitp["accepted_params"])
    _meta = tables / "meta.json"
    n_draws = int(json.loads(_meta.read_text()).get("n_reps", n_fit_draws)) if _meta.exists() else n_fit_draws
    num_days = int(fitp["num_days"])
    start = pd.Timestamp(fitc["run_kwargs"]["start_date"])
    end = start + pd.Timedelta(days=num_days - 1)
    fc = fitc["fit_config"]
    n_knots = sum(1 for k in fitp["best_params"] if k.startswith("m_dlog_"))
    weights = {t["label"]: t["weight"] for t in fitc["targets"]}

    t = lambda name, **kw: pd.read_csv(tables / name, index_col=0, **kw)
    s1 = t("S_A_1.csv")
    s2 = {k: t(f"S_A_2_{k}.csv") for k in ("absolute", "pct_reduction", "per_100k_doses")}
    s4 = pd.read_csv(tables / "S_A_4.csv")
    s5 = {k: t(f"S_A_5_{k}.csv") for k in ("absolute", "pct_reduction", "per_100k")}
    dose = t("DOSE_ACCOUNTING.csv")
    cov = pd.read_csv(tables / "COVERAGE_SUMMARY.csv").set_index("scenario")
    s3 = {T: {k: t(f"S_A_3_{T}pct_{k}.csv") for k in ("absolute", "pct_reduction", "per_100k_doses")}
          for T in COVERAGE_TARGETS}
    dec = {D: {k: t(f"DECREASE_{D}pct_{k}.csv") for k in ("absolute", "pct_increase", "per_100k_doses")}
           for D in COVERAGE_DECREASES}
    fitpar = pd.read_csv(figs / "fit_parameters.csv")
    agepar = pd.read_csv(figs / "age_parameters.csv")
    cmp_ = pd.read_csv(fit_cmp) if fit_cmp.exists() else None

    base = cov.loc["baseline"]
    novax = cov.loc["no vax"]
    tot_av = s1.loc["All", "averted_total"]
    out = []
    w = out.append

    # ---- title + summary ----
    w("# MA_vax: vaccination-impact analysis report\n")
    w(f"Massachusetts, 2025–2026 influenza season ({start:%Y-%m-%d} to {end:%Y-%m-%d}, "
      f"{num_days} days). Age-structured SEIR model with a parallel vaccinated arm, calibrated "
      "to daily hospital admissions by age group with Bayesian MCMC. This report documents "
      "the model, the fit and the vaccination-impact analysis, including scenarios that "
      "raise vaccination coverage to 50–70% or cut the season's doses by 5–20%.\n")
    w(f"Every count is a **hospital admission over the whole {num_days}-day season** unless "
      f"stated otherwise. Intervals are 95% intervals across {n_draws} simulations, one per "
      "posterior parameter draw.\n")
    w("## Summary\n")
    s3_all = {T: s3[T]["absolute"].loc["All", "All"] for T in COVERAGE_TARGETS}
    bullets = [
        f"The fitted baseline produces **{base.hosp_median:,.0f}** "
        f"[{base.hosp_lo:,.0f} – {base.hosp_hi:,.0f}] hospitalizations, against "
        f"**{novax.hosp_median:,.0f}** with no vaccination: the {pct(base.scheduled_coverage_pct)} "
        f"overall coverage achieved averts **{dash(tot_av)}** hospitalizations "
        f"(**{dash(s1.loc['All', 'pct_averted_total'])}**).",
        "Raising every age group to a coverage target (groups already above it unchanged) averts "
        + ", ".join(f"{dash(s3_all[T])} more at {T}%" for T in COVERAGE_TARGETS) + " (vs. baseline).",
        "Removing doses from every age group adds "
        + ", ".join(f"{dash(dec[D]['absolute'].loc['All', 'All'])} at −{D}%" for D in COVERAGE_DECREASES)
        + " hospitalizations.",
    ]
    per_dose_all = {a: med(s2["per_100k_doses"].loc["All", a]) for a in AGE_GROUPS}
    top = sorted(per_dose_all, key=lambda k: per_dose_all[k], reverse=True)[:2]
    bullets.append(
        f"Per dose, vaccinating school-age groups prevents the most hospitalizations across the "
        f"population (Table S.A.2: {top[0]} {per_dose_all[top[0]]:,.0f} and {top[1]} "
        f"{per_dose_all[top[1]]:,.0f} per 100,000 doses), mostly through indirect protection of 65+.")
    w("\n".join(f"- {b}" for b in bullets) + "\n")
    w("---\n")

    # ---- 1. model ----
    w("## 1. Model structure\n")
    w("### 1.1 Compartments\n")
    w("Nine compartments, each stratified by 7 age groups (" + ", ".join(f"`{a}`" for a in AGE_GROUPS) + "):\n")
    w(md_table(pd.DataFrame({"Compartment": ["`S`", "`E`", "`I`", "`R`", "`SV`", "`EV`", "`IV`", "`H`", "`D`"],
                             "Meaning": ["Susceptible, unvaccinated", "Exposed (latent), unvaccinated",
                                         "Infectious, unvaccinated", "Recovered (either arm)",
                                         "Susceptible, vaccinated", "Exposed (latent), vaccinated",
                                         "Infectious, vaccinated", "Hospitalized (either arm)",
                                         "Dead (either arm)"]}), index=False, align=["---", "---"]) + "\n")
    w(fig("model_diagram.png", "Model structure", STATIC) + "\n")
    w("The vaccinated arm (`SV → EV → IV → H/R`) mirrors the unvaccinated arm "
      "(`S → E → I → H/R`) with its own susceptibility and severity parameters. `H`, `R` and "
      "`D` are shared, so vaccination history is not tracked once hospitalized or recovered.\n")
    w("### 1.2 Force of infection\n")
    w("Contacts come from fixed age × age contact matrices (Mistry et al. 2021), with school "
      "and work contacts removed on non-school / non-work days:\n")
    w("```\nC(t) = total_C − (1 − is_school(t))·school_C − (1 − is_work(t))·work_C\n"
      "beta_adj(t) = beta_baseline · m(t) · (1 + humidity_impact · exp(−180 · humidity(t)))\n"
      "wtd_inf_prop(t) = (I·I_relative_infectiousness + IV·IV_relative_infectiousness) / population\n"
      "foi(t) = beta_adj(t) · (C(t) @ wtd_inf_prop(t))\n"
      "S_to_E   = foi(t) · relative_suscept   · S\n"
      "SV_to_EV = foi(t) · vax_susceptibility · SV\n```\n")
    w("`m(t)` is a smoothly time-varying transmission multiplier (§2.3) that absorbs "
      "behavioural and seasonal variation the mechanistic terms do not capture. "
      "`1 − vax_susceptibility` is the vaccine effectiveness (VE) against infection.\n")
    w("### 1.3 Progression, hospitalization, death\n")
    w("```\nE_to_I  = E_to_I_rate · E                 EV_to_IV = EV_to_IV_rate · EV\n"
      "I_to_H  = I_out_rate · I_to_H_prop · I     IV_to_H  = I_out_rate · IV_to_H_prop · IV\n"
      "I_to_R  = I_out_rate · (1−I_to_H_prop)·I   IV_to_R  = I_out_rate · (1−IV_to_H_prop)·IV\n"
      "H_to_D  = H_out_rate · H_to_D_prop · H\nH_to_R  = H_out_rate · (1−H_to_D_prop)·H\n```\n")
    w("### 1.4 Vaccination flow\n")
    delay = int(p.get("vax_transfer_delay_days", 0))
    w(f"Daily doses (an age-specific share of the population, applied {delay} days after the "
      "dose date to represent the delay to immunity) move an exact count from `S` to `SV` each "
      "day, capped at what remains in `S`:\n")
    w("```\nS_to_SV(t) = min(round(vax_prop(t − delay) · (S + SV)), S)\n```\n")
    w("A dose scheduled for someone who has already been infected is not delivered by the model "
      "(the cap). Those doses are still bought and given in reality, so every \"per 100,000 "
      "doses\" figure divides by **scheduled** doses (appendix B).\n")
    w("### 1.5 Simulation and uncertainty\n")
    w(f"Explicit Euler integration with {fitc['run_kwargs']['ts_per_day']} sub-steps per day. "
      f"Every interval comes from a **posterior parameter ensemble**: one deterministic run per "
      f"posterior draw ({n_draws} draws), the same draws for every scenario, so differences "
      "between scenarios are paired. Intervals therefore reflect calibration uncertainty, not "
      "epidemic-process noise.\n")
    w("---\n")

    # ---- 2. parameters ----
    w("## 2. Parameters\n")
    w("### 2.1 Fixed scalar parameters\n")
    scal = [("relative_suscept", "Susceptibility multiplier, unvaccinated arm"),
            ("I_relative_infectiousness", "Infectiousness weight, unvaccinated `I`"),
            ("IV_relative_infectiousness", "Infectiousness weight, vaccinated `IV`"),
            ("E_to_I_rate", "Latent → infectious rate (/day)"),
            ("EV_to_IV_rate", "Same, vaccinated arm (/day)"),
            ("I_out_rate", "Infectious-period exit rate (/day)"),
            ("H_out_rate", "Hospital exit rate (/day)"),
            ("vax_transfer_delay_days", "Days from dose to modelled immunity")]
    w(md_table(pd.DataFrame([{"Parameter": f"`{k}`", "Value": p[k], "Meaning": m} for k, m in scal]),
               index=False, align=["---", "---:", "---"]) + "\n")
    w("### 2.2 Age-stratified parameters\n")
    a = agepar.copy()
    a_tab = pd.DataFrame({
        "Age group": a["age_group"],
        "Population": a["population"].map("{:,}".format),
        "Hosp. risk given infection¹": fitted_risk(a, "hosp_risk_given_infection"),
        "Hosp. risk given breakthrough¹": fitted_risk(a, "hosp_risk_given_breakthrough"),
        "Death risk given hosp.": (a["death_risk_given_hosp"] * 100).map("{:.2f}%".format),
        "Residual susceptibility if vaccinated": a["residual_susceptibility_vaccinated"].map("{:.2f}".format),
        "Initial infections seeded": a["initial_infections_seeded"],
    })
    w(md_table(a_tab, index=False) + "\n")
    w("¹ Fitted: pre-fit value × the age group's fitted multiplier (§2.4), posterior mean "
      "(95% interval), pre-fit value in brackets. Both columns share the multiplier, so VE "
      "against hospitalization given infection is unchanged by the fit. Seeds are pre-fit "
      "values, scaled by one fitted multiplier. A residual susceptibility of 1.00 (65+) means "
      "vaccination blocks no infection there: it only reduces severity, and vaccinating 65+ "
      "has no indirect effect on other age groups.\n")
    w("### 2.3 Fitting method\n")
    b = fc["bounds"]
    w(f"Affine-invariant ensemble MCMC ({max(fc['n_walkers'], 2 * (len(fitp['best_params']) - 1))} "
      f"walkers, {fc['n_iter']} iterations each) over the {num_days}-day target window.\n")
    w(md_table(pd.DataFrame([
        {"Parameter": "`beta_baseline`", "Prior": f"Uniform({b['beta_baseline'][0]}, {b['beta_baseline'][1]})",
         "Role": "Baseline transmission rate"},
        {"Parameter": "`humidity_impact`", "Prior": "Uniform(0, 1)", "Role": "Strength of humidity forcing"},
        {"Parameter": "Initial-seed multiplier", "Prior": f"Log-uniform({b['seed_scale_E'][0]}, {b['seed_scale_E'][1]})",
         "Role": "Multiplies the age-specific seed counts"},
        {"Parameter": "Hospitalization-risk multiplier (× 7)",
         "Prior": f"Uniform({b['IHR_scale'][0]}, {b['IHR_scale'][1]})",
         "Role": "Multiplies both hospitalization-risk columns of one age group"},
        {"Parameter": f"`m(t)` log-increments (× {n_knots})", "Prior": f"Normal(0, {fc['tv_tau']})",
         "Role": f"Random-walk steps in log-transmission, knots every {fc['tv_knot_spacing_days']} days"},
        {"Parameter": "`phi`", "Prior": "—", "Role": "Negative-binomial dispersion (likelihood only)"},
    ]), index=False, align=["---", "---", "---"]) + "\n")
    wt = [f"{lbl.replace('MA_flu_daily_hospitalizations_', '').replace('.csv', '')}: {v:g}"
          for lbl, v in weights.items() if "daily" in lbl]
    w("**Likelihood**: negative-binomial (NB2), jointly over 8 targets: daily admissions for each "
      f"age group (weights {', '.join(wt)}) and end-of-season cumulative admissions by age "
      f"(weight {weights.get('MA_flu_end_of_season_hospitalizations_by_age.csv', 1):g}).\n")
    w(f"**Posterior sample**: first {fc['mcmc_burnin']} iterations discarded, then every "
      f"{fc['mcmc_thin']}th kept, giving **{n_fit_draws} draws**. \"Best\" is the draw with the "
      "highest log-posterior.\n")
    w("### 2.4 Fitted parameters\n")
    fp = fitpar.copy()
    fmt = lambda v: f"{v:.4f}" if abs(v) < 0.1 else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
    fp_tab = pd.DataFrame({"Parameter": fp["parameter"], "Posterior mean": fp["posterior_mean"].map(fmt),
                           "5%": fp["p5"].map(fmt), "95%": fp["p95"].map(fmt), "\"Best\"": fp["best"].map(fmt)})
    w(md_table(fp_tab, index=False) + "\n")
    w(fig("transmission_multiplier.png", "Humidity factor, fitted m(t) and their product "
          "(posterior median and 95% band)") + "\n")
    w("### 2.5 Vaccination coverage\n")
    cv = dose[["population", "scheduled_coverage"]].rename(columns={
        "population": "Population", "scheduled_coverage": "Scheduled coverage"})
    cv.index.name = "Age group"
    w(f"Cumulative share of each age group vaccinated by the schedule over the simulated "
      f"season (after the {delay}-day delay):\n")
    w(md_table(cv, bold_last=True) + "\n")
    w(fig("vax_coverage_cumulative.png", "Cumulative scheduled vaccination coverage") + "\n")
    w("---\n")

    # ---- 3. results ----
    w("## 3. Vaccination impact of the 2025–2026 programme\n")
    w(f"The fitted baseline produces a median of **{base.hosp_median:,.0f}** hospitalizations "
      f"across all ages, against **{novax.hosp_median:,.0f}** with no vaccination.\n")
    w(fig("ts_baseline_infection_protection_no_vax.png",
          "Daily hospitalizations: baseline, infection protection only, no vaccination") + "\n")
    w("### Table S.A.1 — Hospitalizations averted, infection vs. severity protection\n")
    for key, title in (("reduced_infection", "**Infection protection** (no vaccination → VE against infection only)"),
                       ("reduced_severity", "**Severity protection** (adding VE against hospitalization given infection)"),
                       ("total", "**Total** (no vaccination → baseline)")):
        tab = pd.DataFrame({"Hospitalizations averted": s1[f"averted_{key}"],
                            "% averted": s1[f"pct_averted_{key}"],
                            "Per 100,000 population": s1[f"per100k_averted_{key}"],
                            "Per 100,000 doses": s1[f"per100k_doses_averted_{key}"]})
        tab.index.name = "Age group"
        w(title + "\n")
        w(md_table(tab, bold_last=True) + "\n")
    w("### Table S.A.2 — Hospitalizations averted by age group vaccinated\n")
    w("Each column vaccinates one age group only (all others unvaccinated) and compares with no "
      "vaccination; `All` is the full baseline schedule. Rows are the age group in which "
      "hospitalizations are counted, so off-diagonal cells are indirect protection.\n")
    for k, title in (("absolute", "Hospitalizations averted"), ("pct_reduction", "% reduction"),
                     ("per_100k_doses", "Per 100,000 doses given to the column's age group")):
        tab = s2[k].copy()
        tab.columns = [f"{c} vacc." for c in tab.columns]
        tab.index.name = "Counted in"
        w(f"**{title}**\n")
        w(md_table(tab, bold_last=True) + "\n")
    w(fig("S_A_2_pct_reduction_stacked_bar.png",
          "Share of each age group's averted hospitalizations, by age group vaccinated") + "\n")
    w("### Table S.A.4 — Vaccine-effectiveness sensitivity scenarios\n")
    s4t = s4.rename(columns={str(s4.columns[0]): "Scenario", str(s4.columns[1]): "Age group"})
    w(md_table(s4t, index=False, align=["---", "---"] + ["---:"] * (len(s4t.columns) - 2)) + "\n")
    w("### Table S.A.5 — Hospitalizations averted across VE scenarios (vs. no vaccination)\n")
    for k, title in (("absolute", "Hospitalizations averted"), ("pct_reduction", "% reduction")):
        tab = s5[k].rename(columns={"low_ve": "Low VE", "baseline_ve": "Baseline VE", "high_ve": "High VE"})
        tab.index.name = "Age group"
        w(f"**{title}**\n")
        w(md_table(tab, bold_last=True) + "\n")
    w(fig("ts_ve.png", "Daily hospitalizations under low, baseline and high VE") + "\n")
    w("---\n")

    # ---- 4. coverage scenarios ----
    w("## 4. Coverage scenarios\n")
    w("Two families of counterfactual schedules, each applied either to **every age group at "
      "once** or to **one age group at a time** (others keep the baseline schedule), and "
      "compared with the baseline:\n")
    w("- **Coverage targets** (" + ", ".join(f"{T}%" for T in COVERAGE_TARGETS) + "): each "
      "targeted group's daily doses are scaled by `target / baseline coverage`, so its season "
      "coverage reaches the target. A group already above the target keeps its schedule.\n")
    w("- **Fewer doses** (" + ", ".join(f"−{D}%" for D in COVERAGE_DECREASES) + "): each "
      "targeted group's daily doses are cut by that share (a relative decrease, e.g. 60% "
      "coverage becomes 54% at −10%). Timing is unchanged.\n")
    w("### 4.1 All age groups together\n")
    rows = []
    for name in (["no vax"] + [f"{D}% lower coverage (all ages)" for D in sorted(COVERAGE_DECREASES, reverse=True)]
                 + ["baseline"] + [f"{T}% coverage (all ages)" for T in COVERAGE_TARGETS]):
        r = cov.loc[name]
        rows.append({"Scenario": r.label, "Overall coverage": pct(r.scheduled_coverage_pct),
                     "Doses vs. baseline": f"{r.dose_change_vs_baseline:+,.0f}",
                     "Hospitalizations": f"{r.hosp_median:,.0f} [{r.hosp_lo:,.0f} – {r.hosp_hi:,.0f}]",
                     "Change vs. baseline": f"{r.change_vs_baseline_median:+,.0f} "
                                            f"[{r.change_vs_baseline_lo:+,.0f} – {r.change_vs_baseline_hi:+,.0f}]",
                     "% change": f"{r.pct_change_median:+.1f}% [{r.pct_change_lo:+.1f}% – {r.pct_change_hi:+.1f}%]"})
    w(md_table(pd.DataFrame(rows), index=False) + "\n")
    w("Overall coverage = scheduled doses summed over all age groups ÷ total population, i.e. "
      "the population-weighted average of each age group's scheduled coverage.\n")
    w(fig("coverage_response.png", "Season hospitalizations against overall scheduled coverage") + "\n")
    w(fig("ts_coverage_targets.png", "Daily hospitalizations: baseline and coverage targets (all ages)") + "\n")
    w(fig("ts_coverage_decreases.png", "Daily hospitalizations: baseline and fewer doses (all ages)") + "\n")
    w("### 4.2 One age group at a time\n")
    w("Change in **total** (all-ages) hospitalizations when only the row's age group is "
      "changed, median across draws (95% intervals in appendix A):\n")
    w(fig("coverage_heatmap.png", "% change in total hospitalizations, by age group changed and level") + "\n")
    per_target = pd.DataFrame({f"{T}%": [dash(s3[T]["absolute"].loc["All", c]) for c in AGE_GROUPS + ["All"]]
                               for T in COVERAGE_TARGETS},
                              index=[f"{c} only" for c in AGE_GROUPS] + ["All ages"])
    per_target.index.name = "Raised to target"
    w("**Hospitalizations averted vs. baseline, all ages counted**\n")
    w(md_table(per_target) + "\n")
    per_dec = pd.DataFrame({f"−{D}%": [dash(dec[D]["absolute"].loc["All", c]) for c in AGE_GROUPS + ["All"]]
                            for D in COVERAGE_DECREASES},
                           index=[f"{c} only" for c in AGE_GROUPS] + ["All ages"])
    per_dec.index.name = "Doses cut"
    w("**Additional hospitalizations vs. baseline, all ages counted**\n")
    w(md_table(per_dec) + "\n")
    w("### 4.3 Efficiency per dose\n")
    w("Hospitalizations averted per 100,000 **additional** doses (targets) and added per "
      "100,000 doses **removed** (cuts), all ages counted. For targets the denominator is the "
      "extra doses the target requires in that group (`All ages`: in every group).\n")
    eff = pd.DataFrame({**{f"{T}% target": [dash(s3[T]["per_100k_doses"].loc["All", c]) for c in AGE_GROUPS + ["All"]]
                           for T in COVERAGE_TARGETS},
                        **{f"−{D}%": [dash(dec[D]["per_100k_doses"].loc["All", c]) for c in AGE_GROUPS + ["All"]]
                           for D in COVERAGE_DECREASES}},
                       index=[f"{c} only" for c in AGE_GROUPS] + ["All ages"])
    eff.index.name = "Age group changed"
    w(md_table(eff) + "\n")
    lo_T, hi_T = COVERAGE_TARGETS[0], COVERAGE_TARGETS[-1]
    w(fig(f"S_A_3_{lo_T}pct_per_100k_doses_stacked_bar.png",
          f"{lo_T}% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized") + "\n")
    w(fig(f"S_A_3_{hi_T}pct_per_100k_doses_stacked_bar.png",
          f"{hi_T}% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized") + "\n")
    mid_D = COVERAGE_DECREASES[len(COVERAGE_DECREASES) // 2 - 1]
    w(fig(f"DECREASE_{mid_D}pct_per_100k_doses_stacked_bar.png",
          f"{mid_D}% fewer doses: additional hospitalizations per 100,000 doses removed") + "\n")
    w("A relative dose cut removes the most doses where coverage is highest, so per removed "
      "dose its cost mirrors the per-dose benefit of Table S.A.2; coverage targets add doses "
      "mostly to the least-covered groups (18-49 needs the largest relative increase).\n")
    w("---\n")

    # ---- 5. fit ----
    w("## 5. Baseline fit check\n")
    if cmp_ is not None:
        c = cmp_.copy()
        ctab = pd.DataFrame({"Age group": c["age_group"],
                             "Simulated (median)": c["simulated_median"].map("{:,.1f}".format),
                             "Simulated 95% interval": [f"{lo:,.1f} – {hi:,.1f}" for lo, hi in
                                                        zip(c["simulated_ci_lower"], c["simulated_ci_upper"])],
                             "Observed": c["observed"].map("{:,.1f}".format),
                             "% difference": c["pct_difference"].map("{:+.1f}%".format)})
        w(f"Cumulative admissions over {start:%Y-%m-%d} – {end:%Y-%m-%d}:\n")
        w(md_table(ctab, index=False, bold_last=True) + "\n")
    w(fig("fit_check_daily_by_age.png", "Daily admissions: posterior median and 95% band vs. observed") + "\n")
    w(fig("fit_check_cumulative_by_age.png", "Cumulative admissions: posterior median and 95% band vs. observed") + "\n")
    w("---\n")

    # ---- 6/7 ----
    w("## 6. Data sources\n")
    w(md_table(pd.DataFrame([
        {"Input": "Hospital admissions (calibration target)",
         "Source": "Massachusetts Department of Public Health (MA DPH): weekly influenza admissions by county and age group "
                   "(`influenza_timeseries_*.csv`, `ed_admits_count`), summed over counties and "
                   "spread evenly over each MMWR week (`data/prepare_hospitalizations.py`)"},
        {"Input": "Vaccination", "Source": "MA DPH: weekly cumulative first doses by county and age "
                                           "(`MA_influenza_coverage_2025-2026.csv`), differenced and spread "
                                           "over each week (`data/prepare_vaccination.py`)"},
        {"Input": "Population by age group", "Source": "MA DPH: `POP_ESTIMATE` of the vaccination file, summed over counties"},
        {"Input": "Contact matrices", "Source": "Mistry et al. 2021 synthetic contact matrices"},
        {"Input": "Absolute humidity", "Source": "gridMET daily specific humidity averaged over Massachusetts "
                                                 "(× 1.2 kg/m³, `data/extract_ma_humidity.py`)"},
        {"Input": "School/work calendar", "Source": "Constructed state school/work-day calendar"},
    ]), index=False, align=["---", "---"]) + "\n")
    w("## 7. Notes\n")
    w("\n".join(f"- {n}" for n in [
        "Intervals reflect calibration uncertainty (posterior draws, deterministic transitions), "
        "not epidemic-process noise.",
        "`m(t)` is a statistical smoothing device, not an independently measured behavioural signal.",
        "Coverage-target multipliers are computed from the scheduled coverage over the simulated "
        "window; the model's cap means delivered coverage lands slightly below the target "
        "(appendix B).",
        "\"Per 100,000 doses\" uses scheduled doses: doses the model declines to deliver to "
        "already-infected people are still bought and given.",
        "Vaccinating 65+ has no indirect effect on other age groups because vaccination blocks no "
        "infection in 65+ in the fitted model (residual susceptibility 1.00).",
    ]) + "\n")
    w("---\n")

    # ---- appendices ----
    w("## Appendix A — Coverage-scenario tables by age group hospitalized\n")
    for T in COVERAGE_TARGETS:
        tab = s3[T]["absolute"].copy()
        tab.columns = [f"{c} raised" for c in tab.columns]
        tab.index.name = "Counted in"
        w(f"### {T}% coverage target — hospitalizations averted vs. baseline\n")
        w(md_table(tab, bold_last=True) + "\n")
    for D in COVERAGE_DECREASES:
        tab = dec[D]["absolute"].copy()
        tab.columns = [f"{c} cut" for c in tab.columns]
        tab.index.name = "Counted in"
        w(f"### {D}% fewer doses — additional hospitalizations vs. baseline\n")
        w(md_table(tab, bold_last=True) + "\n")
    w("## Appendix B — Dose accounting: scheduled vs. delivered doses\n")
    dt = dose.rename(columns=lambda c: c.replace("_", " ").capitalize())
    dt.index.name = "Age group"
    w(md_table(dt, bold_last=True) + "\n")
    w(fig("vax_coverage_daily.png", "Daily scheduled vaccination proportion") + "\n")
    ar_path = figs / "attack_rates.csv"
    if ar_path.exists():
        ar = pd.read_csv(ar_path).rename(columns={
            "age_group": "Age group", "share_vaccinated": "Share vaccinated", "vaccinated": "Vaccinated",
            "unvaccinated": "Unvaccinated", "total": "Total"})
        ar = ar[["Age group", "Share vaccinated", "Vaccinated", "Unvaccinated", "Total"]]
        w("## Appendix C — Attack rate by vaccination status (baseline)\n")
        w("Share of each group infected over the season, median [95% interval]. Vaccinated = "
          "vaccinated at any point in the season (delivered doses), with their infections after "
          "vaccination; unvaccinated = everyone else, seeded infections included. The model never "
          "vaccinates someone already infected, so an infection before a person's vaccination date "
          "counts as unvaccinated. This timing effect lowers the vaccinated attack rate even where "
          "vaccination blocks no infection (65+).\n")
        w(md_table(ar, index=False, bold_last=True) + "\n")
    return "\n".join(out)


CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; background: #fff; max-width: 980px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22pt; } h2 { font-size: 16pt; border-bottom: 1px solid #ccc; padding-bottom: 4px;
     margin-top: 28px; page-break-after: avoid; } h3 { font-size: 12.5pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0 14px; font-size: 8.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 3px 6px; } th { background: #f2f2f2; }
td { font-variant-numeric: tabular-nums; }
img { max-width: 100%; display: block; margin: 8px auto 2px; page-break-inside: avoid; }
p > em:only-child { display: block; text-align: center; font-size: 9pt; color: #555; }
pre { background: #f6f6f6; padding: 8px; font-size: 8.5pt; overflow-x: auto; }
code { font-size: 90%; }
@page { size: Letter; margin: 14mm 12mm; }
"""


def to_html(md_text: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>MA_vax report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=MA_VAX / "counterfactual_tables_from_db_MA_vax")
    ap.add_argument("--figures", type=Path, default=MA_VAX / "figures" / "output")
    ap.add_argument("--fit-comparison", type=Path,
                    default=MA_VAX / "fit_comparison_output_baseline" / "cumulative_comparison.csv")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    md_text = build(args.tables, args.figures, args.fit_comparison)
    md_path = REPORT_DIR / "report_MA_vax.md"
    md_path.write_text(md_text)
    html_path = REPORT_DIR / "report_MA_vax.html"
    html_path.write_text(to_html(md_text))
    print(f"wrote {md_path}\nwrote {html_path}")
    if not args.no_pdf:
        pdf_path = REPORT_DIR / "report_MA_vax.pdf"
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={pdf_path}",
                        html_path.as_uri()], check=True, capture_output=True)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
