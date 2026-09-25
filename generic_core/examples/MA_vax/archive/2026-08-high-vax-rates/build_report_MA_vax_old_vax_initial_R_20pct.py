"""Assemble report_MA_vax_old_vax_initial_R_20pct.md -- a standalone report,
same layout as ../report.md, for the 20%-initially-recovered refit of the
Aug-2026 high-vaccination-rate model.

Every table is rendered from the saved outputs (nothing is re-simulated):
  counterfactual_tables_from_db_<TAG>_param_set_stochastic/   (build_counterfactual_tables_from_db.py)
  report_assets_<TAG>/                                       (build_report_assets_<TAG>.py)
  ../../transmission_components/attack_rate_and_IHR_by_age_<TAG>.csv
                                                             (plot_transmission_components_MA_vax.py --only <TAG>)
The earlier high-vax run (no initial immunity) is read from
counterfactual_tables_from_db_param_set_stochastic_2026-08-high-vax-rates/ and
../fitted_params.json for the comparison section.

Model sections 1.1-1.4 are copied verbatim from ../report.md (the model
structure is unchanged); everything else is written here.

    python generic_core/examples/MA_vax/archive/2026-08-high-vax-rates/build_report_MA_vax_old_vax_initial_R_20pct.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MA = HERE.parent.parent
sys.path.insert(0, str(HERE))
from report_tables_MA_vax_old_vax_initial_R_20pct import AGES, Tables, _dash, md_table  # noqa: E402

TAG = "MA_vax_old_vax_initial_R_20pct"
T = Tables(HERE / f"counterfactual_tables_from_db_{TAG}_param_set_stochastic")
T_OLD = Tables(HERE / "counterfactual_tables_from_db_param_set_stochastic_2026-08-high-vax-rates")
ASSETS = f"report_assets_{TAG}"
CFG = json.loads((HERE / f"model_config_{TAG}.json").read_text())
FIT = json.loads((HERE / f"fitted_params_{TAG}.json").read_text())
FIT_OLD = json.loads((HERE.parent / "fitted_params.json").read_text())
FITCFG = json.loads((HERE / f"fit_config_{TAG}.json").read_text())["fit_config"]
OUT = HERE / f"report_{TAG}.md"

P = CFG["params"]
IC = CFG["initial_conditions"]["aggregate_pop"]
POP = np.asarray(IC["population"], float).ravel()
E0 = np.asarray(IC["seeds"]["E"], float).ravel()
R0 = np.asarray(IC["seeds"]["R"], float).ravel()
N_DRAWS = len(FIT["accepted_params"])


def first(cell: str) -> float:
    """Median (leading number) of a 'median [lo – hi]' cell."""
    m = re.match(r"-?[\d.]+", str(cell).replace(",", ""))
    assert m, cell
    return float(m.group(0))


def interval(cell: str) -> str:
    return cell.split(" ", 1)[1] if " " in cell else ""


def load_asset(name: str) -> pd.DataFrame:
    return pd.read_csv(HERE / ASSETS / name, index_col=0)


# ---- model sections copied from the archived report --------------------------
_old = (HERE.parent / "report.md").read_text()
MODEL_1_1_TO_1_4 = _old[_old.index("### 1.1 Compartments"):_old.index("### 1.5 Numerical scheme")].rstrip()


# ---- §2 tables -----------------------------------------------------------------
def fixed_scalar_table() -> str:
    rows = [
        ["`num_days`", "250", "Simulation length (2025-09-01 through 2026-05-08)"],
        ["`relative_suscept`", f"{P['relative_suscept']}", "Susceptibility multiplier, unvaccinated arm"],
        ["`I_relative_infectiousness`", f"{P['I_relative_infectiousness']}", "Infectiousness weight, unvaccinated `I`"],
        ["`IV_relative_infectiousness`", f"{P['IV_relative_infectiousness']}", "Infectiousness weight, vaccinated `IV`"],
        ["`E_to_I_rate`", f"{P['E_to_I_rate']} /day", "~2-day latent period"],
        ["`EV_to_IV_rate`", f"{P['EV_to_IV_rate']} /day", "Same, vaccinated arm"],
        ["`I_out_rate`", f"{P['I_out_rate']} /day", "~3-day infectious period"],
        ["`H_out_rate`", f"{P['H_out_rate']} /day", "~6-day hospital stay"],
        ["`vax_transfer_delay_days`", f"{P['vax_transfer_delay_days']}", "Days from dose to modeled immunity"],
    ]
    return md_table(["Parameter", "Value", "Meaning"], rows, bold_last=False)


def age_param_table() -> str:
    def col(name):
        return np.asarray(P[name], float).ravel()
    ih, ivh, hd, vs = col("I_to_H_prop"), col("IV_to_H_prop"), col("H_to_D_prop"), col("vax_susceptibility")
    rows = [[AGES[a], f"{POP[a]:,.0f}", f"{100*ih[a]:.3f}%", f"{100*ivh[a]:.3f}%", f"{100*hd[a]:.2f}%",
             f"{vs[a]:.2f}", f"{E0[a]:.0f}", f"{R0[a]:,.0f} ({100*R0[a]/POP[a]:.0f}%)"] for a in range(len(AGES))]
    return md_table(["Age group", "Population", "Hospitalization risk, given infection¹",
                     "Hospitalization risk, given breakthrough infection¹", "Death risk, given hospitalization",
                     "Residual susceptibility if vaccinated", "Initial infections seeded (E)¹",
                     "Initially recovered (R)"], rows, bold_last=False,
                    align=["---"] + ["---:"] * 7)


def fitted_params_table() -> str:
    fp = load_asset("fitted_params_summary.csv")
    old = pd.DataFrame(FIT_OLD["accepted_params"])
    names = {"beta_baseline": "`beta_baseline`", "humidity_impact": "`humidity_impact`",
             "seed_scale_E": "Initial-seed multiplier", "phi": "`phi` (NB dispersion)"}
    names.update({f"IHR_scale|a{a}": f"Hospitalization-risk multiplier — {AGES[a]}" for a in range(len(AGES))})

    def f(x, k):
        return f"{x:.0f}" if k == "phi" else (f"{x:.4f}" if k == "beta_baseline" else
                                              (f"{x:.3f}" if k == "humidity_impact" else f"{x:.2f}"))
    rows = []
    for k in names:
        r = fp.loc[k]
        rows.append([names[k], f(r.posterior_mean, k), f(r.p05, k), f(r.p95, k), f(r.best, k),
                     f(float(old[k].mean()), k)])
    return md_table(["Parameter", "Posterior mean", "5%", "95%", "\"Best\" point",
                     "Earlier high-vax fit, posterior mean"], rows, bold_last=False,
                    align=["---"] + ["---:"] * 5)


def coverage_table() -> tuple[str, pd.DataFrame]:
    cov = load_asset("vaccination_coverage.csv")
    rows = [[str(a), f"{r.population:,.0f}", f"{100*r.scheduled_coverage:.1f}%"] for a, r in cov.iterrows()]
    rows[-1][0] = "All (population-weighted)"
    return md_table(["Age group", "Population", "Cumulative coverage (scheduled)"], rows,
                    align=["---", "---:", "---:"]), cov


def fit_check_table() -> tuple[str, pd.DataFrame]:
    fc = load_asset("cumulative_hospitalizations_by_age.csv")
    rows = [[str(a), f"{r.simulated_median:.1f}", f"{r.simulated_95pct_lo:.1f} – {r.simulated_95pct_hi:.1f}",
             f"{r.raw_data:.1f}", f"{r.pct_diff_median:.1f}%"] for a, r in fc.iterrows()]
    return md_table(["Age group", "Simulated (median)", "Simulated 95% interval", "Raw data",
                     "% difference (median)"], rows, align=["---"] + ["---:"] * 4), fc


def attack_rate_table() -> str:
    ar = pd.read_csv(MA / "transmission_components" / f"attack_rate_and_IHR_by_age_{TAG}.csv",
                     keep_default_na=False)
    rows = [[r["age group"].replace("0-0", "0").replace("all ages", "All"), r["attack rate, baseline"],
             r["attack rate, no vax"], r["effective IHR, unvaccinated"], r["effective IHR, vaccinated"]]
            for _, r in ar.iterrows()]
    return md_table(["Age group", "Attack rate, baseline", "Attack rate, no vaccination",
                     "Effective IHR, unvaccinated", "Effective IHR, vaccinated"], rows)


# ---- comparison with the earlier high-vax run -------------------------------------
def comparison_table() -> str:
    def row(label, name, r, c):
        return [label, T_OLD.cell(name, r, c), T.cell(name, r, c)]
    old_fp = pd.DataFrame(FIT_OLD["accepted_params"])
    fp = load_asset("fitted_params_summary.csv")
    old_da, new_da = T_OLD.csv("DOSE_ACCOUNTING"), T.csv("DOSE_ACCOUNTING")
    ar_old = pd.read_csv(MA / "transmission_components" / "attack_rate_and_IHR_by_age.csv")
    ar_old = ar_old[(ar_old["model"].str.contains("archived")) & (ar_old["age group"] == "all ages")].iloc[0]
    ar_new = pd.read_csv(MA / "transmission_components" / f"attack_rate_and_IHR_by_age_{TAG}.csv")
    ar_new = ar_new[ar_new["age group"] == "all ages"].iloc[0]
    rows = [
        ["Initially recovered (R)", "0%", "20% of every age group"],
        ["`beta_baseline`, posterior mean", f"{old_fp.beta_baseline.mean():.4f}", f"{fp.loc['beta_baseline'].posterior_mean:.4f}"],
        ["Initial-seed multiplier, posterior mean", f"{old_fp.seed_scale_E.mean():.2f}", f"{fp.loc['seed_scale_E'].posterior_mean:.2f}"],
        ["Attack rate, baseline (all ages, share of total population)", ar_old["attack rate, baseline"], ar_new["attack rate, baseline"]],
        ["Attack rate, no vaccination", ar_old["attack rate, no vax"], ar_new["attack rate, no vax"]],
        # the earlier run's S_A_1.csv predates the absolute-count columns; its
        # S.A.2 All/All cell is the same quantity (no vax -> full baseline)
        row("Hospitalizations averted by vaccination (S.A.1/S.A.2 total)", "S_A_2_absolute", "All", "All"),
        row("% of hospitalizations averted (S.A.1 total)", "S_A_1", "All", "pct_averted_total"),
        row("… via infection protection (% of no-vax burden)", "S_A_1", "All", "pct_averted_reduced_infection"),
        row("… via severity protection (% of no-vax burden)", "S_A_1", "All", "pct_averted_reduced_severity"),
        row("Averted per 100,000 scheduled doses", "S_A_1", "All", "per100k_doses_averted_total"),
        row("% averted, Low VE (S.A.5)", "S_A_5_pct_reduction", "All", "low_ve"),
        row("% averted, High VE (S.A.5)", "S_A_5_pct_reduction", "All", "high_ve"),
        row("Additional averted at 70% coverage floor (S.A.3 All)", "S_A_3_absolute", "All", "All"),
        ["Scheduled doses not delivered", old_da.loc["All", "pct_wasted"], new_da.loc["All", "pct_wasted"]],
        ["Delivered coverage (scheduled 55.9%)", old_da.loc["All", "delivered_coverage"], new_da.loc["All", "delivered_coverage"]],
    ]
    return md_table(["Quantity", "Earlier high-vax fit (no initial immunity)", "This fit (20% initially recovered)"],
                    rows, bold_last=False)


def main() -> None:
    cov_md, cov = coverage_table()
    fc_md, fc = fit_check_table()
    # cells quoted in the text get the same en-dash formatting as the tables
    def csv(t, name):
        return t.csv(name).map(_dash)
    s1, s1o = csv(T, "S_A_1"), csv(T_OLD, "S_A_1")
    s2d = csv(T, "S_A_2_per_100k_doses")
    s3d = csv(T, "S_A_3_per_100k_doses")
    s5p, s6a, s6p = csv(T, "S_A_5_pct_reduction"), csv(T, "S_A_6_absolute"), csv(T, "S_A_6_pct_reduction")
    da = T.csv("DOSE_ACCOUNTING")
    tv = FITCFG

    extra = np.maximum(0.0, 0.70 - cov.loc[AGES, "scheduled_coverage"].to_numpy()) * POP
    extra_desc = ", ".join(f"{extra[i]:,.0f} for age {AGES[i]}" if AGES[i] == "0" else f"{extra[i]:,.0f} for {AGES[i]}"
                           for i in range(len(AGES)) if extra[i] > 0)
    above = [AGES[i] for i in range(len(AGES)) if extra[i] == 0]
    below = [AGES[i] for i in range(len(AGES)) if extra[i] > 0]

    all_inside = all(r.simulated_95pct_lo <= r.raw_data <= r.simulated_95pct_hi for _, r in fc.iterrows())
    worst = fc.drop(index="All")["pct_diff_median"].abs().idxmax()

    wasted_pct = {a: float(da.loc[a, "pct_wasted"].rstrip("%")) for a in AGES}
    pct_r0 = 100 * R0.sum() / POP.sum()

    # peak / season totals (recomputed from the metric-timeseries CSV so the text matches the figure)
    from generic_core import results_io
    con = results_io.load_source(str(HERE / f"simulation_output_{TAG}_param_set_stochastic" / "results_parquet"))
    tot = {}
    for scen in ("baseline", "no vax"):
        df = con.execute("SELECT rep, day, SUM(value) v FROM results_full WHERE scenario=? AND "
                         "compartment IN ('I_to_H','IV_to_H') GROUP BY rep, day", [scen]).df()
        tot[scen] = df.pivot(index="rep", columns="day", values="v").to_numpy()
    con.close()
    med = {s: np.median(v, axis=0) for s, v in tot.items()}
    dates = pd.date_range("2025-09-01", periods=med["baseline"].size, freq="D")
    pk = {s: (med[s].max(), pd.Timestamp(dates[med[s].argmax()]).date()) for s in med}

    md = f"""# MA_vax, Aug-2026 high vaccination rates with 20% initially recovered: vaccination-impact report

Massachusetts, 2025–2026 influenza season. Age-structured SEIR model with a
parallel vaccinated arm, calibrated to daily hospital admissions by age
group via Bayesian MCMC. This report documents the model, the fit, and the
resulting vaccination-impact analysis for the variant that starts the season
with **20% of every age group already recovered (immune)**, run with the
archived August-2026 ("OLD high vax") vaccination schedule.

The model is otherwise identical to the earlier high-vax fit
(`model_config_2026-08-high-vax-rates.json`, reported in `../report.md`):
the same compartments, transitions, fixed parameters, contact matrices,
humidity and calendar inputs, vaccination schedule, fit targets and priors.
The only change to the model is the 20% initial `R`; the calibration was
re-run from scratch for it. §7 compares the two.

Files: `model_config_{TAG}.json`, `fit_config_{TAG}.json`,
`fitted_params_{TAG}.json`,
`run_simulations_{TAG}_param_set_stochastic.py`, and the outputs listed in §8.

---

## 1. Model structure

{MODEL_1_1_TO_1_4}

**Vaccination and the initially recovered.** The vaccination pool is `S + SV`,
so the 20% seeded into `R` never receive a dose in the model. The schedule's
proportions are still applied to `S + SV` only, so the model delivers about 80%
of the scheduled doses to people who can benefit. In the real world the other
~20% of doses still go to people who happen to be immune already, and buy no
protection. This is the same as vaccinating without regard to prior immunity.
It is why about a fifth of the scheduled doses show up as "not delivered" in
the dose-accounting appendix, and why every "per 100,000 doses" figure is
lower than in the earlier fit.

### 1.5 Numerical scheme

Explicit Euler integration with 7 sub-steps per day. The calibration and
every simulation in this report use deterministic transitions (§3). The
outflow groups from `I`, `IV` and `H` are declared as `multinom` groups in
this config (vs. `multinom_deterministic` in the earlier config). That label
is informational only: at run time every group uses the simulation-wide
transition type, and the two configs give bit-identical trajectories.
Vaccination is a deterministic scheduled count in all cases.

### 1.6 Initial conditions

At the start of the simulation (2025-09-01): `R = 20%` of each age group's
population (prior immunity, {R0.sum():,.0f} people in total, {pct_r0:.0f}% of the state),
`E = E0` (age-specific seed counts, scaled by a single fitted multiplier,
§2.3), `S = population − E0 − R0`, and all other compartments start at zero.

---

## 2. Parameters

### 2.1 Fixed scalar parameters

{fixed_scalar_table()}

### 2.2 Age-stratified fixed parameters

{age_param_table()}

¹ These are pre-fit baseline values. The calibration scales both
hospitalization-risk columns by a fitted age-specific multiplier (§2.3), and
the seed counts by a single multiplier, so the values actually used are
these figures × those multipliers. The ratio of the two hospitalization-risk
columns, i.e. vaccine effectiveness against hospitalization given infection,
is unchanged by the scaling. A residual susceptibility of 1.00 for 65+ means
the fitted baseline assumes **no infection-blocking effect** of vaccination
in that age group (only the severity effect applies there).

### 2.3 Fitting method

Free parameters were estimated with an affine-invariant ensemble Markov
Chain Monte Carlo sampler (emcee; {tv['n_iter']} iterations per walker, 58 walkers — the
fitter raises the configured {tv['n_walkers']} to twice the 29 sampled dimensions), run in
parallel. The free parameters and their priors:

| Parameter | Prior | Role |
|---|---|---|
| `beta_baseline` | Uniform({tv['bounds']['beta_baseline'][0]}, {tv['bounds']['beta_baseline'][1]}) | Baseline transmission rate |
| `humidity_impact` | Uniform({tv['bounds']['humidity_impact'][0]}, {tv['bounds']['humidity_impact'][1]}) | Strength of humidity forcing |
| Initial-seed multiplier | Log-uniform({tv['bounds']['seed_scale_E'][0]}, {tv['bounds']['seed_scale_E'][1]}) | Multiplies the age-specific initial-infection seed counts |
| Hospitalization-risk multiplier (× 7, one per age group) | Uniform({tv['bounds']['IHR_scale'][0]}, {tv['bounds']['IHR_scale'][1]}) | Multiplies both hospitalization-risk columns (§2.2) for that age group |
| `m(t)` log-increments (× 18, one per {tv['tv_knot_spacing_days']}-day knot) | Normal(0, {tv['tv_tau']}) | Random-walk steps in log-transmission (§1.2) |
| `phi` | — | Negative-Binomial dispersion (likelihood nuisance parameter, not a model input) |

**Likelihood**: a Negative-Binomial (NB2) observation model jointly across
8 targets — daily hospital admissions **by age group** (7 time series) plus a
single scalar **end-of-season cumulative hospitalizations by age** target.
The vaccination schedule used during the fit is the archived "OLD high vax"
schedule, the same one the simulations use.

**Posterior sampling**: the first {tv['mcmc_burnin']} iterations were discarded as
burn-in, and the remaining chain thinned to every {tv['mcmc_thin']}th sample, leaving
{N_DRAWS} posterior draws (no walkers dropped). Two point estimates are reported:
the **posterior mean** (marginal mean of each parameter) and the **"best"
point** (the single draw with the highest log-posterior, which keeps the
correlation between parameters).

### 2.4 Fitted parameters

Posterior mean and 90% credible interval (5th–95th percentile) across the {N_DRAWS}
posterior draws, the "best" point, and for reference the posterior mean of
the earlier high-vax fit (no initial immunity):

{fitted_params_table()}

Starting with a fifth of the population immune leaves fewer susceptibles, so
the fit compensates with somewhat more transmission (`beta_baseline` up ~7%)
and a larger initial seed (multiplier up ~30%). The age-specific
hospitalization-risk multipliers barely move (all within 0.13 of the earlier
fit), because the admissions they are fitted to are the same. The 18
`m(t)` log-increments are shown in the transmission-components figure (§6)
rather than tabulated.

### 2.5 Cumulative vaccination coverage

Cumulative proportion of each age group vaccinated over the season,
according to the vaccination schedule (summed over the 250-day window):

{cov_md}

Four groups sit below a 70% mark ({", ".join(below)}); {", ".join(above)} are above it.
This matters for the "scale to 70% coverage" rows in the appendix (Tables S.A.3/S.A.6):
only the four groups below the mark are scaled up.

Because the 20% initially recovered are outside the vaccination pool (§1),
the doses the model actually delivers are much lower than scheduled:
{da.loc['All', 'delivered_coverage']} delivered coverage overall vs. {da.loc['All', 'scheduled_coverage']} scheduled
({da.loc['All', 'pct_wasted']} of scheduled doses not delivered, vs. 3.3% in the earlier fit).
About 20 points of that gap come from prior immunity. The rest, as before, is
the §1.4 cap skipping people infected earlier in the season. Every
"per 100,000 doses" panel divides by the **scheduled** count (see the appendix).

---

## 3. Vaccination-impact results

Every table in this section reports a **median and 95% interval across {N_DRAWS}
simulations**: one deterministic simulation per posterior parameter draw
(§2.3), re-using the same draw for every scenario compared within a table so
that the comparison is paired. The intervals reflect **calibration
uncertainty**, not epidemic-process noise.

### New daily hospitalizations: baseline vs. no vaccination

Total-population new hospitalizations per day, posterior median and 95%
interval across the {N_DRAWS} parameter draws:

![Baseline vs. no vaccination]({ASSETS}/baseline_vs_no_vaccination_daily_H.png)

The fitted vaccination program cuts the peak in daily new hospitalizations
to under a third of what it would otherwise be (median peak ≈ {pk['baseline'][0]:.0f}/day
baseline vs. ≈ {pk['no vax'][0]:.0f}/day with no vaccination, both on {pk['baseline'][1]}), and
the season total from a median of {np.median(tot['no vax'].sum(1)):,.0f} to {np.median(tot['baseline'].sum(1)):,.0f} admissions.
It reduces the epidemic's height without changing its timing.

### Table S.A.1 — Hospitalizations averted, infection vs. severity protection

Decomposes the total hospitalizations averted by vaccination into two
channels: protection against getting infected at all, and — *given* a
breakthrough infection still happens — protection against that infection
becoming severe enough to need hospitalization.

**Infection protection** (no vaccination → infection-protection-only, i.e. VE against infection retained but VE against severity zeroed out)

{T.sa1("reduced_infection")}

**Severity protection** (infection-protection-only → full baseline, i.e. adding back VE against severity)

{T.sa1("reduced_severity")}

**Total** (no vaccination → full baseline)

{T.sa1("total")}

As in the earlier fit, almost all of the averted burden comes from
**blocking infection** ({s1.loc['All', 'averted_reduced_infection']} of {s1.loc['All', 'averted_total']}), not from
reducing severity given a breakthrough ({s1.loc['All', 'averted_reduced_severity']}, nearly all in 65+).
The total is somewhat smaller than in the earlier fit ({T_OLD.cell('S_A_2_absolute', 'All', 'All')}):
with a fifth of the population already immune, the no-vaccination epidemic is
smaller, so there is less for vaccination to avert. Per scheduled dose the
drop is larger ({s1.loc['All', 'per100k_doses_averted_total']} vs. {s1o.loc['All', 'per100k_doses_averted_total']} per 100,000),
because the same doses are spread over a population in which ~20% gain nothing from them.

### Table S.A.2 — Hospitalizations averted by age group vaccinated

Each column vaccinates a single age group only (all others left
unvaccinated) and compares to no vaccination at all; "All" is the full
baseline schedule. Rows are the age group in which hospitalizations are
counted, so off-diagonal cells show the indirect (transmission-blocking)
benefit to *other* age groups from vaccinating this one.

**Hospitalizations averted (count)**

{T.age_matrix("S_A_2_absolute", absolute=True)}

**% reduction in hospitalizations**

{T.age_matrix("S_A_2_pct_reduction")}

**Hospitalizations averted per 100,000 population**

{T.age_matrix("S_A_2_per_100k")}

**Hospitalizations averted per 100,000 doses**

Off-diagonal cells divide by the doses scheduled for the **column's** age group
(the only group vaccinated in that scenario). The `All` column keeps each
row's own dose count, matching Table S.A.1.

{T.age_matrix("S_A_2_per_100k_doses")}

Per dose, the indirect benefit is dominated by what it does for **65+**:
vaccinating 13-17 averts {s2d.loc['65+', '13-17']} hospitalizations per 100,000 doses
in the 65+ group alone, against {s2d.loc['13-17', '13-17']} in 13-17 itself. This is
why the `All` row ranks 13-17 ({first(s2d.loc['All', '13-17']):.1f}) and 5-12
({first(s2d.loc['All', '5-12']):.1f}) far above the groups that carry the burden directly.
65+ is the only group with zero indirect effect on every other group, since
its vaccine has no infection-blocking effect (§2.2).

### Table S.A.4 — Vaccine-effectiveness sensitivity scenarios

Implied vaccine effectiveness under three VE presets — a parameter table, not
a simulation result, and identical to the earlier fit (VE parameters are
not fitted). `Baseline VE (fitted)` is the reference point; `Low VE`/`High VE`
bracket it.

{T.sa4()}

65+ has 0% VE against infection in the baseline and in `Low VE` (residual
susceptibility 1.00), so only the severity channel operates there; `High VE`
scales it to 0.86 (14% VE against infection).

### Table S.A.5 — Hospitalizations averted across VE scenarios

Compares each VE sensitivity scenario against no vaccination at all.

**Hospitalizations averted (count)**

{T.ve_table("S_A_5_absolute", absolute=True)}

**% reduction in hospitalizations**

{T.ve_table("S_A_5_pct_reduction")}

**Hospitalizations averted per 100,000 population**

{T.ve_table("S_A_5_per_100k")}

Even under the pessimistic `Low VE` assumption, the vaccination schedule
still averts {s5p.loc['All', 'low_ve'].split(' ')[0]} of hospitalizations overall ({s5p.loc['All', 'high_ve'].split(' ')[0]} under `High VE`).

---

## 4. Baseline fit check — posterior-uncertainty simulation vs. raw data

The baseline scenario, one deterministic simulation per posterior draw (the
same {N_DRAWS} runs used in §3), against the raw data it was fit to.

### Cumulative hospitalizations, by age group

Simulated (per-draw season total, median and 95% interval) vs. raw daily
hospital admissions, summed over the simulation window
(2025-09-01 – 2026-05-08):

{fc_md}

The fit tracks the data closely overall ({fc.loc['All', 'pct_diff_median']:.1f}% on the total){"," if all_inside else ";"}
{"with the raw value inside the 95% interval for every age group" if all_inside else "the raw value falls outside the 95% interval for at least one age group"}.
The largest relative miss is in age {worst} ({fc.loc[worst, 'pct_diff_median']:.1f}%, only
{abs(fc.loc[worst, 'simulated_median'] - fc.loc[worst, 'raw_data']):.0f} admissions off in absolute terms).

### Daily new hospitalizations by age group

![Daily fit check by age]({ASSETS}/fit_check_daily_by_age.png)

### Cumulative hospitalizations by age group

![Cumulative fit check by age]({ASSETS}/fit_check_cumulative_by_age.png)

The same comparison in the layout of the other fits' `fit_comparison_output_*`
folders is in `../../fit_comparison_output_{TAG}/`. Its intervals are
narrower than `fit_comparison_output_baseline/`, because that folder was built
from a run with stochastic transitions; this one reflects parameter
uncertainty only.

---

## 5. Attack rate and effective IHR by age group

Median [95% interval]. Attack rate = share of the age group infected over the
season, `(N − R(0) − S(T) − SV(T)) / N`: it includes seeded infections and
**excludes the 20% seeded as recovered**, and is expressed as a share of the
**whole** age group. Divide by 0.8 for the share of those who started out
susceptible. Effective IHR = hospitalization-risk multiplier × `I_to_H_prop`
(unvaccinated) or × `IV_to_H_prop` (vaccinated), across the posterior draws.

{attack_rate_table()}

---

## 6. Transmission components

![Transmission components]({"../../transmission_components/transmission_components_" + TAG + ".png"})

Posterior median and 95% band of `m(t)`, the humidity term, `beta_adjusted`,
`beta_eff` (with population susceptibility from vaccination), the calendar
effect and the calendar-adjusted `beta_eff`. As noted on the figure, `beta_eff`
accounts for vaccination only: it includes neither depletion from infection
nor the 20% initially recovered. The time series are in
`../../transmission_components/transmission_components_{TAG}.csv`.

---

## 7. Comparison with the earlier high-vax fit (no initial immunity)

Same vaccination schedule, VE assumptions and fit targets; the only
differences are the 20% initial `R` and the refit it required.

{comparison_table()}

Both fits reproduce the observed admissions comparably well (season total
within 2% of the data in both), so the data alone don't pick between them. The impact estimates are fairly robust to the
initial-immunity assumption in relative terms (% averted changes by about 2
points). Per-dose figures drop by about 10%, because a fifth of the doses land
on people who are already immune.

---

## 8. Data sources and outputs

| Input | Source |
|---|---|
| Age-specific vaccination coverage | MIDAS Flu Scenario Modeling Hub resources, age-specific coverage dataset (archived Aug-2026 version, `MA_flu_daily_vaccinations_proportions_array - OLD high vax.csv`) |
| Hospital admissions (calibration target) | MIDAS Flu Scenario Modeling Hub, target-data time series |
| Population by age group | US Census (`tidycensus`) |
| Contact matrices | Mistry et al. 2021 synthetic contact matrices |
| Absolute humidity | gridMET daily specific-humidity data, averaged over Massachusetts |
| School/work calendar | Constructed school/work-day calendar for the state, for each day of the season |

| Output | Produced by |
|---|---|
| `fitted_params_{TAG}.json` | `run_fitting_{TAG}.py` |
| `simulation_output_{TAG}_param_set_stochastic/` | `run_simulations_{TAG}_param_set_stochastic.py` |
| `counterfactual_tables_from_db_{TAG}_param_set_stochastic/` | `../../build_counterfactual_tables_from_db.py` |
| `{TAG}_param_set_stochastic__metric_timeseries.csv`, `../../fit_comparison_output_{TAG}/` | `../../export_metric_timeseries_MA_vax.py`, `../../plot_fit_vs_actual.py` |
| `../../transmission_components/*_{TAG}.*` | `../../plot_transmission_components_MA_vax.py --only {TAG}` |
| `{ASSETS}/`, this report | `build_report_assets_{TAG}.py`, `build_report_{TAG}.py` |

---

## 9. Notes

- **The confidence intervals reflect calibration uncertainty, not
  epidemic-process noise**: one deterministic run per posterior draw.
- **`m(t)` is a statistical smoothing device, not a mechanistic term**: it
  absorbs whatever transmission variation contacts and humidity don't explain.
- **The 20% initial immunity is an assumption, not an estimate**: it is fixed
  in the config, not fitted, and the data fit equally well with or without it
  (§7). Treat the differences in §7 as sensitivity to this assumption.
- **The "scale to 70% coverage" scenarios treat 70% as a floor, not a
  quota**: {", ".join(above)} are already above it and keep their baseline schedule,
  contributing all-zero columns. Coverage here is scheduled coverage of the
  whole age group, including the initially recovered.
- **"Per 100,000 doses" counts scheduled doses, not delivered doses**:
  {da.loc['All', 'pct_wasted']} of scheduled doses are not delivered in the model, mostly
  because they would go to people who are already immune (see the appendix).

---

## Appendix: Tables S.A.3 and S.A.6

### Table S.A.3 — Additional hospitalizations averted at 70% coverage

Each column scales a single age group's vaccination schedule up to 70%
cumulative scheduled coverage; "All" scales every eligible age group. Compared
against the baseline vaccination scenario. 70% is a **floor**: the four groups
below it ({", ".join(below)}) are scaled up, and the three already above it
({", ".join(above)}) keep their baseline schedule and give exact-zero columns.

**Hospitalizations averted (count)**

{T.age_matrix("S_A_3_absolute", absolute=True)}

**% reduction in hospitalizations**

{T.age_matrix("S_A_3_pct_reduction")}

**Hospitalizations averted per 100,000 population**

{T.age_matrix("S_A_3_per_100k")}

**Hospitalizations averted per 100,000 additional doses**

Denominators are the *additional* doses each scenario schedules,
`max(0, 70% − scheduled coverage) × population`: {extra_desc}, and zero for the
three groups already above 70% (shown as `—`). The `All` column's denominator
is the total {extra.sum():,.0f} additional doses. As with the baseline, ~20% of
these extra doses reach people who are already immune.

{T.age_matrix("S_A_3_per_100k_doses", zero_as_dash=True)}

Per additional dose, raising **13-17** to 70% is by far the best buy:
{s3d.loc['All', '13-17']} hospitalizations averted per 100,000 doses, against
{first(s3d.loc['All', '0']):.1f} for age 0 and {first(s3d.loc['All', '18-49']):.1f} for 18-49, and
{first(s3d.loc['65+', '13-17']):.1f} of that lands in **65+**, not in 13-17 itself. 18-49 dominates the
*absolute* totals only because it absorbs {extra[AGES.index('18-49')]:,.0f} of the {extra.sum():,.0f} additional doses.

### Table S.A.6 — Additional hospitalizations averted at 70% coverage, across VE scenarios

For each VE sensitivity scenario, compares that scenario's own baseline
vaccination to the 70%-coverage floor applied to every eligible age group.

**Hospitalizations averted (count)**

{T.ve_table("S_A_6_absolute", absolute=True)}

**% reduction in hospitalizations**

{T.ve_table("S_A_6_pct_reduction")}

**Hospitalizations averted per 100,000 population**

{T.ve_table("S_A_6_per_100k")}

The 70% floor averts {s6a.loc['All', 'baseline_ve'].split(' [')[0]} additional hospitalizations under the
fitted VE ({s6p.loc['All', 'baseline_ve'].split(' ')[0]} of the remaining burden), {s6a.loc['All', 'low_ve'].split(' [')[0]} under `Low VE`
and {s6a.loc['All', 'high_ve'].split(' [')[0]} under `High VE`. The count is lower under `High VE`
than under the fitted VE because the baseline schedule has already
prevented more of the burden, even though the percentage reduction is larger.

---

## Appendix: dose accounting — scheduled vs. delivered doses

**Scheduled** doses are what the vaccination schedule reports: the daily
proportions of §2.5, summed over the season and multiplied by the whole
age group's population. **Delivered** doses are what the model records as an
actual `S → SV` transition. The gap has two sources here:

1. **Prior immunity** (new in this fit): the daily proportion is applied to
   `S + SV`, which excludes the 20% seeded into `R`. Those people's share of
   the doses is never delivered. In the real world they are vaccinated like
   anyone else, and the dose buys no protection.
2. **Infection during the season** (as in the earlier fit): the §1.4 cap
   skips people infected before their dose would have arrived.

Both are real doses that are bought and administered, so every "per
100,000 doses" figure divides by the scheduled count.

Baseline schedule, median across the {N_DRAWS} posterior draws:

{T.dose_accounting()}

{da.loc['All', 'pct_wasted']} of scheduled doses are not delivered overall (3.3% in the earlier fit),
ranging from {min(wasted_pct.values()):.1f}% to {max(wasted_pct.values()):.1f}% by age group. About 20 points
of that are prior immunity. The remainder tracks the attack rate, as before:
highest in 18-49 and 50-64.

---

## Appendix: vaccine-efficacy mechanism check

Flow-level rate ratios (vaccinated rate / unvaccinated rate, so **lower
means more protection**) from the "Vaccinate `<age>` only" scenarios
(diagonal) and the full baseline (`All`). They check that the simulated
vaccinated and unvaccinated arms behave as the parameters imply. They depend
on the VE parameters, not on the fit, and are within ~1 point of the earlier fit.
(The CSVs are named `VAX_CHECK_*_reduction.csv`, but every entry is a ratio.)

**Naive infection attack-rate ratio** (whole season: `SV_to_EV / (SV(0) + S_to_SV)`
vs. `S_to_E / (S(0) − S_to_SV)`). This is biased, because it mixes people
vaccinated before and after the epidemic peak; it is shown as the naive-analysis
comparison point.

{T.vax_check("infection_reduction")}

**Matched-cohort infection attack-rate ratio**, the estimate comparable to
real-world VE. It is ≈ `vax_susceptibility` (0.57 / 0.79 / 1.00) as expected.

{T.vax_check("matched_cohort_infection_reduction")}

**Hospitalization-given-infection rate ratio** (`IV_to_H / SV_to_EV` vs. `I_to_H / S_to_E`),
≈ `IV_to_H_prop / I_to_H_prop` (0.91 for ages under 65, 0.69 for 65+).

{T.vax_check("hospitalization_reduction")}
"""
    OUT.write_text(md)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
