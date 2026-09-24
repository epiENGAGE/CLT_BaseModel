"""Side-by-side comparison of the four MA_vax fits (7-age / single-age, with
and without the m(t) transmission multiplier).

Writes a tidy CSV and a markdown report covering attack rate, fitted vs
observed hospitalizations, IHR scaling and the derived quantities that
explain why the fits disagree.

Posterior draws come from each fit's accepted_params; per-draw trajectories
come from the matching *_param_set_stochastic run's `baseline` scenario.
Requires effective_beta_MA_vax.csv (run compute_effective_beta.py first) for
the R_eff / beta_eff rows.

Usage: python compare_fits.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/rfp437/Work/CityLevelTransmission/CLT_BaseModel")
MA = ROOT / "generic_core/examples/MA_vax"
SA = MA / "MA_vax_single_age"
SEED_BASE = 42                      # see compute_effective_beta.py
BETA_CSV = MA / "effective_beta_MA_vax.csv"
OUT_CSV = MA / "fit_comparison_MA_vax.csv"
OUT_MD = MA / "fit_comparison_MA_vax.md"

CASES = [
    ("7-age, m(t)",    MA / "model_config_MA_vax.json",        MA / "fitted_params_MA_vax.json",
     MA / "simulation_output_MA_vax_param_set_stochastic"),
    ("7-age, no m(t)", MA / "model_config_MA_vax.json",        MA / "fitted_params_MA_vax_no_transmission_multiplier.json",
     MA / "simulation_output_MA_vax_no_transmission_multiplier_param_set_stochastic"),
    ("1-age, m(t)",    SA / "model_config_MA_single_age.json", SA / "fitted_params_MA_single_age.json",
     SA / "simulation_output_single_age"),
    ("1-age, no m(t)", SA / "model_config_MA_single_age.json", SA / "fitted_params_MA_single_age_no_transmission_multiplier.json",
     SA / "simulation_output_single_age_no_transmission_multiplier"),
]

Q = (2.5, 50, 97.5)


def qs(x, axis=0):
    """(median, lo, hi) over posterior draws."""
    lo, med, hi = np.nanpercentile(x, Q, axis=axis)
    return med, lo, hi


def fmt(med, lo, hi, d=1):
    return f"{med:,.{d}f} [{lo:,.{d}f} – {hi:,.{d}f}]"


def load_case_arrays(sim_dir, comps, num_days, n_age, n_draws):
    """{comp: (n_draws, num_days+1, n_age)} from the baseline scenario,
    reindexed from parquet rep order to accepted_params order (the run script
    draws its sets as a shuffled permutation seeded with SEED_BASE)."""
    base = Path(sim_dir) / "results_parquet/results_full/scenario=baseline"
    perm = np.random.default_rng(SEED_BASE).choice(n_draws, size=n_draws, replace=False)
    out = {}
    for comp in comps:
        files = sorted((base / f"compartment={comp}").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{base}/compartment={comp}")
        df = pd.concat([pd.read_parquet(f) for f in files])
        arr = np.zeros((n_draws, num_days, n_age))
        arr[df["rep"].to_numpy(int), df["day"].to_numpy(int), df["age_group"].to_numpy(int)] = df["value"].to_numpy()
        ordered = np.empty_like(arr)
        ordered[perm] = arr
        out[comp] = ordered
    return out


rows = []          # tidy output
report = {}        # per-case dict used to build the markdown

for label, cfg_path, fit_path, sim_dir in CASES:
    cfg = json.loads(cfg_path.read_text())
    fit = json.loads(fit_path.read_text())
    prm = cfg["params"]
    num_days = int(fit["num_days"])
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], float).ravel()
    N = pop.sum()
    w = pop / N
    ages = cfg["age_risk"]["age_groups"]
    n_age = len(ages)
    acc = fit["accepted_params"]
    n_draws = len(acc)

    sim = load_case_arrays(sim_dir, ("S", "SV", "I_to_H", "IV_to_H"), num_days, n_age, n_draws)

    # --- attack rate: everyone who ever left S/SV by infection ---
    ever_inf = pop[None, :] - sim["S"][:, -1, :] - sim["SV"][:, -1, :]
    ar_age = ever_inf / pop[None, :]
    ar_tot = ever_inf.sum(1) / N

    # --- hospitalizations over the season (both arms) ---
    H_age = (sim["I_to_H"] + sim["IV_to_H"])[:, 1:, :].sum(1)
    H_tot = H_age.sum(1)
    H_daily = (sim["I_to_H"] + sim["IV_to_H"]).sum(2)

    # observed targets: the 'ts' entries, in age order (single scalar for 1-age)
    obs = []
    for o, mode in zip(fit["observed"], fit["target_modes"]):
        if mode != "ts":
            continue
        obs.append(np.array([np.nan if v is None else float(v) for v in o]))
    obs_age = np.array([np.nansum(o) for o in obs])          # (n_age,) or (1,)
    obs_daily = np.nansum(np.stack(obs), axis=0)

    # --- IHR: fitted scale x baseline proportion ---
    scale_keys = ([f"IHR_scale|a{i}" for i in range(n_age)] if n_age > 1 else ["IHR_scale"])
    scales = np.array([[q[k] for k in scale_keys] for q in acc])      # (n_draws, n_age)
    ihr_base = np.asarray(prm["I_to_H_prop"], float).ravel()
    ihrv_base = np.asarray(prm["IV_to_H_prop"], float).ravel()
    ihr = scales * ihr_base[None, :]
    ihrv = scales * ihrv_base[None, :]
    # population-weighted "all ages" IHR, and the infection-weighted one that
    # actually governs the hospitalization total
    ihr_pop = ihr @ w
    ihr_infwtd = (ihr * ever_inf).sum(1) / ever_inf.sum(1)
    scale_pop = scales @ w

    # infections per hospitalization
    inf_per_hosp = ever_inf.sum(1) / H_tot

    per_age = {}
    for i, a in enumerate(ages):
        per_age[a] = dict(
            attack_rate=qs(ar_age[:, i]),
            hosp=qs(H_age[:, i]),
            hosp_obs=obs_age[i] if i < len(obs_age) else np.nan,
            ihr_scale=qs(scales[:, i]),
            ihr=qs(ihr[:, i]),
            ihr_vax=qs(ihrv[:, i]),
        )
    report[label] = dict(
        cfg=cfg, fit=fit, ages=ages, pop=pop, N=N, n_draws=n_draws,
        per_age=per_age,
        total=dict(
            attack_rate=qs(ar_tot),
            hosp=qs(H_tot),
            hosp_obs=float(np.nansum(obs_age)),
            ihr_scale=qs(scale_pop),
            ihr=qs(ihr_pop),
            ihr_infwtd=qs(ihr_infwtd),
            inf_per_hosp=qs(inf_per_hosp),
            peak_daily_H=qs(H_daily.max(1)),
            peak_day=qs(H_daily.argmax(1).astype(float)),
            obs_peak=float(np.nanmax(obs_daily)),
        ),
        scalars={k: qs(np.array([q[k] for q in acc]))
                 for k in ("beta_baseline", "humidity_impact", "seed_scale_E", "phi")},
        start=cfg["simulation_settings"]["start_real_date"],
        num_days=num_days,
    )

    # tidy rows
    for a, d in per_age.items():
        for metric, key in [("attack_rate", "attack_rate"), ("hospitalizations", "hosp"),
                            ("IHR_scale", "ihr_scale"), ("IHR_unvaccinated", "ihr"),
                            ("IHR_vaccinated", "ihr_vax")]:
            med, lo, hi = d[key]
            rows.append(dict(case=label, age_group=a, metric=metric,
                             median=med, lo=lo, hi=hi,
                             observed=d["hosp_obs"] if metric == "hospitalizations" else np.nan))
    t = report[label]["total"]
    for metric, key in [("attack_rate", "attack_rate"), ("hospitalizations", "hosp"),
                        ("IHR_scale", "ihr_scale"), ("IHR_unvaccinated", "ihr"),
                        ("IHR_infection_weighted", "ihr_infwtd"),
                        ("infections_per_hospitalization", "inf_per_hosp"),
                        ("peak_daily_hospitalizations", "peak_daily_H")]:
        med, lo, hi = t[key]
        rows.append(dict(case=label, age_group="All", metric=metric, median=med, lo=lo, hi=hi,
                         observed=t["hosp_obs"] if metric == "hospitalizations" else np.nan))
    for k, (med, lo, hi) in report[label]["scalars"].items():
        rows.append(dict(case=label, age_group="All", metric=k, median=med, lo=lo, hi=hi, observed=np.nan))

    print(f"{label:16s} draws={n_draws:4d}  attack={t['attack_rate'][0]:.3f}  "
          f"H={t['hosp'][0]:,.0f} (obs {t['hosp_obs']:,.0f})  "
          f"inf/hosp={t['inf_per_hosp'][0]:.0f}")

# --- R_eff / beta_eff rows from compute_effective_beta.py's output ----------
beta = None
if BETA_CSV.exists():
    beta = pd.read_csv(BETA_CSV, header=[0, 1], index_col=0, parse_dates=[0])
    for label in report:
        if label not in beta.columns.levels[0]:
            continue
        s = beta[label]
        extras = {
            "R_eff_peak": s["R_eff_med"].max(),
            "R_eff_peak_date": s["R_eff_med"].idxmax(),
            "R_eff_season_end": s["R_eff_med"].iloc[-1],
            "R_v_season_end": s["R_v_med"].iloc[-1],
            "R_eff_over_R_v_season_end": s["R_eff_med"].iloc[-1] / s["R_v_med"].iloc[-1],
            "beta_eff_season_mean": s["beta_eff_best"].mean(),
        }
        report[label]["extras"] = extras
        for k, v in extras.items():
            rows.append(dict(case=label, age_group="All", metric=k,
                             median=(v if not isinstance(v, pd.Timestamp) else v.strftime("%Y-%m-%d")),
                             lo=np.nan, hi=np.nan, observed=np.nan))
else:
    print(f"NOTE: {BETA_CSV.name} not found -- R_eff rows skipped "
          "(run compute_effective_beta.py first)")

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print("wrote", OUT_CSV)

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
L = list(report)
AGE_CASES = [k for k in L if report[k]["ages"] != ["0+"]]
ages7 = report[AGE_CASES[0]]["ages"] if AGE_CASES else []


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def header(cells):
    return row(cells) + "\n" + row(["---"] * len(cells))


def pct(med, lo, hi):
    return f"{med:.1%} [{lo:.1%} – {hi:.1%}]"


def pct3(med, lo, hi):
    return f"{med:.3%} [{lo:.3%} – {hi:.3%}]"


md = []
md.append("# MA_vax: comparing the four fits\n")
md.append("Massachusetts 2025–26 influenza season. Four calibrations of the same "
          "model against the same hospital-admission data:\n")
md.append("- **7-age** vs **1-age**: the age-structured model (7 groups) vs the "
          "single-group model built by population-weighting its inputs.\n"
          "- **m(t)** vs **no m(t)**: with and without the fitted time-varying "
          "transmission multiplier (18 log-increments on 14-day knots).\n")
md.append("All intervals are the median and 2.5–97.5 percentile across each fit's "
          "posterior draws (one simulation per accepted parameter set, `baseline` "
          "vaccination scenario).\n")

md.append("\n## 1. Headline comparison\n")
cols = ["Quantity"] + L
lines = [header(cols)]


def add(name, fn):
    lines.append(row([name] + [fn(report[k]) for k in L]))


add("Posterior draws", lambda r: f"{r['n_draws']}")
add("**Infection attack rate**", lambda r: pct(*r["total"]["attack_rate"]))
add("Cumulative hospitalizations (simulated)", lambda r: fmt(*r["total"]["hosp"], 0))
add("Cumulative hospitalizations (observed)", lambda r: f"{r['total']['hosp_obs']:,.0f}")
add("Difference vs observed", lambda r: f"{r['total']['hosp'][0] / r['total']['hosp_obs'] - 1:+.1%}")
add("**Infections per hospitalization**", lambda r: fmt(*r["total"]["inf_per_hosp"], 0))
add("IHR, population-weighted", lambda r: pct3(*r["total"]["ihr"]))
add("IHR, infection-weighted", lambda r: pct3(*r["total"]["ihr_infwtd"]))
add("IHR_scale, population-weighted", lambda r: fmt(*r["total"]["ihr_scale"], 2))
add("Peak daily hospitalizations (simulated)", lambda r: fmt(*r["total"]["peak_daily_H"], 0))
add("Peak daily hospitalizations (observed)", lambda r: f"{r['total']['obs_peak']:,.0f}")
add("`beta_baseline`", lambda r: fmt(*r["scalars"]["beta_baseline"], 4))
add("`humidity_impact`", lambda r: fmt(*r["scalars"]["humidity_impact"], 2))
add("`seed_scale_E`", lambda r: fmt(*r["scalars"]["seed_scale_E"], 2))
add("`phi` (NB dispersion)", lambda r: fmt(*r["scalars"]["phi"], 0))
if all("extras" in report[k] for k in L):
    add("Peak R_eff", lambda r: f"{r['extras']['R_eff_peak']:.2f}")
    add("Peak R_eff date", lambda r: r["extras"]["R_eff_peak_date"].strftime("%Y-%m-%d"))
    add("R_eff, season end", lambda r: f"{r['extras']['R_eff_season_end']:.2f}")
    add("R_v, season end (no infection-acquired immunity)",
        lambda r: f"{r['extras']['R_v_season_end']:.2f}")
    add("**R_eff / R_v, season end** (share of R left after depletion)",
        lambda r: f"{r['extras']['R_eff_over_R_v_season_end']:.2f}")
    add("Season-mean effective beta (per contact per day)",
        lambda r: f"{r['extras']['beta_eff_season_mean']:.4f}")
md.append("\n".join(lines) + "\n")
md.append("""
*Two IHR averages appear above and they answer different questions.*
**Population-weighted** IHR is `Σ_a (N_a/N)·IHR_a` — the risk faced by a
randomly chosen *resident*. It depends only on the fitted parameters and the
age distribution, not on the epidemic, and in an age-structured model it is
dominated by 65+ (1.70 of the 7-age m(t) fit's 2.10 percentage points come from
that one group). **Infection-weighted** IHR is `Σ_a (infections_a/infections)·IHR_a`
— the risk carried by a randomly chosen *infection*, i.e. the number that
multiplied by total infections gives total hospitalizations.

They differ whenever infections are not distributed like the population. In the
7-age fits, 65+ takes 17.5% of the population but only ~10.6% of infections
(low contact rates), while 5-12 takes 8.7% of the population and ~10.4% of
infections — so the infection-weighted average sits well below the
population-weighted one (1.44% vs 2.10%). In the single-age model there is no
age structure, so the two are identical by construction. **Across models,
compare the infection-weighted row**: it is 1.440% (7-age) against 1.438%
(1-age), which is exactly the agreement you would expect from two fits matching
the same hospitalizations with the same attack rate. The population-weighted
row differing (2.10% vs 1.44%) is a statement about demography and where the
fit put the age-specific scaling, not a disagreement about severity.
""")

md.append("\n## 2. Infection attack rate\n")
md.append("Ever-infected fraction at the end of the season, `(N − S − SV) / N`.\n")
lines = [header(["Age group"] + L)]
for i, a in enumerate(ages7):
    lines.append(row([a] + [pct(*report[k]["per_age"][a]["attack_rate"]) if k in AGE_CASES else "—"
                            for k in L]))
lines.append(row(["**All**"] + [f"**{pct(*report[k]['total']['attack_rate'])}**" for k in L]))
md.append("\n".join(lines) + "\n")

md.append("\n## 3. Hospitalizations: fitted vs observed\n")
md.append("Cumulative new hospital admissions over the season "
          "(`I_to_H + IV_to_H`), against the calibration target.\n")
lines = [header(["Age group"] + [f"{k} (simulated)" for k in L] + ["Observed"])]
for a in ages7:
    obs = report[AGE_CASES[0]]["per_age"][a]["hosp_obs"] if AGE_CASES else np.nan
    lines.append(row([a]
                     + [fmt(*report[k]["per_age"][a]["hosp"], 0) if k in AGE_CASES else "—" for k in L]
                     + [f"{obs:,.0f}"]))
lines.append(row(["**All**"]
                 + [f"**{fmt(*report[k]['total']['hosp'], 0)}**" for k in L]
                 + [f"**{report[L[0]]['total']['hosp_obs']:,.0f}**"]))
lines.append(row(["*% difference*"]
                 + [f"*{report[k]['total']['hosp'][0] / report[k]['total']['hosp_obs'] - 1:+.1%}*" for k in L]
                 + [""]))
md.append("\n".join(lines) + "\n")
md.append("\n**Difference from observed, by age group**\n")
lines = [header(["Age group"] + AGE_CASES)]
for a in ages7:
    cells = []
    for k in AGE_CASES:
        d = report[k]["per_age"][a]
        cells.append(f"{d['hosp'][0] / d['hosp_obs'] - 1:+.1%}")
    lines.append(row([a] + cells))
lines.append(row(["**All**"] + [f"**{report[k]['total']['hosp'][0] / report[k]['total']['hosp_obs'] - 1:+.1%}**"
                                for k in AGE_CASES]))
md.append("\n".join(lines) + "\n")

md.append("\nThe single-age fits are calibrated to the population total only "
          "(2 targets), the 7-age fits to the 7 age series plus an "
          "end-of-season by-age scalar (8 targets), so the by-age columns are "
          "a fit check for the former and a fit result for the latter.\n")

md.append("\n## 4. IHR scaling\n")
md.append("`IHR_scale` multiplies both hospitalization-risk parameters "
          "(`I_to_H_prop` and `IV_to_H_prop`) for that age group; the fitted "
          "IHR is that multiplier times the config's baseline value.\n")
md.append("\n**Fitted `IHR_scale`**\n")
lines = [header(["Age group"] + L + ["ratio m(t) / no m(t)"])]
for a in ages7:
    vals = [report[k]["per_age"][a]["ihr_scale"][0] if k in AGE_CASES else np.nan for k in L]
    ratio = vals[0] / vals[1] if AGE_CASES and np.isfinite(vals[1]) else np.nan
    lines.append(row([a] + [fmt(*report[k]["per_age"][a]["ihr_scale"], 2) if k in AGE_CASES else "—"
                            for k in L] + [f"{ratio:.1f}×" if np.isfinite(ratio) else "—"]))
tot = [report[k]["total"]["ihr_scale"][0] for k in L]
lines.append(row(["**All (pop-weighted)**"] + [f"**{fmt(*report[k]['total']['ihr_scale'], 2)}**" for k in L]
                 + [f"**{tot[0]/tot[1]:.1f}× / {tot[2]/tot[3]:.1f}×**"]))
md.append("\n".join(lines) + "\n")

md.append("\n**Fitted IHR (unvaccinated arm, `I_to_H_prop` after scaling)**\n")
lines = [header(["Age group", "Config baseline"] + L)]
base_cfg = report[AGE_CASES[0]]["cfg"] if AGE_CASES else report[L[0]]["cfg"]
base_ihr = np.asarray(base_cfg["params"]["I_to_H_prop"], float).ravel()
for i, a in enumerate(ages7):
    lines.append(row([a, f"{base_ihr[i]:.3%}"]
                     + [pct3(*report[k]["per_age"][a]["ihr"]) if k in AGE_CASES else "—" for k in L]))
lines.append(row(["**All (pop-weighted)**",
                  f"**{base_ihr @ report[AGE_CASES[0]]['pop'] / report[AGE_CASES[0]]['N']:.3%}**"]
                 + [f"**{pct3(*report[k]['total']['ihr'])}**" for k in L]))
md.append("\n".join(lines) + "\n")

md.append("\n**Fitted IHR (vaccinated arm, `IV_to_H_prop` after scaling)**\n")
lines = [header(["Age group"] + L)]
for a in ages7:
    lines.append(row([a] + [pct3(*report[k]["per_age"][a]["ihr_vax"]) if k in AGE_CASES else "—"
                            for k in L]))
md.append("\n".join(lines) + "\n")

md.append("""
## 5. What actually separates the fits

**The m(t) and no-m(t) fits imply the same hospitalizations from ~5.5× different
epidemics.** Hospitalizations are infections × IHR, and hospital-admission data
alone identifies only that product. The two families split it very differently:

- **Without m(t)**, susceptible depletion is the only mechanism in the model
  that can bend the epidemic curve over — R_v (transmission before
  infection-acquired immunity) stays flat near 1.25–1.35 all season. To
  reproduce the observed peak-and-decline the fit must burn through a large
  share of the population, so it lands on a ~37–38% attack rate and a
  correspondingly low IHR.
- **With m(t)**, the multiplier can push transmission down directly (R_v falls
  from ~1.5 in December to ~0.8 in January), so the curve turns over with
  almost no depletion — a ~6.5% attack rate, and an IHR ~5.5× higher to
  compensate.

The IHR_scale ratio between the two families tracks the attack-rate ratio
almost exactly, age group by age group (§4), which is the signature of a
near-degenerate product rather than two genuinely different fits.

**This is a substantive difference, not a technicality.** It changes the
implied infection burden of the season by a factor of ~5.5 and the implied
severity by the same factor in the other direction. A ~6.5% season attack rate
is low for influenza and a ~37% one is on the high side; breaking the
degeneracy needs an infection-side constraint (serology, ILI, wastewater, or an
explicit prior on attack rate or IHR) — the hospitalization series cannot do it.

**The no-m(t) fits fit visibly worse, and the dispersion parameter says so.**
`phi` (NB2 dispersion — smaller means more overdispersion, i.e. more of the
data written off as observation noise) drops from 55 [15 – 306] to 3 [2 – 5] in
the 7-age pair and from 10 to 2 in the single-age pair. Without m(t) the model
cannot track the shape of the curve, so the likelihood buys agreement with
slack instead. The 7-age no-m(t) fit also misses the total by -8.7% and 65+ by
-14.1% (§3), while the m(t) fit is within -1.4% / -2.3%. That is not by itself
grounds to prefer m(t): it adds 18 free parameters under a random-walk prior,
so a fair comparison needs WAIC or a marginal likelihood, not just the
residuals.

**The single-age m(t) fit is the least well identified of the four.** Its
attack rate spans 2.3% – 34.6% across the posterior and its `IHR_scale` spans
0.13 – 1.82 — the whole degeneracy described above, visible *within* one
posterior rather than between fits. The 7-age m(t) fit is much tighter (5.0% –
10.4%), presumably because the by-age targets constrain the infection
distribution. Treat single-age m(t) posterior means with care.

**Age structure matters much less than m(t).** The 7-age and 1-age fits agree
closely on attack rate, hospitalizations and R_eff within each m(t) family; the
population-weighted collapse loses the contact assortativity (which shows up as
a slightly higher R for the same effective beta — see
`effective_beta_MA_vax.png`) but not much else at this level of aggregation.

**Caveats.** Attack rate is `(N − S − SV) / N` at the final day, so it counts
anyone who ever left the susceptible arms by infection. Per-draw trajectories
are paired to their parameter sets by reversing the `default_rng(42)`
permutation the run scripts use; `compute_effective_beta.py` asserts that
pairing on every run. The simulation outputs on disk are rewritten whenever the
corresponding `run_simulations_*` script or counterfactual notebook is re-run,
so regenerate this report after any such run.
""")

OUT_MD.write_text("\n".join(md))
print("wrote", OUT_MD)
