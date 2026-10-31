# MA_vax, Aug-2026 high vaccination rates with 20% initially recovered: vaccination-impact report

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

Files: `model_config_MA_vax_old_vax_initial_R_20pct.json`, `fit_config_MA_vax_old_vax_initial_R_20pct.json`,
`fitted_params_MA_vax_old_vax_initial_R_20pct.json`,
`run_simulations_MA_vax_old_vax_initial_R_20pct_param_set_stochastic.py`, and the outputs listed in §8.

---

## 1. Model structure

### 1.1 Compartments

Nine compartments, each stratified by 7 age groups
(`0`, `1-4`, `5-12`, `13-17`, `18-49`, `50-64`, `65+`):

| Compartment | Meaning |
|---|---|
| `S`  | Susceptible, unvaccinated |
| `E`  | Exposed (latent), unvaccinated |
| `I`  | Infectious, unvaccinated |
| `R`  | Recovered/removed (either arm) |
| `SV` | Susceptible, vaccinated |
| `EV` | Exposed (latent), vaccinated |
| `IV` | Infectious, vaccinated |
| `H`  | Hospitalized (either arm) |
| `D`  | Dead (either arm) |

The vaccinated arm (`SV → EV → IV → H/R`) mirrors the unvaccinated arm
(`S → E → I → H/R`), with its own susceptibility and severity parameters.
`H`, `R`, `D` are shared, pooled compartments — once hospitalized (or
recovered), an individual's vaccination history is no longer tracked.

```mermaid
flowchart LR
    S -- "force of infection" --> E
    S -- "vaccination (S_to_SV)" --> SV
    E -- "E_to_I_rate" --> I
    I -- "I_to_H_prop * I_out_rate" --> H
    I -- "(1-I_to_H_prop) * I_out_rate" --> R
    SV -- "force of infection * vax_susceptibility" --> EV
    EV -- "EV_to_IV_rate" --> IV
    IV -- "IV_to_H_prop * I_out_rate" --> H
    IV -- "(1-IV_to_H_prop) * I_out_rate" --> R
    H -- "H_to_D_prop * H_out_rate" --> D
    H -- "(1-H_to_D_prop) * H_out_rate" --> R
```

### 1.2 Force of infection

Contacts come from fixed age×age contact matrices (Mistry et al. 2021
synthetic contact matrices), with school/work contacts removed on
non-school/work days:

```
C(t) = total_C − (1 − is_school(t))·school_C − (1 − is_work(t))·work_C
beta_adj(t) = beta_baseline · m(t) · (1 + humidity_impact · exp(−180 · humidity(t)))
wtd_inf_prop(t) = (I·I_relative_infectiousness + IV·IV_relative_infectiousness) / population
foi(t) = beta_adj(t) · (C(t) @ wtd_inf_prop(t))
S_to_E   = foi(t) · relative_suscept   · S
SV_to_EV = foi(t) · vax_susceptibility · SV
```

`m(t)` is a smoothly time-varying transmission multiplier (§2.3) that
absorbs behavioral/seasonal variation the mechanistic terms above don't
otherwise capture. `IV_relative_infectiousness = 1.0` — breakthrough
infections (in vaccinated individuals) are assumed exactly as transmissible
as unvaccinated infections. `vax_susceptibility` (age-specific, ≤ 1) is the
residual susceptibility of a vaccinated individual; `1 − vax_susceptibility`
is the model's implied vaccine effectiveness (VE) against infection.

### 1.3 Progression, hospitalization, death

```
E_to_I  = E_to_I_rate · E                 EV_to_IV = EV_to_IV_rate · EV
I_to_H  = I_out_rate · I_to_H_prop · I     IV_to_H  = I_out_rate · IV_to_H_prop · IV
I_to_R  = I_out_rate · (1−I_to_H_prop)·I   IV_to_R  = I_out_rate · (1−IV_to_H_prop)·IV
H_to_D  = H_out_rate · H_to_D_prop · H
H_to_R  = H_out_rate · (1−H_to_D_prop)·H
```

`IV_to_H_prop < I_to_H_prop` for every age group — vaccination reduces
hospitalization risk *given* infection (severity VE), on top of reducing
infection risk.

### 1.4 Vaccination flow

Daily doses (an age-specific proportion of the population, with a 14-day
delay between dose and effective immunity) are applied as an exact `S → SV`
count each day, capped at whatever remains in `S`:

```
base(t) = S + SV
S_to_SV(t) = min(round(vax_prop(t) · base(t)), S)
```

The base pool (`S + SV`) is not eroded by vaccination itself — only
infection depletes it — so a roughly flat input proportion vaccinates a
roughly constant head-count per day, until `S` starts running low late in
the season.

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
population (prior immunity, 1,398,479 people in total, 20% of the state),
`E = E0` (age-specific seed counts, scaled by a single fitted multiplier,
§2.3), `S = population − E0 − R0`, and all other compartments start at zero.

---

## 2. Parameters

### 2.1 Fixed scalar parameters

| Parameter | Value | Meaning |
|---|---|---|
| `num_days` | 250 | Simulation length (2025-09-01 through 2026-05-08) |
| `relative_suscept` | 1.0 | Susceptibility multiplier, unvaccinated arm |
| `I_relative_infectiousness` | 1.0 | Infectiousness weight, unvaccinated `I` |
| `IV_relative_infectiousness` | 1.0 | Infectiousness weight, vaccinated `IV` |
| `E_to_I_rate` | 0.5 /day | ~2-day latent period |
| `EV_to_IV_rate` | 0.5 /day | Same, vaccinated arm |
| `I_out_rate` | 0.333 /day | ~3-day infectious period |
| `H_out_rate` | 0.17 /day | ~6-day hospital stay |
| `vax_transfer_delay_days` | 14 | Days from dose to modeled immunity |

### 2.2 Age-stratified fixed parameters

| Age group | Population | Hospitalization risk, given infection¹ | Hospitalization risk, given breakthrough infection¹ | Death risk, given hospitalization | Residual susceptibility if vaccinated | Initial infections seeded (E)¹ | Initially recovered (R) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 70,067 | 0.697% | 0.636% | 1.74% | 0.57 | 2 | 14,013 (20%) |
| 1-4 | 280,268 | 0.697% | 0.636% | 1.74% | 0.57 | 8 | 56,054 (20%) |
| 5-12 | 606,291 | 0.274% | 0.250% | 1.17% | 0.57 | 17 | 121,258 (20%) |
| 13-17 | 411,782 | 0.274% | 0.250% | 1.17% | 0.57 | 12 | 82,356 (20%) |
| 18-49 | 2,978,204 | 0.561% | 0.511% | 2.63% | 0.79 | 85 | 595,641 (20%) |
| 50-64 | 1,424,434 | 1.060% | 0.966% | 6.30% | 0.79 | 41 | 284,887 (20%) |
| 65+ | 1,221,349 | 9.091% | 6.273% | 7.99% | 1.00 | 35 | 244,270 (20%) |

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
Chain Monte Carlo sampler (emcee; 4000 iterations per walker, 58 walkers — the
fitter raises the configured 40 to twice the 29 sampled dimensions), run in
parallel. The free parameters and their priors:

| Parameter | Prior | Role |
|---|---|---|
| `beta_baseline` | Uniform(0.015, 0.06) | Baseline transmission rate |
| `humidity_impact` | Uniform(1e-08, 1.0) | Strength of humidity forcing |
| Initial-seed multiplier | Log-uniform(0.1, 10.0) | Multiplies the age-specific initial-infection seed counts |
| Hospitalization-risk multiplier (× 7, one per age group) | Uniform(0.1, 2.0) | Multiplies both hospitalization-risk columns (§2.2) for that age group |
| `m(t)` log-increments (× 18, one per 14-day knot) | Normal(0, 0.25) | Random-walk steps in log-transmission (§1.2) |
| `phi` | — | Negative-Binomial dispersion (likelihood nuisance parameter, not a model input) |

**Likelihood**: a Negative-Binomial (NB2) observation model jointly across
8 targets — daily hospital admissions **by age group** (7 time series) plus a
single scalar **end-of-season cumulative hospitalizations by age** target.
The vaccination schedule used during the fit is the archived "OLD high vax"
schedule, the same one the simulations use.

**Posterior sampling**: the first 1700 iterations were discarded as
burn-in, and the remaining chain thinned to every 200th sample, leaving
638 posterior draws (no walkers dropped). Two point estimates are reported:
the **posterior mean** (marginal mean of each parameter) and the **"best"
point** (the single draw with the highest log-posterior, which keeps the
correlation between parameters).

### 2.4 Fitted parameters

Posterior mean and 90% credible interval (5th–95th percentile) across the 638
posterior draws, the "best" point, and for reference the posterior mean of
the earlier high-vax fit (no initial immunity):

| Parameter | Posterior mean | 5% | 95% | "Best" point | Earlier high-vax fit, posterior mean |
|---|---:|---:|---:|---:|---:|
| `beta_baseline` | 0.0391 | 0.0337 | 0.0451 | 0.0383 | 0.0367 |
| `humidity_impact` | 0.621 | 0.225 | 0.945 | 0.419 | 0.595 |
| Initial-seed multiplier | 2.33 | 0.85 | 4.67 | 2.51 | 1.80 |
| `phi` (NB dispersion) | 128 | 28 | 391 | 337 | 116 |
| Hospitalization-risk multiplier — 0 | 1.57 | 1.06 | 1.96 | 1.32 | 1.45 |
| Hospitalization-risk multiplier — 1-4 | 1.65 | 1.18 | 1.98 | 1.54 | 1.63 |
| Hospitalization-risk multiplier — 5-12 | 1.35 | 0.86 | 1.86 | 1.23 | 1.31 |
| Hospitalization-risk multiplier — 13-17 | 0.97 | 0.63 | 1.39 | 0.97 | 0.96 |
| Hospitalization-risk multiplier — 18-49 | 0.54 | 0.37 | 0.73 | 0.48 | 0.52 |
| Hospitalization-risk multiplier — 50-64 | 0.73 | 0.49 | 1.02 | 0.67 | 0.70 |
| Hospitalization-risk multiplier — 65+ | 0.95 | 0.68 | 1.27 | 0.87 | 0.93 |

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

| Age group | Population | Cumulative coverage (scheduled) |
|---|---:|---:|
| 0 | 70,067 | 45.4% |
| 1-4 | 280,268 | 90.7% |
| 5-12 | 606,291 | 71.4% |
| 13-17 | 411,782 | 55.3% |
| 18-49 | 2,978,204 | 40.5% |
| 50-64 | 1,424,434 | 60.4% |
| 65+ | 1,221,349 | 73.2% |
| **All (population-weighted)** | **6,992,395** | **55.9%** |

Four groups sit below a 70% mark (0, 13-17, 18-49, 50-64); 1-4, 5-12, 65+ are above it.
This matters for the "scale to 70% coverage" rows in the appendix (Tables S.A.3/S.A.6):
only the four groups below the mark are scaled up.

Because the 20% initially recovered are outside the vaccination pool (§1),
the doses the model actually delivers are much lower than scheduled:
43.1% delivered coverage overall vs. 55.9% scheduled
(22.9% of scheduled doses not delivered, vs. 3.3% in the earlier fit).
About 20 points of that gap come from prior immunity. The rest, as before, is
the §1.4 cap skipping people infected earlier in the season. Every
"per 100,000 doses" panel divides by the **scheduled** count (see the appendix).

---

## 3. Vaccination-impact results

Every table in this section reports a **median and 95% interval across 638
simulations**: one deterministic simulation per posterior parameter draw
(§2.3), re-using the same draw for every scenario compared within a table so
that the comparison is paired. The intervals reflect **calibration
uncertainty**, not epidemic-process noise.

### New daily hospitalizations: baseline vs. no vaccination

Total-population new hospitalizations per day, posterior median and 95%
interval across the 638 parameter draws:

![Baseline vs. no vaccination](report_assets_MA_vax_old_vax_initial_R_20pct/baseline_vs_no_vaccination_daily_H.png)

The fitted vaccination program cuts the peak in daily new hospitalizations
to under a third of what it would otherwise be (median peak ≈ 167/day
baseline vs. ≈ 569/day with no vaccination, both on 2025-12-28), and
the season total from a median of 18,348 to 6,071 admissions.
It reduces the epidemic's height without changing its timing.

### Table S.A.1 — Hospitalizations averted, infection vs. severity protection

Decomposes the total hospitalizations averted by vaccination into two
channels: protection against getting infected at all, and — *given* a
breakthrough infection still happens — protection against that infection
becoming severe enough to need hospitalization.

**Infection protection** (no vaccination → infection-protection-only, i.e. VE against infection retained but VE against severity zeroed out)

| Age group | Hospitalizations Averted | % Hospitalizations Averted | Averted per 100,000 Population | Averted per 100,000 Doses |
|---|---|---|---|---|
| 0 | 79 [53 – 99] | 66.8% [61.2% – 70.3%] | 112.2 [75.3 – 141.8] | 247.4 [166.0 – 312.6] |
| 1-4 | 421 [306 – 502] | 72.7% [67.6% – 75.7%] | 150.1 [109.3 – 179.0] | 165.4 [120.6 – 197.3] |
| 5-12 | 380 [250 – 509] | 69.5% [63.4% – 73.2%] | 62.6 [41.3 – 83.9] | 87.6 [57.8 – 117.4] |
| 13-17 | 183 [118 – 262] | 66.6% [60.0% – 70.6%] | 44.5 [28.8 – 63.7] | 80.4 [52.0 – 115.1] |
| 18-49 | 1,197 [851 – 1,564] | 61.3% [54.4% – 65.6%] | 40.2 [28.6 – 52.5] | 99.2 [70.5 – 129.6] |
| 50-64 | 1,320 [931 – 1,760] | 62.9% [56.4% – 66.9%] | 92.7 [65.4 – 123.6] | 153.6 [108.3 – 204.7] |
| 65+ | 7,683 [5,823 – 9,258] | 60.5% [54.1% – 64.5%] | 629.0 [476.7 – 758.0] | 859.3 [651.2 – 1035.5] |
| **All** | **11,299 [8,561 – 13,403]** | **61.7% [55.3% – 65.5%]** | **161.6 [122.4 – 191.7]** | **289.2 [219.1 – 343.0]** |

**Severity protection** (infection-protection-only → full baseline, i.e. adding back VE against severity)

| Age group | Hospitalizations Averted | % Hospitalizations Averted | Averted per 100,000 Population | Averted per 100,000 Doses |
|---|---|---|---|---|
| 0 | 1 [1 – 1] | 0.7% [0.6% – 0.8%] | 1.1 [0.8 – 1.6] | 2.5 [1.8 – 3.5] |
| 1-4 | 8 [6 – 10] | 1.4% [1.2% – 1.6%] | 2.9 [2.1 – 3.7] | 3.2 [2.3 – 4.1] |
| 5-12 | 6 [4 – 8] | 1.1% [1.0% – 1.3%] | 1.0 [0.7 – 1.3] | 1.4 [1.0 – 1.8] |
| 13-17 | 3 [2 – 3] | 0.9% [0.8% – 1.1%] | 0.6 [0.4 – 0.8] | 1.1 [0.8 – 1.5] |
| 18-49 | 17 [14 – 21] | 0.9% [0.8% – 1.0%] | 0.6 [0.5 – 0.7] | 1.4 [1.1 – 1.7] |
| 50-64 | 28 [22 – 33] | 1.3% [1.2% – 1.5%] | 1.9 [1.6 – 2.3] | 3.2 [2.6 – 3.9] |
| 65+ | 905 [824 – 984] | 7.1% [6.4% – 8.3%] | 74.1 [67.5 – 80.6] | 101.3 [92.2 – 110.1] |
| **All** | **968 [884 – 1,046]** | **5.3% [4.7% – 6.2%]** | **13.8 [12.6 – 15.0]** | **24.8 [22.6 – 26.8]** |

**Total** (no vaccination → full baseline)

| Age group | Hospitalizations Averted | % Hospitalizations Averted | Averted per 100,000 Population | Averted per 100,000 Doses |
|---|---|---|---|---|
| 0 | 79 [53 – 100] | 67.5% [62.0% – 70.9%] | 113.3 [76.1 – 143.2] | 249.8 [167.8 – 315.7] |
| 1-4 | 429 [313 – 511] | 74.1% [69.3% – 77.0%] | 152.9 [111.7 – 182.5] | 168.6 [123.1 – 201.2] |
| 5-12 | 385 [256 – 516] | 70.6% [64.7% – 74.2%] | 63.5 [42.3 – 85.1] | 88.9 [59.2 – 119.1] |
| 13-17 | 186 [120 – 265] | 67.5% [61.1% – 71.5%] | 45.1 [29.2 – 64.4] | 81.5 [52.7 – 116.5] |
| 18-49 | 1,215 [864 – 1,582] | 62.2% [55.4% – 66.3%] | 40.8 [29.0 – 53.1] | 100.6 [71.6 – 131.1] |
| 50-64 | 1,347 [952 – 1,791] | 64.2% [58.0% – 68.0%] | 94.6 [66.8 – 125.8] | 156.7 [110.8 – 208.4] |
| 65+ | 8,581 [6,683 – 10,218] | 67.6% [62.4% – 70.9%] | 702.6 [547.2 – 836.6] | 959.7 [747.5 – 1142.8] |
| **All** | **12,284 [9,539 – 14,405]** | **66.9% [61.4% – 70.3%]** | **175.7 [136.4 – 206.0]** | **314.4 [244.1 – 368.6]** |

As in the earlier fit, almost all of the averted burden comes from
**blocking infection** (11,299 [8,561 – 13,403] of 12,284 [9,539 – 14,405]), not from
reducing severity given a breakthrough (968 [884 – 1,046], nearly all in 65+).
The total is somewhat smaller than in the earlier fit (13,584 [10,817 – 15,997]):
with a fifth of the population already immune, the no-vaccination epidemic is
smaller, so there is less for vaccination to avert. Per scheduled dose the
drop is larger (314.4 [244.1 – 368.6] vs. 347.6 [276.8 – 409.4] per 100,000),
because the same doses are spread over a population in which ~20% gain nothing from them.

### Table S.A.2 — Hospitalizations averted by age group vaccinated

Each column vaccinates a single age group only (all others left
unvaccinated) and compares to no vaccination at all; "All" is the full
baseline schedule. Rows are the age group in which hospitalizations are
counted, so off-diagonal cells show the indirect (transmission-blocking)
benefit to *other* age groups from vaccinating this one.

**Hospitalizations averted (count)**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 [13 – 25] | 9 [6 – 12] | 27 [17 – 34] | 15 [10 – 19] | 20 [12 – 25] | 10 [6 – 12] | 0 [0 – 0] | 79 [53 – 100] |
| 1-4 | 3 [2 – 3] | 221 [161 – 263] | 141 [95 – 166] | 72 [48 – 86] | 91 [61 – 107] | 44 [30 – 53] | 0 [0 – 0] | 429 [313 – 511] |
| 5-12 | 2 [1 – 3] | 36 [22 – 49] | 244 [162 – 329] | 65 [40 – 91] | 71 [43 – 100] | 36 [22 – 51] | 0 [0 – 0] | 385 [256 – 516] |
| 13-17 | 1 [0 – 1] | 14 [9 – 22] | 53 [32 – 81] | 93 [61 – 133] | 35 [21 – 55] | 19 [11 – 29] | 0 [0 – 0] | 186 [120 – 265] |
| 18-49 | 8 [5 – 10] | 126 [86 – 173] | 399 [268 – 552] | 244 [163 – 339] | 437 [307 – 579] | 163 [110 – 224] | 0 [0 – 0] | 1,215 [864 – 1,582] |
| 50-64 | 7 [5 – 11] | 128 [84 – 181] | 425 [280 – 600] | 270 [177 – 381] | 332 [218 – 465] | 431 [305 – 574] | 0 [0 – 0] | 1,347 [952 – 1,791] |
| 65+ | 51 [37 – 64] | 862 [620 – 1,085] | 2,780 [1,981 – 3,504] | 1,727 [1,226 – 2,190] | 2,082 [1,495 – 2,608] | 1,238 [901 – 1,543] | 2,313 [1,905 – 2,664] | 8,581 [6,683 – 10,218] |
| **All** | **91 [68 – 108]** | **1,412 [1,026 – 1,689]** | **4,102 [2,888 – 5,066]** | **2,502 [1,761 – 3,132]** | **3,073 [2,216 – 3,809]** | **1,949 [1,436 – 2,405]** | **2,313 [1,905 – 2,664]** | **12,284 [9,539 – 14,405]** |

**% reduction in hospitalizations**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.7% [16.1% – 17.0%] | 8.0% [6.9% – 8.8%] | 22.7% [19.0% – 25.3%] | 13.0% [10.8% – 14.7%] | 16.8% [14.2% – 18.7%] | 8.1% [6.8% – 9.2%] | 0.0% [0.0% – 0.0%] | 67.5% [62.0% – 70.9%] |
| 1-4 | 0.5% [0.4% – 0.5%] | 38.0% [36.2% – 39.3%] | 24.1% [20.2% – 26.8%] | 12.3% [10.1% – 14.1%] | 15.5% [12.9% – 17.5%] | 7.6% [6.3% – 8.7%] | 0.0% [0.0% – 0.0%] | 74.1% [69.3% – 77.0%] |
| 5-12 | 0.3% [0.3% – 0.4%] | 6.5% [5.3% – 7.4%] | 44.8% [40.7% – 47.5%] | 11.9% [9.5% – 13.7%] | 13.0% [10.4% – 15.1%] | 6.6% [5.3% – 7.7%] | 0.0% [0.0% – 0.0%] | 70.6% [64.7% – 74.2%] |
| 13-17 | 0.3% [0.2% – 0.3%] | 5.3% [4.2% – 6.1%] | 19.6% [15.8% – 22.5%] | 33.8% [30.7% – 35.9%] | 13.0% [10.4% – 15.1%] | 7.0% [5.6% – 8.1%] | 0.0% [0.0% – 0.0%] | 67.5% [61.1% – 71.5%] |
| 18-49 | 0.4% [0.3% – 0.4%] | 6.5% [5.3% – 7.3%] | 20.5% [16.7% – 23.2%] | 12.5% [10.2% – 14.3%] | 22.3% [19.5% – 24.3%] | 8.4% [6.9% – 9.5%] | 0.0% [0.0% – 0.0%] | 62.2% [55.4% – 66.3%] |
| 50-64 | 0.4% [0.3% – 0.4%] | 6.1% [5.1% – 6.9%] | 20.3% [16.6% – 23.0%] | 12.9% [10.5% – 14.6%] | 15.8% [13.1% – 17.8%] | 20.5% [18.8% – 21.8%] | 0.0% [0.0% – 0.0%] | 64.2% [58.0% – 68.0%] |
| 65+ | 0.4% [0.3% – 0.4%] | 6.8% [5.8% – 7.6%] | 21.9% [18.4% – 24.5%] | 13.6% [11.4% – 15.3%] | 16.3% [13.8% – 18.2%] | 9.7% [8.3% – 10.8%] | 18.2% [18.0% – 18.4%] | 67.6% [62.4% – 70.9%] |
| **All** | **0.5% [0.4% – 0.5%]** | **7.7% [6.6% – 8.3%]** | **22.3% [18.8% – 24.9%]** | **13.6% [11.4% – 15.4%]** | **16.7% [14.2% – 18.7%]** | **10.6% [9.1% – 11.8%]** | **12.6% [12.1% – 13.3%]** | **66.9% [61.4% – 70.3%]** |

**Hospitalizations averted per 100,000 population**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 27.8 [18.7 – 35.5] | 13.5 [8.6 – 17.0] | 38.2 [23.8 – 48.0] | 21.9 [13.8 – 27.6] | 28.4 [17.7 – 35.5] | 13.8 [8.6 – 17.2] | 0.0 [0.0 – 0.0] | 113.3 [76.1 – 143.2] |
| 1-4 | 0.9 [0.6 – 1.1] | 78.7 [57.5 – 93.7] | 50.1 [33.9 – 59.4] | 25.8 [17.1 – 30.6] | 32.3 [21.8 – 38.3] | 15.9 [10.6 – 18.9] | 0.0 [0.0 – 0.0] | 152.9 [111.7 – 182.5] |
| 5-12 | 0.3 [0.2 – 0.4] | 5.9 [3.6 – 8.1] | 40.3 [26.7 – 54.2] | 10.7 [6.6 – 15.0] | 11.7 [7.1 – 16.5] | 6.0 [3.6 – 8.4] | 0.0 [0.0 – 0.0] | 63.5 [42.3 – 85.1] |
| 13-17 | 0.2 [0.1 – 0.3] | 3.5 [2.1 – 5.3] | 12.9 [7.8 – 19.8] | 22.6 [14.7 – 32.3] | 8.6 [5.1 – 13.3] | 4.6 [2.7 – 7.1] | 0.0 [0.0 – 0.0] | 45.1 [29.2 – 64.4] |
| 18-49 | 0.3 [0.2 – 0.3] | 4.2 [2.9 – 5.8] | 13.4 [9.0 – 18.5] | 8.2 [5.5 – 11.4] | 14.7 [10.3 – 19.4] | 5.5 [3.7 – 7.5] | 0.0 [0.0 – 0.0] | 40.8 [29.0 – 53.1] |
| 50-64 | 0.5 [0.3 – 0.7] | 9.0 [5.9 – 12.7] | 29.9 [19.7 – 42.2] | 19.0 [12.4 – 26.7] | 23.3 [15.3 – 32.7] | 30.3 [21.4 – 40.3] | 0.0 [0.0 – 0.0] | 94.6 [66.8 – 125.8] |
| 65+ | 4.2 [3.0 – 5.2] | 70.6 [50.7 – 88.8] | 227.6 [162.2 – 286.9] | 141.4 [100.4 – 179.3] | 170.4 [122.4 – 213.5] | 101.4 [73.8 – 126.4] | 189.4 [155.9 – 218.1] | 702.6 [547.2 – 836.6] |
| **All** | **1.3 [1.0 – 1.5]** | **20.2 [14.7 – 24.2]** | **58.7 [41.3 – 72.4]** | **35.8 [25.2 – 44.8]** | **43.9 [31.7 – 54.5]** | **27.9 [20.5 – 34.4]** | **33.1 [27.2 – 38.1]** | **175.7 [136.4 – 206.0]** |

**Hospitalizations averted per 100,000 doses**

Off-diagonal cells divide by the doses scheduled for the **column's** age group
(the only group vaccinated in that scenario). The `All` column keeps each
row's own dose count, matching Table S.A.1.

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 61.4 [41.3 – 78.3] | 3.7 [2.4 – 4.7] | 6.2 [3.9 – 7.8] | 6.7 [4.2 – 8.5] | 1.6 [1.0 – 2.1] | 1.1 [0.7 – 1.4] | 0.0 [0.0 – 0.0] | 249.8 [167.8 – 315.7] |
| 1-4 | 8.3 [5.7 – 9.9] | 86.7 [63.4 – 103.3] | 32.4 [21.9 – 38.4] | 31.7 [21.0 – 37.6] | 7.5 [5.1 – 8.9] | 5.2 [3.5 – 6.2] | 0.0 [0.0 – 0.0] | 168.6 [123.1 – 201.2] |
| 5-12 | 5.7 [3.5 – 7.9] | 14.0 [8.7 – 19.4] | 56.4 [37.4 – 75.9] | 28.5 [17.5 – 40.0] | 5.9 [3.6 – 8.3] | 4.2 [2.5 – 5.9] | 0.0 [0.0 – 0.0] | 88.9 [59.2 – 119.1] |
| 13-17 | 2.6 [1.5 – 3.9] | 5.6 [3.4 – 8.6] | 12.3 [7.4 – 18.8] | 40.8 [26.6 – 58.5] | 2.9 [1.7 – 4.5] | 2.2 [1.3 – 3.4] | 0.0 [0.0 – 0.0] | 81.5 [52.7 – 116.5] |
| 18-49 | 24.0 [16.3 – 32.6] | 49.7 [33.7 – 68.0] | 92.1 [61.9 – 127.4] | 107.3 [71.4 – 148.6] | 36.2 [25.5 – 48.0] | 19.0 [12.8 – 26.0] | 0.0 [0.0 – 0.0] | 100.6 [71.6 – 131.1] |
| 50-64 | 23.5 [15.4 – 33.0] | 50.5 [33.2 – 71.2] | 98.2 [64.6 – 138.6] | 118.5 [77.8 – 167.2] | 27.5 [18.1 – 38.6] | 50.2 [35.4 – 66.8] | 0.0 [0.0 – 0.0] | 156.7 [110.8 – 208.4] |
| 65+ | 160.0 [115.4 – 200.3] | 339.1 [243.7 – 426.8] | 641.8 [457.4 – 809.0] | 758.2 [538.2 – 961.5] | 172.5 [123.9 – 216.1] | 144.0 [104.8 – 179.5] | 258.7 [213.0 – 297.9] | 959.7 [747.5 – 1142.8] |
| **All** | **287.7 [214.2 – 340.7]** | **555.4 [403.4 – 664.6]** | **947.0 [666.7 – 1169.6]** | **1098.3 [773.1 – 1375.0]** | **254.6 [183.6 – 315.6]** | **226.7 [167.0 – 279.8]** | **258.7 [213.0 – 297.9]** | **314.4 [244.1 – 368.6]** |

Per dose, the indirect benefit is dominated by what it does for **65+**:
vaccinating 13-17 averts 758.2 [538.2 – 961.5] hospitalizations per 100,000 doses
in the 65+ group alone, against 40.8 [26.6 – 58.5] in 13-17 itself. This is
why the `All` row ranks 13-17 (1098.3) and 5-12
(947.0) far above the groups that carry the burden directly.
65+ is the only group with zero indirect effect on every other group, since
its vaccine has no infection-blocking effect (§2.2).

### Table S.A.4 — Vaccine-effectiveness sensitivity scenarios

Implied vaccine effectiveness under three VE presets — a parameter table, not
a simulation result, and identical to the earlier fit (VE parameters are
not fitted). `Baseline VE (fitted)` is the reference point; `Low VE`/`High VE`
bracket it.

| Scenario | Age group | VE against infection | VE against hospitalization (overall) | VE against hospitalization, given infection |
|---|---|---|---|---|
| Low VE | 0 | 33% | 33% | 0% |
| Low VE | 1-4 | 33% | 33% | 0% |
| Low VE | 5-12 | 33% | 33% | 0% |
| Low VE | 13-17 | 33% | 33% | 0% |
| Low VE | 18-49 | 6% | 10% | 4% |
| Low VE | 50-64 | 6% | 10% | 4% |
| Low VE | 65+ | 0% | 21% | 21% |
| Baseline VE (fitted) | 0 | 43% | 48% | 9% |
| Baseline VE (fitted) | 1-4 | 43% | 48% | 9% |
| Baseline VE (fitted) | 5-12 | 43% | 48% | 9% |
| Baseline VE (fitted) | 13-17 | 43% | 48% | 9% |
| Baseline VE (fitted) | 18-49 | 21% | 28% | 9% |
| Baseline VE (fitted) | 50-64 | 21% | 28% | 9% |
| Baseline VE (fitted) | 65+ | 0% | 31% | 31% |
| High VE | 0 | 51% | 78% | 55% |
| High VE | 1-4 | 51% | 78% | 55% |
| High VE | 5-12 | 51% | 78% | 55% |
| High VE | 13-17 | 51% | 78% | 55% |
| High VE | 18-49 | 34% | 44% | 16% |
| High VE | 50-64 | 34% | 44% | 16% |
| High VE | 65+ | 14% | 39% | 29% |

65+ has 0% VE against infection in the baseline and in `Low VE` (residual
susceptibility 1.00), so only the severity channel operates there; `High VE`
scales it to 0.86 (14% VE against infection).

### Table S.A.5 — Hospitalizations averted across VE scenarios

Compares each VE sensitivity scenario against no vaccination at all.

**Hospitalizations averted (count)**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 55 [36 – 69] | 79 [53 – 100] | 96 [64 – 122] |
| 1-4 | 310 [222 – 368] | 429 [313 – 511] | 511 [376 – 613] |
| 5-12 | 274 [175 – 370] | 385 [256 – 516] | 462 [321 – 610] |
| 13-17 | 128 [80 – 186] | 186 [120 – 265] | 225 [150 – 317] |
| 18-49 | 752 [522 – 1,017] | 1,215 [864 – 1,582] | 1,497 [1,104 – 1,912] |
| 50-64 | 827 [561 – 1,141] | 1,347 [952 – 1,791] | 1,655 [1,195 – 2,149] |
| 65+ | 5,883 [4,444 – 7,133] | 8,581 [6,683 – 10,218] | 10,220 [8,134 – 11,957] |
| **All** | **8,276 [6,186 – 9,957]** | **12,284 [9,539 – 14,405]** | **14,734 [11,685 – 16,949]** |

**% reduction in hospitalizations**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 46.3% [41.1% – 49.9%] | 67.5% [62.0% – 70.9%] | 81.6% [77.7% – 83.8%] |
| 1-4 | 53.5% [48.4% – 56.8%] | 74.1% [69.3% – 77.0%] | 88.3% [85.6% – 89.7%] |
| 5-12 | 50.1% [44.2% – 54.0%] | 70.6% [64.7% – 74.2%] | 84.7% [80.8% – 86.8%] |
| 13-17 | 46.6% [40.6% – 50.7%] | 67.5% [61.1% – 71.5%] | 82.1% [77.5% – 84.6%] |
| 18-49 | 38.7% [32.7% – 42.8%] | 62.2% [55.4% – 66.3%] | 76.7% [71.3% – 79.7%] |
| 50-64 | 39.7% [33.9% – 43.6%] | 64.2% [58.0% – 68.0%] | 78.8% [74.0% – 81.4%] |
| 65+ | 46.3% [41.4% – 49.7%] | 67.6% [62.4% – 70.9%] | 80.6% [76.5% – 82.8%] |
| **All** | **45.0% [40.0% – 48.5%]** | **66.9% [61.4% – 70.3%]** | **80.3% [76.2% – 82.6%]** |

**Hospitalizations averted per 100,000 population**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 78.0 [51.6 – 98.7] | 113.3 [76.1 – 143.2] | 136.3 [92.0 – 174.3] |
| 1-4 | 110.6 [79.2 – 131.2] | 152.9 [111.7 – 182.5] | 182.2 [134.2 – 218.7] |
| 5-12 | 45.2 [28.9 – 61.1] | 63.5 [42.3 – 85.1] | 76.3 [52.9 – 100.7] |
| 13-17 | 31.0 [19.4 – 45.1] | 45.1 [29.2 – 64.4] | 54.7 [36.4 – 77.1] |
| 18-49 | 25.3 [17.5 – 34.1] | 40.8 [29.0 – 53.1] | 50.3 [37.1 – 64.2] |
| 50-64 | 58.1 [39.4 – 80.1] | 94.6 [66.8 – 125.8] | 116.2 [83.9 – 150.9] |
| 65+ | 481.6 [363.9 – 584.0] | 702.6 [547.2 – 836.6] | 836.8 [665.9 – 979.0] |
| **All** | **118.4 [88.5 – 142.4]** | **175.7 [136.4 – 206.0]** | **210.7 [167.1 – 242.4]** |

Even under the pessimistic `Low VE` assumption, the vaccination schedule
still averts 45.0% of hospitalizations overall (80.3% under `High VE`).

---

## 4. Baseline fit check — posterior-uncertainty simulation vs. raw data

The baseline scenario, one deterministic simulation per posterior draw (the
same 638 runs used in §3), against the raw data it was fit to.

### Cumulative hospitalizations, by age group

Simulated (per-draw season total, median and 95% interval) vs. raw daily
hospital admissions, summed over the simulation window
(2025-09-01 – 2026-05-08):

| Age group | Simulated (median) | Simulated 95% interval | Raw data | % difference (median) |
|---|---:|---:|---:|---:|
| 0 | 37.3 | 26.5 – 52.5 | 41.1 | -9.3% |
| 1-4 | 150.8 | 108.2 – 192.7 | 164.6 | -8.4% |
| 5-12 | 162.6 | 118.9 – 208.3 | 160.5 | 1.3% |
| 13-17 | 90.3 | 60.3 – 121.6 | 90.3 | -0.0% |
| 18-49 | 737.8 | 602.6 – 916.5 | 726.0 | 1.6% |
| 50-64 | 756.1 | 609.7 – 915.3 | 749.6 | 0.9% |
| 65+ | 4119.1 | 3766.0 – 4476.2 | 4222.3 | -2.4% |
| **All** | **6070.7** | **5648.2 – 6480.9** | **6154.4** | **-1.4%** |

The fit tracks the data closely overall (-1.4% on the total),
with the raw value inside the 95% interval for every age group.
The largest relative miss is in age 0 (-9.3%, only
4 admissions off in absolute terms).

### Daily new hospitalizations by age group

![Daily fit check by age](report_assets_MA_vax_old_vax_initial_R_20pct/fit_check_daily_by_age.png)

### Cumulative hospitalizations by age group

![Cumulative fit check by age](report_assets_MA_vax_old_vax_initial_R_20pct/fit_check_cumulative_by_age.png)

The same comparison in the layout of the other fits' `fit_comparison_output_*`
folders is in `../../fit_comparison_output_MA_vax_old_vax_initial_R_20pct/`. Its intervals are
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

| Age group | Attack rate, baseline | Attack rate, no vaccination | Effective IHR, unvaccinated | Effective IHR, vaccinated |
|---|---|---|---|---|
| 0 | 5.0% [3.7-7.6] | 15.2% [12.4-19.5] | 1.127% [0.676-1.378] | 1.028% [0.616-1.257] |
| 1-4 | 4.9% [3.6-7.4] | 17.8% [14.8-22.7] | 1.177% [0.744-1.385] | 1.073% [0.678-1.264] |
| 5-12 | 7.5% [5.6-11.3] | 24.7% [20.8-30.6] | 0.366% [0.221-0.525] | 0.334% [0.201-0.479] |
| 13-17 | 8.5% [6.3-12.7] | 25.3% [21.4-31.4] | 0.261% [0.153-0.413] | 0.238% [0.140-0.377] |
| 18-49 | 8.5% [6.3-12.7] | 22.0% [18.3-27.8] | 0.298% [0.192-0.425] | 0.272% [0.175-0.388] |
| 50-64 | 7.2% [5.4-10.8] | 19.5% [16.1-24.8] | 0.756% [0.472-1.134] | 0.689% [0.430-1.033] |
| 65+ | 4.8% [3.5-7.2] | 12.0% [9.8-15.7] | 8.658% [5.648-11.802] | 5.974% [3.897-8.143] |
| **All** | **7.3% [5.4-11.0]** | **19.9% [16.6-25.3]** | **n/a** | **n/a** |

---

## 6. Transmission components

![Transmission components](../../transmission_components/transmission_components_MA_vax_old_vax_initial_R_20pct.png)

Posterior median and 95% band of `m(t)`, the humidity term, `beta_adjusted`,
`beta_eff` (with population susceptibility from vaccination), the calendar
effect and the calendar-adjusted `beta_eff`. As noted on the figure, `beta_eff`
accounts for vaccination only: it includes neither depletion from infection
nor the 20% initially recovered. The time series are in
`../../transmission_components/transmission_components_MA_vax_old_vax_initial_R_20pct.csv`.

---

## 7. Comparison with the earlier high-vax fit (no initial immunity)

Same vaccination schedule, VE assumptions and fit targets; the only
differences are the 20% initial `R` and the refit it required.

| Quantity | Earlier high-vax fit (no initial immunity) | This fit (20% initially recovered) |
|---|---|---|
| Initially recovered (R) | 0% | 20% of every age group |
| `beta_baseline`, posterior mean | 0.0367 | 0.0391 |
| Initial-seed multiplier, posterior mean | 1.80 | 2.33 |
| Attack rate, baseline (all ages, share of total population) | 7.5% [5.4-11.6] | 7.3% [5.4-11.0] |
| Attack rate, no vaccination | 22.0% [18.0-29.0] | 19.9% [16.6-25.3] |
| Hospitalizations averted by vaccination (S.A.1/S.A.2 total) | 13,584 [10,817 – 15,997] | 12,284 [9,539 – 14,405] |
| % of hospitalizations averted (S.A.1 total) | 69.1% [64.1% – 72.5%] | 66.9% [61.4% – 70.3%] |
| … via infection protection (% of no-vax burden) | 64.1% [58.4% – 68.1%] | 61.7% [55.3% – 65.5%] |
| … via severity protection (% of no-vax burden) | 4.9% [4.4% – 5.8%] | 5.3% [4.7% – 6.2%] |
| Averted per 100,000 scheduled doses | 347.6 [276.8 – 409.4] | 314.4 [244.1 – 368.6] |
| % averted, Low VE (S.A.5) | 47.3% [42.3% – 51.1%] | 45.0% [40.0% – 48.5%] |
| % averted, High VE (S.A.5) | 81.8% [78.2% – 84.1%] | 80.3% [76.2% – 82.6%] |
| Additional averted at 70% coverage floor (S.A.3 All) | 1,613 [1,472 – 1,753] | 1,575 [1,401 – 1,707] |
| Scheduled doses not delivered | 3.3% | 22.9% |
| Delivered coverage (scheduled 55.9%) | 54.0% | 43.1% |

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
| `fitted_params_MA_vax_old_vax_initial_R_20pct.json` | `run_fitting_MA_vax_old_vax_initial_R_20pct.py` |
| `simulation_output_MA_vax_old_vax_initial_R_20pct_param_set_stochastic/` | `run_simulations_MA_vax_old_vax_initial_R_20pct_param_set_stochastic.py` |
| `counterfactual_tables_from_db_MA_vax_old_vax_initial_R_20pct_param_set_stochastic/` | `../../build_counterfactual_tables_from_db.py` |
| `MA_vax_old_vax_initial_R_20pct_param_set_stochastic__metric_timeseries.csv`, `../../fit_comparison_output_MA_vax_old_vax_initial_R_20pct/` | `../../export_metric_timeseries_MA_vax.py`, `../../plot_fit_vs_actual.py` |
| `../../transmission_components/*_MA_vax_old_vax_initial_R_20pct.*` | `../../plot_transmission_components_MA_vax.py --only MA_vax_old_vax_initial_R_20pct` |
| `report_assets_MA_vax_old_vax_initial_R_20pct/`, this report | `build_report_assets_MA_vax_old_vax_initial_R_20pct.py`, `build_report_MA_vax_old_vax_initial_R_20pct.py` |

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
  quota**: 1-4, 5-12, 65+ are already above it and keep their baseline schedule,
  contributing all-zero columns. Coverage here is scheduled coverage of the
  whole age group, including the initially recovered.
- **"Per 100,000 doses" counts scheduled doses, not delivered doses**:
  22.9% of scheduled doses are not delivered in the model, mostly
  because they would go to people who are already immune (see the appendix).

---

## Appendix: Tables S.A.3 and S.A.6

### Table S.A.3 — Additional hospitalizations averted at 70% coverage

Each column scales a single age group's vaccination schedule up to 70%
cumulative scheduled coverage; "All" scales every eligible age group. Compared
against the baseline vaccination scenario. 70% is a **floor**: the four groups
below it (0, 13-17, 18-49, 50-64) are scaled up, and the three already above it
(1-4, 5-12, 65+) keep their baseline schedule and give exact-zero columns.

**Hospitalizations averted (count)**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 [3 – 6] | 0 [0 – 0] | 0 [0 – 0] | 2 [1 – 3] | 7 [5 – 9] | 1 [1 – 1] | 0 [0 – 0] | 12 [9 – 17] |
| 1-4 | 1 [0 – 1] | 0 [0 – 0] | 0 [0 – 0] | 8 [6 – 10] | 27 [20 – 33] | 3 [2 – 4] | 0 [0 – 0] | 36 [27 – 46] |
| 5-12 | 1 [0 – 1] | 0 [0 – 0] | 0 [0 – 0] | 9 [6 – 12] | 27 [19 – 34] | 3 [2 – 4] | 0 [0 – 0] | 38 [27 – 48] |
| 13-17 | 0 [0 – 0] | 0 [0 – 0] | 0 [0 – 0] | 12 [8 – 17] | 15 [10 – 20] | 2 [1 – 3] | 0 [0 – 0] | 27 [18 – 37] |
| 18-49 | 3 [2 – 3] | 0 [0 – 0] | 0 [0 – 0] | 41 [32 – 51] | 173 [139 – 217] | 17 [14 – 21] | 0 [0 – 0] | 220 [177 – 276] |
| 50-64 | 2 [2 – 3] | 0 [0 – 0] | 0 [0 – 0] | 42 [32 – 52] | 136 [107 – 169] | 36 [29 – 44] | 0 [0 – 0] | 202 [161 – 251] |
| 65+ | 14 [12 – 15] | 0 [0 – 0] | 0 [0 – 0] | 230 [196 – 257] | 742 [646 – 819] | 102 [89 – 113] | 0 [0 – 0] | 1,032 [901 – 1,136] |
| **All** | **24 [21 – 26]** | **0 [0 – 0]** | **0 [0 – 0]** | **346 [296 – 380]** | **1,131 [997 – 1,229]** | **165 [145 – 181]** | **0 [0 – 0]** | **1,575 [1,401 – 1,707]** |

**% reduction in hospitalizations**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 11.2% [11.0% – 11.3%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.5% [4.8% – 5.9%] | 18.8% [17.0% – 19.9%] | 2.2% [2.0% – 2.4%] | 0.0% [0.0% – 0.0%] | 33.5% [31.4% – 34.8%] |
| 1-4 | 0.4% [0.3% – 0.4%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.4% [4.7% – 5.8%] | 17.9% [16.1% – 19.0%] | 2.1% [1.9% – 2.3%] | 0.0% [0.0% – 0.0%] | 24.5% [22.2% – 25.9%] |
| 5-12 | 0.3% [0.3% – 0.3%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.6% [4.8% – 6.1%] | 16.5% [14.6% – 17.7%] | 2.0% [1.7% – 2.2%] | 0.0% [0.0% – 0.0%] | 23.3% [20.8% – 24.8%] |
| 13-17 | 0.3% [0.3% – 0.3%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 13.9% [12.9% – 14.5%] | 16.6% [14.6% – 17.9%] | 2.1% [1.8% – 2.3%] | 0.0% [0.0% – 0.0%] | 30.2% [27.6% – 31.7%] |
| 18-49 | 0.3% [0.3% – 0.4%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.5% [4.8% – 6.0%] | 23.5% [21.6% – 24.7%] | 2.3% [2.0% – 2.5%] | 0.0% [0.0% – 0.0%] | 30.0% [27.5% – 31.4%] |
| 50-64 | 0.3% [0.3% – 0.3%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.6% [4.8% – 6.0%] | 18.2% [16.3% – 19.3%] | 4.8% [4.4% – 5.0%] | 0.0% [0.0% – 0.0%] | 26.9% [24.5% – 28.3%] |
| 65+ | 0.3% [0.3% – 0.4%] | 0.0% [0.0% – 0.0%] | 0.0% [0.0% – 0.0%] | 5.6% [4.9% – 6.0%] | 18.0% [16.3% – 19.1%] | 2.5% [2.2% – 2.7%] | 0.0% [0.0% – 0.0%] | 25.1% [22.9% – 26.4%] |
| **All** | **0.4% [0.4% – 0.4%]** | **0.0% [0.0% – 0.0%]** | **0.0% [0.0% – 0.0%]** | **5.7% [5.0% – 6.1%]** | **18.7% [16.8% – 19.7%]** | **2.7% [2.4% – 2.9%]** | **0.0% [0.0% – 0.0%]** | **26.0% [23.6% – 27.3%]** |

**Hospitalizations averted per 100,000 population**

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.9 [4.2 – 8.3] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 2.9 [2.1 – 3.9] | 10.0 [7.1 – 13.5] | 1.2 [0.8 – 1.6] | 0.0 [0.0 – 0.0] | 17.8 [12.6 – 24.4] |
| 1-4 | 0.2 [0.2 – 0.3] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 2.9 [2.2 – 3.6] | 9.5 [7.1 – 11.9] | 1.1 [0.9 – 1.4] | 0.0 [0.0 – 0.0] | 13.0 [9.7 – 16.3] |
| 5-12 | 0.1 [0.1 – 0.1] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 1.5 [1.1 – 1.9] | 4.4 [3.2 – 5.6] | 0.5 [0.4 – 0.7] | 0.0 [0.0 – 0.0] | 6.2 [4.5 – 7.9] |
| 13-17 | 0.1 [0.0 – 0.1] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 3.0 [2.1 – 4.1] | 3.6 [2.4 – 5.0] | 0.5 [0.3 – 0.6] | 0.0 [0.0 – 0.0] | 6.6 [4.4 – 9.0] |
| 18-49 | 0.1 [0.1 – 0.1] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 1.4 [1.1 – 1.7] | 5.8 [4.7 – 7.3] | 0.6 [0.5 – 0.7] | 0.0 [0.0 – 0.0] | 7.4 [6.0 – 9.3] |
| 50-64 | 0.2 [0.1 – 0.2] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 2.9 [2.3 – 3.7] | 9.6 [7.5 – 11.9] | 2.5 [2.0 – 3.1] | 0.0 [0.0 – 0.0] | 14.2 [11.3 – 17.6] |
| 65+ | 1.1 [1.0 – 1.2] | 0.0 [0.0 – 0.0] | 0.0 [0.0 – 0.0] | 18.8 [16.0 – 21.0] | 60.8 [52.9 – 67.1] | 8.4 [7.2 – 9.3] | 0.0 [0.0 – 0.0] | 84.5 [73.8 – 93.1] |
| **All** | **0.3 [0.3 – 0.4]** | **0.0 [0.0 – 0.0]** | **0.0 [0.0 – 0.0]** | **5.0 [4.2 – 5.4]** | **16.2 [14.3 – 17.6]** | **2.4 [2.1 – 2.6]** | **0.0 [0.0 – 0.0]** | **22.5 [20.0 – 24.4]** |

**Hospitalizations averted per 100,000 additional doses**

Denominators are the *additional* doses each scenario schedules,
`max(0, 70% − scheduled coverage) × population`: 17,270 for age 0, 60,431 for 13-17, 877,865 for 18-49, 137,410 for 50-64, and zero for the
three groups already above 70% (shown as `—`). The `All` column's denominator
is the total 1,092,976 additional doses. As with the baseline, ~20% of
these extra doses reach people who are already immune.

| Age group (counted) | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 24.1 [17.2 – 33.8] | — | — | 3.4 [2.4 – 4.5] | 0.8 [0.6 – 1.1] | 0.6 [0.4 – 0.8] | — | 1.1 [0.8 – 1.6] |
| 1-4 | 3.3 [2.4 – 4.1] | — | — | 13.4 [10.1 – 16.6] | 3.0 [2.3 – 3.8] | 2.3 [1.7 – 2.9] | — | 3.3 [2.5 – 4.2] |
| 5-12 | 2.9 [2.1 – 3.8] | — | — | 14.9 [10.7 – 19.2] | 3.0 [2.2 – 3.9] | 2.4 [1.7 – 3.0] | — | 3.4 [2.5 – 4.4] |
| 13-17 | 1.5 [1.0 – 2.1] | — | — | 20.6 [14.0 – 28.2] | 1.7 [1.1 – 2.3] | 1.4 [0.9 – 1.9] | — | 2.5 [1.7 – 3.4] |
| 18-49 | 14.5 [11.5 – 18.2] | — | — | 67.3 [53.1 – 84.1] | 19.7 [15.9 – 24.7] | 12.4 [9.9 – 15.5] | — | 20.1 [16.2 – 25.3] |
| 50-64 | 13.6 [10.4 – 17.1] | — | — | 68.9 [53.0 – 86.6] | 15.5 [12.2 – 19.3] | 26.0 [20.9 – 32.2] | — | 18.5 [14.7 – 23.0] |
| 65+ | 78.3 [67.4 – 86.9] | — | — | 380.8 [323.8 – 425.1] | 84.6 [73.6 – 93.3] | 74.5 [64.4 – 82.5] | — | 94.4 [82.5 – 104.0] |
| **All** | **139.3 [123.8 – 150.7]** | **—** | **—** | **573.1 [489.0 – 629.4]** | **128.8 [113.6 – 140.1]** | **120.3 [105.8 – 131.9]** | **—** | **144.1 [128.1 – 156.2]** |

Per additional dose, raising **13-17** to 70% is by far the best buy:
573.1 [489.0 – 629.4] hospitalizations averted per 100,000 doses, against
139.3 for age 0 and 128.8 for 18-49, and
380.8 of that lands in **65+**, not in 13-17 itself. 18-49 dominates the
*absolute* totals only because it absorbs 877,865 of the 1,092,976 additional doses.

### Table S.A.6 — Additional hospitalizations averted at 70% coverage, across VE scenarios

For each VE sensitivity scenario, compares that scenario's own baseline
vaccination to the 70%-coverage floor applied to every eligible age group.

**Hospitalizations averted (count)**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 10 [7 – 13] | 12 [9 – 17] | 10 [7 – 15] |
| 1-4 | 24 [18 – 29] | 36 [27 – 46] | 23 [16 – 29] |
| 5-12 | 23 [15 – 31] | 38 [27 – 48] | 27 [20 – 34] |
| 13-17 | 20 [13 – 28] | 27 [18 – 37] | 21 [14 – 29] |
| 18-49 | 131 [97 – 168] | 220 [177 – 276] | 195 [159 – 244] |
| 50-64 | 124 [88 – 162] | 202 [161 – 251] | 173 [139 – 210] |
| 65+ | 631 [490 – 747] | 1,032 [901 – 1,136] | 887 [803 – 969] |
| **All** | **969 [749 – 1,127]** | **1,575 [1,401 – 1,707]** | **1,339 [1,235 – 1,439]** |

**% reduction in hospitalizations**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 15.7% [14.1% – 16.7%] | 33.5% [31.4% – 34.8%] | 48.3% [47.0% – 49.6%] |
| 1-4 | 9.1% [7.5% – 10.1%] | 24.5% [22.2% – 25.9%] | 33.3% [31.7% – 34.8%] |
| 5-12 | 8.6% [7.0% – 9.8%] | 23.3% [20.8% – 24.8%] | 31.7% [30.1% – 33.4%] |
| 13-17 | 13.8% [11.9% – 15.0%] | 30.2% [27.6% – 31.7%] | 42.6% [41.0% – 44.2%] |
| 18-49 | 11.0% [9.3% – 12.1%] | 30.0% [27.5% – 31.4%] | 42.7% [41.2% – 44.0%] |
| 50-64 | 9.8% [8.1% – 10.9%] | 26.9% [24.5% – 28.3%] | 38.5% [37.1% – 39.8%] |
| 65+ | 9.3% [7.8% – 10.3%] | 25.1% [22.9% – 26.4%] | 35.7% [34.4% – 37.0%] |
| **All** | **9.6% [8.1% – 10.7%]** | **26.0% [23.6% – 27.3%]** | **37.0% [35.6% – 38.3%]** |

**Hospitalizations averted per 100,000 population**

| Age group | Low VE | Baseline VE (fitted) | High VE |
|---|---|---|---|
| 0 | 13.9 [9.4 – 18.0] | 17.8 [12.6 – 24.4] | 14.6 [10.4 – 20.8] |
| 1-4 | 8.7 [6.3 – 10.4] | 13.0 [9.7 – 16.3] | 8.1 [5.8 – 10.4] |
| 5-12 | 3.9 [2.5 – 5.1] | 6.2 [4.5 – 7.9] | 4.5 [3.2 – 5.6] |
| 13-17 | 4.9 [3.3 – 6.8] | 6.6 [4.4 – 9.0] | 5.2 [3.4 – 7.0] |
| 18-49 | 4.4 [3.2 – 5.6] | 7.4 [6.0 – 9.3] | 6.5 [5.3 – 8.2] |
| 50-64 | 8.7 [6.2 – 11.4] | 14.2 [11.3 – 17.6] | 12.1 [9.7 – 14.7] |
| 65+ | 51.7 [40.1 – 61.2] | 84.5 [73.8 – 93.1] | 72.6 [65.8 – 79.3] |
| **All** | **13.9 [10.7 – 16.1]** | **22.5 [20.0 – 24.4]** | **19.2 [17.7 – 20.6]** |

The 70% floor averts 1,575 additional hospitalizations under the
fitted VE (26.0% of the remaining burden), 969 under `Low VE`
and 1,339 under `High VE`. The count is lower under `High VE`
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

Baseline schedule, median across the 638 posterior draws:

| Age group | Population | Scheduled doses | Delivered doses (median) | Scheduled but not delivered | % not delivered | Scheduled coverage | Delivered coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 70,067 | 31,777 | 25,066 | 6,711 | 21.1% | 45.4% | 35.8% |
| 1-4 | 280,268 | 254,216 | 200,464 | 53,753 | 21.1% | 90.7% | 71.5% |
| 5-12 | 606,291 | 433,128 | 335,772 | 97,357 | 22.5% | 71.4% | 55.4% |
| 13-17 | 411,782 | 227,816 | 177,595 | 50,221 | 22.0% | 55.3% | 43.1% |
| 18-49 | 2,978,204 | 1,206,878 | 919,221 | 287,657 | 23.8% | 40.5% | 30.9% |
| 50-64 | 1,424,434 | 859,693 | 659,698 | 199,996 | 23.3% | 60.4% | 46.3% |
| 65+ | 1,221,349 | 894,101 | 693,708 | 200,393 | 22.4% | 73.2% | 56.8% |
| **All** | **6,992,395** | **3,907,610** | **3,011,522** | **896,088** | **22.9%** | **55.9%** | **43.1%** |

22.9% of scheduled doses are not delivered overall (3.3% in the earlier fit),
ranging from 21.1% to 23.8% by age group. About 20 points
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

| Age group | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 42.1% [41.6% – 42.9%] | — | — | — | — | — | — | 38.9% [38.2% – 39.9%] |
| 1-4 | — | 20.4% [19.6% – 21.9%] | — | — | — | — | — | 16.2% [15.3% – 17.6%] |
| 5-12 | — | — | 34.8% [33.8% – 36.1%] | — | — | — | — | 30.5% [29.2% – 32.2%] |
| 13-17 | — | — | — | 44.2% [43.3% – 45.5%] | — | — | — | 39.7% [38.5% – 41.3%] |
| 18-49 | — | — | — | — | 57.5% [56.6% – 58.8%] | — | — | 53.3% [52.1% – 55.0%] |
| 50-64 | — | — | — | — | — | 52.3% [51.4% – 53.6%] | — | 47.8% [46.5% – 49.4%] |
| 65+ | — | — | — | — | — | — | 61.7% [60.9% – 62.5%] | 57.0% [55.6% – 58.6%] |

**Matched-cohort infection attack-rate ratio**, the estimate comparable to
real-world VE. It is ≈ `vax_susceptibility` (0.57 / 0.79 / 1.00) as expected.

| Age group | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 59.3% [58.8% – 60.0%] | — | — | — | — | — | — | 57.8% [57.6% – 58.2%] |
| 1-4 | — | 59.4% [58.9% – 60.3%] | — | — | — | — | — | 57.8% [57.6% – 58.4%] |
| 5-12 | — | — | 59.7% [59.1% – 60.8%] | — | — | — | — | 58.3% [57.9% – 59.0%] |
| 13-17 | — | — | — | 60.3% [59.6% – 61.5%] | — | — | — | 58.4% [58.0% – 59.2%] |
| 18-49 | — | — | — | — | 80.8% [80.5% – 81.5%] | — | — | 79.8% [79.5% – 80.2%] |
| 50-64 | — | — | — | — | — | 80.7% [80.4% – 81.4%] | — | 79.7% [79.5% – 80.1%] |
| 65+ | — | — | — | — | — | — | 99.9% [99.9% – 99.9%] | 99.9% [99.9% – 99.9%] |

**Hospitalization-given-infection rate ratio** (`IV_to_H / SV_to_EV` vs. `I_to_H / S_to_E`),
≈ `IV_to_H_prop / I_to_H_prop` (0.91 for ages under 65, 0.69 for 65+).

| Age group | 0 vaccinated | 1-4 vaccinated | 5-12 vaccinated | 13-17 vaccinated | 18-49 vaccinated | 50-64 vaccinated | 65+ vaccinated | All vaccinated |
|---|---|---|---|---|---|---|---|---|
| 0 | 91.1% [91.0% – 91.2%] | — | — | — | — | — | — | 91.0% [90.8% – 91.1%] |
| 1-4 | — | 91.0% [90.7% – 91.1%] | — | — | — | — | — | 90.7% [90.3% – 91.0%] |
| 5-12 | — | — | 91.0% [90.7% – 91.1%] | — | — | — | — | 90.9% [90.6% – 91.1%] |
| 13-17 | — | — | — | 91.1% [90.9% – 91.2%] | — | — | — | 91.0% [90.8% – 91.1%] |
| 18-49 | — | — | — | — | 91.0% [90.9% – 91.1%] | — | — | 90.9% [90.7% – 91.0%] |
| 50-64 | — | — | — | — | — | 91.0% [90.8% – 91.1%] | — | 90.9% [90.6% – 91.0%] |
| 65+ | — | — | — | — | — | — | 68.9% [68.8% – 69.0%] | 68.7% [68.4% – 68.9%] |
