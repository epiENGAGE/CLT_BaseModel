# MA_vax: vaccination-impact analysis report

Massachusetts, 2025–2026 influenza season (2025-09-01 to 2026-07-04, 307 days). Age-structured SEIR model with a parallel vaccinated arm, calibrated to daily hospital admissions by age group with Bayesian MCMC. This report documents the model, the fit and the vaccination-impact analysis, including scenarios that raise vaccination coverage to 50–70% or cut the season's doses by 5–20%.

Every count is a **hospital admission over the whole 307-day season** unless stated otherwise. Intervals are 95% intervals across 715 simulations, one per posterior parameter draw.

## Summary

- The fitted baseline produces **8,264** [7,714 – 8,789] hospitalizations, against **17,709** with no vaccination: the 38.6% overall coverage achieved averts **9,459 [7,293 – 11,426]** hospitalizations (**53.4% [47.1% – 58.1%]**).
- Raising every age group to a coverage target (groups already above it unchanged) averts 2,446 [2,060 – 2,731] more at 50%, 3,146 [2,697 – 3,461] more at 55%, 3,778 [3,298 – 4,122] more at 60%, 4,368 [3,894 – 4,744] more at 65%, 4,898 [4,427 – 5,296] more at 70% (vs. baseline).
- Removing doses from every age group adds 378 [311 – 430] at −5%, 767 [627 – 875] at −10%, 1,166 [949 – 1,333] at −15%, 1,574 [1,276 – 1,809] at −20% hospitalizations.
- Per dose, vaccinating school-age groups prevents the most hospitalizations across the population (Table S.A.2: 13-17 1,070 and 5-12 1,030 per 100,000 doses), mostly through indirect protection of 65+.

---

## 1. Model structure

### 1.1 Compartments

Nine compartments, each stratified by 7 age groups (`0`, `1-4`, `5-12`, `13-17`, `18-49`, `50-64`, `65+`):

| Compartment | Meaning |
|---|---|
| `S` | Susceptible, unvaccinated |
| `E` | Exposed (latent), unvaccinated |
| `I` | Infectious, unvaccinated |
| `R` | Recovered (either arm) |
| `SV` | Susceptible, vaccinated |
| `EV` | Exposed (latent), vaccinated |
| `IV` | Infectious, vaccinated |
| `H` | Hospitalized (either arm) |
| `D` | Dead (either arm) |

![Model structure](../figures/static/model_diagram.png)

*Model structure*

The vaccinated arm (`SV → EV → IV → H/R`) mirrors the unvaccinated arm (`S → E → I → H/R`) with its own susceptibility and severity parameters. `H`, `R` and `D` are shared, so vaccination history is not tracked once hospitalized or recovered.

### 1.2 Force of infection

Contacts come from fixed age × age contact matrices (Mistry et al. 2021), with school and work contacts removed on non-school / non-work days:

```
C(t) = total_C − (1 − is_school(t))·school_C − (1 − is_work(t))·work_C
beta_adj(t) = beta_baseline · m(t) · (1 + humidity_impact · exp(−180 · humidity(t)))
wtd_inf_prop(t) = (I·I_relative_infectiousness + IV·IV_relative_infectiousness) / population
foi(t) = beta_adj(t) · (C(t) @ wtd_inf_prop(t))
S_to_E   = foi(t) · relative_suscept   · S
SV_to_EV = foi(t) · vax_susceptibility · SV
```

`m(t)` is a smoothly time-varying transmission multiplier (§2.3) that absorbs behavioural and seasonal variation the mechanistic terms do not capture. `1 − vax_susceptibility` is the vaccine effectiveness (VE) against infection.

### 1.3 Progression, hospitalization, death

```
E_to_I  = E_to_I_rate · E                 EV_to_IV = EV_to_IV_rate · EV
I_to_H  = I_out_rate · I_to_H_prop · I     IV_to_H  = I_out_rate · IV_to_H_prop · IV
I_to_R  = I_out_rate · (1−I_to_H_prop)·I   IV_to_R  = I_out_rate · (1−IV_to_H_prop)·IV
H_to_D  = H_out_rate · H_to_D_prop · H
H_to_R  = H_out_rate · (1−H_to_D_prop)·H
```

### 1.4 Vaccination flow

Daily doses (an age-specific share of the population, applied 14 days after the dose date to represent the delay to immunity) move an exact count from `S` to `SV` each day, capped at what remains in `S`:

```
S_to_SV(t) = min(round(vax_prop(t − delay) · (S + SV)), S)
```

A dose scheduled for someone who has already been infected is not delivered by the model (the cap). Those doses are still bought and given in reality, so every "per 100,000 doses" figure divides by **scheduled** doses (appendix B).

### 1.5 Simulation and uncertainty

Explicit Euler integration with 7 sub-steps per day. Every interval comes from a **posterior parameter ensemble**: one deterministic run per posterior draw (715 draws), the same draws for every scenario, so differences between scenarios are paired. Intervals therefore reflect calibration uncertainty, not epidemic-process noise.

---

## 2. Parameters

### 2.1 Fixed scalar parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `relative_suscept` | 1.0 | Susceptibility multiplier, unvaccinated arm |
| `I_relative_infectiousness` | 1.0 | Infectiousness weight, unvaccinated `I` |
| `IV_relative_infectiousness` | 1.0 | Infectiousness weight, vaccinated `IV` |
| `E_to_I_rate` | 0.5 | Latent → infectious rate (/day) |
| `EV_to_IV_rate` | 0.5 | Same, vaccinated arm (/day) |
| `I_out_rate` | 0.333 | Infectious-period exit rate (/day) |
| `H_out_rate` | 0.17 | Hospital exit rate (/day) |
| `vax_transfer_delay_days` | 14.0 | Days from dose to modelled immunity |

### 2.2 Age-stratified parameters

| Age group | Population | Hosp. risk given infection¹ | Hosp. risk given breakthrough¹ | Death risk given hosp. | Residual susceptibility if vaccinated | Initial infections seeded |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 68,723 | 1.229% (0.899–1.388%) [0.697%] | 1.121% (0.820–1.266%) [0.636%] | 1.74% | 0.57 | 2 |
| 1-4 | 287,051 | 1.008% (0.633–1.350%) [0.697%] | 0.919% (0.577–1.231%) [0.636%] | 1.74% | 0.57 | 8 |
| 5-12 | 625,439 | 0.354% (0.219–0.505%) [0.274%] | 0.323% (0.200–0.461%) [0.250%] | 1.17% | 0.57 | 18 |
| 13-17 | 406,983 | 0.186% (0.116–0.277%) [0.274%] | 0.170% (0.106–0.253%) [0.250%] | 1.17% | 0.57 | 12 |
| 18-49 | 3,112,693 | 0.215% (0.138–0.322%) [0.561%] | 0.196% (0.126–0.293%) [0.511%] | 2.63% | 0.79 | 89 |
| 50-64 | 1,351,323 | 0.647% (0.407–0.935%) [1.060%] | 0.589% (0.371–0.852%) [0.966%] | 6.30% | 0.79 | 39 |
| 65+ | 1,420,722 | 5.251% (3.439–7.388%) [9.091%] | 3.623% (2.373–5.098%) [6.273%] | 7.99% | 1.00 | 41 |

¹ Fitted: pre-fit value × the age group's fitted multiplier (§2.4), posterior mean (95% interval), pre-fit value in brackets. Both columns share the multiplier, so VE against hospitalization given infection is unchanged by the fit. Seeds are pre-fit values, scaled by one fitted multiplier. A residual susceptibility of 1.00 (65+) means vaccination blocks no infection there: it only reduces severity, and vaccinating 65+ has no indirect effect on other age groups.

### 2.3 Fitting method

Affine-invariant ensemble MCMC (64 walkers, 4000 iterations each) over the 307-day target window.

| Parameter | Prior | Role |
|---|---|---|
| `beta_baseline` | Uniform(0.015, 0.06) | Baseline transmission rate |
| `humidity_impact` | Uniform(0, 1) | Strength of humidity forcing |
| Initial-seed multiplier | Log-uniform(0.1, 10.0) | Multiplies the age-specific seed counts |
| Hospitalization-risk multiplier (× 7) | Uniform(0.1, 2.0) | Multiplies both hospitalization-risk columns of one age group |
| `m(t)` log-increments (× 22) | Normal(0, 0.25) | Random-walk steps in log-transmission, knots every 14 days |
| `phi` | — | Negative-binomial dispersion (likelihood only) |

**Likelihood**: negative-binomial (NB2), jointly over 8 targets: daily admissions for each age group (weights 0_0: 1, 1_4: 1, 5_12: 1, 13_17: 1, 18_49: 2, 50_64: 2, 65plus: 5) and end-of-season cumulative admissions by age (weight 10).

**Posterior sample**: first 1700 iterations discarded, then every 200th kept, giving **715 draws**. "Best" is the draw with the highest log-posterior.

### 2.4 Fitted parameters

| Parameter | Posterior mean | 5% | 95% | "Best" |
|---|---:|---:|---:|---:|
| beta_baseline | 0.0334 | 0.0286 | 0.0396 | 0.0295 |
| humidity_impact | 0.35 | 0.0463 | 0.71 | 0.40 |
| Initial-seed multiplier | 3.96 | 1.43 | 7.30 | 3.27 |
| Hospitalization-risk multiplier – 0 | 1.76 | 1.39 | 1.98 | 1.91 |
| Hospitalization-risk multiplier – 1-4 | 1.45 | 0.97 | 1.89 | 1.35 |
| Hospitalization-risk multiplier – 5-12 | 1.29 | 0.87 | 1.77 | 1.04 |
| Hospitalization-risk multiplier – 13-17 | 0.68 | 0.45 | 0.95 | 0.58 |
| Hospitalization-risk multiplier – 18-49 | 0.38 | 0.27 | 0.53 | 0.29 |
| Hospitalization-risk multiplier – 50-64 | 0.61 | 0.42 | 0.83 | 0.55 |
| Hospitalization-risk multiplier – 65+ | 0.58 | 0.41 | 0.78 | 0.52 |
| phi (NB dispersion) | 105 | 27 | 313 | 57 |

![Humidity factor, fitted m(t) and their product (posterior median and 95% band)](../figures/output/transmission_multiplier.png)

*Humidity factor, fitted m(t) and their product (posterior median and 95% band)*

### 2.5 Vaccination coverage

Cumulative share of each age group vaccinated by the schedule over the simulated season (after the 14-day delay):

| Age group | Population | Scheduled coverage |
|---|---:|---:|
| 0 | 68,723 | 50.6% |
| 1-4 | 287,051 | 47.8% |
| 5-12 | 625,439 | 39.5% |
| 13-17 | 406,983 | 37.6% |
| 18-49 | 3,112,693 | 27.6% |
| 50-64 | 1,351,323 | 38.1% |
| 65+ | 1,420,722 | 60.8% |
| **All** | **7,272,934** | **38.6%** |

![Cumulative scheduled vaccination coverage](../figures/output/vax_coverage_cumulative.png)

*Cumulative scheduled vaccination coverage*

---

## 3. Vaccination impact of the 2025–2026 programme

The fitted baseline produces a median of **8,264** hospitalizations across all ages, against **17,709** with no vaccination.

![Daily hospitalizations: baseline, infection protection only, no vaccination](../figures/output/ts_baseline_infection_protection_no_vax.png)

*Daily hospitalizations: baseline, infection protection only, no vaccination*

### Table S.A.1 — Hospitalizations averted, infection vs. severity protection

**Infection protection** (no vaccination → VE against infection only)

| Age group | Hospitalizations averted | % averted | Per 100,000 population | Per 100,000 doses |
|---|---:|---:|---:|---:|
| 0 | 92 [66 – 108] | 53.4% [46.9% – 58.1%] | 133.5 [95.5 – 156.5] | 264.0 [188.9 – 309.5] |
| 1-4 | 368 [256 – 470] | 54.1% [47.5% – 59.0%] | 128.3 [89.1 – 163.7] | 268.6 [186.6 – 342.7] |
| 5-12 | 367 [237 – 496] | 50.7% [43.2% – 56.4%] | 58.6 [37.9 – 79.3] | 148.2 [95.8 – 200.5] |
| 13-17 | 125 [82 – 175] | 49.5% [41.8% – 55.3%] | 30.7 [20.2 – 43.1] | 81.6 [53.7 – 114.6] |
| 18-49 | 875 [620 – 1,229] | 46.2% [38.5% – 52.0%] | 28.1 [19.9 – 39.5] | 102.0 [72.3 – 143.3] |
| 50-64 | 1,061 [735 – 1,408] | 48.0% [40.7% – 53.5%] | 78.5 [54.4 – 104.2] | 205.9 [142.7 – 273.3] |
| 65+ | 5,401 [3,951 – 6,760] | 46.0% [38.8% – 51.3%] | 380.2 [278.1 – 475.8] | 624.8 [457.1 – 782.0] |
| **All** | **8,306 [6,166 – 10,261]** | **46.9% [39.7% – 52.2%]** | **114.2 [84.8 – 141.1]** | **295.6 [219.5 – 365.3]** |

**Severity protection** (adding VE against hospitalization given infection)

| Age group | Hospitalizations averted | % averted | Per 100,000 population | Per 100,000 doses |
|---|---:|---:|---:|---:|
| 0 | 2 [1 – 2] | 0.9% [0.8% – 1.0%] | 2.2 [1.4 – 3.2] | 4.4 [2.7 – 6.3] |
| 1-4 | 7 [5 – 9] | 1.1% [1.0% – 1.3%] | 2.6 [1.9 – 3.3] | 5.4 [3.9 – 6.8] |
| 5-12 | 7 [5 – 9] | 0.9% [0.8% – 1.1%] | 1.1 [0.8 – 1.4] | 2.7 [1.9 – 3.5] |
| 13-17 | 2 [2 – 3] | 0.9% [0.8% – 1.0%] | 0.5 [0.4 – 0.7] | 1.4 [1.0 – 1.9] |
| 18-49 | 18 [14 – 21] | 0.9% [0.8% – 1.1%] | 0.6 [0.5 – 0.7] | 2.1 [1.7 – 2.5] |
| 50-64 | 29 [24 – 36] | 1.3% [1.2% – 1.5%] | 2.1 [1.8 – 2.6] | 5.6 [4.6 – 6.9] |
| 65+ | 1,092 [990 – 1,186] | 9.3% [8.4% – 10.5%] | 76.9 [69.7 – 83.5] | 126.4 [114.5 – 137.2] |
| **All** | **1,156 [1,054 – 1,253]** | **6.5% [5.8% – 7.4%]** | **15.9 [14.5 – 17.2]** | **41.2 [37.5 – 44.6]** |

**Total** (no vaccination → baseline)

| Age group | Hospitalizations averted | % averted | Per 100,000 population | Per 100,000 doses |
|---|---:|---:|---:|---:|
| 0 | 93 [67 – 110] | 54.2% [47.9% – 58.9%] | 135.7 [97.2 – 159.4] | 268.4 [192.2 – 315.2] |
| 1-4 | 376 [262 – 479] | 55.2% [48.7% – 60.0%] | 130.8 [91.2 – 166.9] | 274.0 [190.9 – 349.6] |
| 5-12 | 373 [242 – 504] | 51.6% [44.3% – 57.2%] | 59.7 [38.7 – 80.6] | 151.0 [98.0 – 203.9] |
| 13-17 | 127 [84 – 178] | 50.3% [42.8% – 56.0%] | 31.2 [20.6 – 43.7] | 82.9 [54.8 – 116.2] |
| 18-49 | 892 [635 – 1,250] | 47.2% [39.6% – 52.8%] | 28.7 [20.4 – 40.2] | 104.1 [74.0 – 145.8] |
| 50-64 | 1,089 [762 – 1,440] | 49.3% [42.2% – 54.6%] | 80.6 [56.4 – 106.5] | 211.4 [147.9 – 279.4] |
| 65+ | 6,473 [5,013 – 7,904] | 55.3% [49.4% – 59.7%] | 455.6 [352.9 – 556.3] | 748.8 [579.9 – 914.4] |
| **All** | **9,459 [7,293 – 11,426]** | **53.4% [47.1% – 58.1%]** | **130.1 [100.3 – 157.1]** | **336.7 [259.6 – 406.7]** |

### Table S.A.2 — Hospitalizations averted by age group vaccinated

Each column vaccinates one age group only (all others unvaccinated) and compares with no vaccination; `All` is the full baseline schedule. Rows are the age group in which hospitalizations are counted, so off-diagonal cells are indirect protection.

**Hospitalizations averted**

| Counted in | 0 vacc. | 1-4 vacc. | 5-12 vacc. | 13-17 vacc. | 18-49 vacc. | 50-64 vacc. | 65+ vacc. | All vacc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25 [17 – 31] | 8 [6 – 9] | 25 [18 – 29] | 15 [11 – 17] | 25 [18 – 28] | 11 [8 – 13] | 0 [-0 – 0] | 93 [67 – 110] |
| 1-4 | 3 [2 – 3] | 142 [103 – 180] | 103 [68 – 135] | 57 [37 – 76] | 90 [59 – 117] | 41 [27 – 54] | 0 [-0 – 0] | 376 [262 – 479] |
| 5-12 | 2 [1 – 3] | 25 [16 – 35] | 201 [133 – 269] | 57 [35 – 81] | 79 [47 – 112] | 37 [22 – 52] | 0 [-0 – 0] | 373 [242 – 504] |
| 13-17 | 1 [0 – 1] | 7 [4 – 11] | 30 [19 – 45] | 57 [38 – 79] | 27 [17 – 41] | 14 [8 – 20] | 0 [-0 – 0] | 127 [84 – 178] |
| 18-49 | 6 [4 – 9] | 66 [46 – 96] | 243 [164 – 357] | 159 [107 – 236] | 349 [252 – 487] | 123 [85 – 177] | 0 [-0 – 0] | 892 [635 – 1,250] |
| 50-64 | 7 [5 – 9] | 75 [49 – 102] | 286 [185 – 394] | 194 [125 – 267] | 300 [198 – 409] | 340 [249 – 441] | 0 [-0 – 0] | 1,089 [762 – 1,440] |
| 65+ | 41 [30 – 53] | 439 [313 – 563] | 1,652 [1,153 – 2,128] | 1,095 [760 – 1,415] | 1,656 [1,185 – 2,120] | 898 [653 – 1,142] | 2,008 [1,707 – 2,321] | 6,473 [5,013 – 7,904] |
| **All** | **86 [71 – 98]** | **763 [568 – 944]** | **2,546 [1,801 – 3,247]** | **1,637 [1,151 – 2,112]** | **2,530 [1,830 – 3,223]** | **1,467 [1,091 – 1,853]** | **2,008 [1,707 – 2,321]** | **9,459 [7,293 – 11,426]** |

**% reduction**

| Counted in | 0 vacc. | 1-4 vacc. | 5-12 vacc. | 13-17 vacc. | 18-49 vacc. | 50-64 vacc. | 65+ vacc. | All vacc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.8% [14.3% – 15.3%] | 4.4% [3.7% – 5.0%] | 14.5% [11.7% – 16.8%] | 8.9% [7.1% – 10.4%] | 14.3% [11.9% – 16.5%] | 6.5% [5.3% – 7.5%] | 0.0% [-0.0% – 0.0%] | 54.2% [47.9% – 58.9%] |
| 1-4 | 0.4% [0.3% – 0.5%] | 20.9% [19.9% – 21.9%] | 15.2% [12.3% – 17.6%] | 8.4% [6.6% – 9.9%] | 13.2% [10.7% – 15.3%] | 6.0% [4.8% – 7.0%] | 0.0% [-0.0% – 0.0%] | 55.2% [48.7% – 60.0%] |
| 5-12 | 0.3% [0.2% – 0.3%] | 3.5% [2.8% – 4.1%] | 27.7% [24.5% – 30.4%] | 7.9% [6.1% – 9.5%] | 10.9% [8.5% – 13.1%] | 5.1% [4.0% – 6.2%] | 0.0% [-0.0% – 0.0%] | 51.6% [44.3% – 57.2%] |
| 13-17 | 0.3% [0.2% – 0.3%] | 2.8% [2.2% – 3.4%] | 12.1% [9.4% – 14.5%] | 22.5% [20.1% – 24.6%] | 10.9% [8.5% – 13.1%] | 5.4% [4.2% – 6.5%] | 0.0% [-0.0% – 0.0%] | 50.3% [42.8% – 56.0%] |
| 18-49 | 0.3% [0.3% – 0.4%] | 3.5% [2.8% – 4.1%] | 12.9% [10.1% – 15.3%] | 8.5% [6.6% – 10.1%] | 18.4% [15.8% – 20.7%] | 6.5% [5.3% – 7.6%] | 0.0% [-0.0% – 0.0%] | 47.2% [39.6% – 52.8%] |
| 50-64 | 0.3% [0.3% – 0.4%] | 3.3% [2.7% – 3.9%] | 12.9% [10.1% – 15.2%] | 8.7% [6.9% – 10.3%] | 13.5% [11.0% – 15.7%] | 15.4% [14.0% – 16.7%] | 0.0% [-0.0% – 0.0%] | 49.3% [42.2% – 54.6%] |
| 65+ | 0.4% [0.3% – 0.4%] | 3.7% [3.1% – 4.3%] | 14.1% [11.4% – 16.3%] | 9.3% [7.5% – 10.9%] | 14.1% [11.7% – 16.2%] | 7.6% [6.4% – 8.7%] | 17.2% [17.0% – 17.3%] | 55.3% [49.4% – 59.7%] |
| **All** | **0.5% [0.4% – 0.5%]** | **4.3% [3.6% – 4.9%]** | **14.4% [11.7% – 16.6%]** | **9.2% [7.4% – 10.8%]** | **14.3% [11.8% – 16.5%]** | **8.3% [7.0% – 9.4%]** | **11.4% [10.8% – 11.9%]** | **53.4% [47.1% – 58.1%]** |

**Per 100,000 doses given to the column's age group**

| Counted in | 0 vacc. | 1-4 vacc. | 5-12 vacc. | 13-17 vacc. | 18-49 vacc. | 50-64 vacc. | 65+ vacc. | All vacc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 72.7 [50.1 – 90.6] | 5.5 [4.0 – 6.4] | 10.1 [7.3 – 11.6] | 10.0 [7.3 – 11.4] | 2.9 [2.1 – 3.3] | 2.2 [1.6 – 2.5] | 0.0 [-0.0 – 0.0] | 268.4 [192.2 – 315.2] |
| 1-4 | 7.7 [5.2 – 10.0] | 103.8 [75.1 – 131.5] | 41.8 [27.4 – 54.7] | 37.3 [24.1 – 49.4] | 10.5 [6.9 – 13.7] | 7.9 [5.2 – 10.4] | 0.0 [-0.0 – 0.0] | 274.0 [190.9 – 349.6] |
| 5-12 | 6.0 [3.6 – 8.4] | 18.5 [11.4 – 25.8] | 81.3 [53.9 – 108.9] | 37.6 [22.7 – 52.9] | 9.2 [5.5 – 13.0] | 7.2 [4.3 – 10.2] | 0.0 [-0.0 – 0.0] | 151.0 [98.0 – 203.9] |
| 13-17 | 1.9 [1.2 – 2.8] | 5.2 [3.2 – 7.7] | 12.3 [7.6 – 18.3] | 37.2 [25.1 – 51.3] | 3.2 [2.0 – 4.8] | 2.6 [1.6 – 3.9] | 0.0 [-0.0 – 0.0] | 82.9 [54.8 – 116.2] |
| 18-49 | 18.5 [12.8 – 26.7] | 48.5 [33.5 – 70.3] | 98.2 [66.5 – 144.5] | 104.2 [70.1 – 153.9] | 40.7 [29.4 – 56.8] | 23.9 [16.5 – 34.4] | 0.0 [-0.0 – 0.0] | 104.1 [74.0 – 145.8] |
| 50-64 | 19.9 [13.1 – 27.3] | 54.4 [35.5 – 74.4] | 115.7 [74.7 – 159.3] | 126.9 [81.6 – 174.6] | 35.0 [23.1 – 47.7] | 66.0 [48.4 – 85.5] | 0.0 [-0.0 – 0.0] | 211.4 [147.9 – 279.4] |
| 65+ | 118.8 [85.5 – 152.3] | 320.5 [228.1 – 410.9] | 668.1 [466.4 – 860.6] | 715.6 [496.7 – 924.7] | 193.1 [138.2 – 247.2] | 174.3 [126.7 – 221.7] | 232.3 [197.5 – 268.5] | 748.8 [579.9 – 914.4] |
| **All** | **246.4 [205.3 – 282.1]** | **556.8 [414.0 – 688.9]** | **1029.5 [728.2 – 1313.2]** | **1069.5 [751.7 – 1379.7]** | **295.0 [213.4 – 375.9]** | **284.7 [211.8 – 359.6]** | **232.3 [197.5 – 268.5]** | **336.7 [259.6 – 406.7]** |

![Share of each age group's averted hospitalizations, by age group vaccinated](../figures/output/S_A_2_pct_reduction_stacked_bar.png)

*Share of each age group's averted hospitalizations, by age group vaccinated*

### Table S.A.4 — Vaccine-effectiveness sensitivity scenarios

| Scenario | Age group | VE_infection | VE_hosp_infection | VE_hosp_given_infection | VE_transmission_blocking |
|---|---|---:|---:|---:|---:|
| low_ve | 0 | 33% | 33% | 0% | 0% |
| low_ve | 1-4 | 33% | 33% | 0% | 0% |
| low_ve | 5-12 | 33% | 33% | 0% | 0% |
| low_ve | 13-17 | 33% | 33% | 0% | 0% |
| low_ve | 18-49 | 6% | 10% | 4% | 0% |
| low_ve | 50-64 | 6% | 10% | 4% | 0% |
| low_ve | 65+ | 0% | 21% | 21% | 0% |
| baseline_ve | 0 | 43% | 48% | 9% | 0% |
| baseline_ve | 1-4 | 43% | 48% | 9% | 0% |
| baseline_ve | 5-12 | 43% | 48% | 9% | 0% |
| baseline_ve | 13-17 | 43% | 48% | 9% | 0% |
| baseline_ve | 18-49 | 21% | 28% | 9% | 0% |
| baseline_ve | 50-64 | 21% | 28% | 9% | 0% |
| baseline_ve | 65+ | 0% | 31% | 31% | 0% |
| high_ve | 0 | 51% | 78% | 55% | 0% |
| high_ve | 1-4 | 51% | 78% | 55% | 0% |
| high_ve | 5-12 | 51% | 78% | 55% | 0% |
| high_ve | 13-17 | 51% | 78% | 55% | 0% |
| high_ve | 18-49 | 34% | 44% | 16% | 0% |
| high_ve | 50-64 | 34% | 44% | 16% | 0% |
| high_ve | 65+ | 14% | 39% | 29% | 0% |

### Table S.A.5 — Hospitalizations averted across VE scenarios (vs. no vaccination)

**Hospitalizations averted**

| Age group | Low VE | Baseline VE | High VE |
|---|---:|---:|---:|
| 0 | 59 [43 – 69] | 93 [67 – 110] | 121 [85 – 147] |
| 1-4 | 242 [167 – 311] | 376 [262 – 479] | 490 [350 – 622] |
| 5-12 | 238 [152 – 326] | 373 [242 – 504] | 497 [329 – 656] |
| 13-17 | 79 [52 – 113] | 127 [84 – 178] | 169 [114 – 232] |
| 18-49 | 500 [347 – 715] | 892 [635 – 1,250] | 1,201 [879 – 1,643] |
| 50-64 | 607 [406 – 824] | 1,089 [762 – 1,440] | 1,451 [1,049 – 1,870] |
| 65+ | 4,124 [3,150 – 5,107] | 6,473 [5,013 – 7,904] | 8,205 [6,525 – 9,868] |
| **All** | **5,863 [4,427 – 7,193]** | **9,459 [7,293 – 11,426]** | **12,186 [9,678 – 14,378]** |

**% reduction**

| Age group | Low VE | Baseline VE | High VE |
|---|---:|---:|---:|
| 0 | 34.3% [29.8% – 38.1%] | 54.2% [47.9% – 58.9%] | 70.9% [65.2% – 74.8%] |
| 1-4 | 35.5% [30.8% – 39.4%] | 55.2% [48.7% – 60.0%] | 72.2% [66.4% – 76.1%] |
| 5-12 | 32.8% [27.7% – 37.2%] | 51.6% [44.3% – 57.2%] | 68.3% [61.3% – 73.2%] |
| 13-17 | 31.5% [26.3% – 35.9%] | 50.3% [42.8% – 56.0%] | 67.1% [59.8% – 72.1%] |
| 18-49 | 26.3% [21.4% – 30.6%] | 47.2% [39.6% – 52.8%] | 63.3% [55.3% – 68.5%] |
| 50-64 | 27.4% [22.5% – 31.5%] | 49.3% [42.2% – 54.6%] | 65.7% [58.5% – 70.6%] |
| 65+ | 35.2% [31.0% – 38.7%] | 55.3% [49.4% – 59.7%] | 70.1% [64.2% – 74.1%] |
| **All** | **33.1% [28.7% – 36.8%]** | **53.4% [47.1% – 58.1%]** | **68.8% [62.4% – 73.0%]** |

![Daily hospitalizations under low, baseline and high VE](../figures/output/ts_ve.png)

*Daily hospitalizations under low, baseline and high VE*

---

## 4. Coverage scenarios

Two families of counterfactual schedules, each applied either to **every age group at once** or to **one age group at a time** (others keep the baseline schedule), and compared with the baseline:

- **Coverage targets** (50%, 55%, 60%, 65%, 70%): each targeted group's daily doses are scaled by `target / baseline coverage`, so its season coverage reaches the target. A group already above the target keeps its schedule.

- **Fewer doses** (−5%, −10%, −15%, −20%): each targeted group's daily doses are cut by that share (a relative decrease, e.g. 60% coverage becomes 54% at −10%). Timing is unchanged.

### 4.1 All age groups together

| Scenario | Overall coverage | Doses vs. baseline | Hospitalizations | Change vs. baseline | % change |
|---|---:|---:|---:|---:|---:|
| No vaccination | 0.0% | -2,809,361 | 17,709 [15,358 – 19,823] | +9,459 [+7,293 – +11,426] | +114.8% [+88.9% – +138.6%] |
| -20% | 30.9% | -561,872 | 9,837 [9,194 – 10,472] | +1,574 [+1,276 – +1,809] | +19.1% [+15.5% – +21.9%] |
| -15% | 32.8% | -421,404 | 9,427 [8,817 – 10,008] | +1,166 [+949 – +1,333] | +14.1% [+11.5% – +16.1%] |
| -10% | 34.8% | -280,936 | 9,031 [8,439 – 9,602] | +767 [+627 – +875] | +9.3% [+7.6% – +10.6%] |
| -5% | 36.7% | -140,468 | 8,639 [8,079 – 9,200] | +378 [+311 – +430] | +4.6% [+3.8% – +5.2%] |
| Baseline | 38.6% | +0 | 8,264 [7,714 – 8,789] | +0 [+0 – +0] | +0.0% [+0.0% – +0.0%] |
| 50% target | 52.1% | +981,541 | 5,835 [5,338 – 6,344] | -2,446 [-2,731 – -2,060] | -29.7% [-32.5% – -24.9%] |
| 55% target | 56.1% | +1,273,763 | 5,133 [4,667 – 5,663] | -3,146 [-3,461 – -2,697] | -38.1% [-41.2% – -32.7%] |
| 60% target | 60.2% | +1,566,373 | 4,495 [4,062 – 5,028] | -3,778 [-4,122 – -3,298] | -45.8% [-48.8% – -40.1%] |
| 65% target | 65.0% | +1,918,046 | 3,893 [3,505 – 4,412] | -4,368 [-4,744 – -3,894] | -53.1% [-56.0% – -47.5%] |
| 70% target | 70.0% | +2,281,693 | 3,361 [3,015 – 3,822] | -4,898 [-5,296 – -4,427] | -59.5% [-62.2% – -54.4%] |

Overall coverage = scheduled doses summed over all age groups ÷ total population, i.e. the population-weighted average of each age group's scheduled coverage.

![Season hospitalizations against overall scheduled coverage](../figures/output/coverage_response.png)

*Season hospitalizations against overall scheduled coverage*

![Daily hospitalizations: baseline and coverage targets (all ages)](../figures/output/ts_coverage_targets.png)

*Daily hospitalizations: baseline and coverage targets (all ages)*

![Daily hospitalizations: baseline and fewer doses (all ages)](../figures/output/ts_coverage_decreases.png)

*Daily hospitalizations: baseline and fewer doses (all ages)*

### 4.2 One age group at a time

Change in **total** (all-ages) hospitalizations when only the row's age group is changed, median across draws (95% intervals in appendix A):

![% change in total hospitalizations, by age group changed and level](../figures/output/coverage_heatmap.png)

*% change in total hospitalizations, by age group changed and level*

**Hospitalizations averted vs. baseline, all ages counted**

| Raised to target | 50% | 55% | 60% | 65% | 70% |
|---|---:|---:|---:|---:|---:|
| 0 only | 0 [0 – 0] | 5 [4 – 5] | 10 [10 – 11] | 16 [15 – 17] | 22 [20 – 23] |
| 1-4 only | 24 [20 – 27] | 77 [63 – 87] | 130 [107 – 146] | 182 [150 – 205] | 235 [193 – 263] |
| 5-12 only | 462 [364 – 532] | 675 [533 – 774] | 881 [700 – 1,007] | 1,083 [863 – 1,232] | 1,279 [1,024 – 1,450] |
| 13-17 only | 376 [294 – 435] | 522 [410 – 602] | 666 [525 – 766] | 806 [638 – 926] | 944 [749 – 1,081] |
| 18-49 only | 1,406 [1,163 – 1,591] | 1,699 [1,413 – 1,912] | 1,982 [1,660 – 2,224] | 2,257 [1,904 – 2,522] | 2,523 [2,139 – 2,807] |
| 50-64 only | 314 [260 – 354] | 443 [368 – 500] | 571 [475 – 644] | 697 [582 – 785] | 822 [687 – 924] |
| 65+ only | 0 [0 – 0] | 0 [0 – 0] | 0 [0 – 0] | 75 [68 – 81] | 164 [149 – 179] |
| All ages | 2,446 [2,060 – 2,731] | 3,146 [2,697 – 3,461] | 3,778 [3,298 – 4,122] | 4,368 [3,894 – 4,744] | 4,898 [4,427 – 5,296] |

**Additional hospitalizations vs. baseline, all ages counted**

| Doses cut | −5% | −10% | −15% | −20% |
|---|---:|---:|---:|---:|
| 0 only | 3 [3 – 3] | 6 [5 – 6] | 8 [8 – 9] | 11 [10 – 12] |
| 1-4 only | 26 [21 – 29] | 51 [42 – 58] | 77 [63 – 87] | 103 [84 – 116] |
| 5-12 only | 90 [70 – 104] | 181 [140 – 210] | 273 [211 – 317] | 365 [283 – 426] |
| 13-17 only | 58 [45 – 68] | 118 [91 – 137] | 177 [137 – 206] | 237 [183 – 276] |
| 18-49 only | 91 [73 – 105] | 183 [147 – 212] | 276 [221 – 319] | 369 [295 – 426] |
| 50-64 only | 51 [42 – 58] | 103 [85 – 117] | 154 [127 – 175] | 206 [170 – 234] |
| 65+ only | 55 [50 – 59] | 109 [99 – 119] | 164 [148 – 178] | 218 [198 – 237] |
| All ages | 378 [311 – 430] | 767 [627 – 875] | 1,166 [949 – 1,333] | 1,574 [1,276 – 1,809] |

### 4.3 Efficiency per dose

Hospitalizations averted per 100,000 **additional** doses (targets) and added per 100,000 doses **removed** (cuts), all ages counted. For targets the denominator is the extra doses the target requires in that group (`All ages`: in every group).

| Age group changed | 50% target | 55% target | 60% target | 65% target | 70% target | −5% | −10% | −15% | −20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 only | — | 161.6 [146.5 – 173.1] | 161.7 [146.7 – 172.7] | 161.7 [147.0 – 172.9] | 161.2 [146.6 – 172.2] | 160.7 [146.1 – 171.9] | 160.6 [145.8 – 171.4] | 161.8 [147.0 – 172.8] | 161.0 [145.9 – 172.2] |
| 1-4 only | 370.9 [304.3 – 417.7] | 370.4 [303.9 – 416.4] | 369.4 [303.6 – 415.6] | 368.3 [303.0 – 414.0] | 367.3 [302.4 – 412.5] | 372.3 [305.4 – 419.1] | 373.2 [305.8 – 420.6] | 373.9 [306.1 – 421.2] | 374.4 [306.3 – 421.9] |
| 5-12 only | 706.8 [555.6 – 812.5] | 697.6 [551.1 – 800.4] | 688.7 [546.6 – 787.2] | 679.8 [542.0 – 773.4] | 671.1 [537.6 – 760.8] | 728.6 [566.6 – 844.9] | 731.3 [568.0 – 849.1] | 735.4 [569.9 – 855.3] | 738.4 [571.3 – 860.5] |
| 13-17 only | 745.2 [583.8 – 861.7] | 737.8 [579.8 – 851.1] | 730.5 [575.8 – 840.8] | 723.1 [572.1 – 830.5] | 715.8 [568.3 – 820.2] | 763.6 [592.9 – 890.2] | 769.7 [596.5 – 896.3] | 771.6 [597.3 – 899.3] | 774.0 [598.3 – 902.4] |
| 18-49 only | 201.2 [166.5 – 227.6] | 198.8 [165.4 – 223.7] | 196.2 [164.3 – 220.2] | 193.6 [163.3 – 216.4] | 191.0 [161.9 – 212.5] | 213.2 [171.1 – 245.8] | 213.7 [171.3 – 246.7] | 214.3 [171.6 – 247.7] | 214.9 [171.8 – 248.6] |
| 50-64 only | 195.5 [162.1 – 220.9] | 194.3 [161.4 – 219.3] | 193.1 [160.8 – 217.7] | 192.0 [160.2 – 216.1] | 190.8 [159.6 – 214.5] | 199.1 [163.9 – 225.7] | 199.4 [164.1 – 226.2] | 199.8 [164.3 – 226.7] | 200.3 [164.5 – 227.4] |
| 65+ only | — | — | — | 126.4 [114.5 – 137.2] | 126.4 [114.5 – 137.2] | 126.4 [114.5 – 137.3] | 126.4 [114.5 – 137.3] | 126.4 [114.5 – 137.3] | 126.4 [114.5 – 137.3] |
| All ages | 249.2 [209.9 – 278.2] | 246.9 [211.8 – 271.7] | 241.2 [210.5 – 263.2] | 227.8 [203.0 – 247.3] | 214.7 [194.0 – 232.1] | 269.0 [221.1 – 305.9] | 273.0 [223.2 – 311.4] | 276.7 [225.2 – 316.4] | 280.1 [227.2 – 322.0] |

![50% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized](../figures/output/S_A_3_50pct_per_100k_doses_stacked_bar.png)

*50% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized*

![70% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized](../figures/output/S_A_3_70pct_per_100k_doses_stacked_bar.png)

*70% target: hospitalizations averted per 100,000 additional doses, by age group hospitalized*

![10% fewer doses: additional hospitalizations per 100,000 doses removed](../figures/output/DECREASE_10pct_per_100k_doses_stacked_bar.png)

*10% fewer doses: additional hospitalizations per 100,000 doses removed*

A relative dose cut removes the most doses where coverage is highest, so per removed dose its cost mirrors the per-dose benefit of Table S.A.2; coverage targets add doses mostly to the least-covered groups (18-49 needs the largest relative increase).

---

## 5. Baseline fit check

Cumulative admissions over 2025-09-01 – 2026-07-04:

| Age group | Simulated (median) | Simulated 95% interval | Observed | % difference |
|---|---:|---:|---:|---:|
| 0 | 78.6 | 49.1 – 110.8 | 114.0 | -31.0% |
| 1-4 | 305.9 | 221.6 – 390.9 | 302.9 | +1.0% |
| 5-12 | 348.7 | 253.0 – 448.8 | 347.0 | +0.5% |
| 13-17 | 125.1 | 90.4 – 163.8 | 123.0 | +1.7% |
| 18-49 | 1,013.6 | 821.3 – 1,227.7 | 984.7 | +2.9% |
| 50-64 | 1,120.5 | 921.0 – 1,385.7 | 1,108.7 | +1.1% |
| 65+ | 5,264.1 | 4,750.6 – 5,733.3 | 5,388.9 | -2.3% |
| **Total** | **8,263.8** | **7,714.3 – 8,789.1** | **8,369.1** | **-1.3%** |

![Daily admissions: posterior median and 95% band vs. observed](../figures/output/fit_check_daily_by_age.png)

*Daily admissions: posterior median and 95% band vs. observed*

![Cumulative admissions: posterior median and 95% band vs. observed](../figures/output/fit_check_cumulative_by_age.png)

*Cumulative admissions: posterior median and 95% band vs. observed*

---

## 6. Data sources

| Input | Source |
|---|---|
| Hospital admissions (calibration target) | Massachusetts Department of Public Health (MA DPH): weekly influenza admissions by county and age group (`influenza_timeseries_*.csv`, `ed_admits_count`), summed over counties and spread evenly over each MMWR week (`data/prepare_hospitalizations.py`) |
| Vaccination | MA DPH: weekly cumulative first doses by county and age (`MA_influenza_coverage_2025-2026.csv`), differenced and spread over each week (`data/prepare_vaccination.py`) |
| Population by age group | MA DPH: `POP_ESTIMATE` of the vaccination file, summed over counties |
| Contact matrices | Mistry et al. 2021 synthetic contact matrices |
| Absolute humidity | gridMET daily specific humidity averaged over Massachusetts (× 1.2 kg/m³, `data/extract_ma_humidity.py`) |
| School/work calendar | Constructed state school/work-day calendar |

## 7. Notes

- Intervals reflect calibration uncertainty (posterior draws, deterministic transitions), not epidemic-process noise.
- `m(t)` is a statistical smoothing device, not an independently measured behavioural signal.
- Coverage-target multipliers are computed from the scheduled coverage over the simulated window; the model's cap means delivered coverage lands slightly below the target (appendix B).
- "Per 100,000 doses" uses scheduled doses: doses the model declines to deliver to already-infected people are still bought and given.
- Vaccinating 65+ has no indirect effect on other age groups because vaccination blocks no infection in 65+ in the fitted model (residual susceptibility 1.00).

---

## Appendix A — Coverage-scenario tables by age group hospitalized

### 50% coverage target — hospitalizations averted vs. baseline

| Counted in | 0 raised | 1-4 raised | 5-12 raised | 13-17 raised | 18-49 raised | 50-64 raised | 65+ raised | All raised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 [0 – 0] | 0 [0 – 0] | 4 [3 – 5] | 3 [2 – 4] | 13 [9 – 17] | 2 [2 – 3] | 0 [0 – 0] | 23 [15 – 29] |
| 1-4 | 0 [0 – 0] | 4 [3 – 5] | 18 [13 – 23] | 13 [9 – 17] | 50 [35 – 63] | 9 [6 – 11] | 0 [0 – 0] | 89 [65 – 113] |
| 5-12 | 0 [0 – 0] | 1 [1 – 1] | 38 [26 – 48] | 15 [10 – 20] | 52 [34 – 67] | 10 [6 – 13] | 0 [0 – 0] | 107 [75 – 139] |
| 13-17 | 0 [0 – 0] | 0 [0 – 0] | 6 [4 – 9] | 13 [9 – 17] | 18 [12 – 25] | 4 [2 – 5] | 0 [0 – 0] | 39 [27 – 52] |
| 18-49 | 0 [0 – 0] | 3 [2 – 3] | 52 [38 – 71] | 44 [32 – 59] | 211 [165 – 273] | 32 [24 – 43] | 0 [0 – 0] | 323 [252 – 417] |
| 50-64 | 0 [0 – 0] | 3 [2 – 3] | 58 [41 – 74] | 49 [35 – 64] | 185 [138 – 234] | 72 [58 – 90] | 0 [0 – 0] | 344 [264 – 429] |
| 65+ | 0 [0 – 0] | 13 [10 – 16] | 283 [220 – 335] | 237 [184 – 281] | 872 [707 – 1,012] | 183 [148 – 214] | 0 [0 – 0] | 1,506 [1,245 – 1,727] |
| **All** | **0 [0 – 0]** | **24 [20 – 27]** | **462 [364 – 532]** | **376 [294 – 435]** | **1,406 [1,163 – 1,591]** | **314 [260 – 354]** | **0 [0 – 0]** | **2,446 [2,060 – 2,731]** |

### 55% coverage target — hospitalizations averted vs. baseline

| Counted in | 0 raised | 1-4 raised | 5-12 raised | 13-17 raised | 18-49 raised | 50-64 raised | 65+ raised | All raised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 [1 – 2] | 1 [1 – 1] | 6 [4 – 8] | 5 [3 – 6] | 16 [11 – 20] | 3 [2 – 4] | 0 [0 – 0] | 30 [20 – 38] |
| 1-4 | 0 [0 – 0] | 13 [9 – 17] | 27 [19 – 34] | 18 [13 – 23] | 60 [43 – 76] | 13 [9 – 16] | 0 [0 – 0] | 118 [87 – 151] |
| 5-12 | 0 [0 – 0] | 3 [2 – 4] | 55 [38 – 70] | 21 [14 – 28] | 63 [42 – 81] | 14 [9 – 18] | 0 [0 – 0] | 141 [98 – 181] |
| 13-17 | 0 [0 – 0] | 1 [1 – 1] | 9 [6 – 13] | 18 [13 – 24] | 22 [15 – 30] | 5 [3 – 7] | 0 [0 – 0] | 50 [36 – 67] |
| 18-49 | 0 [0 – 1] | 8 [6 – 11] | 77 [56 – 103] | 61 [44 – 82] | 254 [199 – 327] | 45 [34 – 60] | 0 [0 – 0] | 410 [322 – 525] |
| 50-64 | 0 [0 – 1] | 9 [6 – 11] | 84 [60 – 108] | 68 [49 – 88] | 224 [168 – 283] | 102 [82 – 126] | 0 [0 – 0] | 442 [346 – 548] |
| 65+ | 2 [2 – 3] | 42 [34 – 50] | 413 [322 – 487] | 329 [256 – 389] | 1,053 [860 – 1,215] | 259 [210 – 302] | 0 [0 – 0] | 1,937 [1,630 – 2,194] |
| **All** | **5 [4 – 5]** | **77 [63 – 87]** | **675 [533 – 774]** | **522 [410 – 602]** | **1,699 [1,413 – 1,912]** | **443 [368 – 500]** | **0 [0 – 0]** | **3,146 [2,697 – 3,461]** |

### 60% coverage target — hospitalizations averted vs. baseline

| Counted in | 0 raised | 1-4 raised | 5-12 raised | 13-17 raised | 18-49 raised | 50-64 raised | 65+ raised | All raised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 [2 – 4] | 1 [1 – 2] | 8 [6 – 10] | 6 [4 – 7] | 19 [13 – 24] | 4 [3 – 5] | 0 [0 – 0] | 37 [24 – 48] |
| 1-4 | 0 [0 – 0] | 22 [16 – 28] | 35 [25 – 44] | 23 [16 – 29] | 70 [51 – 89] | 16 [12 – 21] | 0 [0 – 0] | 144 [106 – 184] |
| 5-12 | 0 [0 – 0] | 5 [3 – 6] | 71 [50 – 91] | 27 [18 – 35] | 73 [49 – 95] | 18 [12 – 23] | 0 [0 – 0] | 169 [120 – 219] |
| 13-17 | 0 [0 – 0] | 1 [1 – 2] | 12 [8 – 17] | 23 [16 – 30] | 26 [18 – 35] | 6 [4 – 9] | 0 [0 – 0] | 61 [43 – 80] |
| 18-49 | 1 [1 – 1] | 14 [10 – 18] | 100 [74 – 134] | 77 [57 – 104] | 295 [231 – 379] | 59 [44 – 78] | 0 [0 – 0] | 488 [385 – 618] |
| 50-64 | 1 [1 – 1] | 14 [10 – 19] | 110 [79 – 141] | 87 [63 – 112] | 262 [196 – 330] | 131 [105 – 162] | 0 [0 – 0] | 530 [420 – 653] |
| 65+ | 5 [4 – 6] | 72 [57 – 84] | 540 [423 – 635] | 419 [327 – 494] | 1,228 [1,012 – 1,412] | 334 [271 – 389] | 0 [0 – 0] | 2,332 [1,997 – 2,615] |
| **All** | **10 [10 – 11]** | **130 [107 – 146]** | **881 [700 – 1,007]** | **666 [525 – 766]** | **1,982 [1,660 – 2,224]** | **571 [475 – 644]** | **0 [0 – 0]** | **3,778 [3,298 – 4,122]** |

### 65% coverage target — hospitalizations averted vs. baseline

| Counted in | 0 raised | 1-4 raised | 5-12 raised | 13-17 raised | 18-49 raised | 50-64 raised | 65+ raised | All raised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 [3 – 6] | 2 [1 – 2] | 10 [7 – 13] | 7 [5 – 9] | 22 [14 – 27] | 5 [4 – 7] | 0 [-0 – 0] | 42 [27 – 56] |
| 1-4 | 1 [0 – 1] | 31 [22 – 39] | 43 [30 – 54] | 28 [20 – 36] | 79 [58 – 101] | 20 [14 – 26] | 0 [-0 – 0] | 167 [123 – 214] |
| 5-12 | 0 [0 – 1] | 7 [5 – 9] | 86 [61 – 111] | 33 [22 – 43] | 84 [56 – 108] | 22 [14 – 28] | 0 [-0 – 0] | 195 [138 – 251] |
| 13-17 | 0 [0 – 0] | 2 [1 – 3] | 15 [10 – 21] | 28 [20 – 37] | 30 [21 – 40] | 8 [5 – 11] | 0 [-0 – 0] | 69 [49 – 91] |
| 18-49 | 1 [1 – 2] | 19 [14 – 26] | 123 [91 – 165] | 94 [69 – 126] | 334 [262 – 428] | 72 [54 – 95] | 0 [-0 – 0] | 557 [439 – 699] |
| 50-64 | 2 [1 – 2] | 20 [15 – 26] | 135 [98 – 174] | 106 [76 – 136] | 298 [225 – 374] | 159 [127 – 197] | 0 [-0 – 0] | 608 [485 – 751] |
| 65+ | 8 [6 – 9] | 101 [80 – 118] | 662 [522 – 778] | 507 [397 – 596] | 1,399 [1,159 – 1,603] | 408 [333 – 474] | 75 [68 – 81] | 2,723 [2,371 – 3,033] |
| **All** | **16 [15 – 17]** | **182 [150 – 205]** | **1,083 [863 – 1,232]** | **806 [638 – 926]** | **2,257 [1,904 – 2,522]** | **697 [582 – 785]** | **75 [68 – 81]** | **4,368 [3,894 – 4,744]** |

### 70% coverage target — hospitalizations averted vs. baseline

| Counted in | 0 raised | 1-4 raised | 5-12 raised | 13-17 raised | 18-49 raised | 50-64 raised | 65+ raised | All raised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 [3 – 8] | 2 [2 – 3] | 12 [8 – 15] | 9 [6 – 11] | 24 [16 – 31] | 6 [4 – 8] | 0 [-0 – 0] | 47 [30 – 63] |
| 1-4 | 1 [0 – 1] | 39 [29 – 50] | 50 [36 – 64] | 33 [23 – 42] | 89 [65 – 113] | 24 [17 – 30] | 0 [-0 – 0] | 187 [137 – 240] |
| 5-12 | 1 [0 – 1] | 9 [6 – 12] | 101 [72 – 130] | 38 [25 – 50] | 94 [64 – 122] | 25 [17 – 33] | 0 [-0 – 0] | 218 [155 – 279] |
| 13-17 | 0 [0 – 0] | 3 [2 – 4] | 18 [12 – 24] | 32 [23 – 43] | 33 [23 – 45] | 9 [6 – 13] | 0 [-0 – 0] | 77 [55 – 102] |
| 18-49 | 2 [2 – 3] | 25 [19 – 33] | 146 [108 – 195] | 110 [81 – 147] | 373 [293 – 474] | 85 [64 – 112] | 0 [-0 – 0] | 619 [491 – 771] |
| 50-64 | 2 [2 – 3] | 26 [19 – 33] | 160 [116 – 205] | 124 [90 – 159] | 333 [254 – 417] | 187 [149 – 231] | 0 [-0 – 0] | 676 [542 – 837] |
| 65+ | 10 [8 – 12] | 129 [104 – 152] | 782 [619 – 918] | 594 [466 – 697] | 1,564 [1,304 – 1,782] | 481 [393 – 558] | 164 [149 – 179] | 3,068 [2,699 – 3,408] |
| **All** | **22 [20 – 23]** | **235 [193 – 263]** | **1,279 [1,024 – 1,450]** | **944 [749 – 1,081]** | **2,523 [2,139 – 2,807]** | **822 [687 – 924]** | **164 [149 – 179]** | **4,898 [4,427 – 5,296]** |

### 5% fewer doses — additional hospitalizations vs. baseline

| Counted in | 0 cut | 1-4 cut | 5-12 cut | 13-17 cut | 18-49 cut | 50-64 cut | 65+ cut | All cut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 [0 – 1] | 0 [0 – 0] | 1 [1 – 1] | 1 [0 – 1] | 1 [1 – 1] | 0 [0 – 0] | 0 [-0 – 0] | 4 [3 – 5] |
| 1-4 | 0 [0 – 0] | 4 [3 – 6] | 4 [2 – 5] | 2 [1 – 3] | 3 [2 – 4] | 1 [1 – 2] | 0 [-0 – 0] | 15 [11 – 19] |
| 5-12 | 0 [0 – 0] | 1 [1 – 1] | 7 [5 – 10] | 2 [2 – 3] | 3 [2 – 4] | 2 [1 – 2] | 0 [-0 – 0] | 16 [11 – 21] |
| 13-17 | 0 [0 – 0] | 0 [0 – 0] | 1 [1 – 2] | 2 [1 – 3] | 1 [1 – 2] | 1 [0 – 1] | 0 [-0 – 0] | 5 [4 – 7] |
| 18-49 | 0 [0 – 0] | 3 [2 – 4] | 10 [7 – 14] | 7 [5 – 9] | 14 [11 – 18] | 5 [4 – 7] | 0 [-0 – 0] | 39 [29 – 53] |
| 50-64 | 0 [0 – 0] | 3 [2 – 4] | 11 [8 – 14] | 8 [5 – 10] | 12 [9 – 15] | 12 [10 – 15] | 0 [-0 – 0] | 46 [34 – 59] |
| 65+ | 1 [1 – 2] | 14 [11 – 17] | 55 [42 – 66] | 37 [28 – 44] | 57 [45 – 67] | 30 [24 – 35] | 55 [50 – 59] | 251 [207 – 291] |
| **All** | **3 [3 – 3]** | **26 [21 – 29]** | **90 [70 – 104]** | **58 [45 – 68]** | **91 [73 – 105]** | **51 [42 – 58]** | **55 [50 – 59]** | **378 [311 – 430]** |

### 10% fewer doses — additional hospitalizations vs. baseline

| Counted in | 0 cut | 1-4 cut | 5-12 cut | 13-17 cut | 18-49 cut | 50-64 cut | 65+ cut | All cut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 [1 – 2] | 0 [0 – 1] | 2 [1 – 2] | 1 [1 – 1] | 2 [1 – 2] | 1 [1 – 1] | 0 [-0 – 0] | 7 [5 – 9] |
| 1-4 | 0 [0 – 0] | 9 [6 – 11] | 7 [5 – 9] | 4 [3 – 5] | 6 [4 – 8] | 3 [2 – 4] | 0 [-0 – 0] | 30 [22 – 38] |
| 5-12 | 0 [0 – 0] | 2 [1 – 3] | 15 [11 – 19] | 5 [3 – 6] | 7 [4 – 9] | 3 [2 – 4] | 0 [-0 – 0] | 32 [21 – 42] |
| 13-17 | 0 [0 – 0] | 1 [0 – 1] | 3 [2 – 3] | 4 [3 – 6] | 2 [2 – 3] | 1 [1 – 2] | 0 [-0 – 0] | 11 [8 – 15] |
| 18-49 | 1 [0 – 1] | 5 [4 – 7] | 20 [15 – 28] | 14 [10 – 18] | 28 [22 – 37] | 10 [8 – 14] | 0 [-0 – 0] | 79 [59 – 106] |
| 50-64 | 1 [0 – 1] | 6 [4 – 7] | 22 [16 – 29] | 15 [11 – 20] | 24 [17 – 31] | 24 [19 – 30] | 0 [-0 – 0] | 93 [68 – 120] |
| 65+ | 3 [2 – 3] | 28 [22 – 33] | 111 [85 – 132] | 74 [57 – 89] | 113 [90 – 135] | 60 [48 – 70] | 109 [99 – 119] | 510 [418 – 594] |
| **All** | **6 [5 – 6]** | **51 [42 – 58]** | **181 [140 – 210]** | **118 [91 – 137]** | **183 [147 – 212]** | **103 [85 – 117]** | **109 [99 – 119]** | **767 [627 – 875]** |

### 15% fewer doses — additional hospitalizations vs. baseline

| Counted in | 0 cut | 1-4 cut | 5-12 cut | 13-17 cut | 18-49 cut | 50-64 cut | 65+ cut | All cut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 [1 – 3] | 1 [1 – 1] | 3 [2 – 3] | 2 [1 – 2] | 3 [2 – 3] | 1 [1 – 1] | 0 [-0 – 0] | 11 [8 – 14] |
| 1-4 | 0 [0 – 0] | 13 [10 – 17] | 11 [8 – 14] | 6 [4 – 8] | 10 [7 – 12] | 4 [3 – 6] | 0 [-0 – 0] | 46 [33 – 58] |
| 5-12 | 0 [0 – 0] | 3 [2 – 4] | 23 [16 – 29] | 7 [5 – 9] | 10 [6 – 13] | 5 [3 – 6] | 0 [-0 – 0] | 49 [33 – 64] |
| 13-17 | 0 [0 – 0] | 1 [1 – 1] | 4 [3 – 5] | 6 [4 – 8] | 4 [2 – 5] | 2 [1 – 2] | 0 [-0 – 0] | 17 [11 – 23] |
| 18-49 | 1 [1 – 1] | 8 [6 – 11] | 31 [22 – 42] | 21 [15 – 28] | 42 [33 – 56] | 16 [12 – 21] | 0 [-0 – 0] | 120 [88 – 161] |
| 50-64 | 1 [1 – 1] | 8 [6 – 11] | 34 [24 – 44] | 23 [16 – 30] | 36 [26 – 46] | 36 [29 – 45] | 0 [-0 – 0] | 141 [103 – 182] |
| 65+ | 4 [3 – 5] | 42 [34 – 50] | 167 [128 – 200] | 112 [85 – 133] | 170 [134 – 203] | 90 [72 – 106] | 164 [148 – 178] | 776 [633 – 909] |
| **All** | **8 [8 – 9]** | **77 [63 – 87]** | **273 [211 – 317]** | **177 [137 – 206]** | **276 [221 – 319]** | **154 [127 – 175]** | **164 [148 – 178]** | **1,166 [949 – 1,333]** |

### 20% fewer doses — additional hospitalizations vs. baseline

| Counted in | 0 cut | 1-4 cut | 5-12 cut | 13-17 cut | 18-49 cut | 50-64 cut | 65+ cut | All cut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 [2 – 4] | 1 [1 – 1] | 3 [2 – 4] | 2 [2 – 3] | 4 [2 – 4] | 2 [1 – 2] | 0 [-0 – 0] | 15 [11 – 19] |
| 1-4 | 0 [0 – 0] | 17 [13 – 22] | 14 [10 – 18] | 8 [6 – 10] | 13 [9 – 16] | 6 [4 – 7] | 0 [-0 – 0] | 62 [45 – 78] |
| 5-12 | 0 [0 – 0] | 4 [3 – 5] | 31 [21 – 39] | 9 [6 – 13] | 13 [9 – 18] | 6 [4 – 8] | 0 [-0 – 0] | 66 [44 – 86] |
| 13-17 | 0 [0 – 0] | 1 [1 – 2] | 5 [3 – 7] | 9 [6 – 11] | 5 [3 – 6] | 2 [2 – 3] | 0 [-0 – 0] | 22 [15 – 30] |
| 18-49 | 1 [1 – 1] | 11 [8 – 14] | 41 [30 – 56] | 27 [20 – 37] | 57 [43 – 75] | 21 [16 – 28] | 0 [-0 – 0] | 161 [118 – 217] |
| 50-64 | 1 [1 – 1] | 11 [8 – 15] | 45 [32 – 59] | 31 [22 – 40] | 48 [35 – 62] | 48 [38 – 60] | 0 [-0 – 0] | 190 [138 – 246] |
| 65+ | 5 [4 – 6] | 56 [45 – 67] | 223 [170 – 268] | 149 [114 – 179] | 228 [179 – 271] | 120 [96 – 141] | 218 [198 – 237] | 1,049 [853 – 1,232] |
| **All** | **11 [10 – 12]** | **103 [84 – 116]** | **365 [283 – 426]** | **237 [183 – 276]** | **369 [295 – 426]** | **206 [170 – 234]** | **218 [198 – 237]** | **1,574 [1,276 – 1,809]** |

## Appendix B — Dose accounting: scheduled vs. delivered doses

| Age group | Population | Scheduled doses | Delivered doses | Wasted doses | Pct wasted | Scheduled coverage | Delivered coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 68,723 | 34,751 | 33,558 | 1,193 | 3.4% | 50.6% | 48.8% |
| 1-4 | 287,051 | 137,081 | 134,036 | 3,045 | 2.2% | 47.8% | 46.7% |
| 5-12 | 625,439 | 247,286 | 238,271 | 9,015 | 3.6% | 39.5% | 38.1% |
| 13-17 | 406,983 | 153,066 | 146,744 | 6,322 | 4.1% | 37.6% | 36.1% |
| 18-49 | 3,112,693 | 857,553 | 833,336 | 24,217 | 2.8% | 27.6% | 26.8% |
| 50-64 | 1,351,323 | 515,217 | 505,033 | 10,184 | 2.0% | 38.1% | 37.4% |
| 65+ | 1,420,722 | 864,407 | 857,022 | 7,385 | 0.9% | 60.8% | 60.3% |
| **All** | **7,272,934** | **2,809,361** | **2,748,000** | **61,361** | **2.2%** | **38.6%** | **37.8%** |

![Daily scheduled vaccination proportion](../figures/output/vax_coverage_daily.png)

*Daily scheduled vaccination proportion*

## Appendix C — Attack rate by vaccination status (baseline)

Share of each group infected over the season, median [95% interval]. Vaccinated = vaccinated at any point in the season (delivered doses), with their infections after vaccination; unvaccinated = everyone else, seeded infections included. The model never vaccinates someone already infected, so an infection before a person's vaccination date counts as unvaccinated. This timing effect lowers the vaccinated attack rate even where vaccination blocks no infection (65+).

| Age group | Share vaccinated | Vaccinated | Unvaccinated | Total |
|---|---:|---:|---:|---:|
| 0 | 49.0% | 4.1% [2.9–6.4%] | 14.4% [10.3–21.1%] | 9.3% [6.6–14.0%] |
| 1-4 | 46.8% | 6.2% [4.3–9.4%] | 15.0% [10.8–22.0%] | 10.9% [7.7–16.1%] |
| 5-12 | 38.2% | 8.9% [6.3–13.4%] | 20.8% [15.1–29.6%] | 16.3% [11.7–23.5%] |
| 13-17 | 36.2% | 9.1% [6.5–13.7%] | 21.6% [15.7–30.7%] | 17.1% [12.3–24.7%] |
| 18-49 | 27.0% | 11.1% [7.9–16.7%] | 17.3% [12.4–25.3%] | 15.6% [11.2–23.0%] |
| 50-64 | 37.6% | 10.0% [7.0–15.1%] | 15.4% [11.0–22.7%] | 13.4% [9.5–19.9%] |
| 65+ | 61.0% | 7.8% [5.5–11.9%] | 9.8% [7.0–14.7%] | 8.6% [6.1–13.0%] |
| **All** | **38.1%** | **9.3% [6.5–14.0%]** | **16.5% [11.8–24.1%]** | **13.7% [9.8–20.3%]** |
