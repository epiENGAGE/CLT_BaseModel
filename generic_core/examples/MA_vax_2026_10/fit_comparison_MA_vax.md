# MA_vax: comparing the four fits

Massachusetts 2025–26 influenza season. Four calibrations of the same model against the same hospital-admission data:

- **7-age** vs **1-age**: the age-structured model (7 groups) vs the single-group model built by population-weighting its inputs.
- **m(t)** vs **no m(t)**: with and without the fitted time-varying transmission multiplier (18 log-increments on 14-day knots).

All intervals are the median and 2.5–97.5 percentile across each fit's posterior draws (one simulation per accepted parameter set, `baseline` vaccination scenario).


## 1. Headline comparison

| Quantity | 7-age, m(t) | 7-age, no m(t) | 1-age, m(t) | 1-age, no m(t) |
| --- | --- | --- | --- | --- |
| Posterior draws | 627 | 429 | 495 | 352 |
| **Infection attack rate** | 6.7% [5.0% – 10.4%] | 36.9% [32.9% – 40.8%] | 6.4% [2.3% – 34.6%] | 38.5% [32.5% – 44.2%] |
| Cumulative hospitalizations (simulated) | 6,070 [5,569 – 6,549] | 5,622 [4,943 – 6,389] | 6,060 [5,110 – 7,075] | 6,225 [5,359 – 7,190] |
| Cumulative hospitalizations (observed) | 6,157 | 6,157 | 6,134 | 6,134 |
| Difference vs observed | -1.4% | -8.7% | -1.2% | +1.5% |
| **Infections per hospitalization** | 78 [58 – 121] | 458 [394 – 532] | 74 [28 – 402] | 434 [359 – 499] |
| IHR, population-weighted | 2.098% [1.355% – 2.799%] | 0.331% [0.279% – 0.393%] | 1.438% [0.266% – 3.865%] | 0.246% [0.215% – 0.297%] |
| IHR, infection-weighted | 1.440% [0.931% – 1.930%] | 0.244% [0.210% – 0.285%] | 1.438% [0.266% – 3.865%] | 0.246% [0.215% – 0.297%] |
| IHR_scale, population-weighted | 0.83 [0.53 – 1.13] | 0.15 [0.13 – 0.17] | 0.68 [0.13 – 1.82] | 0.12 [0.10 – 0.14] |
| Peak daily hospitalizations (simulated) | 165 [142 – 190] | 81 [67 – 97] | 135 [96 – 189] | 81 [60 – 106] |
| Peak daily hospitalizations (observed) | 183 | 183 | 183 | 183 |
| `beta_baseline` | 0.0303 [0.0251 – 0.0374] | 0.0368 [0.0312 – 0.0411] | 0.0324 [0.0235 – 0.0511] | 0.0403 [0.0342 – 0.0444] |
| `humidity_impact` | 0.60 [0.14 – 0.95] | 0.21 [0.01 – 0.54] | 0.35 [0.02 – 0.89] | 0.12 [0.01 – 0.48] |
| `seed_scale_E` | 1.57 [0.37 – 3.72] | 2.03 [0.65 – 7.34] | 1.02 [0.15 – 7.87] | 2.23 [0.64 – 8.36] |
| `phi` (NB dispersion) | 55 [15 – 306] | 3 [2 – 5] | 10 [2 – 39] | 2 [1 – 4] |
| Peak R_eff | 2.09 | 1.71 | 1.97 | 1.59 |
| Peak R_eff date | 2025-12-08 | 2025-10-09 | 2025-12-08 | 2025-10-09 |
| R_eff, season end | 0.45 | 0.43 | 0.54 | 0.45 |
| R_v, season end (no infection-acquired immunity) | 0.48 | 0.69 | 0.60 | 0.74 |
| **R_eff / R_v, season end** (share of R left after depletion) | 0.93 | 0.62 | 0.90 | 0.61 |
| Season-mean effective beta (per contact per day) | 0.0317 | 0.0392 | 0.0359 | 0.0415 |


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


## 2. Infection attack rate

Ever-infected fraction at the end of the season, `(N − S − SV) / N`.

| Age group | 7-age, m(t) | 7-age, no m(t) | 1-age, m(t) | 1-age, no m(t) |
| --- | --- | --- | --- | --- |
| 0-0 | 4.8% [3.6% – 7.4%] | 26.9% [23.7% – 30.0%] | — | — |
| 1-4 | 5.3% [3.9% – 8.1%] | 30.0% [26.7% – 33.4%] | — | — |
| 5-12 | 8.1% [6.0% – 12.3%] | 44.3% [40.1% – 48.3%] | — | — |
| 13-17 | 8.3% [6.2% – 12.6%] | 45.1% [40.8% – 49.1%] | — | — |
| 18-49 | 7.6% [5.7% – 11.7%] | 41.0% [36.6% – 45.2%] | — | — |
| 50-64 | 6.5% [4.9% – 10.1%] | 35.7% [31.7% – 39.7%] | — | — |
| 65+ | 4.1% [3.0% – 6.4%] | 24.0% [21.0% – 27.1%] | — | — |
| **All** | **6.7% [5.0% – 10.4%]** | **36.9% [32.9% – 40.8%]** | **6.4% [2.3% – 34.6%]** | **38.5% [32.5% – 44.2%]** |


## 3. Hospitalizations: fitted vs observed

Cumulative new hospital admissions over the season (`I_to_H + IV_to_H`), against the calibration target.

| Age group | 7-age, m(t) (simulated) | 7-age, no m(t) (simulated) | 1-age, m(t) (simulated) | 1-age, no m(t) (simulated) | Observed |
| --- | --- | --- | --- | --- | --- |
| 0-0 | 36 [22 – 48] | 42 [27 – 58] | — | — | 41 |
| 1-4 | 153 [104 – 201] | 167 [120 – 217] | — | — | 165 |
| 5-12 | 156 [111 – 204] | 162 [112 – 208] | — | — | 161 |
| 13-17 | 95 [63 – 123] | 93 [64 – 125] | — | — | 90 |
| 18-49 | 746 [582 – 895] | 771 [653 – 933] | — | — | 727 |
| 50-64 | 753 [595 – 911] | 740 [589 – 916] | — | — | 750 |
| 65+ | 4,125 [3,700 – 4,555] | 3,630 [3,076 – 4,335] | — | — | 4,224 |
| **All** | **6,070 [5,569 – 6,549]** | **5,622 [4,943 – 6,389]** | **6,060 [5,110 – 7,075]** | **6,225 [5,359 – 7,190]** | **6,157** |
| *% difference* | *-1.4%* | *-8.7%* | *-1.2%* | *+1.5%* |  |


**Difference from observed, by age group**

| Age group | 7-age, m(t) | 7-age, no m(t) |
| --- | --- | --- |
| 0-0 | -13.0% | +1.5% |
| 1-4 | -6.8% | +1.7% |
| 5-12 | -2.6% | +0.6% |
| 13-17 | +5.3% | +2.4% |
| 18-49 | +2.7% | +6.1% |
| 50-64 | +0.4% | -1.4% |
| 65+ | -2.3% | -14.1% |
| **All** | **-1.4%** | **-8.7%** |


The single-age fits are calibrated to the population total only (2 targets), the 7-age fits to the 7 age series plus an end-of-season by-age scalar (8 targets), so the by-age columns are a fit check for the former and a fit result for the latter.


## 4. IHR scaling

`IHR_scale` multiplies both hospitalization-risk parameters (`I_to_H_prop` and `IV_to_H_prop`) for that age group; the fitted IHR is that multiplier times the config's baseline value.


**Fitted `IHR_scale`**

| Age group | 7-age, m(t) | 7-age, no m(t) | 1-age, m(t) | 1-age, no m(t) | ratio m(t) / no m(t) |
| --- | --- | --- | --- | --- | --- |
| 0-0 | 1.54 [0.85 – 1.96] | 0.32 [0.20 – 0.45] | — | — | 4.8× |
| 1-4 | 1.56 [0.86 – 1.97] | 0.29 [0.21 – 0.39] | — | — | 5.4× |
| 5-12 | 1.18 [0.69 – 1.73] | 0.22 [0.15 – 0.30] | — | — | 5.3× |
| 13-17 | 1.01 [0.55 – 1.62] | 0.19 [0.13 – 0.25] | — | — | 5.4× |
| 18-49 | 0.59 [0.36 – 0.85] | 0.11 [0.10 – 0.14] | — | — | 5.2× |
| 50-64 | 0.78 [0.46 – 1.15] | 0.14 [0.11 – 0.18] | — | — | 5.5× |
| 65+ | 1.07 [0.69 – 1.42] | 0.16 [0.13 – 0.19] | — | — | 6.7× |
| **All (pop-weighted)** | **0.83 [0.53 – 1.13]** | **0.15 [0.13 – 0.17]** | **0.68 [0.13 – 1.82]** | **0.12 [0.10 – 0.14]** | **5.5× / 5.8×** |


**Fitted IHR (unvaccinated arm, `I_to_H_prop` after scaling)**

| Age group | Config baseline | 7-age, m(t) | 7-age, no m(t) | 1-age, m(t) | 1-age, no m(t) |
| --- | --- | --- | --- | --- | --- |
| 0-0 | 0.697% | 1.077% [0.594% – 1.366%] | 0.226% [0.142% – 0.312%] | — | — |
| 1-4 | 0.697% | 1.086% [0.599% – 1.375%] | 0.201% [0.147% – 0.269%] | — | — |
| 5-12 | 0.274% | 0.322% [0.188% – 0.473%] | 0.061% [0.042% – 0.081%] | — | — |
| 13-17 | 0.274% | 0.278% [0.151% – 0.444%] | 0.051% [0.036% – 0.069%] | — | — |
| 18-49 | 0.561% | 0.331% [0.201% – 0.478%] | 0.063% [0.057% – 0.079%] | — | — |
| 50-64 | 1.060% | 0.823% [0.489% – 1.217%] | 0.149% [0.116% – 0.191%] | — | — |
| 65+ | 9.091% | 9.712% [6.316% – 12.951%] | 1.459% [1.197% – 1.766%] | — | — |
| **All (pop-weighted)** | **2.118%** | **2.098% [1.355% – 2.799%]** | **0.331% [0.279% – 0.393%]** | **1.438% [0.266% – 3.865%]** | **0.246% [0.215% – 0.297%]** |


**Fitted IHR (vaccinated arm, `IV_to_H_prop` after scaling)**

| Age group | 7-age, m(t) | 7-age, no m(t) | 1-age, m(t) | 1-age, no m(t) |
| --- | --- | --- | --- | --- |
| 0-0 | 0.982% [0.542% – 1.246%] | 0.206% [0.130% – 0.285%] | — | — |
| 1-4 | 0.990% [0.547% – 1.254%] | 0.183% [0.134% – 0.246%] | — | — |
| 5-12 | 0.294% [0.172% – 0.432%] | 0.056% [0.039% – 0.074%] | — | — |
| 13-17 | 0.254% [0.138% – 0.405%] | 0.047% [0.032% – 0.063%] | — | — |
| 18-49 | 0.301% [0.183% – 0.435%] | 0.058% [0.052% – 0.072%] | — | — |
| 50-64 | 0.750% [0.446% – 1.109%] | 0.136% [0.105% – 0.174%] | — | — |
| 65+ | 6.701% [4.358% – 8.936%] | 1.007% [0.826% – 1.219%] | — | — |


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
