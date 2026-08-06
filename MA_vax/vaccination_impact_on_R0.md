# Vaccination's effect on the reproduction number (next-generation-matrix analysis)

Quantifies how much vaccination (coverage + reduced susceptibility) reduces
the model's basic/effective reproduction number, using a next-generation
matrix (NGM) built from the age x vaccination-status structure of
`MA_vax/model.py`. Computed with `MA_vax/ngm_r0.py`.

**Fit used:** `fit_folder=MA_vax/outputs_2026-07-30_age_ihr_scale`,
`method=emcee`, `point=best` (the "best"/MAP point — see `methodology.md`
§3.4). This folder was originally named `outputs_2026-07-30_age_ihr_scale_no_V_infec`
and has since been renamed to `outputs_2026-07-30_age_ihr_scale` — same
underlying fit, confirmed by re-running `ngm_r0.py` against the new path and
reproducing the identical R0/Rv figures below.

---

## 1. Method: reducing the model to a next-generation matrix

New infections only enter via `S_to_E` / `SV_to_EV`, each bilinear in a
susceptible pool (`S` or `SV`) and the weighted infectious prevalence
(`I`, `IV`) reaching it through the age x age contact matrix. `E` and `EV`
have no exit besides progressing to `I` / `IV` with probability 1 (no death
or removal from the latent stage), so they can be integrated out
analytically. That leaves a **2A x 2A next-generation matrix** (A = 7 age
groups) directly over the two infectious classes `(I_1..I_A, IV_1..IV_A)`,
with average infectious duration `1/I_out_rate`:

```
M[I_i,  I_j ] = beta_adj * C[i,j] / (I_out_rate * pop_j) * relative_suscept_i * S0_i  * I_relative_infectiousness
M[I_i,  IV_j] = beta_adj * C[i,j] / (I_out_rate * pop_j) * relative_suscept_i * S0_i  * IV_relative_infectiousness
M[IV_i, I_j ] = beta_adj * C[i,j] / (I_out_rate * pop_j) * vax_susceptibility_i * SV0_i * I_relative_infectiousness
M[IV_i, IV_j] = beta_adj * C[i,j] / (I_out_rate * pop_j) * vax_susceptibility_i * SV0_i * IV_relative_infectiousness
```

R0 (or Rv, Reff) is the **spectral radius** (dominant eigenvalue magnitude)
of `M`, evaluated at a chosen disease-free state `(S0, SV0)`.

**Sanity check:** with no vaccination (`SV0=0`, `S0=population`), this
collapses exactly to the textbook single-type formula
`R0 = beta/I_out_rate * rho(C)` — verified numerically (both give 0.7033
under the calendar-averaged contact matrix, §2).

---

## 2. Static structural R0: calendar-averaged contact matrix, `beta_baseline` only

Uses `C_avg = total_C - (1-mean(is_school))*school_C - (1-mean(is_work))*work_C`
and `beta_baseline` alone (no humidity forcing, no fitted `m(t)` multiplier).
This isolates **only** the structural effect of vaccination coverage /
reduced susceptibility on transmission potential, decoupled from any
time-varying seasonal forcing — and that decoupling is exact: humidity and
`m(t)` are pure scalar multipliers on the whole NGM, so they scale R0 and Rv
by the same factor and **do not affect the relative reduction** reported
here.

| | R |
|---|---:|
| R0 (no vaccination) | **0.703** |
| Rv (with vaccination, full fitted-season coverage) | **0.590** |
| Reduction | 0.113 absolute, **16.1% relative** |

This "structural" R0 (0.70, subcritical on its own) is *not* the actual
epidemic-driving reproduction number — the real outbreak clears threshold
only because of humidity forcing and the fitted `m(t)` surge (§3). It isolates
just the vaccination-driven piece.

### Coverage ramp (0% -> full fitted coverage)

| Coverage (x fitted) | Rv | Reduction from R0 |
|---:|---:|---:|
| 0% | 0.703 | 0.0% |
| 25% | 0.674 | 4.2% |
| 50% | 0.645 | 8.2% |
| 75% | 0.617 | 12.2% |
| 100% | 0.590 | 16.1% |

### Per-age-group contribution (vaccinating only that group, at its own fitted coverage)

| Age | Coverage | Rv | Reduction |
|---|---:|---:|---:|
| 18-49 | 38.6% | 0.680 | 3.3% |
| 5-12 | 69.2% | 0.663 | 5.7% |
| 50-64 | 57.9% | 0.692 | 1.5% |
| 13-17 | 53.9% | 0.681 | 3.2% |
| 1-4 | 89.4% | 0.694 | 1.3% |
| 0 | 44.7% | 0.703 | 0.1% |
| 65+ | 71.0% | 0.703 | 0.0% |

18-49 and 5-12 dominate — the dominant eigenvector of the no-vax NGM puts
~45% of next-generation transmission on 18-49 (large population, high
contact rates) and ~15% on 5-12. 65+ contributes essentially nothing to R0
reduction because `vax_susceptibility=1.0` for that group in this fit (no
infection-blocking effect modeled, only the severity effect, which doesn't
enter R0). Per-group reductions don't sum to the 16.1% total — R0 reduction
is a nonlinear (spectral-radius) function of the coverage vector, not additive
across groups.

This 16.1% relative R0 reduction is much smaller than the ~62% reduction in
total-season infections (see the direct/indirect infection decomposition
discussed separately) — expected, since R0 is a linearized threshold/growth-rate
quantity while total attack size is highly nonlinear over a full epidemic
(small reductions in R compound across many generations of transmission).

---

## 3. Time-varying R0(t) / Reff(t): actual daily contact matrix + humidity + m(t)

Uses each day's actual `C(t)` (school/work calendar) and
`beta_adj(t) = beta_baseline * m(t) * (1 + humidity_impact * exp(-180*humidity(t)))`.
Two variants:

- **R0(t)**: susceptible/vaccinated split by *cumulative vaccination coverage
  only* as of day t (as if the epidemic were freshly seeded that day into an
  otherwise fully susceptible population) — isolates seasonal forcing +
  vaccine rollout from natural-infection depletion.
- **Reff(t)**: the actual simulated `S(t)`/`SV(t)` (depleted by both
  vaccination *and* ongoing infection) — the true effective reproduction
  number for that scenario.

Both are computed for the baseline (with-vaccination) and no-vaccination
scenarios, using the same day's `C(t)`/`beta_adj(t)` in both.

### R0(t): with vs. without vaccination (coverage-only DFE)

| Date | m(t) | R0 no-vax | R0 with-vax | Reduction |
|---|---:|---:|---:|---:|
| 2025-09-01 | 1.00 | 0.48 | 0.48 | 0.0% |
| 2025-10-01 | 2.04 | 2.01 | 1.99 | 0.9% |
| 2025-11-01 | 1.06 | 0.56 | 0.54 | 3.3% |
| **2025-12-01** | **2.74** | **2.97** | **2.61** | **11.9%** |
| 2026-01-01 | 1.35 | 1.03 | 0.94 | 9.0% |
| 2026-02-01 | 0.94 | 0.56 | 0.49 | 11.4% |
| 2026-03-01 | 1.53 | 0.87 | 0.76 | 12.0% |
| 2026-04-01 | 1.53 | 1.47 | 1.20 | 18.2% |

Reduction from vaccination climbs steadily over the season as cumulative
coverage builds (0% -> 18%) — the "clean" comparison, reflecting only
coverage-to-date, not accumulated natural immunity.

### Reff(t): with vs. without vaccination (actual simulated trajectories)

| Date | m(t) | Reff no-vax | Reff with-vax | Reduction |
|---|---:|---:|---:|---:|
| 2025-09-01 | 1.00 | 0.48 | 0.48 | 0.0% |
| 2025-10-01 | 2.04 | 2.01 | 1.99 | 0.8% |
| 2025-11-01 | 1.06 | 0.56 | 0.54 | 3.2% |
| **2025-12-01** | **2.74** | **2.94** | **2.60** | **11.6%** |
| 2026-01-01 | 1.35 | 0.87 | 0.88 | -1.4% |
| 2026-02-01 | 0.94 | 0.42 | 0.45 | -6.5% |
| 2026-03-01 | 1.53 | 0.64 | 0.69 | -6.8% |
| 2026-04-01 | 1.53 | 1.04 | 1.07 | -3.0% |

### At the epidemic's own internal peaks

Peak new-hospitalization / peak prevalence (I+IV) day: **2025-12-28**.
Peak new-infection incidence day: **2025-12-19** (leads the prevalence/
hospitalization peak, as expected from the E/I progression delay).

| | R0 no-vax | R0 with-vax | Reduction | Reff no-vax | Reff with-vax | Reduction |
|---|---:|---:|---:|---:|---:|---:|
| Peak new_H / prevalence (Dec 28) | 0.84 | 0.75 | 10.2% | 0.73 | 0.72 | 1.7% |
| Peak incidence (Dec 19) | 1.85 | 1.58 | **14.6%** | 1.68 | 1.53 | 9.2% |

Season max: R0(t) = 2.61 and Reff(t) = 2.60, both on 2025-12-01, matching the
fitted `m(t)` peak in `methodology.md` §4.1.

### Caveats

- **R0(t) and Reff(t) track almost identically through December** (natural
  infection depletion is still small relative to the population), then
  **diverge from January onward** as accumulated natural immunity builds on
  top of vaccination.
- **The negative "reduction" values from January onward are not vaccination
  increasing transmission risk.** The no-vaccination epidemic burns through
  susceptibles faster and bigger through November-December (nothing holding
  it back), so by January it has accumulated *more* natural immunity than the
  smaller vaccinated-world epidemic. At that later calendar date, the no-vax
  world's Reff is then lower purely because it already "used up" more of its
  susceptible pool earlier — a timing/ordering artifact of comparing two
  differently-paced epidemic trajectories at the same calendar date, not a
  causal effect of vaccination. The R0(t) comparison (coverage-only DFE, no
  natural-immunity confound) avoids this and is the cleaner series to use for
  "how much is vaccination itself doing" at any given date.
- **The incidence peak does not coincide with a single Reff=1 crossing**,
  unlike the classic autonomous-SIR intuition. That intuition assumes Reff
  declines monotonically from susceptible depletion alone. Here Reff is
  driven up and down by exogenous seasonal forcing (school calendar, `m(t)`
  climbing to 2.74 then falling) on a timescale comparable to the epidemic's
  own generation time — Reff already dips below 1 once in mid-November before
  the Dec-1 forcing surge pushes it back above 1, and only settles below 1 for
  good in late December. The actual peak reflects both susceptible depletion
  and the turnover of the `m(t)` forcing itself.
- As in §2, all of this is conditional on the fitted "best" point estimate
  for `beta_baseline`, `m(t)`, `ihr_scale`, etc. — see `methodology.md` §3.4
  for the mean-vs-best caveat.
