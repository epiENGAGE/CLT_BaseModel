# `ts_per_day` impact analysis — generic-core MA_vax fitting

## Background

When fitting the Massachusetts flu+vaccination model (`model_config.json` /
`MA_fit_config.json`) via `generic_core`'s MCMC path (`generic_core/fitting.py`,
`method="mcmc"`), the resulting fits were consistently worse than — and had
systematically different fitted parameters from — a reference calibration of
the "same" model done with a standalone, pure-numpy prototype
(`MA_vax_standalone/fit_bayesian.py` + `MA_vax_standalone/model.py`). This document records the
investigation into why, and what it means for choosing `run_kwargs.ts_per_day`
going forward.

**Conclusion up front:** the discrepancy was not a fitting-config problem
(bounds, priors, likelihood) — those were checked and ruled out. It's a
genuine difference in how the two codebases convert a per-day rate into a
per-timestep transition count. `generic_core` (via `clt_toolkit`) uses the
exact hazard formula `1 - exp(-rate * dt)`; `MA_vax_standalone/model.py` uses the linear
approximation `rate * dt`. These only agree when `rate * dt` is small, and at
coarse `ts_per_day` they diverge enough to produce a >400% difference in
epidemic size for identical parameters. Since the goal going forward is an
**independent calibration within `generic_core`**, not reproducing
`MA_vax`'s specific numbers, the resolution is to keep the (more principled)
hazard formula and run at a `ts_per_day` fine enough that the bias becomes
negligible, rather than changing the formula.

---

## 1. How the discrepancy was found

### 1.1 Ruled out first: fitting-config mismatches

Two config issues were found and fixed along the way, before the deeper
numerical issue was identified:

- **`humidity_impact` prior bounds.** The reference fit (`MA_vax_standalone/fit_bayesian.py`)
  used bounds `[0.0, 1.0]` for `humidity_impact` and found a best-fit value of
  ~0.41. `MA_fit_config.json` initially bounded it to `[0.05, 0.2]` — far
  narrower, and excluding the reference's best-fit value entirely. Widening
  the bound removed one confound but did not close the fit-quality gap.
- **MCMC burn-in/thin.** Unlike `fit_bayesian.py`, which computes burn-in and
  thinning automatically from the chain's autocorrelation time, `generic_core`'s
  MCMC path uses fixed `mcmc_burnin`/`mcmc_thin` values from the config. This
  was checked against the saved `loss_curve` (mean log-posterior per step) and
  found to have already plateaued well before the configured burn-in — so this
  was not the cause of the fit-quality gap either.

### 1.2 The decisive test: apples-to-apples forward simulation

To isolate genuine model-mechanics differences from anything caused by MCMC
posterior degeneracy or config mismatches, a direct comparison script
(`compare_to_generic.py`, in this folder) was built: take one fixed,
realistic parameter set (beta, humidity_impact, seed scale, per-age IHR
scale, and a fitted `m(t)` transmission-multiplier curve), and simulate it
two ways:

1. `MA_vax_standalone/model.py`'s `simulate_new_H` — pure numpy, one Euler step per day
   (`dt=1`, i.e. effectively `ts_per_day=1`).
2. `generic_core`'s `ConfigDrivenMetapopModel`, built from
   `model_config.json`, run at `ts_per_day=1` (matched to `model.py`, so
   timestep resolution could not be the explanation).

**Result, with the identical parameter set and identical timestep on both
sides:**

```
peak ref (model.py):        153.56  (day 118)
peak generic-core:          997.22  (day 118)   — 6.5x higher
total ref (model.py):     6,098.2
total generic-core:      31,909.4                — +423%
```

Same beta, same humidity_impact, same seeds, same IHR, same m(t) curve, same
contact matrices, same timestep — and `generic_core` produced over 5x more
hospitalizations, growing explosively from ~day 40 onward, peaking far
higher, then crashing below the reference in the tail (having burned through
susceptibles much faster). This ruled out `ts_per_day` resolution as the
explanation (both were run at the same resolution) and pointed to something
structural in how each engine turns a rate into a transition count.

---

## 2. Root cause: two different rate → probability/count conversions

### 2.1 The two formulas

Both codebases start from the same per-day rates (`E_to_I_rate=0.5`,
`EV_to_IV_rate=0.5`, `I_out_rate=0.333`, `H_out_rate=0.17`, plus the
force-of-infection rate), and both convert `rate` + step size `dt` into
"what fraction of this compartment leaves in one step" — but via different
formulas:

- **`MA_vax_standalone/model.py` (linear / forward-Euler):** `p = rate · dt`
  A first-order Taylor approximation. Used directly, with no probability
  transform, e.g. `E_to_I_flow = E_to_I_rate * dt * E`.

- **`generic_core` / `clt_toolkit` (exponential / hazard):**
  `p = 1 - exp(-rate · dt)`
  (`clt_toolkit/base_components.py:52`, `approx_binom_probability_from_rate`).
  This is the *exact* probability that a Poisson process with constant rate
  produces at least one event within an interval of length `dt`. It is used
  for **both** the deterministic (`BINOM_DETERMINISTIC_NO_ROUND`, used by the
  MCMC/ABC fitting path) and stochastic (`BINOM`) transition types — the
  deterministic mode uses it so that its output equals the expectation of the
  stochastic mode.

### 2.2 Why they diverge, and by how much

Taylor-expanding: `1 - exp(-r) = r - r²/2 + r³/6 - …`. Since this is always
less than `r` for `r > 0`, **the hazard formula always removes fewer
individuals per step than the linear formula**, for the same nominal rate.
Concretely, at `dt = 1` (`ts_per_day = 1`), using this model's actual rates:

| rate | linear (`r·dt`) | hazard (`1−e^(−r·dt)`) | hazard shortfall | ⇒ effective sojourn time is longer by |
|---|---|---|---|---|
| `E_to_I_rate = 0.5` | 0.500 | 0.393 | 21.3% | 1.27x |
| `EV_to_IV_rate = 0.5` | 0.500 | 0.393 | 21.3% | 1.27x |
| `I_out_rate = 0.333` | 0.333 | 0.283 | 14.9% | 1.18x |
| `H_out_rate = 0.17` | 0.170 | 0.156 | 8.0% | 1.09x |

A ~20% per-day shortfall in exit probability from `E`/`EV`/`I`/`IV` means
people linger in each infectious/exposed stage longer than intended. Since
transmission accumulates roughly as β × (time spent infectious), this
directly inflates the effective reproduction number. And because the bias
applies at *every* timestep, to *every* compartment along the chain, in a
system with strong positive feedback (more people staying infectious longer →
more new infections → who also linger longer …), it compounds into a large,
*growing* discrepancy over the course of a simulation rather than a fixed
percentage offset — this is exactly what was observed in the direct
comparison (§1.2): +26% at day 10, growing to +580% by day 90–100, before the
generic-core run's faster susceptible depletion causes it to fall back below
the reference in the tail.

### 2.3 Neither formula is simply "wrong" — a design/interpretation question

This point matters for deciding what to do about it, so it's worth being
precise:

- The **hazard formula is exact**, but only relative to the assumption (shared
  by *both* formulas) that the rate is frozen at its start-of-step value for
  the duration of `dt`. In reality the rate is not constant — force of
  infection depends on `S` and `I`, which change continuously within the
  step. So "exact" here means exact for an idealized frozen-rate sub-problem,
  not exact for the true nonlinear system.
- Both formulas converge to the **same true continuous-time solution as
  `dt → 0`** (to first order they're identical for small `rate·dt`), but they
  approach it **from opposite directions**: the linear/Euler scheme
  over-empties compartments relative to the true exponential decay (shorter
  effective sojourn than intended); the hazard scheme is exact for the frozen
  sub-problem, so it doesn't share that particular bias.
- The hazard formula is the natural/necessary choice for a toolkit that wants
  **deterministic and stochastic modes to be consistent** (deterministic
  output = expectation of the stochastic Binomial draw), since a Binomial
  probability must stay in `[0, 1]` — which the linear formula does not
  guarantee for large `rate · dt`.
- `MA_vax_standalone/model.py`'s linear scheme isn't a bug either — it's simply *what the
  model is defined to be*: the deterministic Euler map at `dt=1`, exactly
  what `fit_bayesian.py` was calibrated against.

**Decision made:** since the goal going forward is an independent, from-scratch
calibration within `generic_core` (MA_vax was only ever a fast prototyping
ground for the MCMC/ABC fitting methods before porting them to
`generic_core`'s full architecture) — not reproduction of `MA_vax`'s specific
fitted numbers — the more standard, self-consistent hazard formula is kept
as-is (no `clt_toolkit` changes), and instead **`ts_per_day` is resolved
finely enough that the bias becomes negligible**, converging `generic_core`
itself to the true continuous-time solution.

---

## 3. `ts_per_day` convergence study

### 3.1 Method

`ts_per_day_convergence.py` (this folder) runs one fixed, realistic parameter
set (pulled from a previous fit's `best_params`, including its fitted `m(t)`
curve) forward through the `generic_core` engine at a ladder of `ts_per_day`
values, with everything else held fixed, and compares each run's daily
`new_H` trajectory (summed over all 7 age groups) against the **finest
resolution tested**, using three metrics:

- **`peak %diff`**: `100 × (max(total_run) − max(ref)) / max(ref)` — the two
  runs' peak *heights*, independent of what day each peak falls on.
- **`total %diff`**: `100 × (sum(total_run) − sum(ref)) / sum(ref)` — the
  full-simulation cumulative new_H, i.e. overall attack-rate-equivalent
  discrepancy.
- **`max day %diff`**: for every day where the reference exceeds 1.0 (to
  avoid dividing by ~0), `100 × |total_run[t] − ref[t]| / ref[t]`, then the
  **max over all days** — the worst single-day pointwise mismatch anywhere in
  the trajectory. This is the strictest metric: two curves can agree closely
  on peak and total while still being offset in timing, which this metric
  catches and the other two don't.

Wall-clock time per single forward simulation (no MCMC, no parallelism) was
also recorded, to translate resolution into expected MCMC cost (MCMC cost
scales with `ts_per_day` roughly in proportion to per-eval simulation cost).

### 3.2 Results

Parameter set used: `beta_baseline=0.0296`, `humidity_impact=0.596`,
`seed_scale_E=2.240`, `ihr_scale=[1.647, 1.51, 1.144, 0.765, 0.477, 0.616,
0.756]` (from `MA_fitted_params.json`'s `best_params`, 251-day simulation).

| `ts_per_day` | peak | total | wall time (single run) |
|---|---|---|---|
| 7 | 120.91 (day 118) | 5,720.4 | 0.13–0.16s |
| 14 | 100.55 (day 118) | 4,749.9 | 0.23–0.24s |
| 28 | 91.40 (day 118) | 4,309.9 | 0.43–0.44s |
| 56 | 87.08 (day 118) | 4,101.1 | 0.85s |
| 112 | 84.97 (day 118) | 3,999.5 | 1.68s |
| 224 (finest) | 83.94 (day 118) | 3,949.4 | 3.23s |

Relative to the finest resolution tested (`ts_per_day=224`):

| `ts_per_day` | peak %diff | total %diff | max day %diff | cost vs. `ts=7` |
|---|---|---|---|---|
| 7 | 44.05% | 44.84% | 55.13% | 1.00x |
| 14 | 19.79% | 20.27% | 25.04% | 1.76x |
| 28 | 8.90% | 9.13% | 11.28% | 3.24x |
| 56 | 3.74% | 3.84% | 4.74% | 6.28x |
| 112 | 1.23% | 1.27% | 1.57% | 12.50x |

**Convergence is close to first-order**: the error roughly halves each time
`ts_per_day` doubles (44% → 20% → 9% → 3.7% → 1.2%), consistent with the
`O(dt)` nature of the hazard-vs-linear bias derived in §2.2. Cost scales a
little sub-linearly with `ts_per_day` (e.g. `ts=56` costs 6.3x, not 8x, the
`ts=7` cost).

### 3.3 Recommendation

No tested `ts_per_day` below 224 came within a strict 1% tolerance of the
finest run — but a 1% numerical-convergence bar is far stricter than
necessary here. The actual problem being solved was a **>400%** compounding
bias that drove the MCMC into wild, degenerate parameter trade-offs (beta,
humidity_impact, seed_scale_E, and IHR_scale all inflating together across
different `ts_per_day` settings). Once residual bias is down to a few
percent, an MCMC refit should simply absorb it into slightly adjusted
parameter values, the way any minor model-form discrepancy normally gets
absorbed — not the large-scale mode-shifting behavior seen at coarser
resolution.

**Recommended starting point: `ts_per_day=28`** (~9% residual bias, 3.2x the
current per-eval cost). Refit there and check whether the fitted parameters
are now stable across reruns (no more large swings in beta/humidity/IHR). If
still unstable, step up to `ts_per_day=56` (~3.7% bias, 6.3x cost) before
going any higher — the cost of `ts_per_day=112`+ combined with more MCMC
walkers (see below) becomes substantial (potentially many hours per run).

**Cost interaction with walker count:** the existing MCMC run (`ts_per_day=7`,
`n_walkers=40`) takes ~30 minutes. `n_walkers` was sitting at emcee's bare
minimum (`2×ndim`) for a ~20-dimensional posterior, which mixes poorly given
the parameter-degeneracy ridge observed throughout this investigation — more
walkers (e.g. 2x) is a separate, worthwhile improvement, but multiplies
directly with the `ts_per_day` cost multiplier above. E.g. `ts_per_day=28` +
2x walkers ≈ 6.5x ≈ 3.3 hours; `ts_per_day=56` + 2x walkers ≈ 12.6x ≈ 6.3
hours. Worth testing parameter stability at `ts_per_day=28` with the
*current* walker count first before paying for both increases at once.

---

## 4. Does the discrepancy distort the epidemic curve's *shape*, or just its *scale*?

A natural follow-up: is the `ts_per_day=7` bias just an overall magnitude
error (fixable by refitting `beta_baseline` etc.), or does it also distort
the *shape* of the epidemic curve (peak sharpness/width) in a way a scalar
parameter can't correct?

### 4.1 Method

Using the same fixed parameter set as §3, `beta_baseline` at `ts_per_day=7`
was adjusted (via bisection) until its peak height matched the
`ts_per_day=224` run's peak (83.94, on day 118) exactly, with every other
parameter held fixed. The two peak-matched trajectories were then compared
on total cumulative burden and on curve width at several fractions of peak
height.

### 4.2 Results

**Beta adjustment required:** `0.02958 → 0.02911` — only a **~1.6% reduction**
in `beta_baseline` was enough to close what had been a 44% *unadjusted* peak
gap. (This reflects how sensitive epidemic size is to beta near this
operating point, via the nonlinear S–I feedback — small rate changes are
strongly amplified.)

**Total cumulative, once peaks are matched:**

| | cumulative new_H |
|---|---|
| `ts_per_day=224` | 3,949.4 |
| `ts_per_day=7` (beta-adjusted) | 4,079.5 (**+3.3%**) |

**Curve width, at several fractions of peak height:**

| threshold | `ts=224` | `ts=7` (matched) | difference |
|---|---|---|---|
| 50% of peak | 37 days | 38 days | +2.7% |
| 25% of peak | 56 days | 58 days | +3.6% |
| 10% of peak | 84 days | 86 days | +2.4% |

### 4.3 Interpretation

Once magnitude (peak height) is aligned via a small beta adjustment, the two
curves are close in both total burden (+3.3%) and shape. If anything,
`ts_per_day=7` is very slightly **wider/flatter**, not narrower, than
`ts_per_day=224` — by 2–4% at every threshold tested, not a qualitative
shape distortion.

**Practical implication:** most of what the `ts_per_day=7` MCMC fits were
compensating for was overall *scale*, not curve *shape* — consistent with
the observation (throughout the broader investigation) that `beta_baseline`,
`humidity_impact`, `seed_scale_E`, and `IHR_scale` all moved together across
different `ts_per_day` settings, rather than the `m(t)` time-varying-
transmission knots reshaping the curve to compensate. This is a reassuring
result: it means `ts_per_day` mainly needs to be fine enough to get
*magnitude* right; it is not hiding a shape-distortion problem that a scalar
parameter can't fix, which increases confidence that a refit at a moderate
`ts_per_day` (e.g. 28–56) will land on a sensible, stable calibration.

---

## 5. Artifacts produced

All in `generic_core/examples/MA_vax/analysis/` unless noted:

- **`MA_vax_standalone/compare_to_generic.py`** — apples-to-apples forward-simulation
  comparison between `MA_vax_standalone/model.py` and `generic_core`, for a given fitted
  parameter set (§1.2). Also writes `MA_vax_standalone/compare_to_generic_trajectories.csv`
  (full day-by-day, age-by-age trajectories for both engines).
- **`ts_per_day_convergence.py`** — the convergence study in §3; runs a fixed
  parameter set at a ladder of `ts_per_day` values and reports peak/total/
  max-day differences relative to the finest resolution tested. Writes
  `ts_per_day_convergence.csv` (day-by-day totals at each tested `ts_per_day`).
- **`peak_matched_ts7_vs_ts224.csv`** — the peak-matched shape comparison in
  §4 (day-by-day totals for `ts_per_day=224` at its original beta, and
  `ts_per_day=7` at the beta adjusted to match its peak). *Deleted during the
  massachusetts_vax consolidation; regenerate from §4 if needed.*

## 6. Open items / follow-ups

- Rerun the MCMC fit at `ts_per_day=28` (or `56` if needed) and confirm
  fitted-parameter stability across reruns before scaling up further.
- Separately verify the walker-count increase (`n_walkers`) improves posterior
  stability, ideally independent of the `ts_per_day` change so the two
  effects aren't conflated.
- **Stochastic-mode note for later CI work:** the hazard formula is already
  the correct/required choice for genuine stochastic (`Binomial`) draws — no
  formula change needed there. However, resolution needs may differ from the
  deterministic fit: coarse `ts_per_day` can distort the *variance* of
  stochastic trajectories and early-outbreak fadeout probability (small
  E-seed counts of 2–85 people per age group) even at a `ts_per_day` that's
  fine enough for the deterministic *mean* trajectory to converge. Recommend
  a separate check — e.g. comparing the spread/fadeout rate across ~200
  stochastic replicates at two candidate `ts_per_day` values — before
  finalizing the resolution used for confidence-interval analysis. Also avoid
  `TransitionTypes.BINOM_TAYLOR_APPROX` (the linear-probability stochastic
  variant) for this purpose, since it lacks the hazard formula's `[0,1]`
  guarantee and exactness.
