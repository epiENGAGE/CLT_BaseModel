# Tasks: Vaccination Scenario Analysis Readiness

This file tracks concrete implementation tasks for items 5, 6, and 7 from
`ARCHITECTURE.md § What Is Missing Before Running Vaccination Scenario Analyses`.

---

## Item 5 — Scenario Configuration Infrastructure

**Goal:** A reusable layer that takes a baseline model configuration and one or more
counterfactual vaccine schedules, runs each as a multi-replicate `Experiment`, and stores
results in a way that supports downstream comparison.

**What already exists:**
- `Experiment.run_static_inputs()` runs N replicates of a single fixed model
- `updated_dataclass()` creates modified copies of frozen `FluSubpopParams`
- `MetapopModel.modify_subpop_params()` / `modify_simulation_settings()` for in-place updates
- The SQL results schema `(subpop_name, state_var_name, age_group, risk_group, rep, timepoint, value)`

**What is missing:**

### 5a. Programmatic schedule swapping

The `DailyVaccines` schedule is currently loaded from a CSV at model initialization and stored
on `FluSubpopState`. There is no mechanism to replace a schedule (e.g. swap in a higher-coverage
vaccine CSV) without rebuilding the entire model from scratch.

**Task:** Add a `replace_schedule(schedule_name, new_df)` method to `FluSubpopModel`
(and a corresponding method on `FluMetapopModel`) that replaces the underlying DataFrame
for a named schedule and re-indexes it for O(1) lookup. This is the primitive needed by
the scenario runner.

**Acceptance criteria:** A test that swaps `daily_vaccines` on a live model, resets the
simulation, and confirms the new schedule is used in the next run.

---

### 5b. ScenarioRunner class

A `ScenarioRunner` (in `clt_toolkit/` or a new `flu_core/flu_scenarios.py`) that:

1. Accepts a `FluMetapopModel` as the baseline, plus a list of named scenario definitions.
   Each scenario definition specifies which schedules or parameters differ from baseline
   (e.g. `{"daily_vaccines": higher_coverage_df}`).
2. For each scenario:
   - Applies the scenario's schedule/parameter overrides via `replace_schedule()` or
     `updated_dataclass()` + `modify_subpop_params()`
   - Runs an `Experiment` with a user-specified number of replicates
   - Tags all results with a `scenario_name` column before writing to the shared database
3. Restores the baseline configuration between scenarios.

**Schema change needed:** Add a `scenario_name TEXT` column to the `results` SQL table
(or use a separate table per scenario — evaluate trade-offs given expected database size).

**Acceptance criteria:** Running two scenarios (baseline and +20% vaccine coverage) produces
a database with results that can be filtered by `scenario_name`.

---

### 5c. Variance-reduction via paired replicates

To detect small differences between scenarios, paired replicates (same RNG seed across
scenarios) are strongly preferred over independent draws. The current `Experiment` loop
re-seeds from a fresh `numpy.random.Generator` on each call.

**Task:** Allow `ScenarioRunner` to fix a common seed list across scenarios so that
replicate `i` in the baseline and replicate `i` in each counterfactual use the same
stochastic path up to the point where vaccine schedules diverge. This may require passing
a seed or `SeedSequence` into `Experiment.run_static_inputs()`.

---

## Item 6 — Outcome Metrics and Interactive Analysis

**Goal:** Functions that translate raw simulation output (compartment histories, transition
variable histories) into the epidemiological quantities needed for a vaccination impact report,
plus interactive marimo notebooks for parameter exploration and scenario comparison.

All notebooks in this project use **marimo** instead of Jupyter.

**What already exists:**
- `Experiment.get_state_var_df()` returns a (reps × timepoints) DataFrame for any state variable
- `aggregate_daily_tvar_history()` in `sampling.py` sums transition variable histories across
  subpopulations and timesteps-per-day into daily totals
- `clt_toolkit/plotting.py` has matplotlib-based plot functions for compartment histories,
  total infected, deaths, and epi metrics (M/MV), but they are not age/risk stratified and
  cannot overlay multiple runs

**What is missing:**

### 6a. Core outcome metric and plotting functions (`flu_core/flu_outcomes.py`)

Create `flu_core/flu_outcomes.py` with two sections: computation functions and plotting
wrappers. All computation functions should accept optional `age_group` and `risk_group`
arguments for stratification (passing `None` sums across that dimension).

**Computation functions:**

| Function | Inputs | Output |
|----------|--------|--------|
| `daily_hospital_admissions(metapop_model, ...)` | `FluMetapopModel`, optional subpop/age/risk filters | (days,) array of daily new hospital admissions = ISH→HR + ISH→HD transition flows, summed across timesteps-per-day |
| `daily_new_infections(metapop_model, ...)` | same | (days,) array of daily S→E flows |
| `cumulative_hospitalizations(metapop_model, ...)` | same | scalar season-total hospitalizations |
| `cumulative_deaths(metapop_model, ...)` | same | scalar HD→D total |
| `attack_rate(metapop_model, ...)` | same | scalar: cumulative infections / initial susceptible population |
| `vaccine_preventable_events(baseline_model, counterfactual_model, metric_fn)` | two `FluMetapopModel` objects + one of the above metric functions | scalar difference (requires paired runs from 5c) |
| `summarize_outcomes(values, credible_interval=0.95)` | (reps,) array or list | dict with `mean`, `median`, `lower_ci`, `upper_ci` — standard interface for multi-replicate reporting |

**Note:** Hospital admissions and new infections must use transition variable histories
(`ISH_to_HR`, `ISH_to_HD`, `S_to_E`), not compartment differencing. Ensure these appear
in `SimulationSettings.transition_variables_to_save` before running.

**Plotting functions:**

Add corresponding plot wrappers in the same module. Each should accept a `FluMetapopModel`
(or list of models for multi-scenario overlays) and use `matplotlib`. They should optionally
accept an `age_group` and `risk_group` filter to enable stratified views.

| Plot function | What it shows |
|---------------|---------------|
| `plot_compartment_history(model, compartment_names, ...)` | Time series of selected compartments, aggregated or by age/risk group; replaces the unsegmented `plot_metapop_basic_compartment_history` |
| `plot_epi_metrics(model, ...)` | M and MV over time, one line per age group when stratified |
| `plot_daily_new_infections(model, ...)` | Daily S→E flow aggregated to one curve per subpopulation |
| `plot_daily_hospital_admissions(model, ...)` | Daily ISH→HR + ISH→HD; optionally overlaid across multiple models with a legend for each scenario |
| `plot_attack_rate_by_age(model)` | Bar chart of attack rate per age group |
| `plot_scenario_comparison(models_dict, metric_fn, ...)` | Given a `{scenario_name: FluMetapopModel}` dict, computes a scalar metric for each and produces a labeled bar or box plot for comparison |

---

### 6b. Interactive parameter exploration notebook

Create `flu_instances/examples/flu_interactive_explorer.py` as a **marimo** notebook.

**Model setup:** Reuse the exact model initialization from `flu_demo_2popAustin_2024_2025.ipynb`
(east + west Austin subpopulations, `austin_input_files_2024_2025/`, same RNG seeds) so
results are directly comparable to the existing demo. Keep the setup cells non-reactive
(i.e. run once at startup) to avoid re-loading files on every interaction.

**Interactive controls (marimo UI elements):**

- **Simulation mode toggle:** radio button — `Deterministic` (sets
  `transition_type = "binom_deterministic_no_round"`) vs. `Stochastic` (sets
  `transition_type = "binom"`). In stochastic mode, add a numeric input for number of
  replicates (default 1 for speed during exploration).
- **Parameter selector:** dropdown listing all scalar and age-vector parameters from
  `FluSubpopParams` (e.g. `beta_baseline`, `vax_induced_saturation`,
  `vax_immunity_reduction_hosp`, `inf_induced_immune_wane`, `E_to_IA_prop`, etc.).
- **Parameter value inputs:** once a parameter is selected, show either a single numeric
  slider/input (for scalar params) or one slider per age group (for age-vector params).
  Support entering up to three values simultaneously so curves for each value are overlaid
  on the same plots for comparison.
- **Vaccine coverage multiplier:** a separate slider (0.5× to 3×) that scales the
  `daily_vaccines` schedule uniformly across all age groups — the primary vaccination
  scenario knob. This is independent of the parameter selector above.
- **Subpopulation selector:** checkboxes for `east`, `west`, or `combined` to control
  which subpopulation(s) are shown in plots.
- **Age group selector:** multi-select for age groups (0–4 to show stratified view, or
  "all" to sum across groups).
- **Simulation length:** numeric input for number of days to simulate (default 200).

**Outputs — one plot per panel, all reactive to controls above:**

1. **Compartment histories:** line chart of selected compartments over time. Default view
   shows S, E, IP, ISR, ISH, IA, HR, HD, R, D summed across age groups. When an age group
   is selected, show one line per age group for each compartment. Multiple parameter values
   are overlaid as separate line styles/colors with a legend.
2. **M and MV curves:** time series of infection-induced and vaccine-induced immunity.
   When age groups are selected, show one line per age group per metric.
3. **Daily new infections:** daily S→E flow (from `flu_outcomes.daily_new_infections()`).
   Overlaid for multiple parameter values.
4. **Daily hospital admissions:** daily ISH→HR + ISH→HD (from
   `flu_outcomes.daily_hospital_admissions()`). Overlaid for multiple parameter values.
5. **Cumulative summary table:** a small reactive table showing, for each parameter value
   tested: total hospitalizations, total deaths, attack rate, and peak daily admissions.

**Notes on implementation:**
- Use `marimo.ui.slider`, `marimo.ui.dropdown`, `marimo.ui.radio`, `marimo.ui.number`,
  `marimo.ui.multiselect` for controls.
- Re-run simulation in a reactive cell triggered by any control change. For stochastic
  mode with multiple replicates, show median line + shaded 95% interval.
- Use `updated_dataclass()` to create parameter-modified model copies without mutating
  the baseline; use `replace_schedule()` (task 5a) for the vaccine coverage slider.
- Plots should use matplotlib (consistent with the rest of the codebase); wrap each in
  `mo.as_html(fig)` or use `mo.mpl.interactive(fig)` for display.

---

### 6c. Vaccination scenario comparison notebook

Create `flu_instances/examples/vaccination_scenario_analysis.py` as a **marimo** notebook
demonstrating end-to-end multi-scenario analysis using `ScenarioRunner` (task 5b):

1. Load the Austin 2024-2025 baseline model
2. Define counterfactual vaccine schedules: +10% and +20% coverage relative to baseline
3. Run `ScenarioRunner` with all three scenarios, N=100 stochastic replicates each (paired,
   from task 5c)
4. Compute and display:
   - Daily hospital admissions by scenario: median + 95% CI ribbon using
     `plot_daily_hospital_admissions()`
   - Vaccine-preventable hospitalizations (VPH): boxplot across replicates using
     `plot_scenario_comparison()`
   - Age-stratified VPH: bar chart using `plot_attack_rate_by_age()`
   - Cumulative deaths averted
5. Summary table: scenario | mean VPH | 95% CI | mean deaths averted | 95% CI

---

## Item 7 — Sensitivity Analysis Over Uncertain Parameters

**Goal:** Quantify how uncertainty in key model parameters propagates into the vaccination
scenario comparison, so that conclusions about VPH are robust to parameter uncertainty.

**What already exists:**
- `accept_reject_admits()` in `flu_accept_reject.py`: ABC calibration that accepts parameter
  draws when simulated hospitalizations achieve R² ≥ threshold vs. observed data
- `sample_uniform_metapop_params()` and `UniformSamplingSpec` in `sampling.py`: uniform
  prior sampling over arbitrary parameter subsets
- PyTorch deterministic variant (`flu_torch_det_components.py`) suitable for gradient-based
  analysis

**What is missing:**

### 7a. Parameter uncertainty propagation into scenario comparison

The current ABC sampler calibrates to hospitalization curves but does not propagate the
resulting parameter uncertainty into scenario comparisons. The standard approach:

1. Run ABC (or use torch calibration) to obtain M accepted parameter sets
   `{θ_1, …, θ_M}` for the baseline season
2. For each `θ_m`, run baseline and counterfactual scenarios with K stochastic replicates each
3. Compute VPH for each `(θ_m, k)` pair
4. Report VPH uncertainty that spans both parameter uncertainty (across m) and stochastic
   uncertainty (across k)

**Task:** Extend `accept_reject_admits()` (or write a wrapper) to return the list of accepted
`FluSubpopState` snapshots and accepted parameter dicts as Python objects (not only as JSON
files on disk). This makes it practical to feed accepted sets directly into `ScenarioRunner`.

---

### 7b. One-at-a-time (OAT) sensitivity analysis

For rapid iteration, implement a `one_at_a_time_sensitivity()` function that:

1. Takes a baseline `FluMetapopModel`, a list of parameter names with ranges, and a scalar
   outcome function (e.g. `cumulative_hospitalizations`)
2. For each parameter, sweeps it across its range while holding all others at baseline,
   running deterministic replicates
3. Returns a DataFrame of `(param_name, param_value, outcome)` suitable for tornado plots

This is a cheap first-pass screen to identify which parameters the VPH estimate is most
sensitive to, informing which parameters are worth including in the full ABC calibration.

---

### 7c. Sensitivity of VPH to vaccine efficacy assumptions

Vaccine efficacy (VE) is encoded across several parameters: `vax_induced_saturation`,
`vax_induced_immune_wane`, `vax_immunity_reduction_inf`, `vax_immunity_reduction_hosp`,
`vax_immunity_reduction_death`. These are often reported with confidence intervals in the
literature.

**Task:** Using the OAT or uniform sampling infrastructure, produce a dedicated sensitivity
marimo notebook `flu_instances/examples/vax_efficacy_sensitivity.py` that sweeps VE-related
parameters over plausible ranges (derived from the literature source documented in
ARCHITECTURE.md item 8) and shows how the VPH estimate changes. This directly addresses
the policy question: "how sensitive is our conclusion to assumptions about how well the
vaccine works?"

---

### 7d. Multi-metric acceptance criterion for ABC

The current `accept_reject_admits()` uses a single R² threshold on hospitalizations.
For vaccination analyses, matching the timing and age distribution of hospitalizations
simultaneously improves confidence that the model's immunity dynamics are correctly specified.

**Task:** Refactor the acceptance criterion in `accept_reject_admits()` (or add an
alternative entry point) to support a user-supplied `acceptance_fn(simulated, observed) -> bool`
callable. This allows multi-metric criteria such as:
- Total-season R² on hospitalizations ≥ 0.75, AND
- Age-group-specific peak timing within ±2 weeks

The existing R²-only criterion should remain the default to preserve backwards compatibility.
