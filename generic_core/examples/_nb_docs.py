# _nb_docs.py
# Section: Documentation tab cell
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _docs_display(mo, main_tab):
    mo.stop(main_tab.value != "Documentation", None)
    mo.vstack([
        mo.md("""
# CLT Model Builder — User Guide

This notebook lets you build a config-driven compartmental epidemic model, fit it to
data, run forecasts, export scripts for server runs, and compare scenarios — all
without writing code.  Each capability lives in its own tab.  All tabs share the same
model defined in **Model Builder**.

---

## Output directory

The text box at the top of every page sets the folder where results are auto-saved.
The default is `~/clt_outputs/`.  Change it before running anything if you want
output in a specific location.

---

## Tab 1 — Population & Geography

**Purpose:** Define the population dimensions and (optionally) fetch real contact
matrices for a geography. Do this **first** — the rest of the model is built in the
Model Builder tab and depends on the age/risk group counts set here.

### What you configure
- **Age groups** — two modes:
  - *Count only* — just a number `A` (for abstract models with no real age bands).
  - *Named age bands* — e.g. `0-4, 5-17, 18-49, 50-64, 65+`. Bands must start at 0,
    be contiguous, and end in an open band `x+` (with `x ≤ 84`). `A` = number of bands.
- **Risk groups** — the number `R`.
- **Population mode** — *Single population* or *Metapopulation* (with a folder path).
- **Contact matrices** — when using named age bands you can **fetch** the total / school /
  work matrices for a geography (Mistry 2021, via the optional `epydemix` package).

### Contact-matrix fetch
- Requires **named age bands** (the fetcher needs age-group definitions). In count-only
  mode, supply contact-matrix CSVs in Model Builder → Step 4 instead.
- Choose a **US state** or a **Country** (both are searchable dropdowns of the
  epydemix-data location names).
- For metapopulation models, choose the geography **scope**:
  - *Same for all subpops* — one geography; matrices are shared across subpops.
  - *Per-subpopulation* — one geography per subpop; each subpop gets its own matrices,
    written into the config's `subpop_params` section.
- Press **Fetch contact matrices**. Results are written into the config and used at run
  time. Requires `pip install epydemix` and internet access; if unavailable, the tab shows
  an install hint and you can fall back to CSVs.

---

## Tab 2 — Model Builder

**Purpose:** Define the structure of your epidemic model and do a quick preview simulation.

### Steps

| Step | What you configure |
|------|--------------------|
| 0 — Load config | Optionally load a previously saved `model_config.json` to pre-fill all fields. |
| 1 — Compartments | Name each compartment (e.g. `S`, `E`, `I`, `R`).  The first compartment receives the bulk of the initial population. |
| 2 — Transitions | Define flows between compartments.  Each transition needs a name, a "from" compartment, a "to" compartment, and a rate template (`constant_param`, `param_product`, `immunity_modulated`, `force_of_infection`, `force_of_infection_travel`, or `scheduled_exact`). |
| 3 — Parameters | Set numeric values for all parameters referenced by your rate templates (e.g. `beta_baseline`, `sigma`, `gamma`). |
| 4 — Schedules & immunity | Optionally upload CSVs for time-varying schedules: absolute humidity, school/work calendars, mobility, and daily vaccines. Contact matrices come from the Population & Geography tab or CSVs here. |
| 5 — Diagram | Preview the compartment diagram generated from your transitions. |
| 6 — Initial conditions | Set the total population and seed counts for compartments 2–N. |
| 7 — Simulation settings | Choose deterministic vs stochastic, number of replicates, timesteps per day, start date, and which transition variables to save. |
| 8 — Config preview & download | Review the full `model_config.json` and download it. |
| 9 — Run | Click **Run simulation** to see epidemic curves and a summary table. |

**Metapopulation mode:** Enabled in the **Population & Geography** tab, which asks for a
folder path containing:
- `metapop_config.json` — subpopulation names and travel matrix
- `initial_conditions_<SubpopName>.json` — per-subpop initial conditions
- Optional per-subpop schedule CSVs (`school_work_calendar_<name>.csv`, `vaccines_<name>.csv`)
- Optional shared schedule CSVs (`absolute_humidity.csv`, `mobility_modifier.csv`)

**Auto-save:** The model config is written to `{output_dir}/model_config.json` every time
any setting changes, so you never lose your work.

---

## Tab 3 — Fitting

**Purpose:** Estimate unknown parameters by fitting the model to an observed time series.

### Steps

1. **Observed data** — Upload a CSV or provide a file path.  The file must have at least
   two columns; all columns whose names are not `date`, `day`, `time`, or `week` are
   treated as the observed values (the first such column is used).
2. **Target** — Choose which model output to fit.  This can be any compartment name or
   any transition variable name (as listed in Step 8 of Model Builder).
3. **Parameters and bounds** — Enter a comma-separated list of parameter names to fit,
   then provide bounds as a JSON object: `{"beta_baseline": [0.05, 0.8]}`.  If you
   omit bounds for a parameter, the notebook guesses ±80 % around the current value.
4. **Method**
   - *Adam (gradient)* — PyTorch-based gradient descent.  Fast and accurate for smooth
     loss landscapes.  **Requires a transition variable as the target** (not a compartment).
   - *LBFGS (gradient)* — Second-order gradient method.  Often converges in fewer steps
     than Adam but each step is more expensive.  Same target constraint applies.
   - *Accept-reject* — Parameter-space random search that accepts samples with R² above
     a threshold.  Works with any target (compartment or transition) and does not require
     PyTorch.
5. Click **Run fitting**.

### Results

- **Loss / R² curve** — Shows fitting progress over iterations or samples.
- **Best-fit parameters** — The parameter values that minimised the loss (or maximised R²).

Auto-saved to `{output_dir}/fitted_params.json`.

### Tips

- For gradient methods, start with a small learning rate (0.001–0.01) and 100–200
  iterations; watch the loss curve to judge convergence.
- For accept-reject, increase "Max samples" if the best R² is still below the threshold
  after running.
- Gradient fitting fits parameters **globally** (all subpopulations share the same values).
  Use accept-reject for metapopulation models.

---

## Tab 4 — Forecast

**Purpose:** Run an ensemble forward projection using the fitted (or current) parameters.

### Steps

1. **Fitted parameters** — Toggle on "Use fitted params from Fitting tab" to apply the
   best-fit values automatically.  Or point to a `fitted_params.json` on disk.
2. **Settings** — Choose forecast horizon (days beyond the fit period), number of
   replicates, and stochastic vs deterministic.
3. Click **Run forecast**.

### Results

- **Epidemic curves** — Median + 95 % CI ribbon for each compartment.  A dashed vertical
  line and shaded region mark the end of the fit period.
- **Summary table** — Median peak value and peak day per compartment.

Auto-saved to `{output_dir}/forecast_ensemble.json`.

### Notes

- The simulation always starts from day 1 (initial conditions set in Model Builder).
  The "fit period" is just a visual annotation: the model runs for `fit_days + horizon`
  days in a single pass.
- Stochastic replicates use independent random seeds; increase replicates for smoother
  confidence intervals.

---

## Tab 5 — Export

**Purpose:** Generate a standalone Python script that can run your model on a server
or in a batch job, and download all configuration files.

### What is generated

- **`run_simulation.py`** — A self-contained script that loads `model_config.json` and
  optionally `fitted_params.json`, builds the model, runs each entry in a `SCENARIOS`
  dict, and saves results to a SQLite database (`simulation_output/results.db`).

  Edit the top of the script to configure:
  - `NUM_DAYS`, `NUM_REPS`, `STOCHASTIC`, `TIMESTEPS_PER_DAY`, `START_DATE`
  - `SCENARIOS` — a dict mapping scenario name to a `{param: value}` override dict

- **`model_config.json`** — The current model configuration.
- **`fitted_params.json`** — The best-fit parameter values (empty `{}` if fitting has
  not been run).

### Running the script

```bash
# Put all three files in the same directory, then:
python run_simulation.py
```

Results are stored in `simulation_output/results.db` as a table with columns
`scenario`, `rep`, `compartment`, `day`, `value`.

---

## Tab 6 — Analysis

**Purpose:** Compare how model outputs change across scenarios or parameter values.
Sensitivity and scenario analysis share identical output plots.

### Sub-tabs

#### Sensitivity
Vary **one parameter** across N values.  Each value becomes its own scenario, labelled
`param=value`.  Use this to understand how sensitive the model is to a single unknown.

- Select the parameter from the dropdown (populated from your model's `params`).
- Enter values as a comma-separated list: `0.1, 0.2, 0.3, 0.4`.

#### Scenario
Define **N parameter bundles**.  Each bundle is a named scenario with its own set of
parameter overrides.  Use this to compare specific interventions or assumptions.

Enter scenarios as a JSON object:
```json
{
  "baseline":   {},
  "high_beta":  {"beta_baseline": 0.4},
  "vaccination": {"beta_baseline": 0.2, "daily_vaccines_value": 5000}
}
```

### Shared run settings

| Setting | Description |
|---------|-------------|
| Simulation days | How many days to simulate for each scenario. |
| Replicates per scenario | How many stochastic runs per scenario (use 1 for deterministic). |
| Stochastic | Toggle on for binomial draws; off for deterministic (faster). |
| Output metric | Which compartment or transition variable to plot in the main chart. |

Click **Run analysis**.

### Results

- **Scenario comparison plot** — One line per scenario for the selected metric,
  with 95 % CI ribbons when replicates > 1.
- **Summary table** — Peak value, peak day, and day-end value for every
  (scenario, metric) combination.
- **Download summary CSV** — Export the summary table for use in external tools.

Auto-saved to `{output_dir}/analysis_results.json`.

---

## Advanced: modelling vaccination (`scheduled_exact`)

To move an exact, data-driven number of people between compartments each day
(e.g. vaccination `S → V`), use the **`scheduled_exact`** rate template instead
of a rate-based one. It bypasses the usual rate→probability machinery and applies
the scheduled transfer deterministically on the first timestep of each day.

How to set it up in **Model Builder**:

1. **Step 2 — Compartments:** add the destination compartment (e.g. `V`).
2. **Step 3 — Transitions:** add a transition `S → V` with rate template
   `scheduled_exact`. It references a schedule by name (the daily-vaccines
   schedule).
3. **Step 5 — Schedules:** supply the vaccines schedule, either as a constant
   value or via a `vaccines_<name>.csv` (column `daily_vaccines`, a JSON A×R
   array). **The schedule value is a daily _proportion_ of the origin
   compartment** (0–1), not an absolute count; it is rounded to an integer and
   capped at the available population each day.

Notes:
- `scheduled_exact` transitions are deterministic. They cannot be placed in a
  transition group or marked jointly-distributed (the config parser rejects this).
- Vaccination immunity (the `MV` metric) is configured separately in Step 5 — a
  compartment transfer (`scheduled_exact`) and a population-immunity metric
  (`vaccine_induced_immunity`) are independent mechanisms; use either or both.

---

## Advanced: multi-target fitting

The **Fitting** tab can fit several observed series at once. Add targets with the
"number of targets" control; each target independently specifies:

- the model output to match (a compartment or transition variable),
- an optional **slice** (subpopulation / age group / risk group),
- a **mode** — timeseries, scalar total, or proportions, and
- a **weight** `λ` controlling its contribution to the combined loss.

The objective minimised is the weighted sum of per-target losses. Practical advice:

- Keep targets on comparable scales, or use weights to balance them — a large-count
  target will otherwise dominate a small-count one.
- Avoid conflicting targets (e.g. fitting both a compartment and a transition that
  mechanically determine each other) — they can make the loss landscape ill-conditioned.
- Gradient methods (Adam / L-BFGS) require **transition-variable** targets; use
  accept-reject if any target is a compartment.

---

## Typical workflow

```
Population & Geography  →  Model Builder  →  Fitting  →  Forecast  →  Export
                                                ↓
                                             Analysis
```

1. Set age/risk groups, population mode, and (optionally) fetch contact matrices in
   **Population & Geography**.
2. Build your model in **Model Builder** and confirm the epidemic curves look sensible.
3. Go to **Fitting**, upload observed data, and fit key parameters.
4. Check the fit overlay in Fitting results, then switch to **Forecast** to project forward.
5. Use **Analysis** to quantify uncertainty (sensitivity) or compare policy scenarios.
6. When ready to run larger ensembles, go to **Export**, download the script and configs,
   and run them on your server.

---

## File formats

### `model_config.json`
The master configuration file.  It is read/written by Steps 0 and 9 of Model Builder
and auto-saved whenever any setting changes.

### Observed data CSV (Fitting tab)
Any CSV with at least two columns.  The first column whose name is not
`date`, `day`, `time`, or `week` is used as the observed values.

Example:
```
date,hospitalizations
2024-01-01,12
2024-01-02,18
2024-01-03,25
```

### Fitted params JSON
A flat dict of `{param_name: value}` pairs, e.g.:
```json
{"beta_baseline": 0.23, "sigma": 0.5}
```

### Metapop config JSON (`metapop_config.json`)
`subpopulations` is an **ordered list** of names; `travel_matrix` is an N×N
matrix (N = number of subpopulations) whose rows sum to 1.
```json
{
  "subpopulations": ["SubpopA", "SubpopB"],
  "travel_matrix": [[0.95, 0.05], [0.05, 0.95]]
}
```

### Initial conditions JSON (`initial_conditions_<name>.json`)
```json
{
  "compartments": {"S": [[950000]], "I": [[50000]]},
  "epi_metrics":  {}
}
```
Arrays are shape `[age_groups][risk_groups]`.
"""),
    ])
    return

