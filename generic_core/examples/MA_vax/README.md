# MA_vax

Three pipelines built on the same exported MA_vax model (`model_config.json`
\+ `fitted_params.json`), plus their shared inputs. Pipelines 1 and 3 chain
together; pipeline 2 is fully independent of both.

## Files

**Inputs (shared by all three pipelines):**

- `model_config.json` — the model config exported from the Model Builder
  notebook (compartments, transitions, params, initial conditions).
- `fitted_params.json` — fitted/posterior parameters from the Bayesian fit
  (best point estimate + accepted posterior sets + time-varying transmission
  multiplier increments).
- `schedules.json` — uploaded schedule CSVs (humidity, school/work calendar,
  mobility, vaccination), single-population only.
- `scenario_config_MA_vax_small.json` — a saved snapshot of the notebook's
  Analysis-tab scenario configuration (historical record of what was
  configured there; not read by any pipeline below).

**Pipeline 1 — standalone scenario runner:**

- `run_simulations_MA_vax.py` — generated-by-notebook export script. Runs a
  fixed dict of scenarios (`SCENARIOS`) once each, deterministic or
  stochastic per its `STOCHASTIC`/`UNCERTAINTY_SOURCE` settings, and writes
  every compartment/transition-variable daily history to a SQLite database
  (`simulation_output/results.db`), at two granularities: population totals
  (`results` table) and per-replicate/subpop/age-group/risk-group
  (`results_full` table). Self-contained; doesn't import anything below.
  Includes two scenarios beyond what the notebook's Analysis tab exports on
  its own ("Low VE + 70% coverage (all ages)", "High VE + 70% coverage (all
  ages)") — hand-added so pipeline 3's Table S.A.6 has what it needs; see
  the comment above them in `SCENARIOS`.

**Pipeline 2 — counterfactual vaccination-impact tables (live simulation):**

- `counterfactual_generic.py` — the analysis engine. Builds/runs models
  directly against `model_config.json`/`fitted_params.json`/`schedules.json`
  (independent of `run_simulations_MA_vax.py`), defines scenario builders
  (`no_vaccine_scenario`, `single_age_only_scenario`, `ve_scenarios`,
  `coverage_70pct_scenario`, `infection_protection_only_scenario`, and the
  `named_scenarios()` registry that exposes all of them by name), and
  computes the S.A.1-S.A.6 tables plus a vaccine-efficacy mechanism check —
  ported from `MA_vax/counterfactual.py`, but run against this generic_core
  model instead of the hand-written `MA_vax.model` equations, as a check that
  the two agree.
- `run_counterfactual_tables_generic.py` — CLI driver: runs
  `counterfactual_generic.py` and writes the tables to CSV in an output
  folder.

**Pipeline 3 — counterfactual vaccination-impact tables (from pipeline 1's database):**

- `build_counterfactual_tables_from_db.py` — CLI driver: reads pipeline 1's
  `results.db` (specifically `results_full`, since the S.A.* tables need
  per-age-group arrays) and computes the exact same S.A.1-S.A.6 tables +
  vaccine-efficacy check as pipeline 2, but without running any simulation of
  its own — it reuses the same pure numeric helpers from
  `MA_vax.counterfactual` (`averted_summary`, `_rate_ratio_col`,
  `_matched_cohort_ratio_col`) that pipeline 2 does, applied to arrays
  reconstructed from the database instead of a fresh simulation. This is the
  one place `run_simulations_MA_vax.py`'s output feeds into anything else in
  this folder — demonstrates a full round trip (notebook Analysis tab →
  Export tab → exported script → database → tables → notebook) with the
  notebook-exported script as the only thing that ever runs a simulation.
  Writes CSVs in the same filenames/format as pipeline 2, so
  `counterfactual_notebook_generic.py` reads either pipeline's output
  identically.

**Shared display notebook (pipelines 2 and 3 both feed it):**

- `counterfactual_notebook_generic.py` — marimo notebook that loads and
  displays the S.A.* CSVs from whichever results folder you point it at
  (pipeline 2's or pipeline 3's — same format), plus a few sections that
  always run fresh (cheap, deterministic) simulations directly through
  `counterfactual_generic.py`, independent of both CSV pipelines: vaccination
  coverage, matched-cohort attack-probability curves, a baseline-fit-check
  against raw hospitalization data, vaccination-vs-epidemic-timing, and an
  interactive epi-curves-by-scenario explorer.

## Running pipeline 1 (standalone scenario runner)

```bash
python generic_core/examples/MA_vax/run_simulations_MA_vax.py
```

Writes `simulation_output/results.db`. Edit the constants at the top of the
script (`STOCHASTIC`, `UNCERTAINTY_SOURCE`, `NUM_DAYS`, ...) or the
`SCENARIOS`/`DOSE_MULTIPLIER`/`DESIGNED_PARAMS` dicts to change what it runs.

## Running pipeline 2 (counterfactual tables, live simulation)

```bash
# 1. Compute the tables (deterministic; drop --deterministic and add
#    --n-reps for stochastic runs with confidence intervals)
python generic_core/examples/MA_vax/run_counterfactual_tables_generic.py \
    --deterministic --out generic_core/examples/MA_vax/counterfactual_tables_det

# 2. View the results
marimo edit generic_core/examples/MA_vax/counterfactual_notebook_generic.py
```

In the notebook, point the "Results folder" box at the folder from step 1
(it defaults to `counterfactual_tables_det`, matching the example above).

## Running pipeline 3 (counterfactual tables, from pipeline 1's database)

```bash
# 1. Run pipeline 1 first -- pipeline 3 reads its results.db
python generic_core/examples/MA_vax/run_simulations_MA_vax.py

# 2. Turn results.db into the same S.A.* CSVs pipeline 2 writes
python generic_core/examples/MA_vax/build_counterfactual_tables_from_db.py \
    --db generic_core/examples/MA_vax/simulation_output/results.db \
    --out generic_core/examples/MA_vax/counterfactual_tables_from_db

# 3. View the results (same notebook as pipeline 2)
marimo edit generic_core/examples/MA_vax/counterfactual_notebook_generic.py
```

Point the notebook's "Results folder" box at
`counterfactual_tables_from_db` from step 2. `--db`/`--out` default to
exactly those paths (relative to this folder) if omitted.

Pipeline 3 requires `results.db` to have a `results_full` table -- rerun
pipeline 1 if it doesn't (an older `results.db` from before that table was
added won't have it).

Pipelines 2 and 3 produce numerically identical tables (verified against
each other for the deterministic baseline run) -- pipeline 2 is the more
convenient one-shot CLI if you don't already have a `results.db`, and
demonstrates a second, independently-defined simulation pipeline against the
same config; pipeline 3 is the one that demonstrates the whole
notebook-export round trip.
