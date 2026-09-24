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
- `MA_fit_config_14d_tv_w_cml.json` — the fit configuration that produced
  `fitted_params.json` (historical record; its runner script was removed).
- `data/` — the source data this example is built from, consolidated here
  from the former `generic_core/examples/massachusetts_vax/`. Contains the
  observed hospitalization time series (`hospitalizations_ts/`, plus the
  cumulative and end-of-season derivations), the schedule CSVs the model
  actually runs on (`schedules/`, `vaccination/`), population and contact
  matrices (`MA_pop/`, `massachusetts_population.csv`), raw source files
  (`original/`, the gridMET `sph_2025.nc`/`sph_2026.nc` humidity NetCDFs),
  and the scripts that regenerate the derived CSVs from the raw ones
  (`download_contact_matrices.py`, `extract_ma_humidity.py`,
  `clt_get_population.R`).

  Of all this, only `data/hospitalizations_ts/MA_flu_daily_hospitalizations.csv`
  is read at runtime — by `build_report_assets.py` and the notebook's
  baseline-fit-check section, via `ma_vax_shared.DATA_FOLDER`. The pipelines
  themselves read schedules from `schedules.json`, not from `data/`.

**Data-prep scripts (run by hand, only when regenerating inputs):**

- `split_hospitalizations_by_age.py` — splits the combined daily
  hospitalizations CSV into one file per age band.
- `compute_cumulative_hospitalizations.py` — turns those per-age daily files
  into cumulative series plus an end-of-season summary.
- `plot_fit_vs_actual.py` — plots a fitted metric time series against the
  observed data. Note: its default input
  (`MA_fitted_14d_tv__metric_timeseries.csv`) is not in the repo, so it must
  be pointed at a real file via its CLI arguments.

**Analysis (`analysis/`):**

- `analysis_ts_per_day_impact.md` — write-up of how the substep resolution
  (`ts_per_day`) affects this model's numerics.
- `ts_per_day_convergence.py` / `.csv` — the convergence study behind it.
  Note: its default `--fitted-params` file is not in the repo, so it must be
  pointed at a real fitted-params JSON via its CLI arguments.

**Pipeline 1 — standalone scenario runner:**

- `run_simulations_MA_vax.py` — generated-by-notebook export script. Runs a
  fixed dict of scenarios (`SCENARIOS`) once each, deterministic or
  stochastic per its `STOCHASTIC`/`UNCERTAINTY_SOURCE` settings, and writes
  every compartment/transition-variable daily history to a Hive-partitioned
  Parquet directory (`simulation_output/results_parquet/`, via
  `generic_core.results_io.write_results_parquet`), at two granularities:
  population totals (`results` table) and per-replicate/subpop/age-group/
  risk-group (`results_full` table). Self-contained; doesn't import anything
  below.
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
  ported from `MA_vax_standalone/counterfactual.py`, but run against this generic_core
  model instead of the hand-written `MA_vax_standalone/model.py` equations, as a check that
  the two agree.
- `run_counterfactual_tables_generic.py` — CLI driver: runs
  `counterfactual_generic.py` and writes the tables to CSV in an output
  folder.

**Pipeline 3 — counterfactual vaccination-impact tables (from pipeline 1's database):**

- `build_counterfactual_tables_from_db.py` — CLI driver: reads pipeline 1's
  results (`results_parquet/`, or an older `results.db` — both go through
  `generic_core.results_io.load_source`, specifically its `results_full`
  table, since the S.A.* tables need per-age-group arrays) and computes the
  exact same S.A.1-S.A.6 tables + vaccine-efficacy check as pipeline 2, but
  without running any simulation of its own — it reuses the same pure
  numeric helpers from
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

Writes `simulation_output/results_parquet/`. Edit the constants at the top of
the script (`STOCHASTIC`, `UNCERTAINTY_SOURCE`, `NUM_DAYS`, ...) or the
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
# 1. Run pipeline 1 first -- pipeline 3 reads its results
python generic_core/examples/MA_vax/run_simulations_MA_vax.py

# 2. Turn the results into the same S.A.* CSVs pipeline 2 writes
python generic_core/examples/MA_vax/build_counterfactual_tables_from_db.py \
    --db generic_core/examples/MA_vax/simulation_output/results_parquet \
    --out generic_core/examples/MA_vax/counterfactual_tables_from_db

# 3. View the results (same notebook as pipeline 2)
marimo edit generic_core/examples/MA_vax/counterfactual_notebook_generic.py
```

Point the notebook's "Results folder" box at
`counterfactual_tables_from_db` from step 2. `--db`/`--out` default to
exactly those paths (relative to this folder) if omitted.

`--db` accepts either a Parquet results directory (current default output,
read via `generic_core.results_io.load_source`) or an older SQLite
`results.db` (e.g. from the notebook's Analysis tab SQLite export) — both go
through the same loader. Either way it needs a `results_full` table; rerun
pipeline 1 if your source predates that table.

`--model-config`/`--fitted-params` default to this folder's `model_config.json`/
`fitted_params.json`, and only need overriding if you're working with a
differently-named or relocated pair (e.g. multiple fitted variants of the
same model kept side by side) -- pass either as a path, relative to the
current working directory or absolute. They feed only Table S.A.4 (the VE
sensitivity parameters table, recomputed from the params here rather than
read from `--db`); every other table comes from `--db` as already-simulated
arrays and doesn't touch these two flags. Keep them pointed at whatever
`model_config`/`fitted_params` pair `run_simulations_MA_vax.py` actually used
to produce `--db`'s results, or Table S.A.4's percentages won't describe the
same run as the rest of the tables.

Pipelines 2 and 3 produce numerically identical tables (verified against
each other for the deterministic baseline run) -- pipeline 2 is the more
convenient one-shot CLI if you don't already have a results database, and
demonstrates a second, independently-defined simulation pipeline against the
same config; pipeline 3 is the one that demonstrates the whole
notebook-export round trip.

## `MA_vax_single_age/` — single-age-group variant

`MA_vax_single_age/` is a one-age-group ("0+") version of this model, with its
own `model_config_MA_single_age.json` / `fitted_params_MA_single_age.json` /
`schedules.json` and its own `run_simulations_MA_vax_single_age.py` (pipeline
1's analog). Its `SCENARIOS` dict deliberately omits every age-targeted
scenario ("Vaccinate `<age>` only", "70% coverage (`<age>` only)") -- they
have no meaning with a single age group -- see the comment above `SCENARIOS`
in that script.

`build_counterfactual_tables_from_db_single_age.py` is pipeline 3's analog
for this folder: same `results_io`-based reading of a results source
(Parquet or SQLite) into S.A.1, S.A.4, S.A.5, S.A.6, dose accounting, and a
one-column vaccine-efficacy check. Tables S.A.2 and S.A.3 are not produced --
they're defined by the per-age-group breakdown this model doesn't have.
Table S.A.4's VE-preset parameter values are copied from
`run_simulations_MA_vax_single_age.py`'s `SCENARIOS` dict rather than
recomputed, since that script already carries the single-age-refit absolute
values as source of truth.

`counterfactual_generic_single_age.py` is pipeline 2's analog: builds and
runs the single-age model directly (`build_model`/`_run_reps`/scenario
builders), used only by the notebook's "always run fresh" sections below
(no `table_S_A_*`/CLI driver of its own -- that's what
`build_counterfactual_tables_from_db_single_age.py` is for).

`counterfactual_notebook_generic_single_age.py` is the display notebook,
same structure as `counterfactual_notebook_generic.py` minus the S.A.2/S.A.3
sections and collapsed to one age group everywhere (single line/column
instead of a 7-panel grid or an age-group selector).

```bash
# 1. Run pipeline 1 first
python generic_core/examples/MA_vax/MA_vax_single_age/run_simulations_MA_vax_single_age.py

# 2. Turn the results into CSVs
python generic_core/examples/MA_vax/MA_vax_single_age/build_counterfactual_tables_from_db_single_age.py \
    --db generic_core/examples/MA_vax/MA_vax_single_age/simulation_output_single_age/results_parquet \
    --out generic_core/examples/MA_vax/MA_vax_single_age/counterfactual_tables_from_db_single_age

# 3. View the results
marimo edit generic_core/examples/MA_vax/MA_vax_single_age/counterfactual_notebook_generic_single_age.py
```

`--db`/`--out` default to exactly those paths (relative to
`MA_vax_single_age/`) if omitted, matching the notebook's "Results folder"
default.

`--model-config`/`--fitted-params` work the same way as the multi-age
script's (see above) -- default to this folder's
`model_config_MA_single_age.json`/`fitted_params_MA_single_age.json`, only
feed Table S.A.4, and should be kept pointed at whatever pair
`run_simulations_MA_vax_single_age.py` used to produce `--db`. One
difference from the multi-age script: this model's age groups are
themselves read from `--model-config` (`age_risk.age_groups`, currently just
`["0+"]`) rather than fixed at 7, so passing a different `--model-config`
also changes what `AGE_GROUPS` the rest of the script uses.
