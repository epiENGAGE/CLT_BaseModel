# MA_vax_standalone

Massachusetts-specific age-structured SEIR + vaccination flu model: calibration,
counterfactual vaccination scenarios, and a PNAS-style vaccine-impact table
pipeline for the 2025-2026 season.

## File overview

| File | Role |
|---|---|
| [`model.py`](model.py) | Core model library (no CLI). Single source of truth for the model logic. |
| [`ma_vax.py`](ma_vax.py) | Hand-authored marimo notebook for interactively exploring one simulation run (params, input loading, plots). Predates `model.py`; the model logic was lifted from this notebook. |
| [`fit_bayesian.py`](fit_bayesian.py) | CLI. Bayesian calibration of `model.py` to observed hospitalizations (emcee MCMC and/or pyabc ABC-SMC). |
| [`scenarios.py`](scenarios.py) | Plain dict of named counterfactual-scenario overrides, edited by hand. |
| [`run_scenarios.py`](run_scenarios.py) | CLI. Runs every scenario in `scenarios.py` against a fitted run and dumps full daily compartment/transition detail to NetCDF. |
| [`counterfactual.py`](counterfactual.py) | Library (no CLI) implementing the PNAS-2505175122-style paired-stochastic vaccine-impact tables (S.A.1-S.A.6 analogs). |
| [`run_counterfactual_tables.py`](run_counterfactual_tables.py) | CLI. Computes the `counterfactual.py` tables and saves them to CSV. |
| [`counterfactual_notebook.py`](counterfactual_notebook.py) | marimo notebook that loads and displays CSVs written by `run_counterfactual_tables.py`. Never runs a simulation itself. |
| [`data/`](data/) | Raw/source data and data-prep scripts (population, contact matrices, humidity). See below. |

Everything downstream of `model.py` imports it as `from MA_vax_standalone import model` (or
`counterfactual`, `fit_bayesian`, etc.), so run all commands below from the repo
root with `uv run python MA_vax_standalone/<script>.py ...`.

---

## `model.py`

Not run directly (its `__main__` block is just a smoke test: `uv run python
MA_vax_standalone/model.py`). Provides:

- `default_params()` / `load_inputs(data_folder, params)` — build the parameter
  dict and the per-day input arrays (population, contact matrices, humidity,
  school/work calendar, vaccination schedule) from the CSVs/JSON in
  `model.DATA_FOLDER` (`generic_core/examples/massachusetts_vax/` by default —
  see [Data](#data) below).
- `run_simulation(inputs)` — deterministic run, returns a daily summary
  DataFrame.
- `simulate_detailed(inputs, stochastic, n_reps, seed, steps_per_day)` — full
  per-age-group daily compartment counts and transition flows as an
  `xarray.Dataset`, deterministic or chain-binomial stochastic, with optional
  sub-daily stepping. Used by `run_scenarios.py` and `counterfactual.py`.
- `apply_overrides` / `apply_scenario` / `apply_vax_scenario` — apply a dict of
  parameter/vaccination-schedule overrides to an inputs dict; this is the
  common "scenario" interface every other script in this folder builds on.

## `ma_vax.py`

Interactive notebook, not a batch script:

```
uv run marimo edit MA_vax_standalone/ma_vax.py
```

Lets you edit parameters, inspect the loaded inputs, run one simulation, and
look at plots, all inline — useful for manual exploration before scripting
something with `model.py` directly. Produces no file outputs.

## `fit_bayesian.py`

Calibrates `beta_baseline`, `humidity_impact`, an E0 scale, an IHR scale, and
(optionally) a monthly time-varying transmission multiplier m(t), against
`generic_core/examples/massachusetts_vax/MA_flu_daily_hospitalizations_total.csv`.

```
uv run python MA_vax_standalone/fit_bayesian.py --method both            # emcee + pyabc, with m(t)
uv run python MA_vax_standalone/fit_bayesian.py --method emcee --no-tvbeta
```

Key flags: `--method {emcee,pyabc,both}`, `--tvbeta`/`--no-tvbeta`, `--tau`
(m(t) smoothness), `--nwalkers`/`--nsteps`/`--workers` (emcee), `--pop`/`--gens`
(pyabc), `--seed`.

**Inputs:** `model.default_params()` + `model.load_inputs()` (see above) +
the observed hospitalization CSV.

**Outputs:** a timestamped `MA_vax_standalone/outputs_<date>_<time>h<min>m<sec>s/` folder
containing, per method:
- `fit_summary_{method}.csv` — full posterior parameter summary (mean, 5%, 95%)
- `fit_parameters_{method}.csv` / `fit_parameters_best_{method}.csv` — one-row
  "ready to run" override table, posterior mean / single highest-support
  sampled parameter vector respectively (what `load_fitted_run` reads back;
  see `best_fit_theta` — unlike the mean, the "best" point is an actual
  sampled point, so it preserves whatever correlation structure the
  posterior has between parameters)
- `fit_beta_multiplier_{method}.csv` / `fit_beta_multiplier_best_{method}.csv`
  — matching m(t) curve, one row/day (only if m(t) was fit)
- `fit_posterior_predictive.png`, `fit_beta_multiplier.png`, `fit_corner_{method}.png`

`fit_bayesian.load_fitted_run(output_folder, method, point="best")` reads
that folder back into `(overrides, beta_multiplier_arr)` ready for
`model.simulate_new_H` — this is how every downstream script
(`run_scenarios.py`, `counterfactual.py`'s `load_base_inputs`) picks up a
calibrated run. `point="mean"` reads the posterior-mean files instead;
older output folders (predating the `"best"` point estimate) only have the
mean files, and `load_fitted_run(..., point="best")` raises a clear
`FileNotFoundError` if asked to load a folder without them.

## `scenarios.py`

No CLI — just `SCENARIOS: dict[str, dict]`, scenario name -> override dict
consumed by `model.apply_scenario` (vaccination-rate multipliers per age
group, `vax_susceptibility_scale`, `IV_to_H_prop_scale`, or any raw param
override). Edit this file directly to add/change scenarios; see its docstring
for the recognised keys and `model.AGE_GROUP_LABELS` for age-group order.

## `run_scenarios.py`

```
uv run python MA_vax_standalone/run_scenarios.py --fit-folder MA_vax_standalone/outputs_2026-07-08_15h24m30s
uv run python MA_vax_standalone/run_scenarios.py --fit-folder MA_vax_standalone/outputs_2026-07-08_15h24m30s \
    --stochastic --n-reps 200 --steps-per-day 4
```

Key flags: `--fit-folder` (required, an `outputs_*` folder from
`fit_bayesian.py`), `--method {emcee,pyabc}` (default `emcee`), `--stochastic`
(default off = one deterministic replication), `--n-reps`, `--seed`,
`--steps-per-day` (sub-daily Euler/binomial stepping; output stays daily),
`--out`.

**Inputs:** a `fit_bayesian.py` output folder + `scenarios.py`.

**Outputs:** `<fit-folder>/scenarios_<date>_<time>h<min>m<sec>s/scenario_<name>.nc`
— one NetCDF per scenario (`model.simulate_detailed`'s full
`(replication, day, age_group)` compartment + transition Dataset), plus a
per-scenario console summary (total new hospitalizations/deaths, mean ± std).

## `counterfactual.py`

Library (no CLI) that replicates the table structure of PNAS 2505175122
Supplementary Tables S.A.1-S.A.6, adapted to this model's 7 age groups, the
Massachusetts population, and the 2025-2026 season:

- Scenario builders: `no_vaccine_scenario`, `single_age_only_scenario`,
  `infection_protection_only_scenario`, `coverage_70pct_scenario` (bisects a
  vaccination-rate multiplier to hit 70% cumulative coverage), `VE_SCENARIOS`
  (low/baseline/high VE presets — currently placeholder multipliers, tune to
  taste).
- `scenario_totals` / `averted_summary` — run a scenario `n_reps` times
  (paired on a common seed across scenarios, for variance reduction) or once
  deterministically (`stochastic=False`), and turn two scenario runs into a
  DataFrame of percent/per-100K/per-100K-doses hospitalizations averted with
  median + 95% CI.
- `table_S_A_1` ... `table_S_A_6` — compose the above into each paper table's
  shape.
- `load_base_inputs(fit_folder=None, method="emcee", point="best")` — build
  the baseline inputs dict, optionally applying a `fit_bayesian.py` fitted
  run at the given point estimate (`"best"` or `"mean"`, see above).
- `save_all_tables` / `load_saved_tables` — write/read every table as CSV;
  used by `run_counterfactual_tables.py` and `counterfactual_notebook.py`
  respectively.

Not run directly — import it, or use the CLI/notebook below.

## `run_counterfactual_tables.py`

```
uv run python MA_vax_standalone/run_counterfactual_tables.py --n-reps 1000
uv run python MA_vax_standalone/run_counterfactual_tables.py \
    --fit-folder MA_vax_standalone/outputs_2026-07-08_15h24m30s --method emcee --n-reps 500
uv run python MA_vax_standalone/run_counterfactual_tables.py --deterministic   # fast sanity check
```

Key flags: `--fit-folder` (optional; omit to use `model.default_params()`
unfitted), `--method {emcee,pyabc}`, `--point {mean,best}` (default `best` —
see `counterfactual.load_base_inputs`/`fit_bayesian.best_fit_theta`),
`--n-reps` (paired stochastic replications per scenario), `--deterministic`
(single deterministic replication instead, ignores `--n-reps`), `--seed`,
`--out`.

**Inputs:** optionally a `fit_bayesian.py` output folder.

**Outputs:** a timestamped `MA_vax_standalone/counterfactual_tables_<date>_<time>h<min>m<sec>s/`
folder containing `meta.json` (fit folder, method, point, n_reps, seed,
stochastic flag, timestamp) plus one CSV per table (`S_A_1.csv`, `S_A_4.csv`,
and `S_A_{2,3,5,6}_{pct_reduction,per_100k,per_100k_doses}.csv` for the
dict-valued tables — see `counterfactual.DICT_TABLES`).

This is the slow step (many stochastic replications); run it once and reuse
its output folder in the notebook below.

## `counterfactual_notebook.py`

```
uv run marimo edit MA_vax_standalone/counterfactual_notebook.py
```

Enter a `counterfactual_tables_*` folder (from `run_counterfactual_tables.py`)
in the text box and it loads + renders all six tables — these are read
straight from CSV, so they stay fast regardless of how many replications the
CLI run used; re-run the CLI and point at the new folder to refresh numbers.

It also has a "Baseline fit check" section that runs one fresh deterministic
simulation (the only simulation this notebook runs) and compares it to the
raw per-age hospitalization data in `MA_flu_daily_hospitalizations.csv` (loaded
from `model.DATA_FOLDER`, i.e. `generic_core/examples/massachusetts_vax/`)
— cumulative totals by age group plus actual-vs-simulated curves. It reuses
`tables["meta"]`'s `fit_folder`/`method`/`point` so this comparison is always
built from the exact same baseline inputs as the CSV tables above it.

---

## Data

All input data now lives under `generic_core/examples/massachusetts_vax/`; no
`MA_vax_standalone/` code reads from `MA_vax_standalone/` itself:

- **`generic_core/examples/massachusetts_vax/`** — the processed CSVs/JSON
  actually read by `model.load_inputs` (via `model.DATA_FOLDER`): population,
  contact matrices, humidity, school/work calendar, vaccination schedule,
  observed hospitalizations.
- **`generic_core/examples/massachusetts_vax/data/`** — raw/source data and
  the scripts used to (re)build some of those processed inputs. Nothing at
  runtime reads this folder; it exists for provenance and regeneration.
  - `clt_get_population.R` — pulls Census population data via `tidycensus`.
  - `download_contact_matrices.py` — downloads Mistry-2021 contact matrices
    from the epydemix-data repo, writing into `MA_pop/` next to the script.
  - `extract_ma_humidity.py` — extracts daily MA-averaged specific humidity
    from the gridMET NetCDFs (`sph_2025.nc`, `sph_2026.nc`) in this folder.
    Those `.nc` files are gitignored — re-download them if you need to re-run it.
  - `data_source.rtf` — links to the upstream vaccination-coverage and
    hospital-admissions data sources.
  - `original/` — untouched upstream vaccination-coverage and population files,
    before the reformatting that produced the processed CSVs.
