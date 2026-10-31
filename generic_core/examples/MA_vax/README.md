# MA_vax

Primary MA vaccination-impact analysis: 7 age groups, a fitted time-varying
transmission multiplier m(t), and a stochastic scenario ensemble that draws
parameter sets from the MCMC posterior. Set up to re-run with new data.

The previous version of this analysis, with its outputs, reports and side
analyses (single-age model, no-m(t) fit, deterministic runs), is in
`../MA_vax_2026_10/`. That folder also holds the Model Builder notebook used
to make the configs, the counterfactual display notebook, and the report and
presentation builders.

## Age groups

Every file uses the same order, from youngest to oldest. This is the model's
age index 0-6:

| idx | label (`age_groups`) | column / file suffix |
|----:|----------------------|----------------------|
| 0 | `0-0` (Under 1) | `0_0` |
| 1 | `1-4`   | `1_4`   |
| 2 | `5-12`  | `5_12`  |
| 3 | `13-17` | `13_17` |
| 4 | `18-49` | `18_49` |
| 5 | `50-64` | `50_64` |
| 6 | `65+`   | `65+` (hosp. wide CSV) / `65plus` (files) |

Source files may list the groups alphabetically (1-4, 13-17, 18-49, 5-12,
50-64, 65+, Under 1). Reorder them to the order above before building
any input.

## Files

- `model_config_MA_vax.json`: model structure, priors and initial conditions
  (population and seeds). Reads its schedule CSVs from `data/schedules/`.
- `fit_config_MA_vax.json`: fit targets and settings (MCMC, m(t) with 14-day
  knots). **The observed data and the schedule CSVs are embedded in this
  file**, so it must be regenerated whenever the data changes.
- `run_fitting_MA_vax.py`: writes `fitted_params_MA_vax.json`.
- `make_scenarios_MA_vax.py`: rewrites the `SCENARIOS`, `DESIGNED_PARAMS`,
  `DESIGNED_PARAM_RATIOS` and `DOSE_MULTIPLIER` blocks of
  `run_simulations_MA_vax.py` from the fitted parameters and the vaccination
  schedule. Coverage targets (`COVERAGE_TARGETS`, 50-70%) and relative dose
  cuts (`COVERAGE_DECREASES`, 5-20%) are set at its top. Re-run it after every
  new fit or schedule change.
- `run_simulations_MA_vax.py`: the stochastic, posterior-sampled scenario
  ensemble (`UNCERTAINTY_SOURCE = "parameters"`, 85 scenarios). It writes
  `simulation_output_MA_vax_param_set_stochastic/results_parquet/`. This was
  `run_simulations_MA_vax_param_set_stochastic.py` in `MA_vax_2026_10/`.
  Full run: about 1-1.5 h on 11 cores, about 25 GB of Parquet, and up to
  about 55 GB of free disk while it runs (the DuckDB staging file and the
  Parquet export coexist at the end).
- `build_counterfactual_tables_from_db.py`: Tables S.A.1-S.A.5, S.A.3 for each
  coverage target, the dose-decrease tables (`DECREASE_<D>pct_*`),
  `COVERAGE_SUMMARY.csv` and the vaccine checks, built from the simulation
  output. It uses `counterfactual_generic.py` and `ma_vax_shared.py`.
- `build_summary_tables_MA_vax.py`: presentation summary tables (hospitalizations
  averted, cumulative hospitalizations).
- `export_metric_timeseries_MA_vax.py` + `plot_fit_vs_actual.py`: fit check
  against the observed admissions.
- `figures/build_figures.py`: every figure and figure-table of the deck and
  report (Python + the ggplot scripts in `figures/*.R`), written to
  `figures/output/`.
- `report/build_report.py`: the report (`report/report_MA_vax.md`, `.html`,
  `.pdf`), with every number read from the tables above.
- `data/`: inputs and data-prep scripts (see below).

## Pipeline

```bash
cd generic_core/examples/MA_vax
# 0. data prep from the raw files in data/original/ (see "Data preparation")
python data/prepare_hospitalizations.py
python data/prepare_vaccination.py --update-model-config
# 1. fit
python run_fitting_MA_vax.py
# 2. scenarios from the new fit, then simulate them
python make_scenarios_MA_vax.py
python run_simulations_MA_vax.py
# 3. tables
python build_counterfactual_tables_from_db.py    # -> counterfactual_tables_from_db_MA_vax/
python build_summary_tables_MA_vax.py             # -> summary_tables_MA_vax/
# 4. fit check
python export_metric_timeseries_MA_vax.py \
    --db simulation_output_MA_vax_param_set_stochastic/results_parquet \
    --out baseline_new_H_MA_vax.csv --scenario baseline
python plot_fit_vs_actual.py --sim-csv baseline_new_H_MA_vax.csv \
    --output-dir fit_comparison_output_baseline
# 5. figures (needs R with ggplot2, dplyr, tidyr, readr, scales) -> figures/output/
python figures/build_figures.py
# 6. report -> report/report_MA_vax.{md,html,pdf} (PDF via headless Google Chrome)
python report/build_report.py
```

`run_simulations_MA_vax.py` refuses to overwrite an existing
`results_parquet/`: it writes `results_<timestamp>_parquet/` next to it
instead, and steps 3-5 then need `--db` pointing at that folder. Move the old
output away first to keep the default paths.

Figures in `figures/output/`, by where they are used:

| Figure | Made by | Shows |
|---|---|---|
| `vax_coverage_daily.png`, `vax_coverage_cumulative.png` | `build_figures.py` | scheduled vaccination by age group |
| `fit_check_daily_by_age.png`, `fit_check_cumulative_by_age.png` | `build_figures.py` | baseline posterior band vs observed admissions |
| `transmission_multiplier.png` | `build_figures.py` | m(t), humidity factor, their product |
| `ts_baseline_infection_protection_no_vax.png`, `ts_ve.png` | `plot_scenarios_ts.R` | burden averted, VE sensitivity |
| `S_A_2_pct_reduction_stacked_bar.png` | `plot_pct_reduction_stacked_bar.R` | who is protected by vaccinating whom |
| `ts_coverage_targets.png`, `ts_coverage_decreases.png` | `plot_scenarios_ts.R` | coverage scenarios over time |
| `coverage_response.png`, `coverage_heatmap.png` | `build_figures.py` | totals by coverage level and age group targeted |
| `S_A_3_<T>pct_per_100k_doses_stacked_bar.png` | `plot_per_100k_doses_stacked_bar.R` | averted per 100K additional doses |
| `DECREASE_<D>pct_per_100k_doses_stacked_bar.png` | `plot_per_100k_doses_stacked_bar.R` | added per 100K doses removed |
| `fit_parameters.csv`, `age_parameters.csv`, `contact_matrix.csv` | `build_figures.py` | parameter tables |

The previous deck (`temp/2026 10 08 - MA vaccination burden analysis.pptx`)
was built from the equivalent scripts in
`../MA_vax_2026_10/presentation_2026_09_22/graphs/` and
`../MA_vax_2026_10/build_report_assets*.py`; `figures/` replaces them for
this folder. The model diagram and the notebook screenshot are static images.

## Data preparation

The raw files in `data/original/` are restricted, and git ignores them.
`data/original/censored_data/` holds copies with altered values, and those can
be shared. Two scripts build every model input from the raw files:

- `data/prepare_hospitalizations.py` reads the weekly county x age admissions
  file (`influenza_timeseries_*.csv`). It sums the counties, spreads each week
  evenly over its 7 days (Sunday to Saturday) and writes
  `hospitalizations_ts/MA_flu_daily_hospitalizations.csv`. It then runs
  `split_hospitalizations_by_age.py` and `compute_cumulative_hospitalizations.py`
  to make the per-age, cumulative and end-of-season fit targets.
  - A week x age x county with no row counts as 0.
  - `--start/--end` choose the weeks kept (default: weeks ending
    2025-06-29 .. 2026-07-04).
- `data/prepare_vaccination.py` reads the weekly county x age coverage file
  (`MA_influenza_coverage_*.csv`). `DISTINCT_PT_FIRST_SHOT` is cumulative,
  so the script takes weekly increments and spreads them over the days of the
  week. It divides by the summed `POP_ESTIMATE` and writes
  `schedules/MA_flu_daily_vaccinations_proportions_array.csv`, a readable
  copy in `vaccination/`, and `massachusetts_population.csv`.
  - Drops in the cumulative series are reported and handled with a
    running max.
  - `--update-model-config` writes that population into
    `model_config_MA_vax.json` and rescales `seeds.E`.
  - Doses dated before the simulation start date are not applied by the model.

Both scripts aggregate to the whole state by default. `--mapping county_groups.csv`
(columns `county,group`) aggregates counties into groups instead, and writes
each group to `data/by_group/<group>/` with the same sub-folders.
- County names are matched case-, space- and `MA_`-prefix-insensitively, so
  one mapping covers both files (`Berkshire` / `MA_Berkshire`,
  `Cape and Islands` / `MA_CapeandIslands`).
- See `data/county_groups_example.csv`.
- Use `--input` to run on another file, for example the censored copies, and
  `--out-root DIR` to write somewhere other than `data/`.

Age labels are normalized as they are read, including the Excel-mangled
`4-Jan` (1-4) and `12-May` (5-12) in the admissions file.

## Inputs to replace with the new data

The steps below are what the scripts above do. They are kept as a reference
for the formats.

1. **Hospital admissions**
   - Raw download goes in `data/original/`.
   - Build `data/hospitalizations_ts/MA_flu_daily_hospitalizations.csv`.
     Header: `Date,0_0,1_4,5_12,13_17,18_49,50_64,65+`. Dates use `m/d/yy`.
     There is one row per day, holding daily admissions (weekly counts / 7).
   - Then run the two data-prep scripts. They write the per-age files
     (`hospitalizations_ts/`, `hospitalizations_ts_cumul/`) and
     `hospitalizations_end_cumul/MA_flu_end_of_season_hospitalizations_by_age.csv`.
     These are the 8 fit targets.
2. **Vaccination coverage**
   - Raw download goes in `data/original/`.
   - Build `data/schedules/MA_flu_daily_vaccinations_proportions_array.csv`.
     Header: `date,daily_vaccines`. Each value is a quoted 7x1 nested list, for
     example `"[[0],[0.0001],...,[0.0004]]"`. Each entry is the share of that
     age group vaccinated *that day* (daily increments, not cumulative
     coverage), in the age order above.
   - Last time, the wide daily-proportions CSV was turned into this array
     format with
     `../MA_vax_2026_10/data/original/MA_flu_daily_vaccinations_proportions_array_formatting.xlsx`.
3. **Population**
   - Update `model_config_MA_vax.json` → `initial_conditions.aggregate_pop.population`
     (7x1, age order above) and `total_population` (their sum).
   - Rescale `initial_conditions.aggregate_pop.seeds.E`. These seeds were
     about 2.9e-5 × the population of each age group.
   - Optionally keep a record in `data/massachusetts_population.csv`
     (`age,population`, with age index 0-6). `data/clt_get_population.R` is
     the census pull used last time.
   - Vaccination proportions are relative to the population of each age
     group, so build them with the same population numbers.

Kept unchanged from the previous analysis: humidity
(`data/schedules/ma_absolute_humidity.csv`, data through 2026-06-14), the
school/work calendar (to 2026-07-31) and the Mistry 2021 contact matrices
(`data/MA_pop/`, which are also embedded in the model config). If the new
data run past mid-June 2026, regenerate humidity with
`data/extract_ma_humidity.py`. Its gridMET `.nc` inputs are git-ignored and
are in `../MA_vax_2026_10/data/`.

## Configs and scripts to update after the data are in

- `fit_config_MA_vax.json`: regenerate it from the Model Builder notebook
  (load `model_config_MA_vax.json`, upload the 7 per-age daily CSVs and the
  end-of-season CSV as targets, then use the Fitting tab export). Otherwise,
  edit each target's `observed` array, plus `schedule_csvs.daily_vaccines_df`
  (and humidity or calendar if those changed). Check `run_kwargs.start_date`
  and `fit_config.sim_days` against the new data window.
- `run_simulations_MA_vax.py`:
  - `NUM_PARAM_SETS` (627 last time) must not exceed the number of
    `accepted_params` in the new `fitted_params_MA_vax.json`.
  - Check `NUM_DAYS` / `START_DATE`.
  - Recompute the `DOSE_MULTIPLIER` entries for the 50% / 55% / 70%
    coverage scenarios, including `Low/High VE + X% coverage`. They are
    `max(1, target / baseline scheduled coverage)` per age group, so they
    depend on the vaccination schedule. Use
    `counterfactual_generic.scheduled_coverage()` /
    `coverage_multiplier_for_target()`.
  - The literal fitted values in `SCENARIOS` are overwritten by each
    posterior draw, except the parameters listed in `DESIGNED_PARAMS`, which
    are scaled through `DESIGNED_PARAM_RATIOS`. Refresh them anyway (for
    example by re-exporting from the notebook's Analysis tab) so they match
    the new fit.
