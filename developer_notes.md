# Journal

## 2026 03 13

### Scenario configuration infrastructure (`clt_toolkit/`)

**`replace_schedule` on `SubpopModel` and `MetapopModel`** (`base_components.py`)
- Added `SubpopModel.replace_schedule(schedule_name, new_df)`: swaps a named schedule's `timeseries_df` and re-runs `postprocess_data_input()` so the processed index is consistent with the new data. Copies `new_df` before assigning to avoid mutating the caller's DataFrame.
- Added `MetapopModel.replace_schedule(schedule_name, new_df, subpop_name=None)`: fans out to all subpopulations when `subpop_name` is omitted, or targets a single subpopulation when specified.
- Added `MetapopModel.modify_random_seed(seed)`: re-seeds all subpopulation RNGs from a single integer using `numpy.random.SeedSequence.spawn`, giving each subpop a distinct but reproducible child seed.

**`ScenarioRunner`** (`scenario_runner.py`, new file)
- Runs a baseline `MetapopModel` (or `SubpopModel`) and one or more named counterfactual scenarios as paired multi-replicate experiments, writing all results to a single SQLite database with a `scenario_name` column.
- Scenario definitions support three optional keys: `"schedules"` (apply same DataFrame to all subpops), `"subpop_schedules"` (per-subpop DataFrame overrides), and `"params"` (per-subpop parameter updates via `updated_dataclass`).
- Uses a save/restore pattern (`_save_overrideable_state` / `_restore_overrideable_state`) instead of `deepcopy` to avoid infinite recursion triggered by `SubpopModel.__getattr__` during pickling.
- `seeds` parameter enables paired replicates: `seeds[i]` is used to re-seed the model before replicate `i` in every scenario, isolating the effect of each intervention.
- SQL writes are batched: all rows for a replicate are accumulated in a Python list and flushed with a single `executemany` + `conn.commit()` per replicate, instead of one `executemany` call per (subpop × state variable × timepoint).
- Exported from `clt_toolkit/__init__.py` as `ScenarioRunner` and `ScenarioRunnerError`.

**`seeds` parameter in `Experiment.run_static_inputs`** (`experiments.py`)
- Added optional `seeds` list to `run_static_inputs` and `simulate_reps_and_save_results`: before replicate `i`, `model.modify_random_seed(seeds[i])` is called, giving reproducible and cross-scenario-paired replicates.
- Added module-level docstring to `experiments.py` with typical usage and database schema.

**`FluSubpopModel.reset_simulation()` override** (`flu_core/flu_components.py`)
- Overrides `SubpopModel.reset_simulation()` to recompute `VaxInducedImmunity.adjust_initial_value()` from the current vaccine schedule before restoring compartment values. Uses `MV.original_init_val` (the unmodified JSON baseline) to prevent adjustments from compounding across multiple resets.
- Ensures `MV.init_val` (and `current_val`) are consistent with whatever `daily_vaccines` schedule is loaded at the time of reset, so `replace_schedule + reset_simulation` produces the same trajectory as a freshly constructed model using the same schedule.

## 2026 02 19
- Pre-indexed schedule DataFrames by date (or day-of-week) in all 4 schedule classes in `flu_components.py` to eliminate O(n) boolean scans during `prepare_daily_state`. Changes:
  - `DailyVaccines`, `MobilityModifier`, `AbsoluteHumidity`, `FluContactMatrix`: added/updated `postprocess_data_input()` to call `set_index('date')` (or `set_index('day_of_week')` for the day-of-week mobility variant) at the end of setup, after all other DataFrame processing is complete.
  - All four `update_current_val()` methods updated to use `.loc[current_date]` instead of `df[df["date"] == current_date]`.
  - `FluContactMatrix.update_current_val()`: `except IndexError` changed to `except KeyError` to match `.loc` semantics.
  - `MV.adjust_initial_value()`: mask updated to use `vaccines_df.index` instead of `vaccines_df['date']` since `date` is now the DataFrame index.
  - Profiling result: `prepare_daily_state` cumtime dropped from 0.256s (33%) to 0.082s (13%); total simulation time 0.78s → 0.627s (−20%).
- Updated the travel mixing exposure equations in `flu_travel_functions.py` to properly account for the proportion of individuals staying home in each location.
- In `compute_local_to_local_exposure()`: `proportion_staying_home` is now applied to both sides of the contact matrix multiplication — scaling both the susceptible pool and the infectious pool present in the local location. Previously it was only applied to the susceptible side.
- In `compute_outside_visitors_exposure()`: added the susceptible scaling  `proportion_staying_home` of the *local* (destination) location. 
- In `compute_residents_traveling_exposure()`: the infectious pool at the destination is now correctly computed by aggregating over all subpopulations present at the destination (residents staying home + travelers from other locations) via a vectorized `einsum`. Previously only the local-to-local infectious individuals at the destination were considered, undercounting the infectious pool.

## 2026 01 29
- Added vaccine immunity reset functionality to model seasonal vaccine immunity patterns. A new parameter `vax_immunity_reset_date_mm_dd` is added to `FluSubpopParams` in `flu_data_structures.py`. When set (format: "MM_DD", e.g., "08_01" for August 1st), the vaccine-induced immunity (MV) resets to zero on this date each year to represent the start of a new vaccine season.
- Added `start_real_date` parameter to `FluSubpopParams` in `flu_data_structures.py` to track the real-world date corresponding to the simulation start, enabling date-based reset functionality.
- Modified the `VaxInducedImmunity` class in `flu_components.py` to:
  - Adjust the initial vaccine-induced immunity value at simulation start by accounting only for vaccines administered after the most recent reset date (before simulation start), with appropriate waning applied. This ensures vaccines from previous seasons are not counted. The initial value variable `init_val` saved in the object's instance needs to updated so that even when we the simulation is reset the adjustment is taken into account.
  - Check each day whether the current date matches the reset date and reset MV to zero if it does.
- Added `prepare_daily_state()` method override in `FluSubpopModel` to check for vaccine immunity resets at the beginning of each simulated day.
- Added `check_and_apply_MV_reset()` function in `flu_torch_det_components.py` to support vaccine immunity resets in deterministic simulations.
- Enhanced metapopulation model in `flu_components.py` to properly handle non-numerical parameters (strings and datetime objects) across subpopulations, ensuring consistency.
- The reset functionality works in conjunction with `vax_protection_delay_days` to properly account for the delay between vaccination and effective protection.
- Updated example parameter files ([common_subpop_params.json](flu_instances/austin_input_files/common_subpop_params.json), test files) and notebooks to demonstrate the new reset functionality.

## 2026 01 21
- Added a new parameter `vax_protection_delay_days` to model the delay between vaccine administration and protection effectiveness. The parameter is added to `FluSubpopParams` in `flu_data_structures.py` and used in the `DailyVaccines` class in `flu_components.py`. The vaccine timeseries is shifted forward by the specified number of days, with zero-valued entries backfilled at the beginning to preserve the original start date.

## 2026 01 14
- Modified the variable mobility_modifier to be a schedule that varies through time instead of being a static variable. Input can either be a time series (like vaccines) or depend on the day of the week only.
- In function check_rate_input() in file `flu_components.py` we now let transition rate values be equal to zero and only issue a warning if that is the case. Values still need to be positive (>=0).

## 2025 12 11
Added input checks for subpop and metapop models.
We check that humidity, vaccination, contact matrix, and initial compartment values are non-negative. All values at zero are possible but wouldn't make sense.
For transition rates we need strictly positive values.
For vaccination rates we check whether cumulative vaccination rates in each age-risk group are not exceeding 100% in any 365-day period. This only issues a warning.
The mobility matrix (or travel_proportions) should have rows that sum to 1: this ensures people either travel to another subpopulation or stay in their home location.

A new parameter called use_deterministic_softplus is added to the simulation settings. If the object oriented model is run with deterministic transitions this can be used to prevent softplus values instead of zeros in compartments, which leads to strange behaviors when epidemics occur in populations without any exposure.

Small fixes were made to the travel model equations in the file `flu_travel_functions.py`.

## 2025 11 17 - Adding ghost compartments
Updated website notation and made code updates in a lot of places.


# For future developers (from LP)

Technical notes
- After making changes, please make sure ALL tests in `tests/` folder pass, and add new tests for new code.
- Due to the highly complicated influenza model, there are a lot of input combinations and formats -- errors might arise due to incorrect inputs (e.g. dimensions, missing commas, etc...) -- if there is an error in running the model, the inputs should be checked first. Additionally, more work should be spent on writing input validators and error messages.

Tests to add
- Experiments
  - Make sure aggregating over subpopulation/age/risk is correct (e.g. in `get_state_var_df`).
  - Make sure all the ways to create different CSV files lead to consistent results!
- Accept-reject sampling
  - Reproducibility: running the algorithm twice (with the same RNG each time) should give the same result.
  - Make sure the sampling updates are applied correctly (e.g. to the correct subpopulation(s) and with the correct dimensions).

Features to add
- Would be nice to make the "checker" in `FluMetapopModel` `__init__` method more robust -- can check dimensions, check for nonnegativity, etc...