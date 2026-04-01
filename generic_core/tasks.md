# Tasks: generic_core Implementation

Implementation is divided into 5 phases. Each phase is independently testable
against existing `flu_core/` and `SIHR_core/` output. See `architecture.md` for
file responsibilities, interfaces, and invariants.

---

## Phase 1 — Rate Template Infrastructure

**Goal**: Implement all rate templates with numpy and torch paths. No changes to
existing code. All templates verified against existing `get_current_rate()`
implementations.

### Tasks

- [ ] **1.1** Create `generic_core/rate_templates.py`
  - Define `RateTemplate` ABC with `validate_config()`, `numpy_rate()`, `torch_rate()`
  - Define module-level `RATE_TEMPLATE_REGISTRY: dict[str, RateTemplate]`
  - Define `register_rate_template(name, instance)` helper

- [ ] **1.2** Implement `ConstantParamRate`
  - `numpy_rate`: `np.full((A, R), params.params[name])`
  - `torch_rate`: `params_dict[name].expand(L, A, R)` (or `(A, R)` for single-pop)
  - Reference: `RecoveredToSusceptible`, `SympRecoverToRecovered`, `AsympToRecovered`,
    `HospRecoverToRecovered`, `HospDeadToDead` in
    `flu_core/flu_components.py:134–404`

- [ ] **1.3** Implement `ParamProductRate`
  - `rate_config`: `{"factors": ["E_to_I_rate"], "complement_factors": ["E_to_IA_prop"]}`
    → `rate = prod(params[f]) * prod(1 - params[c])`
  - Handles `ExposedToAsymp` and `ExposedToPresymp` patterns
  - Reference: `flu_components.py:152–194`

- [ ] **1.4** Implement `ImmunityModulatedRate`
  - `rate_config`: `{base_rate, proportion, is_complement, inf_reduce_param, vax_reduce_param}`
  - Computes `immunity_force = 1 + (r/(1-r)) * M + (r/(1-r)) * MV`
  - `is_complement=True` → `rate = base_rate * (1 - prop / immunity_force)`
  - `is_complement=False` → `rate = base_rate * (prop / immunity_force)`
  - State access: `state.epi_metrics["M"]` and `state.epi_metrics["MV"]` (numpy);
    `state_dict["M"]`, `state_dict["MV"]` (torch)
  - Reference: `PresympToSympRecover`, `PresympToSympHospital`,
    `SympHospitalToHospRecover`, `SympHospitalToHospDead` in
    `flu_components.py:197–346`

- [ ] **1.5** Implement `ForceOfInfectionRate` (single-population)
  - Reproduces the `else` branch of `SusceptibleToExposed.get_current_rate()`
    (`flu_components.py:119–131`)
  - `rate_config`: `{beta_param, humidity_param, contact_matrix_schedule,
    inf_reduce_param, vax_reduce_param, infectious_compartments,
    relative_susceptibility_param}`
  - `numpy_rate`: contact matrix matmul + beta × humidity + immunity force
  - `torch_rate`: same ops with torch (matches `compute_S_to_E` in
    `flu_torch_det_components.py:102–136`)

- [ ] **1.6** Implement `ForceOfInfectionTravelRate` (metapopulation)
  - Reproduces the `if self.total_mixing_exposure is not None` branch
    (`flu_components.py:109–117`)
  - Calls `generic_core/travel_functions.compute_total_mixing_exposure` (Phase 3)
  - For now, can raise `NotImplementedError` — fully wired in Phase 3/4

- [ ] **1.7** Write unit tests for Phase 1
  - For each template: construct a minimal state/params from the Austin 2024-2025
    instance, call `numpy_rate()`, compare to `get_current_rate()` of the
    corresponding flu class (tolerance: machine epsilon for float64)
  - For each template: `numpy_rate()` and `torch_rate()` produce same values
    (tolerance: float32 precision)

---

## Phase 2 — Generic Data Structures, Config Parser, and ConfigDrivenSubpopModel

**Goal**: Load a JSON config and run a single-population simulation. Validate by
reproducing SIHR model trajectories.

### Tasks

- [ ] **2.1** Create `generic_core/data_structures.py`
  - `GenericSubpopState(SubpopState)`: `compartments`, `epi_metrics`,
    `schedules`, `dynamic_vals` as dicts
  - `GenericSubpopParams(SubpopParams)`: `params` dict, `num_age_groups`,
    `num_risk_groups`, `total_pop_age_risk`
  - Implement `sync_to_current_vals()` to iterate over the `compartments` dict
    (adapting `SubpopState.sync_to_current_vals()` in
    `clt_toolkit/base_data_structures.py`)
  - `GenericTravelTensors`: `compartment_tensors`, `schedule_tensors`,
    `param_tensors` as dicts (see `architecture.md §4`)

- [ ] **2.2** Create `generic_core/metric_templates.py`
  - `MetricTemplate` ABC with `validate_config()` and `build_metric()`
  - `InfectionInducedImmunityTemplate`: factory for a `EpiMetric` subclass that
    references the `R_to_S` transition variable by name — passes
    `transition_variables["R_to_S"]` to the metric at construction time
    (mirrors `InfInducedImmunity.__init__` in `flu_components.py:430–432`)
  - `VaccineInducedImmunityTemplate`: factory for a `EpiMetric` subclass with
    `adjust_initial_value()`, `get_change_in_current_val()`, and
    `check_and_apply_reset()` logic from `VaxInducedImmunity`
    (`flu_components.py:454–566`)

- [ ] **2.3** Create `generic_core/schedule_templates.py`
  - `ScheduleTemplate` ABC
  - `TimeseriesLookupSchedule`: wraps a DataFrame column, O(1) date lookup
  - `ContactMatrixSchedule`: reproduces `FluContactMatrix` logic
    (`flu_components.py` — school/work day interpolation from three base matrices)
  - `VaccineScheduleTemplate`: reproduces `DailyVaccines` with delay and backfill
    (`flu_components.py:596–669`)
  - `MobilityScheduleTemplate`: reproduces `MobilityModifier` date/day-of-week logic
    (`flu_components.py:672+`)

- [ ] **2.4** Create `generic_core/config_parser.py`
  - `ModelConfig` dataclass (parsed/validated form of the JSON)
  - `parse_model_config(json_path, registry=None) -> ModelConfig`
  - Validation checks (see `architecture.md §5`)
  - Call each template's `validate_config()` with the available param/compartment
    names

- [ ] **2.5** Create `generic_core/generic_model.py`
  - `ConfigDrivenTransitionVariable(clt.TransitionVariable)`:
    `get_current_rate()` delegates to `rate_template.numpy_rate()`
  - `ConfigDrivenEpiMetric(clt.EpiMetric)`: constructed via metric template factory
  - `ConfigDrivenSubpopModel(clt.SubpopModel)`: implements all 7 factory methods
    by iterating over `ModelConfig`
  - `prepare_daily_state()`: calls `check_and_apply_reset()` on vaccine metric if
    present (analogous to `FluSubpopModel.prepare_daily_state()`)

- [ ] **2.6** Write SIHR validation test
  - Create a JSON config that replicates `SIHR_core/`
  - Run `ConfigDrivenSubpopModel` and `SIHRSubpopModel` with the same initial
    conditions, same RNG seed, same transition type
  - Assert compartment trajectories are identical

- [ ] **2.7** Write single-population flu validation test (no travel)
  - Create a JSON config for the full flu model
  - Run `ConfigDrivenSubpopModel` alongside `FluSubpopModel` using Austin data
  - Assert trajectories identical (deterministic transition type)

---

## Phase 3 — Decouple Travel Module

**Goal**: Implement config-driven travel functions and `ConfigDrivenMetapopModel`.
Travel results must be numerically identical to `flu_core/flu_travel_functions.py`.

### Tasks

- [ ] **3.1** Create `generic_core/travel_functions.py`
  - Port all functions from `flu_core/flu_travel_functions.py` with field-access
    changed to dict-lookup (see `architecture.md §7` for signature changes):
    - `compute_wtd_infectious_LA(compartment_tensors, param_tensors, infectious_config)`
    - `compute_active_pop_LAR(compartment_tensors, immobile_compartments, precomputed)`
    - `compute_effective_pop_LA(compartment_tensors, param_tensors, precomputed, travel_config)`
    - `compute_wtd_infectious_ratio_LLA(...)`
    - `compute_local_to_local_exposure(...)`
    - `compute_outside_visitors_exposure(...)`
    - `compute_residents_traveling_exposure(...)`
    - `compute_total_mixing_exposure(compartment_tensors, param_tensors, precomputed, travel_config)`
  - `travel_config` carries `infectious_compartments` dict and
    `immobile_compartments` list from the JSON config

- [ ] **3.2** Wire `ForceOfInfectionTravelRate` (Phase 1.6)
  - Replace `NotImplementedError` with a call to
    `generic_core/travel_functions.compute_total_mixing_exposure`
  - The travel config comes from `rate_config["travel_config"]` set at
    `ConfigDrivenSubpopModel` construction time

- [ ] **3.3** Create `generic_core/generic_metapop.py`
  - `ConfigDrivenMetapopModel(clt.MetapopModel)` analogous to `FluMetapopModel`
    (`flu_components.py`)
  - `update_state_tensors()`: builds `GenericTravelTensors` from subpop states
    using compartment names from config
  - `apply_inter_subpop_updates()`: calls `compute_total_mixing_exposure` and
    sets the `total_mixing_exposure` attribute on each subpop's `S_to_E`
    transition variable
  - `compute_total_pop_LAR_tensor()`: sums initial populations across all
    compartments (model-structure-agnostic)

- [ ] **3.4** Write travel validation test
  - Run `ConfigDrivenMetapopModel` on Austin 2-population data (deterministic)
  - Run `FluMetapopModel` on same data
  - Assert all compartment trajectories and hospital admit timeseries are identical
  - **This is the highest-risk test**: any dict lookup bug will show up here

---

## Phase 4 — Decouple Torch Module

**Goal**: Implement generic torch simulation loop using rate templates.
Deterministic path must be bit-identical to `flu_torch_det_components.py`.

### Tasks

- [ ] **4.1** Create `generic_core/torch_generic.py` — state utilities
  - `build_state_dict_from_subpop(subpop_model, config) -> dict[str, torch.Tensor]`
    extracts compartment and metric tensors from a `ConfigDrivenMetapopModel`
  - `build_params_dict(params, config) -> dict[str, torch.Tensor]`
  - `update_state_dict_with_schedules(state_dict, schedules, day_counter, config)`
    mirrors `update_state_with_schedules` in `flu_torch_det_components.py:383–427`
  - `check_and_apply_MV_reset(state_dict, params_dict, config, day_counter)`

- [ ] **4.2** Implement `generic_advance_timestep()`
  - Signature: `(state_dict, params_dict, schedules_dict, model_config,
    rate_templates, precomputed, dt, save_calibration_targets, save_tvar_history)`
    → `(new_state_dict, calibration_targets, transition_vars)`
  - For each transition: call `rate_template.torch_rate()`, apply
    `torch_approx_binom_probability_from_rate()`
  - For each transition group: compute total rate, sample total, split
    proportionally (deterministic multinomial — mirrors
    `flu_torch_det_components.py:471–503`)
  - Update each compartment: `softplus(val + sum(inflows) - sum(outflows))`
  - Update epi metrics: call metric template's `torch_update()` method

- [ ] **4.3** Add `torch_update()` to metric templates
  - `InfectionInducedImmunityTemplate.torch_update()`: mirrors
    `compute_M_change` in `flu_torch_det_components.py:328–345`
  - `VaccineInducedImmunityTemplate.torch_update()`: mirrors
    `compute_MV_change` in `flu_torch_det_components.py:348–360`

- [ ] **4.4** Implement `generic_torch_simulate_full_history()`
  - Mirrors `torch_simulate_full_history` in `flu_torch_det_components.py:572–621`
  - Returns `(state_history_dict, tvar_history_dict)` keyed by compartment/
    transition name

- [ ] **4.5** Implement `generic_torch_simulate_calibration_target()`
  - Mirrors `torch_simulate_hospital_admits` in
    `flu_torch_det_components.py:624–652`
  - Accepts `calibration_transition_names: list[str]` instead of hardcoding
    `ISH_to_HR + ISH_to_HD`

- [ ] **4.6** Write torch validation tests
  - **Deterministic path test**: run `generic_torch_simulate_full_history` and
    `torch_simulate_full_history` with Austin flu config; assert all compartment
    tensors are bit-identical
  - **Gradient flow test**: run `generic_torch_simulate_calibration_target`,
    compute a scalar loss, call `.backward()`, assert gradients exist on
    `beta_baseline` and `IP_to_ISH_prop`

---

## Phase 5 — Calibration, Outcomes, and Integration

**Goal**: Generic calibration interface, outcome utilities, and full end-to-end
validation including a marimo notebook equivalent.

### Tasks

- [ ] **5.1** Create `generic_core/calibration.py`
  - `generic_accept_reject(model, sampling_RNG, sampling_info, target_timeseries,
    calibration_target_fn, num_days, target_accepted_reps, max_reps,
    early_stop_percent, target_rsquared)`
  - `calibration_target_fn(model) -> np.ndarray`: extracts daily target metric
    from simulation state (caller-supplied; model-agnostic)
  - Early stopping logic identical to `flu_accept_reject.accept_reject_admits`
    (`flu_accept_reject.py:80+`)
  - Save accepted params and states to JSON (same format as existing)

- [ ] **5.2** Create `generic_core/outcomes.py`
  - `daily_transition_sum(tvar_history, names)`: sum over named transitions per day
  - `cumulative_compartment(state_history, name)`: running sum of compartment
  - `attack_rate(tvar_history, infection_transition, initial_susceptible)`
  - `summarize_outcomes(outcomes_list)`: mean, median, 95% CI across replicates
  - These mirror `flu_core/flu_outcomes.py` but accept string names instead of
    hardcoded transition references

- [ ] **5.3** End-to-end flu integration test
  - Load Austin 2024-2025 data, construct `ConfigDrivenMetapopModel` from the
    full flu JSON config
  - Run stochastic simulation (N=5 replicates) with same seeds as `FluMetapopModel`
  - Assert hospital admission timeseries within sampling noise of existing model
  - Run accept-reject calibration for 10 iterations; check it accepts/rejects
    correctly and saves JSON output

- [ ] **5.4** Create `generic_core/__init__.py`
  - Export: `ConfigDrivenSubpopModel`, `ConfigDrivenMetapopModel`,
    `parse_model_config`, `generic_accept_reject`, outcome utilities,
    `register_rate_template`, `register_metric_template`,
    `register_schedule_template`

- [ ] **5.5** Write a minimal SIR example config + demo
  - `generic_core/examples/sir_config.json`: 3 compartments, 2 transitions
    (both `constant_param`), no epi metrics, no schedules
  - Short script that loads config, runs simulation, plots S/I/R
  - This serves as the "minimal working example" for new users

---

## Cross-Phase Notes

**Testing strategy**: Each phase adds tests that run `generic_core` alongside the
corresponding existing implementation and compare outputs. Tests are the primary
correctness mechanism — there is no independent mathematical verification.

**Invariant to check in every phase**: `flu_core/` tests still pass. Since no
`flu_core/` files are modified, this should hold automatically, but it should be
verified after each phase.

**Numerical tolerance**:
- Numpy comparisons to existing code: machine epsilon (float64)
- Numpy vs torch comparisons within generic_core: float32 precision (~1e-6)
- Torch vs existing torch: bit-identical (same dtype, same operations)
