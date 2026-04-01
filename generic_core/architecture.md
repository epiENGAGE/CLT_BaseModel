# Architecture: generic_core

This document specifies the files that make up `generic_core/`, their
responsibilities, and the core design decisions that govern how they fit together.

---

## File Map

```
generic_core/
├── __init__.py
├── plan.md                  ← overall goals and approach
├── architecture.md          ← this file
├── tasks.md                 ← implementation phases
│
├── rate_templates.py        ← RateTemplate ABC + all concrete rate classes
├── metric_templates.py      ← MetricTemplate ABC + epi metric update classes
├── schedule_templates.py    ← ScheduleTemplate ABC + schedule type classes
│
├── data_structures.py       ← GenericSubpopState, GenericSubpopParams,
│                               GenericTravelTensors
├── config_parser.py         ← JSON loading, schema validation, registry lookup
│
├── generic_model.py         ← ConfigDrivenTransitionVariable,
│                               ConfigDrivenEpiMetric,
│                               ConfigDrivenSubpopModel
├── generic_metapop.py       ← ConfigDrivenMetapopModel
│
├── travel_functions.py      ← Generic travel/mixing functions (config-driven,
│                               not tied to flu compartment names)
├── torch_generic.py         ← generic_advance_timestep,
│                               generic_torch_simulate_full_history,
│                               generic_torch_simulate_calibration_target
│
├── calibration.py           ← Generic accept-reject calibration interface
└── outcomes.py              ← Generic outcome utilities (transition/compartment
                                name-based, not hardcoded)
```

### Relationship to existing code

| Existing file | Relationship to generic_core |
|---------------|------------------------------|
| `clt_toolkit/base_components.py` | `generic_core` subclasses from here: `ConfigDrivenSubpopModel` extends `SubpopModel`, `ConfigDrivenTransitionVariable` extends `TransitionVariable`, etc. No changes to this file. |
| `clt_toolkit/base_data_structures.py` | `GenericSubpopState` extends `SubpopState`. `GenericSubpopParams` extends `SubpopParams`. No changes to this file. |
| `flu_core/flu_components.py` | Untouched. Rate formulas here are the ground-truth reference for what templates must reproduce. |
| `flu_core/flu_travel_functions.py` | Untouched. `generic_core/travel_functions.py` is a config-driven reimplementation of the same math. |
| `flu_core/flu_torch_det_components.py` | Untouched. `generic_core/torch_generic.py` is a config-driven replacement. |
| `flu_core/flu_accept_reject.py` | Untouched. `generic_core/calibration.py` is a generalisation. |
| `SIHR_core/` | Untouched. Used as validation target. |

---

## Design Decisions

### 1. Rate Templates — not an expression DSL

**Decision**: Every transition rate is implemented as a named Python template class.
No `eval()` or string expression evaluator.

**Rationale**:
- Rate formulas range from trivial scalars to matrix multiplications + immunity
  dynamics + travel mixing. A single expression language cannot cover the full range
  without reimplementing a programming language.
- The torch dual-use requirement breaks any numpy-specific formula string: even
  `eval("np.matmul(a, b)")` cannot be transparently run through torch.
- Template classes give debuggable stack traces, type-safe config validation at
  construction time, and a single place for each formula.

**Interface** (`rate_templates.py`):

```python
class RateTemplate(ABC):
    @abstractmethod
    def validate_config(self, rate_config: dict, param_names: set,
                        compartment_names: set, schedule_names: set): ...

    @abstractmethod
    def numpy_rate(self, state: GenericSubpopState,
                   params: GenericSubpopParams, rate_config: dict) -> np.ndarray:
        """Returns (A, R) numpy array."""

    @abstractmethod
    def torch_rate(self, state_dict: dict, params_dict: dict,
                   rate_config: dict) -> torch.Tensor:
        """Returns (A, R) or (L, A, R) torch tensor."""
```

**Built-in templates**:

| Template name | Formula | Used for |
|---------------|---------|----------|
| `constant_param` | `params[name]` broadcast to (A,R) | R→S, ISR→R, IA→R, HR→R, HD→D |
| `param_product` | `params[a] * params[b] * (1 - params[c]) ...` | E→IP, E→IA |
| `immunity_modulated` | `base_rate * (prop / immunity_force)` or complement | IP→ISR, IP→ISH, ISH→HR, ISH→HD |
| `force_of_infection` | Beta × humidity × contact_matrix × infectious_prop / immune_force | S→E (single-pop) |
| `force_of_infection_travel` | As above but uses `compute_total_mixing_exposure` | S→E (metapop) |

**Registry**: A module-level dict `RATE_TEMPLATE_REGISTRY: dict[str, RateTemplate]`
maps string names to instances. Users call `register_rate_template(name, instance)`
to add custom templates.

---

### 2. Metric Templates

**Interface** (`metric_templates.py`):

```python
class MetricTemplate(ABC):
    @abstractmethod
    def validate_config(self, update_config, param_names, transition_names): ...

    @abstractmethod
    def build_metric(self, init_val, update_config, params,
                     transition_variables, schedules, timesteps_per_day) -> EpiMetric:
        """Returns a concrete EpiMetric instance."""
```

The `build_metric` factory pattern lets each template construct an EpiMetric
subclass with whatever extra constructor arguments it needs (e.g.,
`VaxInducedImmunity` needs `current_real_date`, `schedules`, `timesteps_per_day`).

**Built-in templates**:

| Template name | Concrete class created | Notes |
|---------------|----------------------|-------|
| `infection_induced_immunity` | `InfInducedImmunityGeneric` | Needs reference to `R_to_S` transition variable |
| `vaccine_induced_immunity` | `VaxInducedImmunityGeneric` | Needs schedule + delay params + reset date |

---

### 3. Schedule Templates

**Interface** (`schedule_templates.py`):

```python
class ScheduleTemplate(ABC):
    @abstractmethod
    def build_schedule(self, schedule_config: dict,
                       params: GenericSubpopParams,
                       schedules_input: FluSubpopSchedules) -> Schedule:
        """Returns a concrete Schedule instance."""
```

**Built-in templates**:

| Template name | Behaviour |
|---------------|-----------|
| `timeseries_lookup` | Simple date-indexed lookup from a DataFrame column |
| `contact_matrix` | Computes total − school×(1−is_school) − work×(1−is_work) |
| `vaccine_schedule` | Delay shift + backfill + O(1) date lookup |
| `mobility` | Date-based or day-of-week lookup |

---

### 4. Generic Data Structures

**Decision**: Use dict-based containers rather than typed dataclasses.

**Rationale**: Named dataclass fields cannot be determined at module import time if
the compartment names come from a JSON config. Dict-based containers support any
model structure with the same code. The tradeoff (no static type checking on fields)
is acceptable because validation happens at construction time in `config_parser.py`.

```python
# data_structures.py

@dataclass
class GenericSubpopState(SubpopState):
    compartments: dict[str, np.ndarray]   # {"S": array(A,R), ...}
    epi_metrics: dict[str, np.ndarray]    # {"M": array(A,R), ...}
    schedules: dict[str, Any]             # current schedule values
    dynamic_vals: dict[str, Any]

@dataclass
class GenericSubpopParams(SubpopParams):
    params: dict[str, Any]               # {"beta_baseline": 0.3, ...}
    num_age_groups: int
    num_risk_groups: int
    total_pop_age_risk: np.ndarray

@dataclass
class GenericTravelTensors:
    """
    Dict-of-tensors container used by generic travel functions.
    Replaces flu-specific FluTravelStateTensors / FluFullMetapopStateTensors.
    """
    compartment_tensors: dict[str, torch.Tensor]  # {"S": (L,A,R), ...}
    schedule_tensors: dict[str, torch.Tensor]     # {"mobility_modifier": (L,A,R), ...}
    param_tensors: dict[str, torch.Tensor]        # {"beta_baseline": (L,A,R), ...}
```

---

### 5. Config Parser and Validation

**File**: `config_parser.py`

All validation happens at construction time before any simulation objects are
created. Errors are raised with clear messages pointing to the offending config key.

**Validation checks**:
- Every transition `origin` and `destination` is a declared compartment
- Every transition group member references a declared transition
- Every `rate_template` name exists in `RATE_TEMPLATE_REGISTRY`
- Each rate template's `validate_config()` is called (checks param names exist, etc.)
- Every epi metric `update_template` is in `METRIC_TEMPLATE_REGISTRY`
- Every schedule `schedule_template` is in `SCHEDULE_TEMPLATE_REGISTRY`
- Joint distribution groups: each member's `jointly_distributed_with` is consistent
- Travel config: `infectious_compartments` and `immobile_compartments` reference
  declared compartments

**Output**: A validated, parsed `ModelConfig` dataclass consumed by
`ConfigDrivenSubpopModel`.

---

### 6. ConfigDrivenSubpopModel

**File**: `generic_model.py`

Extends `clt.SubpopModel`. Implements the 7 abstract factory methods by
iterating over the validated `ModelConfig`.

```python
class ConfigDrivenTransitionVariable(clt.TransitionVariable):
    def __init__(self, origin, destination, transition_type,
                 rate_template: RateTemplate, rate_config: dict, ...):
        super().__init__(origin, destination, transition_type, ...)
        self.rate_template = rate_template
        self.rate_config = rate_config

    def get_current_rate(self, state, params) -> np.ndarray:
        return self.rate_template.numpy_rate(state, params, self.rate_config)

class ConfigDrivenSubpopModel(clt.SubpopModel):
    def __init__(self, model_config: ModelConfig, state: GenericSubpopState,
                 params: GenericSubpopParams, settings: SimulationSettings,
                 RNG, schedules_input, name: str): ...

    def create_compartments(self): ...       # loops over config.compartments
    def create_transition_variables(self):   # loops over config.transitions
    def create_transition_variable_groups(self): ...
    def create_epi_metrics(self): ...
    def create_schedules(self): ...
    def create_dynamic_vals(self): ...       # returns empty dict by default
    def run_input_checks(self): ...
```

---

### 7. Travel Functions — Config-Driven

**File**: `travel_functions.py`

A reimplementation of `flu_core/flu_travel_functions.py` where compartment names
are passed as config rather than hardcoded. The mathematical operations are
identical. Field access changes from `state.IP` to `state_tensors["IP"]`.

**Key function signatures** (compare to existing):

```python
# OLD (flu_core):
def compute_wtd_infectious_LA(state: FluTravelStateTensors,
                               params: FluTravelParamsTensors) -> torch.Tensor

# NEW (generic_core):
def compute_wtd_infectious_LA(compartment_tensors: dict[str, torch.Tensor],
                               param_tensors: dict[str, torch.Tensor],
                               infectious_config: dict) -> torch.Tensor
# infectious_config = {"ISR": 1.0, "ISH": 1.0, "IP": "IP_relative_inf", ...}

# OLD:
def compute_active_pop_LAR(state, _params, precomputed) -> torch.Tensor

# NEW:
def compute_active_pop_LAR(compartment_tensors: dict[str, torch.Tensor],
                            immobile_compartments: list[str],
                            precomputed: FluPrecomputedTensors) -> torch.Tensor
```

All other travel functions (`compute_effective_pop_LA`,
`compute_wtd_infectious_ratio_LLA`, `compute_local_to_local_exposure`,
`compute_outside_visitors_exposure`, `compute_residents_traveling_exposure`,
`compute_total_mixing_exposure`) follow the same pattern: replace struct-field
access with dict lookup, keep math identical.

`FluPrecomputedTensors` (nonlocal travel prop, sum_residents, total pop) is still
used — it is computed from the population totals and travel matrix which are
model-structure-independent.

---

### 8. Generic Torch Module

**File**: `torch_generic.py`

A config-driven reimplementation of the core simulation loop from
`flu_torch_det_components.py`. The state is a `dict[str, torch.Tensor]` instead of
`FluFullMetapopStateTensors`.

```python
def generic_advance_timestep(
    state_dict: dict[str, torch.Tensor],     # {"S": (L,A,R), ...}
    params_dict: dict[str, torch.Tensor],
    schedules_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
    rate_templates: dict[str, RateTemplate],
    precomputed: FluPrecomputedTensors,
    dt: float,
    save_calibration_targets: bool = False,
    save_tvar_history: bool = False
) -> tuple[dict, dict, dict]: ...
```

**Transition group handling in torch**: The deterministic multinomial pattern from
`advance_timestep` in `flu_torch_det_components.py` (compute total rate, sample
total, split proportionally) is reproduced mechanically from `config.transition_groups`.

**Calibration targets**: Since the JSON config does not specify which transitions are
calibration targets (Q2 decision), the caller passes a list of transition names to
`save_calibration_targets` in `generic_torch_simulate_calibration_target()`.

---

### 9. Generic Calibration Interface

**File**: `calibration.py`

Generalises `flu_accept_reject.py` to work with any `ConfigDrivenMetapopModel`
(or `ConfigDrivenSubpopModel`).

```python
def generic_accept_reject(
    model,                             # ConfigDrivenMetapopModel
    sampling_RNG: np.random.Generator,
    sampling_info: dict,               # same spec as in flu_accept_reject
    target_timeseries: list[np.ndarray],
    calibration_target_fn: Callable,   # extracts scalar from model state (e.g. daily hospital admits)
    num_days: int,
    target_accepted_reps: int,
    max_reps: int,
    early_stop_percent: float,
    target_rsquared: float
): ...
```

The `calibration_target_fn` is user-supplied and knows which transitions to sum for
the target metric (e.g. `ISH_to_HR + ISH_to_HD`). This keeps the calibration
interface model-agnostic.

---

### 10. Generic Outcome Utilities

**File**: `outcomes.py`

Analogous to `flu_core/flu_outcomes.py` but no hardcoded transition/compartment names.

```python
def daily_transition_sum(history: dict, transition_names: list[str]) -> np.ndarray: ...
def cumulative_compartment(history: dict, compartment_name: str) -> np.ndarray: ...
def attack_rate(history: dict, infection_transition: str,
                initial_susceptible: np.ndarray) -> np.ndarray: ...
def summarize_outcomes(outcomes_list: list[np.ndarray]) -> dict: ...
```

Users of the generic model call these with explicit names:
```python
hospital_admits = daily_transition_sum(history, ["ISH_to_HR", "ISH_to_HD"])
new_infections  = daily_transition_sum(history, ["S_to_E"])
```

---

## Invariants and Contracts

1. **Template registry is populated before `config_parser.parse()` is called.**
   All default templates are auto-registered on import of `rate_templates.py`.

2. **`GenericSubpopState.compartments` keys match `ModelConfig.compartments` exactly.**
   Enforced by `config_parser.py` at construction time.

3. **`RateTemplate.numpy_rate()` and `torch_rate()` must produce numerically
   identical results** for the same inputs (allowing float32/float64 differences).
   Verified in unit tests for each template.

4. **`generic_core/travel_functions.py` must produce bit-identical results to
   `flu_core/flu_travel_functions.py`** when given the same inputs in the flu
   model configuration. Verified by integration test.

5. **`generic_torch_simulate_*` deterministic path must produce bit-identical
   results to `flu_core/flu_torch_det_components.py`** for identical configs.
   Verified before use in gradient-based calibration.

6. **No modifications to `flu_core/`, `SIHR_core/`, or `clt_toolkit/`.**
   `generic_core/` depends on `clt_toolkit/` only via subclassing.
