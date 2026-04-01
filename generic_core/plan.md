# Plan: JSON-Configurable Compartmental Model with Decoupled Travel and Torch Modules

## Context

The flu model structure (compartments, transitions, rate formulas, epi metrics) is
hardcoded across `flu_core/`. To support different disease models (SIR, SEIRV, etc.)
without writing new Python code, the model structure should be JSON-configurable.
Two key modules — the metapopulation travel model and the torch-based differentiable
implementation — are tightly coupled to the flu-specific structure and need to be
decoupled so they can work with any model defined via JSON.

The existing `flu_core/` and `SIHR_core/` code will remain untouched. The new
framework lives in `generic_core/` as a parallel pathway.

---

## Feasibility Assessment: Yes, with caveats

`clt_toolkit/base_components.py` already provides a disease-agnostic simulation
engine (`SubpopModel.simulate_until_day()`, `Compartment`, `TransitionVariable`,
`TransitionVariableGroup`, `EpiMetric`). `SIHR_core/` proves a different model
structure works with these base classes. The main challenges are:

1. Rate formulas range from trivial constants to the complex force-of-infection
2. The torch module fully duplicates the numpy rate logic
3. The travel module hardcodes which compartments are infectious/immobile
4. EpiMetrics and Schedules have bespoke lifecycle logic

---

## Approach: Rate Template Registry

A JSON expression DSL would effectively require reimplementing a programming language
(and breaks for torch dual-use). Instead, use **parameterizable rate templates** —
a registry of Python classes, each implementing both numpy and torch versions of a
rate computation. The JSON config selects which template to use and passes it
parameter names. Complex math stays in Python; model *structure* (compartments,
transitions, wiring) is fully JSON-configurable.

Rate type templates are required for all transitions (no expression evaluator),
for consistency and debuggability. The framework ships with templates covering all
flu model rate patterns; new templates are registered for novel formulas.

---

## Rate Formula Taxonomy

| Tier | Count (flu) | Example | Template |
|------|------------|---------|----------|
| Constant | 6/12 | `IA_to_R_rate` | `constant_param` |
| Param product | — | `E_to_I_rate * E_to_IA_prop` | `param_product` |
| Immunity-modulated | 4/12 | `IP_to_ISR` with `immunity_force` | `immunity_modulated` |
| Force of infection (single-pop) | 1/12 | `S_to_E` with contact matrix | `force_of_infection` |
| Force of infection (travel) | 1/12 | `S_to_E` with travel mixing | `force_of_infection_travel` |

---

## Key Design Decisions

### 1. Rate Templates (not DSL)
Template classes implement `numpy_rate()` and `torch_rate()` in one place.
No `eval()`. Clear stack traces. Type-safe config validation at construction time.
See `architecture.md` §Rate Templates for full interface.

### 2. Travel Module Decoupling
`flu_travel_functions.py` hardcodes which compartments are infectious (IP, IA, ISR,
ISH) and immobile (HR, HD). A new `generic_core/travel_functions.py` accepts config
dicts instead of hardcoded field names. The math is identical; only the data access
pattern changes. `flu_core/` is not modified.

### 3. Torch Module Decoupling
`flu_torch_det_components.py` reimplements all rate logic in torch. In the generic
framework, each `RateTemplate` has a `torch_rate()` method, so formulas are defined
once. A generic `torch_advance_timestep()` iterates over transitions from config and
calls each template's `torch_rate()`.

### 4. Calibration Interface
A generic accept-reject calibration interface analogous to `flu_accept_reject.py`
but operating on any `ConfigDrivenMetapopModel`. The calibration target (which
transitions count as "hospital admits" etc.) is specified by the caller, not the
JSON config.

### 5. Outcome Functions
Outcome functions (hospital admissions, deaths, attack rate) are not specified in
the JSON config. Instead, `generic_core/` will provide generic outcome utilities
that accept transition/compartment names as arguments, paralleling the approach in
`flu_outcomes.py`. Marimo notebooks (`flu_scenario_analysis.py`,
`flu_sensitivity.py`) will be adapted to use the generic model.

### 6. Backward Compatibility
`flu_core/` and `SIHR_core/` are untouched. `generic_core/` is a parallel module.

---

## JSON Config Schema (Conceptual)

```json
{
  "compartments": ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"],
  "epi_metrics": ["M", "MV"],
  "transitions": [
    {
      "name": "R_to_S", "origin": "R", "destination": "S",
      "rate_template": "constant_param",
      "rate_config": {"param_name": "R_to_S_rate"}
    },
    {
      "name": "IP_to_ISR", "origin": "IP", "destination": "ISR",
      "jointly_distributed_with": "IP_to_ISH",
      "rate_template": "immunity_modulated",
      "rate_config": {
        "base_rate": "IP_to_IS_rate",
        "proportion": "IP_to_ISH_prop",
        "is_complement": true,
        "inf_reduce_param": "inf_induced_hosp_risk_reduce",
        "vax_reduce_param": "vax_induced_hosp_risk_reduce"
      }
    },
    {
      "name": "S_to_E", "origin": "S", "destination": "E",
      "rate_template": "force_of_infection_travel",
      "rate_config": {
        "beta_param": "beta_baseline",
        "humidity_param": "humidity_impact",
        "contact_matrix_schedule": "flu_contact_matrix",
        "inf_reduce_param": "inf_induced_inf_risk_reduce",
        "vax_reduce_param": "vax_induced_inf_risk_reduce",
        "infectious_compartments": {
          "ISR": 1.0, "ISH": 1.0,
          "IP": "IP_relative_inf", "IA": "IA_relative_inf"
        },
        "relative_susceptibility_param": "relative_suscept"
      }
    }
  ],
  "transition_groups": [
    {"name": "E_out",  "members": ["E_to_IP",  "E_to_IA"]},
    {"name": "IP_out", "members": ["IP_to_ISR", "IP_to_ISH"]},
    {"name": "ISH_out","members": ["ISH_to_HR", "ISH_to_HD"]}
  ],
  "epi_metrics_config": [
    {
      "name": "M",
      "update_template": "infection_induced_immunity",
      "update_config": {
        "inflow_transition": "R_to_S",
        "inf_saturation_param": "inf_induced_saturation",
        "vax_saturation_param": "vax_induced_saturation",
        "wane_rate_param": "inf_induced_immune_wane"
      }
    },
    {
      "name": "MV",
      "update_template": "vaccine_induced_immunity",
      "update_config": {
        "vaccine_schedule": "daily_vaccines",
        "wane_rate_param": "vax_induced_immune_wane",
        "reset_date_param": "vax_immunity_reset_date_mm_dd",
        "delay_param": "vax_protection_delay_days"
      }
    }
  ],
  "schedules": [
    {"name": "absolute_humidity",  "schedule_template": "timeseries_lookup",
     "column": "absolute_humidity"},
    {"name": "flu_contact_matrix", "schedule_template": "contact_matrix"},
    {"name": "daily_vaccines",     "schedule_template": "vaccine_schedule"},
    {"name": "mobility_modifier",  "schedule_template": "mobility"}
  ],
  "travel": {
    "infectious_compartments": {
      "ISR": 1.0, "ISH": 1.0,
      "IP": "IP_relative_inf", "IA": "IA_relative_inf"
    },
    "immobile_compartments": ["HR", "HD"],
    "contact_matrix_schedule": "flu_contact_matrix",
    "mobility_schedule": "mobility_modifier"
  }
}
```

---

## Open Questions (Resolved)

1. **Expression language vs templates**: Templates required for all rates. No `eval`.
   Consistency and debuggability outweigh convenience for trivial cases.
2. **Outcome functions**: Not in JSON. Generic outcome utilities accept transition/
   compartment names as arguments. Marimo notebooks adapted separately.
3. **Calibration**: Generic accept-reject interface included in `generic_core/`.
4. **Scope**: `flu_core/` kept as-is. `generic_core/` is the new parallel module.

---

## Risk Assessment

| Phase | Risk | Main concern |
|-------|------|-------------|
| Rate templates | Low | Pure addition; no existing code changes |
| Generic data structures + model | Low-Medium | Dict-based access, validation logic |
| Travel decoupling | Medium | Tensor index arithmetic; must be numerically identical |
| Torch decoupling | Medium-High | Deterministic path must be bit-identical for calibration |
| Integration (full flu via JSON) | Medium | Many interacting components |
