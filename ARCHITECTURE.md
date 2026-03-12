# CLT_BaseModel Architecture

## Overview

The **City-Level Transmission (CLT) Toolkit** is a modular Python framework for building compartmental epidemiological models of respiratory virus transmission at the city or regional level. The primary implementation is a detailed influenza model; a simplified SIHR (Susceptible-Infected-Hospitalized-Recovered) model is included as a tutorial example.

The design philosophy is to keep disease-agnostic infrastructure in a generic toolkit (`clt_toolkit/`) and place all disease-specific logic in separate core modules (`flu_core/`, `SIHR_core/`). This makes it straightforward to add new diseases without touching shared machinery.

---

## Directory Structure

```
CLT_BaseModel/
├── clt_toolkit/                    # Generic base classes and utilities
│   ├── base_components.py          # Core abstract classes (SubpopModel, MetapopModel, Compartment, …)
│   ├── base_data_structures.py     # TransitionTypes, SimulationSettings, SubpopState, SubpopParams
│   ├── input_parsers.py            # JSON loading and frozen dataclass construction
│   ├── experiments.py              # Result collection and SQL-backed data management
│   ├── sampling.py                 # Accept-reject (ABC) sampling algorithms
│   ├── plotting.py                 # Visualization utilities
│   └── utils.py                    # Miscellaneous helpers (updated_dataclass, etc.)
│
├── flu_core/                       # Flu-specific model implementation
│   ├── flu_components.py           # All flu TransitionVariable and EpiMetric subclasses
│   ├── flu_data_structures.py      # FluSubpopState, FluSubpopParams, FluSubpopSchedules
│   ├── flu_travel_functions.py     # Travel / metapopulation force-of-infection functions
│   ├── flu_torch_det_components.py # PyTorch deterministic variants (for calibration)
│   └── flu_accept_reject.py        # ABC sampling adapted for the flu model
│
├── SIHR_core/                      # Simplified SIHR tutorial model
│
├── flu_instances/                  # Example model instances and input data
│   ├── austin_input_files/         # Austin metro test case (2 locations, 5 age groups)
│   ├── austin_input_files_2023_2024/
│   ├── austin_input_files_2024_2025/
│   ├── texas_input_files/          # Texas statewide single-location case
│   ├── calibration_research_input_files/
│   ├── derived_inputs_computations/ # Scripts for computing contact matrices
│   └── examples/                   # Jupyter demo notebooks
│
├── SIHR_instances/
├── tests/
│   ├── test_flu_subpop.py
│   ├── test_flu_metapop.py
│   ├── test_flu_travel.py
│   ├── experiments_tests.py
│   └── test_input_files/
│
├── requirements.txt
├── pyproject.toml
├── README.md
├── developer_notes.md
└── ARCHITECTURE.md                 # this file
```

---

## Model Type and Compartmental Structure

The flu model is a **stochastic (or deterministic) compartmental SEIR-variant** with age-risk stratification and optional multi-location travel dynamics.

### Compartments

| Symbol | Name | Description |
|--------|------|-------------|
| S | Susceptible | Not infected; protection modulated by M and MV |
| E | Exposed | Infected, not yet infectious |
| IP | Infected Pre-symptomatic | Infectious, no symptoms yet |
| ISR | Infected Symptomatic → Recover | Will recover without hospitalization |
| ISH | Infected Symptomatic → Hospitalize | Will require hospitalization |
| IA | Infected Asymptomatic | Infectious, never develop symptoms |
| HR | Hospitalized → Recover | In hospital, will survive |
| HD | Hospitalized → Die | In hospital, will die |
| R | Recovered | Immune via infection; feeds back into M metric |
| D | Dead | Removed permanently |

### Epi Metrics (population-level immunity)

| Symbol | Description |
|--------|-------------|
| M | Infection-induced immunity – accumulates from recoveries, wanes over time |
| MV | Vaccine-induced immunity – driven by the daily vaccines schedule, wanes and can reset annually |

Each compartment and metric is indexed by **age group × risk group** (e.g. 5 age groups × 1 risk group in the Austin example), stored as NumPy arrays.

---

## Key Classes

### Generic Toolkit (`clt_toolkit/base_components.py`)

| Class | Role |
|-------|------|
| `StateVariable` | Abstract parent for anything that has a time-varying value (Compartment, EpiMetric, Schedule, DynamicVal) |
| `Compartment` | Tracks inflow, outflow, and current population for one compartment |
| `TransitionVariable` | Models a flow between two compartments (e.g. S→E); subclassed for each disease transition |
| `TransitionVariableGroup` | Jointly samples multiple outflows from one compartment to preserve population conservation |
| `EpiMetric` | Population-level metric updated each timestep (immunity levels) |
| `DynamicVal` | Values recalculated from current simulation state (e.g. alert-level triggers) |
| `Schedule` | Time-indexed external inputs (vaccines, humidity, contact matrices) |
| `SubpopModel` | Owns a single subpopulation's compartments/transitions; runs the simulation loop |
| `MetapopModel` | Coordinates multiple `SubpopModel` instances with inter-location travel |
| `SimulationSettings` | Frozen dataclass: `timesteps_per_day`, `transition_type`, `start_real_date`, history flags |
| `SubpopParams` | Frozen dataclass base for all epidemiological parameters |

### Flu Core (`flu_core/`)

| Class | Role |
|-------|------|
| `FluSubpopState` | Holds current values for all 10 compartments, 2 epi metrics, and schedule dataframes |
| `FluSubpopParams` | ~40 frozen parameters: transmission rates, proportions, contact matrices, immunity scalings |
| `FluSubpopSchedules` | DataFrames for absolute humidity, contact matrix, daily vaccines, mobility modifier |
| `FluSubpopModel` | Main flu simulation class; wires up all compartments, transitions, and metrics |
| `FluMetapopModel` | Multi-location flu model; applies travel-adjusted force of infection across subpopulations |
| `SusceptibleToExposed` | Implements the force-of-infection (most complex transition; handles travel mixing) |
| `InfInducedImmunity` | EpiMetric updating M from recoveries + waning |
| `VaxInducedImmunity` | EpiMetric updating MV from vaccines + waning + annual reset |
| `DailyVaccines` | Schedule wrapping a time-series of daily vaccination counts by age-risk |
| `AbsoluteHumidity` | Schedule wrapping daily humidity values |
| `FluContactMatrix` | Schedule interpolating between school-day, work-day, and home contact matrices |
| `MobilityModifier` | Schedule providing time-varying proportion of people staying home |

---

## Input Files

Each model instance is described by a set of JSON and CSV files:

| File | Contents |
|------|----------|
| `simulation_settings.json` | `timesteps_per_day`, `transition_type`, `start_real_date`, history flags |
| `common_subpop_params.json` | Shared epidemiological parameters (beta, rates, proportions, immunity params, contact matrices) |
| `init_vals_*.json` | Initial compartment and metric values as age×risk arrays |
| `mixing_params.json` | Number of locations and travel proportion matrix (rows sum to 1) |
| `absolute_humidity.csv` | Daily absolute humidity values |
| `contact_matrix.csv` | Daily school/work day indicators for contact matrix interpolation |
| `daily_vaccines.csv` | Daily vaccination counts per age-risk cell (JSON-encoded per row) |
| `mobility_modifier.csv` | Time-varying or day-of-week mobility reduction factors |

---

## Simulation Loop

```
1. Build FluSubpopParams, FluSubpopState, FluSubpopSchedules
2. Instantiate FluSubpopModel (or FluMetapopModel for multi-location)
3. For each simulated day:
   a. prepare_daily_state()         → advance schedules to current_real_date
   b. apply_inter_subpop_updates()  → compute travel-driven exposure across locations
   c. For each timestep within the day (1 … timesteps_per_day):
      i.  update_transition_rates() → recompute per-compartment rates
      ii. sample_transitions()      → draw binomial/Poisson/deterministic flows
      iii.update_epi_metrics()      → update M, MV from flows + waning
      iv. update_compartments()     → apply net flows to all compartments
   d. increment_simulation_day()    → current_real_date += 1 day
   e. save_daily_history()          → (optional) snapshot all state arrays
4. Query results via SubpopModel.compartments[name].history_vals_list
   or via the experiments.py SQL framework
```

---

## Key Implementation Choices

### Stochasticity and Transition Types

Several modes are supported via `SimulationSettings.transition_type`:

| Mode | Description |
|------|-------------|
| `binom` | Stochastic binomial draws — the statistically correct default |
| `binom_deterministic_no_round` | Deterministic binomial means (no rounding); matches PyTorch variant |
| `binom_taylor_approx` | Stochastic with Taylor-series rate approximation |
| `poisson` | Stochastic Poisson draws |
| `poisson_deterministic` | Deterministic Poisson means |

### Immunity Modeling

Both M and MV are **population-level saturation metrics**, not individual-level immunity flags. They reduce per-susceptible transmission, hospitalization, and death rates through multiplicative scaling factors:

- M accumulates from recovered individuals and wanes with rate `inf_induced_immune_wane`
- MV is driven by `DailyVaccines` (with a configurable `vax_protection_delay_days` lag), wanes with `vax_induced_immune_wane`, and can be reset annually (e.g. start of flu season) via `vax_immunity_reset_date_mm_dd`
- Separate immunity-reduction factors are applied for infection risk, hospitalization risk, and death risk

This is an aggregate approximation: it captures herd-immunity-like effects and waning at the population level without tracking individual immunity histories.

### Contact Matrix Dynamics

The model interpolates between a full contact matrix and reduced school/work contact matrices based on daily `is_school_day` and `is_work_day` flags, then scales by a `mobility_modifier`. Matrices are pre-indexed by date and day-of-week at initialization for O(1) lookup during simulation.

### Travel / Metapopulation Model

The force of infection in a multi-location setting decomposes into three terms:

1. **Local-to-local exposure**: residents staying home, infected by others staying home
2. **Residents-traveling exposure**: susceptibles at their travel destinations encountering infectious individuals there
3. **Outside-visitors exposure**: susceptibles at home exposed to infectious visitors

This is computed once per day and passed into `SusceptibleToExposed` for each subpopulation. Hospitalized individuals are excluded from travel.

### PyTorch Deterministic Variant

`flu_torch_det_components.py` provides differentiable versions of the core simulation components. These are used with the `torch` optimizer for parameter calibration (gradient-based fitting), as demonstrated in `torch_calibration_demo.ipynb`.

### Parameter Immutability

`FluSubpopParams` is a frozen dataclass. Modifications for scenario comparisons should use the `updated_dataclass()` utility, which returns a new instance with specified fields changed. This prevents accidental mutation of a baseline configuration.

### Result Management

`experiments.py` provides an SQL-backed framework for batch simulation runs: results from multiple replicates and locations are inserted into a SQLite database and retrieved as Pandas DataFrames aggregated by age, risk, location, and replicate.

---

## Assumptions

- **Closed population**: No births, natural deaths, or migration other than the explicit travel model.
- **Population-level immunity**: M and MV are aggregate metrics, not individual-level tracking. This is an approximation that works at large population sizes but loses individual heterogeneity in immunity history.
- **Homogeneous mixing within age-risk strata**: All individuals in the same age group and risk group are assumed to mix identically.
- **Symmetric contact matrices**: Contact matrices are assumed to be consistent with population demographics (not necessarily symmetric in raw form, but reflecting reciprocal contact patterns).
- **No risk stratification by default**: The Austin example uses 1 risk group. Multi-risk-group simulations are supported but require calibrated risk-stratified parameters.
- **Absolute humidity as seasonality proxy**: Transmission rate β is scaled by absolute humidity; this is a well-established but simplified seasonality mechanism.
- **Vaccine immunity is aggregate**: Vaccines add to the MV pool uniformly; individual-level variation in vaccine response is not modeled.
- **Travel proportions are stationary**: The mixing/travel matrix is constant over the simulation period.

---

## What Is Missing Before Running Vaccination Scenario Analyses

To run scenario analyses that estimate the impact of increased vaccination rates, the following elements need to be in place:

### 1. Validated Baseline Parameters
Before comparing scenarios, the baseline model should be calibrated to observed epidemiological data (e.g. hospitalization curves, seroprevalence). Without calibration, scenario differences are relative to an unvalidated baseline. The `flu_accept_reject.py` and `torch_calibration_demo.ipynb` provide infrastructure for this, but calibrated parameter sets for a specific season and geography need to exist.

### 2. Realistic Vaccine Coverage Schedules
Each scenario requires a `daily_vaccines.csv` that reflects a specific coverage trajectory — both the **baseline** (current coverage) and the **counterfactual** (increased coverage). These schedules need:
- Vaccine uptake curves by age group (older adults typically prioritize early; children lag)
- Plausible administration start dates and rollout speeds
- Geographic availability constraints if modeling a specific city/region

### 3. Age-Stratified Vaccine Efficacy Parameters
The current parameters `vax_induced_saturation`, `vax_induced_immune_wane`, and the immunity reduction factors (`vax_immunity_reduction_*`) control how effective vaccination is. For scenario analyses to be meaningful:
- These should reflect empirically estimated vaccine effectiveness (VE) for the strain and season being modeled
- Age-specific VE differences (e.g. lower effectiveness in elderly) should be encoded in age-stratified parameters

### 4. Defined Initial Conditions for M and MV
Initial values for the immunity metrics M and MV (in `init_vals_*.json`) need to reflect **pre-season immunity levels**, accounting for prior infection history and existing vaccine coverage. Poorly specified initial immunity levels can dwarf the intervention signal.

### 5. Scenario Configuration Infrastructure
Currently there is no high-level "scenario runner" that automates:
- Constructing scenario-specific `FluSubpopParams` and schedule files
- Running multiple stochastic replicates per scenario
- Comparing outcomes (attack rate, hospitalizations, deaths) across scenarios with confidence intervals

This could be built on top of the existing `experiments.py` SQL framework, but the orchestration layer is not yet present.

### 6. Outcome Metrics and Analysis Notebooks
There is no existing notebook or script that computes vaccination scenario comparison outputs such as:
- Vaccine-preventable hospitalizations (VPH) = baseline hospitalizations − counterfactual hospitalizations
- Vaccine-preventable deaths
- Population-level attack rate reduction by age group
- Health-economic estimates (if desired)

These would need to be written on top of the result-querying utilities in `experiments.py`.

### 7. Sensitivity Analysis Over Uncertain Parameters
Key parameters (β_baseline, VE, waning rates, initial immunity) carry uncertainty. A sensitivity or uncertainty quantification analysis over these parameters — ideally using the ABC sampling infrastructure in `sampling.py` and `flu_accept_reject.py` — is needed for the scenario results to be credible.

### 8. Documentation of Parameter Sources
For peer review or policy use, each parameter value (especially VE estimates and age-risk proportions like `IP_to_ISH_prop`, `ISH_to_HD_prop`) should be traceable to a literature or surveillance source. This documentation does not currently exist in a structured form.
