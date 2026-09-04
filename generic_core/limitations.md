# Limitations: generic_core

Known constraints of the current implementation — behaviours that are *by
design* or *not yet built*, as opposed to bugs. The intent is that anything
surprising a user hits at run time should already be written down here, with
the reason and the workaround.

See `architecture.md` for design, `tasks.md` for the phased implementation
plan, `plan.md` for overall goals.

**Adding an entry**: give it (1) what the limitation is, (2) why it exists —
design decision vs. unbuilt, (3) what actually happens if you hit it, (4) the
workaround if any, (5) where in the code. Prefer stating the observed
behaviour over the intent; verify before writing.

---

## Scheduled exact transfers (`scheduled_exact`)

The engine supports **any number** of independent `scheduled_exact`
transitions, each backed by its own named schedule. `ScheduledTransferVariable`
is keyed by `schedule_name`, `create_transition_variables` builds one per
transition, and both the numpy and torch backends resolve each transition's
schedule independently. The limitations below are about the layers *around*
that.

### Competing `scheduled_exact` outflows from the same compartment

**What.** Two or more `scheduled_exact` transitions leaving the *same* origin
compartment are not coordinated with each other.

**Why.** Each `ScheduledTransferVariable` computes its transfer against the
origin's full current value and clamps only itself
(`generic_model.py`, `get_scheduled_exact_realization`):

```python
scheduled_count = np.rint(rate * (origin_val + destination_val))
return np.minimum(scheduled_count, origin_val)
```

Each individually respects the origin; their *sum* need not.

**What happens.** `clt_toolkit.base_components.enforce_outflow_capacity`
detects the over-draw and scales every outflow from that compartment by
`origin / total`. Population is conserved exactly and no compartment goes
negative — but neither schedule moves the count it asked for. Measured, day 1,
`S = 1000`, vaccination at 0.9/day and antivirals at 0.3/day:

| schedule | asked for | actually moved |
|---|---|---|
| vaccination | 900 | 750 |
| antiviral | 300 | 250 |

(combined ask 1200 > 1000, both scaled by 0.8333)

The toolkit also emits a `RuntimeWarning` advising that competing transitions
be declared in a **transition group** — which `config_parser.py` explicitly
*rejects* for `scheduled_exact` transitions ("it is a deterministic, exact
flow, not a competing stochastic branch"). That advice is therefore a dead end
for this case; the warning text is generic to all over-draw causes.

**Workaround.** Keep the combined daily proportions out of any shared origin
below 1.0. The rescale only engages on days the origin is nearly exhausted, so
models with modest daily proportions never trip it. The Model Builder now
raises a config warning at build time when it sees two `scheduled_exact`
outflows from one compartment.

**If this needs fixing properly**, it is a design decision, not a patch: the
engine would need a joint allocation rule for scheduled flows out of a shared
origin (e.g. allocate proportionally to requested counts, computed once per
origin per day) — deciding the intended semantics comes first.

### Dose multipliers are per-schedule (no longer a limitation)

Scaling two `scheduled_exact` schedules by *different* amounts is supported
end to end: the Analysis tab renders one multiplier grid per schedule, the
scenario carries a `{df_attribute: per-age-group vector}` dict, and the
generated script's `DOSE_MULTIPLIER` accepts either a bare list (same scaling
for every schedule) or a per-schedule dict. A schedule omitted from the dict is
left unscaled. Kept here as a pointer because the shape is easy to
misremember — see `_analysis_dose_mult_controls` / `_split_dose_mult` in
`_nb_analysis.py`, `_dose_mult_for` in the `_nb_export.py` codegen, and
`extra_dose_mult` / `extra_dose_mult_per_subpop` in `model_factory.py`.

Single-schedule configs still emit the historical bare-list `DOSE_MULTIPLIER`
format, so existing exported scripts are unaffected.

### Additional schedules are CSV-only, with no transfer delay

**What.** Only the default-named schedule `vaccinated_transfer_schedule` gets
the full data-source UI. Any additional `scheduled_exact` schedule:

- can be fed **only** from a CSV (no constant-value entry, no per-age/risk
  grid editor, no per-subpop constant editor);
- gets **no** transfer-delay parameter — `vax_transfer_delay_days` is wired
  only to the default-named schedule.

**Why.** Unbuilt. The default schedule's constant/CSV/per-subpop machinery is
a large block of reactive marimo cells; replicating it per schedule was scoped
out.

**What happens.** A non-default schedule with no CSV set falls back to
all-zero values, so its transition moves nobody. The Model Builder warns about
this at build time.

**Workaround.** Supply a CSV. For a delay, pre-shift the dates in the CSV, or
add `<slug>_vax_transfer_delay_days` to the schedule's `schedule_config` by
hand-editing the exported `model_config.json` (the schedule template already
reads `vax_protection_delay_days_param` generically).

### Naming conventions for additional schedules

Derived from the schedule name via `slugify_schedule_name`, and **not**
user-editable:

| | default-named schedule | any other schedule name |
|---|---|---|
| `df_attribute` | `daily_vaccines_df` | `<slug>_df` |
| delay param | `vax_transfer_delay_days` | `<slug>_vax_transfer_delay_days` (not wired to UI) |
| reset-date param | `vaccinated_compartment_reset_date_mm_dd` | `<slug>_compartment_reset_date_mm_dd` |
| metapop per-subpop CSV | `vaccines_<subpop>.csv` | `<df_attribute>_<subpop>.csv` |

The default-named schedule keeps its historical names so existing configs,
exported CSV bundles, and `examples/MA_vax/` keep working unchanged.

Note the CSV **value column is always literally `daily_vaccines`**, for every
schedule, regardless of what the schedule represents — it is fixed by the
model-config schema. `scheduled_exact` is not vaccine-specific; the naming is
historical.

---

## Vaccine efficacy

### `resolve_mm_dd_near_date` is duplicated in flu_core

**What.** `generic_core/data_structures.py` and
`flu_core/flu_data_structures.py` each define their own copy of
`resolve_mm_dd_near_date` (and `INFECTION_IMMUNITY_START_DATE_WARN_DAYS`).

**Why.** `generic_core` deliberately does not import `flu_core` for core
logic, so the two copies were allowed to coexist while `ve_update` was still
unmerged.

**Fix when possible.** `ve_update` has now merged, so both copies sit on the
same branch and are free to drift apart silently. Lift the single
implementation into `clt_toolkit` and have both import it from there.

### Exact parity with flu_core is restored (no longer a limitation)

**What.** `generic_core` implements the reworked vaccine-efficacy model
(multiplicative `1 - MV * VE_0`, derived peak efficacy, no
`vax_induced_saturation` in the `M` update). `flu_core` used to implement the
old additive-immunity model, so the exact-equality tests were skipped behind
`conftest.requires_flu_core_new_ve`.

**Status.** `ve_update` has merged into `main` and this branch has been synced
to it, so `flu_core` now carries the same model. The gate evaluates true and
every generic-vs-flu parity test runs and passes. `requires_flu_core_new_ve` is
now a no-op and could be retired.

**Note.** Where the two differ, it is in *where* the derived peak efficacy
lives, not in the arithmetic: `flu_core` precomputes `VE_0` onto params as
`<name>_initial`, while `generic_core` stores the inflation factor
`VE_0 / VE_season` and forms the product at each rate evaluation, so that a
later override of the season-average value still reaches the trajectory.
Anything comparing the two must translate between the two representations —
see `_generic_params` in `tests/test_generic_rate_templates.py`.

### The VE inflation factor is refreshed per run, not per parameter change

**What.** The season-average efficacies (`vax_induced_*_risk_reduce`) are read
live at every rate evaluation. The *inflation factor* they are multiplied by is
recomputed by `ConfigDrivenSubpopModel.update_ve_inflation_factors` at three
points: construction, `reset_simulation`, and the start of day 0 in
`prepare_daily_state`.

**What happens.** The day-0 refresh is what makes overrides of the dose
schedule and `vax_induced_immune_wane` take effect, since every override path
in this repo applies parameters *after* `reset_simulation()`. What is NOT
covered is a change made mid-run — between `simulate_until_day(30)` and
`simulate_until_day(60)`, say. Such a change moves the season-average
efficacies immediately but leaves the factor at its day-0 value until the next
reset.

**Torch.** `build_generic_torch_inputs` snapshots the factor into `params_dict`
at build time and there is no day-0 hook on that path, so a torch fit that
optimizes `vax_induced_immune_wane` holds the factor fixed at its build value.
Optimizing the efficacies themselves is unaffected — they are trainable leaves
and gradients reach them through the live multiplication.

**Why it matters.** The factor is *not* mildly sensitive to the waning rate: on
the Austin dose schedule it runs 1.00 at `wane = 0`, 1.17 at 0.002, and 2.37 at
0.016. Getting it a draw stale is a factor-of-two error in the applied `VE_0`,
not a rounding difference.

### `M(0)` and `MV(0)` adjustments are recomputed only on reset

**What.** `InfInducedImmunityGeneric.adjust_initial_value` (decay-forward of
`M(0)`) and `VaxInducedImmunityGeneric.adjust_initial_value` (pre-simulation
dose accumulation into `MV(0)`) read `inf_induced_immune_wane` and
`vax_induced_immune_wane` respectively. Both run at construction and in
`reset_simulation` — not at day 0, because unlike the inflation factor they
write state (`init_val`/`current_val`), and clobbering state at day 0 would
overwrite a manually set `current_val`.

**What happens.** A waning-rate override applied after `reset_simulation()`
reaches the immunity *dynamics* and the VE inflation factor, but the initial
values were computed with the previous draw's waning rate. The effect is
second-order — it perturbs day 0 only, and decays — but it is not zero.

**Fix when needed.** Re-run `reset_simulation()` after applying overrides, or
move the init-value recomputation behind an explicit "run is starting" hook
that is distinguishable from a user-set `current_val`.

### The VE cap is not re-warned after an override

**What.** `ve_derivation.warn_if_ve_initial_capped` reports age-risk groups
whose implied `VE_0` exceeds 1 (and is therefore capped) using the
season-average values in force at construction/reset.

**What happens.** An override applied later that pushes the product over 1 is
still capped correctly at evaluation time, but is not warned about. Warning
per-timestep would be far noisier than it is worth.

---

## Metapopulation

### Torch recomputed mixing exposure per sub-timestep (no longer a limitation)

**What.** `ForceOfInfectionTravelRate.torch_rate` used to call
`compute_total_mixing_exposure` itself, so the metapopulation mixing exposure
was recomputed at every sub-timestep. The numpy path has always computed it
once per simulated day, in `ConfigDrivenMetapopModel.apply_inter_subpop_updates`,
as does `flu_core` in both its object-oriented and torch paths.

**What happened.** The two paths agreed exactly at `timesteps_per_day = 1` --
one sub-timestep per day makes the two conventions identical -- and diverged
sharply above it: on the caseB two-subpopulation fixture over 50 days, the
worst compartment differed by 17% of peak `S` at `timesteps_per_day = 2` and
28% at 4. Because every parity test ran at one timestep per day, nothing
caught it. In particular torch-based fitting and numpy-based simulation of the
same model would not have agreed at `timesteps_per_day > 1`.

**Fix.** `compute_daily_mixing_exposure` (torch_generic.py) evaluates it once
per day, after the day's schedule and immunity updates, and both simulate
loops thread the result down through `generic_advance_timestep` into the
travel transition's rate config as `_total_mixing_exposure` -- the same key the
numpy path injects, and the same shape of fix as `flu_core`'s `03c2e9f`. The
template still computes it itself when nothing is injected, so direct callers
keep working.

**Regression cover.** `tests/test_generic_torch.py` now parametrizes its
generic-vs-flu parity fixture over `timesteps_per_day` in `(1, 2)`. Disabling
the hoist makes the `tspd2` cases fail while `tspd1` still passes, which is
exactly the blind spot that let this through.

## Fitting

### Log-space parameters must be scalar

**What.** A parameter in `fit_config.log_params` cannot also have age/risk/
subpopulation granularity.

**What happens.** `run_fit` raises `ValueError`: *"Log-space parameter 'X' must
be scalar; remove its age/risk/subpopulation granularity."* Log bounds must
also be strictly positive (log10 is undefined at ≤ 0).

**Code.** `fitting.py`, the `_active_log_params` guard in `run_fit`.

---

## Exported artefacts

### Previously-exported scripts do not pick up codegen changes

**What.** A generated `run_simulation.py` / `run_simulations_*.py` is a
one-time snapshot. Improvements to the export codegen do not propagate to
scripts already downloaded and checked in.

**What happens.** An old script keeps working with the semantics it was
generated under, which may differ from the current notebook's.

**Workaround.** Re-export from the Export tab against the same config.

**Currently affected.** `examples/MA_vax/run_simulations_MA_vax.py` predates
multi-`scheduled_exact` support and assumes a single schedule.
