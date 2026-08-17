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
