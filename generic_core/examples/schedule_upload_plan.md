# Upload-a-CSV schedule override in the Analysis tab

## Context

The Analysis tab can currently modify model inputs in exactly three ways: per-scenario
scalar/array **parameter** overrides, per-subpop variants of those, and a per-age-group
**dose multiplier** that *scales* an existing `vaccine_schedule` (`scale_dose_schedule_df`).
There is no way to supply a genuinely *different* schedule — a new vaccination rollout, a
different school calendar, an alternative humidity trace. Today that requires going back to
the Model Builder tab, editing the CSV filename, and rebuilding, which destroys the ability
to compare two schedules side by side in one run.

This adds a fourth modification type: **per-scenario replacement of any schedule by an
uploaded CSV**, working in both the Analysis tab and the exported `run_simulation.py`.

Decisions taken (confirmed with the user):
- **Per-scenario**, so "current schedule" vs "new schedule" is a single analysis run.
- **All schedules** are replaceable (humidity, school/work calendar, mobility, daily
  vaccines, and any extra `scheduled_exact` schedules) — not just the dose ones.
- **Metapop: per-subpop uploads**, mirroring the dose multiplier's shared-vs-per-subpop
  toggle (a shared upload applies to every subpop; a per-subpop toggle overrides it).

Design note: we deliberately do **not** use `SubpopModel.replace_schedule` /
`MetapopModel.replace_schedule` (`clt_toolkit/base_components.py:1200,1533`). None of the
`*Generic` schedule classes in `generic_core/schedule_templates.py` override
`postprocess_data_input`, so `replace_schedule` would install a raw CSV-shaped DataFrame
(string dates, JSON-string value column, no date index) and blow up at lookup time. Instead
we inject the replacement DataFrame into the `schedules_input` namespace *before* the model
is built, which is the path every existing schedule already takes.

## Implementation

### 1. Shared helpers — `generic_core/examples/_nb_shared.py`

Both helpers go **inside the `_helpers` cell** (`def _helpers(Path, SimpleNamespace, json, np, pd)`,
L91-L516) next to `load_csv_validated` (L118) — these are nested functions in a marimo cell,
not module-level defs — and must be added to that cell's `return` tuple (L485) and to the
arg list of every cell that consumes them.

- `load_csv_bytes_validated(contents: bytes, required_columns) -> (df, error_str)` — same
  column check and `Unnamed*` strip, reading from `io.BytesIO`. Factor the shared body out
  of `load_csv_validated` so both paths validate identically.
- `schedule_upload_specs(config_dict) -> list[(schedule_name, df_attribute, required_columns)]`
  — walks `config_dict["schedules"]` and maps each `schedule_template` to its df attribute
  and expected columns:
  - `timeseries_lookup` → `df_attribute`, `["date", value_column]`
  - `contact_matrix` → `school_work_day_df_attribute`, `["date", "is_school_day", "is_work_day"]`
  - `vaccine_schedule` → `df_attribute`, `["date", "daily_vaccines"]`
  - `mobility` → `df_attribute`, `["mobility_modifier"]` (plus `date` **or** `day_of_week`)
  De-duplicate by df attribute (two schedules can share one).

### 2. Analysis-tab UI — `generic_core/examples/_nb_analysis.py`

New cell `_analysis_schedule_upload_controls`, placed immediately after
`_analysis_dose_mult_controls` (L566) and modelled on it directly:

- `analysis_sched_upload_toggle` — `mo.ui.switch("Replace schedules from uploaded CSVs (per scenario)")`
- `analysis_sched_upload_sel` — `mo.ui.multiselect` over the schedule names from
  `schedule_upload_specs`, so only the chosen schedules get upload rows rendered (keeps the
  grid small; the widget arrays themselves are built over *all* specs so the array shape is
  stable across re-runs).
- `analysis_sched_upload_files` — `mo.ui.array` indexed `[schedule][scenario]` of
  `mo.ui.file(filetypes=[".csv"], multiple=False)`, `_MAX_SC = 5` scenarios as elsewhere.
- Metapop: `analysis_sched_upload_per_subpop_toggle` + `analysis_sched_upload_subpop_files`
  indexed `[subpop][schedule][scenario]`, exactly paralleling
  `analysis_dose_mult_subpop_inputs` (L670).

Widget count: like the dose grid, the arrays are instantiated for *all* schedules × 5
scenarios (× subpops), and the multiselect only gates what is **rendered**. For a 3-schedule,
5-subpop metapop that's 75 `mo.ui.file` widgets — comparable to the existing dose grid's
`5 × 3 × A` numbers, and empty file widgets are cheap.

Do **not** route these through `get_scenario_dose_state`/`set_scenario_dose_state`.
`mo.ui.file` holds its contents in the widget itself, and that state dict is serialized into
the downloadable `scenario_config.json` (`_nb_analysis.py:428-435`) — embedding CSV bytes
there would bloat it and can't be restored anyway (UI element values can't be assigned from
Python; see the comment at `_nb_fitting.py:70-75`). Instead record only
`{schedule: filename}` in a small `mo.state` for display, and note in the UI that uploads
must be re-selected after a config restore.

New cell `_analysis_schedule_upload_parse` producing
`analysis_sched_override_dfs` (`{scenario_idx: {df_attribute: df}}`) and
`analysis_sched_override_dfs_per_subpop` (`{scenario_idx: {subpop_idx: {df_attribute: df}}}`),
parsing with `load_csv_bytes_validated` and rendering per-file success/error callouts in the
Scenario section card (`_analysis_display`, layout L1257-L1327), same style as the Model
Builder's Step 4 callouts.

### 3. Scenario tuples — `_analysis_define_scenarios` (`_nb_analysis.py:1332`)

Extend the 7-tuple to a **9-tuple**:

```
(name, global_overrides, per_subpop_overrides, designed, dose_multiplier,
 dose_multiplier_per_subpop, ratios,
 schedule_df_overrides,            # {df_attribute: DataFrame} or None
 schedule_df_overrides_per_subpop) # list indexed by subpop of that dict, or None
```

Only ever populated in the Scenario sub-tab (Sensitivity sweeps a param); the three
Sensitivity `append` sites (L1405/1426/1445) get `None, None` appended.

Every existing consumer reads these tuples with defensive `scen_tuple[N] if len(...) > N`
indexing (`analysis_runner.py:305-309`, `_nb_export.py:874-877`), so widening to 9 is
backward-compatible with no other edits.

### 4. Runner — `generic_core/analysis_runner.py`

**Do not put the DataFrames in the per-task tuples.** Two reasons:
- `run_analysis_scenarios` builds one task per (scenario × replicate) and `pool.imap`
  pickles each one, so a CSV would be re-pickled hundreds or thousands of times.
- The progress callback reads `tasks[i][-1]` for the replicate index (L349); appending
  slots to the task tuple silently breaks it.

Instead use the module's established "ship once per worker" mechanism (see its docstring):

- `run_analysis_scenarios` (L252): collect `{scenario_name: {attr: df}}` and
  `{scenario_name: [ {attr: df} | None, ... ]}` from tuple slots 7/8 and put them in
  `payload` (L317) as `schedule_overrides_by_scenario` /
  `schedule_overrides_per_subpop_by_scenario`. Task tuples stay exactly as they are.
- `_analysis_setup` (L66): accept the two new keyword args (`payload` is `**`-expanded
  into it, so the signature must match).
- `run_one` (L171-173): the scenario name is currently discarded as `_scenario_name` — use
  it to look the overrides up. For single-pop, build a shallow copy of `run_sched` with the
  override attributes replaced (see the precedence caveat in §5) and pass it as
  `schedule_dfs=`. For metapop, forward the new `make_metapop_from_folder` kwargs.
- Ordering rule (document it in a comment): the uploaded CSV **replaces** the base schedule,
  and any `dose_mult` for that same schedule is then applied **on top** of the replacement.

### 5. Factory — `generic_core/model_factory.py`

- `make_metapop_from_folder` (L351): new kwargs
  `schedule_df_overrides=None, schedule_df_overrides_per_subpop=None`. Apply inside the
  per-subpop loop (L409-463), *before* `scale_dose_schedule_df` at L423/L440 and before
  `build_notebook_schedules_input` at L454, overriding `_shared_ah` / `_cal_df` /
  `_shared_mob` / `_vax_df` / `_extra_dfs[attr]` for that subpop. Per-subpop entries take
  precedence over the shared dict, matching the existing `dose_mult_per_subpop` precedence
  at L420-422.
- `make_single_pop_metapop` (L245) needs no signature change, **but there is a precedence
  trap the runner must work around.** At L288-L311 it collects extra `scheduled_exact`
  dfs as:

  ```python
  _extra_dfs = dict(extra_scheduled_dfs or {})
  if not _extra_dfs:
      _bundled = getattr(_sched_dfs, "extra_scheduled_dfs", None)   # dict wins
      ...
      for _attr, _val in vars(_sched_dfs).items():                  # flat attrs only if dict absent
  ```

  The notebook's `loaded_schedule_dfs` always carries a populated `extra_scheduled_dfs`
  dict (`_nb_model_builder.py:1897`), so setting a flat `<attr>_df` attribute on the
  namespace copy would be **silently ignored** for any extra schedule. The runner must
  therefore override *inside* that dict too (rebuild `extra_scheduled_dfs` with the
  replacement merged in), not just set the flat attribute. Base attributes
  (`absolute_humidity_df` / `school_work_calendar_df` / `mobility_df` /
  `daily_vaccines_df`) are read directly at L322-L326 and are fine as flat attributes.
  Once that is right, `dose_mult` scaling at L286/L313 correctly applies on top of the
  replacement.

### 5b. Adjacent pre-existing bug to fix while here

`schedule_dfs_for` (`analysis_runner.py:123-134`) and its notebook twin
(`_nb_analysis.py:1707-1714`) rebuild the schedule namespace from only the four base
attributes plus `transmission_multiplier_df` — **`extra_scheduled_dfs` is dropped**. So
today, whenever a fitted m(t) is in use, every extra `scheduled_exact` schedule silently
falls back to all-zero constants. This feature routes through exactly that namespace, so
fix it in the same change (carry `extra_scheduled_dfs` through both rebuilds) — otherwise
an uploaded override for an extra schedule would appear to work with m(t) off and vanish
with it on.

### 6. Export — `generic_core/examples/_nb_export.py`

**Payload.** `_schedules_payload` (L1076) is currently flat `{df_attribute: csv_text}` and
skipped entirely for metapop. Keep the flat base keys and add one reserved key. Per-subpop
entries are **lists indexed by subpop order**, matching how `SUBPOP_PARAM_OVERRIDES` and
`DOSE_MULTIPLIER_PER_SUBPOP` already work — `_export_display` has no `analysis_sp_names` in
its signature, so keying by subpop *name* would mean threading a new input through for no
benefit:

```json
{ "absolute_humidity_df": "...csv...",
  "__scenario_overrides__": {
      "per_scenario":  {"<scenario>": {"daily_vaccines_df": "...csv..."}},
      "per_subpop":    {"<scenario>": [{"daily_vaccines_df": "...csv..."}, null]}
  } }
```

Drop the `is_metapop` short-circuit so metapop exports still bundle the override CSVs (the
base per-subpop schedules keep coming from `METAPOP_FOLDER`). Two follow-on edits:
- The download note at L1203-1206 does `sorted(_schedules_payload)` to list the bundled
  attributes — filter out the `__scenario_overrides__` key there, or it shows up as a
  bogus schedule name.
- Update the metapop note at L1135 (it currently states schedules are not bundled for
  metapop, which stops being true).

**Script template** (the `_script` literal from L29):
- `_real_or` (L322) needs **no** `__`-prefix guard: it only ever tests
  `if _name in _csvs` for known `*_df` attribute names, so the reserved key can never
  collide.
- `_build_schedules` (L309) gains a `schedule_overrides=None` param (a `{attr: csv_text}`
  dict) consulted ahead of `_csvs` inside `_real_or`; dose scaling stays applied after,
  preserving the same ordering as the notebook.
- `build_model` (L364) gains `schedule_overrides` / `schedule_overrides_per_subpop`,
  passes the former into `_build_schedules` for the single-pop path, and for the metapop
  path (L421-432 — which does *not* use `_build_schedules` today) parses the CSV text to
  DataFrames and forwards them as the new `make_metapop_from_folder` kwargs.
- `_run_one` (L567) needs **no new task slots**: it already receives `scenario_name`, and
  `SCHEDULE_OVERRIDES` is a module global rebuilt in every worker on re-import, exactly
  like `SCENARIOS`/`DOSE_MULTIPLIER`. It just looks them up and passes them to
  `build_model`. The task-tuple construction at L628-632 is untouched.
- New placeholder `# <<<SCHEDULE_OVERRIDES_BLOCK>>>` after `DOSE_MULTIPLIER_SUBPOP_BLOCK`
  (L129), emitting `SCHEDULE_OVERRIDES = {scenario: {attr: attr}}` /
  `SCHEDULE_OVERRIDES_PER_SUBPOP = {scenario: [{attr: attr} | None, ...]}` — plain
  attribute names indexing into `schedules.json`, not inline CSV text, so the generated
  script stays readable and the CSV data lives in exactly one place.
- Generate that block alongside the existing ones (~L830-990) and add the matching
  `.replace(...)` in the assembly chain (~L1020). Remember the blocks are built at column
  0 and substituted *after* `_textwrap.dedent`.

### 7. Regenerate and document

- `python generic_core/examples/build_notebook.py`, then
  `python generic_core/examples/check_notebook_sync.py` (per `CLAUDE.md`; never hand-edit
  `model_builder_notebook.py`).
- Add a short note to `generic_core/limitations.md`: uploaded schedule CSVs are not saved
  into `scenario_config.json` and must be re-uploaded after a config restore.

## Verification

1. `python generic_core/examples/check_notebook_sync.py` — must pass (pre-commit hook enforces it).
2. `pytest tests/ -x -q` — no regressions; `tests/test_generic_metapop.py` and
   `tests/test_generic_scheduled_transitions.py` cover the factory paths being touched.
3. New test in `tests/test_generic_metapop.py`: build a metapop from
   `generic_core/examples/example_metapop_inputs/` twice — once plain, once with
   `schedule_df_overrides_per_subpop` supplying a modified `vaccines_SubpopA.csv` (e.g. all
   zeros) — and assert the resulting `scheduled_exact` transfer differs only for SubpopA.
   Also assert `schedule_df_overrides` + `dose_mult` compose in the documented order.
4. Manual, single-population: `marimo edit generic_core/examples/model_builder_notebook.py`,
   load the SIR/flu example, Analysis → Scenario → two scenarios, upload a zeroed
   `vaccines_*.csv` for scenario 2 only, run, and confirm the two trajectories diverge in
   the plot while scenario 1 matches the pre-change baseline.
5. Manual, metapop: same with `example_metapop_inputs`, using the per-subpop toggle and
   uploading only for SubpopB; confirm only SubpopB's curves change.
6. Export round-trip: Export tab → download `run_simulation.py`, `model_config.json`,
   `fitted_params.json`, `schedules.json` into one folder → `python run_simulation.py` →
   confirm the scenario outputs match the Analysis tab's (same `SEED_BASE`, deterministic
   run) — this is the check that the override actually survived the export, not just the
   script running without error. Run it once for a single-population model and once for a
   metapop one, since those take different code paths in `build_model`.
7. Regression check for the §5b fix: a config with an extra `scheduled_exact` schedule,
   run with fitted m(t) enabled, must now show a non-zero transfer for that schedule
   (before the fix it is silently all-zero).
8. Back-compat check: an older `scenario_config.json` (no schedule-upload keys) still
   restores cleanly, and an older `schedules.json` (flat, no `__scenario_overrides__`)
   still runs in the exported script.
