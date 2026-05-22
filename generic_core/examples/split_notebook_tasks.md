# Split model_builder_notebook.py into per-tab files

## Objective

`generic_core/examples/model_builder_notebook.py` is a single marimo notebook
that has grown to ~5,031 lines across six tabs: **Model Builder**, **Fitting**,
**Forecast**, **Export**, **Analysis**, and **Documentation**.

The goal is to split it into one file per logical section using marimo's
`app.include()` API, so that future edits to a single tab only require reading
and touching that one file (~200–1,800 lines) rather than the entire 5,000-line
file.

The entry point (`model_builder_notebook.py`) is kept as a thin file (~50 lines)
that owns only the tab selector, output-directory widget, autosave cell, and a
sequence of `app.include()` calls.

Run the notebook the same way as before:

```
marimo run generic_core/examples/model_builder_notebook.py
```

---

## Background: marimo `app.include()`

Each sub-file is a standalone marimo app (`app = marimo.App()`). The entry-point
file merges them all into one reactive DAG via:

```python
# in model_builder_notebook.py
from generic_core.examples._nb_shared import app as _shared_app
app.include(_shared_app)
```

Variables returned by cells in an included app become available to all
subsequent cells in the including app, exactly as if those cells had been
written inline. The include **order** must respect data-flow dependencies (see
the required include order below).

Verify that `app.include()` works as expected for marimo v0.20.4 (the version
listed in `__generated_with`) before writing all sub-files — the API is stable
but worth a quick smoke-test first.

---

## Cross-tab variable dependencies (critical)

The following variables flow **across** tab boundaries and dictate the include
order.

| Variable(s) | Defined in | Consumed by |
|---|---|---|
| helpers, imports | `_nb_shared` | everything |
| `analysis_n_metrics_input`, `analysis_metric_names`, `analysis_metric_tvs` | `_nb_analysis_metric_defs` | `_build_config` (Model Builder) |
| `config_dict`, `compartments`, `is_metapop`, `metapop_folder_input`, `metapop_travel_config`, `num_age_groups`, `num_risk_groups`, `start_date_input`, `timesteps`, `rng_seed`, `total_pop_input`, `seed_inputs`, `param_names`, `scalar_param_names`, `params_inputs`, `array_param_names`, `loaded_config`, `n_transitions`, `t_name`, `transition_vars_input`, `absolute_humidity_input`, `mobility_input`, `daily_vaccines_input`, `loaded_schedule_dfs` | `_nb_model_builder` | Shared Factory, Fitting, Forecast, Export, Analysis |
| `make_single_pop_metapop`, `make_metapop_from_folder`, `extract_history` | `_nb_shared_factory` | Fitting, Forecast, Analysis |
| `fit_result` | `_nb_fitting` | Forecast, Export |
| `forecast_result` | `_nb_forecast` | Export |
| `analysis_n_metrics_input` etc. (display cell) | `_nb_analysis` | (none — display only) |

**Note on `_nb_analysis_metric_defs`:** The three cells
`_analysis_metric_defs_ui`, `_analysis_metric_sel_state`, and
`_analysis_metric_plot_controls` currently appear near the bottom of the
Analysis section in the original file, but they must be extracted to their own
file and included *before* `_nb_model_builder` because `_build_config` depends
on `analysis_n_metrics_input`, `analysis_metric_names`, and
`analysis_metric_tvs`. The `_analysis_metric_defs_show` cell (display only)
stays in `_nb_analysis.py`.

---

## Required include order in `model_builder_notebook.py`

```
_nb_shared
  → _nb_analysis_metric_defs
    → _nb_model_builder
      → _nb_shared_factory
        → _nb_fitting
          → _nb_forecast
            → _nb_export
_nb_analysis      (depends on _nb_shared_factory + _nb_analysis_metric_defs)
_nb_docs          (depends on _nb_shared for `mo`)
```

In the entry-point file, include them sequentially:

```python
app.include(_nb_shared_app)
app.include(_nb_analysis_metric_defs_app)
app.include(_nb_model_builder_app)
app.include(_nb_shared_factory_app)
app.include(_nb_fitting_app)
app.include(_nb_forecast_app)
app.include(_nb_export_app)
app.include(_nb_analysis_app)
app.include(_nb_docs_app)
```

---

## Proposed file layout

All files live in `generic_core/examples/`.

| File | Contents | Approx. lines |
|---|---|---|
| `model_builder_notebook.py` | Entry point: docstring, tab selector, output dir, autosave, `app.include()` calls | ~50 |
| `_nb_shared.py` | `_imports`, `_helpers` cells | ~256 |
| `_nb_analysis_metric_defs.py` | `_analysis_metric_defs_ui`, `_analysis_metric_sel_state`, `_analysis_metric_plot_controls` | ~50 |
| `_nb_model_builder.py` | Steps 0–10, `_build_config`, `_config_preview`, run button, `_run_sim`, `_plot_curves`, `_summary_stats` | ~1,800 |
| `_nb_shared_factory.py` | `_shared_model_factory` cell | ~200 |
| `_nb_fitting.py` | `_fitting_ui`, `_fitting_bounds_ui`, `_fitting_display`, `_fitting_obs_parse`, `_run_fitting`, `_fitting_autosave`, `_fitting_results_display` | ~490 |
| `_nb_forecast.py` | `_forecast_ui`, `_forecast_display`, `_run_forecast`, `_forecast_autosave`, `_forecast_results_display` | ~200 |
| `_nb_export.py` | `_export_display` | ~130 |
| `_nb_analysis.py` | `_analysis_sub_tab`, `_analysis_param_catalog`, all sensitivity/scenario/run/plot cells, `_analysis_metric_defs_show` (display only) | ~1,090 |
| `_nb_docs.py` | `_docs_display` | ~120 |

---

## Task list

- [ ] **Task 1** — Verify `app.include()` API for marimo v0.20.4. Write a minimal
  two-file test (a shared app with one cell, an entry app that includes it) and
  confirm the variable flows through correctly before touching the real notebook.

- [ ] **Task 2** — Create `_nb_shared.py`. Move the `_imports` cell (lines 48–120)
  and the `_helpers` cell (lines 123–304) here verbatim. Add `import marimo` and
  `app = marimo.App()` at the top.

- [ ] **Task 3** — Create `_nb_analysis_metric_defs.py`. Move three cells from the
  Analysis section of the original file:
  - `_analysis_metric_defs_ui` (lines 4559–4589)
  - `_analysis_metric_sel_state` (lines 4592–4595)
  - `_analysis_metric_plot_controls` (lines 4598–4609)
  These export `analysis_n_metrics_input`, `analysis_metric_names`,
  `analysis_metric_tvs`, `tv_opts`, `get_sel_metrics`, `set_sel_metrics`,
  `analysis_plot_metric_sel`.

- [ ] **Task 4** — Create `_nb_model_builder.py`. Move all cells between
  `_load_config_state` and `_summary_stats` (lines 358–2776), which covers:
  Step 0 (load config), Step 1 (population structure), Step 2 (compartments),
  Step 3 (transitions), Step 4 (parameters), Step 5 (schedules/immunity),
  Step 6 (diagram), Step 7 (initial conditions), Step 8 (sim settings),
  `_build_config`, Step 9 (config preview), Step 10 (run button),
  `_run_sim`, `_plot_curves`, `_summary_stats`.

- [ ] **Task 5** — Create `_nb_shared_factory.py`. Move the `_shared_model_factory`
  cell (lines 2783–2946). This cell exports `make_single_pop_metapop`,
  `make_metapop_from_folder`, and `extract_history`.

- [ ] **Task 6** — Create `_nb_fitting.py`. Move cells from `_fitting_ui` through
  `_fitting_results_display` (lines 2952–3442).

- [ ] **Task 7** — Create `_nb_forecast.py`. Move cells from `_forecast_ui` through
  `_forecast_results_display` (lines 3445–3643).

- [ ] **Task 8** — Create `_nb_export.py`. Move the `_export_display` cell
  (lines 3650–3814).

- [ ] **Task 9** — Create `_nb_analysis.py`. Move cells from `_analysis_sub_tab`
  through `_analysis_plot_age_bars` (lines 3817–4904), **excluding** the three
  cells already moved to `_nb_analysis_metric_defs.py` (tasks 3). Keep
  `_analysis_metric_defs_show` here (it is display-only and does not export
  anything needed elsewhere).

- [ ] **Task 10** — Create `_nb_docs.py`. Move the `_docs_display` cell
  (lines 4911–5031).

- [ ] **Task 11** — Rewrite `model_builder_notebook.py` as the entry point.
  Keep: module docstring, `__generated_with`, `app = marimo.App(...)`,
  `_main_tab_selector`, `_output_dir_ui`, `_output_dir`, `_tab_header_display`,
  `_autosave_config`. Replace all moved cells with `app.include()` calls in the
  order listed in the "Required include order" section above.

- [ ] **Task 12** — Smoke-test: run `marimo run generic_core/examples/model_builder_notebook.py`
  and confirm all six tabs render without errors.

- [ ] **Task 13** — Functional test: run a single-population simulation in the
  Model Builder tab (SEIR or similar) and verify epidemic curves appear.

- [ ] **Task 14** — Functional test: run a metapopulation simulation using the
  `generic_core/examples/example_metapop_inputs/` folder.

- [ ] **Task 15** — Functional test: run the Fitting tab and the Analysis tab to
  confirm that cross-tab variable flow (`config_dict`, `fit_result`, etc.) still
  works end-to-end.
