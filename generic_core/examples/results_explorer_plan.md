# Results Explorer Notebook + SQLite-first results format

> **Status: implemented.** This document is kept as the design record — the
> measurements below are what motivated the SQLite-first decision. Delivered:
> `results_explorer_notebook.py`, `_results_explorer_lib.py`, the SQLite +
> `meta` export in `_nb_analysis.py`, the `meta` table in `_nb_export.py`'s
> generated script, and `tests/test_results_explorer_lib.py` (40 tests).

## Context

The Model Builder notebook (`generic_core/examples/model_builder_notebook.py`) produces simulation results two ways: interactively via its Analysis tab (exported as one big JSON by `_analysis_export_full`, e.g. `generic_core/examples/MA_vax/analysis_results_full_example.json`), or via the standalone `run_simulation.py` script from the Export tab (which writes `results.db`, SQLite). There is no tool for exploring either afterwards — the Analysis tab's plots exist only live in-session, and the only code that reads `results.db` back is a bespoke MA_vax-specific report script (`build_counterfactual_tables_from_db.py`).

The goal is a standalone marimo notebook — a generic results explorer — that loads results produced either way and lets the user build charts (time series, histogram, boxplot, stacked bar, scatter) with selectable aggregation (population total / subpop / age group / risk group / all, faceted) and scenario comparison (single, multiple, or each-vs-baseline), dynamically adding and removing chart blocks. It must generalize to any config-driven model, not just MA_vax.

**This revision changes the approach in one important way.** The original plan treated the Analysis tab's JSON as a fixed input to work around. On investigation that format is the actual problem, and it is cheaper to fix it than to build a loader around it. The plan now covers both: a small change to how results are written, and the explorer that reads them.

### Why the JSON format should change (measured, not assumed)

`_analysis_export_full` (`_nb_analysis.py:1871-1946`) accumulates *every* row into Python lists, then `json.dumps` the entire structure into a single string, then writes it. Both the row lists and the complete serialized string are held in memory simultaneously. Benchmarked against a faithful reproduction of that code path at 2.8M rows (212 MB output):

| | peak RSS | time | output |
|---|---|---|---|
| JSON (current approach) | **1.16 GB** | 2.4s | 212 MB |
| SQLite, batched `executemany` | **0.04 GB** | 3.5s | 239 MB |

JSON peak memory is ~5.5× the output size and scales linearly with it; SQLite's is bounded by one array's batch and is *constant* in export size. The real MA_vax export is 20M rows / 1.4 GB, which extrapolates to roughly **8 GB peak just to write the file**, and the reader then needs ~12 GB more to load it back (`json.load` measured at ~8.4× file size; DuckDB's own `read_json_auto` is worse — ~19×, and it additionally trips a 16 MiB default object-size cap because the file is one giant top-level object rather than JSON-lines).

So the JSON path costs many GB on both ends, for a file that then still has to be fully parsed before a single chart can be drawn. Writing the same rows to SQLite instead is bounded-memory to produce, needs no load step at all to read (DuckDB attaches and queries in place with predicate pushdown), and — decisively — is *the format the other half of the pipeline already produces*. Fixing the writer collapses two formats into one.

Verified supporting facts:
- DuckDB 1.5.0 and Altair 6.0.0 are already pinned in `requirements.txt`; `INSTALL sqlite_scanner; LOAD sqlite_scanner` works, and `ATTACH ... (TYPE sqlite, READ_ONLY)` queries `results.db` directly.
- Against the 116 MB deterministic db, dimension-discovery queries run in 0.03–0.13s and a full time-series-with-CI aggregation (2 scenarios × 250 days × 7 age groups) in 0.62s.

### A second gap: results.db carries no metadata

Inspecting `results.db` directly: it contains only the `results` and `results_full` tables. There is **no `start_date`**, no age-group labels, no subpop ordering. The explorer could therefore only ever plot "day 1…250" rather than real dates, and label age groups "0…6" rather than `0-0, 1-4, 5-12, …` — a real regression against the Analysis tab's own plots, which use `pd.date_range(start=analysis_results["start_date"], …)` and the config's `age_groups` labels.

Discovery cost makes this worse. Against the 57 GB `results.db`, `SELECT DISTINCT scenario FROM results` takes **7.1s** (that table alone holds 73.4M rows), and `SELECT DISTINCT subpop FROM results_full` takes **48.7s — to return a single row** (`pop`, this being a single-population model). Populating the subpop/age/risk selectors by scanning would therefore cost roughly a minute every time a file is opened, to learn almost nothing. SQLite is row-oriented, so DuckDB must read every row to answer these; no index avoids it.

Both problems have the same fix: **write a small `meta` table alongside the data**, from both producers. Discovery then becomes a single tiny read regardless of file size, and the explorer gets real dates and readable labels for free.

## Design

### Part 1 — Results format (small changes to existing code)

**1a. `_nb_analysis.py`'s `_analysis_export_full` writes SQLite instead of JSON.**
Replace the row-accumulation + `json.dumps` body with a `sqlite3` writer emitting the exact schema `_nb_export.py:691-708` already defines (`results`, `results_full`, same columns, same `(scenario, compartment)` indexes on both, `PRAGMA journal_mode=WAL` / `synchronous=NORMAL`). Crucially, `executemany` per array inside the existing scenario/rep/key loops rather than building one giant list — that is what makes peak memory constant. Output becomes `output_dir / "analysis_results_full.db"`; relabel the button "Export full results (SQLite)". The existing per-array `np.indices(...)` / `itertools.repeat` row construction is already the right shape and transfers over nearly unchanged — this is a localized rewrite of ~40 lines, not a redesign.

**1b. Both writers also emit a `meta` table.** In `_analysis_export_full` and in the `run_simulation.py` template inside `_nb_export.py`. A single-row (or key/value) table carrying what the data tables structurally cannot: `start_date`, `num_days`, `timesteps_per_day`, `num_age_groups`, `num_risk_groups`, `age_group_labels` (JSON list from the config's `age_risk.age_groups`), `subpop_names`, `scenarios` (in definition order — SQL `DISTINCT` loses the user's intended ordering, which matters for consistent chart colors and for identifying the baseline), `compartments` and `transition_vars` (with their `kind`), `n_reps`, `param_set_indices`, `uncertainty_source`, and `stochastic`. Every one of these is already in scope at both write sites (`analysis_results` dict / the script's module constants), so this is bookkeeping, not new computation.

**1c. Backwards compatibility.** The explorer must not hard-require `meta` — existing `results.db` files (including both MA_vax examples) predate it. When `meta` is absent, fall back to `SELECT DISTINCT` discovery, show a "no metadata — using day indices, showing age groups as indices; re-export to get dates and labels" note, and plot day numbers. Detect via `sqlite_master`, the same way `_nb_export.py:679-690` already checks for older column sets.

**1d. Legacy JSON is read via one-time conversion, not a parallel code path.** `load_source` given a `.json` streams it into a sibling `<name>.db` once (with a progress indicator), then attaches that — so the 1.4 GB example file keeps working and every subsequent open is instant. Use `ijson` for the streaming parse so conversion is bounded-memory rather than the ~12 GB a `json.load` would need; **this is the one new dependency to add to `requirements.txt`** (you approved new packages; it is small, pure-Python-with-C-backend, and used only on this legacy path).

**On vegafusion: recommend not installing it.** It exists to push aggregation server-side for large datasets handed to Vega — but here DuckDB has *already* aggregated before Altair sees anything, so charts receive thousands of rows, not millions. Altair's 5000-row cap is real (`MaxRowsError` reproduced at 12k rows, and a 2-scenario × 250-day × 7-age-group query already returns 3,500 rows, so realistic selections will cross it), but `alt.data_transformers.enable("default", max_rows=None)` lifts it cleanly and was verified to work; the resulting payloads are a few MB at most. vegafusion would add a heavyweight dependency to solve a problem the DuckDB layer already solves. Pair `max_rows=None` with a soft guard that warns above ~50k rows and suggests narrowing the selection.

### Part 2 — The explorer

**Files** (both new, neither touched by `build_notebook.py` — per `CLAUDE.md` only `model_builder_notebook.py` is generated):

1. **`generic_core/examples/_results_explorer_lib.py`** — plain Python, no marimo import, unit-testable:
   - `load_source(path) -> duckdb.DuckDBPyConnection` — dispatch on suffix; `.db`/`.sqlite` → `ATTACH ... (TYPE sqlite, READ_ONLY)` + views named `results`/`results_full`; `.json` → convert-once-then-attach (1d). Both yield identical table names so all downstream SQL is source-agnostic.
   - `read_meta(con) -> dict` — the `meta` table if present, else `None`; and `discover_dims(con, meta)` — returns dims from `meta` when available, else the `SELECT DISTINCT`/`MIN`/`MAX` fallback.
   - `ChartConfig` schema + per-chart-type query builders and Altair builders.
   - `render_chart(con, dims, config) -> alt.Chart` — the single entry point, raising `ValueError` on an incomplete config so the caller can show an inline error.
2. **`generic_core/examples/results_explorer_notebook.py`** — the marimo app (`app = marimo.App(...)`, run via `marimo edit`/`marimo run`, not in `build_notebook.py`'s `SECTIONS`). Imports the lib for all data logic and `_nb_shared` for `step_header`/`section_card`/`CLT_ACCENT` styling consistency. Holds only UI: source picker, cached dims, dynamic chart list, render loop.

**Source picker.** `mo.ui.text` path box as the source of truth (multi-GB local files are not upload targets), optionally pre-filled by an `mo.ui.file_browser(selection_mode='file', filetypes=['.json','.db'])`. Gate the actual load behind `mo.ui.run_button("Load")` — same pattern as `_nb_analysis.py`'s `analysis_export_full_button`. Cache the connection in `mo.state(None)` and the discovered dims in a second `mo.state({})`, closing any previous connection first so a stale attachment doesn't linger.

**Dynamic chart list.**
```python
get_charts, set_charts = mo.state([])   # list[ChartConfig dict]
get_next_id, set_next_id = mo.state(0)  # stable id, independent of list position
```
`ChartConfig`: `id`, `chart_type` (`timeseries`|`histogram`|`boxplot`|`stacked_bar`|`scatter`), `metrics: [str]` (summed when several are selected), `agg_level` (`population`|`subpop`|`age_group`|`risk_group`|`all`), `scenario_mode` (`single`|`multiple`|`compare_baseline`), `scenarios: [str]`, `baseline_scenario`, `compare_metric` (`difference`|`ratio`), `pairing` (`paired`|`unpaired`), `show_uncertainty`, `day_range`, `scatter_x`, `scatter_y`.

Plus, added after the first round of use:

- `subpop_filter` / `age_filter` / `risk_filter` — pin the chart to a single slice. Deliberately **independent of `agg_level`**, rather than a special case of it: "population total, but only age group 6" is a distinct and common request from "broken out by age group". `None` means no restriction. Any active filter forfeits the pre-aggregated `results` fast path (it has no such column) and reads `results_full`.
- `cumulative` — running total over days, taken **per replicate inside the CTE**, so the median and interval describe the spread of cumulative curves rather than the cumulative sum of daily medians. Rejected for compartments: a compartment is already a level, so summing it over days measures nothing. Accumulation starts at the beginning of the selected day range.
- `hide_empty` — drop breakdown groups whose series is entirely zero, so a granular aggregation does not fill the grid with blank panels. Never drops everything: if every group is empty the frame passes through and the caller reports "no data".
- `shared_y` — small multiples share a y scale by default (that is what makes them comparable), but one dominant group flattens every other panel against the axis until it reads as blank. Turning this off rescales each panel to its own data.

The day-range readout is rendered by the notebook rather than by `mo.ui.range_slider`'s `show_value`, which prints the bare tuple — "1 to 250" comes out as `1, 250` and reads as the number 1,250.

Mutate with the **callable-update form** (`set_charts(lambda cs: cs + [new])`, removal `lambda cs: [c for c in cs if c["id"] != id]`) — required because `mo.state`'s setter does not reflect through `get_*()` within the same reactive pass, the convention `_nb_analysis.py` already follows around its scenario-state cells.

Marimo cells are static, so N charts cannot be N cells: one cell iterates `get_charts()`, building each block's edit widgets (via `mo.ui.array`, the idiom `_nb_analysis.py` uses for repeated widget groups) plus a Remove button, wrapping each `render_chart` call in try/except so one bad config yields an inline `mo.callout(kind="danger")` instead of killing the cell, and rendering through `mo.ui.altair_chart(chart)` for marimo's native selection/brush support.

**Query strategy.** `agg_level` maps to source table + `GROUP BY` columns: `population` reads `results` (already aggregated, far smaller); everything else reads `results_full` and `SUM(value)` over the dimensions not grouped on. Every generated query must carry a `WHERE scenario IN (...) AND compartment IN (...)` filter — on the 57 GB file an unfiltered `results_full` scan is minutes, so this is a correctness-of-performance requirement, not an optimization.

- *Time series with CI* (flagship): inner `SELECT scenario, rep, param_set, day, {agg}, SUM(value) v ... GROUP BY ...`, outer `SELECT scenario, day, {agg}, median(v), quantile_cont(v,0.025), quantile_cont(v,0.975) ... GROUP BY scenario, day, {agg}`. Both are native DuckDB aggregates, so percentiles compute in-engine rather than pulling every replicate into numpy as `_analysis_plot_compartments` does today. Drop the CI columns when `n_reps == 1`. Render layered `mark_area(opacity=0.2)` + `mark_line()`, colored by scenario, faceted for `agg_level="all"`. X axis uses real dates derived from `meta.start_date + day`, falling back to day index without `meta`.
- *Scenario vs baseline*: **offer both pairings, default to paired, and state the assumption in the chart's own UI copy** ("Paired by replicate index — assumes rep *i* in each scenario corresponds to rep *i* in baseline"), per your decision. Paired = self-join the per-rep CTE on `(rep, param_set IS NOT DISTINCT FROM, day, {agg})` between baseline and each comparison scenario, compute `v - v_base` or `v / NULLIF(v_base, 0)` per matched pair, then median/quantile the derived quantity. Unpaired = aggregate each scenario independently, then difference/ratio the point estimates. Add a `mark_rule` at 0 (difference) or 1 (ratio). The assumption holds for transitions-only uncertainty (export scripts seed replicates by position) and is shakier under parameter sampling — hence making it visible rather than implicit.
- *Histogram / boxplot*: per-rep totals (`GROUP BY scenario, rep, param_set, {agg}`), handed raw to `mark_bar(bin=True)` / `mark_boxplot()` for client-side binning and quartiles. Row count is bounded by replicate count.
- *Stacked bar*: per-rep totals at a fixed day/window, `median()` across reps in SQL, `mark_bar()` with `color={agg col}`.
- *Scatter*: one query with two `SUM(value) FILTER (WHERE compartment = ?)` columns to avoid a join; `mark_point()` colored by scenario.

## Verification

1. **Round-trip the new writer**: run a small Analysis-tab scenario set with "Keep full per-replicate results" on, export SQLite, and assert the resulting `results`/`results_full` row counts and a sampled set of values match what the equivalent JSON export produced (the two must be numerically identical — this is a format change, not a semantics change). Confirm a `meta` table is written and populated.
2. **Fast explorer smoke test** (every iteration): open `generic_core/examples/MA_vax/simulation_output_deterministic/results.db` (116 MB, has no `meta`) — exercises the legacy/no-metadata fallback, and its single replicate exercises graceful CI degradation. Add and remove one of every chart type.
3. **Legacy JSON conversion**: point the explorer at `analysis_results_full_example.json` (1.4 GB), confirm the one-time conversion completes within bounded memory (measure peak RSS via `/usr/bin/time -l`) and that the resulting `.db` answers the same queries as the original.
4. **Large-file performance**: against the 57 GB `results.db`, confirm discovery is instant with `meta` present (versus the measured 7.1s / 48.7s legacy scans), and audit every generated SQL string for its scenario/compartment filter before any `results_full` `GROUP BY`.
5. **Unit tests** for `_results_explorer_lib.py` against a small hand-built fixture db — `discover_dims` with and without `meta`, and the paired vs unpaired comparison SQL against hand-computable numbers. Check `generic_core/tests/` for the existing layout and match it.

## Files to change / create

- `generic_core/examples/_nb_analysis.py:1855-1946` — `_analysis_export_full` writes SQLite + `meta` (and the button label). **Regenerate afterwards**: `python generic_core/examples/build_notebook.py`, per `CLAUDE.md` — never hand-edit `model_builder_notebook.py`.
- `generic_core/examples/_nb_export.py:691-757` — add the `meta` table to the generated `run_simulation.py` template. Same regeneration step.
- `generic_core/examples/_nb_docs.py:242` — the Documentation tab documents the export as `analysis_results_full.json`; update to `.db`. (A repo-wide grep confirms these are the *only* references — nothing consumes the JSON programmatically, so replacing it outright breaks no downstream code. If you'd rather keep a JSON option alongside for external tooling, say so and I'll leave both buttons.)
- `generic_core/examples/_results_explorer_lib.py` — new.
- `generic_core/examples/results_explorer_notebook.py` — new.
- `requirements.txt` — add `ijson` (legacy JSON conversion only).

## Reference

- `generic_core/examples/_nb_export.py:659-757` — authoritative `results.db` schema to mirror.
- `generic_core/examples/_nb_analysis.py:1950` (`_analysis_plot_compartments`) — existing numpy median/CI logic the DuckDB queries replace; `:329-511` — `mo.ui.array` / `mo.state` callable-update idiom.
- `generic_core/examples/_nb_shared.py` — `step_header` / `section_card` / `CLT_ACCENT`.
- `generic_core/examples/MA_vax/build_counterfactual_tables_from_db.py:120-204` (`ResultsDB`) — existing sqlite→numpy query patterns, useful cross-check.
- `flu_instances/examples/flu_scenario_analysis.py:746+` — precedent for a standalone marimo notebook with dropdown-driven plots over its own results db (different schema; UI shape only).
