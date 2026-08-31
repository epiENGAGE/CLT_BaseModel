"""
results_explorer_notebook.py
============================

Interactive explorer for simulation results, as a directory of
Hive-partitioned Parquet, from either:

  * ``analysis_results_full_parquet/`` — the Analysis tab's "Export full
    results (Parquet)" button in model_builder_notebook.py
  * ``simulation_output/results_parquet/`` — the ``run_simulation.py``
    script downloaded from that notebook's Export tab

Both write the same ``results``/``results_full`` schema, so either opens the
same way.

Run with::

    marimo edit generic_core/examples/results_explorer_notebook.py
    marimo run generic_core/examples/results_explorer_notebook.py

Add as many charts as you like; each one independently chooses its metric(s),
aggregation level (population total, or broken out by subpopulation / age /
risk group), which single slice to restrict to, and how scenarios are
compared. Transition variables can additionally be shown as running totals.
Nothing here is specific to any one model — every selector is built from
whatever the results file contains, in the order its `meta` table records.

Everything is downloadable: each chart offers its own rows as CSV and itself
as an image, and a separate section at the top pulls out bulk data — any set
of scenarios and metrics, per replicate or summarized across them, per day or
totalled — without having to plot it first.

This is a standalone notebook: unlike model_builder_notebook.py it is NOT
assembled by build_notebook.py, so edit it directly.

Design note: queries run against the file in place via DuckDB rather than
loading it into memory. Result sets reach tens of GB (one MA_vax example is
57 GB), so only the aggregated data behind each chart is ever materialized.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import sys
    import traceback
    from pathlib import Path

    import marimo as mo

    # The query/plot helpers live next to this file. Importing by path keeps
    # the notebook runnable from any working directory. NOTEBOOK_DIR is also
    # where the file browser starts, so __file__ is only consulted here.
    NOTEBOOK_DIR = Path(__file__).parent
    sys.path.insert(0, str(NOTEBOOK_DIR))
    import _results_explorer_lib as rex

    return NOTEBOOK_DIR, Path, mo, rex, traceback


@app.cell
def _helpers(Path, mo, rex):
    # Note: these deliberately avoid a leading underscore. In marimo a name
    # starting with "_" is cell-local, so an underscored helper would not be
    # visible to the cells below that need it.
    ACCENT = "#7B5EA7"

    # Controls are pre-built for a fixed number of chart slots (see
    # _chart_widgets for why the count must not vary), so this is also the
    # cap on how many charts can exist at once.
    MAX_CHARTS = 8

    # Label -> stored value. Dropdowns take the *label* as their `value`,
    # so label_for() below inverts these when seeding a widget.
    AGG_OPTIONS = {
        "Population total": "population",
        "By subpopulation": "subpop",
        "By age group": "age_group",
        "By risk group": "risk_group",
        "All (subpop × age × risk)": "all",
    }
    # Two modes, not three: how many scenarios a chart shows is decided by
    # the Select multiselect, so a "single vs multiple" mode would only imply
    # a constraint nothing enforces (and previously did nothing at all).
    MODE_OPTIONS = {
        "Show levels": "levels",
        "Compare to baseline": "compare_baseline",
    }
    COMPARE_OPTIONS = {"Difference": "difference", "Ratio": "ratio"}
    PAIRING_OPTIONS = {"Paired by replicate": "paired", "Unpaired": "unpaired"}

    def default_config_for(spec, dims):
        return rex.default_chart_config(spec["id"], spec["chart_type"], dims)

    def section_card(title, subtitle, body):
        header = mo.Html(
            f'<div style="font-size:1.05rem;font-weight:700;color:{ACCENT};">{title}</div>'
            f'<div style="color:#777;font-size:.85rem;margin:.1rem 0 .5rem;">{subtitle}</div>'
        )
        return mo.vstack([header, body]).style({
            "border": "1px solid #e3e3e3",
            "border-left": f"4px solid {ACCENT}",
            "border-radius": "8px",
            "padding": "0.9rem 1.1rem",
            "margin-bottom": "0.9rem",
        })

    def label_for(options: dict, value):
        """The label a dropdown should display for a stored config value.

        mo.ui.dropdown takes its `value` as the *label*, while configs store
        the underlying value, so this inverts the options mapping.
        """
        for lbl, val in options.items():
            if val == value:
                return lbl
        return next(iter(options))

    def file_name(path):
        return Path(path).name

    def download_buttons(result, config, dims):
        """CSV + image download buttons for one chart.

        Both are lazy (marimo calls back into the kernel when the button is
        clicked) because every chart is re-rendered on every widget change:
        converting each one to a PNG on the way past would dominate the render
        even though almost none of them are ever downloaded.

        The CSV is the chart's own rows — already filtered, relabelled and, for
        a comparison, differenced against the baseline — so it is what is on
        screen rather than a fresh query that might not agree with it.
        """
        kind = rex.image_kind()
        return mo.hstack([
            mo.download(
                data=lambda _r=result: rex.frame_to_csv(_r.data),
                filename=rex.chart_filename(config, dims, "csv"),
                mimetype="text/csv",
                label="⬇ Data (CSV)",
            ),
            mo.download(
                data=lambda _r=result: rex.chart_image(_r.chart)[0],
                filename=rex.chart_filename(config, dims, kind),
                mimetype="image/png" if kind == "png" else "text/html",
                label=f"⬇ Chart ({kind.upper()})",
            ),
        ], justify="start")

    return (
        AGG_OPTIONS, COMPARE_OPTIONS, MAX_CHARTS, MODE_OPTIONS,
        PAIRING_OPTIONS, default_config_for, download_buttons, file_name,
        label_for, section_card,
    )


@app.cell
def _title(mo):
    mo.md(r"""
    # Results Explorer

    Load a results file, then build as many charts as you need.
    """)
    return


@app.cell
def _source_state(mo):
    # Holds the open DuckDB connection plus the dimensions discovered from
    # it. Kept in state (rather than recomputed) so unrelated widget changes
    # never re-open the file — on a results_parquet/ directory without a
    # meta.json, that discovery is the single most expensive thing this
    # notebook does.
    get_source, set_source = mo.state(None)
    return get_source, set_source


@app.cell
def _source_path_state(mo):
    # The chosen path, shared by the browser and the text box so either can
    # set it and both stay in agreement.
    get_source_path, set_source_path = mo.state("")
    return get_source_path, set_source_path


@app.cell
def _source_browser(NOTEBOOK_DIR, mo, set_source_path):
    # Kept in its own cell (independent of the path state) so that editing the
    # text box does not recreate the widget and drop the browsed selection —
    # same reasoning as the Model Builder tab's Browse widget.
    #
    # NOTE: this is mo.ui.file_browser, not mo.ui.file. The Model Builder tab
    # uses mo.ui.file, which uploads the file's *contents* into memory and
    # (per its own comment) cannot report a real filesystem path. That is fine
    # for a small config JSON, but results files run to tens of GB — reading
    # one into memory would defeat the point of querying it in place.
    # file_browser navigates the filesystem where the notebook is running and
    # hands back a real path, which is exactly what the loader wants.
    #
    # selection_mode="file", not "directory": in directory mode, navigating
    # INTO a folder (double-clicking it, the natural thing to do) makes it
    # stop being selectable -- only its children are offered as selectable
    # items from then on, so reaching the results_parquet/ folder itself
    # means selecting it from its *parent* listing instead, which is not
    # how anyone actually uses a file browser. So this selects a file
    # *inside* the directory instead -- every results_parquet/ has a
    # `_manifest.json` (and usually a `meta.json`) sitting right in it,
    # and either's parent directory is the path we actually want.
    source_browser = mo.ui.file_browser(
        initial_path=NOTEBOOK_DIR,
        filetypes=[".json"],
        selection_mode="file",
        multiple=False,
        label="Browse into a results_parquet/ folder, then select its _manifest.json",
        on_change=lambda _files: (
            set_source_path(str(_files[0].path.parent)) if _files else None
        ),
    )
    return (source_browser,)


@app.cell
def _source_controls(mo, get_source_path, set_source_path):
    source_path = mo.ui.text(
        value=get_source_path(),
        on_change=set_source_path,
        placeholder="/path/to/results_parquet  (or use Browse above)",
        label="Or enter the path directly",
        full_width=True,
    )
    load_button = mo.ui.run_button(label="Load")
    return load_button, source_path


@app.cell
def _source_picker(load_button, mo, section_card, source_browser, source_path):
    section_card(
        "① Results file",
        "A results_parquet/ directory written by the Analysis tab's export "
        "or by run_simulation.py — browse into it above and select its "
        "_manifest.json, or type its path below.",
        mo.vstack([
            source_browser,
            source_path,
            mo.hstack([load_button], justify="start"),
        ]),
    )
    return


@app.cell
def _load_source(get_source, load_button, mo, rex, set_source, source_path):
    # Gated on the button so a partially-typed path is never opened.
    mo.stop(not load_button.value, None)
    mo.stop(
        not source_path.value.strip(),
        mo.callout(mo.md("Enter a path to a results file, then click **Load**."),
                   kind="warn"),
    )

    _previous = get_source()
    if _previous is not None:
        # Release the old attachment before opening another, so loading a
        # second file does not keep the first one's handle alive.
        try:
            _previous["con"].close()
        except Exception:
            pass
        set_source(None)

    _loaded = None
    _error = None
    with mo.status.spinner("Opening results file…") as _spinner:
        try:
            _con = rex.load_source(
                source_path.value.strip(),
                progress=lambda _msg: _spinner.update(_msg),
            )
            _spinner.update("Reading dimensions…")
            _loaded = {
                "con": _con,
                "dims": rex.discover_dims(_con),
                "path": source_path.value.strip(),
            }
        except rex.ResultsExplorerError as _exc:
            _error = str(_exc)

    mo.stop(
        _error is not None,
        mo.callout(mo.md(f"**Could not load:** {_error}"), kind="danger"),
    )
    set_source(_loaded)
    return


@app.cell
def _source_summary(file_name, get_source, mo):
    _src = get_source()
    mo.stop(_src is None, mo.md("*No results file loaded yet.*"))

    _dims = _src["dims"]
    _n_tv = sum(1 for _, _k in _dims["metrics"] if _k == "transition")
    _n_comp = len(_dims["metrics"]) - _n_tv
    _summary = mo.md(
        f"**{file_name(_src['path'])}** — "
        f"{len(_dims['scenarios'])} scenario(s), "
        f"{_n_comp} compartment(s) + {_n_tv} transition(s), "
        f"{_dims['n_reps']} replicate(s), "
        f"days {_dims['day_min']}–{_dims['day_max']}, "
        f"{_dims['num_age_groups']} age × {_dims['num_risk_groups']} risk group(s), "
        f"{len(_dims['subpops'])} subpopulation(s)."
    )
    _callout = (
        mo.callout(_summary, kind="success") if _dims["has_meta"]
        else mo.callout(
            mo.vstack([
                _summary,
                mo.md(
                    "*This file predates the `meta` table, so it carries no start "
                    "date, age-group labels or model ordering. Charts show day "
                    "numbers and age indices, and the metric list is alphabetical "
                    "rather than in the model's own compartment/transition order. "
                    "Re-exporting from the Analysis tab (or re-running "
                    "`run_simulation.py`) adds all of it.*"
                ),
            ]),
            kind="info",
        )
    )
    _callout
    return


@app.cell
def _slice_options(get_source, mo):
    # The "restrict to one slice" selectors, built once and shared by the data
    # export and by every chart — they ask the same question of the same file,
    # so they should offer exactly the same choices. "All" maps to None, which
    # the query builders read as "no restriction".
    _src = get_source()
    mo.stop(_src is None, None)
    _dims = _src["dims"]
    _age_labels = _dims.get("age_group_labels") or []

    SUBPOP_OPTIONS = {"All subpopulations": None,
                      **{_s: _s for _s in _dims.get("subpops") or []}}
    AGE_OPTIONS = {"All age groups": None,
                   **{_lbl: _i for _i, _lbl in enumerate(_age_labels)}}
    RISK_OPTIONS = {"All risk groups": None,
                    **{str(_i): _i for _i in range(_dims.get("num_risk_groups") or 0)}}

    # A selector for a dimension the model does not have (one subpopulation,
    # one risk group) would offer exactly one choice besides "All", so it is
    # left out rather than shown inert.
    HAS_SUBPOP = len(_dims.get("subpops") or []) > 1
    HAS_RISK = (_dims.get("num_risk_groups") or 1) > 1
    return AGE_OPTIONS, HAS_RISK, HAS_SUBPOP, RISK_OPTIONS, SUBPOP_OPTIONS


@app.cell
def _export_state(mo):
    # The prepared table, kept in state for the same reason the connection is:
    # mo.ui.run_button resets its value to False once the cells referencing it
    # have run, so a result computed under the click has to be stored somewhere
    # to survive past it.
    get_export, set_export = mo.state(None)
    return get_export, set_export


@app.cell
def _export_controls(
    AGE_OPTIONS, AGG_OPTIONS, HAS_RISK, HAS_SUBPOP, RISK_OPTIONS,
    SUBPOP_OPTIONS, get_source, mo, rex,
):
    _src = get_source()
    mo.stop(_src is None, None)
    _dims = _src["dims"]
    _metric_opts = [_m for _m, _ in _dims["metrics"]]
    _day_lo = int(_dims["day_min"] or 1)
    _day_hi = int(_dims["day_max"] or 1)

    export_scenarios = mo.ui.multiselect(
        options=list(_dims["scenarios"]), value=list(_dims["scenarios"]),
        label="Scenarios")
    # Metric order is the meta table's compartment order then transition order,
    # i.e. the model's own — do not sort it.
    export_metrics = mo.ui.multiselect(
        options=_metric_opts, value=_metric_opts,
        label="Compartments / transition variables")
    export_agg = mo.ui.dropdown(
        options=AGG_OPTIONS, value="Population total", label="Aggregation")
    export_subpop = mo.ui.dropdown(
        options=SUBPOP_OPTIONS, value="All subpopulations", label="Subpopulation")
    export_age = mo.ui.dropdown(
        options=AGE_OPTIONS, value="All age groups", label="Age group")
    export_risk = mo.ui.dropdown(
        options=RISK_OPTIONS, value="All risk groups", label="Risk group")
    export_days = mo.ui.range_slider(
        start=_day_lo, stop=max(_day_hi, _day_lo + 1),
        value=[_day_lo, _day_hi], label="Days", show_value=False)
    export_rows = mo.ui.radio(
        options={"Summary across replicates": "summary",
                 "One row per replicate": "replicates"},
        value="Summary across replicates", label="Rows")
    export_time = mo.ui.radio(
        options={"One row per day": "daily",
                 "One row for the whole day range": "total"},
        value="One row per day", label="Time")
    export_cumulative = mo.ui.checkbox(
        label="Cumulative (transition variables)")
    export_total = mo.ui.checkbox(
        label=f"Add a combined '{rex.TOTAL_LABEL}' row")
    export_button = mo.ui.run_button(label="Prepare table")
    return (
        export_age, export_agg, export_button, export_cumulative, export_days,
        export_metrics, export_risk, export_rows, export_scenarios,
        export_subpop, export_time, export_total,
    )


@app.cell
def _export_form(
    HAS_RISK, HAS_SUBPOP, export_age, export_agg, export_button,
    export_cumulative, export_days, export_metrics, export_risk, export_rows,
    export_scenarios, export_subpop, export_time, export_total, get_source, mo,
    rex, section_card,
):
    mo.stop(get_source() is None, None)
    _lo, _hi = export_days.value
    _slice_row = ([export_subpop] if HAS_SUBPOP else []) + [export_age]
    if HAS_RISK:
        _slice_row.append(export_risk)
    # Offered only where it means something: at population level the single
    # row already is the total.
    _flags = [export_cumulative]
    if rex.supports_export_total({"agg_level": export_agg.value}):
        _flags.append(export_total)
    section_card(
        "② Download data",
        "Pull the numbers out as a CSV, at whatever grain you need — this is "
        "independent of the charts below.",
        mo.vstack([
            export_scenarios,
            export_metrics,
            mo.hstack([export_agg, *_slice_row], justify="start", wrap=True),
            mo.hstack([export_days, mo.md(f"**{_lo} – {_hi}**")],
                      justify="start", wrap=True),
            mo.hstack([export_rows, export_time], justify="start", wrap=True),
            mo.hstack(_flags, justify="start", wrap=True),
            mo.hstack([export_button], justify="start"),
        ]),
    )
    return


@app.cell
def _export_run(
    HAS_RISK, HAS_SUBPOP, export_age, export_agg, export_button,
    export_cumulative, export_days, export_metrics, export_risk, export_rows,
    export_scenarios, export_subpop, export_time, export_total, get_source, mo,
    rex, set_export,
):
    # Gated on the button: the widest selection here can be a million rows, so
    # it must not run on every twiddle of a selector.
    mo.stop(not export_button.value, None)
    _src = get_source()
    mo.stop(_src is None, None)

    # Dropdowns built from a {label: value} dict take the *label* as `value=`
    # but read back the mapped value, so these are used as-is.
    _export_cfg = {
        "scenarios": list(export_scenarios.value or []),
        "metrics": list(export_metrics.value or []),
        "agg_level": export_agg.value,
        "subpop_filter": export_subpop.value if HAS_SUBPOP else None,
        "age_filter": export_age.value,
        "risk_filter": export_risk.value if HAS_RISK else None,
        "day_range": tuple(export_days.value),
        "cumulative": bool(export_cumulative.value),
        "show_total": bool(export_total.value),
        "row_mode": export_rows.value,
        "time_mode": export_time.value,
    }
    with mo.status.spinner("Building table…"):
        try:
            set_export({
                "df": rex.run_export(_src["con"], _src["dims"], _export_cfg),
                "config": _export_cfg,
                "error": None,
            })
        except rex.ResultsExplorerError as _exc:
            set_export({"df": None, "config": _export_cfg, "error": str(_exc)})
    return


@app.cell
def _export_result(get_export, get_source, mo, rex):
    # Deliberately separate from the form cell: this one depends only on the
    # prepared table, so editing a selector does not re-encode the last CSV.
    mo.stop(get_source() is None, None)
    _state = get_export()
    mo.stop(
        _state is None,
        mo.md("*Choose what you need above, then click **Prepare table**.*"),
    )
    mo.stop(
        _state["error"] is not None,
        mo.callout(mo.md(_state["error"] or ""), kind="warn"),
    )

    _df = _state["df"]
    _cfg = _state["config"]
    _preview_rows = 200
    _notes = []
    if _cfg["cumulative"] and _cfg["time_mode"] != "daily":
        _notes.append(
            "*Cumulative is ignored here: the days are already summed into one "
            "row, so a running total over them has nothing left to run over.*"
        )
    if _cfg["time_mode"] == "total":
        _notes.append(
            "*Each row sums its value over the selected days. That is the "
            "seasonal total for a transition variable (a daily flow); for a "
            "compartment, which is already a level, it is a sum of levels and "
            "rarely what you want.*"
        )
    if _cfg.get("show_total") and rex.supports_export_total(_cfg):
        _groups = " × ".join(
            _c.replace("_", " ") for _c in rex.group_columns(_cfg))
        _notes.append(
            f"*Rows marked **{rex.TOTAL_LABEL}** under `{_groups}` sum every "
            f"group — summed per replicate before any median or percentile, so "
            f"each is the total in its own right and not the total of the other "
            f"rows' summaries. **They overlap the rows they sum**, so filter "
            f"them out before totalling a column yourself.*"
        )
    if _cfg["row_mode"] == "summary":
        _notes.append(
            "*Summary columns are taken across replicates, per row — so "
            "`median_value` is the median of the replicate values for that "
            "scenario/metric/day, not a median of anything already summarized.*"
        )
    if len(_df) > _preview_rows:
        _notes.append(
            f"*Preview shows the first {_preview_rows:,} of "
            f"{len(_df):,} rows — the download has all of them.*"
        )

    mo.vstack([
        mo.hstack([
            mo.md(f"**{len(_df):,} rows × {len(_df.columns)} columns**"),
            mo.download(
                data=rex.frame_to_csv(_df),
                filename=rex.export_filename(_cfg),
                mimetype="text/csv",
                label="⬇ Download CSV",
            ),
        ], justify="start"),
        mo.ui.table(_df.head(_preview_rows), selection=None, pagination=True),
        *[mo.md(_n) for _n in _notes],
    ]).style({
        "border": "1px solid #e3e3e3",
        "border-radius": "8px",
        "padding": "0.8rem 1rem",
        "margin-bottom": "1rem",
    })
    return


@app.cell
def _charts_state(mo):
    # Two pieces of state, deliberately separate:
    #
    #   chart_specs  — the structural list: which charts exist, in order, and
    #                  what type each is. Only Add/Remove touch it.
    #   chart_values — every per-chart setting, keyed "field::<chart id>".
    #                  Keying by id (not slot) means removing a chart never
    #                  shifts anyone else's settings.
    get_chart_specs, set_chart_specs = mo.state([])
    get_chart_values, set_chart_values = mo.state({})
    get_next_id, set_next_id = mo.state(0)
    return (
        get_chart_specs, get_chart_values, get_next_id,
        set_chart_specs, set_chart_values, set_next_id,
    )


@app.cell
def _add_chart_controls(get_source, mo):
    mo.stop(get_source() is None, None)
    new_chart_type = mo.ui.dropdown(
        options={
            "Time series": "timeseries",
            "Histogram": "histogram",
            "Box plot": "boxplot",
            "Stacked bar": "stacked_bar",
            "Scatter": "scatter",
        },
        value="Time series",
        label="Type",
    )
    add_chart_button = mo.ui.run_button(label="+ Add chart")
    return add_chart_button, new_chart_type


@app.cell
def _add_chart_ui(
    MAX_CHARTS, add_chart_button, get_chart_specs, get_source, mo,
    new_chart_type, section_card,
):
    mo.stop(get_source() is None, None)
    _full = len(get_chart_specs()) >= MAX_CHARTS
    section_card(
        "③ Charts",
        "Add as many as you need — each keeps its own settings. "
        "Remove one with the ✕ on its card.",
        mo.vstack([
            mo.hstack([new_chart_type, add_chart_button], justify="start"),
            *([mo.md(f"*Maximum of {MAX_CHARTS} charts reached — remove one to add another.*")]
              if _full else []),
        ]),
    )
    return


@app.cell
def _add_chart(
    MAX_CHARTS, add_chart_button, get_chart_specs, get_next_id, get_source,
    mo, new_chart_type, set_chart_specs, set_next_id,
):
    mo.stop(not add_chart_button.value, None)
    mo.stop(get_source() is None, None)
    mo.stop(len(get_chart_specs()) >= MAX_CHARTS, None)

    _spec = {"id": get_next_id(), "chart_type": new_chart_type.value}
    # Callable form: a mo.state setter does not reflect through its getter
    # within the same reactive pass, and it applies atomically against the
    # current value — so two quick edits cannot race each other.
    set_chart_specs(lambda _cs: _cs + [_spec])
    set_next_id(lambda _n: _n + 1)
    return


@app.cell
def _chart_widgets(
    AGE_OPTIONS, AGG_OPTIONS, COMPARE_OPTIONS, MAX_CHARTS, MODE_OPTIONS,
    PAIRING_OPTIONS, RISK_OPTIONS, SUBPOP_OPTIONS, default_config_for,
    get_chart_specs, get_chart_values, get_source, label_for, mo,
    set_chart_specs, set_chart_values,
):
    # Every control for every chart slot is built here, bound to a module-level
    # name, and wrapped in mo.ui.array. That combination is what makes them
    # interactive: marimo can only route a browser interaction back to a UI
    # element it can address, which means elements reachable from a global —
    # an element merely constructed inside another cell's output is rendered
    # but inert. This mirrors _nb_analysis.py's scenario controls, which build
    # a fixed MAX_SC-sized mo.ui.array for exactly the same reason.
    #
    # Hence a fixed MAX_CHARTS slots rather than one widget per existing
    # chart: the array's shape must not depend on how many charts there are,
    # or it would be rebuilt mid-interaction. Slots beyond the current chart
    # count are built but never displayed.
    _src = get_source()
    mo.stop(_src is None, None)
    _dims = _src["dims"]
    _specs = get_chart_specs()
    _st = get_chart_values()
    # Metric order comes straight from dims["metrics"], which is the meta
    # table's compartment order followed by its transition order — i.e. the
    # same order the Model Builder's own selectors use. Do not sort it.
    _metric_opts = [_m for _m, _ in _dims["metrics"]]
    _scenario_opts = list(_dims["scenarios"])
    _day_lo = int(_dims["day_min"] or 1)
    _day_hi = int(_dims["day_max"] or 1)

    # Slice selectors come from _slice_options; they are independent of the
    # Aggregation dropdown, since pinning one age group while still showing the
    # population total across risk groups and subpops is a normal thing to want.
    def _cb(key):
        def _inner(v):
            set_chart_values(lambda d: {**d, key: v})
        return _inner

    def _remove(chart_id):
        def _inner(_v):
            # Values keyed by id are left behind harmlessly; ids are never
            # reused, so they can never be picked up by a later chart.
            set_chart_specs(lambda _cs: [_c for _c in _cs if _c["id"] != chart_id])
        return _inner

    def _slot(j):
        """(chart id, default config) for slot j — placeholders past the end."""
        if j < len(_specs):
            _spec = _specs[j]
            return _spec["id"], default_config_for(_spec, _dims)
        return -1 - j, default_config_for({"id": -1 - j, "chart_type": "timeseries"}, _dims)

    def _val(j, field, default):
        _cid, _ = _slot(j)
        return _st.get(f"{field}::{_cid}", default)

    chart_metrics = mo.ui.array([
        mo.ui.multiselect(
            options=_metric_opts,
            value=[_m for _m in _val(j, "metrics", _slot(j)[1]["metrics"])
                   if _m in _metric_opts],
            label="Metric(s)", on_change=_cb(f"metrics::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_agg = mo.ui.array([
        mo.ui.dropdown(
            options=AGG_OPTIONS,
            value=label_for(AGG_OPTIONS,
                            _val(j, "agg_level", _slot(j)[1]["agg_level"])),
            label="Aggregation", on_change=_cb(f"agg_level::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_mode = mo.ui.array([
        mo.ui.dropdown(
            options=MODE_OPTIONS,
            value=label_for(MODE_OPTIONS,
                            _val(j, "scenario_mode", _slot(j)[1]["scenario_mode"])),
            label="Scenarios", on_change=_cb(f"scenario_mode::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_scenarios = mo.ui.array([
        mo.ui.multiselect(
            options=_scenario_opts,
            value=[_s for _s in _val(j, "scenarios", _slot(j)[1]["scenarios"])
                   if _s in _scenario_opts],
            label="Select", on_change=_cb(f"scenarios::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_baseline = mo.ui.array([
        mo.ui.dropdown(
            options=_scenario_opts,
            value=_val(j, "baseline_scenario", _slot(j)[1]["baseline_scenario"]),
            label="Baseline", on_change=_cb(f"baseline_scenario::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_compare = mo.ui.array([
        mo.ui.dropdown(
            options=COMPARE_OPTIONS,
            value=label_for(COMPARE_OPTIONS,
                            _val(j, "compare_metric", _slot(j)[1]["compare_metric"])),
            label="As", on_change=_cb(f"compare_metric::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_pairing = mo.ui.array([
        mo.ui.dropdown(
            options=PAIRING_OPTIONS,
            value=label_for(PAIRING_OPTIONS,
                            _val(j, "pairing", _slot(j)[1]["pairing"])),
            label="Pairing", on_change=_cb(f"pairing::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_ci = mo.ui.array([
        mo.ui.checkbox(
            value=bool(_val(j, "show_uncertainty", _slot(j)[1]["show_uncertainty"])),
            label="95% interval", on_change=_cb(f"show_uncertainty::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_subpop = mo.ui.array([
        mo.ui.dropdown(
            options=SUBPOP_OPTIONS,
            value=label_for(SUBPOP_OPTIONS, _val(j, "subpop_filter", None)),
            label="Subpopulation", on_change=_cb(f"subpop_filter::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_age = mo.ui.array([
        mo.ui.dropdown(
            options=AGE_OPTIONS,
            value=label_for(AGE_OPTIONS, _val(j, "age_filter", None)),
            label="Age group", on_change=_cb(f"age_filter::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_risk = mo.ui.array([
        mo.ui.dropdown(
            options=RISK_OPTIONS,
            value=label_for(RISK_OPTIONS, _val(j, "risk_filter", None)),
            label="Risk group", on_change=_cb(f"risk_filter::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_cumulative = mo.ui.array([
        mo.ui.checkbox(
            value=bool(_val(j, "cumulative", False)),
            label="Cumulative", on_change=_cb(f"cumulative::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_total = mo.ui.array([
        mo.ui.checkbox(
            value=bool(_val(j, "show_total", True)),
            label="Total panel", on_change=_cb(f"show_total::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_hide_empty = mo.ui.array([
        mo.ui.checkbox(
            value=bool(_val(j, "hide_empty", True)),
            label="Hide empty groups", on_change=_cb(f"hide_empty::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_shared_y = mo.ui.array([
        mo.ui.checkbox(
            value=bool(_val(j, "shared_y", True)),
            label="Shared y axis", on_change=_cb(f"shared_y::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_days = mo.ui.array([
        mo.ui.range_slider(
            start=_day_lo, stop=max(_day_hi, _day_lo + 1),
            value=list(_val(j, "day_range", (_day_lo, _day_hi))),
            label="Days",
            # The built-in readout prints a bare tuple, so "1 to 250" comes out
            # as "1, 250" and reads as the number 1,250. _render_charts writes
            # its own "start – end" caption instead.
            show_value=False,
            on_change=_cb(f"day_range::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_x = mo.ui.array([
        mo.ui.dropdown(
            options=_metric_opts,
            value=_val(j, "scatter_x", _slot(j)[1]["scatter_x"]),
            label="X metric", on_change=_cb(f"scatter_x::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_y = mo.ui.array([
        mo.ui.dropdown(
            options=_metric_opts,
            value=_val(j, "scatter_y", _slot(j)[1]["scatter_y"]),
            label="Y metric", on_change=_cb(f"scatter_y::{_slot(j)[0]}"),
        )
        for j in range(MAX_CHARTS)
    ])
    chart_remove = mo.ui.array([
        mo.ui.button(label="✕", on_change=_remove(_slot(j)[0]))
        for j in range(MAX_CHARTS)
    ])
    return (
        chart_age, chart_agg, chart_baseline, chart_ci, chart_compare,
        chart_cumulative, chart_days, chart_hide_empty, chart_metrics,
        chart_mode, chart_pairing, chart_remove, chart_risk, chart_scenarios,
        chart_shared_y, chart_subpop, chart_total, chart_x, chart_y,
    )


@app.cell
def _render_charts(
    HAS_RISK, HAS_SUBPOP, chart_age, chart_agg, chart_baseline, chart_ci,
    chart_compare, chart_cumulative, chart_days, chart_hide_empty,
    chart_metrics, chart_mode, chart_pairing, chart_remove, chart_risk,
    chart_scenarios, chart_shared_y, chart_subpop, chart_total, chart_x,
    chart_y, default_config_for, download_buttons, get_chart_specs,
    get_source, mo, rex, traceback,
):
    _src = get_source()
    mo.stop(_src is None, mo.md(""))
    _specs = get_chart_specs()
    mo.stop(
        not _specs,
        mo.callout(mo.md("No charts yet — add one above."), kind="info"),
    )
    _con, _dims = _src["con"], _src["dims"]
    _has_subpop, _has_risk = HAS_SUBPOP, HAS_RISK
    _kinds = dict(_dims.get("metrics") or [])

    def _config(j, spec):
        # The widgets are the source of truth for settings; defaults only fill
        # in what the user has not touched.
        #
        # Note the asymmetry in mo.ui.dropdown when options is a {label: value}
        # dict: it takes the *label* as its `value=` argument (hence label_for
        # when the widgets are built), but `.value` reads back the mapped
        # *value*. So these are used directly — mapping them again would
        # KeyError on the already-mapped result.
        _cfg = default_config_for(spec, _dims)
        _cfg.update(
            metrics=list(chart_metrics[j].value or []),
            agg_level=chart_agg[j].value,
            scenario_mode=chart_mode[j].value,
            scenarios=list(chart_scenarios[j].value or []),
            baseline_scenario=chart_baseline[j].value,
            compare_metric=chart_compare[j].value,
            pairing=chart_pairing[j].value,
            show_uncertainty=bool(chart_ci[j].value),
            day_range=tuple(chart_days[j].value),
            scatter_x=chart_x[j].value,
            scatter_y=chart_y[j].value,
            subpop_filter=chart_subpop[j].value if _has_subpop else None,
            age_filter=chart_age[j].value,
            risk_filter=chart_risk[j].value if _has_risk else None,
            cumulative=bool(chart_cumulative[j].value),
            show_total=bool(chart_total[j].value),
            hide_empty=bool(chart_hide_empty[j].value),
            shared_y=bool(chart_shared_y[j].value),
        )
        return _cfg

    _blocks = []
    for _j, _spec in enumerate(_specs):
        _cfg = _config(_j, _spec)
        _kind = _cfg["chart_type"]

        _row1 = ([chart_x[_j], chart_y[_j]] if _kind == "scatter"
                 else [chart_metrics[_j]]) + [chart_agg[_j]]
        # Slice selectors sit next to Aggregation because they answer the
        # neighbouring question: aggregation says how to break the data down,
        # these say which part of it to look at in the first place.
        _row_slice = ([chart_subpop[_j]] if _has_subpop else []) + [chart_age[_j]]
        if _has_risk:
            _row_slice.append(chart_risk[_j])
        # These only bite once the chart is broken into small multiples.
        if _cfg["agg_level"] != "population":
            _row_slice += [chart_hide_empty[_j], chart_shared_y[_j]]
        # Offered only where it means something — see rex.supports_total.
        if rex.supports_total(_cfg):
            _row_slice.append(chart_total[_j])
        _row2 = [chart_mode[_j], chart_scenarios[_j]]
        if rex.is_comparison(_cfg):
            _row2 += [chart_baseline[_j], chart_compare[_j], chart_pairing[_j]]
        _lo, _hi = _cfg["day_range"]
        _row3 = [chart_days[_j], mo.md(f"**{_lo} – {_hi}**")]
        if _kind == "timeseries" and (_dims.get("n_reps") or 1) > 1:
            _row3.insert(0, chart_ci[_j])
        # Cumulative is a running total of a daily flow, so it is offered only
        # for time series and only while every selected metric is a transition
        # variable — see rex._validate for why a compartment cannot be summed
        # over days.
        if _kind == "timeseries" and _cfg["metrics"] and all(
            _kinds.get(_m) == "transition" for _m in _cfg["metrics"]
        ):
            _row3.insert(0, chart_cumulative[_j])

        # No downloads when the chart did not build: there would be nothing
        # behind them but the error message.
        _downloads = None
        try:
            _result = rex.build_chart(_con, _dims, _cfg)
            _downloads = download_buttons(_result, _cfg, _dims)
            try:
                _view = mo.ui.altair_chart(
                    _result.chart, chart_selection=False, legend_selection=False)
            except Exception:
                # Faceted charts cannot carry marimo's selection wrapper; show
                # the chart itself rather than losing it.
                _view = _result.chart
        except rex.ResultsExplorerError as _exc:
            _view = mo.callout(mo.md(str(_exc)), kind="warn")
        except Exception:
            _view = mo.callout(
                mo.md(f"```\n{traceback.format_exc()}\n```"), kind="danger")

        _extra = []
        if _kind != "scatter" and len(_cfg["metrics"]) > 1:
            _extra.append(mo.md(
                f"*Metrics are **summed**: this shows "
                f"{' + '.join(_cfg['metrics'])} as one series.*"
            ))
        if rex.supports_total(_cfg) and _cfg.get("show_total"):
            _extra.append(mo.md(
                f"*The last panel, **{rex.TOTAL_LABEL}**, sums every group — "
                f"it is summed per replicate before any median, interval or "
                f"baseline comparison, so it is the total in its own right and "
                f"not the sum of the other panels' summaries.*"
            ))
        if _cfg.get("cumulative"):
            _extra.append(mo.md(
                "*Cumulative totals run from the start of the selected day "
                "range, and are accumulated per replicate before the median "
                "and interval are taken.*"
            ))
        if rex.is_comparison(_cfg):
            # The axis says "difference from X", but it is worth stating that
            # the baseline is gone from the chart rather than leaving the user
            # to notice a missing box.
            _extra.append(mo.md(
                f"*Values are relative to **{_cfg['baseline_scenario']}**, "
                f"which is therefore not drawn — comparing it with itself "
                f"gives {'1' if _cfg.get('compare_metric') == 'ratio' else '0'} "
                f"everywhere. The dashed line marks no effect.*"
            ))
            if _cfg.get("pairing", "paired") == "paired":
                _extra.append(mo.md(
                    "*Paired by replicate index — assumes replicate i of each "
                    "scenario corresponds to replicate i of the baseline. That "
                    "holds when replicates differ only by transition RNG (runs "
                    "are seeded by position), and less so once parameter "
                    "sampling varies which draw backs each index. Switch to "
                    "Unpaired if in doubt.*"
                ))
            elif _kind in ("boxplot", "histogram", "scatter"):
                # Worth spelling out here specifically: unpaired turns the
                # spread into the compared scenario's own, which is easy to
                # misread as uncertainty about the effect.
                _extra.append(mo.md(
                    "*Unpaired — every replicate is measured against a single "
                    "baseline summary, so the spread shown is the compared "
                    "scenario's own variability, not the uncertainty in the "
                    "effect. Pairing is what narrows this to the effect itself.*"
                ))

        _blocks.append(
            mo.vstack([
                mo.hstack([
                    mo.md(f"**{rex.chart_title(_cfg, _dims)}**"),
                    chart_remove[_j],
                ], justify="space-between"),
                mo.hstack(_row1, justify="start", wrap=True),
                mo.hstack(_row_slice, justify="start", wrap=True),
                mo.hstack(_row2, justify="start", wrap=True),
                mo.hstack(_row3, justify="start", wrap=True),
                _view,
                *([_downloads] if _downloads is not None else []),
                *_extra,
            ]).style({
                "border": "1px solid #e3e3e3",
                "border-radius": "8px",
                "padding": "0.8rem 1rem",
                "margin-bottom": "1rem",
            })
        )

    mo.vstack(_blocks)
    return


if __name__ == "__main__":
    app.run()
