"""
_results_explorer_lib.py
========================

Query and charting helpers behind ``results_explorer_notebook.py``.

Deliberately free of any ``marimo`` import: everything here is plain Python
over a DuckDB connection, so it can be exercised by unit tests without
booting a notebook kernel. The notebook file holds only UI wiring.

Data sources
------------
Both halves of the pipeline write the same tidy row schema:

``results``
    scenario, rep, param_set, compartment, kind, day, value
``results_full``
    the same plus subpop, age_group, risk_group

``results`` is already summed over subpop/age/risk, so population-level
charts read it directly and never touch the (much larger) ``results_full``.
``kind`` is ``"compartment"`` or ``"transition"``; ``day`` is 1-based;
``age_group``/``risk_group`` are 0-based indices.

Newer files also carry a ``meta`` key/value table (start date, age-group
labels, scenario order, ...). It is optional -- files written before it
existed still work, they just fall back to index-based axes and labels.

Why DuckDB rather than loading into pandas: real result sets reach tens of
GB (one MA_vax example ``results.db`` is 57 GB, with 73M rows in ``results``
alone). DuckDB attaches the SQLite file and pushes filters and aggregation
down into it, so only the aggregated chart data is ever materialized.
"""

from __future__ import annotations

import json
import re
import sqlite3
import textwrap
from pathlib import Path
from typing import Any, Literal, NamedTuple, Sequence

import altair as alt
import duckdb
import pandas as pd

#: Builders return whichever Altair wrapper the chart needed -- a plain
#: Chart, a LayerChart (median + interval band), or a FacetChart (small
#: multiples) -- so the shared return type is their common base.
AltairChart = alt.TopLevelMixin

# Charts here are aggregated in DuckDB before Altair ever sees them, so the
# frames handed over are thousands of rows, not millions -- but a wide
# selection (long day range x several scenarios x faceted age groups) does
# cross Altair's default 5000-row cap, which otherwise raises MaxRowsError.
# Lifting the cap is the right call precisely because the aggregation already
# happened; SOFT_ROW_LIMIT below is what actually protects the browser.
alt.data_transformers.enable("default", max_rows=None)

#: Row count past which a chart is refused with a suggestion to narrow the
#: selection. Well above anything a sensible chart needs, low enough that the
#: payload embedded in the page stays a few MB.
SOFT_ROW_LIMIT = 50_000

#: Row cap for the bulk data export, which is downloaded rather than drawn and
#: so can be far larger than a chart -- but is still materialized in memory as
#: a DataFrame before being written out, hence a cap at all.
EXPORT_ROW_LIMIT = 1_000_000

CHART_TYPES = ("timeseries", "histogram", "boxplot", "stacked_bar", "scatter")

AGG_LEVELS = ("population", "subpop", "age_group", "risk_group", "all")

#: How a chart relates its scenarios to each other. Only two modes, because
#: only two behaviours differ: plot the values as they are, or plot them
#: relative to a baseline. *How many* scenarios are shown is decided entirely
#: by ``config["scenarios"]``, so a mode claiming to control that would imply a
#: constraint nothing enforces. (Earlier versions offered "single"/"multiple",
#: which were byte-for-byte identical; is_comparison treats any unrecognized
#: value as "levels", so a config saved with either still opens.)
SCENARIO_MODES = ("levels", "compare_baseline")

#: agg_level -> the columns a query groups by (beyond scenario/day). The
#: "population" level is special: it reads the pre-aggregated `results` table
#: instead of summing `results_full` back down.
_AGG_GROUP_COLS: dict[str, tuple[str, ...]] = {
    "population": (),
    "subpop": ("subpop",),
    "age_group": ("age_group",),
    "risk_group": ("risk_group",),
    "all": ("subpop", "age_group", "risk_group"),
}

#: (column, config key) for the "restrict to one slice" selectors. These are
#: independent of agg_level: pinning a single age group and still aggregating
#: over the population total is a meaningful (and common) request, so a filter
#: narrows *which rows are summed* rather than changing what is grouped on.
_FILTER_COLS: tuple[tuple[str, str], ...] = (
    ("subpop", "subpop_filter"),
    ("age_group", "age_filter"),
    ("risk_group", "risk_filter"),
)

#: Name shown for the extra "every group combined" slice.
TOTAL_LABEL = "Total"

#: The combined slice is carried as an ordinary value *inside* the group
#: column rather than as a separate result set, so every downstream join,
#: GROUP BY and baseline comparison treats it exactly like a real group and
#: needs no special case. That requires a value the data cannot itself
#: contain, and one of the right SQL type -- the `*_group` columns are
#: INTEGER, subpop is TEXT.
_TOTAL_SQL_LITERAL: dict[str, str] = {
    "subpop": "'__total__'", "age_group": "-1", "risk_group": "-1",
}
_TOTAL_VALUE: dict[str, Any] = {
    "subpop": "__total__", "age_group": -1, "risk_group": -1,
}


#: Columns that exist only so a chart can render -- a wrapped axis label, a
#: synthesized legend key. They are presentation, not results, so they are
#: stripped before the chart's data is handed to a download button.
_DISPLAY_ONLY_COLS = ("scenario_label", "_series")


class ChartResult(NamedTuple):
    """A built chart together with the rows it was built from.

    The two travel together because the data is not recoverable from the chart:
    by the time Altair has it, the frame has been filtered, relabelled and (for
    a comparison) differenced against the baseline. Re-running the query to
    offer a CSV would be both wasteful and, if it diverged, wrong -- what the
    user downloads has to be what they are looking at.
    """

    chart: AltairChart
    data: pd.DataFrame


class ResultsExplorerError(Exception):
    """A chart config that cannot be rendered as given (bad or incomplete).

    Raised rather than returning None so the notebook can show the message
    inline against the offending chart instead of failing the whole cell.
    """


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def load_source(path: str | Path, *, progress=None) -> duckdb.DuckDBPyConnection:
    """Open a results file and return a connection exposing ``results``,
    ``results_full`` and (when present) ``meta``.

    ``path`` may be a SQLite ``.db`` -- attached read-only and queried in
    place -- or a legacy ``.json`` export, which is converted once to a
    sibling ``.db`` and then attached (see :func:`convert_json_to_sqlite`).

    ``progress`` is an optional ``callable(str)`` used to report conversion
    steps, since that path can take a while on a multi-GB file.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ResultsExplorerError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        db_path = path.with_suffix(".db")
        if not db_path.exists():
            if progress:
                progress(f"Converting {path.name} to SQLite (one time)…")
            convert_json_to_sqlite(path, db_path, progress=progress)
        path = db_path
    elif suffix not in _SQLITE_SUFFIXES:
        raise ResultsExplorerError(
            f"Unrecognized results file type '{suffix}'. Expected a SQLite "
            f".db (from the Analysis tab's SQLite export or run_simulation.py) "
            f"or a legacy .json export."
        )

    con = duckdb.connect()
    con.execute("INSTALL sqlite_scanner")
    con.execute("LOAD sqlite_scanner")
    # READ_ONLY so opening a results file in the explorer can never modify
    # it -- these are expensive-to-regenerate simulation outputs.
    con.execute(f"ATTACH '{path.as_posix()}' AS src (TYPE sqlite, READ_ONLY)")

    # An ATTACHed database is a *catalog* in DuckDB, so it is table_catalog
    # that carries the alias here -- table_schema is the sqlite file's own
    # 'main'.
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'"
    ).fetchall()}
    missing = {"results", "results_full"} - tables
    if missing:
        con.close()
        raise ResultsExplorerError(
            f"{path.name} is missing table(s) {sorted(missing)}. Expected a "
            f"results database written by the Analysis tab's SQLite export or "
            f"by run_simulation.py."
        )

    for table in ("results", "results_full"):
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM src.{table}")
    if "meta" in tables:
        con.execute("CREATE VIEW meta AS SELECT * FROM src.meta")
    return con


def convert_json_to_sqlite(
    json_path: str | Path, db_path: str | Path, *, progress=None,
    batch_size: int = 100_000,
) -> Path:
    """Stream a legacy ``.json`` export into a SQLite ``.db``.

    The JSON export is one large object holding ``results``/``results_full``
    as arrays of row-arrays. Parsing it with ``json.load`` costs roughly 8x
    the file size in memory (~12 GB for the 1.4 GB MA_vax example), so this
    streams with ``ijson`` and inserts in batches, keeping peak memory flat.

    ``ijson`` is only needed on this legacy path; if it is unavailable the
    caller gets an actionable error rather than a MemoryError halfway through.
    """
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ResultsExplorerError(
            "Converting a legacy .json export needs the 'ijson' package "
            "(pip install ijson). Alternatively, re-export the results as "
            "SQLite from the Analysis tab, which needs no conversion."
        ) from exc

    json_path, db_path = Path(json_path), Path(db_path)
    tmp_path = db_path.with_name(db_path.name + ".partial")
    tmp_path.unlink(missing_ok=True)

    con = sqlite3.connect(tmp_path)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute(
        "CREATE TABLE results "
        "(scenario TEXT, rep INTEGER, param_set INTEGER, compartment TEXT, kind TEXT, "
        "day INTEGER, value REAL)"
    )
    con.execute(
        "CREATE TABLE results_full "
        "(scenario TEXT, rep INTEGER, param_set INTEGER, compartment TEXT, kind TEXT, "
        "subpop TEXT, age_group INTEGER, risk_group INTEGER, day INTEGER, value REAL)"
    )

    try:
        for table, n_cols in (("results", 7), ("results_full", 10)):
            placeholders = ",".join("?" * n_cols)
            insert = f"INSERT INTO {table} VALUES ({placeholders})"
            batch: list[Sequence[Any]] = []
            total = 0
            with json_path.open("rb") as fh:
                # use_float: ijson decodes JSON numbers to Decimal by default,
                # which sqlite3 refuses to bind. These are simulation outputs,
                # so float is the correct (and original) representation.
                for row in ijson.items(fh, f"{table}.item", use_float=True):
                    batch.append(row)
                    if len(batch) >= batch_size:
                        con.executemany(insert, batch)
                        total += len(batch)
                        batch.clear()
                        if progress:
                            progress(f"{table}: {total:,} rows converted…")
            if batch:
                con.executemany(insert, batch)
                total += len(batch)
            if progress:
                progress(f"{table}: {total:,} rows converted.")

        con.execute(
            "CREATE INDEX idx_results_scenario_compartment "
            "ON results (scenario, compartment)"
        )
        con.execute(
            "CREATE INDEX idx_results_full_scenario_compartment "
            "ON results_full (scenario, compartment)"
        )
        con.commit()
    finally:
        con.close()

    # Only claim the real name once the conversion fully succeeded, so an
    # interrupted run doesn't leave a truncated .db that later looks cached.
    tmp_path.replace(db_path)
    for stale in tmp_path.parent.glob(tmp_path.name + "-*"):
        stale.unlink(missing_ok=True)
    return db_path


# ---------------------------------------------------------------------------
# Metadata and dimension discovery
# ---------------------------------------------------------------------------


def read_meta(con: duckdb.DuckDBPyConnection) -> dict[str, Any] | None:
    """Return the ``meta`` table as a dict, or None for files without one."""
    try:
        rows = con.execute("SELECT key, value FROM meta").fetchall()
    except duckdb.Error:
        return None
    out: dict[str, Any] = {}
    for key, value in rows:
        try:
            out[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            out[key] = value
    return out or None


def discover_dims(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Everything the UI needs to build its selectors.

    Served from ``meta`` when the file has it (one tiny read regardless of
    file size). Otherwise recovered with DISTINCT/MIN/MAX scans, which on a
    large file are slow -- seconds over ``results``, tens of seconds over
    ``results_full`` -- hence ``has_meta`` in the result, so the UI can say
    why it is waiting and what re-exporting would buy.
    """
    meta = read_meta(con)
    dims: dict[str, Any] = {"has_meta": meta is not None, "meta": meta or {}}

    if meta:
        dims["scenarios"] = list(meta.get("scenarios") or [])
        dims["subpops"] = list(meta.get("subpop_names") or [])
        dims["start_date"] = meta.get("start_date")
        dims["age_group_labels"] = list(meta.get("age_group_labels") or [])
        dims["num_age_groups"] = meta.get("num_age_groups")
        dims["num_risk_groups"] = meta.get("num_risk_groups")
        comps = list(meta.get("compartments") or [])
        tvs = list(meta.get("transition_vars") or [])
        dims["metrics"] = (
            [(c, "compartment") for c in comps] + [(t, "transition") for t in tvs]
        )
        dims["n_reps"] = meta.get("n_reps")
        dims["uncertainty_source"] = meta.get("uncertainty_source")

    # Anything meta didn't supply (older file, or a key added after it was
    # written) falls back to querying. Each block is guarded so a partial
    # meta still avoids the scans it does cover.
    if not dims.get("scenarios"):
        dims["scenarios"] = [
            r[0] for r in con.execute(
                "SELECT DISTINCT scenario FROM results ORDER BY 1").fetchall()
        ]
    if not dims.get("metrics"):
        dims["metrics"] = [
            (r[0], r[1]) for r in con.execute(
                "SELECT DISTINCT compartment, kind FROM results ORDER BY 2, 1"
            ).fetchall()
        ]
    if not dims.get("subpops"):
        dims["subpops"] = [
            r[0] for r in con.execute(
                "SELECT DISTINCT subpop FROM results_full ORDER BY 1").fetchall()
        ]
    if dims.get("num_age_groups") is None or dims.get("num_risk_groups") is None:
        n_age, n_risk = con.execute(
            "SELECT MAX(age_group) + 1, MAX(risk_group) + 1 FROM results_full"
        ).fetchone()
        dims.setdefault("num_age_groups", n_age)
        dims["num_age_groups"] = dims["num_age_groups"] or n_age
        dims["num_risk_groups"] = dims.get("num_risk_groups") or n_risk
    if not dims.get("n_reps"):
        dims["n_reps"] = con.execute(
            "SELECT COUNT(DISTINCT rep) FROM results").fetchone()[0]

    day_min, day_max = con.execute(
        "SELECT MIN(day), MAX(day) FROM results").fetchone()
    dims["day_min"], dims["day_max"] = day_min, day_max

    # Labels for the age selector: real bands ("5-12") when the file knows
    # them, otherwise the bare indices the rows actually store.
    n_age = dims.get("num_age_groups") or 0
    labels = dims.get("age_group_labels") or []
    dims["age_group_labels"] = (
        list(labels) if len(labels) == n_age else [str(i) for i in range(n_age)]
    )
    return dims


# ---------------------------------------------------------------------------
# Chart configs
# ---------------------------------------------------------------------------


def default_chart_config(chart_id: int, chart_type: str, dims: dict) -> dict:
    """A config that renders something sensible the moment a chart is added."""
    metrics = [m for m, _ in dims.get("metrics", [])]
    scenarios = list(dims.get("scenarios", []))
    # Prefer a transition (an incidence series like new hospitalizations)
    # since that is the usual first thing to look at; fall back to whatever
    # the model has.
    first_transition = next(
        (m for m, k in dims.get("metrics", []) if k == "transition"), None)
    default_metric = first_transition or (metrics[0] if metrics else None)
    return {
        "id": chart_id,
        "chart_type": chart_type,
        "metrics": [default_metric] if default_metric else [],
        # A stacked bar needs a dimension to stack, so it cannot default to
        # the population total the way the other chart types do.
        "agg_level": "age_group" if chart_type == "stacked_bar" else "population",
        # Slice restrictions, independent of agg_level: None means "all of
        # them", which is what an unrestricted chart wants.
        "subpop_filter": None,
        "age_filter": None,
        "risk_filter": None,
        # Running total over days. Only meaningful for transition variables
        # (flows); a compartment series is already a level, so summing it over
        # days measures nothing. Enforced in _validate.
        "cumulative": False,
        # An extra panel summing every group, so the breakdown does not cost
        # sight of the overall number. Ignored unless exactly one dimension is
        # broken out -- see _total_group_col.
        "show_total": True,
        "hide_empty": True,
        # Small multiples share a y axis by default, which is what makes them
        # comparable -- but one dominant group (a peak age band, say) then
        # flattens every other panel against the axis until they read as
        # blank. Turning this off rescales each panel to its own data.
        "shared_y": True,
        "scenario_mode": "levels",
        "scenarios": scenarios[:3],
        "baseline_scenario": scenarios[0] if scenarios else None,
        "compare_metric": "difference",
        # Paired by default: within this pipeline replicate i of every
        # scenario is seeded by position, so pairing cancels the shared RNG
        # draw and tightens the interval. Surfaced in the UI because it is an
        # assumption, not a universal truth (see build_comparison_query).
        "pairing": "paired",
        "show_uncertainty": True,
        "day_range": None,
        "scatter_x": default_metric,
        "scatter_y": metrics[1] if len(metrics) > 1 else default_metric,
    }


def _validate(config: dict, dims: dict) -> None:
    if config.get("chart_type") not in CHART_TYPES:
        raise ResultsExplorerError(f"Unknown chart type {config.get('chart_type')!r}")
    if config.get("agg_level") not in AGG_LEVELS:
        raise ResultsExplorerError(f"Unknown aggregation level {config.get('agg_level')!r}")
    if not config.get("scenarios"):
        raise ResultsExplorerError("Select at least one scenario.")
    if config["chart_type"] == "scatter":
        if not config.get("scatter_x") or not config.get("scatter_y"):
            raise ResultsExplorerError("Scatter charts need both an X and a Y metric.")
    elif not config.get("metrics"):
        raise ResultsExplorerError("Select at least one metric.")
    if is_comparison(config):
        baseline = config.get("baseline_scenario")
        if not baseline:
            raise ResultsExplorerError("Pick a baseline scenario to compare against.")
        others = [s for s in config["scenarios"] if s != baseline]
        if not others:
            raise ResultsExplorerError(
                "Comparison needs at least one scenario other than the baseline."
            )
    if config.get("cumulative"):
        kinds = dict(dims.get("metrics") or [])
        levels = [m for m in config.get("metrics") or []
                  if kinds.get(m, "transition") == "compartment"]
        if levels:
            raise ResultsExplorerError(
                f"Cumulative is only meaningful for transition variables "
                f"(daily flows). {', '.join(levels)} "
                f"{'is a compartment' if len(levels) == 1 else 'are compartments'}"
                f" — already a level, so a running total over days measures "
                f"nothing. Uncheck Cumulative, or select only transitions."
            )


# ---------------------------------------------------------------------------
# SQL building
# ---------------------------------------------------------------------------


def _in_clause(values: Sequence[str]) -> tuple[str, list[str]]:
    return ",".join("?" * len(values)), list(values)


def _active_filters(config: dict) -> list[tuple[str, Any]]:
    """The (column, value) slice restrictions the user pinned, if any."""
    out: list[tuple[str, Any]] = []
    for col, key in _FILTER_COLS:
        value = config.get(key)
        if value is not None and value != "":
            out.append((col, value))
    return out


def _source_table(config: dict) -> str:
    # `results` is pre-summed over subpop/age/risk, so population charts read
    # it and skip results_full entirely -- on the 57 GB example that is a 73M
    # row table instead of a ~500M row one. Pinning any single slice forfeits
    # that shortcut: `results` no longer has the column to filter on.
    if config["agg_level"] == "population" and not _active_filters(config):
        return "results"
    return "results_full"


def _filter_clauses(config: dict) -> tuple[list[str], list[Any]]:
    where, params = [], []
    for col, value in _active_filters(config):
        where.append(f"{col} = ?")
        params.append(int(value) if col.endswith("_group") else value)
    return where, params


def _total_group_col(config: dict) -> str | None:
    """The group column that gets an extra combined-over-everything slice.

    Breaking a chart into small multiples answers "how is this split up?" but
    loses "and how much is it altogether?", which is usually the number the
    reader wants to anchor on. Adding the total as one more panel puts both on
    the same page, at the same scale.

    None unless there is exactly one breakdown dimension: at agg_level "all"
    three dimensions are broken out at once, and a single extra slice could
    not say which of them it totals over. Also None for a stacked bar, where
    the parts are already drawn on top of each other and a total segment would
    stack on top of its own components, doubling the bar.
    """
    if not config.get("show_total"):
        return None
    if config.get("chart_type") == "stacked_bar":
        return None
    cols = _AGG_GROUP_COLS[config["agg_level"]]
    if len(cols) != 1:
        return None
    col = cols[0]
    # Pinned to one slice, the "total" would be a copy of the only panel.
    if col in {c for c, _ in _active_filters(config)}:
        return None
    return col


def supports_total(config: dict) -> bool:
    """Whether the combined-total panel is meaningful for this chart at all.

    Independent of whether it is currently switched on, so the UI can decide
    whether to *offer* the toggle.
    """
    return _total_group_col({**config, "show_total": True}) is not None


def _total_union_sql(
    source: str, cols: Sequence[str], keys: Sequence[str],
    value_cols: Sequence[str],
) -> str:
    """``source`` plus one extra row per key, holding the sum over ``cols``.

    With several columns the extra row is the combined total over all of them
    at once -- every one of them reads as the total marker -- rather than a
    marginal total per column, which would need a row set per column and no
    way to say which is which.
    """
    key_sql = ", ".join(keys)
    all_cols = ", ".join([*keys, *cols, *value_cols])
    totals = ", ".join(f"{_TOTAL_SQL_LITERAL[c]} AS {c}" for c in cols)
    sums = ", ".join(f"SUM({v}) AS {v}" for v in value_cols)
    return (
        f"  SELECT {all_cols} FROM {source}\n"
        f"  UNION ALL\n"
        f"  SELECT {key_sql}, {totals}, {sums}\n"
        f"  FROM {source} GROUP BY {key_sql}\n"
    )


def _per_rep_cte(
    config: dict, metrics: Sequence[str], *, alias: str = "per_rep",
) -> tuple[str, list[Any]]:
    """Per-(scenario, rep, param_set, day, group) totals -- the common base.

    Sums the metric(s) over whichever of subpop/age/risk the chosen
    aggregation level does not break out, leaving replicates intact so the
    caller can take medians/percentiles across them.

    With ``cumulative`` set the running total over ``day`` is taken here, per
    replicate -- before any median/percentile, so the interval describes the
    spread of cumulative curves rather than the (meaningless) cumulative sum
    of daily medians. The running total starts at the beginning of the
    selected day range, not of the run.
    """
    agg_level = config["agg_level"]
    table = _source_table(config)
    group_cols = _AGG_GROUP_COLS[agg_level]
    group_sql = "".join(f", {c}" for c in group_cols)

    scen_ph, params = _in_clause(config["scenarios"])
    metric_ph, metric_params = _in_clause(list(metrics))
    params += metric_params

    # Every query filters scenario AND compartment before aggregating. On a
    # large results_full an unfiltered scan is minutes, so this is a
    # correctness-of-performance requirement, not an optimization.
    where = [f"scenario IN ({scen_ph})", f"compartment IN ({metric_ph})"]

    filter_where, filter_params = _filter_clauses(config)
    where += filter_where
    params += filter_params

    day_range = config.get("day_range")
    if day_range:
        where.append("day BETWEEN ? AND ?")
        params += [int(day_range[0]), int(day_range[1])]

    body = (
        f"  SELECT scenario, rep, param_set, day{group_sql}, SUM(value) AS value\n"
        f"  FROM {table}\n"
        f"  WHERE {' AND '.join(where)}\n"
        f"  GROUP BY scenario, rep, param_set, day{group_sql}\n"
    )

    # A chain of CTEs, each reading the one before: slice -> combined total
    # -> running total. Order matters: the total must be summed across groups
    # before it is accumulated over days, and both must happen per replicate,
    # ahead of any median or percentile.
    keys = ["scenario", "rep", "param_set", "day"]
    stages = [body]
    total_col = _total_group_col(config)
    if total_col:
        stages.append(_total_union_sql(
            f"{alias}_{len(stages) - 1}", [total_col], keys, ["value"]))
    if config.get("cumulative"):
        part_sql = "".join(f", {c}" for c in group_cols)
        stages.append(
            f"  SELECT scenario, rep, param_set, day{group_sql},\n"
            f"         SUM(value) OVER (\n"
            f"           PARTITION BY scenario, rep, param_set{part_sql}\n"
            f"           ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
            f"         ) AS value\n"
            f"  FROM {alias}_{len(stages) - 1}\n"
        )
    # The last stage carries the name the caller expects; the rest are
    # intermediate and numbered.
    names = [f"{alias}_{i}" for i in range(len(stages) - 1)] + [alias]
    return ",\n".join(f"{n} AS (\n{s})" for n, s in zip(names, stages)), params


def build_timeseries_query(config: dict, *, with_ci: bool) -> tuple[str, list[Any]]:
    """Median (and 95% interval) per scenario/day, across replicates."""
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)
    cte, params = _per_rep_cte(config, config["metrics"])

    # median()/quantile_cont() are native DuckDB aggregates, so the percentile
    # work happens in-engine over the replicate rows rather than pulling every
    # replicate series into Python.
    stats = "median(value) AS median_value"
    if with_ci:
        stats += (
            ",\n         quantile_cont(value, 0.025) AS p_lo"
            ",\n         quantile_cont(value, 0.975) AS p_hi"
        )
    sql = (
        f"WITH {cte}\n"
        f"SELECT scenario, day{group_sql},\n         {stats}\n"
        f"FROM per_rep\n"
        f"GROUP BY scenario, day{group_sql}\n"
        f"ORDER BY scenario, day{group_sql}"
    )
    return sql, params


def build_comparison_query(config: dict, *, with_ci: bool) -> tuple[str, list[Any]]:
    """Each scenario against the baseline, as a difference or a ratio.

    Two ways to combine, both offered because they answer slightly different
    questions:

    ``paired``
        Match replicate i of a scenario with replicate i of the baseline,
        take the difference per pair, then summarize those differences.
        Shared noise (the RNG stream, the sampled parameter draw) cancels, so
        the interval is tighter and reflects the effect of the intervention
        alone. Valid only if replicate i really does correspond across
        scenarios -- true here for transition-only uncertainty, since runs
        are seeded by position, and weaker once parameter sampling varies
        which draw backs a given index. The UI states this assumption.

    ``unpaired``
        Summarize each scenario independently, then compare the summaries.
        Always valid, but the interval also carries the between-replicate
        spread that pairing would have cancelled.
    """
    baseline = config["baseline_scenario"]
    others = [s for s in config["scenarios"] if s != baseline]
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)
    is_ratio = config.get("compare_metric") == "ratio"

    # The CTE must span baseline + comparison scenarios regardless of what
    # the user ticked, so the baseline is present to join/compare against.
    cte_config = {**config, "scenarios": [baseline] + others}
    cte, params = _per_rep_cte(cte_config, config["metrics"])

    expr = (
        "c.value / NULLIF(b.value, 0)" if is_ratio else "c.value - b.value"
    )

    if config.get("pairing", "paired") == "paired":
        join_cols = " AND ".join(
            [f"c.{c} = b.{c}" for c in group_cols]
            + ["c.day = b.day", "c.rep = b.rep",
               # A NULL param_set (parameter uncertainty off) must still match
               # a NULL, which `=` would not do.
               "c.param_set IS NOT DISTINCT FROM b.param_set"]
        )
        stats = f"median({expr}) AS median_value"
        if with_ci:
            stats += (
                f",\n         quantile_cont({expr}, 0.025) AS p_lo"
                f",\n         quantile_cont({expr}, 0.975) AS p_hi"
            )
        select_group = "".join(f", c.{c} AS {c}" for c in group_cols)
        group_by = "".join(f", c.{c}" for c in group_cols)
        sql = (
            f"WITH {cte},\n"
            f"baseline AS (SELECT * FROM per_rep WHERE scenario = ?),\n"
            f"compared AS (SELECT * FROM per_rep WHERE scenario <> ?)\n"
            f"SELECT c.scenario AS scenario, c.day AS day{select_group},\n"
            f"         {stats}\n"
            f"FROM compared c JOIN baseline b ON {join_cols}\n"
            f"GROUP BY c.scenario, c.day{group_by}\n"
            f"ORDER BY c.scenario, c.day{group_by}"
        )
        return sql, params + [baseline, baseline]

    # Unpaired: collapse replicates first, then compare the point estimates.
    agg_stats = "median(value) AS median_value"
    if with_ci:
        agg_stats += (
            ",\n           quantile_cont(value, 0.025) AS p_lo"
            ",\n           quantile_cont(value, 0.975) AS p_hi"
        )
    cmp_cols = "c.median_value - b.median_value AS median_value"
    if is_ratio:
        cmp_cols = "c.median_value / NULLIF(b.median_value, 0) AS median_value"
    if with_ci:
        if is_ratio:
            cmp_cols += (
                ",\n       c.p_lo / NULLIF(b.median_value, 0) AS p_lo"
                ",\n       c.p_hi / NULLIF(b.median_value, 0) AS p_hi"
            )
        else:
            cmp_cols += (
                ",\n       c.p_lo - b.median_value AS p_lo"
                ",\n       c.p_hi - b.median_value AS p_hi"
            )
    join_cols = " AND ".join(
        [f"c.{c} = b.{c}" for c in group_cols] + ["c.day = b.day"])
    select_group = "".join(f", c.{c} AS {c}" for c in group_cols)
    sql = (
        f"WITH {cte},\n"
        f"agg AS (\n"
        f"  SELECT scenario, day{group_sql},\n           {agg_stats}\n"
        f"  FROM per_rep GROUP BY scenario, day{group_sql}\n"
        f"),\n"
        f"baseline AS (SELECT * FROM agg WHERE scenario = ?),\n"
        f"compared AS (SELECT * FROM agg WHERE scenario <> ?)\n"
        f"SELECT c.scenario AS scenario, c.day AS day{select_group},\n"
        f"       {cmp_cols}\n"
        f"FROM compared c JOIN baseline b ON {join_cols}\n"
        f"ORDER BY c.scenario, c.day"
    )
    return sql, params + [baseline, baseline]


def is_comparison(config: dict) -> bool:
    """Whether this chart shows scenarios relative to a baseline.

    Tested positively rather than against the "levels" name, so a config
    carrying a retired mode ("single"/"multiple", which behaved identically to
    each other and to "levels") still renders instead of erroring.
    """
    return config.get("scenario_mode") == "compare_baseline"


def comparison_reference(config: dict) -> float | None:
    """The y value that means "no effect" -- 0 for a difference, 1 for a ratio.

    None when the chart is not a comparison, so callers can skip the rule.
    """
    if not is_comparison(config):
        return None
    return 1.0 if config.get("compare_metric") == "ratio" else 0.0


def _compare_expr(config: dict, left: str, right: str) -> str:
    if config.get("compare_metric") == "ratio":
        return f"{left} / NULLIF({right}, 0)"
    return f"{left} - {right}"


def _totals_cte(config: dict) -> tuple[str, list[Any]]:
    """CTE chain ending in ``totals``: one row per (scenario, replicate, group).

    In plain mode ``total_value`` is that replicate's total over the selected
    day range. In comparison mode it is that replicate's total *relative to the
    baseline* -- a difference or a ratio -- and the baseline scenario itself
    drops out, since comparing it to itself is a column of zeros (or ones).

    Getting this right at the replicate level, rather than differencing two
    summaries afterwards, is what lets a box plot or histogram show the
    distribution of the *effect* instead of the distribution of two levels.
    """
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)

    if not is_comparison(config):
        cte, params = _per_rep_cte(config, config["metrics"])
        sql = (
            f"{cte},\n"
            f"totals AS (\n"
            f"  SELECT scenario, rep, param_set{group_sql},\n"
            f"         SUM(value) AS total_value\n"
            f"  FROM per_rep GROUP BY scenario, rep, param_set{group_sql}\n"
            f")"
        )
        return sql, params

    baseline = config["baseline_scenario"]
    others = [s for s in config["scenarios"] if s != baseline]
    # The baseline must be in the scan whether or not the user ticked it,
    # since every compared row needs its counterpart.
    cte, params = _per_rep_cte({**config, "scenarios": [baseline] + others},
                               config["metrics"])
    params = params + [baseline, baseline]

    head = (
        f"{cte},\n"
        f"raw_totals AS (\n"
        f"  SELECT scenario, rep, param_set{group_sql},\n"
        f"         SUM(value) AS total_value\n"
        f"  FROM per_rep GROUP BY scenario, rep, param_set{group_sql}\n"
        f"),\n"
        f"base AS (SELECT * FROM raw_totals WHERE scenario = ?),\n"
        f"cmp AS (SELECT * FROM raw_totals WHERE scenario <> ?),\n"
    )
    select_group = "".join(f", c.{c} AS {c}" for c in group_cols)

    if config.get("pairing", "paired") == "paired":
        # Replicate i of a scenario against replicate i of the baseline, so the
        # shared RNG/parameter draw cancels and the spread shown is the spread
        # of the effect itself. Same assumption as the time-series comparison.
        join = " AND ".join(
            [f"c.{c} = b.{c}" for c in group_cols]
            + ["c.rep = b.rep", "c.param_set IS NOT DISTINCT FROM b.param_set"]
        )
        expr = _compare_expr(config, "c.total_value", "b.total_value")
        sql = (
            f"{head}"
            f"totals AS (\n"
            f"  SELECT c.scenario AS scenario, c.rep AS rep,\n"
            f"         c.param_set AS param_set{select_group},\n"
            f"         {expr} AS total_value\n"
            f"  FROM cmp c JOIN base b ON {join}\n"
            f")"
        )
        return sql, params

    # Unpaired: every replicate is measured against one baseline summary, so
    # the spread shown is the compared scenario's own spread, not the effect's.
    base_group_by = f"\n  GROUP BY {', '.join(group_cols)}" if group_cols else ""
    base_select = "".join(f"{c}, " for c in group_cols)
    join = (" AND ".join(f"c.{c} = b.{c}" for c in group_cols)
            if group_cols else "TRUE")
    expr = _compare_expr(config, "c.total_value", "b.base_value")
    sql = (
        f"{head}"
        f"base_agg AS (\n"
        f"  SELECT {base_select}median(total_value) AS base_value\n"
        f"  FROM base{base_group_by}\n"
        f"),\n"
        f"totals AS (\n"
        f"  SELECT c.scenario AS scenario, c.rep AS rep,\n"
        f"         c.param_set AS param_set{select_group},\n"
        f"         {expr} AS total_value\n"
        f"  FROM cmp c JOIN base_agg b ON {join}\n"
        f")"
    )
    return sql, params


def build_per_rep_totals_query(config: dict) -> tuple[str, list[Any]]:
    """One total per replicate -- the basis for histograms and box plots.

    Sums each replicate's series over the selected day range, giving a
    distribution across replicates (e.g. total season hospitalizations), or of
    per-replicate differences/ratios against the baseline when comparing.
    """
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)
    cte, params = _totals_cte(config)
    sql = (
        f"WITH {cte}\n"
        f"SELECT scenario, rep, param_set{group_sql}, total_value\n"
        f"FROM totals\n"
        f"ORDER BY scenario, rep"
    )
    return sql, params


def build_stacked_bar_query(config: dict) -> tuple[str, list[Any]]:
    """Median per-replicate total, split by the aggregation dimension."""
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    if not group_cols:
        raise ResultsExplorerError(
            "A stacked bar needs something to stack — choose an aggregation "
            "level other than 'population' (e.g. age group)."
        )
    group_sql = "".join(f", {c}" for c in group_cols)
    cte, params = _totals_cte(config)
    sql = (
        f"WITH {cte}\n"
        f"SELECT scenario{group_sql}, median(total_value) AS median_value\n"
        f"FROM totals\n"
        f"GROUP BY scenario{group_sql}\n"
        f"ORDER BY scenario{group_sql}"
    )
    return sql, params


def _scatter_per_rep_sql(config: dict, scenarios: Sequence[str]) -> tuple[str, list[Any]]:
    """Per-replicate X/Y totals for ``scenarios`` -- the scatter's raw rows.

    Uses conditional aggregation (SUM(...) FILTER) rather than joining two
    separate queries, so it is a single pass over the filtered rows.
    """
    x_metric, y_metric = config["scatter_x"], config["scatter_y"]
    table = _source_table(config)
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)

    scen_ph, params = _in_clause(list(scenarios))
    metrics = [x_metric] if x_metric == y_metric else [x_metric, y_metric]
    metric_ph, metric_params = _in_clause(metrics)
    where = [f"scenario IN ({scen_ph})", f"compartment IN ({metric_ph})"]
    params += metric_params

    filter_where, filter_params = _filter_clauses(config)
    where += filter_where
    params += filter_params
    day_range = config.get("day_range")
    if day_range:
        where.append("day BETWEEN ? AND ?")
        params += [int(day_range[0]), int(day_range[1])]

    sql = (
        f"  SELECT scenario, rep, param_set{group_sql},\n"
        f"         SUM(value) FILTER (WHERE compartment = ?) AS x_value,\n"
        f"         SUM(value) FILTER (WHERE compartment = ?) AS y_value\n"
        f"  FROM {table}\n"
        f"  WHERE {' AND '.join(where)}\n"
        f"  GROUP BY scenario, rep, param_set{group_sql}\n"
    )
    # The two FILTER placeholders bind before the WHERE ones in the text, so
    # they lead the parameter list.
    return sql, [x_metric, y_metric] + params


def _scatter_body(config: dict, scenarios: Sequence[str]) -> tuple[str, list[Any]]:
    """:func:`_scatter_per_rep_sql` with the combined-total slice folded in.

    Emitted as a leading CTE rather than inline, since the total needs a second
    pass over the same rows.
    """
    body, params = _scatter_per_rep_sql(config, scenarios)
    total_col = _total_group_col(config)
    if not total_col:
        return body, params
    union = _total_union_sql(
        "scatter_slice", [total_col],
        ["scenario", "rep", "param_set"], ["x_value", "y_value"])
    return f"  WITH scatter_slice AS (\n{body})\n{union}", params


def build_scatter_query(config: dict) -> tuple[str, list[Any]]:
    """Two metrics per replicate, as X and Y.

    When comparing, both axes become per-replicate differences (or ratios)
    against the baseline, so the cloud sits around the no-effect point rather
    than around the two scenarios' levels.
    """
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]

    if not is_comparison(config):
        body, params = _scatter_body(config, config["scenarios"])
        return f"{body}  ORDER BY scenario, rep", params

    baseline = config["baseline_scenario"]
    others = [s for s in config["scenarios"] if s != baseline]
    body, params = _scatter_body(config, [baseline] + others)
    params = params + [baseline, baseline]

    select_group = "".join(f", c.{c} AS {c}" for c in group_cols)
    if config.get("pairing", "paired") == "paired":
        join = " AND ".join(
            [f"c.{c} = b.{c}" for c in group_cols]
            + ["c.rep = b.rep", "c.param_set IS NOT DISTINCT FROM b.param_set"]
        )
        x_expr = _compare_expr(config, "c.x_value", "b.x_value")
        y_expr = _compare_expr(config, "c.y_value", "b.y_value")
        base_rel = "base b"
    else:
        join = (" AND ".join(f"c.{c} = b.{c}" for c in group_cols)
                if group_cols else "TRUE")
        x_expr = _compare_expr(config, "c.x_value", "b.x_value")
        y_expr = _compare_expr(config, "c.y_value", "b.y_value")
        base_group_by = f"\n    GROUP BY {', '.join(group_cols)}" if group_cols else ""
        base_select = "".join(f"{c}, " for c in group_cols)
        base_rel = (
            f"(SELECT {base_select}median(x_value) AS x_value,\n"
            f"          median(y_value) AS y_value\n"
            f"   FROM base{base_group_by}) b"
        )

    sql = (
        f"WITH per_rep AS (\n{body}),\n"
        f"base AS (SELECT * FROM per_rep WHERE scenario = ?),\n"
        f"cmp AS (SELECT * FROM per_rep WHERE scenario <> ?)\n"
        f"SELECT c.scenario AS scenario, c.rep AS rep,\n"
        f"       c.param_set AS param_set{select_group},\n"
        f"       {x_expr} AS x_value,\n"
        f"       {y_expr} AS y_value\n"
        f"FROM cmp c JOIN {base_rel} ON {join}\n"
        f"ORDER BY c.scenario, c.rep"
    )
    return sql, params


# ---------------------------------------------------------------------------
# Query execution + presentation helpers
# ---------------------------------------------------------------------------


def _run(con, sql: str, params: list[Any]) -> pd.DataFrame:
    try:
        return con.execute(sql, params).df()
    except duckdb.Error as exc:
        raise ResultsExplorerError(f"Query failed: {exc}") from exc


def _guard_size(df: pd.DataFrame) -> None:
    if len(df) > SOFT_ROW_LIMIT:
        raise ResultsExplorerError(
            f"This selection produces {len(df):,} rows (limit "
            f"{SOFT_ROW_LIMIT:,}). Narrow it — fewer scenarios, a shorter day "
            f"range, or a coarser aggregation level."
        )
    if df.empty:
        raise ResultsExplorerError(
            "No data for this selection. Check the metric and scenario "
            "choices — a transition variable may not have been recorded."
        )


def _add_date(df: pd.DataFrame, dims: dict) -> tuple[pd.DataFrame, str, str]:
    """Turn 1-based day indices into real dates when the file knows its start.

    Returns the frame plus the field name and Altair type to encode on X.
    """
    start = dims.get("start_date")
    if start and "day" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(start) + pd.to_timedelta(df["day"] - 1, unit="D")
        return df, "date", "T"
    return df, "day", "Q"


def _label_total_slices(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Swap the combined-slice sentinel for :data:`TOTAL_LABEL` in ``cols``."""
    for col in cols:
        if col not in df.columns:
            continue
        df = df.copy()
        # Cast the whole column, not just the renamed row: a mix of ints and
        # one string makes the facet's explicit sort order heterogeneous,
        # which Altair rejects against its schema.
        df[col] = df[col].astype(str).mask(
            df[col] == _TOTAL_VALUE[col], TOTAL_LABEL)
    return df


def _label_groups(df: pd.DataFrame, dims: dict, config: dict | None = None) -> pd.DataFrame:
    """Swap raw group keys for readable labels: 0-based age indices for real
    band labels where available, and the combined-slice sentinel for
    :data:`TOTAL_LABEL`."""
    total_col = _total_group_col(config) if config else None
    if total_col:
        df = _label_total_slices(df, [total_col])

    if "age_group" not in df.columns:
        return df
    labels = dims.get("age_group_labels") or []
    if not labels:
        return df
    df = df.copy()
    df["age_group"] = df["age_group"].map(
        lambda i: i if i == TOTAL_LABEL else
        labels[int(i)] if 0 <= int(i) < len(labels) else str(i)
    )
    return df


def _facet_col(config: dict) -> str | None:
    """Which column becomes small multiples.

    Colour already encodes scenario, so the aggregation dimension needs its
    own channel. "all" breaks out three dimensions at once; facet on the first
    dimension that is not pinned to a single slice -- faceting on a pinned one
    would draw a grid of exactly one panel.
    """
    pinned = {col for col, _ in _active_filters(config)}
    candidates = {
        "population": (),
        "subpop": ("subpop",),
        "age_group": ("age_group",),
        "risk_group": ("risk_group",),
        # Age group usually varies most, so it makes the most informative grid.
        "all": ("age_group", "subpop", "risk_group"),
    }[config["agg_level"]]
    return next((c for c in candidates if c not in pinned), None)


def _residual_group_cols(config: dict) -> list[str]:
    """Grouping columns a line chart faces that the facet does not absorb.

    At agg_level "all" the query groups by subpop, age_group AND risk_group,
    but only one of those becomes small multiples. The rest still split the
    rows, so a line chart would otherwise get several y values per x and draw
    a zig-zag joining unrelated series. They need their own visual channel.
    """
    facet = _facet_col(config)
    return [c for c in _AGG_GROUP_COLS[config["agg_level"]] if c != facet]


def _y_resolve(config: dict) -> Literal["shared", "independent"]:
    """Whether small multiples share one y scale or each get their own.

    Shared is the default because it is what makes panels comparable at a
    glance. It fails badly when one group dominates -- every other panel
    flattens onto the axis and reads as an empty chart -- which is exactly
    when the user wants to turn it off.
    """
    return "shared" if config.get("shared_y", True) else "independent"


def _drop_empty_groups(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop breakdown groups whose series is entirely zero (or missing).

    A granular aggregation over a model whose population is concentrated in a
    few slices produces panels that are flat on zero for their whole length --
    visually blank, and they squeeze the panels that do carry signal. Emptiness
    is defined as "no non-zero value anywhere", so a difference chart showing a
    genuine no-effect group is dropped for the same (correct) reason a group
    with no population at all is.

    Never drops everything: if every group is empty the frame is returned
    untouched, so the caller reports "no data" rather than silently blanking.
    """
    if not config.get("hide_empty", True):
        return df
    cols = [c for c in _AGG_GROUP_COLS[config["agg_level"]] if c in df.columns]
    if not cols:
        return df
    value_cols = [c for c in ("median_value", "total_value", "x_value", "y_value")
                  if c in df.columns]
    if not value_cols:
        return df
    nonzero = df[value_cols].fillna(0).abs().sum(axis=1) > 0
    keep = nonzero.groupby([df[c] for c in cols]).transform("any").to_numpy(dtype=bool)
    if keep.all() or not keep.any():
        return df
    return df.iloc[keep.nonzero()[0]]


def _series_column(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Collapse residual grouping columns into one labelled series column.

    Returns no column when those dimensions do not actually split the data --
    a single-population, single-risk-group model (the common case) would
    otherwise get a legend with exactly one entry in it.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return df, None
    series = df[present].astype(str).agg(" · ".join, axis=1)
    if series.nunique() <= 1:
        return df, None
    df = df.copy()
    df["_series"] = series
    return df, "_series"


def _value_axis_title(config: dict, what: str) -> str:
    """Axis title for a per-replicate quantity, saying what it is measured
    against when the chart is a comparison."""
    if not is_comparison(config):
        return f"{what} (total per replicate)"
    rel = "ratio to" if config.get("compare_metric") == "ratio" else "difference from"
    return f"{what} — {rel} {config.get('baseline_scenario')} (per replicate)"


def _reference_rule(config: dict, channel: str = "y") -> alt.Chart | None:
    """A dashed line at the no-effect value, so a comparison is readable
    against something rather than floating.

    Encoded with ``alt.datum`` rather than its own one-row frame: a layer
    whose subcharts carry different data sources cannot be faceted (Altair
    rejects it outright), and every comparison chart here may be faceted.
    """
    reference = comparison_reference(config)
    if reference is None:
        return None
    return (alt.Chart()
            .mark_rule(strokeDash=[4, 4], color="grey")
            .encode(**{channel: alt.datum(reference)}))


def _panel_order(df: pd.DataFrame, col: str, dims: dict) -> list:
    """The order small multiples should be laid out in.

    Row order out of the query is whatever the GROUP BY hash happened to
    produce, so panels arrive shuffled ("0, 5, 4, 1, 3, 6, 2"). Age groups
    follow the label list from the file's metadata, which is the model's own
    ordering and the only thing that can order bands like "0-4"/"5-12"
    correctly; anything else sorts numerically where it can and
    alphabetically where it cannot. The combined total goes last, after the
    parts it sums.
    """
    values = df[col].drop_duplicates().tolist()
    rank = {}
    if col == "age_group":
        rank = {str(label): i for i, label in enumerate(dims.get("age_group_labels") or [])}

    def key(value: Any) -> tuple[int, float, str]:
        text = str(value)
        if text in rank:
            return (0, rank[text], "")
        try:
            return (0, float(text), "")
        except ValueError:
            return (1, 0.0, text)

    order = sorted((v for v in values if v != TOTAL_LABEL), key=key)
    return order + ([TOTAL_LABEL] if TOTAL_LABEL in values else [])


def _order_panels(df: pd.DataFrame, config: dict, dims: dict) -> pd.DataFrame:
    """Put the rows in panel order, which is how the facet grid reads them.

    The obvious spelling -- handing Vega-Lite the order as ``Facet(sort=[...])``
    -- does not survive a composite mark: for a box plot it compiles the order
    into a ``facet_<field>_sort_index`` column that the box plot's own
    aggregate then drops, leaving the cell sort comparing nulls and the panels
    in whatever order the hash aggregation produced. Ordering the rows here and
    telling the facet to use data order (``sort=None``) works for every mark,
    composite or not. The sort is stable, so within-panel row order (day
    sequence, scenario order) is untouched.
    """
    col = _facet_col(config)
    if not col or col not in df.columns:
        return df
    rank = {value: i for i, value in enumerate(_panel_order(df, col, dims))}
    return df.sort_values(col, key=lambda s: s.map(rank), kind="stable")


#: Roughly how many characters of a categorical axis label fit on one line
#: before it crowds its neighbours.
_LABEL_WRAP_WIDTH = 16


def _wrapped_scenario_x(
    df: pd.DataFrame, title: str = "Scenario",
) -> tuple[pd.DataFrame, Any]:
    """A horizontal, line-wrapped scenario axis instead of a rotated one.

    Scenario names here are sentences ("High VE + 70% coverage (all ages)"),
    and rotated vertically they are slow to read and eat most of the chart's
    height. Vega's expression language has no loop, so a greedy wrap cannot be
    written as a ``labelExpr``; wrap in pandas into a display-only column
    instead and let ``labelExpr`` split it back into the string array Vega
    renders as separate lines. ``scenario`` itself is left untouched, so
    colour, sort order and baseline matching keep working on the real names.
    """
    wrapped = {
        s: "\n".join(textwrap.wrap(str(s), _LABEL_WRAP_WIDTH)) or str(s)
        for s in df["scenario"].drop_duplicates()
    }
    df = df.copy()
    df["scenario_label"] = df["scenario"].map(lambda s: wrapped[s])
    axis = alt.X(
        "scenario_label:N",
        title=title,
        # Keep the query's scenario order rather than re-sorting the wrapped text.
        sort=list(wrapped.values()),
        axis=alt.Axis(
            labelAngle=0, labelLimit=0, labelPadding=6, labelFontSize=11,
            labelExpr="split(datum.label, '\\n')",
        ),
    )
    return df, axis


def _facet_grid(
    chart: Any, df: pd.DataFrame, config: dict, *, width: int, height: int,
) -> AltairChart:
    """Turn one chart into small multiples over the breakdown dimension.

    Returns ``chart`` unchanged when there is nothing to facet on, so callers
    can hand back whatever this gives them.
    """
    col = _facet_col(config)
    if not col or col not in df.columns:
        return chart

    n_panels = df[col].nunique()
    columns = 3
    faceted = chart.properties(width=width, height=height).facet(
        # Data order, laid down by _order_panels -- see the note there for why
        # this is not an explicit sort list.
        facet=alt.Facet(f"{col}:N", title=col.replace("_", " "), sort=None),
        columns=columns,
    ).resolve_scale(y=_y_resolve(config))

    if n_panels % columns:
        # A ragged last row leaves empty cells, and with the default shared
        # axis resolution Vega-Lite still draws each column's x axis at the
        # bottom of the grid -- so the blanks come out as bare axes that read
        # as empty plots. Per-panel axes put each one under its own chart.
        faceted = faceted.resolve_axis(x="independent")
    return faceted


def _metric_label(config: dict) -> str:
    """Y-axis label. Several metrics are summed, which the ``+`` states."""
    label = " + ".join(config.get("metrics") or [])
    return f"cumulative {label}" if config.get("cumulative") else label


def _slice_label(config: dict, dims: dict) -> str:
    """Human-readable description of the pinned slices, e.g. 'age 5-12'."""
    parts = []
    for col, value in _active_filters(config):
        if col == "age_group":
            labels = dims.get("age_group_labels") or []
            idx = int(value)
            name = labels[idx] if 0 <= idx < len(labels) else str(idx)
            parts.append(f"age {name}")
        elif col == "risk_group":
            parts.append(f"risk {value}")
        else:
            parts.append(str(value))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def _export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The rows behind a chart, without the columns that only draw it."""
    return df.drop(columns=[c for c in _DISPLAY_ONLY_COLS if c in df.columns])


def frame_to_csv(df: pd.DataFrame) -> bytes:
    """CSV bytes for a download button.

    Encoded here rather than at the call site so every download in the notebook
    is written the same way -- UTF-8, no index column (the index is a positional
    artifact of the query, not data).
    """
    return df.to_csv(index=False).encode("utf-8")


def slugify(text: str, *, max_len: int = 60) -> str:
    """A filename-safe stub of ``text``.

    Chart titles carry punctuation that browsers and shells both dislike
    ("Time series: IV_to_H + I_to_H (by age)"), so runs of anything that is not
    a word character collapse to a single underscore.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
    return (slug[:max_len].rstrip("_") or "chart").lower()


def image_kind() -> str:
    """Which image format a chart can be downloaded as: ``"png"`` or ``"html"``.

    PNG rendering needs ``vl-convert-python``, which compiles the Vega-Lite
    spec with an embedded browser-free renderer. It is a large wheel and not
    every environment will have it, so the fallback is Altair's own standalone
    HTML: no extra dependency, still a single self-contained file, and
    interactive into the bargain -- it just needs a browser (and, for the Vega
    runtime it loads from a CDN, a network connection) to view.
    """
    try:
        import vl_convert  # noqa: F401
    except ImportError:
        return "html"
    return "png"


#  Charts are built with width="container" (see e.g. _build_timeseries) so
#  they fill whatever card marimo puts them in on screen. vl-convert has no
#  such container to measure, and Vega-Lite's fallback for an unresolved
#  "container" width is a few hundred px -- so a downloaded PNG comes out far
#  narrower than what is on screen unless that width is pinned to a concrete
#  number first.
_DOWNLOAD_WIDTH = 900


def _pin_container_width(spec: dict, width: int) -> None:
    """Replace ``"width": "container"`` (and matching ``"height"``) in-place.

    Recurses into faceted/concatenated specs (``vconcat``/``hconcat``/``layer``/
    ``spec``), since a facet's inner width lives one level down from the root.
    Height is left alone unless it is itself "container" -- charts here always
    give it a concrete number, only width is ever "container".
    """
    if spec.get("width") == "container":
        spec["width"] = width
    if spec.get("height") == "container":
        spec["height"] = width
    for _key in ("spec", "layer", "vconcat", "hconcat", "concat"):
        _child = spec.get(_key)
        if isinstance(_child, dict):
            _pin_container_width(_child, width)
        elif isinstance(_child, list):
            for _c in _child:
                if isinstance(_c, dict):
                    _pin_container_width(_c, width)


def chart_image(chart: AltairChart) -> tuple[bytes, str, str]:
    """Render ``chart`` for download: ``(payload, extension, mimetype)``.

    Deliberately not called until a download button is actually clicked -- PNG
    conversion takes appreciable time per chart, and the notebook re-renders
    every chart on every widget change.
    """
    spec = json.loads(chart.to_json())  # type: ignore[attr-defined]
    _pin_container_width(spec, _DOWNLOAD_WIDTH)
    spec = json.dumps(spec)
    if image_kind() == "png":
        import vl_convert as vlc

        # scale=2 so the PNG is usable in a slide or a paper rather than only
        # on screen.
        return vlc.vegalite_to_png(spec, scale=2), "png", "image/png"
    return (
        chart.to_html().encode("utf-8"),  # type: ignore[attr-defined]
        "html",
        "text/html",
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_timeseries(con, config, dims) -> ChartResult:
    comparing = is_comparison(config)
    # A single replicate has no spread to summarize; asking for percentiles
    # would just draw a zero-width band around the line.
    with_ci = bool(config.get("show_uncertainty")) and (dims.get("n_reps") or 1) > 1

    if comparing:
        sql, params = build_comparison_query(config, with_ci=with_ci)
    else:
        sql, params = build_timeseries_query(config, with_ci=with_ci)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims, config)
    df = _order_panels(df, config, dims)
    df, x_field, x_type = _add_date(df, dims)

    if comparing:
        is_ratio = config.get("compare_metric") == "ratio"
        y_title = (
            f"{_metric_label(config)} — ratio vs {config['baseline_scenario']}"
            if is_ratio else
            f"{_metric_label(config)} — difference vs {config['baseline_scenario']}"
        )
    else:
        y_title = _metric_label(config)

    x_title = "Date" if x_field == "date" else "Day"
    # Any grouping dimension the facet does not absorb still splits the rows,
    # so give it its own channel — otherwise one line would zig-zag between
    # several unrelated series sharing an x value.
    df, series_col = _series_column(df, _residual_group_cols(config))
    base = alt.Chart(df)
    layers: list[Any] = []
    _tooltip = [c for c in df.columns if c != "_series"]

    if with_ci and {"p_lo", "p_hi"} <= set(df.columns):
        _area = {
            "x": alt.X(f"{x_field}:{x_type}", title=x_title),
            "y": alt.Y("p_lo:Q", title=y_title),
            "y2": alt.Y2("p_hi:Q"),
            "color": alt.Color("scenario:N", title="Scenario"),
        }
        if series_col:
            _area["detail"] = alt.Detail(f"{series_col}:N")
        layers.append(base.mark_area(opacity=0.2).encode(**_area))

    _line = {
        "x": alt.X(f"{x_field}:{x_type}", title=x_title),
        "y": alt.Y("median_value:Q", title=y_title),
        "color": alt.Color("scenario:N", title="Scenario"),
        "tooltip": _tooltip,
    }
    if series_col:
        _line["strokeDash"] = alt.StrokeDash(f"{series_col}:N", title="Series")
    layers.append(base.mark_line().encode(**_line))
    rule = _reference_rule(config)
    if rule is not None:
        layers.append(rule)

    # Data at the layer level, not only on the subcharts: the reference rule
    # has none of its own, and a facet needs its child's data at the top.
    chart = alt.layer(*layers, data=df)
    if _facet_col(config) in df.columns:
        built = _facet_grid(chart, df, config, width=320, height=180)
    else:
        built = chart.properties(width="container", height=320).resolve_scale(
            y=_y_resolve(config))
    return ChartResult(built, _export_frame(df))


def _needs_replicates(config: dict, dims: dict) -> None:
    """Reject the replicate-distribution charts on a single-replicate run.

    Histograms and box plots describe the spread *across replicates*. With one
    replicate each scenario contributes a single point, so they render as a
    hair-thin bar or a degenerate box — visually indistinguishable from an
    empty plot, which is worse than saying so plainly.
    """
    if (dims.get("n_reps") or 1) > 1:
        return
    kind = "A histogram" if config["chart_type"] == "histogram" else "A box plot"
    raise ResultsExplorerError(
        f"{kind} shows the spread across replicates, but this results file has "
        f"only one replicate — every scenario would be a single point. Use a "
        f"time series or stacked bar here, or open results from a stochastic "
        f"run (several replicates) to see a distribution."
    )


def _build_histogram(con, config, dims) -> ChartResult:
    _needs_replicates(config, dims)
    sql, params = build_per_rep_totals_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims, config)
    df = _order_panels(df, config, dims)

    # Bin count follows the sample size: 30 bins over a handful of replicates
    # leaves single-count bars too thin to see.
    _bins = max(5, min(30, len(df) // 2))
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("total_value:Q", bin=alt.Bin(maxbins=_bins),
                title=_value_axis_title(config, _metric_label(config))),
        y=alt.Y("count()", title="Replicates", stack=None),
        color=alt.Color("scenario:N", title="Scenario"),
        tooltip=["scenario:N", "count()"],
    )
    # Vertical here, since the compared quantity is on x for a histogram.
    rule = _reference_rule(config, "x")
    if rule is not None:
        chart = alt.layer(chart, rule, data=df)
    if _facet_col(config) in df.columns:
        built = _facet_grid(chart, df, config, width=280, height=180)
    else:
        built = chart.properties(width="container", height=320)
    return ChartResult(built, _export_frame(df))


def _build_boxplot(con, config, dims) -> ChartResult:
    _needs_replicates(config, dims)
    sql, params = build_per_rep_totals_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims, config)
    df = _order_panels(df, config, dims)

    df, x = _wrapped_scenario_x(df)
    # Vega sizes a box plot's box from the band width, which a faceted panel
    # makes narrow enough that the box reads as a bare tick. Pin it instead.
    chart = alt.Chart(df).mark_boxplot(
        extent="min-max", size=34, median=alt.MarkConfig(strokeWidth=3),
        rule=alt.MarkConfig(strokeWidth=2),
    ).encode(
        x=x,
        y=alt.Y("total_value:Q",
                title=_value_axis_title(config, _metric_label(config)),
                # A difference straddles zero, so forcing the axis through it
                # would waste most of the range on empty space.
                scale=alt.Scale(zero=not is_comparison(config))),
        color=alt.Color("scenario:N", title="Scenario", legend=None),
    )
    rule = _reference_rule(config)
    if rule is not None:
        chart = alt.layer(chart, rule, data=df)
    if _facet_col(config) in df.columns:
        # Wider than the other grids: horizontal labels need the room that
        # rotated ones did not.
        built = _facet_grid(chart, df, config, width=300, height=200)
    else:
        built = chart.properties(width="container", height=320)
    return ChartResult(built, _export_frame(df))


def _build_stacked_bar(con, config, dims) -> ChartResult:
    sql, params = build_stacked_bar_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims, config)
    df = _order_panels(df, config, dims)

    stack_col = _AGG_GROUP_COLS[config["agg_level"]][-1]
    comparing = is_comparison(config)
    y_title = (
        f"{_metric_label(config)} — median "
        f"{'ratio to' if config.get('compare_metric') == 'ratio' else 'difference from'} "
        f"{config.get('baseline_scenario')}"
        if comparing else f"{_metric_label(config)} (median total)"
    )
    tooltip = list(df.columns)
    df, x = _wrapped_scenario_x(df)
    encoding: dict[str, Any] = {
        "x": x,
        "y": alt.Y("median_value:Q", title=y_title,
                   scale=alt.Scale(zero=not comparing)),
        "color": alt.Color(f"{stack_col}:N", title=stack_col.replace("_", " ")),
        # The wrapped display column is an axis-rendering detail, not data.
        "tooltip": tooltip,
    }
    if comparing:
        # Differences can be negative, and stacking mixed signs makes the bar
        # heights mean nothing (Vega stacks them from wherever the running
        # total happens to be). Group them side by side instead, so each
        # group's effect is read against the zero rule directly.
        encoding["xOffset"] = alt.XOffset(f"{stack_col}:N")
    chart = alt.Chart(df).mark_bar().encode(**encoding)
    rule = _reference_rule(config)
    if rule is not None:
        chart = alt.layer(chart, rule, data=df)
    return ChartResult(chart.properties(width="container", height=340),
                       _export_frame(df))


def _build_scatter(con, config, dims) -> ChartResult:
    sql, params = build_scatter_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims, config)
    df = _order_panels(df, config, dims)

    chart = alt.Chart(df).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("x_value:Q",
                title=_value_axis_title(config, config["scatter_x"]),
                scale=alt.Scale(zero=False)),
        y=alt.Y("y_value:Q",
                title=_value_axis_title(config, config["scatter_y"]),
                scale=alt.Scale(zero=False)),
        color=alt.Color("scenario:N", title="Scenario"),
        tooltip=list(df.columns),
    )
    if _facet_col(config) in df.columns:
        built = _facet_grid(chart, df, config, width=260, height=200)
    else:
        built = chart.properties(width="container", height=340)
    return ChartResult(built, _export_frame(df))


_BUILDERS = {
    "timeseries": _build_timeseries,
    "histogram": _build_histogram,
    "boxplot": _build_boxplot,
    "stacked_bar": _build_stacked_bar,
    "scatter": _build_scatter,
}


def build_chart(
    con: duckdb.DuckDBPyConnection, dims: dict, config: dict,
) -> ChartResult:
    """Query and build one chart, keeping the rows behind it.

    Raises :class:`ResultsExplorerError` for anything the user can fix (an
    incomplete config, an over-broad selection, an empty result) so the
    notebook can show it inline against that chart rather than failing the
    whole render pass.
    """
    _validate(config, dims)
    return _BUILDERS[config["chart_type"]](con, config, dims)


def render_chart(con: duckdb.DuckDBPyConnection, dims: dict, config: dict) -> AltairChart:
    """Just the chart from :func:`build_chart`, for callers that do not need
    its data."""
    return build_chart(con, dims, config).chart


def chart_title(config: dict, dims: dict | None = None) -> str:
    """Short human-readable summary of what a chart is showing.

    ``dims`` is optional only so a title can still be produced before a file is
    open; pass it whenever available, since it is what turns a pinned age index
    into its band label.
    """
    dims = dims or {}
    kind = {
        "timeseries": "Time series",
        "histogram": "Histogram",
        "boxplot": "Box plot",
        "stacked_bar": "Stacked bar",
        "scatter": "Scatter",
    }[config["chart_type"]]
    if config["chart_type"] == "scatter":
        what = f"{config.get('scatter_y')} vs {config.get('scatter_x')}"
    else:
        what = _metric_label(config) or "—"
    # A pinned dimension is no longer broken out, so describe it as a
    # restriction ("age 5-12") rather than as a breakdown ("by age group").
    pinned = {c for c, _ in _active_filters(config)}
    breakdown = [c for c in _AGG_GROUP_COLS[config["agg_level"]] if c not in pinned]
    if breakdown:
        where = "by " + "/".join(c.replace("_group", "").replace("subpop", "subpop")
                                 for c in breakdown)
    else:
        where = "population total"
    slices = _slice_label(config, dims)
    if slices:
        where = f"{slices}{', ' + where if breakdown else ''}"
    if is_comparison(config):
        rel = "ratio" if config.get("compare_metric") == "ratio" else "difference"
        where += f" · {rel} vs {config.get('baseline_scenario')}"
    return f"{kind}: {what} ({where})"


def chart_filename(config: dict, dims: dict | None = None, ext: str = "csv") -> str:
    """Download name for a chart, derived from the title it is shown under.

    Titles are what distinguishes one chart from another on screen, so naming
    the file after the title is what makes eight downloads in a row tellable
    apart afterwards.
    """
    return f"{slugify(chart_title(config, dims))}.{ext}"


# ---------------------------------------------------------------------------
# Bulk data export
# ---------------------------------------------------------------------------

#: What one exported row represents.
EXPORT_ROW_MODES = ("summary", "replicates")

#: Whether the day dimension is kept or summed away.
EXPORT_TIME_MODES = ("daily", "total")


def default_export_config(dims: dict) -> dict:
    """A selection that downloads something useful without any adjusting.

    Every scenario and every metric, per day, summarized across replicates --
    the shape people usually want to hand to a spreadsheet or another tool.
    """
    return {
        "scenarios": list(dims.get("scenarios") or []),
        "metrics": [m for m, _ in dims.get("metrics") or []],
        "agg_level": "population",
        "subpop_filter": None,
        "age_filter": None,
        "risk_filter": None,
        "day_range": None,
        "cumulative": False,
        # Off by default, unlike the charts' equivalent panel. A chart's total
        # panel sits beside the others and cannot be mistaken for one of them;
        # a Total *row* in a CSV is one more row in the same column, and
        # summing that column would double-count. Opt in, having read the note.
        "show_total": False,
        "row_mode": "summary",
        "time_mode": "daily",
    }


def group_columns(config: dict) -> tuple[str, ...]:
    """The dimensions this config breaks out into their own rows/panels."""
    return _AGG_GROUP_COLS.get(config.get("agg_level", "population"), ())


def supports_export_total(config: dict) -> bool:
    """Whether a combined-total row means anything at this aggregation level.

    Nothing to total at population level: the single row already is the total.
    """
    return bool(group_columns(config))


def _validate_export(config: dict) -> None:
    if not config.get("scenarios"):
        raise ResultsExplorerError("Select at least one scenario.")
    if not config.get("metrics"):
        raise ResultsExplorerError(
            "Select at least one compartment or transition variable.")
    if config.get("agg_level") not in AGG_LEVELS:
        raise ResultsExplorerError(
            f"Unknown aggregation level {config.get('agg_level')!r}")
    if config.get("row_mode", "summary") not in EXPORT_ROW_MODES:
        raise ResultsExplorerError(f"Unknown row mode {config.get('row_mode')!r}")
    if config.get("time_mode", "daily") not in EXPORT_TIME_MODES:
        raise ResultsExplorerError(f"Unknown time mode {config.get('time_mode')!r}")


def export_is_cumulative(config: dict, dims: dict) -> bool:
    """Whether a running total actually applies to this export.

    Cumulative is a running total over days, so it means nothing once the days
    have already been summed away -- and nothing for a compartment, which is a
    level rather than a flow. Both are handled by silently not applying it
    rather than by refusing the export, since here (unlike a chart, which draws
    exactly one series) the selection may legitimately mix flows and levels.
    """
    if not config.get("cumulative"):
        return False
    if config.get("time_mode", "daily") != "daily":
        return False
    kinds = dict(dims.get("metrics") or [])
    return any(kinds.get(m) == "transition" for m in config.get("metrics") or [])


def build_export_query(config: dict, dims: dict) -> tuple[str, list[Any]]:
    """Tidy rows for download, one per (scenario, metric, ...) combination.

    Differs from every chart query in one important way: metrics stay in their
    own rows instead of being summed together. A chart draws a single series,
    so summing "IV_to_H + I_to_H" is the point; a downloaded table is something
    the user will pivot themselves, and a sum they cannot undo is a loss.
    """
    _validate_export(config)
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]
    group_sql = "".join(f", {c}" for c in group_cols)
    daily = config.get("time_mode", "daily") == "daily"
    cumulative = export_is_cumulative(config, dims)

    table = _source_table(config)
    scen_ph, params = _in_clause(config["scenarios"])
    metric_ph, metric_params = _in_clause(list(config["metrics"]))
    params += metric_params
    where = [f"scenario IN ({scen_ph})", f"compartment IN ({metric_ph})"]
    filter_where, filter_params = _filter_clauses(config)
    where += filter_where
    params += filter_params
    day_range = config.get("day_range")
    if day_range:
        where.append("day BETWEEN ? AND ?")
        params += [int(day_range[0]), int(day_range[1])]

    keys = f"scenario, compartment AS metric, rep, param_set, day{group_sql}"
    stages = [
        f"  SELECT {keys}, SUM(value) AS value\n"
        f"  FROM {table}\n"
        f"  WHERE {' AND '.join(where)}\n"
        f"  GROUP BY scenario, compartment, rep, param_set, day{group_sql}\n"
    ]

    if group_cols and config.get("show_total"):
        # Summed across groups per replicate, here at the head of the chain, so
        # every later stage -- running total, day total, median across
        # replicates -- treats it exactly like a real group. That makes it the
        # total in its own right rather than the total of the other rows'
        # summaries, which for a median is not the same number.
        stages.append(_total_union_sql(
            f"export_{len(stages) - 1}", group_cols,
            ["scenario", "metric", "rep", "param_set", "day"], ["value"]))

    if cumulative:
        # Applied per metric and only to the flows: a mixed selection keeps its
        # compartments as the levels they are, so one export can carry both.
        tvs = [m for m, k in dims.get("metrics") or [] if k == "transition"]
        tv_ph, tv_params = _in_clause(tvs)
        part = "".join(f", {c}" for c in group_cols)
        stages.append(
            f"  SELECT scenario, metric, rep, param_set, day{group_sql},\n"
            f"         CASE WHEN metric IN ({tv_ph}) THEN SUM(value) OVER (\n"
            f"           PARTITION BY scenario, metric, rep, param_set{part}\n"
            f"           ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
            f"         ) ELSE value END AS value\n"
            f"  FROM export_{len(stages) - 1}\n"
        )
        params += tv_params

    if not daily:
        # One number per replicate for the whole selected window, summed before
        # any median so the summary describes replicate totals rather than the
        # total of daily medians.
        stages.append(
            f"  SELECT scenario, metric, rep, param_set{group_sql},\n"
            f"         SUM(value) AS value\n"
            f"  FROM export_{len(stages) - 1}\n"
            f"  GROUP BY scenario, metric, rep, param_set{group_sql}\n"
        )

    names = [f"export_{i}" for i in range(len(stages) - 1)] + ["export_rows"]
    cte = ",\n".join(f"{n} AS (\n{s})" for n, s in zip(names, stages))

    day_sql = ", day" if daily else ""
    if config.get("row_mode", "summary") == "replicates":
        sql = (
            f"WITH {cte}\n"
            f"SELECT scenario, metric, rep, param_set{day_sql}{group_sql}, value\n"
            f"FROM export_rows\n"
            f"ORDER BY scenario, metric, rep{day_sql}{group_sql}"
        )
        return sql, params

    # Summary across replicates. Percentiles are dropped for a single-replicate
    # file, where they would be three more columns all equal to the median.
    stats = ["median(value) AS median_value", "AVG(value) AS mean_value"]
    if (dims.get("n_reps") or 1) > 1:
        stats += [
            "quantile_cont(value, 0.025) AS p2_5",
            "quantile_cont(value, 0.975) AS p97_5",
            "MIN(value) AS min_value",
            "MAX(value) AS max_value",
            "COUNT(*) AS n_reps",
        ]
    stats_sql = ",\n       ".join(stats)
    sql = (
        f"WITH {cte}\n"
        f"SELECT scenario, metric{day_sql}{group_sql},\n"
        f"       {stats_sql}\n"
        f"FROM export_rows\n"
        f"GROUP BY scenario, metric{day_sql}{group_sql}\n"
        f"ORDER BY scenario, metric{day_sql}{group_sql}"
    )
    return sql, params


#: Column order for a downloaded table: what each row *is*, then what it
#: measures. Anything unlisted (the value columns) keeps its query order at the
#: end.
_EXPORT_LEAD_COLS = (
    "scenario", "metric", "day", "date", "subpop", "age_group", "risk_group",
    "rep", "param_set",
)


def run_export(
    con: duckdb.DuckDBPyConnection, dims: dict, config: dict,
) -> pd.DataFrame:
    """Run :func:`build_export_query` and dress the result for download.

    Applies the same labelling the charts use -- real age bands, real dates --
    so a downloaded table reads the same way the plots above it do.
    """
    sql, params = build_export_query(config, dims)
    # One row past the cap, so an over-broad selection is caught by the row
    # count rather than by the machine running out of memory building it.
    df = _run(con, f"SELECT * FROM (\n{sql}\n) LIMIT {EXPORT_ROW_LIMIT + 1}", params)
    if len(df) > EXPORT_ROW_LIMIT:
        raise ResultsExplorerError(
            f"This selection exceeds {EXPORT_ROW_LIMIT:,} rows. Narrow it — "
            f"fewer metrics or scenarios, a shorter day range, a coarser "
            f"aggregation level, or summary rows instead of every replicate."
        )
    if df.empty:
        raise ResultsExplorerError(
            "No data for this selection. Check the metric and scenario "
            "choices — a transition variable may not have been recorded."
        )
    group_cols = [c for c in _AGG_GROUP_COLS[config["agg_level"]] if c in df.columns]
    if config.get("show_total"):
        df = _label_total_slices(df, group_cols)
    df = _label_groups(df, dims)
    df, _, _ = _add_date(df, dims)
    df = _order_export_rows(df, group_cols, dims)
    lead = [c for c in _EXPORT_LEAD_COLS if c in df.columns]
    return df.reindex(columns=lead + [c for c in df.columns if c not in lead])


def _order_export_rows(
    df: pd.DataFrame, group_cols: Sequence[str], dims: dict,
) -> pd.DataFrame:
    """Sort the table the way the charts order their small multiples.

    SQL orders the group columns by their raw keys, which puts the combined
    total wherever its sentinel happens to sort -- first for the integer
    ``*_group`` columns (-1), mid-alphabet for ``subpop``. :func:`_panel_order`
    is the same ranking the facet grids use: the model's own age-band order,
    numeric where it can be, alphabetical where it cannot, total last.
    """
    if not group_cols:
        return df
    ranks = {str(c): {v: i for i, v in enumerate(_panel_order(df, c, dims))}
             for c in group_cols}
    keys = [c for c in ("scenario", "metric", "day") if c in df.columns]
    keys += list(group_cols)
    return df.sort_values(
        keys,
        key=lambda s: s.map(ranks[str(s.name)]) if str(s.name) in ranks else s,
        kind="stable",
    ).reset_index(drop=True)


def export_filename(config: dict) -> str:
    """Download name describing the selection, so several exports stay apart."""
    parts = [
        "results",
        config.get("agg_level", "population"),
        config.get("row_mode", "summary"),
        "total" if config.get("time_mode", "daily") == "total" else "daily",
    ]
    if config.get("cumulative"):
        parts.append("cumulative")
    if config.get("show_total") and supports_export_total(config):
        parts.append("with_total")
    return f"{slugify('_'.join(parts))}.csv"
