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
import sqlite3
from pathlib import Path
from typing import Any, Literal, Sequence

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
    if not config.get("cumulative"):
        return f"{alias} AS (\n{body})", params

    part_sql = "".join(f", {c}" for c in group_cols)
    sql = (
        f"{alias}_daily AS (\n{body}),\n"
        f"{alias} AS (\n"
        f"  SELECT scenario, rep, param_set, day{group_sql},\n"
        f"         SUM(value) OVER (\n"
        f"           PARTITION BY scenario, rep, param_set{part_sql}\n"
        f"           ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n"
        f"         ) AS value\n"
        f"  FROM {alias}_daily\n"
        f")"
    )
    return sql, params


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


def build_scatter_query(config: dict) -> tuple[str, list[Any]]:
    """Two metrics per replicate, as X and Y.

    When comparing, both axes become per-replicate differences (or ratios)
    against the baseline, so the cloud sits around the no-effect point rather
    than around the two scenarios' levels.
    """
    group_cols = _AGG_GROUP_COLS[config["agg_level"]]

    if not is_comparison(config):
        body, params = _scatter_per_rep_sql(config, config["scenarios"])
        return f"{body}  ORDER BY scenario, rep", params

    baseline = config["baseline_scenario"]
    others = [s for s in config["scenarios"] if s != baseline]
    body, params = _scatter_per_rep_sql(config, [baseline] + others)
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


def _label_groups(df: pd.DataFrame, dims: dict) -> pd.DataFrame:
    """Swap 0-based age indices for real band labels where available."""
    if "age_group" not in df.columns:
        return df
    labels = dims.get("age_group_labels") or []
    if not labels:
        return df
    df = df.copy()
    df["age_group"] = df["age_group"].map(
        lambda i: labels[int(i)] if 0 <= int(i) < len(labels) else str(i)
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


def _reference_rule(config: dict) -> alt.Chart | None:
    """A dashed line at the no-effect value, so a comparison is readable
    against something rather than floating."""
    reference = comparison_reference(config)
    if reference is None:
        return None
    return (alt.Chart(pd.DataFrame({"_ref": [reference]}))
            .mark_rule(strokeDash=[4, 4], color="grey")
            .encode(y="_ref:Q"))


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
# Chart builders
# ---------------------------------------------------------------------------


def _build_timeseries(con, config, dims) -> AltairChart:
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
    df = _label_groups(df, dims)
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
    reference = comparison_reference(config)

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
    if reference is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"_ref": [reference]}))
            .mark_rule(strokeDash=[4, 4], color="grey")
            .encode(y="_ref:Q")
        )

    chart = alt.layer(*layers)
    facet = _facet_col(config)
    if facet and facet in df.columns:
        chart = chart.properties(width=320, height=180).facet(
            facet=alt.Facet(f"{facet}:N", title=facet.replace("_", " ")), columns=3
        )
    else:
        chart = chart.properties(width="container", height=320)
    return chart.resolve_scale(y=_y_resolve(config))


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


def _build_histogram(con, config, dims) -> AltairChart:
    _needs_replicates(config, dims)
    sql, params = build_per_rep_totals_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims)

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
    reference = comparison_reference(config)
    if reference is not None:
        # Vertical here, since the compared quantity is on x for a histogram.
        chart = alt.layer(chart, alt.Chart(pd.DataFrame({"_ref": [reference]}))
                          .mark_rule(strokeDash=[4, 4], color="grey")
                          .encode(x="_ref:Q"))
    facet = _facet_col(config)
    if facet and facet in df.columns:
        return chart.properties(width=280, height=180).facet(
            facet=alt.Facet(f"{facet}:N", title=facet.replace("_", " ")), columns=3
        ).resolve_scale(y=_y_resolve(config))
    return chart.properties(width="container", height=320)


def _build_boxplot(con, config, dims) -> AltairChart:
    _needs_replicates(config, dims)
    sql, params = build_per_rep_totals_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims)

    chart = alt.Chart(df).mark_boxplot(extent="min-max").encode(
        x=alt.X("scenario:N", title="Scenario"),
        y=alt.Y("total_value:Q",
                title=_value_axis_title(config, _metric_label(config)),
                # A difference straddles zero, so forcing the axis through it
                # would waste most of the range on empty space.
                scale=alt.Scale(zero=not is_comparison(config))),
        color=alt.Color("scenario:N", title="Scenario", legend=None),
    )
    rule = _reference_rule(config)
    if rule is not None:
        chart = alt.layer(chart, rule)
    facet = _facet_col(config)
    if facet and facet in df.columns:
        return chart.properties(width=220, height=200).facet(
            facet=alt.Facet(f"{facet}:N", title=facet.replace("_", " ")), columns=3
        ).resolve_scale(y=_y_resolve(config))
    return chart.properties(width="container", height=320)


def _build_stacked_bar(con, config, dims) -> AltairChart:
    sql, params = build_stacked_bar_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims)

    stack_col = _AGG_GROUP_COLS[config["agg_level"]][-1]
    comparing = is_comparison(config)
    y_title = (
        f"{_metric_label(config)} — median "
        f"{'ratio to' if config.get('compare_metric') == 'ratio' else 'difference from'} "
        f"{config.get('baseline_scenario')}"
        if comparing else f"{_metric_label(config)} (median total)"
    )
    encoding: dict[str, Any] = {
        "x": alt.X("scenario:N", title="Scenario"),
        "y": alt.Y("median_value:Q", title=y_title,
                   scale=alt.Scale(zero=not comparing)),
        "color": alt.Color(f"{stack_col}:N", title=stack_col.replace("_", " ")),
        "tooltip": list(df.columns),
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
        chart = alt.layer(chart, rule)
    return chart.properties(width="container", height=340)


def _build_scatter(con, config, dims) -> AltairChart:
    sql, params = build_scatter_query(config)
    df = _run(con, sql, params)
    _guard_size(df)
    df = _drop_empty_groups(df, config)
    df = _label_groups(df, dims)

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
    facet = _facet_col(config)
    if facet and facet in df.columns:
        return chart.properties(width=260, height=200).facet(
            facet=alt.Facet(f"{facet}:N", title=facet.replace("_", " ")), columns=3
        ).resolve_scale(y=_y_resolve(config))
    return chart.properties(width="container", height=340)


_BUILDERS = {
    "timeseries": _build_timeseries,
    "histogram": _build_histogram,
    "boxplot": _build_boxplot,
    "stacked_bar": _build_stacked_bar,
    "scatter": _build_scatter,
}


def render_chart(con: duckdb.DuckDBPyConnection, dims: dict, config: dict) -> AltairChart:
    """Query and build one chart from its config.

    Raises :class:`ResultsExplorerError` for anything the user can fix (an
    incomplete config, an over-broad selection, an empty result) so the
    notebook can show it inline against that chart rather than failing the
    whole render pass.
    """
    _validate(config, dims)
    return _BUILDERS[config["chart_type"]](con, config, dims)


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
