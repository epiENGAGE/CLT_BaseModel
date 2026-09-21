"""
results_io.py — On-disk format for simulation results.

The storage half of the results pipeline, shared by every writer (the model
builder notebook's Analysis tab export, generated ``run_simulation.py``
scripts) and reader (``generic_core/examples/_results_explorer_lib.py`` and
the results explorer notebook). Kept free of any charting or notebook
dependency so the run scripts can import it cheaply.

Both tables use the same tidy row schema:

``results``
    scenario, rep, param_set, compartment, kind, day, value
``results_full``
    the same plus subpop, age_group, risk_group

plus an optional ``meta`` key/value table (start date, age-group labels,
scenario order, ...). A results source is a Hive-partitioned Parquet
directory (current format, see :func:`write_results_parquet`), a SQLite
``.db``, or a legacy ``.json`` export (see :func:`load_source`).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import duckdb


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
    place -- a legacy ``.json`` export, which is converted once to a sibling
    ``.db`` and then attached (see :func:`convert_json_to_sqlite`) -- or a
    directory of Hive-partitioned Parquet (written by the Analysis tab's
    export or ``run_simulation.py``; see :func:`write_results_parquet`),
    which is read directly (much faster for large ``results_full`` scans).

    ``progress`` is an optional ``callable(str)`` used to report conversion
    steps, since that path can take a while on a multi-GB file.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ResultsExplorerError(f"No such file: {path}")

    if path.is_dir():
        return _load_parquet_source(path)

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
# Parquet source (partitioned by scenario, for faster-than-SQLite scans)
# ---------------------------------------------------------------------------

#: Marks a directory as a converted results source, and records enough to
#: sanity-check it on open (schema version, which tables it holds).
_PARQUET_MANIFEST_NAME = "_manifest.json"

_PARQUET_SCHEMA_VERSION = 1


def create_results_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create empty native ``results``/``results_full`` tables on ``con``,
    ready for :meth:`duckdb.DuckDBPyConnection.append`.

    Shared by every writer of this schema (``_nb_analysis.py``'s Analysis tab
    export, the generated ``run_simulation.py``) so the column list and types
    live in one place. ``DOUBLE`` rather than SQLite's ``REAL`` for ``value``:
    SQLite's ``REAL`` is an 8-byte float, and DuckDB's own ``REAL`` is only
    4 bytes -- ``DOUBLE`` is the 8-byte type here.
    """
    con.execute(
        "CREATE TABLE results "
        "(scenario TEXT, rep INTEGER, param_set INTEGER, compartment TEXT, "
        "kind TEXT, day INTEGER, value DOUBLE)"
    )
    con.execute(
        "CREATE TABLE results_full "
        "(scenario TEXT, rep INTEGER, param_set INTEGER, compartment TEXT, "
        "kind TEXT, subpop TEXT, age_group INTEGER, risk_group INTEGER, "
        "day INTEGER, value DOUBLE)"
    )


def write_results_parquet(
    con: duckdb.DuckDBPyConnection, out_dir: str | Path, *,
    meta: dict[str, Any] | None = None, progress=None,
) -> Path:
    """Write an open connection's ``results``/``results_full`` (tables or
    views) out as Hive-partitioned Parquet -- the shared last step behind
    a run's own writer (native DuckDB tables built up with
    :meth:`duckdb.DuckDBPyConnection.append` as it runs -- faster than
    SQLite's row-by-row inserts). Partitioning is on ``(scenario,
    compartment)`` -- the two columns every chart query filters on first --
    rather than a global sort, which needs an external-sort spill that can
    exceed the table's own size.

    ``meta`` is written as ``meta.json`` directly from a plain dict (native
    Python values, not pre-JSON-encoded).
    """
    import shutil

    out_dir = Path(out_dir).expanduser()
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    for table in ("results", "results_full"):
        if progress:
            progress(f"Writing {table} to Parquet…")
        con.execute(
            f"COPY (SELECT * FROM {table}) "
            f"TO '{(tmp_dir / table).as_posix()}' "
            f"(FORMAT PARQUET, PARTITION_BY (scenario, compartment), "
            f"OVERWRITE_OR_IGNORE true, COMPRESSION ZSTD)"
        )

    if meta is not None:
        if progress:
            progress("Writing meta…")
        (tmp_dir / "meta.json").write_text(
            json.dumps({k: json.dumps(v) for k, v in meta.items()}),
            encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": _PARQUET_SCHEMA_VERSION,
        "tables": ["results", "results_full"],
    }
    (tmp_dir / _PARQUET_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.replace(out_dir)
    return out_dir


def _load_parquet_source(path: Path) -> duckdb.DuckDBPyConnection:
    manifest_path = path / _PARQUET_MANIFEST_NAME
    if not manifest_path.exists():
        raise ResultsExplorerError(
            f"{path} is a directory but not a results source "
            f"(missing {_PARQUET_MANIFEST_NAME})."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = {"results", "results_full"} - set(manifest.get("tables", []))
    if missing:
        raise ResultsExplorerError(
            f"{path} is missing table(s) {sorted(missing)}.")

    con = duckdb.connect()
    for table in ("results", "results_full"):
        glob = (path / table / "**" / "*.parquet").as_posix()
        con.execute(
            f"CREATE VIEW {table} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true)"
        )
    meta_path = path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)", list(meta.items()))
    return con


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
