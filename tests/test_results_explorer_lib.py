"""
Tests for generic_core.examples._results_explorer_lib.

Everything runs against small synthetic SQLite fixtures built in a tmp_path,
so nothing here depends on the multi-GB MA_vax example outputs. The fixtures
reproduce the exact schema both producers write (`results`, `results_full`,
and the optional `meta` table), which is what the library contracts on.

The values are chosen so expected results are computable by hand -- that is
the point of the comparison tests, which check the actual arithmetic of the
paired/unpaired baseline comparison rather than merely that a query runs.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

# The library lives in examples/, which is not an importable package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generic_core" / "examples"))

import _results_explorer_lib as lib  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DAYS = 3
AGE_GROUPS = 2

#: Values for the single metric "new_H", constructed so paired and unpaired
#: comparison give *different* answers -- otherwise the tests below could not
#: tell the two code paths apart.
#:
#: The baseline itself varies a lot between replicates (_REP_BASE_SCALE),
#: standing in for the shared RNG/parameter draw. "treated" is that same
#: replicate's baseline scaled by _REP_FACTOR. Pairing cancels the shared
#: variation and recovers the factors exactly; comparing summaries does not.
_BASE = {0: 10.0, 1: 20.0}          # per age group, before the replicate scale
_REP_BASE_SCALE = {0: 1.0, 1: 3.0}  # replicate-to-replicate spread, both scenarios
_REP_FACTOR = {0: 0.5, 1: 0.9}      # treated/baseline within a replicate

#: Population total per day, by replicate: 30 x scale.
_POP_BASELINE = {r: sum(_BASE.values()) * s for r, s in _REP_BASE_SCALE.items()}
_POP_TREATED = {r: v * _REP_FACTOR[r] for r, v in _POP_BASELINE.items()}


def _rows():
    """Yield (scenario, rep, compartment, kind, subpop, age, risk, day, value)."""
    for rep in (0, 1):
        for age in range(AGE_GROUPS):
            for day in range(1, DAYS + 1):
                base = _BASE[age] * _REP_BASE_SCALE[rep]
                yield ("baseline", rep, "new_H", "transition", "pop", age, 0, day, base)
                yield ("treated", rep, "new_H", "transition", "pop", age, 0, day,
                       base * _REP_FACTOR[rep])


def _build_db(path: Path, *, with_meta: bool) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE results (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, day INTEGER, value REAL)")
    con.execute(
        "CREATE TABLE results_full (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, subpop TEXT, age_group INTEGER,"
        " risk_group INTEGER, day INTEGER, value REAL)")

    full = list(_rows())
    con.executemany(
        "INSERT INTO results_full VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(s, r, None, c, k, sp, a, rg, d, v) for (s, r, c, k, sp, a, rg, d, v) in full])
    # `results` is the same data summed over subpop/age/risk.
    totals: dict[tuple, float] = {}
    for (s, r, c, k, _sp, _a, _rg, d, v) in full:
        totals[(s, r, c, k, d)] = totals.get((s, r, c, k, d), 0.0) + v
    con.executemany(
        "INSERT INTO results VALUES (?,?,?,?,?,?,?)",
        [(s, r, None, c, k, d, v) for (s, r, c, k, d), v in totals.items()])

    if with_meta:
        meta = {
            "schema_version": 1,
            "source": "test",
            "start_date": "2025-09-01",
            "num_days": DAYS,
            "num_age_groups": AGE_GROUPS,
            "num_risk_groups": 1,
            "age_group_labels": ["0-17", "18+"],
            "subpop_names": ["pop"],
            "scenarios": ["baseline", "treated"],
            "compartments": [],
            "transition_vars": ["new_H"],
            "n_reps": 2,
        }
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany("INSERT INTO meta VALUES (?,?)",
                        [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()
    con.close()
    return path


@pytest.fixture
def db_with_meta(tmp_path):
    return _build_db(tmp_path / "with_meta.db", with_meta=True)


@pytest.fixture
def db_no_meta(tmp_path):
    return _build_db(tmp_path / "no_meta.db", with_meta=False)


@pytest.fixture
def con_meta(db_with_meta):
    con = lib.load_source(db_with_meta)
    yield con
    con.close()


@pytest.fixture
def con_no_meta(db_no_meta):
    con = lib.load_source(db_no_meta)
    yield con
    con.close()


def _config(con_dims, **overrides):
    cfg = lib.default_chart_config(0, "timeseries", con_dims)
    cfg.update(metrics=["new_H"], scenarios=["baseline", "treated"],
               agg_level="population")
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_source_rejects_missing_file(tmp_path):
    with pytest.raises(lib.ResultsExplorerError, match="No such file"):
        lib.load_source(tmp_path / "nope.db")


def test_load_source_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "results.parquet"
    p.write_bytes(b"")
    with pytest.raises(lib.ResultsExplorerError, match="Unrecognized"):
        lib.load_source(p)


def test_load_source_rejects_db_without_expected_tables(tmp_path):
    p = tmp_path / "wrong.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(lib.ResultsExplorerError, match="missing table"):
        lib.load_source(p)


def test_load_source_exposes_both_tables(con_meta):
    assert con_meta.execute("SELECT COUNT(*) FROM results").fetchone()[0] > 0
    assert con_meta.execute("SELECT COUNT(*) FROM results_full").fetchone()[0] > 0


# ---------------------------------------------------------------------------
# Legacy JSON conversion
# ---------------------------------------------------------------------------

def _write_legacy_json(path: Path) -> Path:
    """The pre-SQLite export format: one object, rows as arrays."""
    results, results_full = [], []
    for (s, r, _c, _k, _sp, a, rg, d, v) in _rows():
        results_full.append([s, r, None, "new_H", "transition", "pop", a, rg, d, v])
    # `results` is the same data summed over subpop/age/risk.
    totals: dict[tuple, float] = {}
    for row in results_full:
        key = (row[0], row[1], row[8])
        totals[key] = totals.get(key, 0.0) + row[9]
    for (s, r, d), v in totals.items():
        results.append([s, r, None, "new_H", "transition", d, v])

    path.write_text(json.dumps({
        "results_columns": ["scenario", "rep", "param_set", "compartment",
                            "kind", "day", "value"],
        "results": results,
        "results_full_columns": ["scenario", "rep", "param_set", "compartment",
                                 "kind", "subpop", "age_group", "risk_group",
                                 "day", "value"],
        "results_full": results_full,
    }, separators=(",", ":")))
    return path


def test_legacy_json_is_converted_and_queryable(tmp_path):
    src = _write_legacy_json(tmp_path / "analysis_results_full.json")
    con = lib.load_source(src)
    try:
        assert (tmp_path / "analysis_results_full.db").exists()
        dims = lib.discover_dims(con)
        # A legacy export carries no meta table, so labels/dates fall back.
        assert dims["has_meta"] is False
        assert sorted(dims["scenarios"]) == ["baseline", "treated"]
        assert dims["n_reps"] == 2
        # Values must survive the conversion exactly. ijson decodes JSON
        # numbers to Decimal unless told otherwise, which sqlite3 refuses to
        # bind -- this asserts the float round-trip that guards against it.
        got = con.execute(
            "SELECT value FROM results WHERE scenario='baseline' AND rep=0 "
            "AND day=1").fetchone()[0]
        assert isinstance(got, float)
        assert got == pytest.approx(_POP_BASELINE[0])
    finally:
        con.close()


def test_legacy_json_conversion_is_cached(tmp_path):
    src = _write_legacy_json(tmp_path / "analysis_results_full.json")
    seen: list[str] = []
    lib.load_source(src, progress=seen.append).close()
    assert seen, "first open should report conversion progress"
    seen.clear()
    lib.load_source(src, progress=seen.append).close()
    assert not seen, "second open should reuse the converted .db"


# ---------------------------------------------------------------------------
# Metadata / discovery
# ---------------------------------------------------------------------------

def test_discover_dims_uses_meta(con_meta):
    dims = lib.discover_dims(con_meta)
    assert dims["has_meta"] is True
    assert dims["start_date"] == "2025-09-01"
    # Scenario order comes from meta, not alphabetical DISTINCT.
    assert dims["scenarios"] == ["baseline", "treated"]
    assert dims["age_group_labels"] == ["0-17", "18+"]
    assert dims["metrics"] == [("new_H", "transition")]
    assert dims["n_reps"] == 2


def test_discover_dims_falls_back_without_meta(con_no_meta):
    dims = lib.discover_dims(con_no_meta)
    assert dims["has_meta"] is False
    assert sorted(dims["scenarios"]) == ["baseline", "treated"]
    assert dims["subpops"] == ["pop"]
    assert dims["num_age_groups"] == AGE_GROUPS
    assert dims["num_risk_groups"] == 1
    assert dims["n_reps"] == 2
    # No start date and no labels -> index-based fallbacks.
    assert dims.get("start_date") is None
    assert dims["age_group_labels"] == ["0", "1"]


def test_read_meta_absent_returns_none(con_no_meta):
    assert lib.read_meta(con_no_meta) is None


def test_day_range_discovered(con_meta):
    dims = lib.discover_dims(con_meta)
    assert (dims["day_min"], dims["day_max"]) == (1, DAYS)


# ---------------------------------------------------------------------------
# Aggregation correctness
# ---------------------------------------------------------------------------

def test_population_timeseries_matches_hand_computation(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"])
    sql, params = lib.build_timeseries_query(cfg, with_ci=False)
    df = con_meta.execute(sql, params).df()
    # Population totals per replicate are 30 and 90; with two replicates the
    # median sits midway, at 60, on every day.
    expected = (_POP_BASELINE[0] + _POP_BASELINE[1]) / 2
    assert len(df) == DAYS
    assert df["median_value"].tolist() == pytest.approx([expected] * DAYS)


def test_age_group_split_sums_back_to_population(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"], agg_level="age_group")
    sql, params = lib.build_timeseries_query(cfg, with_ci=False)
    df = con_meta.execute(sql, params).df()
    assert len(df) == DAYS * AGE_GROUPS
    day1 = df[df["day"] == 1].set_index("age_group")["median_value"]
    # Per age group, the median across the two replicate scales (1x, 3x).
    for age, base in _BASE.items():
        expected = (base * _REP_BASE_SCALE[0] + base * _REP_BASE_SCALE[1]) / 2
        assert day1[age] == pytest.approx(expected)
    assert day1.sum() == pytest.approx(
        (_POP_BASELINE[0] + _POP_BASELINE[1]) / 2)


def test_per_rep_totals_are_per_replicate(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["treated"])
    sql, params = lib.build_per_rep_totals_query(cfg)
    df = con_meta.execute(sql, params).df().set_index("rep")["total_value"]
    for rep in (0, 1):
        assert df[rep] == pytest.approx(_POP_TREATED[rep] * DAYS)


# ---------------------------------------------------------------------------
# Baseline comparison: paired vs unpaired
# ---------------------------------------------------------------------------

def test_paired_ratio_recovers_per_replicate_factors(con_meta):
    """Paired differencing sees each replicate's own factor.

    Within a replicate the ratio is exactly _REP_FACTOR (0.5 and 0.9), no
    matter how much the baseline itself varies between replicates -- that
    shared variation cancels. Median of {0.5, 0.9} = 0.7.
    """
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenario_mode="compare_baseline",
                  baseline_scenario="baseline", pairing="paired",
                  compare_metric="ratio")
    sql, params = lib.build_comparison_query(cfg, with_ci=True)
    df = con_meta.execute(sql, params).df()
    assert set(df["scenario"]) == {"treated"}
    assert df["median_value"].iloc[0] == pytest.approx(0.7)
    # The interval spans exactly the two replicate-level ratios.
    assert df["p_lo"].iloc[0] == pytest.approx(0.5, abs=0.02)
    assert df["p_hi"].iloc[0] == pytest.approx(0.9, abs=0.02)


def test_unpaired_ratio_differs_from_paired(con_meta):
    """Unpaired compares summaries, so the baseline spread does not cancel.

    median(treated) = median(15, 81) = 48, median(baseline) = median(30, 90)
    = 60, giving 48/60 = 0.8 -- not the 0.7 the paired route returns. The two
    genuinely answer different questions, which is why both are offered.
    """
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenario_mode="compare_baseline",
                  baseline_scenario="baseline", pairing="unpaired",
                  compare_metric="ratio")
    sql, params = lib.build_comparison_query(cfg, with_ci=True)
    df = con_meta.execute(sql, params).df()
    expected = (
        (_POP_TREATED[0] + _POP_TREATED[1]) / 2
        / ((_POP_BASELINE[0] + _POP_BASELINE[1]) / 2)
    )
    assert expected == pytest.approx(0.8)
    assert df["median_value"].iloc[0] == pytest.approx(0.8)


def test_paired_difference_matches_hand_computation(con_meta):
    """Per-pair differences are 15-30 = -15 and 81-90 = -9; median -12.

    For a difference the two routes coincide (with two replicates the median
    is the mean, and differencing is linear) -- unlike the ratio above. The
    pairing choice therefore matters most for ratios.
    """
    dims = lib.discover_dims(con_meta)
    expected = (
        (_POP_TREATED[0] - _POP_BASELINE[0])
        + (_POP_TREATED[1] - _POP_BASELINE[1])
    ) / 2
    assert expected == pytest.approx(-12.0)
    for pairing in ("paired", "unpaired"):
        cfg = _config(dims, scenario_mode="compare_baseline",
                      baseline_scenario="baseline", pairing=pairing,
                      compare_metric="difference")
        sql, params = lib.build_comparison_query(cfg, with_ci=False)
        df = con_meta.execute(sql, params).df()
        assert df["median_value"].iloc[0] == pytest.approx(-12.0)


def test_comparison_excludes_baseline_from_results(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenario_mode="compare_baseline",
                  baseline_scenario="baseline", pairing="paired")
    sql, params = lib.build_comparison_query(cfg, with_ci=False)
    df = con_meta.execute(sql, params).df()
    assert "baseline" not in set(df["scenario"])


def test_comparison_pulls_in_baseline_even_if_unselected(con_meta):
    """The baseline must be queried even when the user did not tick it."""
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["treated"], scenario_mode="compare_baseline",
                  baseline_scenario="baseline", pairing="paired",
                  compare_metric="difference")
    sql, params = lib.build_comparison_query(cfg, with_ci=False)
    df = con_meta.execute(sql, params).df()
    # Same answer as when the baseline was explicitly selected.
    assert df["median_value"].iloc[0] == pytest.approx(-12.0)


# ---------------------------------------------------------------------------
# Validation and rendering
# ---------------------------------------------------------------------------

def test_render_rejects_missing_scenarios(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=[])
    with pytest.raises(lib.ResultsExplorerError, match="at least one scenario"):
        lib.render_chart(con_meta, dims, cfg)


def test_render_rejects_missing_metrics(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, metrics=[])
    with pytest.raises(lib.ResultsExplorerError, match="at least one metric"):
        lib.render_chart(con_meta, dims, cfg)


def test_render_rejects_comparison_without_other_scenario(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"], scenario_mode="compare_baseline",
                  baseline_scenario="baseline")
    with pytest.raises(lib.ResultsExplorerError, match="other than the baseline"):
        lib.render_chart(con_meta, dims, cfg)


def test_stacked_bar_rejects_population_level(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, chart_type="stacked_bar", agg_level="population")
    with pytest.raises(lib.ResultsExplorerError, match="needs something to stack"):
        lib.render_chart(con_meta, dims, cfg)


def test_stacked_bar_default_is_renderable(con_meta):
    """Adding a stacked bar must not land on a config that errors."""
    dims = lib.discover_dims(con_meta)
    cfg = lib.default_chart_config(0, "stacked_bar", dims)
    cfg["scenarios"] = ["baseline", "treated"]
    assert cfg["agg_level"] != "population"
    assert lib.render_chart(con_meta, dims, cfg).to_dict()


@pytest.mark.parametrize("chart_type", ["histogram", "boxplot"])
def test_replicate_distribution_charts_explain_single_replicate(tmp_path, chart_type):
    """With one replicate these collapse to a point and look blank.

    Saying so is more useful than rendering a hair-thin bar or a degenerate
    box that reads as an empty plot.
    """
    path = _build_db(tmp_path / "one_rep.db", with_meta=True)
    con = sqlite3.connect(path)
    con.execute("DELETE FROM results WHERE rep = 1")
    con.execute("DELETE FROM results_full WHERE rep = 1")
    con.execute("UPDATE meta SET value = '1' WHERE key = 'n_reps'")
    con.commit()
    con.close()

    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        cfg = _config(dims, chart_type=chart_type)
        with pytest.raises(lib.ResultsExplorerError, match="only one replicate"):
            lib.render_chart(dcon, dims, cfg)
    finally:
        dcon.close()


def test_render_reports_empty_selection(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, metrics=["not_a_real_metric"])
    with pytest.raises(lib.ResultsExplorerError, match="No data"):
        lib.render_chart(con_meta, dims, cfg)


@pytest.mark.parametrize("chart_type", lib.CHART_TYPES)
def test_every_chart_type_renders(con_meta, chart_type):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, chart_type=chart_type,
                  agg_level="age_group" if chart_type == "stacked_bar" else "population",
                  scatter_x="new_H", scatter_y="new_H")
    chart = lib.render_chart(con_meta, dims, cfg)
    # to_dict() is what actually serializes for the browser, so it catches
    # encoding errors a bare construction would not.
    assert chart.to_dict()


@pytest.mark.parametrize("agg_level", lib.AGG_LEVELS)
def test_every_agg_level_renders(con_meta, agg_level):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, agg_level=agg_level)
    assert lib.render_chart(con_meta, dims, cfg).to_dict()


def test_timeseries_uses_real_dates_with_meta(con_meta):
    dims = lib.discover_dims(con_meta)
    spec = lib.render_chart(con_meta, dims, _config(dims)).to_dict()
    assert "2025-09-01" in json.dumps(spec)


def test_timeseries_falls_back_to_day_index_without_meta(con_no_meta):
    dims = lib.discover_dims(con_no_meta)
    spec = lib.render_chart(con_no_meta, dims, _config(dims)).to_dict()
    text = json.dumps(spec)
    assert "2025-09-01" not in text
    assert '"day"' in text


def test_single_replicate_suppresses_interval_band(tmp_path):
    """A one-replicate run has no spread; the band would be zero-width."""
    path = _build_db(tmp_path / "one_rep.db", with_meta=True)
    con = sqlite3.connect(path)
    con.execute("DELETE FROM results WHERE rep = 1")
    con.execute("DELETE FROM results_full WHERE rep = 1")
    con.execute("UPDATE meta SET value = '1' WHERE key = 'n_reps'")
    con.commit()
    con.close()

    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        assert dims["n_reps"] == 1
        spec = lib.render_chart(dcon, dims, _config(dims, show_uncertainty=True)).to_dict()
        assert "p_lo" not in json.dumps(spec)
    finally:
        dcon.close()


def test_soft_row_limit_is_enforced(con_meta, monkeypatch):
    dims = lib.discover_dims(con_meta)
    monkeypatch.setattr(lib, "SOFT_ROW_LIMIT", 1)
    with pytest.raises(lib.ResultsExplorerError, match="Narrow it"):
        lib.render_chart(con_meta, dims, _config(dims))


def test_all_agg_level_separates_residual_dimensions(tmp_path):
    """agg_level="all" groups by subpop, age AND risk, but only one can be
    the facet. The others must get their own channel, or the line chart draws
    one zig-zagging line through several unrelated series sharing an x value.
    """
    path = tmp_path / "multi.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE results (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, day INTEGER, value REAL)")
    con.execute(
        "CREATE TABLE results_full (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, subpop TEXT, age_group INTEGER,"
        " risk_group INTEGER, day INTEGER, value REAL)")
    full, totals = [], {}
    for subpop in ("north", "south"):
        for age in (0, 1):
            for risk in (0, 1):
                for day in range(1, DAYS + 1):
                    v = (age + 1) * (1 if subpop == "north" else 2) * (risk + 1)
                    full.append(("s1", 0, None, "new_H", "transition",
                                 subpop, age, risk, day, float(v)))
                    totals[day] = totals.get(day, 0.0) + v
    con.executemany("INSERT INTO results_full VALUES (?,?,?,?,?,?,?,?,?,?)", full)
    con.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                    [("s1", 0, None, "new_H", "transition", d, v)
                     for d, v in totals.items()])
    con.commit()
    con.close()

    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        assert dims["subpops"] == ["north", "south"]
        cfg = lib.default_chart_config(0, "timeseries", dims)
        cfg.update(metrics=["new_H"], scenarios=["s1"], agg_level="all")
        spec = json.dumps(lib.render_chart(dcon, dims, cfg).to_dict())
        # Faceted on one dimension, with the remaining two distinguished
        # rather than silently collapsed together.
        assert "age_group" in spec
        assert "strokeDash" in spec
    finally:
        dcon.close()


def test_single_subpop_gets_no_redundant_series_legend(con_meta):
    """One subpop and one risk group means nothing residual to distinguish."""
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, agg_level="all")
    spec = json.dumps(lib.render_chart(con_meta, dims, cfg).to_dict())
    assert "strokeDash" not in spec


def test_chart_title_mentions_comparison(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenario_mode="compare_baseline",
                  baseline_scenario="baseline", compare_metric="ratio")
    title = lib.chart_title(cfg)
    assert "ratio vs baseline" in title


# ---------------------------------------------------------------------------
# Slice filters (pin a single subpop / age group / risk group)
# ---------------------------------------------------------------------------

def test_age_filter_selects_one_age_group(con_meta):
    """A pinned age group restricts which rows are summed, independently of
    the aggregation level -- the population total of age group 1 only."""
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, agg_level="population", age_filter=1, scenarios=["baseline"])
    sql, params = lib.build_timeseries_query(cfg, with_ci=False)
    df = con_meta.execute(sql, params).df()
    # Age group 1's baseline is _BASE[1] scaled per replicate; the median over
    # the two replicates is the mean of the two scaled values.
    expected = sorted(_BASE[1] * s for s in _REP_BASE_SCALE.values())
    assert df["median_value"].tolist() == pytest.approx(
        [sum(expected) / 2] * DAYS)


def test_filters_force_the_full_table(con_meta):
    """`results` has no age_group column, so pinning one must switch the query
    off the pre-aggregated fast path rather than producing invalid SQL."""
    dims = lib.discover_dims(con_meta)
    assert lib._source_table(_config(dims, agg_level="population")) == "results"
    plain = lib.build_timeseries_query(
        _config(dims, agg_level="population"), with_ci=False)[0]
    assert "FROM results\n" in plain
    pinned = lib.build_timeseries_query(
        _config(dims, agg_level="population", age_filter=0), with_ci=False)[0]
    assert "FROM results_full" in pinned


def test_subpop_and_risk_filters_apply(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, agg_level="population", subpop_filter="pop",
                  risk_filter=0, scenarios=["baseline"])
    df = con_meta.execute(*lib.build_timeseries_query(cfg, with_ci=False)).df()
    # The fixture has exactly one subpop and one risk group, so pinning both
    # must leave the population total unchanged -- while still going through
    # the filtered results_full path rather than the `results` shortcut.
    unpinned_cfg = _config(dims, agg_level="population", scenarios=["baseline"])
    unpinned = con_meta.execute(
        *lib.build_timeseries_query(unpinned_cfg, with_ci=False)).df()
    assert lib._source_table(cfg) == "results_full"
    assert df["median_value"].tolist() == pytest.approx(
        unpinned["median_value"].tolist())


def test_pinned_dimension_is_not_also_faceted(con_meta):
    """Faceting on a dimension restricted to one value draws a one-panel grid."""
    dims = lib.discover_dims(con_meta)
    assert lib._facet_col(_config(dims, agg_level="age_group")) == "age_group"
    assert lib._facet_col(
        _config(dims, agg_level="age_group", age_filter=0)) is None


def test_chart_title_names_the_pinned_slice(con_meta):
    dims = lib.discover_dims(con_meta)
    title = lib.chart_title(_config(dims, agg_level="population", age_filter=1), dims)
    # The band label from meta, not the bare index.
    assert "age 18+" in title


# ---------------------------------------------------------------------------
# Cumulative time series
# ---------------------------------------------------------------------------

def test_cumulative_is_the_running_total_of_the_daily_series(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"])
    daily = con_meta.execute(
        *lib.build_timeseries_query(cfg, with_ci=False)).df()
    cumulative = con_meta.execute(
        *lib.build_timeseries_query({**cfg, "cumulative": True}, with_ci=False)).df()
    assert cumulative["median_value"].tolist() == pytest.approx(
        daily["median_value"].cumsum().tolist())


def test_cumulative_accumulates_per_replicate_before_summarizing(con_meta):
    """Accumulating after taking the median would be a different (wrong)
    quantity once replicates differ; check the interval reflects per-replicate
    cumulative curves."""
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"], cumulative=True)
    df = con_meta.execute(*lib.build_timeseries_query(cfg, with_ci=True)).df()
    last = df.iloc[-1]
    # Per replicate the cumulative total is 30 x scale x DAYS days. The band is
    # quantile_cont over those two replicate totals, which interpolates between
    # them rather than returning the endpoints.
    lo, hi = sorted(sum(_BASE.values()) * s * DAYS for s in _REP_BASE_SCALE.values())
    assert last["median_value"] == pytest.approx((lo + hi) / 2)
    assert last["p_lo"] == pytest.approx(lo + 0.025 * (hi - lo))
    assert last["p_hi"] == pytest.approx(lo + 0.975 * (hi - lo))


def test_cumulative_starts_from_the_selected_day_range(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, scenarios=["baseline"], cumulative=True, day_range=(2, 3))
    df = con_meta.execute(*lib.build_timeseries_query(cfg, with_ci=False)).df()
    assert df["day"].tolist() == [2, 3]
    # Day 2 is the first day in range, so it is a total of one day only.
    assert df["median_value"].iloc[1] == pytest.approx(
        2 * df["median_value"].iloc[0])


def test_cumulative_rejected_for_compartments(tmp_path):
    """A compartment is a level, not a flow, so summing it over days is
    meaningless -- say so rather than plotting nonsense."""
    path = _build_db(tmp_path / "c.db", with_meta=True)
    con = sqlite3.connect(path)
    con.execute("UPDATE meta SET value = ? WHERE key = 'compartments'",
                (json.dumps(["H"]),))
    con.commit()
    con.close()
    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        cfg = _config(dims, metrics=["H"], cumulative=True)
        with pytest.raises(lib.ResultsExplorerError, match="only meaningful for transition"):
            lib.render_chart(dcon, dims, cfg)
    finally:
        dcon.close()


def test_cumulative_label_marks_the_axis(con_meta):
    dims = lib.discover_dims(con_meta)
    cfg = _config(dims, cumulative=True, scenarios=["baseline"])
    assert "cumulative new_H" in lib.chart_title(cfg, dims)


# ---------------------------------------------------------------------------
# Empty groups and per-panel scales
# ---------------------------------------------------------------------------

def _db_with_an_empty_age_group(tmp_path):
    """Same shape as the main fixture, but age group 1 is flat zero."""
    path = tmp_path / "empty_group.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE results (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, day INTEGER, value REAL)")
    con.execute(
        "CREATE TABLE results_full (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, subpop TEXT, age_group INTEGER,"
        " risk_group INTEGER, day INTEGER, value REAL)")
    full, totals = [], {}
    for age in (0, 1, 2):
        for day in range(1, DAYS + 1):
            v = 0.0 if age == 1 else float(age + 1)
            full.append(("s1", 0, None, "new_H", "transition", "pop", age, 0, day, v))
            totals[day] = totals.get(day, 0.0) + v
    con.executemany("INSERT INTO results_full VALUES (?,?,?,?,?,?,?,?,?,?)", full)
    con.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                    [("s1", 0, None, "new_H", "transition", d, v)
                     for d, v in totals.items()])
    con.commit()
    con.close()
    return path


def test_all_zero_groups_are_dropped(tmp_path):
    dcon = lib.load_source(_db_with_an_empty_age_group(tmp_path))
    try:
        dims = lib.discover_dims(dcon)
        cfg = lib.default_chart_config(0, "timeseries", dims)
        cfg.update(metrics=["new_H"], scenarios=["s1"], agg_level="age_group")
        sql, params = lib.build_timeseries_query(cfg, with_ci=False)
        raw = dcon.execute(sql, params).df()
        assert sorted(raw["age_group"].unique()) == [0, 1, 2]
        kept = lib._drop_empty_groups(raw, cfg)
        assert sorted(kept["age_group"].unique()) == [0, 2]
    finally:
        dcon.close()


def test_empty_groups_are_kept_when_the_option_is_off(tmp_path):
    dcon = lib.load_source(_db_with_an_empty_age_group(tmp_path))
    try:
        dims = lib.discover_dims(dcon)
        cfg = lib.default_chart_config(0, "timeseries", dims)
        cfg.update(metrics=["new_H"], scenarios=["s1"], agg_level="age_group",
                   hide_empty=False)
        raw = dcon.execute(*lib.build_timeseries_query(cfg, with_ci=False)).df()
        assert sorted(lib._drop_empty_groups(raw, cfg)["age_group"].unique()) == [0, 1, 2]
    finally:
        dcon.close()


def test_dropping_never_empties_the_frame(tmp_path):
    """If every group is zero there is nothing to prefer, so keep the data and
    let the caller say 'no data' rather than rendering a blank chart."""
    path = tmp_path / "all_zero.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE results (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, day INTEGER, value REAL)")
    con.execute(
        "CREATE TABLE results_full (scenario TEXT, rep INTEGER, param_set INTEGER,"
        " compartment TEXT, kind TEXT, subpop TEXT, age_group INTEGER,"
        " risk_group INTEGER, day INTEGER, value REAL)")
    con.executemany("INSERT INTO results_full VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [("s1", 0, None, "new_H", "transition", "pop", a, 0, d, 0.0)
                     for a in (0, 1) for d in range(1, DAYS + 1)])
    con.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?)",
                    [("s1", 0, None, "new_H", "transition", d, 0.0)
                     for d in range(1, DAYS + 1)])
    con.commit()
    con.close()
    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        cfg = lib.default_chart_config(0, "timeseries", dims)
        cfg.update(metrics=["new_H"], scenarios=["s1"], agg_level="age_group")
        raw = dcon.execute(*lib.build_timeseries_query(cfg, with_ci=False)).df()
        assert len(lib._drop_empty_groups(raw, cfg)) == len(raw)
    finally:
        dcon.close()


def test_shared_y_axis_can_be_released_per_panel(con_meta):
    """One dominant group otherwise flattens every other panel onto the axis."""
    dims = lib.discover_dims(con_meta)
    shared = json.dumps(lib.render_chart(
        con_meta, dims, _config(dims, agg_level="age_group")).to_dict())
    independent = json.dumps(lib.render_chart(
        con_meta, dims,
        _config(dims, agg_level="age_group", shared_y=False)).to_dict())
    assert '"y": "shared"' in shared
    assert '"y": "independent"' in independent


# ---------------------------------------------------------------------------
# Metric ordering
# ---------------------------------------------------------------------------

def test_metric_order_follows_meta_not_the_alphabet(tmp_path):
    """The meta table records the model's own compartment/transition order so
    the explorer's metric list matches the Model Builder's. Sorting it here
    (or recovering it with SELECT DISTINCT) would scramble S -> E -> I -> H."""
    path = tmp_path / "ordered.db"
    _build_db(path, with_meta=True)
    con = sqlite3.connect(path)
    con.executemany("UPDATE meta SET value = ? WHERE key = ?", [
        (json.dumps(["S", "E", "I", "H", "R"]), "compartments"),
        (json.dumps(["S_to_E", "E_to_I", "I_to_H"]), "transition_vars"),
    ])
    con.commit()
    con.close()
    dcon = lib.load_source(path)
    try:
        dims = lib.discover_dims(dcon)
        assert [m for m, _ in dims["metrics"]] == [
            "S", "E", "I", "H", "R", "S_to_E", "E_to_I", "I_to_H"]
    finally:
        dcon.close()
