"""
model_builder_notebook.py
=========================

** GENERATED FILE — DO NOT EDIT DIRECTLY **

This file is assembled from section files by build_notebook.py.
Edit the relevant section file instead, then rebuild:

    python generic_core/examples/build_notebook.py

Section files (all in generic_core/examples/):
  _nb_shared.py               — imports and helper functions
  _nb_analysis_metric_defs.py — analysis metric definition widgets
  _nb_entry.py                — tab selector, output directory, autosave
  _nb_population.py           — Population & Geography tab
  _nb_model_builder.py        — Model Builder tab (Steps 0–9)
  _nb_shared_factory.py       — shared model factory functions
  _nb_fitting.py              — Fitting tab
  _nb_forecast.py             — Forecast tab
  _nb_export.py               — Export tab
  _nb_analysis.py             — Analysis tab
  _nb_docs.py                 — Documentation tab

If you edited cells in the marimo browser UI, sync changes back to the
section files first:

    python generic_core/examples/split_notebook.py

Interactive marimo notebook for building, visualising, and running
config-driven epidemic models.

Run with::

    marimo run generic_core/examples/model_builder_notebook.py

Supported rate templates
------------------------
- ``constant_param``
- ``param_product``
- ``immunity_modulated``
- ``force_of_infection``
- ``force_of_infection_travel``

Scope note
----------
This notebook supports single-population and metapopulation models, with
configurable age and risk groups. Age groups can be a plain count or named
bands (e.g. 0-4, 5-17, 65+). For multi-age/risk-group models, contact matrices
are embedded inline in the config JSON; they can be entered as CSVs or, when
named age bands are used, fetched for a US state or country in the
Population & Geography tab (requires the optional ``epydemix`` package:
``pip install epydemix``). Vaccines and mobility can be supplied as CSV files
or as constant scalar values.

Metapopulation folder conventions
----------------------------------
Required files:
  metapop_config.json          keys: subpopulations (ordered list of names), travel_matrix (NxN list of lists)

Optional shared files (all subpops):
  absolute_humidity.csv        cols: date, absolute_humidity
  mobility_modifier.csv        cols: day_of_week, mobility_modifier (JSON A×R array)

Optional per-subpop files ({name} = subpop name):
  school_work_calendar_{name}.csv   cols: date, is_school_day, is_work_day
  vaccines_{name}.csv               cols: date, daily_vaccines (JSON A×R array)
  initial_conditions_{name}.json    keys: compartments {name: A×R list}, epi_metrics {name: A×R list}
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

@app.cell
def _imports():
    import sys
    import json
    import io
    import copy
    import sqlite3
    import datetime
    from pathlib import Path
    from types import SimpleNamespace
    from collections import namedtuple

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    import clt_toolkit as clt
    import flu_core as flu

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import generic_core as gc
    from generic_core import contact_matrix_fetch as cmf
    from generic_core.config_parser import parse_model_config_from_dict
    from generic_core.generic_model import (
        ConfigDrivenSubpopModel,
        build_state_from_config,
        build_params_from_config,
    )
    from generic_core.generic_metapop import ConfigDrivenMetapopModel

    try:
        import torch
        import torch.nn.functional as _F
    except ImportError:
        torch = None
        _F = None

    try:
        from generic_core.torch_generic import (
            build_generic_torch_inputs,
            generic_torch_simulate_calibration_target,
        )
        from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY
    except ImportError:
        build_generic_torch_inputs = None
        generic_torch_simulate_calibration_target = None
        RATE_TEMPLATE_REGISTRY = None

    from generic_core.outcomes import (
        daily_transition_sum,
        compartment_timeseries,
        attack_rate as _generic_attack_rate,
        summarize_outcomes as _generic_summarize_outcomes,
    )
    from generic_core.calibration import compute_rsquared

    FitResult = namedtuple("FitResult", ["best_params", "loss_curve", "num_days", "observed", "method", "accepted_params", "sim_trajectories", "fit_targets", "target_labels", "target_weights", "target_modes", "r2_threshold", "n_ar_accepted"], defaults=[None, None])

    return (
        Path, SimpleNamespace, namedtuple, copy, sqlite3, datetime,
        clt, flu, gc, cmf, io, json, mo, np, pd, plt,
        ConfigDrivenMetapopModel, ConfigDrivenSubpopModel,
        build_state_from_config, build_params_from_config,
        parse_model_config_from_dict,
        torch, _F,
        build_generic_torch_inputs,
        generic_torch_simulate_calibration_target,
        RATE_TEMPLATE_REGISTRY,
        daily_transition_sum, compartment_timeseries,
        _generic_attack_rate, _generic_summarize_outcomes,
        compute_rsquared,
        FitResult,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@app.cell
def _helpers(Path, SimpleNamespace, json, np, pd):
    def parse_csv_list(text: str) -> list[str]:
        """Split a comma-separated string into a list of trimmed, non-empty items."""
        return [_item.strip() for _item in text.split(",") if _item.strip()]

    def rel_inf_param_name(compartment: str) -> str:
        """Auto-generated parameter name for a compartment's relative infectiousness."""
        return f"{compartment}_relative_infectiousness"

    def build_scalar_array(value, num_age_groups: int = 1, num_risk_groups: int = 1) -> "np.ndarray":
        """Return an (A×R) array filled with ``value``."""
        return np.full((num_age_groups, num_risk_groups), float(value), dtype=float)

    def build_notebook_schedules_input(
        start_date,
        num_days: int,
        absolute_humidity: float,
        mobility_value: float,
        daily_vaccines_value,  # float, or an A×R nested list to vary by age/risk group
        num_age_groups: int = 1,
        num_risk_groups: int = 1,
        absolute_humidity_df=None,
        school_work_calendar_df=None,
        mobility_df=None,
        daily_vaccines_df=None,
    ) -> "SimpleNamespace":
        """Assemble a schedules-input namespace for the model from the per-schedule
        DataFrames, falling back to constant-valued DataFrames where a df is None."""
        _horizon = max(int(num_days) + 14, 370)
        _dates = pd.date_range(start=start_date, periods=_horizon, freq="D").date

        _ah_df = absolute_humidity_df
        if _ah_df is None:
            _ah_df = pd.DataFrame({
                "date": _dates,
                "absolute_humidity": [float(absolute_humidity)] * _horizon,
            })

        _cal_df = school_work_calendar_df
        if _cal_df is None:
            _cal_df = pd.DataFrame({
                "date": _dates,
                "is_school_day": [1.0] * _horizon,
                "is_work_day": [1.0] * _horizon,
            })

        _mob_payload = json.dumps(
            np.full((num_age_groups, num_risk_groups), float(mobility_value)).tolist()
        )
        _mob_df = mobility_df
        if _mob_df is None:
            _mob_df = pd.DataFrame({
                "day_of_week": [
                    "monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday",
                ],
                "mobility_modifier": [_mob_payload] * 7,
            })

        if isinstance(daily_vaccines_value, (list, tuple)):
            _vax_arr = np.asarray(daily_vaccines_value, dtype=float)
        else:
            _vax_arr = np.full((num_age_groups, num_risk_groups), float(daily_vaccines_value))
        _vax_payload = json.dumps(_vax_arr.tolist())
        _vax_df = daily_vaccines_df
        if _vax_df is None:
            _vax_df = pd.DataFrame({
                "date": _dates,
                "daily_vaccines": [_vax_payload] * _horizon,
            })

        return SimpleNamespace(
            absolute_humidity_df=_ah_df,
            school_work_calendar_df=_cal_df,
            mobility_df=_mob_df,
            daily_vaccines_df=_vax_df,
        )

    def load_csv_validated(path_str: str, required_columns) -> tuple:
        """Load a CSV from path_str and validate column names.
        Returns (df, error_str) — error_str is None on success."""
        if not path_str or not path_str.strip():
            return None, "No path provided"
        _p = Path(path_str.strip())
        if not _p.exists():
            return None, f"File not found: {_p}"
        if not _p.is_file():
            return None, f"Not a file: {_p}"
        try:
            _df = pd.read_csv(_p)
            _df = _df.loc[:, ~_df.columns.str.match(r"^Unnamed")]
            _missing = set(required_columns) - set(_df.columns)
            if _missing:
                return None, f"Missing columns: {_missing}. Found: {list(_df.columns)}"
            return _df, None
        except Exception as _exc:
            return None, f"CSV read error: {_exc}"

    def load_contact_matrix_csv(path_str: str, expected_size: int) -> tuple:
        """Load an A×A contact matrix CSV (plain floats).
        Returns (nested_list, error_str)."""
        if not path_str or not path_str.strip():
            return None, "No path provided"
        _p = Path(path_str.strip())
        if not _p.exists():
            return None, f"File not found: {_p}"
        try:
            _mat = pd.read_csv(_p, header=None).values.astype(float)
            if _mat.shape != (expected_size, expected_size):
                return None, f"Expected {expected_size}×{expected_size}, got {_mat.shape}"
            return _mat.tolist(), None
        except Exception as _exc:
            return None, f"Matrix CSV error: {_exc}"

    def load_config_json(path_str: str) -> tuple:
        """Load a config JSON from path_str. Returns ({}, None) on empty path."""
        if not path_str or not path_str.strip():
            return {}, None
        _p = Path(path_str.strip())
        if not _p.exists():
            return {}, f"File not found: {_p}"
        try:
            with open(_p) as _f:
                return json.load(_f), None
        except Exception as _exc:
            return {}, f"JSON parse error: {_exc}"

    def resolve_input_path(folder_str: str, name_str: str) -> str:
        """Resolve a CSV entry against the shared input folder.

        ``name_str`` is normally a bare filename living in ``folder_str``.
        An absolute path in ``name_str`` overrides the folder (pathlib join
        semantics), so configs that still store full paths keep working.
        Returns "" when ``name_str`` is empty."""
        if not name_str or not name_str.strip():
            return ""
        if not folder_str or not folder_str.strip():
            return name_str.strip()
        return str(Path(folder_str.strip()) / name_str.strip())

    def validate_metapop_folder(folder_path_str: str) -> tuple:
        """Check that a metapop folder has the required files and a coherent
        metapop_config.json (travel matrix shape / row sums, per-subpop files).
        Returns (is_valid, status_dict)."""
        if not folder_path_str or not folder_path_str.strip():
            return False, {}
        _folder = Path(folder_path_str.strip())
        if not _folder.exists() or not _folder.is_dir():
            return False, {"folder": f"Not found or not a directory: {_folder}"}
        _required = ["metapop_config.json"]
        _optional_shared = [
            "absolute_humidity.csv",
            "mobility_modifier.csv",
        ]
        _status = {}
        _valid = True
        for _fname in _required:
            if (_folder / _fname).exists():
                _status[_fname] = "OK (required)"
            else:
                _status[_fname] = "MISSING (required)"
                _valid = False
        for _fname in _optional_shared:
            if (_folder / _fname).exists():
                _status[_fname] = "OK (optional shared)"
            else:
                _status[_fname] = "absent (will use constant value)"

        # Deeper validation of metapop_config.json contents.
        _cfg_path = _folder / "metapop_config.json"
        if _cfg_path.exists():
            try:
                with open(_cfg_path) as _f:
                    _cfg = json.load(_f)
            except Exception as _exc:
                _status["metapop_config.json"] = f"INVALID JSON: {_exc}"
                return False, _status

            _subpops = _cfg.get("subpopulations")
            _travel = _cfg.get("travel_matrix")
            if not isinstance(_subpops, list) or not _subpops:
                _status["subpopulations"] = "MISSING or empty (expected non-empty list)"
                _valid = False
            else:
                _n = len(_subpops)
                _status["subpopulations"] = f"OK ({_n}: {', '.join(map(str, _subpops))})"

                # Travel matrix must be N×N with rows summing to ~1.
                _tm = np.asarray(_travel, dtype=float) if _travel is not None else None
                if _tm is None or _tm.ndim != 2 or _tm.shape != (_n, _n):
                    _shape = None if _tm is None else _tm.shape
                    _status["travel_matrix"] = (
                        f"INVALID: expected {_n}×{_n}, got {_shape}"
                    )
                    _valid = False
                else:
                    _row_sums = _tm.sum(axis=1)
                    if not np.allclose(_row_sums, 1.0, atol=1e-6):
                        _status["travel_matrix"] = (
                            f"WARNING: rows should sum to 1; got {np.round(_row_sums, 4).tolist()}"
                        )
                    else:
                        _status["travel_matrix"] = f"OK ({_n}×{_n}, rows sum to 1)"

                # Per-subpop files (informational; defaults used when absent).
                for _name in _subpops:
                    for _suffix, _kind in (
                        (f"initial_conditions_{_name}.json", "initial conditions"),
                        (f"school_work_calendar_{_name}.csv", "calendar"),
                        (f"vaccines_{_name}.csv", "vaccines"),
                    ):
                        if (_folder / _suffix).exists():
                            _status[_suffix] = f"OK ({_kind})"
                        else:
                            _status[_suffix] = f"absent ({_kind}; default used)"

        return _valid, _status

    def infectious_mapping_to_str(mapping: dict) -> str:
        """Convert {comp: rel_param | None} back to the text format 'IP:ip_rel, IA:ia_rel, ISR'."""
        _parts = []
        for _k, _v in mapping.items():
            _parts.append(f"{_k}:{_v}" if _v else _k)
        return ", ".join(_parts)

    def is_array_param(cfg: dict, name: str) -> bool:
        """Return True if the named param in cfg has a list (A×R array) value."""
        return isinstance(cfg.get("params", {}).get(name), list)

    def param_grid_columns(age_groups, num_age_groups: int) -> list:
        """Column labels for an age×risk param data_editor: named age bands
        when available (from the Population & Geography tab), else age0..ageN."""
        if age_groups and len(age_groups) == num_age_groups:
            return list(age_groups)
        return [f"age{_a}" for _a in range(num_age_groups)]

    def grid_to_AR_array(grid_value, age_cols, num_age_groups, num_risk_groups):
        """Transpose a risk-row / age-column data_editor value into an A×R array.

        ``grid_value`` is the list-of-row-dicts produced by ``mo.ui.data_editor``
        (one row per risk group, one column per age band). Mirrors the param-grid
        readback in ``_build_config``."""
        _A = int(num_age_groups)
        _R = int(num_risk_groups)
        _rows = list(grid_value)
        return np.array(
            [[float(_rows[_r][age_cols[_a]]) for _r in range(_R)] for _a in range(_A)],
            dtype=float,
        )

    def array_to_grid_rows(arr, age_cols, num_risk_groups):
        """Inverse of grid_to_AR_array: build data_editor rows from an A×R array
        (or None, which becomes all-zero rows)."""
        _R = int(num_risk_groups)
        _A = len(age_cols)
        _arr = np.zeros((_A, _R)) if arr is None else np.asarray(arr, dtype=float)
        return [
            {"risk_group": f"risk{_r}", **{age_cols[_a]: float(_arr[_a][_r]) for _a in range(_A)}}
            for _r in range(_R)
        ]

    def default_seed_row_data(saved_ic, subpop, comp, age_cols, num_risk_groups,
                               is_first_seed_comp):
        """Build initial data_editor rows for one (subpop, compartment) seed grid.

        Pulls from ``saved_ic`` (a loaded config's ``initial_conditions`` dict)
        when present, else defaults to 50 in the first seed compartment's
        (age0, risk0) cell so a freshly built model still produces an epidemic
        out of the box."""
        _R = int(num_risk_groups)
        _seeds = (saved_ic.get(subpop, {}) or {}).get("seeds", {}) or {}
        _arr = _seeds.get(comp)
        _rows = []
        for _r in range(_R):
            _row = {"risk_group": _r}
            for _a, _col in enumerate(age_cols):
                _val = 0.0
                if isinstance(_arr, list):
                    try:
                        _val = float(_arr[_a][_r])
                    except (IndexError, TypeError, ValueError):
                        _val = 0.0
                elif is_first_seed_comp and _a == 0 and _r == 0:
                    _val = 50.0
                _row[_col] = _val
            _rows.append(_row)
        return _rows

    def load_population_csv(path_str, subpop_names, num_age_groups,
                            num_risk_groups, age_groups=None):
        """Parse a population CSV into per-subpop A×R arrays.

        Expected columns: ``age``, ``risk``, ``subpopulation``, ``population``.
        - ``age`` may be a named band (matching ``age_groups``) or a 0-based index.
        - ``risk`` is a 0-based index in ``0..R-1``. Optional when there is only
          one risk group, in which case every row is assumed to be risk 0.
        - ``subpopulation`` must be one of ``subpop_names``. Optional when there
          is only one subpopulation, in which case every row is assigned to it.
        Returns ``(pop_by_subpop, error_str)`` where ``pop_by_subpop`` maps each
        subpop name to an A×R numpy array; ``error_str`` is None on success."""
        if not path_str or not path_str.strip():
            return None, "No path provided"
        _p = Path(path_str.strip())
        if not _p.exists():
            return None, f"File not found: {_p}"
        try:
            _df = pd.read_csv(_p)
        except Exception as _exc:
            return None, f"CSV read error: {_exc}"
        _df = _df.loc[:, ~_df.columns.str.match(r"^Unnamed")]
        _required = {"age", "population"}
        _missing = _required - set(_df.columns)
        if _missing:
            return None, f"Missing columns: {_missing}. Found: {list(_df.columns)}"
        if "risk" not in _df.columns:
            if int(num_risk_groups) != 1:
                return None, (
                    "Missing column: {'risk'} (required when there is more than "
                    "one risk group)."
                )
            _df = _df.assign(risk=0)
        if "subpopulation" not in _df.columns:
            if len(subpop_names) != 1:
                return None, (
                    "Missing column: {'subpopulation'} (required when there is "
                    "more than one subpopulation)."
                )
            _df = _df.assign(subpopulation=subpop_names[0])

        _A = int(num_age_groups)
        _R = int(num_risk_groups)
        # Map an age cell (named band or index string) to a 0-based age index.
        _band_to_idx = {}
        if age_groups and len(age_groups) == _A:
            _band_to_idx = {str(_b): _i for _i, _b in enumerate(age_groups)}

        def _age_index(_val):
            _s = str(_val).strip()
            if _s in _band_to_idx:
                return _band_to_idx[_s]
            try:
                return int(float(_s))
            except ValueError:
                return None

        _pop = {_name: np.zeros((_A, _R), dtype=float) for _name in subpop_names}
        for _row_i, _row in _df.iterrows():
            _sp = str(_row["subpopulation"]).strip()
            if _sp not in _pop:
                return None, (f"Row {_row_i}: unknown subpopulation '{_sp}'. "
                              f"Expected one of {list(subpop_names)}.")
            _ai = _age_index(_row["age"])
            if _ai is None or not (0 <= _ai < _A):
                return None, (f"Row {_row_i}: age '{_row['age']}' is not a valid "
                              f"band/index for A={_A}.")
            try:
                _ri = int(float(_row["risk"]))
            except (ValueError, TypeError):
                return None, f"Row {_row_i}: risk '{_row['risk']}' is not an integer."
            if not (0 <= _ri < _R):
                return None, f"Row {_row_i}: risk {_ri} out of range 0..{_R - 1}."
            try:
                _pop[_sp][_ai, _ri] = float(_row["population"])
            except (ValueError, TypeError):
                return None, f"Row {_row_i}: population '{_row['population']}' is not numeric."
        return _pop, None

    def build_compartment_init(seed_arrays, population_AR, compartments):
        """Build a {compartment: A×R array} init dict from seed counts + population.

        ``seed_arrays`` maps non-first compartment names to A×R arrays of seeded
        counts. The first compartment receives ``population − Σ seeds`` per cell
        (clamped at 0). Returns ``(comp_init, overflow)`` where ``overflow`` is
        True if any cell's seeds exceeded its population."""
        _pop = np.asarray(population_AR, dtype=float)
        _seed_total = np.zeros_like(_pop)
        _comp_init = {}
        for _c in compartments[1:] if len(compartments) > 1 else []:
            _arr = np.asarray(seed_arrays.get(_c, np.zeros_like(_pop)), dtype=float)
            _comp_init[_c] = _arr
            _seed_total = _seed_total + _arr
        _remainder = _pop - _seed_total
        _overflow = bool(np.any(_remainder < 0))
        if compartments:
            _comp_init[compartments[0]] = np.clip(_remainder, 0.0, None)
        for _c in compartments:
            _comp_init.setdefault(_c, np.zeros_like(_pop))
        return _comp_init, _overflow

    def read_initial_conditions(config, subpop_name, compartments,
                                num_age_groups, num_risk_groups):
        """Read per-subpop initial conditions from ``config['initial_conditions']``.

        Returns a ``{compartment: A×R array}`` init dict (first compartment =
        population − Σ seeds), or None when the subpop has no entry. Used by the
        metapop and shared-factory run paths, with the folder JSON as fallback."""
        _ic = (config or {}).get("initial_conditions", {})
        _entry = _ic.get(subpop_name)
        if not _entry:
            return None
        _A = int(num_age_groups)
        _R = int(num_risk_groups)
        _pop = np.asarray(_entry.get("population",
                                     np.zeros((_A, _R))), dtype=float)
        _seeds = {}
        for _c, _arr in (_entry.get("seeds", {}) or {}).items():
            if _c in compartments:
                _seeds[_c] = np.asarray(_arr, dtype=float)
        _comp_init, _ = build_compartment_init(_seeds, _pop, compartments)
        return _comp_init

    return (
        build_notebook_schedules_input,
        build_scalar_array,
        parse_csv_list,
        rel_inf_param_name,
        load_csv_validated,
        load_contact_matrix_csv,
        load_config_json,
        resolve_input_path,
        validate_metapop_folder,
        infectious_mapping_to_str,
        is_array_param,
        array_to_grid_rows,
        param_grid_columns,
        grid_to_AR_array,
        default_seed_row_data,
        load_population_csv,
        build_compartment_init,
        read_initial_conditions,
    )


# ---------------------------------------------------------------------------
# Shared visual style layer
# ---------------------------------------------------------------------------


@app.cell
def _clt_style_helpers(mo):
    import html as _html
    import random as _random

    # Per-tab accent colors — keep section badges/headers visually distinct so
    # the user always knows which part of the workflow they are in.
    CLT_ACCENT = {
        "population": "#2e7d6b",  # teal
        "builder":    "#3b6ea5",  # blue
        "fitting":    "#9c5fb5",  # purple
        "forecast":   "#c2792e",  # amber
        "export":     "#557a46",  # green
        "analysis":   "#b5495b",  # rose
    }

    def tip_label(label_text="", tip_text="", *, html_tip=False, width=None):
        """Inline ⓘ hover tooltip, optionally preceded by ``label_text``.

        ``html_tip=False`` (default): ``tip_text`` is plain text, HTML-escaped,
        shown with line breaks preserved (pre-wrap). ``html_tip=True``:
        ``tip_text`` is treated as raw HTML (use ``<br>`` for breaks).

        The ``<style>`` is emitted inline (scoped to a unique id) rather than
        relying on a global stylesheet, so the tooltip still hides correctly
        when it renders inside a shadow-DOM component such as ``mo.accordion``
        or ``mo.ui.tabs`` (a global stylesheet would not reach inside those)."""
        if html_tip:
            _body = tip_text.replace("\n", "<br>")
            _ws = "normal"
            _w = 520 if width is None else width
        else:
            _body = _html.escape(tip_text)
            _ws = "pre-wrap"
            _w = 300 if width is None else width
        _uid = _random.randint(10**7, 10**8 - 1)
        _lead = f"{label_text}&nbsp;" if label_text else ""
        return mo.Html(
            f"<style>"
            f"#tip{_uid}{{position:relative;display:inline-block;cursor:help;"
            f"color:#888;font-size:0.8em;vertical-align:middle;}}"
            f"#tip{_uid}>span{{visibility:hidden;opacity:0;transition:opacity .15s;"
            f"transition-delay:.2s;position:absolute;bottom:120%;left:0;"
            f"display:block;box-sizing:border-box;"
            f"background:#222;color:#fff;border-radius:4px;padding:6px 10px;"
            f"width:{_w}px;font-size:12px;line-height:1.5;white-space:{_ws};"
            f"pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            # Radix's accordion content panel keeps a permanent
            # `overflow-hidden` class (for its open/close animation), which
            # clips this absolutely-positioned popup when a tooltip lives
            # inside an accordion item. Lifting overflow on the open panel
            # only takes effect once it's fully open, so it doesn't disturb
            # the collapse/expand animation itself.
            f'div[data-state="open"].overflow-hidden{{overflow:visible;}}'
            f"</style>"
            f"<span>{_lead}"
            f'<span id="tip{_uid}">ⓘ<span>{_body}</span></span>'
            f"</span>"
        )

    def with_tip(label_text, tip_text, widget, **kw):
        """Label + ⓘ tooltip on the left, widget on the right."""
        return mo.hstack(
            [tip_label(label_text, tip_text, **kw), widget],
            justify="start", align="center",
        )

    def wtip(widget, tip_text, **kw):
        """Widget on the left, ⓘ tooltip on the right. The widget sits in a
        fit-content box so radio/checkbox widgets don't stretch and shove the
        icon far to the right."""
        return mo.Html(
            '<div style="display:inline-flex;align-items:center;gap:4px;">'
            f'<div style="width:fit-content;">{widget}</div>'
            f'{tip_label("", tip_text, **kw)}'
            "</div>"
        )

    def step_header(n, title, subtitle=None, accent=None):
        """Bold numbered step/section header: colored badge + accented title.

        Styles are inline (no shared stylesheet) so the header renders
        correctly even inside shadow-DOM components."""
        _acc = accent or "#3b6ea5"
        _sub = (
            f'<div style="color:#777;font-size:.82rem;margin:.05rem 0 0 2.25rem;">'
            f"{_html.escape(str(subtitle))}</div>"
            if subtitle else ""
        )
        return mo.Html(
            '<div style="display:flex;align-items:center;gap:.55rem;'
            'margin:.1rem 0 .15rem;">'
            '<span style="display:inline-flex;align-items:center;'
            "justify-content:center;min-width:1.7em;height:1.7em;padding:0 .45em;"
            f"border-radius:999px;background:{_acc};color:#fff;font-weight:700;"
            f'font-size:.95rem;line-height:1;flex:none;">{_html.escape(str(n))}</span>'
            f'<span style="font-size:1.06rem;font-weight:700;color:{_acc};">'
            f"{_html.escape(str(title))}</span>"
            f"</div>{_sub}"
        )

    def section_card(header, body, accent=None):
        """Wrap ``header`` + ``body`` in a bordered card with a colored accent
        stripe down the left edge."""
        _stripe = accent or "#9aa7b8"
        return mo.vstack([header, body], gap=0.5).style({
            "border": "1px solid rgba(127,127,127,0.25)",
            "border-left": f"4px solid {_stripe}",
            "border-radius": "10px",
            "padding": "0.8rem 1rem",
            "margin": "0.45rem 0",
            "background": "rgba(127,127,127,0.03)",
        })

    return CLT_ACCENT, tip_label, with_tip, wtip, step_header, section_card

@app.cell
def _analysis_metric_defs_ui(mo, loaded_config, n_transitions, t_name, transition_vars_input):
    _MAX_MET = 5
    _saved = loaded_config.get("analysis_metrics", [])
    analysis_n_metrics_input = mo.ui.number(
        start=1, stop=_MAX_MET, step=1,
        value=min(max(len(_saved), 1), _MAX_MET),
        label="Number of user-defined metrics",
    )
    analysis_metric_names = mo.ui.array([
        mo.ui.text(
            value=_saved[i]["name"] if i < len(_saved) else f"metric_{i + 1}",
            label="Name",
        )
        for i in range(_MAX_MET)
    ])
    _tvs_explicit = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    tv_opts = _tvs_explicit if _tvs_explicit else [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    analysis_metric_tvs = mo.ui.array([
        mo.ui.multiselect(
            options=tv_opts if tv_opts else [""],
            value=[v for v in (_saved[i].get("transition_variables", []) if i < len(_saved) else []) if v in tv_opts],
            label="Transition variables to sum",
        )
        for i in range(_MAX_MET)
    ])
    return analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs, tv_opts


@app.cell
def _analysis_metric_sel_state(mo):
    get_sel_metrics, set_sel_metrics = mo.state([])
    return get_sel_metrics, set_sel_metrics


@app.cell
def _analysis_metric_plot_controls(mo, analysis_n_metrics_input, analysis_metric_names, get_sel_metrics, set_sel_metrics):
    _n = int(analysis_n_metrics_input.value)
    _opts = [analysis_metric_names.value[i].strip() or f"metric_{i + 1}" for i in range(_n)]
    _saved = [m for m in get_sel_metrics() if m in _opts]
    analysis_plot_metric_sel = mo.ui.multiselect(
        options=_opts if _opts else ["(no metrics defined)"],
        value=_saved if _saved else (_opts[:1] if _opts else []),
        on_change=set_sel_metrics,
        label="Metric(s) to show in plots",
    )
    return (analysis_plot_metric_sel,)

@app.cell
def _tab_style_injection(mo):
    # Must render in its own cell, before the tabs widget's cell, so this
    # stylesheet already exists in the document when marimo-tabs constructs
    # its shadow root and copies stylesheets into it (a one-time, synchronous
    # snapshot at construction — added later is too late).
    mo.Html(
        '<style title="marimo-tab-width">'
        '[role="tablist"] { width: 100%; }'
        '[role="tablist"] [role="tab"] { flex: 1; text-align: center; }'
        "</style>"
    )
    return


@app.cell
def _main_tab_selector(mo):
    main_tab = mo.ui.tabs({
        "Population & Geography": mo.md(""),
        "Model Builder": mo.md(""),
        "Analysis":      mo.md(""),
        "Fitting":       mo.md(""),
        "Forecast":      mo.md(""),
        "Export":        mo.md(""),
        "Documentation": mo.md(""),
    })
    return (main_tab,)


@app.cell
def _output_dir_ui(mo, Path):
    output_dir_input = mo.ui.text(
        value=str(Path.home() / "clt_outputs"),
        label="Output directory (auto-saves go here)",
        full_width=True,
    )
    return (output_dir_input,)


@app.cell
def _output_dir(output_dir_input, Path):
    output_dir = Path(output_dir_input.value).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _tab_header_display(main_tab, output_dir_input, mo):
    mo.vstack([
        main_tab,
        mo.hstack([output_dir_input], justify="start"),
    ])
    return


@app.cell
def _autosave_config(config_dict, output_dir, json):
    _p = output_dir / "model_config.json"
    _ = _p.write_text(json.dumps(config_dict, indent=2))
    return

@app.cell
def _population_structure_ui(mo, loaded_config):
    _ar = loaded_config.get("age_risk", {})
    _inf = loaded_config.get("input_files", {})
    _saved_bands = _ar.get("age_groups") or []

    age_group_mode_radio = mo.ui.radio(
        options=["Count only", "Named age bands"],
        value="Named age bands" if _saved_bands else "Count only",
        label="Age-group specification",
    )
    num_age_groups_input = mo.ui.number(
        start=1, stop=20, step=1,
        value=int(_ar.get("num_age_groups", 1)),
        label="Number of age groups (A)",
    )
    age_bands_input = mo.ui.text(
        value=", ".join(_saved_bands),
        placeholder="0-4, 5-17, 18-49, 50-64, 65+",
        label="Age bands (comma-separated, 0-based, contiguous, last 'x+')",
        full_width=True,
    )
    num_risk_groups_input = mo.ui.number(
        start=1, stop=10, step=1,
        value=int(_ar.get("num_risk_groups", 1)),
        label="Number of risk groups (R)",
    )
    _metapop_folder_saved = _inf.get("metapop_folder", "")
    pop_mode_radio = mo.ui.radio(
        options=["Single population", "Metapopulation"],
        value="Metapopulation" if _metapop_folder_saved else "Single population",
        label="Population mode",
    )
    metapop_folder_input = mo.ui.text(
        value=_metapop_folder_saved,
        placeholder="/path/to/metapop_folder/",
        label="Metapopulation folder path",
        full_width=True,
    )
    return (
        age_group_mode_radio,
        num_age_groups_input,
        age_bands_input,
        num_risk_groups_input,
        pop_mode_radio,
        metapop_folder_input,
    )


@app.cell
def _population_structure_compute(
    age_group_mode_radio,
    num_age_groups_input,
    age_bands_input,
    num_risk_groups_input,
    pop_mode_radio,
    cmf,
):
    # In band mode, A is the number of named bands; in count mode it's the number
    # input. age_groups is the band list (or None when no bands are defined).
    _use_bands = age_group_mode_radio.value == "Named age bands"
    if _use_bands:
        age_groups = cmf.parse_age_bands(age_bands_input.value)
        num_age_groups = max(len(age_groups), 1)
    else:
        age_groups = None
        num_age_groups = int(num_age_groups_input.value)

    num_risk_groups = int(num_risk_groups_input.value)
    is_metapop = pop_mode_radio.value == "Metapopulation"
    age_group_mode = age_group_mode_radio.value
    return (num_age_groups, num_risk_groups, is_metapop, age_groups, age_group_mode)


@app.cell
def _population_structure_show(
    mo,
    main_tab,
    num_age_groups,
    num_risk_groups,
    is_metapop,
    age_groups,
    age_group_mode,
    age_group_mode_radio,
    num_age_groups_input,
    age_bands_input,
    num_risk_groups_input,
    pop_mode_radio,
    metapop_folder_input,
    validate_metapop_folder,
    cmf,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Population & Geography", None)
    _ACC = CLT_ACCENT["population"]

    _parts = [
        mo.md(
            "Define the population dimensions and geography here. The rest of the "
            "model (compartments, transitions, parameters, …) is built in the "
            "**Model Builder** tab."
        ),
        age_group_mode_radio,
    ]

    if age_group_mode == "Named age bands":
        _parts.append(age_bands_input)
        try:
            cmf.validate_age_bands(age_groups or [])
            _parts.append(mo.callout(
                mo.md(f"Parsed **A = {num_age_groups}** age bands: "
                      + ", ".join(f"`{_b}`" for _b in (age_groups or []))
                      + ".\n\nNamed bands enable contact-matrix fetching below."),
                kind="success",
            ))
        except ValueError as _exc:
            _parts.append(mo.callout(mo.md(f"**Age bands:** {_exc}"), kind="danger"))
    else:
        _parts.append(num_age_groups_input)
        if num_age_groups > 1:
            _parts.append(mo.callout(
                mo.md(
                    "Count-only mode: contact matrices cannot be fetched (the "
                    "fetcher needs age-band definitions). For A > 1, either switch "
                    "to **Named age bands** to fetch them, or supply contact-matrix "
                    "CSVs in Model Builder → Step 4."
                ),
                kind="info",
            ))

    _parts.append(num_risk_groups_input)
    _parts.append(pop_mode_radio)

    if is_metapop:
        _parts.append(metapop_folder_input)
        _folder_valid, _folder_status = validate_metapop_folder(metapop_folder_input.value)
        if metapop_folder_input.value.strip():
            _lines = [f"- **{_fname}**: {_msg}" for _fname, _msg in _folder_status.items()]
            _overall_kind = "success" if _folder_valid else "danger"
            _parts.append(mo.callout(mo.md("\n".join(_lines)), kind=_overall_kind))
    else:
        if num_age_groups > 1 or num_risk_groups > 1:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model: A={num_age_groups}, R={num_risk_groups}. "
                    "Use CSV file paths in Model Builder → Step 4 for schedule data."
                ),
                kind="info",
            ))

    section_card(
        step_header("①", "Population Structure",
                    "Age groups, risk groups, and single-population vs. metapopulation.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Contact-matrix geography (fetch via epydemix)
# ---------------------------------------------------------------------------


@app.cell
def _geo_fetch_state(mo):
    # Holds the most recent fetch result:
    #   {"matrices": {scope_key: {param: A×A}}, "scope": "shared"|"per_subpop",
    #    "errors": {...}}
    # scope_key is "__shared__" for one geography, or a subpop name otherwise.
    get_fetched_matrices, set_fetched_matrices = mo.state({})
    return get_fetched_matrices, set_fetched_matrices


@app.cell
def _geo_subpop_names(is_metapop, metapop_folder_input, Path, json):
    # Subpop names for per-subpop geography, read from metapop_config.json.
    geo_subpop_names = []
    if is_metapop and metapop_folder_input.value.strip():
        _cfg_path = Path(metapop_folder_input.value.strip()) / "metapop_config.json"
        if _cfg_path.exists():
            try:
                with open(_cfg_path) as _f:
                    _mc = json.load(_f)
                _sp = _mc.get("subpopulations")
                if isinstance(_sp, list):
                    geo_subpop_names = [str(_s) for _s in _sp]
            except Exception:
                geo_subpop_names = []
    return (geo_subpop_names,)


@app.cell
def _geo_ui(mo, cmf, geo_subpop_names):
    geo_scope_radio = mo.ui.radio(
        options=["Same for all subpops", "Per-subpopulation"],
        value="Same for all subpops",
        label="Contact-matrix geography scope (metapop)",
    )
    geo_kind_radio = mo.ui.radio(
        options=["US state", "Country"], value="US state", label="Geography type",
    )
    geo_state_dropdown = mo.ui.dropdown(
        options=cmf.US_STATES, value="Massachusetts", label="US state", searchable=True,
    )
    geo_country_input = mo.ui.dropdown(
        options=cmf.COUNTRIES, value="United_Kingdom",
        label="Country (epydemix-data name)", searchable=True,
    )
    # Per-subpop selectors (used only in metapop + per-subpopulation scope).
    geo_subpop_kind = mo.ui.array([
        mo.ui.radio(options=["US state", "Country"], value="US state", label=f"{_n}: type")
        for _n in geo_subpop_names
    ])
    geo_subpop_state = mo.ui.array([
        mo.ui.dropdown(options=cmf.US_STATES, value="Massachusetts",
                       label=f"{_n}: US state", searchable=True)
        for _n in geo_subpop_names
    ])
    geo_subpop_country = mo.ui.array([
        mo.ui.dropdown(options=cmf.COUNTRIES, value="United_Kingdom",
                       label=f"{_n}: country", searchable=True)
        for _n in geo_subpop_names
    ])
    geo_fetch_button = mo.ui.run_button(label="Fetch contact matrices & population")
    return (
        geo_scope_radio, geo_kind_radio, geo_state_dropdown, geo_country_input,
        geo_subpop_kind, geo_subpop_state, geo_subpop_country, geo_fetch_button,
    )


@app.cell
def _pop_source_ui(mo, num_risk_groups, loaded_config):
    # Population-source widgets: either fetch per-age-band totals for the
    # geography (epydemix), or load a CSV of population per age/risk/subpop.
    population_source_radio = mo.ui.radio(
        options=["Fetch from geography", "CSV file"],
        value="Fetch from geography",
        label="Population source",
    )
    _saved_rf = (loaded_config.get("age_risk", {}) or {}).get("risk_group_fractions")
    if not isinstance(_saved_rf, list) or len(_saved_rf) != int(num_risk_groups):
        _saved_rf = [1.0 / max(int(num_risk_groups), 1)] * int(num_risk_groups)
    # One fraction per risk group; the fetched (age-only) population is split
    # across risk groups by these fractions (renormalised to sum to 1).
    risk_fraction_inputs = mo.ui.array([
        mo.ui.number(start=0.0, stop=1.0, step=None, value=float(_saved_rf[_r]),
                     label=f"risk {_r}")
        for _r in range(int(num_risk_groups))
    ])
    population_csv_input = mo.ui.text(
        value="",
        placeholder="/path/to/population.csv",
        label="Population CSV (columns: age, population, [risk], [subpopulation])",
        full_width=True,
    )
    return population_source_radio, risk_fraction_inputs, population_csv_input


@app.cell
def _geo_fetch(
    mo, cmf,
    geo_fetch_button,
    age_group_mode, age_groups, num_age_groups,
    is_metapop, geo_scope_radio,
    geo_kind_radio, geo_state_dropdown, geo_country_input,
    geo_subpop_names, geo_subpop_kind, geo_subpop_state, geo_subpop_country,
    population_source_radio,
    set_fetched_matrices,
):
    # Only fetch when the button is pressed and named bands are defined.
    mo.stop(not geo_fetch_button.value, None)

    def _kind_geo(kind_radio, state_dd, country_txt):
        if kind_radio.value == "US state":
            return "us_state", state_dd.value
        return "country", country_txt.value.strip()

    # With a single age group there are no named bands to define; '0+' covers
    # the whole population in one band, which is all fetch_* needs for A=1.
    if age_group_mode == "Named age bands":
        _eff_age_groups = age_groups
    elif num_age_groups == 1:
        _eff_age_groups = ["0+"]
    else:
        _eff_age_groups = None

    if not _eff_age_groups:
        set_fetched_matrices({
            "matrices": {}, "populations": {}, "scope": "shared",
            "errors": {"error": "Define named age bands before fetching contact matrices."},
        })
    else:
        _per_subpop = is_metapop and geo_scope_radio.value == "Per-subpopulation"
        _fetch_pop = population_source_radio.value == "Fetch from geography"
        _results, _pops, _errors = {}, {}, {}
        try:
            if _per_subpop:
                for _i, _name in enumerate(geo_subpop_names):
                    _kind, _geo = _kind_geo(
                        geo_subpop_kind[_i], geo_subpop_state[_i], geo_subpop_country[_i]
                    )
                    _results[_name] = cmf.fetch_contact_matrices(_kind, _geo, _eff_age_groups)
                    if _fetch_pop:
                        _pops[_name] = cmf.fetch_population(_kind, _geo, _eff_age_groups)
            else:
                _kind, _geo = _kind_geo(geo_kind_radio, geo_state_dropdown, geo_country_input)
                _results["__shared__"] = cmf.fetch_contact_matrices(_kind, _geo, _eff_age_groups)
                if _fetch_pop:
                    _pops["__shared__"] = cmf.fetch_population(_kind, _geo, _eff_age_groups)
        except Exception as _exc:
            _errors["error"] = str(_exc)

        set_fetched_matrices({
            "matrices": _results,
            "populations": _pops,
            "scope": "per_subpop" if _per_subpop else "shared",
            "errors": _errors,
        })
    return


@app.cell
def _geo_result(get_fetched_matrices):
    _state = get_fetched_matrices() or {}
    fetched_contact_matrices = _state.get("matrices", {})
    fetched_populations = _state.get("populations", {})
    fetched_matrices_scope = _state.get("scope", "shared")
    fetched_matrices_errors = _state.get("errors", {})
    return (
        fetched_contact_matrices, fetched_populations,
        fetched_matrices_scope, fetched_matrices_errors,
    )


@app.cell
def _geo_show(
    mo, main_tab, cmf,
    age_group_mode, num_age_groups, is_metapop,
    geo_scope_radio, geo_kind_radio, geo_state_dropdown, geo_country_input,
    geo_subpop_names, geo_subpop_kind, geo_subpop_state, geo_subpop_country,
    geo_fetch_button,
    fetched_contact_matrices, fetched_matrices_scope, fetched_matrices_errors,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Population & Geography", None)
    _ACC = CLT_ACCENT["population"]
    _header = step_header(
        "②", "Contact Matrices (geography)",
        "Optionally fetch age-structured contact matrices for a US state or country.",
        accent=_ACC,
    )

    if age_group_mode != "Named age bands" and num_age_groups != 1:
        mo.stop(True, section_card(
            _header,
            mo.callout(
                mo.md("Switch to **Named age bands** above to fetch contact matrices "
                      "for a geography. In count-only mode with A > 1, provide "
                      "contact-matrix CSVs in Model Builder → Step 4 instead."),
                kind="info",
            ),
            accent=_ACC,
        ))

    _parts = []
    if not cmf.epydemix_available():
        _parts.append(mo.callout(
            mo.md("The optional **epydemix** package is not installed, so live fetching "
                  "is unavailable. Install it with `pip install epydemix`, or supply "
                  "contact-matrix CSVs in Model Builder → Step 4."),
            kind="warn",
        ))

    _parts.append(mo.md(
        f"Fetch the **total / school / work** {num_age_groups}×{num_age_groups} contact "
        "matrices (Mistry 2021, via epydemix-data) for your age bands."
    ))

    # Fetch controls collapse into an accordion to keep the tab tidy.
    _ctrl = []
    if is_metapop:
        _ctrl.append(geo_scope_radio)
    if is_metapop and geo_scope_radio.value == "Per-subpopulation":
        if not geo_subpop_names:
            _ctrl.append(mo.callout(
                mo.md("No subpopulations found — set a valid metapop folder above."),
                kind="warn",
            ))
        for _i, _name in enumerate(geo_subpop_names):
            _sel = (geo_subpop_state[_i] if geo_subpop_kind[_i].value == "US state"
                    else geo_subpop_country[_i])
            _ctrl.append(mo.hstack([mo.md(f"**{_name}**"), geo_subpop_kind[_i], _sel],
                                   justify="start"))
    else:
        _ctrl.append(geo_kind_radio)
        _ctrl.append(geo_state_dropdown if geo_kind_radio.value == "US state"
                     else geo_country_input)
    _ctrl.append(geo_fetch_button)
    _parts.append(mo.accordion(
        {"Fetch contact matrices for a geography": mo.vstack(_ctrl)},
    ))

    if fetched_matrices_errors.get("error"):
        _parts.append(mo.callout(mo.md(f"**Fetch failed:** {fetched_matrices_errors['error']}"),
                                 kind="danger"))
    elif fetched_contact_matrices:
        _keys = ", ".join(
            "all subpops" if _k == "__shared__" else _k for _k in fetched_contact_matrices
        )
        _parts.append(mo.callout(
            mo.md(f"Fetched contact matrices ({fetched_matrices_scope}) for: {_keys}. "
                  "They are written into the config and used at run time."),
            kind="success",
        ))

    section_card(_header, mo.vstack(_parts), accent=_ACC)
    return


@app.cell
def _population_data(
    population_source_radio, risk_fraction_inputs, population_csv_input,
    fetched_populations, fetched_matrices_scope,
    is_metapop, geo_subpop_names,
    num_age_groups, num_risk_groups, age_groups,
    loaded_config, load_population_csv, np,
):
    # Resolve the per-subpopulation population into A×R arrays. Not gated on the
    # active tab so population_by_subpop stays available to the Model Builder
    # tab (initial conditions) and the run paths.
    _A = int(num_age_groups)
    _R = int(num_risk_groups)
    pop_subpop_names = (
        list(geo_subpop_names) if (is_metapop and geo_subpop_names) else ["aggregate_pop"]
    )
    population_source = population_source_radio.value
    population_by_subpop = {}
    population_errors = {}

    # Risk-group split fractions (renormalised; uniform fallback).
    _rf = np.array([float(_x) for _x in risk_fraction_inputs.value], dtype=float)
    if _rf.size != _R or _rf.sum() <= 0:
        _rf = np.full(_R, 1.0 / max(_R, 1))
    _rf = _rf / _rf.sum()

    if population_source == "CSV file":
        _pop, _err = load_population_csv(
            population_csv_input.value, pop_subpop_names, _A, _R, age_groups,
        )
        if _err:
            population_errors["error"] = _err
        elif _pop:
            population_by_subpop = _pop
    else:  # Fetch from geography
        if not fetched_populations:
            population_errors["info"] = (
                "No population fetched yet — choose a geography and press "
                "**Fetch contact matrices & population** above."
            )
        elif fetched_matrices_scope == "per_subpop":
            for _name in pop_subpop_names:
                _nk = fetched_populations.get(_name)
                if _nk:
                    population_by_subpop[_name] = np.round(
                        np.outer(np.asarray(_nk, dtype=float), _rf)
                    )
        else:
            _nk = fetched_populations.get("__shared__")
            if _nk:
                _arr = np.round(np.outer(np.asarray(_nk, dtype=float), _rf))
                for _name in pop_subpop_names:
                    population_by_subpop[_name] = _arr

    # Fallback for any subpop without a resolved population: reuse a saved value
    # from config (round-trip), else split total_population uniformly across cells.
    _saved_ic = loaded_config.get("initial_conditions", {}) or {}
    for _name in pop_subpop_names:
        if _name in population_by_subpop:
            continue
        _saved_pop = (_saved_ic.get(_name, {}) or {}).get("population")
        _arr = None
        if isinstance(_saved_pop, list):
            try:
                _cand = np.asarray(_saved_pop, dtype=float)
                if _cand.shape == (_A, _R):
                    _arr = _cand
            except Exception:
                _arr = None
        if _arr is None:
            _total = float(loaded_config.get("total_population", 10000))
            _arr = np.full((_A, _R), _total / max(_A * _R, 1))
        population_by_subpop[_name] = _arr
    return population_by_subpop, population_source, population_errors, pop_subpop_names


@app.cell
def _population_show(
    mo, main_tab,
    population_source_radio, risk_fraction_inputs, population_csv_input,
    num_risk_groups, age_groups, num_age_groups,
    population_by_subpop, population_source, population_errors, pop_subpop_names,
    param_grid_columns, pd,
    tip_label, step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Population & Geography", None)
    _ACC = CLT_ACCENT["population"]

    _CSV_FORMAT_TIP = (
        "Required columns: age, population\n\n"
        "  age — named band (matching the configured\n"
        "        age groups) or a 0-based index\n"
        "  population — count for that row\n\n"
        "Optional columns:\n"
        "  risk — 0-based index in 0..R-1\n"
        "        (required only if there is more\n"
        "        than one risk group)\n"
        "  subpopulation — must match a configured\n"
        "        subpopulation name (required only\n"
        "        if there is more than one)"
    )

    _parts = [
        mo.md(
            "Population totals per age group are fetched for the chosen geography "
            "(US states and countries supported via epydemix), or loaded from a CSV "
            "for custom / per-subpopulation populations."
        ),
        population_source_radio,
    ]
    if population_source == "Fetch from geography":
        if int(num_risk_groups) > 1:
            _parts.append(mo.md(
                "**Risk-group split** — the fetched (age-only) population is split "
                "across risk groups by these fractions (renormalised to sum to 1):"
            ))
            _parts.append(mo.hstack(list(risk_fraction_inputs), justify="start"))
    else:
        _parts.append(
            mo.hstack(
                [population_csv_input, tip_label("", _CSV_FORMAT_TIP)],
                justify="start", align="center", gap=0.5,
            )
        )

    if population_errors.get("error"):
        _parts.append(mo.callout(mo.md(f"**Population error:** {population_errors['error']}"),
                                 kind="danger"))
    elif population_errors.get("info"):
        _parts.append(mo.callout(mo.md(population_errors["info"]), kind="info"))

    _cols = param_grid_columns(age_groups, int(num_age_groups))
    for _name in pop_subpop_names:
        _arr = population_by_subpop.get(_name)
        if _arr is None:
            continue
        _label = "Population" if _name == "aggregate_pop" else f"Population — {_name}"
        _rows = [
            {"risk_group": str(_r), **{_c: _arr[_a, _r] for _a, _c in enumerate(_cols)}}
            for _r in range(_arr.shape[1])
        ]
        # Totals row: sum across risk groups for each age band.
        _rows.append({"risk_group": "Σ all risk",
                      **{_c: _arr[_a, :].sum() for _a, _c in enumerate(_cols)}})
        _df = pd.DataFrame(_rows)
        _parts.append(mo.md(f"**{_label}** (total {_arr.sum():,.0f})"))
        _parts.append(mo.ui.table(_df, selection=None))

    section_card(
        step_header("③", "Population sizes",
                    "Per age / risk-group population counts used as denominators.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return

@app.cell
def _builder_overview(mo, main_tab, CLT_ACCENT):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    _steps = [
        ("0", "Load config"), ("1", "Compartments"), ("2", "Transitions"),
        ("3", "Parameters"), ("4", "Schedules"), ("4", "Immunity"),
        ("5", "Diagram"), ("6", "Initial conditions"), ("7", "Sim settings"),
        ("8", "Config preview"), ("9", "Run"),
    ]
    _chips = "".join(
        '<span style="display:inline-flex;align-items:center;gap:.35rem;'
        "background:rgba(127,127,127,0.08);border:1px solid rgba(127,127,127,0.2);"
        "border-radius:999px;padding:.15rem .6rem .15rem .2rem;font-size:.8rem;"
        'white-space:nowrap;">'
        '<span style="display:inline-flex;align-items:center;justify-content:center;'
        "min-width:1.5em;height:1.5em;border-radius:999px;"
        f"background:{_ACC};color:#fff;font-weight:700;font-size:.78rem;\">{_n}</span>"
        f"{_t}</span>"
        for _n, _t in _steps
    )
    mo.Html(
        f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Model Builder</div>'
        '<div style="color:#777;margin:.1rem 0 .55rem;">Work top to bottom — each '
        "numbered card below is one step.</div>"
        f'<div style="display:flex;flex-wrap:wrap;gap:.4rem;">{_chips}</div>'
    )
    return


@app.cell
def _load_config_state(mo, Path):
    # Default to the example config that ships alongside this notebook, resolved
    # relative to the notebook file so it works on any machine. Falls back to an
    # empty string if the bundled example cannot be located.
    try:
        _default_config_path = str(
            Path(__file__).parent / "example_metapop_inputs" / "model_config.json"
        )
    except NameError:
        _default_config_path = ""

    get_config_path, set_config_path = mo.state(_default_config_path)
    return get_config_path, set_config_path


@app.cell
def _config_file_upload_ui(mo):
    # Kept in its own cell (independent of the path state) so that clearing the
    # path text box does not recreate this widget and wipe the browsed file.
    config_file_upload = mo.ui.file(
        filetypes=[".json"],
        label="Browse for config JSON",
    )
    return (config_file_upload,)


@app.cell
def _load_config_ui(mo, get_config_path, set_config_path):
    config_path_input = mo.ui.text(
        value=get_config_path(),
        on_change=set_config_path,
        placeholder="/path/to/model_config.json  (or use Browse above)",
        label="Or enter config JSON path directly",
        full_width=True,
    )
    return (config_path_input,)


@app.cell
def _clear_path_on_browse(config_file_upload, set_config_path):
    # When a file is browsed via the OS picker, clear the manual-path text box.
    # mo.ui.file can't expose the real filesystem path (browsers withhold it),
    # so we empty the box; the browsed file's name is shown next to Browse and
    # the parse cell already prioritizes the browsed file over the text path.
    if config_file_upload.value:
        set_config_path("")
    return


@app.cell
def _clear_config_button_ui(mo, set_config_path):
    clear_config_button = mo.ui.button(
        label="Clear config",
        on_click=lambda _: set_config_path(""),
    )
    return (clear_config_button,)


@app.cell
def _load_config_parse(config_file_upload, config_path_input, load_config_json, json):
    _loaded_config = {}
    _cfg_err = None
    _source = None

    if config_file_upload.value:
        _file = config_file_upload.value[0]
        try:
            _loaded_config = json.loads(_file.contents.decode("utf-8"))
        except Exception as _exc:
            _cfg_err = f"JSON parse error: {_exc}"
        _source = f"Browsed: **{_file.name}**"
    elif config_path_input.value.strip():
        _loaded_config, _cfg_err = load_config_json(config_path_input.value)
        _source = "path"

    loaded_config = _loaded_config
    return (loaded_config,)


@app.cell
def _load_config_display(
    config_file_upload, config_path_input, clear_config_button,
    loaded_config, load_config_json, mo, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]

    _cfg_err = None
    _source = None
    if config_file_upload.value:
        _source = f"Browsed: **{config_file_upload.value[0].name}**"
        try:
            import json as _json
            _json.loads(config_file_upload.value[0].contents.decode("utf-8"))
        except Exception as _exc:
            _cfg_err = f"JSON parse error: {_exc}"
    elif config_path_input.value.strip():
        _, _cfg_err = load_config_json(config_path_input.value)
        _source = "path"

    _browse_row = mo.hstack([config_file_upload] + (
        [mo.md(f"Selected: `{config_file_upload.value[0].name}`")]
        if config_file_upload.value else []
    ), align="center", gap=1)

    _parts = [
        _browse_row,
        config_path_input,
        clear_config_button,
    ]
    if _source:
        if _cfg_err:
            _parts.append(mo.callout(mo.md(f"**Load error:** {_cfg_err}"), kind="danger"))
        else:
            _n_comp = len(loaded_config.get("compartments", []))
            _n_tr = len(loaded_config.get("transitions", []))
            _ar = loaded_config.get("age_risk", {})
            _A = _ar.get("num_age_groups", 1)
            _R = _ar.get("num_risk_groups", 1)
            if _source == "path":
                _label = f"Loaded from `{config_path_input.value.strip()}`"
            else:
                _label = _source
            _parts.append(mo.callout(
                mo.md(
                    f"{_label} — **{_n_comp}** compartments, **{_n_tr}** transitions, "
                    f"**{_A}** age group(s), **{_R}** risk group(s)."
                ),
                kind="success",
            ))
    else:
        _parts.append(mo.callout(
            mo.md("No config loaded — all fields below use their defaults. Enter a path or browse for a JSON file to pre-populate the form, or fill it in manually."),
            kind="info",
        ))
    section_card(
        step_header(0, "Load Existing Config",
                    "Optional — pre-fill the form from a saved config JSON.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------


@app.cell
def _intro(mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.md(
        """
        # Generic Epidemic Model Builder

        Build, visualise, and run a config-driven epidemic model without editing JSON.
        Supports all rate templates, configurable age/risk groups, CSV-backed schedules,
        and multi-subpopulation (metapopulation) models.

        **Quick start:** set up the **Population & Geography** tab first, then work through
        Steps 1–9 here in order and press **Run simulation**.
        Load a previously saved config in **Step 0** to restore any prior setup.
        """
    )
    return


@app.cell
def _instructions(mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.accordion({
        "📋 Workflow overview": mo.md("""
**Population & Geography tab** *(do this first)*
Choose the number of age groups (A) and risk groups (R), single-population vs.
metapopulation mode, and (when using named age bands) fetch contact matrices for a
chosen geography. For metapop, enter the path to a folder containing the required input
files (see the *Metapopulation folder conventions* section below).

**Step 0 — Load existing config** *(optional)*
Enter the path to a `model_config.json` file to pre-populate all fields below.
Leave blank to start fresh.

**Step 1 — Compartments**
Enter compartment names as a comma-separated list, e.g. `S, E, I, R`.

**Step 2 — Transitions**
Define each transition: origin compartment → destination, and the rate template.
Available templates:
- `constant_param` — fixed rate from a single parameter
- `param_product` — product of multiple parameters (with optional complement factors)
- `immunity_modulated` — base rate adjusted by infection/vaccine immunity (M, MV)
- `force_of_infection` — standard FOI with contact matrix and optional humidity/immunity
- `force_of_infection_travel` — FOI with inter-subpop travel mixing (metapop only)

**Step 3 — Parameters**
Numeric sliders appear automatically for every parameter name referenced by your transitions.

**Step 4 — Schedules and Immunity**
For rate templates that use schedules (humidity, mobility, vaccines):
- Choose *constant* to use a single scalar value for the whole simulation.
- Choose *csv* to load a real time-varying schedule from a CSV file.
Contact matrices (total, school, work) are always stored as inline arrays in the
config JSON. When A > 1, supply them via CSV paths in the contact matrix fields, fetch
them in the Population & Geography tab, or load a saved config that already has them
embedded. Risk groups (R > 1) affect transition and susceptibility parameters but do not
require separate contact matrices.

**Step 5 — Model diagram**
Auto-generated from your compartments and transitions. Requires `graphviz`; falls back
to a simple matplotlib diagram if not installed.

**Step 6 — Initial conditions**
Seed each compartment by age and risk group (absolute counts) in an editable table;
the first compartment receives the remaining population per cell. Population totals
come from the **Population & Geography** tab (fetched per age group, split across risk
groups, or loaded from a CSV). In metapopulation mode, pick a subpopulation to edit its
table — these override `initial_conditions_{name}.json` in the metapop folder, which is
used as a fallback for any subpop left without seeds.

**Step 7 — Simulation settings**
Days, deterministic vs. stochastic, number of replicates, RNG seed, timesteps per day.

**Step 8 — Config preview and download**
The full config JSON (including file paths and age/risk group settings) is shown and
can be downloaded. The downloaded file can be re-loaded in Step 0.

**Step 9 — Run**
Press the *Run simulation* button. Results appear as epidemic curves and a summary table.
        """),

        "📁 Schedule CSV formats": mo.md("""
All CSV files should have a header row. Index columns (unnamed first column) are
ignored automatically.

**`absolute_humidity.csv`** — shared across subpops
```
date,absolute_humidity
2024-01-01,0.0043
2024-01-02,0.0041
```

**`school_work_calendar_{name}.csv`** — per-subpop (or shared)
```
date,is_school_day,is_work_day
2024-01-01,0.0,0.0
2024-01-02,1.0,1.0
```
Values are floats in [0, 1] (fractional school/work day allowed).

**`mobility_modifier.csv`** — shared, day-of-week indexed, JSON A×R array per row
```
day_of_week,mobility_modifier
Monday,"[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]"
Tuesday,"[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]"
```
The JSON array shape must be A rows × R columns.

**`vaccines_{name}.csv`** — per-subpop, date-indexed, JSON A×R array per row

Each value is the **proportion** of that age×risk group vaccinated on that day
(i.e. daily count ÷ group population), not a raw count.
```
date,daily_vaccines
2024-01-01,"[[0.000417, 0.000667], [0.000288, 0.000615], [0.001563, 0.003]]"
```

**Contact matrix CSVs** — plain floats, A×A, no header row, no index column *(optional)*
```
7.0,3.0,0.5
3.0,9.0,1.5
0.5,1.5,4.0
```
Separate files for total, school, and work contact matrices. When not provided,
the matrices embedded in the loaded config JSON are used directly.
        """),

        "🗂️ Metapopulation folder conventions": mo.md("""
Create a folder with files following these naming conventions.
The folder path is entered in the **Population & Geography** tab and saved in the config JSON.

**Required files:**

| File | Description |
|---|---|
| `metapop_config.json` | `subpopulations` (ordered list of names) and `travel_matrix` (N×N list of lists, rows sum to 1) |

**Optional shared files** (used by all subpops if present):

| File | Description |
|---|---|
| `absolute_humidity.csv` | `date`, `absolute_humidity` |
| `mobility_modifier.csv` | `day_of_week`, `mobility_modifier` (JSON A×R) |

**Optional per-subpop files** (`{name}` = a name from `metapop_config.json → subpopulations`):

| File | Description |
|---|---|
| `school_work_calendar_{name}.csv` | `date`, `is_school_day`, `is_work_day` |
| `vaccines_{name}.csv` | `date`, `daily_vaccines` (JSON A×R) |
| `initial_conditions_{name}.json` | `compartments` and `epi_metrics` keys, each mapping name → A×R list |

**Per-subpopulation parameter overrides** (`subpop_params` in `model_config.json`):

Any parameter in the `params` block can be overridden on a per-subpop basis by adding a
`subpop_params` section to `model_config.json`. Each key is a subpopulation name
(matching `metapop_config.json → subpopulations`); each value is a dict of parameter
overrides applied only to that subpopulation. Scalar and A×R array values are both supported.
Parameters not listed under a subpop continue to use the shared value from `params`.

```json
"subpop_params": {
  "East": { "beta_baseline": 0.050 },
  "West": {
    "beta_baseline": 0.038,
    "IP_to_ISH_prop": [[0.008], [0.003], [0.007], [0.012], [0.100]]
  }
}
```

Example `initial_conditions_West.json`:
```json
{
  "compartments": {
    "S": [[31680], [96589], [344716], [116909], [87681]],
    "E": [[0], [0], [30], [0], [0]]
  },
  "epi_metrics": {
    "M":  [[0.1], [0.1], [0.06], [0.08], [0.04]],
    "MV": [[0.0], [0.0], [0.0],  [0.0],  [0.0]]
  }
}
```

Per-subpop initial conditions seeded in the **Step 6** tables take precedence over this
file. The file is used as a fallback for any subpopulation left without seeds in Step 6.
If neither is present, all compartments are initialised to zero and the simulation will
stop with an error when the model has more than one age or risk group.

**Example folder** is included in the repository at
`generic_core/examples/example_metapop_inputs/` (2 subpops, 3 age groups, 2 risk groups,
SEIR model). Re-generate it with::

    python generic_core/examples/generate_example_metapop_data.py
        """),

        "💾 Config save / load round-trip": mo.md("""
The downloaded `model_config.json` contains everything needed to restore the session:

- Compartment names, transition definitions, parameter values
- Age/risk group counts (`age_risk` section)
- CSV schedule files (`input_files` section): a shared `input_folder` plus the
  filename of each CSV used (humidity, calendar, mobility, vaccines, contact matrices)
- Initial conditions (`initial_conditions` section): per-subpopulation population (A×R)
  and per-compartment seed counts, plus the derived `total_population`
- Per-subpopulation parameter overrides (`subpop_params` section, if present)

**To reload:** paste the path into the Step 0 text field. All UI fields (compartments,
transitions, parameters, immunity toggles, file paths, metapop folder) will be
pre-populated automatically.

**Note on contact matrices:** When A > 1, matrix values *are* embedded inline in the
config JSON under `params` (as nested lists). CSV file paths are optional — if provided
in Step 4, they override the inline values at load time. If no CSV paths are set, the
inline param arrays are used as-is.

**Note on `subpop_params`:** These overrides are written directly in `model_config.json`
and are not editable via the UI sliders — edit the JSON file directly to add or change
per-subpop values. They are preserved across save/load cycles.
        """),

        "⚡ Rate template quick reference": mo.md("""
| Template | When to use | Example | Required rate_config keys |
|---|---|---|---|
| `constant_param` | Single fixed-rate transition | E→I recovery at rate `gamma` | `param` |
| `param_product` | Product of two or more parameters | S→H at `sigma × hosp_prop`; complement branch S→I at `sigma × (1 − hosp_prop)` | `factors` (list); optionally `complement_factors` |
| `immunity_modulated` | Rate that scales down as population immunity (M/MV) accumulates | S→E exposure rate suppressed by prior infection/vaccine immunity | `base_rate`, `proportion`, `is_complement`; optionally `inf_reduce_param`, `vax_reduce_param` |
| `force_of_infection` | Standard frequency-dependent incidence with a contact matrix | S→E infection driven by `beta`, contact patterns, and infectious compartments I/A | `beta_param`, `contact_matrix_schedule`, `infectious_compartments`, `relative_susceptibility_param`; optionally humidity/immunity fields |
| `force_of_infection_travel` | FOI with commuter mixing across subpopulations *(metapop only)* | S→E where residents of subpop A contact infectious individuals from subpop B during work hours | Same as above plus `travel_config` with `immobile_compartments`, `mobility_schedule` |
| `scheduled_exact` | Deterministic, exact transfer of a scheduled daily count (not a stochastic rate) | S→Vaccinated moving exactly the (rounded, delay-shifted) vaccinated count each day | `schedule` (name of a schedule, e.g. a `vaccine_schedule` instance) |

**Infectious compartments field** uses the format `CompartmentName:relative_infectivity_param`
(or just `CompartmentName` if all compartments are equally infectious), comma-separated.
Example: `IP:IP_relative_inf, IA:IA_relative_inf, ISR, ISH`
        """),
    })
    return


# ---------------------------------------------------------------------------
# Population structure (A/R, single vs metapop, metapop folder) now lives in the
# Population & Geography tab — see _nb_population.py. It exports num_age_groups,
# num_risk_groups, is_metapop, and metapop_folder_input, which the cells below
# consume unchanged.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1 — Compartments
# ---------------------------------------------------------------------------


@app.cell
def _compartments_ui(mo, loaded_config):
    _default = ", ".join(loaded_config.get("compartments", ["S", "E", "I", "R"]))
    compartments_text = mo.ui.text(
        value=_default,
        placeholder="S, E, I, R",
        label="Compartments (comma-separated)",
        full_width=True,
    )
    return (compartments_text,)


@app.cell
def _compartments_parse(compartments_text):
    raw = [_c.strip() for _c in compartments_text.value.split(",") if _c.strip()]
    compartments = list(dict.fromkeys(raw))
    return (compartments,)


@app.cell
def _compartments_display(
    compartments, compartments_text, mo, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    if compartments:
        _body = mo.md("**Parsed:** " + "  ".join(f"`{_c}`" for _c in compartments))
    else:
        _body = mo.callout(mo.md("Enter at least one compartment name."), kind="warn")
    section_card(
        step_header(1, "Compartments",
                    "The disease states individuals move between (e.g. S, E, I, R).",
                    accent=_ACC),
        mo.vstack([compartments_text, _body]),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 2 — Transitions
# ---------------------------------------------------------------------------


@app.cell
def _transition_count_ui(mo, loaded_config):
    _n_loaded = len(loaded_config.get("transitions", []))
    n_transitions = mo.ui.number(
        start=1,
        stop=12,
        step=1,
        value=min(max(_n_loaded, 1), 12) if _n_loaded else 3,
        label="Number of transitions",
    )
    return (n_transitions,)


@app.cell
def _transition_forms_ui(compartments, loaded_config, mo):
    _max_t = 12
    _comps = compartments if compartments else ["?"]
    _templates = [
        "constant_param",
        "param_product",
        "immunity_modulated",
        "force_of_infection",
        "force_of_infection_travel",
        "scheduled_exact",
    ]
    _t_cfgs = loaded_config.get("transitions", [])

    def _tget(i, key, default):
        if i < len(_t_cfgs):
            return _t_cfgs[i].get(key, default)
        return default

    def _rcget(i, key, default):
        if i < len(_t_cfgs):
            return _t_cfgs[i].get("rate_config", {}).get(key, default)
        return default

    def _origin_default(i):
        _v = _tget(i, "origin", None)
        if _v and _v in _comps:
            return _v
        return _comps[i] if i < len(_comps) else _comps[0]

    def _dest_default(i):
        _v = _tget(i, "destination", None)
        if _v and _v in _comps:
            return _v
        return _comps[i + 1] if i < len(_comps) - 1 else _comps[-1]

    t_name = mo.ui.array([
        mo.ui.text(
            value=_tget(_i, "name", f"{_origin_default(_i)}_to_{_dest_default(_i)}"),
            label="Name",
        )
        for _i in range(_max_t)
    ])
    t_origin = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_origin_default(_i), label="Origin")
        for _i in range(_max_t)
    ])
    t_dest = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_dest_default(_i), label="Destination")
        for _i in range(_max_t)
    ])
    t_template = mo.ui.array([
        mo.ui.dropdown(
            options=_templates,
            value=_tget(_i, "rate_template", "constant_param"),
            label="",
        )
        for _i in range(_max_t)
    ])

    t_param = mo.ui.array([
        mo.ui.text(
            value=_rcget(
                _i, "param", f"{_origin_default(_i)}_to_{_dest_default(_i)}_rate"
            ),
            label="Param name",
        )
        for _i in range(_max_t)
    ])
    t_schedule_name = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "schedule", "vaccinated_transfer_schedule"),
            label="Schedule name",
        )
        for _i in range(_max_t)
    ])
    t_factors = mo.ui.array([
        mo.ui.text(value=", ".join(_rcget(_i, "factors", [])), label="")
        for _i in range(_max_t)
    ])
    t_complements = mo.ui.array([
        mo.ui.text(value=", ".join(_rcget(_i, "complement_factors", [])), label="")
        for _i in range(_max_t)
    ])

    t_base_rate = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "base_rate", "base_rate"), label="Base rate param")
        for _i in range(_max_t)
    ])
    t_proportion = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "proportion", "split_prop"), label="Proportion param")
        for _i in range(_max_t)
    ])
    t_is_complement = mo.ui.array([
        mo.ui.checkbox(
            label="Use complement branch",
            value=bool(_rcget(_i, "is_complement", False)),
        )
        for _i in range(_max_t)
    ])
    t_inf_reduce = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "inf_reduce_param", "inf_risk_reduce"),
            label="Infection reduction param",
        )
        for _i in range(_max_t)
    ])
    t_vax_reduce = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "vax_reduce_param", "vax_risk_reduce"),
            label="Vaccine reduction param",
        )
        for _i in range(_max_t)
    ])

    t_beta = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "beta_param", "beta_baseline"), label="Beta param")
        for _i in range(_max_t)
    ])
    t_rel_sus = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "relative_susceptibility_param", "relative_suscept"),
            label="Relative susceptibility param",
        )
        for _i in range(_max_t)
    ])

    def _travel_config_get(i, key, default):
        """Look up a key inside rate_config.travel_config (for force_of_infection_travel)."""
        _tc = _rcget(i, "travel_config", None)
        if _tc and isinstance(_tc, dict):
            return _tc.get(key, default)
        return default

    def _infectious_default(i):
        _raw = _rcget(i, "infectious_compartments", None)
        if _raw is None:
            _raw = _travel_config_get(i, "infectious_compartments", None)
        if _raw and isinstance(_raw, dict):
            return ", ".join(_raw.keys())
        return "I"

    t_infectious = mo.ui.array([
        mo.ui.text(
            value=_infectious_default(_i),
            label="",
            placeholder="IP, IA, ISR, ISH",
        )
        for _i in range(_max_t)
    ])
    t_use_humidity = mo.ui.array([
        mo.ui.checkbox(
            label="Include humidity modifier",
            value=bool(_rcget(_i, "humidity_impact_param", None)),
        )
        for _i in range(_max_t)
    ])
    t_humidity_impact = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "humidity_impact_param", "humidity_impact"),
            label="Humidity impact param",
        )
        for _i in range(_max_t)
    ])
    t_use_foi_immunity = mo.ui.array([
        mo.ui.checkbox(
            label="Include immunity modifier",
            value=bool(_rcget(_i, "inf_reduce_param", None)),
        )
        for _i in range(_max_t)
    ])
    t_immobile = mo.ui.array([
        mo.ui.text(
            value=", ".join(
                _travel_config_get(_i, "immobile_compartments", None)
                or _rcget(_i, "immobile_compartments", [])
            ),
            label="",
        )
        for _i in range(_max_t)
    ])

    return (
        t_name, t_origin, t_dest, t_template,
        t_param, t_schedule_name, t_factors, t_complements,
        t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
        t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
        t_use_foi_immunity, t_immobile,
    )


@app.cell
def _transition_show(
    mo,
    main_tab,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_schedule_name, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
    tip_label, with_tip,
    step_header, section_card, CLT_ACCENT,
):
    _ACC = CLT_ACCENT["builder"]
    mo.stop(main_tab.value != "Model Builder", None)
    _IMMUNITY_TIP = (
        "Divides the rate or force of infection by a population-level immunity factor:\n\n"
        "  immunity_force =\n"
        "    1\n"
        "    + (r_inf / (1 − r_inf)) × M\n"
        "    + (r_vax / (1 − r_vax)) × MV\n\n"
        "  r_inf = inf_reduce_param ∈ [0, 1)\n"
        "  r_vax = vax_reduce_param ∈ [0, 1)\n"
        "  M  = cumulative infection-induced immunity\n"
        "  MV = cumulative vaccine-induced immunity\n\n"
        "Higher r → stronger rate reduction.\n"
        "Example: r_inf = 0.5, M = 1 → rate halved.\n\n"
        "Requires at least one of M or MV to be enabled\n"
        "in Step 4, otherwise immunity_force stays at 1."
    )

    def _immunity_checkbox(checkbox):
        return mo.hstack([checkbox, tip_label("", _IMMUNITY_TIP)], justify="start", align="center")


    _n = int(n_transitions.value)
    _acc = {}
    for _i in range(_n):
        _template = t_template.value[_i]

        if _template == "constant_param":
            _rate_ui = t_param[_i]
        elif _template == "param_product":
            _rate_ui = mo.vstack([
                with_tip(
                    "Factors",
                    "Comma-separated parameter names multiplied together to form the rate.\n\n"
                    "Example: base_rate, hosp_prop\n"
                    "Rate = base_rate × hosp_prop\n\n"
                    "Each name gets a slider in Step 3.",
                    t_factors[_i],
                ),
                with_tip(
                    "Complement factors",
                    "Parameters applied as (1 − param) factors in the product.\n"
                    "Useful for modelling the fraction that does NOT take a given path.\n\n"
                    "Example: hosp_prop as a complement (with base_rate as a factor)\n"
                    "Rate = base_rate × (1 − hosp_prop)",
                    t_complements[_i],
                ),
            ])
        elif _template == "immunity_modulated":
            _rate_ui = mo.vstack([
                t_base_rate[_i],
                t_proportion[_i],
                t_is_complement[_i],
                _immunity_checkbox(t_use_foi_immunity[_i]),
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        elif _template == "force_of_infection":
            _foi_items = [
                t_beta[_i],
                t_rel_sus[_i],
                with_tip(
                    "Infectious compartments",
                    "Comma-separated names of compartments that contribute to\n"
                    "this force of infection.\n\n"
                    "Example: I, A\n\n"
                    "Each compartment's relative infectiousness (vs. 1x baseline)\n"
                    "is set once for the whole model in Step 1 — Compartments, so\n"
                    "the same value is shared by every transition that lists it.",
                    t_infectious[_i],
                ),
                t_use_humidity[_i],
            ]
            if t_use_humidity.value[_i]:
                _foi_items.append(t_humidity_impact[_i])
            _foi_items.append(_immunity_checkbox(t_use_foi_immunity[_i]))
            if t_use_foi_immunity.value[_i]:
                _foi_items.extend([t_inf_reduce[_i], t_vax_reduce[_i]])
            _rate_ui = mo.vstack(_foi_items)
        elif _template == "scheduled_exact":
            _rate_ui = mo.vstack([
                with_tip(
                    "Schedule name",
                    "Name of the schedule providing the exact daily count of\n"
                    "individuals to move from origin to destination (e.g. a\n"
                    "vaccine_schedule instance backed by a per-subpop CSV with\n"
                    "one AxR array per day).\n\n"
                    "The count is rounded to the nearest integer and capped at\n"
                    "the origin compartment's current population -- this is a\n"
                    "deterministic, exact transfer, not a stochastic rate.\n\n"
                    "Configure the underlying data source and delay in Step 4.",
                    t_schedule_name[_i],
                ),
            ])
        else:
            _foit_items = [
                t_beta[_i],
                t_rel_sus[_i],
                with_tip(
                    "Infectious compartments",
                    "Comma-separated names of compartments that contribute to\n"
                    "this force of infection.\n\n"
                    "Example: I, A\n\n"
                    "Each compartment's relative infectiousness (vs. 1x baseline)\n"
                    "is set once for the whole model in Step 1 — Compartments, so\n"
                    "the same value is shared by every transition that lists it.",
                    t_infectious[_i],
                ),
                t_use_humidity[_i],
            ]
            if t_use_humidity.value[_i]:
                _foit_items.append(t_humidity_impact[_i])
            _foit_items.append(_immunity_checkbox(t_use_foi_immunity[_i]))
            if t_use_foi_immunity.value[_i]:
                _foit_items.extend([t_inf_reduce[_i], t_vax_reduce[_i]])
            _foit_items.append(with_tip(
                "Immobile compartments",
                "Comma-separated compartment names whose members do NOT travel\n"
                "between subpopulations (no cross-subpop mixing).\n\n"
                "Example: H, ICU",
                t_immobile[_i],
            ))
            _rate_ui = mo.vstack(_foit_items)

        _o = str(t_origin.value[_i]).strip() or "?"
        _d = str(t_dest.value[_i]).strip() or "?"
        _nm = str(t_name.value[_i]).strip()
        _label = f"{_i + 1}. {_o} → {_d}  ·  {_template}"
        if _nm:
            _label += f"  ({_nm})"
        _label = f'<span style="font-size: 0.85em;">{_label}</span>'
        _acc[_label] = mo.vstack([
            mo.vstack([
                mo.hstack([t_origin[_i], t_dest[_i]], justify="start"),
                t_name[_i],
                with_tip(
                    "Rate template",
                    "Determines how the transition rate is computed each timestep.\n\n"
                    "constant_param — single fixed rate parameter\n"
                    "  e.g. E→I recovery at rate gamma\n\n"
                    "param_product — product of multiple parameters\n"
                    "  e.g. S→H at sigma × hosp_prop\n\n"
                    "immunity_modulated — rate suppressed by cumulative infection/vaccine immunity (M/MV)\n"
                    "  e.g. S→E exposure dampened by prior immunity\n\n"
                    "force_of_infection — standard frequency-dependent incidence with a contact matrix\n"
                    "  e.g. S→E driven by beta, contact patterns, and infectious compartments\n\n"
                    "force_of_infection_travel — FOI with commuter mixing across subpopulations (metapop only)\n"
                    "  e.g. S→E where residents of subpop A contact infectious people from subpop B\n\n"
                    "scheduled_exact — exact, deterministic transfer of a scheduled daily count\n"
                    "  e.g. S→Vaccinated moving exactly the vaccinated count each day (not stochastic)\n\n"
                    "See the ⚡ Rate template quick reference accordion above for full details.",
                    t_template[_i],
                ),
            ]),
            _rate_ui,
        ])

    section_card(
        step_header(2, "Transitions",
                    "Define each flow between compartments and how its rate is computed. "
                    "Click a transition to expand it.",
                    accent=_ACC),
        mo.vstack([n_transitions, mo.accordion(_acc, multiple=True)]),
        accent=_ACC,
    )
    return


@app.cell
def _template_requirements(
    n_transitions, t_template, t_use_humidity, t_use_foi_immunity,
):
    _n = int(n_transitions.value)
    _uses_contact_matrix = False
    _uses_absolute_humidity = False
    _uses_mobility = False
    _requires_immunity_metrics = False
    _uses_scheduled_transfer = False

    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "immunity_modulated":
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection_travel":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _uses_mobility = True
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "scheduled_exact":
            _uses_scheduled_transfer = True

    uses_absolute_humidity = _uses_absolute_humidity
    uses_contact_matrix = _uses_contact_matrix
    uses_mobility = _uses_mobility
    requires_immunity_metrics = _requires_immunity_metrics
    uses_scheduled_transfer = _uses_scheduled_transfer
    return (
        uses_absolute_humidity, uses_contact_matrix, uses_mobility,
        requires_immunity_metrics, uses_scheduled_transfer,
    )


@app.cell
def _collect_param_names(
    n_transitions, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact, t_use_foi_immunity,
    parse_csv_list, rel_inf_param_name,
):
    _n = int(n_transitions.value)
    _names = []
    _infectious_comp_names = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _p = t_param.value[_i].strip()
            if _p:
                _names.append(_p)
        elif _template == "param_product":
            _names.extend(parse_csv_list(t_factors.value[_i]))
            _names.extend(parse_csv_list(t_complements.value[_i]))
        elif _template == "immunity_modulated":
            for _p in (t_base_rate.value[_i], t_proportion.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
        elif _template == "force_of_infection":
            for _p in (t_beta.value[_i], t_rel_sus.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _infectious_comp_names.extend(parse_csv_list(t_infectious.value[_i]))
        elif _template == "force_of_infection_travel":
            for _p in (t_beta.value[_i], t_rel_sus.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _infectious_comp_names.extend(parse_csv_list(t_infectious.value[_i]))

    param_names = list(dict.fromkeys(_names))

    _reduce_names = set()
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template in ("immunity_modulated", "force_of_infection", "force_of_infection_travel"):
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _reduce_names.add(_p)
    reduce_param_names = _reduce_names

    infectious_compartment_names = list(dict.fromkeys(_infectious_comp_names))
    rel_inf_param_names = [rel_inf_param_name(_c) for _c in infectious_compartment_names]
    param_names = list(dict.fromkeys(param_names + rel_inf_param_names))

    return param_names, reduce_param_names, infectious_compartment_names, rel_inf_param_names


# ---------------------------------------------------------------------------
# Step 3 — Parameters
# ---------------------------------------------------------------------------


@app.cell
def _params_ui(param_names, reduce_param_names, rel_inf_param_names, loaded_config, mo,
               is_array_param, param_grid_columns, num_age_groups, num_risk_groups, age_groups):
    _saved_params = loaded_config.get("params", {})
    _rel_inf_set = set(rel_inf_param_names)
    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)
    _risk_labels = [f"risk{_r}" for _r in range(_R)]

    def _scalar_seed(_name):
        _v = _saved_params.get(_name, 0.5 if _name in reduce_param_names else 1.0)
        # A param saved as an A×R array still needs a single number to seed the
        # scalar input — use its first entry.
        while isinstance(_v, list):
            _v = _v[0]
        return float(_v)

    def _grid_seed(_name, _a, _r):
        # Saved A×R params are nested [age][risk] lists (see _build_config); pull
        # each cell's own value instead of collapsing the whole array to one number.
        _v = _saved_params.get(_name)
        if (isinstance(_v, list) and len(_v) == _A
                and all(isinstance(_row, list) and len(_row) == _R for _row in _v)):
            return float(_v[_a][_r])
        return _scalar_seed(_name)

    # One toggle / scalar input / A×R data_editor per param, all built every run
    # so that flipping one param's toggle doesn't shift the others' widget
    # identity (which would otherwise reset their live values on rerun). Wrapped
    # in mo.ui.dictionary (not a plain dict) so interacting with any nested
    # element triggers reactive reruns of cells that reference the container —
    # a plain dict of UI elements does not propagate value-change reactivity.
    param_vary_toggles = mo.ui.dictionary({
        _name: mo.ui.checkbox(
            value=is_array_param(loaded_config, _name),
            label="Vary by age/risk group",
        )
        for _name in param_names
    })
    param_scalar_inputs = mo.ui.dictionary({
        _name: mo.ui.number(
            start=0.0, stop=10.0, step=None,
            value=_scalar_seed(_name),
            label="" if _name in _rel_inf_set else _name,
        )
        for _name in param_names
    })
    # Rows are risk groups, columns are age groups (named from the Population &
    # Geography tab's age bands when available); "risk_group" is a read-only
    # row-label column.
    param_grid_inputs = mo.ui.dictionary({
        _name: mo.ui.data_editor(
            data=[
                {"risk_group": _risk_labels[_r],
                 **{_c: _grid_seed(_name, _a, _r) for _a, _c in enumerate(_age_cols)}}
                for _r in range(_R)
            ],
            label="" if _name in _rel_inf_set else _name,
            editable_columns=list(_age_cols),
        )
        for _name in param_names
    })
    return param_vary_toggles, param_scalar_inputs, param_grid_inputs


@app.cell
def _params_show(
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    infectious_compartment_names, rel_inf_param_names,
    mo, main_tab,
    tip_label, step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    import html as _html
    _ACC = CLT_ACCENT["builder"]

    _rel_inf_tip = (
        "Relative infectiousness of this compartment vs. baseline (1.0).\n\n"
        "Used by every force-of-infection transition (Step 2) that lists "
        "this compartment as infectious — the same value applies everywhere "
        "it's used."
    )

    def _param_row(_name, _label_node):
        _toggle = param_vary_toggles[_name]
        _content = param_grid_inputs[_name] if _toggle.value else param_scalar_inputs[_name]
        return mo.vstack([
            mo.hstack([_label_node, _toggle], wrap=True, justify="start"),
            _content,
        ])

    _regular_names = [_n for _n in param_names if _n not in rel_inf_param_names]

    _parts = []
    if not param_names:
        _parts.append(mo.callout(mo.md("No transition parameters found yet."), kind="warn"))
    for _name in _regular_names:
        _parts.append(_param_row(_name, mo.md(f"**`{_name}`**")))

    if rel_inf_param_names:
        _rel_rows = []
        for _comp, _name in zip(infectious_compartment_names, rel_inf_param_names):
            _rel_rows.append(_param_row(
                _name,
                tip_label(f"<b><code>{_html.escape(_comp)}</code></b>", _rel_inf_tip),
            ))
        _parts.append(mo.accordion(
            {"Relative infectiousness (per infectious compartment)": mo.vstack(_rel_rows)},
        ))

    section_card(
        step_header(3, "Parameters",
                    "Rates from your transitions. Toggle a parameter to vary it by age / risk group.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 4 — Schedules and Immunity (scalar inputs + CSV file paths)
# ---------------------------------------------------------------------------


@app.cell
def _schedule_and_immunity_ui(
    mo, loaded_config, num_age_groups, num_risk_groups, age_groups, param_grid_columns,
    is_metapop, pop_subpop_names,
):
    _epi_names = [_m["name"] for _m in loaded_config.get("epi_metrics", [])]
    _saved_params = loaded_config.get("params", {})
    include_inf_immunity = mo.ui.checkbox(
        label="Include infection-induced immunity metric (M)",
        value="M" in _epi_names,
    )
    include_vax_immunity = mo.ui.checkbox(
        label="Include vaccine-induced immunity metric (MV)",
        value="MV" in _epi_names,
    )
    total_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=1.0, label="Total contact matrix value",
    )
    school_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=0.0, label="School contact subtraction",
    )
    work_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=0.0, label="Work contact subtraction",
    )
    mobility_input = mo.ui.number(
        start=0.0, stop=5.0, step=None, value=1.0, label="Mobility modifier",
    )
    daily_vaccines_input = mo.ui.number(
        start=0.0, stop=1e9, step=1.0, value=0.0, label="",
    )
    # "Vary by age/risk group" toggle + A×R grid, mirroring the Step 3
    # Parameters scalar/grid pattern — lets a constant daily-vaccines value
    # differ per age/risk group instead of broadcasting one number to all.
    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)
    _risk_labels = [f"risk{_r}" for _r in range(_R)]
    daily_vaccines_vary_toggle = mo.ui.checkbox(
        value=False, label="Vary by age/risk group",
    )
    daily_vaccines_grid_input = mo.ui.data_editor(
        data=[
            {"risk_group": _risk_labels[_r], **{_c: 0.0 for _c in _age_cols}}
            for _r in range(_R)
        ],
        label="",
        editable_columns=list(_age_cols),
    )

    # Optional per-subpopulation override of the constant daily-vaccines value
    # (metapop only). The editor widget for the selected subpop is rebuilt
    # fresh on every render (see _schedule_csv_show) with its on_change
    # writing into this state dict — a single long-lived widget reused across
    # subpops would reset to its construction-time default each time the
    # selector switches back to a previously-edited subpop (see _init_ui's
    # get_seed_values/set_seed_values, which hit the same issue for Step 6).
    get_subpop_vax_values, set_subpop_vax_values = mo.state(
        dict(loaded_config.get("subpop_daily_vaccines", {}) or {})
    )
    daily_vaccines_per_subpop_toggle = mo.ui.checkbox(
        value=False, label="Vary by subpopulation",
    )
    daily_vaccines_subpop_selector = mo.ui.dropdown(
        options=list(pop_subpop_names),
        value=pop_subpop_names[0] if pop_subpop_names else None,
        label="Subpopulation",
    )
    vax_transfer_delay_input = mo.ui.number(
        start=0, stop=60, step=1,
        value=int(loaded_config.get("params", {}).get("vax_transfer_delay_days", 0)),
        label="vax_transfer_delay_days",
    )
    vaccinated_compartment_reset_date_input = mo.ui.text(
        value=str(loaded_config.get("params", {}).get(
            "vaccinated_compartment_reset_date_mm_dd", "")),
        placeholder="07_30",
        label="vaccinated_compartment_reset_date_mm_dd (MM_DD, blank to use all history)",
    )
    return (
        include_inf_immunity,
        include_vax_immunity,
        total_contact_input,
        school_contact_input,
        work_contact_input,
        mobility_input,
        daily_vaccines_input,
        daily_vaccines_vary_toggle,
        daily_vaccines_grid_input,
        daily_vaccines_per_subpop_toggle,
        daily_vaccines_subpop_selector,
        get_subpop_vax_values,
        set_subpop_vax_values,
        vax_transfer_delay_input,
        vaccinated_compartment_reset_date_input,
    )


@app.cell
def _epi_metric_ui(n_transitions, t_name, mo, loaded_config):
    _saved_params = loaded_config.get("params", {})
    _epi_cfgs = {_m["name"]: _m for _m in loaded_config.get("epi_metrics", [])}
    _M_cfg = _epi_cfgs.get("M", {}).get("update_config", {})
    transition_names = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    opts = transition_names if transition_names else [""]
    _rtos_saved = _M_cfg.get("r_to_s_transition", opts[-1])
    r_to_s_picker = mo.ui.dropdown(
        options=opts,
        value=_rtos_saved if _rtos_saved in opts else opts[-1],
        label="Transition used for R→S-style immunity update",
    )
    inf_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("inf_induced_saturation", 0.0)),
        label="inf_induced_saturation",
    )
    vax_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("vax_induced_saturation", 0.0)),
        label="vax_induced_saturation",
    )
    inf_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("inf_induced_immune_wane", 0.01)),
        label="inf_induced_immune_wane",
    )
    _vax_wane_raw = _saved_params.get("vax_induced_immune_wane", 0.0)
    vax_wane_is_array = isinstance(_vax_wane_raw, list)
    vax_wane_loaded_val = _vax_wane_raw
    vax_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=0.0 if vax_wane_is_array else float(_vax_wane_raw),
        label="vax_induced_immune_wane",
    )
    vax_delay_input = mo.ui.number(
        start=0, stop=60, step=1,
        value=int(_saved_params.get("vax_protection_delay_days", 0)),
        label="vax_protection_delay_days",
    )
    vax_reset_date_input = mo.ui.text(
        value=str(_saved_params.get("vax_immunity_reset_date_mm_dd", "")),
        placeholder="07_30",
        label="vax_immunity_reset_date_mm_dd (MM_DD, blank to disable)",
    )
    return (r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input,
            vax_wane_input, vax_wane_is_array, vax_wane_loaded_val,
            vax_delay_input, vax_reset_date_input)


@app.cell
def _schedule_csv_ui(
    mo, loaded_config, num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
    is_metapop, metapop_folder_input, Path,
):
    _inf = loaded_config.get("input_files", {})
    _multi = (num_age_groups > 1) or (num_risk_groups > 1)

    # Single shared folder holding every CSV below. The file fields are bare
    # filenames resolved against it (like the metapop folder). Legacy configs
    # that stored full paths leave this empty — the resolver passes them through.
    _folder_saved = _inf.get("input_folder", "")

    input_folder = mo.ui.text(
        value=_folder_saved,
        placeholder="/path/to/input_folder",
        label="Input folder (all CSV files below live here)",
        full_width=True,
    )

    # Humidity is CSV-only: a constant humidity modifier just scales beta by a
    # constant, which is a no-op. Auto-detect absolute_humidity.csv in the shared
    # folder (bare name), else the metapop folder (full path) for old layouts.
    _ah_csv_saved = _inf.get("absolute_humidity_csv", "")
    if not _ah_csv_saved:
        if _folder_saved and (Path(_folder_saved) / "absolute_humidity.csv").exists():
            _ah_csv_saved = "absolute_humidity.csv"
        elif is_metapop and metapop_folder_input.value.strip():
            _candidate = Path(metapop_folder_input.value.strip()) / "absolute_humidity.csv"
            if _candidate.exists():
                _ah_csv_saved = str(_candidate)

    ah_path = mo.ui.text(
        value=_ah_csv_saved,
        placeholder="absolute_humidity.csv",
        label="Absolute humidity CSV (filename, required for humidity modifier)",
        full_width=True,
    )
    cal_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("school_work_calendar_csv") else "constant",
        label="",
    )
    cal_path = mo.ui.text(
        value=_inf.get("school_work_calendar_csv", ""),
        placeholder="school_work_calendar.csv",
        label="School/work calendar CSV (filename)",
        full_width=True,
    )
    mob_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("mobility_csv") else ("csv" if _multi else "constant"),
        label="",
    )
    mob_path = mo.ui.text(
        value=_inf.get("mobility_csv", ""),
        placeholder="mobility_modifier.csv",
        label="Mobility CSV (filename)",
        full_width=True,
    )
    vax_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("vaccines_csv") else ("csv" if _multi else "constant"),
        label="",
    )
    vax_path = mo.ui.text(
        value=_inf.get("vaccines_csv", ""),
        placeholder="daily_vaccines.csv",
        label="Vaccines CSV (filename)",
        full_width=True,
    )
    total_contact_csv_path = mo.ui.text(
        value=_inf.get("total_contact_matrix_csv", ""),
        placeholder="total_contact_matrix.csv",
        label="Total contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    school_contact_csv_path = mo.ui.text(
        value=_inf.get("school_contact_matrix_csv", ""),
        placeholder="school_contact_matrix.csv",
        label="School contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    work_contact_csv_path = mo.ui.text(
        value=_inf.get("work_contact_matrix_csv", ""),
        placeholder="work_contact_matrix.csv",
        label="Work contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    return (
        input_folder,
        ah_path,
        cal_mode, cal_path,
        mob_mode, mob_path,
        vax_mode, vax_path,
        total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    )


@app.cell
def _schedule_csv_show(
    mo, main_tab,
    num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
    uses_scheduled_transfer,
    input_folder,
    ah_path,
    cal_mode, cal_path,
    mob_mode, mob_path,
    vax_mode, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    total_contact_input, school_contact_input, work_contact_input,
    mobility_input, daily_vaccines_input,
    daily_vaccines_vary_toggle, daily_vaccines_grid_input,
    daily_vaccines_per_subpop_toggle, daily_vaccines_subpop_selector,
    get_subpop_vax_values, set_subpop_vax_values,
    load_csv_validated, load_contact_matrix_csv, resolve_input_path,
    SimpleNamespace,
    fetched_contact_matrices, fetched_matrices_scope,
    loaded_config, is_array_param,
    is_metapop, pop_subpop_names,
    age_groups, param_grid_columns, array_to_grid_rows, grid_to_AR_array,
    tip_label, wtip,
    step_header, section_card, CLT_ACCENT,
):
    _ACC = CLT_ACCENT["builder"]
    _multi = (num_age_groups > 1) or (num_risk_groups > 1)
    _parts = []
    _parts.append(input_folder)

    # Absolute humidity — CSV-only (no constant option)
    _ah_df = None
    if uses_absolute_humidity:
        _parts.append(wtip(ah_path, "CSV columns: date, absolute_humidity"))
        if ah_path.value.strip():
            _ah_df, _ah_err = load_csv_validated(
                resolve_input_path(input_folder.value, ah_path.value),
                ["date", "absolute_humidity"],
            )
            if _ah_err:
                _parts.append(mo.callout(mo.md(f"**Humidity CSV:** {_ah_err}"), kind="danger"))
            else:
                _parts.append(mo.callout(
                    mo.md(f"Humidity CSV: {len(_ah_df)} rows loaded."), kind="success"
                ))
        else:
            _parts.append(mo.callout(
                mo.md("**Humidity modifier is on but no CSV is set.** "
                      "Provide an absolute-humidity CSV filename above."),
                kind="warn",
            ))

    # School/work calendar
    _cal_df = None
    if uses_contact_matrix:
        _parts.append(tip_label(
            "School/work calendar source",
            "constant: no calendar is used — the model always applies the full "
            "total contact matrix (school/work reductions are never applied).\n\n"
            "csv: vary contact patterns day-by-day using the is_school_day / "
            "is_work_day columns.",
        ))
        _parts.append(cal_mode)
        if cal_mode.value == "constant" and num_age_groups == 1:
            _parts.append(mo.hstack(
                [total_contact_input, school_contact_input, work_contact_input],
                wrap=True,
            ))
        if cal_mode.value == "csv":
            _parts.append(wtip(
                cal_path,
                "CSV columns: date, is_school_day, is_work_day "
                "(floats in [0, 1], fractional days allowed)",
            ))
            if cal_path.value.strip():
                _cal_df, _cal_err = load_csv_validated(
                    resolve_input_path(input_folder.value, cal_path.value),
                    ["date", "is_school_day", "is_work_day"],
                )
                if _cal_err:
                    _parts.append(mo.callout(mo.md(f"**Calendar CSV:** {_cal_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Calendar CSV: {len(_cal_df)} rows loaded."), kind="success"
                    ))

    # Mobility
    _mob_df = None
    if uses_mobility:
        _parts.append(tip_label(
            "Mobility source",
            "constant: the fixed Mobility modifier value entered here is "
            "applied every day.\n\n"
            "csv: vary the mobility modifier by day-of-week or date.",
        ))
        _parts.append(mob_mode)
        if mob_mode.value == "constant":
            _parts.append(mobility_input)
        if mob_mode.value == "csv":
            _parts.append(wtip(
                mob_path,
                "CSV with a day_of_week or date column, plus a mobility_modifier "
                "column holding a JSON A×R array per row, e.g.\n"
                '[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]',
            ))
            if mob_path.value.strip():
                _mob_df, _mob_err = load_csv_validated(
                    resolve_input_path(input_folder.value, mob_path.value), []
                )
                if _mob_err:
                    _parts.append(mo.callout(mo.md(f"**Mobility CSV:** {_mob_err}"), kind="danger"))
                else:
                    _has_col = "day_of_week" in _mob_df.columns or "date" in _mob_df.columns
                    if not _has_col:
                        _parts.append(mo.callout(
                            mo.md("**Mobility CSV:** Must have `day_of_week` or `date` column."),
                            kind="danger",
                        ))
                        _mob_df = None
                    else:
                        _parts.append(mo.callout(
                            mo.md(f"Mobility CSV: {len(_mob_df)} rows loaded."), kind="success"
                        ))
        elif _multi:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model (A={num_age_groups}, R={num_risk_groups}): "
                    "scalar mobility will broadcast to all groups."
                ),
                kind="warn",
            ))

    # Vaccines
    _vax_df = None
    if include_vax_immunity.value or uses_scheduled_transfer:
        _parts.append(tip_label(
            "Vaccines source",
            "constant: the value(s) entered here are applied every day.\n\n"
            "csv: vary doses by date (and by age/risk group).",
        ))
        _parts.append(vax_mode)
        if vax_mode.value == "constant":
            _vax_const_tip = (
                "Each value is the proportion of that age/risk group's "
                "population vaccinated per day (e.g. 0.001 = 0.1% of the "
                "group vaccinated that day) — not a raw dose count.\n\n"
                "Off: one value broadcasts to every age/risk group.\n"
                "Vary by age/risk group: enter a separate proportion per cell."
            )
            _parts.append(tip_label("Daily vaccines", _vax_const_tip))
            if is_metapop and len(pop_subpop_names) > 1:
                _parts.append(daily_vaccines_per_subpop_toggle)
            if is_metapop and len(pop_subpop_names) > 1 and daily_vaccines_per_subpop_toggle.value:
                _parts.append(daily_vaccines_subpop_selector)
                _sp = (
                    daily_vaccines_subpop_selector.value
                    if daily_vaccines_subpop_selector.value in pop_subpop_names
                    else pop_subpop_names[0]
                )
                _parts.append(daily_vaccines_vary_toggle)

                _subpop_vax_values = get_subpop_vax_values()
                _saved_sp_val = _subpop_vax_values.get(_sp)
                if daily_vaccines_vary_toggle.value:
                    _age_cols_vax = param_grid_columns(age_groups, num_age_groups)

                    def _on_subpop_vax_grid_change(_new_value, _sp=_sp, _age_cols_vax=_age_cols_vax):
                        _arr = grid_to_AR_array(_new_value, _age_cols_vax, num_age_groups, num_risk_groups)
                        set_subpop_vax_values({**get_subpop_vax_values(), _sp: _arr.tolist()})

                    _sp_grid = mo.ui.data_editor(
                        data=array_to_grid_rows(
                            _saved_sp_val if isinstance(_saved_sp_val, list) else None,
                            _age_cols_vax, num_risk_groups,
                        ),
                        label="",
                        editable_columns=list(_age_cols_vax),
                        on_change=_on_subpop_vax_grid_change,
                    )
                    _parts.append(_sp_grid)
                else:
                    def _on_subpop_vax_number_change(_new_value, _sp=_sp):
                        set_subpop_vax_values({**get_subpop_vax_values(), _sp: float(_new_value)})

                    _sp_number = mo.ui.number(
                        start=0.0, stop=1e9, step=1.0,
                        value=float(_saved_sp_val) if isinstance(_saved_sp_val, (int, float)) else 0.0,
                        label="",
                        on_change=_on_subpop_vax_number_change,
                    )
                    _parts.append(_sp_number)
            else:
                _parts.append(daily_vaccines_vary_toggle)
                if daily_vaccines_vary_toggle.value:
                    _parts.append(daily_vaccines_grid_input)
                else:
                    _parts.append(daily_vaccines_input)
        if vax_mode.value == "csv":
            _parts.append(wtip(
                vax_path,
                "CSV columns: date, daily_vaccines — each value is the "
                "proportion of that age×risk group vaccinated on that day "
                "(JSON A×R array per row), not a raw count.",
            ))
            if vax_path.value.strip():
                _vax_df, _vax_err = load_csv_validated(
                    resolve_input_path(input_folder.value, vax_path.value),
                    ["date", "daily_vaccines"],
                )
                if _vax_err:
                    _parts.append(mo.callout(mo.md(f"**Vaccines CSV:** {_vax_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Vaccines CSV: {len(_vax_df)} rows loaded."), kind="success"
                    ))
        elif _multi:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model (A={num_age_groups}, R={num_risk_groups}): "
                    "scalar vaccine count will broadcast to all groups."
                ),
                kind="warn",
            ))

    # Contact matrices (A > 1)
    _total_contact_mat = None
    _school_contact_mat = None
    _work_contact_mat = None
    if uses_contact_matrix and num_age_groups > 1:
        _fetched_shared = (
            fetched_contact_matrices.get("__shared__", {})
            if fetched_matrices_scope == "shared" else {}
        )
        _fetched_per_subpop = fetched_matrices_scope == "per_subpop" and bool(fetched_contact_matrices)
        _already_fetched = bool(_fetched_shared) or _fetched_per_subpop

        if _already_fetched:
            # Matrices already fetched in the Population & Geography tab take
            # precedence over anything entered here — don't duplicate the
            # inputs or show a misleading "not set" warning for them.
            _parts.append(mo.callout(
                mo.md(
                    "**Contact matrices are set** via the **Population & Geography** "
                    "tab and will be used. To change them, fetch a different "
                    "geography there (or clear the fetch and provide CSVs below "
                    "instead)."
                ),
                kind="success",
            ))
        else:
            _parts.append(mo.md("**Contact matrices (required when A > 1):**"))
            _contact_csv_tip = (
                "Plain floats, A×A, no header row, no index column. Used only "
                "if no contact matrix is fetched in the Population & Geography tab."
            )
            _no_csv_inline = []
            _no_csv_unset = []

            _parts.append(wtip(total_contact_csv_path, _contact_csv_tip))
            if total_contact_csv_path.value.strip():
                _total_contact_mat, _tc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, total_contact_csv_path.value),
                    num_age_groups,
                )
                if _tc_err:
                    _parts.append(mo.callout(mo.md(f"**Total contact matrix:** {_tc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Total contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "total_contact_matrix"):
                _no_csv_inline.append("total")
            else:
                _no_csv_unset.append("total")

            _parts.append(wtip(school_contact_csv_path, _contact_csv_tip))
            if school_contact_csv_path.value.strip():
                _school_contact_mat, _sc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, school_contact_csv_path.value),
                    num_age_groups,
                )
                if _sc_err:
                    _parts.append(mo.callout(mo.md(f"**School contact matrix:** {_sc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"School contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "school_contact_matrix"):
                _no_csv_inline.append("school")
            else:
                _no_csv_unset.append("school")

            _parts.append(wtip(work_contact_csv_path, _contact_csv_tip))
            if work_contact_csv_path.value.strip():
                _work_contact_mat, _wc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, work_contact_csv_path.value),
                    num_age_groups,
                )
                if _wc_err:
                    _parts.append(mo.callout(mo.md(f"**Work contact matrix:** {_wc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Work contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "work_contact_matrix"):
                _no_csv_inline.append("work")
            else:
                _no_csv_unset.append("work")

            if _no_csv_inline:
                _parts.append(mo.callout(
                    mo.md(
                        "No CSV set for **" + ", ".join(_no_csv_inline) + "** — using the "
                        "**inline config value(s)** from the loaded config."
                    ),
                    kind="success",
                ))
            if _no_csv_unset:
                _parts.append(mo.callout(
                    mo.md(
                        "**Not set: " + ", ".join(_no_csv_unset) + " contact matrix(es).** "
                        "No CSV is provided here, no inline value exists in the loaded "
                        "config, and (for total) none was fetched in the "
                        "**Population & Geography** tab — the model will fall back to "
                        "scalar `[[1.0]]` for these."
                    ),
                    kind="warn",
                ))

    if main_tab.value == "Model Builder":
        mo.output.append(section_card(
            step_header(4, "Schedules",
                        "Time-varying inputs: humidity, school/work calendar, "
                        "mobility, vaccines, and contact matrices.",
                        accent=_ACC),
            mo.vstack(_parts),
            accent=_ACC,
        ))

    loaded_schedule_dfs = SimpleNamespace(
        absolute_humidity_df=_ah_df,
        school_work_calendar_df=_cal_df,
        mobility_df=_mob_df,
        daily_vaccines_df=_vax_df,
        total_contact_matrix=_total_contact_mat,
        school_contact_matrix=_school_contact_mat,
        work_contact_matrix=_work_contact_mat,
    )
    return (loaded_schedule_dfs,)


@app.cell
def _schedule_and_immunity_show(
    mo,
    main_tab,
    include_inf_immunity,
    include_vax_immunity,
    vax_transfer_delay_input,
    vaccinated_compartment_reset_date_input,
    r_to_s_picker,
    inf_sat_input,
    vax_sat_input,
    inf_wane_input,
    vax_wane_input,
    vax_wane_is_array,
    vax_wane_loaded_val,
    vax_delay_input,
    vax_reset_date_input,
    uses_absolute_humidity,
    uses_contact_matrix,
    uses_mobility,
    requires_immunity_metrics,
    uses_scheduled_transfer,
    tip_label, wtip,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]

    _parts = [
        mo.hstack([
            wtip(
                include_inf_immunity,
                "Track population-level infection-induced immunity (M).\n\n"
                "For instance, if driven by (R→S) transitions:\n\n"
                "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                "M increases when recently-recovered individuals re-enter\n"
                "the susceptible pool (R→S), and decays via waning.\n\n"
                "Must be enabled for inf_reduce_param (Step 2) to have effect.",
            ),
            wtip(
                include_vax_immunity,
                "Track population-level vaccine-induced immunity (MV).\n\n"
                "MV grows with daily vaccine doses and decays via waning:\n\n"
                "ΔMV = daily_vaccines − wane×MV\n\n"
                "Must be enabled for vax_reduce_param (Step 2) to have effect.",
            ),
        ], wrap=True),
    ]

    if not uses_absolute_humidity and not uses_contact_matrix and not uses_mobility:
        _parts.append(mo.md("*No schedule-backed rate templates selected.*"))
    # The scalar values used when a schedule source is set to "constant"
    # (contact matrices for A=1, mobility, daily vaccines) are entered
    # directly below the matching source radio in Schedule File Inputs,
    # rather than here, so the input sits next to the control that reveals it.

    if uses_scheduled_transfer:
        _parts.append(mo.hstack([
            wtip(
                vax_transfer_delay_input,
                "Days between the scheduled date (e.g. vaccination date) and the\n"
                "date individuals actually move from origin to destination in a\n"
                "'scheduled_exact' transition.\n\n"
                "0 = transfer happens on the scheduled date itself.",
            ),
            wtip(
                vaccinated_compartment_reset_date_input,
                "If the schedule's CSV history starts before the simulation\n"
                "start date (e.g. a vaccine CSV starting months earlier than a\n"
                "fitted start date), doses between this reset date and the\n"
                "start date are replayed into the destination compartment's\n"
                "initial value (and out of the origin's), so pre-simulation\n"
                "vaccination history isn't lost.\n\n"
                "Blank = use all available history before the start date.\n"
                "Set to a date after last year's vaccination season (e.g.\n"
                "07_30) to exclude vaccinations from a previous year.",
            ),
        ], wrap=True))

    if requires_immunity_metrics:
        _parts.append(mo.callout(
            mo.md(
                "Selected rate templates can use `M` and/or `MV`. "
                "Enable whichever immunity metrics you want to track."
            ),
            kind="info",
        ))

    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    if _immunity_active:
        _metric_inputs = []
        if include_inf_immunity.value:
            _metric_inputs.extend([
                wtip(
                    r_to_s_picker,
                    "The transition that drives immunity gain.\n\n"
                    "M increases as people move from R back to S — recently-\n"
                    "recovered individuals re-entering the susceptible pool\n"
                    "still carry partial immunity.\n\n"
                    "Select the transition that represents this R→S flow.",
                ),
                wtip(
                    inf_sat_input,
                    "Limits how much M can grow as immunity accumulates.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → M saturates at a lower level.\n"
                    "0 = no saturation limit.",
                ),
                wtip(
                    vax_sat_input,
                    "How much vaccine immunity (MV) dampens further gain in M.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → MV reduces M accumulation more.\n"
                    "0 = vaccine and infection immunity are independent.",
                ),
                wtip(
                    inf_wane_input,
                    "Daily decay rate of infection-induced immunity M.\n\n"
                    "ΔM = (R→S / N) × (...) − wane×M\n\n"
                    "0 = no waning.\n"
                    "0.01 ≈ half-life of ~70 days.",
                ),
            ])
        if include_vax_immunity.value:
            if vax_wane_is_array:
                _metric_inputs.append(mo.callout(
                    mo.md(
                        "**`vax_induced_immune_wane`** — loaded from config as A×R array "
                        "(slider disabled, value passes through unchanged)\n\n"
                        f"```json\n{vax_wane_loaded_val}\n```"
                    ),
                    kind="info",
                ))
            else:
                _metric_inputs.append(
                    wtip(
                        vax_wane_input,
                        "Daily decay rate of vaccine-induced immunity MV.\n\n"
                        "ΔMV = daily_vaccines − wane×MV\n\n"
                        "0 = no waning.\n"
                        "0.01 ≈ half-life of ~70 days.",
                    )
                )
            _metric_inputs.extend([vax_delay_input, vax_reset_date_input])
        _parts.append(mo.hstack(_metric_inputs, wrap=True))
    else:
        _parts.append(mo.md("*Dynamic immunity metrics disabled.*"))

    section_card(
        step_header(4, "Immunity",
                    "Cumulative infection- and vaccine-induced immunity metrics "
                    "(M / MV) and their waning.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 5 — Model Diagram
# ---------------------------------------------------------------------------


@app.cell
def _diagram(
    compartments, n_transitions, t_name, t_origin, t_dest, t_template, t_infectious,
    parse_csv_list, mo, plt, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    _n = int(n_transitions.value)
    _inner = None
    _graphviz_error = None

    _foi_templates = ("force_of_infection", "force_of_infection_travel")
    _infectious_compartments: set[str] = set()
    _foi_links: list[tuple[str, int]] = []  # (infectious compartment, transition index)
    _foi_transition_idx: set[int] = set()
    for _i in range(_n):
        if t_template.value[_i] in _foi_templates and t_origin.value[_i] and t_dest.value[_i]:
            _foi_transition_idx.add(_i)
            for _c in parse_csv_list(t_infectious.value[_i]):
                _infectious_compartments.add(_c)
                _foi_links.append((_c, _i))

    try:
        import graphviz as gv  # type: ignore[import-untyped]
        _dot = gv.Digraph(
            graph_attr={"rankdir": "LR", "bgcolor": "white", "pad": "0.3"},
            node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#ddeeff"},
        )
        for _c in compartments:
            if _c in _infectious_compartments:
                _dot.node(_c, fillcolor="#ffa64d")
            else:
                _dot.node(_c)
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            _label = t_name.value[_i]
            if not (_origin and _dest):
                continue
            if _i in _foi_transition_idx:
                # Split the edge with an invisible point node so the dashed
                # "drives this transition" arrows can target the edge itself
                # rather than the destination compartment.
                _tnode = f"_t{_i}"
                _dot.node(_tnode, shape="point", width="0.01", label="")
                _dot.edge(_origin, _tnode, arrowhead="none", label=_label)
                _dot.edge(_tnode, _dest)
            else:
                _dot.edge(_origin, _dest, label=_label)
        for _c, _i in _foi_links:
            # constraint=false: keep this edge out of graphviz's rank/position
            # solver so it doesn't drag the point node off the straight line
            # of the transition edge it's attached to (causes a visible kink).
            _dot.edge(
                _c, f"_t{_i}", style="dashed", color="#ffa64d", arrowhead="empty",
                constraint="false",
            )
        _inner = mo.image(_dot.pipe(format="png"), width="100%")
    except Exception as _exc:
        _graphviz_error = f"{type(_exc).__name__}: {_exc}"

    if _inner is None:
        _fig, _ax = plt.subplots(figsize=(max(4, len(compartments) * 2), 2))
        _ax.set_xlim(-0.5, len(compartments) - 0.5)
        _ax.set_ylim(-0.5, 1.5)
        _ax.axis("off")
        _pos = {_c: (_i, 0.5) for _i, _c in enumerate(compartments)}
        for _c, (_x, _y) in _pos.items():
            _facecolor = "#ffa64d" if _c in _infectious_compartments else "#ddeeff"
            _ax.text(_x, _y, _c, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=_facecolor))
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            if _origin in _pos and _dest in _pos:
                _x0, _y0 = _pos[_origin]
                _x1, _y1 = _pos[_dest]
                _ax.annotate(
                    "", xy=(_x1 - 0.15, _y1), xytext=(_x0 + 0.15, _y0),
                    arrowprops=dict(arrowstyle="->", color="#336699"),
                )
        for _c, _i in _foi_links:
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            if _c in _pos and _origin in _pos and _dest in _pos:
                _x0, _y0 = _pos[_c]
                _xo, _yo = _pos[_origin]
                _xd, _yd = _pos[_dest]
                _xm, _ym = (_xo + _xd) / 2, (_yo + _yd) / 2  # midpoint of the transition edge
                _ax.annotate(
                    "", xy=(_xm, _ym + 0.2), xytext=(_x0 + 0.15, _y0 + 0.2),
                    arrowprops=dict(arrowstyle="->", color="#ffa64d", linestyle="dashed"),
                )
        plt.tight_layout()
        _fallback_parts = []
        if _graphviz_error is None:
            _fallback_parts.append(
                mo.callout(
                    mo.md("*Graphviz not available; using a simple fallback diagram.*"),
                    kind="info",
                )
            )
        else:
            _fallback_parts.append(
                mo.callout(
                    mo.md(
                        "**Graphviz rendering failed; using fallback diagram.**\n\n"
                        f"`{_graphviz_error}`"
                    ),
                    kind="warn",
                )
            )
        _fallback_parts.append(_fig)
        _inner = mo.vstack(_fallback_parts)

    section_card(
        step_header(5, "Model Diagram",
                    "Auto-generated compartment-flow diagram from your transitions.",
                    accent=_ACC),
        _inner,
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 6 — Initial Conditions
# ---------------------------------------------------------------------------


@app.cell
def _init_ui(mo, pop_subpop_names):
    # Edited grids are persisted here (keyed by "{subpop}::{compartment}")
    # rather than relying on the data_editor widgets' own internal state,
    # since those widgets are recreated each time the selected subpopulation
    # changes (see _init_show) and would otherwise reset to their
    # construction-time defaults, silently dropping edits made for a subpop
    # the user has switched away from.
    get_seed_values, set_seed_values = mo.state({})
    ic_subpop_selector = mo.ui.dropdown(
        options=list(pop_subpop_names),
        value=pop_subpop_names[0] if pop_subpop_names else None,
        label="Subpopulation",
    )
    return get_seed_values, set_seed_values, ic_subpop_selector


@app.cell
def _init_show(
    compartments, get_seed_values, set_seed_values, ic_subpop_selector,
    population_by_subpop, pop_subpop_names,
    num_age_groups, num_risk_groups, age_groups,
    param_grid_columns, grid_to_AR_array, default_seed_row_data,
    is_metapop, mo, main_tab, np, pd, loaded_config,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    _A = int(num_age_groups)
    _R = int(num_risk_groups)
    _seed_comps = compartments[1:] if len(compartments) > 1 else []
    _age_cols = param_grid_columns(age_groups, _A)
    _saved_ic = loaded_config.get("initial_conditions", {}) or {}

    _parts = [
        mo.md(
            "Seed each compartment by **age** and **risk** group (absolute counts). "
            "The first compartment (`"
            + (compartments[0] if compartments else "?")
            + "`) receives the remaining population in each cell. Population totals "
            "come from the **Population & Geography** tab."
        ),
    ]

    _sp = ic_subpop_selector.value if ic_subpop_selector.value in pop_subpop_names else (
        pop_subpop_names[0] if pop_subpop_names else "aggregate_pop"
    )
    if is_metapop:
        _parts.append(ic_subpop_selector)
        _parts.append(mo.callout(
            mo.md(
                "These per-subpopulation tables **override** "
                "`initial_conditions_{name}.json` in the metapop folder. A subpop "
                "whose tables are left all-zero falls back to its folder file."
            ),
            kind="info",
        ))

    _pop = population_by_subpop.get(_sp)
    if _pop is None:
        _pop = np.zeros((_A, _R))
    _pop = np.asarray(_pop, dtype=float)

    _seed_values = get_seed_values()

    def _make_on_change(_key):
        def _cb(_new_value):
            set_seed_values({**get_seed_values(), _key: _new_value})
        return _cb

    _seed_total = np.zeros((_A, _R))
    for _ci, _c in enumerate(_seed_comps):
        _key = f"{_sp}::{_c}"
        _data = _seed_values.get(_key)
        if _data is None:
            _data = default_seed_row_data(_saved_ic, _sp, _c, _age_cols, _R, _ci == 0)
        _ed = mo.ui.data_editor(
            data=_data,
            label=_c,
            editable_columns=list(_age_cols),
            on_change=_make_on_change(_key),
        )
        _parts.append(mo.md(f"**{_c}**"))
        _parts.append(_ed)
        _seed_total = _seed_total + grid_to_AR_array(_data, _age_cols, _A, _R)

    _remainder = _pop - _seed_total
    _first = compartments[0] if compartments else "?"
    _first_df = pd.DataFrame(
        [{"risk_group": _r,
          **{_col: _remainder[_a, _r] for _a, _col in enumerate(_age_cols)}}
         for _r in range(_R)]
    )
    _parts.append(mo.md(
        f"**{_first}** (auto = population − Σ seeds; total {max(_remainder.sum(), 0):,.0f})"
    ))
    _parts.append(mo.ui.table(_first_df, selection=None))
    if bool(np.any(_remainder < 0)):
        _parts.append(mo.callout(
            mo.md("Seeded counts exceed the population in at least one age/risk cell."),
            kind="danger",
        ))

    section_card(
        step_header(6, "Initial Conditions",
                    "Seed the compartments by age / risk group; the first "
                    "compartment absorbs the remaining population.",
                    accent=_ACC),
        mo.vstack(_parts),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 7 — Simulation Settings
# ---------------------------------------------------------------------------


@app.cell
def _sim_settings_ui(mo, loaded_config):
    _sim = loaded_config.get("simulation_settings", {})
    sim_days = mo.ui.number(start=10, stop=730, step=10, value=250, label="Simulation days")
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    n_reps = mo.ui.number(start=1, stop=100, step=1, value=10, label="Replicates")
    rng_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="RNG seed")
    timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    start_date_input = mo.ui.text(
        value=_sim.get("start_real_date", "2024-01-01"),
        label="Simulation start date (YYYY-MM-DD)",
    )
    transition_vars_input = mo.ui.text(
        value=", ".join(_sim.get("transition_variables_to_save", [])),
        placeholder="ISH_to_HR, ISH_to_HD, S_to_E  (blank = save all)",
        label="Transition variables to save",
        full_width=True,
    )
    return sim_days, sim_mode, n_reps, rng_seed, timesteps, start_date_input, transition_vars_input


@app.cell
def _sim_settings_show(
    mo, sim_days, sim_mode, n_reps, rng_seed, timesteps, start_date_input,
    transition_vars_input, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    section_card(
        step_header(7, "Simulation Settings",
                    "Horizon, deterministic vs. stochastic mode, RNG seed, and "
                    "which transition variables to record.",
                    accent=_ACC),
        mo.vstack([
            mo.hstack([sim_days, sim_mode, timesteps, rng_seed], justify="start"),
            mo.hstack([
                n_reps,
                mo.md("*Ignored in deterministic mode.*") if sim_mode.value == "Deterministic" else mo.md(""),
            ]),
            start_date_input,
            transition_vars_input,
            mo.callout(
                mo.md(
                    "Leaving **Transition variables to save** blank saves *every* "
                    "transition variable each day. For large models or many "
                    "replicates this can use a lot of memory and produce large "
                    "output files — list only the transitions you need (e.g. "
                    "`S_to_E, ISH_to_HR`)."
                ),
                kind="info",
            ) if not transition_vars_input.value.strip() else mo.md(""),
        ]),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Build config dict
# ---------------------------------------------------------------------------


@app.cell
def _build_config(
    compartments,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_schedule_name, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    param_grid_columns,
    include_inf_immunity, include_vax_immunity,
    r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input,
    vax_wane_input, vax_wane_is_array, vax_wane_loaded_val,
    vax_delay_input, vax_reset_date_input,
    vax_transfer_delay_input,
    vaccinated_compartment_reset_date_input,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics,
    uses_scheduled_transfer,
    rel_inf_param_name,
    parse_csv_list,
    total_contact_input, school_contact_input, work_contact_input,
    num_age_groups, num_risk_groups, age_groups,
    fetched_contact_matrices, fetched_matrices_scope,
    is_metapop, metapop_folder_input,
    loaded_schedule_dfs,
    input_folder,
    cal_mode, mob_mode, vax_mode,
    ah_path, cal_path, mob_path, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    get_seed_values, default_seed_row_data, population_by_subpop, pop_subpop_names,
    risk_fraction_inputs, grid_to_AR_array,
    daily_vaccines_per_subpop_toggle, get_subpop_vax_values,
    loaded_config,
    start_date_input, transition_vars_input,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    np,
):
    # Assembles the full model_config.json dict from every Step's widgets.
    # Roadmap of the sections below (search for the banner comments):
    #   1. PARAMS        — seed from loaded config, overlay scalar or A×R grid inputs
    #   2. TRANSITIONS   — per-template rate_config + self-loop warnings
    #   3. CONTACT MATRIX PARAMS
    #   4. SCHEDULES
    #   5. EPI METRICS   — infection-/vaccine-induced immunity
    #   6. INPUT FILES   — CSV references resolved against the shared folder
    #   7. INITIAL CONDITIONS — per-subpop population + seed grids
    #   8. CONFIG DICT   — final assembly
    #   9. ANALYSIS METRICS
    # Returns: (config_dict, immunity_active, metapop_travel_config, config_warnings)
    _n = int(n_transitions.value)
    _A = num_age_groups
    _R = num_risk_groups

    def _infectious_compartments_map(_i):
        return {
            _c: rel_inf_param_name(_c)
            for _c in parse_csv_list(t_infectious.value[_i])
        }

    # --- 1. PARAMS ---
    # Seed from loaded config first, then overlay each param's Step 3 widget:
    # a single float when constant, or an A×R nested list when toggled to vary.
    # The grid's data_editor is row-oriented by risk group with one column per
    # age group, so transpose its rows back into the A×R nested-list shape.
    params_dict: dict = dict(loaded_config.get("params", {}))
    _age_cols = param_grid_columns(age_groups, _A)
    for _name in param_names:
        if param_vary_toggles[_name].value:
            _rows = list(param_grid_inputs[_name].value)
            params_dict[_name] = [
                [float(_rows[_r][_age_cols[_a]]) for _r in range(_R)]
                for _a in range(_A)
            ]
        else:
            params_dict[_name] = float(param_scalar_inputs[_name].value)

    # --- 2. TRANSITIONS ---
    _transitions = []
    _metapop_travel_config = {}
    _config_warnings = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _rate_config = {"param": t_param.value[_i].strip()}
        elif _template == "param_product":
            _factors = parse_csv_list(t_factors.value[_i])
            _complements = parse_csv_list(t_complements.value[_i])
            _rate_config = {"factors": _factors}
            if _complements:
                _rate_config["complement_factors"] = _complements
        elif _template == "immunity_modulated":
            _rate_config = {
                "base_rate": t_base_rate.value[_i].strip(),
                "proportion": t_proportion.value[_i].strip(),
                "is_complement": bool(t_is_complement.value[_i]),
            }
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
        elif _template == "force_of_infection":
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "infectious_compartments": _infectious_compartments_map(_i),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
        elif _template == "scheduled_exact":
            _rate_config = {
                "schedule": t_schedule_name.value[_i].strip(),
                "compartment_reset_date_mm_dd_param": "vaccinated_compartment_reset_date_mm_dd",
            }
            if vaccinated_compartment_reset_date_input.value.strip():
                params_dict["vaccinated_compartment_reset_date_mm_dd"] = (
                    vaccinated_compartment_reset_date_input.value.strip()
                )
        else:
            _travel_config = {
                "infectious_compartments": _infectious_compartments_map(_i),
                "immobile_compartments": parse_csv_list(t_immobile.value[_i]),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "mobility_schedule": "mobility_modifier",
            }
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "travel_config": _travel_config,
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
            if not _metapop_travel_config:
                _metapop_travel_config = _travel_config

        _t_name = t_name.value[_i].strip()
        _t_origin = t_origin.value[_i]
        _t_dest = t_dest.value[_i]
        if _t_origin and _t_dest and _t_origin == _t_dest:
            _config_warnings.append(
                f"Transition '{_t_name or _i + 1}' has the same origin and "
                f"destination ('{_t_origin}') — this self-loop has no net effect."
            )
        _transitions.append({
            "name": _t_name,
            "origin": _t_origin,
            "destination": _t_dest,
            "rate_template": _template,
            "rate_config": _rate_config,
        })

    # --- 3. CONTACT MATRIX PARAMS ---
    if uses_contact_matrix:
        if _A == 1:
            params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
            params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
            params_dict["work_contact_matrix"] = [[float(work_contact_input.value)]]
        else:
            # A > 1: a proper A×A contact matrix is required. Prefer the CSV,
            # then an inline A×A list from the loaded config. Only fall back to a
            # scalar 1×1 matrix as a last resort, and warn loudly because that is
            # the wrong shape and will misbehave at run time.
            _shared_fetched_check = (
                fetched_contact_matrices.get("__shared__", {})
                if fetched_matrices_scope == "shared" else {}
            )
            _per_subpop_fetched = fetched_matrices_scope == "per_subpop" and fetched_contact_matrices
            for _label, _matrix_attr, _scalar_input in (
                ("total", "total_contact_matrix", total_contact_input),
                ("school", "school_contact_matrix", school_contact_input),
                ("work", "work_contact_matrix", work_contact_input),
            ):
                _loaded_mat = getattr(loaded_schedule_dfs, _matrix_attr)
                if _matrix_attr in _shared_fetched_check or _per_subpop_fetched:
                    pass  # fetched in Population & Geography tab — applied below
                elif _loaded_mat is not None:
                    params_dict[_matrix_attr] = _loaded_mat
                elif isinstance(params_dict.get(_matrix_attr), list) and \
                        len(params_dict[_matrix_attr]) == _A:
                    pass  # valid inline A×A matrix from loaded config — keep it
                else:
                    params_dict[_matrix_attr] = [[float(_scalar_input.value)]]
                    _config_warnings.append(
                        f"{_label.capitalize()} contact matrix: no {_A}×{_A} CSV or "
                        f"inline matrix provided for {_A} age groups; falling back to a "
                        f"scalar 1×1 matrix. Provide a {_A}×{_A} contact-matrix CSV in "
                        f"Step 4, or fetch matrices in the Population & Geography tab — "
                        f"the model will not behave correctly otherwise."
                    )

        # Contact matrices fetched in the Population & Geography tab take
        # precedence. Shared scope writes the three params here; per-subpop
        # scope is applied to subpop_params below (section 7).
        if fetched_matrices_scope == "shared":
            _shared_fetched = fetched_contact_matrices.get("__shared__", {})
            for _mname in ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"):
                if _mname in _shared_fetched:
                    params_dict[_mname] = _shared_fetched[_mname]

    # --- 4. SCHEDULES ---
    _schedules = []
    if uses_absolute_humidity:
        _schedules.append({
            "name": "absolute_humidity",
            "schedule_template": "timeseries_lookup",
            "schedule_config": {
                "df_attribute": "absolute_humidity_df",
                "value_column": "absolute_humidity",
            },
        })
    if uses_contact_matrix:
        _schedules.append({
            "name": "flu_contact_matrix",
            "schedule_template": "contact_matrix",
            "schedule_config": {
                "school_work_day_df_attribute": "school_work_calendar_df",
                "total_contact_matrix_param": "total_contact_matrix",
                "school_contact_matrix_param": "school_contact_matrix",
                "work_contact_matrix_param": "work_contact_matrix",
            },
        })
    if uses_mobility:
        _schedules.append({
            "name": "mobility_modifier",
            "schedule_template": "mobility",
            "schedule_config": {
                "df_attribute": "mobility_df",
            },
        })

    # --- 5. EPI METRICS ---
    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    _epi_metrics = []
    if include_vax_immunity.value:
        _schedules.append({
            "name": "daily_vaccines",
            "schedule_template": "vaccine_schedule",
            "schedule_config": {
                "df_attribute": "daily_vaccines_df",
            },
        })
    if uses_scheduled_transfer:
        _transfer_schedule_config = {"df_attribute": "daily_vaccines_df"}
        if int(vax_transfer_delay_input.value) > 0:
            params_dict["vax_transfer_delay_days"] = int(vax_transfer_delay_input.value)
            _transfer_schedule_config["vax_protection_delay_days_param"] = "vax_transfer_delay_days"
        _schedules.append({
            "name": "vaccinated_transfer_schedule",
            "schedule_template": "vaccine_schedule",
            "schedule_config": _transfer_schedule_config,
        })
    if include_inf_immunity.value:
        params_dict.update({
            "inf_induced_saturation": float(inf_sat_input.value),
            "vax_induced_saturation": float(vax_sat_input.value),
            "inf_induced_immune_wane": float(inf_wane_input.value),
        })
        _epi_metrics.append({
            "name": "M",
            "init_val": np.zeros((_A, _R)).tolist(),
            "metric_template": "infection_induced_immunity",
            "update_config": {
                "r_to_s_transition": r_to_s_picker.value,
                "inf_induced_saturation_param": "inf_induced_saturation",
                "vax_induced_saturation_param": "vax_induced_saturation",
                "inf_induced_immune_wane_param": "inf_induced_immune_wane",
            },
        })
    if include_vax_immunity.value:
        if not vax_wane_is_array:
            params_dict["vax_induced_immune_wane"] = float(vax_wane_input.value)
        # else: array already seeded from loaded_config above — vax_wane_loaded_val passes through
        if int(vax_delay_input.value) > 0:
            params_dict["vax_protection_delay_days"] = int(vax_delay_input.value)
        if vax_reset_date_input.value.strip():
            params_dict["vax_immunity_reset_date_mm_dd"] = vax_reset_date_input.value.strip()
        _epi_metrics.append({
            "name": "MV",
            "init_val": np.zeros((_A, _R)).tolist(),
            "metric_template": "vaccine_induced_immunity",
            "update_config": {
                "daily_vaccines_schedule": "daily_vaccines",
                "vax_induced_immune_wane_param": "vax_induced_immune_wane",
                "vax_protection_delay_days_param": "vax_protection_delay_days",
                "vax_immunity_reset_date_mm_dd_param": "vax_immunity_reset_date_mm_dd",
            },
        })

    # --- 6. INPUT FILES ---
    # The shared folder is recorded once and the CSV entries below are bare
    # filenames resolved against it. Humidity is CSV-only.
    _input_files = {}
    if input_folder.value.strip():
        _input_files["input_folder"] = input_folder.value.strip()
    if uses_absolute_humidity and ah_path.value.strip():
        _input_files["absolute_humidity_csv"] = ah_path.value.strip()
    if uses_contact_matrix and cal_mode.value == "csv" and cal_path.value.strip():
        _input_files["school_work_calendar_csv"] = cal_path.value.strip()
    if uses_mobility and mob_mode.value == "csv" and mob_path.value.strip():
        _input_files["mobility_csv"] = mob_path.value.strip()
    if (include_vax_immunity.value or uses_scheduled_transfer) and vax_mode.value == "csv" and vax_path.value.strip():
        _input_files["vaccines_csv"] = vax_path.value.strip()
    if uses_contact_matrix and _A > 1:
        if total_contact_csv_path.value.strip():
            _input_files["total_contact_matrix_csv"] = total_contact_csv_path.value.strip()
        if school_contact_csv_path.value.strip():
            _input_files["school_contact_matrix_csv"] = school_contact_csv_path.value.strip()
        if work_contact_csv_path.value.strip():
            _input_files["work_contact_matrix_csv"] = work_contact_csv_path.value.strip()
    if is_metapop and metapop_folder_input.value.strip():
        _input_files["metapop_folder"] = metapop_folder_input.value.strip()

    # --- 7. INITIAL CONDITIONS (per subpopulation) ---
    # Per-subpop population (A×R) plus per-compartment seed counts (A×R) from the
    # Step 6 tables. The first compartment is reconstructed at run time as
    # population − Σ seeds, so only the seeds are stored here.
    _age_cols_ic = param_grid_columns(age_groups, _A)
    _seed_comps = compartments[1:] if len(compartments) > 1 else []
    _saved_ic_cfg = loaded_config.get("initial_conditions", {}) or {}
    _seed_values = get_seed_values()
    _initial_conditions = {}
    for _sp in pop_subpop_names:
        _pop_arr = np.asarray(population_by_subpop.get(_sp, np.zeros((_A, _R))), dtype=float)
        _seeds = {}
        for _ci, _c in enumerate(_seed_comps):
            _data = _seed_values.get(f"{_sp}::{_c}")
            if _data is None:
                _data = default_seed_row_data(
                    _saved_ic_cfg, _sp, _c, _age_cols_ic, _R, _ci == 0
                )
            _arr = grid_to_AR_array(_data, _age_cols_ic, _A, _R)
            if bool(np.any(_arr != 0)):
                _seeds[_c] = _arr.tolist()
        _initial_conditions[_sp] = {"population": _pop_arr.tolist(), "seeds": _seeds}
    _total_pop = int(round(sum(
        np.asarray(_v["population"], dtype=float).sum()
        for _v in _initial_conditions.values()
    )))
    _risk_fractions = [float(_x) for _x in risk_fraction_inputs.value]

    # --- 8. CONFIG DICT ---
    _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    config_dict = {
        "compartments": compartments,
        "params": params_dict,
        "transitions": _transitions,
        "transition_groups": [],
        "epi_metrics": _epi_metrics,
        "schedules": _schedules,
        "age_risk": {
            "num_age_groups": _A,
            "num_risk_groups": _R,
            "risk_group_fractions": _risk_fractions,
            **({"age_groups": age_groups} if age_groups else {}),
        },
        "total_population": _total_pop,
        "initial_conditions": _initial_conditions,
        "simulation_settings": {
            "start_real_date": start_date_input.value.strip(),
            "transition_variables_to_save": _tvs,
        },
    }
    if _input_files:
        config_dict["input_files"] = _input_files

    # Per-subpopulation parameter overrides. Start from any loaded overrides
    # (authored directly in model_config.json) so the load -> rebuild -> run
    # round-trip stays lossless, then layer on per-subpop contact matrices
    # fetched in the Population & Geography tab. Both feed the metapop run path
    # and the shared factory, which read config_dict["subpop_params"].
    _subpop_params = dict(loaded_config.get("subpop_params", {}))
    if fetched_matrices_scope == "per_subpop":
        for _sp_name, _mats in fetched_contact_matrices.items():
            _entry = dict(_subpop_params.get(_sp_name, {}))
            for _mname in ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"):
                if _mname in _mats:
                    _entry[_mname] = _mats[_mname]
            _subpop_params[_sp_name] = _entry
    if _subpop_params:
        config_dict["subpop_params"] = _subpop_params

    # Per-subpopulation constant daily-vaccines override (metapop only). Kept
    # as its own top-level key rather than folded into subpop_params, since
    # daily_vaccines is a *schedule* value (read by _run_sim), not a model
    # "params" entry — subpop_params gets merged straight into params at run
    # time, which would silently misroute it.
    if is_metapop and daily_vaccines_per_subpop_toggle.value:
        _subpop_vax_values = get_subpop_vax_values()
        _subpop_daily_vaccines = {
            _sp: _subpop_vax_values.get(_sp, 0.0) for _sp in pop_subpop_names
        }
        config_dict["subpop_daily_vaccines"] = _subpop_daily_vaccines

    # --- 9. ANALYSIS METRICS ---
    _n_metrics = int(analysis_n_metrics_input.value)
    _analysis_metrics = []
    for _i in range(_n_metrics):
        _aname = analysis_metric_names.value[_i].strip() or f"metric_{_i + 1}"
        _raw = analysis_metric_tvs.value[_i]
        _atvs = _raw if isinstance(_raw, list) else [t.strip() for t in _raw.split(",") if t.strip()]
        if _atvs:
            _analysis_metrics.append({"name": _aname, "transition_variables": _atvs})
    if _analysis_metrics:
        config_dict["analysis_metrics"] = _analysis_metrics

    immunity_active = _immunity_active
    metapop_travel_config = _metapop_travel_config
    config_warnings = _config_warnings
    return config_dict, immunity_active, metapop_travel_config, config_warnings


# ---------------------------------------------------------------------------
# Step 8 — Config Preview
# ---------------------------------------------------------------------------


@app.cell
def _config_preview(
    config_dict, config_warnings, json, mo, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    json_str = json.dumps(config_dict, indent=2)
    _warn_block = []
    if config_warnings:
        _warn_block.append(mo.callout(
            mo.md(
                "**Config warnings:**\n\n"
                + "\n".join(f"- {_w}" for _w in config_warnings)
            ),
            kind="warn",
        ))
    section_card(
        step_header(8, "Config Preview",
                    "The assembled model config JSON — review or download it.",
                    accent=_ACC),
        mo.vstack([
            *_warn_block,
            mo.accordion({
                "View / download config JSON": mo.vstack([
                    mo.md(f"```json\n{json_str}\n```"),
                    mo.download(
                        data=json_str.encode(),
                        filename="model_config.json",
                        mimetype="application/json",
                        label="Download config JSON",
                    ),
                ])
            }),
        ]),
        accent=_ACC,
    )
    return


# ---------------------------------------------------------------------------
# Step 9 — Run
# ---------------------------------------------------------------------------


@app.cell
def _run_button(mo):
    run_button = mo.ui.run_button(label="Run simulation")
    return (run_button,)


@app.cell
def _run_section_display(
    run_button, mo, main_tab,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _ACC = CLT_ACCENT["builder"]
    section_card(
        step_header(9, "Run",
                    "Run the model and view trajectories below.", accent=_ACC),
        run_button,
        accent=_ACC,
    )
    return


@app.cell
def _run_sim(
    run_button,
    main_tab,
    config_dict,
    metapop_travel_config,
    compartments,
    sim_days,
    sim_mode,
    n_reps,
    rng_seed,
    timesteps,
    start_date_input,
    transition_vars_input,
    mobility_input,
    daily_vaccines_input,
    daily_vaccines_vary_toggle,
    daily_vaccines_grid_input,
    build_notebook_schedules_input,
    build_scalar_array,
    build_compartment_init,
    read_initial_conditions,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt,
    flu,
    np,
    mo,
    json,
    num_age_groups,
    num_risk_groups,
    is_metapop,
    metapop_folder_input,
    loaded_schedule_dfs,
    Path,
    pd,
    age_groups,
    param_grid_columns,
    grid_to_AR_array,
):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.stop(not run_button.value, mo.md(""))

    # Runs the preview simulation for Step 9. Structure of this cell:
    #   - run settings (stochastic/deterministic, reps, days, timesteps)
    #   - nested helper _build_schedules_input_for_subpop(...)
    #   - nested helper _run_once(...)         — single-population path
    #   - nested helper _run_metapop_once(...) — metapopulation path
    #   - dispatch on is_metapop, aggregate replicates, then plot + summary table
    _A = num_age_groups
    _R = num_risk_groups
    start_real_date = start_date_input.value.strip() or "2024-01-01"
    _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    _is_stochastic = sim_mode.value == "Stochastic"
    _transition_type = (
        clt.TransitionTypes.BINOM if _is_stochastic
        else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    )
    _reps = int(n_reps.value) if _is_stochastic else 1
    _num_days = int(sim_days.value)
    _seed = int(rng_seed.value)
    _ts_per_day = int(timesteps.value)

    # Daily-vaccines constant value: either a single scalar broadcast to every
    # age/risk group, or (when "Vary by age/risk group" is on) a per-cell A×R
    # array read back from the grid editor.
    if daily_vaccines_vary_toggle.value:
        _age_cols = param_grid_columns(age_groups, _A)
        _daily_vaccines_value = grid_to_AR_array(
            daily_vaccines_grid_input.value, _age_cols, _A, _R
        ).tolist()
    else:
        _daily_vaccines_value = float(daily_vaccines_input.value)

    # ---- Single-population run helper ----
    def _build_schedules_input_for_subpop(
        ah_df_override=None,
        cal_df_override=None,
        mob_df_override=None,
        vax_df_override=None,
        daily_vaccines_value_override=None,
    ):
        return build_notebook_schedules_input(
            start_date=start_real_date,
            num_days=_num_days,
            absolute_humidity=0.0,  # CSV-only: the humidity df is always supplied when used
            mobility_value=float(mobility_input.value),
            daily_vaccines_value=daily_vaccines_value_override if daily_vaccines_value_override is not None
                else _daily_vaccines_value,
            num_age_groups=_A,
            num_risk_groups=_R,
            absolute_humidity_df=ah_df_override if ah_df_override is not None
                else loaded_schedule_dfs.absolute_humidity_df,
            school_work_calendar_df=cal_df_override if cal_df_override is not None
                else loaded_schedule_dfs.school_work_calendar_df,
            mobility_df=mob_df_override if mob_df_override is not None
                else loaded_schedule_dfs.mobility_df,
            daily_vaccines_df=vax_df_override if vax_df_override is not None
                else loaded_schedule_dfs.daily_vaccines_df,
        )

    def _build_subpop(schedules_input, compartment_init, seed_offset, name="aggregate_pop", epi_metric_init=None, param_overrides=None):
        _config_err = None
        _model_config = None
        _cfg = config_dict
        if param_overrides:
            _cfg = dict(config_dict)
            _cfg["params"] = {**config_dict.get("params", {}), **param_overrides}
        try:
            _model_config = parse_model_config_from_dict(
                _cfg, schedules_input=schedules_input
            )
        except Exception as _exc:
            _config_err = str(_exc)
        if _config_err is not None:
            raise RuntimeError(f"Config error: {_config_err}")
        _state = build_state_from_config(_model_config, compartment_init, epi_metric_init=epi_metric_init or {})
        _params = build_params_from_config(_model_config, num_age_groups=_A, num_risk_groups=_R)
        _settings = clt.SimulationSettings(
            timesteps_per_day=_ts_per_day,
            transition_type=_transition_type,
            start_real_date=start_real_date,
            save_daily_history=True,
            transition_variables_to_save=_tvs,
        )
        _rng = np.random.default_rng(_seed + seed_offset)
        return ConfigDrivenSubpopModel(
            model_config=_model_config,
            state_init=_state,
            params=_params,
            simulation_settings=_settings,
            RNG=_rng,
            schedules_input=schedules_input,
            name=name,
        ), _model_config

    # ---- Pre-flight shape validation ----
    def _validate_shapes():
        """Return a list of human-readable issues for param/schedule shapes vs A×R."""
        _issues = []
        _contact_matrix_params = {"total_contact_matrix", "school_contact_matrix", "work_contact_matrix"}
        for _pname, _pval in config_dict.get("params", {}).items():
            if _pname in _contact_matrix_params:
                continue
            if isinstance(_pval, list):
                try:
                    _arr = np.array(_pval)
                    if _arr.ndim == 2 and (_arr.shape[0] != _A or _arr.shape[1] != _R):
                        _issues.append(
                            f"Param **`{_pname}`**: loaded shape {list(_arr.shape)} "
                            f"does not match A={_A}, R={_R}."
                        )
                except Exception:
                    pass
        for _sched_attr, _col, _label in [
            ("mobility_df", "mobility_modifier", "mobility_modifier"),
            ("daily_vaccines_df", "daily_vaccines", "daily_vaccines"),
        ]:
            _df = getattr(loaded_schedule_dfs, _sched_attr, None)
            if _df is not None and _col in _df.columns:
                try:
                    _arr = np.array(json.loads(_df[_col].iloc[0]))
                    if _arr.shape != (_A, _R):
                        _issues.append(
                            f"Schedule **`{_label}`** CSV: row array shape {list(_arr.shape)} "
                            f"does not match A={_A}, R={_R}."
                        )
                except Exception:
                    pass
        return _issues

    _shape_issues = _validate_shapes()
    mo.stop(
        bool(_shape_issues),
        mo.callout(
            mo.md(
                f"**Shape mismatch** — the following parameters/schedules are incompatible "
                f"with A={_A}, R={_R}. They are likely carried over from the loaded config. "
                f"Switch the affected schedule source to **constant** in Step 4, "
                f"or reload a config that matches the current group counts.\n\n"
                + "\n".join(f"- {_issue}" for _issue in _shape_issues)
            ),
            kind="danger",
        ),
    )

    # ---- Single-population path ----
    if not is_metapop:
        # Initial conditions come from the Step 6 tables via config_dict
        # (population A×R per cell, minus the per-compartment seed grids).
        _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
        _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((_A, _R))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in compartments
        }
        compartment_init, _overflow = build_compartment_init(
            _seed_arrays, _pop_arr, compartments)
        mo.stop(
            _overflow,
            mo.callout(mo.md("**Initial condition error:** seeded counts exceed the "
                             "population in at least one age/risk cell."),
                       kind="danger"),
        )

        def _run_once(seed_offset):
            _sched = _build_schedules_input_for_subpop()
            _subpop, _model_config = _build_subpop(_sched, compartment_init, seed_offset)
            _mixing = flu.FluMixingParams(
                travel_proportions=np.array([[1.0]]),
                num_locations=1,
            )
            _metapop = ConfigDrivenMetapopModel(
                subpop_models=[_subpop],
                mixing_params=_mixing,
                model_config=_model_config,
                travel_config=metapop_travel_config,
            )
            _metapop.simulate_until_day(_num_days)
            return {
                _c: np.array(_subpop.compartments[_c].history_vals_list).sum(axis=(1, 2))
                for _c in compartments
            }

        # Parse config once for error checking before running
        _config_parse_err = None
        try:
            _test_sched = _build_schedules_input_for_subpop()
            parse_model_config_from_dict(config_dict, schedules_input=_test_sched)
        except Exception as _exc:
            _config_parse_err = str(_exc)
        mo.stop(
            _config_parse_err is not None,
            mo.callout(mo.md(f"**Config error:** {_config_parse_err}"), kind="danger"),
        )

        sim_err = None
        histories = []
        with mo.status.spinner("Running simulation..."):
            try:
                histories = [_run_once(_rep) for _rep in range(_reps)]
            except Exception as _exc:
                sim_err = str(_exc)
        mo.stop(
            sim_err is not None,
            mo.callout(mo.md(f"**Simulation error:** {sim_err}"), kind="danger"),
        )

    # ---- Metapopulation path ----
    else:
        _folder = Path(metapop_folder_input.value.strip())
        mo.stop(
            not _folder.exists() or not _folder.is_dir(),
            mo.callout(mo.md(f"**Metapop folder not found:** {_folder}"), kind="danger"),
        )
        _metapop_cfg_path = _folder / "metapop_config.json"
        mo.stop(
            not _metapop_cfg_path.exists(),
            mo.callout(mo.md("**Missing:** `metapop_config.json` in metapop folder."), kind="danger"),
        )
        with open(_metapop_cfg_path) as _f:
            _metapop_cfg = json.load(_f)
        mo.stop(
            "subpopulations" not in _metapop_cfg or "travel_matrix" not in _metapop_cfg,
            mo.callout(mo.md("**Invalid `metapop_config.json`:** must have `subpopulations` and `travel_matrix` keys."), kind="danger"),
        )
        _sp_names = list(_metapop_cfg["subpopulations"])
        _travel_arr = np.array(_metapop_cfg["travel_matrix"], dtype=float)
        _n_subpops = len(_sp_names)

        # Shared optional schedule files
        _shared_ah_df = None
        _shared_mob_df = None
        _ah_shared_path = _folder / "absolute_humidity.csv"
        _mob_shared_path = _folder / "mobility_modifier.csv"
        if _ah_shared_path.exists():
            _shared_ah_df = pd.read_csv(_ah_shared_path)
            _shared_ah_df = _shared_ah_df.loc[:, ~_shared_ah_df.columns.str.match(r"^Unnamed")]
        if _mob_shared_path.exists():
            _shared_mob_df = pd.read_csv(_mob_shared_path)
            _shared_mob_df = _shared_mob_df.loc[:, ~_shared_mob_df.columns.str.match(r"^Unnamed")]

        def _run_metapop_once(seed_offset):
            _subpop_models = []
            _model_config_ref = None
            for _sp_idx, _sp_name in enumerate(_sp_names):
                # Load per-subpop schedule files
                _sp_cal_path = _folder / f"school_work_calendar_{_sp_name}.csv"
                _sp_vax_path = _folder / f"vaccines_{_sp_name}.csv"
                _sp_ic_path  = _folder / f"initial_conditions_{_sp_name}.json"

                _sp_cal_df = None
                _sp_vax_df = None
                if _sp_cal_path.exists():
                    _sp_cal_df = pd.read_csv(_sp_cal_path)
                    _sp_cal_df = _sp_cal_df.loc[:, ~_sp_cal_df.columns.str.match(r"^Unnamed")]
                if _sp_vax_path.exists():
                    _sp_vax_df = pd.read_csv(_sp_vax_path)
                    _sp_vax_df = _sp_vax_df.loc[:, ~_sp_vax_df.columns.str.match(r"^Unnamed")]

                _sched = _build_schedules_input_for_subpop(
                    ah_df_override=_shared_ah_df,
                    cal_df_override=_sp_cal_df,
                    mob_df_override=_shared_mob_df,
                    vax_df_override=_sp_vax_df,
                    daily_vaccines_value_override=config_dict.get(
                        "subpop_daily_vaccines", {}
                    ).get(_sp_name),
                )

                # Initial conditions: the in-notebook Step 6 tables take precedence
                # when they carry seeds, else the per-subpop folder JSON, else the
                # table's population-only (all-susceptible) state.
                _sp_epi_init = {}
                _ic_cfg = (config_dict.get("initial_conditions", {}) or {}).get(_sp_name, {})
                _has_table_seeds = bool(_ic_cfg.get("seeds"))
                _table_comp_init = read_initial_conditions(
                    config_dict, _sp_name, compartments, _A, _R)
                if _has_table_seeds and _table_comp_init is not None:
                    _sp_comp_init = _table_comp_init
                elif _sp_ic_path.exists():
                    _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                    with open(_sp_ic_path) as _f:
                        _ic = json.load(_f)
                    for _c, _arr in _ic.get("compartments", {}).items():
                        if _c in compartments:
                            _sp_comp_init[_c] = np.array(_arr, dtype=float)
                    for _m, _arr in _ic.get("epi_metrics", {}).items():
                        _sp_epi_init[_m] = np.array(_arr, dtype=float)
                elif _table_comp_init is not None:
                    _sp_comp_init = _table_comp_init
                else:
                    _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                    mo.stop(
                        _A > 1 or _R > 1,
                        mo.callout(mo.md(f"**Missing:** initial conditions for `{_sp_name}` — "
                                         f"seed it in Step 6 or provide "
                                         f"`initial_conditions_{_sp_name}.json`."), kind="danger"),
                    )

                _sp_param_overrides = dict(config_dict.get("subpop_params", {}).get(_sp_name, {}))
                _subpop, _mc = _build_subpop(
                    _sched, _sp_comp_init, seed_offset + _sp_idx, name=_sp_name,
                    epi_metric_init=_sp_epi_init,
                    param_overrides=_sp_param_overrides or None,
                )
                _subpop_models.append(_subpop)
                if _model_config_ref is None:
                    _model_config_ref = _mc

            _mixing = flu.FluMixingParams(
                travel_proportions=_travel_arr,
                num_locations=_n_subpops,
            )
            _metapop = ConfigDrivenMetapopModel(
                subpop_models=_subpop_models,
                mixing_params=_mixing,
                model_config=_model_config_ref,
                travel_config=metapop_travel_config,
            )
            _metapop.simulate_until_day(_num_days)

            # Aggregate histories by summing across subpops and age/risk groups
            return {
                _c: sum(
                    np.array(_sp.compartments[_c].history_vals_list).sum(axis=(1, 2))
                    for _sp in _subpop_models
                )
                for _c in compartments
            }

        sim_err = None
        histories = []
        with mo.status.spinner("Running metapopulation simulation..."):
            try:
                histories = [_run_metapop_once(_rep * _n_subpops) for _rep in range(_reps)]
            except Exception as _exc:
                sim_err = str(_exc)
        mo.stop(
            sim_err is not None,
            mo.callout(mo.md(f"**Simulation error:** {sim_err}"), kind="danger"),
        )

    return (histories,)


@app.cell
def _plot_curves(histories, compartments, sim_days, sim_mode, is_metapop, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _num_days = int(sim_days.value)
    _days = np.arange(1, _num_days + 1)
    _is_stochastic = sim_mode.value == "Stochastic"

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for _ci, _comp in enumerate(compartments):
        _color = _colors[_ci % len(_colors)]
        if _is_stochastic and len(histories) > 1:
            _mat = np.stack([_h[_comp] for _h in histories], axis=0)
            _median = np.median(_mat, axis=0)
            _lo = np.percentile(_mat, 2.5, axis=0)
            _hi = np.percentile(_mat, 97.5, axis=0)
            for _rep in range(len(histories)):
                _ax.plot(_days, _mat[_rep], color=_color, alpha=0.15, linewidth=0.8)
            _ax.plot(_days, _median, color=_color, linewidth=2, label=f"{_comp} (median)")
            _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
        else:
            _ax.plot(_days, histories[0][_comp], color=_color, linewidth=2, label=_comp)

    _ax.set_xlabel("Day")
    _ax.set_ylabel("Count")
    _title = "Epidemic Curves"
    if is_metapop:
        _title += " (aggregated across subpopulations)"
    _ax.set_title(_title)
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.vstack([mo.md("### Results — Epidemic Curves"), _fig])
    return


@app.cell
def _summary_stats(histories, compartments, np, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _rows = []
    for _comp in compartments:
        _vals = np.stack([_h[_comp] for _h in histories], axis=0)
        _peak = np.median(np.max(_vals, axis=1))
        _peak_day = int(np.median(np.argmax(_vals, axis=1))) + 1
        _final = np.median(_vals[:, -1])
        _rows.append(f"| `{_comp}` | {_peak:,.0f} | {_peak_day} | {_final:,.0f} |")
    _table = "\n".join(_rows)
    mo.vstack([
        mo.md("### Results — Summary"),
        mo.md(
            "| Compartment | Peak value (median) | Peak day (median) | Final value (median) |\n"
            "|---|---|---|---|\n"
            f"{_table}"
        ),
    ])
    return

@app.cell
def _shared_model_factory(
    build_notebook_schedules_input,
    build_scalar_array,
    read_initial_conditions,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt, flu, np, json, pd, Path,
    loaded_schedule_dfs,
    mobility_input,
    daily_vaccines_input,
    num_age_groups,
    num_risk_groups,
):
    _A = num_age_groups
    _R = num_risk_groups

    def _sched_builder(start_date, num_days, ah_df=None, cal_df=None, mob_df=None, vax_df=None):
        return build_notebook_schedules_input(
            start_date=start_date, num_days=num_days,
            absolute_humidity=0.0,  # CSV-only: the humidity df is always supplied when used
            mobility_value=float(mobility_input.value),
            daily_vaccines_value=float(daily_vaccines_input.value),
            num_age_groups=_A, num_risk_groups=_R,
            absolute_humidity_df=ah_df if ah_df is not None else loaded_schedule_dfs.absolute_humidity_df,
            school_work_calendar_df=cal_df if cal_df is not None else loaded_schedule_dfs.school_work_calendar_df,
            mobility_df=mob_df if mob_df is not None else loaded_schedule_dfs.mobility_df,
            daily_vaccines_df=vax_df if vax_df is not None else loaded_schedule_dfs.daily_vaccines_df,
        )

    def make_single_pop_metapop(
        config, start_date, num_days, compartment_init,
        seed_offset=0, seed_base=0, ts_per_day=7, stochastic=False,
        tvs=None, save_daily=True, epi_metric_init=None, param_overrides=None,
        travel_config=None,
    ):
        _cfg = config
        if param_overrides:
            _cfg = dict(config)
            _cfg["params"] = {**config.get("params", {}), **param_overrides}
        _sched = _sched_builder(start_date, num_days)
        _mc = parse_model_config_from_dict(_cfg, schedules_input=_sched)
        _state = build_state_from_config(_mc, compartment_init, epi_metric_init=epi_metric_init or {})
        _params = build_params_from_config(_mc, num_age_groups=_A, num_risk_groups=_R)
        _tt = clt.TransitionTypes.BINOM if stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
        _settings = clt.SimulationSettings(
            timesteps_per_day=ts_per_day, transition_type=_tt,
            start_real_date=start_date, save_daily_history=save_daily,
            transition_variables_to_save=tvs or [],
        )
        _subpop = ConfigDrivenSubpopModel(
            model_config=_mc, state_init=_state, params=_params,
            simulation_settings=_settings,
            RNG=np.random.default_rng(seed_base + seed_offset),
            schedules_input=_sched, name="pop",
        )
        _mixing = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
        return ConfigDrivenMetapopModel(
            subpop_models=[_subpop], mixing_params=_mixing,
            model_config=_mc, travel_config=travel_config or {},
        ), _mc, _sched

    def make_metapop_from_folder(
        folder_path, config, start_date, num_days, compartments_list,
        seed_offset=0, seed_base=0, ts_per_day=7, stochastic=False,
        tvs=None, save_daily=True, param_overrides=None, travel_config=None,
        param_overrides_per_subpop=None, init_states_override=None,
    ):
        _folder = Path(folder_path)
        with open(_folder / "metapop_config.json") as _f:
            _mc_cfg = json.load(_f)
        _sp_names = list(_mc_cfg["subpopulations"])
        _travel_arr = np.array(_mc_cfg["travel_matrix"], dtype=float)
        _shared_ah = _shared_mob = None
        _ah_p = _folder / "absolute_humidity.csv"
        _mob_p = _folder / "mobility_modifier.csv"
        if _ah_p.exists():
            _shared_ah = pd.read_csv(_ah_p)
            _shared_ah = _shared_ah.loc[:, ~_shared_ah.columns.str.match(r"^Unnamed")]
        if _mob_p.exists():
            _shared_mob = pd.read_csv(_mob_p)
            _shared_mob = _shared_mob.loc[:, ~_shared_mob.columns.str.match(r"^Unnamed")]
        _tt = clt.TransitionTypes.BINOM if stochastic else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
        _subpops = []
        _mc_ref = None
        for _si, _sp_name in enumerate(_sp_names):
            _cal_df = _vax_df = None
            _cal_p = _folder / f"school_work_calendar_{_sp_name}.csv"
            _vax_p = _folder / f"vaccines_{_sp_name}.csv"
            _ic_p = _folder / f"initial_conditions_{_sp_name}.json"
            if _cal_p.exists():
                _cal_df = pd.read_csv(_cal_p)
                _cal_df = _cal_df.loc[:, ~_cal_df.columns.str.match(r"^Unnamed")]
            if _vax_p.exists():
                _vax_df = pd.read_csv(_vax_p)
                _vax_df = _vax_df.loc[:, ~_vax_df.columns.str.match(r"^Unnamed")]
            _sched = _sched_builder(
                start_date, num_days,
                ah_df=_shared_ah, cal_df=_cal_df, mob_df=_shared_mob, vax_df=_vax_df,
            )
            _comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments_list}
            _epi_init = {}
            # Precedence: explicit override > Step 6 tables (with seeds) > folder
            # JSON > population-only table > zeros.
            _ic_cfg = (config.get("initial_conditions", {}) or {}).get(_sp_name, {})
            _has_table_seeds = bool(_ic_cfg.get("seeds"))
            _table_ci = read_initial_conditions(config, _sp_name, compartments_list, _A, _R)
            if init_states_override is not None and _si < len(init_states_override):
                _comp_init, _epi_init = init_states_override[_si]
            elif _has_table_seeds and _table_ci is not None:
                _comp_init = _table_ci
            elif _ic_p.exists():
                with open(_ic_p) as _f:
                    _ic = json.load(_f)
                for _c, _arr in _ic.get("compartments", {}).items():
                    if _c in compartments_list:
                        _comp_init[_c] = np.array(_arr, dtype=float)
                for _m, _arr in _ic.get("epi_metrics", {}).items():
                    _epi_init[_m] = np.array(_arr, dtype=float)
            elif _table_ci is not None:
                _comp_init = _table_ci
            _sp_overrides = dict(param_overrides or {})
            if param_overrides_per_subpop and _si < len(param_overrides_per_subpop) and param_overrides_per_subpop[_si]:
                _sp_overrides.update(param_overrides_per_subpop[_si])
            _sp_overrides.update(config.get("subpop_params", {}).get(_sp_name, {}))
            _cfg = config
            if _sp_overrides:
                _cfg = dict(config)
                _cfg["params"] = {**config.get("params", {}), **_sp_overrides}
            _mc_parsed = parse_model_config_from_dict(_cfg, schedules_input=_sched)
            _state = build_state_from_config(_mc_parsed, _comp_init, epi_metric_init=_epi_init)
            _params = build_params_from_config(_mc_parsed, num_age_groups=_A, num_risk_groups=_R)
            _settings = clt.SimulationSettings(
                timesteps_per_day=ts_per_day, transition_type=_tt,
                start_real_date=start_date, save_daily_history=save_daily,
                transition_variables_to_save=tvs or [],
            )
            _subpop = ConfigDrivenSubpopModel(
                model_config=_mc_parsed, state_init=_state, params=_params,
                simulation_settings=_settings,
                RNG=np.random.default_rng(seed_base + seed_offset + _si),
                schedules_input=_sched, name=_sp_name,
            )
            _subpops.append(_subpop)
            if _mc_ref is None:
                _mc_ref = _mc_parsed
        _mixing = flu.FluMixingParams(travel_proportions=_travel_arr, num_locations=len(_sp_names))
        _kwargs = {}
        if travel_config:
            _kwargs["travel_config"] = travel_config
        return ConfigDrivenMetapopModel(
            subpop_models=_subpops, mixing_params=_mixing, model_config=_mc_ref, **_kwargs
        ), _mc_ref

    def extract_history(metapop, comps, tvs=None):
        _sps = list(metapop.subpop_models.values())
        _out = {}
        for _c in comps:
            _out[_c] = sum(
                np.array(_sp.compartments[_c].history_vals_list).sum(axis=(1, 2))
                for _sp in _sps
            )
        for _tv in (tvs or []):
            _candidates = [_sp for _sp in _sps if _tv in _sp.transition_variables]
            if _candidates:
                _out[_tv] = sum(
                    np.array(_sp.transition_variables[_tv].history_vals_list).sum(axis=(1, 2))
                    for _sp in _candidates
                )
        return _out

    return (make_single_pop_metapop, make_metapop_from_folder, extract_history)


# ============================================================
# Fitting tab
# ============================================================

@app.cell
def _fit_n_targets_state(mo):
    get_n_targets, set_n_targets = mo.state(1)
    return get_n_targets, set_n_targets


@app.cell
def _fit_target_buttons(mo, get_n_targets, set_n_targets):
    add_target_btn = mo.ui.button(
        label="+ Add target",
        on_click=lambda _: set_n_targets(min(get_n_targets() + 1, 20)),
    )
    del_target_btn = mo.ui.button(
        label="− Remove target",
        on_click=lambda _: set_n_targets(max(get_n_targets() - 1, 1)),
    )
    return add_target_btn, del_target_btn


@app.cell
def _fitting_ui(
    mo, compartments, n_transitions, t_name, param_names,
    is_metapop, metapop_folder_input, json, Path,
    num_age_groups, num_risk_groups,
):
    _tvars = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    _all_tgts = list(compartments) + _tvars
    _tgt_opts = _all_tgts if _all_tgts else ["S"]

    # Subpop names for slice dropdowns
    _subpop_names_ui = []
    if is_metapop and metapop_folder_input.value.strip():
        try:
            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                _subpop_names_ui = json.load(_f).get("subpopulations", [])
        except Exception:
            _subpop_names_ui = []

    _sp_opts = ["All (sum)"] + [f"{_i}: {_nm}" for _i, _nm in enumerate(_subpop_names_ui)]
    _age_opts = ["All (sum)"] + [str(_i) for _i in range(int(num_age_groups))]
    _risk_opts = ["All (sum)"] + [str(_i) for _i in range(int(num_risk_groups))]

    fit_target_src = mo.ui.array([
        mo.ui.radio(
            options={"Upload CSV": "upload", "File path": "path"},
            value="Upload CSV",
            label="Data source",
        )
        for _ in range(20)
    ])
    fit_target_upload = mo.ui.array([
        mo.ui.file(label="Upload CSV", filetypes=[".csv"])
        for _ in range(20)
    ])
    fit_target_path = mo.ui.array([
        mo.ui.text(
            label="CSV file path", placeholder="~/data/observed.csv", full_width=True,
        )
        for _ in range(20)
    ])
    fit_target_vars = mo.ui.array([
        mo.ui.multiselect(
            options={t: t for t in _tgt_opts},
            value=[_tgt_opts[0]],
            label="Variables (summed)",
        )
        for _ in range(20)
    ])
    fit_target_mode = mo.ui.array([
        mo.ui.radio(
            options={"Timeseries": "ts", "Scalar total": "scalar", "Proportions": "proportion"},
            value="Timeseries",
            label="Observed data type",
        )
        for _ in range(20)
    ])
    fit_target_weight = mo.ui.array([
        mo.ui.number(value=1.0, start=0.0, stop=1000.0, step=None, label="Weight λ")
        for _ in range(20)
    ])
    fit_target_subpop = mo.ui.array([
        mo.ui.dropdown(options=_sp_opts, value=_sp_opts[0], label="Subpopulation")
        for _ in range(20)
    ])
    fit_target_age = mo.ui.array([
        mo.ui.dropdown(options=_age_opts, value=_age_opts[0], label="Age group")
        for _ in range(20)
    ])
    fit_target_risk = mo.ui.array([
        mo.ui.dropdown(options=_risk_opts, value=_risk_opts[0], label="Risk group")
        for _ in range(20)
    ])

    fit_sim_days_input = mo.ui.number(
        value=180, start=1, stop=3650, step=1,
        label="Simulation days (used when all targets are scalar totals)",
    )
    _seed_scale_opts = {
        f"seed_scale_{_c}": f"seed_scale_{_c}"
        for _c in compartments[1:]
    }
    fit_params_multiselect = mo.ui.multiselect(
        options={**{p: p for p in param_names}, **_seed_scale_opts},
        value=[],
        label="Parameters to fit",
    )
    fit_method = mo.ui.radio(
        options={"Adam (gradient)": "adam", "L-BFGS (gradient)": "lbfgs", "Accept-reject": "ar"},
        value="Adam (gradient)", label="Fitting method",
    )
    fit_n_iter = mo.ui.number(value=200, start=10, stop=2000, step=10, label="Iterations / Max samples")
    fit_r2_thresh = mo.ui.number(value=0.75, start=0.0, stop=1.0, step=None, label="R² acceptance threshold")
    fit_run_button = mo.ui.run_button(label="Run fitting")

    return (
        fit_target_src, fit_target_upload, fit_target_path,
        fit_target_vars, fit_target_mode, fit_target_weight,
        fit_target_subpop, fit_target_age, fit_target_risk,
        fit_sim_days_input,
        fit_params_multiselect,
        fit_method, fit_n_iter, fit_r2_thresh, fit_run_button,
    )


@app.cell
def _fitting_lr_ui(mo, fit_method):
    _lr_default = 0.5 if fit_method.value == "lbfgs" else 0.01
    fit_lr = mo.ui.number(value=_lr_default, start=1e-5, stop=10.0, step=None, label="Learning rate")
    return (fit_lr,)


@app.cell
def _fitting_replications_ui(mo):
    fit_n_replications = mo.ui.number(
        value=5, start=1, stop=200, step=1,
        label="Number of replications",
    )
    return (fit_n_replications,)


@app.cell
def _fitting_bounds_ui(
    mo, fit_params_multiselect, config_dict,
    num_age_groups, num_risk_groups, is_metapop,
):
    _saved_params = config_dict.get("params", {})
    _selected = list(fit_params_multiselect.value)
    _A = num_age_groups
    _R = num_risk_groups

    _dim_opts = []
    if _A > 1:
        _dim_opts.append("age groups")
    if _R > 1:
        _dim_opts.append("risk groups")
    if is_metapop:
        _dim_opts.append("subpopulation")

    def _default_bounds(pn):
        if pn.startswith("seed_scale_"):
            return 0.1, 10.0
        _raw = _saved_params.get(pn, 0.1)
        _dv = float(_raw) if not isinstance(_raw, list) else 0.1
        _lo = round(0.5 * _dv, 8)
        _hi = round(2.0 * _dv, 8)
        if _lo == _hi:
            _lo = max(1e-8, _dv * 0.1)
            _hi = _dv * 5.0
        return _lo, _hi

    fit_bounds_lo = mo.ui.array([
        mo.ui.number(
            start=1e-8, stop=1e8, step=None,
            value=_default_bounds(_pn)[0],
            label="Lower bound",
        )
        for _pn in _selected
    ])
    fit_bounds_hi = mo.ui.array([
        mo.ui.number(
            start=1e-8, stop=1e8, step=None,
            value=_default_bounds(_pn)[1],
            label="Upper bound",
        )
        for _pn in _selected
    ])
    fit_param_dims = mo.ui.array([
        mo.ui.multiselect(
            options=[] if _pn.startswith("seed_scale_") else _dim_opts,
            value=[],
            label="Granularity",
        )
        for _pn in _selected
    ])
    return (fit_bounds_lo, fit_bounds_hi, fit_param_dims)


@app.cell
def _fitting_start_offset_ui(mo):
    fit_start_offset_enable = mo.ui.checkbox(
        label="Fit epidemic start date offset", value=False,
    )
    fit_start_offset_lo = mo.ui.number(
        value=-30, start=-365, stop=0, step=1, label="Min offset (days)",
    )
    fit_start_offset_hi = mo.ui.number(
        value=30, start=0, stop=365, step=1, label="Max offset (days)",
    )
    return fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi


@app.cell
def _fitting_display(
    get_n_targets, add_target_btn, del_target_btn,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_sim_days_input,
    fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_method, fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications, fit_run_button,
    mo, main_tab,
    num_age_groups, num_risk_groups, is_metapop,
    tip_label, wtip,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Fitting", None)
    _ACC = CLT_ACCENT["fitting"]
    _n = get_n_targets()
    _selected_params = list(fit_params_multiselect.value)
    _any_non_ts = any(fit_target_mode.value[_k] in ("scalar", "proportion") for _k in range(_n))
    _any_ts = any(fit_target_mode.value[_k] == "ts" for _k in range(_n))

    # Fitting tooltips carry rich HTML (the shared helpers escape plain text by
    # default, so pass html_tip=True here).
    def _tip(tip_text):
        return tip_label("", tip_text, html_tip=True)

    _PROPORTION_TIP = (
        "<b>Proportions mode</b><br><br>"
        "The <code>value</code> column must be a fraction (0–1) representing the share of the "
        "target variable in a given group. Each row defines one constraint.<br><br>"
        "<b>Denominator rule</b><br>"
        "<code>age=X</code> only → denominator is the grand total across all subpopulations<br>"
        "<code>age=X, subpop=Y</code> → denominator is the total within subpop Y<br>"
        "<code>subpop=Y</code> only → denominator is the grand total "
        "(i.e. the share of all infections attributable to subpop Y)<br>"
        "<code>risk=Z</code> only → grand total; "
        "<code>risk=Z, subpop=Y</code> → total within subpop Y<br><br>"
        "<b>Examples</b> (target = new_infections, 2 age groups, 2 subpops)<br>"
        "<code>age=0</code> value=0.40 → age 0 accounts for 40% of all infections<br>"
        "<code>age=0, subpop=city_A</code> value=0.35 → age 0 is 35% of city A's infections<br>"
        "<code>subpop=city_A</code> value=0.60 → city A has 60% of all infections"
    )

    # ── target cards ──────────────────────────────────────────────────────────
    _target_acc = {}
    for _k in range(_n):
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        _fname_note = mo.md("")
        if _src == "upload" and fit_target_upload.value[_k]:
            _fname = fit_target_upload.value[_k][0].name
            _fname_note = mo.callout(mo.md(f"Loaded: **{_fname}**"), kind="info")

        _data_input = fit_target_upload[_k] if _src == "upload" else fit_target_path[_k]

        _scalar_hint = mo.md("")
        if _mode == "scalar":
            _scalar_hint = mo.accordion({
                "Scalar format reference": mo.callout(
                    mo.md(
                        "**Required column:** `value`.\n\n"
                        "**Optional CSV columns:** `age` (integer index), `risk` (integer index), "
                        "`subpopulation` (name or integer index). "
                        "Each row is one group-specific constraint. When a group column is absent, "
                        "the group selection dropdowns below are used as the target "
                        "(or sum over all if 'All')."
                    ),
                    kind="info",
                )
            })
        elif _mode == "proportion":
            _scalar_hint = mo.accordion({
                "Proportions format reference": mo.callout(
                    mo.md(
                        "**Required column:** `value` (0–1).\n\n"
                        "**Optional CSV columns:** `age` (integer index), `risk` (integer index), "
                        "`subpopulation` (name or integer index).\n\n"
                        "Each row is **numerator / denominator** over the full simulation:\n\n"
                        "- **Numerator** — cumulative target variable restricted to the age, risk, "
                        "and subpopulation specified in the row (falling back to the dropdowns below "
                        "when a column is absent).\n"
                        "- **Denominator** — same variable summed over *all age and risk groups*, "
                        "restricted to the row's subpopulation *only when age or risk is also given*; "
                        "otherwise the grand total (all subpopulations).\n\n"
                        "**Examples** (target = `new_infections`, 2 age groups, 2 subpopulations):\n\n"
                        "| `subpopulation` | `age` | `value` | Meaning |\n"
                        "|---|---|---|---|\n"
                        "| *(absent)* | `0` | `0.40` | Age 0 is 40 % of all infections (both subpops) |\n"
                        "| `city_A` | `0` | `0.35` | Age 0 is 35 % of city A's infections |\n"
                        "| `city_A` | `1` | `0.65` | Age 1 is 65 % of city A's infections |\n"
                        "| `city_A` | *(absent)* | `0.60` | city A accounts for 60 % of all infections |\n\n"
                        "Age rows for the same subpopulation should sum to 1."
                    ),
                    kind="info",
                )
            })

        _prop_tip_row = mo.md("")
        if _mode == "proportion":
            _prop_tip_row = mo.hstack(
                [mo.md("*Proportions quick reference*"), _tip(_PROPORTION_TIP)],
                justify="start", align="center",
            )

        _slice_items = []
        if is_metapop:
            _slice_items.append(fit_target_subpop[_k])
        if int(num_age_groups) > 1:
            _slice_items.append(fit_target_age[_k])
        if int(num_risk_groups) > 1:
            _slice_items.append(fit_target_risk[_k])
        _slice_ui = mo.hstack(_slice_items, justify="start") if _slice_items else mo.md("")

        _tvar = str(fit_target_vars.value[_k]).strip()
        _tlabel = f"Target {_k + 1}  ·  {_mode}" + (f"  ·  {_tvar}" if _tvar else "")
        _target_acc[_tlabel] = mo.vstack([
            fit_target_src[_k],
            _data_input,
            _fname_note,
            fit_target_mode[_k],
            _prop_tip_row,
            _scalar_hint,
            fit_target_vars[_k],
            mo.hstack([fit_target_weight[_k]], justify="start"),
            _slice_ui,
        ])

    # ── parameter bounds ──────────────────────────────────────────────────────
    if _selected_params:
        _rows = []
        for _j, _pn in enumerate(_selected_params):
            _is_seed_scale = _pn.startswith("seed_scale_")
            _bound_widgets = [fit_bounds_lo[_j], fit_bounds_hi[_j]]
            if not _is_seed_scale:
                _bound_widgets.append(fit_param_dims[_j])
            _rows.append(mo.vstack([
                mo.md(f"**`{_pn}`**"),
                mo.hstack(_bound_widgets, justify="start", align="center"),
            ]))
        _bounds_section = mo.vstack(_rows)
    else:
        _bounds_section = mo.md("*Select parameters above to configure bounds.*")

    _LR_TIP = (
        "Step size used by the gradient optimiser.\n\n"
        "Adam: controls how far each parameter moves per gradient step.\n"
        "  Too large → unstable loss; too small → slow convergence.\n"
        "  Typical range: 0.001 – 0.05.\n\n"
        "L-BFGS: initial step size for the internal line search.\n"
        "  L-BFGS is less sensitive than Adam; 0.1 – 1.0 usually works.\n"
        "  The line search can shrink the step automatically."
    )
    _ITER_TIP = (
        "Number of optimisation steps or random draws, depending on method.\n\n"
        "Adam: exact number of gradient update steps per replication.\n"
        "  More iterations → better convergence, but more compute.\n\n"
        "L-BFGS: outer loop runs N ÷ 20 steps; each step performs\n"
        "  up to 20 internal line-search iterations, so total function\n"
        "  evaluations ≈ N (per replication).\n\n"
        "Accept-reject: total number of random parameter sets sampled.\n"
        "  Higher → better coverage of the parameter space."
    )
    _R2_TIP = (
        "Minimum weighted R² a sample must achieve to be 'accepted'.\n\n"
        "With multiple targets, R² is computed per-target and averaged\n"
        "using the target weights (λ).\n\n"
        "Accepted samples form the ensemble used in the Forecast tab.\n"
        "  Higher threshold → fewer but better-fitting accepted sets.\n"
        "  Lower threshold → larger ensemble, more uncertainty.\n\n"
        "If no samples are accepted, lower this value or increase\n"
        "the number of samples (Iterations / Max samples).\n\n"
        "R² = 1 − (SS_res / SS_tot); values above 0.7 are often\n"
        "considered a reasonable fit."
    )
    _REP_TIP = (
        "Number of independent optimisation runs for gradient methods.\n\n"
        "Starting points are spread across the parameter bounds using\n"
        "Latin Hypercube Sampling (LHS) to cover the space evenly and\n"
        "reduce the risk of converging to a local minimum.\n\n"
        "The best-fit result is the replication with the lowest final\n"
        "loss. All replication trajectories and parameter distributions\n"
        "are shown in the results, similar to the accept-reject method."
    )

    _method_val = fit_method.value
    _hyper = [wtip(fit_n_iter, _ITER_TIP, html_tip=True)]
    if _method_val != "ar":
        _hyper = [wtip(fit_lr, _LR_TIP, html_tip=True)] + _hyper
        _hyper.append(wtip(fit_n_replications, _REP_TIP, html_tip=True))
    if _method_val == "ar":
        _hyper.append(wtip(fit_r2_thresh, _R2_TIP, html_tip=True))

    _sim_days_widget = mo.md("")
    if _any_non_ts and not _any_ts:
        _sim_days_widget = mo.callout(
            mo.vstack([
                mo.md(
                    "All targets are scalar totals or proportions. Set the simulation length below "
                    "(number of days over which the totals are accumulated)."
                ),
                fit_sim_days_input,
            ]),
            kind="info",
        )

    _seed_scale_note = mo.md("")
    if any(_pn.startswith("seed_scale_") for _pn in _selected_params):
        _seed_scale_note = mo.callout(
            mo.md(
                "**Seed scaling** multiplies the initial count of the selected compartment "
                "by the fitted scale factor, adjusting the first compartment (susceptibles) "
                "to keep total population constant. "
                "Gradient methods optimise the scale alongside rate parameters."
            ),
            kind="info",
        )

    _start_offset_section = mo.vstack([
        fit_start_offset_enable,
        mo.hstack([fit_start_offset_lo, fit_start_offset_hi], justify="start")
        if fit_start_offset_enable.value else mo.md(""),
    ]) if True else mo.md("")

    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Fitting</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Calibrate model '
            "parameters to observed data.</div>"
        ),
        section_card(
            step_header("①", "Fit Targets",
                        "The data series / totals the model is calibrated against. "
                        "Click a target to expand it.",
                        accent=_ACC),
            mo.vstack([
                mo.accordion(_target_acc, multiple=True),
                mo.hstack([add_target_btn, del_target_btn,
                           mo.md(f"*{_n} of 20 targets*")],
                          justify="start", align="center"),
                _sim_days_widget,
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Parameters to Fit",
                        "Pick which parameters to estimate and their search bounds.",
                        accent=_ACC),
            mo.vstack([
                fit_params_multiselect,
                _seed_scale_note,
                _bounds_section,
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("③", "Epidemic Start Date",
                        "Optionally fit an offset for when the epidemic seeds.",
                        accent=_ACC),
            _start_offset_section,
            accent=_ACC,
        ),
        section_card(
            step_header("④", "Method & Run",
                        "Choose the optimiser, set its hyperparameters, then run.",
                        accent=_ACC),
            mo.vstack([
                fit_method,
                mo.hstack(_hyper, justify="start"),
                fit_run_button,
            ]),
            accent=_ACC,
        ),
    ])
    return


@app.cell
def _fitting_obs_parse(
    get_n_targets,
    fit_target_src, fit_target_upload, fit_target_path, fit_target_mode,
    pd, io, Path,
):
    _n = get_n_targets()
    # fit_obs_arrays: list of (np.array | list-of-dicts | None) per target
    # fit_obs_n_days: list of int (days in timeseries) or 0 for scalar targets
    fit_obs_arrays = []
    fit_obs_n_days = []

    for _k in range(_n):
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        _df = None
        try:
            if _src == "upload" and fit_target_upload.value[_k]:
                _df = pd.read_csv(io.BytesIO(fit_target_upload.value[_k][0].contents))
            elif _src == "path" and fit_target_path.value[_k].strip():
                _df = pd.read_csv(Path(fit_target_path.value[_k].strip()).expanduser())
        except Exception:
            _df = None

        if _df is None:
            fit_obs_arrays.append(None)
            fit_obs_n_days.append(0)
            continue

        if _mode in ("scalar", "proportion"):
            if "value" not in _df.columns:
                fit_obs_arrays.append(None)
                fit_obs_n_days.append(0)
                continue
            _rows = []
            for _, _row in _df.iterrows():
                _entry = {"value": float(_row["value"])}
                for _col in ("age", "risk", "subpopulation"):
                    if _col in _df.columns and pd.notna(_row.get(_col)):
                        _entry[_col] = _row[_col]
                _rows.append(_entry)
            fit_obs_arrays.append(_rows)
            fit_obs_n_days.append(0)
        else:
            _META_COLS_TS = {"date", "day", "time", "week", "subpopulation", "age", "risk"}
            _non_id = [c for c in _df.columns if c.lower() not in _META_COLS_TS]
            if not _non_id:
                _non_id = [c for c in _df.columns if c.lower() not in ("date", "day", "time", "week")]
            _val_col = "value" if "value" in _df.columns else (_non_id[0] if _non_id else None)
            if _val_col:
                _arr = _df[_val_col].dropna().to_numpy(dtype=float)
                fit_obs_arrays.append(_arr)
                fit_obs_n_days.append(len(_arr))
            else:
                fit_obs_arrays.append(None)
                fit_obs_n_days.append(0)

    return fit_obs_arrays, fit_obs_n_days


@app.cell
def _run_fitting(
    fit_run_button, fit_obs_arrays, fit_obs_n_days,
    get_n_targets,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_sim_days_input,
    fit_method, fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications,
    config_dict, compartments, is_metapop,
    build_compartment_init, read_initial_conditions,
    start_date_input, timesteps, rng_seed,
    num_age_groups, num_risk_groups,
    metapop_folder_input, metapop_travel_config,
    make_single_pop_metapop, make_metapop_from_folder,
    build_generic_torch_inputs, generic_torch_simulate_calibration_target, RATE_TEMPLATE_REGISTRY,
    torch, np, json, mo, compute_rsquared, FitResult, build_scalar_array, Path,
):
    fit_result = None
    if fit_run_button.value:
        _n = get_n_targets()
        _selected_params = list(fit_params_multiselect.value)
        _lo_vals = list(fit_bounds_lo.value)
        _hi_vals = list(fit_bounds_hi.value)
        _dim_vals = list(fit_param_dims.value)
        _A = num_age_groups
        _R = num_risk_groups
    
        _target_vars_list = [list(fit_target_vars.value[_k]) for _k in range(_n)]
        _target_modes = [fit_target_mode.value[_k] for _k in range(_n)]
        _target_weights = [float(fit_target_weight.value[_k]) for _k in range(_n)]
        _weight_sum = sum(_target_weights) or 1.0
    
        # ── slice-index helpers ────────────────────────────────────────────────────
        def _parse_idx(val):
            if val == "All (sum)":
                return -1
            return int(str(val).split(":")[0])
    
        _sp_idxs = [_parse_idx(fit_target_subpop.value[_k]) for _k in range(_n)]
        _age_idxs = [_parse_idx(fit_target_age.value[_k]) for _k in range(_n)]
        _risk_idxs = [_parse_idx(fit_target_risk.value[_k]) for _k in range(_n)]
    
        # Target display labels
        _target_labels = []
        for _k in range(_n):
            if fit_target_src.value[_k] == "upload" and fit_target_upload.value[_k]:
                _target_labels.append(fit_target_upload.value[_k][0].name)
            elif fit_target_src.value[_k] == "path" and fit_target_path.value[_k].strip():
                _target_labels.append(Path(fit_target_path.value[_k].strip()).name)
            else:
                _target_labels.append(f"Target {_k + 1}")
    
        # Subpop name → index map (for scalar CSV rows)
        _subpop_names_run = []
        if is_metapop and metapop_folder_input.value.strip():
            try:
                with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                    _subpop_names_run = json.load(_f).get("subpopulations", [])
            except Exception:
                _subpop_names_run = []
        _subpop_name_to_idx = {str(_nm): _i for _i, _nm in enumerate(_subpop_names_run)}
    
        def _parse_scalar_row_idxs(row, sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
            _sp = row.get("subpopulation", None)
            _sp_idx = sp_fallback
            if _sp is not None:
                _sp_s = str(_sp).strip()
                if _sp_s.isdigit():
                    _sp_idx = int(_sp_s)
                elif _sp_s in _subpop_name_to_idx:
                    _sp_idx = _subpop_name_to_idx[_sp_s]
            _ag_idx = int(row["age"]) if "age" in row else ag_fallback
            _rk_idx = int(row["risk"]) if "risk" in row else rk_fallback
            return _sp_idx, _ag_idx, _rk_idx
    
        # ── validation ─────────────────────────────────────────────────────────────
        for _k in range(_n):
            if fit_obs_arrays[_k] is None:
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_k + 1}: no observed data loaded.**"), kind="warn"),
                )
            if not _target_vars_list[_k]:
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_k + 1}: no variables selected.**"), kind="warn"),
                )
        mo.stop(
            not _selected_params,
            mo.callout(mo.md("**No parameters to fit.** Select parameters above."), kind="warn"),
        )
    
        _target_tvs = [[t for t in _target_vars_list[_k] if t not in compartments] for _k in range(_n)]
        _target_comps = [[t for t in _target_vars_list[_k] if t in compartments] for _k in range(_n)]
    
        # Determine simulation length
        _ts_days = [fit_obs_n_days[_k] for _k in range(_n) if _target_modes[_k] == "ts"]
        _sim_days = max(_ts_days) if _ts_days else int(fit_sim_days_input.value)
        mo.stop(
            len(set(_ts_days)) > 1,
            mo.callout(
                mo.md(
                    "**Timeseries targets have mismatched lengths:** "
                    + ", ".join(
                        f"Target {_k + 1}: {fit_obs_n_days[_k]} days"
                        for _k in range(_n)
                        if _target_modes[_k] == "ts"
                    )
                    + ". All timeseries targets must have the same number of observations."
                ),
                kind="danger",
            ),
        )
        _num_fit_days_per_target = [
            fit_obs_n_days[_k] if _target_modes[_k] == "ts" else _sim_days
            for _k in range(_n)
        ]
    
        _start = start_date_input.value.strip() or "2024-01-01"
        _ts = int(timesteps.value)
        _seed_b = int(rng_seed.value)
    
        _ci = None
        if not is_metapop:
            # Initial conditions from the Step 6 tables via config_dict.
            _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
            _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((_A, _R))), dtype=float)
            _seed_arrays = {
                _c: np.asarray(_a, dtype=float)
                for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
                if _c in compartments
            }
            _ci, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)
    
        _bounds_grad = {
            _selected_params[_j]: [float(_lo_vals[_j]), float(_hi_vals[_j])]
            for _j in range(len(_selected_params))
        }

        # Identify seed-scale virtual params (scale initial compartment count)
        _seed_scale_comps = {
            _pn[len("seed_scale_"):]: _j
            for _j, _pn in enumerate(_selected_params)
            if _pn.startswith("seed_scale_") and _pn[len("seed_scale_"):] in compartments
        }

        # Epidemic start date offset settings
        _fit_start_offset = fit_start_offset_enable.value
        _offset_lo_val = int(fit_start_offset_lo.value)
        _offset_hi_val = int(fit_start_offset_hi.value)

        from datetime import datetime as _dt, timedelta as _td
        def _shift_date(base_str, days):
            return (
                _dt.strptime(base_str, "%Y-%m-%d") + _td(days=int(days))
            ).strftime("%Y-%m-%d")

        _loss_curve = []
        _best_params = {}
        _accepted_params_list = []
        _best_trajs = {}
        _accepted_trajectories = []
    
        with mo.status.spinner("Running fitting..."):
            try:
                # ── gradient methods (Adam / L-BFGS) ──────────────────────────────
                if fit_method.value in ("adam", "lbfgs"):
                    mo.stop(
                        torch is None,
                        mo.callout(
                            mo.md("**PyTorch not available.** Install torch to use gradient-based fitting."),
                            kind="danger",
                        ),
                    )
                    # Scale compartment values and total population up by this factor.
                    # PyTorch uses softplus for non-negativity; for small counts it
                    # diverges from the identity — large values (~10000) make it negligible.
                    # N must scale with the compartments so I/N (force of infection) is unchanged.
                    _GRAD_SCALE = 10000.0

                    _all_tvs_union = list(dict.fromkeys(tv for tvs in _target_tvs for tv in tvs))
                    _all_comps_union = list(dict.fromkeys(c for cs in _target_comps for c in cs))

                    # Build initial model to identify valid regular params
                    if is_metapop:
                        _metapop, _mc = make_metapop_from_folder(
                            metapop_folder_input.value.strip(),
                            config_dict, _start, _sim_days, list(compartments),
                            seed_offset=0, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=False, tvs=_all_tvs_union, save_daily=False,
                            travel_config=metapop_travel_config or None,
                        )
                    else:
                        _metapop, _mc, _ = make_single_pop_metapop(
                            config_dict, _start, _sim_days, _ci,
                            ts_per_day=_ts, stochastic=False, tvs=_all_tvs_union,
                            save_daily=False, seed_base=_seed_b,
                            travel_config=metapop_travel_config,
                        )

                    _ti = build_generic_torch_inputs(_metapop, _mc, _sim_days)
                    _ti["params_dict"]["total_pop_age_risk"] = (
                        _ti["params_dict"]["total_pop_age_risk"] * _GRAD_SCALE
                    )
                    _ti["precomputed"].total_pop_LAR_tensor = (
                        _ti["precomputed"].total_pop_LAR_tensor * _GRAD_SCALE
                    )
                    _ti["precomputed"].total_pop_LA = (
                        _ti["precomputed"].total_pop_LA * _GRAD_SCALE
                    )

                    # Valid regular params (seed_scale virtual params are excluded)
                    _valid_pn_idxs = [
                        _j for _j, _pn in enumerate(_selected_params)
                        if _pn in _ti["params_dict"]
                    ]
                    mo.stop(
                        not _valid_pn_idxs and not _seed_scale_comps and not _fit_start_offset,
                        mo.callout(
                            mo.md("**None of the specified parameters found in params dict.** Check names."),
                            kind="danger",
                        ),
                    )

                    # Mutable dicts updated per replication so closures see current values
                    _seed_base_state = {
                        k: v.clone().detach() * _GRAD_SCALE
                        for k, v in _ti["state_dict"].items()
                    }
                    _scale_tensors = {}  # {comp_name: tensor} — cleared and repopulated per rep

                    # Extended LHS: regular params + seed_scale params + offset
                    _seed_scale_keys = list(_seed_scale_comps.keys())
                    _lhs_lo_list = [float(_lo_vals[_j]) for _j in _valid_pn_idxs]
                    _lhs_hi_list = [float(_hi_vals[_j]) for _j in _valid_pn_idxs]
                    for _comp_ss in _seed_scale_keys:
                        _j_ss = _seed_scale_comps[_comp_ss]
                        _lhs_lo_list.append(float(_lo_vals[_j_ss]))
                        _lhs_hi_list.append(float(_hi_vals[_j_ss]))
                    _offset_lhs_col = len(_lhs_lo_list)
                    if _fit_start_offset:
                        _lhs_lo_list.append(float(_offset_lo_val))
                        _lhs_hi_list.append(float(_offset_hi_val))

                    _n_rep = int(fit_n_replications.value)
                    _n_lhs_total = max(len(_lhs_lo_list), 1)
                    try:
                        from scipy.stats.qmc import LatinHypercube as _LHC
                        _lhs_unit = _LHC(d=_n_lhs_total).random(n=_n_rep)
                    except Exception:
                        _lhs_unit = np.random.rand(_n_rep, _n_lhs_total)
                    _lo_ext = np.array(_lhs_lo_list) if _lhs_lo_list else np.array([0.0])
                    _hi_ext = np.array(_lhs_hi_list) if _lhs_hi_list else np.array([1.0])
                    _lhs_scaled = _lo_ext + _lhs_unit[:, :len(_lhs_lo_list)] * (_hi_ext - _lo_ext)

                    def _slice_tensor(sim, sp_idx, ag_idx, rk_idx):
                        """Slice (days, L, A, R) and sum remaining dims → (days,)."""
                        _s = sim
                        if sp_idx >= 0:
                            _s = _s[:, sp_idx:sp_idx + 1, :, :]
                        if ag_idx >= 0:
                            _s = _s[:, :, ag_idx:ag_idx + 1, :]
                        if rk_idx >= 0:
                            _s = _s[:, :, :, rk_idx:rk_idx + 1]
                        return _s.sum(dim=tuple(range(1, _s.dim())))

                    def _build_scaled_state():
                        """Build initial state tensor dict, applying seed-scale factors."""
                        _state_run = {}
                        for _ks, _vs in _seed_base_state.items():
                            if _ks in _scale_tensors:
                                _state_run[_ks] = _seed_base_state[_ks] * _scale_tensors[_ks]
                            elif compartments and _ks == compartments[0] and _scale_tensors:
                                _delta = sum(
                                    _seed_base_state[_c] * (_scale_tensors[_c] - 1.0)
                                    for _c in _scale_tensors
                                    if _c in _seed_base_state
                                )
                                _state_run[_ks] = torch.clamp(
                                    _seed_base_state[_ks] - _delta, min=0.0
                                )
                            else:
                                _state_run[_ks] = _vs.clone()
                        return _state_run

                    def _compute_total_loss():
                        _state_run = _build_scaled_state()
                        _sim_all = generic_torch_simulate_calibration_target(
                            _state_run, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                            _ti["precomputed"], _ti["schedules_dict"],
                            _sim_days, _ts, _all_tvs_union, _all_comps_union,
                        )  # dict: {var_name: (days, L, A, R)}
                        _total = torch.tensor(0.0, dtype=torch.float64)
                        for _k in range(_n):
                            _parts = [_sim_all[v] for v in _target_vars_list[_k]]
                            _sim_k = _parts[0]
                            for _pt in _parts[1:]:
                                _sim_k = _sim_k + _pt
                            _w_k = _target_weights[_k] / _weight_sum
                            if _target_modes[_k] == "scalar":
                                _loss_k = torch.tensor(0.0, dtype=torch.float64)
                                _obs_scalar_sq_sum = sum(
                                    float(_row["value"]) ** 2 for _row in fit_obs_arrays[_k]
                                ) + 1e-10
                                for _row in fit_obs_arrays[_k]:
                                    _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                        _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    _sliced = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i) / _GRAD_SCALE
                                    _obs_v = torch.tensor(float(_row["value"]), dtype=torch.float64)
                                    _loss_k = _loss_k + (_sliced.sum() - _obs_v) ** 2
                                _loss_k = _loss_k / _obs_scalar_sq_sum
                            elif _target_modes[_k] == "proportion":
                                _loss_k = torch.tensor(0.0, dtype=torch.float64)
                                # Normalise by sum(obs²) so loss ≈ 1 when sim=0, ≈ 0 when perfect,
                                # keeping it in [0, ~1] regardless of epidemic size or number of rows.
                                _obs_prop_sq_sum = sum(
                                    float(_row["value"]) ** 2 for _row in fit_obs_arrays[_k]
                                ) + 1e-10
                                for _ri_prop, _row in enumerate(fit_obs_arrays[_k]):
                                    _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                        _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    # When no group is specified, treat row order as subpop order
                                    # so that rows without a "subpopulation" column still compute
                                    # a meaningful per-subpop proportion instead of always 1.0.
                                    if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                        _sp_i = _ri_prop
                                    # Fix subpop in denominator only when age or risk also given;
                                    # subpop-only rows compare against grand total (share of all
                                    # infections attributable to that subpopulation).
                                    _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                                    _den = _slice_tensor(_sim_k, _den_sp, -1, -1).sum()
                                    _num = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i).sum()
                                    _sim_prop = _num / (_den + 1e-10)
                                    _obs_prop = torch.tensor(float(_row["value"]), dtype=torch.float64)
                                    _loss_k = _loss_k + (_sim_prop - _obs_prop) ** 2
                                _loss_k = _loss_k / _obs_prop_sq_sum
                            else:
                                _days_k = _num_fit_days_per_target[_k]
                                _sliced = _slice_tensor(_sim_k, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]) / _GRAD_SCALE
                                _obs_t_k = torch.tensor(
                                    fit_obs_arrays[_k][:_days_k], dtype=torch.float64
                                )
                                # Normalise by mean(obs²) so loss ≈ 1 when sim=0, ≈ 0 when perfect,
                                # matching the [0, ~1] scale of proportion and scalar losses.
                                _obs_sq_mean_k = float((_obs_t_k ** 2).mean()) + 1e-10
                                _loss_k = ((_sliced[:_days_k] - _obs_t_k) ** 2).mean() / _obs_sq_mean_k
                            _total = _total + _w_k * _loss_k
                        return _total

                    def _record_params_grad():
                        _out = {}
                        for _pn in _selected_params:
                            if _pn.startswith("seed_scale_"):
                                _comp = _pn[len("seed_scale_"):]
                                if _comp in _scale_tensors:
                                    _out[_pn] = float(_scale_tensors[_comp].item())
                            elif _pn in _ti["params_dict"]:
                                _v = _ti["params_dict"][_pn].detach()
                                if _v.ndim == 3 and _v.shape[1] == 1 and _v.shape[2] == 1:
                                    for _spi in range(_v.shape[0]):
                                        _out[f"{_pn}_subpop{_spi}"] = float(_v[_spi, 0, 0].item())
                                else:
                                    _out[_pn] = _v.tolist()
                        return _out

                    def _record_trajs_grad():
                        _trajs = {}
                        with torch.no_grad():
                            _state_run = _build_scaled_state()
                            _sim_all = generic_torch_simulate_calibration_target(
                                _state_run, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                                _ti["precomputed"], _ti["schedules_dict"],
                                _sim_days, _ts, _all_tvs_union, _all_comps_union,
                            )
                            for _k in range(_n):
                                _parts = [_sim_all[v] for v in _target_vars_list[_k]]
                                _sim_k = _parts[0]
                                for _pt in _parts[1:]:
                                    _sim_k = _sim_k + _pt
                                if _target_modes[_k] == "scalar":
                                    _row_sims = []
                                    for _row in fit_obs_arrays[_k]:
                                        _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                            _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                        )
                                        _sliced = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i)
                                        _row_sims.append(float(_sliced.sum().item()) / _GRAD_SCALE)
                                    _trajs[f"target_{_k}"] = _row_sims
                                elif _target_modes[_k] == "proportion":
                                    _row_sims = []
                                    for _ri_prop, _row in enumerate(fit_obs_arrays[_k]):
                                        _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                            _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                        )
                                        if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                            _sp_i = _ri_prop
                                        _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                                        _den = _slice_tensor(_sim_k, _den_sp, -1, -1).sum()
                                        _num = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i).sum()
                                        _row_sims.append(float((_num / (_den + 1e-10)).item()))
                                    _trajs[f"target_{_k}"] = _row_sims
                                else:
                                    _days_k = _num_fit_days_per_target[_k]
                                    _sliced = _slice_tensor(
                                        _sim_k, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    _trajs[f"target_{_k}"] = (_sliced[:_days_k] / _GRAD_SCALE).numpy().tolist()
                        return _trajs

                    _n_it = int(fit_n_iter.value)
                    _lr_val = float(fit_lr.value)
                    _best_loss = float("inf")
                    _all_rep_params = []
                    _all_rep_trajs = []
                    _n_regular_lhs = len(_valid_pn_idxs)

                    for _rep_idx in range(_n_rep):
                        # ── Extract seed-scale and offset from extended LHS ────────
                        _scale_tensors.clear()
                        _sampled_offset = 0
                        for _ssi, _comp_ss in enumerate(_seed_scale_keys):
                            _sval = float(_lhs_scaled[_rep_idx, _n_regular_lhs + _ssi])
                            _scale_tensors[_comp_ss] = torch.tensor(
                                _sval, dtype=torch.float64, requires_grad=True,
                            )
                        if _fit_start_offset and len(_lhs_lo_list) > _offset_lhs_col:
                            _sampled_offset = int(round(
                                float(_lhs_scaled[_rep_idx, _offset_lhs_col])
                            ))
                        _start_rep = _shift_date(_start, _sampled_offset) if _fit_start_offset else _start

                        # ── Per-replication model rebuild when start date varies ──
                        if _fit_start_offset:
                            if is_metapop:
                                _metapop, _mc = make_metapop_from_folder(
                                    metapop_folder_input.value.strip(),
                                    config_dict, _start_rep, _sim_days, list(compartments),
                                    seed_offset=_rep_idx, seed_base=_seed_b, ts_per_day=_ts,
                                    stochastic=False, tvs=_all_tvs_union, save_daily=False,
                                    travel_config=metapop_travel_config or None,
                                )
                            else:
                                _metapop, _mc, _ = make_single_pop_metapop(
                                    config_dict, _start_rep, _sim_days, _ci,
                                    ts_per_day=_ts, stochastic=False, tvs=_all_tvs_union,
                                    save_daily=False, seed_base=_seed_b + _rep_idx,
                                    travel_config=metapop_travel_config,
                                )
                            _ti = build_generic_torch_inputs(_metapop, _mc, _sim_days)
                            _ti["params_dict"]["total_pop_age_risk"] = (
                                _ti["params_dict"]["total_pop_age_risk"] * _GRAD_SCALE
                            )
                            _ti["precomputed"].total_pop_LAR_tensor = (
                                _ti["precomputed"].total_pop_LAR_tensor * _GRAD_SCALE
                            )
                            _ti["precomputed"].total_pop_LA = (
                                _ti["precomputed"].total_pop_LA * _GRAD_SCALE
                            )
                            _seed_base_state.clear()
                            _seed_base_state.update(
                                {k: v.clone().detach() * _GRAD_SCALE for k, v in _ti["state_dict"].items()}
                            )

                        # ── Initialise regular param tensors from LHS ─────────────
                        _opt_tensors = list(_scale_tensors.values())
                        _lhs_col = 0
                        for _j, _pn in enumerate(_selected_params):
                            if _pn not in _ti["params_dict"]:
                                continue
                            _dims = _dim_vals[_j] if _dim_vals else []
                            _has_age = "age groups" in _dims
                            _has_risk = "risk groups" in _dims
                            _init_val = float(_lhs_scaled[_rep_idx, _lhs_col])
                            _lhs_col += 1
                            if _has_age and _has_risk:
                                _t = torch.full((_A, _R), _init_val, dtype=torch.float64, requires_grad=True)
                            elif _has_age:
                                _t = torch.full((_A, 1), _init_val, dtype=torch.float64, requires_grad=True)
                            elif _has_risk:
                                _t = torch.full((1, _R), _init_val, dtype=torch.float64, requires_grad=True)
                            elif "subpopulation" in _dims:
                                _L = list(_ti["state_dict"].values())[0].shape[0]
                                _t = torch.full((_L, 1, 1), _init_val, dtype=torch.float64, requires_grad=True)
                            else:
                                _t = torch.tensor(_init_val, dtype=torch.float64, requires_grad=True)
                            _ti["params_dict"][_pn] = _t
                            _opt_tensors.append(_t)

                        _rep_best_loss = float("inf")
                        _rep_best_params = {}
                        _rep_best_trajs = {}

                        _rep_loss_curve = []
                        if not _opt_tensors:
                            # Only start-offset is being searched; evaluate once
                            with torch.no_grad():
                                _lv0 = float(_compute_total_loss().item())
                            _rep_loss_curve = [_lv0]
                            _rep_best_loss = _lv0
                            _rep_best_params = _record_params_grad()
                            _rep_best_trajs = _record_trajs_grad()
                        elif fit_method.value == "adam":
                            _opt = torch.optim.Adam(_opt_tensors, lr=_lr_val)
                            for _ in range(_n_it):
                                _opt.zero_grad()
                                _loss = _compute_total_loss()
                                _loss.backward()
                                _opt.step()
                                for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                    if _pn in _ti["params_dict"]:
                                        _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                                for _comp_ss in _seed_scale_keys:
                                    _j_ss = _seed_scale_comps[_comp_ss]
                                    _scale_tensors[_comp_ss].data.clamp_(
                                        float(_lo_vals[_j_ss]), float(_hi_vals[_j_ss])
                                    )
                                _lv = float(_loss.item())
                                _rep_loss_curve.append(_lv)
                                if _lv < _rep_best_loss:
                                    _rep_best_loss = _lv
                                    _rep_best_params = _record_params_grad()
                                    _rep_best_trajs = _record_trajs_grad()

                        else:  # lbfgs
                            _opt = torch.optim.LBFGS(_opt_tensors, lr=_lr_val, max_iter=20)
                            for _ in range(max(1, _n_it // 20)):
                                def _closure():
                                    _opt.zero_grad()
                                    _l = _compute_total_loss()
                                    _l.backward()
                                    return _l
                                _opt.step(_closure)
                                for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                    if _pn in _ti["params_dict"]:
                                        _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                                for _comp_ss in _seed_scale_keys:
                                    _j_ss = _seed_scale_comps[_comp_ss]
                                    _scale_tensors[_comp_ss].data.clamp_(
                                        float(_lo_vals[_j_ss]), float(_hi_vals[_j_ss])
                                    )
                                with torch.no_grad():
                                    _lv2 = float(_compute_total_loss().item())
                                _rep_loss_curve.append(_lv2)
                                if _lv2 < _rep_best_loss:
                                    _rep_best_loss = _lv2
                                    _rep_best_params = _record_params_grad()
                                    _rep_best_trajs = _record_trajs_grad()
                        _loss_curve.append(_rep_loss_curve)

                        if _fit_start_offset:
                            _rep_best_params["epidemic_start_offset_days"] = _sampled_offset
                            _rep_best_params["epidemic_start_date"] = _start_rep

                        _all_rep_params.append(_rep_best_params)
                        _all_rep_trajs.append(_rep_best_trajs)
                        if _rep_best_loss < _best_loss:
                            _best_loss = _rep_best_loss
                            _best_params = _rep_best_params
                            _best_trajs = _rep_best_trajs

                    _accepted_params_list = _all_rep_params
                    _accepted_trajectories = _all_rep_trajs
    
                # ── accept-reject ──────────────────────────────────────────────────
                else:
                    _all_tvs_ar = list(dict.fromkeys(tv for tvs in _target_tvs for tv in tvs))
                    _all_comps_ar = list(dict.fromkeys(c for cs in _target_comps for c in cs))
                    _has_any_comp_ar = bool(_all_comps_ar)

                    _n_subpops = 1
                    if is_metapop and any(
                        "subpopulation" in (_dim_vals[_j] if _dim_vals else [])
                        for _j in range(len(_selected_params))
                    ):
                        try:
                            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                                _n_subpops = len(json.load(_f).get("subpopulations", []))
                        except Exception:
                            _n_subpops = 1

                    # Pre-load metapop base ICs once for seed scaling
                    _metapop_base_inits = []
                    if _seed_scale_comps and is_metapop and metapop_folder_input.value.strip():
                        _folder_ar = Path(metapop_folder_input.value.strip())
                        for _sp_nm_ar in _subpop_names_run:
                            _ic_path_ar = _folder_ar / f"initial_conditions_{_sp_nm_ar}.json"
                            _ic_cfg_ar = (config_dict.get("initial_conditions", {}) or {}).get(_sp_nm_ar, {})
                            _table_ci_ar = read_initial_conditions(
                                config_dict, _sp_nm_ar, compartments, _A, _R)
                            _comp_base_ar = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                            _epi_base_ar = {}
                            if _ic_cfg_ar.get("seeds") and _table_ci_ar is not None:
                                _comp_base_ar = _table_ci_ar
                            elif _ic_path_ar.exists():
                                try:
                                    with open(_ic_path_ar) as _f:
                                        _ic_data_ar = json.load(_f)
                                    for _c, _arr in _ic_data_ar.get("compartments", {}).items():
                                        if _c in compartments:
                                            _comp_base_ar[_c] = np.array(_arr, dtype=float)
                                    for _em, _arr in _ic_data_ar.get("epi_metrics", {}).items():
                                        _epi_base_ar[_em] = np.array(_arr, dtype=float)
                                except Exception:
                                    pass
                            elif _table_ci_ar is not None:
                                _comp_base_ar = _table_ci_ar
                            _metapop_base_inits.append((_comp_base_ar, _epi_base_ar))

                    def _extract_ts_sliced(metapop_obj, target_vars_k, sp_idx, ag_idx, rk_idx, n_days):
                        """Extract timeseries array with optional subpop/age/risk slicing."""
                        _sps = list(metapop_obj.subpop_models.values())
                        _sp_list = [_sps[sp_idx]] if sp_idx >= 0 else _sps
                        _result = None
                        for _sp_m in _sp_list:
                            for _var in target_vars_k:
                                if _var in _sp_m.transition_variables:
                                    _h = np.array(
                                        _sp_m.transition_variables[_var].history_vals_list
                                    )[:n_days]
                                elif _var in _sp_m.compartments:
                                    _h = np.array(
                                        _sp_m.compartments[_var].history_vals_list
                                    )[:n_days]
                                else:
                                    continue
                                if ag_idx >= 0:
                                    _h = _h[:, ag_idx:ag_idx + 1, :]
                                if rk_idx >= 0:
                                    _h = _h[:, :, rk_idx:rk_idx + 1]
                                _hv = _h.sum(axis=(1, 2))
                                _result = _hv if _result is None else _result + _hv
                        return _result if _result is not None else np.zeros(n_days)

                    def _extract_scalar_rows(metapop_obj, target_vars_k, scalar_rows, n_days,
                                             sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
                        """Extract per-row scalar totals for scalar-mode targets."""
                        _sps = list(metapop_obj.subpop_models.values())
                        _row_sims = []
                        for _row in scalar_rows:
                            _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                _row, sp_fallback, ag_fallback, rk_fallback
                            )
                            _sp_list = [_sps[_sp_i]] if _sp_i >= 0 else _sps
                            _total = 0.0
                            for _sp_m in _sp_list:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    if _ag_i >= 0:
                                        _h = _h[:, _ag_i:_ag_i + 1, :]
                                    if _rk_i >= 0:
                                        _h = _h[:, :, _rk_i:_rk_i + 1]
                                    _total += float(_h.sum())
                            _row_sims.append(_total)
                        return _row_sims

                    def _extract_proportion_rows(metapop_obj, target_vars_k, scalar_rows, n_days,
                                                 sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
                        """Extract per-row proportions for proportion-mode targets.

                        Denominator subpop: fixed to the row's subpop only when age or risk is
                        also specified; otherwise grand total (all subpops). This allows
                        subpop-only rows to express the share of infections attributable to
                        that subpopulation.
                        """
                        _sps = list(metapop_obj.subpop_models.values())
                        _row_sims = []
                        for _ri_prop, _row in enumerate(scalar_rows):
                            _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                _row, sp_fallback, ag_fallback, rk_fallback
                            )
                            # When no group is specified, treat row order as subpop order
                            if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                _sp_i = _ri_prop
                            _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                            _sp_list_num = [_sps[_sp_i]] if (0 <= _sp_i < len(_sps)) else _sps
                            _sp_list_den = [_sps[_den_sp]] if _den_sp >= 0 else _sps
                            _num = 0.0
                            _den = 0.0
                            for _sp_m in _sp_list_den:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h_full = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h_full = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    _den += float(_h_full.sum())
                            for _sp_m in _sp_list_num:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h_full = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h_full = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    _h_sliced = _h_full
                                    if _ag_i >= 0:
                                        _h_sliced = _h_sliced[:, _ag_i:_ag_i + 1, :]
                                    if _rk_i >= 0:
                                        _h_sliced = _h_sliced[:, :, _rk_i:_rk_i + 1]
                                    _num += float(_h_sliced.sum())
                            _row_sims.append(_num / _den if _den > 1e-10 else 0.0)
                        return _row_sims

                    _rng = np.random.default_rng(_seed_b)
                    _n_samples = int(fit_n_iter.value)
                    _r2_thresh = float(fit_r2_thresh.value)
                    _best_r2 = -float("inf")

                    for _s_idx in range(_n_samples):
                        _sampled = {}
                        _per_subpop = [{} for _ in range(_n_subpops)]
                        _sampled_scales = {}  # {comp: scale_value}
                        _sampled_offset = 0

                        for _j, _pn in enumerate(_selected_params):
                            _lo = float(_lo_vals[_j])
                            _hi = float(_hi_vals[_j])
                            if _pn.startswith("seed_scale_"):
                                _comp_ss = _pn[len("seed_scale_"):]
                                if _comp_ss in compartments:
                                    _sampled_scales[_comp_ss] = float(_rng.uniform(_lo, _hi))
                                continue
                            _dims = _dim_vals[_j] if _dim_vals else []
                            _has_age = "age groups" in _dims
                            _has_risk = "risk groups" in _dims
                            _has_meta = "subpopulation" in _dims
                            if _has_age and _has_risk:
                                _inner_shape = (_A, _R)
                            elif _has_age:
                                _inner_shape = (_A, 1)
                            elif _has_risk:
                                _inner_shape = (1, _R)
                            else:
                                _inner_shape = None
                            if _has_meta:
                                for _spi in range(_n_subpops):
                                    _per_subpop[_spi][_pn] = (
                                        _rng.uniform(_lo, _hi, size=_inner_shape).tolist()
                                        if _inner_shape else float(_rng.uniform(_lo, _hi))
                                    )
                            elif _inner_shape:
                                _sampled[_pn] = _rng.uniform(_lo, _hi, size=_inner_shape).tolist()
                            else:
                                _sampled[_pn] = float(_rng.uniform(_lo, _hi))

                        # Sample start date offset (integer)
                        if _fit_start_offset:
                            _sampled_offset = int(_rng.integers(_offset_lo_val, _offset_hi_val + 1))
                        _start_iter = _shift_date(_start, _sampled_offset) if _fit_start_offset else _start

                        # Build scaled initial conditions for this iteration
                        _ci_iter = None
                        _init_override = None
                        if _sampled_scales:
                            if is_metapop and _metapop_base_inits:
                                _init_override = []
                                for _comp_base_sp, _epi_base_sp in _metapop_base_inits:
                                    _scaled_comp = {}
                                    _delta_sp = np.zeros_like(
                                        _comp_base_sp.get(
                                            compartments[0],
                                            build_scalar_array(0.0, _A, _R),
                                        )
                                    ) if compartments else None
                                    for _c in compartments:
                                        _base_arr = _comp_base_sp.get(_c, build_scalar_array(0.0, _A, _R))
                                        if _c in _sampled_scales and _c != (compartments[0] if compartments else None):
                                            _sc = _sampled_scales[_c]
                                            _scaled_comp[_c] = _base_arr * _sc
                                            if _delta_sp is not None:
                                                _delta_sp += _base_arr * (_sc - 1.0)
                                        else:
                                            _scaled_comp[_c] = _base_arr.copy()
                                    if compartments and _delta_sp is not None:
                                        _first_arr = _comp_base_sp.get(
                                            compartments[0], build_scalar_array(0.0, _A, _R)
                                        )
                                        _scaled_comp[compartments[0]] = np.maximum(
                                            _first_arr - _delta_sp, 0.0
                                        )
                                    _init_override.append((_scaled_comp, _epi_base_sp))
                            elif not is_metapop and _ci is not None:
                                _ci_iter = {}
                                _delta = np.zeros_like(
                                    _ci.get(compartments[0], build_scalar_array(0.0, _A, _R))
                                ) if compartments else None
                                for _c in compartments:
                                    _base_arr = _ci.get(_c, build_scalar_array(0.0, _A, _R))
                                    if _c in _sampled_scales and _c != (compartments[0] if compartments else None):
                                        _sc = _sampled_scales[_c]
                                        _ci_iter[_c] = _base_arr * _sc
                                        if _delta is not None:
                                            _delta += _base_arr * (_sc - 1.0)
                                    else:
                                        _ci_iter[_c] = _base_arr.copy()
                                if compartments and _delta is not None:
                                    _first_arr = _ci.get(compartments[0], build_scalar_array(0.0, _A, _R))
                                    _ci_iter[compartments[0]] = np.maximum(_first_arr - _delta, 0.0)

                        _has_per_subpop = any(_per_subpop)
                        if is_metapop:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value.strip(),
                                config_dict, _start_iter, _sim_days, list(compartments),
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_all_tvs_ar,
                                save_daily=_has_any_comp_ar,
                                param_overrides=_sampled or None,
                                param_overrides_per_subpop=_per_subpop if _has_per_subpop else None,
                                travel_config=metapop_travel_config or None,
                                init_states_override=_init_override,
                            )
                        else:
                            _ci_run = _ci_iter if _ci_iter is not None else _ci
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start_iter, _sim_days, _ci_run,
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_all_tvs_ar,
                                save_daily=_has_any_comp_ar,
                                param_overrides=_sampled or None,
                                travel_config=metapop_travel_config,
                            )
                        _m.simulate_until_day(_sim_days)

                        _weighted_r2 = 0.0
                        _per_target_trajs = {}
                        _skip = False
                        for _k in range(_n):
                            _mode_k = _target_modes[_k]
                            _w_k = _target_weights[_k] / _weight_sum
                            _n_days_k = _num_fit_days_per_target[_k]

                            if _mode_k == "scalar":
                                _row_sims = _extract_scalar_rows(
                                    _m, _target_vars_list[_k], fit_obs_arrays[_k], _n_days_k,
                                    sp_fallback=_sp_idxs[_k], ag_fallback=_age_idxs[_k],
                                    rk_fallback=_risk_idxs[_k],
                                )
                                _obs_vals = [_row["value"] for _row in fit_obs_arrays[_k]]
                                _r2_k = compute_rsquared(_obs_vals, _row_sims)
                                _per_target_trajs[f"target_{_k}"] = _row_sims
                            elif _mode_k == "proportion":
                                _row_sims = _extract_proportion_rows(
                                    _m, _target_vars_list[_k], fit_obs_arrays[_k], _n_days_k,
                                    sp_fallback=_sp_idxs[_k], ag_fallback=_age_idxs[_k],
                                    rk_fallback=_risk_idxs[_k],
                                )
                                _obs_vals = [_row["value"] for _row in fit_obs_arrays[_k]]
                                _r2_k = compute_rsquared(_obs_vals, _row_sims)
                                _per_target_trajs[f"target_{_k}"] = _row_sims
                            else:
                                _sim_ts = _extract_ts_sliced(
                                    _m, _target_vars_list[_k],
                                    _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k],
                                    _n_days_k,
                                )
                                _obs_k = fit_obs_arrays[_k][:_n_days_k]
                                if len(_sim_ts) != len(_obs_k):
                                    _skip = True
                                    break
                                _r2_k = compute_rsquared(list(_obs_k), _sim_ts.tolist())
                                _per_target_trajs[f"target_{_k}"] = _sim_ts.tolist()

                            _weighted_r2 += _w_k * _r2_k

                        if _skip:
                            continue

                        _loss_curve.append(float(_weighted_r2))
                        if _weighted_r2 > _best_r2:
                            _best_r2 = _weighted_r2
                            _best_params = dict(_sampled)
                            for _spi, _sp_d in enumerate(_per_subpop):
                                for _kk, _vv in _sp_d.items():
                                    _best_params[f"{_kk}_subpop{_spi}"] = _vv
                            _best_params.update({f"seed_scale_{_c}": _v for _c, _v in _sampled_scales.items()})
                            if _fit_start_offset:
                                _best_params["epidemic_start_offset_days"] = _sampled_offset
                                _best_params["epidemic_start_date"] = _start_iter
                            _best_trajs = dict(_per_target_trajs)
                        if _weighted_r2 >= _r2_thresh:
                            _acc = dict(_sampled)
                            for _spi, _sp_d in enumerate(_per_subpop):
                                for _kk, _vv in _sp_d.items():
                                    _acc[f"{_kk}_subpop{_spi}"] = _vv
                            _acc.update({f"seed_scale_{_c}": _v for _c, _v in _sampled_scales.items()})
                            if _fit_start_offset:
                                _acc["epidemic_start_offset_days"] = _sampled_offset
                                _acc["epidemic_start_date"] = _start_iter
                            _accepted_params_list.append(_acc)
                            _accepted_trajectories.append(dict(_per_target_trajs))
    
            except Exception as _exc:
                import traceback as _tb
                mo.stop(
                    True,
                    mo.callout(
                        mo.md(f"**Fitting error:** {_exc}\n\n```\n{_tb.format_exc()}\n```"),
                        kind="danger",
                    ),
                )
    
        _sim_trajectories = (
            _accepted_trajectories if _accepted_trajectories
            else ([_best_trajs] if _best_trajs else [])
        )
        _is_ar = fit_method.value == "ar"
        fit_result = FitResult(
            best_params=_best_params,
            loss_curve=_loss_curve,
            num_days=_sim_days,
            observed=[list(o) if isinstance(o, np.ndarray) else o for o in fit_obs_arrays],
            method=fit_method.value,
            accepted_params=_accepted_params_list if _accepted_params_list else [_best_params],
            sim_trajectories=_sim_trajectories,
            fit_targets=_target_vars_list,
            target_labels=_target_labels,
            target_weights=_target_weights,
            target_modes=_target_modes,
            r2_threshold=_r2_thresh if _is_ar else None,
            n_ar_accepted=len(_accepted_params_list) if _is_ar else None,
        )
    return (fit_result,)


@app.cell
def _fitting_autosave(fit_result, output_dir, json):
    if fit_result is not None:
        _p = output_dir / "fitted_params.json"
        _p.write_text(json.dumps(fit_result.best_params, indent=2))
    return


@app.cell
def _fitting_results_display(fit_result, np, plt, mo, main_tab, json):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md("*Run fitting to see results.*"))
    _lc = fit_result.loss_curve
    _bp = fit_result.best_params
    _method = fit_result.method
    _accepted = fit_result.accepted_params or []
    _n_runs = len(_accepted)

    def _fmt_val(v):
        if isinstance(v, (int, float)):
            return f"{v:.6g}"
        return str(v)

    # Loss / progress plot
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    if _method == "ar":
        _axes[0].plot(_lc, linewidth=1.5, label="Weighted R²")
        _r2_thr = fit_result.r2_threshold
        if _r2_thr is not None:
            _axes[0].axhline(
                y=_r2_thr, color="red", linestyle="--", linewidth=1.2,
                alpha=0.8, label=f"Threshold ({_r2_thr:.2f})",
            )
            _axes[0].legend(fontsize=8)
        _axes[0].set_ylabel("Weighted R²")
    else:
        # _lc is list of lists — one per replication, each starting at iteration 0
        _n_rep_lc = len(_lc)
        _alpha = min(0.9, max(0.3, 3.0 / max(_n_rep_lc, 1)))
        for _ri, _rep_lc in enumerate(_lc):
            _axes[0].plot(
                _rep_lc,
                linewidth=1.2,
                alpha=_alpha,
                label=f"Rep {_ri + 1}" if _n_rep_lc <= 10 else None,
            )
        if 1 < _n_rep_lc <= 10:
            _axes[0].legend(fontsize=8)
        _axes[0].set_ylabel("Weighted MSE loss")
    _axes[0].set_xlabel("Iterations / Max samples")
    _axes[0].set_title(f"Fitting progress ({_method})")
    _axes[0].grid(True, alpha=0.3)

    # Parameter table — two columns when multiple runs available
    _axes[1].axis("off")
    if _bp:
        if _n_runs > 1:
            _rows = []
            for _pn, _bv in _bp.items():
                _all_vals = [_s.get(_pn) for _s in _accepted if isinstance(_s.get(_pn), (int, float))]
                if _all_vals:
                    _arr = np.array(_all_vals, dtype=float)
                    _med = np.median(_arr)
                    _lo95 = np.percentile(_arr, 2.5)
                    _hi95 = np.percentile(_arr, 97.5)
                    _stat_str = f"{_med:.4g} [{_lo95:.4g}, {_hi95:.4g}]"
                else:
                    _stat_str = "—"
                _rows.append([_pn, _fmt_val(_bv), _stat_str])
            _col_labels = ["Parameter", "Best-fit", "Median [2.5%, 97.5%]"]
        else:
            _rows = [[_pn, _fmt_val(_v)] for _pn, _v in _bp.items()]
            _col_labels = ["Parameter", "Best-fit value"]
        _tbl = _axes[1].table(
            cellText=_rows, colLabels=_col_labels,
            loc="center", cellLoc="left",
        )
        _tbl.auto_set_font_size(True)
        _tbl.scale(1.2, 1.5)
        _axes[1].set_title("Best-fit parameters")
    plt.tight_layout()

    _params_md = "\n".join(f"- `{k}` = **{_fmt_val(v)}**" for k, v in _bp.items())

    _accepted_note = mo.md("")
    _ar_multi_note = mo.md("")
    if _method == "ar":
        _n_acc = fit_result.n_ar_accepted if fit_result.n_ar_accepted is not None else _n_runs
        if _n_acc == 0:
            _accepted_note = mo.callout(
                mo.md(
                    f"**No parameter sets passed the weighted R² threshold "
                    f"({fit_result.r2_threshold:.2f}).** Showing best-fit only. "
                    "Lower the threshold or increase the number of samples."
                ),
                kind="warn",
            )
        else:
            _accepted_note = mo.callout(
                mo.md(
                    f"**{_n_acc} accepted parameter set(s)** passed the weighted R² threshold. "
                    "Enable **Start forecast from fitted end-state** in the Forecast tab to run an ensemble."
                ),
                kind="success" if _n_acc > 1 else "info",
            )
        if len(fit_result.target_modes or []) > 1:
            _ar_multi_note = mo.callout(
                mo.md(
                    "**Accept-reject with multiple targets:** AR samples parameters randomly and "
                    "accepts only those where the *combined* weighted R² clears the threshold. "
                    "With multiple targets the joint probability of a random sample satisfying all "
                    "objectives simultaneously is low — acceptance rates drop sharply. "
                    "Use **Adam** or **L-BFGS** for reliable multi-target fitting."
                ),
                kind="warn",
            )
    elif _method in ("adam", "lbfgs") and _n_runs > 1:
        _accepted_note = mo.callout(
            mo.md(
                f"**{_n_runs} replications** completed with LHS starting points. "
                "Best-fit is the replication with lowest final loss. "
                "Enable **Start forecast from fitted end-state** in the Forecast tab to run an ensemble."
            ),
            kind="info",
        )

    _download = mo.download(
        data=json.dumps({
            "best_params": fit_result.best_params,
            "num_days": fit_result.num_days,
            "method": fit_result.method,
            "accepted_params": fit_result.accepted_params,
        }, indent=2).encode(),
        filename="fitted_params.json",
        mimetype="application/json",
        label="Download fitted_params.json",
    )
    mo.vstack([mo.md("## Fitting Results"), _fig, mo.md(_params_md), _accepted_note, _ar_multi_note, _download])
    return


@app.cell
def _fitting_comparison_ui(mo):
    fit_comparison_style = mo.ui.radio(
        options={"Spaghetti lines": "spaghetti", "Median + 95% CI": "band"},
        value="Median + 95% CI",
        label="Display accepted parameter sets as",
    )
    return (fit_comparison_style,)


@app.cell
def _fitting_comparison_display(
    fit_result, fit_comparison_style,
    np, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))

    _method = fit_result.method
    _trajs = fit_result.sim_trajectories
    _n_tgts = len(fit_result.fit_targets)
    _labels = fit_result.target_labels if fit_result.target_labels else [f"Target {_k+1}" for _k in range(_n_tgts)]
    _modes = list(fit_result.target_modes) if fit_result.target_modes else []
    if not _modes:
        for _k in range(_n_tgts):
            _obs_k = fit_result.observed[_k]
            _is_dict_rows = isinstance(_obs_k, list) and _obs_k and isinstance(_obs_k[0], dict)
            _modes.append("scalar" if _is_dict_rows else "ts")

    _style_ui = mo.md("")
    if _method == "ar" or (_method in ("adam", "lbfgs") and len(_trajs) > 1):
        _style_ui = mo.hstack([fit_comparison_style], justify="start")

    _figs = []
    for _k in range(_n_tgts):
        _label = _labels[_k]
        _obs_k = fit_result.observed[_k]
        _mode_k = _modes[_k]
        _traj_key = f"target_{_k}"

        if _mode_k in ("scalar", "proportion"):
            # Bar chart: observed vs simulated per row
            _obs_vals = [_row["value"] for _row in _obs_k]
            _sim_vals = [
                _traj[_traj_key][_ri] if _traj_key in _traj and _ri < len(_traj[_traj_key]) else 0.0
                for _traj in (_trajs[:1] if _trajs else [{}])
                for _ri in range(len(_obs_vals))
            ]
            # For AR with multiple accepted: show range
            if _method == "ar" and len(_trajs) > 1:
                _all_sim = np.array([
                    [_t[_traj_key][_ri] if _traj_key in _t and _ri < len(_t[_traj_key]) else 0.0
                     for _ri in range(len(_obs_vals))]
                    for _t in _trajs
                ])
                _sim_med = np.median(_all_sim, axis=0)
                _sim_lo = np.percentile(_all_sim, 2.5, axis=0)
                _sim_hi = np.percentile(_all_sim, 97.5, axis=0)
            else:
                _sim_med = np.array(_sim_vals[:len(_obs_vals)])
                _sim_lo = _sim_med
                _sim_hi = _sim_med

            _row_labels = []
            for _ri, _row in enumerate(_obs_k):
                _parts = []
                for _col in ("subpopulation", "age", "risk"):
                    if _col in _row:
                        _parts.append(f"{_col}={_row[_col]}")
                _row_labels.append(", ".join(_parts) if _parts else f"row {_ri}")

            _x = np.arange(len(_obs_vals))
            _w = 0.35
            _fig_k, _ax_k = plt.subplots(figsize=(max(6, len(_obs_vals) * 1.2 + 1), 4))
            _ax_k.bar(_x - _w / 2, _obs_vals, _w, label="Observed", color="k", alpha=0.7)
            _ax_k.bar(_x + _w / 2, _sim_med, _w, label="Simulated", color="steelblue", alpha=0.8)
            if _method == "ar" and len(_trajs) > 1:
                _ax_k.errorbar(
                    _x + _w / 2, _sim_med,
                    yerr=[_sim_med - _sim_lo, _sim_hi - _sim_med],
                    fmt="none", color="steelblue", capsize=4,
                )
            _ax_k.set_xticks(_x)
            _ax_k.set_xticklabels(_row_labels, rotation=30, ha="right", fontsize=9)
            _ax_k.set_ylabel("Proportion" if _mode_k == "proportion" else "Total count")
            _ax_k.set_title(f"{_label} ({'proportion' if _mode_k == 'proportion' else 'scalar total'})")
            _ax_k.legend()
            _ax_k.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _figs.append(_fig_k)

        else:
            # Timeseries plot
            _obs_arr = np.array(_obs_k)
            _num_days = len(_obs_arr)
            _days = np.arange(_num_days)
            _fig_k, _ax_k = plt.subplots(figsize=(10, 4))

            _valid_trajs = [_t[_traj_key] for _t in _trajs if _traj_key in _t]
            _n_valid = len(_valid_trajs)
            if _method in ("adam", "lbfgs") and _n_valid <= 1:
                # Single replication: just draw the fitted curve
                if _valid_trajs:
                    _sim = np.array(_valid_trajs[0])
                    _ax_k.plot(_days, _sim[:_num_days], color="steelblue", linewidth=2, label="Fitted")
                _ax_k.plot(_days, _obs_arr, "ko", markersize=4, label="Observed", zorder=5)
                _ax_k.set_title(f"{_label} — Fitted vs Observed")
            else:
                # Multiple replications or AR accepted runs: spaghetti or band
                _style = fit_comparison_style.value
                _run_label = "accepted" if _method == "ar" else "replication"
                if _valid_trajs:
                    _trajs_arr = np.array([_t[:_num_days] for _t in _valid_trajs])
                    if _style == "spaghetti":
                        for _traj in _trajs_arr:
                            _ax_k.plot(
                                _days, _traj,
                                color="steelblue",
                                alpha=min(1.0, 3.0 / len(_trajs_arr)),
                                linewidth=1,
                            )
                        _ax_k.plot([], [], color="steelblue", alpha=0.6, linewidth=1.5,
                                   label=f"{_n_valid} {_run_label}(s)")
                    else:
                        _med = np.median(_trajs_arr, axis=0)
                        _lo = np.percentile(_trajs_arr, 2.5, axis=0)
                        _hi = np.percentile(_trajs_arr, 97.5, axis=0)
                        _ax_k.fill_between(_days, _lo, _hi, color="steelblue", alpha=0.25, label="95% CI")
                        _ax_k.plot(_days, _med, color="steelblue", linewidth=2, label="Median")
                _ax_k.plot(_days, _obs_arr, "ko", markersize=4, label="Observed", zorder=5)
                _ax_k.set_title(f"{_label} ({_n_valid} {_run_label}(s))")

            _ax_k.set_xlabel("Day")
            _ax_k.set_ylabel(_label)
            _ax_k.legend()
            _ax_k.grid(True, alpha=0.3)
            plt.tight_layout()
            _figs.append(_fig_k)

    mo.vstack([
        mo.md("### Fitted vs Observed"),
        _style_ui,
        *_figs,
    ])
    return


@app.cell
def _fitting_pairplot(fit_result, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))

    _accepted = fit_result.accepted_params
    mo.stop(not _accepted or len(_accepted) <= 1, mo.md(""))

    _first = _accepted[0]
    _scalar_keys = [_k for _k, _v in _first.items() if isinstance(_v, (int, float))]
    mo.stop(not _scalar_keys, mo.md(""))

    _data = np.array([[float(_s.get(_k, float("nan"))) for _k in _scalar_keys] for _s in _accepted])
    _n = len(_scalar_keys)

    def _draw_density(_ax, _vals):
        _vals = _vals[np.isfinite(_vals)]
        if len(_vals) < 2:
            return
        _ax.hist(_vals, bins=max(10, len(_vals) // 5), density=True,
                 color="steelblue", alpha=0.65, edgecolor="white", linewidth=0.4)
        try:
            from scipy.stats import gaussian_kde as _kde
            _xs = np.linspace(_vals.min(), _vals.max(), 300)
            _ax.plot(_xs, _kde(_vals)(_xs), color="navy", linewidth=1.8)
        except Exception:
            pass

    if _n == 1:
        _fig, _ax = plt.subplots(figsize=(5, 4))
        _draw_density(_ax, _data[:, 0])
        _ax.set_xlabel(_scalar_keys[0])
        _ax.set_ylabel("Density")
        _run_noun = "accepted" if fit_result.method == "ar" else "replication"
        _ax.set_title(f"Parameter distribution ({len(_accepted)} {_run_noun}(s))")
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        _run_noun = "accepted" if fit_result.method == "ar" else "replication"
        _fig, _axs = plt.subplots(_n, _n, figsize=(3 * _n, 3 * _n))
        _alpha_sc = min(0.8, 30.0 / max(len(_accepted), 1))
        for _row in range(_n):
            for _col in range(_n):
                _ax = _axs[_row, _col]
                if _row == _col:
                    _draw_density(_ax, _data[:, _row])
                else:
                    _ax.scatter(
                        _data[:, _col], _data[:, _row],
                        alpha=_alpha_sc, s=14,
                        color="steelblue", edgecolors="none",
                    )
                _ax.grid(True, alpha=0.2)
                if _row == _n - 1:
                    _ax.set_xlabel(_scalar_keys[_col], fontsize=9)
                else:
                    _ax.tick_params(labelbottom=False)
                if _col == 0:
                    _ax.set_ylabel(_scalar_keys[_row], fontsize=9)
                else:
                    _ax.tick_params(labelleft=False)
        _fig.suptitle(
            f"Parameter distributions ({len(_accepted)} {_run_noun}(s))",
            y=1.01, fontsize=11,
        )
        plt.tight_layout()

    mo.vstack([
        mo.md("### Accepted Parameter Distributions"),
        _fig,
    ])
    return


# ============================================================
# Forecast tab
# ============================================================

@app.cell
def _forecast_ui(mo):
    forecast_use_fitted = mo.ui.switch(label="Use fitted params from Fitting tab", value=True)
    forecast_params_path = mo.ui.text(
        label="Fitted params JSON path",
        placeholder="~/clt_outputs/fitted_params.json",
        full_width=True,
    )
    forecast_from_fitted_state = mo.ui.switch(
        label="Start forecast from fitted end-state",
        value=False,
    )
    forecast_horizon = mo.ui.number(value=30, start=1, stop=365, step=1, label="Forecast horizon (days)")
    forecast_n_reps = mo.ui.number(value=10, start=1, stop=200, step=1, label="Replicates")
    forecast_stochastic = mo.ui.switch(label="Stochastic simulation", value=True)
    forecast_run_button = mo.ui.run_button(label="Run forecast")
    return (
        forecast_use_fitted, forecast_params_path, forecast_from_fitted_state,
        forecast_horizon, forecast_n_reps, forecast_stochastic, forecast_run_button,
    )


@app.cell
def _forecast_display(
    forecast_use_fitted, forecast_params_path, forecast_from_fitted_state,
    forecast_horizon, forecast_n_reps, forecast_stochastic, forecast_run_button,
    fit_result, mo, main_tab, json, Path,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Forecast", None)
    _ACC = CLT_ACCENT["forecast"]
    _path_w = forecast_params_path if not forecast_use_fitted.value else mo.md("")

    # Multi-set note — shown when AR or gradient replications produced multiple param sets
    _ar_note = mo.md("")
    _is_ar_fitted = False
    _n_accepted = 0
    _fit_method_display = "ar"
    if forecast_use_fitted.value:
        _is_ar_fitted = (
            fit_result is not None
            and len(fit_result.accepted_params) > 1
        )
        if _is_ar_fitted:
            _n_accepted = len(fit_result.accepted_params)
            _fit_method_display = fit_result.method
    else:
        _pp = forecast_params_path.value.strip()
        if _pp:
            try:
                with open(Path(_pp).expanduser()) as _f:
                    _loaded_preview = json.load(_f)
                _loaded_accepted = _loaded_preview.get("accepted_params", [])
                if len(_loaded_accepted) > 1:
                    _is_ar_fitted = True
                    _n_accepted = len(_loaded_accepted)
                    _fit_method_display = _loaded_preview.get("method", "ar")
            except Exception:
                pass

    if _is_ar_fitted:
        _reps_val = int(forecast_n_reps.value)
        _set_noun = (
            "accepted parameter set(s)" if _fit_method_display == "ar"
            else "fitted replication(s)"
        )
        _method_label = (
            "accept-reject" if _fit_method_display == "ar"
            else _fit_method_display
        )
        _set_header = f"**{_n_accepted} {_set_noun}** from {_method_label} fitting."
        if _reps_val < _n_accepted:
            _ar_note = mo.callout(
                mo.md(
                    f"{_set_header} "
                    f"Replicates ({_reps_val}) < {_n_accepted} sets — **{_reps_val}** sets will be "
                    "sampled without replacement, each seeding one trajectory."
                ),
                kind="success",
            )
        else:
            _base = _reps_val // _n_accepted
            _extra = _reps_val % _n_accepted
            if _extra == 0:
                _ar_note_text = (
                    f"{_set_header} "
                    f"Each will seed **{_base}** replicate(s) — **{_reps_val}** trajectories total."
                )
            else:
                _ar_note_text = (
                    f"{_set_header} "
                    f"Each will seed **{_base}** replicate(s); **{_extra}** set(s) sampled without "
                    f"replacement will run one extra — **{_reps_val}** trajectories total."
                )
            _ar_note = mo.callout(mo.md(_ar_note_text), kind="success")

    _fitted_state_note = mo.md("")
    if forecast_from_fitted_state.value:
        if fit_result is not None and forecast_use_fitted.value:
            if not _is_ar_fitted:
                _fitted_state_note = mo.callout(
                    mo.md(
                        "Will run a deterministic warm-up through the fit period, then launch "
                        f"**{forecast_n_reps.value}** stochastic replicate(s) from the fitted end-state."
                    ),
                    kind="info",
                )
        else:
            _fitted_state_note = mo.callout(
                mo.md(
                    "**Start from fitted end-state** requires either:\n\n"
                    "- **Use fitted params from Fitting tab** enabled with fitting already run, or\n"
                    "- A JSON file (via the path field) exported from the Fitting tab, "
                    "which includes `num_days`, `method`, and `accepted_params`."
                ),
                kind="warn",
            )

    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Forecast</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Project the model forward '
            "using fitted parameters.</div>"
        ),
        section_card(
            step_header("①", "Fitted parameters",
                        "Use the fit from the Fitting tab, or load a saved fit JSON.",
                        accent=_ACC),
            mo.vstack([forecast_use_fitted, _path_w]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Settings",
                        "Forecast horizon, replicates, and stochasticity.",
                        accent=_ACC),
            mo.vstack([
                mo.hstack([forecast_horizon, forecast_n_reps], justify="start"),
                forecast_stochastic,
                forecast_from_fitted_state,
                _ar_note,
                _fitted_state_note,
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("③", "Run", "Generate the forecast ensemble.", accent=_ACC),
            forecast_run_button,
            accent=_ACC,
        ),
    ])
    return


@app.cell
def _run_forecast(
    forecast_run_button, forecast_use_fitted, forecast_params_path,
    forecast_from_fitted_state,
    forecast_horizon, forecast_n_reps, forecast_stochastic,
    fit_result, config_dict, compartments, is_metapop,
    metapop_folder_input, metapop_travel_config,
    build_compartment_init, start_date_input, timesteps, rng_seed,
    transition_vars_input,
    make_single_pop_metapop, make_metapop_from_folder, extract_history,
    np, json, mo, Path, build_scalar_array, datetime,
):
    forecast_result = None
    if forecast_run_button.value:
        _fitted_params = {}
        _fit_meta = {"num_days": 0, "method": "ar", "accepted_params": []}
        if forecast_use_fitted.value:
            mo.stop(
                fit_result is None,
                mo.callout(mo.md("**No fitting results.** Run fitting first or disable 'Use fitted params'."), kind="warn"),
            )
            _fitted_params = dict(fit_result.best_params)
            _fit_meta = {
                "num_days": fit_result.num_days,
                "method": fit_result.method,
                "accepted_params": list(fit_result.accepted_params),
            }
        else:
            _pp = forecast_params_path.value.strip()
            if _pp:
                try:
                    with open(Path(_pp).expanduser()) as _f:
                        _loaded = json.load(_f)
                    if "best_params" in _loaded:
                        _fitted_params = _loaded["best_params"]
                        _fit_meta = {
                            "num_days": _loaded.get("num_days", 0),
                            "method": _loaded.get("method", "ar"),
                            "accepted_params": _loaded.get("accepted_params", []),
                        }
                    else:
                        _fitted_params = _loaded
                except Exception as _exc:
                    mo.stop(True, mo.callout(mo.md(f"**Could not load fitted params:** {_exc}"), kind="danger"))

        _fit_n = _fit_meta["num_days"]
        _horizon = int(forecast_horizon.value)
        _total_days = _fit_n + _horizon
        _reps = int(forecast_n_reps.value)
        _stoch = bool(forecast_stochastic.value)
        _start = start_date_input.value.strip() or "2024-01-01"
        _ts = int(timesteps.value)
        _seed_b = int(rng_seed.value)
        _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]

        _ci = None
        if not is_metapop:
            # Initial conditions from the Step 6 tables via config_dict.
            _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
            _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((1, 1))), dtype=float)
            _seed_arrays = {
                _c: np.asarray(_a, dtype=float)
                for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
                if _c in compartments
            }
            _ci, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)

        _histories = []

        # Build param sets and run schedule (shared by both forecast paths)
        _param_sets = _fit_meta["accepted_params"] if _fit_meta["accepted_params"] else [_fitted_params]
        _n_accepted = len(_param_sets)
        _is_ar = _n_accepted > 1  # distribute across AR accepted sets OR gradient replications
        _rng_sched = np.random.default_rng(_seed_b)
        if _is_ar and _reps < _n_accepted:
            _selected = _rng_sched.choice(_n_accepted, size=_reps, replace=False)
            _run_schedule = [(int(_i), 0) for _i in _selected]
        elif _is_ar:
            _base = _reps // _n_accepted
            _extra = _reps % _n_accepted
            _run_schedule = [(_i, _r) for _i in range(_n_accepted) for _r in range(_base)]
            if _extra > 0:
                _extra_idx = _rng_sched.choice(_n_accepted, size=_extra, replace=False)
                _run_schedule += [(int(_i), _base) for _i in _extra_idx]
        else:
            _run_schedule = [(0, _r) for _r in range(_reps)]

        if forecast_from_fitted_state.value:
            # Two-phase: warmup through fit period → extract end-state → run forecast
            mo.stop(
                _fit_n == 0,
                mo.callout(mo.md("**Start from fitted end-state** requires a non-zero fit period. Run fitting first, or load a JSON exported from the Fitting tab (which includes `num_days`)."), kind="warn"),
            )

            _metric_names = [_m["name"] for _m in config_dict.get("epi_metrics", [])]
            _fcast_start = (
                datetime.datetime.strptime(_start, "%Y-%m-%d")
                + datetime.timedelta(days=_fit_n)
            ).strftime("%Y-%m-%d")

            def _extract_end_states(metapop_model, comps, metric_names):
                _states = []
                for _sp in metapop_model.subpop_models.values():
                    _comp = {
                        _c: np.array(_sp.compartments[_c].history_vals_list)[-1]
                        for _c in comps
                    }
                    _epi = {}
                    for _mn in metric_names:
                        try:
                            _h = np.array(_sp.epi_metrics[_mn].history_vals_list)
                            if len(_h) > 0:
                                _epi[_mn] = _h[-1]
                        except Exception:
                            pass
                    _states.append((_comp, _epi))
                return _states

            with mo.status.spinner("Running warmup + forecast from fitted state..."):
                try:
                    _warmup_cache = {}
                    for _traj_idx, (_pset_idx, _rep) in enumerate(_run_schedule):
                        _pset = _param_sets[_pset_idx]

                        # Phase 1: deterministic warmup — cached per param set
                        if _pset_idx not in _warmup_cache:
                            if not is_metapop:
                                _wm, _, _ = make_single_pop_metapop(
                                    config_dict, _start, _fit_n, _ci,
                                    seed_offset=_pset_idx, seed_base=_seed_b, ts_per_day=_ts,
                                    stochastic=False, tvs=_tvs, save_daily=True,
                                    param_overrides=_pset or None,
                                    travel_config=metapop_travel_config,
                                )
                            else:
                                _wm, _ = make_metapop_from_folder(
                                    metapop_folder_input.value, config_dict, _start, _fit_n,
                                    list(compartments),
                                    seed_offset=_pset_idx, seed_base=_seed_b, ts_per_day=_ts,
                                    stochastic=False, tvs=_tvs, save_daily=True,
                                    param_overrides=_pset or None,
                                    travel_config=metapop_travel_config,
                                )
                            _wm.simulate_until_day(_fit_n)
                            _warmup_cache[_pset_idx] = (
                                extract_history(_wm, list(compartments), tvs=_tvs),
                                _extract_end_states(_wm, list(compartments), _metric_names),
                            )
                        _warmup_hist, _end_states = _warmup_cache[_pset_idx]

                        # Phase 2: stochastic forecast from end-state
                        if not is_metapop:
                            _end_comp, _end_epi = _end_states[0]
                            _fm, _, _ = make_single_pop_metapop(
                                config_dict, _fcast_start, _horizon, _end_comp,
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                epi_metric_init=_end_epi or None,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _fm, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _fcast_start, _horizon,
                                list(compartments),
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                                init_states_override=_end_states,
                            )
                        _fm.simulate_until_day(_horizon)
                        _fcast_hist = extract_history(_fm, list(compartments), tvs=_tvs)

                        _combined = {
                            _k: np.concatenate([_warmup_hist[_k], _fcast_hist[_k]])
                            for _k in _warmup_hist
                            if _k in _fcast_hist
                        }
                        _histories.append(_combined)
                except Exception as _exc:
                    mo.stop(True, mo.callout(mo.md(f"**Forecast error:** {_exc}"), kind="danger"))

        else:
            # Standard path: run from Step 7 initial conditions, using run schedule for param sets
            with mo.status.spinner("Running forecast..."):
                try:
                    for _traj_idx, (_pset_idx, _rep) in enumerate(_run_schedule):
                        _pset = _param_sets[_pset_idx]
                        if not is_metapop:
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start, _total_days, _ci,
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _start, _total_days, list(compartments),
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        _m.simulate_until_day(_total_days)
                        _histories.append(extract_history(_m, list(compartments), tvs=_tvs))
                except Exception as _exc:
                    mo.stop(True, mo.callout(mo.md(f"**Forecast error:** {_exc}"), kind="danger"))

        forecast_result = {
            "histories": _histories,
            "fit_n_days": _fit_n,
            "total_days": _total_days,
            "horizon": _horizon,
            "compartments": list(compartments),
            "tvs": _tvs,
        }
    return (forecast_result,)


@app.cell
def _forecast_autosave(forecast_result, output_dir, json):
    if forecast_result is not None:
        _p = output_dir / "forecast_ensemble.json"
        _p.write_text(json.dumps({
            "fit_n_days": forecast_result["fit_n_days"],
            "total_days": forecast_result["total_days"],
            "horizon": forecast_result["horizon"],
            "compartments": forecast_result["compartments"],
            "tvs": forecast_result["tvs"],
            "histories": [
                {k: v.tolist() for k, v in _h.items()}
                for _h in forecast_result["histories"]
            ],
        }, indent=2))
    return


@app.cell
def _forecast_chart_style_ui(mo):
    forecast_chart_style = mo.ui.radio(
        options={"Median + 95% CI": "band", "Spaghetti lines": "spaghetti"},
        value="Median + 95% CI",
        label="Chart style",
    )
    return (forecast_chart_style,)


@app.cell
def _forecast_results_display(forecast_result, forecast_chart_style, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Forecast", None)
    mo.stop(forecast_result is None, mo.md("*Run forecast to see results.*"))
    _hists = forecast_result["histories"]
    _comps = forecast_result["compartments"]
    _fit_n = forecast_result["fit_n_days"]
    _total = forecast_result["total_days"]
    _days = np.arange(1, _total + 1)
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _style = forecast_chart_style.value
    _fig, _ax = plt.subplots(figsize=(12, 5))
    for _ci, _comp in enumerate(_comps):
        _color = _colors[_ci % len(_colors)]
        _arrays = [_h[_comp] for _h in _hists if _comp in _h]
        if not _arrays:
            continue
        _mat = np.stack(_arrays, axis=0)
        _n = min(len(_days), _mat.shape[1])
        if _style == "spaghetti":
            for _row in _mat:
                _ax.plot(_days[:_n], _row[:_n], color=_color, linewidth=0.8, alpha=0.4)
            _med = np.median(_mat[:, :_n], axis=0)
            _ax.plot(_days[:_n], _med, color=_color, linewidth=2, label=f"{_comp} (median)")
        else:
            _med = np.median(_mat[:, :_n], axis=0)
            _lo = np.percentile(_mat[:, :_n], 2.5, axis=0)
            _hi = np.percentile(_mat[:, :_n], 97.5, axis=0)
            _ax.plot(_days[:_n], _med, color=_color, linewidth=2, label=f"{_comp} (median)")
            _ax.fill_between(_days[:_n], _lo, _hi, color=_color, alpha=0.2)
    if _fit_n > 0:
        _ax.axvline(_fit_n, color="black", linestyle="--", alpha=0.6, label=f"Fit end (day {_fit_n})")
        _ax.axvspan(0, _fit_n, alpha=0.05, color="gray")
    _ax.set_xlabel("Day")
    _ax.set_ylabel("Count")
    _ax.set_title("Forecast — Epidemic Curves  (shaded = fit period, right = forecast)")
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _rows = []
    for _comp in _comps:
        _arrays = [_h[_comp] for _h in _hists if _comp in _h]
        if not _arrays:
            continue
        _mat = np.stack(_arrays, axis=0)
        _rows.append(
            f"| `{_comp}` | {float(np.median(np.max(_mat, axis=1))):,.0f} "
            f"| {int(np.median(np.argmax(_mat, axis=1))) + 1} |"
        )
    mo.vstack([
        mo.md("## Forecast Results"),
        forecast_chart_style,
        _fig,
        mo.md(
            "### Summary\n\n"
            "| Compartment | Peak (median) | Peak day (median) |\n|---|---|---|\n"
            + "\n".join(_rows)
        ),
    ])
    return


# ============================================================
# Export tab
# ============================================================

@app.cell
def _export_display(config_dict, fit_result, output_dir, json, mo, main_tab, num_age_groups, num_risk_groups, sim_days, sim_mode, n_reps, timesteps, start_date_input, analysis_scenarios, step_header, section_card, CLT_ACCENT):
    mo.stop(main_tab.value != "Export", None)
    _ACC = CLT_ACCENT["export"]
    _config_str = json.dumps(config_dict, indent=2)
    _fitted_str = json.dumps(fit_result.best_params if fit_result is not None else {}, indent=2)

    _script = """\
#!/usr/bin/env python3
\"\"\"
Generated by CLT Model Builder Notebook.
Usage: python run_simulation.py
\"\"\"

import sys
import json
import copy
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from types import SimpleNamespace

# ---- Configurable ----
MODEL_CONFIG_FILE = "model_config.json"
FITTED_PARAMS_FILE = "fitted_params.json"  # set to None to skip
OUTPUT_DIR = Path("simulation_output")
NUM_DAYS = 100
NUM_REPS = 1
STOCHASTIC = False
TIMESTEPS_PER_DAY = 7
START_DATE = "2024-01-01"
NUM_AGE_GROUPS = 1
NUM_RISK_GROUPS = 1

# <<<SCENARIOS_BLOCK>>>

# <<<SUBPOP_OVERRIDES_BLOCK>>>

# ---- Setup ----
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))

import clt_toolkit as clt
import flu_core as flu
from generic_core.config_parser import parse_model_config_from_dict
from generic_core.generic_model import (
    ConfigDrivenSubpopModel, build_state_from_config, build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel

with open(_HERE / MODEL_CONFIG_FILE) as _f:
    config_dict = json.load(_f)

if FITTED_PARAMS_FILE is not None:
    _fp = _HERE / FITTED_PARAMS_FILE
    if _fp.exists():
        with open(_fp) as _f:
            _fitted = json.load(_f)
        config_dict["params"] = {**config_dict.get("params", {}), **_fitted}
    else:
        print(f"Warning: {FITTED_PARAMS_FILE} not found")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_schedules(start_date, num_days):
    _h = max(num_days + 14, 370)
    _dates = pd.date_range(start=start_date, periods=_h, freq="D").date
    _mob = json.dumps(np.ones((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    _vax = json.dumps(np.zeros((NUM_AGE_GROUPS, NUM_RISK_GROUPS)).tolist())
    return SimpleNamespace(
        absolute_humidity_df=pd.DataFrame({"date": _dates, "absolute_humidity": [0.01] * _h}),
        school_work_calendar_df=pd.DataFrame({"date": _dates, "is_school_day": [1.0] * _h, "is_work_day": [1.0] * _h}),
        mobility_df=pd.DataFrame({"day_of_week": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"], "mobility_modifier": [_mob]*7}),
        daily_vaccines_df=pd.DataFrame({"date": _dates, "daily_vaccines": [_vax] * _h}),
    )


def build_model(cfg, param_overrides, rep, subpop_overrides=None):
    # subpop_overrides: list of {param: value} indexed by subpop order (metapop only)
    _cfg = copy.deepcopy(cfg)
    if param_overrides:
        _cfg["params"] = {**_cfg.get("params", {}), **param_overrides}
    _sched = _build_schedules(START_DATE, NUM_DAYS)
    _mc = parse_model_config_from_dict(_cfg, schedules_input=_sched)
    _A, _R = NUM_AGE_GROUPS, NUM_RISK_GROUPS
    _comps = list(_cfg.get("compartments", {}).keys()) if isinstance(_cfg.get("compartments"), dict) else list(_cfg.get("compartments", ["S"]))
    _first = _comps[0] if _comps else "S"
    _N = _cfg.get("total_population", 100000)
    _comp_init = {_first: np.full((_A, _R), float(_N))}
    for _c in _comps[1:]:
        _comp_init.setdefault(_c, np.zeros((_A, _R)))
    _state = build_state_from_config(_mc, _comp_init, epi_metric_init={})
    _params = build_params_from_config(_mc, num_age_groups=_A, num_risk_groups=_R)
    _tt = clt.TransitionTypes.BINOM if STOCHASTIC else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    _settings = clt.SimulationSettings(
        timesteps_per_day=TIMESTEPS_PER_DAY, transition_type=_tt,
        start_real_date=START_DATE, save_daily_history=True,
    )
    _subpop = ConfigDrivenSubpopModel(
        model_config=_mc, state_init=_state, params=_params,
        simulation_settings=_settings, RNG=np.random.default_rng(rep),
        schedules_input=_sched, name="pop",
    )
    _mixing = flu.FluMixingParams(travel_proportions=np.array([[1.0]]), num_locations=1)
    return ConfigDrivenMetapopModel({"pop": _subpop}, mixing_params=_mixing)


all_results = {}
for scenario_name, overrides in SCENARIOS.items():
    print(f"Running scenario: {scenario_name}")
    _sp_overrides = SUBPOP_PARAM_OVERRIDES.get(scenario_name)
    _reps_data = []
    for _rep in range(NUM_REPS):
        _m = build_model(config_dict, overrides, _rep, subpop_overrides=_sp_overrides)
        _m.simulate_until_day(NUM_DAYS)
        _sps = list(_m.subpop_models.values())
        _h = {}
        for _sp in _sps:
            for _c, _comp in _sp.compartments.items():
                _arr = np.array(_comp.history_vals_list).sum(axis=(1, 2))
                _h[_c] = _h.get(_c, 0) + _arr
        _reps_data.append(_h)
    all_results[scenario_name] = _reps_data

_db = OUTPUT_DIR / "results.db"
_con = sqlite3.connect(_db)
_cur = _con.cursor()
_cur.execute(
    "CREATE TABLE IF NOT EXISTS results "
    "(scenario TEXT, rep INTEGER, compartment TEXT, day INTEGER, value REAL)"
)
for _scen, _reps_data in all_results.items():
    for _ri, _h in enumerate(_reps_data):
        for _c, _arr in _h.items():
            _cur.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?)",
                [(_scen, _ri, _c, _d + 1, float(_v)) for _d, _v in enumerate(_arr)],
            )
_con.commit()
_con.close()
print(f"Results saved to {_db}")
"""

    _scen_lines = [
        "# Define scenarios: {name: {param: value}}",
        "SCENARIOS = {",
        '    "baseline": {},',
    ]
    _subpop_lines = [
        "# Per-subpopulation parameter overrides per scenario (metapop only)",
        "# Format: {scenario_name: [override_dict_or_None, ...]} indexed by subpop order",
        "SUBPOP_PARAM_OVERRIDES = {",
    ]
    for _scen_tuple in analysis_scenarios:
        _sn, _ov = _scen_tuple[0], _scen_tuple[1]
        _sp_list = _scen_tuple[2] if len(_scen_tuple) > 2 else None
        if _ov and _sn != "baseline":
            _inner = ", ".join(f'"{_k}": {repr(_v)}' for _k, _v in _ov.items())
            _scen_lines.append(f"    {repr(_sn)}: {{{_inner}}},")
        if _sp_list and any(_r for _r in _sp_list):
            _subpop_lines.append(f"    {repr(_sn)}: {repr(_sp_list)},")
    _scen_lines.append("}")
    _subpop_lines.append("}")
    _scenarios_block = "\n".join(_scen_lines)
    _subpop_block = "\n".join(_subpop_lines)

    _script = _script.replace("# <<<SCENARIOS_BLOCK>>>", _scenarios_block).replace(
        "# <<<SUBPOP_OVERRIDES_BLOCK>>>", _subpop_block,
    ).replace(
        "NUM_DAYS = 100\n",
        f"NUM_DAYS = {int(sim_days.value)}\n",
    ).replace(
        "NUM_REPS = 1\n",
        f"NUM_REPS = {int(n_reps.value)}\n",
    ).replace(
        "STOCHASTIC = False\n",
        f"STOCHASTIC = {sim_mode.value == 'Stochastic'}\n",
    ).replace(
        "TIMESTEPS_PER_DAY = 7\n",
        f"TIMESTEPS_PER_DAY = {int(timesteps.value)}\n",
    ).replace(
        'START_DATE = "2024-01-01"\n',
        f'START_DATE = "{start_date_input.value}"\n',
    ).replace(
        "NUM_AGE_GROUPS = 1\n",
        f"NUM_AGE_GROUPS = {num_age_groups}\n",
    ).replace(
        "NUM_RISK_GROUPS = 1\n",
        f"NUM_RISK_GROUPS = {num_risk_groups}\n",
    )

    _config_dl = mo.download(
        data=_config_str.encode(), filename="model_config.json",
        label="Download model_config.json", mimetype="application/json",
    )
    _script_dl = mo.download(
        data=_script.encode(), filename="run_simulation.py",
        label="Download run_simulation.py", mimetype="text/x-python",
    )
    _fitted_dl = mo.download(
        data=_fitted_str.encode(), filename="fitted_params.json",
        label="Download fitted_params.json", mimetype="application/json",
    )
    _n_analysis = len(analysis_scenarios)
    _n_sp_overrides = sum(
        1 for _t in analysis_scenarios
        if len(_t) > 2 and _t[2] and any(_r for _r in _t[2])
    )
    if _n_analysis:
        _sp_note_extra = (
            f" `SUBPOP_PARAM_OVERRIDES` includes **{_n_sp_overrides}** scenario(s) with per-subpop overrides."
            if _n_sp_overrides else ""
        )
        _scen_note = mo.callout(
            mo.md(
                f"`SCENARIOS` pre-populated from the Analysis tab "
                f"(**{_n_analysis}** scenario(s) + baseline). "
                f"Edit the script to add or change scenarios.{_sp_note_extra}"
            ),
            kind="info",
        )
    else:
        _scen_note = mo.callout(
            mo.md(
                "No Analysis scenarios defined — only `baseline` is included in `SCENARIOS`. "
                "Define scenarios in the Analysis tab to pre-populate this block."
            ),
            kind="info",
        )

    _how_to = mo.callout(
        mo.md(
            "**How to run**\n\n"
            "1. Download `run_simulation.py`, `model_config.json`, and "
            "`fitted_params.json` below into one folder.\n"
            "2. (Optional) edit the `SCENARIOS` block and top constants in the script.\n"
            "3. Run `python run_simulation.py` — results are written to a "
            "`simulation_output/` folder alongside the script."
        ),
        kind="info",
    )
    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Export</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Generate a standalone '
            "script to run this model outside the notebook.</div>"
        ),
        section_card(
            step_header("①", "Generated script",
                        "A runnable run_simulation.py — edit SCENARIOS and the top "
                        "constants before running.",
                        accent=_ACC),
            mo.vstack([
                _how_to,
                _scen_note,
                mo.accordion({"run_simulation.py": mo.md(f"```python\n{_script}\n```")}),
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Downloads",
                        "Grab the script and its input files.", accent=_ACC),
            mo.vstack([
                mo.hstack([_config_dl, _script_dl, _fitted_dl], justify="start"),
                mo.md(f"*Outputs auto-saved to `{output_dir}/`*"),
            ]),
            accent=_ACC,
        ),
    ])
    return


# ============================================================
# Analysis tab — sub-tab selector (must depend only on mo)
# ============================================================

@app.cell
def _analysis_sub_tab(mo):
    analysis_sub_tab = mo.ui.tabs({"Sensitivity": mo.md(""), "Scenario": mo.md("")})
    return (analysis_sub_tab,)


@app.cell
def _analysis_param_catalog(
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    param_grid_columns, num_age_groups, num_risk_groups, age_groups,
):
    import math as _math

    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)
    _params = {}
    for _name in param_names:
        if param_vary_toggles[_name].value:
            _rows = list(param_grid_inputs[_name].value)
            _params[_name] = [
                [float(_rows[_r][_age_cols[_a]]) for _r in range(_R)]
                for _a in range(_A)
            ]
        else:
            _params[_name] = float(param_scalar_inputs[_name].value)

    ANALYSIS_SCALAR_PARAMS = {k: v for k, v in _params.items() if isinstance(v, (int, float))}
    ANALYSIS_ARRAY_PARAMS = {k: v for k, v in _params.items() if isinstance(v, list)}

    def _slider_range(val):
        if val == 0.0:
            return 0.0, 10.0, 0.01
        mag = 10 ** _math.floor(_math.log10(abs(val)))
        step = round(mag / 100, 10)
        hi = round(max(val * 5, mag * 10), 10)
        return 0.0, hi, step

    ANALYSIS_SCALAR_RANGES = {k: _slider_range(v) for k, v in ANALYSIS_SCALAR_PARAMS.items()}
    return ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, ANALYSIS_SCALAR_RANGES


@app.cell
def _analysis_subpop_names(is_metapop, metapop_folder_input, Path, json):
    _sp_names = []
    if is_metapop and metapop_folder_input.value.strip():
        _mc_path = Path(metapop_folder_input.value.strip()) / "metapop_config.json"
        if _mc_path.exists():
            try:
                with open(_mc_path) as _f:
                    _mc_cfg = json.load(_f)
                _sp_names = list(_mc_cfg.get("subpopulations", []))
            except Exception:
                pass
    analysis_sp_names = _sp_names
    return (analysis_sp_names,)


@app.cell
def _analysis_sensitivity_controls(mo, ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS):
    _scalar_opts = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_opts = list(ANALYSIS_ARRAY_PARAMS.keys())
    analysis_param_type = mo.ui.radio(
        options=["Scalar", "Array (scale factor)"],
        value="Scalar",
        label="Parameter type",
    )
    analysis_scalar_param_sel = mo.ui.dropdown(
        options=_scalar_opts if _scalar_opts else ["(none)"],
        value=_scalar_opts[0] if _scalar_opts else "(none)",
        label="Scalar parameter to vary",
    )
    analysis_array_param_sel = mo.ui.dropdown(
        options=_array_opts if _array_opts else ["(none)"],
        value=_array_opts[0] if _array_opts else "(none)",
        label="Array parameter to scale",
    )
    analysis_n_values = mo.ui.number(start=1, stop=6, step=1, value=3, label="Values to compare")
    return analysis_param_type, analysis_scalar_param_sel, analysis_array_param_sel, analysis_n_values


@app.cell
def _analysis_sensitivity_sliders(
    mo, analysis_param_type,
    analysis_scalar_param_sel, analysis_array_param_sel,
    analysis_n_values, ANALYSIS_SCALAR_PARAMS, ANALYSIS_SCALAR_RANGES,
):
    _n = int(analysis_n_values.value)
    if analysis_param_type.value == "Array (scale factor)":
        analysis_sens_sliders = mo.ui.array([
            mo.ui.slider(start=0.1, stop=3.0, step=0.05, value=1.0, label=f"scale {i + 1}")
            for i in range(_n)
        ])
    else:
        _pname = analysis_scalar_param_sel.value
        if _pname in ANALYSIS_SCALAR_RANGES:
            _lo, _hi, _step = ANALYSIS_SCALAR_RANGES[_pname]
            _base = ANALYSIS_SCALAR_PARAMS.get(_pname, 1.0)
        else:
            _lo, _hi, _step, _base = 0.0, 10.0, 0.01, 1.0
        analysis_sens_sliders = mo.ui.array([
            mo.ui.slider(start=_lo, stop=_hi, step=_step, value=_base, label=f"value {i + 1}")
            for i in range(_n)
        ])
    return (analysis_sens_sliders,)


@app.cell
def _analysis_scenario_controls(mo, ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS):
    _MAX_SC = 5
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())

    analysis_n_scenarios = mo.ui.number(start=1, stop=5, step=1, value=2, label="Number of scenarios")
    analysis_scenario_names = mo.ui.array([
        mo.ui.text(value=f"Scenario {j + 1}", label=f"Name {j + 1}")
        for j in range(_MAX_SC)
    ])

    def _make_scalar_input(pname):
        _base = float(ANALYSIS_SCALAR_PARAMS.get(pname, 1.0))
        _stop = max(10.0, _base * 20) if _base > 0 else 10.0
        return mo.ui.number(start=0.0, stop=_stop, step=None, value=_base)

    analysis_scenario_scalar_inputs = mo.ui.array([
        mo.ui.array([_make_scalar_input(pname) for _ in range(_MAX_SC)])
        for pname in _scalar_names
    ]) if _scalar_names else mo.ui.array([])

    analysis_scenario_array_scales = mo.ui.array([
        mo.ui.array([mo.ui.number(start=0.0, stop=10.0, step=None, value=1.0) for _ in range(_MAX_SC)])
        for _ in _array_names
    ]) if _array_names else mo.ui.array([])

    return (
        analysis_n_scenarios, analysis_scenario_names,
        analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    )


@app.cell
def _analysis_param_subpop_controls(
    mo, is_metapop, analysis_sp_names,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, ANALYSIS_SCALAR_RANGES,
    analysis_scalar_param_sel, analysis_array_param_sel, analysis_param_type,
):
    _MAX_SC = 5
    _MAX_SENS = 6
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _sp_opts = list(analysis_sp_names) if analysis_sp_names else []

    # Scenario sub-tab: per-param subpop multiselects
    analysis_scalar_subpop_sels = mo.ui.array([
        mo.ui.multiselect(options=_sp_opts, value=[], label=f"Subpops for {pname}")
        for pname in _scalar_names
    ]) if is_metapop and _scalar_names and _sp_opts else mo.ui.array([])

    analysis_array_subpop_sels = mo.ui.array([
        mo.ui.multiselect(options=_sp_opts, value=[], label=f"Subpops for {pname}")
        for pname in _array_names
    ]) if is_metapop and _array_names and _sp_opts else mo.ui.array([])

    # Scenario sub-tab: per-param × per-subpop × per-scenario number inputs [param][subpop][scenario]
    def _make_sp_scalar_input(pname):
        _base = float(ANALYSIS_SCALAR_PARAMS.get(pname, 1.0))
        _lo, _hi, _step = ANALYSIS_SCALAR_RANGES.get(pname, (0.0, 10.0, 0.001))
        return mo.ui.number(start=_lo, stop=_hi, step=None, value=_base)

    analysis_scalar_subpop_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.array([_make_sp_scalar_input(pname) for _ in range(_MAX_SC)])
            for _ in _sp_opts
        ])
        for pname in _scalar_names
    ]) if is_metapop and _scalar_names and _sp_opts else mo.ui.array([])

    analysis_array_subpop_scales = mo.ui.array([
        mo.ui.array([
            mo.ui.array([mo.ui.number(start=0.0, stop=10.0, step=None, value=1.0) for _ in range(_MAX_SC)])
            for _ in _sp_opts
        ])
        for _ in _array_names
    ]) if is_metapop and _array_names and _sp_opts else mo.ui.array([])

    # Sensitivity sub-tab: one subpop multiselect for the active parameter
    analysis_sens_subpop_sel = mo.ui.multiselect(
        options=_sp_opts if _sp_opts else ["(none)"],
        value=[],
        label="Apply to subpopulations (empty = all subpops)",
    ) if is_metapop else mo.ui.multiselect(options=["(none)"], value=[], label="")

    # Sensitivity sub-tab: per-subpop sliders [subpop][value_index], pre-allocated _MAX_SENS per subpop
    _is_arr = analysis_param_type.value == "Array (scale factor)"
    _pname_s = analysis_array_param_sel.value if _is_arr else analysis_scalar_param_sel.value

    if is_metapop and _sp_opts:
        if _is_arr:
            _sp_sliders_inner = [
                mo.ui.array([mo.ui.slider(start=0.1, stop=3.0, step=0.05, value=1.0) for _ in range(_MAX_SENS)])
                for _ in _sp_opts
            ]
        else:
            if _pname_s in ANALYSIS_SCALAR_RANGES:
                _lo_s, _hi_s, _step_s = ANALYSIS_SCALAR_RANGES[_pname_s]
                _base_s = float(ANALYSIS_SCALAR_PARAMS.get(_pname_s, 1.0))
            else:
                _lo_s, _hi_s, _step_s, _base_s = 0.0, 10.0, 0.01, 1.0
            _sp_sliders_inner = [
                mo.ui.array([mo.ui.slider(start=_lo_s, stop=_hi_s, step=_step_s, value=_base_s) for _ in range(_MAX_SENS)])
                for _ in _sp_opts
            ]
        analysis_sens_subpop_sliders = mo.ui.array(_sp_sliders_inner)
    else:
        analysis_sens_subpop_sliders = mo.ui.array([])

    return (
        analysis_scalar_subpop_sels, analysis_array_subpop_sels,
        analysis_scalar_subpop_inputs, analysis_array_subpop_scales,
        analysis_sens_subpop_sel, analysis_sens_subpop_sliders,
    )


@app.cell
def _analysis_shared_controls(mo, num_age_groups, is_metapop, metapop_folder_input, Path, json):
    _sp_opts = ["all subpops"]
    if is_metapop and metapop_folder_input.value.strip():
        _mc_path = Path(metapop_folder_input.value.strip()) / "metapop_config.json"
        if _mc_path.exists():
            try:
                with open(_mc_path) as _f:
                    _mc_cfg = json.load(_f)
                _sp_opts = ["all subpops"] + list(_mc_cfg.get("subpopulations", []))
            except Exception:
                pass

    analysis_subpop_selector = mo.ui.multiselect(
        options=_sp_opts, value=["all subpops"], label="Subpopulation(s)",
    )
    analysis_age_selector = mo.ui.multiselect(
        options=["all ages"] + [f"Age {i}" for i in range(num_age_groups)],
        value=["all ages"], label="Age group(s)",
    )
    analysis_sim_days = mo.ui.number(value=250, start=10, stop=730, step=1, label="Simulation days")
    analysis_n_reps = mo.ui.number(value=1, start=1, stop=100, step=1, label="Replicates per scenario")
    analysis_timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    analysis_stochastic = mo.ui.switch(label="Stochastic", value=False)
    analysis_run_button = mo.ui.run_button(label="Run analysis")
    return (
        analysis_subpop_selector, analysis_age_selector,
        analysis_sim_days, analysis_n_reps, analysis_timesteps, analysis_stochastic, analysis_run_button,
    )


@app.cell
def _analysis_compartment_selector(mo, compartments, transition_vars_input, n_transitions, t_name):
    _tvs_explicit = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    _tv_keys = _tvs_explicit if _tvs_explicit else [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    analysis_all_keys = list(compartments) + _tv_keys
    analysis_comp_checkboxes = mo.ui.array([
        mo.ui.checkbox(value=True, label=k) for k in analysis_all_keys
    ])
    return analysis_comp_checkboxes, analysis_all_keys


@app.cell
def _analysis_display(
    mo, main_tab, analysis_sub_tab,
    analysis_param_type, analysis_scalar_param_sel, analysis_array_param_sel,
    analysis_n_values, analysis_sens_sliders,
    analysis_n_scenarios, analysis_scenario_names,
    analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    analysis_subpop_selector, analysis_age_selector,
    analysis_sim_days, analysis_n_reps, analysis_timesteps, analysis_stochastic, analysis_run_button,
    analysis_comp_checkboxes,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS,
    is_metapop, analysis_sp_names,
    analysis_sens_subpop_sel, analysis_sens_subpop_sliders,
    analysis_scalar_subpop_sels, analysis_array_subpop_sels,
    analysis_scalar_subpop_inputs, analysis_array_subpop_scales,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Analysis", None)
    _ACC = CLT_ACCENT["analysis"]
    _n_sc = int(analysis_n_scenarios.value)
    _n_values = int(analysis_n_values.value)
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _use_subpop = is_metapop and bool(analysis_sp_names)

    # --- Sensitivity sub-tab ---
    _is_array = analysis_param_type.value == "Array (scale factor)"
    _param_w = analysis_array_param_sel if _is_array else analysis_scalar_param_sel
    _pname = _param_w.value
    _fmt = "scale factor × each array entry" if _is_array else "value"

    _slider_vals = list(analysis_sens_sliders.value)
    _unique_vals = list(dict.fromkeys(_slider_vals))
    _n_unique = len(_unique_vals)
    _n_total = len(_slider_vals)
    if _n_unique == 1:
        _sens_preview = mo.callout(
            mo.md(
                f"**All {_n_total} slider(s) have the same value ({_unique_vals[0]:.4g}) "
                f"— only 1 scenario will run and all curves will be identical.**\n\n"
                f"Adjust the sliders to distinct values before clicking Run. "
                f"_Note: sliders reset to the base parameter value whenever the Builder config changes._"
            ),
            kind="warn",
        )
    elif _n_unique < _n_total:
        _vals_str = ", ".join(f"`{v:.4g}`" for v in _unique_vals)
        _sens_preview = mo.callout(
            mo.md(
                f"{_n_total - _n_unique} duplicate value(s) removed — "
                f"will run **{_n_unique} scenario(s)**: {_vals_str}"
            ),
            kind="info",
        )
    else:
        _vals_str = ", ".join(f"`{v:.4g}`" for v in _unique_vals)
        _sens_preview = mo.callout(
            mo.md(f"Will run **{_n_unique} scenario(s)**: {_vals_str}"),
            kind="success",
        )

    _sens_parts = [
        mo.md("**Vary one parameter across N values — each value becomes a scenario.**"),
        mo.hstack([analysis_param_type, _param_w, analysis_n_values], justify="start"),
    ]
    if _use_subpop:
        _sens_parts.append(mo.hstack([analysis_sens_subpop_sel], justify="start"))
        _sel_sens_sps = list(analysis_sens_subpop_sel.value or [])
        if _sel_sens_sps:
            _sens_parts.append(mo.callout(
                mo.md(
                    "**Global sliders** (below) apply to all subpops not listed above. "
                    "**Per-subpop sliders** override those subpops for each scenario index i."
                ),
                kind="info",
            ))

    _sens_parts.append(mo.md(f"Varying `{_pname}` ({_fmt}):"))
    _sens_parts.append(mo.hstack(list(analysis_sens_sliders), wrap=True))

    if _use_subpop:
        _sel_sens_sps = list(analysis_sens_subpop_sel.value or [])
        for _sp in _sel_sens_sps:
            if _sp in analysis_sp_names:
                _sp_idx = analysis_sp_names.index(_sp)
                if _sp_idx < len(analysis_sens_subpop_sliders):
                    _sp_slides = list(analysis_sens_subpop_sliders[_sp_idx])[:_n_values]
                    _sens_parts.append(mo.md(f"↳ `{_pname}` for **{_sp}**:"))
                    _sens_parts.append(mo.hstack(_sp_slides, wrap=True))

    _sens_parts.append(_sens_preview)
    _sens_ui = mo.vstack(_sens_parts)

    # --- Scenario sub-tab ---
    _show_sp_col = _use_subpop

    _header_items = [mo.md("**Parameter**")] + [analysis_scenario_names[j] for j in range(_n_sc)]
    if _show_sp_col:
        _header_items.append(mo.md("**Subpopulations**"))
    _header = mo.hstack(_header_items, justify="start")

    _scalar_rows = []
    for _i, _pn in enumerate(_scalar_names):
        _row_items = [mo.md(f"`{_pn}`")] + [analysis_scenario_scalar_inputs[_i][j] for j in range(_n_sc)]
        if _show_sp_col and _i < len(analysis_scalar_subpop_sels):
            _row_items.append(analysis_scalar_subpop_sels[_i])
        _scalar_rows.append(mo.hstack(_row_items, justify="start"))
        if _show_sp_col and _i < len(analysis_scalar_subpop_sels):
            _sel_sps_i = list(analysis_scalar_subpop_sels.value[_i]) if _i < len(analysis_scalar_subpop_sels.value) else []
            for _sp in _sel_sps_i:
                if _sp in analysis_sp_names:
                    _sp_idx = analysis_sp_names.index(_sp)
                    if _i < len(analysis_scalar_subpop_inputs) and _sp_idx < len(analysis_scalar_subpop_inputs[_i]):
                        _sp_inputs = [analysis_scalar_subpop_inputs[_i][_sp_idx][j] for j in range(_n_sc)]
                        _sp_row_items = [mo.md(f"  ↳ *{_sp}*")] + _sp_inputs + [mo.md("")]
                        _scalar_rows.append(mo.hstack(_sp_row_items, justify="start"))

    _array_rows = []
    for _k, _pn in enumerate(_array_names):
        _arr_row_items = [mo.md(f"`{_pn}` ×scale")] + [analysis_scenario_array_scales[_k][j] for j in range(_n_sc)]
        if _show_sp_col and _k < len(analysis_array_subpop_sels):
            _arr_row_items.append(analysis_array_subpop_sels[_k])
        _array_rows.append(mo.hstack(_arr_row_items, justify="start"))
        if _show_sp_col and _k < len(analysis_array_subpop_sels):
            _sel_sps_k = list(analysis_array_subpop_sels.value[_k]) if _k < len(analysis_array_subpop_sels.value) else []
            for _sp in _sel_sps_k:
                if _sp in analysis_sp_names:
                    _sp_idx = analysis_sp_names.index(_sp)
                    if _k < len(analysis_array_subpop_scales) and _sp_idx < len(analysis_array_subpop_scales[_k]):
                        _sp_scale_inputs = [analysis_array_subpop_scales[_k][_sp_idx][j] for j in range(_n_sc)]
                        _arr_sp_row = [mo.md(f"  ↳ *{_sp}* ×scale")] + _sp_scale_inputs + [mo.md("")]
                        _array_rows.append(mo.hstack(_arr_sp_row, justify="start"))

    _scen_body = [mo.md("**Define N scenarios with per-parameter overrides.**"), analysis_n_scenarios, _header]
    if _scalar_rows:
        _scen_body += [mo.md("*Scalar parameters:*")] + _scalar_rows
    if _array_rows:
        _scen_body += [mo.md("*Array parameters (scale factor applied to each entry):*")] + _array_rows
    if not _scalar_rows and not _array_rows:
        _scen_body.append(mo.callout(mo.md("No tunable parameters found in the current config."), kind="info"))
    _scen_ui = mo.vstack(_scen_body)

    _tab_body = {"Sensitivity": _sens_ui, "Scenario": _scen_ui}
    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Analysis</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Sweep parameters or compare '
            "scenarios.</div>"
        ),
        section_card(
            step_header("①", "Design",
                        "Define a sensitivity sweep or a set of scenarios.",
                        accent=_ACC),
            mo.vstack([
                analysis_sub_tab,
                _tab_body.get(analysis_sub_tab.value, mo.md("")),
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Run settings",
                        "Horizon, replicates, slices, and which compartments to display.",
                        accent=_ACC),
            mo.vstack([
                mo.hstack([analysis_sim_days, analysis_n_reps, analysis_timesteps], justify="start"),
                mo.hstack([
                    analysis_stochastic,
                    mo.md("*Ignored — using 1 replicate.*") if not analysis_stochastic.value else mo.md(""),
                ], justify="start"),
                mo.hstack([analysis_subpop_selector, analysis_age_selector], justify="start"),
                mo.md("**Compartments / metrics to display:**"),
                mo.hstack(list(analysis_comp_checkboxes), wrap=True, justify="start"),
                analysis_run_button,
            ]),
            accent=_ACC,
        ),
    ])
    return


@app.cell
def _analysis_define_scenarios(
    analysis_sub_tab,
    analysis_param_type, analysis_scalar_param_sel, analysis_array_param_sel,
    analysis_sens_sliders,
    analysis_n_scenarios, analysis_scenario_names,
    analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, np,
    is_metapop, analysis_sp_names,
    analysis_sens_subpop_sel, analysis_sens_subpop_sliders,
    analysis_scalar_subpop_sels, analysis_array_subpop_sels,
    analysis_scalar_subpop_inputs, analysis_array_subpop_scales,
):
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _use_subpop = is_metapop and bool(analysis_sp_names)
    analysis_scenarios = []

    def _make_per_subpop_list(sp_names, sp_to_override_dict):
        if not sp_to_override_dict:
            return None
        _result = [sp_to_override_dict.get(_sp) for _sp in sp_names]
        return _result if any(_r for _r in _result) else None

    if analysis_sub_tab.value == "Sensitivity":
        _is_array = analysis_param_type.value == "Array (scale factor)"
        if _is_array:
            _pname = analysis_array_param_sel.value
            _base_arr = np.asarray(ANALYSIS_ARRAY_PARAMS.get(_pname, [1.0]))
            for _i, _v in enumerate(list(dict.fromkeys(analysis_sens_sliders.value))):
                _global_ov = {_pname: (_base_arr * _v).tolist()}
                _per_subpop = None
                if _use_subpop:
                    _sel_sps = list(analysis_sens_subpop_sel.value or [])
                    _sp_ov = {}
                    for _sp in _sel_sps:
                        if _sp in analysis_sp_names:
                            _sp_idx = analysis_sp_names.index(_sp)
                            _sp_vals = list(analysis_sens_subpop_sliders.value)
                            if _sp_idx < len(_sp_vals) and _i < len(_sp_vals[_sp_idx]):
                                _sp_scale = float(_sp_vals[_sp_idx][_i])
                                _sp_ov[_sp] = {_pname: (_base_arr * _sp_scale).tolist()}
                    _per_subpop = _make_per_subpop_list(analysis_sp_names, _sp_ov)
                analysis_scenarios.append((f"{_pname} ×{_v:.3g}", _global_ov, _per_subpop))
        else:
            _pname = analysis_scalar_param_sel.value
            for _i, _v in enumerate(list(dict.fromkeys(analysis_sens_sliders.value))):
                _global_ov = {_pname: float(_v)}
                _per_subpop = None
                if _use_subpop:
                    _sel_sps = list(analysis_sens_subpop_sel.value or [])
                    _sp_ov = {}
                    for _sp in _sel_sps:
                        if _sp in analysis_sp_names:
                            _sp_idx = analysis_sp_names.index(_sp)
                            _sp_vals = list(analysis_sens_subpop_sliders.value)
                            if _sp_idx < len(_sp_vals) and _i < len(_sp_vals[_sp_idx]):
                                _sp_ov[_sp] = {_pname: float(_sp_vals[_sp_idx][_i])}
                    _per_subpop = _make_per_subpop_list(analysis_sp_names, _sp_ov)
                analysis_scenarios.append((f"{_pname}={_v:.4g}", _global_ov, _per_subpop))
    else:
        _n = int(analysis_n_scenarios.value)
        for j in range(_n):
            _name = analysis_scenario_names.value[j].strip() or f"Scenario {j + 1}"
            _overrides = {}
            for _i, _pn in enumerate(_scalar_names):
                _overrides[_pn] = float(analysis_scenario_scalar_inputs.value[_i][j])
            for _k, _pn in enumerate(_array_names):
                _scale = float(analysis_scenario_array_scales.value[_k][j])
                if _scale != 1.0:
                    _base = np.asarray(ANALYSIS_ARRAY_PARAMS[_pn])
                    _overrides[_pn] = (_base * _scale).tolist()
            _per_subpop = None
            if _use_subpop:
                _sp_ov_by_name = {}
                for _i, _pn in enumerate(_scalar_names):
                    _sel_sps = list(analysis_scalar_subpop_sels.value[_i]) if _i < len(analysis_scalar_subpop_sels.value) else []
                    for _sp in _sel_sps:
                        if _sp in analysis_sp_names:
                            _sp_idx = analysis_sp_names.index(_sp)
                            _sp_vals = list(analysis_scalar_subpop_inputs.value)
                            if _i < len(_sp_vals) and _sp_idx < len(_sp_vals[_i]) and j < len(_sp_vals[_i][_sp_idx]):
                                _sp_ov_by_name.setdefault(_sp, {})[_pn] = float(_sp_vals[_i][_sp_idx][j])
                for _k, _pn in enumerate(_array_names):
                    _sel_sps = list(analysis_array_subpop_sels.value[_k]) if _k < len(analysis_array_subpop_sels.value) else []
                    for _sp in _sel_sps:
                        if _sp in analysis_sp_names:
                            _sp_idx = analysis_sp_names.index(_sp)
                            _sp_vals = list(analysis_array_subpop_scales.value)
                            if _k < len(_sp_vals) and _sp_idx < len(_sp_vals[_k]) and j < len(_sp_vals[_k][_sp_idx]):
                                _sp_scale = float(_sp_vals[_k][_sp_idx][j])
                                if _sp_scale != 1.0:
                                    _base = np.asarray(ANALYSIS_ARRAY_PARAMS[_pn])
                                    _sp_ov_by_name.setdefault(_sp, {})[_pn] = (_base * _sp_scale).tolist()
                _per_subpop = _make_per_subpop_list(analysis_sp_names, _sp_ov_by_name)
            analysis_scenarios.append((_name, _overrides, _per_subpop))

    return (analysis_scenarios,)


@app.cell
def _analysis_results_state(mo):
    get_analysis_results, set_analysis_results = mo.state(None)
    return get_analysis_results, set_analysis_results


@app.cell
def _analysis_results_reader(get_analysis_results):
    analysis_results = get_analysis_results()
    return (analysis_results,)


@app.cell
def _run_analysis(
    analysis_run_button, analysis_scenarios,
    analysis_sim_days, analysis_n_reps, analysis_timesteps, analysis_stochastic,
    config_dict, compartments, is_metapop,
    metapop_folder_input, metapop_travel_config,
    transition_vars_input,
    build_compartment_init, start_date_input, rng_seed,
    make_single_pop_metapop, make_metapop_from_folder,
    set_analysis_results,
    np, mo, build_scalar_array,
):
    mo.stop(not analysis_run_button.value)

    mo.stop(
        not analysis_scenarios,
        mo.callout(mo.md("**No scenarios defined.** Configure sensitivity or scenario settings above."), kind="warn"),
    )

    _num_days = int(analysis_sim_days.value)
    _n_reps = int(analysis_n_reps.value) if analysis_stochastic.value else 1
    _stoch = bool(analysis_stochastic.value)
    _start = start_date_input.value.strip() or "2024-01-01"
    _ts = int(analysis_timesteps.value)
    _seed_b = int(rng_seed.value)
    _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    if not _tvs:
        _tvs = [t["name"] for t in config_dict.get("transitions", []) if t.get("name")]

    _ci = None
    if not is_metapop:
        # Initial conditions from the Step 6 tables via config_dict.
        _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
        _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((1, 1))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in compartments
        }
        _ci, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)

    def _extract_detailed(metapop, comps, tvs=None):
        _out = {}
        for _sp_name, _sp in metapop.subpop_models.items():
            _sp_out = {}
            for _c in comps:
                _sp_out[_c] = np.array(_sp.compartments[_c].history_vals_list)
            _tv_list = tvs if tvs else list(_sp.transition_variables.keys())
            for _tv in _tv_list:
                _hist = _sp.transition_variables.get(_tv)
                if _hist is not None and _hist.history_vals_list:
                    _raw = np.array(_hist.history_vals_list)
                    # TVs are stored per timestep; aggregate to daily sums
                    _T = _raw.shape[0]
                    if _ts > 1 and _T > 0 and _T % _ts == 0:
                        _raw = _raw.reshape(_T // _ts, _ts, *_raw.shape[1:]).sum(axis=1)
                    _sp_out[_tv] = _raw
            _out[_sp_name] = _sp_out
        return _out

    _all = {}
    with mo.status.spinner(f"Running {len(analysis_scenarios)} scenario(s) × {_n_reps} rep(s)..."):
        try:
            for _scen_tuple in analysis_scenarios:
                _scen_name, _overrides = _scen_tuple[0], _scen_tuple[1]
                _per_subpop = _scen_tuple[2] if len(_scen_tuple) > 2 else None
                _reps_hists = []
                for _rep in range(_n_reps):
                    if not is_metapop:
                        _m, _, _ = make_single_pop_metapop(
                            config_dict, _start, _num_days, _ci,
                            seed_offset=_rep, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=_stoch, tvs=_tvs, save_daily=True,
                            param_overrides=_overrides or None,
                            travel_config=metapop_travel_config,
                        )
                    else:
                        _m, _ = make_metapop_from_folder(
                            metapop_folder_input.value, config_dict, _start, _num_days, list(compartments),
                            seed_offset=_rep, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=_stoch, tvs=_tvs, save_daily=True,
                            param_overrides=_overrides or None,
                            param_overrides_per_subpop=_per_subpop,
                            travel_config=metapop_travel_config,
                        )
                    _m.simulate_until_day(_num_days)
                    _reps_hists.append(_extract_detailed(_m, list(compartments), tvs=_tvs))
                _all[_scen_name] = _reps_hists
        except Exception as _exc:
            mo.stop(True, mo.callout(mo.md(f"**Analysis error:** {_exc}"), kind="danger"))

    _first_rep = next(iter(_all.values()))[0]
    _first_sp_data = next(iter(_first_rep.values()))
    _comp_set = set(compartments)
    _tvs_actual = [k for k in _first_sp_data if k not in _comp_set]
    set_analysis_results({
        "scenarios": _all,
        "subpop_names": list(_first_rep.keys()),
        "compartments": list(compartments),
        "tvs": _tvs_actual,
        "num_days": _num_days,
        "start_date": _start,
    })
    return


@app.cell
def _analysis_autosave(analysis_results, output_dir, json):
    if analysis_results is not None:
        _p = output_dir / "analysis_results.json"
        _p.write_text(json.dumps({
            "compartments": analysis_results["compartments"],
            "tvs": analysis_results["tvs"],
            "num_days": analysis_results["num_days"],
            "start_date": analysis_results.get("start_date", ""),
            "subpop_names": analysis_results["subpop_names"],
            "scenarios": {
                _scen: [
                    {_sp: {k: v.tolist() for k, v in _sp_data.items()}
                     for _sp, _sp_data in _rep.items()}
                    for _rep in _reps
                ]
                for _scen, _reps in analysis_results["scenarios"].items()
            },
        }, indent=2))
    return


@app.cell
def _analysis_plot_compartments(
    analysis_results, analysis_comp_checkboxes, analysis_all_keys,
    analysis_subpop_selector, analysis_age_selector,
    np, pd, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(analysis_results is None, mo.md("*Run analysis to see results.*"))

    _selected = [k for k, v in zip(analysis_all_keys, analysis_comp_checkboxes.value) if v] or analysis_all_keys
    _sel_subpops = analysis_subpop_selector.value or ["all subpops"]
    _sel_ages = analysis_age_selector.value or ["all ages"]
    _scens = analysis_results["scenarios"]
    _sp_names = analysis_results["subpop_names"]
    _start = analysis_results.get("start_date", "2024-01-01")

    _combos = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _LINE_STYLES = ["-", "--", ":", "-."]

    def _agg(rep_data, sp_sel, age_sel, key):
        _sps = (
            [rep_data[sp] for sp in _sp_names if sp in rep_data]
            if sp_sel == "all subpops"
            else ([rep_data[sp_sel]] if sp_sel in rep_data else [])
        )
        if not _sps or key not in _sps[0]:
            return None
        _total = np.stack([d[key] for d in _sps], axis=0).sum(axis=0)  # (days, A, R)
        if age_sel == "all ages":
            return _total.sum(axis=(1, 2))
        return _total[:, int(age_sel.split()[-1]), :].sum(axis=1)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(11, min(4 * _n_combos, 80)), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        for _s_idx, (_scen_name, _reps) in enumerate(_scens.items()):
            _color = _colors[_s_idx % len(_colors)]
            for _k_idx, _key in enumerate(_selected):
                _ls = _LINE_STYLES[_k_idx % len(_LINE_STYLES)]
                _rep_arrs = [_agg(rep, _sp, _ag, _key) for rep in _reps]
                _rep_arrs = [a for a in _rep_arrs if a is not None]
                if not _rep_arrs:
                    continue
                _stacked = np.stack(_rep_arrs, axis=0)
                _dates = pd.date_range(start=_start, periods=_stacked.shape[1], freq="D")
                _med = np.median(_stacked, axis=0)
                _lo = np.percentile(_stacked, 2.5, axis=0)
                _hi = np.percentile(_stacked, 97.5, axis=0)
                _ax.plot(_dates, _med, label=f"{_scen_name} — {_key}",
                         color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.15)

        _ax.set_xlabel("Date")
        _ax.set_ylabel("Count")
        _ax.set_title(f"Compartment histories (median + 95% CI) — {_sp} / {_ag}")
        _handles, _labels_leg = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels_leg, fontsize=7, loc="upper right")

    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("## Analysis — Compartment Histories"), _fig])
    return


@app.cell
def _analysis_detailed_download(
    analysis_results, np, pd, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(analysis_results is None, mo.md(""))

    _scens = analysis_results["scenarios"]
    _sp_names = analysis_results["subpop_names"]
    _all_keys = analysis_results["compartments"] + analysis_results["tvs"]
    _start = analysis_results.get("start_date", "2024-01-01")

    _rows = []
    for _scen_name, _reps in _scens.items():
        for _rep_idx, _rep_data in enumerate(_reps):
            # Aggregate across subpops for each key: shape (days, A, R)
            _agg_by_key = {}
            for _key in _all_keys:
                _arrays = [_rep_data[_sp][_key] for _sp in _sp_names if _key in _rep_data.get(_sp, {})]
                if _arrays:
                    _agg_by_key[_key] = np.stack(_arrays, axis=0).sum(axis=0)

            for _sp in _sp_names + ["aggregated"]:
                for _key in _all_keys:
                    if _sp == "aggregated":
                        _arr = _agg_by_key.get(_key)
                    else:
                        _arr = _rep_data.get(_sp, {}).get(_key)
                    if _arr is None:
                        continue
                    _arr = np.array(_arr)  # (days, A, R)
                    _n_days, _n_ages, _n_risks = _arr.shape
                    _dates = pd.date_range(start=_start, periods=_n_days, freq="D")

                    for _age_idx in range(_n_ages):
                        _series = _arr[:, _age_idx, :].sum(axis=1)
                        for _day_i, (_date, _val) in enumerate(zip(_dates, _series)):
                            _rows.append({
                                "date": _date.date().isoformat(),
                                "scenario": _scen_name,
                                "subpopulation": _sp,
                                "age_group": _age_idx,
                                "replicate": _rep_idx,
                                "metric": _key,
                                "value": float(_val),
                            })

                    # Aggregated age group (sum over all ages and risks)
                    _series_agg = _arr.sum(axis=(1, 2))
                    for _day_i, (_date, _val) in enumerate(zip(_dates, _series_agg)):
                        _rows.append({
                            "date": _date.date().isoformat(),
                            "scenario": _scen_name,
                            "subpopulation": _sp,
                            "age_group": "aggregated",
                            "replicate": _rep_idx,
                            "metric": _key,
                            "value": float(_val),
                        })

    _detail_df = pd.DataFrame(_rows) if _rows else pd.DataFrame(
        columns=["date", "scenario", "subpopulation", "age_group", "replicate", "metric", "value"]
    )
    _detail_csv_dl = mo.download(
        data=_detail_df.to_csv(index=False).encode(),
        filename="analysis_detailed_timeseries.csv",
        label="Download detailed timeseries CSV",
    )
    mo.vstack([_detail_csv_dl])
    return


@app.cell
def _analysis_summary_table(
    analysis_results, analysis_comp_checkboxes, analysis_all_keys,
    analysis_subpop_selector, analysis_age_selector,
    np, pd, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(analysis_results is None, mo.md(""))

    _selected = [k for k, v in zip(analysis_all_keys, analysis_comp_checkboxes.value) if v] or analysis_all_keys
    _sel_subpops = analysis_subpop_selector.value or ["all subpops"]
    _sel_ages = analysis_age_selector.value or ["all ages"]
    _scens = analysis_results["scenarios"]
    _sp_names = analysis_results["subpop_names"]

    def _agg(rep_data, sp_sel, age_sel, key):
        _sps = (
            [rep_data[sp] for sp in _sp_names if sp in rep_data]
            if sp_sel == "all subpops"
            else ([rep_data[sp_sel]] if sp_sel in rep_data else [])
        )
        if not _sps or key not in _sps[0]:
            return None
        _total = np.stack([d[key] for d in _sps], axis=0).sum(axis=0)  # (days, A, R)
        if age_sel == "all ages":
            return _total.sum(axis=(1, 2))
        return _total[:, int(age_sel.split()[-1]), :].sum(axis=1)

    _rows = []
    for _sp in _sel_subpops:
        for _ag in _sel_ages:
            for _scen_name, _reps in _scens.items():
                for _key in _selected:
                    _arrays = [_agg(rep, _sp, _ag, _key) for rep in _reps]
                    _arrays = [a for a in _arrays if a is not None]
                    if not _arrays:
                        continue
                    _mat = np.stack(_arrays, axis=0)
                    _rows.append({
                        "Scenario": _scen_name,
                        "Subpopulation": _sp,
                        "Age group": _ag,
                        "Metric": _key,
                        "Peak (median)": f"{float(np.median(np.max(_mat, axis=1))):,.0f}",
                        "Peak day (median)": int(np.median(np.argmax(_mat, axis=1))) + 1,
                        "Day-end (median)": f"{float(np.median(_mat[:, -1])):,.0f}",
                    })

    _df = pd.DataFrame(_rows) if _rows else pd.DataFrame(
        columns=["Scenario", "Subpopulation", "Age group", "Metric",
                 "Peak (median)", "Peak day (median)", "Day-end (median)"]
    )
    _csv_dl = mo.download(
        data=_df.to_csv(index=False).encode(),
        filename="analysis_summary.csv",
        label="Download summary CSV",
    )
    mo.vstack([
        mo.md("### Analysis — Summary Table"),
        mo.ui.table(_df) if not _df.empty else mo.md("*No data.*"),
        _csv_dl,
    ])
    return


# ---------------------------------------------------------------------------
# Analysis — User-defined metrics (line, box, and bar plots)
# ---------------------------------------------------------------------------


@app.cell
def _analysis_metric_defs_show(
    mo, main_tab,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    analysis_plot_metric_sel, transition_vars_input, tv_opts,
):
    mo.stop(main_tab.value != "Analysis", None)
    _n = int(analysis_n_metrics_input.value)
    _tvars_explicit = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    if tv_opts:
        _hint = (
            ("Saving all transition variables. " if not _tvars_explicit else "")
            + "Available: "
            + ", ".join(f"`{t}`" for t in tv_opts)
        )
        _hint_kind = "info"
    else:
        _hint = "No transitions defined yet. Complete Steps 2–3 in Model Builder first."
        _hint_kind = "warn"
    _rows = []
    for _i in range(_n):
        _rows.append(mo.hstack([analysis_metric_names[_i], analysis_metric_tvs[_i]], justify="start"))
    mo.vstack([
        mo.md("## Analysis — User-defined Metrics"),
        mo.md(
            "Define metrics as the sum of one or more saved transition variables. "
            "For example, in a SEIR model the daily incidence is `S_to_E`. "
            "The three plots below update automatically as you select metrics."
        ),
        mo.callout(mo.md(_hint), kind=_hint_kind),
        analysis_n_metrics_input,
        *_rows,
        mo.md("**Select which metrics to show in the plots below:**"),
        analysis_plot_metric_sel,
    ])
    return


@app.cell
def _analysis_compute_metric_series(
    analysis_results,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    np,
):
    analysis_metric_series = None

    if analysis_results is not None:
        _n = int(analysis_n_metrics_input.value)
        _metric_defs = []
        for _i in range(_n):
            _name = analysis_metric_names.value[_i].strip() or f"metric_{_i + 1}"
            _raw = analysis_metric_tvs.value[_i]
            _tvs = _raw if isinstance(_raw, list) else [t.strip() for t in _raw.split(",") if t.strip()]
            if _tvs:
                _metric_defs.append((_name, _tvs))

        if _metric_defs:
            _metrics_out = {}
            for _mname, _mtvs in _metric_defs:
                _scen_data = {}
                for _scen_name, _reps in analysis_results["scenarios"].items():
                    _rep_list = []
                    for _rep in _reps:
                        _sp_data = {}
                        for _sp_name, _sp_hist in _rep.items():
                            _total = None
                            for _tv in _mtvs:
                                if _tv in _sp_hist:
                                    _arr = np.array(_sp_hist[_tv])
                                    _total = _arr if _total is None else _total + _arr
                            if _total is not None:
                                _sp_data[_sp_name] = _total  # shape (T, A, R)
                        _rep_list.append(_sp_data)
                    _scen_data[_scen_name] = _rep_list
                _metrics_out[_mname] = _scen_data

            analysis_metric_series = {
                "metrics": _metrics_out,
                "sp_names": analysis_results["subpop_names"],
                "start_date": analysis_results.get("start_date", "2024-01-01"),
            }

    return (analysis_metric_series,)


@app.cell
def _analysis_plot_daily_metrics(
    analysis_metric_series, analysis_plot_metric_sel,
    analysis_subpop_selector, analysis_age_selector,
    np, pd, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(
        analysis_metric_series is None,
        mo.md("*Define metrics above and run analysis to see metric plots.*"),
    )
    _sel_metrics = [
        m for m in (analysis_plot_metric_sel.value or [])
        if m in analysis_metric_series["metrics"]
    ]
    mo.stop(not _sel_metrics, mo.md("*Select at least one metric to plot.*"))

    _metrics = analysis_metric_series["metrics"]
    _sp_names = analysis_metric_series["sp_names"]
    _start = analysis_metric_series["start_date"]
    _sel_subpops = analysis_subpop_selector.value or ["all subpops"]
    _sel_ages = analysis_age_selector.value or ["all ages"]
    _combos = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _LINE_STYLES = ["-", "--", ":", "-."]

    def _agg_daily(rep_data, sp_sel, age_sel):
        _sps = (
            [rep_data[sp] for sp in _sp_names if sp in rep_data]
            if sp_sel == "all subpops"
            else ([rep_data[sp_sel]] if sp_sel in rep_data else [])
        )
        if not _sps:
            return None
        _total = np.sum(np.stack(_sps, axis=0), axis=0)  # (T, A, R)
        if age_sel == "all ages":
            return _total.sum(axis=(1, 2))
        return _total[:, int(age_sel.split()[-1]), :].sum(axis=1)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(11, min(4 * _n_combos, 80)), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        for _m_idx, _mname in enumerate(_sel_metrics):
            _ls = _LINE_STYLES[_m_idx % len(_LINE_STYLES)]
            _mdata = _metrics[_mname]
            for _s_idx, (_scen_name, _reps) in enumerate(_mdata.items()):
                _color = _colors[_s_idx % len(_colors)]
                _rep_arrs = [_agg_daily(_rep, _sp, _ag) for _rep in _reps]
                _rep_arrs = [a for a in _rep_arrs if a is not None]
                if not _rep_arrs:
                    continue
                _stacked = np.stack(_rep_arrs, axis=0)
                _dates = pd.date_range(start=_start, periods=_stacked.shape[1], freq="D")
                _med = np.median(_stacked, axis=0)
                _lo = np.percentile(_stacked, 2.5, axis=0)
                _hi = np.percentile(_stacked, 97.5, axis=0)
                _ax.plot(_dates, _med, label=f"{_mname} — {_scen_name}", color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.15)

        _ax.set_xlabel("Date")
        _ax.set_ylabel("Daily count")
        _ax.set_title(f"Daily metric by scenario (median + 95% CI) — {_sp} / {_ag}")
        _handles, _labels_leg = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels_leg, fontsize=8, loc="upper right")

    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Analysis — Daily Metric by Scenario"), _fig])
    return


@app.cell
def _analysis_plot_cumulative_boxplot(
    analysis_metric_series, analysis_plot_metric_sel,
    analysis_subpop_selector, analysis_age_selector,
    np, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(analysis_metric_series is None, mo.md(""))
    _sel_metrics = [
        m for m in (analysis_plot_metric_sel.value or [])
        if m in analysis_metric_series["metrics"]
    ]
    mo.stop(not _sel_metrics, mo.md(""))

    _metrics = analysis_metric_series["metrics"]
    _sp_names = analysis_metric_series["sp_names"]
    _sel_subpops = analysis_subpop_selector.value or ["all subpops"]
    _sel_ages = analysis_age_selector.value or ["all ages"]
    _combos = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)
    _n_met = len(_sel_metrics)

    def _cum_scalar(rep_data, sp_sel, age_sel):
        _sps = (
            [rep_data[sp] for sp in _sp_names if sp in rep_data]
            if sp_sel == "all subpops"
            else ([rep_data[sp_sel]] if sp_sel in rep_data else [])
        )
        if not _sps:
            return None
        _total = np.sum(np.stack(_sps, axis=0), axis=0)  # (T, A, R)
        if age_sel == "all ages":
            return float(_total.sum())
        return float(_total[:, int(age_sel.split()[-1]), :].sum())

    _fig, _axes = plt.subplots(
        _n_combos, _n_met,
        figsize=(max(5 * _n_met, 6), min(5 * _n_combos, 80)),
        squeeze=False,
    )

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        for _m_idx, _mname in enumerate(_sel_metrics):
            _ax = _axes[_c_idx, _m_idx]
            _mdata = _metrics[_mname]
            _scen_names = list(_mdata.keys())
            _box_data = []
            for _scen_name in _scen_names:
                _vals = [_cum_scalar(_rep, _sp, _ag) for _rep in _mdata[_scen_name]]
                _vals = [v for v in _vals if v is not None]
                _box_data.append(_vals if _vals else [0.0])
            _ax.boxplot(_box_data, tick_labels=_scen_names, vert=True)
            _ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
            _ax.set_ylabel(f"Cumulative {_mname}")
            _ax.set_title(f"Cumulative {_mname} — {_sp} / {_ag}")
            _ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    mo.vstack([mo.md("### Analysis — Cumulative Metric Distribution by Scenario"), _fig])
    return


@app.cell
def _analysis_plot_age_bars(
    analysis_metric_series, analysis_plot_metric_sel,
    analysis_subpop_selector,
    num_age_groups, np, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Analysis", None)
    mo.stop(analysis_metric_series is None, mo.md(""))
    _sel_metrics = [
        m for m in (analysis_plot_metric_sel.value or [])
        if m in analysis_metric_series["metrics"]
    ]
    mo.stop(not _sel_metrics, mo.md(""))

    _metrics = analysis_metric_series["metrics"]
    _sp_names = analysis_metric_series["sp_names"]
    _sel_subpops = analysis_subpop_selector.value or ["all subpops"]
    _n_ages = num_age_groups
    _x_labels = [f"Age {_a}" for _a in range(_n_ages)] + ["Total"]
    _x = np.arange(len(_x_labels))
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _cum_per_age(rep_data, sp_sel):
        _sps = (
            [rep_data[sp] for sp in _sp_names if sp in rep_data]
            if sp_sel == "all subpops"
            else ([rep_data[sp_sel]] if sp_sel in rep_data else [])
        )
        if not _sps:
            return None
        _total = np.sum(np.stack(_sps, axis=0), axis=0)  # (T, A, R)
        _per_age = [float(_total[:, _a, :].sum()) for _a in range(_n_ages)]
        return np.array(_per_age + [float(_total.sum())])

    _n_plots = len(_sel_subpops) * len(_sel_metrics)
    _fig, _axes = plt.subplots(_n_plots, 1, figsize=(10, min(5 * _n_plots, 80)), squeeze=False)
    _ax_idx = 0

    for _sp in _sel_subpops:
        for _mname in _sel_metrics:
            _ax = _axes[_ax_idx, 0]
            _mdata = _metrics[_mname]
            _scen_names = list(_mdata.keys())
            _width = 0.8 / max(len(_scen_names), 1)

            for _s_idx, _scen_name in enumerate(_scen_names):
                _offset = (_s_idx - len(_scen_names) / 2) * _width + _width / 2
                _rep_arrs = [_cum_per_age(_rep, _sp) for _rep in _mdata[_scen_name]]
                _rep_arrs = [a for a in _rep_arrs if a is not None]
                if not _rep_arrs:
                    continue
                _mean_vals = np.mean(np.stack(_rep_arrs, axis=0), axis=0)
                _ax.bar(
                    _x + _offset, _mean_vals, _width,
                    label=_scen_name,
                    color=_colors[_s_idx % len(_colors)],
                    alpha=0.8,
                )

            # Subtle separator before the "Total" bar
            _ax.axvline(x=_n_ages - 0.5, color="gray", linestyle=":", alpha=0.5)
            _ax.set_xlabel("Age group")
            _ax.set_ylabel(f"Cumulative {_mname} (mean across replicates)")
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_x_labels)
            _ax.set_title(f"Age-stratified cumulative {_mname} — {_sp}")
            _ax.legend()
            _ax_idx += 1

    plt.tight_layout()
    mo.vstack([mo.md("### Analysis — Age-stratified Metric by Scenario"), _fig])
    return


# ============================================================
# Documentation tab
# ============================================================

@app.cell
def _docs_display(mo, main_tab):
    mo.stop(main_tab.value != "Documentation", None)
    mo.vstack([
        mo.md("""
# CLT Model Builder — User Guide

This notebook lets you build a config-driven compartmental epidemic model, fit it to
data, run forecasts, export scripts for server runs, and compare scenarios — all
without writing code.  Each capability lives in its own tab.  All tabs share the same
model defined in **Model Builder**.

---

## Output directory

The text box at the top of every page sets the folder where results are auto-saved.
The default is `~/clt_outputs/`.  Change it before running anything if you want
output in a specific location.

---

## Tab 1 — Population & Geography

**Purpose:** Define the population dimensions and (optionally) fetch real contact
matrices for a geography. Do this **first** — the rest of the model is built in the
Model Builder tab and depends on the age/risk group counts set here.

### What you configure
- **Age groups** — two modes:
  - *Count only* — just a number `A` (for abstract models with no real age bands).
  - *Named age bands* — e.g. `0-4, 5-17, 18-49, 50-64, 65+`. Bands must start at 0,
    be contiguous, and end in an open band `x+` (with `x ≤ 84`). `A` = number of bands.
- **Risk groups** — the number `R`.
- **Population mode** — *Single population* or *Metapopulation* (with a folder path).
- **Contact matrices** — when using named age bands you can **fetch** the total / school /
  work matrices for a geography (Mistry 2021, via the optional `epydemix` package).

### Contact-matrix fetch
- Requires **named age bands** (the fetcher needs age-group definitions). In count-only
  mode, supply contact-matrix CSVs in Model Builder → Step 4 instead.
- Choose a **US state** or a **Country** (both are searchable dropdowns of the
  epydemix-data location names).
- For metapopulation models, choose the geography **scope**:
  - *Same for all subpops* — one geography; matrices are shared across subpops.
  - *Per-subpopulation* — one geography per subpop; each subpop gets its own matrices,
    written into the config's `subpop_params` section.
- Press **Fetch contact matrices**. Results are written into the config and used at run
  time. Requires `pip install epydemix` and internet access; if unavailable, the tab shows
  an install hint and you can fall back to CSVs.

---

## Tab 2 — Model Builder

**Purpose:** Define the structure of your epidemic model and do a quick preview simulation.

### Steps

| Step | What you configure |
|------|--------------------|
| 0 — Load config | Optionally load a previously saved `model_config.json` to pre-fill all fields. |
| 1 — Compartments | Name each compartment (e.g. `S`, `E`, `I`, `R`).  The first compartment receives the bulk of the initial population. |
| 2 — Transitions | Define flows between compartments.  Each transition needs a name, a "from" compartment, a "to" compartment, and a rate template (`constant_param`, `param_product`, `immunity_modulated`, `force_of_infection`, `force_of_infection_travel`, or `scheduled_exact`). |
| 3 — Parameters | Set numeric values for all parameters referenced by your rate templates (e.g. `beta_baseline`, `sigma`, `gamma`). |
| 4 — Schedules & immunity | Optionally upload CSVs for time-varying schedules: absolute humidity, school/work calendars, mobility, and daily vaccines. Contact matrices come from the Population & Geography tab or CSVs here. |
| 5 — Diagram | Preview the compartment diagram generated from your transitions. |
| 6 — Initial conditions | Set the total population and seed counts for compartments 2–N. |
| 7 — Simulation settings | Choose deterministic vs stochastic, number of replicates, timesteps per day, start date, and which transition variables to save. |
| 8 — Config preview & download | Review the full `model_config.json` and download it. |
| 9 — Run | Click **Run simulation** to see epidemic curves and a summary table. |

**Metapopulation mode:** Enabled in the **Population & Geography** tab, which asks for a
folder path containing:
- `metapop_config.json` — subpopulation names and travel matrix
- `initial_conditions_<SubpopName>.json` — per-subpop initial conditions
- Optional per-subpop schedule CSVs (`school_work_calendar_<name>.csv`, `vaccines_<name>.csv`)
- Optional shared schedule CSVs (`absolute_humidity.csv`, `mobility_modifier.csv`)

**Auto-save:** The model config is written to `{output_dir}/model_config.json` every time
any setting changes, so you never lose your work.

---

## Tab 3 — Fitting

**Purpose:** Estimate unknown parameters by fitting the model to an observed time series.

### Steps

1. **Observed data** — Upload a CSV or provide a file path.  The file must have at least
   two columns; all columns whose names are not `date`, `day`, `time`, or `week` are
   treated as the observed values (the first such column is used).
2. **Target** — Choose which model output to fit.  This can be any compartment name or
   any transition variable name (as listed in Step 8 of Model Builder).
3. **Parameters and bounds** — Enter a comma-separated list of parameter names to fit,
   then provide bounds as a JSON object: `{"beta_baseline": [0.05, 0.8]}`.  If you
   omit bounds for a parameter, the notebook guesses ±80 % around the current value.
4. **Method**
   - *Adam (gradient)* — PyTorch-based gradient descent.  Fast and accurate for smooth
     loss landscapes.  **Requires a transition variable as the target** (not a compartment).
   - *LBFGS (gradient)* — Second-order gradient method.  Often converges in fewer steps
     than Adam but each step is more expensive.  Same target constraint applies.
   - *Accept-reject* — Parameter-space random search that accepts samples with R² above
     a threshold.  Works with any target (compartment or transition) and does not require
     PyTorch.
5. Click **Run fitting**.

### Results

- **Loss / R² curve** — Shows fitting progress over iterations or samples.
- **Best-fit parameters** — The parameter values that minimised the loss (or maximised R²).

Auto-saved to `{output_dir}/fitted_params.json`.

### Tips

- For gradient methods, start with a small learning rate (0.001–0.01) and 100–200
  iterations; watch the loss curve to judge convergence.
- For accept-reject, increase "Max samples" if the best R² is still below the threshold
  after running.
- Gradient fitting fits parameters **globally** (all subpopulations share the same values).
  Use accept-reject for metapopulation models.

---

## Tab 4 — Forecast

**Purpose:** Run an ensemble forward projection using the fitted (or current) parameters.

### Steps

1. **Fitted parameters** — Toggle on "Use fitted params from Fitting tab" to apply the
   best-fit values automatically.  Or point to a `fitted_params.json` on disk.
2. **Settings** — Choose forecast horizon (days beyond the fit period), number of
   replicates, and stochastic vs deterministic.
3. Click **Run forecast**.

### Results

- **Epidemic curves** — Median + 95 % CI ribbon for each compartment.  A dashed vertical
  line and shaded region mark the end of the fit period.
- **Summary table** — Median peak value and peak day per compartment.

Auto-saved to `{output_dir}/forecast_ensemble.json`.

### Notes

- The simulation always starts from day 1 (initial conditions set in Model Builder).
  The "fit period" is just a visual annotation: the model runs for `fit_days + horizon`
  days in a single pass.
- Stochastic replicates use independent random seeds; increase replicates for smoother
  confidence intervals.

---

## Tab 5 — Export

**Purpose:** Generate a standalone Python script that can run your model on a server
or in a batch job, and download all configuration files.

### What is generated

- **`run_simulation.py`** — A self-contained script that loads `model_config.json` and
  optionally `fitted_params.json`, builds the model, runs each entry in a `SCENARIOS`
  dict, and saves results to a SQLite database (`simulation_output/results.db`).

  Edit the top of the script to configure:
  - `NUM_DAYS`, `NUM_REPS`, `STOCHASTIC`, `TIMESTEPS_PER_DAY`, `START_DATE`
  - `SCENARIOS` — a dict mapping scenario name to a `{param: value}` override dict

- **`model_config.json`** — The current model configuration.
- **`fitted_params.json`** — The best-fit parameter values (empty `{}` if fitting has
  not been run).

### Running the script

```bash
# Put all three files in the same directory, then:
python run_simulation.py
```

Results are stored in `simulation_output/results.db` as a table with columns
`scenario`, `rep`, `compartment`, `day`, `value`.

---

## Tab 6 — Analysis

**Purpose:** Compare how model outputs change across scenarios or parameter values.
Sensitivity and scenario analysis share identical output plots.

### Sub-tabs

#### Sensitivity
Vary **one parameter** across N values.  Each value becomes its own scenario, labelled
`param=value`.  Use this to understand how sensitive the model is to a single unknown.

- Select the parameter from the dropdown (populated from your model's `params`).
- Enter values as a comma-separated list: `0.1, 0.2, 0.3, 0.4`.

#### Scenario
Define **N parameter bundles**.  Each bundle is a named scenario with its own set of
parameter overrides.  Use this to compare specific interventions or assumptions.

Enter scenarios as a JSON object:
```json
{
  "baseline":   {},
  "high_beta":  {"beta_baseline": 0.4},
  "vaccination": {"beta_baseline": 0.2, "daily_vaccines_value": 5000}
}
```

### Shared run settings

| Setting | Description |
|---------|-------------|
| Simulation days | How many days to simulate for each scenario. |
| Replicates per scenario | How many stochastic runs per scenario (use 1 for deterministic). |
| Stochastic | Toggle on for binomial draws; off for deterministic (faster). |
| Output metric | Which compartment or transition variable to plot in the main chart. |

Click **Run analysis**.

### Results

- **Scenario comparison plot** — One line per scenario for the selected metric,
  with 95 % CI ribbons when replicates > 1.
- **Summary table** — Peak value, peak day, and day-end value for every
  (scenario, metric) combination.
- **Download summary CSV** — Export the summary table for use in external tools.

Auto-saved to `{output_dir}/analysis_results.json`.

---

## Advanced: modelling vaccination (`scheduled_exact`)

To move an exact, data-driven number of people between compartments each day
(e.g. vaccination `S → V`), use the **`scheduled_exact`** rate template instead
of a rate-based one. It bypasses the usual rate→probability machinery and applies
the scheduled transfer deterministically on the first timestep of each day.

How to set it up in **Model Builder**:

1. **Step 2 — Compartments:** add the destination compartment (e.g. `V`).
2. **Step 3 — Transitions:** add a transition `S → V` with rate template
   `scheduled_exact`. It references a schedule by name (the daily-vaccines
   schedule).
3. **Step 5 — Schedules:** supply the vaccines schedule, either as a constant
   value or via a `vaccines_<name>.csv` (column `daily_vaccines`, a JSON A×R
   array). **The schedule value is a daily _proportion_ of the origin
   compartment** (0–1), not an absolute count; it is rounded to an integer and
   capped at the available population each day.

Notes:
- `scheduled_exact` transitions are deterministic. They cannot be placed in a
  transition group or marked jointly-distributed (the config parser rejects this).
- Vaccination immunity (the `MV` metric) is configured separately in Step 5 — a
  compartment transfer (`scheduled_exact`) and a population-immunity metric
  (`vaccine_induced_immunity`) are independent mechanisms; use either or both.

---

## Advanced: multi-target fitting

The **Fitting** tab can fit several observed series at once. Add targets with the
"number of targets" control; each target independently specifies:

- the model output to match (a compartment or transition variable),
- an optional **slice** (subpopulation / age group / risk group),
- a **mode** — timeseries, scalar total, or proportions, and
- a **weight** `λ` controlling its contribution to the combined loss.

The objective minimised is the weighted sum of per-target losses. Practical advice:

- Keep targets on comparable scales, or use weights to balance them — a large-count
  target will otherwise dominate a small-count one.
- Avoid conflicting targets (e.g. fitting both a compartment and a transition that
  mechanically determine each other) — they can make the loss landscape ill-conditioned.
- Gradient methods (Adam / L-BFGS) require **transition-variable** targets; use
  accept-reject if any target is a compartment.

---

## Typical workflow

```
Population & Geography  →  Model Builder  →  Fitting  →  Forecast  →  Export
                                                ↓
                                             Analysis
```

1. Set age/risk groups, population mode, and (optionally) fetch contact matrices in
   **Population & Geography**.
2. Build your model in **Model Builder** and confirm the epidemic curves look sensible.
3. Go to **Fitting**, upload observed data, and fit key parameters.
4. Check the fit overlay in Fitting results, then switch to **Forecast** to project forward.
5. Use **Analysis** to quantify uncertainty (sensitivity) or compare policy scenarios.
6. When ready to run larger ensembles, go to **Export**, download the script and configs,
   and run them on your server.

---

## File formats

### `model_config.json`
The master configuration file.  It is read/written by Steps 0 and 9 of Model Builder
and auto-saved whenever any setting changes.

### Observed data CSV (Fitting tab)
Any CSV with at least two columns.  The first column whose name is not
`date`, `day`, `time`, or `week` is used as the observed values.

Example:
```
date,hospitalizations
2024-01-01,12
2024-01-02,18
2024-01-03,25
```

### Fitted params JSON
A flat dict of `{param_name: value}` pairs, e.g.:
```json
{"beta_baseline": 0.23, "sigma": 0.5}
```

### Metapop config JSON (`metapop_config.json`)
`subpopulations` is an **ordered list** of names; `travel_matrix` is an N×N
matrix (N = number of subpopulations) whose rows sum to 1.
```json
{
  "subpopulations": ["SubpopA", "SubpopB"],
  "travel_matrix": [[0.95, 0.05], [0.05, 0.95]]
}
```

### Initial conditions JSON (`initial_conditions_<name>.json`)
```json
{
  "compartments": {"S": [[950000]], "I": [[50000]]},
  "epi_metrics":  {}
}
```
Arrays are shape `[age_groups][risk_groups]`.
"""),
    ])
    return


if __name__ == "__main__":
    app.run()


if __name__ == "__main__":
    app.run()
