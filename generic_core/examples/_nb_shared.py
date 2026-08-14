# _nb_shared.py
# Section: Shared imports and helper functions
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _imports():
    import sys
    import json
    import io
    import copy
    import re
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
    from generic_core.fitting import (
        FitResult, FitTarget, FitConfig, run_fit,
        fit_result_to_dict, fit_result_from_dict,
    )

    return (
        Path, SimpleNamespace, namedtuple, copy, re, sqlite3, datetime,
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
        FitResult, FitTarget, FitConfig, run_fit,
        fit_result_to_dict, fit_result_from_dict,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@app.cell
def _helpers(Path, SimpleNamespace, json, np, pd):
    from generic_core.model_factory import (
        build_scalar_array,
        build_notebook_schedules_input,
    )

    def parse_csv_list(text: str) -> list[str]:
        """Split a comma-separated string into a list of trimmed, non-empty items."""
        return [_item.strip() for _item in text.split(",") if _item.strip()]

    def rel_inf_param_name(compartment: str) -> str:
        """Auto-generated parameter name for a compartment's relative infectiousness."""
        return f"{compartment}_relative_infectiousness"

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

    def detect_config_type(filename: str) -> str:
        """Best-effort classification of an uploaded JSON config by filename,
        for the shared multi-file importer (see _nb_import.py). Returns one
        of "model_config", "fit_config", "fitted_params", "scenario_config",
        or "unknown" -- shown as the pre-selected type in a per-file dropdown
        the user confirms/overrides before applying, so a wrong guess here is
        just an extra click, not a silent misfile."""
        _n = (filename or "").lower()
        if "scenario" in _n:
            return "scenario_config"
        if "fit_config" in _n:
            return "fit_config"
        if "fitted" in _n or "fit_result" in _n or "params" in _n:
            return "fitted_params"
        if "config" in _n:
            return "model_config"
        return "unknown"

    def parse_fit_config_targets(raw: dict) -> tuple:
        """Turn a saved fit_config.json's ``targets`` list into the
        (slots, data) shape the Fitting tab's per-slot restore state expects
        (see _fit_config_upload_ui / _nb_import.py's shared importer, which
        share this so the two stay in sync). Raises if ``raw`` has no
        "targets" key."""
        _targets = raw["targets"]
        _new_slots, _new_data = [], {}
        for _i, _t in enumerate(_targets):
            if _i >= 20:
                break
            _new_slots.append(_i)
            _new_data[_i] = {
                "label": _t.get("label", f"Restored {_i + 1}"),
                "vars": _t.get("variables"),
                "mode": _t.get("mode"),
                "weight": _t.get("weight"),
                "subpop_idx": _t.get("subpop_idx"),
                "age_idx": _t.get("age_idx"),
                "risk_idx": _t.get("risk_idx"),
                "observed": _t.get("observed"),
                "point_weights": _t.get("point_weights"),
            }
        return _new_slots, _new_data

    def scenario_state_group(key: str) -> str:
        """Which of the Analysis Scenario sub-tab's four split mo.state
        dicts (see _nb_analysis.py's _analysis_scenario_state and its four
        *_controls cells) a given scenario-config key belongs to. Used to
        partition an uploaded/imported flat scenario_config.json across
        them (see partition_scenario_state, and _nb_import.py's shared
        importer)."""
        if key == "n_scenarios" or key.startswith(("name::", "scalar::", "array::")):
            return "controls"
        if key.startswith(("age_scale_toggle::", "age_scale::")):
            return "agescale"
        if key in ("dose_toggle", "dose_per_subpop_toggle") or key.startswith(
            ("dose::", "dose_subpop::")
        ):
            return "dose"
        if key.startswith((
            "scalar_subpop_sel::", "scalar_subpop::",
            "array_subpop_sel::", "array_subpop::",
        )):
            return "subpop"
        return "controls"  # unrecognized key: harmless default bucket

    def partition_scenario_state(flat: dict) -> dict:
        """Split a flat scenario-config dict (the shape scenario_config.json
        is saved/restored in) into the four group dicts consumed by
        _nb_analysis.py's per-cell scenario state."""
        _out = {"controls": {}, "agescale": {}, "dose": {}, "subpop": {}}
        for _k, _v in (flat or {}).items():
            _out[scenario_state_group(_k)][_k] = _v
        return _out

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

    from generic_core.model_factory import (
        build_compartment_init,
        read_initial_conditions,
    )

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
        detect_config_type,
        parse_fit_config_targets,
        scenario_state_group,
        partition_scenario_state,
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
        """Wrap ``header`` + ``body`` in a bordered, collapsible card (a native
        ``<details>`` accordion, open by default) with a colored accent stripe
        down the left edge.

        ``header``/``body`` are spliced in via ``.text`` rather than an
        f-string ``{}`` placeholder: ``Html.__format__`` joins every line with
        a space (it's meant for inlining short snippets into markdown), which
        silently collapses any ``<pre>`` blocks nested inside ``body`` (e.g.
        the config-preview JSON) onto a single line.
        """
        _stripe = accent or "#9aa7b8"
        _header_html = header.text if isinstance(header, mo.Html) else header
        _body_html = body.text if isinstance(body, mo.Html) else body
        return mo.Html(
            '<details open style="'
            "border:1px solid rgba(127,127,127,0.25);"
            f"border-left:4px solid {_stripe};"
            "border-radius:10px;"
            "padding:0.8rem 1rem;"
            "margin:0.45rem 0;"
            'background:rgba(127,127,127,0.03);">'
            '<summary style="cursor:pointer;list-style:none;'
            'display:flex;align-items:flex-start;gap:.5rem;">'
            '<span class="_clt_acc_arrow" style="display:inline-flex;'
            "align-items:center;justify-content:center;height:1.7em;"
            'flex:none;transition:transform .15s;">▶</span>'
            f'<span style="flex:1 1 auto;min-width:0;">{_header_html}</span>'
            "</summary>"
            "<style>"
            "summary::-webkit-details-marker{display:none;}"
            'details[open]>summary>._clt_acc_arrow{transform:rotate(90deg);}'
            "</style>"
            f'<div style="margin-top:0.5rem;">{_body_html}</div>'
            "</details>"
        )

    return CLT_ACCENT, tip_label, with_tip, wtip, step_header, section_card

