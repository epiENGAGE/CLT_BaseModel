# _nb_shared.py
# Section: Shared imports and helper functions
# Part of model_builder_notebook.py — assembled by build_notebook.py

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

