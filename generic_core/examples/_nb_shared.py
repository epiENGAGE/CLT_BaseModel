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
        clt, flu, gc, io, json, mo, np, pd, plt,
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

    def parse_infectious_mapping(text: str) -> dict[str, str | None]:
        """Parse 'IP:ip_rel, IA:ia_rel, ISR' into {compartment: rel_param | None}."""
        _mapping = {}
        for _item in parse_csv_list(text):
            if ":" in _item:
                _name, _rel = _item.split(":", 1)
                _name = _name.strip()
                _rel = _rel.strip()
                if _name:
                    _mapping[_name] = _rel or None
            elif _item:
                _mapping[_item] = None
        return _mapping

    def build_scalar_array(value, num_age_groups: int = 1, num_risk_groups: int = 1) -> "np.ndarray":
        """Return an (A×R) array filled with ``value``."""
        return np.full((num_age_groups, num_risk_groups), float(value), dtype=float)

    def build_notebook_schedules_input(
        start_date,
        num_days: int,
        absolute_humidity: float,
        mobility_value: float,
        daily_vaccines_value: float,
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

        _vax_payload = json.dumps(
            np.full((num_age_groups, num_risk_groups), float(daily_vaccines_value)).tolist()
        )
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

    return (
        build_notebook_schedules_input,
        build_scalar_array,
        parse_csv_list,
        parse_infectious_mapping,
        load_csv_validated,
        load_contact_matrix_csv,
        load_config_json,
        resolve_input_path,
        validate_metapop_folder,
        infectious_mapping_to_str,
        is_array_param,
    )

