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
  _nb_model_builder.py        — Model Builder tab (Steps 0–10)
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
configurable age and risk groups. For multi-age/risk-group models, contact
matrices are embedded inline in the config JSON; vaccines and mobility can
be supplied as CSV files or as constant scalar values.

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

    FitResult = namedtuple("FitResult", ["best_params", "loss_curve", "num_days", "observed", "method", "accepted_params"])

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
    def parse_csv_list(text):
        return [_item.strip() for _item in text.split(",") if _item.strip()]

    def parse_infectious_mapping(text):
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

    def build_scalar_array(value, num_age_groups=1, num_risk_groups=1):
        return np.full((num_age_groups, num_risk_groups), float(value), dtype=float)

    def build_notebook_schedules_input(
        start_date,
        num_days,
        absolute_humidity,
        mobility_value,
        daily_vaccines_value,
        num_age_groups=1,
        num_risk_groups=1,
        absolute_humidity_df=None,
        school_work_calendar_df=None,
        mobility_df=None,
        daily_vaccines_df=None,
    ):
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

    def load_csv_validated(path_str, required_columns):
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

    def load_contact_matrix_csv(path_str, expected_size):
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

    def load_config_json(path_str):
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

    def validate_metapop_folder(folder_path_str):
        """Check that a metapop folder has the required files.
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
        return _valid, _status

    def infectious_mapping_to_str(mapping):
        """Convert {comp: rel_param | None} back to the text format 'IP:ip_rel, IA:ia_rel, ISR'."""
        _parts = []
        for _k, _v in mapping.items():
            _parts.append(f"{_k}:{_v}" if _v else _k)
        return ", ".join(_parts)

    def is_array_param(cfg, name):
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
        validate_metapop_folder,
        infectious_mapping_to_str,
        is_array_param,
    )

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
def _main_tab_selector(mo):
    main_tab = mo.ui.tabs({
        "Model Builder": mo.md(""),
        "Fitting":        mo.md(""),
        "Forecast":       mo.md(""),
        "Export":         mo.md(""),
        "Analysis":       mo.md(""),
        "Documentation":  mo.md(""),
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
def _load_config_state(mo):
    get_config_path, set_config_path = mo.state(
        "/Users/rfp437/Work/CityLevelTransmission/CLT_BaseModel/generic_core/examples/example_metapop_inputs/model_config.json"
    )
    return get_config_path, set_config_path


@app.cell
def _load_config_ui(mo, get_config_path, set_config_path):
    config_file_upload = mo.ui.file(
        filetypes=[".json"],
        label="Browse for config JSON",
    )
    config_path_input = mo.ui.text(
        value=get_config_path(),
        on_change=set_config_path,
        placeholder="/path/to/model_config.json  (or use Browse above)",
        label="Or enter config JSON path directly",
        full_width=True,
    )
    return (config_file_upload, config_path_input)


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
    loaded_config, load_config_json, mo, main_tab
):
    mo.stop(main_tab.value != "Model Builder", None)

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

    _parts = [
        mo.md("### Step 0 — Load Existing Config"),
        config_file_upload,
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
    mo.vstack(_parts)
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

        **Quick start:** work through Steps 1–10 in order, then press **Run simulation**.
        Load a previously saved config in **Step 0** to restore any prior setup.
        """
    )
    return


@app.cell
def _instructions(mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.accordion({
        "📋 Workflow overview": mo.md("""
**Step 0 — Load existing config** *(optional)*
Enter the path to a `model_config.json` file to pre-populate all fields below.
Leave blank to start fresh.

**Step 1 — Population structure**
Choose the number of age groups (A) and risk groups (R).
Select *Single population* or *Metapopulation* mode.
For metapop, enter the path to a folder containing the required input files (see the
*Metapopulation folder conventions* section below).

**Step 2 — Compartments**
Enter compartment names as a comma-separated list, e.g. `S, E, I, R`.

**Step 3 — Transitions**
Define each transition: origin compartment → destination, and the rate template.
Available templates:
- `constant_param` — fixed rate from a single parameter
- `param_product` — product of multiple parameters (with optional complement factors)
- `immunity_modulated` — base rate adjusted by infection/vaccine immunity (M, MV)
- `force_of_infection` — standard FOI with contact matrix and optional humidity/immunity
- `force_of_infection_travel` — FOI with inter-subpop travel mixing (metapop only)

**Step 4 — Parameters**
Numeric sliders appear automatically for every parameter name referenced by your transitions.

**Step 5 — Schedules and Immunity**
For rate templates that use schedules (humidity, mobility, vaccines):
- Choose *constant* to use a single scalar value for the whole simulation.
- Choose *csv* to load a real time-varying schedule from a CSV file.
Contact matrices (total, school, work) are always stored as inline arrays in the
config JSON. When A > 1, supply them via CSV paths in the contact matrix fields —
or load a saved config that already has them embedded. Risk groups (R > 1) affect
transition and susceptibility parameters but do not require separate contact matrices.

**Step 6 — Model diagram**
Auto-generated from your compartments and transitions. Requires `graphviz`; falls back
to a simple matplotlib diagram if not installed.

**Step 7 — Initial conditions**
Set total population N and the count seeded into each non-first compartment.
In metapopulation mode, per-subpop initial conditions are read from
`initial_conditions_{name}.csv` in the metapop folder (see below).

**Step 8 — Simulation settings**
Days, deterministic vs. stochastic, number of replicates, RNG seed, timesteps per day.

**Step 9 — Config preview and download**
The full config JSON (including file paths and age/risk group settings) is shown and
can be downloaded. The downloaded file can be re-loaded in Step 0.

**Step 10 — Run**
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
The folder path is entered in **Step 1** and saved in the config JSON.

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

If `initial_conditions_{name}.json` is absent, all compartments are initialised to zero and the simulation will stop with an error if the model has more than one age or risk group.

**Example folder** is included in the repository at
`generic_core/examples/example_metapop_inputs/` (2 subpops, 3 age groups, 2 risk groups,
SEIR model). Re-generate it with::

    python generic_core/examples/generate_example_metapop_data.py
        """),

        "💾 Config save / load round-trip": mo.md("""
The downloaded `model_config.json` contains everything needed to restore the session:

- Compartment names, transition definitions, parameter values
- Age/risk group counts (`age_risk` section)
- File paths for all CSV schedules (`input_files` section)
- Total population
- Per-subpopulation parameter overrides (`subpop_params` section, if present)

**To reload:** paste the path into the Step 0 text field. All UI fields (compartments,
transitions, parameters, immunity toggles, file paths, metapop folder) will be
pre-populated automatically.

**Note on contact matrices:** When A > 1, matrix values *are* embedded inline in the
config JSON under `params` (as nested lists). CSV file paths are optional — if provided
in Step 5, they override the inline values at load time. If no CSV paths are set, the
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

**Infectious compartments field** uses the format `CompartmentName:relative_infectivity_param`
(or just `CompartmentName` if all compartments are equally infectious), comma-separated.
Example: `IP:IP_relative_inf, IA:IA_relative_inf, ISR, ISH`
        """),
    })
    return


# ---------------------------------------------------------------------------
# Step 1 — Population Structure
# ---------------------------------------------------------------------------


@app.cell
def _population_structure_ui(mo, loaded_config):
    _ar = loaded_config.get("age_risk", {})
    _inf = loaded_config.get("input_files", {})
    num_age_groups_input = mo.ui.number(
        start=1, stop=20, step=1,
        value=int(_ar.get("num_age_groups", 1)),
        label="Number of age groups (A)",
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
        num_age_groups_input,
        num_risk_groups_input,
        pop_mode_radio,
        metapop_folder_input,
    )


@app.cell
def _population_structure_compute(
    num_age_groups_input,
    num_risk_groups_input,
    pop_mode_radio,
):
    num_age_groups = int(num_age_groups_input.value)
    num_risk_groups = int(num_risk_groups_input.value)
    is_metapop = pop_mode_radio.value == "Metapopulation"
    return (num_age_groups, num_risk_groups, is_metapop)


@app.cell
def _population_structure_show(
    mo,
    main_tab,
    num_age_groups,
    num_risk_groups,
    is_metapop,
    num_age_groups_input,
    num_risk_groups_input,
    pop_mode_radio,
    metapop_folder_input,
    validate_metapop_folder,
):
    mo.stop(main_tab.value != "Model Builder", None)

    _parts = [
        mo.md("### Step 1 — Population Structure"),
        mo.hstack([num_age_groups_input, num_risk_groups_input], justify="start"),
        pop_mode_radio,
    ]

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
                    "Use CSV file paths in Step 5 for schedule data."
                ),
                kind="info",
            ))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 2 — Compartments
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
def _compartments_display(compartments, compartments_text, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    if compartments:
        _body = mo.md("**Parsed:** " + "  ".join(f"`{_c}`" for _c in compartments))
    else:
        _body = mo.callout(mo.md("Enter at least one compartment name."), kind="warn")
    mo.vstack([
        mo.md("### Step 2 — Compartments"),
        compartments_text,
        _body,
    ])
    return


# ---------------------------------------------------------------------------
# Step 3 — Transitions
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
        mo.ui.text(value=_tget(_i, "name", f"t{_i+1}"), label="Name")
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
        mo.ui.text(value=_rcget(_i, "param", f"param_{_i+1}"), label="Param name")
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
            return ", ".join(f"{_k}:{_v}" if _v else _k for _k, _v in _raw.items())
        return "I"

    t_infectious = mo.ui.array([
        mo.ui.text(
            value=_infectious_default(_i),
            label="",
            placeholder="IP:IP_relative_inf, IA:IA_relative_inf, ISR, ISH",
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
        t_param, t_factors, t_complements,
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
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
):
    mo.stop(main_tab.value != "Model Builder", None)
    import html as _html
    import random as _random

    def _tip_label(label_text, tip_text):
        """Render a field label with an inline ⓘ hover tooltip using CSS only."""
        _uid = _random.randint(10**7, 10**8 - 1)
        _esc = _html.escape(tip_text)
        return mo.Html(
            f"<style>"
            f"#tip{_uid}{{position:relative;display:inline-block;"
            f"cursor:help;color:#888;font-size:0.8em;vertical-align:middle;}}"
            f"#tip{_uid}>span{{visibility:hidden;opacity:0;"
            f"transition:opacity .15s;transition-delay:.2s;"
            f"position:absolute;bottom:120%;left:0;"
            f"background:#222;color:#fff;border-radius:4px;"
            f"padding:6px 10px;width:280px;font-size:12px;line-height:1.5;"
            f"white-space:pre-wrap;pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            f"</style>"
            f"<span>"
            f"{label_text}&nbsp;"
            f'<span id="tip{_uid}">ⓘ<span>{_esc}</span></span>'
            f"</span>"
        )

    def _with_tip(label_text, tip_text, widget):
        return mo.hstack([_tip_label(label_text, tip_text), widget], justify="start", align="center")

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
        "in Step 5, otherwise immunity_force stays at 1."
    )

    def _immunity_checkbox(checkbox):
        return mo.hstack([checkbox, _tip_label("", _IMMUNITY_TIP)], justify="start", align="center")


    _n = int(n_transitions.value)
    _rows = []
    for _i in range(_n):
        _template = t_template.value[_i]

        if _template == "constant_param":
            _rate_ui = t_param[_i]
        elif _template == "param_product":
            _rate_ui = mo.vstack([
                _with_tip(
                    "Factors",
                    "Comma-separated parameter names multiplied together to form the rate.\n\n"
                    "Example: base_rate, hosp_prop\n"
                    "Rate = base_rate × hosp_prop\n\n"
                    "Each name gets a slider in Step 4.",
                    t_factors[_i],
                ),
                _with_tip(
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
                _with_tip(
                    "Infectious compartments",
                    "Comma-separated compartment names, optionally followed by\n"
                    ":param_name to supply a relative infectivity parameter.\n\n"
                    "Format:  Compartment  or  Compartment:param_name\n\n"
                    "Example: I, A:a_rel_inf\n"
                    "→ Compartment A has a_rel_inf× the instantaneous infectiousness\n"
                    "  of I (set its value in Step 4).\n\n"
                    "Omit the parameter to treat all listed compartments as equally infectious.",
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
        else:
            _foit_items = [
                t_beta[_i],
                t_rel_sus[_i],
                _with_tip(
                    "Infectious compartments",
                    "Comma-separated compartment names, optionally followed by\n"
                    ":param_name to supply a relative infectivity parameter.\n\n"
                    "Format:  Compartment  or  Compartment:param_name\n\n"
                    "Example: I, A:a_rel_inf\n"
                    "→ Compartment A has a_rel_inf× the instantaneous infectiousness\n"
                    "  of I (set its value in Step 4).\n\n"
                    "Omit the parameter to treat all listed compartments as equally infectious.",
                    t_infectious[_i],
                ),
                t_use_humidity[_i],
            ]
            if t_use_humidity.value[_i]:
                _foit_items.append(t_humidity_impact[_i])
            _foit_items.append(_immunity_checkbox(t_use_foi_immunity[_i]))
            if t_use_foi_immunity.value[_i]:
                _foit_items.extend([t_inf_reduce[_i], t_vax_reduce[_i]])
            _foit_items.append(_with_tip(
                "Immobile compartments",
                "Comma-separated compartment names whose members do NOT travel\n"
                "between subpopulations (no cross-subpop mixing).\n\n"
                "Example: H, ICU",
                t_immobile[_i],
            ))
            _rate_ui = mo.vstack(_foit_items)

        _rows.append(mo.vstack([
            mo.md(f"**Transition {_i + 1}**"),
            mo.vstack([
                mo.hstack([t_name[_i], t_origin[_i], t_dest[_i]], justify="start"),
                _with_tip(
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
                    "See the ⚡ Rate template quick reference accordion above for full details.",
                    t_template[_i],
                ),
            ]),
            _rate_ui,
            mo.md("---"),
        ]))

    mo.vstack([
        mo.md("### Step 3 — Transitions"),
        n_transitions,
        *_rows,
    ])
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

    uses_absolute_humidity = _uses_absolute_humidity
    uses_contact_matrix = _uses_contact_matrix
    uses_mobility = _uses_mobility
    requires_immunity_metrics = _requires_immunity_metrics
    return uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics


@app.cell
def _collect_param_names(
    n_transitions, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact, t_use_foi_immunity,
    parse_csv_list, parse_infectious_mapping,
):
    _n = int(n_transitions.value)
    _names = []
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
            _names.extend([_p for _p in parse_infectious_mapping(t_infectious.value[_i]).values() if _p])
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
            _names.extend([_p for _p in parse_infectious_mapping(t_infectious.value[_i]).values() if _p])

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

    return param_names, reduce_param_names


# ---------------------------------------------------------------------------
# Step 4 — Parameters
# ---------------------------------------------------------------------------


@app.cell
def _params_ui(param_names, reduce_param_names, loaded_config, mo, is_array_param):
    _saved_params = loaded_config.get("params", {})
    scalar_param_names = [_n for _n in param_names if not is_array_param(loaded_config, _n)]
    array_param_names  = [_n for _n in param_names if is_array_param(loaded_config, _n)]
    params_inputs = mo.ui.array([
        mo.ui.number(
            start=0.0, stop=10.0, step=0.001,
            value=float(_saved_params.get(_name, 0.5 if _name in reduce_param_names else 1.0)),
            label=_name,
        )
        for _name in scalar_param_names
    ])
    return params_inputs, scalar_param_names, array_param_names


@app.cell
def _params_show(param_names, params_inputs, scalar_param_names, array_param_names, loaded_config, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _saved_params = loaded_config.get("params", {})
    _parts = [mo.md("### Step 4 — Parameters")]
    if not param_names:
        _parts.append(mo.callout(mo.md("No transition parameters found yet."), kind="warn"))
    if scalar_param_names:
        _parts.append(mo.hstack(list(params_inputs), wrap=True))
    for _name in array_param_names:
        _val = _saved_params[_name]
        _parts.append(mo.callout(
            mo.md(
                f"**`{_name}`** — loaded from config as A×R array "
                f"(slider disabled, value passes through unchanged)\n\n"
                f"```json\n{_val}\n```"
            ),
            kind="info",
        ))
    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 5 — Schedules and Immunity (scalar inputs + CSV file paths)
# ---------------------------------------------------------------------------


@app.cell
def _schedule_and_immunity_ui(mo, loaded_config):
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
    absolute_humidity_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.0001, value=0.006, label="Absolute humidity",
    )
    total_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=1.0, label="Total contact matrix value",
    )
    school_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=0.0, label="School contact subtraction",
    )
    work_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=0.1, value=0.0, label="Work contact subtraction",
    )
    mobility_input = mo.ui.number(
        start=0.0, stop=5.0, step=0.01, value=1.0, label="Mobility modifier",
    )
    daily_vaccines_input = mo.ui.number(
        start=0.0, stop=1e9, step=1.0, value=0.0, label="Daily vaccines",
    )
    return (
        include_inf_immunity,
        include_vax_immunity,
        absolute_humidity_input,
        total_contact_input,
        school_contact_input,
        work_contact_input,
        mobility_input,
        daily_vaccines_input,
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
        start=0.0, stop=1.0, step=0.01,
        value=float(_saved_params.get("inf_induced_saturation", 0.0)),
        label="inf_induced_saturation",
    )
    vax_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.01,
        value=float(_saved_params.get("vax_induced_saturation", 0.0)),
        label="vax_induced_saturation",
    )
    inf_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.001,
        value=float(_saved_params.get("inf_induced_immune_wane", 0.01)),
        label="inf_induced_immune_wane",
    )
    _vax_wane_raw = _saved_params.get("vax_induced_immune_wane", 0.0)
    vax_wane_is_array = isinstance(_vax_wane_raw, list)
    vax_wane_loaded_val = _vax_wane_raw
    vax_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.001,
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

    # Auto-detect absolute_humidity.csv from metapop folder when not explicitly saved
    _ah_csv_saved = _inf.get("absolute_humidity_csv", "")
    if not _ah_csv_saved and is_metapop and metapop_folder_input.value.strip():
        _candidate = Path(metapop_folder_input.value.strip()) / "absolute_humidity.csv"
        if _candidate.exists():
            _ah_csv_saved = str(_candidate)

    ah_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _ah_csv_saved else "constant",
        label="Absolute humidity source",
    )
    ah_path = mo.ui.text(
        value=_ah_csv_saved,
        placeholder="/path/to/absolute_humidity.csv",
        label="Absolute humidity CSV",
        full_width=True,
    )
    cal_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("school_work_calendar_csv") else "constant",
        label="School/work calendar source",
    )
    cal_path = mo.ui.text(
        value=_inf.get("school_work_calendar_csv", ""),
        placeholder="/path/to/school_work_calendar.csv",
        label="School/work calendar CSV",
        full_width=True,
    )
    mob_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("mobility_csv") else ("csv" if _multi else "constant"),
        label="Mobility source",
    )
    mob_path = mo.ui.text(
        value=_inf.get("mobility_csv", ""),
        placeholder="/path/to/mobility_modifier.csv",
        label="Mobility CSV",
        full_width=True,
    )
    vax_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("vaccines_csv") else ("csv" if _multi else "constant"),
        label="Vaccines source",
    )
    vax_path = mo.ui.text(
        value=_inf.get("vaccines_csv", ""),
        placeholder="/path/to/daily_vaccines.csv",
        label="Vaccines CSV",
        full_width=True,
    )
    total_contact_csv_path = mo.ui.text(
        value=_inf.get("total_contact_matrix_csv", ""),
        placeholder="/path/to/total_contact_matrix.csv",
        label="Total contact matrix CSV (A×A plain floats)",
        full_width=True,
    )
    school_contact_csv_path = mo.ui.text(
        value=_inf.get("school_contact_matrix_csv", ""),
        placeholder="/path/to/school_contact_matrix.csv",
        label="School contact matrix CSV (A×A plain floats)",
        full_width=True,
    )
    work_contact_csv_path = mo.ui.text(
        value=_inf.get("work_contact_matrix_csv", ""),
        placeholder="/path/to/work_contact_matrix.csv",
        label="Work contact matrix CSV (A×A plain floats)",
        full_width=True,
    )
    return (
        ah_mode, ah_path,
        cal_mode, cal_path,
        mob_mode, mob_path,
        vax_mode, vax_path,
        total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    )


@app.cell
def _schedule_csv_show(
    mo,
    num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
    ah_mode, ah_path,
    cal_mode, cal_path,
    mob_mode, mob_path,
    vax_mode, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    load_csv_validated, load_contact_matrix_csv,
    SimpleNamespace,
):
    _multi = (num_age_groups > 1) or (num_risk_groups > 1)
    _parts = [mo.md("#### Schedule File Inputs")]

    # Absolute humidity
    _ah_df = None
    if uses_absolute_humidity:
        _parts.append(ah_mode)
        if ah_mode.value == "csv":
            _parts.append(ah_path)
            if ah_path.value.strip():
                _ah_df, _ah_err = load_csv_validated(
                    ah_path.value, ["date", "absolute_humidity"]
                )
                if _ah_err:
                    _parts.append(mo.callout(mo.md(f"**Humidity CSV:** {_ah_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Humidity CSV: {len(_ah_df)} rows loaded."), kind="success"
                    ))

    # School/work calendar
    _cal_df = None
    if uses_contact_matrix:
        _parts.append(cal_mode)
        if cal_mode.value == "csv":
            _parts.append(cal_path)
            if cal_path.value.strip():
                _cal_df, _cal_err = load_csv_validated(
                    cal_path.value, ["date", "is_school_day", "is_work_day"]
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
        _parts.append(mob_mode)
        if mob_mode.value == "csv":
            _parts.append(mob_path)
            if mob_path.value.strip():
                _mob_df, _mob_err = load_csv_validated(mob_path.value, [])
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
    if include_vax_immunity.value:
        _parts.append(vax_mode)
        if vax_mode.value == "csv":
            _parts.append(vax_path)
            if vax_path.value.strip():
                _vax_df, _vax_err = load_csv_validated(
                    vax_path.value, ["date", "daily_vaccines"]
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
        _parts.append(mo.md("**Contact matrices (required when A > 1):**"))
        _parts.append(total_contact_csv_path)
        if total_contact_csv_path.value.strip():
            _total_contact_mat, _tc_err = load_contact_matrix_csv(
                total_contact_csv_path.value, num_age_groups
            )
            if _tc_err:
                _parts.append(mo.callout(mo.md(f"**Total contact matrix:** {_tc_err}"), kind="danger"))
            else:
                _parts.append(mo.callout(
                    mo.md(f"Total contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                    kind="success",
                ))
        else:
            _parts.append(mo.callout(
                mo.md(
                    "Total contact matrix CSV not set. "
                    "Inline value from loaded config will be used if available, "
                    "otherwise scalar `[[1.0]]`."
                ),
                kind="info",
            ))

        _parts.append(school_contact_csv_path)
        if school_contact_csv_path.value.strip():
            _school_contact_mat, _sc_err = load_contact_matrix_csv(
                school_contact_csv_path.value, num_age_groups
            )
            if _sc_err:
                _parts.append(mo.callout(mo.md(f"**School contact matrix:** {_sc_err}"), kind="danger"))
            else:
                _parts.append(mo.callout(
                    mo.md(f"School contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                    kind="success",
                ))

        _parts.append(work_contact_csv_path)
        if work_contact_csv_path.value.strip():
            _work_contact_mat, _wc_err = load_contact_matrix_csv(
                work_contact_csv_path.value, num_age_groups
            )
            if _wc_err:
                _parts.append(mo.callout(mo.md(f"**Work contact matrix:** {_wc_err}"), kind="danger"))
            else:
                _parts.append(mo.callout(
                    mo.md(f"Work contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                    kind="success",
                ))

    mo.vstack(_parts)

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
    absolute_humidity_input,
    total_contact_input,
    school_contact_input,
    work_contact_input,
    mobility_input,
    daily_vaccines_input,
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
    ah_mode,
    cal_mode,
    mob_mode,
    vax_mode,
    num_age_groups,
):
    mo.stop(main_tab.value != "Model Builder", None)
    import html as _html
    import random as _random

    def _tip_label(label_text, tip_text):
        _uid = _random.randint(10**7, 10**8 - 1)
        _esc = _html.escape(tip_text)
        return mo.Html(
            f"<style>"
            f"#tip{_uid}{{position:relative;display:inline-block;"
            f"cursor:help;color:#888;font-size:0.8em;vertical-align:middle;}}"
            f"#tip{_uid}>span{{visibility:hidden;opacity:0;"
            f"transition:opacity .15s;transition-delay:.2s;"
            f"position:absolute;bottom:120%;left:0;"
            f"background:#222;color:#fff;border-radius:4px;"
            f"padding:6px 10px;width:300px;font-size:12px;line-height:1.5;"
            f"white-space:pre-wrap;pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            f"</style>"
            f"<span>"
            f"{label_text}&nbsp;"
            f'<span id="tip{_uid}">ⓘ<span>{_esc}</span></span>'
            f"</span>"
        )

    def _wtip(widget, tip_text):
        return mo.hstack([widget, _tip_label("", tip_text)], justify="start", align="center")

    _parts = [
        mo.md("### Step 5 — Schedules and Immunity"),
        mo.hstack([
            _wtip(
                include_inf_immunity,
                "Track population-level infection-induced immunity (M).\n\n"
                "For instance, if driven by (R→S) transitions:\n\n"
                "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                "M increases when recently-recovered individuals re-enter\n"
                "the susceptible pool (R→S), and decays via waning.\n\n"
                "Must be enabled for inf_reduce_param (Step 3) to have effect.",
            ),
            _wtip(
                include_vax_immunity,
                "Track population-level vaccine-induced immunity (MV).\n\n"
                "MV grows with daily vaccine doses and decays via waning:\n\n"
                "ΔMV = daily_vaccines − wane×MV\n\n"
                "Must be enabled for vax_reduce_param (Step 3) to have effect.",
            ),
        ], wrap=True),
    ]

    _scalar_schedule_inputs = []
    if uses_absolute_humidity and ah_mode.value == "constant":
        _scalar_schedule_inputs.append(absolute_humidity_input)
    if uses_contact_matrix:
        if num_age_groups == 1:
            if cal_mode.value == "constant":
                _scalar_schedule_inputs.extend([
                    total_contact_input, school_contact_input, work_contact_input,
                ])
        # when A > 1 contact matrices come from CSV or inline config params; no scalar fallback shown
    if uses_mobility and mob_mode.value == "constant":
        _scalar_schedule_inputs.append(mobility_input)

    if _scalar_schedule_inputs:
        _parts.append(mo.hstack(_scalar_schedule_inputs, wrap=True))
    elif not uses_absolute_humidity and not uses_contact_matrix and not uses_mobility:
        _parts.append(mo.md("*No schedule-backed rate templates selected.*"))

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
                _wtip(
                    r_to_s_picker,
                    "The transition that drives immunity gain.\n\n"
                    "M increases as people move from R back to S — recently-\n"
                    "recovered individuals re-entering the susceptible pool\n"
                    "still carry partial immunity.\n\n"
                    "Select the transition that represents this R→S flow.",
                ),
                _wtip(
                    inf_sat_input,
                    "Limits how much M can grow as immunity accumulates.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → M saturates at a lower level.\n"
                    "0 = no saturation limit.",
                ),
                _wtip(
                    vax_sat_input,
                    "How much vaccine immunity (MV) dampens further gain in M.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → MV reduces M accumulation more.\n"
                    "0 = vaccine and infection immunity are independent.",
                ),
                _wtip(
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
                    _wtip(
                        vax_wane_input,
                        "Daily decay rate of vaccine-induced immunity MV.\n\n"
                        "ΔMV = daily_vaccines − wane×MV\n\n"
                        "0 = no waning.\n"
                        "0.01 ≈ half-life of ~70 days.",
                    )
                )
            _metric_inputs.extend([vax_delay_input, vax_reset_date_input])
            if vax_mode.value == "constant":
                _metric_inputs.append(daily_vaccines_input)
        _parts.append(mo.hstack(_metric_inputs, wrap=True))
    else:
        _parts.append(mo.md("*Dynamic immunity metrics disabled.*"))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 6 — Model Diagram
# ---------------------------------------------------------------------------


@app.cell
def _diagram(compartments, n_transitions, t_name, t_origin, t_dest, mo, plt, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _n = int(n_transitions.value)
    _inner = None
    _graphviz_error = None
    try:
        import graphviz as gv  # type: ignore[import-untyped]
        _dot = gv.Digraph(
            graph_attr={"rankdir": "LR", "bgcolor": "white", "pad": "0.3"},
            node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#ddeeff"},
        )
        for _c in compartments:
            _dot.node(_c)
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            _label = t_name.value[_i]
            if _origin and _dest:
                _dot.edge(_origin, _dest, label=_label)
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
            _ax.text(_x, _y, _c, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ddeeff"))
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

    mo.vstack([mo.md("### Step 6 — Model Diagram"), _inner])
    return


# ---------------------------------------------------------------------------
# Step 7 — Initial Conditions
# ---------------------------------------------------------------------------


@app.cell
def _init_ui(compartments, mo, loaded_config, num_age_groups, num_risk_groups):
    _saved_N = loaded_config.get("total_population", 10000)
    total_pop_input = mo.ui.number(
        start=1, stop=int(1e9), step=1, value=int(_saved_N), label="Total population N",
    )
    seed_compartments = compartments[1:] if len(compartments) > 1 else []
    seed_inputs = mo.ui.array([
        mo.ui.number(start=0, stop=int(1e9), step=1, value=0 if _j > 0 else 50, label=f"Initial {_c}")
        for _j, _c in enumerate(seed_compartments)
    ])
    return total_pop_input, seed_inputs


@app.cell
def _init_show(compartments, total_pop_input, seed_inputs, is_metapop, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _parts = [mo.md("### Step 7 — Initial Conditions")]
    if is_metapop:
        _parts.append(mo.callout(
            mo.md(
                "**Metapopulation mode:** initial conditions are read from "
                "`initial_conditions_{name}.csv` and `age_risk_fractions_{name}.csv` "
                "in the metapop folder."
            ),
            kind="info",
        ))
    else:
        _N = int(total_pop_input.value)
        _seeded = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
        _remainder = _N - sum(_seeded.values())
        _first = compartments[0] if compartments else "?"
        _table_rows = {_first: _remainder, **_seeded}
        _rows_md = "\n".join(f"| `{_c}` | {_v:,} |" for _c, _v in _table_rows.items())
        _parts += [
            total_pop_input,
            mo.hstack(list(seed_inputs), wrap=True) if seed_inputs.value else mo.md(""),
            mo.md(
                "| Compartment | Initial count |\n"
                "|---|---|\n"
                f"{_rows_md}"
            ),
        ]
        if _remainder < 0:
            _parts.append(mo.callout(mo.md("Seeded counts exceed total population N."), kind="danger"))
    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 8 — Simulation Settings
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
def _sim_settings_show(mo, sim_days, sim_mode, n_reps, rng_seed, timesteps, start_date_input, transition_vars_input, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.vstack([
        mo.md("### Step 8 — Simulation Settings"),
        mo.hstack([sim_days, sim_mode, timesteps, rng_seed], justify="start"),
        mo.hstack([
            n_reps,
            mo.md("*Ignored in deterministic mode.*") if sim_mode.value == "Deterministic" else mo.md(""),
        ]),
        start_date_input,
        transition_vars_input,
    ])
    return


# ---------------------------------------------------------------------------
# Build config dict
# ---------------------------------------------------------------------------


@app.cell
def _build_config(
    compartments,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
    scalar_param_names, params_inputs,
    include_inf_immunity, include_vax_immunity,
    r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input,
    vax_wane_input, vax_wane_is_array, vax_wane_loaded_val,
    vax_delay_input, vax_reset_date_input,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics,
    parse_csv_list, parse_infectious_mapping,
    total_contact_input, school_contact_input, work_contact_input,
    num_age_groups, num_risk_groups,
    is_metapop, metapop_folder_input,
    loaded_schedule_dfs,
    ah_mode, cal_mode, mob_mode, vax_mode,
    ah_path, cal_path, mob_path, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    total_pop_input,
    loaded_config,
    start_date_input, transition_vars_input,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    np,
):
    _n = int(n_transitions.value)
    _A = num_age_groups
    _R = num_risk_groups
    # Seed from loaded config first (preserves A×R array-valued params), then
    # overlay the scalar slider values for any param the user has wired up in Step 3.
    params_dict: dict = dict(loaded_config.get("params", {}))
    for _j, _name in enumerate(scalar_param_names):
        params_dict[_name] = float(params_inputs.value[_j])

    _transitions = []
    _metapop_travel_config = {}
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
                "infectious_compartments": parse_infectious_mapping(t_infectious.value[_i]),
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
        else:
            _travel_config = {
                "infectious_compartments": parse_infectious_mapping(t_infectious.value[_i]),
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

        _transitions.append({
            "name": t_name.value[_i].strip(),
            "origin": t_origin.value[_i],
            "destination": t_dest.value[_i],
            "rate_template": _template,
            "rate_config": _rate_config,
        })

    # Contact matrix params
    if uses_contact_matrix:
        if _A == 1:
            params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
            params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
            params_dict["work_contact_matrix"] = [[float(work_contact_input.value)]]
        else:
            if loaded_schedule_dfs.total_contact_matrix is not None:
                params_dict["total_contact_matrix"] = loaded_schedule_dfs.total_contact_matrix
            elif not isinstance(params_dict.get("total_contact_matrix"), list):
                params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
            if loaded_schedule_dfs.school_contact_matrix is not None:
                params_dict["school_contact_matrix"] = loaded_schedule_dfs.school_contact_matrix
            elif not isinstance(params_dict.get("school_contact_matrix"), list):
                params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
            if loaded_schedule_dfs.work_contact_matrix is not None:
                params_dict["work_contact_matrix"] = loaded_schedule_dfs.work_contact_matrix
            elif not isinstance(params_dict.get("work_contact_matrix"), list):
                params_dict["work_contact_matrix"] = [[float(work_contact_input.value)]]

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
            },
        })

    # Build input_files section (only non-empty paths)
    _input_files = {}
    if uses_absolute_humidity and ah_mode.value == "csv" and ah_path.value.strip():
        _input_files["absolute_humidity_csv"] = ah_path.value.strip()
    if uses_contact_matrix and cal_mode.value == "csv" and cal_path.value.strip():
        _input_files["school_work_calendar_csv"] = cal_path.value.strip()
    if uses_mobility and mob_mode.value == "csv" and mob_path.value.strip():
        _input_files["mobility_csv"] = mob_path.value.strip()
    if include_vax_immunity.value and vax_mode.value == "csv" and vax_path.value.strip():
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
        },
        "total_population": int(total_pop_input.value),
        "simulation_settings": {
            "start_real_date": start_date_input.value.strip(),
            "transition_variables_to_save": _tvs,
        },
    }
    if _input_files:
        config_dict["input_files"] = _input_files

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
    return config_dict, immunity_active, metapop_travel_config


# ---------------------------------------------------------------------------
# Step 9 — Config Preview
# ---------------------------------------------------------------------------


@app.cell
def _config_preview(config_dict, json, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    json_str = json.dumps(config_dict, indent=2)
    mo.vstack([
        mo.md("### Step 9 — Config Preview"),
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
    ])
    return


# ---------------------------------------------------------------------------
# Step 10 — Run
# ---------------------------------------------------------------------------


@app.cell
def _run_button(mo):
    run_button = mo.ui.run_button(label="Run simulation")
    return (run_button,)


@app.cell
def _run_section_display(run_button, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.vstack([mo.md("### Step 10 — Run"), run_button])
    return


@app.cell
def _run_sim(
    run_button,
    main_tab,
    config_dict,
    metapop_travel_config,
    compartments,
    total_pop_input,
    seed_inputs,
    sim_days,
    sim_mode,
    n_reps,
    rng_seed,
    timesteps,
    start_date_input,
    transition_vars_input,
    absolute_humidity_input,
    mobility_input,
    daily_vaccines_input,
    build_notebook_schedules_input,
    build_scalar_array,
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
):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.stop(not run_button.value, mo.md(""))

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

    # ---- Single-population run helper ----
    def _build_schedules_input_for_subpop(
        ah_df_override=None,
        cal_df_override=None,
        mob_df_override=None,
        vax_df_override=None,
    ):
        return build_notebook_schedules_input(
            start_date=start_real_date,
            num_days=_num_days,
            absolute_humidity=float(absolute_humidity_input.value),
            mobility_value=float(mobility_input.value),
            daily_vaccines_value=float(daily_vaccines_input.value),
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
        for _pname, _pval in config_dict.get("params", {}).items():
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
                f"Switch the affected schedule source to **constant** in Step 5, "
                f"or reload a config that matches the current group counts.\n\n"
                + "\n".join(f"- {_issue}" for _issue in _shape_issues)
            ),
            kind="danger",
        ),
    )

    # ---- Single-population path ----
    if not is_metapop:
        _N = int(total_pop_input.value)
        _seed_vals = {compartments[_j + 1]: int(seed_inputs.value[_j])
                     for _j in range(len(seed_inputs.value))}
        _first_comp = compartments[0]
        _remainder = _N - sum(_seed_vals.values())
        mo.stop(
            _remainder < 0,
            mo.callout(mo.md("**Initial condition error:** seeded counts exceed total population."),
                       kind="danger"),
        )

        compartment_init = {_first_comp: build_scalar_array(_remainder, _A, _R)}
        compartment_init.update({_c: build_scalar_array(_v, _A, _R) for _c, _v in _seed_vals.items()})
        for _c in compartments:
            compartment_init.setdefault(_c, build_scalar_array(0.0, _A, _R))

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
                )

                # Build initial conditions from JSON (supports A×R arrays and epi metric init)
                _sp_epi_init = {}
                _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                if _sp_ic_path.exists():
                    with open(_sp_ic_path) as _f:
                        _ic = json.load(_f)
                    for _c, _arr in _ic.get("compartments", {}).items():
                        if _c in compartments:
                            _sp_comp_init[_c] = np.array(_arr, dtype=float)
                    for _m, _arr in _ic.get("epi_metrics", {}).items():
                        _sp_epi_init[_m] = np.array(_arr, dtype=float)
                else:
                    mo.stop(
                        _A > 1 or _R > 1,
                        mo.callout(mo.md(f"**Missing:** `initial_conditions_{_sp_name}.json` is required when the model has more than one age or risk group."), kind="danger"),
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
        _rows.append(f"| `{_comp}` | {_peak:,.0f} | {_peak_day} |")
    _table = "\n".join(_rows)
    mo.vstack([
        mo.md("### Results — Summary"),
        mo.md(
            "| Compartment | Peak value (median) | Peak day (median) |\n"
            "|---|---|---|\n"
            f"{_table}"
        ),
    ])
    return

@app.cell
def _shared_model_factory(
    build_notebook_schedules_input,
    build_scalar_array,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt, flu, np, json, pd, Path,
    loaded_schedule_dfs,
    absolute_humidity_input,
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
            absolute_humidity=float(absolute_humidity_input.value),
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
            if init_states_override is not None and _si < len(init_states_override):
                _comp_init, _epi_init = init_states_override[_si]
            elif _ic_p.exists():
                with open(_ic_p) as _f:
                    _ic = json.load(_f)
                for _c, _arr in _ic.get("compartments", {}).items():
                    if _c in compartments_list:
                        _comp_init[_c] = np.array(_arr, dtype=float)
                for _m, _arr in _ic.get("epi_metrics", {}).items():
                    _epi_init[_m] = np.array(_arr, dtype=float)
            _sp_overrides = dict(param_overrides or {})
            if param_overrides_per_subpop and _si < len(param_overrides_per_subpop):
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
def _fitting_ui(mo, compartments, n_transitions, t_name, param_names):
    _tvars = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    _all_tgts = list(compartments) + _tvars
    _tgt_opts = _all_tgts if _all_tgts else ["S"]
    fit_obs_source = mo.ui.radio(
        options={"Upload CSV": "upload", "File path": "path"},
        value="Upload CSV", label="Observed data source",
    )
    fit_obs_upload = mo.ui.file(label="Upload CSV (columns: date, value)", filetypes=[".csv"])
    fit_obs_path = mo.ui.text(
        label="CSV file path", placeholder="~/data/observed.csv", full_width=True,
    )
    fit_target = mo.ui.multiselect(
        options={t: t for t in _tgt_opts},
        value=[_tgt_opts[0]],
        label="Fit target — select one or more (multiple selections are summed)",
    )
    fit_params_multiselect = mo.ui.multiselect(
        options={p: p for p in param_names},
        value=[],
        label="Parameters to fit",
    )
    fit_method = mo.ui.radio(
        options={"Adam (gradient)": "adam", "LBFGS (gradient)": "lbfgs", "Accept-reject": "ar"},
        value="Adam (gradient)", label="Fitting method",
    )
    fit_lr = mo.ui.number(value=0.01, start=1e-5, stop=1.0, step=1e-4, label="Learning rate")
    fit_n_iter = mo.ui.number(value=200, start=10, stop=2000, step=10, label="Iterations / Max samples")
    fit_r2_thresh = mo.ui.number(value=0.75, start=0.0, stop=1.0, step=0.01, label="R² acceptance threshold")
    fit_run_button = mo.ui.run_button(label="Run fitting")
    return (
        fit_obs_source, fit_obs_upload, fit_obs_path, fit_target,
        fit_params_multiselect,
        fit_method, fit_lr, fit_n_iter, fit_r2_thresh, fit_run_button,
    )


@app.cell
def _fitting_bounds_ui(
    mo, fit_params_multiselect, config_dict,
    num_age_groups, num_risk_groups, is_metapop,
):
    _saved_params = config_dict.get("params", {})
    _selected = list(fit_params_multiselect.value)
    _A = num_age_groups
    _R = num_risk_groups

    _dim_opts = ["scalar"]
    if _A > 1:
        _dim_opts.append("per age group")
    if _R > 1:
        _dim_opts.append("per risk group")
    if _A > 1 and _R > 1:
        _dim_opts.append("per age × risk")
    if is_metapop:
        _dim_opts.append("per subpopulation")

    def _default_bounds(pn):
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
            start=1e-8, stop=1e8, step=0.001,
            value=_default_bounds(_pn)[0],
            label="Lower",
        )
        for _pn in _selected
    ])
    fit_bounds_hi = mo.ui.array([
        mo.ui.number(
            start=1e-8, stop=1e8, step=0.001,
            value=_default_bounds(_pn)[1],
            label="Upper",
        )
        for _pn in _selected
    ])
    fit_param_dims = mo.ui.array([
        mo.ui.radio(options=_dim_opts, value="scalar", label="")
        for _pn in _selected
    ])
    return (fit_bounds_lo, fit_bounds_hi, fit_param_dims)


@app.cell
def _fitting_display(
    fit_obs_source, fit_obs_upload, fit_obs_path,
    fit_target, fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_method, fit_lr, fit_n_iter, fit_r2_thresh, fit_run_button,
    compartments, mo, main_tab,
):
    mo.stop(main_tab.value != "Fitting", None)
    _obs_w = fit_obs_upload if fit_obs_source.value == "upload" else fit_obs_path
    _selected_targets = list(fit_target.value)
    _selected_params = list(fit_params_multiselect.value)
    _tvars_selected = [t for t in _selected_targets if t not in compartments]

    _target_note = (
        mo.callout(
            mo.md(f"Fitting to **sum** of: {', '.join(f'`{t}`' for t in _selected_targets)}"),
            kind="info",
        )
        if len(_selected_targets) > 1 else mo.md("")
    )
    _tv_note = (
        mo.callout(
            mo.md(
                f"Transition variable targets (`{'`, `'.join(_tvars_selected)}`) require gradient "
                "or accept-reject fitting. Make sure these appear in **Transition variables to save** (Step 8)."
            ),
            kind="info",
        )
        if _tvars_selected else mo.md("")
    )

    if _selected_params:
        _header = mo.hstack(
            [mo.md("**Parameter**"), mo.md("**Lower bound**"), mo.md("**Upper bound**"), mo.md("**Dimensionality**")],
            justify="start",
        )
        _rows = [_header]
        for _j, _pn in enumerate(_selected_params):
            _rows.append(mo.hstack(
                [mo.md(f"`{_pn}`"), fit_bounds_lo[_j], fit_bounds_hi[_j], fit_param_dims[_j]],
                justify="start", align="center",
            ))
        _bounds_section = mo.vstack(_rows)
    else:
        _bounds_section = mo.md("*Select parameters above to configure bounds.*")

    mo.vstack([
        mo.md("## Fitting"),
        mo.md("**Step 1 — Observed data**"),
        mo.hstack([fit_obs_source], justify="start"),
        _obs_w,
        mo.md("**Step 2 — Target and parameters**"),
        fit_target,
        _target_note,
        _tv_note,
        fit_params_multiselect,
        _bounds_section,
        mo.md("**Step 3 — Method and hyperparameters**"),
        fit_method,
        mo.hstack([fit_lr, fit_n_iter, fit_r2_thresh], justify="start"),
        mo.md("**Step 4 — Run**"),
        fit_run_button,
    ])
    return


@app.cell
def _fitting_obs_parse(fit_obs_source, fit_obs_upload, fit_obs_path, pd, np, io):
    fit_obs_df = None
    fit_obs_array = None
    fit_obs_n_days = 0
    try:
        if fit_obs_source.value == "upload":
            if fit_obs_upload.value:
                _raw = fit_obs_upload.value[0].contents
                fit_obs_df = pd.read_csv(io.BytesIO(_raw))
        else:
            _p = fit_obs_path.value.strip()
            if _p:
                from pathlib import Path as _Path
                fit_obs_df = pd.read_csv(_Path(_p).expanduser())
    except Exception:
        fit_obs_df = None
    if fit_obs_df is not None:
        _non_id = [c for c in fit_obs_df.columns if c.lower() not in ("date", "day", "time", "week")]
        _val_col = _non_id[0] if _non_id else (fit_obs_df.columns[1] if len(fit_obs_df.columns) >= 2 else None)
        if _val_col:
            fit_obs_array = fit_obs_df[_val_col].to_numpy(dtype=float)
            fit_obs_n_days = len(fit_obs_array)
    return (fit_obs_df, fit_obs_array, fit_obs_n_days)


@app.cell
def _run_fitting(
    fit_run_button, fit_obs_array, fit_obs_n_days,
    fit_target, fit_method, fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_lr, fit_n_iter, fit_r2_thresh,
    config_dict, compartments, is_metapop,
    total_pop_input, seed_inputs,
    start_date_input, timesteps, rng_seed,
    num_age_groups, num_risk_groups,
    metapop_folder_input, metapop_travel_config,
    make_single_pop_metapop, make_metapop_from_folder, extract_history,
    build_generic_torch_inputs, generic_torch_simulate_calibration_target, RATE_TEMPLATE_REGISTRY,
    torch, _F, np, json, mo, compute_rsquared, FitResult, build_scalar_array, Path,
):
    fit_result = None
    if fit_run_button.value:
        _selected_params = list(fit_params_multiselect.value)
        _lo_vals = list(fit_bounds_lo.value)
        _hi_vals = list(fit_bounds_hi.value)
        _dim_vals = list(fit_param_dims.value)
        _fit_targets = list(fit_target.value)
        _A = num_age_groups
        _R = num_risk_groups

        mo.stop(
            not _fit_targets,
            mo.callout(mo.md("**No fit target selected.**"), kind="warn"),
        )
        mo.stop(
            fit_obs_array is None,
            mo.callout(mo.md("**No observed data.** Upload a CSV or provide a file path."), kind="warn"),
        )
        mo.stop(
            not _selected_params,
            mo.callout(mo.md("**No parameters to fit.** Select parameters above."), kind="warn"),
        )
        mo.stop(
            is_metapop and fit_method.value != "ar",
            mo.callout(mo.md("**Gradient fitting only supports single-population models.** Use accept-reject for metapop."), kind="warn"),
        )

        # Build scalar bounds dict used by gradient methods
        _bounds_grad = {
            _selected_params[_j]: [float(_lo_vals[_j]), float(_hi_vals[_j])]
            for _j in range(len(_selected_params))
        }

        _num_fit_days = fit_obs_n_days
        _N = int(total_pop_input.value)
        _seed_vals = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
        _first_comp = compartments[0]
        _ci = {_first_comp: build_scalar_array(_N - sum(_seed_vals.values()), 1, 1)}
        _ci.update({_c: build_scalar_array(_v, 1, 1) for _c, _v in _seed_vals.items()})
        for _c in compartments:
            _ci.setdefault(_c, build_scalar_array(0.0, 1, 1))

        _start = start_date_input.value.strip() or "2024-01-01"
        _ts = int(timesteps.value)
        _seed_b = int(rng_seed.value)
        _target_tvs = [t for t in _fit_targets if t not in compartments]
        _target_comps = [t for t in _fit_targets if t in compartments]
        _target_is_tv = bool(_target_tvs)
        _tvs = _target_tvs
        _obs = fit_obs_array[:_num_fit_days]
        _loss_curve = []
        _best_params = {}
        _accepted_params_list = []

        with mo.status.spinner("Running fitting..."):
            try:
                if fit_method.value in ("adam", "lbfgs"):
                    mo.stop(
                        not _target_tvs,
                        mo.callout(mo.md("**Gradient fitting requires at least one transition variable as target.** Use accept-reject for compartment-only targets."), kind="warn"),
                    )
                    mo.stop(
                        torch is None,
                        mo.callout(mo.md("**PyTorch not available.** Install torch to use gradient-based fitting."), kind="danger"),
                    )
                    _metapop, _mc, _ = make_single_pop_metapop(
                        config_dict, _start, _num_fit_days, _ci,
                        ts_per_day=_ts, stochastic=False, tvs=_tvs,
                        save_daily=False, seed_base=_seed_b,
                        travel_config=metapop_travel_config,
                    )
                    _ti = build_generic_torch_inputs(_metapop, _mc, _num_fit_days)
                    _state0 = {k: v.clone().detach() for k, v in _ti["state_dict"].items()}
                    _opt_tensors = []
                    for _j, _pn in enumerate(_selected_params):
                        if _pn not in _ti["params_dict"]:
                            continue
                        _dim = _dim_vals[_j] if _dim_vals else "scalar"
                        _existing = _ti["params_dict"][_pn]
                        # Scalar initialisation value — use mean if existing tensor is already shaped
                        _init_val = float(_existing.detach().mean().item())
                        if _dim == "per age group":
                            _t = torch.full((_A, 1), _init_val, dtype=torch.float32, requires_grad=True)
                        elif _dim == "per risk group":
                            _t = torch.full((1, _R), _init_val, dtype=torch.float32, requires_grad=True)
                        elif _dim == "per age × risk":
                            _t = torch.full((_A, _R), _init_val, dtype=torch.float32, requires_grad=True)
                        else:
                            _t = _existing.clone().detach().float().requires_grad_(True)
                        _ti["params_dict"][_pn] = _t
                        _opt_tensors.append(_t)
                    mo.stop(
                        not _opt_tensors,
                        mo.callout(mo.md("**None of the specified parameters found in params dict.** Check names."), kind="danger"),
                    )
                    _obs_t = torch.tensor(_obs, dtype=torch.float32)
                    _n_it = int(fit_n_iter.value)
                    _lr_val = float(fit_lr.value)
                    _best_loss = float("inf")

                    def _record_best_params_grad():
                        return {
                            _pn: _ti["params_dict"][_pn].detach().tolist()
                            for _pn in _selected_params if _pn in _ti["params_dict"]
                        }

                    if fit_method.value == "adam":
                        _opt = torch.optim.Adam(_opt_tensors, lr=_lr_val)
                        for _ in range(_n_it):
                            _opt.zero_grad()
                            _state = {k: v.clone() for k, v in _state0.items()}
                            _sim = generic_torch_simulate_calibration_target(
                                _state, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                                _ti["precomputed"], _ti["schedules_dict"],
                                _num_fit_days, _ts, _target_tvs,
                            )
                            _sim_agg = _sim.sum(dim=tuple(range(1, _sim.dim())))
                            _loss = _F.mse_loss(_sim_agg, _obs_t)
                            _loss.backward()
                            _opt.step()
                            for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                if _pn in _ti["params_dict"]:
                                    _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                            _lv = float(_loss.item())
                            _loss_curve.append(_lv)
                            if _lv < _best_loss:
                                _best_loss = _lv
                                _best_params = _record_best_params_grad()
                    else:  # lbfgs
                        _opt = torch.optim.LBFGS(_opt_tensors, lr=_lr_val, max_iter=20)
                        _best_loss = float("inf")
                        for _ in range(max(1, _n_it // 20)):
                            def _closure(_s0=_state0, _t=_ti, _ot=_obs_t):
                                _opt.zero_grad()
                                _s = {k: v.clone() for k, v in _s0.items()}
                                _sim = generic_torch_simulate_calibration_target(
                                    _s, _t["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                                    _t["precomputed"], _t["schedules_dict"],
                                    _num_fit_days, _ts, _target_tvs,
                                )
                                _l = _F.mse_loss(_sim.sum(dim=tuple(range(1, _sim.dim()))), _ot)
                                _l.backward()
                                return _l
                            _opt.step(_closure)
                            with torch.no_grad():
                                _s2 = {k: v.clone() for k, v in _state0.items()}
                                _out2 = generic_torch_simulate_calibration_target(
                                    _s2, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                                    _ti["precomputed"], _ti["schedules_dict"],
                                    _num_fit_days, _ts, _target_tvs,
                                )
                                _lv2 = float(_F.mse_loss(_out2.sum(dim=tuple(range(1, _out2.dim()))), _obs_t).item())
                            for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                if _pn in _ti["params_dict"]:
                                    _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                            _loss_curve.append(_lv2)
                            if _lv2 < _best_loss:
                                _best_loss = _lv2
                                _best_params = _record_best_params_grad()

                else:  # accept-reject
                    # Determine number of subpops for "per subpopulation" sampling
                    _n_subpops = 1
                    if is_metapop and any(_dim_vals[_j] == "per subpopulation" for _j in range(len(_selected_params))):
                        try:
                            _mp_cfg_p = Path(metapop_folder_input.value.strip()) / "metapop_config.json"
                            with open(_mp_cfg_p) as _f:
                                _n_subpops = len(json.load(_f).get("subpopulations", []))
                        except Exception:
                            _n_subpops = 1

                    _rng = np.random.default_rng(_seed_b)
                    _n_samples = int(fit_n_iter.value)
                    _r2_thresh = float(fit_r2_thresh.value)
                    _best_r2 = -float("inf")

                    for _s_idx in range(_n_samples):
                        _sampled = {}
                        _per_subpop = [{} for _ in range(_n_subpops)]

                        for _j, _pn in enumerate(_selected_params):
                            _lo = float(_lo_vals[_j])
                            _hi = float(_hi_vals[_j])
                            _dim = _dim_vals[_j] if _dim_vals else "scalar"
                            if _dim == "per age group":
                                _sampled[_pn] = _rng.uniform(_lo, _hi, size=(_A, 1)).tolist()
                            elif _dim == "per risk group":
                                _sampled[_pn] = _rng.uniform(_lo, _hi, size=(1, _R)).tolist()
                            elif _dim == "per age × risk":
                                _sampled[_pn] = _rng.uniform(_lo, _hi, size=(_A, _R)).tolist()
                            elif _dim == "per subpopulation":
                                _sp_vals = _rng.uniform(_lo, _hi, size=_n_subpops)
                                for _spi in range(_n_subpops):
                                    _per_subpop[_spi][_pn] = float(_sp_vals[_spi])
                            else:
                                _sampled[_pn] = float(_rng.uniform(_lo, _hi))

                        _has_per_subpop = any(_per_subpop)
                        if is_metapop:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value.strip(),
                                config_dict, _start, _num_fit_days, list(compartments),
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_tvs if _target_is_tv else [],
                                save_daily=bool(_target_comps),
                                param_overrides=_sampled or None,
                                param_overrides_per_subpop=_per_subpop if _has_per_subpop else None,
                                travel_config=metapop_travel_config or None,
                            )
                        else:
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start, _num_fit_days, _ci,
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_tvs if _target_is_tv else [],
                                save_daily=bool(_target_comps),
                                param_overrides=_sampled or None,
                                travel_config=metapop_travel_config,
                            )
                        _m.simulate_until_day(_num_fit_days)
                        _hist = extract_history(_m, list(compartments), tvs=_tvs if _target_is_tv else None)
                        _parts = [_hist[_t][:_num_fit_days] for _t in _fit_targets if _t in _hist]
                        if not _parts:
                            continue
                        _sim_s = np.array(_parts).sum(axis=0)
                        if len(_sim_s) != len(_obs):
                            continue
                        _r2 = compute_rsquared(list(_obs), _sim_s.tolist())
                        _loss_curve.append(float(_r2))
                        if _r2 > _best_r2:
                            _best_r2 = _r2
                            # Flatten best params: scalars and per-subpop combined
                            _best_params = dict(_sampled)
                            for _spi, _sp_d in enumerate(_per_subpop):
                                for _k, _v in _sp_d.items():
                                    _best_params[f"{_k}_subpop{_spi}"] = _v
                        if _r2 >= _r2_thresh:
                            _accepted_params_list.append(dict(_sampled))

            except Exception as _exc:
                mo.stop(True, mo.callout(mo.md(f"**Fitting error:** {_exc}"), kind="danger"))

        fit_result = FitResult(
            best_params=_best_params,
            loss_curve=_loss_curve,
            num_days=_num_fit_days,
            observed=list(_obs),
            method=fit_method.value,
            accepted_params=_accepted_params_list if _accepted_params_list else [_best_params],
        )
    return (fit_result,)


@app.cell
def _fitting_autosave(fit_result, output_dir, json):
    if fit_result is not None:
        _p = output_dir / "fitted_params.json"
        _p.write_text(json.dumps(fit_result.best_params, indent=2))
    return


@app.cell
def _fitting_results_display(fit_result, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md("*Run fitting to see results.*"))
    _lc = fit_result.loss_curve
    _bp = fit_result.best_params
    _method = fit_result.method
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))
    _axes[0].plot(_lc, linewidth=1.5)
    _axes[0].set_xlabel("Iteration / Sample")
    _axes[0].set_ylabel("R²" if _method == "ar" else "MSE loss")
    _axes[0].set_title(f"Fitting progress ({_method})")
    _axes[0].grid(True, alpha=0.3)
    _axes[1].axis("off")
    if _bp:
        _rows = [[_pn, f"{_v:.6g}"] for _pn, _v in _bp.items()]
        _tbl = _axes[1].table(
            cellText=_rows, colLabels=["Parameter", "Best-fit value"],
            loc="center", cellLoc="left",
        )
        _tbl.auto_set_font_size(True)
        _tbl.scale(1.2, 1.5)
        _axes[1].set_title("Best-fit parameters")
    plt.tight_layout()
    _params_md = "\n".join(f"- `{k}` = **{v:.6g}**" for k, v in _bp.items())
    _accepted_note = mo.md("")
    if _method == "ar":
        _n_acc = len(fit_result.accepted_params)
        _accepted_note = mo.callout(
            mo.md(
                f"**{_n_acc} accepted parameter set(s)** passed the R² threshold. "
                "Enable **Start forecast from fitted end-state** in the Forecast tab to run an ensemble."
            ),
            kind="success" if _n_acc > 1 else "info",
        )
    mo.vstack([mo.md("## Fitting Results"), _fig, mo.md(_params_md), _accepted_note])
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
    fit_result, mo, main_tab,
):
    mo.stop(main_tab.value != "Forecast", None)
    _path_w = forecast_params_path if not forecast_use_fitted.value else mo.md("")

    _fitted_state_note = mo.md("")
    if forecast_from_fitted_state.value:
        if fit_result is not None and forecast_use_fitted.value:
            _n_accepted = len(fit_result.accepted_params)
            _method = fit_result.method
            if _method == "ar" and _n_accepted > 1:
                _fitted_state_note = mo.callout(
                    mo.md(
                        f"**{_n_accepted} accepted parameter set(s)** from accept-reject fitting. "
                        "Each will warm-up through the fit period and seed a separate forecast trajectory."
                    ),
                    kind="success",
                )
            else:
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
                    "**Start from fitted end-state** requires **Use fitted params from Fitting tab** "
                    "to be enabled and fitting to have been run."
                ),
                kind="warn",
            )

    mo.vstack([
        mo.md("## Forecast"),
        mo.md("**Step 1 — Fitted parameters**"),
        forecast_use_fitted,
        _path_w,
        mo.md("**Step 2 — Settings**"),
        mo.hstack([forecast_horizon, forecast_n_reps], justify="start"),
        forecast_stochastic,
        forecast_from_fitted_state,
        _fitted_state_note,
        mo.md("**Step 3 — Run**"),
        forecast_run_button,
    ])
    return


@app.cell
def _run_forecast(
    forecast_run_button, forecast_use_fitted, forecast_params_path,
    forecast_from_fitted_state,
    forecast_horizon, forecast_n_reps, forecast_stochastic,
    fit_result, config_dict, compartments, is_metapop,
    metapop_folder_input, metapop_travel_config,
    total_pop_input, seed_inputs, start_date_input, timesteps, rng_seed,
    transition_vars_input,
    make_single_pop_metapop, make_metapop_from_folder, extract_history,
    np, json, mo, Path, build_scalar_array, datetime,
):
    forecast_result = None
    if forecast_run_button.value:
        _fitted_params = {}
        if forecast_use_fitted.value:
            mo.stop(
                fit_result is None,
                mo.callout(mo.md("**No fitting results.** Run fitting first or disable 'Use fitted params'."), kind="warn"),
            )
            _fitted_params = dict(fit_result.best_params)
        else:
            _pp = forecast_params_path.value.strip()
            if _pp:
                try:
                    with open(Path(_pp).expanduser()) as _f:
                        _fitted_params = json.load(_f)
                except Exception as _exc:
                    mo.stop(True, mo.callout(mo.md(f"**Could not load fitted params:** {_exc}"), kind="danger"))

        _fit_n = fit_result.num_days if (fit_result is not None and forecast_use_fitted.value) else 0
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
            _N = int(total_pop_input.value)
            _sv = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
            _fc = compartments[0]
            _ci = {_fc: build_scalar_array(_N - sum(_sv.values()), 1, 1)}
            _ci.update({_c: build_scalar_array(_v, 1, 1) for _c, _v in _sv.items()})
            for _c in compartments:
                _ci.setdefault(_c, build_scalar_array(0.0, 1, 1))

        _histories = []

        if forecast_from_fitted_state.value:
            # Two-phase: warmup through fit period → extract end-state → run forecast
            mo.stop(
                not forecast_use_fitted.value or fit_result is None,
                mo.callout(mo.md("**Start from fitted end-state** requires 'Use fitted params from Fitting tab' to be enabled."), kind="warn"),
            )
            mo.stop(
                _fit_n == 0,
                mo.callout(mo.md("**Start from fitted end-state** requires a non-zero fit period. Run fitting first."), kind="warn"),
            )

            # Ensemble of param sets: all accepted for AR, otherwise just best params
            _param_sets = fit_result.accepted_params if fit_result.accepted_params else [_fitted_params]
            _metric_names = [_m["name"] for _m in config_dict.get("epi_metrics", [])]
            _fcast_start = (
                datetime.datetime.strptime(_start, "%Y-%m-%d")
                + datetime.timedelta(days=_fit_n)
            ).strftime("%Y-%m-%d")

            def _extract_end_states(metapop_model, comps, metric_names):
                """Return list of (comp_init, epi_init) per subpop from the last simulated day."""
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
                    for _pset_idx, _pset in enumerate(_param_sets):
                        # Phase 1: deterministic warmup through the fit period
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
                        _warmup_hist = extract_history(_wm, list(compartments), tvs=_tvs)
                        _end_states = _extract_end_states(_wm, list(compartments), _metric_names)

                        # Phase 2: stochastic forecast from end-state, one rep per param set
                        if not is_metapop:
                            _end_comp, _end_epi = _end_states[0]
                            _fm, _, _ = make_single_pop_metapop(
                                config_dict, _fcast_start, _horizon, _end_comp,
                                seed_offset=_pset_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                epi_metric_init=_end_epi or None,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _fm, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _fcast_start, _horizon,
                                list(compartments),
                                seed_offset=_pset_idx, seed_base=_seed_b, ts_per_day=_ts,
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
            # Original path: single simulation from Step 7 initial conditions
            with mo.status.spinner("Running forecast..."):
                try:
                    for _rep in range(_reps):
                        if not is_metapop:
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start, _total_days, _ci,
                                seed_offset=_rep, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_fitted_params or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _start, _total_days, list(compartments),
                                seed_offset=_rep, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_fitted_params or None,
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
def _forecast_results_display(forecast_result, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Forecast", None)
    mo.stop(forecast_result is None, mo.md("*Run forecast to see results.*"))
    _hists = forecast_result["histories"]
    _comps = forecast_result["compartments"]
    _fit_n = forecast_result["fit_n_days"]
    _total = forecast_result["total_days"]
    _days = np.arange(1, _total + 1)
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _fig, _ax = plt.subplots(figsize=(12, 5))
    for _ci, _comp in enumerate(_comps):
        _color = _colors[_ci % len(_colors)]
        _arrays = [_h[_comp] for _h in _hists if _comp in _h]
        if not _arrays:
            continue
        _mat = np.stack(_arrays, axis=0)
        _n = min(len(_days), _mat.shape[1])
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
def _export_display(config_dict, fit_result, output_dir, json, mo, main_tab):
    mo.stop(main_tab.value != "Export", None)
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

# Define scenarios: {name: {param: value}}
SCENARIOS = {
    "baseline": {},
    # "high_beta": {"beta_baseline": 0.4},
}

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


def build_model(cfg, param_overrides, rep):
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
    _reps_data = []
    for _rep in range(NUM_REPS):
        _m = build_model(config_dict, overrides, _rep)
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
    mo.vstack([
        mo.md("## Export"),
        mo.md("### Generated script  *(edit `SCENARIOS` and top constants before running)*"),
        mo.accordion({"run_simulation.py": mo.md(f"```python\n{_script}\n```")}),
        mo.md("### Downloads"),
        mo.hstack([_config_dl, _script_dl, _fitted_dl], justify="start"),
        mo.md(f"*Outputs auto-saved to `{output_dir}/`*"),
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
def _analysis_param_catalog(scalar_param_names, params_inputs, array_param_names, loaded_config):
    import math as _math

    _saved_params = loaded_config.get("params", {})
    _scalar = {
        scalar_param_names[_j]: float(params_inputs.value[_j])
        for _j in range(len(scalar_param_names))
    }
    _array = {_n: _saved_params[_n] for _n in array_param_names if _n in _saved_params}
    _params = {**_scalar, **_array}

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
        _step = max(1e-5, round(_base / 1000, 6)) if _base > 0 else 0.001
        return mo.ui.number(start=0.0, stop=_stop, step=_step, value=_base)

    analysis_scenario_scalar_inputs = mo.ui.array([
        mo.ui.array([_make_scalar_input(pname) for _ in range(_MAX_SC)])
        for pname in _scalar_names
    ]) if _scalar_names else mo.ui.array([])

    analysis_scenario_array_scales = mo.ui.array([
        mo.ui.array([mo.ui.number(start=0.0, stop=10.0, step=0.01, value=1.0) for _ in range(_MAX_SC)])
        for _ in _array_names
    ]) if _array_names else mo.ui.array([])

    return (
        analysis_n_scenarios, analysis_scenario_names,
        analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
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
):
    mo.stop(main_tab.value != "Analysis", None)
    _n_sc = int(analysis_n_scenarios.value)
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())

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

    _sens_ui = mo.vstack([
        mo.md("**Vary one parameter across N values — each value becomes a scenario.**"),
        mo.hstack([analysis_param_type, _param_w, analysis_n_values], justify="start"),
        mo.md(f"Varying `{_pname}` ({_fmt}):"),
        mo.hstack(list(analysis_sens_sliders), wrap=True),
        _sens_preview,
    ])

    # --- Scenario sub-tab ---
    _header = mo.hstack(
        [mo.md("**Parameter**")] + [analysis_scenario_names[j] for j in range(_n_sc)],
        widths="equal",
    )
    _scalar_rows = [
        mo.hstack(
            [mo.md(f"`{_pn}`")] + [analysis_scenario_scalar_inputs[_i][j] for j in range(_n_sc)],
            widths="equal",
        )
        for _i, _pn in enumerate(_scalar_names)
    ]
    _array_rows = [
        mo.hstack(
            [mo.md(f"`{_pn}` ×scale")] + [analysis_scenario_array_scales[_k][j] for j in range(_n_sc)],
            widths="equal",
        )
        for _k, _pn in enumerate(_array_names)
    ]
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
        mo.md("## Analysis"),
        analysis_sub_tab,
        _tab_body.get(analysis_sub_tab.value, mo.md("")),
        mo.md("---\n**Run settings**"),
        mo.hstack([analysis_sim_days, analysis_n_reps, analysis_timesteps], justify="start"),
        mo.hstack([
            analysis_stochastic,
            mo.md("*Ignored — using 1 replicate.*") if not analysis_stochastic.value else mo.md(""),
        ], justify="start"),
        mo.hstack([analysis_subpop_selector, analysis_age_selector], justify="start"),
        mo.md("**Compartments / metrics to display:**"),
        mo.hstack(list(analysis_comp_checkboxes), wrap=True, justify="start"),
        analysis_run_button,
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
):
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    analysis_scenarios = []

    if analysis_sub_tab.value == "Sensitivity":
        _is_array = analysis_param_type.value == "Array (scale factor)"
        if _is_array:
            _pname = analysis_array_param_sel.value
            _base_arr = np.asarray(ANALYSIS_ARRAY_PARAMS.get(_pname, [1.0]))
            for _v in list(dict.fromkeys(analysis_sens_sliders.value)):
                analysis_scenarios.append((
                    f"{_pname} ×{_v:.3g}",
                    {_pname: (_base_arr * _v).tolist()},
                ))
        else:
            _pname = analysis_scalar_param_sel.value
            for _v in list(dict.fromkeys(analysis_sens_sliders.value)):
                analysis_scenarios.append((f"{_pname}={_v:.4g}", {_pname: float(_v)}))
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
            analysis_scenarios.append((_name, _overrides))

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
    total_pop_input, seed_inputs, start_date_input, rng_seed,
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
        _N = int(total_pop_input.value)
        _sv = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
        _fc = compartments[0]
        _ci = {_fc: build_scalar_array(_N - sum(_sv.values()), 1, 1)}
        _ci.update({_c: build_scalar_array(_v, 1, 1) for _c, _v in _sv.items()})
        for _c in compartments:
            _ci.setdefault(_c, build_scalar_array(0.0, 1, 1))

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
            for _scen_name, _overrides in analysis_scenarios:
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

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(11, 4 * _n_combos), squeeze=False)

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

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(11, 4 * _n_combos), squeeze=False)

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
        figsize=(max(5 * _n_met, 6), 5 * _n_combos),
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
    _fig, _axes = plt.subplots(_n_plots, 1, figsize=(10, 5 * _n_plots), squeeze=False)
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

## Tab 1 — Model Builder

**Purpose:** Define the structure of your epidemic model and do a quick preview simulation.

### Steps

| Step | What you configure |
|------|--------------------|
| 0 — Load config | Optionally load a previously saved `model_config.json` to pre-fill all fields. |
| 1 — Population structure | Number of age groups, risk groups, and whether to use a metapopulation. |
| 2 — Compartments | Name each compartment (e.g. `S`, `E`, `I`, `R`).  The first compartment receives the bulk of the initial population. |
| 3 — Transitions | Define flows between compartments.  Each transition needs a name, a "from" compartment, a "to" compartment, and a rate template (e.g. `standard_infection`, `fixed_rate`). |
| 4 — Parameters | Set numeric values for all parameters referenced by your rate templates (e.g. `beta_baseline`, `sigma`, `gamma`). |
| 5 — Schedules & immunity | Optionally upload CSVs for time-varying schedules: absolute humidity, school/work calendars, mobility, and daily vaccines. |
| 6 — Diagram | Preview the compartment diagram generated from your transitions. |
| 7 — Initial conditions | Set the total population and seed counts for compartments 2–N. |
| 8 — Simulation settings | Choose deterministic vs stochastic, number of replicates, timesteps per day, start date, and which transition variables to save. |
| 9 — Config preview & download | Review the full `model_config.json` and download it. |
| 10 — Run | Click **Run simulation** to see epidemic curves and a summary table. |

**Metapopulation mode:** When enabled, Step 1 asks for a folder path containing:
- `metapop_config.json` — subpopulation names and travel matrix
- `initial_conditions_<SubpopName>.json` — per-subpop initial conditions
- Optional per-subpop schedule CSVs (`school_work_calendar_<name>.csv`, `vaccines_<name>.csv`)
- Optional shared schedule CSVs (`absolute_humidity.csv`, `mobility_modifier.csv`)

**Auto-save:** The model config is written to `{output_dir}/model_config.json` every time
any setting changes, so you never lose your work.

---

## Tab 2 — Fitting

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
   - *Accept-reject* — Parameter-space random search that runs all samples and collects
     every draw whose R² meets the threshold.  Works with any target (compartment or
     transition) and does not require PyTorch.  All accepted sets are stored and can be
     used as an ensemble in the Forecast tab.
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

## Tab 3 — Forecast

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

### Starting from the fitted end-state

Enable **Start forecast from fitted end-state** to anchor the forecast at the model
state reached at the *end* of the fitted period rather than at the fixed initial
conditions from Step 7 of Model Builder.

When this switch is on, the notebook runs in two phases:

1. **Warm-up** — A deterministic simulation is run forward from the original initial
   conditions through the entire fit period using the fitted parameters.  This recovers
   the compartment counts *and* the immunity metrics (M, MV) that the model would have
   at the moment the observed time series ends.
2. **Forecast** — A new simulation is started from that end-state, using the correct
   calendar date as its start, and runs forward for the chosen horizon.

**Accept-reject ensemble:** When the Fitting tab has found multiple accepted parameter
sets (any sample whose R² met the threshold), each set drives a separate warm-up and
produces a separate forecast trajectory.  This gives you a proper ensemble that reflects
both parameter uncertainty and stochastic noise.  The number of accepted sets is shown
in the Fitting tab after running.

For gradient-based methods (Adam, LBFGS), there is only one solution, so the warm-up
uses that single parameter set; you can still run multiple stochastic replicates from
the fitted end-state by increasing "Replicates".

> **When to use this:** If you are fitting mid-season (e.g. the observed series ends
> partway through a flu wave), this option ensures the forecast trajectory begins at a
> state the model actually passed through — with the correct levels of susceptibles,
> exposed individuals, and accumulated immunity — rather than projecting forward from an
> artificial starting point.

---

## Tab 4 — Export

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

## Tab 5 — Analysis

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

## Typical workflow

```
Model Builder  →  Fitting  →  Forecast  →  Export
                    ↓
                 Analysis
```

1. Build your model in **Model Builder** and confirm the epidemic curves look sensible.
2. Go to **Fitting**, upload observed data, and fit key parameters.
3. Check the fit overlay in Fitting results, then switch to **Forecast** to project forward.
4. Use **Analysis** to quantify uncertainty (sensitivity) or compare policy scenarios.
5. When ready to run larger ensembles, go to **Export**, download the script and configs,
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
```json
{
  "subpopulations": {"SubpopA": {}, "SubpopB": {}},
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
