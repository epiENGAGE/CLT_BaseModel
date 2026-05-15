"""
model_builder_notebook.py
=========================

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
configurable age and risk groups. For multi-age/risk-group models, schedule
data (contact matrices, vaccines, mobility) must be supplied as CSV files.

Metapopulation folder conventions
----------------------------------
Required files:
  subpopulations.csv           cols: name, total_population
  travel_matrix.csv            NxN matrix, subpop names as header row and index col

Optional shared files (all subpops):
  absolute_humidity.csv        cols: date, absolute_humidity
  mobility_modifier.csv        cols: day_of_week, mobility_modifier (JSON A×R array)

Optional per-subpop files ({name} = subpop name):
  school_work_calendar_{name}.csv   cols: date, is_school_day, is_work_day
  vaccines_{name}.csv               cols: date, daily_vaccines (JSON A×R array)
  initial_conditions_{name}.csv     cols: compartment, value
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import sys
    import json
    import io
    from pathlib import Path
    from types import SimpleNamespace

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

    return (
        Path, SimpleNamespace, clt, flu, gc, io, json, mo, np, pd, plt,
        ConfigDrivenMetapopModel, ConfigDrivenSubpopModel,
        build_state_from_config, build_params_from_config,
        parse_model_config_from_dict,
    )


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
        _required = ["subpopulations.csv", "travel_matrix.csv"]
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
    )


# ---------------------------------------------------------------------------
# Step 0 — Load Existing Config
# ---------------------------------------------------------------------------

@app.cell
def _load_config_ui(mo):
    config_file_upload = mo.ui.file(
        filetypes=[".json"],
        label="Browse for config JSON",
    )
    config_path_input = mo.ui.text(
        value="",
        placeholder="/path/to/model_config.json  (or use Browse above)",
        label="Or enter config JSON path directly",
        full_width=True,
    )
    return (config_file_upload, config_path_input)


@app.cell
def _load_config_parse(config_file_upload, config_path_input, load_config_json, json, mo):
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

    _parts = [
        mo.md("### Step 0 — Load Existing Config"),
        config_file_upload,
        config_path_input,
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
            _label = _source if _source != "path" else f"Loaded from `{config_path_input.value.strip()}`"
            _parts.append(mo.callout(
                mo.md(
                    f"{_label} — **{_n_comp}** compartments, **{_n_tr}** transitions, "
                    f"**{_A}** age group(s), **{_R}** risk group(s)."
                ),
                kind="success",
            ))
    mo.vstack(_parts)
    return (loaded_config,)


# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------

@app.cell
def _intro(mo):
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
def _instructions(mo):
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
For rate templates that use schedules (contact matrix, humidity, mobility, vaccines):
- Choose *constant* to use a single scalar value for the whole simulation.
- Choose *csv* to load a real time-varying schedule from a CSV file.
When A > 1, contact matrix CSVs are required and vaccine/mobility CSVs are
strongly recommended. Risk groups (R > 1) affect transition and susceptibility parameters
but do not require separate contact matrices.

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

**Contact matrix CSVs** — plain floats, A×A, no header row, no index column
```
7.0,3.0,0.5
3.0,9.0,1.5
0.5,1.5,4.0
```
Separate files for total, school, and work contact matrices.
        """),

        "🗂️ Metapopulation folder conventions": mo.md("""
Create a folder with files following these naming conventions.
The folder path is entered in **Step 1** and saved in the config JSON.

**Required files:**

| File | Description |
|---|---|
| `subpopulations.csv` | One row per subpop; columns: `name`, `total_population` |
| `travel_matrix.csv` | N×N matrix; subpop names as header row and index column |

**Optional shared files** (used by all subpops if present):

| File | Description |
|---|---|
| `absolute_humidity.csv` | `date`, `absolute_humidity` |
| `mobility_modifier.csv` | `day_of_week`, `mobility_modifier` (JSON A×R) |

**Optional per-subpop files** (`{name}` = the subpop's name from `subpopulations.csv`):

| File | Description |
|---|---|
| `school_work_calendar_{name}.csv` | `date`, `is_school_day`, `is_work_day` |
| `vaccines_{name}.csv` | `date`, `daily_vaccines` (JSON A×R) |
| `initial_conditions_{name}.csv` | `compartment`, `value` — seeded counts per compartment |

If `initial_conditions_{name}.csv` is absent, the entire subpop population is placed
in the first compartment (e.g., S).

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

**To reload:** paste the path into the Step 0 text field. All UI fields (compartments,
transitions, parameters, immunity toggles, file paths, metapop folder) will be
pre-populated automatically.

**Note on contact matrices:** When A > 1, matrix values are *not* embedded in the JSON
(they stay in the CSV files). The config stores only the CSV paths, which are re-read
at run time. Move or rename the CSV files and the paths in the config will need updating.
        """),

        "⚡ Rate template quick reference": mo.md("""
| Template | When to use | Required rate_config keys |
|---|---|---|
| `constant_param` | Fixed transition rate | `param` |
| `param_product` | Product of factors (e.g., base rate × proportion) | `factors` (list); optionally `complement_factors` |
| `immunity_modulated` | Rate reduced by cumulative immunity M/MV | `base_rate`, `proportion`, `is_complement`; optionally `inf_reduce_param`, `vax_reduce_param` |
| `force_of_infection` | Standard SIR-style infection | `beta_param`, `contact_matrix_schedule`, `infectious_compartments`, `relative_susceptibility_param`; optionally humidity/immunity fields |
| `force_of_infection_travel` | FOI with inter-subpop mixing | Same as above plus `travel_config` with `immobile_compartments`, `mobility_schedule` |

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
def _population_structure_show(
    mo,
    num_age_groups_input,
    num_risk_groups_input,
    pop_mode_radio,
    metapop_folder_input,
    validate_metapop_folder,
):
    num_age_groups = int(num_age_groups_input.value)
    num_risk_groups = int(num_risk_groups_input.value)
    is_metapop = pop_mode_radio.value == "Metapopulation"

    _parts = [
        mo.md("### Step 1 — Population Structure"),
        mo.hstack([num_age_groups_input, num_risk_groups_input], justify="start"),
        pop_mode_radio,
    ]

    if is_metapop:
        _parts.append(metapop_folder_input)
        _folder_valid, _folder_status = validate_metapop_folder(metapop_folder_input.value)
        if metapop_folder_input.value.strip():
            for _fname, _msg in _folder_status.items():
                _kind = "success" if "OK" in _msg else ("warn" if "absent" in _msg else "danger")
                _parts.append(mo.callout(mo.md(f"**{_fname}**: {_msg}"), kind=_kind))
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
    return (num_age_groups, num_risk_groups, is_metapop)


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
def _compartments_display(compartments, compartments_text, mo):
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
            label="Rate template",
        )
        for _i in range(_max_t)
    ])

    t_param = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "param", f"param_{_i+1}"), label="Param name")
        for _i in range(_max_t)
    ])
    t_factors = mo.ui.array([
        mo.ui.text(value=", ".join(_rcget(_i, "factors", [])), label="Factors")
        for _i in range(_max_t)
    ])
    t_complements = mo.ui.array([
        mo.ui.text(
            value=", ".join(_rcget(_i, "complement_factors", [])),
            label="Complement factors",
        )
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

    def _infectious_default(i):
        _raw = _rcget(i, "infectious_compartments", None)
        if _raw and isinstance(_raw, dict):
            return ", ".join(f"{_k}:{_v}" if _v else _k for _k, _v in _raw.items())
        return "I"

    t_infectious = mo.ui.array([
        mo.ui.text(
            value=_infectious_default(_i),
            label="Infectious compartments",
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
            value=", ".join(_rcget(_i, "immobile_compartments", [])),
            label="Immobile compartments",
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
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
):
    _n = int(n_transitions.value)
    _rows = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _rate_ui = t_param[_i]
        elif _template == "param_product":
            _rate_ui = mo.vstack([t_factors[_i], t_complements[_i]])
        elif _template == "immunity_modulated":
            _rate_ui = mo.vstack([
                t_base_rate[_i],
                t_proportion[_i],
                t_is_complement[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        elif _template == "force_of_infection":
            _rate_ui = mo.vstack([
                t_beta[_i],
                t_rel_sus[_i],
                t_infectious[_i],
                t_use_humidity[_i],
                t_humidity_impact[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        else:
            _rate_ui = mo.vstack([
                t_beta[_i],
                t_use_humidity[_i],
                t_humidity_impact[_i],
                t_use_foi_immunity[_i],
                t_inf_reduce[_i],
                t_vax_reduce[_i],
                t_infectious[_i],
                t_rel_sus[_i],
                t_immobile[_i],
            ])

        _rows.append(mo.vstack([
            mo.md(f"**Transition {_i + 1}**"),
            mo.vstack([
                mo.hstack([t_name[_i], t_origin[_i], t_dest[_i]], justify="start"),
                t_template[_i],
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
    return (param_names,)


# ---------------------------------------------------------------------------
# Step 4 — Parameters
# ---------------------------------------------------------------------------

@app.cell
def _params_ui(param_names, loaded_config, mo):
    _saved_params = loaded_config.get("params", {})
    params_inputs = mo.ui.array([
        mo.ui.number(
            start=0.0, stop=10.0, step=0.1,
            value=float(_saved_params.get(_name, 1.0)),
            label=_name,
        )
        for _name in param_names
    ])
    return (params_inputs,)


@app.cell
def _params_show(param_names, params_inputs, mo):
    _body = (
        mo.hstack(list(params_inputs), wrap=True)
        if param_names
        else mo.callout(mo.md("No transition parameters found yet."), kind="warn")
    )
    mo.vstack([mo.md("### Step 4 — Parameters"), _body])
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
    vax_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=0.001,
        value=float(_saved_params.get("vax_induced_immune_wane", 0.0)),
        label="vax_induced_immune_wane",
    )
    return r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input, vax_wane_input


@app.cell
def _schedule_csv_ui(
    mo, loaded_config, num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
):
    _inf = loaded_config.get("input_files", {})
    _multi = (num_age_groups > 1) or (num_risk_groups > 1)

    ah_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("absolute_humidity_csv") else "constant",
        label="Absolute humidity source",
    )
    ah_path = mo.ui.text(
        value=_inf.get("absolute_humidity_csv", ""),
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
                    f"Total contact matrix CSV not set. "
                    "Scalar `[[1.0]]` will be used (may be incorrect for A>1)."
                ),
                kind="warn",
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

    if len(_parts) > 1:
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
    _parts = [
        mo.md("### Step 5 — Schedules and Immunity"),
        mo.hstack([include_inf_immunity, include_vax_immunity], wrap=True),
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
        # when A > 1 contact matrices come from CSV only; no scalar fallback shown
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
            _metric_inputs.extend([r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input])
        if include_vax_immunity.value:
            _metric_inputs.extend([vax_wane_input])
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
def _diagram(compartments, n_transitions, t_name, t_origin, t_dest, mo, plt):
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
def _init_show(compartments, total_pop_input, seed_inputs, mo):
    _N = int(total_pop_input.value)
    _seeded = {compartments[_j + 1]: int(seed_inputs.value[_j]) for _j in range(len(seed_inputs.value))}
    _remainder = _N - sum(_seeded.values())
    _first = compartments[0] if compartments else "?"
    _table_rows = {_first: _remainder, **_seeded}
    _rows_md = "\n".join(f"| `{_c}` | {_v:,} |" for _c, _v in _table_rows.items())
    _parts = [
        mo.md("### Step 7 — Initial Conditions"),
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
    sim_days = mo.ui.number(start=10, stop=730, step=10, value=100, label="Simulation days")
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    n_reps = mo.ui.number(start=1, stop=100, step=1, value=10, label="Replicates")
    rng_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="RNG seed")
    timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    return sim_days, sim_mode, n_reps, rng_seed, timesteps


@app.cell
def _sim_settings_show(mo, sim_days, sim_mode, n_reps, rng_seed, timesteps):
    mo.vstack([
        mo.md("### Step 8 — Simulation Settings"),
        mo.hstack([sim_days, sim_mode, timesteps, rng_seed], justify="start"),
        mo.hstack([
            n_reps,
            mo.md("*Ignored in deterministic mode.*") if sim_mode.value == "Deterministic" else mo.md(""),
        ]),
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
    param_names, params_inputs,
    include_inf_immunity, include_vax_immunity,
    r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input, vax_wane_input,
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
    np,
):
    _n = int(n_transitions.value)
    _A = num_age_groups
    _R = num_risk_groups
    params_dict: dict = {
        _name: float(params_inputs.value[_j])
        for _j, _name in enumerate(param_names)
    }

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
            else:
                params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
            if loaded_schedule_dfs.school_contact_matrix is not None:
                params_dict["school_contact_matrix"] = loaded_schedule_dfs.school_contact_matrix
            else:
                params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
            if loaded_schedule_dfs.work_contact_matrix is not None:
                params_dict["work_contact_matrix"] = loaded_schedule_dfs.work_contact_matrix
            else:
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
        params_dict["vax_induced_immune_wane"] = float(vax_wane_input.value)
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
    }
    if _input_files:
        config_dict["input_files"] = _input_files

    immunity_active = _immunity_active
    metapop_travel_config = _metapop_travel_config
    return config_dict, immunity_active, metapop_travel_config


# ---------------------------------------------------------------------------
# Step 9 — Config Preview
# ---------------------------------------------------------------------------

@app.cell
def _config_preview(config_dict, json, mo):
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
def _run_section_display(run_button, mo):
    mo.vstack([mo.md("### Step 10 — Run"), run_button])
    return


@app.cell
def _run_sim(
    run_button,
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
    num_age_groups,
    num_risk_groups,
    is_metapop,
    metapop_folder_input,
    loaded_schedule_dfs,
    Path,
    pd,
):
    mo.stop(not run_button.value, mo.md(""))

    _A = num_age_groups
    _R = num_risk_groups
    start_real_date = "2024-01-01"
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

    def _build_subpop(schedules_input, compartment_init, seed_offset, name="aggregate_pop"):
        _config_err = None
        _model_config = None
        try:
            _model_config = parse_model_config_from_dict(
                config_dict, schedules_input=schedules_input
            )
        except Exception as _exc:
            _config_err = str(_exc)
        if _config_err is not None:
            raise RuntimeError(f"Config error: {_config_err}")
        _state = build_state_from_config(_model_config, compartment_init, epi_metric_init={})
        _params = build_params_from_config(_model_config, num_age_groups=_A, num_risk_groups=_R)
        _settings = clt.SimulationSettings(
            timesteps_per_day=_ts_per_day,
            transition_type=_transition_type,
            start_real_date=start_real_date,
            save_daily_history=True,
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
                _c: np.array(_subpop.compartments[_c].history_vals_list).squeeze()
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
        _subpops_csv = _folder / "subpopulations.csv"
        _travel_csv = _folder / "travel_matrix.csv"
        mo.stop(
            not _subpops_csv.exists(),
            mo.callout(mo.md("**Missing:** `subpopulations.csv` in metapop folder."), kind="danger"),
        )
        mo.stop(
            not _travel_csv.exists(),
            mo.callout(mo.md("**Missing:** `travel_matrix.csv` in metapop folder."), kind="danger"),
        )

        _subpops_df = pd.read_csv(_subpops_csv)
        _subpops_df = _subpops_df.loc[:, ~_subpops_df.columns.str.match(r"^Unnamed")]
        _travel_df = pd.read_csv(_travel_csv, index_col=0)
        _travel_arr = _travel_df.values.astype(float)
        _sp_names = list(_subpops_df["name"])
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
            for _sp_idx, _sp_row in _subpops_df.iterrows():
                _sp_name = _sp_row["name"]
                _sp_total = int(_sp_row["total_population"])

                # Load per-subpop schedule files
                _sp_cal_path = _folder / f"school_work_calendar_{_sp_name}.csv"
                _sp_vax_path = _folder / f"vaccines_{_sp_name}.csv"
                _sp_ic_path = _folder / f"initial_conditions_{_sp_name}.csv"

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

                # Build initial conditions
                if _sp_ic_path.exists():
                    _ic_df = pd.read_csv(_sp_ic_path)
                    _ic_df = _ic_df.loc[:, ~_ic_df.columns.str.match(r"^Unnamed")]
                    _ic_map = dict(zip(_ic_df["compartment"], _ic_df["value"].astype(float)))
                    _sp_comp_init = {
                        _c: build_scalar_array(_ic_map.get(_c, 0.0), _A, _R)
                        for _c in compartments
                    }
                else:
                    # Default: everyone in first compartment
                    _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                    _sp_comp_init[compartments[0]] = build_scalar_array(float(_sp_total), _A, _R)

                _subpop, _mc = _build_subpop(
                    _sched, _sp_comp_init, seed_offset + _sp_idx, name=_sp_name
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

            # Aggregate histories by summing across subpops
            return {
                _c: sum(
                    np.array(_sp.compartments[_c].history_vals_list).squeeze()
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
def _plot_curves(histories, compartments, sim_days, sim_mode, is_metapop, np, plt, mo):
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
def _summary_stats(histories, compartments, np, mo):
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


if __name__ == "__main__":
    app.run()
