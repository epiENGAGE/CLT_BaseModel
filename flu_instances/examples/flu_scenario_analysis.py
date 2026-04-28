"""
flu_scenario_analysis.py
==================================

Interactive marimo notebook demonstrating end-to-end multi-scenario
vaccination analysis using ``ScenarioRunner``.

Run with::

    marimo run flu_scenario_analysis.py
    marimo run flu_instances/examples/flu_scenario_analysis.py

or open in edit mode::

    marimo edit flu_scenario_analysis.py

Workflow
--------
1. Load the Austin 2024-2025 baseline model.
2. Define counterfactual vaccine schedules with user-chosen coverage scaling
   factors (up to five scenarios).
3. Run all scenarios, N stochastic replicates each (or deterministically),
   paired with the same seeds across scenarios.  All state variable histories
   and transition variable histories are saved to a persistent SQLite database
   alongside this file.
4. Compute and display:
   - Daily hospital admissions / infections by scenario (median + 95 % CI
     ribbon), with dropdowns for metric, subpopulation, and age group.
   - Vaccine-preventable events (VPH): box plot, with dropdowns for metric,
     subpopulation, and age group.
   - Age-stratified metric by scenario, including a «Total» bar, with
     dropdowns for subpopulation and metric.
   - Summary table: scenario | mean VPH | 95 % CI | mean deaths averted | 95 % CI.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import os
    import sys
    os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

    import io
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import marimo as mo
    import clt_toolkit as clt
    import flu_core as flu
    from clt_toolkit.utils import daily_sum_over_timesteps
    from flu_core.flu_outcomes import (
        daily_hospital_admissions,
        daily_new_infections,
        daily_deaths,
        cumulative_hospitalizations,
        cumulative_deaths,
        attack_rate,
        summarize_outcomes,
    )

    return (
        attack_rate,
        clt,
        cumulative_deaths,
        cumulative_hospitalizations,
        daily_deaths,
        daily_hospital_admissions,
        daily_new_infections,
        daily_sum_over_timesteps,
        flu,
        io,
        mo,
        np,
        pd,
        plt,
        summarize_outcomes,
    )
    
@app.cell
def _filename(sys, mo):
    if len(sys.argv) > 0:
        notebook_filename = sys.argv[0]
        # mo.md(f"**Current Notebook Filename:** `{notebook_filename}`")
        mo.md(f"`{notebook_filename}`")
    mo.md(f"`Flu Scenario Analysis`")
    return




@app.cell
def _controls(mo):
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    num_reps_input = mo.ui.number(
        start=10, stop=500, step=10, value=100,
        label="Number of replicates per scenario",
    )
    sim_days_input = mo.ui.number(
        start=50, stop=365, step=10, value=200,
        label="Simulation days",
    )
    start_date_input = mo.ui.date(
        value="2024-08-14",
        label="Simulation start date",
    )
    run_button = mo.ui.run_button(label="Run scenario analysis")
    return num_reps_input, run_button, sim_days_input, sim_mode, start_date_input


@app.cell
def _scenario_controls(mo):
    num_scenarios_input = mo.ui.number(
        start=1, stop=5, step=1, value=2,
        label="Number of counterfactual scenarios",
    )
    # Up to five scale-factor inputs wrapped in mo.ui.array so that marimo
    # tracks value changes on individual elements and re-runs dependent cells.
    scale_inputs = mo.ui.array([
        mo.ui.number(start=0.0, stop=3.0, step=0.05,
                     value=1.10 + i * 0.10, label=f"Scale factor {i + 1}")
        for i in range(5)
    ])
    return num_scenarios_input, scale_inputs


@app.cell
def _tab_selector(mo):
    # Created in its own cell so the widget is never recreated by user input
    # changes — preserving the selected tab across reactive updates.
    scenario_tab = mo.ui.tabs({
        "Vaccination coverage": mo.md(""),
        "Parameter overrides":  mo.md(""),
    })
    return (scenario_tab,)


@app.cell
def _show_controls(
    mo,
    num_reps_input,
    num_scenarios_input,
    param_inputs,
    run_button,
    scale_inputs,
    scenario_name_inputs,
    scenario_tab,
    sim_days_input,
    sim_mode,
    start_date_input,
    ARRAY_BASELINES,
    ARRAY_PARAMS,
    SCALAR_PARAMS,
    E_NONZERO_ENTRIES,
    array_scale_inputs,
    e_init_inputs,
    m_scale_inputs,
    start_date_sc_inputs,
    vax_editors,
):
    _n = int(num_scenarios_input.value)
    _reps_note = (
        mo.md("*Replicates ignored in deterministic mode.*")
        if sim_mode.value == "Deterministic"
        else mo.md("")
    )

    # --- Tab 1: Vaccination coverage ---
    _tab1 = mo.vstack([
        mo.md("### Vaccine coverage scenarios"),
        mo.md(
            "*The tables below show cumulative vaccination rates per age group and risk group over "
            "the simulation (or 1st year if longer). "
            "Use the scale factor as a quick uniform multiplier — changing it resets the table below. "
            "Edit individual cells for per-(subpopulation, age group, risk group) control.*"
        ),
        num_scenarios_input,
        mo.hstack(scale_inputs[:_n]),
        mo.vstack([
            mo.vstack([mo.md(f"**Scenario {_i + 1}**"), vax_editors[_i]])
            for _i in range(_n)
        ]),
    ])

    # --- Tab 2: Parameter overrides ---
    _header_row = mo.hstack(
        [mo.md("**Parameter**")] + [scenario_name_inputs[j] for j in range(_n)],
        widths="equal",
    )
    _vax_row = mo.hstack(
        [mo.md("`vax_scale_factor`")] + [scale_inputs[j] for j in range(_n)],
        widths="equal",
    )
    _scalar_rows = [
        mo.hstack(
            [mo.md(f"`{_pname}`")] + [param_inputs[_i][j] for j in range(_n)],
            widths="equal",
        )
        for _i, _pname in enumerate(SCALAR_PARAMS)
    ]
    _array_rows = [
        mo.hstack(
            [mo.md(f"`{_pname}`\n\n_{_base}_")] +
            [array_scale_inputs[_k][j] for j in range(_n)],
            widths="equal",
        )
        for _k, (_pname, _base) in enumerate(zip(ARRAY_PARAMS, ARRAY_BASELINES))
    ]
    _e_rows = [
        mo.hstack(
            [mo.md(f"`{_spn}_E[{_ii},{_jj}]`")] + [e_init_inputs[_idx][j] for j in range(_n)],
            widths="equal",
        )
        for _idx, (_spn, _ii, _jj) in enumerate(E_NONZERO_ENTRIES)
    ]
    from flu_example_utils import SUBPOP_CONFIG as _SC_M
    _m_rows = [
        mo.hstack(
            [mo.md(f"`{sp['name']}_M_scale`")] + [m_scale_inputs[_mi][j] for j in range(_n)],
            widths="equal",
        )
        for _mi, sp in enumerate(_SC_M)
    ]
    _start_date_row = mo.hstack(
        [mo.md("`start_real_date`")] + [start_date_sc_inputs[j] for j in range(_n)],
        widths="equal",
    )
    _vax_editors_section = mo.vstack([
        mo.md(
            "*The tables below show cumulative vaccination rates per age group and risk group over "
            "the simulation (or 1st year if longer). "
            "Use the scale factor as a quick uniform multiplier — changing it resets the table below. "
            "Edit individual cells for per-(subpopulation, age group, risk group) control.*"
        ),
        mo.vstack([
            mo.vstack([mo.md(f"**Scenario {_j + 1}**"), vax_editors[_j]])
            for _j in range(_n)
        ]),
    ])
    _tab2 = mo.vstack(
        [mo.md("### Parameter overrides"), num_scenarios_input,
         _header_row, _vax_row,
         mo.md("**Vaccination coverage**"), _vax_editors_section,
         mo.md("**Scalar parameters**")] + _scalar_rows +
        [mo.md("**Array parameters** *(scale factor applied to all entries)*")] + _array_rows +
        [mo.md("**Initial conditions**")] + _e_rows + _m_rows +
        [mo.md("**Other**"), _start_date_row]
    )

    _content = _tab1 if scenario_tab.value == "Vaccination coverage" else _tab2

    mo.vstack([
        mo.md("## Scenario analysis settings"),
        mo.hstack([sim_mode, num_reps_input, sim_days_input, start_date_input]),
        _reps_note,
        scenario_tab,
        _content,
        run_button,
    ])
    return


@app.cell
def _cumulative_vax_display(mo, num_scenarios_input, vax_editors):
    import pandas as _pd_cvd

    _n = int(num_scenarios_input.value)
    _any_exceed = False
    for _i in range(_n):
        _df = vax_editors[_i].value
        if not isinstance(_df, _pd_cvd.DataFrame):
            _df = _pd_cvd.DataFrame(_df)
        _risk_cols = [c for c in _df.columns if c.startswith("risk_")]
        if len(_risk_cols) > 0 and float(_df[_risk_cols].to_numpy().max()) > 1.0:
            _any_exceed = True
            break

    mo.callout(
        mo.md("**Warning:** one or more cumulative vaccination rates exceed 1.0."),
        kind="warn",
    ) if _any_exceed else mo.md("")
    return


@app.cell
def _load_files(clt, flu, pd):
    from flu_example_utils import load_flu_inputs, SUBPOP_CONFIG as _SC, SHARED_FILES_CONFIG
    inputs = load_flu_inputs(_SC, SHARED_FILES_CONFIG, clt, flu, pd)
    params = inputs["params_baseline"]
    mixing_params = inputs["mixing_params"]
    settings_base = inputs["settings_base"]
    return inputs, mixing_params, params, settings_base


@app.cell
def _compute_baseline_vax(inputs, np):
    from flu_example_utils import compute_cumulative_vax, SUBPOP_CONFIG as _SC
    baseline_vax = {
        sp["name"]: compute_cumulative_vax(inputs["vaccines_df"][sp["name"]], 1.0, np)
        for sp in _SC
    }
    return (baseline_vax,)


@app.cell
def _vax_editor_inputs(baseline_vax, mo, scale_inputs):
    from flu_example_utils import SUBPOP_CONFIG as _SC
    _MAX_SC = 5
    _first_sp = _SC[0]["name"]
    _n_age, _n_risk = baseline_vax[_first_sp].shape

    def _make_editor_rows(scale):
        rows = []
        for sp in _SC:
            _bl = baseline_vax[sp["name"]]
            _vals = _bl * scale
            for _age in range(_n_age):
                row = {"subpopulation": sp["name"], "age_group": _age}
                for _risk in range(_n_risk):
                    row[f"risk_{_risk}"] = round(float(_vals[_age, _risk]), 3)
                rows.append(row)
        return rows

    vax_editors = mo.ui.array([
        mo.ui.data_editor(
            _make_editor_rows(scale_inputs[i].value),
            label=f"Scenario {i + 1}",
        )
        for i in range(_MAX_SC)
    ])
    return (vax_editors,)


@app.cell
def _param_tab_controls(flu, inputs, mo, params):
    import dataclasses as _dc
    import numpy as _np

    _MAX_SC = 5

    _SKIP = {
        "num_age_groups", "num_risk_groups", "start_real_date",
        "vax_immunity_reset_date_mm_dd", "vax_protection_delay_days",
        "total_pop_age_risk",
    }
    _CONTACT_MATRICES = {"total_contact_matrix", "school_contact_matrix", "work_contact_matrix"}

    # Scalar numeric params
    SCALAR_PARAMS = [
        f.name for f in _dc.fields(flu.FluSubpopParams)
        if f.name not in _SKIP
        and isinstance(getattr(params, f.name), (int, float))
    ]

    # Array params (excludes contact matrices and skipped fields)
    ARRAY_PARAMS = [
        f.name for f in _dc.fields(flu.FluSubpopParams)
        if f.name not in (_SKIP | _CONTACT_MATRICES)
        and not isinstance(getattr(params, f.name), (int, float))
        and getattr(params, f.name) is not None
    ]

    # Baseline vector formatted for display
    def _fmt(val):
        arr = _np.asarray(val)
        if arr.ndim == 2:
            return "[" + ", ".join(
                "[" + ", ".join(f"{x:.4g}" for x in row) + "]" for row in arr
            ) + "]"
        return "[" + ", ".join(f"{x:.4g}" for x in arr.flatten()) + "]"
    ARRAY_BASELINES = [_fmt(getattr(params, p)) for p in ARRAY_PARAMS]

    # Scenario name inputs (one per possible scenario)
    scenario_name_inputs = mo.ui.array([
        mo.ui.text(value=f"Scenario {j + 1}", label=f"Scenario {j + 1} name")
        for j in range(_MAX_SC)
    ])

    # Scalar param value inputs: [param_idx][scenario_idx]
    def _make_scalar_input(pname):
        _base = float(getattr(params, pname) or 0.0)
        _stop = max(10.0, _base * 20) if _base > 0 else 10.0
        _step = max(1e-5, round(_base / 1000, 6)) if _base > 0 else 0.001
        return mo.ui.number(start=0.0, stop=_stop, step=_step, value=_base)

    param_inputs = mo.ui.array([
        mo.ui.array([_make_scalar_input(pname) for _ in range(_MAX_SC)])
        for pname in SCALAR_PARAMS
    ])

    # Array param scale inputs: [param_idx][scenario_idx], default 1.0
    array_scale_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.number(start=0.0, stop=10.0, step=0.01, value=1.0)
            for _ in range(_MAX_SC)
        ])
        for _ in ARRAY_PARAMS
    ])

    # Dynamically find all (i, j) positions where any subpop has non-zero E,
    # then create inputs for every (subpop, i, j) combination.
    from flu_example_utils import SUBPOP_CONFIG as _SPCONF
    _e_nonzero_ij = set()
    for _sp_cfg in _SPCONF:
        _E = _np.asarray(inputs["states"][_sp_cfg["name"]].E)
        for _ii, _jj in zip(*_np.nonzero(_E)):
            _e_nonzero_ij.add((int(_ii), int(_jj)))

    # All subpops × non-zero (i,j) positions, sorted for stable ordering.
    E_NONZERO_ENTRIES = [
        (_sp_cfg["name"], _ii, _jj)
        for _ii, _jj in sorted(_e_nonzero_ij)
        for _sp_cfg in _SPCONF
    ]

    # Flat array: one entry per (sp_name, i, j), each containing _MAX_SC inputs.
    e_init_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.number(
                start=0, stop=10000, step=1,
                value=float(_np.asarray(inputs["states"][_spn].E)[_ii, _jj]),
            )
            for _ in range(_MAX_SC)
        ])
        for _spn, _ii, _jj in E_NONZERO_ENTRIES
    ])

    # M scale inputs: [subpop_idx][scenario_idx], default 1.0
    m_scale_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.number(start=0.0, stop=5.0, step=0.01, value=1.0)
            for _ in range(_MAX_SC)
        ])
        for _ in _SPCONF
    ])

    # Per-scenario simulation start date
    start_date_sc_inputs = mo.ui.array([
        mo.ui.date(value="2024-08-14", label=f"Scenario {j + 1} start date")
        for j in range(_MAX_SC)
    ])

    return (
        ARRAY_BASELINES, ARRAY_PARAMS, SCALAR_PARAMS,
        E_NONZERO_ENTRIES,
        array_scale_inputs, e_init_inputs, m_scale_inputs, param_inputs,
        scenario_name_inputs, start_date_sc_inputs,
    )


@app.cell
def _build_settings(clt, settings_base, sim_mode, start_date_input):
    _transition_type = (
        "binom_deterministic_no_round"
        if sim_mode.value == "Deterministic"
        else "binom"
    )
    settings = clt.updated_dataclass(
        settings_base,
        {
            "transition_type": _transition_type,
            "transition_variables_to_save": ["ISH_to_HR", "ISH_to_HD", "S_to_E", "HD_to_D"],
            "start_real_date": start_date_input.value,
        },
    )
    return (settings,)


@app.cell
def _define_scenarios(
    ARRAY_PARAMS,
    E_NONZERO_ENTRIES,
    SCALAR_PARAMS,
    array_scale_inputs,
    baseline_vax,
    e_init_inputs,
    m_scale_inputs,
    inputs,
    np,
    num_scenarios_input,
    param_inputs,
    scale_inputs,
    scenario_name_inputs,
    scenario_tab,
    start_date_sc_inputs,
    vax_editors,
):
    import pandas as _pd_ds
    from flu_example_utils import SUBPOP_CONFIG as _SC, scale_vaccines_df

    _n = int(num_scenarios_input.value)
    _active_tab = scenario_tab.value

    def _editor_to_scale_mat(editor_val, subpop, baseline):
        df = editor_val if isinstance(editor_val, _pd_ds.DataFrame) else _pd_ds.DataFrame(editor_val)
        rows = df[df["subpopulation"] == subpop].reset_index(drop=True)
        n_age, n_risk = baseline.shape
        scale_mat = np.ones((n_age, n_risk))
        for _, row in rows.iterrows():
            _age = int(row["age_group"])
            for _risk in range(n_risk):
                _b = baseline[_age, _risk]
                scale_mat[_age, _risk] = float(row[f"risk_{_risk}"]) / _b if _b > 0 else 1.0
        return scale_mat

    scenarios = {"baseline": {}}
    scenario_labels = {"baseline": "Baseline"}

    if _active_tab == "Vaccination coverage":
        for _i in range(_n):
            _scale = scale_inputs[_i].value
            _sc_name = f"vax_sc{_i + 1}"
            _editor_val = vax_editors[_i].value
            scenarios[_sc_name] = {
                "subpop_schedules": {
                    sp["name"]: {
                        "daily_vaccines": scale_vaccines_df(
                            inputs["vaccines_df"][sp["name"]],
                            _editor_to_scale_mat(_editor_val, sp["name"], baseline_vax[sp["name"]]),
                            np,
                        )
                    }
                    for sp in _SC
                }
            }
            scenario_labels[_sc_name] = f"Scenario {_i + 1} (\u00d7{_scale:.2f} base)"

    else:  # "Parameter overrides"
        for j in range(_n):
            _sc_display = scenario_name_inputs[j].value.strip() or f"Scenario {j + 1}"
            _sc_name = f"param_sc{j + 1}"
            _scale = scale_inputs[j].value
            _sc_def = {
                "param_overrides": {
                    pname: param_inputs.value[i][j]
                    for i, pname in enumerate(SCALAR_PARAMS)
                },
                "array_param_scales": {
                    pname: array_scale_inputs.value[k][j]
                    for k, pname in enumerate(ARRAY_PARAMS)
                },
                "init_overrides": {
                    f"init:{_spn}:E:{_ii}:{_jj}": e_init_inputs.value[_idx][j]
                    for _idx, (_spn, _ii, _jj) in enumerate(E_NONZERO_ENTRIES)
                },
                "m_scales": {
                    sp["name"]: m_scale_inputs.value[_mi][j]
                    for _mi, sp in enumerate(_SC)
                },
                "start_date": start_date_sc_inputs[j].value,
            }
            _editor_val_j = vax_editors[j].value
            _sc_def["subpop_schedules"] = {
                sp["name"]: {
                    "daily_vaccines": scale_vaccines_df(
                        inputs["vaccines_df"][sp["name"]],
                        _editor_to_scale_mat(_editor_val_j, sp["name"], baseline_vax[sp["name"]]),
                        np,
                    )
                }
                for sp in _SC
            }
            scenarios[_sc_name] = _sc_def
            scenario_labels[_sc_name] = _sc_display

    return scenario_labels, scenarios


@app.cell
def _run_scenarios(
    clt,
    daily_sum_over_timesteps,
    flu,
    inputs,
    io,
    mo,
    np,
    num_reps_input,
    params,
    run_button,
    scenario_labels,
    scenarios,
    settings,
    sim_days_input,
    sim_mode,
):
    mo.stop(not run_button.value, mo.md("Press **Run scenario analysis** to start."))

    import sqlite3

    num_reps = 1 if sim_mode.value == "Deterministic" else num_reps_input.value
    end_day  = sim_days_input.value

    # Persistent database saved alongside this notebook.
    db_file = str(
        clt.utils.PROJECT_ROOT / "flu_instances" / "examples" / "scenario_results.db"
    )

    # ------------------------------------------------------------------
    # Database setup — two tables, one for state vars and one for tvars.
    # Arrays are stored as binary blobs (numpy .npy format) to keep the
    # row count small (one row per scenario/subpop/var/rep combination).
    # ------------------------------------------------------------------
    _STATE_VARS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    _TVAR_NAMES = ["ISH_to_HR", "ISH_to_HD", "S_to_E", "HD_to_D"]

    def _arr_to_blob(arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    con = sqlite3.connect(db_file)
    con.executescript("""
        DROP TABLE IF EXISTS state_vars;
        DROP TABLE IF EXISTS transition_vars;
        CREATE TABLE state_vars (
            scenario_name TEXT,
            subpop_name   TEXT,
            var_name      TEXT,
            rep           INT,
            data          BLOB,
            PRIMARY KEY (scenario_name, subpop_name, var_name, rep)
        );
        CREATE TABLE transition_vars (
            scenario_name TEXT,
            subpop_name   TEXT,
            var_name      TEXT,
            rep           INT,
            data          BLOB,
            PRIMARY KEY (scenario_name, subpop_name, var_name, rep)
        );
    """)
    con.commit()

    # ------------------------------------------------------------------
    # Helper: build one model for a given seed and scenario definition.
    # ------------------------------------------------------------------
    from flu_example_utils import (
        SUBPOP_CONFIG as _SC,
        make_rng_generators,
        build_flu_metapop_model,
        apply_general_init_overrides,
    )

    def _build_model(seed, scenario_def):
        # Per-scenario start date override
        _start_date = scenario_def.get("start_date")
        _settings = clt.updated_dataclass(settings, {"start_real_date": _start_date}) if _start_date else settings

        # Scalar param overrides
        _param_overrides = scenario_def.get("param_overrides", {})
        _params = clt.updated_dataclass(params, _param_overrides) if _param_overrides else params

        # Array param scaling: multiply every entry of each array by its scale factor
        _array_overrides = {
            pname: np.asarray(getattr(_params, pname)) * scale
            if not hasattr(getattr(_params, pname), "numpy")
            else getattr(_params, pname) * scale
            for pname, scale in scenario_def.get("array_param_scales", {}).items()
            if scale != 1.0
        }
        if _array_overrides:
            _params = clt.updated_dataclass(_params, _array_overrides)

        # Initial condition overrides (keys: "init:{sp}:E:{i}:{j}")
        _init_ovr = scenario_def.get("init_overrides", {})
        _states = apply_general_init_overrides(inputs["states"], _init_ovr, _SC, np) if _init_ovr else inputs["states"]

        # M scale overrides
        _m_scales = {
            sp_name: scale
            for sp_name, scale in scenario_def.get("m_scales", {}).items()
            if scale != 1.0
        }
        if _m_scales:
            _m_overrides = {f"M:{sp_name}": scale for sp_name, scale in _m_scales.items()}
            _states = apply_general_init_overrides(_states, _m_overrides, _SC, np)

        # Per-scenario vaccine schedules (fall back to base inputs if not overridden)
        _subpop_schedules = scenario_def.get("subpop_schedules", {})
        _vax_dfs = {
            sp["name"]: _subpop_schedules.get(sp["name"], {}).get(
                "daily_vaccines", inputs["vaccines_df"][sp["name"]]
            )
            for sp in _SC
        }

        _inputs_for_build = {**inputs, "states": _states}
        rngs = make_rng_generators(88888, _SC, np)
        model = build_flu_metapop_model(_SC, _inputs_for_build, _params, _settings, rngs, _vax_dfs, flu)
        model.modify_random_seed(seed)
        model.simulate_until_day(end_day)
        return model

    # ------------------------------------------------------------------
    # Helper: persist one replicate to the database.
    # ------------------------------------------------------------------
    def _save_rep(con, sc_name, rep_idx, model):
        sv_rows, tv_rows = [], []
        for sp_name, sp in model.subpop_models.items():
            n = sp.simulation_settings.timesteps_per_day
            # State variables: end-of-day snapshot → shape (days, A, R)
            for vn in _STATE_VARS:
                hist  = np.asarray(sp.compartments[vn].history_vals_list)
                daily = hist[::n]
                sv_rows.append((sc_name, sp_name, vn, rep_idx, _arr_to_blob(daily)))
            # Transition variables: daily totals → shape (days, A, R)
            for vn in _TVAR_NAMES:
                tvar = sp.transition_variables[vn]
                if not tvar.history_vals_list:
                    continue
                hist  = np.asarray(tvar.history_vals_list)
                daily = daily_sum_over_timesteps(hist, n)
                tv_rows.append((sc_name, sp_name, vn, rep_idx, _arr_to_blob(daily)))
        con.executemany(
            "INSERT OR REPLACE INTO state_vars VALUES (?,?,?,?,?)", sv_rows
        )
        con.executemany(
            "INSERT OR REPLACE INTO transition_vars VALUES (?,?,?,?,?)", tv_rows
        )

    # ------------------------------------------------------------------
    # Run all scenarios once; save state + transition histories to DB.
    # ------------------------------------------------------------------
    scenario_models = {}
    with mo.status.spinner("Running scenarios and saving all data..."):
        for _sc_name, _sc_def in scenarios.items():
            _reps = []
            for _seed in range(num_reps):
                _m = _build_model(_seed, _sc_def)
                _save_rep(con, _sc_name, _seed, _m)
                _reps.append(_m)
            con.commit()
            scenario_models[_sc_name] = _reps

    con.close()

    return db_file, scenario_models


@app.cell
def _compute_outcomes(
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    mo,
    np,
    scenario_models,
):
    scenario_names = list(scenario_models.keys())

    hosp = {
        name: [cumulative_hospitalizations(m) for m in models]
        for name, models in scenario_models.items()
    }
    deaths = {
        name: [cumulative_deaths(m) for m in models]
        for name, models in scenario_models.items()
    }

    # Vaccine-preventable hospitalizations/deaths vs baseline
    vph = {}
    vda = {}
    for _sc_name in scenario_names:
        if _sc_name == "baseline":
            continue
        vph[_sc_name] = [
            b - c for b, c in zip(hosp["baseline"], hosp[_sc_name])
        ]
        vda[_sc_name] = [
            b - c for b, c in zip(deaths["baseline"], deaths[_sc_name])
        ]

    # Age-stratified attack rates (mean across reps)
    num_age_groups = list(
        scenario_models["baseline"][0].subpop_models.values()
    )[0].params.num_age_groups
    age_ar = {
        sc: [
            np.mean([attack_rate(m, age_group=a) for m in models])
            for a in range(num_age_groups)
        ]
        for sc, models in scenario_models.items()
    }
    return age_ar, deaths, hosp, num_age_groups, vda, vph


# ---------------------------------------------------------------------------
# Plot 1 — Daily admissions / infections
# ---------------------------------------------------------------------------

@app.cell
def _daily_admissions_controls(mo):
    adm_metric_dd = mo.ui.dropdown(
        options={
            "Daily hospital admissions": "hosp",
            "Daily new infections":      "infections",
            "Daily deaths":              "deaths",
        },
        value="Daily hospital admissions",
        label="Metric",
    )
    from flu_example_utils import subpop_dropdown_options as _subpop_dd_opts, SUBPOP_CONFIG as _SC
    adm_subpop_dd = mo.ui.multiselect(
        options=_subpop_dd_opts(_SC, aggregate_label="all subpops"),
        value=["all subpops"],
        label="Subpopulation(s)",
    )
    adm_age_dd = mo.ui.multiselect(
        options=["all ages", "Age 0", "Age 1", "Age 2", "Age 3", "Age 4"],
        value=["all ages"],
        label="Age group(s)",
    )
    mo.vstack([
        mo.md("## Daily metric by scenario"),
        mo.hstack([adm_metric_dd, adm_subpop_dd, adm_age_dd]),
    ])
    return adm_age_dd, adm_metric_dd, adm_subpop_dd


@app.cell
def _plot_daily_admissions(
    adm_age_dd,
    adm_metric_dd,
    adm_subpop_dd,
    daily_deaths,
    daily_hospital_admissions,
    daily_new_infections,
    np,
    pd,
    plt,
    scenario_labels,
    scenario_models,
    settings,
):
    _sel_metric  = adm_metric_dd.value
    _sel_subpops = adm_subpop_dd.value or ["all subpops"]
    _sel_ages    = adm_age_dd.value or ["all ages"]

    _metric_fn = {
        "hosp":       daily_hospital_admissions,
        "infections": daily_new_infections,
        "deaths":     daily_deaths,
    }[_sel_metric]
    _y_label = {
        "hosp":       "Daily hospital admissions",
        "infections": "Daily new infections",
        "deaths":     "Daily deaths",
    }[_sel_metric]

    _combos   = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)
    _colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 4 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax        = _axes[_c_idx, 0]
        _subpop    = None if _sp == "all subpops" else _sp
        _age_group = None if _ag == "all ages" else int(_ag.split()[-1])
        _combo_label = f"{_sp} / {_ag}"

        for _i, (_sc_name, _model_list) in enumerate(scenario_models.items()):
            _color    = _colors[_i % len(_colors)]
            _all_vals = np.stack(
                [_metric_fn(m, subpop_name=_subpop, age_group=_age_group)
                 for m in _model_list],
                axis=0,
            )
            _dates  = pd.date_range(start=str(settings.start_real_date), periods=_all_vals.shape[1], freq='D')
            _median = np.median(_all_vals, axis=0)
            _lo     = np.percentile(_all_vals, 2.5,  axis=0)
            _hi     = np.percentile(_all_vals, 97.5, axis=0)
            _ax.plot(_dates, _median, label=scenario_labels.get(_sc_name, _sc_name), color=_color)
            _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.2)

        _ax.set_xlabel("Date")
        _ax.set_ylabel(_y_label)
        _ax.set_title(f"{_y_label} by scenario (median + 95% CI) — {_combo_label}")
        _ax.legend()

    _fig.autofmt_xdate()
    plt.tight_layout()
    _fig
    return


# ---------------------------------------------------------------------------
# Plot 1b — Compartment histories by scenario
# ---------------------------------------------------------------------------

@app.cell
def _comp_line_controls(mo):
    _ALL_COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    comp_checkboxes = mo.ui.array(
        [mo.ui.checkbox(value=(c in {"ISR", "IA"}), label=c)
         for c in _ALL_COMPARTMENTS]
    )
    mo.vstack([
        mo.md("## Compartment histories by scenario"),
        mo.md("*(subpopulation and age group dropdowns above also control this plot)*"),
        mo.md("### Compartments to show"),
        mo.hstack(comp_checkboxes.elements, justify="start"),
    ])
    return (comp_checkboxes,)


@app.cell
def _plot_comp_histories(
    adm_age_dd,
    adm_subpop_dd,
    comp_checkboxes,
    np,
    pd,
    plt,
    scenario_labels,
    scenario_models,
    settings,
):
    _ALL_COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    _LINE_STYLES = ["-", "--", ":", "-."]

    _selected    = [c for c, v in zip(_ALL_COMPARTMENTS, comp_checkboxes.value) if v] or _ALL_COMPARTMENTS
    _sel_subpops = adm_subpop_dd.value or ["all subpops"]
    _sel_ages    = adm_age_dd.value or ["all ages"]
    _combos      = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos    = len(_combos)
    _colors      = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 4 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax        = _axes[_c_idx, 0]
        _subpop    = None if _sp == "all subpops" else _sp
        _age_group = None if _ag == "all ages" else int(_ag.split()[-1])
        _combo_label = f"{_sp} / {_ag}"

        for _i, (_sc_name, _model_list) in enumerate(scenario_models.items()):
            _color = _colors[_i % len(_colors)]
            _n_tpd = list(_model_list[0].subpop_models.values())[0].simulation_settings.timesteps_per_day

            for _j, _comp in enumerate(_selected):
                _ls = _LINE_STYLES[_j % len(_LINE_STYLES)]
                _rep_arrays = []
                for _m in _model_list:
                    _subpops_list = (
                        [_m.subpop_models[_subpop]] if _subpop is not None
                        else list(_m.subpop_models.values())
                    )
                    _arrs  = [np.asarray(sp.compartments[_comp].history_vals_list)
                              for sp in _subpops_list]
                    _total = np.sum(np.stack(_arrs, axis=0), axis=0)  # (T, A, R)
                    _daily = _total[::_n_tpd]                          # (days, A, R)
                    if _age_group is not None:
                        _series = _daily[:, _age_group, :].sum(axis=-1)
                    else:
                        _series = _daily.sum(axis=(1, 2))
                    _rep_arrays.append(_series)

                _stacked = np.stack(_rep_arrays, axis=0)  # (reps, days)
                _dates   = pd.date_range(start=str(settings.start_real_date), periods=_stacked.shape[1], freq='D')
                _median  = np.median(_stacked, axis=0)
                _lo      = np.percentile(_stacked, 2.5,  axis=0)
                _hi      = np.percentile(_stacked, 97.5, axis=0)
                _label   = f"{scenario_labels.get(_sc_name, _sc_name)} — {_comp}"
                _ax.plot(_dates, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.1)

        _ax.set_xlabel("Date")
        _ax.set_ylabel("Count")
        _ax.set_title(f"Compartment histories by scenario (median + 95% CI) — {_combo_label}")
        _ax.legend(fontsize=7, loc="upper right")

    _fig.autofmt_xdate()
    plt.tight_layout()
    _fig
    return


# ---------------------------------------------------------------------------
# Plot 2 — Vaccine-preventable events (box plot)
# ---------------------------------------------------------------------------

@app.cell
def _vph_controls(mo):
    vph_metric_dd = mo.ui.dropdown(
        options={
            "Total hospitalizations": "hosp",
            "Total deaths":           "deaths",
            "Attack rate":            "ar",
        },
        value="Total hospitalizations",
        label="Metric",
    )
    from flu_example_utils import subpop_dropdown_options as _subpop_dd_opts, SUBPOP_CONFIG as _SC
    vph_subpop_dd = mo.ui.multiselect(
        options=_subpop_dd_opts(_SC, aggregate_label="all subpops"),
        value=["all subpops"],
        label="Subpopulation(s)",
    )
    vph_age_dd = mo.ui.multiselect(
        options=["all ages", "Age 0", "Age 1", "Age 2", "Age 3", "Age 4"],
        value=["all ages"],
        label="Age group(s)",
    )
    mo.vstack([
        mo.md("## Vaccine-preventable events vs. baseline"),
        mo.hstack([vph_metric_dd, vph_subpop_dd, vph_age_dd]),
    ])
    return vph_age_dd, vph_metric_dd, vph_subpop_dd


@app.cell
def _plot_vph(
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    mo,
    plt,
    scenario_labels,
    scenario_models,
    vph_age_dd,
    vph_metric_dd,
    vph_subpop_dd,
):
    _sel_metric  = vph_metric_dd.value
    _sel_subpops = vph_subpop_dd.value or ["all subpops"]
    _sel_ages    = vph_age_dd.value or ["all ages"]

    _metric_map = {
        "hosp":   (cumulative_hospitalizations, "Vaccine-preventable hospitalizations"),
        "deaths": (cumulative_deaths,           "Vaccine-preventable deaths"),
        "ar":     (attack_rate,                 "Vaccine-preventable attack rate"),
    }
    _metric_fn, _y_label = _metric_map[_sel_metric]

    _sc_names = [k for k in scenario_models if k != "baseline"]
    mo.stop(not _sc_names, mo.md("No counterfactual scenarios to compare."))

    _combos   = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(7, 5 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax        = _axes[_c_idx, 0]
        _subpop    = None if _sp == "all subpops" else _sp
        _age_group = None if _ag == "all ages" else int(_ag.split()[-1])
        _combo_label = f"{_sp} / {_ag}"

        _vp_data = {}
        for _sc_name in _sc_names:
            _base_vals    = [_metric_fn(m, subpop_name=_subpop, age_group=_age_group)
                             for m in scenario_models["baseline"]]
            _counter_vals = [_metric_fn(m, subpop_name=_subpop, age_group=_age_group)
                             for m in scenario_models[_sc_name]]
            _vp_data[_sc_name] = [b - c for b, c in zip(_base_vals, _counter_vals)]

        _labels = [scenario_labels.get(k, k) for k in _vp_data]
        _ax.boxplot(list(_vp_data.values()), labels=_labels)
        _ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
        _ax.set_ylabel(_y_label)
        _ax.set_title(f"{_y_label} vs. baseline — {_combo_label}")

    plt.tight_layout()
    _fig
    return


# ---------------------------------------------------------------------------
# Plot 2b — Actual metric values for all scenarios (including baseline)
# ---------------------------------------------------------------------------

@app.cell
def _plot_actual_metrics(
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    mo,
    plt,
    scenario_labels,
    scenario_models,
    vph_age_dd,
    vph_metric_dd,
    vph_subpop_dd,
):
    _sel_metric  = vph_metric_dd.value
    _sel_subpops = vph_subpop_dd.value or ["all subpops"]
    _sel_ages    = vph_age_dd.value or ["all ages"]

    _metric_map = {
        "hosp":   (cumulative_hospitalizations, "Total hospitalizations"),
        "deaths": (cumulative_deaths,           "Total deaths"),
        "ar":     (attack_rate,                 "Attack rate"),
    }
    _metric_fn, _y_label = _metric_map[_sel_metric]

    _combos   = [(sp, ag) for sp in _sel_subpops for ag in _sel_ages]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(7, 5 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax        = _axes[_c_idx, 0]
        _subpop    = None if _sp == "all subpops" else _sp
        _age_group = None if _ag == "all ages" else int(_ag.split()[-1])
        _combo_label = f"{_sp} / {_ag}"

        _data = {}
        for _sc_name, _models in scenario_models.items():
            _data[_sc_name] = [
                _metric_fn(m, subpop_name=_subpop, age_group=_age_group)
                for m in _models
            ]

        _labels = [scenario_labels.get(k, k) for k in _data]
        _ax.boxplot(list(_data.values()), labels=_labels)
        _ax.set_ylabel(_y_label)
        _ax.set_title(f"{_y_label} — all scenarios — {_combo_label}")

    mo.vstack([mo.md("## Actual metric values — all scenarios"), _fig])
    return


# ---------------------------------------------------------------------------
# Plot 3 — Age-stratified metric
# ---------------------------------------------------------------------------

@app.cell
def _age_ar_controls(mo):
    ar_metric_dd = mo.ui.dropdown(
        options={
            "Attack rate":            "ar",
            "Total hospitalizations": "hosp",
            "Total deaths":           "deaths",
        },
        value="Attack rate",
        label="Metric",
    )
    from flu_example_utils import subpop_dropdown_options as _subpop_dd_opts, SUBPOP_CONFIG as _SC
    ar_subpop_dd = mo.ui.multiselect(
        options=_subpop_dd_opts(_SC, aggregate_label="all subpops"),
        value=["all subpops"],
        label="Subpopulation(s)",
    )
    mo.vstack([
        mo.md("## Age-stratified metric by scenario"),
        mo.hstack([ar_metric_dd, ar_subpop_dd]),
    ])
    return ar_metric_dd, ar_subpop_dd


@app.cell
def _plot_age_attack_rates(
    ar_metric_dd,
    ar_subpop_dd,
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    np,
    num_age_groups,
    plt,
    scenario_labels,
    scenario_models,
):
    _sel_subpops = ar_subpop_dd.value or ["all subpops"]
    _sel_metric  = ar_metric_dd.value

    _metric_map = {
        "ar":     (attack_rate,                 "Mean attack rate"),
        "hosp":   (cumulative_hospitalizations, "Mean total hospitalizations"),
        "deaths": (cumulative_deaths,           "Mean total deaths"),
    }
    _metric_fn, _y_label = _metric_map[_sel_metric]

    _n_subpops = len(_sel_subpops)
    _n_bars    = num_age_groups + 1
    _x_labels  = [f"Age {a}" for a in range(num_age_groups)] + ["Total"]
    _x         = np.arange(_n_bars)
    _colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _fig, _axes = plt.subplots(_n_subpops, 1, figsize=(10, 4 * _n_subpops), squeeze=False)

    for _s_idx, _sp in enumerate(_sel_subpops):
        _ax     = _axes[_s_idx, 0]
        _subpop = None if _sp == "all subpops" else _sp

        # Per-age mean + a final «Total» (age_group=None) bar
        _age_vals = {}
        for _sc_name, _models in scenario_models.items():
            _per_age = [
                np.mean([_metric_fn(m, subpop_name=_subpop, age_group=a) for m in _models])
                for a in range(num_age_groups)
            ]
            _total = np.mean(
                [_metric_fn(m, subpop_name=_subpop, age_group=None) for m in _models]
            )
            _age_vals[_sc_name] = _per_age + [_total]

        _width = 0.8 / max(len(_age_vals), 1)
        for _i, (_sc_name, _vals) in enumerate(_age_vals.items()):
            _offset = (_i - len(_age_vals) / 2) * _width + _width / 2
            _ax.bar(
                _x + _offset, _vals, _width,
                label=scenario_labels.get(_sc_name, _sc_name),
                color=_colors[_i % len(_colors)],
                alpha=0.8,
            )

        # Subtle separator before the «Total» bar
        _ax.axvline(x=num_age_groups - 0.5, color="gray", linestyle=":", alpha=0.5)
        _ax.set_xlabel("Age group")
        _ax.set_ylabel(_y_label)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_x_labels)
        _ax.set_title(f"Age-stratified {_y_label.lower()} by scenario — {_sp}")
        _ax.legend()

    plt.tight_layout()
    _fig
    return


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

@app.cell
def _summary_selectors(mo, num_age_groups):
    from flu_example_utils import subpop_dropdown_options as _sdo, SUBPOP_CONFIG as _SC2
    summary_subpop_selector = mo.ui.multiselect(
        options=_sdo(_SC2, aggregate_label="combined"),
        value=["combined"],
        label="Subpopulation(s)",
    )
    summary_age_group_selector = mo.ui.multiselect(
        options=["all"] + [str(i) for i in range(num_age_groups)],
        value=["all"],
        label="Age group(s) ('all' = sum)",
    )
    mo.vstack([summary_subpop_selector, summary_age_group_selector])
    return summary_age_group_selector, summary_subpop_selector


@app.cell
def _summary_table(
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    daily_hospital_admissions,
    mo,
    np,
    pd,
    scenario_labels,
    scenario_models,
    summarize_outcomes,
    summary_age_group_selector,
    summary_subpop_selector,
    vda,
    vph,
):
    def fmt(vals):
        s = summarize_outcomes(vals)
        if len(vals) == 1:
            return f"{vals[0]:.1f}"
        return f"{s['mean']:.1f} [{s['lower_ci']:.1f}–{s['upper_ci']:.1f}]"

    _combos = [
        (sp, ag)
        for sp in (summary_subpop_selector.value or ["combined"])
        for ag in (summary_age_group_selector.value or ["all"])
    ]

    rows = []
    for _sp, _ag in _combos:
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)

        for _sc_name, _sc_label in scenario_labels.items():
            _models = scenario_models[_sc_name]

            _hosp_vals  = [cumulative_hospitalizations(m, _subpop, _age_group) for m in _models]
            _death_vals = [cumulative_deaths(m, _subpop, _age_group) for m in _models]
            _ar_vals    = [attack_rate(m, _subpop, _age_group) for m in _models]

            _daily_adm = np.stack(
                [daily_hospital_admissions(m, _subpop, _age_group) for m in _models], axis=0
            )
            _peak_vals         = _daily_adm.max(axis=1).tolist()
            _days_to_peak_vals = np.argmax(_daily_adm, axis=1).tolist()

            _row = {
                "Subpopulation":             _sp,
                "Age group":                 _ag,
                "Scenario":                  _sc_label,
                "Attack rate (mean)":        f"{np.mean(_ar_vals):.3f}",
                "Hosp. (mean [95% CI])":     fmt(_hosp_vals),
                "Deaths (mean [95% CI])":    fmt(_death_vals),
                "Peak daily admissions":     fmt(_peak_vals),
                "Days to peak admissions":   fmt(_days_to_peak_vals),
            }
            if _sc_name in vph:
                _row["VPH (mean [95% CI])"]     = fmt(vph[_sc_name])
                _row["Deaths averted [95% CI]"] = fmt(vda[_sc_name])
            else:
                _row["VPH (mean [95% CI])"]     = "—"
                _row["Deaths averted [95% CI]"] = "—"
            rows.append(_row)
    mo.vstack([mo.md("### Summary"), mo.ui.table(pd.DataFrame(rows))])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
