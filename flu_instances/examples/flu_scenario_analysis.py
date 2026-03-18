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
    run_button = mo.ui.run_button(label="Run scenario analysis")
    return num_reps_input, run_button, sim_days_input, sim_mode


@app.cell
def _scenario_controls(mo):
    num_scenarios_input = mo.ui.number(
        start=1, stop=5, step=1, value=2,
        label="Number of counterfactual scenarios",
    )
    # Up to five scale-factor inputs wrapped in mo.ui.array so that marimo
    # tracks value changes on individual elements and re-runs dependent cells.
    scale_inputs = mo.ui.array([
        mo.ui.number(start=0.1, stop=5.0, step=0.05,
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
    ARRAY_BASELINES,
    ARRAY_PARAMS,
    SCALAR_PARAMS,
    array_scale_inputs,
    e_init_inputs,
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
        num_scenarios_input,
        mo.hstack(scale_inputs[:_n]),
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
            [mo.md(f"`{_lbl}`")] + [e_init_inputs[_ei][j] for j in range(_n)],
            widths="equal",
        )
        for _ei, _lbl in enumerate(["east_E[2,0]", "west_E[2,0]"])
    ]
    _tab2 = mo.vstack(
        [mo.md("### Parameter overrides"), num_scenarios_input,
         _header_row, _vax_row,
         mo.md("**Scalar parameters**")] + _scalar_rows +
        [mo.md("**Array parameters** *(scale factor applied to all entries)*")] + _array_rows +
        [mo.md("**Initial conditions**")] + _e_rows
    )

    _content = _tab1 if scenario_tab.value == "Vaccination coverage" else _tab2

    mo.vstack([
        mo.md("## Scenario analysis settings"),
        mo.hstack([sim_mode, num_reps_input, sim_days_input]),
        _reps_note,
        scenario_tab,
        _content,
        run_button,
    ])
    return


@app.cell
def _load_files(clt, flu, pd):
    base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "austin_input_files_2024_2025"

    east_state = clt.make_dataclass_from_json(
        base_path / "init_vals_east.json", flu.FluSubpopState
    )
    west_state = clt.make_dataclass_from_json(
        base_path / "init_vals_west.json", flu.FluSubpopState
    )
    params = clt.make_dataclass_from_json(
        base_path / "common_subpop_params.json", flu.FluSubpopParams
    )
    mixing_params = clt.make_dataclass_from_json(
        base_path / "mixing_params.json", flu.FluMixingParams
    )
    settings_base = clt.make_dataclass_from_json(
        base_path / "simulation_settings.json", flu.SimulationSettings
    )

    east_vax_df   = pd.read_csv(base_path / "daily_vaccines_East.csv", index_col=0)
    west_vax_df   = pd.read_csv(base_path / "daily_vaccines_West.csv", index_col=0)
    east_cal_df   = pd.read_csv(base_path / "school_work_calendar_austin_East.csv", index_col=0)
    west_cal_df   = pd.read_csv(base_path / "school_work_calendar_austin_West.csv", index_col=0)
    humidity_df   = pd.read_csv(base_path / "absolute_humidity_austin.csv", index_col=0)
    mobility_df   = pd.read_csv(base_path / "mobility_modifier.csv", index_col=0)
    return (
        east_cal_df,
        east_state,
        east_vax_df,
        humidity_df,
        mixing_params,
        mobility_df,
        params,
        settings_base,
        west_cal_df,
        west_state,
        west_vax_df,
    )


@app.cell
def _param_tab_controls(east_state, flu, mo, params, west_state):
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
        return "[" + ", ".join(f"{x:.4g}" for x in _np.asarray(val).flatten()) + "]"
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

    # Initial E(2,0) inputs: [east=0, west=1][scenario_idx]
    _east_e_base = float(east_state.E[2][0])
    _west_e_base = float(west_state.E[2][0])
    e_init_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.number(start=0, stop=10000, step=1, value=_east_e_base)
            for _ in range(_MAX_SC)
        ]),
        mo.ui.array([
            mo.ui.number(start=0, stop=10000, step=1, value=_west_e_base)
            for _ in range(_MAX_SC)
        ]),
    ])

    return (
        ARRAY_BASELINES, ARRAY_PARAMS, SCALAR_PARAMS,
        array_scale_inputs, e_init_inputs, param_inputs, scenario_name_inputs,
    )


@app.cell
def _build_settings(clt, settings_base, sim_mode):
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
        },
    )
    return (settings,)


@app.cell
def _define_scenarios(
    ARRAY_PARAMS,
    SCALAR_PARAMS,
    array_scale_inputs,
    e_init_inputs,
    east_vax_df,
    np,
    num_scenarios_input,
    param_inputs,
    pd,
    scale_inputs,
    scenario_name_inputs,
    scenario_tab,
    west_vax_df,
):
    import json

    _n = int(num_scenarios_input.value)
    _active_tab = scenario_tab.value

    # daily_vaccines values are JSON-encoded 2D arrays (not plain floats),
    # so we must round-trip through JSON to scale them correctly.
    def scale_vax(df, factor):
        scaled = df.copy()
        scaled["daily_vaccines"] = scaled["daily_vaccines"].apply(
            lambda s: json.dumps((np.array(json.loads(s)) * factor).tolist())
        )
        scaled["date"] = pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.date
        return scaled

    scenarios = {"baseline": {}}
    scenario_labels = {"baseline": "Baseline"}

    if _active_tab == "Vaccination coverage":
        for _scale in [inp.value for inp in scale_inputs[:_n]]:
            _sc_name = f"vax_x{_scale:.2f}"
            scenarios[_sc_name] = {
                "subpop_schedules": {
                    "east": {"daily_vaccines": scale_vax(east_vax_df, _scale)},
                    "west": {"daily_vaccines": scale_vax(west_vax_df, _scale)},
                }
            }
            scenario_labels[_sc_name] = f"\u00d7{_scale:.2f} coverage"

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
                    "east_E_2_0": e_init_inputs.value[0][j],
                    "west_E_2_0": e_init_inputs.value[1][j],
                },
            }
            if _scale != 1.0:
                _sc_def["subpop_schedules"] = {
                    "east": {"daily_vaccines": scale_vax(east_vax_df, _scale)},
                    "west": {"daily_vaccines": scale_vax(west_vax_df, _scale)},
                }
            scenarios[_sc_name] = _sc_def
            scenario_labels[_sc_name] = _sc_display

    return scenario_labels, scenarios


@app.cell
def _run_scenarios(
    clt,
    daily_sum_over_timesteps,
    east_cal_df,
    east_state,
    east_vax_df,
    flu,
    humidity_df,
    io,
    mixing_params,
    mo,
    mobility_df,
    np,
    num_reps_input,
    params,
    run_button,
    scenario_labels,
    scenarios,
    settings,
    sim_days_input,
    sim_mode,
    west_cal_df,
    west_state,
    west_vax_df,
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
    def _build_model(seed, scenario_def):
        import copy as _copy

        bit_gen = np.random.MT19937(88888)
        jumped  = np.random.MT19937(88888).jumped(1)

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

        # Initial condition overrides for E(2,0)
        _init_ovr = scenario_def.get("init_overrides", {})
        if _init_ovr:
            _east_state = _copy.deepcopy(east_state)
            _west_state = _copy.deepcopy(west_state)
            _east_state.E[2][0] = _init_ovr.get("east_E_2_0", east_state.E[2][0])
            _west_state.E[2][0] = _init_ovr.get("west_E_2_0", west_state.E[2][0])
        else:
            _east_state = east_state
            _west_state = west_state

        east = flu.FluSubpopModel(
            _east_state, _params, settings,
            np.random.Generator(bit_gen),
            flu.FluSubpopSchedules(
                absolute_humidity=humidity_df,
                flu_contact_matrix=east_cal_df,
                daily_vaccines=east_vax_df,
                mobility_modifier=mobility_df,
            ),
            name="east",
        )
        west = flu.FluSubpopModel(
            _west_state, _params, settings,
            np.random.Generator(jumped),
            flu.FluSubpopSchedules(
                absolute_humidity=humidity_df,
                flu_contact_matrix=west_cal_df,
                daily_vaccines=west_vax_df,
                mobility_modifier=mobility_df,
            ),
            name="west",
        )
        model = flu.FluMetapopModel([east, west], mixing_params)
        for _sp, _sched_map in scenario_def.get("subpop_schedules", {}).items():
            for _sname, _new_df in _sched_map.items():
                model.replace_schedule(_sname, _new_df, subpop_name=_sp)
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
        },
        value="Daily hospital admissions",
        label="Metric",
    )
    adm_subpop_dd = mo.ui.multiselect(
        options=["all subpops", "east", "west"],
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
    daily_hospital_admissions,
    daily_new_infections,
    np,
    plt,
    scenario_labels,
    scenario_models,
):
    _sel_metric  = adm_metric_dd.value
    _sel_subpops = adm_subpop_dd.value or ["all subpops"]
    _sel_ages    = adm_age_dd.value or ["all ages"]

    _metric_fn = (
        daily_hospital_admissions if _sel_metric == "hosp" else daily_new_infections
    )
    _y_label = (
        "Daily hospital admissions" if _sel_metric == "hosp" else "Daily new infections"
    )

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
            _days   = np.arange(_all_vals.shape[1])
            _median = np.median(_all_vals, axis=0)
            _lo     = np.percentile(_all_vals, 2.5,  axis=0)
            _hi     = np.percentile(_all_vals, 97.5, axis=0)
            _ax.plot(_days, _median, label=scenario_labels.get(_sc_name, _sc_name), color=_color)
            _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)

        _ax.set_xlabel("Day")
        _ax.set_ylabel(_y_label)
        _ax.set_title(f"{_y_label} by scenario (median + 95% CI) — {_combo_label}")
        _ax.legend()

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
    plt,
    scenario_labels,
    scenario_models,
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
                _days    = np.arange(_stacked.shape[1])
                _median  = np.median(_stacked, axis=0)
                _lo      = np.percentile(_stacked, 2.5,  axis=0)
                _hi      = np.percentile(_stacked, 97.5, axis=0)
                _label   = f"{scenario_labels.get(_sc_name, _sc_name)} — {_comp}"
                _ax.plot(_days, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.1)

        _ax.set_xlabel("Day")
        _ax.set_ylabel("Count")
        _ax.set_title(f"Compartment histories by scenario (median + 95% CI) — {_combo_label}")
        _ax.legend(fontsize=7, loc="upper right")

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
    vph_subpop_dd = mo.ui.multiselect(
        options=["all subpops", "east", "west"],
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
    ar_subpop_dd = mo.ui.multiselect(
        options=["all subpops", "east", "west"],
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
def _summary_table(deaths, hosp, mo, pd, scenario_labels, summarize_outcomes, vda, vph):
    def ci_str(vals):
        s = summarize_outcomes(vals)
        return f"{s['mean']:.1f} [{s['lower_ci']:.1f}–{s['upper_ci']:.1f}]"

    rows = []
    for _sc_name, _sc_label in scenario_labels.items():
        _row = {
            "Scenario":              _sc_label,
            "Hosp. (mean [95% CI])": ci_str(hosp[_sc_name]),
            "Deaths (mean [95% CI])":ci_str(deaths[_sc_name]),
        }
        if _sc_name in vph:
            _row["VPH (mean [95% CI])"]    = ci_str(vph[_sc_name])
            _row["Deaths averted [95% CI]"] = ci_str(vda[_sc_name])
        else:
            _row["VPH (mean [95% CI])"]    = "—"
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
