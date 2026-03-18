"""
flu_sensitivity.py
============================

Interactive marimo notebook for exploring flu model parameters and visualizing
simulation output.  Run with::

    marimo run flu_sensitivity.py
    marimo run flu_instances/examples/flu_sensitivity.py

or open in edit mode::

    marimo edit flu_sensitivity.py

Model setup reuses the Austin 2024-2025 two-population configuration
(east + west subpopulations) so results are directly comparable to the
existing ``flu_demo_2popAustin_2024_2025.ipynb`` notebook.
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import os
    import sys
    os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import marimo as mo
    import clt_toolkit as clt
    import flu_core as flu
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
        flu,
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
    else:
        mo.md(f"`Flu Sensitivity Analysis`")
    return


@app.cell
def _load_files(clt, flu, pd):
    from pathlib import Path

    base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "austin_input_files_2024_2025"

    east_state = clt.make_dataclass_from_json(
        base_path / "init_vals_east.json", flu.FluSubpopState
    )
    west_state = clt.make_dataclass_from_json(
        base_path / "init_vals_west.json", flu.FluSubpopState
    )

    params_baseline = clt.make_dataclass_from_json(
        base_path / "common_subpop_params.json", flu.FluSubpopParams
    )
    mixing_params = clt.make_dataclass_from_json(
        base_path / "mixing_params.json", flu.FluMixingParams
    )

    east_vaccines_df = pd.read_csv(base_path / "daily_vaccines_East.csv", index_col=0)
    west_vaccines_df = pd.read_csv(base_path / "daily_vaccines_West.csv", index_col=0)
    east_calendar_df = pd.read_csv(base_path / "school_work_calendar_austin_East.csv", index_col=0)
    west_calendar_df = pd.read_csv(base_path / "school_work_calendar_austin_West.csv", index_col=0)
    humidity_df = pd.read_csv(base_path / "absolute_humidity_austin.csv", index_col=0)
    mobility_df = pd.read_csv(base_path / "mobility_modifier.csv", index_col=0)
    return (
        east_calendar_df,
        east_state,
        east_vaccines_df,
        humidity_df,
        mixing_params,
        mobility_df,
        params_baseline,
        west_calendar_df,
        west_state,
        west_vaccines_df,
    )


@app.cell
def _param_catalog(flu, params_baseline):
    import dataclasses as _dc
    import numpy as _np

    _SKIP = {
        "num_age_groups", "num_risk_groups", "start_real_date",
        "vax_immunity_reset_date_mm_dd", "vax_protection_delay_days",
        "total_pop_age_risk",
    }
    _CONTACT_MATRICES = {"total_contact_matrix", "school_contact_matrix", "work_contact_matrix"}

    # Array-valued params (excludes contact matrices, strings, None, and scalars)
    ARRAY_PARAMS = [
        f.name for f in _dc.fields(flu.FluSubpopParams)
        if f.name not in (_SKIP | _CONTACT_MATRICES)
        and not isinstance(getattr(params_baseline, f.name), (int, float, str))
        and getattr(params_baseline, f.name) is not None
    ]

    def _fmt(val):
        return "[" + ", ".join(f"{x:.4g}" for x in _np.asarray(val).flatten()) + "]"

    ARRAY_BASELINES = [_fmt(getattr(params_baseline, p)) for p in ARRAY_PARAMS]

    # Map parameter name → (label, default_value, min, max, step)
    # Only scalar / non-matrix parameters are included.
    SCALAR_PARAMS = {
        "beta_baseline":               ("β baseline (transmission)", 0.042, 0.001, 0.1,   0.001),
        "humidity_impact":             ("Humidity impact",           1.0,   0.0,   5.0,   0.1),
        "inf_induced_saturation":      ("Inf. induced saturation",   1.0,   0.0,   5.0,   0.1),
        "inf_induced_immune_wane":     ("Inf. immunity wane rate",   0.01,  0.0,   0.1,   0.001),
        "vax_induced_saturation":      ("Vax. induced saturation",   1.0,   0.0,   5.0,   0.1),
        "vax_induced_immune_wane":     ("Vax. immunity wane rate",   0.01,  0.0,   0.1,   0.001),
        "inf_induced_inf_risk_reduce": ("Inf. → inf. risk reduction",  0.5, 0.0,   1.0,   0.01),
        "inf_induced_hosp_risk_reduce":("Inf. → hosp. risk reduction", 0.5, 0.0,   1.0,   0.01),
        "vax_induced_inf_risk_reduce": ("Vax. → inf. risk reduction",  0.5, 0.0,   1.0,   0.01),
        "vax_induced_hosp_risk_reduce":("Vax. → hosp. risk reduction", 0.5, 0.0,   1.0,   0.01),
        "vax_induced_death_risk_reduce":("Vax. → death risk reduction",0.5, 0.0,   1.0,   0.01),
        "E_to_I_rate":                 ("E → I rate",                0.33,  0.1,   1.0,   0.01),
        "IP_to_IS_rate":               ("IP → IS rate",              0.5,   0.1,   1.0,   0.01),
        "ISR_to_R_rate":               ("ISR → R rate",              0.2,   0.05,  1.0,   0.01),
        "ISH_to_H_rate":               ("ISH → H rate",              0.2,   0.05,  1.0,   0.01),
        "HR_to_R_rate":                ("HR → R rate",               0.1,   0.01,  0.5,   0.01),
        "HD_to_D_rate":                ("HD → D rate",               0.1,   0.01,  0.5,   0.01),
        "IP_relative_inf":             ("IP relative infectiousness", 1.0,  0.0,   2.0,   0.1),
        "IA_relative_inf":             ("IA relative infectiousness", 0.5,  0.0,   2.0,   0.1),
    }
    return (SCALAR_PARAMS, ARRAY_PARAMS, ARRAY_BASELINES)


@app.cell
def _controls(SCALAR_PARAMS, mo, params_baseline):
    # Simulation mode
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    num_reps_input = mo.ui.number(
        start=1, stop=50, step=1, value=1, label="Replicates (stochastic)"
    )

    # Parameter type selector
    param_type = mo.ui.radio(
        options=["Scalar", "Array (scale factor)"],
        value="Scalar",
        label="Parameter type",
    )

    # Parameter selector (scalar)
    param_selector = mo.ui.dropdown(
        options=list(SCALAR_PARAMS.keys()),
        value="beta_baseline",
        label="Scalar parameter to vary",
    )

    # Number of parameter values to compare
    num_values_input = mo.ui.number(
        start=1, stop=6, step=1, value=3, label="Values to compare"
    )

    # Vaccine coverage multiplier
    vax_multiplier = mo.ui.slider(
        start=0.5, stop=3.0, step=0.05, value=1.0,
        label="Vaccine coverage multiplier",
    )

    # Subpopulation multi-selector
    subpop_selector = mo.ui.multiselect(
        options=["combined", "east", "west"],
        value=["combined"],
        label="Subpopulation(s)",
    )

    # Age group multi-selector
    age_group_selector = mo.ui.multiselect(
        options=["all"] + [str(i) for i in range(int(params_baseline.num_age_groups))],
        value=["all"],
        label="Age group(s) ('all' = sum)",
    )

    # Simulation length
    sim_length = mo.ui.number(
        start=50, stop=365, step=10, value=200, label="Simulation days"
    )

    return (
        age_group_selector,
        num_reps_input,
        num_values_input,
        param_selector,
        param_type,
        sim_length,
        sim_mode,
        subpop_selector,
        vax_multiplier,
    )


@app.cell
def _controls_form(
    age_group_selector,
    mo,
    num_reps_input,
    num_values_input,
    param_selector,
    param_type,
    sim_length,
    sim_mode,
    subpop_selector,
    vax_multiplier,
):
    _scalar_row = (
        mo.hstack([param_selector, num_values_input, vax_multiplier])
        if param_type.value == "Scalar"
        else mo.hstack([num_values_input, vax_multiplier])
    )
    controls_form = mo.vstack([
        mo.md("## Controls"),
        mo.hstack([sim_mode, num_reps_input, param_type]),
        _scalar_row,
        mo.hstack([subpop_selector, age_group_selector, sim_length]),
    ])
    return (controls_form,)


@app.cell
def _param_sliders(SCALAR_PARAMS, mo, num_values_input, param_selector):
    _label, _default_val, _lo, _hi, _step = SCALAR_PARAMS[param_selector.value]
    _n = num_values_input.value

    sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=_lo, stop=_hi, step=_step, value=_default_val,
                label=f"{param_selector.value} value {i + 1}",
            )
            for i in range(_n)
        ]
    )

    sliders_ui = mo.vstack([
        mo.md(f"### Parameter: {_label}"),
        mo.md("Set values to overlay (identical values collapse to one curve):"),
        sliders,
    ])
    return sliders, sliders_ui


@app.cell
def _array_param_selector(ARRAY_PARAMS, mo):
    array_param_selector = mo.ui.dropdown(
        options=ARRAY_PARAMS,
        value=ARRAY_PARAMS[0] if ARRAY_PARAMS else None,
        label="Array parameter to scale",
    )
    return (array_param_selector,)


@app.cell
def _array_param_sliders(
    ARRAY_BASELINES, ARRAY_PARAMS, array_param_selector, mo, num_values_input
):
    _n = num_values_input.value

    array_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.1, stop=3.0, step=0.05, value=1.0,
                label=f"scale factor {i + 1}",
            )
            for i in range(_n)
        ]
    )

    _base_str = ""
    if ARRAY_PARAMS and array_param_selector.value in ARRAY_PARAMS:
        _base_str = ARRAY_BASELINES[ARRAY_PARAMS.index(array_param_selector.value)]

    array_sliders_ui = mo.vstack([
        mo.md(f"### Array parameter: `{array_param_selector.value}`"),
        mo.md(f"Baseline values: {_base_str}"),
        array_param_selector,
        mo.md("Each entry in the array is multiplied by the chosen scale factor:"),
        array_sliders,
    ])
    return array_sliders, array_sliders_ui


@app.cell
def _run_simulation(
    array_param_selector,
    array_sliders,
    clt,
    east_calendar_df,
    east_state,
    east_vaccines_df,
    flu,
    humidity_df,
    mixing_params,
    mo,
    mobility_df,
    np,
    num_reps_input,
    param_selector,
    param_type,
    params_baseline,
    pd,
    sim_length,
    sim_mode,
    sliders,
    vax_multiplier,
    west_calendar_df,
    west_state,
    west_vaccines_df,
):
    transition_type = (
        "binom_deterministic_no_round"
        if sim_mode.value == "Deterministic"
        else "binom"
    )
    num_reps = num_reps_input.value if sim_mode.value == "Stochastic" else 1
    end_day = sim_length.value

    # Scale vaccine schedules by the multiplier.
    # daily_vaccines values are JSON-encoded 2D arrays (not plain floats),
    # so we must round-trip through JSON to scale them correctly.
    import json

    def scale_vaccines(df, scale):
        scaled = df.copy()
        scaled["daily_vaccines"] = scaled["daily_vaccines"].apply(
            lambda s: json.dumps((np.array(json.loads(s)) * scale).tolist())
        )
        scaled["date"] = pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.date
        return scaled

    vax_scale = vax_multiplier.value
    east_vax = scale_vaccines(east_vaccines_df, vax_scale)
    west_vax = scale_vaccines(west_vaccines_df, vax_scale)

    # Parameter values/scale factors to overlay (deduplicated, preserving order)
    _is_array_mode = param_type.value == "Array (scale factor)"

    if _is_array_mode:
        _array_pname = array_param_selector.value
        param_name = f"{_array_pname} ×scale"
        param_values = list(dict.fromkeys(array_sliders.value))

        def _make_params(val):
            _base = np.asarray(getattr(params_baseline, _array_pname))
            return clt.updated_dataclass(
                params_baseline, {_array_pname: _base * val}
            )
    else:
        param_name = param_selector.value
        param_values = list(dict.fromkeys(sliders.value))

        def _make_params(val):
            return clt.updated_dataclass(params_baseline, {param_name: val})

    tvar_to_save = ["ISH_to_HR", "ISH_to_HD", "S_to_E", "HD_to_D"]

    def build_and_run(param_val):
        updated_params = _make_params(param_val)

        east_schedules = flu.FluSubpopSchedules(
            absolute_humidity=humidity_df,
            flu_contact_matrix=east_calendar_df,
            daily_vaccines=east_vax,
            mobility_modifier=mobility_df,
        )
        west_schedules = flu.FluSubpopSchedules(
            absolute_humidity=humidity_df,
            flu_contact_matrix=west_calendar_df,
            daily_vaccines=west_vax,
            mobility_modifier=mobility_df,
        )

        rng_results = []
        for rep in range(num_reps):
            bit_gen = np.random.MT19937(88888 + rep)
            jumped = bit_gen.jumped(1)

            east_model = flu.FluSubpopModel(
                east_state, updated_params,
                clt.updated_dataclass(
                    clt.make_dataclass_from_json(
                        clt.utils.PROJECT_ROOT / "flu_instances"
                        / "austin_input_files_2024_2025"
                        / "simulation_settings.json",
                        flu.SimulationSettings,
                    ),
                    {
                        "transition_type": transition_type,
                        "transition_variables_to_save": tvar_to_save,
                    },
                ),
                np.random.Generator(bit_gen),
                east_schedules,
                name="east",
            )
            west_model = flu.FluSubpopModel(
                west_state, updated_params,
                clt.updated_dataclass(
                    east_model.simulation_settings,
                    {},
                ),
                np.random.Generator(jumped),
                west_schedules,
                name="west",
            )
            model = flu.FluMetapopModel([east_model, west_model], mixing_params)
            model.simulate_until_day(end_day)
            rng_results.append(model)

        return rng_results

    with mo.status.spinner("Running simulation..."):
        results = {str(v): build_and_run(v) for v in param_values}
    return num_reps, param_name, results


@app.cell
def _show_controls(array_sliders_ui, controls_form, mo, param_type, sliders_ui):
    _param_controls = array_sliders_ui if param_type.value == "Array (scale factor)" else sliders_ui
    mo.vstack([controls_form, _param_controls])
    return


@app.cell
def _compartment_and_metric_controls(mo):
    _all_compartments = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    compartment_checkboxes = mo.ui.array(
        [mo.ui.checkbox(value=True, label=c) for c in _all_compartments],
        label="Compartments to show",
    )
    epi_metric_checkboxes = mo.ui.array(
        [mo.ui.checkbox(value=True, label=m) for m in ["M", "MV"]],
        label="Epi metrics to show",
    )
    return compartment_checkboxes, epi_metric_checkboxes


@app.cell
def _plot_compartments(
    age_group_selector,
    compartment_checkboxes,
    mo,
    param_name,
    plt,
    results,
    subpop_selector,
):
    from flu_core.flu_outcomes import plot_compartment_history

    _ALL_COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    _LINE_STYLES = ["-", "--", ":", "-."]

    _selected = [c for c, v in zip(_ALL_COMPARTMENTS, compartment_checkboxes.value) if v] or _ALL_COMPARTMENTS

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 5 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp} / age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            plot_compartment_history(
                _model_list[0],
                compartment_names=_selected,
                ax=_ax,
                subpop_name=_subpop,
                age_group=_age_group,
                title=f"Compartment histories — {_combo_label}",
                linestyle=_ls,
                label_suffix=f" [{param_name}={_scenario_label}]",
            )
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels, fontsize=7, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

    _fig.tight_layout()
    _fig.subplots_adjust(right=0.92)
    # mo.hstack([_fig, mo.vstack([mo.md("### Compartments to show"), compartment_checkboxes])])
    mo.vstack([mo.md("### Compartments to show"), mo.hstack(compartment_checkboxes.elements, justify="start"), _fig])
    return


@app.cell
def _plot_epi_metrics(
    age_group_selector,
    epi_metric_checkboxes,
    mo,
    param_name,
    plt,
    results,
    subpop_selector,
):
    from flu_core.flu_outcomes import plot_epi_metrics

    _ALL_METRICS = ["M", "MV"]
    _LINE_STYLES = ["-", "--", ":", "-."]
    _selected_metrics = [m for m, v in zip(_ALL_METRICS, epi_metric_checkboxes.value) if v] or _ALL_METRICS

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 4 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp} / age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            plot_epi_metrics(
                _model_list[0],
                metric_names=_selected_metrics,
                ax=_ax,
                subpop_name=_subpop,
                age_group=_age_group,
                title=f"{', '.join(_selected_metrics)} — {_combo_label}",
                linestyle=_ls,
                label_suffix=f" [{param_name}={_scenario_label}]",
            )
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

    _fig.tight_layout()
    _fig.subplots_adjust(right=0.92)
    mo.vstack([mo.md("### Epi metrics to show"), mo.hstack(epi_metric_checkboxes.elements, justify="start"), _fig])
    
    return


@app.cell
def _plot_infections(
    age_group_selector,
    daily_new_infections,
    np,
    num_reps,
    param_name,
    plt,
    results,
    subpop_selector,
):
    _LINE_STYLES = ["-", "--", ":", "-."]

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _fig, _ax = plt.subplots(figsize=(10, 4))

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _color = _colors[_c_idx % len(_colors)]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp}/age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{_combo_label} | {param_name}={_scenario_label}"

            if num_reps > 1:
                _all_vals = np.stack(
                    [daily_new_infections(m, _subpop, _age_group) for m in _model_list], axis=0
                )
                _days = np.arange(_all_vals.shape[1])
                _median = np.median(_all_vals, axis=0)
                _lo = np.percentile(_all_vals, 2.5, axis=0)
                _hi = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_days, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
            else:
                _vals = daily_new_infections(_model_list[0], _subpop, _age_group)
                _ax.plot(_vals, label=_label, color=_color, linestyle=_ls)

    _ax.set_xlabel("Day")
    _ax.set_ylabel("Daily new infections")
    _ax.set_title(f"Daily new infections — {param_name}")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell
def _plot_admissions(
    age_group_selector,
    daily_hospital_admissions,
    np,
    num_reps,
    param_name,
    plt,
    results,
    subpop_selector,
):
    _LINE_STYLES = ["-", "--", ":", "-."]

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _fig, _ax = plt.subplots(figsize=(10, 4))

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _color = _colors[_c_idx % len(_colors)]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp}/age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{_combo_label} | {param_name}={_scenario_label}"

            if num_reps > 1:
                _all_vals = np.stack(
                    [daily_hospital_admissions(m, _subpop, _age_group) for m in _model_list], axis=0
                )
                _days = np.arange(_all_vals.shape[1])
                _median = np.median(_all_vals, axis=0)
                _lo = np.percentile(_all_vals, 2.5, axis=0)
                _hi = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_days, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
            else:
                _vals = daily_hospital_admissions(_model_list[0], _subpop, _age_group)
                _ax.plot(_vals, label=_label, color=_color, linestyle=_ls)

    _ax.set_xlabel("Day")
    _ax.set_ylabel("Daily hospital admissions")
    _ax.set_title(f"Daily hospital admissions — {param_name}")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell
def _plot_cumulative_tvars(
    age_group_selector,
    daily_hospital_admissions,
    daily_new_infections,
    np,
    num_reps,
    param_name,
    plt,
    results,
    subpop_selector,
):
    _LINE_STYLES = ["-", "--", ":", "-."]

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Three subplots per combo row: infections, hospitalizations, deaths
    _n_combos = len(_combos)
    _fig, _axes = plt.subplots(_n_combos, 3, figsize=(15, 4 * _n_combos), squeeze=False)

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _subpop    = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp} / age {_ag}"

        _ax_inf  = _axes[_c_idx, 0]
        _ax_hosp = _axes[_c_idx, 1]
        _ax_dth  = _axes[_c_idx, 2]

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _color = _colors[_s_idx % len(_colors)]
            _ls    = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{param_name}={_scenario_label}"

            # Daily deaths from HD_to_D tvar (same pattern as daily_new_infections)
            def _daily_deaths(m, subpop_name=_subpop, age_group=_age_group):
                from flu_core.flu_outcomes import _tvar_daily, _apply_ar_filter
                return _apply_ar_filter(
                    _tvar_daily(m, ["HD_to_D"], subpop_name), age_group, None
                )

            if num_reps > 1:
                for _ax, _fn, _ylabel in [
                    (_ax_inf,  daily_new_infections,     "Cumulative infections"),
                    (_ax_hosp, daily_hospital_admissions,"Cumulative hospitalizations"),
                    (_ax_dth,  _daily_deaths,            "Cumulative deaths"),
                ]:
                    _daily = np.stack([_fn(m, _subpop, _age_group) for m in _model_list], axis=0)
                    _cum   = np.cumsum(_daily, axis=1)
                    _med   = np.median(_cum, axis=0)
                    _lo    = np.percentile(_cum, 2.5,  axis=0)
                    _hi    = np.percentile(_cum, 97.5, axis=0)
                    _days  = np.arange(_med.shape[0])
                    _ax.plot(_days, _med, label=_label, color=_color, linestyle=_ls)
                    _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
                    _ax.set_ylabel(_ylabel)
            else:
                for _ax, _fn, _ylabel in [
                    (_ax_inf,  daily_new_infections,     "Cumulative infections"),
                    (_ax_hosp, daily_hospital_admissions,"Cumulative hospitalizations"),
                    (_ax_dth,  _daily_deaths,            "Cumulative deaths"),
                ]:
                    _cum = np.cumsum(_fn(_model_list[0], _subpop, _age_group))
                    _ax.plot(_cum, label=_label, color=_color, linestyle=_ls)
                    _ax.set_ylabel(_ylabel)

        for _ax, _title in [
            (_ax_inf,  f"Cumulative infections — {_combo_label}"),
            (_ax_hosp, f"Cumulative hospitalizations — {_combo_label}"),
            (_ax_dth,  f"Cumulative deaths — {_combo_label}"),
        ]:
            _ax.set_xlabel("Day")
            _ax.set_title(_title)
            _ax.legend(fontsize=7)

    plt.tight_layout()
    _fig
    return


@app.cell
def _summary_table(
    age_group_selector,
    attack_rate,
    cumulative_deaths,
    cumulative_hospitalizations,
    daily_hospital_admissions,
    mo,
    np,
    param_name,
    pd,
    results,
    subpop_selector,
    summarize_outcomes,
):
    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]

    rows = []
    for _sp, _ag in _combos:
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)

        for _scenario_label, _model_list in results.items():
            hosp_vals = [cumulative_hospitalizations(m, _subpop, _age_group) for m in _model_list]
            death_vals = [cumulative_deaths(m, _subpop, _age_group) for m in _model_list]
            ar_vals = [attack_rate(m, _subpop, _age_group) for m in _model_list]

            daily_adm = np.stack(
                [daily_hospital_admissions(m, _subpop, _age_group) for m in _model_list],
                axis=0,
            )
            peak_vals = daily_adm.max(axis=1).tolist()

            def fmt(vals):
                s = summarize_outcomes(vals)
                if len(vals) == 1:
                    return f"{vals[0]:.1f}"
                return f"{s['mean']:.1f} [{s['lower_ci']:.1f}–{s['upper_ci']:.1f}]"

            rows.append({
                "Subpopulation": _sp,
                "Age group": _ag,
                f"{param_name}": _scenario_label,
                "Total hospitalizations": fmt(hosp_vals),
                "Total deaths": fmt(death_vals),
                "Attack rate": f"{np.mean(ar_vals):.3f}",
                "Peak daily admissions": fmt(peak_vals),
            })
    mo.vstack([mo.md("## Summary"), mo.ui.table(pd.DataFrame(rows))])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
