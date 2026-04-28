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
        flu,
        mo,
        np,
        pd,
        plt,
        summarize_outcomes,
        sys,
    )


@app.cell
def _filename(mo, sys):
    if len(sys.argv) > 0:
        notebook_filename = sys.argv[0]
        # mo.md(f"**Current Notebook Filename:** `{notebook_filename}`")
        mo.md(f"`{notebook_filename}`")
    
    mo.md(f"`Flu Sensitivity Analysis`")
    return


@app.cell
def _load_files(clt, flu, pd):
    from flu_example_utils import load_flu_inputs, SUBPOP_CONFIG as _SC, SHARED_FILES_CONFIG
    inputs = load_flu_inputs(_SC, SHARED_FILES_CONFIG, clt, flu, pd)
    params_baseline = inputs["params_baseline"]
    mixing_params = inputs["mixing_params"]
    settings_base = inputs["settings_base"]
    return inputs, mixing_params, params_baseline, settings_base


@app.cell
def _param_catalog(flu, inputs, params_baseline):
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
        arr = _np.asarray(val)
        if arr.ndim == 2:
            return "[" + ", ".join(
                "[" + ", ".join(f"{x:.4g}" for x in row) + "]" for row in arr
            ) + "]"
        return "[" + ", ".join(f"{x:.4g}" for x in arr.flatten()) + "]"

    ARRAY_BASELINES = [_fmt(getattr(params_baseline, p)) for p in ARRAY_PARAMS]

    # Known descriptions for scalar parameters (first element of each value tuple).
    # Add entries here when new descriptively-named parameters are introduced.
    _DESCRIPTIONS = {
        "beta_baseline":                "β baseline (transmission)",
        "humidity_impact":              "Humidity impact",
        "inf_induced_saturation":       "Inf. induced saturation",
        "inf_induced_immune_wane":      "Inf. immunity wane rate",
        "vax_induced_saturation":       "Vax. induced saturation",
        "inf_induced_inf_risk_reduce":  "Inf. → inf. risk reduction",
        "inf_induced_hosp_risk_reduce": "Inf. → hosp. risk reduction",
        "vax_induced_inf_risk_reduce":  "Vax. → inf. risk reduction",
        "vax_induced_hosp_risk_reduce": "Vax. → hosp. risk reduction",
        "vax_induced_death_risk_reduce":"Vax. → death risk reduction",
        "E_to_I_rate":                  "E → I rate",
        "IP_to_IS_rate":                "IP → IS rate",
        "ISR_to_R_rate":                "ISR → R rate",
        "ISH_to_H_rate":                "ISH → H rate",
        "IP_relative_inf":              "IP relative infectiousness",
        "IA_relative_inf":              "IA relative infectiousness",
    }

    def _slider_range(val):
        """Derive (min, max, step) from a baseline value."""
        val = float(val)
        if val == 0.0:
            return 0.0, 10.0, 0.1
        magnitude = 10 ** _np.floor(_np.log10(abs(val)))
        step = round(magnitude / 10, 10)
        lo = 0.0
        hi = round(max(val * 5, magnitude * 10), 10)
        return lo, hi, step

    # Auto-generate SCALAR_PARAMS as name → (label, default, min, max, step)
    # from all scalar (int/float) fields not in _SKIP.
    SCALAR_PARAMS = {}
    for _f in _dc.fields(flu.FluSubpopParams):
        _name = _f.name
        if _name in _SKIP:
            continue
        _val = getattr(params_baseline, _name)
        if not isinstance(_val, (int, float)):
            continue
        _label = _DESCRIPTIONS.get(_name, _name)
        _lo, _hi, _step = _slider_range(_val)
        SCALAR_PARAMS[_name] = (_label, float(_val), _lo, _hi, _step)

    # Add non-zero init condition scalars from loaded states (excludes S, M, MV)
    from flu_example_utils import SUBPOP_CONFIG as _SC_IC
    _INIT_COMPS = ["E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    for _sp_cfg in _SC_IC:
        _sp_name = _sp_cfg["name"]
        _state = inputs["states"][_sp_name]
        for _comp in _INIT_COMPS:
            _field = getattr(_state, _comp, None)
            if _field is None:
                continue
            _arr = _np.asarray(_field)
            for _idx in _np.argwhere(_arr != 0):
                _i, _j = int(_idx[0]), int(_idx[1])
                _val = float(_arr[_i, _j])
                _key = f"init:{_sp_name}:{_comp}:{_i}:{_j}"
                _lo_ic, _hi_ic, _step_ic = _slider_range(_val)
                SCALAR_PARAMS[_key] = (
                    f"{_sp_name} {_comp}[{_i}][{_j}]",
                    _val, _lo_ic, _hi_ic, _step_ic,
                )

    # Add M arrays: one per subpop plus one aggregate entry
    _M_PARAM_KEYS = [f"M:{sp['name']}" for sp in _SC_IC] + ["M:all"]
    _M_BASELINES_LIST = []
    for _mk in _M_PARAM_KEYS:
        if _mk == "M:all":
            _parts = [
                f"{sp['name']}: " + _fmt(getattr(inputs["states"][sp["name"]], "M"))
                for sp in _SC_IC
            ]
            _M_BASELINES_LIST.append(" | ".join(_parts))
        else:
            _sp_nm = _mk.split(":")[1]
            _M_BASELINES_LIST.append(_fmt(getattr(inputs["states"][_sp_nm], "M")))
    ARRAY_PARAMS = ARRAY_PARAMS + _M_PARAM_KEYS
    ARRAY_BASELINES = ARRAY_BASELINES + _M_BASELINES_LIST

    return ARRAY_BASELINES, ARRAY_PARAMS, SCALAR_PARAMS


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
        options=["Scalar", "Array (scale factor)", "Other"],
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
        start=0.0, stop=3.0, step=0.05, value=1.0,
        label="Vaccine coverage multiplier",
    )

    # Subpopulation multi-selector
    from flu_example_utils import subpop_dropdown_options, SUBPOP_CONFIG as _SC
    subpop_selector = mo.ui.multiselect(
        options=subpop_dropdown_options(_SC, aggregate_label="combined"),
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

    run_button = mo.ui.run_button(label="Run simulation")

    return (
        age_group_selector,
        num_reps_input,
        num_values_input,
        param_selector,
        param_type,
        run_button,
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
    ARRAY_BASELINES,
    ARRAY_PARAMS,
    array_param_selector,
    mo,
    num_values_input,
):
    _n = num_values_input.value

    array_sliders = mo.ui.array(
        [
            mo.ui.slider(
                start=0.0, stop=5.0, step=0.05, value=1.0,
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
def _other_param_controls(mo, num_values_input, settings_base):
    _n = num_values_input.value
    _default_date = str(settings_base.start_real_date)

    start_date_inputs = mo.ui.array(
        [
            mo.ui.date(value=_default_date, label=f"Start date {i + 1}")
            for i in range(_n)
        ]
    )

    other_controls_ui = mo.vstack([
        mo.md("### Other parameter: `start_real_date`"),
        mo.md("Set start date values to overlay (identical values collapse to one curve):"),
        start_date_inputs,
    ])
    return other_controls_ui, start_date_inputs


@app.cell
def _run_simulation(
    array_param_selector,
    array_sliders,
    clt,
    flu,
    inputs,
    mo,
    np,
    num_reps_input,
    param_selector,
    param_type,
    params_baseline,
    run_button,
    settings_base,
    sim_length,
    sim_mode,
    sliders,
    start_date_inputs,
    vax_multiplier,
):
    mo.stop(not run_button.value, mo.md("Press **Run simulation** to start."))

    from flu_example_utils import (
        SUBPOP_CONFIG as _SC,
        scale_vaccines_df,
        make_rng_generators,
        build_flu_metapop_model,
        apply_general_init_overrides,
    )

    transition_type = (
        "binom_deterministic_no_round"
        if sim_mode.value == "Deterministic"
        else "binom"
    )
    num_reps = num_reps_input.value if sim_mode.value == "Stochastic" else 1
    end_day = sim_length.value

    vax_scale = vax_multiplier.value
    scaled_vax = {
        sp["name"]: scale_vaccines_df(inputs["vaccines_df"][sp["name"]], vax_scale, np)
        for sp in _SC
    }

    # Parameter values/scale factors to overlay (deduplicated, preserving order)
    _is_array_mode = param_type.value == "Array (scale factor)"
    _is_other_mode = param_type.value == "Other"

    if _is_array_mode:
        _apn = array_param_selector.value
        param_values = list(dict.fromkeys(array_sliders.value))

        if str(_apn).startswith("M:"):
            _sp_part = _apn.split(":")[1]
            param_name = f"M ({'all subpops' if _sp_part == 'all' else _sp_part}) ×scale"
            _make_params = lambda _: params_baseline
            _make_states = lambda val, _k=_apn: apply_general_init_overrides(
                inputs["states"], {_k: val}, _SC, np)
        else:
            param_name = f"{_apn} ×scale"
            _make_params = lambda val, _a=_apn: clt.updated_dataclass(
                params_baseline, {_a: np.asarray(getattr(params_baseline, _a)) * val})
            _make_states = lambda _: inputs["states"]

    elif _is_other_mode:
        param_name = "start_real_date"
        param_values = list(dict.fromkeys(start_date_inputs.value))
        _make_params = lambda _: params_baseline
        _make_states = lambda _: inputs["states"]

    else:
        _raw_pname = param_selector.value
        param_values = list(dict.fromkeys(sliders.value))

        if str(_raw_pname).startswith("init:"):
            _, _sp_nm, _comp, _ii, _jj = _raw_pname.split(":")
            param_name = f"{_sp_nm} {_comp}[{_ii}][{_jj}]"
            _make_params = lambda _: params_baseline
            _make_states = lambda val, _k=_raw_pname: apply_general_init_overrides(
                inputs["states"], {_k: val}, _SC, np)
        else:
            param_name = _raw_pname
            _make_params = lambda val, _pn=_raw_pname: clt.updated_dataclass(
                params_baseline, {_pn: val})
            _make_states = lambda _: inputs["states"]

    _default_start = str(settings_base.start_real_date)
    if _is_other_mode:
        scenario_start_dates = {str(v): str(v) for v in param_values}
    else:
        scenario_start_dates = {str(v): _default_start for v in param_values}

    tvar_to_save = ["ISH_to_HR", "ISH_to_HD", "S_to_E", "HD_to_D"]
    updated_settings = clt.updated_dataclass(settings_base, {
        "transition_type": transition_type,
        "transition_variables_to_save": tvar_to_save,
    })

    def build_and_run(param_val):
        updated_params = _make_params(param_val)
        updated_states = _make_states(param_val)
        _run_inputs = {**inputs, "states": updated_states}
        rng_results = []
        for rep in range(num_reps):
            rngs = make_rng_generators(88888 + rep, _SC, np)
            model = build_flu_metapop_model(
                _SC, _run_inputs, updated_params, updated_settings,
                rngs, scaled_vax, flu,
            )
            model.simulate_until_day(end_day)
            rng_results.append(model)
        return rng_results

    with mo.status.spinner("Running simulation..."):
        results = {str(v): build_and_run(v) for v in param_values}
    return num_reps, param_name, results, scenario_start_dates


@app.cell
def _show_controls(
    array_sliders_ui,
    controls_form,
    mo,
    other_controls_ui,
    param_type,
    run_button,
    sliders_ui,
):
    if param_type.value == "Array (scale factor)":
        _param_controls = array_sliders_ui
    elif param_type.value == "Other":
        _param_controls = other_controls_ui
    else:
        _param_controls = sliders_ui
    mo.vstack([controls_form, _param_controls, run_button])
    return


@app.cell
def _cumulative_vax_display(inputs, mo, np, vax_multiplier):
    from flu_example_utils import (
        SUBPOP_CONFIG as _SC,
        compute_cumulative_vax,
        make_cumvax_markdown_table,
    )

    _scale = vax_multiplier.value
    _cum_vax = {
        sp["name"]: compute_cumulative_vax(inputs["vaccines_df"][sp["name"]], _scale, np)
        for sp in _SC
    }

    _warn = " ⚠️" if any(v.max() > 1 for v in _cum_vax.values()) else ""
    _title = f"Cumulative vaccination rates (multiplier ×{_scale:.2f}){_warn}"
    mo.accordion({
        _title: mo.hstack([
            mo.md(make_cumvax_markdown_table(_cum_vax[sp["name"]], sp["name"].capitalize()))
            for sp in _SC
        ]),
    })
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
    np,
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
    subpop_selector,
):
    _ALL_COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    _LINE_STYLES = ["-", "--", ":", "-."]
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    _selected = [c for c, v in zip(_ALL_COMPARTMENTS, compartment_checkboxes.value) if v] or _ALL_COMPARTMENTS

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 5 * _n_combos), squeeze=False)

    def _extract_comp(model, comp_name, subpop_name, age_group):
        _subpops = (
            [model.subpop_models[subpop_name]]
            if subpop_name is not None
            else list(model.subpop_models.values())
        )
        _arrays = [np.asarray(sp.compartments[comp_name].history_vals_list) for sp in _subpops]
        _total = np.sum(np.stack(_arrays, axis=0), axis=0)  # (T, A, R)
        if age_group is not None:
            return _total[:, age_group, :].sum(axis=1)
        return _total.sum(axis=(1, 2))

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp} / age {_ag}"

        _date_extents = []
        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label_sfx = f" [{param_name}={_scenario_label}]"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

            for _comp_idx, _comp_name in enumerate(_selected):
                _color = _colors[_comp_idx % len(_colors)]
                _all_vals = np.stack(
                    [_extract_comp(m, _comp_name, _subpop, _age_group) for m in _model_list],
                    axis=0,
                )  # (reps, T) — T is days (save_daily_history=True)
                _dates  = pd.date_range(start=_start, periods=_all_vals.shape[1], freq='D')
                if _comp_idx == 0:
                    _date_extents.append((_dates[0], _dates[-1]))
                _median = np.median(_all_vals, axis=0)
                _lo     = np.percentile(_all_vals, 2.5,  axis=0)
                _hi     = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_dates, _median, label=f"{_comp_name}{_label_sfx}",
                         linestyle=_ls, color=_color, alpha=0.7)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.15)

        if _date_extents:
            _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))
        _ax.set_xlabel("Date")
        _ax.set_ylabel("Number of individuals")
        _ax.set_title(f"Compartment histories (median + 95% CI) — {_combo_label}")
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels, fontsize=7, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig.subplots_adjust(right=0.92)
    mo.vstack([mo.md("### Compartments to show"), mo.hstack(compartment_checkboxes.elements, justify="start"), _fig])
    return


@app.cell
def _plot_epi_metrics(
    age_group_selector,
    epi_metric_checkboxes,
    mo,
    np,
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
    subpop_selector,
):
    _ALL_METRICS = ["M", "MV"]
    _LINE_STYLES = ["-", "--", ":", "-."]
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _selected_metrics = [m for m, v in zip(_ALL_METRICS, epi_metric_checkboxes.value) if v] or _ALL_METRICS

    _combos = [
        (sp, ag)
        for sp in (subpop_selector.value or ["combined"])
        for ag in (age_group_selector.value or ["all"])
    ]
    _n_combos = len(_combos)

    _fig, _axes = plt.subplots(_n_combos, 1, figsize=(10, 4 * _n_combos), squeeze=False)

    def _extract_metric(model, metric_name, subpop_name, age_group):
        _subpops = (
            [model.subpop_models[subpop_name]]
            if subpop_name is not None
            else list(model.subpop_models.values())
        )
        _arrays = [np.asarray(sp.epi_metrics[metric_name].history_vals_list) for sp in _subpops]
        _total = np.mean(np.stack(_arrays, axis=0), axis=0)  # (T, A, R)
        if age_group is not None:
            return _total[:, age_group, :].mean(axis=1)
        return _total.mean(axis=(1, 2))

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _ax = _axes[_c_idx, 0]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp} / age {_ag}"

        _date_extents = []
        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label_sfx = f" [{param_name}={_scenario_label}]"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

            for _metric_idx, _metric_name in enumerate(_selected_metrics):
                _color = _colors[_metric_idx % len(_colors)]
                _all_vals = np.stack(
                    [_extract_metric(m, _metric_name, _subpop, _age_group) for m in _model_list],
                    axis=0,
                )  # (reps, T) — T is days (save_daily_history=True)
                _dates  = pd.date_range(start=_start, periods=_all_vals.shape[1], freq='D')
                if _metric_idx == 0:
                    _date_extents.append((_dates[0], _dates[-1]))
                _median = np.median(_all_vals, axis=0)
                _lo     = np.percentile(_all_vals, 2.5,  axis=0)
                _hi     = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_dates, _median, label=f"{_metric_name}{_label_sfx}",
                         linestyle=_ls, color=_color, alpha=0.7)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.15)

        if _date_extents:
            _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))
        _ax.set_xlabel("Date")
        _ax.set_ylabel("Immunity level")
        _ax.set_title(f"{', '.join(_selected_metrics)} (median + 95% CI) — {_combo_label}")
        _handles, _labels = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), borderaxespad=0)

    _fig.autofmt_xdate()
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
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
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
    _date_extents = []

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _color = _colors[_c_idx % len(_colors)]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp}/age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{_combo_label} | {param_name}={_scenario_label}"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

            if num_reps > 1:
                _all_vals = np.stack(
                    [daily_new_infections(m, _subpop, _age_group) for m in _model_list], axis=0
                )
                _dates = pd.date_range(start=_start, periods=_all_vals.shape[1], freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _median = np.median(_all_vals, axis=0)
                _lo = np.percentile(_all_vals, 2.5, axis=0)
                _hi = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_dates, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.2)
            else:
                _vals = daily_new_infections(_model_list[0], _subpop, _age_group)
                _dates = pd.date_range(start=_start, periods=len(_vals), freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _ax.plot(_dates, _vals, label=_label, color=_color, linestyle=_ls)

    if _date_extents:
        _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Daily new infections")
    _ax.set_title(f"Daily new infections — {param_name}")
    _ax.legend()
    _fig.autofmt_xdate()
    plt.tight_layout()
    _fig
    return


@app.cell
def _plot_admissions(
    age_group_selector,
    daily_hospital_admissions,
    np,
    num_reps,
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
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
    _date_extents = []

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _color = _colors[_c_idx % len(_colors)]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp}/age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{_combo_label} | {param_name}={_scenario_label}"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

            if num_reps > 1:
                _all_vals = np.stack(
                    [daily_hospital_admissions(m, _subpop, _age_group) for m in _model_list], axis=0
                )
                _dates = pd.date_range(start=_start, periods=_all_vals.shape[1], freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _median = np.median(_all_vals, axis=0)
                _lo = np.percentile(_all_vals, 2.5, axis=0)
                _hi = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_dates, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.2)
            else:
                _vals = daily_hospital_admissions(_model_list[0], _subpop, _age_group)
                _dates = pd.date_range(start=_start, periods=len(_vals), freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _ax.plot(_dates, _vals, label=_label, color=_color, linestyle=_ls)

    if _date_extents:
        _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Daily hospital admissions")
    _ax.set_title(f"Daily hospital admissions — {param_name}")
    _ax.legend()
    _fig.autofmt_xdate()
    plt.tight_layout()
    _fig
    return


@app.cell
def _plot_deaths(
    age_group_selector,
    daily_deaths,
    np,
    num_reps,
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
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
    _date_extents = []

    for _c_idx, (_sp, _ag) in enumerate(_combos):
        _color = _colors[_c_idx % len(_colors)]
        _subpop = None if _sp == "combined" else _sp
        _age_group = None if _ag == "all" else int(_ag)
        _combo_label = f"{_sp}/age {_ag}"

        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _ls = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{_combo_label} | {param_name}={_scenario_label}"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

            if num_reps > 1:
                _all_vals = np.stack(
                    [daily_deaths(m, _subpop, _age_group) for m in _model_list], axis=0
                )
                _dates = pd.date_range(start=_start, periods=_all_vals.shape[1], freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _median = np.median(_all_vals, axis=0)
                _lo = np.percentile(_all_vals, 2.5, axis=0)
                _hi = np.percentile(_all_vals, 97.5, axis=0)
                _ax.plot(_dates, _median, label=_label, color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.2)
            else:
                _vals = daily_deaths(_model_list[0], _subpop, _age_group)
                _dates = pd.date_range(start=_start, periods=len(_vals), freq='D')
                _date_extents.append((_dates[0], _dates[-1]))
                _ax.plot(_dates, _vals, label=_label, color=_color, linestyle=_ls)

    if _date_extents:
        _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))
    _ax.set_xlabel("Date")
    _ax.set_ylabel("Daily deaths")
    _ax.set_title(f"Daily deaths — {param_name}")
    _ax.legend()
    _fig.autofmt_xdate()
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
    pd,
    param_name,
    plt,
    results,
    scenario_start_dates,
    settings_base,
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

        _date_extents = []
        for _s_idx, (_scenario_label, _model_list) in enumerate(results.items()):
            _color = _colors[_s_idx % len(_colors)]
            _ls    = _LINE_STYLES[_s_idx % len(_LINE_STYLES)]
            _label = f"{param_name}={_scenario_label}"
            _start = scenario_start_dates.get(_scenario_label, str(settings_base.start_real_date))

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
                    _dates = pd.date_range(start=_start, periods=_med.shape[0], freq='D')
                    _date_extents.append((_dates[0], _dates[-1]))
                    _ax.plot(_dates, _med, label=_label, color=_color, linestyle=_ls)
                    _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.2)
                    _ax.set_ylabel(_ylabel)
            else:
                for _ax, _fn, _ylabel in [
                    (_ax_inf,  daily_new_infections,     "Cumulative infections"),
                    (_ax_hosp, daily_hospital_admissions,"Cumulative hospitalizations"),
                    (_ax_dth,  _daily_deaths,            "Cumulative deaths"),
                ]:
                    _cum = np.cumsum(_fn(_model_list[0], _subpop, _age_group))
                    _dates = pd.date_range(start=_start, periods=len(_cum), freq='D')
                    _date_extents.append((_dates[0], _dates[-1]))
                    _ax.plot(_dates, _cum, label=_label, color=_color, linestyle=_ls)
                    _ax.set_ylabel(_ylabel)

        if _date_extents:
            for _ax in [_ax_inf, _ax_hosp, _ax_dth]:
                _ax.set_xlim(min(d[0] for d in _date_extents), max(d[1] for d in _date_extents))

        for _ax, _title in [
            (_ax_inf,  f"Cumulative infections — {_combo_label}"),
            (_ax_hosp, f"Cumulative hospitalizations — {_combo_label}"),
            (_ax_dth,  f"Cumulative deaths — {_combo_label}"),
        ]:
            _ax.set_xlabel("Date")
            _ax.set_title(_title)
            _ax.legend(fontsize=7)

    _fig.autofmt_xdate()
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
            days_to_peak_vals = np.argmax(daily_adm, axis=1).tolist()

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
                "Days to peak admissions": fmt(days_to_peak_vals),
            })
    mo.vstack([mo.md("## Summary"), mo.ui.table(pd.DataFrame(rows))])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
