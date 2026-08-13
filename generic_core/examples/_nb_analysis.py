# _nb_analysis.py
# Section: Analysis tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py
# (excludes metric-def widgets which are in _nb_analysis_metric_defs.py)

@app.cell
def _analysis_sub_tab(mo):
    analysis_sub_tab = mo.ui.tabs({"Sensitivity": mo.md(""), "Scenario": mo.md("")})
    return (analysis_sub_tab,)


@app.cell
def _analysis_fitted_params_ui(mo):
    analysis_use_fitted = mo.ui.switch(label="Override baseline with fitted params", value=False)
    analysis_fitted_source = mo.ui.radio(
        options=["Fitting tab result", "Upload JSON file"],
        value="Fitting tab result",
        label="Source",
    )
    analysis_fitted_params_path = mo.ui.text(
        label="Fitted params JSON path",
        placeholder="~/clt_outputs/fitted_params.json",
        full_width=True,
    )
    return analysis_use_fitted, analysis_fitted_source, analysis_fitted_params_path


@app.cell
def _analysis_fitted_params_load(
    analysis_use_fitted, analysis_fitted_source, analysis_fitted_params_path,
    fit_result, config_dict, is_metapop, json, Path, np, mo,
):
    analysis_fitted_params = {}
    analysis_fitted_tv_increments = None
    analysis_fitted_tv_spacing = 30
    analysis_fitted_num_days = 0
    # Full, unexpanded fitted-params structure (best_params/scale_groups/
    # num_days/tv_knot_spacing_days/accepted_params) for the Export tab —
    # analysis_fitted_params has already been flattened/expanded/filtered for
    # in-notebook use and has lost the info needed to reproduce seed_scale_*/
    # m(t) in the exported run_simulation.py script.
    analysis_fitted_full = {}
    analysis_fitted_note = mo.md("")
    if analysis_use_fitted.value:
        _raw = None
        _scale_groups = {}
        _tv_spacing = 30
        _fit_num_days = 0
        _err = None
        if analysis_fitted_source.value == "Fitting tab result":
            if fit_result is not None:
                _raw = dict(fit_result.best_params)
                _scale_groups = dict(getattr(fit_result, "scale_groups", {}) or {})
                _tv_spacing = int(getattr(fit_result, "tv_knot_spacing_days", 30) or 30)
                _fit_num_days = int(fit_result.num_days)
                analysis_fitted_full = {
                    "best_params": fit_result.best_params,
                    "num_days": fit_result.num_days,
                    "method": fit_result.method,
                    "accepted_params": fit_result.accepted_params,
                    "scale_groups": fit_result.scale_groups,
                    "tv_knot_spacing_days": fit_result.tv_knot_spacing_days,
                }
            else:
                _err = "No fitting results available. Run fitting first, or switch to 'Upload JSON file'."
        else:
            _pp = analysis_fitted_params_path.value.strip()
            if not _pp:
                _err = "Enter a path to a fitted params JSON file."
            else:
                try:
                    with open(Path(_pp).expanduser()) as _f:
                        _loaded = json.load(_f)
                    _raw = _loaded.get("best_params", _loaded) if isinstance(_loaded, dict) else {}
                    _scale_groups = dict(_loaded.get("scale_groups", {}) or {}) if isinstance(_loaded, dict) else {}
                    _tv_spacing = int(_loaded.get("tv_knot_spacing_days", 30) or 30) if isinstance(_loaded, dict) else 30
                    _fit_num_days = int(_loaded.get("num_days", 0) or 0) if isinstance(_loaded, dict) else 0
                    analysis_fitted_full = (
                        _loaded if isinstance(_loaded, dict) and "best_params" in _loaded
                        else {"best_params": _raw}
                    )
                except Exception as _exc:
                    _err = f"Could not load fitted params: {_exc}"

        if _err is not None:
            analysis_fitted_note = mo.callout(mo.md(f"**{_err}**"), kind="warn")
        else:
            # MCMC/ABC-SMC record one posterior column per age/risk element
            # (`pn|a0`, `pn|a1`, ...) rather than one array-valued `pn` entry
            # (AR/gradient do the latter directly) — reassemble first so a
            # granular param applies the same way regardless of method.
            from generic_core.fitting import merge_param_slots as _merge_param_slots
            _raw = _merge_param_slots(_raw)
            # Expand linked-scale multipliers (from AR/gradient fits) into concrete
            # base-param overrides, same as the Forecast tab does.
            _out = {_k: _v for _k, _v in _raw.items() if _k not in _scale_groups}
            if _scale_groups:
                _baselines = config_dict.get("params", {}) or {}
                for _m, _bases in _scale_groups.items():
                    if _m not in _raw:
                        continue
                    _mult_raw = _raw[_m]
                    _mult = (
                        np.asarray(_mult_raw, dtype=float)
                        if isinstance(_mult_raw, (list, tuple))
                        else float(_mult_raw)
                    )
                    for _b in _bases:
                        _bl = _baselines.get(_b)
                        if isinstance(_bl, (list, tuple)) or isinstance(_mult, np.ndarray):
                            _out[_b] = (np.asarray(_bl, dtype=float) * _mult).tolist()
                        elif _bl is not None:
                            _out[_b] = float(_bl) * _mult

            # m_dlog_* are the log-increments of the fitted time-varying
            # transmission multiplier m(t) (single-population only); phi is
            # the NB2 observation-noise dispersion param. Neither is a
            # config["params"] entry, so both are pulled out of the flat
            # scalar-override dict — m(t) is reconstructed separately below
            # (see _run_analysis), phi is simply not a model parameter.
            _mdlog_items = sorted(
                (
                    (int(_k[len("m_dlog_"):]), float(_v))
                    for _k, _v in _out.items()
                    if _k.startswith("m_dlog_") and _k[len("m_dlog_"):].isdigit()
                ),
                key=lambda _t: _t[0],
            )
            _has_mt = bool(_mdlog_items)

            # Keep both scalar and array-valued overrides (the latter mainly
            # from linked scale groups expanded above, e.g. IHR_scale ->
            # I_to_H_prop/IV_to_H_prop) — _analysis_param_catalog applies
            # either kind directly.
            analysis_fitted_params = {
                _k: (_v if isinstance(_v, list) else float(_v))
                for _k, _v in _out.items()
                if _k != "phi" and not _k.startswith("m_dlog_")
            }
            _n = len(analysis_fitted_params)
            _msgs = []
            if _n:
                _names = ", ".join(f"`{_k}`" for _k in sorted(analysis_fitted_params))
                _msgs.append(
                    f"**{_n} fitted parameter(s)** ({_names}) will override the baseline "
                    "used for sensitivity/scenario analysis."
                )
            else:
                _msgs.append("No matching parameters found in the fitted params.")
            if _has_mt:
                if _fit_num_days <= 0:
                    _msgs.append(
                        "**Time-varying transmission (`m_dlog_*`) found but the fitted params "
                        "are missing `num_days`** — can't reconstruct m(t) without the fit period "
                        "length; ignored."
                    )
                else:
                    analysis_fitted_tv_increments = [_v for _, _v in _mdlog_items]
                    analysis_fitted_tv_spacing = _tv_spacing
                    analysis_fitted_num_days = _fit_num_days
                    if is_metapop:
                        _msgs.append(
                            f"**Time-varying transmission m(t)** (fitted on a single population) "
                            f"reconstructed from {len(_mdlog_items)} fitted log-increment(s) over a "
                            f"{_fit_num_days}-day fit period (knots every {_tv_spacing} day(s)), held "
                            "flat past it, and **broadcast uniformly to every subpopulation**."
                        )
                    else:
                        _msgs.append(
                            f"**Time-varying transmission m(t)** reconstructed from "
                            f"{len(_mdlog_items)} fitted log-increment(s) over a {_fit_num_days}-day "
                            f"fit period (knots every {_tv_spacing} day(s)), held flat past it."
                        )
            analysis_fitted_note = mo.callout(mo.md("\n\n".join(_msgs)), kind="success" if _n or _has_mt else "warn")
    return (
        analysis_fitted_params, analysis_fitted_note,
        analysis_fitted_tv_increments, analysis_fitted_tv_spacing, analysis_fitted_num_days,
        analysis_fitted_full,
    )


@app.cell
def _analysis_fitted_param_sets(analysis_use_fitted, analysis_fitted_full, config_dict):
    # Accepted parameter sets (MCMC/ABC posterior draws, AR-accepted samples,
    # or one best per gradient replication) behind the single best set that
    # _analysis_fitted_params_load extracts. These drive the optional
    # parameter-uncertainty ensemble in _run_analysis; a fit with a single
    # accepted set offers no parameter uncertainty to propagate, so the whole
    # feature stays hidden in that case.
    analysis_param_sets = []
    if analysis_use_fitted.value:
        _accepted = (analysis_fitted_full or {}).get("accepted_params") or []
        if len(_accepted) > 1:
            from generic_core.fitting import prepare_param_sets as _prepare_param_sets
            analysis_param_sets = _prepare_param_sets(
                _accepted,
                dict((analysis_fitted_full or {}).get("scale_groups", {}) or {}),
                config_dict.get("params", {}) or {},
            )
    analysis_n_param_sets_avail = len(analysis_param_sets)
    return analysis_param_sets, analysis_n_param_sets_avail


@app.cell
def _analysis_param_catalog(
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    param_grid_columns, num_age_groups, num_risk_groups, age_groups,
    analysis_fitted_params,
):
    import math as _math
    import numpy as _np

    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)

    def _fitted_grid(_fv):
        # Reshape a fitted value onto the builder's (A, R) grid so it can
        # replace an age/risk-varying baseline. Fitted granular params arrive
        # as (A, R) nested lists (merge_param_slots / scale-group expansion);
        # a scalar or per-age vector is broadcast the same way the model
        # broadcasts a scalar/1-D param_override. Returns None if the shape
        # can't be matched, in which case the caller keeps the grid values.
        _arr = _np.asarray(_fv, dtype=float)
        if _arr.ndim == 0:
            _arr = _np.full((_A, _R), float(_arr))
        elif _arr.ndim == 1 and _arr.shape[0] == _A:
            _arr = _np.repeat(_arr[:, None], _R, axis=1)
        elif _arr.shape != (_A, _R):
            return None
        return [[float(_arr[_a][_r]) for _r in range(_R)] for _a in range(_A)]

    _params = {}
    for _name in param_names:
        # A fitted value always wins over the builder baseline — including for
        # age/risk-varying params, whose fitted per-element values would
        # otherwise be silently replaced by the (unfitted) grid, since every
        # scenario emits `catalog baseline × scale` as an explicit override.
        _fitted = analysis_fitted_params.get(_name)
        if param_vary_toggles[_name].value:
            _grid = _fitted_grid(_fitted) if _fitted is not None else None
            if _grid is None:
                _rows = list(param_grid_inputs[_name].value)
                _grid = [
                    [float(_rows[_r][_age_cols[_a]]) for _r in range(_R)]
                    for _a in range(_A)
                ]
            _params[_name] = _grid
        elif _fitted is not None:
            _params[_name] = list(_fitted) if isinstance(_fitted, list) else float(_fitted)
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
    # The swept parameter is pinned to its slider value for every run AND is
    # deliberately shielded from sampled fitted parameter sets (it lands in
    # `designed` — see _analysis_define_scenarios), so whichever param is
    # selected here silently loses its posterior uncertainty in the ensemble.
    # Pre-selecting the first param in the dropdown therefore mis-specifies the
    # run for anyone who only wanted a fitted baseline — and the first param is
    # often exactly one the fit varied (e.g. beta_baseline). So the sweep
    # defaults to "no parameter": one baseline scenario at the current values,
    # with nothing shielded from the fitted/sampled params.
    ANALYSIS_NO_SWEEP = "— none (baseline only) —"
    _scalar_opts = [ANALYSIS_NO_SWEEP] + list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_opts = [ANALYSIS_NO_SWEEP] + list(ANALYSIS_ARRAY_PARAMS.keys())
    analysis_param_type = mo.ui.radio(
        options=["Scalar", "Array (scale factor)"],
        value="Scalar",
        label="Parameter type",
    )
    analysis_scalar_param_sel = mo.ui.dropdown(
        options=_scalar_opts,
        value=ANALYSIS_NO_SWEEP,
        label="Scalar parameter to vary",
    )
    analysis_array_param_sel = mo.ui.dropdown(
        options=_array_opts,
        value=ANALYSIS_NO_SWEEP,
        label="Array parameter to scale",
    )
    analysis_n_values = mo.ui.number(start=1, stop=6, step=1, value=3, label="Values to compare")
    return (
        analysis_param_type, analysis_scalar_param_sel, analysis_array_param_sel,
        analysis_n_values, ANALYSIS_NO_SWEEP,
    )


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
def _analysis_scenario_state(mo):
    # Live edits to every Scenario sub-tab widget below (names, per-scenario
    # overrides, subpop selections, dose multipliers, ...), keyed by a string
    # identifying the control. Widgets below read this dict as their default
    # value every time their defining cell reruns -- which happens on nearly
    # every edit to the Model Builder param catalog (ANALYSIS_SCALAR_PARAMS /
    # ANALYSIS_ARRAY_PARAMS) or the fitted-params toggle, since those cells
    # depend on it -- so a live edit here survives changes made elsewhere in
    # the notebook instead of being silently reset back to the catalog
    # baseline. Also doubles as the save/restore format for the scenario
    # config upload/download below.
    get_scenario_state, set_scenario_state = mo.state({})
    get_scenario_restore_error, set_scenario_restore_error = mo.state(None)
    return (
        get_scenario_state, set_scenario_state,
        get_scenario_restore_error, set_scenario_restore_error,
    )


@app.cell
def _analysis_scenario_config_upload_ui(mo, set_scenario_state, set_scenario_restore_error, json):
    def _on_scenario_config_upload(_files):
        _files = _files or ()
        if not _files:
            return
        try:
            _raw = json.loads(_files[0].contents.decode())
            if not isinstance(_raw, dict):
                raise ValueError("expected a JSON object")
        except Exception as _exc:
            set_scenario_restore_error(str(_exc))
            return
        # Restoring replaces the whole scenario design (names, overrides,
        # subpop selections, dose multipliers) rather than merging it, so a
        # restore always reflects exactly the uploaded file -- same as the
        # Fitting tab's "Restore a saved configuration".
        set_scenario_state(_raw)
        set_scenario_restore_error(None)

    analysis_scenario_config_upload = mo.ui.file(
        label="Restore a saved scenario config (JSON)",
        filetypes=[".json"],
        multiple=False,
        on_change=_on_scenario_config_upload,
    )
    return (analysis_scenario_config_upload,)


@app.cell
def _analysis_scenario_config_download_ui(mo, get_scenario_state, json):
    analysis_scenario_config_download = mo.download(
        data=json.dumps(get_scenario_state(), indent=2).encode(),
        filename="scenario_config.json",
        mimetype="application/json",
        label="Download current scenario config",
    )
    return (analysis_scenario_config_download,)


@app.cell
def _analysis_scenario_controls(
    mo, ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, get_scenario_state, set_scenario_state,
):
    _MAX_SC = 5
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _st = get_scenario_state()

    def _cb(key):
        def _inner(v):
            set_scenario_state({**get_scenario_state(), key: v})
        return _inner

    analysis_n_scenarios = mo.ui.number(
        start=1, stop=5, step=1,
        value=_st.get("n_scenarios", 2),
        label="Number of scenarios",
        on_change=_cb("n_scenarios"),
    )
    analysis_scenario_names = mo.ui.array([
        mo.ui.text(
            value=_st.get(f"name::{j}", f"Scenario {j + 1}"),
            label=f"Name {j + 1}",
            on_change=_cb(f"name::{j}"),
        )
        for j in range(_MAX_SC)
    ])

    def _make_scalar_input(pname):
        _base = float(ANALYSIS_SCALAR_PARAMS.get(pname, 1.0))
        _stop = max(10.0, _base * 20) if _base > 0 else 10.0
        return [
            mo.ui.number(
                start=0.0, stop=_stop, step=None,
                value=_st.get(f"scalar::{pname}::{j}", _base),
                on_change=_cb(f"scalar::{pname}::{j}"),
            )
            for j in range(_MAX_SC)
        ]

    analysis_scenario_scalar_inputs = mo.ui.array([
        mo.ui.array(_make_scalar_input(pname))
        for pname in _scalar_names
    ]) if _scalar_names else mo.ui.array([])

    def _make_array_input(pname):
        return [
            mo.ui.number(
                start=0.0, stop=10.0, step=None,
                value=_st.get(f"array::{pname}::{j}", 1.0),
                on_change=_cb(f"array::{pname}::{j}"),
            )
            for j in range(_MAX_SC)
        ]

    analysis_scenario_array_scales = mo.ui.array([
        mo.ui.array(_make_array_input(pname))
        for pname in _array_names
    ]) if _array_names else mo.ui.array([])

    return (
        analysis_n_scenarios, analysis_scenario_names,
        analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    )


@app.cell
def _analysis_array_age_scale_controls(
    mo, ANALYSIS_ARRAY_PARAMS, num_age_groups, age_groups, param_grid_columns,
    get_scenario_state, set_scenario_state,
):
    # Per-array-param opt-in: scale each age group by its OWN factor instead of
    # one uniform factor for the whole array (e.g. VE sensitivity presets that
    # scale `vax_susceptibility`/`IV_to_H_prop` differently per age group).
    # Off by default so existing scenarios (uniform ×scale) are unaffected.
    _MAX_SC = 5
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _age_labels = param_grid_columns(age_groups, num_age_groups)
    _st = get_scenario_state()

    def _cb(key):
        def _inner(v):
            set_scenario_state({**get_scenario_state(), key: v})
        return _inner

    analysis_array_age_scale_toggles = mo.ui.array([
        mo.ui.switch(
            label=f"Scale `{_pn}` by age group",
            value=_st.get(f"age_scale_toggle::{_pn}", False),
            on_change=_cb(f"age_scale_toggle::{_pn}"),
        )
        for _pn in _array_names
    ]) if _array_names else mo.ui.array([])

    analysis_array_age_scale_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.array([
                mo.ui.number(
                    start=0.0, stop=10.0, step=None,
                    value=_st.get(f"age_scale::{_pn}::{_a}::{_j}", 1.0),
                    label=_age_labels[_a],
                    on_change=_cb(f"age_scale::{_pn}::{_a}::{_j}"),
                )
                for _j in range(_MAX_SC)
            ])
            for _a in range(num_age_groups)
        ])
        for _pn in _array_names
    ]) if _array_names else mo.ui.array([])

    return analysis_array_age_scale_toggles, analysis_array_age_scale_inputs


@app.cell
def _analysis_dose_mult_controls(
    mo, num_age_groups, config_dict, is_metapop, analysis_sp_names,
    age_groups, param_grid_columns, get_scenario_state, set_scenario_state,
):
    # Per-scenario, per-age-group multiplier on a `scheduled_exact` transition's
    # daily schedule (`daily_vaccines_df`) -- e.g. 0 to zero out an age group (a
    # no-dose / single-age-only scenario), or a scalar to scale toward a
    # coverage target. This is a SCHEDULE override, not a config["params"]
    # entry, so it doesn't fit the scalar/array param catalog above and needs
    # its own control. Despite the "vaccine_schedule"/`daily_vaccines_df`
    # naming (fixed by the model config schema), `scheduled_exact` isn't
    # vaccine-specific -- it can back any deterministic scheduled transfer
    # (antiviral prophylaxis courses, etc.), so the UI below is worded
    # generically ("scheduled doses") rather than assuming vaccination.
    _MAX_SC = 5
    _dose_schedule_names = [
        (_s or {}).get("name") for _s in config_dict.get("schedules", []) or []
        if (_s or {}).get("schedule_template") == "vaccine_schedule"
    ]
    analysis_dose_mult_enabled = bool(_dose_schedule_names)
    _age_labels = param_grid_columns(age_groups, num_age_groups)

    # Surface which schedule(s)/file(s) this actually scales, since the config
    # schema's "vaccine_schedule"/`daily_vaccines_df` naming doesn't tell the
    # user what CSV backs it (or that it might not be vaccination at all).
    if is_metapop:
        _sp_opts = list(analysis_sp_names) if analysis_sp_names else []
        if _sp_opts:
            _file_desc = ", ".join(f"`vaccines_{_sp}.csv`" for _sp in _sp_opts) + " in the metapop folder"
        else:
            _file_desc = "`vaccines_<subpop>.csv` files in the metapop folder"
    else:
        _vax_csv = (config_dict.get("input_files", {}) or {}).get("vaccines_csv")
        _file_desc = f"`{_vax_csv}`" if _vax_csv else "a constant value (no CSV configured)"
    analysis_dose_schedule_desc = (
        f"schedule(s) {', '.join(f'`{_n}`' for _n in _dose_schedule_names)} → {_file_desc}"
        if _dose_schedule_names else ""
    )

    _st = get_scenario_state()

    def _cb(key):
        def _inner(v):
            set_scenario_state({**get_scenario_state(), key: v})
        return _inner

    analysis_dose_mult_toggle = mo.ui.switch(
        label="Scale scheduled doses per scenario (by age group)",
        value=_st.get("dose_toggle", False),
        on_change=_cb("dose_toggle"),
    )
    analysis_dose_mult_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.number(
                start=0.0, stop=10.0, step=None,
                value=_st.get(f"dose::{_j}::{_a}", 1.0),
                label=_age_labels[_a],
                on_change=_cb(f"dose::{_j}::{_a}"),
            )
            for _a in range(num_age_groups)
        ])
        for _j in range(_MAX_SC)
    ])

    # Metapop only: whether the scenario's dose scaling applies identically to
    # every subpop (the grid above) or is set independently per subpop.
    analysis_dose_mult_per_subpop_toggle = mo.ui.switch(
        label="Scale independently per subpopulation (instead of the same scaling everywhere)",
        value=_st.get("dose_per_subpop_toggle", False),
        on_change=_cb("dose_per_subpop_toggle"),
    ) if is_metapop else None
    analysis_dose_mult_subpop_inputs = mo.ui.array([
        mo.ui.array([
            mo.ui.array([
                mo.ui.number(
                    start=0.0, stop=10.0, step=None,
                    value=_st.get(f"dose_subpop::{_sp}::{_j}::{_a}", 1.0),
                    label=_age_labels[_a],
                    on_change=_cb(f"dose_subpop::{_sp}::{_j}::{_a}"),
                )
                for _a in range(num_age_groups)
            ])
            for _j in range(_MAX_SC)
        ])
        for _sp in _sp_opts
    ]) if is_metapop and _sp_opts else mo.ui.array([])

    return (
        analysis_dose_mult_enabled, analysis_dose_mult_toggle, analysis_dose_mult_inputs,
        analysis_dose_mult_per_subpop_toggle, analysis_dose_mult_subpop_inputs,
        analysis_dose_schedule_desc,
    )


@app.cell
def _analysis_param_subpop_controls(
    mo, is_metapop, analysis_sp_names,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, ANALYSIS_SCALAR_RANGES,
    analysis_scalar_param_sel, analysis_array_param_sel, analysis_param_type,
    get_scenario_state, set_scenario_state,
):
    _MAX_SC = 5
    _MAX_SENS = 6
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _sp_opts = list(analysis_sp_names) if analysis_sp_names else []
    _st = get_scenario_state()

    def _cb(key):
        def _inner(v):
            set_scenario_state({**get_scenario_state(), key: v})
        return _inner

    # Scenario sub-tab: per-param subpop multiselects
    analysis_scalar_subpop_sels = mo.ui.array([
        mo.ui.multiselect(
            options=_sp_opts,
            value=_st.get(f"scalar_subpop_sel::{pname}", []),
            label=f"Subpops for {pname}",
            on_change=_cb(f"scalar_subpop_sel::{pname}"),
        )
        for pname in _scalar_names
    ]) if is_metapop and _scalar_names and _sp_opts else mo.ui.array([])

    analysis_array_subpop_sels = mo.ui.array([
        mo.ui.multiselect(
            options=_sp_opts,
            value=_st.get(f"array_subpop_sel::{pname}", []),
            label=f"Subpops for {pname}",
            on_change=_cb(f"array_subpop_sel::{pname}"),
        )
        for pname in _array_names
    ]) if is_metapop and _array_names and _sp_opts else mo.ui.array([])

    # Scenario sub-tab: per-param × per-subpop × per-scenario number inputs [param][subpop][scenario]
    def _make_sp_scalar_col(pname, sp):
        _base = float(ANALYSIS_SCALAR_PARAMS.get(pname, 1.0))
        _lo, _hi, _step = ANALYSIS_SCALAR_RANGES.get(pname, (0.0, 10.0, 0.001))
        return mo.ui.array([
            mo.ui.number(
                start=_lo, stop=_hi, step=None,
                value=_st.get(f"scalar_subpop::{pname}::{sp}::{j}", _base),
                on_change=_cb(f"scalar_subpop::{pname}::{sp}::{j}"),
            )
            for j in range(_MAX_SC)
        ])

    analysis_scalar_subpop_inputs = mo.ui.array([
        mo.ui.array([_make_sp_scalar_col(pname, sp) for sp in _sp_opts])
        for pname in _scalar_names
    ]) if is_metapop and _scalar_names and _sp_opts else mo.ui.array([])

    def _make_sp_array_col(pname, sp):
        return mo.ui.array([
            mo.ui.number(
                start=0.0, stop=10.0, step=None,
                value=_st.get(f"array_subpop::{pname}::{sp}::{j}", 1.0),
                on_change=_cb(f"array_subpop::{pname}::{sp}::{j}"),
            )
            for j in range(_MAX_SC)
        ])

    analysis_array_subpop_scales = mo.ui.array([
        mo.ui.array([_make_sp_array_col(pname, sp) for sp in _sp_opts])
        for pname in _array_names
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
    analysis_n_reps = mo.ui.number(value=1, start=1, stop=1000, step=1, label="Replicates per scenario")
    analysis_timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    analysis_stochastic = mo.ui.switch(label="Stochastic", value=False)
    # Where the spread between replicates comes from. "Transitions only" is the
    # historical behaviour (every replicate uses the single best fitted param
    # set, differing only in the transition RNG stream); the other two options
    # draw parameter sets from the fit's accepted/posterior samples and spread
    # the replicates across them — "Sampled parameters only" keeps transitions
    # deterministic (mirrors the fitting tab's per-set trajectories, just run
    # forward as scenarios/forecasts), while "Sampled parameters + transitions"
    # adds transition RNG noise on top. Only meaningful with the replicate
    # count enabled by **Stochastic** and a fit carrying more than one
    # parameter set — see _analysis_display.
    analysis_uncertainty_source = mo.ui.radio(
        options=[
            "Transitions only",
            "Sampled parameters only",
            "Sampled parameters + transitions",
        ],
        value="Transitions only",
        label="Uncertainty source",
    )
    analysis_n_param_sets = mo.ui.number(
        value=10, start=1, stop=1000, step=1, label="Sampled parameter sets",
    )
    analysis_run_button = mo.ui.run_button(label="Run analysis")
    return (
        analysis_subpop_selector, analysis_age_selector,
        analysis_sim_days, analysis_n_reps, analysis_timesteps, analysis_stochastic,
        analysis_uncertainty_source, analysis_n_param_sets, analysis_run_button,
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
    # Default to the first three compartments only (fewer if the model has
    # fewer): checking everything puts every compartment and transition
    # variable on one axis, which is unreadable for anything but a toy model.
    _default_on = set(list(compartments)[:3])
    analysis_comp_checkboxes = mo.ui.array([
        mo.ui.checkbox(value=k in _default_on, label=k) for k in analysis_all_keys
    ])
    return analysis_comp_checkboxes, analysis_all_keys


@app.cell
def _analysis_display(
    mo, main_tab, analysis_sub_tab,
    analysis_param_type, analysis_scalar_param_sel, analysis_array_param_sel,
    analysis_n_values, analysis_sens_sliders, ANALYSIS_NO_SWEEP,
    analysis_n_scenarios, analysis_scenario_names,
    analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    analysis_subpop_selector, analysis_age_selector,
    analysis_sim_days, analysis_n_reps, analysis_timesteps, analysis_stochastic,
    analysis_uncertainty_source, analysis_n_param_sets, analysis_run_button,
    analysis_n_param_sets_avail,
    analysis_comp_checkboxes,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS,
    is_metapop, analysis_sp_names,
    analysis_sens_subpop_sel, analysis_sens_subpop_sliders,
    analysis_scalar_subpop_sels, analysis_array_subpop_sels,
    analysis_scalar_subpop_inputs, analysis_array_subpop_scales,
    analysis_use_fitted, analysis_fitted_source, analysis_fitted_params_path, analysis_fitted_note,
    analysis_fitted_params,
    analysis_dose_mult_enabled, analysis_dose_mult_toggle, analysis_dose_mult_inputs,
    analysis_dose_mult_per_subpop_toggle, analysis_dose_mult_subpop_inputs,
    analysis_dose_schedule_desc,
    analysis_array_age_scale_toggles, analysis_array_age_scale_inputs,
    age_groups, num_age_groups, param_grid_columns,
    analysis_scenario_config_upload, analysis_scenario_config_download,
    get_scenario_restore_error,
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
    _no_sweep = _pname == ANALYSIS_NO_SWEEP
    _fmt = "scale factor × each array entry" if _is_array else "value"

    _slider_vals = list(analysis_sens_sliders.value)
    _unique_vals = list(dict.fromkeys(_slider_vals))
    _n_unique = len(_unique_vals)
    _n_total = len(_slider_vals)
    if _no_sweep:
        _sens_preview = mo.callout(
            mo.md(
                "**No parameter selected — 1 baseline scenario will run** with the "
                "current parameter values (the Model Builder values, overridden by the "
                "fitted params when ① is on). Nothing is pinned, so every parameter "
                "is free to vary across sampled fitted parameter sets.\n\n"
                "Pick a parameter above to sweep it — but note the swept parameter is "
                "then **held at its slider value in every run**, including runs using a "
                "sampled fitted set, which removes that parameter's fitted uncertainty."
            ),
            kind="info",
        )
    elif _n_unique == 1:
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
        mo.hstack(
            [analysis_param_type, _param_w] if _no_sweep
            else [analysis_param_type, _param_w, analysis_n_values],
            justify="start",
        ),
    ]
    # With no parameter selected there is nothing to slide over: the sliders,
    # per-subpop overrides and value preview all describe a sweep that isn't
    # happening, so only the "baseline only" callout is shown.
    if not _no_sweep:
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

        # Sweeping a parameter the fit estimated pins it across every sampled
        # set, so the ensemble understates uncertainty (and the sweep values
        # override the fitted one) — the mistake is easy to make silently.
        if analysis_use_fitted.value and _pname in analysis_fitted_params:
            _sens_parts.append(mo.callout(
                mo.md(
                    f"**`{_pname}` is one of the fitted parameters.** Sweeping it "
                    "overrides the fitted value and holds it fixed in every run, "
                    "including runs drawn from the fit's accepted sets — so the "
                    "ensemble carries no uncertainty for this parameter. That is the "
                    "right thing for a deliberate what-if; select "
                    f"**{ANALYSIS_NO_SWEEP}** if you only wanted the fitted baseline."
                ),
                kind="warn",
            ))

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

    _array_age_labels = param_grid_columns(age_groups, num_age_groups)
    _array_rows = []
    for _k, _pn in enumerate(_array_names):
        _age_toggle = analysis_array_age_scale_toggles[_k] if _k < len(analysis_array_age_scale_toggles) else None
        if _age_toggle is not None:
            _array_rows.append(mo.hstack([_age_toggle], justify="start"))
        _by_age = bool(_age_toggle is not None and _age_toggle.value)
        if _by_age:
            # One scale factor per age group per scenario, e.g. VE_SCENARIOS-style
            # presets that scale `vax_susceptibility`/`IV_to_H_prop` differently by
            # age. Per-subpop overrides aren't offered on top of this (metapop ×
            # per-age scaling would need a 4-D grid) — use the uniform ×scale above
            # for per-subpop scenarios instead.
            for _a in range(num_age_groups):
                _age_row_items = [mo.md(f"`{_pn}` [{_array_age_labels[_a]}] ×scale")] + [
                    analysis_array_age_scale_inputs[_k][_a][j] for j in range(_n_sc)
                ]
                _array_rows.append(mo.hstack(_age_row_items, justify="start"))
            continue
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

    if analysis_dose_mult_enabled:
        _scen_body.append(mo.md(
            "*Scheduled doses (from a `scheduled_exact` transition's schedule — "
            "vaccine doses, antiviral courses, etc.; schedule scaling, not a "
            "model param — **1 leaves the original schedule untouched**; 0 "
            "zeroes out an age group, e.g. for a no-dose or single-age-only "
            "scenario):*"
        ))
        if analysis_dose_schedule_desc:
            _scen_body.append(mo.md(f"&nbsp;&nbsp;↳ scaling {analysis_dose_schedule_desc}"))
        _scen_body.append(analysis_dose_mult_toggle)
        if analysis_dose_mult_toggle.value:
            _dose_age_labels = param_grid_columns(age_groups, num_age_groups)
            _per_sp_active = bool(
                is_metapop and analysis_dose_mult_per_subpop_toggle is not None
                and analysis_dose_mult_per_subpop_toggle.value
            )
            if is_metapop and analysis_dose_mult_per_subpop_toggle is not None:
                _scen_body.append(analysis_dose_mult_per_subpop_toggle)
            if _per_sp_active and analysis_sp_names:
                for _si, _sp in enumerate(analysis_sp_names):
                    if _si >= len(analysis_dose_mult_subpop_inputs):
                        break
                    _scen_body.append(mo.md(f"↳ **{_sp}**:"))
                    _sp_header = mo.hstack(
                        [mo.md("**Age group**")] + [analysis_scenario_names[j] for j in range(_n_sc)],
                        justify="start",
                    )
                    _scen_body.append(_sp_header)
                    for _a in range(num_age_groups):
                        _label = _dose_age_labels[_a]
                        _scen_body.append(mo.hstack(
                            [mo.md(f"  `{_label}`")] + [
                                analysis_dose_mult_subpop_inputs[_si][j][_a] for j in range(_n_sc)
                            ],
                            justify="start",
                        ))
            else:
                _dose_header = mo.hstack(
                    [mo.md("**Age group**")] + [analysis_scenario_names[j] for j in range(_n_sc)],
                    justify="start",
                )
                _dose_rows = [_dose_header]
                for _a in range(num_age_groups):
                    _label = _dose_age_labels[_a]
                    _dose_rows.append(mo.hstack(
                        [mo.md(f"`{_label}`")] + [analysis_dose_mult_inputs[j][_a] for j in range(_n_sc)],
                        justify="start",
                    ))
                _scen_body += _dose_rows
    _scen_ui = mo.vstack(_scen_body)

    # --- Uncertainty-source controls (Run settings) ---
    # Only offered when there is parameter uncertainty to propagate (a fit with
    # >1 accepted set) AND stochastic transitions are on — deterministic runs
    # always collapse to a single run of the best parameter set.
    _unc_parts = []
    # "Sampled parameters only" runs each drawn set exactly once (transitions
    # are deterministic, so repeating a set just duplicates its trajectory) —
    # the replicate count is meaningless there and the input is hidden below.
    _param_only_ui = (
        analysis_n_param_sets_avail > 1
        and analysis_stochastic.value
        and analysis_uncertainty_source.value == "Sampled parameters only"
    )
    if analysis_n_param_sets_avail > 1 and analysis_stochastic.value:
        _unc_parts.append(analysis_uncertainty_source)
        if analysis_uncertainty_source.value in (
            "Sampled parameters only", "Sampled parameters + transitions",
        ):
            _unc_parts.append(analysis_n_param_sets)
            if _param_only_ui:
                _k_ui = min(int(analysis_n_param_sets.value), analysis_n_param_sets_avail)
                _unc_parts.append(mo.callout(
                    mo.md(
                        f"**{_k_ui} parameter set(s)** drawn at random (without replacement) "
                        f"from the **{analysis_n_param_sets_avail}** accepted set(s) × 1 "
                        f"deterministic run each = **{_k_ui} run(s) per scenario**. "
                        "Transitions run **deterministically**, so a set repeated across "
                        "replicates would give an identical trajectory — *Replicates per "
                        "scenario* is ignored in this mode; raise **Sampled parameter sets** "
                        "to cover more of the posterior."
                    ),
                    kind="success",
                ))
            else:
                _reps_ui = int(analysis_n_reps.value)
                _k_ui = min(int(analysis_n_param_sets.value), _reps_ui, analysis_n_param_sets_avail)
                _base_ui, _extra_ui = divmod(_reps_ui, _k_ui)
                _spread = (
                    f"{_base_ui} stochastic rep(s) each"
                    if not _extra_ui
                    else f"{_base_ui}–{_base_ui + 1} stochastic rep(s) each"
                )
                _unc_parts.append(mo.callout(
                    mo.md(
                        f"**{_k_ui} parameter set(s)** drawn at random (without replacement) from the "
                        f"**{analysis_n_param_sets_avail}** accepted set(s) × {_spread} = "
                        f"**{_reps_ui} run(s) per scenario**."
                    ),
                    kind="success",
                ))
                if int(analysis_n_param_sets.value) > _reps_ui:
                    _unc_parts.append(mo.callout(
                        mo.md(
                            f"**Sampled parameter sets ({int(analysis_n_param_sets.value)}) exceeds "
                            f"replicates per scenario ({_reps_ui})** — clamped to {_k_ui}. Raise the "
                            "replicate count to simulate more of the posterior."
                        ),
                        kind="warn",
                    ))
            if int(analysis_n_param_sets.value) > analysis_n_param_sets_avail:
                _unc_parts.append(mo.callout(
                    mo.md(
                        f"**Only {analysis_n_param_sets_avail} accepted set(s) available** — "
                        f"clamped from {int(analysis_n_param_sets.value)}."
                    ),
                    kind="warn",
                ))
    elif analysis_n_param_sets_avail > 1:
        _unc_parts.append(mo.md(
            f"*{analysis_n_param_sets_avail} fitted parameter set(s) available — turn on "
            "**Stochastic** to sample from them.*"
        ))

    _tab_body = {"Sensitivity": _sens_ui, "Scenario": _scen_ui}
    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Analysis</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Sweep parameters or compare '
            "scenarios.</div>"
        ),
        section_card(
            step_header("①", "Fitted parameters",
                        "Optionally use the fit from the Fitting tab, or a saved fit JSON, "
                        "as the baseline for sensitivity/scenario analysis.",
                        accent=_ACC),
            mo.vstack([
                analysis_use_fitted,
                mo.vstack([
                    analysis_fitted_source,
                    analysis_fitted_params_path if analysis_fitted_source.value == "Upload JSON file" else mo.md(""),
                    analysis_fitted_note,
                ]) if analysis_use_fitted.value else mo.md(""),
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Design",
                        "Define a sensitivity sweep or a set of scenarios.",
                        accent=_ACC),
            mo.vstack([
                mo.hstack(
                    [analysis_scenario_config_upload, analysis_scenario_config_download],
                    justify="start",
                ),
                mo.callout(
                    mo.md(f"**Could not restore scenario config:** {get_scenario_restore_error()}"),
                    kind="warn",
                ) if get_scenario_restore_error() else mo.md(""),
                analysis_sub_tab,
                _tab_body.get(analysis_sub_tab.value, mo.md("")),
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("③", "Run settings",
                        "Horizon, replicates, slices, and which compartments to display.",
                        accent=_ACC),
            mo.vstack([
                mo.hstack(
                    [analysis_sim_days, analysis_timesteps] if _param_only_ui
                    else [analysis_sim_days, analysis_n_reps, analysis_timesteps],
                    justify="start",
                ),
                mo.hstack([
                    analysis_stochastic,
                    mo.md("*Ignored — using 1 replicate of the best parameter set.*")
                    if not analysis_stochastic.value else mo.md(""),
                ], justify="start"),
                mo.vstack(_unc_parts) if _unc_parts else mo.md(""),
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
    analysis_sens_sliders, ANALYSIS_NO_SWEEP,
    analysis_n_scenarios, analysis_scenario_names,
    analysis_scenario_scalar_inputs, analysis_scenario_array_scales,
    ANALYSIS_SCALAR_PARAMS, ANALYSIS_ARRAY_PARAMS, np,
    is_metapop, analysis_sp_names,
    analysis_sens_subpop_sel, analysis_sens_subpop_sliders,
    analysis_scalar_subpop_sels, analysis_array_subpop_sels,
    analysis_scalar_subpop_inputs, analysis_array_subpop_scales,
    analysis_dose_mult_enabled, analysis_dose_mult_toggle, analysis_dose_mult_inputs, num_age_groups,
    analysis_dose_mult_per_subpop_toggle, analysis_dose_mult_subpop_inputs,
    analysis_array_age_scale_toggles, analysis_array_age_scale_inputs,
):
    _scalar_names = list(ANALYSIS_SCALAR_PARAMS.keys())
    _array_names = list(ANALYSIS_ARRAY_PARAMS.keys())
    _use_subpop = is_metapop and bool(analysis_sp_names)
    # Each entry is (name, global_overrides, per_subpop_overrides, designed,
    # dose_multiplier, dose_multiplier_per_subpop). `designed` names the params
    # this scenario deliberately sets — the swept param in Sensitivity, or a
    # cell edited away from the catalog baseline in Scenario. _run_analysis
    # lets a sampled fitted param set overwrite every *other* param, so
    # parameter uncertainty is injected without silently undoing the design.
    # (Both grids are pre-filled from the catalog baseline, so inequality with
    # it is the edit signal.)
    # `dose_multiplier` is a per-age-group scale on a `scheduled_exact`
    # transition's daily schedule (None = unscaled) — a schedule override, not
    # a params entry, so it's carried separately from `overrides`/`designed`.
    # `dose_multiplier_per_subpop` (metapop only) is a list of per-age-group
    # vectors indexed by subpop, overriding `dose_multiplier` for subpops set
    # independently. Both only ever set in the Scenario sub-tab (Sensitivity
    # sweeps a param, not the dose schedule).
    analysis_scenarios = []
    _dose_mult_active = analysis_dose_mult_enabled and analysis_dose_mult_toggle.value
    _dose_mult_per_sp_active = bool(
        _dose_mult_active and is_metapop and analysis_dose_mult_per_subpop_toggle is not None
        and analysis_dose_mult_per_subpop_toggle.value
    )

    def _make_per_subpop_list(sp_names, sp_to_override_dict):
        if not sp_to_override_dict:
            return None
        _result = [sp_to_override_dict.get(_sp) for _sp in sp_names]
        return _result if any(_r for _r in _result) else None

    if analysis_sub_tab.value == "Sensitivity":
        _is_array = analysis_param_type.value == "Array (scale factor)"
        _sel_pname = analysis_array_param_sel.value if _is_array else analysis_scalar_param_sel.value
        if _sel_pname == ANALYSIS_NO_SWEEP:
            # No parameter selected: one baseline run at the current values, with
            # an EMPTY designed set — nothing is shielded, so fitted/sampled
            # parameter sets apply in full (see _apply_pset in _run_analysis).
            analysis_scenarios.append((
                "baseline", {**ANALYSIS_SCALAR_PARAMS, **ANALYSIS_ARRAY_PARAMS}, None, set(), None, None,
            ))
        elif _is_array:
            _pname = analysis_array_param_sel.value
            _base_arr = np.asarray(ANALYSIS_ARRAY_PARAMS.get(_pname, [1.0]))
            for _i, _v in enumerate(list(dict.fromkeys(analysis_sens_sliders.value))):
                _global_ov = {**ANALYSIS_SCALAR_PARAMS, **ANALYSIS_ARRAY_PARAMS}
                _global_ov[_pname] = (_base_arr * _v).tolist()
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
                analysis_scenarios.append((f"{_pname} ×{_v:.3g}", _global_ov, _per_subpop, {_pname}, None, None))
        else:
            _pname = analysis_scalar_param_sel.value
            for _i, _v in enumerate(list(dict.fromkeys(analysis_sens_sliders.value))):
                _global_ov = {**ANALYSIS_SCALAR_PARAMS, **ANALYSIS_ARRAY_PARAMS}
                _global_ov[_pname] = float(_v)
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
                analysis_scenarios.append((f"{_pname}={_v:.4g}", _global_ov, _per_subpop, {_pname}, None, None))
    else:
        _n = int(analysis_n_scenarios.value)
        for j in range(_n):
            _name = analysis_scenario_names.value[j].strip() or f"Scenario {j + 1}"
            _overrides = {}
            _designed = set()
            for _i, _pn in enumerate(_scalar_names):
                _val = float(analysis_scenario_scalar_inputs.value[_i][j])
                _overrides[_pn] = _val
                if _val != float(ANALYSIS_SCALAR_PARAMS[_pn]):
                    _designed.add(_pn)
            for _k, _pn in enumerate(_array_names):
                _base = np.asarray(ANALYSIS_ARRAY_PARAMS[_pn])
                _by_age = (
                    _k < len(analysis_array_age_scale_toggles)
                    and analysis_array_age_scale_toggles.value[_k]
                )
                if _by_age:
                    _age_scales = np.array([
                        float(analysis_array_age_scale_inputs.value[_k][_a][j])
                        for _a in range(num_age_groups)
                    ])
                    _overrides[_pn] = (_base * _age_scales[:, None]).tolist()
                    if np.any(_age_scales != 1.0):
                        _designed.add(_pn)
                else:
                    _scale = float(analysis_scenario_array_scales.value[_k][j])
                    _overrides[_pn] = (_base * _scale).tolist()
                    if _scale != 1.0:
                        _designed.add(_pn)
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
            _dose_mult = None
            _dose_mult_per_subpop = None
            if _dose_mult_active:
                _mult_vec = [float(analysis_dose_mult_inputs.value[j][_a]) for _a in range(num_age_groups)]
                if any(_v != 1.0 for _v in _mult_vec):
                    _dose_mult = _mult_vec
                if _dose_mult_per_sp_active and analysis_sp_names:
                    _sp_vecs = []
                    for _si in range(len(analysis_sp_names)):
                        if _si >= len(analysis_dose_mult_subpop_inputs):
                            _sp_vecs.append(None)
                            continue
                        _sp_vec = [
                            float(analysis_dose_mult_subpop_inputs.value[_si][j][_a])
                            for _a in range(num_age_groups)
                        ]
                        _sp_vecs.append(_sp_vec if any(_v != 1.0 for _v in _sp_vec) else None)
                    if any(_v is not None for _v in _sp_vecs):
                        _dose_mult_per_subpop = _sp_vecs
            analysis_scenarios.append((_name, _overrides, _per_subpop, _designed, _dose_mult, _dose_mult_per_subpop))

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
    analysis_uncertainty_source, analysis_n_param_sets, analysis_param_sets,
    config_dict, compartments, is_metapop,
    metapop_folder_input, metapop_travel_config,
    transition_vars_input,
    build_compartment_init, start_date_input, rng_seed,
    make_single_pop_metapop, make_metapop_from_folder,
    set_analysis_results, loaded_schedule_dfs,
    analysis_use_fitted, analysis_fitted_params,
    analysis_fitted_tv_increments, analysis_fitted_tv_spacing, analysis_fitted_num_days,
    np, mo, build_scalar_array, SimpleNamespace,
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

    # --- Uncertainty source: which parameter set each replicate runs with ---
    # "Transitions only" (and every deterministic run) keeps the historical
    # behaviour: one param set — the fitted best, already folded into the
    # scenario overrides via the param catalog — with the replicates differing
    # only in the transition RNG stream. Otherwise draw `_k` sets at random
    # (without replacement) from the fit's accepted/posterior samples and
    # spread the replicates evenly across them, so the ensemble carries both
    # parameter and transition uncertainty. `_run_schedule` is a list of
    # (param-set index or None, rep index within that set), one entry per run.
    # "Sampled parameters only" draws the same way but keeps the per-run
    # transition engine deterministic (see _stoch_run below) — the spread
    # across runs is then parameter uncertainty alone, mirroring how the
    # fitting tab's accepted sets each produce one deterministic trajectory.
    _param_only = analysis_uncertainty_source.value == "Sampled parameters only"
    _use_psets = (
        _stoch
        and analysis_uncertainty_source.value in (
            "Sampled parameters only", "Sampled parameters + transitions",
        )
        and len(analysis_param_sets) > 1
    )
    _stoch_run = _stoch and not _param_only
    if _use_psets:
        _rng_sched = np.random.default_rng(_seed_b)
        # With deterministic transitions a param set repeated across replicates
        # reproduces the exact same trajectory, so "Sampled parameters only"
        # runs each drawn set once and ignores the replicate count entirely —
        # the ensemble size is the number of sampled sets (see _analysis_display,
        # which hides the replicate input in that mode).
        _k = min(
            int(analysis_n_param_sets.value),
            len(analysis_param_sets),
            *(() if _param_only else (_n_reps,)),
        )
        _sel = _rng_sched.choice(len(analysis_param_sets), size=_k, replace=False)
        _psets = [analysis_param_sets[int(_i)] for _i in _sel]
        if _param_only:
            _run_schedule = [(_i, 0) for _i in range(_k)]
        else:
            _base_r, _extra_r = divmod(_n_reps, _k)
            _run_schedule = [(_i, _r) for _i in range(_k) for _r in range(_base_r)]
            if _extra_r:
                _run_schedule += [
                    (int(_i), _base_r)
                    for _i in _rng_sched.choice(_k, size=_extra_r, replace=False)
                ]
    else:
        _psets = []
        _run_schedule = [(None, _r) for _r in range(_n_reps)]

    _ci_best = None
    _sim_config = config_dict
    # The uploaded schedule CSVs (humidity / school-work calendar / mobility /
    # vaccination) must be passed through exactly as the Fitting and Forecast
    # tabs do — omitting them silently substitutes flat constant schedules
    # (humidity 0, no vaccination), which is a different model from the one that
    # was fitted and inflates the epidemic by an order of magnitude.
    _schedule_dfs_best = loaded_schedule_dfs
    _ci_unscaled = None
    # NOTE: marimo mangles cell-private (underscore) names, but only rewrites a
    # reference inside a nested function if the name is already bound at the
    # cell's top level *earlier in the source*. So every cell-private name a
    # closure below reads must be pre-bound here, before the `def` — otherwise
    # the reference stays unmangled and raises NameError at call time.
    _pop_arr = None

    def _scaled_ci(_pset):
        # Apply a param set's seed_scale_<comp> entries to the unscaled initial
        # conditions. Single-population only — the metapop path builds its own
        # initial conditions per subpop inside make_metapop_from_folder.
        _seed_scales = {
            _sk[len("seed_scale_"):]: float(_sv)
            for _sk, _sv in _pset.items()
            if _sk.startswith("seed_scale_") and _sk[len("seed_scale_"):] in compartments
        }
        if not _seed_scales:
            return _ci_unscaled
        from generic_core.fitting import _scale_compartment_init
        _A, _R = _pop_arr.shape
        return _scale_compartment_init(_ci_unscaled, _seed_scales, compartments, _A, _R)

    if not is_metapop:
        # Initial conditions from the Step 6 tables via config_dict.
        _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
        _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((1, 1))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in compartments
        }
        _ci_unscaled, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)
        _ci_best = _ci_unscaled

        # Fitted seed_scale_<comp> params scale the seeded compartments (mirrors
        # the fitting-time scaling in generic_core.fitting._scale_compartment_init;
        # they are not config["params"] entries, so param_overrides can't apply them).
        if analysis_use_fitted.value and analysis_fitted_params:
            _ci_best = _scaled_ci(analysis_fitted_params)

    # Reconstruct the fitted time-varying transmission multiplier m(t) from its
    # log-increments and wire it into every force_of_infection transition via a
    # 'transmission_multiplier' schedule (mirrors
    # generic_core.fitting._inject_tv_transmission). Held flat at its last
    # fitted value for any simulated days beyond the fit period. In the metapop
    # case, the same single-population-fitted m(t) trajectory is broadcast
    # uniformly to every subpopulation (see make_metapop_from_folder's
    # transmission_multiplier_df param).
    _has_mt = False
    # Imported before the def below (not inside the `if` that uses them) so the
    # closure's references to these cell-private names get mangled — see the
    # marimo note above.
    from generic_core.fitting import (
        _inject_tv_transmission, _tv_knot_days,
        build_transmission_multiplier_array as _build_transmission_multiplier_array,
    )
    import pandas as _pd

    def _schedule_dfs_for(_increments):
        # Build the schedule bundle carrying this param set's m(t). Adds m(t)
        # *alongside* the uploaded schedules rather than replacing them — m(t)
        # is an extra multiplier on the force of infection, not a substitute
        # for humidity/calendar/mobility/vaccination.
        _fit_days = int(analysis_fitted_num_days)
        _knots = _tv_knot_days(_fit_days, int(analysis_fitted_tv_spacing))
        _m_fit = _build_transmission_multiplier_array(list(_increments), _knots, _fit_days)
        if _num_days <= _fit_days:
            _m_full = _m_fit[:_num_days]
        else:
            _m_full = np.concatenate([_m_fit, np.full(_num_days - _fit_days, _m_fit[-1])])
        _dates = _pd.date_range(start=_start, periods=_num_days, freq="D").date
        return SimpleNamespace(
            **{
                _f: getattr(loaded_schedule_dfs, _f, None)
                for _f in ("absolute_humidity_df", "school_work_calendar_df",
                           "mobility_df", "daily_vaccines_df")
            },
            transmission_multiplier_df=_pd.DataFrame(
                {"date": _dates, "transmission_multiplier": _m_full}
            ),
        )

    if analysis_use_fitted.value and analysis_fitted_tv_increments:
        _tv_cfg, _n_foi = _inject_tv_transmission(_sim_config)
        if _n_foi:
            # The FOI transitions are rewired to read a 'transmission_multiplier'
            # schedule once, for every run — only the per-day values differ
            # between param sets. Accepted sets come from the same fit as the
            # best set, so if the best set carries m(t) so do the others.
            _sim_config = _tv_cfg
            _has_mt = True
            _schedule_dfs_best = _schedule_dfs_for(analysis_fitted_tv_increments)

    def _pset_increments(_pset):
        return [
            _v for _, _v in sorted(
                (
                    (int(_pk[len("m_dlog_"):]), float(_pv))
                    for _pk, _pv in _pset.items()
                    if _pk.startswith("m_dlog_") and _pk[len("m_dlog_"):].isdigit()
                ),
                key=lambda _t: _t[0],
            )
        ]

    # Per-param-set initial conditions and m(t) schedules, built lazily and
    # reused across scenarios (both depend only on the param set, not on the
    # scenario overrides). _pset_idx None means "the fitted best set" — the
    # values already computed above, i.e. the pre-existing code path.
    _ci_cache = {}
    _sched_cache = {}

    def _get_ci(_pset_idx):
        if _pset_idx is None or is_metapop:
            return _ci_best
        if _pset_idx not in _ci_cache:
            _ci_cache[_pset_idx] = _scaled_ci(_psets[_pset_idx])
        return _ci_cache[_pset_idx]

    def _get_schedule_dfs(_pset_idx):
        if _pset_idx is None or not _has_mt:
            return _schedule_dfs_best
        if _pset_idx not in _sched_cache:
            _incrs = _pset_increments(_psets[_pset_idx])
            _sched_cache[_pset_idx] = (
                _schedule_dfs_for(_incrs) if _incrs else _schedule_dfs_best
            )
        return _sched_cache[_pset_idx]

    def _apply_pset(_overrides, _pset_idx, _designed):
        # Layer a sampled param set under the scenario's overrides: the set
        # supplies every fitted param the design did not deliberately set, so
        # the ensemble carries parameter uncertainty without undoing the
        # sensitivity sweep or the edited scenario cells. seed_scale_*/m_dlog_*
        # /phi are handled elsewhere (_get_ci / _get_schedule_dfs / not a model
        # param), so they never reach param_overrides.
        if _pset_idx is None:
            return _overrides
        _ov = dict(_overrides or {})
        for _pk, _pv in _psets[_pset_idx].items():
            if _pk == "phi" or _pk.startswith("m_dlog_") or _pk.startswith("seed_scale_"):
                continue
            if _pk in _designed:
                continue
            _ov[_pk] = list(_pv) if isinstance(_pv, (list, tuple)) else float(_pv)
        return _ov

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
    _spin = (
        f"Running {len(analysis_scenarios)} scenario(s) × {len(_run_schedule)} run(s)"
        + (f" across {len(_psets)} parameter set(s)..." if _use_psets else "...")
    )
    with mo.status.spinner(_spin):
        try:
            for _scen_tuple in analysis_scenarios:
                _scen_name, _overrides = _scen_tuple[0], _scen_tuple[1]
                _per_subpop = _scen_tuple[2] if len(_scen_tuple) > 2 else None
                _designed = _scen_tuple[3] if len(_scen_tuple) > 3 else set()
                _dose_mult = _scen_tuple[4] if len(_scen_tuple) > 4 else None
                _dose_mult_per_subpop = _scen_tuple[5] if len(_scen_tuple) > 5 else None
                _reps_hists = []
                # Seed by position in the run schedule, not by the rep index
                # within a param set, so every run gets a distinct RNG stream
                # once replicates are spread across sets.
                for _run_i, (_pset_idx, _rep) in enumerate(_run_schedule):
                    _run_ov = _apply_pset(_overrides, _pset_idx, _designed)
                    _run_sched = _get_schedule_dfs(_pset_idx)
                    if not is_metapop:
                        _m, _, _ = make_single_pop_metapop(
                            _sim_config, _start, _num_days, _get_ci(_pset_idx),
                            seed_offset=_run_i, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=_stoch_run, tvs=_tvs, save_daily=True,
                            param_overrides=_run_ov or None,
                            travel_config=metapop_travel_config,
                            schedule_dfs=_run_sched,
                            dose_mult=_dose_mult,
                        )
                    else:
                        _m, _ = make_metapop_from_folder(
                            metapop_folder_input.value, _sim_config, _start, _num_days, list(compartments),
                            seed_offset=_run_i, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=_stoch_run, tvs=_tvs, save_daily=True,
                            param_overrides=_run_ov or None,
                            param_overrides_per_subpop=_per_subpop,
                            travel_config=metapop_travel_config,
                            transmission_multiplier_df=getattr(
                                _run_sched, "transmission_multiplier_df", None
                            ),
                            dose_mult=_dose_mult,
                            dose_mult_per_subpop=_dose_mult_per_subpop,
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
        # Which sampled param set each replicate used (None when parameter
        # uncertainty is off) — lets downstream code group replicates by draw.
        "param_set_indices": [_i for _i, _ in _run_schedule],
        "uncertainty_source": (
            "parameters" if (_use_psets and _param_only)
            else "parameters+transitions" if _use_psets
            else ("transitions" if _stoch else "deterministic")
        ),
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

    _fig, _axes = plt.subplots(
        _n_combos, 1, figsize=(11, min(4 * _n_combos, 80)), squeeze=False,
        constrained_layout=True,
    )

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
    # Name the uncertainty source these results were produced with: the run
    # settings above can be changed without re-running, so the displayed
    # figure is not necessarily the currently-selected configuration.
    _UNC_LABEL = {
        "deterministic": "deterministic — 1 run of the best parameter set",
        "transitions": "transition RNG noise only — every run uses the best parameter set",
        "parameters": "sampled parameters only — deterministic transitions",
        "parameters+transitions": "sampled parameters + transition RNG noise",
    }
    _unc_used = analysis_results.get("uncertainty_source", "")
    _n_runs_used = len(analysis_results.get("param_set_indices", [])) or 1
    _n_sets_used = len({_i for _i in analysis_results.get("param_set_indices", []) if _i is not None})
    mo.vstack([
        mo.md("## Analysis — Compartment Histories"),
        mo.md(
            f"*Results from: **{_UNC_LABEL.get(_unc_used, _unc_used)}**, "
            f"{_n_runs_used} run(s) per scenario"
            + (f" across {_n_sets_used} parameter set(s)" if _n_sets_used else "")
            + ". Re-click **Run analysis** after changing any run setting.*"
        ),
        _fig,
    ])
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
def _analysis_metric_plot_options(mo):
    # Extra "duplicate the plots" toggles for the user-defined metric section.
    # Both are additive: the total-population / daily views are always kept.
    analysis_metric_per_age = mo.ui.checkbox(
        value=False,
        label="Also plot each age group separately (in addition to the totals above)",
    )
    analysis_metric_cumulative = mo.ui.checkbox(
        value=False,
        label="Also plot cumulative time series (in addition to daily values)",
    )
    return analysis_metric_per_age, analysis_metric_cumulative


@app.cell
def _analysis_metric_defs_show(
    mo, main_tab,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    analysis_plot_metric_sel, transition_vars_input, tv_opts,
    analysis_metric_per_age, analysis_metric_cumulative,
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
        mo.md("**Extra plot views:**"),
        analysis_metric_per_age,
        analysis_metric_cumulative,
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
    analysis_metric_per_age, analysis_metric_cumulative,
    num_age_groups, np, pd, plt, mo, main_tab,
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
    _sel_ages = list(analysis_age_selector.value or ["all ages"])
    if analysis_metric_per_age.value:
        _sel_ages = list(dict.fromkeys(
            _sel_ages + [f"Age {_a}" for _a in range(num_age_groups)]
        ))
    _kinds = ["Daily"] + (["Cumulative"] if analysis_metric_cumulative.value else [])
    _combos = [(sp, ag, k) for sp in _sel_subpops for ag in _sel_ages for k in _kinds]
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

    _fig, _axes = plt.subplots(
        _n_combos, 1, figsize=(11, min(4 * _n_combos, 80)), squeeze=False,
        constrained_layout=True,
    )
    _csv_rows = []  # long-format copy of exactly what gets drawn

    for _c_idx, (_sp, _ag, _kind) in enumerate(_combos):
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
                if _kind == "Cumulative":
                    _stacked = np.cumsum(_stacked, axis=1)
                _dates = pd.date_range(start=_start, periods=_stacked.shape[1], freq="D")
                _med = np.median(_stacked, axis=0)
                _lo = np.percentile(_stacked, 2.5, axis=0)
                _hi = np.percentile(_stacked, 97.5, axis=0)
                _ax.plot(_dates, _med, label=f"{_mname} — {_scen_name}", color=_color, linestyle=_ls)
                _ax.fill_between(_dates, _lo, _hi, color=_color, alpha=0.15)
                for _d, _m_v, _lo_v, _hi_v in zip(_dates, _med, _lo, _hi):
                    _csv_rows.append({
                        "date": _d.date().isoformat(),
                        "series": _kind.lower(),
                        "subpopulation": _sp,
                        "age_group": _ag,
                        "metric": _mname,
                        "scenario": _scen_name,
                        "median": float(_m_v),
                        "ci_lower_2.5": float(_lo_v),
                        "ci_upper_97.5": float(_hi_v),
                    })

        _ax.set_xlabel("Date")
        _ax.set_ylabel(f"{_kind} count")
        _ax.set_title(f"{_kind} metric by scenario (median + 95% CI) — {_sp} / {_ag}")
        _handles, _labels_leg = _ax.get_legend_handles_labels()
        if _handles:
            _ax.legend(_handles, _labels_leg, fontsize=8, loc="upper right")

    _fig.autofmt_xdate()
    _dl = mo.download(
        data=pd.DataFrame(_csv_rows).to_csv(index=False).encode(),
        filename="metric_timeseries.csv",
        label="Download plotted data as CSV",
    )
    mo.vstack([mo.md(f"### Analysis — {' / '.join(_kinds)} Metric by Scenario"), _fig, _dl])
    return


@app.cell
def _analysis_plot_cumulative_boxplot(
    analysis_metric_series, analysis_plot_metric_sel,
    analysis_subpop_selector, analysis_age_selector,
    analysis_metric_per_age, num_age_groups,
    np, pd, plt, mo, main_tab,
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
    _sel_ages = list(analysis_age_selector.value or ["all ages"])
    if analysis_metric_per_age.value:
        _sel_ages = list(dict.fromkeys(
            _sel_ages + [f"Age {_a}" for _a in range(num_age_groups)]
        ))
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
        constrained_layout=True,
    )
    _csv_rows = []  # per-replicate values behind each box

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
                for _r_idx, _v in enumerate(_vals):
                    _csv_rows.append({
                        "subpopulation": _sp,
                        "age_group": _ag,
                        "metric": _mname,
                        "scenario": _scen_name,
                        "replicate": _r_idx,
                        "cumulative_value": float(_v),
                    })
            _ax.boxplot(_box_data, tick_labels=_scen_names, orientation="vertical")
            _ax.axhline(0, linestyle="--", color="gray", alpha=0.4)
            _ax.set_ylabel(f"Cumulative {_mname}")
            _ax.set_title(f"Cumulative {_mname} — {_sp} / {_ag}")
            _ax.tick_params(axis="x", rotation=20)

    _dl = mo.download(
        data=pd.DataFrame(_csv_rows).to_csv(index=False).encode(),
        filename="metric_cumulative_by_replicate.csv",
        label="Download plotted data as CSV",
    )
    mo.vstack([mo.md("### Analysis — Cumulative Metric Distribution by Scenario"), _fig, _dl])
    return


@app.cell
def _analysis_plot_age_bars(
    analysis_metric_series, analysis_plot_metric_sel,
    analysis_subpop_selector,
    num_age_groups, np, pd, plt, mo, main_tab,
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
    _fig, _axes = plt.subplots(
        _n_plots, 1, figsize=(10, min(5 * _n_plots, 80)), squeeze=False,
        constrained_layout=True,
    )
    _ax_idx = 0
    _csv_rows = []  # bar heights (mean across replicates)

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
                for _lbl, _v in zip(_x_labels, _mean_vals):
                    _csv_rows.append({
                        "subpopulation": _sp,
                        "metric": _mname,
                        "scenario": _scen_name,
                        "age_group": _lbl,
                        "cumulative_mean": float(_v),
                    })

            # Subtle separator before the "Total" bar
            _ax.axvline(x=_n_ages - 0.5, color="gray", linestyle=":", alpha=0.5)
            _ax.set_xlabel("Age group")
            _ax.set_ylabel(f"Cumulative {_mname} (mean across replicates)")
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_x_labels)
            _ax.set_title(f"Age-stratified cumulative {_mname} — {_sp}")
            _ax.legend()
            _ax_idx += 1

    _dl = mo.download(
        data=pd.DataFrame(_csv_rows).to_csv(index=False).encode(),
        filename="metric_age_stratified.csv",
        label="Download plotted data as CSV",
    )
    mo.vstack([mo.md("### Analysis — Age-stratified Metric by Scenario"), _fig, _dl])
    return


# ============================================================
# Documentation tab
# ============================================================

