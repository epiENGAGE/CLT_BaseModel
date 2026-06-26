# _nb_fitting.py
# Section: Fitting tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _fit_n_targets_state(mo):
    get_n_targets, set_n_targets = mo.state(1)
    return get_n_targets, set_n_targets


@app.cell
def _fit_target_buttons(mo, get_n_targets, set_n_targets):
    add_target_btn = mo.ui.button(
        label="+ Add target",
        on_click=lambda _: set_n_targets(min(get_n_targets() + 1, 20)),
    )
    del_target_btn = mo.ui.button(
        label="− Remove target",
        on_click=lambda _: set_n_targets(max(get_n_targets() - 1, 1)),
    )
    return add_target_btn, del_target_btn


@app.cell
def _fitting_ui(
    mo, compartments, n_transitions, t_name, param_names,
    is_metapop, metapop_folder_input, json, Path,
    num_age_groups, num_risk_groups,
):
    _tvars = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    _all_tgts = list(compartments) + _tvars
    _tgt_opts = _all_tgts if _all_tgts else ["S"]

    # Subpop names for slice dropdowns
    _subpop_names_ui = []
    if is_metapop and metapop_folder_input.value.strip():
        try:
            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                _subpop_names_ui = json.load(_f).get("subpopulations", [])
        except Exception:
            _subpop_names_ui = []

    _sp_opts = ["All (sum)"] + [f"{_i}: {_nm}" for _i, _nm in enumerate(_subpop_names_ui)]
    _age_opts = ["All (sum)"] + [str(_i) for _i in range(int(num_age_groups))]
    _risk_opts = ["All (sum)"] + [str(_i) for _i in range(int(num_risk_groups))]

    fit_target_src = mo.ui.array([
        mo.ui.radio(
            options={"Upload CSV": "upload", "File path": "path"},
            value="Upload CSV",
            label="Data source",
        )
        for _ in range(20)
    ])
    fit_target_upload = mo.ui.array([
        mo.ui.file(label="Upload CSV", filetypes=[".csv"])
        for _ in range(20)
    ])
    fit_target_path = mo.ui.array([
        mo.ui.text(
            label="CSV file path", placeholder="~/data/observed.csv", full_width=True,
        )
        for _ in range(20)
    ])
    fit_target_vars = mo.ui.array([
        mo.ui.multiselect(
            options={t: t for t in _tgt_opts},
            value=[_tgt_opts[0]],
            label="Variables (summed)",
        )
        for _ in range(20)
    ])
    fit_target_mode = mo.ui.array([
        mo.ui.radio(
            options={"Timeseries": "ts", "Scalar total": "scalar", "Proportions": "proportion"},
            value="Timeseries",
            label="Observed data type",
        )
        for _ in range(20)
    ])
    fit_target_weight = mo.ui.array([
        mo.ui.number(value=1.0, start=0.0, stop=1000.0, step=None, label="Weight λ")
        for _ in range(20)
    ])
    fit_target_subpop = mo.ui.array([
        mo.ui.dropdown(options=_sp_opts, value=_sp_opts[0], label="Subpopulation")
        for _ in range(20)
    ])
    fit_target_age = mo.ui.array([
        mo.ui.dropdown(options=_age_opts, value=_age_opts[0], label="Age group")
        for _ in range(20)
    ])
    fit_target_risk = mo.ui.array([
        mo.ui.dropdown(options=_risk_opts, value=_risk_opts[0], label="Risk group")
        for _ in range(20)
    ])

    fit_sim_days_input = mo.ui.number(
        value=180, start=1, stop=3650, step=1,
        label="Simulation days (used when all targets are scalar totals)",
    )
    _seed_scale_opts = {
        f"seed_scale_{_c}": f"seed_scale_{_c}"
        for _c in compartments[1:]
    }
    fit_params_multiselect = mo.ui.multiselect(
        options={**{p: p for p in param_names}, **_seed_scale_opts},
        value=[],
        label="Parameters to fit",
    )
    fit_method = mo.ui.radio(
        options={"Adam (gradient)": "adam", "L-BFGS (gradient)": "lbfgs", "Accept-reject": "ar"},
        value="Adam (gradient)", label="Fitting method",
    )
    fit_n_iter = mo.ui.number(value=200, start=10, stop=2000, step=10, label="Iterations / Max samples")
    fit_r2_thresh = mo.ui.number(value=0.75, start=0.0, stop=1.0, step=None, label="R² acceptance threshold")
    fit_run_button = mo.ui.run_button(label="Run fitting")

    return (
        fit_target_src, fit_target_upload, fit_target_path,
        fit_target_vars, fit_target_mode, fit_target_weight,
        fit_target_subpop, fit_target_age, fit_target_risk,
        fit_sim_days_input,
        fit_params_multiselect,
        fit_method, fit_n_iter, fit_r2_thresh, fit_run_button,
    )


@app.cell
def _fitting_lr_ui(mo, fit_method):
    _lr_default = 0.5 if fit_method.value == "lbfgs" else 0.01
    fit_lr = mo.ui.number(value=_lr_default, start=1e-5, stop=10.0, step=None, label="Learning rate")
    return (fit_lr,)


@app.cell
def _fitting_replications_ui(mo):
    fit_n_replications = mo.ui.number(
        value=5, start=1, stop=200, step=1,
        label="Number of replications",
    )
    return (fit_n_replications,)


@app.cell
def _fitting_bounds_ui(
    mo, fit_params_multiselect, config_dict,
    num_age_groups, num_risk_groups, is_metapop,
):
    _saved_params = config_dict.get("params", {})
    _selected = list(fit_params_multiselect.value)
    _A = num_age_groups
    _R = num_risk_groups

    _dim_opts = []
    if _A > 1:
        _dim_opts.append("age groups")
    if _R > 1:
        _dim_opts.append("risk groups")
    if is_metapop:
        _dim_opts.append("subpopulation")

    def _default_bounds(pn):
        if pn.startswith("seed_scale_"):
            return 0.1, 10.0
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
            start=1e-8, stop=1e8, step=None,
            value=_default_bounds(_pn)[0],
            label="Lower bound",
        )
        for _pn in _selected
    ])
    fit_bounds_hi = mo.ui.array([
        mo.ui.number(
            start=1e-8, stop=1e8, step=None,
            value=_default_bounds(_pn)[1],
            label="Upper bound",
        )
        for _pn in _selected
    ])
    fit_param_dims = mo.ui.array([
        mo.ui.multiselect(
            options=[] if _pn.startswith("seed_scale_") else _dim_opts,
            value=[],
            label="Granularity",
        )
        for _pn in _selected
    ])
    return (fit_bounds_lo, fit_bounds_hi, fit_param_dims)


@app.cell
def _fitting_start_offset_ui(mo):
    fit_start_offset_enable = mo.ui.checkbox(
        label="Fit epidemic start date offset", value=False,
    )
    fit_start_offset_lo = mo.ui.number(
        value=-30, start=-365, stop=0, step=1, label="Min offset (days)",
    )
    fit_start_offset_hi = mo.ui.number(
        value=30, start=0, stop=365, step=1, label="Max offset (days)",
    )
    return fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi


@app.cell
def _fitting_display(
    get_n_targets, add_target_btn, del_target_btn,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_sim_days_input,
    fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_method, fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications, fit_run_button,
    mo, main_tab,
    num_age_groups, num_risk_groups, is_metapop,
    tip_label, wtip,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Fitting", None)
    _ACC = CLT_ACCENT["fitting"]
    _n = get_n_targets()
    _selected_params = list(fit_params_multiselect.value)
    _any_non_ts = any(fit_target_mode.value[_k] in ("scalar", "proportion") for _k in range(_n))
    _any_ts = any(fit_target_mode.value[_k] == "ts" for _k in range(_n))

    # Fitting tooltips carry rich HTML (the shared helpers escape plain text by
    # default, so pass html_tip=True here).
    def _tip(tip_text):
        return tip_label("", tip_text, html_tip=True)

    _PROPORTION_TIP = (
        "<b>Proportions mode</b><br><br>"
        "The <code>value</code> column must be a fraction (0–1) representing the share of the "
        "target variable in a given group. Each row defines one constraint.<br><br>"
        "<b>Denominator rule</b><br>"
        "<code>age=X</code> only → denominator is the grand total across all subpopulations<br>"
        "<code>age=X, subpop=Y</code> → denominator is the total within subpop Y<br>"
        "<code>subpop=Y</code> only → denominator is the grand total "
        "(i.e. the share of all infections attributable to subpop Y)<br>"
        "<code>risk=Z</code> only → grand total; "
        "<code>risk=Z, subpop=Y</code> → total within subpop Y<br><br>"
        "<b>Examples</b> (target = new_infections, 2 age groups, 2 subpops)<br>"
        "<code>age=0</code> value=0.40 → age 0 accounts for 40% of all infections<br>"
        "<code>age=0, subpop=city_A</code> value=0.35 → age 0 is 35% of city A's infections<br>"
        "<code>subpop=city_A</code> value=0.60 → city A has 60% of all infections"
    )

    # ── target cards ──────────────────────────────────────────────────────────
    _target_acc = {}
    for _k in range(_n):
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        _fname_note = mo.md("")
        if _src == "upload" and fit_target_upload.value[_k]:
            _fname = fit_target_upload.value[_k][0].name
            _fname_note = mo.callout(mo.md(f"Loaded: **{_fname}**"), kind="info")

        _data_input = fit_target_upload[_k] if _src == "upload" else fit_target_path[_k]

        _scalar_hint = mo.md("")
        if _mode == "scalar":
            _scalar_hint = mo.accordion({
                "Scalar format reference": mo.callout(
                    mo.md(
                        "**Required column:** `value`.\n\n"
                        "**Optional CSV columns:** `age` (integer index), `risk` (integer index), "
                        "`subpopulation` (name or integer index). "
                        "Each row is one group-specific constraint. When a group column is absent, "
                        "the group selection dropdowns below are used as the target "
                        "(or sum over all if 'All')."
                    ),
                    kind="info",
                )
            })
        elif _mode == "proportion":
            _scalar_hint = mo.accordion({
                "Proportions format reference": mo.callout(
                    mo.md(
                        "**Required column:** `value` (0–1).\n\n"
                        "**Optional CSV columns:** `age` (integer index), `risk` (integer index), "
                        "`subpopulation` (name or integer index).\n\n"
                        "Each row is **numerator / denominator** over the full simulation:\n\n"
                        "- **Numerator** — cumulative target variable restricted to the age, risk, "
                        "and subpopulation specified in the row (falling back to the dropdowns below "
                        "when a column is absent).\n"
                        "- **Denominator** — same variable summed over *all age and risk groups*, "
                        "restricted to the row's subpopulation *only when age or risk is also given*; "
                        "otherwise the grand total (all subpopulations).\n\n"
                        "**Examples** (target = `new_infections`, 2 age groups, 2 subpopulations):\n\n"
                        "| `subpopulation` | `age` | `value` | Meaning |\n"
                        "|---|---|---|---|\n"
                        "| *(absent)* | `0` | `0.40` | Age 0 is 40 % of all infections (both subpops) |\n"
                        "| `city_A` | `0` | `0.35` | Age 0 is 35 % of city A's infections |\n"
                        "| `city_A` | `1` | `0.65` | Age 1 is 65 % of city A's infections |\n"
                        "| `city_A` | *(absent)* | `0.60` | city A accounts for 60 % of all infections |\n\n"
                        "Age rows for the same subpopulation should sum to 1."
                    ),
                    kind="info",
                )
            })

        _prop_tip_row = mo.md("")
        if _mode == "proportion":
            _prop_tip_row = mo.hstack(
                [mo.md("*Proportions quick reference*"), _tip(_PROPORTION_TIP)],
                justify="start", align="center",
            )

        _slice_items = []
        if is_metapop:
            _slice_items.append(fit_target_subpop[_k])
        if int(num_age_groups) > 1:
            _slice_items.append(fit_target_age[_k])
        if int(num_risk_groups) > 1:
            _slice_items.append(fit_target_risk[_k])
        _slice_ui = mo.hstack(_slice_items, justify="start") if _slice_items else mo.md("")

        _tvar = str(fit_target_vars.value[_k]).strip()
        _tlabel = f"Target {_k + 1}  ·  {_mode}" + (f"  ·  {_tvar}" if _tvar else "")
        _target_acc[_tlabel] = mo.vstack([
            fit_target_src[_k],
            _data_input,
            _fname_note,
            fit_target_mode[_k],
            _prop_tip_row,
            _scalar_hint,
            fit_target_vars[_k],
            mo.hstack([fit_target_weight[_k]], justify="start"),
            _slice_ui,
        ])

    # ── parameter bounds ──────────────────────────────────────────────────────
    if _selected_params:
        _rows = []
        for _j, _pn in enumerate(_selected_params):
            _is_seed_scale = _pn.startswith("seed_scale_")
            _bound_widgets = [fit_bounds_lo[_j], fit_bounds_hi[_j]]
            if not _is_seed_scale:
                _bound_widgets.append(fit_param_dims[_j])
            _rows.append(mo.vstack([
                mo.md(f"**`{_pn}`**"),
                mo.hstack(_bound_widgets, justify="start", align="center"),
            ]))
        _bounds_section = mo.vstack(_rows)
    else:
        _bounds_section = mo.md("*Select parameters above to configure bounds.*")

    _LR_TIP = (
        "Step size used by the gradient optimiser.\n\n"
        "Adam: controls how far each parameter moves per gradient step.\n"
        "  Too large → unstable loss; too small → slow convergence.\n"
        "  Typical range: 0.001 – 0.05.\n\n"
        "L-BFGS: initial step size for the internal line search.\n"
        "  L-BFGS is less sensitive than Adam; 0.1 – 1.0 usually works.\n"
        "  The line search can shrink the step automatically."
    )
    _ITER_TIP = (
        "Number of optimisation steps or random draws, depending on method.\n\n"
        "Adam: exact number of gradient update steps per replication.\n"
        "  More iterations → better convergence, but more compute.\n\n"
        "L-BFGS: outer loop runs N ÷ 20 steps; each step performs\n"
        "  up to 20 internal line-search iterations, so total function\n"
        "  evaluations ≈ N (per replication).\n\n"
        "Accept-reject: total number of random parameter sets sampled.\n"
        "  Higher → better coverage of the parameter space."
    )
    _R2_TIP = (
        "Minimum weighted R² a sample must achieve to be 'accepted'.\n\n"
        "With multiple targets, R² is computed per-target and averaged\n"
        "using the target weights (λ).\n\n"
        "Accepted samples form the ensemble used in the Forecast tab.\n"
        "  Higher threshold → fewer but better-fitting accepted sets.\n"
        "  Lower threshold → larger ensemble, more uncertainty.\n\n"
        "If no samples are accepted, lower this value or increase\n"
        "the number of samples (Iterations / Max samples).\n\n"
        "R² = 1 − (SS_res / SS_tot); values above 0.7 are often\n"
        "considered a reasonable fit."
    )
    _REP_TIP = (
        "Number of independent optimisation runs for gradient methods.\n\n"
        "Starting points are spread across the parameter bounds using\n"
        "Latin Hypercube Sampling (LHS) to cover the space evenly and\n"
        "reduce the risk of converging to a local minimum.\n\n"
        "The best-fit result is the replication with the lowest final\n"
        "loss. All replication trajectories and parameter distributions\n"
        "are shown in the results, similar to the accept-reject method."
    )

    _method_val = fit_method.value
    _hyper = [wtip(fit_n_iter, _ITER_TIP, html_tip=True)]
    if _method_val != "ar":
        _hyper = [wtip(fit_lr, _LR_TIP, html_tip=True)] + _hyper
        _hyper.append(wtip(fit_n_replications, _REP_TIP, html_tip=True))
    if _method_val == "ar":
        _hyper.append(wtip(fit_r2_thresh, _R2_TIP, html_tip=True))

    _sim_days_widget = mo.md("")
    if _any_non_ts and not _any_ts:
        _sim_days_widget = mo.callout(
            mo.vstack([
                mo.md(
                    "All targets are scalar totals or proportions. Set the simulation length below "
                    "(number of days over which the totals are accumulated)."
                ),
                fit_sim_days_input,
            ]),
            kind="info",
        )

    _seed_scale_note = mo.md("")
    if any(_pn.startswith("seed_scale_") for _pn in _selected_params):
        _seed_scale_note = mo.callout(
            mo.md(
                "**Seed scaling** multiplies the initial count of the selected compartment "
                "by the fitted scale factor, adjusting the first compartment (susceptibles) "
                "to keep total population constant. "
                "Gradient methods optimise the scale alongside rate parameters."
            ),
            kind="info",
        )

    _start_offset_section = mo.vstack([
        fit_start_offset_enable,
        mo.hstack([fit_start_offset_lo, fit_start_offset_hi], justify="start")
        if fit_start_offset_enable.value else mo.md(""),
    ]) if True else mo.md("")

    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Fitting</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Calibrate model '
            "parameters to observed data.</div>"
        ),
        section_card(
            step_header("①", "Fit Targets",
                        "The data series / totals the model is calibrated against. "
                        "Click a target to expand it.",
                        accent=_ACC),
            mo.vstack([
                mo.accordion(_target_acc, multiple=True),
                mo.hstack([add_target_btn, del_target_btn,
                           mo.md(f"*{_n} of 20 targets*")],
                          justify="start", align="center"),
                _sim_days_widget,
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("②", "Parameters to Fit",
                        "Pick which parameters to estimate and their search bounds.",
                        accent=_ACC),
            mo.vstack([
                fit_params_multiselect,
                _seed_scale_note,
                _bounds_section,
            ]),
            accent=_ACC,
        ),
        section_card(
            step_header("③", "Epidemic Start Date",
                        "Optionally fit an offset for when the epidemic seeds.",
                        accent=_ACC),
            _start_offset_section,
            accent=_ACC,
        ),
        section_card(
            step_header("④", "Method & Run",
                        "Choose the optimiser, set its hyperparameters, then run.",
                        accent=_ACC),
            mo.vstack([
                fit_method,
                mo.hstack(_hyper, justify="start"),
                fit_run_button,
            ]),
            accent=_ACC,
        ),
    ])
    return


@app.cell
def _fitting_obs_parse(
    get_n_targets,
    fit_target_src, fit_target_upload, fit_target_path, fit_target_mode,
    pd, io, Path,
):
    _n = get_n_targets()
    # fit_obs_arrays: list of (np.array | list-of-dicts | None) per target
    # fit_obs_n_days: list of int (days in timeseries) or 0 for scalar targets
    fit_obs_arrays = []
    fit_obs_n_days = []

    for _k in range(_n):
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        _df = None
        try:
            if _src == "upload" and fit_target_upload.value[_k]:
                _df = pd.read_csv(io.BytesIO(fit_target_upload.value[_k][0].contents))
            elif _src == "path" and fit_target_path.value[_k].strip():
                _df = pd.read_csv(Path(fit_target_path.value[_k].strip()).expanduser())
        except Exception:
            _df = None

        if _df is None:
            fit_obs_arrays.append(None)
            fit_obs_n_days.append(0)
            continue

        if _mode in ("scalar", "proportion"):
            if "value" not in _df.columns:
                fit_obs_arrays.append(None)
                fit_obs_n_days.append(0)
                continue
            _rows = []
            for _, _row in _df.iterrows():
                _entry = {"value": float(_row["value"])}
                for _col in ("age", "risk", "subpopulation"):
                    if _col in _df.columns and pd.notna(_row.get(_col)):
                        _entry[_col] = _row[_col]
                _rows.append(_entry)
            fit_obs_arrays.append(_rows)
            fit_obs_n_days.append(0)
        else:
            _META_COLS_TS = {"date", "day", "time", "week", "subpopulation", "age", "risk"}
            _non_id = [c for c in _df.columns if c.lower() not in _META_COLS_TS]
            if not _non_id:
                _non_id = [c for c in _df.columns if c.lower() not in ("date", "day", "time", "week")]
            _val_col = "value" if "value" in _df.columns else (_non_id[0] if _non_id else None)
            if _val_col:
                _arr = _df[_val_col].dropna().to_numpy(dtype=float)
                fit_obs_arrays.append(_arr)
                fit_obs_n_days.append(len(_arr))
            else:
                fit_obs_arrays.append(None)
                fit_obs_n_days.append(0)

    return fit_obs_arrays, fit_obs_n_days


@app.cell
def _run_fitting(
    fit_run_button, fit_obs_arrays, fit_obs_n_days,
    get_n_targets,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_sim_days_input,
    fit_method, fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications,
    config_dict, compartments, is_metapop,
    build_compartment_init, read_initial_conditions,
    start_date_input, timesteps, rng_seed,
    num_age_groups, num_risk_groups,
    metapop_folder_input, metapop_travel_config,
    make_single_pop_metapop, make_metapop_from_folder,
    build_generic_torch_inputs, generic_torch_simulate_calibration_target, RATE_TEMPLATE_REGISTRY,
    torch, np, json, mo, compute_rsquared, FitResult, build_scalar_array, Path,
):
    fit_result = None
    if fit_run_button.value:
        _n = get_n_targets()
        _selected_params = list(fit_params_multiselect.value)
        _lo_vals = list(fit_bounds_lo.value)
        _hi_vals = list(fit_bounds_hi.value)
        _dim_vals = list(fit_param_dims.value)
        _A = num_age_groups
        _R = num_risk_groups
    
        _target_vars_list = [list(fit_target_vars.value[_k]) for _k in range(_n)]
        _target_modes = [fit_target_mode.value[_k] for _k in range(_n)]
        _target_weights = [float(fit_target_weight.value[_k]) for _k in range(_n)]
        _weight_sum = sum(_target_weights) or 1.0
    
        # ── slice-index helpers ────────────────────────────────────────────────────
        def _parse_idx(val):
            if val == "All (sum)":
                return -1
            return int(str(val).split(":")[0])
    
        _sp_idxs = [_parse_idx(fit_target_subpop.value[_k]) for _k in range(_n)]
        _age_idxs = [_parse_idx(fit_target_age.value[_k]) for _k in range(_n)]
        _risk_idxs = [_parse_idx(fit_target_risk.value[_k]) for _k in range(_n)]
    
        # Target display labels
        _target_labels = []
        for _k in range(_n):
            if fit_target_src.value[_k] == "upload" and fit_target_upload.value[_k]:
                _target_labels.append(fit_target_upload.value[_k][0].name)
            elif fit_target_src.value[_k] == "path" and fit_target_path.value[_k].strip():
                _target_labels.append(Path(fit_target_path.value[_k].strip()).name)
            else:
                _target_labels.append(f"Target {_k + 1}")
    
        # Subpop name → index map (for scalar CSV rows)
        _subpop_names_run = []
        if is_metapop and metapop_folder_input.value.strip():
            try:
                with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                    _subpop_names_run = json.load(_f).get("subpopulations", [])
            except Exception:
                _subpop_names_run = []
        _subpop_name_to_idx = {str(_nm): _i for _i, _nm in enumerate(_subpop_names_run)}
    
        def _parse_scalar_row_idxs(row, sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
            _sp = row.get("subpopulation", None)
            _sp_idx = sp_fallback
            if _sp is not None:
                _sp_s = str(_sp).strip()
                if _sp_s.isdigit():
                    _sp_idx = int(_sp_s)
                elif _sp_s in _subpop_name_to_idx:
                    _sp_idx = _subpop_name_to_idx[_sp_s]
            _ag_idx = int(row["age"]) if "age" in row else ag_fallback
            _rk_idx = int(row["risk"]) if "risk" in row else rk_fallback
            return _sp_idx, _ag_idx, _rk_idx
    
        # ── validation ─────────────────────────────────────────────────────────────
        for _k in range(_n):
            if fit_obs_arrays[_k] is None:
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_k + 1}: no observed data loaded.**"), kind="warn"),
                )
            if not _target_vars_list[_k]:
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_k + 1}: no variables selected.**"), kind="warn"),
                )
        mo.stop(
            not _selected_params,
            mo.callout(mo.md("**No parameters to fit.** Select parameters above."), kind="warn"),
        )
    
        _target_tvs = [[t for t in _target_vars_list[_k] if t not in compartments] for _k in range(_n)]
        _target_comps = [[t for t in _target_vars_list[_k] if t in compartments] for _k in range(_n)]
    
        # Determine simulation length
        _ts_days = [fit_obs_n_days[_k] for _k in range(_n) if _target_modes[_k] == "ts"]
        _sim_days = max(_ts_days) if _ts_days else int(fit_sim_days_input.value)
        mo.stop(
            len(set(_ts_days)) > 1,
            mo.callout(
                mo.md(
                    "**Timeseries targets have mismatched lengths:** "
                    + ", ".join(
                        f"Target {_k + 1}: {fit_obs_n_days[_k]} days"
                        for _k in range(_n)
                        if _target_modes[_k] == "ts"
                    )
                    + ". All timeseries targets must have the same number of observations."
                ),
                kind="danger",
            ),
        )
        _num_fit_days_per_target = [
            fit_obs_n_days[_k] if _target_modes[_k] == "ts" else _sim_days
            for _k in range(_n)
        ]
    
        _start = start_date_input.value.strip() or "2024-01-01"
        _ts = int(timesteps.value)
        _seed_b = int(rng_seed.value)
    
        _ci = None
        if not is_metapop:
            # Initial conditions from the Step 6 tables via config_dict.
            _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
            _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((_A, _R))), dtype=float)
            _seed_arrays = {
                _c: np.asarray(_a, dtype=float)
                for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
                if _c in compartments
            }
            _ci, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)
    
        _bounds_grad = {
            _selected_params[_j]: [float(_lo_vals[_j]), float(_hi_vals[_j])]
            for _j in range(len(_selected_params))
        }

        # Identify seed-scale virtual params (scale initial compartment count)
        _seed_scale_comps = {
            _pn[len("seed_scale_"):]: _j
            for _j, _pn in enumerate(_selected_params)
            if _pn.startswith("seed_scale_") and _pn[len("seed_scale_"):] in compartments
        }

        # Epidemic start date offset settings
        _fit_start_offset = fit_start_offset_enable.value
        _offset_lo_val = int(fit_start_offset_lo.value)
        _offset_hi_val = int(fit_start_offset_hi.value)

        from datetime import datetime as _dt, timedelta as _td
        def _shift_date(base_str, days):
            return (
                _dt.strptime(base_str, "%Y-%m-%d") + _td(days=int(days))
            ).strftime("%Y-%m-%d")

        _loss_curve = []
        _best_params = {}
        _accepted_params_list = []
        _best_trajs = {}
        _accepted_trajectories = []
    
        with mo.status.spinner("Running fitting..."):
            try:
                # ── gradient methods (Adam / L-BFGS) ──────────────────────────────
                if fit_method.value in ("adam", "lbfgs"):
                    mo.stop(
                        torch is None,
                        mo.callout(
                            mo.md("**PyTorch not available.** Install torch to use gradient-based fitting."),
                            kind="danger",
                        ),
                    )
                    # Scale compartment values and total population up by this factor.
                    # PyTorch uses softplus for non-negativity; for small counts it
                    # diverges from the identity — large values (~10000) make it negligible.
                    # N must scale with the compartments so I/N (force of infection) is unchanged.
                    _GRAD_SCALE = 10000.0

                    _all_tvs_union = list(dict.fromkeys(tv for tvs in _target_tvs for tv in tvs))
                    _all_comps_union = list(dict.fromkeys(c for cs in _target_comps for c in cs))

                    # Build initial model to identify valid regular params
                    if is_metapop:
                        _metapop, _mc = make_metapop_from_folder(
                            metapop_folder_input.value.strip(),
                            config_dict, _start, _sim_days, list(compartments),
                            seed_offset=0, seed_base=_seed_b, ts_per_day=_ts,
                            stochastic=False, tvs=_all_tvs_union, save_daily=False,
                            travel_config=metapop_travel_config or None,
                        )
                    else:
                        _metapop, _mc, _ = make_single_pop_metapop(
                            config_dict, _start, _sim_days, _ci,
                            ts_per_day=_ts, stochastic=False, tvs=_all_tvs_union,
                            save_daily=False, seed_base=_seed_b,
                            travel_config=metapop_travel_config,
                        )

                    _ti = build_generic_torch_inputs(_metapop, _mc, _sim_days)
                    _ti["params_dict"]["total_pop_age_risk"] = (
                        _ti["params_dict"]["total_pop_age_risk"] * _GRAD_SCALE
                    )
                    _ti["precomputed"].total_pop_LAR_tensor = (
                        _ti["precomputed"].total_pop_LAR_tensor * _GRAD_SCALE
                    )
                    _ti["precomputed"].total_pop_LA = (
                        _ti["precomputed"].total_pop_LA * _GRAD_SCALE
                    )

                    # Valid regular params (seed_scale virtual params are excluded)
                    _valid_pn_idxs = [
                        _j for _j, _pn in enumerate(_selected_params)
                        if _pn in _ti["params_dict"]
                    ]
                    mo.stop(
                        not _valid_pn_idxs and not _seed_scale_comps and not _fit_start_offset,
                        mo.callout(
                            mo.md("**None of the specified parameters found in params dict.** Check names."),
                            kind="danger",
                        ),
                    )

                    # Mutable dicts updated per replication so closures see current values
                    _seed_base_state = {
                        k: v.clone().detach() * _GRAD_SCALE
                        for k, v in _ti["state_dict"].items()
                    }
                    _scale_tensors = {}  # {comp_name: tensor} — cleared and repopulated per rep

                    # Extended LHS: regular params + seed_scale params + offset
                    _seed_scale_keys = list(_seed_scale_comps.keys())
                    _lhs_lo_list = [float(_lo_vals[_j]) for _j in _valid_pn_idxs]
                    _lhs_hi_list = [float(_hi_vals[_j]) for _j in _valid_pn_idxs]
                    for _comp_ss in _seed_scale_keys:
                        _j_ss = _seed_scale_comps[_comp_ss]
                        _lhs_lo_list.append(float(_lo_vals[_j_ss]))
                        _lhs_hi_list.append(float(_hi_vals[_j_ss]))
                    _offset_lhs_col = len(_lhs_lo_list)
                    if _fit_start_offset:
                        _lhs_lo_list.append(float(_offset_lo_val))
                        _lhs_hi_list.append(float(_offset_hi_val))

                    _n_rep = int(fit_n_replications.value)
                    _n_lhs_total = max(len(_lhs_lo_list), 1)
                    try:
                        from scipy.stats.qmc import LatinHypercube as _LHC
                        _lhs_unit = _LHC(d=_n_lhs_total).random(n=_n_rep)
                    except Exception:
                        _lhs_unit = np.random.rand(_n_rep, _n_lhs_total)
                    _lo_ext = np.array(_lhs_lo_list) if _lhs_lo_list else np.array([0.0])
                    _hi_ext = np.array(_lhs_hi_list) if _lhs_hi_list else np.array([1.0])
                    _lhs_scaled = _lo_ext + _lhs_unit[:, :len(_lhs_lo_list)] * (_hi_ext - _lo_ext)

                    def _slice_tensor(sim, sp_idx, ag_idx, rk_idx):
                        """Slice (days, L, A, R) and sum remaining dims → (days,)."""
                        _s = sim
                        if sp_idx >= 0:
                            _s = _s[:, sp_idx:sp_idx + 1, :, :]
                        if ag_idx >= 0:
                            _s = _s[:, :, ag_idx:ag_idx + 1, :]
                        if rk_idx >= 0:
                            _s = _s[:, :, :, rk_idx:rk_idx + 1]
                        return _s.sum(dim=tuple(range(1, _s.dim())))

                    def _build_scaled_state():
                        """Build initial state tensor dict, applying seed-scale factors."""
                        _state_run = {}
                        for _ks, _vs in _seed_base_state.items():
                            if _ks in _scale_tensors:
                                _state_run[_ks] = _seed_base_state[_ks] * _scale_tensors[_ks]
                            elif compartments and _ks == compartments[0] and _scale_tensors:
                                _delta = sum(
                                    _seed_base_state[_c] * (_scale_tensors[_c] - 1.0)
                                    for _c in _scale_tensors
                                    if _c in _seed_base_state
                                )
                                _state_run[_ks] = torch.clamp(
                                    _seed_base_state[_ks] - _delta, min=0.0
                                )
                            else:
                                _state_run[_ks] = _vs.clone()
                        return _state_run

                    def _compute_total_loss():
                        _state_run = _build_scaled_state()
                        _sim_all = generic_torch_simulate_calibration_target(
                            _state_run, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                            _ti["precomputed"], _ti["schedules_dict"],
                            _sim_days, _ts, _all_tvs_union, _all_comps_union,
                        )  # dict: {var_name: (days, L, A, R)}
                        _total = torch.tensor(0.0, dtype=torch.float64)
                        for _k in range(_n):
                            _parts = [_sim_all[v] for v in _target_vars_list[_k]]
                            _sim_k = _parts[0]
                            for _pt in _parts[1:]:
                                _sim_k = _sim_k + _pt
                            _w_k = _target_weights[_k] / _weight_sum
                            if _target_modes[_k] == "scalar":
                                _loss_k = torch.tensor(0.0, dtype=torch.float64)
                                _obs_scalar_sq_sum = sum(
                                    float(_row["value"]) ** 2 for _row in fit_obs_arrays[_k]
                                ) + 1e-10
                                for _row in fit_obs_arrays[_k]:
                                    _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                        _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    _sliced = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i) / _GRAD_SCALE
                                    _obs_v = torch.tensor(float(_row["value"]), dtype=torch.float64)
                                    _loss_k = _loss_k + (_sliced.sum() - _obs_v) ** 2
                                _loss_k = _loss_k / _obs_scalar_sq_sum
                            elif _target_modes[_k] == "proportion":
                                _loss_k = torch.tensor(0.0, dtype=torch.float64)
                                # Normalise by sum(obs²) so loss ≈ 1 when sim=0, ≈ 0 when perfect,
                                # keeping it in [0, ~1] regardless of epidemic size or number of rows.
                                _obs_prop_sq_sum = sum(
                                    float(_row["value"]) ** 2 for _row in fit_obs_arrays[_k]
                                ) + 1e-10
                                for _ri_prop, _row in enumerate(fit_obs_arrays[_k]):
                                    _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                        _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    # When no group is specified, treat row order as subpop order
                                    # so that rows without a "subpopulation" column still compute
                                    # a meaningful per-subpop proportion instead of always 1.0.
                                    if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                        _sp_i = _ri_prop
                                    # Fix subpop in denominator only when age or risk also given;
                                    # subpop-only rows compare against grand total (share of all
                                    # infections attributable to that subpopulation).
                                    _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                                    _den = _slice_tensor(_sim_k, _den_sp, -1, -1).sum()
                                    _num = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i).sum()
                                    _sim_prop = _num / (_den + 1e-10)
                                    _obs_prop = torch.tensor(float(_row["value"]), dtype=torch.float64)
                                    _loss_k = _loss_k + (_sim_prop - _obs_prop) ** 2
                                _loss_k = _loss_k / _obs_prop_sq_sum
                            else:
                                _days_k = _num_fit_days_per_target[_k]
                                _sliced = _slice_tensor(_sim_k, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]) / _GRAD_SCALE
                                _obs_t_k = torch.tensor(
                                    fit_obs_arrays[_k][:_days_k], dtype=torch.float64
                                )
                                # Normalise by mean(obs²) so loss ≈ 1 when sim=0, ≈ 0 when perfect,
                                # matching the [0, ~1] scale of proportion and scalar losses.
                                _obs_sq_mean_k = float((_obs_t_k ** 2).mean()) + 1e-10
                                _loss_k = ((_sliced[:_days_k] - _obs_t_k) ** 2).mean() / _obs_sq_mean_k
                            _total = _total + _w_k * _loss_k
                        return _total

                    def _record_params_grad():
                        _out = {}
                        for _pn in _selected_params:
                            if _pn.startswith("seed_scale_"):
                                _comp = _pn[len("seed_scale_"):]
                                if _comp in _scale_tensors:
                                    _out[_pn] = float(_scale_tensors[_comp].item())
                            elif _pn in _ti["params_dict"]:
                                _v = _ti["params_dict"][_pn].detach()
                                if _v.ndim == 3 and _v.shape[1] == 1 and _v.shape[2] == 1:
                                    for _spi in range(_v.shape[0]):
                                        _out[f"{_pn}_subpop{_spi}"] = float(_v[_spi, 0, 0].item())
                                else:
                                    _out[_pn] = _v.tolist()
                        return _out

                    def _record_trajs_grad():
                        _trajs = {}
                        with torch.no_grad():
                            _state_run = _build_scaled_state()
                            _sim_all = generic_torch_simulate_calibration_target(
                                _state_run, _ti["params_dict"], _mc, RATE_TEMPLATE_REGISTRY,
                                _ti["precomputed"], _ti["schedules_dict"],
                                _sim_days, _ts, _all_tvs_union, _all_comps_union,
                            )
                            for _k in range(_n):
                                _parts = [_sim_all[v] for v in _target_vars_list[_k]]
                                _sim_k = _parts[0]
                                for _pt in _parts[1:]:
                                    _sim_k = _sim_k + _pt
                                if _target_modes[_k] == "scalar":
                                    _row_sims = []
                                    for _row in fit_obs_arrays[_k]:
                                        _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                            _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                        )
                                        _sliced = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i)
                                        _row_sims.append(float(_sliced.sum().item()) / _GRAD_SCALE)
                                    _trajs[f"target_{_k}"] = _row_sims
                                elif _target_modes[_k] == "proportion":
                                    _row_sims = []
                                    for _ri_prop, _row in enumerate(fit_obs_arrays[_k]):
                                        _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                            _row, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                        )
                                        if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                            _sp_i = _ri_prop
                                        _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                                        _den = _slice_tensor(_sim_k, _den_sp, -1, -1).sum()
                                        _num = _slice_tensor(_sim_k, _sp_i, _ag_i, _rk_i).sum()
                                        _row_sims.append(float((_num / (_den + 1e-10)).item()))
                                    _trajs[f"target_{_k}"] = _row_sims
                                else:
                                    _days_k = _num_fit_days_per_target[_k]
                                    _sliced = _slice_tensor(
                                        _sim_k, _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k]
                                    )
                                    _trajs[f"target_{_k}"] = (_sliced[:_days_k] / _GRAD_SCALE).numpy().tolist()
                        return _trajs

                    _n_it = int(fit_n_iter.value)
                    _lr_val = float(fit_lr.value)
                    _best_loss = float("inf")
                    _all_rep_params = []
                    _all_rep_trajs = []
                    _n_regular_lhs = len(_valid_pn_idxs)

                    for _rep_idx in range(_n_rep):
                        # ── Extract seed-scale and offset from extended LHS ────────
                        _scale_tensors.clear()
                        _sampled_offset = 0
                        for _ssi, _comp_ss in enumerate(_seed_scale_keys):
                            _sval = float(_lhs_scaled[_rep_idx, _n_regular_lhs + _ssi])
                            _scale_tensors[_comp_ss] = torch.tensor(
                                _sval, dtype=torch.float64, requires_grad=True,
                            )
                        if _fit_start_offset and len(_lhs_lo_list) > _offset_lhs_col:
                            _sampled_offset = int(round(
                                float(_lhs_scaled[_rep_idx, _offset_lhs_col])
                            ))
                        _start_rep = _shift_date(_start, _sampled_offset) if _fit_start_offset else _start

                        # ── Per-replication model rebuild when start date varies ──
                        if _fit_start_offset:
                            if is_metapop:
                                _metapop, _mc = make_metapop_from_folder(
                                    metapop_folder_input.value.strip(),
                                    config_dict, _start_rep, _sim_days, list(compartments),
                                    seed_offset=_rep_idx, seed_base=_seed_b, ts_per_day=_ts,
                                    stochastic=False, tvs=_all_tvs_union, save_daily=False,
                                    travel_config=metapop_travel_config or None,
                                )
                            else:
                                _metapop, _mc, _ = make_single_pop_metapop(
                                    config_dict, _start_rep, _sim_days, _ci,
                                    ts_per_day=_ts, stochastic=False, tvs=_all_tvs_union,
                                    save_daily=False, seed_base=_seed_b + _rep_idx,
                                    travel_config=metapop_travel_config,
                                )
                            _ti = build_generic_torch_inputs(_metapop, _mc, _sim_days)
                            _ti["params_dict"]["total_pop_age_risk"] = (
                                _ti["params_dict"]["total_pop_age_risk"] * _GRAD_SCALE
                            )
                            _ti["precomputed"].total_pop_LAR_tensor = (
                                _ti["precomputed"].total_pop_LAR_tensor * _GRAD_SCALE
                            )
                            _ti["precomputed"].total_pop_LA = (
                                _ti["precomputed"].total_pop_LA * _GRAD_SCALE
                            )
                            _seed_base_state.clear()
                            _seed_base_state.update(
                                {k: v.clone().detach() * _GRAD_SCALE for k, v in _ti["state_dict"].items()}
                            )

                        # ── Initialise regular param tensors from LHS ─────────────
                        _opt_tensors = list(_scale_tensors.values())
                        _lhs_col = 0
                        for _j, _pn in enumerate(_selected_params):
                            if _pn not in _ti["params_dict"]:
                                continue
                            _dims = _dim_vals[_j] if _dim_vals else []
                            _has_age = "age groups" in _dims
                            _has_risk = "risk groups" in _dims
                            _init_val = float(_lhs_scaled[_rep_idx, _lhs_col])
                            _lhs_col += 1
                            if _has_age and _has_risk:
                                _t = torch.full((_A, _R), _init_val, dtype=torch.float64, requires_grad=True)
                            elif _has_age:
                                _t = torch.full((_A, 1), _init_val, dtype=torch.float64, requires_grad=True)
                            elif _has_risk:
                                _t = torch.full((1, _R), _init_val, dtype=torch.float64, requires_grad=True)
                            elif "subpopulation" in _dims:
                                _L = list(_ti["state_dict"].values())[0].shape[0]
                                _t = torch.full((_L, 1, 1), _init_val, dtype=torch.float64, requires_grad=True)
                            else:
                                _t = torch.tensor(_init_val, dtype=torch.float64, requires_grad=True)
                            _ti["params_dict"][_pn] = _t
                            _opt_tensors.append(_t)

                        _rep_best_loss = float("inf")
                        _rep_best_params = {}
                        _rep_best_trajs = {}

                        _rep_loss_curve = []
                        if not _opt_tensors:
                            # Only start-offset is being searched; evaluate once
                            with torch.no_grad():
                                _lv0 = float(_compute_total_loss().item())
                            _rep_loss_curve = [_lv0]
                            _rep_best_loss = _lv0
                            _rep_best_params = _record_params_grad()
                            _rep_best_trajs = _record_trajs_grad()
                        elif fit_method.value == "adam":
                            _opt = torch.optim.Adam(_opt_tensors, lr=_lr_val)
                            for _ in range(_n_it):
                                _opt.zero_grad()
                                _loss = _compute_total_loss()
                                _loss.backward()
                                _opt.step()
                                for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                    if _pn in _ti["params_dict"]:
                                        _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                                for _comp_ss in _seed_scale_keys:
                                    _j_ss = _seed_scale_comps[_comp_ss]
                                    _scale_tensors[_comp_ss].data.clamp_(
                                        float(_lo_vals[_j_ss]), float(_hi_vals[_j_ss])
                                    )
                                _lv = float(_loss.item())
                                _rep_loss_curve.append(_lv)
                                if _lv < _rep_best_loss:
                                    _rep_best_loss = _lv
                                    _rep_best_params = _record_params_grad()
                                    _rep_best_trajs = _record_trajs_grad()

                        else:  # lbfgs
                            _opt = torch.optim.LBFGS(_opt_tensors, lr=_lr_val, max_iter=20)
                            for _ in range(max(1, _n_it // 20)):
                                def _closure():
                                    _opt.zero_grad()
                                    _l = _compute_total_loss()
                                    _l.backward()
                                    return _l
                                _opt.step(_closure)
                                for _pn, (_pmin, _pmax) in _bounds_grad.items():
                                    if _pn in _ti["params_dict"]:
                                        _ti["params_dict"][_pn].data.clamp_(_pmin, _pmax)
                                for _comp_ss in _seed_scale_keys:
                                    _j_ss = _seed_scale_comps[_comp_ss]
                                    _scale_tensors[_comp_ss].data.clamp_(
                                        float(_lo_vals[_j_ss]), float(_hi_vals[_j_ss])
                                    )
                                with torch.no_grad():
                                    _lv2 = float(_compute_total_loss().item())
                                _rep_loss_curve.append(_lv2)
                                if _lv2 < _rep_best_loss:
                                    _rep_best_loss = _lv2
                                    _rep_best_params = _record_params_grad()
                                    _rep_best_trajs = _record_trajs_grad()
                        _loss_curve.append(_rep_loss_curve)

                        if _fit_start_offset:
                            _rep_best_params["epidemic_start_offset_days"] = _sampled_offset
                            _rep_best_params["epidemic_start_date"] = _start_rep

                        _all_rep_params.append(_rep_best_params)
                        _all_rep_trajs.append(_rep_best_trajs)
                        if _rep_best_loss < _best_loss:
                            _best_loss = _rep_best_loss
                            _best_params = _rep_best_params
                            _best_trajs = _rep_best_trajs

                    _accepted_params_list = _all_rep_params
                    _accepted_trajectories = _all_rep_trajs
    
                # ── accept-reject ──────────────────────────────────────────────────
                else:
                    _all_tvs_ar = list(dict.fromkeys(tv for tvs in _target_tvs for tv in tvs))
                    _all_comps_ar = list(dict.fromkeys(c for cs in _target_comps for c in cs))
                    _has_any_comp_ar = bool(_all_comps_ar)

                    _n_subpops = 1
                    if is_metapop and any(
                        "subpopulation" in (_dim_vals[_j] if _dim_vals else [])
                        for _j in range(len(_selected_params))
                    ):
                        try:
                            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                                _n_subpops = len(json.load(_f).get("subpopulations", []))
                        except Exception:
                            _n_subpops = 1

                    # Pre-load metapop base ICs once for seed scaling
                    _metapop_base_inits = []
                    if _seed_scale_comps and is_metapop and metapop_folder_input.value.strip():
                        _folder_ar = Path(metapop_folder_input.value.strip())
                        for _sp_nm_ar in _subpop_names_run:
                            _ic_path_ar = _folder_ar / f"initial_conditions_{_sp_nm_ar}.json"
                            _ic_cfg_ar = (config_dict.get("initial_conditions", {}) or {}).get(_sp_nm_ar, {})
                            _table_ci_ar = read_initial_conditions(
                                config_dict, _sp_nm_ar, compartments, _A, _R)
                            _comp_base_ar = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                            _epi_base_ar = {}
                            if _ic_cfg_ar.get("seeds") and _table_ci_ar is not None:
                                _comp_base_ar = _table_ci_ar
                            elif _ic_path_ar.exists():
                                try:
                                    with open(_ic_path_ar) as _f:
                                        _ic_data_ar = json.load(_f)
                                    for _c, _arr in _ic_data_ar.get("compartments", {}).items():
                                        if _c in compartments:
                                            _comp_base_ar[_c] = np.array(_arr, dtype=float)
                                    for _em, _arr in _ic_data_ar.get("epi_metrics", {}).items():
                                        _epi_base_ar[_em] = np.array(_arr, dtype=float)
                                except Exception:
                                    pass
                            elif _table_ci_ar is not None:
                                _comp_base_ar = _table_ci_ar
                            _metapop_base_inits.append((_comp_base_ar, _epi_base_ar))

                    def _extract_ts_sliced(metapop_obj, target_vars_k, sp_idx, ag_idx, rk_idx, n_days):
                        """Extract timeseries array with optional subpop/age/risk slicing."""
                        _sps = list(metapop_obj.subpop_models.values())
                        _sp_list = [_sps[sp_idx]] if sp_idx >= 0 else _sps
                        _result = None
                        for _sp_m in _sp_list:
                            for _var in target_vars_k:
                                if _var in _sp_m.transition_variables:
                                    _h = np.array(
                                        _sp_m.transition_variables[_var].history_vals_list
                                    )[:n_days]
                                elif _var in _sp_m.compartments:
                                    _h = np.array(
                                        _sp_m.compartments[_var].history_vals_list
                                    )[:n_days]
                                else:
                                    continue
                                if ag_idx >= 0:
                                    _h = _h[:, ag_idx:ag_idx + 1, :]
                                if rk_idx >= 0:
                                    _h = _h[:, :, rk_idx:rk_idx + 1]
                                _hv = _h.sum(axis=(1, 2))
                                _result = _hv if _result is None else _result + _hv
                        return _result if _result is not None else np.zeros(n_days)

                    def _extract_scalar_rows(metapop_obj, target_vars_k, scalar_rows, n_days,
                                             sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
                        """Extract per-row scalar totals for scalar-mode targets."""
                        _sps = list(metapop_obj.subpop_models.values())
                        _row_sims = []
                        for _row in scalar_rows:
                            _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                _row, sp_fallback, ag_fallback, rk_fallback
                            )
                            _sp_list = [_sps[_sp_i]] if _sp_i >= 0 else _sps
                            _total = 0.0
                            for _sp_m in _sp_list:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    if _ag_i >= 0:
                                        _h = _h[:, _ag_i:_ag_i + 1, :]
                                    if _rk_i >= 0:
                                        _h = _h[:, :, _rk_i:_rk_i + 1]
                                    _total += float(_h.sum())
                            _row_sims.append(_total)
                        return _row_sims

                    def _extract_proportion_rows(metapop_obj, target_vars_k, scalar_rows, n_days,
                                                 sp_fallback=-1, ag_fallback=-1, rk_fallback=-1):
                        """Extract per-row proportions for proportion-mode targets.

                        Denominator subpop: fixed to the row's subpop only when age or risk is
                        also specified; otherwise grand total (all subpops). This allows
                        subpop-only rows to express the share of infections attributable to
                        that subpopulation.
                        """
                        _sps = list(metapop_obj.subpop_models.values())
                        _row_sims = []
                        for _ri_prop, _row in enumerate(scalar_rows):
                            _sp_i, _ag_i, _rk_i = _parse_scalar_row_idxs(
                                _row, sp_fallback, ag_fallback, rk_fallback
                            )
                            # When no group is specified, treat row order as subpop order
                            if _sp_i < 0 and _ag_i < 0 and _rk_i < 0:
                                _sp_i = _ri_prop
                            _den_sp = _sp_i if (_sp_i >= 0 and (_ag_i >= 0 or _rk_i >= 0)) else -1
                            _sp_list_num = [_sps[_sp_i]] if (0 <= _sp_i < len(_sps)) else _sps
                            _sp_list_den = [_sps[_den_sp]] if _den_sp >= 0 else _sps
                            _num = 0.0
                            _den = 0.0
                            for _sp_m in _sp_list_den:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h_full = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h_full = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    _den += float(_h_full.sum())
                            for _sp_m in _sp_list_num:
                                for _var in target_vars_k:
                                    if _var in _sp_m.transition_variables:
                                        _h_full = np.array(
                                            _sp_m.transition_variables[_var].history_vals_list
                                        )[:n_days]
                                    elif _var in _sp_m.compartments:
                                        _h_full = np.array(
                                            _sp_m.compartments[_var].history_vals_list
                                        )[:n_days]
                                    else:
                                        continue
                                    _h_sliced = _h_full
                                    if _ag_i >= 0:
                                        _h_sliced = _h_sliced[:, _ag_i:_ag_i + 1, :]
                                    if _rk_i >= 0:
                                        _h_sliced = _h_sliced[:, :, _rk_i:_rk_i + 1]
                                    _num += float(_h_sliced.sum())
                            _row_sims.append(_num / _den if _den > 1e-10 else 0.0)
                        return _row_sims

                    _rng = np.random.default_rng(_seed_b)
                    _n_samples = int(fit_n_iter.value)
                    _r2_thresh = float(fit_r2_thresh.value)
                    _best_r2 = -float("inf")

                    for _s_idx in range(_n_samples):
                        _sampled = {}
                        _per_subpop = [{} for _ in range(_n_subpops)]
                        _sampled_scales = {}  # {comp: scale_value}
                        _sampled_offset = 0

                        for _j, _pn in enumerate(_selected_params):
                            _lo = float(_lo_vals[_j])
                            _hi = float(_hi_vals[_j])
                            if _pn.startswith("seed_scale_"):
                                _comp_ss = _pn[len("seed_scale_"):]
                                if _comp_ss in compartments:
                                    _sampled_scales[_comp_ss] = float(_rng.uniform(_lo, _hi))
                                continue
                            _dims = _dim_vals[_j] if _dim_vals else []
                            _has_age = "age groups" in _dims
                            _has_risk = "risk groups" in _dims
                            _has_meta = "subpopulation" in _dims
                            if _has_age and _has_risk:
                                _inner_shape = (_A, _R)
                            elif _has_age:
                                _inner_shape = (_A, 1)
                            elif _has_risk:
                                _inner_shape = (1, _R)
                            else:
                                _inner_shape = None
                            if _has_meta:
                                for _spi in range(_n_subpops):
                                    _per_subpop[_spi][_pn] = (
                                        _rng.uniform(_lo, _hi, size=_inner_shape).tolist()
                                        if _inner_shape else float(_rng.uniform(_lo, _hi))
                                    )
                            elif _inner_shape:
                                _sampled[_pn] = _rng.uniform(_lo, _hi, size=_inner_shape).tolist()
                            else:
                                _sampled[_pn] = float(_rng.uniform(_lo, _hi))

                        # Sample start date offset (integer)
                        if _fit_start_offset:
                            _sampled_offset = int(_rng.integers(_offset_lo_val, _offset_hi_val + 1))
                        _start_iter = _shift_date(_start, _sampled_offset) if _fit_start_offset else _start

                        # Build scaled initial conditions for this iteration
                        _ci_iter = None
                        _init_override = None
                        if _sampled_scales:
                            if is_metapop and _metapop_base_inits:
                                _init_override = []
                                for _comp_base_sp, _epi_base_sp in _metapop_base_inits:
                                    _scaled_comp = {}
                                    _delta_sp = np.zeros_like(
                                        _comp_base_sp.get(
                                            compartments[0],
                                            build_scalar_array(0.0, _A, _R),
                                        )
                                    ) if compartments else None
                                    for _c in compartments:
                                        _base_arr = _comp_base_sp.get(_c, build_scalar_array(0.0, _A, _R))
                                        if _c in _sampled_scales and _c != (compartments[0] if compartments else None):
                                            _sc = _sampled_scales[_c]
                                            _scaled_comp[_c] = _base_arr * _sc
                                            if _delta_sp is not None:
                                                _delta_sp += _base_arr * (_sc - 1.0)
                                        else:
                                            _scaled_comp[_c] = _base_arr.copy()
                                    if compartments and _delta_sp is not None:
                                        _first_arr = _comp_base_sp.get(
                                            compartments[0], build_scalar_array(0.0, _A, _R)
                                        )
                                        _scaled_comp[compartments[0]] = np.maximum(
                                            _first_arr - _delta_sp, 0.0
                                        )
                                    _init_override.append((_scaled_comp, _epi_base_sp))
                            elif not is_metapop and _ci is not None:
                                _ci_iter = {}
                                _delta = np.zeros_like(
                                    _ci.get(compartments[0], build_scalar_array(0.0, _A, _R))
                                ) if compartments else None
                                for _c in compartments:
                                    _base_arr = _ci.get(_c, build_scalar_array(0.0, _A, _R))
                                    if _c in _sampled_scales and _c != (compartments[0] if compartments else None):
                                        _sc = _sampled_scales[_c]
                                        _ci_iter[_c] = _base_arr * _sc
                                        if _delta is not None:
                                            _delta += _base_arr * (_sc - 1.0)
                                    else:
                                        _ci_iter[_c] = _base_arr.copy()
                                if compartments and _delta is not None:
                                    _first_arr = _ci.get(compartments[0], build_scalar_array(0.0, _A, _R))
                                    _ci_iter[compartments[0]] = np.maximum(_first_arr - _delta, 0.0)

                        _has_per_subpop = any(_per_subpop)
                        if is_metapop:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value.strip(),
                                config_dict, _start_iter, _sim_days, list(compartments),
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_all_tvs_ar,
                                save_daily=_has_any_comp_ar,
                                param_overrides=_sampled or None,
                                param_overrides_per_subpop=_per_subpop if _has_per_subpop else None,
                                travel_config=metapop_travel_config or None,
                                init_states_override=_init_override,
                            )
                        else:
                            _ci_run = _ci_iter if _ci_iter is not None else _ci
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start_iter, _sim_days, _ci_run,
                                seed_offset=_s_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=False,
                                tvs=_all_tvs_ar,
                                save_daily=_has_any_comp_ar,
                                param_overrides=_sampled or None,
                                travel_config=metapop_travel_config,
                            )
                        _m.simulate_until_day(_sim_days)

                        _weighted_r2 = 0.0
                        _per_target_trajs = {}
                        _skip = False
                        for _k in range(_n):
                            _mode_k = _target_modes[_k]
                            _w_k = _target_weights[_k] / _weight_sum
                            _n_days_k = _num_fit_days_per_target[_k]

                            if _mode_k == "scalar":
                                _row_sims = _extract_scalar_rows(
                                    _m, _target_vars_list[_k], fit_obs_arrays[_k], _n_days_k,
                                    sp_fallback=_sp_idxs[_k], ag_fallback=_age_idxs[_k],
                                    rk_fallback=_risk_idxs[_k],
                                )
                                _obs_vals = [_row["value"] for _row in fit_obs_arrays[_k]]
                                _r2_k = compute_rsquared(_obs_vals, _row_sims)
                                _per_target_trajs[f"target_{_k}"] = _row_sims
                            elif _mode_k == "proportion":
                                _row_sims = _extract_proportion_rows(
                                    _m, _target_vars_list[_k], fit_obs_arrays[_k], _n_days_k,
                                    sp_fallback=_sp_idxs[_k], ag_fallback=_age_idxs[_k],
                                    rk_fallback=_risk_idxs[_k],
                                )
                                _obs_vals = [_row["value"] for _row in fit_obs_arrays[_k]]
                                _r2_k = compute_rsquared(_obs_vals, _row_sims)
                                _per_target_trajs[f"target_{_k}"] = _row_sims
                            else:
                                _sim_ts = _extract_ts_sliced(
                                    _m, _target_vars_list[_k],
                                    _sp_idxs[_k], _age_idxs[_k], _risk_idxs[_k],
                                    _n_days_k,
                                )
                                _obs_k = fit_obs_arrays[_k][:_n_days_k]
                                if len(_sim_ts) != len(_obs_k):
                                    _skip = True
                                    break
                                _r2_k = compute_rsquared(list(_obs_k), _sim_ts.tolist())
                                _per_target_trajs[f"target_{_k}"] = _sim_ts.tolist()

                            _weighted_r2 += _w_k * _r2_k

                        if _skip:
                            continue

                        _loss_curve.append(float(_weighted_r2))
                        if _weighted_r2 > _best_r2:
                            _best_r2 = _weighted_r2
                            _best_params = dict(_sampled)
                            for _spi, _sp_d in enumerate(_per_subpop):
                                for _kk, _vv in _sp_d.items():
                                    _best_params[f"{_kk}_subpop{_spi}"] = _vv
                            _best_params.update({f"seed_scale_{_c}": _v for _c, _v in _sampled_scales.items()})
                            if _fit_start_offset:
                                _best_params["epidemic_start_offset_days"] = _sampled_offset
                                _best_params["epidemic_start_date"] = _start_iter
                            _best_trajs = dict(_per_target_trajs)
                        if _weighted_r2 >= _r2_thresh:
                            _acc = dict(_sampled)
                            for _spi, _sp_d in enumerate(_per_subpop):
                                for _kk, _vv in _sp_d.items():
                                    _acc[f"{_kk}_subpop{_spi}"] = _vv
                            _acc.update({f"seed_scale_{_c}": _v for _c, _v in _sampled_scales.items()})
                            if _fit_start_offset:
                                _acc["epidemic_start_offset_days"] = _sampled_offset
                                _acc["epidemic_start_date"] = _start_iter
                            _accepted_params_list.append(_acc)
                            _accepted_trajectories.append(dict(_per_target_trajs))
    
            except Exception as _exc:
                import traceback as _tb
                mo.stop(
                    True,
                    mo.callout(
                        mo.md(f"**Fitting error:** {_exc}\n\n```\n{_tb.format_exc()}\n```"),
                        kind="danger",
                    ),
                )
    
        _sim_trajectories = (
            _accepted_trajectories if _accepted_trajectories
            else ([_best_trajs] if _best_trajs else [])
        )
        _is_ar = fit_method.value == "ar"
        fit_result = FitResult(
            best_params=_best_params,
            loss_curve=_loss_curve,
            num_days=_sim_days,
            observed=[list(o) if isinstance(o, np.ndarray) else o for o in fit_obs_arrays],
            method=fit_method.value,
            accepted_params=_accepted_params_list if _accepted_params_list else [_best_params],
            sim_trajectories=_sim_trajectories,
            fit_targets=_target_vars_list,
            target_labels=_target_labels,
            target_weights=_target_weights,
            target_modes=_target_modes,
            r2_threshold=_r2_thresh if _is_ar else None,
            n_ar_accepted=len(_accepted_params_list) if _is_ar else None,
        )
    return (fit_result,)


@app.cell
def _fitting_autosave(fit_result, output_dir, json):
    if fit_result is not None:
        _p = output_dir / "fitted_params.json"
        _p.write_text(json.dumps(fit_result.best_params, indent=2))
    return


@app.cell
def _fitting_results_display(fit_result, np, plt, mo, main_tab, json):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md("*Run fitting to see results.*"))
    _lc = fit_result.loss_curve
    _bp = fit_result.best_params
    _method = fit_result.method
    _accepted = fit_result.accepted_params or []
    _n_runs = len(_accepted)

    def _fmt_val(v):
        if isinstance(v, (int, float)):
            return f"{v:.6g}"
        return str(v)

    # Loss / progress plot
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 4))
    if _method == "ar":
        _axes[0].plot(_lc, linewidth=1.5, label="Weighted R²")
        _r2_thr = fit_result.r2_threshold
        if _r2_thr is not None:
            _axes[0].axhline(
                y=_r2_thr, color="red", linestyle="--", linewidth=1.2,
                alpha=0.8, label=f"Threshold ({_r2_thr:.2f})",
            )
            _axes[0].legend(fontsize=8)
        _axes[0].set_ylabel("Weighted R²")
    else:
        # _lc is list of lists — one per replication, each starting at iteration 0
        _n_rep_lc = len(_lc)
        _alpha = min(0.9, max(0.3, 3.0 / max(_n_rep_lc, 1)))
        for _ri, _rep_lc in enumerate(_lc):
            _axes[0].plot(
                _rep_lc,
                linewidth=1.2,
                alpha=_alpha,
                label=f"Rep {_ri + 1}" if _n_rep_lc <= 10 else None,
            )
        if 1 < _n_rep_lc <= 10:
            _axes[0].legend(fontsize=8)
        _axes[0].set_ylabel("Weighted MSE loss")
    _axes[0].set_xlabel("Iterations / Max samples")
    _axes[0].set_title(f"Fitting progress ({_method})")
    _axes[0].grid(True, alpha=0.3)

    # Parameter table — two columns when multiple runs available
    _axes[1].axis("off")
    if _bp:
        if _n_runs > 1:
            _rows = []
            for _pn, _bv in _bp.items():
                _all_vals = [_s.get(_pn) for _s in _accepted if isinstance(_s.get(_pn), (int, float))]
                if _all_vals:
                    _arr = np.array(_all_vals, dtype=float)
                    _med = np.median(_arr)
                    _lo95 = np.percentile(_arr, 2.5)
                    _hi95 = np.percentile(_arr, 97.5)
                    _stat_str = f"{_med:.4g} [{_lo95:.4g}, {_hi95:.4g}]"
                else:
                    _stat_str = "—"
                _rows.append([_pn, _fmt_val(_bv), _stat_str])
            _col_labels = ["Parameter", "Best-fit", "Median [2.5%, 97.5%]"]
        else:
            _rows = [[_pn, _fmt_val(_v)] for _pn, _v in _bp.items()]
            _col_labels = ["Parameter", "Best-fit value"]
        _tbl = _axes[1].table(
            cellText=_rows, colLabels=_col_labels,
            loc="center", cellLoc="left",
        )
        _tbl.auto_set_font_size(True)
        _tbl.scale(1.2, 1.5)
        _axes[1].set_title("Best-fit parameters")
    plt.tight_layout()

    _params_md = "\n".join(f"- `{k}` = **{_fmt_val(v)}**" for k, v in _bp.items())

    _accepted_note = mo.md("")
    _ar_multi_note = mo.md("")
    if _method == "ar":
        _n_acc = fit_result.n_ar_accepted if fit_result.n_ar_accepted is not None else _n_runs
        if _n_acc == 0:
            _accepted_note = mo.callout(
                mo.md(
                    f"**No parameter sets passed the weighted R² threshold "
                    f"({fit_result.r2_threshold:.2f}).** Showing best-fit only. "
                    "Lower the threshold or increase the number of samples."
                ),
                kind="warn",
            )
        else:
            _accepted_note = mo.callout(
                mo.md(
                    f"**{_n_acc} accepted parameter set(s)** passed the weighted R² threshold. "
                    "Enable **Start forecast from fitted end-state** in the Forecast tab to run an ensemble."
                ),
                kind="success" if _n_acc > 1 else "info",
            )
        if len(fit_result.target_modes or []) > 1:
            _ar_multi_note = mo.callout(
                mo.md(
                    "**Accept-reject with multiple targets:** AR samples parameters randomly and "
                    "accepts only those where the *combined* weighted R² clears the threshold. "
                    "With multiple targets the joint probability of a random sample satisfying all "
                    "objectives simultaneously is low — acceptance rates drop sharply. "
                    "Use **Adam** or **L-BFGS** for reliable multi-target fitting."
                ),
                kind="warn",
            )
    elif _method in ("adam", "lbfgs") and _n_runs > 1:
        _accepted_note = mo.callout(
            mo.md(
                f"**{_n_runs} replications** completed with LHS starting points. "
                "Best-fit is the replication with lowest final loss. "
                "Enable **Start forecast from fitted end-state** in the Forecast tab to run an ensemble."
            ),
            kind="info",
        )

    _download = mo.download(
        data=json.dumps({
            "best_params": fit_result.best_params,
            "num_days": fit_result.num_days,
            "method": fit_result.method,
            "accepted_params": fit_result.accepted_params,
        }, indent=2).encode(),
        filename="fitted_params.json",
        mimetype="application/json",
        label="Download fitted_params.json",
    )
    mo.vstack([mo.md("## Fitting Results"), _fig, mo.md(_params_md), _accepted_note, _ar_multi_note, _download])
    return


@app.cell
def _fitting_comparison_ui(mo):
    fit_comparison_style = mo.ui.radio(
        options={"Spaghetti lines": "spaghetti", "Median + 95% CI": "band"},
        value="Median + 95% CI",
        label="Display accepted parameter sets as",
    )
    return (fit_comparison_style,)


@app.cell
def _fitting_comparison_display(
    fit_result, fit_comparison_style,
    np, plt, mo, main_tab,
):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))

    _method = fit_result.method
    _trajs = fit_result.sim_trajectories
    _n_tgts = len(fit_result.fit_targets)
    _labels = fit_result.target_labels if fit_result.target_labels else [f"Target {_k+1}" for _k in range(_n_tgts)]
    _modes = list(fit_result.target_modes) if fit_result.target_modes else []
    if not _modes:
        for _k in range(_n_tgts):
            _obs_k = fit_result.observed[_k]
            _is_dict_rows = isinstance(_obs_k, list) and _obs_k and isinstance(_obs_k[0], dict)
            _modes.append("scalar" if _is_dict_rows else "ts")

    _style_ui = mo.md("")
    if _method == "ar" or (_method in ("adam", "lbfgs") and len(_trajs) > 1):
        _style_ui = mo.hstack([fit_comparison_style], justify="start")

    _figs = []
    for _k in range(_n_tgts):
        _label = _labels[_k]
        _obs_k = fit_result.observed[_k]
        _mode_k = _modes[_k]
        _traj_key = f"target_{_k}"

        if _mode_k in ("scalar", "proportion"):
            # Bar chart: observed vs simulated per row
            _obs_vals = [_row["value"] for _row in _obs_k]
            _sim_vals = [
                _traj[_traj_key][_ri] if _traj_key in _traj and _ri < len(_traj[_traj_key]) else 0.0
                for _traj in (_trajs[:1] if _trajs else [{}])
                for _ri in range(len(_obs_vals))
            ]
            # For AR with multiple accepted: show range
            if _method == "ar" and len(_trajs) > 1:
                _all_sim = np.array([
                    [_t[_traj_key][_ri] if _traj_key in _t and _ri < len(_t[_traj_key]) else 0.0
                     for _ri in range(len(_obs_vals))]
                    for _t in _trajs
                ])
                _sim_med = np.median(_all_sim, axis=0)
                _sim_lo = np.percentile(_all_sim, 2.5, axis=0)
                _sim_hi = np.percentile(_all_sim, 97.5, axis=0)
            else:
                _sim_med = np.array(_sim_vals[:len(_obs_vals)])
                _sim_lo = _sim_med
                _sim_hi = _sim_med

            _row_labels = []
            for _ri, _row in enumerate(_obs_k):
                _parts = []
                for _col in ("subpopulation", "age", "risk"):
                    if _col in _row:
                        _parts.append(f"{_col}={_row[_col]}")
                _row_labels.append(", ".join(_parts) if _parts else f"row {_ri}")

            _x = np.arange(len(_obs_vals))
            _w = 0.35
            _fig_k, _ax_k = plt.subplots(figsize=(max(6, len(_obs_vals) * 1.2 + 1), 4))
            _ax_k.bar(_x - _w / 2, _obs_vals, _w, label="Observed", color="k", alpha=0.7)
            _ax_k.bar(_x + _w / 2, _sim_med, _w, label="Simulated", color="steelblue", alpha=0.8)
            if _method == "ar" and len(_trajs) > 1:
                _ax_k.errorbar(
                    _x + _w / 2, _sim_med,
                    yerr=[_sim_med - _sim_lo, _sim_hi - _sim_med],
                    fmt="none", color="steelblue", capsize=4,
                )
            _ax_k.set_xticks(_x)
            _ax_k.set_xticklabels(_row_labels, rotation=30, ha="right", fontsize=9)
            _ax_k.set_ylabel("Proportion" if _mode_k == "proportion" else "Total count")
            _ax_k.set_title(f"{_label} ({'proportion' if _mode_k == 'proportion' else 'scalar total'})")
            _ax_k.legend()
            _ax_k.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            _figs.append(_fig_k)

        else:
            # Timeseries plot
            _obs_arr = np.array(_obs_k)
            _num_days = len(_obs_arr)
            _days = np.arange(_num_days)
            _fig_k, _ax_k = plt.subplots(figsize=(10, 4))

            _valid_trajs = [_t[_traj_key] for _t in _trajs if _traj_key in _t]
            _n_valid = len(_valid_trajs)
            if _method in ("adam", "lbfgs") and _n_valid <= 1:
                # Single replication: just draw the fitted curve
                if _valid_trajs:
                    _sim = np.array(_valid_trajs[0])
                    _ax_k.plot(_days, _sim[:_num_days], color="steelblue", linewidth=2, label="Fitted")
                _ax_k.plot(_days, _obs_arr, "ko", markersize=4, label="Observed", zorder=5)
                _ax_k.set_title(f"{_label} — Fitted vs Observed")
            else:
                # Multiple replications or AR accepted runs: spaghetti or band
                _style = fit_comparison_style.value
                _run_label = "accepted" if _method == "ar" else "replication"
                if _valid_trajs:
                    _trajs_arr = np.array([_t[:_num_days] for _t in _valid_trajs])
                    if _style == "spaghetti":
                        for _traj in _trajs_arr:
                            _ax_k.plot(
                                _days, _traj,
                                color="steelblue",
                                alpha=min(1.0, 3.0 / len(_trajs_arr)),
                                linewidth=1,
                            )
                        _ax_k.plot([], [], color="steelblue", alpha=0.6, linewidth=1.5,
                                   label=f"{_n_valid} {_run_label}(s)")
                    else:
                        _med = np.median(_trajs_arr, axis=0)
                        _lo = np.percentile(_trajs_arr, 2.5, axis=0)
                        _hi = np.percentile(_trajs_arr, 97.5, axis=0)
                        _ax_k.fill_between(_days, _lo, _hi, color="steelblue", alpha=0.25, label="95% CI")
                        _ax_k.plot(_days, _med, color="steelblue", linewidth=2, label="Median")
                _ax_k.plot(_days, _obs_arr, "ko", markersize=4, label="Observed", zorder=5)
                _ax_k.set_title(f"{_label} ({_n_valid} {_run_label}(s))")

            _ax_k.set_xlabel("Day")
            _ax_k.set_ylabel(_label)
            _ax_k.legend()
            _ax_k.grid(True, alpha=0.3)
            plt.tight_layout()
            _figs.append(_fig_k)

    mo.vstack([
        mo.md("### Fitted vs Observed"),
        _style_ui,
        *_figs,
    ])
    return


@app.cell
def _fitting_pairplot(fit_result, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))

    _accepted = fit_result.accepted_params
    mo.stop(not _accepted or len(_accepted) <= 1, mo.md(""))

    _first = _accepted[0]
    _scalar_keys = [_k for _k, _v in _first.items() if isinstance(_v, (int, float))]
    mo.stop(not _scalar_keys, mo.md(""))

    _data = np.array([[float(_s.get(_k, float("nan"))) for _k in _scalar_keys] for _s in _accepted])
    _n = len(_scalar_keys)

    def _draw_density(_ax, _vals):
        _vals = _vals[np.isfinite(_vals)]
        if len(_vals) < 2:
            return
        _ax.hist(_vals, bins=max(10, len(_vals) // 5), density=True,
                 color="steelblue", alpha=0.65, edgecolor="white", linewidth=0.4)
        try:
            from scipy.stats import gaussian_kde as _kde
            _xs = np.linspace(_vals.min(), _vals.max(), 300)
            _ax.plot(_xs, _kde(_vals)(_xs), color="navy", linewidth=1.8)
        except Exception:
            pass

    if _n == 1:
        _fig, _ax = plt.subplots(figsize=(5, 4))
        _draw_density(_ax, _data[:, 0])
        _ax.set_xlabel(_scalar_keys[0])
        _ax.set_ylabel("Density")
        _run_noun = "accepted" if fit_result.method == "ar" else "replication"
        _ax.set_title(f"Parameter distribution ({len(_accepted)} {_run_noun}(s))")
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        _run_noun = "accepted" if fit_result.method == "ar" else "replication"
        _fig, _axs = plt.subplots(_n, _n, figsize=(3 * _n, 3 * _n))
        _alpha_sc = min(0.8, 30.0 / max(len(_accepted), 1))
        for _row in range(_n):
            for _col in range(_n):
                _ax = _axs[_row, _col]
                if _row == _col:
                    _draw_density(_ax, _data[:, _row])
                else:
                    _ax.scatter(
                        _data[:, _col], _data[:, _row],
                        alpha=_alpha_sc, s=14,
                        color="steelblue", edgecolors="none",
                    )
                _ax.grid(True, alpha=0.2)
                if _row == _n - 1:
                    _ax.set_xlabel(_scalar_keys[_col], fontsize=9)
                else:
                    _ax.tick_params(labelbottom=False)
                if _col == 0:
                    _ax.set_ylabel(_scalar_keys[_row], fontsize=9)
                else:
                    _ax.tick_params(labelleft=False)
        _fig.suptitle(
            f"Parameter distributions ({len(_accepted)} {_run_noun}(s))",
            y=1.01, fontsize=11,
        )
        plt.tight_layout()

    mo.vstack([
        mo.md("### Accepted Parameter Distributions"),
        _fig,
    ])
    return


# ============================================================
# Forecast tab
# ============================================================

