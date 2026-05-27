# _nb_fitting.py
# Section: Fitting tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py

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

