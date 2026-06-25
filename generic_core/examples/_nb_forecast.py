# _nb_forecast.py
# Section: Forecast tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py

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
    fit_result, mo, main_tab, json, Path,
):
    mo.stop(main_tab.value != "Forecast", None)
    _path_w = forecast_params_path if not forecast_use_fitted.value else mo.md("")

    # Multi-set note — shown when AR or gradient replications produced multiple param sets
    _ar_note = mo.md("")
    _is_ar_fitted = False
    _n_accepted = 0
    _fit_method_display = "ar"
    if forecast_use_fitted.value:
        _is_ar_fitted = (
            fit_result is not None
            and len(fit_result.accepted_params) > 1
        )
        if _is_ar_fitted:
            _n_accepted = len(fit_result.accepted_params)
            _fit_method_display = fit_result.method
    else:
        _pp = forecast_params_path.value.strip()
        if _pp:
            try:
                with open(Path(_pp).expanduser()) as _f:
                    _loaded_preview = json.load(_f)
                _loaded_accepted = _loaded_preview.get("accepted_params", [])
                if len(_loaded_accepted) > 1:
                    _is_ar_fitted = True
                    _n_accepted = len(_loaded_accepted)
                    _fit_method_display = _loaded_preview.get("method", "ar")
            except Exception:
                pass

    if _is_ar_fitted:
        _reps_val = int(forecast_n_reps.value)
        _set_noun = (
            "accepted parameter set(s)" if _fit_method_display == "ar"
            else "fitted replication(s)"
        )
        _method_label = (
            "accept-reject" if _fit_method_display == "ar"
            else _fit_method_display
        )
        _set_header = f"**{_n_accepted} {_set_noun}** from {_method_label} fitting."
        if _reps_val < _n_accepted:
            _ar_note = mo.callout(
                mo.md(
                    f"{_set_header} "
                    f"Replicates ({_reps_val}) < {_n_accepted} sets — **{_reps_val}** sets will be "
                    "sampled without replacement, each seeding one trajectory."
                ),
                kind="success",
            )
        else:
            _base = _reps_val // _n_accepted
            _extra = _reps_val % _n_accepted
            if _extra == 0:
                _ar_note_text = (
                    f"{_set_header} "
                    f"Each will seed **{_base}** replicate(s) — **{_reps_val}** trajectories total."
                )
            else:
                _ar_note_text = (
                    f"{_set_header} "
                    f"Each will seed **{_base}** replicate(s); **{_extra}** set(s) sampled without "
                    f"replacement will run one extra — **{_reps_val}** trajectories total."
                )
            _ar_note = mo.callout(mo.md(_ar_note_text), kind="success")

    _fitted_state_note = mo.md("")
    if forecast_from_fitted_state.value:
        if fit_result is not None and forecast_use_fitted.value:
            if not _is_ar_fitted:
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
                    "**Start from fitted end-state** requires either:\n\n"
                    "- **Use fitted params from Fitting tab** enabled with fitting already run, or\n"
                    "- A JSON file (via the path field) exported from the Fitting tab, "
                    "which includes `num_days`, `method`, and `accepted_params`."
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
        _ar_note,
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
    build_compartment_init, start_date_input, timesteps, rng_seed,
    transition_vars_input,
    make_single_pop_metapop, make_metapop_from_folder, extract_history,
    np, json, mo, Path, build_scalar_array, datetime,
):
    forecast_result = None
    if forecast_run_button.value:
        _fitted_params = {}
        _fit_meta = {"num_days": 0, "method": "ar", "accepted_params": []}
        if forecast_use_fitted.value:
            mo.stop(
                fit_result is None,
                mo.callout(mo.md("**No fitting results.** Run fitting first or disable 'Use fitted params'."), kind="warn"),
            )
            _fitted_params = dict(fit_result.best_params)
            _fit_meta = {
                "num_days": fit_result.num_days,
                "method": fit_result.method,
                "accepted_params": list(fit_result.accepted_params),
            }
        else:
            _pp = forecast_params_path.value.strip()
            if _pp:
                try:
                    with open(Path(_pp).expanduser()) as _f:
                        _loaded = json.load(_f)
                    if "best_params" in _loaded:
                        _fitted_params = _loaded["best_params"]
                        _fit_meta = {
                            "num_days": _loaded.get("num_days", 0),
                            "method": _loaded.get("method", "ar"),
                            "accepted_params": _loaded.get("accepted_params", []),
                        }
                    else:
                        _fitted_params = _loaded
                except Exception as _exc:
                    mo.stop(True, mo.callout(mo.md(f"**Could not load fitted params:** {_exc}"), kind="danger"))

        _fit_n = _fit_meta["num_days"]
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
            # Initial conditions from the Step 6 tables via config_dict.
            _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
            _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((1, 1))), dtype=float)
            _seed_arrays = {
                _c: np.asarray(_a, dtype=float)
                for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
                if _c in compartments
            }
            _ci, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)

        _histories = []

        # Build param sets and run schedule (shared by both forecast paths)
        _param_sets = _fit_meta["accepted_params"] if _fit_meta["accepted_params"] else [_fitted_params]
        _n_accepted = len(_param_sets)
        _is_ar = _n_accepted > 1  # distribute across AR accepted sets OR gradient replications
        _rng_sched = np.random.default_rng(_seed_b)
        if _is_ar and _reps < _n_accepted:
            _selected = _rng_sched.choice(_n_accepted, size=_reps, replace=False)
            _run_schedule = [(int(_i), 0) for _i in _selected]
        elif _is_ar:
            _base = _reps // _n_accepted
            _extra = _reps % _n_accepted
            _run_schedule = [(_i, _r) for _i in range(_n_accepted) for _r in range(_base)]
            if _extra > 0:
                _extra_idx = _rng_sched.choice(_n_accepted, size=_extra, replace=False)
                _run_schedule += [(int(_i), _base) for _i in _extra_idx]
        else:
            _run_schedule = [(0, _r) for _r in range(_reps)]

        if forecast_from_fitted_state.value:
            # Two-phase: warmup through fit period → extract end-state → run forecast
            mo.stop(
                _fit_n == 0,
                mo.callout(mo.md("**Start from fitted end-state** requires a non-zero fit period. Run fitting first, or load a JSON exported from the Fitting tab (which includes `num_days`)."), kind="warn"),
            )

            _metric_names = [_m["name"] for _m in config_dict.get("epi_metrics", [])]
            _fcast_start = (
                datetime.datetime.strptime(_start, "%Y-%m-%d")
                + datetime.timedelta(days=_fit_n)
            ).strftime("%Y-%m-%d")

            def _extract_end_states(metapop_model, comps, metric_names):
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
                    _warmup_cache = {}
                    for _traj_idx, (_pset_idx, _rep) in enumerate(_run_schedule):
                        _pset = _param_sets[_pset_idx]

                        # Phase 1: deterministic warmup — cached per param set
                        if _pset_idx not in _warmup_cache:
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
                            _warmup_cache[_pset_idx] = (
                                extract_history(_wm, list(compartments), tvs=_tvs),
                                _extract_end_states(_wm, list(compartments), _metric_names),
                            )
                        _warmup_hist, _end_states = _warmup_cache[_pset_idx]

                        # Phase 2: stochastic forecast from end-state
                        if not is_metapop:
                            _end_comp, _end_epi = _end_states[0]
                            _fm, _, _ = make_single_pop_metapop(
                                config_dict, _fcast_start, _horizon, _end_comp,
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                epi_metric_init=_end_epi or None,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _fm, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _fcast_start, _horizon,
                                list(compartments),
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
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
            # Standard path: run from Step 7 initial conditions, using run schedule for param sets
            with mo.status.spinner("Running forecast..."):
                try:
                    for _traj_idx, (_pset_idx, _rep) in enumerate(_run_schedule):
                        _pset = _param_sets[_pset_idx]
                        if not is_metapop:
                            _m, _, _ = make_single_pop_metapop(
                                config_dict, _start, _total_days, _ci,
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_pset or None,
                                travel_config=metapop_travel_config,
                            )
                        else:
                            _m, _ = make_metapop_from_folder(
                                metapop_folder_input.value, config_dict, _start, _total_days, list(compartments),
                                seed_offset=_traj_idx, seed_base=_seed_b, ts_per_day=_ts,
                                stochastic=_stoch, tvs=_tvs, save_daily=True,
                                param_overrides=_pset or None,
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
def _forecast_chart_style_ui(mo):
    forecast_chart_style = mo.ui.radio(
        options={"Median + 95% CI": "band", "Spaghetti lines": "spaghetti"},
        value="Median + 95% CI",
        label="Chart style",
    )
    return (forecast_chart_style,)


@app.cell
def _forecast_results_display(forecast_result, forecast_chart_style, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Forecast", None)
    mo.stop(forecast_result is None, mo.md("*Run forecast to see results.*"))
    _hists = forecast_result["histories"]
    _comps = forecast_result["compartments"]
    _fit_n = forecast_result["fit_n_days"]
    _total = forecast_result["total_days"]
    _days = np.arange(1, _total + 1)
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _style = forecast_chart_style.value
    _fig, _ax = plt.subplots(figsize=(12, 5))
    for _ci, _comp in enumerate(_comps):
        _color = _colors[_ci % len(_colors)]
        _arrays = [_h[_comp] for _h in _hists if _comp in _h]
        if not _arrays:
            continue
        _mat = np.stack(_arrays, axis=0)
        _n = min(len(_days), _mat.shape[1])
        if _style == "spaghetti":
            for _row in _mat:
                _ax.plot(_days[:_n], _row[:_n], color=_color, linewidth=0.8, alpha=0.4)
            _med = np.median(_mat[:, :_n], axis=0)
            _ax.plot(_days[:_n], _med, color=_color, linewidth=2, label=f"{_comp} (median)")
        else:
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
        forecast_chart_style,
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

