# _nb_shared_factory.py
# Section: Shared model factory functions
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _shared_model_factory(
    build_notebook_schedules_input,
    build_scalar_array,
    read_initial_conditions,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt, flu, np, json, pd, Path,
    loaded_schedule_dfs,
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
            absolute_humidity=0.0,  # CSV-only: the humidity df is always supplied when used
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
            # Precedence: explicit override > Step 6 tables (with seeds) > folder
            # JSON > population-only table > zeros.
            _ic_cfg = (config.get("initial_conditions", {}) or {}).get(_sp_name, {})
            _has_table_seeds = bool(_ic_cfg.get("seeds"))
            _table_ci = read_initial_conditions(config, _sp_name, compartments_list, _A, _R)
            if init_states_override is not None and _si < len(init_states_override):
                _comp_init, _epi_init = init_states_override[_si]
            elif _has_table_seeds and _table_ci is not None:
                _comp_init = _table_ci
            elif _ic_p.exists():
                with open(_ic_p) as _f:
                    _ic = json.load(_f)
                for _c, _arr in _ic.get("compartments", {}).items():
                    if _c in compartments_list:
                        _comp_init[_c] = np.array(_arr, dtype=float)
                for _m, _arr in _ic.get("epi_metrics", {}).items():
                    _epi_init[_m] = np.array(_arr, dtype=float)
            elif _table_ci is not None:
                _comp_init = _table_ci
            _sp_overrides = dict(param_overrides or {})
            if param_overrides_per_subpop and _si < len(param_overrides_per_subpop) and param_overrides_per_subpop[_si]:
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

