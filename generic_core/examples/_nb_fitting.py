# _nb_fitting.py
# Section: Fitting tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _fit_n_targets_state(mo):
    # Ordered list of active slot indices into the fixed pool of 20
    # fit_target_* UI arrays below. Using slot ids (rather than a plain
    # count) lets a target be removed from anywhere in the list, not just
    # the end — marimo UI element values can't be reassigned from Python
    # (only through user interaction), so removing a target just drops its
    # slot id from this list instead of shifting values between slots.
    get_target_slots, set_target_slots = mo.state([0])
    return get_target_slots, set_target_slots


@app.cell
def _fit_target_buttons(
    mo, get_target_slots, set_target_slots,
    get_bulk_file_data, set_bulk_file_data,
    get_restored_target_data, set_restored_target_data,
):
    def _add_target(_):
        _cur = get_target_slots()
        if len(_cur) >= 20:
            return
        _unused = next(_i for _i in range(20) if _i not in _cur)
        set_target_slots(_cur + [_unused])

    def _remove_target(_slot):
        def _remove(_):
            _cur = get_target_slots()
            if len(_cur) > 1:
                set_target_slots([s for s in _cur if s != _slot])
                # Drop the removed slot's leftover bulk/restored data too —
                # otherwise it's still sitting in these dicts under that slot
                # id, and a later "+ Add target" reusing the same freed slot
                # number silently inherits it instead of starting empty.
                if _slot in get_bulk_file_data():
                    _new_bulk = dict(get_bulk_file_data())
                    del _new_bulk[_slot]
                    set_bulk_file_data(_new_bulk)
                if _slot in get_restored_target_data():
                    _new_restored = dict(get_restored_target_data())
                    del _new_restored[_slot]
                    set_restored_target_data(_new_restored)
        return _remove

    add_target_btn = mo.ui.button(label="+ Add target", on_click=_add_target)
    # Pre-built pool of 20 remove buttons, one per fit_target_* slot — mirrors
    # how fit_target_src/fit_target_weight/etc. are built as mo.ui.array in a
    # dedicated cell rather than ad hoc inside the display loop. Buttons
    # created inline inside _fitting_display's loop never get wired up to the
    # reactive graph (no cell statically references them), so clicks silently
    # do nothing; building them here, indexed like every other target widget,
    # is what makes the click actually register.
    fit_target_remove_btn = mo.ui.array([
        mo.ui.button(
            label="✕",
            tooltip="Remove this target",
            on_click=_remove_target(_i),
        )
        for _i in range(20)
    ])
    return add_target_btn, fit_target_remove_btn


@app.cell
def _fit_bulk_upload_state(mo):
    # {slot_id: {"name": str, "contents": bytes}} for targets created by the
    # bulk-upload widget above — kept separate from fit_target_upload because
    # each slot's own mo.ui.file can only ever hold that one slot's file, and
    # UI element values can't be assigned from Python (only through user
    # interaction), so a bulk-uploaded file can't be pushed into another
    # slot's uploader widget.
    get_bulk_file_data, set_bulk_file_data = mo.state({})
    # {identity (file-name tuple) -> [slot_id-or-None per file]}, so a batch
    # already turned into targets isn't recreated when this reactive cell
    # re-runs for an unrelated reason (e.g. another target being removed
    # changes get_target_slots()) — but IS recreated per-file if the user
    # deletes one of its targets and then re-selects the same file(s).
    get_bulk_batches, set_bulk_batches = mo.state({})
    return get_bulk_file_data, set_bulk_file_data, get_bulk_batches, set_bulk_batches


@app.cell
def _fit_config_restore_state(mo):
    # Parsed fit_config.json from a "Restore a saved configuration" upload —
    # None until a file is uploaded. Read (never mutated in place) by the
    # scalar hyperparameter widget cells below, each of which recreates
    # itself using this as its default whenever it reruns, and by the
    # per-target override logic in _fitting_build_request/_fitting_obs_parse,
    # which follows the same "applied only while a field still shows its
    # untouched default" precedent as the bulk-CSV-upload defaults above.
    get_restored_config, set_restored_config = mo.state(None)
    # {slot_id: {...}} restored-target data, keyed like get_bulk_file_data —
    # kept separate since a restored target has no real uploaded CSV, only
    # the already-parsed observed/point_weights arrays saved in the file.
    get_restored_target_data, set_restored_target_data = mo.state({})
    get_restore_error, set_restore_error = mo.state(None)
    return (
        get_restored_config, set_restored_config,
        get_restored_target_data, set_restored_target_data,
        get_restore_error, set_restore_error,
    )


@app.cell
def _fit_bulk_defaults_ui(
    mo, compartments, n_transitions, t_name,
    is_metapop, metapop_folder_input, json, Path, num_risk_groups,
):
    # Shared defaults applied to every target created by the next bulk-upload
    # batch (snapshotted into each target's bulk-data entry at upload time —
    # see _fit_bulk_upload_ui). Mirrors the per-target options (compartments +
    # named transition variables for "vars"; subpop/risk option lists match
    # _fitting_ui's) — kept as its own small cell, rather than reusing
    # _fitting_ui's copies, so these widgets don't depend on the rest of that
    # larger cell.
    _tvars = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    _all_tgts = list(compartments) + _tvars
    _tgt_opts = _all_tgts if _all_tgts else ["S"]
    fit_bulk_vars = mo.ui.multiselect(
        options={t: t for t in _tgt_opts},
        value=[_tgt_opts[0]],
        label="Variables (summed) for bulk-added targets",
    )

    fit_bulk_weight = mo.ui.number(
        value=1.0, start=0.0, stop=1000.0, step=None,
        label="Weight λ for bulk-added targets",
    )

    _subpop_names_bulk = []
    if is_metapop and metapop_folder_input.value.strip():
        try:
            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                _subpop_names_bulk = json.load(_f).get("subpopulations", [])
        except Exception:
            _subpop_names_bulk = []
    _sp_opts = ["All (sum)"] + [f"{_i}: {_nm}" for _i, _nm in enumerate(_subpop_names_bulk)]
    _risk_opts = ["All (sum)"] + [str(_i) for _i in range(int(num_risk_groups))]

    fit_bulk_subpop = mo.ui.dropdown(
        options=_sp_opts, value=_sp_opts[0], label="Subpopulation for bulk-added targets",
    )
    fit_bulk_risk = mo.ui.dropdown(
        options=_risk_opts, value=_risk_opts[0], label="Risk group for bulk-added targets",
    )
    return (fit_bulk_vars, fit_bulk_weight, fit_bulk_subpop, fit_bulk_risk)


@app.cell
def _fitting_ui(
    mo, compartments, n_transitions, t_name, param_names,
    is_metapop, metapop_folder_input, json, Path,
    num_age_groups, num_risk_groups, get_restored_config,
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

    # Restored fit_config.json values (see _fit_config_upload_ui): the target
    # widgets themselves (per-slot arrays below) can't be reassigned once
    # rendered, so restored target data is applied later via override, not
    # here — but fit_params_multiselect, fit_method, fit_r2_thresh, and
    # fit_sim_days_input are each a single widget recreated fresh whenever
    # this cell reruns, so their restored value can be set directly.
    _restored = get_restored_config()
    _restored_fc = (_restored or {}).get("fit_config", {}) if _restored else {}

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
        value=int(_restored_fc.get("sim_days", 180)), start=1, stop=3650, step=1,
        label="Simulation days (used when all targets are scalar totals)",
    )
    _seed_scale_opts = {
        f"seed_scale_{_c}": f"seed_scale_{_c}"
        for _c in compartments[1:]
    }
    # Scale-group multipliers (synthetic params like "ihr_scale") are saved in
    # selected_params too but aren't valid multiselect options — those are
    # restored separately by _fitting_scale_groups_ui/_fields.
    _valid_param_opts = {**{p: p for p in param_names}, **_seed_scale_opts}
    _restored_params = [
        _p for _p in _restored_fc.get("selected_params", []) if _p in _valid_param_opts
    ]
    fit_params_multiselect = mo.ui.multiselect(
        options=_valid_param_opts,
        value=_restored_params,
        label="Parameters to fit",
    )
    _method_opts = {
        "Adam (gradient)": "adam",
        "L-BFGS (gradient)": "lbfgs",
        "Accept-reject": "ar",
        "MCMC (emcee)": "mcmc",
        "ABC-SMC (pyabc)": "abc-smc",
    }
    _method_labels_by_val = {_v: _k for _k, _v in _method_opts.items()}
    _restored_method_val = _restored_fc.get("method") or ""
    _restored_method_label = _method_labels_by_val.get(_restored_method_val)
    fit_method = mo.ui.radio(
        options=_method_opts,
        value=_restored_method_label or "Adam (gradient)", label="Fitting method",
    )
    fit_r2_thresh = mo.ui.number(
        value=float(_restored_fc.get("r2_threshold", 0.75)),
        start=0.0, stop=1.0, step=None, label="R² acceptance threshold",
    )
    fit_run_button = mo.ui.run_button(label="Run fitting")

    return (
        fit_target_src, fit_target_upload, fit_target_path,
        fit_target_vars, fit_target_mode, fit_target_weight,
        fit_target_subpop, fit_target_age, fit_target_risk,
        fit_sim_days_input,
        fit_params_multiselect,
        fit_method, fit_r2_thresh, fit_run_button,
    )


@app.cell
def _fit_bulk_upload_ui(
    mo, fit_bulk_vars, fit_bulk_weight, fit_bulk_subpop, fit_bulk_risk,
    get_target_slots, set_target_slots,
    get_bulk_file_data, set_bulk_file_data,
    get_bulk_batches, set_bulk_batches,
    fit_target_upload, fit_target_path, fit_target_mode, fit_target_vars,
    fit_target_weight, fit_target_subpop, fit_target_age, fit_target_risk,
    compartments, age_groups, num_age_groups, re,
):
    # Named age bands (e.g. "13-17", "65+") are matched against each
    # uploaded filename to auto-set that target's age group, so per-age-band
    # CSVs (like MA_flu_daily_hospitalizations_13_17.csv) don't need manual
    # dropdown selection. Tokens are normalized to match the underscore-joined
    # style produced by split_hospitalizations_by_age.py; sorted longest-first
    # so e.g. "5-12" can't match inside "50-64"'s token. A filename matching
    # zero or more-than-one band is left ambiguous (no guess).
    def _age_band_tokens():
        if not age_groups or len(age_groups) != int(num_age_groups):
            return []
        _toks = [
            (_idx, str(_band).strip().lower().replace("-", "_").replace("+", "plus"))
            for _idx, _band in enumerate(age_groups)
        ]
        _toks = [(_idx, _tok) for _idx, _tok in _toks if _tok]
        _toks.sort(key=lambda _t: len(_t[1]), reverse=True)
        return _toks

    def _match_age_idx(_fname):
        _norm = _fname.lower()
        _matches = {
            _idx for _idx, _tok in _age_band_tokens()
            if re.search(r"(?<![0-9a-z])" + re.escape(_tok) + r"(?![0-9a-z])", _norm)
        }
        return next(iter(_matches)) if len(_matches) == 1 else None
    # Runs only on a genuine file-selection event from the browser (mo.ui.file
    # calls on_change from its own _update(), never from an unrelated cell
    # rerun) — NOT as a reactive read of fit_bulk_upload.value in a normal
    # cell body. That distinction matters: this cell's own inputs include
    # fit_target_* (needed for the slot-0-reclaim check), so if it read
    # .value passively instead, deleting ANY target — bulk-created or not —
    # would re-trigger it with the *same*, unchanged file list and silently
    # recreate the very target the user just removed.
    def _on_bulk_upload(_files):
        _files = _files or ()
        _identity = tuple(_f.name for _f in _files)
        if not _identity:
            return
        _cur_slots = list(get_target_slots())
        _batches = dict(get_bulk_batches())
        _recorded = list(_batches.get(_identity, []))
        _recorded += [None] * (len(_files) - len(_recorded))

        # Nothing to do if every file in this batch still has a live target.
        # If the user deleted one of the batch's targets and re-selected the
        # same file(s), its recorded slot is no longer in _cur_slots, so
        # this is False and that one file gets a fresh slot.
        if all(_s is not None and _s in _cur_slots for _s in _recorded):
            return

        _new_slots = list(_cur_slots)
        _new_data = dict(get_bulk_file_data())
        _vars = list(fit_bulk_vars.value)
        _weight = float(fit_bulk_weight.value)
        _subpop = fit_bulk_subpop.value
        _risk = fit_bulk_risk.value

        # Reclaim slot 0 (the default Target 1) for a bulk-uploaded file if
        # it's still exactly at its untouched defaults — otherwise bulk
        # upload always leaves that empty target sitting alongside the new
        # ones. Any real customization (a file, a non-default mode/vars/
        # weight/slice) disqualifies it, so we never clobber user work. If
        # slot 0 was already deleted, it's simply absent from _new_slots and
        # gets picked up below like any other free slot (lowest-numbered
        # first).
        _default_var = [compartments[0]] if compartments else ["S"]
        _slot0_untouched = (
            0 in _new_slots
            and not fit_target_upload.value[0]
            and not fit_target_path.value[0].strip()
            and fit_target_mode.value[0] == "ts"
            and list(fit_target_vars.value[0]) == _default_var
            and float(fit_target_weight.value[0]) == 1.0
            and fit_target_subpop.value[0] == "All (sum)"
            and fit_target_age.value[0] == "All (sum)"
            and fit_target_risk.value[0] == "All (sum)"
        )

        for _i, _f in enumerate(_files):
            if _recorded[_i] is not None and _recorded[_i] in _new_slots:
                continue  # this file's target is still alive

            _age_idx = _match_age_idx(_f.name)

            if _i == 0 and _slot0_untouched:
                # Slot 0 is already in _new_slots (it's an active, untouched
                # target) — fill its data in place, don't append it again.
                _new_data[0] = {
                    "name": _f.name, "contents": _f.contents, "vars": _vars, "age_idx": _age_idx,
                    "weight": _weight, "subpop": _subpop, "risk": _risk,
                }
                _recorded[_i] = 0
                continue

            # Otherwise: pick the lowest free slot. If slot 0 was deleted
            # (absent from _new_slots), it's naturally picked up here too.
            _unused = next((_j for _j in range(20) if _j not in _new_slots), None)
            if _unused is None:
                break
            _new_slots.append(_unused)
            _new_data[_unused] = {
                "name": _f.name, "contents": _f.contents, "vars": _vars, "age_idx": _age_idx,
                "weight": _weight, "subpop": _subpop, "risk": _risk,
            }
            _recorded[_i] = _unused

        _batches[_identity] = _recorded
        set_target_slots(_new_slots)
        set_bulk_file_data(_new_data)
        set_bulk_batches(_batches)

    # Selecting several files here auto-creates one target per file, using
    # the fit_bulk_* widgets above (vars/weight/subpop/risk) as that batch's
    # shared defaults, plus any age group auto-detected from each filename —
    # the user then tweaks per target to override.
    fit_bulk_upload = mo.ui.file(
        label="Bulk-add targets from CSVs (select multiple files)",
        filetypes=[".csv"],
        multiple=True,
        on_change=_on_bulk_upload,
    )
    return (fit_bulk_upload,)


@app.cell
def _fit_config_upload_ui(
    mo, json,
    set_target_slots,
    set_restored_target_data,
    set_restored_config, set_restore_error,
):
    # Restores a fit_config.json previously downloaded from this tab (see
    # _fitting_export_display): replaces whatever targets are currently
    # configured with one target per saved target (data + vars/mode/weight/
    # subpop/age/risk) and, via get_restored_config, repopulates the
    # parameters-to-fit, bounds, method, and its hyperparameters below.
    #
    # A restored target's observed data is the exact numbers saved in the
    # file, not a live re-read of any CSV — if the source CSV has since
    # changed, re-upload it to that target to refresh it (see the caveat
    # callout shown next to this widget).
    def _on_config_upload(_files):
        _files = _files or ()
        if not _files:
            return
        try:
            _raw = json.loads(_files[0].contents.decode())
            _targets = _raw["targets"]
        except Exception as _exc:
            set_restore_error(str(_exc))
            return

        # Restoring a saved configuration replaces whatever targets are
        # currently configured (manual, bulk-uploaded, or from an earlier
        # restore) rather than adding to them — unlike bulk CSV upload,
        # which is additive by design (see _fit_bulk_upload_ui). So, unlike
        # that cell, this one starts from an empty slot list instead of the
        # current one.
        _new_slots = []
        _new_data = {}

        for _i, _t in enumerate(_targets):
            if _i >= 20:
                break
            _entry = {
                "label": _t.get("label", f"Restored {_i + 1}"),
                "vars": _t.get("variables"),
                "mode": _t.get("mode"),
                "weight": _t.get("weight"),
                "subpop_idx": _t.get("subpop_idx"),
                "age_idx": _t.get("age_idx"),
                "risk_idx": _t.get("risk_idx"),
                "observed": _t.get("observed"),
                "point_weights": _t.get("point_weights"),
            }
            _new_slots.append(_i)
            _new_data[_i] = _entry

        set_target_slots(_new_slots)
        set_restored_target_data(_new_data)
        set_restore_error(None)
        set_restored_config(_raw)

    fit_config_upload = mo.ui.file(
        label="Restore a saved configuration (fit_config.json)",
        filetypes=[".json"],
        multiple=False,
        on_change=_on_config_upload,
    )
    return (fit_config_upload,)


@app.cell
def _fitting_iter_ui(mo, fit_method, get_restored_config):
    # "Iterations / Max samples" is shared across methods but its sensible default
    # differs: MCMC needs many steps per walker to converge, whereas the gradient
    # and accept-reject methods work fine with a few hundred. Recreated on method
    # change so switching to MCMC bumps the default up to a usable value. A
    # restored config's saved value takes precedence over the method-based default.
    _restored_n_iter = (get_restored_config() or {}).get("fit_config", {}).get("n_iter")
    _iter_default = _restored_n_iter if _restored_n_iter is not None else (
        1500 if fit_method.value == "mcmc" else 200
    )
    fit_n_iter = mo.ui.number(
        value=_iter_default, start=10, stop=20000, step=10,
        label="Iterations / Max samples")
    return (fit_n_iter,)


@app.cell
def _fitting_lr_ui(mo, fit_method, get_restored_config):
    _restored_lr = (get_restored_config() or {}).get("fit_config", {}).get("lr")
    _lr_default = _restored_lr if _restored_lr is not None else (
        0.5 if fit_method.value == "lbfgs" else 0.01
    )
    fit_lr = mo.ui.number(value=_lr_default, start=1e-5, stop=10.0, step=None, label="Learning rate")
    return (fit_lr,)


@app.cell
def _fitting_replications_ui(mo, get_restored_config):
    _restored_n_rep = (get_restored_config() or {}).get("fit_config", {}).get("n_replications")
    fit_n_replications = mo.ui.number(
        value=_restored_n_rep if _restored_n_rep is not None else 5, start=1, stop=200, step=1,
        label="Number of replications",
    )
    return (fit_n_replications,)


@app.cell
def _fitting_robust_ui(mo, get_restored_config):
    _restored_fc = (get_restored_config() or {}).get("fit_config", {})
    fit_robust_steps = mo.ui.checkbox(
        value=bool(_restored_fc.get("robust_steps", True)),
        label="Robust gradient steps (recommended)",
    )
    fit_parallel = mo.ui.checkbox(
        value=bool(_restored_fc.get("parallel", True)),
        label="Parallel replications (recommended)",
    )
    return (fit_robust_steps, fit_parallel)


@app.cell
def _fitting_bayes_ui(mo, get_restored_config):
    # Hyperparameters for the Bayesian samplers (MCMC / ABC-SMC).
    _restored_fc = (get_restored_config() or {}).get("fit_config", {})
    fit_n_walkers = mo.ui.number(
        value=int(_restored_fc.get("n_walkers", 32)), start=4, stop=500, step=1, label="Walkers (ensemble members)")
    fit_mcmc_burnin = mo.ui.number(
        value=int(_restored_fc.get("mcmc_burnin", 300)), start=0, stop=20000, step=50, label="Burn-in steps (discarded)")
    fit_mcmc_thin = mo.ui.number(
        value=int(_restored_fc.get("mcmc_thin", 10)), start=1, stop=500, step=1, label="Thinning (keep every k-th)")
    fit_abc_pop = mo.ui.number(
        value=int(_restored_fc.get("abc_pop_size", 200)), start=20, stop=5000, step=10, label="Population size (particles/generation)")
    fit_abc_gens = mo.ui.number(
        value=int(_restored_fc.get("abc_max_gens", 12)), start=1, stop=100, step=1, label="Max generations")
    # Time-varying transmission m(t)
    fit_tv_enable = mo.ui.checkbox(
        value=bool(_restored_fc.get("tv_transmission", False)), label="Fit time-varying transmission m(t)")
    fit_tv_spacing = mo.ui.number(
        value=int(_restored_fc.get("tv_knot_spacing_days", 30)), start=7, stop=180, step=1, label="Knot spacing (days)")
    fit_tv_tau = mo.ui.number(
        value=float(_restored_fc.get("tv_tau", 0.25)), start=0.01, stop=2.0, step=None, label="Smoothness τ (RW-prior sd)")
    return (
        fit_n_walkers, fit_mcmc_burnin, fit_mcmc_thin,
        fit_abc_pop, fit_abc_gens,
        fit_tv_enable, fit_tv_spacing, fit_tv_tau,
    )


@app.cell
def _fitting_bounds_ui(
    mo, fit_params_multiselect, config_dict,
    num_age_groups, num_risk_groups, is_metapop, get_restored_config,
):
    _saved_params = config_dict.get("params", {})
    _selected = list(fit_params_multiselect.value)
    _A = num_age_groups
    _R = num_risk_groups

    # Restored fit_config.json bounds/granularity/log-space flags (see
    # _fit_config_upload_ui) — this cell already recreates its widget arrays
    # whenever fit_params_multiselect.value changes (which happens on restore,
    # since that widget's own value is set directly there), so restored
    # per-param settings can be applied here as real defaults.
    _restored_fc = (get_restored_config() or {}).get("fit_config", {})
    _restored_bounds = _restored_fc.get("bounds", {})
    _restored_dims = _restored_fc.get("param_dims", {})
    _restored_log = set(_restored_fc.get("log_params", []))

    _dim_opts = []
    if _A > 1:
        _dim_opts.append("age groups")
    if _R > 1:
        _dim_opts.append("risk groups")
    if is_metapop:
        _dim_opts.append("subpopulation")

    def _default_bounds(pn):
        if pn in _restored_bounds:
            _lo, _hi = _restored_bounds[pn]
            return float(_lo), float(_hi)
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
            value=[_d for _d in _restored_dims.get(_pn, []) if _d in _dim_opts],
            label="Granularity",
        )
        for _pn in _selected
    ])
    # Per-param "fit in log10 space" toggle (LogUniform prior / log-space
    # sampler steps). Sensible default ON for seed-scale multipliers, which are
    # positive scale factors best explored multiplicatively. Requires positive
    # bounds; ignored for params given age/risk/subpop granularity.
    fit_bounds_log = mo.ui.array([
        mo.ui.checkbox(
            value=_pn in _restored_log or _pn.startswith("seed_scale_"),
            label="Fit in log space",
        )
        for _pn in _selected
    ])
    return (fit_bounds_lo, fit_bounds_hi, fit_param_dims, fit_bounds_log)


@app.cell
def _fitting_scale_groups_ui(mo, get_restored_config):
    # Linked-scale groups: one fitted multiplier scales several base params by
    # the same factor (preserving their config ratio). Pick how many groups; the
    # per-group fields are built in the next cell.
    _restored_sg = (get_restored_config() or {}).get("fit_config", {}).get("scale_groups", {})
    fit_n_scale_groups = mo.ui.number(
        value=len(_restored_sg), start=0, stop=10, step=1,
        label="Linked-scale groups (one multiplier scales several params)")
    return (fit_n_scale_groups,)


@app.cell
def _fitting_scale_groups_fields(
    mo, fit_n_scale_groups, param_names, get_restored_config,
    num_age_groups, num_risk_groups, is_metapop,
):
    _ng = int(fit_n_scale_groups.value)
    _opts = list(param_names)

    # This cell reruns whenever fit_n_scale_groups.value changes — which
    # happens on restore, since that widget's own value is set directly in
    # _fitting_scale_groups_ui — so restored group names/bases/bounds/log
    # flags (matched to this group's position in the saved scale_groups dict)
    # can be applied here as real widget defaults.
    _restored_fc = (get_restored_config() or {}).get("fit_config", {})
    _restored_sg_items = list(_restored_fc.get("scale_groups", {}).items())
    _restored_bounds = _restored_fc.get("bounds", {})
    _restored_dims = _restored_fc.get("param_dims", {})
    _restored_log = set(_restored_fc.get("log_params", []))

    _dim_opts = []
    if num_age_groups > 1:
        _dim_opts.append("age groups")
    if num_risk_groups > 1:
        _dim_opts.append("risk groups")
    if is_metapop:
        _dim_opts.append("subpopulation")

    def _sg_name(_i):
        return _restored_sg_items[_i][0] if _i < len(_restored_sg_items) else f"scale_{_i + 1}"

    def _sg_bases(_i):
        if _i < len(_restored_sg_items):
            return [_b for _b in _restored_sg_items[_i][1] if _b in _opts]
        return []

    def _sg_bounds(_i):
        _nm = _sg_name(_i)
        if _nm in _restored_bounds:
            _lo, _hi = _restored_bounds[_nm]
            return float(_lo), float(_hi)
        return 0.1, 2.0

    fit_sg_names = mo.ui.array([
        mo.ui.text(value=_sg_name(_i), label="Multiplier name")
        for _i in range(_ng)
    ])
    fit_sg_bases = mo.ui.array([
        mo.ui.multiselect(options=_opts, value=_sg_bases(_i), label="Scales these params")
        for _i in range(_ng)
    ])
    fit_sg_lo = mo.ui.array([
        mo.ui.number(start=1e-8, stop=1e8, step=None, value=_sg_bounds(_i)[0], label="Lower")
        for _i in range(_ng)
    ])
    fit_sg_hi = mo.ui.array([
        mo.ui.number(start=1e-8, stop=1e8, step=None, value=_sg_bounds(_i)[1], label="Upper")
        for _i in range(_ng)
    ])
    fit_sg_dims = mo.ui.array([
        mo.ui.multiselect(
            options=_dim_opts,
            value=[_d for _d in _restored_dims.get(_sg_name(_i), []) if _d in _dim_opts],
            label="Granularity",
        )
        for _i in range(_ng)
    ])
    fit_sg_log = mo.ui.array([
        mo.ui.checkbox(value=_sg_name(_i) in _restored_log, label="Fit in log space")
        for _i in range(_ng)
    ])
    return (fit_sg_names, fit_sg_bases, fit_sg_lo, fit_sg_hi, fit_sg_dims, fit_sg_log)


@app.cell
def _fitting_start_offset_ui(mo, get_restored_config):
    _restored_fc = (get_restored_config() or {}).get("fit_config", {})
    _restored_offset_bounds = _restored_fc.get("start_offset_bounds", [-30, 30])
    fit_start_offset_enable = mo.ui.checkbox(
        label="Fit epidemic start date offset",
        value=bool(_restored_fc.get("fit_start_offset", False)),
    )
    fit_start_offset_lo = mo.ui.number(
        value=int(_restored_offset_bounds[0]), start=-365, stop=0, step=1, label="Min offset (days)",
    )
    fit_start_offset_hi = mo.ui.number(
        value=int(_restored_offset_bounds[1]), start=0, stop=365, step=1, label="Max offset (days)",
    )
    return fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi


@app.cell
def _fitting_display(
    get_target_slots, add_target_btn, fit_target_remove_btn,
    fit_bulk_upload, fit_bulk_vars, fit_bulk_weight, fit_bulk_subpop, fit_bulk_risk,
    get_bulk_file_data,
    fit_config_upload, get_restored_target_data, get_restore_error,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_sim_days_input,
    fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims, fit_bounds_log,
    fit_n_scale_groups, fit_sg_names, fit_sg_bases, fit_sg_lo, fit_sg_hi, fit_sg_dims, fit_sg_log,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_method, fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications,
    fit_robust_steps, fit_parallel, fit_run_button,
    fit_n_walkers, fit_mcmc_burnin, fit_mcmc_thin,
    fit_abc_pop, fit_abc_gens,
    fit_tv_enable, fit_tv_spacing, fit_tv_tau,
    fit_upload_result, fit_upload_error,
    mo, main_tab,
    compartments,
    num_age_groups, num_risk_groups, is_metapop, age_groups,
    tip_label, wtip,
    step_header, section_card, CLT_ACCENT,
):
    mo.stop(main_tab.value != "Fitting", None)
    _ACC = CLT_ACCENT["fitting"]
    _slots = get_target_slots()
    _n = len(_slots)
    _selected_params = list(fit_params_multiselect.value)
    _any_non_ts = any(fit_target_mode.value[_k] in ("scalar", "proportion") for _k in _slots)
    _any_ts = any(fit_target_mode.value[_k] == "ts" for _k in _slots)

    _bulk_default_items = [fit_bulk_vars, fit_bulk_weight]
    if is_metapop:
        _bulk_default_items.append(fit_bulk_subpop)
    if int(num_risk_groups) > 1:
        _bulk_default_items.append(fit_bulk_risk)

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
    for _pos, _k in enumerate(_slots):
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        _bulk_entry = get_bulk_file_data().get(_k)
        _restore_entry = get_restored_target_data().get(_k)
        _fname_note = mo.md("")
        if _src == "upload" and fit_target_upload.value[_k]:
            _fname = fit_target_upload.value[_k][0].name
            _fname_note = mo.callout(mo.md(f"Loaded: **{_fname}**"), kind="info")

        _data_input = fit_target_upload[_k] if _src == "upload" else fit_target_path[_k]

        # A bulk-added target's loaded file, plus its vars/weight/subpop/risk
        # defaults (chosen via the fit_bulk_* widgets at upload time, since
        # UI element values can't be overwritten from code once rendered —
        # see fit_target_upload/_bulk_entry above) and the age group auto-
        # detected from the filename (see _fit_bulk_upload_ui._match_age_idx)
        # — the latter four applied by _fitting_build_request only while the
        # corresponding widget still shows its original default; picking
        # anything else here overrides that one field. A target restored from
        # a saved fit_config.json (see _fit_config_upload_ui) works the same
        # way, using _restore_entry instead of _bulk_entry — its observed
        # data is a frozen snapshot from the file, not a live CSV re-read.
        # Collected into a single callout (rather than one band per field) so
        # a target doesn't show a wall of separate notes.
        _bulk_lines = []
        if _src == "upload" and _bulk_entry:
            _bulk_lines.append(f"File: **{_bulk_entry['name']}**")
        elif _src == "upload" and _restore_entry:
            _bulk_lines.append(f"Target: **{_restore_entry.get('label', '')}**")
            _bulk_lines.append(
                "observed data is a frozen snapshot from the file — re-upload the "
                "CSV here to pick up any changes made since it was saved"
            )
        _default_var = [compartments[0]] if compartments else ["S"]
        if (
            _bulk_entry and _bulk_entry.get("vars")
            and list(fit_target_vars.value[_k]) == _default_var
        ):
            _bulk_lines.append(f"Variables: **{', '.join(_bulk_entry['vars'])}**")
        elif (
            _restore_entry and _restore_entry.get("vars")
            and list(fit_target_vars.value[_k]) == _default_var
        ):
            _bulk_lines.append(f"Variables: **{', '.join(_restore_entry['vars'])}**")

        if (
            _restore_entry and _restore_entry.get("mode")
            and _restore_entry["mode"] != "ts"
            and fit_target_mode.value[_k] == "ts"
        ):
            _bulk_lines.append(f"Observed data type: **{_restore_entry['mode']}**")

        if (
            _bulk_entry and _bulk_entry.get("age_idx") is not None
            and fit_target_age.value[_k] == "All (sum)"
            and int(num_age_groups) > 1
        ):
            _bulk_age_idx = _bulk_entry["age_idx"]
            _band_label = (
                age_groups[_bulk_age_idx]
                if age_groups and len(age_groups) == int(num_age_groups)
                else str(_bulk_age_idx)
            )
            _bulk_lines.append(f"Age group (detected from filename): **{_band_label}**")
        elif (
            _restore_entry and _restore_entry.get("age_idx") is not None
            and int(_restore_entry["age_idx"]) >= 0
            and fit_target_age.value[_k] == "All (sum)"
            and int(num_age_groups) > 1
        ):
            _restore_age_idx = int(_restore_entry["age_idx"])
            _band_label = (
                age_groups[_restore_age_idx]
                if age_groups and len(age_groups) == int(num_age_groups)
                else str(_restore_age_idx)
            )
            _bulk_lines.append(f"Age group: **{_band_label}**")

        if _bulk_entry and _bulk_entry.get("weight") is not None and float(fit_target_weight.value[_k]) == 1.0:
            _bulk_lines.append(f"Weight: **{_bulk_entry['weight']}**")
        elif _restore_entry and _restore_entry.get("weight") is not None and float(fit_target_weight.value[_k]) == 1.0:
            _bulk_lines.append(f"Weight: **{_restore_entry['weight']}**")

        if (
            _bulk_entry and _bulk_entry.get("subpop") is not None
            and fit_target_subpop.value[_k] == "All (sum)"
            and is_metapop
        ):
            _bulk_lines.append(f"Subpopulation: **{_bulk_entry['subpop']}**")
        elif (
            _restore_entry and _restore_entry.get("subpop_idx") is not None
            and int(_restore_entry["subpop_idx"]) >= 0
            and fit_target_subpop.value[_k] == "All (sum)"
            and is_metapop
        ):
            _bulk_lines.append(f"Subpopulation index: **{_restore_entry['subpop_idx']}**")

        if (
            _bulk_entry and _bulk_entry.get("risk") is not None
            and fit_target_risk.value[_k] == "All (sum)"
            and int(num_risk_groups) > 1
        ):
            _bulk_lines.append(f"Risk group: **{_bulk_entry['risk']}**")
        elif (
            _restore_entry and _restore_entry.get("risk_idx") is not None
            and int(_restore_entry["risk_idx"]) >= 0
            and fit_target_risk.value[_k] == "All (sum)"
            and int(num_risk_groups) > 1
        ):
            _bulk_lines.append(f"Risk group: **{_restore_entry['risk_idx']}**")

        _base_line_count = 1 if (_src == "upload" and _bulk_entry) else (
            2 if (_src == "upload" and _restore_entry) else 0
        )
        _has_override_lines = len(_bulk_lines) > _base_line_count
        _bulk_defaults_note = mo.md("")
        if _bulk_lines:
            _prefix = "Restored from config" if (_src == "upload" and _restore_entry) else "Bulk upload"
            _msg = f"{_prefix} — " + "; ".join(_bulk_lines) + "."
            if _has_override_lines:
                _msg += " Change a field above to override just that one."
            _bulk_defaults_note = mo.callout(mo.md(_msg), kind="info")

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
        _tlabel = f"Target {_pos + 1}  ·  {_mode}" + (f"  ·  {_tvar}" if _tvar else "")
        _target_acc[_tlabel] = mo.vstack([
            mo.hstack([fit_target_src[_k], fit_target_remove_btn[_k]],
                      justify="space-between", align="start"),
            _data_input,
            _fname_note,
            fit_target_mode[_k],
            _prop_tip_row,
            _scalar_hint,
            fit_target_vars[_k],
            mo.hstack([fit_target_weight[_k]], justify="start"),
            _slice_ui,
            _bulk_defaults_note,
        ])

    # ── parameter bounds ──────────────────────────────────────────────────────
    if _selected_params:
        _rows = []
        for _j, _pn in enumerate(_selected_params):
            _is_seed_scale = _pn.startswith("seed_scale_")
            _bound_widgets = [fit_bounds_lo[_j], fit_bounds_hi[_j]]
            if not _is_seed_scale:
                _bound_widgets.append(fit_param_dims[_j])
            _bound_widgets.append(fit_bounds_log[_j])
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

    _ROBUST_TIP = (
        "<b>Robust gradient steps</b> (Adam only)<br><br>"
        "Two safeguards that keep the optimiser from overshooting a narrow "
        "best-fit region on steep or stiff loss landscapes — common when a "
        "parameter's effect grows sharply (e.g. a transmission rate, where the "
        "epidemic size rises roughly exponentially):<br>"
        "• each parameter moves at most a fraction of its search-bound width "
        "per step, so one steep gradient can't fling it across the optimum;<br>"
        "• any step that would <i>increase</i> the loss is rejected and shrunk "
        "(backtracking line search), so the loss decreases monotonically.<br><br>"
        "Leave this on unless you specifically want plain, unconstrained Adam "
        "steps. It does not apply to L-BFGS (which already line-searches) or to "
        "accept-reject."
    )

    _WALKERS_TIP = (
        "Number of ensemble members (walkers) in the affine-invariant MCMC.\n"
        "More walkers explore the posterior in parallel; rule of thumb ≥ 2× the\n"
        "number of fitted parameters. Each walker takes 'Iterations' steps."
    )
    _BURN_TIP = (
        "Initial MCMC steps discarded before collecting the posterior, giving\n"
        "the chains time to reach the stationary distribution. Must be < steps."
    )
    _THIN_TIP = (
        "Keep every k-th post-burn-in step. Reduces autocorrelation between\n"
        "retained draws and shrinks the stored ensemble."
    )
    _MCMC_ITER_TIP = (
        "MCMC steps per walker. Total forward simulations ≈ walkers × steps.\n"
        "Posterior draws ≈ walkers × (steps − burn-in) / thinning."
    )
    _POP_TIP = (
        "ABC-SMC particles accepted per generation. This is also (approximately)\n"
        "the number of posterior samples returned. Larger → smoother posterior."
    )
    _GENS_TIP = (
        "Maximum ABC-SMC generations. Each generation shrinks the acceptance\n"
        "threshold ε (tolerance) toward the data, refining the posterior."
    )
    _TV_TIP = (
        "<b>Time-varying transmission m(t)</b><br><br>"
        "Fit a smooth multiplier on the force of infection: knots every 'spacing' "
        "days (default 30), each with a fitted log-increment m_dlog_i (the change "
        "in log(m) from the previous knot). m(t) is <i>interpolated</i> between "
        "knots — linear in log space, so it ramps smoothly rather than jumping — "
        "then exponentiated, and it is anchored so m(t=0) = 1 (no effect at the "
        "start of the fit window). Captures seasonal / behavioural forcing a "
        "constant β can't.<br><br>"
        "Each increment m_dlog_i has an independent Gaussian prior/penalty "
        "centered at 0 with scale τ — a driftless random walk on log(m). Smaller "
        "τ ⇒ smoother, flatter m(t); larger τ ⇒ more freedom to swing between "
        "knots. Note this only shrinks the <i>increments</i> toward 0 (no "
        "change) — there is no mean-reversion pulling the cumulative multiplier "
        "back toward 1 over time, so it can drift and stay away from 1 if the "
        "data supports it. The increments are calibrated by the MCMC / ABC-SMC "
        "sampler. For a metapopulation a single shared m(t) is fit and applied "
        "identically across all subpopulations."
    )

    _method_val = fit_method.value
    _is_bayes = _method_val in ("mcmc", "abc-smc")
    if _method_val in ("adam", "lbfgs"):
        _hyper = [
            wtip(fit_lr, _LR_TIP, html_tip=True),
            wtip(fit_n_iter, _ITER_TIP, html_tip=True),
            wtip(fit_n_replications, _REP_TIP, html_tip=True),
        ]
    elif _method_val == "ar":
        _hyper = [wtip(fit_n_iter, _ITER_TIP, html_tip=True), wtip(fit_r2_thresh, _R2_TIP, html_tip=True)]
    elif _method_val == "mcmc":
        _hyper = [
            wtip(fit_n_iter, _MCMC_ITER_TIP, html_tip=False),
            wtip(fit_n_walkers, _WALKERS_TIP, html_tip=False),
            wtip(fit_mcmc_burnin, _BURN_TIP, html_tip=False),
            wtip(fit_mcmc_thin, _THIN_TIP, html_tip=False),
        ]
    else:  # abc-smc
        _hyper = [wtip(fit_abc_pop, _POP_TIP, html_tip=False), wtip(fit_abc_gens, _GENS_TIP, html_tip=False)]

    _PARALLEL_TIP = (
        "Run independent replications (different LHS starting points) across "
        "multiple CPU processes at once instead of one after another. Each "
        "replication is fully independent, so results are identical either way "
        "— this only changes wall-clock time. Defaults to half the available "
        "CPUs. Turn off if you need to limit CPU usage."
    )

    _robust_widgets = []
    if _method_val == "adam":
        _robust_widgets.append(wtip(fit_robust_steps, _ROBUST_TIP, html_tip=True))
    if _method_val in ("adam", "lbfgs"):
        _robust_widgets.append(wtip(fit_parallel, _PARALLEL_TIP, html_tip=True))
    _robust_row = mo.hstack(_robust_widgets, justify="start") if _robust_widgets else mo.md("")

    # The expected-posterior-size callout lives in its own cell (_fitting_post_note)
    # so that editing walkers / iterations / burn-in / thinning does NOT re-render —
    # and thus reset — the interactive hyperparameter widgets shown here.

    # Time-varying m(t) controls (Bayesian samplers only).
    _tv_row = mo.md("")
    if _is_bayes:
        _tv_children = [wtip(fit_tv_enable, _TV_TIP, html_tip=True)]
        if fit_tv_enable.value:
            _tv_children.append(mo.hstack([fit_tv_spacing, fit_tv_tau], justify="start"))
            if is_metapop:
                _tv_children.append(mo.callout(
                    mo.md(
                        "A **single shared m(t)** is fit and **broadcast uniformly to "
                        "every subpopulation** — one set of increments, identical "
                        "multiplier values across all subpops."
                    ),
                    kind="info",
                ))
        _tv_row = mo.vstack(_tv_children)

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

    # Linked-scale groups: one fitted multiplier scales several base params by
    # the same factor. Only reads the group count here (so editing a group's
    # name/bounds doesn't re-render and reset the other group widgets); the
    # per-group elements are displayed by indexing the arrays.
    _sg_ng = int(fit_n_scale_groups.value)
    _sg_rows = []
    for _gi in range(_sg_ng):
        _sg_rows.append(mo.vstack([
            fit_sg_names[_gi],
            mo.hstack([fit_sg_bases[_gi], fit_sg_lo[_gi], fit_sg_hi[_gi], fit_sg_dims[_gi], fit_sg_log[_gi]],
                      justify="start", align="center"),
        ]))
    _scale_group_section = mo.vstack([
        wtip(fit_n_scale_groups,
             "Fit one multiplier that scales several base parameters by the same "
             "factor, preserving each one's config value ratio (e.g. a single "
             "'ihr_scale' scaling both I_to_H_prop and IV_to_H_prop). Reduces "
             "dimensionality vs. fitting each separately. Base params picked here "
             "must NOT also be selected above. Works with every method.",
             html_tip=False),
        *_sg_rows,
    ])

    _upload_error_note = mo.md("")
    if fit_upload_error:
        _upload_error_note = mo.callout(
            mo.md(f"**Couldn't load that file:** {fit_upload_error}"), kind="danger",
        )

    _restore_error = get_restore_error()
    _restore_error_note = mo.md("")
    if _restore_error:
        _restore_error_note = mo.callout(
            mo.md(f"**Couldn't restore that file:** {_restore_error}"), kind="danger",
        )

    mo.vstack([
        mo.Html(
            f'<div style="font-size:1.35rem;font-weight:800;color:{_ACC};">Fitting</div>'
            '<div style="color:#777;margin:.1rem 0 .2rem;">Calibrate model '
            "parameters to observed data.</div>"
        ),
        section_card(
            step_header("↺", "Load Previous Results",
                        "Upload a fitting result JSON — downloaded from a previous run here, or "
                        "written by a standalone run_fitting.py — to view it below without "
                        "re-running the fit.",
                        accent=_ACC),
            mo.vstack([fit_upload_result, _upload_error_note]),
            accent=_ACC,
        ),
        section_card(
            step_header("⟲", "Restore a Saved Configuration",
                        "Upload a fit_config.json (downloaded below, or from a previous run "
                        "here) to recreate its targets, parameters, epidemic start date, and "
                        "method & run settings, so you can review or tweak them before "
                        "re-running. Restored targets replay the exact observed data saved in "
                        "the file rather than re-reading the source CSV — if a CSV has changed "
                        "since the file was saved, re-upload it to that target to refresh it.",
                        accent=_ACC),
            mo.vstack([fit_config_upload, _restore_error_note]),
            accent=_ACC,
        ),
        section_card(
            step_header("①", "Fit Targets",
                        "The data series / totals the model is calibrated against. "
                        "Click a target to expand it.",
                        accent=_ACC),
            mo.vstack([
                mo.accordion(_target_acc, multiple=True),
                mo.hstack([add_target_btn,
                           mo.md(f"*{_n} of 20 targets — expand a target to remove it*")],
                          justify="start", align="center"),
                mo.hstack([fit_bulk_upload], justify="start"),
                mo.hstack(_bulk_default_items, justify="start"),
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
                mo.md("**Linked-scale groups** *(optional)*"),
                _scale_group_section,
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
                _robust_row,
                _tv_row,
                fit_run_button,
            ]),
            accent=_ACC,
        ),
    ])
    return


@app.cell
def _fitting_post_note(
    mo, main_tab, fit_method,
    fit_n_iter, fit_n_walkers, fit_mcmc_burnin, fit_mcmc_thin,
    fit_abc_pop, fit_abc_gens,
):
    # Expected posterior size + total evaluation count for the Bayesian samplers.
    # Isolated from _fitting_display so that editing these hyperparameters only
    # re-runs this callout — it never re-renders (and resets) the input widgets.
    mo.stop(main_tab.value != "Fitting", None)
    _mv = fit_method.value
    if _mv == "mcmc":
        _nw = max(int(fit_n_walkers.value), 4)
        _ns = int(fit_n_iter.value)
        _bi = min(int(fit_mcmc_burnin.value), max(0, _ns - 1))
        _th = max(1, int(fit_mcmc_thin.value))
        _draws = ((_ns - _bi) // _th) * _nw
        _out = mo.callout(
            mo.md(
                f"**≈ {_draws:,} posterior draws** = walkers × (steps − burn-in) ÷ thinning "
                f"(before dropping any stuck walkers). "
                f"**≈ {_nw * _ns:,} forward simulations** total."
            ),
            kind="info",
        )
    elif _mv == "abc-smc":
        _out = mo.callout(
            mo.md(
                f"**≈ {int(fit_abc_pop.value):,} posterior particles** (the final-generation "
                "population). Total simulations vary with the acceptance rate across "
                f"up to {int(fit_abc_gens.value)} generations."
            ),
            kind="info",
        )
    else:
        _out = None
    _out


@app.cell
def _fitting_obs_parse(
    get_target_slots,
    get_bulk_file_data, get_restored_target_data,
    fit_target_src, fit_target_upload, fit_target_path, fit_target_mode,
    start_date_input, pd, io, Path, np,
):
    _slots = get_target_slots()
    # Simulation day 0 corresponds to this calendar date; timeseries targets are
    # aligned to it by date below (must match the start_date passed to run_fit).
    _sim_start_ts = pd.Timestamp(start_date_input.value.strip() or "2024-01-01").normalize()
    # fit_obs_arrays: {slot -> (np.array | list-of-dicts | None)}, keyed by the
    # target's slot id (not display position) — see get_target_slots.
    # fit_obs_n_days: {slot -> int} days in timeseries, or 0 for scalar targets
    # fit_obs_weights: {slot -> (np.array | None)} — optional per-timepoint
    # weights from a "weight" CSV column (ts targets only), date-aligned the
    # same way as fit_obs_arrays so the two stay in lockstep.
    fit_obs_arrays = {}
    fit_obs_n_days = {}
    fit_obs_weights = {}
    _bulk_data = get_bulk_file_data()
    _restored_data = get_restored_target_data()

    for _k in _slots:
        _src = fit_target_src.value[_k]
        _mode = fit_target_mode.value[_k]

        # A target restored from a saved fit_config.json (see
        # _fit_config_upload_ui) has no real CSV to parse — its observed data
        # is already the exact numbers saved in the file. Used only while the
        # source widget still shows its untouched default (same "still at
        # default" precedent as the bulk-upload fields), so uploading a real
        # CSV to this slot overrides it.
        _restore_entry = _restored_data.get(_k)
        if (
            _restore_entry is not None
            and _src == "upload"
            and not fit_target_upload.value[_k]
            and not _bulk_data.get(_k)
        ):
            _r_mode = _restore_entry.get("mode")
            _obs = _restore_entry.get("observed")
            _pw = _restore_entry.get("point_weights")
            if _r_mode == "ts":
                _arr = np.array([np.nan if _v is None else _v for _v in (_obs or [])], dtype=float)
                fit_obs_arrays[_k] = _arr
                fit_obs_n_days[_k] = len(_arr)
                fit_obs_weights[_k] = (
                    np.array([np.nan if _v is None else _v for _v in _pw], dtype=float)
                    if _pw is not None else None
                )
            else:
                fit_obs_arrays[_k] = _obs
                fit_obs_n_days[_k] = 0
                fit_obs_weights[_k] = None
            continue

        _df = None
        try:
            if _src == "upload" and fit_target_upload.value[_k]:
                _df = pd.read_csv(io.BytesIO(fit_target_upload.value[_k][0].contents))
            elif _src == "upload" and _bulk_data.get(_k):
                _df = pd.read_csv(io.BytesIO(_bulk_data[_k]["contents"]))
            elif _src == "path" and fit_target_path.value[_k].strip():
                _df = pd.read_csv(Path(fit_target_path.value[_k].strip()).expanduser())
        except Exception:
            _df = None

        if _df is None:
            fit_obs_arrays[_k] = None
            fit_obs_n_days[_k] = 0
            fit_obs_weights[_k] = None
            continue

        if _mode in ("scalar", "proportion"):
            if "value" not in _df.columns:
                fit_obs_arrays[_k] = None
                fit_obs_n_days[_k] = 0
                fit_obs_weights[_k] = None
                continue
            _rows = []
            for _, _row in _df.iterrows():
                _entry = {"value": float(_row["value"])}
                for _col in ("age", "risk", "subpopulation"):
                    if _col in _df.columns and pd.notna(_row.get(_col)):
                        _entry[_col] = _row[_col]
                _rows.append(_entry)
            fit_obs_arrays[_k] = _rows
            fit_obs_n_days[_k] = 0
            fit_obs_weights[_k] = None
        else:
            _META_COLS_TS = {"date", "day", "time", "week", "subpopulation", "age", "risk", "weight"}
            _non_id = [c for c in _df.columns if c.lower() not in _META_COLS_TS]
            if not _non_id:
                _non_id = [c for c in _df.columns if c.lower() not in ("date", "day", "time", "week", "weight")]
            _val_col = "value" if "value" in _df.columns else (_non_id[0] if _non_id else None)
            _date_col = next(
                (c for c in _df.columns if c.lower() in ("date", "day", "time", "week")), None
            )
            # Optional per-timepoint weight column (e.g. up-weight the
            # epidemic peak). Aligned onto the same calendar as the value
            # column below so the two arrays stay index-matched.
            _weight_col = next((c for c in _df.columns if c.lower() == "weight"), None)
            if not _val_col:
                fit_obs_arrays[_k] = None
                fit_obs_n_days[_k] = 0
                fit_obs_weights[_k] = None
                continue

            _vals = pd.to_numeric(_df[_val_col], errors="coerce")
            _wts = pd.to_numeric(_df[_weight_col], errors="coerce") if _weight_col else None
            # Date-align the observed series onto the simulation calendar so that
            # observed[0] is the value on the simulation start date. Rows before
            # the start date are dropped; days from the start date up to the last
            # observed date with no data become NaN (gaps + any leading lead-in),
            # which the fitting loss masks out. Without this, the loss compared
            # observed[i] to simulated[i] by row position, silently misaligning
            # any CSV that doesn't begin exactly on the simulation start date.
            _aligned = None
            _weight_aligned = None
            if _date_col is not None:
                try:
                    _dts = pd.to_datetime(_df[_date_col], errors="coerce").dt.normalize()
                    _ser = pd.Series(_vals.to_numpy(dtype=float), index=_dts)
                    _ser = _ser[~_ser.index.isna()]
                    _ser = _ser[~_ser.index.duplicated(keep="last")].sort_index()
                    _last = _ser.index.max()
                    if pd.notna(_last) and _last >= _sim_start_ts:
                        _full = pd.date_range(_sim_start_ts, _last, freq="D")
                        _aligned = _ser.reindex(_full).to_numpy(dtype=float)
                        if _wts is not None:
                            _wser = pd.Series(_wts.to_numpy(dtype=float), index=_dts)
                            _wser = _wser[~_wser.index.isna()]
                            _wser = _wser[~_wser.index.duplicated(keep="last")].sort_index()
                            _weight_aligned = _wser.reindex(_full).to_numpy(dtype=float)
                except Exception:
                    _aligned = None
                    _weight_aligned = None
            if _aligned is None:
                # No usable date column (or no overlap with the sim window) —
                # fall back to assuming the series already starts at day 0.
                _aligned = _vals.dropna().to_numpy(dtype=float)
                if _wts is not None:
                    _weight_aligned = _wts.to_numpy(dtype=float)[:len(_aligned)]

            fit_obs_arrays[_k] = _aligned
            fit_obs_n_days[_k] = len(_aligned)
            fit_obs_weights[_k] = _weight_aligned

    return fit_obs_arrays, fit_obs_n_days, fit_obs_weights


@app.cell
def _fitting_build_request(
    get_target_slots,
    get_bulk_file_data, get_restored_target_data,
    fit_target_src, fit_target_upload, fit_target_path,
    fit_target_vars, fit_target_mode, fit_target_weight,
    fit_target_subpop, fit_target_age, fit_target_risk,
    fit_obs_arrays, fit_obs_weights,
    fit_sim_days_input,
    fit_method, fit_params_multiselect,
    fit_bounds_lo, fit_bounds_hi, fit_param_dims, fit_bounds_log,
    fit_n_scale_groups, fit_sg_names, fit_sg_bases, fit_sg_lo, fit_sg_hi, fit_sg_dims, fit_sg_log,
    fit_start_offset_enable, fit_start_offset_lo, fit_start_offset_hi,
    fit_lr, fit_n_iter, fit_r2_thresh, fit_n_replications, fit_robust_steps,
    fit_parallel,
    fit_n_walkers, fit_mcmc_burnin, fit_mcmc_thin,
    fit_abc_pop, fit_abc_gens,
    fit_tv_enable, fit_tv_spacing, fit_tv_tau,
    config_dict, compartments, is_metapop,
    build_compartment_init,
    start_date_input, timesteps, rng_seed,
    num_age_groups, num_risk_groups,
    metapop_folder_input, metapop_travel_config,
    mobility_input, daily_vaccines_input, loaded_schedule_dfs,
    np, json, FitTarget, FitConfig, Path,
):
    # Build the FitTarget/FitConfig request from the current UI state —
    # independent of the run button, so the export download (fit_config.json +
    # run_fitting.py, see _fitting_export_display below) is available
    # immediately, without first running a fit. _run_fitting reuses these same
    # objects when the button is actually clicked.
    _slots = get_target_slots()

    def _parse_idx(val):
        if val == "All (sum)":
            return -1
        return int(str(val).split(":")[0])

    # Target display labels, keyed by slot id (not display position).
    _bulk_data = get_bulk_file_data()
    _restored_data = get_restored_target_data()
    _target_labels = {}
    for _pos, _k in enumerate(_slots):
        if fit_target_src.value[_k] == "upload" and fit_target_upload.value[_k]:
            _target_labels[_k] = fit_target_upload.value[_k][0].name
        elif fit_target_src.value[_k] == "upload" and _bulk_data.get(_k):
            _target_labels[_k] = _bulk_data[_k]["name"]
        elif fit_target_src.value[_k] == "upload" and _restored_data.get(_k):
            _target_labels[_k] = _restored_data[_k].get("label", f"Target {_pos + 1}")
        elif fit_target_src.value[_k] == "path" and fit_target_path.value[_k].strip():
            _target_labels[_k] = Path(fit_target_path.value[_k].strip()).name
        else:
            _target_labels[_k] = f"Target {_pos + 1}"

    # A bulk-added or restored (see _fit_config_upload_ui) target's variables
    # default is used only while the per-target widget still shows its
    # original single-compartment default — matches the note shown in
    # _fitting_display, and lets picking anything else in the widget override
    # it, since the widget's rendered value can't be reassigned from code.
    _default_var = [compartments[0]] if compartments else ["S"]

    def _resolve_vars(_k):
        _bulk_vars = _bulk_data.get(_k, {}).get("vars")
        _restore_vars = _restored_data.get(_k, {}).get("vars")
        _widget_vars = list(fit_target_vars.value[_k])
        if _widget_vars == _default_var:
            if _bulk_vars:
                return _bulk_vars
            if _restore_vars:
                return _restore_vars
        return _widget_vars

    # A restored target's saved observed-data mode (ts/scalar/proportion) is
    # used only while the widget still shows the default "ts".
    def _resolve_mode(_k):
        _restore_mode = _restored_data.get(_k, {}).get("mode")
        _widget_mode = fit_target_mode.value[_k]
        if _restore_mode and _widget_mode == "ts":
            return _restore_mode
        return _widget_mode

    # Age group auto-detected from the uploaded filename (see
    # _fit_bulk_upload_ui), or restored from a saved config, is used only
    # while the per-target dropdown still shows "All (sum)" — matches the
    # note shown in _fitting_display.
    def _resolve_age_idx(_k):
        _bulk_age = _bulk_data.get(_k, {}).get("age_idx")
        _restore_age = _restored_data.get(_k, {}).get("age_idx")
        _widget_age = fit_target_age.value[_k]
        if _widget_age == "All (sum)":
            if _bulk_age is not None:
                return _bulk_age
            if _restore_age is not None:
                return int(_restore_age)
        return _parse_idx(_widget_age)

    # Weight/subpopulation/risk defaults (fit_bulk_weight/fit_bulk_subpop/
    # fit_bulk_risk at upload time, or the equivalent restored values) follow
    # the same "widget still at default" override precedent as vars/age above.
    def _resolve_weight(_k):
        _bulk_weight = _bulk_data.get(_k, {}).get("weight")
        _restore_weight = _restored_data.get(_k, {}).get("weight")
        _widget_weight = float(fit_target_weight.value[_k])
        if _widget_weight == 1.0:
            if _bulk_weight is not None:
                return float(_bulk_weight)
            if _restore_weight is not None:
                return float(_restore_weight)
        return _widget_weight

    def _resolve_subpop_idx(_k):
        _bulk_subpop = _bulk_data.get(_k, {}).get("subpop")
        _restore_subpop_idx = _restored_data.get(_k, {}).get("subpop_idx")
        _widget_subpop = fit_target_subpop.value[_k]
        if _widget_subpop == "All (sum)":
            if _bulk_subpop is not None:
                return _parse_idx(_bulk_subpop)
            if _restore_subpop_idx is not None:
                return int(_restore_subpop_idx)
        return _parse_idx(_widget_subpop)

    def _resolve_risk_idx(_k):
        _bulk_risk = _bulk_data.get(_k, {}).get("risk")
        _restore_risk_idx = _restored_data.get(_k, {}).get("risk_idx")
        _widget_risk = fit_target_risk.value[_k]
        if _widget_risk == "All (sum)":
            if _bulk_risk is not None:
                return _parse_idx(_bulk_risk)
            if _restore_risk_idx is not None:
                return int(_restore_risk_idx)
        return _parse_idx(_widget_risk)

    fit_targets = [
        FitTarget(
            variables=_resolve_vars(_k),
            mode=_resolve_mode(_k),
            weight=_resolve_weight(_k),
            observed=fit_obs_arrays.get(_k),
            point_weights=fit_obs_weights.get(_k),
            label=_target_labels[_k],
            subpop_idx=_resolve_subpop_idx(_k),
            age_idx=_resolve_age_idx(_k),
            risk_idx=_resolve_risk_idx(_k),
        )
        for _k in _slots
    ]

    # Directly-fitted params (from the multiselect) with their bounds/dims.
    _base_selected = list(fit_params_multiselect.value)
    _bounds = {
        _base_selected[_j]: (float(fit_bounds_lo.value[_j]), float(fit_bounds_hi.value[_j]))
        for _j in range(len(_base_selected))
    }
    _param_dims = {
        _base_selected[_j]: list(fit_param_dims.value[_j])
        for _j in range(len(_base_selected))
    }

    # Linked-scale groups: each becomes a synthetic fitted multiplier (added to
    # selected_params with its own bounds); its base params are driven from it.
    # Params flagged to be fit in log10 space (per-param checkboxes).
    _log_params = [
        _base_selected[_j] for _j in range(len(_base_selected))
        if bool(fit_bounds_log.value[_j])
    ]

    _scale_groups = {}
    for _gi in range(int(fit_n_scale_groups.value)):
        _nm = str(fit_sg_names.value[_gi]).strip()
        _bases = [str(_b) for _b in fit_sg_bases.value[_gi]]
        if _nm and _bases:
            _scale_groups[_nm] = _bases
            _bounds[_nm] = (float(fit_sg_lo.value[_gi]), float(fit_sg_hi.value[_gi]))
            _param_dims[_nm] = list(fit_sg_dims.value[_gi])
            if bool(fit_sg_log.value[_gi]):
                _log_params.append(_nm)

    _selected_params = _base_selected + [_m for _m in _scale_groups if _m not in _base_selected]
    fit_config_obj = FitConfig(
        selected_params=_selected_params,
        bounds=_bounds,
        param_dims=_param_dims,
        scale_groups=_scale_groups,
        log_params=_log_params,
        method=fit_method.value,
        lr=float(fit_lr.value),
        n_iter=int(fit_n_iter.value),
        n_replications=int(fit_n_replications.value),
        r2_threshold=float(fit_r2_thresh.value),
        sim_days=int(fit_sim_days_input.value),
        fit_start_offset=fit_start_offset_enable.value,
        start_offset_bounds=(int(fit_start_offset_lo.value), int(fit_start_offset_hi.value)),
        robust_steps=bool(fit_robust_steps.value),
        parallel=bool(fit_parallel.value),
        n_walkers=int(fit_n_walkers.value),
        mcmc_burnin=int(fit_mcmc_burnin.value),
        mcmc_thin=int(fit_mcmc_thin.value),
        abc_pop_size=int(fit_abc_pop.value),
        abc_max_gens=int(fit_abc_gens.value),
        tv_transmission=bool(fit_tv_enable.value),
        tv_knot_spacing_days=int(fit_tv_spacing.value),
        tv_tau=float(fit_tv_tau.value),
    )

    # Initial conditions from the Step 6 tables via config_dict (single-pop only —
    # the metapop path builds its own per-subpop init inside run_fit).
    fit_compartment_init = None
    if not is_metapop:
        _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
        _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((num_age_groups, num_risk_groups))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in compartments
        }
        fit_compartment_init, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)

    # Subpop names (for resolving "subpopulation" columns in scalar/proportion rows)
    _subpop_names_run = []
    if is_metapop and metapop_folder_input.value.strip():
        try:
            with open(Path(metapop_folder_input.value.strip()) / "metapop_config.json") as _f:
                _subpop_names_run = json.load(_f).get("subpopulations", [])
        except Exception:
            _subpop_names_run = []

    fit_run_kwargs = dict(
        start_date=start_date_input.value.strip() or "2024-01-01",
        ts_per_day=int(timesteps.value),
        seed_base=int(rng_seed.value),
        num_age_groups=num_age_groups,
        num_risk_groups=num_risk_groups,
        metapop_folder=metapop_folder_input.value.strip() if is_metapop else None,
        metapop_travel_config=metapop_travel_config or None,
        subpop_names=tuple(_subpop_names_run),
        mobility_value=float(mobility_input.value),
        daily_vaccines_value=float(daily_vaccines_input.value),
    )

    # JSON-serializable record of this exact request, downloadable so a standalone
    # script can reproduce the same run_fit(...) call on a server. Built
    # best-effort: if the current UI state can't be serialized yet (e.g. no
    # targets configured), the download is simply unavailable until it can be.
    try:
        fit_run_config = {
            "compartments": list(compartments),
            "is_metapop": is_metapop,
            "targets": [
                {
                    "variables": _t.variables,
                    "mode": _t.mode,
                    "weight": _t.weight,
                    "observed": (
                        [None if (isinstance(_v, float) and np.isnan(_v)) else _v for _v in _t.observed]
                        if isinstance(_t.observed, np.ndarray) else _t.observed
                    ),
                    "point_weights": (
                        [None if (isinstance(_v, float) and np.isnan(_v)) else _v for _v in _t.point_weights]
                        if isinstance(_t.point_weights, np.ndarray) else _t.point_weights
                    ),
                    "label": _t.label,
                    "subpop_idx": _t.subpop_idx,
                    "age_idx": _t.age_idx,
                    "risk_idx": _t.risk_idx,
                }
                for _t in fit_targets
            ],
            "fit_config": {
                "selected_params": fit_config_obj.selected_params,
                "bounds": {_k: list(_v) for _k, _v in fit_config_obj.bounds.items()},
                "param_dims": fit_config_obj.param_dims,
                "scale_groups": fit_config_obj.scale_groups,
                "log_params": fit_config_obj.log_params,
                "method": fit_config_obj.method,
                "lr": fit_config_obj.lr,
                "n_iter": fit_config_obj.n_iter,
                "n_replications": fit_config_obj.n_replications,
                "r2_threshold": fit_config_obj.r2_threshold,
                "sim_days": fit_config_obj.sim_days,
                "fit_start_offset": fit_config_obj.fit_start_offset,
                "start_offset_bounds": list(fit_config_obj.start_offset_bounds),
                "robust_steps": fit_config_obj.robust_steps,
                "parallel": fit_config_obj.parallel,
                "n_walkers": fit_config_obj.n_walkers,
                "mcmc_burnin": fit_config_obj.mcmc_burnin,
                "mcmc_thin": fit_config_obj.mcmc_thin,
                "abc_pop_size": fit_config_obj.abc_pop_size,
                "abc_max_gens": fit_config_obj.abc_max_gens,
                "tv_transmission": fit_config_obj.tv_transmission,
                "tv_knot_spacing_days": fit_config_obj.tv_knot_spacing_days,
                "tv_tau": fit_config_obj.tv_tau,
            },
            "run_kwargs": fit_run_kwargs,
            # The real uploaded schedule CSVs (humidity/calendar/mobility/vaccines),
            # single-pop only — the metapop path reads its own per-subpop CSVs from
            # the metapop folder regardless of this field. Needed so the exported
            # script reproduces any real, time-varying vaccination schedule (and its
            # backfill/delay preprocessing) instead of falling back to a constant.
            # Any attribute the user didn't upload a CSV for is None (the notebook
            # falls back to mobility_value/daily_vaccines_value for it) and is
            # omitted here so the export script falls back the same way.
            "schedule_csvs": None if is_metapop else {
                _name: _df.to_csv(index=False)
                for _name, _df in (
                    ("absolute_humidity_df", loaded_schedule_dfs.absolute_humidity_df),
                    ("school_work_calendar_df", loaded_schedule_dfs.school_work_calendar_df),
                    ("mobility_df", loaded_schedule_dfs.mobility_df),
                    ("daily_vaccines_df", loaded_schedule_dfs.daily_vaccines_df),
                )
                if _df is not None
            },
        }
    except Exception:
        fit_run_config = None

    # Signature of the current fit request — used to detect when previously
    # displayed results (kept around via mo.state, see _fit_result_state) no
    # longer correspond to the targets/parameters/method configured above.
    import hashlib as _hashlib
    fit_run_config_signature = (
        _hashlib.sha256(json.dumps(fit_run_config, sort_keys=True, default=str).encode()).hexdigest()
        if fit_run_config is not None else None
    )

    return (
        fit_targets, fit_config_obj, fit_compartment_init, fit_run_kwargs,
        fit_run_config, fit_run_config_signature,
    )


@app.cell
def _fitting_export_display(fit_run_config, mo, main_tab, json):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(
        fit_run_config is None,
        mo.callout(
            mo.md(
                "Configure at least one fit target and select parameters above to "
                "enable downloading a standalone fitting script."
            ),
            kind="info",
        ),
    )

    _fit_config_dl = mo.download(
        data=json.dumps(fit_run_config, indent=2).encode(),
        filename="fit_config.json",
        mimetype="application/json",
        label="Download fit_config.json",
    )
    _run_fitting_script = """\
    #!/usr/bin/env python3
    \"\"\"
    Generated by CLT Model Builder Notebook (Fitting tab export).
    Usage: python run_fitting.py
    Reads model_config.json + fit_config.json from this directory and writes
    fitted_params.json.

    Location: this file must sit exactly two directory levels below the
    repo root that contains generic_core/ — e.g. <repo_root>/some_folder/
    some_subfolder/run_fitting.py — since it adds
    Path(__file__).parent.parent.parent to sys.path to import generic_core.
    model_config.json and fit_config.json must sit alongside it in that
    same directory.
    If you move this file, update the sys.path.insert(...) line below:
    count how many directories separate this file from the repo root
    (the one containing generic_core/), then use that many + 1 .parent
    calls from __file__ (equivalently, .parent calls on _HERE equal to
    that count). Also make sure model_config.json / fit_config.json are
    still next to this file.
    \"\"\"

    import sys
    import json
    import io
    from pathlib import Path
    from types import SimpleNamespace

    MODEL_CONFIG_FILE = "model_config.json"
    FIT_CONFIG_FILE = "fit_config.json"
    OUTPUT_FILE = "fitted_params.json"

    _HERE = Path(__file__).parent
    sys.path.insert(0, str(_HERE.parent.parent))

    import numpy as np
    import pandas as pd
    from generic_core.model_factory import build_compartment_init
    from generic_core.fitting import FitTarget, FitConfig, run_fit, fit_result_to_dict


    def main():
        with open(_HERE / MODEL_CONFIG_FILE) as _f:
            config_dict = json.load(_f)
        with open(_HERE / FIT_CONFIG_FILE) as _f:
            fit_cfg_raw = json.load(_f)

        compartments = fit_cfg_raw["compartments"]
        is_metapop = fit_cfg_raw["is_metapop"]
        targets = [
            FitTarget(
                variables=_t["variables"], mode=_t["mode"], weight=_t["weight"],
                observed=(np.array(_t["observed"], dtype=float) if _t["mode"] == "ts" else _t["observed"]),
                point_weights=(
                    np.array(_t["point_weights"], dtype=float)
                    if _t.get("point_weights") is not None else None
                ),
                label=_t["label"], subpop_idx=_t["subpop_idx"], age_idx=_t["age_idx"], risk_idx=_t["risk_idx"],
            )
            for _t in fit_cfg_raw["targets"]
        ]
        _fc = fit_cfg_raw["fit_config"]
        fit_config = FitConfig(
            selected_params=_fc["selected_params"],
            bounds={_k: tuple(_v) for _k, _v in _fc["bounds"].items()},
            param_dims=_fc["param_dims"],
            scale_groups=_fc.get("scale_groups", {}),
            log_params=_fc.get("log_params", []),
            method=_fc["method"], lr=_fc["lr"], n_iter=_fc["n_iter"],
            n_replications=_fc["n_replications"], r2_threshold=_fc["r2_threshold"],
            sim_days=_fc["sim_days"], fit_start_offset=_fc["fit_start_offset"],
            start_offset_bounds=tuple(_fc["start_offset_bounds"]),
            robust_steps=_fc.get("robust_steps", True),
            parallel=_fc.get("parallel", True),
            n_workers=_fc.get("n_workers"),
            n_walkers=_fc.get("n_walkers", 32),
            mcmc_burnin=_fc.get("mcmc_burnin", 500),
            mcmc_thin=_fc.get("mcmc_thin", 10),
            abc_pop_size=_fc.get("abc_pop_size", 200),
            abc_max_gens=_fc.get("abc_max_gens", 12),
            tv_transmission=_fc.get("tv_transmission", False),
            tv_knot_spacing_days=_fc.get("tv_knot_spacing_days", 30),
            tv_tau=_fc.get("tv_tau", 0.25),
        )

        _rk = fit_cfg_raw["run_kwargs"]
        num_age_groups = _rk["num_age_groups"]
        num_risk_groups = _rk["num_risk_groups"]

        compartment_init = None
        if not is_metapop:
            _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
            _pop_arr = np.asarray(
                _ic_entry.get("population", np.zeros((num_age_groups, num_risk_groups))), dtype=float,
            )
            _seed_arrays = {
                _c: np.asarray(_a, dtype=float)
                for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
                if _c in compartments
            }
            compartment_init, _ = build_compartment_init(_seed_arrays, _pop_arr, compartments)

        # Reconstruct the real uploaded schedule CSVs (so any time-varying vaccination
        # schedule, and its backfill/delay preprocessing, is reproduced exactly). Any
        # field missing from schedule_csvs (no CSV uploaded for it) becomes None, same
        # as the notebook's loaded_schedule_dfs, and model_factory falls back to a
        # constant for that field accordingly.
        schedule_dfs = None
        _sched_csvs = fit_cfg_raw.get("schedule_csvs")
        if _sched_csvs is not None:
            _sched_fields = ("absolute_humidity_df", "school_work_calendar_df", "mobility_df", "daily_vaccines_df")
            schedule_dfs = SimpleNamespace(**{
                _name: (pd.read_csv(io.StringIO(_sched_csvs[_name])) if _name in _sched_csvs else None)
                for _name in _sched_fields
            })

        result = run_fit(
            config_dict=config_dict,
            compartments=compartments,
            is_metapop=is_metapop,
            targets=targets,
            fit_config=fit_config,
            compartment_init=compartment_init,
            schedule_dfs=schedule_dfs,
            start_date=_rk["start_date"],
            ts_per_day=_rk["ts_per_day"],
            seed_base=_rk["seed_base"],
            num_age_groups=num_age_groups,
            num_risk_groups=num_risk_groups,
            metapop_folder=_rk["metapop_folder"],
            metapop_travel_config=_rk["metapop_travel_config"],
            subpop_names=tuple(_rk["subpop_names"]),
            mobility_value=_rk["mobility_value"],
            daily_vaccines_value=_rk["daily_vaccines_value"],
        )

        Path(OUTPUT_FILE).write_text(json.dumps(fit_result_to_dict(result), indent=2))
        print(f"Wrote {OUTPUT_FILE}")


    if __name__ == "__main__":
        # Gradient methods (Adam/L-BFGS) run replications across a process pool
        # by default (FitConfig.parallel=True) — this guard is required on
        # macOS/Windows, which spawn fresh interpreters for worker processes that
        # re-import this file.
        main()
"""
    _run_fitting_script_dl = mo.download(
        data=_run_fitting_script.encode(),
        filename="run_fitting.py",
        mimetype="text/x-python",
        label="Download run_fitting.py",
    )
    mo.callout(
        mo.vstack([
            mo.md(
                "**Run this fit on a server.** Download `run_fitting.py` and "
                "`fit_config.json` here, plus `model_config.json` from the "
                "**Export** tab, into one folder, then run "
                "`python run_fitting.py` — it reruns this exact calibration "
                "headlessly and writes `fitted_params.json`."
            ),
            mo.hstack([_fit_config_dl, _run_fitting_script_dl], justify="start"),
        ]),
        kind="info",
    )
    return


@app.cell
def _fit_result_state(mo):
    # Persists the last fitting result (run or uploaded) independent of the
    # reactive graph, so it survives switching tabs or tweaking any other
    # widget on this tab — both of which would otherwise re-run _run_fitting
    # with fit_run_button.value back to False and lose the result.
    get_fit_result_state, set_fit_result_state = mo.state(None)
    return get_fit_result_state, set_fit_result_state


@app.cell
def _fit_result_reader(get_fit_result_state):
    _fit_state = get_fit_result_state()
    fit_result = _fit_state["result"] if _fit_state else None
    fit_result_signature = _fit_state["signature"] if _fit_state else None
    fit_result_source = _fit_state["source"] if _fit_state else None
    return fit_result, fit_result_signature, fit_result_source


@app.cell
def _fitting_staleness(fit_result, fit_result_signature, fit_run_config_signature):
    # Stale = can't be verified to match the currently configured targets /
    # parameters / method (signature unknown, e.g. an uploaded file, or the
    # config has changed since this result was produced).
    fit_result_is_stale = fit_result is not None and not (
        fit_result_signature is not None
        and fit_run_config_signature is not None
        and fit_result_signature == fit_run_config_signature
    )
    return (fit_result_is_stale,)


@app.cell
def _fitting_upload_ui(mo):
    fit_upload_result = mo.ui.file(
        label="Upload fitting result JSON", filetypes=[".json"], multiple=False,
    )
    return (fit_upload_result,)


@app.cell
def _fitting_load_uploaded(fit_upload_result, set_fit_result_state, fit_result_from_dict, json):
    # Loads a previously downloaded/exported fitting result JSON (same schema
    # as fitted_params.json, see fit_result_to_dict) and displays it exactly
    # like a freshly run fit, via the same persisted-result state.
    fit_upload_error = None
    if fit_upload_result.value:
        try:
            _raw = json.loads(fit_upload_result.value[0].contents.decode())
            _loaded = fit_result_from_dict(_raw)
            set_fit_result_state({"result": _loaded, "signature": None, "source": "uploaded"})
        except Exception as _exc:
            fit_upload_error = str(_exc)
    return (fit_upload_error,)


@app.cell
def _run_fitting(
    fit_run_button, fit_obs_arrays, fit_obs_n_days,
    get_target_slots,
    fit_target_vars, fit_target_mode, fit_params_multiselect,
    fit_targets, fit_config_obj, fit_compartment_init, fit_run_kwargs,
    fit_run_config_signature, set_fit_result_state,
    config_dict, compartments, is_metapop, loaded_schedule_dfs,
    mo, run_fit,
):
    # Thin UI wrapper: on click, validate and delegate the actual optimization
    # to generic_core.fitting.run_fit (the same function an exported standalone
    # fitting script calls) using the request built in _fitting_build_request.
    # The result is written to persisted state (_fit_result_state) rather than
    # returned directly — see _fit_result_reader for why.
    if fit_run_button.value:
        _slots = get_target_slots()

        # ── validation (kept here so the UI shows inline callouts) ───────────
        for _pos, _k in enumerate(_slots):
            if fit_obs_arrays.get(_k) is None:
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_pos + 1}: no observed data loaded.**"), kind="warn"),
                )
            if not list(fit_target_vars.value[_k]):
                mo.stop(
                    True,
                    mo.callout(mo.md(f"**Target {_pos + 1}: no variables selected.**"), kind="warn"),
                )
        mo.stop(
            not list(fit_params_multiselect.value),
            mo.callout(mo.md("**No parameters to fit.** Select parameters above."), kind="warn"),
        )
        _ts_days = [fit_obs_n_days.get(_k, 0) for _k in _slots if fit_target_mode.value[_k] == "ts"]
        mo.stop(
            len(set(_ts_days)) > 1,
            mo.callout(
                mo.md(
                    "**Timeseries targets have mismatched lengths:** "
                    + ", ".join(
                        f"Target {_pos + 1}: {fit_obs_n_days.get(_k, 0)} days"
                        for _pos, _k in enumerate(_slots)
                        if fit_target_mode.value[_k] == "ts"
                    )
                    + ". All timeseries targets must have the same number of observations."
                ),
                kind="danger",
            ),
        )

        def _fmt_progress(info):
            _method = info.get("method", "")
            if _method in ("adam", "lbfgs"):
                _phase = info.get("phase", "")
                _rep = info.get("rep", 0)
                _total = info.get("total", 1)
                if _phase == "start_parallel":
                    _nw = info.get("n_workers", 1)
                    return f"Running {_total} replication(s) in parallel ({_nw} workers)…"
                elif _phase == "rep_starting":
                    return f"Replication {_rep}/{_total} running…"
                elif _phase == "rep_done":
                    _loss = info.get("loss", float("nan"))
                    return f"Replication {_rep}/{_total} done  (loss = {_loss:.4g})"
                return f"Gradient fitting…"
            elif _method == "ar":
                _s = info.get("samples", 0)
                _t = info.get("total", 1)
                _acc = info.get("accepted", 0)
                _r2 = info.get("best_r2", float("nan"))
                _pct = int(100 * _s / max(_t, 1))
                return f"Sample {_s:,}/{_t:,} explored ({_pct}%)  —  {_acc} accepted  |  best R² = {_r2:.3f}"
            elif _method == "mcmc":
                _step = info.get("step", 0)
                _total = info.get("total", 1)
                _nw = info.get("walkers", "?")
                _lp = info.get("mean_log_prob", float("nan"))
                _pct = int(100 * _step / max(_total, 1))
                return (
                    f"MCMC step {_step:,}/{_total:,} ({_pct}%)  —  {_nw} walkers  "
                    f"|  mean log-posterior = {_lp:.3g}"
                )
            elif _method == "abc-smc":
                _gen = info.get("generation", 0)
                _total = info.get("total", 1)
                _eps = info.get("epsilon", float("nan"))
                return f"ABC-SMC generation {_gen}/{_total}  |  ε = {_eps:.4g}"
            return "Running fitting…"

        mo.output.replace(mo.callout(mo.md("Running fitting…"), kind="info"))

        def _on_progress(info):
            mo.output.replace(mo.callout(mo.md(_fmt_progress(info)), kind="info"))

        try:
            _fit_result_new = run_fit(
                config_dict=config_dict,
                compartments=list(compartments),
                is_metapop=is_metapop,
                targets=fit_targets,
                fit_config=fit_config_obj,
                compartment_init=fit_compartment_init,
                schedule_dfs=loaded_schedule_dfs,
                progress_callback=_on_progress,
                **fit_run_kwargs,
            )
        except (ValueError, RuntimeError) as _exc:
            mo.stop(True, mo.callout(mo.md(f"**Fitting error:** {_exc}"), kind="danger"))
        except Exception as _exc:
            import traceback as _tb
            mo.stop(
                True,
                mo.callout(
                    mo.md(f"**Fitting error:** {_exc}\n\n```\n{_tb.format_exc()}\n```"),
                    kind="danger",
                ),
            )
        set_fit_result_state({
            "result": _fit_result_new,
            "signature": fit_run_config_signature,
            "source": "run",
        })
    return


@app.cell
def _fitting_autosave(fit_result, fit_result_source, output_dir, fit_result_to_dict, json):
    # Only autosave results from a run in this session — an uploaded result
    # already lives in its own file and re-writing fitted_params.json from it
    # would silently overwrite whatever this session last actually computed.
    if fit_result is not None and fit_result_source == "run":
        _p = output_dir / "fitted_params.json"
        _p.write_text(json.dumps(fit_result_to_dict(fit_result), indent=2))
    return


@app.cell
def _fitting_results_display(
    fit_result, fit_result_is_stale, fit_result_source,
    fit_result_to_dict, np, plt, mo, main_tab, json,
):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None)

    _status_banner = mo.md("")
    if fit_result_source == "uploaded":
        _status_banner = mo.callout(
            mo.md(
                "**Showing a loaded fitting result** (uploaded file), not a run from "
                "this session. It may not correspond to the targets/parameters "
                "currently configured above."
            ),
            kind="info",
        )
    elif fit_result_is_stale:
        _status_banner = mo.callout(
            mo.md(
                "**Stale:** these results were produced with a different set of "
                "targets, parameters, or method than what's currently configured "
                "above. Re-run fitting to refresh them."
            ),
            kind="warn",
        )

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
    elif _method == "mcmc":
        # _lc is the mean log-probability across walkers per step.
        _axes[0].plot(_lc, linewidth=1.2, color="steelblue")
        _axes[0].set_ylabel("Mean log-posterior (across walkers)")
        _axes[0].set_title("MCMC convergence")
    elif _method == "abc-smc":
        # _lc is the ε (acceptance-threshold) schedule per generation.
        _axes[0].plot(range(1, len(_lc) + 1), _lc, marker="o", linewidth=1.5, color="darkorange")
        _axes[0].set_ylabel("ε (acceptance threshold, weighted 1−R²)")
        _axes[0].set_xlabel("Generation")
        _axes[0].set_title("ABC-SMC tolerance schedule")
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
        # Flat-fit baseline: the weighted MSE loss is normalised so that a model
        # predicting zero everywhere scores exactly 1.0. Anything plateauing at
        # or above this line has not found a meaningful fit. Drawn explicitly
        # (plus a log y-scale when the spread is large) so a converged-looking
        # plateau near 1.0 isn't visually mistaken for "loss → 0" when early
        # iterations overshoot into the hundreds.
        _axes[0].axhline(
            1.0, color="0.4", linestyle=":", linewidth=1.3,
            label="flat-fit baseline (loss = 1)",
        )
        _all_lv = [_v for _rep in _lc for _v in _rep if _v is not None and _v > 0]
        if _all_lv and max(_all_lv) / min(_all_lv) > 20:
            _axes[0].set_yscale("log")
        _best_final = min((_rep[-1] for _rep in _lc if _rep), default=None)
        if _best_final is not None:
            _axes[0].annotate(
                f"best final loss = {_best_final:.3g}",
                xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
            )
        _axes[0].legend(fontsize=8)
        _axes[0].set_ylabel("Weighted MSE loss")
    if _method in ("ar", "adam", "lbfgs"):
        _axes[0].set_xlabel("Iterations / Max samples")
        _axes[0].set_title(f"Fitting progress ({_method})")
    elif _method == "mcmc":
        _axes[0].set_xlabel("Step")
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
    elif _method == "mcmc":
        _accepted_note = mo.callout(
            mo.md(
                f"**{_n_runs:,} posterior draws** from the MCMC ensemble (post-burn-in, thinned, "
                "stuck walkers dropped). The parameter table shows the posterior median and "
                "central 95% credible interval; the distributions are plotted below. "
                "Enable **Start forecast from fitted end-state** in the Forecast tab to run a "
                "posterior-predictive ensemble."
            ),
            kind="success" if _n_runs > 1 else "info",
        )
    elif _method == "abc-smc":
        _accepted_note = mo.callout(
            mo.md(
                f"**{_n_runs:,} posterior particles** from the final ABC-SMC generation. "
                "The parameter table shows the posterior median and central 95% interval; "
                "the distributions are plotted below. Enable **Start forecast from fitted "
                "end-state** in the Forecast tab to run a posterior-predictive ensemble."
            ),
            kind="success" if _n_runs > 1 else "info",
        )

    _download = mo.download(
        data=json.dumps(fit_result_to_dict(fit_result), indent=2).encode(),
        filename="fitted_params.json",
        mimetype="application/json",
        label="Download fitted_params.json",
    )

    _loss_note = mo.md("")
    if _method in ("adam", "lbfgs"):
        _loss_note = mo.callout(
            mo.md(
                "The dotted **flat-fit baseline (loss = 1)** marks the loss of a model "
                "that predicts zero everywhere. A replication that settles at or above "
                "this line has **not** found a meaningful fit — a good calibration drives "
                "the loss well below 1. (Timeseries targets are now aligned to the "
                "simulation start date automatically, so the loss compares matching "
                "calendar dates.)"
            ),
            kind="info",
        )

    # ── "Fit configuration used" accordion — the request (bounds, granularity,
    # log-space, targets, method hyperparameters) that produced this result, so
    # a saved/uploaded result is self-describing rather than just the outcome.
    _fc = fit_result.fit_config or {}
    _config_accordion = mo.md("")
    if _fc:
        _param_rows = []
        for _pn in _fc.get("selected_params", []):
            _b = (_fc.get("bounds", {}) or {}).get(_pn)
            _b_str = f"[{_b[0]:.4g}, {_b[1]:.4g}]" if _b else "—"
            _dims = (_fc.get("param_dims", {}) or {}).get(_pn) or []
            _dims_str = ", ".join(_dims) if _dims else "scalar"
            _log_str = "log10" if _pn in (_fc.get("log_params") or []) else "linear"
            _param_rows.append(f"| `{_pn}` | {_b_str} | {_dims_str} | {_log_str} |")
        _params_table_md = (
            "| Parameter | Bounds | Granularity | Space |\n|---|---|---|---|\n" + "\n".join(_param_rows)
            if _param_rows else "*(none recorded)*"
        )

        _tgt_rows = []
        _fit_targets_cfg = fit_result.fit_targets or []
        _tgt_slices = fit_result.target_slices or []
        for _k in range(len(_fit_targets_cfg)):
            _vars_str = ", ".join(_fit_targets_cfg[_k])
            _lbl = (fit_result.target_labels or [])[_k] if _k < len(fit_result.target_labels or []) else f"Target {_k + 1}"
            _mode = (fit_result.target_modes or [])[_k] if _k < len(fit_result.target_modes or []) else "?"
            _wt = (fit_result.target_weights or [])[_k] if _k < len(fit_result.target_weights or []) else 1.0
            _slice = _tgt_slices[_k] if _k < len(_tgt_slices) else {}
            _slice_str = ", ".join(
                f"{_sk}={_sv}" for _sk, _sv in _slice.items() if _sv != -1
            ) or "All (sum)"
            _tgt_rows.append(f"| {_lbl} | {_vars_str} | {_mode} | {_wt:.3g} | {_slice_str} |")
        _targets_table_md = (
            "| Label | Variables | Mode | Weight | Slice |\n|---|---|---|---|---|\n" + "\n".join(_tgt_rows)
            if _tgt_rows else "*(none recorded)*"
        )

        _hyper_lines = [f"- **Method:** `{_fc.get('method', _method)}`", f"- **Simulation days:** {_fc.get('sim_days', '—')}"]
        if _method in ("adam", "lbfgs"):
            _hyper_lines += [
                f"- **Learning rate:** {_fc.get('lr')}",
                f"- **Iterations:** {_fc.get('n_iter')}",
                f"- **Replications:** {_fc.get('n_replications')}",
                f"- **Robust steps:** {_fc.get('robust_steps')}",
            ]
        elif _method == "ar":
            _hyper_lines += [
                f"- **Samples:** {_fc.get('n_iter')}",
                f"- **R² threshold:** {_fc.get('r2_threshold')}",
            ]
        elif _method == "mcmc":
            _hyper_lines += [
                f"- **Walkers:** {_fc.get('n_walkers')}",
                f"- **Steps:** {_fc.get('n_iter')}",
                f"- **Burn-in:** {_fc.get('mcmc_burnin')}",
                f"- **Thinning:** {_fc.get('mcmc_thin')}",
            ]
        elif _method == "abc-smc":
            _hyper_lines += [
                f"- **Population size:** {_fc.get('abc_pop_size')}",
                f"- **Max generations:** {_fc.get('abc_max_gens')}",
            ]
        if _fc.get("tv_transmission"):
            _hyper_lines.append(
                f"- **Time-varying m(t):** knot spacing {_fc.get('tv_knot_spacing_days')} days, "
                f"τ = {_fc.get('tv_tau')}"
            )
        if _fc.get("fit_start_offset"):
            _sob = _fc.get("start_offset_bounds", [None, None])
            _hyper_lines.append(f"- **Epidemic start offset:** [{_sob[0]}, {_sob[1]}] days")
        if _fc.get("scale_groups"):
            _hyper_lines.append("- **Linked-scale groups:** " + "; ".join(
                f"`{_g}` → {', '.join(_bases)}" for _g, _bases in _fc.get("scale_groups", {}).items()
            ))

        _config_accordion = mo.accordion({
            "Fit configuration used": mo.vstack([
                mo.md("\n".join(_hyper_lines)),
                mo.md("**Targets**"), mo.md(_targets_table_md),
                mo.md("**Fitted parameters**"), mo.md(_params_table_md),
            ]),
        })

    mo.vstack([
        mo.md("## Fitting Results"), _status_banner, _fig, _loss_note, mo.md(_params_md),
        _accepted_note, _ar_multi_note, _config_accordion, _download,
    ])
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

    _MULTI_METHODS = ("ar", "mcmc", "abc-smc")
    _run_label = {"ar": "accepted", "mcmc": "posterior draw",
                  "abc-smc": "posterior draw"}.get(_method, "replication")

    _style_ui = mo.md("")
    if _method in _MULTI_METHODS or (_method in ("adam", "lbfgs") and len(_trajs) > 1):
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
            if _method in _MULTI_METHODS and len(_trajs) > 1:
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
            if _method in _MULTI_METHODS and len(_trajs) > 1:
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
def _fitting_mt_display(fit_result, fit_tv_spacing, np, plt, mo, main_tab):
    # Time-varying transmission m(t): shown only when the fit produced m(t)
    # log-increment parameters (m_dlog_*), i.e. the Bayesian samplers with
    # "Fit time-varying transmission m(t)" enabled.
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))
    _acc = fit_result.accepted_params or []
    _incr_keys = sorted(
        [_k for _k in (_acc[0].keys() if _acc else []) if _k.startswith("m_dlog_")],
        key=lambda _s: int(_s.split("_")[-1]),
    )
    mo.stop(not _incr_keys, mo.md(""))

    from generic_core.fitting import build_transmission_multiplier_array, _tv_knot_days

    _num_days = int(fit_result.num_days)
    _n_incr = len(_incr_keys)
    _knots = _tv_knot_days(_num_days, int(fit_tv_spacing.value))
    if len(_knots) - 1 != _n_incr:  # spacing changed since the run — rebuild a matching grid
        _knots = np.linspace(0, _num_days - 1, _n_incr + 1).round().astype(int).tolist()

    _curves = np.array([
        build_transmission_multiplier_array(
            [float(_s.get(_k, 0.0)) for _k in _incr_keys], _knots, _num_days)
        for _s in _acc
    ])
    _days = np.arange(_num_days)
    _med = np.median(_curves, axis=0)
    _lo = np.percentile(_curves, 2.5, axis=0)
    _hi = np.percentile(_curves, 97.5, axis=0)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    _ax.fill_between(_days, _lo, _hi, color="seagreen", alpha=0.25, label="95% CI")
    _ax.plot(_days, _med, color="seagreen", linewidth=2, label="median m(t)")
    _ax.set_xlabel("Day")
    _ax.set_ylabel("m(t) transmission multiplier")
    _ax.set_title("Fitted time-varying transmission multiplier m(t)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        mo.md("### Time-varying Transmission m(t)"),
        mo.callout(
            mo.md(
                "m(t) multiplies the force of infection over the simulation "
                "(**1.0 = the baseline β**): values above 1 raise transmission, below 1 "
                f"lower it. Knots sit every **{int(fit_tv_spacing.value)} days**; between "
                "knots m(t) is **interpolated, not constant** — piecewise-linear in log "
                "space, so it ramps smoothly toward the next knot rather than jumping on "
                "the knot date. m(t) is anchored to **1.0 at day 0** (start of the fit "
                "window).\n\n"
                "The knot-to-knot log-increments are fit with an independent, **zero-mean "
                "Gaussian random-walk prior** (scale τ) — this only shrinks each *step* "
                "toward 'no change'; there is **no mean-reversion pulling m(t) itself back "
                "toward 1** over time, so the fitted curve can drift and stay away from 1 "
                "for as long as the data support it.\n\n"
                "The shaded band is the 95% credible interval across the posterior; it "
                "widens where the data don't constrain transmission (typically the "
                "low-incidence tail)."
            ),
            kind="info",
        ),
        _fig,
    ])
    return


@app.cell
def _fitting_pairplot(fit_result, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Fitting", None)
    mo.stop(fit_result is None, mo.md(""))

    _accepted = fit_result.accepted_params
    mo.stop(not _accepted or len(_accepted) <= 1, mo.md(""))

    _first = _accepted[0]
    # m_dlog_* time-varying-transmission increments are excluded here: with
    # monthly knots over a long fit window there can be a dozen+ of them,
    # blowing up the n×n grid into a huge, slow-to-render raster (marimo caps
    # output size and refuses to display it past ~35MB for a busy MCMC corner
    # plot). They're already visualized as the assembled m(t) curve above.
    _scalar_keys = [
        _k for _k, _v in _first.items()
        if isinstance(_v, (int, float)) and not _k.startswith("m_dlog_")
    ]
    mo.stop(not _scalar_keys, mo.md(""))

    _data = np.array([[float(_s.get(_k, float("nan"))) for _k in _scalar_keys] for _s in _accepted])
    _n = len(_scalar_keys)
    _method = fit_result.method
    _run_noun = {"ar": "accepted", "mcmc": "posterior draw",
                 "abc-smc": "posterior draw"}.get(_method, "replication")
    _is_posterior = _method in ("mcmc", "abc-smc")

    # Cap rendered output size two ways, independent of the fix above (which
    # only helps the m_dlog_* case): (1) subsample the scatter/histogram
    # points — thousands of overlapping alpha-blended dots inflate PNG size
    # far more than they add visual information; (2) cap the total figure
    # footprint so a large parameter count (e.g. per-age/risk/subpop
    # granularity) can't blow the panel size past what marimo will display.
    _MAX_POINTS = 3000
    if len(_data) > _MAX_POINTS:
        _rng = np.random.default_rng(0)
        _plot_idx = _rng.choice(len(_data), size=_MAX_POINTS, replace=False)
        _plot_data = _data[_plot_idx]
    else:
        _plot_data = _data
    _panel_size = min(3.0, 18.0 / max(_n, 1))
    _fig_dpi = 100 if _n <= 8 else max(60, int(800 / _n))

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
        _ax.set_title(f"Parameter distribution ({len(_accepted)} {_run_noun}(s))")
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        _fig, _axs = plt.subplots(
            _n, _n, figsize=(_panel_size * _n, _panel_size * _n), dpi=_fig_dpi,
        )
        _alpha_sc = min(0.8, 30.0 / max(len(_plot_data), 1))
        for _row in range(_n):
            for _col in range(_n):
                _ax = _axs[_row, _col]
                if _row == _col:
                    _draw_density(_ax, _data[:, _row])
                else:
                    _ax.scatter(
                        _plot_data[:, _col], _plot_data[:, _row],
                        alpha=_alpha_sc, s=14,
                        color="steelblue", edgecolors="none",
                        rasterized=True,
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
            f"Parameter distributions ({len(_accepted)} {_run_noun}(s))"
            + (f" — {_MAX_POINTS:,} shown" if len(_data) > _MAX_POINTS else ""),
            y=1.01, fontsize=11,
        )
        plt.tight_layout()

    _title = "### Posterior Parameter Distributions" if _is_posterior else "### Accepted Parameter Distributions"
    _corner_help = mo.accordion({
        "How to read this corner plot": mo.md(
            "This is a **corner plot** (pair plot) of the "
            f"{len(_accepted):,} {_run_noun} parameter sets:\n\n"
            "- **Diagonal** — the 1-D distribution of each parameter on its own "
            "(histogram + smoothed density). For the Bayesian methods this is the "
            "**marginal posterior**: a narrow, single peak means the data pin the "
            "parameter down; a broad or rail-hugging spread means it's weakly "
            "identified.\n"
            "- **Off-diagonal** — each scatter shows the joint distribution of a "
            "*pair* of parameters. A tilted, cigar-shaped cloud means the two are "
            "**correlated / trading off** (e.g. a higher transmission rate "
            "compensated by a lower seed size) — the data constrain their "
            "combination better than either alone. A round blob means they're "
            "roughly independent.\n\n"
            + ("- **`m_dlog_*`** time-varying-transmission log-increments are "
               "omitted from this plot (there can be a dozen+ with monthly knots, "
               "which would make the grid unreadable and slow to render) — see the "
               "m(t) plot above for the assembled curve instead. "
               "**`phi`** is the Negative-Binomial dispersion (smaller ⇒ noisier data) "
               "and is still shown below.\n\n"
               if _is_posterior else "")
            + (f"- Showing a random subsample of **{_MAX_POINTS:,}** of the "
               f"{len(_accepted):,} {_run_noun}s for the scatter panels (marginal "
               "histograms/densities still use the full set).\n\n"
               if len(_data) > _MAX_POINTS else "")
            + "Well-identified fits show tight diagonals; wide marginals or strong "
            "off-diagonal correlations flag parameters the data can't separate."
        )
    })

    mo.vstack([
        mo.md(_title),
        _corner_help,
        _fig,
    ])
    return


# ============================================================
# Forecast tab
# ============================================================

