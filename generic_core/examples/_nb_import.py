# _nb_import.py
# Section: Shared multi-file config importer (model/fit/fitted-params/scenario)
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _shared_import_state(mo):
    # {"model_config": {"name": str, "contents": bytes}, "fit_config": {...},
    # "fitted_params": {...}, "scenario_config": {...}} -- one entry per
    # config type, holding the bytes of the last file applied under that
    # type. Read as a fallback source by _load_config_parse (Model Builder)
    # when neither a direct browse nor a path is set; the other three types
    # are applied straight into their own tab's existing restore state at
    # click time in _shared_import_apply below, since those already need
    # one-shot "apply, don't keep re-applying on every unrelated rerun"
    # semantics (same as each tab's own upload widget).
    get_shared_imports, set_shared_imports = mo.state({})
    # Staged files awaiting Apply: [{"name": str, "contents": bytes, "type":
    # str}, ...]. Built additively across possibly-many browse actions (see
    # _shared_import_upload_ui) rather than read straight off the file
    # widget's .value, since re-opening the browser dialog replaces .value
    # wholesale -- without this, picking files from a second folder would
    # silently drop whatever was picked from the first.
    get_shared_import_files, set_shared_import_files = mo.state([])
    return (
        get_shared_imports, set_shared_imports,
        get_shared_import_files, set_shared_import_files,
    )


@app.cell
def _shared_import_upload_ui(
    mo, detect_config_type, get_shared_import_files, set_shared_import_files,
):
    # Runs only on a genuine file-selection event from the browser (mo.ui.file
    # calls on_change from its own _update(), never from an unrelated cell
    # rerun) -- same reasoning as the Fitting tab's bulk CSV uploader. That's
    # what makes this additive: each browse only ever hands us the files
    # picked in that one dialog, so merging them into the staged list (rather
    # than replacing it) is what lets the user pick files from separate
    # folders across more than one browse.
    def _on_upload(_files):
        _files = _files or ()
        if not _files:
            return
        def _update(_cur):
            _new = list(_cur)
            _seen = {(_e["name"], _e["contents"]) for _e in _new}
            for _f in _files:
                _key = (_f.name, _f.contents)
                if _key in _seen:
                    continue
                _new.append({
                    "name": _f.name,
                    "contents": _f.contents,
                    "type": detect_config_type(_f.name),
                })
                _seen.add(_key)
            return _new
        set_shared_import_files(_update)

    shared_import_upload = mo.ui.file(
        multiple=True,
        filetypes=[".json"],
        label="Import config files",
        on_change=_on_upload,
    )
    shared_import_upload_note = mo.md(
        "Drop any combination of a model config, fit config, fitted "
        "params / fit result, and scenario config -- browsing again adds "
        "to the files below rather than replacing them, so files from "
        "separate folders can be picked one browse at a time. Once you "
        "confirm each file's type below, click Apply to import it."
    )
    return shared_import_upload, shared_import_upload_note


@app.cell
def _shared_import_rows_ui(mo, get_shared_import_files, set_shared_import_files):
    # One dropdown + remove button per staged file. The dropdown is pre-set
    # to a filename-based guess (see detect_config_type) but always
    # user-confirmable before Apply -- the guess is just a time-saver, never
    # applied blind. Selecting a type writes straight into the staged-files
    # state (rather than being read back out by the Apply cell via a
    # separate array), so a file's chosen type survives later browses/
    # removals instead of resetting to the guess every time this cell
    # rebuilds.
    _type_opts = {
        "Model config": "model_config",
        "Fit config": "fit_config",
        "Fitted params / fit result": "fitted_params",
        "Scenario config": "scenario_config",
        "Skip (ignore this file)": "skip",
    }
    _label_by_value = {v: k for k, v in _type_opts.items()}
    _singular_types = {"model_config", "fit_config", "fitted_params", "scenario_config"}

    def _set_type(_idx):
        def _on_change(_val):
            def _update(_cur):
                if _idx >= len(_cur):
                    return _cur
                _new = list(_cur)
                _new[_idx] = {**_new[_idx], "type": _val}
                return _new
            set_shared_import_files(_update)
        return _on_change

    def _remove(_idx):
        def _on_click(_):
            def _update(_cur):
                return _cur[:_idx] + _cur[_idx + 1:]
            set_shared_import_files(_update)
        return _on_click

    _files = get_shared_import_files()
    shared_import_type_sels = mo.ui.array([
        mo.ui.dropdown(
            options=_type_opts,
            value=_label_by_value.get(_e["type"], "Skip (ignore this file)"),
            label=_e["name"],
            on_change=_set_type(_i),
        )
        for _i, _e in enumerate(_files)
    ])
    shared_import_remove_btns = [
        mo.ui.button(label="✕", tooltip=f"Remove {_e['name']}", on_click=_remove(_i))
        for _i, _e in enumerate(_files)
    ]
    shared_import_rows = [
        mo.hstack([_dd, _btn], justify="start", align="center", gap=1)
        for _dd, _btn in zip(shared_import_type_sels, shared_import_remove_btns)
    ]

    # Guard against e.g. two Fit config files being staged at once -- Apply
    # only ever applies one file per type (see _shared_import_apply), so
    # surface the conflict here before the user clicks it.
    _type_counts = {}
    for _e in _files:
        if _e["type"] in _singular_types:
            _type_counts[_e["type"]] = _type_counts.get(_e["type"], 0) + 1
    _dupe_labels = [_label_by_value[_t] for _t, _c in _type_counts.items() if _c > 1]
    shared_import_dupe_warning = (
        mo.callout(
            mo.md(
                "More than one file is set to: " + ", ".join(_dupe_labels) + ". "
                "Only one file per type is applied -- change the type on all "
                "but one (or remove it) before clicking Apply."
            ),
            kind="warn",
        )
        if _dupe_labels else mo.md("")
    )
    return (
        shared_import_type_sels, shared_import_remove_btns,
        shared_import_rows, shared_import_dupe_warning,
    )


@app.cell
def _shared_import_apply_button(mo):
    shared_import_apply_btn = mo.ui.run_button(label="Apply imported configs")
    return (shared_import_apply_btn,)


@app.cell
def _shared_import_apply(
    mo, json,
    shared_import_apply_btn,
    get_shared_import_files, set_shared_import_files,
    get_shared_imports, set_shared_imports,
    parse_fit_config_targets, partition_scenario_state,
    set_target_slots, set_restored_target_data, set_restore_error, set_restored_config,
    set_fit_result_state, fit_result_from_dict,
    set_scenario_controls_state, set_scenario_agescale_state,
    set_scenario_dose_state, set_scenario_subpop_state, set_scenario_restore_error,
    set_config_path,
):
    _TYPE_LABELS = {
        "model_config": "Model config",
        "fit_config": "Fit config",
        "fitted_params": "Fitted params / fit result",
        "scenario_config": "Scenario config",
    }
    shared_import_apply_note = mo.md("")
    if shared_import_apply_btn.value:
        _entries = get_shared_import_files()
        _bytes_by_type = dict(get_shared_imports())
        _applied = []
        _errors = []
        # Entries that stay staged after this click -- only ones that
        # couldn't be applied (parse error, or a same-type conflict), so the
        # user can fix and re-apply without having to re-browse everything.
        _kept = []

        _type_counts = {}
        for _e in _entries:
            if _e["type"] in _TYPE_LABELS:
                _type_counts[_e["type"]] = _type_counts.get(_e["type"], 0) + 1
        _dupe_types = {t for t, c in _type_counts.items() if c > 1}

        for _e in _entries:
            _kind = _e["type"]
            _name = _e["name"]
            if _kind == "skip":
                continue
            if _kind in _dupe_types:
                _errors.append(
                    f"**{_name}**: multiple {_TYPE_LABELS[_kind]} files staged "
                    "-- resolve before applying"
                )
                _kept.append(_e)
                continue
            try:
                _raw = json.loads(_e["contents"].decode("utf-8"))
            except Exception as _exc:
                _errors.append(f"**{_name}**: JSON parse error: {_exc}")
                _kept.append(_e)
                continue

            if _kind == "model_config":
                # Applied lazily by _load_config_parse (it re-reads this
                # dict every time it reruns), not here -- no one-shot state
                # to mutate for this type. But that cell only falls back to
                # the shared import when BOTH the browsed-file widget and the
                # path text box are empty, and the path box defaults to the
                # bundled example config (non-empty) -- so without clearing
                # it here, the import would silently never take effect.
                _bytes_by_type["model_config"] = {"name": _name, "contents": _e["contents"]}
                set_config_path("")
            elif _kind == "fit_config":
                try:
                    _new_slots, _new_data = parse_fit_config_targets(_raw)
                except Exception as _exc:
                    _errors.append(f"**{_name}**: {_exc}")
                    _kept.append(_e)
                    continue
                set_target_slots(_new_slots)
                set_restored_target_data(_new_data)
                set_restore_error(None)
                set_restored_config(_raw)
            elif _kind == "fitted_params":
                try:
                    # Same leniency as the Analysis tab's own "Upload JSON
                    # file" fitted-params mode: accept either a full
                    # fit_result_to_dict export or a bare {param: value} dict.
                    _for_fit = _raw if "best_params" in _raw else {"best_params": _raw}
                    _loaded = fit_result_from_dict(_for_fit)
                except Exception as _exc:
                    _errors.append(f"**{_name}**: {_exc}")
                    _kept.append(_e)
                    continue
                set_fit_result_state({"result": _loaded, "signature": None, "source": "uploaded"})
            elif _kind == "scenario_config":
                if not isinstance(_raw, dict):
                    _errors.append(f"**{_name}**: expected a JSON object")
                    _kept.append(_e)
                    continue
                _groups = partition_scenario_state(_raw)
                set_scenario_controls_state(_groups["controls"])
                set_scenario_agescale_state(_groups["agescale"])
                set_scenario_dose_state(_groups["dose"])
                set_scenario_subpop_state(_groups["subpop"])
                set_scenario_restore_error(None)

            _applied.append(f"**{_name}** → {_TYPE_LABELS.get(_kind, _kind)}")

        set_shared_imports(_bytes_by_type)
        set_shared_import_files(_kept)

        _parts = []
        if _applied:
            _parts.append("Imported " + ", ".join(_applied) + ".")
        if _errors:
            _parts.append("Failed: " + "; ".join(_errors))
        if not _applied and not _errors:
            _parts.append("Nothing to import -- set a type per file above (or add files first).")
        shared_import_apply_note = mo.callout(
            mo.md(" ".join(_parts)),
            kind="warn" if _errors else ("success" if _applied else "info"),
        )
    return (shared_import_apply_note,)
