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
    return get_shared_imports, set_shared_imports


@app.cell
def _shared_import_ui(mo):
    shared_import_upload = mo.ui.file(
        multiple=True,
        filetypes=[".json"],
        label=(
            "Import config files (drop any combination of a model config, "
            "fit config, fitted params / fit result, and scenario config)"
        ),
    )
    return (shared_import_upload,)


@app.cell
def _shared_import_type_dropdowns(mo, shared_import_upload, detect_config_type):
    # One dropdown per uploaded file, pre-set to a filename-based guess (see
    # detect_config_type) but always user-confirmable before Apply -- the
    # guess is just a time-saver, never applied blind. Rebuilt fresh whenever
    # the file selection changes, same as the Fitting tab's per-slot target
    # widgets; there's nothing here worth persisting past a single import.
    _files = shared_import_upload.value or ()
    _type_opts = {
        "Model config": "model_config",
        "Fit config": "fit_config",
        "Fitted params / fit result": "fitted_params",
        "Scenario config": "scenario_config",
        "Skip (ignore this file)": "skip",
    }
    _label_by_value = {v: k for k, v in _type_opts.items()}
    shared_import_type_sels = mo.ui.array([
        mo.ui.dropdown(
            options=_type_opts,
            value=_label_by_value.get(detect_config_type(_f.name), "Skip (ignore this file)"),
            label=_f.name,
        )
        for _f in _files
    ])
    return (shared_import_type_sels,)


@app.cell
def _shared_import_apply_button(mo):
    shared_import_apply_btn = mo.ui.run_button(label="Apply imported configs")
    return (shared_import_apply_btn,)


@app.cell
def _shared_import_apply(
    mo, json,
    shared_import_apply_btn, shared_import_upload, shared_import_type_sels,
    get_shared_imports, set_shared_imports,
    parse_fit_config_targets, partition_scenario_state,
    set_target_slots, set_restored_target_data, set_restore_error, set_restored_config,
    set_fit_result_state, fit_result_from_dict,
    set_scenario_controls_state, set_scenario_agescale_state,
    set_scenario_dose_state, set_scenario_subpop_state, set_scenario_restore_error,
    set_config_path,
):
    shared_import_apply_note = mo.md("")
    if shared_import_apply_btn.value:
        _files = shared_import_upload.value or ()
        _bytes_by_type = dict(get_shared_imports())
        _applied = []
        _errors = []

        for _f, _sel in zip(_files, shared_import_type_sels):
            _kind = _sel.value
            if _kind == "skip":
                continue
            try:
                _raw = json.loads(_f.contents.decode("utf-8"))
            except Exception as _exc:
                _errors.append(f"**{_f.name}**: JSON parse error: {_exc}")
                continue

            if _kind == "model_config":
                # Applied lazily by _load_config_parse (it re-reads this
                # dict every time it reruns), not here -- no one-shot state
                # to mutate for this type. But that cell only falls back to
                # the shared import when BOTH the browsed-file widget and the
                # path text box are empty, and the path box defaults to the
                # bundled example config (non-empty) -- so without clearing
                # it here, the import would silently never take effect.
                _bytes_by_type["model_config"] = {"name": _f.name, "contents": _f.contents}
                set_config_path("")
            elif _kind == "fit_config":
                try:
                    _new_slots, _new_data = parse_fit_config_targets(_raw)
                except Exception as _exc:
                    _errors.append(f"**{_f.name}**: {_exc}")
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
                    _errors.append(f"**{_f.name}**: {_exc}")
                    continue
                set_fit_result_state({"result": _loaded, "signature": None, "source": "uploaded"})
            elif _kind == "scenario_config":
                if not isinstance(_raw, dict):
                    _errors.append(f"**{_f.name}**: expected a JSON object")
                    continue
                _groups = partition_scenario_state(_raw)
                set_scenario_controls_state(_groups["controls"])
                set_scenario_agescale_state(_groups["agescale"])
                set_scenario_dose_state(_groups["dose"])
                set_scenario_subpop_state(_groups["subpop"])
                set_scenario_restore_error(None)

            _applied.append(f"**{_f.name}** → {_kind}")

        set_shared_imports(_bytes_by_type)

        _parts = []
        if _applied:
            _parts.append("Imported " + ", ".join(_applied) + ".")
        if _errors:
            _parts.append("Failed: " + "; ".join(_errors))
        if not _applied and not _errors:
            _parts.append("Nothing to import -- set a type per file above (or upload files first).")
        shared_import_apply_note = mo.callout(
            mo.md(" ".join(_parts)),
            kind="warn" if _errors else ("success" if _applied else "info"),
        )
    return (shared_import_apply_note,)
