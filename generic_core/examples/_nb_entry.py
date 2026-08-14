# _nb_entry.py
# Section: Notebook entry-point UI (tab selector, output directory, autosave)
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _tab_style_injection(mo):
    # Must render in its own cell, before the tabs widget's cell, so this
    # stylesheet already exists in the document when marimo-tabs constructs
    # its shadow root and copies stylesheets into it (a one-time, synchronous
    # snapshot at construction — added later is too late).
    mo.Html(
        '<style title="marimo-tab-width">'
        '[role="tablist"] { width: 100%; }'
        '[role="tablist"] [role="tab"] { flex: 1; text-align: center; }'
        "</style>"
    )
    return


@app.cell
def _main_tab_selector(mo):
    main_tab = mo.ui.tabs({
        "Population & Geography": mo.md(""),
        "Model Builder": mo.md(""),
        "Analysis":      mo.md(""),
        "Fitting":       mo.md(""),
        "Forecast":      mo.md(""),
        "Export":        mo.md(""),
        "Documentation": mo.md(""),
    })
    return (main_tab,)


@app.cell
def _output_dir_ui(mo, Path):
    output_dir_input = mo.ui.text(
        value=str(Path.home() / "clt_outputs"),
        label="Output directory (auto-saves go here)",
        full_width=True,
    )
    return (output_dir_input,)


@app.cell
def _output_dir(output_dir_input, Path):
    output_dir = Path(output_dir_input.value).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir,)


@app.cell
def _tab_header_display(
    main_tab, output_dir_input, mo,
    shared_import_upload, shared_import_type_sels,
    shared_import_apply_btn, shared_import_apply_note,
):
    mo.vstack([
        main_tab,
        mo.hstack([output_dir_input], justify="start"),
        mo.accordion({
            "Import config files": mo.vstack([
                shared_import_upload,
                mo.vstack(list(shared_import_type_sels)) if len(shared_import_type_sels) else mo.md(""),
                shared_import_apply_btn if len(shared_import_type_sels) else mo.md(""),
                shared_import_apply_note,
            ]),
        }),
    ])
    return


@app.cell
def _autosave_config(config_dict, output_dir, json):
    _p = output_dir / "model_config.json"
    _ = _p.write_text(json.dumps(config_dict, indent=2))
    return

