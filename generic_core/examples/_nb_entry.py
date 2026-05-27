# _nb_entry.py
# Section: Notebook entry-point UI (tab selector, output directory, autosave)
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _main_tab_selector(mo):
    main_tab = mo.ui.tabs({
        "Model Builder": mo.md(""),
        "Fitting":        mo.md(""),
        "Forecast":       mo.md(""),
        "Export":         mo.md(""),
        "Analysis":       mo.md(""),
        "Documentation":  mo.md(""),
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
def _tab_header_display(main_tab, output_dir_input, mo):
    mo.vstack([
        main_tab,
        mo.hstack([output_dir_input], justify="start"),
    ])
    return


@app.cell
def _autosave_config(config_dict, output_dir, json):
    _p = output_dir / "model_config.json"
    _ = _p.write_text(json.dumps(config_dict, indent=2))
    return

