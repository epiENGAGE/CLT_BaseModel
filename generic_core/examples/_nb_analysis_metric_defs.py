# _nb_analysis_metric_defs.py
# Section: Analysis metric definition widgets
# Part of model_builder_notebook.py — assembled by build_notebook.py
#
# IMPORTANT: These cells must be assembled BEFORE _nb_model_builder.py because
# _build_config depends on analysis_n_metrics_input, analysis_metric_names,
# and analysis_metric_tvs.

@app.cell
def _analysis_metric_defs_ui(mo, loaded_config, n_transitions, t_name, transition_vars_input):
    _MAX_MET = 5
    _saved = loaded_config.get("analysis_metrics", [])
    analysis_n_metrics_input = mo.ui.number(
        start=1, stop=_MAX_MET, step=1,
        value=min(max(len(_saved), 1), _MAX_MET),
        label="Number of user-defined metrics",
    )
    analysis_metric_names = mo.ui.array([
        mo.ui.text(
            value=_saved[i]["name"] if i < len(_saved) else f"metric_{i + 1}",
            label="Name",
        )
        for i in range(_MAX_MET)
    ])
    _tvs_explicit = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    tv_opts = _tvs_explicit if _tvs_explicit else [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    analysis_metric_tvs = mo.ui.array([
        mo.ui.multiselect(
            options=tv_opts if tv_opts else [""],
            value=[v for v in (_saved[i].get("transition_variables", []) if i < len(_saved) else []) if v in tv_opts],
            label="Transition variables to sum",
        )
        for i in range(_MAX_MET)
    ])
    return analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs, tv_opts


@app.cell
def _analysis_metric_sel_state(mo):
    get_sel_metrics, set_sel_metrics = mo.state([])
    return get_sel_metrics, set_sel_metrics


@app.cell
def _analysis_metric_plot_controls(mo, analysis_n_metrics_input, analysis_metric_names, get_sel_metrics, set_sel_metrics):
    _n = int(analysis_n_metrics_input.value)
    _opts = [analysis_metric_names.value[i].strip() or f"metric_{i + 1}" for i in range(_n)]
    _saved = [m for m in get_sel_metrics() if m in _opts]
    analysis_plot_metric_sel = mo.ui.multiselect(
        options=_opts if _opts else ["(no metrics defined)"],
        value=_saved if _saved else (_opts[:1] if _opts else []),
        on_change=set_sel_metrics,
        label="Metric(s) to show in plots",
    )
    return (analysis_plot_metric_sel,)

