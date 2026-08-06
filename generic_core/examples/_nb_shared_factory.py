# _nb_shared_factory.py
# Section: Shared model factory functions
# Part of model_builder_notebook.py — assembled by build_notebook.py
#
# Thin notebook-context wrappers around generic_core.model_factory (the model-
# building logic itself lives there, with no marimo dependency, so it can be
# reused by generic_core.fitting and by exported standalone scripts).

@app.cell
def _shared_model_factory(
    loaded_schedule_dfs,
    mobility_input,
    daily_vaccines_input,
    num_age_groups,
    num_risk_groups,
):
    from functools import partial
    from generic_core import model_factory as _mf

    _A = num_age_groups
    _R = num_risk_groups

    make_single_pop_metapop = partial(
        _mf.make_single_pop_metapop,
        num_age_groups=_A, num_risk_groups=_R,
        mobility_value=float(mobility_input.value),
        daily_vaccines_value=float(daily_vaccines_input.value),
        schedule_dfs=loaded_schedule_dfs,
    )
    make_metapop_from_folder = partial(
        _mf.make_metapop_from_folder,
        num_age_groups=_A, num_risk_groups=_R,
        mobility_value=float(mobility_input.value),
        daily_vaccines_value=float(daily_vaccines_input.value),
    )
    extract_history = _mf.extract_history

    return (make_single_pop_metapop, make_metapop_from_folder, extract_history)


# ============================================================
# Fitting tab
# ============================================================

