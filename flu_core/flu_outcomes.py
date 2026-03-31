"""
flu_outcomes.py
===============

Computation functions and plotting wrappers for vaccination scenario analysis.

Computation functions translate raw simulation output (compartment histories
and transition variable histories) into epidemiological quantities needed for
a vaccination impact report.

All computation functions accept optional ``subpop_name``, ``age_group``, and
``risk_group`` arguments; passing ``None`` sums across that dimension.

Plotting functions produce matplotlib figures and accept a single
``FluMetapopModel`` or a dict of models for multi-scenario overlays.

Important
---------
Hospital admissions and new infections use **transition variable histories**
(``ISH_to_HR``, ``ISH_to_HD``, ``S_to_E``, ``HD_to_D``), not compartment
differencing.  These must appear in
``SimulationSettings.transition_variables_to_save`` before the simulation runs.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Optional, Callable

from clt_toolkit.base_components import MetapopModel
from clt_toolkit.utils import daily_sum_over_timesteps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_subpop(metapop_model: MetapopModel):
    """Return the first SubpopModel from the metapopulation."""
    return next(iter(metapop_model.subpop_models.values()))


def _timesteps_per_day(metapop_model: MetapopModel) -> int:
    return _first_subpop(metapop_model).simulation_settings.timesteps_per_day


def _tvar_daily(
    metapop_model: MetapopModel,
    tvar_names: list,
    subpop_name: Optional[str] = None,
) -> np.ndarray:
    """
    Return an (days, A, R) array of daily totals for one or more transition
    variables, summed across the selected subpopulations.

    Parameters
    ----------
    metapop_model : MetapopModel
    tvar_names : list[str]
        Names of transition variables to sum together (e.g.
        ``["ISH_to_HR", "ISH_to_HD"]``).
    subpop_name : str or None
        Restrict to one named subpopulation; ``None`` sums all.

    Returns
    -------
    np.ndarray, shape (days, A, R)
    """
    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    n_per_day = _timesteps_per_day(metapop_model)

    arrays = []
    for subpop in subpops:
        for name in tvar_names:
            tvar = subpop.transition_variables[name]
            if not tvar.history_vals_list:
                raise ValueError(
                    f"Transition variable '{name}' has no saved history. "
                    f"Add it to SimulationSettings.transition_variables_to_save "
                    f"before running the simulation."
                )
            arrays.append(np.asarray(tvar.history_vals_list))

    total = np.sum(np.stack(arrays, axis=0), axis=0)  # (T, A, R)
    return daily_sum_over_timesteps(total, n_per_day)  # (days, A, R)


def _apply_ar_filter(
    arr: np.ndarray,
    age_group: Optional[int],
    risk_group: Optional[int],
) -> np.ndarray:
    """
    Reduce (days, A, R) → (days,) by slicing and then summing age/risk dims.
    """
    if age_group is not None:
        arr = arr[:, age_group : age_group + 1, :]
    if risk_group is not None:
        arr = arr[:, :, risk_group : risk_group + 1]
    return arr.sum(axis=(1, 2))


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------

def daily_hospital_admissions(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> np.ndarray:
    """
    Daily new hospital admissions.

    Computed as the sum of ISH→HR and ISH→HD transition flows,
    summed across subpopulations and timesteps-per-day.

    Parameters
    ----------
    metapop_model : MetapopModel
    subpop_name : str or None
        Restrict to one subpopulation; None sums all.
    age_group : int or None
        Index of age group; None sums all age groups.
    risk_group : int or None
        Index of risk group; None sums all risk groups.

    Returns
    -------
    np.ndarray, shape (days,)
    """
    arr = _tvar_daily(metapop_model, ["ISH_to_HR", "ISH_to_HD"], subpop_name)
    return _apply_ar_filter(arr, age_group, risk_group)


def daily_new_infections(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> np.ndarray:
    """
    Daily new infections = S→E transition flows, aggregated to daily totals.

    Returns
    -------
    np.ndarray, shape (days,)
    """
    arr = _tvar_daily(metapop_model, ["S_to_E"], subpop_name)
    return _apply_ar_filter(arr, age_group, risk_group)


def cumulative_hospitalizations(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> float:
    """
    Season-total hospital admissions (scalar).
    """
    return float(
        daily_hospital_admissions(metapop_model, subpop_name, age_group, risk_group).sum()
    )


def daily_deaths(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> np.ndarray:
    """
    Daily deaths = HD→D transition flows, aggregated to daily totals.

    Returns
    -------
    np.ndarray, shape (days,)
    """
    arr = _tvar_daily(metapop_model, ["HD_to_D"], subpop_name)
    return _apply_ar_filter(arr, age_group, risk_group)


def cumulative_deaths(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> float:
    """
    Season-total deaths = sum of HD→D transition flows (scalar).
    """
    arr = _tvar_daily(metapop_model, ["HD_to_D"], subpop_name)
    return float(_apply_ar_filter(arr, age_group, risk_group).sum())


def attack_rate(
    metapop_model: MetapopModel,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
) -> float:
    """
    Attack rate = cumulative infections / initial susceptible population.

    Returns
    -------
    float
    """
    infections = daily_new_infections(
        metapop_model, subpop_name, age_group, risk_group
    ).sum()

    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    # Initial susceptible = S compartment at first recorded timestep
    init_S_arrays = [
        np.asarray(subpop.compartments["S"].history_vals_list[0])
        for subpop in subpops
    ]
    init_S = np.sum(np.stack(init_S_arrays, axis=0), axis=0)  # (A, R)

    if age_group is not None:
        init_S = init_S[age_group : age_group + 1, :]
    if risk_group is not None:
        init_S = init_S[:, risk_group : risk_group + 1]

    return float(infections / init_S.sum())


def vaccine_preventable_events(
    baseline_model: MetapopModel,
    counterfactual_model: MetapopModel,
    metric_fn: Callable,
    **kwargs,
) -> float:
    """
    Vaccine-preventable events = metric(baseline) − metric(counterfactual).

    A positive result means the higher-coverage counterfactual prevented events
    relative to the baseline.

    Parameters
    ----------
    baseline_model : MetapopModel
        Model run under baseline vaccine coverage.
    counterfactual_model : MetapopModel
        Model run under a higher vaccine coverage scenario.
    metric_fn : callable
        One of ``cumulative_hospitalizations``, ``cumulative_deaths``,
        ``attack_rate``, etc.
    **kwargs
        Forwarded to ``metric_fn`` (e.g. ``subpop_name``, ``age_group``).

    Returns
    -------
    float
    """
    return metric_fn(baseline_model, **kwargs) - metric_fn(counterfactual_model, **kwargs)


def summarize_outcomes(
    values,
    credible_interval: float = 0.95,
) -> dict:
    """
    Summarize a (reps,) array of scalar outcomes across replicates.

    Parameters
    ----------
    values : array-like, shape (reps,)
    credible_interval : float
        Width of the central credible interval (default 0.95 → 2.5th–97.5th
        percentiles).

    Returns
    -------
    dict with keys: ``mean``, ``median``, ``lower_ci``, ``upper_ci``
    """
    values = np.asarray(values, dtype=float)
    half = (1.0 - credible_interval) / 2.0
    return {
        "mean":     float(np.mean(values)),
        "median":   float(np.median(values)),
        "lower_ci": float(np.percentile(values, 100 * half)),
        "upper_ci": float(np.percentile(values, 100 * (1.0 - half))),
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_compartment_history(
    metapop_model: MetapopModel,
    compartment_names=("S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"),
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
    title: str = None,
    linestyle: str = "-",
    label_suffix: str = "",
) -> matplotlib.axes.Axes:
    """
    Time series of selected compartments, aggregated or stratified by age/risk.

    Replaces the unsegmented ``plot_metapop_basic_compartment_history``.

    Parameters
    ----------
    metapop_model : MetapopModel
    compartment_names : sequence of str
        Compartments to include.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; created if not provided.
    savefig_filename : str or None
    subpop_name : str or None
        Restrict to one subpopulation; None combines all.
    age_group : int or None
        When an integer, restrict to that specific age group.
    risk_group : int or None
        Filter to a single risk group (summed otherwise).
    title : str or None
    linestyle : str
        Matplotlib linestyle (e.g. ``"-"``, ``"--"``, ``":"``).
    label_suffix : str
        Appended to each legend entry (e.g. `` "[beta=0.05]"``).
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    for comp_name in compartment_names:
        arrays = [
            np.asarray(sp.compartments[comp_name].history_vals_list)
            for sp in subpops
        ]
        total = np.sum(np.stack(arrays, axis=0), axis=0)  # (T, A, R)

        if age_group is not None:
            if risk_group is not None:
                vals = total[:, age_group, risk_group]
            else:
                vals = total[:, age_group, :].sum(axis=1)
        else:
            if risk_group is not None:
                vals = total[:, :, risk_group].sum(axis=1)
            else:
                vals = total.sum(axis=(1, 2))
        ax.plot(vals, label=f"{comp_name}{label_suffix}", alpha=0.7, linestyle=linestyle)

    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title or "Compartment histories")
    ax.legend(fontsize=7)

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax


def plot_epi_metrics(
    metapop_model: MetapopModel,
    metric_names=("M", "MV"),
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    title: str = None,
    linestyle: str = "-",
    label_suffix: str = "",
) -> matplotlib.axes.Axes:
    """
    M and MV over time, restricted to ``age_group`` when provided.

    Parameters
    ----------
    metapop_model : MetapopModel
    metric_names : sequence of str
        Epi metrics to plot; subset of ``("M", "MV")``.
    age_group : int or None
        When not None, restrict to that specific age group.
    linestyle : str
        Matplotlib linestyle (e.g. ``"-"``, ``"--"``, ``":"``).
    label_suffix : str
        Appended to each legend entry.
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if subpop_name is not None:
        subpops = [metapop_model.subpop_models[subpop_name]]
    else:
        subpops = list(metapop_model.subpop_models.values())

    for metric_name in metric_names:
        arrays = [
            np.asarray(sp.epi_metrics[metric_name].history_vals_list)
            for sp in subpops
        ]
        # Average across subpops (epi metrics are fractions, not counts)
        total = np.mean(np.stack(arrays, axis=0), axis=0)  # (T, A, R)

        if age_group is not None:
            vals = total[:, age_group, :].mean(axis=1)
        else:
            vals = total.mean(axis=(1, 2))
        ax.plot(vals, label=f"{metric_name}{label_suffix}", alpha=0.7, linestyle=linestyle)

    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Immunity level")
    ax.set_title(title or "Epi metrics (M, MV)")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax


def plot_daily_new_infections(
    metapop_model: MetapopModel,
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
    label: str = None,
    color=None,
    title: str = None,
) -> matplotlib.axes.Axes:
    """
    Daily S→E flow aggregated to one curve per subpopulation (or combined).
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    vals = daily_new_infections(metapop_model, subpop_name, age_group, risk_group)

    kw = {"alpha": 0.8}
    if color is not None:
        kw["color"] = color
    ax.plot(vals, label=label or "New infections", **kw)

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily new infections")
    ax.set_title(title or "Daily new infections")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax


def plot_daily_hospital_admissions(
    models,
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    subpop_name: Optional[str] = None,
    age_group: Optional[int] = None,
    risk_group: Optional[int] = None,
    title: str = None,
) -> matplotlib.axes.Axes:
    """
    Daily ISH→HR + ISH→HD, optionally overlaid across multiple models.

    Parameters
    ----------
    models : MetapopModel  or  dict[str, MetapopModel]
        A single model, or ``{scenario_name: model}`` dict for overlay.
        For multi-replicate data, pass
        ``{scenario_name: list_of_MetapopModels}`` — in that case the median
        line and shaded 95 % interval are shown.
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if isinstance(models, dict):
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (scenario_name, model_or_list) in enumerate(models.items()):
            color = prop_cycle[i % len(prop_cycle)]
            if isinstance(model_or_list, list):
                # Multi-replicate: median + 95 % CI ribbon
                all_vals = np.stack(
                    [daily_hospital_admissions(m, subpop_name, age_group, risk_group)
                     for m in model_or_list],
                    axis=0,
                )  # (reps, days)
                median = np.median(all_vals, axis=0)
                lo = np.percentile(all_vals, 2.5, axis=0)
                hi = np.percentile(all_vals, 97.5, axis=0)
                days = np.arange(median.shape[0])
                ax.plot(days, median, label=scenario_name, color=color, alpha=0.9)
                ax.fill_between(days, lo, hi, color=color, alpha=0.2)
            else:
                vals = daily_hospital_admissions(
                    model_or_list, subpop_name, age_group, risk_group
                )
                ax.plot(vals, label=scenario_name, color=color, alpha=0.8)
    else:
        vals = daily_hospital_admissions(models, subpop_name, age_group, risk_group)
        ax.plot(vals, alpha=0.8)

    ax.set_xlabel("Day")
    ax.set_ylabel("Daily hospital admissions")
    ax.set_title(title or "Daily hospital admissions")
    ax.legend()

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax


def plot_attack_rate_by_age(
    metapop_model: MetapopModel,
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    subpop_name: Optional[str] = None,
    title: str = None,
) -> matplotlib.axes.Axes:
    """
    Bar chart of attack rate per age group.
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    num_age_groups = _first_subpop(metapop_model).params.num_age_groups
    rates = [
        attack_rate(metapop_model, subpop_name, age_group=a)
        for a in range(num_age_groups)
    ]

    ax.bar(range(num_age_groups), rates, alpha=0.8)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Attack rate")
    ax.set_xticks(range(num_age_groups))
    ax.set_xticklabels([f"Age {a}" for a in range(num_age_groups)])
    ax.set_title(title or "Attack rate by age group")

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax


def plot_scenario_comparison(
    models_dict: dict,
    metric_fn: Callable,
    ax: matplotlib.axes.Axes = None,
    savefig_filename: str = None,
    metric_name: str = None,
    title: str = None,
    **metric_kwargs,
) -> matplotlib.axes.Axes:
    """
    Bar or box plot comparing a scalar metric across scenarios.

    Parameters
    ----------
    models_dict : dict
        ``{scenario_name: model}`` for single-run bar chart, or
        ``{scenario_name: [model_rep1, model_rep2, ...]}`` for multi-replicate
        box plots.
    metric_fn : callable
        Scalar metric function (e.g. ``cumulative_hospitalizations``).
    metric_name : str or None
        Label for the y-axis; defaults to ``metric_fn.__name__``.
    **metric_kwargs
        Forwarded to ``metric_fn``.
    """
    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    names = list(models_dict.keys())
    values = []
    multi_rep = False

    for model_or_list in models_dict.values():
        if isinstance(model_or_list, list):
            multi_rep = True
            values.append([metric_fn(m, **metric_kwargs) for m in model_or_list])
        else:
            values.append(metric_fn(model_or_list, **metric_kwargs))

    if multi_rep:
        ax.boxplot(values, labels=names)
    else:
        ax.bar(names, values, alpha=0.8)

    ax.set_ylabel(metric_name or metric_fn.__name__)
    ax.set_title(title or "Scenario comparison")

    if savefig_filename:
        plt.savefig(savefig_filename, dpi=300, bbox_inches="tight")
    if not ax_provided:
        plt.tight_layout()
        plt.show()

    return ax
