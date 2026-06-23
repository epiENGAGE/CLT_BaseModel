"""
torch_generic.py — Config-driven torch simulation loop.

Ports flu_core/flu_torch_det_components.py replacing hard-coded field access
with dict-lookup keyed by names from ModelConfig.

Functions
---------
build_state_dict                         — stack compartments/metrics → (L, A, R) tensors
build_params_dict                        — standardize params → tensors
build_schedules_dict                     — pre-compute per-day schedule tensor lists
build_generic_torch_inputs               — combine all inputs (entry point)
update_state_dict_with_schedules         — inject per-day schedule values into state_dict
check_and_apply_MV_reset                 — reset vaccine immunity on anniversary date
generic_advance_timestep                 — one dt-sized step (differentiable)
generic_torch_simulate_full_history      — full compartment+tvar history
generic_torch_simulate_calibration_target — calibration target only
"""

from __future__ import annotations

import copy
import datetime
import numpy as np
import torch
from typing import Tuple

from .config_parser import ModelConfig
from .travel_functions import GenericPrecomputedTensors, compute_total_mixing_exposure
from .rate_templates import RATE_TEMPLATE_REGISTRY


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def torch_approx_binom_probability_from_rate(
    rate: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    Differentiable rate-to-probability conversion.
    Mirrors flu_torch_det_components.torch_approx_binom_probability_from_rate.
    """
    return 1 - torch.exp(-rate * dt)


# ---------------------------------------------------------------------------
# Tensor-building helpers (tasks 4.1)
# ---------------------------------------------------------------------------

def build_state_dict(metapop_model, model_config: ModelConfig) -> dict:
    """
    Stack current compartment and epi-metric values from all subpops into
    (L, A, R) float32 tensors.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: compartment names and epi-metric names.
        Values: (L, A, R) tensors.
    """
    subpop_models = metapop_model._subpop_models_ordered
    state_dict = {}

    for comp_name in model_config.compartments:
        vals = [m.compartments[comp_name].current_val for m in subpop_models.values()]
        state_dict[comp_name] = torch.tensor(
            np.asarray(vals, dtype=np.float64), dtype=torch.float64
        )

    for mc in model_config.epi_metrics:
        vals = [m.epi_metrics[mc.name].current_val for m in subpop_models.values()]
        state_dict[mc.name] = torch.tensor(
            np.asarray(vals, dtype=np.float64), dtype=torch.float64
        )

    return state_dict


def _standardize_param_tensor(
    val, L: int, A: int, R: int, requires_grad: bool
) -> torch.Tensor:
    """
    Convert a raw param value (scalar, list, nested list) to a standardized tensor.

    Standardization rules (mirrors FluTravelParamsTensors.standardize_shapes):
        scalar      → scalar tensor (shape [])
        (A, R)      → expanded to (L, A, R)
        (A, A)      → expanded to (L, A, A)    (contact matrices)
        other       → converted as-is
    """
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        t = torch.tensor(float(arr), dtype=torch.float64)
    elif arr.shape == (A, R):
        t = torch.tensor(arr).view(1, A, R).expand(L, A, R).contiguous()
    elif arr.shape == (A, A):
        t = torch.tensor(arr).view(1, A, A).expand(L, A, A).contiguous()
    elif arr.ndim == 1 and arr.size == 1:
        t = torch.tensor(float(arr[0]), dtype=torch.float64)
    else:
        t = torch.tensor(arr, dtype=torch.float64)

    if requires_grad and t.is_floating_point():# and t.ndim == 0:
        t = t.requires_grad_(True)
    return t


def build_params_dict(
    metapop_model,
    model_config: ModelConfig,
    requires_grad: bool = True,
) -> dict:
    """
    Build a standardized params dict from the metapop model.

    Includes:
        - All model_config.params as standardized tensors
        - "travel_proportions" : (L, L) float32 tensor from mixing_params
        - "total_pop_age_risk" : (L, A, R) float32 tensor
        - "num_age_groups", "num_risk_groups" : scalar int tensors

    Parameters
    ----------
    requires_grad : bool
        If True, turns on gradient tracking for float params
        (enabled for more than just scalars, which are the typical 
        calibration targets, such as beta_baseline).
    """
    subpop_models = list(metapop_model._subpop_models_ordered.values())
    first = subpop_models[0]
    L = len(subpop_models)
    A = first.params.num_age_groups
    R = first.params.num_risk_groups

    params_dict: dict = {}
    for name, val in model_config.params.items():
        params_dict[name] = _standardize_param_tensor(val, L, A, R, requires_grad)

    # travel_proportions (L, L) — not trainable
    tp = np.asarray(metapop_model.mixing_params.travel_proportions, dtype=np.float64)
    params_dict["travel_proportions"] = torch.tensor(tp)

    # total_pop_age_risk (L, A, R) — not trainable
    total_pop = np.asarray(
        [m.params.total_pop_age_risk for m in subpop_models], dtype=np.float64
    )
    params_dict["total_pop_age_risk"] = torch.tensor(total_pop)

    return params_dict


def _get_schedule_values_for_dates(
    schedule, params, start_date: datetime.date, num_days: int
) -> list:
    """
    Compute schedule values for num_days days starting from start_date.
    Temporarily calls schedule.update_current_val for each date.

    Returns list of num_days deep-copied values.
    """
    vals = []
    for i in range(num_days):
        date = start_date + datetime.timedelta(days=i)
        schedule.update_current_val(params, date)
        vals.append(copy.deepcopy(schedule.current_val))
    return vals


def build_schedules_dict(
    metapop_model,
    model_config: ModelConfig,
    num_days: int,
) -> dict:
    """
    Pre-compute per-day schedule values for all schedules in model_config.

    For each schedule, iterates over dates from start_real_date up to
    num_days days, collecting per-subpop values and stacking them into
    (L, *shape) tensors.

    Returns
    -------
    dict[str, list[torch.Tensor]]
        Keys: schedule names.
        Values: list of num_days tensors.
            - scalar schedule  → (L, A, R) after broadcasting
            - contact matrix   → (L, A, A)
            - (A, R) schedule  → (L, A, R)
    """
    subpop_models = metapop_model._subpop_models_ordered
    first = list(subpop_models.values())[0]
    L = len(subpop_models)
    A = first.params.num_age_groups
    R = first.params.num_risk_groups

    start_date = first.start_real_date

    schedules_dict: dict = {}

    for sc_cfg in model_config.schedules:
        sname = sc_cfg.name

        # Collect per-subpop, per-day values: shape [L][num_days]
        per_subpop_series = []
        for model in subpop_models.values():
            schedule = model.schedules[sname]
            series = _get_schedule_values_for_dates(
                schedule, model.params, start_date, num_days
            )
            per_subpop_series.append(series)

        # Transpose to [num_days][L] and build tensor list
        day_tensors = []
        for day_i in range(num_days):
            day_vals = [per_subpop_series[l][day_i] for l in range(L)]
            arr = np.asarray(day_vals, dtype=np.float64)  # (L, *val_shape)

            # Expand to (L, A, R) or (L, A, A) depending on shape
            if arr.ndim == 1:
                # scalar per subpop → broadcast to (L, A, R)
                arr = np.broadcast_to(arr.reshape(L, 1, 1), (L, A, R)).copy()
            elif arr.ndim == 3 and arr.shape == (L, A, A):
                pass  # contact matrix, keep as (L, A, A)
            elif arr.ndim == 3 and arr.shape == (L, A, R):
                pass
            else:
                # Attempt generic broadcast to (L, A, R)
                try:
                    arr = np.broadcast_to(
                        arr.reshape((L,) + arr.shape[1:]), (L, A, R)
                    ).copy()
                except ValueError:
                    pass  # leave as-is; caller must handle

            day_tensors.append(torch.tensor(arr))

        schedules_dict[sname] = day_tensors

    return schedules_dict


def build_generic_torch_inputs(
    metapop_model,
    model_config: ModelConfig,
    num_days: int,
    requires_grad: bool = True,
) -> dict:
    """
    Prepare all inputs required for the generic torch simulation loop.

    Mirrors FluMetapopModel.get_flu_torch_inputs() but produces generic
    dict-based containers.

    Parameters
    ----------
    metapop_model : ConfigDrivenMetapopModel
        A fully initialised (but not yet simulated) metapop model.
    model_config : ModelConfig
        Parsed config from parse_model_config().
    num_days : int
        Number of simulation days; used to pre-compute schedule lists.
    requires_grad : bool
        Passed to build_params_dict; enables gradient tracking on scalar params.

    Returns
    -------
    dict with keys:
        "state_dict"    : dict[str, Tensor]
        "params_dict"   : dict[str, Tensor]
        "schedules_dict": dict[str, list[Tensor]]
        "precomputed"   : GenericPrecomputedTensors
    """
    return {
        "state_dict":     build_state_dict(metapop_model, model_config),
        "params_dict":    build_params_dict(metapop_model, model_config, requires_grad),
        "schedules_dict": build_schedules_dict(metapop_model, model_config, num_days),
        "precomputed":    copy.deepcopy(metapop_model.precomputed),
    }


# ---------------------------------------------------------------------------
# Per-day state preparation (task 4.1)
# ---------------------------------------------------------------------------

def update_state_dict_with_schedules(
    state_dict: dict,
    schedules_dict: dict,
    model_config: ModelConfig,
    day_counter: int,
) -> dict:
    """
    Return a new state_dict with schedule values updated for day_counter.

    Mirrors update_state_with_schedules in flu_torch_det_components.py.
    Does NOT mutate state_dict; returns a shallow copy with updated keys.
    """
    new_state = dict(state_dict)
    for sc_cfg in model_config.schedules:
        new_state[sc_cfg.name] = schedules_dict[sc_cfg.name][day_counter]
    return new_state


def check_and_apply_MV_reset(
    state_dict: dict,
    model_config: ModelConfig,
    day_counter: int,
    start_real_date: datetime.date,
) -> dict:
    """
    Reset vaccine-induced immunity metric to zero on its reset date.

    Mirrors check_and_apply_MV_reset in flu_torch_det_components.py.
    Returns possibly-updated state_dict (new dict if reset applied).
    """
    for mc in model_config.epi_metrics:
        if mc.metric_template != "vaccine_induced_immunity":
            continue
        reset_param = mc.update_config.get("vax_immunity_reset_date_mm_dd_param")
        if reset_param is None:
            continue
        reset_date_str = model_config.params.get(reset_param)
        if reset_date_str is None:
            continue

        current_date = start_real_date + datetime.timedelta(days=day_counter)
        month, day = reset_date_str.split("_")
        if current_date.month == int(month) and current_date.day == int(day):
            new_state = dict(state_dict)
            new_state[mc.name] = torch.zeros_like(state_dict[mc.name])
            print(f"VaxInducedImmunityGeneric '{mc.name}' reset to 0 on {current_date}")
            return new_state

    return state_dict


# ---------------------------------------------------------------------------
# One-timestep advance (tasks 4.2, 4.3)
# ---------------------------------------------------------------------------

def _build_torch_rate_config(
    tc, precomputed: GenericPrecomputedTensors
) -> dict:
    """
    Return rate_config for torch_rate, injecting _precomputed for travel
    transitions so ForceOfInfectionTravelRate can call compute_total_mixing_exposure.
    """
    if tc.rate_template == "force_of_infection_travel":
        rc = dict(tc.rate_config)
        rc["_precomputed"] = precomputed
        return rc
    return tc.rate_config


def _find_vax_metric_name(model_config: ModelConfig) -> str | None:
    """Return the name of the first vaccine-induced immunity metric, or None."""
    for mc in model_config.epi_metrics:
        if mc.metric_template == "vaccine_induced_immunity":
            return mc.name
    return None


def generic_advance_timestep(
    state_dict: dict,
    params_dict: dict,
    model_config: ModelConfig,
    rate_registry: dict,
    precomputed: GenericPrecomputedTensors,
    dt: float,
    save_calibration_targets: bool = False,
    save_tvar_history: bool = False,
    calibration_transition_names: list | None = None,
    timestep_idx: int = 0,
) -> Tuple[dict, dict, dict]:
    """
    Advance the model one dt-sized timestep.

    Mirrors advance_timestep in flu_torch_det_components.py but is driven by
    ModelConfig instead of hard-coded compartment names.

    Transition groups use the deterministic multinomial pattern (same as the
    OOP binom_deterministic_no_round transition type).

    Parameters
    ----------
    state_dict : dict[str, Tensor]
        Current compartment values, epi-metric values, and schedule values.
    params_dict : dict[str, Tensor]
        Model parameters.
    model_config : ModelConfig
        Validated config.
    rate_registry : dict[str, RateTemplate]
        Template registry (defaults to RATE_TEMPLATE_REGISTRY).
    precomputed : GenericPrecomputedTensors
        Precomputed travel quantities.
    dt : float
        Timestep size (1 / timesteps_per_day).
    save_calibration_targets : bool
        If True, populate calibration_targets in the returned dict.
    save_tvar_history : bool
        If True, populate tvar_history with all transition amounts.
    calibration_transition_names : list[str] | None
        Transition names to include in calibration_targets when
        save_calibration_targets is True.
    timestep_idx : int
        Index of this timestep within the current simulation day (0-based).
        Used by 'scheduled_exact' transitions, which apply their full
        scheduled count only on the first timestep of each day (mirroring
        ScheduledTransferVariable in generic_model.py) and 0 otherwise.

    Returns
    -------
    (new_state_dict, calibration_targets, tvar_history)
    """
    is_first_timestep_of_day = timestep_idx == 0

    # Names that belong to a jointly-distributed group
    jointly_grouped: set = {
        name
        for gc in model_config.transition_groups
        for name in gc.members
    }

    # Quick lookup: transition name → TransitionConfig
    tc_by_name = {tc.name: tc for tc in model_config.transitions}

    transition_amounts: dict = {}  # name → (L, A, R) tensor

    # -------------------------------------------------------------------
    # 1. Jointly-distributed transition groups
    #    Mirrors the deterministic-multinomial implementation in
    #    advance_timestep (flu_torch_det_components.py:471–503).
    # -------------------------------------------------------------------
    for gc in model_config.transition_groups:
        rates = {}
        for tname in gc.members:
            tc = tc_by_name[tname]
            template = rate_registry[tc.rate_template]
            rc = _build_torch_rate_config(tc, precomputed)
            rates[tname] = template.torch_rate(state_dict, params_dict, rc)

        # All members share the same origin
        origin_name = tc_by_name[gc.members[0]].origin
        origin = state_dict[origin_name]

        rate_list = list(rates.values())
        total_rate: torch.Tensor = rate_list[0] + sum(rate_list[1:], torch.zeros_like(rate_list[0]))
        total_amount = origin * torch_approx_binom_probability_from_rate(total_rate, dt)

        for tname, rate in rates.items():
            transition_amounts[tname] = total_amount * (rate / total_rate)

    # -------------------------------------------------------------------
    # 2. Non-grouped transitions
    # -------------------------------------------------------------------
    for tc in model_config.transitions:
        if tc.name in jointly_grouped:
            continue
        if tc.rate_template == "scheduled_exact":
            schedule_name = tc.rate_config["schedule"]
            # schedule value is a proportion of the origin population
            # vaccinated that day, not an absolute count (matches the
            # existing vaccine_schedule input format) -- convert before
            # clamping to the available population.
            origin = state_dict[tc.origin]
            scheduled_count = state_dict[schedule_name] * origin
            transition_amounts[tc.name] = (
                torch.minimum(scheduled_count, origin)
                if is_first_timestep_of_day
                else torch.zeros_like(origin)
            )
            continue
        template = rate_registry[tc.rate_template]
        rc = _build_torch_rate_config(tc, precomputed)
        rate = template.torch_rate(state_dict, params_dict, rc)
        origin = state_dict[tc.origin]
        transition_amounts[tc.name] = origin * torch_approx_binom_probability_from_rate(
            rate, dt
        )

    # -------------------------------------------------------------------
    # 3. Compartment updates with softplus non-negativity
    # -------------------------------------------------------------------
    inflows: dict = {name: [] for name in model_config.compartments}
    outflows: dict = {name: [] for name in model_config.compartments}
    for tc in model_config.transitions:
        outflows[tc.origin].append(transition_amounts[tc.name])
        inflows[tc.destination].append(transition_amounts[tc.name])

    new_state = dict(state_dict)
    for comp_name in model_config.compartments:
        val = state_dict[comp_name]
        in_sum = (
            sum(inflows[comp_name])
            if inflows[comp_name]
            else torch.zeros_like(val)
        )
        out_sum = (
            sum(outflows[comp_name])
            if outflows[comp_name]
            else torch.zeros_like(val)
        )
        new_state[comp_name] = torch.nn.functional.softplus(val + in_sum - out_sum)

    # -------------------------------------------------------------------
    # 4. Epi-metric updates (task 4.3)
    #    Mirrors compute_M_change / compute_MV_change in
    #    flu_torch_det_components.py:328–360.
    # -------------------------------------------------------------------
    vax_metric_name = _find_vax_metric_name(model_config)

    for mc in model_config.epi_metrics:
        if mc.metric_template == "infection_induced_immunity":
            r_to_s_name = mc.update_config["r_to_s_transition"]
            R_to_S = transition_amounts[r_to_s_name]
            M = state_dict[mc.name]
            MV = (
                state_dict[vax_metric_name]
                if vax_metric_name
                else torch.zeros_like(M)
            )

            inf_sat = params_dict[mc.update_config["inf_induced_saturation_param"]]
            vax_sat = params_dict[mc.update_config["vax_induced_saturation_param"]]
            wane = params_dict[mc.update_config["inf_induced_immune_wane_param"]]

            total_pop = precomputed.total_pop_LAR_tensor.to(R_to_S.dtype)

            # Mirrors compute_M_change: R_to_S already includes dt,
            # so only the waning term is multiplied by dt.
            M_change = (
                (R_to_S / total_pop) * (1 - inf_sat * M - vax_sat * MV)
                - wane * M * dt
            )
            new_state[mc.name] = M + M_change

        elif mc.metric_template == "vaccine_induced_immunity":
            MV = state_dict[mc.name]
            vax_schedule = mc.update_config["daily_vaccines_schedule"]
            daily_vaccines = state_dict[vax_schedule]
            wane = params_dict[mc.update_config["vax_induced_immune_wane_param"]]
            MV_change = (daily_vaccines - wane * MV) * dt
            new_state[mc.name] = MV + MV_change

    # -------------------------------------------------------------------
    # 5. Output dicts
    # -------------------------------------------------------------------
    calibration_targets: dict = {}
    if save_calibration_targets and calibration_transition_names:
        for tname in calibration_transition_names:
            calibration_targets[tname] = transition_amounts[tname]

    tvar_history: dict = {}
    if save_tvar_history:
        for tc in model_config.transitions:
            tvar_history[tc.name] = transition_amounts[tc.name]

    return new_state, calibration_targets, tvar_history


# ---------------------------------------------------------------------------
# Full simulation loops (tasks 4.4, 4.5)
# ---------------------------------------------------------------------------

def generic_torch_simulate_full_history(
    state_dict: dict,
    params_dict: dict,
    model_config: ModelConfig,
    rate_registry: dict,
    precomputed: GenericPrecomputedTensors,
    schedules_dict: dict,
    num_days: int,
    timesteps_per_day: int,
    start_real_date: datetime.date | None = None,
) -> Tuple[dict, dict]:
    """
    Simulate the model for num_days days, saving full compartment and
    transition-variable histories.

    Mirrors torch_simulate_full_history in flu_torch_det_components.py.

    Parameters
    ----------
    start_real_date : datetime.date | None
        Used for vaccine immunity reset check.  If None, reset is skipped.

    Returns
    -------
    (state_history_dict, tvar_history_dict)
        state_history_dict : dict[str, list[Tensor]]
            One (L, A, R) tensor per day for each compartment and epi-metric.
        tvar_history_dict  : dict[str, list[Tensor]]
            One (L, A, R) tensor per day for each transition (last timestep
            of each day, matching flu reference).
    """
    from collections import defaultdict

    dt = 1.0 / float(timesteps_per_day)

    state_history: dict = defaultdict(list)
    tvar_history: dict = defaultdict(list)

    for day in range(num_days):
        state_dict = update_state_dict_with_schedules(
            state_dict, schedules_dict, model_config, day
        )
        if start_real_date is not None:
            state_dict = check_and_apply_MV_reset(
                state_dict, model_config, day, start_real_date
            )

        for timestep in range(timesteps_per_day):
            save_last = (timestep == timesteps_per_day - 1)
            state_dict, _, step_tvars = generic_advance_timestep(
                state_dict,
                params_dict,
                model_config,
                rate_registry,
                precomputed,
                dt,
                save_tvar_history=save_last,
                timestep_idx=timestep,
            )
            if save_last:
                for tname, val in step_tvars.items():
                    tvar_history[tname].append(val)

        # Save compartments and metrics at end of day
        for comp_name in model_config.compartments:
            state_history[comp_name].append(state_dict[comp_name].clone())
        for mc in model_config.epi_metrics:
            state_history[mc.name].append(state_dict[mc.name].clone())

    return dict(state_history), dict(tvar_history)


def generic_torch_simulate_calibration_target(
    state_dict: dict,
    params_dict: dict,
    model_config: ModelConfig,
    rate_registry: dict,
    precomputed: GenericPrecomputedTensors,
    schedules_dict: dict,
    num_days: int,
    timesteps_per_day: int,
    calibration_transition_names: list,
    calibration_compartment_names: list | None = None,
    start_real_date: datetime.date | None = None,
) -> dict:
    """
    Simulate for num_days, returning only the sum of named calibration-target
    transitions and/or end-of-day compartment values for each day.

    Mirrors torch_simulate_hospital_admits in flu_torch_det_components.py
    but accepts any list of transition names instead of hardcoding ISH_to_HR
    and ISH_to_HD.

    Parameters
    ----------
    calibration_transition_names : list[str]
        Transition variable names to accumulate (summed across timesteps).
        May be empty when compartment targets are used alone.
    calibration_compartment_names : list[str] | None
        Compartment names whose end-of-day values are added to the output.
        Gradients flow through these values in the same way as transitions.

    Returns
    -------
    dict[str, torch.Tensor]
        One tensor of shape (num_days, L, A, R) per requested variable.
        Transitions are accumulated over all intra-day timesteps; compartments
        are end-of-day snapshots.
    """
    dt = 1.0 / float(timesteps_per_day)
    _comp_names = calibration_compartment_names or []
    _all_names = list(calibration_transition_names) + _comp_names

    # Per-variable accumulator: {name: list of (L, A, R) tensors, one per day}
    daily_per_var: dict = {name: [] for name in _all_names}

    for day in range(num_days):
        state_dict = update_state_dict_with_schedules(
            state_dict, schedules_dict, model_config, day
        )
        if start_real_date is not None:
            state_dict = check_and_apply_MV_reset(
                state_dict, model_config, day, start_real_date
            )

        day_tv_accum: dict = {t: None for t in calibration_transition_names}
        for timestep in range(timesteps_per_day):
            state_dict, cal_targets, _ = generic_advance_timestep(
                state_dict,
                params_dict,
                model_config,
                rate_registry,
                precomputed,
                dt,
                save_calibration_targets=bool(calibration_transition_names),
                calibration_transition_names=calibration_transition_names,
                timestep_idx=timestep,
            )
            for tname in calibration_transition_names:
                val = cal_targets[tname]
                day_tv_accum[tname] = (
                    val if day_tv_accum[tname] is None else day_tv_accum[tname] + val
                )

        for tname in calibration_transition_names:
            daily_per_var[tname].append(day_tv_accum[tname].clone())

        for cname in _comp_names:
            daily_per_var[cname].append(state_dict[cname].clone())

    return {name: torch.stack(daily_per_var[name]) for name in _all_names}
