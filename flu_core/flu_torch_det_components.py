###################################################################################
######################## MetroFluSim: pytorch implementation ######################
###################################################################################

# Dimensions
#   L (int):
#       number of locations/subpopulations
#   A (int):
#       number of age groups
#   R (int):
#       number of risk groups

import torch
import numpy as np
import pandas as pd
import clt_toolkit as clt

import datetime

from typing import Tuple

from collections import defaultdict
from dataclasses import dataclass, fields, field, replace

from .flu_data_structures import FluFullMetapopStateTensors, \
    FluFullMetapopParamsTensors, FluPrecomputedTensors, \
    FluFullMetapopScheduleTensors, BASE_COMPARTMENTS
from .flu_travel_functions import compute_total_mixing_exposure

base_path = clt.utils.PROJECT_ROOT / "flu_instances" / "texas_input_files"


def torch_approx_binom_probability_from_rate(rate, dt):
    """
    Torch-compatible implementation of converting a
    rate into a probability. See analogous numpy implementation
    `base_components/approx_binom_probability_from_rate()` docstring
    for details.
    """

    return 1 - torch.exp(-rate * dt)


def create_dict_of_tensors(d: dict,
                           requires_grad: bool = True) -> dict:
    """
    Converts dictionary entries to `tensor` (of type `torch.float32`)
    and if `requires_grad` is `True`, turns on gradient tracking for
    each entry -- returns new dictionary.
    """

    def to_tensor(k, v):
        if v is None:
            return None
        else:
            return torch.tensor(v, dtype=torch.float32, requires_grad=requires_grad)

    return {k: to_tensor(k, v) for k, v in d.items()}


def compute_beta_adjusted(state: FluFullMetapopStateTensors,
                          params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Computes beta-adjusted humidity.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    absolute_humidity = state.absolute_humidity
    beta_adjusted = params.beta_baseline * (1 + params.humidity_impact * np.exp(-180 * absolute_humidity))

    return beta_adjusted


def compute_flu_contact_matrix(params: FluFullMetapopParamsTensors,
                               schedules: FluFullMetapopScheduleTensors,
                               day_counter: int) -> torch.Tensor:
    """
    Computes flu model contact matrix in tensor format -- makes
    adjustments based on whether day is school day or work day.

    Returns:
        (torch.Tensor of size (L, A, A))
    """

    # Here, using schedules.is_school_day[day_counter][:,:,0] and similarly for
    #   is_work_day because each contact matrix (as a metapop tensor) is L x A x A --
    #   we don't use risk -- assume here that we do not have a different school/work-day
    #   schedule based on risk, so just grab the first risk group
    # But then we have to take (1 - schedules.is_school_day[day_counter][:, :, 0]), which is
    #   L x A, and then make it L x A x 1 (unsqueeze the last dimension) to make the
    #   broadcasting work (because this gets element-wise multiplied by params.school_contact_matrix)
    flu_contact_matrix = \
        params.total_contact_matrix - \
        params.school_contact_matrix * (1 - schedules.is_school_day[day_counter][:, :, 0]).unsqueeze(dim=2) - \
        params.work_contact_matrix * (1 - schedules.is_work_day[day_counter][:, :, 0]).unsqueeze(dim=2)

    return flu_contact_matrix


def compute_S_to_E_rate(state: FluFullMetapopStateTensors,
                        params: FluFullMetapopParamsTensors,
                        precomputed: FluPrecomputedTensors,
                        total_mixing_exposure: torch.Tensor = None) -> torch.Tensor:
    """
    Returns the "S" to "E" rate -- the "S_V" to "E_V" rate is this
    rate multiplied by `1 - vax_induced_inf_risk_reduce` (see
    `advance_timestep`).

    Returns:
        (torch.Tensor of size (L, A, R))

    If total_mixing_exposure is provided, use it directly (for daily-update mode
    matching the numpy metapop model). Otherwise compute it from current state.
    """

    if total_mixing_exposure is None:
        total_mixing_exposure = compute_total_mixing_exposure(state, params, precomputed)

    if total_mixing_exposure.size() != torch.Size([precomputed.L,
                                                   precomputed.A,
                                                   precomputed.R]):
        raise Exception("force_of_infection must be L x A x R corresponding \n"
                        "to number of locations (subpopulations), age groups, \n"
                        "and risk groups.")

    beta_adjusted = compute_beta_adjusted(state, params)

    inf_induced_inf_risk_reduce = params.inf_induced_inf_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_inf_risk_reduce / (1 - inf_induced_inf_risk_reduce)

    immune_force = 1 + inf_induced_proportional_risk_reduce * state.M

    vax_immunity_factor = 1 - state.MV * params.vax_induced_inf_risk_reduce_initial

    rate = beta_adjusted * total_mixing_exposure * vax_immunity_factor / immune_force

    return rate


def compute_E_to_IP_rate(params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    return params.E_to_I_rate * (1 - params.E_to_IA_prop)


def compute_E_to_IA_rate(params: FluFullMetapopParamsTensors) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R))
    """

    return params.E_to_I_rate * params.E_to_IA_prop


def compute_IP_to_ISR_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors,
                           hosp_multiplier=1.0) -> torch.Tensor:
    """
    `hosp_multiplier` multiplies the probability of hospitalization --
    1 for "IP", the conditional multiplier from
    `torch_compute_vax_conditional_multipliers` for "IP_V".

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

    vax_immunity_factor = 1 - state.MV * params.vax_induced_hosp_risk_reduce_initial

    prob_hosp = (params.IP_to_ISH_prop / immunity_force) * vax_immunity_factor * hosp_multiplier

    rate = params.IP_to_IS_rate * (1 - prob_hosp)

    return rate


def compute_IP_to_ISH_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors,
                           hosp_multiplier=1.0) -> torch.Tensor:
    """
    See `compute_IP_to_ISR_rate` for `hosp_multiplier`.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
    inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

    immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

    vax_immunity_factor = 1 - state.MV * params.vax_induced_hosp_risk_reduce_initial

    prob_hosp = (params.IP_to_ISH_prop / immunity_force) * vax_immunity_factor * hosp_multiplier

    rate = params.IP_to_IS_rate * prob_hosp

    return rate


def compute_ISH_to_HR_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors,
                           death_multiplier=1.0) -> torch.Tensor:
    """
    `death_multiplier` multiplies the probability of death --
    1 for "ISH", the conditional multiplier from
    `torch_compute_vax_conditional_multipliers` for "ISH_V".

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

    vax_immunity_factor = 1 - state.MV * params.vax_induced_death_risk_reduce_initial

    prob_death = (params.ISH_to_HD_prop / immunity_force) * vax_immunity_factor * death_multiplier

    rate = (1 - prob_death) * params.ISH_to_H_rate

    return rate


def compute_ISH_to_HD_rate(state: FluFullMetapopStateTensors,
                           params: FluFullMetapopParamsTensors,
                           death_multiplier=1.0) -> torch.Tensor:
    """
    See `compute_ISH_to_HR_rate` for `death_multiplier`.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce

    inf_induced_proportional_risk_reduce = \
        inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

    immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

    vax_immunity_factor = 1 - state.MV * params.vax_induced_death_risk_reduce_initial

    prob_death = (params.ISH_to_HD_prop / immunity_force) * vax_immunity_factor * death_multiplier

    rate = prob_death * params.ISH_to_H_rate

    return rate


# Infection-induced (M) and vaccine-induced (MV) immunity are switched
#   off in the vaccinated-track model -- they always stay at zero, like
#   their numpy counterparts `InfInducedImmunity` and `VaxInducedImmunity`.
#   Vaccination is modeled by moving people from "S" to "S_V" instead.


def compute_M_change(state: FluFullMetapopStateTensors, params: FluFullMetapopParamsTensors,
                     precomputed: FluPrecomputedTensors,
                     dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R)) of zeros -- M is switched off.
    """

    return torch.zeros_like(state.M)


def compute_MV_change(state: FluFullMetapopStateTensors,
                      params: FluFullMetapopParamsTensors,
                      precomputed: FluPrecomputedTensors,
                      dt: float) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor of size (L, A, R)) of zeros -- MV is switched off.
    """

    return torch.zeros_like(state.MV)


def _vax_conditional_ratio(ve_outcome: torch.Tensor,
                           ve_prior_step: torch.Tensor) -> torch.Tensor:
    """
    Torch counterpart of `flu_components._vax_conditional_ratio`:
    `(1 - ve_outcome) / (1 - ve_prior_step)`, capped at 1.0, and 1.0
    where `ve_prior_step` is 1.
    """

    denom = 1 - ve_prior_step
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    ratio = torch.where(denom > 0, (1 - ve_outcome) / safe_denom, torch.ones_like(denom))

    return torch.clamp(ratio, min=0.0, max=1.0)


def torch_compute_vax_conditional_multipliers(params: FluFullMetapopParamsTensors) -> tuple:
    """
    Torch counterpart of `flu_components.compute_vax_conditional_multipliers`
    -- converts the overall `vax_induced_hosp_risk_reduce` and
    `vax_induced_death_risk_reduce` into the conditional multipliers
    applied at "IP_V" -> "ISH_V" and "ISH_V" -> "HD_V":

        hosp_mult  = (1 - VE_hosp)  / (1 - VE_inf)
        death_mult = (1 - VE_death) / (1 - VE_hosp)

    each capped at 1.0.

    Returns:
        (hosp_mult, death_mult), each a torch.Tensor of size (L, A, R).
    """

    hosp_mult = _vax_conditional_ratio(params.vax_induced_hosp_risk_reduce,
                                       params.vax_induced_inf_risk_reduce)
    death_mult = _vax_conditional_ratio(params.vax_induced_death_risk_reduce,
                                        params.vax_induced_hosp_risk_reduce)

    return hosp_mult, death_mult


def compute_daily_vax_expected(state: FluFullMetapopStateTensors,
                               params: FluFullMetapopParamsTensors,
                               precomputed: FluPrecomputedTensors) -> torch.Tensor:
    """
    Torch counterpart of the daily computation in
    `ScheduledVaccination.get_current_rate`: the expected number of
    people moving from "S" to "S_V" over the day, i.e. the day's
    `daily_vaccines` proportion times the dose pool -- S + S_V when
    `vax_dose_pool` is "susceptible", the whole population when it is
    "total_population". Should be computed from the state at the
    start of the day.

    Returns:
        (torch.Tensor of size (L, A, R))
    """

    if params.vax_dose_pool == "total_population":
        pool = precomputed.total_pop_LAR_tensor
    else:
        pool = state.S + state.S_V

    return state.daily_vaccines * pool


def check_and_apply_vax_track_reset(state: FluFullMetapopStateTensors,
                                    params: FluFullMetapopParamsTensors,
                                    day_counter: int) -> FluFullMetapopStateTensors:
    """
    Torch counterpart of `FluSubpopModel.check_and_apply_vax_track_reset`.

    If the current date matches `vax_immunity_reset_date_mm_dd`, returns
    a new state with everyone in "S_V" moved back to "S"; otherwise
    returns `state` unchanged. Like the numpy version, this fires on
    every matching month/day, so it repeats each year in a
    multi-season run.

    Args:
        state (FluFullMetapopStateTensors):
            current state.
        params (FluFullMetapopParamsTensors):
            holds `vax_immunity_reset_date_mm_dd` and `start_real_date`.
        day_counter (int):
            days elapsed since `start_real_date`.

    Returns:
        FluFullMetapopStateTensors
    """

    if params.vax_immunity_reset_date_mm_dd is None:
        return state

    current_date = params.start_real_date + datetime.timedelta(days=day_counter)

    # Parse reset date (format: "MM_DD")
    month, day = params.vax_immunity_reset_date_mm_dd.split('_')

    # Check if current date matches the reset date (month and day)
    if current_date.month != int(month) or current_date.day != int(day):
        return state

    print(f"Vaccinated track reset: S_V moved back to S on {current_date}")

    return replace(state, S=state.S + state.S_V, S_V=torch.zeros_like(state.S_V))


def update_state_with_schedules(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                day_counter: int) -> FluFullMetapopStateTensors:
    """
    Returns new dataclass formed by copying the current `state`
    and updating specific values according to `schedules` and
    the simulation's current `day_counter`.

    Returns:
        (FluFullMetapopStateTensors):
            New state with updated schedule-related values:
              - `flu_contact_matrix`
              - `absolute_humidity`
              - `daily_vaccines`
              - `mobility_modifier`
            All other fields remain unchanged from the input `state`.
    """

    return replace(state,
                   flu_contact_matrix=compute_flu_contact_matrix(params, schedules, day_counter),
                   absolute_humidity=schedules.absolute_humidity[day_counter],
                   daily_vaccines=schedules.daily_vaccines[day_counter],
                   mobility_modifier=schedules.mobility_modifier[day_counter])


def _vax_track_name(tvar_name: str) -> str:
    """
    Maps a base-track transition name to its vaccinated-track
    counterpart, e.g. "E_to_IP" -> "E_V_to_IP_V".
    """

    origin, destination = tvar_name.split("_to_")

    return f"{origin}_V_to_{destination}_V"


def compute_track_transitions(state: FluFullMetapopStateTensors,
                              params: FluFullMetapopParamsTensors,
                              dt: float,
                              suffix: str = "",
                              hosp_multiplier=1.0,
                              death_multiplier=1.0) -> dict:
    """
    Computes every transition within one track except "S" to "E"
    (and entry into the vaccinated track) -- `suffix` is "" for the
    base track and "_V" for the vaccinated track, whose only differences
    here are `hosp_multiplier` (see `compute_IP_to_ISR_rate`) and
    `death_multiplier` (see `compute_ISH_to_HR_rate`).

    Implements the "mean" deterministic multinomial for compartments
    with multiple outflows, to match the object-oriented version.

    Returns:
        (dict):
            base-track transition names (e.g. "E_to_IP") mapped to
            torch.Tensor of size (L, A, R).
    """

    c = {name: getattr(state, name + suffix) for name in BASE_COMPARTMENTS}

    flows = {}

    E_to_IP_rate = compute_E_to_IP_rate(params)
    E_to_IA_rate = compute_E_to_IA_rate(params)
    E_outgoing_total_rate = E_to_IP_rate + E_to_IA_rate
    E_outgoing_total = c["E"] * \
        torch_approx_binom_probability_from_rate(E_outgoing_total_rate, dt)
    flows["E_to_IA"] = E_outgoing_total * (E_to_IA_rate / E_outgoing_total_rate)
    flows["E_to_IP"] = E_outgoing_total * (E_to_IP_rate / E_outgoing_total_rate)

    flows["IA_to_R"] = c["IA"] * torch_approx_binom_probability_from_rate(params.IA_to_R_rate, dt)

    IP_to_ISR_rate = compute_IP_to_ISR_rate(state, params, hosp_multiplier)
    IP_to_ISH_rate = compute_IP_to_ISH_rate(state, params, hosp_multiplier)
    IP_outgoing_total_rate = IP_to_ISR_rate + IP_to_ISH_rate
    IP_outgoing_total = c["IP"] * \
        torch_approx_binom_probability_from_rate(IP_outgoing_total_rate, dt)
    flows["IP_to_ISR"] = IP_outgoing_total * (IP_to_ISR_rate / IP_outgoing_total_rate)
    flows["IP_to_ISH"] = IP_outgoing_total * (IP_to_ISH_rate / IP_outgoing_total_rate)

    flows["ISR_to_R"] = c["ISR"] * torch_approx_binom_probability_from_rate(params.ISR_to_R_rate, dt)

    ISH_to_HR_rate = compute_ISH_to_HR_rate(state, params, death_multiplier)
    ISH_to_HD_rate = compute_ISH_to_HD_rate(state, params, death_multiplier)
    ISH_outgoing_total_rate = ISH_to_HR_rate + ISH_to_HD_rate
    ISH_outgoing_total = c["ISH"] * \
        torch_approx_binom_probability_from_rate(ISH_outgoing_total_rate, dt)
    flows["ISH_to_HR"] = ISH_outgoing_total * (ISH_to_HR_rate / ISH_outgoing_total_rate)
    flows["ISH_to_HD"] = ISH_outgoing_total * (ISH_to_HD_rate / ISH_outgoing_total_rate)

    flows["HR_to_R"] = c["HR"] * torch_approx_binom_probability_from_rate(params.HR_to_R_rate, dt)
    flows["HD_to_D"] = c["HD"] * torch_approx_binom_probability_from_rate(params.HD_to_D_rate, dt)

    flows["R_to_S"] = c["R"] * torch_approx_binom_probability_from_rate(params.R_to_S_rate, dt)

    return flows


def compute_track_new_compartments(state: FluFullMetapopStateTensors,
                                   flows: dict,
                                   S_to_E: torch.Tensor,
                                   S_net_vaccination: torch.Tensor,
                                   suffix: str = "") -> dict:
    """
    Applies one track's transitions (from `compute_track_transitions`,
    plus "S" to "E" and the net vaccination flow into "S", which is
    negative for the base track and positive for the vaccinated track)
    to its compartments.

    Uses `softplus`, a smooth approximation to the ReLU function, to
    keep compartments nonnegative.

    Returns:
        (dict):
            compartment names (with `suffix`) mapped to their new
            torch.Tensor values of size (L, A, R).
    """

    c = {name: getattr(state, name + suffix) for name in BASE_COMPARTMENTS}
    f = flows
    softplus = torch.nn.functional.softplus

    new_vals = {
        "S": softplus(c["S"] + f["R_to_S"] - S_to_E + S_net_vaccination),
        "E": softplus(c["E"] + S_to_E - f["E_to_IP"] - f["E_to_IA"]),
        "IP": softplus(c["IP"] + f["E_to_IP"] - f["IP_to_ISR"] - f["IP_to_ISH"]),
        "ISR": softplus(c["ISR"] + f["IP_to_ISR"] - f["ISR_to_R"]),
        "ISH": softplus(c["ISH"] + f["IP_to_ISH"] - f["ISH_to_HR"] - f["ISH_to_HD"]),
        "IA": softplus(c["IA"] + f["E_to_IA"] - f["IA_to_R"]),
        "HR": softplus(c["HR"] + f["ISH_to_HR"] - f["HR_to_R"]),
        "HD": softplus(c["HD"] + f["ISH_to_HD"] - f["HD_to_D"]),
        "R": softplus(c["R"] + f["ISR_to_R"] + f["IA_to_R"] + f["HR_to_R"] - f["R_to_S"]),
        "D": softplus(c["D"] + f["HD_to_D"]),
    }

    return {name + suffix: val for name, val in new_vals.items()}


def advance_timestep(state: FluFullMetapopStateTensors,
                     params: FluFullMetapopParamsTensors,
                     precomputed: FluPrecomputedTensors,
                     dt: float,
                     save_calibration_targets: bool=False,
                     save_tvar_history: bool=False,
                     total_mixing_exposure: torch.Tensor = None,
                     daily_vax_expected: torch.Tensor = None) -> Tuple[FluFullMetapopStateTensors, dict, dict]:
    """
    Advance the simulation one timestep, with length `dt`.
    Updates state corresponding to compartments and
    epidemiological metrics after computing transition variables
    and metric changes.

    Note that in this torch "mean" deterministic implementation...
    - We compute rates in the same way as the
        `get_binom_deterministic_no_round`
        transition type in the OOP code -- see
        `TransitionVariables` class in
        `clt_toolkit / base_components` for more details.
    - We also implement a "mean" deterministic analog
        of the multinomial distribution to handle
        multiple outflows from the same compartment
    - We do not round the transition variables
    - We also use `softplus`, a smooth approximation to the
        ReLU function, to ensure that compartments are
        nonnegative (which is not guaranteed using
        the mean of a binomial/multinomial random variable
        rather than sampling from those distributions).

    Both tracks are advanced: the base track and the vaccinated track
    (compartments with a "_V" suffix). People enter the vaccinated
    track from "S" into "S_V": each timestep moves
    `daily_vax_expected * dt` people -- spreading the day's expected
    vaccinations evenly over its timesteps, like `ScheduledVaccination`
    -- capped at what "S" to "E" leaves in "S". `daily_vax_expected`
    should come from `compute_daily_vax_expected` at the start of the
    day; if it is None, it is computed from the current state, which
    is only equivalent when there is one timestep per day.

    Returns:
        (Tuple[FluFullMetapopStateTensors, dict, dict]):
            New `FluFullMetapopStateTensors` with updated state,
            `dict` of calibration targets corresponding to state
            values or transition variable values used for calibration,
            and `dict` of transition variable values to save this
            history. If `save_calibration_targets` is `False`,
            then the corresponding `dict` is empty, and similarly with
            `save_tvar_history`.
    """

    if daily_vax_expected is None:
        daily_vax_expected = compute_daily_vax_expected(state, params, precomputed)

    S_to_E_rate = compute_S_to_E_rate(state, params, precomputed,
                                      total_mixing_exposure=total_mixing_exposure)
    S_to_E = state.S * torch_approx_binom_probability_from_rate(S_to_E_rate, dt)
    S_V_to_E_V = state.S_V * torch_approx_binom_probability_from_rate(
        S_to_E_rate * (1 - params.vax_induced_inf_risk_reduce), dt)

    # Entry into the vaccinated track, capped so "S" cannot go negative
    S_to_S_V = torch.minimum(daily_vax_expected * dt,
                             torch.clamp(state.S - S_to_E, min=0.0))

    base_flows = compute_track_transitions(state, params, dt)
    vax_hosp_multiplier, vax_death_multiplier = torch_compute_vax_conditional_multipliers(params)
    vax_flows = compute_track_transitions(state, params, dt, suffix="_V",
                                          hosp_multiplier=vax_hosp_multiplier,
                                          death_multiplier=vax_death_multiplier)

    new_compartments = {
        **compute_track_new_compartments(state, base_flows, S_to_E, -S_to_S_V),
        **compute_track_new_compartments(state, vax_flows, S_V_to_E_V, S_to_S_V, suffix="_V"),
    }

    # Immunity variables are switched off -- see `compute_M_change`
    M_change = compute_M_change(state, params, precomputed, dt)
    MV_change = compute_MV_change(state, params, precomputed, dt)

    state_new = replace(state,
                        **new_compartments,
                        M=state.M + M_change,
                        MV=state.MV + MV_change)

    calibration_targets = {}
    if save_calibration_targets:
        calibration_targets["ISH_to_H"] = base_flows["ISH_to_HR"] + base_flows["ISH_to_HD"] + \
            vax_flows["ISH_to_HR"] + vax_flows["ISH_to_HD"]

    transition_variables = {}
    if save_tvar_history:
        transition_variables["S_to_E"] = S_to_E
        transition_variables.update(base_flows)
        transition_variables["S_to_S_V"] = S_to_S_V
        transition_variables["S_V_to_E_V"] = S_V_to_E_V
        transition_variables.update({_vax_track_name(name): val for name, val in vax_flows.items()})
        transition_variables["M_change"] = M_change
        transition_variables["MV_change"] = MV_change

    return state_new, calibration_targets, transition_variables


def prepare_daily_torch_state(state: FluFullMetapopStateTensors,
                              params: FluFullMetapopParamsTensors,
                              precomputed: FluPrecomputedTensors,
                              schedules: FluFullMetapopScheduleTensors,
                              day: int) -> Tuple[FluFullMetapopStateTensors, torch.Tensor, torch.Tensor]:
    """
    Once-a-day updates at the start of simulation day `day`, matching
    `FluSubpopModel.prepare_daily_state` and `FluMetapopModel`'s
    once-a-day mixing exposure: updates schedule values, applies the
    vaccinated-track reset, and computes the day's mixing exposure and
    expected vaccinations (both from the post-reset state).

    Returns:
        (Tuple[FluFullMetapopStateTensors, torch.Tensor, torch.Tensor]):
            new state, daily mixing exposure, and daily expected
            vaccinations (see `compute_daily_vax_expected`).
    """

    state = update_state_with_schedules(state, params, schedules, day)
    state = check_and_apply_vax_track_reset(state, params, day)

    daily_mixing_exposure = compute_total_mixing_exposure(state, params, precomputed)
    daily_vax_expected = compute_daily_vax_expected(state, params, precomputed)

    return state, daily_mixing_exposure, daily_vax_expected


def torch_simulate_full_history(state: FluFullMetapopStateTensors,
                                params: FluFullMetapopParamsTensors,
                                precomputed: FluPrecomputedTensors,
                                schedules: FluFullMetapopScheduleTensors,
                                num_days: int,
                                timesteps_per_day: int) -> Tuple[dict, dict]:
    """
    Simulates the flu model with a differentiable torch implementation
    that carries out `binom_deterministic_no_round` transition types --
    returns hospital admits for calibration use.

    See subroutine `advance_timestep` for additional details.

    Returns:
        (Tuple[dict, dict]):
            Returns compartment states and transition variables
            for day, location, age, risk, in tensor format.
    """

    dt = 1 / float(timesteps_per_day)

    state_history_dict = defaultdict(list)
    tvar_history_dict = defaultdict(list)

    for day in range(num_days):
        state, daily_mixing_exposure, daily_vax_expected = \
            prepare_daily_torch_state(state, params, precomputed, schedules, day)

        daily_tvar = None
        for timestep in range(timesteps_per_day):
            state, _, tvar_history = \
                advance_timestep(state, params, precomputed, dt, save_tvar_history=True,
                                 total_mixing_exposure=daily_mixing_exposure,
                                 daily_vax_expected=daily_vax_expected)
            if daily_tvar is None:
                daily_tvar = {key: val.clone() for key, val in tvar_history.items()}
            else:
                for key in tvar_history:
                    daily_tvar[key] = daily_tvar[key] + tvar_history[key]

        for key in daily_tvar:
            tvar_history_dict[key].append(daily_tvar[key])

        for field in fields(state):
            if field.name == "init_vals":
                continue
            state_history_dict[str(field.name)].append(getattr(state, field.name).clone())

    return state_history_dict, tvar_history_dict


def torch_simulate_hospital_admits(state: FluFullMetapopStateTensors,
                                     params: FluFullMetapopParamsTensors,
                                     precomputed: FluPrecomputedTensors,
                                     schedules: FluFullMetapopScheduleTensors,
                                     num_days: int,
                                     timesteps_per_day: int) -> torch.Tensor:
    """
    Analogous to `torch_simulate_full_history` but only saves and
    returns hospital admits for calibration use.

    Returns:
        (torch.Tensor of size (num_days, L, A, R)):
            Returns hospital admits (the ISH to HR and HD
            transition variable values, summed over both tracks)
            for day, location, age, risk, in tensor format.
    """

    hospital_admits_history = []

    dt = 1 / float(timesteps_per_day)

    for day in range(num_days):
        state, daily_mixing_exposure, daily_vax_expected = \
            prepare_daily_torch_state(state, params, precomputed, schedules, day)
        daily_admits = None
        for timestep in range(timesteps_per_day):
            state, calibration_targets, _ = \
                advance_timestep(state, params, precomputed, dt, save_calibration_targets=True,
                                 total_mixing_exposure=daily_mixing_exposure,
                                 daily_vax_expected=daily_vax_expected)
            if daily_admits is None:
                daily_admits = calibration_targets["ISH_to_H"].clone()
            else:
                daily_admits = daily_admits + calibration_targets["ISH_to_H"]
        hospital_admits_history.append(daily_admits)

    return torch.stack(hospital_admits_history)
