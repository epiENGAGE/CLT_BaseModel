import datetime
import copy

import numpy as np
import pandas as pd
import sciris as sc
from typing import Optional
from abc import ABC
import json
import warnings

from functools import reduce

import torch
import clt_toolkit as clt

from dataclasses import fields, asdict
from .flu_travel_functions import compute_total_mixing_exposure
from .flu_data_structures import FluSubpopState, FluSubpopParams, \
    FluTravelStateTensors, FluTravelParamsTensors, \
    FluFullMetapopStateTensors, FluFullMetapopParamsTensors, \
    FluMixingParams, FluPrecomputedTensors, FluFullMetapopScheduleTensors, \
    FluSubpopSchedules, ALL_COMPARTMENTS


class FluSubpopModelError(clt.SubpopModelError):
    """Custom exceptions for flu subpopulation simulation model errors."""
    pass


class FluMetapopModelError(clt.MetapopModelError):
    """Custom exceptions for flu metapopulation simulation model errors."""
    pass


VAX_DOSE_POOLS = ("susceptible", "total_population")


# Note: for dataclasses, Optional is used to help with static type checking
# -- it means that an attribute can either hold a value with the specified
# datatype or it can be None


class SusceptibleToExposed(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "S" to "E" compartment. The functional form is the same across
    subpopulations.

    The rate depends on the corresponding subpopulation's
    contact matrix, transmission rate beta, number
    infected (symptomatic, asymptomatic, and pre-symptomatic),
    and population-level immunity against infection,
    among other parameters.

    This is the most complicated transition variable in the
    flu model. If using metapopulation model (travel model), then
    the rate depends on the `total_mixing_exposure` attribute,
    which is a function of other subpopulations' states and
    parameters, and travel between subpopulations.

    If there is no metapopulation model, the rate
    is much simpler.

    Attributes:
        total_mixing_exposure (np.ndarray of positive floats):
            weighted infectious count (exposure) from movement
            within home location, travel to other locations,
            and visitors from other locations

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 origin: clt.Compartment,
                 destination: clt.Compartment,
                 transition_type: clt.TransitionTypes,
                 is_jointly_distributed: str = False):

        super().__init__(origin,
                         destination,
                         transition_type,
                         is_jointly_distributed)

        self.total_mixing_exposure = None

    def susceptibility_multiplier(self,
                                  params: FluSubpopParams):
        """
        Multiplier on the relative susceptibility of the origin
        compartment -- 1 for "S", overridden for "S_V" in
        `VaxSusceptibleToExposed`.
        """

        return 1.0

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        return np.asarray(self.get_unadjusted_rate(state, params) *
                          self.susceptibility_multiplier(params))

    def get_unadjusted_rate(self,
                            state: FluSubpopState,
                            params: FluSubpopParams) -> np.ndarray:
        """
        Rate for someone in "S" -- see `get_current_rate`.

        Returns:
            np.ndarray of shape (A, R)
        """

        # If `total_mixing_exposure` has not been updated,
        #   then there is no travel model -- so, simulate
        #   this subpopulation entirely independently and
        #   use the simplified transition rate that does not
        #   depend on travel dynamics

        beta_adjusted = compute_beta_adjusted(state, params)

        inf_induced_inf_risk_reduce = params.inf_induced_inf_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_inf_risk_reduce / (1 - inf_induced_inf_risk_reduce)

        immune_force = 1 + inf_induced_proportional_risk_reduce * state.M

        # Vaccine-induced protection against infection is modeled as a
        #   multiplicative reduction (rather than folded into the additive
        #   `immune_force` denominator above) -- see
        #   `compute_vax_induced_risk_reduce_initial` for how
        #   `vax_induced_inf_risk_reduce_initial` is derived.
        vax_immunity_factor = 1 - state.MV * params.vax_induced_inf_risk_reduce_initial

        if self.total_mixing_exposure is not None:

            # Note here `self.total_mixing_exposure` includes
            #   `suscept_by_age` -- see `compute_total_mixing_exposure_prop`
            #   in `flu_travel_functions`

            # Need to convert tensor into array because combining np.ndarrays and
            #   tensors doesn't work, and everything else is an array
            # Note: `self.total_mixing_exposure` (a Tensor) must be the left
            #   operand of the first multiplication -- np.ndarray * Tensor
            #   raises TypeError, but Tensor * np.ndarray works fine.
            return np.asarray(
                (beta_adjusted * self.total_mixing_exposure / immune_force) * vax_immunity_factor)

        else:
            wtd_presymp_asymp_by_age = compute_wtd_presymp_asymp_by_age(state, params)

            # Super confusing syntax... but this is the pain of having A x R,
            #   but having the contact matrix (contact patterns) be for
            #   ONLY age groups
            # Symptomatic people on both tracks are infectious
            wtd_infectious_prop = np.divide(
                np.sum(compute_symp_infectious(state), axis=1, keepdims=True) +\
                    wtd_presymp_asymp_by_age,
                compute_pop_by_age(params))

            raw_total_exposure = np.matmul(state.flu_contact_matrix, wtd_infectious_prop)

            # The total rate is only age-dependent -- it's the same rate across age groups
            return params.relative_suscept * (beta_adjusted * vax_immunity_factor * raw_total_exposure / immune_force)


class VaxSusceptibleToExposed(SusceptibleToExposed):
    """
    SusceptibleToExposed-derived class for movement from the
    "S_V" to "E_V" compartment (vaccinated track).

    Identical to `SusceptibleToExposed` except that the relative
    susceptibility of people in "S_V" is that of people in "S"
    multiplied by `1 - vax_induced_inf_risk_reduce`.
    """

    def susceptibility_multiplier(self,
                                  params: FluSubpopParams):
        return 1 - np.asarray(params.vax_induced_inf_risk_reduce)


class RecoveredToSusceptible(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "R" to "S" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.R_to_S_rate)


class ExposedToAsymp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "E" to "IA" compartment. The functional form is the same across
    subpopulations.

    Each ExposedToAsymp instance forms a TransitionVariableGroup with
    a corresponding ExposedToPresymp instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * params.E_to_IA_prop)


class ExposedToPresymp(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "E" to "IP" compartment. The functional form is the same across
    subpopulations.

    Each ExposedToPresymp instance forms a TransitionVariableGroup with
    a corresponding ExposedToAsymp instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.E_to_I_rate * (1 - params.E_to_IA_prop))


class PresympToSympRecover(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IP" to "ISR" compartment. The functional form is the same across
    subpopulations.
    
    Each PresympToSympRecover instance forms a TransitionVariableGroup with
    a corresponding PresympToSympHospital instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

        immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

        vax_immunity_factor = 1 - state.MV * params.vax_induced_hosp_risk_reduce_initial

        prob_hosp = (params.IP_to_ISH_prop / immunity_force) * vax_immunity_factor * \
            self.hosp_multiplier(params)

        return np.asarray((1 - prob_hosp) * params.IP_to_IS_rate)

    def hosp_multiplier(self,
                        params: FluSubpopParams):
        """
        Multiplier on the probability of hospitalization -- 1 for
        "IP", overridden for "IP_V" in `VaxPresympToSympRecover`.
        """

        return 1.0


class PresympToSympHospital(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IP" to "ISH" compartment. The functional form is the same across
    subpopulations.
    
    Each PresympToSympHospital instance forms a TransitionVariableGroup with
    a corresponding PresympToSympRecover instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        inf_induced_hosp_risk_reduce = params.inf_induced_hosp_risk_reduce
        inf_induced_proportional_risk_reduce = inf_induced_hosp_risk_reduce / (1 - inf_induced_hosp_risk_reduce)

        immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

        vax_immunity_factor = 1 - state.MV * params.vax_induced_hosp_risk_reduce_initial

        prob_hosp = (params.IP_to_ISH_prop / immunity_force) * vax_immunity_factor * \
            self.hosp_multiplier(params)

        return np.asarray(prob_hosp * params.IP_to_IS_rate)

    def hosp_multiplier(self,
                        params: FluSubpopParams):
        """
        Multiplier on the probability of hospitalization -- 1 for
        "IP", overridden for "IP_V" in `VaxPresympToSympHospital`.
        """

        return 1.0


class VaxPresympToSympRecover(PresympToSympRecover):
    """
    PresympToSympRecover-derived class for movement from the
    "IP_V" to "ISR_V" compartment (vaccinated track) -- the
    complement of `VaxPresympToSympHospital`.
    """

    def hosp_multiplier(self,
                        params: FluSubpopParams):
        return compute_vax_conditional_multipliers(params)[0]


class VaxPresympToSympHospital(PresympToSympHospital):
    """
    PresympToSympHospital-derived class for movement from the
    "IP_V" to "ISH_V" compartment (vaccinated track).

    The probability of hospitalization is that of people in "IP"
    multiplied by `(1 - vax_induced_hosp_risk_reduce) /
    (1 - vax_induced_inf_risk_reduce)` -- the conditional (given
    infection) multiplier, since vaccinated people already had their
    infection risk reduced at "S_V" -> "E_V". See
    `compute_vax_conditional_multipliers`.
    """

    def hosp_multiplier(self,
                        params: FluSubpopParams):
        return compute_vax_conditional_multipliers(params)[0]


class SympRecoverToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "ISR" to "R" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.ISR_to_R_rate)


class SympHospitalToHospRecover(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "ISH" to "HR" compartment. The functional form is the same across
    subpopulations.
    
    Each SympHospitalToHospRecover instance forms a TransitionVariableGroup with
    a corresponding SympHospitalToHospDead instance (these two
    transition variables are jointly distributed).
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        
        inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce

        inf_induced_proportional_risk_reduce = \
            inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

        immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

        vax_immunity_factor = 1 - state.MV * params.vax_induced_death_risk_reduce_initial

        prob_death = (params.ISH_to_HD_prop / immunity_force) * vax_immunity_factor * \
            self.death_multiplier(params)

        return np.asarray((1 - prob_death) * params.ISH_to_H_rate)

    def death_multiplier(self,
                         params: FluSubpopParams):
        """
        Multiplier on the probability of death -- 1 for "ISH",
        overridden for "ISH_V" in `VaxSympHospitalToHospRecover`.
        """

        return 1.0


class SympHospitalToHospDead(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "ISH" to "HD" compartment. The functional form is the same across
    subpopulations.
    
    Each SympHospitalToHospDead instance forms a TransitionVariableGroup with
    a corresponding SympHospitalToHospRecover instance (these two
    transition variables are jointly distributed).
    
    The rate of SympHospitalToHospDead decreases as population-level immunity
    against hospitalization increases.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        
        inf_induced_death_risk_reduce = params.inf_induced_death_risk_reduce

        inf_induced_proportional_risk_reduce = \
            inf_induced_death_risk_reduce / (1 - inf_induced_death_risk_reduce)

        immunity_force = 1 + inf_induced_proportional_risk_reduce * state.M

        vax_immunity_factor = 1 - state.MV * params.vax_induced_death_risk_reduce_initial

        prob_death = (params.ISH_to_HD_prop / immunity_force) * vax_immunity_factor * \
            self.death_multiplier(params)

        return np.asarray(prob_death * params.ISH_to_H_rate)

    def death_multiplier(self,
                         params: FluSubpopParams):
        """
        Multiplier on the probability of death -- 1 for "ISH",
        overridden for "ISH_V" in `VaxSympHospitalToHospDead`.
        """

        return 1.0


class VaxSympHospitalToHospRecover(SympHospitalToHospRecover):
    """
    SympHospitalToHospRecover-derived class for movement from the
    "ISH_V" to "HR_V" compartment (vaccinated track) -- the
    complement of `VaxSympHospitalToHospDead`.
    """

    def death_multiplier(self,
                         params: FluSubpopParams):
        return compute_vax_conditional_multipliers(params)[1]


class VaxSympHospitalToHospDead(SympHospitalToHospDead):
    """
    SympHospitalToHospDead-derived class for movement from the
    "ISH_V" to "HD_V" compartment (vaccinated track).

    The probability of death is that of people in "ISH"
    multiplied by `(1 - vax_induced_death_risk_reduce) /
    (1 - vax_induced_hosp_risk_reduce)` -- the conditional (given
    hospitalization) multiplier. See
    `compute_vax_conditional_multipliers`.
    """

    def death_multiplier(self,
                         params: FluSubpopParams):
        return compute_vax_conditional_multipliers(params)[1]

        

class AsympToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "IA" to "R" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.IA_to_R_rate)


class HospRecoverToRecovered(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "HR" to "R" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """
        
        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.HR_to_R_rate)


class HospDeadToDead(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "HD" to "D" compartment. The functional form is the same across
    subpopulations.
    """

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R)
        """

        return np.full((params.num_age_groups, params.num_risk_groups),
                       params.HD_to_D_rate)


def compute_vax_dose_pool(S: np.ndarray,
                          S_V: np.ndarray,
                          total_pop_age_risk: np.ndarray,
                          vax_dose_pool: str) -> np.ndarray:
    """
    Returns the number of people the `daily_vaccines` proportions
    apply to, for each age-risk group -- see `vax_dose_pool` in
    `FluSubpopParams`.

    Returns:
        np.ndarray of shape (A, R)
    """

    if vax_dose_pool == "total_population":
        return np.asarray(total_pop_age_risk, dtype=float)
    elif vax_dose_pool == "susceptible":
        return np.asarray(S, dtype=float) + np.asarray(S_V, dtype=float)
    else:
        raise FluSubpopModelError(
            f"`vax_dose_pool` must be one of {VAX_DOSE_POOLS} -- got {vax_dose_pool!r}.")


def round_with_carry(expected: np.ndarray,
                     carry: np.ndarray,
                     is_integer_valued: bool) -> tuple:
    """
    Turns an expected (fractional) number of people to move into the
    number actually moved this timestep, for each age-risk group.

    For integer-valued transition types, `expected` is added to the
    fractional `carry` owed from earlier timesteps and only the rounded
    total is moved, so `carry` stays in [-0.5, 0.5). The cumulative
    number moved is therefore always within 0.5 of the cumulative
    number expected -- rounding every timestep (or every day)
    independently would instead drift, and a group expecting e.g. 0.3
    people per day would never move anyone. Otherwise `expected` is
    moved as-is and `carry` is unchanged.

    Returns:
        (np.ndarray, np.ndarray):
            number to move and updated carry, each of shape (A, R).
    """

    if not is_integer_valued:
        return expected, carry

    carry = carry + expected
    target = np.floor(carry + 0.5)

    return target, carry - target


class ScheduledVaccination(clt.TransitionVariable):
    """
    TransitionVariable-derived class for movement from the
    "S" to "S_V" compartment -- the entry point of the
    vaccinated track.

    The number moved is deterministic and set by the `daily_vaccines`
    schedule rather than drawn from a distribution, similar to the
    `scheduled_exact` transitions of `generic_core`. At the start of
    each day, `FluSubpopModel.prepare_daily_state` calls
    `set_daily_expected` with the expected number of people vaccinated
    that day: the day's `daily_vaccines` proportion times the dose pool
    (see `compute_vax_dose_pool`). This expected number is spread
    evenly over the day's timesteps, the same way `daily_vaccines` was
    divided by the number of timesteps when it fed the (now switched
    off) `VaxInducedImmunity` epi metric, and rounded with
    `round_with_carry` for integer-valued transition types.

    The amount moved is capped at what is left in "S" after this
    timestep's "S" to "E" transition, so "S" never goes negative. A
    capped shortfall is recorded in `cumulative_capped_shortfall`
    (and warned about once) but is not carried forward -- those
    people were not in "S" to be vaccinated.

    Attributes:
        competing_outflow (SusceptibleToExposed):
            the other outflow from "S" -- must be realized before
            this transition variable each timestep (see
            `FluSubpopModel.create_transition_variables`).
        is_integer_valued (bool):
            whether to round the amount moved (see above).
        daily_expected (np.ndarray of shape (A, R)):
            expected number vaccinated over the current day.
        carry (np.ndarray of shape (A, R)):
            fractional amount owed but not yet moved, in [-0.5, 0.5).
        cumulative_capped_shortfall (np.ndarray of shape (A, R)):
            people the schedule called for since the start of the
            simulation who could not be moved because "S" ran out.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 origin: clt.Compartment,
                 destination: clt.Compartment,
                 competing_outflow: clt.TransitionVariable,
                 transition_type: clt.TransitionTypes):

        super().__init__(origin,
                         destination,
                         "scheduled_exact",
                         is_jointly_distributed=False)

        self.competing_outflow = competing_outflow
        self.is_integer_valued = "no_round" not in transition_type

        self._reset_counters()

    def _reset_counters(self) -> None:

        self._warned_cap = False

        self.daily_expected = None
        self.carry = None
        self.cumulative_capped_shortfall = None

    def set_daily_expected(self,
                           state: FluSubpopState,
                           params: FluSubpopParams) -> None:
        """
        Sets the expected number of people vaccinated over the current
        day, from the dose pool at the start of the day -- called once
        a day by `FluSubpopModel.prepare_daily_state`, after the
        schedules are updated and the vaccinated-track reset is applied.
        """

        pool = compute_vax_dose_pool(state.S,
                                     state.S_V,
                                     params.total_pop_age_risk,
                                     params.vax_dose_pool)

        self.daily_expected = np.asarray(state.daily_vaccines, dtype=float) * pool

    def get_current_rate(self,
                         state: FluSubpopState,
                         params: FluSubpopParams) -> np.ndarray:
        """
        Returns the `daily_vaccines` proportion for the current day --
        the amount moved comes from `daily_expected` instead.

        Returns:
            np.ndarray of shape (A, R)
        """

        return np.asarray(state.daily_vaccines, dtype=float)

    def get_scheduled_exact_realization(self,
                                        RNG: np.random.Generator,
                                        num_timesteps: int) -> np.ndarray:
        """
        See class docstring. The `RNG` parameter is not used.

        Returns:
            np.ndarray of shape (A, R)
        """

        if self.daily_expected is None:
            raise FluSubpopModelError(
                "`S_to_S_V.daily_expected` has not been set -- "
                "`FluSubpopModel.prepare_daily_state` must run at the start "
                "of each simulation day.")

        expected = self.daily_expected / num_timesteps

        if self.carry is None:
            self.carry = np.zeros_like(expected)
            self.cumulative_capped_shortfall = np.zeros_like(expected)

        target, self.carry = round_with_carry(expected, self.carry, self.is_integer_valued)

        competing_val = self.competing_outflow.current_val
        if competing_val is None:
            competing_val = 0.0
        available = np.maximum(np.asarray(self.origin.current_val, dtype=float) -
                               np.asarray(competing_val, dtype=float), 0.0)

        moved = np.minimum(target, available)
        shortfall = target - moved

        self.cumulative_capped_shortfall = self.cumulative_capped_shortfall + shortfall

        if np.any(shortfall > 0) and not self._warned_cap:
            self._warned_cap = True
            warnings.warn(
                "Scheduled vaccinations exceeded the number of people left in "
                "\"S\" for some age-risk group(s) and were capped. The shortfall "
                "is recorded in `S_to_S_V.cumulative_capped_shortfall`. Further "
                "occurrences are not reported.")

        return moved

    def reset(self) -> None:
        super().reset()
        self._reset_counters()


def compute_pre_start_vaccination_shift(S_init: np.ndarray,
                                        S_V_init: np.ndarray,
                                        total_pop_age_risk: np.ndarray,
                                        current_real_date: datetime.date,
                                        params: FluSubpopParams,
                                        schedules: sc.objdict,
                                        timesteps_per_day: int,
                                        is_integer_valued: bool) -> np.ndarray:
    """
    Returns the number of people to move from "S" to "S_V" at the
    start of the simulation to account for vaccinations scheduled
    before the simulation start date.

    Mirrors how `VaxInducedImmunity` used to adjust its initial value:
    if `vax_immunity_reset_date_mm_dd` is None, there is no
    adjustment. Otherwise, the window runs from the most recent
    occurrence of the reset date before `current_real_date` up to
    (not including) `current_real_date`. The `daily_vaccines` index is
    already the protection date (see `DailyVaccines.postprocess_data_input`),
    so the protection delay is NOT added to the reset date.

    Within the window, the vaccination rule of `ScheduledVaccination`
    is replayed day by day and timestep by timestep (same dose pool,
    same `round_with_carry` rounding, capped at what is left in "S"),
    starting from the input `S_init` and `S_V_init` -- which are
    therefore treated as the values at the reset date. As in the old
    `VaxInducedImmunity` adjustment, no other transitions are replayed.

    Returns:
        np.ndarray of shape (A, R)
    """

    S_remaining = np.asarray(S_init, dtype=float).copy()
    moved_total = np.zeros_like(S_remaining)

    if params.vax_immunity_reset_date_mm_dd is None:
        return moved_total

    month, day = params.vax_immunity_reset_date_mm_dd.split('_')
    reset_date = datetime.date(current_real_date.year, int(month), int(day))
    if reset_date >= current_real_date:
        reset_date = datetime.date(current_real_date.year - 1, int(month), int(day))

    vaccines_df = schedules['daily_vaccines'].timeseries_df

    mask = (vaccines_df.index >= reset_date) & (vaccines_df.index < current_real_date)
    relevant_vaccines = vaccines_df.loc[mask, "daily_vaccines"].sort_index()

    S_V_current = np.asarray(S_V_init, dtype=float).copy()
    carry = np.zeros_like(S_remaining)

    for daily_vaccines in relevant_vaccines:

        pool = compute_vax_dose_pool(S_remaining, S_V_current + moved_total,
                                     total_pop_age_risk, params.vax_dose_pool)
        expected = np.asarray(daily_vaccines, dtype=float) * pool / timesteps_per_day

        for _ in range(timesteps_per_day):
            target, carry = round_with_carry(expected, carry, is_integer_valued)
            moved = np.minimum(target, S_remaining)
            S_remaining = S_remaining - moved
            moved_total = moved_total + moved

    return moved_total


def _zero_epi_metric_init_val(init_val,
                              params: FluSubpopParams,
                              metric_name: str) -> np.ndarray:
    """
    Returns zeros shaped like `init_val` (A x R when `init_val` is None),
    warning if the input initial value was nonzero -- in the
    vaccinated-track model, the M and MV epi metrics are switched off
    and always stay at zero, so any input value is ignored.
    """

    if init_val is None:
        return np.zeros((params.num_age_groups, params.num_risk_groups))

    init_val = np.asarray(init_val, dtype=float)

    if np.any(init_val != 0):
        warnings.warn(
            f"Initial value of {metric_name} is nonzero ({init_val.tolist()}) but "
            f"{metric_name} is switched off in the vaccinated-track model and "
            "always stays at 0 -- the input value is ignored.")

    return np.zeros_like(init_val)


class InfInducedImmunity(clt.EpiMetric):
    """
    EpiMetric-derived class for infection-induced
    population-level immunity.

    Switched off in the vaccinated-track model: M starts at zero
    (whatever the input initial value), never changes, and is never
    injected. The class is kept so that M still exists as a state
    variable for code that reads it (plotting, torch tensors, JSON
    loaders), and so that the M terms in the transition rates
    simply evaluate to 1.

    Params:
        R_to_S (RecoveredToSusceptible):
            RecoveredToSusceptible TransitionVariable
            in the SubpopModel -- kept for interface compatibility.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val,
                 R_to_S,
                 current_real_date: datetime.date,
                 params: FluSubpopParams,
                 timesteps_per_day: int):
        self.R_to_S = R_to_S
        self.pending_injection_date = None

        adjusted_init_val = self.adjust_initial_value(
            init_val, current_real_date, params, timesteps_per_day)
        super().__init__(adjusted_init_val)

    def adjust_initial_value(self,
                             init_val: np.ndarray,
                             current_real_date: datetime.date,
                             params: FluSubpopParams,
                             timesteps_per_day: int):
        """
        Returns zeros -- see class docstring.
        """

        self.original_init_val = copy.deepcopy(init_val)
        self.adjusted_init_val = _zero_epi_metric_init_val(init_val, params, "M")
        self.pending_injection_date = None

        return self.adjusted_init_val

    def check_and_apply_injection(self,
                                  current_date: datetime.date,
                                  params: FluSubpopParams):
        """
        No-op -- see class docstring.
        """

        pass

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R) of zeros -- see class docstring.
        """

        return np.zeros_like(np.asarray(state.M, dtype=float))


class VaxInducedImmunity(clt.EpiMetric):
    """
    EpiMetric-derived class for vaccine-induced
    population-level immunity.

    Switched off in the vaccinated-track model: vaccination is
    modeled by moving people from "S" to "S_V" instead (see
    `ScheduledVaccination`). MV starts at zero (whatever the input
    initial value) and never changes. The class is kept so that MV
    still exists as a state variable for code that reads it, and so
    that the MV terms in the transition rates simply evaluate to 1.
    """

    def __init__(self,
                 init_val,
                 current_real_date: datetime.date,
                 params: FluSubpopParams,
                 schedules: clt.Schedule,
                 timesteps_per_day: int):

        adjusted_init_val = self.adjust_initial_value(
            init_val, current_real_date, params, schedules, timesteps_per_day)
        super().__init__(adjusted_init_val)

    def adjust_initial_value(self,
                             init_val: np.ndarray,
                             current_real_date: datetime.date,
                             params: FluSubpopParams,
                             schedules: clt.Schedule,
                             timesteps_per_day: int):
        """
        Returns zeros -- see class docstring. Vaccinations before the
        simulation start date are accounted for by the initial "S_V"
        value instead (see `compute_pre_start_vaccination_shift`).
        """

        self.original_init_val = copy.deepcopy(init_val)
        self.adjusted_init_val = _zero_epi_metric_init_val(init_val, params, "MV")

        return self.adjusted_init_val

    def get_change_in_current_val(self,
                                  state: FluSubpopState,
                                  params: FluSubpopParams,
                                  num_timesteps: int) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (A, R) of zeros -- see class docstring.
        """

        return np.zeros_like(np.asarray(state.MV, dtype=float))

    def check_and_apply_reset(self,
                              current_date: datetime.date,
                              params: FluSubpopParams):
        """
        No-op -- the vaccine immunity reset now moves people from
        "S_V" back to "S" (see `FluSubpopModel.check_and_apply_vax_track_reset`).
        """

        pass


def compute_vax_induced_risk_reduce_initial(params: FluSubpopParams,
                                            schedules: sc.objdict,
                                            start_real_date: datetime.date) -> tuple:
    """
    Computes the "zero-waning" (peak, just-after-protection-delay) vaccine
    efficacy values `vax_induced_inf_risk_reduce_initial`,
    `vax_induced_hosp_risk_reduce_initial`, and `vax_induced_death_risk_reduce_initial`
    from the corresponding season-average efficacy values
    (`vax_induced_inf_risk_reduce`, `vax_induced_hosp_risk_reduce`,
    `vax_induced_death_risk_reduce`), `vax_induced_immune_wane`, and the
    `daily_vaccines` schedule.

    For each age-risk group, given the season's vaccination timing
    (`p_prot`, the proportion of effective doses -- i.e. vaccination date
    plus protection delay -- given on each day of the season) and waning
    rate w_V, this solves for the peak efficacy VE_0 such that the
    dose-timing-weighted average realized efficacy over the season equals
    the input (season-average) efficacy value:

        VE_0 = VE_season * w_V * T /
               ((1 - exp(-w_V)) * sum_{tau=t0}^{T-1} [
                   (sum_{u=t0}^{tau} p_prot(u) * exp(-w_V * (tau - u))) /
                   (sum_{u=t0}^{tau} p_prot(u))
               ])

    Note that VE_0 is *linear* in VE_season -- the bracketed quantity
    depends only on w_V and the dose timing profile, so it is computed
    once per age-risk group and reused for all three efficacy fields.

    Season window
    -------------
    The season window is the period between two consecutive occurrences of
    `vax_immunity_reset_date_mm_dd` (the occurrence on or before
    `start_real_date`, through one year later), or if that parameter is
    not set, the 12 months starting from the first date covered by the
    `daily_vaccines` schedule. This window is further intersected with
    the actual date range covered by the `daily_vaccines` schedule (which
    matters when the reset-date window extends past the end of the
    schedule, or -- in the no-reset-date case -- when the schedule itself
    spans less than 12 months).

    Within that window, t0 and T are by default the first day with a
    nonzero dose and the number of days through the last day with a
    nonzero dose. Because the average above is an *unweighted* average
    over the T days, a long low-dose tail (a schedule that trickles a
    handful of doses through the spring, say) pulls the average down and
    so inflates VE_0. Setting `params.VE_season_dose_window_quantile` to a
    value q in [0, 0.5) trims the window to the days spanning the central
    (1 - 2q) of the season's cumulative doses, dropping the sparse tails
    at both ends -- see that parameter's docstring.

    Capping
    -------
    VE_0 is a probability and is capped at 1.0. Because VE_0 is
    VE_season divided by the season's mean waning factor (which is < 1),
    plausible inputs can push it above 1: VE_season = 0.8 with
    `vax_induced_immune_wane` = 0.004 over a year-long schedule gives
    VE_0 ~ 1.07. Values above 1 would make the applied factor
    `1 - MV * VE_0` negative at high vaccination coverage (a negative
    force of infection). Any capped entry raises a warning naming the
    affected age-risk groups, since a cap means the requested
    season-average efficacy is not actually achievable under the given
    waning rate and dose timing.

    Edge cases:
        - If waning (w_V) is 0 for a given age-risk group, VE_0 equals the
          input (season-average) value for that group -- no adjustment is
          needed since there is no waning to correct for.
        - If there are no vaccine doses in the (intersected) season window
          for a given age-risk group, VE_0 equals the input value for that
          group -- vaccine-induced immunity is always 0 for that group, so
          the value is never actually applied.
        - If `params.adjust_VE_for_seasonal_waning` is False, this
          adjustment is skipped entirely and the input (season-average)
          values are returned unchanged (broadcast to shape (A, R)), still
          capped at 1.0.

    Args:
        params (FluSubpopParams):
            holds `vax_induced_inf_risk_reduce`, `vax_induced_hosp_risk_reduce`,
            `vax_induced_death_risk_reduce`, `vax_induced_immune_wane`,
            `vax_immunity_reset_date_mm_dd`,
            `VE_season_dose_window_quantile`, `num_age_groups`, and
            `num_risk_groups`.
        schedules (sc.objdict):
            holds the `daily_vaccines` `Schedule` instance (already shifted
            by `vax_protection_delay_days`, date-indexed, one A x R array
            per day).
        start_real_date (datetime.date):
            real-world date corresponding to the start of the simulation --
            used to anchor the season window.

    Returns:
        (vax_induced_inf_risk_reduce_initial, vax_induced_hosp_risk_reduce_initial,
         vax_induced_death_risk_reduce_initial), each an np.ndarray of shape (A, R).
    """

    target_shape = (params.num_age_groups, params.num_risk_groups)

    field_names = ("vax_induced_inf_risk_reduce",
                   "vax_induced_hosp_risk_reduce",
                   "vax_induced_death_risk_reduce")

    ve_season_arrs = {
        name: np.broadcast_to(
            np.asarray(getattr(params, name), dtype=float), target_shape).copy()
        for name in field_names
    }

    if params.adjust_VE_for_seasonal_waning is False:
        return tuple(_cap_vax_induced_risk_reduce_initial(ve_season_arrs[name], name)
                     for name in field_names)

    w_arr = np.broadcast_to(
        np.asarray(params.vax_induced_immune_wane, dtype=float), target_shape)

    doses_stack = _season_window_doses(params, schedules, start_real_date, target_shape)

    # `inflation_factor[a, r]` is VE_0 / VE_season for that age-risk group --
    #   it depends only on the waning rate and the dose timing profile, so
    #   it is shared across all three efficacy fields
    inflation_factor = np.ones(target_shape)

    trim_quantile = params.VE_season_dose_window_quantile

    if trim_quantile is not None and not 0 <= trim_quantile < 0.5:
        raise FluSubpopModelError(
            f"`VE_season_dose_window_quantile` must be None or in [0, 0.5) -- "
            f"got {trim_quantile}. It trims that fraction of cumulative doses "
            "off each end of the season window, so 0.5 or more would leave "
            "nothing behind.")

    for a in range(target_shape[0]):
        for r in range(target_shape[1]):

            w_ar = w_arr[a, r]

            if w_ar == 0 or doses_stack.shape[0] == 0:
                continue

            sub = _trimmed_dose_profile(doses_stack[:, a, r], trim_quantile)

            if sub is None:
                continue

            T = sub.size
            p_prot = sub / sub.sum()
            cumsum_p_prot = np.cumsum(p_prot)

            # numer[n] = sum_{u=0}^{n} p_prot(u) * exp(-w * (n - u))
            #          = p_prot(n) + exp(-w) * numer[n - 1]
            decay = np.exp(-w_ar)
            numer = np.empty(T)
            acc = 0.0
            for n in range(T):
                acc = p_prot[n] + decay * acc
                numer[n] = acc

            S = np.sum(numer / cumsum_p_prot)

            inflation_factor[a, r] = w_ar * T / ((1 - decay) * S)

    return tuple(
        _cap_vax_induced_risk_reduce_initial(ve_season_arrs[name] * inflation_factor, name)
        for name in field_names
    )


def _cap_vax_induced_risk_reduce_initial(ve_initial_arr: np.ndarray,
                                         field_name: str) -> np.ndarray:
    """
    Caps peak vaccine efficacy at 1.0, warning if any age-risk group
    was actually capped.

    A capped entry means the requested season-average efficacy is not
    achievable given the waning rate and dose timing -- even perfect
    (100%) protection at the moment of vaccination would average out to
    less than the requested value over the season. The simulation stays
    well-defined (`1 - MV * VE_0` remains nonnegative), but realized
    efficacy will fall short of the input value, so this is worth
    surfacing rather than silently clipping.

    Args:
        ve_initial_arr (np.ndarray of shape (A, R)):
            uncapped peak efficacy values.
        field_name (str):
            name of the source season-average parameter, used in the
            warning message.

    Returns:
        np.ndarray of shape (A, R), with all entries <= 1.0.
    """

    over_idxs = np.argwhere(ve_initial_arr > 1.0)

    if over_idxs.size > 0:
        max_val = float(np.max(ve_initial_arr))
        groups_str = ", ".join(f"(age {a}, risk {r})" for a, r in over_idxs)
        warnings.warn(
            f"`{field_name}_initial` exceeded 1.0 (max {max_val:.4f}) for "
            f"age-risk group(s) {groups_str} and was capped at 1.0. This means "
            f"`{field_name}` is not achievable given `vax_induced_immune_wane` "
            "and the `daily_vaccines` timing -- even 100% protection at the "
            "moment of vaccination would average to less than the requested "
            "season-average value. Realized efficacy will be lower than "
            "requested for these groups. Consider lowering "
            f"`{field_name}`, lowering `vax_induced_immune_wane`, or setting "
            "`VE_season_dose_window_quantile` to trim sparse dose tails from "
            "the season window."
        )

    return np.minimum(ve_initial_arr, 1.0)


def _vax_conditional_ratio(ve_outcome, ve_prior_step) -> np.ndarray:
    """
    Returns `(1 - ve_outcome) / (1 - ve_prior_step)`, capped at 1.0, and
    set to 1.0 where `ve_prior_step` is 1 (nobody reaches that step, so the
    value is never applied). See `compute_vax_conditional_multipliers`.
    """

    ve_outcome = np.asarray(ve_outcome, dtype=float)
    ve_prior_step = np.asarray(ve_prior_step, dtype=float)

    denom = 1 - ve_prior_step
    safe_denom = np.where(denom > 0, denom, 1.0)
    ratio = np.where(denom > 0, (1 - ve_outcome) / safe_denom, 1.0)

    return np.minimum(ratio, 1.0)


def compute_vax_conditional_multipliers(params: FluSubpopParams) -> tuple:
    """
    Converts the overall (unconditional) vaccine efficacies
    `vax_induced_hosp_risk_reduce` and `vax_induced_death_risk_reduce` into
    the conditional multipliers applied along the vaccinated track.

    The vaccinated track applies its reductions sequentially --
    `1 - vax_induced_inf_risk_reduce` at "S_V" -> "E_V", then the hosp
    multiplier at "IP_V" -> "ISH_V", then the death multiplier at
    "ISH_V" -> "HD_V" -- so the multipliers must be conditional on the
    previous step for the products to equal the overall efficacies:

        hosp_mult  = (1 - VE_hosp)  / (1 - VE_inf)
        death_mult = (1 - VE_death) / (1 - VE_hosp)

    so that (1 - VE_inf) * hosp_mult = 1 - VE_hosp and
    (1 - VE_inf) * hosp_mult * death_mult = 1 - VE_death.

    Each multiplier is capped at 1.0 (conditional efficacy floored at 0)
    -- a ratio above 1 means the overall efficacy for that outcome is
    lower than for the previous step, which the sequential structure
    cannot represent. See `_warn_vax_conditional_clipping`.

    Returns:
        (hosp_mult, death_mult), each an np.ndarray broadcastable to (A, R).
    """

    hosp_mult = _vax_conditional_ratio(params.vax_induced_hosp_risk_reduce,
                                       params.vax_induced_inf_risk_reduce)
    death_mult = _vax_conditional_ratio(params.vax_induced_death_risk_reduce,
                                        params.vax_induced_hosp_risk_reduce)

    return hosp_mult, death_mult


def _warn_vax_conditional_clipping(params: FluSubpopParams) -> None:
    """
    Warns if `compute_vax_conditional_multipliers` caps either multiplier
    at 1.0 for any age-risk group -- i.e. if `vax_induced_hosp_risk_reduce`
    < `vax_induced_inf_risk_reduce` or `vax_induced_death_risk_reduce`
    < `vax_induced_hosp_risk_reduce`. Realized overall efficacy against
    that outcome then equals the previous step's efficacy, which is
    higher than requested.
    """

    target_shape = (params.num_age_groups, params.num_risk_groups)

    def as_arr(name):
        return np.broadcast_to(np.asarray(getattr(params, name), dtype=float), target_shape)

    pairs = (("vax_induced_hosp_risk_reduce", "vax_induced_inf_risk_reduce"),
             ("vax_induced_death_risk_reduce", "vax_induced_hosp_risk_reduce"))

    for outcome_name, prior_name in pairs:
        outcome_arr = as_arr(outcome_name)
        prior_arr = as_arr(prior_name)
        under_idxs = np.argwhere((outcome_arr < prior_arr) & (prior_arr < 1))

        if under_idxs.size > 0:
            groups_str = ", ".join(f"(age {a}, risk {r})" for a, r in under_idxs)
            warnings.warn(
                f"`{outcome_name}` is lower than `{prior_name}` for age-risk "
                f"group(s) {groups_str}. Vaccine efficacies are applied "
                "sequentially along the vaccinated track, so the conditional "
                f"efficacy for `{outcome_name}` is floored at 0 and the "
                f"realized overall efficacy equals `{prior_name}` for these "
                "groups -- higher than requested."
            )


def _season_window_doses(params: FluSubpopParams,
                         schedules: sc.objdict,
                         start_real_date: datetime.date,
                         target_shape: tuple) -> np.ndarray:
    """
    Returns the `daily_vaccines` doses falling inside the vaccination
    season window, as an np.ndarray of shape (T, A, R) -- see
    `compute_vax_induced_risk_reduce_initial` for how the window is
    defined. Returns an array with T == 0 if the window contains no
    schedule days.
    """

    vaccines_df = schedules["daily_vaccines"].timeseries_df

    schedule_min_date = vaccines_df.index.min()
    schedule_max_date = vaccines_df.index.max()

    if params.vax_immunity_reset_date_mm_dd is not None:
        month, day = (int(x) for x in params.vax_immunity_reset_date_mm_dd.split('_'))
        window_start = datetime.date(start_real_date.year, month, day)
        if window_start >= start_real_date:
            window_start = datetime.date(start_real_date.year - 1, month, day)
        window_end = datetime.date(window_start.year + 1, month, day)
    else:
        window_start = schedule_min_date
        window_end = schedule_min_date + datetime.timedelta(days=365)

    window_start = max(window_start, schedule_min_date)
    window_end = min(window_end, schedule_max_date + datetime.timedelta(days=1))

    if window_start < window_end:
        mask = (vaccines_df.index >= window_start) & (vaccines_df.index < window_end)
        window_doses_df = vaccines_df.loc[mask]
    else:
        window_doses_df = vaccines_df.iloc[0:0]

    if window_doses_df.empty:
        return np.zeros((0,) + target_shape)

    doses_stack = np.stack(window_doses_df["daily_vaccines"].values, axis=0)

    if doses_stack.shape[1:] != target_shape:
        # Time series has a different age-risk resolution than the
        # risk-reduce parameters -- aggregate (sum) across all
        # dimensions and broadcast the resulting total evenly across
        # every age-risk group.
        doses_stack = doses_stack.reshape(doses_stack.shape[0], -1).sum(axis=1, keepdims=True)
        doses_stack = np.broadcast_to(doses_stack, (doses_stack.shape[0],) + target_shape)

    return doses_stack


def _trimmed_dose_profile(cell_doses: np.ndarray,
                          trim_quantile: Optional[float]) -> Optional[np.ndarray]:
    """
    Returns the slice of one age-risk group's daily doses that defines
    the season window for the VE_0 calculation, or None if the group has
    no doses at all.

    The base window runs from the first to the last day with a nonzero
    dose. If `trim_quantile` is a value q in [0, 0.5), the window is
    further narrowed to the days spanning the central (1 - 2q) of the
    group's cumulative doses -- i.e. days before the qth and after the
    (1 - q)th quantile of the cumulative dose distribution are dropped.
    This removes sparse dose tails that would otherwise stretch the
    (unweighted) season average over months with almost no vaccination.

    Args:
        cell_doses (np.ndarray of shape (T,)):
            daily doses for one age-risk group over the season window.
        trim_quantile (Optional[float]):
            q as described above -- None or 0 leaves the window untrimmed.

    Returns:
        np.ndarray of shape (T',) with a positive sum, or None.
    """

    nonzero_idxs = np.flatnonzero(cell_doses > 0)

    if nonzero_idxs.size == 0:
        return None

    sub = cell_doses[nonzero_idxs[0]:nonzero_idxs[-1] + 1]

    if not trim_quantile:
        return sub

    cumulative_prop = np.cumsum(sub) / sub.sum()

    # `lo` is the first day by which the trimmed-off leading mass has
    #   accumulated, `hi` the first day reaching the upper cutoff --
    #   both are kept, so the window spans the central mass inclusively
    lo = int(np.searchsorted(cumulative_prop, trim_quantile, side="left"))
    hi = int(np.searchsorted(cumulative_prop, 1.0 - trim_quantile, side="left"))

    return sub[lo:hi + 1]


class BetaReduce(clt.DynamicVal):
    """
    "Toy" function representing staged-alert policy
        that reduces transmission by 50% when more than 5%
        of the total population is infected. Note: the
        numbers are completely made up :)
    The "permanent_lockdown" toggle is to avoid "bang-bang"
        behavior where the staged-alert policy gets triggered
        one day and then is off the next, and then is on the
        day after, and so on... but as the name suggests,
        it IS permanent.
    TODO: replace with realistic function.
    """

    def __init__(self, init_val, is_enabled):
        super().__init__(init_val, is_enabled)
        self.permanent_lockdown = False

    def update_current_val(self, state, params):
        if np.sum(compute_symp_infectious(state)) / np.sum(params.total_pop_age_risk) > 0.05:
            self.current_val = .5
            self.permanent_lockdown = True
        else:
            if not self.permanent_lockdown:
                self.current_val = 0.0


class DailyVaccines(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None,
                 vax_protection_delay_days: int = 0):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                must have "date" and "daily_vaccines" -- "date" entries must
                correspond to consecutive calendar days and must either
                be strings with `"YYYY-MM-DD"` format or `datetime.date`
                objects -- "value" entries correspond to historical
                number vaccinated on those days. Identical to
                `FluSubpopSchedules` field of same name.
            vax_protection_delay_days (int):
                number of days to delay vaccine protection effect.
                Vaccines administered on day X become effective on day X + delay.
        """

        super().__init__(init_val)

        self.timeseries_df = timeseries_df
        self.vax_protection_delay_days = vax_protection_delay_days

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.timeseries_df.loc[current_date, "daily_vaccines"]

    def postprocess_data_input(self) -> None:
        """
            Converts daily_vaccines column from
            a string representation of a list of lists
            (each day) of format AxR into np.ndarray.
            Shifts dates forward by vax_protection_delay_days
            to model delayed vaccine protection, backfilling
            the beginning with zero entries.
            Pre-indexes the DataFrame by date for O(1) lookups.
        """

        self.timeseries_df['daily_vaccines'] = \
            self.timeseries_df['daily_vaccines'].apply(json.loads)
        self.timeseries_df.loc[:, 'daily_vaccines'] = \
            self.timeseries_df['daily_vaccines'].apply(
                lambda x: np.asarray(x)
                )

        if self.vax_protection_delay_days > 0:
            # Get the original start date and array shape for zero entries
            original_start_date = self.timeseries_df['date'].min()
            zero_array = np.zeros_like(self.timeseries_df['daily_vaccines'].iloc[0])

            # Shift all dates forward by the delay
            self.timeseries_df['date'] = self.timeseries_df['date'].apply(
                lambda d: d + datetime.timedelta(days=self.vax_protection_delay_days)
            )

            # Create backfill rows for the gap at the beginning using pd.date_range
            backfill_dates = pd.date_range(
                start=original_start_date,
                periods=self.vax_protection_delay_days,
                freq='D'
            ).date
            backfill_df = pd.DataFrame({
                'date': backfill_dates,
                'daily_vaccines': [zero_array.copy()] * self.vax_protection_delay_days
            })

            # Concatenate and sort by date
            self.timeseries_df = pd.concat([backfill_df, self.timeseries_df], ignore_index=True)
            self.timeseries_df = self.timeseries_df.sort_values('date').reset_index(drop=True)

        self.timeseries_df = self.timeseries_df.set_index('date')


class MobilityModifier(clt.Schedule):
    """
    Schedule for time-varying mobility modifier values.

    Attributes:
        timeseries_df (pd.DataFrame):
            There are 2 possible input formats:
            i) a standard schedule that must have columns "date" and
            "mobility_modifier" where "date" entries must correspond
            to consecutive calendar days and must either be strings with
            `"YYYY-MM-DD"` format or `datetime.date` objects 
            ii) a day of week schedule that must have columns "day_of_week"
            and "mobility_modifier" where "day_of_week" entries are
            strings with values from Monday to Sunday (case doesn't matter).
            The code will automatically detect which format is being used
            by looking at the column name.
            In both cases, "mobility_modifier" entries are
            JSON-encoded A x R arrays representing the proportion of
            time spent away from home by age-risk group on those days.
            Identical to `FluSubpopSchedules` field of same name.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                must have columns ("date" or "day_of_week") 
                and "mobility_modifier" --
                see class docstring for format details.
        """

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        if self.is_day_of_week_schedule:
            current_day_of_week = current_date.strftime('%A').lower()
            self.current_val = self.timeseries_df.loc[current_day_of_week, "mobility_modifier"]
        else:
            self.current_val = self.timeseries_df.loc[current_date, "mobility_modifier"]

    def postprocess_data_input(self) -> None:
        """
            Converts mobility_modifier column from
            a string representation of a list of lists
            (each day) of format AxR into np.ndarray.
            Check whether day_of_week schedule is being used.
            Make days of week lower case if being used.
            Pre-indexes the DataFrame by date or day_of_week for O(1) lookups.
        """

        if 'day_of_week' in self.timeseries_df.columns:
            self.is_day_of_week_schedule = True

        self.timeseries_df['mobility_modifier'] = \
            self.timeseries_df['mobility_modifier'].apply(json.loads)
        self.timeseries_df.loc[:, 'mobility_modifier'] = \
            self.timeseries_df['mobility_modifier'].apply(
                lambda x: np.asarray(x)
                )

        if self.is_day_of_week_schedule:
            self.timeseries_df['day_of_week'] = \
                self.timeseries_df['day_of_week'].str.lower()
            self.timeseries_df = self.timeseries_df.set_index('day_of_week')
        else:
            self.timeseries_df = self.timeseries_df.set_index('date')


class AbsoluteHumidity(clt.Schedule):

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None):
        """
        Args:
            init_val (Optional[np.ndarray | float]):
                starting value(s) at the beginning of the simulation
            timeseries_df (Optional[pd.DataFrame] = None):
                must have columns "date" and "absolute_humidity" --
                "date" entries must correspond to consecutive calendar days
                and must either be strings with `"YYYY-MM-DD"` format or
                `datetime.date` objects -- "value" entries correspond to
                absolute humidity on those days. Identical to
                `FluSubpopSchedules` field of same name.
        """

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self, params, current_date: datetime.date) -> None:
        self.current_val = self.timeseries_df.loc[current_date, "absolute_humidity"]

    def postprocess_data_input(self) -> None:
        self.timeseries_df = self.timeseries_df.set_index('date')


class FluContactMatrix(clt.Schedule):
    """
    Flu contact matrix.

    Attributes:
        timeseries_df (pd.DataFrame):
            must have columns "date", "is_school_day", and "is_work_day"
            -- "date" entries must correspond to consecutive calendar
            days and must either be strings with `"YYYY-MM-DD"` format
            or `datetime.date` object and "is_school_day" and
            "is_work_day" entries are floats between 0 and 1 indicating if 
            that date is a school day or work day. Identical to 
            `FluSubpopSchedules` field of same name.

    See parent class docstring for other attributes.
    """

    def __init__(self,
                 init_val: Optional[np.ndarray | float] = None,
                 timeseries_df: pd.DataFrame = None):

        super().__init__(init_val)

        self.timeseries_df = timeseries_df

    def update_current_val(self,
                           subpop_params: FluSubpopParams,
                           current_date: datetime.date) -> None:

        try:
            current_row = self.timeseries_df.loc[current_date]
            self.current_val = subpop_params.total_contact_matrix - \
                               (1 - current_row["is_school_day"]) * subpop_params.school_contact_matrix - \
                               (1 - current_row["is_work_day"]) * subpop_params.work_contact_matrix
        except KeyError:
            # print(f"Error: {current_date} is not in `timeseries_df`. Using total contact matrix.")
            self.current_val = subpop_params.total_contact_matrix

    def postprocess_data_input(self) -> None:
        self.timeseries_df = self.timeseries_df.set_index('date')


def compute_wtd_presymp_asymp_by_age(subpop_state: FluSubpopState,
                                     subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns weighted sum of IP and IA compartment for
        subpopulation with given state and parameters.
        IP and IA are weighted by their relative infectiousness
        respectively, and then summed over risk groups.

    Returns:
        np.ndarray of shape (A, R)
    """

    # Sum both tracks, then sum over risk groups
    wtd_IP = \
        subpop_params.IP_relative_inf * np.sum(subpop_state.IP + subpop_state.IP_V,
                                               axis=1, keepdims=True)
    wtd_IA = \
        subpop_params.IA_relative_inf * np.sum(subpop_state.IA + subpop_state.IA_V,
                                               axis=1, keepdims=True)

    return wtd_IP + wtd_IA


def compute_symp_infectious(subpop_state: FluSubpopState) -> np.ndarray:
    """
    Returns symptomatic infectious people (ISR and ISH) summed
        across the base and vaccinated tracks.

    Returns:
        np.ndarray of shape (A, R)
    """

    return subpop_state.ISR + subpop_state.ISH + \
        subpop_state.ISR_V + subpop_state.ISH_V


def compute_beta_adjusted(subpop_state: FluSubpopState,
                          subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Computes humidity-adjusted beta
    """

    return subpop_params.beta_baseline * (1 + subpop_params.humidity_impact *
                                          np.exp(-180 * subpop_state.absolute_humidity))


def compute_pop_by_age(subpop_params: FluSubpopParams) -> np.ndarray:
    """
    Returns:
        np.ndarray:
            A x 1 array -- where A is the number of age groups --
            where ith element corresponds to total population
            (across all compartments, including "D", and across all risk groups)
            in age group i
    """

    return np.sum(subpop_params.total_pop_age_risk, axis=1, keepdims=True)


def create_timeseries_df_from_day_of_week_schedule(
        day_of_week_schedule: pd.DataFrame,
        start_date: datetime.date) -> pd.DataFrame:
    """
    Creates a dataframe containing a timeseries of values
    for each date starting from start_date for 10 years.

    Parameters
    ----------
    day_of_week_schedule : pd.DataFrame
        Column day_of_week with values monday, tuesday, ...
        Second column has values for that day of week.
    start_date : datetime.date
        First day in timeseries.

    Returns
    -------
    pd.DataFrame
        Column date with all dates from start_date for 10 years.
        Second column has values for that date.
    """
    
    df_day_of_week = day_of_week_schedule.copy()
    
    # Create full timeseries dataframe by repeating day of week schedule
    duration_days = 10 * 365 # extend to 10 years to be safe
    new_dates = pd.date_range(start=start_date, periods=duration_days, freq='D')
    df = pd.DataFrame({'date': new_dates})
    
    df['day_of_week'] = df['date'].dt.day_name().str.lower()
    df = pd.merge(
        df, df_day_of_week, 
        on='day_of_week', how='left'
        ).drop(columns=['day_of_week'])
    
    df = df.set_index('date')
    
    return df


def parse_schedule_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the "date" column of a schedule DataFrame in place from
    "YYYY-MM-DD" strings (or `datetime.date` objects, left as dates)
    to `datetime.date` objects, and returns the DataFrame. DataFrames
    with a "day_of_week" column instead of dates are returned as-is.
    """

    try:
        if 'day_of_week' not in df.columns:
            df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d').dt.date
    except ValueError as e:
        raise ValueError("Error: dates should be strings in YYYY-MM-DD format or "
                         "`date.datetime` objects.") from e

    return df


class FluSubpopModel(clt.SubpopModel):
    """
    Class for creating ImmunoSEIRS flu model with predetermined fixed
    structure -- initial values and epidemiological structure are
    populated by user-specified `JSON` files.

    Key method create_transmission_model returns a `SubpopModel`
    instance with S-E-I-H-R-D compartments, a parallel vaccinated
    track of S_V-E_V-I_V-H_V-R_V-D_V compartments, and M and MV epi
    metrics (both switched off -- they always stay at zero, and
    `R_to_S_rate` is forced to zero).

    The update structure is as follows:
        - S <- S + R_to_S - S_to_E - S_to_S_V
        - E <- E + S_to_E - E_to_IP - E_to_IA
        - IA <- IA + E_to_IA - IA_to_R 
        - IP <- IP + E_to_IP - IP_to_ISR - IP_to_ISH
        - ISR <- ISR + IP_to_ISR - ISR_to_R
        - ISH <- ISH + IP_to_ISH - ISH_to_HR - ISH_to_HD
        - HR <- HR + ISH_to_HR - HR_to_R
        - HD <- HD + ISH_to_HD - HD_to_D
        - R <- R + ISR_to_R + HR_to_R - R_to_S
        - D <- D + HD_to_D

    The vaccinated track X_V has the same structure (with transition
    variables named X_V_to_Y_V), except that:
        - S_V <- S_V + S_to_S_V + R_V_to_S_V - S_V_to_E_V
        - S_V_to_E_V is a VaxSusceptibleToExposed instance -- relative
          susceptibility is multiplied by 1 - vax_induced_inf_risk_reduce
        - IP_V_to_ISR_V / IP_V_to_ISH_V are VaxPresympToSympRecover /
          VaxPresympToSympHospital instances -- the probability of
          hospitalization is multiplied by
          (1 - vax_induced_hosp_risk_reduce) / (1 - vax_induced_inf_risk_reduce)
        - ISH_V_to_HR_V / ISH_V_to_HD_V are VaxSympHospitalToHospRecover /
          VaxSympHospitalToHospDead instances -- the probability of
          death is multiplied by
          (1 - vax_induced_death_risk_reduce) / (1 - vax_induced_hosp_risk_reduce)
      These are conditional multipliers, so the overall reductions in
      hospitalization and death risk for a vaccinated person equal
      vax_induced_hosp_risk_reduce and vax_induced_death_risk_reduce --
      see `compute_vax_conditional_multipliers`.
    S_to_S_V is a ScheduledVaccination instance driven by the
    `daily_vaccines` schedule. On `vax_immunity_reset_date_mm_dd`,
    everyone in S_V moves back to S.

    The following are TransitionVariable instances:
        - R_to_S is a RecoveredToSusceptible instance
        - S_to_E is a SusceptibleToExposed instance
        - IP_to_ISR is a PresympToSympRecover instance
        - IP_to_ISH is a PresympToSympHospital instance
        - ISH_to_HR is a SympHospitalToHospRecover instance
        - ISH_to_HD is a SympHospitalToHospDead instance
        - ISR_to_R is a SympRecoverToRecovered instance
        - HR_to_R is a HospRecoverToRecovered instance 
        - HD_to_D is a HospDeadToDead instance

    There are six TransitionVariableGroups:
        - E_out (handles E_to_IP and E_to_IA)
        - IP_out (handles IP_to_ISR and IP_to_ISH)
        - ISH_out (handles ISH_to_HR and ISH_to_HD)
        - E_V_out, IP_V_out, ISH_V_out (vaccinated-track counterparts)

    The following are EpiMetric instances:
        - M is a InfInducedImmunity instance
        - MV is a VaxInducedImmunity instance

    Transition rates and update formulas are specified in
    corresponding classes.

    See parent class `SubpopModel`'s docstring for additional attributes.
    """

    def __init__(self,
                 state: FluSubpopState,
                 params: FluSubpopParams,
                 simulation_settings: FluSubpopSchedules,
                 RNG: np.random.Generator,
                 schedules_spec: FluSubpopSchedules,
                 name: str):
        """
        Args:
            state (FluSubpopState):
                holds current simulation state information,
                such as current values of epidemiological compartments
                and epi metrics.
            params (FluSubpopParams):
                holds epidemiological parameter values.
            simulation_settings (SimulationSettings):
                holds simulation settings.
            RNG (np.random.Generator):
                numpy random generator object used to obtain
                random numbers.
            schedules_spec (FluSubpopSchedules):
                holds dataframes that specify `Schedule` instances.
            name (str):
                unique name of MetapopModel instance.
        """

        self.schedules_spec = schedules_spec

        # IMPORTANT NOTE: as always, we must be careful with mutable objects
        # and generally use deep copies to avoid modification of the same
        # object. But in this function call, using deep copies is unnecessary
        # (redundant) because the parent class `SubpopModel`'s `__init__`
        # creates deep copies.
        super().__init__(state, params, simulation_settings, RNG, name)

        self.params = clt.updated_dataclass(self.params, {"start_real_date": self.start_real_date})

        # Infection-induced immunity is switched off in the
        #   vaccinated-track model -- recovered people stay in R
        if np.any(np.asarray(self.params.R_to_S_rate) != 0):
            warnings.warn(
                f"`R_to_S_rate` is {self.params.R_to_S_rate} but is forced to 0 "
                "in the vaccinated-track model (infection-induced immunity is "
                "switched off).")
        self.params = clt.updated_dataclass(self.params, {"R_to_S_rate": 0.0})

        self.update_vax_induced_risk_reduce_initial()
        self.update_infection_immunity_injection_val()

        # `InfInducedImmunity` and `VaxInducedImmunity` adjust their
        #   initial values in their constructors (decaying M(0) forward,
        #   deferring it, or adding pre-start vaccine doses to MV(0)),
        #   but `self.state` still holds the raw values read from the
        #   init-vals JSON. Sync so that anything reading `self.state`
        #   before the first simulated day sees the adjusted values --
        #   notably `get_flu_torch_inputs`, which builds the torch
        #   model's starting tensors straight off `self.state` and would
        #   otherwise start the torch run from different initial
        #   immunity than the numpy run.
        # Same for the compartments: "S" and "S_V" start from values
        #   shifted by pre-start vaccinations (see `create_compartments`),
        #   and the vaccinated-track fields are None when the init-vals
        #   JSON omits them.
        self.state.sync_to_current_vals(self.epi_metrics)
        self.state.sync_to_current_vals(self.compartments)

    def update_vax_induced_risk_reduce_initial(self) -> None:
        """
        Recomputes `vax_induced_inf_risk_reduce_initial`,
        `vax_induced_hosp_risk_reduce_initial`, and
        `vax_induced_death_risk_reduce_initial` from the current
        `daily_vaccines` schedule and updates `self.params` in place.

        This must be re-run (not just computed once at construction)
        whenever the `daily_vaccines` schedule or any of the underlying
        base parameters (`vax_induced_*_risk_reduce`,
        `vax_induced_immune_wane`) change after construction -- e.g. via
        `replace_schedule` or a `ScenarioRunner` parameter override --
        otherwise these derived values would silently keep reflecting
        the schedule/params from construction time. See
        `reset_simulation`, which calls this for the same reason
        `VaxInducedImmunity`'s initial value is recomputed there.
        """

        _warn_vax_conditional_clipping(self.params)

        inf_initial, hosp_initial, death_initial = compute_vax_induced_risk_reduce_initial(
            self.params, self.schedules, self.start_real_date)
        self.params = clt.updated_dataclass(self.params, {
            "vax_induced_inf_risk_reduce_initial": inf_initial,
            "vax_induced_hosp_risk_reduce_initial": hosp_initial,
            "vax_induced_death_risk_reduce_initial": death_initial,
        })

    def update_infection_immunity_injection_val(self) -> None:
        """
        Mirrors the `InfInducedImmunity` epi metric's pending injection
        onto `self.params.infection_immunity_injection_val` -- the
        amount to add to M when
        `infection_immunity_start_date_mm_dd` is reached, or zeros if
        no injection is pending.

        The numpy model applies the injection straight off the epi
        metric (`InfInducedImmunity.check_and_apply_injection`) and does
        not need this. The torch metapopulation model has no epi metric
        objects -- it only sees `params` and state tensors -- so the
        value has to travel on `params` for
        `check_and_apply_M_injection` to be able to apply it there.

        Like `update_vax_induced_risk_reduce_initial`, this is re-run on
        `reset_simulation` so it tracks any post-construction change to
        `infection_immunity_start_date_mm_dd` or M's initial value.
        """

        M = self.epi_metrics["M"]

        if M.pending_injection_date is not None:
            injection_val = np.asarray(M.original_init_val, dtype=float).copy()
        else:
            injection_val = np.zeros((self.params.num_age_groups,
                                      self.params.num_risk_groups))

        self.params = clt.updated_dataclass(
            self.params, {"infection_immunity_injection_val": injection_val})

    def check_humidity_input(self) -> None:
        """
        Check that absolute humidity values are non-negative.
        """

        humidity_values = self.schedules['absolute_humidity'].timeseries_df['absolute_humidity'].values
        if np.any(humidity_values < 0):
            raise FluSubpopModelError("Error: absolute humidity values must be non-negative.")
    
    def check_vaccination_input(self) -> None:
        """
        Check that vaccination values are positive.
        If vaccinations exceed 100% over a year, issue a warning.
        """
        
        df_vaccine = self.schedules['daily_vaccines'].timeseries_df.copy()
        
        ## Check all entries are positive
        all_positive = all([
            (x >= 0).all() for x in df_vaccine['daily_vaccines'].values
            ])
        if not(all_positive):
            raise FluSubpopModelError("Error: vaccination values must be non-negative.")
        
        ## Check cumulative vaccination never exceeds 100% over 365 days
        df_vaccine['datetime'] = pd.to_datetime(df_vaccine.index)
        df_vaccine.set_index('datetime', inplace=True)

        # Ensure there is one row per day within time range
        full_date_range = pd.date_range(start=df_vaccine.index.min(), end=df_vaccine.index.max(), freq='D')
        df_vaccine = df_vaccine.reindex(full_date_range)

        # Replace NAs due to reindexing (if any)
        na_rows = df_vaccine['daily_vaccines'].isna()
        value_shape = df_vaccine['daily_vaccines'].values[0].shape
        df_vaccine.loc[na_rows, 'daily_vaccines'] = \
            pd.Series(
                [np.zeros(value_shape)] * na_rows.sum(), 
                index=df_vaccine.loc[na_rows].index
                )

        # Calculate rolling 1-year sum of vaccination rates
        window_size_days = min(365, len(df_vaccine))

        data_windows = np.lib.stride_tricks.sliding_window_view(
            df_vaccine['daily_vaccines'].values, 
            window_size_days
            )
        vaccines_rolling_sum = np.sum(data_windows, axis=-1)

        # Check whether any cumulative vaccinations exceed 100%
        max_values_above_one = [x.max() > 1 for x in vaccines_rolling_sum]

        # Find first index rolling sum exceeds 100% for some age group (if there is one)
        if sum(max_values_above_one) > 0:
            first_exceeds_idx = max_values_above_one.index(True)
            
            vaccines_cml_exceeds = vaccines_rolling_sum[first_exceeds_idx]
            exceeds_first_date = df_vaccine.index[first_exceeds_idx + window_size_days - 1]
            
            msg = 'Cumulative vaccination over a 365-day period exceeds 100% on (at least) ' +\
                f'the following date: {exceeds_first_date}. Cumulative vaccination by that date is \n' +\
                str(vaccines_cml_exceeds)
            warnings.warn(msg)
        
    def check_calendar_variables_input(self) -> None:
        """
        Check school and calendar variables in flu_contact_matrix
        schedule are between 0 and 1
        """
        
        flu_contact_matrix = self.schedules['flu_contact_matrix'].timeseries_df
        
        for variable in ['is_school_day', 'is_work_day']:
            values = flu_contact_matrix[variable].values
            
            if (values.min() < 0) or (values.max() > 1):
                msg = f'Error: {variable} values must be between 0 and 1.'
                raise FluSubpopModelError(msg)
    
    def check_contact_matrix_input(self) -> None:
        """
        Check contact matrix entries are non-negative.
        Check total contact is greater than the sum of the
        school and work matrices.
        """   
        
        if not(np.all(self.params.total_contact_matrix >= 0)):
            raise FluSubpopModelError(
                'Some entries of the total contact matrix are negative \n'+\
                f'{self.params.total_contact_matrix}'
                )
        
        if not(np.all(self.params.school_contact_matrix >= 0)):
            raise FluSubpopModelError(
                'Some entries of the school contact matrix are negative \n'+\
                f'{self.params.school_contact_matrix}'
                )
        
        if not(np.all(self.params.work_contact_matrix >= 0)):
            raise FluSubpopModelError(
                'Some entries of the work contact matrix are negative \n'+\
                f'{self.params.work_contact_matrix}'
                )
        
        if not(np.all((
            self.params.total_contact_matrix - self.params.school_contact_matrix - 
            self.params.work_contact_matrix) >= 0)):
            raise FluSubpopModelError(
                'The total contact matrix must be at least greater than the sum of ' +\
                'the work and school contact matrices.'
            )
    
    def check_rate_input(self) -> None:
        """
        Ensure all rate values are strictly positive, and other
        variables (waning, saturation, reductions) are non-negative.
        """
        
        p = self.params
        # `R_to_S_rate` is not checked -- it is forced to 0 (see `__init__`)
        rates_list = [
            p.E_to_I_rate, p.IP_to_IS_rate, p.ISH_to_H_rate,
            p.ISR_to_R_rate, p.IA_to_R_rate, p.HR_to_R_rate, p.HD_to_D_rate,
            p.E_to_IA_prop]
        
        other_params_list = [
            p.humidity_impact, p.inf_induced_saturation, p.inf_induced_immune_wane,
            p.vax_induced_immune_wane, p.inf_induced_inf_risk_reduce,
            p.inf_induced_hosp_risk_reduce, p.inf_induced_death_risk_reduce, 
            p.vax_induced_inf_risk_reduce, p.vax_induced_hosp_risk_reduce,
            p.vax_induced_death_risk_reduce, p.IP_relative_inf,
            p.IA_relative_inf, p.relative_suscept,
            p.ISH_to_HD_prop, p.IP_to_ISH_prop, p.beta_baseline
        ]
        
        for value in rates_list:
            if not(np.all(value >= 0)):
                raise FluSubpopModelError('All transition rates must be positive values.')
            if not(np.all(value > 0)):
                msg = 'Some transition rates are equal to zero.'
                warnings.warn(msg)
        
        for value in other_params_list:
            if not(np.all(value >= 0)):
                raise FluSubpopModelError('Some parameter values are negative.')
        
    def check_initial_compartment_input(self) -> None:
        """
        Ensure all initial compartment and saturation values are non-negative.
        """
        
        # Read compartments off the `Compartment` objects rather than
        #   `self.state` -- vaccinated-track fields on `self.state` are
        #   None until synced when the init-vals JSON omits them
        values = {name: self.compartments[name].current_val for name in ALL_COMPARTMENTS}
        values.update({name: self.epi_metrics[name].current_val for name in ("M", "MV")})

        for state_name, value in values.items():
            if not(np.all(np.asarray(value) >= 0)):
                raise FluSubpopModelError(
                    'Initial compartment and immunity values must be non-negative. ' +\
                    f'{state_name} is negative: {value} for subpopulation ' +\
                    f'{self.name}.'
                )

    def check_vax_dose_pool_input(self) -> None:
        """
        Ensure `vax_dose_pool` is one of the supported options.
        """

        if self.params.vax_dose_pool not in VAX_DOSE_POOLS:
            raise FluSubpopModelError(
                f"`vax_dose_pool` must be one of {VAX_DOSE_POOLS} -- got "
                f"{self.params.vax_dose_pool!r}.")
    
    def run_input_checks(self) -> None:
        """
        Check the following:
            - if total vaccinations exceed 100% over a year we issue a warning
            - school and work calendar variables must be between 0 and 1
            - absolute humidity values, contact matrix entries, daily vaccination
              must be non-negative
            - total contact matrix must be greater than the sum of the school
              and work contact matrices
            - all rate values must be strictly positive
            - initial compartmental values must be non-negative
            - `vax_dose_pool` must be a supported option
        """

        self.check_humidity_input()
        self.check_vaccination_input()
        self.check_calendar_variables_input()
        self.check_contact_matrix_input()
        self.check_rate_input()
        self.check_initial_compartment_input()
        self.check_vax_dose_pool_input()

    def prepare_daily_state(self) -> None:
        """
        Override parent method to add the vaccinated-track updates.
        At beginning of each day, update schedules and dynamic values,
        move everyone in "S_V" back to "S" if today is the vaccine
        immunity reset date, and set the day's expected number of
        vaccinations on `S_to_S_V`.
        """
        # Call parent implementation first to update schedules and dynamic vals
        super().prepare_daily_state()

        self.check_and_apply_vax_track_reset()

        # The day's vaccinations come from the pool after the reset, which
        #   moves people between S and S_V -- matches `prepare_daily_torch_state`
        self.transition_variables.S_to_S_V.set_daily_expected(self.state, self.params)

    def check_and_apply_vax_track_reset(self) -> None:
        """
        If the current date matches `vax_immunity_reset_date_mm_dd`
        (month and day, so this repeats every year), move everyone in
        "S_V" back to "S". People elsewhere on the vaccinated track
        stay there.
        """

        if self.params.vax_immunity_reset_date_mm_dd is None:
            return

        month, day = self.params.vax_immunity_reset_date_mm_dd.split('_')

        if self.current_real_date.month != int(month) or \
                self.current_real_date.day != int(day):
            return

        S = self.compartments.S
        S_V = self.compartments.S_V

        S.current_val = np.asarray(S.current_val, dtype=float) + \
            np.asarray(S_V.current_val, dtype=float)
        S_V.current_val = np.zeros_like(np.asarray(S_V.current_val, dtype=float))

        print(f"Vaccinated track reset: S_V moved back to S on {self.current_real_date}")

        # Sync `self.state` immediately so that today's first timestep
        #   (which reads `state.S`/`state.S_V`) sees the reset values
        self.state.sync_to_current_vals(self.compartments)

    def create_compartments(self) -> sc.objdict[str, clt.Compartment]:

        # Create `Compartment` instances S-E-IA-IP-IS-H-R-D and their
        #   vaccinated-track copies (20 compartments total)
        # Save instances in `sc.objdict` and return objdict

        A = self.params.num_age_groups
        R = self.params.num_risk_groups

        init_vals = {}
        for name in ALL_COMPARTMENTS:
            val = getattr(self.state, name)
            init_vals[name] = np.zeros((A, R)) if val is None else val

        # Keep the input values so the pre-start shift can be recomputed
        #   from scratch on `reset_simulation` (see there)
        self._original_S_init = copy.deepcopy(init_vals["S"])
        self._original_S_V_init = copy.deepcopy(init_vals["S_V"])

        shift = self.compute_pre_start_vaccination_shift(
            sum(np.asarray(v, dtype=float) for v in init_vals.values()))
        init_vals["S"] = np.asarray(self._original_S_init, dtype=float) - shift
        init_vals["S_V"] = np.asarray(self._original_S_V_init, dtype=float) + shift

        compartments = sc.objdict()

        for name in ALL_COMPARTMENTS:
            compartments[name] = clt.Compartment(init_vals[name])

        return compartments

    def compute_pre_start_vaccination_shift(self,
                                            total_pop_age_risk: np.ndarray) -> np.ndarray:
        """
        Returns the number of people to move from the input "S" to
        the input "S_V" to account for vaccinations scheduled before
        the simulation start date -- see
        `compute_pre_start_vaccination_shift`.
        """

        return compute_pre_start_vaccination_shift(
            self._original_S_init,
            self._original_S_V_init,
            total_pop_age_risk,
            self.start_real_date,
            self.params,
            self.schedules,
            self.simulation_settings.timesteps_per_day,
            "no_round" not in self.simulation_settings.transition_type)

    def create_dynamic_vals(self) -> sc.objdict[str, clt.DynamicVal]:
        """
        Create all `DynamicVal` instances, save in `sc.objdict`, and return objdict
        """

        dynamic_vals = sc.objdict()

        dynamic_vals["beta_reduce"] = BetaReduce(init_val=0.0,
                                                 is_enabled=False)

        return dynamic_vals

    def create_schedules(self) -> sc.objdict[str, clt.Schedule]:
        """
        Create all `Schedule` instances, save in `sc.objdict`, and return objdict
        """

        schedules = sc.objdict()

        schedules["absolute_humidity"] = AbsoluteHumidity()
        schedules["flu_contact_matrix"] = FluContactMatrix()
        schedules["daily_vaccines"] = DailyVaccines(
            vax_protection_delay_days=self.params.vax_protection_delay_days
        )
        schedules["mobility_modifier"] = MobilityModifier()

        for field, df in asdict(self.schedules_spec).items():

            schedules[field].timeseries_df = parse_schedule_dates(df)
            schedules[field].postprocess_data_input()

        return schedules

    def replace_schedule(self,
                         schedule_name: str,
                         new_df: pd.DataFrame) -> None:
        """
        Extends the base `replace_schedule` to parse "date" strings
        into `datetime.date` objects first, as `create_schedules` does
        -- the `Schedule` lookups index by `datetime.date`, and
        `DailyVaccines` shifts the dates by its protection delay.
        """

        # Leave an unknown `schedule_name` to the base method, which
        #   raises a clear error for it
        if schedule_name in self.schedules:
            new_df = parse_schedule_dates(new_df.copy())

        super().replace_schedule(schedule_name, new_df)

    def create_transition_variables(self) -> sc.objdict[str, clt.TransitionVariable]:
        """
        Create all `TransitionVariable` instances,
        save in `sc.objdict`, and return objdict
        """

        # NOTE: see the parent class `SubpopModel`'s `__init__` --
        # `create_transition_variables` is called after
        # `simulation_settings` is assigned

        transition_type = self.simulation_settings.transition_type

        transition_variables = sc.objdict()

        S = self.compartments.S
        E = self.compartments.E
        IP = self.compartments.IP
        ISR = self.compartments.ISR
        ISH = self.compartments.ISH
        IA = self.compartments.IA
        HR = self.compartments.HR
        HD = self.compartments.HD
        R = self.compartments.R
        D = self.compartments.D

        transition_variables.R_to_S = RecoveredToSusceptible(R, S, transition_type)
        transition_variables.S_to_E = SusceptibleToExposed(S, E, transition_type)
        transition_variables.IP_to_ISR = PresympToSympRecover(IP, ISR, transition_type, True)
        transition_variables.IP_to_ISH = PresympToSympHospital(IP, ISH, transition_type, True)
        transition_variables.IA_to_R = AsympToRecovered(IA, R, transition_type)
        transition_variables.E_to_IP = ExposedToPresymp(E, IP, transition_type, True)
        transition_variables.E_to_IA = ExposedToAsymp(E, IA, transition_type, True)
        transition_variables.ISR_to_R = SympRecoverToRecovered(ISR, R, transition_type)
        transition_variables.ISH_to_HR = SympHospitalToHospRecover(ISH, HR, transition_type, True)
        transition_variables.ISH_to_HD = SympHospitalToHospDead(ISH, HD, transition_type, True)
        transition_variables.HR_to_R = HospRecoverToRecovered(HR, R, transition_type)
        transition_variables.HD_to_D = HospDeadToDead(HD, D, transition_type)

        # Entry into the vaccinated track -- must come AFTER S_to_E:
        #   both are marginally distributed, so `sample_transitions`
        #   realizes them in this order, and S_to_S_V caps itself at
        #   what S_to_E leaves in S
        transition_variables.S_to_S_V = ScheduledVaccination(S, self.compartments.S_V,
                                                             transition_variables.S_to_E,
                                                             transition_type)

        # Vaccinated track -- same structure, except for S_V_to_E_V
        #   and the IP_V and ISH_V splits
        S_V = self.compartments.S_V
        E_V = self.compartments.E_V
        IP_V = self.compartments.IP_V
        ISR_V = self.compartments.ISR_V
        ISH_V = self.compartments.ISH_V
        IA_V = self.compartments.IA_V
        HR_V = self.compartments.HR_V
        HD_V = self.compartments.HD_V
        R_V = self.compartments.R_V
        D_V = self.compartments.D_V

        transition_variables.R_V_to_S_V = RecoveredToSusceptible(R_V, S_V, transition_type)
        transition_variables.S_V_to_E_V = VaxSusceptibleToExposed(S_V, E_V, transition_type)
        transition_variables.IP_V_to_ISR_V = VaxPresympToSympRecover(IP_V, ISR_V, transition_type, True)
        transition_variables.IP_V_to_ISH_V = VaxPresympToSympHospital(IP_V, ISH_V, transition_type, True)
        transition_variables.IA_V_to_R_V = AsympToRecovered(IA_V, R_V, transition_type)
        transition_variables.E_V_to_IP_V = ExposedToPresymp(E_V, IP_V, transition_type, True)
        transition_variables.E_V_to_IA_V = ExposedToAsymp(E_V, IA_V, transition_type, True)
        transition_variables.ISR_V_to_R_V = SympRecoverToRecovered(ISR_V, R_V, transition_type)
        transition_variables.ISH_V_to_HR_V = VaxSympHospitalToHospRecover(ISH_V, HR_V, transition_type, True)
        transition_variables.ISH_V_to_HD_V = VaxSympHospitalToHospDead(ISH_V, HD_V, transition_type, True)
        transition_variables.HR_V_to_R_V = HospRecoverToRecovered(HR_V, R_V, transition_type)
        transition_variables.HD_V_to_D_V = HospDeadToDead(HD_V, D_V, transition_type)

        return transition_variables

    def create_transition_variable_groups(self) -> sc.objdict[str, clt.TransitionVariableGroup]:
        """
        Create all transition variable groups described in docstring (6 transition
        variable groups total), save in `sc.objdict`, return objdict
        """

        # Shortcuts for attribute access
        # NOTE: see the parent class `SubpopModel`'s `__init__` --
        # `create_transition_variable_groups` is called after
        # `simulation_settings` is assigned

        transition_type = self.simulation_settings.transition_type

        transition_variable_groups = sc.objdict()

        transition_variable_groups.E_out = clt.TransitionVariableGroup(self.compartments.E,
                                                                       transition_type,
                                                                       (self.transition_variables.E_to_IP,
                                                                        self.transition_variables.E_to_IA))

        transition_variable_groups.IP_out = clt.TransitionVariableGroup(self.compartments.IP,
                                                                        transition_type,
                                                                        (self.transition_variables.IP_to_ISR,
                                                                         self.transition_variables.IP_to_ISH))

        transition_variable_groups.ISH_out = clt.TransitionVariableGroup(self.compartments.ISH,
                                                                         transition_type,
                                                                         (self.transition_variables.ISH_to_HR,
                                                                          self.transition_variables.ISH_to_HD))

        transition_variable_groups.E_V_out = clt.TransitionVariableGroup(self.compartments.E_V,
                                                                         transition_type,
                                                                         (self.transition_variables.E_V_to_IP_V,
                                                                          self.transition_variables.E_V_to_IA_V))

        transition_variable_groups.IP_V_out = clt.TransitionVariableGroup(self.compartments.IP_V,
                                                                          transition_type,
                                                                          (self.transition_variables.IP_V_to_ISR_V,
                                                                           self.transition_variables.IP_V_to_ISH_V))

        transition_variable_groups.ISH_V_out = clt.TransitionVariableGroup(self.compartments.ISH_V,
                                                                           transition_type,
                                                                           (self.transition_variables.ISH_V_to_HR_V,
                                                                            self.transition_variables.ISH_V_to_HD_V))

        return transition_variable_groups

    def create_epi_metrics(self) -> sc.objdict[str, clt.EpiMetric]:
        """
        Create all epi metric described in docstring (2 state
        variables total), save in `sc.objdict`, and return objdict
        """

        epi_metrics = sc.objdict()

        epi_metrics.M = \
            InfInducedImmunity(getattr(self.state, "M"),
                               self.transition_variables.R_to_S,
                               self.current_real_date,
                               self.params,
                               self.simulation_settings.timesteps_per_day)

        epi_metrics.MV = \
            VaxInducedImmunity(getattr(self.state, "MV"),
                               self.current_real_date,
                               self.params,
                               self.schedules,
                               self.simulation_settings.timesteps_per_day)

        return epi_metrics

    def modify_subpop_params(self,
                             updates_dict: dict):
        """
        This method lets users safely modify a single subpopulation
        parameters field; if this subpop model is associated with
        a metapop model, the metapopulation-wide tensors are updated
        automatically afterward. See also `modify_subpop_params` method on
        `FluMetapopModel`.

        Parameters:
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`.
        """

        # If associated with metapop model, run this method
        #   on the metapop model itself to handle metapopulation-wide
        #   tensor updating
        if self.metapop_model:
            self.metapop_model.modify_subpop_params(self.name,
                                                    updates_dict)
        else:
            # Since `SubpopParams` is frozen, we return a new instance
            #   with the reflected updates
            self.params = clt.updated_dataclass(self.params, updates_dict)

    def reset_simulation(self) -> None:
        """
        Extends the base `reset_simulation` to recompute the initial
        "S" and "S_V" values and `vax_induced_*_risk_reduce_initial`
        from the currently loaded vaccine schedule (and current base
        params) before resetting.

        This ensures that if the `daily_vaccines` schedule has been
        replaced (e.g. via `replace_schedule`), or
        `vax_immunity_reset_date_mm_dd`/`vax_dose_pool` have been
        overridden (e.g. by `ScenarioRunner`), the model resets to
        values consistent with the current schedule/params, rather
        than the values computed at construction time.

        The pre-start shift is recomputed from `_original_S_init` and
        `_original_S_V_init` -- the unmodified values from the state
        JSON -- so shifts do not compound across calls. M and MV are
        switched off, so their initial values are always zero.
        """

        shift = self.compute_pre_start_vaccination_shift(self.params.total_pop_age_risk)

        # Use the init_val setter so current_val is also updated immediately,
        # before super()'s reset loop overwrites it again (harmlessly).
        self.compartments.S.init_val = np.asarray(self._original_S_init, dtype=float) - shift
        self.compartments.S_V.init_val = np.asarray(self._original_S_V_init, dtype=float) + shift

        self.update_vax_induced_risk_reduce_initial()
        self.update_infection_immunity_injection_val()

        super().reset_simulation()


class FluMetapopModel(clt.MetapopModel, ABC):
    """
    MetapopModel-derived class specific to flu model.
    """

    def __init__(self,
                 subpop_models: list[dict],
                 mixing_params: FluMixingParams,
                 name: str = ""):

        super().__init__(subpop_models,
                         mixing_params,
                         name)

        # Confirm validity and consistency of `FluMixingParams`
        try:
            num_locations = mixing_params.num_locations
        except KeyError:
            raise FluMetapopModelError("'mixing_params' must contain the key 'num_locations'. \n"
                                       "Please specify it before continuing.")
        if num_locations != len(subpop_models):
            raise FluMetapopModelError("'num_locations' should equal the number of items in \n"
                                       "'subpop_models'. Please amend before continuing.")

        self.travel_state_tensors = FluTravelStateTensors()
        self.update_travel_state_tensors()

        # `FluMixingParams` info is stored on `FluTravelParamsTensors` --
        # this order of operations below is important, because
        # `mixing_params` attribute must be defined before `update_travel_params_tensors()`
        # is called.
        self.mixing_params = mixing_params
        self.travel_params_tensors = FluTravelParamsTensors()
        self.update_travel_params_tensors()

        total_pop_LAR_tensor = self.compute_total_pop_LAR_tensor()

        self.precomputed = FluPrecomputedTensors(total_pop_LAR_tensor,
                                                 self.travel_params_tensors)

        # Generally not used unless using torch version
        self._full_metapop_params_tensors = None
        self._full_metapop_state_tensors = None
        self._full_metapop_schedule_tensors = None
    
    def check_mobility_input(self) -> None:
        """
        Check that all elements of the mobility matrix are positive,
        and that rows sum to 1.
        """
        
        travel_proportions = self.mixing_params.travel_proportions
        if np.any(travel_proportions < 0):
            raise FluSubpopModelError(
                f'All entries of the travel matrix must be non-negative:\n {travel_proportions}.')
        
        if not(np.allclose(travel_proportions.sum(axis=1), 1)):
            raise FluSubpopModelError(
                f'Rows of the travel matrix must all sum to 1:\n {travel_proportions}.')

    def run_input_checks(self) -> None:
        """
        Check the following:
            - rows of mobility matrix must sum to 1
            - mobility matrix entries are between 0 and 1
        """
        
        # Mobility matrix check
        self.check_mobility_input()

    def modify_subpop_params(self,
                             subpop_name: str,
                             updates_dict: dict):
        """
        This method lets users safely modify a single subpopulation
        parameters field; the metapopulation-wide tensors are updated
        automatically afterward.

        In a `FluMetapopModel`, subpopulation parameters are combined into
        (L, A, R) tensors across L subpopulations.`FluSubpopParams` is a frozen
        dataclass to avoid users naively changing parameter values and getting
        undesirable results -- thus, `FluSubpopParams` on a subpopulation
        model cannot be updated directly.

        Parameters:
            subpop_name (str):
               Value must match the `name` attribute of one of the
               `FluSubpopModel` instances contained in this metapopulation
                model's `subpop_models` attribute.
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`.
        """

        # Since `FluSubpopParams` is frozen, we return a new instance
        #   with the reflected updates
        self.subpop_models[subpop_name].params = clt.updated_dataclass(
            self.subpop_models[subpop_name].params, updates_dict
        )

        self.update_travel_params_tensors()

        # Adding this for extra safety in case the user does not
        # call `get_flu_torch_inputs` for accessing the
        # `FullMetapopParams` instance.

        # If this attribute is not `None`, it means we are using
        # the `torch` implementation, and we should update the
        # corresponding `FullMetapopParams` instance with the new
        # `FluMixingParams` values.
        if self._full_metapop_params_tensors:
            self.update_full_metapop_params_tensors()

    def modify_mixing_params(self,
                             updates_dict: dict):
        """
        This method lets users safely modify flu mixing parameters;
        the metapopulation-wide tensors are updated automatically afterward.
        `FluMixingParams` is a frozen dataclass to avoid users
        naively changing parameter values and getting undesirable results --
        thus, `FluMixingParams` cannot be updated directly.

        Parameters:
            updates_dict (dict):
                Dictionary specifying values to update in a
                `FluSubpopParams` instance -- keys must match the
                field names of `FluSubpopParams`. 
        """

        self.mixing_params = clt.updated_dataclass(self.mixing_params, updates_dict)
        self.update_travel_params_tensors()

        nonlocal_travel_prop = self.travel_params_tensors.travel_proportions.clone().fill_diagonal_(0.0)

        self.precomputed.sum_residents_nonlocal_travel_prop = nonlocal_travel_prop.sum(dim=1)

        # Adding this for extra safety in case the user does not
        # call `get_flu_torch_inputs` for accessing the
        # `FullMetapopParams` instance.

        # If this attribute is not `None`, it means we are using
        # the `torch` implementation, and we should update the
        # corresponding `FullMetapopParams` instance with the new
        # `FluMixingParams` values.
        if self._full_metapop_params_tensors:
            self.update_full_metapop_params_tensors()

    def compute_total_pop_LAR_tensor(self) -> torch.tensor:
        """
        For each subpopulation, sum initial values of population
        in each compartment for age-risk groups. Store all information
        as tensor and return tensor.

        Returns:
        --------
        torch.tensor of size (L, A, R):
            Total population (across all compartments) for
            location-age-risk (l, a, r).
        """

        # ORDER MATTERS! USE ORDERED DICTIONARY HERE
        #   to preserve correct index order in tensors!
        #   See `update_travel_params_tensors` for detailed note.
        subpop_models_ordered = self._subpop_models_ordered

        total_pop_LAR_tensor = torch.zeros(self.travel_params_tensors.num_locations,
                                           self.travel_params_tensors.num_age_groups,
                                           self.travel_params_tensors.num_risk_groups)

        # All subpop models should have the same compartments' keys
        for name in subpop_models_ordered[0].compartments.keys():

            metapop_vals = []

            for model in subpop_models_ordered.values():
                compartment = getattr(model.compartments, name)
                metapop_vals.append(compartment.current_val)

            total_pop_LAR_tensor = total_pop_LAR_tensor + torch.tensor(np.asarray(metapop_vals))

        return total_pop_LAR_tensor

    def update_state_tensors(self,
                             target: FluTravelStateTensors) -> None:
        """
        Update `target` instance in-place with current simulation
        values. Each field of `target` corresponds to a field in
        `FluSubpopState`, and contains either a tensor of size
        (L, A, R) or a tensor of size (L), where (l, a, r) refers to
        location-age-risk.
        """

        # ORDER MATTERS! USE ORDERED DICTIONARY HERE
        #   to preserve correct index order in tensors!
        #   See `update_travel_params_tensors` for detailed note.
        subpop_models_ordered = self._subpop_models_ordered

        for field in fields(target):

            name = field.name

            # FluTravelStateTensors has an attribute
            #   that is a dictionary called `init_vals` --
            #   disregard, as this only used to store
            #   initial values for resetting, but is not
            #   used in the travel model computation
            if name == "init_vals":
                continue

            metapop_vals = []

            for model in subpop_models_ordered.values():
                current_val = getattr(model.state, name)
                metapop_vals.append(current_val)

            # Probably want to update this to be cleaner...
            # `SubpopState` fields that correspond to `Schedule` instances
            # have initial values of `None` -- but we cannot build a tensor
            # with `None` values, so we convert values to 0s.
            if any(v is None for v in metapop_vals):
                setattr(target, name, torch.tensor(np.full(np.shape(metapop_vals), 0.0)))
            else:
                setattr(target, name, torch.tensor(np.asarray(metapop_vals)))

            # Only fields corresponding to `Schedule` instances can be
            # size (L) -- this is because the schedule value may be scalar for
            # each subpopulation. Other fields should all be size (L, A, R). 

    def update_travel_state_tensors(self) -> None:
        """
        Update `travel_state_tensors` attribute in-place.
        `FluTravelStateTensors` only has fields corresponding
        to state variables relevant for the travel model.
        Converts subpopulation-specific state to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        self.update_state_tensors(self.travel_state_tensors)

    def update_full_metapop_state_tensors(self) -> None:
        """
        Update `_full_metapop_state_tensors` attribute in-place.
        `FluFullMetapopStateTensors` has fields corresponding
        to all state variables in the simulation.
        Converts subpopulation-specific state to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        if self._full_metapop_state_tensors is None:
            self._full_metapop_state_tensors = FluFullMetapopStateTensors()
        self.update_state_tensors(self._full_metapop_state_tensors)

    def update_params_tensors(self,
                              target: FluTravelParamsTensors) -> FluTravelParamsTensors:
        """
        Update `target` in-place. Converts subpopulation-specific
        parameters to tensors of size (L, A, R) for location-age-risk,
        except for `num_locations` and `travel_proportions`, which
        have size 1 and (L, L) respectively.
        """

        # USE THE ORDERED DICTIONARY HERE FOR SAFETY!
        #   AGAIN, ORDER MATTERS BECAUSE ORDER DETERMINES
        #   THE SUBPOPULATION INDEX IN THE METAPOPULATION
        #   TENSOR!
        subpop_models_ordered = self._subpop_models_ordered

        # Subpop models should have the same A, R so grab
        #   from the first subpop model
        A = subpop_models_ordered[0].params.num_age_groups
        R = subpop_models_ordered[0].params.num_risk_groups

        for field in fields(target):

            name = field.name
            is_non_numerical = False

            metapop_vals = []

            if name == "num_locations" or name == "travel_proportions":
                setattr(target, name, torch.tensor(getattr(self.mixing_params, name)))

            else:

                for model in subpop_models_ordered.values():
                    metapop_vals.append(getattr(model.params, name))
                
                # If all values are equal to each other, then
                #   simply store the first value (since its value is common
                #   across metapopulations)
                first_val = metapop_vals[0]
                if first_val is None or isinstance(first_val, str) or isinstance(first_val, datetime.date):
                    is_non_numerical = True
                    if all(x == first_val for x in metapop_vals):
                        metapop_vals = first_val
                    else:
                        raise FluMetapopModelError(
                            f"Error: non-numerical parameter '{name}' has values that differ "
                            "across subpopulations; values should be the same."
                        )
                else:
                    if all(np.allclose(x, first_val) for x in metapop_vals):
                        metapop_vals = first_val

                # Converting list of arrays to tensors is slow --
                #   better to convert to array first
                if isinstance(metapop_vals, list):
                    metapop_vals = np.asarray(metapop_vals)
                    # metapop_vals = np.stack([clt.to_AR_array(x, A, R) for x in metapop_vals])

                if is_non_numerical:
                    setattr(target, name, metapop_vals)
                else:
                    setattr(target, name, torch.tensor(metapop_vals))

        # Convert all tensors to correct size!
        target.standardize_shapes()

    def update_travel_params_tensors(self) -> None:
        """
        Update `travel_params_tensors` attribute in-place.
        `FluTravelParamsTensors` only has fields corresponding
        to parameters relevant for the travel model.
        Converts subpopulation-specific parameters to
        tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        self.update_params_tensors(target=self.travel_params_tensors)

    def update_full_metapop_params_tensors(self) -> None:
        """
        Update `_full_metapop_params_tensors` attribute in-place.
        `FluFullMetapopParamsTensors` has fields corresponding
        to all parameters in the simulation. Converts subpopulation-specific
        parameters to tensors of size (L, A, R) for location-age-risk
        (except for a few exceptions that have different dimensions).
        """

        if self._full_metapop_params_tensors is None:
            self._full_metapop_params_tensors = FluFullMetapopParamsTensors()
        self.update_params_tensors(target=self._full_metapop_params_tensors)

    def apply_inter_subpop_updates(self) -> None:
        """
        Update the `FluTravelStateTensors` according to the simulation state
        and compute the total mixing exposure, which includes across-subpopulation
        mixing/travel. Update the `total_mixing_exposure` attribute on each
        subpopulation's `SusceptibleToExposed` instance accordingly, so each
        of these transition variables can compute its transition rate.

        See `apply_inter_subpop_updates` on `MetapopModel` base class
        for logic of how/when this is called in the simulation.
        """

        self.update_travel_state_tensors()

        total_mixing_exposure = compute_total_mixing_exposure(self.travel_state_tensors,
                                                              self.travel_params_tensors,
                                                              self.precomputed)

        # Again, `self.subpop_models` is an ordered dictionary --
        #   so iterating over the dictionary like this is well-defined
        #   and responsible -- the order is important because it
        #   determines the order (index) in any metapopulation tensors
        subpop_models = self._subpop_models_ordered

        # Updates `total_mixing_exposure` attribute on each `SusceptibleToExposed`
        # instance -- this value captures across-population travel/mixing.
        # The vaccinated track's S_V_to_E_V uses the same exposure --
        #   its reduced susceptibility is applied in its own rate
        for i in range(len(subpop_models)):
            subpop_tvars = subpop_models.values()[i].transition_variables
            subpop_tvars.S_to_E.total_mixing_exposure = total_mixing_exposure[i, :, :]
            subpop_tvars.S_V_to_E_V.total_mixing_exposure = total_mixing_exposure[i, :, :]

    def setup_full_metapop_schedule_tensors(self):
        """
        Creates `FluFullMetapopScheduleTensors` instance and assigns to
        `_full_metapop_schedule_tensors` attribute.

        For the metapopulation model's L locations/subpopulations, for each day,
        each value-related column in each schedule is either a float or
        array of size (A, R) for age-risk groups.

        We aggregate and reformat this schedule information and put it
        into a `FluFullMetapopScheduleTensors` instance, where fields
        correspond to a schedule value, and values are lists of tensors of
        size (L, A, R). The ith element of each list corresponds to the
        ith simulation day.
        """

        self._full_metapop_schedule_tensors = FluFullMetapopScheduleTensors()

        L = self.precomputed.L
        A = self.precomputed.A
        R = self.precomputed.R

        # Note: there is probably a more consistent way to do this,
        # because now `flu_contact_matrix` has two values: "is_school_day"
        # and "is_work_day" -- other schedules' dataframes only have one
        # relevant column value rather than two
        for item in [("absolute_humidity", "absolute_humidity"),
                     ("flu_contact_matrix", "is_school_day"),
                     ("flu_contact_matrix", "is_work_day"),
                     ("daily_vaccines", "daily_vaccines"),
                     ("mobility_modifier", "mobility_modifier")]:

            schedule_name = item[0]
            values_column_name = item[1]

            metapop_vals = []

            for subpop_model in self._subpop_models_ordered.values():
                df = subpop_model.schedules[schedule_name].timeseries_df

                # Using the `start_real_date` specification given in subpop's `SimulationSettings`,
                # extract the relevant part of the dataframe with dates >= the simulation start date.
                # Note that `start_real_date` should be the same for each subpopulation
                start_date = datetime.datetime.strptime(subpop_model.simulation_settings.start_real_date, "%Y-%m-%d")
                
                # If schedule uses day_of_week scheduling, we need to create the full date range
                # for the schedule dataframe
                if subpop_model.schedules[schedule_name].is_day_of_week_schedule:
                    df = create_timeseries_df_from_day_of_week_schedule(
                        df, start_date)
                
                df["simulation_day"] = (pd.to_datetime(df.index, format="%Y-%m-%d") - start_date).to_series().dt.days.values
                df = df[df["simulation_day"] >= 0]

                # Make each day's value an A x R array
                # Pandas complains about `SettingWithCopyWarning` so we work on a copy explicitly to stop it
                #   from complaining...
                df = df.copy()
                
                if schedule_name in ['daily_vaccines', 'mobility_modifier']:
                    # daily_vaccines and mobility_modifier are already given as A x R arrays
                    if df[values_column_name].values[0].shape != (A, R):
                        raise ValueError(f"Error: {schedule_name} arrays must have shape ({A}, {R}). " \
                            f"Current input has shape {df[values_column_name].values[0].shape}.")
                else:
                    df[values_column_name] = df[values_column_name].astype(object)
                    df.loc[:, values_column_name] = df[values_column_name].apply(
                        lambda x, A=A, R=R: np.broadcast_to(np.asarray(x).reshape(1, 1), (A, R))
                    )

                metapop_vals.append(np.asarray(df[values_column_name]))

            # IMPORTANT: tedious array/tensor shape/size manipulation here
            # metapop_vals: list of L arrays, each shape (num_days, A, R)
            # We need to transpose this... to be a list of num_days tensors, of size L x A x R
            num_items = metapop_vals[0].shape[0]

            # This is ugly and inefficient -- but at least we only do this once, when we get the initial
            #   state of a metapopulation model in tensor form
            transposed_metapop_vals = [torch.tensor(np.array([metapop_vals[l][i] for l in range(L)])) for i in
                                       range(num_items)]

            setattr(self._full_metapop_schedule_tensors, values_column_name, transposed_metapop_vals)

    def get_flu_torch_inputs(self) -> dict:
        """
        Prepares and returns metapopulation simulation data in tensor format
        that can be directly used for `torch` implementation.

        Returns:
             d (dict):
                Has keys "state_tensors", "params_tensors", "schedule_tensors",
                and "precomputed". Corresponds to `FluFullMetapopStateTensors`,
                `FluFullMetapopParamsTensors`, `FluFullMetapopScheduleTensors`,
                and `FluPrecomputedTensors` instances respectively.
        """

        # Note: does not support dynamic variables (yet). If want to
        #   run pytorch with dynamic variables, will need to create
        #   a method similar to `setup_full_metapop_schedule_tensors`
        #   but for dynamic variables. Also note that we cannot differentiate
        #   with respect to dynamic variables that are discontinuous
        #   (e.g. a 0-1 intervention) -- so we cannot optimize discontinuous
        #   dynamic variables.

        self.update_full_metapop_state_tensors()
        self.update_full_metapop_params_tensors()
        self._full_metapop_params_tensors.standardize_shapes()
        self.setup_full_metapop_schedule_tensors()

        d = {}

        d["state_tensors"] = copy.deepcopy(self._full_metapop_state_tensors)
        d["params_tensors"] = copy.deepcopy(self._full_metapop_params_tensors)
        d["schedule_tensors"] = copy.deepcopy(self._full_metapop_schedule_tensors)
        d["precomputed"] = copy.deepcopy(self.precomputed)

        return d
