###########################################################
######## Tests for the vaccinated track (X_V) #############
###########################################################

# People enter a parallel vaccinated track from "S" into "S_V"
#   according to the `daily_vaccines` schedule -- see
#   `ScheduledVaccination` and `FluSubpopModel` in `flu_components.py`,
#   and `advance_timestep` in `flu_torch_det_components.py`.

import flu_core as flu
import clt_toolkit as clt

import copy
import datetime
import json
import numpy as np
import pandas as pd
import pytest
import torch

from typing import Optional

from conftest import subpop_inputs


def _vaccines_df(start_date: str,
                 num_days: int,
                 proportion: float,
                 num_age_groups: int,
                 num_risk_groups: int) -> pd.DataFrame:
    """
    Constant `daily_vaccines` schedule -- `proportion` of the dose pool
    vaccinated every day, for every age-risk group.
    """

    dates = pd.date_range(start=start_date, periods=num_days, freq="D").date
    value = json.dumps((np.ones((num_age_groups, num_risk_groups)) * proportion).tolist())

    return pd.DataFrame({"date": [str(d) for d in dates],
                         "daily_vaccines": [value] * num_days})


def _make_subpop_model(case_id_str: str = "caseA",
                       transition_type: str = "binom",
                       timesteps_per_day: int = 4,
                       params_updates: Optional[dict] = None,
                       vaccines_df: Optional[pd.DataFrame] = None,
                       settings_updates: Optional[dict] = None,
                       name: str = "subpop_model",
                       seed_jump: int = 0) -> flu.FluSubpopModel:
    """
    Builds a `FluSubpopModel` from the test fixtures, with a lower
    transmission rate than the fixtures by default (their epidemic
    otherwise empties "S" before the vaccine protection delay is over,
    leaving nobody to vaccinate).
    """

    state, params, _, settings, schedules_info = subpop_inputs(case_id_str)

    params = clt.updated_dataclass(params, {"beta_baseline": params.beta_baseline * 0.05,
                                            **(params_updates or {})})
    settings = clt.updated_dataclass(settings, {"transition_type": transition_type,
                                                "timesteps_per_day": timesteps_per_day,
                                                **(settings_updates or {})})

    if vaccines_df is not None:
        schedules_info = flu.FluSubpopSchedules(
            absolute_humidity=schedules_info.absolute_humidity,
            flu_contact_matrix=schedules_info.flu_contact_matrix,
            daily_vaccines=vaccines_df.copy(),
            mobility_modifier=schedules_info.mobility_modifier)

    RNG = np.random.Generator(np.random.MT19937(123456789).jumped(seed_jump))

    return flu.FluSubpopModel(state, params, settings, RNG, schedules_info, name)


def _caseA_vaccines(proportion: float = 0.004, days_before_start: int = 0) -> pd.DataFrame:
    start = datetime.date(2022, 8, 8) - datetime.timedelta(days=days_before_start)
    return _vaccines_df(str(start), 400, proportion, 4, 3)


def _total_population(model: flu.FluSubpopModel) -> np.ndarray:
    return np.sum([np.asarray(c.current_val, dtype=float) for c in model.compartments.values()], axis=0)


# ---------------------------------------------------------------------------
# Structure, conservation, switched-off immunity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transition_type", ["binom", "binom_deterministic",
                                             "binom_deterministic_no_round", "poisson"])
def test_population_conserved_and_nonnegative(transition_type):
    """
    The total over all 20 compartments stays constant, and no compartment
    ever goes negative.
    """

    model = _make_subpop_model(transition_type=transition_type,
                               vaccines_df=_caseA_vaccines(0.004, days_before_start=30))

    total_0 = _total_population(model)

    for day in [1, 10, 50, 150]:
        model.simulate_until_day(day)
        assert np.allclose(_total_population(model), total_0, rtol=1e-9, atol=1e-6)
        for name, compartment in model.compartments.items():
            assert np.all(np.asarray(compartment.current_val) >= 0), name

    # Sanity check: people actually moved into and through the vaccinated track
    assert np.sum(model.compartments.S_V.current_val) > 0
    assert np.sum(model.compartments.R_V.current_val) > 0


@pytest.mark.parametrize("transition_type", ["binom", "binom_deterministic_no_round"])
def test_immunity_metrics_and_R_to_S_stay_zero(transition_type):
    """
    M and MV are switched off, and R_to_S_rate is forced to 0 -- even if
    the inputs say otherwise.
    """

    state, params, _, settings, schedules_info = subpop_inputs("caseA")
    state.M = np.full_like(np.asarray(state.M, dtype=float), 0.3)
    state.MV = np.full_like(np.asarray(state.MV, dtype=float), 0.2)
    settings = clt.updated_dataclass(settings, {"transition_type": transition_type,
                                                "transition_variables_to_save": ["R_to_S", "R_V_to_S_V"]})

    with pytest.warns(UserWarning):
        model = flu.FluSubpopModel(state, params, settings,
                                   np.random.Generator(np.random.MT19937(1)),
                                   schedules_info, "subpop_model")

    assert model.params.R_to_S_rate == 0

    model.simulate_until_day(60)

    assert np.all(np.asarray(model.epi_metrics.M.history_vals_list) == 0)
    assert np.all(np.asarray(model.epi_metrics.MV.history_vals_list) == 0)
    assert np.all(np.asarray(model.transition_variables.R_to_S.history_vals_list) == 0)
    assert np.all(np.asarray(model.transition_variables.R_V_to_S_V.history_vals_list) == 0)


# ---------------------------------------------------------------------------
# Number of people vaccinated, rounding, and the negativity guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transition_type", ["binom", "binom_deterministic_no_round"])
@pytest.mark.parametrize("vax_dose_pool", ["susceptible", "total_population"])
@pytest.mark.parametrize("timesteps_per_day", [1, 4])
def test_number_vaccinated_matches_schedule(transition_type, vax_dose_pool, timesteps_per_day):
    """
    The cumulative number moved from S to S_V matches the schedule times
    the dose pool at the start of each day, within 0.5 per age-risk group
    for integer transition types (exact otherwise). The expected number is
    recomputed independently here from the saved compartment history.
    """

    num_days = 60
    proportion = 0.003

    model = _make_subpop_model(transition_type=transition_type,
                               timesteps_per_day=timesteps_per_day,
                               params_updates={"vax_dose_pool": vax_dose_pool,
                                               "vax_immunity_reset_date_mm_dd": None},
                               settings_updates={"transition_variables_to_save": ["S_to_S_V"]},
                               vaccines_df=_caseA_vaccines(proportion))

    S_0 = np.asarray(model.compartments.S.current_val, dtype=float)
    S_V_0 = np.asarray(model.compartments.S_V.current_val, dtype=float)

    model.simulate_until_day(num_days)

    # The schedule is shifted by the protection delay, so look up each
    #   day's proportion off the model's own (post-shift) schedule
    vaccines_df = model.schedules.daily_vaccines.timeseries_df
    start = model.start_real_date

    S_hist = np.asarray(model.compartments.S.history_vals_list, dtype=float)
    S_V_hist = np.asarray(model.compartments.S_V.history_vals_list, dtype=float)
    # Pool at the start of day d = end of day d - 1 (initial values for day 0)
    S_start = np.concatenate([S_0[None], S_hist[:-1]])
    S_V_start = np.concatenate([S_V_0[None], S_V_hist[:-1]])

    expected = np.zeros_like(S_0)
    for day in range(num_days):
        daily_vaccines = np.asarray(
            vaccines_df.loc[start + datetime.timedelta(days=day), "daily_vaccines"], dtype=float)
        if vax_dose_pool == "susceptible":
            pool = S_start[day] + S_V_start[day]
        else:
            pool = model.params.total_pop_age_risk
        expected += daily_vaccines * pool

    vax = model.transition_variables.S_to_S_V

    # History is saved every timestep, so this is everyone moved
    moved = np.sum(np.asarray(vax.history_vals_list, dtype=float), axis=0)

    assert np.all(vax.cumulative_capped_shortfall == 0)

    tolerance = 0.5 + 1e-6 if transition_type == "binom" else 1e-6 * np.max(expected)
    assert np.all(np.abs(moved - expected) <= tolerance)

    if transition_type == "binom":
        # Whole people only
        assert np.all(moved == np.round(moved))

    # Sanity check: the schedule actually vaccinated people
    assert np.sum(moved) > 1000


@pytest.mark.parametrize("transition_type", ["binom", "binom_deterministic_no_round"])
def test_aggressive_schedule_never_negative_and_records_shortfall(transition_type):
    """
    A schedule vaccinating half the pool every day runs out of people in S
    within a couple of days. S must never go negative, and the people who
    could not be vaccinated are recorded (and warned about), not moved.
    """

    model = _make_subpop_model(transition_type=transition_type,
                               params_updates={"vax_immunity_reset_date_mm_dd": None,
                                               "vax_protection_delay_days": 0},
                               vaccines_df=_caseA_vaccines(0.5))

    with pytest.warns(UserWarning, match="capped"):
        model.simulate_until_day(20)

    vax = model.transition_variables.S_to_S_V

    assert np.all(np.asarray(model.compartments.S.history_vals_list) >= 0)
    assert np.sum(vax.cumulative_capped_shortfall) > 0


@pytest.mark.parametrize("transition_type,expect_integer", [
    ("binom", True),
    ("binom_deterministic", True),
    ("poisson_deterministic", True),
    ("binom_deterministic_no_round", False),
])
def test_outflow_capacity_scaling_keeps_integer_types_whole(transition_type, expect_integer):
    """
    When outflows from a compartment exceed its population,
    `enforce_outflow_capacity` scales them down to fit -- and, for every
    transition type that yields whole numbers (all but the "_no_round"
    one, deterministic types included), floors the result.
    """

    model = _make_subpop_model(transition_type=transition_type)
    model.prepare_daily_state()

    S = np.asarray(model.compartments.S.current_val, dtype=float)
    tvars_from_S = [tvar for tvar in model.transition_variables.values()
                    if tvar.origin is model.compartments.S]
    for tvar in model.transition_variables.values():
        tvar.current_val = np.zeros_like(S)
    # Each outflow from S takes 0.7 of S -- together they over-draw it
    for tvar in tvars_from_S:
        tvar.current_val = np.floor(0.7 * S) + 1

    with pytest.warns(RuntimeWarning, match="exceeded its population"):
        model.enforce_outflow_capacity()

    total = sum(np.asarray(tvar.current_val, dtype=float) for tvar in tvars_from_S)
    assert np.all(total <= S + 1e-9)
    for tvar in tvars_from_S:
        val = np.asarray(tvar.current_val, dtype=float)
        assert np.array_equal(val, np.floor(val)) == expect_integer


# ---------------------------------------------------------------------------
# Vaccinations before the start date, and the reset date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vax_dose_pool", ["susceptible", "total_population"])
def test_pre_start_vaccinations_go_into_S_V(vax_dose_pool):
    """
    Vaccinations scheduled between the reset date and the start date put
    people in S_V at time 0 -- with a constant schedule and a constant
    pool, that is (number of days) x proportion x pool, up to rounding.
    Nothing moves when the reset date is None.
    """

    proportion = 0.002
    # Schedule starts 60 days before the simulation; the protection delay
    #   (14 days) shifts it forward, so protection starts 46 days before
    model = _make_subpop_model(params_updates={"vax_dose_pool": vax_dose_pool,
                                               "vax_immunity_reset_date_mm_dd": "07_20"},
                               vaccines_df=_caseA_vaccines(proportion, days_before_start=60))

    state, _, _, _, _ = subpop_inputs("caseA")
    S_input = np.asarray(state.S, dtype=float)

    # Window: 2022-07-20 up to (not including) 2022-08-08 -- 19 days, all
    #   of them after the protection delay
    num_days = (datetime.date(2022, 8, 8) - datetime.date(2022, 7, 20)).days

    if vax_dose_pool == "susceptible":
        pool = S_input  # S + S_V, and S_V starts at 0
    else:
        pool = model.params.total_pop_age_risk
    expected_shift = num_days * proportion * pool

    S_V_0 = np.asarray(model.compartments.S_V.current_val, dtype=float)
    assert np.all(np.abs(S_V_0 - expected_shift) <= 0.5 + 1e-6)
    assert np.allclose(np.asarray(model.compartments.S.current_val), S_input - S_V_0)
    # `self.state` is synced too, so the torch inputs see the shift
    assert np.allclose(np.asarray(model.state.S_V), S_V_0)

    no_reset_model = _make_subpop_model(params_updates={"vax_immunity_reset_date_mm_dd": None},
                                        vaccines_df=_caseA_vaccines(proportion, days_before_start=60))
    assert np.all(np.asarray(no_reset_model.compartments.S_V.current_val) == 0)


def test_reset_date_moves_S_V_back_to_S():
    """
    On `vax_immunity_reset_date_mm_dd`, everyone in S_V moves back to S,
    and nobody else on the vaccinated track moves.
    """

    model = _make_subpop_model(params_updates={"vax_immunity_reset_date_mm_dd": "09_15"},
                               vaccines_df=_caseA_vaccines(0.004, days_before_start=30))

    days_to_reset = (datetime.date(2022, 9, 15) - model.start_real_date).days
    model.simulate_until_day(days_to_reset)
    assert model.current_real_date == datetime.date(2022, 9, 15)

    before = {name: copy.deepcopy(np.asarray(c.current_val, dtype=float))
              for name, c in model.compartments.items()}
    assert np.sum(before["S_V"]) > 0

    model.prepare_daily_state()

    after = {name: np.asarray(c.current_val, dtype=float) for name, c in model.compartments.items()}

    assert np.all(after["S_V"] == 0)
    assert np.allclose(after["S"], before["S"] + before["S_V"])
    for name in flu.ALL_COMPARTMENTS:
        if name not in ("S", "S_V"):
            assert np.array_equal(after[name], before[name]), name
    assert np.allclose(np.asarray(model.state.S_V), 0)


# ---------------------------------------------------------------------------
# Vaccine effect on the vaccinated track
# ---------------------------------------------------------------------------


def test_zero_efficacy_vaccinated_track_matches_no_vaccination():
    """
    With zero vaccine efficacy, the vaccinated track behaves exactly like
    the base track -- total infections and hospital admissions over both
    tracks equal those of a run with no vaccination at all.
    """

    settings_updates = {"transition_variables_to_save":
                        flu.NEW_INFECTION_TVARS + flu.HOSPITAL_ADMISSION_TVARS}
    params_updates = {"vax_induced_inf_risk_reduce": 0.0,
                      "vax_induced_hosp_risk_reduce": 0.0,
                      "vax_induced_death_risk_reduce": 0.0,
                      "vax_immunity_reset_date_mm_dd": None}

    vaccinated = _make_subpop_model(transition_type="binom_deterministic_no_round",
                                    params_updates=params_updates,
                                    settings_updates=settings_updates,
                                    vaccines_df=_caseA_vaccines(0.004))
    unvaccinated = _make_subpop_model(transition_type="binom_deterministic_no_round",
                                      params_updates=params_updates,
                                      settings_updates=settings_updates,
                                      vaccines_df=_caseA_vaccines(0.0))

    vaccinated.simulate_until_day(100)
    unvaccinated.simulate_until_day(100)

    def total(model, names):
        return sum(np.sum(model.transition_variables[n].history_vals_list) for n in names)

    assert np.sum(vaccinated.compartments.S_V.current_val) > 0
    assert np.isclose(total(vaccinated, flu.NEW_INFECTION_TVARS),
                      total(unvaccinated, flu.NEW_INFECTION_TVARS), rtol=1e-6)
    assert np.isclose(total(vaccinated, flu.HOSPITAL_ADMISSION_TVARS),
                      total(unvaccinated, flu.HOSPITAL_ADMISSION_TVARS), rtol=1e-6)


def test_vaccine_effects_on_S_V_to_E_V_IP_V_and_ISH_V_splits():
    """
    S_V_to_E_V's rate is S_to_E's times (1 - vax_induced_inf_risk_reduce).
    The hosp and death efficacies are overall (unconditional) values, so
    the IP_V hospitalization probability is IP's times the conditional
    multiplier (1 - VE_hosp) / (1 - VE_inf), and the ISH_V death
    probability is ISH's times (1 - VE_death) / (1 - VE_hosp) -- with the
    recover transitions the complements. The products along the track
    then equal 1 - VE_hosp and 1 - VE_death.
    """

    VE_inf, VE_hosp, VE_death = 0.3, 0.6, 0.7

    model = _make_subpop_model(transition_type="binom_deterministic_no_round",
                               params_updates={"vax_induced_inf_risk_reduce": VE_inf,
                                               "vax_induced_hosp_risk_reduce": VE_hosp,
                                               "vax_induced_death_risk_reduce": VE_death})
    model.prepare_daily_state()
    model.update_transition_rates()

    tvars = model.transition_variables

    inf_mult = tvars.S_V_to_E_V.current_rate / tvars.S_to_E.current_rate
    assert np.allclose(inf_mult, 1 - VE_inf)

    IP_to_IS_rate = model.params.IP_to_IS_rate
    prob_hosp = tvars.IP_to_ISH.current_rate / IP_to_IS_rate
    prob_hosp_V = tvars.IP_V_to_ISH_V.current_rate / IP_to_IS_rate
    hosp_mult = prob_hosp_V / prob_hosp

    assert np.allclose(hosp_mult, (1 - VE_hosp) / (1 - VE_inf))
    assert np.allclose(tvars.IP_V_to_ISR_V.current_rate + tvars.IP_V_to_ISH_V.current_rate,
                       IP_to_IS_rate)

    ISH_to_H_rate = model.params.ISH_to_H_rate
    prob_death = tvars.ISH_to_HD.current_rate / ISH_to_H_rate
    prob_death_V = tvars.ISH_V_to_HD_V.current_rate / ISH_to_H_rate
    death_mult = prob_death_V / prob_death

    assert np.allclose(death_mult, (1 - VE_death) / (1 - VE_hosp))
    assert np.allclose(tvars.ISH_V_to_HR_V.current_rate + tvars.ISH_V_to_HD_V.current_rate,
                       ISH_to_H_rate)

    # Overall reductions equal the input (overall) efficacies
    assert np.allclose(inf_mult * hosp_mult, 1 - VE_hosp)
    assert np.allclose(inf_mult * hosp_mult * death_mult, 1 - VE_death)

    # Everything else on the vaccinated track matches the base track
    assert np.allclose(tvars.E_V_to_IP_V.current_rate, tvars.E_to_IP.current_rate)


def test_vax_conditional_multipliers_clip_and_warn():
    """
    When an outcome's overall efficacy is lower than the previous step's
    (VE_hosp < VE_inf, or VE_death < VE_hosp), the conditional multiplier
    is capped at 1 and the model warns at construction. A previous-step
    efficacy of 1 gives a multiplier of 1 (that step is never reached).
    """

    with pytest.warns(UserWarning, match="vax_induced_hosp_risk_reduce` is lower"):
        model = _make_subpop_model(params_updates={"vax_induced_inf_risk_reduce": 0.6,
                                                   "vax_induced_hosp_risk_reduce": 0.3,
                                                   "vax_induced_death_risk_reduce": 0.5})

    hosp_mult, death_mult = flu.flu_components.compute_vax_conditional_multipliers(model.params)
    assert np.allclose(hosp_mult, 1.0)
    assert np.allclose(death_mult, 0.5 / 0.7)

    params = clt.updated_dataclass(model.params, {"vax_induced_inf_risk_reduce": 1.0,
                                                  "vax_induced_hosp_risk_reduce": 1.0,
                                                  "vax_induced_death_risk_reduce": 1.0})
    hosp_mult, death_mult = flu.flu_components.compute_vax_conditional_multipliers(params)
    assert np.allclose(hosp_mult, 1.0)
    assert np.allclose(death_mult, 1.0)


def test_infectious_vaccinated_people_spread_infection():
    """
    Infectious people on the vaccinated track count toward the force of
    infection: moving them from the base track to the vaccinated track
    leaves the S_to_E rate unchanged.
    """

    model = _make_subpop_model()
    model.prepare_daily_state()
    model.update_transition_rates()
    rate_base = copy.deepcopy(model.transition_variables.S_to_E.current_rate)

    for name in ("IP", "ISR", "ISH", "IA"):
        model.compartments[name + "_V"].current_val = model.compartments[name].current_val
        model.compartments[name].current_val = np.zeros_like(model.compartments[name].current_val)
    model.state.sync_to_current_vals(model.compartments)
    model.update_transition_rates()

    assert np.allclose(model.transition_variables.S_to_E.current_rate, rate_base)


# ---------------------------------------------------------------------------
# numpy / torch agreement
# ---------------------------------------------------------------------------


def _make_metapop_model(vax_dose_pool: str,
                        timesteps_per_day: int) -> flu.FluMetapopModel:
    """
    Two-subpopulation deterministic model with vaccinations before the start
    date (so S_V starts nonzero) and a reset date 10 days into the run.
    """

    vaccines_df = _vaccines_df("2022-06-01", 400, 0.003, 5, 1)

    _, _, mixing_params, _, _ = subpop_inputs("caseB_subpop1")

    subpops = []
    for ix, case_id_str in enumerate(["caseB_subpop1", "caseB_subpop2"]):
        subpops.append(_make_subpop_model(
            case_id_str=case_id_str,
            transition_type="binom_deterministic_no_round",
            timesteps_per_day=timesteps_per_day,
            params_updates={"vax_dose_pool": vax_dose_pool,
                            "vax_immunity_reset_date_mm_dd": "08_18",
                            "vax_induced_death_risk_reduce": 0.4},
            settings_updates={"use_deterministic_softplus": True},
            vaccines_df=vaccines_df,
            name=f"subpop{ix + 1}",
            seed_jump=ix))

    return flu.FluMetapopModel(subpops, mixing_params)


@pytest.mark.parametrize("vax_dose_pool", ["susceptible", "total_population"])
@pytest.mark.parametrize("timesteps_per_day", [1, 3])
def test_oop_and_torch_agree_on_vaccinated_track(vax_dose_pool, timesteps_per_day):
    """
    The torch metapopulation model applies the same vaccinated track as
    the numpy model: same pre-start S_V, same daily transfers spread over
    timesteps, same reset, same V-track dynamics.
    """

    num_days = 40

    oop_model = _make_metapop_model(vax_dose_pool, timesteps_per_day)
    d = oop_model.get_flu_torch_inputs()

    torch_state_history, _ = flu.torch_simulate_full_history(
        d["state_tensors"], d["params_tensors"], d["precomputed"],
        d["schedule_tensors"], num_days, timesteps_per_day)

    torch_admits = flu.torch_simulate_hospital_admits(
        d["state_tensors"], d["params_tensors"], d["precomputed"],
        d["schedule_tensors"], num_days, timesteps_per_day)

    oop_model.modify_simulation_settings(
        {"transition_variables_to_save": flu.HOSPITAL_ADMISSION_TVARS})
    oop_model.simulate_until_day(num_days)

    for subpop_ix in range(oop_model.precomputed.L):
        subpop_model = oop_model._subpop_models_ordered[subpop_ix]
        for name in flu.ALL_COMPARTMENTS:
            oop_history = np.asarray(subpop_model.compartments[name].history_vals_list)
            torch_history = torch.stack(torch_state_history[name])[:, subpop_ix]
            assert torch.allclose(torch.tensor(oop_history, dtype=torch.float64),
                                  torch_history.to(torch.float64), rtol=1e-4, atol=1e-3), \
                f"{name} diverged for subpop {subpop_ix}"

    oop_admits = clt.aggregate_daily_tvar_history(oop_model, flu.HOSPITAL_ADMISSION_TVARS)
    # (days, A, R) summed over subpopulations, versus (days, L, A, R)
    assert np.allclose(np.asarray(oop_admits), torch_admits.sum(dim=1).detach().numpy(), rtol=1e-4)

    # Sanity checks: S_V starts nonzero, and the reset (day 10) emptied it
    S_V_history = np.asarray(
        oop_model._subpop_models_ordered[0].compartments["S_V"].history_vals_list)
    S_V_start = oop_model._subpop_models_ordered[0].compartments["S_V"].init_val
    assert np.sum(S_V_start) > 0
    assert np.sum(S_V_history[10]) < 0.05 * np.sum(S_V_history[9])


def test_oop_and_torch_agree_on_vax_conditional_multipliers():
    """
    The numpy and torch helpers give the same conditional hosp and death
    multipliers, including where a multiplier is capped at 1.
    """

    oop_model = _make_metapop_model("susceptible", 1)
    params_tensors = oop_model.get_flu_torch_inputs()["params_tensors"]

    for VE_inf, VE_hosp, VE_death in [(0.3, 0.6, 0.7), (0.6, 0.3, 0.5)]:
        params_tensors.vax_induced_inf_risk_reduce = torch.full_like(
            params_tensors.vax_induced_inf_risk_reduce, VE_inf)
        params_tensors.vax_induced_hosp_risk_reduce = torch.full_like(
            params_tensors.vax_induced_hosp_risk_reduce, VE_hosp)
        params_tensors.vax_induced_death_risk_reduce = torch.full_like(
            params_tensors.vax_induced_death_risk_reduce, VE_death)

        subpop_params = clt.updated_dataclass(
            oop_model._subpop_models_ordered[0].params,
            {"vax_induced_inf_risk_reduce": VE_inf,
             "vax_induced_hosp_risk_reduce": VE_hosp,
             "vax_induced_death_risk_reduce": VE_death})

        np_mults = flu.compute_vax_conditional_multipliers(subpop_params)
        torch_mults = flu.torch_compute_vax_conditional_multipliers(params_tensors)

        for np_mult, torch_mult in zip(np_mults, torch_mults):
            assert np.allclose(torch_mult.detach().numpy(),
                               np.broadcast_to(np_mult, torch_mult.shape))
