import flu_core as flu
import clt_toolkit as clt

import datetime
import numpy as np
import pandas as pd
import copy
import pytest

from conftest import subpop_inputs
from helpers import binom_transition_types_list, binom_random_transition_types_list, \
    binom_no_taylor_transition_types_list, inputs_id_list, check_state_variables_same_history

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"


def test_num_timesteps(make_flu_subpop_model):
    """
    If "timesteps_per_day" in SimulationSettings increases (number of timesteps per day
        increases), then step sizes are smaller.

    Using binomial deterministic transitions, realizations will be smaller
        for more timesteps per day.
    """

    few_timesteps_model = make_flu_subpop_model("few_timesteps",
                                            clt.TransitionTypes.BINOM_DETERMINISTIC,
                                            timesteps_per_day = 2)

    few_timesteps_model.prepare_daily_state()
    few_timesteps_model._simulate_timesteps(1)

    many_timesteps_model = make_flu_subpop_model("many_timesteps",
                                             clt.TransitionTypes.BINOM_DETERMINISTIC,
                                             timesteps_per_day = 20)

    many_timesteps_model.prepare_daily_state()
    many_timesteps_model._simulate_timesteps(1)

    for name in few_timesteps_model.transition_variables.keys():
        assert (few_timesteps_model.transition_variables[name].current_val >=
                many_timesteps_model.transition_variables[name].current_val).all()


def test_subpop_correct_object_count(make_flu_subpop_model):
    """
    For each SubpopModel, there should be 8 epi compartments,
        10 transition variables, 2 transition variable groups,
        and 3 epi metrics
    """

    model = make_flu_subpop_model("model")

    assert len(model.compartments) == 10
    assert len(model.transition_variables) == 12
    assert len(model.transition_variable_groups) == 3

    assert len(model.epi_metrics) == 2

    assert len(model.dynamic_vals) == 1


@pytest.mark.parametrize("transition_type", binom_transition_types_list)
def test_subpop_constructor_no_unintended_sharing(make_flu_subpop_model, transition_type):
    """
    Regression test: there was a previous bug where the same
        SubpopState object was being shared across multiple models created
        by subsequent creation calls on the model constructor.
        This is remedied using deep copies -- we make sure that
        the model constructor always creates a transmission model
        with its own distinct/independent SubpopState OBJECT,
        even if the actual initial SubpopState VALUES are the same
        across models.

    This test makes sure that SubpopState objects across models
        created by the same constructor are indeed distinct/independent.
    """

    first_model = make_flu_subpop_model("first", transition_type)
    second_model = make_flu_subpop_model("second", transition_type)

    init_vals = copy.deepcopy(second_model.state)

    first_model.simulate_until_day(100)

    # The initial state of the second model should still be the same
    #   initial state -- it should not have been affected by simulating
    #   the first model

    for key, value in vars(init_vals).items():
        if isinstance(value, (np.ndarray, list)):
            try:
                assert (getattr(second_model.state, key) ==
                        getattr(init_vals, key))
            # if it's an array, have to check equality of each element --
            #   Python will complain that the truth value of an array is ambiguous
            except ValueError:
                assert (getattr(second_model.state, key) ==
                        getattr(init_vals, key)).all()


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list)
def test_subpop_constructor_reproducible_results(make_flu_subpop_model, transition_type):
    """
    If the flu model constructor creates two identical models
        with the same starting random number seed, they should give
        the same results. Specifically, if the first model is simulated
        before the second model is created, the results should still
        be the same.

    Also a way of ensuring there is no unintended object sharing
        or unintended mutability issues with model constructors.
        Specifically, simulating a model created from a constructor
        should not modify objects on that constructor.
    """

    first_model = make_flu_subpop_model("first", transition_type)
    second_model = make_flu_subpop_model("second", transition_type)

    first_model.simulate_until_day(100)
    second_model.simulate_until_day(100)

    check_state_variables_same_history(first_model, second_model)


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list)
def test_subpop_no_transmission_when_beta_zero(make_flu_subpop_model, transition_type):
    """
    If the transmission rate beta_baseline = 0, then S should not decrease over time
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type)
    subpop_model.reset_simulation()
    subpop_model.modify_subpop_params({"beta_baseline": 0})
    subpop_model.simulate_until_day(300)

    S_history = subpop_model.compartments["S"].history_vals_list

    assert np.sum((np.diff(np.sum(S_history, axis=(1, 2))) >= 0)) == len(S_history) - 1


@pytest.mark.parametrize("transition_type", binom_transition_types_list)
def test_subpop_dead_compartment_monotonic(make_flu_subpop_model, transition_type):
    """
    People do not rise from the dead; the dead compartment
        should not decrease over time
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type)

    subpop_model.reset_simulation()
    subpop_model.modify_subpop_params({"beta_baseline": 1.1})
    subpop_model.simulate_until_day(300)

    D_history = subpop_model.compartments["D"].history_vals_list

    assert np.sum(np.diff(np.sum(D_history, axis=(1, 2))) >= 0) == len(D_history) - 1


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_subpop_population_is_constant(make_flu_subpop_model, transition_type, inputs_id):
    """
    The total population (summed over all compartments and age-risk groups)
        should be constant over time, equal to the initial total population.
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type, case_id_str = inputs_id)

    for day in range(300):
        subpop_model.simulate_until_day(day)

        current_sum_all_compartments = 0
        for compartment in subpop_model.compartments.values():
            current_sum_all_compartments += np.sum(compartment.current_val)

        assert np.abs(current_sum_all_compartments -
                      np.sum(subpop_model.params.total_pop_age_risk)) < 1e-6


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_subpop_reset_reproducible_results(make_flu_subpop_model, transition_type, inputs_id):
    """
    Resetting the random number generator and simulating should
        give the same results as the initial run.
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type, case_id_str = inputs_id)

    subpop_model.modify_random_seed(123456789123456789)
    subpop_model.simulate_until_day(100)

    original_model_history_dict = {}

    for name, compartment in subpop_model.compartments.items():
        original_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    reset_model_history_dict = {}

    subpop_model.reset_simulation()
    subpop_model.modify_random_seed(123456789123456789)
    subpop_model.simulate_until_day(100)

    for name, compartment in subpop_model.compartments.items():
        reset_model_history_dict[name] = \
            copy.deepcopy(compartment.history_vals_list)

    for name in subpop_model.compartments.keys():
        assert np.array_equal(np.array(original_model_history_dict[name]),
                              np.array(reset_model_history_dict[name]))


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list)
@pytest.mark.parametrize("inputs_id", inputs_id_list)
def test_subpop_reset_matches_fresh_model_with_same_params(make_flu_subpop_model, transition_type, inputs_id):
    """
    A model that is built fresh with some params already modified from
    their default values should give identical simulation results to a
    model that is built with default params, simulated once, then has
    those same params modified (via `modify_subpop_params`) and is reset
    (via `reset_simulation`) before simulating again.

    This guards `reset_simulation`'s recomputation of derived quantities
    that depend on `params` -- `MV.init_val` and the
    `vax_induced_*_risk_reduce_initial` fields (see
    `FluSubpopModel.reset_simulation` and
    `update_vax_induced_risk_reduce_initial` in flu_components.py) -- to
    make sure they reflect the *current* params after reset rather than
    stale values left over from construction time or from a prior run.
    """

    params_updates = {
        "beta_baseline": 0.9,
        "vax_induced_immune_wane": 0.02,
        "vax_induced_inf_risk_reduce": 0.4,
    }
    seed = 123456789123456789

    # Model built fresh, with the modified params in place before it is
    #   ever simulated
    scratch_model = make_flu_subpop_model("scratch_model", transition_type, case_id_str=inputs_id)
    scratch_model.modify_subpop_params(params_updates)
    scratch_model.reset_simulation()
    scratch_model.modify_random_seed(seed)
    scratch_model.simulate_until_day(100)

    # Model that runs once under the original (unmodified) params, and is
    #   then modified and reset before being simulated again
    reused_model = make_flu_subpop_model("reused_model", transition_type, case_id_str=inputs_id)
    reused_model.modify_random_seed(seed)
    reused_model.simulate_until_day(100)

    reused_model.modify_subpop_params(params_updates)
    reused_model.reset_simulation()
    reused_model.modify_random_seed(seed)
    reused_model.simulate_until_day(100)

    check_state_variables_same_history(scratch_model, reused_model)


@pytest.mark.parametrize("inputs_id,vaccines_csv_name", [
    ("caseA", "caseA_daily_vaccines_constant.csv"),
    ("caseB_subpop1", "caseB_daily_vaccines_constant.csv"),
])
def test_subpop_reset_recomputes_MV_init_val(make_flu_subpop_model, inputs_id, vaccines_csv_name):
    """
    `reset_simulation` should recompute `MV.init_val` from the model's
    *current* schedule/params, not silently keep the value computed at
    construction time -- see `FluSubpopModel.reset_simulation`'s docstring,
    which specifically calls out `replace_schedule` and param overrides as
    the two ways this value can go stale.

    With the default test fixtures, `MV.init_val`'s reset-date adjustment
    (in `VaxInducedImmunity.adjust_initial_value`) is a no-op: the
    `daily_vaccines` schedule has no history before the simulation start
    date, so the window of vaccines counted toward the adjustment is
    always empty. To meaningfully exercise the recomputation, this test
    prepends pre-simulation vaccination history to the schedule (via
    `replace_schedule`) and moves `vax_immunity_reset_date_mm_dd` earlier
    (via `modify_subpop_params`) so that window is non-empty, then checks
    that after `reset_simulation`, `MV.init_val`:
        1. actually changed from its construction-time value (so this test
           would fail to exercise anything if it hadn't), and
        2. matches a value computed independently (calling
           `adjust_initial_value` directly), so the recomputed value is not
           just different but *correct*.
    """

    subpop_model = make_flu_subpop_model("subpop_model", case_id_str=inputs_id)

    MV = subpop_model.epi_metrics["MV"]
    original_MV_init_val = copy.deepcopy(MV.init_val)

    # Prepend 100 days of nonzero vaccination history before the schedule's
    #   original start date, so the reset-date window has doses to count
    raw_df = pd.read_csv(base_path / vaccines_csv_name, index_col=0)
    raw_df["date"] = pd.to_datetime(raw_df["date"], format="%Y-%m-%d").dt.date
    fill_val = raw_df["daily_vaccines"].iloc[0]
    min_date = pd.Timestamp(raw_df["date"].min())
    extra_dates = pd.date_range(end=min_date - pd.Timedelta(days=1), periods=100, freq="D").date
    extra_df = pd.DataFrame({"date": extra_dates, "daily_vaccines": [fill_val] * len(extra_dates)})
    combined_df = pd.concat([extra_df, raw_df], ignore_index=True)

    subpop_model.replace_schedule("daily_vaccines", combined_df)

    params_updates = {
        "vax_induced_immune_wane": 0.02,
        "vax_immunity_reset_date_mm_dd": "06_01",
    }
    subpop_model.modify_subpop_params(params_updates)
    subpop_model.reset_simulation()

    expected_MV_init_val = MV.adjust_initial_value(
        MV.original_init_val,
        subpop_model.start_real_date,
        subpop_model.params,
        subpop_model.schedules,
        subpop_model.simulation_settings.timesteps_per_day)

    # Sanity check: the schedule/param change must actually move
    #   MV.init_val, otherwise this test would pass even if reset never
    #   recomputed it
    assert not np.array_equal(np.asarray(original_MV_init_val),
                              np.asarray(MV.init_val))

    # The value reset_simulation left in place should match the value
    #   independently recomputed from the current schedule/params
    assert np.allclose(np.asarray(MV.init_val), np.asarray(expected_MV_init_val))
    assert np.allclose(np.asarray(MV.current_val), np.asarray(expected_MV_init_val))


def _make_flu_subpop_model_with_M(mm_dd, M_val=None, case_id_str="caseA"):
    """
    Helper: build a `FluSubpopModel` with a nonzero `M` initial value
    and a given `infection_immunity_start_date_mm_dd`, so
    `InfInducedImmunity.adjust_initial_value` can be exercised
    meaningfully (the default test fixtures' `M` init val is all zeros).
    """

    init_vals, params, mixing_params, simulation_settings, schedules_info = \
        subpop_inputs(case_id_str)

    if M_val is None:
        M_val = np.full_like(np.asarray(init_vals.M, dtype=float), 0.1)
    init_vals.M = M_val

    params = clt.updated_dataclass(
        params, {"infection_immunity_start_date_mm_dd": mm_dd})

    starting_random_seed = 123456789123456789
    bit_generator = np.random.MT19937(starting_random_seed)

    model = flu.FluSubpopModel(init_vals,
                               params,
                               simulation_settings,
                               np.random.Generator(bit_generator),
                               schedules_info,
                               "subpop_model")

    return model


def test_M_adjust_initial_value_same_date():
    """
    If infection_immunity_start_date_mm_dd equals start_real_date,
    M.init_val should be used as-is (no adjustment).
    """

    model = _make_flu_subpop_model_with_M("08_08")

    assert model.start_real_date == datetime.date(2022, 8, 8)

    M = model.epi_metrics["M"]
    assert np.allclose(np.asarray(M.init_val), np.asarray(M.original_init_val))
    assert M.pending_injection_date is None


def test_M_adjust_initial_value_past_date_decays_with_waning():
    """
    If infection_immunity_start_date_mm_dd is before start_real_date,
    M.init_val should be decayed forward from the input M(0) using only
    the waning term (matching the waning piece of
    InfInducedImmunity.get_change_in_current_val).
    """

    model = _make_flu_subpop_model_with_M("08_01")

    assert model.start_real_date == datetime.date(2022, 8, 8)

    M = model.epi_metrics["M"]

    timesteps_per_day = model.simulation_settings.timesteps_per_day
    wane = model.params.inf_induced_immune_wane

    expected = np.asarray(M.original_init_val, dtype=float).copy()
    num_days = (datetime.date(2022, 8, 8) - datetime.date(2022, 8, 1)).days
    for _ in range(num_days):
        for _ in range(timesteps_per_day):
            expected = expected - wane * expected / timesteps_per_day

    assert np.allclose(np.asarray(M.init_val), expected)
    assert M.pending_injection_date is None
    # Sanity check: waning should have actually reduced the value
    assert np.all(expected < np.asarray(M.original_init_val, dtype=float))


def test_M_adjust_initial_value_future_date_defers_injection():
    """
    If infection_immunity_start_date_mm_dd is after start_real_date,
    M.init_val should be zero at simulation start, and the original
    M(0) should only be added to current_val once that date is reached
    (via check_and_apply_injection), exactly once.
    """

    model = _make_flu_subpop_model_with_M("08_10")

    assert model.start_real_date == datetime.date(2022, 8, 8)

    M = model.epi_metrics["M"]

    assert np.allclose(np.asarray(M.init_val), np.zeros_like(np.asarray(M.init_val)))
    assert M.pending_injection_date == datetime.date(2022, 8, 10)

    # A day before the injection date: no change
    M.check_and_apply_injection(datetime.date(2022, 8, 9), model.params)
    assert np.allclose(np.asarray(M.current_val), np.zeros_like(np.asarray(M.current_val)))
    assert M.pending_injection_date == datetime.date(2022, 8, 10)

    # On the injection date: current_val jumps by the original init val
    M.check_and_apply_injection(datetime.date(2022, 8, 10), model.params)
    assert np.allclose(np.asarray(M.current_val), np.asarray(M.original_init_val))
    assert M.pending_injection_date is None

    # Calling again on/after the same date should not double-inject
    M.check_and_apply_injection(datetime.date(2022, 8, 10), model.params)
    assert np.allclose(np.asarray(M.current_val), np.asarray(M.original_init_val))


def test_subpop_reset_recomputes_M_init_val():
    """
    Analogous to test_subpop_reset_recomputes_MV_init_val: reset_simulation
    should recompute M.init_val (and pending_injection_date) from the
    model's *current* params, not silently keep the value/state computed
    at construction time.
    """

    model = _make_flu_subpop_model_with_M("08_10")

    M = model.epi_metrics["M"]
    original_M_init_val = copy.deepcopy(M.init_val)
    assert M.pending_injection_date == datetime.date(2022, 8, 10)

    # Change infection_immunity_start_date_mm_dd to a date before
    #   start_real_date, so the adjustment now decays M(0) instead of
    #   deferring an injection
    model.modify_subpop_params({"infection_immunity_start_date_mm_dd": "08_01"})
    model.reset_simulation()

    expected_M_init_val = M.adjust_initial_value(
        M.original_init_val,
        model.start_real_date,
        model.params,
        model.simulation_settings.timesteps_per_day)

    # Sanity check: the param change must actually move M.init_val
    assert not np.array_equal(np.asarray(original_M_init_val),
                              np.asarray(M.init_val))

    assert np.allclose(np.asarray(M.init_val), np.asarray(expected_M_init_val))
    assert np.allclose(np.asarray(M.current_val), np.asarray(expected_M_init_val))
    assert M.pending_injection_date is None


def _analytic_pure_wane_decay(M0, wane, timesteps_per_day, num_days):
    """
    Closed-form value of M after `num_days` of pure exponential waning
    (no growth term), applied with the same per-timestep discretization
    as InfInducedImmunity.get_change_in_current_val's waning piece:
        M <- M - wane * M / timesteps_per_day, each timestep.
    """

    factor = (1 - wane / timesteps_per_day) ** (num_days * timesteps_per_day)
    return np.asarray(M0, dtype=float) * factor


def _make_isolated_M_model(mm_dd, wane, M0_val=0.2, timesteps_per_day=7,
                           case_id_str="caseA"):
    """
    Helper: build a `FluSubpopModel` with no epidemic activity at all
    (E, IP, ISR, ISH, IA, HR, HD, R, D all zeroed out, S left as-is) so
    R_to_S stays exactly zero throughout the simulation. This isolates
    InfInducedImmunity's waning term -- with R_to_S == 0, the ODE
    reduces exactly to dM/dt = -wane * M -- so M's simulated trajectory
    can be checked against a closed-form decay curve.
    """

    init_vals, params, mixing_params, simulation_settings, schedules_info = \
        subpop_inputs(case_id_str)

    for compartment_name in ("E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"):
        zeros = np.zeros_like(np.asarray(getattr(init_vals, compartment_name)))
        setattr(init_vals, compartment_name, zeros)

    init_vals.M = np.full_like(np.asarray(init_vals.M, dtype=float), M0_val)

    params = clt.updated_dataclass(
        params, {"infection_immunity_start_date_mm_dd": mm_dd,
                 "inf_induced_immune_wane": wane})

    simulation_settings = clt.updated_dataclass(
        simulation_settings, {"timesteps_per_day": timesteps_per_day})

    starting_random_seed = 123456789123456789
    bit_generator = np.random.MT19937(starting_random_seed)

    model = flu.FluSubpopModel(init_vals,
                               params,
                               simulation_settings,
                               np.random.Generator(bit_generator),
                               schedules_info,
                               "subpop_model")

    return model


def test_M_simulated_curve_same_date_matches_analytic_decay():
    """
    infection_immunity_start_date_mm_dd == start_real_date: M(0) is used
    as-is, and (with no epidemic activity) M should follow the pure
    waning decay curve exactly as the simulation progresses.
    """

    wane = 0.05
    M0 = 0.2
    timesteps_per_day = 7

    model = _make_isolated_M_model("08_08", wane, M0_val=M0,
                                   timesteps_per_day=timesteps_per_day)
    assert model.start_real_date == datetime.date(2022, 8, 8)

    num_days = 5
    model.simulate_until_day(num_days)

    expected = _analytic_pure_wane_decay(M0, wane, timesteps_per_day, num_days)

    assert np.allclose(np.asarray(model.epi_metrics["M"].current_val), expected)
    # Sanity check: waning should have actually reduced M below M(0)
    assert np.all(expected < M0)


def test_M_simulated_curve_past_date_shows_immunity_declining():
    """
    infection_immunity_start_date_mm_dd < start_real_date, with
    inf_induced_immune_wane > 0: M(0) is decayed forward to
    start_real_date, and then continues decaying identically as the
    simulation progresses -- i.e. the pre-simulation adjustment and the
    in-simulation ODE waning compose into one continuous decay curve,
    and immunity is strictly lower than in the same-date case (since it
    has been waning for longer).
    """

    wane = 0.05
    M0 = 0.2
    timesteps_per_day = 7

    model = _make_isolated_M_model("08_01", wane, M0_val=M0,
                                   timesteps_per_day=timesteps_per_day)
    assert model.start_real_date == datetime.date(2022, 8, 8)

    days_before_start = (datetime.date(2022, 8, 8) - datetime.date(2022, 8, 1)).days

    num_simulated_days = 5
    model.simulate_until_day(num_simulated_days)

    total_decay_days = days_before_start + num_simulated_days
    expected = _analytic_pure_wane_decay(M0, wane, timesteps_per_day, total_decay_days)

    actual = np.asarray(model.epi_metrics["M"].current_val)
    assert np.allclose(actual, expected)

    # Immunity should be strictly lower than the same-date case simulated
    #   for the same number of days, since it started decaying earlier
    same_date_expected = _analytic_pure_wane_decay(
        M0, wane, timesteps_per_day, num_simulated_days)
    assert np.all(actual < same_date_expected)


def test_M_simulated_curve_future_date_injects_then_wanes():
    """
    infection_immunity_start_date_mm_dd > start_real_date, with
    inf_induced_immune_wane > 0: M stays at 0 until the injection date is
    reached, then jumps to M(0) and wanes from there identically to the
    same-date case (just shifted in time).
    """

    wane = 0.05
    M0 = 0.2
    timesteps_per_day = 7

    model = _make_isolated_M_model("08_10", wane, M0_val=M0,
                                   timesteps_per_day=timesteps_per_day)
    assert model.start_real_date == datetime.date(2022, 8, 8)

    days_until_injection = (datetime.date(2022, 8, 10) - datetime.date(2022, 8, 8)).days

    # Before the injection date: M stays at 0
    model.simulate_until_day(days_until_injection)
    assert np.allclose(np.asarray(model.epi_metrics["M"].current_val),
                       np.zeros_like(np.asarray(model.epi_metrics["M"].current_val)))

    # Continue past the injection date, then check the post-injection
    #   waning curve
    num_days_after_injection = 3
    model.simulate_until_day(days_until_injection + num_days_after_injection)

    expected = _analytic_pure_wane_decay(
        M0, wane, timesteps_per_day, num_days_after_injection)

    assert np.allclose(np.asarray(model.epi_metrics["M"].current_val), expected)
    assert np.all(expected < M0)


@pytest.mark.parametrize("transition_type", binom_random_transition_types_list + ["poisson"])
def test_compartments_integer_population(make_flu_subpop_model, transition_type):
    """
    Compartment populations should be integer-valued.
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type)

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for compartment in subpop_model.compartments.values():
            assert (compartment.current_val ==
                    np.asarray(compartment.current_val, dtype=int)).all()


@pytest.mark.parametrize("transition_type", binom_transition_types_list)
def test_transition_format(make_flu_subpop_model, transition_type):
    """
    Transition variables' transition rates and
        current value should be A x L, where
        A is the number of risk groups and L is the
        number of age groups.

    Transition rates should also be floats, even though the
        transition variable realization is integer
        (so that population counts in compartments
        always stay integer). Transition rates should be
        floats to prevent premature rounding. Binomial
        and Poisson random variables are always integer,
        but their deterministic equivalents may not be
        under our implementation -- so we round them
        after the fact.
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type)

    A = subpop_model.params.num_age_groups
    L = subpop_model.params.num_risk_groups

    for day in [1, 10, 100]:
        subpop_model.simulate_until_day(day)

        for tvar in subpop_model.transition_variables.values():
            assert np.shape(tvar.current_rate) == (A, L)
            assert np.shape(tvar.current_val) == (A, L)

            for element in tvar.current_rate.flatten():
                assert isinstance(element, float)


@pytest.mark.parametrize("transition_type", binom_transition_types_list)
def test_M_no_waning_no_saturation(make_flu_subpop_model, transition_type):

    """
    From Anass:
        I set the waning constant to zero and removed the saturation factor.
        The equation becomes dM/dt = sigma_{R-S}(t) / N.
        It means that the final immunity should be (1/N) integral from 0 to T of [sigma_{R-S}(t)] dt.
    """

    subpop_model = make_flu_subpop_model("subpop_model", transition_type)

    subpop_model.simulation_settings = clt.updated_dataclass(subpop_model.simulation_settings,
                                                             {"transition_variables_to_save": ["R_to_S"]})
    subpop_model.modify_subpop_params({"inf_induced_immune_wane": 0,
                                       "inf_induced_saturation": 0,})

    subpop_model.simulate_until_day(100)

    assert np.all(np.isclose(subpop_model.M.current_val,
                             np.sum(np.asarray(subpop_model.R_to_S.history_vals_list), axis=0) / subpop_model.params.total_pop_age_risk,
                             rtol=0.02))