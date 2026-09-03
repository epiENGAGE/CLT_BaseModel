"""
Tests for the season-average -> peak vaccine efficacy derivation
(`compute_vax_induced_risk_reduce_initial` and friends), for the
year resolution of "MM_DD" parameters, and for numpy/torch agreement
on the once-a-day MV reset and M injection.
"""

import flu_core as flu
import clt_toolkit as clt

import datetime
import json
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import subpop_inputs

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"


def _make_model(params_updates=None,
                vaccines_df=None,
                case_id_str="caseA",
                settings_updates=None):
    """
    Builds a `FluSubpopModel` from the `caseA`/`caseB` fixtures with
    optional parameter, schedule, and settings overrides.
    """

    init_vals, params, mixing_params, simulation_settings, schedules_info = \
        subpop_inputs(case_id_str)

    if params_updates:
        params = clt.updated_dataclass(params, params_updates)

    if settings_updates:
        simulation_settings = clt.updated_dataclass(
            simulation_settings, settings_updates)

    if vaccines_df is not None:
        schedules_info = clt.updated_dataclass(
            schedules_info, {"daily_vaccines": vaccines_df})

    return flu.FluSubpopModel(init_vals,
                              params,
                              simulation_settings,
                              np.random.Generator(np.random.MT19937(1234)),
                              schedules_info,
                              "subpop_model")


def _constant_vaccines_df(start_date, num_days, dose, num_age_groups, num_risk_groups):
    """
    Builds a `daily_vaccines` input dataframe giving `dose` to every
    age-risk group on each of `num_days` consecutive days.
    """

    arr = json.dumps([[dose] * num_risk_groups] * num_age_groups)
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]

    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                         "daily_vaccines": [arr] * num_days})


def _campaign_vaccines_df(start_date, num_days, campaign_days, campaign_dose,
                          tail_dose, num_age_groups, num_risk_groups):
    """
    Builds a `daily_vaccines` input dataframe shaped like a real
    vaccination campaign: `campaign_dose` for the first `campaign_days`
    days, then a long low-dose tail of `tail_dose` for the remainder.
    """

    big = json.dumps([[campaign_dose] * num_risk_groups] * num_age_groups)
    small = json.dumps([[tail_dose] * num_risk_groups] * num_age_groups)
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    doses = [big] * campaign_days + [small] * (num_days - campaign_days)

    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                         "daily_vaccines": doses})


# ---------------------------------------------------------------------------
# compute_vax_induced_risk_reduce_initial
# ---------------------------------------------------------------------------


def test_VE_initial_equals_input_when_no_waning():
    """
    With `vax_induced_immune_wane` == 0 there is no waning to correct
    for, so the derived peak efficacy must equal the input
    season-average efficacy exactly.
    """

    model = _make_model({"vax_induced_immune_wane": 0.0,
                         "vax_induced_inf_risk_reduce": 0.5,
                         "vax_induced_hosp_risk_reduce": 0.4,
                         "vax_induced_death_risk_reduce": 0.3})

    assert np.allclose(model.params.vax_induced_inf_risk_reduce_initial, 0.5)
    assert np.allclose(model.params.vax_induced_hosp_risk_reduce_initial, 0.4)
    assert np.allclose(model.params.vax_induced_death_risk_reduce_initial, 0.3)


def test_VE_initial_equals_input_when_adjustment_disabled():
    """
    `adjust_VE_for_seasonal_waning = False` skips the derivation
    entirely, so the `_initial` values are the input values -- even
    though waning is nonzero and would otherwise inflate them.
    """

    updates = {"vax_induced_immune_wane": 0.01,
               "vax_induced_inf_risk_reduce": 0.5,
               "vax_induced_hosp_risk_reduce": 0.4,
               "vax_induced_death_risk_reduce": 0.3}

    disabled = _make_model({**updates, "adjust_VE_for_seasonal_waning": False})
    enabled = _make_model({**updates, "adjust_VE_for_seasonal_waning": True})

    assert np.allclose(disabled.params.vax_induced_inf_risk_reduce_initial, 0.5)
    assert np.allclose(disabled.params.vax_induced_hosp_risk_reduce_initial, 0.4)
    assert np.allclose(disabled.params.vax_induced_death_risk_reduce_initial, 0.3)

    # Sanity check: with the adjustment on, waning does move the values,
    #   so the test above is not passing trivially
    assert not np.allclose(enabled.params.vax_induced_inf_risk_reduce_initial, 0.5)


def test_VE_initial_matches_closed_form_for_single_dose_day():
    """
    When the whole season's doses land on a single day, the dose-timing
    profile collapses to p_prot = [1] and T = 1, so the general formula

        VE_0 = VE_season * w * T / ((1 - exp(-w)) * S),  S = sum numer/cumsum

    reduces to the closed form VE_0 = VE_season * w / (1 - exp(-w)) --
    the pure within-day continuous-averaging correction, with no
    across-day waning. This pins the formula against a value that can be
    checked by hand.
    """

    wane = 0.01
    ve_season = 0.5

    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    # One dose day, then zeros -- zero-dose days are excluded from the
    #   window, so only the single dose day defines the profile
    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 200, 0.0, A, R)
    one_day = json.dumps([[0.01] * R] * A)
    df.loc[0, "daily_vaccines"] = one_day

    model = _make_model({"vax_induced_immune_wane": wane,
                         "vax_induced_inf_risk_reduce": ve_season,
                         "vax_protection_delay_days": 0,
                         "vax_immunity_reset_date_mm_dd": None},
                        vaccines_df=df)

    expected = ve_season * wane / (1 - np.exp(-wane))

    assert np.allclose(model.params.vax_induced_inf_risk_reduce_initial, expected)


def test_VE_initial_exceeds_input_and_grows_with_waning():
    """
    Correcting for waning always raises the peak above the
    season-average value, and the longer immunity has been waning by the
    end of the season, the larger that correction has to be.
    """

    ve_season = 0.5
    initials = []

    for wane in (0.002, 0.004, 0.008):
        model = _make_model({"vax_induced_immune_wane": wane,
                             "vax_induced_inf_risk_reduce": ve_season})
        initials.append(float(model.params.vax_induced_inf_risk_reduce_initial[0, 0]))

    assert all(v > ve_season for v in initials)
    assert initials[0] < initials[1] < initials[2]


def test_VE_initial_is_capped_at_one_with_warning():
    """
    A high season-average efficacy combined with fast waning implies a
    peak efficacy above 1, which is not a probability and would make
    `1 - MV * VE_0` negative at high coverage. It must be capped at 1
    and must warn, since capping means the requested season-average
    efficacy is not actually achievable.
    """

    with pytest.warns(UserWarning, match="exceeded 1.0"):
        model = _make_model({"vax_induced_immune_wane": 0.02,
                             "vax_induced_inf_risk_reduce": 0.9})

    initial = model.params.vax_induced_inf_risk_reduce_initial

    assert np.all(initial <= 1.0)
    assert np.allclose(initial, 1.0)


def test_VE_initial_not_capped_for_reasonable_values():
    """
    Guard against the cap firing spuriously: values in the range the
    real instance files use must pass through uncapped and unchanged in
    behavior (no warning).
    """

    with pytest.warns() as record:
        model = _make_model({"vax_induced_immune_wane": 0.0025,
                             "vax_induced_inf_risk_reduce": 0.52})

    assert not any("exceeded 1.0" in str(w.message) for w in record)
    assert np.all(model.params.vax_induced_inf_risk_reduce_initial < 1.0)


# ---------------------------------------------------------------------------
# VE_season_dose_window_quantile
# ---------------------------------------------------------------------------


def test_dose_window_quantile_no_effect_on_flat_schedule():
    """
    Trimming the sparse tails of the dose distribution should do nothing
    when doses are spread perfectly evenly -- there are no sparse tails,
    so the trimmed window is (almost) the full window.
    """

    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 300, 0.001, A, R)

    updates = {"vax_induced_immune_wane": 0.004,
               "vax_induced_inf_risk_reduce": 0.5,
               "vax_immunity_reset_date_mm_dd": None}

    untrimmed = _make_model(updates, vaccines_df=df.copy())
    trimmed = _make_model({**updates, "VE_season_dose_window_quantile": 0.10},
                          vaccines_df=df.copy())

    assert np.allclose(untrimmed.params.vax_induced_inf_risk_reduce_initial,
                       trimmed.params.vax_induced_inf_risk_reduce_initial,
                       rtol=0.15)


def test_dose_window_quantile_lowers_VE_initial_on_campaign_schedule():
    """
    A campaign-shaped schedule -- most doses in a short burst, then a
    long low-dose tail -- is exactly the case the trim exists for. The
    untrimmed window averages realized efficacy over months in which
    almost nobody is vaccinated, dragging the average down and inflating
    the derived peak. Trimming should pull the peak back toward the
    input season-average value, and trimming more should pull it further.
    """

    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    df = _campaign_vaccines_df(datetime.date(2022, 8, 8), num_days=300,
                               campaign_days=60, campaign_dose=0.004,
                               tail_dose=0.00001,
                               num_age_groups=A, num_risk_groups=R)

    ve_season = 0.5
    updates = {"vax_induced_immune_wane": 0.004,
               "vax_induced_inf_risk_reduce": ve_season,
               "vax_immunity_reset_date_mm_dd": None}

    initials = []
    for quantile in (None, 0.05, 0.10):
        model = _make_model({**updates, "VE_season_dose_window_quantile": quantile},
                            vaccines_df=df.copy())
        initials.append(float(model.params.vax_induced_inf_risk_reduce_initial[0, 0]))

    untrimmed, trim_05, trim_10 = initials

    # Every variant still corrects upward from the season average
    assert all(v > ve_season for v in initials)

    # Trimming moves the derived peak strictly down, toward the input
    assert untrimmed > trim_05 > trim_10

    # ...and the effect is substantial for this schedule shape, not a
    #   rounding-level difference
    assert (untrimmed - trim_10) / untrimmed > 0.05


def test_dose_window_quantile_recomputed_on_reset():
    """
    Like the other derived VE quantities, the trim setting must take
    effect on `reset_simulation` rather than being frozen at
    construction time.
    """

    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    df = _campaign_vaccines_df(datetime.date(2022, 8, 8), num_days=300,
                               campaign_days=60, campaign_dose=0.004,
                               tail_dose=0.00001,
                               num_age_groups=A, num_risk_groups=R)

    model = _make_model({"vax_induced_immune_wane": 0.004,
                         "vax_induced_inf_risk_reduce": 0.5,
                         "vax_immunity_reset_date_mm_dd": None},
                        vaccines_df=df)

    before = np.array(model.params.vax_induced_inf_risk_reduce_initial, copy=True)

    model.modify_subpop_params({"VE_season_dose_window_quantile": 0.10})
    model.reset_simulation()

    after = np.array(model.params.vax_induced_inf_risk_reduce_initial, copy=True)

    assert np.all(after < before)


# ---------------------------------------------------------------------------
# resolve_mm_dd_near_date
# ---------------------------------------------------------------------------


def test_resolve_mm_dd_picks_nearest_year_across_new_year():
    """
    The season-spanning case that a naive "use the reference year" rule
    gets backwards: for a simulation starting mid-August 2024, "01_01"
    means the coming January, not the one seven months past.
    """

    resolved = flu.resolve_mm_dd_near_date("01_01", datetime.date(2024, 8, 15))

    assert resolved == datetime.date(2025, 1, 1)


def test_resolve_mm_dd_picks_previous_year_when_nearer():
    """
    Mirror image of the above: for a simulation starting in early
    February, "12_01" means the December just gone.
    """

    resolved = flu.resolve_mm_dd_near_date("12_01", datetime.date(2025, 2, 3))

    assert resolved == datetime.date(2024, 12, 1)


def test_resolve_mm_dd_same_year_when_nearest():
    """
    The ordinary case still resolves within the reference year.
    """

    resolved = flu.resolve_mm_dd_near_date("08_01", datetime.date(2022, 8, 8))

    assert resolved == datetime.date(2022, 8, 1)


def test_resolve_mm_dd_warns_when_far_from_reference():
    """
    A date roughly half a year out is ambiguous under a nearest-year
    rule and almost certainly a mis-specification, so it warns.
    """

    with pytest.warns(UserWarning, match="days from"):
        flu.resolve_mm_dd_near_date("02_15", datetime.date(2024, 8, 15))


def test_resolve_mm_dd_handles_feb_29_in_non_leap_year():
    """
    Feb 29 does not exist in every candidate year -- the resolver must
    skip those years rather than raising.
    """

    resolved = flu.resolve_mm_dd_near_date(
        "02_29", datetime.date(2023, 3, 1), warn_days=None)

    assert resolved == datetime.date(2024, 2, 29)


def test_M_injection_date_resolves_to_following_year():
    """
    End-to-end version of the nearest-year rule: a model starting in
    mid-December with `infection_immunity_start_date_mm_dd` of "01_15"
    should defer the M injection to the coming January, not decay M(0)
    forward from eleven months earlier.
    """

    init_vals, params, mixing_params, settings, schedules_info = subpop_inputs("caseA")

    init_vals.M = np.full_like(np.asarray(init_vals.M, dtype=float), 0.1)
    params = clt.updated_dataclass(
        params, {"infection_immunity_start_date_mm_dd": "01_15"})
    settings = clt.updated_dataclass(
        settings, {"start_real_date": "2022-12-20"})

    model = flu.FluSubpopModel(init_vals, params, settings,
                               np.random.Generator(np.random.MT19937(1)),
                               schedules_info, "subpop_model")

    assert model.epi_metrics["M"].pending_injection_date == datetime.date(2023, 1, 15)
    assert np.allclose(np.asarray(model.epi_metrics["M"].init_val), 0.0)


# ---------------------------------------------------------------------------
# numpy / torch agreement on the daily MV reset and M injection
# ---------------------------------------------------------------------------


def _make_metapop_model(params_updates, settings_updates=None):

    state1, params1, mixing_params, settings, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, _, _, _ = subpop_inputs("caseB_subpop2")

    settings = clt.updated_dataclass(
        settings, {"transition_type": "binom_deterministic_no_round",
                   "timesteps_per_day": 1,
                   "use_deterministic_softplus": True,
                   **(settings_updates or {})})

    subpops = []
    for ix, (state, params) in enumerate([(state1, params1), (state2, params2)]):
        state.M = np.full_like(np.asarray(state.M, dtype=float), 0.05)
        params = clt.updated_dataclass(params, params_updates)
        subpops.append(flu.FluSubpopModel(
            state, params, settings,
            np.random.Generator(np.random.MT19937(88888).jumped(ix)),
            schedules_info, name=f"subpop{ix + 1}"))

    return flu.FluMetapopModel(subpops, mixing_params)


@pytest.mark.parametrize("mm_dd_updates,expect_M_jump,expect_MV_reset", [
    # Injection 5 days in, no MV reset inside the window
    ({"infection_immunity_start_date_mm_dd": "08_13",
      "vax_immunity_reset_date_mm_dd": None}, True, False),
    # MV reset 10 days in, no M injection
    ({"infection_immunity_start_date_mm_dd": None,
      "vax_immunity_reset_date_mm_dd": "08_18"}, False, True),
    # Both, on different days
    ({"infection_immunity_start_date_mm_dd": "08_13",
      "vax_immunity_reset_date_mm_dd": "08_18"}, True, True),
])
def test_oop_and_torch_agree_on_M_injection_and_MV_reset(
        mm_dd_updates, expect_M_jump, expect_MV_reset):
    """
    The torch metapopulation model must apply the same once-a-day M
    injection and MV reset as the object-oriented numpy model. Both
    events are scheduled a few days into the run so a short simulation
    crosses them.
    """

    num_days = 20

    oop_model = _make_metapop_model(mm_dd_updates)
    d = oop_model.get_flu_torch_inputs()

    torch_state_history, _ = flu.torch_simulate_full_history(
        d["state_tensors"], d["params_tensors"], d["precomputed"],
        d["schedule_tensors"], num_days, 1)

    oop_model.simulate_until_day(num_days)

    for subpop_ix in range(oop_model.precomputed.L):
        subpop_model = oop_model._subpop_models_ordered[subpop_ix]
        for name in ("M", "MV"):
            oop_history = np.asarray(subpop_model.epi_metrics[name].history_vals_list)
            torch_final = torch_state_history[name][num_days - 1][subpop_ix]
            assert torch.allclose(torch.tensor(oop_history[num_days - 1]),
                                  torch_final.to(torch.float64), rtol=1e-4), \
                f"{name} diverged for subpop {subpop_ix}"

    # Sanity checks: the events actually fired, so agreement above is not
    #   just two implementations both doing nothing
    M_history = np.asarray(
        oop_model._subpop_models_ordered[0].epi_metrics["M"].history_vals_list)
    MV_history = np.asarray(
        oop_model._subpop_models_ordered[0].epi_metrics["MV"].history_vals_list)

    if expect_M_jump:
        # M starts at 0 (deferred) and jumps once the injection lands
        assert np.allclose(M_history[0], 0.0)
        assert np.any(M_history[-1] > 0.04)

    if expect_MV_reset:
        # MV accumulates, then is zeroed on the reset date
        assert np.any(np.all(np.isclose(MV_history, 0.0), axis=(1, 2))[1:])
