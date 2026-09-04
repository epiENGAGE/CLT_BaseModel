"""
Tests for generic_core's port of the `ve_update` vaccine-efficacy model.

Covers:
  - the season-average -> peak efficacy derivation
    (`ve_derivation.compute_ve_inflation_factors`)
  - the year resolution of "MM_DD" parameters (`resolve_mm_dd_near_date`)
  - the deferred / decayed-forward infection-induced immunity initial value
  - the vaccine-immunity initial value's protection-delay handling
  - recomputation of all of the above on `reset_simulation`
  - numpy/torch agreement on the once-a-day MV reset and M injection

Ported from tests/test_flu_vaccine_efficacy.py on the `ve_update` branch,
driving a ConfigDrivenSubpopModel instead of a FluSubpopModel.
"""

import copy
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
from conftest import subpop_inputs

from generic_core.config_parser import parse_model_config, validate_ve_derivation_config
from generic_core.data_structures import (
    GenericSubpopParams,
    resolve_mm_dd_near_date,
)
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)
from generic_core.metric_templates import injection_val_param_name
from generic_core.rate_templates import _vax_induced_peak_efficacy_np
from generic_core.ve_derivation import ve_inflation_param_name

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
FLU_CONFIG_PATH = BASE_PATH / "flu_generic_config.json"

COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]

RNG_SEED = 123456789123456789

INF_SEASON = "vax_induced_inf_risk_reduce"
HOSP_SEASON = "vax_induced_hosp_risk_reduce"
DEATH_SEASON = "vax_induced_death_risk_reduce"


def ve0(model, season_param):
    """
    The peak (zero-waning) efficacy the model actually applies for a given
    season-average param: `min(season_average * derived inflation, 1)`.

    Read through the same helper the rate templates use, so these tests pin
    the value that reaches the trajectory rather than an intermediate.
    """
    return np.asarray(_vax_induced_peak_efficacy_np(
        model.params, {"vax_reduce_param": season_param}), dtype=float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(param_updates=None,
                vaccines_df=None,
                case_id_str="caseA",
                settings_updates=None,
                epi_metric_init=None,
                metric_config_updates=None):
    """
    Build a ConfigDrivenSubpopModel from flu_generic_config.json and the
    caseA/caseB fixtures, with optional param, schedule, and settings
    overrides.

    `param_updates` are applied to the config's `params` block before
    parsing, so they flow through validation exactly as a user-supplied
    config would.
    """

    state, flu_params, _, settings, schedules_info = subpop_inputs(case_id_str)

    if settings_updates:
        settings = clt.updated_dataclass(settings, settings_updates)

    if vaccines_df is not None:
        schedules_info = clt.updated_dataclass(
            schedules_info, {"daily_vaccines": vaccines_df})

    model_config = parse_model_config(FLU_CONFIG_PATH, schedules_input=schedules_info)

    if param_updates:
        # Applied post-parse so that DERIVED params (which must not appear in
        # the config's own "params" block) can also be injected by tests.
        model_config.params.update(param_updates)
        model_config.param_names |= set(param_updates)

    # Wire optional update_config keys (e.g. the infection-immunity start-date
    # param) onto a named epi metric.
    for metric_name, extra in (metric_config_updates or {}).items():
        mc = next(m for m in model_config.epi_metrics if m.name == metric_name)
        mc.update_config = {**mc.update_config, **extra}

    compartment_init = {name: getattr(state, name) for name in COMPARTMENTS}
    metric_init = epi_metric_init or {
        "M": np.asarray(state.M, dtype=float),
        "MV": np.asarray(state.MV, dtype=float),
    }

    state_init = build_state_from_config(model_config, compartment_init, metric_init)
    params = build_params_from_config(
        model_config,
        num_age_groups=flu_params.num_age_groups,
        num_risk_groups=flu_params.num_risk_groups,
    )

    return ConfigDrivenSubpopModel(
        model_config=model_config,
        state_init=state_init,
        params=params,
        simulation_settings=settings,
        RNG=np.random.Generator(np.random.MT19937(RNG_SEED)),
        schedules_input=schedules_info,
        name="ve_test",
    )


def _start_date(settings) -> datetime.date:
    """
    `SimulationSettings.start_real_date` comes out of the test fixtures as an
    ISO string; the model converts it internally. Tests that do date
    arithmetic need the real date object.
    """
    value = settings.start_real_date
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    return value


def _constant_vaccines_df(start_date, num_days, dose, num_age_groups, num_risk_groups):
    """
    A `daily_vaccines` input dataframe giving `dose` to every age-risk group
    on each of `num_days` consecutive days.
    """
    arr = json.dumps([[dose] * num_risk_groups] * num_age_groups)
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                         "daily_vaccines": [arr] * num_days})


def _campaign_vaccines_df(start_date, num_days, campaign_days, campaign_dose,
                          tail_dose, num_age_groups, num_risk_groups):
    """
    A `daily_vaccines` input dataframe shaped like a real vaccination
    campaign: `campaign_dose` for the first `campaign_days` days, then a long
    low-dose tail of `tail_dose` for the remainder.
    """
    big = json.dumps([[campaign_dose] * num_risk_groups] * num_age_groups)
    small = json.dumps([[tail_dose] * num_risk_groups] * num_age_groups)
    dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    doses = [big] * campaign_days + [small] * (num_days - campaign_days)
    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                         "daily_vaccines": doses})


# ---------------------------------------------------------------------------
# compute_ve_inflation_factors / applied peak efficacy
# ---------------------------------------------------------------------------

def test_VE_initial_equals_input_when_no_waning():
    """
    With a waning rate of 0 there is no waning to correct for, so the
    derived peak efficacy must equal the input season-average efficacy
    exactly.
    """
    model = _make_model({"vax_induced_immune_wane": 0.0,
                         "vax_induced_inf_risk_reduce": 0.5,
                         "vax_induced_hosp_risk_reduce": 0.4,
                         "vax_induced_death_risk_reduce": 0.3})

    assert np.allclose(ve0(model, INF_SEASON), 0.5)
    assert np.allclose(ve0(model, HOSP_SEASON), 0.4)
    assert np.allclose(ve0(model, DEATH_SEASON), 0.3)


def test_VE_initial_equals_input_when_adjustment_disabled():
    """
    `adjust_VE_for_seasonal_waning = False` skips the derivation entirely,
    so the derived values are the input values -- even though waning is
    nonzero and would otherwise inflate them.
    """
    updates = {"vax_induced_immune_wane": 0.01,
               "vax_induced_inf_risk_reduce": 0.5,
               "vax_induced_hosp_risk_reduce": 0.4,
               "vax_induced_death_risk_reduce": 0.3}

    enabled = _make_model(updates)

    disabled = _make_model(updates)
    disabled.model_config.ve_derivation["adjust_VE_for_seasonal_waning"] = False
    disabled.update_ve_inflation_factors()

    assert np.allclose(ve0(disabled, INF_SEASON), 0.5)
    assert np.allclose(ve0(disabled, HOSP_SEASON), 0.4)
    assert np.allclose(ve0(disabled, DEATH_SEASON), 0.3)

    # Sanity check: with the adjustment on, waning does move the values,
    #   so the assertions above are not passing trivially
    assert not np.allclose(ve0(enabled, INF_SEASON), 0.5)


def test_VE_initial_matches_closed_form_for_single_dose_day():
    """
    When the whole season's doses land on a single day, the dose-timing
    profile collapses to p_prot = [1] and T = 1, so the general formula

        VE_0 = VE_season * w * T / ((1 - exp(-w)) * S)

    reduces to the closed form VE_0 = VE_season * w / (1 - exp(-w)) -- the
    pure within-day continuous-averaging correction, with no across-day
    waning. This pins the formula against a value checkable by hand.
    """
    wane = 0.01
    ve_season = 0.5

    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    # One dose day, then zeros -- zero-dose days are excluded from the
    #   window, so only the single dose day defines the profile
    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 200, 0.0, A, R)
    df.loc[0, "daily_vaccines"] = json.dumps([[0.01] * R] * A)

    model = _make_model({"vax_induced_immune_wane": wane,
                         "vax_induced_inf_risk_reduce": ve_season,
                         "vax_protection_delay_days": 0,
                         "vax_immunity_reset_date_mm_dd": None},
                        vaccines_df=df)

    expected = ve_season * wane / (1 - np.exp(-wane))

    assert np.allclose(ve0(model, INF_SEASON), expected)


def test_VE_initial_exceeds_input_and_grows_with_waning():
    """
    Correcting for waning always raises the peak above the season-average
    value, and the longer immunity has been waning by the end of the
    season, the larger that correction has to be.
    """
    ve_season = 0.5
    initials = []

    for wane in (0.002, 0.004, 0.008):
        model = _make_model({"vax_induced_immune_wane": wane,
                             "vax_induced_inf_risk_reduce": ve_season})
        initials.append(float(ve0(model, INF_SEASON)[0, 0]))

    assert all(v > ve_season for v in initials)
    assert initials[0] < initials[1] < initials[2]


def test_VE_initial_is_capped_at_one_with_warning():
    """
    A high season-average efficacy combined with fast waning implies a peak
    efficacy above 1, which is not a probability and would make
    `1 - MV * VE_0` negative at high coverage. It must be capped at 1 and
    must warn, since capping means the requested season-average efficacy is
    not actually achievable.
    """
    with pytest.warns(UserWarning, match="exceeds 1.0"):
        model = _make_model({"vax_induced_immune_wane": 0.02,
                             "vax_induced_inf_risk_reduce": 0.9})

    initial = ve0(model, INF_SEASON)

    assert np.all(initial <= 1.0)
    assert np.allclose(initial, 1.0)

    # The cap is applied where the product is formed, not baked into the
    #   stored factor, so lowering the season-average efficacy afterwards
    #   must bring the applied value back down below 1
    model.params.params[INF_SEASON] = 0.3
    assert np.all(ve0(model, INF_SEASON) < 1.0)


def test_VE_initial_not_capped_for_reasonable_values():
    """A plausible efficacy/waning combination must not hit the cap."""
    model = _make_model({"vax_induced_immune_wane": 0.004,
                         "vax_induced_inf_risk_reduce": 0.5})

    initial = ve0(model, INF_SEASON)

    assert np.all(initial < 1.0)
    assert np.all(initial > 0.5)


def test_dose_window_quantile_no_effect_on_flat_schedule():
    """
    Trimming removes sparse dose tails. A schedule with a flat dose profile
    has no tails to trim, so the quantile must not change the result
    materially.
    """
    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 180, 0.001, A, R)

    updates = {"vax_induced_immune_wane": 0.004,
               "vax_induced_inf_risk_reduce": 0.5,
               "vax_immunity_reset_date_mm_dd": None}

    untrimmed = _make_model(updates, vaccines_df=df)
    trimmed = _make_model(updates, vaccines_df=df)
    trimmed.model_config.ve_derivation["VE_season_dose_window_quantile"] = 0.05
    trimmed.update_ve_inflation_factors()

    assert np.allclose(ve0(untrimmed, INF_SEASON),
                       ve0(trimmed, INF_SEASON),
                       rtol=0.05)


def test_dose_window_quantile_lowers_VE_initial_on_campaign_schedule():
    """
    On a campaign-shaped schedule (a burst of doses, then a long low-dose
    tail), the untrimmed unweighted season average is stretched over months
    with almost no vaccination, inflating the derived peak. Trimming the
    tail must bring the derived peak back down.
    """
    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    df = _campaign_vaccines_df(datetime.date(2022, 8, 8), 300,
                               campaign_days=30, campaign_dose=0.01,
                               tail_dose=0.00001,
                               num_age_groups=A, num_risk_groups=R)

    updates = {"vax_induced_immune_wane": 0.004,
               "vax_induced_inf_risk_reduce": 0.4,
               "vax_immunity_reset_date_mm_dd": None}

    untrimmed = _make_model(updates, vaccines_df=df)
    trimmed = _make_model(updates, vaccines_df=df)
    trimmed.model_config.ve_derivation["VE_season_dose_window_quantile"] = 0.10
    trimmed.update_ve_inflation_factors()

    assert np.all(ve0(trimmed, INF_SEASON)
                  < ve0(untrimmed, INF_SEASON))


def test_VE_initial_recomputed_on_reset():
    """
    The derived peak efficacy must track post-construction parameter
    changes: `reset_simulation` re-runs the derivation, so a model reset
    after a waning-rate override must not keep the construction-time value.
    """
    model = _make_model({"vax_induced_immune_wane": 0.002,
                         "vax_induced_inf_risk_reduce": 0.5})
    before = ve0(model, INF_SEASON).copy()

    model.params.params["vax_induced_immune_wane"] = 0.008
    model.reset_simulation()
    after = ve0(model, INF_SEASON)

    assert np.all(after > before)


def test_ve_derivation_rejects_unknown_season_param():
    """
    Every name in `fields` must be a declared param -- it is the
    season-average efficacy a rate config will name in `vax_reduce_param`.
    """
    with pytest.raises(ValueError, match="not in model params"):
        validate_ve_derivation_config(
            {"daily_vaccines_schedule": "daily_vaccines",
             "vax_induced_immune_wane_param": "wane",
             "fields": ["nope"]},
            param_names={"wane", "ve"},
            schedule_names={"daily_vaccines"},
        )


def test_ve_derivation_rejects_non_string_field():
    with pytest.raises(ValueError, match="a string"):
        validate_ve_derivation_config(
            {"daily_vaccines_schedule": "daily_vaccines",
             "vax_induced_immune_wane_param": "wane",
             "fields": [{"season_average_param": "ve"}]},
            param_names={"wane", "ve"},
            schedule_names={"daily_vaccines"},
        )


def test_ve_derivation_rejects_out_of_range_quantile():
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        validate_ve_derivation_config(
            {"daily_vaccines_schedule": "daily_vaccines",
             "vax_induced_immune_wane_param": "wane",
             "VE_season_dose_window_quantile": 0.6,
             "fields": ["ve"]},
            param_names={"wane", "ve"},
            schedule_names={"daily_vaccines"},
        )


# ---------------------------------------------------------------------------
# resolve_mm_dd_near_date
# ---------------------------------------------------------------------------

def test_resolve_mm_dd_picks_nearest_year_across_new_year():
    """
    For a season starting in August, "01_01" means the following January,
    not the January 7 months before the simulation start.
    """
    resolved = resolve_mm_dd_near_date(
        "01_01", datetime.date(2024, 8, 15), warn_days=None)
    assert resolved == datetime.date(2025, 1, 1)


def test_resolve_mm_dd_picks_previous_year_when_nearer():
    resolved = resolve_mm_dd_near_date(
        "12_01", datetime.date(2025, 1, 15), warn_days=None)
    assert resolved == datetime.date(2024, 12, 1)


def test_resolve_mm_dd_same_year_when_nearest():
    resolved = resolve_mm_dd_near_date(
        "09_01", datetime.date(2024, 8, 15), warn_days=None)
    assert resolved == datetime.date(2024, 9, 1)


def test_resolve_mm_dd_warns_when_far_from_reference():
    with pytest.warns(UserWarning, match="days from"):
        resolve_mm_dd_near_date("02_15", datetime.date(2024, 8, 15))


def test_resolve_mm_dd_handles_feb_29_in_non_leap_year():
    """
    2023 has no Feb 29, so the candidate for that year is skipped rather
    than raising -- the nearest real Feb 29 is chosen instead.
    """
    resolved = resolve_mm_dd_near_date(
        "02_29", datetime.date(2023, 12, 1), warn_days=None)
    assert resolved == datetime.date(2024, 2, 29)


# ---------------------------------------------------------------------------
# Infection-induced immunity initial value
# ---------------------------------------------------------------------------

# Wires the optional start-date key onto the M metric — the shipped test
# config leaves it out, since it is opt-in.
_M_START_DATE_CONFIG = {
    "M": {"infection_immunity_start_date_mm_dd_param":
              "infection_immunity_start_date_mm_dd"}
}


def _M_init(A, R, value=0.3):
    return np.full((A, R), value)


def test_M_initial_unchanged_when_start_date_absent():
    """Without the start-date param, M(0) is used exactly as supplied."""
    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    model = _make_model(epi_metric_init={"M": M0, "MV": np.zeros((A, R))})

    assert np.allclose(model.epi_metrics["M"].init_val, M0)
    assert model.epi_metrics["M"].pending_injection_date is None


def test_M_initial_decayed_forward_from_past_start_date():
    """
    A start date before the simulation start means M(0) is a past
    measurement: it must be decayed forward by waning only, so the adjusted
    value is strictly smaller than the input.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    start = _start_date(settings)
    past = start - datetime.timedelta(days=30)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{past.month:02d}_{past.day:02d}",
         "inf_induced_immune_wane": 0.01},
        epi_metric_init={"M": M0, "MV": np.zeros((A, R))},
        metric_config_updates=_M_START_DATE_CONFIG)

    adjusted = model.epi_metrics["M"].init_val

    assert np.all(adjusted < M0)
    assert model.epi_metrics["M"].pending_injection_date is None

    # Decay is (1 - wane/ts_per_day) applied ts_per_day * num_days times
    ts = model.simulation_settings.timesteps_per_day
    expected = M0 * (1 - 0.01 / ts) ** (ts * 30)
    assert np.allclose(adjusted, expected)


def test_M_initial_deferred_to_future_start_date_then_injected():
    """
    A start date after the simulation start means M(0) is a future
    measurement: M starts at zero and the input value is added once, on
    that date.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    start = _start_date(settings)
    future = start + datetime.timedelta(days=10)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}",
         # no waning, so the only change to M is the injection itself
         "inf_induced_immune_wane": 0.0,
         "inf_induced_saturation": 0.0,
         "R_to_S_rate": 0.0},
        epi_metric_init={"M": M0, "MV": np.zeros((A, R))},
        settings_updates={"save_daily_history": True},
        metric_config_updates=_M_START_DATE_CONFIG)

    assert np.allclose(model.epi_metrics["M"].init_val, 0.0)
    assert model.epi_metrics["M"].pending_injection_date == future

    # The pending value is mirrored onto params for the torch path
    assert np.allclose(
        model.params.params[injection_val_param_name("M")], M0)

    model.simulate_until_day(20)

    history = np.asarray([np.asarray(h) for h in model.epi_metrics["M"].history_vals_list])

    # Day index 9 is the end of the 10th simulated day, i.e. still before the
    #   injection; by the end of the injection day M carries the full value.
    assert np.allclose(history[8], 0.0)
    assert np.allclose(history[-1], M0)


def test_M_injection_applies_only_once():
    """The injection is a one-shot event, not a daily top-up."""
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    future = _start_date(settings) + datetime.timedelta(days=5)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}",
         "inf_induced_immune_wane": 0.0,
         "inf_induced_saturation": 0.0,
         "R_to_S_rate": 0.0},
        epi_metric_init={"M": M0, "MV": np.zeros((A, R))},
        settings_updates={"save_daily_history": True},
        metric_config_updates=_M_START_DATE_CONFIG)

    model.simulate_until_day(30)

    final = np.asarray(model.epi_metrics["M"].current_val)

    assert np.allclose(final, M0)
    assert model.epi_metrics["M"].pending_injection_date is None


def test_M_initial_recomputed_on_reset():
    """
    `reset_simulation` re-derives M(0) from `original_init_val`, so the
    deferred injection is armed again and adjustments do not compound.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    future = _start_date(settings) + datetime.timedelta(days=5)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}",
         "inf_induced_immune_wane": 0.0,
         "inf_induced_saturation": 0.0,
         "R_to_S_rate": 0.0},
        epi_metric_init={"M": M0, "MV": np.zeros((A, R))},
        settings_updates={"save_daily_history": True},
        metric_config_updates=_M_START_DATE_CONFIG)

    model.simulate_until_day(10)
    assert model.epi_metrics["M"].pending_injection_date is None

    model.reset_simulation()

    assert model.epi_metrics["M"].pending_injection_date == future
    assert np.allclose(model.epi_metrics["M"].init_val, 0.0)
    assert np.allclose(model.epi_metrics["M"].original_init_val, M0)


def test_decayed_M_initial_does_not_compound_across_resets():
    """
    The decay-forward path re-derives from `original_init_val`, so
    resetting repeatedly must not decay the value again and again.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    M0 = _M_init(A, R)

    past = _start_date(settings) - datetime.timedelta(days=30)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{past.month:02d}_{past.day:02d}",
         "inf_induced_immune_wane": 0.01},
        epi_metric_init={"M": M0, "MV": np.zeros((A, R))},
        metric_config_updates=_M_START_DATE_CONFIG)

    first = np.asarray(model.epi_metrics["M"].init_val).copy()

    model.reset_simulation()
    model.reset_simulation()

    assert np.allclose(model.epi_metrics["M"].init_val, first)


# ---------------------------------------------------------------------------
# Vaccine-induced immunity initial value
# ---------------------------------------------------------------------------

def test_MV_initial_counts_doses_in_protection_delay_window():
    """
    Regression test for the double-counted protection delay.

    The `daily_vaccines` schedule is indexed by PROTECTION date -- the
    schedule template has already shifted every date forward by
    `vax_protection_delay_days`. Masking from `reset_date + delay` would
    drop the doses that become protective during the delay window just
    after the reset date, so MV(0) would be too small.

    With doses present only in that window, a delay-shifted mask yields
    MV(0) == 0 while the correct mask yields MV(0) > 0.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    start = _start_date(settings)
    reset_date = start - datetime.timedelta(days=40)
    delay = 14

    # Doses given only in the [reset_date, reset_date + delay) protection
    #   window -- exactly the doses the buggy mask discarded
    dates = [reset_date + datetime.timedelta(days=i) for i in range(delay - 1)]
    arr = json.dumps([[0.01] * R] * A)
    df = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates],
                       "daily_vaccines": [arr] * len(dates)})

    with pytest.warns(UserWarning, match="Vaccine immunity reset date"):
        model = _make_model(
            {"vax_immunity_reset_date_mm_dd":
                 f"{reset_date.month:02d}_{reset_date.day:02d}",
             "vax_protection_delay_days": delay,
             "vax_induced_immune_wane": 0.004},
            vaccines_df=df,
            epi_metric_init={"M": np.zeros((A, R)), "MV": np.zeros((A, R))})

    assert np.all(np.asarray(model.epi_metrics["MV"].init_val) > 0)


def test_MV_initial_does_not_compound_across_resets():
    """
    `reset_simulation` re-derives MV(0) from `original_init_val`, so the
    pre-simulation dose adjustment is not added twice.
    """
    state, params, _, settings, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups

    with pytest.warns(UserWarning, match="Vaccine immunity reset date"):
        model = _make_model(
            epi_metric_init={"M": np.zeros((A, R)), "MV": np.zeros((A, R))})

    first = np.asarray(model.epi_metrics["MV"].init_val).copy()

    model.reset_simulation()
    model.reset_simulation()

    assert np.allclose(model.epi_metrics["MV"].init_val, first)


# ---------------------------------------------------------------------------
# VE mechanism actually bites
# ---------------------------------------------------------------------------

def test_vaccine_protection_reduces_infections():
    """
    End-to-end check that the multiplicative `1 - MV * VE_0` factor is
    wired into the force of infection: raising the season-average efficacy
    must leave more people susceptible after a run.
    """
    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 300, 0.005, A, R)

    finals = []
    for ve in (0.0, 0.7):
        model = _make_model({"vax_induced_inf_risk_reduce": ve,
                             "vax_induced_hosp_risk_reduce": ve,
                             "vax_induced_death_risk_reduce": ve,
                             "vax_immunity_reset_date_mm_dd": None},
                            vaccines_df=df,
                            settings_updates={"save_daily_history": True})
        model.simulate_until_day(120)
        finals.append(float(np.sum(model.compartments["S"].current_val)))

    assert finals[1] > finals[0]


def test_vaccine_factor_multiplies_severe_outcome_probability():
    """
    The severe-outcome splits apply the vaccine factor to the PROBABILITY,
    not to the rate, so it does NOT cancel out of the complement branch.

    With vaccine immunity present, the direct branch (e.g. IP->ISH) must
    fall and the complement branch (IP->ISR) must rise by exactly the same
    amount -- the two must still sum to the base rate, which is what makes
    this a redistribution between branches rather than a change in the
    total outflow. Checked on both backends.
    """
    torch = pytest.importorskip("torch")

    from generic_core.rate_templates import ImmunityModulatedRate

    template = ImmunityModulatedRate()
    A, R = 2, 1

    base_rate, proportion, ve_season = 1.1, 0.4, 0.6
    MV = np.full((A, R), 0.5)

    # No ve_derivation block here, so no inflation factor is stored and the
    #   season-average value is used directly as the peak -- which is exactly
    #   what `adjust_VE_for_seasonal_waning = false` means. That keeps this
    #   test focused on the branch arithmetic.
    param_vals = {"IP_to_IS_rate": base_rate,
                  "IP_to_ISH_prop": proportion,
                  "inf_induced_hosp_risk_reduce": 0.0,
                  "vax_induced_hosp_risk_reduce": ve_season}

    params = GenericSubpopParams(
        params=param_vals,
        num_age_groups=A,
        num_risk_groups=R,
        total_pop_age_risk=np.full((A, R), 1000.0),
    )

    rate_config = {"base_rate": "IP_to_IS_rate",
                   "proportion": "IP_to_ISH_prop",
                   "inf_reduce_param": "inf_induced_hosp_risk_reduce",
                   "vax_reduce_param": "vax_induced_hosp_risk_reduce"}

    class _State:
        epi_metrics = {"M": np.zeros((A, R)), "MV": MV}

    class _StateNoVax:
        epi_metrics = {"M": np.zeros((A, R)), "MV": np.zeros((A, R))}

    direct = template.numpy_rate(_State, params, {**rate_config, "is_complement": False})
    complement = template.numpy_rate(_State, params, {**rate_config, "is_complement": True})

    direct0 = template.numpy_rate(_StateNoVax, params, {**rate_config, "is_complement": False})
    complement0 = template.numpy_rate(_StateNoVax, params, {**rate_config, "is_complement": True})

    # Vaccination redistributes flow away from the severe branch ...
    assert np.all(direct < direct0)
    assert np.all(complement > complement0)
    # ... without changing the total outflow from the origin compartment
    assert np.allclose(direct + complement, base_rate)
    assert np.allclose(direct0 + complement0, base_rate)

    # Exact expected value: prob = (prop / immunity_force) * (1 - MV * VE_0),
    #   with immunity_force = 1 here since inf_reduce_param is 0
    expected_prob = proportion * (1.0 - MV * ve_season)
    assert np.allclose(direct, expected_prob * base_rate)

    # torch path must agree
    state_dict = {"M": torch.zeros((A, R), dtype=torch.float64),
                  "MV": torch.tensor(MV)}
    params_dict = {k: torch.tensor(float(v)) for k, v in param_vals.items()}

    torch_direct = template.torch_rate(
        state_dict, params_dict, {**rate_config, "is_complement": False})
    torch_complement = template.torch_rate(
        state_dict, params_dict, {**rate_config, "is_complement": True})

    assert np.allclose(torch_direct.numpy(), direct)
    assert np.allclose(torch_complement.numpy(), complement)


def test_season_average_efficacy_override_after_reset_takes_effect():
    """
    Regression: overriding a season-average efficacy AFTER reset_simulation
    must change the trajectory.

    This is the ordering every override path in the repo uses --
    `fitting._reuse_simulate` and `calibration.run_accept_reject` both call
    reset_simulation() and only then apply the draw's parameters. When the
    peak efficacy was precomputed onto params during reset, such an override
    could not reach the rates and fitting these params was a silent no-op.
    Storing the inflation factor instead keeps the season-average value live.
    """
    _, params, _, _, _ = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    df = _constant_vaccines_df(datetime.date(2022, 8, 8), 300, 0.005, A, R)

    base_updates = {"vax_induced_inf_risk_reduce": 0.0,
                    "vax_induced_hosp_risk_reduce": 0.0,
                    "vax_induced_death_risk_reduce": 0.0,
                    "vax_immunity_reset_date_mm_dd": None}

    model = _make_model(base_updates, vaccines_df=df,
                        settings_updates={"save_daily_history": True})
    model.simulate_until_day(120)
    unprotected = float(np.sum(model.compartments["S"].current_val))

    # Exactly the reset-then-override sequence the fitting reuse path uses
    model.reset_simulation()
    model.params.params["vax_induced_inf_risk_reduce"] = 0.7
    model.simulate_until_day(120)
    protected = float(np.sum(model.compartments["S"].current_val))

    assert protected > unprotected


def test_reset_reproduces_identical_run():
    """A run, a reset, and a second run must produce identical trajectories."""
    model = _make_model(settings_updates={"save_daily_history": True})

    model.simulate_until_day(60)
    first = copy.deepcopy(
        [np.asarray(h) for h in model.compartments["S"].history_vals_list])

    model.reset_simulation()
    model.RNG = np.random.Generator(np.random.MT19937(RNG_SEED))
    model.simulate_until_day(60)
    second = [np.asarray(h) for h in model.compartments["S"].history_vals_list]

    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# numpy / torch agreement
# ---------------------------------------------------------------------------

METAPOP_CONFIG_PATH = BASE_PATH / "caseb_flu_generic_metapop_config.json"


def _make_generic_metapop(settings, param_updates=None, metric_config_updates=None):
    """
    Build a two-subpopulation ConfigDrivenMetapopModel on the caseB fixtures,
    mirroring tests/test_generic_torch.py's factory.
    """
    import torch  # noqa: F401  (ensures torch is importable before use)

    from generic_core.generic_metapop import ConfigDrivenMetapopModel

    state1, params1, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, _, _, _ = subpop_inputs("caseB_subpop2")

    model_config = parse_model_config(METAPOP_CONFIG_PATH, schedules_input=schedules_info)

    if param_updates:
        model_config.params.update(param_updates)
        model_config.param_names |= set(param_updates)

    for metric_name, extra in (metric_config_updates or {}).items():
        mc = next(m for m in model_config.epi_metrics if m.name == metric_name)
        mc.update_config = {**mc.update_config, **extra}

    A, R = params1.num_age_groups, params1.num_risk_groups

    def _make_subpop(flu_state, rng_bitgen, name):
        compartment_init = {c: getattr(flu_state, c) for c in COMPARTMENTS}
        epi_metric_init = {"M": np.asarray(flu_state.M, dtype=float),
                           "MV": np.asarray(flu_state.MV, dtype=float)}
        state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
        params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)
        return ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=state_init,
            params=params,
            simulation_settings=settings,
            RNG=np.random.Generator(rng_bitgen),
            schedules_input=schedules_info,
            name=name,
        )

    bit_gen1 = np.random.MT19937(88888)
    bit_gen2 = bit_gen1.jumped(1)

    s_to_e_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_E")
    travel_config = s_to_e_tc.rate_config["travel_config"]

    metapop = ConfigDrivenMetapopModel(
        subpop_models=[_make_subpop(state1, bit_gen1, "subpop1"),
                       _make_subpop(state2, bit_gen2, "subpop2")],
        mixing_params=mixing_params,
        model_config=model_config,
        travel_config=travel_config,
    )
    return metapop, model_config


def _run_torch(metapop, model_config, num_days, start):
    """
    Run the generic torch path and return its state history dict.

    NOTE: build_generic_torch_inputs reads the metapop model's CURRENT state,
    so this must be called before the numpy model is simulated — otherwise
    torch starts from the numpy run's final state rather than its initial one.
    """
    from generic_core.torch_generic import (
        build_generic_torch_inputs,
        generic_torch_simulate_full_history,
    )
    from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY

    torch_inputs = build_generic_torch_inputs(
        metapop, model_config, num_days, requires_grad=False)

    history, _ = generic_torch_simulate_full_history(
        torch_inputs["state_dict"],
        torch_inputs["params_dict"],
        model_config,
        RATE_TEMPLATE_REGISTRY,
        torch_inputs["precomputed"],
        torch_inputs["schedules_dict"],
        num_days,
        timesteps_per_day=1,
        start_real_date=start,
    )
    return history, torch_inputs


def _torch_settings():
    _s, _p, _mp, settings, _sched = subpop_inputs("caseB_subpop1")
    return clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND,
        "timesteps_per_day": 1,
        "use_deterministic_softplus": True,
        "save_daily_history": True,
    })


def test_derived_params_reach_torch_params_dict():
    """
    The derived peak-efficacy and M-injection values are computed onto each
    subpop's own params, not onto model_config.params, so they would be
    missing from the torch params dict unless build_params_dict stacks them
    explicitly. Without them the torch rate templates would KeyError or
    silently drop the vaccine factor.
    """
    num_days = 20
    settings = _torch_settings()
    start = _start_date(settings)
    future = start + datetime.timedelta(days=7)

    metapop, model_config = _make_generic_metapop(
        settings,
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}"},
        _M_START_DATE_CONFIG)

    _history, torch_inputs = _run_torch(metapop, model_config, num_days, start)
    params_dict = torch_inputs["params_dict"]

    subpops = list(metapop._subpop_models_ordered.values())
    L = len(subpops)
    A = subpops[0].params.num_age_groups
    R = subpops[0].params.num_risk_groups

    for name in (ve_inflation_param_name(INF_SEASON),
                 ve_inflation_param_name(HOSP_SEASON),
                 ve_inflation_param_name(DEATH_SEASON),
                 injection_val_param_name("M")):
        assert name in params_dict, f"{name} missing from torch params dict"
        assert tuple(params_dict[name].shape) == (L, A, R)

        # Each subpop's own derived value must land in its own slice
        for ix, subpop in enumerate(subpops):
            expected = np.broadcast_to(
                np.asarray(subpop.params.params[name], dtype=float), (A, R))
            assert np.allclose(params_dict[name][ix].detach().numpy(), expected)


def test_torch_applies_M_injection_on_its_date_only():
    """
    The torch path has no epi metric objects, so the deferred M(0) has to be
    applied by check_and_apply_M_injection reading the mirrored params value.
    M must stay near zero until the injection date, then jump by the input
    M(0), and must not jump again.
    """
    num_days = 30
    settings = _torch_settings()
    start = _start_date(settings)
    future = start + datetime.timedelta(days=10)

    M0_value = 0.3
    _s, params, _mp, _set, _sched = subpop_inputs("caseB_subpop1")
    A, R = params.num_age_groups, params.num_risk_groups

    metapop, model_config = _make_generic_metapop(
        settings,
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}",
         # freeze every other channel into M so the injection is the only change
         "inf_induced_immune_wane": 0.0,
         "inf_induced_saturation": 0.0,
         "R_to_S_rate": 0.0},
        _M_START_DATE_CONFIG)

    # Override M(0) on both subpops, then re-derive the mirrored injection value
    for subpop in metapop._subpop_models_ordered.values():
        M = subpop.epi_metrics["M"]
        M.init_val = M.adjust_initial_value(
            np.full((A, R), M0_value),
            subpop.start_real_date,
            subpop.params,
            subpop.simulation_settings.timesteps_per_day,
        )
        subpop.update_infection_immunity_injection_val()
        subpop.state.sync_to_current_vals(subpop.epi_metrics)

    history, _ = _run_torch(metapop, model_config, num_days, start)

    M_history = [h[0].detach().numpy() for h in history["M"]]
    days_before = (future - start).days - 1

    assert np.allclose(M_history[days_before], 0.0, atol=1e-9)
    assert np.allclose(M_history[days_before + 1], M0_value, atol=1e-9)
    # One-shot: no further growth over the rest of the run
    assert np.allclose(M_history[-1], M0_value, atol=1e-9)


@pytest.mark.parametrize("offset_days", [None, -20, 0, 10])
def test_torch_M_start_matches_numpy_for_every_start_date_case(offset_days):
    """
    The torch path must start from the same M(0) as numpy in all four
    cases, and must inject only in the deferred one.

    `pending_injection_date is None` covers three quite different
    situations -- no start date, a start date on the simulation start, and a
    PAST start date whose M(0) was decayed forward. In all three the
    mirrored injection param is zeros and torch must not add anything; the
    adjusted M(0) reaches torch through the state tensors instead. Only a
    FUTURE start date defers, and only then may torch inject.
    """
    pytest.importorskip("torch")

    num_days = 30
    settings = _torch_settings()
    start = _start_date(settings)

    if offset_days is None:
        mm_dd = None
    else:
        d = start + datetime.timedelta(days=offset_days)
        mm_dd = f"{d.month:02d}_{d.day:02d}"

    M0_value = 0.3
    _s, params, _mp, _set, _sched = subpop_inputs("caseB_subpop1")
    A, R = params.num_age_groups, params.num_risk_groups

    metapop, model_config = _make_generic_metapop(
        settings,
        {"infection_immunity_start_date_mm_dd": mm_dd,
         # freeze every other channel into M so only the start-date
         #   handling can move it
         "inf_induced_immune_wane": 0.01,
         "inf_induced_saturation": 0.0,
         "R_to_S_rate": 0.0},
        _M_START_DATE_CONFIG)

    for subpop in metapop._subpop_models_ordered.values():
        M = subpop.epi_metrics["M"]
        M.init_val = M.adjust_initial_value(
            np.full((A, R), M0_value),
            subpop.start_real_date,
            subpop.params,
            subpop.simulation_settings.timesteps_per_day,
        )
        subpop.update_infection_immunity_injection_val()
        subpop.state.sync_to_current_vals(subpop.epi_metrics)

    subpop0 = list(metapop._subpop_models_ordered.values())[0]
    numpy_init = np.asarray(subpop0.epi_metrics["M"].init_val, dtype=float)
    deferred = offset_days is not None and offset_days > 0

    history, torch_inputs = _run_torch(metapop, model_config, num_days, start)

    # Torch's starting tensor must be the ADJUSTED value, not the raw input
    assert np.allclose(
        torch_inputs["state_dict"]["M"][0].detach().numpy(), numpy_init)

    injection = torch_inputs["params_dict"][injection_val_param_name("M")]
    if deferred:
        assert np.allclose(numpy_init, 0.0)
        assert np.allclose(injection[0].detach().numpy(), M0_value)
    else:
        assert np.allclose(injection[0].detach().numpy(), 0.0)

    # Only the deferred case may show M rising; the others decay monotonically
    M_hist = [h[0].detach().numpy() for h in history["M"]]
    rises = any(np.any(M_hist[d + 1] > M_hist[d] + 1e-12)
                for d in range(len(M_hist) - 1))
    assert rises == deferred


def test_M_pending_injection_is_consumed_then_restored_by_reset():
    """
    The deferred injection must fire exactly once per run: the metric clears
    `pending_injection_date` when it fires, the mirrored params value is
    zeroed to match, and `reset_simulation` restores both so the next run
    injects again.
    """
    _s, params, _mp, settings, _sched = subpop_inputs("caseA")
    A, R = params.num_age_groups, params.num_risk_groups
    start = _start_date(settings)
    future = start + datetime.timedelta(days=10)

    model = _make_model(
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}"},
        metric_config_updates=_M_START_DATE_CONFIG,
        epi_metric_init={"M": _M_init(A, R), "MV": np.zeros((A, R))},
        settings_updates={"save_daily_history": True})

    M = model.epi_metrics["M"]
    assert M.pending_injection_date == future

    # Before the injection date the mirror still advertises it
    model.simulate_until_day((future - start).days - 1)
    assert M.pending_injection_date == future
    assert np.allclose(model.params.params[injection_val_param_name("M")],
                       np.asarray(M.original_init_val, dtype=float))

    model.simulate_until_day(30)
    assert M.pending_injection_date is None, "injection must be consumed"

    # ... and once spent, the mirror must be zeroed to match. A mirror left
    #   saying "pending" would make build_generic_torch_inputs apply the same
    #   injection a second time.
    assert np.allclose(model.params.params[injection_val_param_name("M")], 0.0), \
        "spent injection must be cleared from the params mirror"

    model.reset_simulation()
    assert M.pending_injection_date == future, "reset must restore the pending date"
    assert np.allclose(
        model.params.params[injection_val_param_name("M")],
        np.asarray(M.original_init_val, dtype=float))


def test_torch_vaccine_protection_reduces_infections():
    """
    End-to-end check that the multiplicative vaccine factor is wired into
    the torch force of infection too — raising the season-average efficacy
    must leave more people susceptible in the torch run.
    """
    num_days = 60
    settings = _torch_settings()
    start = _start_date(settings)

    finals = []
    for ve in (0.0, 0.7):
        metapop, model_config = _make_generic_metapop(
            settings,
            {"vax_induced_inf_risk_reduce": ve,
             "vax_induced_hosp_risk_reduce": ve,
             "vax_induced_death_risk_reduce": ve})
        history, _ = _run_torch(metapop, model_config, num_days, start)
        finals.append(float(history["S"][-1].sum()))

    assert finals[1] > finals[0]


def test_numpy_and_torch_agree_with_VE_and_M_injection():
    """
    The numpy and torch paths must produce the same trajectories with the new
    VE model active, a pending M injection, and the MV reset in play.

    This is the check that catches a derived param that never reached the
    torch params dict, a missing `sync_to_current_vals`, or a torch/numpy
    formula divergence — all of which would leave the two paths silently
    disagreeing rather than erroring.
    """
    num_days = 40
    settings = _torch_settings()
    start = _start_date(settings)
    future = start + datetime.timedelta(days=7)

    metapop, model_config = _make_generic_metapop(
        settings,
        {"infection_immunity_start_date_mm_dd": f"{future.month:02d}_{future.day:02d}"},
        _M_START_DATE_CONFIG)

    subpops = list(metapop._subpop_models_ordered.values())

    # Sanity: the injection really is pending, so the test is not vacuous
    assert subpops[0].epi_metrics["M"].pending_injection_date == future

    # Torch inputs must be built from the un-simulated model — see _run_torch
    history, _ = _run_torch(metapop, model_config, num_days, start)

    metapop.simulate_until_day(num_days)

    for subpop_ix, subpop in enumerate(subpops):
        for name in COMPARTMENTS + ["M", "MV"]:
            lookup = (subpop.compartments if name in subpop.compartments
                      else subpop.epi_metrics)
            numpy_final = np.asarray(lookup[name].history_vals_list[-1], dtype=float)
            torch_final = history[name][-1][subpop_ix].detach().numpy()
            assert np.allclose(numpy_final, torch_final, rtol=1e-10, atol=1e-8), \
                (f"{name} (subpop {subpop_ix}) diverged between numpy and torch: "
                 f"max diff {np.max(np.abs(numpy_final - torch_final))}")


def test_season_average_efficacy_is_a_torch_gradient_leaf():
    """
    The torch path must route gradients to the SEASON-AVERAGE efficacy
    param.

    When the finished peak efficacy was precomputed onto params, the value
    the rates read was a non-trainable constant and the gradient w.r.t. the
    season-average param was identically zero -- so gradient-based fitting
    of vaccine efficacy silently did nothing. Storing only the inflation
    factor keeps the efficacy itself in the graph.
    """
    torch = pytest.importorskip("torch")

    from generic_core.torch_generic import (
        build_generic_torch_inputs,
        generic_torch_simulate_full_history,
    )
    from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY

    num_days = 10
    settings = _torch_settings()
    metapop, model_config = _make_generic_metapop(settings)

    torch_inputs = build_generic_torch_inputs(
        metapop, model_config, num_days, requires_grad=True)

    ve = torch_inputs["params_dict"][INF_SEASON]
    assert ve.requires_grad, "season-average efficacy must be a trainable leaf"

    history, _ = generic_torch_simulate_full_history(
        torch_inputs["state_dict"],
        torch_inputs["params_dict"],
        model_config,
        RATE_TEMPLATE_REGISTRY,
        torch_inputs["precomputed"],
        torch_inputs["schedules_dict"],
        num_days,
        timesteps_per_day=1,
        start_real_date=metapop.subpop_models["subpop1"].start_real_date,
    )

    torch.stack([history["S"][d] for d in range(num_days)]).sum().backward()

    assert ve.grad is not None
    assert torch.any(ve.grad != 0), "gradient must actually reach the efficacy"
