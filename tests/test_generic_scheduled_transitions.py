"""
Tests for the scheduled-transition / vaccination feature (commit 550fc87) and
a regression test for the optional vax_induced_saturation_param on the torch path.

Covers:
  1. ScheduledTransferVariable.get_scheduled_exact_realization() unit behaviour:
     first-timestep-of-day gating, proportion->count rounding, and capping at
     the origin population.
  2. ScheduledTransferVariable.reset() restores the within-day timestep counter.
  3. Config-parser rejection of scheduled_exact inside transition groups and in
     jointly_distributed_with (these constraints are coded in config_parser.py).
  4. Regression: an infection-induced-immunity model WITHOUT
     vax_induced_saturation_param must run on the torch path without raising
     KeyError (previously torch_generic.generic_advance_timestep accessed the
     optional key unconditionally).
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import clt_toolkit as clt

from generic_core.generic_model import ScheduledTransferVariable
from generic_core.config_parser import parse_model_config_from_dict

sys.path.insert(0, str(Path(__file__).parent))

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
CASEB_CONFIG_PATH = BASE_PATH / "caseb_flu_generic_metapop_config.json"


# ---------------------------------------------------------------------------
# 1 + 2. ScheduledTransferVariable unit behaviour
# ---------------------------------------------------------------------------

def _make_stv(origin_val, rate):
    """Build a ScheduledTransferVariable with a fixed origin population and rate."""
    origin = clt.Compartment(np.asarray(origin_val, dtype=float))
    destination = clt.Compartment(np.zeros_like(np.asarray(origin_val, dtype=float)))
    stv = ScheduledTransferVariable(origin, destination, schedule_name="vax_sched")
    # current_rate is the daily proportion of the origin compartment to move
    # (set by the simulation loop from the schedule value before realization).
    stv.current_rate = np.asarray(rate, dtype=float)
    return stv


def test_scheduled_exact_applies_on_first_timestep_only():
    # 2 timesteps per day: count applied on the first, zero on the second.
    stv = _make_stv([[100.0, 50.0]], [[0.1, 0.2]])
    num_timesteps = 2

    first = stv.get_scheduled_exact_realization(None, num_timesteps)
    np.testing.assert_array_equal(first, np.array([[10.0, 10.0]]))

    second = stv.get_scheduled_exact_realization(None, num_timesteps)
    np.testing.assert_array_equal(second, np.zeros((1, 2)))

    # Counter wraps back to the start of the next day -> applied again.
    third = stv.get_scheduled_exact_realization(None, num_timesteps)
    np.testing.assert_array_equal(third, np.array([[10.0, 10.0]]))


def test_scheduled_exact_rounds_to_integer_count():
    # 100 * 0.125 = 12.5 -> rint -> 12 (banker's rounding rounds .5 to even)
    stv = _make_stv([[100.0]], [[0.125]])
    realized = stv.get_scheduled_exact_realization(None, num_timesteps=1)
    np.testing.assert_array_equal(realized, np.array([[12.0]]))


def test_scheduled_exact_caps_at_origin_population():
    # A proportion > 1 must never move more than the available population.
    stv = _make_stv([[100.0]], [[2.0]])
    realized = stv.get_scheduled_exact_realization(None, num_timesteps=1)
    np.testing.assert_array_equal(realized, np.array([[100.0]]))


def test_scheduled_exact_reset_restores_day_counter():
    stv = _make_stv([[100.0]], [[0.1]])
    num_timesteps = 3
    # Advance partway through a day.
    stv.get_scheduled_exact_realization(None, num_timesteps)  # first timestep
    stv.get_scheduled_exact_realization(None, num_timesteps)  # mid-day -> zeros
    stv.reset()
    # reset() nulls current_rate (the simulation loop sets it each timestep);
    # restore it as the loop would before the next realization.
    stv.current_rate = np.array([[0.1]])
    # After reset, the next call is treated as the first timestep of a day again.
    realized = stv.get_scheduled_exact_realization(None, num_timesteps)
    np.testing.assert_array_equal(realized, np.array([[10.0]]))


# ---------------------------------------------------------------------------
# 3. Config-parser rejection paths for scheduled_exact
# ---------------------------------------------------------------------------

def _minimal_scheduled_exact_config():
    """A tiny valid-ish config with one scheduled_exact S->V transition."""
    return {
        "compartments": ["S", "V"],
        "params": {"num_age_groups": 1, "num_risk_groups": 1},
        "transitions": [
            {
                "name": "S_to_V",
                "origin": "S",
                "destination": "V",
                "rate_template": "scheduled_exact",
                "rate_config": {"schedule": "vax_sched"},
            }
        ],
        "schedules": [
            {
                "name": "vax_sched",
                "schedule_template": "timeseries_lookup",
                "schedule_config": {
                    "df_attribute": "vax_sched_df",
                    "value_column": "proportion",
                },
            }
        ],
    }


class _SchedulesInput:
    """Minimal stand-in for FluSubpopSchedules exposing one DataFrame attribute."""

    def __init__(self, df):
        self.vax_sched_df = df


def _vax_schedules_input():
    import pandas as pd

    df = pd.DataFrame(
        {"date": ["2022-08-08", "2022-08-09"], "proportion": [0.1, 0.1]}
    )
    return _SchedulesInput(df)


def test_scheduled_exact_rejected_in_transition_group():
    config = _minimal_scheduled_exact_config()
    config["transition_groups"] = [
        {
            "name": "bad_group",
            "transition_type": "binom",
            "members": ["S_to_V"],
        }
    ]
    with pytest.raises(ValueError):
        parse_model_config_from_dict(config, schedules_input=_vax_schedules_input())


def test_scheduled_exact_rejected_in_jointly_distributed():
    # A scheduled_exact transition is deterministic and may not participate in
    # a jointly-distributed (competing stochastic branches) relationship.
    config = _minimal_scheduled_exact_config()
    config["transitions"][0]["jointly_distributed_with"] = "S_to_other"
    config["transitions"].append(
        {
            "name": "S_to_other",
            "origin": "S",
            "destination": "V",
            "rate_template": "constant_param",
            "rate_config": {"param": "some_rate"},
            "jointly_distributed_with": "S_to_V",
        }
    )
    config["params"]["some_rate"] = 0.1
    with pytest.raises(ValueError):
        parse_model_config_from_dict(config, schedules_input=_vax_schedules_input())


# ---------------------------------------------------------------------------
# 4b. Pre-simulation history of a scheduled_exact transition (vaccination
#     before the simulation start date) feeding into the initial compartment
#     values -- exercised directly against generic_core, no notebook involved.
# ---------------------------------------------------------------------------

def _build_scheduled_exact_model(start_date, reset_param_value, *, with_reset_key=True):
    """Build a minimal S->V ConfigDrivenSubpopModel with a vaccine_schedule CSV
    starting well before start_date, to exercise pre-simulation accounting."""
    import pandas as pd
    from generic_core.generic_model import (
        ConfigDrivenSubpopModel,
        build_state_from_config,
        build_params_from_config,
    )

    rate_config = {"schedule": "vax_sched"}
    if with_reset_key:
        rate_config["compartment_reset_date_mm_dd_param"] = "vaccinated_compartment_reset_date_mm_dd"

    config_dict = {
        "compartments": ["S", "V"],
        "params": {
            "num_age_groups": 1,
            "num_risk_groups": 1,
            "vax_protection_delay_days": 0,
            "vaccinated_compartment_reset_date_mm_dd": reset_param_value,
        },
        "transitions": [
            {
                "name": "S_to_V",
                "origin": "S",
                "destination": "V",
                "rate_template": "scheduled_exact",
                "rate_config": rate_config,
            }
        ],
        "schedules": [
            {
                "name": "vax_sched",
                "schedule_template": "vaccine_schedule",
                "schedule_config": {
                    "df_attribute": "vax_sched_df",
                    "vax_protection_delay_days_param": "vax_protection_delay_days",
                },
            }
        ],
    }

    # Daily proportion of S vaccinated: 0.1/day for 5 days starting 2024-08-01,
    # well before start_date (2024-10-01) -- mirrors a CSV with pre-sim history.
    dates = [f"2024-08-0{d}" for d in range(1, 6)]
    df = pd.DataFrame({
        "date": dates,
        "daily_vaccines": [json.dumps([[0.1]])] * len(dates),
    })

    class _SchedulesInput:
        def __init__(self, df):
            self.vax_sched_df = df

    schedules_input = _SchedulesInput(df)
    model_config = parse_model_config_from_dict(config_dict, schedules_input=schedules_input)
    A, R = 1, 1

    compartment_init = {"S": np.array([[1000.0]]), "V": np.array([[0.0]])}
    state_init = build_state_from_config(model_config, compartment_init, {})
    params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)

    settings = clt.SimulationSettings(
        timesteps_per_day=1,
        start_real_date=start_date.isoformat(),
    )

    return ConfigDrivenSubpopModel(
        model_config=model_config,
        state_init=state_init,
        params=params,
        simulation_settings=settings,
        RNG=np.random.Generator(np.random.MT19937(0)),
        schedules_input=schedules_input,
        name="subpop",
    )


def test_scheduled_exact_replays_pre_simulation_history_into_compartments():
    """A vaccine schedule with 5 days of history before start_date, and no
    reset date set (None), must replay all 5 days into V at sim start."""
    model = _build_scheduled_exact_model(
        start_date=datetime.date(2024, 10, 1), reset_param_value=None
    )
    # 1000 * 0.1 = 100/day for 5 days, depleting S each day: 100, 90, 81, 72.9->73, 65.6->66
    # exact replay of rint(rate * remaining), capped at remaining.
    s = 1000.0
    moved_total = 0.0
    for _ in range(5):
        moved = min(round(0.1 * s), s)
        s -= moved
        moved_total += moved
    np.testing.assert_allclose(model.compartments["V"].current_val, [[moved_total]])
    np.testing.assert_allclose(model.compartments["S"].current_val, [[1000.0 - moved_total]])
    assert moved_total > 0


def test_scheduled_exact_reset_date_excludes_history_before_it():
    """Setting the reset-date param to a date after all CSV history must
    yield a zero adjustment (no vaccines counted)."""
    model = _build_scheduled_exact_model(
        start_date=datetime.date(2024, 10, 1), reset_param_value="09_15"
    )
    np.testing.assert_allclose(model.compartments["V"].current_val, [[0.0]])
    np.testing.assert_allclose(model.compartments["S"].current_val, [[1000.0]])


def test_scheduled_exact_no_reset_key_means_no_adjustment():
    """If rate_config omits 'compartment_reset_date_mm_dd_param' entirely,
    pre-simulation history must be ignored (existing/back-compat behavior)."""
    model = _build_scheduled_exact_model(
        start_date=datetime.date(2024, 10, 1), reset_param_value=None, with_reset_key=False
    )
    np.testing.assert_allclose(model.compartments["V"].current_val, [[0.0]])
    np.testing.assert_allclose(model.compartments["S"].current_val, [[1000.0]])


# ---------------------------------------------------------------------------
# 4. Regression: optional vax_induced_saturation_param on the torch path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drop_vax_sat", [False, True])
def test_torch_runs_with_optional_vax_saturation(drop_vax_sat):
    """
    Build the caseB generic metapop model on the torch path and run a short
    history. With drop_vax_sat=True the M-metric's vax_induced_saturation_param
    is removed from update_config; the torch path must treat the vax-immunity
    saturation term as zero rather than raising KeyError.
    """
    torch = pytest.importorskip("torch")

    import flu_core as flu  # noqa: F401  (ensures flu_core import side effects)
    from conftest import subpop_inputs
    from generic_core.generic_model import (
        ConfigDrivenSubpopModel,
        build_state_from_config,
        build_params_from_config,
    )
    from generic_core.generic_metapop import ConfigDrivenMetapopModel
    from generic_core.torch_generic import (
        build_generic_torch_inputs,
        generic_torch_simulate_full_history,
    )
    from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY

    compartments = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]

    config_dict = json.loads(CASEB_CONFIG_PATH.read_text())
    if drop_vax_sat:
        for m in config_dict["epi_metrics"]:
            if m["name"] == "M":
                m["update_config"].pop("vax_induced_saturation_param", None)

    state1, params1, mixing_params, settings, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, _, _, _ = subpop_inputs("caseB_subpop2")

    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND,
        "timesteps_per_day": 1,
        "use_deterministic_softplus": True,
    })

    model_config = parse_model_config_from_dict(config_dict, schedules_input=schedules_info)
    A, R = params1.num_age_groups, params1.num_risk_groups

    def _make_subpop(flu_state, bit_gen, name):
        compartment_init = {c: getattr(flu_state, c) for c in compartments}
        epi_metric_init = {
            "M": np.asarray(flu_state.M, dtype=float),
            "MV": np.asarray(flu_state.MV, dtype=float),
        }
        state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
        params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)
        return ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=state_init,
            params=params,
            simulation_settings=settings,
            RNG=np.random.Generator(bit_gen),
            schedules_input=schedules_info,
            name=name,
        )

    bit_gen1 = np.random.MT19937(88888)
    bit_gen2 = bit_gen1.jumped(1)
    s_to_e_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_E")
    travel_config = s_to_e_tc.rate_config["travel_config"]

    gen_model = ConfigDrivenMetapopModel(
        subpop_models=[
            _make_subpop(state1, bit_gen1, "subpop1"),
            _make_subpop(state2, bit_gen2, "subpop2"),
        ],
        mixing_params=mixing_params,
        model_config=model_config,
        travel_config=travel_config,
    )

    num_days = 5
    start_date = datetime.date.fromisoformat(
        list(gen_model._subpop_models_ordered.values())[0].simulation_settings.start_real_date
    )
    torch_inputs = build_generic_torch_inputs(gen_model, model_config, num_days, requires_grad=False)

    state_history, _ = generic_torch_simulate_full_history(
        torch_inputs["state_dict"],
        torch_inputs["params_dict"],
        model_config,
        RATE_TEMPLATE_REGISTRY,
        torch_inputs["precomputed"],
        torch_inputs["schedules_dict"],
        num_days,
        timesteps_per_day=1,
        start_real_date=start_date,
    )

    M_series = torch.stack([state_history["M"][d] for d in range(num_days)])
    assert torch.isfinite(M_series).all(), "M trajectory contains non-finite values"
