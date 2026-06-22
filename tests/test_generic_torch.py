"""
Torch validation test — Phase 4, tasks 4.4 and 4.6.

Runs generic_torch_simulate_full_history alongside flu.torch_simulate_full_history
on the same caseB two-subpopulation data (5 age groups, 1 risk group), using the
same initial conditions.

Deterministic path test (task 4.6):
    Assert that all compartment trajectories are close (rtol=1e-2, matching the
    tolerance used in the existing flu torch tests in test_flu_metapop.py).

Gradient flow test (task 4.6):
    Run generic_torch_simulate_calibration_target (returns dict[str, Tensor]),
    compute scalar loss, call .backward(), assert gradients exist on
    beta_baseline and IP_to_ISH_prop.
"""

import numpy as np
import pytest
import sys
import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
import flu_core as flu
from conftest import subpop_inputs

from generic_core.config_parser import parse_model_config
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)
from generic_core.generic_metapop import ConfigDrivenMetapopModel
from generic_core.torch_generic import (
    build_generic_torch_inputs,
    generic_torch_simulate_full_history,
    generic_torch_simulate_calibration_target,
)
from generic_core.rate_templates import RATE_TEMPLATE_REGISTRY

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
CONFIG_PATH = BASE_PATH / "caseb_flu_generic_metapop_config.json"

NUM_DAYS = 50
RNG_SEED_1 = 88888
RNG_SEED_2 = 88888 + 1

COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]


# ---------------------------------------------------------------------------
# Shared factory: build ConfigDrivenMetapopModel on caseB data
# ---------------------------------------------------------------------------

def _make_generic_metapop_model(settings):
    state1_flu, params1_flu, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2_flu, params2_flu, _, _, _ = subpop_inputs("caseB_subpop2")

    model_config = parse_model_config(CONFIG_PATH, schedules_input=schedules_info)

    A = params1_flu.num_age_groups
    R = params1_flu.num_risk_groups

    def _make_subpop(flu_state, flu_params, rng_seed, name):
        compartment_init = {c: getattr(flu_state, c) for c in COMPARTMENTS}
        epi_metric_init = {
            "M":  np.asarray(flu_state.M,  dtype=float),
            "MV": np.asarray(flu_state.MV, dtype=float),
        }
        state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
        params = build_params_from_config(model_config, num_age_groups=A, num_risk_groups=R)
        RNG = np.random.Generator(rng_seed)
        return ConfigDrivenSubpopModel(
            model_config=model_config,
            state_init=state_init,
            params=params,
            simulation_settings=settings,
            RNG=RNG,
            schedules_input=schedules_info,
            name=name,
        )

    bit_gen1 = np.random.MT19937(RNG_SEED_1)
    bit_gen2 = bit_gen1.jumped(1)

    s_to_e_tc = next(tc for tc in model_config.transitions if tc.name == "S_to_E")
    travel_config = s_to_e_tc.rate_config["travel_config"]

    subpop1 = _make_subpop(state1_flu, params1_flu, bit_gen1, "subpop1")
    subpop2 = _make_subpop(state2_flu, params2_flu, bit_gen2, "subpop2")

    return ConfigDrivenMetapopModel(
        subpop_models=[subpop1, subpop2],
        mixing_params=mixing_params,
        model_config=model_config,
        travel_config=travel_config,
    ), model_config


def _make_flu_metapop_model(settings):
    state1, params1, mixing_params, _, schedules_info = subpop_inputs("caseB_subpop1")
    state2, params2, _, _, _ = subpop_inputs("caseB_subpop2")

    bit_gen1 = np.random.MT19937(RNG_SEED_1)
    bit_gen2 = bit_gen1.jumped(1)

    subpop1 = flu.FluSubpopModel(
        state1, params1, settings,
        np.random.Generator(bit_gen1), schedules_info, name="subpop1"
    )
    subpop2 = flu.FluSubpopModel(
        state2, params2, settings,
        np.random.Generator(bit_gen2), schedules_info, name="subpop2"
    )
    return flu.FluMetapopModel([subpop1, subpop2], mixing_params)


# ---------------------------------------------------------------------------
# Fixture: both torch histories computed once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_torch_histories():
    """
    Returns (flu_state_history, gen_state_history) dicts, both over NUM_DAYS.
    """
    _state, _params, _mp, settings, _sched = subpop_inputs("caseB_subpop1")
    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND,
        "timesteps_per_day": 1,
        "use_deterministic_softplus": True,
    })

    # --- flu torch reference ---
    flu_model = _make_flu_metapop_model(settings)
    d = flu_model.get_flu_torch_inputs()
    flu_state_history, _ = flu.torch_simulate_full_history(
        d["state_tensors"], d["params_tensors"], d["precomputed"],
        d["schedule_tensors"], NUM_DAYS, 1
    )

    # --- generic torch ---
    gen_model, model_config = _make_generic_metapop_model(settings)
    start_date = datetime.date.fromisoformat(
        list(gen_model._subpop_models_ordered.values())[0].simulation_settings.start_real_date
    )
    torch_inputs = build_generic_torch_inputs(gen_model, model_config, NUM_DAYS, requires_grad=False)
    gen_state_history, _ = generic_torch_simulate_full_history(
        torch_inputs["state_dict"],
        torch_inputs["params_dict"],
        model_config,
        RATE_TEMPLATE_REGISTRY,
        torch_inputs["precomputed"],
        torch_inputs["schedules_dict"],
        NUM_DAYS,
        timesteps_per_day=1,
        start_real_date=start_date,
    )

    return flu_state_history, gen_state_history


# ---------------------------------------------------------------------------
# Task 4.6 — Deterministic path: compartment histories match flu torch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subpop_ix", [0, 1])
@pytest.mark.parametrize("compartment", COMPARTMENTS)
def test_compartment_torch_history_matches_flu(both_torch_histories, subpop_ix, compartment):
    flu_hist, gen_hist = both_torch_histories

    # Both store lists of (L, A, R) tensors; extract the subpop slice
    flu_series = torch.stack([flu_hist[compartment][d][subpop_ix] for d in range(NUM_DAYS)])
    gen_series = torch.stack([gen_hist[compartment][d][subpop_ix] for d in range(NUM_DAYS)])

    assert torch.allclose(flu_series.float(), gen_series.float(), rtol=1e-2, atol=1.0), (
        f"subpop {subpop_ix} compartment '{compartment}' mismatch; "
        f"max diff = {(flu_series.float() - gen_series.float()).abs().max():.4g}"
    )


@pytest.mark.parametrize("subpop_ix", [0, 1])
def test_M_torch_history_matches_flu(both_torch_histories, subpop_ix):
    flu_hist, gen_hist = both_torch_histories
    flu_series = torch.stack([flu_hist["M"][d][subpop_ix] for d in range(NUM_DAYS)])
    gen_series = torch.stack([gen_hist["M"][d][subpop_ix] for d in range(NUM_DAYS)])
    assert torch.allclose(flu_series.float(), gen_series.float(), rtol=1e-2, atol=1e-4), (
        f"subpop {subpop_ix} M mismatch; max diff = {(flu_series - gen_series).abs().max():.4g}"
    )


@pytest.mark.parametrize("subpop_ix", [0, 1])
def test_MV_torch_history_matches_flu(both_torch_histories, subpop_ix):
    flu_hist, gen_hist = both_torch_histories
    flu_series = torch.stack([flu_hist["MV"][d][subpop_ix] for d in range(NUM_DAYS)])
    gen_series = torch.stack([gen_hist["MV"][d][subpop_ix] for d in range(NUM_DAYS)])
    assert torch.allclose(flu_series.float(), gen_series.float(), rtol=1e-2, atol=1e-4), (
        f"subpop {subpop_ix} MV mismatch; max diff = {(flu_series - gen_series).abs().max():.4g}"
    )


# ---------------------------------------------------------------------------
# Task 4.6 — Gradient flow test
# ---------------------------------------------------------------------------

def test_gradient_flows_through_calibration_target():
    """
    Run generic_torch_simulate_calibration_target (dict-returning API),
    compute a scalar loss, call .backward(), and confirm gradients exist
    on beta_baseline and IP_to_ISH_prop.
    """
    _state, _params, _mp, settings, _sched = subpop_inputs("caseB_subpop1")
    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND,
        "timesteps_per_day": 1,
        "use_deterministic_softplus": True,
    })

    gen_model, model_config = _make_generic_metapop_model(settings)
    start_date = datetime.date.fromisoformat(
        list(gen_model._subpop_models_ordered.values())[0].simulation_settings.start_real_date
    )

    num_days = 10
    torch_inputs = build_generic_torch_inputs(gen_model, model_config, num_days, requires_grad=True)

    result = generic_torch_simulate_calibration_target(
        torch_inputs["state_dict"],
        torch_inputs["params_dict"],
        model_config,
        RATE_TEMPLATE_REGISTRY,
        torch_inputs["precomputed"],
        torch_inputs["schedules_dict"],
        num_days=num_days,
        timesteps_per_day=1,
        calibration_transition_names=["ISH_to_HR", "ISH_to_HD"],
        start_real_date=start_date,
    )

    assert set(result.keys()) == {"ISH_to_HR", "ISH_to_HD"}, (
        f"Expected keys {{'ISH_to_HR', 'ISH_to_HD'}}, got {set(result.keys())}"
    )
    loss = result["ISH_to_HR"].sum() + result["ISH_to_HD"].sum()
    loss.backward()

    beta_grad = torch_inputs["params_dict"]["beta_baseline"].grad
    prop_grad = torch_inputs["params_dict"]["IP_to_ISH_prop"].grad

    assert beta_grad is not None, "No gradient on beta_baseline"
    assert prop_grad is not None, "No gradient on IP_to_ISH_prop"
    assert beta_grad.abs() > 0, "Gradient on beta_baseline is zero"
    assert prop_grad.abs() > 0, "Gradient on IP_to_ISH_prop is zero"
