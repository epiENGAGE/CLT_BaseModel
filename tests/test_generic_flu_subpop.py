"""
Single-population flu validation test — Phase 2, task 2.7.

Runs ConfigDrivenSubpopModel (loaded from flu_generic_config.json) and
FluSubpopModel with identical initial conditions, RNG seed, and simulation
settings, then asserts that all 10 compartment trajectories are exactly equal.

Both models use the caseA test inputs (4 age groups, 3 risk groups).
Transition type: binom_deterministic, timesteps_per_day = 7.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
import flu_core as flu
from conftest import subpop_inputs
from conftest import requires_flu_core_new_ve

from generic_core.config_parser import parse_model_config
from generic_core.generic_model import (
    ConfigDrivenSubpopModel,
    build_state_from_config,
    build_params_from_config,
)

BASE_PATH = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"
FLU_CONFIG_PATH = BASE_PATH / "flu_generic_config.json"

NUM_DAYS = 150
RNG_SEED = 123456789123456789

COMPARTMENTS = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_flu_model(state, params, settings, schedules_info, name):
    RNG = np.random.Generator(np.random.MT19937(RNG_SEED))
    return flu.FluSubpopModel(state, params, settings, RNG, schedules_info, name=name)


def _make_generic_model(state, params_flu, settings, schedules_info):
    model_config = parse_model_config(FLU_CONFIG_PATH, schedules_input=schedules_info)

    # Build initial compartment dict from the FluSubpopState
    compartment_init = {name: getattr(state, name) for name in COMPARTMENTS}
    epi_metric_init = {
        "M": np.asarray(state.M, dtype=float),
        "MV": np.asarray(state.MV, dtype=float),
    }

    state_init = build_state_from_config(model_config, compartment_init, epi_metric_init)
    generic_params = build_params_from_config(
        model_config,
        num_age_groups=params_flu.num_age_groups,
        num_risk_groups=params_flu.num_risk_groups,
    )

    RNG = np.random.Generator(np.random.MT19937(RNG_SEED))
    return ConfigDrivenSubpopModel(
        model_config=model_config,
        state_init=state_init,
        params=generic_params,
        simulation_settings=settings,
        RNG=RNG,
        schedules_input=schedules_info,
        name="flu_generic",
    )


# ---------------------------------------------------------------------------
# Fixture: both models simulated for NUM_DAYS
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def both_models():
    flu_state, flu_params, _, settings, schedules_info = subpop_inputs("caseA")
    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC,
        "timesteps_per_day": 7,
        "save_daily_history": True,
    })

    ref_model = _make_flu_model(flu_state, flu_params, settings, schedules_info, name="flu_ref")
    gen_model = _make_generic_model(flu_state, flu_params, settings, schedules_info)

    ref_model.simulate_until_day(NUM_DAYS)
    gen_model.simulate_until_day(NUM_DAYS)

    return ref_model, gen_model


# ---------------------------------------------------------------------------
# Tests: identical compartment trajectories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compartment", COMPARTMENTS)
@requires_flu_core_new_ve
def test_compartment_trajectory_identical(both_models, compartment):
    ref, gen = both_models
    ref_hist = np.array(ref.compartments[compartment].history_vals_list)
    gen_hist = np.array(gen.compartments[compartment].history_vals_list)
    assert np.array_equal(ref_hist, gen_hist), (
        f"Compartment '{compartment}' mismatch at day(s) where "
        f"max diff = {np.max(np.abs(ref_hist.astype(float) - gen_hist.astype(float))):.6g}; "
        f"first divergence at day index "
        f"{int(np.argmax(np.any(ref_hist != gen_hist, axis=(1, 2))))}"
    )


@requires_flu_core_new_ve
def test_epi_metric_M_trajectory(both_models):
    ref, gen = both_models
    ref_hist = np.array(ref.epi_metrics["M"].history_vals_list)
    gen_hist = np.array(gen.epi_metrics["M"].history_vals_list)
    assert np.allclose(ref_hist, gen_hist, rtol=1e-14), (
        f"EpiMetric 'M' mismatch; max diff = "
        f"{np.max(np.abs(ref_hist - gen_hist)):.6g}"
    )


def test_epi_metric_MV_trajectory(both_models):
    ref, gen = both_models
    ref_hist = np.array(ref.epi_metrics["MV"].history_vals_list)
    gen_hist = np.array(gen.epi_metrics["MV"].history_vals_list)
    assert np.allclose(ref_hist, gen_hist, rtol=1e-14), (
        f"EpiMetric 'MV' mismatch; max diff = "
        f"{np.max(np.abs(ref_hist - gen_hist)):.6g}"
    )


def test_population_conserved(both_models):
    ref, gen = both_models
    ref_total_pop = float(np.sum(ref.params.total_pop_age_risk))
    for day_idx in range(NUM_DAYS):
        day_total = sum(
            float(np.sum(gen.compartments[c].history_vals_list[day_idx]))
            for c in COMPARTMENTS
        )
        assert abs(day_total - ref_total_pop) < 1e-4, (
            f"Population not conserved on day {day_idx}: "
            f"{day_total:.2f} != {ref_total_pop:.2f}"
        )
