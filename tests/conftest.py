import clt_toolkit as clt
import flu_core as flu
import numpy as np
import pandas as pd
import pytest

from typing import Tuple

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"


# generic_core implements the reworked vaccine-efficacy model from the
#   `ve_update` branch: vaccine protection is a multiplicative
#   `1 - MV * VE_0` factor (with VE_0 back-derived from a season-average
#   efficacy) rather than a term in the additive immunity denominator, and
#   `vax_induced_saturation` no longer damps the M update.
#
# flu_core has NOT been ported yet on this branch, so the generic-vs-flu
#   exact-equality tests are expected to diverge until `ve_update` merges.
#   They are gated rather than deleted: this flips back on automatically
#   once flu_core gains the derived-efficacy params, at which point the
#   parity assertions become meaningful again (and must pass).
FLU_CORE_HAS_NEW_VE = hasattr(flu.FluSubpopParams, "vax_induced_inf_risk_reduce_initial")

requires_flu_core_new_ve = pytest.mark.skipif(
    not FLU_CORE_HAS_NEW_VE,
    reason="generic_core uses the ve_update vaccine-efficacy model; flu_core on this "
           "branch still uses the old additive-immunity model, so exact parity with "
           "flu_core cannot hold. Re-enables automatically once ve_update merges.",
)


def subpop_inputs(id: str) -> Tuple[flu.FluSubpopState,
                                    flu.FluSubpopParams,
                                    flu.FluMixingParams,
                                    flu.SimulationSettings,
                                    flu.FluSubpopSchedules]:
    """
    Generate data structures needed to construct `FluSubpopModel`
    instance based on `id`.
    """

    # 4 age groups, 3 risk groups
    # mixing params for 2 subpopulations
    if id == "caseA":
        init_vals_filepath = base_path / "caseA_init_vals.json"
        params_filepath = base_path / "caseA_subpop_params.json"
        mixing_params_filepath = base_path / "caseA_mixing_params.json"
        vaccines_df = pd.read_csv(base_path / "caseA_daily_vaccines_constant.csv", index_col=0)
        mobility_modifier_df = pd.read_csv(base_path / "caseA_mobility_modifier.csv", index_col=0)

    # 5 age groups, 1 risk group
    # mixing params for 2 subpopulations
    elif id == "caseB_subpop1":
        init_vals_filepath = base_path / "caseB_subpop1_init_vals.json"
        params_filepath = base_path / "caseB_subpop_params.json"
        mixing_params_filepath = base_path / "caseB_mixing_params.json"
        vaccines_df = pd.read_csv(base_path / "caseB_daily_vaccines_constant.csv", index_col=0)
        mobility_modifier_df = pd.read_csv(base_path / "caseB_mobility_modifier.csv", index_col=0)

    # 5 age groups, 1 risk group -- roughly 1/3 of population of caseB_subpop1
    # mixing params for 2 subpopulations
    elif id == "caseB_subpop2":
        init_vals_filepath = base_path / "caseB_subpop2_init_vals.json"
        params_filepath = base_path / "caseB_subpop_params.json"
        mixing_params_filepath = base_path / "caseB_mixing_params.json"
        vaccines_df = pd.read_csv(base_path / "caseB_daily_vaccines_constant.csv", index_col=0)
        mobility_modifier_df = pd.read_csv(base_path / "caseB_mobility_modifier.csv", index_col=0)

    simulation_settings_filepath = base_path / "simulation_settings.json"
    calendar_filepath = base_path / "school_work_calendar.csv"

    state = clt.make_dataclass_from_json(init_vals_filepath,
                                         flu.FluSubpopState)
    params = clt.make_dataclass_from_json(params_filepath, flu.FluSubpopParams)
    mixing_params = clt.make_dataclass_from_json(mixing_params_filepath, flu.FluMixingParams)
    settings = clt.make_dataclass_from_json(simulation_settings_filepath, flu.SimulationSettings)

    calendar_df = pd.read_csv(calendar_filepath, index_col=0)
    humidity_df = pd.read_csv(base_path / "absolute_humidity_austin_2023_2024.csv", index_col=0)
    

    schedules_info = flu.FluSubpopSchedules(absolute_humidity=humidity_df,
                                            flu_contact_matrix=calendar_df,
                                            daily_vaccines=vaccines_df,
                                            mobility_modifier=mobility_modifier_df)

    return state, params, mixing_params, \
           settings, schedules_info


# Factory function
# Need factory because pytest only runs a fixture once
#   per test! So, to be able to use this function
#   twice (or more) in a test -- for example, to
#   create two models, this actually needs to be
#   a function that returns a function.
# pytest documentation:
#   https://docs.pytest.org/en/6.2.x/fixture.html
# See section named 'Factories as fixtures'
# Also pytest doesn’t allow passing arguments to fixtures
#   like regular functions -- that's why arguments are
#   in the inner function, not the outer function
@pytest.fixture
def make_flu_subpop_model():
    def _make_flu_subpop_model(name: str,
                               transition_type: clt.TransitionTypes = clt.TransitionTypes.BINOM,
                               num_jumps: int = 0,
                               timesteps_per_day: int = 7,
                               case_id_str: str = "caseA"):
        init_vals, params, mixing_params, simulation_settings, schedules_info = \
            subpop_inputs(case_id_str)

        simulation_settings = clt.updated_dataclass(simulation_settings, {"timesteps_per_day": timesteps_per_day,
                                                                          "transition_type": transition_type})

        starting_random_seed = 123456789123456789
        bit_generator = np.random.MT19937(starting_random_seed)

        model = flu.FluSubpopModel(init_vals,
                                   params,
                                   simulation_settings,
                                   np.random.Generator(bit_generator.jumped(num_jumps)),
                                   schedules_info,
                                   name)

        return model

    return _make_flu_subpop_model


@pytest.fixture
def make_flu_metapop_model():
    def _make_flu_metapop_model(transition_type: clt.TransitionTypes,
                                subpop1_id: str = "caseB_subpop1",
                                subpop2_id: str = "caseB_subpop2",
                                settings_updates: dict = {}) -> flu.FluMetapopModel:
        state1, params1, mixing_params, settings, schedules_info = subpop_inputs(subpop1_id)
        state2, params2, mixing_params, settings, schedules_info = subpop_inputs(subpop2_id)

        settings = clt.updated_dataclass(settings,
                                         {"transition_type": transition_type,
                                          "timesteps_per_day": 1})
        
        if len(settings_updates) > 0:
            settings = clt.updated_dataclass(settings, settings_updates)
            

        bit_generator = np.random.MT19937(88888)
        jumped_bit_generator = bit_generator.jumped(1)

        subpop1 = flu.FluSubpopModel(state1,
                                     params1,
                                     settings,
                                     np.random.Generator(bit_generator),
                                     schedules_info,
                                     name="subpop1")

        subpop2 = flu.FluSubpopModel(state2,
                                     params2,
                                     settings,
                                     np.random.Generator(jumped_bit_generator),
                                     schedules_info,
                                     name="subpop2")

        model = flu.FluMetapopModel([subpop1, subpop2], mixing_params)

        return model

    return _make_flu_metapop_model
