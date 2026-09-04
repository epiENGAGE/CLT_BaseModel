"""
Tests for generic_core rate templates — Phase 1, task 1.7.

For each built-in template:
  1. numpy_rate() matches the corresponding FluSubpopModel transition's
     get_current_rate() to machine epsilon (rtol = 1e-12 or better).
  2. numpy_rate() and torch_rate() agree to float32 precision (rtol = 1e-5).

All comparisons use the caseA FluSubpopModel state after 50 days of
deterministic simulation, which ensures M and MV are non-trivially nonzero.
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import clt_toolkit as clt
import flu_core as flu
from conftest import subpop_inputs
from conftest import requires_flu_core_new_ve

from generic_core.rate_templates import (
    ConstantParamRate,
    ParamProductRate,
    ImmunityModulatedRate,
    ForceOfInfectionRate,
)
from generic_core.data_structures import GenericSubpopState, GenericSubpopParams
from generic_core.ve_derivation import ve_inflation_param_name

NUM_DAYS_WARMUP = 50


# ---------------------------------------------------------------------------
# Helpers to extract generic state / params from a warmed-up FluSubpopModel
# ---------------------------------------------------------------------------

def _make_flu_model():
    state, params, _, settings, schedules_info = subpop_inputs("caseA")
    settings = clt.updated_dataclass(settings, {
        "transition_type": clt.TransitionTypes.BINOM_DETERMINISTIC,
        "timesteps_per_day": 1,
    })
    RNG = np.random.Generator(np.random.MT19937(42))
    model = flu.FluSubpopModel(state, params, settings, RNG, schedules_info, name="test")
    model.simulate_until_day(NUM_DAYS_WARMUP)
    return model


def _generic_state(flu_model) -> GenericSubpopState:
    comp_names = ["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"]
    gs = GenericSubpopState(
        compartment_names=comp_names,
        epi_metric_names=["M", "MV"],
        schedule_names=["absolute_humidity", "flu_contact_matrix"],
    )
    for name in comp_names:
        gs._cvals[name] = flu_model.compartments[name].current_val.copy()
    gs._evals["M"] = flu_model.epi_metrics["M"].current_val.copy()
    gs._evals["MV"] = flu_model.epi_metrics["MV"].current_val.copy()
    gs._svals["absolute_humidity"] = flu_model.schedules["absolute_humidity"].current_val
    gs._svals["flu_contact_matrix"] = flu_model.schedules["flu_contact_matrix"].current_val.copy()
    return gs


def _generic_params(flu_model) -> GenericSubpopParams:
    p = flu_model.params
    params_dict = {
        "R_to_S_rate":                   p.R_to_S_rate,
        "E_to_I_rate":                   p.E_to_I_rate,
        "E_to_IA_prop":                  p.E_to_IA_prop,
        "IP_to_IS_rate":                 p.IP_to_IS_rate,
        "IP_to_ISH_prop":                float(np.asarray(p.IP_to_ISH_prop)),
        "ISH_to_HD_prop":                float(np.asarray(p.ISH_to_HD_prop)),
        "ISH_to_H_rate":                 p.ISH_to_H_rate,
        "ISR_to_R_rate":                 p.ISR_to_R_rate,
        "IA_to_R_rate":                  p.IA_to_R_rate,
        "HR_to_R_rate":                  p.HR_to_R_rate,
        "HD_to_D_rate":                  p.HD_to_D_rate,
        "beta_baseline":                 p.beta_baseline,
        "humidity_impact":               p.humidity_impact,
        "inf_induced_inf_risk_reduce":   p.inf_induced_inf_risk_reduce,
        "inf_induced_hosp_risk_reduce":  p.inf_induced_hosp_risk_reduce,
        "inf_induced_death_risk_reduce": p.inf_induced_death_risk_reduce,
        "vax_induced_inf_risk_reduce":   p.vax_induced_inf_risk_reduce,
        "vax_induced_hosp_risk_reduce":  p.vax_induced_hosp_risk_reduce,
        "vax_induced_death_risk_reduce": p.vax_induced_death_risk_reduce,
        "IP_relative_inf":               p.IP_relative_inf,
        "IA_relative_inf":               p.IA_relative_inf,
        "relative_suscept":              np.asarray(p.relative_suscept, dtype=float),
    }

    # The new VE model applies a PEAK efficacy VE_0, derived from the
    # season-average value the user supplies. flu_core precomputes VE_0 onto
    # params as `<name>_initial`; generic_core instead stores the inflation
    # factor VE_0 / VE_season and forms the product live at each rate
    # evaluation (see `_vax_induced_peak_efficacy_np`), so that overriding the
    # season-average value still reaches the trajectory. Carry flu's derived
    # value across in the form generic_core actually reads, so both sides
    # apply the same VE_0.
    for _name in ("vax_induced_inf_risk_reduce",
                  "vax_induced_hosp_risk_reduce",
                  "vax_induced_death_risk_reduce"):
        _season = np.asarray(getattr(p, _name), dtype=float)
        _peak = getattr(p, f"{_name}_initial", None)
        _inflation = (np.ones_like(_season) if _peak is None
                      else np.asarray(_peak, dtype=float) / _season)
        params_dict[ve_inflation_param_name(_name)] = _inflation
    return GenericSubpopParams(
        params=params_dict,
        num_age_groups=p.num_age_groups,
        num_risk_groups=p.num_risk_groups,
        total_pop_age_risk=p.total_pop_age_risk.copy(),
    )


def _torch_state_dict(gs: GenericSubpopState) -> dict:
    sd = {}
    for name, val in gs._cvals.items():
        sd[name] = torch.tensor(val, dtype=torch.float32)
    sd["M"] = torch.tensor(gs._evals["M"], dtype=torch.float32)
    sd["MV"] = torch.tensor(gs._evals["MV"], dtype=torch.float32)
    sd["absolute_humidity"] = torch.tensor(
        float(gs._svals["absolute_humidity"]), dtype=torch.float32
    )
    sd["flu_contact_matrix"] = torch.tensor(
        gs._svals["flu_contact_matrix"], dtype=torch.float32
    )
    return sd


def _torch_params_dict(gp: GenericSubpopParams) -> dict:
    A, R = gp.num_age_groups, gp.num_risk_groups
    pd_ = {}
    for name, val in gp.params.items():
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, (A, R)).copy()
        pd_[name] = torch.tensor(arr, dtype=torch.float32)
    pd_["total_pop_age_risk"] = torch.tensor(
        gp.total_pop_age_risk.astype(np.float32), dtype=torch.float32
    )
    return pd_


# ---------------------------------------------------------------------------
# Module-level fixtures (computed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flu_model():
    return _make_flu_model()


@pytest.fixture(scope="module")
def gstate(flu_model):
    return _generic_state(flu_model)


@pytest.fixture(scope="module")
def gparams(flu_model):
    return _generic_params(flu_model)


@pytest.fixture(scope="module")
def torch_state(gstate):
    return _torch_state_dict(gstate)


@pytest.fixture(scope="module")
def torch_params(gparams):
    return _torch_params_dict(gparams)


# ---------------------------------------------------------------------------
# 1.  ConstantParamRate  vs  flu get_current_rate
# ---------------------------------------------------------------------------

class TestConstantParamRateNumpyVsFlu:

    def _check(self, flu_model, gstate, gparams, tvar_name, param_name):
        template = ConstantParamRate()
        flu_tv = flu_model.transition_variables[tvar_name]
        expected = flu_tv.get_current_rate(flu_model.state, flu_model.params)
        got = template.numpy_rate(gstate, gparams, {"param": param_name})
        assert np.allclose(got, expected, atol=0, rtol=0), \
            f"{tvar_name}: got {got}, expected {expected}"

    def test_R_to_S(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "R_to_S", "R_to_S_rate")

    def test_ISR_to_R(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "ISR_to_R", "ISR_to_R_rate")

    def test_IA_to_R(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "IA_to_R", "IA_to_R_rate")

    def test_HR_to_R(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "HR_to_R", "HR_to_R_rate")

    def test_HD_to_D(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "HD_to_D", "HD_to_D_rate")


# ---------------------------------------------------------------------------
# 2.  ParamProductRate  vs  flu get_current_rate
# ---------------------------------------------------------------------------

class TestParamProductRateNumpyVsFlu:

    def test_E_to_IA(self, flu_model, gstate, gparams):
        template = ParamProductRate()
        flu_tv = flu_model.transition_variables["E_to_IA"]
        expected = flu_tv.get_current_rate(flu_model.state, flu_model.params)
        got = template.numpy_rate(
            gstate, gparams,
            {"factors": ["E_to_I_rate", "E_to_IA_prop"]}
        )
        assert np.allclose(got, expected, atol=0, rtol=0)

    def test_E_to_IP(self, flu_model, gstate, gparams):
        template = ParamProductRate()
        flu_tv = flu_model.transition_variables["E_to_IP"]
        expected = flu_tv.get_current_rate(flu_model.state, flu_model.params)
        got = template.numpy_rate(
            gstate, gparams,
            {"factors": ["E_to_I_rate"], "complement_factors": ["E_to_IA_prop"]}
        )
        assert np.allclose(got, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# 3.  ImmunityModulatedRate  vs  flu get_current_rate
# ---------------------------------------------------------------------------

@requires_flu_core_new_ve
class TestImmunityModulatedRateNumpyVsFlu:

    _TEMPLATE = ImmunityModulatedRate()

    def _check(self, flu_model, gstate, gparams, tvar_name, rate_config):
        flu_tv = flu_model.transition_variables[tvar_name]
        expected = flu_tv.get_current_rate(flu_model.state, flu_model.params)
        got = self._TEMPLATE.numpy_rate(gstate, gparams, rate_config)
        assert np.allclose(got, expected, rtol=1e-14), \
            f"{tvar_name}: max diff = {np.max(np.abs(got - expected))}"

    def test_IP_to_ISR(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "IP_to_ISR", {
            "base_rate": "IP_to_IS_rate", "proportion": "IP_to_ISH_prop",
            "is_complement": True,
            "inf_reduce_param": "inf_induced_hosp_risk_reduce",
            "vax_reduce_param": "vax_induced_hosp_risk_reduce",
        })

    def test_IP_to_ISH(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "IP_to_ISH", {
            "base_rate": "IP_to_IS_rate", "proportion": "IP_to_ISH_prop",
            "is_complement": False,
            "inf_reduce_param": "inf_induced_hosp_risk_reduce",
            "vax_reduce_param": "vax_induced_hosp_risk_reduce",
        })

    def test_ISH_to_HR(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "ISH_to_HR", {
            "base_rate": "ISH_to_H_rate", "proportion": "ISH_to_HD_prop",
            "is_complement": True,
            "inf_reduce_param": "inf_induced_death_risk_reduce",
            "vax_reduce_param": "vax_induced_death_risk_reduce",
        })

    def test_ISH_to_HD(self, flu_model, gstate, gparams):
        self._check(flu_model, gstate, gparams, "ISH_to_HD", {
            "base_rate": "ISH_to_H_rate", "proportion": "ISH_to_HD_prop",
            "is_complement": False,
            "inf_reduce_param": "inf_induced_death_risk_reduce",
            "vax_reduce_param": "vax_induced_death_risk_reduce",
        })


# ---------------------------------------------------------------------------
# 4.  ForceOfInfectionRate  vs  flu get_current_rate  (single-population)
# ---------------------------------------------------------------------------

_FOI_RATE_CONFIG = {
    "beta_param":                    "beta_baseline",
    "humidity_impact_param":         "humidity_impact",
    "humidity_schedule":             "absolute_humidity",
    "contact_matrix_schedule":       "flu_contact_matrix",
    "inf_reduce_param":              "inf_induced_inf_risk_reduce",
    "vax_reduce_param":      "vax_induced_inf_risk_reduce",
    "infectious_compartments": {
        "ISR": None,
        "ISH": None,
        "IP":  "IP_relative_inf",
        "IA":  "IA_relative_inf",
    },
    "relative_susceptibility_param": "relative_suscept",
}


@requires_flu_core_new_ve
class TestForceOfInfectionRateNumpyVsFlu:

    def test_S_to_E(self, flu_model, gstate, gparams):
        template = ForceOfInfectionRate()
        flu_tv = flu_model.transition_variables["S_to_E"]
        # Ensure single-pop branch (no travel)
        assert flu_tv.total_mixing_exposure is None
        expected = flu_tv.get_current_rate(flu_model.state, flu_model.params)
        got = template.numpy_rate(gstate, gparams, _FOI_RATE_CONFIG)
        assert np.allclose(got, expected, rtol=1e-12), \
            f"max diff = {np.max(np.abs(got - expected))}"


# ---------------------------------------------------------------------------
# 5.  numpy_rate  vs  torch_rate  (float32 precision, rtol = 1e-5)
# ---------------------------------------------------------------------------

class TestNumpyVsTorch:
    """
    For each template, verify that torch_rate() produces the same
    numerical output as numpy_rate() when using float32 tensors.
    Tolerance: 1e-5 (float32 precision).
    """

    _ATOL = 1e-5

    def _compare(self, np_result, torch_result):
        np_ref = np.asarray(np_result, dtype=np.float32)
        torch_np = torch_result.detach().numpy()
        # broadcast torch output to match numpy shape if needed
        torch_np = np.broadcast_to(torch_np, np_ref.shape)
        assert np.allclose(np_ref, torch_np, atol=self._ATOL), \
            f"numpy vs torch mismatch — max diff = {np.max(np.abs(np_ref - torch_np))}"

    def test_constant_param_R_to_S(self, gstate, gparams, torch_state, torch_params):
        t = ConstantParamRate()
        rc = {"param": "R_to_S_rate"}
        np_r = t.numpy_rate(gstate, gparams, rc)
        torch_r = t.torch_rate(torch_state, torch_params, rc)
        self._compare(np_r, torch_r)

    def test_param_product_E_to_IA(self, gstate, gparams, torch_state, torch_params):
        t = ParamProductRate()
        rc = {"factors": ["E_to_I_rate", "E_to_IA_prop"]}
        np_r = t.numpy_rate(gstate, gparams, rc)
        torch_r = t.torch_rate(torch_state, torch_params, rc)
        self._compare(np_r, torch_r)

    def test_param_product_E_to_IP(self, gstate, gparams, torch_state, torch_params):
        t = ParamProductRate()
        rc = {"factors": ["E_to_I_rate"], "complement_factors": ["E_to_IA_prop"]}
        np_r = t.numpy_rate(gstate, gparams, rc)
        torch_r = t.torch_rate(torch_state, torch_params, rc)
        self._compare(np_r, torch_r)

    def test_immunity_modulated_IP_to_ISR(self, gstate, gparams, torch_state, torch_params):
        t = ImmunityModulatedRate()
        rc = {
            "base_rate": "IP_to_IS_rate", "proportion": "IP_to_ISH_prop",
            "is_complement": True,
            "inf_reduce_param": "inf_induced_hosp_risk_reduce",
            "vax_reduce_param": "vax_induced_hosp_risk_reduce",
        }
        np_r = t.numpy_rate(gstate, gparams, rc)
        torch_r = t.torch_rate(torch_state, torch_params, rc)
        self._compare(np_r, torch_r)

    def test_immunity_modulated_IP_to_ISH(self, gstate, gparams, torch_state, torch_params):
        t = ImmunityModulatedRate()
        rc = {
            "base_rate": "IP_to_IS_rate", "proportion": "IP_to_ISH_prop",
            "is_complement": False,
            "inf_reduce_param": "inf_induced_hosp_risk_reduce",
            "vax_reduce_param": "vax_induced_hosp_risk_reduce",
        }
        np_r = t.numpy_rate(gstate, gparams, rc)
        torch_r = t.torch_rate(torch_state, torch_params, rc)
        self._compare(np_r, torch_r)

    def test_force_of_infection_S_to_E(self, gstate, gparams, torch_state, torch_params):
        t = ForceOfInfectionRate()
        np_r = t.numpy_rate(gstate, gparams, _FOI_RATE_CONFIG)
        torch_r = t.torch_rate(torch_state, torch_params, _FOI_RATE_CONFIG)
        self._compare(np_r, torch_r)
