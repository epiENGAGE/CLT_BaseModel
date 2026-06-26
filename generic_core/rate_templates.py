"""
rate_templates.py — RateTemplate ABC and built-in concrete implementations.

Each template encapsulates one rate formula with both a numpy path (used by
ConfigDrivenSubpopModel at simulation time) and a torch path (used by
torch_generic.py for gradient-based calibration).

Registry
--------
RATE_TEMPLATE_REGISTRY : dict[str, RateTemplate]
    Maps string template names → singleton instances.
    Built-in templates are registered at module import time.
    Users call register_rate_template() to add custom templates.

Reference implementations in flu_core/flu_components.py and
flu_core/flu_torch_det_components.py are the ground truth; all templates
must produce numerically identical output for the same inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce

import numpy as np
import torch


def _validate_optional_param_key(rate_name: str, rate_config: dict, key: str, param_names: set) -> None:
    if key not in rate_config:
        return
    pname = rate_config[key]
    if pname not in param_names:
        raise ValueError(
            f"{rate_name}: param '{pname}' (from key '{key}') not in model params"
        )


def _validate_optional_humidity_keys(
    rate_name: str,
    rate_config: dict,
    param_names: set,
    schedule_names: set,
) -> None:
    has_humidity_param = "humidity_impact_param" in rate_config
    has_humidity_schedule = "humidity_schedule" in rate_config
    if has_humidity_param != has_humidity_schedule:
        raise ValueError(
            f"{rate_name}: 'humidity_impact_param' and 'humidity_schedule' must either "
            "both be present or both be absent"
        )
    if has_humidity_param:
        _validate_optional_param_key(rate_name, rate_config, "humidity_impact_param", param_names)
        schedule_name = rate_config["humidity_schedule"]
        if schedule_name not in schedule_names:
            raise ValueError(
                f"{rate_name}: schedule '{schedule_name}' (from key 'humidity_schedule') "
                "not in model schedules"
            )


def _beta_adjusted_np(state, params, rate_config: dict):
    beta_adjusted = params.params[rate_config["beta_param"]]
    if "humidity_impact_param" in rate_config:
        humidity_impact = params.params[rate_config["humidity_impact_param"]]
        absolute_humidity = state.schedules[rate_config["humidity_schedule"]]
        beta_adjusted = beta_adjusted * (
            1.0 + humidity_impact * np.exp(-180.0 * absolute_humidity)
        )
    return beta_adjusted


def _beta_adjusted_torch(state_dict: dict, params_dict: dict, rate_config: dict):
    beta_adjusted = params_dict[rate_config["beta_param"]]
    if "humidity_impact_param" in rate_config:
        humidity_impact = params_dict[rate_config["humidity_impact_param"]]
        absolute_humidity = state_dict[rate_config["humidity_schedule"]]
        beta_adjusted = beta_adjusted * (
            1.0 + humidity_impact * torch.exp(-180.0 * absolute_humidity)
        )
    return beta_adjusted


def _immunity_force_np(epi_metrics: dict, params, rate_config: dict):
    immunity_force = 1.0
    if "inf_reduce_param" in rate_config and "M" in epi_metrics:
        inf_reduce = params.params[rate_config["inf_reduce_param"]]
        inf_prop = inf_reduce / (1.0 - inf_reduce)
        immunity_force = immunity_force + inf_prop * epi_metrics["M"]
    if "vax_reduce_param" in rate_config and "MV" in epi_metrics:
        vax_reduce = params.params[rate_config["vax_reduce_param"]]
        vax_prop = vax_reduce / (1.0 - vax_reduce)
        immunity_force = immunity_force + vax_prop * epi_metrics["MV"]
    return immunity_force


def _immunity_force_torch(state_dict: dict, params_dict: dict, rate_config: dict):
    immunity_force = 1.0
    if "inf_reduce_param" in rate_config and "M" in state_dict:
        inf_reduce = params_dict[rate_config["inf_reduce_param"]]
        inf_prop = inf_reduce / (1.0 - inf_reduce)
        immunity_force = immunity_force + inf_prop * state_dict["M"]
    if "vax_reduce_param" in rate_config and "MV" in state_dict:
        vax_reduce = params_dict[rate_config["vax_reduce_param"]]
        vax_prop = vax_reduce / (1.0 - vax_reduce)
        immunity_force = immunity_force + vax_prop * state_dict["MV"]
    return immunity_force


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class RateTemplate(ABC):
    """
    Abstract base class for all rate templates.

    Subclasses must implement:
        validate_config  — called once at model construction to catch config errors
        numpy_rate       — used by ConfigDrivenTransitionVariable.get_current_rate()
        torch_rate       — used by generic_advance_timestep()
    """

    @abstractmethod
    def validate_config(
        self,
        rate_config: dict,
        param_names: set,
        compartment_names: set,
        schedule_names: set,
    ) -> None:
        """
        Validate that rate_config references known params/compartments/schedules.
        Raises ValueError with a descriptive message on the first problem found.
        """

    @abstractmethod
    def numpy_rate(
        self,
        state,
        params,
        rate_config: dict,
    ) -> np.ndarray:
        """
        Compute the per-timestep transition rate.

        Returns
        -------
        np.ndarray of shape (A, R)
            Rate for each age × risk group.
        """

    @abstractmethod
    def torch_rate(
        self,
        state_dict: dict,
        params_dict: dict,
        rate_config: dict,
    ) -> torch.Tensor:
        """
        Torch-differentiable version of numpy_rate.

        Returns
        -------
        torch.Tensor of shape (A, R) or (L, A, R)
        """


# ---------------------------------------------------------------------------
# 1. ConstantParamRate
# ---------------------------------------------------------------------------

class ConstantParamRate(RateTemplate):
    """
    Constant rate equal to a single scalar parameter, broadcast to (A, R).

    rate_config keys
    ----------------
    param : str
        Name of the scalar parameter in params.params.

    Reference: RecoveredToSusceptible, SympRecoverToRecovered,
    AsympToRecovered, HospRecoverToRecovered, HospDeadToDead in
    flu_core/flu_components.py.
    """

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        if "param" not in rate_config:
            raise ValueError("ConstantParamRate: rate_config must contain 'param'")
        name = rate_config["param"]
        if name not in param_names:
            raise ValueError(
                f"ConstantParamRate: param '{name}' not found in model params"
            )

    def numpy_rate(self, state, params, rate_config):
        return np.full(
            (params.num_age_groups, params.num_risk_groups),
            params.params[rate_config["param"]],
        )

    def torch_rate(self, state_dict, params_dict, rate_config):
        # params_dict values are already correctly shaped tensors
        return params_dict[rate_config["param"]]


# ---------------------------------------------------------------------------
# 2. ParamProductRate
# ---------------------------------------------------------------------------

class ParamProductRate(RateTemplate):
    """
    Rate equal to the product of zero or more parameter factors, each
    optionally complemented (1 − factor), broadcast to (A, R).

    rate_config keys
    ----------------
    factors : list[str]
        Parameter names whose values are multiplied together.
    complement_factors : list[str]
        Parameter names used as (1 − value) before multiplying.

    Example (ExposedToAsymp):
        {"factors": ["E_to_I_rate"], "complement_factors": []}
    →   rate = E_to_I_rate * E_to_IA_prop     (but that is separate key)

    Actual usage:
        ExposedToAsymp:   {"factors": ["E_to_I_rate", "E_to_IA_prop"]}
        ExposedToPresymp: {"factors": ["E_to_I_rate"], "complement_factors": ["E_to_IA_prop"]}

    Reference: ExposedToAsymp, ExposedToPresymp in flu_components.py:152–194.
    """

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        for fname in rate_config.get("factors", []):
            if fname not in param_names:
                raise ValueError(
                    f"ParamProductRate: factor param '{fname}' not in model params"
                )
        for cname in rate_config.get("complement_factors", []):
            if cname not in param_names:
                raise ValueError(
                    f"ParamProductRate: complement_factor param '{cname}' not in model params"
                )

    def numpy_rate(self, state, params, rate_config):
        result = 1.0
        for fname in rate_config.get("factors", []):
            result = result * params.params[fname]
        for cname in rate_config.get("complement_factors", []):
            result = result * (1.0 - params.params[cname])
        return np.full((params.num_age_groups, params.num_risk_groups), result)

    def torch_rate(self, state_dict, params_dict, rate_config):
        factors = rate_config.get("factors", [])
        complements = rate_config.get("complement_factors", [])
        tensors = (
            [params_dict[f] for f in factors]
            + [1.0 - params_dict[c] for c in complements]
        )
        if not tensors:
            raise ValueError("ParamProductRate: at least one factor or complement_factor required")
        return reduce(lambda a, b: a * b, tensors)


# ---------------------------------------------------------------------------
# 3. ImmunityModulatedRate
# ---------------------------------------------------------------------------

class ImmunityModulatedRate(RateTemplate):
    """
    Rate modulated by population-level infection- and vaccine-induced immunity.

    immunity_force = 1 + (r_inf / (1 − r_inf)) × M + (r_vax / (1 − r_vax)) × MV

    If is_complement is True  (e.g. IP→ISR — complement of hospitalization):
        rate = base_rate × (1 − proportion / immunity_force)

    If is_complement is False (e.g. IP→ISH — hospitalization):
        rate = base_rate × (proportion / immunity_force)

    rate_config keys
    ----------------
    base_rate : str
        Parameter name for the base transition rate (e.g. "IP_to_IS_rate").
    proportion : str
        Parameter name for the proportional split (e.g. "IP_to_ISH_prop").
    is_complement : bool
        True → complement branch; False → direct branch.
    inf_reduce_param : str
        Parameter name for infection-induced risk reduction (in [0, 1)).
    vax_reduce_param : str
        Parameter name for vaccine-induced risk reduction (in [0, 1)).

    Reference: PresympToSympRecover, PresympToSympHospital,
    SympHospitalToHospRecover, SympHospitalToHospDead in
    flu_components.py:197–346.
    """

    _REQUIRED = ("base_rate", "proportion")

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        for key in self._REQUIRED:
            if key not in rate_config:
                raise ValueError(
                    f"ImmunityModulatedRate: missing required key '{key}' in rate_config"
                )
        for key in ("base_rate", "proportion"):
            pname = rate_config[key]
            if pname not in param_names:
                raise ValueError(
                    f"ImmunityModulatedRate: param '{pname}' (from key '{key}') not in model params"
                )
        _validate_optional_param_key("ImmunityModulatedRate", rate_config, "inf_reduce_param", param_names)
        _validate_optional_param_key("ImmunityModulatedRate", rate_config, "vax_reduce_param", param_names)

    def numpy_rate(self, state, params, rate_config):
        base_rate = params.params[rate_config["base_rate"]]
        proportion = params.params[rate_config["proportion"]]
        is_complement = rate_config.get("is_complement", False)
        immunity_force = _immunity_force_np(state.epi_metrics, params, rate_config)

        if is_complement:
            return np.asarray((1.0 - proportion / immunity_force) * base_rate)
        else:
            return np.asarray((proportion / immunity_force) * base_rate)

    def torch_rate(self, state_dict, params_dict, rate_config):
        base_rate = params_dict[rate_config["base_rate"]]
        proportion = params_dict[rate_config["proportion"]]
        is_complement = rate_config.get("is_complement", False)
        immunity_force = _immunity_force_torch(state_dict, params_dict, rate_config)

        if is_complement:
            return (1.0 - proportion / immunity_force) * base_rate
        else:
            return (proportion / immunity_force) * base_rate


# ---------------------------------------------------------------------------
# 4. ForceOfInfectionRate  (single-population)
# ---------------------------------------------------------------------------

class ForceOfInfectionRate(RateTemplate):
    """
    Force-of-infection rate for a single-population model, with optional
    humidity and immunity modifiers.

    Reproduces the `else` branch of SusceptibleToExposed.get_current_rate()
    (flu_components.py:119–131) and the equivalent torch formula in
    flu_torch_det_components.py:102–136 (for L = 1 with trivial travel).

    Formula
    -------
    beta_adjusted  = beta_baseline × (1 + humidity_impact × exp(−180 × absolute_humidity))
                     if humidity is configured, else beta_baseline
    immune_force   = 1 + optional infection-induced term + optional vaccine-induced term
    wtd_inf_by_age = Σ_c  [rel_inf_c × Σ_R compartment_c]   (A × 1)
    wtd_inf_prop   = wtd_inf_by_age / pop_by_age              (A × 1)
    raw_exposure   = contact_matrix @ wtd_inf_prop            (A × 1)
    rate           = relative_suscept × beta_adjusted × raw_exposure / immune_force   (A × R)

    rate_config keys
    ----------------
    beta_param : str
        Parameter name for beta_baseline.
    humidity_impact_param : str, optional
        Parameter name for humidity_impact (scale factor on humidity).
    humidity_schedule : str, optional
        Schedule name for the absolute humidity time series.
    contact_matrix_schedule : str
        Schedule name for the current contact matrix (A × A ndarray).
    inf_reduce_param : str, optional
        Parameter name for inf_induced_inf_risk_reduce.
    vax_reduce_param : str, optional
        Parameter name for vax_induced_inf_risk_reduce.
    infectious_compartments : dict[str, str | None]
        Keys are compartment names; values are either None (relative
        infectiousness = 1.0) or a parameter name whose value gives the
        relative infectiousness (e.g. {"IP": "IP_relative_inf"}).
    relative_susceptibility_param : str
        Parameter name for relative_suscept (A × R array).
    """

    _REQUIRED_KEYS = (
        "beta_param",
        "contact_matrix_schedule",
        "infectious_compartments",
        "relative_susceptibility_param",
    )

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        for key in self._REQUIRED_KEYS:
            if key not in rate_config:
                raise ValueError(
                    f"ForceOfInfectionRate: missing required key '{key}' in rate_config"
                )
        infectious_compartments = rate_config["infectious_compartments"]
        if not infectious_compartments:
            raise ValueError(
                "ForceOfInfectionRate: 'infectious_compartments' must be a non-empty mapping"
            )
        for pkey in ("beta_param", "relative_susceptibility_param"):
            pname = rate_config[pkey]
            if pname not in param_names:
                raise ValueError(
                    f"ForceOfInfectionRate: param '{pname}' (from key '{pkey}') not in model params"
                )
        _validate_optional_humidity_keys(
            "ForceOfInfectionRate", rate_config, param_names, schedule_names
        )
        _validate_optional_param_key("ForceOfInfectionRate", rate_config, "inf_reduce_param", param_names)
        _validate_optional_param_key("ForceOfInfectionRate", rate_config, "vax_reduce_param", param_names)
        contact_schedule = rate_config["contact_matrix_schedule"]
        if contact_schedule not in schedule_names:
            raise ValueError(
                f"ForceOfInfectionRate: schedule '{contact_schedule}' (from key "
                "'contact_matrix_schedule') not in model schedules"
            )
        for comp_name, rel_inf_param in rate_config["infectious_compartments"].items():
            if comp_name not in compartment_names:
                raise ValueError(
                    f"ForceOfInfectionRate: infectious compartment '{comp_name}' not in model compartments"
                )
            if rel_inf_param is not None and rel_inf_param not in param_names:
                raise ValueError(
                    f"ForceOfInfectionRate: relative infectiousness param '{rel_inf_param}' not in model params"
                )

    def _wtd_infectious_by_age_np(self, compartments, params_vals, infectious_compartments):
        """Returns (A, 1) weighted-infectious-count array. Caller ensures non-empty dict."""
        items = list(infectious_compartments.items())
        first_name, first_rel = items[0]
        first_comp = compartments[first_name]
        result: np.ndarray = np.sum(first_comp, axis=1, keepdims=True)
        if first_rel is not None:
            result = params_vals[first_rel] * result
        for comp_name, rel_inf_param in items[1:]:
            comp = compartments[comp_name]
            wtd = np.sum(comp, axis=1, keepdims=True)
            if rel_inf_param is not None:
                wtd = params_vals[rel_inf_param] * wtd
            result = result + wtd
        return result

    def _wtd_infectious_by_age_torch(self, state_dict, params_dict, infectious_compartments):
        """Returns (L, A, 1) or (A, 1) weighted-infectious-count tensor. Caller ensures non-empty dict."""
        items = list(infectious_compartments.items())
        first_name, first_rel = items[0]
        first_comp = state_dict[first_name]
        result: torch.Tensor = first_comp.sum(dim=-1, keepdim=True)
        if first_rel is not None:
            result = params_dict[first_rel] * result
        for comp_name, rel_inf_param in items[1:]:
            comp = state_dict[comp_name]
            wtd = comp.sum(dim=-1, keepdim=True)
            if rel_inf_param is not None:
                wtd = params_dict[rel_inf_param] * wtd
            result = result + wtd
        return result

    def numpy_rate(self, state, params, rate_config):
        beta_adjusted = _beta_adjusted_np(state, params, rate_config)
        immune_force = _immunity_force_np(state.epi_metrics, params, rate_config)

        # --- weighted infectious proportion (A × 1) ---
        pop_by_age = np.sum(params.total_pop_age_risk, axis=1, keepdims=True)
        wtd_inf = self._wtd_infectious_by_age_np(
            state.compartments, params.params, rate_config["infectious_compartments"]
        )
        wtd_inf_prop = np.divide(wtd_inf, pop_by_age)

        # --- contact matrix × infectious proportion (A × 1) ---
        contact_matrix = state.schedules[rate_config["contact_matrix_schedule"]]
        raw_exposure = np.matmul(contact_matrix, wtd_inf_prop)

        # --- final rate (A × R) ---
        relative_suscept = params.params[rate_config["relative_susceptibility_param"]]
        return relative_suscept * (beta_adjusted * raw_exposure / immune_force)

    def torch_rate(self, state_dict, params_dict, rate_config):
        beta_adjusted = _beta_adjusted_torch(state_dict, params_dict, rate_config)
        immune_force = _immunity_force_torch(state_dict, params_dict, rate_config)

        # --- weighted infectious proportion ---
        # total_pop_age_risk is (A, R); sum over R → (A, 1)
        # For torch metapop tensors (L, A, R), sum over last dim → (L, A, 1)
        total_pop = params_dict["total_pop_age_risk"]
        pop_by_age = total_pop.sum(dim=-1, keepdim=True)
        wtd_inf = self._wtd_infectious_by_age_torch(
            state_dict, params_dict, rate_config["infectious_compartments"]
        )
        wtd_inf_prop = wtd_inf / pop_by_age

        # --- contact matrix × infectious proportion ---
        contact_matrix = state_dict[rate_config["contact_matrix_schedule"]]
        # contact_matrix: (A, A) or (L, A, A); wtd_inf_prop: (A, 1) or (L, A, 1)
        raw_exposure = torch.matmul(contact_matrix, wtd_inf_prop)

        # --- final rate ---
        relative_suscept = params_dict[rate_config["relative_susceptibility_param"]]
        return relative_suscept * (beta_adjusted * raw_exposure / immune_force)


# ---------------------------------------------------------------------------
# 5. ForceOfInfectionOptionalRate  (backward-compatible alias)
# ---------------------------------------------------------------------------

class ForceOfInfectionOptionalRate(ForceOfInfectionRate):
    """
    Backward-compatible alias for ForceOfInfectionRate.
    """
    pass


# ---------------------------------------------------------------------------
# 6. ForceOfInfectionTravelRate  (metapopulation)
# ---------------------------------------------------------------------------

class ForceOfInfectionTravelRate(RateTemplate):
    """
    Force-of-infection rate for the metapopulation (travel) model.

    Reproduces the `if self.total_mixing_exposure is not None` branch of
    SusceptibleToExposed.get_current_rate() (flu_components.py:109–117).

    This template delegates the heavy lifting to
    generic_core.travel_functions.compute_total_mixing_exposure, which is
    wired in Phase 3. Until Phase 3, calling numpy_rate or torch_rate raises
    NotImplementedError.

    rate_config keys
    ----------------
    beta_param : str
    humidity_impact_param : str, optional
    humidity_schedule : str, optional
    inf_reduce_param : str, optional
    vax_reduce_param : str, optional
    travel_config : dict
        Passed directly to compute_total_mixing_exposure; carries
        infectious_compartments and immobile_compartments.
    """

    _REQUIRED_KEYS = (
        "beta_param",
        "travel_config",
    )

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        for key in self._REQUIRED_KEYS:
            if key not in rate_config:
                raise ValueError(
                    f"ForceOfInfectionTravelRate: missing required key '{key}' in rate_config"
                )
        _validate_optional_param_key("ForceOfInfectionTravelRate", rate_config, "beta_param", param_names)
        _validate_optional_humidity_keys(
            "ForceOfInfectionTravelRate", rate_config, param_names, schedule_names
        )
        _validate_optional_param_key("ForceOfInfectionTravelRate", rate_config, "inf_reduce_param", param_names)
        _validate_optional_param_key("ForceOfInfectionTravelRate", rate_config, "vax_reduce_param", param_names)

    def numpy_rate(self, state, params, rate_config):
        """
        Compute S→E rate using travel-model total_mixing_exposure.

        Requires ConfigDrivenMetapopModel to have injected
        rate_config["_total_mixing_exposure"] before calling get_current_rate().
        That injection happens in ConfigDrivenMetapopModel.apply_inter_subpop_updates().

        Mirrors the `if self.total_mixing_exposure is not None` branch in
        SusceptibleToExposed.get_current_rate() (flu_components.py:109-117).
        """
        total_mixing_exposure = rate_config.get("_total_mixing_exposure")
        if total_mixing_exposure is None:
            raise RuntimeError(
                "ForceOfInfectionTravelRate.numpy_rate: '_total_mixing_exposure' not set "
                "in rate_config. Ensure this transition belongs to a ConfigDrivenMetapopModel "
                "and that apply_inter_subpop_updates() has been called."
            )

        beta_adjusted = _beta_adjusted_np(state, params, rate_config)
        immune_force = _immunity_force_np(state.epi_metrics, params, rate_config)

        return np.asarray(beta_adjusted * total_mixing_exposure / immune_force)

    def torch_rate(self, state_dict, params_dict, rate_config):
        """
        Compute S→E rate using travel-model total_mixing_exposure.

        Requires _precomputed to be injected into rate_config before calling
        (done by generic_advance_timestep via _build_torch_rate_config).

        Mirrors compute_S_to_E in flu_torch_det_components.py:102–136 but
        uses generic travel functions.
        """
        from .travel_functions import compute_total_mixing_exposure

        precomputed = rate_config.get("_precomputed")
        if precomputed is None:
            raise RuntimeError(
                "ForceOfInfectionTravelRate.torch_rate: '_precomputed' not set in rate_config. "
                "Ensure generic_advance_timestep is used, which injects _precomputed."
            )

        travel_config = rate_config["travel_config"]

        # Build sub-dicts for compute_total_mixing_exposure
        compartment_tensors = {}
        for c in travel_config["immobile_compartments"]:
            compartment_tensors[c] = state_dict[c]
        for c in travel_config["infectious_compartments"]:
            compartment_tensors[c] = state_dict[c]

        schedule_tensors = {
            travel_config["contact_matrix_schedule"]: state_dict[
                travel_config["contact_matrix_schedule"]
            ],
            travel_config["mobility_schedule"]: state_dict[
                travel_config["mobility_schedule"]
            ],
        }

        rel_suscept_param = travel_config["relative_susceptibility_param"]
        param_tensors = {
            "travel_proportions": params_dict["travel_proportions"],
            rel_suscept_param: params_dict[rel_suscept_param],
        }
        for rel_inf_param in travel_config["infectious_compartments"].values():
            if rel_inf_param is not None and rel_inf_param not in param_tensors:
                param_tensors[rel_inf_param] = params_dict[rel_inf_param]

        total_mixing_exposure = compute_total_mixing_exposure(
            compartment_tensors, param_tensors, schedule_tensors, precomputed, travel_config
        )

        beta_adjusted = _beta_adjusted_torch(state_dict, params_dict, rate_config)
        immune_force = _immunity_force_torch(state_dict, params_dict, rate_config)

        return beta_adjusted * total_mixing_exposure / immune_force


# ---------------------------------------------------------------------------
# 7. ScheduledExactTransferRate  (validation-only marker)
# ---------------------------------------------------------------------------

class ScheduledExactTransferRate(RateTemplate):
    """
    Validation-only marker for 'scheduled_exact' transitions.

    numpy_rate/torch_rate are never called at simulation time --
    ConfigDrivenSubpopModel.create_transition_variables (numpy) and
    generic_advance_timestep (torch) special-case transitions with this
    rate_template name and apply the exact scheduled transfer directly,
    bypassing the RateTemplate machinery entirely. This class exists only
    so 'scheduled_exact' can be validated and looked up uniformly wherever
    other rate templates are (e.g. parse_model_config_from_dict).

    rate_config keys
    -----------------
    schedule : str
        Name of the schedule (e.g. a vaccine_schedule instance) providing the
        daily *proportion* of the origin compartment to move to the destination
        each day. The proportion is converted to an exact (rounded, capped)
        count in ScheduledTransferVariable.get_scheduled_exact_realization
        (numpy) and generic_advance_timestep (torch).
    compartment_reset_date_mm_dd_param : str, optional
        Parameter name holding a "MM_DD" string (or None). If set, the
        schedule's pre-simulation history (between this reset date and the
        simulation start date) is summed and moved from origin to destination
        in the initial compartment values, so a schedule (e.g. a vaccine CSV)
        that starts before the simulation start date still counts toward the
        initial vaccinated population. If the param is absent/None, all
        history before the simulation start date is used. If this key is
        omitted entirely from rate_config, no adjustment is made (the
        transition's pre-simulation history is ignored, as before). See
        ConfigDrivenSubpopModel._compute_scheduled_exact_pre_simulation_adjustments
        in generic_model.py.
    """

    def validate_config(self, rate_config, param_names, compartment_names, schedule_names):
        if "schedule" not in rate_config:
            raise ValueError("ScheduledExactTransferRate: rate_config must contain 'schedule'")
        sname = rate_config["schedule"]
        if sname not in schedule_names:
            raise ValueError(
                f"ScheduledExactTransferRate: schedule '{sname}' not in model schedules"
            )

    def numpy_rate(self, state, params, rate_config):
        raise NotImplementedError(
            "scheduled_exact transitions bypass RateTemplate.numpy_rate; "
            "see ConfigDrivenSubpopModel.create_transition_variables"
        )

    def torch_rate(self, state_dict, params_dict, rate_config):
        raise NotImplementedError(
            "scheduled_exact transitions bypass RateTemplate.torch_rate; "
            "see generic_advance_timestep"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RATE_TEMPLATE_REGISTRY: dict[str, RateTemplate] = {}


def register_rate_template(name: str, instance: RateTemplate) -> None:
    """
    Add a RateTemplate instance to the global registry under `name`.

    Built-in templates are registered automatically on import.
    Call this function to register custom templates before parsing a config.
    """
    RATE_TEMPLATE_REGISTRY[name] = instance


# Register all built-in templates
register_rate_template("constant_param", ConstantParamRate())
register_rate_template("param_product", ParamProductRate())
register_rate_template("immunity_modulated", ImmunityModulatedRate())
register_rate_template("force_of_infection", ForceOfInfectionRate())
register_rate_template("force_of_infection_optional", ForceOfInfectionOptionalRate())
register_rate_template("force_of_infection_travel", ForceOfInfectionTravelRate())
register_rate_template("scheduled_exact", ScheduledExactTransferRate())
