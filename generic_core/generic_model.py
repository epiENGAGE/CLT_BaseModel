"""
generic_model.py — Config-driven subpopulation model.

Classes
-------
ConfigDrivenTransitionVariable
    TransitionVariable subclass whose get_current_rate() delegates to a RateTemplate.

ConfigDrivenEpiMetric
    Thin wrapper — constructed by a MetricTemplate factory; check_and_apply_reset
    forwarded here if present on the underlying metric.

ConfigDrivenSubpopModel
    SubpopModel subclass that implements all 7 factory methods by iterating over
    a validated ModelConfig.

The implementation mirrors FluSubpopModel (flu_core/flu_components.py:901+) but
is fully driven by configuration rather than hard-coded compartment/transition names.
"""

from __future__ import annotations

import copy
import datetime
import sciris as sc
import numpy as np

import clt_toolkit as clt
from clt_toolkit.base_data_structures import SimulationSettings

from .data_structures import GenericSubpopState, GenericSubpopParams
from .config_parser import ModelConfig
from .rate_templates import RATE_TEMPLATE_REGISTRY
from .metric_templates import METRIC_TEMPLATE_REGISTRY, VaxInducedImmunityGeneric
from .schedule_templates import SCHEDULE_TEMPLATE_REGISTRY


# ---------------------------------------------------------------------------
# ConfigDrivenTransitionVariable
# ---------------------------------------------------------------------------

class ConfigDrivenTransitionVariable(clt.TransitionVariable):
    """
    TransitionVariable whose rate is computed by a RateTemplate instance.

    get_current_rate() delegates entirely to rate_template.numpy_rate().
    """

    def __init__(
        self,
        origin: clt.Compartment,
        destination: clt.Compartment,
        transition_type: str,
        rate_template,
        rate_config: dict,
        is_jointly_distributed: bool = False,
    ):
        super().__init__(origin, destination, transition_type, is_jointly_distributed)
        self._rate_template = rate_template
        self._rate_config = rate_config

    def get_current_rate(self, state, params) -> np.ndarray:
        return self._rate_template.numpy_rate(state, params, self._rate_config)


# ---------------------------------------------------------------------------
# ScheduledTransferVariable
# ---------------------------------------------------------------------------

class ScheduledTransferVariable(clt.TransitionVariable):
    """
    Moves an exact, rounded count of people from origin to destination on
    the first timestep of each simulation day (0 on subsequent timesteps
    within the same day).

    The count comes from a Schedule (e.g. a vaccine_schedule instance)
    that already encodes any delay/backfill -- this class only applies
    that schedule's current value as an exact compartment transfer,
    bypassing the rate-to-probability machinery used by every other
    TransitionVariable in this codebase.
    """

    def __init__(self, origin: clt.Compartment, destination: clt.Compartment, schedule_name: str):
        super().__init__(origin, destination, "scheduled_exact", is_jointly_distributed=False)
        self.schedule_name = schedule_name
        self._timestep_in_day = 0

    def get_current_rate(self, state, params) -> np.ndarray:
        # NOTE: this is a proportion of the origin compartment's population
        # vaccinated that day (matching the existing vaccine_schedule input
        # format, e.g. FluSubpopState.daily_vaccines), not an absolute count.
        # It is converted to a count in get_scheduled_exact_realization.
        return state.schedules[self.schedule_name]  # type: ignore[attr-defined]

    def get_scheduled_exact_realization(self, RNG, num_timesteps) -> np.ndarray:
        is_first_timestep = self._timestep_in_day == 0
        self._timestep_in_day = (self._timestep_in_day + 1) % num_timesteps
        origin_val = np.asarray(self.origin.current_val)
        if not is_first_timestep:
            return np.zeros_like(origin_val)
        scheduled_count = np.rint(np.asarray(self.current_rate) * origin_val)
        return np.minimum(scheduled_count, origin_val)

    def reset(self) -> None:
        super().reset()
        self._timestep_in_day = 0


# ---------------------------------------------------------------------------
# ConfigDrivenSubpopModel
# ---------------------------------------------------------------------------

class ConfigDrivenSubpopModel(clt.SubpopModel):
    """
    SubpopModel whose entire structure is determined by a ModelConfig.

    Parameters
    ----------
    model_config : ModelConfig
        Fully validated config from parse_model_config().
    state_init : GenericSubpopState
        Initial state (compartment and metric values); populated from the
        config by the caller.
    params : GenericSubpopParams
        Model parameters; populated from config.params by the caller.
    simulation_settings : SimulationSettings
    RNG : np.random.Generator
    schedules_input : Any
        Raw schedules input (e.g. FluSubpopSchedules); passed through to
        schedule template build_schedule() calls.
    name : str
        Unique identifier for this subpopulation.
    rate_registry : dict | None
        Override rate template registry. Defaults to RATE_TEMPLATE_REGISTRY.
    metric_registry : dict | None
        Override metric template registry. Defaults to METRIC_TEMPLATE_REGISTRY.
    schedule_registry : dict | None
        Override schedule template registry. Defaults to SCHEDULE_TEMPLATE_REGISTRY.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        state_init: GenericSubpopState,
        params: GenericSubpopParams,
        simulation_settings: SimulationSettings,
        RNG: np.random.Generator,
        schedules_input,
        name: str,
        rate_registry: dict | None = None,
        metric_registry: dict | None = None,
        schedule_registry: dict | None = None,
    ):
        self.model_config = model_config
        self.schedules_input = schedules_input
        self._rate_registry = rate_registry or RATE_TEMPLATE_REGISTRY
        self._metric_registry = metric_registry or METRIC_TEMPLATE_REGISTRY
        self._schedule_registry = schedule_registry or SCHEDULE_TEMPLATE_REGISTRY

        # Store initial compartment values so create_compartments() can read them
        self._state_init = state_init

        # super().__init__ calls create_schedules, create_compartments, ...,
        # run_input_checks in that order.
        super().__init__(state_init, params, simulation_settings, RNG, name)

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    def create_compartments(self) -> sc.objdict:
        """
        Create one Compartment per name declared in model_config.compartments.
        Initial values come from _state_init.compartments (populated by caller).
        """
        compartments = sc.objdict()
        A, R = self.params.num_age_groups, self.params.num_risk_groups
        for name in self.model_config.compartments:
            init = np.asarray(
                self._state_init._cvals.get(name, np.zeros((A, R))),
                dtype=float,
            )
            compartments[name] = clt.Compartment(init)
        return compartments

    def create_transition_variables(self) -> sc.objdict:
        """
        Create one ConfigDrivenTransitionVariable per transition in model_config.
        """
        transition_type = self.simulation_settings.transition_type
        tvars = sc.objdict()

        for tc in self.model_config.transitions:
            origin = self.compartments[tc.origin]
            dest = self.compartments[tc.destination]

            if tc.rate_template == "scheduled_exact":
                tvars[tc.name] = ScheduledTransferVariable(
                    origin=origin,
                    destination=dest,
                    schedule_name=tc.rate_config["schedule"],
                )
                continue

            template = self._rate_registry[tc.rate_template]
            is_joint = tc.jointly_distributed_with is not None
            tvars[tc.name] = ConfigDrivenTransitionVariable(
                origin=origin,
                destination=dest,
                transition_type=transition_type,
                rate_template=template,
                rate_config=dict(tc.rate_config),
                is_jointly_distributed=is_joint,
            )
        return tvars

    def create_transition_variable_groups(self) -> sc.objdict:
        """
        Create one TransitionVariableGroup per group in model_config.transition_groups.
        """
        transition_type = self.simulation_settings.transition_type
        groups = sc.objdict()

        for gc in self.model_config.transition_groups:
            members = [self.transition_variables[m] for m in gc.members]
            origin = members[0].origin
            groups[gc.name] = clt.TransitionVariableGroup(
                origin, transition_type, members
            )
        return groups

    def create_epi_metrics(self) -> sc.objdict:
        """
        Create EpiMetric instances via metric template factories.
        """
        metrics = sc.objdict()

        for mc in self.model_config.epi_metrics:
            template = self._metric_registry[mc.metric_template]

            # Use state_init value if the caller supplied one; fall back to JSON init_val.
            # This allows passing M/MV from a warm-started flu model without baking the
            # initial values into the JSON config.
            init_val = self._state_init._evals.get(mc.name)
            if init_val is None:
                init_val = mc.init_val

            # Inject current_real_date into update_config for VaxInducedImmunity
            update_config = dict(mc.update_config)
            update_config["_current_real_date"] = self.current_real_date

            schedules_dict = {
                sc_cfg.name: self.schedules.get(sc_cfg.name)
                for sc_cfg in self.model_config.schedules
            }
            transition_variables_dict = dict(self.transition_variables)

            metrics[mc.name] = template.build_metric(
                init_val=init_val,
                update_config=update_config,
                params=self.params,
                transition_variables=transition_variables_dict,
                schedules=schedules_dict,
                timesteps_per_day=self.simulation_settings.timesteps_per_day,
            )
        return metrics

    def create_schedules(self) -> sc.objdict:
        """
        Create Schedule instances via schedule template factories.
        """
        schedules = sc.objdict()
        for sc_cfg in self.model_config.schedules:
            template = self._schedule_registry[sc_cfg.schedule_template]
            schedules[sc_cfg.name] = template.build_schedule(
                sc_cfg.schedule_config, self.params, self.schedules_input
            )
        return schedules

    def create_dynamic_vals(self) -> sc.objdict:
        """
        No dynamic vals by default. Override to add model-specific dynamic vals.
        """
        return sc.objdict()

    def run_input_checks(self) -> None:
        """
        Ensure initial compartment values are non-negative.
        Skips flu-specific checks (humidity, vaccination, etc.).
        """
        # Populate state from the just-created compartments and metrics
        # (state dicts may be empty at this point since sync_to_current_vals
        #  has not been called yet by the simulation loop)
        self.state.sync_to_current_vals(self.compartments)
        self.state.sync_to_current_vals(self.epi_metrics)

        for name, arr in self.state.compartments.items():
            if arr is not None and not np.all(arr >= 0):
                raise ValueError(
                    f"ConfigDrivenSubpopModel '{self.name}': "
                    f"initial value of compartment '{name}' is negative"
                )
        for name, arr in self.state.epi_metrics.items():
            if arr is not None and not np.all(arr >= 0):
                raise ValueError(
                    f"ConfigDrivenSubpopModel '{self.name}': "
                    f"initial value of epi metric '{name}' is negative"
                )

    def prepare_daily_state(self) -> None:
        """
        Override to also call check_and_apply_reset on vaccine immunity metrics.

        Mirrors FluSubpopModel.prepare_daily_state.
        """
        super().prepare_daily_state()

        for metric in self.epi_metrics.values():
            if isinstance(metric, VaxInducedImmunityGeneric):
                metric.check_and_apply_reset(self.current_real_date, self.params)


# ---------------------------------------------------------------------------
# Factory helper: build GenericSubpopState from a ModelConfig and initial values
# ---------------------------------------------------------------------------

def build_state_from_config(
    model_config: ModelConfig,
    compartment_init: dict[str, np.ndarray],
    epi_metric_init: dict[str, np.ndarray],
) -> GenericSubpopState:
    """
    Construct a GenericSubpopState from initial value dicts.

    Parameters
    ----------
    model_config : ModelConfig
        Provides compartment, metric, schedule, and dynval name sets.
    compartment_init : dict[str, np.ndarray]
        Initial compartment arrays, keyed by compartment name.
    epi_metric_init : dict[str, np.ndarray]
        Initial epi metric arrays, keyed by metric name.

    Returns
    -------
    GenericSubpopState
        State with _cvals and _evals pre-populated.
    """
    state = GenericSubpopState(
        compartment_names=model_config.compartment_names,
        epi_metric_names=model_config.epi_metric_names,
        schedule_names=model_config.schedule_names,
        dynamic_val_names=set(),
    )
    for name in model_config.compartments:
        if name not in compartment_init:
            raise ValueError(
                f"build_state_from_config: missing initial value for compartment '{name}'"
            )
        state._cvals[name] = np.asarray(compartment_init[name], dtype=float)

    for mc in model_config.epi_metrics:
        if mc.name in epi_metric_init:
            state._evals[mc.name] = np.asarray(epi_metric_init[mc.name], dtype=float)
        else:
            state._evals[mc.name] = mc.init_val.copy()

    return state


def build_params_from_config(
    model_config: ModelConfig,
    num_age_groups: int,
    num_risk_groups: int,
) -> GenericSubpopParams:
    """
    Construct GenericSubpopParams from a validated ModelConfig.

    total_pop_age_risk is initialized to zeros; it is overwritten by
    SubpopModel.compute_total_pop_age_risk() during SubpopModel.__init__.
    """
    return GenericSubpopParams(
        params=copy.deepcopy(model_config.params),
        num_age_groups=num_age_groups,
        num_risk_groups=num_risk_groups,
        total_pop_age_risk=np.zeros((num_age_groups, num_risk_groups)),
    )
