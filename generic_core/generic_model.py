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
from .metric_templates import (
    METRIC_TEMPLATE_REGISTRY,
    InfInducedImmunityGeneric,
    VaxInducedImmunityGeneric,
    injection_val_param_name,
)
from .schedule_templates import SCHEDULE_TEMPLATE_REGISTRY
from .ve_derivation import (
    compute_ve_inflation_factors,
    warn_if_ve_initial_capped,
)


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
        # NOTE: this is a proportion of the origin+destination pool (the
        # "susceptible" vax_pool -- everyone not yet infected, whether or not
        # already scheduled-transferred) vaccinated that day (matching the
        # existing vaccine_schedule input format, e.g. FluSubpopState.daily_vaccines),
        # not an absolute count. It is converted to a count in
        # get_scheduled_exact_realization.
        return state.schedules[self.schedule_name]  # type: ignore[attr-defined]

    def get_scheduled_exact_realization(self, RNG, num_timesteps) -> np.ndarray:
        is_first_timestep = self._timestep_in_day == 0
        self._timestep_in_day = (self._timestep_in_day + 1) % num_timesteps
        origin_val = np.asarray(self.origin.current_val)
        if not is_first_timestep:
            return np.zeros_like(origin_val)
        # vax_pool="susceptible": proportion applies to origin+destination (the
        # not-yet-infected pool), not origin alone -- vaccinating someone doesn't
        # shrink the base future proportions are applied to, only infection does.
        destination_val = np.asarray(self.destination.current_val)
        scheduled_count = np.rint(np.asarray(self.current_rate) * (origin_val + destination_val))
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

        self.update_ve_inflation_factors()
        self.update_infection_immunity_injection_val()

        # The immunity metrics adjust their own initial values in their
        #   constructors (decaying M(0) forward, deferring it, or adding
        #   pre-start vaccine doses to MV(0)), but `self.state` still holds
        #   the raw values from the config. Sync so that anything reading
        #   `self.state` before the first simulated day sees the adjusted
        #   values -- notably the torch path, which builds its starting
        #   tensors straight off `self.state` and would otherwise start from
        #   different initial immunity than the numpy run.
        self.state.sync_to_current_vals(self.epi_metrics)

    # -----------------------------------------------------------------------
    # Derived vaccine-efficacy / immunity params
    # -----------------------------------------------------------------------

    def update_ve_inflation_factors(self, warn: bool = True) -> None:
        """
        Recompute the derived VE inflation factors from the current dose
        schedule and waning rate, writing them into `self.params.params`.

        Each factor is `VE_0 / VE_season` for one configured season-average
        efficacy param; the rate templates multiply it by that param's live
        value at every evaluation (see
        `rate_templates._vax_induced_peak_efficacy_np`). Storing the factor
        rather than the finished `VE_0` is what keeps season-average efficacy
        overrides -- scenario overrides, fitting draws, torch autodiff --
        effective; see ve_derivation's module docstring.

        The factor itself depends on the dose schedule and the waning rate,
        so it is also refreshed on `reset_simulation` and again at the start
        of day 0 in `prepare_daily_state`.

        The day-0 refresh is the one that matters for overrides. Every
        override path in this repo applies parameters AFTER
        `reset_simulation()` -- `fitting._reuse_simulate` additionally
        restores a cached baseline params dict, which overwrites whatever
        the reset just computed -- so a reset-time refresh alone would leave
        the factor pinned to its construction-time waning rate for an entire
        fit. Refreshing once more when the run actually starts picks up
        every override regardless of ordering.

        Args:
            warn: whether to report age-risk groups whose implied peak
                efficacy exceeds 1.0 and will be capped. False for the
                day-0 refresh, which would otherwise repeat the same
                warning on every one of a fit's thousands of evaluations.

        No-op when the config has no "ve_derivation" block.
        """

        derivation_config = self.model_config.ve_derivation

        if not derivation_config:
            return

        self.params.params.update(compute_ve_inflation_factors(
            self.params, self.schedules, self.start_real_date, derivation_config
        ))

        if warn:
            warn_if_ve_initial_capped(self.params, derivation_config)

    def update_infection_immunity_injection_val(self) -> None:
        """
        Mirror each infection-induced immunity metric's pending injection onto
        `self.params.params` -- the amount to add to M when its start date is
        reached, or zeros if no injection is pending.

        The numpy model applies the injection straight off the epi metric
        (`InfInducedImmunityGeneric.check_and_apply_injection`) and does not
        need this. The torch model has no epi metric objects -- it only sees
        params and state tensors -- so the value has to travel on params for
        `torch_generic.check_and_apply_M_injection` to apply it there.

        Like `update_ve_inflation_factors`, this is re-run on
        `reset_simulation` so it tracks post-construction changes.

        Note that `pending_injection_date is None` covers three different
        situations -- no start date, a start date landing on the simulation
        start, and a PAST start date whose value was decayed forward into
        `init_val`. All three want zeros here: the adjusted M(0) reaches the
        torch path through the state tensors, not through this param. Only a
        FUTURE start date defers, and only then is there anything to inject.

        The mirror is kept in step with the metric throughout: it is set at
        construction and on `reset_simulation`, and `prepare_daily_state`
        re-runs this method on the day an injection fires so the spent
        injection zeroes out. That matters because `build_generic_torch_inputs`
        reads `params` (and the model's current state) at whatever point it is
        called -- a mirror left claiming "pending" after the fact would make
        the torch path apply the same injection a second time.
        """

        A, R = self.params.num_age_groups, self.params.num_risk_groups

        for name, metric in self.epi_metrics.items():
            if not isinstance(metric, InfInducedImmunityGeneric):
                continue
            if metric.pending_injection_date is not None:
                injection_val = np.asarray(metric.original_init_val, dtype=float).copy()
            else:
                injection_val = np.zeros((A, R))
            self.params.params[injection_val_param_name(name)] = injection_val

    def reset_simulation(self) -> None:
        """
        Extend the base `reset_simulation` to recompute the immunity metrics'
        initial values and the derived vaccine-efficacy params from the
        currently loaded schedules and params, before resetting.

        This ensures that if a schedule has been replaced (e.g. via
        `replace_schedule`), or base params have been overridden (e.g. by
        `ScenarioRunner`), the model resets to values consistent with the
        current schedules/params rather than those computed at construction
        time.

        Both recomputations start from each metric's `original_init_val` --
        the unmodified config value -- so adjustments do not compound across
        calls.
        """

        schedules_dict = dict(self.schedules)

        for metric in self.epi_metrics.values():
            if isinstance(metric, VaxInducedImmunityGeneric):
                # Use the init_val setter so current_val is also updated
                # immediately, before super()'s reset loop overwrites it again
                # (harmlessly).
                metric.init_val = metric.adjust_initial_value(
                    metric.original_init_val,
                    self.start_real_date,
                    metric.update_config,
                    self.params,
                    schedules_dict,
                    self.simulation_settings.timesteps_per_day,
                )
            elif isinstance(metric, InfInducedImmunityGeneric):
                metric.init_val = metric.adjust_initial_value(
                    metric.original_init_val,
                    self.start_real_date,
                    self.params,
                    self.simulation_settings.timesteps_per_day,
                )

        self.update_ve_inflation_factors()
        self.update_infection_immunity_injection_val()

        super().reset_simulation()

    # -----------------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------------

    def create_compartments(self) -> sc.objdict:
        """
        Create one Compartment per name declared in model_config.compartments.
        Initial values come from _state_init.compartments (populated by caller),
        adjusted for any pre-simulation history of scheduled_exact transitions
        (see _compute_scheduled_exact_pre_simulation_adjustments).
        """
        compartments = sc.objdict()
        A, R = self.params.num_age_groups, self.params.num_risk_groups
        adjustments = self._compute_scheduled_exact_pre_simulation_adjustments()
        for name in self.model_config.compartments:
            init = np.asarray(
                self._state_init._cvals.get(name, np.zeros((A, R))),
                dtype=float,
            )
            if name in adjustments:
                init = np.clip(init + adjustments[name], 0.0, None)
            compartments[name] = clt.Compartment(init)
        return compartments

    def _compute_scheduled_exact_pre_simulation_adjustments(self) -> dict[str, np.ndarray]:
        """
        Move counts from origin to destination to account for a scheduled_exact
        transition's history before the simulation start date.

        A scheduled_exact transition (e.g. vaccination) moves
        rint(schedule_value * origin_val) people from origin to destination on
        each simulated day (ScheduledTransferVariable.get_scheduled_exact_realization)
        -- schedule_value is a daily *proportion* of the origin compartment, not
        an absolute count. If the schedule's timeseries starts before
        current_real_date (e.g. a vaccine CSV starting well before a fitted
        simulation start date), those days are never simulated and the
        transfers they represent are otherwise lost entirely.

        This replays that same exact-realization rule day by day, starting
        from the origin's caller-supplied initial value, for all schedule
        dates between a reset date and current_real_date, and returns the
        resulting cumulative origin -> destination movement to apply to the
        initial compartment values. This mirrors
        VaxInducedImmunityGeneric._adjust_initial_value's pre-simulation
        integration, but replays the actual transfer rule instead of a wane
        decay (the compartment is a stock with no decay, unlike the MV
        metric). It assumes the origin compartment is depleted only by this
        transition during the lookback window (no other inflows/outflows are
        replayed), the same simplifying assumption MV's adjustment makes.

        The reset date comes from the param named by
        'compartment_reset_date_mm_dd_param' in rate_config (a "MM_DD"
        string); if that param is unset/None, all schedule history before
        current_real_date is replayed. If rate_config omits the key entirely,
        no adjustment is made for that transition (existing behavior, opt-in
        per transition).
        """
        adjustments: dict[str, np.ndarray] = {}
        for tc in self.model_config.transitions:
            if tc.rate_template != "scheduled_exact":
                continue
            if "compartment_reset_date_mm_dd_param" not in tc.rate_config:
                continue

            schedule_name = tc.rate_config["schedule"]
            schedule = self.schedules.get(schedule_name)
            timeseries_df = getattr(schedule, "timeseries_df", None)
            if timeseries_df is None:
                continue
            value_column = getattr(schedule, "value_column", "daily_vaccines")

            mask = timeseries_df.index < self.current_real_date

            reset_param = tc.rate_config["compartment_reset_date_mm_dd_param"]
            reset_date_str = self.params.params.get(reset_param) if reset_param else None
            if reset_date_str:
                month, day = reset_date_str.split("_")
                year = self.current_real_date.year
                reset_date = datetime.date(year, int(month), int(day))
                if reset_date >= self.current_real_date:
                    reset_date = datetime.date(year - 1, int(month), int(day))

                schedule_config = next(
                    (sc_cfg.schedule_config for sc_cfg in self.model_config.schedules
                     if sc_cfg.name == schedule_name),
                    {},
                )
                delay_param = schedule_config.get("vax_protection_delay_days_param")
                delay_days = int(self.params.params.get(delay_param, 0)) if delay_param else 0
                # Values in timeseries_df are already shifted by delay_days
                reset_date = reset_date + datetime.timedelta(days=delay_days)
                mask &= timeseries_df.index >= reset_date

            relevant = timeseries_df.loc[mask, value_column].sort_index()
            if len(relevant) == 0:
                continue

            A, R = self.params.num_age_groups, self.params.num_risk_groups
            remaining = np.asarray(
                self._state_init._cvals.get(tc.origin, np.zeros((A, R))), dtype=float
            )
            moved_total = np.zeros_like(remaining)
            for proportion in relevant:
                # Same vax_pool="susceptible" rule as get_scheduled_exact_realization:
                # proportion applies to origin+destination (remaining + moved_total).
                pool = remaining + moved_total
                moved = np.minimum(np.rint(np.asarray(proportion, dtype=float) * pool), remaining)
                remaining = remaining - moved
                moved_total = moved_total + moved

            adjustments.setdefault(tc.destination, np.zeros_like(moved_total))
            adjustments[tc.destination] = adjustments[tc.destination] + moved_total
            adjustments.setdefault(tc.origin, np.zeros_like(moved_total))
            adjustments[tc.origin] = adjustments[tc.origin] - moved_total
        return adjustments

    def create_transition_variables(self) -> sc.objdict:
        """
        Create one ConfigDrivenTransitionVariable per transition in model_config.
        """
        transition_type = self.simulation_settings.transition_type
        tvars = sc.objdict()

        # A transition is jointly distributed when its realization comes from a
        # TransitionVariableGroup's joint draw rather than its own marginal one.
        # Group membership is what actually determines that, so derive it from
        # transition_groups (the pairwise `jointly_distributed_with` field is
        # still honoured). Reading only that field meant a config declaring the
        # groups but omitting it built the groups and then silently overwrote
        # every joint realization with an independent marginal draw -- see
        # SubpopModel.sample_transitions, which skips only is_jointly_distributed
        # variables.
        grouped_names = {
            m for gc in self.model_config.transition_groups for m in gc.members
        }

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
            is_joint = tc.jointly_distributed_with is not None or tc.name in grouped_names
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
        Override to also apply the vaccine-immunity reset and the
        infection-immunity injection at the start of each day.

        Mirrors FluSubpopModel.prepare_daily_state.
        """
        super().prepare_daily_state()

        # Refresh the derived VE inflation factors once, as the run starts.
        #   By now any post-reset parameter override (a fitting draw, a
        #   scenario override) is in place, which is not true at
        #   `reset_simulation` time -- see `update_ve_inflation_factors`.
        #   Params only; no state is touched, so this cannot disturb a
        #   manually set `current_val`.
        if self.current_simulation_day == 0:
            self.update_ve_inflation_factors(warn=False)

        injection_fired = False

        for metric in self.epi_metrics.values():
            if isinstance(metric, VaxInducedImmunityGeneric):
                metric.check_and_apply_reset(self.current_real_date, self.params)
            elif isinstance(metric, InfInducedImmunityGeneric):
                was_pending = metric.pending_injection_date is not None
                metric.check_and_apply_injection(self.current_real_date, self.params)
                if was_pending and metric.pending_injection_date is None:
                    injection_fired = True

        # An injection that just fired is spent, so zero its mirrored params
        #   value -- otherwise `params` would keep claiming an injection is
        #   pending after the metric knows better, and anything reading params
        #   mid-run (notably `build_generic_torch_inputs`) would apply it a
        #   second time. `reset_simulation` restores it for the next run.
        if injection_fired:
            self.update_infection_immunity_injection_val()

        # The reset/injection above set `current_val` directly on the epi
        #   metric objects -- sync `self.state` immediately so that today's
        #   update (which reads state.epi_metrics, not the metric's own
        #   current_val) sees the up-to-date value on its very first
        #   timestep, instead of a stale pre-reset/pre-injection value.
        self.state.sync_to_current_vals(self.epi_metrics)


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
