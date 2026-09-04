"""
metric_templates.py — MetricTemplate ABC and built-in concrete implementations.

Each template is a factory that constructs a concrete EpiMetric subclass.
The factory pattern is necessary because VaxInducedImmunity needs extra
constructor arguments (schedules, dates) that are not known until the model
is constructed.

Registry
--------
METRIC_TEMPLATE_REGISTRY : dict[str, MetricTemplate]
    Built-in templates are registered at import time.
    Users call register_metric_template() to add custom templates.

Reference implementations:
    InfInducedImmunity  — flu_core/flu_components.py:407–451
    VaxInducedImmunity  — flu_core/flu_components.py:454–566
"""

from __future__ import annotations

import copy
import datetime
import warnings
from abc import ABC, abstractmethod

from typing import Any

import numpy as np

import clt_toolkit as clt

from .data_structures import resolve_mm_dd_near_date


def injection_val_param_name(metric_name: str) -> str:
    """
    Name of the derived param carrying an infection-induced immunity
    metric's pending injection value.

    The numpy model reads the pending injection off the metric object
    itself; the torch model has no metric objects, so
    ConfigDrivenSubpopModel mirrors the value onto params under this name
    and torch_generic.check_and_apply_M_injection reads it back. The
    leading underscore marks it as derived, not user-supplied.
    """
    return f"_{metric_name}_injection_val"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MetricTemplate(ABC):
    """
    Abstract base class for metric templates.

    Subclasses implement:
        validate_config  — called at model construction to catch config errors
        build_metric     — factory method that returns a concrete EpiMetric instance
    """

    @abstractmethod
    def validate_config(
        self,
        update_config: dict,
        param_names: set,
        transition_names: set,
    ) -> None:
        """
        Validate that update_config references known params and transitions.
        Raises ValueError with a descriptive message on the first problem found.
        """

    @abstractmethod
    def build_metric(
        self,
        init_val: np.ndarray,
        update_config: dict,
        params,
        transition_variables: dict,
        schedules: dict,
        timesteps_per_day: int,
    ) -> clt.EpiMetric:
        """
        Construct and return a concrete EpiMetric instance.

        Parameters
        ----------
        init_val : np.ndarray
            Initial value of shape (A, R).
        update_config : dict
            Template-specific configuration dict from the model config.
        params : GenericSubpopParams
            Model parameters (used by VaxInducedImmunityGeneric for init
            adjustment).
        transition_variables : dict[str, TransitionVariable]
            All transition variable instances in the model (used by
            InfInducedImmunityGeneric to get the R_to_S reference).
        schedules : dict[str, Schedule]
            All schedule instances in the model (used by
            VaxInducedImmunityGeneric for vaccine schedule access).
        timesteps_per_day : int
            Used by VaxInducedImmunityGeneric for initial value adjustment.
        """


# ---------------------------------------------------------------------------
# InfInducedImmunityGeneric
# ---------------------------------------------------------------------------

class InfInducedImmunityGeneric(clt.EpiMetric):
    """
    Infection-induced population-level immunity.

    Mirrors InfInducedImmunity from flu_components.py:407–451.

    Population-level immunity increases as people move from R to S
    (recently-recovered individuals re-entering the susceptible pool
    still carry partial immunity).

    Configured via update_config:
        r_to_s_transition : str
            Name of the R→S transition variable.
        inf_induced_saturation_param : str
            Parameter name for inf_induced_saturation.
        inf_induced_immune_wane_param : str
            Parameter name for inf_induced_immune_wane.
        infection_immunity_start_date_mm_dd_param : str, optional
            Parameter name holding a "MM_DD" string (or None) giving the
            date that the input M(0) corresponds to — see
            `adjust_initial_value`.
    """

    def __init__(
        self,
        init_val: np.ndarray,
        R_to_S: clt.TransitionVariable,
        update_config: dict,
        params,
        current_real_date: datetime.date,
        timesteps_per_day: int,
    ):
        self.R_to_S = R_to_S
        self.update_config = update_config
        self.params = params
        self.pending_injection_date = None

        adjusted_init_val = self.adjust_initial_value(
            init_val, current_real_date, params, timesteps_per_day
        )
        super().__init__(adjusted_init_val)

    def adjust_initial_value(
        self,
        init_val: np.ndarray,
        current_real_date: datetime.date,
        params,
        timesteps_per_day: int,
    ) -> np.ndarray:
        """
        Adjusts the initial value of infection-induced immunity based on
        `infection_immunity_start_date_mm_dd_param`, the date that
        init_val (M(0)) corresponds to.

        If that parameter is unset, or resolves to `current_real_date`,
        init_val is used as-is. If it resolves to a date before
        `current_real_date`, init_val is decayed forward (waning only) to
        `current_real_date`. If it resolves to a date after, the adjusted
        initial value is zero and init_val is instead added to
        `current_val` once that date is reached (see
        `check_and_apply_injection`).

        The "MM_DD" string carries no year, so the year is resolved by
        `resolve_mm_dd_near_date` — see that function for why the
        nearest-year rule matters for seasons that span a new year.

        Mirrors InfInducedImmunity.adjust_initial_value in
        flu_core/flu_components.py.
        """

        self.original_init_val = copy.deepcopy(init_val)
        self.adjusted_init_val = copy.deepcopy(init_val)
        self.pending_injection_date = None

        start_param = self.update_config.get("infection_immunity_start_date_mm_dd_param")
        start_date_str = params.params.get(start_param) if start_param else None

        if start_date_str is not None:

            immunity_start_date = resolve_mm_dd_near_date(
                start_date_str,
                current_real_date,
                param_name=start_param)

            if immunity_start_date < current_real_date:
                # init_val is a past value -- decay it forward
                #   (waning only) to current_real_date
                wane = params.params[self.update_config["inf_induced_immune_wane_param"]]
                num_days = (current_real_date - immunity_start_date).days
                for _ in range(num_days):
                    for _ in range(timesteps_per_day):
                        self.adjusted_init_val = self.adjusted_init_val - \
                            wane * self.adjusted_init_val / timesteps_per_day
            elif immunity_start_date > current_real_date:
                # init_val is a future value -- start at 0 and add
                #   init_val once immunity_start_date is reached
                self.adjusted_init_val = np.zeros_like(self.adjusted_init_val)
                self.pending_injection_date = immunity_start_date

        return self.adjusted_init_val

    def check_and_apply_injection(self, current_date: datetime.date, params) -> None:
        """
        If `current_date` matches the pending injection date, add the
        original initial value to `current_val` (one time only).

        Mirrors InfInducedImmunity.check_and_apply_injection.
        """

        if self.pending_injection_date is not None and \
                current_date == self.pending_injection_date:
            self.current_val = self.current_val + self.original_init_val
            self.pending_injection_date = None
            print(f"InfInducedImmunityGeneric increased by initial value on {current_date}")

    def get_change_in_current_val(self, state: Any, params: Any, num_timesteps: int) -> np.ndarray:
        """
        Returns (A, R) change in infection-induced immunity.

        Mirrors InfInducedImmunity.get_change_in_current_val.
        Note: R_to_S.current_val already includes discretization.
        """
        inf_sat = params.params[self.update_config["inf_induced_saturation_param"]]
        wane = params.params[self.update_config["inf_induced_immune_wane_param"]]

        M = state.epi_metrics["M"]

        return (
            (self.R_to_S.current_val / params.total_pop_age_risk)
            * (1.0 - inf_sat * M)
            - wane * M / num_timesteps
        )


class InfectionInducedImmunityTemplate(MetricTemplate):
    """
    Factory for InfInducedImmunityGeneric.

    update_config keys
    ------------------
    r_to_s_transition : str
        Name of the R→S transition variable in the model.
    inf_induced_saturation_param : str
    inf_induced_immune_wane_param : str
    infection_immunity_start_date_mm_dd_param : str, optional
        Parameter name holding a "MM_DD" string (or None) giving the date
        that the configured init_val corresponds to.

    NOTE: `vax_induced_saturation_param` was REMOVED — vaccine-induced
    immunity no longer damps the M update. See generic_core/limitations.md.
    """

    _REQUIRED = (
        "r_to_s_transition",
        "inf_induced_saturation_param",
        "inf_induced_immune_wane_param",
    )

    def validate_config(self, update_config, param_names, transition_names):
        for key in self._REQUIRED:
            if key not in update_config:
                raise ValueError(
                    f"InfectionInducedImmunityTemplate: missing key '{key}' in update_config"
                )
        tname = update_config["r_to_s_transition"]
        if tname not in transition_names:
            raise ValueError(
                f"InfectionInducedImmunityTemplate: transition '{tname}' not found in model"
            )
        for pkey in ("inf_induced_saturation_param", "inf_induced_immune_wane_param"):
            pname = update_config[pkey]
            if pname not in param_names:
                raise ValueError(
                    f"InfectionInducedImmunityTemplate: param '{pname}' (from '{pkey}') not in model params"
                )
        if "vax_induced_saturation_param" in update_config:
            raise ValueError(
                "InfectionInducedImmunityTemplate: 'vax_induced_saturation_param' is no "
                "longer supported -- vaccine-induced immunity no longer damps the "
                "infection-induced immunity update. Remove the key from update_config."
            )
        start_param = update_config.get("infection_immunity_start_date_mm_dd_param")
        if start_param is not None and start_param not in param_names:
            raise ValueError(
                f"InfectionInducedImmunityTemplate: param '{start_param}' (from "
                "'infection_immunity_start_date_mm_dd_param') not in model params"
            )

    def build_metric(self, init_val, update_config, params, transition_variables, schedules, timesteps_per_day):
        R_to_S = transition_variables[update_config["r_to_s_transition"]]
        current_real_date = update_config.get("_current_real_date")
        if current_real_date is None:
            raise ValueError(
                "InfectionInducedImmunityTemplate: '_current_real_date' must be set in "
                "update_config before calling build_metric "
                "(done automatically by ConfigDrivenSubpopModel)"
            )
        return InfInducedImmunityGeneric(
            init_val,
            R_to_S,
            update_config,
            params,
            current_real_date,
            timesteps_per_day,
        )


# ---------------------------------------------------------------------------
# VaxInducedImmunityGeneric
# ---------------------------------------------------------------------------

class VaxInducedImmunityGeneric(clt.EpiMetric):
    """
    Vaccine-induced population-level immunity.

    Mirrors VaxInducedImmunity from flu_components.py:454–566.

    Configured via update_config:
        daily_vaccines_schedule : str
            Name of the daily vaccines schedule.
        vax_induced_immune_wane_param : str
            Parameter name for vax_induced_immune_wane.
        vax_immunity_reset_date_mm_dd_param : str | None
            Parameter name for the reset date string ("MM_DD"), or None
            if no reset is used.
        vax_protection_delay_days_param : str | None
            Unused here — the protection delay is applied once, by
            VaccineScheduleTemplate, when building the schedule.
    """

    def __init__(
        self,
        init_val: np.ndarray,
        current_real_date: datetime.date,
        update_config: dict,
        params,
        schedules: dict,
        timesteps_per_day: int,
    ):
        self.update_config = update_config
        adjusted_init_val = self.adjust_initial_value(
            init_val, current_real_date, update_config, params, schedules, timesteps_per_day
        )
        super().__init__(adjusted_init_val)

    def adjust_initial_value(
        self,
        init_val: np.ndarray,
        current_real_date: datetime.date,
        update_config: dict,
        params,
        schedules: dict,
        timesteps_per_day: int,
    ) -> np.ndarray:
        """
        Mirrors VaxInducedImmunity.adjust_initial_value.

        If vax_immunity_reset_date_mm_dd_param is set (points to a non-None
        parameter), counts vaccines administered after the reset date up to
        the simulation start, applying waning, and adds this adjustment to
        init_val.

        `original_init_val` is stored so that reset_simulation can re-derive
        from the unmodified config value without adjustments compounding
        across calls.
        """
        self.original_init_val = copy.deepcopy(init_val)

        reset_param = update_config.get("vax_immunity_reset_date_mm_dd_param")
        reset_date_str = None
        if reset_param is not None:
            reset_date_str = params.params.get(reset_param)

        if reset_date_str is None:
            return copy.deepcopy(init_val)

        msg = (
            f"Vaccine immunity reset date is set as "
            f"{reset_date_str.replace('_', '/')}.\n"
            "Initial vaccine-induced immunity value is being adjusted "
            "by resetting immunity to 0 at that date, and by taking into "
            "account vaccines administered after this date, and before simulation start date."
        )
        warnings.warn(msg)

        month, day = reset_date_str.split("_")
        current_year = current_real_date.year
        reset_date = datetime.date(current_year, int(month), int(day))
        if reset_date >= current_real_date:
            reset_date = datetime.date(current_year - 1, int(month), int(day))

        vaccines_schedule = schedules[update_config["daily_vaccines_schedule"]]
        vaccines_df = vaccines_schedule.timeseries_df.copy()

        # The schedule index is already the PROTECTION date --
        #   VaccineScheduleGeneric.preprocess has shifted every date forward
        #   by vax_protection_delay_days. Adding the delay to `reset_date`
        #   here as well would drop the doses that become protective during
        #   the delay window just after the reset date.
        mask = (vaccines_df.index >= reset_date) & (vaccines_df.index < current_real_date)
        relevant_vaccines = vaccines_df[mask]

        wane = params.params[update_config["vax_induced_immune_wane_param"]]
        MV_adjustment = np.zeros_like(vaccines_df["daily_vaccines"].iloc[0])
        for _, row in relevant_vaccines.iterrows():
            for _ in range(timesteps_per_day):
                MV_adjustment += (
                    row["daily_vaccines"] / timesteps_per_day
                    - wane * MV_adjustment / timesteps_per_day
                )

        return copy.deepcopy(init_val) + MV_adjustment

    def get_change_in_current_val(self, state: Any, params: Any, num_timesteps: int) -> np.ndarray:
        """
        Returns (A, R) change in vaccine-induced immunity.

        Mirrors VaxInducedImmunity.get_change_in_current_val.
        """
        wane = params.params[self.update_config["vax_induced_immune_wane_param"]]
        daily_vaccines = state.schedules[self.update_config["daily_vaccines_schedule"]]
        MV = state.epi_metrics["MV"]
        return daily_vaccines / num_timesteps - wane * MV / num_timesteps

    def check_and_apply_reset(self, current_date: datetime.date, params) -> None:
        """
        Reset MV to zero on the anniversary of vax_immunity_reset_date_mm_dd.

        Mirrors VaxInducedImmunity.check_and_apply_reset.
        """
        reset_param = self.update_config.get("vax_immunity_reset_date_mm_dd_param")
        if reset_param is None:
            return
        reset_date_str = params.params.get(reset_param)
        if reset_date_str is None:
            return

        month, day = reset_date_str.split("_")
        if current_date.month == int(month) and current_date.day == int(day):
            self.current_val = np.zeros_like(self.current_val)
            print(f"VaxInducedImmunityGeneric reset to 0 on {current_date}")


class VaccineInducedImmunityTemplate(MetricTemplate):
    """
    Factory for VaxInducedImmunityGeneric.

    update_config keys
    ------------------
    daily_vaccines_schedule : str
        Name of the daily vaccines schedule in the model.
    vax_induced_immune_wane_param : str
        Parameter name for the waning rate.
    vax_immunity_reset_date_mm_dd_param : str | None
        Parameter name for the reset date (value is "MM_DD" string), or
        None/absent if no reset is used.
    vax_protection_delay_days_param : str | None
        Unused here — the protection delay is applied once, by
        VaccineScheduleTemplate, when building the schedule.
    """

    _REQUIRED = (
        "daily_vaccines_schedule",
        "vax_induced_immune_wane_param",
    )

    def validate_config(self, update_config, param_names, transition_names):
        for key in self._REQUIRED:
            if key not in update_config:
                raise ValueError(
                    f"VaccineInducedImmunityTemplate: missing key '{key}' in update_config"
                )
        wane_param = update_config["vax_induced_immune_wane_param"]
        if wane_param not in param_names:
            raise ValueError(
                f"VaccineInducedImmunityTemplate: param '{wane_param}' not in model params"
            )
        reset_param = update_config.get("vax_immunity_reset_date_mm_dd_param")
        if reset_param is not None and reset_param not in param_names:
            raise ValueError(
                f"VaccineInducedImmunityTemplate: reset param '{reset_param}' not in model params"
            )
        delay_param = update_config.get("vax_protection_delay_days_param")
        if delay_param is not None and delay_param not in param_names:
            raise ValueError(
                f"VaccineInducedImmunityTemplate: delay param '{delay_param}' not in model params"
            )

    def build_metric(self, init_val, update_config, params, transition_variables, schedules, timesteps_per_day):
        import datetime as dt
        # current_real_date must be provided via update_config at build time
        # (set by ConfigDrivenSubpopModel before calling build_metric)
        current_real_date = update_config.get("_current_real_date")
        if current_real_date is None:
            raise ValueError(
                "VaccineInducedImmunityTemplate: '_current_real_date' must be set in "
                "update_config before calling build_metric "
                "(done automatically by ConfigDrivenSubpopModel)"
            )
        return VaxInducedImmunityGeneric(
            init_val,
            current_real_date,
            update_config,
            params,
            schedules,
            timesteps_per_day,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRIC_TEMPLATE_REGISTRY: dict[str, MetricTemplate] = {}


def register_metric_template(name: str, instance: MetricTemplate) -> None:
    """
    Add a MetricTemplate instance to the global registry under `name`.

    Built-in templates are registered at import time.
    """
    METRIC_TEMPLATE_REGISTRY[name] = instance


register_metric_template("infection_induced_immunity", InfectionInducedImmunityTemplate())
register_metric_template("vaccine_induced_immunity", VaccineInducedImmunityTemplate())
