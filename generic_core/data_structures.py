"""
data_structures.py — Generic containers for subpopulation state, parameters,
and travel tensors.

Design note — GenericSubpopState
---------------------------------
The base class SubpopModel.__init__ assigns:
    self.state.compartments = self.compartments   # sc.objdict of Compartment objects
    self.state.epi_metrics  = self.epi_metrics    # sc.objdict of EpiMetric objects
    self.state.schedules    = self.schedules       # sc.objdict of Schedule objects
    self.state.dynamic_vals = self.dynamic_vals   # sc.objdict of DynamicVal objects

For FluSubpopState the state stores numpy arrays as direct attributes (state.S,
state.E, ...) that are set by sync_to_current_vals via setattr. The sc.objdict
assignments above just add extra references.

GenericSubpopState cannot use typed dataclass fields because compartment names
come from a JSON config. Instead it uses four private numpy-array dicts
(_cvals, _evals, _svals, _dvals) exposed as properties (compartments,
epi_metrics, schedules, dynamic_vals). The property setters are no-ops so that
the base class assignments above are silently ignored. sync_to_current_vals is
overridden to route values into the correct private dict using pre-populated
name sets.

This means:
    state.compartments["S"]       — numpy array  ✓ (used by rate templates)
    state.epi_metrics["M"]        — numpy array  ✓ (used by rate templates)
    state.schedules["flu_contact_matrix"] — numpy array  ✓
"""

from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

import clt_toolkit as clt


# A "MM_DD" parameter describing a date near the simulation start that
#   resolves further than this many days away is almost certainly a
#   mis-specification rather than a deliberate choice -- see
#   `resolve_mm_dd_near_date`
INFECTION_IMMUNITY_START_DATE_WARN_DAYS = 120


def resolve_mm_dd_near_date(mm_dd: str,
                            reference_date: datetime.date,
                            param_name: str = "date",
                            warn_days: Optional[int] = INFECTION_IMMUNITY_START_DATE_WARN_DAYS
                            ) -> datetime.date:
    """
    Resolves a year-less "MM_DD" parameter to the actual calendar date
    closest to `reference_date`.

    A "MM_DD" string is ambiguous about which year it means, and simply
    reusing `reference_date`'s year silently picks the wrong one for any
    season that spans a new year. For a simulation starting 2024-08-15,
    "01_01" under that rule resolves to 2024-01-01 -- 227 days *before*
    the start -- when the intended date is almost certainly 2025-01-01,
    139 days after it. Choosing whichever of the previous, current, and
    next year lands closest to `reference_date` resolves both cases the
    way a user would expect, and is unambiguous as long as the intended
    date is within ~6 months of the reference date.

    NOTE: this duplicates `flu_core.flu_data_structures.resolve_mm_dd_near_date`
    -- generic_core deliberately does not import flu_core for core logic.
    See generic_core/limitations.md.

    Args:
        mm_dd (str):
            date in "MM_DD" format, e.g. "01_01".
        reference_date (datetime.date):
            the date to resolve relative to -- generally the
            simulation's `start_real_date`.
        param_name (str):
            name of the parameter being resolved, used in messages.
        warn_days (Optional[int]):
            warn if the resolved date is more than this many days from
            `reference_date` -- pass None to skip the check.

    Returns:
        datetime.date
    """

    month, day = (int(x) for x in mm_dd.split("_"))

    candidates = []
    for year in (reference_date.year - 1, reference_date.year, reference_date.year + 1):
        try:
            candidates.append(datetime.date(year, month, day))
        except ValueError:
            # Feb 29 in a non-leap year -- that year simply has no such
            #   date, so it is not a candidate
            continue

    if not candidates:
        raise ValueError(
            f"`{param_name}` = '{mm_dd}' does not correspond to a real date in any "
            f"year adjacent to {reference_date}.")

    resolved = min(candidates, key=lambda d: abs((d - reference_date).days))

    if warn_days is not None and abs((resolved - reference_date).days) > warn_days:
        warnings.warn(
            f"`{param_name}` = '{mm_dd}' resolved to {resolved}, which is "
            f"{abs((resolved - reference_date).days)} days from {reference_date}. "
            f"This parameter is meant to describe a date near the simulation start "
            f"(within a couple of months); a gap this large usually means the "
            f"month/day is mis-specified.")

    return resolved


# ---------------------------------------------------------------------------
# GenericSubpopState
# ---------------------------------------------------------------------------

class GenericSubpopState(clt.SubpopState):
    """
    Config-driven subpopulation state container.

    Attributes accessed by rate/metric templates
    --------------------------------------------
    compartments : dict[str, np.ndarray]   (property)
    epi_metrics  : dict[str, np.ndarray]   (property)
    schedules    : dict[str, Any]          (property)
    dynamic_vals : dict[str, Any]          (property)

    Parameters
    ----------
    compartment_names : iterable[str]
        Names of all compartments declared in the model config.
    epi_metric_names : iterable[str]
        Names of all epi metrics declared in the model config.
    schedule_names : iterable[str]
        Names of all schedules declared in the model config.
    dynamic_val_names : iterable[str]
        Names of all dynamic vals declared in the model config.
    """

    def __init__(
        self,
        compartment_names,
        epi_metric_names=(),
        schedule_names=(),
        dynamic_val_names=(),
    ):
        # Note: clt.SubpopState is a dataclass with no required __init__ args.
        # We do NOT call super().__init__() because SubpopState.__init__ takes
        # no arguments and just sets _init_val=None. We replicate that here.
        self._init_val = None
        self.current_val = None
        self.history_vals_list = []

        # Name sets for routing in sync_to_current_vals
        self._compartment_names: frozenset[str] = frozenset(compartment_names)
        self._epi_metric_names: frozenset[str] = frozenset(epi_metric_names)
        self._schedule_names: frozenset[str] = frozenset(schedule_names)
        self._dynamic_val_names: frozenset[str] = frozenset(dynamic_val_names)

        # Private dicts holding numpy arrays (the actual state)
        self._cvals: dict[str, Any] = {}
        self._evals: dict[str, Any] = {}
        self._svals: dict[str, Any] = {}
        self._dvals: dict[str, Any] = {}

    # --- Properties with no-op setters ----------------------------------
    # The no-op setters are needed because SubpopModel.__init__ does:
    #   self.state.compartments = self.compartments  (sc.objdict of Compartment objs)
    # We silently ignore those assignments; _cvals holds the numpy arrays.

    @property
    def compartments(self) -> dict[str, Any]:
        return self._cvals

    @compartments.setter
    def compartments(self, value) -> None:
        pass  # Intentionally ignore base-class assignment

    @property
    def epi_metrics(self) -> dict[str, Any]:
        return self._evals

    @epi_metrics.setter
    def epi_metrics(self, value) -> None:
        pass

    @property
    def schedules(self) -> dict[str, Any]:
        return self._svals

    @schedules.setter
    def schedules(self, value) -> None:
        pass

    @property
    def dynamic_vals(self) -> dict[str, Any]:
        return self._dvals

    @dynamic_vals.setter
    def dynamic_vals(self, value) -> None:
        pass

    # --- sync_to_current_vals -------------------------------------------

    def sync_to_current_vals(self, lookup_dict: dict) -> None:
        """
        Copy current_val from each StateVariable into the appropriate sub-dict.

        Called by SubpopModel.simulate_until_day with all_state_variables,
        compartments, epi_metrics, schedules, or dynamic_vals.
        """
        for name, item in lookup_dict.items():
            val = item.current_val
            if name in self._compartment_names:
                self._cvals[name] = val
            elif name in self._epi_metric_names:
                self._evals[name] = val
            elif name in self._schedule_names:
                self._svals[name] = val
            elif name in self._dynamic_val_names:
                self._dvals[name] = val
            # else: name not in any known set — silently skip
            #       (handles the case where SubpopModel passes all_state_variables
            #        which may include entries not in any of our name sets)


# ---------------------------------------------------------------------------
# GenericSubpopParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenericSubpopParams(clt.SubpopParams):
    """
    Config-driven subpopulation parameter container.

    All model parameters are stored in a single flat dict keyed by name.
    The num_age_groups, num_risk_groups, and total_pop_age_risk fields are
    separated out because the base class machinery (compute_total_pop_age_risk,
    etc.) needs them as explicit attributes.

    Attributes
    ----------
    params : dict[str, Any]
        All model parameters keyed by name.
        Example: {"beta_baseline": 0.3, "R_to_S_rate": 0.01, ...}
    num_age_groups : int
    num_risk_groups : int
    total_pop_age_risk : np.ndarray of shape (A, R)
        Updated by SubpopModel.compute_total_pop_age_risk() after construction.
    """

    params: dict[str, Any] = field(default_factory=dict)
    num_age_groups: int = 1
    num_risk_groups: int = 1
    total_pop_age_risk: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1))
    )


# ---------------------------------------------------------------------------
# GenericTravelTensors
# ---------------------------------------------------------------------------

@dataclass
class GenericTravelTensors:
    """
    Dict-of-tensors container used by generic travel functions.

    Replaces flu-specific FluTravelStateTensors / FluFullMetapopStateTensors
    with dict-based equivalents whose keys come from the model config.

    Attributes
    ----------
    compartment_tensors : dict[str, torch.Tensor]
        Compartment tensors, each of shape (L, A, R).
    schedule_tensors : dict[str, torch.Tensor]
        Schedule tensors, each of shape (L, A, R) or broadcastable.
    param_tensors : dict[str, torch.Tensor]
        Parameter tensors, each of shape (L, A, R) or broadcastable.
    """

    compartment_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    schedule_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    param_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
