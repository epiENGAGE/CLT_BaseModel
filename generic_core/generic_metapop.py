"""
generic_metapop.py — Config-driven metapopulation model.

Classes
-------
ConfigDrivenMetapopModel
    MetapopModel subclass whose inter-subpopulation travel computation is driven
    entirely by config dicts rather than flu-specific typed dataclasses.

Mirrors FluMetapopModel (flu_core/flu_components.py:1387+) but replaces all
field-access on FluTravelStateTensors / FluTravelParamsTensors with dict-lookup
via generic_core/travel_functions.py.

Usage
-----
travel_config = {
    "infectious_compartments": {"ISR": None, "ISH": None, "IP": "IP_relative_inf", "IA": "IA_relative_inf"},
    "immobile_compartments":   ["HR", "HD"],
    "relative_susceptibility_param": "relative_suscept",
    "contact_matrix_schedule": "flu_contact_matrix",
    "mobility_schedule":       "mobility_modifier",
}

model = ConfigDrivenMetapopModel(
    subpop_models=[subpop1, subpop2],
    mixing_params=flu_mixing_params,   # any object with .travel_proportions and .num_locations
    model_config=model_config,
    travel_config=travel_config,
)
model.simulate_until_day(N)
"""

from __future__ import annotations

import numpy as np
import torch

import clt_toolkit as clt

from .config_parser import ModelConfig
from .travel_functions import GenericPrecomputedTensors, compute_total_mixing_exposure


class ConfigDrivenMetapopModel(clt.MetapopModel):
    """
    MetapopModel driven by ModelConfig + travel_config dicts.

    Parameters
    ----------
    subpop_models : list[ConfigDrivenSubpopModel]
        Constituent subpopulation models. All must share the same
        model_config structure and simulation settings.
    mixing_params : object
        Must expose:
            .travel_proportions : np.ndarray of shape (L, L)
            .num_locations      : int
    model_config : ModelConfig
        Validated config from parse_model_config(); used to discover
        which transitions use force_of_infection_travel.
    travel_config : dict
        Keys required by compute_total_mixing_exposure:
            infectious_compartments      : dict[str, str|None]
            immobile_compartments        : list[str]
            relative_susceptibility_param: str
            contact_matrix_schedule      : str   (name of Schedule in subpop)
            mobility_schedule            : str   (name of Schedule in subpop)
    name : str
        Optional identifier for this metapopulation model.
    """

    def __init__(
        self,
        subpop_models: list,
        mixing_params,
        model_config: ModelConfig,
        travel_config: dict,
        name: str = "",
    ) -> None:
        # MetapopModel.__init__ validates settings consistency and sets
        # self.subpop_models, self._subpop_models_ordered, self.mixing_params,
        # then calls self.run_input_checks().
        super().__init__(subpop_models, mixing_params, name)

        self.model_config = model_config
        self.travel_config = travel_config

        # Store travel proportions as a float64 tensor for travel computations
        travel_proportions = torch.tensor(
            np.asarray(mixing_params.travel_proportions, dtype=float)
        )
        self._travel_proportions_tensor = travel_proportions

        # Precompute population totals and travel-proportion summaries once at init.
        # (Mirrors FluMetapopModel which calls compute_total_pop_LAR_tensor in __init__.)
        total_pop_LAR_tensor = self.compute_total_pop_LAR_tensor()
        self.precomputed = GenericPrecomputedTensors(total_pop_LAR_tensor, travel_proportions)

        # Find all transitions that use the travel rate template; we'll inject
        # _total_mixing_exposure into their rate_config each timestep.
        self._travel_transition_names = [
            tc.name
            for tc in model_config.transitions
            if tc.rate_template == "force_of_infection_travel"
        ]

    # -----------------------------------------------------------------------
    # MetapopModel interface
    # -----------------------------------------------------------------------

    def run_input_checks(self) -> None:
        """Validate travel_proportions rows sum to 1."""
        travel_proportions = np.asarray(self.mixing_params.travel_proportions)
        if np.any(travel_proportions < 0):
            raise ValueError(
                "ConfigDrivenMetapopModel: all entries of travel_proportions must be non-negative."
            )
        if not np.allclose(travel_proportions.sum(axis=1), 1):
            raise ValueError(
                "ConfigDrivenMetapopModel: rows of travel_proportions must each sum to 1."
            )

    def compute_total_pop_LAR_tensor(self) -> torch.Tensor:
        """
        Sum initial compartment values across all subpopulations.

        Returns
        -------
        torch.Tensor of shape (L, A, R)
        """
        subpop_models = self._subpop_models_ordered
        first = list(subpop_models.values())[0]
        A = first.params.num_age_groups
        R = first.params.num_risk_groups
        L = len(subpop_models)

        total_pop = torch.zeros(L, A, R, dtype=torch.float64)
        for comp_name in self.model_config.compartments:
            metapop_vals = [
                model.compartments[comp_name].current_val
                for model in subpop_models.values()
            ]
            total_pop = total_pop + torch.tensor(np.asarray(metapop_vals, dtype=float))
        return total_pop

    def apply_inter_subpop_updates(self) -> None:
        """
        Compute total_mixing_exposure from current subpop states and inject it
        into the rate_config of each travel-rate transition variable on each subpop.

        Called once per simulation day (before transitions are computed) by
        the MetapopModel simulation loop.

        Mirrors FluMetapopModel.apply_inter_subpop_updates().
        No-op when no transitions use force_of_infection_travel.
        """
        if not self._travel_transition_names:
            return

        compartment_tensors = self._build_compartment_tensors()
        schedule_tensors = self._build_schedule_tensors()
        param_tensors = self._build_param_tensors()

        total_mixing_exposure = compute_total_mixing_exposure(
            compartment_tensors,
            param_tensors,
            schedule_tensors,
            self.precomputed,
            self.travel_config,
        )

        # Inject per-subpopulation slices into each travel transition's rate_config.
        subpop_models = self._subpop_models_ordered
        for i, model in enumerate(subpop_models.values()):
            exposure_i = np.asarray(total_mixing_exposure[i, :, :])
            for tvar_name in self._travel_transition_names:
                tv = model.transition_variables[tvar_name]
                tv._rate_config["_total_mixing_exposure"] = exposure_i

    # -----------------------------------------------------------------------
    # Private tensor-building helpers
    # -----------------------------------------------------------------------

    def modify_subpop_params(self, subpop_name: str, updates_dict: dict) -> None:
        """
        Update parameters on a named subpopulation.

        For GenericSubpopParams the parameters live in params.params (a mutable
        dict inside a frozen dataclass), so we update that dict in-place.
        This is the generic equivalent of FluMetapopModel.modify_subpop_params.

        Parameters
        ----------
        subpop_name : str
            Must match the name of one of the subpop models.
        updates_dict : dict[str, Any]
            Mapping of parameter name → new value.
        """
        subpop = self.subpop_models[subpop_name]
        for param_name, value in updates_dict.items():
            subpop.params.params[param_name] = value

    def _build_compartment_tensors(self) -> dict:
        """
        Stack current compartment values from each subpop into (L, A, R) tensors.
        """
        result = {}
        for comp_name in self.model_config.compartments:
            vals = [
                model.compartments[comp_name].current_val
                for model in self._subpop_models_ordered.values()
            ]
            result[comp_name] = torch.tensor(np.asarray(vals, dtype=float))
        return result

    def _build_schedule_tensors(self) -> dict:
        """
        Stack current schedule values from each subpop.

        Contact matrix: (A, A) per subpop → stacked to (L, A, A).
        Mobility modifier: (A, R) per subpop → stacked to (L, A, R).
        """
        result = {}
        contact_name = self.travel_config["contact_matrix_schedule"]
        mobility_name = self.travel_config["mobility_schedule"]

        contact_vals = [
            model.schedules[contact_name].current_val
            for model in self._subpop_models_ordered.values()
        ]
        result[contact_name] = torch.tensor(np.asarray(contact_vals, dtype=float))

        mobility_vals = [
            model.schedules[mobility_name].current_val
            for model in self._subpop_models_ordered.values()
        ]
        result[mobility_name] = torch.tensor(np.asarray(mobility_vals, dtype=float))

        return result

    def _build_param_tensors(self) -> dict:
        """
        Build parameter tensors required by compute_total_mixing_exposure.

        travel_proportions : (L, L) — from mixing_params
        relative_suscept   : (L, A, R) — same across subpops, expanded from (A, R)
        Relative infectiousness params (e.g. IP_relative_inf) : scalar tensors
        """
        subpop_models = self._subpop_models_ordered
        first = list(subpop_models.values())[0]
        A = first.params.num_age_groups
        R = first.params.num_risk_groups
        L = len(subpop_models)

        result: dict = {"travel_proportions": self._travel_proportions_tensor}

        # relative_suscept: assumed identical across subpops; expand to (L, A, R)
        rel_suscept_param = self.travel_config["relative_susceptibility_param"]
        rs = np.broadcast_to(
            np.asarray(first.params.params[rel_suscept_param], dtype=float), (A, R)
        ).copy()
        rs_tensor = torch.tensor(rs).unsqueeze(0).expand(L, A, R)
        result[rel_suscept_param] = rs_tensor

        # Scalar relative-infectiousness params (e.g. IP_relative_inf, IA_relative_inf)
        for _comp, rel_inf_param in self.travel_config["infectious_compartments"].items():
            if rel_inf_param is not None and rel_inf_param not in result:
                val = float(first.params.params[rel_inf_param])
                result[rel_inf_param] = torch.tensor(val, dtype=torch.float64)

        return result
