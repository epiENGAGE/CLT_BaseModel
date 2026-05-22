"""
travel_functions.py — Generic (config-driven) travel computation functions.

Ports all functions from flu_core/flu_travel_functions.py, replacing
struct field-access with dict-lookup. All functions produce numerically
identical output to their flu counterparts when given equivalent inputs.

The key substitutions versus the flu version:
  flu_travel_functions                  generic_travel_functions
  ----------------------------          -----------------------------------
  state.ISR / state.IP / ...            compartment_tensors["ISR"] / ...
  params.IP_relative_inf                param_tensors["IP_relative_inf"]
  params.relative_suscept               param_tensors[config["relative_susceptibility_param"]]
  params.travel_proportions             param_tensors["travel_proportions"]
  state.flu_contact_matrix              schedule_tensors[config["contact_matrix_schedule"]]
  state.mobility_modifier               schedule_tensors[config["mobility_schedule"]]
  precomputed.total_pop_LAR_tensor etc. GenericPrecomputedTensors (same fields as Flu*)

Functions
---------
GenericPrecomputedTensors   — replaces FluPrecomputedTensors
compute_wtd_infectious_LA
compute_active_pop_LAR
compute_effective_pop_LA
compute_wtd_infectious_ratio_LLA
compute_local_to_local_exposure
compute_outside_visitors_exposure
compute_residents_traveling_exposure
compute_total_mixing_exposure
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# GenericPrecomputedTensors
# ---------------------------------------------------------------------------

class GenericPrecomputedTensors:
    """
    Stores precomputed quantities for repeated use in travel computations.
    Mirrors FluPrecomputedTensors without flu-specific types.

    Parameters
    ----------
    total_pop_LAR_tensor : torch.Tensor of shape (L, A, R)
        Total population per location × age × risk, summed across all compartments.
    travel_proportions_tensor : torch.Tensor of shape (L, L)
        Row l gives the fraction of location l's population that travels to each
        destination; diagonal entries are the fraction staying home.
    """

    def __init__(
        self,
        total_pop_LAR_tensor: torch.Tensor,
        travel_proportions_tensor: torch.Tensor,
    ) -> None:
        L, A, R = total_pop_LAR_tensor.shape
        self.L = L
        self.A = A
        self.R = R
        self.total_pop_LAR_tensor = total_pop_LAR_tensor
        self.total_pop_LA = torch.sum(total_pop_LAR_tensor, dim=2)

        # Zero the diagonal so nonlocal_travel_prop[l, k] = fraction of l going to k≠l
        self.nonlocal_travel_prop = travel_proportions_tensor.clone().fill_diagonal_(0.0)

        # sum_residents_nonlocal_travel_prop[l] = total fraction of l leaving home
        self.sum_residents_nonlocal_travel_prop = self.nonlocal_travel_prop.sum(dim=1)


# ---------------------------------------------------------------------------
# Core compute functions
# ---------------------------------------------------------------------------

def compute_wtd_infectious_LA(
    compartment_tensors: dict,
    param_tensors: dict,
    infectious_config: dict,
) -> torch.Tensor:
    """
    Weighted infectious count summed over risk groups.

    Parameters
    ----------
    infectious_config : dict[str, str | None]
        Keys are compartment names that contribute to infectiousness;
        values are either None (relative inf = 1) or a param name whose
        tensor value gives the relative infectiousness.

    Returns
    -------
    torch.Tensor of shape (L, A)
    """
    result: torch.Tensor | None = None
    for comp_name, rel_inf_param in infectious_config.items():
        comp = compartment_tensors[comp_name]          # (L, A, R)
        if rel_inf_param is not None:
            comp = param_tensors[rel_inf_param] * comp  # broadcast before summing over R
        wtd = torch.einsum("lar->la", comp)            # (L, A)
        result = wtd if result is None else result + wtd
    return result


def compute_active_pop_LAR(
    compartment_tensors: dict,
    immobile_compartments: list,
    precomputed: GenericPrecomputedTensors,
) -> torch.Tensor:
    """
    Active population: total minus individuals in immobile compartments.

    Immobile compartments (e.g. hospitalized) are subtracted from total
    population; the remainder are those who travel and infect normally.

    Returns
    -------
    torch.Tensor of shape (L, A, R)
    """
    active = precomputed.total_pop_LAR_tensor.clone()
    for comp_name in immobile_compartments:
        active = active - compartment_tensors[comp_name]
    return active


def compute_effective_pop_LA(
    compartment_tensors: dict,
    schedule_tensors: dict,
    precomputed: GenericPrecomputedTensors,
    travel_config: dict,
) -> torch.Tensor:
    """
    Effective population accounting for incoming visitors and outgoing travelers.

    Mirrors compute_effective_pop_LA in flu_travel_functions.py.

    Returns
    -------
    torch.Tensor of shape (L, A)
    """
    active_pop_LAR = compute_active_pop_LAR(
        compartment_tensors,
        travel_config["immobile_compartments"],
        precomputed,
    )

    # Visitors from other locations entering location l
    outside_visitors_LAR = torch.einsum(
        "kl,kar->lar",
        precomputed.nonlocal_travel_prop,
        active_pop_LAR,
    )

    # Residents of location l traveling out (same fraction regardless of destination)
    traveling_residents_LAR = (
        precomputed.sum_residents_nonlocal_travel_prop[:, None, None] * active_pop_LAR
    )

    # mobility_modifier is (L, A, R); take first risk dim (same pattern as flu)
    mobility_modifier = schedule_tensors[travel_config["mobility_schedule"]]
    mobility_modifier_LA = mobility_modifier[:, :, 0]   # (L, A)

    effective_pop_LA = precomputed.total_pop_LA + mobility_modifier_LA * torch.sum(
        outside_visitors_LAR - traveling_residents_LAR, dim=2
    )
    return effective_pop_LA


def compute_wtd_infectious_ratio_LLA(
    compartment_tensors: dict,
    param_tensors: dict,
    schedule_tensors: dict,
    precomputed: GenericPrecomputedTensors,
    travel_config: dict,
) -> torch.Tensor:
    """
    Ratio of weighted infectious individuals in each location to effective
    population in each other location.

    Mirrors compute_wtd_infectious_ratio_LLA in flu_travel_functions.py.

    Returns
    -------
    torch.Tensor of shape (L, L, A)
        Element [k, l, a] = wtd_infectious in k for age a / effective_pop in l for age a.
    """
    wtd_infectious_LA = compute_wtd_infectious_LA(
        compartment_tensors, param_tensors, travel_config["infectious_compartments"]
    )
    effective_pop_LA = compute_effective_pop_LA(
        compartment_tensors, schedule_tensors, precomputed, travel_config
    )

    # prop_wtd_infectious[k, l, a] = wtd_infectious[k, a] / effective_pop[l, a]
    prop_wtd_infectious = torch.einsum(
        "ka,la->kla",
        wtd_infectious_LA,
        1 / effective_pop_LA,
    )
    return prop_wtd_infectious


def compute_local_to_local_exposure(
    contact_matrix: torch.Tensor,
    mobility_modifier: torch.Tensor,
    sum_residents_nonlocal_travel_prop: torch.Tensor,
    wtd_infectious_ratio_LLA: torch.Tensor,
    location_ix: int,
) -> torch.Tensor:
    """
    Transmission within location_ix from residents who stayed home.

    Returns
    -------
    torch.Tensor of shape (A,)
    """
    # Fraction of residents who did not travel away
    proportion_staying_home = np.maximum(
        0,
        (1 - mobility_modifier[location_ix, :, 0] * sum_residents_nonlocal_travel_prop[location_ix])
    )

    result = torch.mul(
        proportion_staying_home,
        torch.matmul(
            contact_matrix[location_ix, :, :],
            torch.mul(
                proportion_staying_home,
                wtd_infectious_ratio_LLA[location_ix, location_ix, :]
            )
        )
    )
    return result


def compute_outside_visitors_exposure(
    contact_matrix: torch.Tensor,
    mobility_modifier: torch.Tensor,
    sum_residents_nonlocal_travel_prop: torch.Tensor,
    travel_proportions: torch.Tensor,
    wtd_infectious_ratio_LLA: torch.Tensor,
    local_ix: int,
    visitors_ix: int,
) -> torch.Tensor:
    """
    Transmission to local_ix residents caused by visitors from visitors_ix.

    Returns
    -------
    torch.Tensor of shape (A,)
    """
    proportion_staying_home = np.maximum(
        0,
        (1 - mobility_modifier[local_ix, :, 0] * sum_residents_nonlocal_travel_prop[local_ix])
    )

    result = torch.mul(
        proportion_staying_home * travel_proportions[visitors_ix, local_ix],
        torch.matmul(
            mobility_modifier[visitors_ix, :, 0] * contact_matrix[local_ix, :, :],
            wtd_infectious_ratio_LLA[visitors_ix, local_ix, :]
        )
    )
    return result


def compute_residents_traveling_exposure(
    contact_matrix: torch.Tensor,
    mobility_modifier: torch.Tensor,
    sum_residents_nonlocal_travel_prop: torch.Tensor,
    travel_proportions: torch.Tensor,
    wtd_infectious_ratio_LLA: torch.Tensor,
    local_ix: int,
    dest_ix: int,
) -> torch.Tensor:
    """
    Transmission acquired by local_ix residents who traveled to dest_ix.

    Returns
    -------
    torch.Tensor of shape (A,)
    """
    proportion_staying_home_at_dest = np.maximum(
        0,
        (1 - mobility_modifier[dest_ix, :, 0] * sum_residents_nonlocal_travel_prop[dest_ix])
    )

    # For each source location k: fraction of k's infectious who are in dest_ix
    infectious_proportion = mobility_modifier[:, :, 0] * travel_proportions[:, dest_ix].unsqueeze(1)
    # Residents of dest_ix who stayed home are exposed to the home-stayers
    infectious_proportion[dest_ix, :] = proportion_staying_home_at_dest

    result = torch.mul(
        mobility_modifier[local_ix, :, 0] * travel_proportions[local_ix, dest_ix],
        torch.matmul(
            contact_matrix[dest_ix, :, :],
            torch.einsum("ka,ka->a", wtd_infectious_ratio_LLA[:, dest_ix, :], infectious_proportion)
        )
    )
    return result


def compute_total_mixing_exposure(
    compartment_tensors: dict,
    param_tensors: dict,
    schedule_tensors: dict,
    precomputed: GenericPrecomputedTensors,
    travel_config: dict,
) -> torch.Tensor:
    """
    Total mixing exposure for each location × age × risk, normalized by
    relative susceptibility.

    Combines local-to-local, visitor, and traveling-resident contributions.
    Mirrors compute_total_mixing_exposure in flu_travel_functions.py.

    Parameters
    ----------
    travel_config : dict
        Must contain:
            infectious_compartments : dict[str, str|None]
            immobile_compartments   : list[str]
            relative_susceptibility_param : str
            contact_matrix_schedule : str
            mobility_schedule       : str

    Returns
    -------
    torch.Tensor of shape (L, A, R)
    """
    L, A, R = precomputed.L, precomputed.A, precomputed.R

    contact_matrix = schedule_tensors[travel_config["contact_matrix_schedule"]]  # (L, A, A)
    mobility_modifier = schedule_tensors[travel_config["mobility_schedule"]]      # (L, A, R)
    travel_proportions = param_tensors["travel_proportions"]                      # (L, L)

    rel_suscept_param = travel_config["relative_susceptibility_param"]
    _rs = param_tensors[rel_suscept_param]
    # Normalize to (L, A, R) so per-location slicing works for any input shape.
    # Scalar→(1,A,R), (A,R)→(1,A,R); (L,A,R) passes through. expand() is zero-copy.
    if _rs.dim() == 0:
        _rs = _rs.expand(1, A, R)
    elif _rs.dim() == 2:
        _rs = _rs.unsqueeze(0)                                                    # (1, A, R)

    sum_residents_nonlocal = precomputed.sum_residents_nonlocal_travel_prop
    wtd_infectious_ratio_LLA = compute_wtd_infectious_ratio_LLA(
        compartment_tensors, param_tensors, schedule_tensors, precomputed, travel_config
    )

    total_mixing_exposure = torch.tensor(np.zeros((L, A, R)))

    for l in np.arange(L):
        raw_exposure = torch.tensor(np.zeros(A))

        # Contribution 1: local residents who stayed home
        raw_exposure = raw_exposure + compute_local_to_local_exposure(
            contact_matrix, mobility_modifier, sum_residents_nonlocal,
            wtd_infectious_ratio_LLA, l
        )

        for k in np.arange(L):
            if k == l:
                continue  # visitor and travel-back terms only apply for k ≠ l

            # Contribution 2: visitors from k infecting l's residents
            raw_exposure = raw_exposure + compute_outside_visitors_exposure(
                contact_matrix, mobility_modifier, sum_residents_nonlocal,
                travel_proportions, wtd_infectious_ratio_LLA, l, k
            )

            # Contribution 3: l's residents who traveled to k and got infected
            raw_exposure = raw_exposure + compute_residents_traveling_exposure(
                contact_matrix, mobility_modifier, sum_residents_nonlocal,
                travel_proportions, wtd_infectious_ratio_LLA, l, k
            )

        # Slice per-location relative susceptibility; broadcast over risk groups.
        _l = min(int(l), _rs.shape[0] - 1)                                       # clamp for (1,A,R) case
        relative_suscept = _rs[_l, :, 0]                                         # (A,)
        normalized = relative_suscept * raw_exposure
        total_mixing_exposure[l, :, :] = normalized.view(A, 1).expand((A, R))

    return total_mixing_exposure
