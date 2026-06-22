"""
calibration.py — Generic accept-reject calibration for ConfigDrivenMetapopModel.

Mirrors flu_core/flu_accept_reject.py but accepts any MetapopModel subclass
and delegates calibration-target extraction to a caller-supplied function,
removing the flu-specific ISH_to_HR / ISH_to_HD hardcoding.

Functions
---------
compute_rsquared              — R² between two timeseries
generic_accept_reject         — accept-reject sampler
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

import clt_toolkit as clt


def compute_rsquared(
    reference_timeseries: list[np.ndarray],
    simulated_timeseries: list[np.ndarray],
) -> float:
    """
    Compute R² between reference and simulated timeseries.

    Mirrors compute_rsquared in flu_core/flu_accept_reject.py.

    Parameters
    ----------
    reference_timeseries : list or array-like, length D
    simulated_timeseries : list or array-like, length D

    Returns
    -------
    float
    """
    if len(reference_timeseries) != len(simulated_timeseries):
        raise ValueError(
            "Reference and simulated timeseries must have the same length."
        )
    ref = np.asarray(reference_timeseries, dtype=float)
    sim = np.asarray(simulated_timeseries, dtype=float)
    ybar = ref.mean(axis=0)
    ss_residual = np.sum(np.square(sim - ref))
    ss_total = np.sum(np.square(ref - ybar))
    if ss_total == 0:
        return 1.0 if ss_residual == 0 else -np.inf
    return 1.0 - ss_residual / ss_total


def generic_accept_reject(
    metapop_model,
    sampling_RNG: np.random.Generator,
    sampling_info: dict,
    target_timeseries: list[np.ndarray],
    calibration_target_fn: Callable,
    transition_variables_to_save: list[str],
    num_days: int = 50,
    target_accepted_reps: int = 100,
    max_reps: int = 1000,
    early_stop_percent: float = 0.5,
    target_rsquared: float = 0.75,
) -> None:
    """
    Accept-reject sampler for a ConfigDrivenMetapopModel (or any MetapopModel).

    Mirrors accept_reject_admits in flu_core/flu_accept_reject.py but is model-
    agnostic: calibration target extraction is delegated to ``calibration_target_fn``.

    Parameters
    ----------
    metapop_model : MetapopModel
        The metapopulation model to simulate and sample parameters for.
        Must be reset-able via ``metapop_model.reset_simulation()``.
    sampling_RNG : np.random.Generator
        Used for uniform parameter sampling.
    sampling_info : dict[str, dict[str, clt.UniformSamplingSpec]]
        See ``clt_toolkit.sample_uniform_metapop_params`` for format.
    target_timeseries : list[np.ndarray], length num_days
        Reference daily target values used to compute R².
    calibration_target_fn : Callable[[MetapopModel], list[np.ndarray]]
        Function that extracts the calibration target timeseries from the
        model after simulation.  Signature: ``fn(model) -> list[np.ndarray]``.
        For the flu model this would be ``lambda m: clt.aggregate_daily_tvar_history(m, ["ISH_to_HR", "ISH_to_HD"])``.
    transition_variables_to_save : list[str]
        Transition variable names to save during simulation (needed by
        ``calibration_target_fn``).
    num_days : int
        Number of days to simulate for accepted parameter sets.
    target_accepted_reps : int
        Stop when this many parameter sets are accepted.
    max_reps : int
        Hard upper limit on total sampling attempts.
    early_stop_percent : float
        Fraction of ``num_days`` simulated before the first R² check.
    target_rsquared : float
        Minimum R² for acceptance.
    """
    if target_accepted_reps > max_reps:
        max_reps = 10 * target_accepted_reps

    num_days_early_stop = int(num_days * early_stop_percent)
    reps_counter = 0
    accepted_reps_counter = 0

    while reps_counter < max_reps and accepted_reps_counter < target_accepted_reps:
        reps_counter += 1

        metapop_model.reset_simulation()

        param_samples = clt.sample_uniform_metapop_params(
            metapop_model, sampling_RNG, sampling_info
        )

        for subpop_name, updates_dict in param_samples.items():
            metapop_model.modify_subpop_params(subpop_name, updates_dict)
            metapop_model.modify_simulation_settings({
                "transition_variables_to_save": transition_variables_to_save,
                "save_daily_history": False,
            })

        # Early-stop check
        metapop_model.simulate_until_day(num_days_early_stop)
        sim_target_early = calibration_target_fn(metapop_model)
        early_rsq = compute_rsquared(
            target_timeseries[:num_days_early_stop], sim_target_early
        )
        if early_rsq < target_rsquared:
            continue

        # Full simulation
        metapop_model.simulate_until_day(num_days)
        sim_target_full = calibration_target_fn(metapop_model)
        full_rsq = compute_rsquared(target_timeseries, sim_target_full)
        if full_rsq < target_rsquared:
            continue

        accepted_reps_counter += 1

        for subpop_name, subpop in metapop_model.subpop_models.items():
            with open(
                f"subpop_{subpop_name}_rep_{accepted_reps_counter}_accepted_sample_params.json",
                "w",
            ) as f:
                json.dump(clt.serialize_dataclass(param_samples[subpop_name]), f, indent=4)
            with open(
                f"subpop_{subpop_name}_rep_{accepted_reps_counter}_accepted_state.json",
                "w",
            ) as f:
                json.dump(clt.serialize_dataclass(subpop.state), f, indent=4)
