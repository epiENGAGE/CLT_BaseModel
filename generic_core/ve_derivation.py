"""
ve_derivation.py — back-derive the "peak" (zero-waning) vaccine efficacy that a
season-average efficacy input implies.

Generic port of `compute_vax_induced_risk_reduce_initial` and its helpers in
flu_core/flu_components.py. The arithmetic is identical; two things are
generalised:

1.  Where the inputs come from — a `GenericSubpopParams.params` dict plus a
    named schedule and a list of season-average parameter names, rather than
    the fixed `FluSubpopParams` inf/hosp/death triple.

2.  WHAT IS STORED. flu_core stores the finished product
    `VE_0 = VE_season * inflation` on params. generic_core stores only the
    `inflation` factor, and the rate templates form the product at evaluation
    time (`rate_templates._vax_immunity_factor_np` / `_torch`).

    The reason is that `VE_0` is *linear* in `VE_season` — the inflation factor
    depends only on the waning rate and the dose-timing profile. Storing the
    product bakes in whatever `VE_season` happened to be at construction time,
    so any later override of the season-average parameter (a `ScenarioRunner`
    override, a fitting draw applied after `reset_simulation`, a torch
    autodiff leaf) silently has no effect on the trajectory. Storing the
    factor keeps `VE_season` live on every path, including torch gradients.

    See generic_core/limitations.md, "The VE inflation factor is fixed at
    construction/reset time", for what remains construction-time-bound.

Entry point
-----------
compute_ve_inflation_factors(params, schedules, start_real_date,
                             derivation_config) -> dict[str, np.ndarray]

The returned dict maps `ve_inflation_param_name(<season_average_param>)` to an
(A, R) array, which `ConfigDrivenSubpopModel` writes back into `params.params`.
Rate templates read it via `rate_config["vax_reduce_param"]`, which names the
season-average parameter itself.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Optional

import numpy as np


class VEDerivationError(ValueError):
    """Raised for an invalid ve_derivation config or parameter value."""



def ve_inflation_param_name(season_average_param: str) -> str:
    """
    Name of the derived param holding the VE inflation factor for a given
    season-average efficacy parameter.

    The leading underscore marks it as derived rather than user-supplied,
    matching `metric_templates.injection_val_param_name`.
    """
    return f"_{season_average_param}_ve_inflation"


def compute_ve_inflation_factors(
    params,
    schedules: dict,
    start_real_date: datetime.date,
    derivation_config: dict,
) -> dict[str, np.ndarray]:
    """
    Computes the VE inflation factor for each configured season-average
    efficacy parameter, from the waning rate and the dose schedule.

    The inflation factor is `VE_0 / VE_season`, where `VE_0` is the
    "zero-waning" (peak, just-after-protection-delay) efficacy. For each
    age-risk group, given the season's vaccination timing (`p_prot`, the
    proportion of effective doses -- i.e. vaccination date plus protection
    delay -- given on each day of the season) and waning rate w_V, VE_0 is
    the peak efficacy for which the dose-timing-weighted average realized
    efficacy over the season equals the input (season-average) value:

        VE_0 = VE_season * w_V * T /
               ((1 - exp(-w_V)) * sum_{tau=t0}^{T-1} [
                   (sum_{u=t0}^{tau} p_prot(u) * exp(-w_V * (tau - u))) /
                   (sum_{u=t0}^{tau} p_prot(u))
               ])

    VE_0 is *linear* in VE_season, so everything to the right of
    `VE_season` above is the inflation factor -- it depends only on w_V and
    the dose timing profile, is computed once per age-risk group, and is
    shared by every configured field. Returning the factor rather than the
    product is what keeps `VE_season` live for overrides and gradients; see
    the module docstring.

    Season window
    -------------
    The season window is the period between two consecutive occurrences of
    the reset date (the occurrence on or before `start_real_date`, through
    one year later), or if that parameter is not set, the 12 months
    starting from the first date covered by the dose schedule. This window
    is further intersected with the actual date range covered by the
    schedule.

    Within that window, t0 and T are by default the first day with a
    nonzero dose and the number of days through the last day with a
    nonzero dose. Because the average above is an *unweighted* average
    over the T days, a long low-dose tail pulls the average down and so
    inflates VE_0. Setting `VE_season_dose_window_quantile` to a value q
    in [0, 0.5) trims the window to the days spanning the central
    (1 - 2q) of the season's cumulative doses.

    Capping
    -------
    Not done here. `VE_0 = VE_season * inflation` is a probability and must
    not exceed 1 (a value above 1 would make the applied factor
    `1 - MV * VE_0` negative at high vaccination coverage), but since
    `VE_season` is free to change after this function runs, the cap can only
    be applied where the product is formed -- see
    `rate_templates._vax_immunity_factor_np`. This function still *warns*
    about capping, using the season-average values as they stand at
    construction time; see `warn_if_ve_initial_capped`.

    Edge cases
    ----------
    - Waning of 0 for an age-risk group: inflation factor is 1.
    - No doses in the (intersected) season window for an age-risk group:
      inflation factor is 1 (MV is always 0 there anyway).
    - `adjust_VE_for_seasonal_waning` False: the derivation is skipped
      entirely and every factor is 1, i.e. the configured
      `vax_induced_*_risk_reduce` values are used directly as peak
      efficacies.

    Args:
        params (GenericSubpopParams):
            holds `params` (the name -> value dict), `num_age_groups`,
            and `num_risk_groups`.
        schedules (dict):
            name -> Schedule, holding the dose schedule named by
            `daily_vaccines_schedule` (already shifted by the protection
            delay by VaccineScheduleGeneric.preprocess, date-indexed, one
            A x R array per day).
        start_real_date (datetime.date):
            real-world date corresponding to the start of the simulation.
        derivation_config (dict):
            the validated "ve_derivation" config block -- see
            config_parser.validate_ve_derivation_config.

    Returns:
        dict mapping `ve_inflation_param_name(season_param)` to an
        np.ndarray of shape (A, R), for each configured season param.
    """

    target_shape = (params.num_age_groups, params.num_risk_groups)
    season_params = derivation_config["fields"]

    # `inflation_factor[a, r]` is VE_0 / VE_season for that age-risk group --
    #   it depends only on the waning rate and the dose timing profile, so it
    #   is computed once and shared by every configured season param
    inflation_factor = np.ones(target_shape)

    if not derivation_config.get("adjust_VE_for_seasonal_waning", True):
        return {ve_inflation_param_name(name): inflation_factor.copy()
                for name in season_params}

    wane_param = derivation_config["vax_induced_immune_wane_param"]
    w_arr = np.broadcast_to(
        np.asarray(params.params[wane_param], dtype=float), target_shape)

    doses_stack = _season_window_doses(
        params, schedules, start_real_date, target_shape, derivation_config)

    trim_quantile = derivation_config.get("VE_season_dose_window_quantile")

    if trim_quantile is not None and not 0 <= trim_quantile < 0.5:
        raise VEDerivationError(
            f"`VE_season_dose_window_quantile` must be None or in [0, 0.5) -- "
            f"got {trim_quantile}. It trims that fraction of cumulative doses "
            "off each end of the season window, so 0.5 or more would leave "
            "nothing behind.")

    for a in range(target_shape[0]):
        for r in range(target_shape[1]):

            w_ar = w_arr[a, r]

            if w_ar == 0 or doses_stack.shape[0] == 0:
                continue

            sub = _trimmed_dose_profile(doses_stack[:, a, r], trim_quantile)

            if sub is None:
                continue

            T = sub.size
            p_prot = sub / sub.sum()
            cumsum_p_prot = np.cumsum(p_prot)

            # numer[n] = sum_{u=0}^{n} p_prot(u) * exp(-w * (n - u))
            #          = p_prot(n) + exp(-w) * numer[n - 1]
            decay = np.exp(-w_ar)
            numer = np.empty(T)
            acc = 0.0
            for n in range(T):
                acc = p_prot[n] + decay * acc
                numer[n] = acc

            S = np.sum(numer / cumsum_p_prot)

            inflation_factor[a, r] = w_ar * T / ((1 - decay) * S)

    return {ve_inflation_param_name(name): inflation_factor.copy()
            for name in season_params}


def warn_if_ve_initial_capped(params, derivation_config: dict) -> None:
    """
    Warns for each configured season-average efficacy whose implied peak
    efficacy `VE_season * inflation` exceeds 1.0 and will therefore be
    capped when applied.

    A capped entry means the requested season-average efficacy is not
    achievable given the waning rate and dose timing -- even perfect (100%)
    protection at the moment of vaccination would average out to less than
    the requested value over the season. The simulation stays well-defined
    (`rate_templates._vax_immunity_factor_np` caps the product, so
    `1 - MV * VE_0` remains nonnegative), but realized efficacy will fall
    short of the input value, so this is worth surfacing rather than
    silently clipping.

    This runs where the inflation factors are (re)computed -- construction
    and `reset_simulation` -- so it reflects the season-average values in
    force at that moment. A later override that pushes the product over 1
    is still capped correctly at evaluation time, but is not warned about;
    warning per-timestep would be far noisier than it is worth.

    Args:
        params (GenericSubpopParams):
            holds `params`, already containing the freshly computed
            inflation factors.
        derivation_config (dict):
            the validated "ve_derivation" config block.
    """

    for season_param in derivation_config["fields"]:

        inflation = params.params.get(ve_inflation_param_name(season_param))

        if inflation is None:
            continue

        ve_initial_arr = np.asarray(
            params.params[season_param], dtype=float) * inflation

        over_idxs = np.argwhere(np.atleast_2d(ve_initial_arr) > 1.0)

        if over_idxs.size == 0:
            continue

        max_val = float(np.max(ve_initial_arr))
        groups_str = ", ".join(f"(age {a}, risk {r})" for a, r in over_idxs)
        warnings.warn(
            f"Derived peak efficacy from `{season_param}` exceeds 1.0 "
            f"(max {max_val:.4f}) for age-risk group(s) {groups_str} and will "
            "be capped at 1.0. This means "
            f"`{season_param}` is not achievable given the vaccine waning rate "
            "and the dose schedule timing -- even 100% protection at the "
            "moment of vaccination would average to less than the requested "
            "season-average value. Realized efficacy will be lower than "
            f"requested for these groups. Consider lowering `{season_param}`, "
            "lowering the waning rate, or setting "
            "`VE_season_dose_window_quantile` to trim sparse dose tails from "
            "the season window."
        )



def _season_window_doses(params,
                         schedules: dict,
                         start_real_date: datetime.date,
                         target_shape: tuple,
                         derivation_config: dict) -> np.ndarray:
    """
    Returns the doses falling inside the vaccination season window, as an
    np.ndarray of shape (T, A, R) -- see
    `compute_ve_inflation_factors` for how the window is
    defined. Returns an array with T == 0 if the window contains no
    schedule days.
    """

    schedule_name = derivation_config["daily_vaccines_schedule"]
    vaccines_df = schedules[schedule_name].timeseries_df

    schedule_min_date = vaccines_df.index.min()
    schedule_max_date = vaccines_df.index.max()

    reset_param = derivation_config.get("vax_immunity_reset_date_mm_dd_param")
    reset_date_str = params.params.get(reset_param) if reset_param else None

    if reset_date_str is not None:
        month, day = (int(x) for x in reset_date_str.split("_"))
        window_start = datetime.date(start_real_date.year, month, day)
        if window_start >= start_real_date:
            window_start = datetime.date(start_real_date.year - 1, month, day)
        window_end = datetime.date(window_start.year + 1, month, day)
    else:
        window_start = schedule_min_date
        window_end = schedule_min_date + datetime.timedelta(days=365)

    window_start = max(window_start, schedule_min_date)
    window_end = min(window_end, schedule_max_date + datetime.timedelta(days=1))

    if window_start < window_end:
        mask = (vaccines_df.index >= window_start) & (vaccines_df.index < window_end)
        window_doses_df = vaccines_df.loc[mask]
    else:
        window_doses_df = vaccines_df.iloc[0:0]

    if window_doses_df.empty:
        return np.zeros((0,) + target_shape)

    doses_stack = np.stack(window_doses_df["daily_vaccines"].values, axis=0)

    if doses_stack.shape[1:] != target_shape:
        # Time series has a different age-risk resolution than the
        # risk-reduce parameters -- aggregate (sum) across all
        # dimensions and broadcast the resulting total evenly across
        # every age-risk group.
        doses_stack = doses_stack.reshape(doses_stack.shape[0], -1).sum(axis=1, keepdims=True)
        doses_stack = np.broadcast_to(doses_stack, (doses_stack.shape[0],) + target_shape)

    return doses_stack


def _trimmed_dose_profile(cell_doses: np.ndarray,
                          trim_quantile: Optional[float]) -> Optional[np.ndarray]:
    """
    Returns the slice of one age-risk group's daily doses that defines
    the season window for the VE_0 calculation, or None if the group has
    no doses at all.

    The base window runs from the first to the last day with a nonzero
    dose. If `trim_quantile` is a value q in [0, 0.5), the window is
    further narrowed to the days spanning the central (1 - 2q) of the
    group's cumulative doses -- i.e. days before the qth and after the
    (1 - q)th quantile of the cumulative dose distribution are dropped.
    This removes sparse dose tails that would otherwise stretch the
    (unweighted) season average over months with almost no vaccination.

    Args:
        cell_doses (np.ndarray of shape (T,)):
            daily doses for one age-risk group over the season window.
        trim_quantile (Optional[float]):
            q as described above -- None or 0 leaves the window untrimmed.

    Returns:
        np.ndarray of shape (T',) with a positive sum, or None.
    """

    nonzero_idxs = np.flatnonzero(cell_doses > 0)

    if nonzero_idxs.size == 0:
        return None

    sub = cell_doses[nonzero_idxs[0]:nonzero_idxs[-1] + 1]

    if not trim_quantile:
        return sub

    cumulative_prop = np.cumsum(sub) / sub.sum()

    # `lo` is the first day by which the trimmed-off leading mass has
    #   accumulated, `hi` the first day reaching the upper cutoff --
    #   both are kept, so the window spans the central mass inclusively
    lo = int(np.searchsorted(cumulative_prop, trim_quantile, side="left"))
    hi = int(np.searchsorted(cumulative_prop, 1.0 - trim_quantile, side="left"))

    return sub[lo:hi + 1]
