"""Plain-Python MA_vax SEIR+vaccination model.

Single source of truth for the model logic used by both the marimo notebook
(`ma_vax.py`) and the Bayesian fitting prototype (`fit_bayesian.py`).

The functions here are lifted verbatim from the notebook cells so behaviour is
identical, plus a lightweight `simulate_new_H` helper for the MCMC inner loop
that applies parameter overrides and returns just the daily new-hospitalization
array (skipping DataFrame construction).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Default location of the input CSVs / config (relative to the repo root).
DATA_FOLDER = "generic_core/examples/massachusetts_vax"

# Age groups, in the order used by every age-indexed array (population,
# E0_counts, I_to_H_prop, vax_arr columns, ...).
AGE_GROUP_LABELS = ["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"]

COMPARTMENTS = ["S", "E", "I", "R", "SV", "EV", "IV", "H", "D"]
TRANSITIONS = [
    "S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R",
    "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R",
]

# How the daily vaccination proportions in `vax_arr` are interpreted:
#   "susceptible" — proportion of the *remaining susceptible* pool S (the
#                   count vaccinated shrinks as S is depleted; cumulative
#                   coverage saturates and is nonlinear in a schedule scale).
#   "population"  — proportion of the *total (constant) population* (the count
#                   is a fixed head-count independent of S, so cumulative
#                   coverage is linear in a schedule scale; matches the usual
#                   "% of the population vaccinated per day" reading of the
#                   source data). Still capped at the available S each step.
# Flip this one line (or override params["vax_pool"] per run) to switch.
VAX_POOL_DEFAULT = "susceptible"


def _vax_flow(vax_prop, S, SV, population, vax_pool: str, dt: float = 1.0):
    """Scheduled S -> SV count for one (sub-)step, capped at the available S.

    `vax_pool="population"` applies `vax_prop` to the constant total
    `population` (== sum over all compartments, which this closed model
    conserves). `vax_pool="susceptible"` applies it to `S + SV` instead — the
    not-yet-infected pool — so that vaccinating someone doesn't itself reduce
    the base future vaccination rates are applied to; only infection (moving
    people out of S/SV into E/EV) shrinks that base. Both clamp to S: you can
    never vaccinate more unvaccinated-susceptibles than exist.
    """
    base = population if vax_pool == "population" else S + SV
    return np.minimum(np.rint(vax_prop * dt * base), S)


def default_params() -> dict:
    """Return a fresh copy of the baseline parameter dict."""
    return {
        # ── Simulation settings ──────────────────────────────────────────────
        "start_date": "2025-09-01",
        "num_days":   250,

        # ── Transmission ─────────────────────────────────────────────────────
        "beta_baseline":             0.039,
        "humidity_impact":           0.25,
        "relative_suscept":          1.0,
        "I_relative_infectiousness": 1.0,
        "IV_relative_infectiousness": 1.0, #0.5,

        # ── Progression rates (per day) ──────────────────────────────────────
        "E_to_I_rate":   0.5,    # ~2-day latent period
        "EV_to_IV_rate": 0.5,
        "I_out_rate":    0.333,  # ~3-day infectious period
        "H_out_rate":    0.17,   # ~6-day hospital stay

        # ── Age-stratified proportions (7 groups: 0, 1-4, 5-12, 13-17, 18-49, 50-64, 65+) ──
        "I_to_H_prop": np.array([
            0.006971411624, 0.006971411624, 0.002741943394, 0.002741943394,
            0.005613008764, 0.01060470654,  0.09090913146,
        ]),
        "IV_to_H_prop": np.array([   # lower for vaccinated
            0.006357927401, 0.006357927401, 0.002500652376, 0.002500652376,
            0.005113450984, 0.009660887656, 0.06272730071,
        ]),
        "H_to_D_prop": np.array([
            0.01741109215, 0.01741109215, 0.0117181651, 0.0117181651,
            0.02633277317, 0.06301864177,  0.07988898095,
        ]),

        # Fraction of susceptibility retained after vaccination (1=no protection)
        "vax_susceptibility": np.array([0.57, 0.57, 0.57, 0.57, 0.79, 0.79, 1.0]),

        # Days from vaccination to immunity (shifts the vax schedule forward)
        "vax_transfer_delay_days": 14,

        # How vax_arr proportions are applied: "population" (fraction of the
        # constant total population) or "susceptible" (fraction of remaining S).
        # See VAX_POOL_DEFAULT / _vax_flow.
        "vax_pool": VAX_POOL_DEFAULT,

        # ── Initial exposed seeds ────────────────────────────────────────────
        "E0_counts": np.array([2.0, 8.0, 17.0, 12.0, 85.0, 41.0, 35.0]),
    }


def load_inputs(data_folder: str, params: dict) -> dict:
    """
    Load all CSVs, build per-day schedule arrays, and return a single dict
    that contains everything run_simulation() needs.
    """
    folder = Path(data_folder)
    num_days = params["num_days"]
    delay    = params["vax_transfer_delay_days"]

    # Population
    df_pop     = pd.read_csv(folder / "data" / "massachusetts_population.csv")
    population = df_pop["population"].to_numpy(dtype=float)

    # Contact matrices from JSON config
    config_path = folder / "model_config_MA.json"
    with open(config_path) as f:
        cfg = json.load(f)
    total_C  = np.array(cfg["params"]["total_contact_matrix"])
    school_C = np.array(cfg["params"]["school_contact_matrix"])
    work_C   = np.array(cfg["params"]["work_contact_matrix"])

    # Date range
    dates = pd.date_range(start=params["start_date"], periods=num_days, freq="D")

    # Humidity timeseries
    df_hum = pd.read_csv(folder / "data" / "schedules" / "ma_absolute_humidity.csv")
    df_hum["date"] = pd.to_datetime(df_hum["date"])
    df_hum = df_hum.set_index("date")

    # School/work calendar
    df_cal = pd.read_csv(folder / "data" / "schedules" / "MA_school_work_calendar.csv")
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_cal = df_cal.set_index("date")

    # Vaccination schedule (proportions per age group)
    df_vax = pd.read_csv(folder / "data" / "vaccination" / "MA_flu_daily_vaccinations_proportions_array.csv")
    df_vax["date"] = pd.to_datetime(df_vax["date"])
    df_vax["daily_vaccines"] = (
        df_vax["daily_vaccines"].apply(json.loads).apply(lambda x: np.array(x).flatten())
    )
    df_vax = df_vax.set_index("date")

    # Observed hospitalizations (new daily admissions)
    df_obs = pd.read_csv(folder / "data" / "hospitalizations_ts" / "MA_flu_daily_hospitalizations_total.csv")
    df_obs["date"] = pd.to_datetime(df_obs["Date"])
    df_obs = df_obs.set_index("date")[["total"]]

    # Pre-compute per-day arrays
    A = len(population)
    humidity_arr  = np.zeros(num_days)
    is_school_arr = np.zeros(num_days)
    is_work_arr   = np.zeros(num_days)
    vax_arr       = np.zeros((num_days, A))

    for t, date in enumerate(dates):
        if date in df_hum.index:
            humidity_arr[t] = df_hum.loc[date, "absolute_humidity"]
        if date in df_cal.index:
            is_school_arr[t] = df_cal.loc[date, "is_school_day"]
            is_work_arr[t]   = df_cal.loc[date, "is_work_day"]
        raw_date = date - pd.Timedelta(days=delay)
        if raw_date in df_vax.index:
            vax_arr[t] = df_vax.loc[raw_date, "daily_vaccines"]

    return {
        "params":         params,
        "population":     population,
        "dates":          dates,
        "total_C":        total_C,
        "school_C":       school_C,
        "work_C":         work_C,
        "humidity_arr":   humidity_arr,
        "is_school_arr":  is_school_arr,
        "is_work_arr":    is_work_arr,
        "vax_arr":        vax_arr,
        "df_obs":         df_obs,
    }


def _simulate_core(inputs: dict):
    """
    Run the SEIR+vaccination model.

    Returns per-day arrays: (dates, new_H, new_H_by_age, records) where
    `new_H_by_age` has shape (num_days, num_age_groups) (new_H is just its
    row-sum) and `records` is the list of per-day summary dicts (used to
    build the full DataFrame in `run_simulation`). Splitting the core out
    lets the fast `simulate_new_H` path skip DataFrame construction.

    Compartments (all shape: num_age_groups):
        S, E, I, R, SV, EV, IV, H, D
    """
    p          = inputs["params"]
    population = inputs["population"]
    dates      = inputs["dates"]
    num_days   = p["num_days"]

    # Unpack params
    beta            = p["beta_baseline"]
    hum_impact      = p["humidity_impact"]
    rel_suscept     = p["relative_suscept"]
    I_rel_inf       = p["I_relative_infectiousness"]
    IV_rel_inf      = p["IV_relative_infectiousness"]
    E_to_I          = p["E_to_I_rate"]
    EV_to_IV        = p["EV_to_IV_rate"]
    I_out           = p["I_out_rate"]
    H_out           = p["H_out_rate"]
    I_to_H          = p["I_to_H_prop"]
    IV_to_H         = p["IV_to_H_prop"]
    H_to_D          = p["H_to_D_prop"]
    vax_suscept     = p["vax_susceptibility"]
    vax_pool        = p.get("vax_pool", VAX_POOL_DEFAULT)

    # Contact matrices
    total_C  = inputs["total_C"]
    school_C = inputs["school_C"]
    work_C   = inputs["work_C"]

    # Schedule arrays
    humidity_arr  = inputs["humidity_arr"]
    is_school_arr = inputs["is_school_arr"]
    is_work_arr   = inputs["is_work_arr"]
    vax_arr       = inputs["vax_arr"]

    # Optional time-varying transmission multiplier m(t) (defaults to 1.0).
    beta_mult = inputs.get("beta_multiplier_arr")
    if beta_mult is None:
        beta_mult = np.ones(num_days)

    # Initial conditions
    E0 = p["E0_counts"]
    S  = population - E0;  E  = E0.copy()
    I  = np.zeros_like(population);  R  = np.zeros_like(population)
    SV = np.zeros_like(population);  EV = np.zeros_like(population)
    IV = np.zeros_like(population);  H  = np.zeros_like(population)
    D  = np.zeros_like(population)

    new_H_arr = np.zeros(num_days)
    new_H_by_age_arr = np.zeros((num_days, len(population)))
    records = []

    for t in range(num_days):

        # ── Effective contact matrix ─────────────────────────────────────────
        # C = total - (1-school_open)*school_C - (1-work_open)*work_C
        C = (
            total_C
            - (1.0 - is_school_arr[t]) * school_C
            - (1.0 - is_work_arr[t])   * work_C
        )

        # ── Humidity modifier: higher humidity → lower beta ──────────────────
        beta_adj = beta * beta_mult[t] * (1.0 + hum_impact * np.exp(-180.0 * humidity_arr[t]))

        # ── Force of infection ───────────────────────────────────────────────
        wtd_inf_prop = (I * I_rel_inf + IV * IV_rel_inf) / population
        foi = beta_adj * (C @ wtd_inf_prop)   # shape (A,)

        # ── Flows: unvaccinated pathway ──────────────────────────────────────
        S_to_E = foi * rel_suscept * S
        E_to_I_flow = E_to_I * E
        I_to_H_flow = I_out * I_to_H * I
        I_to_R_flow = I_out * (1.0 - I_to_H) * I

        # ── Vaccination: exact count S → SV (delay already in vax_arr) ───────
        S_to_SV = _vax_flow(vax_arr[t], S, SV, population, vax_pool)

        # ── Flows: vaccinated pathway ────────────────────────────────────────
        SV_to_EV  = foi * vax_suscept * SV
        EV_to_IV_flow = EV_to_IV * EV
        IV_to_H_flow  = I_out * IV_to_H * IV
        IV_to_R_flow  = I_out * (1.0 - IV_to_H) * IV

        # ── Flows: hospital ──────────────────────────────────────────────────
        H_to_D_flow = H_out * H_to_D * H
        H_to_R_flow = H_out * (1.0 - H_to_D) * H

        # ── Euler update ─────────────────────────────────────────────────────
        S  = S  - S_to_E   - S_to_SV
        E  = E  + S_to_E   - E_to_I_flow
        I  = I  + E_to_I_flow - I_to_H_flow - I_to_R_flow
        R  = R  + I_to_R_flow + IV_to_R_flow + H_to_R_flow
        SV = SV + S_to_SV  - SV_to_EV
        EV = EV + SV_to_EV - EV_to_IV_flow
        IV = IV + EV_to_IV_flow - IV_to_H_flow - IV_to_R_flow
        H  = H  + I_to_H_flow + IV_to_H_flow - H_to_D_flow - H_to_R_flow
        D  = D  + H_to_D_flow

        new_H_by_age = I_to_H_flow + IV_to_H_flow
        new_H_by_age_arr[t] = new_H_by_age
        new_H_arr[t] = new_H_by_age.sum()

        records.append({
            "date":           dates[t],
            "S":              S.sum(),   "E":  E.sum(),  "I":  I.sum(),
            "R":              R.sum(),   "SV": SV.sum(), "EV": EV.sum(),
            "IV":             IV.sum(),  "H":  H.sum(),  "D":  D.sum(),
            "new_H":          new_H_arr[t],
            "new_D":          H_to_D_flow.sum(),
            "new_infections": S_to_E.sum(),
        })

    return dates, new_H_arr, new_H_by_age_arr, records


def run_simulation(inputs: dict) -> pd.DataFrame:
    """Run the model and return a DataFrame of daily outputs (indexed by date)."""
    _, _, _, records = _simulate_core(inputs)
    return pd.DataFrame(records).set_index("date")


# ── Fitting helper ───────────────────────────────────────────────────────────

def apply_overrides(base_params: dict, overrides: dict) -> dict:
    """
    Return a copy of `base_params` with fitting overrides applied.

    Recognised keys:
        beta_baseline, humidity_impact  -> set directly
        E0_scale                        -> multiplies E0_counts
        ihr_scale                       -> multiplies I_to_H_prop and IV_to_H_prop;
                                            scalar (applied to every age group) or
                                            a length-len(AGE_GROUP_LABELS) per-age
                                            array

    Any other key that already exists in `base_params` is set directly.
    """
    p = dict(base_params)
    # array-valued entries must be copied so we never mutate the caller's arrays
    for key in ("E0_counts", "I_to_H_prop", "IV_to_H_prop"):
        p[key] = np.array(p[key], dtype=float)

    for key, val in overrides.items():
        if key == "E0_scale":
            p["E0_counts"] = p["E0_counts"] * float(val)
        elif key == "ihr_scale":
            scale = np.asarray(val, dtype=float)
            p["I_to_H_prop"] = p["I_to_H_prop"] * scale
            p["IV_to_H_prop"] = p["IV_to_H_prop"] * scale
        else:
            p[key] = val
    return p


def build_beta_multiplier(increments, knot_days, num_days: int) -> np.ndarray:
    """
    Build a time-varying transmission multiplier m(t) from monthly knots.

    `g` (= log m) is anchored at 0 on the first knot and follows a random walk:
    g[k] = g[k-1] + increments[k-1]. m(t) is the exp of the piecewise-linear
    interpolation of g over `knot_days`; np.interp clamps to the endpoint values
    outside the knot range (so m is flat before the first / after the last knot).

    `increments` has length len(knot_days) - 1.
    """
    increments = np.asarray(increments, dtype=float)
    g = np.concatenate([[0.0], np.cumsum(increments)])
    g_interp = np.interp(np.arange(num_days), np.asarray(knot_days), g)
    return np.exp(g_interp)


def simulate_new_H(overrides: dict, inputs: dict,
                   beta_multiplier_arr: np.ndarray | None = None,
                   by_age: bool = False) -> np.ndarray:
    """
    Apply parameter overrides and return the daily new-hospitalization array:
    shape (num_days,), or (num_days, num_age_groups) if `by_age`.

    `inputs` is the dict from `load_inputs`; its "params" entry is used as the
    baseline over which `overrides` are applied. Schedule arrays and contact
    matrices are reused as-is (they do not depend on the fitted params).
    An optional `beta_multiplier_arr` (length num_days) scales transmission over
    time; defaults to 1.0 everywhere.
    """
    p = apply_overrides(inputs["params"], overrides)
    sim_inputs = dict(inputs)
    sim_inputs["params"] = p
    if beta_multiplier_arr is not None:
        sim_inputs["beta_multiplier_arr"] = beta_multiplier_arr
    _, new_H_arr, new_H_by_age_arr, _ = _simulate_core(sim_inputs)
    return new_H_by_age_arr if by_age else new_H_arr


# ── Counterfactual scenarios ─────────────────────────────────────────────────

def apply_vax_scenario(inputs: dict, vax_multiplier) -> dict:
    """
    Return a copy of `inputs` with `vax_arr` scaled per age group by
    `vax_multiplier` (length-A array, ordered like `AGE_GROUP_LABELS`; 0
    zeroes out a group's vaccination schedule, >1 boosts it).
    """
    vax_multiplier = np.asarray(vax_multiplier, dtype=float)
    new_inputs = dict(inputs)
    new_inputs["vax_arr"] = inputs["vax_arr"] * vax_multiplier[None, :]
    return new_inputs


def apply_scenario(inputs: dict, scenario: dict) -> dict:
    """
    Apply a counterfactual scenario dict to `inputs`, returning a new inputs
    dict (`inputs` itself is not mutated). Recognised keys:

        vax_multiplier            -> per-age-group multiplier on the daily
                                      vaccination schedule (see
                                      `apply_vax_scenario`)
        vax_susceptibility_scale  -> multiplies params["vax_susceptibility"]
                                      (vaccine effectiveness against infection)
        IV_to_H_prop_scale        -> multiplies params["IV_to_H_prop"]
                                      (vaccine effectiveness against
                                      hospitalization given infection)

    Any other key is passed through to `apply_overrides` on params, so a
    scenario can also set a full replacement array (e.g. `vax_susceptibility`)
    or override a structural param (e.g. `beta_baseline`) directly.
    """
    scenario = dict(scenario)
    new_inputs = dict(inputs)

    if "vax_multiplier" in scenario:
        new_inputs = apply_vax_scenario(new_inputs, scenario.pop("vax_multiplier"))

    def _scaled(param_name: str, scale) -> np.ndarray:
        """`inputs["params"][param_name]` times `scale`. `scale` may be a
        scalar (applied to every age group) or a per-age-group array matching
        the base param's length; any other length is a mistake, not a
        broadcast, so reject it rather than silently mis-broadcasting."""
        base = np.array(inputs["params"][param_name], dtype=float)
        scale_arr = np.asarray(scale, dtype=float)
        if scale_arr.ndim > 0 and scale_arr.shape != base.shape:
            raise ValueError(
                f"{param_name}_scale has length {scale_arr.shape[0]}, expected "
                f"a scalar or a length-{base.shape[0]} per-age-group sequence"
            )
        return base * scale_arr

    param_overrides = {}
    if "vax_susceptibility_scale" in scenario:
        param_overrides["vax_susceptibility"] = _scaled(
            "vax_susceptibility", scenario.pop("vax_susceptibility_scale"))
    if "IV_to_H_prop_scale" in scenario:
        param_overrides["IV_to_H_prop"] = _scaled(
            "IV_to_H_prop", scenario.pop("IV_to_H_prop_scale"))
    param_overrides.update(scenario)  # any remaining keys pass straight through

    if param_overrides:
        new_inputs["params"] = apply_overrides(new_inputs["params"], param_overrides)

    return new_inputs


# ── Detailed (per-age-group, per-compartment, per-transition) simulation ─────

def _simulate_detailed_core(inputs: dict, stochastic: bool, rng: np.random.Generator | None,
                            steps_per_day: int = 1):
    """
    Run one replication, tracking every compartment and transition flow per
    age group per day. Deterministic mode reproduces the same Euler update as
    `_simulate_core` when `steps_per_day=1`.

    Stochastic mode is a discrete-time chain-binomial model: each per-day
    rate r is used directly as a Binomial draw probability (not 1-exp(-r)),
    because the deterministic model IS a discrete-time Euler map, not a
    discretization of a continuous-time hazard — matching the rate directly
    keeps E[stochastic] equal to the deterministic trajectory (see inline
    comment below for why 1-exp(-r) would bias this). I/IV/H each have a
    single total hazard (I_out_rate / H_out_rate) shared by their two
    competing exits, so "leaves the compartment" is one Binomial draw, then
    split by a second Binomial on the age-specific proportion (I_to_H_prop /
    IV_to_H_prop / H_to_D_prop) — exact, not an approximation. Vaccination
    (S_to_SV) stays a deterministic scheduled count in both modes, since it
    comes from external vaccine-delivery data rather than a hazard rate;
    infection draws are taken from whatever susceptibles remain after that
    sub-step's vaccinations.

    `steps_per_day` subdivides each day into that many Euler/binomial
    sub-steps (rates scaled by dt=1/steps_per_day), so within-day nonlinear
    feedback (S depleting, I growing) is resolved more finely without
    changing the daily output granularity. Each day's transitions are the
    *sum* of its sub-step flows (so `TRANSITIONS` values are still "how many
    moved during this day" regardless of steps_per_day); each day's
    compartment counts are a *snapshot taken before that day's sub-steps run*
    (so day 0 is the initial condition, and day t is the state that day t's
    flows act on) — the natural convention once "the state" is no longer a
    single well-defined value within a day that has multiple sub-steps.

    Returns (compartments, transitions): dicts of name -> (num_days, A) array.
    """
    p          = inputs["params"]
    population = inputs["population"]
    num_days   = p["num_days"]
    A          = len(population)

    beta            = p["beta_baseline"]
    hum_impact      = p["humidity_impact"]
    rel_suscept     = p["relative_suscept"]
    I_rel_inf       = p["I_relative_infectiousness"]
    IV_rel_inf      = p["IV_relative_infectiousness"]
    E_to_I          = p["E_to_I_rate"]
    EV_to_IV        = p["EV_to_IV_rate"]
    I_out           = p["I_out_rate"]
    H_out           = p["H_out_rate"]
    I_to_H          = p["I_to_H_prop"]
    IV_to_H         = p["IV_to_H_prop"]
    H_to_D          = p["H_to_D_prop"]
    vax_suscept     = p["vax_susceptibility"]
    vax_pool        = p.get("vax_pool", VAX_POOL_DEFAULT)

    total_C  = inputs["total_C"]
    school_C = inputs["school_C"]
    work_C   = inputs["work_C"]

    humidity_arr  = inputs["humidity_arr"]
    is_school_arr = inputs["is_school_arr"]
    is_work_arr   = inputs["is_work_arr"]
    vax_arr       = inputs["vax_arr"]

    beta_mult = inputs.get("beta_multiplier_arr")
    if beta_mult is None:
        beta_mult = np.ones(num_days)

    E0 = p["E0_counts"]
    S  = population - E0;  E  = E0.copy()
    I  = np.zeros_like(population);  R  = np.zeros_like(population)
    SV = np.zeros_like(population);  EV = np.zeros_like(population)
    IV = np.zeros_like(population);  H  = np.zeros_like(population)
    D  = np.zeros_like(population)

    compartments = {c: np.zeros((num_days, A)) for c in COMPARTMENTS}
    transitions  = {t: np.zeros((num_days, A)) for t in TRANSITIONS}

    if stochastic:
        assert rng is not None, "stochastic=True requires an rng"

    dt = 1.0 / steps_per_day

    def binom(n, prob):
        assert rng is not None
        n = np.maximum(np.rint(n), 0.0)
        return rng.binomial(n.astype(np.int64), np.clip(prob, 0.0, 1.0)).astype(float)

    for t in range(num_days):
        # Snapshot compartments as of the *start* of day t, before any of
        # today's sub-steps run (day 0 is the initial condition).
        compartments["S"][t], compartments["E"][t]  = S, E
        compartments["I"][t], compartments["R"][t]  = I, R
        compartments["SV"][t], compartments["EV"][t] = SV, EV
        compartments["IV"][t], compartments["H"][t] = IV, H
        compartments["D"][t] = D

        C = (
            total_C
            - (1.0 - is_school_arr[t]) * school_C
            - (1.0 - is_work_arr[t])   * work_C
        )
        beta_adj = beta * beta_mult[t] * (1.0 + hum_impact * np.exp(-180.0 * humidity_arr[t]))

        day_flows = {name: np.zeros(A) for name in TRANSITIONS}

        for _ in range(steps_per_day):
            wtd_inf_prop = (I * I_rel_inf + IV * IV_rel_inf) / population
            foi = beta_adj * (C @ wtd_inf_prop)

            # Vaccination: exact scheduled count (this sub-step's share of
            # the day's schedule), same in both modes.
            S_to_SV = _vax_flow(vax_arr[t], S, SV, population, vax_pool, dt)
            S_after_vax = S - S_to_SV

            if stochastic:
                # Draw probabilities are the raw per-sub-step rates (rate*dt,
                # not 1-exp(-rate*dt)): the deterministic model IS a
                # discrete-time Euler map, not a discretization of a
                # continuous-time hazard, so matching the rate directly as a
                # Binomial probability keeps E[stochastic] equal to the
                # deterministic trajectory. Using 1-exp(-rate*dt) instead
                # would silently lengthen mean sojourn times (e.g. infectious
                # period 1/(1-exp(-I_out*dt))/dt > 1/I_out) and inflate
                # effective R, biasing every stochastic run high relative to
                # the calibrated deterministic/fitted dynamics.
                S_to_E        = binom(S_after_vax, foi * rel_suscept * dt)
                E_to_I_flow   = binom(E, E_to_I * dt)
                I_leave       = binom(I, I_out * dt)
                I_to_H_flow   = binom(I_leave, I_to_H)
                I_to_R_flow   = I_leave - I_to_H_flow
                SV_to_EV      = binom(SV, foi * vax_suscept * dt)
                EV_to_IV_flow = binom(EV, EV_to_IV * dt)
                IV_leave      = binom(IV, I_out * dt)
                IV_to_H_flow  = binom(IV_leave, IV_to_H)
                IV_to_R_flow  = IV_leave - IV_to_H_flow
                H_leave       = binom(H, H_out * dt)
                H_to_D_flow   = binom(H_leave, H_to_D)
                H_to_R_flow   = H_leave - H_to_D_flow
            else:
                S_to_E        = foi * rel_suscept * dt * S
                E_to_I_flow   = E_to_I * dt * E
                I_to_H_flow   = I_out * dt * I_to_H * I
                I_to_R_flow   = I_out * dt * (1.0 - I_to_H) * I
                SV_to_EV      = foi * vax_suscept * dt * SV
                EV_to_IV_flow = EV_to_IV * dt * EV
                IV_to_H_flow  = I_out * dt * IV_to_H * IV
                IV_to_R_flow  = I_out * dt * (1.0 - IV_to_H) * IV
                H_to_D_flow   = H_out * dt * H_to_D * H
                H_to_R_flow   = H_out * dt * (1.0 - H_to_D) * H

            S  = S  - S_to_E   - S_to_SV
            E  = E  + S_to_E   - E_to_I_flow
            I  = I  + E_to_I_flow - I_to_H_flow - I_to_R_flow
            R  = R  + I_to_R_flow + IV_to_R_flow + H_to_R_flow
            SV = SV + S_to_SV  - SV_to_EV
            EV = EV + SV_to_EV - EV_to_IV_flow
            IV = IV + EV_to_IV_flow - IV_to_H_flow - IV_to_R_flow
            H  = H  + I_to_H_flow + IV_to_H_flow - H_to_D_flow - H_to_R_flow
            D  = D  + H_to_D_flow

            day_flows["S_to_E"]   += S_to_E;    day_flows["S_to_SV"]  += S_to_SV
            day_flows["E_to_I"]   += E_to_I_flow
            day_flows["I_to_H"]   += I_to_H_flow;  day_flows["I_to_R"]  += I_to_R_flow
            day_flows["SV_to_EV"] += SV_to_EV;  day_flows["EV_to_IV"] += EV_to_IV_flow
            day_flows["IV_to_H"]  += IV_to_H_flow; day_flows["IV_to_R"] += IV_to_R_flow
            day_flows["H_to_D"]   += H_to_D_flow;  day_flows["H_to_R"]  += H_to_R_flow

        for name in TRANSITIONS:
            transitions[name][t] = day_flows[name]

    return compartments, transitions


def simulate_detailed(inputs: dict, stochastic: bool = False, n_reps: int = 1,
                      seed: int | None = None, steps_per_day: int = 1) -> xr.Dataset:
    """
    Run the model and return an xarray Dataset with per-age-group, per-day
    compartment counts (`COMPARTMENTS`) and transition flows (`TRANSITIONS`),
    dims `(replication, day, age_group)`.

    `stochastic=False` runs `n_reps` identical deterministic replications
    (kept only for API symmetry — pass `n_reps=1` in that case).
    `stochastic=True` draws `n_reps` independent chain-binomial replications
    from `seed` (see `_simulate_detailed_core`).
    `steps_per_day` subdivides each day into that many sub-steps for finer
    within-day dynamics; output stays daily (transitions summed, compartment
    counts snapshotted at day-start — see `_simulate_detailed_core`).
    """
    dates = inputs["dates"]
    num_days = len(dates)
    A = len(inputs["population"])

    comp_arr = {c: np.zeros((n_reps, num_days, A)) for c in COMPARTMENTS}
    trans_arr = {t: np.zeros((n_reps, num_days, A)) for t in TRANSITIONS}

    rng = np.random.default_rng(seed) if stochastic else None
    for r in range(n_reps):
        comps_r, trans_r = _simulate_detailed_core(inputs, stochastic, rng, steps_per_day)
        for c in COMPARTMENTS:
            comp_arr[c][r] = comps_r[c]
        for name in TRANSITIONS:
            trans_arr[name][r] = trans_r[name]

    data_vars = {c: (("replication", "day", "age_group"), comp_arr[c]) for c in COMPARTMENTS}
    data_vars.update({t: (("replication", "day", "age_group"), trans_arr[t]) for t in TRANSITIONS})

    return xr.Dataset(
        data_vars,
        coords={
            "replication": np.arange(n_reps),
            "day": dates,
            "age_group": AGE_GROUP_LABELS,
        },
    )


def total_transitions(ds: xr.Dataset, by_age: bool = False) -> xr.Dataset:
    """
    Sum every transition variable over the simulation (the `day` dimension),
    leaving `replication` (and `age_group` if `by_age=True`) intact. E.g. for
    a stochastic run, `total_transitions(ds)["I_to_H"]` is a length-`n_reps`
    array of total age-aggregated I->H counts, one per replication.
    """
    trans = ds[TRANSITIONS]
    summed = trans.sum(dim="day")
    if not by_age:
        summed = summed.sum(dim="age_group")
    return summed


if __name__ == "__main__":
    # Smoke test: default run, print a few summary stats of new_H.
    params = default_params()
    inputs = load_inputs(DATA_FOLDER, params)
    df = run_simulation(inputs)
    fast = simulate_new_H({}, inputs)
    assert np.allclose(df["new_H"].to_numpy(), fast), "simulate_new_H disagrees with run_simulation"
    print(f"num_days={params['num_days']}  peak new_H={df['new_H'].max():.2f} "
          f"on {df['new_H'].idxmax().date()}  total_H={df['new_H'].sum():.1f}")
    print("simulate_new_H matches run_simulation ✓")
