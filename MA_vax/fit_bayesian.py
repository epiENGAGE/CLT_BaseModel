"""Bayesian calibration of the MA_vax model — emcee + pyabc prototype.

Fits the deterministic SEIR+vaccination model in `model.py` to the observed
daily-hospitalization series, two ways:

  * emcee  — affine-invariant ensemble MCMC with a Negative-Binomial likelihood.
  * pyabc  — ABC-SMC (likelihood-free) with a weighted-RMSE distance.

Both optionally fit a low-dimensional time-varying transmission multiplier m(t):
monthly knots, piecewise-linear in log space, regularized by a Gaussian
random-walk prior on the month-to-month increments (smoothness controlled by
`--tau`). This is the anti-overfitting device and, for ABC, the prior is what
enforces smoothness.

Run:
    uv run python MA_vax/fit_bayesian.py --method both            # with m(t)
    uv run python MA_vax/fit_bayesian.py --method both --age-specific-ihr  # with m(t), ihr scaling per age group
    uv run python MA_vax/fit_bayesian.py --method both --no-tvbeta
    uv run python MA_vax/fit_bayesian.py --method emcee --age-specific-ihr
    uv run python MA_vax/fit_bayesian.py --method emcee --age-specific-ihr --no-tvbeta

Outputs figures + a parameter table into MA_vax/.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln

from MA_vax import model

warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))

IHR_SCALE_LOW, IHR_SCALE_HIGH = 0.1, 2.0


# ── Free-parameter specification ─────────────────────────────────────────────
# Structural params. `log10_E0_scale` is sampled in log10 space (LogUniform
# prior on the E0 multiplier); everything else is linear. Bounds double as the
# uniform prior box for both samplers.
@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float


def build_param_specs(age_specific_ihr: bool) -> list[ParamSpec]:
    """Structural param specs. `ihr_scale` is either one shared parameter, or
    one per age group (named `ihr_scale_{label}`) when `age_specific_ihr` is
    set, each with the same [IHR_SCALE_LOW, IHR_SCALE_HIGH] prior box."""
    specs = [
        ParamSpec("beta_baseline",   0.02, 0.06),
        ParamSpec("humidity_impact", 0.0,  1.0),
        ParamSpec("log10_E0_scale",  np.log10(0.2), np.log10(5.0)),
    ]
    if age_specific_ihr:
        specs += [ParamSpec(f"ihr_scale_{lbl}", IHR_SCALE_LOW, IHR_SCALE_HIGH)
                  for lbl in model.AGE_GROUP_LABELS]
    else:
        specs.append(ParamSpec("ihr_scale", IHR_SCALE_LOW, IHR_SCALE_HIGH))
    return specs


PHI_SPEC = ParamSpec("phi", 0.1, 100.0)  # NB dispersion, nuisance, emcee only

# Module-level structural-param state, set by `set_param_specs` (default:
# age_specific_ihr=False). Must be re-applied inside spawned emcee worker
# processes, which re-import this module fresh and don't inherit the parent's
# post-argparse assignment — see `_init_emcee_worker`.
PARAM_SPECS: list[ParamSpec] = build_param_specs(False)
N_STRUCT = len(PARAM_SPECS)  # number of structural (non-increment, non-phi) parameters
STRUCT_NAMES = [s.name for s in PARAM_SPECS]


def set_param_specs(age_specific_ihr: bool) -> None:
    global PARAM_SPECS, N_STRUCT, STRUCT_NAMES
    PARAM_SPECS = build_param_specs(age_specific_ihr)
    N_STRUCT = len(PARAM_SPECS)
    STRUCT_NAMES = [s.name for s in PARAM_SPECS]


def compute_knot_days(inputs: dict) -> np.ndarray:
    """Day indices of the 1st of each month in the sim window (anchored at 0)."""
    dates = inputs["dates"]
    knots = [t for t, d in enumerate(dates) if d.day == 1]
    if not knots or knots[0] != 0:
        knots = [0] + knots
    return np.array(knots, dtype=int)


def increment_names(n_incr: int) -> list[str]:
    return [f"dlogm{i + 1}" for i in range(n_incr)]


# ── Parameter-vector layout ──────────────────────────────────────────────────
# emcee:  [structural(4)] + [increments(n_incr)] + [phi]
# pyabc:  [structural(4)] + [increments(n_incr)]
def theta_to_overrides(struct) -> dict:
    beta, humidity, log10_e0 = struct[0], struct[1], struct[2]
    ihr = struct[3:N_STRUCT]  # 1 shared value, or 1 per age group
    return {
        "beta_baseline":   float(beta),
        "humidity_impact": float(humidity),
        "E0_scale":        float(10.0 ** log10_e0),
        "ihr_scale":       float(ihr[0]) if len(ihr) == 1 else np.asarray(ihr, dtype=float),
    }


def simulate_full(theta, inputs, knot_days, by_age: bool = False) -> np.ndarray:
    """Full-length new_H trajectory for a parameter vector (struct + increments).
    Shape (num_days,), or (num_days, num_age_groups) if `by_age`."""
    n_incr = len(knot_days) - 1
    overrides = theta_to_overrides(theta[:N_STRUCT])
    if n_incr > 0:
        m = model.build_beta_multiplier(
            theta[N_STRUCT:N_STRUCT + n_incr], knot_days, len(inputs["dates"]))
    else:
        m = None
    return model.simulate_new_H(overrides, inputs, beta_multiplier_arr=m, by_age=by_age)


def predict(theta, inputs, obs_idx, knot_days, by_age: bool = False) -> np.ndarray:
    """Simulated new_H at the observed day indices."""
    return simulate_full(theta, inputs, knot_days, by_age=by_age)[obs_idx]


# ── Observed-data alignment ──────────────────────────────────────────────────
def load_observed(inputs: dict):
    """
    Align the observed (total, all-ages) series onto the model day index.
    Returns (obs_values, obs_idx, weights); only days inside [0, num_days)
    are kept.
    """
    dates = inputs["dates"]
    start = dates[0]
    num_days = len(dates)

    df = pd.read_csv(os.path.join(model.DATA_FOLDER, "MA_flu_daily_hospitalizations_total.csv"))
    df["date"] = pd.to_datetime(df["Date"])
    idx = (df["date"] - start).dt.days.to_numpy()
    weight = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones(len(df))

    in_range = (idx >= 0) & (idx < num_days)
    return (
        df["total"].to_numpy(dtype=float)[in_range],
        idx[in_range],
        weight[in_range],
    )


# Column order in MA_flu_daily_hospitalizations.csv, matching model.AGE_GROUP_LABELS.
AGE_RAW_CSV_COLUMNS = ["0_0", "1_4", "5_12", "13_17", "18_49", "50_64", "65+"]


def load_observed_by_age(inputs: dict):
    """
    Per-age-group counterpart of `load_observed`. Returns
    (obs_values (n_obs, num_age_groups), obs_idx, weights), reusing the same
    `obs_idx`/`weights` as `load_observed` — the per-age CSV's date range is
    a strict superset of the total CSV's, so date alignment and the
    epidemic-peak time-weighting are identical, just evaluated per age group
    instead of on the age-summed total.
    """
    _, obs_idx, weights = load_observed(inputs)
    obs_dates = inputs["dates"][obs_idx]

    df = pd.read_csv(os.path.join(model.DATA_FOLDER, "MA_flu_daily_hospitalizations.csv"))
    df["date"] = pd.to_datetime(df["Date"])
    df = df.set_index("date")

    obs_values = df.loc[obs_dates, AGE_RAW_CSV_COLUMNS].to_numpy(dtype=float)
    return obs_values, obs_idx, weights


# ── Negative-Binomial (NB2) log-likelihood ───────────────────────────────────
def neg_binom_loglike(obs, mu, phi, weights=None) -> float:
    """
    NB2 log-likelihood, mean `mu`, dispersion `phi` (variance = mu + mu^2/phi).
    Uses gammaln so fractional observed counts (7-day-smoothed data) are handled.

    `obs`/`mu` are either shape (n_obs,) (total series) or (n_obs, n_age)
    (per-age-group series, one log-likelihood term per age per day).

    `weights` (optional, shape (n_obs,) — always per-day, even when `obs` is
    per-age) scales each day's log-likelihood contribution — mirrors the
    per-timepoint emphasis (e.g. up-weighting the epidemic peak) the generic
    accept-reject/gradient fitters apply via `point_weights`, normalized to
    mean 1 by the caller.
    """
    mu = np.maximum(mu, 1e-6)
    ll = (
        gammaln(obs + phi) - gammaln(phi) - gammaln(obs + 1.0)
        + phi * (np.log(phi) - np.log(phi + mu))
        + obs * (np.log(mu) - np.log(phi + mu))
    )
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if ll.ndim > w.ndim:
            w = w.reshape(w.shape + (1,) * (ll.ndim - w.ndim))
        ll = ll * w
    return float(np.sum(ll))


# ── emcee ────────────────────────────────────────────────────────────────────
# Per-worker context, populated once by the Pool initializer to avoid pickling
# the (large) inputs dict on every likelihood evaluation.
_CTX: dict | None = None


def _init_emcee_worker(tau, use_tvbeta, age_specific_ihr):
    global _CTX
    # Spawned workers re-import this module fresh, so the parent's post-argparse
    # set_param_specs() call in main() never reaches them — redo it here.
    set_param_specs(age_specific_ihr)
    params = model.default_params()
    inputs = model.load_inputs(model.DATA_FOLDER, params)
    # With age-specific ihr_scale, fit each age group's hospitalization curve
    # directly (otherwise the total alone barely constrains the extra
    # per-age dimensions); without it, fit the age-summed total as before.
    load_fn = load_observed_by_age if age_specific_ihr else load_observed
    obs_values, obs_idx, weights = load_fn(inputs)
    # Normalize so per-timepoint weights average to 1 — this keeps the overall
    # likelihood scale unchanged while shifting emphasis between days (e.g.
    # up-weighting the epidemic peak), matching the generic fitters' convention.
    w_mean = float(weights.mean()) if weights.size else 1.0
    if w_mean > 0:
        weights = weights / w_mean
    knot_days = compute_knot_days(inputs) if use_tvbeta else np.array([0])
    _CTX = {"inputs": inputs, "obs_values": obs_values, "obs_idx": obs_idx,
            "weights": weights, "knot_days": knot_days, "tau": tau,
            "by_age": age_specific_ihr}


def log_prob_emcee(theta) -> float:
    """log posterior: uniform structural priors + N(0,tau) RW prior on the
    m(t) increments + NB likelihood. Reads global _CTX."""
    ctx = _CTX
    kd, tau = ctx["knot_days"], ctx["tau"]
    n_incr = len(kd) - 1

    struct = theta[:N_STRUCT]
    incr = theta[N_STRUCT:N_STRUCT + n_incr]
    phi = theta[N_STRUCT + n_incr]

    # structural + phi hard bounds
    for val, spec in zip(struct, PARAM_SPECS):
        if not (spec.low <= val <= spec.high):
            return -np.inf
    if not (PHI_SPEC.low <= phi <= PHI_SPEC.high):
        return -np.inf

    # Gaussian random-walk prior on increments (smoothness regularizer)
    lp_incr = -0.5 * float(np.sum((incr / tau) ** 2)) if n_incr else 0.0

    try:
        mu = predict(theta, ctx["inputs"], ctx["obs_idx"], kd, by_age=ctx["by_age"])
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(mu)):
        return -np.inf
    ll = neg_binom_loglike(ctx["obs_values"], mu, phi, weights=ctx["weights"])
    total = ll + lp_incr
    return total if np.isfinite(total) else -np.inf


def run_emcee(knot_days, tau, use_tvbeta, age_specific_ihr, nwalkers=32, nsteps=3000, workers=None, seed=0):
    import emcee

    n_incr = len(knot_days) - 1 if use_tvbeta else 0
    names = STRUCT_NAMES + increment_names(n_incr) + [PHI_SPEC.name]
    ndim = len(names)
    rng = np.random.default_rng(seed)

    # emcee's red-blue ensemble moves require nwalkers >= 2*ndim; ndim grows
    # with --age-specific-ihr (+6) and --tvbeta (+n_incr), so the requested
    # --nwalkers can fall short without the caller doing that arithmetic.
    min_walkers = 2 * ndim + 2  # +2 margin above the hard minimum
    if nwalkers < min_walkers:
        # round up to even, emcee splits the ensemble into two halves per move
        min_walkers += min_walkers % 2
        print(f"[emcee] bumping nwalkers {nwalkers} -> {min_walkers} "
              f"(ndim={ndim} requires nwalkers >= {2 * ndim})")
        nwalkers = min_walkers

    # Initialise walkers: structural/phi in the inner 60% of their box,
    # increments in a tight ball near 0 (m(t) ≈ 1).
    p0 = np.empty((nwalkers, ndim))
    for j, spec in enumerate(PARAM_SPECS):
        pad = 0.2 * (spec.high - spec.low)
        p0[:, j] = rng.uniform(spec.low + pad, spec.high - pad, size=nwalkers)
    for j in range(N_STRUCT, N_STRUCT + n_incr):
        p0[:, j] = rng.normal(0.0, 0.05, size=nwalkers)
    pad = 0.2 * (PHI_SPEC.high - PHI_SPEC.low)
    p0[:, -1] = rng.uniform(PHI_SPEC.low + pad, PHI_SPEC.high - pad, size=nwalkers)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[emcee] {nwalkers} walkers x {nsteps} steps on {workers} workers, "
          f"ndim={ndim} ({n_incr} m(t) increments)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_emcee_worker,
                  initargs=(tau, use_tvbeta, age_specific_ihr)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_emcee, pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    # Burn-in / thinning from the integrated autocorrelation time.
    try:
        tau_ac = sampler.get_autocorr_time(tol=0)
        burn = int(3 * np.nanmax(tau_ac))
        thin = max(1, int(0.5 * np.nanmin(tau_ac)))
    except Exception:
        burn, thin = nsteps // 3, 1
    burn = min(burn, nsteps // 2)

    # Drop walkers stuck in a bad mode (mean log-prob far below the cluster).
    lp = sampler.get_log_prob(discard=burn, thin=thin)
    m = np.mean(lp, axis=0)
    med, mad = np.median(m), np.median(np.abs(m - np.median(m)))
    good = m >= med - 5.0 * (mad + 1e-9)
    if good.sum() < 0.75 * nwalkers:
        good = m >= np.percentile(m, 25)
    n_dropped = int(nwalkers - good.sum())

    chain = sampler.get_chain(discard=burn, thin=thin)
    flat = chain[:, good, :].reshape(-1, ndim)
    log_prob_flat = lp[:, good].reshape(-1)  # same (step, walker) ordering as `flat`
    acc = float(np.mean(sampler.acceptance_fraction[good]))
    print(f"[emcee] acceptance={acc:.2f}  burn={burn}  thin={thin}  "
          f"dropped {n_dropped}/{nwalkers} stuck walkers  posterior draws={len(flat)}")
    return {"names": names, "samples": flat, "log_prob": log_prob_flat,
            "acceptance": acc, "knot_days": knot_days, "age_specific_ihr": age_specific_ihr}


# ── pyabc ─────────────────────────────────────────────────────────────────────
def run_pyabc(knot_days, tau, use_tvbeta, age_specific_ihr, pop_size=200, max_gens=12, seed=0):
    import pyabc

    params = model.default_params()
    inputs = model.load_inputs(model.DATA_FOLDER, params)
    # See run_emcee: fit each age group's curve directly when ihr_scale is
    # age-specific, otherwise fit the age-summed total as before.
    load_fn = load_observed_by_age if age_specific_ihr else load_observed
    obs_values, obs_idx, weights = load_fn(inputs)
    w = weights / weights.sum()
    # `w` is always per-day; broadcast against a per-age (n_obs, n_age) diff.
    w_b = w[:, None] if age_specific_ihr else w
    n_incr = len(knot_days) - 1 if use_tvbeta else 0
    incr_names = increment_names(n_incr)
    names = STRUCT_NAMES + incr_names

    def pyabc_model(p):
        theta = [p[n] for n in names]
        return {"y": predict(theta, inputs, obs_idx, knot_days, by_age=age_specific_ihr)}

    def distance(x, x0):
        diff = x["y"] - x0["y"]
        return float(np.sqrt(np.sum(w_b * diff * diff)))

    prior_dict = {s.name: pyabc.RV("uniform", s.low, s.high - s.low) for s in PARAM_SPECS}
    # Gaussian random-walk prior on increments (same smoothness regularizer).
    prior_dict.update({n: pyabc.RV("norm", 0.0, tau) for n in incr_names})
    prior = pyabc.Distribution(**prior_dict)

    print(f"[pyabc] ABC-SMC pop={pop_size} x up to {max_gens} generations "
          f"(single-core), ndim={len(names)} ({n_incr} m(t) increments)")
    abc = pyabc.ABCSMC(
        pyabc_model, prior, distance,
        population_size=pop_size,
        sampler=pyabc.sampler.SingleCoreSampler(),
    )
    db_path = os.path.join(tempfile.gettempdir(), f"ma_vax_abc_{seed}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    abc.new("sqlite:///" + db_path, {"y": obs_values})
    try:
        history = abc.run(max_nr_populations=max_gens, minimum_epsilon=0.0)
    except (AssertionError, ZeroDivisionError, FloatingPointError):
        history = abc.history
        print(f"[pyabc] Weight collapse detected — stopping early, "
              f"using last stable generation t={history.max_t}")

    df, weights_post = history.get_distribution(t=history.max_t)
    eps = history.get_all_populations()["epsilon"].to_numpy()
    print(f"[pyabc] generations={history.max_t + 1}  final eps={eps[-1]:.3f}  "
          f"eps schedule={np.round(eps, 2)}")

    return {"names": names, "samples": df[names].to_numpy(),
            "weights": np.asarray(weights_post), "epsilon": eps,
            "knot_days": knot_days, "age_specific_ihr": age_specific_ihr}


# ── Diagnostics / comparison output ──────────────────────────────────────────
def _weighted_quantile(values, q, weights=None):
    if weights is None:
        return np.quantile(values, q)
    order = np.argsort(values)
    v, wq = values[order], weights[order]
    cdf = np.cumsum(wq) - 0.5 * wq
    cdf /= np.sum(wq)
    return np.interp(q, cdf, v)


def summarize(result, only=None) -> pd.DataFrame:
    names = result["names"]
    samples = result["samples"]
    weights = result.get("weights")
    rows = []
    for j, name in enumerate(names):
        if only is not None and name not in only:
            continue
        col = samples[:, j]
        rows.append({
            "param": name,
            "mean": np.average(col, weights=weights),
            "q05": _weighted_quantile(col, 0.05, weights),
            "q95": _weighted_quantile(col, 0.95, weights),
        })
    return pd.DataFrame(rows).set_index("param")


def _draw_indices(result, n_draws, rng):
    n = len(result["samples"])
    weights = result.get("weights")
    if weights is not None:
        return rng.choice(n, size=min(n_draws, n), p=weights / weights.sum())
    return rng.choice(n, size=min(n_draws, n), replace=False)


def posterior_predictive(result, inputs, n_draws=300, seed=0):
    """Percentile bands over all model days for new_H."""
    rng = np.random.default_rng(seed)
    pick = _draw_indices(result, n_draws, rng)
    kd = result["knot_days"]
    num_days = len(inputs["dates"])
    trajs = np.empty((len(pick), num_days))
    for i, k in enumerate(pick):
        trajs[i] = simulate_full(result["samples"][k], inputs, kd)
    return np.percentile(trajs, [5, 25, 50, 75, 95], axis=0)


def posterior_predictive_by_age(result, inputs, n_draws=300, seed=0):
    """Per-age-group counterpart of `posterior_predictive`: percentile bands
    over all model days for new_H, shape (5, num_days, num_age_groups)."""
    rng = np.random.default_rng(seed)
    pick = _draw_indices(result, n_draws, rng)
    kd = result["knot_days"]
    num_days = len(inputs["dates"])
    n_age = len(model.AGE_GROUP_LABELS)
    trajs = np.empty((len(pick), num_days, n_age))
    for i, k in enumerate(pick):
        trajs[i] = simulate_full(result["samples"][k], inputs, kd, by_age=True)
    return np.percentile(trajs, [5, 25, 50, 75, 95], axis=0)


def m_of_t_bands(result, num_days, n_draws=500, seed=1):
    """Percentile bands over all model days for the m(t) multiplier."""
    kd = result["knot_days"]
    n_incr = len(kd) - 1
    if n_incr <= 0:
        return None
    rng = np.random.default_rng(seed)
    pick = _draw_indices(result, n_draws, rng)
    curves = np.empty((len(pick), num_days))
    for i, k in enumerate(pick):
        incr = result["samples"][k][N_STRUCT:N_STRUCT + n_incr]
        curves[i] = model.build_beta_multiplier(incr, kd, num_days)
    return np.percentile(curves, [5, 50, 95], axis=0)


def point_estimate_theta(res) -> np.ndarray:
    """Posterior-mean parameter vector (weighted mean if `res` carries weights)."""
    samples, weights = res["samples"], res.get("weights")
    return np.array([np.average(samples[:, j], weights=weights)
                      for j in range(samples.shape[1])])


def best_fit_theta(res) -> np.ndarray:
    """
    Single highest-support parameter vector: the emcee sample with the
    highest log-posterior (`res["log_prob"]`), or the highest-weight pyabc
    particle (`res["weights"]`). Unlike `point_estimate_theta`, which averages
    each parameter marginally and can land on a combination the sampler never
    actually visited (breaking whatever correlation structure the posterior
    has between parameters), this is an actual sampled point.
    """
    samples = res["samples"]
    if "log_prob" in res:
        idx = int(np.argmax(res["log_prob"]))
    else:
        idx = int(np.argmax(res["weights"]))
    return samples[idx]


def _expand_ihr_scale_for_csv(overrides: dict) -> dict:
    """`overrides["ihr_scale"]` may be a scalar or a per-age array (see
    `theta_to_overrides`); a DataFrame row can't hold an array cell, so an
    array is spread into one `ihr_scale__{age_label}` column per age group.
    Inverse of `_collapse_ihr_scale_from_csv`."""
    val = overrides.get("ihr_scale")
    if not isinstance(val, np.ndarray):
        return overrides
    overrides = dict(overrides)
    del overrides["ihr_scale"]
    for lbl, v in zip(model.AGE_GROUP_LABELS, val):
        overrides[f"ihr_scale__{lbl}"] = float(v)
    return overrides


def _collapse_ihr_scale_from_csv(overrides: dict) -> dict:
    """Inverse of `_expand_ihr_scale_for_csv`: reassemble any
    `ihr_scale__{age_label}` columns back into `overrides["ihr_scale"]` as a
    per-age array, in `model.AGE_GROUP_LABELS` order."""
    prefix = "ihr_scale__"
    ihr_cols = {k: v for k, v in overrides.items() if k.startswith(prefix)}
    if not ihr_cols:
        return overrides
    overrides = {k: v for k, v in overrides.items() if k not in ihr_cols}
    overrides["ihr_scale"] = np.array(
        [ihr_cols[prefix + lbl] for lbl in model.AGE_GROUP_LABELS], dtype=float)
    return overrides


def save_csv_outputs(results: dict, inputs, output_folder: str):
    """Write, per method: a full posterior summary, and two 'ready to run'
    point-estimate parameter tables + (if m(t) was fit) beta-multiplier
    curves — one for the posterior mean (`fit_parameters_{method}.csv`), one
    for the single highest-support sample (`fit_parameters_best_{method}.csv`,
    see `best_fit_theta`). Together each parameter table and its matching
    multiplier curve are everything `model.simulate_new_H` needs to reproduce
    that point-estimate fit."""
    dates = inputs["dates"]
    num_days = len(dates)

    for method, res in results.items():
        names = res["names"]

        summary_path = os.path.join(output_folder, f"fit_summary_{method}.csv")
        summarize(res).to_csv(summary_path)
        print(f"[csv] {summary_path}")

        kd = res["knot_days"]
        n_incr = len(kd) - 1

        for tag, theta in (("", point_estimate_theta(res)), ("_best", best_fit_theta(res))):
            overrides = theta_to_overrides(theta[:N_STRUCT])
            if PHI_SPEC.name in names:
                overrides[PHI_SPEC.name] = float(theta[names.index(PHI_SPEC.name)])
            overrides = _expand_ihr_scale_for_csv(overrides)
            params_path = os.path.join(output_folder, f"fit_parameters{tag}_{method}.csv")
            pd.DataFrame([overrides]).to_csv(params_path, index=False)
            print(f"[csv] {params_path}")

            if n_incr > 0:
                incr = theta[N_STRUCT:N_STRUCT + n_incr]
                m = model.build_beta_multiplier(incr, kd, num_days)
                mult_path = os.path.join(output_folder, f"fit_beta_multiplier{tag}_{method}.csv")
                pd.DataFrame({"date": dates, "beta_multiplier": m}).to_csv(mult_path, index=False)
                print(f"[csv] {mult_path}")


def load_fitted_run(output_folder: str, method: str, point: str = "best") -> tuple[dict, np.ndarray | None]:
    """
    Read back a `save_csv_outputs` run and return `(overrides, beta_multiplier_arr)`
    ready to pass straight to `model.simulate_new_H`:

        overrides, m = load_fitted_run(output_folder, "emcee")
        new_H = model.simulate_new_H(overrides, inputs, beta_multiplier_arr=m)

    `point`: `"mean"` (posterior-mean parameter vector, averaged marginally
    per parameter -- ignores correlations between parameters, and can land on
    a combination the sampler never actually visited) or `"best"` (the single
    sampled parameter vector with the highest posterior support -- see
    `best_fit_theta`; preserves whatever correlation structure the posterior
    has, since it's an actual sampled point rather than a per-parameter
    average).

    `beta_multiplier_arr` is None if the corresponding
    `fit_beta_multiplier*_{method}.csv` isn't present (constant-transmission
    run, `--no-tvbeta`).
    """
    tag = "" if point == "mean" else f"_{point}"
    params_path = os.path.join(output_folder, f"fit_parameters{tag}_{method}.csv")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"{params_path} not found -- point='{point}' requires re-running "
            "fit_bayesian.py (older output folders only have the 'mean' point estimate)."
        )
    overrides = pd.read_csv(params_path).iloc[0].to_dict()
    overrides.pop(PHI_SPEC.name, None)  # NB dispersion is a likelihood nuisance param, not a model input
    overrides = _collapse_ihr_scale_from_csv(overrides)

    mult_path = os.path.join(output_folder, f"fit_beta_multiplier{tag}_{method}.csv")
    beta_multiplier_arr = (
        pd.read_csv(mult_path)["beta_multiplier"].to_numpy() if os.path.exists(mult_path) else None
    )
    return overrides, beta_multiplier_arr


def make_figures(results: dict, inputs, obs_values, obs_idx, output_folder: str):
    import corner

    dates = inputs["dates"]
    num_days = len(dates)
    colors = {"emcee": "C0", "pyabc": "C1"}

    # 1) Posterior predictive overlay
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(dates[obs_idx], obs_values, s=14, color="red", zorder=5, label="Observed")
    for method, res in results.items():
        qs = posterior_predictive(res, inputs)
        c = colors.get(method, "C2")
        ax.plot(dates, qs[2], color=c, lw=1.8, label=f"{method} median")
        ax.fill_between(dates, qs[0], qs[4], color=c, alpha=0.15)
        ax.fill_between(dates, qs[1], qs[3], color=c, alpha=0.25)
    ax.set_ylabel("Daily new hospitalizations")
    ax.set_title("Posterior predictive vs observed")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    p = os.path.join(output_folder, "fit_posterior_predictive.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"[fig] {p}")

    # 1b) Per-age-group posterior predictive (methods fit with age-specific ihr_scale)
    age_specific_results = {m: r for m, r in results.items() if r.get("age_specific_ihr")}
    if age_specific_results:
        obs_age_values, obs_age_idx, _ = load_observed_by_age(inputs)
        fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
        axes = axes.flatten()
        for i, age in enumerate(model.AGE_GROUP_LABELS):
            ax = axes[i]
            ax.scatter(dates[obs_age_idx], obs_age_values[:, i], s=10, color="red",
                       zorder=5, label="Observed")
            for method, res in age_specific_results.items():
                qs = posterior_predictive_by_age(res, inputs)
                c = colors.get(method, "C2")
                ax.plot(dates, qs[2][:, i], color=c, lw=1.5, label=f"{method} median")
                ax.fill_between(dates, qs[0][:, i], qs[4][:, i], color=c, alpha=0.15)
                ax.fill_between(dates, qs[1][:, i], qs[3][:, i], color=c, alpha=0.25)
            ax.set_title(age)
            ax.grid(True, alpha=0.3)
        axes[-1].axis("off")
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle("Posterior predictive vs observed, by age group")
        fig.autofmt_xdate()
        fig.tight_layout()
        p = os.path.join(output_folder, "fit_posterior_predictive_by_age.png")
        fig.savefig(p, dpi=130); plt.close(fig); print(f"[fig] {p}")

    # 2) m(t) transmission multiplier bands
    if any(len(r["knot_days"]) > 1 for r in results.values()):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        for method, res in results.items():
            band = m_of_t_bands(res, num_days)
            if band is None:
                continue
            c = colors.get(method, "C2")
            ax.plot(dates, band[1], color=c, lw=1.8, label=f"{method} median")
            ax.fill_between(dates, band[0], band[2], color=c, alpha=0.2)
        ax.set_ylabel("m(t) transmission multiplier")
        ax.set_title("Fitted time-varying transmission multiplier (90% CI)")
        ax.legend()
        fig.autofmt_xdate(); fig.tight_layout()
        p = os.path.join(output_folder, "fit_beta_multiplier.png")
        fig.savefig(p, dpi=130); plt.close(fig); print(f"[fig] {p}")

    # 3) Corner plot — structural params + phi only (skip increments for legibility)
    keep = set(STRUCT_NAMES + [PHI_SPEC.name])
    for method, res in results.items():
        idx = [i for i, n in enumerate(res["names"]) if n in keep]
        fig = corner.corner(
            res["samples"][:, idx], labels=[res["names"][i] for i in idx],
            weights=res.get("weights"), show_titles=True,
            title_fmt=".3g", quantiles=[0.05, 0.5, 0.95],
        )
        p = os.path.join(output_folder, f"fit_corner_{method}.png")
        fig.savefig(p, dpi=110); plt.close(fig); print(f"[fig] {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["emcee", "pyabc", "both"], default="both")
    ap.add_argument("--tvbeta", dest="tvbeta", action="store_true", default=True,
                    help="fit time-varying transmission m(t) [default]")
    ap.add_argument("--no-tvbeta", dest="tvbeta", action="store_false",
                    help="disable m(t); constant transmission")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="RW-prior smoothness for m(t) increments")
    ap.add_argument("--age-specific-ihr", dest="age_specific_ihr", action="store_true",
                    default=False,
                    help="fit a separate ihr_scale per age group instead of one shared value")
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_param_specs(args.age_specific_ihr)

    params = model.default_params()
    inputs = model.load_inputs(model.DATA_FOLDER, params)
    obs_values, obs_idx, _weights = load_observed(inputs)
    knot_days = compute_knot_days(inputs) if args.tvbeta else np.array([0])
    n_incr = len(knot_days) - 1 if args.tvbeta else 0
    print(f"Observed points: {len(obs_values)} (days {obs_idx.min()}-{obs_idx.max()} "
          f"of {len(inputs['dates'])})")
    print(f"m(t): tvbeta={args.tvbeta}  knots={len(knot_days)}  increments={n_incr}  tau={args.tau}")
    print(f"ihr_scale: age_specific={args.age_specific_ihr}")

    results = {}
    if args.method in ("emcee", "both"):
        results["emcee"] = run_emcee(knot_days, args.tau, args.tvbeta, args.age_specific_ihr,
                                     args.nwalkers, args.nsteps, args.workers, args.seed)
    if args.method in ("pyabc", "both"):
        results["pyabc"] = run_pyabc(knot_days, args.tau, args.tvbeta, args.age_specific_ihr,
                                     args.pop, args.gens, args.seed)

    only = set(STRUCT_NAMES + [PHI_SPEC.name])
    print("\n=== Posterior summary (mean [5%, 95%]) — structural params ===")
    for method, res in results.items():
        print(f"\n[{method}]")
        print(summarize(res, only=only).to_string(float_format=lambda x: f"{x:.4g}"))

    now = datetime.now()
    output_folder = os.path.join(HERE, "outputs_" + now.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    os.makedirs(output_folder, exist_ok=True)

    save_csv_outputs(results, inputs, output_folder)
    make_figures(results, inputs, obs_values, obs_idx, output_folder)
    print("\nDone.")


if __name__ == "__main__":
    main()
