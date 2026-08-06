"""
Next-generation-matrix (NGM) reproduction-number analysis for MA_vax.

Computes R0/Reff, with and without vaccination, decomposed by age group and
vaccination status, accounting for the age x age contact matrices. Two modes:

  - Static: a single reproduction number using the school/work-calendar-averaged
    contact matrix and beta_baseline only (no humidity/m(t) modulation) --
    isolates the *structural* effect of vaccination coverage/reduced
    susceptibility on transmission potential, independent of any time-varying
    forcing.
  - Time-varying: R0(t)/Reff(t) day by day, using each day's actual contact
    matrix C(t) and beta_adj(t) = beta_baseline * m(t) * humidity forcing.

See `vaccination_impact_on_R0.md` for the full write-up and results.

Derivation: new infections enter only via S_to_E/SV_to_EV (bilinear in the
susceptible/vaccinated-susceptible pool x weighted infectious prevalence
through the contact matrix); E/EV have no exit besides progressing to I/IV
with probability 1, so they can be integrated out analytically, leaving a
2A x 2A NGM directly over (I_1..I_A, IV_1..IV_A) with average infectious
duration 1/I_out_rate. R0/Reff is the spectral radius of that matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from MA_vax import counterfactual as cf
from MA_vax import model

AGE = model.AGE_GROUP_LABELS
A = len(AGE)


def ngm_matrix(beta_adj: float, C: np.ndarray, S0: np.ndarray, SV0: np.ndarray,
                population: np.ndarray, params: dict) -> np.ndarray:
    """2A x 2A next-generation matrix over (I_1..I_A, IV_1..IV_A) at the
    disease-free state (S0, SV0) [E=I=EV=IV=0 elsewhere]."""
    rel_suscept = params["relative_suscept"]
    I_rel_inf = params["I_relative_infectiousness"]
    IV_rel_inf = params["IV_relative_infectiousness"]
    I_out_rate = params["I_out_rate"]
    vax_suscept = np.asarray(params["vax_susceptibility"], dtype=float)

    M = np.zeros((2 * A, 2 * A))
    for i in range(A):
        for j in range(A):
            common = beta_adj * C[i, j] / (I_out_rate * population[j])
            M[i, j]         = common * rel_suscept * S0[i] * I_rel_inf     # I_i  <- I_j
            M[i, A + j]     = common * rel_suscept * S0[i] * IV_rel_inf    # I_i  <- IV_j
            M[A + i, j]     = common * vax_suscept[i] * SV0[i] * I_rel_inf   # IV_i <- I_j
            M[A + i, A + j] = common * vax_suscept[i] * SV0[i] * IV_rel_inf  # IV_i <- IV_j
    return M


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def load_default_base_inputs():
    return cf.load_base_inputs(
        fit_folder="MA_vax/outputs_2026-07-30_age_ihr_scale",
        method="emcee", point="best")


def average_contact_matrix(base_inputs: dict) -> np.ndarray:
    """Effective contact matrix averaged over the school/work calendar."""
    mean_school = base_inputs["is_school_arr"].mean()
    mean_work = base_inputs["is_work_arr"].mean()
    return (
        base_inputs["total_C"]
        - (1.0 - mean_school) * base_inputs["school_C"]
        - (1.0 - mean_work) * base_inputs["work_C"]
    )


def static_r0_analysis(base_inputs: dict) -> dict:
    """Structural R0/Rv using the calendar-averaged contact matrix and
    beta_baseline (no humidity/m(t)). Returns a dict of results."""
    p = base_inputs["params"]
    population = np.asarray(base_inputs["population"], dtype=float)
    beta = p["beta_baseline"]
    C_avg = average_contact_matrix(base_inputs)

    coverage = np.array([cf._cumulative_coverage(base_inputs, i) for i in range(A)])

    S0_novax, SV0_novax = population.copy(), np.zeros(A)
    R0_novax = spectral_radius(ngm_matrix(beta, C_avg, S0_novax, SV0_novax, population, p))

    S0_vax, SV0_vax = population * (1 - coverage), population * coverage
    Rv = spectral_radius(ngm_matrix(beta, C_avg, S0_vax, SV0_vax, population, p))

    coverage_sweep = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cov = coverage * frac
        Sf, SVf = population * (1 - cov), population * cov
        Rf = spectral_radius(ngm_matrix(beta, C_avg, Sf, SVf, population, p))
        coverage_sweep.append((frac, Rf))

    per_age = []
    for i, lbl in enumerate(AGE):
        cov = np.zeros(A)
        cov[i] = coverage[i]
        Si, SVi = population * (1 - cov), population * cov
        Ri = spectral_radius(ngm_matrix(beta, C_avg, Si, SVi, population, p))
        per_age.append((lbl, coverage[i], Ri))

    return {
        "R0_novax": R0_novax, "Rv": Rv, "coverage": coverage,
        "coverage_sweep": coverage_sweep, "per_age": per_age,
    }


def time_varying_r0_analysis(base_inputs: dict) -> pd.DataFrame:
    """Day-by-day R0(t)/Reff(t), with and without vaccination, using each
    day's actual contact matrix and beta_adj(t) (humidity + fitted m(t))."""
    p = base_inputs["params"]
    population = np.asarray(base_inputs["population"], dtype=float)
    beta_baseline = p["beta_baseline"]
    hum_impact = p["humidity_impact"]

    dates = base_inputs["dates"]
    num_days = len(dates)
    humidity_arr = base_inputs["humidity_arr"]
    is_school_arr = base_inputs["is_school_arr"]
    is_work_arr = base_inputs["is_work_arr"]
    total_C, school_C, work_C = base_inputs["total_C"], base_inputs["school_C"], base_inputs["work_C"]
    beta_mult = base_inputs.get("beta_multiplier_arr")
    if beta_mult is None:
        beta_mult = np.ones(num_days)

    def beta_adj_at(t):
        return beta_baseline * beta_mult[t] * (1.0 + hum_impact * np.exp(-180.0 * humidity_arr[t]))

    def C_at(t):
        return total_C - (1.0 - is_school_arr[t]) * school_C - (1.0 - is_work_arr[t]) * work_C

    # With-vaccination run: actual S(t), SV(t) (vaccination + natural depletion).
    ds = model.simulate_detailed(base_inputs, stochastic=False, n_reps=1).isel(replication=0)
    S_t, SV_t = ds["S"].to_numpy(), ds["SV"].to_numpy()
    newH_t = (ds["I_to_H"] + ds["IV_to_H"]).to_numpy().sum(axis=1)
    prevalence_t = (ds["I"] + ds["IV"]).to_numpy().sum(axis=1)
    incidence_t = (ds["S_to_E"] + ds["SV_to_EV"]).to_numpy().sum(axis=1)
    cum_vax_frac = np.cumsum(ds["S_to_SV"].to_numpy(), axis=0) / population[None, :]

    # No-vaccination run: actual S(t) (natural depletion only, no SV pool).
    novax_inputs = model.apply_scenario(base_inputs, cf.no_vaccine_scenario())
    ds_nv = model.simulate_detailed(novax_inputs, stochastic=False, n_reps=1).isel(replication=0)
    S_t_nv = ds_nv["S"].to_numpy()
    zeros_A = np.zeros(A)

    rows = []
    for t in range(num_days):
        beta_adj_t = beta_adj_at(t)
        C_t = C_at(t)

        S0_dfe = population * (1 - cum_vax_frac[t])
        SV0_dfe = population * cum_vax_frac[t]
        R0_t = spectral_radius(ngm_matrix(beta_adj_t, C_t, S0_dfe, SV0_dfe, population, p))
        R0_novax_t = spectral_radius(ngm_matrix(beta_adj_t, C_t, population, zeros_A, population, p))

        Reff_t = spectral_radius(ngm_matrix(beta_adj_t, C_t, S_t[t], SV_t[t], population, p))
        Reff_novax_t = spectral_radius(ngm_matrix(beta_adj_t, C_t, S_t_nv[t], zeros_A, population, p))

        rows.append({
            "date": dates[t], "m_t": beta_mult[t], "beta_adj": beta_adj_t,
            "R0_t": R0_t, "R0_novax_t": R0_novax_t,
            "Reff_t": Reff_t, "Reff_novax_t": Reff_novax_t,
            "new_H": newH_t[t], "prevalence": prevalence_t[t], "incidence": incidence_t[t],
        })

    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    base_inputs = load_default_base_inputs()

    static = static_r0_analysis(base_inputs)
    print("=== Static (calendar-averaged contact matrix, beta_baseline only) ===")
    print(f"R0 (no vaccination): {static['R0_novax']:.3f}")
    print(f"Rv (fitted coverage): {static['Rv']:.3f}  "
          f"({(static['R0_novax']-static['Rv'])/static['R0_novax']*100:.1f}% reduction)")

    df = time_varying_r0_analysis(base_inputs)
    df.to_csv("MA_vax/R_timeseries.csv")
    print("\nTime-varying R0(t)/Reff(t) written to MA_vax/R_timeseries.csv")
