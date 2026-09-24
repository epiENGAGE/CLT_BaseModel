"""Effective (population-collapsed) beta over the season for the four MA_vax fits.

beta_eff(t) = beta_baseline * m(t) * (1 + humidity_impact*exp(-180*h(t)))
              * sum_a w_a * susc_a(t)

i.e. the transmission rate per contact, averaged over the population's
susceptibility -- the contact matrix is deliberately NOT folded in (the
reproduction numbers below carry it instead).

with  susc_a(t) = (1 - v_a(t))*relative_suscept + v_a(t)*vax_susceptibility_a
      v_a(t)    = cumulative SCHEDULED coverage of age a through day t, lagged
                  vax_transfer_delay_days
      w_a       = N_a / N

Two reproduction numbers come from the same ingredients, as the dominant
eigenvalue of the next-generation matrix

      K_ab(t) = beta_adj(t) * C_ab(t) * susc_pop_a(t) / (N_b * I_out_rate)

      R_v(t)   susc_pop_a = N_a * susc_a(t)  -- vaccination and contacts only,
               NO infection-acquired immunity (available for every posterior
               draw without simulating)
      R_eff(t) susc_pop_a = relative_suscept*S_a(t) + vax_susceptibility_a*SV_a(t)
               taken from the baseline-scenario parquet of the matching
               param-set run, i.e. the true effective R including susceptible
               depletion
"""
import ast, csv, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/rfp437/Work/CityLevelTransmission/CLT_BaseModel")
MA = ROOT / "generic_core/examples/MA_vax"
SA = MA / "MA_vax_single_age"
ARCH = MA / "archive/2026-08-high-vax-rates"

CASES = [
    ("7-age, m(t)",    MA / "model_config_MA_vax.json",        MA / "fitted_params_MA_vax.json",
     MA / "simulation_output_MA_vax_param_set_stochastic"),
    ("7-age, no m(t)", MA / "model_config_MA_vax.json",        MA / "fitted_params_MA_vax_no_transmission_multiplier.json",
     MA / "simulation_output_MA_vax_no_transmission_multiplier_param_set_stochastic"),
    ("1-age, m(t)",    SA / "model_config_MA_single_age.json", SA / "fitted_params_MA_single_age.json",
     SA / "simulation_output_single_age"),
    ("1-age, no m(t)", SA / "model_config_MA_single_age.json", SA / "fitted_params_MA_single_age_no_transmission_multiplier.json",
     SA / "simulation_output_single_age_no_transmission_multiplier"),
    # Archived Aug-2026 run of the 7-age m(t) model, fitted and simulated under
    # the higher vaccination schedule (55.9% population coverage vs 38.2% in
    # the current data). That schedule CSV is no longer on disk, so this case
    # takes its vaccination coverage from the run's own S_to_SV transfers --
    # see COVERAGE_FROM_SIM.
    ("7-age, m(t), high vax", ARCH / "model_config_2026-08-high-vax-rates.json",
     MA / "archive/fitted_params.json",
     ARCH / "simulation_output_param_set_stochastic_2026-08-high-vax-rates"),
]

# Cases whose per-age vaccination coverage v_a(t) cannot be read from the
# config's schedule CSV and is reconstructed instead as the cumulative
# DELIVERED S_to_SV transfers over that age group's population. The model's
# 14-day dose->immunity delay is already applied inside S_to_SV, so no further
# lag is taken. Delivered coverage runs 0.6-2.3 pp below scheduled (the cap in
# the vaccination flow skips anyone no longer in S).
COVERAGE_FROM_SIM = {"7-age, m(t), high vax"}

# run_simulations_*_param_set_stochastic.py draws its param sets as
# default_rng(SEED_BASE).choice(n, size=n, replace=False) -- a full but
# shuffled permutation -- so parquet rep i is accepted_params[SEED_PERM[i]].
SEED_BASE = 42


def tv_knot_days(sim_days, spacing):
    knots = list(range(0, int(sim_days), max(1, int(spacing))))
    if not knots or knots[-1] != int(sim_days) - 1:
        knots.append(int(sim_days) - 1)
    return knots


def m_of_t(params, knot_days, num_days):
    incr = [params[k] for k in sorted((k for k in params if k.startswith("m_dlog_")),
                                      key=lambda s: int(s.split("_")[-1]))]
    if not incr:
        return np.ones(num_days)
    g = np.concatenate([[0.0], np.cumsum(np.asarray(incr, float))])
    return np.exp(np.interp(np.arange(num_days), np.asarray(knot_days, float), g))


def load_schedules(cfg, num_days, start_date):
    folder = ROOT / cfg["input_files"]["input_folder"]
    dates = pd.date_range(start_date, periods=num_days, freq="D")

    h = pd.read_csv(folder / cfg["input_files"]["absolute_humidity_csv"])
    h["date"] = pd.to_datetime(h["date"], format="%m/%d/%y")
    hum = h.set_index("date")["absolute_humidity"].reindex(dates).ffill().to_numpy()

    cal = pd.read_csv(folder / cfg["input_files"]["school_work_calendar_csv"])
    cal["date"] = pd.to_datetime(cal["date"], format="%m/%d/%y")
    cal = cal.set_index("date").reindex(dates).ffill()
    is_school = cal["is_school_day"].to_numpy(float)
    is_work = cal["is_work_day"].to_numpy(float)

    vax = pd.read_csv(folder / cfg["input_files"]["vaccines_csv"])
    vax["date"] = pd.to_datetime(vax["date"], format="%m/%d/%y")
    arr = np.stack([np.asarray(ast.literal_eval(s), float).ravel() for s in vax["daily_vaccines"]])
    vdf = pd.DataFrame(arr, index=vax["date"]).reindex(dates).fillna(0.0)
    return hum, is_school, is_work, vdf.to_numpy()


def ngm_R(beta_adj, C_rows_full, susc_pop, pop, I_out_rate):
    """Dominant eigenvalue of K_ab(t) = beta_adj(t)*C_ab(t)*susc_pop_a(t)
    / (N_b * I_out_rate), given C as a (T, A, A) stack."""
    K = (beta_adj[:, None, None] * C_rows_full
         * susc_pop[:, :, None] / pop[None, None, :] / I_out_rate)
    return np.abs(np.linalg.eigvals(K)).max(axis=1)


def load_sim_susceptibles(sim_dir, num_days, n_age, n_draws):
    """(n_draws, num_days, n_age) arrays of S and SV from the baseline
    scenario of a param-set run, reindexed from parquet rep order back to
    accepted_params order. Day 0 is left as NaN (filled by the caller from
    the initial conditions). Returns None if the run isn't on disk."""
    base = Path(sim_dir) / "results_parquet/results_full/scenario=baseline"
    if not base.is_dir():
        return None
    perm = np.random.default_rng(SEED_BASE).choice(n_draws, size=n_draws, replace=False)
    out = {}
    for comp in ("S", "SV"):
        files = sorted((base / f"compartment={comp}").glob("*.parquet"))
        if not files:
            return None
        df = pd.concat([pd.read_parquet(f) for f in files])
        arr = np.full((n_draws, num_days, n_age), np.nan)
        arr[df["rep"].to_numpy(int), df["day"].to_numpy(int), df["age_group"].to_numpy(int)] = df["value"].to_numpy()
        ordered = np.full_like(arr, np.nan)
        ordered[perm] = arr                      # rep i  ->  accepted_params[perm[i]]
        out[comp] = ordered
    return out


def delivered_coverage(sim_dir, num_days, n_age, n_draws, pop):
    """(n_draws, num_days, n_age) cumulative delivered vaccination coverage,
    from the baseline scenario's S_to_SV transfers."""
    base = Path(sim_dir) / "results_parquet/results_full/scenario=baseline/compartment=S_to_SV"
    df = pd.concat([pd.read_parquet(f) for f in sorted(base.glob("*.parquet"))])
    arr = np.zeros((n_draws, num_days, n_age))
    arr[df["rep"].to_numpy(int), df["day"].to_numpy(int), df["age_group"].to_numpy(int)] = df["value"].to_numpy()
    perm = np.random.default_rng(SEED_BASE).choice(n_draws, size=n_draws, replace=False)
    ordered = np.empty_like(arr)
    ordered[perm] = arr
    return np.cumsum(ordered, axis=1) / pop[None, None, :]


def beta_eff_series(cfg, params, num_days, start_date, cov_override=None):
    p = cfg["params"]
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], float).ravel()
    w = pop / pop.sum()
    totC, schC, wrkC = (np.asarray(p[k], float) for k in
                        ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"))
    hum, is_school, is_work, vprop = load_schedules(cfg, num_days, start_date)

    if cov_override is None:
        # cumulative scheduled coverage, lagged by the dose->immunity delay
        lag = int(p["vax_transfer_delay_days"])
        cov = np.cumsum(vprop, axis=0)
        cov = np.vstack([np.zeros((lag, cov.shape[1])), cov[:-lag]])
    else:
        cov = np.asarray(cov_override, float)      # already delivered + lagged

    vs = np.asarray(p["vax_susceptibility"], float).ravel()
    rs = float(p["relative_suscept"])
    susc = (1.0 - cov) * rs + cov * vs                      # (T, A)

    # per-day contact matrix C(t), (T, A, A)
    Ct = (totC[None, :, :]
          - (1 - is_school)[:, None, None] * schC[None, :, :]
          - (1 - is_work)[:, None, None] * wrkC[None, :, :])

    beta_adj = (float(params["beta_baseline"])
                * (1.0 + float(params["humidity_impact"]) * np.exp(-180.0 * hum))
                * m_of_t(params, tv_knot_days(num_days, 14), num_days))
    susc_term = (w[None, :] * susc).sum(1)
    R_v = ngm_R(beta_adj, Ct, pop[None, :] * susc, pop, float(p["I_out_rate"]))
    return dict(beta_eff=beta_adj * susc_term, beta_adj=beta_adj,
                susc_term=susc_term, coverage=cov @ w, R_v=R_v,
                Ct=Ct, pop=pop, susc=susc)


out = {}
for label, cfg_path, fit_path, sim_dir in CASES:
    cfg = json.loads(cfg_path.read_text())
    fit = json.loads(fit_path.read_text())
    num_days = int(fit["num_days"])
    start = cfg["simulation_settings"]["start_real_date"]
    dates = pd.date_range(start, periods=num_days, freq="D")
    I_out_rate = float(cfg["params"]["I_out_rate"])
    rs = float(cfg["params"]["relative_suscept"])
    vs = np.asarray(cfg["params"]["vax_susceptibility"], float).ravel()

    n_draws = len(fit["accepted_params"])
    n_age = len(cfg["age_risk"]["age_groups"])
    cov_draws = (delivered_coverage(sim_dir, num_days, n_age, n_draws,
                                    np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"],
                                               float).ravel())
                 if label in COVERAGE_FROM_SIM else None)
    # the "best point" has no simulation of its own, so it borrows the
    # posterior-median delivered coverage
    best = beta_eff_series(cfg, fit["best_params"], num_days, start,
                           None if cov_draws is None else np.median(cov_draws, axis=0))
    draws = [beta_eff_series(cfg, q, num_days, start,
                             None if cov_draws is None else cov_draws[j])
             for j, q in enumerate(fit["accepted_params"])]
    be = np.stack([d["beta_eff"] for d in draws])
    rv = np.stack([d["R_v"] for d in draws])

    # R_eff: same NGM, but with the simulated susceptible pools of the
    # matching posterior draw in place of the no-depletion ones.
    sim = load_sim_susceptibles(sim_dir, num_days, n_age, n_draws)
    if sim is None:
        print(f"{label:16s} no param-set simulation output -- R_eff skipped")
        re_med = re_lo = re_hi = np.full(num_days, np.nan)
    else:
        # sanity-check the rep -> accepted_params pairing: day-1 susceptibles
        # are N - E0*seed_scale_E almost exactly, so this correlation is ~1
        # when the permutation is right and ~0 when it is not.
        _ss = np.array([q.get("seed_scale_E", 1.0) for q in fit["accepted_params"]])
        _r = np.corrcoef(_ss, -sim["S"][:, 1, :].sum(1))[0, 1]
        if _r < 0.99:
            print(f"  WARNING {label}: param-set pairing check failed (r={_r:.3f}) -- "
                  "R_eff draws may be mismatched to their parameters")

        S0 = best["pop"] - np.asarray(
            cfg["initial_conditions"]["aggregate_pop"]["seeds"]["E"], float).ravel() \
            * float(fit["best_params"].get("seed_scale_E", 1.0))
        re_all = np.empty((n_draws, num_days))
        for j, d in enumerate(draws):
            S, SV = sim["S"][j].copy(), np.nan_to_num(sim["SV"][j])
            S[0] = S0                                   # parquet starts at day 1
            re_all[j] = ngm_R(d["beta_adj"], d["Ct"], rs * S + vs[None, :] * SV,
                              best["pop"], I_out_rate)
        re_lo, re_med, re_hi = np.nanpercentile(re_all, [2.5, 50, 97.5], axis=0)

    out[label] = pd.DataFrame({
        "date": dates,
        "beta_eff_best": best["beta_eff"],
        "beta_eff_med": np.percentile(be, 50, axis=0),
        "beta_eff_lo": np.percentile(be, 2.5, axis=0),
        "beta_eff_hi": np.percentile(be, 97.5, axis=0),
        "R_v_best": best["R_v"],
        "R_v_med": np.percentile(rv, 50, axis=0),
        "R_v_lo": np.percentile(rv, 2.5, axis=0),
        "R_v_hi": np.percentile(rv, 97.5, axis=0),
        "R_eff_med": re_med, "R_eff_lo": re_lo, "R_eff_hi": re_hi,
        "beta_adj": best["beta_adj"],
        "susc_term": best["susc_term"],
        "coverage": best["coverage"],
    })
    print(f"{label:16s} n_draws={n_draws:4d}  beta_eff(best): "
          f"mean={best['beta_eff'].mean():.4f} min={best['beta_eff'].min():.4f} "
          f"max={best['beta_eff'].max():.4f} | R_v(best) mean={best['R_v'].mean():.2f} "
          f"peak={best['R_v'].max():.2f} | R_eff(med) peak={np.nanmax(re_med):.2f} "
          f"end={re_med[-1]:.2f}")

wide = pd.concat({k: v.set_index("date") for k, v in out.items()}, axis=1)
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MA / "effective_beta_MA_vax.csv"
wide.to_csv(csv_path)
print("wrote", csv_path)

# ---- monthly summary + figure -------------------------------------------
for col, title in [("beta_eff_best", "Monthly mean effective beta (best-fit point)"),
                   ("R_v_med", "Monthly mean R_v (posterior median; vaccination only, no infection-acquired immunity)"),
                   ("R_eff_med", "Monthly mean R_eff (posterior median; simulated susceptibles)")]:
    summary = pd.DataFrame({k: v.set_index("date")[col].resample("MS").mean()
                            for k, v in out.items()})
    summary.index = summary.index.strftime("%Y-%m")
    print(f"\n{title}:")
    print(summary.round(3).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "7-age, m(t)": "#1b6fb8",
    "7-age, no m(t)": "#8ab6dd",
    "1-age, m(t)": "#c2410c",
    "1-age, no m(t)": "#f0a888",
    "7-age, m(t), high vax": "#3f8f5a",
}
# The four current-data fits share one pair of figures; the archived high-vax
# run gets its own pair, since its vaccination schedule (and hence its
# coverage panel) is not comparable to theirs.
FIGURE_GROUPS = [
    ("", [k for k in out if k not in COVERAGE_FROM_SIM]),
    ("_high_vax", [k for k in out if k in COVERAGE_FROM_SIM]),
]


def combined_figure(labels, path, cov_label):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.4]})
    # beta_eff carries no contact term, so it has no weekday/school sawtooth
    # and is plotted daily; the R panel keeps a 7-day mean because C(t) is in
    # there.
    for label in labels:
        d = out[label].set_index("date")
        c = COLORS[label]
        axes[0].plot(d.index, d["beta_eff_best"], label=label, color=c, lw=2.0)
        axes[0].fill_between(d.index, d["beta_eff_lo"], d["beta_eff_hi"],
                             color=c, alpha=0.13, lw=0)
        # both lines are posterior medians, so the gap between them is exactly
        # what infection-acquired immunity has taken off R
        axes[1].plot(d.index, d["R_v_med"].rolling(7, center=True).mean(),
                     color=c, lw=1.3, ls="--", alpha=0.8)
        axes[1].plot(d.index, d["R_eff_med"].rolling(7, center=True).mean(),
                     label=label, color=c, lw=2.0)
        axes[1].fill_between(d.index,
                             d["R_eff_lo"].rolling(7, center=True).mean(),
                             d["R_eff_hi"].rolling(7, center=True).mean(),
                             color=c, alpha=0.13, lw=0)
        axes[2].plot(d.index, d["coverage"], color=c, lw=1.4)
    axes[0].set_ylabel("effective beta (per contact per day)")
    axes[0].set_title("Effective beta and effective R through the season\n"
                      "(beta daily; R as a 7-day rolling mean; "
                      "bands = 95% across posterior draws)")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="0.4", lw=0.8, ls=":")
    axes[1].set_ylabel("reproduction number")
    axes[1].annotate("dashed = R$_v$ (vaccination only, no infection-acquired immunity)\n"
                     "solid = R$_{eff}$ (simulated susceptibles)",
                     xy=(0.015, 0.05), xycoords="axes fraction", fontsize=9, color="0.3")
    axes[2].set_ylabel(f"pop. vaccination coverage\n({cov_label})")
    axes[2].set_xlabel("date")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def reff_figure(labels, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label in labels:
        d = out[label].set_index("date")
        ax.plot(d.index, d["R_eff_med"].rolling(7, center=True).mean(),
                label=label, color=COLORS[label], lw=2.0)
        ax.fill_between(d.index,
                        d["R_eff_lo"].rolling(7, center=True).mean(),
                        d["R_eff_hi"].rolling(7, center=True).mean(),
                        color=COLORS[label], alpha=0.13, lw=0)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_ylabel("effective reproduction number R$_{eff}$")
    ax.set_xlabel("date")
    ax.set_title("Effective reproduction number through the season\n"
                 "(7-day rolling mean of the posterior median; "
                 "bands = 95% across posterior draws)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


for suffix, labels in FIGURE_GROUPS:
    if not labels:
        continue
    cov_label = ("delivered, from S_to_SV" if suffix else "scheduled, lagged 14d")
    stem = csv_path.stem + suffix
    combined_figure(labels, csv_path.with_name(stem + ".png"), cov_label)
    reff_figure(labels, csv_path.with_name(stem.replace("effective_beta", "effective_Reff") + ".png"))
