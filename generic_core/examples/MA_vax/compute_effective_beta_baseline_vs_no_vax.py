"""Effective beta and effective R for the 7-age m(t) MA_vax fit: baseline
scenario vs the "no vax" scenario of the same param-set run.

Same quantities and figure layout as compute_effective_beta.py (see its
docstring for the definitions of beta_eff, R_v and R_eff). The "no vax"
scenario in run_simulations_MA_vax_param_set_stochastic.py keeps every
sampled parameter (DESIGNED_PARAMS['no vax'] == []) and only zeroes the dose
schedule (DOSE_MULTIPLIER), so here it is the same posterior draws with
coverage v_a(t) = 0 and S/SV taken from its own parquet partition.
"""
import ast, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/rfp437/Work/CityLevelTransmission/CLT_BaseModel")
MA = ROOT / "generic_core/examples/MA_vax"
CFG_PATH = MA / "model_config_MA_vax.json"
FIT_PATH = MA / "fitted_params_MA_vax.json"
SIM_DIR = MA / "simulation_output_MA_vax_param_set_stochastic"

# (label, parquet scenario partition, vaccinate?)
SCENARIOS = [
    ("baseline", "baseline", True),
    ("no vax", "no%20vax", False),
]

# run_simulations_MA_vax_param_set_stochastic.py draws its param sets as
# default_rng(SEED_BASE).choice(n, size=n, replace=False), so parquet rep i is
# accepted_params[SEED_PERM[i]].
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
    K = (beta_adj[:, None, None] * C_rows_full
         * susc_pop[:, :, None] / pop[None, None, :] / I_out_rate)
    return np.abs(np.linalg.eigvals(K)).max(axis=1)


def load_sim_susceptibles(scenario_dir, num_days, n_age, n_draws):
    base = SIM_DIR / "results_parquet/results_full" / f"scenario={scenario_dir}"
    perm = np.random.default_rng(SEED_BASE).choice(n_draws, size=n_draws, replace=False)
    out = {}
    for comp in ("S", "SV"):
        files = sorted((base / f"compartment={comp}").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {comp} parquet under {base}")
        df = pd.concat([pd.read_parquet(f) for f in files])
        arr = np.full((n_draws, num_days, n_age), np.nan)
        arr[df["rep"].to_numpy(int), df["day"].to_numpy(int), df["age_group"].to_numpy(int)] = df["value"].to_numpy()
        ordered = np.full_like(arr, np.nan)
        ordered[perm] = arr
        out[comp] = ordered
    return out


def beta_eff_series(cfg, params, num_days, start_date, vaccinate):
    p = cfg["params"]
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], float).ravel()
    w = pop / pop.sum()
    totC, schC, wrkC = (np.asarray(p[k], float) for k in
                        ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"))
    hum, is_school, is_work, vprop = load_schedules(cfg, num_days, start_date)

    lag = int(p["vax_transfer_delay_days"])
    cov = np.cumsum(vprop, axis=0)
    cov = np.vstack([np.zeros((lag, cov.shape[1])), cov[:-lag]])
    if not vaccinate:
        cov = np.zeros_like(cov)

    vs = np.asarray(p["vax_susceptibility"], float).ravel()
    rs = float(p["relative_suscept"])
    susc = (1.0 - cov) * rs + cov * vs

    Ct = (totC[None, :, :]
          - (1 - is_school)[:, None, None] * schC[None, :, :]
          - (1 - is_work)[:, None, None] * wrkC[None, :, :])

    beta_adj = (float(params["beta_baseline"])
                * (1.0 + float(params["humidity_impact"]) * np.exp(-180.0 * hum))
                * m_of_t(params, tv_knot_days(num_days, 14), num_days))
    susc_term = (w[None, :] * susc).sum(1)
    R_v = ngm_R(beta_adj, Ct, pop[None, :] * susc, pop, float(p["I_out_rate"]))
    return dict(beta_eff=beta_adj * susc_term, beta_adj=beta_adj,
                susc_term=susc_term, coverage=cov @ w, R_v=R_v, Ct=Ct, pop=pop)


cfg = json.loads(CFG_PATH.read_text())
fit = json.loads(FIT_PATH.read_text())
num_days = int(fit["num_days"])
start = cfg["simulation_settings"]["start_real_date"]
dates = pd.date_range(start, periods=num_days, freq="D")
I_out_rate = float(cfg["params"]["I_out_rate"])
rs = float(cfg["params"]["relative_suscept"])
vs = np.asarray(cfg["params"]["vax_susceptibility"], float).ravel()
n_draws = len(fit["accepted_params"])
n_age = len(cfg["age_risk"]["age_groups"])

out = {}
for label, scen_dir, vaccinate in SCENARIOS:
    best = beta_eff_series(cfg, fit["best_params"], num_days, start, vaccinate)
    draws = [beta_eff_series(cfg, q, num_days, start, vaccinate) for q in fit["accepted_params"]]
    be = np.stack([d["beta_eff"] for d in draws])
    rv = np.stack([d["R_v"] for d in draws])

    sim = load_sim_susceptibles(scen_dir, num_days, n_age, n_draws)
    _ss = np.array([q.get("seed_scale_E", 1.0) for q in fit["accepted_params"]])
    _r = np.corrcoef(_ss, -sim["S"][:, 1, :].sum(1))[0, 1]
    if _r < 0.99:
        print(f"  WARNING {label}: param-set pairing check failed (r={_r:.3f})")

    S0 = best["pop"] - np.asarray(
        cfg["initial_conditions"]["aggregate_pop"]["seeds"]["E"], float).ravel() \
        * float(fit["best_params"].get("seed_scale_E", 1.0))
    re_all = np.empty((n_draws, num_days))
    for j, d in enumerate(draws):
        S, SV = sim["S"][j].copy(), np.nan_to_num(sim["SV"][j])
        S[0] = S0
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
        "coverage": best["coverage"],
    })
    print(f"{label:9s} n_draws={n_draws}  beta_eff(best) mean={best['beta_eff'].mean():.4f} | "
          f"R_v(med) peak={np.percentile(rv, 50, axis=0).max():.2f} | "
          f"R_eff(med) peak={np.nanmax(re_med):.2f} end={re_med[-1]:.2f}")

wide = pd.concat({k: v.set_index("date") for k, v in out.items()}, axis=1)
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MA / "effective_beta_MA_vax_baseline_vs_no_vax.csv"
wide.to_csv(csv_path)
print("wrote", csv_path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"baseline": "#1b6fb8", "no vax": "#c2410c"}

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 1.4]})
for label in out:
    d = out[label].set_index("date")
    c = COLORS[label]
    axes[0].plot(d.index, d["beta_eff_best"], label=label, color=c, lw=2.0)
    axes[0].fill_between(d.index, d["beta_eff_lo"], d["beta_eff_hi"], color=c, alpha=0.13, lw=0)
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
axes[0].set_title("Effective beta and effective R through the season -- 7-age m(t) fit, "
                  "baseline vs no vaccination\n"
                  "(beta daily; R as a 7-day rolling mean; bands = 95% across posterior draws)")
axes[0].legend(frameon=False)
axes[1].axhline(1.0, color="0.4", lw=0.8, ls=":")
axes[1].set_ylabel("reproduction number")
axes[1].annotate("dashed = R$_v$ (vaccination only, no infection-acquired immunity)\n"
                 "solid = R$_{eff}$ (simulated susceptibles)",
                 xy=(0.015, 0.05), xycoords="axes fraction", fontsize=9, color="0.3")
axes[2].set_ylabel("pop. vaccination coverage\n(scheduled, lagged 14d)")
axes[2].set_xlabel("date")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
png_path = csv_path.with_suffix(".png")
fig.savefig(png_path, dpi=150)
plt.close(fig)
print("wrote", png_path)
