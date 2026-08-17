"""One-off script: builds the simulation-derived assets (CSVs + PNGs) for
report.md in this folder.

Two things this computes that aren't already saved anywhere else:

1. Posterior-uncertainty baseline fit check -- runs the *baseline
   (fitted-vaccination) scenario once per posterior parameter set in
   `fitted_params.json["accepted_params"]` (638 deterministic sims), instead
   of the single best-point deterministic run
   `counterfactual_notebook_generic.py`'s "Baseline fit check" section uses,
   to get a median + 95% CI band -- same style as the Fitting tab
   (`_nb_fitting.py`: steelblue fill_between at 2.5/97.5 percentiles).
2. Daily new hospitalizations (I_to_H + IV_to_H), baseline vs. no-vaccination
   (best-point deterministic, single run each -- no posterior variation
   needed for this comparison).

Run from repo root:
    python generic_core/examples/MA_vax/build_report_assets.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import counterfactual_generic as cf
from MA_vax.model import DATA_FOLDER

OUT_DIR = os.path.join(HERE, "report_assets")
os.makedirs(OUT_DIR, exist_ok=True)

BAND_COLOR = "steelblue"
NO_VAX_COLOR = "firebrick"
RAW_COLOR = "black"


def _load_raw_age_hospitalizations():
    raw = pd.read_csv(
        os.path.join(DATA_FOLDER, "data", "hospitalizations_ts", "MA_flu_daily_hospitalizations.csv")
    )
    raw["date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("date").drop(columns=["Date"])
    raw.columns = cf.AGE_GROUPS
    return raw


def _posterior_param_sets():
    """Yield (param_overrides, seed_scales, tv_increments) for every accepted
    posterior draw, applying the same scale-group expansion
    `load_base_inputs` applies to the single best-point estimate (see
    counterfactual_generic.load_base_inputs / generic_core.fitting.prepare_param_sets)."""
    from generic_core.fitting import prepare_param_sets

    with open(os.path.join(HERE, cf.FITTED_PARAMS_FILE)) as f:
        fitted_raw = json.load(f)
    accepted = fitted_raw.get("accepted_params") or []
    scale_groups = fitted_raw.get("scale_groups", {}) or {}

    # Original (pre-fit) config params, needed by prepare_param_sets as the
    # baseline the scale-group multipliers scale -- reload fresh since
    # base_inputs["config_dict"]["params"] has already been overwritten with
    # the best-point fitted values by load_base_inputs.
    with open(os.path.join(HERE, cf.MODEL_CONFIG_FILE)) as f:
        orig_config = json.load(f)
    orig_params = dict(orig_config.get("params", {}) or {})

    expanded_sets = prepare_param_sets(accepted, scale_groups, orig_params)
    for expanded in expanded_sets:
        yield cf._split_pset(expanded)


def run_posterior_baseline_bands(base_inputs):
    """Returns (dates, new_H_by_age) where new_H_by_age has shape
    (n_posterior_draws, num_days, num_age_groups) -- new_H = I_to_H + IV_to_H,
    one deterministic simulation per posterior draw, baseline scenario."""
    param_sets = list(_posterior_param_sets())
    n = len(param_sets)
    print(f"Running {n} posterior baseline simulations...")
    A = cf.NUM_AGE_GROUPS
    out = np.zeros((n, cf.NUM_DAYS, A))
    dates = None
    for i, (model_params, seed_scales, tv_increments) in enumerate(param_sets):
        bi = dict(base_inputs)
        bi["seed_scales"] = seed_scales
        bi["tv_increments"] = tv_increments
        m = cf.build_model(bi, param_overrides=model_params, rng_seed=cf.SEED_BASE, stochastic=False)
        m.simulate_until_day(cf.NUM_DAYS)
        subpop = list(m.subpop_models.values())[0]
        h = cf._extract_age_arrays(subpop)
        out[i] = (h["I_to_H"] + h["IV_to_H"])[: cf.NUM_DAYS]
        if dates is None:
            dates = pd.date_range(start=cf.START_DATE, periods=cf.NUM_DAYS, freq="D")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")
    assert dates is not None, "no posterior parameter sets found in fitted_params.json"
    return dates, out


def main():
    base_inputs = cf.load_base_inputs()
    raw_age_H = _load_raw_age_hospitalizations()

    # ---- 1. Posterior baseline bands ----
    cache = os.path.join(OUT_DIR, "posterior_baseline_new_H.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        dates, new_H = pd.to_datetime(z["dates"]), z["new_H"]
    else:
        dates, new_H = run_posterior_baseline_bands(base_inputs)
        np.savez(cache, dates=dates.astype(str), new_H=new_H)

    sim_dates = pd.DatetimeIndex(dates)
    common_dates = sim_dates.intersection(raw_age_H.index)
    common_idx = sim_dates.get_indexer(common_dates)

    med = np.median(new_H, axis=0)
    lo = np.percentile(new_H, 2.5, axis=0)
    hi = np.percentile(new_H, 97.5, axis=0)

    # cumulative hospitalizations table (median sim vs raw, over common dates)
    sim_cum_med = med[common_idx].sum(axis=0)
    sim_cum_lo = lo[common_idx].sum(axis=0)
    sim_cum_hi = hi[common_idx].sum(axis=0)
    raw_cum = raw_age_H.loc[common_dates].sum()

    cum_table = pd.DataFrame({
        "simulated_median": sim_cum_med,
        "simulated_95pct_lo": sim_cum_lo,
        "simulated_95pct_hi": sim_cum_hi,
        "raw_data": raw_cum.to_numpy(),
    }, index=cf.AGE_GROUPS).round(1)
    cum_table["pct_diff_median"] = (
        (cum_table["simulated_median"] - cum_table["raw_data"]) / cum_table["raw_data"] * 100
    ).round(1)
    cum_table.loc["All"] = [
        round(sim_cum_med.sum(), 1), round(sim_cum_lo.sum(), 1), round(sim_cum_hi.sum(), 1),
        round(raw_cum.sum(), 1),
        round((sim_cum_med.sum() - raw_cum.sum()) / raw_cum.sum() * 100, 1),
    ]
    cum_table.index.name = "age_group"
    cum_table.to_csv(os.path.join(OUT_DIR, "cumulative_hospitalizations_by_age.csv"))
    print(cum_table)

    # daily plot, by age + total, band + median + raw
    def _band_plot(cumulative: bool, fname: str, title: str):
        fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
        axes = axes.flatten()
        for i, age in enumerate(cf.AGE_GROUPS):
            ax = axes[i]
            m_, l_, h_ = med[:, i], lo[:, i], hi[:, i]
            if cumulative:
                m_, l_, h_ = m_.cumsum(), l_.cumsum(), h_.cumsum()
            ax.fill_between(sim_dates, l_, h_, color=BAND_COLOR, alpha=0.25, label="95% CI (posterior)")
            ax.plot(sim_dates, m_, color=BAND_COLOR, linewidth=1.5, label="Median (posterior)")
            raw_series = raw_age_H[age].reindex(sim_dates)
            if cumulative:
                raw_series = raw_series.reindex(common_dates).cumsum().reindex(sim_dates)
            ax.plot(raw_series.index, raw_series, color=RAW_COLOR, linewidth=1, label="Raw data")
            ax.set_title(age)
            ax.grid(True, alpha=0.3)
        m_all, l_all, h_all = med.sum(axis=1), lo.sum(axis=1), hi.sum(axis=1)
        if cumulative:
            m_all, l_all, h_all = m_all.cumsum(), l_all.cumsum(), h_all.cumsum()
        ax_all = axes[-1]
        ax_all.fill_between(sim_dates, l_all, h_all, color=BAND_COLOR, alpha=0.25, label="95% CI (posterior)")
        ax_all.plot(sim_dates, m_all, color=BAND_COLOR, linewidth=1.5, label="Median (posterior)")
        raw_all = raw_age_H.sum(axis=1).reindex(sim_dates)
        if cumulative:
            raw_all = raw_all.reindex(common_dates).cumsum().reindex(sim_dates)
        ax_all.plot(raw_all.index, raw_all, color=RAW_COLOR, linewidth=1, label="Raw data")
        ax_all.set_title("All ages combined")
        ax_all.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(title)
        fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close(fig)

    _band_plot(False, "fit_check_daily_by_age.png",
               "Daily new hospitalizations: posterior median + 95% CI vs. raw data, by age group")
    _band_plot(True, "fit_check_cumulative_by_age.png",
               "Cumulative hospitalizations: posterior median + 95% CI vs. raw data, by age group")

    # ---- 2. Baseline vs no-vaccination (best point, deterministic) ----
    ds_base = cf._run_reps(base_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False)
    ds_novax = cf._run_reps(base_inputs, cf.no_vaccine_scenario(), n_reps=1, seed=0, stochastic=False)
    base_new_H = (ds_base["I_to_H"] + ds_base["IV_to_H"]).isel(replication=0).sum(dim="age_group").to_numpy()
    novax_new_H = (ds_novax["I_to_H"] + ds_novax["IV_to_H"]).isel(replication=0).sum(dim="age_group").to_numpy()
    plot_dates = pd.to_datetime(ds_base["day"].to_numpy())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(plot_dates, novax_new_H, color=NO_VAX_COLOR, linewidth=1.5, label="No vaccination")
    ax.plot(plot_dates, base_new_H, color=BAND_COLOR, linewidth=1.5, label="Baseline (fitted vaccination)")
    ax.set_ylabel("New hospitalizations / day (all ages)")
    ax.set_xlabel("date")
    ax.set_title("Daily new hospitalizations: baseline vs. no vaccination")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "baseline_vs_no_vaccination_daily_H.png"), dpi=150)
    plt.close(fig)

    print("Done. Assets written to", OUT_DIR)


if __name__ == "__main__":
    main()
