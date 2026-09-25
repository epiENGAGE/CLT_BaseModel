"""Simulation-derived assets (CSVs + PNGs) for report_<TAG>.md, where TAG is
MA_vax_old_vax_initial_R_20pct (default) or, with --tag, another variant with
the same file naming (e.g. MA_vax_old_vax_initial_R_20pct_vax_total_pop).

Unlike ../../build_report_assets.py (which re-simulates through
counterfactual_generic.py), everything here is read from the param-set
simulation output of run_simulations_MA_vax_old_vax_initial_R_20pct_param_set_stochastic.py
-- the same 638 runs per scenario the counterfactual tables are built from --
so the fit check, the baseline-vs-no-vax figure and the tables all describe
one set of simulations.

Writes to report_assets_<TAG>/:
  fitted_params_summary.csv             posterior mean / 5% / 95% / best point
  vaccination_coverage.csv              scheduled cumulative coverage by age
  cumulative_hospitalizations_by_age.csv  simulated (per-draw cumulative) vs raw
  fit_check_daily_by_age.png / fit_check_cumulative_by_age.png
  baseline_vs_no_vaccination_daily_H.png  (+ peak summary printed)

Run from anywhere:
    python generic_core/examples/MA_vax/archive/2026-08-high-vax-rates/build_report_assets_MA_vax_old_vax_initial_R_20pct.py [--tag TAG]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
MA = HERE.parent.parent
sys.path.insert(0, str(MA))
import counterfactual_generic as cf  # noqa: E402
from generic_core import results_io  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", default="MA_vax_old_vax_initial_R_20pct")
TAG = _ap.parse_args().tag
MODEL_CONFIG = HERE / f"model_config_{TAG}.json"
FITTED_PARAMS = HERE / f"fitted_params_{TAG}.json"
DB = HERE / f"simulation_output_{TAG}_param_set_stochastic" / "results_parquet"
RAW_H = MA / "data" / "hospitalizations_ts" / "MA_flu_daily_hospitalizations.csv"
OUT = HERE / f"report_assets_{TAG}"
AGES = cf.AGE_GROUPS

BAND_COLOR = "steelblue"
NO_VAX_COLOR = "firebrick"
RAW_COLOR = "black"


def new_H_cube(con, scenario: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(dates, (reps, days, ages)) of daily I_to_H + IV_to_H."""
    df = con.execute(
        "SELECT rep, day, age_group, SUM(value) AS value FROM results_full "
        "WHERE scenario = ? AND compartment IN ('I_to_H', 'IV_to_H') "
        "GROUP BY rep, day, age_group", [scenario]).df()
    reps, days = sorted(df.rep.unique()), sorted(df.day.unique())
    arr = np.zeros((len(reps), len(days), len(AGES)))
    arr[df.rep.map({r: i for i, r in enumerate(reps)}).to_numpy(),
        df.day.map({d: i for i, d in enumerate(days)}).to_numpy(),
        df.age_group.to_numpy()] = df.value.to_numpy()
    dates = pd.date_range(cf.START_DATE, periods=len(days), freq="D")
    return dates, arr


def fitted_params_summary(fit: dict) -> pd.DataFrame:
    draws = fit["accepted_params"]
    best = fit["best_params"]
    keys = [k for k in draws[0] if not k.startswith("m_dlog_")]
    rows = []
    for k in keys:
        v = np.array([float(q[k]) for q in draws])
        rows.append({"parameter": k, "posterior_mean": v.mean(),
                     "p05": np.percentile(v, 5), "p95": np.percentile(v, 95),
                     "best": float(best[k]) if k in best else np.nan})
    return pd.DataFrame(rows).set_index("parameter")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg = json.loads(MODEL_CONFIG.read_text())
    fit = json.loads(FITTED_PARAMS.read_text())
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], float).ravel()

    fp = fitted_params_summary(fit)
    fp.to_csv(OUT / "fitted_params_summary.csv")
    print(fp.round(4), f"\n{len(fit['accepted_params'])} posterior draws")

    cov = np.asarray(cf.scheduled_coverage(model_config_file=str(MODEL_CONFIG)), float)
    cov_df = pd.DataFrame({"population": pop, "scheduled_coverage": cov}, index=pd.Index(AGES, name="age_group"))
    cov_df.loc["All"] = [pop.sum(), (cov * pop).sum() / pop.sum()]
    cov_df.to_csv(OUT / "vaccination_coverage.csv")
    print(cov_df)

    con = results_io.load_source(str(DB))
    dates, base = new_H_cube(con, "baseline")
    _, novax = new_H_cube(con, "no vax")
    con.close()

    raw = pd.read_csv(RAW_H)
    raw["date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("date").drop(columns=["Date"])
    raw.columns = AGES
    common = dates.intersection(raw.index)
    idx = dates.get_indexer(common)

    # cumulative over the common window: quantiles of each draw's own sum
    cum = base[:, idx, :].sum(axis=1)                        # (reps, ages)
    cum = np.concatenate([cum, cum.sum(axis=1, keepdims=True)], axis=1)
    raw_cum = np.append(raw.loc[common].sum().to_numpy(), raw.loc[common].to_numpy().sum())
    med = np.median(cum, axis=0)
    tbl = pd.DataFrame({
        "simulated_median": med,
        "simulated_95pct_lo": np.percentile(cum, 2.5, axis=0),
        "simulated_95pct_hi": np.percentile(cum, 97.5, axis=0),
        "raw_data": raw_cum,
        "pct_diff_median": (med - raw_cum) / raw_cum * 100,
    }, index=pd.Index(AGES + ["All"], name="age_group")).round(1)
    tbl.to_csv(OUT / "cumulative_hospitalizations_by_age.csv")
    print(tbl, f"\ncommon window {pd.Timestamp(common.min()).date()} - {pd.Timestamp(common.max()).date()}")

    def band_plot(cumulative: bool, fname: str, title: str):
        fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
        axes = axes.flatten()
        series = [(AGES[i], base[:, :, i], raw[AGES[i]]) for i in range(len(AGES))]
        series.append(("All ages combined", base.sum(axis=2), raw.sum(axis=1)))
        for ax, (title_i, x, r) in zip(axes, series):
            if cumulative:
                x = np.cumsum(x, axis=1)
                r = r.reindex(common).cumsum()
            lo, m, hi = np.percentile(x, [2.5, 50, 97.5], axis=0)
            ax.fill_between(dates, lo, hi, color=BAND_COLOR, alpha=0.25, label="95% interval (posterior)")
            ax.plot(dates, m, color=BAND_COLOR, lw=1.5, label="Median (posterior)")
            r = r.reindex(dates)
            ax.plot(r.index, r, color=RAW_COLOR, lw=1, label="Raw data")
            ax.set_title(title_i)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(title)
        fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)

    band_plot(False, "fit_check_daily_by_age.png",
              "Daily new hospitalizations: posterior median + 95% interval vs. raw data, by age group")
    band_plot(True, "fit_check_cumulative_by_age.png",
              "Cumulative hospitalizations: posterior median + 95% interval vs. raw data, by age group")

    bt, nt = base.sum(axis=2), novax.sum(axis=2)
    b_lo, b_m, b_hi = np.percentile(bt, [2.5, 50, 97.5], axis=0)
    n_lo, n_m, n_hi = np.percentile(nt, [2.5, 50, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(dates, n_lo, n_hi, color=NO_VAX_COLOR, alpha=0.2)
    ax.plot(dates, n_m, color=NO_VAX_COLOR, lw=1.5, label="No vaccination (median, 95% interval)")
    ax.fill_between(dates, b_lo, b_hi, color=BAND_COLOR, alpha=0.2)
    ax.plot(dates, b_m, color=BAND_COLOR, lw=1.5, label="Baseline (fitted vaccination) (median, 95% interval)")
    ax.set_ylabel("New hospitalizations / day (all ages)")
    ax.set_xlabel("date")
    ax.set_title("Daily new hospitalizations: baseline vs. no vaccination")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUT / "baseline_vs_no_vaccination_daily_H.png", dpi=150)
    plt.close(fig)
    print(f"Baseline median peak: {b_m.max():.1f} on {pd.Timestamp(dates[b_m.argmax()]).date()}")
    print(f"No-vaccination median peak: {n_m.max():.1f} on {pd.Timestamp(dates[n_m.argmax()]).date()}")
    print(f"Ratio of peaks (no-vax / baseline): {n_m.max() / b_m.max():.2f}x")
    print(f"Season totals, median: baseline {np.median(bt.sum(1)):.0f}, no vax {np.median(nt.sum(1)):.0f}")
    print("Assets written to", OUT)


if __name__ == "__main__":
    main()
