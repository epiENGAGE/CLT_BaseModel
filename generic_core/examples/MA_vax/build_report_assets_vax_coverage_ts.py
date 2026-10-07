"""One-off script: builds the vaccination-coverage time-series assets
(CSV + 2 PNGs) for report.md's dose-accounting appendix.

Proportion of each age group's population vaccinated per day, and the
cumulative version of the same series, straight from the vaccination
schedule's INPUT (`daily_vaccines_df`, the Model Builder's per-age-group
daily vaccination proportion) -- not adjusted for the §1.4 cap that
declines to deliver a scheduled dose to someone already infected. That
adjustment is what the report's "scheduled vs. delivered" dose-accounting
table/discussion is about; this script deliberately uses the raw
scheduled input, matching `counterfactual_generic.scheduled_coverage`'s
per-day values (whose season-end cumulative sum reproduces the §2.5 table).

Run from repo root:
    python generic_core/examples/MA_vax/build_report_assets_vax_coverage_ts.py
"""
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import counterfactual_generic as cf

OUT_DIR = os.path.join(HERE, "report_assets")
os.makedirs(OUT_DIR, exist_ok=True)


def _daily_schedule_array() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """(num_days, num_age_groups) proportion-of-population-vaccinated array,
    over the simulation window -- same windowing/parsing as
    counterfactual_generic.scheduled_coverage, but keeping every day instead
    of summing them away."""
    csvs = cf._load_schedule_csvs(model_config_file="model_config_MA_vax.json")
    if "daily_vaccines_df" not in csvs:
        raise ValueError("no daily_vaccines_df schedule found")
    df = pd.read_csv(io.StringIO(csvs["daily_vaccines_df"]))
    dates = pd.to_datetime(df["date"], format="mixed")
    start = pd.Timestamp(cf.START_DATE)
    window = (dates >= start) & (dates < start + pd.Timedelta(days=cf.NUM_DAYS))
    sub = df.loc[window].copy()
    sub["_date"] = dates.loc[window]
    sub = sub.sort_values("_date")
    daily = np.array([np.asarray(json.loads(v), dtype=float).ravel()
                       for v in sub["daily_vaccines"]])
    return daily, pd.DatetimeIndex(sub["_date"])


def main():
    base_inputs = cf.load_base_inputs(
        model_config_file="model_config_MA_vax.json", fitted_params_file="fitted_params_MA_vax.json")
    population = base_inputs["population"]  # (num_age_groups,)
    daily, dates = _daily_schedule_array()  # (num_days, num_age_groups)
    assert daily.shape[0] == cf.NUM_DAYS, f"expected {cf.NUM_DAYS} days, got {daily.shape[0]}"

    cum = daily.cumsum(axis=0)

    # Population-weighted "All ages" aggregate: proportion of the TOTAL
    # population vaccinated, not an unweighted average across age groups.
    total_pop = population.sum()
    daily_all = (daily * population).sum(axis=1) / total_pop
    cum_all = (cum * population).sum(axis=1) / total_pop

    # ---- CSV, long format: date, age_group, daily_proportion, cumulative_proportion ----
    rows = []
    for i, age in enumerate(cf.AGE_GROUPS):
        for d, day_val, cum_val in zip(dates, daily[:, i], cum[:, i]):
            rows.append({"date": d.date(), "age_group": age,
                         "daily_proportion": day_val, "cumulative_proportion": cum_val})
    for d, day_val, cum_val in zip(dates, daily_all, cum_all):
        rows.append({"date": d.date(), "age_group": "All",
                     "daily_proportion": day_val, "cumulative_proportion": cum_val})
    out_csv = os.path.join(OUT_DIR, "vax_coverage_timeseries.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # ---- Plots ----
    labels = list(cf.AGE_GROUPS) + ["All"]
    series_daily = list(daily.T) + [daily_all]
    series_cum = list(cum.T) + [cum_all]
    colors = list(cm.tab10.colors[:7]) + ["black"]

    def _plot(series_list, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, series, color in zip(labels, series_list, colors):
            lw = 2.2 if label == "All" else 1.4
            ls = "--" if label == "All" else "-"
            ax.plot(dates, series * 100, label=label, color=color, linewidth=lw, linestyle=ls)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("date")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Age group", loc="upper left", fontsize=9)
        fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
        plt.close(fig)

    _plot(series_daily, "Proportion of population vaccinated / day (%)",
          "Daily scheduled vaccination proportion, by age group",
          "vax_coverage_daily_proportion.png")
    _plot(series_cum, "Cumulative proportion of population vaccinated (%)",
          "Cumulative scheduled vaccination coverage, by age group",
          "vax_coverage_cumulative_proportion.png")

    print("Final cumulative coverage (should match §2.5 table):")
    for label, series in zip(labels, series_cum):
        print(f"  {label}: {series[-1] * 100:.1f}%")

    print("Done. Assets written to", OUT_DIR)


if __name__ == "__main__":
    main()
