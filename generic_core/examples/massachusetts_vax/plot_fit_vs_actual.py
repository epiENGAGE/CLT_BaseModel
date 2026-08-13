"""Compare simulated (fitted) hospitalization trajectories against observed data.

Reads:
  - a `*_metric_timeseries.csv` produced by the model-builder notebook's
    analysis/export step (median + 95% CI, daily and cumulative, per age
    group and "all ages"), e.g. MA_fitted_14d_tv__metric_timeseries.csv
  - the per-age-group and total observed hospitalization CSVs
    (MA_flu_daily_hospitalizations_<age>.csv / _total.csv)

Produces:
  - a PNG with one row per age group (+ total), daily on the left and
    cumulative on the right: solid blue median line, blue CI band, black
    dots for observed data.
  - a CSV table comparing cumulative totals (simulated median + CI vs.
    observed) at the end of the simulated window, per age group and total.

To point this at a different simulation run, edit SIM_TIMESERIES_CSV (or pass
--sim-csv) — everything else is derived from it and from the fixed observed
data files in this directory.

Run:
    uv run python generic_core/examples/massachusetts_vax/plot_fit_vs_actual.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent

SIM_TIMESERIES_CSV = HERE / "MA_fitted_14d_tv__metric_timeseries.csv"
OUTPUT_DIR = HERE / "fit_comparison_output"

# (age_group label in the simulation CSV, observed-data file suffix, display name)
AGE_GROUPS = [
    ("Age 0", "0_0", "0"),
    ("Age 1", "1_4", "1-4"),
    ("Age 2", "5_12", "5-12"),
    ("Age 3", "13_17", "13-17"),
    ("Age 4", "18_49", "18-49"),
    ("Age 5", "50_64", "50-64"),
    ("Age 6", "65plus", "65+"),
]
TOTAL_LABEL = ("all ages", "total", "Total")

MEDIAN_COLOR = "#1a56db"
BAND_COLOR = "#1a56db"
BAND_ALPHA = 0.2
ACTUAL_COLOR = "#111827"


def load_simulation(sim_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sim_csv, parse_dates=["date"])
    return df[df["metric"] == "new_H"]


def load_observed_age(suffix: str) -> pd.DataFrame:
    path = HERE / "data" / "hospitalizations_ts" / f"MA_flu_daily_hospitalizations_{suffix}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df.rename(columns={"value": "observed"})


def load_observed_total() -> pd.DataFrame:
    path = HERE / "data" / "hospitalizations_ts" / "MA_flu_daily_hospitalizations_total.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df[["date", "total"]].rename(columns={"total": "observed"})


def restrict_to_sim_window(observed: pd.DataFrame, sim_dates: pd.Series) -> pd.DataFrame:
    start, end = sim_dates.min(), sim_dates.max()
    observed = observed[(observed["date"] >= start) & (observed["date"] <= end)].copy()
    observed["observed_cumulative"] = observed["observed"].cumsum()
    return observed


def plot_group(ax_daily, ax_cum, sim_group: pd.DataFrame, observed: pd.DataFrame, title: str) -> None:
    daily = sim_group[sim_group["series"] == "daily"].sort_values("date")
    cumulative = sim_group[sim_group["series"] == "cumulative"].sort_values("date")

    for ax, sim_df, obs_col in (
        (ax_daily, daily, "observed"),
        (ax_cum, cumulative, "observed_cumulative"),
    ):
        ax.fill_between(
            sim_df["date"], sim_df["ci_lower_2.5"], sim_df["ci_upper_97.5"],
            color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0, label="Simulated 95% CI",
        )
        ax.plot(sim_df["date"], sim_df["median"], color=MEDIAN_COLOR, linewidth=2, label="Simulated median")
        ax.scatter(
            observed["date"], observed[obs_col], color=ACTUAL_COLOR, s=10, zorder=3, label="Observed",
        )
        ax.tick_params(axis="x", rotation=45)

    ax_daily.set_ylabel(title, fontsize=9, rotation=0, ha="right", va="center", labelpad=25)


def make_figure(sim: pd.DataFrame, observed_by_group: dict[str, pd.DataFrame], output_path: Path) -> None:
    groups = AGE_GROUPS + [TOTAL_LABEL]
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 3 * len(groups)), sharex=True)

    for row, (sim_label, suffix, display_name) in enumerate(groups):
        sim_group = sim[sim["age_group"] == sim_label]
        observed = observed_by_group[suffix]
        plot_group(axes[row, 0], axes[row, 1], sim_group, observed, display_name)

    axes[0, 0].set_title("Daily")
    axes[0, 1].set_title("Cumulative")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_table(sim: pd.DataFrame, observed_by_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sim_label, suffix, display_name in AGE_GROUPS + [TOTAL_LABEL]:
        sim_final = sim[(sim["age_group"] == sim_label) & (sim["series"] == "cumulative")].sort_values("date").iloc[-1]
        observed = observed_by_group[suffix]
        observed_final = observed["observed_cumulative"].iloc[-1] if len(observed) else float("nan")
        rows.append({
            "age_group": display_name,
            "simulated_median": sim_final["median"],
            "simulated_ci_lower": sim_final["ci_lower_2.5"],
            "simulated_ci_upper": sim_final["ci_upper_97.5"],
            "observed": observed_final,
            "difference": sim_final["median"] - observed_final,
            "pct_difference": 100 * (sim_final["median"] - observed_final) / observed_final,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-csv", type=Path, default=SIM_TIMESERIES_CSV, help="Simulated metric timeseries CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to write outputs into")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = load_simulation(args.sim_csv)
    sim_dates = sim["date"]

    observed_by_group = {}
    for _, suffix, _ in AGE_GROUPS:
        observed_by_group[suffix] = restrict_to_sim_window(load_observed_age(suffix), sim_dates)
    observed_by_group["total"] = restrict_to_sim_window(load_observed_total(), sim_dates)

    figure_path = args.output_dir / "fit_vs_actual.png"
    make_figure(sim, observed_by_group, figure_path)
    print(f"wrote {figure_path}")

    table = make_table(sim, observed_by_group)
    table_path = args.output_dir / "cumulative_comparison.csv"
    table.to_csv(table_path, index=False)
    print(f"wrote {table_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
