"""Build every figure and figure-table for the MA_vax deck and report from the
pipeline's outputs, so a re-run (or an alternative model version) only needs:

    python generic_core/examples/MA_vax/figures/build_figures.py

Inputs (defaults are the pipeline's standard locations in MA_vax/):
  --db       simulation_output_MA_vax_param_set_stochastic/results_parquet
  --tables   counterfactual_tables_from_db_MA_vax/        (build_counterfactual_tables_from_db.py)
  --fit-csv  baseline_new_H_MA_vax.csv                    (export_metric_timeseries_MA_vax.py)
  model_config_MA_vax.json, fitted_params_MA_vax.json, data/ (observed admissions, schedules)

Everything is written to figures/output/ (or --out):

  Python (this script)
    scenario_timeseries.csv            daily new hospitalizations (I_to_H + IV_to_H), whole
                                       population, median / 2.5% / 97.5% per scenario
    attack_rates.csv                   baseline season attack rate by age group, vaccinated /
                                       unvaccinated / total (median [95% CI])
    vax_coverage_daily.png / _cumulative.png / vax_coverage_timeseries.csv
                                       scheduled vaccination by age group (model input)
    fit_check_daily_by_age.png / fit_check_cumulative_by_age.png
                                       baseline posterior band vs observed admissions
    transmission_multiplier.png / .csv m(t), humidity factor and their product (posterior)
    coverage_response.png              total hospitalizations vs overall scheduled coverage,
                                       every "(all ages)" coverage scenario
    coverage_heatmap.png               % change in total hospitalizations by age group
                                       targeted x coverage level (targets and decreases)
    coverage_by_age_targeted.csv       the numbers behind the heatmap (median [95% CI]): averted /
                                       % reduction for targets, added / % increase for dose cuts
    fit_parameters.csv / age_parameters.csv / contact_matrix.csv
                                       parameter tables for the deck / report (age_parameters:
                                       pre-fit values plus fitted hospitalization risks)
    S_A_2_pct_reduction_normalized.csv input of the S.A.2 stacked bar
  R (ggplot, deck styling -- skipped with --no-r)
    ts_baseline_infection_protection_no_vax.png, ts_ve.png,
    ts_coverage_targets.png, ts_coverage_decreases.png         plot_scenarios_ts.R
    S_A_2_pct_reduction_stacked_bar.png                        plot_pct_reduction_stacked_bar.R
    S_A_3_<T>pct_per_100k_doses_stacked_bar.png,
    DECREASE_<D>pct_per_100k_doses_stacked_bar.png             plot_per_100k_doses_stacked_bar.R
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
MA_VAX = FIG_DIR.parent
sys.path.insert(0, str(MA_VAX.parent.parent.parent))
sys.path.insert(0, str(MA_VAX))

import counterfactual_generic as cg
from generic_core import results_io
from generic_core.fitting import _tv_knot_days, build_transmission_multiplier_array, prepare_param_sets
from make_scenarios_MA_vax import COVERAGE_TARGETS, COVERAGE_DECREASES
from ma_vax_shared import AGE_GROUP_LABELS as AGE_GROUPS

MODEL_CONFIG = MA_VAX / "model_config_MA_vax.json"
FITTED_PARAMS = MA_VAX / "fitted_params_MA_vax.json"
OBS_DIR = MA_VAX / "data" / "hospitalizations_ts"
OBS_SUFFIX = ["0_0", "1_4", "5_12", "13_17", "18_49", "50_64", "65plus"]

TS_SCENARIOS = (
    ["baseline", "no vax", "Infection protection only", "Low VE", "High VE"]
    + [f"{t}% coverage (all ages)" for t in COVERAGE_TARGETS]
    + [f"{d}% lower coverage (all ages)" for d in COVERAGE_DECREASES]
)
R_TS_PLOTS = {
    "ts_baseline_infection_protection_no_vax.png": ["baseline", "Infection protection only", "no vax"],
    "ts_ve.png": ["baseline", "no vax", "High VE", "Low VE"],
    "ts_coverage_targets.png": ["baseline"] + [f"{t}% coverage (all ages)" for t in COVERAGE_TARGETS],
    "ts_coverage_decreases.png": ["baseline"] + [f"{d}% lower coverage (all ages)"
                                                 for d in COVERAGE_DECREASES],
}

BAND = "steelblue"


def _median(cell) -> float:
    """Median of a "<median> [<lo> - <hi>]" table cell ("—" -> nan)."""
    m = re.match(r"^\s*(-?[\d,]*\.?\d+)", str(cell))
    return float(m.group(1).replace(",", "")) if m else np.nan


# ── Simulation-derived data ──────────────────────────────────────────────────

def scenario_timeseries(con, start_date: str) -> pd.DataFrame:
    names = ",".join("?" * len(TS_SCENARIOS))
    df = con.execute(
        f"""WITH t AS (
                SELECT scenario, rep, day, SUM(value) AS v FROM results
                WHERE compartment IN ('I_to_H', 'IV_to_H') AND scenario IN ({names})
                GROUP BY scenario, rep, day)
            SELECT scenario, day, median(v) AS median_value,
                   quantile_cont(v, 0.025) AS p_lo, quantile_cont(v, 0.975) AS p_hi
            FROM t GROUP BY scenario, day ORDER BY scenario, day""",
        TS_SCENARIOS,
    ).df()
    # The R plot puts day 1 on the start date; results days start at 0 or 1
    # depending on the writer, so re-base to 1 here.
    df["day"] = df["day"] - df["day"].min() + 1
    df["date"] = (pd.Timestamp(start_date) + pd.to_timedelta(df["day"] - 1, unit="D")).dt.strftime("%Y-%m-%d")
    return df


# ── Vaccination schedule ─────────────────────────────────────────────────────

def vax_coverage(out: Path) -> None:
    """Scheduled daily / cumulative vaccination proportion by age group over the
    simulation window, after the model's vax_transfer_delay_days shift (same
    windowing as counterfactual_generic.scheduled_coverage)."""
    csvs = cg._load_schedule_csvs(model_config_file=str(MODEL_CONFIG))
    cfg = json.loads(MODEL_CONFIG.read_text())
    delay = int(cfg["params"].get("vax_transfer_delay_days", 0) or 0)
    df = pd.read_csv(io.StringIO(csvs["daily_vaccines_df"]))
    dates = pd.to_datetime(df["date"], format="mixed") + pd.Timedelta(days=delay)
    start = pd.Timestamp(cg.START_DATE)
    window = (dates >= start) & (dates < start + pd.Timedelta(days=cg.NUM_DAYS))
    daily = np.array([np.asarray(json.loads(v), dtype=float).ravel() for v in df.loc[window, "daily_vaccines"]])
    dates = dates[window]
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], dtype=float).ravel()
    all_daily = daily @ pop / pop.sum()
    cols = {**{a: daily[:, i] for i, a in enumerate(AGE_GROUPS)}, "All": all_daily}

    rows = []
    for label, d in cols.items():
        rows.append(pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d").to_numpy(), "age_group": label,
                                  "daily_proportion": d, "cumulative_proportion": np.cumsum(d)}))
    pd.concat(rows).to_csv(out / "vax_coverage_timeseries.csv", index=False)

    colors = [plt.get_cmap("tab10")(k) for k in range(10)]
    for kind, fname, ylabel, title in (
        ("daily", "vax_coverage_daily.png", "Proportion of population vaccinated / day (%)",
         "Daily scheduled vaccination proportion, by age group"),
        ("cumulative", "vax_coverage_cumulative.png", "Cumulative proportion of population vaccinated (%)",
         "Cumulative scheduled vaccination coverage, by age group"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for k, (label, d) in enumerate(cols.items()):
            y = (np.cumsum(d) if kind == "cumulative" else d) * 100
            ax.plot(dates, y, label=label, color="black" if label == "All" else colors[k],
                    ls="--" if label == "All" else "-", lw=2 if label == "All" else 1.4,
                    drawstyle="steps-post" if kind == "daily" else "default")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(title="Age group", fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


# ── Fit check ────────────────────────────────────────────────────────────────

def fit_check(fit_csv: Path, out: Path) -> None:
    sim = pd.read_csv(fit_csv, parse_dates=["date"])
    sim = sim[sim["metric"] == "new_H"]
    start, end = sim["date"].min(), sim["date"].max()
    obs = {}
    for i, suffix in enumerate(OBS_SUFFIX):
        o = pd.read_csv(OBS_DIR / f"MA_flu_daily_hospitalizations_{suffix}.csv", parse_dates=["date"])
        obs[f"Age {i}"] = o[(o["date"] >= start) & (o["date"] <= end)].set_index("date")["value"]
    obs["all ages"] = sum(obs.values())
    titles = {**{f"Age {i}": a for i, a in enumerate(AGE_GROUPS)}, "all ages": "All ages combined"}

    for series, fname, sup in (
        ("daily", "fit_check_daily_by_age.png", "Daily new hospitalizations"),
        ("cumulative", "fit_check_cumulative_by_age.png", "Cumulative hospitalizations"),
    ):
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
        for ax, (key, title) in zip(axes.ravel(), titles.items()):
            s = sim[(sim["age_group"] == key) & (sim["series"] == series)].sort_values("date")
            o = obs[key].cumsum() if series == "cumulative" else obs[key]
            ax.fill_between(s["date"], s["ci_lower_2.5"], s["ci_upper_97.5"], color=BAND, alpha=0.25,
                            lw=0, label="95% CI (posterior)")
            ax.plot(s["date"], s["median"], color=BAND, lw=1.8, label="Median (posterior)")
            ax.plot(o.index, o.to_numpy(), color="black", lw=1, label="Observed")
            ax.set_title(title, fontsize=11)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{sup}: posterior median + 95% CI vs. observed, by age group")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


# ── m(t) and humidity ────────────────────────────────────────────────────────

def transmission_multiplier(out: Path) -> None:
    cfg = json.loads(MODEL_CONFIG.read_text())
    fit = json.loads(FITTED_PARAMS.read_text())
    sets = prepare_param_sets(fit["accepted_params"], fit.get("scale_groups") or {}, cfg["params"])
    n_days = int(fit["num_days"])
    knots = _tv_knot_days(n_days, int(fit.get("tv_knot_spacing_days", 30)))
    csvs = cg._load_schedule_csvs(model_config_file=str(MODEL_CONFIG))
    h = pd.read_csv(io.StringIO(csvs["absolute_humidity_df"]))
    h["date"] = pd.to_datetime(h["date"], format="mixed")
    dates = pd.date_range(cg.START_DATE, periods=n_days, freq="D")
    hum = h.set_index("date")["absolute_humidity"].reindex(dates).ffill().to_numpy()

    m_all, hf_all = [], []
    for p in sets:
        incr = [p[k] for k in sorted((k for k in p if k.startswith("m_dlog_")), key=lambda k: int(k[7:]))]
        m_all.append(build_transmission_multiplier_array(incr, knots, n_days))
        hf_all.append(1 + float(p["humidity_impact"]) * np.exp(-180 * hum))
    m_all, hf_all = np.array(m_all), np.array(hf_all)
    series = {"m(t)": m_all, "Humidity factor": hf_all, "m(t) x humidity factor": m_all * hf_all}

    tab: dict[str, object] = {"date": dates.strftime("%Y-%m-%d")}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for (label, arr), color in zip(series.items(), ["#1f77b4", "#2ca02c", "#d62728"]):
        med = np.median(arr, axis=0)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        tab.update({f"{label} median": med, f"{label} 2.5%": lo, f"{label} 97.5%": hi})
        ax.fill_between(dates, lo, hi, color=color, alpha=0.2, lw=0)
        ax.plot(dates, med, color=color, lw=2, label=label)
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("Multiplier on beta_baseline")
    ax.set_title("Force of infection: humidity forcing and fitted time-varying multiplier m(t)\n"
                 "(posterior median + 95% band)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "transmission_multiplier.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(tab).to_csv(out / "transmission_multiplier.csv", index=False)


# ── Attack rates ─────────────────────────────────────────────────────────────

def attack_rates(con, out: Path, scenario: str = "baseline") -> pd.DataFrame:
    """Share of each age group (and of everyone) infected over the season,
    split by vaccination status, median [95% interval] across draws.

    Vaccinated = everyone vaccinated at some point in the season (SV(0) + all
    S_to_SV); the model never vaccinates someone already infected, so their
    infections are exactly SV_to_EV. Unvaccinated = everyone else; their
    infections are the rest, seeded infections included. Total = (N - S(T) -
    SV(T)) / N."""
    cfg = json.loads(MODEL_CONFIG.read_text())
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], dtype=float).ravel()
    df = con.execute(
        """SELECT rep, age_group, compartment, day, SUM(value) AS v FROM results_full
           WHERE scenario = ? AND compartment IN ('S', 'SV', 'S_to_SV', 'SV_to_EV')
           GROUP BY rep, age_group, compartment, day""", [scenario]).df()
    reps = np.sort(df["rep"].unique())
    first, last = df["day"].min(), df["day"].max()

    def cube(comp, how):
        d = df[df["compartment"] == comp]
        if how == "sum":
            d = d.groupby(["rep", "age_group"])["v"].sum()
        else:
            d = d[d["day"] == (last if how == "last" else first)].set_index(["rep", "age_group"])["v"]
        return d.unstack("age_group").reindex(index=reps, columns=range(len(AGE_GROUPS))).to_numpy()

    vacc = cube("SV", "first") + cube("S_to_SV", "sum")
    inf_v = cube("SV_to_EV", "sum")
    inf = pop[None, :] - cube("S", "last") - cube("SV", "last")
    inf_u, unvacc = inf - inf_v, pop[None, :] - vacc

    def cell(num, den):
        x = num / den * 100
        return f"{np.median(x):.1f}% [{np.percentile(x, 2.5):.1f}–{np.percentile(x, 97.5):.1f}%]"

    rows = [{"age_group": a, "vaccinated": cell(inf_v[:, i], vacc[:, i]),
             "unvaccinated": cell(inf_u[:, i], unvacc[:, i]), "total": cell(inf[:, i], pop[i]),
             "share_vaccinated": f"{np.median(vacc[:, i] / pop[i]) * 100:.1f}%"}
            for i, a in enumerate(AGE_GROUPS)]
    rows.append({"age_group": "All", "vaccinated": cell(inf_v.sum(1), vacc.sum(1)),
                 "unvaccinated": cell(inf_u.sum(1), unvacc.sum(1)), "total": cell(inf.sum(1), pop.sum()),
                 "share_vaccinated": f"{np.median(vacc.sum(1) / pop.sum()) * 100:.1f}%"})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "attack_rates.csv", index=False)
    return tab


# ── Coverage scenarios ───────────────────────────────────────────────────────

def coverage_response(tables: Path, out: Path) -> None:
    s = pd.read_csv(tables / "COVERAGE_SUMMARY.csv")
    s = s[s["scenario"] != "no vax"].sort_values("scheduled_coverage_pct")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, ylabel) in zip(axes, (("hosp", "Total hospitalizations (season)"),
                                        ("change_vs_baseline", "Change vs. baseline"))):
        y = s[f"{col}_median"]
        err = [y - s[f"{col}_lo"], s[f"{col}_hi"] - y]
        is_dec = s["label"].str.startswith("-")
        is_base = s["scenario"] == "baseline"
        for mask, color, lab in ((is_dec, "#3182bd", "Fewer doses"), (is_base, "black", "Baseline"),
                                 (~is_dec & ~is_base, "#d73027", "Coverage target")):
            ax.errorbar(s.loc[mask, "scheduled_coverage_pct"], y[mask],
                        yerr=[err[0][mask], err[1][mask]], fmt="o", color=color, capsize=4, label=lab)
        ax.plot(s["scheduled_coverage_pct"], y, color="grey", lw=0.8, zorder=0)
        for x, yy, lab in zip(s["scheduled_coverage_pct"], y, s["label"]):
            ax.annotate(lab, (x, yy), textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.set_xlabel("Overall scheduled vaccination coverage (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[0].legend()
    fig.suptitle("Hospitalizations by vaccination coverage (all age groups changed together; "
                 "median + 95% interval)")
    fig.tight_layout()
    fig.savefig(out / "coverage_response.png", dpi=150)
    plt.close(fig)


def coverage_by_age_targeted(tables: Path, out: Path) -> None:
    """% change in total (all-ages) hospitalizations vs baseline, by age group
    targeted (rows) x level (columns), read from the "All" row of each table."""
    targets, decreases, rows_out = {}, {}, []
    cols = AGE_GROUPS + ["All"]
    for t in COVERAGE_TARGETS:
        tab = pd.read_csv(tables / f"S_A_3_{t}pct_pct_reduction.csv", index_col=0)
        absd = pd.read_csv(tables / f"S_A_3_{t}pct_absolute.csv", index_col=0)
        targets[f"{t}%"] = [-_median(tab.loc["All", c]) for c in cols]
        rows_out += [{"scenario": f"{t}% coverage", "age_group_targeted": c,
                      "hosp_averted": absd.loc["All", c], "pct_reduction": tab.loc["All", c]}
                     for c in cols]
    for d in COVERAGE_DECREASES:
        tab = pd.read_csv(tables / f"DECREASE_{d}pct_pct_increase.csv", index_col=0)
        absd = pd.read_csv(tables / f"DECREASE_{d}pct_absolute.csv", index_col=0)
        decreases[f"-{d}%"] = [_median(tab.loc["All", c]) for c in cols]
        rows_out += [{"scenario": f"{d}% fewer doses", "age_group_targeted": c,
                      "hosp_added": absd.loc["All", c], "pct_increase": tab.loc["All", c]}
                     for c in cols]
    pd.DataFrame(rows_out).to_csv(out / "coverage_by_age_targeted.csv", index=False)

    labels = [f"{c} only" for c in AGE_GROUPS] + ["All ages"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, cmap, title in (
        (axes[0], pd.DataFrame(targets, index=labels), "Reds_r", "Raise coverage to target"),
        (axes[1], pd.DataFrame(decreases, index=labels), "Blues", "Remove a share of the baseline doses"),
    ):
        lim = np.nanmax(np.abs(data.to_numpy()))
        im = ax.imshow(data.to_numpy(), cmap=cmap, aspect="auto",
                       vmin=-lim if cmap == "Reds_r" else 0, vmax=0 if cmap == "Reds_r" else lim)
        ax.set_xticks(range(data.shape[1]), data.columns)
        ax.set_yticks(range(data.shape[0]), data.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.iat[i, j]
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.6 * lim else "black")
        ax.set_title(title)
        ax.set_ylabel("Age group(s) changed")
        fig.colorbar(im, ax=ax, label="% change in total hospitalizations")
    fig.suptitle("Change in total hospitalizations vs. baseline (median across posterior draws)")
    fig.tight_layout()
    fig.savefig(out / "coverage_heatmap.png", dpi=150)
    plt.close(fig)


# ── Parameter tables ─────────────────────────────────────────────────────────

def parameter_tables(out: Path) -> None:
    cfg = json.loads(MODEL_CONFIG.read_text())
    fit = json.loads(FITTED_PARAMS.read_text())
    acc = pd.DataFrame(fit["accepted_params"])
    best = fit["best_params"]
    names = {"beta_baseline": "beta_baseline", "humidity_impact": "humidity_impact",
             "seed_scale_E": "Initial-seed multiplier",
             **{f"IHR_scale|a{i}": f"Hospitalization-risk multiplier - {a}" for i, a in enumerate(AGE_GROUPS)},
             "phi": "phi (NB dispersion)"}
    rows = [{"parameter": label, "posterior_mean": acc[k].mean(), "p5": acc[k].quantile(0.05),
             "p95": acc[k].quantile(0.95), "best": best[k]}
            for k, label in names.items() if k in acc]
    pd.DataFrame(rows).to_csv(out / "fit_parameters.csv", index=False)

    p = cfg["params"]
    agg = cfg["initial_conditions"]["aggregate_pop"]
    flat = lambda v: np.asarray(v, dtype=float).ravel()
    # Fitted hospitalization risks: pre-fit value x each draw's age-specific
    # multiplier (both columns share it), summarized across posterior draws.
    fitted = {}
    for col, base in (("hosp_risk_given_infection", flat(p["I_to_H_prop"])),
                      ("hosp_risk_given_breakthrough", flat(p["IV_to_H_prop"]))):
        draws = np.column_stack([acc[f"IHR_scale|a{i}"] for i in range(len(AGE_GROUPS))]) * base
        fitted[f"{col}_fitted_mean"] = draws.mean(axis=0)
        fitted[f"{col}_fitted_p2.5"] = np.percentile(draws, 2.5, axis=0)
        fitted[f"{col}_fitted_p97.5"] = np.percentile(draws, 97.5, axis=0)
    pd.DataFrame({
        "age_group": AGE_GROUPS,
        "population": flat(agg["population"]).astype(int),
        "hosp_risk_given_infection": flat(p["I_to_H_prop"]),
        "hosp_risk_given_breakthrough": flat(p["IV_to_H_prop"]),
        "death_risk_given_hosp": flat(p["H_to_D_prop"]),
        "residual_susceptibility_vaccinated": flat(p["vax_susceptibility"]),
        "initial_infections_seeded": flat(agg["seeds"]["E"]).astype(int),
        **fitted,
    }).to_csv(out / "age_parameters.csv", index=False)

    cm = np.asarray(p["total_contact_matrix"], dtype=float)
    pd.DataFrame(cm, index=AGE_GROUPS, columns=AGE_GROUPS).round(2).to_csv(
        out / "contact_matrix.csv", index_label="age_group")


def pct_reduction_normalized(tables: Path, out: Path) -> Path:
    df = pd.read_csv(tables / "S_A_2_pct_reduction.csv", index_col="age_group").drop(columns=["All"])
    df = df.map(_median)
    df = df.div(df.sum(axis=1), axis=0)
    path = out / "S_A_2_pct_reduction_normalized.csv"
    df.round(6).to_csv(path)
    return path


# ── R figures ────────────────────────────────────────────────────────────────

def run_r(out: Path, tables: Path, ts_csv: Path, norm_csv: Path) -> None:
    def r(script, *args):
        cmd = ["Rscript", str(FIG_DIR / script), *map(str, args)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"{' '.join(cmd)} failed:\n{res.stderr}")

    ts = pd.read_csv(ts_csv)
    for fname, scen in R_TS_PLOTS.items():
        ymax = ts[ts["scenario"].isin(scen)]["p_hi"].max() * 1.02
        r("plot_scenarios_ts.R", ts_csv, out / fname, "|".join(scen), f"{ymax:.0f}")
    r("plot_pct_reduction_stacked_bar.R", norm_csv, out / "S_A_2_pct_reduction_stacked_bar.png")
    for t in COVERAGE_TARGETS:
        r("plot_per_100k_doses_stacked_bar.R", tables / f"S_A_3_{t}pct_per_100k_doses.csv",
          out / f"S_A_3_{t}pct_per_100k_doses_stacked_bar.png", f"{t}% coverage target")
    for d in COVERAGE_DECREASES:
        r("plot_per_100k_doses_stacked_bar.R", tables / f"DECREASE_{d}pct_per_100k_doses.csv",
          out / f"DECREASE_{d}pct_per_100k_doses_stacked_bar.png", f"{d}% fewer doses",
          "Additional hospitalizations per 100,000 doses removed")


def main() -> None:
    global MODEL_CONFIG, FITTED_PARAMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path,
                    default=MA_VAX / "simulation_output_MA_vax_param_set_stochastic" / "results_parquet")
    ap.add_argument("--tables", type=Path, default=MA_VAX / "counterfactual_tables_from_db_MA_vax")
    ap.add_argument("--fit-csv", type=Path, default=MA_VAX / "baseline_new_H_MA_vax.csv")
    ap.add_argument("--out", type=Path, default=FIG_DIR / "output")
    ap.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    ap.add_argument("--fitted-params", type=Path, default=FITTED_PARAMS)
    ap.add_argument("--no-r", action="store_true", help="skip the R (ggplot) figures")
    args = ap.parse_args()
    MODEL_CONFIG, FITTED_PARAMS = args.model_config, args.fitted_params
    args.out.mkdir(parents=True, exist_ok=True)

    print("scenario time series ...")
    con = results_io.load_source(str(args.db))
    meta = results_io.read_meta(con) or {}
    start = str(meta.get("start_date", cg.START_DATE)).strip('"')
    ts_csv = args.out / "scenario_timeseries.csv"
    scenario_timeseries(con, start).to_csv(ts_csv, index=False)
    print("attack rates ...")
    attack_rates(con, args.out)
    con.close()

    print("vaccination coverage ...")
    vax_coverage(args.out)
    print("fit check ...")
    fit_check(args.fit_csv, args.out)
    print("m(t) ...")
    transmission_multiplier(args.out)
    print("coverage scenarios ...")
    coverage_response(args.tables, args.out)
    coverage_by_age_targeted(args.tables, args.out)
    print("parameter tables ...")
    parameter_tables(args.out)
    norm_csv = pct_reduction_normalized(args.tables, args.out)
    if not args.no_r:
        print("R figures ...")
        run_r(args.out, args.tables, ts_csv, norm_csv)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
