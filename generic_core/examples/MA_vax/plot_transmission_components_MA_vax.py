"""Transmission components and per-age attack rate / calibrated IHR for the two
7-age m(t) MA_vax fits:

  current   model_config_MA_vax.json + fitted_params_MA_vax.json
  high vax  archive/2026-08-high-vax-rates/model_config_2026-08-high-vax-rates.json
            + archive/fitted_params.json, with the vaccination schedule it was
            fitted/simulated under ("... - OLD high vax.csv", same folder)

plus, with `--only <tag>`, the refits of the high-vax model with 20% of each
age group initially recovered (own config/fit/simulations in
archive/2026-08-high-vax-rates/, table written to
attack_rate_and_IHR_by_age_<tag>.{csv,md}):
  MA_vax_old_vax_initial_R_20pct                daily doses = proportion x (S + SV)
  MA_vax_old_vax_initial_R_20pct_vax_total_pop  daily doses = proportion x total population

Time series, each as posterior median + 95% band over accepted_params:

  m(t)            = exp(interp of cumsum(m_dlog_k) on the 14-day knots)
  humidity term   = 1 + humidity_impact * exp(-180 * h(t))
  beta_adjusted   = beta_baseline * m(t) * humidity term
  beta_eff        = beta_adjusted * sum_a w_a * susc_a(t)
        susc_a(t) = (1 - v_a(t))*relative_suscept + v_a(t)*vax_susceptibility_a
        v_a(t)    = cumulative scheduled coverage, lagged vax_transfer_delay_days
        w_a       = N_a / N
  C(t)            = total - (1-school_day)*school - (1-work_day)*work, using the
                    scalar contact values of MA_vax_single_age (calendar effect)
  beta_eff * C(t)/C_total  -- beta_eff with the calendar effect folded in

The "no vax" scenario keeps every parameter and only zeroes doses, so only
beta_eff (and the calendar-adjusted beta_eff) differ between scenarios.

Table, per age group:
  attack rate     = (N_a - R_a(0) - S_a(T) - SV_a(T)) / N_a from the param-set stochastic
                    runs (baseline and no vax), cross-checked against the
                    cumulative S_to_E + SV_to_EV flows plus seeded E
  effective IHR   = IHR_scale|a * I_to_H_prop_a   (unvaccinated)
                    IHR_scale|a * IV_to_H_prop_a  (vaccinated)
"""
import argparse, ast, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path("/Users/rfp437/Work/CityLevelTransmission/CLT_BaseModel")
MA = ROOT / "generic_core/examples/MA_vax"
ARCH = MA / "archive/2026-08-high-vax-rates"
SINGLE_AGE_CFG = MA / "MA_vax_single_age/model_config_MA_single_age.json"
OUT = MA / "transmission_components"

# (tag, label, config, fit, vaccination CSV, param-set simulation dir)
MODELS = [
    ("current", "7-age m(t), current vaccination data",
     MA / "model_config_MA_vax.json", MA / "fitted_params_MA_vax.json",
     MA / "data/schedules/MA_flu_daily_vaccinations_proportions_array.csv",
     MA / "simulation_output_MA_vax_param_set_stochastic"),
    ("high_vax", "7-age m(t), archived Aug-2026 high vaccination rates",
     ARCH / "model_config_2026-08-high-vax-rates.json", MA / "archive/fitted_params.json",
     ARCH / "MA_flu_daily_vaccinations_proportions_array - OLD high vax.csv",
     ARCH / "simulation_output_param_set_stochastic_2026-08-high-vax-rates"),
    ("MA_vax_old_vax_initial_R_20pct", "7-age m(t), Aug-2026 high vaccination rates, 20% initially recovered",
     ARCH / "model_config_MA_vax_old_vax_initial_R_20pct.json",
     ARCH / "fitted_params_MA_vax_old_vax_initial_R_20pct.json",
     ARCH / "MA_flu_daily_vaccinations_proportions_array - OLD high vax.csv",
     ARCH / "simulation_output_MA_vax_old_vax_initial_R_20pct_param_set_stochastic"),
    ("MA_vax_old_vax_initial_R_20pct_vax_total_pop",
     "7-age m(t), Aug-2026 high vaccination rates, 20% initially recovered, doses as a share of total population",
     ARCH / "model_config_MA_vax_old_vax_initial_R_20pct_vax_total_pop.json",
     ARCH / "fitted_params_MA_vax_old_vax_initial_R_20pct_vax_total_pop.json",
     ARCH / "MA_flu_daily_vaccinations_proportions_array - OLD high vax.csv",
     ARCH / "simulation_output_MA_vax_old_vax_initial_R_20pct_vax_total_pop_param_set_stochastic"),
]
DEFAULT_TAGS = ("current", "high_vax")
SCENARIOS = [("baseline", "baseline"), ("no vax", "no%20vax")]
TV_KNOT_SPACING = 14

# reference categorical palette (dataviz skill), slots 1-2; text stays in ink
C_BASE, C_NOVAX, C_SINGLE = "#2a78d6", "#eb6834", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_dated_csv(path, dates):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    return df.set_index("date").reindex(dates)


def m_of_t(params, num_days):
    knots = list(range(0, num_days, TV_KNOT_SPACING))
    if knots[-1] != num_days - 1:
        knots.append(num_days - 1)
    incr = [params[k] for k in sorted((k for k in params if k.startswith("m_dlog_")),
                                      key=lambda s: int(s.split("_")[-1]))]
    g = np.concatenate([[0.0], np.cumsum(np.asarray(incr, float))])
    return np.exp(np.interp(np.arange(num_days), np.asarray(knots, float), g))


def band(x):
    """(median, 2.5%, 97.5%) along the draw axis."""
    return np.percentile(x, [50, 2.5, 97.5], axis=0)


def load_final_state(sim_dir, scen_dir, comps, num_days):
    """{comp: (n_reps, n_age)} -- value on the last day for compartments, sum
    over all days for transition variables."""
    base = sim_dir / "results_parquet/results_full" / f"scenario={scen_dir}"
    out = {}
    for comp in comps:
        df = pd.concat([pd.read_parquet(f) for f in sorted((base / f"compartment={comp}").glob("*.parquet"))])
        if df["kind"].iloc[0] == "compartment":
            df = df[df["day"] == num_days - 1]
        out[comp] = df.pivot_table(index="rep", columns="age_group", values="value",
                                   aggfunc="sum").sort_index().to_numpy()
    return out


def to_markdown(df):
    rows = [list(df.columns), ["---"] * df.shape[1]] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def fmt_pct(med, lo, hi, digits=1):
    return f"{100*med:.{digits}f}% [{100*lo:.{digits}f}-{100*hi:.{digits}f}]"


# calendar contacts from the single-age model (scalars)
sa = json.loads(SINGLE_AGE_CFG.read_text())["params"]
C_TOT = float(np.ravel(sa["total_contact_matrix"])[0])
C_SCH = float(np.ravel(sa["school_contact_matrix"])[0])
C_WRK = float(np.ravel(sa["work_contact_matrix"])[0])

# --only TAG runs a single model and writes its attack-rate table to
# attack_rate_and_IHR_by_age_<TAG>.{csv,md}, leaving the default
# (current + high_vax) table untouched.
_ap = argparse.ArgumentParser()
_ap.add_argument("--only", choices=[m[0] for m in MODELS])
_args = _ap.parse_args()
_tags = (_args.only,) if _args.only else DEFAULT_TAGS
TABLE_STEM = f"attack_rate_and_IHR_by_age_{_args.only}" if _args.only else "attack_rate_and_IHR_by_age"

OUT.mkdir(exist_ok=True)
table_rows = []

for tag, label, cfg_path, fit_path, vax_csv, sim_dir in (m for m in MODELS if m[0] in _tags):
    cfg = json.loads(cfg_path.read_text())
    fit = json.loads(fit_path.read_text())
    p = cfg["params"]
    draws = fit["accepted_params"]
    num_days = int(fit["num_days"])
    dates = pd.date_range(cfg["simulation_settings"]["start_real_date"], periods=num_days, freq="D")
    ages = cfg["age_risk"]["age_groups"]
    n_age = len(ages)
    pop = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["population"], float).ravel()
    w = pop / pop.sum()

    folder = ROOT / cfg["input_files"]["input_folder"]
    hum = read_dated_csv(folder / cfg["input_files"]["absolute_humidity_csv"], dates)["absolute_humidity"].ffill().to_numpy()
    cal = read_dated_csv(folder / cfg["input_files"]["school_work_calendar_csv"], dates).ffill()
    is_school = cal["is_school_day"].to_numpy(float)
    is_work = cal["is_work_day"].to_numpy(float)
    vdf = pd.read_csv(vax_csv)
    vdf["date"] = pd.to_datetime(vdf["date"], format="%m/%d/%y")
    if vdf.loc[vdf["date"] < dates[0], "daily_vaccines"].map(
            lambda s: np.sum(ast.literal_eval(s))).sum() > 0:
        raise ValueError(f"{vax_csv.name}: doses before the simulation start would be dropped")
    varr = np.stack([np.asarray(ast.literal_eval(s), float).ravel() for s in vdf["daily_vaccines"]])
    vprop = pd.DataFrame(varr, index=pd.DatetimeIndex(vdf["date"])).reindex(dates).fillna(0.0).to_numpy()

    lag = int(p["vax_transfer_delay_days"])
    cov = np.cumsum(vprop, axis=0)
    cov = np.vstack([np.zeros((lag, n_age)), cov[:-lag]])
    vs = np.asarray(p["vax_susceptibility"], float).ravel()
    rs = float(p["relative_suscept"])
    susc_term = {"baseline": ((1 - cov) * rs + cov * vs) @ w,
                 "no vax": np.full(num_days, rs)}

    C_t = C_TOT - (1 - is_school) * C_SCH - (1 - is_work) * C_WRK
    cal_mult = C_t / C_TOT

    m = np.stack([m_of_t(q, num_days) for q in draws])
    hterm = np.stack([1 + float(q["humidity_impact"]) * np.exp(-180.0 * hum) for q in draws])
    beta_adj = np.array([float(q["beta_baseline"]) for q in draws])[:, None] * m * hterm
    beta_eff = {s: beta_adj * susc_term[s][None, :] for s in susc_term}

    series = {"m(t)": band(m), "humidity term": band(hterm), "beta_adjusted": band(beta_adj)}
    for s in beta_eff:
        series[f"beta_eff [{s}]"] = band(beta_eff[s])
        series[f"beta_eff x C(t)/C_total [{s}]"] = band(beta_eff[s] * cal_mult[None, :])

    # ---- CSV of the time series ------------------------------------------------
    cols = {"C(t) single-age contacts": C_t, "C(t)/C_total": cal_mult,
            "population vax coverage (lagged)": cov @ w}
    for name, (md, lo, hi) in series.items():
        cols[f"{name} median"], cols[f"{name} lo95"], cols[f"{name} hi95"] = md, lo, hi
    pd.DataFrame(cols, index=pd.Index(dates.strftime("%Y-%m-%d"), name="date")).to_csv(
        OUT / f"transmission_components_{tag}.csv", float_format="%.6g")

    # ---- figure ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 12.5), sharex=True)
    ax = axes.ravel()

    def draw(a, key, color, lab=None):
        md, lo, hi = series[key]
        a.fill_between(dates, lo, hi, color=color, alpha=0.18, lw=0)
        a.plot(dates, md, color=color, lw=2, label=lab)

    draw(ax[0], "m(t)", C_BASE)
    ax[0].axhline(1, color=INK2, lw=0.8, ls=":")
    ax[0].set_title("(a) Transmission multiplier m(t)")
    draw(ax[1], "humidity term", C_BASE)
    ax[1].set_title("(b) Humidity term  1 + humidity_impact·exp(−180·h(t))")
    draw(ax[2], "beta_adjusted", C_BASE)
    ax[2].set_title("(c) beta_adjusted = beta_baseline · m(t) · humidity term")
    for s, c in (("baseline", C_BASE), ("no vax", C_NOVAX)):
        draw(ax[3], f"beta_eff [{s}]", c, s)
        draw(ax[5], f"beta_eff x C(t)/C_total [{s}]", c, s)
    ax[3].set_title("(d) beta_eff = beta_adjusted · population susceptibility")
    ax[4].step(dates, C_t, where="mid", color=C_SINGLE, lw=1.2)
    ax[4].set_ylim(0, C_TOT * 1.08)
    ax[4].set_title("(e) Calendar effect: contacts/day C(t), single-age values")
    ax[5].set_title("(f) Calendar-adjusted beta_eff = beta_eff · C(t)/C_total")
    for a in (ax[3], ax[5]):
        a.legend(frameon=False, loc="upper left", labelcolor=INK)
    for a in ax:
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=9)
        a.title.set_color(INK)
        a.title.set_fontsize(10.5)
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    cov_end = (cov @ w)[-1]
    r0 = np.asarray(cfg["initial_conditions"]["aggregate_pop"].get("seeds", {}).get("R", np.zeros((n_age, 1))),
                    float).ravel()
    notes = (
        f"{label}.  Lines = posterior median, shaded = 95% interval over the {len(draws)} accepted MCMC draws "
        f"(panel e is deterministic).\n"
        "(a) m(t) = exp of the cumulative m_dlog_k increments, linearly interpolated between knots every "
        f"{TV_KNOT_SPACING} days.   (b) h(t) = daily absolute humidity (kg/m³); humidity_impact is fitted.\n"
        "(c) beta_adjusted is the per-contact transmission rate before susceptibility; identical in both scenarios "
        "(no vax only zeroes doses), so one curve is shown in (a)–(c).\n"
        "(d) beta_eff multiplies (c) by Σ_a (N_a/N)·[(1−v_a)·relative_suscept + v_a·vax_susceptibility_a], with v_a the "
        f"cumulative scheduled coverage lagged {lag} days\n"
        f"      (population coverage {100*cov_end:.1f}% by season end in baseline, 0 in no vax). "
        "Susceptible depletion from infection is NOT included"
        + (f" (nor the {100*r0.sum()/pop.sum():.0f}% initially recovered).\n" if r0.sum() > 0 else ".\n") +
        f"(e) C(t) = {C_TOT:.2f} − (1−school day)·{C_SCH:.2f} − (1−work day)·{C_WRK:.2f} contacts/day, from "
        "model_config_MA_single_age.json and the MA school/work calendar\n"
        f"      ({C_TOT:.2f} on school+work days, {C_TOT-C_SCH:.2f} on work-only days, {C_TOT-C_SCH-C_WRK:.2f} "
        "on weekends/holidays). The 7-age model applies the same calendar to its age-structured matrices.\n"
        "(f) beta_eff scaled by C(t)/C_total, i.e. transmission relative to a full school+work day, "
        "including the weekly and holiday dips."
    )
    fig.suptitle(f"MA_vax transmission components — {label}", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.145, 1, 0.97))
    fig.text(0.02, 0.005, notes, fontsize=8.6, color=INK2, va="bottom", ha="left", linespacing=1.45)
    fig.savefig(OUT / f"transmission_components_{tag}.png", dpi=150)
    plt.close(fig)

    # ---- table: attack rate + effective IHR --------------------------------------
    scale = np.array([[float(q[f"IHR_scale|a{a}"]) for a in range(n_age)] for q in draws])
    ihr_u = band(scale * np.asarray(p["I_to_H_prop"], float).ravel()[None, :])
    ihr_v = band(scale * np.asarray(p["IV_to_H_prop"], float).ravel()[None, :])

    seeds = np.asarray(cfg["initial_conditions"]["aggregate_pop"]["seeds"]["E"], float).ravel()
    ar = {}
    for s, scen_dir in SCENARIOS:
        st = load_final_state(sim_dir, scen_dir, ["S", "SV", "S_to_E", "SV_to_EV"], num_days)
        # people seeded directly into R (prior immunity) were never infected
        # this season, so they are not part of the attack rate
        inf = pop[None, :] - r0[None, :] - st["S"] - st["SV"]
        # (N-S-SV) - flows must be the seeded E, i.e. seeds_E * seed_scale_E of
        # that rep: the same ratio in every age group, within the posterior range
        ratio = (inf - (st["S_to_E"] + st["SV_to_EV"])) / seeds[None, :]
        spread = np.abs(ratio - ratio[:, :1]).max()
        ss = np.array([float(q.get("seed_scale_E", 1.0)) for q in draws])
        print(f"{tag:9s} {s:9s} reps={inf.shape[0]}  implied seed scale "
              f"{ratio.min():.3f}–{ratio.max():.3f} (posterior seed_scale_E {ss.min():.3f}–{ss.max():.3f}), "
              f"max cross-age spread {spread:.1e}")
        ar[s] = band(inf / pop[None, :])
        ar[s + " all"] = band(inf.sum(1) / pop.sum())

    for a in range(n_age):
        table_rows.append({
            "model": label, "age group": ages[a],
            "attack rate, baseline": fmt_pct(*ar["baseline"][:, a]),
            "attack rate, no vax": fmt_pct(*ar["no vax"][:, a]),
            "effective IHR, unvaccinated": fmt_pct(*ihr_u[:, a], digits=3),
            "effective IHR, vaccinated": fmt_pct(*ihr_v[:, a], digits=3),
        })
    table_rows.append({
        "model": label, "age group": "all ages",
        "attack rate, baseline": fmt_pct(*ar["baseline all"]),
        "attack rate, no vax": fmt_pct(*ar["no vax all"]),
        "effective IHR, unvaccinated": "n/a", "effective IHR, vaccinated": "n/a",
    })

tbl = pd.DataFrame(table_rows)
tbl.to_csv(OUT / f"{TABLE_STEM}.csv", index=False)

md = ["# Attack rate and effective calibrated IHR by age group", "",
      "Median [95% interval]. Attack rate = share of the age group infected over the 250-day season "
      "(N − R(0) − S(T) − SV(T)) / N, including seeded infections and excluding anyone seeded as "
      "recovered, across the param-set stochastic runs. "
      "Effective IHR = IHR_scale|age × I_to_H_prop (unvaccinated) or × IV_to_H_prop (vaccinated), "
      "i.e. the probability that an infection is hospitalized, across the accepted MCMC draws.", ""]
for model, g in tbl.groupby("model", sort=False):
    md += [f"## {model}", "", to_markdown(g.drop(columns="model")), ""]
(OUT / f"{TABLE_STEM}.md").write_text("\n".join(md))
print(f"wrote outputs to {OUT}")
