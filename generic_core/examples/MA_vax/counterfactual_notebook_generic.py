"""
  marimo run generic_core/examples/MA_vax/counterfactual_notebook_generic.py
"""


import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import os
    import sys

    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    import counterfactual_generic as cf

    return cf, io, json, mo, np, os, pd, plt


@app.cell(hide_code=True)
def _(mo):
    def show_table(df, filename: str, page_size: int = 10):
        """Render `df` as an interactive table, or a clear warning if the CSV
        wasn't found (e.g. the results folder was wrong, or that table's CSV
        is missing) instead of crashing on `.reset_index()`."""
        if df is None:
            return mo.md(f"⚠️ **missing `{filename}`** — check the results folder path above.").callout(kind="warn")
        return mo.ui.table(df.reset_index(), page_size=page_size)

    return (show_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Counterfactual vaccination-impact tables (generic_core)

    Replicates the structure of PNAS 2505175122 Supplementary Tables
    S.A.1-S.A.6 -- adapted to this model's 7 age groups, the Massachusetts
    population, and the 2025-2026 season -- but built through generic_core's
    exported-script machinery (`counterfactual_generic.py` in this folder)
    instead of the hand-written `MA_vax.model` equations that
    `MA_vax/counterfactual_notebook.py` uses. This is the same analysis,
    run against the notebook-built/exported model, as a check that the two
    reproduce the same vaccination-impact conclusions. Scenario definitions
    live in `counterfactual_generic.py` (`single_age_only_scenario`,
    `coverage_70pct_scenario`, `ve_scenarios`, ...).

    This notebook only **loads and displays** results that were already
    computed by

    ```
    python generic_core/examples/MA_vax/run_counterfactual_tables_generic.py --n-reps 1000
    ```

    Point it at that run's output folder below. Re-running the CLI script
    and pointing at the new folder is how you refresh the numbers -- this
    notebook itself never runs the S.A.* table simulations, so it stays fast
    regardless of `n_reps` (the epi-curve and coverage sections further down
    *do* run fresh deterministic simulations directly, since those are cheap).

    **70%-coverage caveat:** unlike `MA_vax/counterfactual.py`'s
    `coverage_multiplier_for_target` (which can bisection-refine), the
    coverage-target scenarios here use a single naive cross-product
    multiplier (`target / baseline_coverage`) with no verification -- see
    the "Vaccination coverage" section below for how far off 70% it actually
    lands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    results_folder = mo.ui.text(
        value="generic_core/examples/MA_vax/counterfactual_tables_from_db",
        placeholder="generic_core/examples/MA_vax/counterfactual_tables_from_db",
        label="Results folder (from run_counterfactual_tables_generic.py)",
        full_width=True,
    )
    results_folder
    return (results_folder,)


@app.cell
def _(cf, mo, os, results_folder):
    mo.stop(
        not results_folder.value,
        mo.md("Enter a results folder produced by `run_counterfactual_tables_generic.py` above."),
    )
    mo.stop(
        not os.path.isdir(results_folder.value),
        mo.md(
            f"⚠️ **`{results_folder.value}` is not a folder.** "
            "Paths are relative to wherever `marimo edit`/`marimo run` was launched from "
            "(usually the repo root) -- a leading `/` makes it absolute from the filesystem "
            "root instead, which is a common way to hit this."
        ).callout(kind="danger"),
    )
    tables = cf.load_saved_tables(results_folder.value)
    meta = tables["meta"]
    mo.md(
        f"**model_config:** `{meta.get('model_config_file')}` &nbsp;·&nbsp; "
        f"**fitted_params:** `{meta.get('fitted_params_file')}` &nbsp;·&nbsp; "
        f"**n_reps:** `{meta.get('n_reps')}` &nbsp;·&nbsp; "
        f"**seed:** `{meta.get('seed')}` &nbsp;·&nbsp; "
        f"**generated:** `{meta.get('generated_at')}`"
    ) if meta else mo.md("_No `meta.json` found in this folder._")
    return (tables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.1 — Hospitalizations averted, infection vs. severity protection

    Compares *no vaccination* -> *infection-protection-only* (VE against
    infection retained, VE against severity zeroed out) -> *baseline*
    (full VE), by age group and aggregate.
    """)
    return


@app.cell
def _(show_table, tables):
    show_table(tables["S_A_1"], "S_A_1.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.2 — Hospitalizations averted by age group vaccinated

    Each column vaccinates a single age group only (all others
    unvaccinated) and compares to no vaccination; "All" is the full
    baseline schedule. Rows are the age group in which hospitalizations
    are counted.
    """)
    return


@app.cell
def _(mo, show_table, tables):
    t2 = tables["S_A_2"]
    mo.vstack([
        mo.md("**Percent reduction in hospitalizations**"),
        show_table(t2["pct_reduction"], "S_A_2_pct_reduction.csv"),
        mo.md("**Hospitalizations averted per 100K population**"),
        show_table(t2["per_100k"], "S_A_2_per_100k.csv"),
        mo.md("**Hospitalizations averted per 100K doses**"),
        show_table(t2["per_100k_doses"], "S_A_2_per_100k_doses.csv"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.3 — Additional hospitalizations averted at 70% coverage

    Each column scales a single age group's vaccination schedule to reach
    (a naive estimate of) 70% cumulative coverage; "All" scales every age
    group. Compared against the baseline vaccination scenario.
    """)
    return


@app.cell
def _(mo, show_table, tables):
    t3 = tables["S_A_3"]
    mo.vstack([
        mo.md("**Percent reduction in hospitalizations**"),
        show_table(t3["pct_reduction"], "S_A_3_pct_reduction.csv"),
        mo.md("**Hospitalizations averted per 100K population**"),
        show_table(t3["per_100k"], "S_A_3_per_100k.csv"),
        mo.md("**Hospitalizations averted per 100K additional doses**"),
        show_table(t3["per_100k_doses"], "S_A_3_per_100k_doses.csv"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.4 — VE sensitivity scenarios

    Implied vaccine effectiveness against infection and against
    hospitalization-given-infection for each preset in `cf.ve_scenarios`.
    Same placeholder multipliers as `MA_vax.counterfactual.VE_SCENARIOS`, so
    directly comparable to the equivalent table there.
    """)
    return


@app.cell
def _(show_table, tables):
    show_table(tables["S_A_4"], "S_A_4.csv", page_size=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.5 — Hospitalizations averted across VE scenarios

    Compares each VE sensitivity scenario against no vaccination.
    """)
    return


@app.cell
def _(mo, show_table, tables):
    t5 = tables["S_A_5"]
    mo.vstack([
        mo.md("**Percent reduction in hospitalizations**"),
        show_table(t5["pct_reduction"], "S_A_5_pct_reduction.csv"),
        mo.md("**Hospitalizations averted per 100K population**"),
        show_table(t5["per_100k"], "S_A_5_per_100k.csv"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.6 — Additional hospitalizations averted at 70% coverage, across VE scenarios

    For each VE sensitivity scenario, compares that scenario's own baseline
    vaccination to (naive) 70% coverage in every age group.
    """)
    return


@app.cell
def _(mo, show_table, tables):
    t6 = tables["S_A_6"]
    mo.vstack([
        mo.md("**Percent reduction in hospitalizations**"),
        show_table(t6["pct_reduction"], "S_A_6_pct_reduction.csv"),
        mo.md("**Hospitalizations averted per 100K population**"),
        show_table(t6["per_100k"], "S_A_6_per_100k.csv"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vaccine-efficacy mechanism check

    Same layout as Table S.A.2 (columns = age group vaccinated, "All" =
    baseline schedule; rows = age group the flow is counted in), but instead
    of population-level hospitalizations averted, these look directly at the
    model's internal flows to confirm vaccination is doing what it should.
    All three entries are rate ratios (vaccinated rate / unvaccinated rate)
    shown as a percentage -- below 100% means vaccination is reducing that
    risk. See `MA_vax/counterfactual.py`'s `table_vax_efficacy_check`
    docstring for the immortal-time-bias discussion behind why the
    matched-cohort version is the one comparable to real-world VE.
    """)
    return


@app.cell
def _(mo, show_table, tables):
    tvc = tables["VAX_CHECK"]
    mo.vstack([
        mo.md("**Infection reduction (attack-rate ratio — biased, naive-analysis comparison point)**"),
        show_table(tvc["infection_reduction"], "VAX_CHECK_infection_reduction.csv"),
        mo.md("**Infection reduction (matched-cohort attack-rate ratio — the real-world-comparable estimate)**"),
        show_table(tvc["matched_cohort_infection_reduction"], "VAX_CHECK_matched_cohort_infection_reduction.csv"),
        mo.md("**Hospitalization reduction (hospitalization-given-infection rate ratio)**"),
        show_table(tvc["hospitalization_reduction"], "VAX_CHECK_hospitalization_reduction.csv"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vaccination coverage — baseline vs. 70%-coverage scenario

    Cumulative proportion of each age group's population vaccinated
    (`S_to_SV` summed over the season, divided by population) under the
    baseline schedule, versus the "All ages -> 70%" scenario. Since the
    coverage multiplier here is a naive cross-product estimate (not
    bisection-refined -- see the notebook intro), this table is also the
    check for **how far off 70% the scaled scenario actually lands** in
    each age group.
    """)
    return


@app.cell
def _(cf, np, pd):
    _base_inputs = cf.load_base_inputs()
    _target70_scenario = cf.coverage_70pct_scenario(_base_inputs, None)

    _pop = _base_inputs["population"]

    def _cumulative_coverage(scenario, age_idx):
        _ds = cf._run_reps(_base_inputs, scenario, n_reps=1, seed=0, stochastic=False)
        return float(_ds["S_to_SV"].isel(replication=0).sum(dim="day").to_numpy()[age_idx] / _pop[age_idx])

    _baseline_cov = [_cumulative_coverage(cf.baseline_scenario(), i) for i in range(len(cf.AGE_GROUPS))]
    _target70_cov = [_cumulative_coverage(_target70_scenario, i) for i in range(len(cf.AGE_GROUPS))]

    coverage_table = pd.DataFrame({
        "population": _pop,
        "baseline_coverage": _baseline_cov,
        "70pct_scenario_coverage": _target70_cov,
    }, index=cf.AGE_GROUPS)
    coverage_table.loc["All"] = [
        _pop.sum(),
        float(np.average(_baseline_cov, weights=_pop)),
        float(np.average(_target70_cov, weights=_pop)),
    ]
    coverage_table["baseline_coverage"] = (coverage_table["baseline_coverage"] * 100).round(1).astype(str) + "%"
    coverage_table["70pct_scenario_coverage"] = (coverage_table["70pct_scenario_coverage"] * 100).round(1).astype(str) + "%"
    coverage_table.index.name = "age_group"
    return (coverage_table,)


@app.cell
def _(coverage_table, show_table):
    show_table(coverage_table, "vaccination_coverage.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matched-cohort attack-probability curves

    For the baseline vaccination schedule, `attack_SV_from(d)` (solid) and
    `attack_S_from(d)` (dashed) by age group -- the counterfactual
    season-end attack probability for a hypothetical individual entering
    `SV`/`S` on day `d` and followed to season end. See
    `MA_vax.counterfactual.table_vax_efficacy_check`'s docstring (the
    "matched_cohort_infection_reduction" entry) for the full derivation --
    `attack_probability_curves` is reused unchanged from there, since it
    only operates on already-simulated (reps, day, age) arrays.
    """)
    return


@app.cell
def _(cf, np):
    _base_inputs = cf.load_base_inputs()
    _d = cf._scenario_daily_arrays(_base_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False)
    _attack_S_from, _attack_SV_from = cf.attack_probability_curves(
        _d["S"], _d["SV"], _d["S_to_E"], _d["SV_to_EV"])

    attack_curve_dates = cf._run_reps(_base_inputs, cf.baseline_scenario(), 1, 0, stochastic=False)["day"].to_numpy()
    attack_S_from = _attack_S_from[0]   # (day, A), single deterministic replication
    attack_SV_from = _attack_SV_from[0]
    # Before an age group's first-ever vaccination day, SV=0 and attack_SV_from(d) falls
    # back to a "zero hazard that day" placeholder -- meaningless for a day nobody was
    # actually vaccinated on, so mask it out of the plot.
    attack_SV_from = np.where(_d["SV"][0] > 0, attack_SV_from, np.nan)
    return attack_S_from, attack_SV_from, attack_curve_dates


@app.cell
def _(attack_S_from, attack_SV_from, attack_curve_dates, cf, mo, plt):
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    _axes = _axes.flatten()
    for _i, _age in enumerate(cf.AGE_GROUPS):
        _ax = _axes[_i]
        _ax.plot(attack_curve_dates, attack_SV_from[:, _i] * 100, label="attack_SV_from(d)",
                  color="C1", linewidth=1.5)
        _ax.plot(attack_curve_dates, attack_S_from[:, _i] * 100, label="attack_S_from(d)",
                  color="black", linewidth=1, linestyle="--")
        _ax.set_title(_age)
        _ax.set_ylabel("attack probability (%)")
        _ax.grid(True, alpha=0.3)
    _axes[-1].axis("off")
    _axes[0].legend(loc="upper right")
    _fig.suptitle("Matched-cohort attack-probability curves by age group (baseline schedule)")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### attack_SV_from(d) vs. attack_S_from(d), by age group"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline fit check — simulated vs. raw hospitalizations by age group

    Compares a fresh deterministic **baseline-vaccination** simulation
    (re-run here, not one of the loaded S.A.* tables) to the raw per-age
    daily hospitalization series in
    `MA_flu_daily_hospitalizations.csv` (from `MA_vax.model.DATA_FOLDER`,
    i.e. `generic_core/examples/massachusetts_vax/data/hospitalizations_ts/`),
    over the dates the two series have in common. This is a fit-quality
    check, separate from the counterfactual tables above, and specifically a
    check that the notebook-built/exported model tracks the fitted data as
    well as `MA_vax.model` did when it was fit -- both should be close if
    the export round-trip preserved the fit, since they share the same
    `fitted_params.json`-style inputs.
    """)
    return


@app.cell
def _(cf, os, pd):
    from MA_vax.model import DATA_FOLDER as _DATA_FOLDER

    _base_inputs = cf.load_base_inputs()
    _ds = cf._run_reps(_base_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False)
    sim_new_H = (_ds["I_to_H"] + _ds["IV_to_H"]).isel(replication=0).to_pandas()
    sim_new_H.index.name = "date"

    raw_age_H = pd.read_csv(
        os.path.join(_DATA_FOLDER, "data", "hospitalizations_ts", "MA_flu_daily_hospitalizations.csv")
    )
    raw_age_H["date"] = pd.to_datetime(raw_age_H["Date"])
    raw_age_H = raw_age_H.set_index("date").drop(columns=["Date"])
    raw_age_H.columns = cf.AGE_GROUPS
    return raw_age_H, sim_new_H


@app.cell
def _(mo, pd, raw_age_H, show_table, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_age_H.index)
    _sim_cum = sim_new_H.loc[_common_dates].sum()
    _raw_cum = raw_age_H.loc[_common_dates].sum()
    _pct_diff = (_sim_cum - _raw_cum) / _raw_cum.replace(0, pd.NA) * 100

    cum_hosp_table = pd.DataFrame({
        "simulated": _sim_cum,
        "raw_data": _raw_cum,
        "pct_diff": _pct_diff,
    }).round(1)
    cum_hosp_table.loc["All"] = [
        round(_sim_cum.sum(), 1),
        round(_raw_cum.sum(), 1),
        round((_sim_cum.sum() - _raw_cum.sum()) / _raw_cum.sum() * 100, 1),
    ]
    cum_hosp_table.index.name = "age_group"

    mo.vstack([
        mo.md(
            f"**Cumulative hospitalizations, {_common_dates.min().date()} to "
            f"{_common_dates.max().date()}** ({len(_common_dates)} days common to both series)"
        ),
        show_table(cum_hosp_table, "cumulative_hospitalizations_by_age.csv"),
    ])
    return


@app.cell
def _(cf, mo, plt, raw_age_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_age_H.index)
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    _axes = _axes.flatten()
    for _i, _age in enumerate(cf.AGE_GROUPS):
        _ax = _axes[_i]
        _ax.plot(_common_dates, raw_age_H.loc[_common_dates, _age], label="Raw data",
                  color="black", linewidth=1)
        _ax.plot(_common_dates, sim_new_H.loc[_common_dates, _age], label="Simulated (baseline)",
                  color="C1", linewidth=1.5)
        _ax.set_title(_age)
        _ax.grid(True, alpha=0.3)
    _ax_all = _axes[-1]
    _ax_all.plot(_common_dates, raw_age_H.loc[_common_dates].sum(axis=1), label="Raw data",
                 color="black", linewidth=1)
    _ax_all.plot(_common_dates, sim_new_H.loc[_common_dates].sum(axis=1), label="Simulated (baseline)",
                 color="C1", linewidth=1.5)
    _ax_all.set_title("All ages combined")
    _ax_all.grid(True, alpha=0.3)
    _axes[0].legend(loc="upper right")
    _fig.suptitle("Daily new hospitalizations: simulated (baseline) vs. raw data, by age group")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Actual vs. simulated daily hospitalizations by age group"), _fig])
    return


@app.cell
def _(cf, mo, plt, raw_age_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_age_H.index)
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    _axes = _axes.flatten()
    for _i, _age in enumerate(cf.AGE_GROUPS):
        _ax = _axes[_i]
        _ax.plot(_common_dates, raw_age_H.loc[_common_dates, _age].cumsum(), label="Raw data",
                  color="black", linewidth=1)
        _ax.plot(_common_dates, sim_new_H.loc[_common_dates, _age].cumsum(), label="Simulated (baseline)",
                  color="C1", linewidth=1.5)
        _ax.set_title(_age)
        _ax.grid(True, alpha=0.3)
    _ax_all = _axes[-1]
    _ax_all.plot(_common_dates, raw_age_H.loc[_common_dates].sum(axis=1).cumsum(), label="Raw data",
                 color="black", linewidth=1)
    _ax_all.plot(_common_dates, sim_new_H.loc[_common_dates].sum(axis=1).cumsum(), label="Simulated (baseline)",
                 color="C1", linewidth=1.5)
    _ax_all.set_title("All ages combined")
    _ax_all.grid(True, alpha=0.3)
    _axes[0].legend(loc="upper right")
    _fig.suptitle("Cumulative hospitalizations: simulated (baseline) vs. raw data, by age group")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Actual vs. simulated cumulative hospitalizations by age group"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vaccination vs. epidemic timing (interactive)

    Compare the vaccination time series against epidemic-curve metrics, by
    age group, to see how vaccination timing lines up with the epidemic.
    Both series are shown as **proportion of population** (not raw counts),
    so different-sized age groups are directly comparable and on the same
    scale as each other -- though vaccination and the epidemic metric are on
    separate y-axes (left = epidemic metric, right = vaccination), since
    their typical magnitudes differ a lot.

    Both series come from the same deterministic baseline simulation as the
    "Baseline fit check" section above. Two vaccination sources are available
    (select either or both):

    - **Simulated (`S_to_SV`)** -- doses that actually landed in the model,
      after the `scheduled_exact` transition's own capping against available
      `S`.
    - **Raw input schedule** -- the uploaded `daily_vaccines_df`'s per-day,
      per-age-group proportion (from `schedules.json`, already a proportion
      of that age group's total population, matching `MA_vax.model`'s
      `vax_arr` semantics), aligned to the simulation's date range. This is
      the *nominal* schedule as given, before any capping the model's
      transition engine applies -- it diverges from the simulated series
      wherever that capping actually binds. For the "All" entity, both
      sources are population-weighted (summed counts / summed population),
      not an average of per-age proportions.
    """)
    return


@app.cell
def _(cf, io, json, np, pd):
    _base_inputs = cf.load_base_inputs()
    vax_timing_ds = cf._run_reps(_base_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False).isel(replication=0)
    vax_timing_population = _base_inputs["population"]

    _csvs = cf._load_schedule_csvs()
    _df_vax = pd.read_csv(io.StringIO(_csvs["daily_vaccines_df"]))
    _df_vax["date"] = pd.to_datetime(_df_vax["date"])
    _dates_sim = pd.to_datetime(vax_timing_ds["day"].to_numpy())
    _df_vax = _df_vax.set_index("date").reindex(_dates_sim)
    # reindex introduces NaN rows for any simulation date absent from the uploaded
    # schedule -- treat those as zero vaccination rather than crashing json.loads.
    vax_timing_vax_arr = np.array([
        [_age_val[0] for _age_val in json.loads(_row)] if isinstance(_row, str) else [0.0] * cf.NUM_AGE_GROUPS
        for _row in _df_vax["daily_vaccines"]
    ])
    return vax_timing_ds, vax_timing_population, vax_timing_vax_arr


@app.cell
def _(mo):
    _metric_options = (
        [f"{c} (compartment)" for c in ["S", "E", "I", "R", "SV", "EV", "IV", "H", "D"]]
        + [f"{t} (daily)" for t in
           ["S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R", "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R"]]
        + [f"{t} (cumulative)" for t in
           ["S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R", "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R"]]
    )
    vax_timing_age_selector = mo.ui.multiselect(
        options=["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+", "All"],
        value=["All"],
        label="Age group(s)",
    )
    vax_timing_metric_selector = mo.ui.dropdown(
        options=_metric_options,
        value="I (compartment)",
        label="Epidemic metric",
    )
    vax_timing_source_selector = mo.ui.multiselect(
        options=["Simulated (S_to_SV)", "Raw input schedule (vax_arr)"],
        value=["Simulated (S_to_SV)"],
        label="Vaccination source(s)",
    )
    vax_timing_mode_selector = mo.ui.radio(
        options=["daily", "cumulative"],
        value="daily",
        label="Vaccination series",
    )
    mo.hstack(
        [vax_timing_age_selector, vax_timing_metric_selector,
         vax_timing_source_selector, vax_timing_mode_selector],
        justify="start", gap=2,
    )
    return (
        vax_timing_age_selector,
        vax_timing_metric_selector,
        vax_timing_mode_selector,
        vax_timing_source_selector,
    )


@app.cell
def _(
    cf,
    mo,
    plt,
    vax_timing_age_selector,
    vax_timing_ds,
    vax_timing_mode_selector,
    vax_timing_metric_selector,
    vax_timing_source_selector,
    vax_timing_population,
    vax_timing_vax_arr,
):
    mo.stop(
        not vax_timing_age_selector.value,
        mo.md("Select at least one age group above.").callout(kind="warn"),
    )
    mo.stop(
        not vax_timing_source_selector.value,
        mo.md("Select at least one vaccination source above.").callout(kind="warn"),
    )

    _dates = vax_timing_ds["day"].to_numpy()
    _labels = cf.AGE_GROUPS

    def _select_np(counts2d, entity):
        if entity == "All":
            return counts2d.sum(axis=1), vax_timing_population.sum()
        idx = _labels.index(entity)
        return counts2d[:, idx], vax_timing_population[idx]

    def _select(da, entity):
        return _select_np(da.to_numpy(), entity)

    # vax_timing_vax_arr is already a proportion of each age group's total population
    # (see daily_vaccines_df in schedules.json), so recover counts by multiplying back
    # by population -- no capping here, this is the raw nominal schedule, uncapped by
    # the model's `scheduled_exact` transition (unlike S_to_SV, which is capped by it).
    _raw_implied_counts = vax_timing_vax_arr * vax_timing_population[None, :]

    _vax_sources = {
        "Simulated (S_to_SV)": lambda entity: _select(vax_timing_ds["S_to_SV"], entity),
        "Raw input schedule (vax_arr)": lambda entity: _select_np(_raw_implied_counts, entity),
    }

    _metric = vax_timing_metric_selector.value
    if _metric.endswith(" (compartment)"):
        _metric_da, _metric_cumulative = vax_timing_ds[_metric.removesuffix(" (compartment)")], False
    elif _metric.endswith(" (cumulative)"):
        _metric_da, _metric_cumulative = vax_timing_ds[_metric.removesuffix(" (cumulative)")], True
    else:
        _metric_da, _metric_cumulative = vax_timing_ds[_metric.removesuffix(" (daily)")], False

    _fig, _ax1 = plt.subplots(figsize=(12, 5))
    _ax2 = _ax1.twinx()
    _n_lines = len(vax_timing_age_selector.value) * (1 + len(vax_timing_source_selector.value))
    _colors = plt.cm.tab10.colors if _n_lines <= 10 else plt.cm.tab20.colors
    _color_idx = 0

    _handles = []
    for _entity in vax_timing_age_selector.value:
        _metric_counts, _metric_pop = _select(_metric_da, _entity)
        _metric_series = (_metric_counts.cumsum() if _metric_cumulative else _metric_counts) / _metric_pop
        _h1, = _ax1.plot(_dates, _metric_series, color=_colors[_color_idx % len(_colors)], linestyle="-",
                          label=f"{_entity} — {_metric}")
        _handles.append(_h1)
        _color_idx += 1

        _vax_cumulative = vax_timing_mode_selector.value == "cumulative"
        for _source in vax_timing_source_selector.value:
            _vax_counts, _vax_pop = _vax_sources[_source](_entity)
            _vax_series = (_vax_counts.cumsum() if _vax_cumulative else _vax_counts) / _vax_pop
            _h2, = _ax2.plot(_dates, _vax_series, color=_colors[_color_idx % len(_colors)], linestyle="-",
                              label=f"{_entity} — {_source} ({vax_timing_mode_selector.value})")
            _handles.append(_h2)
            _color_idx += 1

    _ax1.set_ylabel(f"{_metric} (proportion of population)")
    _ax2.set_ylabel(f"Vaccinations, {vax_timing_mode_selector.value} (proportion of population)")
    _ax1.set_xlabel("date")
    _ax1.grid(True, alpha=0.3)
    _ax1.legend(handles=_handles, loc="upper left", fontsize=8)
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Vaccination vs. epidemic timing"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Epi curves by scenario (interactive)

    Compare any compartment or transition (as a time series or cumulative),
    for any age group (or "All"), across the scenarios in
    `counterfactual_generic.py`. Each scenario is a fresh deterministic
    simulation built via `cf.build_model`/`cf._run_reps` directly (not one of
    the loaded S.A.* tables). Values shown are proportion of population, so
    different-sized age groups/scenarios are directly comparable.
    """)
    return


@app.cell
def _():
    epi_curve_composite_metrics = {
        "S_to_E + SV_to_EV (total new infections)": ("S_to_E", "SV_to_EV"),
        "E_to_I + EV_to_IV (total new infectious)": ("E_to_I", "EV_to_IV"),
        "I_to_H + IV_to_H (total new hospitalizations)": ("I_to_H", "IV_to_H"),
    }
    return (epi_curve_composite_metrics,)


@app.cell
def _(cf):
    _base_inputs = cf.load_base_inputs()
    epi_curve_scenarios = {
        "Baseline (fitted vaccination)": cf.baseline_scenario(),
        "No vaccination": cf.no_vaccine_scenario(),
        "Low VE": cf.ve_scenarios(_base_inputs)["low_ve"],
        "High VE": cf.ve_scenarios(_base_inputs)["high_ve"],
        "70% coverage (all ages)": cf.coverage_70pct_scenario(_base_inputs, None),
    }
    for _i, _label in enumerate(cf.AGE_GROUPS):
        epi_curve_scenarios[f"Vaccinate {_label} only"] = cf.single_age_only_scenario(_i)
    epi_curve_base_inputs = _base_inputs
    return epi_curve_base_inputs, epi_curve_scenarios


@app.cell
def _(epi_curve_composite_metrics, epi_curve_scenarios, mo):
    _metric_options = (
        [f"{c} (compartment)" for c in ["S", "E", "I", "R", "SV", "EV", "IV", "H", "D"]]
        + [f"{t} (daily)" for t in
           ["S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R", "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R"]]
        + [f"{t} (cumulative)" for t in
           ["S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R", "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R"]]
        + [f"{name} (daily)" for name in epi_curve_composite_metrics]
        + [f"{name} (cumulative)" for name in epi_curve_composite_metrics]
    )
    epi_curve_scenario_selector = mo.ui.multiselect(
        options=list(epi_curve_scenarios.keys()),
        value=["Baseline (fitted vaccination)", "No vaccination"],
        label="Scenario(s)",
    )
    epi_curve_age_selector = mo.ui.multiselect(
        options=["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+", "All"],
        value=["All"],
        label="Age group(s)",
    )
    epi_curve_metric_selector = mo.ui.multiselect(
        options=_metric_options,
        value=["H (compartment)"],
        label="Metric(s)",
    )
    mo.hstack(
        [epi_curve_scenario_selector, epi_curve_age_selector, epi_curve_metric_selector],
        justify="start", gap=2,
    )
    return epi_curve_age_selector, epi_curve_metric_selector, epi_curve_scenario_selector


@app.cell
def _(
    cf,
    epi_curve_base_inputs,
    epi_curve_scenario_selector,
    epi_curve_scenarios,
):
    epi_curve_datasets = {}
    for _name in epi_curve_scenario_selector.value:
        epi_curve_datasets[_name] = cf._run_reps(
            epi_curve_base_inputs, epi_curve_scenarios[_name], n_reps=1, seed=0, stochastic=False
        ).isel(replication=0)
    epi_curve_population = epi_curve_base_inputs["population"]
    return epi_curve_datasets, epi_curve_population


@app.cell
def _(
    epi_curve_age_selector,
    epi_curve_composite_metrics,
    epi_curve_datasets,
    epi_curve_metric_selector,
    epi_curve_population,
    mo,
    plt,
):
    mo.stop(
        not epi_curve_datasets,
        mo.md("Select at least one scenario above.").callout(kind="warn"),
    )
    mo.stop(
        not epi_curve_age_selector.value,
        mo.md("Select at least one age group above.").callout(kind="warn"),
    )
    mo.stop(
        not epi_curve_metric_selector.value,
        mo.md("Select at least one metric above.").callout(kind="warn"),
    )

    _labels = ["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"]
    _linestyles = ["-", "--", ":", "-."]

    def _metric_series(metric: str):
        if metric.endswith(" (compartment)"):
            return metric.removesuffix(" (compartment)"), False, False
        if metric.endswith(" (cumulative)"):
            _name = metric.removesuffix(" (cumulative)")
            return _name, True, _name in epi_curve_composite_metrics
        _name = metric.removesuffix(" (daily)")
        return _name, False, _name in epi_curve_composite_metrics

    def _select(counts2d, entity):
        if entity == "All":
            return counts2d.sum(axis=1), epi_curve_population.sum()
        idx = _labels.index(entity)
        return counts2d[:, idx], epi_curve_population[idx]

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _n_colors = len(epi_curve_datasets) * len(epi_curve_age_selector.value)
    _colors = plt.cm.tab10.colors if _n_colors <= 10 else plt.cm.tab20.colors

    for _metric_idx, _metric in enumerate(epi_curve_metric_selector.value):
        _metric_name, _cumulative, _composite = _metric_series(_metric)
        _linestyle = _linestyles[_metric_idx % len(_linestyles)]
        _color_idx = 0
        for _scen_name, _ds in epi_curve_datasets.items():
            _dates = _ds["day"].to_numpy()
            if _composite:
                _part_a, _part_b = epi_curve_composite_metrics[_metric_name]
                _counts2d = _ds[_part_a].to_numpy() + _ds[_part_b].to_numpy()
            else:
                _counts2d = _ds[_metric_name].to_numpy()
            for _entity in epi_curve_age_selector.value:
                _values, _pop = _select(_counts2d, _entity)
                _series = (_values.cumsum() if _cumulative else _values) / _pop
                _ax.plot(
                    _dates, _series * 100,
                    color=_colors[_color_idx % len(_colors)],
                    linestyle=_linestyle,
                    label=f"{_scen_name} — {_entity} — {_metric}",
                )
                _color_idx += 1

    _ax.set_ylabel("% of population")
    _ax.set_xlabel("date")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper left", fontsize=7)
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Epi curves by scenario"), _fig])
    return


if __name__ == "__main__":
    app.run()
