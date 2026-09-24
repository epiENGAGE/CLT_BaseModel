"""
  marimo run generic_core/examples/MA_vax/MA_vax_single_age/counterfactual_notebook_generic_single_age.py
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
    import counterfactual_generic_single_age as cf

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
    # Counterfactual vaccination-impact tables (generic_core, single age group)

    Single-age-group ("0+") counterpart to
    `../counterfactual_notebook_generic.py` -- same PNAS 2505175122
    Supplementary Table structure, but for `MA_vax_single_age/`'s one-age-group
    model instead of the 7-age-group MA_vax model. Built through
    `counterfactual_generic_single_age.py` in this folder (this notebook's
    live-simulation engine) and
    `build_counterfactual_tables_from_db_single_age.py` (the S.A.* table CSVs
    loaded below).

    **Tables S.A.2 and S.A.3 are not shown here** -- they're defined by a
    per-age-group breakdown ("vaccinate age X only", "scale age X to 70%
    coverage") that has no equivalent with a single age group. See the
    comment above `SCENARIOS` in `run_simulations_MA_vax_single_age.py` and
    the module docstrings of the two scripts above.

    This notebook only **loads and displays** the S.A.* results that were
    already computed by

    ```
    python generic_core/examples/MA_vax/MA_vax_single_age/run_simulations_MA_vax_single_age.py
    python generic_core/examples/MA_vax/MA_vax_single_age/build_counterfactual_tables_from_db_single_age.py
    ```

    Point it at that script's output folder below. Re-running it and pointing
    at the new folder is how you refresh the numbers -- this notebook itself
    never runs the S.A.* table simulations, so it stays fast regardless of
    how many replicates that run used (the epi-curve and coverage sections
    further down *do* run fresh deterministic simulations directly, since
    those are cheap).

    **70%-coverage caveat:** the coverage-target scenario uses a single naive
    cross-product multiplier (`target / baseline_coverage`) with no
    bisection-refinement or verification -- see the "Vaccination coverage"
    section below for how far off 70% it actually lands.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    results_folder = mo.ui.text(
        value="generic_core/examples/MA_vax/MA_vax_single_age/counterfactual_tables_from_db_single_age",
        placeholder="generic_core/examples/MA_vax/MA_vax_single_age/counterfactual_tables_from_db_single_age",
        label="Results folder (from build_counterfactual_tables_from_db_single_age.py)",
        full_width=True,
    )
    results_folder
    return (results_folder,)


@app.cell
def _(cf, mo, os, results_folder):
    mo.stop(
        not results_folder.value,
        mo.md("Enter a results folder produced by `build_counterfactual_tables_from_db_single_age.py` above."),
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
        f"**age_groups:** `{meta.get('age_groups')}` &nbsp;·&nbsp; "
        f"**n_reps:** `{meta.get('n_reps')}` &nbsp;·&nbsp; "
        f"**generated:** `{meta.get('generated_at')}`"
    ) if meta else mo.md("_No `meta.json` found in this folder._")
    return (tables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.1 — Hospitalizations averted, infection vs. severity protection

    Compares *no vaccination* -> *infection-protection-only* (VE against
    infection retained, VE against severity zeroed out) -> *baseline*
    (full VE).
    """)
    return


@app.cell
def _(show_table, tables):
    show_table(tables["S_A_1"], "S_A_1.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Table S.A.4 — VE sensitivity scenarios

    Implied vaccine effectiveness against infection and against
    hospitalization-given-infection for each VE preset -- absolute
    `vax_susceptibility`/`IV_to_H_prop` overrides copied from
    `run_simulations_MA_vax_single_age.py`'s `SCENARIOS` dict (that script's
    source of truth for the single-age-refit presets), not ratios on the
    fitted baseline.
    """)
    return


@app.cell
def _(show_table, tables):
    show_table(tables["S_A_4"], "S_A_4.csv", page_size=25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dose accounting — scheduled vs. delivered

    Doses the baseline schedule ships (`scheduled_doses`) vs. doses the model
    actually delivers (`S_to_SV`, median across replicates) -- the gap is
    doses the §1.4 cap declines to hand out because the intended recipient
    was already infected. See
    `build_counterfactual_tables_from_db_single_age.table_dose_accounting`'s
    docstring.
    """)
    return


@app.cell
def _(show_table, tables):
    show_table(tables.get("DOSE_ACCOUNTING"), "DOSE_ACCOUNTING.csv")
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
    vaccination to (naive) 70% coverage.
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

    Instead of population-level hospitalizations averted, these look
    directly at the model's internal flows to confirm vaccination is doing
    what it should. All three entries are rate ratios (vaccinated rate /
    unvaccinated rate) shown as a percentage -- below 100% means vaccination
    is reducing that risk. A single "All" column, since there's no per-age
    breakdown to compare it against. See
    `MA_vax_standalone/counterfactual.py`'s `table_vax_efficacy_check`
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

    Cumulative proportion of the population vaccinated (`S_to_SV` summed over
    the season, divided by population) under the baseline schedule, versus
    the "70% coverage" scenario. Since the coverage multiplier here is a
    naive cross-product estimate (not bisection-refined -- see the notebook
    intro), this is also the check for **how far off 70% the scaled scenario
    actually lands**.
    """)
    return


@app.cell
def _(cf, pd):
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
    ## Matched-cohort attack-probability curve

    For the baseline vaccination schedule, `attack_SV_from(d)` (solid) and
    `attack_S_from(d)` (dashed) -- the counterfactual season-end attack
    probability for a hypothetical individual entering `SV`/`S` on day `d`
    and followed to season end. See
    `MA_vax_standalone/counterfactual.py`'s `table_vax_efficacy_check`'s
    docstring (the "matched_cohort_infection_reduction" entry) for the full
    derivation -- `attack_probability_curves` is reused unchanged from there,
    since it only operates on already-simulated (reps, day, age) arrays.
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
    # Before the first-ever vaccination day, SV=0 and attack_SV_from(d) falls back to a
    # "zero hazard that day" placeholder -- meaningless for a day nobody was actually
    # vaccinated on, so mask it out of the plot.
    attack_SV_from = np.where(_d["SV"][0] > 0, attack_SV_from, np.nan)
    return attack_S_from, attack_SV_from, attack_curve_dates


@app.cell
def _(attack_S_from, attack_SV_from, attack_curve_dates, cf, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _ax.plot(attack_curve_dates, attack_SV_from[:, 0] * 100, label="attack_SV_from(d)",
             color="C1", linewidth=1.5)
    _ax.plot(attack_curve_dates, attack_S_from[:, 0] * 100, label="attack_S_from(d)",
             color="black", linewidth=1, linestyle="--")
    _ax.set_title(cf.AGE_GROUPS[0])
    _ax.set_ylabel("attack probability (%)")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper right")
    _fig.suptitle("Matched-cohort attack-probability curve (baseline schedule)")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### attack_SV_from(d) vs. attack_S_from(d)"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline fit check — simulated vs. raw hospitalizations

    Compares a fresh deterministic **baseline-vaccination** simulation
    (re-run here, not one of the loaded S.A.* tables) to the raw
    total daily hospitalization series in
    `MA_flu_daily_hospitalizations_total.csv` (shared from
    `../data/hospitalizations_ts/` -- this folder has no per-age breakdown to
    compare against, so it reads the same total series
    `run_simulations_MA_vax_single_age.py`'s own fit used), over the dates
    the two series have in common. This is a fit-quality check, separate
    from the counterfactual tables above, and specifically a check that the
    single-age-group export tracks the fitted data as well as the fit config
    it was refit from (`fit_config_MA_single_age.json`) did.
    """)
    return


@app.cell
def _(cf, os, pd):
    _base_inputs = cf.load_base_inputs()
    _ds = cf._run_reps(_base_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False)
    sim_new_H = (_ds["I_to_H"] + _ds["IV_to_H"]).isel(replication=0).to_pandas()
    sim_new_H.index.name = "date"
    sim_new_H.columns = ["total"]

    raw_H = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "hospitalizations_ts",
                     "MA_flu_daily_hospitalizations_total.csv")
    )
    raw_H["date"] = pd.to_datetime(raw_H["Date"])
    raw_H = raw_H.set_index("date")[["total"]]
    return raw_H, sim_new_H


@app.cell
def _(mo, pd, raw_H, show_table, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_H.index)
    _sim_cum = sim_new_H.loc[_common_dates, "total"]
    _raw_cum = raw_H.loc[_common_dates, "total"]
    _sim_total, _raw_total = float(_sim_cum.sum()), float(_raw_cum.sum())
    _pct_diff = (_sim_total - _raw_total) / _raw_total * 100 if _raw_total else float("nan")

    cum_hosp_table = pd.DataFrame({
        "simulated": [round(_sim_total, 1)],
        "raw_data": [round(_raw_total, 1)],
        "pct_diff": [round(_pct_diff, 1)],
    }, index=pd.Index(["total"], name="series"))

    mo.vstack([
        mo.md(
            f"**Cumulative hospitalizations, {_common_dates.min().date()} to "
            f"{_common_dates.max().date()}** ({len(_common_dates)} days common to both series)"
        ),
        show_table(cum_hosp_table, "cumulative_hospitalizations.csv"),
    ])
    return


@app.cell
def _(mo, plt, raw_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_H.index)
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _ax.plot(_common_dates, raw_H.loc[_common_dates, "total"], label="Raw data",
             color="black", linewidth=1)
    _ax.plot(_common_dates, sim_new_H.loc[_common_dates, "total"], label="Simulated (baseline)",
             color="C1", linewidth=1.5)
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper right")
    _fig.suptitle("Daily new hospitalizations: simulated (baseline) vs. raw data")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Actual vs. simulated daily hospitalizations"), _fig])
    return


@app.cell
def _(mo, plt, raw_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_H.index)
    _fig, _ax = plt.subplots(figsize=(9, 4.5))
    _ax.plot(_common_dates, raw_H.loc[_common_dates, "total"].cumsum(), label="Raw data",
             color="black", linewidth=1)
    _ax.plot(_common_dates, sim_new_H.loc[_common_dates, "total"].cumsum(), label="Simulated (baseline)",
             color="C1", linewidth=1.5)
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="upper right")
    _fig.suptitle("Cumulative hospitalizations: simulated (baseline) vs. raw data")
    _fig.autofmt_xdate()
    plt.tight_layout()
    mo.vstack([mo.md("### Actual vs. simulated cumulative hospitalizations"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vaccination vs. epidemic timing (interactive)

    Compare the vaccination time series against epidemic-curve metrics to
    see how vaccination timing lines up with the epidemic. Both series are
    shown as **proportion of population** -- vaccination and the epidemic
    metric are on separate y-axes (left = epidemic metric, right =
    vaccination), since their typical magnitudes differ a lot.

    Both series come from the same deterministic baseline simulation as the
    "Baseline fit check" section above. Two vaccination sources are available
    (select either or both):

    - **Simulated (`S_to_SV`)** -- doses that actually landed in the model,
      after the `scheduled_exact` transition's own capping against available
      `S`.
    - **Raw input schedule** -- the uploaded `daily_vaccines_df`'s per-day
      proportion (from `schedules.json`, already a proportion of the
      population), aligned to the simulation's date range. This is the
      *nominal* schedule as given, before any capping the model's transition
      engine applies -- it diverges from the simulated series wherever that
      capping actually binds.
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
        [vax_timing_metric_selector, vax_timing_source_selector, vax_timing_mode_selector],
        justify="start", gap=2,
    )
    return vax_timing_metric_selector, vax_timing_mode_selector, vax_timing_source_selector


@app.cell
def _(
    cf,
    mo,
    plt,
    vax_timing_ds,
    vax_timing_mode_selector,
    vax_timing_metric_selector,
    vax_timing_source_selector,
    vax_timing_population,
    vax_timing_vax_arr,
):
    mo.stop(
        not vax_timing_source_selector.value,
        mo.md("Select at least one vaccination source above.").callout(kind="warn"),
    )

    _dates = vax_timing_ds["day"].to_numpy()

    def _select_np(counts2d):
        return counts2d.sum(axis=1), vax_timing_population.sum()

    def _select(da):
        return _select_np(da.to_numpy())

    # vax_timing_vax_arr is already a proportion of the population (see
    # daily_vaccines_df in schedules.json), so recover counts by multiplying back
    # by population -- no capping here, this is the raw nominal schedule, uncapped
    # by the model's `scheduled_exact` transition (unlike S_to_SV, which is capped).
    _raw_implied_counts = vax_timing_vax_arr * vax_timing_population[None, :]

    _vax_sources = {
        "Simulated (S_to_SV)": lambda: _select(vax_timing_ds["S_to_SV"]),
        "Raw input schedule (vax_arr)": lambda: _select_np(_raw_implied_counts),
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

    _metric_counts, _metric_pop = _select(_metric_da)
    _metric_series = (_metric_counts.cumsum() if _metric_cumulative else _metric_counts) / _metric_pop
    _h1, = _ax1.plot(_dates, _metric_series, color="C0", linestyle="-", label=_metric)
    _handles = [_h1]

    _vax_cumulative = vax_timing_mode_selector.value == "cumulative"
    for _color_idx, _source in enumerate(vax_timing_source_selector.value):
        _vax_counts, _vax_pop = _vax_sources[_source]()
        _vax_series = (_vax_counts.cumsum() if _vax_cumulative else _vax_counts) / _vax_pop
        _h2, = _ax2.plot(_dates, _vax_series, color=f"C{_color_idx + 1}", linestyle="-",
                          label=f"{_source} ({vax_timing_mode_selector.value})")
        _handles.append(_h2)

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
    across the scenarios in `counterfactual_generic_single_age.py`. Each
    scenario is a fresh deterministic simulation built via `cf.build_model`/
    `cf._run_reps` directly (not one of the loaded S.A.* tables). Values
    shown are proportion of population.
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
        "Infection protection only": cf.infection_protection_only_scenario(_base_inputs),
        "Low VE": cf.ve_scenarios(_base_inputs)["low_ve"],
        "High VE": cf.ve_scenarios(_base_inputs)["high_ve"],
        "70% coverage": cf.coverage_70pct_scenario(_base_inputs, None),
    }
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
    epi_curve_metric_selector = mo.ui.multiselect(
        options=_metric_options,
        value=["H (compartment)"],
        label="Metric(s)",
    )
    mo.hstack(
        [epi_curve_scenario_selector, epi_curve_metric_selector],
        justify="start", gap=2,
    )
    return epi_curve_metric_selector, epi_curve_scenario_selector


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
        not epi_curve_metric_selector.value,
        mo.md("Select at least one metric above.").callout(kind="warn"),
    )

    _linestyles = ["-", "--", ":", "-."]

    def _metric_series(metric: str):
        if metric.endswith(" (compartment)"):
            return metric.removesuffix(" (compartment)"), False, False
        if metric.endswith(" (cumulative)"):
            _name = metric.removesuffix(" (cumulative)")
            return _name, True, _name in epi_curve_composite_metrics
        _name = metric.removesuffix(" (daily)")
        return _name, False, _name in epi_curve_composite_metrics

    _fig, _ax = plt.subplots(figsize=(12, 5))
    _n_colors = len(epi_curve_datasets)
    _colors = plt.cm.tab10.colors if _n_colors <= 10 else plt.cm.tab20.colors

    for _metric_idx, _metric in enumerate(epi_curve_metric_selector.value):
        _metric_name, _cumulative, _composite = _metric_series(_metric)
        _linestyle = _linestyles[_metric_idx % len(_linestyles)]
        for _color_idx, (_scen_name, _ds) in enumerate(epi_curve_datasets.items()):
            _dates = _ds["day"].to_numpy()
            if _composite:
                _part_a, _part_b = epi_curve_composite_metrics[_metric_name]
                _counts2d = _ds[_part_a].to_numpy() + _ds[_part_b].to_numpy()
            else:
                _counts2d = _ds[_metric_name].to_numpy()
            _values = _counts2d.sum(axis=1)
            _series = (_values.cumsum() if _cumulative else _values) / epi_curve_population.sum()
            _ax.plot(
                _dates, _series * 100,
                color=_colors[_color_idx % len(_colors)],
                linestyle=_linestyle,
                label=f"{_scen_name} — {_metric}",
            )

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
