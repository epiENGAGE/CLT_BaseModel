"""
  marimo run MA_vax/counterfactual_notebook.py
"""


import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from MA_vax import counterfactual as cf
    from MA_vax import model

    return cf, mo, model, np, os, pd, plt


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
    # Counterfactual vaccination-impact tables

    Replicates the structure of PNAS 2505175122 Supplementary Tables
    S.A.1-S.A.6, adapted to this model's 7 age groups, the Massachusetts
    population, and the 2025-2026 season. Scenario definitions live in
    `MA_vax/counterfactual.py` (`single_age_only_scenario`,
    `coverage_70pct_scenario`, `VE_SCENARIOS`, ...) and can be edited there
    without touching the table-computation logic.

    This notebook only **loads and displays** results that were already
    computed by

    ```
    uv run python MA_vax/run_counterfactual_tables.py --n-reps 1000
    ```

    Point it at that run's output folder below. Re-running the CLI script
    (e.g. with more replications for tighter confidence intervals) and
    pointing at the new folder is how you refresh the numbers — this
    notebook itself never runs a simulation, so it stays fast regardless of
    `n_reps`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    results_folder = mo.ui.text(
        value="MA_vax/counterfactual_tables_det",
        placeholder="MA_vax/counterfactual_tables_det",
        label="Results folder (from run_counterfactual_tables.py)",
        full_width=True,
    )
    results_folder
    return (results_folder,)


@app.cell
def _(cf, mo, os, results_folder):
    mo.stop(
        not results_folder.value,
        mo.md("Enter a results folder produced by `run_counterfactual_tables.py` above."),
    )
    mo.stop(
        not os.path.isdir(results_folder.value),
        mo.md(
            f"⚠️ **`{results_folder.value}` is not a folder.** "
            "Paths are relative to wherever `marimo edit`/`marimo run` was launched from "
            "(usually the repo root) — a leading `/` makes it absolute from the filesystem "
            "root instead, which is a common way to hit this."
        ).callout(kind="danger"),
    )
    tables = cf.load_saved_tables(results_folder.value)
    meta = tables["meta"]
    fit_point = meta.get("point", "best")
    mo.md(
        f"**fit_folder:** `{meta.get('fit_folder') or 'default_params()'}` &nbsp;·&nbsp; "
        f"**method:** `{meta.get('method')}` &nbsp;·&nbsp; "
        f"**point:** `{fit_point}` &nbsp;·&nbsp; "
        f"**n_reps:** `{meta.get('n_reps')}` &nbsp;·&nbsp; "
        f"**seed:** `{meta.get('seed')}` &nbsp;·&nbsp; "
        f"**generated:** `{meta.get('generated_at')}`"
    ) if meta else mo.md("_No `meta.json` found in this folder._")
    return fit_point, tables


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
    70% cumulative coverage (all others left at baseline); "All" scales
    every age group to 70%. Compared against the baseline vaccination
    scenario.
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
    hospitalization-given-infection for each preset in
    `cf.VE_SCENARIOS`. **Placeholder multipliers** — tune
    `VE_SCENARIOS` in `counterfactual.py` to match real low/high VE
    bounds; a naive uniform scale can push `IV_to_H_prop` above
    `I_to_H_prop` for some age groups (negative VE), as seen below for
    `low_ve`.
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

    For each VE sensitivity scenario, compares that scenario's own
    baseline vaccination to 70% coverage in every age group.
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
    ## Vaccination coverage — baseline vs. 70%-coverage scenarios

    Cumulative proportion of each age group's population vaccinated
    (`S_to_SV` summed over the season, divided by population) under the
    baseline schedule, versus the "All ages -> 70%" scenario
    (`cf.coverage_70pct_scenario(inputs, None)`) used as the "All" column of
    Table S.A.3 and as the sole scenario in Table S.A.6. VE-scenario scaling
    (`VE_SCENARIOS`) only rescales `vax_susceptibility`/`IV_to_H_prop`, not
    the vaccination schedule itself, so this same 70% coverage applies
    across every column of Table S.A.6 — it isn't recomputed per VE
    scenario.

    Baseline coverage is well under 70% in every age group (this model's
    fitted 2025-2026 season schedule), which is why scaling every group up
    to 70% in Table S.A.6 averts such a large share of remaining
    hospitalizations — it's closing a large absolute coverage gap, not
    adding a small margin on an already-near-saturated baseline.
    """)
    return


@app.cell
def _(cf, fit_point, model, np, pd, tables):
    _meta = tables["meta"]
    _fit_folder = _meta.get("fit_folder")
    _method = _meta.get("method") or "emcee"
    _base_inputs = cf.load_base_inputs(_fit_folder, _method, fit_point)
    _target70_inputs = model.apply_scenario(_base_inputs, cf.coverage_70pct_scenario(_base_inputs, None))

    _pop = _base_inputs["population"]
    _baseline_cov = [cf._cumulative_coverage(_base_inputs, i) for i in range(len(cf.AGE_GROUPS))]
    _target70_cov = [cf._cumulative_coverage(_target70_inputs, i) for i in range(len(cf.AGE_GROUPS))]

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
    ## Vaccine-efficacy mechanism check

    Same layout as Table S.A.2 (columns = age group vaccinated, "All" =
    baseline schedule; rows = age group the flow is counted in), but instead
    of population-level hospitalizations averted, these look directly at the
    model's internal flows to confirm vaccination is doing what it should.
    All three entries are rate ratios (vaccinated rate / unvaccinated rate)
    shown as a percentage — below 100% means vaccination is reducing that
    risk. Off-diagonal entries are `—` (undefined) except in the "All"
    column, since `single_age_only_scenario` only gives the vaccinated age
    group any `SV`/`IV` population.

    **Infection reduction** — season-total attack-rate ratio, each rate's
    denominator being the pool ever at risk of that flow over the season:

    $$\frac{\mathrm{sum(SV\_to\_EV)} \,/\, (\mathrm{SV}[0] + \mathrm{sum(S\_to\_SV)})}{\mathrm{sum(S\_to\_E)} \,/\, (\mathrm{S}[0] - \mathrm{sum(S\_to\_SV)})}$$

    **This formula is biased** ("immortal-time bias") and is kept only as a
    *"what a naive analysis would show"* comparison point: vaccination is
    staggered across the season, so the `SV` cohort's average follow-up
    window (from whenever each person was vaccinated to season end) is
    shorter than the reference `S[0]` cohort's full-season window. Checked
    against a case with **zero** modeled infection protection
    (`vax_susceptibility=1.0`, the 65+ age group): this formula still
    reports a spurious reduction purely from that timing effect, even
    though there's nothing to detect.

    **Infection reduction (matched-cohort)** — the bias-corrected version,
    and the one actually comparable to real-world reported VE (RCTs
    synchronize enrollment; surveillance/cohort studies use vaccination as
    a time-varying covariate specifically to avoid this same immortal-time
    bias). Uses the fact that in this model the per-capita daily hazard is
    a pure function of calendar day `t`, not of compartment size:

    $$\mathrm{haz}_S(t) = \frac{\mathrm{S\_to\_E}(t)}{\mathrm{S}(t)} = \mathrm{foi}(t)\cdot\mathrm{rel\_suscept}\cdot dt
    \qquad
    \mathrm{haz}_{SV}(t) = \frac{\mathrm{SV\_to\_EV}(t)}{\mathrm{SV}(t)} = \mathrm{foi}(t)\cdot\mathrm{vax\_susceptibility}\cdot dt$$

    So for any day `d`, the counterfactual season-end attack probability
    for a hypothetical individual entering the pool on day `d` and followed
    to season end `T` is:

    $$\mathrm{attack}_S(d) = 1 - \prod_{t=d}^{T-1}\bigl(1-\mathrm{haz}_S(t)\bigr)
    \qquad
    \mathrm{attack}_{SV}(d) = 1 - \prod_{t=d}^{T-1}\bigl(1-\mathrm{haz}_{SV}(t)\bigr)$$

    Weighting across the *actual* vaccination-day distribution using
    `S_to_SV(d)` (how many people were actually vaccinated on day `d`) and
    averaging gives a season-total attack-rate ratio with matched start
    days between the two arms — eliminating the immortal-time bias:

    $$\frac{\sum_d \mathrm{S\_to\_SV}(d)\cdot\mathrm{attack}_{SV}(d) \,/\, \sum_d \mathrm{S\_to\_SV}(d)}
    {\sum_d \mathrm{S\_to\_SV}(d)\cdot\mathrm{attack}_S(d) \,/\, \sum_d \mathrm{S\_to\_SV}(d)}$$

    For the 65+ zero-protection case above, this correctly comes out at
    100% (no artifact) — verified against the biased formula's spurious
    reduction in the same scenario.

    **Hospitalization reduction** — rate ratio of hospitalization-given-
    infection, directly comparable to `IV_to_H_prop / I_to_H_prop` in Table
    S.A.4:

    $$\frac{\mathrm{sum(IV\_to\_H)} \,/\, \mathrm{sum(SV\_to\_EV)}}{\mathrm{sum(I\_to\_H)} \,/\, \mathrm{sum(S\_to\_E)}}$$

    This one needs no timing correction: `I_to_H_prop`/`IV_to_H_prop` are
    constant rates conditional on already being infected, not driven by the
    time-varying force of infection the way `S_to_E`/`SV_to_EV` are, so
    there's no staggered-entry confound here.
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
    ## Matched-cohort attack-probability curves

    For the baseline vaccination schedule, `attack_SV_from(d)` (solid) and
    `attack_S_from(d)` (dashed) by age group — the counterfactual season-end
    attack probability for a hypothetical individual entering `SV`/`S` on
    day `d` and followed to season end, as defined in the "Vaccine-efficacy
    mechanism check" section above. Where the two lines overlap, vaccination
    confers no infection protection that day (as with the 65+ group, where
    `vax_susceptibility=1.0`); where the solid line sits below the dashed
    line, the gap is the actual protective effect for someone vaccinated on
    that day. The `matched_cohort_infection_reduction` table entries are a
    single `S_to_SV(d)`-weighted summary of the gap between these two
    curves across the whole season.

    Lines stop shortly before season end since `attack_*_from(d)` for `d`
    near `T` is computed from very little remaining follow-up time and gets
    noisy; deterministic single-replication run (baseline schedule).
    """)
    return


@app.cell
def _(cf, fit_point, np, tables):
    _meta = tables["meta"]
    _fit_folder = _meta.get("fit_folder")
    _method = _meta.get("method") or "emcee"
    _attack_curve_inputs = cf.load_base_inputs(_fit_folder, _method, fit_point)
    _attack_curve_dates = _attack_curve_inputs["dates"]

    _d = cf._scenario_daily_arrays(_attack_curve_inputs, cf.baseline_scenario(), n_reps=1, seed=0, stochastic=False)
    _attack_S_from, _attack_SV_from = cf.attack_probability_curves(
        _d["S"], _d["SV"], _d["S_to_E"], _d["SV_to_EV"])

    attack_curve_dates = _attack_curve_dates
    attack_S_from = _attack_S_from[0]   # (day, A), single deterministic replication
    attack_SV_from = _attack_SV_from[0]
    # Before an age group's first-ever vaccination day, SV=0 and attack_SV_from(d) falls
    # back to a "zero hazard that day" placeholder (nobody there to face one) -- meaningless
    # for a day nobody was actually vaccinated on, so mask it out of the plot.
    attack_SV_from = np.where(_d["SV"][0] > 0, attack_SV_from, np.nan)
    return attack_curve_dates, attack_S_from, attack_SV_from


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
    (re-run here, not one of the loaded `S_A_*` tables) to the raw per-age
    daily hospitalization series in
    `MA_flu_daily_hospitalizations.csv` (from `model.DATA_FOLDER`, i.e.
    `generic_core/examples/massachusetts_vax/`), over the dates the two
    series have in common. This is a fit-quality check, separate from the
    counterfactual tables above, but it deliberately reuses the exact same
    `fit_folder`/`method`/**point estimate** as those tables (from
    `tables["meta"]`, shown above) rather than letting this section pick its
    own — so this simulation and the S.A.* tables above are always built
    from identical baseline inputs.

    **Point estimate:** `"mean"` averages each fitted parameter independently
    across posterior draws, which can land on a parameter *combination* the
    sampler never actually visited if parameters are correlated (this is why
    a mean-parameter run can peak much higher than the observed data even
    when `fit_posterior_predictive.png`'s median-of-draws curve tracks it
    closely). `"best"` instead uses the single sampled parameter vector with
    the highest posterior support, preserving that correlation structure —
    it's the default for `run_counterfactual_tables.py`, but requires a
    `fit_folder` produced after `fit_bayesian.py` started saving best-fit
    CSVs; older folders will show a warning below instead of erroring.
    """)
    return


@app.cell
def _(cf, fit_point, mo, model, tables):
    _meta = tables["meta"]
    _fit_folder = _meta.get("fit_folder")
    _method = _meta.get("method") or "emcee"
    try:
        base_inputs = cf.load_base_inputs(_fit_folder, _method, fit_point)
    except FileNotFoundError as _exc:
        mo.stop(True, mo.md(f"⚠️ {_exc}").callout(kind="warn"))
    _ds = model.simulate_detailed(base_inputs, stochastic=False, n_reps=1)
    sim_new_H = (_ds["I_to_H"] + _ds["IV_to_H"]).isel(replication=0).to_pandas()
    sim_new_H.index.name = "date"
    return (sim_new_H,)


@app.cell
def _(model, os, pd):
    raw_age_H = pd.read_csv(
        os.path.join(model.DATA_FOLDER, "MA_flu_daily_hospitalizations.csv")
    )
    raw_age_H["date"] = pd.to_datetime(raw_age_H["Date"])
    raw_age_H = raw_age_H.set_index("date").drop(columns=["Date"])
    raw_age_H.columns = model.AGE_GROUP_LABELS
    return (raw_age_H,)


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
def _(mo, model, plt, raw_age_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_age_H.index)
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    _axes = _axes.flatten()
    for _i, _age in enumerate(model.AGE_GROUP_LABELS):
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
def _(mo, model, plt, raw_age_H, sim_new_H):
    _common_dates = sim_new_H.index.intersection(raw_age_H.index)
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    _axes = _axes.flatten()
    for _i, _age in enumerate(model.AGE_GROUP_LABELS):
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
    scale as each other — though vaccination and the epidemic metric are on
    separate y-axes (left = epidemic metric, right = vaccination), since
    their typical magnitudes differ a lot.

    Both series come from the same deterministic baseline simulation as the
    "Baseline fit check" section above (same `fit_folder`/`method`/`point`).
    Two vaccination sources are available (select either or both):

    - **Simulated (`S_to_SV`)** — doses that actually landed in the model,
      after the 14-day delay-shift and susceptible-pool capping.
    - **Raw input schedule (`vax_arr`)** — the input CSV's daily proportion
      (`MA_flu_daily_vaccinations_proportions_array.csv` — already a
      proportion of that age group's total population, not of the
      susceptible pool), delay-shifted to line up with the simulation dates.
      This is the *nominal* schedule as given, before the model's
      `vax_pool="susceptible"` capping is applied — it diverges from the
      simulated series wherever that capping actually binds (which can be
      substantial once S is significantly depleted, unlike the raw schedule
      which has no such ceiling). For the "All" entity, both sources are
      population-weighted (summed counts / summed population), not an
      average of per-age proportions.
    """)
    return


@app.cell
def _(mo, model):
    _metric_options = (
        [f"{c} (compartment)" for c in model.COMPARTMENTS]
        + [f"{t} (daily)" for t in model.TRANSITIONS]
        + [f"{t} (cumulative)" for t in model.TRANSITIONS]
    )
    vax_timing_age_selector = mo.ui.multiselect(
        options=model.AGE_GROUP_LABELS + ["All"],
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
def _(cf, fit_point, model, tables):
    _meta = tables["meta"]
    _fit_folder = _meta.get("fit_folder")
    _method = _meta.get("method") or "emcee"
    _vax_timing_inputs = cf.load_base_inputs(_fit_folder, _method, fit_point)
    vax_timing_ds = model.simulate_detailed(_vax_timing_inputs, stochastic=False, n_reps=1)
    vax_timing_population = _vax_timing_inputs["population"]
    vax_timing_vax_arr = _vax_timing_inputs["vax_arr"]
    return vax_timing_ds, vax_timing_population, vax_timing_vax_arr


@app.cell
def _(
    model,
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

    _ds = vax_timing_ds.isel(replication=0)
    _dates = _ds["day"].to_numpy()
    _labels = model.AGE_GROUP_LABELS

    def _select_np(counts2d, entity):
        """Counts (summed over `entity`'s age groups) and matching population,
        for a plain (day, age_group) numpy array."""
        if entity == "All":
            return counts2d.sum(axis=1), vax_timing_population.sum()
        idx = _labels.index(entity)
        return counts2d[:, idx], vax_timing_population[idx]

    def _select(da, entity):
        """Counts (summed over `entity`'s age groups) and matching population,
        for an (day, age_group) xarray DataArray."""
        return _select_np(da.to_numpy(), entity)

    # `vax_arr` is already a proportion of each age group's total population
    # (see MA_flu_daily_vaccinations_proportions_array.csv), so recover
    # counts by multiplying back by population -- no S-based rescaling here,
    # this is the raw nominal schedule, uncapped by the model's internal
    # susceptible-pool depletion (unlike `S_to_SV`, which is capped by it).
    _raw_implied_counts = vax_timing_vax_arr * vax_timing_population[None, :]

    _vax_sources = {
        "Simulated (S_to_SV)": lambda entity: _select(_ds["S_to_SV"], entity),
        "Raw input schedule (vax_arr)": lambda entity: _select_np(_raw_implied_counts, entity),
    }

    _metric = vax_timing_metric_selector.value
    if _metric.endswith(" (compartment)"):
        _metric_da, _metric_cumulative = _ds[_metric.removesuffix(" (compartment)")], False
    elif _metric.endswith(" (cumulative)"):
        _metric_da, _metric_cumulative = _ds[_metric.removesuffix(" (cumulative)")], True
    else:
        _metric_da, _metric_cumulative = _ds[_metric.removesuffix(" (daily)")], False

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
    for any age group (or "All"), across the scenarios already defined in
    `MA_vax/counterfactual.py`:

    - **Baseline (fitted vaccination)** / **No vaccination**
    - **Low VE** / **High VE** — `cf.VE_SCENARIOS["low_ve"/"high_ve"]`
    - **70% coverage (all ages)** — `cf.coverage_70pct_scenario(base_inputs,
      None)`, every age group's schedule scaled to reach 70% cumulative
      coverage
    - **Vaccinate `<age>` only** — `cf.single_age_only_scenario`, one per age
      group, all others left unvaccinated

    Several metrics can be selected at once — each metric gets its own line
    style (solid/dashed/dotted/dash-dot), while each scenario x age-group
    combination gets its own color. Besides the raw compartments/transitions,
    a few **combined vaccinated + unvaccinated** metrics are offered (e.g.
    `S_to_E + SV_to_EV` = total new infections regardless of vaccination
    status), since the raw `model.TRANSITIONS` split every flow by
    vaccination status.

    Each scenario is a fresh deterministic simulation (not one of the loaded
    `S_A_*` tables), built from the same `fit_folder`/`method`/`point` as the
    "Baseline fit check" section above (`base_inputs`, from `tables["meta"]`).
    Values shown are proportion of population, so different-sized age groups
    / scenarios are directly comparable.
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
def _(base_inputs, cf, model):
    epi_curve_scenarios = {
        "Baseline (fitted vaccination)": cf.baseline_scenario(),
        "No vaccination": cf.no_vaccine_scenario(),
        "Low VE": cf.VE_SCENARIOS["low_ve"],
        "High VE": cf.VE_SCENARIOS["high_ve"],
        "70% coverage (all ages)": cf.coverage_70pct_scenario(base_inputs, None),
    }
    for _i, _label in enumerate(model.AGE_GROUP_LABELS):
        epi_curve_scenarios[f"Vaccinate {_label} only"] = cf.single_age_only_scenario(_i)
    return (epi_curve_scenarios,)


@app.cell
def _(epi_curve_composite_metrics, epi_curve_scenarios, mo, model):
    _metric_options = (
        [f"{c} (compartment)" for c in model.COMPARTMENTS]
        + [f"{t} (daily)" for t in model.TRANSITIONS]
        + [f"{t} (cumulative)" for t in model.TRANSITIONS]
        + [f"{name} (daily)" for name in epi_curve_composite_metrics]
        + [f"{name} (cumulative)" for name in epi_curve_composite_metrics]
    )
    epi_curve_scenario_selector = mo.ui.multiselect(
        options=list(epi_curve_scenarios.keys()),
        value=["Baseline (fitted vaccination)", "No vaccination"],
        label="Scenario(s)",
    )
    epi_curve_age_selector = mo.ui.multiselect(
        options=model.AGE_GROUP_LABELS + ["All"],
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
    base_inputs,
    epi_curve_scenario_selector,
    epi_curve_scenarios,
    model,
):
    epi_curve_datasets = {}
    for _name in epi_curve_scenario_selector.value:
        _inputs = model.apply_scenario(base_inputs, epi_curve_scenarios[_name])
        epi_curve_datasets[_name] = model.simulate_detailed(
            _inputs, stochastic=False, n_reps=1
        ).isel(replication=0)
    epi_curve_population = base_inputs["population"]
    return epi_curve_datasets, epi_curve_population


@app.cell
def _(
    epi_curve_age_selector,
    epi_curve_composite_metrics,
    epi_curve_datasets,
    epi_curve_metric_selector,
    epi_curve_population,
    model,
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

    _labels = model.AGE_GROUP_LABELS
    _linestyles = ["-", "--", ":", "-."]

    def _metric_series(metric: str):
        """(metric_name_or_composite_key, cumulative, is_composite)."""
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
