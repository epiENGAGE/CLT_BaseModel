"""
scenario_and_outcomes_demo.py
==============================

Standalone runnable script demonstrating:

  1. ScenarioRunner — baseline + counterfactual vaccine scenarios, paired
     replicates, SQL-backed results.
  2. flu_outcomes — every computation function and every plotting function.

Run from the project root::

    python flu_instances/examples/scenario_and_outcomes_demo.py

Design note
-----------
``ScenarioRunner`` stores state-variable snapshots in SQLite but does *not*
keep per-replicate transition-variable histories in memory (it resets the
model after each replicate).  ``flu_outcomes`` functions need those in-memory
histories.  The standard pattern is therefore:

  a. Run ``ScenarioRunner`` for multi-replicate, SQL-backed statistics.
  b. Re-run each scenario once (with ``transition_variables_to_save`` set)
     to get an in-memory model whose histories can be passed to
     ``flu_outcomes``.

This script shows both steps clearly.
"""

import datetime
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless — swap to "TkAgg" / remove for interactive

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import clt_toolkit as clt
import flu_core as flu
import flu_core.flu_outcomes as outcomes


class _Tee:
    """Write to both the original stdout and a file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_PATH = clt.utils.PROJECT_ROOT / "flu_instances" / "austin_input_files_2024_2025"
SIMULATION_DAYS = 200
NUM_REPS        = 5      # keep small so the demo runs quickly
SEEDS           = list(range(NUM_REPS))

TVAR_SAVE = ("ISH_to_HR", "ISH_to_HD", "S_to_E", "HD_to_D")


# ---------------------------------------------------------------------------
# Helper: scale a daily_vaccines DataFrame
# ---------------------------------------------------------------------------

def scale_vaccines_df(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Return a scaled copy of a raw daily_vaccines DataFrame.

    The ``daily_vaccines`` column holds JSON-encoded 2-D arrays (one entry
    per age×risk group), so scaling requires JSON round-tripping rather than
    simple arithmetic.  Dates are also converted to ``datetime.date`` objects
    to match the format expected by ``DailyVaccines.postprocess_data_input``
    when called via ``replace_schedule``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw vaccines DataFrame as loaded from CSV.
    scale : float
        Multiplicative factor to apply to every daily_vaccines value.

    Returns
    -------
    pd.DataFrame
    """
    scaled = df.copy()
    scaled["daily_vaccines"] = scaled["daily_vaccines"].apply(
        lambda s: json.dumps((np.array(json.loads(s)) * scale).tolist())
    )
    scaled["date"] = pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.date
    return scaled


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    east_vax_df: pd.DataFrame,
    west_vax_df: pd.DataFrame,
    transition_type: str = "binom_deterministic_no_round",
    save_tvars: bool = True,
    seed: int = 88888,
) -> flu.FluMetapopModel:
    """
    Build a fresh FluMetapopModel from the Austin 2024-2025 inputs.

    Parameters
    ----------
    east_vax_df, west_vax_df : pd.DataFrame
        Vaccine schedules (raw CSV format, possibly scaled).
    transition_type : str
        ``"binom_deterministic_no_round"`` for deterministic runs,
        ``"binom"`` for stochastic.
    save_tvars : bool
        Whether to populate ``transition_variables_to_save`` so that
        ``flu_outcomes`` functions can access histories afterwards.
    seed : int
        RNG seed for the east subpopulation; west uses ``seed + 1``.
    """
    east_state = clt.make_dataclass_from_json(
        BASE_PATH / "init_vals_east.json", flu.FluSubpopState
    )
    west_state = clt.make_dataclass_from_json(
        BASE_PATH / "init_vals_west.json", flu.FluSubpopState
    )
    params = clt.make_dataclass_from_json(
        BASE_PATH / "common_subpop_params.json", flu.FluSubpopParams
    )
    mixing_params = clt.make_dataclass_from_json(
        BASE_PATH / "mixing_params.json", flu.FluMixingParams
    )
    settings_raw = clt.make_dataclass_from_json(
        BASE_PATH / "simulation_settings.json", flu.SimulationSettings
    )

    tvar_save = TVAR_SAVE if save_tvars else ()
    settings = clt.updated_dataclass(
        settings_raw,
        {
            "transition_type": transition_type,
            "transition_variables_to_save": tvar_save,
        },
    )

    east_cal_df  = pd.read_csv(BASE_PATH / "school_work_calendar_austin_East.csv", index_col=0)
    west_cal_df  = pd.read_csv(BASE_PATH / "school_work_calendar_austin_West.csv", index_col=0)
    humidity_df  = pd.read_csv(BASE_PATH / "absolute_humidity_austin.csv", index_col=0)
    mobility_df  = pd.read_csv(BASE_PATH / "mobility_modifier.csv", index_col=0)

    bg  = np.random.MT19937(seed)
    bg2 = np.random.MT19937(seed + 1)

    east = flu.FluSubpopModel(
        east_state, params, settings,
        np.random.Generator(bg),
        flu.FluSubpopSchedules(
            absolute_humidity=humidity_df,
            flu_contact_matrix=east_cal_df,
            daily_vaccines=east_vax_df,
            mobility_modifier=mobility_df,
        ),
        name="east",
    )
    west = flu.FluSubpopModel(
        west_state, params, settings,
        np.random.Generator(bg2),
        flu.FluSubpopSchedules(
            absolute_humidity=humidity_df,
            flu_contact_matrix=west_cal_df,
            daily_vaccines=west_vax_df,
            mobility_modifier=mobility_df,
        ),
        name="west",
    )
    return flu.FluMetapopModel([east, west], mixing_params)


# ---------------------------------------------------------------------------
# Load raw vaccine schedules (needed for ScenarioRunner and scale_vaccines_df)
# ---------------------------------------------------------------------------

east_vax_raw = pd.read_csv(BASE_PATH / "daily_vaccines_East.csv", index_col=0)
west_vax_raw = pd.read_csv(BASE_PATH / "daily_vaccines_West.csv", index_col=0)

# Scaled schedules for counterfactual scenarios
east_vax_10 = scale_vaccines_df(east_vax_raw, 1.10)
west_vax_10 = scale_vaccines_df(west_vax_raw, 1.10)
east_vax_20 = scale_vaccines_df(east_vax_raw, 1.20)
west_vax_20 = scale_vaccines_df(west_vax_raw, 1.20)


# ===========================================================================
# Output directory + stdout tee (captures all prints to a log file)
# ===========================================================================

out_dir = os.path.join(
    os.path.dirname(__file__), "outputs",
    datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
)
os.makedirs(out_dir, exist_ok=True)
_tee = _Tee(open(os.path.join(out_dir, "output.txt"), "w"))
print(f"Output directory: {out_dir}")

# ===========================================================================
# Part 1 — Single deterministic run for flu_outcomes demonstration
# ===========================================================================

print("=" * 60)
print("Part 1: single deterministic run")
print("=" * 60)

baseline_model = build_model(east_vax_raw, west_vax_raw, save_tvars=True)
baseline_model.simulate_until_day(SIMULATION_DAYS)

subpop_names   = list(baseline_model.subpop_models.keys())
num_age_groups = list(baseline_model.subpop_models.values())[0].params.num_age_groups

# ---- Computation functions ------------------------------------------------

print("\n--- daily_hospital_admissions ---")
daily_ha = outcomes.daily_hospital_admissions(baseline_model)
print(f"  [all] shape: {daily_ha.shape},  peak day: {int(np.argmax(daily_ha))},  peak: {daily_ha.max():.1f}")
for sp in subpop_names:
    ha = outcomes.daily_hospital_admissions(baseline_model, subpop_name=sp)
    print(f"  [{sp}] shape: {ha.shape},  peak: {ha.max():.2f}")
    for a in range(num_age_groups):
        ha = outcomes.daily_hospital_admissions(baseline_model, subpop_name=sp, age_group=a)
        print(f"  [{sp}, age={a}] shape: {ha.shape},  peak: {ha.max():.2f}")

print("\n--- daily_new_infections ---")
daily_inf = outcomes.daily_new_infections(baseline_model)
print(f"  [all] shape: {daily_inf.shape},  peak: {daily_inf.max():.1f} on day {int(np.argmax(daily_inf))}")
for sp in subpop_names:
    inf = outcomes.daily_new_infections(baseline_model, subpop_name=sp)
    print(f"  [{sp}] peak: {inf.max():.1f} on day {int(np.argmax(inf))}")
    for a in range(num_age_groups):
        inf = outcomes.daily_new_infections(baseline_model, subpop_name=sp, age_group=a)
        print(f"  [{sp}, age={a}] peak: {inf.max():.1f} on day {int(np.argmax(inf))}")

print("\n--- cumulative_hospitalizations ---")
print(f"  [all] total: {outcomes.cumulative_hospitalizations(baseline_model):.1f}")
for sp in subpop_names:
    print(f"  [{sp}]: {outcomes.cumulative_hospitalizations(baseline_model, subpop_name=sp):.1f}")
    for a in range(num_age_groups):
        print(f"  [{sp}, age={a}]: {outcomes.cumulative_hospitalizations(baseline_model, subpop_name=sp, age_group=a):.1f}")

print("\n--- cumulative_deaths ---")
print(f"  [all] total: {outcomes.cumulative_deaths(baseline_model):.1f}")
for sp in subpop_names:
    print(f"  [{sp}]: {outcomes.cumulative_deaths(baseline_model, subpop_name=sp):.1f}")
    for a in range(num_age_groups):
        print(f"  [{sp}, age={a}]: {outcomes.cumulative_deaths(baseline_model, subpop_name=sp, age_group=a):.1f}")

print("\n--- attack_rate ---")
print(f"  [all]: {outcomes.attack_rate(baseline_model):.4f}")
for sp in subpop_names:
    print(f"  [{sp}]: {outcomes.attack_rate(baseline_model, subpop_name=sp):.4f}")
    for a in range(num_age_groups):
        print(f"  [{sp}, age={a}]: {outcomes.attack_rate(baseline_model, subpop_name=sp, age_group=a):.4f}")

print("\n--- summarize_outcomes (pretend 5 replicate values) ---")
fake_rep_values = np.random.default_rng(0).uniform(100, 200, 5)
summary = outcomes.summarize_outcomes(fake_rep_values)
print(f"  mean={summary['mean']:.1f}, median={summary['median']:.1f}, "
      f"95% CI=[{summary['lower_ci']:.1f}, {summary['upper_ci']:.1f}]")

# Run a counterfactual for vaccine_preventable_events
counterfactual_model = build_model(east_vax_20, west_vax_20, save_tvars=True)
counterfactual_model.simulate_until_day(SIMULATION_DAYS)

print("\n--- vaccine_preventable_events (+20% coverage vs baseline) ---")
for metric_fn, label in [
    (outcomes.cumulative_hospitalizations, "VPH"),
    (outcomes.cumulative_deaths,           "VPD"),
]:
    print(f"  {label} [all]: {outcomes.vaccine_preventable_events(baseline_model, counterfactual_model, metric_fn):.1f}")
    for sp in subpop_names:
        vpe = outcomes.vaccine_preventable_events(
            baseline_model, counterfactual_model, metric_fn, subpop_name=sp,
        )
        print(f"  {label} [{sp}]: {vpe:.1f}")
        for a in range(num_age_groups):
            vpe = outcomes.vaccine_preventable_events(
                baseline_model, counterfactual_model, metric_fn,
                subpop_name=sp, age_group=a,
            )
            print(f"  {label} [{sp}, age={a}]: {vpe:.1f}")


# ---- Plotting functions ---------------------------------------------------

print("\n--- plotting functions ---")
print(f"  writing figures to {out_dir}")

# plot_compartment_history — default (all compartments, all subpops)
fig, ax = plt.subplots(figsize=(10, 5))
outcomes.plot_compartment_history(
    baseline_model,
    compartment_names=["S", "E", "IP", "ISR", "ISH", "IA", "HR", "HD", "R", "D"],
    ax=ax,
    title="Baseline compartment history (all subpops)",
)
fig.savefig(os.path.join(out_dir, "compartment_history_all.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# plot_compartment_history — age-stratified, one figure per subpop
for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 5))
    outcomes.plot_compartment_history(
        baseline_model,
        compartment_names=["S", "HR", "D"],
        ax=ax,
        subpop_name=sp,
        age_group=0,   # triggers one-line-per-age-group mode
        title=f"{sp} subpop, age-stratified: S / HR / D",
    )
    fig.savefig(os.path.join(out_dir, f"compartment_history_{sp}_age.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# plot_epi_metrics
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_epi_metrics(baseline_model, ax=ax, title="M and MV (baseline)")
fig.savefig(os.path.join(out_dir, "epi_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# plot_epi_metrics — age-stratified, one figure per subpop
for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 4))
    outcomes.plot_epi_metrics(baseline_model, ax=ax, subpop_name=sp, age_group=0,
                              title=f"M and MV by age group ({sp}, baseline)")
    fig.savefig(os.path.join(out_dir, f"epi_metrics_by_age_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# plot_daily_new_infections — overall + one figure per subpop and age group
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_daily_new_infections(baseline_model, ax=ax, label="Baseline",
                                   title="Daily new infections (all)")
outcomes.plot_daily_new_infections(counterfactual_model, ax=ax,
                                   label="+20% vaccines", color="tab:orange")
fig.savefig(os.path.join(out_dir, "daily_infections_overlay.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 4))
    outcomes.plot_daily_new_infections(
        baseline_model, ax=ax, subpop_name=sp,
        label="Baseline", title=f"Daily new infections ({sp})",
    )
    outcomes.plot_daily_new_infections(
        counterfactual_model, ax=ax, subpop_name=sp,
        label="+20% vaccines", color="tab:orange",
    )
    fig.savefig(os.path.join(out_dir, f"daily_infections_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(10, 4))
        outcomes.plot_daily_new_infections(
            baseline_model, ax=ax, subpop_name=sp, age_group=a,
            label="Baseline", title=f"Daily new infections ({sp}, age={a})",
        )
        outcomes.plot_daily_new_infections(
            counterfactual_model, ax=ax, subpop_name=sp, age_group=a,
            label="+20% vaccines", color="tab:orange",
        )
        fig.savefig(os.path.join(out_dir, f"daily_infections_{sp}_age{a}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_daily_hospital_admissions — single model, overall + per subpop and age group
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_daily_hospital_admissions(baseline_model, ax=ax,
                                        title="Daily admissions (baseline, all)")
fig.savefig(os.path.join(out_dir, "daily_admissions_single.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 4))
    outcomes.plot_daily_hospital_admissions(
        baseline_model, ax=ax, subpop_name=sp,
        title=f"Daily admissions ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"daily_admissions_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(10, 4))
        outcomes.plot_daily_hospital_admissions(
            baseline_model, ax=ax, subpop_name=sp, age_group=a,
            title=f"Daily admissions ({sp}, age={a})",
        )
        fig.savefig(os.path.join(out_dir, f"daily_admissions_{sp}_age{a}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_daily_hospital_admissions — dict overlay, overall + per subpop and age group
scenario_dict = {"Baseline": baseline_model, "+20% vaccines": counterfactual_model}
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_daily_hospital_admissions(
    scenario_dict, ax=ax, title="Daily admissions: scenario overlay (all)",
)
fig.savefig(os.path.join(out_dir, "daily_admissions_overlay.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 4))
    outcomes.plot_daily_hospital_admissions(
        scenario_dict, ax=ax, subpop_name=sp,
        title=f"Daily admissions: scenario overlay ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"daily_admissions_overlay_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(10, 4))
        outcomes.plot_daily_hospital_admissions(
            scenario_dict, ax=ax, subpop_name=sp, age_group=a,
            title=f"Daily admissions: scenario overlay ({sp}, age={a})",
        )
        fig.savefig(os.path.join(out_dir, f"daily_admissions_overlay_{sp}_age{a}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_daily_hospital_admissions — dict with list-of-models (ribbon CI)
# (using same model twice for demo; real use would be separate reps)
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_daily_hospital_admissions(
    {
        "Baseline": [baseline_model, baseline_model],
        "+20% vaccines": [counterfactual_model, counterfactual_model],
    },
    ax=ax,
    title="Daily admissions: median + 95% CI ribbon (demo with 2 reps)",
)
fig.savefig(os.path.join(out_dir, "daily_admissions_ribbon.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# plot_attack_rate_by_age — overall + one figure per subpop
fig, ax = plt.subplots(figsize=(8, 4))
outcomes.plot_attack_rate_by_age(baseline_model, ax=ax, title="Attack rate by age group (all)")
fig.savefig(os.path.join(out_dir, "attack_rate_by_age.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(8, 4))
    outcomes.plot_attack_rate_by_age(baseline_model, ax=ax, subpop_name=sp,
                                     title=f"Attack rate by age group ({sp})")
    fig.savefig(os.path.join(out_dir, f"attack_rate_by_age_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# plot_scenario_comparison — single run (bar chart), overall + per subpop and age group
sc_dict_single = {"Baseline": baseline_model, "+20% vaccines": counterfactual_model}
fig, ax = plt.subplots(figsize=(7, 5))
outcomes.plot_scenario_comparison(
    sc_dict_single, outcomes.cumulative_hospitalizations,
    ax=ax, metric_name="Cumulative hospitalizations",
    title="Scenario comparison (single run, all)",
)
fig.savefig(os.path.join(out_dir, "scenario_comparison_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(7, 5))
    outcomes.plot_scenario_comparison(
        sc_dict_single, outcomes.cumulative_hospitalizations,
        ax=ax, metric_name="Cumulative hospitalizations",
        subpop_name=sp,
        title=f"Scenario comparison ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"scenario_comparison_{sp}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(7, 5))
        outcomes.plot_scenario_comparison(
            sc_dict_single, outcomes.cumulative_hospitalizations,
            ax=ax, metric_name="Cumulative hospitalizations",
            subpop_name=sp, age_group=a,
            title=f"Scenario comparison ({sp}, age={a})",
        )
        fig.savefig(os.path.join(out_dir, f"scenario_comparison_{sp}_age{a}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_scenario_comparison — multi-replicate (box plot)
fig, ax = plt.subplots(figsize=(7, 5))
outcomes.plot_scenario_comparison(
    {
        "Baseline": [baseline_model, baseline_model],
        "+20% vaccines": [counterfactual_model, counterfactual_model],
    },
    outcomes.cumulative_hospitalizations,
    ax=ax,
    metric_name="Cumulative hospitalizations",
    title="Scenario comparison (multi-replicate box plot, demo with 2 reps)",
)
fig.savefig(os.path.join(out_dir, "scenario_comparison_box.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("  done.")


# ===========================================================================
# Part 2 — ScenarioRunner: multi-replicate, SQL-backed
# ===========================================================================

print("\n" + "=" * 60)
print("Part 2: ScenarioRunner multi-replicate run")
print("=" * 60)

# ScenarioRunner mutates the baseline model in-place (applying and then
# restoring overrides), so we build a fresh model here with save_tvars=False.
# Transition variable histories are not used by ScenarioRunner itself —
# they are collected in Part 3 below.
runner_model = build_model(east_vax_raw, west_vax_raw, save_tvars=False)

scenarios = {
    "baseline": {},
    "vax_plus_10pct": {
        "subpop_schedules": {
            "east": {"daily_vaccines": east_vax_10},
            "west": {"daily_vaccines": west_vax_10},
        }
    },
    "vax_plus_20pct": {
        "subpop_schedules": {
            "east": {"daily_vaccines": east_vax_20},
            "west": {"daily_vaccines": west_vax_20},
        }
    },
}

db_path = os.path.join(out_dir, "scenario_results.db")

runner = clt.ScenarioRunner(
    baseline_model=runner_model,
    state_variables_to_record=["S", "HR", "HD", "D"],
    database_filename=db_path,
)

print(f"\nRunning {NUM_REPS} paired replicates × {len(scenarios)} scenarios …")
runner.run(
    scenarios=scenarios,
    num_reps=NUM_REPS,
    simulation_end_day=SIMULATION_DAYS,
    seeds=SEEDS,          # same seeds across scenarios → paired replicates
)
print("Done.")

# ---- Query the SQL database -----------------------------------------------

print("\n--- runner.get_results_df() ---")

# All scenarios, all variables
df_all = runner.get_results_df()
print(f"  Full table: {len(df_all):,} rows × {len(df_all.columns)} columns")
print(f"  Scenarios present: {sorted(df_all['scenario_name'].unique())}")

# Filter to one scenario
df_baseline = runner.get_results_df(scenario_name="baseline")
print(f"  Baseline only: {len(df_baseline):,} rows")

# Filter to one state variable across all scenarios
df_D = runner.get_results_df(state_var_name="D")
print(f"  Deaths (D) all scenarios: {len(df_D):,} rows")

# Filter to specific scenario + variable + subpop + age group
df_specific = runner.get_results_df(
    scenario_name="vax_plus_20pct",
    state_var_name="HR",
    subpop_name="east",
    age_group=0,
)
print(f"  +20% / HR / east / age_group=0: {len(df_specific):,} rows")
print(f"  Columns: {list(df_specific.columns)}")

# Compute mean deaths at final timepoint per scenario
final_day = df_all["timepoint"].max()
mean_deaths = (
    df_all[df_all["timepoint"] == final_day]
    .query("state_var_name == 'D'")
    .groupby("scenario_name")["value"]
    .mean()
)
print("\n  Mean cumulative deaths at final timepoint by scenario:")
for sc, val in mean_deaths.items():
    print(f"    {sc}: {val:.2f}")


# ===========================================================================
# Part 3 — Per-scenario in-memory runs for flu_outcomes
# ===========================================================================

print("\n" + "=" * 60)
print("Part 3: per-scenario re-runs for flu_outcomes outcome metrics")
print("=" * 60)

# Build one deterministic model per scenario with tvar histories saved.
# Using seed=SEEDS[0] gives the same stochastic path as replicate 0 from
# the ScenarioRunner run above (when transition_type is binom_deterministic
# the seed doesn't matter, but the pattern generalises to stochastic runs).

scenario_models = {}

raw_vax = {"east": east_vax_raw, "west": west_vax_raw}
scaled_10 = {"east": east_vax_10, "west": west_vax_10}
scaled_20 = {"east": east_vax_20, "west": west_vax_20}

vax_by_scenario = {
    "baseline":      raw_vax,
    "vax_plus_10pct": scaled_10,
    "vax_plus_20pct": scaled_20,
}

for sc_name, vax_dfs in vax_by_scenario.items():
    model = build_model(vax_dfs["east"], vax_dfs["west"], save_tvars=True)
    model.simulate_until_day(SIMULATION_DAYS)
    scenario_models[sc_name] = model
    print(f"  {sc_name}: simulated {SIMULATION_DAYS} days")

# ---- Outcome metrics per scenario -----------------------------------------

print("\n--- Outcome metrics per scenario ---")
for label, kwargs in [("all", {})] + [
    item
    for sp in subpop_names
    for item in [(sp, {"subpop_name": sp})] + [
        (f"{sp}, age={a}", {"subpop_name": sp, "age_group": a})
        for a in range(num_age_groups)
    ]
]:
    print(f"\n  [{label}]")
    print(f"  {'Scenario':<22} {'CumHosp':>10} {'CumDeaths':>11} {'AttackRate':>11}")
    print("  " + "-" * 56)
    for sc_name, model in scenario_models.items():
        ch = outcomes.cumulative_hospitalizations(model, **kwargs)
        cd = outcomes.cumulative_deaths(model, **kwargs)
        ar = outcomes.attack_rate(model, **kwargs)
        print(f"  {sc_name:<22} {ch:>10.1f} {cd:>11.1f} {ar:>11.4f}")

# ---- Vaccine-preventable events -------------------------------------------

print("\n--- vaccine_preventable_events ---")
for sc_name in ["vax_plus_10pct", "vax_plus_20pct"]:
    for label, kwargs in [("all", {})] + [
        (f"{sp}, age={a}", {"subpop_name": sp, "age_group": a})
        for sp in subpop_names for a in range(num_age_groups)
    ]:
        vph = outcomes.vaccine_preventable_events(
            scenario_models["baseline"], scenario_models[sc_name],
            outcomes.cumulative_hospitalizations, **kwargs,
        )
        vpd = outcomes.vaccine_preventable_events(
            scenario_models["baseline"], scenario_models[sc_name],
            outcomes.cumulative_deaths, **kwargs,
        )
        print(f"  {sc_name} [{label}]: VPH={vph:.1f}, VPD={vpd:.1f}")

# ---- Attack rate by age for each scenario ---------------------------------

print("\n--- attack_rate by age group per scenario ---")
for sp_label, sp_kw in [("all subpops", {})] + [(sp, {"subpop_name": sp}) for sp in subpop_names]:
    header = f"  [{sp_label}]  {'Scenario':<22}" + "".join(f"  age{a}" for a in range(num_age_groups))
    print(header)
    print("  " + "-" * len(header))
    for sc_name, model in scenario_models.items():
        rates = [outcomes.attack_rate(model, age_group=a, **sp_kw) for a in range(num_age_groups)]
        row = f"  {' ' * (len(sp_label) + 3)}{sc_name:<22}" + "".join(f"  {r:.3f}" for r in rates)
        print(row)
    print()

# ---- summarize_outcomes over the scenario models --------------------------

print("\n--- summarize_outcomes (cumulative hospitalizations across scenarios) ---")
hosp_values = [
    outcomes.cumulative_hospitalizations(m) for m in scenario_models.values()
]
summary = outcomes.summarize_outcomes(hosp_values)
print(f"  across {len(hosp_values)} scenarios: {summary}")

# ---- Multi-scenario plots -------------------------------------------------

print("\n--- multi-scenario plots ---")

# plot_daily_hospital_admissions — overlay, overall + per subpop and age group
fig, ax = plt.subplots(figsize=(10, 4))
outcomes.plot_daily_hospital_admissions(
    scenario_models, ax=ax, title="Daily hospital admissions by scenario (all)",
)
fig.savefig(os.path.join(out_dir, "multi_scenario_daily_admissions.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(10, 4))
    outcomes.plot_daily_hospital_admissions(
        scenario_models, ax=ax, subpop_name=sp,
        title=f"Daily hospital admissions by scenario ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"multi_scenario_daily_admissions_{sp}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(10, 4))
        outcomes.plot_daily_hospital_admissions(
            scenario_models, ax=ax, subpop_name=sp, age_group=a,
            title=f"Daily hospital admissions by scenario ({sp}, age={a})",
        )
        fig.savefig(os.path.join(out_dir, f"multi_scenario_daily_admissions_{sp}_age{a}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_scenario_comparison — bar chart, overall + per subpop and age group
fig, ax = plt.subplots(figsize=(8, 5))
outcomes.plot_scenario_comparison(
    scenario_models, outcomes.cumulative_hospitalizations,
    ax=ax, metric_name="Cumulative hospitalizations",
    title="Cumulative hospitalizations by scenario (all)",
)
fig.savefig(os.path.join(out_dir, "multi_scenario_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes.plot_scenario_comparison(
        scenario_models, outcomes.cumulative_hospitalizations,
        ax=ax, metric_name="Cumulative hospitalizations",
        subpop_name=sp,
        title=f"Cumulative hospitalizations by scenario ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"multi_scenario_comparison_{sp}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    for a in range(num_age_groups):
        fig, ax = plt.subplots(figsize=(8, 5))
        outcomes.plot_scenario_comparison(
            scenario_models, outcomes.cumulative_hospitalizations,
            ax=ax, metric_name="Cumulative hospitalizations",
            subpop_name=sp, age_group=a,
            title=f"Cumulative hospitalizations by scenario ({sp}, age={a})",
        )
        fig.savefig(os.path.join(out_dir, f"multi_scenario_comparison_{sp}_age{a}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

# plot_attack_rate_by_age — overall + one figure per subpop
fig, ax = plt.subplots(figsize=(8, 4))
outcomes.plot_attack_rate_by_age(
    scenario_models["baseline"], ax=ax,
    title="Baseline attack rate by age group (all)",
)
fig.savefig(os.path.join(out_dir, "attack_rate_by_age_baseline.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

for sp in subpop_names:
    fig, ax = plt.subplots(figsize=(8, 4))
    outcomes.plot_attack_rate_by_age(
        scenario_models["baseline"], ax=ax, subpop_name=sp,
        title=f"Baseline attack rate by age group ({sp})",
    )
    fig.savefig(os.path.join(out_dir, f"attack_rate_by_age_baseline_{sp}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"  Figures written to {out_dir}")
print("\nAll done.")
_tee.close()
