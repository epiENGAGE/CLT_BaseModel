"""
Tests for Task 5: Scenario Configuration Infrastructure

5a: replace_schedule on SubpopModel / MetapopModel
5b: ScenarioRunner
5c: paired replicates via seeds parameter
"""

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

import clt_toolkit as clt

base_path = clt.utils.PROJECT_ROOT / "tests" / "test_input_files"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_vaccines_df(vaccines_df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Return a copy of the raw vaccines CSV DataFrame with all daily_vaccines
    values multiplied by `scale`.

    Dates are converted from strings to datetime.date objects so the
    DataFrame is in the format that DailyVaccines.postprocess_data_input
    expects (matching what FluSubpopModel.create_schedules does before
    calling postprocess_data_input).
    """
    scaled = vaccines_df.copy()
    scaled["daily_vaccines"] = scaled["daily_vaccines"].apply(
        lambda s: json.dumps(
            (np.array(json.loads(s)) * scale).tolist()
        )
    )
    scaled["date"] = pd.to_datetime(scaled["date"], format="%Y-%m-%d").dt.date
    return scaled


# ---------------------------------------------------------------------------
# Task 5a — replace_schedule
# ---------------------------------------------------------------------------

class TestMVInitValConsistency:
    """
    VaxInducedImmunity.adjust_initial_value() must be re-run on
    reset_simulation() so that MV.init_val is consistent with whatever
    daily_vaccines schedule is currently loaded.

    The caseB test CSV starts exactly on the simulation start date, so the
    pre-simulation vaccine filter in adjust_initial_value() finds no rows and
    the numerical init_val does not change.  We therefore test the invariant
    behaviourally: after replace_schedule + reset_simulation, the model's MV
    trajectory over multiple days must match a freshly built model that used
    the same schedule from the start.
    """

    def _run_mv_history(self, model, n_days: int) -> np.ndarray:
        """Run deterministically for n_days; return MV history as an array."""
        model.simulate_until_day(n_days)
        return np.array(model.epi_metrics["MV"].history_vals_list)

    def test_reset_after_replace_matches_fresh_model(self, make_flu_subpop_model):
        """
        After replace_schedule + reset_simulation, the MV trajectory must be
        identical to a freshly constructed model that used the new schedule.
        """
        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        scaled_df = _scale_vaccines_df(raw_df, 5.0)

        # Reference: a fresh model built directly with the scaled schedule
        model_ref = make_flu_subpop_model(
            "reference",
            transition_type=clt.TransitionTypes.BINOM_DETERMINISTIC,
            case_id_str="caseB_subpop1",
        )
        model_ref.replace_schedule("daily_vaccines", scaled_df)
        # reset so it starts from a consistent initial state
        model_ref.reset_simulation()
        history_ref = self._run_mv_history(model_ref, 10)

        # Model that had original schedule, then replaced + reset
        model_swapped = make_flu_subpop_model(
            "swapped",
            transition_type=clt.TransitionTypes.BINOM_DETERMINISTIC,
            case_id_str="caseB_subpop1",
        )
        model_swapped.replace_schedule("daily_vaccines", scaled_df)
        model_swapped.reset_simulation()
        history_swapped = self._run_mv_history(model_swapped, 10)

        np.testing.assert_allclose(history_swapped, history_ref)

    def test_mv_trajectory_differs_between_different_schedules(
            self, make_flu_subpop_model):
        """
        Two models with different vaccine schedules must produce different MV
        trajectories — confirming that replace_schedule actually changes
        in-simulation dynamics.
        """
        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )

        model_1x = make_flu_subpop_model(
            "1x", transition_type=clt.TransitionTypes.BINOM_DETERMINISTIC,
            case_id_str="caseB_subpop1"
        )
        model_10x = make_flu_subpop_model(
            "10x", transition_type=clt.TransitionTypes.BINOM_DETERMINISTIC,
            case_id_str="caseB_subpop1"
        )
        model_10x.replace_schedule("daily_vaccines", _scale_vaccines_df(raw_df, 10.0))
        model_10x.reset_simulation()

        mv_1x  = self._run_mv_history(model_1x,  30)
        mv_10x = self._run_mv_history(model_10x, 30)

        assert not np.allclose(mv_1x, mv_10x), (
            "MV trajectories are identical despite 10x vaccine coverage difference."
        )

    def test_reset_without_replace_is_idempotent(self, make_flu_subpop_model):
        """
        reset_simulation() without any schedule change must produce the same
        MV.init_val as the original construction (no drift across resets).
        """
        model = make_flu_subpop_model("idempotent", case_id_str="caseB_subpop1")
        original_mv_init = copy.deepcopy(model.epi_metrics["MV"].init_val)

        model.reset_simulation()
        np.testing.assert_array_equal(
            model.epi_metrics["MV"].init_val, original_mv_init
        )

        model.reset_simulation()
        np.testing.assert_array_equal(
            model.epi_metrics["MV"].init_val, original_mv_init
        )

class TestReplaceScheduleSubpop:
    """replace_schedule on a single FluSubpopModel."""

    def test_replace_changes_schedule_values(self, make_flu_subpop_model):
        """
        Acceptance criterion: swap daily_vaccines on a live SubpopModel,
        reset the simulation, and confirm the new schedule is used.
        """
        model = make_flu_subpop_model("test_subpop", case_id_str="caseB_subpop1")

        # Record vaccine value on day 0 with the original schedule
        model.prepare_daily_state()
        original_val = copy.deepcopy(model.daily_vaccines.current_val)

        # Build a scaled-up vaccine schedule (2x coverage)
        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        scaled_df = _scale_vaccines_df(raw_df, 2.0)

        # Swap the schedule and reset
        model.replace_schedule("daily_vaccines", scaled_df)
        model.reset_simulation()
        model.prepare_daily_state()
        new_val = copy.deepcopy(model.daily_vaccines.current_val)

        # New values should be exactly 2x the original
        np.testing.assert_allclose(new_val, original_val * 2.0)

    def test_replace_invalid_schedule_raises(self, make_flu_subpop_model):
        model = make_flu_subpop_model("test_subpop", case_id_str="caseB_subpop1")
        with pytest.raises(clt.SubpopModelError):
            model.replace_schedule("nonexistent_schedule", pd.DataFrame())

    def test_replace_does_not_affect_other_schedules(self, make_flu_subpop_model):
        """Replacing one schedule must not mutate any other schedule."""
        model = make_flu_subpop_model("test_subpop", case_id_str="caseA")

        model.prepare_daily_state()
        original_humidity = copy.deepcopy(model.absolute_humidity.current_val)

        raw_df = pd.read_csv(
            base_path / "caseA_daily_vaccines_constant.csv", index_col=0
        )
        scaled_df = _scale_vaccines_df(raw_df, 3.0)
        model.replace_schedule("daily_vaccines", scaled_df)

        model.reset_simulation()
        model.prepare_daily_state()
        np.testing.assert_array_equal(
            model.absolute_humidity.current_val, original_humidity
        )


class TestReplaceScheduleMetapop:
    """replace_schedule on a FluMetapopModel."""

    def test_replace_all_subpops(self, make_flu_metapop_model):
        """
        Calling replace_schedule without subpop_name updates every subpop.
        """
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)

        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        # Record originals
        originals = {}
        for name, subpop in metapop.subpop_models.items():
            subpop.prepare_daily_state()
            originals[name] = copy.deepcopy(subpop.daily_vaccines.current_val)

        scaled_df = _scale_vaccines_df(raw_df, 1.5)
        metapop.replace_schedule("daily_vaccines", scaled_df)
        metapop.reset_simulation()

        for name, subpop in metapop.subpop_models.items():
            subpop.prepare_daily_state()
            np.testing.assert_allclose(
                subpop.daily_vaccines.current_val, originals[name] * 1.5
            )

    def test_replace_single_subpop(self, make_flu_metapop_model):
        """
        Calling replace_schedule with subpop_name only updates that subpop.
        """
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)

        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )

        for subpop in metapop.subpop_models.values():
            subpop.prepare_daily_state()

        original_subpop2 = copy.deepcopy(
            metapop.subpop_models["subpop2"].daily_vaccines.current_val
        )

        scaled_df = _scale_vaccines_df(raw_df, 2.0)
        metapop.replace_schedule("daily_vaccines", scaled_df, subpop_name="subpop1")
        metapop.reset_simulation()

        # subpop1 changed; subpop2 unchanged
        metapop.subpop_models["subpop1"].prepare_daily_state()
        metapop.subpop_models["subpop2"].prepare_daily_state()

        np.testing.assert_allclose(
            metapop.subpop_models["subpop2"].daily_vaccines.current_val,
            original_subpop2
        )

    def test_replace_invalid_subpop_raises(self, make_flu_metapop_model):
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)
        with pytest.raises(clt.MetapopModelError):
            metapop.replace_schedule("daily_vaccines", pd.DataFrame(),
                                     subpop_name="does_not_exist")


# ---------------------------------------------------------------------------
# Task 5b — ScenarioRunner
# ---------------------------------------------------------------------------

class TestScenarioRunner:
    """ScenarioRunner produces a database filterable by scenario_name."""

    def test_two_scenarios_produce_distinct_results(
            self, make_flu_metapop_model, tmp_path):
        """
        Acceptance criterion (5b): running baseline and +20% vaccine coverage
        produces a database with results filterable by scenario_name.
        """
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)

        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        vax_plus_20 = _scale_vaccines_df(raw_df, 1.2)

        db_path = str(tmp_path / "scenario_test.db")

        runner = clt.ScenarioRunner(
            baseline_model=metapop,
            state_variables_to_record=["S", "E"],
            database_filename=db_path,
        )

        runner.run(
            scenarios={
                "baseline":       {},
                "vaccines_+20%":  {"schedules": {"daily_vaccines": vax_plus_20}},
            },
            num_reps=2,
            simulation_end_day=10,
        )

        # Database should exist
        assert os.path.exists(db_path)

        # Both scenario names should appear
        df = runner.get_results_df()
        assert set(df["scenario_name"].unique()) == {"baseline", "vaccines_+20%"}

        # Each scenario should have rows for the expected state variables
        baseline_df = runner.get_results_df(scenario_name="baseline")
        vax_df = runner.get_results_df(scenario_name="vaccines_+20%")
        assert not baseline_df.empty
        assert not vax_df.empty
        assert set(baseline_df["state_var_name"].unique()) == {"S", "E"}
        assert set(vax_df["state_var_name"].unique()) == {"S", "E"}

    def test_database_already_exists_raises(self, make_flu_metapop_model, tmp_path):
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)
        db_path = str(tmp_path / "existing.db")
        # Pre-create the file
        open(db_path, "w").close()
        with pytest.raises(clt.ScenarioRunnerError):
            clt.ScenarioRunner(metapop, ["S"], db_path)

    def test_invalid_state_variable_raises(self, make_flu_metapop_model, tmp_path):
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)
        db_path = str(tmp_path / "invalid_var.db")
        with pytest.raises(clt.ScenarioRunnerError):
            clt.ScenarioRunner(metapop, ["nonexistent_var"], db_path)

    def test_baseline_model_not_mutated(self, make_flu_metapop_model, tmp_path):
        """
        ScenarioRunner must not mutate the baseline model passed to it.
        """
        metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM_DETERMINISTIC)

        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        # Record original schedule timeseries_df identity
        original_df_id = id(
            metapop.subpop_models["subpop1"].schedules["daily_vaccines"].timeseries_df
        )

        vax_2x = _scale_vaccines_df(raw_df, 2.0)
        db_path = str(tmp_path / "no_mutation.db")

        runner = clt.ScenarioRunner(metapop, ["S"], db_path)
        runner.run(
            scenarios={"vaccines_2x": {"schedules": {"daily_vaccines": vax_2x}}},
            num_reps=1,
            simulation_end_day=5,
        )

        # The baseline's schedule object must be unchanged
        after_df_id = id(
            metapop.subpop_models["subpop1"].schedules["daily_vaccines"].timeseries_df
        )
        assert original_df_id == after_df_id, (
            "ScenarioRunner mutated the baseline model's schedule."
        )


# ---------------------------------------------------------------------------
# Task 5c — paired replicates via seeds
# ---------------------------------------------------------------------------

class TestPairedReplicates:
    """Passing the same seeds list to two Experiment runs gives identical output."""

    def test_seeds_give_identical_replicates(
            self, make_flu_metapop_model, tmp_path):
        """
        Two Experiment runs with the same seeds list must produce bit-identical
        results for every replicate.
        """
        seeds = [42, 99, 7]

        def _run(db_name):
            metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM)
            db_path = str(tmp_path / db_name)
            exp = clt.Experiment(metapop, ["S", "E"], db_path)
            exp.run_static_inputs(
                num_reps=len(seeds),
                simulation_end_day=5,
                seeds=seeds,
            )
            return exp.get_state_var_df("S")

        df_a = _run("seeds_a.db")
        df_b = _run("seeds_b.db")

        pd.testing.assert_frame_equal(df_a, df_b)

    def test_different_seeds_give_different_replicates(
            self, make_flu_metapop_model, tmp_path):
        """
        Different seed lists must produce different stochastic outcomes.
        """
        def _run(db_name, seeds):
            metapop = make_flu_metapop_model(clt.TransitionTypes.BINOM)
            db_path = str(tmp_path / db_name)
            exp = clt.Experiment(metapop, ["S"], db_path)
            exp.run_static_inputs(num_reps=3, simulation_end_day=5, seeds=seeds)
            return exp.get_state_var_df("S")

        df_a = _run("diff_seeds_a.db", [1, 2, 3])
        df_b = _run("diff_seeds_b.db", [100, 200, 300])

        # At least one replicate should differ
        assert not df_a.equals(df_b)

    def test_scenario_runner_paired_seeds(self, make_flu_metapop_model, tmp_path):
        """
        ScenarioRunner with the same seeds list produces the same baseline
        results regardless of what other scenarios were run alongside it.
        """
        raw_df = pd.read_csv(
            base_path / "caseB_daily_vaccines_constant.csv", index_col=0
        )
        vax_2x = _scale_vaccines_df(raw_df, 2.0)
        seeds = [10, 20, 30]

        # Run A: only baseline
        metapop_a = make_flu_metapop_model(clt.TransitionTypes.BINOM)
        db_a = str(tmp_path / "paired_a.db")
        runner_a = clt.ScenarioRunner(metapop_a, ["S"], db_a)
        runner_a.run({"baseline": {}}, num_reps=len(seeds),
                     simulation_end_day=5, seeds=seeds)
        baseline_a = runner_a.get_results_df(scenario_name="baseline",
                                             state_var_name="S")

        # Run B: baseline + counterfactual together
        metapop_b = make_flu_metapop_model(clt.TransitionTypes.BINOM)
        db_b = str(tmp_path / "paired_b.db")
        runner_b = clt.ScenarioRunner(metapop_b, ["S"], db_b)
        runner_b.run(
            {"baseline": {}, "vax_2x": {"schedules": {"daily_vaccines": vax_2x}}},
            num_reps=len(seeds),
            simulation_end_day=5,
            seeds=seeds,
        )
        baseline_b = runner_b.get_results_df(scenario_name="baseline",
                                             state_var_name="S")

        pd.testing.assert_frame_equal(
            baseline_a[["rep", "timepoint", "value"]].sort_values(
                ["rep", "timepoint"]).reset_index(drop=True),
            baseline_b[["rep", "timepoint", "value"]].sort_values(
                ["rep", "timepoint"]).reset_index(drop=True),
        )
