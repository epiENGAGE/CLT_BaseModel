"""
ScenarioRunner
==============

Runs a baseline `MetapopModel` and one or more counterfactual scenarios as
paired, multi-replicate experiments and writes results to a single SQLite
database that includes a `scenario_name` column.

Typical use
-----------
::

    runner = ScenarioRunner(
        baseline_model=metapop_model,
        state_variables_to_record=["S", "HR", "HD", "D"],
        database_filename="results/scenario_results.db",
    )

    seeds = list(range(100))          # same 100 seeds used for every scenario

    runner.run(
        scenarios={
            "baseline":          {},
            "vaccines_plus_10":  {"schedules": {"daily_vaccines": vax_10_df}},
            "vaccines_plus_20":  {
                "schedules": {"daily_vaccines": vax_20_df}
                "subpop_schedules": {'subpop_1': {"daily_vaccines": vax_20_df}},
                },
        },
        num_reps=100,
        simulation_end_day=200,
        seeds=seeds,
    )

    df = runner.get_results_df(scenario_name="vaccines_plus_20")

Scenario definition format
--------------------------
Each scenario is a plain ``dict`` with optional keys:

* ``"schedules"`` – ``{schedule_name: new_df}`` – passed to
  ``SubpopModel.replace_schedule`` on every subpopulation.
* ``"subpop_schedules"`` – ``{subpop_name: {schedule_name: new_df}}`` –
  like ``"schedules"`` but targets individual subpopulations.  Use this
  when different subpopulations need different DataFrames for the same
  schedule.  Both keys may appear in the same scenario definition.
* ``"params"`` – ``{subpop_name: updates_dict}`` – for each named
  subpopulation, ``updates_dict`` is forwarded to ``updated_dataclass`` on
  that subpop's frozen params object.  The original params are restored
  after the scenario run.

An empty dict (``{}``) means the scenario is run with the baseline
configuration unchanged.
"""

import sqlite3
import os
from typing import Optional

import numpy as np
import pandas as pd

from .base_components import MetapopModel, SubpopModel
from .utils import updated_dataclass


class ScenarioRunnerError(Exception):
    """Custom exceptions for ScenarioRunner errors."""
    pass


def _format_row_for_sql(subpop_model: SubpopModel,
                        state_var_name: str,
                        scenario_name: str,
                        rep: int) -> list:
    """
    Same as `format_current_val_for_sql` in experiments.py but prepends a
    ``scenario_name`` column so that results from multiple scenarios can live
    in the same table.

    Returns a list of rows, each with 8 elements:
        scenario_name, subpop_name, state_var_name, age_group, risk_group,
        rep, timepoint, value
    """

    current_val = subpop_model.all_state_variables[state_var_name].current_val

    A, R = np.shape(current_val)

    current_val_reshaped = current_val.reshape(-1, 1)
    age_group_indices = np.repeat(np.arange(A), R).reshape(-1, 1)
    risk_group_indices = np.tile(np.arange(R), A).reshape(-1, 1)

    data = np.column_stack(
        (np.full((A * R, 1), scenario_name),
         np.full((A * R, 1), subpop_model.name),
         np.full((A * R, 1), state_var_name),
         age_group_indices,
         risk_group_indices,
         np.full((A * R, 1), rep),
         np.full((A * R, 1), subpop_model.current_simulation_day),
         current_val_reshaped)
    ).tolist()

    return data


class ScenarioRunner:
    """
    Runs a baseline model and a set of named counterfactual scenarios as
    paired multi-replicate experiments.

    All results are written to a single SQLite database with an additional
    ``scenario_name`` column so that any scenario can be retrieved with a
    simple ``WHERE scenario_name = ?`` filter.

    Parameters
    ----------
    baseline_model : MetapopModel | SubpopModel
        The model in its baseline configuration.  The runner makes a deep
        copy before applying any scenario overrides, so the original object
        is never mutated.
    state_variables_to_record : list[str]
        Names of state variables to log at every save-point.  Must be valid
        state variable names on every SubpopModel in the metapop.
    database_filename : str
        Path to the SQLite ``.db`` file to create.  The file must not already
        exist (to prevent accidental overwrites).
    """

    def __init__(self,
                 baseline_model: MetapopModel | SubpopModel,
                 state_variables_to_record: list,
                 database_filename: str):

        if not isinstance(baseline_model, (MetapopModel, SubpopModel)):
            raise ScenarioRunnerError(
                "baseline_model must be a MetapopModel or SubpopModel instance."
            )

        if os.path.exists(database_filename):
            raise ScenarioRunnerError(
                f"Database '{database_filename}' already exists. "
                "Delete it or choose a different filename."
            )

        self.baseline_model = baseline_model
        self.state_variables_to_record = state_variables_to_record
        self.database_filename = database_filename

        # Validate that every state variable name exists on the model
        if isinstance(baseline_model, MetapopModel):
            subpop_models = list(baseline_model.subpop_models.values())
        else:
            subpop_models = [baseline_model]

        for subpop_model in subpop_models:
            missing = [v for v in state_variables_to_record
                       if v not in subpop_model.all_state_variables]
            if missing:
                raise ScenarioRunnerError(
                    f"State variable(s) {missing} not found on subpop "
                    f"'{subpop_model.name}'."
                )

        self._create_results_sql_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self,
            scenarios: dict,
            num_reps: int,
            simulation_end_day: int,
            days_between_save_history: int = 1,
            seeds: Optional[list] = None) -> None:
        """
        Run all scenarios and write results to the database.

        Parameters
        ----------
        scenarios : dict
            ``{scenario_name: scenario_definition}`` mapping.  Each
            ``scenario_definition`` is a plain dict; see module docstring for
            the supported keys (``"schedules"`` and ``"params"``).
        num_reps : int
            Number of simulation replications per scenario.
        simulation_end_day : int
            Exclusive end day (simulate days 0 … simulation_end_day-1).
        days_between_save_history : int
            How often (in days) to log state variables to the database.
        seeds : Optional[list]
            If provided, must contain at least ``num_reps`` entries.
            ``seeds[i]`` is used to re-seed the model's RNG before
            replicate ``i`` in *every* scenario, giving paired replicates
            (same stochastic path up to the point where schedules/params
            diverge).
        """

        if seeds is not None and len(seeds) < num_reps:
            raise ScenarioRunnerError(
                f"seeds list has {len(seeds)} entries but num_reps is {num_reps}."
            )

        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()

        model = self.baseline_model

        for scenario_name, scenario_def in scenarios.items():
            # Save baseline state for the parts the scenario will override
            saved = self._save_overrideable_state(model, scenario_def)

            self._apply_scenario(model, scenario_def)
            self._run_scenario(model, scenario_name, num_reps, simulation_end_day,
                               days_between_save_history, seeds, conn, cursor)

            # Restore the baseline so the next scenario starts from scratch
            self._restore_overrideable_state(model, saved)

        conn.close()

    def get_results_df(self,
                       scenario_name: str = None,
                       state_var_name: str = None,
                       subpop_name: str = None,
                       age_group: int = None,
                       risk_group: int = None) -> pd.DataFrame:
        """
        Load results from the database, optionally filtered.

        All ``None`` arguments mean "sum across that dimension" when the
        result is aggregated, except for ``scenario_name`` which simply
        removes the filter so all scenarios are returned.

        Returns a DataFrame with columns:
        ``scenario_name, subpop_name, state_var_name, age_group,
        risk_group, rep, timepoint, value``.
        """

        conn = sqlite3.connect(self.database_filename)

        query = "SELECT * FROM results"
        params = []
        conditions = []

        if scenario_name is not None:
            conditions.append("scenario_name = ?")
            params.append(scenario_name)
        if state_var_name is not None:
            conditions.append("state_var_name = ?")
            params.append(state_var_name)
        if subpop_name is not None:
            conditions.append("subpop_name = ?")
            params.append(subpop_name)
        if age_group is not None:
            conditions.append("age_group = ?")
            params.append(age_group)
        if risk_group is not None:
            conditions.append("risk_group = ?")
            params.append(risk_group)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        chunks = []
        try:
            for chunk in pd.read_sql_query(query, conn, chunksize=int(1e4),
                                           params=params if params else None):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        except Exception as exc:
            df = pd.DataFrame()
            print(f"Warning: could not read results table: {exc}")
        finally:
            conn.close()

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_results_sql_table(self) -> None:
        conn = sqlite3.connect(self.database_filename)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            scenario_name TEXT,
            subpop_name TEXT,
            state_var_name TEXT,
            age_group INT,
            risk_group INT,
            rep INT,
            timepoint INT,
            value FLOAT,
            PRIMARY KEY (scenario_name, subpop_name, state_var_name,
                         age_group, risk_group, rep, timepoint)
        )
        """)
        conn.commit()
        conn.close()

    def _save_overrideable_state(self,
                                 model: MetapopModel | SubpopModel,
                                 scenario_def: dict) -> dict:
        """
        Snapshot the parts of *model* that *scenario_def* will override so
        they can be restored after the scenario run.

        Returns a dict with keys:
        - ``"schedules"``: ``{(subpop_name, schedule_name): processed_timeseries_df}``
        - ``"params"``: ``{subpop_name: params_object}``
        """

        saved: dict = {"schedules": {}, "params": {}}

        if isinstance(model, MetapopModel):
            subpop_models = model.subpop_models
        else:
            subpop_models = {model.name: model}

        for schedule_name in scenario_def.get("schedules", {}):
            for subpop_name, subpop in subpop_models.items():
                if schedule_name in subpop.schedules:
                    saved["schedules"][(subpop_name, schedule_name)] = \
                        subpop.schedules[schedule_name].timeseries_df

        for subpop_name, sched_map in scenario_def.get("subpop_schedules", {}).items():
            if subpop_name in subpop_models:
                subpop = subpop_models[subpop_name]
                for schedule_name in sched_map:
                    if schedule_name in subpop.schedules:
                        saved["schedules"][(subpop_name, schedule_name)] = \
                            subpop.schedules[schedule_name].timeseries_df

        for subpop_name in scenario_def.get("params", {}):
            if subpop_name in subpop_models:
                saved["params"][subpop_name] = subpop_models[subpop_name].params

        return saved

    def _restore_overrideable_state(self,
                                    model: MetapopModel | SubpopModel,
                                    saved: dict) -> None:
        """
        Restore the parts of *model* that were saved by
        ``_save_overrideable_state``, without calling ``postprocess_data_input``
        (the saved DataFrames are already processed).
        """

        if isinstance(model, MetapopModel):
            subpop_models = model.subpop_models
        else:
            subpop_models = {model.name: model}

        for (subpop_name, schedule_name), original_df in saved["schedules"].items():
            subpop_models[subpop_name].schedules[schedule_name].timeseries_df = \
                original_df

        for subpop_name, original_params in saved["params"].items():
            subpop_models[subpop_name].params = original_params

    def _apply_scenario(self,
                        model: MetapopModel | SubpopModel,
                        scenario_def: dict) -> None:
        """Apply schedule and parameter overrides from *scenario_def* to *model*."""

        schedules = scenario_def.get("schedules", {})
        subpop_schedules = scenario_def.get("subpop_schedules", {})
        params = scenario_def.get("params", {})

        for schedule_name, new_df in schedules.items():
            model.replace_schedule(schedule_name, new_df)

        for subpop_name, sched_map in subpop_schedules.items():
            if isinstance(model, MetapopModel):
                if subpop_name not in model.subpop_models:
                    raise ScenarioRunnerError(
                        f"Scenario references unknown subpopulation '{subpop_name}'."
                    )
                for schedule_name, new_df in sched_map.items():
                    model.replace_schedule(schedule_name, new_df,
                                           subpop_name=subpop_name)
            else:
                for schedule_name, new_df in sched_map.items():
                    model.replace_schedule(schedule_name, new_df)

        for subpop_name, updates_dict in params.items():
            if isinstance(model, MetapopModel):
                if subpop_name not in model.subpop_models:
                    raise ScenarioRunnerError(
                        f"Scenario references unknown subpopulation '{subpop_name}'."
                    )
                subpop = model.subpop_models[subpop_name]
                subpop.params = updated_dataclass(subpop.params, updates_dict)
            else:
                # SubpopModel — ignore subpop_name, apply directly
                model.params = updated_dataclass(model.params, updates_dict)

    def _run_scenario(self,
                      model: MetapopModel | SubpopModel,
                      scenario_name: str,
                      num_reps: int,
                      end_day: int,
                      days_per_save: int,
                      seeds: Optional[list],
                      conn: sqlite3.Connection,
                      cursor: sqlite3.Cursor) -> None:
        """Simulate *num_reps* replicates of *model* and log to *cursor*."""

        # Disable automatic daily history saving — we log manually
        if isinstance(model, MetapopModel):
            subpop_models = list(model.subpop_models.values())
        else:
            subpop_models = [model]

        for subpop_model in subpop_models:
            subpop_model.simulation_settings = updated_dataclass(
                subpop_model.simulation_settings, {"save_daily_history": False}
            )

        for rep in range(num_reps):
            if seeds is not None:
                model.modify_random_seed(seeds[rep])

            model.reset_simulation()

            # Accumulate all rows for this replicate, then flush in one call
            batch = []
            while model.current_simulation_day < end_day:
                model.simulate_until_day(
                    min(model.current_simulation_day + days_per_save, end_day)
                )
                for subpop_model in subpop_models:
                    for state_var_name in self.state_variables_to_record:
                        batch.extend(_format_row_for_sql(
                            subpop_model, state_var_name, scenario_name, rep
                        ))

            cursor.executemany(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
            )
            conn.commit()
