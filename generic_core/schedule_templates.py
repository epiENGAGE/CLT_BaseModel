"""
schedule_templates.py — ScheduleTemplate ABC and built-in concrete implementations.

Each template is a factory that constructs a concrete clt.Schedule subclass.
The built-in templates mirror the flu-specific schedule classes in
flu_core/flu_components.py, but accept config dicts instead of typed params.

Registry
--------
SCHEDULE_TEMPLATE_REGISTRY : dict[str, ScheduleTemplate]
    Built-in templates are registered at import time.
    Users call register_schedule_template() to add custom templates.

Reference implementations:
    DailyVaccines       — flu_core/flu_components.py:596–669
    MobilityModifier    — flu_core/flu_components.py:672+
    FluContactMatrix    — flu_core/flu_components.py:775–814
    AbsoluteHumidity    — flu_core/flu_components.py:746–772 (simple timeseries)
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

import clt_toolkit as clt


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ScheduleTemplate(ABC):
    """
    Abstract base class for schedule templates.

    Subclasses implement:
        validate_config  — called at model construction to catch config errors
        build_schedule   — factory method that returns a concrete Schedule instance
    """

    @abstractmethod
    def validate_config(
        self,
        schedule_config: dict,
        param_names: set,
        schedules_input,
    ) -> None:
        """
        Validate schedule_config.
        Raises ValueError with a descriptive message on the first problem found.
        """

    @abstractmethod
    def build_schedule(
        self,
        schedule_config: dict,
        params,
        schedules_input,
    ) -> clt.Schedule:
        """
        Construct and return a concrete Schedule instance.

        Parameters
        ----------
        schedule_config : dict
            Template-specific config from the model config JSON.
        params : GenericSubpopParams
            Model parameters (some schedules need param values at build time).
        schedules_input : FluSubpopSchedules or equivalent
            Raw input data object providing DataFrames for timeseries schedules.
        """


# ---------------------------------------------------------------------------
# Date-column helper
# ---------------------------------------------------------------------------

def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'date' column to datetime.date objects if it exists and
    contains strings.  Mirrors the date-conversion step in
    FluSubpopModel.create_schedules().  No-op if 'date' is absent or already
    a proper type.
    """
    if "date" not in df.columns:
        return df
    first = df["date"].iloc[0]
    if isinstance(first, str):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
    return df


# ---------------------------------------------------------------------------
# TimeseriesLookupSchedule — generic date-indexed timeseries
# ---------------------------------------------------------------------------

class TimeseriesLookupScheduleGeneric(clt.Schedule):
    """
    Schedule backed by a date-indexed DataFrame column.

    Supports any single-column timeseries: absolute humidity, school/work-day
    flags, or other scalar/array values.

    schedule_config keys
    --------------------
    df_attribute : str
        Attribute name on schedules_input that holds the DataFrame.
    value_column : str
        Column name in the DataFrame containing the schedule values.
    """

    def __init__(
        self,
        init_val: Any,
        timeseries_df: pd.DataFrame,
        value_column: str,
    ):
        super().__init__(init_val)
        self.timeseries_df = timeseries_df
        self.value_column = value_column

    def update_current_val(self, params: Any, current_date: datetime.date) -> None:
        try:
            self.current_val = self.timeseries_df.loc[current_date, self.value_column]
        except KeyError:
            pass  # keep previous value if date not found


class TimeseriesLookupTemplate(ScheduleTemplate):
    """
    Factory for TimeseriesLookupScheduleGeneric.

    schedule_config keys
    --------------------
    df_attribute : str
        Attribute on schedules_input holding the DataFrame.
    value_column : str
        Column in that DataFrame holding the values.
    init_val : optional
        Starting value. If absent, the first row's value is used.
    """

    def validate_config(self, schedule_config, param_names, schedules_input):
        for key in ("df_attribute", "value_column"):
            if key not in schedule_config:
                raise ValueError(
                    f"TimeseriesLookupTemplate: missing required key '{key}' in schedule_config"
                )
        attr = schedule_config["df_attribute"]
        if not hasattr(schedules_input, attr):
            raise ValueError(
                f"TimeseriesLookupTemplate: schedules_input has no attribute '{attr}'"
            )
        df = getattr(schedules_input, attr)
        col = schedule_config["value_column"]
        if col not in df.columns:
            raise ValueError(
                f"TimeseriesLookupTemplate: column '{col}' not found in DataFrame '{attr}'"
            )

    def build_schedule(self, schedule_config, params, schedules_input):
        attr = schedule_config["df_attribute"]
        col = schedule_config["value_column"]
        df = _normalize_date_column(getattr(schedules_input, attr).copy())
        if "date" in df.columns:
            df = df.set_index("date")
        init_val = schedule_config.get("init_val", df[col].iloc[0])
        return TimeseriesLookupScheduleGeneric(init_val, df, col)


# ---------------------------------------------------------------------------
# ContactMatrixScheduleGeneric — school/work-day contact matrix interpolation
# ---------------------------------------------------------------------------

class ContactMatrixScheduleGeneric(clt.Schedule):
    """
    Schedule that returns the current A×A contact matrix, adjusted for
    whether the current day is a school day and/or work day.

    Formula (mirrors FluContactMatrix.update_current_val):
        matrix = total_contact_matrix
                 - (1 − is_school_day) × school_contact_matrix
                 - (1 − is_work_day)   × work_contact_matrix

    schedule_config keys (resolved from params at build time):
        total_contact_matrix_param  : str
        school_contact_matrix_param : str
        work_contact_matrix_param   : str
        school_work_day_df_attribute: str  — attribute on schedules_input
    """

    def __init__(
        self,
        init_val: np.ndarray,
        timeseries_df: pd.DataFrame,
        total_matrix: np.ndarray,
        school_matrix: np.ndarray,
        work_matrix: np.ndarray,
    ):
        super().__init__(init_val)
        self.timeseries_df = timeseries_df
        self.total_matrix = total_matrix
        self.school_matrix = school_matrix
        self.work_matrix = work_matrix

    def update_current_val(self, params: Any, current_date: datetime.date) -> None:
        try:
            row = self.timeseries_df.loc[current_date]
            self.current_val = (
                self.total_matrix
                - (1.0 - row["is_school_day"]) * self.school_matrix
                - (1.0 - row["is_work_day"]) * self.work_matrix
            )
        except KeyError:
            self.current_val = self.total_matrix


class ContactMatrixTemplate(ScheduleTemplate):
    """
    Factory for ContactMatrixScheduleGeneric.

    schedule_config keys
    --------------------
    school_work_day_df_attribute : str
        Attribute on schedules_input with a DataFrame having columns
        "date", "is_school_day", "is_work_day".
    total_contact_matrix_param : str
        Parameter name for the A×A total contact matrix.
    school_contact_matrix_param : str
        Parameter name for the A×A school contact matrix.
    work_contact_matrix_param : str
        Parameter name for the A×A work contact matrix.
    """

    _REQUIRED = (
        "school_work_day_df_attribute",
        "total_contact_matrix_param",
        "school_contact_matrix_param",
        "work_contact_matrix_param",
    )

    def validate_config(self, schedule_config, param_names, schedules_input):
        for key in self._REQUIRED:
            if key not in schedule_config:
                raise ValueError(
                    f"ContactMatrixTemplate: missing required key '{key}' in schedule_config"
                )
        for pkey in ("total_contact_matrix_param", "school_contact_matrix_param",
                     "work_contact_matrix_param"):
            pname = schedule_config[pkey]
            if pname not in param_names:
                raise ValueError(
                    f"ContactMatrixTemplate: param '{pname}' (from '{pkey}') not in model params"
                )
        attr = schedule_config["school_work_day_df_attribute"]
        if not hasattr(schedules_input, attr):
            raise ValueError(
                f"ContactMatrixTemplate: schedules_input has no attribute '{attr}'"
            )

    def build_schedule(self, schedule_config, params, schedules_input):
        attr = schedule_config["school_work_day_df_attribute"]
        df = _normalize_date_column(getattr(schedules_input, attr).copy())
        if "date" in df.columns:
            df = df.set_index("date")

        total = params.params[schedule_config["total_contact_matrix_param"]]
        school = params.params[schedule_config["school_contact_matrix_param"]]
        work = params.params[schedule_config["work_contact_matrix_param"]]

        # init_val: use total matrix as default
        return ContactMatrixScheduleGeneric(total, df, total, school, work)


# ---------------------------------------------------------------------------
# VaccineScheduleGeneric — delay shift + backfill + date-indexed lookup
# ---------------------------------------------------------------------------

class VaccineScheduleGeneric(clt.Schedule):
    """
    Schedule for daily vaccination counts, with optional protection delay
    and backfill.

    Mirrors DailyVaccines from flu_components.py:596–669.
    The timeseries_df is already preprocessed (date-shifted, backfilled,
    and index-set) at construction time.
    """

    def __init__(self, init_val: Any, timeseries_df: pd.DataFrame):
        super().__init__(init_val)
        self.timeseries_df = timeseries_df

    def update_current_val(self, params: Any, current_date: datetime.date) -> None:
        self.current_val = self.timeseries_df.loc[current_date, "daily_vaccines"]

    @staticmethod
    def preprocess(df: pd.DataFrame, delay_days: int) -> pd.DataFrame:
        """
        Apply delay shift, backfill, and index the DataFrame.

        Mirrors DailyVaccines.postprocess_data_input.
        """
        df = _normalize_date_column(df.copy())
        df["daily_vaccines"] = df["daily_vaccines"].apply(json.loads)
        df["daily_vaccines"] = df["daily_vaccines"].apply(np.asarray)

        if delay_days > 0:
            original_start = df["date"].min()
            zero_array = np.zeros_like(df["daily_vaccines"].iloc[0])
            df["date"] = df["date"].apply(
                lambda d: d + datetime.timedelta(days=delay_days)
            )
            backfill_dates = [
                original_start + datetime.timedelta(days=i) for i in range(delay_days)
            ]
            backfill_df = pd.DataFrame({
                "date": backfill_dates,
                "daily_vaccines": [zero_array.copy() for _ in range(delay_days)],
            })
            df = pd.concat([backfill_df, df], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)

        return df.set_index("date")


class VaccineScheduleTemplate(ScheduleTemplate):
    """
    Factory for VaccineScheduleGeneric.

    schedule_config keys
    --------------------
    df_attribute : str
        Attribute on schedules_input holding the raw DataFrame
        (columns: "date", "daily_vaccines").
    vax_protection_delay_days_param : str | None
        Parameter name for the protection delay in days.
        If absent or None, delay = 0.
    """

    def validate_config(self, schedule_config, param_names, schedules_input):
        if "df_attribute" not in schedule_config:
            raise ValueError("VaccineScheduleTemplate: missing 'df_attribute' in schedule_config")
        attr = schedule_config["df_attribute"]
        if not hasattr(schedules_input, attr):
            raise ValueError(
                f"VaccineScheduleTemplate: schedules_input has no attribute '{attr}'"
            )
        delay_param = schedule_config.get("vax_protection_delay_days_param")
        if delay_param is not None and delay_param not in param_names:
            raise ValueError(
                f"VaccineScheduleTemplate: param '{delay_param}' not in model params"
            )

    def build_schedule(self, schedule_config, params, schedules_input):
        attr = schedule_config["df_attribute"]
        df_raw = getattr(schedules_input, attr).copy()

        delay_param = schedule_config.get("vax_protection_delay_days_param")
        delay_days = 0
        if delay_param is not None:
            delay_days = int(params.params.get(delay_param, 0))

        df = VaccineScheduleGeneric.preprocess(df_raw, delay_days)
        init_val = df["daily_vaccines"].iloc[0]
        return VaccineScheduleGeneric(init_val, df)


# ---------------------------------------------------------------------------
# MobilityScheduleGeneric — date or day-of-week mobility modifier
# ---------------------------------------------------------------------------

class MobilityScheduleGeneric(clt.Schedule):
    """
    Schedule for mobility modifier values (A×R arrays).

    Mirrors MobilityModifier from flu_components.py:672+.
    Supports both date-indexed and day-of-week-indexed schedules.
    """

    def __init__(
        self,
        init_val: Any,
        timeseries_df: pd.DataFrame,
        is_day_of_week: bool,
    ):
        super().__init__(init_val)
        self.timeseries_df = timeseries_df
        self.is_day_of_week = is_day_of_week

    def update_current_val(self, params: Any, current_date: datetime.date) -> None:
        if self.is_day_of_week:
            key = current_date.strftime("%A").lower()
        else:
            key = current_date
        self.current_val = self.timeseries_df.loc[key, "mobility_modifier"]


class MobilityScheduleTemplate(ScheduleTemplate):
    """
    Factory for MobilityScheduleGeneric.

    schedule_config keys
    --------------------
    df_attribute : str
        Attribute on schedules_input holding the DataFrame.
        Must have either ("date", "mobility_modifier") columns
        or ("day_of_week", "mobility_modifier") columns.
    """

    def validate_config(self, schedule_config, param_names, schedules_input):
        if "df_attribute" not in schedule_config:
            raise ValueError("MobilityScheduleTemplate: missing 'df_attribute' in schedule_config")
        attr = schedule_config["df_attribute"]
        if not hasattr(schedules_input, attr):
            raise ValueError(
                f"MobilityScheduleTemplate: schedules_input has no attribute '{attr}'"
            )

    def build_schedule(self, schedule_config, params, schedules_input):
        attr = schedule_config["df_attribute"]
        df = getattr(schedules_input, attr).copy()

        is_day_of_week = "day_of_week" in df.columns
        if not is_day_of_week:
            df = _normalize_date_column(df)
        df["mobility_modifier"] = df["mobility_modifier"].apply(json.loads)
        df["mobility_modifier"] = df["mobility_modifier"].apply(np.asarray)

        if is_day_of_week:
            df["day_of_week"] = df["day_of_week"].str.lower()
            df = df.set_index("day_of_week")
        else:
            df = df.set_index("date")

        init_val = df["mobility_modifier"].iloc[0]
        return MobilityScheduleGeneric(init_val, df, is_day_of_week)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULE_TEMPLATE_REGISTRY: dict[str, ScheduleTemplate] = {}


def register_schedule_template(name: str, instance: ScheduleTemplate) -> None:
    """
    Add a ScheduleTemplate instance to the global registry under `name`.

    Built-in templates are registered at import time.
    """
    SCHEDULE_TEMPLATE_REGISTRY[name] = instance


register_schedule_template("timeseries_lookup", TimeseriesLookupTemplate())
register_schedule_template("contact_matrix", ContactMatrixTemplate())
register_schedule_template("vaccine_schedule", VaccineScheduleTemplate())
register_schedule_template("mobility", MobilityScheduleTemplate())
