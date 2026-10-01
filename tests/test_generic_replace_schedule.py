"""
Tests for `ConfigDrivenSubpopModel.replace_schedule`.

Generic schedules do all their preprocessing (date parsing, JSON decoding,
protection-delay shift, date indexing) in their template's `build_schedule`,
not in `postprocess_data_input` -- so a replaced DataFrame must be rebuilt
through the template, exactly as at construction. In particular, "date"
strings must be accepted, as they are at construction.
"""

import copy
import datetime

import numpy as np
import pytest

import clt_toolkit as clt

from test_generic_vaccine_efficacy import (
    COMPARTMENTS,
    _make_model,
    _start_date,
    _constant_vaccines_df,
)

DETERMINISTIC = {"transition_type": "binom_deterministic_no_round"}


def _vaccines_df(model, dose=0.01):
    """String-dated `daily_vaccines` input starting 30 days before the start."""
    start = _start_date(model.simulation_settings)
    return _constant_vaccines_df(start - datetime.timedelta(days=30), 200, dose,
                                 model.params.num_age_groups,
                                 model.params.num_risk_groups)


def test_replaced_vaccine_schedule_matches_model_built_with_it():
    """
    Replacing the vaccine schedule (string dates) and resetting gives the
    same trajectory as a model constructed with that schedule from the
    start -- the replaced DataFrame is preprocessed (dates parsed, JSON
    decoded, protection delay applied, date-indexed) the same way.
    """
    params = {"vax_protection_delay_days": 7}

    replaced = _make_model(params, settings_updates=DETERMINISTIC)
    new_df = _vaccines_df(replaced)
    replaced.replace_schedule("daily_vaccines", new_df)
    replaced.reset_simulation()

    built = _make_model(params, vaccines_df=new_df, settings_updates=DETERMINISTIC)

    replaced_ts = replaced.schedules["daily_vaccines"].timeseries_df
    built_ts = built.schedules["daily_vaccines"].timeseries_df
    assert isinstance(replaced_ts.index[0], datetime.date)
    assert list(replaced_ts.index) == list(built_ts.index)

    replaced.simulate_until_day(20)
    built.simulate_until_day(20)

    for name in COMPARTMENTS + ["MV"]:
        replaced_var = replaced.compartments.get(name) or replaced.epi_metrics[name]
        built_var = built.compartments.get(name) or built.epi_metrics[name]
        assert np.allclose(np.asarray(replaced_var.history_vals_list),
                           np.asarray(built_var.history_vals_list)), name


def test_replace_schedule_keeps_schedule_object_and_does_not_mutate_input():
    """
    The existing schedule object is updated in place (code holding a
    reference to it sees the new data), and the caller's DataFrame is left
    untouched -- the same DataFrame may be passed to several subpopulations.
    """
    model = _make_model(settings_updates=DETERMINISTIC)
    schedule = model.schedules["daily_vaccines"]

    new_df = _vaccines_df(model, dose=0.02)
    new_df_before = copy.deepcopy(new_df)

    model.replace_schedule("daily_vaccines", new_df)

    assert model.schedules["daily_vaccines"] is schedule
    assert np.allclose(schedule.timeseries_df["daily_vaccines"].iloc[-1], 0.02)
    assert new_df.equals(new_df_before)


def test_replace_contact_matrix_schedule_with_string_dates():
    """
    The contact matrix schedule is built from its school/work-day calendar
    (`school_work_day_df_attribute`, not `df_attribute`) -- replacing that
    calendar with string dates works, and the new calendar takes effect.
    """
    model = _make_model(settings_updates=DETERMINISTIC)
    schedule = model.schedules["flu_contact_matrix"]

    # Same dates as the original calendar, as strings, with no school or
    #   work on any day
    calendar = schedule.timeseries_df.reset_index()
    calendar["date"] = [d.strftime("%Y-%m-%d") for d in calendar["date"]]
    calendar["is_school_day"] = 0
    calendar["is_work_day"] = 0

    model.replace_schedule("flu_contact_matrix", calendar)

    assert isinstance(schedule.timeseries_df.index[0], datetime.date)

    model.prepare_daily_state()
    expected = schedule.total_matrix - schedule.school_matrix - schedule.work_matrix
    assert np.allclose(schedule.current_val, expected)


def test_replace_unknown_schedule_raises():
    model = _make_model(settings_updates=DETERMINISTIC)

    with pytest.raises(clt.SubpopModelError):
        model.replace_schedule("not_a_schedule", _vaccines_df(model))
