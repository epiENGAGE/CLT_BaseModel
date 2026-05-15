"""
generate_example_metapop_data.py
=================================

Generates synthetic input files for a 2-subpopulation, 3-age-group,
2-risk-group SEIR model. Run from the project root::

    python generic_core/examples/generate_example_metapop_data.py

Output is written to::

    generic_core/examples/example_metapop_inputs/

Files produced
--------------
Required (metapop structure):
  subpopulations.csv
  travel_matrix.csv

Shared schedules (all subpops):
  absolute_humidity.csv      — seasonal cosine wave
  mobility_modifier.csv      — day-of-week modifiers per age×risk group

Per-subpop schedules:
  school_work_calendar_SubpopA.csv
  school_work_calendar_SubpopB.csv
  vaccines_SubpopA.csv
  vaccines_SubpopB.csv
  initial_conditions_SubpopA.csv
  initial_conditions_SubpopB.csv

Contact matrices (A×A plain floats, no headers):
  total_contact_matrix.csv
  school_contact_matrix.csv
  work_contact_matrix.csv

Population structure
--------------------
  Age groups  : 0-17 (young), 18-64 (adult), 65+ (older)
  Risk groups : low-risk, high-risk
  SubpopA     : 100 000 total  (urban)
  SubpopB     : 150 000 total  (suburban)
"""

import datetime
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "example_metapop_inputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation horizon
# ---------------------------------------------------------------------------
START_DATE = datetime.date(2024, 1, 1)
N_DAYS = 365
DATES = [START_DATE + datetime.timedelta(days=i) for i in range(N_DAYS)]

# ---------------------------------------------------------------------------
# Population structure
# ---------------------------------------------------------------------------
# 3 age groups × 2 risk groups
A, R = 3, 2
SUBPOPS = [
    {"name": "SubpopA", "total_population": 100_000},
    {"name": "SubpopB", "total_population": 150_000},
]

# Age-group fraction of total population (young, adult, older)
AGE_FRACS = np.array([0.15, 0.65, 0.20])
# Risk-group fraction within each age group (low-risk, high-risk)
RISK_FRACS = np.array([0.80, 0.20])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arr_json(arr: np.ndarray) -> str:
    """Encode an A×R numpy array as a compact JSON string."""
    return json.dumps(arr.tolist())


def _is_school_day(date: datetime.date) -> float:
    """Return 1.0 on school days, 0.0 otherwise (simplified US academic calendar)."""
    if date.weekday() >= 5:
        return 0.0
    # Summer: July, August
    if date.month in (7, 8):
        return 0.0
    # Major holidays (approximate)
    _holidays = {
        datetime.date(2024, 1, 1),   # New Year's
        datetime.date(2024, 5, 27),  # Memorial Day
        datetime.date(2024, 7, 4),   # Independence Day
        datetime.date(2024, 11, 28), # Thanksgiving
        datetime.date(2024, 11, 29), # Thanksgiving Friday
        datetime.date(2024, 12, 25), # Christmas
    }
    if date in _holidays:
        return 0.0
    return 1.0


def _is_work_day(date: datetime.date) -> float:
    """Return 1.0 on weekdays, 0.0 on weekends/major holidays."""
    if date.weekday() >= 5:
        return 0.0
    _holidays = {
        datetime.date(2024, 1, 1),
        datetime.date(2024, 5, 27),
        datetime.date(2024, 7, 4),
        datetime.date(2024, 11, 28),
        datetime.date(2024, 12, 25),
    }
    return 0.0 if date in _holidays else 1.0


# ---------------------------------------------------------------------------
# subpopulations.csv
# ---------------------------------------------------------------------------
pd.DataFrame(SUBPOPS).to_csv(os.path.join(OUT_DIR, "subpopulations.csv"), index=False)
print("Wrote subpopulations.csv")

# ---------------------------------------------------------------------------
# travel_matrix.csv  (rows sum to 1)
# ---------------------------------------------------------------------------
# SubpopA: 95% stay, 5% visit SubpopB
# SubpopB: 3% visit SubpopA, 97% stay
_travel = pd.DataFrame({"SubpopA": [0.95, 0.03], "SubpopB": [0.05, 0.97]})
_travel.index = pd.Index(["SubpopA", "SubpopB"])  # type: ignore[assignment]
_travel.to_csv(os.path.join(OUT_DIR, "travel_matrix.csv"))
print("Wrote travel_matrix.csv")

# ---------------------------------------------------------------------------
# absolute_humidity.csv  (seasonal cosine, range 0.004 – 0.014)
# ---------------------------------------------------------------------------
_t = np.arange(N_DAYS)
# Peak in summer (day ~180), trough in winter
_ah = 0.009 + 0.005 * np.cos(2 * np.pi * (_t - 180) / 365)
pd.DataFrame({"date": DATES, "absolute_humidity": _ah}).to_csv(
    os.path.join(OUT_DIR, "absolute_humidity.csv"), index=False
)
print("Wrote absolute_humidity.csv")

# ---------------------------------------------------------------------------
# mobility_modifier.csv  (day-of-week, A×R JSON blobs)
# ---------------------------------------------------------------------------
# Mobility modifiers by age group and risk group (A=3, R=2)
# Older adults (65+) have lower weekend mobility; high-risk groups similar to low-risk
_mob_by_dow = {
    "Monday":    np.array([[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]),
    "Tuesday":   np.array([[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]),
    "Wednesday": np.array([[0.97, 0.95], [0.97, 0.95], [0.88, 0.88]]),
    "Thursday":  np.array([[0.97, 0.95], [0.97, 0.95], [0.88, 0.88]]),
    "Friday":    np.array([[0.97, 0.95], [0.97, 0.95], [0.90, 0.90]]),
    "Saturday":  np.array([[0.75, 0.73], [0.75, 0.73], [0.60, 0.60]]),
    "Sunday":    np.array([[0.65, 0.63], [0.65, 0.63], [0.55, 0.55]]),
}
pd.DataFrame({
    "day_of_week": list(_mob_by_dow.keys()),
    "mobility_modifier": [_arr_json(v) for v in _mob_by_dow.values()],
}).to_csv(os.path.join(OUT_DIR, "mobility_modifier.csv"), index=False)
print("Wrote mobility_modifier.csv")

# ---------------------------------------------------------------------------
# Contact matrices (A×A, plain floats, no header)
# Polymod-inspired: rows = age of contact initiator, cols = age of contact
# ---------------------------------------------------------------------------
# Total contacts (symmetric)
_total_contact = np.array([
    [7.0, 3.0, 0.5],
    [3.0, 9.0, 1.5],
    [0.5, 1.5, 4.0],
])
# School contacts (mostly young–young)
_school_contact = np.array([
    [3.0, 0.5, 0.0],
    [0.5, 0.2, 0.0],
    [0.0, 0.0, 0.0],
])
# Work contacts (mostly adult–adult)
_work_contact = np.array([
    [0.0, 0.2, 0.0],
    [0.2, 4.0, 0.3],
    [0.0, 0.3, 0.5],
])

for _fname, _mat in [
    ("total_contact_matrix.csv", _total_contact),
    ("school_contact_matrix.csv", _school_contact),
    ("work_contact_matrix.csv", _work_contact),
]:
    pd.DataFrame(_mat).to_csv(os.path.join(OUT_DIR, _fname), index=False, header=False)
    print(f"Wrote {_fname}")

# ---------------------------------------------------------------------------
# Per-subpop files
# ---------------------------------------------------------------------------
for _sp in SUBPOPS:
    _name = _sp["name"]
    _total = _sp["total_population"]

    # school_work_calendar_{name}.csv
    pd.DataFrame({
        "date": DATES,
        "is_school_day": [_is_school_day(d) for d in DATES],
        "is_work_day": [_is_work_day(d) for d in DATES],
    }).to_csv(os.path.join(OUT_DIR, f"school_work_calendar_{_name}.csv"), index=False)
    print(f"Wrote school_work_calendar_{_name}.csv")

    # vaccines_{name}.csv
    # Daily vaccines (A×R matrix) — proportion of each age×risk group vaccinated
    # per day. Higher for older adults and high-risk groups.
    # Computed as: base_count / (AGE_FRAC * RISK_FRAC * total_population)
    _pop_ar = np.outer(AGE_FRACS, RISK_FRACS) * _total
    _base_counts = np.array([
        [5.0,  2.0],   # young: low-risk, high-risk
        [15.0, 8.0],   # adult
        [25.0, 12.0],  # older
    ])
    _vax_ar = _base_counts / _pop_ar
    pd.DataFrame({
        "date": DATES,
        "daily_vaccines": [_arr_json(_vax_ar)] * N_DAYS,
    }).to_csv(os.path.join(OUT_DIR, f"vaccines_{_name}.csv"), index=False)
    print(f"Wrote vaccines_{_name}.csv")

    # initial_conditions_{name}.csv
    # SubpopA: 10 exposed (seed), everyone else susceptible
    # SubpopB: fully susceptible
    _E_seed = 10 if _name == "SubpopA" else 0
    _S_init = _total - _E_seed
    pd.DataFrame({
        "compartment": ["S", "E", "I", "R"],
        "value": [_S_init, _E_seed, 0, 0],
    }).to_csv(os.path.join(OUT_DIR, f"initial_conditions_{_name}.csv"), index=False)
    print(f"Wrote initial_conditions_{_name}.csv")

print(f"\nAll files written to: {OUT_DIR}")
print("Use this folder as the metapop folder path in model_builder_notebook.py")
print(f"Contact matrix files are {A}×{A} plain-float CSVs (no header row).")
