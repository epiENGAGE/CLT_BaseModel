"""Helpers shared by prepare_hospitalizations.py and prepare_vaccination.py:
age-group label normalization, county-name normalization, and the optional
county -> group mapping.

Age groups are always emitted in the model's order (age_risk.age_groups in
model_config_MA_vax.json): 0-0, 1-4, 5-12, 13-17, 18-49, 50-64, 65+.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# Column / file-suffix keys, in model age order.
AGE_KEYS = ["0_0", "1_4", "5_12", "13_17", "18_49", "50_64", "65+"]

# Raw label (after normalize_age_label) -> age key. Covers both source
# files ("Under 1 years", "1-4 years", ...; "<1", "1-4", ...) plus the
# Excel-mangled forms the admissions file has ("4-Jan" = 1-4,
# "12-May" = 5-12).
_AGE_ALIASES = {
    "under1": "0_0", "<1": "0_0", "0": "0_0", "0-0": "0_0", "0_0": "0_0",
    "1-4": "1_4", "4-jan": "1_4", "jan-4": "1_4",
    "5-12": "5_12", "12-may": "5_12", "may-12": "5_12",
    "13-17": "13_17",
    "18-49": "18_49",
    "50-64": "50_64",
    "65+": "65+", "65plus": "65+",
}

STATE_GROUP = "MA"


def normalize_age_label(label: str) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"\s*years?$", "", s)
    s = s.replace(" ", "")
    if s not in _AGE_ALIASES:
        raise ValueError(f"unrecognized age group label: {label!r}")
    return _AGE_ALIASES[s]


def normalize_county(name: str) -> str:
    """'MA_CapeandIslands', 'Cape and Islands', 'capeandislands' -> 'capeandislands'."""
    s = str(name).strip()
    s = re.sub(r"^MA[_\s-]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+county$", "", s, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_county_mapping(path: str | Path | None) -> dict[str, str] | None:
    """Read a `county,group` CSV into {normalized county: group}.

    County names are normalized (see normalize_county), so the same mapping
    works for both source files ('Berkshire' and 'MA_Berkshire'), and may
    list either form. None -> no mapping (everything goes to STATE_GROUP).
    """
    if path is None:
        return None
    df = pd.read_csv(path, dtype=str)
    missing = {"county", "group"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: mapping CSV needs columns 'county,group' (missing {sorted(missing)})")
    mapping: dict[str, str] = {}
    for county, group in zip(df["county"], df["group"]):
        key = normalize_county(county)
        group = str(group).strip()
        if key in mapping and mapping[key] != group:
            raise ValueError(f"{path}: county {county!r} mapped to both {mapping[key]!r} and {group!r}")
        mapping[key] = group
    return mapping


def assign_groups(counties: pd.Series, mapping: dict[str, str] | None) -> pd.Series:
    """Group label per row; every county must be in the mapping."""
    if mapping is None:
        return pd.Series(STATE_GROUP, index=counties.index)
    keys = counties.map(normalize_county)
    unmapped = sorted(set(counties[~keys.isin(mapping.keys())]))
    if unmapped:
        raise ValueError(f"counties missing from the mapping CSV: {unmapped}")
    return keys.map(mapping)


def output_root(group: str, mapping_used: bool, out_root: str | Path | None) -> Path:
    """Folder holding a group's hospitalizations_*/ and schedules/ outputs.

    No mapping -> this data/ folder itself (the paths the pipeline reads).
    With a mapping -> data/by_group/<group>/, same sub-structure.
    """
    base = Path(out_root) if out_root is not None else HERE
    return base / "by_group" / group if mapping_used else base


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y")


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse a date column written either as YYYY-MM-DD or as m/d/yy(yy) --
    the raw downloads use ISO dates, copies re-saved by Excel use m/d/yy.
    The whole column must use one format."""
    s = s.astype(str).str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"dates in column {s.name!r} match none of {_DATE_FORMATS} "
                     f"(first values: {list(s.head(3))})")


def week_days(week_end: pd.Timestamp) -> pd.DatetimeIndex:
    """The 7 days of the week ending on `week_end` (Sun..Sat for MMWR weeks)."""
    return pd.date_range(end=week_end, periods=7, freq="D")


def fmt_date(d: pd.Series | pd.DatetimeIndex) -> list[str]:
    """m/d/yy, no zero padding -- the format the existing schedule CSVs use."""
    return [f"{x.month}/{x.day}/{x.year % 100:02d}" for x in pd.DatetimeIndex(d)]
