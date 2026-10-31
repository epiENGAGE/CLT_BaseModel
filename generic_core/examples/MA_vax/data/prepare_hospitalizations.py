"""Turn the weekly county x age admissions file into the model's daily
hospitalization inputs (fit targets).

Input: CSV with columns `mmwr_week_ending,agegroup,facility_county,ed_admits_count`
(weekly counts, county names like `MA_Berkshire`). A week x age x county
combination with no row counts as 0 admissions.

For each county group (the whole state by default, or the groups of
--mapping) it:
  1. sums counts over the group's counties, per week and age group;
  2. spreads each weekly count evenly over the 7 days ending on
     mmwr_week_ending (count / 7 per day);
  3. writes hospitalizations_ts/MA_flu_daily_hospitalizations.csv
     (`Date,0_0,1_4,5_12,13_17,18_49,50_64,65+`, dates m/d/yy);
  4. runs split_hospitalizations_by_age.py and
     compute_cumulative_hospitalizations.py on it, which write the per-age
     daily / cumulative files and the end-of-season-by-age file used as fit
     targets.

Without --mapping, outputs go to this data/ folder (where the pipeline reads
them). With --mapping, each group goes to data/by_group/<group>/ with the
same sub-folders.

Run (from the repo root):
    python generic_core/examples/MA_vax/data/prepare_hospitalizations.py
    python generic_core/examples/MA_vax/data/prepare_hospitalizations.py \\
        --mapping generic_core/examples/MA_vax/data/county_groups_example.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import compute_cumulative_hospitalizations
import split_hospitalizations_by_age
from prep_common import (
    AGE_KEYS, HERE, assign_groups, fmt_date, load_county_mapping,
    normalize_age_label, output_root, parse_dates, week_days,
)

DEFAULT_INPUT = HERE / "original" / "influenza_timeseries_30Jun2019-04Jul2026_2026-09-10.csv"
DATE_COL, AGE_COL, COUNTY_COL, COUNT_COL = "mmwr_week_ending", "agegroup", "facility_county", "ed_admits_count"


def weekly_by_group(raw: pd.DataFrame, mapping, start, end) -> pd.DataFrame:
    """Weekly counts, index (group, week_end), one column per age key, zero-filled."""
    df = raw.copy()
    df["week_end"] = parse_dates(df[DATE_COL])
    df["age"] = df[AGE_COL].map(normalize_age_label)
    df["group"] = assign_groups(df[COUNTY_COL], mapping)

    weeks = pd.date_range(df["week_end"].min(), df["week_end"].max(), freq="7D")
    if not set(df["week_end"]) <= set(weeks):
        raise ValueError(f"{DATE_COL} values are not all on the same weekday")
    weeks = weeks[(weeks >= start) & (weeks <= end)]
    if len(weeks) == 0:
        raise ValueError(f"no weeks ending between {start.date()} and {end.date()}")
    df = df[df["week_end"].isin(weeks)]

    groups = sorted(set(assign_groups(raw[COUNTY_COL], mapping)))
    full = pd.MultiIndex.from_product([groups, weeks, AGE_KEYS], names=["group", "week_end", "age"])
    counts = (df.groupby(["group", "week_end", "age"])[COUNT_COL].sum()
                .reindex(full, fill_value=0))
    return counts.unstack("age")[AGE_KEYS]


def to_daily(weekly: pd.DataFrame) -> pd.DataFrame:
    """One group's weekly table (index week_end) -> daily rows of count / 7."""
    days = []
    for week_end, row in weekly.iterrows():
        for d in week_days(week_end):
            days.append({"date": d, **(row / 7.0).to_dict()})
    return pd.DataFrame(days)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="weekly admissions CSV")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="CSV with columns county,group; default: aggregate the whole state")
    ap.add_argument("--start", default="2025-06-29",
                    help="keep weeks ending on/after this date (YYYY-MM-DD); default 2025-06-29")
    ap.add_argument("--end", default="2026-07-04",
                    help="keep weeks ending on/before this date (YYYY-MM-DD); default 2026-07-04")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write under this folder instead of data/ (same sub-structure)")
    args = ap.parse_args()

    raw = pd.read_csv(args.input)
    missing = {DATE_COL, AGE_COL, COUNTY_COL, COUNT_COL} - set(raw.columns)
    if missing:
        raise SystemExit(f"{args.input}: missing columns {sorted(missing)}")
    mapping = load_county_mapping(args.mapping)
    weekly = weekly_by_group(raw, mapping, pd.Timestamp(args.start), pd.Timestamp(args.end))

    for group, wk in weekly.groupby(level="group"):
        wk = wk.droplevel("group")
        daily = to_daily(wk)
        root = output_root(group, mapping is not None, args.out_root)
        for sub in ("hospitalizations_ts", "hospitalizations_ts_cumul", "hospitalizations_end_cumul"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        out = pd.DataFrame({"Date": fmt_date(daily["date"])})
        out = pd.concat([out, daily[AGE_KEYS].reset_index(drop=True)], axis=1)
        out_path = root / "hospitalizations_ts" / "MA_flu_daily_hospitalizations.csv"
        out.to_csv(out_path, index=False)
        print(f"[{group}] {len(wk)} weeks ({wk.index.min().date()} .. {wk.index.max().date()}), "
              f"{int(wk.to_numpy().sum())} admissions -> {out_path}")

        split_hospitalizations_by_age.main(root)
        compute_cumulative_hospitalizations.main(root)


if __name__ == "__main__":
    main()
