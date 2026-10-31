"""Turn the weekly county x age vaccination coverage file into the model's
daily vaccination schedule, plus the population by age group.

Input: CSV with columns `COUNTY,AGE_GROUP,WEEK_END,DISTINCT_PT_FIRST_SHOT,POP_ESTIMATE`
(other columns are ignored). DISTINCT_PT_FIRST_SHOT is the CUMULATIVE number
of people with a first dose since the season start, per county and age
group; POP_ESTIMATE is that county x age group's population.

For each county group (the whole state by default, or the groups of
--mapping) it:
  1. sums cumulative doses and population over the group's counties;
  2. forces the cumulative series to be non-decreasing (running max -- a
     drop, e.g. a data correction, is reported and contributes 0 doses);
  3. takes weekly increments (the first week's increment is its cumulative
     value) and spreads each evenly over the 7 days ending on WEEK_END;
  4. divides by the group's population, giving the daily share of each age
     group vaccinated, and writes:
       schedules/MA_flu_daily_vaccinations_proportions_array.csv
           `date,daily_vaccines`, values like "[[0.0],[1.2e-05],...]" (the
           format model_config_MA_vax.json's vaccines_csv expects)
       vaccination/MA_flu_daily_vaccinations_proportions.csv
           same numbers, one column per age group (for checking)
       massachusetts_population.csv
           `age,population`, age = model age index 0-6

The model only reads schedule rows from its start date on (after shifting
them by vax_transfer_delay_days), so doses dated before START_DATE are not
applied.

Without --mapping, outputs go to this data/ folder (where the pipeline reads
them). With --mapping, each group goes to data/by_group/<group>/.

--update-model-config writes the (state-level) population into
model_config_MA_vax.json: initial_conditions.aggregate_pop.population,
total_population, and seeds.E rescaled to keep each age group's current
seeds-per-person ratio.

Run (from the repo root):
    python generic_core/examples/MA_vax/data/prepare_vaccination.py
    python generic_core/examples/MA_vax/data/prepare_vaccination.py --update-model-config
    python generic_core/examples/MA_vax/data/prepare_vaccination.py \\
        --mapping generic_core/examples/MA_vax/data/county_groups_example.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prep_common import (
    AGE_KEYS, HERE, assign_groups, fmt_date, load_county_mapping,
    normalize_age_label, output_root, parse_dates, week_days,
)

DEFAULT_INPUT = HERE / "original" / "MA_influenza_coverage_2025-2026.csv"
MODEL_CONFIG = HERE.parent / "model_config_MA_vax.json"
COUNTY_COL, AGE_COL, WEEK_COL, CUM_COL, POP_COL = (
    "COUNTY", "AGE_GROUP", "WEEK_END", "DISTINCT_PT_FIRST_SHOT", "POP_ESTIMATE")


def load(path: Path, mapping) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {COUNTY_COL, AGE_COL, WEEK_COL, CUM_COL, POP_COL} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    df["week_end"] = parse_dates(df[WEEK_COL])
    df["age"] = df[AGE_COL].map(normalize_age_label)
    df["group"] = assign_groups(df[COUNTY_COL], mapping)

    dup = df.duplicated([COUNTY_COL, "age", "week_end"])
    if dup.any():
        raise SystemExit(f"{path}: {int(dup.sum())} duplicate county x age x week rows")
    pop_varies = df.groupby([COUNTY_COL, "age"])[POP_COL].nunique() > 1
    if pop_varies.any():
        print(f"note: {POP_COL} changes over the season for {int(pop_varies.sum())} county x age "
              f"series; using each one's last value")
    return df


def county_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative doses, index week_end, columns (county, age); a county x age
    week with no row carries the previous week's value (0 before its first row)."""
    weeks = pd.date_range(df["week_end"].min(), df["week_end"].max(), freq="7D")
    if not set(df["week_end"]) <= set(weeks):
        raise SystemExit(f"{WEEK_COL} values are not all on the same weekday")
    wide = df.pivot(index="week_end", columns=[COUNTY_COL, "age"], values=CUM_COL).reindex(weeks)
    n_missing = int(wide.isna().sum().sum())
    if n_missing:
        print(f"note: {n_missing} missing county x age x week values, filled with the previous week's cumulative")
    return wide.ffill().fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="weekly coverage CSV")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="CSV with columns county,group; default: aggregate the whole state")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write under this folder instead of data/ (same sub-structure)")
    ap.add_argument("--update-model-config", nargs="?", const=MODEL_CONFIG, default=None, type=Path,
                    metavar="PATH", help=f"write the population into this model config "
                                         f"(default when given without PATH: {MODEL_CONFIG.name}); "
                                         f"only without --mapping")
    args = ap.parse_args()

    mapping = load_county_mapping(args.mapping)
    if args.update_model_config is not None and mapping is not None:
        raise SystemExit("--update-model-config only applies to the state-level (no --mapping) run")

    df = load(args.input, mapping)
    cum_county = county_cumulative(df)
    pop_county = df.sort_values("week_end").groupby([COUNTY_COL, "age"])[POP_COL].last()
    county_group = df.drop_duplicates(COUNTY_COL).set_index(COUNTY_COL)["group"]

    for group in sorted(county_group.unique()):
        counties = county_group.index[county_group == group]
        cum = (cum_county.loc[:, cum_county.columns.get_level_values(0).isin(counties)]
                         .T.groupby(level="age").sum().T.reindex(columns=AGE_KEYS, fill_value=0.0))
        pop = (pop_county[pop_county.index.get_level_values(0).isin(counties)]
                         .groupby(level="age").sum().reindex(AGE_KEYS))
        if pop.isna().any() or (pop <= 0).any():
            raise SystemExit(f"[{group}] no population for age groups {list(pop.index[pop.isna() | (pop <= 0)])}")

        drops = (cum.diff() < 0).sum()
        if drops.any():
            print(f"[{group}] cumulative doses decrease in {int(drops.sum())} week x age cells "
                  f"({', '.join(f'{a}: {int(n)}' for a, n in drops.items() if n)}); running max used")
        cum = cum.cummax()
        weekly = cum.diff()
        weekly.iloc[0] = cum.iloc[0]

        rows = []
        for week_end, inc in weekly.iterrows():
            share = (inc / pop / 7.0).to_numpy(dtype=float)
            rows.extend((d, share) for d in week_days(week_end))
        dates = [d for d, _ in rows]
        shares = np.array([s for _, s in rows])

        root = output_root(group, mapping is not None, args.out_root)
        for sub in ("schedules", "vaccination"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        arr = pd.DataFrame({
            "date": fmt_date(dates),
            "daily_vaccines": ["[" + ",".join(f"[{float(v)!r}]" for v in s) + "]" for s in shares],
        })
        arr_path = root / "schedules" / "MA_flu_daily_vaccinations_proportions_array.csv"
        arr.to_csv(arr_path, index=False)

        wide = pd.DataFrame(shares, columns=AGE_KEYS)
        wide.insert(0, "date", fmt_date(dates))
        wide.to_csv(root / "vaccination" / "MA_flu_daily_vaccinations_proportions.csv", index=False)

        pop_path = root / "massachusetts_population.csv"
        pd.DataFrame({"age": range(len(AGE_KEYS)), "population": pop.astype(int).to_numpy()}).to_csv(
            pop_path, index=False)

        final_cov = cum.iloc[-1] / pop
        print(f"[{group}] {len(weekly)} weeks ({weekly.index.min().date()} .. {weekly.index.max().date()}) "
              f"-> {arr_path}")
        print(f"[{group}] population {int(pop.sum())}: " + ", ".join(f"{a}={int(p)}" for a, p in pop.items()))
        print(f"[{group}] end-of-data coverage: " + ", ".join(f"{a}={c:.1%}" for a, c in final_cov.items()))

        if args.update_model_config is not None:
            update_model_config(args.update_model_config, pop.to_numpy(dtype=float))


def update_model_config(path: Path, pop: np.ndarray) -> None:
    with open(path) as f:
        cfg = json.load(f)
    agg = cfg["initial_conditions"]["aggregate_pop"]
    old_pop = np.asarray(agg["population"], dtype=float).reshape(-1)
    if old_pop.shape != pop.shape:
        raise SystemExit(f"{path}: population has {old_pop.size} age groups, data has {pop.size}")
    agg["population"] = [[float(p)] for p in pop]
    cfg["total_population"] = int(pop.sum())
    for comp, seeds in (agg.get("seeds") or {}).items():
        s = np.asarray(seeds, dtype=float).reshape(-1)
        agg["seeds"][comp] = [[float(round(v))] for v in s * pop / old_pop]
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"updated {path}: population, total_population, seeds ({', '.join(agg.get('seeds') or {})})")


if __name__ == "__main__":
    main()
