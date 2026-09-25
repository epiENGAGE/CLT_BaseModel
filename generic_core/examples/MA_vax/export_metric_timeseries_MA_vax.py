"""Write a `*_metric_timeseries.csv` (the format plot_fit_vs_actual.py reads)
straight from a run_simulations_MA_vax*.py results_parquet directory, instead
of going through the Model Builder notebook's Analysis tab.

Same computation as the Analysis tab's metric plot export (_nb_analysis.py):
for each age group and "all ages", the per-replicate daily new_H
(I_to_H + IV_to_H) and its per-replicate cumulative sum, summarised across
replicates as median / 2.5th / 97.5th percentile.

Usage:
    python generic_core/examples/MA_vax/export_metric_timeseries_MA_vax.py \
        --db <simulation_output_dir>/results_parquet --out <file>.csv [--scenario baseline]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generic_core import results_io

METRIC = "new_H"
METRIC_TVS = ("I_to_H", "IV_to_H")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True, help="results_parquet directory (or results.db)")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--start-date", default="2025-09-01")
    args = ap.parse_args()

    con = results_io.load_source(str(args.db))
    df = con.execute(
        "SELECT rep, age_group, day, SUM(value) AS value FROM results_full "
        f"WHERE scenario = ? AND compartment IN ({','.join('?' * len(METRIC_TVS))}) "
        "GROUP BY rep, age_group, day",
        [args.scenario, *METRIC_TVS],
    ).df()
    con.close()
    if df.empty:
        raise SystemExit(f"no {METRIC_TVS} rows for scenario {args.scenario!r} in {args.db}")

    # (reps, days, ages)
    cube = df.pivot_table(index=["rep", "day"], columns="age_group", values="value", aggfunc="sum")
    reps = cube.index.get_level_values("rep").unique()
    days = cube.index.get_level_values("day").unique()
    arr = cube.to_numpy().reshape(len(reps), len(days), cube.shape[1])
    dates = pd.date_range(args.start_date, periods=len(days), freq="D")

    groups = {f"Age {a}": arr[:, :, i] for i, a in enumerate(cube.columns)}
    groups["all ages"] = arr.sum(axis=2)

    rows = []
    for series in ("daily", "cumulative"):
        for label, x in groups.items():
            x = np.cumsum(x, axis=1) if series == "cumulative" else x
            med = np.median(x, axis=0)
            lo, hi = np.percentile(x, [2.5, 97.5], axis=0)
            rows.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"), "series": series, "subpopulation": "all subpops",
                "age_group": label, "metric": METRIC, "scenario": args.scenario,
                "median": med, "ci_lower_2.5": lo, "ci_upper_97.5": hi,
            }))
    out = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(reps)} replicates, {len(days)} days)")


if __name__ == "__main__":
    main()
