"""Turn each per-age-band daily hospitalizations CSV (produced by
split_hospitalizations_by_age.py) into a cumulative-hospitalizations CSV, for
use as a Fitting-tab `ts` target against a cumulative compartment/transition
variable in model_builder_notebook.py.

Output filenames keep the same age-band token as their source
(e.g. `_13_17`, `_65plus`) so the Fitting tab's bulk-CSV-upload age-band
auto-detection (see _nb_fitting.py's _match_age_idx) still picks up the
correct age group without manual dropdown selection.

Also writes MA_flu_end_of_season_hospitalizations_by_age.csv, a single
`age,value` file with each age group's final (end-of-season) cumulative
total — usable as a `scalar` mode Fitting-tab target, with `age` giving the
row's age index directly (see _nb_fitting.py's Scalar format reference).

Run:
    uv run python generic_core/examples/MA_vax/compute_cumulative_hospitalizations.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# Age-band suffixes produced by split_hospitalizations_by_age.py, in
# model_config_MA.json's age_risk.age_groups order (their list index is the
# `age` value used in the end-of-season summary file).
AGE_SUFFIXES = ["0_0", "1_4", "5_12", "13_17", "18_49", "50_64", "65plus"]


def main() -> None:
    end_of_season = []

    for age_idx, suffix in enumerate(AGE_SUFFIXES):
        src_path = HERE / "data" / "hospitalizations_ts" / f"MA_flu_daily_hospitalizations_{suffix}.csv"
        df = pd.read_csv(src_path)
        df = df.sort_values("date")
        df["value"] = df["value"].cumsum()

        out_path = HERE / "data" / "hospitalizations_ts_cumul" / f"MA_flu_cumulative_hospitalizations_{suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

        end_of_season.append({"age": age_idx, "value": df["value"].iloc[-1]})

    summary_path = HERE / "data" / "hospitalizations_end_cumul" / "MA_flu_end_of_season_hospitalizations_by_age.csv"
    pd.DataFrame(end_of_season).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
