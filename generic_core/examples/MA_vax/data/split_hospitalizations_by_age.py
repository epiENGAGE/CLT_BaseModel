"""Split MA_flu_daily_hospitalizations.csv (one column per age group) into
one `date,value` CSV per age group, for use as separate Fitting-tab targets
in model_builder_notebook.py (each target's CSV format expects a single
`value` column; age is selected via that target's age-slice dropdown, not
inferred from CSV column names).

Run:
    uv run python generic_core/examples/MA_vax/split_hospitalizations_by_age.py

Output columns correspond, in order, to model_config_MA.json's
age_risk.age_groups (0-0, 1-4, 5-12, 13-17, 18-49, 50-64, 65+), so the
age index chosen in each target's dropdown should match its position here.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SOURCE_CSV = HERE / "data" / "hospitalizations_ts" / "MA_flu_daily_hospitalizations.csv"

# (source column name, output file suffix) — suffix is filesystem-safe.
AGE_COLUMNS = [
    ("0_0", "0_0"),
    ("1_4", "1_4"),
    ("5_12", "5_12"),
    ("13_17", "13_17"),
    ("18_49", "18_49"),
    ("50_64", "50_64"),
    ("65+", "65plus"),
]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)
    dates = pd.to_datetime(df["Date"], format="%m/%d/%y").dt.strftime("%Y-%m-%d")

    for column, suffix in AGE_COLUMNS:
        out_path = HERE / "data" / "hospitalizations_ts" / f"MA_flu_daily_hospitalizations_{suffix}.csv"
        pd.DataFrame({"date": dates, "value": df[column]}).to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
