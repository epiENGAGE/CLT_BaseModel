"""Row-normalize Table S.A.2's percent-reduction medians into the shares that
plot_pct_reduction_stacked_bar.R plots.

Reads S_A_2_pct_reduction.csv (cells are "<median>% [<lo>% - <hi>%]"), keeps the
median only, drops the "All" column (a row total, not a stack segment), and
divides each row by its own sum so the segments of one bar sum to 1.

Run with: python make_pct_reduction_normalized.py
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(HERE, "S_A_2_pct_reduction.csv")
OUT_FILE = os.path.join(HERE, "S_A_2_pct_reduction_normalized.csv")


def median(cell):
    m = re.match(r"^-?[0-9.]+", str(cell).strip())
    return float(m.group()) if m else 0.0


df = pd.read_csv(IN_FILE, index_col="age_group").drop(columns=["All"])
df = df.map(median)
df = df.div(df.sum(axis=1), axis=0)
df.round(6).to_csv(OUT_FILE)
print(f"wrote {OUT_FILE}")
