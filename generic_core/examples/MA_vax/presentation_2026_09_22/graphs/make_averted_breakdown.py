"""Hospitalizations averted, split into infection protection vs hospitalization
(severity) reduction, whole population, for the four MA_vax models.

Reads each model's Table S.A.1 ("All" row). Infection protection = no vax -
infection protection only; hospitalization reduction = infection protection
only - baseline; total = no vax - baseline. Percentages are of no-vax
hospitalizations.

Counts come from S.A.1's averted_* columns where present (7 ages only);
otherwise they are per100k_averted_* x population / 100k, which is exact for
the medians and interval bounds alike since the population is fixed (up to
the 0.1-per-100k rounding in the table, i.e. a few hospitalizations). Each cell is a separate median across
replicates, so the two components need not sum to the total.

Run with: python make_averted_breakdown.py
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MA_VAX = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT_FILE = os.path.join(HERE, "hospitalizations_averted_infection_vs_severity.csv")

MODELS = {
    "7 ages": (
        "counterfactual_tables_from_db_MA_vax/S_A_1.csv",
        "model_config_MA_vax.json"),
    "7 ages, no transmission multiplier": (
        "counterfactual_tables_from_db_MA_vax_no_transmission_multiplier/S_A_1.csv",
        "model_config_MA_vax.json"),
    "Single age": (
        "MA_vax_single_age/counterfactual_tables_from_db_MA_vax_single_age/S_A_1.csv",
        "MA_vax_single_age/model_config_MA_single_age.json"),
    "Single age, no transmission multiplier": (
        "MA_vax_single_age/counterfactual_tables_from_db_MA_vax_single_age_no_transmission_multiplier/S_A_1.csv",
        "MA_vax_single_age/model_config_MA_single_age.json"),
}
COMPONENTS = {
    "reduced_infection": "Infection protection",
    "reduced_severity": "Hospitalization reduction",
    "total": "Total",
}


def total_population(config_file):
    with open(config_file) as f:
        cfg = json.load(f)
    return float(np.sum(cfg["initial_conditions"]["aggregate_pop"]["population"]))


def numbers(cell):
    return [float(x) for x in re.findall(r"-?[\d.]+", str(cell).replace(",", ""))]


rows = []
for model, (table, config) in MODELS.items():
    pop = total_population(os.path.join(MA_VAX, config))
    all_row = pd.read_csv(os.path.join(MA_VAX, table), index_col=0).loc["All"]
    count_row = {"model": model, "measure": "count"}
    pct_row = {"model": model, "measure": "percent"}
    for key, label in COMPONENTS.items():
        if f"averted_{key}" in all_row:  # table already has counts
            count_row[label] = all_row[f"averted_{key}"]
        else:
            med, lo, hi = (v * pop / 1e5 for v in numbers(all_row[f"per100k_averted_{key}"]))
            count_row[label] = f"{med:,.0f} [{lo:,.0f} - {hi:,.0f}]"
        pct_row[label] = all_row[f"pct_averted_{key}"]
    rows += [count_row, pct_row]

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"wrote {OUT_FILE}")
