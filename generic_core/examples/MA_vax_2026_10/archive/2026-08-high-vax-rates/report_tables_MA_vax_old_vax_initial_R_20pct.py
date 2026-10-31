"""Markdown renderers for the counterfactual CSVs, used to assemble
report_MA_vax_old_vax_initial_R_20pct.md. Same table layouts as ../report.md.

    from report_tables_MA_vax_old_vax_initial_R_20pct import Tables
    t = Tables(<counterfactual_tables_dir>)
    t.sa1("reduced_infection")  -> markdown table
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

AGES = ["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"]
VE_COLS = {"low_ve": "Low VE", "baseline_ve": "Baseline VE (fitted)", "high_ve": "High VE"}


def _dash(cell: object) -> str:
    """'40 [24 - 52]' -> '40 [24 – 52]', and '-0' -> '0'."""
    s = str(cell).replace(" - ", " – ")
    return re.sub(r"(?<![\d.])-0(\.0+)?(?![\d.])", lambda m: "0" + (m.group(1) or ""), s)


def _thousands(cell: object) -> str:
    """Add thousands separators to every integer in an absolute-count cell."""
    return re.sub(r"(?<![\d.,])(\d{4,})(?![\d.,])", lambda m: f"{int(m.group(1)):,}", str(cell))


def md_table(header: list[str], rows: list[list[str]], bold_last: bool = True,
             align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for i, r in enumerate(rows):
        if bold_last and i == len(rows) - 1:
            r = [f"**{c}**" for c in r]
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


class Tables:
    def __init__(self, folder: Path):
        self.d = Path(folder)

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.d / f"{name}.csv", index_col=0, dtype=str, keep_default_na=False)

    # Table S.A.1: one block per channel
    def sa1(self, channel: str) -> str:
        df = self.csv("S_A_1")
        rows = []
        for age, r in df.iterrows():
            rows.append([str(age), _thousands(_dash(r[f"averted_{channel}"])),
                         _dash(r[f"pct_averted_{channel}"]), _dash(r[f"per100k_averted_{channel}"]),
                         _dash(r[f"per100k_doses_averted_{channel}"])])
        return md_table(["Age group", "Hospitalizations Averted", "% Hospitalizations Averted",
                         "Averted per 100,000 Population", "Averted per 100,000 Doses"], rows)

    # Tables S.A.2 / S.A.3: age (counted) x age (vaccinated)
    def age_matrix(self, name: str, absolute: bool = False, zero_as_dash: bool = False) -> str:
        df = self.csv(name)
        header = ["Age group (counted)"] + [f"{c} vaccinated" for c in df.columns]
        rows = []
        for age, r in df.iterrows():
            cells = []
            for c in df.columns:
                v = _dash(r[c])
                if absolute:
                    v = _thousands(v)
                # a zero-dose denominator (group already above 70%) divides to inf/nan
                if zero_as_dash and (v == "" or "inf" in v or "nan" in v):
                    v = "—"
                cells.append(v)
            rows.append([str(age)] + cells)
        return md_table(header, rows)

    # Tables S.A.5 / S.A.6: age x VE scenario
    def ve_table(self, name: str, absolute: bool = False) -> str:
        df = self.csv(name)
        rows = [[str(age)] + [(_thousands(_dash(r[c])) if absolute else _dash(r[c])) for c in VE_COLS]
                for age, r in df.iterrows()]
        return md_table(["Age group"] + list(VE_COLS.values()), rows)

    def sa4(self) -> str:
        df = pd.read_csv(self.d / "S_A_4.csv", dtype=str)
        rows = [[VE_COLS[r["scenario"]], r["age_group"], r["VE_infection"], r["VE_hosp_infection"],
                 r["VE_hosp_given_infection"]] for r in df.to_dict("records")]
        return md_table(["Scenario", "Age group", "VE against infection",
                         "VE against hospitalization (overall)",
                         "VE against hospitalization, given infection"], rows, bold_last=False)

    def dose_accounting(self) -> str:
        df = self.csv("DOSE_ACCOUNTING")
        rows = [[str(age), r.population, r.scheduled_doses, r.delivered_doses, r.wasted_doses,
                 r.pct_wasted, r.scheduled_coverage, r.delivered_coverage] for age, r in df.iterrows()]
        return md_table(["Age group", "Population", "Scheduled doses", "Delivered doses (median)",
                         "Scheduled but not delivered", "% not delivered", "Scheduled coverage",
                         "Delivered coverage"], rows, align=["---"] + ["---:"] * 7)

    def vax_check(self, name: str) -> str:
        df = self.csv(f"VAX_CHECK_{name}")
        header = ["Age group"] + [f"{c} vaccinated" for c in df.columns]
        rows = [[str(age)] + [_dash(r[c]) for c in df.columns] for age, r in df.iterrows()]
        return md_table(header, rows, bold_last=False)

    def cell(self, name: str, row: str, col: str) -> str:
        return _dash(self.csv(name).loc[row, col])
