"""Shared MA_vax constants and table utilities.

Self-contained extraction of the handful of things this folder used to import
from the standalone prototype (`MA_vax/model.py` and `MA_vax/counterfactual.py`,
now `MA_vax_standalone/`), so that nothing here depends on that folder any more.

Contents (verbatim from the prototype unless noted):

  Constants, from `model.py`
    DATA_FOLDER, AGE_GROUP_LABELS, COMPARTMENTS, TRANSITIONS

  Summary/table helpers, from `counterfactual.py`
    _summ, averted_summary, attack_probability_curves, _rate_ratio_col,
    _matched_cohort_ratio_col, DICT_TABLES, load_saved_tables

Only pure post-processing helpers were brought across -- everything here takes
plain numpy arrays or reads CSVs, so none of it touches the prototype's
hand-written simulation equations. The scenario builders, the `table_S_A_*`
computations and `save_all_tables` were deliberately left behind: this folder's
`counterfactual_generic.py` reimplements those against the generic_core model.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# ── Constants (from MA_vax/model.py) ─────────────────────────────────────────

# Location of the input CSVs / config. The prototype hard-coded this
# repo-root-relative (`"generic_core/examples/massachusetts_vax"`), which only
# resolved when the process ran from the repo root; resolving it against this
# file's location instead makes it valid from any working directory. That
# sibling folder has since been consolidated into this one, so the inputs now
# live in `./data/` -- DATA_FOLDER is this directory, and every call site still
# joins "data" onto it exactly as before.
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Age groups, in the order used by every age-indexed array (population,
# E0_counts, I_to_H_prop, vax_arr columns, ...).
AGE_GROUP_LABELS = ["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"]

COMPARTMENTS = ["S", "E", "I", "R", "SV", "EV", "IV", "H", "D"]
TRANSITIONS = [
    "S_to_E", "S_to_SV", "E_to_I", "I_to_H", "I_to_R",
    "SV_to_EV", "EV_to_IV", "IV_to_H", "IV_to_R", "H_to_D", "H_to_R",
]

AGE_GROUPS = AGE_GROUP_LABELS


# ── Summary helpers (from MA_vax/counterfactual.py) ──────────────────────────

def _summ(values: np.ndarray) -> tuple[float, float, float]:
    """median, 2.5th pct, 97.5th pct across replications."""
    return (
        float(np.median(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def averted_summary(reference: dict, scenario: dict, by_age: bool = True,
                     pct_reference: dict | None = None,
                     doses_reference: dict | None = None,
                     doses_override: np.ndarray | float | None = None) -> pd.DataFrame:
    """
    Hospitalizations averted by `scenario` relative to `reference` (both
    dicts from `scenario_totals`, run with the same n_reps/seed so
    replications are paired). Returns a DataFrame indexed by age group (plus
    an aggregate "All" row) with median [95% CI] strings for percent averted,
    hospitalizations averted per 100K population, and (when doses differ)
    per 100K additional doses -- mirroring Tables S.A.1-S.A.3/S.A.5-S.A.6.

    `pct_reference` overrides the denominator used for `pct_averted` only
    (per100k/per100k_doses always use `reference`/`scenario`'s own
    population/doses). Needed when chaining multiple `averted_summary` calls
    that must decompose additively in percentage terms, e.g. Table S.A.1's
    reduced-infection + reduced-severity = total split -- the raw averted
    *counts* always sum correctly across such a chain, but the percentages
    only sum to the chain's overall percentage if every link's `pct_averted`
    shares the same denominator (the chain's starting `new_H`), not each
    link's own intermediate `reference`.

    `doses_reference` overrides which scenario's doses are subtracted to get
    the per-100K-doses denominator (default: `reference`'s). Needed when the
    two scenarios being differenced share an identical schedule, so their
    dose difference is exactly zero -- e.g. Table S.A.1's reduced-severity
    link (infection-protection-only -> full baseline), where the meaningful
    denominator is the full baseline dose count, i.e. `doses_reference=no_vax`.

    `doses_override` replaces the denominator for EVERY row (age rows and
    "All" alike) with one scalar or per-replicate dose count, instead of each
    row using the doses delivered to its own age group. Needed whenever the
    numerator and denominator refer to different age groups -- e.g. Table
    S.A.2/S.A.3's off-diagonal cells, which count hospitalizations averted in
    the ROW's age group per dose delivered to the COLUMN's age group. Without
    it those cells divide by zero doses (S.A.2) or by an incidental handful of
    dose-cap spillover doses (S.A.3) and are dropped or meaningless.
    """
    ref_H, scen_H = reference["new_H"], scenario["new_H"]   # (n_reps, A)
    averted = ref_H - scen_H
    pop = reference["population"]
    doses_ref = reference if doses_reference is None else doses_reference
    doses = scenario["doses"] - doses_ref["doses"]           # (A,) or (n_reps, A) additional doses given
    pct_denom_H = reference["new_H"] if pct_reference is None else pct_reference["new_H"]

    def row(label, averted_col, ref_col, pop_val, doses_val):
        pct = np.where(ref_col > 0, averted_col / np.where(ref_col > 0, ref_col, 1) * 100, np.nan)
        per100k = averted_col / pop_val * 1e5
        # doses_val is either a scalar (one schedule shared by every replicate)
        # or a per-replicate array (paired with averted_col) -- broadcasting
        # handles both the same way.
        doses_arr = np.broadcast_to(np.asarray(doses_val, dtype=float), np.shape(averted_col))
        if np.any(doses_arr > 0):
            per100k_doses = averted_col / np.where(doses_arr > 0, doses_arr, 1) * 1e5
            dm, dl, dh = _summ(per100k_doses)
            doses_str = f"{dm:.1f} [{dl:.1f} - {dh:.1f}]"
        else:
            doses_str = "—"
        pm, pl, ph = _summ(pct)
        rm, rl, rh = _summ(per100k)
        return {
            "age_group": label,
            "pct_averted": f"{pm:.1f}% [{pl:.1f}% - {ph:.1f}%]",
            "per100k_averted": f"{rm:.1f} [{rl:.1f} - {rh:.1f}]",
            "per100k_doses_averted": doses_str,
        }

    rows = []
    A = ref_H.shape[1]
    if by_age:
        for i in range(A):
            doses_val = (np.take(doses, i, axis=-1) if doses_override is None
                          else doses_override)
            rows.append(row(AGE_GROUPS[i], averted[:, i], pct_denom_H[:, i], pop[i], doses_val))
    all_doses = doses.sum(axis=-1) if doses_override is None else doses_override
    rows.append(row("All", averted.sum(axis=1), pct_denom_H.sum(axis=1), pop.sum(), all_doses))
    return pd.DataFrame(rows).set_index("age_group")


def _rate_ratio_col(vax_num: np.ndarray, vax_den: np.ndarray,
                     unvax_num: np.ndarray, unvax_den: np.ndarray) -> list[str]:
    """Per age group (columns of the (reps, A) arrays), median [95% CI] of
    the per-replication rate ratio (vax_num/vax_den) / (unvax_num/unvax_den),
    as a percentage string. `"—"` where undefined (either denominator zero)
    in every replication."""
    A = vax_num.shape[1]
    out = []
    for i in range(A):
        with np.errstate(divide="ignore", invalid="ignore"):
            vax_rate = np.where(vax_den[:, i] > 0, vax_num[:, i] / vax_den[:, i], np.nan)
            unvax_rate = np.where(unvax_den[:, i] > 0, unvax_num[:, i] / unvax_den[:, i], np.nan)
            ratio = vax_rate / unvax_rate
        if np.all(np.isnan(ratio)):
            out.append("—")
            continue
        m = float(np.nanmedian(ratio)) * 100
        lo = float(np.nanpercentile(ratio, 2.5)) * 100
        hi = float(np.nanpercentile(ratio, 97.5)) * 100
        out.append(f"{m:.1f}% [{lo:.1f}% - {hi:.1f}%]")
    return out


def attack_probability_curves(S: np.ndarray, SV: np.ndarray, S_to_E: np.ndarray,
                               SV_to_EV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given (reps, day, A) compartment/flow arrays, return `(attack_S_from,
    attack_SV_from)`, each (reps, day, A): `attack_S_from[r, d, a]` /
    `attack_SV_from[r, d, a]` is the counterfactual season-end attack
    probability for a hypothetical individual entering the S/SV pool on day
    `d` (in replication `r`, age group `a`) and followed to season end --
    see `table_vax_efficacy_check`'s docstring (the
    "matched_cohort_infection_reduction" entry) for the full derivation.
    Used both by `_matched_cohort_ratio_col` (which reduces these into a
    single ratio) and directly by the counterfactual notebooks to plot the
    curves themselves.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        haz_S = np.where(S > 0, S_to_E / S, 0.0)
        haz_SV = np.where(SV > 0, SV_to_EV / SV, 0.0)
    haz_S = np.clip(haz_S, 0.0, 1.0)
    haz_SV = np.clip(haz_SV, 0.0, 1.0)

    # Suffix log-survival along the day axis: logsurv[r, d, a] = sum_{t=d}^{T-1} log(1 - haz[r, t, a]).
    # A reversed cumsum of the reversed log-hazard array gives this in one vectorized pass.
    def suffix_log_survival(haz: np.ndarray) -> np.ndarray:
        log1m = np.log1p(-haz)
        return np.cumsum(log1m[:, ::-1, :], axis=1)[:, ::-1, :]

    attack_S_from = 1.0 - np.exp(suffix_log_survival(haz_S))    # (reps, day, A)
    attack_SV_from = 1.0 - np.exp(suffix_log_survival(haz_SV))  # (reps, day, A)
    return attack_S_from, attack_SV_from


def _matched_cohort_ratio_col(S: np.ndarray, SV: np.ndarray, S_to_E: np.ndarray,
                               SV_to_EV: np.ndarray, S_to_SV: np.ndarray) -> list[str]:
    """
    Per age group (last axis of the (reps, day, A) arrays), median [95% CI]
    of the per-replication matched-cohort attack-rate ratio -- see
    `table_vax_efficacy_check` docstring for the formula and rationale.
    `"—"` where no one was ever vaccinated in that (rep, age) slice.
    """
    attack_S_from, attack_SV_from = attack_probability_curves(S, SV, S_to_E, SV_to_EV)

    weight = S_to_SV  # (reps, day, A)
    total_weight = weight.sum(axis=1)  # (reps, A)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_attack_vax = np.where(
            total_weight > 0, (weight * attack_SV_from).sum(axis=1) / total_weight, np.nan)
        weighted_attack_unvax = np.where(
            total_weight > 0, (weight * attack_S_from).sum(axis=1) / total_weight, np.nan)
        ratio = weighted_attack_vax / weighted_attack_unvax  # (reps, A)

    A = ratio.shape[1]
    out = []
    for i in range(A):
        col = ratio[:, i]
        if np.all(np.isnan(col)):
            out.append("—")
            continue
        m = float(np.nanmedian(col)) * 100
        lo = float(np.nanpercentile(col, 2.5)) * 100
        hi = float(np.nanpercentile(col, 97.5)) * 100
        out.append(f"{m:.1f}% [{lo:.1f}% - {hi:.1f}%]")
    return out


# ── Saved-table I/O (from MA_vax/counterfactual.py) ──────────────────────────

DICT_TABLES = {
    "S_A_2": ["pct_reduction", "per_100k", "per_100k_doses"],
    "S_A_3": ["pct_reduction", "per_100k", "per_100k_doses"],
    "S_A_5": ["pct_reduction", "per_100k"],
    "S_A_6": ["pct_reduction", "per_100k"],
    "VAX_CHECK": ["infection_reduction", "matched_cohort_infection_reduction", "hospitalization_reduction"],
}


def load_saved_tables(results_folder: str) -> dict:
    """
    Read back everything `save_all_tables` wrote, in the same shapes the
    `table_S_A_*` functions return (a DataFrame, or a dict of DataFrame for
    the multi-metric tables), plus a `"meta"` key with the run metadata.
    Returns `None` for any table whose CSV isn't present in `results_folder`.
    """
    def read(filename, index_col: int | list[int] = 0):
        path = os.path.join(results_folder, filename)
        return pd.read_csv(path, index_col=index_col) if os.path.exists(path) else None

    meta_path = os.path.join(results_folder, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    tables = {"meta": meta, "S_A_1": read("S_A_1.csv"), "S_A_4": read("S_A_4.csv", index_col=[0, 1])}
    for name, subs in DICT_TABLES.items():
        tables[name] = {sub: read(f"{name}_{sub}.csv") for sub in subs}
    return tables
