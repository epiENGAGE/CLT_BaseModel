"""Regenerate the scenario blocks of run_simulations_MA_vax.py from the
current fit and vaccination schedule.

Rewrites four dict literals in place -- SCENARIOS, DESIGNED_PARAMS,
DESIGNED_PARAM_RATIOS and DOSE_MULTIPLIER -- so that after a new fit (or an
alternative model version) one command brings the scenario definitions back
in sync:

    python make_scenarios_MA_vax.py            # rewrite run_simulations_MA_vax.py
    python make_scenarios_MA_vax.py --dry-run  # print the multipliers only

    # a variant run: other script + fit, a subset of scenarios
    python make_scenarios_MA_vax.py --run-sim-file run_simulations_MA_vax_IHR_ub_5.py \
        --fitted-params fitted_params_MA_vax_IHR_ub_5.json \
        --only baseline "no vax" "50% coverage (all ages)"

Inputs (all in this folder):
  model_config_MA_vax.json   pre-fit params (baselines the fit's IHR_scale multiplies)
  fitted_params_MA_vax.json  best_params -> literal param values in SCENARIOS
  the vaccination schedule   daily_vaccines_df via the model config's input_files
                             (counterfactual_generic.scheduled_coverage)

Scenarios written:
  baseline, "children vax only", "Infection protection only", "no vax"
  "Vaccinate <age> only"                 dose multiplier 1 for that age, 0 elsewhere
  "Low VE", "High VE"                    vax_susceptibility / IV_to_H_prop scaled
                                         (counterfactual_generic.ve_scenarios multipliers)
  "<T>% coverage (all ages | <age> only)" for T in COVERAGE_TARGETS: multiplier
                                         max(1, T / scheduled baseline coverage), applied
                                         to every age group or to one (others stay at 1)
  "<D>% lower coverage (all ages | <age> only)" for D in COVERAGE_DECREASES:
                                         multiplier 1 - D/100 (relative decrease of the
                                         baseline schedule), every age group or one

Literal parameter values are the fitted best set. In "parameters" mode
run_simulations_MA_vax.py replaces them with each posterior draw, except
DESIGNED_PARAMS, which are scaled by DESIGNED_PARAM_RATIOS -- so the literals
mainly document the best fit.

The scheduled-coverage window is NUM_DAYS days from START_DATE, both read from
run_simulations_MA_vax.py so the two never drift apart.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent.parent))
sys.path.insert(0, str(_HERE))

import counterfactual_generic as cg
from generic_core.fitting import prepare_param_sets
from ma_vax_shared import AGE_GROUP_LABELS as AGE_GROUPS

RUN_SIM_FILE = _HERE / "run_simulations_MA_vax.py"
MODEL_CONFIG_FILE = _HERE / "model_config_MA_vax.json"
FITTED_PARAMS_FILE = _HERE / "fitted_params_MA_vax.json"

COVERAGE_TARGETS = [50, 55, 60, 65, 70]   # % cumulative coverage
COVERAGE_DECREASES = [5, 10, 15, 20]      # % relative decrease of the baseline schedule

# Params written into each SCENARIOS literal (the scalar/age-array model
# params; contact matrices and vax_transfer_delay_days are left to the config).
LITERAL_PARAMS = [
    "beta_baseline", "relative_suscept", "humidity_impact", "E_to_I_rate", "I_out_rate",
    "EV_to_IV_rate", "H_out_rate", "I_relative_infectiousness", "IV_relative_infectiousness",
    "I_to_H_prop", "vax_susceptibility", "IV_to_H_prop", "H_to_D_prop",
]

VE_PRESETS = {  # scenario name -> counterfactual_generic.ve_scenarios key
    "Low VE": "low_ve",
    "High VE": "high_ve",
}


def _sim_setting(src: str, name: str) -> str:
    m = re.search(rf"^{name} = (.+)$", src, re.M)
    if not m:
        raise SystemExit(f"{RUN_SIM_FILE.name}: no top-level `{name} = ...` line")
    return m.group(1).split("#")[0].strip()


def fitted_params() -> dict:
    """Pre-fit config params with the fit's best set applied (IHR_scale expanded)."""
    cfg = json.loads(MODEL_CONFIG_FILE.read_text())
    fit = json.loads(FITTED_PARAMS_FILE.read_text())
    best = prepare_param_sets([fit["best_params"]], fit.get("scale_groups") or {}, cfg["params"])[0]
    model = {k: v for k, v in best.items()
             if k != "phi" and not k.startswith(("m_dlog_", "seed_scale_"))}
    return {**cfg["params"], **model}


def _ratio_arrays(scale) -> list:
    return [[float(x)] for x in scale]


def build(num_days: int, start_date: str):
    cg.NUM_DAYS, cg.START_DATE = num_days, start_date
    cov = np.asarray(cg.scheduled_coverage(), dtype=float)
    params = fitted_params()
    lit = {k: params[k] for k in LITERAL_PARAMS}
    n = len(AGE_GROUPS)

    # VE presets: the multipliers counterfactual_generic.ve_scenarios applies.
    ve = cg.ve_scenarios({"config_dict": {"params": params}})
    p_vs = np.asarray(params["vax_susceptibility"], dtype=float)
    p_ivh = np.asarray(params["IV_to_H_prop"], dtype=float)
    ve_ratios = {}
    for name, key in VE_PRESETS.items():
        o = ve[key]["param_overrides"]
        ve_ratios[name] = {
            "vax_susceptibility": _ratio_arrays(np.asarray(o["vax_susceptibility"]).ravel() / p_vs.ravel()),
            "IV_to_H_prop": _ratio_arrays(np.asarray(o["IV_to_H_prop"]).ravel() / p_ivh.ravel()),
        }
    # Infection protection only: IV_to_H_prop := I_to_H_prop, i.e. ratio I/IV.
    ip_ratio = _ratio_arrays(np.asarray(params["I_to_H_prop"], dtype=float).ravel() / p_ivh.ravel())

    scenarios, designed, ratios, dose = {}, {}, {}, {}

    def add(name, dose_mult=None, designed_params=(), ratio=None):
        p = {k: (list(v) if isinstance(v, list) else v) for k, v in lit.items()}
        for param, r in (ratio or {}).items():
            p[param] = (np.asarray(params[param], dtype=float) * np.asarray(r)).tolist()
        scenarios[name] = p
        designed[name] = list(designed_params)
        if ratio:
            ratios[name] = ratio
        if dose_mult is not None:
            dose[name] = [float(x) for x in dose_mult]

    scenarios["baseline"] = {}
    add("children vax only", [1.0] * 4 + [0.0] * (n - 4))
    add("Infection protection only", None, ["IV_to_H_prop"], {"IV_to_H_prop": ip_ratio})
    add("no vax", [0.0] * n)
    for i, label in enumerate(AGE_GROUPS):
        add(f"Vaccinate {label} only", [1.0 if j == i else 0.0 for j in range(n)])
    for name in VE_PRESETS:
        add(name, None, ["vax_susceptibility", "IV_to_H_prop"], ve_ratios[name])

    cov_mult = {}
    for t in COVERAGE_TARGETS:
        mult = np.maximum(1.0, (t / 100.0) / cov)
        cov_mult[t] = mult
        add(f"{t}% coverage (all ages)", mult)
        for i, label in enumerate(AGE_GROUPS):
            add(f"{t}% coverage ({label} only)", [mult[j] if j == i else 1.0 for j in range(n)])
    for d in COVERAGE_DECREASES:
        f = 1.0 - d / 100.0
        add(f"{d}% lower coverage (all ages)", [f] * n)
        for i, label in enumerate(AGE_GROUPS):
            add(f"{d}% lower coverage ({label} only)", [f if j == i else 1.0 for j in range(n)])
    return scenarios, designed, ratios, dose, cov, cov_mult


def _fmt_dict(d: dict, indent: str = "    ") -> str:
    return "".join(f"{indent}{json.dumps(k)}: {v!r},\n" for k, v in d.items())


def render(scenarios, designed, ratios, dose, cov, num_days, start_date) -> dict[str, str]:
    cov_txt = ", ".join(f"{a}={c:.2%}" for a, c in zip(AGE_GROUPS, cov))
    header = (
        "    # GENERATED by make_scenarios_MA_vax.py -- edit that script (targets,\n"
        "    # decreases, VE presets) and re-run it rather than editing here.\n"
        f"    # Literal values = fitted best set ({FITTED_PARAMS_FILE.name}); in\n"
        "    # 'parameters' mode each posterior draw replaces them, except\n"
        "    # DESIGNED_PARAMS (scaled by DESIGNED_PARAM_RATIOS).\n"
    )
    dose_header = (
        "    # GENERATED by make_scenarios_MA_vax.py.\n"
        "    # '<T>% coverage': max(1, T / scheduled baseline coverage), so groups\n"
        "    # already above T keep their schedule; untargeted groups stay at 1.0.\n"
        f"    # Scheduled baseline coverage ({num_days} days from {start_date}):\n"
        f"    #   {cov_txt}\n"
        "    # '<D>% lower coverage': 1 - D/100, a relative cut of the baseline schedule.\n"
    )
    return {
        "SCENARIOS": "SCENARIOS = {\n" + header + _fmt_dict(scenarios) + "}\n",
        "DESIGNED_PARAMS": "DESIGNED_PARAMS = {\n    # GENERATED by make_scenarios_MA_vax.py.\n"
                           + _fmt_dict(designed) + "}\n",
        "DESIGNED_PARAM_RATIOS": "DESIGNED_PARAM_RATIOS = {\n    # GENERATED by make_scenarios_MA_vax.py.\n"
                                 + _fmt_dict(ratios) + "}\n",
        "DOSE_MULTIPLIER": "DOSE_MULTIPLIER = {\n" + dose_header + _fmt_dict(dose) + "}\n",
    }


def replace_block(src: str, name: str, new: str) -> str:
    start = re.search(rf"^{name} = \{{\n", src, re.M)
    if not start:
        raise SystemExit(f"{RUN_SIM_FILE.name}: no `{name} = {{` block")
    end = src.index("\n}\n", start.start()) + 3
    return src[:start.start()] + new + src[end:]


def main() -> None:
    global RUN_SIM_FILE, FITTED_PARAMS_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the multipliers, don't rewrite")
    ap.add_argument("--run-sim-file", default=RUN_SIM_FILE.name,
                    help="simulation script to rewrite (default: %(default)s)")
    ap.add_argument("--fitted-params", default=FITTED_PARAMS_FILE.name,
                    help="fit whose best set fills the literals (default: %(default)s)")
    ap.add_argument("--only", nargs="+", metavar="SCENARIO",
                    help="keep only these scenarios (default: all)")
    args = ap.parse_args()
    RUN_SIM_FILE = _HERE / args.run_sim_file
    FITTED_PARAMS_FILE = _HERE / args.fitted_params

    src = RUN_SIM_FILE.read_text()
    num_days = int(_sim_setting(src, "NUM_DAYS"))
    start_date = json.loads(_sim_setting(src, "START_DATE").replace("'", '"'))
    scenarios, designed, ratios, dose, cov, cov_mult = build(num_days, start_date)
    if args.only:
        missing = [n for n in args.only if n not in scenarios]
        if missing:
            raise SystemExit(f"--only: unknown scenario(s) {missing}")
        keep = set(args.only)
        scenarios = {k: v for k, v in scenarios.items() if k in keep}
        designed = {k: v for k, v in designed.items() if k in keep}
        ratios = {k: v for k, v in ratios.items() if k in keep}
        dose = {k: v for k, v in dose.items() if k in keep}

    print(f"Scheduled baseline coverage ({num_days} days from {start_date}):")
    print("  " + ", ".join(f"{a}={c:.2%}" for a, c in zip(AGE_GROUPS, cov)))
    for t, m in cov_mult.items():
        print(f"  {t}% target multipliers: " + ", ".join(f"{a}={x:.3f}" for a, x in zip(AGE_GROUPS, m)))
    print(f"{len(scenarios)} scenarios")
    if args.dry_run:
        return

    for name, block in render(scenarios, designed, ratios, dose, cov, num_days, start_date).items():
        src = replace_block(src, name, block)
    RUN_SIM_FILE.write_text(src)
    print(f"rewrote SCENARIOS, DESIGNED_PARAMS, DESIGNED_PARAM_RATIOS, DOSE_MULTIPLIER in {RUN_SIM_FILE.name}")


if __name__ == "__main__":
    main()
