"""Run counterfactual vaccination scenarios against the fitted MA_vax model.

Loads a fitted run produced by `fit_bayesian.py` (posterior-mean overrides +
fitted m(t)), applies each scenario in `scenarios.py`, simulates either
deterministically (a single replication) or stochastically (many
chain-binomial replications), and writes one NetCDF file per scenario with
per-age-group daily compartment counts and transition flows
(`model.simulate_detailed`).

Run:
    uv run python MA_vax_standalone/run_scenarios.py --fit-folder MA_vax_standalone/outputs_2026-07-08
    uv run python MA_vax_standalone/run_scenarios.py --fit-folder MA_vax_standalone/outputs_2026-07-08 \
        --stochastic --n-reps 200
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from MA_vax_standalone import model
from MA_vax_standalone.fit_bayesian import load_fitted_run
from MA_vax_standalone.scenarios import SCENARIOS

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-folder", required=True,
                     help="figures_* output folder from fit_bayesian.py")
    ap.add_argument("--method", choices=["emcee", "pyabc"], default="emcee")
    ap.add_argument("--stochastic", action="store_true",
                     help="run n-reps chain-binomial replications instead of one deterministic run")
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-per-day", type=int, default=1,
                     help="sub-daily Euler/binomial steps per day (output stays daily)")
    ap.add_argument("--out", default=None, help="output folder [default: <fit-folder>/scenarios]")
    args = ap.parse_args()

    overrides, beta_multiplier_arr = load_fitted_run(args.fit_folder, args.method)

    params = model.default_params()
    base_inputs = model.load_inputs(model.DATA_FOLDER, params)
    base_inputs["params"] = model.apply_overrides(base_inputs["params"], overrides)
    if beta_multiplier_arr is not None:
        base_inputs["beta_multiplier_arr"] = beta_multiplier_arr

    now = datetime.now()
    out_folder = args.out or os.path.join(args.fit_folder, "scenarios_" + now.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    os.makedirs(out_folder, exist_ok=True)

    n_reps = args.n_reps if args.stochastic else 1
    print(f"Fitted run: {args.fit_folder} [{args.method}]  "
          f"stochastic={args.stochastic}  n_reps={n_reps}")

    for name, scenario in SCENARIOS.items():
        inputs = model.apply_scenario(base_inputs, scenario)
        ds = model.simulate_detailed(inputs, stochastic=args.stochastic, n_reps=n_reps,
                                     seed=args.seed, steps_per_day=args.steps_per_day)

        path = os.path.join(out_folder, f"scenario_{name}.nc")
        ds.to_netcdf(path)

        totals = model.total_transitions(ds)
        new_H_total = (totals["I_to_H"] + totals["IV_to_H"])
        deaths_total = totals["H_to_D"]
        print(f"[{name}] saved {path}  |  total new_H mean={float(new_H_total.mean()):.1f} "
              f"(std={float(new_H_total.std()):.1f})  total deaths mean={float(deaths_total.mean()):.1f}")


if __name__ == "__main__":
    main()
