"""Compute the counterfactual vaccination-impact tables (S.A.1-S.A.6 analogs,
see `counterfactual_generic.py`) for the generic_core-exported MA_vax model,
and save them to CSV.

Running many stochastic replications (needed for tight confidence intervals)
can take a while, so this is split from display: run this script once, then
point `counterfactual_notebook_generic.py` at its output folder to load and
render the results instantly, without re-simulating.

Run (from the repo root):
    python generic_core/examples/run_sim_test_MA_vax/run_counterfactual_tables_generic.py --n-reps 1000
    python generic_core/examples/run_sim_test_MA_vax/run_counterfactual_tables_generic.py --deterministic --out generic_core/examples/run_sim_test_MA_vax/counterfactual_tables_det
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import counterfactual_generic as cf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-reps", type=int, default=500,
                     help="stochastic replications per scenario (paired across scenarios); "
                          "ignored if --deterministic is set")
    ap.add_argument("--deterministic", action="store_true",
                     help="run a single deterministic replication per scenario instead of "
                          "n-reps stochastic ones (median/CI bounds collapse to the point estimate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                     help="output folder [default: <this folder>/counterfactual_tables_<timestamp>]")
    args = ap.parse_args()
    stochastic = not args.deterministic

    now = datetime.now()
    out_folder = args.out or os.path.join(
        HERE, "counterfactual_tables_" + now.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    os.makedirs(out_folder, exist_ok=True)

    print(f"Loading base inputs [model_config={cf.MODEL_CONFIG_FILE} "
          f"fitted_params={cf.FITTED_PARAMS_FILE}] ...")
    base_inputs = cf.load_base_inputs()

    mode = "stochastic" if stochastic else "deterministic"
    print(f"mode={mode}  n_reps={args.n_reps if stochastic else 1}  seed={args.seed}  out={out_folder}")
    cf.save_all_tables(
        base_inputs, out_folder, n_reps=args.n_reps, seed=args.seed, stochastic=stochastic,
        meta={
            "model_config_file": cf.MODEL_CONFIG_FILE,
            "fitted_params_file": cf.FITTED_PARAMS_FILE,
            "stochastic": stochastic,
            "generated_at": now.isoformat(),
        },
    )
    print(f"\nDone. Tables saved to {out_folder}")


if __name__ == "__main__":
    main()
