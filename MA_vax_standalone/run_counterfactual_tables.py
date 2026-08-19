"""Compute the counterfactual vaccination-impact tables (S.A.1-S.A.6 analogs,
see `MA_vax_standalone/counterfactual.py`) and save them to CSV.

Running many stochastic replications (needed for tight confidence intervals)
can take a while, so this is split from display: run this script once, then
point `counterfactual_notebook.py` at its output folder to load and render
the results instantly, without re-simulating.

Run:
    uv run python MA_vax_standalone/run_counterfactual_tables.py --n-reps 1000 --point best
    uv run python MA_vax_standalone/run_counterfactual_tables.py \
        --fit-folder MA_vax_standalone/outputs_2026-07-29 --method emcee --n-reps 500 --point best
    uv run python MA_vax_standalone/run_counterfactual_tables.py --method emcee --fit-folder MA_vax_standalone/outputs_2026-07-30 --deterministic --out MA_vax_standalone/counterfactual_tables_det --point best
    uv run python MA_vax_standalone/run_counterfactual_tables.py --method emcee --fit-folder MA_vax_standalone/outputs_2026-07-30_age_ihr_scale --deterministic --out MA_vax_standalone/counterfactual_tables_det_age_ihr_scale --point best
    uv run python MA_vax_standalone/run_counterfactual_tables.py --method emcee --fit-folder MA_vax_standalone/outputs_2026-07-30_age_ihr_scale_no_V_infec --deterministic --out MA_vax_standalone/counterfactual_tables_det_age_ihr_scale_no_V_infec --point best
    

"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from MA_vax_standalone import counterfactual as cf

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-folder", default=None,
                     help="outputs_* output folder from fit_bayesian.py [default: unfitted default_params()]")
    ap.add_argument("--method", choices=["emcee", "pyabc"], default="emcee")
    ap.add_argument("--point", choices=["mean", "best"], default="best",
                     help="which posterior point estimate to build the baseline inputs from -- "
                          "'best' (single highest-support sampled parameter vector, preserves "
                          "correlations between parameters) or 'mean' (per-parameter posterior "
                          "mean, ignores correlations) [default: best]")
    ap.add_argument("--n-reps", type=int, default=500,
                     help="stochastic replications per scenario (paired across scenarios); "
                          "ignored if --deterministic is set")
    ap.add_argument("--deterministic", action="store_true",
                     help="run a single deterministic replication per scenario instead of "
                          "n-reps stochastic ones (median/CI bounds collapse to the point estimate)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                     help="output folder [default: MA_vax_standalone/counterfactual_tables_<timestamp>]")
    args = ap.parse_args()
    stochastic = not args.deterministic

    now = datetime.now()
    out_folder = args.out or os.path.join(
        HERE, "counterfactual_tables_" + now.strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    os.makedirs(out_folder, exist_ok=True)

    print(f"Loading base inputs [fit_folder={args.fit_folder} method={args.method} point={args.point}] ...")
    base_inputs = cf.load_base_inputs(args.fit_folder, args.method, args.point)

    mode = "stochastic" if stochastic else "deterministic"
    print(f"mode={mode}  n_reps={args.n_reps if stochastic else 1}  seed={args.seed}  out={out_folder}")
    cf.save_all_tables(
        base_inputs, out_folder, n_reps=args.n_reps, seed=args.seed, stochastic=stochastic,
        meta={
            "fit_folder": args.fit_folder,
            "method": args.method,
            "point": args.point,
            "stochastic": stochastic,
            "generated_at": now.isoformat(),
        },
    )
    print(f"\nDone. Tables saved to {out_folder}")


if __name__ == "__main__":
    main()
