#!/usr/bin/env python3
"""
Split a combined (legacy) model_config.json into a disease config and a
population config — see generic_core/config_split.py.

Usage:
    python generic_core/examples/split_config.py path/to/model_config.json \
        [--out-dir DIR] [--prefix NAME]

Writes <prefix>disease_config.json and <prefix>population_config.json into
--out-dir (default: the input file's folder), then checks that merging them
back reproduces the original config. Relative paths inside the config (e.g. a
metapop folder) are resolved against the current directory and the repo root.
"""

import argparse
import json
import sys
from pathlib import Path

from generic_core.config_split import merge_configs, split_model_config


def main(argv=None) -> int:
    _ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ap.add_argument("config", type=Path, help="combined model_config.json to split")
    _ap.add_argument("--out-dir", type=Path, default=None, help="output folder (default: next to the input)")
    _ap.add_argument("--prefix", default="", help="filename prefix, e.g. 'MA_vax_'")
    _args = _ap.parse_args(argv)

    with open(_args.config) as _f:
        _combined = json.load(_f)
    _disease, _population, _notes = split_model_config(_combined)

    _out = _args.out_dir or _args.config.parent
    _out.mkdir(parents=True, exist_ok=True)
    _dp = _out / f"{_args.prefix}disease_config.json"
    _pp = _out / f"{_args.prefix}population_config.json"
    _dp.write_text(json.dumps(_disease, indent=2) + "\n")
    _pp.write_text(json.dumps(_population, indent=2) + "\n")
    print(f"Wrote {_dp}\nWrote {_pp}")
    for _n in _notes:
        print(f"Note: {_n}")

    _merged, _report = merge_configs(_disease, _population, strict=False)
    _merged.pop("metapop", None)
    _diff = sorted(_k for _k in set(_combined) | set(_merged) if _combined.get(_k) != _merged.get(_k))
    for _w in _report.warnings:
        print(f"Note: {_w}")
    if _report.errors or _diff:
        print(f"WARNING: merging the split files does not reproduce the original (differs in: {_diff}; "
              f"errors: {_report.errors})")
        return 1
    print("Round trip OK: merging the two files reproduces the original config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
