"""Counterfactual vaccination scenario definitions for `run_scenarios.py`.

Each entry in SCENARIOS maps a scenario name to a dict of overrides consumed
by `model.apply_scenario` (see its docstring for the full set of recognised
keys). Age-group order for `vax_multiplier` and the `_scale` params matches
`model.AGE_GROUP_LABELS`: ["0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"].
"""

SCENARIOS = {
    "baseline": {},

    # i) Reallocate vaccination across age groups.
    "vax_0_only": {
        "vax_multiplier": [1, 0, 0, 0, 0, 0, 0],
    },
    "vax_1_4_only": {
        "vax_multiplier": [0, 1, 0, 0, 0, 0, 0],
    },
    "vax_5_12_only": {
        "vax_multiplier": [0, 0, 1, 0, 0, 0, 0],
    },
    "vax_13_17_only": {
        "vax_multiplier": [0, 0, 0, 1, 0, 0, 0],
    },
    "vax_18_49_only": {
        "vax_multiplier": [0, 0, 0, 0, 1, 0, 0],
    },
    "vax_50_64_only": {
        "vax_multiplier": [0, 0, 0, 0, 0, 1, 0],
    },
    "vax_65plus_only": {
        "vax_multiplier": [0, 0, 0, 0, 0, 0, 1],
    },
    "no_vax_0": {
        "vax_multiplier": [0, 1, 1, 1, 1, 1, 1],
    },
    "no_vax_1_4": {
        "vax_multiplier": [1, 0, 1, 1, 1, 1, 1],
    },
    "no_vax_5_12": {
        "vax_multiplier": [1, 1, 0, 1, 1, 1, 1],
    },
    "no_vax_13_17": {
        "vax_multiplier": [1, 1, 1, 0, 1, 1, 1],
    },
    "no_vax_18_49": {
        "vax_multiplier": [1, 1, 1, 1, 0, 1, 1],
    },
    "no_vax_50_64": {
        "vax_multiplier": [1, 1, 1, 1, 1, 0, 1],
    },
    "no_vax_65plus": {
        "vax_multiplier": [1, 1, 1, 1, 1, 1, 0],
    },
    "boost_65plus_2x": {
        "vax_multiplier": [1, 1, 1, 1, 1, 1, 2],
    },

    # ii) Change vaccine effectiveness.
    "vax_50pct_more_effective": {
        "vax_susceptibility_scale": 0.5,
        "IV_to_H_prop_scale": 0.5,
    },
    "vax_50pct_less_effective": {
        "vax_susceptibility_scale": 1.5,
        "IV_to_H_prop_scale": 1.5,
    },
}
