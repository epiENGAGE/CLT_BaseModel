"""
flu_example_utils_Calib.py
==========================
Backward-compatibility shim.  Calibration utilities were merged into
``flu_example_utils.py``; this module re-exports everything from there.

Mutable module-level variables (``CURRENT_LOCATION``, ``SHARED_FILES_CONFIG``,
``SUBPOP_CONFIG``) are proxied live via ``__getattr__`` so that callers always
see the current values even after ``load_calibrated_austin_inputs`` mutates them.
"""

import flu_example_utils as _base  # noqa: F401

# Import all stable names directly so IDE completion works.
from flu_example_utils import (  # noqa: F401
    EXAMPLES_ROOT,
    FLU_INSTANCES_ROOT,
    CLT_BASEMODEL_ROOT,
    REPO_ROOT,
    AUSTIN_L2_SHARED_FILES_CONFIG,
    AUSTIN_L3_SHARED_FILES_CONFIG,
    DALLAS_L6_SHARED_FILES_CONFIG,
    AUSTIN_L2_SUBPOP_CONFIG,
    AUSTIN_L3_SUBPOP_CONFIG,
    DALLAS_L6_SUBPOP_CONFIG,
    ALL_LOCATION_CONFIGS,
    HIGH_RISK_IHR_MULTIPLIERS,
    IHR_MAX_PROB,
    CALIBRATION_POP_SCALE,
    RATE_PARAMS_TO_SCALE,
    get_shared_files_config,
    get_subpop_config,
    get_austin_subpop_config,
    get_austin_shared_files_config,
    get_calibrated_austin_l_path,
    load_flu_inputs,
    convert_1risk_to_2risk,
    convert_1risk_to_2risk2,
    load_calibrated_austin_inputs,
    load_calibrated_austin_l2_2024_2025_inputs,
    scale_vaccines_df,
    compute_cumulative_vax,
    make_cumvax_markdown_table,
    make_rng_generators,
    build_flu_metapop_model,
    apply_general_init_overrides,
    apply_init_overrides,
    subpop_dropdown_options,
    load_austin_observed_hosp,
)


def __getattr__(name: str):
    """Proxy live mutable globals to the base module."""
    return getattr(_base, name)
