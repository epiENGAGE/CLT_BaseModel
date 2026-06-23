"""
config_parser.py — Load, validate, and parse a generic model JSON config.

Entry point: parse_model_config(json_path, registry=None) -> ModelConfig

All validation happens here before any simulation objects are created.
Errors are raised with messages pointing to the offending config key.

Expected JSON structure
-----------------------
{
    "compartments": ["S", "E", "IP", ...],

    "params": {
        "beta_baseline": 0.3,
        ...
    },

    "transitions": [
        {
            "name": "S_to_E",
            "origin": "S",
            "destination": "E",
            "rate_template": "force_of_infection",
            "rate_config": { ... },
            "jointly_distributed_with": "E_to_IP"   // optional
        },
        ...
    ],

    "transition_groups": [
        {
            "name": "E_split",
            "transition_type": "multinom_deterministic",
            "members": ["E_to_IP", "E_to_IA"]
        },
        ...
    ],

    "epi_metrics": [
        {
            "name": "M",
            "init_val": [[0.0]],
            "metric_template": "infection_induced_immunity",
            "update_config": { ... }
        },
        ...
    ],

    "schedules": [
        {
            "name": "flu_contact_matrix",
            "schedule_template": "contact_matrix",
            "schedule_config": { ... }
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .rate_templates import RATE_TEMPLATE_REGISTRY, RateTemplate
from .metric_templates import METRIC_TEMPLATE_REGISTRY, MetricTemplate
from .schedule_templates import SCHEDULE_TEMPLATE_REGISTRY, ScheduleTemplate


# ---------------------------------------------------------------------------
# Validated config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransitionConfig:
    name: str
    origin: str
    destination: str
    rate_template: str
    rate_config: dict
    jointly_distributed_with: str | None = None


@dataclass
class TransitionGroupConfig:
    name: str
    transition_type: str
    members: list[str]


@dataclass
class EpiMetricConfig:
    name: str
    init_val: Any          # np.ndarray after parsing
    metric_template: str
    update_config: dict


@dataclass
class ScheduleConfig:
    name: str
    schedule_template: str
    schedule_config: dict


@dataclass
class ModelConfig:
    """
    Fully validated, parsed model configuration.

    Consumed by ConfigDrivenSubpopModel to construct all simulation objects.
    """
    compartments: list[str]
    params: dict[str, Any]
    transitions: list[TransitionConfig]
    transition_groups: list[TransitionGroupConfig]
    epi_metrics: list[EpiMetricConfig]
    schedules: list[ScheduleConfig]

    # Per-subpopulation parameter overrides: {subpop_name: {param_name: raw_value}}
    # Raw JSON values (numbers, lists) — not yet parsed through _parse_param_value.
    subpop_params: dict[str, dict] = field(default_factory=dict)

    # Derived name sets (pre-computed for fast lookup)
    param_names: set[str] = field(default_factory=set)
    compartment_names: set[str] = field(default_factory=set)
    transition_names: set[str] = field(default_factory=set)
    schedule_names: set[str] = field(default_factory=set)
    epi_metric_names: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.param_names = set(self.params.keys())
        self.compartment_names = set(self.compartments)
        self.transition_names = {t.name for t in self.transitions}
        self.schedule_names = {s.name for s in self.schedules}
        self.epi_metric_names = {m.name for m in self.epi_metrics}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_model_config(
    json_path: str | Path,
    rate_registry: dict[str, RateTemplate] | None = None,
    metric_registry: dict[str, MetricTemplate] | None = None,
    schedule_registry: dict[str, ScheduleTemplate] | None = None,
    schedules_input: Any = None,
) -> ModelConfig:
    """
    Load and validate a model config JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON config file.
    rate_registry : dict | None
        Rate template registry to use. Defaults to RATE_TEMPLATE_REGISTRY.
    metric_registry : dict | None
        Metric template registry to use. Defaults to METRIC_TEMPLATE_REGISTRY.
    schedule_registry : dict | None
        Schedule template registry to use. Defaults to SCHEDULE_TEMPLATE_REGISTRY.
    schedules_input : Any | None
        Raw schedules input object (e.g. FluSubpopSchedules). Required if
        any schedule templates need to validate their config against it.
        Pass None to skip schedule-input-dependent validation.

    Returns
    -------
    ModelConfig
        Fully validated, parsed config.

    Raises
    ------
    ValueError
        On any validation failure, with a message pointing to the issue.
    """
    with open(json_path) as f:
        raw = json.load(f)
    return parse_model_config_from_dict(
        raw,
        rate_registry=rate_registry,
        metric_registry=metric_registry,
        schedule_registry=schedule_registry,
        schedules_input=schedules_input,
    )


def parse_model_config_from_dict(
    config_dict: dict,
    rate_registry: dict[str, RateTemplate] | None = None,
    metric_registry: dict[str, MetricTemplate] | None = None,
    schedule_registry: dict[str, ScheduleTemplate] | None = None,
    schedules_input: Any = None,
) -> ModelConfig:
    """
    Validate and parse a model config supplied as a Python dict.

    Equivalent to ``parse_model_config`` but accepts an already-loaded dict
    instead of a file path. Useful when the config is built programmatically
    (e.g. from a notebook UI) without writing a temporary file.

    Parameters
    ----------
    config_dict : dict
        Config dict with the same structure as the JSON file.
    rate_registry, metric_registry, schedule_registry : dict | None
        Override the default template registries.
    schedules_input : Any | None
        See ``parse_model_config``.

    Returns
    -------
    ModelConfig
    """
    raw = config_dict
    if rate_registry is None:
        rate_registry = RATE_TEMPLATE_REGISTRY
    if metric_registry is None:
        metric_registry = METRIC_TEMPLATE_REGISTRY
    if schedule_registry is None:
        schedule_registry = SCHEDULE_TEMPLATE_REGISTRY

    # --- 1. Compartments ---
    compartments = _require_list(raw, "compartments", "top level")
    if len(compartments) == 0:
        raise ValueError("ModelConfig: 'compartments' must be non-empty")
    compartment_names = set(compartments)

    # --- 2. Params ---
    params_raw = raw.get("params", {})
    if not isinstance(params_raw, dict):
        raise ValueError("ModelConfig: 'params' must be a JSON object")
    params = {k: _parse_param_value(v, k) for k, v in params_raw.items()}
    param_names = set(params.keys())

    # --- 3. Schedules (parsed early so names are available for rate/metric validation) ---
    schedules_raw = raw.get("schedules", [])
    schedules = []
    for i, s in enumerate(schedules_raw):
        loc = f"schedules[{i}]"
        name = _require_str(s, "name", loc)
        tname = _require_str(s, "schedule_template", loc)
        if tname not in schedule_registry:
            raise ValueError(
                f"{loc}: schedule_template '{tname}' not found in SCHEDULE_TEMPLATE_REGISTRY"
            )
        cfg = s.get("schedule_config", {})
        template = schedule_registry[tname]
        # Only validate if schedules_input is provided
        if schedules_input is not None:
            template.validate_config(cfg, param_names, schedules_input)
        schedules.append(ScheduleConfig(name=name, schedule_template=tname, schedule_config=cfg))
    schedule_names = {s.name for s in schedules}

    # --- 4. Transitions ---
    transitions_raw = _require_list(raw, "transitions", "top level")
    transitions = []
    for i, t in enumerate(transitions_raw):
        loc = f"transitions[{i}]"
        name = _require_str(t, "name", loc)
        origin = _require_str(t, "origin", loc)
        destination = _require_str(t, "destination", loc)
        rate_tname = _require_str(t, "rate_template", loc)
        rate_cfg = t.get("rate_config", {})

        if origin not in compartment_names:
            raise ValueError(f"{loc} ('{name}'): origin '{origin}' not in compartments")
        if destination not in compartment_names:
            raise ValueError(f"{loc} ('{name}'): destination '{destination}' not in compartments")
        if rate_tname not in rate_registry:
            raise ValueError(
                f"{loc} ('{name}'): rate_template '{rate_tname}' not found in RATE_TEMPLATE_REGISTRY"
            )
        rate_template = rate_registry[rate_tname]
        rate_template.validate_config(rate_cfg, param_names, compartment_names, schedule_names)

        jointly = t.get("jointly_distributed_with")
        transitions.append(TransitionConfig(
            name=name,
            origin=origin,
            destination=destination,
            rate_template=rate_tname,
            rate_config=rate_cfg,
            jointly_distributed_with=jointly,
        ))
    transition_names = {t.name for t in transitions}

    # --- 5. Transition groups ---
    transitions_by_name = {t.name: t for t in transitions}
    groups_raw = raw.get("transition_groups", [])
    groups = []
    for i, g in enumerate(groups_raw):
        loc = f"transition_groups[{i}]"
        name = _require_str(g, "name", loc)
        ttype = _require_str(g, "transition_type", loc)
        members = _require_list(g, "members", loc)
        for m in members:
            if m not in transition_names:
                raise ValueError(
                    f"{loc} ('{name}'): member '{m}' is not a declared transition name"
                )
            if transitions_by_name[m].rate_template == "scheduled_exact":
                raise ValueError(
                    f"{loc} ('{name}'): member '{m}' is a 'scheduled_exact' transition, "
                    "which cannot belong to a transition group (it is a deterministic, "
                    "exact flow, not a competing stochastic branch)"
                )
        groups.append(TransitionGroupConfig(name=name, transition_type=ttype, members=members))

    # --- 6. Jointly-distributed consistency check ---
    _validate_jointly_distributed(transitions)

    # --- 7. Epi metrics ---
    metrics_raw = raw.get("epi_metrics", [])
    epi_metrics = []
    for i, m in enumerate(metrics_raw):
        loc = f"epi_metrics[{i}]"
        name = _require_str(m, "name", loc)
        metric_tname = _require_str(m, "metric_template", loc)
        init_val_raw = m.get("init_val")
        if init_val_raw is None:
            raise ValueError(f"{loc} ('{name}'): 'init_val' is required")
        init_val = np.asarray(init_val_raw, dtype=float)
        update_cfg = m.get("update_config", {})

        if metric_tname not in metric_registry:
            raise ValueError(
                f"{loc} ('{name}'): metric_template '{metric_tname}' not found in METRIC_TEMPLATE_REGISTRY"
            )
        metric_template = metric_registry[metric_tname]
        metric_template.validate_config(update_cfg, param_names, transition_names)

        epi_metrics.append(EpiMetricConfig(
            name=name,
            init_val=init_val,
            metric_template=metric_tname,
            update_config=update_cfg,
        ))

    # --- 8. Subpop params ---
    subpop_params_raw = raw.get("subpop_params", {})
    if not isinstance(subpop_params_raw, dict):
        raise ValueError("ModelConfig: 'subpop_params' must be a JSON object")
    for sp_name, sp_overrides in subpop_params_raw.items():
        if not isinstance(sp_overrides, dict):
            raise ValueError(
                f"ModelConfig: 'subpop_params[\"{sp_name}\"]' must be a JSON object"
            )

    config = ModelConfig(
        compartments=compartments,
        params=params,
        transitions=transitions,
        transition_groups=groups,
        epi_metrics=epi_metrics,
        schedules=schedules,
        subpop_params=subpop_params_raw,
    )
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_str(d: dict, key: str, loc: str) -> str:
    if key not in d:
        raise ValueError(f"{loc}: missing required key '{key}'")
    val = d[key]
    if not isinstance(val, str):
        raise ValueError(f"{loc}: '{key}' must be a string, got {type(val).__name__}")
    return val


def _require_list(d: dict, key: str, loc: str) -> list:
    if key not in d:
        raise ValueError(f"{loc}: missing required key '{key}'")
    val = d[key]
    if not isinstance(val, list):
        raise ValueError(f"{loc}: '{key}' must be a list, got {type(val).__name__}")
    return val


def _parse_param_value(val: Any, key: str) -> Any:
    """
    Convert a JSON param value to the appropriate Python type.
    Lists become np.ndarray; scalars stay as-is.
    """
    if isinstance(val, list):
        return np.asarray(val, dtype=float)
    return val


def _validate_jointly_distributed(transitions: list[TransitionConfig]) -> None:
    """
    Check that jointly_distributed_with references are consistent:
    if A references B, then B must reference A.
    """
    by_name = {t.name: t for t in transitions}
    for t in transitions:
        jd = t.jointly_distributed_with
        if jd is None:
            continue
        if t.rate_template == "scheduled_exact":
            raise ValueError(
                f"Transition '{t.name}': 'scheduled_exact' transitions cannot use "
                "jointly_distributed_with (they are deterministic, exact flows, not "
                "competing stochastic branches)"
            )
        if jd not in by_name:
            raise ValueError(
                f"Transition '{t.name}': jointly_distributed_with references "
                f"'{jd}' which is not a declared transition"
            )
        partner = by_name[jd]
        if partner.jointly_distributed_with != t.name:
            raise ValueError(
                f"Transition '{t.name}' references '{jd}' as jointly_distributed_with, "
                f"but '{jd}' does not reference '{t.name}' back "
                f"(got '{partner.jointly_distributed_with}')"
            )
