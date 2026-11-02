"""
config_split.py — Split a model config into a DISEASE file and a POPULATION
file, and merge the two back into the combined config dict the rest of
generic_core consumes.

The disease file describes the pathogen (compartments, transitions, rates,
vaccine effects, which schedules exist and how they're wired, initial seeds as
fractions of the population). The population file describes the place (age
bands, risk groups and high-risk proportion, population per subpopulation,
contact matrices, travel matrix, and the location's schedule CSVs). Swapping
one file keeps the other: same disease in a new region, or a new disease in
the same region.

``merge_configs(disease, population)`` rebuilds exactly the combined dict
shape that ``model_config.json`` has always had, so every downstream consumer
(``parse_model_config_from_dict``, ``model_factory``, ``fitting``, the
notebook run paths, exported scripts) is unchanged. Legacy combined files are
still accepted everywhere: ``resolve_disease_population`` splits them on load.

Per-age disease parameters are stored keyed by age-band label
(``{"by_age_band": {"0-4": [r1..rR], ...}}``) and must match the population's
bands exactly — there is no automatic remapping. A mismatch is reported as an
error (``strict=True``) or the parameter is dropped and listed in the report
(``strict=False``, used by the notebook so the user can re-enter it).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DISEASE_TYPE = "disease"
POPULATION_TYPE = "population"
COMBINED_TYPE = "combined"
SCHEMA_VERSION = 1
SINGLE_POP_NAME = "aggregate_pop"
CONTACT_KINDS = ("total", "school", "work")

# Combined-config top-level keys that belong to the disease file verbatim.
_DISEASE_KEYS = (
    "compartments", "transitions", "transition_groups", "epi_metrics",
    "schedules", "ve_derivation", "analysis_metrics", "simulation_settings",
)
# Combined-config top-level keys that are rebuilt from the population file.
_POPULATION_KEYS = (
    "age_risk", "total_population", "initial_conditions", "input_files",
    "subpop_params", "subpop_daily_vaccines", "metapop",
)
# Keys that only identify the file; never copied into the combined dict.
_HEADER_KEYS = ("config_type", "schema_version", "name", "description", "geography")

# Default search roots for relative paths in configs: the current directory
# (normally the repo root when the notebook runs) and the repo root itself.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOTS = (Path.cwd(), _REPO_ROOT)


class ConfigCompatibilityError(ValueError):
    """Raised by ``merge_configs(strict=True)`` when the disease and
    population files can't be combined. ``.errors`` lists every problem."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__(
            "Disease and population configs are incompatible:\n  - "
            + "\n  - ".join(self.errors)
        )


@dataclass
class MergeReport:
    """What ``merge_configs`` couldn't apply cleanly.

    ``errors`` are real incompatibilities (age-band / risk-group mismatch,
    malformed travel matrix); with ``strict=False`` the offending items are
    dropped and named in ``dropped_params`` / ``dropped_seeds`` /
    ``dropped_epi_metrics`` instead of raising. ``warnings`` are things that
    still run but may not be what the user intended (e.g. a schedule the
    disease uses with no data file in the population)."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_params: list[str] = field(default_factory=list)
    dropped_seeds: list[str] = field(default_factory=list)
    dropped_epi_metrics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def age_band_labels(age_risk: dict | None) -> list[str]:
    """Labels used to key per-age values: the named bands when the population
    defines them, else ``group_1..group_A`` (count-only age groups)."""
    _ar = age_risk or {}
    _bands = _ar.get("age_groups") or []
    if _bands:
        return [str(_b) for _b in _bands]
    return [f"group_{_i + 1}" for _i in range(int(_ar.get("num_age_groups", 1) or 1))]


def config_kind(raw: dict) -> str:
    """``"disease"``, ``"population"`` or ``"combined"`` (legacy single file).

    Uses the explicit ``config_type`` header when present, otherwise sniffs
    the keys: a population file has ``subpopulations``/``data_files`` and no
    ``compartments``; anything with ``compartments`` and population-ish keys
    (or nothing else) is a legacy combined config."""
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object")
    _t = raw.get("config_type")
    if _t in (DISEASE_TYPE, POPULATION_TYPE):
        return _t
    if "compartments" not in raw and (
        "subpopulations" in raw or "data_files" in raw or "contact_matrices" in raw
    ):
        return POPULATION_TYPE
    return COMBINED_TYPE


def contact_matrix_param_names(config: dict) -> dict[str, str]:
    """``{"total"|"school"|"work": param_name}`` the config's contact_matrix
    schedule(s) read; defaults to ``<kind>_contact_matrix`` (the names the
    notebook always writes)."""
    _out = {_k: f"{_k}_contact_matrix" for _k in CONTACT_KINDS}
    for _s in (config or {}).get("schedules", []) or []:
        if _s.get("schedule_template") != "contact_matrix":
            continue
        _sc = _s.get("schedule_config", {}) or {}
        for _k in CONTACT_KINDS:
            _p = _sc.get(f"{_k}_contact_matrix_param")
            if isinstance(_p, str) and _p:
                _out[_k] = _p
    return _out


def _is_age_risk_array(value, A: int, R: int) -> bool:
    if not isinstance(value, list) or len(value) != A:
        return False
    for _row in value:
        if not isinstance(_row, list) or len(_row) != R:
            return False
        if not all(isinstance(_x, (int, float)) and not isinstance(_x, bool) for _x in _row):
            return False
    return True


def _to_bands(arr, labels: list[str]) -> dict:
    return {"by_age_band": {_lab: [float(_x) for _x in _row] for _lab, _row in zip(labels, arr)}}


def _is_banded(value) -> bool:
    return isinstance(value, dict) and "by_age_band" in value


def _from_bands(spec: dict, labels: list[str], R: int, what: str):
    """Rebuild an A×R nested list from a ``by_age_band`` spec. Returns
    ``(array, error_message_or_None)``."""
    _bands = spec.get("by_age_band") or {}
    _have = list(_bands.keys())
    # Same bands in a different order is fine -- values are looked up by label.
    if set(_have) != set(labels):
        _detail = ""
        if set(_have) & set(labels):
            _missing = [_l for _l in labels if _l not in _bands]
            _extra = [_l for _l in _have if _l not in labels]
            _bits = ([f"missing {_missing}"] if _missing else []) + (
                [f"extra {_extra}"] if _extra else []
            )
            _detail = f" ({'; '.join(_bits)})"
        return None, (
            f"{what}: defined for age bands {_have}, but the population has "
            f"{labels}{_detail}"
        )
    _out = []
    for _lab in labels:
        _row = _bands[_lab]
        if not isinstance(_row, list):
            _row = [_row]
        if len(_row) != R:
            return None, (
                f"{what}: age band '{_lab}' has {len(_row)} risk-group value(s) "
                f"but the population has {R} risk group(s)"
            )
        _out.append([float(_x) for _x in _row])
    return _out, None


def _snap(x: float) -> float:
    """Undo float noise from fraction × population (2.0000000000000004 → 2.0)."""
    _r = round(x)
    return float(_r) if abs(x - _r) < 1e-6 else float(x)


def _resolve_path(path: str, search_roots) -> Path | None:
    _p = Path(path)
    if _p.is_absolute():
        return _p if _p.exists() else None
    for _root in search_roots:
        _cand = Path(_root) / _p
        if _cand.exists():
            return _cand
    return None


def _read_folder_layout(folder: str, search_roots) -> dict | None:
    """``{"subpopulations": [...], "travel_matrix": [...]}`` from a metapop
    folder's ``metapop_config.json``, or None if it can't be found."""
    if not folder or not str(folder).strip():
        return None
    _f = _resolve_path(str(folder).strip(), search_roots)
    if _f is None or not (_f / "metapop_config.json").is_file():
        return None
    with open(_f / "metapop_config.json") as _fh:
        _mc = json.load(_fh)
    return {
        "subpopulations": list(_mc.get("subpopulations", [])),
        "travel_matrix": _mc.get("travel_matrix"),
    }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_model_config(combined: dict, search_roots=DEFAULT_SEARCH_ROOTS):
    """Split a legacy combined config into ``(disease, population, notes)``.

    ``notes`` is a list of human-readable remarks about anything that couldn't
    be carried over exactly (e.g. subpops seeded with different fractions,
    which a single disease-level seed pattern can't represent).
    ``merge_configs(*split_model_config(c)[:2])`` reproduces ``c`` apart from
    the added ``metapop`` key (and these noted approximations)."""
    _c = copy.deepcopy(combined or {})
    _notes: list[str] = []
    _ar = dict(_c.get("age_risk", {}) or {})
    _A = int(_ar.get("num_age_groups", 1) or 1)
    _R = int(_ar.get("num_risk_groups", 1) or 1)
    _labels = age_band_labels(_ar)
    _cm_names = contact_matrix_param_names(_c)
    _cm_param_to_kind = {_v: _k for _k, _v in _cm_names.items()}

    disease: dict = {"config_type": DISEASE_TYPE, "schema_version": SCHEMA_VERSION}
    population: dict = {"config_type": POPULATION_TYPE, "schema_version": SCHEMA_VERSION}

    # --- disease: structural keys ---
    for _k in _DISEASE_KEYS:
        if _k in _c:
            disease[_k] = _c[_k]

    # --- params: contact matrices → population; A×R → band-keyed ---
    _params = {}
    _contacts = {}
    for _name, _val in (_c.get("params", {}) or {}).items():
        if _name in _cm_param_to_kind:
            _contacts[_cm_param_to_kind[_name]] = _val
        elif _is_age_risk_array(_val, _A, _R):
            _params[_name] = _to_bands(_val, _labels)
        else:
            _params[_name] = _val
    disease["params"] = _params
    if any(_is_banded(_v) for _v in _params.values()):
        disease["num_risk_groups"] = _R

    # --- epi-metric starting values ---
    if disease.get("epi_metrics"):
        _metrics = []
        for _m in disease["epi_metrics"]:
            _m = dict(_m)
            if _is_age_risk_array(_m.get("init_val"), _A, _R):
                _m["init_val"] = _to_bands(_m["init_val"], _labels)
                disease["num_risk_groups"] = _R
            _metrics.append(_m)
        disease["epi_metrics"] = _metrics

    # --- population: structure + data ---
    if _ar:
        population["age_risk"] = _ar
    if _contacts:
        population["contact_matrices"] = _contacts
    if "input_files" in _c:
        population["data_files"] = _c["input_files"]
    if "subpop_daily_vaccines" in _c:
        population["subpop_daily_vaccines"] = _c["subpop_daily_vaccines"]

    _ic = _c.get("initial_conditions", {}) or {}
    _sp_params = _c.get("subpop_params", {}) or {}
    _layout = _c.get("metapop") or _read_folder_layout(
        (_c.get("input_files", {}) or {}).get("metapop_folder", ""), search_roots,
    )
    if _layout and _layout.get("subpopulations"):
        _sp_names = list(_layout["subpopulations"])
        if _layout.get("travel_matrix") is not None:
            population["travel_matrix"] = _layout["travel_matrix"]
    else:
        _sp_names = list(_ic.keys())
    for _n in list(_ic.keys()) + list(_sp_params.keys()):
        if _n not in _sp_names:
            _sp_names.append(_n)

    _subpops = {}
    for _n in _sp_names:
        _entry = {}
        if _n in _ic and "population" in _ic[_n]:
            _entry["population"] = _ic[_n]["population"]
        _ov = dict(_sp_params.get(_n, {}) or {})
        _sp_cm = {_cm_param_to_kind[_p]: _ov.pop(_p) for _p in list(_ov) if _p in _cm_param_to_kind}
        if _sp_cm:
            _entry["contact_matrices"] = _sp_cm
        if _ov:
            _entry["params"] = _ov
        _subpops[_n] = _entry
    if _subpops:
        population["subpopulations"] = _subpops
    if not any("population" in _e for _e in _subpops.values()) and "total_population" in _c:
        # No per-cell population (e.g. a metapop folder whose sizes live in
        # the initial_conditions_<name>.json files) -- keep the scalar total.
        population["total_population"] = _c["total_population"]

    # --- seeds → fractions of population (pooled over subpops) ---
    _seed_num: dict[str, np.ndarray] = {}
    _pop_den = np.zeros((_A, _R))
    _per_sp_frac: dict[str, list[np.ndarray]] = {}
    for _n, _e in _ic.items():
        if "population" not in _e:
            continue
        _pop = np.asarray(_e["population"], dtype=float)
        if _pop.shape != (_A, _R):
            _notes.append(f"Initial conditions for '{_n}' have shape {_pop.shape}, not {(_A, _R)}; seeds skipped.")
            continue
        _pop_den = _pop_den + _pop
        for _comp, _arr in (_e.get("seeds", {}) or {}).items():
            _arr = np.asarray(_arr, dtype=float)
            _seed_num[_comp] = _seed_num.get(_comp, np.zeros((_A, _R))) + _arr
            if np.any((_pop == 0) & (_arr > 0)):
                _notes.append(
                    f"'{_n}' seeds {_comp} into an age/risk cell with zero population; "
                    "that seed can't be expressed as a fraction and is dropped."
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                _per_sp_frac.setdefault(_comp, []).append(np.where(_pop > 0, _arr / _pop, 0.0))
    if _seed_num:
        _seeds = {}
        for _comp, _num in _seed_num.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                _frac = np.where(_pop_den > 0, _num / _pop_den, 0.0)
            _fracs = _per_sp_frac.get(_comp, [])
            if len(_fracs) > 1 and not all(np.allclose(_f, _fracs[0], rtol=1e-9, atol=1e-15) for _f in _fracs):
                _notes.append(
                    f"Subpopulations were seeded with different fractions of {_comp}; "
                    "the disease file stores one pooled fraction applied to every subpopulation."
                )
            _seeds[_comp] = _to_bands(_frac.tolist(), _labels)
        disease["initial_seeds"] = _seeds
        disease["num_risk_groups"] = _R

    _known = set(_DISEASE_KEYS) | set(_POPULATION_KEYS) | {"params"} | set(_HEADER_KEYS)
    for _k in _c:
        if _k not in _known:
            disease[_k] = _c[_k]
            _notes.append(f"Unrecognised key '{_k}' kept in the disease file.")
    return disease, population, _notes


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _schedule_attr_for_input_key(key: str) -> str | None:
    # Local mirror of model_factory._INPUT_FILE_SCHEDULE_KEYS (importing
    # model_factory here would pull in torch/flu_core for a pure-dict helper).
    _base = {
        "absolute_humidity_csv": "absolute_humidity_df",
        "school_work_calendar_csv": "school_work_calendar_df",
        "mobility_csv": "mobility_df",
        "vaccines_csv": "daily_vaccines_df",
    }
    if key in _base:
        return _base[key]
    if key.endswith("_schedule_csv"):
        return key[: -len("_schedule_csv")] + "_df"
    return None


def _check_travel(travel, n: int, report: MergeReport):
    try:
        _arr = np.asarray(travel, dtype=float)
    except Exception:
        report.errors.append("travel_matrix is not a numeric matrix")
        return
    if _arr.shape != (n, n):
        report.errors.append(f"travel_matrix has shape {_arr.shape}; expected {(n, n)} for {n} subpopulations")
        return
    _rows = _arr.sum(axis=1)
    if not np.allclose(_rows, 1.0, atol=1e-6):
        report.errors.append(f"travel_matrix rows must each sum to 1 (got {np.round(_rows, 4).tolist()})")


def merge_configs(disease: dict, population: dict, strict: bool = True):
    """Combine a disease file and a population file into the legacy combined
    config dict. Returns ``(combined, report)``.

    With ``strict=True`` any ``report.errors`` raise ``ConfigCompatibilityError``.
    With ``strict=False`` incompatible per-age params / seeds / epi-metric
    starting values are left out (named in the report) so a UI can ask the
    user to re-enter them."""
    _d = copy.deepcopy(disease or {})
    _p = copy.deepcopy(population or {})
    report = MergeReport()
    _ar = dict(_p.get("age_risk", {}) or {})
    _A = int(_ar.get("num_age_groups", 1) or 1)
    _R = int(_ar.get("num_risk_groups", 1) or 1)
    _labels = age_band_labels(_ar)

    combined: dict = {}
    for _k, _v in _d.items():
        if _k in _HEADER_KEYS or _k in ("params", "initial_seeds", "num_risk_groups", "epi_metrics"):
            continue
        combined[_k] = _v

    # --- params ---
    _params = {}
    for _name, _val in (_d.get("params", {}) or {}).items():
        if _is_banded(_val):
            _arr, _err = _from_bands(_val, _labels, _R, f"Parameter '{_name}'")
            if _err:
                report.errors.append(_err)
                report.dropped_params.append(_name)
                continue
            _params[_name] = _arr
        else:
            _params[_name] = _val
    _cm_names = contact_matrix_param_names(_d)
    for _kind, _mat in (_p.get("contact_matrices", {}) or {}).items():
        if _kind not in _cm_names:
            report.warnings.append(f"Unknown contact matrix kind '{_kind}' ignored (expected {list(CONTACT_KINDS)})")
            continue
        _mat_arr = np.asarray(_mat, dtype=float)
        if _mat_arr.ndim == 2 and _mat_arr.shape != (_A, _A):
            report.errors.append(f"{_kind} contact matrix is {_mat_arr.shape[0]}×{_mat_arr.shape[1]} but the population has {_A} age group(s)")
        _params[_cm_names[_kind]] = _mat
    combined["params"] = _params
    if any(_s.get("schedule_template") == "contact_matrix" for _s in _d.get("schedules", []) or []):
        _missing = [_k for _k in CONTACT_KINDS if _cm_names[_k] not in _params]
        _sp_has = any(
            "contact_matrices" in _e for _e in (_p.get("subpopulations", {}) or {}).values()
        )
        _csv_has = any(
            (_p.get("data_files", {}) or {}).get(f"{_k}_contact_matrix_csv") for _k in CONTACT_KINDS
        )
        if _missing and not _sp_has and not _csv_has:
            report.warnings.append(
                "The disease uses a contact-matrix schedule but the population file has no "
                f"{', '.join(_missing)} contact matrix; defaults will be used."
            )

    # --- epi metrics ---
    if "epi_metrics" in _d:
        _metrics = []
        for _m in _d["epi_metrics"] or []:
            _m = dict(_m)
            if _is_banded(_m.get("init_val")):
                _arr, _err = _from_bands(_m["init_val"], _labels, _R, f"Starting value of '{_m.get('name')}'")
                if _err:
                    report.errors.append(_err)
                    report.dropped_epi_metrics.append(str(_m.get("name")))
                    _arr = np.zeros((_A, _R)).tolist()
                _m["init_val"] = _arr
            _metrics.append(_m)
        combined["epi_metrics"] = _metrics

    # --- population structure ---
    if _ar:
        combined["age_risk"] = _ar
    if "data_files" in _p:
        combined["input_files"] = _p["data_files"]
    if "subpop_daily_vaccines" in _p:
        combined["subpop_daily_vaccines"] = _p["subpop_daily_vaccines"]

    _subpops = _p.get("subpopulations", {}) or {}
    _sp_names = list(_subpops.keys())
    if "travel_matrix" in _p:
        _check_travel(_p["travel_matrix"], len(_sp_names), report)
        combined["metapop"] = {"subpopulations": _sp_names, "travel_matrix": _p["travel_matrix"]}

    _subpop_params = {}
    for _n, _e in _subpops.items():
        _ov = dict(_e.get("params", {}) or {})
        for _kind, _mat in (_e.get("contact_matrices", {}) or {}).items():
            if _kind in _cm_names:
                _ov[_cm_names[_kind]] = _mat
        if _ov:
            _subpop_params[_n] = _ov
    if _subpop_params:
        combined["subpop_params"] = _subpop_params

    # --- initial conditions: population + seeds = fraction × population ---
    _seed_fracs = {}
    for _comp, _spec in (_d.get("initial_seeds", {}) or {}).items():
        if _comp not in (_d.get("compartments") or []):
            report.warnings.append(f"Seed for unknown compartment '{_comp}' ignored")
            continue
        if _is_banded(_spec):
            _arr, _err = _from_bands(_spec, _labels, _R, f"Initial seed fraction for '{_comp}'")
            if _err:
                report.errors.append(_err)
                report.dropped_seeds.append(_comp)
                continue
            _seed_fracs[_comp] = np.asarray(_arr, dtype=float)
        else:
            _seed_fracs[_comp] = np.full((_A, _R), float(_spec))
    _ic = {}
    _seeds_unused = []
    for _n, _e in _subpops.items():
        if "population" not in _e:
            if _seed_fracs:
                _seeds_unused.append(_n)
            continue
        _pop = np.asarray(_e["population"], dtype=float)
        if _pop.shape != (_A, _R):
            report.errors.append(f"Population for '{_n}' is {_pop.shape}; expected {(_A, _R)} from age_risk")
            continue
        _ic[_n] = {
            "population": _e["population"],
            "seeds": {
                _comp: [[_snap(_x) for _x in _row] for _row in (_frac * _pop).tolist()]
                for _comp, _frac in _seed_fracs.items()
            },
        }
    if _seeds_unused:
        report.warnings.append(
            f"Subpopulation(s) {_seeds_unused} have no population in the population file, so the "
            "disease's initial seeds can't be applied there; initial conditions come from the "
            "metapop folder instead."
        )
    if _ic:
        combined["initial_conditions"] = _ic
        combined["total_population"] = _snap(float(sum(np.sum(_v["population"]) for _v in _ic.values())))
        if float(combined["total_population"]).is_integer():
            combined["total_population"] = int(combined["total_population"])
    elif "total_population" in _p:
        combined["total_population"] = _p["total_population"]

    # --- schedule data the disease needs but the population doesn't supply ---
    _df = _p.get("data_files", {}) or {}
    if not (_df.get("metapop_folder") or "").strip():
        _supplied = {_schedule_attr_for_input_key(_k) for _k, _v in _df.items() if isinstance(_v, str) and _v.strip()}
        _needed = set()
        for _s in _d.get("schedules", []) or []:
            for _k, _v in (_s.get("schedule_config", {}) or {}).items():
                if _k.endswith("df_attribute") and isinstance(_v, str):
                    _needed.add(_v)
        _needed.discard("transmission_multiplier_df")
        _missing = sorted(_needed - _supplied)
        if _missing:
            report.warnings.append(
                f"The disease uses schedule data {_missing} but the population file names no CSV "
                "for it; constant values will be used unless set in the Schedules step."
            )

    if strict and report.errors:
        raise ConfigCompatibilityError(report.errors)
    return combined, report


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(path) -> dict:
    with open(path) as _f:
        return json.load(_f)


def resolve_disease_population(disease_raw=None, population_raw=None, search_roots=DEFAULT_SEARCH_ROOTS):
    """Normalise whatever the user supplied into ``(disease, population, notes)``.

    Either slot may hold a legacy combined config: it is split, and its other
    half fills the other slot unless that slot was given explicitly. A file of
    the wrong kind in a slot (a population file loaded as the disease) raises
    ``ValueError``. Missing halves come back as ``{}``."""
    _notes = []
    disease = population = None
    _legacy_pop = _legacy_dis = None
    if disease_raw:
        _k = config_kind(disease_raw)
        if _k == POPULATION_TYPE:
            raise ValueError("The file loaded as the disease config is a population config")
        if _k == COMBINED_TYPE:
            disease, _legacy_pop, _n = split_model_config(disease_raw, search_roots)
            _notes.append("Disease slot held a combined (legacy) model config; it was split into disease + population.")
            _notes.extend(_n)
        else:
            disease = disease_raw
    if population_raw:
        _k = config_kind(population_raw)
        if _k == DISEASE_TYPE:
            raise ValueError("The file loaded as the population config is a disease config")
        if _k == COMBINED_TYPE:
            _legacy_dis, population, _n = split_model_config(population_raw, search_roots)
            _notes.append("Population slot held a combined (legacy) model config; only its population part is used.")
            _notes.extend(_n)
        else:
            population = population_raw
    if population is None and _legacy_pop is not None:
        population = _legacy_pop
    if disease is None and _legacy_dis is not None:
        disease = _legacy_dis
    return disease or {}, population or {}, _notes


def load_merged_config(disease_path, population_path=None, strict=True, search_roots=DEFAULT_SEARCH_ROOTS) -> dict:
    """Read a disease file + population file (either may be a legacy combined
    config; ``population_path`` may be omitted for a combined file) and return
    the merged combined config dict. For scripts."""
    _d, _p, _ = resolve_disease_population(
        load_json(disease_path),
        load_json(population_path) if population_path else None,
        search_roots,
    )
    combined, _report = merge_configs(_d, _p, strict=strict)
    for _w in _report.warnings:
        print(f"Note: {_w}")
    return combined


def parse_split_config(disease_path, population_path=None, **kwargs):
    """``parse_model_config`` for a disease + population file pair."""
    from generic_core.config_parser import parse_model_config_from_dict
    return parse_model_config_from_dict(load_merged_config(disease_path, population_path), **kwargs)
