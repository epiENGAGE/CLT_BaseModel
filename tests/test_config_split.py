"""
Disease / population config split (generic_core/config_split.py).

A legacy combined model_config.json is split into a disease file and a
population file; merging them back must reproduce the original exactly, so
every downstream consumer keeps working. Also covers the compatibility checks
(age bands, risk groups, travel matrix), seed fraction <-> count conversion,
and the metapop layout coming from the config rather than the folder.
"""

import copy
import json

import numpy as np
import pytest

import clt_toolkit as clt

from generic_core.config_split import (
    COMBINED_TYPE,
    DISEASE_TYPE,
    POPULATION_TYPE,
    ConfigCompatibilityError,
    config_kind,
    load_merged_config,
    merge_configs,
    resolve_disease_population,
    split_model_config,
)

EXAMPLES = clt.utils.PROJECT_ROOT / "generic_core" / "examples"
EXAMPLE_METAPOP_FOLDER = EXAMPLES / "example_metapop_inputs"
LEGACY_CONFIGS = [
    EXAMPLES / "sir_config.json",
    EXAMPLES / "MA_vax_2026_10" / "model_config_MA_vax.json",
    EXAMPLE_METAPOP_FOLDER / "model_config.json",
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _without_metapop(cfg):
    cfg = dict(cfg)
    cfg.pop("metapop", None)
    return cfg


@pytest.mark.parametrize("path", LEGACY_CONFIGS, ids=lambda p: p.name)
def test_split_then_merge_round_trips(path):
    combined = _load(path)
    disease, population, notes = split_model_config(combined)
    assert config_kind(disease) == DISEASE_TYPE
    assert config_kind(population) == POPULATION_TYPE
    assert notes == []
    merged, report = merge_configs(disease, population)
    assert report.ok
    assert _without_metapop(merged) == combined


def test_disease_file_has_no_location_data():
    disease, population, _ = split_model_config(_load(LEGACY_CONFIGS[1]))
    for key in ("age_risk", "initial_conditions", "input_files", "total_population"):
        assert key not in disease
    assert "total_contact_matrix" not in disease["params"]
    assert set(population["contact_matrices"]) == {"total", "school", "work"}
    # Per-age params are keyed by the population's age bands.
    assert list(disease["params"]["I_to_H_prop"]["by_age_band"]) == population["age_risk"]["age_groups"]


def test_seeds_stored_as_fractions_and_rescaled_to_new_population():
    combined = _load(LEGACY_CONFIGS[1])
    disease, population, _ = split_model_config(combined)
    pop = np.asarray(population["subpopulations"]["aggregate_pop"]["population"], dtype=float)
    seeds = np.asarray(combined["initial_conditions"]["aggregate_pop"]["seeds"]["E"], dtype=float)
    frac = np.array([v for v in disease["initial_seeds"]["E"]["by_age_band"].values()])
    np.testing.assert_allclose(frac, seeds / pop)

    # Same disease, a population twice as large -> twice the seeds.
    bigger = copy.deepcopy(population)
    bigger["subpopulations"]["aggregate_pop"]["population"] = (2 * pop).tolist()
    merged, _ = merge_configs(disease, bigger)
    np.testing.assert_allclose(merged["initial_conditions"]["aggregate_pop"]["seeds"]["E"], 2 * seeds)
    assert merged["total_population"] == int(2 * pop.sum())


def test_age_band_mismatch_strict_raises_and_lenient_drops():
    disease, _, _ = split_model_config(_load(LEGACY_CONFIGS[1]))  # 7 MA age bands
    _, other_pop, _ = split_model_config(_load(LEGACY_CONFIGS[2]))  # 3 count-only groups
    with pytest.raises(ConfigCompatibilityError) as exc:
        merge_configs(disease, other_pop)
    assert any("I_to_H_prop" in e for e in exc.value.errors)

    merged, report = merge_configs(disease, other_pop, strict=False)
    assert set(report.dropped_params) == {"I_to_H_prop", "IV_to_H_prop", "H_to_D_prop", "vax_susceptibility"}
    assert report.dropped_seeds == ["E"]
    assert "I_to_H_prop" not in merged["params"]
    assert merged["params"]["beta_baseline"] == disease["params"]["beta_baseline"]


def test_risk_group_mismatch_is_an_error():
    disease, population, _ = split_model_config(_load(LEGACY_CONFIGS[1]))
    two_risk = copy.deepcopy(population)
    two_risk["age_risk"]["num_risk_groups"] = 2
    two_risk["subpopulations"]["aggregate_pop"]["population"] = [
        [p[0] / 2, p[0] / 2] for p in population["subpopulations"]["aggregate_pop"]["population"]
    ]
    _, report = merge_configs(disease, two_risk, strict=False)
    assert "I_to_H_prop" in report.dropped_params
    assert any("risk-group" in e for e in report.errors)


def test_band_order_does_not_matter():
    disease, population, _ = split_model_config(_load(LEGACY_CONFIGS[1]))
    shuffled = copy.deepcopy(disease)
    bands = shuffled["params"]["I_to_H_prop"]["by_age_band"]
    shuffled["params"]["I_to_H_prop"]["by_age_band"] = dict(reversed(list(bands.items())))
    a, _ = merge_configs(disease, population)
    b, _ = merge_configs(shuffled, population)
    assert a["params"]["I_to_H_prop"] == b["params"]["I_to_H_prop"]


def test_missing_schedule_data_is_a_warning():
    disease, population, _ = split_model_config(_load(LEGACY_CONFIGS[1]))
    no_files = copy.deepcopy(population)
    del no_files["data_files"]["absolute_humidity_csv"]
    _, report = merge_configs(disease, no_files)
    assert report.ok
    assert any("absolute_humidity_df" in w for w in report.warnings)


def test_bad_travel_matrix_is_an_error():
    disease, population, _ = split_model_config(_load(LEGACY_CONFIGS[2]))
    assert population["travel_matrix"] == _load(EXAMPLE_METAPOP_FOLDER / "metapop_config.json")["travel_matrix"]
    bad = copy.deepcopy(population)
    bad["travel_matrix"] = [[0.5, 0.2], [0.03, 0.97]]
    with pytest.raises(ConfigCompatibilityError, match="sum to 1"):
        merge_configs(disease, bad)


def test_legacy_file_in_either_slot():
    combined = _load(LEGACY_CONFIGS[1])
    d, p, notes = resolve_disease_population(combined, None)
    assert config_kind(combined) == COMBINED_TYPE
    assert d["config_type"] == DISEASE_TYPE and p["config_type"] == POPULATION_TYPE
    assert notes
    with pytest.raises(ValueError):
        resolve_disease_population(p, None)  # population file in the disease slot


def test_load_merged_config_matches_legacy(tmp_path):
    combined = _load(LEGACY_CONFIGS[1])
    disease, population, _ = split_model_config(combined)
    (tmp_path / "d.json").write_text(json.dumps(disease))
    (tmp_path / "p.json").write_text(json.dumps(population))
    assert _without_metapop(load_merged_config(tmp_path / "d.json", tmp_path / "p.json")) == combined
    assert load_merged_config(LEGACY_CONFIGS[1]) == combined


def test_metapop_layout_from_config_overrides_folder():
    from generic_core.model_factory import metapop_layout

    combined = _load(LEGACY_CONFIGS[2])
    names, travel = metapop_layout(combined, EXAMPLE_METAPOP_FOLDER)
    assert names == ["SubpopA", "SubpopB"]

    cfg = dict(combined)
    cfg["metapop"] = {"subpopulations": ["SubpopA", "SubpopB"], "travel_matrix": [[0.9, 0.1], [0.2, 0.8]]}
    names2, travel2 = metapop_layout(cfg, EXAMPLE_METAPOP_FOLDER)
    assert names2 == names
    np.testing.assert_allclose(travel2, [[0.9, 0.1], [0.2, 0.8]])
    assert not np.allclose(travel, travel2)


def test_metapop_simulation_identical_from_split_files():
    from generic_core.model_factory import extract_history_full, make_metapop_from_folder

    combined = _load(LEGACY_CONFIGS[2])
    merged, _ = merge_configs(*split_model_config(combined)[:2])
    comps = list(combined["compartments"])
    travel_config = next(
        t["rate_config"]["travel_config"] for t in combined["transitions"]
        if t["rate_template"] == "force_of_infection_travel"
    )

    def _run(cfg):
        m, _ = make_metapop_from_folder(
            EXAMPLE_METAPOP_FOLDER, cfg, "2024-01-01", 10, comps,
            seed_base=0, ts_per_day=1, stochastic=False, save_daily=True,
            num_age_groups=3, num_risk_groups=2, travel_config=travel_config,
        )
        m.simulate_until_day(10)
        return extract_history_full(m, comps)

    a, b = _run(combined), _run(merged)
    for comp in comps:
        for sp in a[comp]:
            np.testing.assert_array_equal(a[comp][sp], b[comp][sp])
