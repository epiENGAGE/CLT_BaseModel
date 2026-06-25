"""
Network-free tests for generic_core.contact_matrix_fetch.

Covers the pure helpers (parse / validate / name resolution / band mappings) and
exercises fetch_contact_matrices with a fake epydemix module injected into
sys.modules, so no package install or network access is needed.
"""

import sys
import types

import numpy as np
import pytest

from generic_core.contact_matrix_fetch import (
    parse_age_bands,
    validate_age_bands,
    age_band_mappings,
    resolve_population_name,
    fetch_contact_matrices,
    fetch_population,
    COUNTRIES,
)


# ---------------------------------------------------------------------------
# parse / validate
# ---------------------------------------------------------------------------

def test_parse_age_bands():
    assert parse_age_bands("0-4, 5-17, 18-49, 50-64, 65+") == \
        ["0-4", "5-17", "18-49", "50-64", "65+"]
    assert parse_age_bands("  ,, 0-4 , ") == ["0-4"]


def test_validate_age_bands_accepts_valid():
    validate_age_bands(["0-4", "5-17", "18-49", "50-64", "65+"])  # no raise
    validate_age_bands(["0-0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+"])
    validate_age_bands(["0+"])  # single band covering the whole population (A=1)


@pytest.mark.parametrize("bands, needle", [
    ([], "non-empty"),
    (["5-17", "18+"], "start at 0"),
    (["0-4", "5-17"], "form 'x+'"),
    (["0-4", "6-17", "18+"], "Gap"),
    (["0-84", "85+"], "greater than 84"),
])
def test_validate_age_bands_rejects(bands, needle):
    with pytest.raises(ValueError) as exc:
        validate_age_bands(bands)
    assert needle in str(exc.value)


# ---------------------------------------------------------------------------
# mappings / name resolution
# ---------------------------------------------------------------------------

def test_age_band_mappings():
    m = age_band_mappings(["0-4", "65+"])
    assert m["0-4"] == ["0", "1", "2", "3", "4"]
    assert m["65+"][0] == "65"
    assert m["65+"][-1] == "84+"


def test_resolve_population_name_us_state():
    assert resolve_population_name("us_state", "New-York") == "United_States__New_York"
    assert resolve_population_name("us_state", "Massachusetts") == "United_States__Massachusetts"


def test_resolve_population_name_country():
    # spaces -> underscores
    assert resolve_population_name("country", "United Kingdom") == "United_Kingdom"
    assert resolve_population_name("country", "Italy") == "Italy"
    # hyphens are KEPT (epydemix-data convention)
    assert resolve_population_name("country", "Guinea-Bissau") == "Guinea-Bissau"
    assert resolve_population_name("country", "Timor-Leste") == "Timor-Leste"
    # apostrophes dropped
    assert resolve_population_name("country", "Cote d'Ivoire") == "Cote_dIvoire"
    # already-canonical names pass through unchanged
    assert resolve_population_name("country", "Bosnia_and_Herzegovina") == "Bosnia_and_Herzegovina"


@pytest.mark.parametrize("kind, name", [("us_state", ""), ("planet", "Mars")])
def test_resolve_population_name_errors(kind, name):
    with pytest.raises(ValueError):
        resolve_population_name(kind, name)


def test_canonical_countries_pass_through_unchanged():
    # COUNTRIES are already canonical epydemix-data names, so resolving them must
    # be a no-op (guards against accidental hyphen-stripping regressions).
    assert COUNTRIES, "COUNTRIES list should not be empty"
    for name in COUNTRIES:
        assert resolve_population_name("country", name) == name


# ---------------------------------------------------------------------------
# fetch_contact_matrices with a fake epydemix module
# ---------------------------------------------------------------------------

def _install_fake_epydemix(monkeypatch, captured):
    """Inject a fake epydemix.population.load_epydemix_population into sys.modules."""
    def _fake_load(population_name, contacts_source, layers, age_group_mapping):
        captured["population_name"] = population_name
        captured["contacts_source"] = contacts_source
        captured["layers"] = list(layers)
        captured["age_group_mapping"] = age_group_mapping
        a = len(age_group_mapping)
        pop = types.SimpleNamespace()
        # distinct matrices per layer so we can tell them apart
        pop.contact_matrices = {
            layer: np.full((a, a), float(i + 1))
            for i, layer in enumerate(layers)
        }
        # per-band population totals (distinct per band)
        pop.Nk = np.array([1000.0 * (j + 1) for j in range(a)], dtype=float)
        pop.Nk_names = list(age_group_mapping.keys())
        return pop

    pkg = types.ModuleType("epydemix")
    sub = types.ModuleType("epydemix.population")
    sub.load_epydemix_population = _fake_load
    pkg.population = sub
    monkeypatch.setitem(sys.modules, "epydemix", pkg)
    monkeypatch.setitem(sys.modules, "epydemix.population", sub)


def test_fetch_contact_matrices_with_fake_epydemix(monkeypatch):
    captured = {}
    _install_fake_epydemix(monkeypatch, captured)

    bands = ["0-4", "5-17", "18-49", "50-64", "65+"]
    out = fetch_contact_matrices("us_state", "Massachusetts", bands)

    assert set(out.keys()) == {
        "total_contact_matrix", "school_contact_matrix", "work_contact_matrix",
    }
    # A×A shape for every matrix
    for mat in out.values():
        arr = np.asarray(mat)
        assert arr.shape == (len(bands), len(bands))

    # population name resolved correctly and passed through
    assert captured["population_name"] == "United_States__Massachusetts"
    assert captured["contacts_source"] == "mistry_2021"
    assert set(captured["layers"]) == {"work", "school", "all"}


def test_fetch_contact_matrices_propagates_band_errors(monkeypatch):
    captured = {}
    _install_fake_epydemix(monkeypatch, captured)
    with pytest.raises(ValueError):
        fetch_contact_matrices("us_state", "Massachusetts", ["5-17", "18+"])


# ---------------------------------------------------------------------------
# fetch_population with a fake epydemix module
# ---------------------------------------------------------------------------

def test_fetch_population_with_fake_epydemix(monkeypatch):
    captured = {}
    _install_fake_epydemix(monkeypatch, captured)

    bands = ["0-4", "5-17", "18-49", "50-64", "65+"]
    pop = fetch_population("us_state", "Massachusetts", bands)

    # one population total per band, in band order, as plain floats
    assert isinstance(pop, list)
    assert len(pop) == len(bands)
    assert pop == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    assert all(isinstance(x, float) for x in pop)

    # geography name resolved and passed through; only the "all" layer is loaded
    assert captured["population_name"] == "United_States__Massachusetts"
    assert captured["contacts_source"] == "mistry_2021"
    assert captured["layers"] == ["all"]


def test_fetch_population_country(monkeypatch):
    captured = {}
    _install_fake_epydemix(monkeypatch, captured)
    pop = fetch_population("country", "United_Kingdom", ["0-17", "18+"])
    assert len(pop) == 2
    assert captured["population_name"] == "United_Kingdom"


def test_fetch_population_propagates_band_errors(monkeypatch):
    captured = {}
    _install_fake_epydemix(monkeypatch, captured)
    with pytest.raises(ValueError):
        fetch_population("us_state", "Massachusetts", ["5-17", "18+"])
