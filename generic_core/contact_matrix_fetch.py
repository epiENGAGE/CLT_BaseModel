"""
contact_matrix_fetch.py — fetch age-structured contact matrices for a geography.

Reusable, importable refactor of MA_vax/download_contact_matrices.py. Given a
geography (US state or country) and a list of named age bands, returns the
total / school / work contact matrices as nested Python lists, ready to drop
into a generic_core model config's contact-matrix params.

Data source: the epydemix-data repo (Mistry 2021 contact matrices), via the
``epydemix`` package. epydemix is an OPTIONAL dependency and is imported lazily
inside fetch_contact_matrices(), so importing this module never requires it.
Install with::

    pip install epydemix

This module is intentionally NOT imported by generic_core/__init__.py, so the
core package has no epydemix dependency.

Pure helpers (parse / validate / name-resolution / mappings) are network-free
and unit-tested in tests/test_contact_matrix_fetch.py.
"""

from __future__ import annotations

# Layer name (epydemix) -> contact-matrix param name (generic_core config).
_SETTING_TO_PARAM = {
    "all": "total_contact_matrix",
    "school": "school_contact_matrix",
    "work": "work_contact_matrix",
}

# US states available from epydemix-data (United_States__<State>). Hyphenated
# multi-word names match the epydemix-data location naming.
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District-of-Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New-Hampshire",
    "New-Jersey", "New-Mexico", "New-York", "North-Carolina", "North-Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode-Island",
    "South-Carolina", "South-Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West-Virginia", "Wisconsin", "Wyoming",
]

# Country-level populations available from epydemix-data, as canonical location
# names (the form passed straight to load_epydemix_population). Mirrors the
# country rows of https://raw.githubusercontent.com/epistorm/epydemix-data/main/locations.csv
# (spaces -> underscores, hyphens kept). May need updating if epydemix-data changes.
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia_and_Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina_Faso", "Burundi", "Cabo_Verde", "Cambodia", "Cameroon", "Canada",
    "Chile", "China", "Colombia", "Congo", "Costa_Rica", "Cote_dIvoire",
    "Croatia", "Cuba", "Cyprus", "Czech_Republic",
    "Democratic_Republic_of_the_Congo", "Denmark", "Dominican_Republic",
    "Egypt", "El_Salvador", "Equatorial_Guinea", "Eritrea", "Estonia",
    "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Greece", "Guatemala", "Guinea", "Guinea-Bissau",
    "Guyana", "Haiti", "Honduras", "Hong_Kong", "Hungary", "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait",
    "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Lithuania", "Luxembourg", "Macao", "Madagascar", "Malawi", "Malaysia",
    "Maldives", "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Moldova",
    "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nepal", "Netherlands", "New_Zealand", "Nicaragua", "Niger",
    "Nigeria", "North_Korea", "North_Macedonia", "Norway", "Oman", "Pakistan",
    "Palestine", "Panama", "Papua_New_Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Puerto_Rico", "Qatar", "Romania",
    "Russia", "Rwanda", "Saint_Lucia", "Saint_Vincent_and_the_Grenadines",
    "Samoa", "Sao_Tome_and_Principe", "Saudi_Arabia", "Senegal", "Serbia",
    "Seychelles", "Sierra_Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon_Islands", "South_Africa", "South_Korea", "South_Sudan", "Spain",
    "Sri_Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
    "Syrian_Arab_Republic", "Taiwan", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad_and_Tobago", "Tunisia", "Turkey",
    "Turkmenistan", "Uganda", "Ukraine", "United_Arab_Emirates",
    "United_Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe",
]


# ---------------------------------------------------------------------------
# Pure helpers (network-free)
# ---------------------------------------------------------------------------

def parse_age_bands(text: str) -> list[str]:
    """Split a comma-separated age-band string into a trimmed list.

    e.g. '0-4, 5-17, 18-49, 50-64, 65+' -> ['0-4','5-17','18-49','50-64','65+'].
    """
    return [item.strip() for item in str(text).split(",") if item.strip()]


def validate_age_bands(age_groups: list[str]) -> None:
    """Validate age-band formatting; raise ValueError with a clear message.

    Rules (mirrors MA_vax/download_contact_matrices.check_age_groups_formatting):
    - non-empty list of strings,
    - first band starts at 0,
    - last band is of the form 'x+' with x <= 84,
    - bands are contiguous (each starts where the previous ends + 1).
    """
    if not isinstance(age_groups, list) or not age_groups:
        raise ValueError("Age groups must be a non-empty list of band strings.")
    if not all(isinstance(x, str) for x in age_groups):
        raise ValueError("All age-group entries must be strings, e.g. '0-4'.")

    first_lower = age_groups[0].split("-")[0]
    if first_lower != "0":
        raise ValueError(
            f"Age groups must start at 0; first band lower bound is {first_lower}."
        )

    last = age_groups[-1]
    if not last.endswith("+"):
        raise ValueError(f"Last band must be of the form 'x+', e.g. '65+', not '{last}'.")
    last_start = int(last.replace("+", ""))
    if last_start > 84:
        raise ValueError("Last age-band start should not be greater than 84.")

    for i in range(1, len(age_groups)):
        if i == len(age_groups) - 1:
            new_start = int(age_groups[i].replace("+", ""))
        else:
            new_start = int(age_groups[i].split("-")[0])
        prev_end = int(age_groups[i - 1].split("-")[-1])
        if prev_end != new_start - 1:
            raise ValueError(
                f"Gap between bands '{age_groups[i - 1]}' and '{age_groups[i]}': "
                "each band must start where the previous one ends + 1."
            )


def age_band_mappings(age_groups: list[str]) -> dict[str, list[str]]:
    """Map each band ('0-4') to the underlying 1-year bin labels (['0',..,'4']).

    The open-ended top band maps to bins up to '84+' (epydemix-data convention).
    """
    mapping: dict[str, list[str]] = {}
    for band in age_groups:
        if band.endswith("+"):
            lower = int(band.replace("+", ""))
            idx = [str(x) for x in range(lower, 85)]
            idx[-1] = "84+"
        else:
            lower = int(band.split("-")[0])
            upper = int(band.split("-")[1])
            idx = [str(x) for x in range(lower, upper + 1)]
        mapping[band] = idx
    return mapping


def resolve_population_name(geography_kind: str, geography_name: str) -> str:
    """Resolve a UI geography selection to an epydemix-data population name.

    Naming conventions (epydemix-data locations.csv):
    - US states:  'United_States__<State>', spaces -> underscores
      (e.g. 'New-York' / 'New York' -> 'United_States__New_York').
    - Countries:  spaces -> underscores, hyphens KEPT, apostrophes dropped
      (e.g. 'United Kingdom' -> 'United_Kingdom', 'Guinea-Bissau' -> 'Guinea-Bissau',
      "Cote d'Ivoire" -> 'Cote_dIvoire'). Already-canonical names pass through.
    """
    name = str(geography_name).strip()
    if not name:
        raise ValueError("Geography name is empty.")
    kind = str(geography_kind).strip().lower()
    if kind in ("us_state", "us state", "state"):
        core = name.replace("-", "_").replace(" ", "_")
        return f"United_States__{core}"
    if kind in ("country",):
        return name.replace("'", "").replace("’", "").replace(" ", "_")
    raise ValueError(f"Unknown geography_kind '{geography_kind}' (use 'us_state' or 'country').")


# ---------------------------------------------------------------------------
# Availability + fetch (needs epydemix + network)
# ---------------------------------------------------------------------------

def epydemix_available() -> bool:
    """Return True if the optional ``epydemix`` package can be imported."""
    try:
        import epydemix  # noqa: F401
        return True
    except Exception:
        return False


def fetch_contact_matrices(
    geography_kind: str,
    geography_name: str,
    age_groups: list[str],
    contacts_source: str = "mistry_2021",
) -> dict[str, list]:
    """Fetch total/school/work contact matrices for a geography and age bands.

    Returns a dict with keys 'total_contact_matrix', 'school_contact_matrix',
    'work_contact_matrix', each an A×A nested list (A = len(age_groups)).

    Raises:
        ImportError: if the optional ``epydemix`` package is not installed.
        ValueError: if the age bands are malformed or the geography is unknown.
        Exception: propagated from epydemix if the geography/data is unavailable.
    """
    try:
        from epydemix.population import load_epydemix_population
    except Exception as exc:  # ImportError or any epydemix import-time failure
        raise ImportError(
            "Fetching contact matrices requires the optional 'epydemix' package. "
            "Install it with: pip install epydemix"
        ) from exc

    validate_age_bands(age_groups)
    population_name = resolve_population_name(geography_kind, geography_name)
    mappings = age_band_mappings(age_groups)

    layers = ["work", "school", "all"]
    population = load_epydemix_population(
        population_name=population_name,
        contacts_source=contacts_source,
        layers=layers,
        age_group_mapping=mappings,
    )

    out: dict[str, list] = {}
    for layer in layers:
        matrix = population.contact_matrices[layer]
        # numpy array -> nested list (config-friendly)
        out[_SETTING_TO_PARAM[layer]] = matrix.tolist()
    return out
