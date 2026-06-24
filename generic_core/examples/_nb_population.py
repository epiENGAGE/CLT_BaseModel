# _nb_population.py
# Section: Population & Geography tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py
#
# IMPORTANT: This section must be assembled AFTER _nb_entry.py and BEFORE
# _nb_model_builder.py, because it defines num_age_groups / num_risk_groups /
# is_metapop / metapop_folder_input / age_groups, which the Model Builder tab
# consumes.
#
# The compute cell (_population_structure_compute) is intentionally NOT gated on
# the active tab, so the population variables it exports stay available to every
# downstream tab regardless of which tab is selected.

@app.cell
def _population_structure_ui(mo, loaded_config):
    _ar = loaded_config.get("age_risk", {})
    _inf = loaded_config.get("input_files", {})
    _saved_bands = _ar.get("age_groups") or []

    age_group_mode_radio = mo.ui.radio(
        options=["Count only", "Named age bands"],
        value="Named age bands" if _saved_bands else "Count only",
        label="Age-group specification",
    )
    num_age_groups_input = mo.ui.number(
        start=1, stop=20, step=1,
        value=int(_ar.get("num_age_groups", 1)),
        label="Number of age groups (A)",
    )
    age_bands_input = mo.ui.text(
        value=", ".join(_saved_bands),
        placeholder="0-4, 5-17, 18-49, 50-64, 65+",
        label="Age bands (comma-separated, 0-based, contiguous, last 'x+')",
        full_width=True,
    )
    num_risk_groups_input = mo.ui.number(
        start=1, stop=10, step=1,
        value=int(_ar.get("num_risk_groups", 1)),
        label="Number of risk groups (R)",
    )
    _metapop_folder_saved = _inf.get("metapop_folder", "")
    pop_mode_radio = mo.ui.radio(
        options=["Single population", "Metapopulation"],
        value="Metapopulation" if _metapop_folder_saved else "Single population",
        label="Population mode",
    )
    metapop_folder_input = mo.ui.text(
        value=_metapop_folder_saved,
        placeholder="/path/to/metapop_folder/",
        label="Metapopulation folder path",
        full_width=True,
    )
    return (
        age_group_mode_radio,
        num_age_groups_input,
        age_bands_input,
        num_risk_groups_input,
        pop_mode_radio,
        metapop_folder_input,
    )


@app.cell
def _population_structure_compute(
    age_group_mode_radio,
    num_age_groups_input,
    age_bands_input,
    num_risk_groups_input,
    pop_mode_radio,
    cmf,
):
    # In band mode, A is the number of named bands; in count mode it's the number
    # input. age_groups is the band list (or None when no bands are defined).
    _use_bands = age_group_mode_radio.value == "Named age bands"
    if _use_bands:
        age_groups = cmf.parse_age_bands(age_bands_input.value)
        num_age_groups = max(len(age_groups), 1)
    else:
        age_groups = None
        num_age_groups = int(num_age_groups_input.value)

    num_risk_groups = int(num_risk_groups_input.value)
    is_metapop = pop_mode_radio.value == "Metapopulation"
    age_group_mode = age_group_mode_radio.value
    return (num_age_groups, num_risk_groups, is_metapop, age_groups, age_group_mode)


@app.cell
def _population_structure_show(
    mo,
    main_tab,
    num_age_groups,
    num_risk_groups,
    is_metapop,
    age_groups,
    age_group_mode,
    age_group_mode_radio,
    num_age_groups_input,
    age_bands_input,
    num_risk_groups_input,
    pop_mode_radio,
    metapop_folder_input,
    validate_metapop_folder,
    cmf,
):
    mo.stop(main_tab.value != "Population & Geography", None)

    _parts = [
        mo.md("## Population Structure"),
        mo.md(
            "Define the population dimensions and geography here. The rest of the "
            "model (compartments, transitions, parameters, …) is built in the "
            "**Model Builder** tab."
        ),
        age_group_mode_radio,
    ]

    if age_group_mode == "Named age bands":
        _parts.append(age_bands_input)
        try:
            cmf.validate_age_bands(age_groups or [])
            _parts.append(mo.callout(
                mo.md(f"Parsed **A = {num_age_groups}** age bands: "
                      + ", ".join(f"`{_b}`" for _b in (age_groups or []))
                      + ".\n\nNamed bands enable contact-matrix fetching below."),
                kind="success",
            ))
        except ValueError as _exc:
            _parts.append(mo.callout(mo.md(f"**Age bands:** {_exc}"), kind="danger"))
    else:
        _parts.append(num_age_groups_input)
        if num_age_groups > 1:
            _parts.append(mo.callout(
                mo.md(
                    "Count-only mode: contact matrices cannot be fetched (the "
                    "fetcher needs age-band definitions). For A > 1, either switch "
                    "to **Named age bands** to fetch them, or supply contact-matrix "
                    "CSVs in Model Builder → Step 4."
                ),
                kind="info",
            ))

    _parts.append(num_risk_groups_input)
    _parts.append(pop_mode_radio)

    if is_metapop:
        _parts.append(metapop_folder_input)
        _folder_valid, _folder_status = validate_metapop_folder(metapop_folder_input.value)
        if metapop_folder_input.value.strip():
            _lines = [f"- **{_fname}**: {_msg}" for _fname, _msg in _folder_status.items()]
            _overall_kind = "success" if _folder_valid else "danger"
            _parts.append(mo.callout(mo.md("\n".join(_lines)), kind=_overall_kind))
    else:
        if num_age_groups > 1 or num_risk_groups > 1:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model: A={num_age_groups}, R={num_risk_groups}. "
                    "Use CSV file paths in Model Builder → Step 4 for schedule data."
                ),
                kind="info",
            ))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Contact-matrix geography (fetch via epydemix)
# ---------------------------------------------------------------------------

@app.cell
def _geo_fetch_state(mo):
    # Holds the most recent fetch result:
    #   {"matrices": {scope_key: {param: A×A}}, "scope": "shared"|"per_subpop",
    #    "errors": {...}}
    # scope_key is "__shared__" for one geography, or a subpop name otherwise.
    get_fetched_matrices, set_fetched_matrices = mo.state({})
    return get_fetched_matrices, set_fetched_matrices


@app.cell
def _geo_subpop_names(is_metapop, metapop_folder_input, Path, json):
    # Subpop names for per-subpop geography, read from metapop_config.json.
    geo_subpop_names = []
    if is_metapop and metapop_folder_input.value.strip():
        _cfg_path = Path(metapop_folder_input.value.strip()) / "metapop_config.json"
        if _cfg_path.exists():
            try:
                with open(_cfg_path) as _f:
                    _mc = json.load(_f)
                _sp = _mc.get("subpopulations")
                if isinstance(_sp, list):
                    geo_subpop_names = [str(_s) for _s in _sp]
            except Exception:
                geo_subpop_names = []
    return (geo_subpop_names,)


@app.cell
def _geo_ui(mo, cmf, geo_subpop_names):
    geo_scope_radio = mo.ui.radio(
        options=["Same for all subpops", "Per-subpopulation"],
        value="Same for all subpops",
        label="Contact-matrix geography scope (metapop)",
    )
    geo_kind_radio = mo.ui.radio(
        options=["US state", "Country"], value="US state", label="Geography type",
    )
    geo_state_dropdown = mo.ui.dropdown(
        options=cmf.US_STATES, value="Massachusetts", label="US state", searchable=True,
    )
    geo_country_input = mo.ui.dropdown(
        options=cmf.COUNTRIES, value="United_Kingdom",
        label="Country (epydemix-data name)", searchable=True,
    )
    # Per-subpop selectors (used only in metapop + per-subpopulation scope).
    geo_subpop_kind = mo.ui.array([
        mo.ui.radio(options=["US state", "Country"], value="US state", label=f"{_n}: type")
        for _n in geo_subpop_names
    ])
    geo_subpop_state = mo.ui.array([
        mo.ui.dropdown(options=cmf.US_STATES, value="Massachusetts",
                       label=f"{_n}: US state", searchable=True)
        for _n in geo_subpop_names
    ])
    geo_subpop_country = mo.ui.array([
        mo.ui.dropdown(options=cmf.COUNTRIES, value="United_Kingdom",
                       label=f"{_n}: country", searchable=True)
        for _n in geo_subpop_names
    ])
    geo_fetch_button = mo.ui.run_button(label="Fetch contact matrices")
    return (
        geo_scope_radio, geo_kind_radio, geo_state_dropdown, geo_country_input,
        geo_subpop_kind, geo_subpop_state, geo_subpop_country, geo_fetch_button,
    )


@app.cell
def _geo_fetch(
    mo, cmf,
    geo_fetch_button,
    age_group_mode, age_groups,
    is_metapop, geo_scope_radio,
    geo_kind_radio, geo_state_dropdown, geo_country_input,
    geo_subpop_names, geo_subpop_kind, geo_subpop_state, geo_subpop_country,
    set_fetched_matrices,
):
    # Only fetch when the button is pressed and named bands are defined.
    mo.stop(not geo_fetch_button.value, None)

    def _kind_geo(kind_radio, state_dd, country_txt):
        if kind_radio.value == "US state":
            return "us_state", state_dd.value
        return "country", country_txt.value.strip()

    if age_group_mode != "Named age bands" or not age_groups:
        set_fetched_matrices({
            "matrices": {}, "scope": "shared",
            "errors": {"error": "Define named age bands before fetching contact matrices."},
        })
    else:
        _per_subpop = is_metapop and geo_scope_radio.value == "Per-subpopulation"
        _results, _errors = {}, {}
        try:
            if _per_subpop:
                for _i, _name in enumerate(geo_subpop_names):
                    _kind, _geo = _kind_geo(
                        geo_subpop_kind[_i], geo_subpop_state[_i], geo_subpop_country[_i]
                    )
                    _results[_name] = cmf.fetch_contact_matrices(_kind, _geo, age_groups)
            else:
                _kind, _geo = _kind_geo(geo_kind_radio, geo_state_dropdown, geo_country_input)
                _results["__shared__"] = cmf.fetch_contact_matrices(_kind, _geo, age_groups)
        except Exception as _exc:
            _errors["error"] = str(_exc)

        set_fetched_matrices({
            "matrices": _results,
            "scope": "per_subpop" if _per_subpop else "shared",
            "errors": _errors,
        })
    return


@app.cell
def _geo_result(get_fetched_matrices):
    _state = get_fetched_matrices() or {}
    fetched_contact_matrices = _state.get("matrices", {})
    fetched_matrices_scope = _state.get("scope", "shared")
    fetched_matrices_errors = _state.get("errors", {})
    return fetched_contact_matrices, fetched_matrices_scope, fetched_matrices_errors


@app.cell
def _geo_show(
    mo, main_tab, cmf,
    age_group_mode, num_age_groups, is_metapop,
    geo_scope_radio, geo_kind_radio, geo_state_dropdown, geo_country_input,
    geo_subpop_names, geo_subpop_kind, geo_subpop_state, geo_subpop_country,
    geo_fetch_button,
    fetched_contact_matrices, fetched_matrices_scope, fetched_matrices_errors,
):
    mo.stop(main_tab.value != "Population & Geography", None)

    _parts = [mo.md("## Contact Matrices (geography)")]

    if age_group_mode != "Named age bands":
        mo.stop(True, mo.vstack([
            *_parts,
            mo.callout(
                mo.md("Switch to **Named age bands** above to fetch contact matrices "
                      "for a geography. In count-only mode, provide contact-matrix CSVs "
                      "in Model Builder → Step 4 instead."),
                kind="info",
            ),
        ]))

    if not cmf.epydemix_available():
        _parts.append(mo.callout(
            mo.md("The optional **epydemix** package is not installed, so live fetching "
                  "is unavailable. Install it with `pip install epydemix`, or supply "
                  "contact-matrix CSVs in Model Builder → Step 4."),
            kind="warn",
        ))

    _parts.append(mo.md(
        f"Fetch the **total / school / work** {num_age_groups}×{num_age_groups} contact "
        "matrices (Mistry 2021, via epydemix-data) for your age bands."
    ))

    if is_metapop:
        _parts.append(geo_scope_radio)

    if is_metapop and geo_scope_radio.value == "Per-subpopulation":
        if not geo_subpop_names:
            _parts.append(mo.callout(
                mo.md("No subpopulations found — set a valid metapop folder above."),
                kind="warn",
            ))
        for _i, _name in enumerate(geo_subpop_names):
            _sel = (geo_subpop_state[_i] if geo_subpop_kind[_i].value == "US state"
                    else geo_subpop_country[_i])
            _parts.append(mo.hstack([mo.md(f"**{_name}**"), geo_subpop_kind[_i], _sel],
                                    justify="start"))
    else:
        _parts.append(geo_kind_radio)
        _parts.append(geo_state_dropdown if geo_kind_radio.value == "US state"
                      else geo_country_input)

    _parts.append(geo_fetch_button)

    if fetched_matrices_errors.get("error"):
        _parts.append(mo.callout(mo.md(f"**Fetch failed:** {fetched_matrices_errors['error']}"),
                                 kind="danger"))
    elif fetched_contact_matrices:
        _keys = ", ".join(
            "all subpops" if _k == "__shared__" else _k for _k in fetched_contact_matrices
        )
        _parts.append(mo.callout(
            mo.md(f"Fetched contact matrices ({fetched_matrices_scope}) for: {_keys}. "
                  "They are written into the config and used at run time."),
            kind="success",
        ))

    mo.vstack(_parts)
    return
