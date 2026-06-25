# _nb_population.py
# Section: Population & Geography tab cells
# Part of model_builder_notebook.py — assembled by build_notebook.py
#
# IMPORTANT: This section must be assembled AFTER _nb_entry.py and BEFORE
# _nb_model_builder.py, because it defines num_age_groups / num_risk_groups /
# is_metapop / metapop_folder_input / age_groups / population_by_subpop /
# pop_subpop_names, which the Model Builder tab consumes.

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
    geo_fetch_button = mo.ui.run_button(label="Fetch contact matrices & population")
    return (
        geo_scope_radio, geo_kind_radio, geo_state_dropdown, geo_country_input,
        geo_subpop_kind, geo_subpop_state, geo_subpop_country, geo_fetch_button,
    )


@app.cell
def _pop_source_ui(mo, num_risk_groups, loaded_config):
    # Population-source widgets: either fetch per-age-band totals for the
    # geography (epydemix), or load a CSV of population per age/risk/subpop.
    population_source_radio = mo.ui.radio(
        options=["Fetch from geography", "CSV file"],
        value="Fetch from geography",
        label="Population source",
    )
    _saved_rf = (loaded_config.get("age_risk", {}) or {}).get("risk_group_fractions")
    if not isinstance(_saved_rf, list) or len(_saved_rf) != int(num_risk_groups):
        _saved_rf = [1.0 / max(int(num_risk_groups), 1)] * int(num_risk_groups)
    # One fraction per risk group; the fetched (age-only) population is split
    # across risk groups by these fractions (renormalised to sum to 1).
    risk_fraction_inputs = mo.ui.array([
        mo.ui.number(start=0.0, stop=1.0, step=None, value=float(_saved_rf[_r]),
                     label=f"risk {_r}")
        for _r in range(int(num_risk_groups))
    ])
    population_csv_input = mo.ui.text(
        value="",
        placeholder="/path/to/population.csv",
        label="Population CSV (columns: age, population, [risk], [subpopulation])",
        full_width=True,
    )
    return population_source_radio, risk_fraction_inputs, population_csv_input


@app.cell
def _geo_fetch(
    mo, cmf,
    geo_fetch_button,
    age_group_mode, age_groups, num_age_groups,
    is_metapop, geo_scope_radio,
    geo_kind_radio, geo_state_dropdown, geo_country_input,
    geo_subpop_names, geo_subpop_kind, geo_subpop_state, geo_subpop_country,
    population_source_radio,
    set_fetched_matrices,
):
    # Only fetch when the button is pressed and named bands are defined.
    mo.stop(not geo_fetch_button.value, None)

    def _kind_geo(kind_radio, state_dd, country_txt):
        if kind_radio.value == "US state":
            return "us_state", state_dd.value
        return "country", country_txt.value.strip()

    # With a single age group there are no named bands to define; '0+' covers
    # the whole population in one band, which is all fetch_* needs for A=1.
    if age_group_mode == "Named age bands":
        _eff_age_groups = age_groups
    elif num_age_groups == 1:
        _eff_age_groups = ["0+"]
    else:
        _eff_age_groups = None

    if not _eff_age_groups:
        set_fetched_matrices({
            "matrices": {}, "populations": {}, "scope": "shared",
            "errors": {"error": "Define named age bands before fetching contact matrices."},
        })
    else:
        _per_subpop = is_metapop and geo_scope_radio.value == "Per-subpopulation"
        _fetch_pop = population_source_radio.value == "Fetch from geography"
        _results, _pops, _errors = {}, {}, {}
        try:
            if _per_subpop:
                for _i, _name in enumerate(geo_subpop_names):
                    _kind, _geo = _kind_geo(
                        geo_subpop_kind[_i], geo_subpop_state[_i], geo_subpop_country[_i]
                    )
                    _results[_name] = cmf.fetch_contact_matrices(_kind, _geo, _eff_age_groups)
                    if _fetch_pop:
                        _pops[_name] = cmf.fetch_population(_kind, _geo, _eff_age_groups)
            else:
                _kind, _geo = _kind_geo(geo_kind_radio, geo_state_dropdown, geo_country_input)
                _results["__shared__"] = cmf.fetch_contact_matrices(_kind, _geo, _eff_age_groups)
                if _fetch_pop:
                    _pops["__shared__"] = cmf.fetch_population(_kind, _geo, _eff_age_groups)
        except Exception as _exc:
            _errors["error"] = str(_exc)

        set_fetched_matrices({
            "matrices": _results,
            "populations": _pops,
            "scope": "per_subpop" if _per_subpop else "shared",
            "errors": _errors,
        })
    return


@app.cell
def _geo_result(get_fetched_matrices):
    _state = get_fetched_matrices() or {}
    fetched_contact_matrices = _state.get("matrices", {})
    fetched_populations = _state.get("populations", {})
    fetched_matrices_scope = _state.get("scope", "shared")
    fetched_matrices_errors = _state.get("errors", {})
    return (
        fetched_contact_matrices, fetched_populations,
        fetched_matrices_scope, fetched_matrices_errors,
    )


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

    if age_group_mode != "Named age bands" and num_age_groups != 1:
        mo.stop(True, mo.vstack([
            *_parts,
            mo.callout(
                mo.md("Switch to **Named age bands** above to fetch contact matrices "
                      "for a geography. In count-only mode with A > 1, provide "
                      "contact-matrix CSVs in Model Builder → Step 4 instead."),
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


@app.cell
def _population_data(
    population_source_radio, risk_fraction_inputs, population_csv_input,
    fetched_populations, fetched_matrices_scope,
    is_metapop, geo_subpop_names,
    num_age_groups, num_risk_groups, age_groups,
    loaded_config, load_population_csv, np,
):
    # Resolve the per-subpopulation population into A×R arrays. Not gated on the
    # active tab so population_by_subpop stays available to the Model Builder
    # tab (initial conditions) and the run paths.
    _A = int(num_age_groups)
    _R = int(num_risk_groups)
    pop_subpop_names = (
        list(geo_subpop_names) if (is_metapop and geo_subpop_names) else ["aggregate_pop"]
    )
    population_source = population_source_radio.value
    population_by_subpop = {}
    population_errors = {}

    # Risk-group split fractions (renormalised; uniform fallback).
    _rf = np.array([float(_x) for _x in risk_fraction_inputs.value], dtype=float)
    if _rf.size != _R or _rf.sum() <= 0:
        _rf = np.full(_R, 1.0 / max(_R, 1))
    _rf = _rf / _rf.sum()

    if population_source == "CSV file":
        _pop, _err = load_population_csv(
            population_csv_input.value, pop_subpop_names, _A, _R, age_groups,
        )
        if _err:
            population_errors["error"] = _err
        elif _pop:
            population_by_subpop = _pop
    else:  # Fetch from geography
        if not fetched_populations:
            population_errors["info"] = (
                "No population fetched yet — choose a geography and press "
                "**Fetch contact matrices & population** above."
            )
        elif fetched_matrices_scope == "per_subpop":
            for _name in pop_subpop_names:
                _nk = fetched_populations.get(_name)
                if _nk:
                    population_by_subpop[_name] = np.round(
                        np.outer(np.asarray(_nk, dtype=float), _rf)
                    )
        else:
            _nk = fetched_populations.get("__shared__")
            if _nk:
                _arr = np.round(np.outer(np.asarray(_nk, dtype=float), _rf))
                for _name in pop_subpop_names:
                    population_by_subpop[_name] = _arr

    # Fallback for any subpop without a resolved population: reuse a saved value
    # from config (round-trip), else split total_population uniformly across cells.
    _saved_ic = loaded_config.get("initial_conditions", {}) or {}
    for _name in pop_subpop_names:
        if _name in population_by_subpop:
            continue
        _saved_pop = (_saved_ic.get(_name, {}) or {}).get("population")
        _arr = None
        if isinstance(_saved_pop, list):
            try:
                _cand = np.asarray(_saved_pop, dtype=float)
                if _cand.shape == (_A, _R):
                    _arr = _cand
            except Exception:
                _arr = None
        if _arr is None:
            _total = float(loaded_config.get("total_population", 10000))
            _arr = np.full((_A, _R), _total / max(_A * _R, 1))
        population_by_subpop[_name] = _arr
    return population_by_subpop, population_source, population_errors, pop_subpop_names


@app.cell
def _population_show(
    mo, main_tab,
    population_source_radio, risk_fraction_inputs, population_csv_input,
    num_risk_groups, age_groups, num_age_groups,
    population_by_subpop, population_source, population_errors, pop_subpop_names,
    param_grid_columns, pd,
):
    mo.stop(main_tab.value != "Population & Geography", None)

    import html as _html
    import random as _random

    def _tip_label(label_text, tip_text):
        """Render a field label with an inline ⓘ hover tooltip using CSS only."""
        _uid = _random.randint(10**7, 10**8 - 1)
        _esc = _html.escape(tip_text)
        return mo.Html(
            f"<style>"
            f"#tip{_uid}{{position:relative;display:inline-block;"
            f"cursor:help;color:#888;font-size:0.8em;vertical-align:middle;}}"
            f"#tip{_uid}>span{{visibility:hidden;opacity:0;"
            f"transition:opacity .15s;transition-delay:.2s;"
            f"position:absolute;bottom:120%;left:0;"
            f"background:#222;color:#fff;border-radius:4px;"
            f"padding:6px 10px;width:280px;font-size:12px;line-height:1.5;"
            f"white-space:pre-wrap;pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            f"</style>"
            f"<span>"
            f"{label_text}&nbsp;"
            f'<span id="tip{_uid}">ⓘ<span>{_esc}</span></span>'
            f"</span>"
        )

    _CSV_FORMAT_TIP = (
        "Required columns: age, population\n\n"
        "  age — named band (matching the configured\n"
        "        age groups) or a 0-based index\n"
        "  population — count for that row\n\n"
        "Optional columns:\n"
        "  risk — 0-based index in 0..R-1\n"
        "        (required only if there is more\n"
        "        than one risk group)\n"
        "  subpopulation — must match a configured\n"
        "        subpopulation name (required only\n"
        "        if there is more than one)"
    )

    _parts = [
        mo.md("## Population sizes (per age / risk group)"),
        mo.md(
            "Population totals per age group are fetched for the chosen geography "
            "(US states and countries supported via epydemix), or loaded from a CSV "
            "for custom / per-subpopulation populations."
        ),
        population_source_radio,
    ]
    if population_source == "Fetch from geography":
        if int(num_risk_groups) > 1:
            _parts.append(mo.md(
                "**Risk-group split** — the fetched (age-only) population is split "
                "across risk groups by these fractions (renormalised to sum to 1):"
            ))
            _parts.append(mo.hstack(list(risk_fraction_inputs), justify="start"))
    else:
        _parts.append(
            mo.hstack(
                [population_csv_input, _tip_label("", _CSV_FORMAT_TIP)],
                justify="start", align="center", gap=0.5,
            )
        )

    if population_errors.get("error"):
        _parts.append(mo.callout(mo.md(f"**Population error:** {population_errors['error']}"),
                                 kind="danger"))
    elif population_errors.get("info"):
        _parts.append(mo.callout(mo.md(population_errors["info"]), kind="info"))

    _cols = param_grid_columns(age_groups, int(num_age_groups))
    for _name in pop_subpop_names:
        _arr = population_by_subpop.get(_name)
        if _arr is None:
            continue
        _label = "Population" if _name == "aggregate_pop" else f"Population — {_name}"
        _df = pd.DataFrame(
            [{"risk_group": _r, **{_c: _arr[_a, _r] for _a, _c in enumerate(_cols)}}
             for _r in range(_arr.shape[1])]
        )
        _parts.append(mo.md(f"**{_label}** (total {_arr.sum():,.0f})"))
        _parts.append(mo.ui.table(_df, selection=None))

    mo.vstack(_parts)
    return

