# _nb_model_builder.py
# Section: Model Builder tab cells (Steps 0-10)
# Part of model_builder_notebook.py — assembled by build_notebook.py

@app.cell
def _load_config_state(mo, Path):
    # Default to the example config that ships alongside this notebook, resolved
    # relative to the notebook file so it works on any machine. Falls back to an
    # empty string if the bundled example cannot be located.
    try:
        _default_config_path = str(
            Path(__file__).parent / "example_metapop_inputs" / "model_config.json"
        )
    except NameError:
        _default_config_path = ""

    get_config_path, set_config_path = mo.state(_default_config_path)
    return get_config_path, set_config_path


@app.cell
def _config_file_upload_ui(mo):
    # Kept in its own cell (independent of the path state) so that clearing the
    # path text box does not recreate this widget and wipe the browsed file.
    config_file_upload = mo.ui.file(
        filetypes=[".json"],
        label="Browse for config JSON",
    )
    return (config_file_upload,)


@app.cell
def _load_config_ui(mo, get_config_path, set_config_path):
    config_path_input = mo.ui.text(
        value=get_config_path(),
        on_change=set_config_path,
        placeholder="/path/to/model_config.json  (or use Browse above)",
        label="Or enter config JSON path directly",
        full_width=True,
    )
    return (config_path_input,)


@app.cell
def _clear_path_on_browse(config_file_upload, set_config_path):
    # When a file is browsed via the OS picker, clear the manual-path text box.
    # mo.ui.file can't expose the real filesystem path (browsers withhold it),
    # so we empty the box; the browsed file's name is shown next to Browse and
    # the parse cell already prioritizes the browsed file over the text path.
    if config_file_upload.value:
        set_config_path("")
    return


@app.cell
def _clear_config_button_ui(mo, set_config_path):
    clear_config_button = mo.ui.button(
        label="Clear config",
        on_click=lambda _: set_config_path(""),
    )
    return (clear_config_button,)


@app.cell
def _load_config_parse(config_file_upload, config_path_input, load_config_json, json):
    _loaded_config = {}
    _cfg_err = None
    _source = None

    if config_file_upload.value:
        _file = config_file_upload.value[0]
        try:
            _loaded_config = json.loads(_file.contents.decode("utf-8"))
        except Exception as _exc:
            _cfg_err = f"JSON parse error: {_exc}"
        _source = f"Browsed: **{_file.name}**"
    elif config_path_input.value.strip():
        _loaded_config, _cfg_err = load_config_json(config_path_input.value)
        _source = "path"

    loaded_config = _loaded_config
    return (loaded_config,)


@app.cell
def _load_config_display(
    config_file_upload, config_path_input, clear_config_button,
    loaded_config, load_config_json, mo, main_tab
):
    mo.stop(main_tab.value != "Model Builder", None)

    _cfg_err = None
    _source = None
    if config_file_upload.value:
        _source = f"Browsed: **{config_file_upload.value[0].name}**"
        try:
            import json as _json
            _json.loads(config_file_upload.value[0].contents.decode("utf-8"))
        except Exception as _exc:
            _cfg_err = f"JSON parse error: {_exc}"
    elif config_path_input.value.strip():
        _, _cfg_err = load_config_json(config_path_input.value)
        _source = "path"

    _browse_row = mo.hstack([config_file_upload] + (
        [mo.md(f"Selected: `{config_file_upload.value[0].name}`")]
        if config_file_upload.value else []
    ), align="center", gap=1)

    _parts = [
        mo.md("### Step 0 — Load Existing Config"),
        _browse_row,
        config_path_input,
        clear_config_button,
    ]
    if _source:
        if _cfg_err:
            _parts.append(mo.callout(mo.md(f"**Load error:** {_cfg_err}"), kind="danger"))
        else:
            _n_comp = len(loaded_config.get("compartments", []))
            _n_tr = len(loaded_config.get("transitions", []))
            _ar = loaded_config.get("age_risk", {})
            _A = _ar.get("num_age_groups", 1)
            _R = _ar.get("num_risk_groups", 1)
            if _source == "path":
                _label = f"Loaded from `{config_path_input.value.strip()}`"
            else:
                _label = _source
            _parts.append(mo.callout(
                mo.md(
                    f"{_label} — **{_n_comp}** compartments, **{_n_tr}** transitions, "
                    f"**{_A}** age group(s), **{_R}** risk group(s)."
                ),
                kind="success",
            ))
    else:
        _parts.append(mo.callout(
            mo.md("No config loaded — all fields below use their defaults. Enter a path or browse for a JSON file to pre-populate the form, or fill it in manually."),
            kind="info",
        ))
    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------


@app.cell
def _intro(mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.md(
        """
        # Generic Epidemic Model Builder

        Build, visualise, and run a config-driven epidemic model without editing JSON.
        Supports all rate templates, configurable age/risk groups, CSV-backed schedules,
        and multi-subpopulation (metapopulation) models.

        **Quick start:** set up the **Population & Geography** tab first, then work through
        Steps 1–9 here in order and press **Run simulation**.
        Load a previously saved config in **Step 0** to restore any prior setup.
        """
    )
    return


@app.cell
def _instructions(mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.accordion({
        "📋 Workflow overview": mo.md("""
**Population & Geography tab** *(do this first)*
Choose the number of age groups (A) and risk groups (R), single-population vs.
metapopulation mode, and (when using named age bands) fetch contact matrices for a
chosen geography. For metapop, enter the path to a folder containing the required input
files (see the *Metapopulation folder conventions* section below).

**Step 0 — Load existing config** *(optional)*
Enter the path to a `model_config.json` file to pre-populate all fields below.
Leave blank to start fresh.

**Step 1 — Compartments**
Enter compartment names as a comma-separated list, e.g. `S, E, I, R`.

**Step 2 — Transitions**
Define each transition: origin compartment → destination, and the rate template.
Available templates:
- `constant_param` — fixed rate from a single parameter
- `param_product` — product of multiple parameters (with optional complement factors)
- `immunity_modulated` — base rate adjusted by infection/vaccine immunity (M, MV)
- `force_of_infection` — standard FOI with contact matrix and optional humidity/immunity
- `force_of_infection_travel` — FOI with inter-subpop travel mixing (metapop only)

**Step 3 — Parameters**
Numeric sliders appear automatically for every parameter name referenced by your transitions.

**Step 4 — Schedules and Immunity**
For rate templates that use schedules (humidity, mobility, vaccines):
- Choose *constant* to use a single scalar value for the whole simulation.
- Choose *csv* to load a real time-varying schedule from a CSV file.
Contact matrices (total, school, work) are always stored as inline arrays in the
config JSON. When A > 1, supply them via CSV paths in the contact matrix fields, fetch
them in the Population & Geography tab, or load a saved config that already has them
embedded. Risk groups (R > 1) affect transition and susceptibility parameters but do not
require separate contact matrices.

**Step 5 — Model diagram**
Auto-generated from your compartments and transitions. Requires `graphviz`; falls back
to a simple matplotlib diagram if not installed.

**Step 6 — Initial conditions**
Seed each compartment by age and risk group (absolute counts) in an editable table;
the first compartment receives the remaining population per cell. Population totals
come from the **Population & Geography** tab (fetched per age group, split across risk
groups, or loaded from a CSV). In metapopulation mode, pick a subpopulation to edit its
table — these override `initial_conditions_{name}.json` in the metapop folder, which is
used as a fallback for any subpop left without seeds.

**Step 7 — Simulation settings**
Days, deterministic vs. stochastic, number of replicates, RNG seed, timesteps per day.

**Step 8 — Config preview and download**
The full config JSON (including file paths and age/risk group settings) is shown and
can be downloaded. The downloaded file can be re-loaded in Step 0.

**Step 9 — Run**
Press the *Run simulation* button. Results appear as epidemic curves and a summary table.
        """),

        "📁 Schedule CSV formats": mo.md("""
All CSV files should have a header row. Index columns (unnamed first column) are
ignored automatically.

**`absolute_humidity.csv`** — shared across subpops
```
date,absolute_humidity
2024-01-01,0.0043
2024-01-02,0.0041
```

**`school_work_calendar_{name}.csv`** — per-subpop (or shared)
```
date,is_school_day,is_work_day
2024-01-01,0.0,0.0
2024-01-02,1.0,1.0
```
Values are floats in [0, 1] (fractional school/work day allowed).

**`mobility_modifier.csv`** — shared, day-of-week indexed, JSON A×R array per row
```
day_of_week,mobility_modifier
Monday,"[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]"
Tuesday,"[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]"
```
The JSON array shape must be A rows × R columns.

**`vaccines_{name}.csv`** — per-subpop, date-indexed, JSON A×R array per row

Each value is the **proportion** of that age×risk group vaccinated on that day
(i.e. daily count ÷ group population), not a raw count.
```
date,daily_vaccines
2024-01-01,"[[0.000417, 0.000667], [0.000288, 0.000615], [0.001563, 0.003]]"
```

**Contact matrix CSVs** — plain floats, A×A, no header row, no index column *(optional)*
```
7.0,3.0,0.5
3.0,9.0,1.5
0.5,1.5,4.0
```
Separate files for total, school, and work contact matrices. When not provided,
the matrices embedded in the loaded config JSON are used directly.
        """),

        "🗂️ Metapopulation folder conventions": mo.md("""
Create a folder with files following these naming conventions.
The folder path is entered in the **Population & Geography** tab and saved in the config JSON.

**Required files:**

| File | Description |
|---|---|
| `metapop_config.json` | `subpopulations` (ordered list of names) and `travel_matrix` (N×N list of lists, rows sum to 1) |

**Optional shared files** (used by all subpops if present):

| File | Description |
|---|---|
| `absolute_humidity.csv` | `date`, `absolute_humidity` |
| `mobility_modifier.csv` | `day_of_week`, `mobility_modifier` (JSON A×R) |

**Optional per-subpop files** (`{name}` = a name from `metapop_config.json → subpopulations`):

| File | Description |
|---|---|
| `school_work_calendar_{name}.csv` | `date`, `is_school_day`, `is_work_day` |
| `vaccines_{name}.csv` | `date`, `daily_vaccines` (JSON A×R) |
| `initial_conditions_{name}.json` | `compartments` and `epi_metrics` keys, each mapping name → A×R list |

**Per-subpopulation parameter overrides** (`subpop_params` in `model_config.json`):

Any parameter in the `params` block can be overridden on a per-subpop basis by adding a
`subpop_params` section to `model_config.json`. Each key is a subpopulation name
(matching `metapop_config.json → subpopulations`); each value is a dict of parameter
overrides applied only to that subpopulation. Scalar and A×R array values are both supported.
Parameters not listed under a subpop continue to use the shared value from `params`.

```json
"subpop_params": {
  "East": { "beta_baseline": 0.050 },
  "West": {
    "beta_baseline": 0.038,
    "IP_to_ISH_prop": [[0.008], [0.003], [0.007], [0.012], [0.100]]
  }
}
```

Example `initial_conditions_West.json`:
```json
{
  "compartments": {
    "S": [[31680], [96589], [344716], [116909], [87681]],
    "E": [[0], [0], [30], [0], [0]]
  },
  "epi_metrics": {
    "M":  [[0.1], [0.1], [0.06], [0.08], [0.04]],
    "MV": [[0.0], [0.0], [0.0],  [0.0],  [0.0]]
  }
}
```

Per-subpop initial conditions seeded in the **Step 6** tables take precedence over this
file. The file is used as a fallback for any subpopulation left without seeds in Step 6.
If neither is present, all compartments are initialised to zero and the simulation will
stop with an error when the model has more than one age or risk group.

**Example folder** is included in the repository at
`generic_core/examples/example_metapop_inputs/` (2 subpops, 3 age groups, 2 risk groups,
SEIR model). Re-generate it with::

    python generic_core/examples/generate_example_metapop_data.py
        """),

        "💾 Config save / load round-trip": mo.md("""
The downloaded `model_config.json` contains everything needed to restore the session:

- Compartment names, transition definitions, parameter values
- Age/risk group counts (`age_risk` section)
- CSV schedule files (`input_files` section): a shared `input_folder` plus the
  filename of each CSV used (humidity, calendar, mobility, vaccines, contact matrices)
- Initial conditions (`initial_conditions` section): per-subpopulation population (A×R)
  and per-compartment seed counts, plus the derived `total_population`
- Per-subpopulation parameter overrides (`subpop_params` section, if present)

**To reload:** paste the path into the Step 0 text field. All UI fields (compartments,
transitions, parameters, immunity toggles, file paths, metapop folder) will be
pre-populated automatically.

**Note on contact matrices:** When A > 1, matrix values *are* embedded inline in the
config JSON under `params` (as nested lists). CSV file paths are optional — if provided
in Step 4, they override the inline values at load time. If no CSV paths are set, the
inline param arrays are used as-is.

**Note on `subpop_params`:** These overrides are written directly in `model_config.json`
and are not editable via the UI sliders — edit the JSON file directly to add or change
per-subpop values. They are preserved across save/load cycles.
        """),

        "⚡ Rate template quick reference": mo.md("""
| Template | When to use | Example | Required rate_config keys |
|---|---|---|---|
| `constant_param` | Single fixed-rate transition | E→I recovery at rate `gamma` | `param` |
| `param_product` | Product of two or more parameters | S→H at `sigma × hosp_prop`; complement branch S→I at `sigma × (1 − hosp_prop)` | `factors` (list); optionally `complement_factors` |
| `immunity_modulated` | Rate that scales down as population immunity (M/MV) accumulates | S→E exposure rate suppressed by prior infection/vaccine immunity | `base_rate`, `proportion`, `is_complement`; optionally `inf_reduce_param`, `vax_reduce_param` |
| `force_of_infection` | Standard frequency-dependent incidence with a contact matrix | S→E infection driven by `beta`, contact patterns, and infectious compartments I/A | `beta_param`, `contact_matrix_schedule`, `infectious_compartments`, `relative_susceptibility_param`; optionally humidity/immunity fields |
| `force_of_infection_travel` | FOI with commuter mixing across subpopulations *(metapop only)* | S→E where residents of subpop A contact infectious individuals from subpop B during work hours | Same as above plus `travel_config` with `immobile_compartments`, `mobility_schedule` |
| `scheduled_exact` | Deterministic, exact transfer of a scheduled daily count (not a stochastic rate) | S→Vaccinated moving exactly the (rounded, delay-shifted) vaccinated count each day | `schedule` (name of a schedule, e.g. a `vaccine_schedule` instance) |

**Infectious compartments field** uses the format `CompartmentName:relative_infectivity_param`
(or just `CompartmentName` if all compartments are equally infectious), comma-separated.
Example: `IP:IP_relative_inf, IA:IA_relative_inf, ISR, ISH`
        """),
    })
    return


# ---------------------------------------------------------------------------
# Population structure (A/R, single vs metapop, metapop folder) now lives in the
# Population & Geography tab — see _nb_population.py. It exports num_age_groups,
# num_risk_groups, is_metapop, and metapop_folder_input, which the cells below
# consume unchanged.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1 — Compartments
# ---------------------------------------------------------------------------


@app.cell
def _compartments_ui(mo, loaded_config):
    _default = ", ".join(loaded_config.get("compartments", ["S", "E", "I", "R"]))
    compartments_text = mo.ui.text(
        value=_default,
        placeholder="S, E, I, R",
        label="Compartments (comma-separated)",
        full_width=True,
    )
    return (compartments_text,)


@app.cell
def _compartments_parse(compartments_text):
    raw = [_c.strip() for _c in compartments_text.value.split(",") if _c.strip()]
    compartments = list(dict.fromkeys(raw))
    return (compartments,)


@app.cell
def _compartments_display(compartments, compartments_text, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    if compartments:
        _body = mo.md("**Parsed:** " + "  ".join(f"`{_c}`" for _c in compartments))
    else:
        _body = mo.callout(mo.md("Enter at least one compartment name."), kind="warn")
    mo.vstack([
        mo.md("### Step 1 — Compartments"),
        compartments_text,
        _body,
    ])
    return


# ---------------------------------------------------------------------------
# Step 2 — Transitions
# ---------------------------------------------------------------------------


@app.cell
def _transition_count_ui(mo, loaded_config):
    _n_loaded = len(loaded_config.get("transitions", []))
    n_transitions = mo.ui.number(
        start=1,
        stop=12,
        step=1,
        value=min(max(_n_loaded, 1), 12) if _n_loaded else 3,
        label="Number of transitions",
    )
    return (n_transitions,)


@app.cell
def _transition_forms_ui(compartments, loaded_config, mo):
    _max_t = 12
    _comps = compartments if compartments else ["?"]
    _templates = [
        "constant_param",
        "param_product",
        "immunity_modulated",
        "force_of_infection",
        "force_of_infection_travel",
        "scheduled_exact",
    ]
    _t_cfgs = loaded_config.get("transitions", [])

    def _tget(i, key, default):
        if i < len(_t_cfgs):
            return _t_cfgs[i].get(key, default)
        return default

    def _rcget(i, key, default):
        if i < len(_t_cfgs):
            return _t_cfgs[i].get("rate_config", {}).get(key, default)
        return default

    def _origin_default(i):
        _v = _tget(i, "origin", None)
        if _v and _v in _comps:
            return _v
        return _comps[i] if i < len(_comps) else _comps[0]

    def _dest_default(i):
        _v = _tget(i, "destination", None)
        if _v and _v in _comps:
            return _v
        return _comps[i + 1] if i < len(_comps) - 1 else _comps[-1]

    t_name = mo.ui.array([
        mo.ui.text(
            value=_tget(_i, "name", f"{_origin_default(_i)}_to_{_dest_default(_i)}"),
            label="Name",
        )
        for _i in range(_max_t)
    ])
    t_origin = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_origin_default(_i), label="Origin")
        for _i in range(_max_t)
    ])
    t_dest = mo.ui.array([
        mo.ui.dropdown(options=_comps, value=_dest_default(_i), label="Destination")
        for _i in range(_max_t)
    ])
    t_template = mo.ui.array([
        mo.ui.dropdown(
            options=_templates,
            value=_tget(_i, "rate_template", "constant_param"),
            label="",
        )
        for _i in range(_max_t)
    ])

    t_param = mo.ui.array([
        mo.ui.text(
            value=_rcget(
                _i, "param", f"{_origin_default(_i)}_to_{_dest_default(_i)}_rate"
            ),
            label="Param name",
        )
        for _i in range(_max_t)
    ])
    t_schedule_name = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "schedule", "vaccinated_transfer_schedule"),
            label="Schedule name",
        )
        for _i in range(_max_t)
    ])
    t_factors = mo.ui.array([
        mo.ui.text(value=", ".join(_rcget(_i, "factors", [])), label="")
        for _i in range(_max_t)
    ])
    t_complements = mo.ui.array([
        mo.ui.text(value=", ".join(_rcget(_i, "complement_factors", [])), label="")
        for _i in range(_max_t)
    ])

    t_base_rate = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "base_rate", "base_rate"), label="Base rate param")
        for _i in range(_max_t)
    ])
    t_proportion = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "proportion", "split_prop"), label="Proportion param")
        for _i in range(_max_t)
    ])
    t_is_complement = mo.ui.array([
        mo.ui.checkbox(
            label="Use complement branch",
            value=bool(_rcget(_i, "is_complement", False)),
        )
        for _i in range(_max_t)
    ])
    t_inf_reduce = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "inf_reduce_param", "inf_risk_reduce"),
            label="Infection reduction param",
        )
        for _i in range(_max_t)
    ])
    t_vax_reduce = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "vax_reduce_param", "vax_risk_reduce"),
            label="Vaccine reduction param",
        )
        for _i in range(_max_t)
    ])

    t_beta = mo.ui.array([
        mo.ui.text(value=_rcget(_i, "beta_param", "beta_baseline"), label="Beta param")
        for _i in range(_max_t)
    ])
    t_rel_sus = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "relative_susceptibility_param", "relative_suscept"),
            label="Relative susceptibility param",
        )
        for _i in range(_max_t)
    ])

    def _travel_config_get(i, key, default):
        """Look up a key inside rate_config.travel_config (for force_of_infection_travel)."""
        _tc = _rcget(i, "travel_config", None)
        if _tc and isinstance(_tc, dict):
            return _tc.get(key, default)
        return default

    def _infectious_default(i):
        _raw = _rcget(i, "infectious_compartments", None)
        if _raw is None:
            _raw = _travel_config_get(i, "infectious_compartments", None)
        if _raw and isinstance(_raw, dict):
            return ", ".join(_raw.keys())
        return "I"

    t_infectious = mo.ui.array([
        mo.ui.text(
            value=_infectious_default(_i),
            label="",
            placeholder="IP, IA, ISR, ISH",
        )
        for _i in range(_max_t)
    ])
    t_use_humidity = mo.ui.array([
        mo.ui.checkbox(
            label="Include humidity modifier",
            value=bool(_rcget(_i, "humidity_impact_param", None)),
        )
        for _i in range(_max_t)
    ])
    t_humidity_impact = mo.ui.array([
        mo.ui.text(
            value=_rcget(_i, "humidity_impact_param", "humidity_impact"),
            label="Humidity impact param",
        )
        for _i in range(_max_t)
    ])
    t_use_foi_immunity = mo.ui.array([
        mo.ui.checkbox(
            label="Include immunity modifier",
            value=bool(_rcget(_i, "inf_reduce_param", None)),
        )
        for _i in range(_max_t)
    ])
    t_immobile = mo.ui.array([
        mo.ui.text(
            value=", ".join(
                _travel_config_get(_i, "immobile_compartments", None)
                or _rcget(_i, "immobile_compartments", [])
            ),
            label="",
        )
        for _i in range(_max_t)
    ])

    return (
        t_name, t_origin, t_dest, t_template,
        t_param, t_schedule_name, t_factors, t_complements,
        t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
        t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
        t_use_foi_immunity, t_immobile,
    )


@app.cell
def _transition_show(
    mo,
    main_tab,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_schedule_name, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
):
    mo.stop(main_tab.value != "Model Builder", None)
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

    def _with_tip(label_text, tip_text, widget):
        return mo.hstack([_tip_label(label_text, tip_text), widget], justify="start", align="center")

    _IMMUNITY_TIP = (
        "Divides the rate or force of infection by a population-level immunity factor:\n\n"
        "  immunity_force =\n"
        "    1\n"
        "    + (r_inf / (1 − r_inf)) × M\n"
        "    + (r_vax / (1 − r_vax)) × MV\n\n"
        "  r_inf = inf_reduce_param ∈ [0, 1)\n"
        "  r_vax = vax_reduce_param ∈ [0, 1)\n"
        "  M  = cumulative infection-induced immunity\n"
        "  MV = cumulative vaccine-induced immunity\n\n"
        "Higher r → stronger rate reduction.\n"
        "Example: r_inf = 0.5, M = 1 → rate halved.\n\n"
        "Requires at least one of M or MV to be enabled\n"
        "in Step 4, otherwise immunity_force stays at 1."
    )

    def _immunity_checkbox(checkbox):
        return mo.hstack([checkbox, _tip_label("", _IMMUNITY_TIP)], justify="start", align="center")


    _n = int(n_transitions.value)
    _rows = []
    for _i in range(_n):
        _template = t_template.value[_i]

        if _template == "constant_param":
            _rate_ui = t_param[_i]
        elif _template == "param_product":
            _rate_ui = mo.vstack([
                _with_tip(
                    "Factors",
                    "Comma-separated parameter names multiplied together to form the rate.\n\n"
                    "Example: base_rate, hosp_prop\n"
                    "Rate = base_rate × hosp_prop\n\n"
                    "Each name gets a slider in Step 3.",
                    t_factors[_i],
                ),
                _with_tip(
                    "Complement factors",
                    "Parameters applied as (1 − param) factors in the product.\n"
                    "Useful for modelling the fraction that does NOT take a given path.\n\n"
                    "Example: hosp_prop as a complement (with base_rate as a factor)\n"
                    "Rate = base_rate × (1 − hosp_prop)",
                    t_complements[_i],
                ),
            ])
        elif _template == "immunity_modulated":
            _rate_ui = mo.vstack([
                t_base_rate[_i],
                t_proportion[_i],
                t_is_complement[_i],
                _immunity_checkbox(t_use_foi_immunity[_i]),
                t_inf_reduce[_i],
                t_vax_reduce[_i],
            ])
        elif _template == "force_of_infection":
            _foi_items = [
                t_beta[_i],
                t_rel_sus[_i],
                _with_tip(
                    "Infectious compartments",
                    "Comma-separated names of compartments that contribute to\n"
                    "this force of infection.\n\n"
                    "Example: I, A\n\n"
                    "Each compartment's relative infectiousness (vs. 1x baseline)\n"
                    "is set once for the whole model in Step 1 — Compartments, so\n"
                    "the same value is shared by every transition that lists it.",
                    t_infectious[_i],
                ),
                t_use_humidity[_i],
            ]
            if t_use_humidity.value[_i]:
                _foi_items.append(t_humidity_impact[_i])
            _foi_items.append(_immunity_checkbox(t_use_foi_immunity[_i]))
            if t_use_foi_immunity.value[_i]:
                _foi_items.extend([t_inf_reduce[_i], t_vax_reduce[_i]])
            _rate_ui = mo.vstack(_foi_items)
        elif _template == "scheduled_exact":
            _rate_ui = mo.vstack([
                _with_tip(
                    "Schedule name",
                    "Name of the schedule providing the exact daily count of\n"
                    "individuals to move from origin to destination (e.g. a\n"
                    "vaccine_schedule instance backed by a per-subpop CSV with\n"
                    "one AxR array per day).\n\n"
                    "The count is rounded to the nearest integer and capped at\n"
                    "the origin compartment's current population -- this is a\n"
                    "deterministic, exact transfer, not a stochastic rate.\n\n"
                    "Configure the underlying data source and delay in Step 4.",
                    t_schedule_name[_i],
                ),
            ])
        else:
            _foit_items = [
                t_beta[_i],
                t_rel_sus[_i],
                _with_tip(
                    "Infectious compartments",
                    "Comma-separated names of compartments that contribute to\n"
                    "this force of infection.\n\n"
                    "Example: I, A\n\n"
                    "Each compartment's relative infectiousness (vs. 1x baseline)\n"
                    "is set once for the whole model in Step 1 — Compartments, so\n"
                    "the same value is shared by every transition that lists it.",
                    t_infectious[_i],
                ),
                t_use_humidity[_i],
            ]
            if t_use_humidity.value[_i]:
                _foit_items.append(t_humidity_impact[_i])
            _foit_items.append(_immunity_checkbox(t_use_foi_immunity[_i]))
            if t_use_foi_immunity.value[_i]:
                _foit_items.extend([t_inf_reduce[_i], t_vax_reduce[_i]])
            _foit_items.append(_with_tip(
                "Immobile compartments",
                "Comma-separated compartment names whose members do NOT travel\n"
                "between subpopulations (no cross-subpop mixing).\n\n"
                "Example: H, ICU",
                t_immobile[_i],
            ))
            _rate_ui = mo.vstack(_foit_items)

        _rows.append(mo.vstack([
            mo.md(f"**Transition {_i + 1}**"),
            mo.vstack([
                mo.hstack([t_origin[_i], t_dest[_i]], justify="start"),
                t_name[_i],
                _with_tip(
                    "Rate template",
                    "Determines how the transition rate is computed each timestep.\n\n"
                    "constant_param — single fixed rate parameter\n"
                    "  e.g. E→I recovery at rate gamma\n\n"
                    "param_product — product of multiple parameters\n"
                    "  e.g. S→H at sigma × hosp_prop\n\n"
                    "immunity_modulated — rate suppressed by cumulative infection/vaccine immunity (M/MV)\n"
                    "  e.g. S→E exposure dampened by prior immunity\n\n"
                    "force_of_infection — standard frequency-dependent incidence with a contact matrix\n"
                    "  e.g. S→E driven by beta, contact patterns, and infectious compartments\n\n"
                    "force_of_infection_travel — FOI with commuter mixing across subpopulations (metapop only)\n"
                    "  e.g. S→E where residents of subpop A contact infectious people from subpop B\n\n"
                    "scheduled_exact — exact, deterministic transfer of a scheduled daily count\n"
                    "  e.g. S→Vaccinated moving exactly the vaccinated count each day (not stochastic)\n\n"
                    "See the ⚡ Rate template quick reference accordion above for full details.",
                    t_template[_i],
                ),
            ]),
            _rate_ui,
            mo.md("---"),
        ]))

    mo.vstack([
        mo.md("### Step 2 — Transitions"),
        n_transitions,
        *_rows,
    ])
    return


@app.cell
def _template_requirements(
    n_transitions, t_template, t_use_humidity, t_use_foi_immunity,
):
    _n = int(n_transitions.value)
    _uses_contact_matrix = False
    _uses_absolute_humidity = False
    _uses_mobility = False
    _requires_immunity_metrics = False
    _uses_scheduled_transfer = False

    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "immunity_modulated":
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "force_of_infection_travel":
            _uses_contact_matrix = True
            _uses_absolute_humidity = _uses_absolute_humidity or bool(t_use_humidity.value[_i])
            _uses_mobility = True
            _requires_immunity_metrics = _requires_immunity_metrics or bool(t_use_foi_immunity.value[_i])
        elif _template == "scheduled_exact":
            _uses_scheduled_transfer = True

    uses_absolute_humidity = _uses_absolute_humidity
    uses_contact_matrix = _uses_contact_matrix
    uses_mobility = _uses_mobility
    requires_immunity_metrics = _requires_immunity_metrics
    uses_scheduled_transfer = _uses_scheduled_transfer
    return (
        uses_absolute_humidity, uses_contact_matrix, uses_mobility,
        requires_immunity_metrics, uses_scheduled_transfer,
    )


@app.cell
def _collect_param_names(
    n_transitions, t_template,
    t_param, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact, t_use_foi_immunity,
    parse_csv_list, rel_inf_param_name,
):
    _n = int(n_transitions.value)
    _names = []
    _infectious_comp_names = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _p = t_param.value[_i].strip()
            if _p:
                _names.append(_p)
        elif _template == "param_product":
            _names.extend(parse_csv_list(t_factors.value[_i]))
            _names.extend(parse_csv_list(t_complements.value[_i]))
        elif _template == "immunity_modulated":
            for _p in (t_base_rate.value[_i], t_proportion.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
        elif _template == "force_of_infection":
            for _p in (t_beta.value[_i], t_rel_sus.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _infectious_comp_names.extend(parse_csv_list(t_infectious.value[_i]))
        elif _template == "force_of_infection_travel":
            for _p in (t_beta.value[_i], t_rel_sus.value[_i]):
                _p = _p.strip()
                if _p:
                    _names.append(_p)
            if t_use_humidity.value[_i]:
                _p = t_humidity_impact.value[_i].strip()
                if _p:
                    _names.append(_p)
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _names.append(_p)
            _infectious_comp_names.extend(parse_csv_list(t_infectious.value[_i]))

    param_names = list(dict.fromkeys(_names))

    _reduce_names = set()
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template in ("immunity_modulated", "force_of_infection", "force_of_infection_travel"):
            if t_use_foi_immunity.value[_i]:
                for _p in (t_inf_reduce.value[_i], t_vax_reduce.value[_i]):
                    _p = _p.strip()
                    if _p:
                        _reduce_names.add(_p)
    reduce_param_names = _reduce_names

    infectious_compartment_names = list(dict.fromkeys(_infectious_comp_names))
    rel_inf_param_names = [rel_inf_param_name(_c) for _c in infectious_compartment_names]
    param_names = list(dict.fromkeys(param_names + rel_inf_param_names))

    return param_names, reduce_param_names, infectious_compartment_names, rel_inf_param_names


# ---------------------------------------------------------------------------
# Step 3 — Parameters
# ---------------------------------------------------------------------------


@app.cell
def _params_ui(param_names, reduce_param_names, rel_inf_param_names, loaded_config, mo,
               is_array_param, param_grid_columns, num_age_groups, num_risk_groups, age_groups):
    _saved_params = loaded_config.get("params", {})
    _rel_inf_set = set(rel_inf_param_names)
    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)
    _risk_labels = [f"risk{_r}" for _r in range(_R)]

    def _scalar_seed(_name):
        _v = _saved_params.get(_name, 0.5 if _name in reduce_param_names else 1.0)
        # A param saved as an A×R array still needs a single number to seed the
        # scalar input — use its first entry.
        while isinstance(_v, list):
            _v = _v[0]
        return float(_v)

    def _grid_seed(_name, _a, _r):
        # Saved A×R params are nested [age][risk] lists (see _build_config); pull
        # each cell's own value instead of collapsing the whole array to one number.
        _v = _saved_params.get(_name)
        if (isinstance(_v, list) and len(_v) == _A
                and all(isinstance(_row, list) and len(_row) == _R for _row in _v)):
            return float(_v[_a][_r])
        return _scalar_seed(_name)

    # One toggle / scalar input / A×R data_editor per param, all built every run
    # so that flipping one param's toggle doesn't shift the others' widget
    # identity (which would otherwise reset their live values on rerun). Wrapped
    # in mo.ui.dictionary (not a plain dict) so interacting with any nested
    # element triggers reactive reruns of cells that reference the container —
    # a plain dict of UI elements does not propagate value-change reactivity.
    param_vary_toggles = mo.ui.dictionary({
        _name: mo.ui.checkbox(
            value=is_array_param(loaded_config, _name),
            label="Vary by age/risk group",
        )
        for _name in param_names
    })
    param_scalar_inputs = mo.ui.dictionary({
        _name: mo.ui.number(
            start=0.0, stop=10.0, step=None,
            value=_scalar_seed(_name),
            label="" if _name in _rel_inf_set else _name,
        )
        for _name in param_names
    })
    # Rows are risk groups, columns are age groups (named from the Population &
    # Geography tab's age bands when available); "risk_group" is a read-only
    # row-label column.
    param_grid_inputs = mo.ui.dictionary({
        _name: mo.ui.data_editor(
            data=[
                {"risk_group": _risk_labels[_r],
                 **{_c: _grid_seed(_name, _a, _r) for _a, _c in enumerate(_age_cols)}}
                for _r in range(_R)
            ],
            label="" if _name in _rel_inf_set else _name,
            editable_columns=list(_age_cols),
        )
        for _name in param_names
    })
    return param_vary_toggles, param_scalar_inputs, param_grid_inputs


@app.cell
def _params_show(
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    infectious_compartment_names, rel_inf_param_names,
    mo, main_tab,
):
    mo.stop(main_tab.value != "Model Builder", None)
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

    _rel_inf_tip = (
        "Relative infectiousness of this compartment vs. baseline (1.0).\n\n"
        "Used by every force-of-infection transition (Step 2) that lists "
        "this compartment as infectious — the same value applies everywhere "
        "it's used."
    )

    def _param_row(_name, _label_node):
        _toggle = param_vary_toggles[_name]
        _content = param_grid_inputs[_name] if _toggle.value else param_scalar_inputs[_name]
        return mo.vstack([
            mo.hstack([_label_node, _toggle], wrap=True, justify="start"),
            _content,
        ])

    _regular_names = [_n for _n in param_names if _n not in rel_inf_param_names]

    _parts = [mo.md("### Step 3 — Parameters")]
    if not param_names:
        _parts.append(mo.callout(mo.md("No transition parameters found yet."), kind="warn"))
    for _name in _regular_names:
        _parts.append(_param_row(_name, mo.md(f"**`{_name}`**")))

    if rel_inf_param_names:
        _parts.append(mo.md("#### **Relative infectiousness**"))
        for _comp, _name in zip(infectious_compartment_names, rel_inf_param_names):
            _parts.append(_param_row(
                _name,
                _tip_label(f"<b><code>{_html.escape(_comp)}</code></b>", _rel_inf_tip),
            ))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 4 — Schedules and Immunity (scalar inputs + CSV file paths)
# ---------------------------------------------------------------------------


@app.cell
def _schedule_and_immunity_ui(
    mo, loaded_config, num_age_groups, num_risk_groups, age_groups, param_grid_columns,
    is_metapop, pop_subpop_names,
):
    _epi_names = [_m["name"] for _m in loaded_config.get("epi_metrics", [])]
    _saved_params = loaded_config.get("params", {})
    include_inf_immunity = mo.ui.checkbox(
        label="Include infection-induced immunity metric (M)",
        value="M" in _epi_names,
    )
    include_vax_immunity = mo.ui.checkbox(
        label="Include vaccine-induced immunity metric (MV)",
        value="MV" in _epi_names,
    )
    total_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=1.0, label="Total contact matrix value",
    )
    school_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=0.0, label="School contact subtraction",
    )
    work_contact_input = mo.ui.number(
        start=0.0, stop=100.0, step=None, value=0.0, label="Work contact subtraction",
    )
    mobility_input = mo.ui.number(
        start=0.0, stop=5.0, step=None, value=1.0, label="Mobility modifier",
    )
    daily_vaccines_input = mo.ui.number(
        start=0.0, stop=1e9, step=1.0, value=0.0, label="",
    )
    # "Vary by age/risk group" toggle + A×R grid, mirroring the Step 3
    # Parameters scalar/grid pattern — lets a constant daily-vaccines value
    # differ per age/risk group instead of broadcasting one number to all.
    _A = num_age_groups
    _R = num_risk_groups
    _age_cols = param_grid_columns(age_groups, _A)
    _risk_labels = [f"risk{_r}" for _r in range(_R)]
    daily_vaccines_vary_toggle = mo.ui.checkbox(
        value=False, label="Vary by age/risk group",
    )
    daily_vaccines_grid_input = mo.ui.data_editor(
        data=[
            {"risk_group": _risk_labels[_r], **{_c: 0.0 for _c in _age_cols}}
            for _r in range(_R)
        ],
        label="",
        editable_columns=list(_age_cols),
    )

    # Optional per-subpopulation override of the constant daily-vaccines value
    # (metapop only). The editor widget for the selected subpop is rebuilt
    # fresh on every render (see _schedule_csv_show) with its on_change
    # writing into this state dict — a single long-lived widget reused across
    # subpops would reset to its construction-time default each time the
    # selector switches back to a previously-edited subpop (see _init_ui's
    # get_seed_values/set_seed_values, which hit the same issue for Step 6).
    get_subpop_vax_values, set_subpop_vax_values = mo.state(
        dict(loaded_config.get("subpop_daily_vaccines", {}) or {})
    )
    daily_vaccines_per_subpop_toggle = mo.ui.checkbox(
        value=False, label="Vary by subpopulation",
    )
    daily_vaccines_subpop_selector = mo.ui.dropdown(
        options=list(pop_subpop_names),
        value=pop_subpop_names[0] if pop_subpop_names else None,
        label="Subpopulation",
    )
    vax_transfer_delay_input = mo.ui.number(
        start=0, stop=60, step=1,
        value=int(loaded_config.get("params", {}).get("vax_transfer_delay_days", 0)),
        label="vax_transfer_delay_days",
    )
    return (
        include_inf_immunity,
        include_vax_immunity,
        total_contact_input,
        school_contact_input,
        work_contact_input,
        mobility_input,
        daily_vaccines_input,
        daily_vaccines_vary_toggle,
        daily_vaccines_grid_input,
        daily_vaccines_per_subpop_toggle,
        daily_vaccines_subpop_selector,
        get_subpop_vax_values,
        set_subpop_vax_values,
        vax_transfer_delay_input,
    )


@app.cell
def _epi_metric_ui(n_transitions, t_name, mo, loaded_config):
    _saved_params = loaded_config.get("params", {})
    _epi_cfgs = {_m["name"]: _m for _m in loaded_config.get("epi_metrics", [])}
    _M_cfg = _epi_cfgs.get("M", {}).get("update_config", {})
    transition_names = [
        t_name.value[_i].strip()
        for _i in range(int(n_transitions.value))
        if t_name.value[_i].strip()
    ]
    opts = transition_names if transition_names else [""]
    _rtos_saved = _M_cfg.get("r_to_s_transition", opts[-1])
    r_to_s_picker = mo.ui.dropdown(
        options=opts,
        value=_rtos_saved if _rtos_saved in opts else opts[-1],
        label="Transition used for R→S-style immunity update",
    )
    inf_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("inf_induced_saturation", 0.0)),
        label="inf_induced_saturation",
    )
    vax_sat_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("vax_induced_saturation", 0.0)),
        label="vax_induced_saturation",
    )
    inf_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=float(_saved_params.get("inf_induced_immune_wane", 0.01)),
        label="inf_induced_immune_wane",
    )
    _vax_wane_raw = _saved_params.get("vax_induced_immune_wane", 0.0)
    vax_wane_is_array = isinstance(_vax_wane_raw, list)
    vax_wane_loaded_val = _vax_wane_raw
    vax_wane_input = mo.ui.number(
        start=0.0, stop=1.0, step=None,
        value=0.0 if vax_wane_is_array else float(_vax_wane_raw),
        label="vax_induced_immune_wane",
    )
    vax_delay_input = mo.ui.number(
        start=0, stop=60, step=1,
        value=int(_saved_params.get("vax_protection_delay_days", 0)),
        label="vax_protection_delay_days",
    )
    vax_reset_date_input = mo.ui.text(
        value=str(_saved_params.get("vax_immunity_reset_date_mm_dd", "")),
        placeholder="07_30",
        label="vax_immunity_reset_date_mm_dd (MM_DD, blank to disable)",
    )
    return (r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input,
            vax_wane_input, vax_wane_is_array, vax_wane_loaded_val,
            vax_delay_input, vax_reset_date_input)


@app.cell
def _schedule_csv_ui(
    mo, loaded_config, num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
    is_metapop, metapop_folder_input, Path,
):
    _inf = loaded_config.get("input_files", {})
    _multi = (num_age_groups > 1) or (num_risk_groups > 1)

    # Single shared folder holding every CSV below. The file fields are bare
    # filenames resolved against it (like the metapop folder). Legacy configs
    # that stored full paths leave this empty — the resolver passes them through.
    _folder_saved = _inf.get("input_folder", "")

    input_folder = mo.ui.text(
        value=_folder_saved,
        placeholder="/path/to/input_folder",
        label="Input folder (all CSV files below live here)",
        full_width=True,
    )

    # Humidity is CSV-only: a constant humidity modifier just scales beta by a
    # constant, which is a no-op. Auto-detect absolute_humidity.csv in the shared
    # folder (bare name), else the metapop folder (full path) for old layouts.
    _ah_csv_saved = _inf.get("absolute_humidity_csv", "")
    if not _ah_csv_saved:
        if _folder_saved and (Path(_folder_saved) / "absolute_humidity.csv").exists():
            _ah_csv_saved = "absolute_humidity.csv"
        elif is_metapop and metapop_folder_input.value.strip():
            _candidate = Path(metapop_folder_input.value.strip()) / "absolute_humidity.csv"
            if _candidate.exists():
                _ah_csv_saved = str(_candidate)

    ah_path = mo.ui.text(
        value=_ah_csv_saved,
        placeholder="absolute_humidity.csv",
        label="Absolute humidity CSV (filename, required for humidity modifier)",
        full_width=True,
    )
    cal_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("school_work_calendar_csv") else "constant",
        label="",
    )
    cal_path = mo.ui.text(
        value=_inf.get("school_work_calendar_csv", ""),
        placeholder="school_work_calendar.csv",
        label="School/work calendar CSV (filename)",
        full_width=True,
    )
    mob_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("mobility_csv") else ("csv" if _multi else "constant"),
        label="",
    )
    mob_path = mo.ui.text(
        value=_inf.get("mobility_csv", ""),
        placeholder="mobility_modifier.csv",
        label="Mobility CSV (filename)",
        full_width=True,
    )
    vax_mode = mo.ui.radio(
        options=["constant", "csv"],
        value="csv" if _inf.get("vaccines_csv") else ("csv" if _multi else "constant"),
        label="",
    )
    vax_path = mo.ui.text(
        value=_inf.get("vaccines_csv", ""),
        placeholder="daily_vaccines.csv",
        label="Vaccines CSV (filename)",
        full_width=True,
    )
    total_contact_csv_path = mo.ui.text(
        value=_inf.get("total_contact_matrix_csv", ""),
        placeholder="total_contact_matrix.csv",
        label="Total contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    school_contact_csv_path = mo.ui.text(
        value=_inf.get("school_contact_matrix_csv", ""),
        placeholder="school_contact_matrix.csv",
        label="School contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    work_contact_csv_path = mo.ui.text(
        value=_inf.get("work_contact_matrix_csv", ""),
        placeholder="work_contact_matrix.csv",
        label="Work contact matrix CSV (filename, A×A plain floats)",
        full_width=True,
    )
    return (
        input_folder,
        ah_path,
        cal_mode, cal_path,
        mob_mode, mob_path,
        vax_mode, vax_path,
        total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    )


@app.cell
def _schedule_csv_show(
    mo,
    num_age_groups, num_risk_groups,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, include_vax_immunity,
    uses_scheduled_transfer,
    input_folder,
    ah_path,
    cal_mode, cal_path,
    mob_mode, mob_path,
    vax_mode, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    total_contact_input, school_contact_input, work_contact_input,
    mobility_input, daily_vaccines_input,
    daily_vaccines_vary_toggle, daily_vaccines_grid_input,
    daily_vaccines_per_subpop_toggle, daily_vaccines_subpop_selector,
    get_subpop_vax_values, set_subpop_vax_values,
    load_csv_validated, load_contact_matrix_csv, resolve_input_path,
    SimpleNamespace,
    fetched_contact_matrices, fetched_matrices_scope,
    loaded_config, is_array_param,
    is_metapop, pop_subpop_names,
    age_groups, param_grid_columns, array_to_grid_rows, grid_to_AR_array,
):
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
            f"padding:6px 10px;width:300px;font-size:12px;line-height:1.5;"
            f"white-space:pre-wrap;pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            f"</style>"
            f"<span>"
            f"{label_text}&nbsp;"
            f'<span id="tip{_uid}">ⓘ<span>{_esc}</span></span>'
            f"</span>"
        )

    def _wtip(widget, tip_text):
        # Wrap the widget in a fit-content div so the tooltip sits right next
        # to it — without this, radio/checkbox widgets stretch to fill the
        # hstack item and push the tooltip far to the right.
        return mo.Html(
            f'<div style="display:inline-flex;align-items:center;gap:4px;">'
            f'<div style="width:fit-content;">{widget}</div>'
            f'{_tip_label("", tip_text)}'
            f'</div>'
        )

    _multi = (num_age_groups > 1) or (num_risk_groups > 1)
    _parts = [mo.md("#### **Schedule File Inputs**")]
    _parts.append(input_folder)

    # Absolute humidity — CSV-only (no constant option)
    _ah_df = None
    if uses_absolute_humidity:
        _parts.append(_wtip(ah_path, "CSV columns: date, absolute_humidity"))
        if ah_path.value.strip():
            _ah_df, _ah_err = load_csv_validated(
                resolve_input_path(input_folder.value, ah_path.value),
                ["date", "absolute_humidity"],
            )
            if _ah_err:
                _parts.append(mo.callout(mo.md(f"**Humidity CSV:** {_ah_err}"), kind="danger"))
            else:
                _parts.append(mo.callout(
                    mo.md(f"Humidity CSV: {len(_ah_df)} rows loaded."), kind="success"
                ))
        else:
            _parts.append(mo.callout(
                mo.md("**Humidity modifier is on but no CSV is set.** "
                      "Provide an absolute-humidity CSV filename above."),
                kind="warn",
            ))

    # School/work calendar
    _cal_df = None
    if uses_contact_matrix:
        _parts.append(_tip_label(
            "School/work calendar source",
            "constant: no calendar is used — the model always applies the full "
            "total contact matrix (school/work reductions are never applied).\n\n"
            "csv: vary contact patterns day-by-day using the is_school_day / "
            "is_work_day columns.",
        ))
        _parts.append(cal_mode)
        if cal_mode.value == "constant" and num_age_groups == 1:
            _parts.append(mo.hstack(
                [total_contact_input, school_contact_input, work_contact_input],
                wrap=True,
            ))
        if cal_mode.value == "csv":
            _parts.append(_wtip(
                cal_path,
                "CSV columns: date, is_school_day, is_work_day "
                "(floats in [0, 1], fractional days allowed)",
            ))
            if cal_path.value.strip():
                _cal_df, _cal_err = load_csv_validated(
                    resolve_input_path(input_folder.value, cal_path.value),
                    ["date", "is_school_day", "is_work_day"],
                )
                if _cal_err:
                    _parts.append(mo.callout(mo.md(f"**Calendar CSV:** {_cal_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Calendar CSV: {len(_cal_df)} rows loaded."), kind="success"
                    ))

    # Mobility
    _mob_df = None
    if uses_mobility:
        _parts.append(_tip_label(
            "Mobility source",
            "constant: the fixed Mobility modifier value entered here is "
            "applied every day.\n\n"
            "csv: vary the mobility modifier by day-of-week or date.",
        ))
        _parts.append(mob_mode)
        if mob_mode.value == "constant":
            _parts.append(mobility_input)
        if mob_mode.value == "csv":
            _parts.append(_wtip(
                mob_path,
                "CSV with a day_of_week or date column, plus a mobility_modifier "
                "column holding a JSON A×R array per row, e.g.\n"
                '[[0.94, 0.92], [0.94, 0.92], [0.85, 0.85]]',
            ))
            if mob_path.value.strip():
                _mob_df, _mob_err = load_csv_validated(
                    resolve_input_path(input_folder.value, mob_path.value), []
                )
                if _mob_err:
                    _parts.append(mo.callout(mo.md(f"**Mobility CSV:** {_mob_err}"), kind="danger"))
                else:
                    _has_col = "day_of_week" in _mob_df.columns or "date" in _mob_df.columns
                    if not _has_col:
                        _parts.append(mo.callout(
                            mo.md("**Mobility CSV:** Must have `day_of_week` or `date` column."),
                            kind="danger",
                        ))
                        _mob_df = None
                    else:
                        _parts.append(mo.callout(
                            mo.md(f"Mobility CSV: {len(_mob_df)} rows loaded."), kind="success"
                        ))
        elif _multi:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model (A={num_age_groups}, R={num_risk_groups}): "
                    "scalar mobility will broadcast to all groups."
                ),
                kind="warn",
            ))

    # Vaccines
    _vax_df = None
    if include_vax_immunity.value or uses_scheduled_transfer:
        _parts.append(_tip_label(
            "Vaccines source",
            "constant: the value(s) entered here are applied every day.\n\n"
            "csv: vary doses by date (and by age/risk group).",
        ))
        _parts.append(vax_mode)
        if vax_mode.value == "constant":
            _vax_const_tip = (
                "Each value is the proportion of that age/risk group's "
                "population vaccinated per day (e.g. 0.001 = 0.1% of the "
                "group vaccinated that day) — not a raw dose count.\n\n"
                "Off: one value broadcasts to every age/risk group.\n"
                "Vary by age/risk group: enter a separate proportion per cell."
            )
            _parts.append(_tip_label("Daily vaccines", _vax_const_tip))
            if is_metapop and len(pop_subpop_names) > 1:
                _parts.append(daily_vaccines_per_subpop_toggle)
            if is_metapop and len(pop_subpop_names) > 1 and daily_vaccines_per_subpop_toggle.value:
                _parts.append(daily_vaccines_subpop_selector)
                _sp = (
                    daily_vaccines_subpop_selector.value
                    if daily_vaccines_subpop_selector.value in pop_subpop_names
                    else pop_subpop_names[0]
                )
                _parts.append(daily_vaccines_vary_toggle)

                _subpop_vax_values = get_subpop_vax_values()
                _saved_sp_val = _subpop_vax_values.get(_sp)
                if daily_vaccines_vary_toggle.value:
                    _age_cols_vax = param_grid_columns(age_groups, num_age_groups)

                    def _on_subpop_vax_grid_change(_new_value, _sp=_sp, _age_cols_vax=_age_cols_vax):
                        _arr = grid_to_AR_array(_new_value, _age_cols_vax, num_age_groups, num_risk_groups)
                        set_subpop_vax_values({**get_subpop_vax_values(), _sp: _arr.tolist()})

                    _sp_grid = mo.ui.data_editor(
                        data=array_to_grid_rows(
                            _saved_sp_val if isinstance(_saved_sp_val, list) else None,
                            _age_cols_vax, num_risk_groups,
                        ),
                        label="",
                        editable_columns=list(_age_cols_vax),
                        on_change=_on_subpop_vax_grid_change,
                    )
                    _parts.append(_sp_grid)
                else:
                    def _on_subpop_vax_number_change(_new_value, _sp=_sp):
                        set_subpop_vax_values({**get_subpop_vax_values(), _sp: float(_new_value)})

                    _sp_number = mo.ui.number(
                        start=0.0, stop=1e9, step=1.0,
                        value=float(_saved_sp_val) if isinstance(_saved_sp_val, (int, float)) else 0.0,
                        label="",
                        on_change=_on_subpop_vax_number_change,
                    )
                    _parts.append(_sp_number)
            else:
                _parts.append(daily_vaccines_vary_toggle)
                if daily_vaccines_vary_toggle.value:
                    _parts.append(daily_vaccines_grid_input)
                else:
                    _parts.append(daily_vaccines_input)
        if vax_mode.value == "csv":
            _parts.append(_wtip(
                vax_path,
                "CSV columns: date, daily_vaccines — each value is the "
                "proportion of that age×risk group vaccinated on that day "
                "(JSON A×R array per row), not a raw count.",
            ))
            if vax_path.value.strip():
                _vax_df, _vax_err = load_csv_validated(
                    resolve_input_path(input_folder.value, vax_path.value),
                    ["date", "daily_vaccines"],
                )
                if _vax_err:
                    _parts.append(mo.callout(mo.md(f"**Vaccines CSV:** {_vax_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Vaccines CSV: {len(_vax_df)} rows loaded."), kind="success"
                    ))
        elif _multi:
            _parts.append(mo.callout(
                mo.md(
                    f"Multi-group model (A={num_age_groups}, R={num_risk_groups}): "
                    "scalar vaccine count will broadcast to all groups."
                ),
                kind="warn",
            ))

    # Contact matrices (A > 1)
    _total_contact_mat = None
    _school_contact_mat = None
    _work_contact_mat = None
    if uses_contact_matrix and num_age_groups > 1:
        _fetched_shared = (
            fetched_contact_matrices.get("__shared__", {})
            if fetched_matrices_scope == "shared" else {}
        )
        _fetched_per_subpop = fetched_matrices_scope == "per_subpop" and bool(fetched_contact_matrices)
        _already_fetched = bool(_fetched_shared) or _fetched_per_subpop

        if _already_fetched:
            # Matrices already fetched in the Population & Geography tab take
            # precedence over anything entered here — don't duplicate the
            # inputs or show a misleading "not set" warning for them.
            _parts.append(mo.callout(
                mo.md(
                    "**Contact matrices are set** via the **Population & Geography** "
                    "tab and will be used. To change them, fetch a different "
                    "geography there (or clear the fetch and provide CSVs below "
                    "instead)."
                ),
                kind="success",
            ))
        else:
            _parts.append(mo.md("**Contact matrices (required when A > 1):**"))
            _contact_csv_tip = (
                "Plain floats, A×A, no header row, no index column. Used only "
                "if no contact matrix is fetched in the Population & Geography tab."
            )
            _no_csv_inline = []
            _no_csv_unset = []

            _parts.append(_wtip(total_contact_csv_path, _contact_csv_tip))
            if total_contact_csv_path.value.strip():
                _total_contact_mat, _tc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, total_contact_csv_path.value),
                    num_age_groups,
                )
                if _tc_err:
                    _parts.append(mo.callout(mo.md(f"**Total contact matrix:** {_tc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Total contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "total_contact_matrix"):
                _no_csv_inline.append("total")
            else:
                _no_csv_unset.append("total")

            _parts.append(_wtip(school_contact_csv_path, _contact_csv_tip))
            if school_contact_csv_path.value.strip():
                _school_contact_mat, _sc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, school_contact_csv_path.value),
                    num_age_groups,
                )
                if _sc_err:
                    _parts.append(mo.callout(mo.md(f"**School contact matrix:** {_sc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"School contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "school_contact_matrix"):
                _no_csv_inline.append("school")
            else:
                _no_csv_unset.append("school")

            _parts.append(_wtip(work_contact_csv_path, _contact_csv_tip))
            if work_contact_csv_path.value.strip():
                _work_contact_mat, _wc_err = load_contact_matrix_csv(
                    resolve_input_path(input_folder.value, work_contact_csv_path.value),
                    num_age_groups,
                )
                if _wc_err:
                    _parts.append(mo.callout(mo.md(f"**Work contact matrix:** {_wc_err}"), kind="danger"))
                else:
                    _parts.append(mo.callout(
                        mo.md(f"Work contact matrix: {num_age_groups}×{num_age_groups} loaded."),
                        kind="success",
                    ))
            elif is_array_param(loaded_config, "work_contact_matrix"):
                _no_csv_inline.append("work")
            else:
                _no_csv_unset.append("work")

            if _no_csv_inline:
                _parts.append(mo.callout(
                    mo.md(
                        "No CSV set for **" + ", ".join(_no_csv_inline) + "** — using the "
                        "**inline config value(s)** from the loaded config."
                    ),
                    kind="success",
                ))
            if _no_csv_unset:
                _parts.append(mo.callout(
                    mo.md(
                        "**Not set: " + ", ".join(_no_csv_unset) + " contact matrix(es).** "
                        "No CSV is provided here, no inline value exists in the loaded "
                        "config, and (for total) none was fetched in the "
                        "**Population & Geography** tab — the model will fall back to "
                        "scalar `[[1.0]]` for these."
                    ),
                    kind="warn",
                ))

    mo.output.append(mo.vstack(_parts))

    loaded_schedule_dfs = SimpleNamespace(
        absolute_humidity_df=_ah_df,
        school_work_calendar_df=_cal_df,
        mobility_df=_mob_df,
        daily_vaccines_df=_vax_df,
        total_contact_matrix=_total_contact_mat,
        school_contact_matrix=_school_contact_mat,
        work_contact_matrix=_work_contact_mat,
    )
    return (loaded_schedule_dfs,)


@app.cell
def _schedule_and_immunity_show(
    mo,
    main_tab,
    include_inf_immunity,
    include_vax_immunity,
    vax_transfer_delay_input,
    r_to_s_picker,
    inf_sat_input,
    vax_sat_input,
    inf_wane_input,
    vax_wane_input,
    vax_wane_is_array,
    vax_wane_loaded_val,
    vax_delay_input,
    vax_reset_date_input,
    uses_absolute_humidity,
    uses_contact_matrix,
    uses_mobility,
    requires_immunity_metrics,
    uses_scheduled_transfer,
):
    mo.stop(main_tab.value != "Model Builder", None)
    import html as _html
    import random as _random

    def _tip_label(label_text, tip_text):
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
            f"padding:6px 10px;width:300px;font-size:12px;line-height:1.5;"
            f"white-space:pre-wrap;pointer-events:none;z-index:9999;}}"
            f"#tip{_uid}:hover>span{{visibility:visible;opacity:1;}}"
            f"</style>"
            f"<span>"
            f"{label_text}&nbsp;"
            f'<span id="tip{_uid}">ⓘ<span>{_esc}</span></span>'
            f"</span>"
        )

    def _wtip(widget, tip_text):
        # Wrap the widget in a fit-content div so the tooltip sits right next
        # to it — without this, radio/checkbox widgets stretch to fill the
        # hstack item and push the tooltip far to the right.
        return mo.Html(
            f'<div style="display:inline-flex;align-items:center;gap:4px;">'
            f'<div style="width:fit-content;">{widget}</div>'
            f'{_tip_label("", tip_text)}'
            f'</div>'
        )

    _parts = [
        mo.md("### Step 4 — Schedules and Immunity"),
        mo.hstack([
            _wtip(
                include_inf_immunity,
                "Track population-level infection-induced immunity (M).\n\n"
                "For instance, if driven by (R→S) transitions:\n\n"
                "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                "M increases when recently-recovered individuals re-enter\n"
                "the susceptible pool (R→S), and decays via waning.\n\n"
                "Must be enabled for inf_reduce_param (Step 2) to have effect.",
            ),
            _wtip(
                include_vax_immunity,
                "Track population-level vaccine-induced immunity (MV).\n\n"
                "MV grows with daily vaccine doses and decays via waning:\n\n"
                "ΔMV = daily_vaccines − wane×MV\n\n"
                "Must be enabled for vax_reduce_param (Step 2) to have effect.",
            ),
        ], wrap=True),
    ]

    if not uses_absolute_humidity and not uses_contact_matrix and not uses_mobility:
        _parts.append(mo.md("*No schedule-backed rate templates selected.*"))
    # The scalar values used when a schedule source is set to "constant"
    # (contact matrices for A=1, mobility, daily vaccines) are entered
    # directly below the matching source radio in Schedule File Inputs,
    # rather than here, so the input sits next to the control that reveals it.

    if uses_scheduled_transfer:
        _parts.append(_wtip(
            vax_transfer_delay_input,
            "Days between the scheduled date (e.g. vaccination date) and the\n"
            "date individuals actually move from origin to destination in a\n"
            "'scheduled_exact' transition.\n\n"
            "0 = transfer happens on the scheduled date itself.",
        ))

    if requires_immunity_metrics:
        _parts.append(mo.callout(
            mo.md(
                "Selected rate templates can use `M` and/or `MV`. "
                "Enable whichever immunity metrics you want to track."
            ),
            kind="info",
        ))

    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    if _immunity_active:
        _metric_inputs = []
        if include_inf_immunity.value:
            _metric_inputs.extend([
                _wtip(
                    r_to_s_picker,
                    "The transition that drives immunity gain.\n\n"
                    "M increases as people move from R back to S — recently-\n"
                    "recovered individuals re-entering the susceptible pool\n"
                    "still carry partial immunity.\n\n"
                    "Select the transition that represents this R→S flow.",
                ),
                _wtip(
                    inf_sat_input,
                    "Limits how much M can grow as immunity accumulates.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → M saturates at a lower level.\n"
                    "0 = no saturation limit.",
                ),
                _wtip(
                    vax_sat_input,
                    "How much vaccine immunity (MV) dampens further gain in M.\n\n"
                    "ΔM = (R→S / N) × (1 − inf_sat×M − vax_sat×MV) − wane×M\n\n"
                    "Higher values → MV reduces M accumulation more.\n"
                    "0 = vaccine and infection immunity are independent.",
                ),
                _wtip(
                    inf_wane_input,
                    "Daily decay rate of infection-induced immunity M.\n\n"
                    "ΔM = (R→S / N) × (...) − wane×M\n\n"
                    "0 = no waning.\n"
                    "0.01 ≈ half-life of ~70 days.",
                ),
            ])
        if include_vax_immunity.value:
            if vax_wane_is_array:
                _metric_inputs.append(mo.callout(
                    mo.md(
                        "**`vax_induced_immune_wane`** — loaded from config as A×R array "
                        "(slider disabled, value passes through unchanged)\n\n"
                        f"```json\n{vax_wane_loaded_val}\n```"
                    ),
                    kind="info",
                ))
            else:
                _metric_inputs.append(
                    _wtip(
                        vax_wane_input,
                        "Daily decay rate of vaccine-induced immunity MV.\n\n"
                        "ΔMV = daily_vaccines − wane×MV\n\n"
                        "0 = no waning.\n"
                        "0.01 ≈ half-life of ~70 days.",
                    )
                )
            _metric_inputs.extend([vax_delay_input, vax_reset_date_input])
        _parts.append(mo.hstack(_metric_inputs, wrap=True))
    else:
        _parts.append(mo.md("*Dynamic immunity metrics disabled.*"))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 5 — Model Diagram
# ---------------------------------------------------------------------------


@app.cell
def _diagram(
    compartments, n_transitions, t_name, t_origin, t_dest, t_template, t_infectious,
    parse_csv_list, mo, plt, main_tab,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _n = int(n_transitions.value)
    _inner = None
    _graphviz_error = None

    _foi_templates = ("force_of_infection", "force_of_infection_travel")
    _infectious_compartments: set[str] = set()
    _foi_links: list[tuple[str, int]] = []  # (infectious compartment, transition index)
    _foi_transition_idx: set[int] = set()
    for _i in range(_n):
        if t_template.value[_i] in _foi_templates and t_origin.value[_i] and t_dest.value[_i]:
            _foi_transition_idx.add(_i)
            for _c in parse_csv_list(t_infectious.value[_i]):
                _infectious_compartments.add(_c)
                _foi_links.append((_c, _i))

    try:
        import graphviz as gv  # type: ignore[import-untyped]
        _dot = gv.Digraph(
            graph_attr={"rankdir": "LR", "bgcolor": "white", "pad": "0.3"},
            node_attr={"shape": "box", "style": "rounded,filled", "fillcolor": "#ddeeff"},
        )
        for _c in compartments:
            if _c in _infectious_compartments:
                _dot.node(_c, fillcolor="#ffa64d")
            else:
                _dot.node(_c)
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            _label = t_name.value[_i]
            if not (_origin and _dest):
                continue
            if _i in _foi_transition_idx:
                # Split the edge with an invisible point node so the dashed
                # "drives this transition" arrows can target the edge itself
                # rather than the destination compartment.
                _tnode = f"_t{_i}"
                _dot.node(_tnode, shape="point", width="0.01", label="")
                _dot.edge(_origin, _tnode, arrowhead="none", label=_label)
                _dot.edge(_tnode, _dest)
            else:
                _dot.edge(_origin, _dest, label=_label)
        for _c, _i in _foi_links:
            # constraint=false: keep this edge out of graphviz's rank/position
            # solver so it doesn't drag the point node off the straight line
            # of the transition edge it's attached to (causes a visible kink).
            _dot.edge(
                _c, f"_t{_i}", style="dashed", color="#ffa64d", arrowhead="empty",
                constraint="false",
            )
        _inner = mo.image(_dot.pipe(format="png"), width="100%")
    except Exception as _exc:
        _graphviz_error = f"{type(_exc).__name__}: {_exc}"

    if _inner is None:
        _fig, _ax = plt.subplots(figsize=(max(4, len(compartments) * 2), 2))
        _ax.set_xlim(-0.5, len(compartments) - 0.5)
        _ax.set_ylim(-0.5, 1.5)
        _ax.axis("off")
        _pos = {_c: (_i, 0.5) for _i, _c in enumerate(compartments)}
        for _c, (_x, _y) in _pos.items():
            _facecolor = "#ffa64d" if _c in _infectious_compartments else "#ddeeff"
            _ax.text(_x, _y, _c, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=_facecolor))
        for _i in range(_n):
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            if _origin in _pos and _dest in _pos:
                _x0, _y0 = _pos[_origin]
                _x1, _y1 = _pos[_dest]
                _ax.annotate(
                    "", xy=(_x1 - 0.15, _y1), xytext=(_x0 + 0.15, _y0),
                    arrowprops=dict(arrowstyle="->", color="#336699"),
                )
        for _c, _i in _foi_links:
            _origin = t_origin.value[_i]
            _dest = t_dest.value[_i]
            if _c in _pos and _origin in _pos and _dest in _pos:
                _x0, _y0 = _pos[_c]
                _xo, _yo = _pos[_origin]
                _xd, _yd = _pos[_dest]
                _xm, _ym = (_xo + _xd) / 2, (_yo + _yd) / 2  # midpoint of the transition edge
                _ax.annotate(
                    "", xy=(_xm, _ym + 0.2), xytext=(_x0 + 0.15, _y0 + 0.2),
                    arrowprops=dict(arrowstyle="->", color="#ffa64d", linestyle="dashed"),
                )
        plt.tight_layout()
        _fallback_parts = []
        if _graphviz_error is None:
            _fallback_parts.append(
                mo.callout(
                    mo.md("*Graphviz not available; using a simple fallback diagram.*"),
                    kind="info",
                )
            )
        else:
            _fallback_parts.append(
                mo.callout(
                    mo.md(
                        "**Graphviz rendering failed; using fallback diagram.**\n\n"
                        f"`{_graphviz_error}`"
                    ),
                    kind="warn",
                )
            )
        _fallback_parts.append(_fig)
        _inner = mo.vstack(_fallback_parts)

    mo.vstack([mo.md("### Step 5 — Model Diagram"), _inner])
    return


# ---------------------------------------------------------------------------
# Step 6 — Initial Conditions
# ---------------------------------------------------------------------------


@app.cell
def _init_ui(mo, pop_subpop_names):
    # Edited grids are persisted here (keyed by "{subpop}::{compartment}")
    # rather than relying on the data_editor widgets' own internal state,
    # since those widgets are recreated each time the selected subpopulation
    # changes (see _init_show) and would otherwise reset to their
    # construction-time defaults, silently dropping edits made for a subpop
    # the user has switched away from.
    get_seed_values, set_seed_values = mo.state({})
    ic_subpop_selector = mo.ui.dropdown(
        options=list(pop_subpop_names),
        value=pop_subpop_names[0] if pop_subpop_names else None,
        label="Subpopulation",
    )
    return get_seed_values, set_seed_values, ic_subpop_selector


@app.cell
def _init_show(
    compartments, get_seed_values, set_seed_values, ic_subpop_selector,
    population_by_subpop, pop_subpop_names,
    num_age_groups, num_risk_groups, age_groups,
    param_grid_columns, grid_to_AR_array, default_seed_row_data,
    is_metapop, mo, main_tab, np, pd, loaded_config,
):
    mo.stop(main_tab.value != "Model Builder", None)
    _A = int(num_age_groups)
    _R = int(num_risk_groups)
    _seed_comps = compartments[1:] if len(compartments) > 1 else []
    _age_cols = param_grid_columns(age_groups, _A)
    _saved_ic = loaded_config.get("initial_conditions", {}) or {}

    _parts = [
        mo.md("### Step 6 — Initial Conditions"),
        mo.md(
            "Seed each compartment by **age** and **risk** group (absolute counts). "
            "The first compartment (`"
            + (compartments[0] if compartments else "?")
            + "`) receives the remaining population in each cell. Population totals "
            "come from the **Population & Geography** tab."
        ),
    ]

    _sp = ic_subpop_selector.value if ic_subpop_selector.value in pop_subpop_names else (
        pop_subpop_names[0] if pop_subpop_names else "aggregate_pop"
    )
    if is_metapop:
        _parts.append(ic_subpop_selector)
        _parts.append(mo.callout(
            mo.md(
                "These per-subpopulation tables **override** "
                "`initial_conditions_{name}.json` in the metapop folder. A subpop "
                "whose tables are left all-zero falls back to its folder file."
            ),
            kind="info",
        ))

    _pop = population_by_subpop.get(_sp)
    if _pop is None:
        _pop = np.zeros((_A, _R))
    _pop = np.asarray(_pop, dtype=float)

    _seed_values = get_seed_values()

    def _make_on_change(_key):
        def _cb(_new_value):
            set_seed_values({**get_seed_values(), _key: _new_value})
        return _cb

    _seed_total = np.zeros((_A, _R))
    for _ci, _c in enumerate(_seed_comps):
        _key = f"{_sp}::{_c}"
        _data = _seed_values.get(_key)
        if _data is None:
            _data = default_seed_row_data(_saved_ic, _sp, _c, _age_cols, _R, _ci == 0)
        _ed = mo.ui.data_editor(
            data=_data,
            label=_c,
            editable_columns=list(_age_cols),
            on_change=_make_on_change(_key),
        )
        _parts.append(mo.md(f"**{_c}**"))
        _parts.append(_ed)
        _seed_total = _seed_total + grid_to_AR_array(_data, _age_cols, _A, _R)

    _remainder = _pop - _seed_total
    _first = compartments[0] if compartments else "?"
    _first_df = pd.DataFrame(
        [{"risk_group": _r,
          **{_col: _remainder[_a, _r] for _a, _col in enumerate(_age_cols)}}
         for _r in range(_R)]
    )
    _parts.append(mo.md(
        f"**{_first}** (auto = population − Σ seeds; total {max(_remainder.sum(), 0):,.0f})"
    ))
    _parts.append(mo.ui.table(_first_df, selection=None))
    if bool(np.any(_remainder < 0)):
        _parts.append(mo.callout(
            mo.md("Seeded counts exceed the population in at least one age/risk cell."),
            kind="danger",
        ))

    mo.vstack(_parts)
    return


# ---------------------------------------------------------------------------
# Step 7 — Simulation Settings
# ---------------------------------------------------------------------------


@app.cell
def _sim_settings_ui(mo, loaded_config):
    _sim = loaded_config.get("simulation_settings", {})
    sim_days = mo.ui.number(start=10, stop=730, step=10, value=250, label="Simulation days")
    sim_mode = mo.ui.radio(
        options=["Deterministic", "Stochastic"],
        value="Deterministic",
        label="Simulation mode",
    )
    n_reps = mo.ui.number(start=1, stop=100, step=1, value=10, label="Replicates")
    rng_seed = mo.ui.number(start=0, stop=99999, step=1, value=42, label="RNG seed")
    timesteps = mo.ui.number(start=1, stop=24, step=1, value=7, label="Timesteps per day")
    start_date_input = mo.ui.text(
        value=_sim.get("start_real_date", "2024-01-01"),
        label="Simulation start date (YYYY-MM-DD)",
    )
    transition_vars_input = mo.ui.text(
        value=", ".join(_sim.get("transition_variables_to_save", [])),
        placeholder="ISH_to_HR, ISH_to_HD, S_to_E  (blank = save all)",
        label="Transition variables to save",
        full_width=True,
    )
    return sim_days, sim_mode, n_reps, rng_seed, timesteps, start_date_input, transition_vars_input


@app.cell
def _sim_settings_show(mo, sim_days, sim_mode, n_reps, rng_seed, timesteps, start_date_input, transition_vars_input, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.vstack([
        mo.md("### Step 7 — Simulation Settings"),
        mo.hstack([sim_days, sim_mode, timesteps, rng_seed], justify="start"),
        mo.hstack([
            n_reps,
            mo.md("*Ignored in deterministic mode.*") if sim_mode.value == "Deterministic" else mo.md(""),
        ]),
        start_date_input,
        transition_vars_input,
        mo.callout(
            mo.md(
                "Leaving **Transition variables to save** blank saves *every* "
                "transition variable each day. For large models or many "
                "replicates this can use a lot of memory and produce large "
                "output files — list only the transitions you need (e.g. "
                "`S_to_E, ISH_to_HR`)."
            ),
            kind="info",
        ) if not transition_vars_input.value.strip() else mo.md(""),
    ])
    return


# ---------------------------------------------------------------------------
# Build config dict
# ---------------------------------------------------------------------------


@app.cell
def _build_config(
    compartments,
    n_transitions,
    t_name, t_origin, t_dest, t_template,
    t_param, t_schedule_name, t_factors, t_complements,
    t_base_rate, t_proportion, t_is_complement, t_inf_reduce, t_vax_reduce,
    t_beta, t_rel_sus, t_infectious, t_use_humidity, t_humidity_impact,
    t_use_foi_immunity, t_immobile,
    param_names, param_vary_toggles, param_scalar_inputs, param_grid_inputs,
    param_grid_columns,
    include_inf_immunity, include_vax_immunity,
    r_to_s_picker, inf_sat_input, vax_sat_input, inf_wane_input,
    vax_wane_input, vax_wane_is_array, vax_wane_loaded_val,
    vax_delay_input, vax_reset_date_input,
    vax_transfer_delay_input,
    uses_absolute_humidity, uses_contact_matrix, uses_mobility, requires_immunity_metrics,
    uses_scheduled_transfer,
    rel_inf_param_name,
    parse_csv_list,
    total_contact_input, school_contact_input, work_contact_input,
    num_age_groups, num_risk_groups, age_groups,
    fetched_contact_matrices, fetched_matrices_scope,
    is_metapop, metapop_folder_input,
    loaded_schedule_dfs,
    input_folder,
    cal_mode, mob_mode, vax_mode,
    ah_path, cal_path, mob_path, vax_path,
    total_contact_csv_path, school_contact_csv_path, work_contact_csv_path,
    get_seed_values, default_seed_row_data, population_by_subpop, pop_subpop_names,
    risk_fraction_inputs, grid_to_AR_array,
    daily_vaccines_per_subpop_toggle, get_subpop_vax_values,
    loaded_config,
    start_date_input, transition_vars_input,
    analysis_n_metrics_input, analysis_metric_names, analysis_metric_tvs,
    np,
):
    # Assembles the full model_config.json dict from every Step's widgets.
    # Roadmap of the sections below (search for the banner comments):
    #   1. PARAMS        — seed from loaded config, overlay scalar or A×R grid inputs
    #   2. TRANSITIONS   — per-template rate_config + self-loop warnings
    #   3. CONTACT MATRIX PARAMS
    #   4. SCHEDULES
    #   5. EPI METRICS   — infection-/vaccine-induced immunity
    #   6. INPUT FILES   — CSV references resolved against the shared folder
    #   7. INITIAL CONDITIONS — per-subpop population + seed grids
    #   8. CONFIG DICT   — final assembly
    #   9. ANALYSIS METRICS
    # Returns: (config_dict, immunity_active, metapop_travel_config, config_warnings)
    _n = int(n_transitions.value)
    _A = num_age_groups
    _R = num_risk_groups

    def _infectious_compartments_map(_i):
        return {
            _c: rel_inf_param_name(_c)
            for _c in parse_csv_list(t_infectious.value[_i])
        }

    # --- 1. PARAMS ---
    # Seed from loaded config first, then overlay each param's Step 3 widget:
    # a single float when constant, or an A×R nested list when toggled to vary.
    # The grid's data_editor is row-oriented by risk group with one column per
    # age group, so transpose its rows back into the A×R nested-list shape.
    params_dict: dict = dict(loaded_config.get("params", {}))
    _age_cols = param_grid_columns(age_groups, _A)
    for _name in param_names:
        if param_vary_toggles[_name].value:
            _rows = list(param_grid_inputs[_name].value)
            params_dict[_name] = [
                [float(_rows[_r][_age_cols[_a]]) for _r in range(_R)]
                for _a in range(_A)
            ]
        else:
            params_dict[_name] = float(param_scalar_inputs[_name].value)

    # --- 2. TRANSITIONS ---
    _transitions = []
    _metapop_travel_config = {}
    _config_warnings = []
    for _i in range(_n):
        _template = t_template.value[_i]
        if _template == "constant_param":
            _rate_config = {"param": t_param.value[_i].strip()}
        elif _template == "param_product":
            _factors = parse_csv_list(t_factors.value[_i])
            _complements = parse_csv_list(t_complements.value[_i])
            _rate_config = {"factors": _factors}
            if _complements:
                _rate_config["complement_factors"] = _complements
        elif _template == "immunity_modulated":
            _rate_config = {
                "base_rate": t_base_rate.value[_i].strip(),
                "proportion": t_proportion.value[_i].strip(),
                "is_complement": bool(t_is_complement.value[_i]),
            }
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
        elif _template == "force_of_infection":
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "infectious_compartments": _infectious_compartments_map(_i),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
        elif _template == "scheduled_exact":
            _rate_config = {"schedule": t_schedule_name.value[_i].strip()}
        else:
            _travel_config = {
                "infectious_compartments": _infectious_compartments_map(_i),
                "immobile_compartments": parse_csv_list(t_immobile.value[_i]),
                "relative_susceptibility_param": t_rel_sus.value[_i].strip(),
                "contact_matrix_schedule": "flu_contact_matrix",
                "mobility_schedule": "mobility_modifier",
            }
            _rate_config = {
                "beta_param": t_beta.value[_i].strip(),
                "travel_config": _travel_config,
            }
            if t_use_humidity.value[_i]:
                _rate_config["humidity_impact_param"] = t_humidity_impact.value[_i].strip()
                _rate_config["humidity_schedule"] = "absolute_humidity"
            if t_use_foi_immunity.value[_i]:
                _inf_r = t_inf_reduce.value[_i].strip()
                _vax_r = t_vax_reduce.value[_i].strip()
                if _inf_r:
                    _rate_config["inf_reduce_param"] = _inf_r
                if _vax_r:
                    _rate_config["vax_reduce_param"] = _vax_r
            if not _metapop_travel_config:
                _metapop_travel_config = _travel_config

        _t_name = t_name.value[_i].strip()
        _t_origin = t_origin.value[_i]
        _t_dest = t_dest.value[_i]
        if _t_origin and _t_dest and _t_origin == _t_dest:
            _config_warnings.append(
                f"Transition '{_t_name or _i + 1}' has the same origin and "
                f"destination ('{_t_origin}') — this self-loop has no net effect."
            )
        _transitions.append({
            "name": _t_name,
            "origin": _t_origin,
            "destination": _t_dest,
            "rate_template": _template,
            "rate_config": _rate_config,
        })

    # --- 3. CONTACT MATRIX PARAMS ---
    if uses_contact_matrix:
        if _A == 1:
            params_dict["total_contact_matrix"] = [[float(total_contact_input.value)]]
            params_dict["school_contact_matrix"] = [[float(school_contact_input.value)]]
            params_dict["work_contact_matrix"] = [[float(work_contact_input.value)]]
        else:
            # A > 1: a proper A×A contact matrix is required. Prefer the CSV,
            # then an inline A×A list from the loaded config. Only fall back to a
            # scalar 1×1 matrix as a last resort, and warn loudly because that is
            # the wrong shape and will misbehave at run time.
            _shared_fetched_check = (
                fetched_contact_matrices.get("__shared__", {})
                if fetched_matrices_scope == "shared" else {}
            )
            _per_subpop_fetched = fetched_matrices_scope == "per_subpop" and fetched_contact_matrices
            for _label, _matrix_attr, _scalar_input in (
                ("total", "total_contact_matrix", total_contact_input),
                ("school", "school_contact_matrix", school_contact_input),
                ("work", "work_contact_matrix", work_contact_input),
            ):
                _loaded_mat = getattr(loaded_schedule_dfs, _matrix_attr)
                if _matrix_attr in _shared_fetched_check or _per_subpop_fetched:
                    pass  # fetched in Population & Geography tab — applied below
                elif _loaded_mat is not None:
                    params_dict[_matrix_attr] = _loaded_mat
                elif isinstance(params_dict.get(_matrix_attr), list) and \
                        len(params_dict[_matrix_attr]) == _A:
                    pass  # valid inline A×A matrix from loaded config — keep it
                else:
                    params_dict[_matrix_attr] = [[float(_scalar_input.value)]]
                    _config_warnings.append(
                        f"{_label.capitalize()} contact matrix: no {_A}×{_A} CSV or "
                        f"inline matrix provided for {_A} age groups; falling back to a "
                        f"scalar 1×1 matrix. Provide a {_A}×{_A} contact-matrix CSV in "
                        f"Step 4, or fetch matrices in the Population & Geography tab — "
                        f"the model will not behave correctly otherwise."
                    )

        # Contact matrices fetched in the Population & Geography tab take
        # precedence. Shared scope writes the three params here; per-subpop
        # scope is applied to subpop_params below (section 7).
        if fetched_matrices_scope == "shared":
            _shared_fetched = fetched_contact_matrices.get("__shared__", {})
            for _mname in ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"):
                if _mname in _shared_fetched:
                    params_dict[_mname] = _shared_fetched[_mname]

    # --- 4. SCHEDULES ---
    _schedules = []
    if uses_absolute_humidity:
        _schedules.append({
            "name": "absolute_humidity",
            "schedule_template": "timeseries_lookup",
            "schedule_config": {
                "df_attribute": "absolute_humidity_df",
                "value_column": "absolute_humidity",
            },
        })
    if uses_contact_matrix:
        _schedules.append({
            "name": "flu_contact_matrix",
            "schedule_template": "contact_matrix",
            "schedule_config": {
                "school_work_day_df_attribute": "school_work_calendar_df",
                "total_contact_matrix_param": "total_contact_matrix",
                "school_contact_matrix_param": "school_contact_matrix",
                "work_contact_matrix_param": "work_contact_matrix",
            },
        })
    if uses_mobility:
        _schedules.append({
            "name": "mobility_modifier",
            "schedule_template": "mobility",
            "schedule_config": {
                "df_attribute": "mobility_df",
            },
        })

    # --- 5. EPI METRICS ---
    _immunity_active = include_inf_immunity.value or include_vax_immunity.value
    _epi_metrics = []
    if include_vax_immunity.value:
        _schedules.append({
            "name": "daily_vaccines",
            "schedule_template": "vaccine_schedule",
            "schedule_config": {
                "df_attribute": "daily_vaccines_df",
            },
        })
    if uses_scheduled_transfer:
        _transfer_schedule_config = {"df_attribute": "daily_vaccines_df"}
        if int(vax_transfer_delay_input.value) > 0:
            params_dict["vax_transfer_delay_days"] = int(vax_transfer_delay_input.value)
            _transfer_schedule_config["vax_protection_delay_days_param"] = "vax_transfer_delay_days"
        _schedules.append({
            "name": "vaccinated_transfer_schedule",
            "schedule_template": "vaccine_schedule",
            "schedule_config": _transfer_schedule_config,
        })
    if include_inf_immunity.value:
        params_dict.update({
            "inf_induced_saturation": float(inf_sat_input.value),
            "vax_induced_saturation": float(vax_sat_input.value),
            "inf_induced_immune_wane": float(inf_wane_input.value),
        })
        _epi_metrics.append({
            "name": "M",
            "init_val": np.zeros((_A, _R)).tolist(),
            "metric_template": "infection_induced_immunity",
            "update_config": {
                "r_to_s_transition": r_to_s_picker.value,
                "inf_induced_saturation_param": "inf_induced_saturation",
                "vax_induced_saturation_param": "vax_induced_saturation",
                "inf_induced_immune_wane_param": "inf_induced_immune_wane",
            },
        })
    if include_vax_immunity.value:
        if not vax_wane_is_array:
            params_dict["vax_induced_immune_wane"] = float(vax_wane_input.value)
        # else: array already seeded from loaded_config above — vax_wane_loaded_val passes through
        if int(vax_delay_input.value) > 0:
            params_dict["vax_protection_delay_days"] = int(vax_delay_input.value)
        if vax_reset_date_input.value.strip():
            params_dict["vax_immunity_reset_date_mm_dd"] = vax_reset_date_input.value.strip()
        _epi_metrics.append({
            "name": "MV",
            "init_val": np.zeros((_A, _R)).tolist(),
            "metric_template": "vaccine_induced_immunity",
            "update_config": {
                "daily_vaccines_schedule": "daily_vaccines",
                "vax_induced_immune_wane_param": "vax_induced_immune_wane",
            },
        })

    # --- 6. INPUT FILES ---
    # The shared folder is recorded once and the CSV entries below are bare
    # filenames resolved against it. Humidity is CSV-only.
    _input_files = {}
    if input_folder.value.strip():
        _input_files["input_folder"] = input_folder.value.strip()
    if uses_absolute_humidity and ah_path.value.strip():
        _input_files["absolute_humidity_csv"] = ah_path.value.strip()
    if uses_contact_matrix and cal_mode.value == "csv" and cal_path.value.strip():
        _input_files["school_work_calendar_csv"] = cal_path.value.strip()
    if uses_mobility and mob_mode.value == "csv" and mob_path.value.strip():
        _input_files["mobility_csv"] = mob_path.value.strip()
    if (include_vax_immunity.value or uses_scheduled_transfer) and vax_mode.value == "csv" and vax_path.value.strip():
        _input_files["vaccines_csv"] = vax_path.value.strip()
    if uses_contact_matrix and _A > 1:
        if total_contact_csv_path.value.strip():
            _input_files["total_contact_matrix_csv"] = total_contact_csv_path.value.strip()
        if school_contact_csv_path.value.strip():
            _input_files["school_contact_matrix_csv"] = school_contact_csv_path.value.strip()
        if work_contact_csv_path.value.strip():
            _input_files["work_contact_matrix_csv"] = work_contact_csv_path.value.strip()
    if is_metapop and metapop_folder_input.value.strip():
        _input_files["metapop_folder"] = metapop_folder_input.value.strip()

    # --- 7. INITIAL CONDITIONS (per subpopulation) ---
    # Per-subpop population (A×R) plus per-compartment seed counts (A×R) from the
    # Step 6 tables. The first compartment is reconstructed at run time as
    # population − Σ seeds, so only the seeds are stored here.
    _age_cols_ic = param_grid_columns(age_groups, _A)
    _seed_comps = compartments[1:] if len(compartments) > 1 else []
    _saved_ic_cfg = loaded_config.get("initial_conditions", {}) or {}
    _seed_values = get_seed_values()
    _initial_conditions = {}
    for _sp in pop_subpop_names:
        _pop_arr = np.asarray(population_by_subpop.get(_sp, np.zeros((_A, _R))), dtype=float)
        _seeds = {}
        for _ci, _c in enumerate(_seed_comps):
            _data = _seed_values.get(f"{_sp}::{_c}")
            if _data is None:
                _data = default_seed_row_data(
                    _saved_ic_cfg, _sp, _c, _age_cols_ic, _R, _ci == 0
                )
            _arr = grid_to_AR_array(_data, _age_cols_ic, _A, _R)
            if bool(np.any(_arr != 0)):
                _seeds[_c] = _arr.tolist()
        _initial_conditions[_sp] = {"population": _pop_arr.tolist(), "seeds": _seeds}
    _total_pop = int(round(sum(
        np.asarray(_v["population"], dtype=float).sum()
        for _v in _initial_conditions.values()
    )))
    _risk_fractions = [float(_x) for _x in risk_fraction_inputs.value]

    # --- 8. CONFIG DICT ---
    _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    config_dict = {
        "compartments": compartments,
        "params": params_dict,
        "transitions": _transitions,
        "transition_groups": [],
        "epi_metrics": _epi_metrics,
        "schedules": _schedules,
        "age_risk": {
            "num_age_groups": _A,
            "num_risk_groups": _R,
            "risk_group_fractions": _risk_fractions,
            **({"age_groups": age_groups} if age_groups else {}),
        },
        "total_population": _total_pop,
        "initial_conditions": _initial_conditions,
        "simulation_settings": {
            "start_real_date": start_date_input.value.strip(),
            "transition_variables_to_save": _tvs,
        },
    }
    if _input_files:
        config_dict["input_files"] = _input_files

    # Per-subpopulation parameter overrides. Start from any loaded overrides
    # (authored directly in model_config.json) so the load -> rebuild -> run
    # round-trip stays lossless, then layer on per-subpop contact matrices
    # fetched in the Population & Geography tab. Both feed the metapop run path
    # and the shared factory, which read config_dict["subpop_params"].
    _subpop_params = dict(loaded_config.get("subpop_params", {}))
    if fetched_matrices_scope == "per_subpop":
        for _sp_name, _mats in fetched_contact_matrices.items():
            _entry = dict(_subpop_params.get(_sp_name, {}))
            for _mname in ("total_contact_matrix", "school_contact_matrix", "work_contact_matrix"):
                if _mname in _mats:
                    _entry[_mname] = _mats[_mname]
            _subpop_params[_sp_name] = _entry
    if _subpop_params:
        config_dict["subpop_params"] = _subpop_params

    # Per-subpopulation constant daily-vaccines override (metapop only). Kept
    # as its own top-level key rather than folded into subpop_params, since
    # daily_vaccines is a *schedule* value (read by _run_sim), not a model
    # "params" entry — subpop_params gets merged straight into params at run
    # time, which would silently misroute it.
    if is_metapop and daily_vaccines_per_subpop_toggle.value:
        _subpop_vax_values = get_subpop_vax_values()
        _subpop_daily_vaccines = {
            _sp: _subpop_vax_values.get(_sp, 0.0) for _sp in pop_subpop_names
        }
        config_dict["subpop_daily_vaccines"] = _subpop_daily_vaccines

    # --- 9. ANALYSIS METRICS ---
    _n_metrics = int(analysis_n_metrics_input.value)
    _analysis_metrics = []
    for _i in range(_n_metrics):
        _aname = analysis_metric_names.value[_i].strip() or f"metric_{_i + 1}"
        _raw = analysis_metric_tvs.value[_i]
        _atvs = _raw if isinstance(_raw, list) else [t.strip() for t in _raw.split(",") if t.strip()]
        if _atvs:
            _analysis_metrics.append({"name": _aname, "transition_variables": _atvs})
    if _analysis_metrics:
        config_dict["analysis_metrics"] = _analysis_metrics

    immunity_active = _immunity_active
    metapop_travel_config = _metapop_travel_config
    config_warnings = _config_warnings
    return config_dict, immunity_active, metapop_travel_config, config_warnings


# ---------------------------------------------------------------------------
# Step 8 — Config Preview
# ---------------------------------------------------------------------------


@app.cell
def _config_preview(config_dict, config_warnings, json, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    json_str = json.dumps(config_dict, indent=2)
    _warn_block = []
    if config_warnings:
        _warn_block.append(mo.callout(
            mo.md(
                "**Config warnings:**\n\n"
                + "\n".join(f"- {_w}" for _w in config_warnings)
            ),
            kind="warn",
        ))
    mo.vstack([
        mo.md("### Step 8 — Config Preview"),
        *_warn_block,
        mo.accordion({
            "View / download config JSON": mo.vstack([
                mo.md(f"```json\n{json_str}\n```"),
                mo.download(
                    data=json_str.encode(),
                    filename="model_config.json",
                    mimetype="application/json",
                    label="Download config JSON",
                ),
            ])
        }),
    ])
    return


# ---------------------------------------------------------------------------
# Step 9 — Run
# ---------------------------------------------------------------------------


@app.cell
def _run_button(mo):
    run_button = mo.ui.run_button(label="Run simulation")
    return (run_button,)


@app.cell
def _run_section_display(run_button, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.vstack([mo.md("### Step 9 — Run"), run_button])
    return


@app.cell
def _run_sim(
    run_button,
    main_tab,
    config_dict,
    metapop_travel_config,
    compartments,
    sim_days,
    sim_mode,
    n_reps,
    rng_seed,
    timesteps,
    start_date_input,
    transition_vars_input,
    mobility_input,
    daily_vaccines_input,
    daily_vaccines_vary_toggle,
    daily_vaccines_grid_input,
    build_notebook_schedules_input,
    build_scalar_array,
    build_compartment_init,
    read_initial_conditions,
    parse_model_config_from_dict,
    ConfigDrivenSubpopModel,
    ConfigDrivenMetapopModel,
    build_state_from_config,
    build_params_from_config,
    clt,
    flu,
    np,
    mo,
    json,
    num_age_groups,
    num_risk_groups,
    is_metapop,
    metapop_folder_input,
    loaded_schedule_dfs,
    Path,
    pd,
    age_groups,
    param_grid_columns,
    grid_to_AR_array,
):
    mo.stop(main_tab.value != "Model Builder", None)
    mo.stop(not run_button.value, mo.md(""))

    # Runs the preview simulation for Step 9. Structure of this cell:
    #   - run settings (stochastic/deterministic, reps, days, timesteps)
    #   - nested helper _build_schedules_input_for_subpop(...)
    #   - nested helper _run_once(...)         — single-population path
    #   - nested helper _run_metapop_once(...) — metapopulation path
    #   - dispatch on is_metapop, aggregate replicates, then plot + summary table
    _A = num_age_groups
    _R = num_risk_groups
    start_real_date = start_date_input.value.strip() or "2024-01-01"
    _tvs = [v.strip() for v in transition_vars_input.value.split(",") if v.strip()]
    _is_stochastic = sim_mode.value == "Stochastic"
    _transition_type = (
        clt.TransitionTypes.BINOM if _is_stochastic
        else clt.TransitionTypes.BINOM_DETERMINISTIC_NO_ROUND
    )
    _reps = int(n_reps.value) if _is_stochastic else 1
    _num_days = int(sim_days.value)
    _seed = int(rng_seed.value)
    _ts_per_day = int(timesteps.value)

    # Daily-vaccines constant value: either a single scalar broadcast to every
    # age/risk group, or (when "Vary by age/risk group" is on) a per-cell A×R
    # array read back from the grid editor.
    if daily_vaccines_vary_toggle.value:
        _age_cols = param_grid_columns(age_groups, _A)
        _daily_vaccines_value = grid_to_AR_array(
            daily_vaccines_grid_input.value, _age_cols, _A, _R
        ).tolist()
    else:
        _daily_vaccines_value = float(daily_vaccines_input.value)

    # ---- Single-population run helper ----
    def _build_schedules_input_for_subpop(
        ah_df_override=None,
        cal_df_override=None,
        mob_df_override=None,
        vax_df_override=None,
        daily_vaccines_value_override=None,
    ):
        return build_notebook_schedules_input(
            start_date=start_real_date,
            num_days=_num_days,
            absolute_humidity=0.0,  # CSV-only: the humidity df is always supplied when used
            mobility_value=float(mobility_input.value),
            daily_vaccines_value=daily_vaccines_value_override if daily_vaccines_value_override is not None
                else _daily_vaccines_value,
            num_age_groups=_A,
            num_risk_groups=_R,
            absolute_humidity_df=ah_df_override if ah_df_override is not None
                else loaded_schedule_dfs.absolute_humidity_df,
            school_work_calendar_df=cal_df_override if cal_df_override is not None
                else loaded_schedule_dfs.school_work_calendar_df,
            mobility_df=mob_df_override if mob_df_override is not None
                else loaded_schedule_dfs.mobility_df,
            daily_vaccines_df=vax_df_override if vax_df_override is not None
                else loaded_schedule_dfs.daily_vaccines_df,
        )

    def _build_subpop(schedules_input, compartment_init, seed_offset, name="aggregate_pop", epi_metric_init=None, param_overrides=None):
        _config_err = None
        _model_config = None
        _cfg = config_dict
        if param_overrides:
            _cfg = dict(config_dict)
            _cfg["params"] = {**config_dict.get("params", {}), **param_overrides}
        try:
            _model_config = parse_model_config_from_dict(
                _cfg, schedules_input=schedules_input
            )
        except Exception as _exc:
            _config_err = str(_exc)
        if _config_err is not None:
            raise RuntimeError(f"Config error: {_config_err}")
        _state = build_state_from_config(_model_config, compartment_init, epi_metric_init=epi_metric_init or {})
        _params = build_params_from_config(_model_config, num_age_groups=_A, num_risk_groups=_R)
        _settings = clt.SimulationSettings(
            timesteps_per_day=_ts_per_day,
            transition_type=_transition_type,
            start_real_date=start_real_date,
            save_daily_history=True,
            transition_variables_to_save=_tvs,
        )
        _rng = np.random.default_rng(_seed + seed_offset)
        return ConfigDrivenSubpopModel(
            model_config=_model_config,
            state_init=_state,
            params=_params,
            simulation_settings=_settings,
            RNG=_rng,
            schedules_input=schedules_input,
            name=name,
        ), _model_config

    # ---- Pre-flight shape validation ----
    def _validate_shapes():
        """Return a list of human-readable issues for param/schedule shapes vs A×R."""
        _issues = []
        _contact_matrix_params = {"total_contact_matrix", "school_contact_matrix", "work_contact_matrix"}
        for _pname, _pval in config_dict.get("params", {}).items():
            if _pname in _contact_matrix_params:
                continue
            if isinstance(_pval, list):
                try:
                    _arr = np.array(_pval)
                    if _arr.ndim == 2 and (_arr.shape[0] != _A or _arr.shape[1] != _R):
                        _issues.append(
                            f"Param **`{_pname}`**: loaded shape {list(_arr.shape)} "
                            f"does not match A={_A}, R={_R}."
                        )
                except Exception:
                    pass
        for _sched_attr, _col, _label in [
            ("mobility_df", "mobility_modifier", "mobility_modifier"),
            ("daily_vaccines_df", "daily_vaccines", "daily_vaccines"),
        ]:
            _df = getattr(loaded_schedule_dfs, _sched_attr, None)
            if _df is not None and _col in _df.columns:
                try:
                    _arr = np.array(json.loads(_df[_col].iloc[0]))
                    if _arr.shape != (_A, _R):
                        _issues.append(
                            f"Schedule **`{_label}`** CSV: row array shape {list(_arr.shape)} "
                            f"does not match A={_A}, R={_R}."
                        )
                except Exception:
                    pass
        return _issues

    _shape_issues = _validate_shapes()
    mo.stop(
        bool(_shape_issues),
        mo.callout(
            mo.md(
                f"**Shape mismatch** — the following parameters/schedules are incompatible "
                f"with A={_A}, R={_R}. They are likely carried over from the loaded config. "
                f"Switch the affected schedule source to **constant** in Step 4, "
                f"or reload a config that matches the current group counts.\n\n"
                + "\n".join(f"- {_issue}" for _issue in _shape_issues)
            ),
            kind="danger",
        ),
    )

    # ---- Single-population path ----
    if not is_metapop:
        # Initial conditions come from the Step 6 tables via config_dict
        # (population A×R per cell, minus the per-compartment seed grids).
        _ic_entry = config_dict.get("initial_conditions", {}).get("aggregate_pop", {})
        _pop_arr = np.asarray(_ic_entry.get("population", np.zeros((_A, _R))), dtype=float)
        _seed_arrays = {
            _c: np.asarray(_a, dtype=float)
            for _c, _a in (_ic_entry.get("seeds", {}) or {}).items()
            if _c in compartments
        }
        compartment_init, _overflow = build_compartment_init(
            _seed_arrays, _pop_arr, compartments)
        mo.stop(
            _overflow,
            mo.callout(mo.md("**Initial condition error:** seeded counts exceed the "
                             "population in at least one age/risk cell."),
                       kind="danger"),
        )

        def _run_once(seed_offset):
            _sched = _build_schedules_input_for_subpop()
            _subpop, _model_config = _build_subpop(_sched, compartment_init, seed_offset)
            _mixing = flu.FluMixingParams(
                travel_proportions=np.array([[1.0]]),
                num_locations=1,
            )
            _metapop = ConfigDrivenMetapopModel(
                subpop_models=[_subpop],
                mixing_params=_mixing,
                model_config=_model_config,
                travel_config=metapop_travel_config,
            )
            _metapop.simulate_until_day(_num_days)
            return {
                _c: np.array(_subpop.compartments[_c].history_vals_list).sum(axis=(1, 2))
                for _c in compartments
            }

        # Parse config once for error checking before running
        _config_parse_err = None
        try:
            _test_sched = _build_schedules_input_for_subpop()
            parse_model_config_from_dict(config_dict, schedules_input=_test_sched)
        except Exception as _exc:
            _config_parse_err = str(_exc)
        mo.stop(
            _config_parse_err is not None,
            mo.callout(mo.md(f"**Config error:** {_config_parse_err}"), kind="danger"),
        )

        sim_err = None
        histories = []
        with mo.status.spinner("Running simulation..."):
            try:
                histories = [_run_once(_rep) for _rep in range(_reps)]
            except Exception as _exc:
                sim_err = str(_exc)
        mo.stop(
            sim_err is not None,
            mo.callout(mo.md(f"**Simulation error:** {sim_err}"), kind="danger"),
        )

    # ---- Metapopulation path ----
    else:
        _folder = Path(metapop_folder_input.value.strip())
        mo.stop(
            not _folder.exists() or not _folder.is_dir(),
            mo.callout(mo.md(f"**Metapop folder not found:** {_folder}"), kind="danger"),
        )
        _metapop_cfg_path = _folder / "metapop_config.json"
        mo.stop(
            not _metapop_cfg_path.exists(),
            mo.callout(mo.md("**Missing:** `metapop_config.json` in metapop folder."), kind="danger"),
        )
        with open(_metapop_cfg_path) as _f:
            _metapop_cfg = json.load(_f)
        mo.stop(
            "subpopulations" not in _metapop_cfg or "travel_matrix" not in _metapop_cfg,
            mo.callout(mo.md("**Invalid `metapop_config.json`:** must have `subpopulations` and `travel_matrix` keys."), kind="danger"),
        )
        _sp_names = list(_metapop_cfg["subpopulations"])
        _travel_arr = np.array(_metapop_cfg["travel_matrix"], dtype=float)
        _n_subpops = len(_sp_names)

        # Shared optional schedule files
        _shared_ah_df = None
        _shared_mob_df = None
        _ah_shared_path = _folder / "absolute_humidity.csv"
        _mob_shared_path = _folder / "mobility_modifier.csv"
        if _ah_shared_path.exists():
            _shared_ah_df = pd.read_csv(_ah_shared_path)
            _shared_ah_df = _shared_ah_df.loc[:, ~_shared_ah_df.columns.str.match(r"^Unnamed")]
        if _mob_shared_path.exists():
            _shared_mob_df = pd.read_csv(_mob_shared_path)
            _shared_mob_df = _shared_mob_df.loc[:, ~_shared_mob_df.columns.str.match(r"^Unnamed")]

        def _run_metapop_once(seed_offset):
            _subpop_models = []
            _model_config_ref = None
            for _sp_idx, _sp_name in enumerate(_sp_names):
                # Load per-subpop schedule files
                _sp_cal_path = _folder / f"school_work_calendar_{_sp_name}.csv"
                _sp_vax_path = _folder / f"vaccines_{_sp_name}.csv"
                _sp_ic_path  = _folder / f"initial_conditions_{_sp_name}.json"

                _sp_cal_df = None
                _sp_vax_df = None
                if _sp_cal_path.exists():
                    _sp_cal_df = pd.read_csv(_sp_cal_path)
                    _sp_cal_df = _sp_cal_df.loc[:, ~_sp_cal_df.columns.str.match(r"^Unnamed")]
                if _sp_vax_path.exists():
                    _sp_vax_df = pd.read_csv(_sp_vax_path)
                    _sp_vax_df = _sp_vax_df.loc[:, ~_sp_vax_df.columns.str.match(r"^Unnamed")]

                _sched = _build_schedules_input_for_subpop(
                    ah_df_override=_shared_ah_df,
                    cal_df_override=_sp_cal_df,
                    mob_df_override=_shared_mob_df,
                    vax_df_override=_sp_vax_df,
                    daily_vaccines_value_override=config_dict.get(
                        "subpop_daily_vaccines", {}
                    ).get(_sp_name),
                )

                # Initial conditions: the in-notebook Step 6 tables take precedence
                # when they carry seeds, else the per-subpop folder JSON, else the
                # table's population-only (all-susceptible) state.
                _sp_epi_init = {}
                _ic_cfg = (config_dict.get("initial_conditions", {}) or {}).get(_sp_name, {})
                _has_table_seeds = bool(_ic_cfg.get("seeds"))
                _table_comp_init = read_initial_conditions(
                    config_dict, _sp_name, compartments, _A, _R)
                if _has_table_seeds and _table_comp_init is not None:
                    _sp_comp_init = _table_comp_init
                elif _sp_ic_path.exists():
                    _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                    with open(_sp_ic_path) as _f:
                        _ic = json.load(_f)
                    for _c, _arr in _ic.get("compartments", {}).items():
                        if _c in compartments:
                            _sp_comp_init[_c] = np.array(_arr, dtype=float)
                    for _m, _arr in _ic.get("epi_metrics", {}).items():
                        _sp_epi_init[_m] = np.array(_arr, dtype=float)
                elif _table_comp_init is not None:
                    _sp_comp_init = _table_comp_init
                else:
                    _sp_comp_init = {_c: build_scalar_array(0.0, _A, _R) for _c in compartments}
                    mo.stop(
                        _A > 1 or _R > 1,
                        mo.callout(mo.md(f"**Missing:** initial conditions for `{_sp_name}` — "
                                         f"seed it in Step 6 or provide "
                                         f"`initial_conditions_{_sp_name}.json`."), kind="danger"),
                    )

                _sp_param_overrides = dict(config_dict.get("subpop_params", {}).get(_sp_name, {}))
                _subpop, _mc = _build_subpop(
                    _sched, _sp_comp_init, seed_offset + _sp_idx, name=_sp_name,
                    epi_metric_init=_sp_epi_init,
                    param_overrides=_sp_param_overrides or None,
                )
                _subpop_models.append(_subpop)
                if _model_config_ref is None:
                    _model_config_ref = _mc

            _mixing = flu.FluMixingParams(
                travel_proportions=_travel_arr,
                num_locations=_n_subpops,
            )
            _metapop = ConfigDrivenMetapopModel(
                subpop_models=_subpop_models,
                mixing_params=_mixing,
                model_config=_model_config_ref,
                travel_config=metapop_travel_config,
            )
            _metapop.simulate_until_day(_num_days)

            # Aggregate histories by summing across subpops and age/risk groups
            return {
                _c: sum(
                    np.array(_sp.compartments[_c].history_vals_list).sum(axis=(1, 2))
                    for _sp in _subpop_models
                )
                for _c in compartments
            }

        sim_err = None
        histories = []
        with mo.status.spinner("Running metapopulation simulation..."):
            try:
                histories = [_run_metapop_once(_rep * _n_subpops) for _rep in range(_reps)]
            except Exception as _exc:
                sim_err = str(_exc)
        mo.stop(
            sim_err is not None,
            mo.callout(mo.md(f"**Simulation error:** {sim_err}"), kind="danger"),
        )

    return (histories,)


@app.cell
def _plot_curves(histories, compartments, sim_days, sim_mode, is_metapop, np, plt, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _num_days = int(sim_days.value)
    _days = np.arange(1, _num_days + 1)
    _is_stochastic = sim_mode.value == "Stochastic"

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for _ci, _comp in enumerate(compartments):
        _color = _colors[_ci % len(_colors)]
        if _is_stochastic and len(histories) > 1:
            _mat = np.stack([_h[_comp] for _h in histories], axis=0)
            _median = np.median(_mat, axis=0)
            _lo = np.percentile(_mat, 2.5, axis=0)
            _hi = np.percentile(_mat, 97.5, axis=0)
            for _rep in range(len(histories)):
                _ax.plot(_days, _mat[_rep], color=_color, alpha=0.15, linewidth=0.8)
            _ax.plot(_days, _median, color=_color, linewidth=2, label=f"{_comp} (median)")
            _ax.fill_between(_days, _lo, _hi, color=_color, alpha=0.2)
        else:
            _ax.plot(_days, histories[0][_comp], color=_color, linewidth=2, label=_comp)

    _ax.set_xlabel("Day")
    _ax.set_ylabel("Count")
    _title = "Epidemic Curves"
    if is_metapop:
        _title += " (aggregated across subpopulations)"
    _ax.set_title(_title)
    _ax.legend(loc="best")
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mo.vstack([mo.md("### Results — Epidemic Curves"), _fig])
    return


@app.cell
def _summary_stats(histories, compartments, np, mo, main_tab):
    mo.stop(main_tab.value != "Model Builder", None)
    _rows = []
    for _comp in compartments:
        _vals = np.stack([_h[_comp] for _h in histories], axis=0)
        _peak = np.median(np.max(_vals, axis=1))
        _peak_day = int(np.median(np.argmax(_vals, axis=1))) + 1
        _final = np.median(_vals[:, -1])
        _rows.append(f"| `{_comp}` | {_peak:,.0f} | {_peak_day} | {_final:,.0f} |")
    _table = "\n".join(_rows)
    mo.vstack([
        mo.md("### Results — Summary"),
        mo.md(
            "| Compartment | Peak value (median) | Peak day (median) | Final value (median) |\n"
            "|---|---|---|---|\n"
            f"{_table}"
        ),
    ])
    return

