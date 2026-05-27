"""
build_notebook.py
=================

Assembles model_builder_notebook.py from its section files.

Run from the repo root:

    python generic_core/examples/build_notebook.py

Section files are NOT standalone scripts — they reference `app` which is
defined in the header written by this assembler. Edit a section file, then
run this script to regenerate model_builder_notebook.py.

Assembly order must respect data-flow dependencies:
  _nb_shared → _nb_analysis_metric_defs → _nb_entry
    → _nb_model_builder → _nb_shared_factory → _nb_fitting
      → _nb_forecast → _nb_export
  _nb_analysis  (depends on _nb_shared_factory + _nb_analysis_metric_defs)
  _nb_docs      (depends on _nb_shared for `mo`)
"""

from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "model_builder_notebook.py"

HEADER = '''\
"""
model_builder_notebook.py
=========================

** GENERATED FILE — DO NOT EDIT DIRECTLY **

This file is assembled from section files by build_notebook.py.
Edit the relevant section file instead, then rebuild:

    python generic_core/examples/build_notebook.py

Section files (all in generic_core/examples/):
  _nb_shared.py               — imports and helper functions
  _nb_analysis_metric_defs.py — analysis metric definition widgets
  _nb_entry.py                — tab selector, output directory, autosave
  _nb_model_builder.py        — Model Builder tab (Steps 0–10)
  _nb_shared_factory.py       — shared model factory functions
  _nb_fitting.py              — Fitting tab
  _nb_forecast.py             — Forecast tab
  _nb_export.py               — Export tab
  _nb_analysis.py             — Analysis tab
  _nb_docs.py                 — Documentation tab

If you edited cells in the marimo browser UI, sync changes back to the
section files first:

    python generic_core/examples/split_notebook.py

Interactive marimo notebook for building, visualising, and running
config-driven epidemic models.

Run with::

    marimo run generic_core/examples/model_builder_notebook.py

Supported rate templates
------------------------
- ``constant_param``
- ``param_product``
- ``immunity_modulated``
- ``force_of_infection``
- ``force_of_infection_travel``

Scope note
----------
This notebook supports single-population and metapopulation models, with
configurable age and risk groups. For multi-age/risk-group models, contact
matrices are embedded inline in the config JSON; vaccines and mobility can
be supplied as CSV files or as constant scalar values.

Metapopulation folder conventions
----------------------------------
Required files:
  metapop_config.json          keys: subpopulations (ordered list of names), travel_matrix (NxN list of lists)

Optional shared files (all subpops):
  absolute_humidity.csv        cols: date, absolute_humidity
  mobility_modifier.csv        cols: day_of_week, mobility_modifier (JSON A×R array)

Optional per-subpop files ({name} = subpop name):
  school_work_calendar_{name}.csv   cols: date, is_school_day, is_work_day
  vaccines_{name}.csv               cols: date, daily_vaccines (JSON A×R array)
  initial_conditions_{name}.json    keys: compartments {name: A×R list}, epi_metrics {name: A×R list}
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

'''

FOOTER = '''\

if __name__ == "__main__":
    app.run()
'''

SECTIONS = [
    "_nb_shared.py",
    "_nb_analysis_metric_defs.py",
    "_nb_entry.py",
    "_nb_model_builder.py",
    "_nb_shared_factory.py",
    "_nb_fitting.py",
    "_nb_forecast.py",
    "_nb_export.py",
    "_nb_analysis.py",
    "_nb_docs.py",
]


def _strip_file_header(text: str) -> str:
    """Remove the leading comment block (# lines) from a section file."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or lines[i].strip() == ""):
        i += 1
    return "".join(lines[i:])


def build() -> None:
    parts = [HEADER]
    for name in SECTIONS:
        path = HERE / name
        if not path.exists():
            raise FileNotFoundError(f"Section file not found: {path}")
        content = _strip_file_header(path.read_text(encoding="utf-8"))
        parts.append(content)
        if not content.endswith("\n\n"):
            parts.append("\n")

    parts.append(FOOTER)
    result = "".join(parts)
    OUT.write_text(result, encoding="utf-8")

    cell_count = result.count("\n@app.cell\n")
    print(f"Assembled {len(SECTIONS)} sections, {cell_count} cells → {OUT}")


if __name__ == "__main__":
    build()
