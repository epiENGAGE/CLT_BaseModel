"""
split_notebook.py
=================

Splits model_builder_notebook.py back into its section files.

Run from the repo root after editing cells in the marimo browser UI:

    python generic_core/examples/split_notebook.py

This is the inverse of build_notebook.py. It reads the assembled notebook,
maps each cell to its section file by cell name, and rewrites each section
file with the current cell content from the assembled notebook.
"""

from pathlib import Path
import re

HERE = Path(__file__).parent
NB = HERE / "model_builder_notebook.py"

# Maps cell function name → section file stem
CELL_TO_SECTION: dict[str, str] = {
    # _nb_shared.py
    "_imports": "_nb_shared",
    "_helpers": "_nb_shared",
    # _nb_analysis_metric_defs.py
    "_analysis_metric_defs_ui": "_nb_analysis_metric_defs",
    "_analysis_metric_sel_state": "_nb_analysis_metric_defs",
    "_analysis_metric_plot_controls": "_nb_analysis_metric_defs",
    # _nb_entry.py
    "_main_tab_selector": "_nb_entry",
    "_output_dir_ui": "_nb_entry",
    "_output_dir": "_nb_entry",
    "_tab_header_display": "_nb_entry",
    "_autosave_config": "_nb_entry",
    # _nb_model_builder.py
    "_load_config_state": "_nb_model_builder",
    "_load_config_ui": "_nb_model_builder",
    "_clear_config_button_ui": "_nb_model_builder",
    "_load_config_parse": "_nb_model_builder",
    "_load_config_display": "_nb_model_builder",
    "_intro": "_nb_model_builder",
    "_instructions": "_nb_model_builder",
    "_population_structure_ui": "_nb_model_builder",
    "_population_structure_compute": "_nb_model_builder",
    "_population_structure_show": "_nb_model_builder",
    "_compartments_ui": "_nb_model_builder",
    "_compartments_parse": "_nb_model_builder",
    "_compartments_display": "_nb_model_builder",
    "_transition_count_ui": "_nb_model_builder",
    "_transition_forms_ui": "_nb_model_builder",
    "_transition_show": "_nb_model_builder",
    "_template_requirements": "_nb_model_builder",
    "_collect_param_names": "_nb_model_builder",
    "_params_ui": "_nb_model_builder",
    "_params_show": "_nb_model_builder",
    "_schedule_and_immunity_ui": "_nb_model_builder",
    "_epi_metric_ui": "_nb_model_builder",
    "_schedule_csv_ui": "_nb_model_builder",
    "_schedule_csv_show": "_nb_model_builder",
    "_schedule_and_immunity_show": "_nb_model_builder",
    "_diagram": "_nb_model_builder",
    "_init_ui": "_nb_model_builder",
    "_init_show": "_nb_model_builder",
    "_sim_settings_ui": "_nb_model_builder",
    "_sim_settings_show": "_nb_model_builder",
    "_build_config": "_nb_model_builder",
    "_config_preview": "_nb_model_builder",
    "_run_button": "_nb_model_builder",
    "_run_section_display": "_nb_model_builder",
    "_run_sim": "_nb_model_builder",
    "_plot_curves": "_nb_model_builder",
    "_summary_stats": "_nb_model_builder",
    # _nb_shared_factory.py
    "_shared_model_factory": "_nb_shared_factory",
    # _nb_fitting.py
    "_fitting_ui": "_nb_fitting",
    "_fitting_bounds_ui": "_nb_fitting",
    "_fitting_display": "_nb_fitting",
    "_fitting_obs_parse": "_nb_fitting",
    "_run_fitting": "_nb_fitting",
    "_fitting_autosave": "_nb_fitting",
    "_fitting_results_display": "_nb_fitting",
    # _nb_forecast.py
    "_forecast_ui": "_nb_forecast",
    "_forecast_display": "_nb_forecast",
    "_run_forecast": "_nb_forecast",
    "_forecast_autosave": "_nb_forecast",
    "_forecast_results_display": "_nb_forecast",
    # _nb_export.py
    "_export_display": "_nb_export",
    # _nb_analysis.py
    "_analysis_sub_tab": "_nb_analysis",
    "_analysis_param_catalog": "_nb_analysis",
    "_analysis_sensitivity_controls": "_nb_analysis",
    "_analysis_sensitivity_sliders": "_nb_analysis",
    "_analysis_scenario_controls": "_nb_analysis",
    "_analysis_shared_controls": "_nb_analysis",
    "_analysis_compartment_selector": "_nb_analysis",
    "_analysis_display": "_nb_analysis",
    "_analysis_define_scenarios": "_nb_analysis",
    "_analysis_results_state": "_nb_analysis",
    "_analysis_results_reader": "_nb_analysis",
    "_run_analysis": "_nb_analysis",
    "_analysis_autosave": "_nb_analysis",
    "_analysis_plot_compartments": "_nb_analysis",
    "_analysis_summary_table": "_nb_analysis",
    "_analysis_metric_defs_show": "_nb_analysis",
    "_analysis_compute_metric_series": "_nb_analysis",
    "_analysis_plot_daily_metrics": "_nb_analysis",
    "_analysis_plot_cumulative_boxplot": "_nb_analysis",
    "_analysis_plot_age_bars": "_nb_analysis",
    # _nb_docs.py
    "_docs_display": "_nb_docs",
}

SECTION_HEADERS: dict[str, str] = {
    "_nb_shared": (
        "# _nb_shared.py\n"
        "# Section: Shared imports and helper functions\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_analysis_metric_defs": (
        "# _nb_analysis_metric_defs.py\n"
        "# Section: Analysis metric definition widgets\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
        "#\n"
        "# IMPORTANT: These cells must be assembled BEFORE _nb_model_builder.py because\n"
        "# _build_config depends on analysis_n_metrics_input, analysis_metric_names,\n"
        "# and analysis_metric_tvs.\n"
    ),
    "_nb_entry": (
        "# _nb_entry.py\n"
        "# Section: Notebook entry-point UI (tab selector, output directory, autosave)\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_model_builder": (
        "# _nb_model_builder.py\n"
        "# Section: Model Builder tab cells (Steps 0-10)\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_shared_factory": (
        "# _nb_shared_factory.py\n"
        "# Section: Shared model factory functions\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_fitting": (
        "# _nb_fitting.py\n"
        "# Section: Fitting tab cells\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_forecast": (
        "# _nb_forecast.py\n"
        "# Section: Forecast tab cells\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_export": (
        "# _nb_export.py\n"
        "# Section: Export tab cell\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
    "_nb_analysis": (
        "# _nb_analysis.py\n"
        "# Section: Analysis tab cells\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
        "# (excludes metric-def widgets which are in _nb_analysis_metric_defs.py)\n"
    ),
    "_nb_docs": (
        "# _nb_docs.py\n"
        "# Section: Documentation tab cell\n"
        "# Part of model_builder_notebook.py — assembled by build_notebook.py\n"
    ),
}

SECTION_ORDER = [
    "_nb_shared",
    "_nb_analysis_metric_defs",
    "_nb_entry",
    "_nb_model_builder",
    "_nb_shared_factory",
    "_nb_fitting",
    "_nb_forecast",
    "_nb_export",
    "_nb_analysis",
    "_nb_docs",
]


def _parse_cells(text: str) -> list[tuple[str, str]]:
    """Return list of (cell_name, full_cell_text) from a notebook.

    Each element covers from `@app.cell` (or `@app.cell(...)`) through the
    blank lines that follow the return statement.
    """
    cell_start_re = re.compile(r"^@app\.cell", re.MULTILINE)
    def_re = re.compile(r"^def (\w+)\(", re.MULTILINE)

    starts = [m.start() for m in cell_start_re.finditer(text)]
    cells: list[tuple[str, str]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        chunk = text[start:end].rstrip() + "\n"
        m = def_re.search(chunk)
        name = m.group(1) if m else f"_unknown_{idx}"
        cells.append((name, chunk))
    return cells


def split() -> None:
    text = NB.read_text(encoding="utf-8")
    cells = _parse_cells(text)

    # Group cells by section
    sections: dict[str, list[str]] = {s: [] for s in SECTION_ORDER}
    unknown: list[str] = []

    for name, chunk in cells:
        section = CELL_TO_SECTION.get(name)
        if section is None:
            unknown.append(name)
            section = "_nb_analysis"  # fallback
        sections[section].append(chunk)

    if unknown:
        print(f"WARNING: unknown cell names (added to _nb_analysis): {unknown}")

    written = 0
    for section_name in SECTION_ORDER:
        path = HERE / f"{section_name}.py"
        header = SECTION_HEADERS.get(section_name, f"# {section_name}.py\n")
        body = "\n\n".join(sections[section_name])
        content = header + "\n" + body + "\n"
        path.write_text(content, encoding="utf-8")
        print(f"  {len(sections[section_name])} cells → {path.name}")
        written += 1

    print(f"\nSplit {len(cells)} cells into {written} section files.")


if __name__ == "__main__":
    split()
