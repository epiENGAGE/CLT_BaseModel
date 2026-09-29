"""Side-by-side summary of the current-vaccination-rate fit with and without the
fitted transmission multiplier m(t), per age group and total. Both start with
no initial immunity:

  With m(t)      model_config_MA_vax.json + fitted_params_MA_vax.json
  Without m(t)   model_config_MA_vax.json + fitted_params_MA_vax_no_transmission_multiplier.json
                 (m(t) = 1 throughout; beta_baseline, humidity_impact,
                 seed_scale_E and IHR_scale refit)

Same metrics and formatting as the Aug-2026 three-model summary
(archive/2026-08-high-vax-rates/build_summary_three_models_MA_vax.py, whose
per-model summarize() is reused here): doses, hospitalizations averted by
mechanism, hospitalizations and attack rate in baseline / no vax, effective
IHR, and the fitted beta_baseline / humidity_impact / seed_scale_E.

Writes summary_mt_vs_no_mt_MA_vax/{summary.md, summary.xlsx, summary_long.csv,
fitted_params.csv}.

    python generic_core/examples/MA_vax/build_summary_mt_vs_no_mt_MA_vax.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "archive" / "2026-08-high-vax-rates"))
import build_summary_three_models_MA_vax as three  # noqa: E402

OUT = HERE / "summary_mt_vs_no_mt_MA_vax"

MODELS = {
    "With m(t)": dict(
        config=HERE / "model_config_MA_vax.json",
        fit=HERE / "fitted_params_MA_vax.json",
        db=HERE / "simulation_output_MA_vax_param_set_stochastic" / "results_parquet"),
    "Without m(t)": dict(
        config=HERE / "model_config_MA_vax.json",
        fit=HERE / "fitted_params_MA_vax_no_transmission_multiplier.json",
        db=HERE / "simulation_output_MA_vax_no_transmission_multiplier_param_set_stochastic"
        / "results_parquet"),
}


def main() -> None:
    results, params = {}, {}
    for label, spec in MODELS.items():
        results[label], params[label] = three.summarize(label, spec)

    metrics = list(next(iter(results.values())))
    tables = {m: pd.DataFrame({lab: results[lab][m] for lab in MODELS},
                              index=pd.Index(three.ROWS, name="Age group")) for m in metrics}
    ptab = pd.DataFrame(params)
    ptab.index.name = "Parameter"

    long = pd.concat([t.reset_index().melt(id_vars="Age group", var_name="Model", value_name="Value")
                      .assign(Metric=m) for m, t in tables.items()], ignore_index=True)
    long = long[["Metric", "Model", "Age group", "Value"]]

    OUT.mkdir(exist_ok=True)
    long.to_csv(OUT / "summary_long.csv", index=False, encoding="utf-8-sig")
    ptab.to_csv(OUT / "fitted_params.csv", encoding="utf-8-sig")

    # one sheet per metric; the .xlsx stores age groups as text (a CSV opened
    # in Excel turns "1-4"/"5-12" into dates)
    with pd.ExcelWriter(OUT / "summary.xlsx", engine="openpyxl") as xw:
        assert len(three.SHEETS) == len(tables)
        for sheet, t in zip(three.SHEETS, tables.values()):
            t.to_excel(xw, sheet_name=sheet)
        ptab.to_excel(xw, sheet_name="Fitted parameters")
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value)) for c in col) + 2

    md = ["# Current-vaccination-rate fits: with vs. without the transmission multiplier m(t)", "",
          "Median [95% interval] across posterior draws (one deterministic simulation per draw). "
          "Both models start with no initial immunity and use the current vaccination schedule "
          "(`model_config_MA_vax.json`).",
          "", "- **With m(t)**: the fit with the 14-day-knot transmission multiplier (`fitted_params_MA_vax.json`).",
          "- **Without m(t)**: m(t) = 1 throughout; beta_baseline, humidity_impact, seed_scale_E and IHR_scale "
          "refit (`fitted_params_MA_vax_no_transmission_multiplier.json`).",
          "", "Hospitalizations averted: *infection protection only* = no vaccination → infection-protection-only "
          "scenario; *hospitalization reduction* = infection-protection-only → full baseline; *total* = no vaccination → full "
          "baseline (the sum of the two). Percentages are of the no-vaccination hospitalizations. "
          "Attack rate = (N − S(T) − SV(T)) / N. Effective IHR = fitted IHR_scale × I_to_H_prop "
          "(unvaccinated) or × IV_to_H_prop (vaccinated).", ""]
    md += ["## Fitted parameters", "", three.to_md(ptab), ""]
    for m, t in tables.items():
        md += [f"## {m}", "", three.to_md(t), ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
