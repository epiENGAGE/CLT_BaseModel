# Graphs -- 2026-09-22 refresh

Copy of `presentation_2026_08_24/graphs/`, re-rendered against the counterfactual
tables regenerated on 2026-09-22 20:46 in
`generic_core/examples/MA_vax/counterfactual_tables_from_db_MA_vax/`.

## Regenerated here

| Figure | Script | Input |
| --- | --- | --- |
| `S_A_3_per_100k_doses_stacked_bar.png` | `plot_per_100k_doses_stacked_bar.R` | `S_A_3_per_100k_doses.csv` (copied from the table folder) |
| `S_A_3_50pct_per_100k_doses_stacked_bar.png` / `S_A_3_55pct_per_100k_doses_stacked_bar.png` | `plot_per_100k_doses_stacked_bar.R <csv> <png> <title>` | `S_A_3_50pct_per_100k_doses.csv` / `S_A_3_55pct_per_100k_doses.csv` (copied from the table folder's new 50%/55%-coverage-target equivalents of S.A.3 -- separate panels, same age-group breakdown as the 70% figure) |
| `S_A_2_pct_reduction_stacked_bar.png` | `plot_pct_reduction_stacked_bar.R` | `S_A_2_pct_reduction_normalized.csv`, built by `make_pct_reduction_normalized.py` from `S_A_2_pct_reduction.csv` |
| `scenario_comparison.png` | `plot_scenarios_ts.R` | `time_series_i_to_h_iv_to_h_population_total.csv` (results explorer export, 2026-09-23; population total, `i_to_h + iv_to_h`, 2.5/97.5% band) |
| `baseline_infection_protection_no_vax_<model>.png` (4 files) | `plot_scenarios_ts.R` with CLI args | `time_series_i_to_h_iv_to_h_population_total<suffix>.csv`, one per model: 7 ages (no suffix), 7 ages `_no_transmission_multiplier`, `_single_age`, `_single_age_no_transmission_multiplier` |

Change vs. 2026-08-24, beyond the numbers: the new S.A.3 table has dose data for
1-4, 5-12 and 65+ (those cells were "--" in August, and the script dropped
them), so the per-100k-doses figure now shows all eight bars.

Added 2026-10-05: 50%/55% coverage-target panels, as separate figures
alongside the existing 70% one (not merged into a single faceted plot).
`plot_per_100k_doses_stacked_bar.R` now takes optional
`<input_csv> <output_png> <plot_title>` CLI args -- run with no args to
regenerate the original 70% figure unchanged. The 50%/55% dose-target
scenarios/multipliers were added by hand to
`run_simulations_MA_vax_param_set_stochastic.py` (not exported by the
notebook's Analysis tab), and `build_counterfactual_tables_from_db.py`
gained a `table_S_A_3_for_target()` to build their table equivalents.

Added 2026-10-05: `baseline_50_55_70_coverage.png`, a time series of
baseline vs. 50%/55%/70% coverage (all ages) incident hospitalizations --
the 70%-coverage analog of `stale_from_2026_08_24/baseline and 70 pct
vax.png`, extended to all three coverage targets. Built from
`time_series_i_to_h_iv_to_h_population_total_50_55_70.csv`, which (unlike
the other time-series CSVs here) was generated directly from the
`results` table in `simulation_output_MA_vax_param_set_stochastic`'s
Parquet output via DuckDB (median + 2.5/97.5 percentile of daily
I_to_H + IV_to_H across replicates), not exported from the results
explorer. `plot_scenarios_ts.R`'s `SCENARIO_COLORS`/`SCENARIO_LABELS` grew
entries for "50% coverage (all ages)" / "55% coverage (all ages)".
Rendered with:
```sh
Rscript plot_scenarios_ts.R \
  time_series_i_to_h_iv_to_h_population_total_50_55_70.csv \
  baseline_50_55_70_coverage.png \
  "baseline|50% coverage (all ages)|55% coverage (all ages)|70% coverage (all ages)"
```

To re-render the four model plots:

```sh
S="baseline|Infection protection only|no vax"; Y=425  # shared y-axis; >= max p_hi of all four
Rscript plot_scenarios_ts.R time_series_i_to_h_iv_to_h_population_total.csv baseline_infection_protection_no_vax_7_ages.png "$S" $Y
Rscript plot_scenarios_ts.R time_series_i_to_h_iv_to_h_population_total_no_transmission_multiplier.csv baseline_infection_protection_no_vax_7_ages_no_transmission_multiplier.png "$S" $Y
Rscript plot_scenarios_ts.R time_series_i_to_h_iv_to_h_population_total_single_age.csv baseline_infection_protection_no_vax_single_age.png "$S" $Y
Rscript plot_scenarios_ts.R time_series_i_to_h_iv_to_h_population_total_single_age_no_transmission_multiplier.csv baseline_infection_protection_no_vax_single_age_no_transmission_multiplier.png "$S" $Y
```

## Still from 2026-08-24

`stale_from_2026_08_24/` holds the August time-series export and the four
scenario PNGs rendered from it ("baseline and no vax", "baseline and all VE",
...). Each was a run of `plot_scenarios_ts.R` with a different
`SELECTED_SCENARIOS` / `REPORTED_FILE`, saved under a new name. Only the
baseline vs 70% coverage version has been re-rendered so far.
