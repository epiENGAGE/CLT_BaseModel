# Graphs -- 2026-09-22 refresh

Copy of `presentation_2026_08_24/graphs/`, re-rendered against the counterfactual
tables regenerated on 2026-09-22 20:46 in
`generic_core/examples/MA_vax/counterfactual_tables_from_db_MA_vax/`.

## Regenerated here

| Figure | Script | Input |
| --- | --- | --- |
| `S_A_3_per_100k_doses_stacked_bar.png` | `plot_per_100k_doses_stacked_bar.R` | `S_A_3_per_100k_doses.csv` (copied from the table folder) |
| `S_A_2_pct_reduction_stacked_bar.png` | `plot_pct_reduction_stacked_bar.R` | `S_A_2_pct_reduction_normalized.csv`, built by `make_pct_reduction_normalized.py` from `S_A_2_pct_reduction.csv` |
| `scenario_comparison.png` | `plot_scenarios_ts.R` | `time_series_i_to_h_iv_to_h_population_total.csv` (results explorer export, 2026-09-23; population total, `i_to_h + iv_to_h`, 2.5/97.5% band) |
| `baseline_infection_protection_no_vax_<model>.png` (4 files) | `plot_scenarios_ts.R` with CLI args | `time_series_i_to_h_iv_to_h_population_total<suffix>.csv`, one per model: 7 ages (no suffix), 7 ages `_no_transmission_multiplier`, `_single_age`, `_single_age_no_transmission_multiplier` |

Change vs. 2026-08-24, beyond the numbers: the new S.A.3 table has dose data for
1-4, 5-12 and 65+ (those cells were "--" in August, and the script dropped
them), so the per-100k-doses figure now shows all eight bars.

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
