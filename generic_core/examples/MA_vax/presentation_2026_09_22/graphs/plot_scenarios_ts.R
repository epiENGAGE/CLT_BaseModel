## Plot simulated incident hospitalizations by scenario, styled after the
## MA vax presentation figure (dashed colored medians + shaded uncertainty
## bands, with reported hospitalizations overlaid as a solid black line).
##
## Run with: Rscript generic_core/examples/MA_vax/presentation_2026_09_22/graphs/plot_scenarios_ts.R
## or source() it interactively / from an RStudio session.
##
## Optional positional arguments override the CONFIG defaults below, so one
## script can render several models / scenario sets:
##   Rscript plot_scenarios_ts.R [DATA_CSV] [OUTPUT_PNG] [SCENARIOS] [Y_MAX]
## DATA_CSV / OUTPUT_PNG are relative to this script's folder (or absolute);
## SCENARIOS is a "|"-separated list, e.g. "baseline|Infection protection only|no vax".
## Y_MAX fixes the top of the y-axis, so several figures share one scale.

library(ggplot2)
library(dplyr)
library(readr)
library(scales)

## Resolve paths relative to this script's own location, so it can be run
## with `Rscript path/to/plot_scenarios_ts.R` from any working directory.
get_script_dir <- function() {
  file_arg <- grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg[1]))))
  }
  frame_file <- sys.frames()[[1]]$ofile
  if (!is.null(frame_file)) {
    return(dirname(normalizePath(frame_file)))
  }
  getwd()  # fall back to cwd when run interactively (e.g. line-by-line)
}
SCRIPT_DIR <- get_script_dir()

## ---------------------------------------------------------------------
## CONFIG - edit these to control what gets plotted
## ---------------------------------------------------------------------

# Which scenarios to show, in legend order. Must match the "scenario"
# column of the input CSV. Comment out lines to drop a scenario.
SELECTED_SCENARIOS <- c(
  "baseline",
  # "no vax",
  "70% coverage (all ages)"
  # "High VE",
  # "Low VE"
)

# Colors keyed by scenario name (any scenario not listed falls back to a
# default ggplot hue).
SCENARIO_COLORS <- c(
  "baseline"                = "#1f77b4",  # blue
  "no vax"                  = "#2ca02c",  # green
  "50% coverage (all ages)" = "#ff7f0e",  # orange
  "55% coverage (all ages)" = "#9467bd",  # purple
  "70% coverage (all ages)" = "#d62728",  # red
  "High VE"                 = "#9467bd",  # purple
  "Low VE"                  = "#ff7f0e",  # orange
  "Infection protection only" = "#8c564b" # brown
)

# Human-readable labels for the legend.
SCENARIO_LABELS <- c(
  "baseline"                = "Baseline",
  "no vax"                  = "No Vaccination",
  "50% coverage (all ages)" = "50% Coverage",
  "55% coverage (all ages)" = "55% Coverage",
  "70% coverage (all ages)" = "70% Coverage",
  "High VE"                 = "High VE",
  "Low VE"                  = "Low VE",
  "Infection protection only" = "Infection Protection Only"
)

SIM_START_DATE <- as.Date("2025-09-01")  # day 1 of the simulation

DATA_FILE <- file.path(SCRIPT_DIR, "time_series_i_to_h_iv_to_h_population_total.csv")

# Reported hospitalization overlay (set to NULL to skip it).
# REPORTED_FILE <- file.path(SCRIPT_DIR, "../../data/hospitalizations_ts/MA_flu_daily_hospitalizations_total.csv")
REPORTED_FILE <- NULL

OUTPUT_FILE <- file.path(SCRIPT_DIR, "scenario_comparison.png")

# Top of the y-axis; NA lets ggplot pick it from the data.
Y_MAX <- NA

## Command-line overrides (see header).
resolve_path <- function(path) {
  if (grepl("^(/|~)", path)) path.expand(path) else file.path(SCRIPT_DIR, path)
}
cli_args <- commandArgs(trailingOnly = TRUE)
if (length(cli_args) >= 1) DATA_FILE <- resolve_path(cli_args[1])
if (length(cli_args) >= 2) OUTPUT_FILE <- resolve_path(cli_args[2])
if (length(cli_args) >= 3) SELECTED_SCENARIOS <- strsplit(cli_args[3], "|", fixed = TRUE)[[1]]
if (length(cli_args) >= 4) Y_MAX <- as.numeric(cli_args[4])

## ---------------------------------------------------------------------
## Load + prep simulation data
## ---------------------------------------------------------------------

sim <- read_csv(DATA_FILE, show_col_types = FALSE) %>%
  filter(scenario %in% SELECTED_SCENARIOS) %>%
  mutate(
    scenario = factor(scenario, levels = SELECTED_SCENARIOS),
    date = SIM_START_DATE + (day - 1)
  )

missing_scenarios <- setdiff(SELECTED_SCENARIOS, unique(as.character(sim$scenario)))
if (length(missing_scenarios) > 0) {
  stop("Scenario(s) not found in ", DATA_FILE, ": ",
       paste(missing_scenarios, collapse = ", "))
}

## ---------------------------------------------------------------------
## Load reported hospitalizations (optional overlay)
## ---------------------------------------------------------------------

reported <- NULL
if (!is.null(REPORTED_FILE) && file.exists(REPORTED_FILE)) {
  reported <- read_csv(REPORTED_FILE, show_col_types = FALSE) %>%
    mutate(date = as.Date(Date, format = "%m/%d/%y")) %>%
    filter(date >= min(sim$date), date <= max(sim$date))
}

## ---------------------------------------------------------------------
## Plot
## ---------------------------------------------------------------------

legend_breaks <- c(SELECTED_SCENARIOS, if (!is.null(reported)) "Reported Hospitalization")
legend_colors <- c(SCENARIO_COLORS, "Reported Hospitalization" = "black")[legend_breaks]
legend_labels <- c(SCENARIO_LABELS, "Reported Hospitalization" = "Reported Hospitalization")[legend_breaks]

p <- ggplot() +
  geom_ribbon(
    data = sim,
    aes(x = date, ymin = p_lo, ymax = p_hi, fill = scenario),
    alpha = 0.2
  ) +
  geom_line(
    data = sim,
    aes(x = date, y = median_value, color = scenario),
    linetype = "solid",
    linewidth = 0.9
  ) +
  scale_color_manual(
    values = legend_colors,
    labels = legend_labels,
    breaks = legend_breaks
  ) +
  scale_fill_manual(
    values = SCENARIO_COLORS[SELECTED_SCENARIOS],
    labels = SCENARIO_LABELS[SELECTED_SCENARIOS],
    breaks = SELECTED_SCENARIOS
  ) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
  scale_y_continuous(labels = comma) +
  labs(x = NULL, y = "Incident hospitalizations", color = NULL, fill = NULL) +
  guides(fill = "none") +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 22),
    axis.text.y = element_text(size = 22),
    axis.title.x = element_text(size = 24),
    axis.title.y = element_text(size = 24),
    legend.text = element_text(size = 22),
    legend.title = element_text(size = 20),
    plot.title = element_text(size = 24),
    plot.subtitle = element_text(size = 22),
    plot.caption = element_text(size = 10),
    strip.text = element_text(size = 12),
    legend.position = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", color = "grey80"),
    panel.grid.minor = element_blank()
  )

if (!is.null(reported)) {
  p <- p +
    geom_line(
      data = reported,
      aes(x = date, y = total, color = "Reported Hospitalization", group = 1)
    ) +
    geom_point(
      data = reported,
      aes(x = date, y = total, color = "Reported Hospitalization"),
      size = 0.8
    )
}

# coord_cartesian (not scale limits) so ribbons above Y_MAX are clipped
# rather than dropped.
if (!is.na(Y_MAX)) {
  p <- p + coord_cartesian(ylim = c(0, Y_MAX))
}

print(p)
ggsave(OUTPUT_FILE, plot = p, width = 12, height = 7, dpi = 200)
