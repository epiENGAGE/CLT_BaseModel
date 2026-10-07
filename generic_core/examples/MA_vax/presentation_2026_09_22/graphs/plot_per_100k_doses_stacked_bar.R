## Stacked horizontal bar chart showing, for each vaccinated age group, the
## breakdown (by hospitalized age group) of hospitalizations averted per
## 100,000 additional doses (medians only, NOT normalized -- bar lengths
## reflect the actual per-100k-doses magnitude).
##
## Run with: Rscript generic_core/examples/MA_vax/presentation_2026_09_22/graphs/plot_per_100k_doses_stacked_bar.R
## or source() it interactively / from an RStudio session.
##
## Optional CLI args (all positional, defaulting to the 70%-coverage table
## for backward compatibility): <input_csv> <output_png> <plot_title>
## e.g. for the 50%/55% coverage-target panels:
##   Rscript plot_per_100k_doses_stacked_bar.R S_A_3_50pct_per_100k_doses.csv S_A_3_50pct_per_100k_doses_stacked_bar.png "50% coverage target"

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)

## Resolve paths relative to this script's own location, so it can be run
## with `Rscript path/to/plot_per_100k_doses_stacked_bar.R` from any working directory.
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
## CONFIG
## ---------------------------------------------------------------------

ARGS <- commandArgs(trailingOnly = TRUE)
DATA_FILE <- if (length(ARGS) >= 1) ARGS[1] else file.path(SCRIPT_DIR, "S_A_3_per_100k_doses.csv")
OUTPUT_FILE <- if (length(ARGS) >= 2) ARGS[2] else file.path(SCRIPT_DIR, "S_A_3_per_100k_doses_stacked_bar.png")
PLOT_TITLE <- if (length(ARGS) >= 3) ARGS[3] else NULL
## A bare filename (no "/") is resolved relative to this script's own
## folder, so it keeps working when invoked from a different cwd; a path
## containing "/" (relative or absolute) is used as-is.
if (!grepl("/", DATA_FILE, fixed = TRUE)) {
  DATA_FILE <- file.path(SCRIPT_DIR, DATA_FILE)
}
if (!grepl("/", OUTPUT_FILE, fixed = TRUE)) {
  OUTPUT_FILE <- file.path(SCRIPT_DIR, OUTPUT_FILE)
}

# Vaccinated age groups (originally the CSV columns) shown as bars, in
# top-to-bottom order. Every group has dose data in this table set, so all
# eight bars are shown (the Aug 24 version dropped 1-4, 5-12 and 65+, whose
# cells were "--" back then).
VAX_GROUP_ORDER <- c("All", "65+", "50-64", "18-49", "13-17", "5-12", "1-4", "0")

# Hospitalized age groups (originally the CSV rows) stacked within each
# bar as colored segments, in left-to-right / legend order.
HOSP_GROUP_ORDER <- c("0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+")

HOSP_GROUP_COLORS <- c(
  "0"     = "#9467bd",  # purple
  "1-4"   = "#c564b0",
  "5-12"  = "#e0559a",
  "13-17" = "#e91e63",  # pink/magenta
  "18-49" = "#008080",  # teal
  "50-64" = "#4fa89a",
  "65+"   = "#ff7f0e"   # orange
)

## ---------------------------------------------------------------------
## Load + reshape data
## ---------------------------------------------------------------------

extract_median <- function(cell) {
  m <- regmatches(cell, regexpr("^[0-9.]+", cell))
  if (length(m) == 0) NA_real_ else as.numeric(m)
}

raw <- read_csv(DATA_FILE, show_col_types = FALSE)

dat <- raw %>%
  filter(age_group %in% HOSP_GROUP_ORDER) %>%
  select(age_group, all_of(VAX_GROUP_ORDER)) %>%
  rename(hosp_group = age_group) %>%
  pivot_longer(
    cols = -hosp_group,
    names_to = "vax_group",
    values_to = "value"
  ) %>%
  mutate(
    value = vapply(value, extract_median, numeric(1)),
    hosp_group = factor(hosp_group, levels = HOSP_GROUP_ORDER),
    vax_group = factor(vax_group, levels = rev(VAX_GROUP_ORDER))
  ) %>%
  filter(!is.na(value))

## ---------------------------------------------------------------------
## Plot
## ---------------------------------------------------------------------

p <- ggplot(dat, aes(x = value, y = vax_group, fill = hosp_group)) +
  geom_col(width = 0.7, position = position_stack(reverse = TRUE)) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_fill_manual(values = HOSP_GROUP_COLORS, breaks = HOSP_GROUP_ORDER) +
  labs(
    x = "Hospitalizations averted per 100,000 additional doses",
    y = "Age group vaccinated",
    fill = "Age group hospitalized",
    title = PLOT_TITLE
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(size = 20),
    axis.title.y = element_text(size = 20),
    plot.title = element_text(size = 20, hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18),
    legend.position = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.margin = margin(t = 10, r = 25, b = 10, l = 10)
  ) +
  guides(fill = guide_legend(nrow = 1))

print(p)
ggsave(OUTPUT_FILE, plot = p, width = 10, height = 7, dpi = 200)
