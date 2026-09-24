## Stacked horizontal bar chart showing, for each hospitalized age group,
## the share of averted hospitalizations attributable to vaccination of
## each age group (medians only, normalized to sum to 1 per row).
##
## Run with: Rscript generic_core/examples/MA_vax/presentation_2026_09_22/graphs/plot_pct_reduction_stacked_bar.R
## or source() it interactively / from an RStudio session.

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)

## Resolve paths relative to this script's own location, so it can be run
## with `Rscript path/to/plot_pct_reduction_stacked_bar.R` from any working directory.
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

DATA_FILE <- file.path(SCRIPT_DIR, "S_A_2_pct_reduction_normalized.csv")
OUTPUT_FILE <- file.path(SCRIPT_DIR, "S_A_2_pct_reduction_stacked_bar.png")

# Age groups in the order stacked left-to-right within each bar
# (= legend order), and the order hospitalized-group rows appear
# top-to-bottom on the y-axis.
VAX_GROUP_ORDER <- c("0", "1-4", "5-12", "13-17", "18-49", "50-64", "65+")
HOSP_GROUP_ORDER <- c("All", "65+", "50-64", "18-49", "13-17", "5-12", "1-4", "0")

VAX_GROUP_COLORS <- c(
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

dat <- read_csv(DATA_FILE, show_col_types = FALSE) %>%
  pivot_longer(
    cols = -age_group,
    names_to = "vax_group",
    values_to = "share"
  ) %>%
  mutate(
    age_group = factor(age_group, levels = rev(HOSP_GROUP_ORDER)),
    vax_group = factor(vax_group, levels = VAX_GROUP_ORDER)
  )

## ---------------------------------------------------------------------
## Plot
## ---------------------------------------------------------------------

p <- ggplot(dat, aes(x = share, y = age_group, fill = vax_group)) +
  geom_col(width = 0.7, position = position_stack(reverse = TRUE)) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = c(0, 0)) +
  scale_fill_manual(values = VAX_GROUP_COLORS, breaks = VAX_GROUP_ORDER) +
  labs(
    x = "Share of hospitalizations averted",
    y = "Age group hospitalized",
    fill = "Age group vaccinated"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 18),
    axis.title.x = element_text(size = 20),
    axis.title.y = element_text(size = 20),
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
