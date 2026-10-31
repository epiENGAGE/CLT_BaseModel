library(tidycensus)
library(tidyverse)
library(dplyr)

state_list <- c("TX")
state_list <- c("MA")
# List of geo values available at: https://walker-data.com/tidycensus/articles/basic-usage.html
geo_level <- "zcta" # "cbg" by state, "zcta" for all of US
#geo_level <- "county"

age_groups_target <- array(c(
  c(0, 4),
  c(5, 9),
  c(10, 17),
  c(18, 49),
  c(50, 64),
  c(65, 99)
), c(2, 6))
age_groups_target <- array(c(
  c(0, 0),
  c(1, 4),
  c(5, 12),
  c(13, 17),
  c(18, 49),
  c(50, 64),
  c(65, 199)
))
age_groups_target <- t(age_groups_target)

# Request a key here: https://api.census.gov/data/key_signup.html
census_api_key("e0b20b169974d7b2f103b35bcebf5cf8cb8a7f37", install=TRUE, overwrite = TRUE)
readRenviron("~/.Renviron")

var_list <- load_variables(2023, "acs5", cache = TRUE)

age_variables <- c(
  male_0y_4y = "B01001_003", 
  male_5y_9y = "B01001_004", 
  male_10y_14y = "B01001_005", 
  male_15y_17y = "B01001_006", 
  male_18y_19y = "B01001_007", 
  male_20y_20y = "B01001_008", 
  male_21y_21y = "B01001_009", 
  male_22y_24y = "B01001_010", 
  male_25y_29y = "B01001_011", 
  male_30y_34y = "B01001_012", 
  male_35y_39y = "B01001_013", 
  male_40y_44y = "B01001_014", 
  male_45y_49y = "B01001_015", 
  male_50y_54y = "B01001_016", 
  male_55y_59y = "B01001_017", 
  male_60y_61y = "B01001_018", 
  male_62y_64y = "B01001_019", 
  male_65y_66y = "B01001_020", 
  male_67y_69y = "B01001_021", 
  male_70y_74y = "B01001_022", 
  male_75y_79y = "B01001_023", 
  male_80y_84y = "B01001_024", 
  male_85y_89y = "B01001_025", 
  female_0y_4y = "B01001_027", 
  female_5y_9y = "B01001_028", 
  female_10y_14y = "B01001_029", 
  female_15y_17y = "B01001_030", 
  female_18y_19y = "B01001_031", 
  female_20y_20y = "B01001_032", 
  female_21y_21y = "B01001_033", 
  female_22y_24y = "B01001_034", 
  female_25y_29y = "B01001_035", 
  female_30y_34y = "B01001_036", 
  female_35y_39y = "B01001_037", 
  female_40y_44y = "B01001_038", 
  female_45y_49y = "B01001_039", 
  female_50y_54y = "B01001_040", 
  female_55y_59y = "B01001_041", 
  female_60y_61y = "B01001_042", 
  female_62y_64y = "B01001_043", 
  female_65y_66y = "B01001_044", 
  female_67y_69y = "B01001_045", 
  female_70y_74y = "B01001_046", 
  female_75y_79y = "B01001_047", 
  female_80y_84y = "B01001_048",
  female_85y_89y = "B01001_049")

# Load variables: population per age group and sex, per census block group
if (geo_level == "cbg") {
  number_states_fetched <- 0
  for (state in state_list) {
    df_pop_details_state <- get_acs(
      geography = geo_level, 
      variables = age_variables, 
      state = state, 
      year = 2023)
    
    if (number_states_fetched == 0) {
      df_pop_details <- df_pop_details_state
    } else {
      df_pop_details <- rbind(df_pop_details, df_pop_details_state)
    }
    
    number_states_fetched <- number_states_fetched + 1
  }
} else {
  df_pop_details <- get_acs(
    geography = geo_level, 
    variables = age_variables, 
    year = 2023)
}



# Add columns for row age group
df_pop_details <- df_pop_details %>% mutate(
  age_group=ifelse(grepl("female",variable), substring(variable,8), substring(variable,6)))

# Aggregate per census block group and age group
df_pop_age_groups <- df_pop_details %>% 
  group_by(GEOID, NAME, age_group) %>%
  summarise(population = sum(estimate))

# Get lower and upper bounds of age groups
df_pop_age_groups <- df_pop_age_groups %>% mutate(
  age_lb=gsub("y", "", substr(age_group, 1, 2)),
  age_ub=gsub("_", "", substr(age_group, nchar(age_group)-2, nchar(age_group)-1)))

df_pop_age_groups$age_lb <- as.numeric(df_pop_age_groups$age_lb)
df_pop_age_groups$age_ub <- as.numeric(df_pop_age_groups$age_ub)

df_pop_age_groups <- df_pop_age_groups %>% mutate(
  number_single_years = 1 + age_ub - age_lb)

df_pop_age_groups <- df_pop_age_groups %>% mutate(
  population_single_year = population / number_single_years)

# Create dataframe with population per single year of age
df_pop_single_year <- data.frame(matrix(ncol=4, nrow=0))
colnames(df_pop_single_year) <- c("GEOID", "NAME", "age_year", "population_single_year")

max_age <- max(df_pop_age_groups$age_ub)

for (age_year in 0:max_age) {
  df_year <- df_pop_age_groups %>% filter(age_lb <= age_year & age_year <= age_ub)
  df_year$age_year <- age_year
  
  df_pop_single_year <- rbind(df_pop_single_year, df_year[colnames(df_pop_single_year)])
}

# Population per target age group
num_age_groups_target <- dim(age_groups_target)[1]

df_pop <- data.frame(matrix(ncol=6, nrow=0))
colnames(df_pop) <- c("GEOID", "NAME", "population", "age_lb", "age_ub", "age_group")

for (i_group in (1:num_age_groups_target)) {
  if (i_group == num_age_groups_target) {
    age_ub_filter <- max_age
  } else {
    age_ub_filter <- age_groups_target[i_group,2]
  }
  age_lb_filter <- age_groups_target[i_group,1]
  
  df_i <- df_pop_single_year %>% filter(age_lb_filter <= age_year & age_year <= age_ub_filter)
  
  df_i_agg <- df_i %>% 
    group_by(GEOID, NAME) %>%
    summarise(population = sum(population_single_year)) %>%
    mutate(
      age_lb = age_lb_filter,
      age_ub = age_ub_filter,
      age_group = paste(age_lb_filter, age_ub_filter, sep="_")
    )
  
  df_pop <- rbind(df_pop, df_i_agg)
  
}

filename_out <- paste("population_per_", geo_level, "-", Sys.Date(), ".csv")
write.csv(df_pop, filename_out)

# Make numbers integers

# Need to aggregate census block groups into zip codes or some other geography

