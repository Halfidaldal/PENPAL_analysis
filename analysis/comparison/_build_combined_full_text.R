#!/usr/bin/env Rscript
# Build combined full-text CSV across conditions.
# Produces analysis/comparison/combined_data_2.csv with conversation_id and full_story
# pulled from each condition's interim stories_full_text_filtered.csv.
# Run with: Rscript analysis/comparison/_build_combined_full_text.R

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, here)

folder_path <- here::here("data")
file_names <- list.files(
  folder_path,
  pattern = "*stories_full_text_filtered.csv",
  full.names = TRUE,
  recursive = TRUE
)

excluded_story_ids <- c(
  "conv_ed575a06c11d42358e3eeb7826d2f959",
  "conv_a79338efb1384551affc0d7597822b0f"
)

combined_data <- file_names %>%
  lapply(read.csv) %>%
  lapply(function(df) df %>% select("conversation_id", "full_story")) %>%
  bind_rows() %>%
  filter(!conversation_id %in% excluded_story_ids)

out_path <- here::here("analysis", "comparison", "combined_data_2.csv")
write.csv(combined_data, out_path, row.names = FALSE)
message(sprintf("Wrote %d rows to %s", nrow(combined_data), out_path))
