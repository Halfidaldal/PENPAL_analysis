# Comparison Analysis Utilities
# Standardized data loading and preprocessing for cross-condition analyses
# 
# This file provides helper functions for loading and combining data from all
# three experimental conditions: human-ai, human-human, and ai-ai.

# Required packages
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, arrow, here)

# Project root
PROJECT_ROOT <- here()

# =============================================================================
# Data Loading Functions
# =============================================================================

#' Load processed data from a specific condition
#'
#' @param condition Character: "human-ai", "human-human", or "ai-ai"
#' @param filename Character: Name of the file (without path)
#' @param format Character: "parquet" or "csv" (default: "parquet")
#' @return A tibble with the data and a 'condition' column added
load_condition_data <- function(condition, filename, format = "parquet") {
  valid_conditions <- c("human-ai", "human-human", "ai-ai")
  if (!condition %in% valid_conditions) {
    stop(paste("Invalid condition:", condition, 
               ". Must be one of:", paste(valid_conditions, collapse = ", ")))
  }
  
  path <- here("data", condition, "processed", filename)
  
  if (!file.exists(path)) {
    warning(paste("File not found:", path))
    return(NULL)
  }
  
  df <- if (format == "parquet") {
    read_parquet(path)
  } else {
    read_csv(path, show_col_types = FALSE)
  }
  
  df %>%
    mutate(condition = condition)
}

#' Load and combine data from all available conditions
#'
#' @param filename Character: Name of the file to load from each condition
#' @param conditions Character vector: Which conditions to load (default: all)
#' @param format Character: "parquet" or "csv" (default: "parquet")
#' @return A combined tibble with 'condition' column
load_all_conditions <- function(filename, 
                                 conditions = c("human-ai", "human-human", "ai-ai"),
                                 format = "parquet") {
  dfs <- conditions %>%
    map(~load_condition_data(.x, filename, format)) %>%
    compact()  # Remove NULLs from missing files
  
  if (length(dfs) == 0) {
    stop(paste("No data found for any condition. File:", filename))
  }
  
  bind_rows(dfs)
}

# =============================================================================
# Schema Normalization
# =============================================================================

#' Normalize column names to unified schema
#'
#' Converts condition-specific column names to standard author_1/author_2 schema.
#' Also handles sentiment column naming variations.
#'
#' @param df Data frame to normalize
#' @param condition Character: The condition of this data
#' @return Data frame with normalized column names
normalize_schema <- function(df, condition = NULL) {
  # Auto-detect condition if present
  if (is.null(condition) && "condition" %in% names(df)) {
    condition <- df$condition[1]
  }
  
  # Column mappings per condition
  mappings <- list(
    "human-ai" = c(
      "user" = "author_1", "ai" = "author_2",
      "user_sentiment_projection" = "author_1_valence",
      "ai_sentiment_projection" = "author_2_valence",
      "user_embedding" = "author_1_embedding",
      "ai_embedding" = "author_2_embedding"
    ),
    "human-human" = c(
      "user" = "author_1", "user2" = "author_2",
      "user_sentiment_projection" = "author_1_valence",
      "user2_sentiment_projection" = "author_2_valence",
      "user_embedding" = "author_1_embedding",
      "user2_embedding" = "author_2_embedding"
    ),
    "ai-ai" = c(
      "agent_1" = "author_1", "agent_2" = "author_2",
      "agent_1_valence" = "author_1_valence",
      "agent_2_valence" = "author_2_valence",
      "agent_1_embedding" = "author_1_embedding",
      "agent_2_embedding" = "author_2_embedding"
    )
  )
  
  if (!is.null(condition) && condition %in% names(mappings)) {
    for (old_name in names(mappings[[condition]])) {
      if (old_name %in% names(df)) {
        df <- df %>% rename(!!mappings[[condition]][[old_name]] := !!sym(old_name))
      }
    }
  }
  
  df
}

#' Load and normalize sentiment/valence data for comparison
#'
#' @param conditions Character vector: Which conditions to load
#' @return Combined, normalized tibble ready for analysis
load_valence_data <- function(conditions = c("human-ai", "human-human", "ai-ai")) {
  # Try different possible filenames
  filenames <- c("dyadic_sentiment_scores.parquet", 
                 "sentiment_scores.parquet",
                 "valence_scores.parquet")
  
  all_data <- list()
  
  for (cond in conditions) {
    df <- NULL
    for (fname in filenames) {
      df <- tryCatch(
        load_condition_data(cond, fname),
        warning = function(w) NULL
      )
      if (!is.null(df)) break
    }
    
    if (!is.null(df)) {
      df <- normalize_schema(df, cond)
      all_data[[cond]] <- df
    }
  }
  
  if (length(all_data) == 0) {
    stop("No valence data found for any condition")
  }
  
  bind_rows(all_data)
}

#' Load surface metrics data for comparison
#'
#' @param conditions Character vector: Which conditions to load
#' @return Combined tibble with surface metrics
load_surface_metrics <- function(conditions = c("human-ai", "human-human", "ai-ai")) {
  filenames <- c("surface_metrics.parquet", "textdescriptives.parquet")
  
  all_data <- list()
  
  for (cond in conditions) {
    df <- NULL
    for (fname in filenames) {
      df <- tryCatch(
        load_condition_data(cond, fname),
        warning = function(w) NULL
      )
      if (!is.null(df)) break
    }
    
    if (!is.null(df)) {
      df <- normalize_schema(df, cond)
      all_data[[cond]] <- df
    }
  }
  
  if (length(all_data) == 0) {
    stop("No surface metrics data found for any condition")
  }
  
  bind_rows(all_data)
}

#' Load semantic exploration data for comparison
#'
#' @param conditions Character vector: Which conditions to load
#' @return Combined tibble with exploration metrics
load_exploration_data <- function(conditions = c("human-ai", "human-human", "ai-ai")) {
  filenames <- c("semantic_exploration.parquet", "exploration_metrics.parquet")
  
  all_data <- list()
  
  for (cond in conditions) {
    df <- NULL
    for (fname in filenames) {
      df <- tryCatch(
        load_condition_data(cond, fname),
        warning = function(w) NULL
      )
      if (!is.null(df)) break
    }
    
    if (!is.null(df)) {
      df <- normalize_schema(df, cond)
      all_data[[cond]] <- df
    }
  }
  
  if (length(all_data) == 0) {
    stop("No exploration data found for any condition")
  }
  
  bind_rows(all_data)
}

# =============================================================================
# Factor Preparation
# =============================================================================

#' Prepare condition factor with proper ordering and labels
#'
#' @param df Data frame with 'condition' column
#' @param reference Character: Reference level for contrasts (default: "human-ai")
#' @return Data frame with condition as ordered factor
prepare_condition_factor <- function(df, reference = "human-ai") {
  levels <- c("human-ai", "human-human", "ai-ai")
  
  # Reorder so reference is first
  if (reference %in% levels) {
    levels <- c(reference, setdiff(levels, reference))
  }
  
  labels <- c(
    "human-ai" = "Human-AI",
    "human-human" = "Human-Human", 
    "ai-ai" = "AI-AI"
  )
  
  df %>%
    mutate(
      condition = factor(condition, levels = levels, labels = labels[levels])
    )
}

#' Add agent type factor for within-story comparisons
#'
#' Creates a long-format dataset with 'agent_type' distinguishing author roles.
#'
#' @param df Data frame with author_1 and author_2 columns
#' @param value_col Character: Name of the value column (e.g., "valence", "word_count")
#' @return Long-format data frame with agent_type column
pivot_to_agent_long <- function(df, value_cols) {
  # Handle multiple value columns
  if (length(value_cols) == 1) {
    author_1_col <- paste0("author_1_", value_cols)
    author_2_col <- paste0("author_2_", value_cols)
    
    df %>%
      pivot_longer(
        cols = c(all_of(author_1_col), all_of(author_2_col)),
        names_to = "agent_type",
        values_to = value_cols
      ) %>%
      mutate(
        agent_type = case_when(
          str_detect(agent_type, "author_1") ~ "author_1",
          str_detect(agent_type, "author_2") ~ "author_2"
        )
      )
  } else {
    # For multiple columns, need to reshape differently
    stop("Multiple value columns not yet supported. Use single column or reshape manually.")
  }
}

# =============================================================================
# Summary Statistics
# =============================================================================

#' Compute summary statistics by condition
#'
#' @param df Data frame with condition column
#' @param var Character: Variable name to summarize
#' @param group_vars Character vector: Additional grouping variables
#' @return Summary tibble with mean, sd, se, n, ci
summarize_by_condition <- function(df, var, group_vars = NULL) {
  group_cols <- c("condition", group_vars)
  
  df %>%
    group_by(across(all_of(group_cols))) %>%
    summarize(
      n = n(),
      mean = mean(!!sym(var), na.rm = TRUE),
      sd = sd(!!sym(var), na.rm = TRUE),
      se = sd / sqrt(n),
      ci_lower = mean - 1.96 * se,
      ci_upper = mean + 1.96 * se,
      .groups = "drop"
    )
}

# =============================================================================
# Plotting Helpers
# =============================================================================

#' Standard theme for comparison plots
theme_comparison <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5)
    )
}

#' Color palette for conditions
condition_colors <- c(
  "Human-AI" = "#1f77b4",      # Blue
  "Human-Human" = "#2ca02c",   # Green
  "AI-AI" = "#ff7f0e"          # Orange
)

#' Plot comparison across conditions
#'
#' @param summary_df Summary data frame from summarize_by_condition()
#' @param title Character: Plot title
#' @param ylab Character: Y-axis label
#' @return ggplot object
plot_condition_comparison <- function(summary_df, title = "", ylab = "Value") {
  ggplot(summary_df, aes(x = condition, y = mean, fill = condition)) +
    geom_col(width = 0.7, alpha = 0.8) +
    geom_errorbar(
      aes(ymin = ci_lower, ymax = ci_upper),
      width = 0.2,
      linewidth = 0.8
    ) +
    scale_fill_manual(values = condition_colors) +
    labs(title = title, y = ylab, x = "Condition") +
    theme_comparison() +
    theme(legend.position = "none")
}

# =============================================================================
# Model Fitting Helpers
# =============================================================================

#' Fit mixed model comparing conditions
#'
#' @param df Data frame
#' @param formula Formula for lmer model
#' @param REML Logical: Use REML estimation? (default: TRUE)
#' @return lmer model object
fit_condition_model <- function(df, formula, REML = TRUE) {
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("Package 'lme4' required. Install with install.packages('lme4')")
  }
  
  lme4::lmer(formula, data = df, REML = REML)
}

#' Get pairwise contrasts between conditions
#'
#' @param model lmer model object
#' @param spec Character: emmeans specification (default: "condition")
#' @return emmeans pairs object
get_condition_contrasts <- function(model, spec = "condition") {
  if (!requireNamespace("emmeans", quietly = TRUE)) {
    stop("Package 'emmeans' required. Install with install.packages('emmeans')")
  }
  
  emmeans::emmeans(model, specs = spec) %>%
    emmeans::pairs(adjust = "bonferroni")
}

# =============================================================================
# Report Data Availability
# =============================================================================

#' Check which processed files exist for each condition
#'
#' @return Tibble with file availability per condition
check_data_availability <- function() {
  conditions <- c("human-ai", "human-human", "ai-ai")
  
  common_files <- c(
    "dyadic_sentiment_scores.parquet",
    "sentiment_scores.parquet",
    "surface_metrics.parquet",
    "semantic_exploration.parquet",
    "embeddings.parquet",
    "cleaned_stories.csv"
  )
  
  results <- expand_grid(condition = conditions, file = common_files) %>%
    rowwise() %>%
    mutate(
      path = here("data", condition, "processed", file),
      exists = file.exists(path)
    ) %>%
    ungroup()
  
  results %>%
    select(condition, file, exists) %>%
    pivot_wider(names_from = condition, values_from = exists)
}

# Print availability on source
if (interactive()) {
  cat("Cross-condition comparison utilities loaded.\n")
  cat("Use check_data_availability() to see which data files exist.\n")
}
