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

clean_conversation_id <- function(x) {
  if (is.list(x)) {
    x <- purrr::map_chr(x, ~ paste(unlist(.x), collapse = ""))
  }
  x <- as.character(x)
  x %>%
    str_replace_all("^\\[|\\]$", "") %>%
    str_replace_all("^['\"]|['\"]$", "")
}

coerce_numeric_cols <- function(df, cols = c("turn", "interaction_count", "analysis_turn")) {
  present_cols <- intersect(cols, names(df))
  if (length(present_cols) == 0) {
    return(df)
  }

  df %>%
    mutate(across(all_of(present_cols), ~ suppressWarnings(as.numeric(.x))))
}

coerce_text_reason_cols <- function(df) {
  reason_cols <- names(df)[str_detect(names(df), "(^|_)qc_reasons$|_reasons$")]
  if (length(reason_cols) == 0) {
    return(df)
  }

  df %>%
    mutate(across(all_of(reason_cols), ~ na_if(as.character(.x), "")))
}

normalize_condition_id <- function(x) {
  as.character(x) %>%
    str_to_lower() %>%
    str_replace_all("_", "-")
}

provider_error_patterns <- c(
  "Network error while generating a response",
  "Claude adapter: no text returned from provider"
)

# Provider failures are excluded post-hoc because recomputing GPU-derived
# features is not currently available. Turn-level metrics drop only the failed
# exchange unless a full story is explicitly listed below.
provider_error_exchanges <- tibble::tribble(
  ~condition, ~conversation_id, ~turn,
  "human-ai", "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256", 6,
  "human-ai", "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80", 7,
  "human-ai", "conv_7c23347e-6172-4c84-9fd0-45ef34290bd5", 3
)

provider_error_stories <- tibble::tribble(
  ~condition, ~conversation_id,
  "ai-ai", "conv_a79338efb1384551affc0d7597822b0f"
)

is_provider_error_text <- function(x) {
  replace_na(str_detect(
    as.character(x),
    regex(paste(provider_error_patterns, collapse = "|"), ignore_case = TRUE)
  ), FALSE)
}

drop_provider_error_exchanges <- function(df) {
  if (!"conversation_id" %in% names(df) || nrow(df) == 0) {
    return(df)
  }

  df_out <- df %>%
    mutate(conversation_id = clean_conversation_id(conversation_id))

  text_cols <- intersect(
    c("author_1", "author_2", "text", "full_text", "story_text", "author_1_text", "author_2_text"),
    names(df_out)
  )
  has_error_text <- rep(FALSE, nrow(df_out))
  for (text_col in text_cols) {
    has_error_text <- has_error_text | is_provider_error_text(df_out[[text_col]])
  }

  turn_cols <- intersect(c("turn", "interaction_count"), names(df_out))
  exact_error_exchange <- rep(FALSE, nrow(df_out))
  exact_error_story <- rep(FALSE, nrow(df_out))

  condition_id <- if ("condition" %in% names(df_out)) {
    normalize_condition_id(df_out$condition)
  } else {
    rep(NA_character_, nrow(df_out))
  }

  story_keys <- tibble(
    .row = seq_len(nrow(df_out)),
    condition = condition_id,
    conversation_id = df_out$conversation_id
  )

  story_rows <- story_keys %>%
    inner_join(provider_error_stories, by = c("condition", "conversation_id")) %>%
    pull(.row)

  exact_error_story[story_rows] <- TRUE

  if (length(turn_cols) > 0) {
    turn_index <- suppressWarnings(as.numeric(df_out[[turn_cols[[1]]]]))
    if (length(turn_cols) > 1) {
      for (turn_col in turn_cols[-1]) {
        turn_index <- coalesce(turn_index, suppressWarnings(as.numeric(df_out[[turn_col]])))
      }
    }

    row_keys <- tibble(
      .row = seq_len(nrow(df_out)),
      condition = condition_id,
      conversation_id = df_out$conversation_id,
      turn = turn_index
    )

    exact_rows <- row_keys %>%
      inner_join(provider_error_exchanges, by = c("condition", "conversation_id", "turn")) %>%
      pull(.row)

    exact_error_exchange[exact_rows] <- TRUE
  }

  df_out %>%
    filter(!(has_error_text | exact_error_exchange | exact_error_story))
}

drop_provider_error_stories <- function(df) {
  if (!"conversation_id" %in% names(df) || nrow(df) == 0) {
    return(df)
  }

  bad_story_keys <- bind_rows(
    provider_error_exchanges %>% select(condition, conversation_id),
    provider_error_stories
  ) %>%
    distinct()

  df_out <- df %>%
    mutate(conversation_id = clean_conversation_id(conversation_id))

  condition_id <- if ("condition" %in% names(df_out)) {
    normalize_condition_id(df_out$condition)
  } else {
    rep(NA_character_, nrow(df_out))
  }

  story_keys <- tibble(
    .row = seq_len(nrow(df_out)),
    condition = condition_id,
    conversation_id = df_out$conversation_id
  )

  story_rows <- story_keys %>%
    inner_join(bad_story_keys, by = c("condition", "conversation_id")) %>%
    pull(.row)

  if (length(story_rows) == 0) {
    return(df_out)
  }

  df_out[-story_rows, , drop = FALSE]
}

load_interaction_metadata <- function(condition) {
  path <- here("data", condition, "interim", "interaction_level_stories_filtered.csv")

  if (!file.exists(path)) {
    warning(paste("Interaction metadata not found:", path))
    return(NULL)
  }

  read_csv(path, show_col_types = FALSE) %>%
    mutate(timestamp = if ("timestamp" %in% names(.)) as.character(timestamp) else NULL) %>%
    mutate(conversation_id = clean_conversation_id(conversation_id)) %>%
    coerce_numeric_cols()
}

attach_interaction_metadata <- function(df, condition = NULL) {
  if (!"conversation_id" %in% names(df)) {
    return(df)
  }

  if (is.null(condition)) {
    if ("condition" %in% names(df) && n_distinct(df$condition) == 1) {
      condition <- unique(df$condition)
    } else {
      return(df %>% mutate(conversation_id = clean_conversation_id(conversation_id)))
    }
  }

  metadata_df <- load_interaction_metadata(condition)
  if (is.null(metadata_df)) {
    return(df %>% mutate(conversation_id = clean_conversation_id(conversation_id)))
  }

  df <- df %>%
    mutate(conversation_id = clean_conversation_id(conversation_id)) %>%
    coerce_numeric_cols()

  key_candidates <- c("conversation_id", "turn", "interaction_count")
  join_keys <- intersect(key_candidates, intersect(names(df), names(metadata_df)))

  if ("conversation_id" %in% join_keys && length(join_keys) >= 2) {
    metadata_join <- metadata_df %>%
      distinct(!!!rlang::syms(join_keys), .keep_all = TRUE)
  } else if ("conversation_id" %in% names(df)) {
    conversation_level_cols <- intersect(
      c(
        "conversation_id", "starter", "starter_side", "starter_type",
        "author_1", "author_2", "author_1_type", "author_2_type",
        "respondent_id", "respondent_id_u1", "respondent_id_u2",
        "llm_type", "model_id"
      ),
      names(metadata_df)
    )

    metadata_join <- metadata_df %>%
      select(all_of(conversation_level_cols)) %>%
      distinct(conversation_id, .keep_all = TRUE)
    join_keys <- "conversation_id"
  } else {
    return(df)
  }

  df_joined <- df %>%
    left_join(metadata_join, by = join_keys, suffix = c("", ".meta"))

  meta_suffix_cols <- names(df_joined)[str_detect(names(df_joined), "\\.meta$")]

  for (meta_col in meta_suffix_cols) {
    base_col <- str_remove(meta_col, "\\.meta$")
    if (base_col %in% names(df_joined)) {
      cast_meta <- tryCatch(
        vctrs::vec_cast(df_joined[[meta_col]], df_joined[[base_col]]),
        error = function(e) NULL
      )

      if (!is.null(cast_meta)) {
        df_joined[[base_col]] <- dplyr::coalesce(df_joined[[base_col]], cast_meta)
      }
    } else {
      df_joined[[base_col]] <- df_joined[[meta_col]]
    }
  }

  df_joined %>%
    select(-all_of(meta_suffix_cols))
}

ensure_role_metadata <- function(df) {
  df <- df %>%
    mutate(conversation_id = clean_conversation_id(conversation_id))

  if (!"starter_side" %in% names(df) && "starter" %in% names(df)) {
    df <- df %>% mutate(starter_side = starter)
  }

  if (!"speaker_slot" %in% names(df)) {
    if ("type" %in% names(df)) {
      df <- df %>%
        mutate(
          speaker_slot = case_when(
            type %in% c("author_1", "user", "agent_1") ~ "author_1",
            type %in% c("author_2", "user2", "ai", "agent_2") ~ "author_2",
            TRUE ~ as.character(type)
          )
        )
    } else if ("agent" %in% names(df)) {
      df <- df %>%
        mutate(
          speaker_slot = case_when(
            agent %in% c("author_1", "author_2") ~ as.character(agent),
            TRUE ~ NA_character_
          )
        )
    }
  }

  if (!"speaker_type" %in% names(df) &&
      all(c("speaker_slot", "author_1_type", "author_2_type") %in% names(df))) {
    df <- df %>%
      mutate(
        speaker_type = case_when(
          speaker_slot == "author_1" ~ author_1_type,
          speaker_slot == "author_2" ~ author_2_type,
          TRUE ~ NA_character_
        )
      )
  }

  if (!"speaker_type" %in% names(df) && "type" %in% names(df)) {
    df <- df %>%
      mutate(
        speaker_type = case_when(
          type %in% c("user", "user2") ~ "human",
          type %in% c("ai", "agent_1", "agent_2") ~ "ai",
          TRUE ~ NA_character_
        )
      )
  }

  if (!"partner_type" %in% names(df) &&
      all(c("speaker_slot", "author_1_type", "author_2_type") %in% names(df))) {
    df <- df %>%
      mutate(
        partner_type = case_when(
          speaker_slot == "author_1" ~ author_2_type,
          speaker_slot == "author_2" ~ author_1_type,
          TRUE ~ NA_character_
        )
      )
  }

  if (!"partner_type" %in% names(df) &&
      all(c("condition", "speaker_type") %in% names(df))) {
    df <- df %>%
      mutate(
        partner_type = case_when(
          condition == "human-ai" & speaker_type == "human" ~ "ai",
          condition == "human-ai" & speaker_type == "ai" ~ "human",
          condition == "human-human" & !is.na(speaker_type) ~ "human",
          condition == "ai-ai" & !is.na(speaker_type) ~ "ai",
          TRUE ~ NA_character_
        )
      )
  }

  if (!"speaker_is_starter" %in% names(df) &&
      all(c("speaker_slot", "starter_side") %in% names(df))) {
    df <- df %>%
      mutate(
        speaker_is_starter = case_when(
          is.na(speaker_slot) | is.na(starter_side) ~ NA,
          TRUE ~ speaker_slot == starter_side
        )
      )
  }

  char_cols <- intersect(
    c("speaker_slot", "speaker_type", "partner_type", "starter_side", "starter_type"),
    names(df)
  )
  if (length(char_cols) > 0) {
    df <- df %>%
      mutate(across(all_of(char_cols), ~ na_if(as.character(.x), "")))
  }

  if ("analysis_turn" %in% names(df)) {
    df <- df %>%
      mutate(analysis_turn = suppressWarnings(as.numeric(analysis_turn)))
  }

  if ("complete_exchange" %in% names(df)) {
    df <- df %>%
      mutate(complete_exchange = as.logical(complete_exchange))
  }

  df
}

add_turn_index <- function(df) {
  if ("analysis_turn" %in% names(df) && any(!is.na(df$analysis_turn))) {
    return(df %>% mutate(turn_index = analysis_turn))
  }
  if ("turn" %in% names(df)) {
    return(df %>% mutate(turn_index = suppressWarnings(as.numeric(turn))))
  }
  if ("interaction_count" %in% names(df)) {
    return(df %>% mutate(turn_index = suppressWarnings(as.numeric(interaction_count))))
  }
  df
}

filter_complete_exchange_window <- function(df, min_analysis_turn = 1, max_analysis_turn = 9) {
  df_out <- drop_provider_error_exchanges(df)

  if ("complete_exchange" %in% names(df_out)) {
    df_out <- df_out %>%
      filter(is.na(complete_exchange) | complete_exchange)
  }

  if ("analysis_turn" %in% names(df_out) && any(!is.na(df_out$analysis_turn))) {
    df_out <- df_out %>%
      filter(between(analysis_turn, min_analysis_turn, max_analysis_turn))
  } else if ("turn" %in% names(df_out)) {
    df_out <- df_out %>%
      filter(between(suppressWarnings(as.numeric(turn)), min_analysis_turn, max_analysis_turn))
  } else if ("interaction_count" %in% names(df_out)) {
    df_out <- df_out %>%
      filter(between(suppressWarnings(as.numeric(interaction_count)), min_analysis_turn, max_analysis_turn))
  }

  add_turn_index(df_out)
}

derive_condition_role <- function(df) {
  df %>%
    ensure_role_metadata() %>%
    mutate(
      condition_role = case_when(
        condition == "human-ai" & !is.na(speaker_type) & speaker_type == "human" ~ "Human",
        condition == "human-ai" & !is.na(speaker_type) & speaker_type == "ai" ~ "AI",
        !is.na(speaker_is_starter) & speaker_is_starter ~ "Starter-side",
        !is.na(speaker_is_starter) & !speaker_is_starter ~ "Non-starter-side",
        speaker_slot == "author_1" ~ "Slot 1",
        speaker_slot == "author_2" ~ "Slot 2",
        TRUE ~ NA_character_
      ),
      slot_label = case_when(
        speaker_slot == "author_1" ~ "Slot 1",
        speaker_slot == "author_2" ~ "Slot 2",
        TRUE ~ NA_character_
      )
    )
}

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

  if ("timestamp" %in% names(df)) {
    df <- df %>%
      mutate(timestamp = as.character(timestamp))
  }
  
  df %>%
    mutate(
      condition = condition,
      conversation_id = if ("conversation_id" %in% names(.)) clean_conversation_id(conversation_id) else NULL
    ) %>%
    coerce_text_reason_cols() %>%
    coerce_numeric_cols()
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
  # NOTE: After standardization in Python pipeline (02_clean_dataset.py),
  # data already uses author_1/author_2. These mappings handle legacy data
  # or data loaded from raw sources.
  mappings <- list(
    "human-ai" = c(
      # Legacy/raw column names -> standardized names
      "user" = "author_1", "ai" = "author_2",
      "user_sentiment_projection" = "author_1_valence",
      "ai_sentiment_projection" = "author_2_valence",
      "user_sentiment_score" = "author_1_sentiment_score",
      "ai_sentiment_score" = "author_2_sentiment_score",
      "user_embedding" = "author_1_embedding",
      "ai_embedding" = "author_2_embedding"
    ),
    "human-human" = c(
      "user" = "author_1", "user2" = "author_2",
      "user_sentiment_projection" = "author_1_valence",
      "user2_sentiment_projection" = "author_2_valence",
      "user_sentiment_score" = "author_1_sentiment_score",
      "user2_sentiment_score" = "author_2_sentiment_score",
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
  
  # Also rename new standardized names to the analysis-friendly names
  # author_1_sentiment_projection -> author_1_valence (for consistency in R analysis)
  standard_mappings <- c(
    "author_1_sentiment_projection" = "author_1_valence",
    "author_2_sentiment_projection" = "author_2_valence"
  )
  
  # Apply standard mappings first
  for (old_name in names(standard_mappings)) {
    if (old_name %in% names(df)) {
      df <- df %>% rename(!!standard_mappings[[old_name]] := !!sym(old_name))
    }
  }
  
  # Apply condition-specific mappings for legacy data
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
                 "dyadic_sentiment_scores_simulated.parquet",
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
      df <- df %>%
        normalize_schema(cond) %>%
        attach_interaction_metadata(cond) %>%
        ensure_role_metadata() %>%
        add_turn_index()
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
  filenames <- c(
    "interaction_level_surface_metrics.parquet",
    "interaction_level_surface_metrics_simulated.parquet",
    "textdescriptives.parquet"
  )
  
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
      df <- df %>%
        normalize_schema(cond) %>%
        attach_interaction_metadata(cond) %>%
        ensure_role_metadata() %>%
        derive_condition_role() %>%
        add_turn_index()
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
  filenames <- c(
    "semantic_exploration_binned.parquet",
    "semantic_exploration_binned_simulated.parquet",
    "exploration_metrics.parquet"
  )
  
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
      df <- df %>%
        normalize_schema(cond) %>%
        attach_interaction_metadata(cond) %>%
        ensure_role_metadata() %>%
        derive_condition_role() %>%
        add_turn_index() %>%
        drop_provider_error_stories()
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

load_novelty_data <- function(conditions = c("human-ai", "human-human", "ai-ai")) {
  filenames <- c(
    "novelty_scores.csv",
    "novelty_scores_simulated.csv",
    "novelty_scores.csv"
  )

  all_data <- list()

  for (cond in conditions) {
    df <- NULL
    for (fname in filenames) {
      df <- tryCatch(
        load_condition_data(cond, fname, format = "csv"),
        warning = function(w) NULL
      )
      if (!is.null(df)) break
    }

    if (!is.null(df)) {
      df <- df %>%
        normalize_schema(cond) %>%
        attach_interaction_metadata(cond) %>%
        ensure_role_metadata() %>%
        add_turn_index()
      all_data[[cond]] <- df
    }
  }

  if (length(all_data) == 0) {
    stop("No novelty data found for any condition")
  }

  bind_rows(all_data)
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
# Cross-Condition Inference Helpers
# =============================================================================
# Helpers for the per-story unsigned dyad-asymmetry magnitude A_m(i).
# All operate on a long data frame with columns `condition` and a numeric
# `value_col` (one row per story per metric).

#' Bootstrap CIs on per-condition mean of a per-story asymmetry magnitude
#'
#' @param df Data frame with `condition` and the value column.
#' @param value_col Character name of the value column.
#' @param R Bootstrap replicates (default 10000).
#' @param ci Confidence level (default 0.95).
#' @param seed RNG seed.
bootstrap_condition_means <- function(df, value_col, R = 10000, ci = 0.95, seed = 42) {
  set.seed(seed)
  alpha <- (1 - ci) / 2

  df %>%
    dplyr::filter(!is.na(.data[[value_col]])) %>%
    dplyr::group_by(condition) %>%
    dplyr::group_modify(function(g, key) {
      x <- g[[value_col]]
      n <- length(x)
      if (n < 2) {
        return(tibble::tibble(
          n = n, mean = mean(x, na.rm = TRUE),
          ci_lower = NA_real_, ci_upper = NA_real_
        ))
      }
      boots <- replicate(R, mean(sample(x, n, replace = TRUE)))
      tibble::tibble(
        n = n,
        mean = mean(x),
        ci_lower = unname(quantile(boots, alpha)),
        ci_upper = unname(quantile(boots, 1 - alpha))
      )
    }) %>%
    dplyr::ungroup()
}

#' Pairwise effect sizes (Cliff's delta) between conditions
#'
#' Cliff's delta is a non-parametric effect size for two ordinal samples,
#' bounded in [-1, 1]. Reported alongside KW / Wilcoxon tests.
#'
#' @param df Data frame with `condition` and the value column.
#' @param value_col Character.
effect_sizes_pairwise <- function(df, value_col) {
  conds <- levels(df$condition)
  if (is.null(conds)) conds <- sort(unique(as.character(df$condition)))
  pairs_df <- utils::combn(conds, 2, simplify = FALSE)

  cliff_delta <- function(x, y) {
    x <- x[!is.na(x)]; y <- y[!is.na(y)]
    if (length(x) == 0 || length(y) == 0) return(NA_real_)
    n_more <- sum(outer(x, y, ">"))
    n_less <- sum(outer(x, y, "<"))
    (n_more - n_less) / (length(x) * length(y))
  }

  purrr::map_dfr(pairs_df, function(p) {
    a <- df[[value_col]][as.character(df$condition) == p[[1]]]
    b <- df[[value_col]][as.character(df$condition) == p[[2]]]
    tibble::tibble(
      contrast = paste(p[[1]], "vs", p[[2]]),
      n_a = sum(!is.na(a)),
      n_b = sum(!is.na(b)),
      cliff_delta = cliff_delta(a, b),
      magnitude = dplyr::case_when(
        is.na(cliff_delta)        ~ NA_character_,
        abs(cliff_delta) < 0.147  ~ "negligible",
        abs(cliff_delta) < 0.330  ~ "small",
        abs(cliff_delta) < 0.474  ~ "medium",
        TRUE                      ~ "large"
      )
    )
  })
}

#' Single-df planned contrast: HA vs (HH + AA) / 2
#'
#' Tests the article-headline claim that mixed dyads are more asymmetric than
#' the average of the two same-type conditions. Uses a Welch-style approximation
#' on the contrast variance.
#'
#' @param df Data frame with `condition` and the value column.
#' @param value_col Character.
#' @param ha_label Character label for the HA condition (default detects).
planned_contrast_HA_vs_same <- function(df, value_col,
                                        ha_label = NULL,
                                        hh_label = NULL,
                                        aa_label = NULL) {
  level_set <- if (is.factor(df$condition)) levels(df$condition) else unique(df$condition)
  pick <- function(provided, patterns) {
    if (!is.null(provided)) return(provided)
    for (p in patterns) {
      hit <- level_set[grepl(p, level_set, ignore.case = TRUE)]
      if (length(hit) >= 1) return(hit[[1]])
    }
    NA_character_
  }
  ha <- pick(ha_label, c("^Human-AI$", "human-ai", "human.?ai"))
  hh <- pick(hh_label, c("^Human-Human$", "human-human", "human.?human"))
  aa <- pick(aa_label, c("^AI-AI$", "ai-ai", "ai.?ai"))

  vals <- function(lab) df[[value_col]][as.character(df$condition) == lab]
  ha_v <- vals(ha); hh_v <- vals(hh); aa_v <- vals(aa)

  m <- function(x) mean(x, na.rm = TRUE)
  v <- function(x) var(x, na.rm = TRUE)
  n <- function(x) sum(!is.na(x))

  est <- m(ha_v) - 0.5 * (m(hh_v) + m(aa_v))
  se <- sqrt(v(ha_v) / n(ha_v) +
             0.25 * v(hh_v) / n(hh_v) +
             0.25 * v(aa_v) / n(aa_v))
  t_stat <- est / se
  df_welch <- (v(ha_v) / n(ha_v) +
               0.25 * v(hh_v) / n(hh_v) +
               0.25 * v(aa_v) / n(aa_v))^2 /
    ( (v(ha_v) / n(ha_v))^2 / (n(ha_v) - 1) +
      (0.25 * v(hh_v) / n(hh_v))^2 / (n(hh_v) - 1) +
      (0.25 * v(aa_v) / n(aa_v))^2 / (n(aa_v) - 1) )
  p <- 2 * pt(-abs(t_stat), df = df_welch)

  tibble::tibble(
    contrast = sprintf("%s vs (%s + %s)/2", ha, hh, aa),
    estimate = est,
    se = se,
    t = t_stat,
    df = df_welch,
    p = p,
    ci_lower = est - qt(0.975, df_welch) * se,
    ci_upper = est + qt(0.975, df_welch) * se
  )
}

#' Permutation test for A_m ~ condition
#'
#' Permutes `condition` labels at the unit-of-row level (one row per story).
#' Reports a two-sided p-value based on the F-statistic from a one-way ANOVA
#' on the per-row asymmetry value.
#'
#' @param df Data frame with `condition` and the value column.
#' @param value_col Character.
#' @param R Permutation replicates.
#' @param seed RNG seed.
permutation_test_A <- function(df, value_col, R = 10000, seed = 42) {
  set.seed(seed)
  d <- df[!is.na(df[[value_col]]), c("condition", value_col)]
  d$condition <- as.factor(d$condition)
  if (nlevels(d$condition) < 2 || nrow(d) < 4) {
    return(tibble::tibble(F_obs = NA_real_, p_perm = NA_real_, R = R))
  }

  f_stat <- function(values, groups) {
    fit <- aov(values ~ groups)
    summary(fit)[[1]][["F value"]][1]
  }

  obs <- f_stat(d[[value_col]], d$condition)
  null_F <- replicate(R, f_stat(d[[value_col]], sample(d$condition)))
  p_perm <- (sum(null_F >= obs, na.rm = TRUE) + 1) / (sum(!is.na(null_F)) + 1)
  tibble::tibble(F_obs = obs, p_perm = p_perm, R = R)
}

#' Directional stability of a SIGNED per-story asymmetry
#'
#' Given a per-story signed slot-difference (e.g., signed Fisher-z difference
#' for valence; signed slope difference for novelty/semexp), this helper
#' summarizes whether the signed asymmetry is *consistently in the same
#' direction* across stories within each condition.
#'
#' Reported per condition:
#'   - n, mean, sd, abs(mean) / sd  (Cohen's-d-style stability index)
#'   - sign_consistency = max(P(signed > 0), P(signed < 0))
#'   - one-sample t-test of mean(signed) against 0
#' Reported across conditions:
#'   - Brown-Forsythe variance-equality p-value (homogeneity of variances)
#'
#' Interpretation: HA-specific identity-tied directionality predicts
#'   * within-HA: high |mean|/sd, sign_consistency near 1, t-test rejects 0
#'   * within-HH/AA: |mean|/sd near 0, sign_consistency near 0.5, t-test fails to reject
#'   * variance-equality test: variances may be similar OR HH/AA larger
#'
#' @param df Data frame with `condition` and a numeric signed-difference column.
#' @param value_col Character.
directional_stability <- function(df, value_col) {
  d <- df[!is.na(df[[value_col]]), c("condition", value_col)]
  if (!nrow(d)) {
    return(list(by_condition = tibble::tibble(), variance_equality_p = NA_real_))
  }
  d$condition <- as.factor(d$condition)

  by_cond <- d %>%
    dplyr::group_by(condition) %>%
    dplyr::summarise(
      n = dplyr::n(),
      mean = mean(.data[[value_col]]),
      sd = sd(.data[[value_col]]),
      stability = abs(mean(.data[[value_col]])) / sd(.data[[value_col]]),
      sign_consistency = max(
        mean(.data[[value_col]] > 0, na.rm = TRUE),
        mean(.data[[value_col]] < 0, na.rm = TRUE)
      ),
      one_sample_t_p = tryCatch(
        t.test(.data[[value_col]])$p.value,
        error = function(e) NA_real_
      ),
      .groups = "drop"
    )

  bf_p <- if (nlevels(d$condition) >= 2 && nrow(d) >= 6) {
    tryCatch({
      med <- ave(d[[value_col]], d$condition, FUN = function(x) median(x, na.rm = TRUE))
      z <- abs(d[[value_col]] - med)
      summary(aov(z ~ d$condition))[[1]][["Pr(>F)"]][1]
    }, error = function(e) NA_real_)
  } else NA_real_

  list(by_condition = by_cond, variance_equality_p = bf_p)
}

#' Winsorize a numeric column to upper/lower percentile bounds
#'
#' Use on per-turn or per-story metric values before computing per-story A_m
#' to dampen single-outlier effects. By default clips to the 1st/99th
#' percentile within each level of `group_col` (typically condition).
#'
#' @param df Data frame.
#' @param value_col Character, numeric column to winsorize.
#' @param lower Lower percentile in [0,1] (default 0.01).
#' @param upper Upper percentile in [0,1] (default 0.99).
#' @param group_col Optional grouping column; if NULL, winsorize globally.
winsorize_metric <- function(df, value_col, lower = 0.01, upper = 0.99,
                             group_col = NULL) {
  clip_one <- function(x) {
    if (all(is.na(x))) return(x)
    qs <- stats::quantile(x, probs = c(lower, upper), na.rm = TRUE)
    pmin(pmax(x, qs[[1]]), qs[[2]])
  }
  if (is.null(group_col) || !(group_col %in% names(df))) {
    df[[value_col]] <- clip_one(df[[value_col]])
    return(df)
  }
  df %>%
    dplyr::group_by(.data[[group_col]]) %>%
    dplyr::mutate(!!value_col := clip_one(.data[[value_col]])) %>%
    dplyr::ungroup()
}

#' Structural-null permutation test for the rubber-band effect
#'
#' Pools all turn-level (slot1, slot2) valence pairs across stories and
#' conditions, randomly repartitions into pseudo-stories of `story_size`
#' turns, computes per-pseudo-story R = -beta from gap_change ~ gap_lag1,
#' and aggregates to a null distribution of mean R. If observed R is inside
#' this distribution, the rubber-band slope is a structural artifact.
#'
#' @param df Long data frame with `author_1_valence` and `author_2_valence`.
#' @param B Number of permutation replicates (default 1000).
#' @param story_size Number of turns per pseudo-story (default 10).
#' @param seed RNG seed.
#' @return List with `null_means` (length B), `pool_size`, `n_pseudo_stories`.
null_rubber_band <- function(df, B = 1000, story_size = 10, seed = 42) {
  set.seed(seed)
  pool <- df %>%
    dplyr::select(author_1_valence, author_2_valence) %>%
    tidyr::drop_na()
  n_pool <- nrow(pool)
  n_pseudo_stories <- floor(n_pool / story_size)
  if (n_pseudo_stories < 5) {
    stop("Pool too small to construct pseudo-stories of size ", story_size)
  }

  story_R <- function(slot1, slot2) {
    gap <- slot1 - slot2
    gap_lag <- c(NA_real_, gap[-length(gap)])
    gap_change <- gap - gap_lag
    ok <- !is.na(gap_lag) & !is.na(gap_change)
    if (sum(ok) < 4) return(NA_real_)
    coefs <- tryCatch(coef(lm(gap_change[ok] ~ gap_lag[ok]))[2], error = function(e) NA_real_)
    -unname(coefs)
  }

  null_means <- numeric(B)
  for (b in seq_len(B)) {
    idx <- sample.int(n_pool)
    block <- idx[seq_len(n_pseudo_stories * story_size)]
    story_id <- rep(seq_len(n_pseudo_stories), each = story_size)
    s1 <- pool$author_1_valence[block]
    s2 <- pool$author_2_valence[block]
    R_vals <- vapply(split(seq_along(block), story_id), function(rows) {
      story_R(s1[rows], s2[rows])
    }, numeric(1))
    null_means[b] <- mean(R_vals, na.rm = TRUE)
  }

  list(
    null_means = null_means,
    pool_size = n_pool,
    n_pseudo_stories = n_pseudo_stories,
    B = B
  )
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
    "interaction_level_surface_metrics.parquet",
    "novelty_scores.csv",
    "semantic_exploration_binned.parquet",
    "story_embeddings_interaction_level.parquet",
    "interaction_level_stories_filtered.csv"
  )
  
  results <- expand_grid(condition = conditions, file = common_files) %>%
    rowwise() %>%
    mutate(
      path = if (file == "interaction_level_stories_filtered.csv") {
        here("data", condition, "interim", file)
      } else {
        here("data", condition, "processed", file)
      },
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
