# =============================================================================
# Annotation x computational-metric cross-analysis (reviewer response) -- CORRECTED
# -----------------------------------------------------------------------------
# SUPERSEDES the resonance-based LLM analysis in annotation_resonance_crossanalysis.R.
#
# Correction (per metrics-reference definitions):
#   Transience[t] = decay of turn[t]'s OWN forward influence; Resonance = Novelty - Transience.
#   Resonance is therefore a FORWARD-looking measure of a turn's own persistence, NOT a
#   backward-looking, relational measure of whether a turn *elaborates on what preceded it*.
#   Using AI resonance as a proxy for "elaboration" was a category error, and is mechanically
#   confounded: if AI novelty is small by construction, AI resonance is pinned near -transience.
#
# Elaboration is operationalized here with metrics that actually match the claim:
#   (LLM-1) cross-author similarity  cos_sim(human_turn[t], LLM_turn[t+1])   -- relational
#   (LLM-2) LLM turn's OWN novelty (author_2 surprise)                       -- low = elaboration
# Human side (unchanged): human resonance (own novelty that persists forward) vs creativity.
#
# Run:  Rscript --vanilla analysis/comparison/annotation_elaboration_crossanalysis.R
# =============================================================================

suppressWarnings(suppressMessages({ library(arrow); library(ggplot2) }))

ROOT <- Sys.getenv("PENPAL_ROOT", getwd())
while (!dir.exists(file.path(ROOT, "analysis")) && dirname(ROOT) != ROOT) ROOT <- dirname(ROOT)
NOV   <- file.path(ROOT, "data", "human-ai", "processed", "novelty_scores.csv")
EMB   <- file.path(ROOT, "data", "human-ai", "processed", "story_embeddings_interaction_level.parquet")
ANNOT <- file.path(ROOT, "analysis", "annotations", "penpal_annotations_with_ids.csv")
FIG   <- file.path(ROOT, "analysis", "figures")
OUT   <- file.path(ROOT, "analysis", "comparison")

clean_id <- function(x) gsub("^\\[|\\]$|^['\"]|['\"]$", "", as.character(x))
excl <- c("conv_ed575a06c11d42358e3eeb7826d2f959", "conv_a63a08273d0a4704a7638e4cd6850225",
          "conv_0bb56093-3033-4615-bb70-ebfa4135589a", "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256",
          "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80", "conv_a79338efb1384551affc0d7597822b0f")

# ---- cross-author similarity (human -> LLM, within exchange) ----------------
e <- read_parquet(EMB); e$conversation_id <- clean_id(e$conversation_id)
e <- e[(is.na(e$complete_exchange) | e$complete_exchange == TRUE) & !(e$conversation_id %in% excl), ]
cos_sim <- function(a, b) { if (is.null(a) || is.null(b) || length(a) != length(b)) return(NA_real_)
  d <- sqrt(sum(a * a)) * sqrt(sum(b * b)); if (!is.finite(d) || d == 0) return(NA_real_); sum(a * b) / d }
e$xsim_h2a <- mapply(cos_sim, e$author_1_embedding, e$author_2_embedding)  # author_1=human, author_2=AI
xagg <- aggregate(xsim_h2a ~ conversation_id, e, function(x) mean(x, na.rm = TRUE))

# ---- resonance + own-novelty (turn-window filters as in novelty_comparison) --
d <- read.csv(NOV, stringsAsFactors = FALSE); d$conversation_id <- clean_id(d$conversation_id)
is_pe <- (d$conversation_id == "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256" & d$turn == 6) |
         (d$conversation_id == "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80" & d$turn == 7) |
         (d$conversation_id == "conv_7c23347e-6172-4c84-9fd0-45ef34290bd5" & d$turn == 3)
ce <- tolower(as.character(d$complete_exchange))
d <- d[(is.na(d$complete_exchange) | ce == "true") & !is.na(d$analysis_turn) &
       d$analysis_turn > 1 & d$analysis_turn <= 9 & !(d$conversation_id %in% excl) & !is_pe, ]
d$human_res <- d$author_1_surprise - d$author_1_transience
d$human_nov <- d$author_1_surprise
d$ai_nov    <- d$author_2_surprise
agg <- aggregate(cbind(human_res, human_nov, ai_nov) ~ conversation_id, d, function(x) mean(x, na.rm = TRUE))

# ---- ratings (mean over 2 annotators; drop reordered + pipeline-filtered) ----
ann <- read.csv(ANNOT, stringsAsFactors = FALSE)
ann <- ann[!ann$mixed_up_turns & !ann$filtered_in_pipeline, ]
items <- c("creativity_originality", "creativity_surprisingness",
           "coherence_element_consistency", "coherence_logical_progression")
rat <- aggregate(ann[items], by = list(conversation_id = ann$conversation_id), FUN = function(x) mean(x, na.rm = TRUE))
rat$creativity <- rowMeans(rat[c("creativity_originality", "creativity_surprisingness")], na.rm = TRUE)
rat$coherence  <- rowMeans(rat[c("coherence_element_consistency", "coherence_logical_progression")], na.rm = TRUE)

m <- Reduce(function(a, b) merge(a, b, by = "conversation_id"), list(xagg, agg, rat))
message(sprintf("Merged human-AI stories: %d", nrow(m)))

# ---- correlations with bootstrap CIs ----------------------------------------
boot_spear <- function(x, y, R = 5000, seed = 42) {
  set.seed(seed); ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]; n <- length(x)
  r <- suppressWarnings(cor(x, y, method = "spearman"))
  p <- suppressWarnings(cor.test(x, y, method = "spearman"))$p.value
  bs <- replicate(R, { i <- sample.int(n, n, TRUE); suppressWarnings(cor(x[i], y[i], method = "spearman")) })
  data.frame(r = r, ci_lo = quantile(bs, .025, names = FALSE), ci_hi = quantile(bs, .975, names = FALSE), p = p, n = n)
}
rows <- list(
  c("LLM cross-author sim", "coherence", "xsim_h2a"),  c("LLM cross-author sim", "creativity", "xsim_h2a"),
  c("LLM own novelty",      "coherence", "ai_nov"),    c("LLM own novelty",      "creativity", "ai_nov"),
  c("Human resonance",      "creativity", "human_res"),c("Human resonance",      "coherence",  "human_res")
)
tab <- do.call(rbind, lapply(rows, function(r) cbind(metric = r[1], rating = r[2], boot_spear(m[[r[3]]], m[[r[2]]]))))
tab$p_holm <- p.adjust(tab$p, method = "holm")

# ---- computational elaboration asymmetry (metric-level, no annotations) ------
hs <- aggregate(author_1_surprise ~ conversation_id, d, mean)
as <- aggregate(author_2_surprise ~ conversation_id, d, mean)
asym <- merge(hs, as, by = "conversation_id")
asym_test <- t.test(asym$author_1_surprise, asym$author_2_surprise, paired = TRUE)

write.csv(m,   file.path(OUT, "annotation_elaboration_perstory.csv"), row.names = FALSE)
write.csv(tab, file.path(OUT, "annotation_elaboration_correlations.csv"), row.names = FALSE)

cat("\n===== Annotation correlations (Spearman, 95% bootstrap CI) =====\n")
print(within(tab, { r<-round(r,3); ci_lo<-round(ci_lo,3); ci_hi<-round(ci_hi,3); p<-signif(p,3); p_holm<-signif(p_holm,3) }), row.names = FALSE)
cat(sprintf("\nComputational asymmetry: human novelty %.3f vs AI novelty %.3f | paired diff %.3f, p=%.1e | human>AI in %.0f%% of stories\n",
    mean(asym$author_1_surprise), mean(asym$author_2_surprise),
    mean(asym$author_1_surprise - asym$author_2_surprise), asym_test$p.value,
    100 * mean(asym$author_1_surprise > asym$author_2_surprise)))

# ---- figure: corrected coefficient plot -------------------------------------
pd <- tab
pd$label <- factor(paste0(pd$metric, "  ↔  ", pd$rating),
                   levels = rev(c("LLM cross-author sim  ↔  coherence", "LLM cross-author sim  ↔  creativity",
                                  "LLM own novelty  ↔  coherence", "LLM own novelty  ↔  creativity",
                                  "Human resonance  ↔  creativity", "Human resonance  ↔  coherence")))
pd$rating <- factor(pd$rating, levels = c("creativity", "coherence"))
p <- ggplot(pd, aes(label, r, colour = rating)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey55") +
  geom_pointrange(aes(ymin = ci_lo, ymax = ci_hi), linewidth = 1.1, fatten = 4) +
  geom_text(aes(label = sprintf("%+.2f", r)), vjust = -0.9, size = 3.5, show.legend = FALSE) +
  coord_flip() + ylim(-0.5, 0.5) +
  scale_colour_manual(values = c(creativity = "#c1440e", coherence = "#1f77b4")) +
  labs(title = "Corrected: elaboration/novelty metrics vs annotation ratings (Human-AI)",
       subtitle = sprintf("Spearman r, 95%% bootstrap CI (n = %d). All near zero: annotations do not track\nthe correctly-specified turn-level metrics. (Withdraws prior resonance-based r=+0.33.)", nrow(m)),
       x = NULL, y = "Spearman correlation", colour = "Rating") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"),
        plot.title.position = "plot", panel.grid.minor = element_blank())
ggsave(file.path(FIG, "annotation_elaboration_crossanalysis.png"), p, width = 10, height = 5.6, dpi = 300)
message("Figure: analysis/figures/annotation_elaboration_crossanalysis.png")
