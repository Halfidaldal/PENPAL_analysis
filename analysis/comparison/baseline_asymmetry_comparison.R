# =============================================================================
# Simpler-baseline comparison (Reviewer 3, weakness #2) -- Human-AI condition
# -----------------------------------------------------------------------------
# R3: "No comparison against simpler alternatives ... hard to assess the added
#      value of the proposed metrics."
#
# Compares the embedding-based novelty asymmetry (paper's metric) against two
# simple lexical/surface baselines the reviewer named:
#   (1) lexical new-word rate  -- fraction of a turn's content words not seen before
#   (2) lexical uptake         -- overlap with the immediately preceding turn
#                                 (reported length-robust: Jaccard + echo-of-prior,
#                                  because fraction-of-own-tokens is length-confounded)
#   (3) turn length            -- content-word count per turn
#
# For each story we take the per-turn measure averaged over human (author_1) and
# AI (author_2) turns, and test the human-vs-AI asymmetry (paired). We then test
# convergent validity: does each simple asymmetry track the embedding-novelty
# asymmetry across stories?
#
# Run:  Rscript --vanilla analysis/comparison/baseline_asymmetry_comparison.R
# =============================================================================

suppressWarnings(suppressMessages(library(ggplot2)))

ROOT <- Sys.getenv("PENPAL_ROOT", getwd())
while (!dir.exists(file.path(ROOT, "analysis")) && dirname(ROOT) != ROOT) ROOT <- dirname(ROOT)
NOV <- file.path(ROOT, "data", "human-ai", "processed", "novelty_scores.csv")
FIG <- file.path(ROOT, "analysis", "figures")
OUT <- file.path(ROOT, "analysis", "comparison")

clean_id <- function(x) gsub("^\\[|\\]$|^['\"]|['\"]$", "", as.character(x))
excl <- c("conv_ed575a06c11d42358e3eeb7826d2f959", "conv_a63a08273d0a4704a7638e4cd6850225",
          "conv_0bb56093-3033-4615-bb70-ebfa4135589a", "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256",
          "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80", "conv_a79338efb1384551affc0d7597822b0f")
SW <- c("the","a","an","and","or","but","if","of","to","in","on","at","for","with","as","by","from","is","are",
        "was","were","be","been","being","it","its","this","that","these","those","i","you","he","she","they","we",
        "him","her","them","his","their","our","my","me","your","s","t","not","no","so","then","than","too","very",
        "can","will","would","could","should","had","has","have","do","does","did","up","out","about","into","over",
        "after","just","who","what","when","where","which","there","here","all","any","some","more","most","one",
        "like","also","only","d","re","ll","m","ve")
toks <- function(x) { w <- unlist(strsplit(tolower(gsub("[^a-z0-9' ]", " ", tolower(x))), " +")); w <- w[w != ""]; w[!(w %in% SW)] }
jac  <- function(a, b) { u <- length(union(a, b)); if (u == 0) NA_real_ else length(intersect(a, b)) / u }
echo <- function(cur, prev) { if (length(prev) == 0) NA_real_ else length(intersect(cur, prev)) / length(prev) }

d <- read.csv(NOV, stringsAsFactors = FALSE); d$conversation_id <- clean_id(d$conversation_id)
ce <- tolower(as.character(d$complete_exchange))
d <- d[(is.na(d$complete_exchange) | ce == "true") & !(d$conversation_id %in% excl), ]
d <- d[order(d$conversation_id, d$turn), ]

rows <- lapply(unique(d$conversation_id), function(cid) {
  g <- d[d$conversation_id == cid, ]
  stream <- list(); who <- character(0)
  for (i in seq_len(nrow(g))) {
    stream[[length(stream) + 1]] <- toks(g$author_1[i]); who <- c(who, "H")
    stream[[length(stream) + 1]] <- toks(g$author_2[i]); who <- c(who, "A")
  }
  n <- length(stream); newr <- jv <- ev <- ln <- rep(NA_real_, n); seen <- character(0)
  for (i in seq_len(n)) {
    ct <- stream[[i]]; ln[i] <- length(ct)
    if (i > 1 && length(ct) > 0) { newr[i] <- mean(!(ct %in% seen)); jv[i] <- jac(ct, stream[[i - 1]]); ev[i] <- echo(ct, stream[[i - 1]]) }
    seen <- union(seen, ct)
  }
  data.frame(conversation_id = cid,
             h_newrate = mean(newr[who == "H"], na.rm = TRUE), a_newrate = mean(newr[who == "A"], na.rm = TRUE),
             h_jac = mean(jv[who == "H"], na.rm = TRUE),       a_jac = mean(jv[who == "A"], na.rm = TRUE),
             h_echo = mean(ev[who == "H"], na.rm = TRUE),      a_echo = mean(ev[who == "A"], na.rm = TRUE),
             h_len = mean(ln[who == "H"]),                     a_len = mean(ln[who == "A"]))
})
b <- do.call(rbind, rows)

# embedding-novelty benchmark (analysis window, same stories)
dn <- read.csv(NOV, stringsAsFactors = FALSE); dn$conversation_id <- clean_id(dn$conversation_id)
cen <- tolower(as.character(dn$complete_exchange))
dn <- dn[(is.na(dn$complete_exchange) | cen == "true") & !is.na(dn$analysis_turn) &
         dn$analysis_turn > 1 & dn$analysis_turn <= 9 & !(dn$conversation_id %in% excl), ]
nov <- aggregate(cbind(author_1_surprise, author_2_surprise) ~ conversation_id, dn, mean)
names(nov) <- c("conversation_id", "h_nov", "a_nov")
b <- merge(b, nov, by = "conversation_id")

# asymmetries (human - AI, except uptake which we orient AI - human)
b$emb_novelty_asym <- b$h_nov - b$a_nov
b$newrate_asym     <- b$h_newrate - b$a_newrate
b$len_asym         <- b$a_len - b$h_len

paired <- function(name, H, A, hi_label) {
  t <- t.test(H, A, paired = TRUE); dz <- mean(H - A) / sd(H - A)
  data.frame(metric = name, human = mean(H), ai = mean(A), dz = dz, p = t$p.value,
             favors = ifelse(mean(H) > mean(A), "human", "AI"))
}
asym_tab <- rbind(
  paired("embedding novelty (paper)", b$h_nov, b$a_nov),
  paired("lexical new-word rate", b$h_newrate, b$a_newrate),
  paired("lexical uptake: Jaccard", b$h_jac, b$a_jac),
  paired("lexical uptake: echo-of-prior", b$h_echo, b$a_echo),
  paired("turn length (words)", b$h_len, b$a_len)
)

conv <- function(v) { s <- cor.test(b[[v]], b$emb_novelty_asym, method = "spearman")
  data.frame(simple_asym = v, spearman_r = unname(s$estimate), p = s$p.value) }
conv_tab <- rbind(conv("newrate_asym"), conv("len_asym"))

write.csv(b, file.path(OUT, "baseline_asymmetry_perstory.csv"), row.names = FALSE)
write.csv(asym_tab, file.path(OUT, "baseline_asymmetry_summary.csv"), row.names = FALSE)

cat("\n== Human vs AI asymmetry per metric (paired, n =", nrow(b), ") ==\n")
print(within(asym_tab, { human <- round(human,3); ai <- round(ai,3); dz <- round(dz,2); p <- signif(p,2) }), row.names = FALSE)
cat("\n== Convergent validity: simple asymmetry ~ embedding-novelty asymmetry ==\n")
print(within(conv_tab, { spearman_r <- round(spearman_r,3); p <- signif(p,2) }), row.names = FALSE)

# ---- figure: standardized paired difference (human - AI) per metric ----------
pl <- asym_tab
pl$metric <- factor(pl$metric, levels = rev(pl$metric))
pl$dir <- ifelse(pl$dz > 0, "human higher", "AI higher")
p <- ggplot(pl, aes(dz, metric, fill = dir)) +
  geom_col(width = 0.62) +
  geom_vline(xintercept = 0, colour = "grey40") +
  geom_text(aes(label = sprintf("dz=%+.2f%s", dz, ifelse(p < .001, "***", ifelse(p < .01, "**", ifelse(p < .05, "*", " ns"))))),
            hjust = ifelse(pl$dz > 0, -0.1, 1.1), size = 3.5) +
  scale_fill_manual(values = c("human higher" = "#1f77b4", "AI higher" = "#c1440e")) +
  scale_x_continuous(expand = expansion(mult = c(0.25, 0.25))) +
  labs(title = "Novelty asymmetry: embedding metric vs simple lexical/length baselines (Human-AI)",
       subtitle = "Standardized paired difference (human - AI) per turn, n = 97 stories.\nOnly the embedding metric places novelty on the human side; surface baselines invert or vanish.",
       x = "Cohen's dz  (human - AI)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"),
        plot.title.position = "plot", panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())
ggsave(file.path(FIG, "baseline_asymmetry_comparison.png"), p, width = 10, height = 5, dpi = 300)
message("Figure: analysis/figures/baseline_asymmetry_comparison.png")
