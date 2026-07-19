---
type: project
status: active
created: 2026-07-02
updated: 2026-07-19
tags: [llm-effects, scientific-writing, information-flow, research-methods, causal-inference]
related:
  - [[endpoint-predictability]]
  - [[decision-confound-defense]]
  - [[methodology-controls]]
  - [[03-model-contamination-study]]
needs-review: true
---

# LLM Effects on Research Paper Information Flow

## Goal

Measure whether post-2022 (ChatGPT era) research papers show different information-flow signatures in how they build toward their conclusions, compared to pre-2022 papers. Use endpoint surprisal framework to test whether LLM-edited or LLM-influenced prose is more coherent, or simply more stylistically uniform.

## Hypothesis (Refined)

**H1 (Coherence claim, Hard)**: Post-2022 papers show steeper, more linear surprisal-drop curves toward conclusions because LLM editing/influence improves argumentative structure.

**H2 (Stylistic claim, Easier)**: Post-2022 papers show steeper, more linear curves, and this difference is primarily observable with post-2022 scorers; the effect diminishes with pre-2022 scorers, suggesting scorer-distribution proximity rather than text coherence.

**H3 (Discontinuity claim, Moderate)**: The post-2022 change is a discontinuous inflection in pre-existing stylistic drift (papers have been getting more uniform for decades); LLM arrival marks a clear breakpoint.

We are building controls to test H1. Without them, we measure the confound (H2). H3 is the placebo that lets us distinguish genuine acceleration from continuing trends.

## The Confound Problem

### Why It's Fatal Here (Unlike Co-Writing Paper)

In the novelty/transience paper, the confound works like this:
- **Confound claim**: "LLM-scorer finds LLM text lower-surprisal because it's in the scorer's distribution"
- **Our finding**: "LLM text is lower-novelty than human text"
- **Relationship**: Collinear but defensible by convergent measures (embedding distance, valence) that don't touch the scorer

Here, the confound is collinear in a worse way:
- **Confound claim**: "Post-2022 text sits closer to post-2022-scorer distribution, so it reads lower-surprisal and more linear"
- **Our finding**: "Post-2022 papers show lower-surprisal, more linear curves"
- **Relationship**: Identical observable. No convergent measure saved us in the design (yet). The confound and hypothesis are not distinguishable within surprisal alone.

**Core risk**: You can measure an entirely real effect and it tells you nothing about whether it's coherence or scorer proximity.

### Why Pre-2022 Scorer Is Load-Bearing

A pre-2022 base model has a frozen representation of what was "normal" before LLM text arrived. Post-2022 text—whether LLM-edited or not—will sit closer to modern norms than to 2020 norms. But the distance is the same for all 2023-2024 papers regardless of coherence.

So if H1 is true (LLM editing improves structure), then:
- Post-2022 papers should show **steeper curves with both pre-2022 and post-2022 scorers**
- The effect should hold because it's about argument architecture, not distribution proximity

If H2 is true (confound), then:
- Effect should be **large with post-2022 scorer, small or absent with pre-2022 scorer**
- The scorer-era comparison directly indexes how much is confound vs. signal

This is not just a robustness check. It's the diagnostic that separates the two theories.

## Causal Design Elements

### 1. Within-Author Panel (Strongest Identification)

Find researchers with papers on both sides of the 2022 boundary. Track **the same person's** curve features across years.

**Why this matters**: Differences out author-level style (voice, typical argument structure, subfield), field evolution (NLP wasn't using LLMs in 2019, obviously wasn't in 2024 either for everyone), and venue drift (ICLR 2019 vs 2024 had different acceptance bars).

**What remains as residual**: The within-author change that co-occurs with LLM access. Much closer to causal than cross-sectional.

**Specification**: 
```
ΔS_features ~ post_2022 + (1 | author) + (1 | venue)
   + (1 | field)
```
where features are: mean ΔS, variance of ΔS, R² of linear fit (linearity index), slope of cumulative curve.

**Sample size**: Target ~50–100 author pairs (at least 2 papers per author, one pre-2022, one post-2022). Achievable from arXiv/ACL anthology metadata.

### 2. Pre-2022 Placebo Trend

Run the same pipeline on year-binned papers from 2010–2021. Is "steeper, lower-variance curves" already a trend before ChatGPT?

**Why this matters**: If pre-2022 ΔS-variance is already declining through 2015–2021, then post-2022 is just continuation, not breakpoint. If it was flat or increasing, then post-2022 discontinuity is real.

**What you're testing**: Are papers naturally homogenizing (template-ification of IMRaD, ESL authorship rise), or is something new happening after 2022?

**Specification**:
```
ΔS_features ~ year + (1 | venue) + (1 | field)
   [run 2010–2024, inspect for slope change in 2022–2023 region]
```

**Visualization**: Plot mean ΔS-variance by year. If pre-2022 is flat/rising and 2023+ drops sharply, you have a discontinuity. If pre-2022 already declining, you have a confound.

### 3. Venue / Section Normalization

Longer papers and more standardized IMRaD sections have more text, which can mechanically flatten ΔS curves (more small increments rather than sharp jumps).

**Control**:
- Compute ΔS per section (Introduction, Methods, Results, Discussion, Conclusion) instead of per-sentence
- Normalize section position: compute ΔS as position within that section (0–1), not raw text offset
- Model: Mixed effects with section type as a factor; estimate year effect within section

**Why**: Ensures you're comparing like with like. A longer Methods section in 2024 shouldn't create the appearance of a flatter curve.

### 4. Scorer-Era Panel (Circularity Detection)

Run the entire pipeline with:
- **Pre-2021 scorer**: GPT-2 or GPT-Neo base model; trained before LLM-text prevalence
- **Transition scorer**: A 2021–2022 base model (if available; otherwise skip)
- **Post-2023 scorer**: Modern base model (Claude, GPT-J-scale or larger)

**Key result**: Effect size (Cohen's d or Δ slope) plotted against scorer age.

**Prediction under H1**: Effect size stable across scorers.
**Prediction under H2**: Effect size large post-2023, attenuates with older scorers.

**Why GPT-2 matters with a caveat**: Literature (2026 retrieval papers) shows even GPT-2 has mild source bias. So "pre-2021" is not perfectly clean, but it's cleaner than 2024. Report the dose-response; claim relative, not absolute, exoneration.

### 5. Human-Coherence Bridge (Optional, Strengthens)

On a stratified subsample (20–30 papers):
- Have 2–3 raters score overall argumentative coherence (Likert 1–5)
- Correlate ratings with your curve features (mean ΔS, variance, linearity-R²)

**If correlations are significant (ρ > 0.3, p < 0.05)**: Curve shapes track coherence beyond just scorer distribution. Licenses C interpretation.

**If not significant**: You still have a real stylometric signal (H2, the confound), but you don't get to claim it's coherence. Report it as such.

**Why this matters**: Converts vague "scorer independence" into grounded mapping: your measurement reflects something a human cares about.

## Core Measurement Specs

### Primary Statistic: Marginal Contribution

```
Δᵢ = s̄(Conclusion | C_{i−1}) − s̄(Conclusion | C_i)

where C_i = all text through section i (or paragraph i)
```

### Derived Features (Tested Per Paper)

1. **Mean ΔS**: Average incremental drop across sections (higher = steeper on average)
2. **Variance of ΔS**: Std. dev. across sections (lower = more linear, each section contributes evenly)
3. **Linearity index**: R² of regression [cumulative Σ ΔS] ~ [section position]
   - Higher R² = more linear trajectory (good fit to straight line)
   - Lower R² = erratic, nonmonotonic (some sections spike, others flat)
4. **Tail-drop ratio**: |ΔS[Discussion+Conclusion] / mean(ΔS[other sections])|
   - Measures whether ending resolution is concentrated late
5. **Cumulative slope**: Fitted slope of [Σ ΔS] vs. section index

### Per-Paper Specification

```
Paper i has feature vector: 
  [mean_ΔS, var_ΔS, linearity_R2, tail_drop, cum_slope, year, venue, author, field]

Model: 
  feature ~ post_2022 + year + (1 | venue) + (1 | field) + (1 | author_id)
  
  [within-author version: feature_2024 - feature_2019 ~ 1, grouped by author]
```

## Dataset & Workflow

### Corpora

**Primary**: arXiv papers, NLP/ML subdomains (2010–2024)
- Sample: 500–1000 papers per year, stratified by venue (NeurIPS, ICML, ICLR, ACL, EMNLP, TACL, journals)
- Text extraction: grobid + manual cleanup
- Metadata: author, year, venue, parsed IMRaD sections

**Secondary**: If within-author effect is strong, extend to 3–4 other fields (biology, economics, physics) as validation.

### Candidate Corpora (survey, 2026-07)

**Key gap**: Prior LLM-effect studies analyze *abstracts* (or reviews) with corpus-level distribution methods — not full-text information flow. Our differentiator is needing **full text + section structure + author IDs across the 2022 boundary**, which points to different sources than the field standard.

**Full text (for per-section ΔS):**
- **arXiv LaTeX source** (via [Kaggle arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) / [bulk](https://info.arxiv.org/help/bulk_data.html); tooling: [arxiv-public-datasets](https://github.com/mattbierbaum/arxiv-public-datasets)) — `\section{}` gives parse-free IMRaD + conclusion, plus version history. **Primary source for CS/ML core.**
- **[PMC Open Access](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/)** — JATS XML with native section tags (biomedical arm, no parsing). Overlaps Kobak's corpus.
- **[S2ORC](https://github.com/allenai/s2orc) / peS2o** — structured full text, many disciplines; cross-field validation.
- **[unarXive](https://arxiv.org/abs/2303.14957)** — arXiv full text pre-parsed with structure + citations.

**Author/metadata layer (for within-author panel):**
- **[OpenAlex](https://openalex.org)** — free full DB, author disambiguation (author IDs), venue/field/dates. **Recommended matching backbone.** Fallbacks: Semantic Scholar (S2AG), ORCID, Crossref.

**Prior meta-pattern studies to benchmark against:** Liang et al. *Mapping the Increasing Use of LLMs* (arXiv+bioRxiv+Nature, 2020–2024, [2404.01268](https://arxiv.org/abs/2404.01268)); Kobak et al. *excess vocabulary* (14M PubMed abstracts, [2406.07016](https://arxiv.org/abs/2406.07016)); Geng & Trotta (~1M arXiv abstracts, [2404.08627](https://arxiv.org/abs/2404.08627)); Liang et al. *Monitoring AI-Modified Content* (peer reviews, [2403.07183](https://arxiv.org/abs/2403.07183)).

### Preprocessing

1. Extract conclusion/abstract (target)
2. Tokenize sections (Introduction, Methods, Results, Discussion)
3. Compute per-section surprisal via scorer LLM
4. Aggregate to feature vectors
5. Remove outliers (papers <1k tokens, >100k tokens; papers with absent sections)

### Scorer Setup

- **Base models**: Acquire checkpoints
  - Pre-2021: GPT-2 (1.5B), GPT-Neo (1.3B) — openly available
  - Post-2023: Use Claude 3.5 or GPT-J (6B)
- **Tokenization**: Normalize across models (bits-per-byte not per-token, to account for vocab drift)
- **Batch inference**: Cache-friendly setup (same papers with all scorers)

## Hypothesis Testing Plan

### Test 1: Baseline Effect (Descriptive)

**Question**: Do post-2022 papers show steeper/more-linear curves?

**Specification**: Simple t-test or linear regression, feature ~ post_2022

**Expected outcome**: Yes, post-2022 ≠ pre-2022 for multiple features

**Reporting**: Effect sizes (Cohen's d), 95% CIs

### Test 2: Confound Isolation (Via Scorer)

**Question**: Is the effect scorer-independent?

**Specification**: Repeat Test 1 for each scorer independently; plot effect size vs. scorer year

**Expected outcome (if H1)**: Effect size stable across scorers
**Expected outcome (if H2)**: Effect size attenuates with older scorers

**Critical**: If only post-2023 scorer shows effect, that is **not a publication-ready finding**. Requires additional work.

### Test 3: Discontinuity (Placebo Trend)

**Question**: Is post-2022 a breakpoint or continuation?

**Specification**: Fit piecewise-linear model to 2010–2024 trend; test for knot at 2022

**Expected outcome**: Knot at 2022 if real LLM effect; flat or smooth trend if drift

**If no knot, what then?**: Papers were already homogenizing; 2022 is just acceleration. Reframe as "LLM accelerates a pre-existing trend" and cite the placebo as evidence.

### Test 4: Within-Author Effect (Causal)

**Question**: Do individual authors' papers change post-2022?

**Specification**: Paired t-test (same author, pre- vs post-2022); mixed-model with author random effect

**Expected outcome**: Significant within-author shift post-2022 if true effect

**Power**: Depends on N authors. With 50–100 authors, sufficient to detect medium effect.

### Test 5: Human Coherence (Optional)

**Question**: Do curve features correlate with rated coherence?

**Specification**: Spearman ρ between human ratings and features; Bayesian mixed model

**Expected outcome**: ρ > 0.3 if features track something linguistic beyond scorer bias

## Blocking Issues & Resolutions

### Issue 1: Pre-2021 Scorer Quality

GPT-2 is small; it's unclear if it has long-range context to score full papers well.

**Resolution**: 
- Test on a subsample: Do surprisal trajectories make sense (monotone, reasonable values)?
- Compare to modern scorer on same papers (correlation of feature rankings, not absolute levels)
- Document limitations; present as "robustness check" rather than "definitive signature"

### Issue 2: Metadata Fragmentation

Not all papers on arXiv have clean section boundaries or author information.

**Resolution**:
- Use parsed versions (grobid output + manual review on sample)
- Accept ~20% incompleteness; report sample sizes transparently
- Run analyses on complete-case and imputed versions; confirm robustness

### Issue 3: Publication Bias & Field Heterogeneity

ML papers post-2022 might be more mainstream/corporate-authored (less diversity in authorship language), while biology papers might not. This is orthogonal to LLM but could confound.

**Resolution**:
- Field-stratified analysis: Run Test 1–4 separately per field (NLP, vision, theory, etc.)
- Report findings per-field and pooled
- Interpret field × year interaction if large

### Issue 4: Timing Confusion

Some authors pre-submitted to arXiv in late 2022 but published in 2023. ChatGPT public release was Nov 2022. Exact timing of "when could someone have used an LLM" is blurry.

**Resolution**:
- Use submission date, not publication date
- Treat 2022-Q4 as transitional (sensivity analysis: include/exclude)
- Main analysis: 2022 vs. 2023+ (no one argues 2023 authors didn't have access)

## Timeline

- **Month 1**: Corpus assembly, metadata extraction, scorer setup
- **Month 2**: Compute features on full dataset, exploratory analysis
- **Month 3**: Run Tests 1–3 (baseline effect, scorer-era diagnostics, placebo trend); interpret
- **Month 4**: Within-author analysis (Test 4), optional human-coherence bridge; write-up

## Expected Outcomes & Publication

### If H1 (Coherence Effect is Real)

**Findings**: 
- Post-2022 papers steeper/more-linear across scorers
- Within-author effect shows significant shift
- Placebo trend shows discontinuity at 2022
- (Ideally) Human coherence correlates with features

**Narrative**: "LLM tools or influence have improved the logical flow of scientific argumentation, detectable as more systematic resolution toward conclusions."

**Venue**: TACL, TEXT, or computational-linguistics track (novel measurement of language quality; repute + novelty)

**Caution**: If human-coherence correlation is weak, soften claim to "structural flow signatures" and de-emphasize the coherence interpretation.

### If H2 (Confound Dominates)

**Findings**: 
- Effect disappears with pre-2021 scorer
- Scorer-era correlation strong (larger effect with newer scorers)
- No within-author effect or flat placebo trend

**Narrative**: "Post-2022 text is stylistically closer to modern language models due to LLM influence or author exposure, but this does not reflect improved argumentative structure."

**Venue**: Same, but reframed as "methodological caution" paper: "Why perplexity-based metrics are insufficient for detecting coherence changes in scientific writing."

**Value**: Defensive contribution, but still publishable. Shows what not to do, which helps the field.

### If H3 (Continuation of Trend)

**Findings**:
- Pre-2022 ΔS-variance already declining (papers homogenizing for years)
- Post-2022 shows acceleration but no discontinuity
- Placebo trend smooth

**Narrative**: "The adoption of LLMs has accelerated a long-run standardization of scientific prose, consistent with field-level templating (IMRaD rigidity, ESL expansion)."

**Venue**: Same, but contextualized in longue-durée trends.

## Next Actions

- [ ] Finalize arXiv corpus (year range, fields, sample strategy)
- [ ] Implement grobid + section-parsing pipeline
- [ ] Acquire GPT-2, GPT-Neo base checkpoints
- [ ] Compute surprisal on 50–100 pilot papers (all scorers)
- [ ] Visualize feature distributions and year trends
- [ ] Recruit annotators for within-author matching + human-coherence subsample
- [ ] Run full Tests 1–3; assess severity of confound
- [ ] Decide: Proceed to within-author + human-coherence, or pivot to H2 narrative?

## References

- 2026 Springer paper: Perplexity on research papers, scorer model agreement
- Model collapse / recursive training literature
- Barron et al. on transience (conceptual foundation)
- [[decision-confound-defense]] for full defense strategy
- [[methodology-controls]] for formal control specifications

---

**Last updated**: 2026-07-02
**Status**: Awaiting corpus assembly and causal-design finalization
**Critical blocker**: Decide on within-author matching feasibility before committing resources
