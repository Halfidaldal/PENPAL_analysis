---
type: project
status: active
created: 2026-07-02
updated: 2026-07-02
tags: [model-contamination, scorer-bias, source-bias, training-data, methodology]
related:
  - [[endpoint-predictability]]
  - [[02-research-papers-llm-effect]]
  - [[decision-scorer-selection]]
  - [[methodology-controls]]
needs-review: false
---

# Model Contamination & Scorer Bias Calibration

## Goal

Quantify and isolate how training on LLM-generated text affects scorer (language model) distributions and their perception of text. Measure the magnitude and dose-response of the "circularity confound" that threatens interpretation of the research-paper LLM-effect project.

**Core insight**: The confound is not a nuisance — it's a phenomenon worth measuring. This project makes it the object of study, turning a threat into a methods foundation.

## The Problem This Solves

In the research-paper project, we need to know: when a post-2022 scorer assigns lower-surprisal to a post-2022 paper, how much is due to:
1. **True coherence**: The paper's argument actually builds more monotonically toward the conclusion
2. **Scorer proximity**: Post-2022 text sits closer to the post-2022 scorer's training distribution, so it's inherently lower-surprisal (confound)

These are collinear in the endpoint-prediction measurement. The only way to break the tie is to show how much each scorer's distribution has shifted with respect to fixed reference text.

**This project directly measures that shift.**

## Research Questions

### Primary

**RQ1**: How much does the human-minus-LLM surprisal gap change as a function of scorer training era?

```
For identical human/LLM text pairs, compute:

gap(scorer) = s̄(LLM_text | scorer) − s̄(human_text | scorer)

Plot gap vs. scorer training cutoff.

Hypothesis: gap narrows with newer scorers (LLM text becomes less surprising relative to human text to modern models)
```

**RQ2**: Can we decompose the total observed post-2022 shift in research papers into a "capability" component and a "contamination-bias" component?

```
Observed_shift = Δ_feature(post-2022 papers, post-2022 scorer)

True_shift = Δ_feature(post-2022 papers, pre-2021 scorer)  [uncontaminated baseline]

Contamination_bias = Observed_shift − True_shift
```

### Secondary

**RQ3**: Does the contamination dose-response scale linearly with amount of LLM-text in training data?

**RQ4**: Are certain scorer architectures or tokenization schemes more vulnerable to contamination bias?

**RQ5**: Does contamination bias affect relative rankings (which text is more surprising) or absolute levels (how surprised)?

## Method

### Core Design: Scorer-Era × Text-Provenance Grid

Systematically vary two dimensions:

#### Dimension 1: Scorer Training Era

Assemble base models stratified by training cutoff:

| Era | Models | Key date |
|-----|--------|----------|
| Pre-2021 | GPT-2 (1.5B), GPT-Neo (1.3B), GPT-J-6B (2021-05) | Before LLM-generation prevalence |
| Transition | GPT-J-4B (2021), early OPT (2022-05) | Boundary region (ambiguous) |
| Post-2023 | Claude 3.5 base, LLaMA-2 (2023), newer | After ChatGPT public |

**Caveat on "pre-2021"**: Literature (2026 retrieval work) shows even GPT-2 has mild source bias from overlap with earlier LLM training corpora. Claim: it's cleaner than 2024, not perfectly uncontaminated. Report it as dose-response, not binary threshold.

#### Dimension 2: Text Provenance

Assemble parallel text pairs:

| Provenance | Source |
|------------|--------|
| **Human-written** | Published papers, books, curated text corpora (pre-LLM) |
| **LLM-generated** | Outputs from GPT-3.5, GPT-4, Claude, controlled prompts |
| **Human-revised LLM** | LLM generation + human editing (hyperparameter: % edited) |

**Pairing strategy**: Use established parallel datasets:
- **SciFact / NQ pairs**: Retrieval literature has human / model-generated text pairs for same factual content
- **Human-LLM rewrites**: Commission small samples of authors rewriting LLM outputs vs. writing from scratch
- **Synthetic pairs**: Passage + high-quality paraphrase (human rewrite vs. LLM paraphrase)

**Advantage**: Controlling for content, varying only style/provenance.

### Measurement: The Contamination Index

For each (scorer, text) pair:

```
S(text | scorer) = surprisal of text under scorer

gap(scorer) = mean[S(LLM_text | scorer)] − mean[S(human_text | scorer)]

Contamination_index(scorer) = gap(scorer) − gap(baseline_scorer)
```

where baseline_scorer is GPT-2 or earliest available.

**Interpretation**:
- **Contamination_index > 0**: Newer scorer finds LLM text even lower-surprisal than old scorer did (supports RQ1)
- **Dose-response**: Plot Contamination_index vs. scorer training date; fit linear/exponential trend
- **Attribution**: If dose-response is strong, the shift is "scored effect"; weak, it's "text properties"

### Secondary Analyses

#### Analysis 1: Entropy Over Possible Continuations

Surprisal is realized; entropy is expectation. Measure both:

```
entropy(text_prefix | scorer) = H[next word | context]
```

**Hypothesis**: LLM text might have lower entropy-per-position (fewer plausible continuations; more stylistically constrained), independent of how probable the actual continuation was.

**This tests**: Is contamination about predictability of what was written, or narrowing of what could be written?

#### Analysis 2: Relative vs. Absolute Rankings

Compute:
- **Absolute surprisal**: Raw s̄(text | scorer)
- **Relative ranking**: For a set of texts {T1, T2, ...}, does scorer A rank them the same as scorer B?

**Hypothesis**: Scorer-era differences might affect magnitudes but preserve rankings (both scorers agree which papers are most surprising).

**If true**: Relative measures (Spearman correlations of surprisal rankings) are more robust than absolute-level measures.

#### Analysis 3: Specific vs. Emergent Biases

Does contamination bias live in:
- **Lexical level**: Certain tokens are consistently lower-surprisal to newer scorers in LLM text? (e.g., "intricate," "delve")
- **Structural level**: Syntax, transition words, sentence length patterns?
- **Emergent level**: Holistic text properties (coherence, formality) that arise from training-distribution proximity?

**Method**: Analyze token-level surprisal contributions; compare token distributions in human vs. LLM text; inspect high/low-ΔS passages.

### Implementation

#### Corpora

**Text 1: SciFact / NQ pairs**
- Retrieval-focused: Human summaries + model-generated alternatives for same content
- Already parallel; usable out-of-the-box
- Size: ~1k pairs per dataset

**Text 2: Human-LLM Comparison**
- Recruit 5–10 authors
- Prompt: "Write a paragraph on [topic] from scratch"
- Prompt: "Here's an LLM draft on [topic]. Revise it"
- Collect: (human original, LLM-generated draft, human-revised version)
- Vary prompts (technical, creative, expository)

**Text 3: Synthetic Paraphrases**
- Take published passages
- Paraphrase using LLMs
- Pair: (original, LLM-paraphrase)
- Size: 100–200 pairs

#### Inference Pipeline

1. **Acquire base model checkpoints** for each scorer era
2. **Tokenize and compute surprisal**:
   - Per-token: s(w_i | context)
   - Per-text: aggregate to s̄
   - Bits-per-byte normalization (account for vocab differences)
3. **Compute gap(scorer)** across all text pairs
4. **Fit contamination dose-response**: linear regression, Contamination_index ~ scorer_year
5. **Quantify uncertainty**: Bootstrap CIs around gap estimates

### Statistical Design

```
Linear model:

log_surprisal[i] ~ provenance[i] + scorer_era[i] 
                    + provenance[i]:scorer_era[i]
                    + (1 | text_pair[i])

Key terms:
  - provenance effect: main effect of human vs. LLM
  - scorer_era effect: main effect of model training cutoff
  - interaction: Does provenance bias scale with scorer era? (RQ1)
```

**Visualization**:
- **Figure 1**: Mean surprisal (human vs. LLM) by scorer era (two lines, convergence/divergence pattern)
- **Figure 2**: Contamination_index vs. scorer training date (scatter + fitted trend)
- **Figure 3**: Ranked surprisal correlation between scorer pairs (heatmap; robustness of ranking)

## Expected Outcomes

### If Contamination Is Substantial

**Pattern**:
- GPT-2 shows small or zero gap between human and LLM text
- Gap widens monotonically as scorer year increases
- By 2024, gap is ~20–30% of baseline surprisal (large effect)

**Implication**: The research-paper project must use pre-2021 scorer OR apply correction term.

**Correction formula**:
```
corrected_feature = observed_feature − Contamination_index(observed_scorer) 
                                       × (2024_slope / baseline_slope)
```

### If Contamination Is Mild

**Pattern**:
- Gap present but small (~2–5%)
- Doesn't increase monotonically with scorer era
- Relative rankings stable across scorers

**Implication**: Surprising; suggests training-data overlap is limited or that LLM-text specificity isn't in the distribution. Argue that surprisal-based measures are more robust than feared.

### If Contamination Scales with Training Data

**Pattern**:
- Models with explicit LLM-training-data (e.g., LLaMA-2 trained on LLM output) show larger gap than base models
- Can estimate "% LLM in training" from gap magnitude

**Implication**: Supports causal model of contamination. Opens path to quantifying exposure in future models.

## Connections to Other Projects

### To Research-Paper LLM-Effect Project

**Direct**: This project measures the confound that threatens that project. Results feed back:

- **If contamination is large**: Pre-2021 scorer becomes non-negotiable; reframe research-paper project as "robustness against contamination"
- **If contamination is small**: Gain confidence in post-2022-scorer findings; less need for within-author control
- **If contamination scales**: Use the correction formula; report corrected and uncorrected results side-by-side

### To Novel-Endings Project

**Indirect**: Genre-signature results (mystery vs. literary fiction resolution curves) should hold across scorers if real. This project validates that:

- Same novel shows consistent ΔS rankings across scorer eras (relative robustness)
- But absolute ΔS magnitude might drift (need to report normalized versions too)

## Timeline

- **Week 1–2**: Acquire corpora (SciFact/NQ public, commission human-LLM pairs)
- **Week 3**: Scorer checkpoint acquisition and setup
- **Week 4–5**: Compute surprisal across grid
- **Week 6–7**: Analyze, visualize, fit models
- **Week 8**: Write methods + results paper

**Output**: Methods paper (1–2 conference abstract or workshop paper) + technical appendix for research-paper project.

## Deliverables

### Primary
- **Contamination index by scorer era** (main result)
- **Dose-response fit** (linear regression + 95% CI)
- **Corrected baseline for research-paper project** (if contamination large enough to matter)

### Secondary
- **Entropy analysis**: Do LLM texts have lower entropy (constrained continuations)?
- **Ranking robustness**: Spearman ρ between scorer pairs (how much relative ranking flips)
- **Token-level bias inspection**: Which lexical/syntactic features drive the gap?

## Open Questions

1. **GPT-2 as baseline**: Is it truly "uncontaminated," or does it already show bias? The 2026 retrieval literature says mild bias even in GPT-2. Strategy: treat as continuous, not binary; report dose-response; note limitations.

2. **Tokenizer drift**: GPT-2 tokenizer vs. modern ones segment text differently. Per-token surprisal not directly comparable. **Solution**: Use bits-per-byte, not bits-per-token.

3. **Which LLM text represents the threat?** Train on ChatGPT, or on Reddit/forum text written by humans who were inspired by LLMs? **Scope**: Focus on direct LLM outputs; human-inspired-by-LLM is later-phase research.

4. **Effect of fine-tuning**: Base models are untuned; instruct-tuned models (Claude, ChatGPT) might show different biases. **Decision**: Measure base models (cleaner signal), but flag instruct-tuning as future direction.

## Related Concepts

- **Source bias**: Established finding that models prefer model-generated text (retrieval lit)
- **Model collapse**: Synthetic data in training degrades diversity (recursive training)
- **Distribution shift**: Covariate shift from training to deployment
- **Fairness / calibration**: Are model predictions calibrated across demographic groups? (Analogous question for text provenance)

## References

- 2026 Springer paper on retrieval + perplexity agreement
- Aitchison et al. on model collapse
- Mahabadi et al. on source bias detection
- [[methodology-controls]] (formal specs for controls)
- [[decision-scorer-selection]] (choosing pre-2022 baseline)

## Next Steps

- [ ] Finalize corpus strategy (SciFact size, human-LLM pair count)
- [ ] Acquire model checkpoints and test inference speed
- [ ] Compute surprisal on 100-pair pilot grid (small scorers × small text set)
- [ ] Visualize gap vs. scorer era on pilot data
- [ ] Decide: Proceed to full scale, or recalibrate design?
- [ ] Draft abstract for workshop or methods venue

---

**Last updated**: 2026-07-02
**Status**: Ready for protocol finalization
**Critical decision point**: Week 2, after corpus assembly — confirm feasibility before full inference run
