---
type: project
status: draft
created: 2026-07-02
updated: 2026-07-19
tags: [narrative, literary-analysis, computational-literature, surprisal, endpoint-prediction]
related:
  - [[endpoint-predictability]]
  - [[novelty-transience-resonance-framework]]
  - [[surprisal-entropy-perplexity-guide]]
needs-review: false
---

# Novel-Endings Endpoint Predictability

## Goal

Detect and characterize narrative structure by measuring how each passage shapes the predictability of a novel's ending. Use surprisal trajectories to uncover which sentences are foreshadowing, identify genre signatures, and operationalize narrative concepts like Chekhov's gun and dramatic irony.

**Foundation**: This project extends [[../PENPAL_Analysis/PENPAL]] (dyadic turn-taking and narrative agency) from co-writing dynamics to single-author narrative architecture. See [[../PENPAL_Analysis/02-research-architecture]] for how endpoint-predictability operationalizes PENPAL's novelty/transience/resonance metrics at the architectural scale.

## Core Idea

Hold a novel's final chapter as fixed target. Feed a growing window of text to a language model, measuring the surprisal of the ending at each step. As context accumulates, surprise about the ending decreases. The per-passage contribution (ΔS) reveals which passages do the "load-bearing work" toward the conclusion.

**ELI5**: You secretly know the last page. As you read each new page, ask: did this help me see the ending coming? Pages that dramatically lower ending-surprisal are setups; pages with ΔS ≈ 0 are texture. Map these scores and you've found the story's hidden machinery.

## Hypotheses

### Primary
- Mystery novels show sharp late drops in S(t) (surprise about ending only resolves at reveal)
- Greek tragedies with announced fate show low S(t) maintained across text (dramatic irony: ending known, surprise low)
- Literary fiction spreads ending-resolution diffusely across the text (many small contributions)

### Secondary
- High-ΔS passages align with human annotations of "setup" or "foreshadowing"
- Shuffle-control shows real-ordering produces more structured curves than randomized text
- Decoy-ending divergence grows monotonically (true ending becomes progressively more predictable relative to alternatives)

## Method

### Core Measurement

```
S(t) = s̄(End | C_t) = (1/n_End) Σ_j −log₂ p(w_j | C_t, w_<j)

Novelty_End(t) = S(t) − S(BOS)  [baseline-corrected]

ΔS(t) = S(t−1) − S(t)  [marginal contribution; positive = lowers ending surprisal]

Resolution Profile = trajectory of cumulative Σ ΔS(t)
```

### Resolution-Profile Typologies

**Mystery**: 
- Low S(t) early (many suspect endings plausible)
- Sharp drop near climax
- Steep cumulative resolution curve concentrated in final 10–20% of text

**Tragic (fate-announced)**:
- Low S(t) maintained throughout (ending already known)
- Flat or slowly declining cumulative curve
- High initial novelty (reader doesn't yet know the announced ending is inevitable)

**Literary/Diffuse**:
- Gradual monotonic decline in S(t)
- Smooth cumulative curve with constant small increments
- ΔS variance low (no single "reveal" moment)

**Thriller/Surprise**:
- Moderate early S(t)
- Late-middle inflection (plot twist)
- Asymmetrical curve (steep rise then steep fall)

## Alternative Framing: Sliding-Window Resonance (No Fixed Ending)

Instead of holding the final chapter as a fixed target, an alternative option is to apply the existing [[novelty-transience-resonance-framework]] as a **sliding window** over the text. For each windowed segment, compute:

- **Novelty**: surprisal of the segment given the preceding context window (minus BOS baseline).
- **Resonance**: how much the segment lowers the surprisal of a *following context window* (a stretch ahead), rather than of the known ending.

This is the same measurement used in [[../PENPAL_Analysis/PENPAL]], but **without the turn-taking structure** defining the units — segmentation is a sliding window over the prose instead of conversational turns. It recovers dialogue transience when the window aligns to turns (h=1) and approaches the endpoint framing above as the following window grows to cover the rest of the novel (w→∞).

**Trade-off vs. fixed-endpoint framing**: The sliding-window variant yields a dense per-position novelty/resonance signal across the whole text and needs no pre-defined "ending", making it applicable to open-ended or ending-ambiguous texts. The fixed-endpoint framing (above) is cleaner for the specific claims here (foreshadowing, Chekhov's gun, genre resolution curves) because the target is definite. Both can be run on the same corpus and compared. See [[novelty-transience-resonance-framework]] (Sliding-Window Resonance) for the formal specification.

## Controls & Rigor

### Baseline Correction
Subtract BOS baseline to isolate contextual facilitation from turn-intrinsic surprisal.

### Permutation Null
Shuffle paragraph order within novel. Real ordering should produce:
- Steeper, more structured curve than shuffled
- Significantly higher R² of linear/polynomial fit to cumulative curve
- Larger ΔS on "setup" passages, smaller on "texture"

**This directly tests**: Is the signal narrative architecture or just topical overlap?

### Decoy-Ending Control
Compute S(t) for:
- True ending
- Decoy endings (other novels' endings, model-generated alternatives)

Measure gap: S_decoy(t) − S_true(t)

**Expectation**: True ending becomes progressively more predictable (gap widens) as context accumulates. Shuffled ordering should show flat or inverted gap.

### Human Annotation Bridge
On subsample:
- Annotate "setup" vs. "texture" passages via close reading
- Compute inter-rater agreement (Cohen's κ)
- Correlate ΔS with setup annotations

**This grounds**: Computational signal in interpretive practice.

### Genre Validation
- Collect novels stratified by genre (mystery, literary fiction, thriller, etc.)
- Fit curve-shape models (logistic, S-curve, exponential) per genre
- Compare curve parameters across genres
- Predict genre from curve shape alone (as validation task)

## Measurement Artifacts to Watch

### Long-Context Degradation
When a far-earlier passage shows ΔS ≈ 0, could mean:
- Genuinely didn't matter for the ending, OR
- Model couldn't reach 200+ pages back

**Mitigation**: Use modern long-context models (Claude 3.5+) and validate positional effects with shuffled null. If shuffle produces same zero-ΔS for far-back passages, then artifact. If real ordering shows high ΔS while shuffled doesn't, architecture-signal wins.

### Model-Specific Resolution
Surprisal is model-relative. GPT-trained models might assign different baseline S(t) to GPT-written endings vs. human-written ones.

**Mitigation**: 
- Use recent base models (not instruction-tuned, to avoid RLHF style bias)
- Run on 2+ model families (confirm typologies hold across scorers)
- Report qualitative curve shapes (not just magnitudes)

## Data & Corpora

### Novels
- **Genre diversity**: Mystery (Agatha Christie, modern), Literary (Austen, contemporary), Thriller (le Carré, modern), Science Fiction, Romance
- **Era diversity**: Pre-1950, 1950–2000, 2000+
- **Size**: ~30–50 novels initially; scale to 100+ if validating
- **Format**: Plain text, standardized preprocessing (remove formatting, tokenize)

### Annotation
- Subsample: 5–10 novels
- Annotators: 2–3 people trained on setup/texture distinction
- Passages: sentence or paragraph level (discuss with annotators)

## Generalization

This framework transfers directly to:

**Scientific writing**: Treat conclusion as ending. Measure which sections (intro, methods, results) do the work of making the conclusion inevitable. Well-structured papers show monotonic resolution; papers with unsupported conclusions show flat or late-jumping curves. **Venue**: TEXT, TACL, or computational-linguistics venue.

**Narrative in film/music**: Same logic with different surprisal models:
- Film: Use visual embeddings or shot-sequence tokens; climax as target
- Music: Use harmonic/pitch sequences; cadence as target
- Both share typologies (anticipated vs. twist vs. diffuse resolution)

**Legal reasoning**: Holding as target; measure which paragraphs of an opinion do the work. Which are dicta (low ΔS) vs. holdings (high ΔS)?

## Timeline & Milestones

- **Month 1**: Assemble corpora, select scorer(s), implement pipeline
- **Month 2**: Compute baselines; run genre-stratified curves
- **Month 3**: Validate against permutation null and human annotations
- **Month 4**: Write up, prepare figures (curve families by genre, genre-prediction from shape)

## Related Concepts

- **Chekhov's gun**: Computationally = high-ΔS passage that later turns out to be "needed." Conversely, low-ΔS passage that later becomes emotionally important = false setup (red herring)
- **Dramatic irony**: Computationally = low S(t) maintained throughout (ending predictable to audience but not character). Foreshadowing = early high-ΔS passages.
- **Plot twist**: Computationally = sudden inflection in ΔS or reversal of S(t) slope (was climbing, then drops sharply; or vice versa)

## Next Actions

- [ ] Finalize novel list and obtain full texts
- [ ] Select 2–3 base model scorers
- [ ] Write preprocessing pipeline (tokenization, chapter/section boundaries)
- [ ] Implement S(t) and ΔS computation
- [ ] Run on 3–5 pilot novels and visualize curves
- [ ] Recruit annotators for setup/texture labels
- [ ] Implement permutation and decoy-ending nulls
- [ ] Document genre-typology findings

## Open Questions

1. **Sentence vs. paragraph vs. section granularity**: How fine-grained should ΔS be? Sentence is noisy; section might be too coarse. Test on pilot corpus.
2. **Which LLM as scorer**: Use Claude for long-context stability, or GPT for comparison? Plan for both.
3. **Ending definition**: Just final chapter, or final scene? Discuss with collaborators.
4. **Genre breadth**: Do typologies hold across languages (translate and rerun)? Phase 2.

## Notes for Presentation

**Strength of the idea**: Solves conceptual issue with dialogue paper (in dialogue, "future" is only next turn; here future is known, which is cleaner). Gives novel way to operationalize literary-criticism concepts (foreshadowing, dramatic irony, Chekhov's gun) that are otherwise interpretive.

**Caution**: Long-context measurement fragility is real. Make sure permutation null and decoy-ending controls are bulletproof before claiming narrative-structure findings.

---

**Last updated**: 2026-07-02
**Status**: Ready for pilot experiments
