---
type: decision
status: active
created: 2026-07-02
updated: 2026-07-06
tags: [methodology, confounds, causal-inference, research-design]
related:
  - [[endpoint-predictability]]
  - [[02-research-papers-llm-effect]]
  - [[03-model-contamination-study]]
  - [[surprisal-entropy-perplexity-guide]]
  - [[Large Language Models Are Overconfident in Their Own Responses]]
needs-review: false
---

# Decision: How to Defend Against LLM-Scorer Circularity Confound

## Context

In the research-paper LLM-effect project, we measure whether post-2022 papers show steeper, more linear surprisal curves toward their conclusions. The confound is:

**If you score with a post-2022 LLM, then post-2022 text reads as lower-surprisal simply because it's closer to that model's training distribution. The measurement artifact and the hypothesis predict the identical observable, making them collinear.**

This is different from the novelty/transience paper, where the confound is a threat but is collinear with the predicted direction and defended by convergent measures (embedding distance, valence).

**Key difference**: Here, the confound IS the entire hypothesis. There's no natural convergence unless built in.

## Options Considered

### Option A: Deny the Problem (Not Viable)

"The confound doesn't exist because BOS baseline subtraction controls for it."

**Rejection**: BOS baseline removes intrinsic surprisal (rare words), not distributional proximity. Post-2022 text sits closer to post-2022 scorer distribution for both the contextual and baseline terms, so the confound survives baseline subtraction.

### Option B: Argue Collinearity Exonerates (Wrong)

"The confound and the hypothesis point the same way, so the confound can only shrink a real effect, not create it."

**Rejection**: This is the critical error. When confound and hypothesis predict the same observable in the same direction, the confound is a **fully sufficient alternative explanation**. It could be producing all of the gap, some of it, or none — the surprisal numbers alone cannot tell. The two stories are indistinguishable within surprisal alone.

**Why I got this wrong**: This logic only works if you've independently established the effect is real (as we have for novelty/transience, defended by convergent measures). Without that, "collinear direction" is irrelevant.

### Option C: Convergent Measures (Not Available)

"Measure with non-surprisal tools (embedding distance, valence projection, human ratings) that don't depend on the scorer's distribution."

**Status**: Partially available. We can add human-coherence annotations, but there's not a natural "semantics" measure that works on research papers the way embedding distance works on dialogue contributions.

**Caution**: Human ratings on 20–30 papers is expensive and might not correlate with curve features (see [[02-research-papers-llm-effect]] for analysis).

### Option D: Pre-2022 Scorer (Load-Bearing)

"Use a base model trained before LLM-text contamination. Measure curve features with both pre-2022 and post-2022 scorers. If the effect is real, it should persist; if it's confound, it should vanish with older scorer."

**Status**: **Chosen as primary defense**. 

**Why it works**:
- Pre-2022 scorer has a frozen representation of "normal language" (no LLM exposure in training).
- Post-2022 text sits closer to 2024 norms than to 2020 norms for everyone, regardless of coherence.
- But the distance is mechanically the same for all 2023-2024 papers, coherent or not.
- So if H1 (argument structure improved) is true → effect with both scorers.
- If H2 (confound) is true → effect disappears with pre-2022 scorer.

**Caveat**: Even GPT-2 has mild source bias from earlier LLM overlap. Not perfectly clean, but cleaner than 2024. Report as dose-response, not binary threshold.

### Option E: Within-Author Design (Causal Identification)

"Track the same authors across the 2022 boundary. Differences out author-level style, field, venue — what remains is the within-author shift co-occurring with LLM access."

**Status**: **Chosen as secondary defense** (reinforces Option D).

**Why it matters**:
- Removes confounding from field evolution, venue acceptance-rate changes, ESL-author population shifts.
- Residual effect is much closer to causal.
- But still doesn't address the scorer-proximity confound directly — that's where Option D comes in.

### Option F: Placebo Trend (Discontinuity Test)

"Measure pre-2022 papers (2010–2021) for same features. If 'steeper, lower-variance curves' is already trending upward before ChatGPT, then post-2022 is just continuation of drift, not breakpoint."

**Status**: **Chosen as tertiary defense**.

**Why it matters**:
- Shows whether the observed shift is novel to the LLM era or part of long-run stylistic drift.
- If pre-2022 trend is flat and post-2022 drops sharply, discontinuity is real.
- If pre-2022 already declining, reframe as "LLM accelerates a pre-existing trend" rather than "LLM causes the trend."

### Option G: Entropy-Over-Conclusions (Out-of-Band Measure)

"In addition to endpoint-surprisal of the true conclusion, measure entropy over plausible conclusions. Does post-2022 text narrow the space of live conclusions?"

**Status**: **Exploratory** (can add for robustness).

**Why it might help**:
- Entropy is less bound to one particular scorer's distribution.
- If post-2022 papers have lower entropy-over-conclusions (fewer conclusion options seem live), that's evidence of tighter structure, not just scorer proximity.
- Independent signal from surprisal; if both converge, more credible.

**Cost**: More expensive to compute (sample over possible conclusions).

---

## Decision

**Use Options D + E + F as the core defense strategy.**

### Core Pipeline

1. **Pre-2022 scorer** (GPT-2 or GPT-Neo base):
   - Compute surprisal of research paper conclusions
   - Measure curve features (mean ΔS, variance, linearity-R²)
   - Establish baseline effect (do post-2022 papers differ from pre-2022?)

2. **Post-2023 scorer** (Claude, GPT-J, or equivalent):
   - Repeat on same papers
   - Compare effect sizes
   - **Key result**: Effect size vs. scorer age

3. **Within-author analysis**:
   - Track ~50–100 authors with papers pre- and post-2022
   - Difference: feature_2024 − feature_2019 (per author)
   - Test significance of within-author shift

4. **Placebo trend**:
   - Bin papers 2010–2024 by year
   - Plot mean curve features vs. year (pre-2022 and post-2022 separately)
   - Test for discontinuity at 2022 boundary

### Interpretation

**If all four converge (effect holds across scorers, within-author, with placebo discontinuity)**:
- Strong evidence that the effect is not pure confound
- Argue for H1 (argument structure changed) or at least H3 (acceleration of pre-existing trend)
- Can claim: "Structural information-flow signature distinct from scorer-distribution proximity"

**If effect disappears with pre-2022 scorer but appears with post-2023**:
- Strong evidence for confound (H2)
- Honest finding: "Post-2022 text sits closer to modern models, not necessarily more coherent"
- Still publishable as cautionary tale (what not to measure with naive surprisal)

**If pre-2022 placebo trend is already steep**:
- Reframe: "LLM era accelerates a long-run homogenization of scientific prose"
- Cite field-level drift (template-ification, ESL expansion, impact of structured databases like PubMed)
- Still novel (quantifies the acceleration), still publishable

---

## Specific Actions

### Immediate (Months 1–2)

- [ ] Acquire GPT-2, GPT-Neo base checkpoints; test inference speed on 100-paper sample
- [ ] Select 500–1000 research papers (stratified by year 2010–2024, venue, field)
- [ ] Extract conclusions / abstracts; tokenize and compute surprisal under both scorers
- [ ] Generate feature vectors (mean ΔS, variance, linearity-R²)
- [ ] Visualize: effect size (mean difference in features) post- vs. pre-2022, by scorer

### Mid-stage (Month 3)

- [ ] Identify author cohort (50–100 with papers on both sides of 2022)
- [ ] Extract their papers, compute features
- [ ] Run paired t-tests and mixed-effect models (within-author effect)
- [ ] Bin all papers by year; plot trend 2010–2024
- [ ] Test for knot/discontinuity at 2022 boundary

### Late stage (Month 4)

- [ ] Write up findings; create Figure 1 (effect by scorer), Figure 2 (within-author shift), Figure 3 (placebo trend)
- [ ] If effect survives all tests → report as evidence of structural change
- [ ] If effect vanishes with pre-2022 scorer → honest confound narrative
- [ ] Optional: entropy-over-conclusions analysis (if time permits)

---

## What This Decision Commits To

✅ **Commits to**:
- Both pre-2021 and post-2023 scorers (non-negotiable)
- Within-author matching (adds causal credibility)
- Placebo trend (proves discontinuity vs. drift)
- Transparent reporting (show effect size by scorer)
- Honest interpretation (if confound dominates, say so)

❌ **Does not commit to**:
- Entropy analysis (nice-to-have, not essential)
- Human-coherence bridge (optional, expensive)
- Multiple alternative corpora (use single, well-sourced set)

---

## Why This Addresses the Confound

The strategy has three layers:

1. **Diagnostic (pre-2022 scorer)**: Directly tests whether effect scales with scorer-era proximity
2. **Causal (within-author)**: Removes field-level confounds; what remains is more plausibly causal
3. **Temporal (placebo trend)**: Proves that post-2022 shift is discontinuous, not long-run drift

Together, they isolate the effect from the confound. Individually, each is weak; together, they're strong.

---

## Precedent & Justification

**Co-writing paper**: Successfully defended against same confound using convergent measures (embedding, valence). That work proved the framework works when the signal is real.

**This project**: Inverts the threat: doesn't assume the signal is real, uses design to prove/disprove it. More honest, more rigorous.

**Analogy**: Clinical trials use intent-to-treat (ITT) + per-protocol (PP) analyses. ITT has confounds; PP has selection bias. Together, they triangulate. We're doing the same with scorer-era + within-author.

---

## Risks This Decision Accepts

1. **GPT-2 isn't clean**: It already has mild source bias. Mitigation: report dose-response, acknowledge limitation, use as one leg of a tripod (not the only support).

2. **Within-author effect might be small**: Not every author changed their writing post-2022. Mitigation: test effect before committing; if small, rely more on scorers and placebo.

3. **Placebo trend might be ambiguous**: Pre-2022 drift could be gradual, making 2022 boundary unclear. Mitigation: test for knot using piecewise-linear fit; report slope estimates on both sides.

4. **All controls could fail**: Maybe the effect IS pure confound. Mitigation: that's still a publishable finding (cautionary tale). Reframe and submit.

---

## Review Checkpoints

Before finalizing, consult:
- Are there newer survey papers on source bias that change the GPT-2 assumption?
- Have other authors published within-author LLM-effect studies we should cite?
- Is the placebo-trend setup sound (are we comparing the right pre-2022 baseline)?

---

## Related Decisions

- [[decision-scorer-selection]]: Which specific pre-2021 and post-2023 models to use
- [[methodology-controls]]: Formal specs for within-author and placebo implementations

---

**Decision made**: 2026-07-02
**Status**: Final (pending Week-2 pilot results to confirm feasibility)
**Reviewed by**: Halfdan
