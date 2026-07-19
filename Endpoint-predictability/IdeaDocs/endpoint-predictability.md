---
type: project
status: active
created: 2026-07-02
updated: 2026-07-19
tags: [research, nlp, narrative, information-theory, llm-effects]
related: 
  - [[00-Research-Index]]
  - [[novelty-transience-resonance-framework]]
needs-review: false
---

# Endpoint Predictability: Information Flow in Narrative & Scientific Text

## Goal

Develop and apply a unified framework for measuring how text reveals, predicts, and shapes understanding of distant endpoints (final chapters, research conclusions, model outputs). Use this to:

1. Detect narrative structure in novels through terminal surprisal
2. Measure LLM effects on research paper argumentative flow
3. Calibrate scorer contamination across model eras
4. Establish a generalizable method for analyzing information-flow in temporal sequences

## Core Insight

In dialogue, "the future" is only the next turn. In complete narratives, we already know the whole future. Inverting temporal logic: instead of "how does this turn shape the next," ask "how does each passage shape the surprisal of the known endpoint?" This unifies storytelling (short horizon), scientific papers (medium), and forecast analysis (long) as instantiations of the same measure at different time scales.

## Sub-Projects

### 1. [[01-narrative-novel-endings]] 
**Status:** Draft | Exploring predictive framing of narrative structure through endpoint surprisal

### 2. [[02-research-papers-llm-effect]]
**Status:** Active | Testing whether post-2022 papers show steeper, more linear surprisal curves toward conclusions

### 3. [[03-model-contamination-study]]
**Status:** Active | Calibrating how training on LLM text affects scorer distributions and baseline assumptions

## Key Concepts

- **Novelty**: Departure from predicted trajectory (contextual surprisal minus BOS baseline)
- **Local resonance**: How much a passage shapes immediate future (next turn/chapter)
- **Terminal resonance**: How much a passage shapes distant endpoint (ending/conclusion)
- **Resonance horizon**: The influence-over-distance curve; shows which passages matter immediately vs. distantly
- **Predictive contribution (ΔS)**: Marginal surprisal change from adding a passage; identifies setup passages
- **Surprisal trajectory S(t)**: Cumulative curve showing resolution profile as context grows

## Methodological Framework

### Core Measure
For context window C_t (all text up to position t), endpoint surprisal:
```
S(t) = s̄(End | C_t) = (1/n_End) Σ_j −log₂ p(w_j | C_t, w_<j)

ΔS(t) = S(t) − S(t−1)  [marginal contribution]
```

### Baselines & Controls
- **Baseline correction**: Novelty_End(t) = S(t) − S(BOS)
- **Permutation/shuffle control**: Real ordering vs. shuffled paragraphs
- **Decoy-ending control**: True ending vs. alternative endings; measure divergence
- **Pre-2022 scorer**: Evaluate with models trained before LLM-text contamination
- **Within-author design**: Track same authors across 2022 boundary
- **Pre-2022 placebo trend**: Baseline drift before ChatGPT to detect discontinuity

### Terminology Discipline

This is **not** Barron's transience (which is strictly local). We're measuring predictive information / information gain over a fixed horizon. The framework encompasses:

- Local resonance (dialogue paper) at h=1 step
- Terminal resonance at h=∞ (end of narrative)
- General case: resonance at arbitrary horizon h

See [[novelty-transience-resonance-framework]] for full definitions.

### Optional Framing: Sliding-Window Resonance (No Fixed Endpoint)

Alongside the fixed-endpoint framing (terminal resonance at h=∞), the same novelty/transience/resonance machinery can be applied as a **sliding window** over a segment rather than toward a known endpoint. For each windowed segment, compute its novelty given the preceding context and its resonance onto a *following context window* (a stretch ahead), exactly as in [[../PENPAL_Analysis/PENPAL]] but **without the turn-taking structure** defining the units.

- Recovers dialogue transience when windows align to turns (h=1) and approaches terminal resonance as the following window grows (w→∞).
- Yields a dense per-position novelty/resonance signal instead of a single endpoint-resolution curve, and does not require an "ending" to be defined in advance — useful for essays, transcripts, or open-ended documents where a fixed endpoint is unnatural.

See [[novelty-transience-resonance-framework]] (Sliding-Window Resonance) for the formal specification.

## The Confound Problem

**The core threat**: If you score with a post-2022 LLM, then post-2022 text reads as less surprising simply because it's closer to that model's training distribution. The measurement artifact and the hypothesis predict the same observable, making them collinear.

**Why it matters differently across projects:**
- **Co-writing paper** (novelty/transience): Confound is collinear with predicted direction but defended by convergent measures (embedding distance, valence) that don't touch the surprisal model
- **Research-paper LLM-effect**: Confound is the entire hypothesis; no natural convergence unless built in; requires out-of-band measures
- **Model-contamination study**: Makes the confound the phenomenon; directly measures how much scorer era contaminates the signal

See [[decision-confound-defense]] and [[methodology-controls]] for defense strategies.

## Generalization Targets

The framework applies anywhere you have a known outcome and temporally extended antecedent:

- **Scientific writing**: Conclusion as target; measure which sections do the work
- **Legal reasoning**: Holding as target; identify which precedent/facts are dispositive
- **Therapy/clinical**: Breakthrough/outcome as target; locate transformative exchanges
- **Music/film**: Cadence/climax as target; analyze harmonic/shot-sequence buildup
- **Forecasting**: Outcome as target; identify which disclosures moved predictability
- **Historical analysis**: Event as target; retrospective information-flow analysis

## Current Status & Next Steps

### Completed
- Conceptual framework unified across projects
- Identified core measurement (ΔS, resonance horizon)
- Mapped confound attacks and defenses
- Scoped three initial applications

### In Progress
- Assembling text corpora (novels, research papers, pre/post 2022)
- Scorer panel selection (pre-2021, transition-era, post-2023)
- Design specification for causal inference (within-author, placebo)

### Blocking Issues
- Pre-2022 base model selection (tokenizer drift, availability)
- Research-paper corpus annotation (determine human-coherence bridge)
- LLM-text provenance labels (parallel human/LLM pairs for model-contamination study)

## Key References & Decisions

### Decisions
- [[decision-scorer-selection]]: Why pre-2022 models are load-bearing
- [[decision-confound-defense]]: Convergence vs. out-of-band measures
- [[decision-surprisal-vs-entropy]]: Why surprisal is correct for this framework

### Supporting Concepts
- [[novelty-transience-resonance-framework]]: Formal definitions
- [[surprisal-entropy-perplexity-guide]]: Measurement tradeoffs
- [[methodology-controls]]: Detailed control specification

## Related Work

- Barron et al. on transience (local temporal dynamics)
- Model collapse / recursive training literature
- Perplexity-as-surprise on scientific text (2025 Springer work)
- Source bias detection in retrieval
- Narrative theory on foreshadowing and dramatic irony

## For Presentation

**ELI5 core idea (4–5 minutes)**:
We take the dialogue framework and turn it inward. Instead of asking "how does this turn shape the next," we fix the ending as known and ask "how much did adding this paragraph reduce the ending's surprise?" That score is computational foreshadowing. Multiply by different horizons (next chapter, midpoint, ending) and you get curves that describe storytelling style as the shape of how meaning propagates forward in time.

**Five-part structure for oral presentation**:
1. The core move (inversion of temporal logic)
2. How it works (growing context, endpoint surprisal, marginal contribution)
3. What it means (Chekhov detector, genre signatures)
4. The generalization (same object at different horizons)
5. The honest caveat (measurement fragility at long horizons)

---

**Last reviewed**: 2026-07-02
**Maintainer**: Halfdan
