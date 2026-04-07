# PENPAL: Directional Alignment and Narrative Agency in Human–LLM Co-Writing

## Overview

PENPAL (Pen Pal for Never-Ending Story) is a research project investigating the dynamics of collaborative storytelling between humans and large language models (LLMs). The study examines how emotional tone, narrative novelty, and creative agency flow between co-authors when one (or both) is an AI.

The central question: **When humans write stories with AI, who leads and who follows?**

## The Experiment

Participants engaged in a turn-based collaborative fiction task via a web platform. Each story began with the prompt *"This is the story of..."* and unfolded over 10 alternating turns. The experiment ran across multiple workshops and data collection sessions.

### Three Experimental Conditions

| Condition | Authors | Description |
|-----------|---------|-------------|
| **Human-AI** | Human + LLM | A human writes alternating turns with an LLM (GPT-4.1, Claude, Llama, or Qwen) |
| **Human-Human** | Human + Human | Two humans write alternating turns together |
| **AI-AI** | LLM + LLM | Two AI agents write alternating turns (simulated baseline) |

The AI-AI condition serves as a computational baseline—showing what happens when both authors are LLMs with identical instructions, removing human agency from the equation entirely.

## Research Questions

1. **Valence Alignment**: Do co-authors converge emotionally? Does the AI accommodate the human's emotional tone, or vice versa?

2. **Asymmetric Accommodation**: Is alignment bidirectional (mutual adaptation) or asymmetric (one author consistently adjusts to the other)?

3. **Narrative Novelty**: Who drives the story forward? Do human contributions introduce more surprise, or does the AI play it safe?

4. **Condition Differences**: How do Human-AI dynamics differ from Human-Human collaboration? What does the AI-AI baseline reveal about LLM behavior?

## Metrics & Measurement

### Valence (Emotional Tone)

We measure emotional valence using **semantic concept projection**—projecting text embeddings onto a valence axis defined by positive/negative anchor words. This produces a continuous score from negative (dark, tense) to positive (bright, hopeful) for each turn.

- **Valence alignment**: Correlation between Author 1 and Author 2 valence within a turn
- **Lagged alignment**: Does Author 2's valence predict Author 1's next turn?
- **Rubber-band effect**: Does a large valence gap at turn *t* predict gap reduction at turn *t+1*?

### Novelty, Transience & Resonance

Using information-theoretic measures inspired by computational linguistics:

- **Novelty (Surprisal)**: How unexpected is a turn given the story so far? Computed via embedding distance from prior context.
- **Transience**: How quickly does a turn's influence fade? Does subsequent text "forget" this contribution?
- **Resonance** = Novelty − Transience: Net forward influence. High resonance means a surprising turn that shapes the story's future direction.

### Surface Metrics

Text descriptives computed via spaCy and textdescriptives:

- Word count, sentence count, vocabulary richness
- Readability scores (Flesch-Kincaid, etc.)
- Part-of-speech distributions
- Lexical diversity measures

### Semantic Exploration

Embedding-based measures of how authors navigate semantic space:

- **Cosine similarity** between consecutive turns (same author and cross-author)
- **Semantic distance** from story centroid
- **Exploration trajectory** over the story arc

## Statistical Modeling

Analyses use **linear mixed-effects models** (via R's `lme4`) with:

- Fixed effects for condition, author role, turn number
- Random intercepts for story (conversation_id)
- Random slopes where supported by data

Key contrasts:
- Human-AI vs Human-Human (effect of AI partner)
- Human-AI vs AI-AI (effect of human presence)
- Author 1 vs Author 2 within each condition

## Data Pipeline

```
Raw Data (Firestore) 
    ↓ 01_download_stories.py
Interim (cleaned, filtered)
    ↓ 02_clean_dataset.py
    ↓ 03_compute_embeddings.py
    ↓ 04_compute_sentiment.py
    ↓ 05_simulate_baseline.py (AI-AI only)
    ↓ 06_compute_textdescriptives.py
    ↓ 07_compute_novelty.py
    ↓ 08_compute_semantic_exploration.py
Processed (analysis-ready)
    ↓
R Markdown Analysis Notebooks
```

All data uses standardized column naming (`author_1`, `author_2`) across conditions, enabling unified cross-condition comparisons.

## Repository Structure

```
PENPAL_analysis/
├── data/
│   ├── human-ai/         # Human-AI experiment data
│   ├── human-human/      # Human-Human experiment data
│   └── ai-ai/            # AI-AI simulation data
├── src/nes/              # Core analysis modules (Python)
├── scripts/              # Numbered pipeline scripts
├── analysis/
│   ├── human-ai/         # Within-condition analyses
│   ├── human-human/      # Within-condition analyses
│   ├── ai-ai/            # Within-condition analyses
│   └── comparison/       # Cross-condition contrasts
├── Paper/                # LaTeX manuscript
└── config.yaml           # Pipeline configuration
```

## Citation

If you use this codebase or data, please cite:

```
[Citation pending publication]
```

## Contact

For questions about the project, data access, or collaboration inquiries, contact the project maintainers.
