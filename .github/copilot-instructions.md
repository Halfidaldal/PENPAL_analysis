# Copilot Instructions for PENPAL Analysis

## Project Overview

This is a data processing and analysis pipeline for the PENPAL (Never-Ending Story) study — a human-AI, human-human, and AI-AI collaborative storytelling experiment. The project is bilingual (Python for data processing, R for statistical analysis).

**Three experimental conditions:**
- `human-ai` — Humans co-writing stories with an LLM
- `human-human` — Humans co-writing stories with each other
- `ai-ai` — Two AI agents co-writing (simulated baseline for 2×2 analysis)

## Architecture

**Data flows through three stages:** `data/<experiment>/raw/` → `data/<experiment>/interim/` → `data/<experiment>/processed/`

The `active_experiment` field in `config.yaml` controls which experiment is used (`"human-ai"`, `"human-human"`, or `"ai-ai"`). All scripts read this to resolve paths and experiment-specific parameters automatically.

**Three-layer separation:**
- `src/nes/` — Pure functions (no side effects except I/O helpers). This is the `nes` package.
- `scripts/` — Numbered pipeline entry points (`01_` through `08_`). Thin wrappers that load config, call `src/nes/` functions, and save results.
- `analysis/` — R Markdown notebooks organized by experiment:
  - `analysis/human-ai/` — Human-AI condition analysis
  - `analysis/human-human/` — Human-human condition analysis
  - `analysis/ai-ai/` — AI-AI baseline analysis
  - `analysis/comparison/` — Cross-condition statistical contrasts (all three conditions)

**Scripts add `src/` to `sys.path`** via `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))` so that `from nes.<module> import ...` works without installing the package.

## Setup & Running

```bash
# Python environment
python3 -m venv venv && source venv/bin/activate
pip install -r environment/requirements.txt

# Switch experiment by editing config.yaml:
#   active_experiment: "human-ai"   # or "human-human" or "ai-ai"

# Run a single pipeline step
python scripts/02_clean_dataset.py

# Run the full pipeline sequentially (each step depends on the previous)
python scripts/01_download_stories.py      # Download from Firestore (human-ai/human-human only)
python scripts/02_clean_dataset.py         # Filter and clean
python scripts/03_compute_embeddings.py    # Compute embeddings
python scripts/04_compute_sentiment.py     # Sentiment analysis
python scripts/05_simulate_baseline.py     # AI-AI simulation (for ai-ai condition)
python scripts/06_compute_textdescriptives.py
python scripts/07_compute_novelty.py
python scripts/08_compute_semantic_exploration.py
```

**For AI-AI simulation:**
```bash
# Set API keys (one or more, depending on models to run)
export OPENAI_API_KEY=your_key        # for gpt-4.1
export ANTHROPIC_API_KEY=your_key     # for claude-sonnet
export OPENROUTER_API_KEY=your_key    # for llama-70b, qwen-72b

# Generate 40 stories (10 per model)
python scripts/05_simulate_baseline.py

# Then set active_experiment to "ai-ai" and run pipeline 02-08
```

R analysis notebooks use `renv` for dependency management and `pacman::p_load()` to load packages:

```r
# Restore R dependencies (from project root)
# R -e "renv::restore()"
```

## Key Conventions

- **All configurable parameters live in `config.yaml`** — no magic numbers in code. Experiment-specific parameters are nested under `experiments.human-ai`, `experiments.human-human`, or `experiments.ai-ai`. Shared parameters are under `shared`.
- **Use `nes.io` helpers** (`load_csv`, `save_csv`, `load_parquet`, `save_parquet`, `load_npy`, `save_npy`) for all data I/O. These resolve paths via `config.yaml` and the active experiment automatically. Pass `experiment="human-ai"` to override.
- **New metrics follow the pattern:** add a pure function in `src/nes/<module>.py`, then create a `scripts/NN_<name>.py` wrapper that loads data, calls the function, and saves results.
- **Scripts should print progress** — no silent scripts. Use `print()` for status updates and `tqdm` for batch processing.
- **Analysis notebooks must not create canonical columns** that other scripts depend on. If a new metric is needed, add it to the pipeline.
- **R notebooks use `pacman::p_load()`** to load packages and `here()` to resolve paths relative to project root.
- **Comparison notebooks** load all three conditions (where data exists) and bind with a `condition` column for statistical contrasts. Use `analysis/comparison/_comparison_utils.R` for standardized data loading.

## AI-AI Simulation

The `ai-ai` condition simulates collaborative stories between two AI agents using 4 models:
- **gpt-4.1-2025-04-14** (OpenAI)
- **claude-sonnet-4-5-20250929** (Anthropic)
- **Llama-3.3-70B-Instruct** (via OpenRouter)
- **Qwen2.5-72B-Instruct** (via OpenRouter)

Configuration is in `config.yaml` under `experiments.ai-ai.simulation.models`. The simulation generates 10 stories per model (40 total) with 10 turns each.

## Paper

The accompanying research paper is at `Paper/acl_latex.tex` ("Directional Alignment and Narrative Agency in Human–LLM Co-Writing"), formatted for ACL submission using the `acl` LaTeX package.

## External Services

- **Firebase/Firestore** — Used by `scripts/01_download_stories.py` for data download. Two schemas supported:
  - `flat` (human-ai): All interactions in a single collection
  - `nested` (human-human): Sessions with nested turns subcollection
- **OpenAI API** — Used for gpt-4.1 simulation. Set `OPENAI_API_KEY` env var.
- **Anthropic API** — Used for Claude simulation. Set `ANTHROPIC_API_KEY` env var.
- **OpenRouter API** — Used for Llama and Qwen simulation. Set `OPENROUTER_API_KEY` env var.
- **HuggingFace models** — Embedding and sentiment models are downloaded from HuggingFace Hub. Model names are configured in `config.yaml` under `shared`.
