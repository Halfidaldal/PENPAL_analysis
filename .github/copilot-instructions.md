# Copilot Instructions for PENPAL Analysis

## Project Overview

This is a data processing and analysis pipeline for the PENPAL (Never-Ending Story) study — a human-AI collaborative storytelling experiment. The project is bilingual (Python for data processing, R for statistical analysis).

## Architecture

**Data flows through three stages:** `data/<dataset>/raw/` → `data/<dataset>/interim/` → `data/<dataset>/processed/`

The `active_dataset` field in `config.yaml` controls which dataset is used (`"TEXT"` or `"Berlin"`). All scripts read this to resolve paths and model choices automatically.

**Three-layer separation:**
- `src/nes/` — Pure functions (no side effects except I/O helpers). This is the `nes` package.
- `scripts/` — Numbered pipeline entry points (`01_` through `08_`). Thin wrappers that load config, call `src/nes/` functions, and save results.
- `analysis/` — R Markdown notebooks that consume only `data/<dataset>/processed/` outputs. These must be runnable top-to-bottom.

**Scripts add `src/` to `sys.path`** via `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))` so that `from nes.<module> import ...` works without installing the package.

## Setup & Running

```bash
# Python environment
python3 -m venv venv && source venv/bin/activate
pip install -r environment/requirements.txt

# Run a single pipeline step
python scripts/02_clean_dataset.py

# Run the full pipeline sequentially (each step depends on the previous)
python scripts/01_download_stories.py
python scripts/02_clean_dataset.py
python scripts/03_compute_embeddings.py
python scripts/04_compute_sentiment.py
```

R analysis notebooks use `renv` for dependency management and `pacman::p_load()` to load packages.

## Key Conventions

- **All configurable parameters live in `config.yaml`** — no magic numbers in code. Parameters are dataset-specific (nested under `TEXT:` or `Berlin:`).
- **Use `nes.io` helpers** (`load_csv`, `save_csv`, `load_parquet`, `save_parquet`, `load_npy`, `save_npy`) for all data I/O. These resolve paths via `config.yaml` and the active dataset automatically.
- **New metrics follow the pattern:** add a pure function in `src/nes/<module>.py`, then create a `scripts/NN_<name>.py` wrapper that loads data, calls the function, and saves results.
- **Scripts should print progress** — no silent scripts. Use `print()` for status updates and `tqdm` for batch processing.
- **Analysis notebooks must not create canonical columns** that other scripts depend on. If a new metric is needed, add it to the pipeline.

## Paper

The accompanying research paper is at `Paper/acl_latex.tex` ("Directional Alignment and Narrative Agency in Human–LLM Co-Writing"), formatted for ACL submission using the `acl` LaTeX package.

## External Services

- **Firebase/Firestore** — Used by `scripts/01_download_stories.py` for data download. Requires a credentials JSON file (path in `config.yaml`).
- **OpenAI API** — Used by `scripts/05_simulate_baseline.py` for AI-AI baseline generation and `scripts/02_clean_dataset.py` for optional spell correction. Pass via `--api-key` or `OPENAI_API_KEY` env var.
- **HuggingFace models** — Embedding and sentiment models are downloaded from HuggingFace Hub. Model names are configured per-dataset in `config.yaml`.
