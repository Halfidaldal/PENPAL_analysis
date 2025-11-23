# PENPAL Analysis Pipeline

This repository contains the complete data processing and analysis pipeline for the PENPAL (Never-Ending Story) study.

## Overview

The project follows a structured pipeline approach where:
- **Scripts** (`scripts/`) handle data processing and metric computation
- **Modules** (`src/nes/`) contain reusable functions with no side effects
- **Analysis** (`analysis/`) contains notebooks that consume processed data
- **Data** flows through `data/raw/` → `data/interim/` → `data/processed/`

## Project Structure

```
PENPAL_analysis/
├── data/
│   ├── raw/              # Raw data exports from Firestore
│   ├── interim/          # Cleaned but not final (spell-checked, filtered)
│   └── processed/        # Final analysis tables (embeddings, sentiment, etc.)
├── src/
│   └── nes/              # Core analysis package
│       ├── __init__.py
│       ├── io.py         # Load/save helpers
│       ├── cleaning.py   # Rectifying, edit distance, filters
│       ├── embeddings.py # Embedding computation
│       ├── sentiment.py  # Sentiment analysis
│       └── simulation.py # AI-AI baseline generation
├── scripts/              # Pipeline entry points (numbered for execution order)
│   ├── 01_download_stories.py
│   ├── 02_clean_dataset.py
│   ├── 03_compute_embeddings.py
│   └── 04_compute_sentiment.py
├── analysis/             # Read-only analysis notebooks
│   └── (notebooks go here)
├── environment/
│   └── requirements.txt  # Python dependencies
├── config.yaml           # Configuration parameters
├── Berlin/               # Legacy code (kept during migration)
└── README.md             # This file
```

## Setup

### 1. Create a Python environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Using conda
conda create -n penpal python=3.11
conda activate penpal
```

### 2. Install dependencies

```bash
pip install -r environment/requirements.txt
```

### 3. Configure credentials

Place your Firebase admin SDK JSON file in the appropriate location (see `config.yaml` for the path).

## Running the Pipeline

The pipeline is designed to be run sequentially. Each script produces output that the next script consumes.

### Full pipeline from scratch

```bash
# 1. Download stories from Firestore (requires credentials)
python scripts/01_download_stories.py

# 2. Clean and filter data
python scripts/02_clean_dataset.py

# 3. Compute embeddings (requires GPU for speed, ~30 min on CPU)
python scripts/03_compute_embeddings.py

# 4. Compute sentiment scores
python scripts/04_compute_sentiment.py
```

### Optional: Delete incomplete stories from Firestore

```bash
python scripts/01_download_stories.py --delete-incomplete
```

**⚠️ Warning:** This permanently deletes data from Firestore. Use with caution.

## Configuration

All parameters are centralized in `config.yaml`:

- **Data paths**: Where to save raw/interim/processed data
- **Firestore settings**: Collection names, minimum interactions
- **Cleaning parameters**: Edit distance threshold, filters
- **Model settings**: Embedding models, sentiment models, batch sizes
- **Analysis parameters**: Random seeds, language filters

To change a parameter (e.g., edit distance threshold), edit `config.yaml` and rerun the relevant script.

## Data Outputs

After running the pipeline, you'll have:

| File | Location | Description |
|------|----------|-------------|
| `finished_stories_raw.csv` | `data/raw/` | Raw story interactions from Firestore |
| `stories_filtered.csv` | `data/interim/` | After edit distance filtering |
| `stories_full_text.csv` | `data/interim/` | Story-level with full_story/full_user/full_ai |
| `story_embeddings_field.parquet` | `data/processed/` | Stories with embeddings (Jina) |
| `story_embeddings_jina_field.npy` | `data/processed/` | NumPy array of story embeddings |
| `story_user_embeddings_jina_field.npy` | `data/processed/` | NumPy array of user embeddings |
| `story_ai_embeddings_jina_field.npy` | `data/processed/` | NumPy array of AI embeddings |
| `story_sentiment_scores.parquet` | `data/processed/` | Story-level sentiment scores |
| `dyadic_sentiment_scores.parquet` | `data/processed/` | Turn-by-turn sentiment |

## Analysis Notebooks

Notebooks in `analysis/` should:
- Load only from `data/processed/`
- Not create new canonical columns that other scripts depend on
- Be runnable top-to-bottom without manual intervention

To add a new analysis:
1. If it requires new metrics, add a function to `src/nes/` and a script to `scripts/`
2. Run the script to generate processed data
3. Create a notebook in `analysis/` that loads the processed data

## Development

### Adding a new metric

1. **Write the function** in the appropriate module (`src/nes/`):
   ```python
   # src/nes/mymetric.py
   def compute_my_metric(df):
       # ... logic here
       return df_with_metric
   ```

2. **Create a script** in `scripts/`:
   ```python
   # scripts/05_compute_my_metric.py
   from nes.mymetric import compute_my_metric
   from nes.io import load_parquet, save_parquet
   
   def main():
       df = load_parquet("input.parquet", stage="processed")
       df_metric = compute_my_metric(df)
       save_parquet(df_metric, "my_metric.parquet", stage="processed")
   
   if __name__ == "__main__":
       main()
   ```

3. **Use in analysis**:
   ```python
   # analysis/my_analysis.ipynb
   df = pd.read_parquet("../data/processed/my_metric.parquet")
   ```

### Code style

- Functions in `src/nes/` should be pure (no side effects) except for I/O functions
- Scripts should be thin wrappers that load config, call functions, and save results
- All magic numbers should go in `config.yaml`
- Use `print()` statements to log progress (no silent scripts)

## Reproducing Results

To fully reproduce the study from raw data:

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r environment/requirements.txt

# 2. Run pipeline
python scripts/01_download_stories.py
python scripts/02_clean_dataset.py
python scripts/03_compute_embeddings.py
python scripts/04_compute_sentiment.py

# 3. Run analysis notebooks (in Jupyter)
jupyter notebook analysis/
```

## Legacy Code

The `Berlin/` directory contains the original code structure. It is kept for reference during migration but should not be used for new analyses.

## Questions?

For questions about the pipeline or to report issues, contact the project maintainer.

## License

[Add license information]
