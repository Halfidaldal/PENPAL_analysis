# Repository Restructuring Summary

## What Was Done

I've restructured your PENPAL analysis repository to follow best practices for reproducible computational research. Here's what was created:

### 1. New Directory Structure ✅

```
PENPAL_analysis/
├── data/
│   ├── raw/          # Firestore exports, never modified
│   ├── interim/      # Cleaned, filtered data
│   └── processed/    # Final analysis-ready data
├── src/
│   └── nes/          # Core analysis package (5 modules)
├── scripts/          # Pipeline entry points (4 scripts)
├── analysis/         # Analysis-only notebooks
├── environment/      # requirements.txt
├── config.yaml       # All configuration in one place
├── README.md         # Full documentation
└── MIGRATION.md      # Guide for moving from Berlin/
```

### 2. Core Modules Created (`src/nes/`)

**io.py** - Standardized data loading/saving
- Functions: `load_csv()`, `save_parquet()`, `load_npy()`, etc.
- Handles path management automatically
- Supports raw/interim/processed stages

**cleaning.py** - Data cleaning and preprocessing
- `init_firestore()` - Initialize Firestore client
- `download_stories_from_firestore()` - Download complete stories only
- `delete_incomplete_stories_from_firestore()` - Clean up Firestore
- `filter_by_edit_distance()` - Apply edit distance threshold
- `build_full_story_text()` - Create full_story/full_user/full_ai columns

**embeddings.py** - Text embedding computation
- `compute_embeddings_batch()` - Batch embedding with progress bars
- `embed_story_columns()` - Embed multiple columns
- `compute_story_embeddings_field()` - Full pipeline for story/user/AI embeddings
- Supports Jina, E5, and other SentenceTransformer models

**sentiment.py** - Sentiment analysis
- `compute_sentiment_batch()` - Batch sentiment scoring
- `continuous_valence_score()` - Three methods: simple/amplify/dampen
- `add_sentiment_to_dataframe()` - Add sentiment columns
- `compute_dyadic_sentiment()` - Turn-by-turn sentiment analysis

**simulation.py** - AI-AI baseline generation
- Placeholder for future LLM-based simulation
- Framework for human-like variation

### 3. Pipeline Scripts (`scripts/`)

**01_download_stories.py**
- Downloads stories from Firestore
- Filters to complete stories (≥10 interactions)
- Saves to `data/raw/finished_stories_raw.csv`
- Optional: `--delete-incomplete` flag to clean Firestore

**02_clean_dataset.py**
- Loads raw data
- Applies edit distance filtering
- Builds full story text columns
- Saves to `data/interim/`

**03_compute_embeddings.py**
- Computes Jina embeddings for stories, users, AI
- Saves both Parquet (with list columns) and NumPy arrays
- Output: `data/processed/story_embeddings_field.parquet` + `.npy` files

**04_compute_sentiment.py**
- Computes story-level sentiment (full_user, full_ai)
- Computes dyadic turn-by-turn sentiment
- Saves to `data/processed/`

### 4. Configuration (`config.yaml`)

Centralized all parameters:
- Firestore settings (collection names, credentials path)
- Cleaning parameters (edit distance threshold = 100)
- Model settings (Jina embeddings, German BERT sentiment)
- Batch sizes, random seeds

### 5. Documentation

**README.md** - Complete pipeline documentation
- Setup instructions
- How to run the full pipeline
- Data outputs table
- Development guidelines
- Code style conventions

**MIGRATION.md** - Step-by-step migration guide
- How to move from Berlin/ structure
- Common patterns for extracting notebook logic
- Before/after examples
- Checklist for complete migration

**environment/requirements.txt** - Clean dependency list
- Core: pandas, numpy, pyarrow
- Firebase: firebase-admin
- ML: torch, transformers, sentence-transformers
- Utils: tqdm, pyyaml, matplotlib, seaborn

## How to Use

### Quick Start

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r environment/requirements.txt

# 2. Run pipeline
python scripts/01_download_stories.py
python scripts/02_clean_dataset.py
python scripts/03_compute_embeddings.py
python scripts/04_compute_sentiment.py

# 3. Analyze
jupyter notebook analysis/
```

### Adding New Metrics

1. Write function in `src/nes/mymetric.py`
2. Create `scripts/05_compute_mymetric.py` that calls it
3. Run script to generate processed data
4. Load in notebook: `pd.read_parquet("../data/processed/mymetric.parquet")`

## Key Principles

### Scripts vs. Notebooks

**Scripts** (modules + entry points):
- Data cleaning, filtering
- Computing embeddings, sentiment, metrics
- Any transformation that later analyses depend on
- Versioned, testable, reproducible

**Notebooks** (analysis only):
- Load processed data
- Visualization, statistical tests, summaries
- No new canonical columns
- Runnable top-to-bottom

### Data Flow

```
Firestore → raw/ → interim/ → processed/ → analysis notebooks
            ↓       ↓          ↓
         script01 script02  script03/04
```

### Configuration Over Code

Instead of:
```python
THRESHOLD = 100  # scattered throughout code
```

Use:
```yaml
# config.yaml
cleaning:
  edit_distance_threshold: 100
```

## Next Steps for You

### Immediate (To Start Using New Structure)

1. **Install environment**:
   ```bash
   pip install -r environment/requirements.txt
   ```

2. **Copy your existing processed data** (optional, to skip recomputation):
   ```bash
   cp Berlin/Data/finished_stories_corrected.csv data/interim/
   cp Berlin/Data/*.parquet data/processed/
   cp Berlin/Data/*.npy data/processed/
   ```

3. **Test a script**:
   ```bash
   # If you have Firestore access
   python scripts/01_download_stories.py
   
   # Or if you copied data
   python scripts/02_clean_dataset.py
   ```

### Short Term (Migration)

4. **Identify your key analyses**:
   - List the notebooks in `Berlin/src/` that produce paper figures/tables
   - Note which ones contain pipeline logic (data processing)

5. **Extract pipeline logic one notebook at a time**:
   - For sentiment: Already done! (scripts/04_compute_sentiment.py)
   - For novelty/exploration: Create scripts/05-06
   - For alignment: Create script 07

6. **Clean and move notebooks to `analysis/`**:
   - Remove all data processing cells
   - Keep only visualization + statistical analysis
   - Update paths to load from `../data/processed/`

### Long Term (Best Practices)

7. **Add unit tests** (optional but recommended):
   ```python
   # tests/test_cleaning.py
   def test_filter_by_edit_distance():
       df = pd.DataFrame({'edit_distance': [50, 150, 80]})
       filtered = filter_by_edit_distance(df, threshold=100)
       assert len(filtered) == 2
   ```

8. **Version control your config**:
   - Track `config.yaml` in git
   - Document parameter changes in commit messages

9. **Create a final "paper reproduction" script**:
   ```bash
   #!/bin/bash
   # reproduce_paper.sh
   python scripts/01_download_stories.py
   python scripts/02_clean_dataset.py
   ...
   jupyter nbconvert --execute analysis/*.ipynb
   ```

## What You Can Do Right Now

### Option A: Start Fresh (Recommended)

Run the full pipeline from scratch to verify everything works:

```bash
python scripts/01_download_stories.py  # ~5 min
python scripts/02_clean_dataset.py      # ~1 min
python scripts/03_compute_embeddings.py # ~20-60 min depending on GPU
python scripts/04_compute_sentiment.py  # ~10-30 min
```

This ensures:
- You understand each step
- Output files are in the right place
- No hidden dependencies on old structure

### Option B: Quick Migration (If Time Constrained)

Copy existing processed files and jump to analysis:

```bash
# Copy processed data
cp Berlin/Data/story_text_field_w_embeddings_jina.parquet data/processed/story_embeddings_field.parquet

# Pick one notebook to migrate as a test
# Example: sentiment.ipynb
# 1. Copy it: cp Berlin/src/sentiment.ipynb analysis/01_sentiment.ipynb
# 2. Open and remove cells 1-15 (data processing)
# 3. Replace with: df = pd.read_parquet("../data/processed/dyadic_sentiment_scores.parquet")
# 4. Run and verify plots still work
```

### Option C: Parallel Work

Keep using `Berlin/` for active work while gradually migrating:

1. New analyses → use new structure
2. Old analyses → keep in Berlin/ for now
3. Migrate one notebook per week
4. After 1-2 months, archive Berlin/

## Benefits You'll See

✅ **Reproducibility**: "Clone repo, run 4 scripts, generate all results"

✅ **Speed**: Recomputing embeddings takes 30 min, but you only do it once

✅ **Clarity**: Anyone can understand: `scripts/01 → 02 → 03 → 04`

✅ **Collaboration**: "Run script 05 to get novelty scores" vs. "Run cells 34-67 in notebook X"

✅ **Version Control**: See what actually changed when you `git diff` a `.py` file

✅ **Testing**: Can verify that `filter_by_edit_distance(df, 100)` works correctly

## Questions?

Common questions answered in README.md and MIGRATION.md:

- "How do I add a new metric?" → See README "Development" section
- "Can I still use notebooks?" → Yes! For analysis and exploration
- "What if I need to change a parameter?" → Edit config.yaml and rerun script
- "Do I have to delete Berlin/?" → No, keep it as reference during migration

---

**Status**: ✅ Structure created, core modules written, pipeline scripts ready, documentation complete

**Your next action**: Choose Option A, B, or C above and start testing!
