# Migration Guide: From Berlin/ to New Structure

This guide helps you transition from the old `Berlin/` structure to the new pipeline-based organization.

## What's Changed?

### Before (Berlin/)
- Mixed scripts and notebooks
- Pipeline logic embedded in notebooks
- Hardcoded paths everywhere
- Unclear execution order
- Data scattered in multiple locations

### After (New Structure)
- Clear separation: scripts for pipeline, notebooks for analysis
- All pipeline logic in versioned modules (`src/nes/`)
- Centralized configuration (`config.yaml`)
- Numbered scripts show execution order
- Organized data flow: raw → interim → processed

## Migration Steps

### Step 1: Install Dependencies

```bash
# Create new environment
python3 -m venv venv
source venv/bin/activate

# Install from new requirements
pip install -r environment/requirements.txt
```

### Step 2: Migrate Your Data

Copy existing data files to the new structure:

```bash
# Raw data (original Firestore exports)
cp Berlin/Data/finished_stories_raw.csv data/raw/

# Processed data (if you want to skip recomputing)
cp Berlin/Data/finished_stories_corrected.csv data/interim/stories_filtered.csv
cp Berlin/Data/story_text_field_w_embeddings_jina.parquet data/processed/story_embeddings_field.parquet
cp Berlin/Data/*.npy data/processed/
```

### Step 3: Run the Pipeline

If you're starting fresh (recommended for reproducibility):

```bash
python scripts/01_download_stories.py  # Download from Firestore
python scripts/02_clean_dataset.py      # Clean and filter
python scripts/03_compute_embeddings.py # Compute embeddings
python scripts/04_compute_sentiment.py  # Compute sentiment
```

Or if you're using existing processed data, you can skip to analysis.

### Step 4: Migrate Your Notebooks

For each notebook in `Berlin/src/`:

1. **Identify pipeline logic vs. analysis**
   - Pipeline: data loading, cleaning, embedding, metric computation
   - Analysis: plots, statistical tests, summaries

2. **Extract pipeline logic**
   - If a notebook computes something new (e.g., novelty scores), move that logic to `src/nes/novelty.py`
   - Create a corresponding script like `scripts/05_compute_novelty.py`

3. **Simplify the notebook**
   - Remove all pipeline cells
   - Replace with simple data loading:
     ```python
     import pandas as pd
     df = pd.read_parquet("../data/processed/my_metric.parquet")
     ```
   - Keep only visualization and statistical analysis

4. **Move to analysis/**
   ```bash
   cp Berlin/src/sentiment.ipynb analysis/01_sentiment_analysis.ipynb
   # Then clean it as described above
   ```

### Example: Migrating sentiment.ipynb

**Before** (Berlin/src/sentiment.ipynb):
```python
# Cell 1: Setup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
...

# Cell 2: Load data
df = pd.read_parquet('/work/PENPAL/...')

# Cell 3-10: Compute embeddings, sentiment scores, etc.
model = AutoModelForSequenceClassification.from_pretrained(...)
df['sentiment'] = ...

# Cell 11-20: Analysis and plots
sns.lineplot(...)
```

**After** (analysis/01_sentiment_analysis.ipynb):
```python
# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Load processed data
df_sentiment = pd.read_parquet("../data/processed/story_sentiment_scores.parquet")
df_dyadic = pd.read_parquet("../data/processed/dyadic_sentiment_scores.parquet")

# Cell 3-N: Analysis and plots only
sns.lineplot(data=df_dyadic, x="pct_turn", y="sentiment_score", hue="type")
plt.title('Dyadic sentiment over story progression')
```

The computation logic goes into:
- `src/nes/sentiment.py` (functions)
- `scripts/04_compute_sentiment.py` (script that calls those functions)

### Step 5: Update Your Workflow

**Old workflow:**
1. Open notebook
2. Run cells 1-50 to process data
3. Run cells 51-100 for analysis
4. Repeat for every analysis

**New workflow:**
1. Run pipeline scripts once: `python scripts/0*.py`
2. Open analysis notebook
3. Run all cells for analysis
4. Analysis notebooks are fast and reproducible

## Common Migration Patterns

### Pattern 1: Embedding Computation

**Old code (in notebook):**
```python
model = SentenceTransformer("jinaai/jina-embeddings-v3")
embeddings = model.encode(df['full_story'].tolist())
df['embedding'] = embeddings.tolist()
```

**New approach:**
- Function lives in `src/nes/embeddings.py`
- Script `scripts/03_compute_embeddings.py` calls it
- Notebook just loads: `df = pd.read_parquet("../data/processed/story_embeddings_field.parquet")`

### Pattern 2: Custom Metrics

**Old code (scattered in notebook):**
```python
def compute_novelty(embeddings):
    # ... complex logic ...
    return novelty_scores

novelty = compute_novelty(embeddings)
df['novelty'] = novelty
```

**New approach:**
1. Create `src/nes/novelty.py`:
   ```python
   def compute_novelty(embeddings):
       # ... same logic ...
       return novelty_scores
   ```

2. Create `scripts/05_compute_novelty.py`:
   ```python
   from nes.novelty import compute_novelty
   from nes.io import load_npy, load_parquet, save_parquet
   
   embeddings = load_npy("story_embeddings_jina_field.npy", stage="processed")
   df = load_parquet("story_embeddings_field.parquet", stage="processed")
   
   novelty = compute_novelty(embeddings)
   df['novelty'] = novelty
   
   save_parquet(df, "story_novelty.parquet", stage="processed")
   ```

3. In notebook:
   ```python
   df = pd.read_parquet("../data/processed/story_novelty.parquet")
   ```

### Pattern 3: Handling Hardcoded Paths

**Old:**
```python
df = pd.read_csv("/Users/you/Documents/PENPAL/Berlin/Data/stories.csv")
```

**New:**
```python
from nes.io import load_csv
df = load_csv("stories.csv", stage="processed")
```

## Configuration Changes

All parameters now live in `config.yaml`:

```yaml
# Instead of magic numbers in code
EDIT_DISTANCE_THRESHOLD = 100
MODEL_NAME = "jinaai/jina-embeddings-v3"
BATCH_SIZE = 32

# Use config
cleaning:
  edit_distance_threshold: 100
  
embeddings:
  model_name: "jinaai/jina-embeddings-v3"
  batch_size: 32
```

Scripts automatically load from config:
```python
config = load_config()
threshold = config['cleaning']['edit_distance_threshold']
```

## Checklist for Complete Migration

- [ ] Installed new environment from `environment/requirements.txt`
- [ ] Copied/moved data files to `data/raw/`, `data/interim/`, `data/processed/`
- [ ] Ran pipeline scripts to regenerate processed data
- [ ] Identified all unique analyses in old notebooks
- [ ] Extracted pipeline logic to `src/nes/` modules
- [ ] Created corresponding scripts in `scripts/`
- [ ] Cleaned analysis notebooks to be read-only consumers
- [ ] Moved cleaned notebooks to `analysis/`
- [ ] Updated `config.yaml` with project-specific parameters
- [ ] Tested running the entire pipeline from scratch
- [ ] Verified analysis notebooks run top-to-bottom

## Benefits of New Structure

1. **Reproducibility**: Anyone can run `python scripts/0*.py` and get the same results
2. **Efficiency**: Don't recompute embeddings every time you make a plot
3. **Clarity**: Clear separation between data processing and analysis
4. **Version control**: Easier to track changes in `.py` files than notebooks
5. **Collaboration**: Team members can work on different scripts/analyses independently
6. **Testing**: Can write unit tests for functions in `src/nes/`
7. **Documentation**: Pipeline flow is explicit and documented

## Keeping Berlin/ for Reference

The `Berlin/` directory is kept as-is during migration. You can:
- Compare old vs. new outputs for validation
- Reference complex logic that hasn't been migrated yet
- Keep as a backup

Once you're confident in the new structure, you can archive `Berlin/` or add a note that it's deprecated.

## Need Help?

If you encounter issues during migration:
1. Check that paths in `config.yaml` are correct
2. Verify Python environment has all dependencies
3. Look at existing scripts (01-04) as examples
4. Test small pieces incrementally

Happy migrating! 🚀
