import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path("src")))
from nes.io import load_csv, get_active_experiment
df = load_csv("interaction_level_stories_filtered.csv", stage="interim")
print(df.columns.tolist())
if 'author_1' in df.columns:
    texts = df['author_1'].tolist()
    print(f"Total texts: {len(texts)}")
    print(f"Text 188: {repr(texts[188])}")
    print(f"Text 189: {repr(texts[189])}")
    print(f"Text 190: {repr(texts[190])}")
