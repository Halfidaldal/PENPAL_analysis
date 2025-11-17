import sys
from pathlib import Path
import argparse
import yaml
import os
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.io import load_parquet, save_parquet, get_project_root, load_config
from nes.surface_metrics import get_descriptive_metrics_dual_long

df_full = pd.read_csv('/work/PENPAL/PENPAL_analysis/data/TEXT/interim/stories_full_text_filtered.csv')

def main(): 
    df_descriptives = get_descriptive_metrics_dual_long(df=df_full)
    save_parquet(df=df_descriptives, filename='surface_metrics.parquet')

if __name__ == "__main__":
    main()
