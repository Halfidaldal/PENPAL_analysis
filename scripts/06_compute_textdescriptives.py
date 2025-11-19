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

def main():
    path = "/work/PENPAL/PENPAL_analysis/data/TEXT/interim/stories_full_text_filtered.csv"
    df = pd.read_csv(path)
    parser = argparse.ArgumentParser(description="Compute textdescriptives for story data.")
    
    # Load config
    config = load_config()
    spacy_mdl = config['surface_metrics']['spacy_mdl']
    batch_size = config['surface_metrics']['batch_size']
    n_process = config['surface_metrics']['n_process']

    print(f"Cleaning Simulated: {config['active_dataset']}")

    df_descriptives = get_descriptive_metrics_dual_long(df,
        spacy_mdl = spacy_mdl, 
        batch_size = batch_size,
        n_process = n_process
        )
    save_parquet(df=df_descriptives, filename='surface_metrics.parquet')

if __name__ == "__main__":
    main()
