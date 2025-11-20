import sys
from pathlib import Path
import argparse
import yaml
import os
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.io import load_parquet, save_parquet, get_project_root, load_config
from nes.surface_metrics import get_descriptive_metrics_dual_full_long, get_descriptive_metrics_dual_inter_long

def main():

    df_full= pd.read_csv("/work/PENPAL/PENPAL_analysis/data/TEXT/interim/stories_full_text_filtered.csv")
    df_inter = pd.read_csv("/work/PENPAL/PENPAL_analysis/data/TEXT/interim/interaction_level_stories_filtered.csv")

    # Load config
    config = load_config()
    spacy_mdl = config['surface_metrics']['spacy_mdl']
    batch_size = config['surface_metrics']['batch_size']
    n_process = config['surface_metrics']['n_process']

    print(f"Computing Text Descriptives for: {config['active_dataset']}")

    df_descriptives_full = get_descriptive_metrics_dual_full_long(df_full,
        spacy_mdl = spacy_mdl, 
        batch_size = batch_size,
        n_process = n_process
        )
    save_parquet(df=df_descriptives_full, filename='full_story_surface_metrics.parquet')
    print('\nFinished computing for full stories\n')

    df_descriptives_inter = get_descriptive_metrics_dual_inter_long(df_inter,
    spacy_mdl = spacy_mdl, 
    batch_size = batch_size,
    n_process = n_process
    )
    print('\nFinished computing for interaction level stories\n')

    save_parquet(df=df_descriptives_inter, filename='interaction_level_surface_metrics.parquet')

if __name__ == "__main__":
    main()
