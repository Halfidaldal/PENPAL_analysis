#!/usr/bin/env python
"""
Script 03: Compute embeddings for story data.

This script:
1. Loads cleaned story data
2. Computes embeddings for full_story, full_user, full_ai
3. Saves embeddings as both:
   - Parquet file (with embeddings as list columns)
   - Separate .npy files for each embedding type

Usage:
    python scripts/03_compute_embeddings.py
"""

import sys
from pathlib import Path
import yaml
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.embeddings import compute_story_embeddings_full_stories, embed_story_columns
from nes.io import load_csv, save_parquet, save_npy, get_project_root


def load_config():
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()
    embeddings_config = config['embeddings']
    
    parser = argparse.ArgumentParser(description="Compute embeddings for story data.")
    parser.add_argument(
        "--simulated-data",
        type=bool,
        default=False,
        help="Whether the input data is simulated (affects file naming).")

    args = parser.parse_args()

    # Load cleaned story data
    print("Loading cleaned story data...")
    df_full = load_csv("stories_full_text_filtered.csv" if not args.simulated_data else "stories_full_text_filtered_simulated.csv", stage="interim")
    df_interactions = load_csv("interaction_level_stories_filtered.csv" if not args.simulated_data else "interaction_level_stories_filtered_simulated.csv", stage="interim")
    print(f"Loaded {len(df_full)} full stories and {len(df_interactions)} interaction-level stories")
    
    # Compute embeddings
    print(f"\nComputing embeddings using {embeddings_config['model_name']}...")
    df_embedded, story_emb, user_emb, ai_emb = compute_story_embeddings_full_stories(
        df_full,
        model_name=embeddings_config['model_name'],
        batch_size=embeddings_config['batch_size']
    )
    
    df_embedded_interaction, embeddings_interaction_dict = embed_story_columns(
        df_interactions,
        ['user', 'ai'],
        model_name=embeddings_config['model_name'],
        batch_size=embeddings_config['batch_size']
    )
    
    # Save parquet with embeddings as list columns
    print("\nSaving embeddings...")
    save_parquet(df_embedded, "story_embeddings_field_full.parquet" if not args.simulated_data else "story_embeddings_full_simulated.parquet", stage="processed")
    save_parquet(df_embedded_interaction, "story_embeddings_interaction_level.parquet" if not args.simulated_data else "story_embeddings_interaction_level_simulated.parquet", stage="processed") 
    
    # Save individual .npy files for numpy arrays
    save_npy(story_emb, "story_embeddings_jina_field_full.npy" if not args.simulated_data else "story_embeddings_jina_full_simulated.npy",  stage="processed")
    save_npy(user_emb, "story_user_embeddings_jina_field_full.npy" if not args.simulated_data else "story_user_embeddings_jina_full_simulated.npy", stage="processed")
    save_npy(ai_emb, "story_ai_embeddings_jina_field_full.npy" if not args.simulated_data else "story_ai_embeddings_jina_full_simulated.npy", stage="processed")
        
    print(f"\n✓ Computed embeddings for {len(df_embedded)} stories")
    print(f"✓ Embedding dimension: {story_emb.shape[1]}")
    print("✓ Saved to data/processed/")
    print("\n✅ Script 03 complete!")


if __name__ == "__main__":
    main()
