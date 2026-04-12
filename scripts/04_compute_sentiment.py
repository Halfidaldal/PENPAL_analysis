#!/usr/bin/env python
"""
Script 04: Compute sentiment scores using semantic projection.

This script:
1. Loads story data with embeddings
2. Computes sentiment via concept vector projection for author_1 and author_2
3. Saves results to data/<experiment>/processed/

Usage:
    python scripts/04_compute_sentiment.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import compute_semantic_projection_batch
from nes.io import (
    backfill_interaction_metadata,
    get_active_experiment,
    get_experiment_config,
    get_shared_config,
    load_parquet,
    save_parquet,
)


def main():
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    sentiment_config = shared_config['sentiment']
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")
    
    # Load story embeddings data
    print("Loading story data with embeddings...")
    
    df_interaction_level = load_parquet(
        "story_embeddings_interaction_level_simulated.parquet" if simulated else "story_embeddings_interaction_level.parquet",
        stage="processed"
    )
    df_interaction_level = backfill_interaction_metadata(df_interaction_level, simulated=simulated)
    print(f"Loaded {len(df_interaction_level)} interaction rows")
    df_dyadic = df_interaction_level.copy()
    
    # Compute Semantic Projection Sentiment
    print(f"\nComputing Semantic Projection Sentiment (batch_size={sentiment_config['batch_size']})...")
    
    # Author 1 turns
    print("Projecting author_1 turns...")
    df_dyadic['author_1_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_1'].tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # Author 2 turns
    print("Projecting author_2 turns...")
    df_dyadic['author_2_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_2'].tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # Save results
    output_file = "dyadic_sentiment_scores_simulated.parquet" if simulated else "dyadic_sentiment_scores.parquet"
    save_parquet(df_dyadic, output_file, stage="processed")
    print(f"✓ Saved sentiment for {len(df_dyadic)} turns to {output_file}")
    
    print("\n✅ Script 04 complete!")


if __name__ == "__main__":
    main()
