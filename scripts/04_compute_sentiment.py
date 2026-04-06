#!/usr/bin/env python
"""
Script 04: Compute sentiment scores.

This script:
1. Loads story data with embeddings
2. Computes sentiment scores for author_1 and author_2
3. Computes turn-by-turn dyadic sentiment
4. Saves results to data/<experiment>/processed/

Usage:
    python scripts/04_compute_sentiment.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import add_sentiment_to_dataframe, compute_dyadic_sentiment, compute_semantic_projection_batch
from nes.io import load_parquet, save_parquet, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config
from nes.cleaning import append_turn_numbers


def main():
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    sentiment_config = shared_config['sentiment']
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")
    
    # Determine column names based on experiment
    author_2_col = 'ai' if experiment == 'human-ai' else 'user2'
    
    # Load story embeddings data
    print("Loading story data with embeddings...")
    
    df_full = load_parquet("story_embeddings_full_simulated.parquet" if simulated else "story_embeddings_full.parquet", stage="processed")
    df_interaction_level = load_parquet("story_embeddings_interaction_level_simulated.parquet" if simulated else "story_embeddings_interaction_level.parquet", stage="processed")
    print(f"Loaded {len(df_full)} stories")
    
    df_de_full = df_full.copy()
    df_de_interaction_level = df_interaction_level.copy()
    
    # Compute sentiment
    print(f"\nComputing sentiment using {sentiment_config['model_name']}...")
    
    # Compute dyadic (turn-by-turn) sentiment
    print("\nComputing turn-by-turn sentiment...")
    df_dyadic = compute_dyadic_sentiment(
        append_turn_numbers(df_de_interaction_level),
        valence_method=sentiment_config['valence_method'],
        batch_size=sentiment_config['batch_size'],
        model_name=sentiment_config['model_name']
    )
    
    # Compute Semantic Projection Sentiment
    print("\nComputing Semantic Projection Sentiment...")
    # Author 1 turns
    print("Projecting author_1 turns...")
    df_dyadic['user_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['user'].astype(str).tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # Author 2 turns
    print(f"Projecting {author_2_col} turns...")
    df_dyadic[f'{author_2_col}_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic[author_2_col].astype(str).tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # Save dyadic sentiment
    save_parquet(df_dyadic, "dyadic_sentiment_scores_simulated.parquet" if simulated else "dyadic_sentiment_scores.parquet", stage="processed")
    print(f"✓ Saved dyadic sentiment for {len(df_dyadic)} turns")
    
    print("\n✅ Script 04 complete!")


if __name__ == "__main__":
    main()
