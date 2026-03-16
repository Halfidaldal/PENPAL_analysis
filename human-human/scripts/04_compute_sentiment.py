#!/usr/bin/env python
"""
Script 04: Compute sentiment scores.

This script:
1. Loads story data with embeddings
2. Computes sentiment scores for full_user and full_ai
3. Computes turn-by-turn dyadic sentiment
4. Saves results to data/processed/

Usage:
    python scripts/04_compute_sentiment.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import add_sentiment_to_dataframe, compute_dyadic_sentiment, compute_semantic_projection_batch
from nes.io import load_parquet, save_parquet, get_project_root, load_config
from nes.cleaning import append_turn_numbers


def main():
    # Load config
    config = load_config()
    active_dataset = config.get('active_dataset', 'TEXT')
    sentiment_config = config['sentiment'][active_dataset]
    simulated = config['cleaning'].get('simulated', False)
    
    print(f"Active dataset: {active_dataset}")
    
    # Load story embeddings data
    print("Loading story data with embeddings...")
    

    df_full = load_parquet("story_embeddings_full_simulated.parquet" if simulated else "story_embeddings_full.parquet", stage="processed")
    df_interaction_level = load_parquet("story_embeddings_interaction_level_simulated.parquet" if simulated else "story_embeddings_interaction_level.parquet", stage="processed")
    print(f"Loaded {len(df_full)} stories")
    
    # Filter to German stories only (adjust as needed)
    if config.get('active_dataset') == 'Berlin':
        df_de_full = df_full[df_full['language'] == 'de'].copy()
        df_de_interaction_level = df_interaction_level[df_interaction_level['language'] == 'de'].copy()
        print(f"Filtered to {len(df_de_full)} German stories")
    else:
        df_de_full = df_full.copy()
        df_de_interaction_level = df_interaction_level.copy()
    
    # Compute sentiment for full_user and full_ai
    print(f"\nComputing sentiment using {sentiment_config['model_name']}...")
    #df_with_sentiment = add_sentiment_to_dataframe(
    #    df_de_full,
    #    text_columns=['full_user', 'full_ai'],
    #    model_name=sentiment_config['model_name'],
    #    batch_size=sentiment_config['batch_size'],
    #    valence_method=sentiment_config['valence_method']
    #)
    
    # Save story-level sentiment
    #save_parquet(df_with_sentiment, "story_sentiment_scores_simulated.parquet" if args.simulated_data else "story_sentiment_scores.parquet", stage="processed")
    #print("✓ Saved story-level sentiment scores")
    
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
    # User turns
    print("Projecting user turns...")
    df_dyadic['user_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['user'].astype(str).tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # AI turns
    print("Projecting AI turns...")
    df_dyadic['ai_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['ai'].astype(str).tolist(),
        batch_size=sentiment_config['batch_size']
    )
    
    # Save dyadic sentiment
    save_parquet(df_dyadic, "dyadic_sentiment_scores_simulated.parquet" if simulated else "dyadic_sentiment_scores.parquet", stage="processed")
    print(f"✓ Saved dyadic sentiment for {len(df_dyadic)} turns")
    
    print("\n✅ Script 04 complete!")


if __name__ == "__main__":
    main()
