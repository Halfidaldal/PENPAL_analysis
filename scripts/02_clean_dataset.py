#!/usr/bin/env python
"""
Script 02: Filter and clean story data.

This script:
1. Loads raw story data
2. Applies edit distance filtering (optional, human experiments only)
3. Removes "This is the story of" prefix
4. Builds full story text (full_story, full_author_1, full_author_2 columns)
5. Saves cleaned data to data/<experiment>/interim/

Supports all three conditions:
- human-ai: Human-AI collaborative stories
- human-human: Human-Human collaborative stories  
- ai-ai: Simulated AI-AI stories

Usage:
    python scripts/02_clean_dataset.py
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.process_spelling_openai import correct_spelling, compute_edit_distance
from nes.cleaning import filter_by_edit_distance, build_full_story_text, filter_by_respondent_id, clean_user_ai_start, clean_ai_ai_data
from nes.io import load_csv, save_csv, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config


def main():
    
    parser = argparse.ArgumentParser(description="Filter and clean story data.")
    parser.add_argument(
        "--input-csv-raw",
        type=str,
        default=None,
        help="Path to input CSV file with raw story data (auto-detected based on experiment)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    
    edit_distance_threshold = shared_config['cleaning']['edit_distance_threshold']
    simulated = shared_config['cleaning']['simulated']
    max_turns = exp_config['cleaning']['max_turns']

    print("=" * 60)
    print(f"Script 02: Clean Dataset ({experiment})")
    print("=" * 60)
    print(f"Simulated mode: {simulated}")

    # Determine input file based on experiment
    if args.input_csv_raw:
        input_file = args.input_csv_raw
    elif experiment == 'ai-ai':
        input_file = "simulated_stories.csv"
    else:
        input_file = "finished_stories_raw.csv"
    
    # Load raw data
    print(f"\nLoading raw story data from {input_file}...")
    df = load_csv(input_file, stage="raw")
    print(f"Loaded {len(df)} rows")
    
    # AI-AI has its own cleaning path
    if experiment == 'ai-ai':
        print("\n--- AI-AI Cleaning Pipeline ---")
        df_filtered = clean_ai_ai_data(df, max_turns=max_turns)
        
    else:
        # Human experiments: optional spell correction + edit distance filtering
        if api_key and experiment in ['human-ai', 'human-human']:
            print("\nApplying spell correction to user inputs...")
            df['user_corrected'] = [correct_spelling(text, api_key=api_key) for text in tqdm(df["user"], desc="Spell Correction")]
            print("Spell correction complete.")
            df['edit_distance'] = [compute_edit_distance(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Edit Distance Computation")]
            print("Edit distance computation complete.")
            
            df['user'] = df['user_corrected']
            df.drop(columns=['user_corrected'], inplace=True)
        else:
            print("\nSkipping spell correction (no API key or not applicable)")
        
        # Filter by edit distance (if column exists)
        if 'edit_distance' in df.columns:
            print(f"\nFiltering by edit distance (threshold={edit_distance_threshold})...")
            df_filtered = filter_by_edit_distance(df, threshold=edit_distance_threshold)
        else:
            print("\nNo edit_distance column found, skipping filter")
            df_filtered = df.copy()
        
        # Clean starter text and identify who started
        df_filtered = clean_user_ai_start(df_filtered, max_turns=max_turns, experiment=experiment)
        
        # Filter by respondent_id (human-ai only)
        if 'respondent_id' in df_filtered.columns and experiment == 'human-ai':
            print("\nFiltering by respondent ID...")
            df_filtered = filter_by_respondent_id(df_filtered, threshold=12)
            print(f"✓ Filtered to {len(df_filtered)} rows with valid respondent IDs")
        else:
            print("\nNo respondent_id filtering (not applicable for this experiment)")

    print("\nBuilding full story text...")
    # Save filtered interaction-level data
    output_interaction = "interaction_level_stories_filtered_simulated.csv" if simulated else "interaction_level_stories_filtered.csv"
    save_csv(df_filtered, output_interaction, stage="interim")
    
    df_stories = build_full_story_text(df_filtered, experiment=experiment)
    output_stories = "stories_full_text_filtered_simulated.csv" if simulated else "stories_full_text_filtered.csv"
    save_csv(df_stories, output_stories, stage="interim")
    
    print(f"\n✓ Filtered to {len(df_filtered)} interaction rows")
    print(f"✓ Built {len(df_stories)} complete stories")
    print(f"✓ Saved to {exp_config['interim_dir']}/")
    print("\n✅ Script 02 complete!")


if __name__ == "__main__":
    main()
