#!/usr/bin/env python
"""
Script 02: Filter and clean story data.

This script:
1. Loads raw story data
2. Applies edit distance filtering
3. Builds full story text (full_story, full_user, full_ai columns)
4. Saves cleaned data to data/interim/

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
from nes.cleaning import filter_by_edit_distance, build_full_story_text, filter_by_respondent_id
from nes.io import load_csv, save_csv, get_project_root, load_config


def main():
    
    parser = argparse.ArgumentParser(description="Filter and clean story data.")
    parser.add_argument(
        "--input-csv-raw",
        type=str,
        default="finished_stories_raw.csv",
        help="Path to input CSV file with raw story data."
    )
    parser.add_argument(
        "--include-spell-correction",
        type=bool,
        default=False,
        help="Whether to include spell correction in the cleaning process."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key and args.include_spell_correction:
        print("❌ Error: OpenAI API key required for spell correction!")
        print("   Set via --api-key flag or OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Load config
    config = load_config()
    edit_distance_threshold = config['cleaning']['edit_distance_threshold']
    
    print(f"Active dataset: {config.get('active_dataset', 'TEXT')}")
    
    # Load raw data
    print("Loading raw story data...")
    df = load_csv(args.input_csv_raw, stage="raw")
    print(f"Loaded {len(df)} rows")
    
    if args.include_spell_correction is True:
        print("\nApplying spell correction to user inputs...")
        df['user_corrected'] = [correct_spelling(text, api_key=api_key) for text in tqdm(df["user"], desc="Spell Correction")]
        print("Spell correction complete.")
        #compute edit distance
        df['edit_distance'] = [compute_edit_distance(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Edit Distance Computation")]
        print("Edit distance computation complete.")
        
        # Use corrected user input for further processing
        df['user'] = df['user_corrected']
        df.drop(columns=['user_corrected'], inplace=True)
    else:
        print("\nSkipping spell correction as per user request.")
    
    # Filter by edit distance (if column exists)
    if 'edit_distance' in df.columns:
        print(f"\nFiltering by edit distance (threshold={edit_distance_threshold})...")
        df_filtered = filter_by_edit_distance(df, threshold=edit_distance_threshold)
    else:
        print("\nNo edit_distance column found, skipping filter")
        df_filtered = df.copy()
        
    if 'respondent_id' in df.columns:
        print("\nFiltering by respondent ID...")
        df_filtered = filter_by_respondent_id(df_filtered, threshold=12)
        print(f"✓ Filtered to {len(df_filtered)} rows with valid respondent IDs")
    
    if 'interaction_count' in df.columns:
        df_filtered = clean_user_ai_start(df_filtered)
    
    else: # adds 'turn' to df in no interaction_count found 
        df_filtered = clean_user_ai_start(df_filtered, interaction_count=False)

    print("\nBuilding full story text...")
    # Save filtered interaction-level data
    save_csv(df_filtered, "interaction_level_stories_filtered.csv" if args.include_spell_correction else "interaction_level_stories_filtered_simulated.csv", stage="interim")
    df_stories = build_full_story_text(df_filtered)
    save_csv(df_stories, "stories_full_text_filtered.csv" if args.include_spell_correction else "stories_full_text_filtered_simulated.csv", stage="interim")
    
    
    print(f"\n✓ Filtered to {len(df_filtered)} interaction rows")
    print(f"✓ Built {len(df_stories)} complete stories")
    print(f"✓ Saved to data/{config.get('active_dataset', 'TEXT')}/interim/")
    print("\n✅ Script 02 complete!")


if __name__ == "__main__":
    main()
