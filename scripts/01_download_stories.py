#!/usr/bin/env python
"""
Script 01: Download and clean raw story data from Firestore.

This script:
1. Downloads stories from Firestore
2. Filters to keep only complete stories (≥10 interactions)
3. Saves raw data to data/raw/
4. Optionally deletes incomplete stories from Firestore

Usage:
    python scripts/01_download_stories.py [--delete-incomplete]
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.cleaning import init_firestore, download_stories_from_firestore
from nes.io import save_csv, get_project_root, load_config


def main():
    
    # Load config
    config = load_config()
    firestore_config = config['firestore']
    
    print(f"Active dataset: {config.get('active_dataset', 'TEXT')}")
    
    # Initialize Firestore
    print("Initializing Firestore client...")
    credentials_path = get_project_root() / firestore_config['credentials_path']
    db = init_firestore(str(credentials_path))
    
    # Download stories
    print(f"\nDownloading stories from collection: {firestore_config['collection_name']}")
    df_stories = download_stories_from_firestore(
        db,
        collection_name=firestore_config['collection_name'],
        min_interactions=firestore_config['min_interactions']
    )
    
    # Save to raw data
    output_filename = "finished_stories_raw.csv"
    save_csv(df_stories, output_filename, stage="raw")
    
    print(f"\n✓ Downloaded {len(df_stories)} interaction rows")
    print(f"✓ Saved to data/{config.get('active_dataset', 'TEXT')}/raw/{output_filename}")
    
    print("\n✅ Script 01 complete!")


if __name__ == "__main__":
    main()
