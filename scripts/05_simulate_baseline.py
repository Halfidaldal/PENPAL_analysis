#!/usr/bin/env python
"""
Script 05: Simulate AI-AI baseline conversations.

This script:
1. Generates simulated conversations between two AI agents
2. Uses workshop-specific prompts (1, 2, 3)
3. Uses client-specific prompts (Utopian/Dystopian)
4. Saves simulated data for null hypothesis testing

Usage:
    python scripts/05_simulate_baseline.py --api-key YOUR_API_KEY
    
    Or set environment variable:
    export OPENAI_API_KEY=your_key
    python scripts/05_simulate_baseline.py
"""

import sys
from pathlib import Path
import argparse
import yaml
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.simulation import simulate_ai_ai_conversations
from nes.io import save_csv, get_project_root


def load_config():
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Simulate AI-AI baseline conversations")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=None,
        help="Number of turns per conversation (defaults to config)"
    )
    parser.add_argument(
        "--client-ids",
        nargs="+",
        default=["1", "2"],
        help="Client IDs to simulate (default: 1 2)"
    )
    parser.add_argument(
        "--workshop-ids",
        nargs="+",
        default=["1", "2", "3"],
        help="Workshop IDs to simulate (default: 1 2 3)"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    simulation_config = config['simulation']
    
    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OpenAI API key required!")
        print("   Set via --api-key flag or OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Get n_interactions
    n_interactions = args.n_interactions or simulation_config.get('n_turns_per_story', 10)
    n_simulations = simulation_config.get('n_simulations', 10)
    
    # Simulate conversations
    print(f"Simulating conversations:")
    print(f"  Client IDs: {args.client_ids}")
    print(f"  Workshop IDs: {args.workshop_ids}")
    print(f"  Interactions per conversation: {n_interactions}")
    print(f"  Total conversations: {len(args.client_ids) * len(args.workshop_ids)}")
    
    df_simulated = simulate_ai_ai_conversations(
        client_ids=args.client_ids,
        workshop_ids=args.
        workshop_ids,
        n_interactions=n_interactions,
        n_simulations=n_simulations,
        api_key=api_key
    )
    
    # Save to processed data
    output_filename = "simulated_ai_ai_baseline.csv"
    save_csv(df_simulated, output_filename, stage="processed")
    
    print(f"\n✓ Simulated {len(df_simulated)} interaction rows")
    print(f"✓ Saved to data/processed/{output_filename}")
    print("\n✅ Script 05 complete!")


if __name__ == "__main__":
    main()
