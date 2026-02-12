#!/usr/bin/env python3
"""
Script 05: Compute novelty and transience scores.

Uses a causal language model (Mistral-7B) to compute:
- Novelty (surprise): how surprising this utterance is given prior context
- Transience: how surprising future text is given this utterance

Input:  data/interim/clean_stories.csv
Output: data/processed/novelty_scores.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.io import load_config, load_csv, save_csv
from nes.novelty import load_language_model, compute_novelty_scores, compute_transience_scores
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute novelty and transience scores")
    parser.add_argument("--input", default="interaction_level_stories_filtered.csv", help="Input CSV path")
    parser.add_argument("--output", default="novelty_scores.csv", help="Output CSV path")
    args = parser.parse_args()
    
    config = load_config()
    model_name = config['novelty'].get('model_name', 'mistral-7b')
    window_size = config['novelty'].get('window_size', 128)

    
    print(f"Loading clean data from {args.input}")
    df = load_csv(Path(args.input).name, stage="interim")
    
    print("Loading language model...")
    tokenizer, model, device = load_language_model(model_name)
    
    print("Computing novelty scores...")
    df = compute_novelty_scores(df, tokenizer, model)
    
    print("Computing transience scores...")
    df = compute_transience_scores(df, tokenizer, model, window_size=window_size)
    
    # Drop token ID columns (not needed in output)
    df = df.drop(columns=["user_ids", "ai_ids"], errors="ignore")
    
    save_csv(df, Path(args.output).name)
    print(f"Saved novelty scores to {args.output}")


if __name__ == "__main__":
    main()
