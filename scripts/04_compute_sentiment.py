#!/usr/bin/env python
"""
Script 04: Compute sentiment scores via semantic projection.

This script computes two parallel projection-based sentiment signals on the
interaction-level story data:

1. Legacy / sanity-check baseline (kept unchanged):
   `compute_semantic_projection_batch` with
   `paraphrase-multilingual-mpnet-base-v2` and the pre-shipped Sentiment.csv
   concept vector. Each turn embedded independently.
   -> author_1_sentiment_projection, author_2_sentiment_projection

2. Context-aware projection (new):
   Encoder configured under shared.sentiment.projection_model_name +
   a Fiction4Sentiment concept vector rebuilt with that encoder
   (see scripts/04b_build_concept_vector.py).
   For each conversation we embed the cumulative-prefix sequence
   (empty, +author_1_t1, +author_2_t1, +author_1_t2, ...) and project all
   prefixes onto the unit concept direction. From those we derive:
     * raw         : projection of each turn embedded alone
     * cumulative  : running story-sentiment after that turn was added
     * marginal    : that turn's contribution (first difference of cumulative)

   By linearity of dot products, marginal == proj(E(ctx+turn) - E(ctx)),
   so it directly captures the turn's shift along the sentiment axis given
   the prior context.
   -> author_{1,2}_sentiment_{raw,cumulative,marginal}

Usage:
    python scripts/04_compute_sentiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import (
    compute_semantic_projection_batch,
    compute_dyadic_contextual_projection,
)
from nes.io import (
    backfill_interaction_metadata,
    get_active_experiment,
    get_experiment_config,
    get_shared_config,
    get_project_root,
    load_parquet,
    save_parquet,
)


def main():
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    sentiment_config = shared_config['sentiment']
    simulated = shared_config['cleaning'].get('simulated', False)

    print(f"Active experiment: {experiment}")

    # Load story embeddings data
    print("Loading story data with embeddings...")
    df_interaction_level = load_parquet(
        "story_embeddings_interaction_level_simulated.parquet" if simulated
        else "story_embeddings_interaction_level.parquet",
        stage="processed",
    )
    df_interaction_level = backfill_interaction_metadata(
        df_interaction_level, simulated=simulated
    )
    print(f"Loaded {len(df_interaction_level)} interaction rows")
    df_dyadic = df_interaction_level.copy()

    # ------------------------------------------------------------------
    # 1. Legacy projection (mpnet + shipped Sentiment.csv) -- sanity check.
    # ------------------------------------------------------------------
    print(
        f"\n[Legacy] Computing per-turn projection with "
        f"paraphrase-multilingual-mpnet-base-v2 (batch_size={sentiment_config['batch_size']})..."
    )
    print("[Legacy] Projecting author_1 turns...")
    df_dyadic['author_1_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_1'].tolist(),
        batch_size=sentiment_config['batch_size'],
    )
    print("[Legacy] Projecting author_2 turns...")
    df_dyadic['author_2_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_2'].tolist(),
        batch_size=sentiment_config['batch_size'],
    )

    # ------------------------------------------------------------------
    # 2. Context-aware projection (new encoder + rebuilt concept vector).
    # ------------------------------------------------------------------
    projection_model = sentiment_config.get("projection_model_name")
    projection_vector_rel = sentiment_config.get("projection_vector_path")

    if projection_model and projection_vector_rel:
        project_root = get_project_root()
        projection_vector_path = project_root / projection_vector_rel
        if not projection_vector_path.exists():
            raise FileNotFoundError(
                f"Concept vector not found at {projection_vector_path}. "
                "Run scripts/04b_build_concept_vector.py first."
            )

        projection_batch_size = int(sentiment_config.get("projection_batch_size", 4))
        projection_task = sentiment_config.get("projection_task")
        separator = sentiment_config.get("projection_context_separator", "\n")

        print(
            f"\n[Contextual] Computing raw/cumulative/marginal projection with "
            f"{projection_model} (batch_size={projection_batch_size})..."
        )
        df_dyadic = compute_dyadic_contextual_projection(
            df_dyadic,
            model_name=projection_model,
            concept_vector_path=str(projection_vector_path),
            batch_size=projection_batch_size,
            task=projection_task,
            separator=separator,
        )
    else:
        print(
            "\n[Contextual] Skipped: set shared.sentiment.projection_model_name "
            "and shared.sentiment.projection_vector_path in config.yaml to enable."
        )

    # Save results
    output_file = (
        "dyadic_sentiment_scores_simulated.parquet" if simulated
        else "dyadic_sentiment_scores.parquet"
    )
    save_parquet(df_dyadic, output_file, stage="processed")
    print(f"\n✓ Saved sentiment for {len(df_dyadic)} turns to {output_file}")

    print("\n✅ Script 04 complete!")


if __name__ == "__main__":
    main()
