#!/usr/bin/env python
"""
Embed 'user' and 'ai' columns from finished_stories_raw.csv using
jinaai/jina-embeddings-v3 and save the result with two new columns:

    user_embedded_jina
    ai_embedded_jina

Usage:
    python embed_finished_stories_jina.py
"""

import os
import math
import pandas as pd
from tqdm.auto import tqdm

import torch
from sentence_transformers import SentenceTransformer


# --------- Config ---------
INPUT_CSV = "finished_stories_raw.csv"
OUTPUT_PARQUET = "finished_stories_with_jina_embeddings.parquet"
BATCH_SIZE = 256  # tweak up/down depending on VRAM

MODEL_NAME = "jinaai/jina-embeddings-v3"
TASK = "text-matching"  # good default for symmetric similarity (user ↔ ai)

# If you’re non-commercial you’re fine. For commercial use, Jina v3 is CC-BY-NC-4.0.


def main():
    print("Loading data…")
    df = pd.read_csv(INPUT_CSV)

    if "user" not in df.columns or "ai" not in df.columns:
        raise ValueError("CSV must contain 'user' and 'ai' columns.")

    # Ensure text (avoid NaNs)
    df["user"] = df["user"].fillna("").astype(str)
    df["ai"] = df["ai"].fillna("").astype(str)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Optional: use bf16 on H100 (comment out if you hit issues)
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
    )

    # Jina recommends passing both `task` and `prompt_name=task`
    encode_kwargs = dict(
        task=TASK,
        prompt_name=TASK,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,  # we handle progress ourselves
    )

    n_rows = len(df)
    user_embeddings = []
    ai_embeddings = []

    print(f"Embedding {n_rows} rows with batch size {BATCH_SIZE}…")
    for start in tqdm(range(0, n_rows, BATCH_SIZE), desc="Embedding rows"):
        end = min(start + BATCH_SIZE, n_rows)

        user_batch = df["user"].iloc[start:end].tolist()
        ai_batch = df["ai"].iloc[start:end].tolist()

        # Two passes so you keep semantics of 'text-matching' symmetric task
        user_emb_batch = model.encode(user_batch, **encode_kwargs)
        ai_emb_batch = model.encode(ai_batch, **encode_kwargs)

        user_embeddings.extend(user_emb_batch)
        ai_embeddings.extend(ai_emb_batch)

    # Sanity check
    assert len(user_embeddings) == n_rows
    assert len(ai_embeddings) == n_rows

    # Store as object columns (each cell = list/np.ndarray of floats)
    df["user_embedded_jina"] = user_embeddings
    df["ai_embedded_jina"] = ai_embeddings

    print(f"Saving to {OUTPUT_PARQUET} …")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
