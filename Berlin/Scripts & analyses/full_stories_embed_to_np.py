#!/usr/bin/env python
"""
full_stories_embed_to_np.py

Pipeline:

1) Load per-turn CSV (one row per interaction with columns like
   timestamp, user, ai, client_id, workshop_id, language, conversation_id).
2) Group by conversation_id and build "full_story" by concatenating
   all USER/AI turns in time order.
3) Keep meta columns (client_id, workshop_id, language) from first row.
4) Embed each full_story with jinaai/jina-embeddings-v3.
5) Save:
    - NumPy array: data/story_embeddings_jina_field.npy
    - Parquet with text + embeddings: data/story_text_field_w_embeddings_jina.parquet
"""

import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

# Input: the CSV you created in download_finished_stories.py
# Adjust this if you are using a cleaned/corrected CSV instead.
INPUT_CSV = os.path.join(
    os.path.expanduser("~"),
    "Documents",
    "PENPAL_analysis",
    "Berlin",
    "Data",
    "finished_stories_corrected.csv",
)

OUTPUT_DIR = "data"
OUTPUT_NPY = os.path.join(OUTPUT_DIR, "story_embeddings_jina_field.npy")
OUTPUT_USER_NPY = os.path.join(OUTPUT_DIR, "story_user_embeddings_jina_field.npy")
OUTPUT_AI_NPY = os.path.join(OUTPUT_DIR, "story_ai_embeddings_jina_field.npy")
OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR, "story_text_field_w_embeddings_jina.parquet"
)

MODEL_NAME = "jinaai/jina-embeddings-v3"
TASK = "text-matching"  # same as your other Jina script
MIN_TURNS_PER_STORY = 4  # only keep stories with >= 4 interactions


# -----------------------------
# BUILD FULL STORIES
# -----------------------------
def build_full_story_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-interaction rows into one row per conversation_id.

    For each conversation_id:
      - sort by timestamp (if present)
      - concatenate USER / AI text into a single 'full_story' string
      - keep client_id, workshop_id, language from first row
    """

    # Make sure conversation_id exists
    if "conversation_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'conversation_id' column.")

    # Sort within each conversation by timestamp if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["conversation_id", "timestamp"])
    else:
        df = df.sort_values(["conversation_id"])

    groups = df.groupby("conversation_id", sort=False)

    records = []
    for conv_id, g in groups:
        # Optionally filter small / partial stories
        if len(g) < MIN_TURNS_PER_STORY:
            continue

        client_id = g["client_id"].iloc[0] if "client_id" in g.columns else None
        workshop_id = g["workshop_id"].iloc[0] if "workshop_id" in g.columns else None
        language = g["language"].iloc[0] if "language" in g.columns else None

        turn_texts = []
        ai_turns = []
        user_turns = []
        for _, row in g.iterrows():
            user_text = (row.get("user_corrected") or "").strip()
            ai_text = (row.get("ai") or "").strip()

            parts = []
            if user_text:
                parts.append(f"USER: {user_text}")
                user_turns.append(user_text)
            if ai_text:
                parts.append(f"AI: {ai_text}")
                ai_turns.append(ai_text)

            if parts:
                turn_texts.append("\n".join(parts))

        full_story = "\n\n".join(turn_texts)
        full_user = "\n".join(user_turns)
        full_ai = "\n".join(ai_turns)

        # Skip if somehow we ended up with empty text
        if not full_story.strip():
            continue

        records.append(
            {
                "conversation_id": conv_id,
                "client_id": client_id,
                "workshop_id": workshop_id,
                "language": language,
                "full_story": full_story,
                "full_user": full_user,
                "full_ai": full_ai,
            }
        )

    full_df = pd.DataFrame.from_records(records)
    return full_df


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading per-turn data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows in raw dataframe: {len(df)}")
    df["user_corrected"] = df["user_corrected"].fillna("").astype(str)
    df["ai"] = df["ai"].fillna("").astype(str)

    full_df = build_full_story_df(df)
    print(
        f"Stories after grouping (one per conversation_id, min turns {MIN_TURNS_PER_STORY}): "
        f"{len(full_df)}"
    )

    corpus = full_df["full_story"].tolist()
    corpus_user = full_df["full_user"].tolist()
    corpus_ai = full_df["full_ai"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
        # Use dtype instead of deprecated torch_dtype
        model_kwargs={"dtype": torch.bfloat16} if torch.cuda.is_available() else {},
    )

    encode_kwargs = dict(
        task=TASK,
        prompt_name=TASK,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Encoding full stories with Jina embeddings …")
    embeddings = model.encode(corpus, **encode_kwargs)
    embeddings = np.asarray(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")  # (n_stories, dim)

    print("Encoding user turns with Jina embeddings …")
    user_embeddings = model.encode(corpus_user, **encode_kwargs)
    user_embeddings = np.asarray(user_embeddings)
    print(f"User embeddings shape: {user_embeddings.shape}")  # (n_stories, dim)

    print("Encoding AI turns with Jina embeddings …")
    ai_embeddings = model.encode(corpus_ai, **encode_kwargs)
    ai_embeddings = np.asarray(ai_embeddings)
    print(f"AI embeddings shape: {ai_embeddings.shape}")  # (n_stories, dim)


    print(f"Saving NumPy array to: {OUTPUT_NPY}")
    np.save(OUTPUT_NPY, embeddings)
    np.save(OUTPUT_USER_NPY, user_embeddings)
    np.save(OUTPUT_AI_NPY, ai_embeddings)

    # Optional but very handy: store text + meta + embedding in a parquet
    full_df = full_df.copy()
    full_df["embedding_jina"] = list(embeddings)
    full_df["user_embedding_jina"] = list(user_embeddings)
    full_df["ai_embedding_jina"] = list(ai_embeddings)
    
    print(f"Saving story text + embeddings to: {OUTPUT_PARQUET}")
    full_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("\nDone, Bossman.")
    print(f'In your notebook you can now do:\n  embeddings_field = np.load("{OUTPUT_NPY}")')
    print(f'  user_embeddings_field = np.load("{OUTPUT_USER_NPY}")')
    print(f'  ai_embeddings_field = np.load("{OUTPUT_AI_NPY}")')


if __name__ == "__main__":
    main()