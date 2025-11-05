#!/usr/bin/env python
"""
Embed 'user' and 'ai' columns from finished_stories_raw.csv using
jinaai/jina-embeddings-v3 and save the result with two new columns:

    user_embedded_jina
    ai_embedded_jina

The script will:
  1) pip install/upgrade all required packages,
  2) load the CSV,
  3) run batched embeddings with progress bars,
  4) save a Parquet file with vector columns.

Usage:
    python embed_finished_stories_jina.py
"""

import sys
import subprocess

# -------------------------------------------------------------------
# Step 1: Install dependencies via pip
# -------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "pandas",
    "tqdm",
    "torch",
    "sentence-transformers>=2.6.0",
    "transformers",
    "pyarrow",
]


def install_dependencies():
    """Install or upgrade required packages."""
    print("Installing/upgrading required packages (may be quick if already installed)…")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + REQUIRED_PACKAGES
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Dependency installation complete.\n")


# -------------------------------------------------------------------
# Step 2: Main embedding logic
# -------------------------------------------------------------------

INPUT_CSV = "../Data/finished_stories_corrected.csv"
OUTPUT_PARQUET = "finished_stories_with_jina_embeddings.parquet"

# You can push this higher on the H100 if VRAM allows
BATCH_SIZE = 256

MODEL_NAME = "jinaai/jina-embeddings-v3"
TASK = "text-matching"  # symmetrical semantic similarity (user ↔ ai)


def main():
    # Make sure deps are there before importing them
    install_dependencies()

    import pandas as pd
    from tqdm.auto import tqdm
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"Loading data from {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    n_rows = len(df)
    print(f"Loaded {n_rows} rows.")

    if "user_corrected" not in df.columns or "ai" not in df.columns:
        raise ValueError("CSV must contain 'user_corrected' and 'ai' columns.")

    # Clean up text columns
    df["user_corrected"] = df["user_corrected"].fillna("").astype(str)
    df["ai"] = df["ai"].fillna("").astype(str)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model {MODEL_NAME} … (first run downloads weights)")
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=device,
        # H100: bfloat16 is nice; comment this out if it causes issues
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
    )
    print("Model loaded.")

    # Encoding options; SentenceTransformers progress bar turned on
    encode_kwargs = dict(
        task=TASK,
        prompt_name=TASK,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    user_embeddings = []
    ai_embeddings = []

    print(f"Embedding {n_rows} rows with batch size {BATCH_SIZE} …")
    for start in tqdm(range(0, n_rows, BATCH_SIZE), desc="Batches processed"):
        end = min(start + BATCH_SIZE, n_rows)

        user_batch = df["user_corrected"].iloc[start:end].tolist()
        ai_batch = df["ai"].iloc[start:end].tolist()

        # Separate calls keeps the symmetric text-matching behaviour clear
        user_emb_batch = model.encode(user_batch, **encode_kwargs)
        ai_emb_batch = model.encode(ai_batch, **encode_kwargs)

        user_embeddings.extend(user_emb_batch)
        ai_embeddings.extend(ai_emb_batch)

    assert len(user_embeddings) == n_rows
    assert len(ai_embeddings) == n_rows

    print("Attaching embedding columns …")
    df["user_embedded_jina"] = user_embeddings
    df["ai_embedded_jina"] = ai_embeddings

    print(f"Saving to {OUTPUT_PARQUET} …")
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print("All done, Kong Gulerod ✔")


if __name__ == "__main__":
    main()
