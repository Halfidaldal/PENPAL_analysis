"""
Embedding computation functions.

This module handles text embedding using various models:
- Jina embeddings (jinaai/jina-embeddings-v3)
- E5 embeddings (intfloat/multilingual-e5-large-instruct)
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def compute_embeddings_batch(
    texts: List[str],
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 32,
    task: Optional[str] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a SentenceTransformer model.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the SentenceTransformer model
        batch_size: Number of texts to process at once
        task: Optional task hint for the model (e.g., "text-matching")
        device: Torch device to use (auto-detected if None)
        
    Returns:
        NumPy array of shape (len(texts), embedding_dim)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=str(device))
    
    print(f"Computing embeddings for {len(texts)} texts (batch_size={batch_size})...")
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        
        # Encode with optional task parameter
        if task and hasattr(model, 'encode') and 'task' in model.encode.__code__.co_varnames:
            embeddings = model.encode(batch, task=task, show_progress_bar=False)
        else:
            embeddings = model.encode(batch, show_progress_bar=False)
        
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    return all_embeddings


def embed_story_columns(
    df: pd.DataFrame,
    text_columns: List[str],
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 32,
    task: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute embeddings for multiple text columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names to embed
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for embedding
        task: Optional task parameter for the model
        
    Returns:
        Tuple of:
        - DataFrame with new embedding columns (as lists of floats)
        - Dictionary mapping column names to numpy arrays of embeddings
    """
    df_out = df.copy()
    embeddings_dict = {}
    
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue
        
        print(f"\n=== Embedding column: {col} ===")
        texts = df[col].astype(str).tolist()
        
        embeddings = compute_embeddings_batch(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            task=task
        )
        
        # Store as separate arrays
        embeddings_dict[col] = embeddings
        
        # Also store in DataFrame as list column (for parquet compatibility)
        df_out[f"{col}_embedding"] = embeddings.tolist()
    
    return df_out, embeddings_dict


def compute_story_embeddings_full_stories(
    df: pd.DataFrame,
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 32
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute embeddings for full_story, full_user, and full_ai columns.
    
    This is the "field" embedding approach where we embed the complete texts.
    
    Args:
        df: DataFrame with 'full_story', 'full_user', 'full_ai' columns
        model_name: Model to use for embeddings
        batch_size: Batch size for processing
        
    Returns:
        Tuple of:
        - DataFrame with embedding columns added
        - story_embeddings (np.ndarray)
        - user_embeddings (np.ndarray)
        - ai_embeddings (np.ndarray)
    """
    columns_to_embed = ['full_story', 'full_user', 'full_ai']
    
    df_embedded, embeddings_dict = embed_story_columns(
        df,
        text_columns=columns_to_embed,
        model_name=model_name,
        batch_size=batch_size,
        task="text-matching"
    )
    
    story_embeddings = embeddings_dict.get('full_story')
    user_embeddings = embeddings_dict.get('full_user')
    ai_embeddings = embeddings_dict.get('full_ai')
    
    return df_embedded, story_embeddings, user_embeddings, ai_embeddings
