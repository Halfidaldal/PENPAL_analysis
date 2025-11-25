"""
Semantic exploration metrics using binned embedding distances.

Computes non-overlapping semantic jumps at different timescales to measure
how much the narrative explores semantic space over time.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_embedding(x):
    """
    Parse embedding from various formats (string, list, array, None).
    
    Parameters
    ----------
    x : various
        Embedding in any format
        
    Returns
    -------
    np.ndarray or None
        Parsed embedding as 1D array, or None if invalid
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    
    if isinstance(x, str):
        try:
            arr = np.fromstring(x.strip('[]'), sep=',')
        except Exception:
            return None
    else:
        try:
            arr = np.array(x)
        except Exception:
            return None
    
    return arr if (isinstance(arr, np.ndarray) and arr.ndim == 1) else None


def interleave_and_align(user_embs, ai_embs):
    """
    Interleave user and AI embeddings, then shift to align properly.
    
    Returns array starting from first AI embedding (after first user).
    
    Parameters
    ----------
    user_embs : np.ndarray
        User embeddings, shape (T, D)
    ai_embs : np.ndarray
        AI embeddings, shape (T, D)
        
    Returns
    -------
    np.ndarray
        Interleaved and aligned embeddings, shape (2*T - 1, D)
    """
    T, D = user_embs.shape
    E = np.empty((2 * T, D), dtype=float)
    E[0::2] = user_embs
    E[1::2] = ai_embs
    return E[1:]  # Drop first user, start from first AI


def compute_nonoverlap_distances(embeddings, window_length):
    """
    Compute cosine distances between centroids of consecutive non-overlapping windows.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Sequence of embeddings, shape (N, D)
    window_length : int
        Number of embeddings in each window
        
    Returns
    -------
    np.ndarray
        Cosine distances between adjacent window centroids
    """
    N, D = embeddings.shape
    n_bins = N // window_length
    
    if n_bins < 2:
        return np.array([])
    
    # Reshape into non-overlapping bins
    bins = embeddings[:n_bins * window_length].reshape(n_bins, window_length, D)
    
    # Compute centroids
    centroids = bins.mean(axis=1)
    
    # Normalize
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / norms
    
    # Cosine similarity between adjacent centroids
    similarities = np.einsum('ij,ij->i', centroids_norm[:-1], centroids_norm[1:])
    similarities = np.clip(similarities, -1.0, 1.0)
    
    # Convert to distance
    distances = 1.0 - similarities
    
    return distances


def compute_semantic_exploration_metrics(
    df,
    user_embedding_col="user_embedding",
    ai_embedding_col="ai_embedding",
    max_k=9
):
    """
    Compute semantic exploration metrics at multiple timescales.
    
    For each story, computes non-overlapping semantic jumps for window sizes
    from 2 to max_k+1 turns. Larger k = longer timescale.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embedding columns
    user_embedding_col : str
        Name of user embedding column
    ai_embedding_col : str
        Name of AI embedding column
    max_k : int
        Maximum bin-width parameter (window_length = k + 1)
        
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        - conversation_id
        - k (bin-width parameter)
        - bin_index (which consecutive window pair)
        - distance (cosine distance between centroids)
    """
    # Parse embeddings
    df = df.copy()
    df['user_emb'] = df[user_embedding_col].apply(parse_embedding)
    df['ai_emb'] = df[ai_embedding_col].apply(parse_embedding)
    
    # Filter out invalid embeddings
    before = len(df)
    df = df[df['user_emb'].notnull() & df['ai_emb'].notnull()].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with invalid embeddings.")
    
    # Group by story
    story_groups = df.groupby(['conversation_id'])
    
    records = []
    for conversation_id, grp in tqdm(story_groups, desc="Computing semantic exploration"):
        user_list = grp['user_emb'].tolist()
        ai_list = grp['ai_emb'].tolist()
        
        try:
            user_embs = np.vstack(user_list)
            ai_embs = np.vstack(ai_list)
        except Exception:
            continue
        
        if user_embs.shape != ai_embs.shape:
            continue
        
        # Interleave and align
        E = interleave_and_align(user_embs, ai_embs)
        
        # Compute distances at different window sizes
        for k in range(1, max_k + 1):
            window_length = k + 1
            distances = compute_nonoverlap_distances(E, window_length)
            
            for idx, dist in enumerate(distances):
                records.append({
                    'conversation_id': conversation_id,
                    'k': k,
                    'bin_index': idx,
                    'distance': float(dist)
                })
    
    return pd.DataFrame.from_records(records)


def compute_ai_ai_semantic_exploration(
    df,
    ai1_embedding_col="ai1_embedding",
    ai2_embedding_col="ai2_embedding",
    max_k=9
):
    """
    Compute semantic exploration metrics for AI-AI baseline.
    
    Same as compute_semantic_exploration_metrics but for AI1 and AI2 columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with AI-AI embedding columns
    ai1_embedding_col : str
        Name of AI1 embedding column
    ai2_embedding_col : str
        Name of AI2 embedding column
    max_k : int
        Maximum bin-width parameter
        
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with semantic exploration metrics
    """
    # Parse embeddings
    df = df.copy()
    df['user_emb'] = df[ai1_embedding_col].apply(parse_embedding)
    df['ai_emb'] = df[ai2_embedding_col].apply(parse_embedding)
    
    # Filter out invalid embeddings
    before = len(df)
    df = df[df['user_emb'].notnull() & df['ai_emb'].notnull()].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with invalid embeddings.")
    
    # Group by story
    story_groups = df.groupby(['conversation_id'])
    
    records = []
    for conversation_id, grp in tqdm(story_groups, desc="Computing AI-AI semantic exploration"):
        user_list = grp['user_emb'].tolist()
        ai_list = grp['ai_emb'].tolist()
        
        try:
            user_embs = np.vstack(user_list)
            ai_embs = np.vstack(ai_list)
        except Exception:
            continue
        
        if user_embs.shape != ai_embs.shape:
            continue
        
        # Interleave and align
        E = interleave_and_align(user_embs, ai_embs)
        
        # Compute distances at different window sizes
        for k in range(1, max_k + 1):
            window_length = k + 1
            distances = compute_nonoverlap_distances(E, window_length)
            
            for idx, dist in enumerate(distances):
                records.append({
                    'conversation_id': conversation_id,
                    'k': k,
                    'bin_index': idx,
                    'distance': float(dist)
                })
    
    return pd.DataFrame.from_records(records)
