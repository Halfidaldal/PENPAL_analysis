"""
Sentiment analysis functions.

This module provides sentiment scoring using:
- German BERT sentiment model (oliverguhr/german-sentiment-bert)
- Continuous valence scoring
- Batch processing for efficiency
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import os


_MISSING_TEXT_VALUES = {"", "nan", "none", "null"}


def _normalize_metric_text(value) -> Optional[str]:
    """Normalize arbitrary cell values; return None for missing/non-substantive text."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        return None
    return text


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_sentiment_model(
    model_name: str = "oliverguhr/german-sentiment-bert",
    device: Optional[torch.device] = None
):
    """
    Load a sentiment classification model.
    
    Args:
        model_name: HuggingFace model identifier
        device: Torch device (auto-detected if None)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading sentiment model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    
    return tokenizer, model, device


def continuous_valence_score(
    probs: torch.Tensor,
    method: str = "simple"
) -> torch.Tensor:
    """
    Convert sentiment probabilities to continuous valence scores.
    
    Assumes 3-class model: [positive, negative, neutral]
    
    Args:
        probs: Probability tensor of shape (batch_size, 3)
        method: Scoring method:
            - "simple": P(pos) - P(neg)
            - "amplify": (P(pos) - P(neg)) / (1 - P(neutral) + eps)
            - "dampen": (P(pos) - P(neg)) * (1 - P(neutral))
            
    Returns:
        Valence scores of shape (batch_size,)
    """
    if method == "simple":
        # Ignore neutral, just positive minus negative
        valence = probs[:, 2] - probs[:, 0]
    elif method == "amplify":
        # Amplify when neutral is high
        valence = (probs[:, 2] - probs[:, 0]) / (1 - probs[:, 1] + 1e-6)
    elif method == "dampen":
        # Dampen when neutral is high
        valence = (probs[:, 2] - probs[:, 0]) * (1 - probs[:, 1])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return valence


def compute_sentiment_batch(
    texts: List[str],
    model_name: str = "oliverguhr/german-sentiment-bert",
    batch_size: int = 64,
    valence_method: str = "simple",
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute continuous sentiment scores for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model identifier
        batch_size: Number of texts to process at once
        valence_method: Method for computing continuous valence
        device: Torch device (auto-detected if None)
        
    Returns:
        NumPy array of sentiment scores (float, range approx -1 to +1)
    """
    normalized_texts = [_normalize_metric_text(text) for text in texts]
    valid_positions = [idx for idx, text in enumerate(normalized_texts) if text is not None]
    scores_out = np.full(len(texts), np.nan, dtype=float)

    if not valid_positions:
        return scores_out

    tokenizer, model, device = load_sentiment_model(model_name, device)

    valid_texts = [normalized_texts[idx] for idx in valid_positions]
    valid_scores = []
    print(f"Computing sentiment for {len(valid_texts)} substantive texts (batch_size={batch_size})...")

    for i in tqdm(range(0, len(valid_texts), batch_size), desc="Sentiment batches"):
        batch = valid_texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            valence = continuous_valence_score(probs, method=valence_method)
            valid_scores.extend(valence.cpu().numpy())

    scores_out[valid_positions] = np.array(valid_scores, dtype=float)
    return scores_out


def add_sentiment_to_dataframe(
    df: pd.DataFrame,
    text_columns: List[str],
    model_name: str = "oliverguhr/german-sentiment-bert",
    batch_size: int = 64,
    valence_method: str = "simple"
) -> pd.DataFrame:
    """
    Add sentiment score columns to a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names to score
        model_name: Sentiment model to use
        batch_size: Batch size for processing
        valence_method: Method for computing valence
        
    Returns:
        DataFrame with new sentiment_score columns
    """
    df_out = df.copy()
    
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue
        
        print(f"\n=== Computing sentiment for column: {col} ===")
        texts = df[col].astype(str).tolist()
        
        scores = compute_sentiment_batch(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            valence_method=valence_method
        )
        
        df_out[f"{col}_sentiment"] = scores
    
    return df_out


def compute_dyadic_sentiment(
    df: pd.DataFrame,
    valence_method: str = "simple",
    batch_size: int = 64,
    model_name: str = "oliverguhr/german-sentiment-bert"
) -> pd.DataFrame:
    """
    Compute turn-by-turn sentiment for dyadic conversations.
    
    Uses standardized column names: author_1, author_2.
    
    Args:
        df: DataFrame with author_1 and author_2 text columns
        valence_method: Method for valence scoring
        batch_size: Batch size for processing
        model_name: Sentiment model to use
        
    Returns:
        DataFrame with added sentiment columns (author_1_sentiment_score, author_2_sentiment_score)
    """
        
    
    turn_axis = "analysis_turn" if "analysis_turn" in df.columns and df["analysis_turn"].notna().any() else "turn"

    def _scale_turn_positions(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        out = pd.Series(np.nan, index=series.index, dtype=float)
        if valid.empty:
            return out
        if valid.max() > valid.min():
            out.loc[valid.index] = (valid - valid.min()) / (valid.max() - valid.min())
        else:
            out.loc[valid.index] = 0.0
        return out

    df["pct_turn"] = df.groupby("conversation_id")[turn_axis].transform(_scale_turn_positions)
    
    # Compute sentiment for all turns using standardized column names
    print(f"\nComputing sentiment for {len(df)} turns...")
    author_1_scores = compute_sentiment_batch(
        df['author_1'].astype(str).tolist(),
        batch_size=batch_size,
        valence_method=valence_method,
        model_name=model_name
    )
    author_2_scores = compute_sentiment_batch(
        df['author_2'].astype(str).tolist(),
        batch_size=batch_size,
        valence_method=valence_method,
        model_name=model_name
    )
    df['author_1_sentiment_score'] = author_1_scores
    df['author_2_sentiment_score'] = author_2_scores
    sort_columns = ['conversation_id']
    if 'turn' in df.columns:
        sort_columns.append('turn')
    elif turn_axis in df.columns:
        sort_columns.append(turn_axis)
    df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


class SemanticProjectionSentiment:
    """
    Sentiment analysis using Semantic Projection (embedding projection onto a concept vector).
    Reference: https://github.com/lauritswl/SemanticProjection
    """
    def __init__(
        self, 
        model_name: str = "paraphrase-multilingual-mpnet-base-v2", 
        vector_path: str = "src/nes/data/Sentiment.csv", 
        device: Optional[torch.device] = None
    ):
        self.model_name = model_name
        self.vector_path = vector_path
        self.device = device if device else get_device()
        
        print(f"Loading Semantic Projection model: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.vector = self._load_vector()

    def _load_vector(self) -> np.ndarray:
        if not os.path.exists(self.vector_path):
            # Try relative path if absolute fails
            rel_path = os.path.join(os.path.dirname(__file__), "data", "Sentiment.csv")
            if os.path.exists(rel_path):
                self.vector_path = rel_path
            else:
                # Try one level up if we are in src/nes
                rel_path_2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nes", "data", "Sentiment.csv")
                if os.path.exists(rel_path_2):
                    self.vector_path = rel_path_2
                else:
                    raise FileNotFoundError(f"Vector file not found at {self.vector_path}. Please download it first.")
        
        # The file has a header 0,1,2... so we read it normally
        df = pd.read_csv(self.vector_path)
        return df.values.flatten()

    def compute_score(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print(f"Encoding {len(texts)} texts for projection...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        # Project onto the sentiment vector
        v = self.vector
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            raise ValueError("Concept vector has zero norm.")
            
        unit_v = v / norm_v
        
        # Scalar projection: dot product with unit vector
        # This gives the component of the embedding along the sentiment direction
        scores = np.dot(embeddings, unit_v)
        return scores


def compute_semantic_projection_batch(
    texts: List[str],
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    vector_path: str = "src/nes/data/Sentiment.csv",
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute sentiment scores using Semantic Projection.

    Args:
        texts: List of text strings
        model_name: SentenceTransformer model name
        vector_path: Path to the concept vector CSV
        batch_size: Batch size for encoding
        device: Torch device

    Returns:
        NumPy array of sentiment scores
    """
    normalized_texts = [_normalize_metric_text(text) for text in texts]
    valid_positions = [idx for idx, text in enumerate(normalized_texts) if text is not None]
    scores_out = np.full(len(texts), np.nan, dtype=float)

    if not valid_positions:
        return scores_out

    projector = SemanticProjectionSentiment(model_name, vector_path, device)
    valid_texts = [normalized_texts[idx] for idx in valid_positions]
    valid_scores = projector.compute_score(valid_texts, batch_size)
    scores_out[valid_positions] = valid_scores
    return scores_out


# ---------------------------------------------------------------------------
# Context-aware projection pipeline.
#
# Builds a concept vector with the CHC procedure
# (https://github.com/centre-for-humanities-computing/embedding-projection):
#   v = mean(embed(positive_anchors)) - mean(embed(negative_anchors))
# using the same anchor texts as the published Fiction4Sentiment vector
# (positive label >= 7, negative label <= 3, 60/40 stratified split, random_state=42).
# Projection score for an embedding m is m·v / ||v||, which equals
# proj(E(ctx+turn)) - proj(E(ctx)) by linearity -- making the marginal
# contribution of a turn well-defined as a first difference of cumulative
# scores along a single fixed direction.
# ---------------------------------------------------------------------------


def _load_sentence_encoder(
    model_name: str,
    device: Optional[torch.device] = None,
    trust_remote_code: bool = True,
) -> SentenceTransformer:
    if device is None:
        device = get_device()
    print(f"Loading projection encoder: {model_name}")
    return SentenceTransformer(
        model_name,
        device=str(device),
        trust_remote_code=trust_remote_code,
    )


def _front_truncate_to_model_window(
    model: SentenceTransformer,
    texts: List[str],
) -> List[str]:
    """Keep the tail of each text so the most-recent turn always survives truncation.

    SentenceTransformer's default tokenizer truncates from the right
    (cuts the end); for cumulative-prefix encoding we want the opposite.
    """
    max_len = getattr(model, "max_seq_length", None)
    if not max_len:
        return texts
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return texts

    # Reserve a small budget for special tokens added by the model.
    budget = max(1, int(max_len) - 4)

    truncated: List[str] = []
    for text in texts:
        if not text:
            truncated.append(text)
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            truncated.append(text)
            continue
        tail_ids = ids[-budget:]
        truncated.append(tokenizer.decode(tail_ids, skip_special_tokens=True))
    return truncated


def _encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 8,
    task: Optional[str] = None,
    normalize_embeddings: bool = True,
    front_truncate: bool = True,
) -> np.ndarray:
    if front_truncate:
        texts = _front_truncate_to_model_window(model, texts)

    encode_kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    if task is not None and "task" in model.encode.__code__.co_varnames:
        encode_kwargs["task"] = task
    return model.encode(texts, **encode_kwargs)


def build_concept_vector_fiction4(
    model_name: str,
    output_vector_path: str,
    *,
    concept_text_path: str = "src/nes/data/Fiction4Sentiment_concept_text.csv",
    batch_size: int = 8,
    task: Optional[str] = None,
    normalize_embeddings: bool = True,
    device: Optional[torch.device] = None,
    overwrite: bool = False,
) -> np.ndarray:
    """Build a sentiment concept vector via the CHC procedure with a chosen encoder.

    Uses the published Fiction4Sentiment anchor texts (positive/negative labels)
    and computes v = mean(positive_embeddings) - mean(negative_embeddings).
    Saves v as a 1-row CSV (columns 0..d-1) matching the format already consumed
    by SemanticProjectionSentiment._load_vector.

    Returns the concept vector as a 1-D numpy array.
    """
    if os.path.exists(output_vector_path) and not overwrite:
        print(f"Concept vector already exists at {output_vector_path}; loading existing.")
        return pd.read_csv(output_vector_path).values.flatten()

    if not os.path.exists(concept_text_path):
        raise FileNotFoundError(
            f"Concept anchor file not found at {concept_text_path}. "
            "Expected the Fiction4Sentiment_concept_text.csv from "
            "centre-for-humanities-computing/embedding-projection."
        )

    anchors = pd.read_csv(concept_text_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(anchors.columns):
        raise ValueError(
            f"Anchor file must have columns {required_cols}, got {set(anchors.columns)}"
        )

    anchors["label"] = anchors["label"].astype(str).str.strip().str.lower()
    valid_labels = {"positive", "negative"}
    bad = set(anchors["label"].unique()) - valid_labels
    if bad:
        raise ValueError(f"Anchor file contains unexpected labels: {bad}")

    pos_texts = anchors.loc[anchors["label"] == "positive", "text"].astype(str).tolist()
    neg_texts = anchors.loc[anchors["label"] == "negative", "text"].astype(str).tolist()
    print(f"Anchors: {len(pos_texts)} positive, {len(neg_texts)} negative")

    model = _load_sentence_encoder(model_name, device=device)

    pos_emb = _encode_texts(
        model, pos_texts,
        batch_size=batch_size, task=task,
        normalize_embeddings=normalize_embeddings, front_truncate=True,
    )
    neg_emb = _encode_texts(
        model, neg_texts,
        batch_size=batch_size, task=task,
        normalize_embeddings=normalize_embeddings, front_truncate=True,
    )

    concept_vector = pos_emb.mean(axis=0) - neg_emb.mean(axis=0)
    norm = float(np.linalg.norm(concept_vector))
    if norm == 0.0:
        raise ValueError("Computed concept vector has zero norm; check anchor embeddings.")
    print(f"Concept vector dim={concept_vector.shape[0]}, ||v||={norm:.4f}")

    os.makedirs(os.path.dirname(output_vector_path) or ".", exist_ok=True)
    pd.DataFrame(concept_vector.reshape(1, -1)).to_csv(output_vector_path, index=False)
    print(f"Saved concept vector to {output_vector_path}")
    return concept_vector


def _load_concept_unit_vector(vector_path: str) -> np.ndarray:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(
            f"Concept vector not found at {vector_path}. "
            "Run scripts/04b_build_concept_vector.py first."
        )
    v = pd.read_csv(vector_path).values.flatten().astype(np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError(f"Concept vector at {vector_path} has zero norm.")
    return v / n


def _build_conversation_prefixes(
    a1_turns: List[str],
    a2_turns: List[str],
    separator: str = "\n",
) -> List[str]:
    """Return [empty, +a1_t1, +a2_t1, +a1_t2, +a2_t2, ...] as cumulative strings.

    Length is 2*N + 1 for N turn-pairs. The empty string anchors the trajectory
    so the first turn's marginal is well-defined (cumulative_a1_t1 - baseline).
    """
    prefixes: List[str] = [""]
    running_parts: List[str] = []
    for a1, a2 in zip(a1_turns, a2_turns):
        a1_text = "" if a1 is None else str(a1)
        a2_text = "" if a2 is None else str(a2)
        if a1_text:
            running_parts.append(a1_text)
        prefixes.append(separator.join(running_parts))
        if a2_text:
            running_parts.append(a2_text)
        prefixes.append(separator.join(running_parts))
    return prefixes


def compute_dyadic_contextual_projection(
    df: pd.DataFrame,
    model_name: str,
    concept_vector_path: str,
    *,
    conversation_col: str = "conversation_id",
    turn_col: Optional[str] = None,
    author_1_col: str = "author_1",
    author_2_col: str = "author_2",
    batch_size: int = 8,
    task: Optional[str] = None,
    normalize_embeddings: bool = True,
    separator: str = "\n",
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Compute raw, cumulative, and marginal sentiment projections per turn.

    For each conversation, builds the sequence of cumulative-prefix strings
    (empty, +a1_t1, +a2_t1, +a1_t2, ...), embeds them with `model_name`,
    projects onto the unit concept direction loaded from `concept_vector_path`,
    and emits six new columns:

      author_1_sentiment_raw          : projection of author_1 text alone
      author_2_sentiment_raw          : projection of author_2 text alone
      author_1_sentiment_cumulative   : projection of full context up to and incl. author_1's turn
      author_2_sentiment_cumulative   : projection of full context up to and incl. author_2's turn
      author_1_sentiment_marginal     : cumulative_a1 - cumulative_just_before_a1
      author_2_sentiment_marginal     : cumulative_a2 - cumulative_a1 (same turn)

    Note: by linearity of dot products, the marginal column equals the
    projection of (E(ctx+turn) - E(ctx)) onto the unit concept direction.
    """
    if conversation_col not in df.columns:
        raise ValueError(f"Missing required column '{conversation_col}' in dataframe.")
    if author_1_col not in df.columns or author_2_col not in df.columns:
        raise ValueError(
            f"Missing required columns '{author_1_col}'/'{author_2_col}' in dataframe."
        )

    if turn_col is None:
        if "analysis_turn" in df.columns and df["analysis_turn"].notna().any():
            turn_col = "analysis_turn"
        elif "turn" in df.columns:
            turn_col = "turn"
        else:
            raise ValueError("Could not infer turn ordering column ('analysis_turn'/'turn').")

    unit_v = _load_concept_unit_vector(concept_vector_path)

    df_sorted = df.copy()
    df_sorted["_orig_index"] = np.arange(len(df_sorted))
    df_sorted[turn_col] = pd.to_numeric(df_sorted[turn_col], errors="coerce")
    df_sorted = df_sorted.sort_values([conversation_col, turn_col], kind="mergesort")

    # Build per-conversation prefix lists and remember slice spans.
    all_prefixes: List[str] = []
    all_raw_a1: List[str] = []
    all_raw_a2: List[str] = []
    conv_spans: List[Tuple[str, int, int, List[int]]] = []  # (conv_id, prefix_start, prefix_end, row_indices)

    for conv_id, group in df_sorted.groupby(conversation_col, sort=False):
        a1_turns = [_sanitize_for_prefix(v) for v in group[author_1_col].tolist()]
        a2_turns = [_sanitize_for_prefix(v) for v in group[author_2_col].tolist()]
        prefixes = _build_conversation_prefixes(a1_turns, a2_turns, separator=separator)
        start = len(all_prefixes)
        all_prefixes.extend(prefixes)
        end = len(all_prefixes)
        conv_spans.append((str(conv_id), start, end, group["_orig_index"].tolist()))
        all_raw_a1.extend(a1_turns)
        all_raw_a2.extend(a2_turns)

    model = _load_sentence_encoder(model_name, device=device)

    print(f"Encoding {len(all_prefixes)} conversation prefixes...")
    prefix_emb = _encode_texts(
        model, all_prefixes,
        batch_size=batch_size, task=task,
        normalize_embeddings=normalize_embeddings, front_truncate=True,
    )
    prefix_scores = prefix_emb @ unit_v  # shape (n_prefixes,)

    print(f"Encoding {len(all_raw_a1)} author_1 turns alone (raw)...")
    raw_a1_emb = _encode_texts(
        model, all_raw_a1,
        batch_size=batch_size, task=task,
        normalize_embeddings=normalize_embeddings, front_truncate=True,
    )
    raw_a1_scores = raw_a1_emb @ unit_v

    print(f"Encoding {len(all_raw_a2)} author_2 turns alone (raw)...")
    raw_a2_emb = _encode_texts(
        model, all_raw_a2,
        batch_size=batch_size, task=task,
        normalize_embeddings=normalize_embeddings, front_truncate=True,
    )
    raw_a2_scores = raw_a2_emb @ unit_v

    # Allocate output columns aligned with original df order.
    n_rows = len(df)
    a1_cum = np.full(n_rows, np.nan, dtype=float)
    a2_cum = np.full(n_rows, np.nan, dtype=float)
    a1_marg = np.full(n_rows, np.nan, dtype=float)
    a2_marg = np.full(n_rows, np.nan, dtype=float)
    a1_raw = np.full(n_rows, np.nan, dtype=float)
    a2_raw = np.full(n_rows, np.nan, dtype=float)

    raw_cursor = 0
    for _conv_id, start, end, row_indices in conv_spans:
        scores = prefix_scores[start:end]
        # scores[0] is baseline (empty prefix). For pair k (0-indexed):
        #   cumulative_a1 = scores[2k + 1]; cumulative_a2 = scores[2k + 2]
        for k, orig_idx in enumerate(row_indices):
            before = scores[2 * k]
            after_a1 = scores[2 * k + 1]
            after_a2 = scores[2 * k + 2]
            a1_cum[orig_idx] = after_a1
            a2_cum[orig_idx] = after_a2
            a1_marg[orig_idx] = after_a1 - before
            a2_marg[orig_idx] = after_a2 - after_a1
            a1_raw[orig_idx] = raw_a1_scores[raw_cursor]
            a2_raw[orig_idx] = raw_a2_scores[raw_cursor]
            raw_cursor += 1

    df_out = df.copy()
    df_out["author_1_sentiment_raw"] = a1_raw
    df_out["author_2_sentiment_raw"] = a2_raw
    df_out["author_1_sentiment_cumulative"] = a1_cum
    df_out["author_2_sentiment_cumulative"] = a2_cum
    df_out["author_1_sentiment_marginal"] = a1_marg
    df_out["author_2_sentiment_marginal"] = a2_marg
    return df_out


def _sanitize_for_prefix(value) -> str:
    text = _normalize_metric_text(value)
    return "" if text is None else text
