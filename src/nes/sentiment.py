"""
Sentiment scoring via semantic concept-vector projection.

The returned score is a cosine projection onto the sentiment concept vector.
It is bounded to [-1, 1]: lower values indicate more negative tone, higher
values indicate more positive tone.
"""

from typing import List, Optional
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


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


class SemanticProjectionSentiment:
    """
    Sentiment analysis using semantic projection onto a concept vector.

    Scores are cosine projections: both the text embedding and concept vector
    are L2-normalized before the dot product, so valid scores are in [-1, 1].
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        vector_path: str = "src/nes/data/Sentiment.csv",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.vector_path = vector_path
        self.device = device if device else get_device()

        print(f"Loading Semantic Projection model: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.vector = self._load_vector()

    def _load_vector(self) -> np.ndarray:
        if not os.path.exists(self.vector_path):
            rel_path = os.path.join(os.path.dirname(__file__), "data", "Sentiment.csv")
            if os.path.exists(rel_path):
                self.vector_path = rel_path
            else:
                raise FileNotFoundError(
                    f"Vector file not found at {self.vector_path}. "
                    "Expected src/nes/data/Sentiment.csv."
                )

        df = pd.read_csv(self.vector_path)
        return df.values.flatten().astype(float)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cannot normalize zero-length embedding or concept vector.")
        return matrix / norms

    def compute_score(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        print(f"Encoding {len(texts)} texts for sentiment projection...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        unit_embeddings = self._l2_normalize(embeddings)
        unit_vector = self._l2_normalize(self.vector.reshape(1, -1)).ravel()

        scores = np.dot(unit_embeddings, unit_vector)
        return np.clip(scores, -1.0, 1.0)


def compute_semantic_projection_batch(
    texts: List[str],
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    vector_path: str = "src/nes/data/Sentiment.csv",
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute sentiment scores using semantic concept-vector projection.

    Returns a NumPy array of cosine-projection scores in [-1, 1]. Missing or
    non-substantive input texts are returned as NaN.
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

