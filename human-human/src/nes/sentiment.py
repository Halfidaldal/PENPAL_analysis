"""
Sentiment analysis functions.

This module provides sentiment scoring using:
- German BERT sentiment model (oliverguhr/german-sentiment-bert)
- Continuous valence scoring
- Batch processing for efficiency
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import os


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
    tokenizer, model, device = load_sentiment_model(model_name, device)
    
    all_scores = []
    print(f"Computing sentiment for {len(texts)} texts (batch_size={batch_size})...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment batches"):
        batch = texts[i:i+batch_size]
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
            all_scores.extend(valence.cpu().numpy())
    
    return np.array(all_scores)


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
    
    Splits stories into USER/AI turns and computes sentiment for each turn.
    
    Args:
        df: DataFrame with story text
        story_column: Column containing full story text with USER:/AI: markers
        valence_method: Method for valence scoring
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with columns: conversation_id, turn, type (user/ai), text, sentiment_score, pct_turn
    """
        
    
    # Compute relative turn position
    df["pct_turn"] = (
        df.groupby("conversation_id")["turn"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
    )
    
    # Compute sentiment for all turns
    print(f"\nComputing sentiment for {len(df)} turns...")
    user_scores = compute_sentiment_batch(
        df['user'].astype(str).tolist(),
        batch_size=batch_size,
        valence_method=valence_method,
        model_name=model_name
    )
    ai_scores = compute_sentiment_batch(
        df['ai'].astype(str).tolist(),
        batch_size=batch_size,
        valence_method=valence_method,
        model_name=model_name
    )
    df['user_sentiment_score'] = user_scores
    df['ai_sentiment_score'] = ai_scores
    df = df.sort_values(['conversation_id', 'turn']).reset_index(drop=True)
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
    projector = SemanticProjectionSentiment(model_name, vector_path, device)
    return projector.compute_score(texts, batch_size)
