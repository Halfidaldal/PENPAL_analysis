"""
Novelty, transience, and resonance computation using language models.
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_language_model(model_path, device=None):
    """
    Load a causal language model and tokenizer.
    
    Parameters
    ----------
    model_path : str
        Path to local model or HuggingFace model name
    device : torch.device or None
        Device to run on
        
    Returns
    -------
    tuple
        (tokenizer, model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading language model from {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    
    return tokenizer, model, device


def calc_sentence_surprisal(context_ids, target_ids, model, window_size=128):
    """
    Compute average surprisal (bits/token) and total surprisal (bits)
    for target tokens given context.
    
    Parameters
    ----------
    context_ids : list of int
        Token IDs for context
    target_ids : list of int
        Token IDs for target sentence
    model : transformers.PreTrainedModel
        Causal language model
    window_size : int
        Maximum context window size
        
    Returns
    -------
    tuple
        (average_surprisal, total_surprisal)
    """
    # Trim context to last window_size tokens
    context_ids = context_ids[-window_size:]
    combined_ids = context_ids + target_ids
    
    if len(combined_ids) < 2 or not target_ids:
        return 0.0, 0.0
    
    # Forward pass
    input_tensor = torch.tensor([combined_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    logits = outputs.logits[:, :-1, :]  # Distributions for each next token
    
    total_nll = 0.0
    for idx, token_id in enumerate(target_ids):
        pos = len(context_ids) + idx - 1
        dist = logits[0, pos]
        log_probs = torch.nn.functional.log_softmax(dist, dim=-1)
        log2p = log_probs[token_id] / math.log(2)
        total_nll += -log2p.item()
    
    avg_nll = total_nll / len(target_ids)
    return avg_nll, total_nll


def compute_novelty_scores(df, tokenizer, model, window_size=128):
    """
    Compute novelty (surprise) scores for user and AI utterances.
    
    Novelty = how surprising this utterance is given prior context.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with user and AI text columns
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    model : transformers.PreTrainedModel
        Causal language model
    window_size : int
        Context window size
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added novelty columns
    """
    # Pre-tokenize
    df["user_ids"] = df["user_corrected"].apply(
        lambda txt: tokenizer(txt, add_special_tokens=False)["input_ids"]
    )
    df["ai_ids"] = df["ai"].apply(
        lambda txt: tokenizer(txt, add_special_tokens=False)["input_ids"]
    )
    
    context_buffer = []
    last_client = None
    user_surprise, user_entropy = [], []
    ai_surprise, ai_entropy = [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing novelty"):
        client = row.get("client_id", None)
        
        # Reset context at new session
        if last_client is None or (client is not None and client != last_client):
            context_buffer = []
            last_client = client
        
        # User novelty
        u_ids = row["user_ids"]
        avg_s, total_s = calc_sentence_surprisal(context_buffer, u_ids, model, window_size)
        user_surprise.append(avg_s)
        user_entropy.append(total_s)
        context_buffer.extend(u_ids)
        
        # AI novelty
        a_ids = row["ai_ids"]
        avg_a, total_a = calc_sentence_surprisal(context_buffer, a_ids, model, window_size)
        ai_surprise.append(avg_a)
        ai_entropy.append(total_a)
        context_buffer.extend(a_ids)
    
    df["user_surprise"] = user_surprise
    df["ai_surprise"] = ai_surprise
    df["user_entropy"] = user_entropy
    df["ai_entropy"] = ai_entropy
    
    return df


def compute_transience_scores(df, tokenizer, model, window_size=128):
    """
    Compute transience scores for user and AI utterances.
    
    Transience = how surprising future text is given this utterance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must already have user_ids and ai_ids columns)
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    model : transformers.PreTrainedModel
        Causal language model
    window_size : int
        Context window size
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added transience columns
    """
    def gather_future_tokens(start_idx, df, window_size):
        """Gather up to window_size tokens from subsequent rows."""
        future_ids = []
        for j in range(start_idx + 1, len(df)):
            for col in ["user_ids", "ai_ids"]:
                for tid in df.at[j, col]:
                    future_ids.append(tid)
                    if len(future_ids) >= window_size:
                        return future_ids
        return future_ids
    
    user_transience = []
    ai_transience = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Computing transience"):
        future_ids = gather_future_tokens(i, df, window_size)
        
        # User transience
        u_context = row["user_ids"][-window_size:]
        if future_ids:
            avg_fut, _ = calc_sentence_surprisal(u_context, future_ids, model, window_size)
        else:
            avg_fut = 0.0
        user_transience.append(avg_fut)
        
        # AI transience
        a_context = row["ai_ids"][-window_size:]
        if future_ids:
            avg_fut_ai, _ = calc_sentence_surprisal(a_context, future_ids, model, window_size)
        else:
            avg_fut_ai = 0.0
        ai_transience.append(avg_fut_ai)
    
    df["user_transience"] = user_transience
    df["ai_transience"] = ai_transience
    
    return df
