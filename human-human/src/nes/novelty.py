"""
Novelty, transience, and resonance computation using language models.
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_language_model(model_path, device=None):
    """
    Load a causal language model and tokenizer.
    Returns (tokenizer, model, device).
    """
    # We don’t actually use `device` directly anymore; let device_map decide.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading language model from {model_path} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use a lighter dtype on GPU and let HF/accelerate place layers
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",        # shard/put on GPU/CPU as needed
        low_cpu_mem_usage=True,   # stream weights instead of loading all at once
    )
    model.eval()

    # Infer “device” from model params (first parameter’s device)
    device = next(model.parameters()).device
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
    # Use larger context for prediction history (up to model max or reasonable limit like 1024)
    # The 'window_size' parameter will now strictly control the FUTURE/TARGET evaluation window size
    prediction_history_limit = 1024 
    context_ids = context_ids[-prediction_history_limit:]
    
    # We only care about the target tokens being predicted
    # If target is longer than window_size, we crop it (for Transience consistency)
    # For Novelty, target_ids is the full turn, so we usually want the full turn.
    # To support both, we'll respect the passed target_ids length, but caller should trim if needed.
    # But wait, for Transience dilution check, we want to limit target_ids to say 40.
    # Let's handle trimming in the caller functions instead of here for flexibility.
    
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
        # Safety check for index
        if pos >= logits.shape[1]:
             break 
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
    df["user_ids"] = df["user"].apply(
        lambda txt: tokenizer(
            "" if pd.isna(txt) else str(txt),
            add_special_tokens=False
        )["input_ids"]
    )

    df["ai_ids"] = df["ai"].apply(
        lambda txt: tokenizer(
            "" if pd.isna(txt) else str(txt),
            add_special_tokens=False
        )["input_ids"]
    )
    
    # Identify a safe start token for unconditional probability
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    base_context = [bos_token_id] if bos_token_id is not None else []
    
    # Initialize context buffer with BOS token to ensure first turn works correctly
    # Otherwise calc_sentence_surprisal can have negative index issue on first turn (context=[])
    context_buffer = [bos_token_id] if bos_token_id is not None else []
    
    last_client = None
    user_novelty, user_raw, user_entropy = [], [], []
    ai_novelty, ai_raw, ai_entropy = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing novelty"):
        client = row.get("conversation_id", None)
        
        # Reset context at new session
        if last_client is None or (client is not None and client != last_client):
            context_buffer = [bos_token_id] if bos_token_id is not None else []
            last_client = client
        
        # User novelty
        u_ids = row["user_ids"]
        # 1. Nominal (Conditional)
        avg_s, total_s = calc_sentence_surprisal(context_buffer, u_ids, model, window_size)
        # 2. Baseline (Unconditional)
        avg_base, _ = calc_sentence_surprisal(base_context, u_ids, model, window_size)
        
        user_raw.append(avg_s) # Keep original for reference
        # The new metric: Cond - Uncond. 
        # If context helps, Cond < Uncond => Negative value.
        # High Novelty (Deviation) => Cond ~ Uncond => Closer to 0.
        user_novelty.append(avg_s - avg_base) 
        user_entropy.append(total_s)
        
        context_buffer.extend(u_ids)
        
        # AI novelty
        a_ids = row["ai_ids"]
        avg_a, total_a = calc_sentence_surprisal(context_buffer, a_ids, model, window_size)
        avg_base_a, _ = calc_sentence_surprisal(base_context, a_ids, model, window_size)
        
        ai_raw.append(avg_a)
        ai_novelty.append(avg_a - avg_base_a)
        ai_entropy.append(total_a)
        context_buffer.extend(a_ids)
    
    # We overwrite 'user_surprise' with the new metric to propagate the fix,
    # but we save 'user_surprise_raw' just in case.
    df["user_surprise"] = user_novelty
    df["ai_surprise"] = ai_novelty
    df["user_surprise_raw"] = user_raw
    df["ai_surprise_raw"] = ai_raw
    
    df["user_entropy"] = user_entropy
    df["ai_entropy"] = ai_entropy
    
    return df


def compute_transience_scores(df, tokenizer, model, window_size=40):
    """
    Compute transience scores for user and AI utterances.

    Transience = how surprising the immediate next turn is given this utterance.
    For user turns, the next turn is the paired AI response in the same row.
    For AI turns, the next turn is the next row's user response in the same
    conversation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must already have user_ids and ai_ids columns)
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    model : transformers.PreTrainedModel
        Causal language model
    window_size : int
        Kept for backward compatibility; not used when scoring full next turn.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added transience columns
    """
    # Identify a safe start token for unconditional probability
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    base_context = [bos_token_id] if bos_token_id is not None else []

    def gather_next_user_ids(pos, frame, current_conversation_id):
        """Get full user token sequence from the next row in same conversation."""
        next_pos = pos + 1
        if next_pos >= len(frame):
            return []
        next_row = frame.iloc[next_pos]
        if next_row.get("conversation_id", None) != current_conversation_id:
            return []
        return next_row["user_ids"]
    
    user_transience, user_raw = [], []
    ai_transience, ai_raw = [], []
    
    for pos in tqdm(range(len(df)), total=len(df), desc="Computing transience"):
        row = df.iloc[pos]
        current_conversation_id = row.get("conversation_id", None)
        
        # User transience target: full paired AI response (same row)
        future_ids_user = row["ai_ids"]
        # AI transience target: full next user turn (next row, same conversation)
        future_ids_ai = gather_next_user_ids(pos, df, current_conversation_id)
        
        # User transience (Predictor is just the current turn)
        # We assume Transience context is just the immediate past turn (no full history), 
        # as defined in Barron approx.
        u_context = row["user_ids"] # Use full user turn as context
        
        if future_ids_user:
            avg_fut, _ = calc_sentence_surprisal(u_context, future_ids_user, model, window_size)
            avg_base, _ = calc_sentence_surprisal(base_context, future_ids_user, model, window_size)
            # Metric: Cond - Uncond
            user_transience.append(avg_fut - avg_base)
            user_raw.append(avg_fut)
        else:
            user_transience.append(0.0)
            user_raw.append(0.0)
        
        # AI transience
        a_context = row["ai_ids"] # Use full AI turn as context
        if future_ids_ai:
            avg_fut_ai, _ = calc_sentence_surprisal(a_context, future_ids_ai, model, window_size)
            avg_base_ai, _ = calc_sentence_surprisal(base_context, future_ids_ai, model, window_size)
            ai_transience.append(avg_fut_ai - avg_base_ai)
            ai_raw.append(avg_fut_ai)
        else:
            ai_transience.append(0.0)
            ai_raw.append(0.0)
    
    df["user_transience"] = user_transience
    df["ai_transience"] = ai_transience
    df["user_transience_raw"] = user_raw
    df["ai_transience_raw"] = ai_raw
    
    return df
