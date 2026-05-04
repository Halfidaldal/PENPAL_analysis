"""
Novelty, transience, and resonance computation using language models.
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


_MISSING_TEXT_VALUES = {"", "nan", "none", "null"}


def _normalize_metric_text(value):
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


def _tokenize_metric_text(value, tokenizer):
    """Normalize scalar text-like values before tokenization."""
    text = _normalize_metric_text(value)
    if text is None:
        return []
    return tokenizer(text, add_special_tokens=False)["input_ids"]

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

def calc_sentence_surprisal(context_ids, target_ids, model, window_size=None):
    """
    Compute average surprisal (bits/token) and total surprisal (bits)
    for target tokens given context, using dynamic context window limits.
    """
    # Dynamically extract model's maximum context length, defaulting to 128k for Maverick if undefined
    max_context = getattr(model.config, "max_position_embeddings", 131072) 
    
    # Define how much history we can safely keep (leaving room for the target sentence)
    prediction_history_limit = max_context - len(target_ids) - 1
    
    # Truncate context to fit within the model's actual limits rather than a hardcoded 1024
    context_ids = context_ids[-prediction_history_limit:]
    combined_ids = context_ids + target_ids
    
    if len(combined_ids) < 2 or not target_ids:
        return 0.0, 0.0
    
    # Convert to tensors and explicitly create an attention mask
    input_tensor = torch.tensor([combined_ids], device=model.device)
    attention_mask = torch.ones_like(input_tensor, device=model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
    
    logits = outputs.logits[:, :-1, :]  # Distributions for each next token
    
    total_nll = 0.0
    for idx, token_id in enumerate(target_ids):
        pos = len(context_ids) + idx - 1
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
    Compute novelty (surprise) scores for author_1 and author_2 utterances.
    
    Uses standardized column names: author_1, author_2.
    
    Novelty = how surprising this utterance is given prior context.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with author_1 and author_2 text columns
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
    df = df.copy()
    sort_columns = [col for col in ["conversation_id", "turn", "timestamp"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    author_1_text = df["author_1"].apply(_normalize_metric_text)
    author_2_text = df["author_2"].apply(_normalize_metric_text)
    author_1_has_text = author_1_text.notna()
    author_2_has_text = author_2_text.notna()

    df["author_1_ids"] = author_1_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    df["author_2_ids"] = author_2_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    
    # Identify a safe start token for unconditional probability
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    bos_ids = [bos_token_id] if bos_token_id is not None else []
    anchor_text = "Speaker:"
    anchor_ids = tokenizer(anchor_text, add_special_tokens=False)["input_ids"]
    # base_context = unconditional baseline (BOS + anchor); reused across all turns
    base_context = bos_ids + anchor_ids
    # context_buffer grows with story tokens and resets per conversation
    context_buffer = bos_ids + anchor_ids

    
    last_client = None
    author_1_novelty, author_1_raw, author_1_entropy = [], [], []
    author_2_novelty, author_2_raw, author_2_entropy = [], [], []
    
    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Computing novelty")):
        client = row.get("conversation_id", None)
        
        # Reset context at new session
        if last_client is None or (client is not None and client != last_client):
            context_buffer = bos_ids + anchor_ids
            last_client = client
        
        # Author 1 novelty
        a1_ids = row["author_1_ids"]
        if author_1_has_text.iloc[pos] and a1_ids:
            avg_s, total_s = calc_sentence_surprisal(context_buffer, a1_ids, model, window_size)
            avg_base, _ = calc_sentence_surprisal(base_context, a1_ids, model, window_size)
            author_1_raw.append(avg_s)
            author_1_novelty.append(avg_s - avg_base)
            author_1_entropy.append(total_s)
            context_buffer.extend(a1_ids)
        else:
            author_1_raw.append(np.nan)
            author_1_novelty.append(np.nan)
            author_1_entropy.append(np.nan)
        
        # Author 2 novelty
        a2_ids = row["author_2_ids"]
        if author_2_has_text.iloc[pos] and a2_ids:
            avg_a, total_a = calc_sentence_surprisal(context_buffer, a2_ids, model, window_size)
            avg_base_a, _ = calc_sentence_surprisal(base_context, a2_ids, model, window_size)
            author_2_raw.append(avg_a)
            author_2_novelty.append(avg_a - avg_base_a)
            author_2_entropy.append(total_a)
            context_buffer.extend(a2_ids)
        else:
            author_2_raw.append(np.nan)
            author_2_novelty.append(np.nan)
            author_2_entropy.append(np.nan)
    
    # Use standardized column names
    df["author_1_surprise"] = author_1_novelty
    df["author_2_surprise"] = author_2_novelty
    df["author_1_surprise_raw"] = author_1_raw
    df["author_2_surprise_raw"] = author_2_raw
    
    df["author_1_entropy"] = author_1_entropy
    df["author_2_entropy"] = author_2_entropy
    
    return df


def compute_transience_scores(df, tokenizer, model, window_size=40):
    """
    Compute transience scores for author_1 and author_2 utterances.

    Uses standardized column names: author_1, author_2.

    Transience = how surprising the immediate next turn is given this utterance.
    For author_1 turns, the next turn is the paired author_2 response in the same row.
    For author_2 turns, the next turn is the next row's author_1 response in the same
    conversation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must already have author_1_ids and author_2_ids columns)
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
    df = df.copy()
    sort_columns = [col for col in ["conversation_id", "turn", "timestamp"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    author_1_has_text = df["author_1"].apply(_normalize_metric_text).notna()
    author_2_has_text = df["author_2"].apply(_normalize_metric_text).notna()

    # Identify a safe start token for unconditional probability
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    base_context = [bos_token_id] if bos_token_id is not None else []

    def gather_next_author_1_ids(pos, frame, current_conversation_id):
        """Get full author_1 token sequence from the next row in same conversation."""
        next_pos = pos + 1
        if next_pos >= len(frame):
            return []
        next_row = frame.iloc[next_pos]
        if next_row.get("conversation_id", None) != current_conversation_id:
            return []
        return next_row["author_1_ids"]
    
    author_1_transience, author_1_raw = [], []
    author_2_transience, author_2_raw = [], []
    
    for pos in tqdm(range(len(df)), total=len(df), desc="Computing transience"):
        row = df.iloc[pos]
        current_conversation_id = row.get("conversation_id", None)
        
        # Author 1 transience target: full paired author_2 response (same row)
        future_ids_author_1 = row["author_2_ids"]
        # Author 2 transience target: full next author_1 turn (next row, same conversation)
        future_ids_author_2 = gather_next_author_1_ids(pos, df, current_conversation_id)
        
        # Author 1 transience (Predictor is just the current turn)
        a1_context = row["author_1_ids"]
        
        if author_1_has_text.iloc[pos] and future_ids_author_1:
            avg_fut, _ = calc_sentence_surprisal(a1_context, future_ids_author_1, model, window_size)
            avg_base, _ = calc_sentence_surprisal(base_context, future_ids_author_1, model, window_size)
            # Metric: Cond - Uncond
            author_1_transience.append(avg_fut - avg_base)
            author_1_raw.append(avg_fut)
        else:
            author_1_transience.append(np.nan)
            author_1_raw.append(np.nan)
        
        # Author 2 transience
        a2_context = row["author_2_ids"]
        if author_2_has_text.iloc[pos] and future_ids_author_2:
            avg_fut_a2, _ = calc_sentence_surprisal(a2_context, future_ids_author_2, model, window_size)
            avg_base_a2, _ = calc_sentence_surprisal(base_context, future_ids_author_2, model, window_size)
            author_2_transience.append(avg_fut_a2 - avg_base_a2)
            author_2_raw.append(avg_fut_a2)
        else:
            author_2_transience.append(np.nan)
            author_2_raw.append(np.nan)
    
    df["author_1_transience"] = author_1_transience
    df["author_2_transience"] = author_2_transience
    df["author_1_transience_raw"] = author_1_raw
    df["author_2_transience_raw"] = author_2_raw
    
    return df
