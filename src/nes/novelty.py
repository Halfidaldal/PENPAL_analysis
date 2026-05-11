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
    scored_tokens = 0
    for idx, token_id in enumerate(target_ids):
        pos = len(context_ids) + idx - 1
        if pos < 0:
            continue
        if pos >= logits.shape[1]:
             break 
        dist = logits[0, pos]
        log_probs = torch.nn.functional.log_softmax(dist, dim=-1)
        log2p = log_probs[token_id] / math.log(2)
        total_nll += -log2p.item()
        scored_tokens += 1
    
    if scored_tokens == 0:
        return 0.0, 0.0

    avg_nll = total_nll / scored_tokens
    return avg_nll, total_nll


def _compute_chronological_order(df):
    """
    Determine the chronological within-row slot order for each conversation.

    A row stores one exchange = a pair of consecutive turns. The data
    convention is "author_1 holds the chronologically-first turn within each
    complete row" -- which is true pre-randomization for every condition. The
    50/50 column-swap applied to HH and AA by `randomize_author_assignment`
    inverts that order for half the stories without reordering the row
    contents, so the surprisal computation must process those rows in
    (author_2, author_1) order to respect chronology.

    Detection rule (intrinsic to the data, no need to know the condition):
        - starter == 'author_1'                          -> ('author_1', 'author_2')
        - starter == 'author_2' AND first row's author_1
          is empty/non-substantive                       -> ('author_1', 'author_2')
              (HA AI-started case: row 1 is an incomplete primer holding the
              AI's opener in author_2 only; from row 2 onwards each complete
              row still pairs (user/author_1, AI/author_2) chronologically.)
        - starter == 'author_2' AND first row's author_1
          is substantive                                 -> ('author_2', 'author_1')
              (HH/AA randomization-swap case: author_2 holds the chronological
              starter of every row.)

    Returns
    -------
    dict
        Maps conversation_id -> ordered tuple of slot names. Conversations
        without a recognizable `starter` value default to ('author_1','author_2').
    """
    default = ("author_1", "author_2")
    if "conversation_id" not in df.columns:
        return {}
    if "starter" not in df.columns:
        return {conv_id: default for conv_id in df["conversation_id"].unique()}

    sort_cols = [c for c in ("conversation_id", "turn", "timestamp") if c in df.columns]
    sorted_df = df.sort_values(sort_cols, kind="stable") if sort_cols else df

    order_map = {}
    swap_count = 0
    for conv_id, group in sorted_df.groupby("conversation_id", sort=False):
        starter_val = group["starter"].iloc[0]
        if starter_val != "author_2":
            order_map[conv_id] = default
            continue
        first_row = group.iloc[0]
        first_a1_text = _normalize_metric_text(first_row.get("author_1", None))
        if first_a1_text is None:
            # HA AI-primer row: keep default order; the empty author_1 slot
            # is handled naturally inside the main loop.
            order_map[conv_id] = default
        else:
            order_map[conv_id] = ("author_2", "author_1")
            swap_count += 1

    total = len(order_map)
    if total > 0:
        print(
            f"Chronological order: {swap_count}/{total} conversation(s) "
            f"processed in (author_2, author_1) order (post-randomization swap); "
            f"{total - swap_count} in default (author_1, author_2) order."
        )
    return order_map


def compute_novelty_scores(df, tokenizer, model, window_size=128):
    """
    Compute novelty (surprise) scores for author_1 and author_2 utterances.

    Uses standardized column names: author_1, author_2.

    Novelty = how surprising this utterance is given prior context.

    Within-row slot processing follows the chronological order returned by
    `_compute_chronological_order`. This ensures that Past is built up in the
    actual order turns were produced, even for HH/AA stories whose columns
    were swapped by `randomize_author_assignment`.
    
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
    has_text = {
        "author_1": author_1_text.notna(),
        "author_2": author_2_text.notna(),
    }

    df["author_1_ids"] = author_1_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    df["author_2_ids"] = author_2_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))

    chrono_order_map = _compute_chronological_order(df)
    default_order = ("author_1", "author_2")

    # Start without an artificial BOS or speaker anchor.
    base_context = []
    # context_buffer grows with story tokens and resets per conversation
    context_buffer = []

    last_client = None
    out = {
        "author_1": {"novelty": [], "raw": [], "entropy": []},
        "author_2": {"novelty": [], "raw": [], "entropy": []},
    }

    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Computing novelty")):
        client = row.get("conversation_id", None)

        # Reset context at new session
        if last_client is None or (client is not None and client != last_client):
            context_buffer = []
            last_client = client

        slot_order = chrono_order_map.get(client, default_order)
        row_results = {}

        for slot in slot_order:
            slot_ids = row[f"{slot}_ids"]
            if has_text[slot].iloc[pos] and slot_ids:
                avg_s, total_s = calc_sentence_surprisal(context_buffer, slot_ids, model, window_size)
                avg_base, _ = calc_sentence_surprisal(base_context, slot_ids, model, window_size)
                row_results[slot] = (avg_s - avg_base, avg_s, total_s)
                context_buffer.extend(slot_ids)
            else:
                row_results[slot] = (np.nan, np.nan, np.nan)

        # Append in fixed slot order so per-slot lists stay aligned with df rows
        # regardless of the chronological iteration order above.
        for slot in ("author_1", "author_2"):
            novelty_val, raw_val, entropy_val = row_results[slot]
            out[slot]["novelty"].append(novelty_val)
            out[slot]["raw"].append(raw_val)
            out[slot]["entropy"].append(entropy_val)

    # Use standardized column names
    df["author_1_surprise"] = out["author_1"]["novelty"]
    df["author_2_surprise"] = out["author_2"]["novelty"]
    df["author_1_surprise_raw"] = out["author_1"]["raw"]
    df["author_2_surprise_raw"] = out["author_2"]["raw"]

    df["author_1_entropy"] = out["author_1"]["entropy"]
    df["author_2_entropy"] = out["author_2"]["entropy"]

    return df


def compute_transience_scores(df, tokenizer, model, window_size=40):
    """
    Compute transience scores for author_1 and author_2 utterances using a
    counterfactual / ablation formulation:

        Transience_t = S(Future | Past) - S(Future | Past + s_t)

    Interpretation: how much does s_t help predict the immediate next turn,
    *over and above* the past context already shared. Positive values mean
    s_t added predictive value beyond the past; values near zero mean the
    past alone was already as informative as past + s_t.

    Within-row slot processing follows the chronological order returned by
    `_compute_chronological_order`. For the chronologically-first slot in a
    row, the future is the same-row second slot; for the second slot, the
    future is the next row's chronologically-first slot. This makes Future
    always equal "the immediately next turn in the conversation," even when
    the row's columns were swapped by `randomize_author_assignment`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must already have author_1_ids and author_2_ids columns
        from compute_novelty_scores; if not, they are tokenized here).
    tokenizer : transformers.PreTrainedTokenizer
    model : transformers.PreTrainedModel
    window_size : int
        Kept for backward compatibility; not used here.

    Returns
    -------
    pd.DataFrame
        DataFrame with added transience columns:
            author_1_transience, author_2_transience  (counterfactual)
            author_1_transience_raw, author_2_transience_raw (S(F | Past + s_t))
            author_1_transience_baseline, author_2_transience_baseline (S(F | Past))
    """
    df = df.copy()
    sort_columns = [col for col in ["conversation_id", "turn", "timestamp"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    author_1_text = df["author_1"].apply(_normalize_metric_text)
    author_2_text = df["author_2"].apply(_normalize_metric_text)
    has_text = {
        "author_1": author_1_text.notna(),
        "author_2": author_2_text.notna(),
    }

    # Tokenize if not already done by compute_novelty_scores
    if "author_1_ids" not in df.columns:
        df["author_1_ids"] = author_1_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    if "author_2_ids" not in df.columns:
        df["author_2_ids"] = author_2_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))

    chrono_order_map = _compute_chronological_order(df)
    default_order = ("author_1", "author_2")

    # Start without an artificial BOS or speaker anchor.
    base_context = []

    # We rebuild the past buffer turn by turn, exactly as in novelty.
    # This way Past_t means the same thing in both metrics.
    context_buffer = list(base_context)
    last_client = None

    def _gather_next_first_slot_ids(pos, current_conversation_id):
        """Return the chronologically-first slot's ids in the next row of the
        same conversation (the immediately next turn in the dialog), or [] if
        the next row belongs to a different conversation / does not exist."""
        next_pos = pos + 1
        if next_pos >= len(df):
            return []
        next_row = df.iloc[next_pos]
        if next_row.get("conversation_id", None) != current_conversation_id:
            return []
        next_order = chrono_order_map.get(current_conversation_id, default_order)
        next_first_slot = next_order[0]
        return next_row[f"{next_first_slot}_ids"]

    out = {
        "author_1": {"trans": [], "with": [], "without": []},
        "author_2": {"trans": [], "with": [], "without": []},
    }

    for pos in tqdm(range(len(df)), total=len(df), desc="Computing transience (counterfactual)"):
        row = df.iloc[pos]
        client = row.get("conversation_id", None)

        # Reset past buffer at session boundaries, same logic as novelty.
        if last_client is None or (client is not None and client != last_client):
            context_buffer = list(base_context)
            last_client = client

        slot_order = chrono_order_map.get(client, default_order)
        first_slot, second_slot = slot_order

        first_ids = row[f"{first_slot}_ids"]
        second_ids = row[f"{second_slot}_ids"]
        row_results = {}

        # ---- First (chronologically) slot transience ----
        # Past   = everything before this row (already in context_buffer).
        # s_t    = first_slot turn (first_ids).
        # Future = second_slot turn in this same row (second_ids).
        if has_text[first_slot].iloc[pos] and first_ids and second_ids:
            past = list(context_buffer)
            past_plus_st = past + first_ids

            avg_with, _ = calc_sentence_surprisal(past_plus_st, second_ids, model, window_size)
            avg_without, _ = calc_sentence_surprisal(past, second_ids, model, window_size)

            # S(F | Past) - S(F | Past + s_t): positive = s_t helped.
            row_results[first_slot] = (avg_with - avg_without, avg_with, avg_without)
        else:
            row_results[first_slot] = (np.nan, np.nan, np.nan)

        # Append first_slot to the past buffer BEFORE second_slot transience,
        # so that for the second_slot measure Past correctly contains first_ids.
        if has_text[first_slot].iloc[pos] and first_ids:
            context_buffer.extend(first_ids)

        # ---- Second (chronologically) slot transience ----
        # Past   = everything up to and including first_slot this row.
        # s_t    = second_slot turn (second_ids).
        # Future = next row's chronologically-first slot (i.e., the next turn
        #          in the conversation), same conversation.
        future_second = _gather_next_first_slot_ids(pos, client)
        if has_text[second_slot].iloc[pos] and second_ids and future_second:
            past = list(context_buffer)
            past_plus_st = past + second_ids

            avg_with, _ = calc_sentence_surprisal(past_plus_st, future_second, model, window_size)
            avg_without, _ = calc_sentence_surprisal(past, future_second, model, window_size)

            row_results[second_slot] = (avg_with - avg_without, avg_with, avg_without)
        else:
            row_results[second_slot] = (np.nan, np.nan, np.nan)

        # Append second_slot to past buffer for subsequent turns.
        if has_text[second_slot].iloc[pos] and second_ids:
            context_buffer.extend(second_ids)

        # Append in fixed slot order so per-slot lists stay aligned with df rows
        # regardless of the chronological iteration order above.
        for slot in ("author_1", "author_2"):
            trans_val, with_val, without_val = row_results[slot]
            out[slot]["trans"].append(trans_val)
            out[slot]["with"].append(with_val)
            out[slot]["without"].append(without_val)

    df["author_1_transience"] = out["author_1"]["trans"]
    df["author_2_transience"] = out["author_2"]["trans"]
    df["author_1_transience_raw"] = out["author_1"]["with"]         # S(F | Past + s_t)
    df["author_2_transience_raw"] = out["author_2"]["with"]
    df["author_1_transience_baseline"] = out["author_1"]["without"] # S(F | Past)
    df["author_2_transience_baseline"] = out["author_2"]["without"]

    return df
