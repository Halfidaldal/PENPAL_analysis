from pandas._libs import indexing
from pandas._libs import indexing
import math
import numpy as np
import pandas as pd
import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

_MISSING_TEXT_VALUES = {"", "nan", "none", "null"}

def build_dataset(filepath):
    records = []
    body, conclusion = load_and_parse_text(filepath)
    records.append({"body": body, "conclusion": conclusion})
    return pd.DataFrame(records)
    
    

def load_and_parse_text(filepath):
    with open(filepath, 'r') as f:
        latex_content = f.read()
    
    # Preprocess to strip \href{url} prefixes, leaving {text} to prevent pylatexenc parser crashes
    import re
    latex_content = re.sub(r'\\href\s*\{[^{}]*\}', '', latex_content)

    plain_text = LatexNodes2Text().latex_to_text(latex_content)

    paragraphs = [p.strip() for p in plain_text.replace('\r\n', '\n').split('\n\n') if p.strip()]
    conclusion_para_idx = None
    for i, para in enumerate(paragraphs):
        if "conclusion" in para.lower():
            if len(para) < 30 and i + 1 < len(paragraphs):
                conclusion_para_idx = i + 1 
            else:
                conclusion_para_idx = i
            break
    if conclusion_para_idx is None:
        raise ValueError("Could not find conclusion in the text.")
    body_text = " ".join(paragraphs[:conclusion_para_idx])
    body = nltk.sent_tokenize(body_text)
    conclusion = paragraphs[conclusion_para_idx] #we only want the conclusion as a single string and not what comes 
    return body, conclusion


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

def compute_endpoint_surprisal_scores(df, tokenizer, model):
    """
    Compute endpoint surprisal trajectories and derive information-flow features.
    
    Adds the following columns to the DataFrame:
      - mean_delta_s
      - var_delta_s
      - linearity_r2
      - cum_slope
      - tail_drop_ratio
      - raw_surprisals
      - raw_drops
    """
    df = df.copy()
    
    # Identify a safe start token for the baseline (BOS or EOS)
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    bos_ids = [bos_token_id] if bos_token_id is not None else []

    # Lists to accumulate row-wise metrics
    mean_delta_s_list = []
    var_delta_s_list = []
    linearity_r2_list = []
    cum_slope_list = []
    tail_drop_ratio_list = []
    raw_surprisals_list = []
    raw_drops_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing endpoint surprisals"):
        body = row['body']  # List of sentences
        conclusion = row['conclusion']  # Single string
        
        # Tokenize conclusion and body sentences
        target_ids = _tokenize_metric_text(conclusion, tokenizer)
        sentence_ids_list = [_tokenize_metric_text(sent, tokenizer) for sent in body]
        
        # 1. Compute baseline surprisal S(0)
        context_buffer = list(bos_ids)
        avg_s_0, _ = calc_sentence_surprisal(context_buffer, target_ids, model)
        surprisals = [avg_s_0]
        
        # 2. Iterate sentence-by-sentence to compute S(t)
        for sent_ids in sentence_ids_list:
            context_buffer.extend(sent_ids)
            avg_s_t, _ = calc_sentence_surprisal(context_buffer, target_ids, model)
            surprisals.append(avg_s_t)
            
        # 3. Calculate surprisal drops (Δ_i = S(i-1) - S(i))
        drops = [surprisals[i-1] - surprisals[i] for i in range(1, len(surprisals))]
        
        # 4. Compute features
        if len(drops) > 0:
            mean_delta_s = np.mean(drops)
            var_delta_s = np.var(drops)
            
            # Regression metrics (Cumulative Drop vs. Position index)
            y = np.array([surprisals[0] - S_t for S_t in surprisals[1:]])
            x = np.array(list(range(1, len(y) + 1)))
            
            if len(y) >= 2:
                # Fit linear regression line: y = mx + c
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                
                # Calculate R^2 (coefficient of determination)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                slope = 0.0
                r2 = 0.0
                
            # Tail-drop ratio (e.g., last 20% of sentences vs. first 80%)
            tail_size = max(1, int(len(drops) * 0.2))
            drops_tail = drops[-tail_size:]
            drops_other = drops[:-tail_size]
            
            mean_tail = np.mean(drops_tail)
            mean_other = np.mean(drops_other) if len(drops_other) > 0 else 0.0
            
            if abs(mean_other) > 1e-9:
                tail_drop_ratio = abs(mean_tail / mean_other)
            else:
                tail_drop_ratio = 0.0
        else:
            mean_delta_s = 0.0
            var_delta_s = 0.0
            r2 = 0.0
            slope = 0.0
            tail_drop_ratio = 0.0

        # Append to row-wise lists
        mean_delta_s_list.append(mean_delta_s)
        var_delta_s_list.append(var_delta_s)
        linearity_r2_list.append(r2)
        cum_slope_list.append(slope)
        tail_drop_ratio_list.append(tail_drop_ratio)
        raw_surprisals_list.append(surprisals)
        raw_drops_list.append(drops)

    # Assign new columns to copied DataFrame
    df["mean_delta_s"] = mean_delta_s_list
    df["var_delta_s"] = var_delta_s_list
    df["linearity_r2"] = linearity_r2_list
    df["cum_slope"] = cum_slope_list
    df["tail_drop_ratio"] = tail_drop_ratio_list
    df["raw_surprisals"] = raw_surprisals_list
    df["raw_drops"] = raw_drops_list
    
    return df
