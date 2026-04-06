import pandas as pd 
import textdescriptives as td
import spacy


nlp = spacy.load('en_core_web_md')  

def get_descriptive_metrics_dual_full_long(
        df: pd.DataFrame,
        author_1_col: str = "full_author_1_dot",
        author_2_col: str = "full_author_2_dot",
        spacy_mdl: str = "en_core_web_md",
        batch_size: int = 10,
        n_process: int = 5):
    """
    Compute text descriptives for full story texts.
    
    Uses standardized column names: author_1, author_2.
    """
    print(f"[INFO:] Loading spaCy model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/all")

    # ----- AUTHOR 1 -----
    print(f"[INFO:] Extracting Author 1 metrics...")
    docs_author_1 = nlp.pipe(df[author_1_col], batch_size=batch_size, n_process=n_process)
    author_1_metrics = td.extract_df(docs_author_1, include_text=True)
    author_1_metrics.index = df.index
    author_1_metrics["type"] = "author_1"
    if "conversation_id" in df.columns:
        author_1_metrics["conversation_id"] = df["conversation_id"]

    # ----- AUTHOR 2 -----
    print(f"[INFO:] Extracting Author 2 metrics...")
    docs_author_2 = nlp.pipe(df[author_2_col], batch_size=batch_size, n_process=n_process)
    author_2_metrics = td.extract_df(docs_author_2, include_text=True)
    author_2_metrics.index = df.index
    author_2_metrics["type"] = "author_2"
    if "conversation_id" in df.columns:
        author_2_metrics["conversation_id"] = df["conversation_id"]

    # ----- STACK LONG -----
    print("[INFO:] Combining metrics (long format)...")
    metrics_long = pd.concat([author_1_metrics, author_2_metrics], axis=0)

    return metrics_long

def get_descriptive_metrics_dual_inter_long(
        df: pd.DataFrame,
        author_1_col: str = "author_1",
        author_2_col: str = "author_2",
        spacy_mdl: str = "en_core_web_md",
        batch_size: int = 10,
        n_process: int = 5):
    """
    Compute text descriptives for interaction-level texts.
    
    Uses standardized column names: author_1, author_2.
    """
    print(f"[INFO:] Loading spaCy model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/all")
    
    # Ensure NaNs don't crash the script 
    df[author_1_col] = df[author_1_col].fillna("")
    df[author_2_col] = df[author_2_col].fillna("")

    # ----- AUTHOR 1 -----
    print(f"[INFO:] Extracting Author 1 metrics...")
    
    docs_author_1 = nlp.pipe(df[author_1_col], batch_size=batch_size, n_process=n_process)
    author_1_metrics = td.extract_df(docs_author_1, include_text=True)
    author_1_metrics.index = df.index
    author_1_metrics["type"] = "author_1"
    author_1_metrics["interaction_count"] = df["interaction_count"]
    author_1_metrics["starter"] = df["starter"]
    
    if "conversation_id" in df.columns:
        author_1_metrics["conversation_id"] = df["conversation_id"]

    # ----- AUTHOR 2 -----
    print(f"[INFO:] Extracting Author 2 metrics...")
    docs_author_2 = nlp.pipe(df[author_2_col], batch_size=batch_size, n_process=n_process)
    author_2_metrics = td.extract_df(docs_author_2, include_text=True)
    author_2_metrics.index = df.index
    author_2_metrics["type"] = "author_2"
    author_2_metrics["interaction_count"] = df["interaction_count"]
    author_2_metrics["starter"] = df["starter"]

    if "conversation_id" in df.columns:
        author_2_metrics["conversation_id"] = df["conversation_id"]

    # ----- STACK LONG -----
    print("[INFO:] Combining metrics (long format)...")
    metrics_long = pd.concat([author_1_metrics, author_2_metrics], axis=0)
    metrics_long = metrics_long.reset_index(drop=True)

    return metrics_long
