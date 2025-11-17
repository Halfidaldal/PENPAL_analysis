import pandas as pd 
import textdescriptives as td
import spacy

nlp = spacy.load('en_core_web_md')  

def get_descriptive_metrics_dual_long(
        df: pd.DataFrame,
        user_col: str = "full_user",
        ai_col: str = "full_ai",
        spacy_mdl: str = "en_core_web_md",
        batch_size: int = 10,
        n_process: int = 5):

    print(f"[INFO:] Loading spaCy model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    nlp.add_pipe("textdescriptives/all")

    # ----- USER -----
    print(f"[INFO:] Extracting USER metrics...")
    docs_user = nlp.pipe(df[user_col], batch_size=batch_size, n_process=n_process)
    user_metrics = td.extract_df(docs_user, include_text=True)
    user_metrics.index = df.index
    user_metrics["type"] = "user"
    if "conversation_id" in df.columns:
        user_metrics["conversation_id"] = df["conversation_id"]

    # ----- AI -----
    print(f"[INFO:] Extracting AI metrics...")
    docs_ai = nlp.pipe(df[ai_col], batch_size=batch_size, n_process=n_process)
    ai_metrics = td.extract_df(docs_ai, include_text=True)
    ai_metrics.index = df.index
    ai_metrics["type"] = "ai"
    if "conversation_id" in df.columns:
        ai_metrics["conversation_id"] = df["conversation_id"]

    # ----- STACK LONG -----
    print("[INFO:] Combining metrics (long format)...")
    metrics_long = pd.concat([user_metrics, ai_metrics], axis=0)

    return metrics_long