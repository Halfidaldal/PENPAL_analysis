import pandas as pd
import numpy as np

def check_length_bias():
    file_path = "data/TEXT/processed/novelty_scores_128.csv"
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    # Filter as in the analysis
    df = df[(df['interaction_count'] > 1) & (df['interaction_count'] < 10)]
    df = df.dropna(subset=['user_surprise', 'ai_surprise', 'user', 'ai'])

    # Approx token length (words)
    df['user_len'] = df['user'].apply(lambda x: len(str(x).split()))
    df['ai_len'] = df['ai'].apply(lambda x: len(str(x).split()))

    print("\n--- Length Stats ---")
    print(f"User Length: Mean={df['user_len'].mean():.1f}, SD={df['user_len'].std():.1f}")
    print(f"AI Length:   Mean={df['ai_len'].mean():.1f}, SD={df['ai_len'].std():.1f}")

    print("\n--- Correlation with Novelty (Raw Surprisal) ---")
    u_corr = df['user_len'].corr(df['user_surprise'])
    a_corr = df['ai_len'].corr(df['ai_surprise'])

    print(f"User Length vs Novelty Corr: {u_corr:.3f}")
    print(f"AI Length vs Novelty Corr:   {a_corr:.3f}")

if __name__ == "__main__":
    check_length_bias()
