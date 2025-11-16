import pandas as pd
import os

file_path = "../Data/finished_stories_corrected.csv"  # Change if needed
df = pd.read_csv(file_path)

def filter_by_edit_distance(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Filter rows based on edit distance threshold."""
    filtered_df = df[df["edit_distance"] <= threshold].copy()
    return filtered_df

FILE_NAME = "../Data/finished_stories_corrected.csv"
filtered_df = filter_by_edit_distance(df, threshold=100)
filtered_df.to_csv(FILE_NAME, index=False)
