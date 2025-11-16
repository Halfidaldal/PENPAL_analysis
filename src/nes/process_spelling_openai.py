from typing import List, Optional
from pathlib import Path
import openai
import pandas as pd
import time
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

# Function to correct spelling
NaS = 0

def correct_spelling(text, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    global NaS  # Ensure global variable access
    if pd.isna(text) or not isinstance(text, str):
        NaS += 1
        return text  # Return original if NaN or not a string
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only corrects spelling mistakes while keeping the original meaning intact. If nonsense, respond with a long message indicating that."},
            {"role": "user", "content": text}
        ],
        temperature=0.1  # Ensures minimal alteration
    )
    return response.choices[0].message.content

# Function to calculate edit distance
def compute_edit_distance(row):
    if pd.isna(row["user"]) or pd.isna(row["user_corrected"]):
        return None
    return levenshtein_distance(str(row["user"]), str(row["user_corrected"]))
