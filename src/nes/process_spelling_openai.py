from typing import List, Optional
from google import genai
from google.genai import types
import pandas as pd
import time
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

# Function to correct spelling
NaS = 0

def correct_spelling(text, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    global NaS  # Ensure global variable access
    if pd.isna(text) or not isinstance(text, str):
        NaS += 1
        return text  # Return original if NaN or not a string
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant that only corrects spelling mistakes while keeping the original meaning intact. If input is nonsense or error messages, reply with a long message explaining.",
            temperature=0.2),        
        contents=[
            text
        ]
    )
    return response.text


def compute_edit_distance_values(original_text, corrected_text):
    """Compute Levenshtein distance directly from two text values."""
    if pd.isna(original_text) or pd.isna(corrected_text):
        return None
    return levenshtein_distance(str(original_text), str(corrected_text))
