import openai
import pandas as pd
import time
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

# Load the CSV file
file_path = "data/finished_stories.csv"  # Change if needed
df = pd.read_csv(file_path)

# OpenAI API Key (Replace with your actual API key)
client = openai.OpenAI(api_key="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")

# Function to correct spelling
NaS = 0

def correct_spelling(text):
    global NaS  # Ensure global variable access
    if pd.isna(text) or not isinstance(text, str):
        NaS += 1
        return text  # Return original if NaN or not a string
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that only corrects spelling mistakes while keeping the original meaning intact."},
            {"role": "user", "content": text}
        ],
        temperature=0.0  # Ensures minimal alteration
    )
    return response.choices[0].message.content

# Function to calculate edit distance
def compute_edit_distance(row):
    if pd.isna(row["user"]) or pd.isna(row["user_corrected"]):
        return None
    return levenshtein_distance(str(row["user"]), str(row["user_corrected"]))

# Start timing
start_time = time.time()

# Apply the function to the 'user' column with a progress bar
df["user_corrected"] = [correct_spelling(text) for text in tqdm(df["user"], desc="Processing Text")]

# Compute edit distance with a progress bar
df["edit_distance"] = [compute_edit_distance(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Edit Distance")]

# Save the corrected CSV
FILE_NAME = "data_steno_corrected.csv"
df.to_csv(FILE_NAME, index=False)

# End timing
end_time = time.time()
print(f"Number of missing texts: {NaS}")
print(f"Spelling correction complete. Saved as '{FILE_NAME}'. Time taken: {end_time - start_time:.2f} seconds")
