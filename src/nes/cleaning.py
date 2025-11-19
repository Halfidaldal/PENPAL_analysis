"""
Data cleaning and filtering functions.

This module handles:
- Firestore data download and filtering
- Edit distance filtering
- Spell-checking and rectification
- Story grouping (10 interactions = 1 story)
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter


def init_firestore(credentials_path: str) -> firestore.Client:
    """
    Initialize Firestore client.
    
    Args:
        credentials_path: Path to Firebase admin SDK JSON file
        
    Returns:
        Firestore client instance
    """
    try:
        # Check if already initialized
        firebase_admin.get_app()
    except ValueError:
        # Not initialized yet
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    
    return firestore.client()


def download_stories_from_firestore(
    db: firestore.Client,
    collection_name: str = "story_data_TEXT",
    min_interactions: int = 10
) -> pd.DataFrame:
    """
    Download stories from Firestore, keeping only complete stories
    (conversations with at least min_interactions).
    
    Args:
        db: Firestore client
        collection_name: Name of the Firestore collection
        min_interactions: Minimum number of interactions to count as a story
        
    Returns:
        DataFrame with story interaction rows (only complete stories)
    """
    story_data_ref = db.collection(collection_name)
    
    # Build per-conversation chunks and return only rows that belong to full stories
    count = {}
    out_rows = []  # flattened list of dicts for all full-story rows
    full_story_data = []  # list of story groups (each is a list of dicts)
    current_conv = None
    conv_docs = []

    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    
    for doc in docs:
        doc_conv = doc.get("conversation_id")

        # When conversation changes, process the accumulated conv_docs
        if current_conv is not None and doc_conv != current_conv:
            # Count stories in the accumulated docs
            n = len(conv_docs)
            num_full = n // min_interactions
            if num_full:
                # Split into full min_interactions-item stories
                for i in range(num_full):
                    story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                    full_story_data.append(story_slice)
                    out_rows.extend(story_slice)
                count[current_conv] = count.get(current_conv, 0) + num_full
            # Reset for new conversation
            conv_docs = []
            current_conv = doc_conv
        else:
            # First doc of a new conversation
            if current_conv is None:
                current_conv = doc_conv

        # Build row dict for current doc and append to conv_docs
        doc_dictionary = doc.to_dict()
        row = {
            "timestamp": doc_dictionary.get("timestamp") if "timestamp" in doc_dictionary else None,
            "user": doc_dictionary.get("user") if "user" in doc_dictionary else None,
            "ai": doc_dictionary.get("ai") if "ai" in doc_dictionary else None,
            #"combined_prompt": doc_dictionary.get("combined_prompt") if "combined_prompt" in doc_dictionary else None,
            #"client_id": doc_dictionary.get("client_id") if "client_id" in doc_dictionary else None,
            #"workshop_id": doc_dictionary.get("workshop_id") if "workshop_id" in doc_dictionary else None,
            #"language": doc_dictionary.get("language") if "language" in doc_dictionary else None,
            "conversation_id": doc_conv,
            "respondent_id": doc_dictionary.get("respondent_id") if "respondent_id" in doc_dictionary else None,
            "interaction_count": doc_dictionary.get("interaction_count") if "interaction_count" in doc_dictionary else None,
            "llm_type": doc_dictionary.get("llm_type") if "llm_type" in doc_dictionary else None,
            
        }
        conv_docs.append(row)

    # Process the final conversation after the loop
    if current_conv is not None and conv_docs:
        n = len(conv_docs)
        num_full = n // min_interactions
        if num_full:
            for i in range(num_full):
                story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                full_story_data.append(story_slice)
                out_rows.extend(story_slice)
            count[current_conv] = count.get(current_conv, 0) + num_full

    total_stories = sum(count.values()) if count else 0
    print(f"Downloaded {len(out_rows)} interactions from {total_stories} complete stories")
    
    return pd.DataFrame(out_rows)

def delete_incomplete_stories_from_firestore(
    db: firestore.Client,
    story_data_collection: str = "story_data_TEXT",
    stories_collection: str = "stories_TEXT",
    min_interactions: int = 10
) -> int:
    """
    Delete stories from Firestore that have fewer than min_interactions.
    
    Args:
        db: Firestore client
        story_data_collection: Name of the story_data collection
        stories_collection: Name of the stories collection
        min_interactions: Minimum interactions required
        
    Returns:
        Number of stories deleted
    """
    story_data_ref = db.collection(story_data_collection)
    stories_ref = db.collection(stories_collection)
    
    counts = {}
    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    conversation_id = None
    deleted_count = 0
    
    for doc in docs:
        # Only consider deletion when we have a previous conversation_id
        if (conversation_id is not None and 
            conversation_id != doc.get("conversation_id") and 
            counts.get(conversation_id, 0) < min_interactions):
            print(f"Deleting story with conversation_id: {conversation_id}")
            story_query = stories_ref.where(
                FieldFilter("conversation_id", "==", conversation_id)
            )
            story_docs = story_query.stream()
            for story_doc in story_docs:
                print(f" - Deleting story document ID: {story_doc.id}")
                stories_ref.document(story_doc.id).delete()
                deleted_count += 1
            
        conversation_id = doc.get("conversation_id")
        if conversation_id is None:
            continue
        counts[conversation_id] = counts.get(conversation_id, 0) + 1

    # Final check for the last conversation after streaming all documents
    if (conversation_id is not None and 
        counts.get(conversation_id, 0) < min_interactions):
        print(f"Deleting story with conversation_id: {conversation_id}")
        story_query = stories_ref.where(
            FieldFilter("conversation_id", "==", conversation_id)
        )
        story_docs = story_query.stream()
        for story_doc in story_docs:
            print(f" - Deleting story document ID: {story_doc.id}")
            stories_ref.document(story_doc.id).delete()
            deleted_count += 1

    story_count = sum(cnt // min_interactions for cnt in counts.values())
    print(f"Number of complete stories remaining: {story_count}")
    print(f"Total story documents deleted: {deleted_count}")
    
    return deleted_count


def filter_by_edit_distance(
    df: pd.DataFrame,
    threshold: int,
    column: str = "edit_distance"
) -> pd.DataFrame:
    """
    Filter rows based on edit distance threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum edit distance to keep
        column: Name of the edit distance column
        
    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    n_before = len(df)
    filtered_df = df[df[column] <= threshold].copy()
    n_after = len(filtered_df)
    n_removed = n_before - n_after
    
    print(f"Edit distance filtering (threshold={threshold}):")
    print(f"  Before: {n_before} rows")
    print(f"  After: {n_after} rows")
    print(f"  Removed: {n_removed} rows ({100*n_removed/n_before:.1f}%)")
    
    return filtered_df

def append_turn_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append turn numbers within each conversation_id.
    
    Args:
        df: Input DataFrame with 'conversation_id' column
        
    Returns:
        DataFrame with new 'turn' column
    """
    df_out = df.copy()
    df_out['turn'] = df_out.groupby('conversation_id').cumcount() + 1
    return df_out


def filter_by_respondent_id(
    df: pd.DataFrame,
    threshold: int = 12,
    column: str = "respondent_id"
) -> pd.DataFrame:
    """
    Filter DataFrame by respondent_id length.
    
    Args:
        df: Input DataFrame
        column: Name of the respondent ID column
        threshold: Required length of respondent_id string
        
    Returns:
        Filtered DataFrame (keeping only rows where len(respondent_id) == threshold)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    n_before = len(df)
    filtered_df = df[df[column].astype(str).str.len() == threshold].copy()

    # removing remaining test_ids
    filtered_df = filtered_df[~filtered_df[column].str.startswith("test-")]

    n_after = len(filtered_df)
    n_removed = n_before - n_after
    
    print(f"Filtering by respondent_id length={threshold}:")
    print(f"  Before: {n_before} rows")
    print(f"  After: {n_after} rows")
    print(f"  Removed: {n_removed} rows ({100*n_removed/n_before:.1f}%)")
    
    return filtered_df

def build_full_story_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full_story, full_user, and full_ai columns by concatenating
    user/ai text within each conversation_id.
    
    Args:
        df: DataFrame with 'conversation_id', 'user', 'ai' columns
        
    Returns:
        DataFrame grouped by conversation_id with concatenated text columns
    """
    # Group by conversation and concatenate
    story_df = df.groupby('conversation_id').agg({
        'user': lambda x: ' '.join(x.astype(str)),
        'ai': lambda x: ' '.join(x.astype(str)),
        **({'language': 'first'} if 'language' in df.columns else {}),
        **({'client_id': 'first'} if 'client_id' in df.columns else {}),
        **({'workshop_id': 'first'} if 'workshop_id' in df.columns else {}),
        'timestamp': 'first',
        'respondent_id': 'first',
        'starter': 'first',
        'llm_type': 'first'
    }).reset_index()
    
    # Rename columns
    story_df = story_df.rename(columns={
        'user': 'full_user',
        'ai': 'full_ai',
    })
    
    # Build full_story with USER: and AI: markers
    def build_story(row):
        users = df[df['conversation_id'] == row['conversation_id']]['user'].tolist()
        ais = df[df['conversation_id'] == row['conversation_id']]['ai'].tolist()
        parts = []
        for u, a in zip(users, ais):
            parts.append(f"{u}")
            parts.append(f"{a}")
        return ' '.join(parts)
    
    story_df['full_story'] = story_df.apply(build_story, axis=1)
    
    print(f"Built full story text for {len(story_df)} stories")
    
    return story_df

def clean_user_ai_start(df: pd.DataFrame, interaction_count: bool = True) -> pd.DataFrame: 

    # adds ai starter for that respondent_id if turn 10 ai-text == None  
    if interaction_count == False:
        df['turn'] = df.groupby('respondent_id').cumcount() + 1
        starter_map = df.loc[df['turn'] == 10].set_index('respondent_id')['ai'].isna().map(
            {True: 'ai', False: 'user'}
        )
        df['starter'] = df['respondent_id'].map(starter_map)
    
    else: # called interaction_count instead
        starter_map = df.loc[df['interaction_count'] == 10].set_index('respondent_id')['ai'].isna().map(
        {True: 'ai', False: 'user'}
        )
        df['starter'] = df['respondent_id'].map(starter_map)

    for rid, group in df.groupby('respondent_id'):
        # Only modify groups where starter == 'ai'
        if group['starter'].iloc[0] != 'ai':
            continue
        
        # Identify the first user row and first ai row
        first_user_idx = group[group['user'].notna()].index.min()
        first_ai_idx   = group[group['ai'].notna()].index.min()
        
        # If either is missing, skip
        if pd.isna(first_user_idx) or pd.isna(first_ai_idx):
            continue
        
        # Prepend: user_text + " " + ai_text
        user_text = df.loc[first_user_idx, 'user']
        ai_text   = df.loc[first_ai_idx, 'ai']
        
        df.loc[first_ai_idx, 'ai'] = f"{user_text} {ai_text}"
        # df.loc[first_user_idx, 'user'] = np.nan
        df.loc[first_user_idx, 'user'] = ""

    return df 
