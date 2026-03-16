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
    collection_name: str = "sessions_TEXT",
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
    conv_docs = []

    docs = story_data_ref.stream()
    for story_doc in docs:
        story_meta = story_doc.to_dict() or {}
        doc_conv = story_doc.id
        participants = story_meta.get("participants", [])
        if not isinstance(participants, list) or len(participants) != 2:
            print(f"Skipping conversation {doc_conv} due to invalid participants field")
            continue

        turn_docs = list(
            story_doc.reference.collection("turns").order_by("timestamp").stream()
        )
        turn_dicts = [t.to_dict() or {} for t in turn_docs]

        n_turns = len(turn_dicts)
        n_interactions = n_turns // 2
        print(f"{doc_conv}: turns={n_turns}, interactions={n_interactions}, needed={min_interactions}")

        # If odd number of turns, last one is incomplete interaction
        if len(turn_dicts) % 2 != 0 or len(story_meta.get("participants", [])) != 2:
            #drop the story if it has an odd number of turns or if it doesn't have exactly 2 participants
            print(f"Skipping conversation {doc_conv} due to odd number of turns or incorrect participants")
            continue

        conv_docs = []
        for i in range(0, len(turn_dicts) - 1, 2):
            t1 = turn_dicts[i]
            t2 = turn_dicts[i + 1]

            row = {
                "timestamp": t2.get("timestamp") or t1.get("timestamp"),
                "user": t1.get("text"),
                "user2":t2.get("text"),
                "conversation_id": doc_conv,
                "respondent_id_u1": t1.get("userId"),
                "respondent_id_u2": t2.get("userId"),
                "interaction_count": (i // 2) + 1,
            }
            conv_docs.append(row)

        # Process the final conversation after the loop
        if conv_docs:
            n = len(conv_docs)
            num_full = n // min_interactions
            if num_full:
                for i in range(num_full):
                    story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                    out_rows.extend(story_slice)
                count[doc_conv] = count.get(doc_conv, 0) + num_full

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
        df: DataFrame with 'conversation_id', 'user', 'user2' columns
        
    Returns:
        DataFrame grouped by conversation_id with concatenated text columns
    """
    ai_col = 'user2' if 'user2' in df.columns else 'ai'
    if ai_col not in df.columns:
        raise ValueError("Expected either 'user2' or 'ai' column in DataFrame")

    # Group by conversation and concatenate
    story_df = df.groupby('conversation_id').agg({
        'user': lambda x: ' '.join(x.astype(str)),
        ai_col: lambda x: ' '.join(x.astype(str)),
        # metadata
        **({'language': 'first'} if 'language' in df.columns else {}),
        **({'client_id': 'first'} if 'client_id' in df.columns else {}),
        **({'workshop_id': 'first'} if 'workshop_id' in df.columns else {}),
        **({'timestamp': 'first'} if 'timestamp' in df.columns else {}),
        **({'respondent_id_u1': 'first'} if 'respondent_id_u1' in df.columns else {}),
        **({'respondent_id_u2': 'first'} if 'respondent_id_u2' in df.columns else {}),
        **({'respondent_id': 'first'} if 'respondent_id' in df.columns else {}),
        **({'interaction_count': 'first'} if 'interaction_count' in df.columns else {}),
        **({'starter': 'first'} if 'starter' in df.columns else {}),
        **({'llm_type': 'first'} if 'llm_type' in df.columns else {})
    }).reset_index()


    # padded versions for parsing textdescriptives
    story_df['full_user_dot'] = df.groupby('conversation_id')['user'].apply(
        lambda x: ' '.join((x.astype(str) + '.').tolist())).values
    story_df['full_ai_dot'] = df.groupby('conversation_id')[ai_col].apply(
        lambda x: ' '.join((x.astype(str) + '.').tolist())).values

    # Rename columns
    story_df = story_df.rename(columns={
        'user': 'full_user',
        ai_col: 'full_ai',
    })
    
    # Build full_story with USER: and AI: markers
    def build_story(row):
        users = df[df['conversation_id'] == row['conversation_id']]['user'].tolist()
        ais = df[df['conversation_id'] == row['conversation_id']][ai_col].tolist()
        parts = []
        for u, a in zip(users, ais):
            parts.append(f"{u}")
            parts.append(f"{a}")
        return ' '.join(parts)
    
    story_df['full_story'] = story_df.apply(build_story, axis=1)
    
    print(f"Built full story text for {len(story_df)} stories")
    
    return story_df

def clean_user_ai_start(df: pd.DataFrame, interaction_count: bool = True, max_turns: int = 10) -> pd.DataFrame: 

    # adds starter label (user or user2) per respondent_id_u1 based on baseline text

    if 'respondent_id_u1' in df.columns:
        df['turn'] = df.groupby('respondent_id_u1').cumcount() + 1
        starter_map = (df['user'] == 'This is the story of').groupby(df['respondent_id_u1']).any().map({True: 'user2', False: 'user'})
        df['starter'] = df['respondent_id_u1'].map(starter_map)
        print(f"Identified starters for {len(df[df['starter'] == 'user2']['respondent_id_u1'].unique())} user2")
        print(f"Identified starters for {len(df[df['starter'] == 'user']['respondent_id_u1'].unique())} user")
    #sort by max_turns:
    if interaction_count:
        df = df[df['interaction_count'] <= max_turns].copy()
    else:
        df = df[df['turn'] <= max_turns].copy()
        
    #remove all the baseline text from the beginning of the story if it exists (This is the story of)
    df['user'] = df['user'].str.replace("This is the story of", "", regex=False).str.strip()
    df['user2'] = df['user2'].str.replace("This is the story of", "", regex=False).str.strip()


    for rid, group in df.groupby('respondent_id_u1'):
        # Only modify groups where starter == 'user2'
        if group['starter'].iloc[0] != 'user2':
            continue
        
        # Identify the first user row and first user2 row
        first_user_idx = group[group['user'].notna()].index.min()
        first_user2_idx   = group[group['user2'].notna()].index.min()
        last_user2_idx    = group[group['user2'].isna()].index        # should be the only one 
        
        # If either is missing, skip
        if pd.isna(first_user_idx) or pd.isna(first_user2_idx):
            continue
        
        # Prepend: user_text + " " + user2_text
        user_text = df.loc[first_user_idx, 'user']
        user2_text   = df.loc[first_user2_idx, 'user2']
        
        df.loc[first_user2_idx, 'user2'] = f"{user_text} {user2_text}"
        # df.loc[first_user_idx, 'user'] = np.nan
        df.loc[first_user_idx, 'user'] = ""
        df.loc[last_user2_idx, 'user2'] = ""                # change from NaN value to empty string in missing user2 turn

    return df
