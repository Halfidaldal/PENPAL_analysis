"""
Data cleaning and filtering functions.

This module handles:
- Firestore data download and filtering (both human-ai and human-human schemas)
- Edit distance filtering
- Spell-checking and rectification
- Story grouping (10 interactions = 1 story)
- Schema normalization (author_1/author_2 with condition column)
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


def normalize_columns(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """
    Normalize column names to unified schema: author_1, author_2, condition.
    
    This should be called after loading/cleaning data to ensure consistent
    column names across experiments for comparative analysis.
    
    Args:
        df: Input DataFrame with experiment-specific column names
        experiment: Experiment name ('human-ai' or 'human-human')
        
    Returns:
        DataFrame with normalized columns
    """
    df = df.copy()
    
    # Add condition column
    df['condition'] = experiment
    
    # Rename author columns based on experiment
    if experiment == 'human-ai':
        # user -> author_1, ai -> author_2
        if 'user' in df.columns:
            df = df.rename(columns={'user': 'author_1'})
        if 'ai' in df.columns:
            df = df.rename(columns={'ai': 'author_2'})
        # Also handle full_user, full_ai if present
        if 'full_user' in df.columns:
            df = df.rename(columns={'full_user': 'full_author_1'})
        if 'full_ai' in df.columns:
            df = df.rename(columns={'full_ai': 'full_author_2'})
        if 'full_user_dot' in df.columns:
            df = df.rename(columns={'full_user_dot': 'full_author_1_dot'})
        if 'full_ai_dot' in df.columns:
            df = df.rename(columns={'full_ai_dot': 'full_author_2_dot'})
            
    elif experiment == 'human-human':
        # user -> author_1, user2 -> author_2
        if 'user' in df.columns:
            df = df.rename(columns={'user': 'author_1'})
        if 'user2' in df.columns:
            df = df.rename(columns={'user2': 'author_2'})
        # Also handle full_user, full_ai (used in human-human despite the name)
        if 'full_user' in df.columns:
            df = df.rename(columns={'full_user': 'full_author_1'})
        if 'full_ai' in df.columns:
            df = df.rename(columns={'full_ai': 'full_author_2'})
        if 'full_user_dot' in df.columns:
            df = df.rename(columns={'full_user_dot': 'full_author_1_dot'})
        if 'full_ai_dot' in df.columns:
            df = df.rename(columns={'full_ai_dot': 'full_author_2_dot'})
            
    elif experiment == 'ai-ai':
        # agent_1 -> author_1, agent_2 -> author_2
        if 'agent_1' in df.columns:
            df = df.rename(columns={'agent_1': 'author_1'})
        if 'agent_2' in df.columns:
            df = df.rename(columns={'agent_2': 'author_2'})
        # Handle full text columns if present
        if 'full_agent_1' in df.columns:
            df = df.rename(columns={'full_agent_1': 'full_author_1'})
        if 'full_agent_2' in df.columns:
            df = df.rename(columns={'full_agent_2': 'full_author_2'})
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    print(f"Normalized columns for {experiment} experiment")
    return df


def download_stories_from_firestore(
    db: firestore.Client,
    collection_name: str = "story_data_TEXT",
    min_interactions: int = 10,
    schema: str = "flat"
) -> pd.DataFrame:
    """
    Download stories from Firestore, keeping only complete stories.
    
    Supports two Firestore schemas:
    - "flat": Human-AI data (all interactions in single collection)
    - "nested": Human-Human data (sessions with nested turns subcollection)
    
    Args:
        db: Firestore client
        collection_name: Name of the Firestore collection
        min_interactions: Minimum number of interactions to count as a story
        schema: Firestore schema type ("flat" or "nested")
        
    Returns:
        DataFrame with story interaction rows (only complete stories)
    """
    if schema == "flat":
        return _download_flat_schema(db, collection_name, min_interactions)
    elif schema == "nested":
        return _download_nested_schema(db, collection_name, min_interactions)
    else:
        raise ValueError(f"Unknown schema: {schema}. Must be 'flat' or 'nested'")


def _download_flat_schema(
    db: firestore.Client,
    collection_name: str,
    min_interactions: int
) -> pd.DataFrame:
    """Download from flat collection schema (human-ai)."""
    story_data_ref = db.collection(collection_name)
    
    count = {}
    out_rows = []
    full_story_data = []
    current_conv = None
    conv_docs = []

    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    
    for doc in docs:
        doc_conv = doc.get("conversation_id")

        # When conversation changes, process the accumulated conv_docs
        if current_conv is not None and doc_conv != current_conv:
            n = len(conv_docs)
            num_full = n // min_interactions
            if num_full:
                for i in range(num_full):
                    story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                    full_story_data.append(story_slice)
                    out_rows.extend(story_slice)
                count[current_conv] = count.get(current_conv, 0) + num_full
            conv_docs = []
            current_conv = doc_conv
        else:
            if current_conv is None:
                current_conv = doc_conv

        doc_dictionary = doc.to_dict()
        row = {
            "timestamp": doc_dictionary.get("timestamp"),
            "user": doc_dictionary.get("user"),
            "ai": doc_dictionary.get("ai"),
            "conversation_id": doc_conv,
            "respondent_id": doc_dictionary.get("respondent_id"),
            "interaction_count": doc_dictionary.get("interaction_count"),
            "llm_type": doc_dictionary.get("llm_type"),
        }
        conv_docs.append(row)

    # Process the final conversation
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
    print(f"Downloaded {len(out_rows)} interactions from {total_stories} complete stories (flat schema)")
    
    return pd.DataFrame(out_rows)


def _download_nested_schema(
    db: firestore.Client,
    collection_name: str,
    min_interactions: int
) -> pd.DataFrame:
    """Download from nested collection schema (human-human)."""
    story_data_ref = db.collection(collection_name)
    
    count = {}
    out_rows = []

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

        # Skip if odd number of turns or incorrect participants
        if len(turn_dicts) % 2 != 0 or len(story_meta.get("participants", [])) != 2:
            print(f"Skipping conversation {doc_conv} due to odd number of turns or incorrect participants")
            continue

        conv_docs = []
        for i in range(0, len(turn_dicts) - 1, 2):
            t1 = turn_dicts[i]
            t2 = turn_dicts[i + 1]

            row = {
                "timestamp": t2.get("timestamp") or t1.get("timestamp"),
                "user": t1.get("text"),
                "user2": t2.get("text"),
                "conversation_id": doc_conv,
                "respondent_id_u1": t1.get("userId"),
                "respondent_id_u2": t2.get("userId"),
                "interaction_count": (i // 2) + 1,
            }
            conv_docs.append(row)

        # Keep the entire conversation if it meets the threshold
        if conv_docs and len(conv_docs) >= min_interactions:
            out_rows.extend(conv_docs)
            count[doc_conv] = 1

    total_stories = sum(count.values()) if count else 0
    print(f"Downloaded {len(out_rows)} interactions from {total_stories} complete stories (nested schema)")
        
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

def build_full_story_text(df: pd.DataFrame, experiment: str = "human-ai") -> pd.DataFrame:
    """
    Build full_story, full_author_1, and full_author_2 columns by concatenating
    text within each conversation_id or story_id.
    
    Args:
        df: DataFrame with grouping column and author columns
        experiment: Experiment type ('human-ai', 'human-human', or 'ai-ai')
        
    Returns:
        DataFrame grouped by conversation/story with concatenated text columns
    """
    # Determine column names based on experiment
    if experiment == 'ai-ai':
        author_1_col = 'agent_1'
        author_2_col = 'agent_2'
        group_col = 'story_id'
    else:
        author_1_col = 'user'
        author_2_col = 'ai' if experiment == 'human-ai' else 'user2'
        group_col = 'conversation_id'
        
        if author_2_col not in df.columns:
            # Fallback: try the other column
            author_2_col = 'user2' if 'user2' in df.columns else 'ai'
    
    if author_1_col not in df.columns:
        raise ValueError(f"Expected '{author_1_col}' column in DataFrame")
    if author_2_col not in df.columns:
        raise ValueError(f"Expected '{author_2_col}' column in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    
    # Group by conversation/story and concatenate
    agg_dict = {
        author_1_col: lambda x: ' '.join(x.astype(str)),
        author_2_col: lambda x: ' '.join(x.astype(str)),
    }
    # Add metadata columns if present
    for col in ['language', 'client_id', 'workshop_id', 'timestamp', 
                'respondent_id', 'respondent_id_u1', 'respondent_id_u2',
                'interaction_count', 'starter', 'llm_type', 'model_id', 'turn']:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    story_df = df.groupby(group_col).agg(agg_dict).reset_index()
    
    # Padded versions for parsing textdescriptives
    story_df['full_user_dot'] = df.groupby(group_col)[author_1_col].apply(
        lambda x: ' '.join((x.astype(str) + '.').tolist())).values
    story_df['full_ai_dot'] = df.groupby(group_col)[author_2_col].apply(
        lambda x: ' '.join((x.astype(str) + '.').tolist())).values

    # Rename columns to unified names
    story_df = story_df.rename(columns={
        author_1_col: 'full_user',
        author_2_col: 'full_ai',
    })
    
    # Also rename group column to conversation_id for consistency
    if group_col != 'conversation_id':
        story_df = story_df.rename(columns={group_col: 'conversation_id'})
    
    # Build full_story by interleaving
    def build_story(row):
        group_val = row['conversation_id']
        # Handle both original column names
        if group_col in df.columns:
            mask = df[group_col] == group_val
        else:
            mask = df['conversation_id'] == group_val
        
        author1s = df[mask][author_1_col].tolist()
        author2s = df[mask][author_2_col].tolist()
        parts = []
        for u, a in zip(author1s, author2s):
            parts.append(f"{u}")
            parts.append(f"{a}")
        return ' '.join(parts)
    
    story_df['full_story'] = story_df.apply(build_story, axis=1)
    
    print(f"Built full story text for {len(story_df)} stories")
    
    return story_df


def clean_user_ai_start(df: pd.DataFrame, interaction_count: bool = True, max_turns: int = 10, experiment: str = "human-ai") -> pd.DataFrame: 
    """
    Clean starter text and identify which author started.
    
    Args:
        df: Input DataFrame
        interaction_count: Whether to filter by interaction_count column
        max_turns: Maximum turns to keep
        experiment: Experiment type ('human-ai' or 'human-human')
        
    Returns:
        Cleaned DataFrame with 'starter' column
    """
    # Determine column names based on experiment
    if experiment == 'human-ai':
        author_2_col = 'ai'
        respondent_col = 'respondent_id'
        author_2_starter_label = 'ai'
    else:
        author_2_col = 'user2'
        respondent_col = 'respondent_id_u1' if 'respondent_id_u1' in df.columns else 'respondent_id'
        author_2_starter_label = 'user2'
    
    # Identify starters
    if respondent_col in df.columns:
        df['turn'] = df.groupby(respondent_col).cumcount() + 1
        starter_map = (df['user'] == 'This is the story of').groupby(df[respondent_col]).any().map(
            {True: author_2_starter_label, False: 'user'}
        )
        df['starter'] = df[respondent_col].map(starter_map)
        print(f"Identified starters for {len(df[df['starter'] == author_2_starter_label][respondent_col].unique())} {author_2_starter_label}")
        print(f"Identified starters for {len(df[df['starter'] == 'user'][respondent_col].unique())} user")
    
    # Filter by max_turns
    if interaction_count:
        df = df[df['interaction_count'] <= max_turns].copy()
    else:
        df = df[df['turn'] <= max_turns].copy()
    
    # Remove baseline text
    df['user'] = df['user'].str.replace("This is the story of", "", regex=False).str.strip()
    df[author_2_col] = df[author_2_col].str.replace("This is the story of", "", regex=False).str.strip()

    # Handle starter adjustments
    for rid, group in df.groupby(respondent_col):
        if group['starter'].iloc[0] != author_2_starter_label:
            continue
        
        first_user_idx = group[group['user'].notna()].index.min()
        first_author2_idx = group[group[author_2_col].notna()].index.min()
        last_author2_idx = group[group[author_2_col].isna()].index
        
        if pd.isna(first_user_idx) or pd.isna(first_author2_idx):
            continue
        
        user_text = df.loc[first_user_idx, 'user']
        author2_text = df.loc[first_author2_idx, author_2_col]
        
        df.loc[first_author2_idx, author_2_col] = f"{user_text} {author2_text}"
        df.loc[first_user_idx, 'user'] = ""
        df.loc[last_author2_idx, author_2_col] = ""

    return df 


def clean_ai_ai_data(df: pd.DataFrame, max_turns: int = 10) -> pd.DataFrame:
    """
    Clean AI-AI simulated data.
    
    Handles:
    - Removing "This is the story of" prefix from first turn
    - Filtering to max_turns
    - Adding metadata columns for compatibility with other conditions
    
    Args:
        df: Raw AI-AI simulation data with columns:
            turn, agent_1, agent_2, story_id, model_id, timestamp
        max_turns: Maximum turns per story (default 10)
        
    Returns:
        Cleaned DataFrame with standardized structure
    """
    df = df.copy()
    
    # Story prefix used in simulation
    STORY_PREFIX = "This is the story of"
    
    print(f"Cleaning AI-AI data: {len(df)} rows, {df['story_id'].nunique()} stories")
    
    # Remove "This is the story of" prefix from agent_1's first turn
    # The prefix is prepended with \n in simulation, so handle both cases
    df['agent_1'] = df['agent_1'].str.replace(f"{STORY_PREFIX}\n", "", regex=False)
    df['agent_1'] = df['agent_1'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Also clean agent_2 just in case (shouldn't have it, but for safety)
    df['agent_2'] = df['agent_2'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Filter to max_turns
    if 'turn' in df.columns:
        df = df[df['turn'] <= max_turns].copy()
    
    # Add starter column (agent_1 always starts in AI-AI)
    df['starter'] = 'agent_1'
    
    # Add interaction_count column for compatibility
    df['interaction_count'] = df['turn']
    
    # Count prefix removals
    n_prefixes_removed = len(df[df['turn'] == 1])
    print(f"Removed '{STORY_PREFIX}' prefix from {n_prefixes_removed} first turns")
    print(f"After cleaning: {len(df)} rows")
    
    return df
