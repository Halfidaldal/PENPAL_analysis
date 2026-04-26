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
from uuid import uuid4
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
        # author_1 -> author_1, author_2 -> author_2
        if 'author_1' in df.columns:
            df = df.rename(columns={'author_1': 'author_1'})
        if 'author_2' in df.columns:
            df = df.rename(columns={'author_2': 'author_2'})
        # Handle full text columns if present
        if 'full_author_1' in df.columns:
            df = df.rename(columns={'full_author_1': 'full_author_1'})
        if 'full_author_2' in df.columns:
            df = df.rename(columns={'full_author_2': 'full_author_2'})
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    print(f"Normalized columns for {experiment} experiment")
    return df


def download_stories_from_firestore(
    db: firestore.Client,
    collection_name: str = "story_data_TEXT",
    min_interactions: int = 9,
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
    text within each conversation_id.
    
    Uses standardized column names: author_1, author_2, conversation_id.
    
    Args:
        df: DataFrame with author_1, author_2, conversation_id columns
        experiment: Experiment type (unused after standardization, kept for API compatibility)
        
    Returns:
        DataFrame grouped by conversation_id with concatenated text columns
    """
    group_col = 'conversation_id'
    
    # Validate columns
    if 'author_1' not in df.columns:
        raise ValueError("Expected 'author_1' column in DataFrame")
    if 'author_2' not in df.columns:
        raise ValueError("Expected 'author_2' column in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    
    def _normalize_story_text(value: Any) -> str:
        """Collapse embedded newlines so story text is built as a single line."""
        if pd.isna(value):
            return ''
        return ' '.join(str(value).split())

    def _concat_series(series: pd.Series) -> str:
        return ' '.join(
            text for text in series.map(_normalize_story_text).tolist() if text
        )

    def _concat_series_with_period(series: pd.Series) -> str:
        return ' '.join(
            f"{text}." for text in series.map(_normalize_story_text).tolist() if text
        )

    # Group by conversation and concatenate
    agg_dict = {
        'author_1': _concat_series,
        'author_2': _concat_series,
    }
    # Add metadata columns if present
    for col in ['condition', 'language', 'client_id', 'workshop_id', 'timestamp', 
                'respondent_id', 'respondent_id_u1', 'respondent_id_u2',
                'interaction_count', 'starter', 'starter_side', 'starter_type',
                'author_1_type', 'author_2_type', 'llm_type', 'model_id', 'turn']:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    story_df = df.groupby(group_col).agg(agg_dict).reset_index()
    
    # Padded versions for parsing textdescriptives
    story_df['full_author_1_dot'] = df.groupby(group_col)['author_1'].apply(
        _concat_series_with_period
    ).values
    story_df['full_author_2_dot'] = df.groupby(group_col)['author_2'].apply(
        _concat_series_with_period
    ).values

    # Rename columns to full_ prefix
    story_df = story_df.rename(columns={
        'author_1': 'full_author_1',
        'author_2': 'full_author_2',
    })
    
    # Build full_story by interleaving
    def build_story(row):
        group_val = row['conversation_id']
        mask = df['conversation_id'] == group_val
        starter = row.get('starter', 'author_1')

        author1s = df[mask]['author_1'].map(_normalize_story_text).tolist()
        author2s = df[mask]['author_2'].map(_normalize_story_text).tolist()
        parts = []

        if starter == 'author_2':
            ordered_pairs = zip(author2s, author1s)
        else:
            ordered_pairs = zip(author1s, author2s)

        for first, second in ordered_pairs:
            if first:
                parts.append(first)
            if second:
                parts.append(second)
        return ' '.join(parts)
    
    story_df['full_story'] = story_df.apply(build_story, axis=1)
    
    print(f"Built full story text for {len(story_df)} stories")
    
    return story_df


def clean_user_ai_start(df: pd.DataFrame, interaction_count: bool = True, max_turns: int = 10, experiment: str = "human-ai") -> pd.DataFrame: 
    """
    Clean starter text and identify which author started.
    
    For human-ai experiment:
    - If author_1's first turn is ONLY "This is the story of" (no additional content),
      then author_2 (AI) started the story
    - If author_1's first turn has "This is the story of" + additional content,
      then author_1 (human) started the story
    
    Uses standardized column names: author_1, author_2.
    
    Args:
        df: Input DataFrame with author_1, author_2 columns
        interaction_count: Whether to filter by interaction_count column
        max_turns: Maximum turns to keep
        experiment: Experiment type ('human-ai' or 'human-human')
        
    Returns:
        Cleaned DataFrame with 'starter' column
    """
    BASELINE = "This is the story of"
    MIN_CONTENT_LENGTH = 10  # Minimum chars beyond baseline to count as "started"
    
    df = df.copy()
    
    # Story-level logic must use the conversation identifier.
    # Participant identifiers can be missing or reused and are therefore not
    # reliable story boundaries for starter detection or turn numbering.
    group_col = 'conversation_id'
    
    # Add turn numbers within each group
    df['turn'] = df.groupby(group_col).cumcount() + 1
    
    # Identify starters based on first turn content
    def detect_starter(group):
        """Detect who started based on first turn's author_1 content."""
        first_row = group[group['turn'] == 1]
        if first_row.empty:
            return 'author_1'  # Default
        
        author_1_text = str(first_row['author_1'].iloc[0]) if pd.notna(first_row['author_1'].iloc[0]) else ''
        
        # Remove baseline and check remaining content
        remainder = author_1_text.replace(BASELINE, '').strip()
        
        # If author_1 only had the baseline placeholder (no real content), author_2 started
        if len(remainder) < MIN_CONTENT_LENGTH:
            return 'author_2'
        else:
            return 'author_1'
    
    # Build starter map
    starter_map = df.groupby(group_col, group_keys=False).apply(detect_starter, include_groups=False)
    df['starter'] = df[group_col].map(starter_map)
    
    # Report starter distribution
    author_1_started = (starter_map == 'author_1').sum()
    author_2_started = (starter_map == 'author_2').sum()
    print(f"Starter detection: {author_1_started} author_1-started, {author_2_started} author_2-started")
    
    # Filter by max_turns
    if interaction_count and 'interaction_count' in df.columns:
        df = df[df['interaction_count'] <= max_turns].copy()
    else:
        df = df[df['turn'] <= max_turns].copy()
    
    # Remove baseline text from both columns
    df['author_1'] = df['author_1'].str.replace(BASELINE, "", regex=False).str.strip()
    df['author_2'] = df['author_2'].str.replace(BASELINE, "", regex=False).str.strip()

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
            turn, author_1, author_2, story_id, model_id, timestamp
        max_turns: Maximum turns per story (default 10)
        
    Returns:
        Cleaned DataFrame with standardized structure
    """
    df = df.copy()
    
    # Story prefix used in simulation
    STORY_PREFIX = "This is the story of"
    
    print(f"Cleaning AI-AI data: {len(df)} rows, {df['story_id'].nunique()} stories")
    
    # Remove "This is the story of" prefix from author_1's first turn
    # The prefix is prepended with \n in simulation, so handle both cases
    df['author_1'] = df['author_1'].str.replace(f"{STORY_PREFIX}\n", "", regex=False)
    df['author_1'] = df['author_1'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Also clean author_2 just in case (shouldn't have it, but for safety)
    df['author_2'] = df['author_2'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Filter to max_turns
    if 'turn' in df.columns:
        df = df[df['turn'] <= max_turns].copy()
    
    # Replace model-revealing story IDs with opaque conversation IDs.
    story_ids = df['story_id'].dropna().unique()
    conversation_id_map = {
        story_id: f"conv_{uuid4().hex}"
        for story_id in story_ids
    }
    df['conversation_id'] = df['story_id'].map(conversation_id_map)
    
    # Add starter column (author_1 always starts in AI-AI)
    df['starter'] = 'author_1'
    
    # Add interaction_count column for compatibility
    df['interaction_count'] = df['turn']
    
    # Count prefix removals
    n_prefixes_removed = len(df[df['turn'] == 1])
    print(f"Removed '{STORY_PREFIX}' prefix from {n_prefixes_removed} first turns")
    print(f"Generated opaque conversation IDs for {len(conversation_id_map)} stories")
    print(f"After cleaning: {len(df)} rows")
    
    return df


def _is_substantive_text(
    series: pd.Series,
    min_substantive_chars: int = 2
) -> pd.Series:
    """
    Identify rows with substantive text after removing whitespace and punctuation.

    A minimum of two remaining word characters treats ".", "," or other
    punctuation-only cells as empty while still counting very short real content.
    """
    normalized = (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.replace(r'[\W_]+', '', regex=True)
    )
    return normalized.str.len() >= min_substantive_chars


def add_exchange_aligned_metadata(
    df: pd.DataFrame,
    experiment: Optional[str] = None,
    group_col: str = 'conversation_id',
    min_substantive_chars: int = 2
) -> pd.DataFrame:
    """
    Add analysis-ready exchange metadata without changing stored author slots.

    Added columns:
    - author_1_type / author_2_type
    - complete_exchange
    - analysis_turn
    - starter_side
    - starter_type
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    if 'starter' not in df.columns:
        raise ValueError("Expected 'starter' column in DataFrame")

    df = df.copy()

    if 'condition' not in df.columns:
        if experiment is None:
            raise ValueError("Either provide 'experiment' or include a 'condition' column")
        df['condition'] = experiment

    if 'turn' not in df.columns:
        df['turn'] = df.groupby(group_col).cumcount() + 1

    author_1_type_map = {
        'human-ai': 'human',
        'human-human': 'human',
        'ai-ai': 'ai',
    }
    author_2_type_map = {
        'human-ai': 'ai',
        'human-human': 'human',
        'ai-ai': 'ai',
    }

    unknown_conditions = sorted(set(df['condition'].dropna().unique()) - set(author_1_type_map))
    if unknown_conditions:
        raise ValueError(f"Unknown condition(s): {unknown_conditions}")

    df['author_1_type'] = df['condition'].map(author_1_type_map)
    df['author_2_type'] = df['condition'].map(author_2_type_map)

    df['starter_side'] = df['starter']
    invalid_starters = sorted(set(df['starter_side'].dropna().unique()) - {'author_1', 'author_2'})
    if invalid_starters:
        raise ValueError(f"Invalid starter values: {invalid_starters}")

    df['starter_type'] = np.where(
        df['starter_side'].eq('author_1'),
        df['author_1_type'],
        df['author_2_type']
    )

    author_1_substantive = _is_substantive_text(df['author_1'], min_substantive_chars=min_substantive_chars)
    author_2_substantive = _is_substantive_text(df['author_2'], min_substantive_chars=min_substantive_chars)
    df['complete_exchange'] = author_1_substantive & author_2_substantive

    ordered = df.sort_values([group_col, 'turn'], kind='stable').copy()
    ordered['analysis_turn'] = ordered.groupby(group_col)['complete_exchange'].cumsum()
    ordered['analysis_turn'] = (
        ordered['analysis_turn']
        .where(ordered['complete_exchange'], pd.NA)
        .astype('Int64')
    )
    df['analysis_turn'] = ordered.sort_index()['analysis_turn']

    complete_count = int(df['complete_exchange'].sum())
    incomplete_count = int((~df['complete_exchange']).sum())
    print(
        "Exchange alignment metadata added: "
        f"{complete_count} complete exchanges, {incomplete_count} incomplete rows"
    )

    return df


def build_long_format_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dyadic interaction rows into a long-format analysis export.

    Each interaction row becomes two contribution rows, one for each author slot.
    """
    required = [
        'conversation_id', 'condition', 'turn', 'analysis_turn',
        'author_1', 'author_2', 'author_1_type', 'author_2_type',
        'complete_exchange', 'starter_side', 'starter_type'
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for long-format export: {missing}")

    shared_columns = [
        'conversation_id', 'condition', 'turn', 'analysis_turn',
        'complete_exchange', 'starter_side', 'starter_type'
    ]

    frames = []
    slot_specs = [
        ('author_1', 'author_1_type', 'author_2_type'),
        ('author_2', 'author_2_type', 'author_1_type'),
    ]

    for speaker_slot, speaker_type_col, partner_type_col in slot_specs:
        frame = df[shared_columns].copy()
        frame['speaker_slot'] = speaker_slot
        frame['speaker_type'] = df[speaker_type_col]
        frame['speaker_is_starter'] = df['starter_side'].eq(speaker_slot)
        frame['partner_type'] = df[partner_type_col]
        frame['text'] = df[speaker_slot]
        frames.append(frame)

    df_long = pd.concat(frames, ignore_index=True)
    df_long = df_long.sort_values(
        ['conversation_id', 'turn', 'speaker_slot'],
        kind='stable'
    ).reset_index(drop=True)

    print(f"Built long-format analysis export with {len(df_long)} rows")

    return df_long


def keep_complete_conversations(
    df: pd.DataFrame,
    group_col: str = 'conversation_id',
    expected_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Remove partial conversation fragments after row-level filtering.

    By default, the expected conversation length is inferred as the modal number
    of rows per conversation in the current DataFrame. This works well when a
    small number of corrupted or partially filtered stories remain alongside a
    dominant complete-story length.

    Args:
        df: Input DataFrame with repeated interaction rows per conversation
        group_col: Story identifier column
        expected_length: Required number of rows per conversation. If None,
            infer from the modal conversation length.

    Returns:
        DataFrame containing only complete conversations
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")

    df = df.copy()
    conversation_sizes = df.groupby(group_col).size()

    if conversation_sizes.empty:
        return df

    if expected_length is None:
        expected_length = int(conversation_sizes.mode().iloc[0])

    complete_ids = conversation_sizes[conversation_sizes == expected_length].index
    removed = conversation_sizes[conversation_sizes != expected_length]

    if len(removed) > 0:
        print(
            f"Dropping {len(removed)} incomplete conversation(s) with sizes "
            f"{removed.value_counts().sort_index().to_dict()} (expected {expected_length} rows)"
        )

    return df[df[group_col].isin(complete_ids)].copy()


def randomize_author_assignment(
    df: pd.DataFrame,
    group_col: str = 'conversation_id',
    seed: int = 42,
    swap_probability: float = 0.5
) -> pd.DataFrame:
    """
    Randomly swap author_1 and author_2 for each story to remove starter confound.
    
    In conditions where author_1 always starts (human-human, ai-ai), this introduces
    randomness matching the human-ai condition where starter was randomly assigned.
    
    When swapped:
    - author_1 <-> author_2 columns are swapped
    - starter column is updated to reflect who now has turn 1 content
    
    Args:
        df: DataFrame with author_1, author_2, conversation_id, starter columns
        group_col: Column to group stories by
        seed: Random seed for reproducibility
        swap_probability: Probability of swapping each story (default 0.5)
        
    Returns:
        DataFrame with randomized author assignment
    """
    import numpy as np
    
    df = df.copy()
    rng = np.random.default_rng(seed)
    
    # Get unique stories
    story_ids = df[group_col].unique()
    
    # Decide which stories to swap
    swap_mask = rng.random(len(story_ids)) < swap_probability
    stories_to_swap = set(story_ids[swap_mask])
    
    n_swapped = len(stories_to_swap)
    n_total = len(story_ids)
    print(f"Author randomization: swapping {n_swapped}/{n_total} stories ({100*n_swapped/n_total:.1f}%)")
    
    # Swap author columns for selected stories
    swap_rows = df[group_col].isin(stories_to_swap)
    
    # Swap author_1 and author_2
    df.loc[swap_rows, ['author_1', 'author_2']] = df.loc[swap_rows, ['author_2', 'author_1']].values
    
    # Update starter column: if we swapped, the original author_1 (now author_2) was the starter
    # So starter becomes 'author_2' for swapped stories
    if 'starter' in df.columns:
        # For swapped stories, flip the starter label
        original_starter = df.loc[swap_rows, 'starter'].copy()
        df.loc[swap_rows, 'starter'] = original_starter.map({
            'author_1': 'author_2',
            'author_2': 'author_1'
        })
    
    return df
