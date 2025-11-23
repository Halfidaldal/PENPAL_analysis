"""
I/O utilities for loading and saving data.

This module provides standardized functions for reading and writing
data files in various formats (CSV, Parquet, NumPy arrays).
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import yaml


# Default paths - can be overridden by config
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_path(stage: str = "processed") -> Path:
    """
    Get the path to a data directory for the active dataset.
    
    Args:
        stage: One of 'raw', 'interim', 'processed'
        
    Returns:
        Path object to the data directory
    """
    try:
        config = load_config()
        active_dataset = config.get('active_dataset', 'TEXT')
        dataset_config = config['datasets'][active_dataset]
        path_key = f"{stage}_dir"
        
        if path_key not in dataset_config:
            raise ValueError(f"Unknown stage: {stage}. Must be one of ['raw', 'interim', 'processed']")
        
        return PROJECT_ROOT / dataset_config[path_key]
    except (KeyError, FileNotFoundError):
        # Fallback to old behavior if config is missing
        paths = {
            "raw": DATA_RAW,
            "interim": DATA_INTERIM,
            "processed": DATA_PROCESSED,
        }
        if stage not in paths:
            raise ValueError(f"Unknown stage: {stage}. Must be one of {list(paths.keys())}")
        return paths[stage]


def load_csv(filename: str, stage: str = "processed", **kwargs) -> pd.DataFrame:
    """
    Load a CSV file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to load from ('raw', 'interim', 'processed')
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage) / filename
    print(f"Loading CSV from: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, filename: str, stage: str = "processed", **kwargs) -> None:
    """
    Save a DataFrame to CSV in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to save to ('raw', 'interim', 'processed')
        **kwargs: Additional arguments passed to df.to_csv
    """
    path = get_data_path(stage) / filename
    print(f"Saving CSV to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_parquet(filename: str, stage: str = "processed", **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to load from
        **kwargs: Additional arguments passed to pd.read_parquet
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage) / filename
    print(f"Loading Parquet from: {path}")
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, filename: str, stage: str = "processed", **kwargs) -> None:
    """
    Save a DataFrame to Parquet in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to save to
        **kwargs: Additional arguments passed to df.to_parquet
    """
    path = get_data_path(stage) / filename
    print(f"Saving Parquet to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)


def load_npy(filename: str, stage: str = "processed") -> np.ndarray:
    """
    Load a NumPy array from a .npy file.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to load from
        
    Returns:
        NumPy array
    """
    path = get_data_path(stage) / filename
    print(f"Loading .npy from: {path}")
    return np.load(path)


def save_npy(arr: np.ndarray, filename: str, stage: str = "processed") -> None:
    """
    Save a NumPy array to a .npy file.
    
    Args:
        arr: NumPy array to save
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to save to
    """
    path = get_data_path(stage) / filename
    print(f"Saving .npy to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
