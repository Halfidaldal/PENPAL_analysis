"""
I/O utilities for loading and saving data.

This module provides standardized functions for reading and writing
data files in various formats (CSV, Parquet, NumPy arrays).

The active experiment is determined by `active_experiment` in config.yaml.
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import yaml


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_active_experiment() -> str:
    """Get the active experiment name from config."""
    config = load_config()
    return config.get('active_experiment', 'human-ai')


def get_experiment_config(experiment: Optional[str] = None) -> dict:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment: Experiment name ('human-ai' or 'human-human'). 
                   If None, uses active_experiment from config.
    
    Returns:
        Dict with experiment-specific configuration
    """
    config = load_config()
    if experiment is None:
        experiment = config.get('active_experiment', 'human-ai')
    
    if experiment not in config.get('experiments', {}):
        raise ValueError(f"Unknown experiment: {experiment}. Must be one of {list(config['experiments'].keys())}")
    
    return config['experiments'][experiment]


def get_shared_config() -> dict:
    """Get shared configuration that applies to all experiments."""
    config = load_config()
    return config.get('shared', {})


def get_data_path(stage: str = "processed", experiment: Optional[str] = None) -> Path:
    """
    Get the path to a data directory for an experiment.
    
    Args:
        stage: One of 'raw', 'interim', 'processed'
        experiment: Experiment name. If None, uses active_experiment from config.
        
    Returns:
        Path object to the data directory
    """
    exp_config = get_experiment_config(experiment)
    path_key = f"{stage}_dir"
    
    if path_key not in exp_config:
        raise ValueError(f"Unknown stage: {stage}. Must be one of ['raw', 'interim', 'processed']")
    
    return PROJECT_ROOT / exp_config[path_key]


def load_csv(filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to load from ('raw', 'interim', 'processed')
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading CSV from: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> None:
    """
    Save a DataFrame to CSV in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to save to ('raw', 'interim', 'processed')
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to df.to_csv
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving CSV to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_parquet(filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to load from
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to pd.read_parquet
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading Parquet from: {path}")
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> None:
    """
    Save a DataFrame to Parquet in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to save to
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to df.to_parquet
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving Parquet to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)


def load_npy(filename: str, stage: str = "processed", experiment: Optional[str] = None) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to load from
        experiment: Experiment name. If None, uses active_experiment from config.
        
    Returns:
        NumPy array
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading .npy from: {path}")
    return np.load(path)


def save_npy(arr: np.ndarray, filename: str, stage: str = "processed", experiment: Optional[str] = None) -> None:
    """
    Save a NumPy array to a .npy file.
    
    Args:
        arr: NumPy array to save
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to save to
        experiment: Experiment name. If None, uses active_experiment from config.
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving .npy to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
