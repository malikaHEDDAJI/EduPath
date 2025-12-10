"""
Utility functions for the Learning Analytics Platform.
"""

from pathlib import Path
from typing import Dict, Tuple
import yaml


def load_settings(config_path: str = "config/settings.yaml") -> Dict:
    """
    Load configuration settings from a YAML file.
    
    Args:
        config_path: Path to the settings YAML file
        
    Returns:
        Dictionary containing the configuration settings.
        Returns empty dict if file doesn't exist or is invalid.
    """
    settings_path = Path(config_path)
    
    if not settings_path.exists():
        return {}
    
    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f) or {}
        return settings
    except Exception as e:
        print(f"Warning: Could not load settings from {config_path}: {e}")
        return {}


def get_data_paths(base_dir: str = ".") -> Tuple[Path, Path]:
    """
    Get paths for raw and processed data directories.
    
    Args:
        base_dir: Base directory of the project (default: current directory)
        
    Returns:
        Tuple of (raw_data_path, processed_data_path) as Path objects
    """
    base = Path(base_dir)
    raw_path = base / "data" / "raw"
    processed_path = base / "data" / "processed"
    
    return raw_path, processed_path


