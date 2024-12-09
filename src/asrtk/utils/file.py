"""File handling utilities."""
from pathlib import Path
import json
from typing import Dict, Any
import shutil
from datetime import datetime

def backup_file(file_path: str) -> str:
    """Create a backup of a file with timestamp.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".{timestamp}.bak")
    shutil.copy2(file_path, backup_path)
    return str(backup_path)

def is_valid_json_file(file_path: Path) -> bool:
    """Check if a file is a valid JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        True if file is valid JSON
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

def load_cache(cache_file: Path) -> Dict[str, Any]:
    """Load a JSON cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        Cache contents as dictionary
    """
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"Warning: Cache file {cache_file} corrupted, starting fresh")
    return {}

def save_cache(cache: Dict[str, Any], cache_file: Path) -> None:
    """Save a dictionary to a JSON cache file.

    Args:
        cache: Dictionary to save
        cache_file: Path to cache file
    """
    cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding='utf-8')
