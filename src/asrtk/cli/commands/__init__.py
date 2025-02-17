"""CLI commands package."""
from typing import Callable

COMMANDS = {
    'create_wordset': '.wordset',
    'apply_patch': '.patch',
    'download_playlist': '.download',
    'download_channel': '.download',
    'download_channel_wosub': '.download',
    'fix': '.fix',
    'find_words': '.find',
    'find_arabic': '.find',
    'find_patterns': '.find',
    'find_brackets': '.find',
    'split': '.split',
    'merge_lines': '.merge',
    'remove_lines': '.remove',
    'count_numbers': '.numbers',
    'find_abbreviations': '.abbreviations',
    'fix_timestamps': '.fix_timestamps',
    'convert': '.convert',
    'probe_audio': '.convert',
    'probe_mp3': '.convert',
    'chunk': '.chunk',
    'duplicates': '.duplicates'
}

def get_command(command_name: str) -> Callable:
    """Lazily import and return the requested command."""
    module_path = COMMANDS[command_name]
    module = __import__(f'{__package__}{module_path}', fromlist=[command_name])
    return getattr(module, command_name)

__all__ = list(COMMANDS.keys())
