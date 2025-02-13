"""CLI commands package."""
from .wordset import create_wordset
from .patch import apply_patch
from .download import download_playlist, download_channel
from .fix import fix
from .find import find_words, find_arabic, find_patterns, find_brackets
from .split import split
from .merge import merge_lines
from .remove import remove_lines
from .numbers import count_numbers
from .abbreviations import find_abbreviations
from .fix_timestamps import fix_timestamps
from .convert import convert, probe_audio, probe_mp3
from .chunk import chunk
from .duplicates import duplicates

__all__ = [
    'create_wordset',
    'apply_patch',
    'download_playlist',
    'download_channel',
    'fix',
    'find_words',
    'find_arabic',
    'find_patterns',
    'find_brackets',
    'split',
    'merge_lines',
    'remove_lines',
    'count_numbers',
    'find_abbreviations',
    'fix_timestamps',
    'convert',
    'probe_audio',
    'probe_mp3',
    'chunk',
    'duplicates',
]
