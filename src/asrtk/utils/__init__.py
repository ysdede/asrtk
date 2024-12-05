"""Utility functions for asrtk."""

from .console import (
    create_highlighted_text,
    print_file_header,
    print_replacement_example
)
from .file import (
    backup_file,
    is_valid_json_file,
    load_cache,
    save_cache
)
from .text import (
    natural_sort_key,
    add_space_after_punctuation,
    remove_text_in_brackets,
    remove_html_tags,
    remove_mismatched_characters,
    fix_punctuation,
    format_time,
    split_words,
    count_words,
    split_into_sentences_turkish
)

__all__ = [
    # Console utilities
    'create_highlighted_text',
    'print_file_header',
    'print_replacement_example',

    # File utilities
    'backup_file',
    'is_valid_json_file',
    'load_cache',
    'save_cache',

    # Text utilities
    'natural_sort_key',
    'add_space_after_punctuation',
    'remove_text_in_brackets',
    'remove_html_tags',
    'remove_mismatched_characters',
    'fix_punctuation',
    'format_time',
    'split_words',
    'count_words',
    'split_into_sentences_turkish'
]
