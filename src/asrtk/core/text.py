"""Text processing utilities."""
from typing import Tuple, Dict, List
from collections import Counter
import re
from transformers import pipeline
import os
import torch

# Turkish character mappings
turkish_upper_chars = {
    "ı": "I",
    "i": "İ",
    "ş": "Ş",
    "ğ": "Ğ",
    "ü": "Ü",
    "ö": "Ö",
    "ç": "Ç"
}
turkish_lower_chars = {v: k for k, v in turkish_upper_chars.items()}

def turkish_upper(s: str) -> str:
    """Convert text to uppercase using Turkish-specific rules."""
    return "".join(turkish_upper_chars.get(c, c.upper()) for c in s)

def turkish_lower(s: str) -> str:
    """Convert text to lowercase using Turkish-specific rules."""
    return "".join(turkish_lower_chars.get(c, c.lower()) for c in s)

def is_turkish_upper(s: str) -> bool:
    """Check if text is uppercase according to Turkish rules."""
    return s == turkish_upper(s)

def fix_spaces(text: str) -> str:
    """Fix spaces in text."""
    return text.replace('   ', ' ').replace('  ', ' ')

def sanitize(text: str) -> str:
    """Sanitize text by removing HTML tags and normalizing whitespace.

    Args:
        text: Input text

    Returns:
        Sanitized text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return fix_spaces(text).strip()

def sanitize_for_merge(text: str) -> str:
    """Sanitize text for merging."""
    text = sanitize(text).strip()

    if text.endswith('...'):
        text = f"{text[:-3]} "

    if text.startswith('...'):
        text = text[3:]

    return fix_spaces(text).strip()

def get_unique_words_with_frequencies(text: str) -> Tuple[List[str], Dict[str, int]]:
    """Get unique words and their frequencies from text.

    Args:
        text: Input text

    Returns:
        Tuple of (list of unique words, frequency dictionary)
    """
    # Split into words and filter empty strings
    words = [word for word in text.split() if word]

    # Count frequencies
    frequencies = Counter(words)

    # Get unique words sorted by frequency
    unique_words = sorted(frequencies.keys(), key=lambda x: (-frequencies[x], x))

    return unique_words, frequencies

def find_sample_sentences(text: str, pattern: str, max_samples: int = 3, max_length: int = None) -> List[str]:
    """Find sample sentences containing the given pattern.

    Args:
        text: Text to search in
        pattern: Pattern to find
        max_samples: Maximum number of samples to return
        max_length: Maximum length of each sample (if None, shows full lines)

    Returns:
        List of sample sentences/contexts
    """
    samples = []
    lines = text.split('\n')

    for line in lines:
        if pattern in line and not line.startswith('WEBVTT') and '-->' not in line:
            # Only trim if max_length is specified
            if max_length and len(line) > max_length:
                # Find the pattern position
                pos = line.find(pattern)
                # Take some context before and after
                start = max(0, pos - max_length//2)
                end = min(len(line), pos + max_length//2)
                sample = ('...' if start > 0 else '') + \
                        line[start:end] + \
                        ('...' if end < len(line) else '')
            else:
                sample = line.strip()  # Use full line, just strip whitespace

            if sample not in samples:  # Avoid duplicates
                samples.append(sample)
                if len(samples) >= max_samples:
                    break

    return samples

def has_arabic(text: str) -> bool:
    """Check if text contains Arabic characters.

    Args:
        text: Text to check

    Returns:
        True if text contains Arabic characters
    """
    return any(ord(char) in range(0x0600, 0x06FF) or  # Arabic
              ord(char) in range(0xFE70, 0xFEFF) or   # Arabic Presentation Forms-B
              ord(char) in range(0xFB50, 0xFDFF)      # Arabic Presentation Forms-A
              for char in text)

def test_punc(captions, n_samples=10):
    """Test punctuation in captions."""
    # Join the first 'count' captions into a single string
    sample = " ".join(caption.text for caption in captions[:n_samples])
    # Count the number of periods in the sample
    period_count = sample.count(".")
    return period_count

def remove_mismatched_characters(text):
    """Remove mismatched quotes and parentheses."""
    # Regex to find unclosed or unopened double quotes
    text = re.sub(r'(^|[^"])(")([^"]*$|[^"])', r'\1\3', text)

    # Regex to find unclosed or unopened parentheses
    text = re.sub(r'(^|[^(\)])\)([^)\]]*$|[^(\)])', r'\1\2', text)  # Remove unopened closing parentheses
    text = re.sub(r'(^|[^)\]])\(([^(\[]*$|[^)\]])', r'\1\2', text)  # Remove unclosed opening parentheses

    return text.strip()

def format_time(ms):
    """Format milliseconds into VTT timestamp format."""
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(ms):03}"

def natural_sort_key(s: str) -> list:
    """Create a key for natural sorting of strings containing numbers.

    Args:
        s: String to create sort key for

    Returns:
        List of components for sorting (numbers as ints, text as lowercase)
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]

def romanize_turkish(text: str) -> str:
    """Convert Turkish text to romanized form.

    Args:
        text: Turkish text to romanize

    Returns:
        Romanized text
    """
    # Map of Turkish characters to their Romanized equivalents
    roman_map = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c",
        "İ": "i",
        "Ğ": "g",
        "Ü": "u",
        "Ş": "s",
        "Ö": "o",
        "Ç": "c",
    }

    # Replace each Turkish character with its Romanized equivalent
    romanized_text = "".join(roman_map.get(c, c) for c in text).lower()

    # Additional normalization steps
    romanized_text = re.sub(r"[.,;:!?\-()\"/]", " ", romanized_text)  # Replace punctuation with space
    romanized_text = romanized_text.replace("'", "'")  # Specific replacement
    romanized_text = re.sub(r"[^a-zA-Z' ]", "", romanized_text)  # Keep only letters, apostrophes, and spaces
    romanized_text = re.sub(r"\s+", " ", romanized_text)  # Replace multiple spaces with a single space

    return romanized_text.strip()

def turkish_capitalize(s: str) -> str:
    """Capitalize text using Turkish-specific rules.

    Args:
        s: Text to capitalize

    Returns:
        Capitalized text
    """
    if not s:
        return s
    return turkish_upper(s[0]) + s[1:]

class PunctuationRestorer:
    """Handles Turkish text punctuation restoration using BERT model."""

    _instance = None
    _model = None

    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super(PunctuationRestorer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model only once."""
        if PunctuationRestorer._model is None:
            PunctuationRestorer._model = pipeline(
                task="token-classification",
                model="uygarkurt/bert-restore-punctuation-turkish"
            )

    def restore(self, text):
        """
        Restore punctuation in the given text.
        Only applies predictions with confidence score above 90%.
        """
        predictions = [
            pred for pred in self._model(text)
            if pred['score'] >= 0.9  # Only keep high confidence predictions
        ]
        return self._restore_punctuation(text, predictions)

    def _restore_punctuation(self, text, model_output):
        """
        Internal method to restore punctuation using model output.
        Handles agglutinative suffixes, apostrophes, quotes, and Turkish punctuation rules.
        Only applies predictions with confidence score above 90%.
        """
        predictions = sorted(model_output, key=lambda x: x['start'])
        result = list(text)
        offset = 0

        i = 0
        while i < len(predictions):
            current_pred = predictions[i]

            # Skip predictions with low confidence
            if current_pred['score'] < 0.9:
                i += 1
                continue

            # Skip if current token is a suffix or part of apostrophe/quote
            if (current_pred['word'].startswith('##') or
                current_pred['word'] in ["'", '"'] or
                (i > 0 and predictions[i-1]['word'] in ["'", '"'])):
                i += 1
                continue

            # Find the last part of the current word
            last_pos = i
            while last_pos + 1 < len(predictions):
                next_pred = predictions[last_pos + 1]
                if (next_pred['word'].startswith('##') or
                    next_pred['start'] == predictions[last_pos]['end']):
                    last_pos += 1
                else:
                    break

            # Only process punctuation if this is the last token of a word
            if current_pred['entity'] in ['PERIOD', 'QUESTION_MARK', 'COMMA']:
                # Skip if we're in the middle of a word (more tokens follow)
                if last_pos > i:
                    i = last_pos + 1
                    continue

                insert_pos = predictions[last_pos]['end'] + offset

                # Don't insert punctuation in the middle of a word with apostrophe/quote
                if (insert_pos < len(result) and
                    (result[insert_pos] in ["'", '"'] or
                     (insert_pos > 0 and result[insert_pos-1] in ["'", '"']))):
                    i = last_pos + 1
                    continue

                # For comma, check if there's a space after the insertion point
                if (current_pred['entity'] == 'COMMA' and
                    insert_pos < len(result) - 1 and
                    not result[insert_pos].isspace()):
                    i = last_pos + 1
                    continue

                # Check if there's already a punctuation mark
                if (insert_pos < len(result) and
                    result[insert_pos] in ['.', ',', '?']):
                    # Replace existing punctuation
                    punct = {
                        'PERIOD': '.',
                        'QUESTION_MARK': '?',
                        'COMMA': ','
                    }[current_pred['entity']]
                    result[insert_pos] = punct
                else:
                    # Check for existing punctuation in surrounding positions
                    has_punct_before = (insert_pos > 0 and
                                      result[insert_pos - 1] in ['.', ',', '?'])
                    has_punct_after = (insert_pos < len(result) and
                                     result[insert_pos] in ['.', ',', '?'])
                    has_quote_after = (insert_pos < len(result) and
                                     result[insert_pos] == '"')

                    # Skip if there's already punctuation nearby or quote after
                    if has_punct_before or has_punct_after or has_quote_after:
                        i = last_pos + 1
                        continue

                    # Insert new punctuation if no existing punctuation nearby
                    punct = {
                        'PERIOD': '.',
                        'QUESTION_MARK': '?',
                        'COMMA': ','
                    }[current_pred['entity']]
                    result.insert(insert_pos, punct)
                    offset += 1

            i = last_pos + 1

        return ''.join(result)
