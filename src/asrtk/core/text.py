"""Text processing utilities."""
from typing import Tuple, Dict, List
from collections import Counter
import re

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

    return text

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
