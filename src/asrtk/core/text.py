"""Text processing utilities."""
from typing import Tuple, Dict, List
from collections import Counter
import re

def turkish_lower(text: str) -> str:
    """Convert text to lowercase using Turkish-specific rules.

    Args:
        text: Input text

    Returns:
        Lowercase text with Turkish character handling
    """
    # Turkish-specific lowercase mappings
    tr_map = {
        'İ': 'i',
        'I': 'ı',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ş': 'ş',
        'Ö': 'ö',
        'Ç': 'ç'
    }

    # Apply Turkish mappings first
    for upper, lower in tr_map.items():
        text = text.replace(upper, lower)

    # Then do standard lowercase
    return text.lower()

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
