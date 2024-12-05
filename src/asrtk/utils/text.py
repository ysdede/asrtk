"""Text utility functions."""
import re
from bs4 import BeautifulSoup

def natural_sort_key(s: str) -> list:
    """Create a key for natural sorting of strings with numbers.

    Args:
        s: String to create sort key for

    Returns:
        List of components for sorting
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]

def add_space_after_punctuation(text: str) -> str:
    """Add a space after punctuation marks in text, while avoiding decimal numbers.

    Args:
        text: Text to process

    Returns:
        Text with spaces added after punctuation
    """
    pattern = r"([!?]|[,\.](?!\d))(?=[^\s])"
    return re.sub(pattern, r"\1 ", text)

def remove_text_in_brackets(text: str) -> str:
    """Remove all text within square brackets or parentheses.

    Args:
        text: Text to process

    Returns:
        Text with bracketed content removed
    """
    pattern = r"\[.*?\]|\(.*?\)"
    return re.sub(pattern, "", text).strip()

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: Text containing HTML

    Returns:
        Plain text without HTML tags
    """
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_mismatched_characters(text: str) -> str:
    """Remove unclosed quotes and parentheses.

    Args:
        text: Text to process

    Returns:
        Text with mismatched characters removed
    """
    # Remove unopened or unclosed double quotes
    text = re.sub(r'(^|[^"])(")([^"]*$|[^"])', r'\1\3', text)

    # Remove unopened closing parentheses
    text = re.sub(r'(^|[^(\)])\)([^)\]]*$|[^(\)])', r'\1\2', text)

    # Remove unclosed opening parentheses
    text = re.sub(r'(^|[^)\]])\(([^(\[]*$|[^)\]])', r'\1\2', text)

    return text.strip()

def fix_punctuation(text: str, add_punct: bool = True) -> str:
    """Fix punctuation spacing and optionally add/remove ending punctuation.

    Args:
        text: Text to process
        add_punct: Whether to add ending punctuation if missing

    Returns:
        Text with fixed punctuation
    """
    if not text:
        return text

    # Ensure space after punctuation
    text = re.sub(r"([.,;:?])(?=[^\s])", r"\1 ", text)

    # Add ending period if needed
    if add_punct and not re.search(r"[.,;:?]$", text):
        text += "."
    # Remove ending punctuation if not wanted
    elif not add_punct and re.search(r"[.,;:?]$", text):
        text = text[:-1]

    return text

def format_time(ms: int) -> str:
    """Format milliseconds as HH:MM:SS.mmm.

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted time string
    """
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(ms):03}"

def split_words(text: str) -> list[str]:
    """Split text into words.

    Args:
        text: Text to split

    Returns:
        List of words
    """
    return re.findall(r"\b\w+\b", text)

def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words
    """
    return len(split_words(text))

def split_into_sentences_turkish(text: str) -> list[str]:
    """Split Turkish text into sentences.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence]
