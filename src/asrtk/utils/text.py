"""Text utility functions."""
import re
from bs4 import BeautifulSoup

# regex used in get_unique_words
punctuation_re = re.compile(r'[()?:;]')
end_punctuation_re = re.compile(r'[\.,]$')
double_space_re = re.compile(r'  +')
#

# Türkçe karakterler için özel büyütme ve küçültme eşlemeleri
turkish_upper_chars = {"ı": "I", "i": "İ", "ş": "Ş", "ğ": "Ğ", "ü": "Ü", "ö": "Ö", "ç": "Ç"}
turkish_lower_chars = {v: k for k, v in turkish_upper_chars.items()}


def turkish_upper(s):
    return "".join(turkish_upper_chars.get(c, c.upper()) for c in s)


def turkish_lower(s):
    return "".join(turkish_lower_chars.get(c, c.lower()) for c in s)


def is_turkish_upper(s):
    return s == turkish_upper(s)


def turkish_capitalize(s):
    if not s:
        return s
    return turkish_upper(s[0]) + s[1:]


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

def normalize_text(text: str) -> str:
    """
    Apply normalization to text described in Moonshine: Speech Recognition for Live Transcription and Voice Commands
    https://arxiv.org/html/2410.15608v2

    Handle Turkish specific characters, use helper functions defined earlier.

    Preprocessing noisily-labeled speech:
    Many speech sources available on the web have subtitles or captions available, which can serve as labels.
    However, captions tend be noisy—they may be manually-generated and thus contain text that is orthogonal
    to the audio content, or they may contain the names of speakers or verbal descriptions of non-speech content.
    In cases where a manually-generated but possibly-unreliable caption is available, we use a heuristic process to
    filter out low-quality instances. First, we lowercase and normalize the caption text, removing or replacing, e.g.,
    ambiguous unicode characters, emoji, and punctuation. We then use Whisper large v3 to generate a pseudo-label
    of the audio content, applying the same text normalization to this pseudo-label as we do the caption.
    Finally, we compute a normalized Levenshtein distance (between [0.0, 1.0],
    where 0.0 is identical and 1.0 is orthogonal) between the normalized caption and the pseudo-label, filtering out labels
    with a distance above a threshold.
    This allows us to treat the human-generated labels in captions as ground truth without introducing excessive noise.
    After filtering out noisy labels, we prepare the remaining text by applying standardized punctuation and capitalization.

    Preprocessing unlabeled speech:
    The majority of speech available on the web is unlabeled. In these cases, we leverage the Whisper large v3 model
    to generate training labels for our lighter-weight Moonshine model.
    The risk inherent in training one model on another model’s outputs is that the new model learns the old model’s errors.
    From inspection, we noted that the majority of hallucinated outputs from Whisper large v3 occurred below a predictable
    value of the average log probability of the output. We thus mitigate the risk of introducing hallucination and
    other noise in the training set by filtering out instances with an average log probability below this threshold.
    During this process, we benefited from speed-ups provided by batched inference in the WhisperX implementation (Bain et al., 2023).
    """

    text = turkish_lower(text)
    text = re.sub(r'[^a-zçğıöşü]', ' ', text).replace("  ", " ")
    return text


if __name__ == "__main__":
    print(normalize_text("Çok iyi ve nazik biriydi. Prusya’daki ilk karşılaşmamızda onu konuşturmayı başarmıştım. Bana o yaz North Cape’de bulunduğunu ve Nijni Novgorod panayırına gitmeyi çok istediğini anlatmıştı.,;)([-*])"))
