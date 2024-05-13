import os
import re
import json
import hashlib
import time
from contextlib import contextmanager

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


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def is_valid_json_file(file_path, size_threshold=1024):
    """Check if the file exists, has a size greater than the threshold and is a valid JSON."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < size_threshold:
        return False
    try:
        with open(file_path, "r") as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        return False


def romanize_turkish(text):
    # Map of Turkish characters to their Romanized equivalents
    import re

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

    # Additional normalization steps can be added here if needed
    # For example, removing or replacing punctuation
    romanized_text = re.sub(r"[.,;:!?\-()\"/]", " ", romanized_text)  # Replace punctuation with space
    romanized_text = romanized_text.replace("’", "'")  # Specific replacement
    romanized_text = re.sub(r"[^a-zA-Z' ]", "", romanized_text)  # Keep only letters, apostrophes, and spaces
    romanized_text = re.sub(r"\s+", " ", romanized_text)  # Replace multiple spaces with a single space

    return romanized_text.strip()


def add_space_after_punctuation(text):
    """
    Adds a space after punctuation marks in the given text, while avoiding
    decimal numbers (with either periods or commas).

    Args:
        text (str): The text to add spaces after punctuation marks.

    Returns:
        str: The text with spaces added after punctuation marks.
    """
    # The pattern now excludes periods and commas followed by digits
    pattern = r"([!?]|[,\.](?!\d))(?=[^\s])"
    return re.sub(pattern, r"\1 ", text)


def remove_text_in_brackets(text):
    """
    Removes all text within square brackets or parentheses from the given text.

    Parameters:
        text (str): The input text from which to remove the text within brackets.

    Returns:
        str: The text with the text within brackets removed.
    """

    pattern = r"\[.*?\]|\(.*?\)"
    return re.sub(pattern, "", text).strip()


def sanitize(s):
    """
    Sanitizes a given string by performing various replacements and transformations.

    :param s: The string to be sanitized.
    :type s: str
    :return: The sanitized string.
    :rtype: str
    """
    import html

    replacements = {
        "...": " ",
        "…": " ",
        "\n": " ",
        " ?": "?",
        " !": "!",
        "..": ".",
        # "-": " ",
        " .": ".",
        "’": "'"
        # Note: "  " (double space) is not added here due to its special handling
    }

    s = html.unescape(s)
    for old, new in replacements.items():
        s = s.replace(old, new)

    s = remove_text_in_brackets(s)
    s = add_space_after_punctuation(s)

    # Special handling for multiple spaces
    while "  " in s:
        s = s.replace("  ", " ")

    return s.strip()


def test_punc(captions, n_samples=10):
    # Join the first 'count' captions into a single string
    sample = " ".join(caption.text for caption in captions[:n_samples])
    # Count the number of periods in the sample
    period_count = sample.count(".")
    # print(sample, len(sample), period_count)
    return period_count


def fix_punctuation(s, add_punct=True):

    if not s:
        return s

    # Ensure space after commas and periods (and other specified punctuations)  # TODO sayılar?
    s = re.sub(r"([.,;:?])(?=[^\s])", r"\1 ", s)

    # Add a period at the end if `add_punct` is True and the string doesn't end with punctuation
    if add_punct and not re.search(r"[.,;:?]$", s):
        s += "."

    # Remove punctuation at the end if `add_punct` is False and the string ends with punctuation
    elif not add_punct and re.search(r"[.,;:?]$", s):
        s = s[:-1]

    return s


def format_time(ms):
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(ms):03}"


def split_words(text):
    return re.findall(r"\b\w+\b", text)


def count_words(text):
    return len(split_words(text))


def split_into_sentences_turkish(text):
    # Define the regex pattern for Turkish sentence endings
    sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
    # Use the re.split() function to split the text into sentences
    sentences = re.split(sentence_endings, text)
    # Filter out any empty strings that may be in the list
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def replace_attached_apostrophes_with_space(text):

    """
    Rakamdan sonra gelen bitişik tırnak işaretlerini (örneğin ' ) bir boşluk ile değiştiren fonksiyon.
    Çıktıda tırnak işaretleri kaldırılır.
    """
    text = replace_attached_hyphens_with_space(text)  # TODO: Bu filtreleri birleştir.
    pattern = r"(\d+)(\'[a-zA-ZğüşıöçĞÜŞİÖÇ]+)"

    def replace_with_space(match):
        return match.group(1) + " " + match.group(2)[1:]

    return re.sub(pattern, replace_with_space, text)


def replace_attached_hyphens_with_space(text):
    """
    Rakamdan sonra gelen bitişik tire işaretlerini (örneğin 19- ) bir boşluk ile değiştiren fonksiyon.
    Çıktıda tırnak işaretleri kaldırılır.
    """
    pattern = r"(\d+)(\-[a-zA-ZğüşıöçĞÜŞİÖÇ]+)"

    def replace_with_space(match):
        return match.group(1) + " " + match.group(2)[1:]

    return re.sub(pattern, replace_with_space, text)


def read_vtt_as_text(vtt_file_path):
    import webvtt

    captions = webvtt.read(vtt_file_path)
    return " ".join(caption.text.strip() for caption in captions)

def most_common_audio_format(audio_folder):
    from collections import Counter
    # Supported audio formats
    supported_formats = ['.wav', '.flac', '.mp3']

    # List all files in the folder and get their extensions
    files = [f for f in os.listdir(audio_folder) if os.path.isfile(os.path.join(audio_folder, f))]
    audio_extensions = [os.path.splitext(f)[1].lower() for f in files if os.path.splitext(f)[1].lower() in supported_formats]

    # Count the occurrences of each audio format
    format_counts = Counter(audio_extensions)

    # Identify the most common format
    if format_counts:
        most_common_format, _ = format_counts.most_common(1)[0]
        return most_common_format
    else:
        return None  # No supported audio files found

def calculate_md5_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.

    Args:
        file_path (str): The path to the file for which to calculate the MD5 checksum.

    Returns:
        str: The MD5 checksum of the file as a hexadecimal string.

    Example:
        >>> file_path = "path/to/file.txt"
        >>> md5_checksum = calculate_md5_checksum(file_path)
        >>> print(md5_checksum)
        "e4d909c290d0fb1ca068ffaddf22cbd0"

    This method reads the file in chunks of 4KB to efficiently calculate the MD5 checksum
    even for large files. It uses the `hashlib` module to compute the MD5 hash.

    Note:
        The MD5 algorithm is used here for the purpose of tracking changes and versioning
        files, not for security-critical applications.
    """

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_unique_words(s):
    if isinstance(s, str):
        s = s.split(' ')

    cleaned = set()
    for word in s:
        word = word.replace('"', '')
        # Önceden derlenmiş regex ifadelerini kullan
        word = punctuation_re.sub(' ', word)
        word = end_punctuation_re.sub('', word)
        word = double_space_re.sub(' ', word)
        word = word.strip()
        if len(word) > 0:
            cleaned.add(word)
    return cleaned

def get_unique_words_with_frequencies(s, threshold=1, reverse=False):
    if isinstance(s, str):
        s = s.split(' ')

    cleaned = set()
    word_frequencies = {}  # Kelime frekanslarını saklamak için bir sözlük

    for word in s:
        if "/2022" in word or "/2023" in word or "/2024" in word or "/2025" in word:
            continue
        word = word.replace('"', '')
        # Önceden derlenmiş regex ifadelerini kullan
        word = punctuation_re.sub(' ', word)
        word = end_punctuation_re.sub('', word)
        word = double_space_re.sub(' ', word)
        word = word.strip()

        if len(word) > 0:
            cleaned.add(word)
            # Kelime frekansını güncelle
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1

    if reverse:
        filtered_words = {word for word in cleaned if word_frequencies[word] <= threshold}
    else:
        # Frekans filtresi uygula: En az 3 kez geçen kelimeleri içeren bir set oluştur
        filtered_words = {word for word in cleaned if word_frequencies[word] >= threshold}

    return filtered_words, word_frequencies

@contextmanager
def measure_time(label):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"⏱'{label}' işlem süresi: {end_time - start_time:.2f} saniye.")
