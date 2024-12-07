import re
import regex
import hashlib
import random
from datetime import datetime
from asrtk import utils
from asrtk.variables import abbreviations_dict
from asrtk.variables import unit_translations
from asrtk.variables import punctuation_dict
from asrtk.ordinals import normalize_ordinals
from .core.text import (
    turkish_capitalize,
    turkish_upper,
    turkish_lower,
    is_turkish_upper
)
from asrtk.variables import special_cases

def detect_decimal_separator(s: str) -> str:
    pattern = r"(\d+)(\.|,)(\d+)"
    match = re.search(pattern, s)

    if match:
        return match.group(2)  # Group 2 is the separator
    return None


class Normalizer:
    """
    - For more details about the algorithms and datasets, see `Readme <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/normalizer/ReadMe.md>`_.
    """

    def __init__(self):
        self.decimal_seperator = None

    def _convert_dates_to_words(self, text, merge_words):
        # Regular expression to match dates
        date_pattern = r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b'
        # Function to replace dates with their word form
        def replace_with_words(match):
            day, month, year = match.groups()
            day_words = self._num_to_words(int(day), 0, merge_words=merge_words)
            month_words = self._num_to_words(int(month), 0, merge_words=merge_words)
            year_words = self._num_to_words(int(year), 0, merge_words=merge_words)
            return f"{day_words} - {month_words} - {year_words}"

        return re.sub(date_pattern, replace_with_words, text)

    def convert_numbers_to_words(self, input_text, num_dec_digits=6, decimal_seperator=",", merge_words=False):
        # TODO: Bug:  6+-7     segment 6 artı -7 segment
        """
        Inherited from 'https://github.com/vngrs-ai/vnlp/blob/main/vnlp/normalizer/normalizer.py'

        Converts numbers to word form.

        Args:
            tokens:
                List of input tokens.
            num_dec_digits:
                Number of precision (decimal points) for floats.
            decimal_seperator:
                Decimal seperator character. Can be either "." or ",".

        Returns:
            List of converted tokens

        """
        input_text = input_text.replace(", ", " |$| ")
        input_text = input_text.replace("-", " ~ ")
        self.decimal_seperator = decimal_seperator
        half_spells = {
            ",5 gün": "buçuk gün",
            ",5 hafta": "buçuk hafta",
            ",5 dakika": "buçuk dakika",
            ",5 saat": "buçuk saat",
            ",5 ay": "buçuk ay",
            ",5 yıl": "buçuk yıl",
            # ",5 mm": "buçuk mm",
            # ",5 cm": "buçuk cm",
            ",5 metre": "buçuk metre",
        }
        input_text_ = input_text
        # replace if half_speels keys in input_text with half_speels values
        for key, value in half_spells.items():
            input_text = input_text.replace(key, f" {value}")

        modify_flag = input_text != input_text_

        alt_seperator = ""
        half_speel_chance_for_alternative_separator = random.randint(0, 100)
        if half_speel_chance_for_alternative_separator > 80:
            alt_seperator = " onda "

        half_speel_chance_for_dimensions = random.randint(0, 100)
        if half_speel_chance_for_dimensions > 60 or modify_flag:
            input_text = input_text.replace(",5 x ", " buçuk x ")
            input_text = input_text.replace(",5 mm", " buçuk mm ")
            input_text = input_text.replace(",5 cm", " buçuk cm ")


        # Convert dates in the input text to words
        text_with_converted_dates = self._convert_dates_to_words(input_text, merge_words=True)

        # Split the text into tokens
        tokens = text_with_converted_dates.split()

        converted_tokens = []

        for token in tokens:
            append_comma = False
            # if there's any numeric character in token
            if any([char.isnumeric() for char in token]):
                # if not decimal_seperator:
                self.decimal_seperator = detect_decimal_separator(token)
                # print(f'{token} self.decimal_seperator: {self.decimal_seperator}')
                if self.decimal_seperator == ",":
                    if token.endswith(",") and token[-2].isdigit():
                        token = token[:-1]
                        append_comma = True
                    # if decimal seperator is comma, then thousands seperator is dot and it will be converted to python's
                    # thousands seperator underscore.
                    # furthermore, comma will be converted to dot, python's decimal seperator.
                    token = token.replace(".", "_").replace(",", ".")
                elif self.decimal_seperator == ".":
                    # if decimal seperator is dot, then thousands seperator is comma and it will be converted to python's
                    # thousands seperator underscore.
                    token = token.replace(",", "_")
                # else:
                #     raise ValueError(decimal_seperator, 'is not a valid decimal seperator value. Use either "." or ","')

            # Try to convert token to number
            try:
                num = float(token)
                c_t = self._num_to_words(num, num_dec_digits, merge_words, alt_seperator)
                if append_comma:
                    c_t += ","
                converted_tokens += c_t.split()
            # If fails, then return it as string
            except:
                converted_tokens.append(token)

        tmp = " ".join(converted_tokens)
        tmp = tmp.replace(" |$| ", ", ")
        tmp = tmp.replace(" ~ ", "-")
        return tmp


    def _is_token_valid_turkish(self, token):
        """
        Checks whether given token is valid according to Turkish.
        """
        valid_according_to_stemmer_analyzer = not (
            self._stemmer_analyzer.candidate_generator.get_analysis_candidates(token)[0][-1] == "Unknown"
        )
        valid_according_to_lexicon = token in self._words_lexicon
        return valid_according_to_stemmer_analyzer or valid_according_to_lexicon

    def _int_to_words(self, main_num, put_commas=False, merge_words=False):
        """
        This function is adapted from:
        https://github.com/Omerktn/Turkish-Lexical-Representation-of-Numbers/blob/master/src.py
        It had a few bugs with numbers like 1000 and 1010, which are resolved.
        """

        # yüz=10^2 ve vigintilyon=10^63, ith element is 10^3 times greater then (i-1)th.
        tp = [
            " yüz",
            " bin",
            "",
            "",
            " milyon",
            " milyar",
            " trilyon",
            " katrilyon",
            " kentilyon",
            " seksilyon",
            " septilyon",
            " oktilyon",
            " nonilyon",
            " desilyon",
            " undesilyon",
            " dodesilyon",
            " tredesilyon",
            " katordesilyon",
            " seksdesilyon",
            " septendesilyon",
            " oktodesilyon",
            " nove mdesilyon",
            " vigintilyon",
        ]

        # dec[]: every decimal digit,  ten[]: every tenth number
        dec = ["", " bir", " iki", " üç", " dört", " beş", " altı", " yedi", " sekiz", " dokuz"]
        ten = ["", " on", " yirmi", " otuz", " kırk", " elli", " altmış", " yetmiş", " seksen", " doksan"]

        text = ""

        # get length of main_num
        num = main_num
        leng = 0
        while num != 0:
            num = num // 10
            leng += 1

        if main_num == 0:
            text = " sıfır"

        # split main_num to (three digit) pieces and read them by mod 3.
        for i in range(leng, 0, -1):
            digit = int((main_num // (10 ** (i - 1))) % 10)
            if i % 3 == 0:
                if digit == 1:
                    text += tp[0]
                elif digit == 0:
                    text += dec[digit]
                else:
                    text += dec[digit] + tp[0]
            elif i % 3 == 1:
                if i > 3:
                    if main_num > 1999:
                        text += dec[digit] + tp[i - 3]
                    else:
                        text += tp[i - 3]
                else:
                    text += dec[digit]
                if i > 3 and put_commas:
                    text += ","
            elif i % 3 == 2:
                text += ten[digit]

        return text[1:].replace(" ", "") if merge_words else text[1:]


    def _num_to_words(self, num, num_dec_digits, merge_words=False, alt_seperator=""):
        integer_part = int(num)
        decimal_part = round(num % 1, num_dec_digits)

        # if number is int (considering significant decimal digits)
        if decimal_part < 10**-num_dec_digits:
            return self._int_to_words(integer_part, merge_words=merge_words)
        # if number is float
        else:
            str_decimal = "{:f}".format(round(num % 1, num_dec_digits))[2:]

            zeros_after_decimal = 0
            for char in str_decimal:
                if char == "0":
                    zeros_after_decimal += 1
                else:
                    break
            str_decimal_stripped_from_zeros = str_decimal.strip(
                "0"
            )  # strip gets rid of heading and trailing 0s in string form
            if str_decimal_stripped_from_zeros == "":
                decimal_part = 0
            else:
                decimal_part = int(str_decimal_stripped_from_zeros)

            seperator_string = " nokta " if self.decimal_seperator == "." else " virgül "

            int_words = self._int_to_words(integer_part, merge_words=merge_words)
            decimal_words = self._int_to_words(decimal_part, merge_words=merge_words)

            if len(alt_seperator) > 0 and decimal_part < 10:
                seperator_string = alt_seperator

            return int_words + seperator_string + "sıfır " * zeros_after_decimal + decimal_words


def tts_normalize(text):
    """
    Preprocess text for TTS synthesis by normalizing numbers, units, and symbols.

    Args:
        text (str): The input text to be normalized.
        unit_translations (dict): A dictionary mapping abbreviations to their full forms.

    Returns:
        str: The normalized text.

    """
    spelled_out_numbers = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]

    # Define a regex pattern that matches abbreviations preceded by numbers/spelled-out numbers
    pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(re.escape(unit) for unit in unit_translations.keys()) + r')\b'
    text = text.replace("°", " ° ").replace("  ", " ")

    def replace_abbr(match):
        """Replace matched abbreviation with its full form."""
        number, unit = match.groups()
        unit = unit.lower()

        if any(number.startswith(word) for word in spelled_out_numbers) or number.isdigit() or '.' in number:
            replacement = f"{number} {unit_translations[unit]}"
        else:
            replacement = match.group(0)

        return replacement

    # Replace all occurrences of abbreviations with their full form
    normalized_text = re.sub(pattern, replace_abbr, text)

    normalized_text = normalized_text.replace("(+)", " pozitif ").replace("(-)", " negatif ").replace("+", " artı ").replace("  ", " ").strip()  # TODO translation'a taşı.

    if normalized_text.endswith("."):
        # Noktadan önceki karakteri kontrol et
        if not re.search(r'\d\.$', normalized_text):
            # Noktadan önce sayı yoksa, noktayı virgülle değiştir
            normalized_text = normalized_text[:-1] + ","

    # Ensure the text ends with a comma, unless it ends with a question mark or period
    if normalized_text[-1] not in ["?", ","]:
        normalized_text += ","

    return normalized_text


def dio_normalizer(text):
    """
    Preprocess text by replacing abbreviations with their full forms.

    Args:
        text (str): The input text to be normalized.
        abbreviations_dict (dict): A dictionary mapping abbreviations to their full forms.

    Returns:
        str: The normalized text.

    """

    # apply all special cases as key, value replace to text
    for key, value in special_cases.items():
        text = text.replace(key, value)

    # Sort abbreviations by length to match longer ones first
    sorted_abbr = sorted(abbreviations_dict.keys(), key=len, reverse=True)

    # Create a regular expression pattern that matches any abbreviation as a whole word
    pattern = r'\b(?:' + '|'.join(re.escape(abbr) for abbr in sorted_abbr) + r')\b'

    def replace_abbr(match):
        """Replace matched abbreviation with its full form."""
        word = match.group(0)
        # Get the first long form from the dictionary
        replacement = abbreviations_dict.get(word, [word])[0]
        return replacement

    # Replace all occurrences of any abbreviation with their full form
    normalized_text = re.sub(pattern, replace_abbr, text)

    return normalized_text


def replace_multiplication_symbol_in_dimensions(text):
    """
    Replace the multiplication symbol 'x' used in mathematical dimensions with a more descriptive term.

    Args:
        text (str): The input text containing dimensional expressions.

    Returns:
        str: The text with the multiplication symbol 'x' replaced by a descriptive term in dimensional expressions.

    """
    # Define a regex pattern that matches 'x' between numbers (with optional decimal parts and optional units)
    pattern = r'(\d+(?:\.\d+)?\s*(?:cm|mm)?)(\s*x\s*)(\d+(?:\.\d+)?\s*(?:cm|mm)?)(?:(\s*x\s*)(\d+(?:\.\d+)?\s*(?:cm|mm)?))?'

    def replacement(match):
        """Construct the replacement string with a descriptive term."""
        number1, x1, number2, x2, number3 = match.groups()
        replacement = f"{number1.strip()} çarpı {number2.strip()}"
        if x2 and number3:
            replacement += f" çarpı {number3.strip()}"
        return replacement

    # Replace all occurrences of 'x' between numbers with a descriptive term
    return re.sub(pattern, replacement, text).replace("  ", " ")



def split_text_improved(text, max_length=208, overlap=0):
    additional_splitters=["etmiş", "sahip", "ve"]
    def find_split_index(sentence, max_len):
        # Check for additional splitters and punctuation, and ensure not to split in the middle of a word
        best_split = -1
        for splitter in additional_splitters:  #  + [',']:  # Split by comma disabled due to unnatural stitches with forced alignment.
            idx = sentence.rfind(splitter, 0, max_len)
            if idx != -1:
                split_candidate = idx + len(splitter)
                if (split_candidate < max_len) and (split_candidate > best_split):
                    best_split = split_candidate

        # If no splitter or punctuation found, split at the last space before max_length
        if best_split == -1:
            best_split = sentence.rfind(' ', 0, max_len)
            if best_split == -1:
                # If no space found, enforce max_length split
                best_split = max_len

        return best_split

    if len(text) <= max_length:
        return [text]

    sentences = []
    while text:
        if len(text) <= max_length:
            sentences.append(text.strip())
            break

        split_index = find_split_index(text, max_length)
        if split_index <= 0:
            split_index = max_length

        part, text = text[:split_index].strip(), text[split_index:].strip()
        sentences.append(part.strip())

    processed_sentences = []

    for i, sentence in enumerate(sentences):
        # Prepend the last 'overlap' words of the previous sentence to the current sentence
        if i > 0 and overlap > 0:
            prev_sentence = sentences[i-1].split()
            # Check if the previous sentence has enough words to overlap
            if len(prev_sentence) >= overlap:
                overlap_words = " ".join(prev_sentence[-overlap:])
                sentence = f"{overlap_words} {sentence}"

        processed_sentences.append(sentence.strip())

    return processed_sentences


def create_random_date(splitter="/", seed=None):
    random.seed(seed)

    # Generate a random number between 1250000000 and 1700000000
    random_number = random.randint(1250000000, 1700000000)

    # Convert this number to a date
    random_date = datetime.fromtimestamp(random_number)

    return random_date.strftime("%d" + splitter + "%m" + splitter + "%Y")


def fill_dates(input_text, splitter="/"):
    """
    Create reproducible random dates if the input text contains "gg/aa/yyyy"
    """
    input_text = input_text.replace("Gg/aa/yyyy", "gg/aa/yyyy")

    timeout_counter = 0
    while "gg/aa/yyyy" in input_text:
        # Create a hash of the input text
        text_hash = int(hashlib.sha256(input_text.encode()).hexdigest(), 16)

        # Use the hash as a seed to create a random date
        rd = create_random_date(splitter, text_hash)
        input_text = input_text.replace("gg/aa/yyyy", rd, 1)
        timeout_counter += 1
        if timeout_counter > 10:
            break

    return input_text

def convert_numbers_to_words_wrapper(text):
    normalizer = Normalizer()
    return normalizer.convert_numbers_to_words(text)

def replace_punctuation_with_spelling(text):
    for punct, spelling in punctuation_dict.items():
        text = text.replace(punct, f" {spelling},")
    text = text.replace("  ", " ")
    return text

def replace_parentheses_with_spoken_form(text):
    """
    Replace parentheses with their spoken equivalents in the text.

    This function replaces opening and closing parentheses with "aç parantez" and "kapa parantez",
    respectively, in sentences that contain both opening and closing parentheses. It also handles
    sentences entirely enclosed in parentheses by adding "Parantez içinde" at the beginning.

    Args:
        text (str): The input text containing sentences with parentheses.

    Returns:
        str: The text with parentheses replaced by their spoken equivalents.
    """
    # Define a regular expression pattern to match sentences with text in parentheses
    pattern = r'(.+?)([.!?]*)\s*\(([^)]+)\)\s*([.!?]*)(\n|$)'

    def add_instruction(match):
        sentence, punct_before, in_parenthesis, punct_after, end_char = match.groups()
        # Remove the punctuation after the sentence if it's the same as the punctuation before the parentheses
        if punct_before and punct_before == punct_after:
            punct_after = ""
        # Replace the parentheses in the sentence
        in_parenthesis_replaced = f"aç parantez {in_parenthesis} kapa parantez,"
        return f"{sentence}{punct_before} {in_parenthesis_replaced}{punct_after}{end_char}"

    # Apply the transformation to all matching sentences
    normalized_text = re.sub(pattern, add_instruction, text)

    # Handle sentences entirely enclosed in parentheses
    normalized_text = re.sub(r'^\(([^)]+)\)\s*([.!?]*)(\n|$)', r'Parantez içinde \1\2\3', normalized_text)

    # Replace double periods with a single period
    normalized_text = normalized_text.replace("..", ".")

    normalized_text = normalized_text.replace(",,", ",")

    # Ensure the text ends with a comma, unless it ends with a question mark or period
    # if normalized_text[-1].strip() not in ["?", ","]:
    #     normalized_text = normalized_text.strip() + ","

    return normalized_text

def apply_normalizers(text, normalizers=[fill_dates, turkish_capitalize, replace_multiplication_symbol_in_dimensions, dio_normalizer, tts_normalize, normalize_ordinals, convert_numbers_to_words_wrapper, replace_punctuation_with_spelling]):
    """
    Apply a sequence of normalization functions to a given text.

    🚨 Note: The `fill_dates` normalizer is applied first to ensure reproducibility
    when generating random dates, as the text hash is used as a random seed.

    Args:
        text (str): The input text to be normalized.
        normalizers (list): A list of normalization functions to be applied
            to the text in the given order.

    Returns:
        str: The normalized text with double spaces replaced by single spaces.
    """
    for normalizer in normalizers:
        text = normalizer(text)
    return text.replace("  ", " ")

def normalized_hash(text, normalize=True):
    """ Added option to skip normalization cause we may need to hash already normalized text """
    if normalize:
        text = apply_normalizers(text.strip())
    return hashlib.sha256(text.encode()).hexdigest()

def merge_sentences(sentences, max_len=224):
    merged = []
    current_sentence = ''

    for sentence in sentences:
        sentence = sentence.strip()
        # Check if adding this sentence to current_sentence exceeds max_len
        if len(current_sentence + ' ' + sentence) <= max_len:
            current_sentence = current_sentence + ' ' + sentence if current_sentence else sentence
        else:
            if current_sentence:
                merged.append(current_sentence)
            current_sentence = sentence

    # Append the last sentence if it exists
    if current_sentence:
        merged.append(current_sentence)

    return merged



def for_vits(text):
    text = utils.turkish_lower(text)
    # Replace all characters not in a-z or Turkish specific chars with space
    # Adding Turkish special characters explicitly: ç, ğ, ı, ö, ş, ü
    text = re.sub(r'[^a-zçğıöşü]', ' ', text).replace("  ", " ")
    return text

def normalize_dictation(text):
    med_stopwords = ["nokta.", "paragraf", "bitti"]

    stopwords_sorted = sorted(med_stopwords, key=len, reverse=True)

    def remove_stopword(match):
        return ''

    normalized_text = regex.sub(r'[\n\t\[\]\{\}]', ' ', text)
    normalized_text = regex.sub(r'\s+', ' ', normalized_text).strip()

    for stopword in stopwords_sorted:
        # Adjusting the pattern to optionally include following punctuation like '.', ',', etc.
        pattern = r'\b' + regex.escape(stopword) + r'\b[.,]?'
        normalized_text = regex.sub(pattern, remove_stopword, normalized_text, flags=regex.IGNORECASE | regex.UNICODE)

    normalized_text = regex.sub(r'\s+', ' ', normalized_text).strip()

    # Handle edge cases for punctuation
    normalized_text = regex.sub(r'\.\s+\.', '.', normalized_text)  # Replace '. .' with '.'
    normalized_text = regex.sub(r'\s+\.', '.', normalized_text)  # Ensure no space before '.'

    return normalized_text


if __name__ == "__main__":
    # Example usage
    converter = Normalizer()
    print(converter.convert_numbers_to_words('"123", "456,789", "0.12"'))
    print(converter.convert_numbers_to_words("Her yıl Mart ayının 13. günü Pi günü olarak kutlanır. 3.14"))
    print(converter.convert_numbers_to_words("9876.43 ve 3,14 aynı cümlede."))
    print("*" * 50)

    print(converter.convert_numbers_to_words("Küsuratlı bazı sayılar, 175.5 try."))
    print(converter.convert_numbers_to_words("Virgülle ayrılmış bazı sayılar, 1,5 x 2,6, 3,2 x 6,8 milimetre."))
    print(converter.convert_numbers_to_words("1,5 gün önce."))
    print(converter.convert_numbers_to_words("yaklaşık 4,5-5 cm'ye kadar"))
