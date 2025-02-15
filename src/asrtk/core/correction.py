"""Text correction utilities using transformer models."""
from typing import Optional, Dict, Any

class TextCorrector:
    """Handles text correction using transformer models for Turkish text."""

    def __init__(
        self,
        use_new_models: bool = False,
        device: Optional[str] = None,
        force_cpu: bool = False
    ):
        """Initialize text corrector with specified models.

        Args:
            use_new_models: Whether to use new BERT models for correction
            device: Device to run models on ("cuda" or "cpu"). If None, auto-detect.
            force_cpu: Force BERT models to run on CPU even if CUDA is available
        """
        from transformers import (
            pipeline,
            AutoTokenizer,
            BertForTokenClassification,
            PreTrainedModel,
            PreTrainedTokenizer
        )
        print(__name__)
        import torch
        # Auto-detect device if not specified
        if device is None:
            self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.use_new_models = use_new_models

        # Initialize as None - will be loaded on demand
        self._cap_model: Optional[PreTrainedModel] = None
        self._punc_model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._cap_pipeline = None
        self._punc_pipeline = None

        # Legacy punctuation model
        self._legacy_punc_model = None

    @property
    def legacy_punctuation_model(self):
        """Get legacy punctuation model."""
        if not self._legacy_punc_model:
            from asrtk.core.text import PunctuationRestorer
            self._legacy_punc_model = PunctuationRestorer()
        return self._legacy_punc_model

    def _load_new_models(self):
        """Load new BERT models if not already loaded."""
        try:
            if not self._cap_model:
                self._cap_model = BertForTokenClassification.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-capitalization-correction"
                )
                # Move model to appropriate device
                self._cap_model = self._cap_model.to(self.device)

            if not self._punc_model:
                self._punc_model = BertForTokenClassification.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-punctuation-correction"
                )
                # Move model to appropriate device
                self._punc_model = self._punc_model.to(self.device)

            if not self._tokenizer:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-capitalization-correction"
                )

            if not self._cap_pipeline:
                # Convert device string to pipeline device parameter
                pipeline_device = -1 if self.device == "cpu" else 0
                self._cap_pipeline = pipeline(
                    "ner",
                    model=self._cap_model,
                    tokenizer=self._tokenizer,
                    device=pipeline_device
                )

            if not self._punc_pipeline:
                # Convert device string to pipeline device parameter
                pipeline_device = -1 if self.device == "cpu" else 0
                self._punc_pipeline = pipeline(
                    "ner",
                    model=self._punc_model,
                    tokenizer=self._tokenizer,
                    device=pipeline_device
                )

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            # Retry loading on CPU
            if not self._cap_model:
                self._cap_model = BertForTokenClassification.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-capitalization-correction"
                ).cpu()
            if not self._punc_model:
                self._punc_model = BertForTokenClassification.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-punctuation-correction"
                ).cpu()
            if not self._tokenizer:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "ytu-ce-cosmos/turkish-base-bert-capitalization-correction"
                )
            if not self._cap_pipeline:
                self._cap_pipeline = pipeline(
                    "ner",
                    model=self._cap_model,
                    tokenizer=self._tokenizer,
                    device=-1
                )
            if not self._punc_pipeline:
                self._punc_pipeline = pipeline(
                    "ner",
                    model=self._punc_model,
                    tokenizer=self._tokenizer,
                    device=-1
                )

    def _preprocess(self, text: str) -> str:
        """Preprocess text for correction.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        punctuation = ['!', '?', '.', ',', '-', ':', ';', "'"]
        # Keep only alphanumeric, spaces and punctuation
        text = "".join(
            char for char in text
            if char in punctuation or char.isalnum() or char.isspace()
        )

        # Create pure text without punctuation
        pure_text = "".join(
            char for char in text
            if char.isalnum() or char.isspace() or char in ["'", "-"]
        )
        pure_text = pure_text.replace("'", " ").replace("-", " ")

        # Convert to lowercase and handle Turkish I/ı
        return pure_text.replace("I", "ı").lower()

    def correct_text(self, text: str, restore_punctuation: bool = True) -> str:
        """Correct text using loaded models.

        Args:
            text: Text to correct
            restore_punctuation: Whether to restore punctuation

        Returns:
            Corrected text
        """
        if not restore_punctuation:
            return text

        if not self.use_new_models:
            return self.legacy_punctuation_model.restore(text)

        # Load models if needed
        self._load_new_models()

        # Preprocess text
        processed_text = self._preprocess(text)

        # Get corrections
        cap_results = self._cap_pipeline(processed_text)
        punc_results = self._punc_pipeline(processed_text)

        # Get tokens
        tokens = self._tokenizer.tokenize(processed_text)

        # Build corrected text
        final_text = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Apply capitalization
            if cap_results[i]['entity'] == 'one':
                token = token.capitalize()
            elif cap_results[i]['entity'] == 'cap':
                token = token.upper()
                # Handle continuation tokens
                while i + 1 < len(tokens) and tokens[i + 1].startswith("##"):
                    token += tokens[i + 1][2:].upper()
                    i += 1

            # Apply punctuation
            if punc_results[i]['entity'] != 'non':
                token += punc_results[i]['entity']

            # Add space unless it's an apostrophe
            if punc_results[i]['entity'] != "'":
                token += ' '

            final_text.append(token)
            i += 1

        result = ''.join(final_text)
        return result.replace(' ##', '').strip()

    def __call__(self, text: str, restore_punctuation: bool = True) -> str:
        """Convenience method to correct text.

        Args:
            text: Text to correct
            restore_punctuation: Whether to restore punctuation

        Returns:
            Corrected text
        """
        return self.correct_text(text, restore_punctuation)
