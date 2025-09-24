"""
Kannada-Specific Text Preprocessor

Advanced preprocessing module specifically designed for Kannada language,
handling script-specific normalization, morphological analysis, and
text cleaning optimized for the Kannada writing system.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import yaml
from pathlib import Path

from .text_cleaner import TextCleaner


class KannadaPreprocessor(TextCleaner):
    """
    Kannada-specific text preprocessor extending the base TextCleaner.
    
    Handles Kannada script normalization, morphological analysis,
    compound word processing, and language-specific cleaning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Kannada preprocessor.
        
        Args:
            config_path: Path to Kannada configuration file
        """
        super().__init__(language='kn')
        
        # Load Kannada-specific configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "kannada_config.yaml"
        
        self.kannada_config = self._load_config(config_path)
        
        # Kannada Unicode ranges and characters
        self.kannada_unicode_range = (0x0C80, 0x0CFF)
        self.kannada_digits = "೦೧೨೩೪೫೬೭೮೯"
        self.ascii_digits = "0123456789"
        
        # Kannada-specific character mappings
        self.vowels = set([
            'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ೡ', 
            'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ'
        ])
        
        self.consonants = set([
            'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ',
            'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ', 'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ',
            'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಲ', 'ವ', 'ಶ',
            'ಷ', 'ಸ', 'ಹ', 'ಳ', 'ೞ', 'ೱ', 'ೲ'
        ])
        
        # Kannada diacritics (vowel signs)
        self.diacritics = set([
            'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್'
        ])
        
        # Virama (halant) - the inherent vowel killer
        self.virama = '್'
        
        # Common Kannada conjuncts and ligatures
        self.conjuncts = [
            'ಕ್ಕ', 'ಗ್ಗ', 'ಙ್ಙ', 'ಚ್ಚ', 'ಜ್ಜ', 'ಞ್ಞ', 'ಟ್ಟ', 'ಡ್ಡ', 'ಣ್ಣ',
            'ತ್ತ', 'ದ್ದ', 'ನ್ನ', 'ಪ್ಪ', 'ಬ್ಬ', 'ಮ್ಮ', 'ಯ್ಯ', 'ರ್ರ', 'ಲ್ಲ',
            'ವ್ವ', 'ಶ್ಶ', 'ಷ್ಷ', 'ಸ್ಸ', 'ಹ್ಹ', 'ಳ್ಳ'
        ]
        
        # Kannada stop words
        self.stop_words = set(self.kannada_config.get('stop_words', []))
        
        # Morphological patterns
        self.prefixes = self.kannada_config.get('morphology', {}).get('prefixes', [])
        self.suffixes = self.kannada_config.get('morphology', {}).get('suffixes', [])
        self.verb_endings = self.kannada_config.get('morphology', {}).get('verb_endings', [])
    
    def _load_config(self, config_path: str) -> Dict:
        """Load Kannada configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {}
    
    def clean_kannada_text(self, 
                          text: str,
                          normalize_unicode: bool = True,
                          normalize_digits: bool = True,
                          remove_mixed_script: bool = False,
                          preserve_conjuncts: bool = True,
                          handle_virama: bool = True) -> str:
        """
        Clean Kannada text with script-specific processing.
        
        Args:
            text: Input Kannada text
            normalize_unicode: Whether to normalize Unicode characters
            normalize_digits: Whether to normalize Kannada digits to ASCII
            remove_mixed_script: Whether to remove non-Kannada script text
            preserve_conjuncts: Whether to keep conjunct characters together
            handle_virama: Whether to handle virama/halant properly
            
        Returns:
            Cleaned Kannada text
        """
        if not text:
            return ""
        
        # Base cleaning
        cleaned = self.clean_text(text, normalize_unicode=normalize_unicode)
        
        # Kannada-specific normalization
        if normalize_digits:
            cleaned = self._normalize_kannada_digits(cleaned)
        
        if remove_mixed_script:
            cleaned = self._remove_non_kannada_text(cleaned)
        
        if handle_virama:
            cleaned = self._handle_virama_sequences(cleaned)
        
        if preserve_conjuncts:
            cleaned = self._preserve_conjuncts(cleaned)
        
        # Additional Kannada-specific cleaning
        cleaned = self._normalize_kannada_punctuation(cleaned)
        cleaned = self._handle_kannada_spacing(cleaned)
        
        return cleaned.strip()
    
    def _normalize_kannada_digits(self, text: str) -> str:
        """Convert Kannada digits to ASCII digits."""
        digit_map = str.maketrans(self.kannada_digits, self.ascii_digits)
        return text.translate(digit_map)
    
    def _remove_non_kannada_text(self, text: str) -> str:
        """Remove text that's not in Kannada script."""
        # Keep Kannada characters, ASCII digits, punctuation, and whitespace
        kannada_pattern = r'[^\u0C80-\u0CFF\u0020-\u007F\s]+'
        return re.sub(kannada_pattern, ' ', text)
    
    def _handle_virama_sequences(self, text: str) -> str:
        """Properly handle virama (halant) sequences in Kannada."""
        # Normalize multiple consecutive viramas
        text = re.sub(r'್+', '್', text)
        
        # Remove virama at the end of words (often incorrect)
        text = re.sub(r'್\s', ' ', text)
        text = re.sub(r'್$', '', text)
        
        return text
    
    def _preserve_conjuncts(self, text: str) -> str:
        """Ensure conjunct characters are kept together."""
        # This is a placeholder - in practice, you'd implement
        # more sophisticated conjunct preservation
        for conjunct in self.conjuncts:
            # Ensure conjuncts aren't split by extra spaces
            pattern = conjunct.replace('್', r'್\s*')
            text = re.sub(pattern, conjunct, text)
        
        return text
    
    def _normalize_kannada_punctuation(self, text: str) -> str:
        """Normalize Kannada punctuation marks."""
        # Map common punctuation
        punctuation_map = {
            '।': '.',  # Devanagari danda to period
            '॥': '.',   # Double danda to period
            '‍': '',    # Zero-width joiner
            '‌': '',    # Zero-width non-joiner
        }
        
        for kannada_punct, replacement in punctuation_map.items():
            text = text.replace(kannada_punct, replacement)
        
        return text
    
    def _handle_kannada_spacing(self, text: str) -> str:
        """Handle spacing issues specific to Kannada text."""
        # Remove extra spaces around Kannada characters
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])\s+', r'\1 ', text)
        
        return text
    
    def tokenize_kannada_words(self, text: str) -> List[str]:
        """
        Tokenize Kannada text into words, handling morphology.
        
        Args:
            text: Input Kannada text
            
        Returns:
            List of Kannada word tokens
        """
        # Clean the text first
        clean_text = self.clean_kannada_text(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'[\u0C80-\u0CFF]+|[^\s\u0C80-\u0CFF]+', clean_text)
        
        # Filter out empty tokens and punctuation-only tokens
        kannada_tokens = []
        for token in tokens:
            if token.strip() and any(c in self.vowels or c in self.consonants for c in token):
                kannada_tokens.append(token.strip())
        
        return kannada_tokens
    
    def analyze_kannada_morphology(self, word: str) -> Dict[str, str]:
        """
        Analyze morphological structure of a Kannada word.
        
        Args:
            word: Kannada word to analyze
            
        Returns:
            Dictionary with morphological analysis
        """
        analysis = {
            'root': word,
            'prefixes': [],
            'suffixes': [],
            'word_type': 'unknown'
        }
        
        # Check for prefixes
        for prefix in self.prefixes:
            if word.startswith(prefix):
                analysis['prefixes'].append(prefix)
                word = word[len(prefix):]
                break
        
        # Check for suffixes
        for suffix in self.suffixes:
            if word.endswith(suffix):
                analysis['suffixes'].append(suffix)
                word = word[:-len(suffix)]
                break
        
        # Check for verb endings
        for ending in self.verb_endings:
            if word.endswith(ending):
                analysis['word_type'] = 'verb'
                analysis['suffixes'].append(ending)
                word = word[:-len(ending)]
                break
        
        analysis['root'] = word
        return analysis
    
    def remove_kannada_stop_words(self, text: str) -> str:
        """
        Remove Kannada stop words from text.
        
        Args:
            text: Input Kannada text
            
        Returns:
            Text with stop words removed
        """
        words = self.tokenize_kannada_words(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def get_kannada_text_statistics(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Get comprehensive statistics for Kannada text.
        
        Args:
            text: Input Kannada text
            
        Returns:
            Dictionary with text statistics
        """
        # Get base statistics
        stats = self.get_text_statistics(text)
        
        # Add Kannada-specific statistics
        kannada_chars = sum(1 for c in text if self.kannada_unicode_range[0] <= ord(c) <= self.kannada_unicode_range[1])
        vowel_count = sum(1 for c in text if c in self.vowels)
        consonant_count = sum(1 for c in text if c in self.consonants)
        diacritic_count = sum(1 for c in text if c in self.diacritics)
        virama_count = text.count(self.virama)
        
        # Conjunct analysis
        conjunct_count = sum(text.count(conjunct) for conjunct in self.conjuncts)
        
        # Word analysis
        words = self.tokenize_kannada_words(text)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        kannada_stats = {
            'kannada_characters': kannada_chars,
            'vowel_count': vowel_count,
            'consonant_count': consonant_count,
            'diacritic_count': diacritic_count,
            'virama_count': virama_count,
            'conjunct_count': conjunct_count,
            'kannada_words': len(words),
            'avg_word_length': avg_word_length,
            'script_purity': kannada_chars / len(text) if text else 0
        }
        
        # Merge with base statistics
        stats.update(kannada_stats)
        return stats
    
    def extract_kannada_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from Kannada text using script-specific rules.
        
        Args:
            text: Input Kannada text
            
        Returns:
            List of Kannada sentences
        """
        # Clean the text first
        clean_text = self.clean_kannada_text(text)
        
        # Split on Kannada sentence boundaries
        # Common Kannada sentence enders: period, question mark, exclamation
        sentence_pattern = r'[.!?।॥]+\s*'
        sentences = re.split(sentence_pattern, clean_text)
        
        # Clean and filter sentences
        kannada_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum length threshold
                # Check if sentence contains significant Kannada content
                kannada_ratio = sum(1 for c in sentence if self.kannada_unicode_range[0] <= ord(c) <= self.kannada_unicode_range[1]) / len(sentence)
                if kannada_ratio > 0.3:  # At least 30% Kannada characters
                    kannada_sentences.append(sentence)
        
        return kannada_sentences
    
    def preprocess_kannada_dataset(self, 
                                  data: Union[List[str], pd.DataFrame],
                                  text_column: str = 'content',
                                  clean_options: Optional[Dict] = None) -> Union[List[str], pd.DataFrame]:
        """
        Preprocess a dataset of Kannada text.
        
        Args:
            data: Input data (list of strings or DataFrame)
            text_column: Column name if data is DataFrame
            clean_options: Cleaning options for Kannada text
            
        Returns:
            Preprocessed data in same format as input
        """
        if clean_options is None:
            clean_options = {
                'normalize_unicode': True,
                'normalize_digits': True,
                'remove_mixed_script': False,
                'preserve_conjuncts': True,
                'handle_virama': True
            }
        
        def clean_kannada_wrapper(text):
            return self.clean_kannada_text(text, **clean_options)
        
        if isinstance(data, list):
            return [clean_kannada_wrapper(text) for text in data]
        elif isinstance(data, pd.DataFrame):
            processed_data = data.copy()
            processed_data[text_column] = processed_data[text_column].apply(clean_kannada_wrapper)
            return processed_data
        else:
            raise TypeError("Data must be either list of strings or pandas DataFrame")
    
    def validate_kannada_text(self, text: str, min_kannada_ratio: float = 0.7) -> bool:
        """
        Validate if text is primarily in Kannada script.
        
        Args:
            text: Text to validate
            min_kannada_ratio: Minimum ratio of Kannada characters required
            
        Returns:
            True if text meets Kannada criteria
        """
        if not text:
            return False
        
        kannada_chars = sum(1 for c in text if self.kannada_unicode_range[0] <= ord(c) <= self.kannada_unicode_range[1])
        total_chars = len(re.sub(r'\s+', '', text))  # Exclude whitespace
        
        if total_chars == 0:
            return False
        
        kannada_ratio = kannada_chars / total_chars
        return kannada_ratio >= min_kannada_ratio


# Utility functions for Kannada text processing
def load_kannada_text_samples() -> List[str]:
    """Load sample Kannada texts for testing and demonstration."""
    return [
        "ಕನ್ನಡ ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಒಂದು ಭಾಷೆಯಾಗಿದೆ.",
        "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿದೆ.",
        "ತಂತ್ರಜ್ಞಾನದ ಪ್ರಗತಿಯು ನಮ್ಮ ಜೀವನವನ್ನು ಬದಲಾಯಿಸಿದೆ.",
        "ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಬಹಳ ಶ್ರೀಮಂತವಾದ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದೆ.",
        "ಶಿಕ್ಷಣವು ಪ್ರತಿಯೊಬ್ಬರಿಗೂ ಮುಖ್ಯವಾದುದು.",
        "ಪ್ರಕೃತಿಯನ್ನು ಸಂರಕ್ಷಿಸುವುದು ನಮ್ಮ ಜವಾಬ್ದಾರಿ.",
        "ಕಲೆ ಮತ್ತು ಸಂಸ್ಕೃತಿಯು ಸಮಾಜದ ಆಧಾರವಾಗಿದೆ.",
        "ಆರೋಗ್ಯಕರ ಆಹಾರ ಮತ್ತು ವ್ಯಾಯಾಮ ಅತ್ಯಗತ್ಯ.",
    ]


def demonstrate_kannada_preprocessing():
    """Demonstrate Kannada preprocessing capabilities."""
    print("🔤 Kannada Preprocessing Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = KannadaPreprocessor()
    
    # Sample texts
    samples = load_kannada_text_samples()
    
    for i, text in enumerate(samples[:3], 1):
        print(f"\n📝 Sample {i}:")
        print(f"Original: {text}")
        
        # Clean text
        cleaned = processor.clean_kannada_text(text)
        print(f"Cleaned: {cleaned}")
        
        # Get statistics
        stats = processor.get_kannada_text_statistics(text)
        print(f"Stats: {stats['kannada_characters']} Kannada chars, "
              f"{stats['vowel_count']} vowels, {stats['consonant_count']} consonants")
        
        # Tokenize
        tokens = processor.tokenize_kannada_words(text)
        print(f"Tokens: {tokens}")
        
        print("-" * 30)


if __name__ == "__main__":
    demonstrate_kannada_preprocessing()
