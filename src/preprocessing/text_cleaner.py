"""
Text Cleaner for Indian Languages

Handles cleaning, normalization, and preprocessing of Indian language text
with support for multiple scripts and complex morphology.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Union
import pandas as pd
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


class TextCleaner:
    """
    Comprehensive text cleaner for Indian languages.
    
    Handles script normalization, cleaning, and preprocessing
    for various Indian language scripts including Devanagari,
    Bengali, Tamil, Telugu, etc.
    """
    
    def __init__(self, language: str):
        """
        Initialize text cleaner for specific language.
        
        Args:
            language: Language code (hi, bn, ta, te, mr, gu, pa, or, ml, kn, etc.)
        """
        self.language = language
        self.normalizer = IndicNormalizerFactory().get_normalizer(language)
        
        # Language-specific configurations
        self.config = self._get_language_config(language)
        
        # Common patterns for cleaning
        self.unwanted_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # www links
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            r'#\w+',  # Hashtags
            r'@\w+',  # Mentions
            r'\+91[-\s]?\d{10}',  # Indian phone numbers
            r'\d{4}[-\s]?\d{3}[-\s]?\d{3}',  # Phone numbers
        ]
    
    def _get_language_config(self, language: str) -> Dict:
        """Get language-specific configuration."""
        configs = {
            'hi': {
                'script': 'devanagari',
                'digits': '०१२३४५६७८९',
                'punctuation': '।॥',
                'common_prefixes': ['श्री', 'डॉ', 'प्रो'],
                'stop_words_file': 'hindi_stopwords.txt'
            },
            'bn': {
                'script': 'bengali', 
                'digits': '০১২৩৪৫৬৭৮৯',
                'punctuation': '।॥',
                'common_prefixes': ['শ্রী', 'ডাঃ', 'প্রফেসর'],
                'stop_words_file': 'bengali_stopwords.txt'
            },
            'ta': {
                'script': 'tamil',
                'digits': '௦௧௨௩௪௫௬௭௮௯', 
                'punctuation': '।',
                'common_prefixes': ['திரு', 'திருமதி', 'டாக்டர்'],
                'stop_words_file': 'tamil_stopwords.txt'
            },
            'te': {
                'script': 'telugu',
                'digits': '౦౧౨౩౪౫౬౭౮౯',
                'punctuation': '।',
                'common_prefixes': ['శ్రీ', 'డాక్టర్'],
                'stop_words_file': 'telugu_stopwords.txt'
            }
        }
        
        return configs.get(language, {
            'script': 'unknown',
            'digits': '',
            'punctuation': '।',
            'common_prefixes': [],
            'stop_words_file': f'{language}_stopwords.txt'
        })
    
    def clean_text(self, text: str, 
                   normalize_unicode: bool = True,
                   remove_extra_whitespace: bool = True,
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phone_numbers: bool = True,
                   normalize_digits: bool = True,
                   preserve_sentence_boundaries: bool = True) -> str:
        """
        Clean and preprocess Indian language text.
        
        Args:
            text: Input text to clean
            normalize_unicode: Whether to normalize Unicode characters
            remove_extra_whitespace: Whether to remove extra whitespace
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phone_numbers: Whether to remove phone numbers
            normalize_digits: Whether to normalize digits to ASCII
            preserve_sentence_boundaries: Whether to preserve sentence boundaries
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        cleaned_text = text
        
        # Unicode normalization
        if normalize_unicode:
            cleaned_text = unicodedata.normalize('NFC', cleaned_text)
        
        # Remove unwanted patterns
        if remove_urls:
            cleaned_text = re.sub(self.unwanted_patterns[0], '', cleaned_text)
            cleaned_text = re.sub(self.unwanted_patterns[1], '', cleaned_text)
        
        if remove_emails:
            cleaned_text = re.sub(self.unwanted_patterns[2], '', cleaned_text)
        
        if remove_phone_numbers:
            for pattern in self.unwanted_patterns[4:]:
                cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # Remove hashtags and mentions
        cleaned_text = re.sub(self.unwanted_patterns[3], '', cleaned_text)  # hashtags
        cleaned_text = re.sub(self.unwanted_patterns[4], '', cleaned_text)  # mentions
        
        # Normalize using Indic NLP library
        if self.normalizer:
            cleaned_text = self.normalizer.normalize(cleaned_text)
        
        # Normalize digits
        if normalize_digits:
            cleaned_text = self._normalize_digits(cleaned_text)
        
        # Handle punctuation and sentence boundaries
        if preserve_sentence_boundaries:
            cleaned_text = self._normalize_punctuation(cleaned_text)
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _normalize_digits(self, text: str) -> str:
        """Normalize digits to ASCII format."""
        if not self.config.get('digits'):
            return text
        
        # Map native digits to ASCII
        native_digits = self.config['digits']
        ascii_digits = '0123456789'
        
        digit_map = str.maketrans(native_digits, ascii_digits)
        return text.translate(digit_map)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Normalize common punctuation
        punctuation_map = {
            '।': '.',  # Devanagari danda to period
            '॥': '.',   # Double danda to period
            '‍': '',    # Zero-width joiner (sometimes problematic)
            '‌': '',    # Zero-width non-joiner (sometimes problematic)
        }
        
        for native, replacement in punctuation_map.items():
            text = text.replace(native, replacement)
        
        return text
    
    def remove_stop_words(self, text: str, 
                         custom_stop_words: Optional[List[str]] = None) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Input text
            custom_stop_words: Additional stop words to remove
            
        Returns:
            Text with stop words removed
        """
        # Get default stop words for the language
        stop_words = self._get_stop_words()
        
        if custom_stop_words:
            stop_words.extend(custom_stop_words)
        
        # Tokenize and filter
        tokens = indic_tokenize.trivial_tokenize(text, self.language)
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        
        return ' '.join(filtered_tokens)
    
    def _get_stop_words(self) -> List[str]:
        """Get stop words for the language."""
        # Common Hindi stop words as example
        hindi_stop_words = [
            'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'और', 'है', 'हैं', 'था', 'थे', 'थी',
            'एक', 'यह', 'वह', 'जो', 'कि', 'न', 'नहीं', 'तो', 'ही', 'भी', 'अब', 'तक',
            'पर', 'या', 'इस', 'उस', 'अपने', 'कई', 'सब', 'कुछ', 'लिए', 'बाद', 'फिर'
        ]
        
        # Language-specific stop words
        stop_words_dict = {
            'hi': hindi_stop_words,
            'bn': ['এর', 'এক', 'একটি', 'সে', 'তার', 'তাদের', 'আমার', 'আমাদের', 'যে', 'যা', 'কি'],
            'ta': ['அந்த', 'இந்த', 'என்', 'அவர்', 'அவன்', 'அவள்', 'நான்', 'நாம்', 'எந்த', 'என்ன'],
            'te': ['ఆ', 'ఈ', 'ఎవర్', 'అతను', 'ఆమె', 'నేను', 'మేము', 'ఏ', 'ఏమి', 'ఎలా']
        }
        
        return stop_words_dict.get(self.language, [])
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences using language-specific rules.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Use simple sentence splitting (fallback since sentence_split doesn't exist)
        # Split on common sentence endings
        sentence_endings = ['.', '!', '?', '।', '॥']
        sentences = [text]
        for ending in sentence_endings:
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split(ending)
                for i, part in enumerate(parts):
                    if part.strip():
                        if i < len(parts) - 1:  # Add back the ending except for last part
                            new_sentences.append(part.strip() + ending)
                        else:
                            new_sentences.append(part.strip())
            sentences = new_sentences
        
        # Clean each sentence
        cleaned_sentences = []
        for sentence in sentences:
            clean_sentence = self.clean_text(sentence, preserve_sentence_boundaries=False)
            if clean_sentence and len(clean_sentence.strip()) > 5:  # Minimum length
                cleaned_sentences.append(clean_sentence.strip())
        
        return cleaned_sentences
    
    def preprocess_dataset(self, 
                          data: Union[List[str], pd.DataFrame],
                          text_column: str = 'content',
                          batch_size: int = 1000) -> Union[List[str], pd.DataFrame]:
        """
        Preprocess a dataset of text.
        
        Args:
            data: Input data (list of strings or DataFrame)
            text_column: Column name if data is DataFrame
            batch_size: Number of texts to process at once
            
        Returns:
            Preprocessed data in same format as input
        """
        if isinstance(data, list):
            # Process list of strings
            processed_texts = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_processed = [self.clean_text(text) for text in batch]
                processed_texts.extend(batch_processed)
            return processed_texts
        
        elif isinstance(data, pd.DataFrame):
            # Process DataFrame
            if text_column not in data.columns:
                raise ValueError(f"Column '{text_column}' not found in DataFrame")
            
            processed_data = data.copy()
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch_indices = range(i, min(i + batch_size, len(data)))
                batch_texts = data.iloc[batch_indices][text_column].tolist()
                
                batch_processed = [self.clean_text(text) for text in batch_texts]
                processed_data.iloc[batch_indices, processed_data.columns.get_loc(text_column)] = batch_processed
            
            return processed_data
        
        else:
            raise TypeError("Data must be either list of strings or pandas DataFrame")
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with various text statistics
        """
        if not text:
            return {}
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        
        # Count different types of characters
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        digit_chars = len(re.findall(r'[0-9]', text))
        
        # Script-specific character counting
        script_chars = 0
        if self.config['script'] == 'devanagari':
            script_chars = len(re.findall(r'[\u0900-\u097F]', text))
        elif self.config['script'] == 'bengali':
            script_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        elif self.config['script'] == 'tamil':
            script_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        elif self.config['script'] == 'telugu':
            script_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        
        # Sentence count (approximate)
        sentence_count = len(self.sentence_tokenize(text))
        
        return {
            'character_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'latin_characters': latin_chars,
            'digit_characters': digit_chars,
            'script_characters': script_chars,
            'avg_words_per_sentence': word_count / max(sentence_count, 1)
        }
    
    def detect_mixed_script(self, text: str, threshold: float = 0.1) -> bool:
        """
        Detect if text contains mixed scripts.
        
        Args:
            text: Input text
            threshold: Minimum ratio of non-script characters to consider mixed
            
        Returns:
            True if text has mixed scripts above threshold
        """
        if not text:
            return False
        
        stats = self.get_text_statistics(text)
        total_chars = stats['character_count']
        script_chars = stats['script_characters']
        latin_chars = stats['latin_characters']
        
        if total_chars == 0:
            return False
        
        # Calculate ratio of non-script characters
        non_script_ratio = (latin_chars) / total_chars
        
        return non_script_ratio > threshold
