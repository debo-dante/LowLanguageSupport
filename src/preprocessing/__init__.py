"""
Preprocessing Module for Indian Languages

This module provides comprehensive text preprocessing capabilities
specifically designed for Indian languages and scripts.
"""

from .text_cleaner import TextCleaner
from .kannada_preprocessor import KannadaPreprocessor
# TODO: Add these modules when implemented
# from .tokenizer import IndianLanguageTokenizer  
# from .normalizer import ScriptNormalizer
# from .augmentation import TextAugmentor

__all__ = [
    'TextCleaner',
    'KannadaPreprocessor',
    # 'IndianLanguageTokenizer',
    # 'ScriptNormalizer', 
    # 'TextAugmentor'
]
