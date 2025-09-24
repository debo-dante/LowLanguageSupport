"""
Data Collection Module for Indian Languages

This module provides tools and utilities for collecting text data
for underrepresented Indian languages from various sources.
"""

from .web_scraper import WebScraper
from .kannada_collector import KannadaDataCollector
# TODO: Add these modules when implemented
# from .corpus_collector import CorpusCollector
# from .language_detector import LanguageDetector
# from .data_validator import DataValidator

__all__ = [
    'WebScraper',
    'KannadaDataCollector',
    # 'CorpusCollector', 
    # 'LanguageDetector',
    # 'DataValidator'
]
