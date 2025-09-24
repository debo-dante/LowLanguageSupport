"""
Evaluation Module for Indian Language Models

This module provides comprehensive evaluation metrics and benchmarking
tools for assessing Indian language model performance.
"""

from .evaluator import ModelEvaluator
from .metrics import IndianLanguageMetrics
from .benchmarks import BenchmarkSuite
from .cross_lingual_eval import CrossLingualEvaluator

__all__ = [
    'ModelEvaluator',
    'IndianLanguageMetrics',
    'BenchmarkSuite',
    'CrossLingualEvaluator'
]
