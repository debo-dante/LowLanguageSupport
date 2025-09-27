"""
TokenizerTrainer for Indian Language Models

This module provides functionality to train custom tokenizers
optimized for Indian languages.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Iterator
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """Trainer for custom tokenizers optimized for Indian languages"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tokenizer trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.tokenizer = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration for Indian languages
        return {
            'tokenizer': {
                'type': 'BPE',  # BPE, WordPiece, Unigram
                'vocab_size': 32000,
                'min_frequency': 2,
                'special_tokens': [
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'
                ],
                'clean_text': True,
                'handle_chinese_chars': False,
                'strip_accents': False,
                'lowercase': False,  # Keep original case for Indian languages
                'wordpiece_prefix': '##',
                'continuing_subword_prefix': '##',
                'end_of_word_suffix': '</w>',
            }
        }
    
    def create_tokenizer(self, tokenizer_type: Optional[str] = None) -> Tokenizer:
        """
        Create a new tokenizer
        
        Args:
            tokenizer_type: Type of tokenizer (BPE, WordPiece, Unigram)
            
        Returns:
            Initialized tokenizer
        """
        tokenizer_type = tokenizer_type or self.config['tokenizer']['type']
        
        logger.info(f"Creating {tokenizer_type} tokenizer")
        
        # Initialize model based on type
        if tokenizer_type == 'BPE':
            model = models.BPE(unk_token='[UNK]')
        elif tokenizer_type == 'WordPiece':
            model = models.WordPiece(unk_token='[UNK]')
        elif tokenizer_type == 'Unigram':
            model = models.Unigram()
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        # Create tokenizer
        tokenizer = Tokenizer(model)
        
        # Set normalizer (minimal normalization for Indian languages)
        if self.config['tokenizer']['clean_text']:
            normalizers = []
            if self.config['tokenizer']['strip_accents']:
                normalizers.append(StripAccents())
            if self.config['tokenizer']['lowercase']:
                normalizers.append(Lowercase())
            
            if normalizers:
                from tokenizers.normalizers import Sequence
                tokenizer.normalizer = Sequence(normalizers)
        
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Set decoder
        if tokenizer_type == 'BPE':
            tokenizer.decoder = decoders.BPEDecoder(
                suffix=self.config['tokenizer']['end_of_word_suffix']
            )
        elif tokenizer_type == 'WordPiece':
            tokenizer.decoder = decoders.WordPiece(
                prefix=self.config['tokenizer']['wordpiece_prefix']
            )
        
        return tokenizer
    
    def train_tokenizer(self, texts: List[str], output_dir: str, 
                       tokenizer_type: Optional[str] = None) -> PreTrainedTokenizerFast:
        """
        Train a custom tokenizer
        
        Args:
            texts: List of text samples for training
            output_dir: Directory to save the trained tokenizer
            tokenizer_type: Type of tokenizer to train
            
        Returns:
            Trained tokenizer
        """
        logger.info(f"Training tokenizer on {len(texts)} text samples")
        
        # Create tokenizer
        self.tokenizer = self.create_tokenizer(tokenizer_type)
        
        # Get trainer based on type
        tokenizer_type = tokenizer_type or self.config['tokenizer']['type']
        trainer = self._get_trainer(tokenizer_type)
        
        # Train the tokenizer
        self.tokenizer.train_from_iterator(iter(texts), trainer=trainer)
        
        # Set up post-processor for BERT-like models
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the tokenizer
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        
        # Create PreTrainedTokenizerFast wrapper
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            pad_token='[PAD]',
            unk_token='[UNK]',
            cls_token='[CLS]',
            sep_token='[SEP]',
            mask_token='[MASK]',
        )
        
        # Save the wrapper
        fast_tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, 'tokenizer_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        
        return fast_tokenizer
    
    def _get_trainer(self, tokenizer_type: str):
        """Get appropriate trainer for tokenizer type"""
        config = self.config['tokenizer']
        
        if tokenizer_type == 'BPE':
            return trainers.BpeTrainer(
                vocab_size=config['vocab_size'],
                min_frequency=config['min_frequency'],
                special_tokens=config['special_tokens'],
                end_of_word_suffix=config['end_of_word_suffix'],
                continuing_subword_prefix=config['continuing_subword_prefix'],
            )
        
        elif tokenizer_type == 'WordPiece':
            return trainers.WordPieceTrainer(
                vocab_size=config['vocab_size'],
                min_frequency=config['min_frequency'],
                special_tokens=config['special_tokens'],
                continuing_subword_prefix=config['wordpiece_prefix'],
            )
        
        elif tokenizer_type == 'Unigram':
            return trainers.UnigramTrainer(
                vocab_size=config['vocab_size'],
                special_tokens=config['special_tokens'],
            )
        
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    
    def train_from_files(self, file_paths: List[str], output_dir: str,
                        tokenizer_type: Optional[str] = None) -> PreTrainedTokenizerFast:
        """
        Train tokenizer from text files
        
        Args:
            file_paths: List of paths to text files
            output_dir: Directory to save the trained tokenizer
            tokenizer_type: Type of tokenizer to train
            
        Returns:
            Trained tokenizer
        """
        logger.info(f"Training tokenizer from {len(file_paths)} files")
        
        # Create tokenizer
        self.tokenizer = self.create_tokenizer(tokenizer_type)
        
        # Get trainer
        tokenizer_type = tokenizer_type or self.config['tokenizer']['type']
        trainer = self._get_trainer(tokenizer_type)
        
        # Train from files
        self.tokenizer.train(file_paths, trainer=trainer)
        
        # Set up post-processor
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
        
        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        
        # Create and save wrapper
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            pad_token='[PAD]',
            unk_token='[UNK]',
            cls_token='[CLS]',
            sep_token='[SEP]',
            mask_token='[MASK]',
        )
        
        fast_tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, 'tokenizer_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Tokenizer trained and saved to {output_dir}")
        
        return fast_tokenizer
    
    def evaluate_tokenizer(self, tokenizer_path: str, test_texts: List[str]) -> Dict:
        """
        Evaluate a trained tokenizer
        
        Args:
            tokenizer_path: Path to the trained tokenizer
            test_texts: List of test texts
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating tokenizer on {len(test_texts)} samples")
        
        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Calculate metrics
        total_tokens = 0
        total_chars = 0
        unk_tokens = 0
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)
            total_chars += len(text)
            unk_tokens += tokens.count('[UNK]')
        
        # Calculate averages
        avg_tokens_per_sample = total_tokens / len(test_texts)
        avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        unk_percentage = (unk_tokens / total_tokens) * 100 if total_tokens > 0 else 0
        
        metrics = {
            'total_samples': len(test_texts),
            'total_tokens': total_tokens,
            'total_characters': total_chars,
            'avg_tokens_per_sample': avg_tokens_per_sample,
            'avg_chars_per_token': avg_chars_per_token,
            'unk_tokens': unk_tokens,
            'unk_percentage': unk_percentage,
            'vocab_size': tokenizer.vocab_size,
        }
        
        logger.info(f"Tokenizer evaluation complete: {metrics}")
        
        return metrics
    
    def load_tokenizer(self, tokenizer_path: str) -> PreTrainedTokenizerFast:
        """
        Load a trained tokenizer
        
        Args:
            tokenizer_path: Path to the trained tokenizer
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        # Load config if available
        config_path = os.path.join(tokenizer_path, 'tokenizer_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        logger.info("Tokenizer loaded successfully")
        
        return tokenizer