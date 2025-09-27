"""
ModelTrainer for Indian Language Models

This module provides training functionality for transformer-based 
language models optimized for Indian languages.
"""

import os
import yaml
import torch
import logging
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for Indian Language Models"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model': {
                'name': 'bert-base-multilingual-cased',
                'vocab_size': 32000,
                'hidden_size': 768,
                'num_attention_heads': 12,
                'num_hidden_layers': 12,
                'intermediate_size': 3072,
                'max_position_embeddings': 512,
            },
            'training': {
                'output_dir': 'models/checkpoint',
                'per_device_train_batch_size': 16,
                'per_device_eval_batch_size': 16,
                'num_train_epochs': 3,
                'learning_rate': 2e-5,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'logging_steps': 100,
                'save_steps': 1000,
                'eval_steps': 1000,
                'evaluation_strategy': 'steps',
                'save_total_limit': 3,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'loss',
                'greater_is_better': False,
                'dataloader_num_workers': 4,
                'fp16': True,
            }
        }
    
    def setup_model(self, model_name: Optional[str] = None):
        """
        Setup model and tokenizer
        
        Args:
            model_name: Name of pre-trained model to use
        """
        model_name = model_name or self.config['model']['name']
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = {
                'pad_token': '[PAD]',
                'cls_token': '[CLS]', 
                'sep_token': '[SEP]',
                'mask_token': '[MASK]',
                'unk_token': '[UNK]'
            }
            
            num_added = self.tokenizer.add_special_tokens(special_tokens)
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                
            logger.info(f"Model setup complete. Vocab size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def prepare_data(self, texts: List[str], max_length: int = 512) -> Dataset:
        """
        Prepare training data
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            Prepared dataset
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")
        
        logger.info(f"Preparing {len(texts)} text samples")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True
            )
        
        # Create dataset
        dataset = Dataset.from_dict({'text': texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Create data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Setup training arguments
        training_args = TrainingArguments(**self.config['training'])
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Starting training...")
        
        try:
            # Train the model
            self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            logger.info(f"Training completed. Model saved to {training_args.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict:
        """
        Evaluate the model
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        logger.info("Evaluating model...")
        results = self.trainer.evaluate(eval_dataset)
        
        return results
    
    def save_model(self, output_dir: str):
        """
        Save the trained model
        
        Args:
            output_dir: Directory to save the model
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, 'trainer_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """
        Load a trained model
        
        Args:
            model_dir: Directory containing the saved model
        """
        logger.info(f"Loading model from {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        
        # Load config if available
        config_path = os.path.join(model_dir, 'trainer_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        logger.info("Model loaded successfully")