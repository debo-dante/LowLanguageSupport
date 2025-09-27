"""
FineTuner for Indian Language Models

This module provides fine-tuning functionality for pre-trained models
on specific Indian language tasks.
"""

import os
import yaml
import torch
import logging
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class FineTuner:
    """Fine-tuner for Indian Language Models"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the fine-tuner
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.task_type = 'classification'  # classification, ner, qa
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model': {
                'name': 'bert-base-multilingual-cased',
                'num_labels': 2,
                'max_length': 512,
            },
            'training': {
                'output_dir': 'models/finetuned',
                'per_device_train_batch_size': 8,
                'per_device_eval_batch_size': 8,
                'num_train_epochs': 5,
                'learning_rate': 2e-5,
                'warmup_steps': 100,
                'weight_decay': 0.01,
                'logging_steps': 50,
                'save_steps': 500,
                'eval_steps': 500,
                'evaluation_strategy': 'steps',
                'save_total_limit': 2,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_accuracy',
                'greater_is_better': True,
                'dataloader_num_workers': 2,
                'fp16': True,
                'early_stopping_patience': 3,
            }
        }
    
    def setup_model(self, model_name: Optional[str] = None, num_labels: Optional[int] = None):
        """
        Setup model and tokenizer for fine-tuning
        
        Args:
            model_name: Name of pre-trained model to use
            num_labels: Number of labels for classification
        """
        model_name = model_name or self.config['model']['name']
        num_labels = num_labels or self.config['model']['num_labels']
        
        logger.info(f"Loading model for fine-tuning: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_labels
            )
            
            logger.info(f"Model setup complete. Labels: {num_labels}")
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def prepare_classification_data(self, texts: List[str], labels: List[int], 
                                   max_length: Optional[int] = None) -> Dataset:
        """
        Prepare classification data
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            max_length: Maximum sequence length
            
        Returns:
            Prepared dataset
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call setup_model() first.")
        
        max_length = max_length or self.config['model']['max_length']
        
        logger.info(f"Preparing {len(texts)} samples for classification")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Fine-tune the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Setup training arguments
        training_args = TrainingArguments(**self.config['training'])
        
        # Setup callbacks
        callbacks = []
        if 'early_stopping_patience' in self.config['training']:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['training']['early_stopping_patience']
                )
            )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        logger.info("Starting fine-tuning...")
        
        try:
            # Fine-tune the model
            self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(training_args.output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {training_args.output_dir}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise
    
    def evaluate(self, eval_dataset: Dataset) -> Dict:
        """
        Evaluate the fine-tuned model
        
        Args:
            eval_dataset: Evaluation dataset
            
        Returns:
            Evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Run fine_tune() first.")
        
        logger.info("Evaluating fine-tuned model...")
        results = self.trainer.evaluate(eval_dataset)
        
        return results
    
    def predict(self, texts: List[str], max_length: Optional[int] = None) -> List[int]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of text samples
            max_length: Maximum sequence length
            
        Returns:
            List of predicted labels
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")
        
        max_length = max_length or self.config['model']['max_length']
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1).cpu().numpy()
        
        return predicted_labels.tolist()
    
    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model
        
        Args:
            output_dir: Directory to save the model
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, 'finetuner_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Fine-tuned model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """
        Load a fine-tuned model
        
        Args:
            model_dir: Directory containing the saved model
        """
        logger.info(f"Loading fine-tuned model from {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Load config if available
        config_path = os.path.join(model_dir, 'finetuner_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        logger.info("Fine-tuned model loaded successfully")