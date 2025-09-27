"""
Metrics Module for Indian Language Models

Provides specialized metrics for evaluating NLP models
on Indian language tasks.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


class IndianLanguageMetrics:
    """Specialized metrics for Indian language model evaluation"""
    
    def __init__(self):
        self.supported_languages = [
            'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'pa', 'or', 'ml', 'kn', 'as'
        ]
    
    def compute_classification_metrics(self, y_true: List[int], y_pred: List[int], 
                                     labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names for reporting
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add per-class metrics if labels provided
        if labels:
            report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
            metrics['per_class'] = report
        
        return metrics
    
    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from loss
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity value
        """
        return np.exp(loss)
    
    def compute_bleu_score(self, references: List[str], candidates: List[str]) -> float:
        """
        Compute BLEU score (placeholder implementation)
        
        Args:
            references: Reference texts
            candidates: Generated texts
            
        Returns:
            BLEU score
        """
        # Simplified BLEU implementation
        # In practice, you'd use libraries like nltk.translate.bleu_score
        total_score = 0.0
        for ref, cand in zip(references, candidates):
            ref_words = ref.split()
            cand_words = cand.split()
            
            if not cand_words:
                continue
                
            # Simple word overlap score
            overlap = len(set(ref_words) & set(cand_words))
            score = overlap / len(cand_words) if cand_words else 0.0
            total_score += score
        
        return total_score / len(references) if references else 0.0
    
    def compute_semantic_similarity(self, embeddings1: np.ndarray, 
                                   embeddings2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Average cosine similarity
        """
        similarities = []
        
        for emb1, emb2 in zip(embeddings1, embeddings2):
            # Normalize embeddings
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarities.append(0.0)
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                similarities.append(similarity)
        
        return np.mean(similarities)
    
    def evaluate_language_modeling(self, model, texts: List[str], 
                                 tokenizer) -> Dict[str, float]:
        """
        Evaluate language modeling performance
        
        Args:
            model: Language model
            texts: Test texts
            tokenizer: Tokenizer
            
        Returns:
            Language modeling metrics
        """
        total_loss = 0.0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                
                if hasattr(model, 'transformer'):
                    # For our custom model, we need to compute loss manually
                    # This is a simplified implementation
                    outputs = model(**inputs)
                    # Placeholder loss calculation
                    loss = 0.0
                else:
                    # For standard language models
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                
                total_loss += loss
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / len(texts)
        perplexity = self.compute_perplexity(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens
        }
    
    def compute_cross_lingual_similarity(self, embeddings_lang1: np.ndarray,
                                       embeddings_lang2: np.ndarray) -> Dict[str, float]:
        """
        Compute cross-lingual similarity metrics
        
        Args:
            embeddings_lang1: Embeddings for language 1
            embeddings_lang2: Embeddings for language 2
            
        Returns:
            Cross-lingual metrics
        """
        # Compute pairwise similarities
        similarities = []
        for emb1, emb2 in zip(embeddings_lang1, embeddings_lang2):
            sim = self.compute_semantic_similarity([emb1], [emb2])
            similarities.append(sim)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def evaluate_script_awareness(self, model, texts_by_script: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate model's script awareness
        
        Args:
            model: Language model
            texts_by_script: Dictionary mapping script names to texts
            
        Returns:
            Script awareness metrics
        """
        script_metrics = {}
        
        for script, texts in texts_by_script.items():
            # Generate embeddings for each script
            embeddings = []
            for text in texts:
                emb = model.get_embeddings(text)
                embeddings.append(emb.numpy())
            
            embeddings = np.array(embeddings)
            
            # Compute intra-script similarity
            intra_similarities = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    sim = self.compute_semantic_similarity([embeddings[i]], [embeddings[j]])
                    intra_similarities.append(sim)
            
            script_metrics[script] = {
                'intra_script_similarity': np.mean(intra_similarities) if intra_similarities else 0.0,
                'embedding_variance': np.var(embeddings.flatten()) if len(embeddings) > 0 else 0.0,
                'num_samples': len(texts)
            }
        
        return script_metrics
    
    def generate_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Formatted evaluation report
        """
        report = "📊 Indian Language Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Classification metrics
        if 'classification' in metrics:
            report += "🎯 Classification Performance:\n"
            cls_metrics = metrics['classification']
            report += f"  Accuracy: {cls_metrics.get('accuracy', 0):.3f}\n"
            report += f"  Precision: {cls_metrics.get('precision', 0):.3f}\n"
            report += f"  Recall: {cls_metrics.get('recall', 0):.3f}\n"
            report += f"  F1 Score: {cls_metrics.get('f1', 0):.3f}\n\n"
        
        # Language modeling metrics
        if 'language_modeling' in metrics:
            report += "📝 Language Modeling Performance:\n"
            lm_metrics = metrics['language_modeling']
            report += f"  Loss: {lm_metrics.get('loss', 0):.3f}\n"
            report += f"  Perplexity: {lm_metrics.get('perplexity', 0):.2f}\n"
            report += f"  Total Tokens: {lm_metrics.get('total_tokens', 0):,}\n\n"
        
        # Cross-lingual metrics
        if 'cross_lingual' in metrics:
            report += "🌍 Cross-lingual Performance:\n"
            xl_metrics = metrics['cross_lingual']
            report += f"  Mean Similarity: {xl_metrics.get('mean_similarity', 0):.3f}\n"
            report += f"  Std Similarity: {xl_metrics.get('std_similarity', 0):.3f}\n\n"
        
        # Script awareness
        if 'script_awareness' in metrics:
            report += "📜 Script Awareness:\n"
            for script, script_metrics in metrics['script_awareness'].items():
                report += f"  {script}:\n"
                report += f"    Intra-script similarity: {script_metrics.get('intra_script_similarity', 0):.3f}\n"
                report += f"    Samples: {script_metrics.get('num_samples', 0)}\n"
            report += "\n"
        
        report += "✅ Evaluation Complete!"
        
        return report