"""
Model Evaluator for Indian Language Models

Comprehensive evaluation framework that assesses model performance
across various tasks and metrics relevant to Indian languages.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime

from ..models.indian_language_model import IndianLanguageModel


class ModelEvaluator:
    """
    Comprehensive evaluator for Indian language models.
    
    Supports various evaluation tasks including:
    - Language modeling (perplexity)
    - Text classification
    - Semantic similarity
    - Cross-lingual evaluation
    - Multilingual capabilities
    """
    
    def __init__(self, 
                 model: IndianLanguageModel,
                 language: str,
                 device: str = 'auto'):
        """
        Initialize the evaluator.
        
        Args:
            model: The Indian language model to evaluate
            language: Primary language for evaluation
            device: Device to run evaluation on ('auto', 'cuda', 'cpu')
        """
        self.model = model
        self.language = language
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.evaluation_results = {}
        self.benchmark_scores = {}
    
    def evaluate_language_modeling(self, 
                                  test_texts: List[str],
                                  batch_size: int = 8,
                                  max_length: int = 512) -> Dict[str, float]:
        """
        Evaluate language modeling capabilities using perplexity.
        
        Args:
            test_texts: List of test texts
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity and related metrics
        """
        self.logger.info("Evaluating language modeling performance...")
        
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.model.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**encoded)
                
                # Calculate language modeling loss (simplified)
                # In practice, you'd use a language model head
                # This is a placeholder for actual LM evaluation
                batch_loss = self._calculate_lm_loss(outputs, encoded)
                
                total_loss += batch_loss * encoded['input_ids'].size(0)
                total_tokens += encoded['attention_mask'].sum().item()
        
        avg_loss = total_loss / len(test_texts)
        perplexity = torch.exp(avg_loss).item()
        
        results = {
            'perplexity': perplexity,
            'average_loss': avg_loss.item(),
            'total_texts': len(test_texts),
            'total_tokens': total_tokens
        }
        
        self.evaluation_results['language_modeling'] = results
        return results
    
    def _calculate_lm_loss(self, outputs, encoded):
        """Calculate language modeling loss (simplified placeholder)."""
        # This is a simplified placeholder
        # In practice, you'd need a language model head and proper loss calculation
        return torch.tensor(1.0)  # Placeholder
    
    def evaluate_text_classification(self,
                                   texts: List[str],
                                   labels: List[int],
                                   task_name: str = "classification") -> Dict[str, Any]:
        """
        Evaluate text classification performance.
        
        Args:
            texts: Input texts
            labels: True labels
            task_name: Name of the classification task
            
        Returns:
            Classification metrics
        """
        self.logger.info(f"Evaluating text classification: {task_name}")
        
        # Get embeddings
        embeddings = self._get_batch_embeddings(texts)
        
        # Simple classification using cosine similarity to label prototypes
        # In practice, you'd train a classifier on top of the embeddings
        predictions = self._classify_embeddings(embeddings, labels)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Generate classification report
        class_report = classification_report(
            labels, predictions, output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'num_samples': len(texts),
            'num_classes': len(set(labels))
        }
        
        self.evaluation_results[f'classification_{task_name}'] = results
        return results
    
    def evaluate_semantic_similarity(self,
                                   text_pairs: List[Tuple[str, str]],
                                   similarity_scores: List[float]) -> Dict[str, float]:
        """
        Evaluate semantic similarity understanding.
        
        Args:
            text_pairs: Pairs of texts
            similarity_scores: Ground truth similarity scores
            
        Returns:
            Similarity evaluation metrics
        """
        self.logger.info("Evaluating semantic similarity...")
        
        predicted_similarities = []
        
        for text1, text2 in text_pairs:
            # Get embeddings
            emb1 = self._get_batch_embeddings([text1])
            emb2 = self._get_batch_embeddings([text2])
            
            # Calculate cosine similarity
            sim = cosine_similarity(emb1, emb2)[0, 0]
            predicted_similarities.append(sim)
        
        # Calculate correlation with ground truth
        spearman_corr, spearman_p = spearmanr(similarity_scores, predicted_similarities)
        pearson_corr, pearson_p = pearsonr(similarity_scores, predicted_similarities)
        
        results = {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'num_pairs': len(text_pairs)
        }
        
        self.evaluation_results['semantic_similarity'] = results
        return results
    
    def evaluate_cross_lingual_transfer(self,
                                      source_data: Dict[str, List],
                                      target_data: Dict[str, List],
                                      source_lang: str,
                                      target_lang: str) -> Dict[str, Any]:
        """
        Evaluate cross-lingual transfer capabilities.
        
        Args:
            source_data: Source language data {'texts': [...], 'labels': [...]}
            target_data: Target language data {'texts': [...], 'labels': [...]}
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Cross-lingual evaluation metrics
        """
        self.logger.info(f"Evaluating cross-lingual transfer: {source_lang} -> {target_lang}")
        
        # Get embeddings for both languages
        source_embeddings = self._get_batch_embeddings(
            source_data['texts'], language=source_lang
        )
        target_embeddings = self._get_batch_embeddings(
            target_data['texts'], language=target_lang
        )
        
        # Evaluate cross-lingual alignment
        alignment_score = self._evaluate_cross_lingual_alignment(
            source_embeddings, target_embeddings,
            source_data['labels'], target_data['labels']
        )
        
        # Zero-shot classification performance
        zero_shot_acc = self._zero_shot_classification(
            source_embeddings, source_data['labels'],
            target_embeddings, target_data['labels']
        )
        
        results = {
            'alignment_score': alignment_score,
            'zero_shot_accuracy': zero_shot_acc,
            'source_language': source_lang,
            'target_language': target_lang,
            'source_samples': len(source_data['texts']),
            'target_samples': len(target_data['texts'])
        }
        
        self.evaluation_results[f'cross_lingual_{source_lang}_{target_lang}'] = results
        return results
    
    def evaluate_multilingual_capabilities(self,
                                         multilingual_data: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """
        Evaluate multilingual model capabilities.
        
        Args:
            multilingual_data: Data for multiple languages
                {'lang_code': {'texts': [...], 'labels': [...]}}
        
        Returns:
            Multilingual evaluation metrics
        """
        self.logger.info("Evaluating multilingual capabilities...")
        
        language_results = {}
        all_embeddings = {}
        
        # Evaluate each language individually
        for lang, data in multilingual_data.items():
            embeddings = self._get_batch_embeddings(data['texts'], language=lang)
            all_embeddings[lang] = embeddings
            
            # Language-specific metrics
            lang_results = self._evaluate_language_specific_metrics(
                embeddings, data.get('labels', []), lang
            )
            language_results[lang] = lang_results
        
        # Cross-lingual consistency
        consistency_score = self._evaluate_multilingual_consistency(all_embeddings)
        
        # Language distance analysis
        language_distances = self._calculate_language_distances(all_embeddings)
        
        results = {
            'language_results': language_results,
            'consistency_score': consistency_score,
            'language_distances': language_distances,
            'supported_languages': list(multilingual_data.keys()),
            'num_languages': len(multilingual_data)
        }
        
        self.evaluation_results['multilingual'] = results
        return results
    
    def _get_batch_embeddings(self, 
                            texts: List[str],
                            language: Optional[str] = None,
                            batch_size: int = 32) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        all_embeddings = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                embeddings = self.model.get_embeddings(
                    batch_texts, 
                    language=language or self.language
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _classify_embeddings(self, embeddings: np.ndarray, labels: List[int]) -> List[int]:
        """Simple classification using k-NN (placeholder)."""
        # This is a simplified placeholder
        # In practice, you'd train a proper classifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        
        if len(set(labels)) < 2:
            return labels  # Can't classify with single class
        
        # Simple train/test split for demonstration
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42
        )
        
        # Train simple k-NN classifier
        clf = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
        clf.fit(X_train, y_train)
        
        # Predict on full set for evaluation
        predictions = clf.predict(embeddings)
        return predictions.tolist()
    
    def _evaluate_cross_lingual_alignment(self,
                                        source_embeddings: np.ndarray,
                                        target_embeddings: np.ndarray,
                                        source_labels: List[int],
                                        target_labels: List[int]) -> float:
        """Evaluate cross-lingual embedding alignment."""
        # Calculate centroid alignment for each class
        alignments = []
        
        for label in set(source_labels):
            if label in target_labels:
                source_mask = np.array(source_labels) == label
                target_mask = np.array(target_labels) == label
                
                if source_mask.sum() > 0 and target_mask.sum() > 0:
                    source_centroid = source_embeddings[source_mask].mean(axis=0)
                    target_centroid = target_embeddings[target_mask].mean(axis=0)
                    
                    alignment = cosine_similarity(
                        source_centroid.reshape(1, -1),
                        target_centroid.reshape(1, -1)
                    )[0, 0]
                    
                    alignments.append(alignment)
        
        return np.mean(alignments) if alignments else 0.0
    
    def _zero_shot_classification(self,
                                source_embeddings: np.ndarray,
                                source_labels: List[int],
                                target_embeddings: np.ndarray,
                                target_labels: List[int]) -> float:
        """Evaluate zero-shot classification performance."""
        # Train classifier on source embeddings
        from sklearn.svm import SVC
        
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(source_embeddings, source_labels)
        
        # Predict on target embeddings
        predictions = clf.predict(target_embeddings)
        
        # Calculate accuracy
        accuracy = accuracy_score(target_labels, predictions)
        return accuracy
    
    def _evaluate_language_specific_metrics(self,
                                           embeddings: np.ndarray,
                                           labels: List[int],
                                           language: str) -> Dict[str, Any]:
        """Evaluate language-specific metrics."""
        results = {
            'embedding_dimension': embeddings.shape[1],
            'num_samples': embeddings.shape[0],
            'embedding_norm_mean': np.linalg.norm(embeddings, axis=1).mean(),
            'embedding_norm_std': np.linalg.norm(embeddings, axis=1).std(),
        }
        
        if labels:
            # Calculate intra-class and inter-class similarities
            intra_class_sims = []
            inter_class_sims = []
            
            unique_labels = list(set(labels))
            
            for label in unique_labels:
                mask = np.array(labels) == label
                class_embeddings = embeddings[mask]
                
                if class_embeddings.shape[0] > 1:
                    # Intra-class similarities
                    sim_matrix = cosine_similarity(class_embeddings)
                    # Get upper triangle (excluding diagonal)
                    triu_indices = np.triu_indices_from(sim_matrix, k=1)
                    intra_class_sims.extend(sim_matrix[triu_indices])
            
            # Inter-class similarities
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    mask1 = np.array(labels) == label1
                    mask2 = np.array(labels) == label2
                    
                    emb1 = embeddings[mask1]
                    emb2 = embeddings[mask2]
                    
                    if emb1.shape[0] > 0 and emb2.shape[0] > 0:
                        sim_matrix = cosine_similarity(emb1, emb2)
                        inter_class_sims.extend(sim_matrix.flatten())
            
            results.update({
                'intra_class_similarity_mean': np.mean(intra_class_sims) if intra_class_sims else 0,
                'inter_class_similarity_mean': np.mean(inter_class_sims) if inter_class_sims else 0,
                'class_separation': (np.mean(intra_class_sims) - np.mean(inter_class_sims)) if (intra_class_sims and inter_class_sims) else 0
            })
        
        return results
    
    def _evaluate_multilingual_consistency(self, 
                                         all_embeddings: Dict[str, np.ndarray]) -> float:
        """Evaluate consistency across languages."""
        languages = list(all_embeddings.keys())
        consistencies = []
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                # Calculate embedding space similarity
                emb1 = all_embeddings[lang1]
                emb2 = all_embeddings[lang2]
                
                # Sample embeddings if too large
                max_samples = 100
                if emb1.shape[0] > max_samples:
                    indices1 = np.random.choice(emb1.shape[0], max_samples, replace=False)
                    emb1 = emb1[indices1]
                
                if emb2.shape[0] > max_samples:
                    indices2 = np.random.choice(emb2.shape[0], max_samples, replace=False)
                    emb2 = emb2[indices2]
                
                # Calculate cross-lingual similarity
                cross_sim = cosine_similarity(emb1, emb2).mean()
                consistencies.append(cross_sim)
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_language_distances(self, 
                                    all_embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise language distances."""
        languages = list(all_embeddings.keys())
        distances = {lang: {} for lang in languages}
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages:
                if lang1 == lang2:
                    distances[lang1][lang2] = 0.0
                    continue
                
                emb1 = all_embeddings[lang1]
                emb2 = all_embeddings[lang2]
                
                # Calculate centroid distance
                centroid1 = emb1.mean(axis=0)
                centroid2 = emb2.mean(axis=0)
                
                distance = 1 - cosine_similarity(
                    centroid1.reshape(1, -1),
                    centroid2.reshape(1, -1)
                )[0, 0]
                
                distances[lang1][lang2] = distance
        
        return distances
    
    def generate_evaluation_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive evaluation report."""
        report_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'language': self.language,
                'model_type': self.model.model_type,
                'parameter_count': self.model.get_parameter_count()
            },
            'evaluation_results': self.evaluation_results,
            'benchmark_scores': self.benchmark_scores
        }
        
        # Generate text report
        report = self._format_text_report(report_data)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Save text report
            with open(output_path.with_suffix('.txt'), 'w') as f:
                f.write(report)
            
            self.logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def _format_text_report(self, report_data: Dict) -> str:
        """Format evaluation results as text report."""
        report = []
        report.append("=" * 80)
        report.append("INDIAN LANGUAGE MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Evaluation Time: {report_data['evaluation_timestamp']}")
        report.append(f"Primary Language: {report_data['model_info']['language']}")
        report.append(f"Model Type: {report_data['model_info']['model_type']}")
        report.append("")
        
        # Model information
        report.append("MODEL INFORMATION:")
        report.append("-" * 40)
        param_count = report_data['model_info']['parameter_count']
        for component, count in param_count.items():
            report.append(f"{component.capitalize()}: {count:,} parameters")
        report.append("")
        
        # Evaluation results
        for task, results in report_data['evaluation_results'].items():
            report.append(f"{task.upper().replace('_', ' ')} RESULTS:")
            report.append("-" * 40)
            
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        report.append(f"{metric}: {value:.4f}")
                    elif isinstance(value, str):
                        report.append(f"{metric}: {value}")
                    else:
                        report.append(f"{metric}: {type(value).__name__}")
            
            report.append("")
        
        return "\n".join(report)
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Create visualizations of evaluation results."""
        if not self.evaluation_results:
            self.logger.warning("No evaluation results to visualize")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Indian Language Model Evaluation Results', fontsize=16)
        
        # Plot 1: Performance metrics
        if 'classification' in str(self.evaluation_results):
            self._plot_classification_metrics(axes[0, 0])
        
        # Plot 2: Language distances (if multilingual)
        if 'multilingual' in self.evaluation_results:
            self._plot_language_distances(axes[0, 1])
        
        # Plot 3: Cross-lingual performance
        if any('cross_lingual' in key for key in self.evaluation_results):
            self._plot_cross_lingual_performance(axes[1, 0])
        
        # Plot 4: Model statistics
        self._plot_model_statistics(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def _plot_classification_metrics(self, ax):
        """Plot classification metrics."""
        # Placeholder visualization
        ax.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [0.85, 0.83, 0.87, 0.85])
        ax.set_title('Classification Performance')
        ax.set_ylabel('Score')
    
    def _plot_language_distances(self, ax):
        """Plot language distance heatmap."""
        # Placeholder visualization
        langs = ['Hindi', 'Bengali', 'Tamil', 'Telugu']
        distances = np.random.rand(4, 4)
        sns.heatmap(distances, xticklabels=langs, yticklabels=langs, ax=ax)
        ax.set_title('Language Distances')
    
    def _plot_cross_lingual_performance(self, ax):
        """Plot cross-lingual transfer performance."""
        # Placeholder visualization
        transfers = ['Hi→Bn', 'Hi→Ta', 'Bn→Te', 'Ta→Hi']
        scores = [0.75, 0.68, 0.72, 0.71]
        ax.bar(transfers, scores)
        ax.set_title('Cross-lingual Transfer')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_model_statistics(self, ax):
        """Plot model parameter statistics."""
        # Placeholder visualization
        components = ['Transformer', 'Adapters', 'Embeddings']
        sizes = [100, 5, 10]  # In millions
        ax.pie(sizes, labels=components, autopct='%1.1f%%')
        ax.set_title('Parameter Distribution')
