#!/usr/bin/env python3
"""
Evaluation Script for Indian Language Models

Command-line interface for evaluating trained language models
on various NLP tasks.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import IndianLanguageModel
from evaluation import ModelEvaluator
from preprocessing import KannadaPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description='Evaluate Indian Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model models/kannada_bert --test-data data/test/kannada.txt
  %(prog)s --model models/hindi_roberta --task classification --test-data data/hindi_sentiment.json
  %(prog)s --model models/tamil_gpt2 --task generation --prompt "தமிழ் மொழி"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model directory'
    )
    
    # Task arguments
    parser.add_argument(
        '--task',
        type=str,
        choices=['language_modeling', 'classification', 'generation', 'embedding'],
        default='language_modeling',
        help='Evaluation task (default: language_modeling)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data file'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        help='Target language for evaluation'
    )
    
    # Generation task arguments
    parser.add_argument(
        '--prompt',
        type=str,
        help='Text prompt for generation task'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum generation length (default: 100)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of generation samples (default: 5)'
    )
    
    # Classification task arguments
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='Classification labels'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Evaluation batch size (default: 32)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['accuracy', 'f1', 'precision', 'recall'],
        help='Evaluation metrics to compute'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for evaluation results'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU if available'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    
    try:
        # Load model
        logger.info("Loading model...")
        if Path(args.model).exists() and (Path(args.model) / "model_config.json").exists():
            # Load custom IndianLanguageModel
            model = IndianLanguageModel.load_model(args.model)
        else:
            # Try loading as standard model
            logger.error(f"Model not found or not a valid IndianLanguageModel: {args.model}")
            return 1
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model)
        
        results = {}
        
        if args.task == 'language_modeling':
            if not args.test_data:
                logger.error("--test-data required for language modeling evaluation")
                return 1
            
            # Load test data
            test_texts = []
            with open(args.test_data, 'r', encoding='utf-8') as f:
                test_texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Evaluating language modeling on {len(test_texts)} samples")
            
            # Compute perplexity and other language modeling metrics
            results = evaluator.evaluate_language_modeling(test_texts)
            
        elif args.task == 'embedding':
            if not args.test_data:
                # Use sample texts for embedding evaluation
                test_texts = [
                    "This is a sample sentence for embedding evaluation.",
                    "Another example sentence in the target language.",
                    "Testing the quality of text embeddings."
                ]
            else:
                with open(args.test_data, 'r', encoding='utf-8') as f:
                    test_texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Generating embeddings for {len(test_texts)} samples")
            
            # Generate embeddings
            embeddings = []
            for text in test_texts:
                embedding = model.get_embeddings(text, language=args.language)
                embeddings.append(embedding.numpy())
            
            results = {
                'num_samples': len(test_texts),
                'embedding_dim': embeddings[0].shape[-1],
                'sample_embeddings': [emb.tolist()[:5] for emb in embeddings[:3]]  # First 5 dims of first 3 samples
            }
            
        elif args.task == 'generation':
            if not args.prompt:
                logger.error("--prompt required for generation task")
                return 1
            
            logger.info(f"Generating text from prompt: '{args.prompt}'")
            
            # Text generation is not directly implemented in our current model
            # This would require adding generation capabilities
            logger.info("Text generation not implemented yet")
            results = {
                'prompt': args.prompt,
                'note': 'Generation functionality to be implemented'
            }
            
        elif args.task == 'classification':
            logger.info("Classification evaluation not implemented yet")
            results = {
                'note': 'Classification evaluation to be implemented'
            }
        
        # Print results
        logger.info("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
        
        logger.info("Evaluation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())