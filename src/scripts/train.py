#!/usr/bin/env python3
"""
Training Script for Indian Language Models

Command-line interface for training language models
on Indian language datasets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelTrainer
from data_collection import KannadaDataCollector
from preprocessing import KannadaPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train Indian Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --language kannada --data-dir data/kannada/
  %(prog)s --config configs/kannada_config.yaml --epochs 10
  %(prog)s --language hindi --model-type bert --vocab-size 50000
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        help='Target language (e.g., kannada, hindi, tamil)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type',
        type=str,
        default='bert',
        choices=['bert', 'roberta', 'gpt2', 'custom'],
        help='Type of model to train (default: bert)'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='Vocabulary size (default: 32000)'
    )
    
    # Training arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='Directory to save trained model (default: models/trained)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size (default: 16)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    
    # Data arguments
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of training samples to use'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
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
    
    logger.info(f"Starting training for {args.language} language")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(config_path=args.config)
        
        # Setup model
        trainer.setup_model(f"bert-base-multilingual-cased")
        
        # Prepare data
        texts = []
        
        if args.data_dir:
            # Load data from directory
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return 1
            
            # Load text files
            for text_file in data_dir.glob("*.txt"):
                logger.info(f"Loading {text_file}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    texts.extend(f.readlines())
        else:
            # Use data collector to gather sample data
            logger.info("No data directory specified, collecting sample data...")
            
            if args.language == 'kannada':
                collector = KannadaDataCollector()
                sample_data = collector.collect_sample_data()
                texts = [item['text'] for item in sample_data]
            else:
                logger.error(f"No sample data available for {args.language}")
                return 1
        
        # Clean texts
        texts = [text.strip() for text in texts if text.strip()]
        
        if args.max_samples:
            texts = texts[:args.max_samples]
        
        logger.info(f"Using {len(texts)} training samples")
        
        # Prepare datasets
        train_size = int(len(texts) * (1 - args.val_split))
        train_texts = texts[:train_size]
        val_texts = texts[train_size:]
        
        train_dataset = trainer.prepare_data(train_texts)
        val_dataset = trainer.prepare_data(val_texts) if val_texts else None
        
        # Train model
        logger.info("Starting model training...")
        trainer.train(train_dataset, val_dataset)
        
        # Save model
        output_path = Path(args.output_dir) / f"{args.language}_{args.model_type}"
        trainer.save_model(str(output_path))
        
        logger.info(f"Training completed! Model saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())