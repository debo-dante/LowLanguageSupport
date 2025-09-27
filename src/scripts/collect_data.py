#!/usr/bin/env python3
"""
Data Collection Script for Indian Languages

Command-line interface for collecting and preprocessing
text data for Indian languages.
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

from data_collection import KannadaDataCollector
from preprocessing import KannadaPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main data collection function"""
    parser = argparse.ArgumentParser(
        description='Collect text data for Indian Language Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --language kannada --source web --output data/kannada/
  %(prog)s --language hindi --source sample --max-samples 1000
  %(prog)s --language tamil --source file --input-file data/tamil.txt
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--language',
        type=str,
        required=True,
        help='Target language (e.g., kannada, hindi, tamil)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['web', 'file', 'sample'],
        help='Data source: web scraping, file input, or sample data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for collected data'
    )
    
    # Data source arguments
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input text file (for --source file)'
    )
    
    parser.add_argument(
        '--urls',
        type=str,
        nargs='+',
        help='URLs to scrape (for --source web)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum number of samples to collect (default: 10000)'
    )
    
    # Processing arguments
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean and preprocess collected text'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum text length in characters (default: 10)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help='Maximum text length in characters (default: 1000)'
    )
    
    # Output format arguments
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'json', 'csv'],
        default='txt',
        help='Output format (default: txt)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split into train/validation/test sets'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without saving data'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting data collection for {args.language} language")
    logger.info(f"Data source: {args.source}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Initialize data collector based on language
        if args.language == 'kannada':
            collector = KannadaDataCollector()
            preprocessor = KannadaPreprocessor() if args.clean else None
        else:
            logger.error(f"Language '{args.language}' not supported yet")
            return 1
        
        # Collect data based on source
        data = []
        
        if args.source == 'sample':
            logger.info("Collecting sample data...")
            data = collector.collect_sample_data()
            
        elif args.source == 'file':
            if not args.input_file:
                logger.error("--input-file required for --source file")
                return 1
            
            logger.info(f"Reading from file: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data = [{'text': line.strip(), 'source': 'file'} for line in lines if line.strip()]
            
        elif args.source == 'web':
            logger.info("Web scraping not implemented yet")
            logger.info("Using sample data instead...")
            data = collector.collect_sample_data()
        
        # Limit number of samples
        if args.max_samples and len(data) > args.max_samples:
            data = data[:args.max_samples]
        
        logger.info(f"Collected {len(data)} samples")
        
        # Filter by length
        original_count = len(data)
        data = [
            item for item in data 
            if args.min_length <= len(item['text']) <= args.max_length
        ]
        filtered_count = len(data)
        
        if filtered_count < original_count:
            logger.info(f"Filtered {original_count - filtered_count} samples by length")
        
        # Clean data if requested
        if args.clean and preprocessor:
            logger.info("Cleaning and preprocessing text...")
            for item in data:
                item['text'] = preprocessor.clean_text(item['text'])
                item['processed'] = True
        
        if not data:
            logger.error("No data collected!")
            return 1
        
        # Create output directory
        output_dir = Path(args.output)
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data if requested
        if args.split:
            train_size = int(len(data) * args.train_ratio)
            val_size = int(len(data) * args.val_ratio)
            
            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]
            
            splits = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        else:
            splits = {'all': data}
        
        # Save data
        for split_name, split_data in splits.items():
            if not split_data:
                continue
            
            filename = f"{args.language}_{split_name}"
            
            if args.format == 'txt':
                output_file = output_dir / f"{filename}.txt"
                if not args.dry_run:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for item in split_data:
                            f.write(item['text'] + '\n')
                logger.info(f"Saved {len(split_data)} samples to {output_file}")
                
            elif args.format == 'json':
                output_file = output_dir / f"{filename}.json"
                if not args.dry_run:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(split_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(split_data)} samples to {output_file}")
                
            elif args.format == 'csv':
                import csv
                output_file = output_dir / f"{filename}.csv"
                if not args.dry_run:
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        fieldnames = ['text', 'source', 'processed'] if args.clean else ['text', 'source']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for item in split_data:
                            writer.writerow({k: v for k, v in item.items() if k in fieldnames})
                logger.info(f"Saved {len(split_data)} samples to {output_file}")
        
        # Save metadata
        if not args.dry_run:
            metadata = {
                'language': args.language,
                'source': args.source,
                'total_samples': len(data),
                'format': args.format,
                'cleaned': args.clean,
                'min_length': args.min_length,
                'max_length': args.max_length,
                'splits': {name: len(split_data) for name, split_data in splits.items()}
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info("Data collection completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())