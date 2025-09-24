#!/usr/bin/env python3
"""
Kannada NLP Demonstration Script

A simple script to showcase Kannada language processing capabilities
using our specialized framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from preprocessing.kannada_preprocessor import KannadaPreprocessor
from data_collection.kannada_collector import KannadaDataCollector
from models import IndianLanguageModel


def main():
    """Main demonstration function."""
    
    print("🔤 ಕನ್ನಡ NLP ಪ್ರದರ್ಶನ (Kannada NLP Demonstration)")
    print("=" * 60)
    
    # Initialize Kannada preprocessor
    print("\n1. 🧹 ಪ್ರೀ-ಪ್ರೊಸೆಸರ್ ಆರಂಭ (Initializing Preprocessor)...")
    preprocessor = KannadaPreprocessor()
    
    # Sample Kannada texts for demonstration
    sample_texts = [
        "ಕನ್ನಡ ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಒಂದು ಸುಂದರ ಭಾಷೆಯಾಗಿದೆ.",
        "ಬೆಂಗಳೂರು ಭಾರತದ ಸಿಲಿಕನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ.",
        "ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆ ಭವಿಷ್ಯದ ತಂತ್ರಜ್ಞಾನಗಳು.",
        "ಕರ್ನಾಟಕದ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಬಹಳ ಶ್ರೀಮಂತವಾಗಿದೆ."
    ]
    
    print(f"📝 ಮಾದರಿ ಪಾಠ್ಯಗಳು (Sample Texts): {len(sample_texts)} texts")
    
    # Demonstrate text processing
    print("\n2. 🔤 ಪಾಠ್ಯ ಪ್ರಕ್ರಿಯೆ (Text Processing)...")
    processed_texts = []
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n   📄 Text {i}:")
        print(f"   Original: {text}")
        
        # Clean text
        cleaned = preprocessor.clean_kannada_text(text)
        processed_texts.append(cleaned)
        print(f"   Cleaned:  {cleaned}")
        
        # Get statistics
        stats = preprocessor.get_kannada_text_statistics(text)
        print(f"   Stats: {stats['kannada_characters']} Kannada chars, "
              f"{stats['word_count']} words")
        
        # Tokenize
        tokens = preprocessor.tokenize_kannada_words(text)
        print(f"   Tokens: {tokens[:5]}...")  # Show first 5 tokens
    
    # Initialize model
    print(f"\n3. 🤖 ಮಾದರಿ ಆರಂಭ (Model Initialization)...")
    try:
        model = IndianLanguageModel(
            language='kn',
            model_type='bert',
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=6,  # Smaller for demo
            num_attention_heads=12,
            max_position_embeddings=1024
        )
        
        print("✅ Model initialized successfully!")
        
        # Get model parameters info
        param_count = model.get_parameter_count()
        print(f"   Total parameters: {param_count['total']:,}")
        
        # Test embeddings
        print(f"\n4. 🧠 ಎಂಬೆಡಿಂಗ್ ಪರೀಕ್ಷೆ (Embedding Test)...")
        embeddings = model.get_embeddings(sample_texts[:2], language='kn')
        print(f"   Generated embeddings shape: {embeddings.shape}")
        print(f"   Sample embedding (first 3 dims): {embeddings[0][:3].numpy()}")
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("   This is normal without proper dependencies installed.")
    
    # Initialize data collector
    print(f"\n5. 📊 ದತ್ತ ಸಂಗ್ರಾಹಕ (Data Collector)...")
    try:
        collector = KannadaDataCollector()
        
        # Create sample dataset
        sample_data = []
        for i, text in enumerate(processed_texts):
            sample_data.append({
                'title': f'ಮಾದರಿ ಲೇಖನ {i+1}',
                'content': text,
                'source': 'ಪ್ರದರ್ಶನ',
                'category': 'sample',
                'language': 'kn'
            })
        
        df = pd.DataFrame(sample_data)
        df['timestamp'] = pd.Timestamp.now()
        df['text_length'] = df['content'].str.len()
        df['word_count'] = df['content'].str.split().str.len()
        
        print(f"   Created sample dataset: {len(df)} items")
        print(f"   Average text length: {df['text_length'].mean():.0f} characters")
        
        # Show statistics
        stats = collector.get_collection_statistics(df)
        print(f"   Total words: {stats.get('total_words', 0)}")
        
    except Exception as e:
        print(f"❌ Data collector error: {e}")
    
    print(f"\n🎉 ಪ್ರದರ್ಶನ ಪೂರ್ಣ! (Demonstration Complete!)")
    print("=" * 60)
    print("✨ Kannada NLP framework is ready for advanced language processing!")
    print("🔍 Explore the notebooks for detailed workflows.")
    print("📚 Check the configs/ directory for Kannada-specific settings.")
    print("🚀 Start building amazing Kannada NLP applications!")


if __name__ == "__main__":
    main()