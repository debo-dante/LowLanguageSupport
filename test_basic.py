#!/usr/bin/env python3
"""
Basic functionality test for Kannada NLP framework
"""

import sys
import os
sys.path.append('src')

print("🔤 ಕನ್ನಡ NLP ಮೂಲಭೂತ ಪರೀಕ್ಷೆ (Kannada NLP Basic Test)")
print("=" * 60)

# Test 1: Basic imports
print("\n1. 📦 Testing imports...")
try:
    from preprocessing.kannada_preprocessor import KannadaPreprocessor
    print("   ✅ KannadaPreprocessor imported successfully")
except Exception as e:
    print(f"   ❌ KannadaPreprocessor import failed: {e}")

try:
    from data_collection.kannada_collector import KannadaDataCollector
    print("   ✅ KannadaDataCollector imported successfully")
except Exception as e:
    print(f"   ❌ KannadaDataCollector import failed: {e}")

# Test 2: Configuration loading
print("\n2. ⚙️ Testing configuration...")
try:
    import yaml
    with open('configs/kannada_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"   ✅ Kannada config loaded: {config['language']['primary']}")
    supported_langs = config['language'].get('supported_variants', config['language'].get('supported', []))
    print(f"   📊 Supported languages: {len(supported_langs)}")
except Exception as e:
    print(f"   ❌ Config loading failed: {e}")

# Test 3: Basic preprocessing
print("\n3. 🧹 Testing Kannada preprocessing...")
try:
    preprocessor = KannadaPreprocessor()
    
    # Test text
    test_text = "ಕನ್ನಡ ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಒಂದು ಸುಂದರ ಭಾಷೆಯಾಗಿದೆ."
    print(f"   📝 Original: {test_text}")
    
    # Clean text
    cleaned = preprocessor.clean_kannada_text(test_text)
    print(f"   ✨ Cleaned: {cleaned}")
    
    # Get statistics
    stats = preprocessor.get_kannada_text_statistics(test_text)
    print(f"   📊 Stats: {stats.get('kannada_characters', 0)} Kannada chars, "
          f"{stats.get('word_count', 0)} words")
    
    # Tokenize
    tokens = preprocessor.tokenize_kannada_words(test_text)
    print(f"   🔤 Tokens: {tokens[:3]}... ({len(tokens)} total)")
    
    print("   ✅ Preprocessing test successful!")
    
except Exception as e:
    print(f"   ❌ Preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Data collector initialization
print("\n4. 📊 Testing data collector...")
try:
    collector = KannadaDataCollector()
    print("   ✅ KannadaDataCollector initialized successfully")
    
    # Test with sample data
    import pandas as pd
    sample_data = [{
        'title': 'ಪರೀಕ್ಷಾ ಶೀರ್ಷಿಕೆ',
        'content': 'ಇದು ಒಂದು ಪರೀಕ್ಷಾ ಪಾಠ್ಯವಾಗಿದೆ.',
        'source': 'ಪರೀಕ್ಷೆ',
        'language': 'kn'
    }]
    
    df = pd.DataFrame(sample_data)
    # Add columns that the collector expects
    df['word_count'] = df['content'].str.split().str.len()
    df['text_length'] = df['content'].str.len()
    stats = collector.get_collection_statistics(df)
    print(f"   📈 Sample stats: {stats.get('total_items', 0)} items")
    
    print("   ✅ Data collector test successful!")
    
except Exception as e:
    print(f"   ❌ Data collector test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: File structure check
print("\n5. 📁 Checking file structure...")
expected_files = [
    'configs/kannada_config.yaml',
    'src/preprocessing/kannada_preprocessor.py',
    'src/data_collection/kannada_collector.py',
    'notebooks/02_kannada_workflow.ipynb',
    'scripts/kannada_demo.py',
    'README_KANNADA.md'
]

for file_path in expected_files:
    if os.path.exists(file_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} (missing)")

print(f"\n🎉 Basic test completed!")
print("=" * 60)
print("🚀 Ready to run advanced features!")