# ಕನ್ನಡ NLP ಚೌಕಟ್ಟು (Kannada NLP Framework)

## ಪರಿಚಯ (Introduction)

ಈ ಯೋಜನೆಯು ಕನ್ನಡ ಭಾಷೆಗೆ ವಿಶೇಷವಾಗಿ ವಿನ್ಯಾಸಗೊಳಿಸಲಾದ ಸಮಗ್ರ NLP ಚೌಕಟ್ಟಾಗಿದೆ. This project is a comprehensive NLP framework specifically designed for the Kannada language, built on top of our Indian Language NLP foundation.

## ವೈಶಿಷ್ಟ್ಯಗಳು (Features)

### 🔤 ಕನ್ನಡ-ನಿರ್ದಿಷ್ಟ ಪ್ರೀ-ಪ್ರೊಸೆಸಿಂಗ್ (Kannada-Specific Preprocessing)
- **ಯೂನಿಕೋಡ್ ನಾರ್ಮಲೈಸೇಶನ್**: Unicode normalization for Kannada script
- **ವಿರಾಮ ನಿರ್ವಹಣೆ**: Proper virama/halant handling
- **ಸಂಯುಕ್ತಾಕ್ಷರ ಸಂರಕ್ಷಣೆ**: Conjunct character preservation
- **ರೂಪವಿಜ್ಞಾನ ವಿಶ್ಲೇಷಣೆ**: Morphological analysis
- **ನಿಲುಪದ ಸಿಕ್ಕಿಸಿ**: Stop word filtering

### 📊 ದತ್ತ ಸಂಗ್ರಹಣೆ (Data Collection)
- **ಸುದ್ದಿ ಸಂಗ್ರಹಣೆ**: News collection from major Kannada portals
- **ಸಾಹಿತ್ಯ ಸಂಗ್ರಹಣೆ**: Literature content gathering
- **ಸರ್ಕಾರಿ ಮೂಲಗಳು**: Government content collection
- **ಪಾಠ್ಯ ಮೌಲ್ಯಮಾಪನ**: Text validation and quality assessment

### 🤖 ಮಾದರಿ ಆಪ್ಟಿಮೈಸೇಶನ್ (Model Optimization)
- **ದ್ರಾವಿಡ ಭಾಷಾ ಬೆಂಬಲ**: Dravidian language family support
- **ಸ್ಕ್ರಿಪ್ಟ್ ಎಂಬೆಡಿಂಗ್ಗಳು**: Script-specific embeddings
- **ಅಡಾಪ್ಟರ್ ಲೇಯರ್ಗಳು**: Efficient adapter layers for fine-tuning
- **ಅಂತರ್-ಭಾಷೆ ವರ್ಗಾವಣೆ**: Cross-lingual transfer capabilities

## ಸ್ಥಾಪನೆ (Installation)

### ಪ್ರಾಥಮಿಕ ಸೆಟಪ್ (Basic Setup)

```bash
# Clone the repository
git clone <repository-url>
cd NLP

# Create virtual environment
python -m venv kannada-nlp
source kannada-nlp/bin/activate  # On macOS/Linux
# kannada-nlp\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### ಕನ್ನಡ-ನಿರ್ದಿಷ್ಟ ಅವಲಂಬನೆಗಳು (Kannada-Specific Dependencies)

```bash
# Install additional Kannada language support
pip install indic-nlp-library
pip install polyglot
pip install langdetect

# For better Kannada text rendering (optional)
# Install Kannada fonts on your system
```

## ತ್ವರಿತ ಪ್ರಾರಂಭ (Quick Start)

### 1. ಸರಳ ಪ್ರದರ್ಶನ ಚಲಾಯಿಸಿ (Run Simple Demo)

```bash
python scripts/kannada_demo.py
```

### 2. ಕನ್ನಡ ಪ್ರೀ-ಪ್ರೊಸೆಸಿಂಗ್ (Kannada Preprocessing)

```python
from src.preprocessing.kannada_preprocessor import KannadaPreprocessor

# Initialize preprocessor
preprocessor = KannadaPreprocessor()

# Clean Kannada text
kannada_text = "ಕನ್ನಡ ಭಾಷೆ ಬಹಳ ಸುಂದರವಾಗಿದೆ."
cleaned_text = preprocessor.clean_kannada_text(kannada_text)

# Get text statistics
stats = preprocessor.get_kannada_text_statistics(kannada_text)
print(stats)

# Tokenize words
tokens = preprocessor.tokenize_kannada_words(kannada_text)
print(tokens)
```

### 3. ದತ್ತ ಸಂಗ್ರಹಣೆ (Data Collection)

```python
from src.data_collection.kannada_collector import KannadaDataCollector

# Initialize collector
collector = KannadaDataCollector()

# Collect news articles (demo mode)
articles = collector.collect_kannada_news(max_articles_per_site=5)

# Create comprehensive dataset
dataset = collector.collect_comprehensive_dataset(
    news_articles=20,
    include_literature=True,
    include_government=True
)
```

### 4. ಮಾದರಿ ಬಳಕೆ (Model Usage)

```python
from src.models import IndianLanguageModel

# Initialize Kannada-optimized model
model = IndianLanguageModel(
    language='kn',
    model_type='bert',
    vocab_size=32000,
    max_position_embeddings=1024
)

# Get embeddings
kannada_sentences = [
    "ಇದು ಒಂದು ಉದಾಹರಣೆ.",
    "ಕನ್ನಡ ತಂತ್ರಜ್ಞಾನ ಅಭಿವೃದ್ಧಿ."
]
embeddings = model.get_embeddings(kannada_sentences, language='kn')
```

## ನೋಟ್‌ಬುಕ್‌ಗಳು (Notebooks)

### 📓 ಪ್ರಮುಖ ನೋಟ್‌ಬುಕ್‌ಗಳು
- **`notebooks/02_kannada_workflow.ipynb`**: ಸಂಪೂರ್ಣ ಕನ್ನಡ ವರ್ಕ್‌ಫ್ಲೋ
- **`notebooks/01_getting_started.ipynb`**: ಸಾಮಾನ್ಯ ಪರಿಚಯ

### ನೋಟ್‌ಬುಕ್ ಚಲಾಯಿಸುವುದು (Running Notebooks)

```bash
# Start Jupyter
jupyter notebook

# Open the Kannada workflow notebook
# Navigate to: notebooks/02_kannada_workflow.ipynb
```

## ಕಾನ್ಫಿಗರೇಶನ್ (Configuration)

### ಕನ್ನಡ ಸೆಟ್ಟಿಂಗ್ಗಳು (Kannada Settings)

ಫೈಲ್: `configs/kannada_config.yaml`

```yaml
language:
  primary: "kn"
  script: "kannada"
  unicode_range: [0x0C80, 0x0CFF]

data_sources:
  news_sites:
    - name: "Prajavani"
      url: "https://www.prajavani.net/"
    - name: "Kannada Prabha" 
      url: "https://kannadaprabha.com/"

model:
  vocab_size: 32000
  max_position_embeddings: 1024
```

## ದತ್ತ ಮೂಲಗಳು (Data Sources)

### ಬೆಂಬಲಿತ ಮೂಲಗಳು (Supported Sources)

#### 📰 ಸುದ್ದಿ ಪತ್ರಿಕೆಗಳು (News Sources)
- **ಪ್ರಜಾವಾಣಿ**: https://www.prajavani.net/
- **ಕನ್ನಡ ಪ್ರಭ**: https://kannadaprabha.com/
- **ಉದಯವಾಣಿ**: https://www.udayavani.com/
- **ವಿಜಯ ಕರ್ನಾಟಕ**: https://vijayakarnataka.com/

#### 📚 ಸಾಹಿತ್ಯ ಮೂಲಗಳು (Literature Sources)
- **ಕನ್ನಡ ಸಾಹಿತ್ಯ ಪರಿಷತ್**: https://kannadasahityaparishat.org/
- **ಕುವೆಂಪು ವಿಶ್ವವಿದ್ಯಾಲಯ ಡಿಜಿಟಲ್ ಲೈಬ್ರರಿ**

#### 🏛️ ಸರ್ಕಾರಿ ಮೂಲಗಳು (Government Sources)
- **ಕರ್ನಾಟಕ ಸರ್ಕಾರ**: https://www.karnataka.gov.in/kannada/
- **ಬೆಂಗಳೂರು ನಗರ ನಿಗಮ**: https://www.bengaluru.gov.in/

## ಮೌಲ್ಯಮಾಪನ (Evaluation)

### ಬೆಂಬಲಿತ ಕಾರ್ಯಗಳು (Supported Tasks)
- **ಪಾಠ್ಯ ವರ್ಗೀಕರಣ**: Text classification
- **ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ**: Sentiment analysis  
- **ಶಬ್ದಾರ್ಥದ ಹೋಲಿಕೆ**: Semantic similarity
- **ಅಂತರ್-ಭಾಷೆ ವರ್ಗಾವಣೆ**: Cross-lingual transfer
- **ರೂಪವಿಜ್ಞಾನ ವಿಶ್ಲೇಷಣೆ**: Morphological analysis

### ಮೌಲ್ಯಮಾಪನ ಚಲಾಯಿಸುವುದು (Running Evaluation)

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model=your_model, language='kn')

# Text classification
results = evaluator.evaluate_text_classification(texts, labels)

# Cross-lingual with other Dravidian languages
cross_results = evaluator.evaluate_cross_lingual_transfer(
    source_data={'texts': kn_texts, 'labels': kn_labels},
    target_data={'texts': ta_texts, 'labels': ta_labels},
    source_lang='kn', target_lang='ta'
)
```

## ಅಭಿವೃದ್ಧಿ (Development)

### ಕೊಡುಗೆ ನೀಡುವುದು (Contributing)

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/kannada-enhancement`
3. **Make changes and test**
4. **Submit pull request**

### ಪರೀಕ್ಷೆಗಳು ಚಲಾಯಿಸುವುದು (Running Tests)

```bash
# Run all tests
pytest

# Run Kannada-specific tests
pytest tests/test_kannada_preprocessing.py
pytest tests/test_kannada_collector.py
```

## ಸಾಮಾನ್ಯ ಸಮಸ್ಯೆಗಳು (Common Issues)

### ಫಾಂಟ್ ಪ್ರದರ್ಶನ ಸಮಸ್ಯೆಗಳು (Font Rendering Issues)

```bash
# Install Kannada fonts on Ubuntu/Debian
sudo apt-get install fonts-kn

# Install Kannada fonts on macOS
# Download and install Noto Sans Kannada from Google Fonts
```

### ಅವಲಂಬನೆ ದೋಷಗಳು (Dependency Errors)

```bash
# If indic-nlp-library installation fails
pip install --upgrade pip setuptools wheel
pip install indic-nlp-library --no-cache-dir

# If polyglot fails
pip install pyicu pycld2
pip install polyglot
```

## ಉದಾಹರಣೆಗಳು (Examples)

### ಸಂಪೂರ್ಣ ಪೈಪ್‌ಲೈನ್ ಉದಾಹರಣೆ (Complete Pipeline Example)

```python
from src.preprocessing.kannada_preprocessor import KannadaPreprocessor
from src.models import IndianLanguageModel
from src.evaluation import ModelEvaluator

# 1. Preprocess
preprocessor = KannadaPreprocessor()
clean_texts = [preprocessor.clean_kannada_text(text) for text in raw_texts]

# 2. Train/Load model  
model = IndianLanguageModel(language='kn', model_type='bert')

# 3. Evaluate
evaluator = ModelEvaluator(model=model, language='kn')
results = evaluator.evaluate_text_classification(clean_texts, labels)

# 4. Generate report
report = evaluator.generate_evaluation_report('kannada_results')
```

## ಸಂಪನ್ಮೂಲಗಳು (Resources)

### ಕಲಿಕಾ ಸಾಮಗ್ರಿಗಳು (Learning Materials)
- [ಕನ್ನಡ ವಿಕಿಪೀಡಿಯಾ](https://kn.wikipedia.org/)
- [AI4Bharat Kannada Resources](https://ai4bharat.org/)
- [IndicNLP Documentation](https://github.com/anoopkunchukuttan/indic_nlp_library)

### ಸಂಶೋಧನಾ ಪತ್ರಗಳು (Research Papers)
- Dravidian language processing papers
- Kannada NLP research articles
- Cross-lingual transfer learning studies

## ಪರವಾನಗಿ (License)

MIT License - ವಿವರಗಳಿಗೆ [LICENSE](LICENSE) ಫೈಲ್ ನೋಡಿ

## ಸಹಾಯ ಮತ್ತು ಬೆಂಬಲ (Help & Support)

- **ಸಮಸ್ಯೆಗಳು**: GitHub Issues ನಲ್ಲಿ ವರದಿ ಮಾಡಿ
- **ಚರ್ಚೆಗಳು**: GitHub Discussions ನಲ್ಲಿ ಭಾಗವಹಿಸಿ  
- **ಕೊಡುಗೆಗಳು**: Pull Requests ಸ್ವಾಗತ

---

## ಯಶಸ್ಸಿನ ಕಥೆ (Success Story)

🎉 **ಕನ್ನಡ ಭಾಷೆಯ ಡಿಜಿಟಲ್ ಭವಿಷ್ಯವನ್ನು ನಿರ್ಮಿಸಲು ನಮ್ಮೊಂದಿಗೆ ಸೇರಿಕೊಳ್ಳಿ!**

**Join us in building the digital future of the Kannada language!** 🌟🇮🇳

---

*ಈ ಯೋಜನೆಯು ಕನ್ನಡ ಭಾಷೆಯ ತಾಂತ್ರಿಕ ಅಭಿವೃದ್ಧಿಗೆ ಸಮರ್ಪಿತವಾಗಿದೆ.*
