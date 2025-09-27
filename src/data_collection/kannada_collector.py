"""
Kannada Data Collector

Specialized data collection module for gathering Kannada language text
from various sources including news websites, literature, and government portals.
"""

import requests
import time
import logging
import re
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import yaml

from .web_scraper import WebScraper


class KannadaDataCollector(WebScraper):
    """
    Specialized data collector for Kannada language content.
    
    Extends the base WebScraper with Kannada-specific sources,
    content validation, and processing capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, delay: float = 2.0):
        """
        Initialize Kannada data collector.
        
        Args:
            config_path: Path to Kannada configuration file
            delay: Delay between requests in seconds
        """
        super().__init__(language='kn', delay=delay, max_retries=3)
        
        # Load Kannada configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "kannada_config.yaml"
        
        self.kannada_config = self._load_config(config_path)
        
        # Kannada Unicode range for validation
        self.kannada_unicode_range = (0x0C80, 0x0CFF)
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Load data sources
        self.news_sites = self.kannada_config.get('data_sources', {}).get('news_sites', [])
        self.literature_sources = self.kannada_config.get('data_sources', {}).get('literature_sources', [])
        self.government_sources = self.kannada_config.get('data_sources', {}).get('government_sources', [])
    
    def _load_config(self, config_path: str) -> Dict:
        """Load Kannada configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def collect_kannada_news(self, max_articles_per_site: int = 20) -> List[Dict[str, str]]:
        """
        Collect Kannada news articles from configured news sites.
        
        Args:
            max_articles_per_site: Maximum articles to collect per site
            
        Returns:
            List of collected articles with metadata
        """
        self.logger.info("Starting Kannada news collection...")
        all_articles = []
        
        for site in self.news_sites:
            self.logger.info(f"Collecting from {site['name']}...")
            
            try:
                # Get main page
                response = self.session.get(site['url'])
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract article links using site-specific selectors
                article_links = self._extract_kannada_article_links(
                    soup, site['url'], site.get('selectors', [])
                )
                
                # Limit articles per site
                article_links = article_links[:max_articles_per_site]
                
                # Scrape individual articles
                for link in article_links:
                    article_data = self.scrape_kannada_article(link, site['selectors'])
                    if article_data:
                        article_data['source'] = site['name']
                        article_data['category'] = 'news'
                        all_articles.append(article_data)
                    
                    time.sleep(self.delay)
                
                self.logger.info(f"Collected {len([a for a in all_articles if a['source'] == site['name']])} articles from {site['name']}")
                
            except Exception as e:
                self.logger.error(f"Failed to collect from {site['name']}: {e}")
        
        self.logger.info(f"Total collected: {len(all_articles)} Kannada news articles")
        return all_articles
    
    def collect_kannada_literature(self) -> List[Dict[str, str]]:
        """
        Collect Kannada literature content.
        
        Returns:
            List of collected literature texts with metadata
        """
        self.logger.info("Starting Kannada literature collection...")
        literature_texts = []
        
        for source in self.literature_sources:
            self.logger.info(f"Collecting literature from {source['name']}...")
            
            try:
                # This is a placeholder for literature collection
                # In practice, you'd implement specific scrapers for each source
                sample_literature = self._collect_sample_literature(source)
                literature_texts.extend(sample_literature)
                
            except Exception as e:
                self.logger.error(f"Failed to collect literature from {source['name']}: {e}")
        
        return literature_texts
    
    def collect_government_content(self) -> List[Dict[str, str]]:
        """
        Collect Kannada content from government sources.
        
        Returns:
            List of collected government content with metadata
        """
        self.logger.info("Starting government content collection...")
        government_texts = []
        
        for source in self.government_sources:
            self.logger.info(f"Collecting from {source['name']}...")
            
            try:
                # Collect government portal content
                content = self._collect_government_content(source)
                government_texts.extend(content)
                
            except Exception as e:
                self.logger.error(f"Failed to collect from {source['name']}: {e}")
        
        return government_texts
    
    def scrape_kannada_article(self, url: str, selectors: List[str]) -> Optional[Dict[str, str]]:
        """
        Scrape a single Kannada article with validation.
        
        Args:
            url: Article URL
            selectors: CSS selectors for content extraction
            
        Returns:
            Article data if valid Kannada content found
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_selectors = ['h1', '.title', '.headline', '.article-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content using provided selectors
            content = ""
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            # Fallback content extraction
            if not content:
                content_tags = soup.find_all(['p', 'div'], string=re.compile(r'[\u0C80-\u0CFF]+'))
                content = ' '.join([tag.get_text(strip=True) for tag in content_tags])
            
            # Validate Kannada content
            if self._validate_kannada_content(content, title):
                return {
                    'url': url,
                    'title': title,
                    'content': content,
                    'language': 'kn',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'word_count': len(content.split()),
                    'kannada_ratio': self._calculate_kannada_ratio(content)
                }
        
        except Exception as e:
            self.logger.warning(f"Failed to scrape {url}: {e}")
        
        return None
    
    def _extract_kannada_article_links(self, soup: BeautifulSoup, base_url: str, selectors: List[str]) -> List[str]:
        """
        Extract article links from Kannada news site homepage.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL of the site
            selectors: CSS selectors for finding article links
            
        Returns:
            List of article URLs
        """
        links = []
        
        # Use provided selectors first
        for selector in selectors:
            link_elements = soup.select(f"{selector} a")
            for elem in link_elements:
                href = elem.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_likely_article_url(full_url) and full_url not in self.visited_urls:
                        links.append(full_url)
        
        # Fallback: look for links with Kannada text
        if not links:
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if href and self._contains_kannada_text(link_text):
                    full_url = urljoin(base_url, href)
                    if self._is_likely_article_url(full_url) and full_url not in self.visited_urls:
                        links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def _is_likely_article_url(self, url: str) -> bool:
        """Check if URL is likely to be an article."""
        article_patterns = [
            r'/news/',
            r'/article/',
            r'/story/',
            r'/\d{4}/',  # Year in URL
            r'/\d{2}/',  # Month/day in URL
            r'-\d+',     # Article ID
        ]
        
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in article_patterns)
    
    def _contains_kannada_text(self, text: str) -> bool:
        """Check if text contains Kannada characters."""
        if not text:
            return False
        
        kannada_chars = sum(1 for c in text if self.kannada_unicode_range[0] <= ord(c) <= self.kannada_unicode_range[1])
        return kannada_chars > 0
    
    def _validate_kannada_content(self, content: str, title: str = "") -> bool:
        """
        Validate if content is substantial Kannada text.
        
        Args:
            content: Main content text
            title: Article title
            
        Returns:
            True if content meets Kannada validation criteria
        """
        if not content or len(content) < 100:
            return False
        
        # Check word count
        words = content.split()
        if len(words) < 20:
            return False
        
        # Check Kannada character ratio
        kannada_ratio = self._calculate_kannada_ratio(content)
        if kannada_ratio < 0.6:  # At least 60% Kannada characters
            return False
        
        # Check title if provided
        if title and not self._contains_kannada_text(title):
            return False
        
        return True
    
    def _calculate_kannada_ratio(self, text: str) -> float:
        """Calculate ratio of Kannada characters in text."""
        if not text:
            return 0.0
        
        # Count only non-whitespace characters
        non_space_text = re.sub(r'\s+', '', text)
        if not non_space_text:
            return 0.0
        
        kannada_chars = sum(1 for c in non_space_text if self.kannada_unicode_range[0] <= ord(c) <= self.kannada_unicode_range[1])
        return kannada_chars / len(non_space_text)
    
    def _collect_sample_literature(self, source: Dict) -> List[Dict[str, str]]:
        """
        Collect sample Kannada literature (placeholder implementation).
        
        Args:
            source: Literature source configuration
            
        Returns:
            List of literature samples
        """
        # This is a placeholder - in practice you'd implement specific scrapers
        sample_texts = [
            {
                'title': 'ಮಾಲೆ ನದಿಯ ಮೇಲೆ',
                'content': 'ಮಾಲೆ ನದಿಯ ಮೇಲೆ ಬಂದ ಮೋಡವು ಕವಿತೆಯಾಗಿ ಬರೆದುಕೊಂಡಿತು. ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯವು ಕನ್ನಡ ಸಾಹಿತ್ಯದಲ್ಲಿ ಯಾವಾಗಲೂ ಪ್ರಮುಖ ಸ್ಥಾನವನ್ನು ಹೊಂದಿದೆ.',
                'author': 'ಅಜ್ಞಾತ',
                'source': source['name'],
                'category': 'literature',
                'language': 'kn',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        ]
        
        return sample_texts
    
    def _collect_government_content(self, source: Dict) -> List[Dict[str, str]]:
        """
        Collect government content (placeholder implementation).
        
        Args:
            source: Government source configuration
            
        Returns:
            List of government content
        """
        # Placeholder implementation
        sample_content = [
            {
                'title': 'ಕರ್ನಾಟಕ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು',
                'content': 'ಕರ್ನಾಟಕ ಸರ್ಕಾರವು ಶಿಕ್ಷಣ, ಆರೋಗ್ಯ ಮತ್ತು ಕೃಷಿ ಕ್ಷೇತ್ರಗಳಲ್ಲಿ ಹಲವು ಯೋಜನೆಗಳನ್ನು ಜಾರಿಗೊಳಿಸಿದೆ. ಇವುಗಳ ಮೂಲಕ ಜನರ ಜೀವನ ಮಟ್ಟವನ್ನು ಸುಧಾರಿಸಲು ಪ್ರಯತ್ನಿಸುತ್ತಿದೆ.',
                'source': source['name'],
                'category': 'government',
                'language': 'kn',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        ]
        
        return sample_content
    
    def collect_comprehensive_dataset(self, 
                                    news_articles: int = 50,
                                    include_literature: bool = True,
                                    include_government: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive Kannada dataset from all sources.
        
        Args:
            news_articles: Number of news articles to collect
            include_literature: Whether to include literature content
            include_government: Whether to include government content
            
        Returns:
            DataFrame with collected Kannada content
        """
        self.logger.info("Starting comprehensive Kannada dataset collection...")
        
        all_content = []
        
        # Collect news articles
        if news_articles > 0:
            news_data = self.collect_kannada_news(max_articles_per_site=news_articles // len(self.news_sites))
            all_content.extend(news_data)
        
        # Collect literature
        if include_literature:
            literature_data = self.collect_kannada_literature()
            all_content.extend(literature_data)
        
        # Collect government content
        if include_government:
            government_data = self.collect_government_content()
            all_content.extend(government_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_content)
        
        if not df.empty:
            # Add additional metadata
            df['collection_date'] = pd.Timestamp.now().date()
            df['text_length'] = df['content'].str.len()
            df['word_count'] = df['content'].str.split().str.len()
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp', ascending=False)
        
        self.logger.info(f"Dataset collection complete: {len(df)} items collected")
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str, format: str = 'csv'):
        """
        Save collected dataset to file.
        
        Args:
            df: DataFrame with collected data
            output_path: Path to save the dataset
            format: Output format ('csv', 'json', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format == 'json':
            df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Dataset saved to {output_path}")
    
    def get_collection_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the collected dataset.
        
        Args:
            df: Collected dataset DataFrame
            
        Returns:
            Dictionary with collection statistics
        """
        if df.empty:
            return {}
        
        stats = {
            'total_items': len(df),
            'total_characters': df['content'].str.len().sum(),
            'total_words': df['word_count'].sum(),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
        }
        
        # Category breakdown
        if 'category' in df.columns:
            stats['by_category'] = df['category'].value_counts().to_dict()
        
        # Source breakdown
        if 'source' in df.columns:
            stats['by_source'] = df['source'].value_counts().to_dict()
        
        # Kannada ratio statistics
        if 'kannada_ratio' in df.columns:
            stats['avg_kannada_ratio'] = df['kannada_ratio'].mean()
            stats['min_kannada_ratio'] = df['kannada_ratio'].min()
            stats['max_kannada_ratio'] = df['kannada_ratio'].max()
        
        return stats
    
    def collect_sample_data(self, num_samples: int = 20) -> List[Dict[str, str]]:
        """
        Collect sample Kannada data for testing and development.
        
        Args:
            num_samples: Number of sample texts to generate
            
        Returns:
            List of sample text data with metadata
        """
        self.logger.info(f"Generating {num_samples} sample Kannada texts")
        
        # Sample Kannada texts for development and testing
        sample_texts = [
            {
                'text': 'ಕನ್ನಡ ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಒಂದು ಸುಂದರ ಭಾಷೆಯಾಗಿದೆ.',
                'source': 'sample',
                'category': 'language',
                'title': 'ಕನ್ನಡ ಭಾಷೆ'
            },
            {
                'text': 'ಬೆಂಗಳೂರು ಭಾರತದ ಸಿಲಿಕನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯಲಾಗುತ್ತದೆ.',
                'source': 'sample',
                'category': 'geography',
                'title': 'ಬೆಂಗಳೂರು ನಗರ'
            },
            {
                'text': 'ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆ ಭವಿಷ್ಯದ ತಂತ್ರಜ್ಞಾನಗಳು.',
                'source': 'sample',
                'category': 'technology',
                'title': 'ಆಧುನಿಕ ತಂತ್ರಜ್ಞಾನ'
            },
            {
                'text': 'ಕರ್ನಾಟಕದ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಬಹಳ ಶ್ರೀಮಂತವಾಗಿದೆ.',
                'source': 'sample',
                'category': 'culture',
                'title': 'ಕರ್ನಾಟಕ ಸಂಸ್ಕೃತಿ'
            },
            {
                'text': 'ಹಂಪಿಯು ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ರಾಜಧಾನಿಯಾಗಿತ್ತು. ಇದು ಯುನೆಸ್ಕೋ ವಿಶ್ವ ಪರಂಪರೆ ತಾಣವಾಗಿದೆ.',
                'source': 'sample',
                'category': 'history',
                'title': 'ಹಂಪಿ ಇತಿಹಾಸ'
            },
            {
                'text': 'ಕನ್ನಡ ಸಾಹಿತ್ಯದಲ್ಲಿ ಪಂಪ, ರನ್ನ, ಪೊನ್ನ ಮುಂತಾದ ಮಹಾಕವಿಗಳು ಇದ್ದಾರೆ.',
                'source': 'sample',
                'category': 'literature',
                'title': 'ಕನ್ನಡ ಸಾಹಿತ್ಯ'
            },
            {
                'text': 'ಮೈಸೂರು ಅರಮನೆಯು ಕರ್ನಾಟಕದ ಪ್ರಸಿದ್ಧ ವಾಸ್ತುಶಿಲ್ಪದ ಉದಾಹರಣೆಯಾಗಿದೆ.',
                'source': 'sample',
                'category': 'architecture',
                'title': 'ಮೈಸೂರು ಅರಮನೆ'
            },
            {
                'text': 'ಕರ್ನಾಟಕ ಸಂಗೀತವು ಭಾರತೀಯ ಶಾಸ್ತ್ರೀಯ ಸಂಗೀತದ ಪ್ರಮುಖ ಅಂಗವಾಗಿದೆ.',
                'source': 'sample',
                'category': 'music',
                'title': 'ಕರ್ನಾಟಕ ಸಂಗೀತ'
            },
            {
                'text': 'ಉಡುಪಿ ಸಾಂಬಾರ್ ಮತ್ತು ರಾಗಿ ಮುದ್ದೆ ಕರ್ನಾಟಕದ ಪ್ರಸಿದ್ಧ ಆಹಾರಗಳು.',
                'source': 'sample',
                'category': 'food',
                'title': 'ಕನ್ನಡ ಆಹಾರ'
            },
            {
                'text': 'ಕೋಡಗು ಕಾಫಿ ತೋಟಗಳಿಗೆ ಪ್ರಸಿದ್ಧವಾಗಿದೆ. ಇಲ್ಲಿನ ಹವಾಮಾನವು ಕಾಫಿ ಬೆಳೆಗೆ ಅನುಕೂಲವಾಗಿದೆ.',
                'source': 'sample',
                'category': 'agriculture',
                'title': 'ಕೋಡಗು ಕಾಫಿ'
            },
            {
                'text': 'ಬೆಳಗಾವಿ ಮತ್ತು ಹುಬ್ಳಿ ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಕೈಗಾರಿಕಾ ಕೇಂದ್ರಗಳಾಗಿವೆ.',
                'source': 'sample',
                'category': 'industry',
                'title': 'ಕೈಗಾರಿಕಾ ಅಭಿವೃದ್ಧಿ'
            },
            {
                'text': 'ಕನ್ನಡ ಯೋಗ ಮತ್ತು ಆಯುರ್ವೇದ ಪರಂಪರೆಯು ಅತ್ಯಂತ ಪ್ರಾಚೀನವಾಗಿದೆ.',
                'source': 'sample',
                'category': 'wellness',
                'title': 'ಆಯುರ್ವೇದ ಪರಂಪರೆ'
            },
            {
                'text': 'ನಾಲಂದಾ ಮತ್ತು ತಕ್ಷಶಿಲಾದಂತಹ ಪ್ರಾಚೀನ ವಿಶ್ವವಿದ್ಯಾಲಯಗಳು ಭಾರತದ ಶಿಕ್ಷಣ ಪರಂಪರೆಯನ್ನು ತೋರಿಸುತ್ತವೆ.',
                'source': 'sample',
                'category': 'education',
                'title': 'ಪ್ರಾಚೀನ ಶಿಕ್ಷಣ'
            },
            {
                'text': 'ಶ್ರೀ ರಂಗಪಟ್ಟಣವು ಟಿಪ್ಪು ಸುಲ್ತಾನನ ರಾಜಧಾನಿಯಾಗಿತ್ತು.',
                'source': 'sample',
                'category': 'history',
                'title': 'ಟಿಪ್ಪು ಸುಲ್ತಾನ'
            },
            {
                'text': 'ಬಸವೇಶ್ವರರು ಸಮಾಜ ಸುಧಾರಕರಾಗಿ ಲಿಂಗಾಯತ ಧರ್ಮವನ್ನು ಸ್ಥಾಪಿಸಿದರು.',
                'source': 'sample',
                'category': 'philosophy',
                'title': 'ಬಸವೇಶ್ವರ ತತ್ವಜ್ಞಾನ'
            },
            {
                'text': 'ಗೋಕರ್ಣ ಮತ್ತು ಮುರುಡೇಶ್ವರ ಕರಾವಳಿ ಪ್ರದೇಶದ ಪ್ರಸಿದ್ಧ ದೇವಾಲಯಗಳಾಗಿವೆ.',
                'source': 'sample',
                'category': 'religion',
                'title': 'ಕರಾವಳಿ ದೇವಾಲಯಗಳು'
            },
            {
                'text': 'ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಪಂಚತಂತ್ರ ಮತ್ತು ಮಹಾಭಾರತದಂತಹ ಮಹಾಕಾವ್ಯಗಳು ಬರೆಯಲಾಗಿವೆ.',
                'source': 'sample',
                'category': 'literature',
                'title': 'ಮಹಾಕಾವ್ಯ ಸಾಹಿತ್ಯ'
            },
            {
                'text': 'ಚಂದ್ರಗುಪ್ತ ಮೌರ್ಯನು ಶ್ರವಣಬೆಳಗೊಳದಲ್ಲಿ ಜೈನ ಸಂತನಾಗಿ ಜೀವನ ಕಳೆದನು.',
                'source': 'sample',
                'category': 'history',
                'title': 'ಚಂದ್ರಗುಪ್ತ ಮೌರ್ಯ'
            },
            {
                'text': 'ಬಾದಾಮಿ ಗುಹೆಗಳು ಚಾಲುಕ್ಯ ವಾಸ್ತುಶಿಲ್ಪದ ಅದ್ಭುತ ಉದಾಹರಣೆಗಳಾಗಿವೆ.',
                'source': 'sample',
                'category': 'architecture',
                'title': 'ಬಾದಾಮಿ ಗುಹೆಗಳು'
            },
            {
                'text': 'ಕರ್ನಾಟಕದ ಯಕ್ಷಗಾನ ಮತ್ತು ಭರತನಾಟ್ಯ ಶಾಸ್ತ್ರೀಯ ನೃತ್ಯ ರೂಪಗಳಾಗಿವೆ.',
                'source': 'sample',
                'category': 'dance',
                'title': 'ಶಾಸ್ತ್ರೀಯ ನೃತ್ಯ'
            }
        ]
        
        # Return requested number of samples (cycle through if needed)
        selected_samples = []
        for i in range(num_samples):
            sample = sample_texts[i % len(sample_texts)].copy()
            sample['timestamp'] = pd.Timestamp.now().isoformat()
            sample['word_count'] = len(sample['text'].split())
            sample['text_length'] = len(sample['text'])
            selected_samples.append(sample)
        
        self.logger.info(f"Generated {len(selected_samples)} sample texts")
        return selected_samples


def demonstrate_kannada_collection():
    """Demonstrate Kannada data collection capabilities."""
    print("📰 Kannada Data Collection Demo")
    print("=" * 50)
    
    # Initialize collector
    collector = KannadaDataCollector()
    
    # Create sample dataset (using placeholder data)
    print("Collecting sample Kannada dataset...")
    
    # In practice, this would collect real data from websites
    # For demo, we'll create sample data
    sample_data = []
    
    # Add sample news articles
    news_samples = [
        {
            'title': 'ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹೊಸ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿ',
            'content': 'ಬೆಂಗಳೂರಿನಲ್ಲಿ ಹೊಸ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿಯು ಪ್ರಾರಂಭವಾಗಿದೆ. ಈ ಕಂಪನಿಯು ಕೃತ್ರಿಮ ಬುದ್ಧಿಮತ್ತೆ ಮತ್ತು ಯಂತ್ರ ಕಲಿಕೆಯ ಕ್ಷೇತ್ರದಲ್ಲಿ ಕೆಲಸ ಮಾಡುತ್ತದೆ.',
            'source': 'ಪ್ರಜಾವಾಣಿ',
            'category': 'news',
            'language': 'kn'
        },
        {
            'title': 'ಕನ್ನಡ ಭಾಷೆಯ ಮಹತ್ವ',
            'content': 'ಕನ್ನಡ ಭಾಷೆಯು ಕರ್ನಾಟಕದ ಸಾಂಸ್ಕೃತಿಕ ಗುರುತಾಗಿದೆ. ಇದು ಸಾವಿರಾರು ವರ್ಷಗಳ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದ ಶ್ರೀಮಂತ ಭಾಷೆಯಾಗಿದೆ.',
            'source': 'ಕನ್ನಡ ಪ್ರಭ',
            'category': 'culture',
            'language': 'kn'
        }
    ]
    
    sample_data.extend(news_samples)
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    df['timestamp'] = pd.Timestamp.now()
    df['text_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()
    
    # Show statistics
    stats = collector.get_collection_statistics(df)
    print(f"\n📊 Collection Statistics:")
    print(f"Total items: {stats['total_items']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Average text length: {stats['avg_text_length']:.0f} characters")
    
    # Show sample content
    print(f"\n📄 Sample Content:")
    for i, row in df.head(2).iterrows():
        print(f"\nTitle: {row['title']}")
        print(f"Content: {row['content'][:100]}...")
        print(f"Source: {row['source']}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    demonstrate_kannada_collection()
