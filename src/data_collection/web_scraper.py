"""
Web Scraper for Indian Language Content

Collects text data from various Indian language websites, news sources,
blogs, and other online resources.
"""

import requests
import time
import logging
import re
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langdetect import detect
import pandas as pd


class WebScraper:
    """
    Web scraper for collecting Indian language text data.
    
    Supports both static scraping with requests/BeautifulSoup
    and dynamic scraping with Selenium for JavaScript-heavy sites.
    """
    
    def __init__(self, 
                 language: str,
                 delay: float = 1.0,
                 max_retries: int = 3,
                 use_selenium: bool = False):
        """
        Initialize the web scraper.
        
        Args:
            language: Target language code (e.g., 'hi', 'bn', 'ta')
            delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
            use_selenium: Whether to use Selenium for dynamic content
        """
        self.language = language
        self.delay = delay
        self.max_retries = max_retries
        self.use_selenium = use_selenium
        self.session = requests.Session()
        self.visited_urls: Set[str] = set()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Common headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Selenium setup if needed
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver with appropriate options."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
        except Exception as e:
            self.logger.warning(f"Failed to setup Selenium: {e}")
            self.use_selenium = False
    
    def scrape_url(self, url: str, 
                   content_selectors: List[str] = None) -> Dict[str, str]:
        """
        Scrape content from a single URL.
        
        Args:
            url: URL to scrape
            content_selectors: CSS selectors to find main content
            
        Returns:
            Dictionary with scraped data
        """
        if url in self.visited_urls:
            return {}
        
        if content_selectors is None:
            content_selectors = [
                'article', '.content', '.post-content', 
                '.article-body', 'main', '.story-body'
            ]
        
        try:
            if self.use_selenium:
                return self._scrape_with_selenium(url, content_selectors)
            else:
                return self._scrape_with_requests(url, content_selectors)
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return {}
    
    def _scrape_with_requests(self, url: str, 
                            content_selectors: List[str]) -> Dict[str, str]:
        """Scrape using requests and BeautifulSoup."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract main content
                text_content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        text_content = ' '.join([elem.get_text(strip=True) 
                                               for elem in elements])
                        break
                
                if not text_content:
                    # Fallback to body text
                    body = soup.find('body')
                    if body:
                        text_content = body.get_text(strip=True)
                
                # Clean and validate text
                clean_text = self._clean_text(text_content)
                
                if self._is_valid_content(clean_text):
                    self.visited_urls.add(url)
                    return {
                        'url': url,
                        'title': soup.title.string if soup.title else '',
                        'content': clean_text,
                        'language': self._detect_language(clean_text),
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                
                break
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        
        return {}
    
    def _scrape_with_selenium(self, url: str, 
                            content_selectors: List[str]) -> Dict[str, str]:
        """Scrape using Selenium for dynamic content."""
        try:
            self.driver.get(url)
            
            # Wait for content to load
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Extract content using selectors
            text_content = ""
            for selector in content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        text_content = ' '.join([elem.text.strip() for elem in elements])
                        break
                except:
                    continue
            
            if not text_content:
                # Fallback to body text
                body = self.driver.find_element(By.TAG_NAME, "body")
                text_content = body.text.strip()
            
            # Clean and validate text
            clean_text = self._clean_text(text_content)
            
            if self._is_valid_content(clean_text):
                self.visited_urls.add(url)
                return {
                    'url': url,
                    'title': self.driver.title,
                    'content': clean_text,
                    'language': self._detect_language(clean_text),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
        
        except Exception as e:
            self.logger.error(f"Selenium scraping failed for {url}: {e}")
        
        return {}
    
    def scrape_news_sites(self, news_sites: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Scrape multiple news sites for the target language.
        
        Args:
            news_sites: List of dictionaries with 'url' and 'name' keys
            
        Returns:
            List of scraped articles
        """
        all_articles = []
        
        for site in news_sites:
            self.logger.info(f"Scraping {site['name']}...")
            
            try:
                # Get main page
                response = self.session.get(site['url'])
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = self._extract_article_links(soup, site['url'])
                
                # Scrape individual articles
                for link in article_links[:10]:  # Limit to 10 articles per site
                    article_data = self.scrape_url(link)
                    if article_data:
                        article_data['source'] = site['name']
                        all_articles.append(article_data)
                    
                    time.sleep(self.delay)
            
            except Exception as e:
                self.logger.error(f"Failed to scrape {site['name']}: {e}")
        
        return all_articles
    
    def _extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract article links from a news site homepage."""
        links = []
        
        # Common patterns for article links
        link_selectors = [
            'a[href*="/news/"]',
            'a[href*="/article/"]', 
            'a[href*="/story/"]',
            'a[href*="/post/"]',
            '.article-link a',
            '.news-item a'
        ]
        
        for selector in link_selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if full_url not in self.visited_urls:
                        links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def _clean_text(self, text: str) -> str:
        """Clean scraped text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common navigation/footer text
        unwanted_patterns = [
            r'cookie policy',
            r'privacy policy', 
            r'terms of service',
            r'subscribe to newsletter',
            r'follow us on',
            r'share this article'
        ]
        
        for pattern in unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _is_valid_content(self, text: str) -> bool:
        """Check if scraped content is valid and substantial."""
        if not text or len(text) < 100:
            return False
        
        # Check for minimum word count
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check if text is mostly in target language
        try:
            detected_lang = detect(text)
            # Map common language codes
            lang_mapping = {
                'hi': ['hi', 'ur'],  # Hindi/Urdu
                'bn': ['bn'],        # Bengali
                'ta': ['ta'],        # Tamil
                'te': ['te'],        # Telugu
                'mr': ['mr'],        # Marathi
                'gu': ['gu'],        # Gujarati
                'pa': ['pa'],        # Punjabi
                'or': ['or'],        # Odia
                'ml': ['ml'],        # Malayalam
                'kn': ['kn']         # Kannada
            }
            
            expected_langs = lang_mapping.get(self.language, [self.language])
            if detected_lang not in expected_langs:
                return False
                
        except:
            # If language detection fails, accept the text
            pass
        
        return True
    
    def _detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            return detect(text)
        except:
            return 'unknown'
    
    def save_data(self, data: List[Dict[str, str]], filepath: str):
        """Save scraped data to file."""
        df = pd.DataFrame(data)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', force_ascii=False, indent=2)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv, .json, or .parquet")
        
        self.logger.info(f"Saved {len(data)} articles to {filepath}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'driver'):
            self.driver.quit()


# Example usage and Indian language news sites
INDIAN_NEWS_SITES = {
    'hindi': [
        {'name': 'BBC Hindi', 'url': 'https://www.bbc.com/hindi'},
        {'name': 'Amar Ujala', 'url': 'https://www.amarujala.com/'},
        {'name': 'Navbharat Times', 'url': 'https://navbharattimes.indiatimes.com/'}
    ],
    'bengali': [
        {'name': 'Anandabazar', 'url': 'https://www.anandabazar.com/'},
        {'name': 'Prothom Alo', 'url': 'https://www.prothomalo.com/'}
    ],
    'tamil': [
        {'name': 'Dinamalar', 'url': 'https://www.dinamalar.com/'},
        {'name': 'Hindu Tamil', 'url': 'https://www.hindutamil.in/'}
    ]
}
