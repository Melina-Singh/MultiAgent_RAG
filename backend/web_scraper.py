# web_scraper.py
import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List
from urllib.parse import urljoin, urlparse
import time
import logging

class WebScraper:
    """Handle web scraping functionality"""
    
    def __init__(self, timeout: int = 30, max_content_length: int = 100000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.logger = logging.getLogger(__name__)
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_url(self, url: str) -> str:
        """Scrape content from a URL"""
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                raise ValueError(f"URL does not contain HTML content: {content_type}")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text = self._extract_text_content(soup)
            
            # Clean and limit text
            text = self._clean_text(text)
            
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length] + "..."
                self.logger.warning(f"Content truncated to {self.max_content_length} characters")
            
            if not text.strip():
                raise ValueError("No meaningful text content found on the page")
            
            return text
        
        except requests.RequestException as e:
            self.logger.error(f"Request error for URL {url}: {str(e)}")
            raise ValueError(f"Failed to fetch URL: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error scraping URL {url}: {str(e)}")
            raise
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Remove ads and common non-content elements
        for element in soup.find_all(class_=re.compile("ad|advertisement|sidebar|menu", re.I)):
            element.decompose()
        
        for element in soup.find_all(id=re.compile("ad|advertisement|sidebar|menu", re.I)):
            element.decompose()
        
        # Try to find main content areas
        main_content = ""
        
        # Look for main content containers
        content_selectors = [
            'main', 'article', '[role="main"]', '.main-content', '.content',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    main_content += element.get_text(separator='\n', strip=True) + "\n\n"
        
        # If no main content found, extract from body
        if not main_content.strip():
            # Look for paragraphs and headers
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']):
                text = tag.get_text(strip=True)
                if len(text) > 20:  # Only include substantial text
                    main_content += text + "\n\n"
        
        # Fallback to all text
        if not main_content.strip():
            main_content = soup.get_text(separator='\n', strip=True)
        
        return main_content
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        # Remove very short lines (likely navigation or ads)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep lines that are substantial or are likely headers
            if len(line) > 15 or (len(line) > 3 and line.isupper()) or re.match(r'^[A-Z][a-z]+.*[.!?]$', line):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up extra spaces after URL/email removal
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'url': url,
                'status_code': response.status_code,
                'title': '',
                'description': '',
                'keywords': '',
                'author': '',
                'published_date': '',
                'language': '',
                'content_length': len(response.content)
            }
            
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                property_name = tag.get('property', '').lower()
                content = tag.get('content', '')
                
                if name in ['description', 'og:description'] or property_name == 'og:description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author':
                    metadata['author'] = content
                elif name in ['published_time', 'article:published_time'] or property_name == 'article:published_time':
                    metadata['published_date'] = content
                elif name == 'language' or tag.get('http-equiv', '').lower() == 'content-language':
                    metadata['language'] = content
            
            # Language from html tag
            html_tag = soup.find('html')
            if html_tag and not metadata['language']:
                metadata['language'] = html_tag.get('lang', '')
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
    
    def batch_scrape_urls(self, urls: List[str], delay: float = 1.0) -> Dict[str, Any]:
        """Scrape multiple URLs with delays to be respectful"""
        results = {}
        
        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Scraping URL {i+1}/{len(urls)}: {url}")
                content = self.scrape_url(url)
                results[url] = {
                    'status': 'success',
                    'content': content,
                    'content_length': len(content)
                }
            except Exception as e:
                results[url] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Add delay between requests
            if i < len(urls) - 1:
                time.sleep(delay)
        
        return results
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """Validate if URL is accessible and scrapeable"""
        validation_result = {
            'is_valid': False,
            'is_accessible': False,
            'content_type': '',
            'status_code': 0,
            'estimated_content_size': 0,
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check URL format
            if not self._is_valid_url(url):
                validation_result['warnings'].append("Invalid URL format")
                return validation_result
            
            validation_result['is_valid'] = True
            
            # Make HEAD request first
            try:
                head_response = requests.head(url, headers=self.headers, timeout=10)
                validation_result['status_code'] = head_response.status_code
                validation_result['content_type'] = head_response.headers.get('content-type', '')
                
                content_length = head_response.headers.get('content-length')
                if content_length:
                    validation_result['estimated_content_size'] = int(content_length)
            except:
                pass
            
            # Make GET request to verify accessibility
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            validation_result['is_accessible'] = True
            validation_result['status_code'] = response.status_code
            validation_result['content_type'] = response.headers.get('content-type', '')
            
            # Check content type
            if 'text/html' not in validation_result['content_type'].lower():
                validation_result['warnings'].append(f"Content type is not HTML: {validation_result['content_type']}")
            
            # Check content size
            actual_size = len(response.content)
            validation_result['estimated_content_size'] = actual_size
            
            if actual_size > 1000000:  # 1MB
                validation_result['recommendations'].append("Large page - scraping may take longer")
            
            # Quick check for content
            soup = BeautifulSoup(response.content[:10000], 'html.parser')  # Check first 10KB
            text_content = soup.get_text(strip=True)
            
            if len(text_content) < 100:
                validation_result['warnings'].append("Page appears to have very little text content")
            
        except requests.RequestException as e:
            validation_result['warnings'].append(f"Request failed: {str(e)}")
        except Exception as e:
            validation_result['warnings'].append(f"Validation error: {str(e)}")
        
        return validation_result