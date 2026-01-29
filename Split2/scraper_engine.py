# scraper_engine.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
import time
from collections import defaultdict
from config import SCRAPER_SETTINGS

class IntegratedScraper:
    """
    Scraper engine integrated for the File Service.
    Crawls pages and returns a list of dictionaries.
    """
    
    def __init__(self, max_pages=None, task_id=None, update_callback=None):
        self.visited_urls = set()
        self.pages_data = []
        self.stats = defaultdict(int)
        self.max_pages = max_pages or SCRAPER_SETTINGS["MAX_PAGES_DEFAULT"]
        self.task_id = task_id
        self.update_callback = update_callback  # Function to call with progress updates
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPER_SETTINGS["USER_AGENT"],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def normalize_url(self, url):
        try:
            parsed = urlparse(url)
            url_no_frag = urlunparse(parsed._replace(fragment=""))
            return url_no_frag.rstrip('/')
        except:
            return url
    
    def is_allowed(self, url):
        try:
            netloc = urlparse(url).netloc.replace('www.', '')
            return any(domain in netloc for domain in SCRAPER_SETTINGS["ALLOWED_DOMAINS"])
        except:
            return False
    
    def should_skip(self, url):
        return any(url.lower().endswith(ext) for ext in SCRAPER_SETTINGS["SKIP_EXTENSIONS"])
    
    def fetch_page(self, url):
        try:
            time.sleep(SCRAPER_SETTINGS["DELAY_BETWEEN_REQUESTS"])
            response = self.session.get(url, timeout=SCRAPER_SETTINGS["TIMEOUT"], stream=True)
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                response.close()
                return None
            
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"Scrape Error {url}: {str(e)[:50]}")
        return None
    
    def extract_text(self, soup):
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                           'aside', 'iframe', 'noscript', 'form']):
            element.decompose()
        
        main_content = (
            soup.find('main') or 
            soup.find(class_='content') or 
            soup.find(id='content') or
            soup.body
        )
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return '\n'.join(lines)
        return ""
    
    def extract_links(self, soup, base_url):
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if href.startswith('#') or not href:
                continue
            full_url = urljoin(base_url, href)
            normalized = self.normalize_url(full_url)
            if self.is_allowed(normalized) and not self.should_skip(normalized):
                links.add(normalized)
        return links

    def categorize_page(self, url):
        netloc = urlparse(url).netloc.replace('www.', '')
        path = url.lower()
        
        if 'msheirebproperties' in netloc:
            domain_name = "Msheireb Properties"
            domain = "msheireb_properties"
        elif 'msheirebmuseums' in netloc:
            domain_name = "Msheireb Museums"
            domain = "msheireb_museums"
        else:
            domain_name = "Msheireb Downtown Doha"
            domain = "msheireb_downtown"
        
        category = "general"
        if '/event' in path: category = "events"
        elif '/news' in path: category = "news"
        elif '/about' in path: category = "about"
        elif any(x in path for x in ['sales', 'leasing', 'residential', 'commercial']): category = "real_estate"
        elif any(x in path for x in ['museum', 'house']): category = "heritage"
        
        return {"domain": domain, "domain_name": domain_name, "category": category}

    def scrape_page(self, url):
        normalized = self.normalize_url(url)
        if normalized in self.visited_urls:
            return None
        
        self.visited_urls.add(normalized)
        content_bytes = self.fetch_page(url)
        if not content_bytes:
            return None
        
        try:
            soup = BeautifulSoup(content_bytes, 'html.parser')
            title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
            content = self.extract_text(soup)
            links = self.extract_links(soup, url)
            cat = self.categorize_page(url)
            
            page_data = {
                "url": url,
                "title": title,
                "domain": cat["domain"],
                "domain_name": cat["domain_name"],
                "category": cat["category"],
                "content": content,
                "scraped_at": datetime.now().isoformat()
            }
            return page_data, links
        except Exception:
            return None

    def run(self):
        queue = [self.normalize_url(url) for url in SCRAPER_SETTINGS["PRIMARY_URLS"] 
                if self.normalize_url(url) not in self.visited_urls]
        
        pages_count = 0
        
        while queue and pages_count < self.max_pages:
            url = queue.pop(0)
            result = self.scrape_page(url)
            
            if result:
                page_data, new_links = result
                self.pages_data.append(page_data)
                self.stats[page_data["domain"]] += 1
                pages_count += 1
                
                # Update progress if callback provided
                if self.update_callback:
                    self.update_callback(self.task_id, pages_count, self.max_pages, url)
                
                for link in new_links:
                    if link not in self.visited_urls and link not in queue:
                        queue.append(link)
                        
        return self.pages_data