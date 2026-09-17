"""
HH.ru Data Collector - Backend Module
Handles rate-limited data collection with checkpointing
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional, Callable
from tqdm import tqdm

class HHCollector:
    """Rate-limited collector for hh.ru vacancy data"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.hh.ru"
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'BigDataCourseProject/1.0',
            'Content-Type': 'application/json'
        }
        self.vacancies = []
        self.seen_ids = set()
        self.duplicate_skipped = 0
        self.new_collected = 0  
        self.request_count = 0
        self.last_request_time = 0
        self.consecutive_errors = 0
        self.is_paused = False
        
    def rate_limit(self):
        """Respect rate limits - 1 second between requests"""
        if self.is_paused:
            time.sleep(5)
            return
            
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self.last_request_time = time.time()
    
    def exponential_backoff(self):
        """Exponential backoff when rate limited"""
        self.consecutive_errors += 1
        wait_time = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
        print(f"  Rate limited. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
    
    def search_vacancies(self, text: str, page: int = 0, area: int = 1, only_with_salary: bool = False) -> Dict:
        """Search for vacancies with optional salary filter"""
        self.rate_limit()
        
        url = f"{self.base_url}/vacancies"
        params = {
            'text': text,
            'area': area,
            'page': page,
            'per_page': 100
        }
        
        # Add salary filter if requested
        if only_with_salary:
            params['only_with_salary'] = True
        
        self.request_count += 1
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if response.status_code == 200:
                self.consecutive_errors = 0
                return response.json()
            elif response.status_code == 403 or response.status_code == 429:
                self.exponential_backoff()
                return self.search_vacancies(text, page, area, only_with_salary)
            else:
                print(f"  Error {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"  Exception: {e}")
            self.exponential_backoff()
            return {}
    
    def get_vacancy_details(self, vacancy_id: str) -> Dict:
        """Get detailed vacancy information"""
        self.rate_limit()
        
        url = f"{self.base_url}/vacancies/{vacancy_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}
    
    def collect_for_query(self, query: str, area: int = 1, 
                      max_pages: int = 20,
                      only_with_salary: bool = False,
                      progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Collect all vacancies for a search query (no duplicates)"""
        collected = []
        new_count = 0
        duplicate_count = 0
        
        for page in range(max_pages):
            if progress_callback:
                progress_callback(f"Page {page + 1}/{max_pages}", page, max_pages)
            
            result = self.search_vacancies(query, page, area, only_with_salary)
            items = result.get('items', [])
            
            if not items:
                break
            
            for item in items:
                vacancy_id = item['id']
                
                # CHECK FOR DUPLICATE BEFORE FETCHING DETAILS
                if vacancy_id in self.seen_ids:
                    duplicate_count += 1
                    self.duplicate_skipped += 1
                    continue  # Skip this one - already collected
                
                # Mark as seen BEFORE fetching to prevent concurrent duplicates
                self.seen_ids.add(vacancy_id)
                self.new_collected += 1
                
                details = self.get_vacancy_details(vacancy_id)
                if details:
                    details['_search_query'] = query
                    details['_area_id'] = area
                    details['_collected_at'] = datetime.now().isoformat()
                    collected.append(details)
                    new_count += 1
                
                # Progress update every 10 new items
                if new_count % 10 == 0 and progress_callback:
                    # Call with just message and let the callback handle formatting
                    progress_callback(f"New: {new_count} | Dup: {duplicate_count}", page, max_pages)
            
            if page >= result.get('pages', 0) - 1:
                break
        
        # Print summary for this query
        print(f"  Query '{query}': {new_count} new, {duplicate_count} duplicate (total unique: {len(self.seen_ids)})")
        
        return collected
    
    def collect_large_dataset(self, queries: List[str], areas: List[int],
                         max_pages_per_query: int = 15,
                         only_with_salary: bool = False,
                         progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Collect large dataset across multiple queries and regions (NO DUPLICATES)"""
        all_vacancies = []
        total_queries = len(queries) * len(areas)
        current = 0
        
        # Reset seen IDs at start of collection
        self.seen_ids = set()
        self.vacancies = []
        
        for query in queries:
            for area in areas:
                current += 1
                
                if progress_callback:
                    progress_callback(f"Query: {query} (Area: {area}) - Unique so far: {len(self.seen_ids)}", current, total_queries)
                
                # This returns only NEW vacancies (not seen before)
                new_vacancies = self.collect_for_query(query, area, max_pages_per_query, only_with_salary)
                
                if new_vacancies:
                    all_vacancies.extend(new_vacancies)
                    self.vacancies.extend(new_vacancies)
                    
                    # Save checkpoint after each query with duplicate stats
                    self.save_checkpoint(self.vacancies)
                    
                    if progress_callback:
                        progress_callback(f"Total UNIQUE: {len(self.vacancies)} vacancies (prevented duplicates across queries)", current, total_queries)
        
        return self.vacancies
    
    def save_checkpoint(self, vacancies: List[Dict]):
        """Save intermediate checkpoint"""
        os.makedirs("data/checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/checkpoints/checkpoint_{timestamp}.json"
        
        # Only save essential fields to save space
        checkpoint_data = {
            'timestamp': timestamp,
            'count': len(vacancies),
            'vacancies': vacancies
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Checkpoint: {len(vacancies)} vacancies saved")
    
    def save_final(self, filename: str = None) -> str:
        """Save final dataset"""
        os.makedirs("data/output", exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/output/hh_vacancies_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.vacancies, f, ensure_ascii=False, indent=2)
        
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"Final saved: {filename} ({size_mb:.2f} MB)")
        
        return filename
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        total_size = len(json.dumps(self.vacancies, ensure_ascii=False))
        return {
            'count': len(self.vacancies),
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024),
            'size_gb': total_size / (1024 * 1024 * 1024),
            'requests': self.request_count,
            'avg_size_per_vacancy': total_size / len(self.vacancies) if self.vacancies else 0
        }

    def get_duplicate_stats(self) -> Dict:
        """Get accurate statistics about prevented duplicates"""
        return {
            'unique_vacancies': len(self.seen_ids),
            'new_vacancies_collected': self.new_collected,
            'duplicate_attempts_prevented': self.duplicate_skipped,
            'total_attempts': self.new_collected + self.duplicate_skipped,
            'efficiency_percent': (self.new_collected / max(1, self.new_collected + self.duplicate_skipped)) * 100,
            'api_calls_made': self.request_count
        }
    
    def collect_continuously(self, queries, areas, interval_minutes=10):
        """Run collection continuously at specified intervals"""
        while True:
            print(f"Starting collection cycle at {datetime.now()}")
            
            for query in queries:
                for area in areas:
                    self.collect_for_query(query, area, max_pages=5)
            
            self.save_checkpoint(self.vacancies)
            print(f"   Collected {len(self.vacancies)} new vacancies")
            print(f"Waiting {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
# Predefined search queries for large volume
SEARCH_QUERIES = [
    "Python developer",
    "Java developer",
    "JavaScript developer",
    "Data Scientist",
    "Data Analyst",
    "DevOps engineer",
    "Frontend developer",
    "Backend developer",
    "Fullstack developer",
    "Machine Learning engineer",
    "QA engineer",
    "System administrator",
    "Project manager",
    "Product manager",
    "UX/UI designer",
    "C++ developer",
    "C# developer",
    "Go developer",
    "Ruby developer",
    "PHP developer",
    "Android developer",
    "iOS developer",
    "Security analyst",
    "Database administrator",
    "Network engineer",
    "1C developer",
    "SAP consultant",
    "Oracle developer",
    "Salesforce developer",
    "SharePoint developer",
    "WordPress developer",
    "Shopify developer",
    "Mobile developer",
    "Flutter developer",
    "React Native developer",
    "Unity developer",
    "Game developer",
    "Embedded engineer",
    "Firmware engineer",
    "Hardware engineer",
    "Network architect",
    "Cloud architect",
    "AWS engineer",
    "Azure engineer",
    "GCP engineer",
    "IT project coordinator",
    "Scrum master",
    "Agile coach",
    "IT recruiter",
    "Technical writer",
    "DevSecOps engineer",
    "Site reliability engineer",
    "Data engineer",
    "Business analyst",
    "Systems analyst",
]

# Regions: 1=Moscow, 2=Saint Petersburg, 113=Russia
REGIONS = {
    # "Moscow": 1,
    # "Saint Petersburg": 2,
    # "Novosibirsk": 99,
    # "Ekaterinburg": 3,
    # "Kazan": 88,
    # "Nizhny Novgorod": 66,
    "All Russia": 113
}