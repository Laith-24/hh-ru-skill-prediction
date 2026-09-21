"""
HH.ru Authentication Module
Handles OAuth2 token management with unlimited lifetime
"""

import requests
import json
import os
import time
from datetime import datetime
from typing import Optional

class HHAuth:
    """Manages hh.ru OAuth2 authentication"""
    
    TOKEN_FILE = "hh_token.json"
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://hh.ru"
        self.api_url = "https://api.hh.ru"
        self.token_data = None
        
    def get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get access token. Loads from file if available, otherwise requests new one.
        Token has unlimited lifetime once obtained.
        """
        # Try to load existing token
        if not force_refresh and self._load_token():
            print("Loaded existing token from file")
            return self.token_data.get('access_token')
        
        # Request new token (only once!)
        print("Requesting new access token...")
        
        url = f"{self.base_url}/oauth/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        try:
            response = requests.post(url, data=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                self.token_data = response.json()
                self._save_token()
                print("Access token obtained and saved")
                return self.token_data.get('access_token')
            else:
                print(f"Token request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Token request error: {e}")
            return None
    
    def _load_token(self) -> bool:
        """Load token from file"""
        if os.path.exists(self.TOKEN_FILE):
            try:
                with open(self.TOKEN_FILE, 'r') as f:
                    self.token_data = json.load(f)
                return True
            except:
                pass
        return False
    
    def _save_token(self):
        """Save token to file for reuse"""
        with open(self.TOKEN_FILE, 'w') as f:
            json.dump(self.token_data, f, indent=2)
    
    def get_headers(self) -> dict:
        """Get authorization headers for API requests"""
        token = self.get_access_token()
        if token:
            return {
                'Authorization': f'Bearer {token}',
                'User-Agent': 'BigDataCourseProject/1.0 (student@university.edu)',
                'Content-Type': 'application/json'
            }
        return {}