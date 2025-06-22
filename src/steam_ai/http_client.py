import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class SyncHTTPClient:
    """HTTP client with connection pooling and retry logic using requests."""
    
    def __init__(self, 
                 pool_size: int = 10, 
                 timeout: int = 30,
                 retry_config: Optional[RetryConfig] = None):
        self.pool_size = pool_size
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.retry_config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=self.retry_config.base_delay,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({'User-Agent': 'SteamAI/1.0'})
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
            
        return delay
        
    def get(self, url: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Synchronous GET request."""
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GET request failed: {e}")
            raise
            
    def post(self, url: str, json_data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Synchronous POST request."""
        try:
            response = self.session.post(
                url, 
                json=json_data, 
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"POST request failed: {e}")
            raise
            
    def batch_post(self, 
                  url: str, 
                  payloads: List[Dict[str, Any]], 
                  max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Synchronous batch POST requests with rate limiting."""
        results = []
        
        for i, payload in enumerate(payloads):
            try:
                result = self.post(url, json_data=payload)
                results.append(result)
                
                # Rate limiting between requests
                if i < len(payloads) - 1 and max_concurrent > 1:
                    time.sleep(0.1)  # Small delay between requests
                    
            except Exception as e:
                logger.error(f"Batch request {i} failed: {e}")
                # Continue with other requests
                results.append({"error": str(e)})
                
        return results
    
    def close(self):
        """Close the session."""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Alias for backward compatibility
HTTPClient = SyncHTTPClient