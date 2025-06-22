import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitWindow:
    """Rate limit window tracking."""
    requests: deque
    limit: int
    window_seconds: int
    
    def __post_init__(self):
        if not isinstance(self.requests, deque):
            self.requests = deque(self.requests or [])


class InMemoryRateLimiter:
    """In-memory rate limiter for API requests."""
    
    def __init__(self):
        self._windows: Dict[str, Dict[str, RateLimitWindow]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._cleanup_interval = 3600  # Cleanup every hour
        self._last_cleanup = time.time()
    
    def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int,
        endpoint: str = "default"
    ) -> Tuple[bool, Dict]:
        """
        Check if request is allowed within rate limit.
        
        Args:
            key: Unique identifier (usually IP address)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            endpoint: Endpoint identifier for separate limits
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        with self._lock:
            current_time = time.time()
            
            # Cleanup old entries periodically
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_entries()
                self._last_cleanup = current_time
            
            # Get or create window for this key/endpoint
            if key not in self._windows:
                self._windows[key] = {}
            
            if endpoint not in self._windows[key]:
                self._windows[key][endpoint] = RateLimitWindow(
                    requests=deque(),
                    limit=limit,
                    window_seconds=window_seconds
                )
            
            window = self._windows[key][endpoint]
            
            # Remove expired requests
            cutoff_time = current_time - window_seconds
            while window.requests and window.requests[0] < cutoff_time:
                window.requests.popleft()
            
            # Check if request is allowed
            current_count = len(window.requests)
            is_allowed = current_count < limit
            
            # Add current request if allowed
            if is_allowed:
                window.requests.append(current_time)
            
            # Calculate reset time
            if window.requests:
                oldest_request = window.requests[0]
                reset_time = oldest_request + window_seconds
            else:
                reset_time = current_time + window_seconds
            
            # Prepare rate limit info
            rate_limit_info = {
                "requests_remaining": max(0, limit - len(window.requests)),
                "requests_limit": limit,
                "window_reset_timestamp": datetime.fromtimestamp(reset_time),
                "retry_after_seconds": None if is_allowed else int(reset_time - current_time)
            }
            
            return is_allowed, rate_limit_info
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory leaks."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, endpoints in self._windows.items():
            endpoints_to_remove = []
            
            for endpoint, window in endpoints.items():
                # Remove expired requests
                cutoff_time = current_time - window.window_seconds
                while window.requests and window.requests[0] < cutoff_time:
                    window.requests.popleft()
                
                # If no recent requests, mark endpoint for removal
                if not window.requests:
                    endpoints_to_remove.append(endpoint)
            
            # Remove empty endpoints
            for endpoint in endpoints_to_remove:
                del endpoints[endpoint]
            
            # If no endpoints left, mark key for removal
            if not endpoints:
                keys_to_remove.append(key)
        
        # Remove empty keys
        for key in keys_to_remove:
            del self._windows[key]
        
        logger.debug(f"Cleaned up {len(keys_to_remove)} inactive rate limit keys")
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        with self._lock:
            stats = {
                "total_keys": len(self._windows),
                "total_endpoints": sum(len(endpoints) for endpoints in self._windows.values()),
                "memory_usage_bytes": self._estimate_memory_usage()
            }
            return stats
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimation
        base_size = 0
        for key, endpoints in self._windows.items():
            base_size += len(key) * 8  # String overhead
            for endpoint, window in endpoints.items():
                base_size += len(endpoint) * 8
                base_size += len(window.requests) * 8  # Each timestamp
                base_size += 64  # Window object overhead
        return base_size
    
    def reset_key(self, key: str, endpoint: str = None):
        """Reset rate limit for a specific key/endpoint."""
        with self._lock:
            if key in self._windows:
                if endpoint:
                    if endpoint in self._windows[key]:
                        del self._windows[key][endpoint]
                else:
                    del self._windows[key]


class RedisRateLimiter:
    """Redis-based rate limiter for distributed deployments."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self._lua_script = None
        self._load_lua_script()
    
    def _load_lua_script(self):
        """Load rate limiting Lua script."""
        lua_script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        -- Remove expired entries
        redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)
        
        -- Count current requests
        local current_count = redis.call('ZCARD', key)
        
        if current_count < limit then
            -- Add current request
            redis.call('ZADD', key, current_time, current_time)
            redis.call('EXPIRE', key, window)
            
            -- Get oldest request for reset calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local reset_time = current_time + window
            if #oldest > 0 then
                reset_time = tonumber(oldest[2]) + window
            end
            
            return {1, limit - current_count - 1, reset_time, -1}
        else
            -- Get oldest request for retry calculation
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = window
            if #oldest > 0 then
                retry_after = math.ceil(tonumber(oldest[2]) + window - current_time)
            end
            
            return {0, 0, current_time + window, retry_after}
        end
        """
        self._lua_script = self.redis.register_script(lua_script)
    
    def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int,
        endpoint: str = "default"
    ) -> Tuple[bool, Dict]:
        """
        Check if request is allowed within rate limit using Redis.
        
        Args:
            key: Unique identifier (usually IP address)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            endpoint: Endpoint identifier for separate limits
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        try:
            redis_key = f"rate_limit:{endpoint}:{key}"
            current_time = time.time()
            
            # Execute Lua script
            if self._lua_script is not None:
                result = self._lua_script(
                    keys=[redis_key],
                    args=[window_seconds, limit, current_time]
                )
            else:
                # Fallback if script not loaded
                return True, {
                    "requests_remaining": limit - 1,
                    "requests_limit": limit,
                    "window_reset_timestamp": datetime.fromtimestamp(time.time() + window_seconds),
                    "retry_after_seconds": None
                }
            
            is_allowed = bool(result[0])
            requests_remaining = int(result[1])
            reset_timestamp = float(result[2])
            retry_after = int(result[3]) if result[3] > 0 else None
            
            rate_limit_info = {
                "requests_remaining": requests_remaining,
                "requests_limit": limit,
                "window_reset_timestamp": datetime.fromtimestamp(reset_timestamp),
                "retry_after_seconds": retry_after
            }
            
            return is_allowed, rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fallback to allowing request on Redis failure
            return True, {
                "requests_remaining": limit - 1,
                "requests_limit": limit,
                "window_reset_timestamp": datetime.fromtimestamp(time.time() + window_seconds),
                "retry_after_seconds": None
            }
    
    def reset_key(self, key: str, endpoint: str = "default"):
        """Reset rate limit for a specific key/endpoint."""
        try:
            redis_key = f"rate_limit:{endpoint}:{key}"
            if self.redis:
                self.redis.delete(redis_key)
        except Exception as e:
            logger.error(f"Error resetting rate limit key: {e}")
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "type": "redis",
            "connected": self.redis is not None
        }


def get_client_ip(request, trust_forwarded_for: bool = True, forwarded_header: str = "X-Forwarded-For") -> str:
    """
    Extract client IP address from request, handling proxy forwarding.
    
    Args:
        request: FastAPI request object
        trust_forwarded_for: Whether to trust forwarded headers
        forwarded_header: Header name for forwarded IPs
        
    Returns:
        str: Client IP address
    """
    if trust_forwarded_for and forwarded_header in request.headers:
        # Get first IP from forwarded header (original client)
        forwarded_ips = request.headers[forwarded_header].split(',')
        client_ip = forwarded_ips[0].strip()
        
        # Validate IP format (basic check)
        if _is_valid_ip(client_ip):
            return client_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


def _is_valid_ip(ip: str) -> bool:
    """Basic IP address validation."""
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        for part in parts:
            if not (0 <= int(part) <= 255):
                return False
        
        return True
    except (ValueError, AttributeError):
        return False


class RateLimitManager:
    """Main rate limit manager that handles different endpoints and limits."""
    
    def __init__(self, redis_client=None):
        if redis_client:
            self.limiter = RedisRateLimiter(redis_client)
        else:
            self.limiter = InMemoryRateLimiter()
        
        # Default rate limits
        self.default_limits = {
            "general": {"limit": 100, "window": 86400},  # 100 requests per day
            "search": {"limit": 50, "window": 3600},     # 50 searches per hour
            "question": {"limit": 10, "window": 3600},   # 10 questions per hour
            "session": {"limit": 20, "window": 3600},    # 20 session operations per hour
        }
    
    def check_rate_limit(
        self, 
        client_ip: str, 
        endpoint: str, 
        custom_limit: Optional[int] = None,
        custom_window: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """
        Check rate limit for a specific endpoint.
        
        Args:
            client_ip: Client IP address
            endpoint: Endpoint name
            custom_limit: Override default limit
            custom_window: Override default window
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        # Get limits for endpoint
        limits = self.default_limits.get(endpoint, self.default_limits["general"])
        limit = custom_limit or limits["limit"]
        window = custom_window or limits["window"]
        
        return self.limiter.is_allowed(client_ip, limit, window, endpoint)
    
    def update_limits(self, endpoint: str, limit: int, window: int):
        """Update rate limits for an endpoint."""
        self.default_limits[endpoint] = {"limit": limit, "window": window}
    
    def reset_client(self, client_ip: str, endpoint: str = None):
        """Reset rate limits for a client."""
        if endpoint:
            self.limiter.reset_key(client_ip, endpoint)
        else:
            # Reset all endpoints for this client
            for endpoint_name in self.default_limits.keys():
                self.limiter.reset_key(client_ip, endpoint_name)
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return self.limiter.get_stats()