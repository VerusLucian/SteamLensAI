import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIConfig:
    """Configuration class for SteamLens AI API."""
    
    # API Server settings
    host: str
    port: int
    workers: int
    debug: bool
    
    # Rate limiting
    rate_limit_requests_per_day: int
    rate_limit_questions_per_hour: int
    redis_url: Optional[str]
    
    # Firebase tunnel settings
    trust_forwarded_for: bool
    forwarded_for_header: str
    
    # Session settings
    session_timeout_hours: int
    max_concurrent_sessions: int
    
    # Security
    api_key_required: bool
    allowed_origins: list
    
    # Logging
    log_level: str
    log_file: Optional[str]
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create configuration from environment variables."""
        return cls(
            # API Server settings
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            
            # Rate limiting
            rate_limit_requests_per_day=int(os.getenv("RATE_LIMIT_REQUESTS_PER_DAY", "100")),
            rate_limit_questions_per_hour=int(os.getenv("RATE_LIMIT_QUESTIONS_PER_HOUR", "10")),
            redis_url=os.getenv("REDIS_URL"),
            
            # Firebase tunnel settings
            trust_forwarded_for=os.getenv("TRUST_FORWARDED_FOR", "true").lower() == "true",
            forwarded_for_header=os.getenv("FORWARDED_FOR_HEADER", "X-Forwarded-For"),
            
            # Session settings
            session_timeout_hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "24")),
            max_concurrent_sessions=int(os.getenv("MAX_CONCURRENT_SESSIONS", "100")),
            
            # Security
            api_key_required=os.getenv("API_KEY_REQUIRED", "false").lower() == "true",
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
            
            # Logging
            log_level=os.getenv("API_LOG_LEVEL", "INFO"),
            log_file=os.getenv("API_LOG_FILE"),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.rate_limit_requests_per_day < 1:
            raise ValueError("rate_limit_requests_per_day must be positive")
        if self.rate_limit_questions_per_hour < 1:
            raise ValueError("rate_limit_questions_per_hour must be positive")
        if self.session_timeout_hours < 1:
            raise ValueError("session_timeout_hours must be positive")
        if self.max_concurrent_sessions < 1:
            raise ValueError("max_concurrent_sessions must be positive")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log_level must be a valid logging level")