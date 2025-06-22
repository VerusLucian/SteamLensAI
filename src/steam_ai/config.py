import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration class for Steam AI application."""
    
    # Steam API settings
    target_reviews: int
    steam_api_sleep: float
    review_max_length: int
    
    # Ollama settings
    embed_model: str
    llm_model: str
    ollama_embed_url: str
    ollama_llm_url: str
    ollama_timeout_embed: int
    ollama_timeout_llm: int
    
    # Embedding settings
    embedding_batch_size: int
    similarity_top_k: int
    
    # Application settings
    save_dir: str
    app_language: str
    i18n_dir: str
    
    # Performance settings
    max_concurrent_requests: int
    connection_pool_size: int
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            # Steam API settings
            target_reviews=int(os.getenv("STEAM_TARGET_REVIEWS", "600")),
            steam_api_sleep=float(os.getenv("STEAM_API_SLEEP", "0.5")),
            review_max_length=int(os.getenv("REVIEW_MAX_LENGTH", "1024")),
            
            # Ollama settings
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "deepseek-r1:14b"),
            ollama_embed_url=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
            ollama_llm_url=os.getenv("OLLAMA_LLM_URL", "http://localhost:11434/api/generate"),
            ollama_timeout_embed=int(os.getenv("OLLAMA_TIMEOUT_EMBED", "120")),
            ollama_timeout_llm=int(os.getenv("OLLAMA_TIMEOUT_LLM", "180")),
            
            # Embedding settings
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "40")),
            
            # Application settings
            save_dir=os.getenv("SAVE_DIR", "sessions"),
            app_language=os.getenv("APP_LANGUAGE", "pl").lower(),
            i18n_dir=os.getenv("I18N_DIR", "i18n"),
            
            # Performance settings
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
            connection_pool_size=int(os.getenv("CONNECTION_POOL_SIZE", "10")),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.target_reviews <= 0:
            raise ValueError("target_reviews must be positive")
        if self.steam_api_sleep < 0:
            raise ValueError("steam_api_sleep must be non-negative")
        if self.review_max_length <= 0:
            raise ValueError("review_max_length must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if self.similarity_top_k <= 0:
            raise ValueError("similarity_top_k must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.connection_pool_size <= 0:
            raise ValueError("connection_pool_size must be positive")