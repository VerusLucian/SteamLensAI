"""
Steam AI - Optimized Steam Review Analysis with RAG
==================================================

A high-performance tool for querying Steam reviews using local LLM and FAISS index
with memory optimization, batch processing, and modular architecture.

Main Components:
- SteamAIApp: Main application class
- Config: Configuration management
- SteamReviewFetcher: Steam API client with streaming
- EmbeddingService: Optimized embedding generation
- FAISSIndexManager: Optimized search indexing
- LLMClient: LLM interaction with caching
- SessionManager: Session storage with compression

Example usage:
    >>> from steam_ai import SteamAIApp, Config
    >>> config = Config.from_env()
    >>> app = SteamAIApp(config)
    >>> app.run_interactive()
"""

__version__ = "2.0.0"
__author__ = "Steam AI Team"
__license__ = "CC BY-NC 4.0"

# Main components
from .app import SteamAIApp
from .config import Config

# Core services
from .steam_client import SteamReviewFetcher, ReviewProcessor, Review
from .steam_store_api import SteamStoreAPI, GameDetails, GameSearchResult
from .embedding_service import EmbeddingService, EmbeddingManager
from .index_manager import FAISSIndexManager, IndexMetadata
from .llm_client import LLMClient, PromptContext, PromptTemplate, LLMResponse
from .session_manager import SessionManager, SessionMetadata, CompressionType
from .http_client import HTTPClient, SyncHTTPClient
from .i18n import TranslationManager

# Convenience imports
from .app import main

__all__ = [
    # Main application
    "SteamAIApp",
    "main",
    
    # Configuration
    "Config",
    
    # Steam API
    "SteamReviewFetcher",
    "ReviewProcessor", 
    "Review",
    
    # Steam Store API
    "SteamStoreAPI",
    "GameDetails",
    "GameSearchResult",
    
    # Embeddings
    "EmbeddingService",
    "EmbeddingManager",
    
    # Search index
    "FAISSIndexManager",
    "IndexMetadata",
    
    # LLM client
    "LLMClient",
    "PromptContext",
    "PromptTemplate",
    "LLMResponse",
    
    # Session management
    "SessionManager",
    "SessionMetadata",
    "CompressionType",
    
    # HTTP client
    "HTTPClient",
    "SyncHTTPClient",
    
    # Internationalization
    "TranslationManager",
]


def get_version() -> str:
    """Get the current version of Steam AI."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "steam-ai",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Optimized Steam Review Analysis with RAG",
    }