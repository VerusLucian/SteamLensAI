#!/usr/bin/env python3
"""
SteamLens AI API - Main Entry Point

This module serves as the main entry point for the SteamLens AI API server.
It handles configuration loading, logging setup, and starts the FastAPI application.
"""

import os
import sys
import logging
import signal
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    print("Error: uvicorn is not installed. Please install with: pip install -r requirements.txt")

from steam_ai_api.config import APIConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_logging(config: APIConfig):
    """Setup logging configuration."""
    # Set log level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    
    # Add file handler if specified
    if config.log_file:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(config.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            
            logger.info(f"Logging to file: {config.log_file}")
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        # The FastAPI lifespan manager will handle cleanup
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_environment():
    """Validate required environment variables and dependencies."""
    required_dirs = [
        "sessions",  # For session storage
        "i18n"      # For internationalization
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.warning(f"Directory {dir_name} does not exist, creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if Ollama is accessible (optional, will be checked during health checks)
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
        base_url = ollama_url.replace("/api/embed", "/api/tags")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            logger.info("Ollama service is accessible")
        else:
            logger.warning("Ollama service is not accessible, some features may not work")
    except Exception as e:
        logger.warning(f"Could not check Ollama service: {e}")


def main():
    """Main entry point for the API server."""
    try:
        logger.info("Starting SteamLens AI API...")
        
        # Load configuration
        config = APIConfig.from_env()
        config.validate()
        
        logger.info("API Configuration:")
        logger.info(f"  Host: {config.host}")
        logger.info(f"  Port: {config.port}")
        logger.info(f"  Workers: {config.workers}")
        logger.info(f"  Debug: {config.debug}")
        logger.info(f"  Rate Limit (requests/day): {config.rate_limit_requests_per_day}")
        logger.info(f"  Rate Limit (questions/hour): {config.rate_limit_questions_per_hour}")
        logger.info(f"  Session Timeout: {config.session_timeout_hours} hours")
        logger.info(f"  Trust Forwarded Headers: {config.trust_forwarded_for}")
        
        # Setup logging
        setup_logging(config)
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Validate environment
        validate_environment()
        
        # Configure uvicorn
        uvicorn_config = {
            "app": "steam_ai_api.app:app",
            "host": config.host,
            "port": config.port,
            "reload": config.debug,
            "log_level": config.log_level.lower(),
            "access_log": True,
            "use_colors": True,
            "loop": "asyncio"
        }
        
        # Add workers only in production mode
        if not config.debug and config.workers > 1:
            uvicorn_config["workers"] = config.workers
            logger.info(f"Running with {config.workers} workers")
        else:
            logger.info("Running in single worker mode")
        
        # Check if uvicorn is available
        if not UVICORN_AVAILABLE:
            logger.error("uvicorn is not available. Cannot start API server.")
            sys.exit(1)
        
        # Start the server
        logger.info(f"SteamLens AI API starting on http://{config.host}:{config.port}")
        logger.info(f"API Documentation available at http://{config.host}:{config.port}/docs")
        logger.info(f"Health check available at http://{config.host}:{config.port}/health")
        
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)
    finally:
        logger.info("SteamLens AI API shutdown complete")


if __name__ == "__main__":
    main()