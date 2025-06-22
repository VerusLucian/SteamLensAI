#!/usr/bin/env python3
"""
Steam AI - Optimized Main Entry Point
====================================

High-performance Steam review analysis with RAG using modular architecture.
This is the main entry point that uses the optimized components for better
memory management, performance, and maintainability.

Features:
- Memory-optimized streaming processing
- Batch embedding generation
- Connection pooling for HTTP requests
- FAISS index optimization
- Session compression and versioning
- Better error handling and logging
- Modular, maintainable architecture

Usage:
    python main.py

Environment Variables:
    STEAM_TARGET_REVIEWS=600        # Number of reviews to fetch
    OLLAMA_EMBED_MODEL=bge-m3       # Embedding model name
    OLLAMA_LLM_MODEL=gemma2:12b      # LLM model name
    APP_LANGUAGE=en                 # Interface language (en/pl)
    EMBEDDING_BATCH_SIZE=10         # Batch size for embeddings
    LOG_LEVEL=INFO                  # Logging level
    DEBUG=false                     # Enable debug mode
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from steam_ai import SteamAIApp, Config, get_version, get_info
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """Setup logging configuration."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'

    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if debug_mode:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Add file logging if not in debug mode
    if not debug_mode:
        try:
            # Create logs directory if it doesn't exist
            logs_dir = Path('logs')
            logs_dir.mkdir(exist_ok=True)

            file_handler = logging.FileHandler(logs_dir / 'steam_ai.log')
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            print(f"⚠️ Could not setup file logging: {e}")

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing_deps.append("numpy")

    try:
        import faiss  # noqa: F401
    except ImportError:
        missing_deps.append("faiss-cpu")

    try:
        import requests  # noqa: F401
    except ImportError:
        missing_deps.append("requests")

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        # aiohttp is optional for now
        pass

    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing_deps.append("tqdm")

    try:
        from dotenv import load_dotenv  # noqa: F401
    except ImportError:
        missing_deps.append("python-dotenv")

    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall them with:")
        print("pip install -r requirements.txt")
        return False

    return True


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        import requests

        # Check embedding endpoint
        embed_url = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed")
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

        test_payload = {
            "model": embed_model,
            "input": ["test"]
        }

        response = requests.post(embed_url, json=test_payload, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Ollama embedding service not responding properly (status: {response.status_code})")
            print(f"   URL: {embed_url}")
            print(f"   Model: {embed_model}")
            return False

        # Check LLM endpoint
        llm_url = os.getenv("OLLAMA_LLM_URL", "http://localhost:11434/api/generate")
        llm_model = os.getenv("OLLAMA_LLM_MODEL", "gemma2:12b")

        test_payload = {
            "model": llm_model,
            "prompt": "test",
            "stream": False,
            "options": {"num_predict": 1}
        }

        response = requests.post(llm_url, json=test_payload, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Ollama LLM service not responding properly (status: {response.status_code})")
            print(f"   URL: {llm_url}")
            print(f"   Model: {llm_model}")
            return False

        return True

    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        llm_model = os.getenv("OLLAMA_LLM_MODEL", "gemma2:12b")
        print("\nMake sure Ollama is running with the required models:")
        print(f"   ollama pull {embed_model}")
        print(f"   ollama pull {llm_model}")
        return False


def show_startup_info():
    """Show startup information."""
    info = get_info()
    print(f"""
🎮 {info['description']} v{info['version']}
📄 License: {info['license']}

🔧 Configuration:
   Target reviews: {os.getenv('STEAM_TARGET_REVIEWS', '600')}
   Embedding model: {os.getenv('OLLAMA_EMBED_MODEL', 'bge-m3')}
   LLM model: {os.getenv('OLLAMA_LLM_MODEL', 'gemma2:12b')}
   Language: {os.getenv('APP_LANGUAGE', 'en')}
   Batch size: {os.getenv('EMBEDDING_BATCH_SIZE', '10')}
   Log level: {os.getenv('LOG_LEVEL', 'INFO')}

💾 Data directory: {os.getenv('SAVE_DIR', 'sessions')}
🌐 Translation directory: {os.getenv('I18N_DIR', 'i18n')}
""")


def main():
    """Main entry point with comprehensive error handling and validation."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)

        logger.info(f"Starting Steam AI v{get_version()}")

        # Show startup info
        show_startup_info()

        # Check dependencies
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✅ Dependencies OK")

        # Check Ollama connection
        print("🔍 Checking Ollama connection...")
        if not check_ollama_connection():
            print("\n⚠️ Ollama connection failed, but continuing anyway.")
            print("The application may not work properly without Ollama.")

            # Ask user if they want to continue
            try:
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes', '1']:
                    print("Exiting...")
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(1)
        else:
            print("✅ Ollama connection OK")

        # Load configuration
        print("🔧 Loading configuration...")
        try:
            config = Config.from_env()
            config.validate()
            print("✅ Configuration loaded")
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            print(f"❌ Configuration error: {e}")
            sys.exit(1)

        # Create and run application
        print("🚀 Starting application...\n")
        app = SteamAIApp(config)
        app.run_interactive()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")

        # Show debug info if in debug mode
        if os.getenv('DEBUG', 'false').lower() == 'true':
            import traceback
            print("\n🔧 Debug traceback:")
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
