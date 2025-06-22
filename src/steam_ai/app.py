import os
import sys
import logging
import time
from typing import List, Optional
from pathlib import Path

from .config import Config
from .steam_client import SteamReviewFetcher, ReviewProcessor
from .embedding_service import EmbeddingManager
from .index_manager import FAISSIndexManager
from .llm_client import LLMClient, PromptContext
from .session_manager import SessionManager
from .i18n import TranslationManager
from .steam_store_api import SteamStoreAPI, GameDetails

logger = logging.getLogger(__name__)


class SteamAIApp:
    """Main application class with optimized workflow and error handling."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the application with configuration."""
        self.config = config or Config.from_env()
        self.config.validate()
        
        # Initialize components
        self.steam_fetcher = SteamReviewFetcher(self.config)
        self.review_processor = ReviewProcessor(self.config)
        self.embedding_manager = EmbeddingManager(self.config)
        self.index_manager = FAISSIndexManager(self.config)
        self.llm_client = LLMClient(self.config)
        self.session_manager = SessionManager(self.config)
        self.translation_manager = TranslationManager(self.config)
        self.store_api = SteamStoreAPI(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Application state
        self.current_app_id: Optional[str] = None
        self.current_app_name: Optional[str] = None
        self.current_game_details: Optional[GameDetails] = None
        self.current_session_loaded = False
        self.should_exit = False
        
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(logs_dir / 'steam_ai.log')
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    def run_interactive(self) -> None:
        """Run the application in interactive mode."""
        try:
            print(self.translation_manager.tr("welcome"))
            
            # Main application loop
            while True:
                try:
                    if not self.current_session_loaded:
                        if not self._setup_session():
                            continue
                    
                    # Interactive Q&A loop
                    self._qa_loop()
                    
                    # Check if user wants to exit after Q&A loop
                    if self.should_exit:
                        break
                    
                except KeyboardInterrupt:
                    print(f"\n{self.translation_manager.tr('goodbye')}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in main loop: {e}")
                    print(f"❌ {self.translation_manager.tr('unexpected_error', error=str(e))}")
                    
                    # Ask if user wants to continue
                    if not self._ask_yes_no(self.translation_manager.tr("continue_after_error")):
                        break
                        
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            print(f"❌ Fatal error: {e}")
            sys.exit(1)
    
    def _setup_session(self) -> bool:
        """Setup or load a session for an app."""
        try:
            # Get game name or ID from user
            query = input(self.translation_manager.tr("ask_game_name")).strip()
            if not query:
                return False
            
            # Check for exit commands
            if query.lower() in {'exit', 'quit', 'q', self.translation_manager.tr('exit_cmd')}:
                raise KeyboardInterrupt()
            
            # Check if input looks like an app ID (numbers only)
            if query.isdigit():
                app_id = query
                # Try to get game details
                self.current_game_details = self.store_api.get_game_details(app_id)
                if self.current_game_details:
                    self.current_app_id = app_id
                    self.current_app_name = self.current_game_details.name
                    print(f"🎮 Found: {self.current_app_name}")
                else:
                    print(f"⚠️ {self.translation_manager.tr('game_not_found')}")
                    return False
            else:
                # Search for games by name
                print(f"🔍 {self.translation_manager.tr('searching_games', query=query)}")
                search_results = self.store_api.search_and_get_details(query, max_results=5)
                
                if not search_results:
                    print(f"❌ {self.translation_manager.tr('no_games_found')}")
                    return False
                
                # Display search results
                print(f"\n{self.store_api.format_search_results_for_user(search_results)}")
                
                # Ask user to select a game
                while True:
                    try:
                        choice = input(self.translation_manager.tr("select_game")).strip()
                        if not choice:
                            return False
                        
                        # Check for exit commands in game selection
                        if choice.lower() in {'exit', 'quit', 'q', self.translation_manager.tr('exit_cmd')}:
                            raise KeyboardInterrupt()
                        
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(search_results):
                            selected_result, selected_details = search_results[choice_num - 1]
                            self.current_app_id = selected_result.appid
                            self.current_app_name = selected_result.name
                            self.current_game_details = selected_details
                            print(f"✅ {self.translation_manager.tr('selected_game', name=self.current_app_name)}")
                            break
                        else:
                            print(f"❌ {self.translation_manager.tr('invalid_selection')}")
                    except ValueError:
                        print(f"❌ {self.translation_manager.tr('invalid_number')}")
            
            # Check for existing session
            if self._check_existing_session(self.current_app_id):
                return True
            
            # Create new session
            return self._create_new_session(self.current_app_id)
            
        except KeyboardInterrupt:
            # Let KeyboardInterrupt propagate to main loop for graceful exit
            raise
        except Exception as e:
            logger.error(f"Error setting up session: {e}")
            print(f"❌ {self.translation_manager.tr('session_setup_error', error=str(e))}")
            return False
    
    def _check_existing_session(self, app_id: str) -> bool:
        """Check if existing session can be loaded."""
        session_data = self.session_manager.load_session(app_id)
        if session_data:
            if self._ask_yes_no(self.translation_manager.tr("use_saved", app_id=app_id)):
                try:
                    reviews, processed_texts, embeddings, index, index_metadata, session_metadata = session_data
                    
                    # Load into managers
                    self.index_manager.index = index
                    self.index_manager.metadata = index_metadata
                    self.index_manager.optimize_index()
                    
                    print(f"✅ {self.translation_manager.tr('session_loaded', count=len(reviews))}")
                    self.current_session_loaded = True
                    return True
                    
                except Exception as e:
                    logger.error(f"Error loading session: {e}")
                    print(f"❌ {self.translation_manager.tr('session_load_error', error=str(e))}")
                    return False
        
        return False
    
    def _create_new_session(self, app_id: str) -> bool:
        """Create a new session by fetching and processing reviews."""
        try:
            print(f"🔄 {self.translation_manager.tr('creating_session')}")
            
            # Estimate total reviews
            estimated_total = self.steam_fetcher.estimate_total_reviews(app_id)
            if estimated_total:
                print(f"📊 Estimated total reviews: {estimated_total}")
                target_count = min(self.config.target_reviews, estimated_total)
            else:
                target_count = self.config.target_reviews
            
            # Fetch reviews with streaming
            all_reviews = []
            all_processed_texts = []
            all_embeddings = []
            
            print(f"📥 Fetching up to {target_count} reviews...")
            
            for review_chunk in self.steam_fetcher.fetch_reviews_streaming(app_id, target_count):
                # Process reviews for quality
                filtered_reviews = self.review_processor.filter_reviews_by_quality(review_chunk)
                deduplicated_reviews = self.review_processor.deduplicate_reviews(filtered_reviews)
                
                # Process for embeddings (include game details)
                processed_texts = self.review_processor.process_reviews_for_embedding(
                    deduplicated_reviews, 
                    self.current_game_details
                )
                
                # Generate embeddings
                embeddings = self.embedding_manager.process_texts_optimized(processed_texts, enable_streaming=True)
                
                # Accumulate results
                all_reviews.extend(deduplicated_reviews)
                all_processed_texts.extend(processed_texts)
                all_embeddings.extend(embeddings)
                
                # Show progress
                print(f"📈 Processed {len(all_reviews)} reviews so far...")
                
                # Memory optimization
                if len(all_reviews) % 1000 == 0:
                    self.embedding_manager.service.optimize_memory()
            
            if not all_reviews:
                print(f"❌ {self.translation_manager.tr('no_reviews_found')}")
                return False
            
            print(f"✅ Fetched {len(all_reviews)} unique reviews")
            
            # Create optimized index
            print("🔧 Creating optimized search index...")
            index = self.index_manager.create_optimized_index(all_embeddings)
            
            # Save session
            print("💾 Saving session...")
            if self.index_manager.metadata is not None:
                self.session_manager.save_session(
                    app_id=app_id,
                    reviews=all_reviews,
                    processed_texts=all_processed_texts,
                    embeddings=all_embeddings,
                    index=index,
                    index_metadata=self.index_manager.metadata,
                    app_name=self.current_app_name,
                    compression=CompressionType.GZIP
                )
            
            print(f"✅ {self.translation_manager.tr('session_ready')}")
            self.current_session_loaded = True
            
            # Show session stats
            self._show_session_stats(all_reviews, all_embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            print(f"❌ {self.translation_manager.tr('session_create_error', error=str(e))}")
            return False
    
    def _qa_loop(self) -> None:
        """Interactive question-answering loop."""
        print(f"\n✅ {self.translation_manager.tr('ready')}")
        print(self.translation_manager.tr('exit_hint'))
        
        while True:
            try:
                # Get question from user
                question = input(f"\n{self.translation_manager.tr('your_question')}").strip()
                
                if not question:
                    continue
                
                # Check for exit commands
                if question.lower() in {'exit', 'quit', 'q', self.translation_manager.tr('exit_cmd')}:
                    self.should_exit = True
                    break
                
                # Check for special commands
                if self._handle_special_commands(question):
                    continue
                
                # Process question
                self._process_question(question)
                
            except KeyboardInterrupt:
                print(f"\n{self.translation_manager.tr('interrupted')}")
                self.should_exit = True
                break
            except EOFError:
                # Handle end of input gracefully (e.g., when piping input)
                print(f"\n{self.translation_manager.tr('goodbye')}")
                self.should_exit = True
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"❌ {self.translation_manager.tr('question_error', error=str(e))}")
    
    def _process_question(self, question: str) -> None:
        """Process a user question and generate response."""
        try:
            # Search for relevant reviews
            print(f"🔍 {self.translation_manager.tr('searching')}")
            
            # Get query embedding
            query_embedding = self.embedding_manager.service.get_single_embedding(question)
            
            # Search index
            distances, indices = self.index_manager.search(query_embedding)
            
            # Get relevant texts (need to load them from session)
            if self.current_app_id is None:
                print(f"❌ {self.translation_manager.tr('session_error')}")
                return
            
            session_data = self.session_manager.load_session(self.current_app_id)
            if not session_data:
                print(f"❌ {self.translation_manager.tr('session_error')}")
                return
            
            _, processed_texts, _, _, _, _ = session_data
            relevant_texts = [processed_texts[i] for i in indices if i < len(processed_texts)]
            
            if not relevant_texts:
                print(f"❌ {self.translation_manager.tr('no_relevant_reviews')}")
                return
            
            # Generate response
            print(f"🤖 {self.translation_manager.tr('generating')}")
            
            # Enhanced context with game details
            additional_context = {}
            if self.current_game_details:
                additional_context["game_summary"] = self.current_game_details.get_summary_text()
                
            context = PromptContext(
                reviews=relevant_texts,
                question=question,
                app_name=self.current_app_name,
                additional_context=additional_context
            )
            
            # Show progress indicator
            import threading
            import itertools
            
            stop_event = threading.Event()
            
            def spin():
                for ch in itertools.cycle('|/-\\'):
                    if stop_event.is_set():
                        break
                    sys.stdout.write(f"\r{self.translation_manager.tr('generating')} {ch}")
                    sys.stdout.flush()
                    time.sleep(0.1)
                sys.stdout.write("\r")
                sys.stdout.flush()
            
            spinner_thread = threading.Thread(target=spin)
            spinner_thread.start()
            
            try:
                response = self.llm_client.generate_response(context)
                
                # Show response
                print(f"\n{self.translation_manager.tr('answer')}")
                print("-" * 50)
                print(response.text)
                print("-" * 50)
                
                # Show metadata if in debug mode
                if os.getenv('DEBUG', '').lower() == 'true':
                    print("\n🔧 Debug info:")
                    print(f"   Processing time: {response.processing_time:.2f}s")
                    print(f"   Model: {response.model}")
                    print(f"   Cached: {response.cached}")
                    print(f"   Relevant reviews: {len(relevant_texts)}")
                
            finally:
                stop_event.set()
                spinner_thread.join()
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"❌ {self.translation_manager.tr('processing_error', error=str(e))}")
    
    def _handle_special_commands(self, command: str) -> bool:
        """Handle special commands like stats, help, etc."""
        command_lower = command.lower().strip()
        
        if command_lower in ['help', 'h', '?']:
            self._show_help()
            return True
        elif command_lower in ['stats', 'statistics']:
            self._show_detailed_stats()
            return True
        elif command_lower in ['sessions', 'list']:
            self._list_sessions()
            return True
        elif command_lower in ['clear', 'cls']:
            os.system('cls' if os.name == 'nt' else 'clear')
            return True
        elif command_lower.startswith('switch '):
            query = command_lower[7:].strip()
            self._switch_session(query)
            return True
        elif command_lower in ['reload', 'refresh']:
            self._reload_session()
            return True
        
        return False
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
🔧 Available Commands:
   help, h, ?         - Show this help
   stats              - Show detailed statistics
   sessions, list     - List all sessions
   switch <app_id>    - Switch to different app session
   reload, refresh    - Reload current session
   clear, cls         - Clear screen
   exit, quit, q      - Exit application

💡 Tips:
   - Ask specific questions about game features, performance, bugs
   - Try questions like "Is this game worth buying?" or "What are the main issues?"
   - Use detailed questions for better responses
        """
        print(help_text)
    
    def _show_session_stats(self, reviews: List, embeddings: List) -> None:
        """Show basic session statistics."""
        print("\n📊 Session Stats:")
        print(f"   Reviews: {len(reviews)}")
        print(f"   Embeddings: {len(embeddings)}")
        if embeddings:
            print(f"   Embedding dimension: {len(embeddings[0])}")
        
        # Memory usage
        memory_stats = self.index_manager.get_memory_usage()
        print(f"   Index memory: {memory_stats.get('total_mb', 0):.1f} MB")
    
    def _show_detailed_stats(self) -> None:
        """Show detailed application statistics."""
        print("\n📊 Detailed Statistics:")
        print(f"Current Session: {self.current_app_name} ({self.current_app_id})")
        
        # Index stats
        index_info = self.index_manager.get_index_info()
        if index_info.get('status') == 'loaded':
            print("\n🔍 Index Info:")
            print(f"   Type: {index_info.get('index_type', 'unknown')}")
            print(f"   Vectors: {index_info.get('ntotal', 0)}")
            print(f"   Dimension: {index_info.get('dimension', 0)}")
            print(f"   Memory: {self.index_manager.get_memory_usage().get('total_mb', 0):.1f} MB")
        
        # Embedding service stats
        embedding_stats = self.embedding_manager.get_service_stats()
        print("\n🔧 Embedding Service:")
        print(f"   Cache hits: {embedding_stats['cache_stats']['cache_hits']}")
        print(f"   Cache misses: {embedding_stats['cache_stats']['cache_misses']}")
        print(f"   Hit rate: {embedding_stats['cache_stats']['hit_rate']:.2%}")
        
        # LLM client stats
        llm_stats = self.llm_client.get_cache_stats()
        print("\n🤖 LLM Service:")
        print(f"   Response cache hits: {llm_stats['cache_hits']}")
        print(f"   Response cache misses: {llm_stats['cache_misses']}")
        print(f"   Hit rate: {llm_stats['hit_rate']:.2%}")
        
        # Session manager stats
        session_stats = self.session_manager.get_session_stats()
        print("\n💾 Session Storage:")
        print(f"   Total sessions: {session_stats['total_sessions']}")
        print(f"   Total size: {session_stats['total_size_mb']:.1f} MB")
        print(f"   Current version: {session_stats['current_version']}")
    
    def _list_sessions(self) -> None:
        """List all available sessions."""
        sessions = self.session_manager.list_sessions()
        
        if not sessions:
            print("📭 No sessions found")
            return
        
        print(f"\n📂 Available Sessions ({len(sessions)}):")
        print("-" * 80)
        print(f"{'App ID':<12} {'Name':<30} {'Reviews':<10} {'Version':<10} {'Modified'}")
        print("-" * 80)
        
        for session in sessions[:20]:  # Show max 20
            app_id = session.get('app_id', 'unknown')[:11]
            app_name = session.get('app_name', 'Unknown')[:29]
            review_count = session.get('total_reviews', 0)
            version = session.get('version', 'unknown')[:9]
            
            # Format modification time
            mod_time = session.get('last_modified', 0)
            if mod_time:
                import datetime
                mod_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
            else:
                mod_str = 'unknown'
            
            print(f"{app_id:<12} {app_name:<30} {review_count:<10} {version:<10} {mod_str}")
        
        if len(sessions) > 20:
            print(f"... and {len(sessions) - 20} more")
    
    def _switch_session(self, query: str) -> None:
        """Switch to a different session."""
        if query == self.current_app_id:
            print("Already using that session")
            return
        
        # If query is a number, treat as app ID
        if query.isdigit():
            self.current_app_id = query
            self.current_game_details = None
            self.current_session_loaded = False
            print(f"🔄 Switching to app {query}...")
        else:
            # Search for game by name
            search_results = self.store_api.search_and_get_details(query, max_results=3)
            if search_results:
                first_result, first_details = search_results[0]
                self.current_app_id = first_result.appid
                self.current_app_name = first_result.name
                self.current_game_details = first_details
                self.current_session_loaded = False
                print(f"🔄 Switching to {first_result.name} ({first_result.appid})...")
            else:
                print(f"❌ Game not found: {query}")
    
    def _reload_session(self) -> None:
        """Reload the current session."""
        if not self.current_app_id:
            print("❌ No session loaded")
            return
        
        print(f"🔄 Reloading session for {self.current_app_id}...")
        self.current_session_loaded = False
    
    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question."""
        yes_text = self.translation_manager.tr("yes")
        no_text = self.translation_manager.tr("no")
        
        if default:
            prompt = f"{question} ({yes_text}/{no_text}) [{yes_text}]: "
        else:
            prompt = f"{question} ({yes_text}/{no_text}) [{no_text}]: "
        
        while True:
            response = input(prompt).strip().lower()
            
            if not response:
                return default
            
            # Check for exit commands
            if response in {'exit', 'quit', 'q', self.translation_manager.tr('exit_cmd')}:
                raise KeyboardInterrupt()
            
            if response in [yes_text.lower(), 'y', 'yes', '1', 'true']:
                return True
            elif response in [no_text.lower(), 'n', 'no', '0', 'false']:
                return False
            else:
                print(f"Please answer {yes_text} or {no_text}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Create and run application
        app = SteamAIApp(config)
        app.run_interactive()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()