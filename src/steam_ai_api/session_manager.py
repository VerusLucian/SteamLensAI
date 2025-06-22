import os
import uuid
import json
import logging
import asyncio
import threading
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time

from steam_ai.config import Config
from steam_ai.steam_store_api import SteamStoreAPI
from steam_ai.steam_client import SteamReviewFetcher, Review

from steam_ai.index_manager import FAISSIndexManager
from steam_ai.llm_client import LLMClient

from .models import SessionStatus, SessionInfo

logger = logging.getLogger(__name__)


@dataclass
class APISession:
    """API session information."""
    session_id: str
    appid: str
    game_name: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    reviews_count: int
    target_reviews: int
    progress_percentage: float
    error_message: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Internal tracking
    _download_task: Optional[asyncio.Task] = None
    _lock: Optional[threading.Lock] = None
    
    def __post_init__(self):
        if self._lock is None:
            self._lock = threading.Lock()
    
    def to_session_info(self) -> SessionInfo:
        """Convert to API session info model."""
        return SessionInfo(
            session_id=self.session_id,
            appid=self.appid,
            game_name=self.game_name,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            reviews_count=self.reviews_count,
            target_reviews=self.target_reviews,
            progress_percentage=self.progress_percentage,
            error_message=self.error_message,
            expires_at=self.expires_at
        )
    
    def update_progress(self, reviews_count: int, status: Optional[SessionStatus] = None):
        """Update session progress."""
        if self._lock:
            with self._lock:
                self.reviews_count = reviews_count
                self.progress_percentage = min(100.0, (reviews_count / self.target_reviews) * 100)
                self.updated_at = datetime.now()
                
                if status:
                    self.status = status
    
    def set_error(self, error_message: str):
        """Set session error state."""
        if self._lock:
            with self._lock:
                self.status = SessionStatus.ERROR
                self.error_message = error_message
                self.updated_at = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False


class APISessionManager:
    """Manages API sessions for game review analysis."""
    
    def __init__(self, config: Config, session_timeout_hours: int = 24):
        self.config = config
        self.session_timeout_hours = session_timeout_hours
        self.sessions: Dict[str, APISession] = {}
        self.sessions_lock = threading.RLock()
        
        # Initialize core components
        self.steam_api = SteamStoreAPI(config)
        self.steam_client = SteamReviewFetcher(config)
        
        # Background task tracking
        self._background_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Session persistence
        self.persistence_file = os.path.join(config.save_dir, "api_sessions.json")
        self._load_sessions()
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            logger.warning("No event loop running, session cleanup will be manual")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                self._cleanup_expired_sessions()
                self._save_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        with self.sessions_lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.is_expired()
            ]
            
            for session_id in expired_sessions:
                logger.info(f"Removing expired session: {session_id}")
                self._remove_session(session_id)
    
    def _remove_session(self, session_id: str):
        """Remove a session and clean up resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Cancel any background tasks
            if session._download_task and not session._download_task.done():
                session._download_task.cancel()
            
            # Remove from tracking
            del self.sessions[session_id]
            
            if session_id in self._background_tasks:
                task = self._background_tasks[session_id]
                if not task.done():
                    task.cancel()
                del self._background_tasks[session_id]
    
    def _load_sessions(self):
        """Load sessions from persistence file."""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                
                for session_data in data.get('sessions', []):
                    try:
                        session = APISession(
                            session_id=session_data['session_id'],
                            appid=session_data['appid'],
                            game_name=session_data['game_name'],
                            status=SessionStatus(session_data['status']),
                            created_at=datetime.fromisoformat(session_data['created_at']),
                            updated_at=datetime.fromisoformat(session_data['updated_at']),
                            reviews_count=session_data['reviews_count'],
                            target_reviews=session_data['target_reviews'],
                            progress_percentage=session_data['progress_percentage'],
                            error_message=session_data.get('error_message'),
                            expires_at=datetime.fromisoformat(session_data['expires_at']) if session_data.get('expires_at') else None
                        )
                        
                        # Only restore non-expired sessions
                        if not session.is_expired():
                            self.sessions[session.session_id] = session
                            
                            # Reset status for incomplete sessions
                            if session.status in [SessionStatus.DOWNLOADING, SessionStatus.PROCESSING]:
                                session.status = SessionStatus.INITIALIZING
                        
                    except Exception as e:
                        logger.warning(f"Error loading session data: {e}")
                        continue
                
                logger.info(f"Loaded {len(self.sessions)} sessions from persistence")
        
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    def _save_sessions(self):
        """Save sessions to persistence file."""
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            
            sessions_data = []
            with self.sessions_lock:
                for session in self.sessions.values():
                    session_data = {
                        'session_id': session.session_id,
                        'appid': session.appid,
                        'game_name': session.game_name,
                        'status': session.status.value,
                        'created_at': session.created_at.isoformat(),
                        'updated_at': session.updated_at.isoformat(),
                        'reviews_count': session.reviews_count,
                        'target_reviews': session.target_reviews,
                        'progress_percentage': session.progress_percentage,
                        'error_message': session.error_message,
                        'expires_at': session.expires_at.isoformat() if session.expires_at else None
                    }
                    sessions_data.append(session_data)
            
            data = {
                'sessions': sessions_data,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    async def init_session(self, appid: str) -> Tuple[bool, Optional[APISession], str]:
        """
        Initialize a new session or return existing one.
        
        Args:
            appid: Steam App ID
            
        Returns:
            Tuple of (success, session, message)
        """
        try:
            # Validate app ID and get game details
            game_details = self.steam_api.get_game_details(appid)
            if not game_details:
                return False, None, f"Game with ID {appid} not found"
            
            # Check for existing session
            existing_session = self._find_session_by_appid(appid)
            if existing_session:
                if existing_session.status == SessionStatus.READY:
                    return True, existing_session, "Session already exists and is ready"
                elif existing_session.status in [SessionStatus.DOWNLOADING, SessionStatus.PROCESSING]:
                    return True, existing_session, "Session is still being prepared"
                elif existing_session.status == SessionStatus.ERROR:
                    # Remove failed session and create new one
                    self._remove_session(existing_session.session_id)
                else:
                    return True, existing_session, "Session found"
            
            # Create new session
            session_id = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(hours=self.session_timeout_hours)
            
            session = APISession(
                session_id=session_id,
                appid=appid,
                game_name=game_details.name,
                status=SessionStatus.INITIALIZING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                reviews_count=0,
                target_reviews=self.config.target_reviews,
                progress_percentage=0.0,
                expires_at=expires_at
            )
            
            with self.sessions_lock:
                self.sessions[session_id] = session
            
            # Start background download task
            download_task = asyncio.create_task(
                self._download_reviews_background(session)
            )
            session._download_task = download_task
            self._background_tasks[session_id] = download_task
            
            self._save_sessions()
            
            logger.info(f"Created new session {session_id} for game {game_details.name}")
            return True, session, "Session created successfully"
        
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            return False, None, f"Error initializing session: {str(e)}"
    
    def _find_session_by_appid(self, appid: str) -> Optional[APISession]:
        """Find existing session by app ID."""
        with self.sessions_lock:
            for session in self.sessions.values():
                if session.appid == appid and not session.is_expired():
                    return session
        return None
    
    async def _download_reviews_background(self, session: APISession):
        """Download reviews in background task."""
        try:
            session.status = SessionStatus.DOWNLOADING
            session.updated_at = datetime.now()
            
            # Check if session already exists on disk
            session_dir = os.path.join(self.config.save_dir, session.appid)
            if os.path.exists(session_dir):
                # Try to load existing session
                try:
                    reviews_file = os.path.join(session_dir, "reviews.json")
                    if os.path.exists(reviews_file):
                        with open(reviews_file, 'r') as f:
                            reviews_data = json.load(f)
                            session.reviews_count = len(reviews_data)
                        session.progress_percentage = 100.0
                        session.status = SessionStatus.READY
                        session.updated_at = datetime.now()
                        logger.info(f"Loaded existing session for {session.game_name}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load existing session: {e}")
            
            # Download new reviews
            session.status = SessionStatus.DOWNLOADING
            
            # Set up progress callback
            def progress_callback(current: int, total: int):
                session.update_progress(current)
            
            # Download reviews
            reviews = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.steam_client.fetch_reviews_batch(
                    session.appid,
                    session.target_reviews
                )
            )
            
            if not reviews:
                session.set_error("No reviews found for this game")
                return
            
            # Process reviews
            # Process reviews and create embeddings
            session.status = SessionStatus.PROCESSING
            session.updated_at = datetime.now()
            
            # Save reviews and create embeddings
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._process_reviews(reviews, session)
            )
            
            session.status = SessionStatus.READY
            session.progress_percentage = 100.0
            session.updated_at = datetime.now()
            
            logger.info(f"Session {session.session_id} is ready with {len(reviews)} reviews")
        
        except asyncio.CancelledError:
            session.set_error("Session initialization was cancelled")
            logger.info(f"Session {session.session_id} download cancelled")
        except Exception as e:
            session.set_error(f"Error downloading reviews: {str(e)}")
            logger.error(f"Error in background download for session {session.session_id}: {e}")
        finally:
            self._save_sessions()
    
    def _process_reviews(self, reviews: List[Review], session: APISession):
        """Process reviews and create embeddings."""
        try:
            # Save reviews to file
            session_dir = os.path.join(self.config.save_dir, session.appid)
            os.makedirs(session_dir, exist_ok=True)
            
            reviews_file = os.path.join(session_dir, "reviews.json")
            reviews_data = [review.__dict__ if hasattr(review, '__dict__') else review for review in reviews]
            
            with open(reviews_file, 'w') as f:
                json.dump(reviews_data, f, indent=2)
            
            # Create index manager and generate embeddings
            index_manager = FAISSIndexManager(self.config)
            # Process reviews into embeddings and create index
            from steam_ai.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(self.config)
            
            # Convert reviews to text for embedding
            review_texts = [review.text if hasattr(review, 'text') else str(review) for review in reviews]
            embeddings = embedding_service.get_embeddings_batch(review_texts)
            index_manager.create_optimized_index(embeddings)
            
            # Store review texts separately for retrieval during search
            review_texts_file = os.path.join(session_dir, "review_texts.json")
            with open(review_texts_file, 'w') as f:
                json.dump(review_texts, f, indent=2)
            
            # Save index to session directory
            index_manager.save_index(session.appid, session_dir)
            
            session.reviews_count = len(reviews)
            
        except Exception as e:
            raise Exception(f"Error processing reviews: {e}")
    
    def _generate_answer_with_llm(self, llm_client: LLMClient, question: str, search_results: List[Dict]) -> Optional[str]:
        """Generate answer using LLM client."""
        try:
            # Format context from search results
            review_texts = []
            for result in search_results:
                if 'text' in result:
                    review_texts.append(result['text'])
            
            # Create prompt context
            from steam_ai.llm_client import PromptContext
            context = PromptContext(
                reviews=review_texts[:10],
                question=question
            )
            
            # Call the LLM client's generate_response method
            response = llm_client.generate_response(context)
            
            if response and hasattr(response, 'text'):
                return response.text
            elif isinstance(response, str):
                return response
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[APISession]:
        """Get session by ID."""
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if session and session.is_expired():
                self._remove_session(session_id)
                return None
            return session
    
    async def ask_question(self, session_id: str, question: str) -> Tuple[bool, Optional[str], List[Dict], str]:
        """
        Ask a question about the game reviews.
        
        Args:
            session_id: Session identifier
            question: Question to ask
            
        Returns:
            Tuple of (success, answer, sources, message)
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False, None, [], "Session not found or expired"
            
            if session.status != SessionStatus.READY:
                return False, None, [], f"Session is not ready (status: {session.status.value})"
            
            # Load session data if needed
            session_dir = os.path.join(self.config.save_dir, session.appid)
            
            # Initialize components
            index_manager = FAISSIndexManager(self.config)
            if not index_manager.load_index(session.appid, session_dir):
                return False, None, [], "Session index not found"
            
            llm_client = LLMClient(self.config)
            
            # Search for relevant reviews
            from steam_ai.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(self.config)
            
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._search_similar_reviews(index_manager, embedding_service, question, self.config.similarity_top_k, session_dir)
            )
            
            if not search_results:
                return False, None, [], "No relevant reviews found"
            
            # Generate answer
            answer = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._generate_answer_with_llm(llm_client, question, search_results)
            )
            
            if not answer:
                return False, None, [], "Failed to generate answer"
            
            # Prepare sources (always include 3 sources)
            sources = self._prepare_sources(search_results[:3])
            
            # Update session access time
            session.updated_at = datetime.now()
            
            return True, answer, sources, "Question answered successfully"
        
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return False, None, [], f"Error processing question: {str(e)}"
    
    def _prepare_sources(self, search_results: List) -> List[Dict]:
        """Prepare source excerpts from search results."""
        sources = []
        
        for i, result in enumerate(search_results):
            try:
                review = result.get('review', {})
                source = {
                    "review_id": f"review_{i}",
                    "author": review.get('author', 'Anonymous'),
                    "helpful_score": review.get('helpful_score', 0),
                    "playtime_hours": review.get('playtime_hours', 0),
                    "posted_date": review.get('timestamp_created', ''),
                    "excerpt": result.get('text', '')[:500] + ('...' if len(result.get('text', '')) > 500 else ''),
                    "similarity_score": result.get('similarity', 0.0)
                }
                sources.append(source)
            except Exception as e:
                logger.warning(f"Error preparing source {i}: {e}")
                continue
        
        return sources
    
    def get_all_sessions(self) -> List[APISession]:
        """Get all active sessions."""
        with self.sessions_lock:
            # Clean up expired sessions first
            self._cleanup_expired_sessions()
            return list(self.sessions.values())
    
    def get_session_stats(self) -> Dict:
        """Get session statistics."""
        with self.sessions_lock:
            total_sessions = len(self.sessions)
            active_sessions = sum(1 for s in self.sessions.values() if s.status == SessionStatus.READY)
            initializing_sessions = sum(1 for s in self.sessions.values() if s.status in [SessionStatus.INITIALIZING, SessionStatus.DOWNLOADING, SessionStatus.PROCESSING])
            error_sessions = sum(1 for s in self.sessions.values() if s.status == SessionStatus.ERROR)
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "initializing_sessions": initializing_sessions,
                "error_sessions": error_sessions,
                "background_tasks": len(self._background_tasks)
            }
    
    def cleanup(self):
        """Clean up resources."""
        # Cancel all background tasks
        for task in self._background_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        # Save sessions before shutdown
        self._save_sessions()
        
        logger.info("API session manager cleanup completed")
    
    def _search_similar_reviews(self, index_manager: FAISSIndexManager, embedding_service, question: str, top_k: int, session_dir: str) -> List[Dict]:
        """Search for similar reviews using the index."""
        try:
            # Generate embedding for the question
            question_embeddings = embedding_service.get_embeddings_batch([question])
            if not question_embeddings:
                return []
            
            # Search in the index
            similarities, indices = index_manager.search(question_embeddings[0], top_k)
            
            # Load the stored review texts
            review_texts_file = os.path.join(session_dir, "review_texts.json")
            review_texts = []
            if os.path.exists(review_texts_file):
                with open(review_texts_file, 'r') as f:
                    review_texts = json.load(f)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities, indices)):
                # Get actual review text if available
                review_text = review_texts[idx] if idx < len(review_texts) else f"Review text not found for index {idx}"
                results.append({
                    'text': review_text,
                    'similarity': float(similarity),
                    'review': {'author': 'Unknown', 'helpful_score': 0, 'playtime_hours': 0}
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar reviews: {e}")
            return []