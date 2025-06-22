import asyncio
import logging
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator, Tuple
from dataclasses import dataclass
import time
from tqdm import tqdm

from .config import Config
from .http_client import SyncHTTPClient
from .steam_store_api import SteamStoreAPI, GameDetails
from .steam_store_api import SteamStoreAPI, GameDetails

logger = logging.getLogger(__name__)


@dataclass
class Review:
    """Steam review data structure."""
    text: str
    author: str
    helpful: int
    funny: int
    timestamp_created: int
    timestamp_updated: int
    voted_up: bool
    votes_up: int
    votes_funny: int
    weighted_vote_score: float
    comment_count: int
    steam_purchase: bool
    received_for_free: bool
    written_during_early_access: bool
    playtime_forever: int
    playtime_at_review: int
    
    @classmethod
    def from_steam_data(cls, data: Dict[str, Any]) -> 'Review':
        """Create Review from Steam API response data."""
        return cls(
            text=data.get('review', '').strip().replace('\n', ' '),
            author=data.get('author', {}).get('steamid', ''),
            helpful=data.get('votes_up', 0),
            funny=data.get('votes_funny', 0),
            timestamp_created=data.get('timestamp_created', 0),
            timestamp_updated=data.get('timestamp_updated', 0),
            voted_up=data.get('voted_up', False),
            votes_up=data.get('votes_up', 0),
            votes_funny=data.get('votes_funny', 0),
            weighted_vote_score=data.get('weighted_vote_score', 0.0),
            comment_count=data.get('comment_count', 0),
            steam_purchase=data.get('steam_purchase', True),
            received_for_free=data.get('received_for_free', False),
            written_during_early_access=data.get('written_during_early_access', False),
            playtime_forever=data.get('author', {}).get('playtime_forever', 0),
            playtime_at_review=data.get('author', {}).get('playtime_at_review', 0),
        )
    
    def get_processed_text(self, max_length: int) -> str:
        """Get processed review text with length limit."""
        text = self.text
        if len(text) > max_length:
            # Try to cut at sentence boundary
            sentences = text.split('. ')
            result = ''
            for sentence in sentences:
                if len(result + sentence + '. ') <= max_length:
                    result += sentence + '. '
                else:
                    break
            if result:
                return result.rstrip()
            else:
                # Fallback to hard cut
                return text[:max_length].rstrip()
        return text


class SteamReviewFetcher:
    """Steam review fetcher with streaming and memory optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.http_client = SyncHTTPClient(
            pool_size=config.connection_pool_size,
            timeout=30
        )
        self.store_api = SteamStoreAPI(config)
        
    def fetch_reviews_streaming(self, 
                              app_id: str, 
                              target_count: int,
                              chunk_size: int = 100) -> Generator[List[Review], None, None]:
        """
        Fetch reviews in chunks to optimize memory usage.
        
        Args:
            app_id: Steam app ID
            target_count: Total number of reviews to fetch
            chunk_size: Number of reviews per chunk
            
        Yields:
            List[Review]: Chunks of reviews
        """
        fetched_count = 0
        cursor = "*"
        pbar = tqdm(total=target_count, desc="Fetching reviews")
        
        try:
            while fetched_count < target_count:
                try:
                    remaining = target_count - fetched_count
                    fetch_count = min(chunk_size, remaining)
                    
                    reviews_chunk = self._fetch_review_page(app_id, cursor, fetch_count)
                    
                    if not reviews_chunk['reviews']:
                        logger.warning(f"No more reviews available for app {app_id}")
                        break
                    
                    # Process reviews
                    processed_reviews = []
                    for review_data in reviews_chunk['reviews']:
                        if review_data.get('review'):
                            review = Review.from_steam_data(review_data)
                            processed_reviews.append(review)
                            
                    if not processed_reviews:
                        logger.warning("No valid reviews in this batch")
                        break
                        
                    # Update progress
                    actual_count = min(len(processed_reviews), remaining)
                    processed_reviews = processed_reviews[:actual_count]
                    fetched_count += len(processed_reviews)
                    pbar.update(len(processed_reviews))
                    
                    # Yield chunk
                    yield processed_reviews
                    
                    # Update cursor for next page
                    cursor = reviews_chunk.get('cursor', cursor)
                    
                    # Rate limiting
                    if self.config.steam_api_sleep > 0:
                        time.sleep(self.config.steam_api_sleep)
                        
                except Exception as e:
                    logger.error(f"Error fetching review page: {e}")
                    break
                    
        finally:
            pbar.close()
    
    def fetch_reviews_batch(self, app_id: str, target_count: int) -> List[Review]:
        """
        Fetch all reviews at once (legacy method for compatibility).
        
        Args:
            app_id: Steam app ID
            target_count: Number of reviews to fetch
            
        Returns:
            List[Review]: All fetched reviews
        """
        all_reviews = []
        for review_chunk in self.fetch_reviews_streaming(app_id, target_count):
            all_reviews.extend(review_chunk)
        return all_reviews
    
    def _fetch_review_page(self, app_id: str, cursor: str, count: int) -> Dict[str, Any]:
        """
        Fetch a single page of reviews from Steam API.
        
        Args:
            app_id: Steam app ID
            cursor: Pagination cursor
            count: Number of reviews to fetch
            
        Returns:
            Dict[str, Any]: Steam API response
        """
        url = f"https://store.steampowered.com/appreviews/{app_id}"
        params = {
            "json": 1,
            "filter": "recent",
            "language": "english",
            "cursor": cursor,
            "day_range": 365,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": min(count, 100)  # Steam API limit
        }
        
        try:
            response = self.http_client.get(url, params=params)
            
            # Validate response
            if not isinstance(response, dict):
                raise ValueError("Invalid response format")
                
            if 'reviews' not in response:
                logger.warning("No reviews key in response")
                return {'reviews': [], 'cursor': cursor}
                
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch review page: {e}")
            raise
    
    def search_games(self, query: str, max_results: int = 5) -> List[Tuple[Any, Optional[GameDetails]]]:
        """
        Search for games by name and return results with details.
        
        Args:
            query: Game name to search for
            max_results: Maximum number of results
            
        Returns:
            List[Tuple[GameSearchResult, Optional[GameDetails]]]: Search results with details
        """
        return self.store_api.search_and_get_details(query, max_results)
    
    def get_game_details(self, app_id: str) -> Optional[GameDetails]:
        """
        Get comprehensive game details.
        
        Args:
            app_id: Steam app ID
            
        Returns:
            Optional[GameDetails]: Game details or None if failed
        """
        return self.store_api.get_game_details(app_id)
    
    def get_app_info(self, app_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic app information from Steam (legacy method).
        
        Args:
            app_id: Steam app ID
            
        Returns:
            Optional[Dict[str, Any]]: App information or None if failed
        """
        details = self.get_game_details(app_id)
        if details:
            return {
                'name': details.name,
                'type': details.type,
                'steam_appid': details.appid,
                'short_description': details.short_description
            }
        return None
    
    def estimate_total_reviews(self, app_id: str) -> Optional[int]:
        """
        Estimate total number of reviews available for an app.
        
        Args:
            app_id: Steam app ID
            
        Returns:
            Optional[int]: Estimated total reviews or None if failed
        """
        try:
            # Fetch first page to get query summary
            response = self._fetch_review_page(app_id, "*", 1)
            query_summary = response.get('query_summary', {})
            return query_summary.get('total_reviews', None)
            
        except Exception as e:
            logger.error(f"Failed to estimate total reviews: {e}")
            return None


class ReviewProcessor:
    """Process reviews for optimal embedding generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.store_api = SteamStoreAPI(config)
        
    def process_reviews_for_embedding(self, reviews: List[Review], game_details: Optional[GameDetails] = None) -> List[str]:
        """
        Process reviews into text suitable for embedding generation.
        
        Args:
            reviews: List of Review objects
            game_details: Optional game details for context
            
        Returns:
            List[str]: Processed review texts
        """
        processed_texts = []
        
        # Add game information as the first "document" if available
        if game_details:
            game_summary = self._create_game_summary_text(game_details)
            processed_texts.append(f"[GAME_INFO] {game_summary}")
        
        for review in reviews:
            text = review.get_processed_text(self.config.review_max_length)
            
            # Skip very short reviews
            if len(text.strip()) < 10:
                continue
                
            # Add metadata for better context
            metadata_prefix = ""
            if review.voted_up:
                metadata_prefix = "[POSITIVE] "
            else:
                metadata_prefix = "[NEGATIVE] "
                
            # Add playtime context if available
            if review.playtime_at_review > 0:
                hours = review.playtime_at_review / 60
                if hours >= 100:
                    metadata_prefix += "[EXPERIENCED] "
                elif hours <= 2:
                    metadata_prefix += "[NEW_PLAYER] "
                    
            processed_text = metadata_prefix + text
            processed_texts.append(processed_text)
            
        return processed_texts
    
    def filter_reviews_by_quality(self, reviews: List[Review], min_length: int = 50) -> List[Review]:
        """
        Filter reviews by quality metrics.
        
        Args:
            reviews: List of Review objects
            min_length: Minimum review text length
            
        Returns:
            List[Review]: Filtered reviews
        """
        filtered_reviews = []
        
        for review in reviews:
            # Skip very short reviews
            if len(review.text.strip()) < min_length:
                continue
                
            # Skip reviews that are mostly punctuation or numbers
            alpha_ratio = sum(c.isalpha() for c in review.text) / len(review.text)
            if alpha_ratio < 0.6:
                continue
                
            # Skip reviews from accounts with no playtime
            if review.playtime_forever == 0 and review.playtime_at_review == 0:
                continue
                
            filtered_reviews.append(review)
            
        return filtered_reviews
    
    def deduplicate_reviews(self, reviews: List[Review]) -> List[Review]:
        """
        Remove duplicate reviews based on text similarity.
        
        Args:
            reviews: List of Review objects
            
        Returns:
            List[Review]: Deduplicated reviews
        """
        if not reviews:
            return reviews
            
        # Simple deduplication based on text content
        seen_texts = set()
        unique_reviews = []
        
        for review in reviews:
            # Normalize text for comparison
            normalized_text = review.text.lower().strip()
            
            # Skip if we've seen very similar text
            if normalized_text not in seen_texts:
                # Check for substring matches in existing texts
                is_duplicate = False
                for seen_text in seen_texts:
                    if (normalized_text in seen_text and len(normalized_text) > len(seen_text) * 0.8) or \
                       (seen_text in normalized_text and len(seen_text) > len(normalized_text) * 0.8):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    seen_texts.add(normalized_text)
                    unique_reviews.append(review)
                    
        logger.info(f"Deduplicated {len(reviews)} reviews to {len(unique_reviews)} unique reviews")
        return unique_reviews
    
    def _create_game_summary_text(self, game_details: GameDetails) -> str:
        """
        Create a comprehensive game summary for embedding.
        
        Args:
            game_details: Game details from Steam API
            
        Returns:
            str: Formatted game summary
        """
        parts = []
        
        # Basic game information
        parts.append(f"Game: {game_details.name}")
        
        if game_details.developers:
            parts.append(f"Developer: {', '.join(game_details.developers)}")
        
        if game_details.publishers:
            parts.append(f"Publisher: {', '.join(game_details.publishers)}")
        
        # Release information
        if game_details.release_date.get('date'):
            parts.append(f"Release Date: {game_details.release_date['date']}")
        
        # Game description
        if game_details.short_description:
            parts.append(f"Description: {game_details.short_description}")
        
        # Genres and categories
        if game_details.genres:
            genre_names = [g.get('description', '') for g in game_details.genres]
            parts.append(f"Genres: {', '.join(genre_names)}")
        
        if game_details.categories:
            category_names = [c.get('description', '') for c in game_details.categories]
            # Limit to most important categories
            important_categories = category_names[:5]
            parts.append(f"Features: {', '.join(important_categories)}")
        
        # Platform support
        platforms = []
        if game_details.platforms.get('windows'): platforms.append('Windows')
        if game_details.platforms.get('mac'): platforms.append('Mac')
        if game_details.platforms.get('linux'): platforms.append('Linux')
        if platforms:
            parts.append(f"Platforms: {', '.join(platforms)}")
        
        # Price information
        if game_details.price_overview:
            price_info = game_details.price_overview.get('final_formatted', 'N/A')
            parts.append(f"Price: {price_info}")
        elif game_details.is_free:
            parts.append("Price: Free to Play")
        
        # System requirements (brief)
        if game_details.pc_requirements.get('minimum'):
            req_text = game_details._clean_html(game_details.pc_requirements['minimum'])
            # Keep requirements brief
            if req_text and len(req_text) < 300:
                parts.append(f"PC Requirements: {req_text}")
        
        # Metacritic score if available
        if game_details.metacritic and game_details.metacritic.get('score'):
            parts.append(f"Metacritic Score: {game_details.metacritic['score']}/100")
        
        # Content rating
        if game_details.required_age > 0:
            parts.append(f"Age Rating: {game_details.required_age}+")
        
        return " | ".join(parts)