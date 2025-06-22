import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .config import Config
from .http_client import SyncHTTPClient

logger = logging.getLogger(__name__)


@dataclass
class GameSearchResult:
    """Steam game search result."""
    appid: str
    name: str
    type: str
    price: Optional[str] = None
    discount: Optional[str] = None
    header_image: Optional[str] = None
    
    @classmethod
    def from_search_data(cls, data: Dict[str, Any]) -> 'GameSearchResult':
        """Create from Steam search API data."""
        return cls(
            appid=str(data.get('id', '')),
            name=data.get('name', ''),
            type=data.get('type', ''),
            price=data.get('price', {}).get('final_formatted') if data.get('price') else None,
            discount=data.get('price', {}).get('discount_percent') if data.get('price') else None,
            header_image=data.get('header_image', '')
        )


@dataclass
class GameDetails:
    """Comprehensive game details from Steam."""
    appid: str
    name: str
    type: str
    description: str
    short_description: str
    about_the_game: str
    detailed_description: str
    
    # Basic info
    is_free: bool
    price_overview: Optional[Dict[str, Any]]
    developers: List[str]
    publishers: List[str]
    release_date: Dict[str, Any]
    platforms: Dict[str, bool]
    
    # Content info
    categories: List[Dict[str, Any]]
    genres: List[Dict[str, Any]]
    screenshots: List[Dict[str, Any]]
    movies: List[Dict[str, Any]]
    
    # Ratings and requirements
    required_age: int
    pc_requirements: Dict[str, Any]
    mac_requirements: Dict[str, Any]
    linux_requirements: Dict[str, Any]
    
    # Additional metadata
    header_image: str
    background: str
    website: Optional[str]
    support_info: Dict[str, Any]
    content_descriptors: Dict[str, Any]
    achievements: Optional[Dict[str, Any]]
    
    # Review info
    metacritic: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]
    
    @classmethod
    def from_api_data(cls, appid: str, data: Dict[str, Any]) -> 'GameDetails':
        """Create from Steam Store API data."""
        return cls(
            appid=appid,
            name=data.get('name', ''),
            type=data.get('type', ''),
            description=cls._clean_html(data.get('detailed_description', '')),
            short_description=cls._clean_html(data.get('short_description', '')),
            about_the_game=cls._clean_html(data.get('about_the_game', '')),
            detailed_description=cls._clean_html(data.get('detailed_description', '')),
            
            is_free=data.get('is_free', False),
            price_overview=data.get('price_overview'),
            developers=data.get('developers', []),
            publishers=data.get('publishers', []),
            release_date=data.get('release_date', {}),
            platforms=data.get('platforms', {}),
            
            categories=data.get('categories', []),
            genres=data.get('genres', []),
            screenshots=data.get('screenshots', []),
            movies=data.get('movies', []),
            
            required_age=data.get('required_age', 0),
            pc_requirements=data.get('pc_requirements', {}),
            mac_requirements=data.get('mac_requirements', {}),
            linux_requirements=data.get('linux_requirements', {}),
            
            header_image=data.get('header_image', ''),
            background=data.get('background', ''),
            website=data.get('website'),
            support_info=data.get('support_info', {}),
            content_descriptors=data.get('content_descriptors', {}),
            achievements=data.get('achievements'),
            
            metacritic=data.get('metacritic'),
            recommendations=data.get('recommendations')
        )
    
    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Simple HTML tag removal
        clean = re.sub(r'<[^>]+>', '', text)
        # Clean up extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean
    
    def get_summary_text(self) -> str:
        """Get a comprehensive summary text for embedding."""
        parts = []
        
        # Basic info
        parts.append(f"Game: {self.name}")
        if self.developers:
            parts.append(f"Developer: {', '.join(self.developers)}")
        if self.publishers:
            parts.append(f"Publisher: {', '.join(self.publishers)}")
        
        # Description
        if self.short_description:
            parts.append(f"Description: {self.short_description}")
        
        # Genres and categories
        if self.genres:
            genre_names = [g.get('description', '') for g in self.genres]
            parts.append(f"Genres: {', '.join(genre_names)}")
        
        if self.categories:
            category_names = [c.get('description', '') for c in self.categories]
            parts.append(f"Features: {', '.join(category_names)}")
        
        # Release info
        if self.release_date.get('date'):
            parts.append(f"Release Date: {self.release_date['date']}")
        
        # Platform info
        platform_list = []
        if self.platforms.get('windows'): platform_list.append('Windows')
        if self.platforms.get('mac'): platform_list.append('Mac')
        if self.platforms.get('linux'): platform_list.append('Linux')
        if platform_list:
            parts.append(f"Platforms: {', '.join(platform_list)}")
        
        # Price info
        if self.price_overview:
            parts.append(f"Price: {self.price_overview.get('final_formatted', 'N/A')}")
        elif self.is_free:
            parts.append("Price: Free to Play")
        
        # Requirements
        if self.pc_requirements.get('minimum'):
            req_text = self._clean_html(self.pc_requirements['minimum'])
            if req_text and len(req_text) < 500:
                parts.append(f"PC Requirements: {req_text}")
        
        # Metacritic
        if self.metacritic and self.metacritic.get('score'):
            parts.append(f"Metacritic Score: {self.metacritic['score']}/100")
        
        return " | ".join(parts)


class SteamStoreAPI:
    """Steam Store API client for game search and details."""
    
    def __init__(self, config: Config):
        self.config = config
        self.http_client = SyncHTTPClient(
            pool_size=config.connection_pool_size,
            timeout=30
        )
        
        # Steam Store API endpoints
        self.search_url = "https://store.steampowered.com/api/storesearch"
        self.app_details_url = "https://store.steampowered.com/api/appdetails"
        self.app_list_url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    
    def search_games(self, query: str, max_results: int = 5) -> List[GameSearchResult]:
        """
        Search for games by name.
        
        Args:
            query: Game name to search for
            max_results: Maximum number of results to return
            
        Returns:
            List[GameSearchResult]: Search results
        """
        try:
            params = {
                'term': query,
                'l': 'english',
                'cc': 'US',
                'category1': '998',  # Games only
                'supportedlang': 'english',
                'ndl': '1'
            }
            
            response = self.http_client.get(self.search_url, params=params)
            
            if not response or 'items' not in response:
                logger.warning(f"No search results for query: {query}")
                return []
            
            results = []
            for item in response['items'][:max_results * 2]:  # Get more results to filter
                try:
                    # Filter to games only - Steam API returns 'app' for games
                    if item.get('type') in ['app', 'dlc']:
                        # Additional filtering to exclude soundtracks and non-game content
                        name = item.get('name', '').lower()
                        if any(keyword in name for keyword in [
                            'soundtrack', 'ost', 'original soundtrack', 'music',
                            'artbook', 'art book', 'wallpaper', 'avatar',
                            'theme pack', 'cosmetic', 'skin pack'
                        ]):
                            continue
                        
                        result = GameSearchResult.from_search_data(item)
                        results.append(result)
                        
                        # Stop when we have enough actual games
                        if len(results) >= max_results:
                            break
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} games for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for games: {e}")
            return []
    
    def get_game_details(self, appid: str) -> Optional[GameDetails]:
        """
        Get detailed information about a game.
        
        Args:
            appid: Steam App ID
            
        Returns:
            Optional[GameDetails]: Game details or None if not found
        """
        try:
            params = {
                'appids': appid,
                'l': 'english',
                'cc': 'US'
            }
            
            response = self.http_client.get(self.app_details_url, params=params)
            
            if not response or appid not in response:
                logger.warning(f"No details found for app ID: {appid}")
                return None
            
            app_data = response[appid]
            
            if not app_data.get('success', False):
                logger.warning(f"Steam API returned success=false for app ID: {appid}")
                return None
            
            game_data = app_data.get('data', {})
            if not game_data:
                logger.warning(f"No game data for app ID: {appid}")
                return None
            
            details = GameDetails.from_api_data(appid, game_data)
            logger.info(f"Retrieved details for game: {details.name}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting game details for {appid}: {e}")
            return None
    
    def search_and_get_details(self, query: str, max_results: int = 5) -> List[Tuple[GameSearchResult, Optional[GameDetails]]]:
        """
        Search for games and get detailed information for each result.
        
        Args:
            query: Game name to search for
            max_results: Maximum number of results to return
            
        Returns:
            List[Tuple[GameSearchResult, Optional[GameDetails]]]: Results with details
        """
        search_results = self.search_games(query, max_results)
        
        results_with_details = []
        for search_result in search_results:
            details = self.get_game_details(search_result.appid)
            results_with_details.append((search_result, details))
        
        return results_with_details
    
    def format_search_results_for_user(self, results: List[Tuple[GameSearchResult, Optional[GameDetails]]]) -> str:
        """
        Format search results for display to user.
        
        Args:
            results: Search results with details
            
        Returns:
            str: Formatted results text
        """
        if not results:
            return "No games found."
        
        lines = []
        lines.append(f"Found {len(results)} games:")
        lines.append("")
        
        for i, (search_result, details) in enumerate(results, 1):
            line_parts = [f"{i}. {search_result.name} (ID: {search_result.appid})"]
            
            if details:
                # Add developer info
                if details.developers:
                    line_parts.append(f"by {', '.join(details.developers)}")
                
                # Add release date
                if details.release_date.get('date'):
                    line_parts.append(f"({details.release_date['date']})")
                
                # Add price info
                if details.price_overview:
                    price = details.price_overview.get('final_formatted', 'N/A')
                    line_parts.append(f"- {price}")
                elif details.is_free:
                    line_parts.append("- Free")
                
                # Add genres
                if details.genres:
                    genre_names = [g.get('description', '') for g in details.genres[:3]]
                    line_parts.append(f"[{', '.join(genre_names)}]")
            
            lines.append(" ".join(line_parts))
            
            # Add short description if available
            if details and details.short_description:
                desc = details.short_description[:100]
                if len(details.short_description) > 100:
                    desc += "..."
                lines.append(f"   {desc}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def validate_app_id(self, appid: str) -> bool:
        """
        Validate that an app ID exists and is a game.
        
        Args:
            appid: Steam App ID to validate
            
        Returns:
            bool: True if valid game, False otherwise
        """
        try:
            details = self.get_game_details(appid)
            return details is not None and details.type in ['game', 'dlc']
        except Exception:
            return False
    
    def get_app_name(self, appid: str) -> Optional[str]:
        """
        Get the name of an app by ID.
        
        Args:
            appid: Steam App ID
            
        Returns:
            Optional[str]: App name or None if not found
        """
        try:
            details = self.get_game_details(appid)
            return details.name if details else None
        except Exception:
            return None