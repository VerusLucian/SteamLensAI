import logging
import time
from datetime import datetime
from typing import Dict, Optional
from contextlib import asynccontextmanager

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.openapi.utils import get_openapi
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    FASTAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    print(f"FastAPI dependencies not available: {e}")
    print("Please install with: pip install -r requirements.txt")
    
    # Create dummy classes to prevent import errors
    class FastAPI: pass
    class HTTPException: pass
    class Request: pass
    class JSONResponse: pass
    class Limiter: pass
    def Depends(): pass
    def asynccontextmanager(): pass

from steam_ai.config import Config
from steam_ai.steam_store_api import SteamStoreAPI

from .config import APIConfig
from .models import (
    GameSearchRequest, GameSearchResponse, GameInfo,
    SessionInitRequest, SessionInitResponse, SessionStatusResponse,
    QuestionRequest, QuestionResponse, ReviewSource,
    HealthResponse, ErrorResponse, APIStatsResponse
)
from .session_manager import APISessionManager
from .rate_limiter import RateLimitManager, get_client_ip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
app_start_time = time.time()
api_config: Optional[APIConfig] = None
core_config: Optional[Config] = None
session_manager: Optional[APISessionManager] = None
steam_api: Optional[SteamStoreAPI] = None
rate_limiter: Optional[RateLimitManager] = None


def get_rate_limiter():
    """Dependency to get rate limiter instance."""
    return rate_limiter


def get_session_manager():
    """Dependency to get session manager instance."""
    return session_manager


def get_steam_api():
    """Dependency to get Steam API instance."""
    return steam_api


# Rate limiter setup
if FASTAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global api_config, core_config, session_manager, steam_api, rate_limiter
    
    try:
        # Load configurations
        api_config = APIConfig.from_env()
        api_config.validate()
        
        core_config = Config.from_env()
        core_config.validate()
        
        # Initialize components
        steam_api = SteamStoreAPI(core_config)
        session_manager = APISessionManager(core_config, api_config.session_timeout_hours)
        rate_limiter = RateLimitManager()
        
        # Update rate limits from config
        rate_limiter.update_limits("general", api_config.rate_limit_requests_per_day, 86400)
        rate_limiter.update_limits("question", api_config.rate_limit_questions_per_hour, 3600)
        
        logger.info("SteamLens AI API started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    finally:
        # Cleanup
        if session_manager:
            session_manager.cleanup()
        logger.info("SteamLens AI API shutdown completed")


# Create FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SteamLens AI API",
        description="RESTful API for Steam game review analysis using AI",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on api_config.allowed_origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None


def create_error_response(error_code: str, error_message: str, details: Optional[Dict] = None) -> JSONResponse:
    """Create standardized error response."""
    error_response = ErrorResponse(
        error_code=error_code,
        error_message=error_message,
        details=details,
        timestamp=datetime.now()
    )
    
    status_codes = {
        "RATE_LIMITED": 429,
        "NOT_FOUND": 404,
        "BAD_REQUEST": 400,
        "INTERNAL_ERROR": 500,
        "SERVICE_UNAVAILABLE": 503
    }
    
    status_code = status_codes.get(error_code, 500)
    return JSONResponse(
        status_code=status_code,
        content=error_response.dict()
    )


def get_client_ip_from_request(request: Request) -> str:
    """Extract client IP from request."""
    return get_client_ip(
        request, 
        trust_forwarded_for=api_config.trust_forwarded_for if api_config else True,
        forwarded_header=api_config.forwarded_for_header if api_config else "X-Forwarded-For"
    )


def check_rate_limit(request: Request, endpoint: str, custom_limit: Optional[int] = None):
    """Check rate limit for endpoint."""
    if not rate_limiter:
        return {}
    client_ip = get_client_ip_from_request(request)
    is_allowed, rate_info = rate_limiter.check_rate_limit(client_ip, endpoint, custom_limit)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "rate_limit_info": rate_info
            }
        )
    
    return rate_info


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        services = {
            "steam_api": "healthy" if steam_api else "unavailable",
            "session_manager": "healthy" if session_manager else "unavailable",
            "rate_limiter": "healthy" if rate_limiter else "unavailable"
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            services=services,
            uptime_seconds=time.time() - app_start_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )


@app.post("/api/v1/games/search", response_model=GameSearchResponse, tags=["Games"])
async def search_games(
    request: Request,
    search_request: GameSearchRequest,
    steam_api: SteamStoreAPI = Depends(get_steam_api)
):
    """Search for games by name."""
    try:
        # Rate limiting
        check_rate_limit(request, "search")
        
        # Search for games
        search_results = steam_api.search_and_get_details(
            search_request.query, 
            search_request.max_results
        )
        
        games = []
        for search_result, details in search_results:
            game_info = GameInfo(
                appid=search_result.appid,
                name=search_result.name,
                type=search_result.type,
                price=search_result.price,
                discount=search_result.discount,
                header_image=search_result.header_image,
                developers=details.developers if details else [],
                publishers=details.publishers if details else [],
                release_date=details.release_date.get('date') if details and details.release_date else None,
                short_description=details.short_description if details else None,
                genres=[g.get('description', '') for g in details.genres] if details and details.genres else []
            )
            games.append(game_info)
        
        return GameSearchResponse(
            success=True,
            games=games,
            total_found=len(games),
            message=f"Found {len(games)} games matching '{search_request.query}'"
        )
        
    except Exception as e:
        logger.error(f"Error searching games: {e}")
        return create_error_response(
            "INTERNAL_ERROR",
            "Failed to search games",
            {"query": search_request.query, "error": str(e)}
        )


@app.post("/api/v1/sessions/init", response_model=SessionInitResponse, tags=["Sessions"])
async def init_session(
    request: Request,
    init_request: SessionInitRequest,
    session_mgr: APISessionManager = Depends(get_session_manager)
):
    """Initialize a new analysis session for a game."""
    try:
        # Rate limiting
        check_rate_limit(request, "session")
        
        # Initialize session with default settings
        success, session, message = await session_mgr.init_session(
            init_request.appid
        )
        
        if success and session:
            return SessionInitResponse(
                success=True,
                session_info=session.to_session_info(),
                message=message
            )
        else:
            return create_error_response(
                "BAD_REQUEST",
                message or "Failed to initialize session",
                {"appid": init_request.appid}
            )
            
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        return create_error_response(
            "INTERNAL_ERROR",
            "Failed to initialize session",
            {"appid": init_request.appid, "error": str(e)}
        )


@app.get("/api/v1/sessions/{session_id}/status", response_model=SessionStatusResponse, tags=["Sessions"])
async def get_session_status(
    request: Request,
    session_id: str,
    session_mgr: APISessionManager = Depends(get_session_manager)
):
    """Get the status of a session."""
    try:
        # Rate limiting
        check_rate_limit(request, "general")
        
        session = session_mgr.get_session(session_id)
        if not session:
            return create_error_response(
                "NOT_FOUND",
                "Session not found or expired",
                {"session_id": session_id}
            )
        
        return SessionStatusResponse(
            success=True,
            session_info=session.to_session_info(),
            message="Session status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return create_error_response(
            "INTERNAL_ERROR",
            "Failed to get session status",
            {"session_id": session_id, "error": str(e)}
        )


@app.post("/api/v1/sessions/{session_id}/question", response_model=QuestionResponse, tags=["Analysis"])
@limiter.limit("10/hour")
async def ask_question(
    request: Request,
    session_id: str,
    question_request: QuestionRequest,
    session_mgr: APISessionManager = Depends(get_session_manager)
):
    """Ask a question about the game reviews."""
    try:
        # Additional rate limiting for questions
        if api_config:
            check_rate_limit(request, "question", api_config.rate_limit_questions_per_hour)
        
        start_time = time.time()
        
        # Ask question with default settings
        success, answer, sources, message = await session_mgr.ask_question(
            session_id,
            question_request.question
        )
        
        processing_time = time.time() - start_time
        
        if success:
            # Convert sources to response model
            review_sources = []
            for source in sources:
                review_source = ReviewSource(
                    review_id=source.get("review_id", ""),
                    author=source.get("author", ""),
                    helpful_score=source.get("helpful_score", 0),
                    playtime_hours=source.get("playtime_hours", 0),
                    posted_date=source.get("posted_date", ""),
                    excerpt=source.get("excerpt", ""),
                    similarity_score=source.get("similarity_score", 0.0)
                )
                review_sources.append(review_source)
            
            return QuestionResponse(
                success=True,
                answer=answer,
                sources=review_sources,
                processing_time_seconds=processing_time,
                session_id=session_id,
                question=question_request.question,
                message=message
            )
        else:
            return create_error_response(
                "BAD_REQUEST" if "not found" in message else "INTERNAL_ERROR",
                message,
                {"session_id": session_id, "question": question_request.question}
            )
            
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        return create_error_response(
            "INTERNAL_ERROR",
            "Failed to process question",
            {"session_id": session_id, "error": str(e)}
        )


@app.get("/api/v1/stats", response_model=APIStatsResponse, tags=["System"])
async def get_api_stats(
    request: Request,
    session_mgr: APISessionManager = Depends(get_session_manager)
):
    """Get API usage statistics."""
    try:
        # Rate limiting
        check_rate_limit(request, "general")
        
        session_stats = session_mgr.get_session_stats()
        
        # Get top games (placeholder - would need to track this in production)
        top_games = []
        
        return APIStatsResponse(
            success=True,
            total_sessions=session_stats["total_sessions"],
            active_sessions=session_stats["active_sessions"],
            total_questions=0,  # Would need to track this
            average_response_time_seconds=0.0,  # Would need to track this
            top_games=top_games,
            uptime_seconds=time.time() - app_start_time
        )
        
    except Exception as e:
        logger.error(f"Error getting API stats: {e}")
        return create_error_response(
            "INTERNAL_ERROR",
            "Failed to get API statistics",
            {"error": str(e)}
        )


def custom_openapi():
    """Custom OpenAPI schema with additional information."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="SteamLens AI API",
        version="1.0.0",
        description="""
        # SteamLens AI API
        
        A RESTful API for analyzing Steam game reviews using AI.
        
        ## Features
        - Search for Steam games
        - Initialize analysis sessions for games
        - Download and process game reviews
        - Ask questions about games based on review analysis
        - Rate limiting and security features
        
        ## Rate Limits
        - General endpoints: 100 requests per day per IP
        - Question endpoint: 10 requests per hour per IP
        - Search endpoint: 50 requests per hour per IP
        - Session operations: 20 requests per hour per IP
        
        ## Usage Flow
        1. **Search for a game** using `/api/v1/games/search`
        2. **Initialize a session** using `/api/v1/sessions/init` with the game's App ID
        3. **Check session status** using `/api/v1/sessions/{session_id}/status` until ready
        4. **Ask questions** using `/api/v1/sessions/{session_id}/question`
        
        ## Firebase Tunnel Support
        This API is designed to work behind Firebase tunnels and properly handles `X-Forwarded-For` headers for rate limiting.
        """,
        routes=app.routes,
    )
    
    # Ensure ErrorResponse is in components schemas
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    # Add ErrorResponse schema manually since it's not used as a response_model
    from .models import ErrorResponse
    openapi_schema["components"]["schemas"]["ErrorResponse"] = ErrorResponse.schema()
    
    # Add rate limiting info to responses
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if "responses" in openapi_schema["paths"][path][method]:
                openapi_schema["paths"][path][method]["responses"]["429"] = {
                    "description": "Rate limit exceeded",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


if app:
    app.openapi = custom_openapi


def main():
    """Main entry point when run directly."""
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI dependencies are not installed.")
        print("Please install them with: pip install -r requirements.txt")
        return 1
    
    try:
        # Load configuration
        config = APIConfig.from_env()
        
        # Configure logging
        log_level = getattr(logging, config.log_level.upper())
        logging.basicConfig(level=log_level)
        
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        # Run the API
        uvicorn.run(
            "steam_ai_api.app:app",
            host=config.host,
            port=config.port,
            workers=config.workers,
            reload=config.debug,
            log_level=config.log_level.lower()
        )
    except Exception as e:
        print(f"Failed to start API: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())