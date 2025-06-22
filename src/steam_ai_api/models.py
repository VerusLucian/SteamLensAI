from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class SessionStatus(str, Enum):
    """Session status enumeration."""
    INITIALIZING = "initializing"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class GameSearchRequest(BaseModel):
    """Request model for game search."""
    query: str = Field(..., min_length=1, max_length=100, description="Game name to search for")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")


class GameInfo(BaseModel):
    """Basic game information."""
    appid: str = Field(..., description="Steam App ID")
    name: str = Field(..., description="Game name")
    type: str = Field(..., description="Content type (game, dlc, etc.)")
    price: Optional[str] = Field(None, description="Formatted price")
    discount: Optional[str] = Field(None, description="Discount percentage")
    header_image: Optional[str] = Field(None, description="Header image URL")
    developers: List[str] = Field(default_factory=list, description="Game developers")
    publishers: List[str] = Field(default_factory=list, description="Game publishers")
    release_date: Optional[str] = Field(None, description="Release date")
    short_description: Optional[str] = Field(None, description="Short description")
    genres: List[str] = Field(default_factory=list, description="Game genres")


class GameSearchResponse(BaseModel):
    """Response model for game search."""
    success: bool = Field(..., description="Whether the search was successful")
    games: List[GameInfo] = Field(default_factory=list, description="List of found games")
    total_found: int = Field(..., description="Total number of games found")
    message: Optional[str] = Field(None, description="Additional message or error description")


class SessionInitRequest(BaseModel):
    """Request model for session initialization."""
    appid: str = Field(..., description="Steam App ID")


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str = Field(..., description="Unique session identifier")
    appid: str = Field(..., description="Steam App ID")
    game_name: str = Field(..., description="Game name")
    status: SessionStatus = Field(..., description="Current session status")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    reviews_count: int = Field(default=0, description="Number of reviews processed")
    target_reviews: int = Field(default=1000, description="Target number of reviews")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Processing progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    expires_at: Optional[datetime] = Field(None, description="Session expiration timestamp")


class SessionInitResponse(BaseModel):
    """Response model for session initialization."""
    success: bool = Field(..., description="Whether initialization was successful")
    session_info: Optional[SessionInfo] = Field(None, description="Session information")
    message: Optional[str] = Field(None, description="Additional message or error description")


class SessionStatusResponse(BaseModel):
    """Response model for session status check."""
    success: bool = Field(..., description="Whether the request was successful")
    session_info: Optional[SessionInfo] = Field(None, description="Session information")
    message: Optional[str] = Field(None, description="Additional message or error description")


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, max_length=500, description="Question about the game")
    language: str = Field(default="en", description="Language for the response (en/pl)")

    @validator('question')
    def validate_question(cls, v):
        """Validate question content."""
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()

    @validator('language')
    def validate_language(cls, v):
        """Validate language parameter."""
        if v.lower() not in ['en', 'pl']:
            raise ValueError('Language must be either "en" or "pl"')
        return v.lower()


class ReviewSource(BaseModel):
    """Source review excerpt."""
    review_id: str = Field(..., description="Review identifier")
    author: str = Field(..., description="Review author")
    helpful_score: int = Field(..., description="Review helpfulness score")
    playtime_hours: float = Field(..., description="Author's playtime in hours")
    posted_date: str = Field(..., description="Review post date")
    excerpt: str = Field(..., description="Relevant excerpt from the review")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score to the question")


class QuestionResponse(BaseModel):
    """Response model for questions."""
    success: bool = Field(..., description="Whether the question was processed successfully")
    answer: Optional[str] = Field(None, description="AI-generated answer")
    processing_time_seconds: float = Field(..., description="Time taken to process the question")
    session_id: str = Field(..., description="Session identifier")
    question: str = Field(..., description="Original question")
    message: Optional[str] = Field(None, description="Additional message or error description")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Status of dependent services")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    success: bool = Field(default=False, description="Always false for error responses")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    requests_remaining: int = Field(..., description="Requests remaining in current window")
    requests_limit: int = Field(..., description="Total requests allowed in window")
    window_reset_timestamp: datetime = Field(..., description="When the rate limit window resets")
    retry_after_seconds: Optional[int] = Field(None, description="Seconds to wait before retry if rate limited")


class APIStatsResponse(BaseModel):
    """Response model for API statistics."""
    success: bool = Field(..., description="Whether the request was successful")
    total_sessions: int = Field(..., description="Total number of sessions created")
    active_sessions: int = Field(..., description="Number of currently active sessions")
    total_questions: int = Field(..., description="Total number of questions processed")
    average_response_time_seconds: float = Field(..., description="Average response time for questions")
    top_games: List[Dict[str, Any]] = Field(default_factory=list, description="Most popular games by session count")
    uptime_seconds: float = Field(..., description="API uptime in seconds")