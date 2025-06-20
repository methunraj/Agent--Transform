# app/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from .core.config import settings

class ProcessRequest(BaseModel):
    json_data: str = Field(..., description="JSON data to be converted")
    file_name: Optional[str] = Field("data", description="Base name for the output file")
    description: Optional[str] = Field("", description="Description of the data")
    model: Optional[str] = Field(None, description="AI model to use")
    processing_mode: Optional[Literal["auto", "ai_only", "direct_only"]] = Field(
        "auto",
        description="Processing mode: auto (smart selection), ai_only, or direct_only"
    )
    chunk_size: Optional[int] = Field(1000, description="Chunk size for large data processing")
    user_id: Optional[str] = Field(None, description="User ID for session management and conversation continuity")
    session_id: Optional[str] = Field(None, description="Session ID to continue existing conversation")
    
    @validator('json_data')
    def validate_json_size(cls, v):
        # Check JSON data size (in bytes)
        max_size_bytes = settings.MAX_JSON_SIZE_MB * 1024 * 1024
        if len(v.encode('utf-8')) > max_size_bytes:
            raise ValueError(f"JSON data exceeds maximum size of {settings.MAX_JSON_SIZE_MB}MB")
        return v
    
    @validator('file_name')
    def sanitize_filename(cls, v):
        if v:
            # Remove potentially dangerous characters
            import re
            # Allow only alphanumeric, spaces, hyphens, underscores, and dots
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', v)
            # Remove multiple dots to prevent path traversal
            sanitized = re.sub(r'\.{2,}', '.', sanitized)
            # Remove leading/trailing dots and spaces
            sanitized = sanitized.strip('. ')
            # Limit length
            sanitized = sanitized[:100]
            return sanitized if sanitized else "data"
        return v
    
    @validator('chunk_size')
    def validate_chunk_size(cls, v):
        if v and v < 10:
            raise ValueError("Chunk size must be at least 10")
        if v and v > 10000:
            raise ValueError("Chunk size cannot exceed 10000")
        return v

class ProcessResponse(BaseModel):
    success: bool
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    download_url: Optional[str] = None
    ai_analysis: Optional[str] = None
    processing_method: Optional[str] = None
    processing_time: Optional[float] = None
    data_size: Optional[int] = None
    error: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class SystemMetrics(BaseModel):
    total_requests: int
    successful_conversions: int
    ai_conversions: int
    direct_conversions: int
    failed_conversions: int
    success_rate: float
    average_processing_time: float
    active_files: int
    temp_directory: str
    active_jobs: Optional[int] = None
    pending_jobs_in_queue: Optional[int] = None
    jobs_created_total: Optional[int] = None
    job_worker_count: Optional[int] = None


# --- New Schemas for Asynchronous Job System ---
from .job_manager import JobStatus # Import JobStatus enum
from datetime import datetime

class JobCreationResponse(BaseModel):
    job_id: str
    status_url: str
    stream_url: str
    websocket_url: str

class JobStatusResponse(BaseModel):
    id: str
    name: Optional[str] = None
    status: JobStatus
    progress: float = 0.0
    current_step: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    result: Optional[dict] = None # To store final job output, including potential download URLs or errors


# --- Schema for Model Information ---
class ModelInfo(BaseModel):
    id: str # e.g., "gemini-1.5-flash-latest" (used in API calls)
    name: str # e.g., "Gemini 1.5 Flash" (display name)
    description: Optional[str] = None
    version: Optional[str] = None
    provider: str = "Google" # Could be extended if other providers are added
    # supported_generation_methods: Optional[List[str]] = None # From Google API
    input_token_limit: Optional[int] = None # Example: 128000 for Gemini 1.0 Pro
    output_token_limit: Optional[int] = None # Example: 2048 for Gemini 1.0 Pro
    pricing_details_url: Optional[str] = Field(None, description="Link to official pricing documentation")
    notes: Optional[str] = None # For any additional info, e.g., "Preview model"

    class Config:
        orm_mode = True # or from_attributes = True for Pydantic v2
