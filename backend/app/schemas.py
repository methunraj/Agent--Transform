# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class ProcessRequest(BaseModel):
    json_data: str = Field(..., description="JSON data to be converted")
    file_name: Optional[str] = Field("data", description="Base name for the output file")
    description: Optional[str] = Field("", description="Description of the data")
    model: Optional[str] = Field("gemini-2.5-flash-preview-05-20", description="AI model to use")
    processing_mode: Optional[Literal["auto", "ai_only", "direct_only"]] = Field(
        "auto",
        description="Processing mode: auto (smart selection), ai_only, or direct_only"
    )
    chunk_size: Optional[int] = Field(1000, description="Chunk size for large data processing")

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