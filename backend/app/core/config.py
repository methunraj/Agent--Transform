# app/core/config.py
import os
import logging
from pydantic_settings import BaseSettings
from pydantic import validator

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "3.0.0-refactored"
    APP_TITLE: str = "IntelliExtract Agno AI JSON to XLSX Processing API"
    TEMP_DIR_PREFIX: str = "intelliextract_agno_xlsx_"
    
    # Model Configuration
    DEFAULT_AI_MODEL: str = os.getenv("DEFAULT_AI_MODEL", "gemini-2.0-flash-001")
    
    # Monitoring Configuration
    AGNO_API_KEY: str = os.getenv("AGNO_API_KEY", "")
    AGNO_MONITOR: bool = os.getenv("AGNO_MONITOR", "false").lower() == "true"
    AGNO_DEBUG: bool = os.getenv("AGNO_DEBUG", "false").lower() == "true"
    DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    
    # Performance & Security Settings
    MAX_FILE_SIZE_MB: int = 100  # Maximum file size in MB
    MAX_JSON_SIZE_MB: int = 50   # Maximum JSON payload size in MB
    REQUEST_TIMEOUT_SECONDS: int = 2700  # 45 minutes (increased from 20 min)
    CLEANUP_DELAY_SECONDS: int = 300  # 5 minutes delay before cleanup
    MAX_POOL_SIZE: int = 10  # Maximum number of agents in pool

    # Per-job type timeout configuration (example)
    # This could be a simple dict or parsed from a JSON string in env
    JOB_TYPE_TIMEOUTS: dict = {
        "default": 2700,  # Default job timeout: 45 minutes
        "document_conversion": 3000, # Specific timeout for this job type
        "data_analysis_report": 3600, # 1 hour
        "quick_task": 300 # 5 minutes for short tasks
    }
    AGENT_STORAGE_CLEANUP_HOURS: int = 24  # Clean up agent storage after 24 hours
    ABANDONED_JOB_TIMEOUT_SECONDS: int = 60 # Timeout for auto-cancelling jobs with no WebSocket connections

    # Job Worker Configuration
    JOB_WORKERS: int = os.getenv("JOB_WORKERS", 2)

    # SSE/WebSocket Configuration
    SSE_TIMEOUT_SECONDS: int = 30 # How long SSE connection waits for message before sending keepalive
    WEBSOCKET_KEEP_ALIVE_TIMEOUT_SECONDS: int = 60 # Timeout for WebSocket receive to check connection
    
    @validator('GOOGLE_API_KEY')
    def validate_google_api_key(cls, v):
        if not v:
            logger.warning("GOOGLE_API_KEY is not set. AI processing will fail.")
        elif len(v) < 20:  # Basic length check
            logger.warning("GOOGLE_API_KEY appears to be invalid (too short).")
        return v
    
    @validator('AGNO_API_KEY')
    def validate_agno_api_key(cls, v, values):
        if values.get('AGNO_MONITOR') and not v:
            logger.warning("AGNO_MONITOR is enabled but AGNO_API_KEY is not set. Monitoring may not work properly.")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Validate critical settings on startup
if not settings.GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY is not configured. AI features will not work!")
else:
    logger.info("Configuration loaded successfully")
