# app/core/config.py
import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Enhanced with performance optimization settings.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "3.1.0-performance-optimized"
    APP_TITLE: str = "IntelliExtract Agno AI JSON to XLSX Processing API - Performance Optimized"
    TEMP_DIR_PREFIX: str = "intelliextract_agno_xlsx_"
    
    # Model Configuration - Enhanced for Performance
    DEFAULT_AI_MODEL: str = os.getenv("DEFAULT_AI_MODEL", "gemini-2.0-flash-001")
    
    # PERFORMANCE OPTIMIZATION: Fast model selection
    FAST_MODEL_SIMPLE_TASKS: str = os.getenv("FAST_MODEL_SIMPLE_TASKS", "gemini-2.0-flash-lite")
    FAST_MODEL_MEDIUM_TASKS: str = os.getenv("FAST_MODEL_MEDIUM_TASKS", "gemini-2.0-flash-001")
    FAST_MODEL_COMPLEX_TASKS: str = os.getenv("FAST_MODEL_COMPLEX_TASKS", "gemini-2.5-pro-preview-05-06")
    
    # Auto-select optimal models based on task complexity
    ENABLE_AUTO_MODEL_SELECTION: bool = os.getenv("ENABLE_AUTO_MODEL_SELECTION", "true").lower() == "true"
    
    # Monitoring Configuration
    AGNO_API_KEY: str = os.getenv("AGNO_API_KEY", "")
    AGNO_MONITOR: bool = os.getenv("AGNO_MONITOR", "false").lower() == "true"
    AGNO_DEBUG: bool = os.getenv("AGNO_DEBUG", "false").lower() == "true"
    DEVELOPMENT_MODE: bool = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
    
    # PERFORMANCE OPTIMIZATION: Enhanced agent pooling
    MAX_POOL_SIZE: int = int(os.getenv("MAX_POOL_SIZE", "20"))  # Increased from 10 for better hit rates
    AGENT_POOL_CLEANUP_INTERVAL_MINUTES: int = int(os.getenv("AGENT_POOL_CLEANUP_INTERVAL_MINUTES", "30"))
    
    # PERFORMANCE OPTIMIZATION: Cache performance targets
    TARGET_CACHE_HIT_RATE_PERCENT: float = float(os.getenv("TARGET_CACHE_HIT_RATE_PERCENT", "80.0"))
    TARGET_AGENT_REUSE_TIME_MS: float = float(os.getenv("TARGET_AGENT_REUSE_TIME_MS", "5.0"))
    
    # Performance & Security Settings
    MAX_FILE_SIZE_MB: int = 100  # Maximum file size in MB
    MAX_JSON_SIZE_MB: int = 50   # Maximum JSON payload size in MB
    REQUEST_TIMEOUT_SECONDS: int = 2700  # 45 minutes (increased from 20 min)
    CLEANUP_DELAY_SECONDS: int = 300  # 5 minutes delay before cleanup
    STORAGE_CLEANUP_HOURS: int = int(os.getenv("STORAGE_CLEANUP_HOURS", "1"))  # Agent storage cleanup interval

    # PERFORMANCE OPTIMIZATION: Streaming settings
    STREAMING_JSON_THRESHOLD_MB: float = float(os.getenv("STREAMING_JSON_THRESHOLD_MB", "10.0"))
    ENABLE_STREAMING_JSON: bool = os.getenv("ENABLE_STREAMING_JSON", "true").lower() == "true"
    
    # PERFORMANCE OPTIMIZATION: Auto-retry settings
    DEFAULT_MAX_RETRIES: int = int(os.getenv("DEFAULT_MAX_RETRIES", "3"))
    ENABLE_AUTO_RETRY: bool = os.getenv("ENABLE_AUTO_RETRY", "true").lower() == "true"
    ENABLE_MODEL_ESCALATION: bool = os.getenv("ENABLE_MODEL_ESCALATION", "true").lower() == "true"
    
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
    JOB_WORKERS: int = int(os.getenv("JOB_WORKERS", "2"))

    # SSE/WebSocket Configuration
    SSE_TIMEOUT_SECONDS: int = 30 # How long SSE connection waits for message before sending keepalive
    WEBSOCKET_KEEP_ALIVE_TIMEOUT_SECONDS: int = 60 # Timeout for WebSocket receive to check connection

    # Redis Configuration for Job Management
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_JOB_QUEUE_KEY: str = "intelliextract_job_queue"
    REDIS_JOB_KEY_PREFIX: str = "intelliextract_job:"
    REDIS_JOB_TTL_SECONDS: int = 24 * 3600  # 24 hours

    # Disk Space Monitoring
    MIN_DISK_SPACE_MB: int = int(os.getenv("MIN_DISK_SPACE_MB", "50")) # Minimum free disk space in MB required

    # API Key Validation Setting
    SKIP_API_KEY_VALIDATION_ON_STARTUP: bool = os.getenv("SKIP_API_KEY_VALIDATION_ON_STARTUP", "false").lower() == "true"
    
    # PERFORMANCE OPTIMIZATION: Monitoring and metrics
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
    PERFORMANCE_METRICS_LOG_INTERVAL: int = int(os.getenv("PERFORMANCE_METRICS_LOG_INTERVAL", "10"))  # Log every N operations
    
    @field_validator('GOOGLE_API_KEY', mode='before')
    @classmethod
    def validate_google_api_key(cls, v, info):
        skip_validation = info.data.get('SKIP_API_KEY_VALIDATION_ON_STARTUP', False) if info.data else False
        if not v or v.isspace():
            if not skip_validation:
                raise ValueError("GOOGLE_API_KEY must be set and not be empty or whitespace.")
            logger.warning("GOOGLE_API_KEY is not set or is whitespace. AI processing will likely fail.")
        elif len(v) < 20:  # Basic length check
            if not skip_validation:
                # Allowing short keys if validation is skipped, but still warn
                logger.warning("GOOGLE_API_KEY appears to be invalid (too short), but validation is skipped.")
            else:
                logger.warning("GOOGLE_API_KEY appears to be invalid (too short).")
        return v
    
    @field_validator('AGNO_API_KEY', mode='after')
    @classmethod
    def validate_agno_api_key(cls, v, info):
        if info.data and info.data.get('AGNO_MONITOR') and not v:
            logger.warning("AGNO_MONITOR is enabled but AGNO_API_KEY is not set. Monitoring may not work properly.")
        return v
    
    @field_validator('MAX_POOL_SIZE', mode='after')
    @classmethod
    def validate_pool_size(cls, v):
        if v < 5:
            logger.warning("MAX_POOL_SIZE is very low - consider increasing for better performance")
        elif v > 50:
            logger.warning("MAX_POOL_SIZE is very high - may consume excessive memory")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# Log successful configuration loading with performance settings
logger.info("Configuration loaded successfully with performance optimizations.")
logger.info(f"Performance settings: Pool size={settings.MAX_POOL_SIZE}, Auto-retry={settings.ENABLE_AUTO_RETRY}, Streaming={settings.ENABLE_STREAMING_JSON}")

if settings.ENABLE_PERFORMANCE_MONITORING:
    logger.info("🚀 Performance monitoring ENABLED - detailed metrics will be tracked")
else:
    logger.info("Performance monitoring disabled - set ENABLE_PERFORMANCE_MONITORING=true to enable")
