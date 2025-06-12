# app/core/config.py
import os
from pydantic_settings import BaseSettings

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

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
