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
    APP_TITLE: str = "Enhanced Agno AI JSON to XLSX Processing API"
    TEMP_DIR_PREFIX: str = "enhanced_agno_xlsx_"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
