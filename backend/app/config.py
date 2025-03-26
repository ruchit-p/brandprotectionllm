import os
from typing import Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with environment variable validation
    """
    # Database settings
    DATABASE_URL: str = "postgresql://postgres:postgres@postgres:5432/brand_protection"
    
    # Redis settings
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Qdrant settings
    QDRANT_URL: str = "http://qdrant:6333"
    
    # LLM Provider settings
    LLM_PROVIDER: str = "anthropic"  # One of: anthropic, openai, ollama
    LLM_MODEL: Optional[str] = None  # If None, use provider default
    
    # Anthropic settings
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "llama2"
    
    # AWS settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Application settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Storage paths
    STORAGE_PATH: str = Field("./storage", env="STORAGE_PATH")
    
    # Celery settings
    CELERY_BROKER_URL: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field("redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # JPlag settings
    JPLAG_JAR_PATH: str = Field("./lib/jplag.jar", env="JPLAG_JAR_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("STORAGE_PATH")
    def create_storage_dirs(cls, v):
        """Create storage directories if they don't exist"""
        os.makedirs(v, exist_ok=True)
        os.makedirs(os.path.join(v, "logos"), exist_ok=True)
        os.makedirs(os.path.join(v, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(v, "assets"), exist_ok=True)
        os.makedirs(os.path.join(v, "evidence"), exist_ok=True)
        os.makedirs(os.path.join(v, "custom_datasets"), exist_ok=True)
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings from environment variables with caching
    """
    try:
        return Settings()
    except Exception as e:
        missing_fields = []
        if hasattr(e, "errors"):
            for error in e.errors():
                if error["type"] == "missing":
                    missing_fields.append(error["loc"][0])
        
        if missing_fields:
            error_message = f"Missing required environment variables: {', '.join(missing_fields)}"
            raise ValueError(error_message) from e
        raise e 