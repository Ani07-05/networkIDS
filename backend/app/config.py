"""
Application configuration management.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Network IDS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/nids_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Clerk Authentication
    CLERK_SECRET_KEY: str = "placeholder_key"
    CLERK_PUBLISHABLE_KEY: str = "placeholder_key"
    CLERK_JWT_KEY: str = "placeholder_key"
    CLERK_ISSUER: str = "https://expert-killdeer-77.clerk.accounts.dev"
    
    # Security
    SECRET_KEY: str = "default_secret_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Allowed hosts (for production)
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1", "*.your-domain.com"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # ML Models
    BINARY_MODEL_PATH: str = "ml/models/nids_model_binary.onnx"
    MULTICLASS_MODEL_PATH: str = "ml/models/nids_model_multiclass.onnx"
    PREPROCESSOR_PATH: str = "ml/data/processed/preprocessor.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

