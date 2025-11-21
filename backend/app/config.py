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
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into list."""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS
    
    # Allowed hosts (for production)
    ALLOWED_HOSTS: str = "localhost,127.0.0.1,*.onrender.com"
    
    @property
    def allowed_hosts_list(self) -> list[str]:
        """Parse ALLOWED_HOSTS string into list."""
        if isinstance(self.ALLOWED_HOSTS, str):
            return [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
        return self.ALLOWED_HOSTS
    
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

