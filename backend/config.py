"""
Configuration management for the backend
"""

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/cs2_market"
    
    # Application
    app_name: str = "CS2 Market Intelligence API"
    environment: str = "development"
    debug: bool = True
    
    # API
    api_title: str = "CS2 Market Intelligence"
    api_version: str = "0.1.0"
    
    # Steam Integration
    steam_api_key: Optional[str] = None
    frontend_url: str = "http://localhost:3000"
    
    # Security
    secret_key: str = "your-secret-key-for-sessions"  # Should be changed in production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"

    def is_production(self) -> bool:
        """Return True when the app should avoid demo bootstrap behavior."""
        return self.environment.lower() in {"production", "prod"}

    def demo_bootstrap_enabled(self) -> bool:
        """
        Return True when synthetic catalog/history bootstrap should run.

        Demo and development environments keep the synthetic backfill available
        for local iteration, while production stays on the live collection path.
        """
        return not self.is_production()

settings = Settings()
