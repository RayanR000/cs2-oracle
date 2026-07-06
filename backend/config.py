"""
Configuration management for the backend
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///backend/cs2_market.db"
    
    # Application
    app_name: str = "CS2 Market Intelligence API"
    environment: str = "development"
    debug: bool = True
    
    # API
    api_title: str = "CS2 Market Intelligence"
    api_version: str = "0.1.0"
    
    # Steam Integration
    # Steam Web API key from https://steamcommunity.com/dev/apikey
    # Daily limit: 100,000 calls per day (https://steamcommunity.com/dev/apiterms)
    # Used for: GetAssetClassInfo, GetSchemaItems, inventory lookups
    steam_api_key: Optional[str] = None
    cs2sh_api_key: Optional[str] = None

    # Steam session cookies (from browser DevTools → Application → Cookies → steamcommunity.com)
    # Required for /market/pricehistory/ endpoint (historical price data)
    steam_session_id: Optional[str] = None
    steam_login_secure: Optional[str] = None

    # CSMarketAPI keys (https://csmarketapi.com)
    # Each key gets 1,000 free requests/month. Add account name for tracking.
    csmarketapi_key_1: Optional[str] = None
    csmarketapi_account_1: Optional[str] = None
    csmarketapi_key_2: Optional[str] = None
    csmarketapi_account_2: Optional[str] = None
    csmarketapi_key_3: Optional[str] = None
    csmarketapi_account_3: Optional[str] = None
    csmarketapi_key_4: Optional[str] = None
    csmarketapi_account_4: Optional[str] = None
    csmarketapi_key_5: Optional[str] = None
    csmarketapi_account_5: Optional[str] = None
    csmarketapi_key_6: Optional[str] = None
    csmarketapi_account_6: Optional[str] = None

    @property
    def csmarketapi_keys(self) -> list[dict[str, str]]:
        """Return all configured CSMarketAPI keys as (account, key) pairs."""
        keys = []
        for i in range(1, 7):
            key = getattr(self, f"csmarketapi_key_{i}", None)
            account = getattr(self, f"csmarketapi_account_{i}", None) or f"account_{i}"
            if key:
                keys.append({"account": account, "key": key})
        return keys

    frontend_url: str = "http://localhost:3000"
    api_url: str = "http://localhost:8000"
    
    # Security
    secret_key: str = "your-secret-key-for-sessions"  # Should be changed in production
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

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
