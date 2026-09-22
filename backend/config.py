"""
Configuration management for the backend
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///backend/cs2_market.db"

    # Application
    environment: str = "development"
    debug: bool = True

    # API
    api_title: str = "CS2 Market Intelligence"
    api_version: str = "0.1.0"

    # Steam Integration
    # Steam Web API key from https://steamcommunity.com/dev/apikey
    # Daily limit: 100,000 calls per day (https://steamcommunity.com/dev/apiterms)
    # Used for: GetAssetClassInfo, GetSchemaItems, inventory lookups
    steam_api_key: str | None = None

    # Steam Community login cookies (for the authenticated /market/pricehistory/ endpoint).
    # Grab from a logged-in browser: DevTools > Application > Cookies > steamcommunity.com.
    #   STEAM_SESSION_ID   = cookie "sessionid"
    #   STEAM_LOGIN_SECURE = cookie "steamLoginSecure"
    # These expire (esp. steamLoginSecure) — refresh when the backfill reports session invalid.
    steam_session_id: str | None = None
    steam_login_secure: str | None = None

    # CSMarketAPI keys (https://csmarketapi.com)
    # Each key gets 1,000 free requests/month. Add account name for tracking.
    csmarketapi_key_1: str | None = None
    csmarketapi_account_1: str | None = None
    csmarketapi_key_2: str | None = None
    csmarketapi_account_2: str | None = None
    csmarketapi_key_3: str | None = None
    csmarketapi_account_3: str | None = None
    csmarketapi_key_4: str | None = None
    csmarketapi_account_4: str | None = None
    csmarketapi_key_5: str | None = None
    csmarketapi_account_5: str | None = None
    csmarketapi_key_6: str | None = None
    csmarketapi_account_6: str | None = None

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

    def check_secret_key(self) -> None:
        """Fail fast when the app boots in production with the default key.

        Called from main.lifespan (server startup), not at import: settings
        objects are also constructed by tests and offline scripts, which must
        not fail on a key they never use for signing.
        """
        if self.is_production() and self.secret_key == "your-secret-key-for-sessions":
            raise ValueError("secret_key must be set to a non-default value in production")

    def is_production(self) -> bool:
        """Return True when the app should avoid demo bootstrap behavior."""
        return self.environment.lower() in {"production", "prod"}


settings = Settings()
