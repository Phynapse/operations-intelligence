"""Application configuration.

All runtime configuration is read from environment variables (optionally from a
local ``.env`` file). Secrets are never stored in the repository and are never
returned to the frontend: :class:`LLMSettings.public_summary` exists specifically
so API responses can describe the active model without leaking credentials.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root (backend/..)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
#: Default location of the synthetic CSV datasets.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

AppEnvironment = Literal["local", "staging", "production"]


class Settings(BaseSettings):
    """Root settings object. Load with :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---- Application -------------------------------------------------
    app_name: str = "Operations Intelligence"
    app_env: AppEnvironment = "local"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # ---- API ---------------------------------------------------------
    api_cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    # ---- Data --------------------------------------------------------
    data_dir: Path = DEFAULT_DATA_DIR
    max_analysis_rows: int = Field(default=200_000, ge=1_000, le=5_000_000)
    min_observations_for_trend: int = Field(default=4, ge=2, le=365)
    evidence_top_n: int = Field(default=5, ge=1, le=50)

    # ---- Input guard rails -------------------------------------------
    max_question_length: int = Field(default=600, ge=32, le=8_000)

    # ---- LLM provider ------------------------------------------------
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = Field(default=45.0, ge=1.0, le=300.0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1_200, ge=128, le=8_192)

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        """Accept either a JSON list or a comma-separated string."""
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return ()
            if raw.startswith("["):
                return raw
            return tuple(item.strip() for item in raw.split(",") if item.strip())
        return value

    @field_validator("llm_base_url")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("data_dir", mode="before")
    @classmethod
    def _resolve_data_dir(cls, value: object) -> object:
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
        return value

    # ---- Derived helpers --------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """True when enough configuration exists to call an LLM endpoint."""
        return bool(self.llm_api_key and self.llm_api_key.get_secret_value().strip())

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


class LLMSettings(BaseSettings):
    """Isolated view of the LLM configuration, used by the provider layer."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    api_key: SecretStr | None = Field(default=None, alias="LLM_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    timeout: float = Field(default=45.0, alias="LLM_TIMEOUT")
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=1_200, alias="LLM_MAX_TOKENS")

    @field_validator("base_url")
    @classmethod
    def _strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    def public_summary(self) -> dict[str, object]:
        """Non-sensitive description safe to expose through the API."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "configured": bool(self.api_key and self.api_key.get_secret_value().strip()),
            "timeout_seconds": self.timeout,
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached application settings."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that override the environment)."""
    get_settings.cache_clear()
