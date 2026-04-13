"""Centralised application settings loaded from environment / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Top-level configuration container.

    Individual modules (LLMSettings, EmbeddingSettings) handle their own
    env loading.  This class is provided for convenience if you want a
    single source of truth at app startup.
    """

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"
