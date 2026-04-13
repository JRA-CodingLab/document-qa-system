"""LLM provider abstraction — OpenAI, Anthropic, Ollama."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMSettings(BaseSettings):
    """Settings for the LLM backend (loaded from env / .env)."""

    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 1024

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_llm(settings: Optional[LLMSettings] = None):
    """Return a LangChain chat model for the configured provider.

    Raises ``ValueError`` if the provider requires an API key that is
    not set.
    """
    if settings is None:
        settings = LLMSettings()

    if settings.llm_provider == LLMProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using the OpenAI LLM provider"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if settings.llm_provider == LLMProvider.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using the Anthropic LLM provider"
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            anthropic_api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    if settings.llm_provider == LLMProvider.OLLAMA:
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def list_providers() -> List[Dict[str, object]]:
    """Return metadata about every supported LLM provider."""
    return [
        {
            "provider": "openai",
            "default_model": "gpt-4o-mini",
            "requires_api_key": True,
        },
        {
            "provider": "anthropic",
            "default_model": "claude-3-5-sonnet-20241022",
            "requires_api_key": True,
        },
        {
            "provider": "ollama",
            "default_model": "llama3.2",
            "requires_api_key": False,
        },
    ]
