"""Embedding providers — OpenAI and HuggingFace."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic_settings import BaseSettings


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class EmbeddingSettings(BaseSettings):
    """Settings for the embedding layer (loaded from env / .env)."""

    openai_api_key: str = ""
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-ada-002"
    huggingface_model: str = "intfloat/multilingual-e5-large"

    class Config:
        env_file = ".env"
        extra = "ignore"


def get_embeddings(settings: Optional[EmbeddingSettings] = None):
    """Return a LangChain-compatible embedding model.

    Raises ``ValueError`` if the selected provider cannot be initialised
    (e.g. missing API key).
    """
    if settings is None:
        settings = EmbeddingSettings()

    if settings.embedding_provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using the OpenAI embedding provider"
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def embed_texts(
    texts: List[str],
    settings: Optional[EmbeddingSettings] = None,
) -> List[List[float]]:
    """Embed a list of texts, returning a list of vectors."""
    model = get_embeddings(settings)
    return model.embed_documents(texts)


def embed_query(
    query: str,
    settings: Optional[EmbeddingSettings] = None,
) -> List[float]:
    """Embed a single query string."""
    model = get_embeddings(settings)
    return model.embed_query(query)
