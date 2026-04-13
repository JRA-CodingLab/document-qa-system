"""Vector store — embeddings, storage, and language detection."""

from docqa.vectordb.embeddings import (
    EmbeddingProvider,
    EmbeddingSettings,
    get_embeddings,
    embed_texts,
    embed_query,
)
from docqa.vectordb.store import ChromaStore, VectorStoreError
from docqa.vectordb.language import detect_language

__all__ = [
    "EmbeddingProvider",
    "EmbeddingSettings",
    "get_embeddings",
    "embed_texts",
    "embed_query",
    "ChromaStore",
    "VectorStoreError",
    "detect_language",
]
