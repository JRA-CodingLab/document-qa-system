"""ChromaDB vector store wrapper with lazy initialisation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from docqa.vectordb.embeddings import EmbeddingSettings, get_embeddings


class VectorStoreError(Exception):
    """Raised when a vector-store operation fails."""


class ChromaStore:
    """Thin wrapper around LangChain's Chroma integration.

    The underlying database connection is created **lazily** on the first
    access of :pyattr:`store`.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[Path] = None,
        embedding_settings: Optional[EmbeddingSettings] = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._embedding_settings = embedding_settings
        self._store = None

    # ── lazy init ─────────────────────────────────────────────────

    @property
    def store(self):
        """Return (and lazily create) the underlying Chroma instance."""
        if self._store is None:
            from langchain_chroma import Chroma

            embed_fn = get_embeddings(self._embedding_settings)
            kwargs = {
                "collection_name": self._collection_name,
                "embedding_function": embed_fn,
            }
            if self._persist_directory is not None:
                kwargs["persist_directory"] = str(self._persist_directory)
            self._store = Chroma(**kwargs)
        return self._store

    # ── public API ─────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add *documents* and return their IDs."""
        if not documents:
            return []
        try:
            return self.store.add_documents(documents)
        except Exception as exc:
            raise VectorStoreError(f"Failed to add documents: {exc}") from exc

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> List[Document]:
        """Return the *k* most relevant documents for *query*."""
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Like :pymeth:`similarity_search` but also returns scores."""
        return self.store.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        self.store.delete(ids)

    def count(self) -> int:
        """Return the number of documents in the store."""
        data = self.store.get()
        return len(data["ids"])

    def clear(self) -> None:
        """Remove **all** documents from the collection."""
        data = self.store.get()
        ids = data["ids"]
        if ids:
            self.store.delete(ids)
