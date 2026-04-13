"""Tests for embeddings settings and ChromaStore operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docqa.vectordb.embeddings import EmbeddingProvider, EmbeddingSettings, get_embeddings
from docqa.vectordb.store import ChromaStore, VectorStoreError


class TestEmbeddingSettings:
    def test_default_provider(self):
        s = EmbeddingSettings(openai_api_key="sk-test")
        assert s.embedding_provider == EmbeddingProvider.OPENAI

    def test_default_model(self):
        s = EmbeddingSettings(openai_api_key="sk-test")
        assert s.embedding_model == "text-embedding-ada-002"

    def test_custom_settings(self):
        s = EmbeddingSettings(embedding_provider=EmbeddingProvider.HUGGINGFACE, huggingface_model="my-model")
        assert s.embedding_provider == EmbeddingProvider.HUGGINGFACE
        assert s.huggingface_model == "my-model"

    def test_huggingface_config(self):
        s = EmbeddingSettings(embedding_provider=EmbeddingProvider.HUGGINGFACE, huggingface_model="intfloat/multilingual-e5-large")
        assert "multilingual" in s.huggingface_model


def _fake_embeddings():
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture()
def chroma(tmp_path: Path):
    with patch("docqa.vectordb.store.get_embeddings", return_value=_fake_embeddings()):
        store = ChromaStore(collection_name="test_col", persist_directory=tmp_path / "db")
        _ = store.store
        return store


class TestChromaStoreInit:
    def test_creates_store(self, chroma):
        assert chroma.store is not None


class TestChromaStoreAdd:
    def test_add_returns_ids(self, chroma):
        docs = [Document(page_content="cat", metadata={"source": "a.txt"})]
        ids = chroma.add_documents(docs)
        assert isinstance(ids, list)
        assert len(ids) == 1

    def test_count_matches(self, chroma):
        docs = [Document(page_content="one", metadata={}), Document(page_content="two", metadata={})]
        chroma.add_documents(docs)
        assert chroma.count() == 2

    def test_add_empty_list(self, chroma):
        ids = chroma.add_documents([])
        assert ids == []


class TestChromaStoreSearch:
    def test_similarity_search(self, chroma):
        docs = [
            Document(page_content="Dogs are loyal", metadata={"source": "pets.txt"}),
            Document(page_content="Cats are independent", metadata={"source": "pets.txt"}),
        ]
        chroma.add_documents(docs)
        results = chroma.similarity_search("loyal animals", k=1)
        assert len(results) >= 1

    def test_search_with_score(self, chroma):
        docs = [Document(page_content="hello", metadata={})]
        chroma.add_documents(docs)
        results = chroma.similarity_search_with_score("hello", k=1)
        assert len(results) >= 1
        assert isinstance(results[0], tuple)


class TestChromaStoreDelete:
    def test_delete_by_id(self, chroma):
        docs = [Document(page_content="remove me", metadata={})]
        ids = chroma.add_documents(docs)
        assert chroma.count() == 1
        chroma.delete(ids)
        assert chroma.count() == 0

    def test_clear_all(self, chroma):
        docs = [Document(page_content="a", metadata={}), Document(page_content="b", metadata={})]
        chroma.add_documents(docs)
        assert chroma.count() == 2
        chroma.clear()
        assert chroma.count() == 0

    def test_count_empty_store(self, chroma):
        assert chroma.count() == 0


class TestGetEmbeddings:
    @patch("docqa.vectordb.embeddings.OpenAIEmbeddings")
    def test_returns_model_with_key(self, mock_cls):
        settings = EmbeddingSettings(openai_api_key="sk-test")
        model = get_embeddings(settings)
        mock_cls.assert_called_once()

    def test_raises_without_key(self):
        settings = EmbeddingSettings(openai_api_key="", embedding_provider=EmbeddingProvider.OPENAI)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_embeddings(settings)
