"""Tests for FastAPI endpoints."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from docqa.api.endpoints import app, _app_state
from docqa.api.schemas import QueryRequest


@pytest.fixture(autouse=True)
def mock_vector_store():
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store.similarity_search.return_value = []
    mock_store.add_documents.return_value = ["id1"]
    _app_state["vector_store"] = mock_store
    yield mock_store
    _app_state["vector_store"] = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestModels:
    def test_returns_200(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200

    def test_three_providers(self, client):
        data = client.get("/api/v1/models").json()
        assert len(data["providers"]) == 3
        names = {p["provider"] for p in data["providers"]}
        assert names == {"openai", "anthropic", "ollama"}


class TestDocuments:
    def test_count_returns_200(self, client):
        resp = client.get("/api/v1/documents/count")
        assert resp.status_code == 200
        assert "count" in resp.json()

    def test_clear_returns_200(self, client):
        resp = client.delete("/api/v1/documents")
        assert resp.status_code == 200


class TestQuery:
    def test_empty_store_returns_400(self, client, mock_vector_store):
        mock_vector_store.count.return_value = 0
        resp = client.post("/api/v1/query", json={"question": "hello?"})
        assert resp.status_code == 400
        assert "No documents" in resp.json()["detail"]

    def test_missing_question_returns_422(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    @patch("docqa.api.endpoints.RAGChain")
    def test_with_mocked_chain(self, mock_chain_cls, client, mock_vector_store):
        mock_vector_store.count.return_value = 5
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "answer": "Test answer",
            "sources": [{"content": "src", "metadata": {}}],
        }
        mock_chain_cls.return_value = mock_chain
        resp = client.post("/api/v1/query", json={"question": "What?"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Test answer"


class TestIngest:
    def test_missing_files_returns_422(self, client):
        resp = client.post("/api/v1/ingest")
        assert resp.status_code == 422

    def test_unsupported_type_returns_400(self, client):
        bad_file = BytesIO(b"data")
        resp = client.post(
            "/api/v1/ingest",
            files=[("files", ("test.docx", bad_file, "application/octet-stream"))],
        )
        assert resp.status_code == 400

    @patch("docqa.api.endpoints.chunk_documents")
    @patch("docqa.api.endpoints.load_document")
    def test_success_returns_counts(self, mock_load, mock_chunk, client, mock_vector_store):
        mock_load.return_value = [Document(page_content="hi", metadata={})]
        mock_chunk.return_value = [Document(page_content="hi", metadata={"chunk_index": 0})]
        mock_vector_store.add_documents.return_value = ["id1"]
        txt_file = BytesIO(b"Hello world")
        resp = client.post("/api/v1/ingest", files=[("files", ("test.txt", txt_file, "text/plain"))])
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 1

    @patch("docqa.api.endpoints.chunk_documents")
    @patch("docqa.api.endpoints.load_document")
    def test_custom_chunk_params(self, mock_load, mock_chunk, client, mock_vector_store):
        mock_load.return_value = [Document(page_content="ok", metadata={})]
        mock_chunk.return_value = [Document(page_content="ok", metadata={})]
        mock_vector_store.add_documents.return_value = ["id1"]
        txt_file = BytesIO(b"content")
        resp = client.post("/api/v1/ingest?chunk_size=500&chunk_overlap=50", files=[("files", ("t.txt", txt_file, "text/plain"))])
        assert resp.status_code == 200


class TestPydanticModels:
    def test_query_defaults(self):
        q = QueryRequest(question="hi")
        assert q.k == 4
        assert q.use_history is False

    def test_query_k_range(self):
        with pytest.raises(Exception):
            QueryRequest(question="hi", k=0)
        with pytest.raises(Exception):
            QueryRequest(question="hi", k=11)

    def test_question_min_length(self):
        with pytest.raises(Exception):
            QueryRequest(question="")
