"""Tests for the RAG chain, LLM settings, document formatting, and prompts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from docqa.llm.prompts import format_documents, get_prompts
from docqa.llm.providers import LLMProvider, LLMSettings, get_llm, list_providers
from docqa.retrieval.chain import RAGChain


class TestLLMSettings:
    def test_defaults(self):
        s = LLMSettings(openai_api_key="sk-test")
        assert s.llm_provider == LLMProvider.OPENAI
        assert s.openai_model == "gpt-4o-mini"

    def test_custom_settings(self):
        s = LLMSettings(
            llm_provider=LLMProvider.ANTHROPIC,
            anthropic_api_key="sk-ant-test",
            anthropic_model="claude-3-haiku-20240307",
        )
        assert s.llm_provider == LLMProvider.ANTHROPIC
        assert s.anthropic_model == "claude-3-haiku-20240307"


class TestFormatDocuments:
    def test_single_doc(self):
        docs = [Document(page_content="Cats are pets.", metadata={"source": "pets.txt"})]
        result = format_documents(docs)
        assert "[Document 1]" in result
        assert "pets.txt" in result

    def test_multiple_docs(self):
        docs = [
            Document(page_content="First.", metadata={"source": "a.txt"}),
            Document(page_content="Second.", metadata={"source": "b.txt"}),
        ]
        result = format_documents(docs)
        assert "[Document 1]" in result
        assert "[Document 2]" in result
        assert "---" in result

    def test_unknown_source(self):
        docs = [Document(page_content="hi", metadata={})]
        result = format_documents(docs)
        assert "Unknown" in result


def _mock_store(docs=None):
    store = MagicMock()
    store.similarity_search.return_value = docs or []
    store.count.return_value = len(docs or [])
    return store


def _mock_llm(response_text="mocked answer"):
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=response_text)
    return mock


class TestRAGChainInit:
    def test_creates_with_params(self):
        store = _mock_store()
        chain = RAGChain(vector_store=store, k=3)
        assert chain._k == 3
        assert chain._vector_store is store


class TestRAGChainRetrieve:
    def test_delegates_to_store(self):
        docs = [Document(page_content="hi", metadata={})]
        store = _mock_store(docs)
        chain = RAGChain(vector_store=store, k=2)
        result = chain.retrieve("test query")
        store.similarity_search.assert_called_once_with("test query", k=2)
        assert result == docs


class TestRAGChainInvoke:
    @patch("docqa.retrieval.chain.get_llm")
    def test_returns_answer_and_sources(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm("The answer is 42")
        docs = [Document(page_content="Context text here", metadata={"source": "f.txt"})]
        store = _mock_store(docs)
        chain = RAGChain(vector_store=store, k=1)
        result = chain.invoke("What is the answer?")
        assert "answer" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)

    @patch("docqa.retrieval.chain.get_llm")
    def test_clear_history(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm()
        store = _mock_store([Document(page_content="x", metadata={})])
        chain = RAGChain(vector_store=store)
        chain.invoke("q", use_history=True)
        assert len(chain.chat_history) > 0
        chain.clear_history()
        assert len(chain.chat_history) == 0

    @patch("docqa.retrieval.chain.get_llm")
    def test_chat_history_is_copy(self, mock_get_llm):
        mock_get_llm.return_value = _mock_llm()
        store = _mock_store([Document(page_content="x", metadata={})])
        chain = RAGChain(vector_store=store)
        chain.invoke("q", use_history=True)
        history = chain.chat_history
        history.clear()
        assert len(chain.chat_history) > 0


class TestRAGPrompt:
    def test_rag_prompt_has_variables(self):
        prompts = get_prompts("en")
        rag = prompts["rag"]
        input_vars = rag.input_variables
        assert "context" in input_vars
        assert "question" in input_vars


class TestGetLLM:
    def test_raises_without_api_keys(self):
        settings = LLMSettings(openai_api_key="", llm_provider=LLMProvider.OPENAI)
        with pytest.raises(ValueError):
            get_llm(settings)


class TestListProviders:
    def test_returns_three(self):
        providers = list_providers()
        assert len(providers) == 3
        names = {p["provider"] for p in providers}
        assert names == {"openai", "anthropic", "ollama"}
