"""Tests for document loading and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docqa.ingestion.chunking import ChunkingConfig, chunk_documents, chunk_text
from docqa.ingestion.loaders import (
    DocumentLoaderError,
    load_directory,
    load_document,
    load_markdown,
    load_text,
)


class TestLoadText:
    def test_returns_documents(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello world")
        docs = load_text(f)
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_content_present(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("Some important content")
        docs = load_text(f)
        assert "important content" in docs[0].page_content

    def test_metadata_correct(self, tmp_path: Path):
        f = tmp_path / "meta.txt"
        f.write_text("meta test")
        docs = load_text(f)
        assert docs[0].metadata["source"] == str(f)
        assert docs[0].metadata["file_type"] == "text"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(DocumentLoaderError):
            load_text(tmp_path / "no_such_file.txt")


class TestLoadMarkdown:
    def test_returns_documents(self, tmp_path: Path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\nBody text")
        docs = load_markdown(f)
        assert isinstance(docs, list)
        assert len(docs) >= 1

    def test_content_present(self, tmp_path: Path):
        f = tmp_path / "notes.md"
        f.write_text("# Heading\nSome markdown body")
        docs = load_markdown(f)
        assert "markdown body" in docs[0].page_content

    def test_metadata_correct(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("content")
        docs = load_markdown(f)
        assert docs[0].metadata["file_type"] == "markdown"


class TestLoadDocument:
    def test_routes_txt(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        docs = load_document(f)
        assert docs[0].metadata["file_type"] == "text"

    def test_routes_md(self, tmp_path: Path):
        f = tmp_path / "b.md"
        f.write_text("# hi")
        docs = load_document(f)
        assert docs[0].metadata["file_type"] == "markdown"

    def test_rejects_unsupported(self, tmp_path: Path):
        f = tmp_path / "x.docx"
        f.write_text("nope")
        with pytest.raises(DocumentLoaderError, match="Unsupported"):
            load_document(f)


class TestLoadDirectory:
    def test_loads_all_files(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("one")
        (tmp_path / "b.md").write_text("two")
        docs = load_directory(tmp_path)
        assert len(docs) >= 2

    def test_filters_by_extension(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("one")
        (tmp_path / "b.md").write_text("two")
        docs = load_directory(tmp_path, extensions=[".txt"])
        for d in docs:
            assert d.metadata["file_type"] == "text"

    def test_missing_dir_raises(self, tmp_path: Path):
        with pytest.raises(DocumentLoaderError):
            load_directory(tmp_path / "nope")

    def test_file_not_dir_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("oops")
        with pytest.raises(DocumentLoaderError):
            load_directory(f)


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200

    def test_custom_values(self):
        cfg = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="positive"):
            ChunkingConfig(chunk_size=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            ChunkingConfig(chunk_overlap=-1)

    def test_overlap_gte_size(self):
        with pytest.raises(ValueError, match="less than"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)


class TestChunkDocuments:
    def test_returns_list(self):
        doc = Document(page_content="Hello world " * 100, metadata={"source": "test"})
        chunks = chunk_documents([doc], ChunkingConfig(chunk_size=50, chunk_overlap=10))
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_preserves_metadata(self):
        doc = Document(page_content="A " * 500, metadata={"source": "s", "custom": 42})
        chunks = chunk_documents([doc], ChunkingConfig(chunk_size=50, chunk_overlap=10))
        assert chunks[0].metadata["source"] == "s"
        assert chunks[0].metadata["custom"] == 42

    def test_adds_chunk_index(self):
        doc = Document(page_content="word " * 500, metadata={})
        chunks = chunk_documents([doc], ChunkingConfig(chunk_size=50, chunk_overlap=10))
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_respects_config(self):
        doc = Document(page_content="word " * 500, metadata={})
        small = chunk_documents([doc], ChunkingConfig(chunk_size=50, chunk_overlap=10))
        large = chunk_documents([doc], ChunkingConfig(chunk_size=500, chunk_overlap=10))
        assert len(small) > len(large)


class TestChunkText:
    def test_returns_documents(self):
        text = "Hello world " * 100
        docs = chunk_text(text, ChunkingConfig(chunk_size=50, chunk_overlap=10))
        assert all(isinstance(d, Document) for d in docs)

    def test_attaches_metadata(self):
        text = "data " * 100
        docs = chunk_text(
            text, ChunkingConfig(chunk_size=50, chunk_overlap=10),
            metadata={"origin": "test"},
        )
        assert docs[0].metadata["origin"] == "test"
        assert "chunk_index" in docs[0].metadata

    def test_smaller_chunks_produce_more(self):
        text = "word " * 200
        small = chunk_text(text, ChunkingConfig(chunk_size=30, chunk_overlap=5))
        large = chunk_text(text, ChunkingConfig(chunk_size=200, chunk_overlap=5))
        assert len(small) > len(large)
