"""Document ingestion — loaders and chunking."""

from docqa.ingestion.loaders import (
    DocumentLoaderError,
    load_pdf,
    load_markdown,
    load_text,
    load_document,
    load_directory,
)
from docqa.ingestion.chunking import ChunkingConfig, chunk_documents, chunk_text

__all__ = [
    "DocumentLoaderError",
    "load_pdf",
    "load_markdown",
    "load_text",
    "load_document",
    "load_directory",
    "ChunkingConfig",
    "chunk_documents",
    "chunk_text",
]
