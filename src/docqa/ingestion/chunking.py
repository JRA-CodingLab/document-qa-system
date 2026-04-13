"""Text chunking with configurable overlap and separators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkingConfig:
    """Parameters controlling how text is split into chunks."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {self.chunk_size}"
            )
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less "
                f"than chunk_size ({self.chunk_size})"
            )


def _build_splitter(config: Optional[ChunkingConfig] = None) -> RecursiveCharacterTextSplitter:
    """Create a text splitter from *config* (defaults applied if ``None``)."""
    if config is None:
        config = ChunkingConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=config.separators,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: List[Document],
    config: Optional[ChunkingConfig] = None,
) -> List[Document]:
    """Split *documents* into smaller chunks, preserving metadata.

    Each chunk gets an additional ``chunk_index`` in its metadata.
    """
    splitter = _build_splitter(config)
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
    return chunks


def chunk_text(
    text: str,
    config: Optional[ChunkingConfig] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> List[Document]:
    """Split raw *text* into ``Document`` objects.

    Each document carries the supplied *metadata* (if any) plus a
    ``chunk_index`` key.
    """
    splitter = _build_splitter(config)
    pieces = splitter.split_text(text)
    base_meta = metadata or {}
    docs: List[Document] = []
    for idx, piece in enumerate(pieces):
        doc_meta = {**base_meta, "chunk_index": idx}
        docs.append(Document(page_content=piece, metadata=doc_meta))
    return docs
